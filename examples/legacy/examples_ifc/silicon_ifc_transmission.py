"""Plot silicon phonon transmission from QE IFC using corrected IFC mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission, transmission_kavg_adaptive, transmission_kavg_adaptive_global_eta
from negfpy.io import read_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.models import material_kspace_device, material_kspace_lead


DEFAULT_ETA_VALUES: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)


def _kmesh(nk: int, mode: str) -> tuple[np.ndarray, str]:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if nk == 1:
        return np.array([0.0], dtype=float), "gamma"
    mode_eff = mode
    if mode == "auto":
        mode_eff = "shifted" if (nk % 2 == 0) else "centered"
    if mode_eff == "shifted":
        if nk % 2 != 0:
            raise ValueError("shifted k-mesh requires even nk to avoid sampling Gamma exactly.")
        return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk), mode_eff
    if mode_eff == "centered":
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step, mode_eff
    raise ValueError(f"Unknown k-mesh mode: {mode}")


def _resolve_eta_values(
    omega: float,
    omega_cm1: float,
    *,
    eta_values: tuple[float, ...],
    max_eta_over_omega: float | None,
    eta_ratio_cm1_min: float | None,
) -> tuple[float, ...]:
    vals = eta_values
    if max_eta_over_omega is None:
        return vals
    apply_cap = eta_ratio_cm1_min is None or omega_cm1 >= eta_ratio_cm1_min
    if not apply_cap:
        return vals
    filtered = tuple(e for e in vals if (e / omega) <= max_eta_over_omega)
    if len(filtered) == 0:
        return (min(vals),)
    return filtered


def _compute_kavg(
    *,
    omega: float,
    kpts: list[tuple[float, float]],
    device,
    lead,
    eta_scheme: str,
    eta_fixed: float,
    eta_values: tuple[float, ...],
    surface_gf_method: str,
    collect_rejected: bool,
) -> tuple[float, dict[str, object]]:
    common_kwargs = dict(
        omega=float(omega),
        device=device,
        lead_left=lead,
        lead_right=lead,
        kpoints=kpts,
        min_success_fraction=0.7,
        surface_gf_method=surface_gf_method,
        nonnegative_tolerance=1e-8,
        max_channel_factor=1.5,
        collect_rejected=collect_rejected,
    )
    if eta_scheme == "fixed":
        return transmission_kavg_adaptive_global_eta(
            **common_kwargs,
            eta_values=(float(eta_fixed),),
        )
    if eta_scheme == "adaptive-global":
        return transmission_kavg_adaptive_global_eta(
            **common_kwargs,
            eta_values=eta_values,
        )
    return transmission_kavg_adaptive(
        **common_kwargs,
        eta_values=eta_values,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("si444.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--principal-layer-size", type=int, default=2)
    parser.add_argument("--nk", type=int, default=6, help="ky/kz mesh size; nk=1 uses Gamma only")
    parser.add_argument(
        "--nk-low",
        type=int,
        default=24,
        help="Dense ky/kz mesh used below --low-cm1 for accurate low-frequency averaging",
    )
    parser.add_argument(
        "--low-cm1",
        type=float,
        default=120.0,
        help="Use dense k-mesh below this frequency (cm^-1)",
    )
    parser.add_argument(
        "--kmesh",
        type=str,
        default="auto",
        choices=["auto", "shifted", "centered"],
        help=(
            "k-mesh style for k-averaging. auto: shifted for even nk, centered for odd nk. "
            "shifted avoids Gamma-point overweighting at low omega."
        ),
    )
    parser.add_argument("--nw", type=int, default=160)
    parser.add_argument("--omega-min", type=float, default=0.0, help="Internal QE omega unit")
    parser.add_argument("--omega-max", type=float, default=0.006, help="Internal QE omega unit")
    parser.add_argument("--eta", type=float, default=1e-4)
    parser.add_argument(
        "--eta-values",
        type=float,
        nargs="+",
        default=list(DEFAULT_ETA_VALUES),
        help="Adaptive eta ladder (used by adaptive/adaptive-global schemes).",
    )
    parser.add_argument(
        "--max-eta-over-omega",
        type=float,
        default=None,
        help="Optional cap for eta/omega in adaptive ladders (e.g. 0.01 for 1%).",
    )
    parser.add_argument(
        "--eta-ratio-cm1-min",
        type=float,
        default=None,
        help="Apply --max-eta-over-omega only above this frequency (cm^-1).",
    )
    parser.add_argument(
        "--eta-scheme",
        type=str,
        default="adaptive",
        choices=["fixed", "adaptive", "adaptive-global"],
        help=(
            "Broadening strategy for k-averaging. fixed uses one eta for all k-points "
            "(recommended for strict comparability). adaptive tries an eta ladder per k-point. "
            "adaptive-global chooses one eta per omega for all k-points."
        ),
    )
    parser.add_argument(
        "--surface-gf-method",
        type=str,
        default="generalized_eigen_svd",
        choices=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd", "legacy_eigen_svd"],
        help="Surface Green's function solver",
    )
    parser.add_argument(
        "--print-rejected",
        action="store_true",
        help="Print rejected (omega, k, eta, T) samples from adaptive filtering",
    )
    parser.add_argument(
        "--max-rejected-print",
        type=int,
        default=300,
        help="Maximum number of rejected records to print",
    )
    parser.add_argument(
        "--rejected-log",
        type=Path,
        default=None,
        help="Optional text file path to save all rejected samples",
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_transmission_corrected.png"))
    args = parser.parse_args()

    eta_values_base = tuple(float(e) for e in args.eta_values)
    if len(eta_values_base) == 0:
        raise ValueError("--eta-values must contain at least one value.")
    if any(e <= 0.0 for e in eta_values_base):
        raise ValueError("All --eta-values must be positive.")
    if args.max_eta_over_omega is not None and args.max_eta_over_omega <= 0.0:
        raise ValueError("--max-eta-over-omega must be positive.")

    ifc = read_ifc(args.ifc, reader=args.reader)
    cfg = BuildConfig(
        principal_layer_size=args.principal_layer_size,
        infer_fc01_from_negative_dx=True,
    )
    params = build_material_kspace_params(ifc=ifc, config=cfg)
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=args.n_layers, params=params)

    omegas = np.linspace(args.omega_min, args.omega_max, args.nw)
    if args.nk <= 1:
        tvals = np.array(
            [
                transmission(
                    w,
                    device=device,
                    lead_left=lead,
                    lead_right=lead,
                    kpar=(0.0, 0.0),
                    eta=args.eta,
                    surface_gf_method=args.surface_gf_method,
                )
                for w in omegas
            ]
        )
    else:
        kvals_hi, kmesh_hi = _kmesh(args.nk, mode=args.kmesh)
        kpts_hi = [(ky, kz) for ky in kvals_hi for kz in kvals_hi]
        nk_low = max(args.nk_low, args.nk)
        kvals_lo, kmesh_lo = _kmesh(nk_low, mode=args.kmesh)
        kpts_lo = [(ky, kz) for ky in kvals_lo for kz in kvals_lo]
        print(
            f"k-mesh resolved: high={kmesh_hi} (nk={args.nk}), "
            f"low={kmesh_lo} (nk_low={nk_low})"
        )
        print(f"eta scheme: {args.eta_scheme}, eta_fixed={args.eta:.1e}")
        if args.max_eta_over_omega is not None:
            print(
                "eta/omega cap enabled: "
                f"max={args.max_eta_over_omega:.3e}, "
                f"cm^-1 threshold={args.eta_ratio_cm1_min}"
            )
        tvals = []
        rejected_records: list[dict[str, object]] = []
        for w in omegas:
            if w <= 0.0:
                tvals.append(0.0)
                continue
            w_cm1 = float(qe_omega_to_cm1(w))
            kpts = kpts_lo if w_cm1 <= args.low_cm1 else kpts_hi
            eta_values = _resolve_eta_values(
                omega=float(w),
                omega_cm1=w_cm1,
                eta_values=eta_values_base,
                max_eta_over_omega=args.max_eta_over_omega,
                eta_ratio_cm1_min=args.eta_ratio_cm1_min,
            )
            try:
                tavg, info = _compute_kavg(
                    omega=float(w),
                    kpts=kpts,
                    device=device,
                    lead=lead,
                    eta_scheme=args.eta_scheme,
                    eta_fixed=float(args.eta),
                    eta_values=eta_values,
                    surface_gf_method=args.surface_gf_method,
                    collect_rejected=args.print_rejected,
                )
            except RuntimeError as exc:
                if args.eta_scheme != "fixed":
                    raise
                raise RuntimeError(
                    "Fixed-eta k-averaging failed quality checks. "
                    "Try a larger --eta (e.g. 1e-4 to 1e-3), denser --nk, "
                    "or switch to --eta-scheme adaptive."
                ) from exc
            tvals.append(float(tavg))
            if args.print_rejected and "rejected" in info:
                rejected_records.extend(info["rejected"])  # type: ignore[index]
        tvals = np.asarray(tvals, dtype=float)
        # Suppress tiny negative values from numerical roundoff.
        tvals = np.where(tvals < 0.0, np.maximum(tvals, -1e-12), tvals)
        tvals = np.clip(tvals, 0.0, None)
        if args.print_rejected and len(rejected_records) > 0:
            maxn = max(0, int(args.max_rejected_print))
            print("Rejected samples (omega_cm^-1, kpar, eta, T, reason):")
            for rec in rejected_records[:maxn]:
                omega_cm1 = float(qe_omega_to_cm1(float(rec["omega"])))
                t_str = "None" if rec["t"] is None else f"{float(rec['t']):.6e}"
                print(
                    f"{omega_cm1:10.4f}  {rec['kpar']}  {float(rec['eta']):.1e}  "
                    f"{t_str}  {rec['reason']}"
                )
            if len(rejected_records) > maxn:
                print(f"... ({len(rejected_records) - maxn} more rejected samples not shown)")
        if args.rejected_log is not None:
            args.rejected_log.parent.mkdir(parents=True, exist_ok=True)
            with args.rejected_log.open("w", encoding="utf-8") as fh:
                fh.write("# omega_cm^-1\tkpar\teta\tT\treason\n")
                for rec in rejected_records:
                    omega_cm1 = float(qe_omega_to_cm1(float(rec["omega"])))
                    t_str = "None" if rec["t"] is None else f"{float(rec['t']):.12e}"
                    fh.write(
                        f"{omega_cm1:.8f}\t{rec['kpar']}\t{float(rec['eta']):.3e}\t"
                        f"{t_str}\t{rec['reason']}\n"
                    )
            print(f"Saved rejected log: {args.rejected_log}")

    x_cm1 = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(x_cm1, tvals, color="tab:blue", lw=1.6)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel(r"$T(\omega)$")
    if args.nk <= 1:
        ax.set_title(f"Silicon Transmission (Gamma-only, {args.surface_gf_method})")
    else:
        ax.set_title(f"Silicon Transmission (k-avg, nk={args.nk}x{args.nk}, {args.surface_gf_method})")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved transmission plot: {args.save}")
    print(f"T(omega) min={float(np.min(tvals)):.6e}, max={float(np.max(tvals)):.6e}")


if __name__ == "__main__":
    main()
