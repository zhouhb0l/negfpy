"""Plot silicon phonon transmission from QE IFC using corrected IFC mapping."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import transmission, transmission_kavg_adaptive
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.models import material_kspace_device, material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def _kmesh(nk: int, mode: str) -> np.ndarray:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if nk == 1:
        return np.array([0.0], dtype=float)
    if mode == "centered":
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step
    if mode == "shifted":
        if nk % 2 != 0:
            raise ValueError("shifted k-mesh requires even nk to avoid sampling Gamma exactly.")
        return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk)
    raise ValueError(f"Unknown k-mesh mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("si444.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--n-layers", type=int, default=30)
    parser.add_argument("--principal-layer-size", type=int, default=2)
    parser.add_argument("--nk", type=int, default=5, help="ky/kz mesh size; nk=1 uses Gamma only")
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
        default="shifted",
        choices=["shifted", "centered"],
        help="k-mesh style for k-averaging; shifted avoids Gamma-point overweighting at low omega",
    )
    parser.add_argument("--nw", type=int, default=160)
    parser.add_argument("--omega-min", type=float, default=0.0, help="Internal QE omega unit")
    parser.add_argument("--omega-max", type=float, default=0.006, help="Internal QE omega unit")
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument(
        "--surface-gf-method",
        type=str,
        default="sancho_rubio",
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

    ifc = _read_ifc(args.ifc, reader=args.reader)
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
        kvals_hi = _kmesh(args.nk, mode=args.kmesh)
        kpts_hi = [(ky, kz) for ky in kvals_hi for kz in kvals_hi]
        nk_low = max(args.nk_low, args.nk)
        kvals_lo = _kmesh(nk_low, mode=args.kmesh)
        kpts_lo = [(ky, kz) for ky in kvals_lo for kz in kvals_lo]
        tvals = []
        rejected_records: list[dict[str, object]] = []
        for w in omegas:
            if w <= 0.0:
                tvals.append(0.0)
                continue
            w_cm1 = float(qe_omega_to_cm1(w))
            kpts = kpts_lo if w_cm1 <= args.low_cm1 else kpts_hi
            tavg, _ = transmission_kavg_adaptive(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                kpoints=kpts,
                eta_values=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
                min_success_fraction=0.7,
                surface_gf_method=args.surface_gf_method,
                nonnegative_tolerance=1e-8,
                max_channel_factor=1.5,
                collect_rejected=args.print_rejected,
            )
            tvals.append(tavg)
            if args.print_rejected and "rejected" in _:
                rejected_records.extend(_["rejected"])  # type: ignore[index]
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


if __name__ == "__main__":
    main()
