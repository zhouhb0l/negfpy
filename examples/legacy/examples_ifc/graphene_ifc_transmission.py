"""Evaluate graphene phonon transmission T(omega) from IFC with ky averaging."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from negfpy.core import transmission, transmission_kavg_adaptive
from negfpy.io import read_ifc
from negfpy.modeling import (
    BuildConfig,
    enforce_translational_asr_on_self_term,
    qe_ev_to_omega,
    qe_omega_to_cm1,
    qe_omega_to_thz,
)
from negfpy.modeling.builders import build_material_kspace_params
from negfpy.models import material_kspace_device, material_kspace_lead


def _kmesh_1d(nk: int, mode: str) -> np.ndarray:
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


def _default_kgrid(nk: int, centered: bool) -> np.ndarray:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if centered:
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step
    return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk)


def _estimate_omega_max_from_ifc(ifc, *, nkx: int, nky: int, centered_grid: bool, safety_factor: float = 1.02) -> float:
    kx = _default_kgrid(nkx, centered=centered_grid)
    ky = _default_kgrid(nky, centered=centered_grid)
    ndof = len(ifc.masses) * ifc.dof_per_atom
    mrep = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(mrep))
    terms = [(int(t.dx), int(t.dy), int(t.dz), np.asarray(t.block, dtype=np.complex128)) for t in ifc.terms]

    wmax = 0.0
    for kxv in kx:
        for kyv in ky:
            phi = np.zeros((ndof, ndof), dtype=np.complex128)
            for dx, dy, dz, block in terms:
                phi += block * np.exp(1j * (kxv * dx + kyv * dy + 0.0 * dz))
            dmat = minv @ phi @ minv
            dmat = 0.5 * (dmat + dmat.conj().T)
            vals = np.linalg.eigvalsh(dmat)
            local_max = float(np.sqrt(max(float(np.max(vals.real)), 0.0)))
            if local_max > wmax:
                wmax = local_max
    return float(max(wmax * safety_factor, 1e-8))


def _thz_to_qe_omega(freq_thz: float) -> float:
    if freq_thz <= 0.0:
        raise ValueError("THz frequency must be positive.")
    thz_per_qe = float(np.asarray(qe_omega_to_thz(1.0), dtype=float))
    return float(freq_thz / thz_per_qe)


def _cm1_to_qe_omega(freq_cm1: float) -> float:
    if freq_cm1 <= 0.0:
        raise ValueError("cm^-1 frequency must be positive.")
    cm1_per_qe = float(np.asarray(qe_omega_to_cm1(1.0), dtype=float))
    return float(freq_cm1 / cm1_per_qe)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("studies/graphene_bulk_2026q1/inputs/ifc/graphene.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--n-layers", type=int, default=40)
    parser.add_argument("--principal-layer-size", type=int, default=4)
    parser.add_argument("--enforce-asr", action="store_true")
    parser.add_argument("--nk", type=int, default=48, help="ky mesh size for 2D k-averaging")
    parser.add_argument("--kmesh", type=str, default="shifted", choices=["shifted", "centered"])
    parser.add_argument("--nw", type=int, default=220)
    parser.add_argument("--omega-min", type=float, default=0.0, help="Internal QE omega unit")
    parser.add_argument("--omega-max", type=float, default=None, help="Internal QE omega unit")
    parser.add_argument("--fmax-thz", type=float, default=None, help="Upper limit in THz; overrides --omega-max")
    parser.add_argument("--fmax-cm1", type=float, default=None, help="Upper limit in cm^-1; overrides --fmax-thz/--omega-max")
    parser.add_argument("--x-unit", type=str, default="thz", choices=["thz", "cm-1"])
    parser.add_argument("--eta", type=float, default=1e-8)
    parser.add_argument(
        "--eta-device",
        type=float,
        default=1e-8,
        help="Imaginary broadening used in device matrix; smaller values reduce artificial damping but may cause solver failures",
    )
    parser.add_argument(
        "--omega-scale-mode",
        type=str,
        default="none",
        choices=["none", "ev"],
        help="Internal numerical scaling. 'ev' solves in E=ħω (eV) units.",
    )
    parser.add_argument(
        "--surface-gf-method",
        type=str,
        default="sancho_rubio",
        choices=["sancho_rubio", "generalized_eigen", "generalized_eigen_svd", "legacy_eigen_svd"],
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/graphene_transmission_kavg.png"))
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Print progress every N omega points (set <=0 to disable).",
    )
    args = parser.parse_args()
    omega_scale = None
    if args.omega_scale_mode == "ev":
        omega_scale = float(np.asarray(qe_ev_to_omega(1.0), dtype=float))
        print(f"Omega scaling enabled: mode=ev, omega_scale={omega_scale:.6e} (internal omega per 1 eV)")

    ifc = read_ifc(args.ifc, reader=args.reader)
    if args.enforce_asr:
        ifc, asr_residual_max = enforce_translational_asr_on_self_term(ifc)
        print(f"ASR enforced: yes (max pre-correction residual={asr_residual_max:.6e})")
    else:
        print("ASR enforced: no")

    cfg = BuildConfig(
        principal_layer_size=args.principal_layer_size,
        infer_fc01_from_negative_dx=True,
    )
    params = build_material_kspace_params(ifc=ifc, config=cfg)
    lead = material_kspace_lead(params)
    device = material_kspace_device(n_layers=args.n_layers, params=params)

    if args.fmax_cm1 is not None:
        omega_max = _cm1_to_qe_omega(float(args.fmax_cm1))
        omega_max_source = f"from --fmax-cm1={args.fmax_cm1:g}"
    elif args.fmax_thz is not None:
        omega_max = _thz_to_qe_omega(float(args.fmax_thz))
        omega_max_source = f"from --fmax-thz={args.fmax_thz:g}"
    elif args.omega_max is not None:
        omega_max = float(args.omega_max)
        omega_max_source = "from --omega-max"
    else:
        omega_max = _estimate_omega_max_from_ifc(ifc, nkx=24, nky=24, centered_grid=False)
        omega_max_source = "auto-estimated from IFC"
    if omega_max <= args.omega_min:
        raise ValueError("Resolved omega-max must be greater than omega-min.")

    omegas = np.linspace(args.omega_min, omega_max, args.nw)
    start_time = time.perf_counter()
    total_omega = int(len(omegas))
    progress_every = int(args.progress_every)

    def _report_progress(done: int) -> None:
        if progress_every <= 0:
            return
        if done == total_omega or done == 1 or (done % progress_every == 0):
            elapsed = time.perf_counter() - start_time
            frac = done / total_omega
            eta_s = (elapsed / frac) - elapsed if frac > 0.0 else float("nan")
            print(
                f"[progress] {done}/{total_omega} ({100.0 * frac:5.1f}%) "
                f"elapsed={elapsed:7.1f}s eta={eta_s:7.1f}s"
            )

    tvals: npt.NDArray[np.float64]
    if args.nk <= 1:
        tvals_list: list[float] = []
        for i, w in enumerate(omegas, start=1):
            tvals_list.append(
                transmission(
                    w,
                    device=device,
                    lead_left=lead,
                    lead_right=lead,
                    kpar=(0.0,),
                    eta=args.eta,
                    eta_device=args.eta_device,
                    surface_gf_method=args.surface_gf_method,
                    omega_scale=omega_scale,
                )
            )
            _report_progress(i)
        tvals = np.asarray(tvals_list, dtype=float)
    else:
        kys = _kmesh_1d(args.nk, mode=args.kmesh)
        kpts = [(ky,) for ky in kys]
        tvals_list: list[float] = []
        for i, w in enumerate(omegas, start=1):
            if w <= 0.0:
                tvals_list.append(0.0)
                _report_progress(i)
                continue
            tavg, _ = transmission_kavg_adaptive(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                kpoints=kpts,
                eta_values=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4),
                eta_device=args.eta_device,
                min_success_fraction=0.7,
                surface_gf_method=args.surface_gf_method,
                omega_scale=omega_scale,
                nonnegative_tolerance=1e-8,
                max_channel_factor=1.5,
                collect_rejected=False,
            )
            tvals_list.append(float(tavg))
            _report_progress(i)
        tvals = np.asarray(tvals_list, dtype=float)

    tvals = np.where(tvals < 0.0, np.maximum(tvals, -1e-12), tvals)
    tvals = np.clip(tvals, 0.0, None)

    if args.x_unit == "cm-1":
        xvals = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
        xlabel = r"Frequency (cm$^{-1}$)"
    else:
        xvals = np.asarray(qe_omega_to_thz(omegas), dtype=float)
        xlabel = "Frequency (THz)"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(xvals, tvals, color="tab:blue", lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$T(\omega)$")
    if args.nk <= 1:
        ax.set_title(f"Graphene Transmission (Gamma-only, {args.surface_gf_method})")
    else:
        ax.set_title(f"Graphene Transmission (ky-avg, nk={args.nk}, {args.surface_gf_method})")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)

    print(f"Saved transmission plot: {args.save}")
    print(f"omega range (QE units): [{args.omega_min:.6e}, {omega_max:.6e}] ({omega_max_source})")
    print(f"eta (lead)={args.eta:.3e}, eta_device={args.eta_device:.3e}")
    print(
        "omega_max conversions: "
        f"{float(np.asarray(qe_omega_to_thz(omega_max), dtype=float)):.3f} THz, "
        f"{float(np.asarray(qe_omega_to_cm1(omega_max), dtype=float)):.3f} cm^-1"
    )
    print(f"T(omega) min={float(np.min(tvals)):.6e}, max={float(np.max(tvals)):.6e}")


if __name__ == "__main__":
    main()
