"""Plot silicon lead phonon DOS from IFC data (left/right leads)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import lead_surface_dos, lead_surface_dos_kavg_adaptive
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.models import material_kspace_lead


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


def _build_lead(ifc_path: Path, reader: str, principal_layer_size: int) -> object:
    ifc = _read_ifc(ifc_path, reader=reader)
    cfg = BuildConfig(principal_layer_size=principal_layer_size, infer_fc01_from_negative_dx=True, onsite_pinning=0.0)
    params = build_material_kspace_params(ifc=ifc, config=cfg)
    return material_kspace_lead(params)


def _dos_spectrum(
    lead: object,
    omegas: np.ndarray,
    nk: int,
    kmesh: str,
) -> np.ndarray:
    if nk <= 1:
        vals = [lead_surface_dos(float(w), lead=lead, eta=1e-6, kpar=(0.0, 0.0)) for w in omegas]
        arr = np.asarray(vals, dtype=float)
        return np.clip(arr, 0.0, None)

    kvals = _kmesh(nk, mode=kmesh)
    kpts = [(float(ky), float(kz)) for ky in kvals for kz in kvals]
    vals: list[float] = []
    for w in omegas:
        if w <= 0.0:
            vals.append(0.0)
            continue
        dval, _ = lead_surface_dos_kavg_adaptive(
            omega=float(w),
            lead=lead,
            kpoints=kpts,
            eta_values=(1e-8, 1e-7, 1e-6, 1e-5),
            min_success_fraction=0.7,
        )
        vals.append(float(max(dval, 0.0)))
    return np.asarray(vals, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-ifc", type=Path, default=Path("si444.fc"))
    parser.add_argument("--right-ifc", type=Path, default=None)
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--principal-layer-size", type=int, default=2)
    parser.add_argument("--nk", type=int, default=12)
    parser.add_argument("--kmesh", type=str, choices=["shifted", "centered"], default="shifted")
    parser.add_argument("--nw", type=int, default=180)
    parser.add_argument("--omega-min", type=float, default=0.0, help="Internal QE omega unit")
    parser.add_argument("--omega-max", type=float, default=0.006, help="Internal QE omega unit")
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_lead_dos.png"))
    args = parser.parse_args()

    right_ifc = args.right_ifc if args.right_ifc is not None else args.left_ifc
    left_lead = _build_lead(args.left_ifc, reader=args.reader, principal_layer_size=args.principal_layer_size)
    right_lead = _build_lead(right_ifc, reader=args.reader, principal_layer_size=args.principal_layer_size)

    omegas = np.linspace(args.omega_min, args.omega_max, args.nw)
    dos_left = _dos_spectrum(left_lead, omegas, nk=args.nk, kmesh=args.kmesh)
    dos_right = _dos_spectrum(right_lead, omegas, nk=args.nk, kmesh=args.kmesh)

    x = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, dos_left, lw=1.6, color="tab:blue", label="Left lead")
    ax.plot(x, dos_right, lw=1.4, color="tab:orange", ls="--", label="Right lead")
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Surface DOS (arb. unit)")
    ax.set_title(f"Silicon Lead Phonon DOS (nk={args.nk}x{args.nk}, {args.kmesh})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved DOS plot: {args.save}")
    print(f"Max |DOS_L - DOS_R| = {float(np.max(np.abs(dos_left - dos_right))):.6e}")


if __name__ == "__main__":
    main()
