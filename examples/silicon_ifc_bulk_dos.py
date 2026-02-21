"""Plot silicon bulk phonon DOS with full 3D periodicity (kx, ky, kz)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import bulk_phonon_dos_3d, bulk_phonon_dos_3d_from_ifc_terms
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.models import material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("si444.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--principal-layer-size", type=int, default=2)
    parser.add_argument("--nkx", type=int, default=20)
    parser.add_argument("--nky", type=int, default=20)
    parser.add_argument("--nkz", type=int, default=20)
    parser.add_argument(
        "--method",
        type=str,
        default="direct_ifc",
        choices=["direct_ifc", "lead_model"],
        help="direct_ifc matches primitive bulk IFC Fourier sum; lead_model uses transport lead mapping",
    )
    parser.add_argument("--nw", type=int, default=500)
    parser.add_argument("--omega-min", type=float, default=0.0, help="Internal QE omega unit")
    parser.add_argument("--omega-max", type=float, default=0.006, help="Internal QE omega unit")
    parser.add_argument("--sigma", type=float, default=3e-5, help="Gaussian broadening in internal QE omega unit")
    parser.add_argument("--centered-grid", action="store_true", help="Use centered k-grid instead of midpoint grid")
    parser.add_argument(
        "--normalize-per-mode",
        action="store_true",
        help="Normalize DOS by number of modes in the chosen representation",
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_bulk_dos.png"))
    args = parser.parse_args()

    ifc = _read_ifc(args.ifc, reader=args.reader)
    cfg = BuildConfig(principal_layer_size=args.principal_layer_size, infer_fc01_from_negative_dx=True, onsite_pinning=0.0)
    params = build_material_kspace_params(ifc=ifc, config=cfg)
    lead = material_kspace_lead(params)

    omegas = np.linspace(args.omega_min, args.omega_max, args.nw)
    if args.method == "direct_ifc":
        term_tuples = [(t.dx, t.dy, t.dz, t.block) for t in ifc.terms]
        dos = bulk_phonon_dos_3d_from_ifc_terms(
            omegas=omegas,
            masses=np.asarray(ifc.masses, dtype=float),
            dof_per_atom=ifc.dof_per_atom,
            terms=term_tuples,
            nkx=args.nkx,
            nky=args.nky,
            nkz=args.nkz,
            sigma=args.sigma,
            centered_grid=args.centered_grid,
            normalize_per_mode=args.normalize_per_mode,
        )
    else:
        dos = bulk_phonon_dos_3d(
            lead=lead,
            omegas=omegas,
            nkx=args.nkx,
            nky=args.nky,
            nkz=args.nkz,
            sigma=args.sigma,
            centered_grid=args.centered_grid,
            normalize_per_mode=args.normalize_per_mode,
        )

    x = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, dos, color="tab:green", lw=1.6)
    ax.set_xlabel(r"$\omega$ (cm$^{-1}$)")
    ax.set_ylabel("Bulk DOS (arb. unit)")
    ax.set_title(
        "Silicon Bulk Phonon DOS "
        f"[{args.method}] (nk={args.nkx}x{args.nky}x{args.nkz}, sigma={args.sigma:g})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved bulk DOS plot: {args.save}")


if __name__ == "__main__":
    main()
