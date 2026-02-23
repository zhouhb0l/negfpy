"""Plot bulk phonon DOS from IFC with full periodicity (kx, ky, kz)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import bulk_phonon_dos_3d, bulk_phonon_dos_3d_from_ifc_terms
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1, qe_omega_to_thz
from negfpy.models import material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def _default_kgrid(nk: int, centered: bool) -> np.ndarray:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if centered:
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step
    return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk)


def _estimate_omega_max_from_ifc(
    ifc,
    *,
    nkx: int,
    nky: int,
    nkz: int,
    centered_grid: bool,
    safety_factor: float = 1.02,
) -> float:
    """Estimate max phonon frequency in internal QE omega units from IFC."""

    kx = _default_kgrid(nkx, centered=centered_grid)
    ky = _default_kgrid(nky, centered=centered_grid)
    kz = _default_kgrid(nkz, centered=centered_grid)
    ndof = len(ifc.masses) * ifc.dof_per_atom
    mrep = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(mrep))
    terms = [(int(t.dx), int(t.dy), int(t.dz), np.asarray(t.block, dtype=np.complex128)) for t in ifc.terms]

    wmax = 0.0
    for kxv in kx:
        for kyv in ky:
            for kzv in kz:
                phi = np.zeros((ndof, ndof), dtype=np.complex128)
                for dx, dy, dz, block in terms:
                    phi += block * np.exp(1j * (kxv * dx + kyv * dy + kzv * dz))
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
    parser.add_argument("--omega-max", type=float, default=None, help="Internal QE omega unit; if omitted, auto-estimated from IFC")
    parser.add_argument("--fmax-thz", type=float, default=None, help="Upper plotting/calculation limit in THz (overrides --omega-max)")
    parser.add_argument("--x-unit", type=str, default="cm-1", choices=["cm-1", "thz", "qe"], help="X-axis unit")
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

    if args.fmax_thz is not None:
        omega_max = _thz_to_qe_omega(float(args.fmax_thz))
        omega_max_source = f"from --fmax-thz={args.fmax_thz:g}"
    elif args.omega_max is not None:
        omega_max = float(args.omega_max)
        omega_max_source = "from --omega-max"
    else:
        omega_max = _estimate_omega_max_from_ifc(
            ifc,
            nkx=args.nkx,
            nky=args.nky,
            nkz=args.nkz,
            centered_grid=args.centered_grid,
        )
        omega_max_source = "auto-estimated from IFC"
    if omega_max <= args.omega_min:
        raise ValueError("Resolved omega-max must be greater than omega-min.")

    omegas = np.linspace(args.omega_min, omega_max, args.nw)
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

    if args.x_unit == "thz":
        x = np.asarray(qe_omega_to_thz(omegas), dtype=float)
        xlabel = "Frequency (THz)"
    elif args.x_unit == "qe":
        x = np.asarray(omegas, dtype=float)
        xlabel = "Frequency (QE internal omega unit)"
    else:
        x = np.asarray(qe_omega_to_cm1(omegas), dtype=float)
        xlabel = r"$\omega$ (cm$^{-1}$)"

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(x, dos, color="tab:green", lw=1.6)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Bulk DOS (arb. unit)")
    ax.set_title(
        "Bulk Phonon DOS "
        f"[{args.method}] (nk={args.nkx}x{args.nky}x{args.nkz}, sigma={args.sigma:g})"
    )
    ax.grid(alpha=0.3)
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)
    print(f"Saved bulk DOS plot: {args.save}")
    print(f"omega range (QE units): [{args.omega_min:.6e}, {omega_max:.6e}] ({omega_max_source})")
    print(
        "omega_max conversions: "
        f"{float(np.asarray(qe_omega_to_thz(omega_max), dtype=float)):.3f} THz, "
        f"{float(np.asarray(qe_omega_to_cm1(omega_max), dtype=float)):.3f} cm^-1"
    )


if __name__ == "__main__":
    main()
