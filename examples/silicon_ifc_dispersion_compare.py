"""Compare silicon phonon dispersion (Gamma->X) from primitive IFC vs supercell lead model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import lead_dynamical_matrix
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.models import material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def _primitive_dispersion_gamma_x(ifc, kx: np.ndarray) -> np.ndarray:
    ndof = len(ifc.masses) * ifc.dof_per_atom
    m = np.repeat(np.asarray(ifc.masses, dtype=float), ifc.dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(m))
    out = np.zeros((kx.size, ndof), dtype=float)
    terms = [(t.dx, t.dy, t.dz, np.asarray(t.block, dtype=np.complex128)) for t in ifc.terms]
    for i, k in enumerate(kx):
        phi = np.zeros((ndof, ndof), dtype=np.complex128)
        for dx, dy, dz, block in terms:
            # Gamma->X path: ky=kz=0.
            phi += block * np.exp(1j * (k * dx))
        dmat = minv @ phi @ minv
        dmat = 0.5 * (dmat + dmat.conj().T)
        w2 = np.linalg.eigvalsh(dmat)
        out[i, :] = np.sqrt(np.clip(w2.real, 0.0, None))
    return out


def _supercell_dispersion_gamma_x(ifc, kx: np.ndarray, principal_layer_size: int) -> np.ndarray:
    cfg = BuildConfig(principal_layer_size=principal_layer_size, infer_fc01_from_negative_dx=True, onsite_pinning=0.0)
    params = build_material_kspace_params(ifc=ifc, config=cfg)
    lead = material_kspace_lead(params)
    out = np.zeros((kx.size, params.ndof), dtype=float)
    for i, k in enumerate(kx):
        dmat = lead_dynamical_matrix(lead=lead, kx=float(k), kpar=(0.0, 0.0), hermitize=True)
        w2 = np.linalg.eigvalsh(dmat)
        out[i, :] = np.sqrt(np.clip(w2.real, 0.0, None))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ifc", type=Path, default=Path("si444.fc"))
    parser.add_argument("--reader", type=str, default="qe")
    parser.add_argument("--pl-size", type=int, default=2, help="Supercell principal layer size")
    parser.add_argument("--nkx", type=int, default=240)
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_dispersion_compare_gamma_x.png"))
    args = parser.parse_args()

    ifc = _read_ifc(args.ifc, reader=args.reader)
    kx = np.linspace(0.0, np.pi, args.nkx)

    w_prim = _primitive_dispersion_gamma_x(ifc=ifc, kx=kx)
    w_sc = _supercell_dispersion_gamma_x(ifc=ifc, kx=kx, principal_layer_size=args.pl_size)

    y_prim = np.asarray(qe_omega_to_cm1(w_prim), dtype=float)
    y_sc = np.asarray(qe_omega_to_cm1(w_sc), dtype=float)

    fig, ax = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True)
    for m in range(y_prim.shape[1]):
        ax[0].plot(kx / np.pi, y_prim[:, m], color="tab:blue", lw=1.0, alpha=0.65)
    for m in range(y_sc.shape[1]):
        ax[1].plot(kx / np.pi, y_sc[:, m], color="tab:orange", lw=1.0, alpha=0.65)

    ax[0].set_title("Primitive IFC (direct)")
    ax[1].set_title(f"Supercell lead-model (PL={args.pl_size})")
    for a in ax:
        a.set_xlabel(r"kx / pi  (Gamma->X)")
        a.grid(alpha=0.25)
    ax[0].set_ylabel("Frequency (cm^-1)")
    fig.suptitle("Silicon Dispersion Comparison Along Gamma->X (ky=kz=0)")
    fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)

    print(f"Saved comparison plot: {args.save}")
    print(f"Primitive modes: {y_prim.shape[1]}, Supercell modes: {y_sc.shape[1]}")
    print(f"Gamma (primitive, first 6 cm^-1): {np.sort(y_prim[0])[:6]}")
    print(f"Gamma (supercell, first 6 cm^-1): {np.sort(y_sc[0])[:6]}")


if __name__ == "__main__":
    main()
