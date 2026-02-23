"""Compare silicon phonon dispersion (Gamma->X) from primitive IFC vs supercell lead model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from negfpy.core import lead_dynamical_matrix
from negfpy.io import read_ifc, read_phonopy_ifc, read_qe_q2r_ifc
from negfpy.modeling import BuildConfig, build_material_kspace_params, qe_omega_to_cm1
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.models import material_kspace_lead


def _read_ifc(path: Path, reader: str):
    if reader == "qe":
        return read_qe_q2r_ifc(path)
    if reader == "phonopy":
        return read_phonopy_ifc(path)
    return read_ifc(path, reader=reader)


def _enforce_asr_on_self_terms(ifc: IFCData) -> IFCData:
    ndof = len(ifc.masses) * ifc.dof_per_atom
    terms = list(ifc.terms)
    phi_sum = np.zeros((ndof, ndof), dtype=np.complex128)
    r0_idx = None
    for idx, term in enumerate(terms):
        phi_sum += np.asarray(term.block, dtype=np.complex128)
        if term.dx == 0 and term.dy == 0 and term.dz == 0:
            r0_idx = idx
    if r0_idx is None:
        raise ValueError("Cannot enforce ASR: missing IFC term at translation (0,0,0).")

    residual = np.zeros((ndof, ifc.dof_per_atom), dtype=np.complex128)
    for beta in range(ifc.dof_per_atom):
        cols = np.arange(beta, ndof, ifc.dof_per_atom)
        residual[:, beta] = np.sum(phi_sum[:, cols], axis=1)

    corrected = np.asarray(terms[r0_idx].block, dtype=np.complex128).copy()
    for row in range(ndof):
        atom_i = row // ifc.dof_per_atom
        for beta in range(ifc.dof_per_atom):
            col = atom_i * ifc.dof_per_atom + beta
            corrected[row, col] -= residual[row, beta]
    corrected = 0.5 * (corrected + corrected.conj().T)
    terms[r0_idx] = IFCTerm(dx=0, dy=0, dz=0, block=corrected)

    md = dict(ifc.metadata)
    md["asr_enforced"] = True
    return IFCData(
        masses=np.asarray(ifc.masses, dtype=float),
        dof_per_atom=ifc.dof_per_atom,
        terms=tuple(terms),
        units=ifc.units,
        metadata=md,
        lattice_vectors=ifc.lattice_vectors,
        atom_positions=ifc.atom_positions,
        atom_symbols=ifc.atom_symbols,
        index_convention=ifc.index_convention,
    )


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


def _primitive_dispersion_folded_supercell_bz(ifc, k_super: np.ndarray, pl_size: int) -> np.ndarray:
    """Fold primitive bands into the supercell BZ used by the lead model."""

    if pl_size <= 0:
        raise ValueError("pl_size must be positive.")
    ndof = len(ifc.masses) * ifc.dof_per_atom
    out = np.zeros((k_super.size, ndof * pl_size), dtype=float)

    for m in range(pl_size):
        k_prim = (k_super + 2.0 * np.pi * m) / float(pl_size)
        # Map to [-pi, pi) for numerical stability.
        k_prim = ((k_prim + np.pi) % (2.0 * np.pi)) - np.pi
        bands_m = _primitive_dispersion_gamma_x(ifc=ifc, kx=k_prim)
        out[:, m * ndof : (m + 1) * ndof] = bands_m
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
    parser.add_argument("--enforce-asr", action="store_true", help="Apply translational ASR correction")
    parser.add_argument("--overlay", action="store_true", help="Overlay primitive and supercell bands on one axis")
    parser.add_argument(
        "--fold-primitive",
        action="store_true",
        help="Fold primitive bands into supercell BZ for direct mode-by-mode comparison",
    )
    parser.add_argument("--save", type=Path, default=Path("outputs/silicon_dispersion_compare_gamma_x.png"))
    args = parser.parse_args()

    ifc = _read_ifc(args.ifc, reader=args.reader)
    if args.enforce_asr:
        ifc = _enforce_asr_on_self_terms(ifc)
    kx = np.linspace(0.0, np.pi, args.nkx)

    if args.fold_primitive:
        w_prim = _primitive_dispersion_folded_supercell_bz(ifc=ifc, k_super=kx, pl_size=args.pl_size)
    else:
        w_prim = _primitive_dispersion_gamma_x(ifc=ifc, kx=kx)
    w_sc = _supercell_dispersion_gamma_x(ifc=ifc, kx=kx, principal_layer_size=args.pl_size)

    y_prim = np.asarray(qe_omega_to_cm1(w_prim), dtype=float)
    y_sc = np.asarray(qe_omega_to_cm1(w_sc), dtype=float)

    if args.overlay:
        fig, ax = plt.subplots(1, 1, figsize=(8.4, 4.8))
        first_prim = True
        for m in range(y_prim.shape[1]):
            ax.plot(
                kx / np.pi,
                y_prim[:, m],
                color="tab:blue",
                lw=1.2,
                alpha=0.80 if args.fold_primitive else 0.90,
                label=("Primitive IFC (folded)" if args.fold_primitive else "Primitive IFC") if first_prim else None,
            )
            first_prim = False
        first_sc = True
        for m in range(y_sc.shape[1]):
            ax.scatter(
                kx / np.pi,
                y_sc[:, m],
                color="tab:orange",
                s=6.0,
                alpha=0.55,
                edgecolors="none",
                label=f"Supercell lead-model (PL={args.pl_size})" if first_sc else None,
            )
            first_sc = False
        ax.set_xlabel(r"kx / pi  (Gamma->X supercell BZ)")
        ax.set_ylabel("Frequency (cm^-1)")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right")
        ax.set_title("Dispersion Comparison Along Gamma->X (ky=kz=0)")
        fig.tight_layout()
    else:
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
        fig.suptitle("Dispersion Comparison Along Gamma->X (ky=kz=0)")
        fig.tight_layout()

    args.save.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=220)

    print(f"Saved comparison plot: {args.save}")
    print(f"Primitive modes: {y_prim.shape[1]}, Supercell modes: {y_sc.shape[1]}")
    print(f"Gamma (primitive, first 6 cm^-1): {np.sort(y_prim[0])[:6]}")
    print(f"Gamma (supercell, first 6 cm^-1): {np.sort(y_sc[0])[:6]}")


if __name__ == "__main__":
    main()
