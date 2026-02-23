"""Utility transforms for IFC datasets."""

from __future__ import annotations

import numpy as np

from .schema import IFCData, IFCTerm


def enforce_translational_asr_on_self_term(ifc: IFCData) -> tuple[IFCData, float]:
    """Enforce translational ASR by correcting only the R=(0,0,0) IFC block.

    Returns:
        (corrected_ifc, max_residual_before_correction)
    """

    ndof = len(ifc.masses) * ifc.dof_per_atom
    terms = list(ifc.terms)
    phi_sum = np.zeros((ndof, ndof), dtype=np.complex128)
    r0_idx: int | None = None
    for idx, term in enumerate(terms):
        block = np.asarray(term.block, dtype=np.complex128)
        phi_sum += block
        if term.dx == 0 and term.dy == 0 and term.dz == 0:
            r0_idx = idx
    if r0_idx is None:
        raise ValueError("Missing IFC term at translation (0,0,0); cannot enforce ASR.")

    residual = np.zeros((ndof, ifc.dof_per_atom), dtype=np.complex128)
    for beta in range(ifc.dof_per_atom):
        cols = np.arange(beta, ndof, ifc.dof_per_atom)
        residual[:, beta] = np.sum(phi_sum[:, cols], axis=1)
    residual_max = float(np.max(np.abs(residual)))

    corrected = np.asarray(terms[r0_idx].block, dtype=np.complex128).copy()
    for row in range(ndof):
        atom_i = row // ifc.dof_per_atom
        for beta in range(ifc.dof_per_atom):
            col = atom_i * ifc.dof_per_atom + beta
            corrected[row, col] -= residual[row, beta]
    corrected = 0.5 * (corrected + corrected.conj().T)
    terms[r0_idx] = IFCTerm(dx=0, dy=0, dz=0, block=corrected)

    metadata = dict(ifc.metadata)
    metadata["asr_enforced"] = True
    return (
        IFCData(
            masses=np.asarray(ifc.masses, dtype=float),
            dof_per_atom=ifc.dof_per_atom,
            terms=tuple(terms),
            units=ifc.units,
            metadata=metadata,
            lattice_vectors=ifc.lattice_vectors,
            atom_positions=ifc.atom_positions,
            atom_symbols=ifc.atom_symbols,
            index_convention=ifc.index_convention,
        ),
        residual_max,
    )
