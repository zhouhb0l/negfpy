"""Validation helpers for IFC intermediate schema."""

from __future__ import annotations

import numpy as np

from negfpy.modeling.schema import IFCData


def validate_ifc_data(ifc: IFCData) -> None:
    masses = np.asarray(ifc.masses, dtype=float)
    if masses.ndim != 1 or masses.size == 0:
        raise ValueError("IFCData.masses must be a non-empty 1D array.")
    if np.any(masses <= 0.0):
        raise ValueError("All IFCData masses must be positive.")
    if ifc.dof_per_atom <= 0:
        raise ValueError("IFCData.dof_per_atom must be positive.")
    if len(ifc.terms) == 0:
        raise ValueError("IFCData.terms must contain at least one term.")

    ndof = masses.size * ifc.dof_per_atom
    for term in ifc.terms:
        block = np.asarray(term.block)
        if block.shape != (ndof, ndof):
            raise ValueError("All IFC terms must have shape (n_atoms*dof, n_atoms*dof).")


def validate_transport_connectivity(ifc: IFCData) -> None:
    has_dx0 = any(term.dx == 0 for term in ifc.terms)
    has_x_coupling = any(term.dx != 0 for term in ifc.terms)
    if not has_dx0:
        raise ValueError("IFCData must include at least one dx=0 term for onsite blocks.")
    if not has_x_coupling:
        raise ValueError("IFCData must include at least one nonzero dx term for layer coupling.")
