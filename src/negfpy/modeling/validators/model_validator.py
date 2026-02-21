"""Validation helpers for built model objects."""

from __future__ import annotations

import numpy as np

from negfpy.models import MaterialKspaceParams


def validate_material_kspace_params(params: MaterialKspaceParams) -> None:
    ndof = params.ndof
    d00 = params.fc00_terms.get((0, 0))
    if d00 is None:
        raise ValueError("fc00_terms must contain the (0, 0) term.")
    if np.asarray(d00).shape != (ndof, ndof):
        raise ValueError("fc00_terms[(0, 0)] has invalid shape.")

