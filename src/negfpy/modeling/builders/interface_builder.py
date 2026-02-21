"""Build interface/contact matrices from InterfaceData schema."""

from __future__ import annotations

import numpy as np

from negfpy.modeling.schema import InterfaceData


Array = np.ndarray


def build_interface_contacts(interface: InterfaceData, ndof: int) -> tuple[Array | None, Array | None]:
    def _validate(mat: Array | None, side: str) -> Array | None:
        if mat is None:
            return None
        arr = np.asarray(mat, dtype=np.complex128)
        if arr.shape != (ndof, ndof):
            raise ValueError(f"{side}_contact must have shape (ndof, ndof).")
        return arr

    return _validate(interface.left_contact, "left"), _validate(interface.right_contact, "right")

