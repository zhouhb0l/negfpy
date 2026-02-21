"""Phonopy IFC adapter to the unified IFC schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.modeling.validators import validate_ifc_data


def _load_source_payload(source: Any) -> dict[str, Any]:
    if isinstance(source, dict):
        return source
    path = Path(source)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_term(item: dict[str, Any]) -> IFCTerm:
    if "translation" in item:
        dx, dy, dz = item["translation"]
    else:
        dx, dy, dz = item["dx"], item["dy"], item["dz"]
    block = np.asarray(item["block"], dtype=np.complex128)
    return IFCTerm(dx=int(dx), dy=int(dy), dz=int(dz), block=block)


def read_phonopy_ifc(source: Any) -> IFCData:
    """Parse a normalized phonopy-like payload into IFCData.

    Supported input:
    - ``dict`` with keys ``masses``, ``dof_per_atom``, ``terms``.
    - JSON file path containing the same structure.
    """

    payload = _load_source_payload(source)
    terms = tuple(_parse_term(item) for item in payload["terms"])
    ifc = IFCData(
        masses=np.asarray(payload["masses"], dtype=float),
        dof_per_atom=int(payload.get("dof_per_atom", 3)),
        terms=terms,
        units=str(payload.get("units", "unknown")),
        metadata=dict(payload.get("metadata", {})),
        lattice_vectors=(
            np.asarray(payload["lattice_vectors"], dtype=float) if payload.get("lattice_vectors") is not None else None
        ),
        atom_positions=(
            np.asarray(payload["atom_positions"], dtype=float) if payload.get("atom_positions") is not None else None
        ),
        atom_symbols=tuple(payload["atom_symbols"]) if payload.get("atom_symbols") is not None else None,
        index_convention=str(payload.get("index_convention", "layer-major")),
    )
    validate_ifc_data(ifc)
    return ifc

