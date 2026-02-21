"""Intermediate schema for force-constant based model construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class IFCTerm:
    """One real-space force-constant block connecting translated layers/cells."""

    dx: int
    dy: int
    dz: int
    block: Array


@dataclass(frozen=True)
class IFCData:
    """Unified intermediate representation for lattice IFC data."""

    masses: Array
    dof_per_atom: int
    terms: tuple[IFCTerm, ...]
    units: str = "unknown"
    metadata: dict[str, Any] = field(default_factory=dict)
    lattice_vectors: Array | None = None
    atom_positions: Array | None = None
    atom_symbols: tuple[str, ...] | None = None
    index_convention: str = "layer-major"

