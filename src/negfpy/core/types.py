"""Core data structures for block-based phonon NEGF."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class LeadBlocks:
    """Periodic lead represented by nearest-neighbor principal-layer blocks."""

    d00: Array
    d01: Array


@dataclass(frozen=True)
class Device1D:
    """Finite block-tridiagonal device with nearest-neighbor couplings."""

    onsite_blocks: list[Array]
    coupling_blocks: list[Array]

    def __post_init__(self) -> None:
        n_layers = len(self.onsite_blocks)
        if n_layers == 0:
            raise ValueError("Device must contain at least one principal layer.")
        if len(self.coupling_blocks) != n_layers - 1:
            raise ValueError("coupling_blocks length must be n_layers - 1.")

    @property
    def n_layers(self) -> int:
        return len(self.onsite_blocks)

    @property
    def dof_per_layer(self) -> int:
        return int(self.onsite_blocks[0].shape[0])
