"""Core data structures for block-based phonon NEGF."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
KPar = tuple[float, ...] | None


@dataclass(frozen=True)
class LeadBlocks:
    """Periodic lead represented by nearest-neighbor principal-layer blocks."""

    d00: Array
    d01: Array
    d10: Array | None = None

    def __post_init__(self) -> None:
        if self.d00.ndim != 2 or self.d00.shape[0] != self.d00.shape[1]:
            raise ValueError("d00 must be a square 2D array.")
        if self.d01.ndim != 2:
            raise ValueError("d01 must be a 2D array.")
        if self.d01.shape != self.d00.shape:
            raise ValueError("d01 must have the same shape as d00.")
        if self.d10 is not None:
            if self.d10.ndim != 2:
                raise ValueError("d10 must be a 2D array when provided.")
            if self.d10.shape != self.d00.shape:
                raise ValueError("d10 must have the same shape as d00.")


@dataclass(frozen=True)
class LeadKSpace:
    """k_parallel-dependent lead block provider."""

    blocks_builder: Callable[[KPar], tuple[Array, Array] | tuple[Array, Array, Array]]

    def blocks(self, kpar: KPar = None) -> tuple[Array, Array] | tuple[Array, Array, Array]:
        return self.blocks_builder(kpar)


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
        first_shape = self.onsite_blocks[0].shape
        if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
            raise ValueError("Each onsite block must be a square 2D array.")
        for block in self.onsite_blocks:
            if block.shape != first_shape:
                raise ValueError("All onsite blocks must share the same shape.")
        for block in self.coupling_blocks:
            if block.shape != first_shape:
                raise ValueError("All coupling blocks must match onsite block shape.")

    @property
    def n_layers(self) -> int:
        return len(self.onsite_blocks)

    @property
    def dof_per_layer(self) -> int:
        return int(self.onsite_blocks[0].shape[0])


@dataclass(frozen=True)
class DeviceKSpace:
    """k_parallel-dependent device block provider."""

    device_builder: Callable[[KPar], Device1D]

    def device(self, kpar: KPar = None) -> Device1D:
        return self.device_builder(kpar)
