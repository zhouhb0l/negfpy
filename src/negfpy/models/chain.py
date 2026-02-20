"""Toy 1D chain builders for phonon NEGF benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks


Array = np.ndarray


@dataclass(frozen=True)
class ChainParams:
    mass: float
    spring: float


def lead_blocks(params: ChainParams) -> LeadBlocks:
    d00 = np.array([[2.0 * params.spring / params.mass]], dtype=float)
    d01 = np.array([[-params.spring / params.mass]], dtype=float)
    return LeadBlocks(d00=d00, d01=d01)


def device_perfect_chain(n_layers: int, params: ChainParams) -> Device1D:
    # One onsite block per principal layer; these fill the block diagonal.
    # For this monoatomic 1D model each block is 1x1 with value 2k/m.
    onsite = [np.array([[2.0 * params.spring / params.mass]], dtype=float) for _ in range(n_layers)]
    # One nearest-neighbor coupling block between adjacent layers; these fill
    # the first off-diagonals (upper/lower via Hermitian transpose in assembly).
    # There are n_layers - 1 such links, each 1x1 with value -k/m.
    coupling = [np.array([[-params.spring / params.mass]], dtype=float) for _ in range(n_layers - 1)]
    return Device1D(onsite_blocks=onsite, coupling_blocks=coupling)


def device_mass_defect(
    n_layers: int,
    params: ChainParams,
    defect_index: int,
    defect_mass: float,
) -> Device1D:
    if defect_index < 0 or defect_index >= n_layers:
        raise ValueError("defect_index out of range")

    onsite = []
    for i in range(n_layers):
        mass = defect_mass if i == defect_index else params.mass
        onsite.append(np.array([[2.0 * params.spring / mass]], dtype=float))

    coupling = [np.array([[-params.spring / params.mass]], dtype=float) for _ in range(n_layers - 1)]
    return Device1D(onsite_blocks=onsite, coupling_blocks=coupling)


def analytic_band_max(params: ChainParams) -> float:
    return float(2.0 * np.sqrt(params.spring / params.mass))
