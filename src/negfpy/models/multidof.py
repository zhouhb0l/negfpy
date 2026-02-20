"""General multi-atom, multi-DOF 1D builders from force-constant blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks


Array = np.ndarray


@dataclass(frozen=True)
class MultiDofCellParams:
    """Principal-layer force constants and masses for a periodic 1D crystal.

    Notes:
    - ``fc00`` and ``fc01`` are force-constant blocks (not mass-normalized).
    - Block size is ``n_atoms * dof_per_atom``.
    - Dynamical blocks are built as ``D = M^{-1/2} Phi M^{-1/2}``.
    """

    masses: Array
    fc00: Array
    fc01: Array
    dof_per_atom: int = 3
    onsite_pinning: float = 0.0

    def __post_init__(self) -> None:
        masses = np.asarray(self.masses, dtype=float)
        if masses.ndim != 1 or masses.size == 0:
            raise ValueError("masses must be a non-empty 1D array.")
        if np.any(masses <= 0.0):
            raise ValueError("All masses must be positive.")
        if self.dof_per_atom <= 0:
            raise ValueError("dof_per_atom must be positive.")

        ndof = masses.size * self.dof_per_atom
        if self.fc00.shape != (ndof, ndof):
            raise ValueError("fc00 shape must be (n_atoms*dof_per_atom, n_atoms*dof_per_atom).")
        if self.fc01.shape != (ndof, ndof):
            raise ValueError("fc01 shape must match fc00 shape.")

    @property
    def n_atoms(self) -> int:
        return int(np.asarray(self.masses).size)

    @property
    def ndof(self) -> int:
        return self.n_atoms * self.dof_per_atom


def _inv_sqrt_mass_matrix(masses: Array, dof_per_atom: int) -> Array:
    masses = np.asarray(masses, dtype=float)
    m_rep = np.repeat(masses, dof_per_atom)
    return np.diag(1.0 / np.sqrt(m_rep))


def _dynamical_blocks(
    masses_left: Array,
    masses_right: Array,
    fc00: Array,
    fc01: Array,
    dof_per_atom: int,
    onsite_pinning: float = 0.0,
) -> tuple[Array, Array]:
    mls = _inv_sqrt_mass_matrix(masses_left, dof_per_atom)
    mrs = _inv_sqrt_mass_matrix(masses_right, dof_per_atom)

    fc00_eff = np.asarray(fc00, dtype=np.complex128)
    if onsite_pinning != 0.0:
        fc00_eff = fc00_eff + onsite_pinning * np.eye(fc00.shape[0], dtype=np.complex128)

    d00 = mls @ fc00_eff @ mls
    d01 = mls @ np.asarray(fc01, dtype=np.complex128) @ mrs
    return d00, d01


def multidof_lead_blocks(params: MultiDofCellParams) -> LeadBlocks:
    """Return periodic lead blocks for a multi-atom multi-DOF principal layer."""

    d00, d01 = _dynamical_blocks(
        masses_left=np.asarray(params.masses, dtype=float),
        masses_right=np.asarray(params.masses, dtype=float),
        fc00=params.fc00,
        fc01=params.fc01,
        dof_per_atom=params.dof_per_atom,
        onsite_pinning=params.onsite_pinning,
    )
    return LeadBlocks(d00=d00, d01=d01)


def multidof_device(
    n_layers: int,
    params: MultiDofCellParams,
    layer_masses: Array | None = None,
) -> Device1D:
    """Return finite device blocks from force constants.

    ``layer_masses`` optionally sets per-layer masses with shape ``(n_layers, n_atoms)``.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    if layer_masses is None:
        masses = np.repeat(np.asarray(params.masses, dtype=float)[None, :], n_layers, axis=0)
    else:
        masses = np.asarray(layer_masses, dtype=float)
        if masses.shape != (n_layers, params.n_atoms):
            raise ValueError("layer_masses shape must be (n_layers, n_atoms).")
        if np.any(masses <= 0.0):
            raise ValueError("layer_masses entries must be positive.")

    onsite_blocks: list[Array] = []
    for i in range(n_layers):
        d00_i, _ = _dynamical_blocks(
            masses_left=masses[i],
            masses_right=masses[i],
            fc00=params.fc00,
            fc01=params.fc01,
            dof_per_atom=params.dof_per_atom,
            onsite_pinning=params.onsite_pinning,
        )
        onsite_blocks.append(d00_i)

    coupling_blocks: list[Array] = []
    for i in range(n_layers - 1):
        _, d01_i = _dynamical_blocks(
            masses_left=masses[i],
            masses_right=masses[i + 1],
            fc00=params.fc00,
            fc01=params.fc01,
            dof_per_atom=params.dof_per_atom,
            onsite_pinning=0.0,
        )
        coupling_blocks.append(d01_i)

    return Device1D(onsite_blocks=onsite_blocks, coupling_blocks=coupling_blocks)

