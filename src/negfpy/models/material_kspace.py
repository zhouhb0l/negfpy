"""Material-oriented k-space builders for multi-atom, multi-DOF phonon transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import Device1D, DeviceKSpace, KPar, LeadKSpace


Array = np.ndarray
Shift2D = tuple[int, int]


@dataclass(frozen=True)
class MaterialKspaceParams:
    """Harmonic force-constant terms for x-transport with periodic transverse directions.

    Parameters
    - ``masses``: per-atom masses for one principal layer (length n_atoms).
    - ``dof_per_atom``: DOF per atom (typically 3 for x/y/z).
    - ``fc00_terms``: dict of same-layer force-constant blocks keyed by (dy, dz).
      The k-space block is sum[ Phi00(dy,dz) * exp(i*(ky*dy + kz*dz)) ].
    - ``fc01_terms``: dict of +x-layer force-constant blocks keyed by (dy, dz).
      The k-space block is sum[ Phi01(dy,dz) * exp(i*(ky*dy + kz*dz)) ].
    - ``fc10_terms``: optional dict of -x-layer force-constant blocks keyed by
      (dy, dz). When omitted, ``fc10_terms`` is assumed Hermitian-conjugate of
      ``fc01_terms``.
    - ``onsite_pinning``: optional small positive onsite term added to Phi00(k).
    """

    masses: Array
    fc00_terms: dict[Shift2D, Array]
    fc01_terms: dict[Shift2D, Array]
    fc10_terms: dict[Shift2D, Array] | None = None
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
        if len(self.fc00_terms) == 0:
            raise ValueError("fc00_terms must contain at least one term.")
        if len(self.fc01_terms) == 0:
            raise ValueError("fc01_terms must contain at least one term.")

        ndof = masses.size * self.dof_per_atom
        for key, block in self.fc00_terms.items():
            if len(key) != 2:
                raise ValueError("fc00_terms keys must be (dy, dz).")
            if block.shape != (ndof, ndof):
                raise ValueError("fc00_terms blocks must have shape (n_atoms*dof, n_atoms*dof).")
        for key, block in self.fc01_terms.items():
            if len(key) != 2:
                raise ValueError("fc01_terms keys must be (dy, dz).")
            if block.shape != (ndof, ndof):
                raise ValueError("fc01_terms blocks must have shape (n_atoms*dof, n_atoms*dof).")
        if self.fc10_terms is not None:
            if len(self.fc10_terms) == 0:
                raise ValueError("fc10_terms must be non-empty when provided.")
            for key, block in self.fc10_terms.items():
                if len(key) != 2:
                    raise ValueError("fc10_terms keys must be (dy, dz).")
                if block.shape != (ndof, ndof):
                    raise ValueError("fc10_terms blocks must have shape (n_atoms*dof, n_atoms*dof).")

    @property
    def n_atoms(self) -> int:
        return int(np.asarray(self.masses).size)

    @property
    def ndof(self) -> int:
        return self.n_atoms * self.dof_per_atom


def _parse_ky_kz(kpar: KPar) -> tuple[float, float]:
    if kpar is None:
        return 0.0, 0.0
    if len(kpar) == 0:
        return 0.0, 0.0
    if len(kpar) > 2:
        raise ValueError("kpar must contain at most two transverse components: (ky,) or (ky, kz).")
    if len(kpar) == 1:
        return float(kpar[0]), 0.0
    return float(kpar[0]), float(kpar[1])


def _inv_sqrt_mass_diag(masses: Array, dof_per_atom: int) -> Array:
    m_rep = np.repeat(np.asarray(masses, dtype=float), dof_per_atom)
    return np.diag(1.0 / np.sqrt(m_rep))


def _sum_k_terms(terms: dict[Shift2D, Array], ky: float, kz: float) -> Array:
    nd = next(iter(terms.values())).shape[0]
    out = np.zeros((nd, nd), dtype=np.complex128)
    for (dy, dz), block in terms.items():
        out += np.asarray(block, dtype=np.complex128) * np.exp(1j * (ky * dy + kz * dz))
    return out


def _build_dynamical_blocks(
    params: MaterialKspaceParams,
    ky: float,
    kz: float,
    masses_left: Array,
    masses_right: Array,
) -> tuple[Array, Array, Array]:
    phi00_k = _sum_k_terms(params.fc00_terms, ky=ky, kz=kz)
    if params.onsite_pinning != 0.0:
        phi00_k = phi00_k + params.onsite_pinning * np.eye(params.ndof, dtype=np.complex128)
    phi01_k = _sum_k_terms(params.fc01_terms, ky=ky, kz=kz)
    if params.fc10_terms is None:
        phi10_k = phi01_k.conj().T
    else:
        phi10_k = _sum_k_terms(params.fc10_terms, ky=ky, kz=kz)

    ml = _inv_sqrt_mass_diag(masses_left, params.dof_per_atom)
    mr = _inv_sqrt_mass_diag(masses_right, params.dof_per_atom)
    d00 = ml @ phi00_k @ ml
    d01 = ml @ phi01_k @ mr
    d10 = mr @ phi10_k @ ml
    return d00, d01, d10


def material_kspace_lead(params: MaterialKspaceParams) -> LeadKSpace:
    """Return a k_parallel-dependent lead from general force-constant terms."""

    def _builder(kpar: KPar) -> tuple[Array, Array, Array]:
        ky, kz = _parse_ky_kz(kpar)
        return _build_dynamical_blocks(
            params=params,
            ky=ky,
            kz=kz,
            masses_left=params.masses,
            masses_right=params.masses,
        )

    return LeadKSpace(blocks_builder=_builder)


def material_kspace_device(
    n_layers: int,
    params: MaterialKspaceParams,
    layer_masses: Array | None = None,
) -> DeviceKSpace:
    """Return a k_parallel-dependent device from general force-constant terms.

    ``layer_masses`` optionally provides per-layer masses with shape
    ``(n_layers, n_atoms)`` to represent mass disorder along x.
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
            raise ValueError("All layer_masses entries must be positive.")

    def _builder(kpar: KPar) -> Device1D:
        ky, kz = _parse_ky_kz(kpar)
        onsite_blocks: list[Array] = []
        for i in range(n_layers):
            d00_i, _, _ = _build_dynamical_blocks(
                params=params,
                ky=ky,
                kz=kz,
                masses_left=masses[i],
                masses_right=masses[i],
            )
            onsite_blocks.append(d00_i)

        coupling_blocks: list[Array] = []
        for i in range(n_layers - 1):
            _, d01_i, _ = _build_dynamical_blocks(
                params=params,
                ky=ky,
                kz=kz,
                masses_left=masses[i],
                masses_right=masses[i + 1],
            )
            coupling_blocks.append(d01_i)
        return Device1D(onsite_blocks=onsite_blocks, coupling_blocks=coupling_blocks)

    return DeviceKSpace(device_builder=_builder)
