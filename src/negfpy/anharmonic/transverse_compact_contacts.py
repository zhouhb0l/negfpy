"""Exact boundary storage for the existing paired-q thermal contact builder.

Only contact rows are stored. ``device_indices`` retains their original order
in the full harmonic device; no lead, coupling, or zero-limit mathematics is
changed. Byte limits bound the same individual payloads as the dense builder,
not process RSS or temporary/library allocation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import LeadBlocks

from ._harmonic import normalize_contact_indices
from .acoustic_contacts import _lead_arrays, _validate_q_samples
from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .transverse_contacts import (
    ReservoirBuilder,
    TransverseContact,
    build_transverse_contact,
)
from .transverse_mapping import _integer, _kpoints, _real
from .transverse_solver import TransverseSpectrum
from .types import HarmonicReservoir

_MAX_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True, eq=False)
class CompactTransverseContact:
    """Contact-sized fields and their ordered embedding into the full device."""

    contact: TransverseContact
    full_ndof: int
    device_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.contact, TransverseContact):
            raise TypeError("contact must be a TransverseContact.")
        dimension = _integer(self.full_ndof, "full_ndof", minimum=1)
        indices = tuple(_integer(v, "device_indices", minimum=0) for v in self.device_indices)
        if len(indices) != self.contact.spectrum.retarded.shape[-1]:
            raise ValueError("device_indices must match the compact contact dimension.")
        if len(set(indices)) != len(indices) or max(indices) >= dimension:
            raise ValueError("device_indices must be unique and inside the full device.")
        object.__setattr__(self, "full_ndof", dimension)
        object.__setattr__(self, "device_indices", indices)

    @property
    def spectrum(self) -> TransverseSpectrum:
        return self.contact.spectrum

    @property
    def mesh(self) -> UniformBZMesh:
        return self.contact.mesh

    @property
    def grid(self) -> SignedFrequencyGrid:
        return self.contact.grid

    @property
    def zero_limits(self) -> np.ndarray:
        return self.contact.zero_limits


def build_compact_transverse_contact(
    mesh: UniformBZMesh,
    grid: SignedFrequencyGrid,
    reservoirs: ReservoirBuilder,
    ndof: int,
    eta_lead: float,
    *,
    side: str,
    surface_gf_method: str = "sancho_rubio",
    q_samples: int = 257,
    max_bytes: int = _MAX_BYTES,
    max_base_evaluations: int = 100_000,
) -> CompactTransverseContact:
    """Build boundary fields with the original full-device contact embedding.

    Resolve default left/right coupling and indices against ``ndof`` first,
    then relabel those rows in their existing order to a compact local basis.
    The unchanged contact builder supplies every physical validation, signed
    spectrum and continuous fixed-eta zero limit. Each callback is called once
    and its arrays are detached before another q is requested.
    """
    kpoints = _kpoints(mesh)
    if not isinstance(grid, SignedFrequencyGrid) or not callable(reservoirs):
        raise TypeError("Supply a SignedFrequencyGrid and reservoir callback.")
    dimension = _integer(ndof, "ndof", minimum=1)
    eta = _real(eta_lead, "eta_lead")
    if eta <= 0 or side not in ("left", "right"):
        raise ValueError("Positive eta_lead and side='left' or 'right' are required.")
    samples = _validate_q_samples(q_samples)
    limit = min(_integer(max_bytes, "max_bytes", minimum=1), _MAX_BYTES)
    work = _integer(max_base_evaluations, "max_base_evaluations", minimum=1)
    if not isinstance(surface_gf_method, str):
        raise TypeError("surface_gf_method must be a string.")
    # At least one contacted coordinate is required. The actual compact size
    # cannot be known before the first callback, but work can be bounded now.
    unit_payload = (3 * grid.n_points + 1) * mesh.n_qpoints * 16
    if unit_payload > limit:
        raise ValueError("Contact spectra and zero limits exceed max_bytes.")
    if mesh.n_qpoints * (grid.n_points + 2 * samples) > work:
        raise ValueError("Contact construction exceeds max_base_evaluations.")

    sources: dict[tuple[float, float], HarmonicReservoir] = {}
    embedding: tuple[int, ...] | None = None
    lead_payload = 0
    for point in kpoints:
        key = tuple(float(value) for value in point)
        raw = reservoirs(key)
        if not isinstance(raw, HarmonicReservoir):
            raise TypeError("Each callback result must be a HarmonicReservoir.")
        if raw.lead.d00.ndim != 2 or raw.lead.d00.shape[0] < 1:
            raise ValueError("lead d00 must be a nonempty square matrix.")
        size = raw.lead.d00.shape[0]
        if 16 * size**4 > limit:
            raise ValueError("Surface derivative operator exceeds max_bytes.")
        coupling = raw.resolved_device_to_lead(side)
        indices = tuple(int(value) for value in normalize_contact_indices(raw, dimension, side))
        if embedding is None:
            embedding = indices
        elif indices != embedding:
            raise ValueError("Reservoir device embedding must be identical across q.")
        if unit_payload * len(indices)**2 > limit:
            raise ValueError("Contact spectra and zero limits exceed max_bytes.")
        lead_payload += 16 * (3 * size**2 + coupling.size)
        if lead_payload > limit:
            raise ValueError("Input lead blocks exceed max_bytes.")
        d00, d01, d10, _, _ = _lead_arrays(raw.lead)
        lead = LeadBlocks(*(np.array(a, dtype=complex, copy=True) for a in (d00, d01, d10)))
        sources[key] = HarmonicReservoir(
            lead, raw.temperature, device_to_lead=coupling,
            device_indices=tuple(range(len(indices))),
        )

    # A valid transverse mesh always contains Gamma and hence one source.
    assert embedding is not None
    contact = build_transverse_contact(
        mesh, grid, sources.__getitem__, len(embedding), eta, side=side,
        surface_gf_method=surface_gf_method, q_samples=samples, max_bytes=limit,
        max_base_evaluations=work,
    )
    return CompactTransverseContact(contact, dimension, embedding)


__all__ = ["CompactTransverseContact", "build_compact_transverse_contact"]
