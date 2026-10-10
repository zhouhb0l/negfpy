"""Explicit harmonic lead padding for fixed-interaction buffer studies.

Padding moves reservoir boundaries; it does not enlarge the cubic interaction
region, rerun SCBA or establish buffer convergence. At finite lead eta with
zero device eta, moving the boundary also moves the regulator discontinuity.
"""

from dataclasses import dataclass, replace

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks

from .frequency_refinement import _immutable_array
from .sparse_cubic import SparseCubicVertex
from .types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    KeldyshSpectrum,
)


def _count(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < int(positive):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}.")
    return int(value)


def _copy(value):
    return _immutable_array(value, np.complex128)


@dataclass(frozen=True, slots=True)
class LeadBufferEmbedding:
    """New problem and explicit original-device coordinate correspondence."""

    problem: CubicSCBAProblem
    original_ndof: int
    left_layers: int
    right_layers: int

    @property
    def original_indices(self):
        offset = self.left_layers * self.problem.device.dof_per_layer
        return _immutable_array(np.arange(self.original_ndof) + offset, np.int64)

    @property
    def original_cuts(self):
        layers = self.original_ndof // self.problem.device.dof_per_layer
        return _immutable_array(np.arange(1, layers) + self.left_layers, np.int64)

    @property
    def metadata(self):
        return {
            "schema": "negfpy.research.lead-buffer-embedding.v1",
            "original_ndof": self.original_ndof, "extended_ndof": self.problem.ndof,
            "left_added_layers": self.left_layers, "right_added_layers": self.right_layers,
            "original_indices": self.original_indices.tolist(),
            "original_cuts": self.original_cuts.tolist(),
            "interaction_support_extended": False, "self_energy_recomputed": False,
            "buffer_convergence_verified": False, "changes_solver_acceptance": False,
            "finite_lead_eta_boundary_moves": True,
        }

    def embed_self_energy(self, sigma, *, max_output_bytes=512 * 2**20):
        """Zero-pad all three fixed fields exactly, without reconstruction.

        The byte limit covers returned arrays, not peak RSS: KeldyshSpectrum
        copies its inputs. This is a fixed-field diagnostic or fresh solver
        input, never a resumed controller/checkpoint or an accepted solution.
        """
        if not isinstance(sigma, KeldyshSpectrum):
            raise TypeError("sigma must be a KeldyshSpectrum.")
        expected = (self.problem.grid.n_points, self.original_ndof, self.original_ndof)
        if sigma.shape != expected:
            raise ValueError("Self-energy shape differs from the original problem grid/device.")
        limit = _count(max_output_bytes, "max_output_bytes", positive=True)
        shape = (sigma.n_frequency, self.problem.ndof, self.problem.ndof)
        needed = 3 * 16 * shape[0] * shape[1] * shape[2]
        if needed > limit:
            raise MemoryError(f"Embedded self-energy arrays need {needed} bytes; limit is {limit}.")
        indices = self.original_indices
        fields = {}
        for name in ("retarded", "lesser", "greater"):
            value = np.zeros(shape, dtype=np.complex128)
            value[:, indices[:, None], indices] = getattr(sigma, name)
            fields[name] = value
        return KeldyshSpectrum(**fields)


def add_harmonic_lead_buffers(problem, *, left_layers, right_layers,
                              max_dense_vertex_bytes=128 * 2**20):
    """Add explicit periodic lead layers without changing the interaction crop.

    Added lead layers must have the device principal-layer dimension. Contacts
    on a padded side must cover exactly that boundary layer; arbitrary ordering
    and explicit (including complex) contact matrices are preserved at the old
    device/new-buffer junction. Outer contacts become periodic lead contacts.
    Left and right leads and padding counts can differ. Counts must be explicit.
    Sparse vertices retain every coefficient and never become dense tensors.
    """
    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    left_layers = _count(left_layers, "left_layers")
    right_layers = _count(right_layers, "right_layers")
    if left_layers + right_layers == 0:
        raise ValueError("At least one lead buffer layer is required.")
    limit = _count(max_dense_vertex_bytes, "max_dense_vertex_bytes", positive=True)
    npl = problem.device.dof_per_layer
    offset = left_layers * npl
    ndof = problem.ndof + (left_layers + right_layers) * npl
    leads, junctions = {}, {}
    for side, count in (("left", left_layers), ("right", right_layers)):
        reservoir = getattr(problem, side)
        lead = reservoir.lead
        reverse = lead.d10 if lead.d10 is not None else lead.d01.conj().T
        if not all(np.isfinite(block).all() for block in (lead.d00, lead.d01, reverse)):
            raise ValueError("Lead blocks must be finite.")
        if not np.allclose(lead.d00, lead.d00.conj().T, rtol=1e-12, atol=1e-12):
            raise ValueError("Lead onsite block must be Hermitian.")
        if not np.allclose(reverse, lead.d01.conj().T, rtol=1e-12, atol=1e-12):
            raise ValueError("Lead reverse coupling must equal the forward adjoint.")
        leads[side] = LeadBlocks(_copy(lead.d00), _copy(lead.d01), _copy(reverse))
        if count:
            indices = reservoir.resolved_device_indices(dim=problem.ndof, side=side)
            boundary = np.arange(npl) + (0 if side == "left" else problem.ndof - npl)
            if lead.d00.shape != (npl, npl) or not np.array_equal(np.sort(indices), boundary):
                raise ValueError("Padded contact must cover exactly one full device boundary layer.")
            ordered = reservoir.resolved_device_to_lead(side)[np.argsort(indices)]
            junctions[side] = ordered.conj().T if side == "left" else ordered
    onsite = ([_copy(leads['left'].d00) for _ in range(left_layers)]
              + [_copy(block) for block in problem.device.onsite_blocks]
              + [_copy(leads['right'].d00) for _ in range(right_layers)])
    couplings = []
    if left_layers:
        couplings.extend(_copy(leads['left'].d01) for _ in range(left_layers - 1))
        couplings.append(_copy(junctions['left']))
    couplings.extend(_copy(block) for block in problem.device.coupling_blocks)
    if right_layers:
        couplings.append(_copy(junctions['right']))
        couplings.extend(_copy(leads['right'].d01) for _ in range(right_layers - 1))
    reservoirs = {}
    for side, count in (("left", left_layers), ("right", right_layers)):
        original = getattr(problem, side)
        if count:
            reservoirs[side] = HarmonicReservoir(leads[side], original.temperature)
        else:
            reservoirs[side] = HarmonicReservoir(leads[side], original.temperature,
                original.resolved_device_to_lead(side),
                original.resolved_device_indices(dim=problem.ndof, side=side) + offset)
    vertex = problem.vertex
    metadata = dict(vertex.metadata, lead_buffer_embedding={
        "original_ndof": problem.ndof, "coordinate_offset": offset,
        "interaction_support_extended": False})
    if isinstance(vertex, SparseCubicVertex):
        extended_vertex = SparseCubicVertex(ndof, vertex.indices + offset,
                                            vertex.coefficients, metadata=metadata)
    else:
        needed = 8 * ndof**3
        if needed > limit:
            raise MemoryError(f"Dense padded vertex needs {needed} bytes; limit is {limit}.")
        values = np.zeros((ndof, ndof, ndof))
        old = slice(offset, offset + problem.ndof)
        values[old, old, old] = vertex.values
        extended_vertex = DenseCubicVertex(values, metadata=metadata)
    extended = replace(problem, device=Device1D(onsite, couplings), vertex=extended_vertex,
                       left=reservoirs['left'], right=reservoirs['right'])
    return LeadBufferEmbedding(extended, problem.ndof, left_layers, right_layers)


__all__ = ["LeadBufferEmbedding", "add_harmonic_lead_buffers"]
