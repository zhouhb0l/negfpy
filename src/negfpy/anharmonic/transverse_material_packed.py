"""Audited material FC3 image blocks streamed directly into exact packed support.

The full harmonic device survives through explicit coordinate addresses. No full
D**3 array or scalar interaction schedule is built, and no support cutoff or
force-constant correction is applied. Coupling strength remains a caller choice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ._canonical import domain_digest
from .layered import LayeredDeviceMapping
from .mapped_fc3 import (
    MassNormalizedTranslationalFC3,
    _effective_terms,
    _layer_tuple,
    _normalized_payload,
    _positive_integer,
    _validate_mapping_identity,
)
from .momentum import UniformBZMesh
from .transverse_mapping import _kpoints
from .transverse_material import _geometry, _position
from .transverse_packed import PackedTransverseCubicVertex
from .transverse_q import difference_q_indices
from .types import SolverScale


@dataclass(frozen=True, slots=True)
class FourierPackedMaterialCubic:
    """Exact packed vertex before coupling, with immutable mapping metadata."""

    vertex: PackedTransverseCubicVertex
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def fourier_material_cubic_packed(
    mesh: UniformBZMesh,
    fc3: MassNormalizedTranslationalFC3,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: Sequence[int],
    scale: SolverScale,
    max_bytes: int = 64 * 1024**2,
    max_geometry_evaluations: int = 1_000_000,
    max_phase_evaluations: int = 1_000_000,
    max_workspace_bytes: int = 256 * 1024**2,
) -> FourierPackedMaterialCubic:
    """Map normalized physical-SI image blocks without a full-device vertex.

    The first geometry pass discovers the exact union of nonzero source
    coordinates surviving the all-leg crop and bounds phase work. The second
    accumulates directly into that candidate packed array. A bounded scan then
    drops coordinates cancelled to *exactly* zero on the sampled mesh. Tiny
    nonzero coefficients remain. Geometry, weights, r0 ordering and phases are
    identical to ``fourier_material_cubic``; only storage changes.

    ``max_bytes`` bounds the candidate coefficient/address payload (before
    possible cancellation). ``max_workspace_bytes`` bounds a conservative
    numeric-array plan for mapping and vertex construction, including coexisting
    candidate/compacted or owned copies and validation scratch. Caller inputs,
    Python objects, source identity hashing and geometry-audit internals are
    excluded; this is not an RSS limit. Geometry and source expansion have their
    independent work cap, and no scalar schedule is materialized.
    """
    if not isinstance(mesh, UniformBZMesh):
        raise TypeError("mesh must be a UniformBZMesh.")
    if mesh.shape[2] != 1:
        raise ValueError("The transverse mesh must have third dimension one.")
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("fc3 must be a MassNormalizedTranslationalFC3.")
    if not isinstance(device_mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    n_layers = _positive_integer(n_device_layers, name="n_device_layers")
    byte_limit = _positive_integer(max_bytes, name="max_bytes")
    workspace_limit = _positive_integer(max_workspace_bytes, name="max_workspace_bytes")
    geometry_limit = _positive_integer(
        max_geometry_evaluations, name="max_geometry_evaluations"
    )
    phase_limit = _positive_integer(max_phase_evaluations, name="max_phase_evaluations")
    interactions = _layer_tuple(
        interaction_layers, name="interaction_layers", n_device_layers=n_layers
    )
    nq = mesh.n_qpoints
    source_legs = 3 if fc3.make_r0_average else 1
    if len(fc3.terms) * source_legs > geometry_limit:
        raise ValueError("Source FC3 expansion exceeds max_geometry_evaluations.")
    # Reserve routing, small phase/image arrays and the newly owned second/third
    # source positions before expanding r0 legs. Values are views, not copies.
    auxiliary_bytes = 48 * len(fc3.terms) + 2048 * nq**2 + 128 * nq + 8192
    if auxiliary_bytes > workspace_limit:
        raise ValueError("Material packed mapping auxiliaries exceed max_workspace_bytes.")
    if 8 * nq**2 > byte_limit:
        raise ValueError("Momentum routing table exceeds max_bytes.")
    _geometry(device_mapping)
    _validate_mapping_identity(fc3, device_mapping)
    if fc3.output_units != "J m^-3 kg^-3/2":
        raise ValueError("FC3 must use mass-normalized physical SI units.")
    if (
        domain_digest("NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3", _normalized_payload(fc3))
        != fc3.content_sha256
    ):
        raise ValueError("Normalized FC3 content no longer matches its identity.")
    effective = _effective_terms(fc3)
    representatives = device_mapping.direct_coset_representatives
    nonzero_effective = sum(bool(np.any(term.values != 0)) for term in effective)
    geometry_evaluations = 2 * nonzero_effective * len(representatives)
    if geometry_evaluations > geometry_limit:
        raise ValueError("Two-pass FC3 coset mapping exceeds max_geometry_evaluations.")
    p = device_mapping.principal_cells
    n_basis = device_mapping.n_transport_basis_atoms
    layer_dofs = device_mapping.n_principal_layer_dofs
    ndof = n_layers * layer_dofs
    if ndof > np.iinfo(np.int64).max:
        raise ValueError("Full-device addresses exceed the supported int64 range.")
    minimum = interactions[0] * p
    maximum = (interactions[-1] + 1) * p - 1
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        solver_factor = float(np.sqrt(scale.hbar) / np.float64(scale.omega0)**2.5)
    if not np.isfinite(solver_factor) or solver_factor <= 0:
        raise ValueError("FC3 solver conversion is not representable.")

    def mapped_images():
        for term in effective:
            nonzero = term.values != 0
            components = int(np.count_nonzero(nonzero))
            if not components:
                continue
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                scaled = term.values * solver_factor
                weight = term.multiplicity_weight * term.anchor_weight * term.r0_leg_factor
                weighted = scaled * weight
            if not np.isfinite(scaled).all() or np.any(scaled[nonzero] == 0):
                raise ValueError("Nonzero FC3 coefficient over/underflows solver units.")
            if not np.isfinite(weighted).all() or np.any(weighted[nonzero] == 0):
                raise ValueError("Nonzero weighted FC3 coefficient over/underflows.")
            cartesian = [
                np.flatnonzero(np.any(nonzero, axis=tuple(j for j in range(3) if j != i)))
                for i in range(3)
            ]
            for representative in representatives:
                mapped = [
                    _position(atom, position + representative, device_mapping)
                    for atom, position in zip(term.primitive_indices, term.positions, strict=True)
                ]
                cells = [int(cell[0]) for cell, _ in mapped]
                low = max(minimum - cell for cell in cells)
                high = min(maximum - cell for cell in cells)
                yield mapped, low, high, components, cartesian, weighted

    def bases_at(mapped, anchor):
        bases = []
        for cell, basis in mapped:
            layer, subcell = divmod(int(cell[0]) + anchor, p)
            bases.append(layer * layer_dofs + 3 * (subcell * n_basis + basis))
        return bases

    support: set[int] = set()
    expanded_images = cropped_images = scalar_terms = 0
    for mapped, low, high, components, cartesian, _ in mapped_images():
        if low > high:
            cropped_images += 1
            continue
        count = high - low + 1
        expanded_images += count
        scalar_terms += count * components
        if nq**2 * scalar_terms > phase_limit:
            raise ValueError("Material Fourier sum exceeds max_phase_evaluations.")
        for anchor in range(low, high + 1):
            for base, components_for_leg in zip(bases_at(mapped, anchor), cartesian, strict=True):
                support.update(base + int(component) for component in components_for_leg)
        if 16 * nq**2 * len(support)**3 + 8 * len(support) > byte_limit:
            raise ValueError("Candidate packed coefficient/address payload exceeds max_bytes.")

    a = len(support)
    coefficient_bytes = 16 * nq**2 * a**3
    candidate_bytes = coefficient_bytes + 8 * a
    # At most two candidate-sized coefficient arrays coexist. Constructor finite
    # checking may use Q**2*A**3 booleans; allclose validation uses one A**3
    # conjugate and bounded arithmetic temporaries. 128*A**3 conservatively
    # covers those complex/real/bool blocks and compaction/support scan scratch.
    peak_numeric_bytes = (
        2 * coefficient_bytes + nq**2 * a**3 + 128 * a**3
        + a**2 + 64 * a + auxiliary_bytes
    )
    if candidate_bytes > byte_limit:
        raise ValueError("Candidate packed coefficient/address payload exceeds max_bytes.")
    if peak_numeric_bytes > workspace_limit:
        raise ValueError("Packed mapping/construction plan exceeds max_workspace_bytes.")
    active = np.array(sorted(support), dtype=np.int64)
    lookup = {int(address): i for i, address in enumerate(active)}
    del support
    kpoints = _kpoints(mesh)
    routes = difference_q_indices(mesh, max_bytes=byte_limit)
    qindices = np.arange(nq)
    values = np.zeros((nq, nq, a, a, a), dtype=np.complex128)
    for mapped, low, high, _, cartesian, weighted in mapped_images():
        if low > high:
            continue
        second = tuple(
            (int(b) - int(c)) % period
            for c, b, period in zip(mapped[0][0][1:], mapped[1][0][1:], mesh.shape[:2], strict=True)
        )
        third = tuple(
            (int(b) - int(c)) % period
            for c, b, period in zip(mapped[0][0][1:], mapped[2][0][1:], mesh.shape[:2], strict=True)
        )
        phase_p = np.exp(1j * (kpoints @ second))
        phase_partner = np.exp(1j * (kpoints @ third))
        block = (phase_p[None, :] * phase_partner[routes])[:, :, None, None, None]
        block = block * weighted[np.ix_(*cartesian)][None, None]
        for anchor in range(low, high + 1):
            coordinates = [
                np.array([lookup[base + int(c)] for c in components], dtype=np.int64)
                for base, components in zip(bases_at(mapped, anchor), cartesian, strict=True)
            ]
            selection = np.ix_(qindices, qindices, *coordinates)
            with np.errstate(over="ignore", invalid="ignore"):
                values[selection] += block

    # Exact support after all source-image interference/cancellation. Scan one
    # Cartesian matrix at a time; do not allocate a Q**2*A**3 support mask.
    retained = np.zeros(a, dtype=bool)
    for q in values:
        for block in q:
            for i, matrix in enumerate(block):
                if not np.isfinite(matrix).all():
                    raise ValueError("Material Fourier accumulation produced nonfinite values.")
                mask = matrix != 0
                if mask.any():
                    retained[i] = True
                    retained |= mask.any(axis=0)
                    retained |= mask.any(axis=1)
    kept = np.flatnonzero(retained)
    if len(kept) != a:
        compact = np.empty((nq, nq, len(kept), len(kept), len(kept)), dtype=np.complex128)
        for q in range(nq):
            for internal in range(nq):
                for i, old_i in enumerate(kept):
                    for j, old_j in enumerate(kept):
                        compact[q, internal, i, j] = values[q, internal, old_i, old_j, kept]
        values = compact
        active = active[kept]
    # Release remaining views before constructor detaches its owned copy.
    if a:
        del block, matrix, mask
    vertex = PackedTransverseCubicVertex(mesh, values, active, ndof, max_bytes=byte_limit)
    return FourierPackedMaterialCubic(vertex, {
        "source_normalized_fc3_sha256": fc3.content_sha256,
        "source_fc3_sha256": fc3.source_fc3_content_sha256,
        "source_fc2_sha256": device_mapping.fc2_content_sha256,
        "primitive_crystal_sha256": device_mapping.primitive_crystal_content_sha256,
        "layered_geometry_sha256": device_mapping.layered_geometry_sha256,
        "n_device_layers": n_layers,
        "interaction_layers": interactions,
        "principal_cells": p,
        "transport_basis_atoms": n_basis,
        "make_r0_average": fc3.make_r0_average,
        "source_image_terms": len(fc3.terms),
        "effective_image_terms": len(effective),
        "expanded_finite_images": expanded_images,
        "images_with_no_allowed_anchor": cropped_images,
        "output_scalar_terms": scalar_terms,
        "geometry_passes": 2,
        "geometry_evaluations": geometry_evaluations,
        "phase_evaluations": nq**2 * scalar_terms,
        "full_ndof": ndof,
        "candidate_active_dofs": a,
        "active_dofs": vertex.ndof,
        "exactly_cancelled_dofs": a - vertex.ndof,
        "candidate_vertex_bytes": candidate_bytes,
        "vertex_bytes": vertex.coefficients.nbytes + vertex.active_dofs.nbytes,
        "planned_peak_numeric_bytes": peak_numeric_bytes,
        "full_dense_vertex_allocated": False,
        "scalar_schedule_materialized": False,
        "support_cutoff": 0.0,
        "output_units": "solver",
        "weights_applied": True,
        "coupling_applied": False,
        "omega0": scale.omega0,
        "hbar": scale.hbar,
        "symmetrized": False,
        "boundary_corrected": False,
    })


__all__ = ["FourierPackedMaterialCubic", "fourier_material_cubic_packed"]
