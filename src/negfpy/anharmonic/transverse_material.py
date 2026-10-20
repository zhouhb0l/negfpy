"""Audited crystal IFC images to finite-device, transverse-periodic inputs.

Reuse the existing integer transport-cell geometry and audited raw FC3 image
expansion. Retain transverse cell winding instead of summing it at q=0.
No force-constant repair, interaction-boundary correction or sum rule is added.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from negfpy.core.types import DeviceKSpace, LeadKSpace
from negfpy.models.material_kspace import MaterialKspaceParams

from ._canonical import (
    crystal_mapping_content_sha256,
    domain_digest,
    fc2_content_sha256,
    primitive_crystal_content_sha256,
)
from .layered import (
    LayeredDeviceMapping,
    _decompose_transport,
    _find_basis_atom,
    _primitive_ordinal,
)
from .mapped_fc3 import (
    MassNormalizedTranslationalFC3,
    _effective_terms,
    _layer_tuple,
    _normalized_payload,
    _positive_integer,
    _validate_mapping_identity,
)
from .momentum import UniformBZMesh
from .transverse_mapping import (
    TransverseCubicTerm,
    _kpoints,
    material_harmonic_providers,
)
from .transverse_q import TransverseCubicVertex, difference_q_indices
from .types import SolverScale

_MAX_BYTES = 64 * 1024**2


def _geometry(mapping: LayeredDeviceMapping) -> None:
    if not isinstance(mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    if mapping.validation is None or not mapping.validation.certification_passed:
        raise ValueError("Layered geometry must carry a passing audit.")
    if (
        fc2_content_sha256(mapping._canonical_fc2.as_fc2())
        != mapping.fc2_content_sha256
    ):
        raise ValueError("Embedded FC2 content no longer matches the mapping.")
    if (
        primitive_crystal_content_sha256(mapping._crystal)
        != mapping.primitive_crystal_content_sha256
    ):
        raise ValueError("Embedded primitive identity no longer matches the mapping.")
    if (
        crystal_mapping_content_sha256(mapping._crystal)
        != mapping.fc2_crystal_mapping_content_sha256
    ):
        raise ValueError("Embedded FC2 crystal mapping content has changed.")


def _position(
    primitive_atom: int,
    position: np.ndarray,
    mapping: LayeredDeviceMapping,
) -> tuple[np.ndarray, int]:
    """Return exact transport cell and basis, including periodic match winding."""
    transport = position @ mapping.primitive_to_transport
    if not np.isfinite(transport).all() or np.max(np.abs(transport)) >= 2**50:
        raise ValueError("IFC image coordinates exceed the supported exact-cell range.")
    cell, fractional = _decompose_transport(transport)
    basis, winding, _, _ = _find_basis_atom(
        primitive_atom=primitive_atom,
        fractional=fractional,
        basis_primitive_atoms=mapping.basis_primitive_atoms,
        basis_scaled_positions=mapping.basis_scaled_positions,
        transport_lattice_angstrom=mapping.transport_lattice_angstrom,
    )
    return cell + winding, basis


@dataclass(frozen=True, slots=True)
class MappedTransverseCubic:
    """Ordered scalar terms in solver units; weights are still unapplied."""

    ndof: int
    terms: tuple[TransverseCubicTerm, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", tuple(self.terms))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class FourierMaterialCubic:
    """Complete reduced vertex from streamed source blocks, in solver units."""

    vertex: TransverseCubicVertex
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def fourier_material_cubic(
    mesh: UniformBZMesh,
    fc3: MassNormalizedTranslationalFC3,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: Sequence[int],
    scale: SolverScale,
    max_bytes: int = _MAX_BYTES,
    max_geometry_evaluations: int = 1_000_000,
    max_phase_evaluations: int = 1_000_000,
) -> FourierMaterialCubic:
    """Map complete FC3 image blocks directly into the reduced Fourier vertex.

    No scalar ``TransverseCubicTerm`` schedule is allocated. A count-only
    geometry pass checks the exact nonzero-scalar/momentum-pair work before
    allocating the dense vertex. A second pass accumulates each 3x3x3 block.
    ``max_geometry_evaluations`` counts both image/coset traversals; each
    traversal maps the three legs together. Source r0 expansion is also
    bounded before copying. ``max_bytes`` bounds one vertex payload, not RSS.

    The same source identities, all-leg crop, cell winding, mass/SI solver
    factor and image/anchor/r0 weights as ``map_material_cubic`` apply.
    Coefficients have no additional coupling-strength multiplier.
    """
    kpoints = _kpoints(mesh)
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("fc3 must be a MassNormalizedTranslationalFC3.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    n_layers = _positive_integer(n_device_layers, name="n_device_layers")
    byte_limit = _positive_integer(max_bytes, name="max_bytes")
    geometry_limit = _positive_integer(
        max_geometry_evaluations,
        name="max_geometry_evaluations",
    )
    phase_limit = _positive_integer(max_phase_evaluations, name="max_phase_evaluations")
    interactions = _layer_tuple(
        interaction_layers,
        name="interaction_layers",
        n_device_layers=n_layers,
    )
    if not isinstance(device_mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    ndof = n_layers * device_mapping.n_principal_layer_dofs
    nq = mesh.n_qpoints
    vertex_bytes = 16 * nq**2 * ndof**3
    if vertex_bytes > byte_limit:
        raise ValueError("Dense material Fourier vertex exceeds max_bytes.")
    source_legs = 3 if fc3.make_r0_average else 1
    if len(fc3.terms) * source_legs > geometry_limit:
        raise ValueError("Source FC3 expansion exceeds max_geometry_evaluations.")
    _geometry(device_mapping)
    _validate_mapping_identity(fc3, device_mapping)
    if fc3.output_units != "J m^-3 kg^-3/2":
        raise ValueError("FC3 must use mass-normalized physical SI units.")
    if (
        domain_digest(
            "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3",
            _normalized_payload(fc3),
        )
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
    minimum = interactions[0] * p
    maximum = (interactions[-1] + 1) * p - 1
    solver_factor = np.sqrt(scale.hbar) / scale.omega0**2.5
    if not np.isfinite(solver_factor) or solver_factor <= 0:
        raise ValueError("FC3 solver conversion is not representable.")

    def mapped_images():
        for term in effective:
            nonzero = term.values != 0
            components = int(np.count_nonzero(nonzero))
            if not components:
                continue
            scaled = term.values * solver_factor
            if not np.isfinite(scaled).all() or np.any(scaled[nonzero] == 0):
                raise ValueError(
                    "Nonzero FC3 coefficient over/underflows solver units."
                )
            weight = term.multiplicity_weight * term.anchor_weight * term.r0_leg_factor
            weighted = scaled * weight
            if not np.isfinite(weighted).all() or np.any(weighted[nonzero] == 0):
                raise ValueError("Nonzero weighted FC3 coefficient over/underflows.")
            for representative in representatives:
                mapped = [
                    _position(atom, position + representative, device_mapping)
                    for atom, position in zip(
                        term.primitive_indices, term.positions, strict=True
                    )
                ]
                cells = [int(cell[0]) for cell, _ in mapped]
                low = max(minimum - cell for cell in cells)
                high = min(maximum - cell for cell in cells)
                yield mapped, low, high, components, weighted

    expanded_images = cropped_images = scalar_terms = 0
    for _, low, high, components, _ in mapped_images():
        if low > high:
            cropped_images += 1
            continue
        count = high - low + 1
        expanded_images += count
        scalar_terms += count * components
        if nq**2 * scalar_terms > phase_limit:
            raise ValueError("Material Fourier sum exceeds max_phase_evaluations.")

    routes = difference_q_indices(mesh, max_bytes=byte_limit)
    values = np.zeros((nq, nq, ndof, ndof, ndof), dtype=np.complex128)
    for mapped, low, high, _, weighted in mapped_images():
        if low > high:
            continue
        # Fold integer cell differences before converting them to phases.
        second = tuple(
            (int(b) - int(a)) % period
            for a, b, period in zip(
                mapped[0][0][1:], mapped[1][0][1:], mesh.shape[:2], strict=True
            )
        )
        third = tuple(
            (int(b) - int(a)) % period
            for a, b, period in zip(
                mapped[0][0][1:], mapped[2][0][1:], mesh.shape[:2], strict=True
            )
        )
        phase_p = np.exp(1j * (kpoints @ second))
        phase_partner = np.exp(1j * (kpoints @ third))
        block = (phase_p[None, :] * phase_partner[routes])[
            :, :, None, None, None
        ] * weighted[None, None]
        for anchor in range(low, high + 1):
            bases = []
            for cell, basis in mapped:
                layer, subcell = divmod(int(cell[0]) + anchor, p)
                bases.append(layer * layer_dofs + 3 * (subcell * n_basis + basis))
            a, c, d = bases
            values[:, :, a : a + 3, c : c + 3, d : d + 3] += block
    vertex = TransverseCubicVertex(mesh, values, max_bytes=byte_limit)
    return FourierMaterialCubic(
        vertex,
        {
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
            "vertex_bytes": vertex_bytes,
            "output_units": "solver",
            "weights_applied": True,
            "scalar_schedule_materialized": False,
            "coupling_applied": False,
            "omega0": scale.omega0,
            "hbar": scale.hbar,
            "symmetrized": False,
            "boundary_corrected": False,
        },
    )


def map_material_cubic(
    fc3: MassNormalizedTranslationalFC3,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: Sequence[int],
    scale: SolverScale,
    max_terms: int = 100_000,
    max_geometry_evaluations: int = 1_000_000,
) -> MappedTransverseCubic:
    """Map audited raw FC3 images without losing transverse phases.

    Start with ``convert_fc3_to_translational`` and
    ``mass_normalize_translational_fc3`` from the raw full IFC source and its
    audited crystal mapping. Their output is mass-normalized physical SI;
    this adapter applies only ``sqrt(hbar)/omega0**2.5``. Each source image,
    anchor and r0 factor remains in the returned term's weight exactly once.

    All three legs must lie inside the contiguous interaction-layer crop.
    No periodic wrapping occurs along transport. Cubic permutations and
    joint reality are checked when these terms are Fourier transformed.
    """
    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("fc3 must be a MassNormalizedTranslationalFC3.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    n_layers = _positive_integer(n_device_layers, name="n_device_layers")
    limit = _positive_integer(max_terms, name="max_terms")
    work_limit = _positive_integer(
        max_geometry_evaluations, name="max_geometry_evaluations"
    )
    interactions = _layer_tuple(
        interaction_layers,
        name="interaction_layers",
        n_device_layers=n_layers,
    )
    # Bound expansion before copying the optional three r0 legs or hashing.
    legs = 3 if fc3.make_r0_average else 1
    if len(fc3.terms) * legs > work_limit:
        raise ValueError("Source FC3 expansion exceeds max_geometry_evaluations.")
    _geometry(device_mapping)
    _validate_mapping_identity(fc3, device_mapping)
    if fc3.output_units != "J m^-3 kg^-3/2":
        raise ValueError("FC3 must use mass-normalized physical SI units.")
    if (
        domain_digest(
            "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3",
            _normalized_payload(fc3),
        )
        != fc3.content_sha256
    ):
        raise ValueError("Normalized FC3 content no longer matches its identity.")
    effective = _effective_terms(fc3)
    representatives = device_mapping.direct_coset_representatives
    if len(effective) * len(representatives) > work_limit:
        raise ValueError("FC3 coset mapping exceeds max_geometry_evaluations.")
    p = device_mapping.principal_cells
    n_basis = device_mapping.n_transport_basis_atoms
    layer_dofs = device_mapping.n_principal_layer_dofs
    minimum = interactions[0] * p
    maximum = (interactions[-1] + 1) * p - 1
    solver_factor = np.sqrt(scale.hbar) / scale.omega0**2.5
    if not np.isfinite(solver_factor) or solver_factor <= 0:
        raise ValueError("FC3 solver conversion is not representable.")
    output: list[TransverseCubicTerm] = []
    expanded_images = 0
    cropped_images = 0
    for term in effective:
        component_indices = np.argwhere(term.values != 0)
        if not len(component_indices):
            continue
        weight = term.multiplicity_weight * term.anchor_weight * term.r0_leg_factor
        for representative in representatives:
            mapped = [
                _position(atom, position + representative, device_mapping)
                for atom, position in zip(
                    term.primitive_indices, term.positions, strict=True
                )
            ]
            cells = [int(cell[0]) for cell, _ in mapped]
            low = max(minimum - cell for cell in cells)
            high = min(maximum - cell for cell in cells)
            if low > high:
                cropped_images += 1
                continue
            count = high - low + 1
            if len(output) + count * len(component_indices) > limit:
                raise ValueError("Finite-device FC3 schedule exceeds max_terms.")
            second = tuple(int(v) for v in mapped[1][0][1:] - mapped[0][0][1:])
            third = tuple(int(v) for v in mapped[2][0][1:] - mapped[0][0][1:])
            expanded_images += count
            for anchor in range(low, high + 1):
                bases = []
                for cell, basis in mapped:
                    longitudinal = int(cell[0]) + anchor
                    layer, subcell = divmod(longitudinal, p)
                    bases.append(layer * layer_dofs + 3 * (subcell * n_basis + basis))
                for cartesian in component_indices:
                    raw = float(term.values[tuple(cartesian)])
                    coefficient = raw * solver_factor
                    if not np.isfinite(coefficient) or coefficient == 0:
                        raise ValueError(
                            "Nonzero FC3 coefficient over/underflows solver units."
                        )
                    output.append(
                        TransverseCubicTerm(
                            tuple(
                                base + int(c)
                                for base, c in zip(bases, cartesian, strict=True)
                            ),
                            second,
                            third,
                            coefficient,
                            weight,
                        )
                    )
    return MappedTransverseCubic(
        n_layers * layer_dofs,
        tuple(output),
        {
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
            "output_scalar_terms": len(output),
            "output_units": "solver",
            "weights_applied": False,
            "omega0": scale.omega0,
            "hbar": scale.hbar,
            "symmetrized": False,
            "boundary_corrected": False,
        },
    )


def material_transverse_harmonic(
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    scale: SolverScale,
    max_bytes: int = _MAX_BYTES,
    max_image_evaluations: int = 1_000_000,
) -> tuple[DeviceKSpace, LeadKSpace]:
    """Rebuild FC2 transverse image blocks before the old q=0 summation.

    Uses the audited source values, source masses, shortest-vector multiplicity,
    and the same integer transport geometry as ``map_material_cubic``. All
    device layers have the source principal-layer composition. Reservoir
    temperatures, broadening and acoustic zero limits are caller decisions.
    """
    n_layers = _positive_integer(n_device_layers, name="n_device_layers")
    limit = _positive_integer(max_bytes, name="max_bytes")
    work_limit = _positive_integer(max_image_evaluations, name="max_image_evaluations")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    _geometry(device_mapping)
    mapping = device_mapping
    crystal, canonical = mapping._crystal, mapping._canonical_fc2
    p = mapping.principal_cells
    n_basis = mapping.n_transport_basis_atoms
    dimension = mapping.n_principal_layer_dofs
    block_bytes = 8 * dimension**2
    if 3 * block_bytes > limit:
        raise ValueError("Harmonic image blocks exceed max_bytes.")
    image_count = p * sum(
        int(np.sum(crystal.multiplicity[:, int(atom), 0]))
        for atom in mapping.basis_primitive_atoms
    )
    if image_count > work_limit:
        raise ValueError("FC2 image mapping exceeds max_image_evaluations.")
    blocks: dict[int, dict[tuple[int, int], np.ndarray]] = {
        layer: {(0, 0): np.zeros((dimension, dimension))} for layer in (-1, 0, 1)
    }
    stored_blocks = 3
    for row_basis, raw_atom in enumerate(mapping.basis_primitive_atoms):
        atom = int(raw_atom)
        source_first = atom if canonical.compact else int(crystal.p2s[atom])
        origin = (
            crystal.primitive.scaled_positions[atom]
            + mapping.basis_integer_cells[row_basis]
        )
        origin_cell, found_basis = _position(atom, origin, mapping)
        if found_basis != row_basis or np.any(origin_cell):
            raise ValueError("Harmonic transport basis does not anchor in cell zero.")
        for source_second in range(crystal.n_supercell_atoms):
            other = _primitive_ordinal(crystal, source_second)
            count, start = (int(v) for v in crystal.multiplicity[source_second, atom])
            values = canonical.values[source_first, source_second] / count
            if not np.any(values != 0):
                continue
            for image in crystal.shortest_vectors[start : start + count]:
                destination_cell, column_basis = _position(
                    other, origin + image, mapping
                )
                transverse = tuple(int(v) for v in destination_cell[1:])
                for subcell in range(p):
                    layer, destination_subcell = divmod(
                        subcell + int(destination_cell[0]), p
                    )
                    if layer not in blocks:
                        raise ValueError(
                            "FC2 support exceeds the declared principal layer."
                        )
                    if transverse not in blocks[layer]:
                        stored_blocks += 1
                        if stored_blocks * block_bytes > limit:
                            raise ValueError("Harmonic image blocks exceed max_bytes.")
                        blocks[layer][transverse] = np.zeros((dimension, dimension))
                    row = 3 * (subcell * n_basis + row_basis)
                    column = 3 * (destination_subcell * n_basis + column_basis)
                    blocks[layer][transverse][row : row + 3, column : column + 3] += (
                        values
                    )
    masses = np.tile(crystal.primitive.masses[mapping.basis_primitive_atoms], p)
    params = MaterialKspaceParams(masses, blocks[0], blocks[1], blocks[-1])
    return material_harmonic_providers(
        params,
        n_layers=n_layers,
        force_constant_units=canonical.units,
        mass_units=canonical.mass_units,
        scale=scale,
    )


__all__ = [
    "FourierMaterialCubic",
    "MappedTransverseCubic",
    "fourier_material_cubic",
    "map_material_cubic",
    "material_transverse_harmonic",
]
