"""Exact layered harmonic construction for the material ``N_q=1`` path.

This module is deliberately limited to Phase A of the sparse material
development contract.  It maps an audited phono3py-style :class:`FC2Data`
and its :class:`CrystalMapping` onto an exactly block-tridiagonal harmonic
operator.  It does not read or map FC3 and it does not run an anharmonic
calculation.

Lattice vectors and fractional coordinates are row vectors throughout.  The
transport transform therefore obeys ``L_t = S @ L_p`` and
``x_t = x_p @ inv(S)``.  All shortest-vector images are retained with their
stored inverse multiplicity; transverse translations are reduced only after
the exact image has been assigned to the transport basis.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks

from ._canonical import (
    crystal_mapping_content_sha256,
    digest_bytes,
    domain_digest,
    enumerate_direct_reciprocal_cosets,
    fc2_content_sha256,
    fc2_provenance_sha256,
    primitive_crystal_content_sha256,
    validate_sha256,
)
from .crystal import CrystalMapping, validate_crystal_mapping
from .fc2 import (
    ANGSTROM_M,
    ATOMIC_MASS_UNIT_KG,
    ELECTRON_VOLT_J,
    FC2Data,
    fc2_force_constant_to_si_factor,
    mass_to_si_factor,
    validate_fc2,
)
from .types import SolverScale


Array = np.ndarray

_INTEGER_MATRIX_ATOL = 1.0e-12
_FRACTIONAL_SNAP_ATOL = 1.0e-10
_FRACTIONAL_MATCH_ATOL = 5.0e-10
_CARTESIAN_MATCH_ATOL_M = 1.0e-12
_MASS_MATCH_RTOL = 1.0e-12
_MASS_MATCH_ATOL_KG = 0.0
_DIRECT_RECIPROCAL_PHASE_ATOL = 1.0e-13
_OPERATOR_RTOL = 1.0e-10
_OPERATOR_ATOL = 1.0e-12

_GEOMETRY_TOLERANCES = MappingProxyType(
    {
        "integer_matrix_atol": _INTEGER_MATRIX_ATOL,
        "fractional_snap_atol": _FRACTIONAL_SNAP_ATOL,
        "fractional_match_atol": _FRACTIONAL_MATCH_ATOL,
        "cartesian_match_atol_m": _CARTESIAN_MATCH_ATOL_M,
        "mass_match_rtol": _MASS_MATCH_RTOL,
        "mass_match_atol_kg": _MASS_MATCH_ATOL_KG,
        "direct_reciprocal_phase_atol": _DIRECT_RECIPROCAL_PHASE_ATOL,
    }
)


def _readonly(values: Any, *, dtype: Any, name: str) -> Array:
    raw = np.asarray(values)
    if dtype in (np.float64, np.complex128) and raw.dtype.kind == "O":
        raise TypeError(f"{name} must be a numerical array.")
    array = np.array(values, dtype=dtype, order="C", copy=True)
    if dtype in (np.float64, np.complex128) and not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    if dtype in (np.float64, np.complex128):
        # Canonical physics arrays never retain negative zero.
        if np.iscomplexobj(array):
            real = np.where(array.real == 0.0, 0.0, array.real)
            imag = np.where(array.imag == 0.0, 0.0, array.imag)
            array = np.array(real + 1j * imag, dtype=dtype, order="C")
        else:
            array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _readonly_int(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain exact integers.")
    return _readonly(raw, dtype=np.int64, name=name)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, order="C", copy=True)
        array.setflags(write=False)
        return array
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )


def _integer_transform(values: Any) -> Array:
    raw = np.asarray(values)
    if raw.shape != (3, 3):
        raise ValueError("transport_supercell_matrix must have shape (3, 3).")
    if raw.dtype.kind == "c":
        raise TypeError("transport_supercell_matrix must be real-valued.")
    try:
        numerical = np.array(raw, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "transport_supercell_matrix must be a numerical matrix."
        ) from exc
    if not np.all(np.isfinite(numerical)):
        raise ValueError("transport_supercell_matrix must be finite.")
    rounded = np.rint(numerical)
    if np.max(np.abs(numerical - rounded)) > _INTEGER_MATRIX_ATOL:
        raise ValueError(
            "transport_supercell_matrix is not integer-valued within the "
            "frozen absolute tolerance 1e-12."
        )
    matrix = np.array(rounded, dtype=np.int64, order="C")
    determinant = _determinant_int3(matrix)
    if determinant == 0:
        raise ValueError("transport_supercell_matrix must be nonsingular.")
    matrix.setflags(write=False)
    return matrix


def _determinant_int3(matrix: Array) -> int:
    a = [[int(value) for value in row] for row in np.asarray(matrix)]
    return (
        a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
    )


def _snap_fractional(values: Any) -> Array:
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    nearest = np.rint(result)
    mask = np.abs(result - nearest) <= _FRACTIONAL_SNAP_ATOL
    result[mask] = nearest[mask]
    result[result == 0.0] = 0.0
    return result


def _decompose_transport(values: Any) -> tuple[Array, Array]:
    snapped = _snap_fractional(values)
    cell = np.floor(snapped).astype(np.int64)
    fractional = snapped - cell
    near_one = np.abs(fractional - 1.0) <= _FRACTIONAL_SNAP_ATOL
    if np.any(near_one):
        cell[near_one] += 1
        fractional[near_one] = 0.0
    fractional[np.abs(fractional) <= _FRACTIONAL_SNAP_ATOL] = 0.0
    return cell, fractional


def _periodic_residual(left: Array, right: Array) -> Array:
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    delta -= np.rint(delta)
    delta[delta == 0.0] = 0.0
    return delta


def _primitive_ordinal(mapping: CrystalMapping, supercell_atom: int) -> int:
    representative = int(mapping.s2p[supercell_atom])
    matches = np.flatnonzero(mapping.p2s == representative)
    if matches.size != 1:
        raise ValueError(
            "Each supercell atom must resolve to exactly one primitive "
            "representative."
        )
    return int(matches[0])


@dataclass(frozen=True, slots=True, eq=False)
class _CanonicalFC2Input:
    values: Array
    masses: Array
    compact: bool
    p2s_map: Array | None
    s2p_map: Array | None
    units: str
    mass_units: str
    version: str | None
    cutoff: float | None
    provenance: Mapping[str, Any]

    @classmethod
    def from_fc2(cls, fc2: FC2Data) -> "_CanonicalFC2Input":
        values = _readonly(fc2.values, dtype=np.float64, name="FC2 values")
        masses = _readonly(fc2.masses, dtype=np.float64, name="FC2 masses")
        if not np.array_equal(values, np.asarray(fc2.values)):
            raise ValueError("FC2 float64 canonicalization changed a value.")
        if not np.array_equal(masses, np.asarray(fc2.masses)):
            raise ValueError("FC2 mass float64 canonicalization changed a value.")
        p2s = (
            None
            if fc2.p2s_map is None
            else _readonly_int(fc2.p2s_map, name="FC2 p2s_map")
        )
        s2p = (
            None
            if fc2.s2p_map is None
            else _readonly_int(fc2.s2p_map, name="FC2 s2p_map")
        )
        return cls(
            values=values,
            masses=masses,
            compact=bool(fc2.compact),
            p2s_map=p2s,
            s2p_map=s2p,
            units=str(fc2.units),
            mass_units=str(fc2.mass_units),
            version=fc2.version,
            cutoff=fc2.cutoff,
            provenance=_freeze_mapping(fc2.provenance),
        )

    def as_fc2(self) -> FC2Data:
        return FC2Data(
            values=self.values,
            masses=self.masses,
            compact=self.compact,
            p2s_map=self.p2s_map,
            s2p_map=self.s2p_map,
            units=self.units,
            mass_units=self.mass_units,
            version=self.version,
            cutoff=self.cutoff,
            provenance=self.provenance,
        )


@dataclass(frozen=True, slots=True, eq=False)
class _TranslationalFC2Term:
    row_basis: int
    column_basis: int
    longitudinal_shift: int
    primitive_i: int
    primitive_j: int
    source_first: int
    source_second: int
    image_index: int
    multiplicity: int
    values: Array

    def __post_init__(self) -> None:
        for name in (
            "row_basis",
            "column_basis",
            "primitive_i",
            "primitive_j",
            "source_first",
            "source_second",
            "image_index",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 0:
                raise ValueError(f"{name} must be nonnegative.")
            object.__setattr__(self, name, int(value))
        object.__setattr__(self, "longitudinal_shift", int(self.longitudinal_shift))
        if int(self.multiplicity) < 1:
            raise ValueError("multiplicity must be positive.")
        object.__setattr__(self, "multiplicity", int(self.multiplicity))
        values = _readonly(self.values, dtype=np.float64, name="FC2 term values")
        if values.shape != (3, 3):
            raise ValueError("FC2 term values must have shape (3, 3).")
        object.__setattr__(self, "values", values)


def _translation_to_representative(
    mapping: CrystalMapping,
    *,
    source_first: int,
    primitive_anchor: int,
) -> Array:
    translations = mapping.translation_permutations
    if translations is None:
        raise NotImplementedError(
            "Full FC2 redundant-row certification requires explicit "
            "CrystalMapping.translation_permutations."
        )
    representative = int(mapping.p2s[primitive_anchor])
    candidates: dict[tuple[int, ...], Array] = {}
    for row in translations:
        if int(row[source_first]) == representative:
            candidates[tuple(int(value) for value in row)] = row
    if len(candidates) != 1:
        raise ValueError(
            "Exactly one pure translation must map a full FC2 first-axis "
            f"atom to representative {representative}; found {len(candidates)}."
        )
    return next(iter(candidates.values()))


def _audit_full_redundant_rows(
    canonical: _CanonicalFC2Input,
    crystal: CrystalMapping,
) -> Mapping[str, Any]:
    if canonical.compact:
        return MappingProxyType(
            {
                "passed": True,
                "checked_first_axis_rows": 0,
                "maximum_absolute_residual": 0.0,
            }
        )
    representatives = set(int(value) for value in crystal.p2s)
    redundant = [
        source
        for source in range(crystal.n_supercell_atoms)
        if source not in representatives
    ]
    maximum = 0.0
    for source_first in redundant:
        primitive_anchor = _primitive_ordinal(crystal, source_first)
        representative = int(crystal.p2s[primitive_anchor])
        permutation = _translation_to_representative(
            crystal,
            source_first=source_first,
            primitive_anchor=primitive_anchor,
        )
        expected = canonical.values[representative][permutation]
        actual = canonical.values[source_first]
        residual = float(np.max(np.abs(actual - expected)))
        scale = max(
            float(np.max(np.abs(actual))),
            float(np.max(np.abs(expected))),
            np.finfo(np.float64).tiny,
        )
        maximum = max(maximum, residual)
        if residual > _OPERATOR_ATOL + _OPERATOR_RTOL * scale:
            raise ValueError(
                "Full FC2 redundant translation audit failed for first-axis "
                f"atom {source_first}: residual {residual:.6e}."
            )
    return MappingProxyType(
        {
            "passed": True,
            "checked_first_axis_rows": len(redundant),
            "maximum_absolute_residual": maximum,
        }
    )


def _source_identity(
    canonical: _CanonicalFC2Input,
    content_sha256: str,
) -> tuple[str, str]:
    provenance = canonical.provenance
    raw_source = provenance.get("source_sha256")
    duplicate = provenance.get("sha256")
    if raw_source is not None:
        source = validate_sha256(raw_source, name="FC2 source_sha256")
        if duplicate is not None and validate_sha256(
            duplicate, name="FC2 provenance sha256"
        ) != source:
            raise ValueError("FC2 provenance source_sha256 and sha256 differ.")
        provenance_hash = fc2_provenance_sha256(canonical.as_fc2())
        return source, provenance_hash

    # Synthetic fixtures have no source file.  They remain deterministic but
    # are not mistaken for sealed external evidence.
    source = domain_digest(
        "NEGFPY-SYNTHETIC-FC2-SOURCE",
        {"fc2_content_sha256": digest_bytes(content_sha256)},
    )
    provenance_hash = domain_digest(
        "NEGFPY-FC2-PROVENANCE",
        {
            "schema_version": "negfpy.fc2-provenance.synthetic.v1",
            "fc2_content_sha256": digest_bytes(content_sha256),
        },
    )
    return source, provenance_hash


def _find_basis_atom(
    *,
    primitive_atom: int,
    fractional: Array,
    basis_primitive_atoms: Array,
    basis_scaled_positions: Array,
    transport_lattice_angstrom: Array,
) -> tuple[int, Array, float, float]:
    candidates: list[tuple[int, Array, float, float]] = []
    for basis in np.flatnonzero(basis_primitive_atoms == primitive_atom):
        raw_delta = np.asarray(fractional) - basis_scaled_positions[int(basis)]
        winding = np.rint(raw_delta).astype(np.int64)
        residual = raw_delta - winding
        fractional_max = float(np.max(np.abs(residual)))
        if fractional_max > _FRACTIONAL_MATCH_ATOL:
            continue
        cartesian_m = residual @ transport_lattice_angstrom * ANGSTROM_M
        cartesian_max = float(np.max(np.abs(cartesian_m)))
        if cartesian_max > _CARTESIAN_MATCH_ATOL_M:
            continue
        candidates.append(
            (int(basis), winding, fractional_max, cartesian_max)
        )
    if len(candidates) != 1:
        raise ValueError(
            "A projected FC2 image must match exactly one transport-basis "
            f"atom; found {len(candidates)} for primitive atom {primitive_atom}."
        )
    return candidates[0]


def _project_terms(
    canonical: _CanonicalFC2Input,
    crystal: CrystalMapping,
    *,
    primitive_to_transport: Array,
    transport_lattice_angstrom: Array,
    basis_primitive_atoms: Array,
    basis_integer_cells: Array,
    basis_scaled_positions: Array,
) -> tuple[
    tuple[_TranslationalFC2Term, ...],
    Array,
    Array,
    int,
    float,
    float,
]:
    terms: list[_TranslationalFC2Term] = []
    n_images = 0
    fractional_match_max = 0.0
    cartesian_match_max_m = 0.0
    primitive_positions = crystal.primitive.scaled_positions
    for row_basis, primitive_i_raw in enumerate(basis_primitive_atoms):
        primitive_i = int(primitive_i_raw)
        source_first = (
            primitive_i
            if canonical.compact
            else int(crystal.p2s[primitive_i])
        )
        anchor_primitive = (
            primitive_positions[primitive_i] + basis_integer_cells[row_basis]
        )
        for source_second in range(crystal.n_supercell_atoms):
            primitive_j = _primitive_ordinal(crystal, source_second)
            count = int(crystal.multiplicity[source_second, primitive_i, 0])
            offset = int(crystal.multiplicity[source_second, primitive_i, 1])
            if count < 1:
                raise ValueError("Every FC2 image multiplicity must be positive.")
            block = canonical.values[source_first, source_second] / float(count)
            for image_index in range(offset, offset + count):
                vector = crystal.shortest_vectors[image_index]
                destination_transport = (
                    anchor_primitive + vector
                ) @ primitive_to_transport
                destination_cell, destination_fractional = _decompose_transport(
                    destination_transport
                )
                (
                    column_basis,
                    cell_winding,
                    fractional_residual,
                    cartesian_residual_m,
                ) = _find_basis_atom(
                    primitive_atom=primitive_j,
                    fractional=destination_fractional,
                    basis_primitive_atoms=basis_primitive_atoms,
                    basis_scaled_positions=basis_scaled_positions,
                    transport_lattice_angstrom=transport_lattice_angstrom,
                )
                # Matching is periodic.  Preserve the corresponding winding
                # in the transport-cell index so a coordinate just below one
                # is never silently assigned to the wrong longitudinal layer.
                destination_cell = destination_cell + cell_winding
                fractional_match_max = max(
                    fractional_match_max,
                    fractional_residual,
                )
                cartesian_match_max_m = max(
                    cartesian_match_max_m,
                    cartesian_residual_m,
                )
                terms.append(
                    _TranslationalFC2Term(
                        row_basis=row_basis,
                        column_basis=column_basis,
                        longitudinal_shift=int(destination_cell[0]),
                        primitive_i=primitive_i,
                        primitive_j=primitive_j,
                        source_first=source_first,
                        source_second=source_second,
                        image_index=image_index,
                        multiplicity=count,
                        values=block,
                    )
                )
                n_images += 1

    terms.sort(
        key=lambda term: (
            term.row_basis,
            term.column_basis,
            term.longitudinal_shift,
            term.source_first,
            term.source_second,
            term.image_index,
        )
    )

    n_basis = int(basis_primitive_atoms.size)
    aggregated: dict[tuple[int, int, int], Array] = {}
    for term in terms:
        key = (term.row_basis, term.column_basis, term.longitudinal_shift)
        if key not in aggregated:
            aggregated[key] = np.zeros((3, 3), dtype=np.float64)
        aggregated[key] += term.values
    envelope: list[tuple[int, int, int]] = []
    numerical_support: list[tuple[int, int, int]] = []
    for (row_basis, column_basis, shift), block in sorted(aggregated.items()):
        for alpha in range(3):
            for beta in range(3):
                record = (
                    3 * row_basis + alpha,
                    3 * column_basis + beta,
                    shift,
                )
                envelope.append(record)
                if block[alpha, beta] != 0.0:
                    numerical_support.append(record)
    envelope_array = np.array(envelope, dtype=np.int64).reshape((-1, 3))
    numerical_support_array = np.array(
        numerical_support,
        dtype=np.int64,
    ).reshape((-1, 3))
    envelope_array.setflags(write=False)
    numerical_support_array.setflags(write=False)
    return (
        tuple(terms),
        envelope_array,
        numerical_support_array,
        n_images,
        fractional_match_max,
        cartesian_match_max_m,
    )


def _infer_principal_cells(terms: tuple[_TranslationalFC2Term, ...]) -> int:
    maximum = max(
        (abs(term.longitudinal_shift) for term in terms),
        default=0,
    )
    for candidate in range(1, max(1, maximum) + 1):
        valid = True
        for subcell in range(candidate):
            for term in terms:
                layer = (subcell + term.longitudinal_shift) // candidate
                if layer not in (-1, 0, 1):
                    valid = False
                    break
            if not valid:
                break
        if valid:
            return candidate
    raise ValueError(
        "No finite principal layer represents the complete FC2 support as "
        "nearest-neighbour blocks."
    )


def _principal_candidate_valid(
    terms: tuple[_TranslationalFC2Term, ...],
    candidate: int,
) -> bool:
    if candidate < 1:
        return False
    for subcell in range(candidate):
        for term in terms:
            if (subcell + term.longitudinal_shift) // candidate not in (-1, 0, 1):
                return False
    return True


def _coordinate_roundtrip_residual(
    crystal: CrystalMapping,
    transform: Array,
    basis_primitive_atoms: Array,
    basis_integer_cells: Array,
    basis_scaled_positions: Array,
) -> float:
    maximum = 0.0
    for basis, primitive_atom_raw in enumerate(basis_primitive_atoms):
        primitive_atom = int(primitive_atom_raw)
        expected = (
            crystal.primitive.scaled_positions[primitive_atom]
            + basis_integer_cells[basis]
        )
        reconstructed = basis_scaled_positions[basis] @ transform
        maximum = max(maximum, float(np.max(np.abs(expected - reconstructed))))
    return maximum


def _direct_reciprocal_phase_residual(
    transform: Array,
    direct_representatives: Array,
    reciprocal_representatives: Array,
) -> float:
    inverse = np.linalg.inv(np.asarray(transform, dtype=np.float64))
    inverse_transpose = inverse.T
    maximum = 0.0
    for direct in direct_representatives:
        transport_coordinate = np.asarray(direct, dtype=np.float64) @ inverse
        for reciprocal in reciprocal_representatives:
            primitive_q = (
                np.asarray(reciprocal, dtype=np.float64) @ inverse_transpose
            )
            primitive_phase = np.exp(
                2j * np.pi * float(np.dot(direct, primitive_q))
            )
            transport_phase = np.exp(
                2j * np.pi * float(np.dot(transport_coordinate, reciprocal))
            )
            maximum = max(maximum, float(abs(primitive_phase - transport_phase)))
    return maximum


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LayeredDeviceMapping:
    """Immutable shared transport geometry derived from one FC2 input."""

    transport_supercell_matrix: Array
    signed_determinant: int
    transport_lattice_angstrom: Array
    primitive_to_transport: Array
    transport_to_primitive: Array
    direct_coset_keys: Array
    direct_coset_representatives: Array
    reciprocal_coset_keys: Array
    reciprocal_coset_representatives: Array
    direct_cosets_sha256: str
    reciprocal_cosets_sha256: str
    basis_coset_ordinals: Array
    basis_primitive_atoms: Array
    basis_integer_cells: Array
    basis_scaled_positions: Array
    basis_cartesian_positions_m: Array
    principal_cells: int
    principal_layer_dof_records: Array
    transverse_repeats: tuple[int, int]
    longitudinal_fc2_support: Array
    longitudinal_fc2_exact_numerical_support: Array
    longitudinal_fc2_exact_numerical_support_sha256: str
    layer_offsets: Array
    fc2_content_sha256: str
    fc2_layout: Literal["full", "compact"]
    fc2_source_sha256: str
    fc2_provenance_sha256: str
    oracle_v0_2_fc2_mapping_sha256: str | None
    primitive_crystal_content_sha256: str
    fc2_crystal_mapping_content_sha256: str
    layered_geometry_sha256: str
    validation: Any
    _canonical_fc2: _CanonicalFC2Input
    _crystal: CrystalMapping
    _terms: tuple[_TranslationalFC2Term, ...]
    _geometry_metrics: Mapping[str, Any]

    def __init__(self) -> None:
        raise TypeError(
            "LayeredDeviceMapping is builder-created; use "
            "build_layered_device_mapping(...)."
        )

    @property
    def n_transport_basis_atoms(self) -> int:
        return int(self.basis_primitive_atoms.size)

    @property
    def n_principal_layer_dofs(self) -> int:
        return int(self.principal_layer_dof_records.shape[0])


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LayeredHarmonicModel:
    """Immutable solver-unit harmonic principal-layer model."""

    mapping: LayeredDeviceMapping
    scale: SolverScale
    d00: Array
    d01: Array
    d10: Array
    principal_layer_atom_masses_kg: Array
    rigid_translation_vectors: Array
    fc2_content_sha256: str
    fc2_layout: Literal["full", "compact"]
    fc2_source_sha256: str
    fc2_provenance_sha256: str
    oracle_v0_2_fc2_mapping_sha256: str | None
    primitive_crystal_content_sha256: str
    fc2_crystal_mapping_content_sha256: str
    layered_geometry_sha256: str
    model_physics_sha256: str
    validation: Any
    lead: LeadBlocks
    _d00_audit: Array
    _d01_audit: Array
    _d10_audit: Array
    _source_blocks_by_offset: Mapping[int, Array]
    _audit_metrics: Mapping[str, Any]

    def __init__(self) -> None:
        raise TypeError(
            "LayeredHarmonicModel is builder-created; use "
            "build_layered_harmonic_model(...)."
        )

    def device(self, n_layers: int) -> Device1D:
        if isinstance(n_layers, (bool, np.bool_)) or not isinstance(
            n_layers, (int, np.integer)
        ):
            raise TypeError("n_layers must be a positive integer.")
        count = int(n_layers)
        if count < 1:
            raise ValueError("n_layers must be a positive integer.")
        return Device1D(
            onsite_blocks=[np.array(self.d00, copy=True) for _ in range(count)],
            coupling_blocks=[
                np.array(self.d01, copy=True) for _ in range(count - 1)
            ],
        )


def _mapping_payload(
    *,
    primitive_sha256: str,
    fc2_mapping_sha256: str,
    transport_supercell_matrix: Array,
    signed_determinant: int,
    transport_lattice_angstrom: Array,
    primitive_to_transport: Array,
    transport_to_primitive: Array,
    direct: Any,
    reciprocal: Any,
    basis_coset_ordinals: Array,
    basis_primitive_atoms: Array,
    basis_integer_cells: Array,
    basis_scaled_positions: Array,
    basis_cartesian_positions_m: Array,
    principal_cells: int,
    principal_layer_dof_records: Array,
    support: Array,
    layer_offsets: Array,
) -> Mapping[str, Any]:
    return {
        "schema_version": "negfpy.layered-geometry.v1",
        "primitive_crystal_content_sha256": digest_bytes(primitive_sha256),
        "fc2_crystal_mapping_content_sha256": digest_bytes(fc2_mapping_sha256),
        "transport_supercell_matrix": transport_supercell_matrix,
        "signed_determinant": signed_determinant,
        "transport_lattice_angstrom": transport_lattice_angstrom,
        "primitive_to_transport": primitive_to_transport,
        "transport_to_primitive": transport_to_primitive,
        "direct_coset_keys": direct.keys,
        "direct_coset_representatives": direct.representatives,
        "direct_cosets_sha256": digest_bytes(direct.sha256),
        "reciprocal_coset_keys": reciprocal.keys,
        "reciprocal_coset_representatives": reciprocal.representatives,
        "reciprocal_cosets_sha256": digest_bytes(reciprocal.sha256),
        "basis_coset_ordinals": basis_coset_ordinals,
        "basis_primitive_atoms": basis_primitive_atoms,
        "basis_integer_cells": basis_integer_cells,
        "basis_scaled_positions": basis_scaled_positions,
        "basis_cartesian_positions_m": basis_cartesian_positions_m,
        "principal_cells": principal_cells,
        "principal_layer_dof_records": principal_layer_dof_records,
        "transverse_repeats": (1, 1),
        "longitudinal_fc2_support": support,
        "layer_offsets": layer_offsets,
        "tolerances": dict(_GEOMETRY_TOLERANCES),
    }


def build_layered_device_mapping(
    crystal: CrystalMapping,
    fc2: FC2Data,
    *,
    transport_supercell_matrix: Any,
) -> LayeredDeviceMapping:
    """Build one exact immutable layered geometry from FC2 and its mapping."""

    if not isinstance(crystal, CrystalMapping):
        raise TypeError("crystal must be a CrystalMapping instance.")
    if not isinstance(fc2, FC2Data):
        raise TypeError("fc2 must be an FC2Data instance.")
    fc2_report = validate_fc2(fc2)
    if not fc2_report.passed:
        raise ValueError("FC2 validation failed: " + "; ".join(fc2_report.errors))
    crystal_report = validate_crystal_mapping(crystal)
    if not crystal_report.passed:
        raise ValueError(
            "CrystalMapping validation failed: " + "; ".join(crystal_report.errors)
        )
    if crystal.purpose != "fc2":
        raise ValueError("CrystalMapping purpose must be 'fc2'.")
    if fc2.n_supercell_atoms != crystal.n_supercell_atoms:
        raise ValueError("FC2 and CrystalMapping supercell sizes differ.")
    if not np.array_equal(np.asarray(fc2.masses), crystal.supercell.masses):
        raise ValueError(
            "FC2 masses must exactly match the ordered crystal supercell masses."
        )
    if fc2.p2s_map is not None and not np.array_equal(fc2.p2s_map, crystal.p2s):
        raise ValueError("FC2 p2s_map conflicts with CrystalMapping.p2s.")
    if fc2.s2p_map is not None and not np.array_equal(fc2.s2p_map, crystal.s2p):
        raise ValueError("FC2 s2p_map conflicts with CrystalMapping.s2p.")
    if fc2.compact and fc2.values.shape[0] != crystal.n_primitive_atoms:
        raise ValueError(
            "Compact FC2 first-axis size must equal the primitive atom count."
        )

    raw_transform = np.asarray(transport_supercell_matrix)
    transform = _integer_transform(transport_supercell_matrix)
    integer_matrix_max_abs = float(
        np.max(
            np.abs(
                np.asarray(raw_transform, dtype=np.float64)
                - transform.astype(np.float64)
            )
        )
    )
    signed_determinant = _determinant_int3(transform)
    inverse = _readonly(
        np.linalg.inv(transform.astype(np.float64)),
        dtype=np.float64,
        name="primitive_to_transport",
    )
    transform_float = _readonly(
        transform.astype(np.float64),
        dtype=np.float64,
        name="transport_to_primitive",
    )
    transport_lattice = _readonly(
        transform @ crystal.primitive.lattice,
        dtype=np.float64,
        name="transport_lattice_angstrom",
    )
    direct, reciprocal = enumerate_direct_reciprocal_cosets(transform)

    basis_cosets: list[int] = []
    basis_atoms: list[int] = []
    basis_integer_cells: list[Array] = []
    basis_scaled: list[Array] = []
    for ordinal, representative in enumerate(direct.representatives):
        for primitive_atom in range(crystal.n_primitive_atoms):
            raw = (
                crystal.primitive.scaled_positions[primitive_atom]
                + representative
            ) @ inverse
            transport_cell, fractional = _decompose_transport(raw)
            primitive_cell = representative - transport_cell @ transform
            basis_cosets.append(ordinal)
            basis_atoms.append(primitive_atom)
            basis_integer_cells.append(primitive_cell)
            basis_scaled.append(fractional)

    basis_coset_array = _readonly_int(basis_cosets, name="basis_coset_ordinals")
    basis_atom_array = _readonly_int(basis_atoms, name="basis_primitive_atoms")
    basis_cell_array = _readonly_int(basis_integer_cells, name="basis_integer_cells")
    basis_scaled_array = _readonly(
        basis_scaled, dtype=np.float64, name="basis_scaled_positions"
    )
    basis_cartesian_m = _readonly(
        basis_scaled_array @ transport_lattice * ANGSTROM_M,
        dtype=np.float64,
        name="basis_cartesian_positions_m",
    )

    expected_basis = abs(signed_determinant) * crystal.n_primitive_atoms
    if basis_atom_array.size != expected_basis:
        raise RuntimeError("Transport-basis coset construction is incomplete.")
    # Fail closed on a duplicate (primitive class, periodic position).
    for left in range(expected_basis):
        for right in range(left + 1, expected_basis):
            if basis_atom_array[left] != basis_atom_array[right]:
                continue
            residual = _periodic_residual(
                basis_scaled_array[left], basis_scaled_array[right]
            )
            if np.max(np.abs(residual)) <= _FRACTIONAL_MATCH_ATOL:
                raise ValueError("Transport-cell atom mapping is not bijective.")

    canonical = _CanonicalFC2Input.from_fc2(fc2)
    canonical_fc2 = canonical.as_fc2()
    content_sha256 = fc2_content_sha256(canonical_fc2)
    source_sha256, provenance_sha256 = _source_identity(canonical, content_sha256)
    primitive_sha256 = primitive_crystal_content_sha256(crystal)
    mapping_sha256 = crystal_mapping_content_sha256(crystal)
    redundant_audit = _audit_full_redundant_rows(canonical, crystal)

    (
        terms,
        support,
        exact_numerical_support,
        n_images,
        atom_fractional_match_max,
        atom_cartesian_match_max_m,
    ) = _project_terms(
        canonical,
        crystal,
        primitive_to_transport=inverse,
        transport_lattice_angstrom=transport_lattice,
        basis_primitive_atoms=basis_atom_array,
        basis_integer_cells=basis_cell_array,
        basis_scaled_positions=basis_scaled_array,
    )
    principal_cells = _infer_principal_cells(terms)
    dof_records = _readonly_int(
        [
            (subcell, basis, cartesian)
            for subcell in range(principal_cells)
            for basis in range(expected_basis)
            for cartesian in range(3)
        ],
        name="principal_layer_dof_records",
    )
    layer_offsets = _readonly_int(
        sorted(
            {
                (subcell + term.longitudinal_shift) // principal_cells
                for subcell in range(principal_cells)
                for term in terms
            }
        ),
        name="layer_offsets",
    )
    if np.any(np.abs(layer_offsets) > 1):
        raise RuntimeError("Principal-layer inference retained a long-range block.")

    coordinate_roundtrip_max_abs = _coordinate_roundtrip_residual(
        crystal,
        transform,
        basis_atom_array,
        basis_cell_array,
        basis_scaled_array,
    )
    direct_reciprocal_phase_max_abs = _direct_reciprocal_phase_residual(
        transform,
        direct.representatives,
        reciprocal.representatives,
    )
    mass_factor = mass_to_si_factor(canonical.mass_units)
    source_masses_kg = canonical.masses * mass_factor
    mapped_masses_kg = crystal.supercell.masses * mass_factor
    mass_denominator = np.maximum(
        np.maximum(np.abs(source_masses_kg), np.abs(mapped_masses_kg)),
        np.finfo(np.float64).tiny,
    )
    mass_match_max_relative = float(
        np.max(np.abs(source_masses_kg - mapped_masses_kg) / mass_denominator)
    )
    principal_layer_minimal = _principal_candidate_valid(
        terms,
        principal_cells,
    ) and (
        principal_cells == 1
        or not _principal_candidate_valid(terms, principal_cells - 1)
    )

    geometry_sha256 = domain_digest(
        "NEGFPY-LAYERED-GEOMETRY",
        _mapping_payload(
            primitive_sha256=primitive_sha256,
            fc2_mapping_sha256=mapping_sha256,
            transport_supercell_matrix=transform,
            signed_determinant=signed_determinant,
            transport_lattice_angstrom=transport_lattice,
            primitive_to_transport=inverse,
            transport_to_primitive=transform_float,
            direct=direct,
            reciprocal=reciprocal,
            basis_coset_ordinals=basis_coset_array,
            basis_primitive_atoms=basis_atom_array,
            basis_integer_cells=basis_cell_array,
            basis_scaled_positions=basis_scaled_array,
            basis_cartesian_positions_m=basis_cartesian_m,
            principal_cells=principal_cells,
            principal_layer_dof_records=dof_records,
            support=support,
            layer_offsets=layer_offsets,
        ),
    )
    exact_numerical_support_sha256 = domain_digest(
        "NEGFPY-LAYERED-FC2-EXACT-NUMERICAL-SUPPORT",
        {
            "schema_version": (
                "negfpy.layered-fc2-exact-numerical-support.v1"
            ),
            "fc2_content_sha256": digest_bytes(content_sha256),
            "primitive_crystal_content_sha256": digest_bytes(
                primitive_sha256
            ),
            "fc2_crystal_mapping_content_sha256": digest_bytes(
                mapping_sha256
            ),
            "transport_supercell_matrix": transform,
            "records": exact_numerical_support,
        },
    )

    legacy = canonical.provenance.get("oracle_v0_2_fc2_mapping_sha256")
    if legacy is not None:
        legacy = validate_sha256(
            legacy, name="oracle_v0_2_fc2_mapping_sha256"
        )
    metrics = _freeze_mapping(
        {
            "n_fc2_image_records": n_images,
            "n_fc2_image_envelope_records": int(support.shape[0]),
            "n_fc2_exact_numerical_support_records": int(
                exact_numerical_support.shape[0]
            ),
            "longitudinal_fc2_exact_numerical_support_sha256": (
                exact_numerical_support_sha256
            ),
            "n_redundant_first_axis_rows": redundant_audit[
                "checked_first_axis_rows"
            ],
            "redundant_rows_max_abs_source": redundant_audit[
                "maximum_absolute_residual"
            ],
            "fc2_schema": True,
            "crystal_schema": True,
            "source_identity": True,
            "integer_transform": (
                integer_matrix_max_abs <= _INTEGER_MATRIX_ATOL
            ),
            "direct_cosets": (
                direct.keys.shape == (abs(signed_determinant), 3)
            ),
            "reciprocal_cosets": (
                reciprocal.keys.shape == (abs(signed_determinant), 3)
            ),
            "integer_matrix_max_abs": integer_matrix_max_abs,
            "coordinate_roundtrip": (
                coordinate_roundtrip_max_abs <= _INTEGER_MATRIX_ATOL
            ),
            "coordinate_roundtrip_max_abs": coordinate_roundtrip_max_abs,
            "atom_bijection": True,
            "mass_match": mass_match_max_relative <= _MASS_MATCH_RTOL,
            "mass_match_max_relative": mass_match_max_relative,
            "support_complete": True,
            "principal_layer_minimal": principal_layer_minimal,
            "nearest_layer_range": bool(np.all(np.abs(layer_offsets) <= 1)),
            "direct_quotient_max_integer_residual": 0.0,
            "reciprocal_quotient_max_integer_residual": 0.0,
            "direct_reciprocal_phase_max_abs": (
                direct_reciprocal_phase_max_abs
            ),
            "direct_reciprocal_phase": (
                direct_reciprocal_phase_max_abs
                <= _DIRECT_RECIPROCAL_PHASE_ATOL
            ),
            "atom_fractional_match_max_abs": atom_fractional_match_max,
            "atom_cartesian_match_max_m": atom_cartesian_match_max_m,
        }
    )
    result = object.__new__(LayeredDeviceMapping)
    fields = {
        "transport_supercell_matrix": transform,
        "signed_determinant": signed_determinant,
        "transport_lattice_angstrom": transport_lattice,
        "primitive_to_transport": inverse,
        "transport_to_primitive": transform_float,
        "direct_coset_keys": direct.keys,
        "direct_coset_representatives": direct.representatives,
        "reciprocal_coset_keys": reciprocal.keys,
        "reciprocal_coset_representatives": reciprocal.representatives,
        "direct_cosets_sha256": direct.sha256,
        "reciprocal_cosets_sha256": reciprocal.sha256,
        "basis_coset_ordinals": basis_coset_array,
        "basis_primitive_atoms": basis_atom_array,
        "basis_integer_cells": basis_cell_array,
        "basis_scaled_positions": basis_scaled_array,
        "basis_cartesian_positions_m": basis_cartesian_m,
        "principal_cells": principal_cells,
        "principal_layer_dof_records": dof_records,
        "transverse_repeats": (1, 1),
        "longitudinal_fc2_support": support,
        "longitudinal_fc2_exact_numerical_support": (
            exact_numerical_support
        ),
        "longitudinal_fc2_exact_numerical_support_sha256": (
            exact_numerical_support_sha256
        ),
        "layer_offsets": layer_offsets,
        "fc2_content_sha256": content_sha256,
        "fc2_layout": "compact" if canonical.compact else "full",
        "fc2_source_sha256": source_sha256,
        "fc2_provenance_sha256": provenance_sha256,
        "oracle_v0_2_fc2_mapping_sha256": legacy,
        "primitive_crystal_content_sha256": primitive_sha256,
        "fc2_crystal_mapping_content_sha256": mapping_sha256,
        "layered_geometry_sha256": geometry_sha256,
        "validation": None,
        "_canonical_fc2": canonical,
        "_crystal": crystal,
        "_terms": terms,
        "_geometry_metrics": metrics,
    }
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    from .layered_validation import validate_layered_device_mapping

    validation = validate_layered_device_mapping(result)
    if not validation.certification_passed:
        raise ValueError(
            "Layered geometry validation failed: "
            + ", ".join(validation.failure_codes)
        )
    object.__setattr__(result, "validation", validation)
    return result


def _model_payload(
    mapping: LayeredDeviceMapping,
    scale: SolverScale,
    *,
    d00: Array,
    d01: Array,
    d10: Array,
    atom_masses_kg: Array,
    rigid_translation_vectors: Array,
) -> Mapping[str, Any]:
    return {
        "schema_version": "negfpy.layered-harmonic-model-physics.v1",
        "fc2_content_sha256": digest_bytes(mapping.fc2_content_sha256),
        "primitive_crystal_content_sha256": digest_bytes(
            mapping.primitive_crystal_content_sha256
        ),
        "fc2_crystal_mapping_content_sha256": digest_bytes(
            mapping.fc2_crystal_mapping_content_sha256
        ),
        "layered_geometry_sha256": digest_bytes(mapping.layered_geometry_sha256),
        "scale": {
            "omega0_rad_s": scale.omega0,
            "hbar_j_s": scale.hbar,
        },
        "unit_conversion": {
            "convention": "negfpy-si-codata-2018-v1",
            "electron_volt_j": ELECTRON_VOLT_J,
            "angstrom_m": ANGSTROM_M,
            "atomic_mass_unit_kg": ATOMIC_MASS_UNIT_KG,
            "fc2_to_j_m2": fc2_force_constant_to_si_factor(
                mapping._canonical_fc2.units
            ),
            "mass_to_kg": mass_to_si_factor(
                mapping._canonical_fc2.mass_units
            ),
            "dynamical_to_solver": 1.0 / scale.omega0**2,
        },
        "d00": d00,
        "d01": d01,
        "d10": d10,
        "principal_layer_atom_masses_kg": atom_masses_kg,
        "rigid_translation_vectors": rigid_translation_vectors,
    }


def _phase_a_primitive_qpoints() -> Array:
    import hashlib

    points: list[tuple[float, float, float]] = [
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.0, 0.0, 0.5),
        (0.5, 0.5, 0.0),
        (0.5, 0.0, 0.5),
        (0.0, 0.5, 0.5),
        (0.5, 0.5, 0.5),
        (1.0 / 3.0, -2.0 / 7.0, 3.0 / 11.0),
    ]
    seed = 0x4E45474650593033
    for counter in range(32):
        digest = hashlib.sha256(
            b"NEGFPY-PHASE-A-QPOINT\x00"
            + (1).to_bytes(2, "little")
            + seed.to_bytes(8, "little")
            + counter.to_bytes(8, "little")
        ).digest()
        coordinate = []
        for axis in range(3):
            integer = int.from_bytes(
                digest[8 * axis : 8 * axis + 8],
                "little",
            )
            coordinate.append(((integer >> 11) + 0.5) / 2**53 - 0.5)
        points.append(tuple(coordinate))
    return np.asarray(points, dtype=np.float64)


def _phase_a_longitudinal_qpoints() -> Array:
    return np.asarray(
        [(n / 65.0, 0.0, 0.0) for n in range(-32, 33)]
        + [(0.5, 0.0, 0.0)],
        dtype=np.float64,
    )


def _primitive_dynamical_literal(
    canonical: _CanonicalFC2Input,
    crystal: CrystalMapping,
    qpoint: Array,
) -> Array:
    n_primitive = crystal.n_primitive_atoms
    matrix = np.zeros((3 * n_primitive, 3 * n_primitive), dtype=np.complex128)
    masses = crystal.primitive.masses
    for primitive_i, representative_i_raw in enumerate(crystal.p2s):
        representative_i = int(representative_i_raw)
        source_first = primitive_i if canonical.compact else representative_i
        for primitive_j, representative_j_raw in enumerate(crystal.p2s):
            representative_j = int(representative_j_raw)
            block = np.zeros((3, 3), dtype=np.complex128)
            for source_second in range(crystal.n_supercell_atoms):
                if int(crystal.s2p[source_second]) != representative_j:
                    continue
                count = int(crystal.multiplicity[source_second, primitive_i, 0])
                offset = int(crystal.multiplicity[source_second, primitive_i, 1])
                vectors = crystal.shortest_vectors[offset : offset + count]
                phase = np.exp(2j * np.pi * (vectors @ qpoint)).sum() / count
                block += canonical.values[source_first, source_second] * phase
            block /= np.sqrt(masses[primitive_i] * masses[primitive_j])
            row = slice(3 * primitive_i, 3 * primitive_i + 3)
            column = slice(3 * primitive_j, 3 * primitive_j + 3)
            matrix[row, column] = block
    return matrix


def _relative_residual(left: Array, right: Array, *, atol: float) -> tuple[float, float]:
    absolute_array = np.abs(np.asarray(left) - np.asarray(right))
    absolute = float(np.max(absolute_array)) if absolute_array.size else 0.0
    denominator = np.maximum(
        np.maximum(np.abs(left), np.abs(right)),
        atol if atol > 0.0 else np.finfo(np.float64).tiny,
    )
    relative = (
        float(np.max(absolute_array / denominator))
        if absolute_array.size
        else 0.0
    )
    return absolute, relative


def _folding_matrix(
    mapping: LayeredDeviceMapping,
    transport_q: Array,
) -> tuple[Array, tuple[Array, ...]]:
    principal_transform = np.array(
        mapping.transport_supercell_matrix,
        dtype=np.int64,
        order="C",
        copy=True,
    )
    principal_transform[0] *= mapping.principal_cells
    _principal_direct, principal_reciprocal = (
        enumerate_direct_reciprocal_cosets(principal_transform)
    )
    determinant = abs(principal_reciprocal.signed_determinant)
    n_primitive = mapping._crystal.n_primitive_atoms
    dimension = (
        mapping.principal_cells * mapping.n_transport_basis_atoms * 3
    )
    inverse_transpose = np.linalg.inv(
        principal_transform.astype(np.float64)
    ).T
    folding = np.zeros((dimension, dimension), dtype=np.complex128)
    primitive_qpoints: list[Array] = []
    for reciprocal_ordinal, reciprocal in enumerate(
        principal_reciprocal.representatives
    ):
        primitive_q = (transport_q + reciprocal) @ inverse_transpose
        primitive_qpoints.append(primitive_q)
        for subcell in range(mapping.principal_cells):
            subcell_translation = (
                subcell * mapping.transport_supercell_matrix[0]
            )
            for basis, primitive_atom_raw in enumerate(
                mapping.basis_primitive_atoms
            ):
                primitive_atom = int(primitive_atom_raw)
                anchor = (
                    mapping._crystal.primitive.scaled_positions[primitive_atom]
                    + mapping.basis_integer_cells[basis]
                    + subcell_translation
                )
                phase = np.exp(
                    2j * np.pi * float(anchor @ primitive_q)
                ) / np.sqrt(determinant)
                for cartesian in range(3):
                    row = 3 * (
                        subcell * mapping.n_transport_basis_atoms + basis
                    ) + cartesian
                    column = (
                        reciprocal_ordinal * 3 * n_primitive
                        + 3 * primitive_atom
                        + cartesian
                    )
                    folding[row, column] = phase
    return folding, tuple(primitive_qpoints)


def _harmonic_audit_metrics(
    mapping: LayeredDeviceMapping,
    scale: SolverScale,
    *,
    d00: Array,
    d01: Array,
    d10: Array,
) -> Mapping[str, Any]:
    from .bulk_harmonic import dynamical_matrix_from_fc2

    canonical = mapping._canonical_fc2
    crystal = mapping._crystal
    fc2 = canonical.as_fc2()
    reciprocal_max_abs = 0.0
    reciprocal_max_relative = 0.0
    reciprocity_max_abs = 0.0
    reciprocal_scale = 0.0
    for qpoint in _phase_a_primitive_qpoints():
        literal = _primitive_dynamical_literal(canonical, crystal, qpoint)
        oracle = dynamical_matrix_from_fc2(
            fc2,
            crystal,
            qpoint,
            hermitianize=False,
        )
        absolute, relative = _relative_residual(literal, oracle, atol=1.0e-12)
        reciprocal_max_abs = max(reciprocal_max_abs, absolute)
        reciprocal_max_relative = max(reciprocal_max_relative, relative)
        reciprocity_max_abs = max(
            reciprocity_max_abs,
            float(np.max(np.abs(literal - literal.conj().T))),
        )
        reciprocal_scale = max(reciprocal_scale, float(np.max(np.abs(literal))))

    asr_max_abs = 0.0
    source_scale = float(np.max(np.abs(canonical.values)))
    for primitive_i, representative_raw in enumerate(crystal.p2s):
        source_first = primitive_i if canonical.compact else int(representative_raw)
        row_sum = np.sum(canonical.values[source_first], axis=0)
        asr_max_abs = max(asr_max_abs, float(np.max(np.abs(row_sum))))
    asr_normalized = asr_max_abs / max(source_scale, np.finfo(np.float64).tiny)

    source_conversion = fc2_force_constant_to_si_factor(
        canonical.units
    ) / mass_to_si_factor(canonical.mass_units)
    source_d00 = np.asarray(d00) * scale.omega0**2 / source_conversion
    source_d01 = np.asarray(d01) * scale.omega0**2 / source_conversion
    source_d10 = np.asarray(d10) * scale.omega0**2 / source_conversion
    dimension = source_d00.shape[0]
    identity = np.eye(dimension, dtype=np.complex128)
    unitary_left = 0.0
    unitary_right = 0.0
    folding_forward_abs = 0.0
    folding_forward_relative = 0.0
    folding_inverse_abs = 0.0
    folding_inverse_relative = 0.0
    folding_forward_passed = True
    folding_inverse_passed = True
    for transport_q in _phase_a_longitudinal_qpoints():
        folding, primitive_qpoints = _folding_matrix(mapping, transport_q)
        unitary_left = max(
            unitary_left,
            float(np.max(np.abs(folding @ folding.conj().T - identity))),
        )
        unitary_right = max(
            unitary_right,
            float(np.max(np.abs(folding.conj().T @ folding - identity))),
        )
        primitive_block = np.zeros((dimension, dimension), dtype=np.complex128)
        n_primitive_dof = 3 * crystal.n_primitive_atoms
        for ordinal, primitive_q in enumerate(primitive_qpoints):
            block = _primitive_dynamical_literal(canonical, crystal, primitive_q)
            start = ordinal * n_primitive_dof
            primitive_block[
                start : start + n_primitive_dof,
                start : start + n_primitive_dof,
            ] = block
        phase = np.exp(2j * np.pi * float(transport_q[0]))
        layered = source_d00 + source_d01 * phase + source_d10 / phase
        folded = folding @ primitive_block @ folding.conj().T
        absolute, relative = _relative_residual(layered, folded, atol=1.0e-12)
        folding_forward_abs = max(folding_forward_abs, absolute)
        folding_forward_relative = max(folding_forward_relative, relative)
        forward_solver_residual = np.abs(layered - folded) * (
            source_conversion / scale.omega0**2
        )
        forward_solver_scale = np.maximum(
            np.abs(layered), np.abs(folded)
        ) * (source_conversion / scale.omega0**2)
        folding_forward_passed &= bool(
            np.all(
                forward_solver_residual
                <= 1.0e-12 + 1.0e-10 * forward_solver_scale
            )
        )
        inverted = folding.conj().T @ layered @ folding
        absolute, relative = _relative_residual(
            primitive_block,
            inverted,
            atol=1.0e-12,
        )
        folding_inverse_abs = max(folding_inverse_abs, absolute)
        folding_inverse_relative = max(folding_inverse_relative, relative)
        inverse_solver_residual = np.abs(primitive_block - inverted) * (
            source_conversion / scale.omega0**2
        )
        inverse_solver_scale = np.maximum(
            np.abs(primitive_block), np.abs(inverted)
        ) * (source_conversion / scale.omega0**2)
        folding_inverse_passed &= bool(
            np.all(
                inverse_solver_residual
                <= 1.0e-12 + 1.0e-10 * inverse_solver_scale
            )
        )

    minimum_omega2 = np.inf
    maximum_abs_omega2 = 0.0
    for numerator in range(-2048, 2048):
        q = numerator / 4096.0
        phase = np.exp(2j * np.pi * q)
        matrix = source_d00 + source_d01 * phase + source_d10 / phase
        hermitian = (matrix + matrix.conj().T) / 2.0
        eigenvalues = np.linalg.eigvalsh(hermitian)
        minimum_omega2 = min(minimum_omega2, float(np.min(eigenvalues)))
        maximum_abs_omega2 = max(
            maximum_abs_omega2,
            float(np.max(np.abs(eigenvalues))),
        )

    reciprocity_allowed = (
        1.0e-12 + 1.0e-10 * max(reciprocal_scale, np.finfo(float).tiny)
    )
    stable_floor = -1.0e-10 * maximum_abs_omega2
    return _freeze_mapping(
        {
            "full_redundant_rows": True,
            "mass_conversion": True,
            "reciprocity": reciprocity_max_abs <= reciprocity_allowed,
            "reciprocity_passed": reciprocity_max_abs <= reciprocity_allowed,
            "reciprocity_max_abs_source": reciprocity_max_abs,
            "reciprocity_allowed": reciprocity_allowed,
            "asr": asr_normalized <= 1.0e-10,
            "asr_passed": asr_normalized <= 1.0e-10,
            "asr_max_abs_source": asr_max_abs,
            "asr_normalized": asr_normalized,
            "range": True,
            "stability": minimum_omega2 >= stable_floor,
            "stability_passed": minimum_omega2 >= stable_floor,
            "minimum_omega2_source": minimum_omega2,
            "maximum_abs_omega2_source": maximum_abs_omega2,
            "reciprocal_reconstruction": (
                reciprocal_max_abs
                <= 1.0e-12 + 1.0e-10 * max(reciprocal_scale, 1.0e-12)
            ),
            "reciprocal_reconstruction_passed": (
                reciprocal_max_abs
                <= 1.0e-12 + 1.0e-10 * max(reciprocal_scale, 1.0e-12)
            ),
            "reciprocal_matrix_max_abs_source": reciprocal_max_abs,
            "reciprocal_matrix_max_relative": reciprocal_max_relative,
            "folding_unitarity": max(unitary_left, unitary_right) <= 1.0e-13,
            "folding_unitarity_passed": (
                max(unitary_left, unitary_right) <= 1.0e-13
            ),
            "folding_unitarity_left_max_abs": unitary_left,
            "folding_unitarity_right_max_abs": unitary_right,
            "folding_forward": folding_forward_passed,
            "folding_forward_passed": folding_forward_passed,
            "folding_forward_max_abs_solver": (
                folding_forward_abs * source_conversion / scale.omega0**2
            ),
            "folding_forward_max_relative": folding_forward_relative,
            "folding_inverse": folding_inverse_passed,
            "folding_inverse_passed": folding_inverse_passed,
            "folding_inverse_max_abs_source": folding_inverse_abs,
            "folding_inverse_max_relative": folding_inverse_relative,
        }
    )


def build_layered_harmonic_model(
    device_mapping: LayeredDeviceMapping,
    scale: SolverScale,
) -> LayeredHarmonicModel:
    """Mass-normalize one layered FC2 operator in explicit SI solver units."""

    if not isinstance(device_mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    canonical = device_mapping._canonical_fc2
    if fc2_content_sha256(canonical.as_fc2()) != device_mapping.fc2_content_sha256:
        raise ValueError("Embedded FC2 content hash no longer matches the mapping.")
    if (
        primitive_crystal_content_sha256(device_mapping._crystal)
        != device_mapping.primitive_crystal_content_sha256
    ):
        raise ValueError("Embedded primitive-crystal hash no longer matches.")
    if (
        crystal_mapping_content_sha256(device_mapping._crystal)
        != device_mapping.fc2_crystal_mapping_content_sha256
    ):
        raise ValueError("Embedded FC2 crystal-mapping hash no longer matches.")
    if device_mapping.validation is None or not device_mapping.validation.certification_passed:
        raise ValueError("LayeredDeviceMapping does not carry a passing audit.")

    p = device_mapping.principal_cells
    n_basis = device_mapping.n_transport_basis_atoms
    dimension = p * n_basis * 3
    audit = {
        -1: np.zeros((dimension, dimension), dtype=np.complex128),
        0: np.zeros((dimension, dimension), dtype=np.complex128),
        1: np.zeros((dimension, dimension), dtype=np.complex128),
    }
    force_factor = fc2_force_constant_to_si_factor(canonical.units)
    mass_factor = mass_to_si_factor(canonical.mass_units)
    primitive_masses_kg = (
        device_mapping._crystal.primitive.masses * mass_factor
    )
    solver_factor = force_factor / scale.omega0**2
    for subcell in range(p):
        for term in device_mapping._terms:
            if not np.any(term.values != 0.0):
                continue
            destination = subcell + term.longitudinal_shift
            layer_offset = destination // p
            destination_subcell = destination % p
            if layer_offset not in audit:
                raise ValueError(
                    "Mapped FC2 support exceeds the inferred principal layer."
                )
            row = 3 * (subcell * n_basis + term.row_basis)
            column = 3 * (
                destination_subcell * n_basis + term.column_basis
            )
            normalization = np.sqrt(
                primitive_masses_kg[term.primitive_i]
                * primitive_masses_kg[term.primitive_j]
            )
            audit[layer_offset][row : row + 3, column : column + 3] += (
                term.values * solver_factor / normalization
            )

    d00_audit = audit[0]
    d01_audit = audit[1]
    d10_audit = audit[-1]
    onsite_residual = float(np.max(np.abs(d00_audit - d00_audit.conj().T)))
    forward_reverse_residual = float(
        np.max(np.abs(d10_audit - d01_audit.conj().T))
    )
    onsite_scale = max(float(np.max(np.abs(d00_audit))), np.finfo(float).tiny)
    coupling_scale = max(
        float(np.max(np.abs(d01_audit))),
        float(np.max(np.abs(d10_audit))),
        np.finfo(float).tiny,
    )
    if onsite_residual > _OPERATOR_ATOL + _OPERATOR_RTOL * onsite_scale:
        raise ValueError(
            "Raw layered onsite block is not Hermitian within the frozen gate."
        )
    if (
        forward_reverse_residual
        > _OPERATOR_ATOL + _OPERATOR_RTOL * coupling_scale
    ):
        raise ValueError(
            "Raw layered forward/reverse blocks are not reciprocal within "
            "the frozen gate."
        )
    d00 = _readonly(
        (d00_audit + d00_audit.conj().T) / 2.0,
        dtype=np.complex128,
        name="d00",
    )
    d01 = _readonly(
        (d01_audit + d10_audit.conj().T) / 2.0,
        dtype=np.complex128,
        name="d01",
    )
    d10 = _readonly(d01.conj().T, dtype=np.complex128, name="d10")
    d00_audit_ro = _readonly(d00_audit, dtype=np.complex128, name="d00_audit")
    d01_audit_ro = _readonly(d01_audit, dtype=np.complex128, name="d01_audit")
    d10_audit_ro = _readonly(d10_audit, dtype=np.complex128, name="d10_audit")

    atom_masses = _readonly(
        [
            primitive_masses_kg[int(primitive_atom)]
            for _subcell in range(p)
            for primitive_atom in device_mapping.basis_primitive_atoms
        ],
        dtype=np.float64,
        name="principal_layer_atom_masses_kg",
    )
    rigid = np.zeros((dimension, 3), dtype=np.float64)
    for atom, mass in enumerate(atom_masses):
        root_mass = np.sqrt(mass)
        for cartesian in range(3):
            rigid[3 * atom + cartesian, cartesian] = root_mass
    for cartesian in range(3):
        norm = np.linalg.norm(rigid[:, cartesian])
        if norm == 0.0:
            raise RuntimeError("Rigid-translation vector has zero norm.")
        rigid[:, cartesian] /= norm
    rigid_ro = _readonly(
        rigid, dtype=np.float64, name="rigid_translation_vectors"
    )

    physics_sha256 = domain_digest(
        "NEGFPY-LAYERED-HARMONIC-MODEL-PHYSICS",
        _model_payload(
            device_mapping,
            scale,
            d00=d00,
            d01=d01,
            d10=d10,
            atom_masses_kg=atom_masses,
            rigid_translation_vectors=rigid_ro,
        ),
    )
    source_conversion = force_factor / mass_factor
    source_blocks = MappingProxyType(
        {
            -1: _readonly(
                d10_audit_ro * scale.omega0**2 / source_conversion,
                dtype=np.complex128,
                name="source d10 audit",
            ),
            0: _readonly(
                d00_audit_ro * scale.omega0**2 / source_conversion,
                dtype=np.complex128,
                name="source d00 audit",
            ),
            1: _readonly(
                d01_audit_ro * scale.omega0**2 / source_conversion,
                dtype=np.complex128,
                name="source d01 audit",
            ),
        }
    )
    canonicalization_delta = max(
        float(np.max(np.abs(d00 - d00_audit))),
        float(np.max(np.abs(d01 - d01_audit))),
        float(np.max(np.abs(d10 - d10_audit))),
    )
    scientific_metrics = dict(
        _harmonic_audit_metrics(
            device_mapping,
            scale,
            d00=d00,
            d01=d01,
            d10=d10,
        )
    )
    audit_metrics = _freeze_mapping(
        {
            **scientific_metrics,
            "raw_onsite_hermiticity_max_abs_solver": onsite_residual,
            "raw_forward_reverse_max_abs_solver": forward_reverse_residual,
            "canonicalization_delta_max_abs_solver": canonicalization_delta,
            "redundant_rows_max_abs_source": device_mapping._geometry_metrics[
                "redundant_rows_max_abs_source"
            ],
        }
    )
    lead = LeadBlocks(d00=d00, d01=d01, d10=d10)
    result = object.__new__(LayeredHarmonicModel)
    fields = {
        "mapping": device_mapping,
        "scale": scale,
        "d00": d00,
        "d01": d01,
        "d10": d10,
        "principal_layer_atom_masses_kg": atom_masses,
        "rigid_translation_vectors": rigid_ro,
        "fc2_content_sha256": device_mapping.fc2_content_sha256,
        "fc2_layout": device_mapping.fc2_layout,
        "fc2_source_sha256": device_mapping.fc2_source_sha256,
        "fc2_provenance_sha256": device_mapping.fc2_provenance_sha256,
        "oracle_v0_2_fc2_mapping_sha256": (
            device_mapping.oracle_v0_2_fc2_mapping_sha256
        ),
        "primitive_crystal_content_sha256": (
            device_mapping.primitive_crystal_content_sha256
        ),
        "fc2_crystal_mapping_content_sha256": (
            device_mapping.fc2_crystal_mapping_content_sha256
        ),
        "layered_geometry_sha256": device_mapping.layered_geometry_sha256,
        "model_physics_sha256": physics_sha256,
        "validation": None,
        "lead": lead,
        "_d00_audit": d00_audit_ro,
        "_d01_audit": d01_audit_ro,
        "_d10_audit": d10_audit_ro,
        "_source_blocks_by_offset": source_blocks,
        "_audit_metrics": audit_metrics,
    }
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    from .layered_validation import validate_layered_harmonic_model

    validation = validate_layered_harmonic_model(result)
    if not validation.certification_passed:
        raise ValueError(
            "Layered harmonic validation failed: "
            + ", ".join(validation.failure_codes)
        )
    object.__setattr__(result, "validation", validation)
    return result


__all__ = [
    "LayeredDeviceMapping",
    "LayeredHarmonicModel",
    "build_layered_device_mapping",
    "build_layered_harmonic_model",
]
