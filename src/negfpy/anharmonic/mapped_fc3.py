"""Finite-device FC3 stages for the sparse material ``N_q=1`` path.

The objects in this module deliberately separate three operations:

* unit and mass normalization of an audited translational FC3 schedule;
* exact finite-domain mapping in physical SI units; and
* conversion to the dimensionless cubic vertex used by the solver.

No operation symmetrizes, thresholds, or range-truncates FC3.  Image and
anchor weights remain distinct from the normalized source block until the
finite mapper applies them exactly once.  Compact FC3 is intentionally
rejected in this first Phase-B slice: accepting it requires the immutable
full/compact paired-certification object specified by the 0.3 design.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import math
import struct
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._canonical import (
    canonical_encode,
    crystal_mapping_content_sha256,
    digest_bytes,
    domain_digest,
    primitive_crystal_content_sha256,
    validate_sha256,
)
from .crystal import CrystalMapping, validate_crystal_mapping
from .fc3 import (
    ANGSTROM_M,
    FC3ValidationReport,
    fc3_force_constant_to_si_factor,
    mass_to_si_factor,
)
from .layered import LayeredDeviceMapping
from .translational_fc3 import FC3Term, TranslationalFC3
from .translational_fc3_validation import validate_translational_fc3
from .types import SolverScale


Array = np.ndarray
Metadata = Mapping[str, Any]
CertificationKind = Literal["full_validation", "paired_full_compact"]

_PERMUTATION_ATOL = 1.0e-12
_PERMUTATION_RTOL = 1.0e-10
_FRACTIONAL_SNAP_ATOL = 1.0e-10
_FRACTIONAL_MATCH_ATOL = 5.0e-10
_CARTESIAN_MATCH_ATOL_M = 1.0e-12
_PAIRS = ((0, 1), (0, 2), (1, 2))
_UINT64_MAX = int(np.iinfo(np.uint64).max)
_UINT64_BYTES = int(np.dtype(np.uint64).itemsize)
_FLOAT64_BYTES = int(np.dtype(np.float64).itemsize)
_INDEX_BYTES = int(np.dtype(np.intp).itemsize)


def _readonly(values: Any, *, dtype: Any, name: str) -> Array:
    raw = np.asarray(values)
    if dtype == np.int64 and (raw.dtype.kind not in "iu" or raw.dtype.kind == "b"):
        raise TypeError(f"{name} must contain exact integers.")
    if dtype == np.float64 and (raw.dtype.kind == "c" or raw.dtype.kind == "O"):
        raise TypeError(f"{name} must be a real numerical array.")
    try:
        result = np.array(raw, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} cannot be represented by its declared dtype.") from exc
    if dtype == np.int64 and not np.array_equal(result, raw):
        raise ValueError(f"{name} conversion to int64 changes a value.")
    if dtype == np.float64:
        if not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must contain only finite values.")
        result[result == 0.0] = 0.0
    result.setflags(write=False)
    return result


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = np.array(value, order="C", copy=True)
        if result.dtype.kind in "fc":
            if not np.all(np.isfinite(result)):
                raise ValueError("Metadata arrays must contain only finite values.")
            if result.dtype.kind == "f":
                result[result == 0.0] = 0.0
            else:
                components = result.view(np.float64).reshape(result.shape + (2,))
                components[components == 0.0] = 0.0
        result.setflags(write=False)
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        raise TypeError("Sets are not valid deterministic metadata.")
    return value


def _freeze_mapping(value: Metadata | None) -> Metadata:
    if value is None:
        return MappingProxyType({})
    if not isinstance(value, Mapping):
        raise TypeError("metadata must be a mapping.")
    frozen = MappingProxyType(
        {str(key): _freeze_value(item) for key, item in value.items()}
    )
    # Fail here rather than at a later checkpoint/hash boundary.
    canonical_encode(frozen)
    return frozen


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return result


def _layer_tuple(
    values: Sequence[int],
    *,
    name: str,
    n_device_layers: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence.")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer sequence.") from exc
    if not raw:
        raise ValueError(f"{name} must be nonempty.")
    result: list[int] = []
    for item in raw:
        if isinstance(item, (bool, np.bool_)) or not isinstance(
            item, (int, np.integer)
        ):
            raise TypeError(f"{name} must contain only integers.")
        integer = int(item)
        if not 0 <= integer < n_device_layers:
            raise ValueError(f"{name} contains an out-of-device layer.")
        result.append(integer)
    normalized = tuple(result)
    if normalized != tuple(sorted(set(normalized))):
        raise ValueError(f"{name} must be ordered and unique.")
    if normalized != tuple(range(normalized[0], normalized[-1] + 1)):
        raise ValueError(f"{name} must be contiguous.")
    return normalized


def _translational_tolerances(fc3: TranslationalFC3) -> tuple[float, float]:
    audit = fc3.provenance.get("redundant_translation_audit", {})
    if not isinstance(audit, Mapping):
        raise ValueError("Translational FC3 has an invalid translation audit.")
    atol = audit.get("atol", fc3.provenance.get("translation_atol", 1.0e-12))
    rtol = audit.get("rtol", fc3.provenance.get("translation_rtol", 1.0e-10))
    result = (float(atol), float(rtol))
    if any(not np.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("Translation tolerances must be finite and nonnegative.")
    return result


def _term_payload(term: FC3Term) -> Mapping[str, Any]:
    return {
        "primitive_indices": np.asarray(term.primitive_indices, dtype=np.int64),
        "source_indices": np.asarray(term.source_indices, dtype=np.int64),
        "translated_supercell_indices": np.asarray(
            term.translated_supercell_indices, dtype=np.int64
        ),
        "anchor_position": np.asarray(term.anchor_position, dtype=np.float64),
        "second_offset": np.asarray(term.second_offset, dtype=np.float64),
        "third_offset": np.asarray(term.third_offset, dtype=np.float64),
        "second_vector_index": term.second_vector_index,
        "third_vector_index": term.third_vector_index,
        "second_multiplicity": term.second_multiplicity,
        "third_multiplicity": term.third_multiplicity,
        "anchor_weight": term.anchor_weight,
        "multiplicity_weight": term.multiplicity_weight,
        "values": np.asarray(term.values, dtype=np.float64),
    }


def _translational_content_sha256(fc3: TranslationalFC3) -> str:
    atol, rtol = _translational_tolerances(fc3)
    payload = {
        "terms": tuple(_term_payload(term) for term in fc3.terms),
        "n_primitive_atoms": fc3.n_primitive_atoms,
        "n_supercell_atoms": fc3.n_supercell_atoms,
        "units": fc3.units,
        "mass_units": fc3.mass_units,
        "source_layout": fc3.source_layout,
        "phase_convention": fc3.phase_convention,
        "anchor_convention": fc3.anchor_convention,
        "make_r0_average": fc3.make_r0_average,
        "all_shortest": (
            None
            if fc3.all_shortest is None
            else np.asarray(fc3.all_shortest, dtype=np.uint8)
        ),
        "effective_symprec": fc3.symprec,
        "translation_atol": atol,
        "translation_rtol": rtol,
    }
    return domain_digest("NEGFPY-TRANSLATIONAL-FC3-BRIDGE", payload)


def _source_fc3_content_sha256(fc3: TranslationalFC3) -> str:
    candidates = [
        fc3.provenance.get("source_fc3_content_sha256"),
    ]
    nested = fc3.provenance.get("source_fc3_provenance")
    if isinstance(nested, Mapping):
        candidates.append(nested.get("source_fc3_content_sha256"))
    present = [value for value in candidates if value is not None]
    if not present:
        raise ValueError(
            "TranslationalFC3 is missing source_fc3_content_sha256; "
            "Phase-B normalization cannot authenticate the raw FC3 source."
        )
    normalized = tuple(
        validate_sha256(value, name="source_fc3_content_sha256")
        for value in present
    )
    if len(set(normalized)) != 1:
        raise ValueError("Conflicting source FC3 content identities were supplied.")
    return normalized[0]


def _full_validation_sha256(
    report: FC3ValidationReport,
    *,
    source_fc3_content_sha256: str,
) -> str:
    payload = {
        "schema_version": "negfpy.fc3-validation-bridge.v1",
        "source_fc3_content_sha256": digest_bytes(source_fc3_content_sha256),
        "dimension_valid": report.dimension_valid,
        "finite_valid": report.finite_valid,
        "masses_valid": report.masses_valid,
        "units_valid": report.units_valid,
        "mapping_valid": report.mapping_valid,
        "permutation_valid": report.permutation_valid,
        "permutation_complete": report.permutation_complete,
        "errors": report.errors,
        "warnings": report.warnings,
        "metrics": dict(report.metrics),
        "checked_permutations": report.checked_permutations,
    }
    return domain_digest("NEGFPY-FC3-VALIDATION-BRIDGE", payload)


@dataclass(frozen=True, slots=True, eq=False)
class _MassNormalizedFC3Term:
    source_term_index: int
    primitive_indices: tuple[int, int, int]
    source_indices: tuple[int, int, int]
    translated_supercell_indices: tuple[int, int, int]
    anchor_position: Array
    second_offset: Array
    third_offset: Array
    second_vector_index: int
    third_vector_index: int
    second_multiplicity: int
    third_multiplicity: int
    anchor_weight: float
    multiplicity_weight: float
    values: Array

    def __post_init__(self) -> None:
        if self.source_term_index < 0:
            raise ValueError("source_term_index must be nonnegative.")
        object.__setattr__(
            self, "values", _readonly(self.values, dtype=np.float64, name="values")
        )
        for name in ("anchor_position", "second_offset", "third_offset"):
            value = _readonly(
                getattr(self, name), dtype=np.float64, name=name
            )
            if value.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True, eq=False, init=False)
class MassNormalizedTranslationalFC3:
    """Physical-SI, mass-normalized FC3 with weights still unapplied."""

    terms: tuple[_MassNormalizedFC3Term, ...]
    n_primitive_atoms: int
    n_supercell_atoms: int
    source_units: str
    source_mass_units: str
    output_units: str
    source_layout: Literal["full", "compact"]
    phase_convention: str
    anchor_convention: str
    make_r0_average: bool
    all_shortest: Array | None
    effective_symprec: float | None
    translation_atol: float
    translation_rtol: float
    force_constant_to_si_factor: float
    mass_to_si_factor: float
    source_fc3_content_sha256: str
    source_translational_sha256: str
    primitive_crystal_content_sha256: str
    fc3_crystal_mapping_content_sha256: str
    source_certification_kind: CertificationKind
    source_certification_sha256: str
    content_sha256: str
    metadata: Metadata

    def __init__(self) -> None:
        raise TypeError(
            "MassNormalizedTranslationalFC3 is builder-created; use "
            "mass_normalize_translational_fc3(...)."
        )


def _normalized_payload(
    result: MassNormalizedTranslationalFC3,
) -> Mapping[str, Any]:
    return {
        "schema_version": "negfpy.mass-normalized-translational-fc3.v1",
        "terms": tuple(
            {
                "source_term_index": term.source_term_index,
                **{
                    key: value
                    for key, value in _term_payload(
                        FC3Term(
                            primitive_indices=term.primitive_indices,
                            source_indices=term.source_indices,
                            translated_supercell_indices=(
                                term.translated_supercell_indices
                            ),
                            anchor_position=term.anchor_position,
                            second_offset=term.second_offset,
                            third_offset=term.third_offset,
                            second_vector_index=term.second_vector_index,
                            third_vector_index=term.third_vector_index,
                            second_multiplicity=term.second_multiplicity,
                            third_multiplicity=term.third_multiplicity,
                            anchor_weight=term.anchor_weight,
                            multiplicity_weight=term.multiplicity_weight,
                            values=term.values,
                        )
                    ).items()
                },
            }
            for term in result.terms
        ),
        "n_primitive_atoms": result.n_primitive_atoms,
        "n_supercell_atoms": result.n_supercell_atoms,
        "source_units": result.source_units,
        "source_mass_units": result.source_mass_units,
        "output_units": result.output_units,
        "source_layout": result.source_layout,
        "phase_convention": result.phase_convention,
        "anchor_convention": result.anchor_convention,
        "make_r0_average": result.make_r0_average,
        "all_shortest": result.all_shortest,
        "effective_symprec": result.effective_symprec,
        "translation_atol": result.translation_atol,
        "translation_rtol": result.translation_rtol,
        "force_constant_to_si_factor": result.force_constant_to_si_factor,
        "mass_to_si_factor": result.mass_to_si_factor,
        "source_fc3_content_sha256": digest_bytes(
            result.source_fc3_content_sha256
        ),
        "source_translational_sha256": digest_bytes(
            result.source_translational_sha256
        ),
        "primitive_crystal_content_sha256": digest_bytes(
            result.primitive_crystal_content_sha256
        ),
        "fc3_crystal_mapping_content_sha256": digest_bytes(
            result.fc3_crystal_mapping_content_sha256
        ),
        "source_certification_kind": result.source_certification_kind,
        "source_certification_sha256": digest_bytes(
            result.source_certification_sha256
        ),
    }


def mass_normalize_translational_fc3(
    fc3: TranslationalFC3,
    mapping: CrystalMapping,
    *,
    source_certification: FC3ValidationReport,
    output_units: Literal["si"] = "si",
) -> MassNormalizedTranslationalFC3:
    """Convert source blocks to physical mass-normalized SI exactly once.

    The two image weights are copied verbatim and are *not* included in the
    normalized ``values`` field.  This first implementation accepts only a
    completely audited full source.  Compact input fails closed until the
    paired full/compact certification type exists.
    """

    if not isinstance(fc3, TranslationalFC3):
        raise TypeError("fc3 must be a TranslationalFC3.")
    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping.")
    if output_units != "si":
        raise ValueError("output_units must be exactly 'si'.")
    mapping_report = validate_crystal_mapping(mapping)
    if not mapping_report.passed or mapping.purpose != "fc3":
        raise ValueError("mapping must be a passing FC3 CrystalMapping.")
    if fc3.source_layout == "compact":
        raise NotImplementedError(
            "Compact TranslationalFC3 requires the not-yet-implemented "
            "FC3PairedCertificationReport paired-certification stage; a "
            "raw compact validation is insufficient."
        )
    if not isinstance(source_certification, FC3ValidationReport):
        raise TypeError(
            "A full source requires its FC3ValidationReport certification."
        )
    if not source_certification.passed or not source_certification.permutation_complete:
        raise ValueError(
            "Full FC3 certification must pass and have a complete "
            "permutation audit."
        )
    if (
        fc3.n_primitive_atoms != mapping.n_primitive_atoms
        or fc3.n_supercell_atoms != mapping.n_supercell_atoms
    ):
        raise ValueError("TranslationalFC3 atom counts conflict with mapping.")

    translation_report = validate_translational_fc3(fc3, mapping)
    if not translation_report.passed:
        raise ValueError(
            "Translational FC3 validation failed: "
            + "; ".join(translation_report.errors)
        )

    source_hash = _source_fc3_content_sha256(fc3)
    if source_certification.source_fc3_content_sha256 != source_hash:
        raise ValueError(
            "Full FC3 certification does not authenticate the translated "
            "raw FC3 source."
        )
    source_translational_hash = _translational_content_sha256(fc3)
    primitive_hash = primitive_crystal_content_sha256(mapping)
    mapping_hash = crystal_mapping_content_sha256(mapping)
    declared_mapping = fc3.provenance.get("fc3_crystal_mapping_content_sha256")
    if declared_mapping is None:
        raise ValueError(
            "TranslationalFC3 is missing "
            "fc3_crystal_mapping_content_sha256."
        )
    if validate_sha256(
        declared_mapping, name="fc3_crystal_mapping_content_sha256"
    ) != mapping_hash:
        raise ValueError("Translational FC3 mapping identity conflicts with mapping.")
    declared_primitive = fc3.provenance.get("primitive_crystal_content_sha256")
    if declared_primitive is None:
        raise ValueError(
            "TranslationalFC3 is missing primitive_crystal_content_sha256."
        )
    if validate_sha256(
        declared_primitive, name="primitive_crystal_content_sha256"
    ) != primitive_hash:
        raise ValueError("Translational FC3 primitive identity conflicts with mapping.")

    force_factor = fc3_force_constant_to_si_factor(fc3.units)
    mass_factor = mass_to_si_factor(fc3.mass_units)
    primitive_masses = np.asarray(mapping.primitive.masses, dtype=np.float64)
    primitive_masses_si = primitive_masses * mass_factor
    if not np.all(np.isfinite(primitive_masses_si)) or np.any(
        primitive_masses_si <= 0.0
    ):
        raise ValueError("FC3 primitive masses must be positive finite SI values.")

    normalized_terms: list[_MassNormalizedFC3Term] = []
    for source_index, term in enumerate(fc3.terms):
        values = np.array(term.values, dtype=np.float64, order="C", copy=True)
        values *= force_factor
        for primitive_atom in term.primitive_indices:
            values /= math.sqrt(float(primitive_masses_si[primitive_atom]))
        values[values == 0.0] = 0.0
        normalized_terms.append(
            _MassNormalizedFC3Term(
                source_term_index=source_index,
                primitive_indices=term.primitive_indices,
                source_indices=term.source_indices,
                translated_supercell_indices=term.translated_supercell_indices,
                anchor_position=term.anchor_position,
                second_offset=term.second_offset,
                third_offset=term.third_offset,
                second_vector_index=term.second_vector_index,
                third_vector_index=term.third_vector_index,
                second_multiplicity=term.second_multiplicity,
                third_multiplicity=term.third_multiplicity,
                anchor_weight=term.anchor_weight,
                multiplicity_weight=term.multiplicity_weight,
                values=values,
            )
        )

    atol, rtol = _translational_tolerances(fc3)
    certification_hash = _full_validation_sha256(
        source_certification,
        source_fc3_content_sha256=source_hash,
    )
    result = object.__new__(MassNormalizedTranslationalFC3)
    fields = {
        "terms": tuple(normalized_terms),
        "n_primitive_atoms": fc3.n_primitive_atoms,
        "n_supercell_atoms": fc3.n_supercell_atoms,
        "source_units": fc3.units,
        "source_mass_units": fc3.mass_units,
        "output_units": "J m^-3 kg^-3/2",
        "source_layout": fc3.source_layout,
        "phase_convention": fc3.phase_convention,
        "anchor_convention": fc3.anchor_convention,
        "make_r0_average": fc3.make_r0_average,
        "all_shortest": (
            None
            if fc3.all_shortest is None
            else _readonly(fc3.all_shortest, dtype=np.int64, name="all_shortest")
        ),
        "effective_symprec": fc3.symprec,
        "translation_atol": atol,
        "translation_rtol": rtol,
        "force_constant_to_si_factor": force_factor,
        "mass_to_si_factor": mass_factor,
        "source_fc3_content_sha256": source_hash,
        "source_translational_sha256": source_translational_hash,
        "primitive_crystal_content_sha256": primitive_hash,
        "fc3_crystal_mapping_content_sha256": mapping_hash,
        "source_certification_kind": "full_validation",
        "source_certification_sha256": certification_hash,
        "content_sha256": "",
        "metadata": _freeze_mapping(
            {
                "schema_version": "negfpy.mass-normalized-translational-fc3.v1",
                "weights_applied": False,
                "symmetrized": False,
                "corrected": False,
                "thresholded": False,
                "translational_validation_passed": True,
            }
        ),
    }
    for name, value in fields.items():
        object.__setattr__(result, name, value)
    content_hash = domain_digest(
        "NEGFPY-MASS-NORMALIZED-TRANSLATIONAL-FC3",
        _normalized_payload(result),
    )
    object.__setattr__(result, "content_sha256", content_hash)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class _EffectiveFC3Term:
    effective_leg: int
    source_term_index: int
    source_group: tuple[int, int, int]
    primitive_indices: tuple[int, int, int]
    positions: tuple[Array, Array, Array]
    second_vector_index: int
    third_vector_index: int
    values: Array
    multiplicity_weight: float
    anchor_weight: float
    r0_leg_factor: float


def _effective_terms(
    fc3: MassNormalizedTranslationalFC3,
) -> tuple[_EffectiveFC3Term, ...]:
    result: list[_EffectiveFC3Term] = []
    for term in fc3.terms:
        anchor = term.anchor_position
        second = anchor + term.second_offset
        third = anchor + term.third_offset
        if not fc3.make_r0_average:
            result.append(
                _EffectiveFC3Term(
                    effective_leg=1,
                    source_term_index=term.source_term_index,
                    source_group=term.primitive_indices,
                    primitive_indices=term.primitive_indices,
                    positions=(anchor, second, third),
                    second_vector_index=term.second_vector_index,
                    third_vector_index=term.third_vector_index,
                    values=term.values,
                    multiplicity_weight=term.multiplicity_weight,
                    anchor_weight=term.anchor_weight,
                    r0_leg_factor=1.0,
                )
            )
            continue

        assert fc3.all_shortest is not None
        flag = bool(
            fc3.all_shortest[
                term.primitive_indices[0],
                term.source_indices[1],
                term.source_indices[2],
            ]
        )
        result.append(
            _EffectiveFC3Term(
                effective_leg=1,
                source_term_index=term.source_term_index,
                source_group=term.primitive_indices,
                primitive_indices=term.primitive_indices,
                positions=(anchor, second, third),
                second_vector_index=term.second_vector_index,
                third_vector_index=term.third_vector_index,
                values=term.values,
                multiplicity_weight=term.multiplicity_weight,
                anchor_weight=term.anchor_weight,
                r0_leg_factor=1.0 if flag else 1.0 / 3.0,
            )
        )
        if flag:
            continue
        result.append(
            _EffectiveFC3Term(
                effective_leg=2,
                source_term_index=term.source_term_index,
                source_group=term.primitive_indices,
                primitive_indices=(
                    term.primitive_indices[1],
                    term.primitive_indices[0],
                    term.primitive_indices[2],
                ),
                positions=(second, anchor, third),
                second_vector_index=term.second_vector_index,
                third_vector_index=term.third_vector_index,
                values=np.transpose(term.values, (1, 0, 2)),
                multiplicity_weight=term.multiplicity_weight,
                anchor_weight=term.anchor_weight,
                r0_leg_factor=1.0 / 3.0,
            )
        )
        result.append(
            _EffectiveFC3Term(
                effective_leg=3,
                source_term_index=term.source_term_index,
                source_group=term.primitive_indices,
                primitive_indices=(
                    term.primitive_indices[2],
                    term.primitive_indices[1],
                    term.primitive_indices[0],
                ),
                positions=(third, second, anchor),
                second_vector_index=term.second_vector_index,
                third_vector_index=term.third_vector_index,
                values=np.transpose(term.values, (2, 1, 0)),
                multiplicity_weight=term.multiplicity_weight,
                anchor_weight=term.anchor_weight,
                r0_leg_factor=1.0 / 3.0,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda term: (
                term.primitive_indices,
                term.effective_leg,
                term.source_group,
                term.source_term_index,
                term.second_vector_index,
                term.third_vector_index,
            ),
        )
    )


def _decompose_transport(values: Array) -> tuple[Array, Array]:
    snapped = np.array(values, dtype=np.float64, order="C", copy=True)
    nearest = np.rint(snapped)
    mask = np.abs(snapped - nearest) <= _FRACTIONAL_SNAP_ATOL
    snapped[mask] = nearest[mask]
    cell = np.floor(snapped).astype(np.int64)
    fractional = snapped - cell
    near_one = np.abs(fractional - 1.0) <= _FRACTIONAL_SNAP_ATOL
    cell[near_one] += 1
    fractional[near_one] = 0.0
    fractional[np.abs(fractional) <= _FRACTIONAL_SNAP_ATOL] = 0.0
    return cell, fractional


def _map_position(
    *,
    primitive_atom: int,
    primitive_position: Array,
    mapping: LayeredDeviceMapping,
) -> tuple[Array, int]:
    transport = primitive_position @ mapping.primitive_to_transport
    cell, fractional = _decompose_transport(transport)
    matches: list[int] = []
    for raw_basis in np.flatnonzero(
        mapping.basis_primitive_atoms == primitive_atom
    ):
        basis = int(raw_basis)
        residual = fractional - mapping.basis_scaled_positions[basis]
        residual -= np.rint(residual)
        if np.max(np.abs(residual)) > _FRACTIONAL_MATCH_ATOL:
            continue
        cartesian = residual @ mapping.transport_lattice_angstrom * ANGSTROM_M
        if np.max(np.abs(cartesian)) > _CARTESIAN_MATCH_ATOL_M:
            continue
        matches.append(basis)
    if len(matches) != 1:
        raise ValueError(
            "An FC3 image must map to exactly one transport-basis atom; "
            f"found {len(matches)} for primitive atom {primitive_atom}."
        )
    return cell, matches[0]


def _mapped_geometry(
    term: _EffectiveFC3Term,
    representative: Array,
    mapping: LayeredDeviceMapping,
) -> tuple[tuple[Array, int], tuple[Array, int], tuple[Array, int]]:
    return tuple(
        _map_position(
            primitive_atom=primitive_atom,
            primitive_position=position + representative,
            mapping=mapping,
        )
        for primitive_atom, position in zip(
            term.primitive_indices, term.positions, strict=True
        )
    )  # type: ignore[return-value]


def _permutation_audit(
    ndof: int,
    indices: Array,
    values: Array,
) -> tuple[float, int, bool, bool]:
    """Audit an already lexicographically ordered sparse cubic vertex.

    A dictionary is deliberately not constructed here.  The compact output
    keys are monotone, so binary search provides the same exact membership
    test without reintroducing one Python object per mapped component.
    """

    if len(values) == 0:
        return 0.0, 0, True, True
    keys = (
        (
            np.asarray(indices[:, 0], dtype=np.uint64) * np.uint64(ndof)
            + np.asarray(indices[:, 1], dtype=np.uint64)
        )
        * np.uint64(ndof)
        + np.asarray(indices[:, 2], dtype=np.uint64)
    )
    maximum = 0.0
    missing = 0
    exact = True
    for ordinal, raw_key in enumerate(indices):
        key = tuple(int(value) for value in raw_key)
        reference = float(values[ordinal])
        reference_bytes = struct.pack("<d", reference)
        for permutation in sorted(set(itertools.permutations(key))):
            permutation_key = (
                (permutation[0] * ndof + permutation[1]) * ndof
                + permutation[2]
            )
            position = int(
                np.searchsorted(keys, np.uint64(permutation_key), side="left")
            )
            if position == len(keys) or int(keys[position]) != permutation_key:
                missing += 1
                exact = False
                maximum = max(maximum, abs(reference))
                continue
            observed = float(values[position])
            maximum = max(maximum, abs(reference - observed))
            if struct.pack("<d", observed) != reference_bytes:
                exact = False
    scale = float(np.max(np.abs(values), initial=0.0))
    tolerance_passed = missing == 0 and maximum <= (
        _PERMUTATION_ATOL + _PERMUTATION_RTOL * scale
    )
    return maximum, missing, tolerance_passed, exact


@dataclass(frozen=True, slots=True)
class _CompactAccumulationPreflight:
    """Conservative workspace model for the compact exact accumulator."""

    schema_version: str
    nonzero_contribution_count: int
    maximum_final_entry_count: int
    compact_payload_bytes: int
    stable_order_bytes: int
    stable_sort_scratch_bytes: int
    final_compact_upper_bound_bytes: int
    estimated_peak_bytes: int
    geometry_pass_count: int
    retained_periodic_record_count: int
    retained_python_contribution_record_count: int


def _compact_accumulation_preflight(
    nonzero_contribution_count: int,
    *,
    ndof: int,
) -> _CompactAccumulationPreflight:
    """Return a no-allocation upper bound for the accumulator workspace.

    The model covers the two compact contribution arrays, the stable-sort
    permutation and scratch space, compact group output, and final decoding.
    It intentionally does not estimate the immutable normalized FC3 input or
    the final returned vertex, which exist independently of accumulation.
    """

    count = int(nonzero_contribution_count)
    if count < 0:
        raise ValueError("nonzero_contribution_count must be nonnegative.")
    if ndof < 1:
        raise ValueError("ndof must be positive.")
    key_space = ndof**3
    if key_space - 1 > _UINT64_MAX:
        raise OverflowError(
            "The finite FC3 output key space exceeds deterministic uint64 "
            "accumulator addressing."
        )
    maximum_final = min(count, key_space)
    payload = count * (_UINT64_BYTES + _FLOAT64_BYTES)
    stable_order = count * _INDEX_BYTES
    # NumPy's stable indirect sort may require one index-sized scratch array.
    stable_scratch = count * _INDEX_BYTES
    final_compact = maximum_final * (_UINT64_BYTES + _FLOAT64_BYTES)
    sort_peak = payload + stable_order + stable_scratch
    accumulation_peak = payload + stable_order + final_compact
    # Final decoding overlaps the caller-owned compact payload and needs at
    # most compact keys/values, a three-column int64 result, a result-value
    # copy, and one uint64 ufunc temporary per final entry.
    finalization_peak = payload + maximum_final * (
        _UINT64_BYTES  # compact final keys
        + _FLOAT64_BYTES  # compact grouped values or the result-value copy
        + 3 * int(np.dtype(np.int64).itemsize)  # decoded result indices
        + _UINT64_BYTES  # one decoding ufunc temporary
    )
    return _CompactAccumulationPreflight(
        schema_version="negfpy.fc3-compact-accumulator-preflight.v1",
        nonzero_contribution_count=count,
        maximum_final_entry_count=maximum_final,
        compact_payload_bytes=payload,
        stable_order_bytes=stable_order,
        stable_sort_scratch_bytes=stable_scratch,
        final_compact_upper_bound_bytes=final_compact,
        estimated_peak_bytes=max(
            sort_peak,
            accumulation_peak,
            finalization_peak,
        ),
        geometry_pass_count=2,
        retained_periodic_record_count=0,
        retained_python_contribution_record_count=0,
    )


def _preflight_metadata(
    preflight: _CompactAccumulationPreflight,
) -> Mapping[str, Any]:
    return {
        name: getattr(preflight, name)
        for name in preflight.__dataclass_fields__
    }


def _weighted_nonzero_components(
    term: _EffectiveFC3Term,
) -> tuple[Array, Array]:
    """Return C-order component ordinals and legacy-ordered weighted values."""

    weighted = np.array(
        np.asarray(term.values, dtype=np.float64).reshape(-1),
        dtype=np.float64,
        order="C",
        copy=True,
    )
    # Keep the scalar mapper's exact multiplication order.  Combining these
    # factors would be mathematically equivalent but can change the last bit.
    weighted *= term.multiplicity_weight
    weighted *= term.anchor_weight
    weighted *= term.r0_leg_factor
    nonzero = np.flatnonzero(weighted != 0.0).astype(np.int64, copy=False)
    return nonzero, weighted[nonzero]


def _record_ordered_effective_terms(
    effective: Sequence[_EffectiveFC3Term],
) -> tuple[_EffectiveFC3Term, ...]:
    """Order terms as the legacy per-contribution record suffix did."""

    return tuple(
        sorted(
            effective,
            key=lambda term: (
                term.effective_leg,
                term.source_group,
                term.source_term_index,
                term.second_vector_index,
                term.third_vector_index,
            ),
        )
    )


def _accumulate_compact(
    output_keys: Array,
    contribution_values: Array,
    *,
    ndof: int,
) -> tuple[Array, Array, int, int]:
    """Stably group compact records and apply one ``math.fsum`` per output."""

    count = int(len(output_keys))
    if contribution_values.shape != (count,):
        raise ValueError("Compact contribution keys and values must align.")
    if count == 0:
        indices = np.empty((0, 3), dtype=np.int64)
        values = np.empty((0,), dtype=np.float64)
        indices.setflags(write=False)
        values.setflags(write=False)
        return indices, values, 0, 0

    # The fill stream is in the legacy record-key order.  A stable indirect
    # sort by output therefore reproduces the old per-output tuple sort while
    # retaining only 16 compact bytes per contribution.
    order = np.argsort(output_keys, kind="stable")
    group_count = 1
    previous = int(output_keys[int(order[0])])
    for raw_ordinal in order[1:]:
        observed = int(output_keys[int(raw_ordinal)])
        if observed != previous:
            group_count += 1
            previous = observed

    final_keys = np.empty(group_count, dtype=np.uint64)
    final_values = np.empty(group_count, dtype=np.float64)
    cursor = 0
    write = 0
    cancellations = 0
    while cursor < count:
        key = int(output_keys[int(order[cursor])])
        end = cursor + 1
        while end < count and int(output_keys[int(order[end])]) == key:
            end += 1
        value = math.fsum(
            float(contribution_values[int(raw_ordinal)])
            for raw_ordinal in order[cursor:end]
        )
        if value == 0.0:
            cancellations += 1
        else:
            final_keys[write] = np.uint64(key)
            final_values[write] = value
            write += 1
        cursor = end

    duplicate_count = count - group_count
    # Release the stable-order array before decoding the compact result.  The
    # preflight conservatively accounts for caller-owned compact inputs until
    # this function returns.
    del order
    result_values = np.array(final_values[:write], dtype=np.float64, copy=True)
    result_values[result_values == 0.0] = 0.0
    result_values.setflags(write=False)
    del final_values

    result_indices = np.empty((write, 3), dtype=np.int64)
    keys = final_keys[:write]
    ndof_squared = ndof * ndof
    result_indices[:, 0] = keys // np.uint64(ndof_squared)
    result_indices[:, 1] = (keys // np.uint64(ndof)) % np.uint64(ndof)
    result_indices[:, 2] = keys % np.uint64(ndof)
    result_indices.setflags(write=False)
    return result_indices, result_values, duplicate_count, cancellations


def _cubic_physics_sha256(
    ndof: int,
    indices: Array,
    values: Array,
) -> str:
    records = [b"negfpy-cubic-physics-v1\x00", struct.pack("<QQ", ndof, len(values))]
    for index, value in zip(indices, values, strict=True):
        number = 0.0 if float(value) == 0.0 else float(value)
        records.append(
            struct.pack(
                "<qqqd",
                int(index[0]),
                int(index[1]),
                int(index[2]),
                number,
            )
        )
    return hashlib.sha256(b"".join(records)).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class FC3DeviceMappingReport:
    """Immutable finite-crop audit and provenance join."""

    schema_version: str
    raw_term_count: int
    nonzero_raw_block_count: int
    exact_zero_raw_block_count: int
    effective_term_count: int
    nonzero_effective_term_count: int
    expanded_image_count: int
    expanded_cartesian_component_count: int
    exact_zero_contribution_count: int
    duplicate_accumulation_count: int
    exact_cancellation_count: int
    final_nonzero_count: int
    support_density: float
    pairwise_max_transport_cells: Array
    pairwise_max_principal_layers: Array
    pairwise_max_distance_m: Array
    required_left_halo_layers: int
    required_right_halo_layers: int
    n_device_layers: int
    contact_layers: tuple[int, int]
    harmonic_buffer_layers: tuple[tuple[int, ...], tuple[int, ...]]
    interaction_layers: tuple[int, ...]
    observation_layers: tuple[int, ...]
    permutation_max_abs: float
    permutation_missing_count: int
    permutation_tolerance_passed: bool
    exact_permutation_equality: bool
    permutation_atol: float
    permutation_rtol: float
    source_fc3_content_sha256: str
    source_translational_sha256: str
    mass_normalized_fc3_sha256: str
    source_certification_kind: CertificationKind
    source_certification_sha256: str
    primitive_crystal_content_sha256: str
    fc3_crystal_mapping_content_sha256: str
    fc2_content_sha256: str
    fc2_crystal_mapping_content_sha256: str
    layered_geometry_sha256: str
    cubic_physics_sha256: str
    symmetrized: bool
    corrected: bool
    truncated: bool
    thresholded: bool
    failure_codes: tuple[str, ...]
    certification_passed: bool
    report_sha256: str

    def __post_init__(self) -> None:
        for name, dtype in (
            ("pairwise_max_transport_cells", np.int64),
            ("pairwise_max_principal_layers", np.int64),
            ("pairwise_max_distance_m", np.float64),
        ):
            array = _readonly(
                getattr(self, name), dtype=dtype, name=name
            )
            if array.shape != (3,):
                raise ValueError(f"{name} must have shape (3,).")
            object.__setattr__(self, name, array)


def _report_payload(report: FC3DeviceMappingReport) -> Mapping[str, Any]:
    return {
        name: (
            digest_bytes(value)
            if name.endswith("_sha256") and name != "report_sha256"
            else value
        )
        for name, value in (
            (field, getattr(report, field))
            for field in report.__dataclass_fields__
            if field != "report_sha256"
        )
    }


@dataclass(frozen=True, slots=True, eq=False, init=False)
class DeviceMappedFC3:
    """Mass-normalized physical-SI cubic Hamiltonian on a finite device."""

    ndof: int
    indices: Array
    values: Array
    interaction_layers: tuple[int, ...]
    observation_layers: tuple[int, ...]
    report: FC3DeviceMappingReport
    metadata: Metadata
    cubic_physics_sha256: str
    content_sha256: str

    def __init__(self) -> None:
        raise TypeError(
            "DeviceMappedFC3 is builder-created; use map_fc3_to_device(...)."
        )

    @property
    def nnz(self) -> int:
        return int(self.values.size)

    @property
    def support_density(self) -> float:
        return float(self.nnz / self.ndof**3)


def _validate_mapping_identity(
    fc3: MassNormalizedTranslationalFC3,
    mapping: LayeredDeviceMapping,
) -> None:
    if not isinstance(mapping, LayeredDeviceMapping):
        raise TypeError("device_mapping must be a LayeredDeviceMapping.")
    if mapping.validation is None or not mapping.validation.certification_passed:
        raise ValueError("LayeredDeviceMapping must carry a passing audit.")
    if (
        primitive_crystal_content_sha256(mapping._crystal)
        != mapping.primitive_crystal_content_sha256
    ):
        raise ValueError("Embedded layered primitive identity is inconsistent.")
    if fc3.primitive_crystal_content_sha256 != mapping.primitive_crystal_content_sha256:
        raise ValueError("FC2 and FC3 primitive-crystal identities differ.")
    if not fc3.source_certification_sha256:
        raise ValueError("Normalized FC3 is missing its certification identity.")


def map_fc3_to_device(
    fc3: MassNormalizedTranslationalFC3,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: Sequence[int],
    observation_layers: Sequence[int],
) -> DeviceMappedFC3:
    """Map every effective FC3 image into an exact finite interaction crop."""

    if not isinstance(fc3, MassNormalizedTranslationalFC3):
        raise TypeError("fc3 must be a MassNormalizedTranslationalFC3.")
    _validate_mapping_identity(fc3, device_mapping)
    n_layers = _positive_integer(n_device_layers, name="n_device_layers")
    interactions = _layer_tuple(
        interaction_layers,
        name="interaction_layers",
        n_device_layers=n_layers,
    )
    observations = _layer_tuple(
        observation_layers,
        name="observation_layers",
        n_device_layers=n_layers,
    )
    if not set(observations).issubset(interactions):
        raise ValueError("observation_layers must be contained in interaction_layers.")

    effective = _effective_terms(fc3)
    principal_cells = device_mapping.principal_cells
    n_basis = device_mapping.n_transport_basis_atoms
    layer_dofs = device_mapping.n_principal_layer_dofs
    ndof = n_layers * layer_dofs
    allowed_min_cell = interactions[0] * principal_cells
    allowed_max_cell = (interactions[-1] + 1) * principal_cells - 1

    # Complete pre-crop ranges are measured before selecting a finite anchor.
    max_cells = np.zeros(3, dtype=np.int64)
    max_layers = np.zeros(3, dtype=np.int64)
    max_distances = np.zeros(3, dtype=np.float64)
    required_left = 0
    required_right = 0
    primitive_lattice = device_mapping._crystal.primitive.lattice
    expanded_images = 0
    expanded_components = 0
    exact_zero_contributions = 0
    nonzero_contributions = 0
    for term in effective:
        if not np.any(term.values != 0.0):
            continue
        component_ordinals, _ = _weighted_nonzero_components(term)
        component_count = int(len(component_ordinals))
        for representative in device_mapping.direct_coset_representatives:
            mapped = _mapped_geometry(term, representative, device_mapping)
            cells = [int(item[0][0]) for item in mapped]
            for pair_index, (left, right) in enumerate(_PAIRS):
                max_cells[pair_index] = max(
                    max_cells[pair_index], abs(cells[right] - cells[left])
                )
                physical_delta = term.positions[right] - term.positions[left]
                distance = float(
                    np.linalg.norm(physical_delta @ primitive_lattice * ANGSTROM_M)
                )
                max_distances[pair_index] = max(max_distances[pair_index], distance)
            for residue in range(principal_cells):
                layers = [
                    (cell + residue) // principal_cells for cell in cells
                ]
                for pair_index, (left, right) in enumerate(_PAIRS):
                    max_layers[pair_index] = max(
                        max_layers[pair_index],
                        abs(layers[right] - layers[left]),
                    )
                required_left = max(
                    required_left, max(layers[0] - value for value in layers)
                )
                required_right = max(
                    required_right, max(value - layers[0] for value in layers)
                )
            anchor_low = max(allowed_min_cell - value for value in cells)
            anchor_high = min(allowed_max_cell - value for value in cells)
            if anchor_low > anchor_high:
                continue
            image_count = anchor_high - anchor_low + 1
            expanded_images += image_count
            expanded_components += 27 * image_count
            exact_zero_contributions += (27 - component_count) * image_count
            nonzero_contributions += component_count * image_count

    workspace_preflight = _compact_accumulation_preflight(
        nonzero_contributions,
        ndof=ndof,
    )
    try:
        output_keys = np.empty(nonzero_contributions, dtype=np.uint64)
        contribution_values = np.empty(nonzero_contributions, dtype=np.float64)
    except (MemoryError, ValueError) as exc:
        raise MemoryError(
            "Finite FC3 compact accumulation requires at most "
            f"{workspace_preflight.estimated_peak_bytes} modeled workspace "
            "bytes for "
            f"{workspace_preflight.nonzero_contribution_count} nonzero "
            "contributions."
        ) from exc

    # The legacy record tuple sorted by effective leg, source group/source
    # ordinal, coset key, anchor, image ordinals, and Cartesian components.
    # Each source ordinal has at most one term for an effective leg, so this
    # loop emits that same suffix order without retaining the tuple itself.
    record_ordered = _record_ordered_effective_terms(effective)
    coset_ordinals = tuple(
        sorted(
            range(len(device_mapping.direct_coset_representatives)),
            key=lambda ordinal: tuple(
                int(value)
                for value in device_mapping.direct_coset_keys[ordinal]
            ),
        )
    )
    write = 0
    for term in record_ordered:
        if not np.any(term.values != 0.0):
            continue
        component_ordinals, weighted = _weighted_nonzero_components(term)
        if len(component_ordinals) == 0:
            continue
        alpha = component_ordinals // 9
        beta = (component_ordinals // 3) % 3
        gamma = component_ordinals % 3
        for coset_ordinal in coset_ordinals:
            representative = device_mapping.direct_coset_representatives[
                coset_ordinal
            ]
            mapped = _mapped_geometry(term, representative, device_mapping)
            base_cells = [int(item[0][0]) for item in mapped]
            anchor_low = max(
                allowed_min_cell - value for value in base_cells
            )
            anchor_high = min(
                allowed_max_cell - value for value in base_cells
            )
            if anchor_low > anchor_high:
                continue
            for longitudinal_anchor in range(anchor_low, anchor_high + 1):
                output_bases: list[int] = []
                output_layers: list[int] = []
                for cell_basis in mapped:
                    cell = int(cell_basis[0][0]) + longitudinal_anchor
                    layer = cell // principal_cells
                    subcell = cell % principal_cells
                    basis = int(cell_basis[1])
                    output_layers.append(layer)
                    output_bases.append(
                        layer * layer_dofs
                        + 3 * (subcell * n_basis + basis)
                    )
                if any(layer not in interactions for layer in output_layers):
                    raise RuntimeError(
                        "Finite FC3 anchor enumeration escaped its crop."
                    )

                stop = write + len(component_ordinals)
                first = output_bases[0] + alpha
                second = output_bases[1] + beta
                third = output_bases[2] + gamma
                output_keys[write:stop] = (
                    (
                        np.asarray(first, dtype=np.uint64) * np.uint64(ndof)
                        + np.asarray(second, dtype=np.uint64)
                    )
                    * np.uint64(ndof)
                    + np.asarray(third, dtype=np.uint64)
                )
                contribution_values[write:stop] = weighted
                write = stop
    if write != nonzero_contributions:
        raise RuntimeError(
            "Finite FC3 compact preflight and fill contribution counts differ."
        )

    try:
        indices, values, duplicate_count, cancellation_count = (
            _accumulate_compact(
                output_keys,
                contribution_values,
                ndof=ndof,
            )
        )
    except MemoryError as exc:
        raise MemoryError(
            "Finite FC3 stable compact accumulation could not allocate its "
            f"modeled {workspace_preflight.estimated_peak_bytes}-byte peak "
            "workspace."
        ) from exc
    del output_keys, contribution_values
    physics_hash = _cubic_physics_sha256(ndof, indices, values)
    permutation_max, permutation_missing, permutation_passed, exact_permutation = (
        _permutation_audit(ndof, indices, values)
    )

    # The explicit contact device layers are harmonic-only, noninteracting
    # layers and therefore count toward the FC3 range halo.
    left_buffer = tuple(range(0, interactions[0]))
    right_buffer = tuple(range(interactions[-1] + 1, n_layers))
    checks = {
        "left_harmonic_buffer": len(left_buffer) >= required_left,
        "right_harmonic_buffer": len(right_buffer) >= required_right,
        "observation_left_padding": (
            observations[0] - interactions[0] >= required_left
        ),
        "observation_right_padding": (
            interactions[-1] - observations[-1] >= required_right
        ),
        "permutation_tolerance": permutation_passed,
        "exact_permutation_equality": exact_permutation,
    }
    failures = tuple(sorted(name for name, passed in checks.items() if not passed))

    raw_blocks: dict[tuple[Any, ...], bool] = {}
    for term in fc3.terms:
        key = (term.primitive_indices, term.source_indices)
        nonzero = bool(np.any(term.values != 0.0))
        if key in raw_blocks and raw_blocks[key] != nonzero:
            raise ValueError(
                "Shortest images of one raw FC3 block disagree on exact-zero "
                "support."
            )
        raw_blocks[key] = nonzero

    provisional = FC3DeviceMappingReport(
        schema_version="negfpy.fc3-device-mapping.v1",
        raw_term_count=len(fc3.terms),
        nonzero_raw_block_count=sum(raw_blocks.values()),
        exact_zero_raw_block_count=len(raw_blocks) - sum(raw_blocks.values()),
        effective_term_count=len(effective),
        nonzero_effective_term_count=sum(
            bool(np.any(term.values != 0.0)) for term in effective
        ),
        expanded_image_count=expanded_images,
        expanded_cartesian_component_count=expanded_components,
        exact_zero_contribution_count=exact_zero_contributions,
        duplicate_accumulation_count=duplicate_count,
        exact_cancellation_count=cancellation_count,
        final_nonzero_count=len(values),
        support_density=float(len(values) / ndof**3),
        pairwise_max_transport_cells=max_cells,
        pairwise_max_principal_layers=max_layers,
        pairwise_max_distance_m=max_distances,
        required_left_halo_layers=required_left,
        required_right_halo_layers=required_right,
        n_device_layers=n_layers,
        contact_layers=(0, n_layers - 1),
        harmonic_buffer_layers=(left_buffer, right_buffer),
        interaction_layers=interactions,
        observation_layers=observations,
        permutation_max_abs=permutation_max,
        permutation_missing_count=permutation_missing,
        permutation_tolerance_passed=permutation_passed,
        exact_permutation_equality=exact_permutation,
        permutation_atol=_PERMUTATION_ATOL,
        permutation_rtol=_PERMUTATION_RTOL,
        source_fc3_content_sha256=fc3.source_fc3_content_sha256,
        source_translational_sha256=fc3.source_translational_sha256,
        mass_normalized_fc3_sha256=fc3.content_sha256,
        source_certification_kind=fc3.source_certification_kind,
        source_certification_sha256=fc3.source_certification_sha256,
        primitive_crystal_content_sha256=fc3.primitive_crystal_content_sha256,
        fc3_crystal_mapping_content_sha256=(
            fc3.fc3_crystal_mapping_content_sha256
        ),
        fc2_content_sha256=device_mapping.fc2_content_sha256,
        fc2_crystal_mapping_content_sha256=(
            device_mapping.fc2_crystal_mapping_content_sha256
        ),
        layered_geometry_sha256=device_mapping.layered_geometry_sha256,
        cubic_physics_sha256=physics_hash,
        symmetrized=False,
        corrected=False,
        truncated=False,
        thresholded=False,
        failure_codes=failures,
        certification_passed=not failures,
        report_sha256="",
    )
    report_hash = domain_digest(
        "NEGFPY-FC3-DEVICE-MAPPING", _report_payload(provisional)
    )
    object.__setattr__(provisional, "report_sha256", report_hash)

    result = object.__new__(DeviceMappedFC3)
    metadata = _freeze_mapping(
        {
            "schema_version": "negfpy.device-mapped-fc3.v1",
            "output_units": fc3.output_units,
            "weights_applied": True,
            "solver_scaled": False,
            "symmetrized": False,
            "corrected": False,
            "truncated": False,
            "thresholded": False,
            "mapping_report_sha256": report_hash,
            "compact_accumulation_preflight": _preflight_metadata(
                workspace_preflight
            ),
        }
    )
    content_hash = domain_digest(
        "NEGFPY-DEVICE-MAPPED-FC3",
        {
            "ndof": ndof,
            "indices": indices,
            "values": values,
            "interaction_layers": interactions,
            "observation_layers": observations,
            "report_sha256": digest_bytes(report_hash),
        },
    )
    for name, value in {
        "ndof": ndof,
        "indices": indices,
        "values": values,
        "interaction_layers": interactions,
        "observation_layers": observations,
        "report": provisional,
        "metadata": metadata,
        "cubic_physics_sha256": physics_hash,
        "content_sha256": content_hash,
    }.items():
        object.__setattr__(result, name, value)
    return result


def scale_device_fc3(
    mapped: DeviceMappedFC3,
    scale: SolverScale,
) -> Any:
    """Scale physical FC3 into a late-bound :class:`SparseCubicVertex`.

    The late import keeps Phase B independently testable while the Phase-C
    sparse container is under development.  It never falls back to a dense
    allocation.
    """

    if not isinstance(mapped, DeviceMappedFC3):
        raise TypeError("mapped must be a DeviceMappedFC3.")
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    if not mapped.report.certification_passed:
        raise ValueError(
            "DeviceMappedFC3 cannot be scaled because its mapping report "
            f"failed: {mapped.report.failure_codes!r}."
        )
    try:
        from .sparse_cubic import SparseCubicVertex
    except ImportError as exc:  # pragma: no cover - Phase-C integration boundary
        raise RuntimeError(
            "SparseCubicVertex is unavailable; complete Phase C before "
            "solver scaling."
        ) from exc

    factor = math.sqrt(scale.hbar) / scale.omega0**2.5
    values = np.asarray(mapped.values, dtype=np.float64) * factor
    metadata = {
        **dict(mapped.metadata),
        "schema_version": "negfpy.sparse-cubic-vertex.material.v1",
        "device_mapped_fc3_sha256": mapped.content_sha256,
        "mapping_report_sha256": mapped.report.report_sha256,
        "cubic_physics_sha256_physical": mapped.cubic_physics_sha256,
        "fc2_content_sha256": mapped.report.fc2_content_sha256,
        "primitive_crystal_content_sha256": (
            mapped.report.primitive_crystal_content_sha256
        ),
        "fc2_crystal_mapping_content_sha256": (
            mapped.report.fc2_crystal_mapping_content_sha256
        ),
        "fc3_crystal_mapping_content_sha256": (
            mapped.report.fc3_crystal_mapping_content_sha256
        ),
        "layered_geometry_sha256": mapped.report.layered_geometry_sha256,
        "source_fc3_content_sha256": mapped.report.source_fc3_content_sha256,
        "source_certification_kind": mapped.report.source_certification_kind,
        "source_certification_sha256": (
            mapped.report.source_certification_sha256
        ),
        "solver_scale_omega0": scale.omega0,
        "solver_scale_hbar": scale.hbar,
        "cubic_vertex_to_solver_factor": factor,
        "solver_scaled": True,
    }
    return SparseCubicVertex(
        ndof=mapped.ndof,
        indices=mapped.indices,
        coefficients=values,
        metadata=metadata,
    )


__all__ = [
    "DeviceMappedFC3",
    "FC3DeviceMappingReport",
    "MassNormalizedTranslationalFC3",
    "map_fc3_to_device",
    "mass_normalize_translational_fc3",
    "scale_device_fc3",
]
