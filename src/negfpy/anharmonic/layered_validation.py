"""Validation reports for the Phase-A layered material harmonic oracle.

The public builders live in :mod:`negfpy.anharmonic.layered`.  This module is
kept separate so validation is recomputed at construction, serialization, and
release boundaries instead of becoming an unchecked flag stored by a caller.
No routine in this module corrects force constants or an assembled operator.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._canonical import domain_digest


Array = np.ndarray

_GEOMETRY_SCHEMA = "negfpy.layered-geometry-validation.v1"
_HARMONIC_SCHEMA = "negfpy.layered-harmonic-validation.v1"
_COMPARISON_SCHEMA = "negfpy.layered-harmonic-comparison.v1"

_GEOMETRY_TOLERANCES = {
    "integer_matrix_atol": 1.0e-12,
    "fractional_snap_atol": 1.0e-10,
    "fractional_match_atol": 5.0e-10,
    "cartesian_match_atol_m": 1.0e-12,
    "mass_match_rtol": 1.0e-12,
    "mass_match_atol_kg": 0.0,
    "direct_reciprocal_phase_atol": 1.0e-13,
}

_HARMONIC_TOLERANCES = {
    "coordinate_roundtrip_atol": 1.0e-12,
    "direct_reciprocal_phase_atol": 1.0e-13,
    "folding_unitarity_atol": 1.0e-13,
    "operator_rtol": 1.0e-10,
    "operator_atol_solver": 1.0e-12,
    "reciprocal_matrix_rtol": 1.0e-10,
    "reciprocal_matrix_atol_scaled": 1.0e-12,
    "asr_normalized_max": 1.0e-10,
    "folding_matrix_rtol": 1.0e-10,
    "folding_matrix_atol_solver": 1.0e-12,
    "same_convention_frequency_rtol": 1.0e-10,
    "same_convention_frequency_atol_thz": 1.0e-8,
    "omega0_invariance_rtol": 1.0e-10,
    "omega0_invariance_atol_scaled": 1.0e-12,
    "omega0_invariance_atol_physical_s_minus2": 1.5791367041742972e16,
    "redundant_rows_rtol": 1.0e-10,
    "redundant_rows_atol_source": 1.0e-12,
    "stability_eigenvalue_floor_relative": -1.0e-10,
    "derived_constant_max_ulps": 2,
    "relative_denominator_policy": (
        "max_abs_pair_then_comparison_atol_then_float64_tiny"
    ),
}

_COMPARISON_TOLERANCES = {
    key: _HARMONIC_TOLERANCES[key]
    for key in (
        "operator_rtol",
        "operator_atol_solver",
        "reciprocal_matrix_rtol",
        "reciprocal_matrix_atol_scaled",
        "same_convention_frequency_rtol",
        "same_convention_frequency_atol_thz",
        "omega0_invariance_rtol",
        "omega0_invariance_atol_scaled",
        "omega0_invariance_atol_physical_s_minus2",
        "relative_denominator_policy",
    )
}

_UNIT_CONVENTIONS = {
    "si_convention": "negfpy-si-codata-2018-v1",
    "electron_volt_j": 1.602176634e-19,
    "angstrom_m": 1.0e-10,
    "atomic_mass_unit_kg": 1.66053906660e-27,
    "hbar_j_s": 1.0545718176461565e-34,
    "frequency_factor_si_thz": 15.633304239856193,
    "phono3py_convention": "phono3py-4.4-compatibility-v1",
    "frequency_factor_phono3py_4_4_thz": 15.633302300230191,
    "si_to_phono3py_ratio": 1.0000001240701397,
}


def _freeze_mapping(values: Mapping[str, Any]) -> Mapping[str, Any]:
    frozen: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, Mapping):
            frozen[str(key)] = _freeze_mapping(value)
        elif isinstance(value, np.ndarray):
            array = np.array(value, copy=True, order="C")
            array.setflags(write=False)
            frozen[str(key)] = array
        elif isinstance(value, list):
            frozen[str(key)] = tuple(value)
        else:
            frozen[str(key)] = value
    return MappingProxyType(frozen)


def _report_digest(domain: str, payload: Mapping[str, Any]) -> str:
    return domain_digest(domain, payload)


def _max_abs(values: Any) -> float:
    array = np.asarray(values)
    return float(np.max(np.abs(array))) if array.size else 0.0


def _pair_metrics(left: Any, right: Any, *, atol: float) -> tuple[float, float]:
    a = np.asarray(left)
    b = np.asarray(right)
    absolute = _max_abs(a - b)
    denominator = np.maximum(np.maximum(np.abs(a), np.abs(b)), atol)
    if atol == 0.0:
        denominator = np.maximum(denominator, np.finfo(np.float64).tiny)
    relative = float(np.max(np.abs(a - b) / denominator)) if a.size else 0.0
    return absolute, relative


def _within(left: Any, right: Any, *, rtol: float, atol: float) -> bool:
    a = np.asarray(left)
    b = np.asarray(right)
    return bool(np.all(np.abs(a - b) <= atol + rtol * np.maximum(np.abs(a), np.abs(b))))


def _qset(name: str) -> Array:
    if name == "primitive_reciprocal_v1":
        fixed = [
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
        import hashlib

        for counter in range(32):
            digest = hashlib.sha256(
                b"NEGFPY-PHASE-A-QPOINT\x00"
                + (1).to_bytes(2, "little")
                + seed.to_bytes(8, "little")
                + counter.to_bytes(8, "little")
            ).digest()
            point = []
            for axis in range(3):
                integer = int.from_bytes(digest[8 * axis : 8 * axis + 8], "little")
                point.append(((integer >> 11) + 0.5) / 2**53 - 0.5)
            fixed.append(tuple(point))
        return np.asarray(fixed, dtype=np.float64)
    if name == "longitudinal_folding_v1":
        return np.asarray(
            [(n / 65.0, 0.0, 0.0) for n in range(-32, 33)]
            + [(0.5, 0.0, 0.0)],
            dtype=np.float64,
        )
    if name == "longitudinal_stability_v1":
        return np.asarray(
            [(n / 4096.0, 0.0, 0.0) for n in range(-2048, 2048)],
            dtype=np.float64,
        )
    raise ValueError(f"Unknown Phase-A q-set {name!r}.")


def _qset_sha256(name: str) -> str:
    if name == "primitive_reciprocal_v1":
        generator, seed = "explicit9_sha256_counter32_v1", 0x4E45474650593033
    elif name == "longitudinal_folding_v1":
        generator, seed = "rational_n_over_65_plus_half_v1", None
    elif name == "longitudinal_stability_v1":
        generator, seed = "rational_n_over_4096_periodic_v1", None
    else:
        raise ValueError(f"Unknown Phase-A q-set {name!r}.")
    return domain_digest(
        "NEGFPY-PHASE-A-QSET",
        {"name": name, "generator": generator, "seed": seed, "qpoints": _qset(name)},
    )


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LayeredGeometryValidationReport:
    schema_version: str
    fc2_content_sha256: str
    fc2_source_sha256: str
    fc2_provenance_sha256: str
    fc2_layout: str
    oracle_v0_2_fc2_mapping_sha256: str | None
    primitive_crystal_content_sha256: str
    fc2_crystal_mapping_content_sha256: str
    direct_cosets_sha256: str
    reciprocal_cosets_sha256: str
    layered_geometry_sha256: str
    checks: Mapping[str, bool]
    counts: Mapping[str, int]
    residuals: Mapping[str, float]
    tolerances: Mapping[str, Any]
    failure_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    certification_passed: bool
    report_sha256: str

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "LayeredGeometryValidationReport":
        frozen = dict(payload)
        frozen["checks"] = _freeze_mapping(frozen["checks"])
        frozen["counts"] = _freeze_mapping(frozen["counts"])
        frozen["residuals"] = _freeze_mapping(frozen["residuals"])
        frozen["tolerances"] = _freeze_mapping(frozen["tolerances"])
        obj = object.__new__(cls)
        for key, value in frozen.items():
            object.__setattr__(obj, key, value)
        object.__setattr__(obj, "report_sha256", _report_digest("NEGFPY-LAYERED-GEOMETRY-VALIDATION", payload))
        return obj


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LayeredHarmonicValidationReport:
    schema_version: str
    fc2_content_sha256: str
    fc2_source_sha256: str
    fc2_provenance_sha256: str
    fc2_layout: str
    oracle_v0_2_fc2_mapping_sha256: str | None
    primitive_crystal_content_sha256: str
    fc2_crystal_mapping_content_sha256: str
    layered_geometry_sha256: str
    model_physics_sha256: str
    primitive_qset_sha256: str
    longitudinal_qset_sha256: str
    stability_qset_sha256: str
    unit_conventions: Mapping[str, Any]
    d00_audit: Array
    d01_audit: Array
    d10_audit: Array
    d00_active: Array
    d01_active: Array
    d10_active: Array
    checks: Mapping[str, bool]
    residuals: Mapping[str, float]
    tolerances: Mapping[str, Any]
    failure_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    certification_passed: bool
    report_sha256: str

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "LayeredHarmonicValidationReport":
        frozen = dict(payload)
        for key in ("unit_conventions", "checks", "residuals", "tolerances"):
            frozen[key] = _freeze_mapping(frozen[key])
        for key in ("d00_audit", "d01_audit", "d10_audit", "d00_active", "d01_active", "d10_active"):
            array = np.array(frozen[key], dtype=np.complex128, copy=True, order="C")
            array.setflags(write=False)
            frozen[key] = array
        obj = object.__new__(cls)
        for key, value in frozen.items():
            object.__setattr__(obj, key, value)
        object.__setattr__(obj, "report_sha256", _report_digest("NEGFPY-LAYERED-HARMONIC-VALIDATION", payload))
        return obj


@dataclass(frozen=True, slots=True, eq=False, init=False)
class LayeredHarmonicComparisonReport:
    schema_version: str
    comparison: str
    left_model_physics_sha256: str
    right_model_physics_sha256: str
    common_primitive_crystal_content_sha256: str
    common_layered_geometry_sha256: str
    left_fc2_content_sha256: str
    right_fc2_content_sha256: str
    qset_sha256: str
    full_compact_metrics_or_null: Mapping[str, Any] | None
    omega0_metrics_or_null: Mapping[str, Any] | None
    tolerances: Mapping[str, Any]
    failure_codes: tuple[str, ...]
    warning_codes: tuple[str, ...]
    certification_passed: bool
    report_sha256: str

    @classmethod
    def _from_payload(cls, payload: Mapping[str, Any]) -> "LayeredHarmonicComparisonReport":
        frozen = dict(payload)
        for key in ("full_compact_metrics_or_null", "omega0_metrics_or_null", "tolerances"):
            if frozen[key] is not None:
                frozen[key] = _freeze_mapping(frozen[key])
        obj = object.__new__(cls)
        for key, value in frozen.items():
            object.__setattr__(obj, key, value)
        object.__setattr__(obj, "report_sha256", _report_digest("NEGFPY-LAYERED-HARMONIC-COMPARISON", payload))
        return obj


def validate_layered_device_mapping(mapping: Any) -> LayeredGeometryValidationReport:
    """Recompute the fail-closed geometry report for one layered mapping."""

    from .layered import LayeredDeviceMapping

    if not isinstance(mapping, LayeredDeviceMapping):
        raise TypeError("mapping must be a LayeredDeviceMapping.")
    metrics = dict(getattr(mapping, "_geometry_metrics", {}))
    offsets = np.asarray(mapping.layer_offsets, dtype=np.int64)
    support = np.asarray(mapping.longitudinal_fc2_support, dtype=np.int64)
    determinant_abs = abs(int(mapping.signed_determinant))
    principal_cells = int(mapping.principal_cells)
    checks = {
        "fc2_schema": bool(metrics.get("fc2_schema", False)),
        "crystal_schema": bool(metrics.get("crystal_schema", False)),
        "source_identity": bool(metrics.get("source_identity", False)),
        "integer_transform": bool(metrics.get("integer_transform", False)),
        "direct_cosets": bool(metrics.get("direct_cosets", False)),
        "reciprocal_cosets": bool(metrics.get("reciprocal_cosets", False)),
        "coordinate_roundtrip": bool(metrics.get("coordinate_roundtrip", False)),
        "atom_bijection": bool(metrics.get("atom_bijection", False)),
        "mass_match": bool(metrics.get("mass_match", False)),
        "support_complete": bool(metrics.get("support_complete", False)),
        "principal_layer_minimal": bool(metrics.get("principal_layer_minimal", False)),
        "nearest_layer_range": bool(metrics.get("nearest_layer_range", False)),
        "direct_reciprocal_phase": bool(metrics.get("direct_reciprocal_phase", False)),
    }
    counts = {
        "n_primitive_atoms": int(mapping._crystal.n_primitive_atoms),
        "n_supercell_atoms": int(mapping._crystal.n_supercell_atoms),
        "determinant_abs": determinant_abs,
        "n_direct_cosets": int(len(mapping.direct_coset_keys)),
        "n_reciprocal_cosets": int(len(mapping.reciprocal_coset_keys)),
        "n_transport_basis_atoms": int(len(mapping.basis_primitive_atoms)),
        "principal_cells": principal_cells,
        "n_principal_layer_dofs": int(len(mapping.principal_layer_dof_records)),
        "n_fc2_image_records": int(len(mapping._terms)),
        "n_fc2_support_records": int(len(support)),
        "n_layer_offsets": int(len(offsets)),
        "n_redundant_first_axis_rows": int(metrics.get("n_redundant_first_axis_rows", 0)),
    }
    residuals = {
        "integer_matrix_max_abs": float(metrics.get("integer_matrix_max_abs", 0.0)),
        "coordinate_roundtrip_max_abs": float(metrics.get("coordinate_roundtrip_max_abs", 0.0)),
        "atom_fractional_match_max_abs": float(metrics.get("atom_fractional_match_max_abs", 0.0)),
        "atom_cartesian_match_max_m": float(metrics.get("atom_cartesian_match_max_m", 0.0)),
        "mass_match_max_relative": float(metrics.get("mass_match_max_relative", 0.0)),
        "direct_quotient_max_integer_residual": float(metrics.get("direct_quotient_max_integer_residual", 0.0)),
        "reciprocal_quotient_max_integer_residual": float(metrics.get("reciprocal_quotient_max_integer_residual", 0.0)),
        "direct_reciprocal_phase_max_abs": float(metrics.get("direct_reciprocal_phase_max_abs", 0.0)),
    }
    failure_codes = tuple(sorted(key for key, passed in checks.items() if not passed))
    payload = {
        "schema_version": _GEOMETRY_SCHEMA,
        "fc2_content_sha256": mapping.fc2_content_sha256,
        "fc2_source_sha256": mapping.fc2_source_sha256,
        "fc2_provenance_sha256": mapping.fc2_provenance_sha256,
        "fc2_layout": mapping.fc2_layout,
        "oracle_v0_2_fc2_mapping_sha256": mapping.oracle_v0_2_fc2_mapping_sha256,
        "primitive_crystal_content_sha256": mapping.primitive_crystal_content_sha256,
        "fc2_crystal_mapping_content_sha256": mapping.fc2_crystal_mapping_content_sha256,
        "direct_cosets_sha256": mapping.direct_cosets_sha256,
        "reciprocal_cosets_sha256": mapping.reciprocal_cosets_sha256,
        "layered_geometry_sha256": mapping.layered_geometry_sha256,
        "checks": checks,
        "counts": counts,
        "residuals": residuals,
        "tolerances": dict(_GEOMETRY_TOLERANCES),
        "failure_codes": failure_codes,
        "warning_codes": (),
        "certification_passed": not failure_codes and all(checks.values()),
    }
    return LayeredGeometryValidationReport._from_payload(payload)


def _layered_matrix(model: Any, q: float) -> Array:
    phase = np.exp(2j * np.pi * q)
    return model.d00 + model.d01 * phase + model.d10 / phase


def validate_layered_harmonic_model(model: Any) -> LayeredHarmonicValidationReport:
    """Recompute the fixed Phase-A harmonic validation report."""

    from .layered import LayeredHarmonicModel

    if not isinstance(model, LayeredHarmonicModel):
        raise TypeError("model must be a LayeredHarmonicModel.")
    mapping_report = validate_layered_device_mapping(model.mapping)
    metrics = dict(getattr(model, "_audit_metrics", {}))
    d00_audit = np.asarray(model._d00_audit, dtype=np.complex128)
    d01_audit = np.asarray(model._d01_audit, dtype=np.complex128)
    d10_audit = np.asarray(model._d10_audit, dtype=np.complex128)
    onsite_residual = _max_abs(d00_audit - d00_audit.conj().T)
    reverse_residual = _max_abs(d10_audit - d01_audit.conj().T)
    canonical_delta = max(
        _max_abs(model.d00 - d00_audit),
        _max_abs(model.d01 - d01_audit),
        _max_abs(model.d10 - d10_audit),
    )
    eig_min = np.inf
    eig_max = 0.0
    for point in _qset("longitudinal_stability_v1")[:, 0]:
        eigenvalues = np.linalg.eigvalsh((_layered_matrix(model, float(point)) + _layered_matrix(model, float(point)).conj().T) / 2.0)
        eig_min = min(eig_min, float(np.min(eigenvalues)))
        eig_max = max(eig_max, float(np.max(np.abs(eigenvalues))))
    residuals = {
        "redundant_rows_max_abs_source": float(metrics.get("redundant_rows_max_abs_source", 0.0)),
        "reciprocity_max_abs_source": float(metrics.get("reciprocity_max_abs_source", 0.0)),
        "asr_max_abs_source": float(metrics.get("asr_max_abs_source", 0.0)),
        "asr_normalized": float(metrics.get("asr_normalized", 0.0)),
        "raw_onsite_hermiticity_max_abs_solver": onsite_residual,
        "raw_forward_reverse_max_abs_solver": reverse_residual,
        "canonicalization_delta_max_abs_solver": canonical_delta,
        "minimum_omega2_source": float(metrics.get("minimum_omega2_source", eig_min * model.scale.omega0**2)),
        "maximum_abs_omega2_source": float(metrics.get("maximum_abs_omega2_source", eig_max * model.scale.omega0**2)),
        "reciprocal_matrix_max_abs_source": float(metrics.get("reciprocal_matrix_max_abs_source", 0.0)),
        "reciprocal_matrix_max_relative": float(metrics.get("reciprocal_matrix_max_relative", 0.0)),
        "folding_unitarity_left_max_abs": float(metrics.get("folding_unitarity_left_max_abs", 0.0)),
        "folding_unitarity_right_max_abs": float(metrics.get("folding_unitarity_right_max_abs", 0.0)),
        "folding_forward_max_abs_solver": float(metrics.get("folding_forward_max_abs_solver", 0.0)),
        "folding_forward_max_relative": float(metrics.get("folding_forward_max_relative", 0.0)),
        "folding_inverse_max_abs_source": float(metrics.get("folding_inverse_max_abs_source", 0.0)),
        "folding_inverse_max_relative": float(metrics.get("folding_inverse_max_relative", 0.0)),
    }
    scale = max(_max_abs(d00_audit), _max_abs(d01_audit), _max_abs(d10_audit), np.finfo(float).tiny)
    op_allowed = _HARMONIC_TOLERANCES["operator_atol_solver"] + _HARMONIC_TOLERANCES["operator_rtol"] * scale
    stable_floor = _HARMONIC_TOLERANCES["stability_eigenvalue_floor_relative"] * residuals["maximum_abs_omega2_source"]
    checks = {
        "mapping_join": mapping_report.certification_passed,
        "full_redundant_rows": bool(metrics.get("full_redundant_rows", False)),
        "mass_conversion": bool(metrics.get("mass_conversion", False)),
        "reciprocity": bool(metrics.get("reciprocity_passed", False)),
        "asr": bool(metrics.get("asr_passed", False)),
        "range": bool(np.max(np.abs(model.mapping.layer_offsets)) <= 1),
        "raw_onsite_hermiticity": onsite_residual <= op_allowed,
        "raw_forward_reverse": reverse_residual <= op_allowed,
        "operator_canonicalization": canonical_delta <= op_allowed,
        "stability": bool(metrics.get("stability_passed", residuals["minimum_omega2_source"] >= stable_floor)),
        "reciprocal_reconstruction": bool(metrics.get("reciprocal_reconstruction_passed", False)),
        "folding_unitarity": bool(metrics.get("folding_unitarity_passed", False)),
        "folding_forward": bool(metrics.get("folding_forward_passed", False)),
        "folding_inverse": bool(metrics.get("folding_inverse_passed", False)),
    }
    failure_codes = tuple(sorted(key for key, passed in checks.items() if not passed))
    payload = {
        "schema_version": _HARMONIC_SCHEMA,
        "fc2_content_sha256": model.fc2_content_sha256,
        "fc2_source_sha256": model.fc2_source_sha256,
        "fc2_provenance_sha256": model.fc2_provenance_sha256,
        "fc2_layout": model.fc2_layout,
        "oracle_v0_2_fc2_mapping_sha256": model.oracle_v0_2_fc2_mapping_sha256,
        "primitive_crystal_content_sha256": model.primitive_crystal_content_sha256,
        "fc2_crystal_mapping_content_sha256": model.fc2_crystal_mapping_content_sha256,
        "layered_geometry_sha256": model.layered_geometry_sha256,
        "model_physics_sha256": model.model_physics_sha256,
        "primitive_qset_sha256": _qset_sha256("primitive_reciprocal_v1"),
        "longitudinal_qset_sha256": _qset_sha256("longitudinal_folding_v1"),
        "stability_qset_sha256": _qset_sha256("longitudinal_stability_v1"),
        "unit_conventions": dict(_UNIT_CONVENTIONS),
        "d00_audit": d00_audit,
        "d01_audit": d01_audit,
        "d10_audit": d10_audit,
        "d00_active": model.d00,
        "d01_active": model.d01,
        "d10_active": model.d10,
        "checks": checks,
        "residuals": residuals,
        "tolerances": dict(_HARMONIC_TOLERANCES),
        "failure_codes": failure_codes,
        "warning_codes": (),
        "certification_passed": not failure_codes and all(checks.values()),
    }
    return LayeredHarmonicValidationReport._from_payload(payload)


def _frequencies_thz(model: Any) -> Array:
    values: list[Array] = []
    for q in _qset("longitudinal_folding_v1")[:, 0]:
        eig = np.linalg.eigvalsh((_layered_matrix(model, float(q)) + _layered_matrix(model, float(q)).conj().T) / 2.0)
        angular = np.sign(eig) * np.sqrt(np.abs(eig)) * model.scale.omega0
        values.append(angular / (2.0 * np.pi * 1.0e12))
    return np.asarray(values)


def _longitudinal_eigenvalues_solver(model: Any) -> Array:
    values: list[Array] = []
    for q in _qset("longitudinal_folding_v1")[:, 0]:
        matrix = _layered_matrix(model, float(q))
        values.append(np.linalg.eigvalsh((matrix + matrix.conj().T) / 2.0))
    return np.asarray(values)


def _cross_reciprocal_metrics(left: Any, right: Any) -> tuple[float, float, bool]:
    """Compare full/compact source matrices on the frozen primitive q-set."""

    from .bulk_harmonic import dynamical_matrix_from_fc2

    left_fc2 = left.mapping._canonical_fc2.as_fc2()
    right_fc2 = right.mapping._canonical_fc2.as_fc2()
    maximum_absolute = 0.0
    maximum_relative = 0.0
    passed = True
    for qpoint in _qset("primitive_reciprocal_v1"):
        a = dynamical_matrix_from_fc2(
            left_fc2,
            left.mapping._crystal,
            qpoint,
            hermitianize=False,
        )
        b = dynamical_matrix_from_fc2(
            right_fc2,
            right.mapping._crystal,
            qpoint,
            hermitianize=False,
        )
        shared = max(_max_abs(a), _max_abs(b), np.finfo(np.float64).tiny)
        a_scaled = a / shared
        b_scaled = b / shared
        absolute = _max_abs(a - b)
        _, relative = _pair_metrics(
            a_scaled,
            b_scaled,
            atol=_HARMONIC_TOLERANCES["reciprocal_matrix_atol_scaled"],
        )
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
        passed &= _within(
            a_scaled,
            b_scaled,
            rtol=_HARMONIC_TOLERANCES["reciprocal_matrix_rtol"],
            atol=_HARMONIC_TOLERANCES["reciprocal_matrix_atol_scaled"],
        )
    return maximum_absolute, maximum_relative, bool(passed)


def _full_compact_frequency_metrics(
    left: Any,
    right: Any,
) -> tuple[Mapping[str, Any], bool]:
    """Apply the operator-derived near-acoustic comparison correction."""

    left_lambda = _longitudinal_eigenvalues_solver(left)
    right_lambda = _longitudinal_eigenvalues_solver(right)
    left_frequency = _frequencies_thz(left)
    right_frequency = _frequencies_thz(right)
    crossover = (
        np.sqrt(_HARMONIC_TOLERANCES["operator_atol_solver"])
        * max(left.scale.omega0, right.scale.omega0)
        / (2.0 * np.pi * 1.0e12)
    )
    acoustic = (np.abs(left_frequency) <= crossover) | (
        np.abs(right_frequency) <= crossover
    )
    lambda_difference = np.abs(left_lambda - right_lambda)
    lambda_allowed = _HARMONIC_TOLERANCES["operator_atol_solver"] + _HARMONIC_TOLERANCES[
        "operator_rtol"
    ] * np.maximum(np.abs(left_lambda), np.abs(right_lambda))
    acoustic_passed = bool(np.all(lambda_difference[acoustic] <= lambda_allowed[acoustic]))
    ordinary = ~acoustic
    if np.any(ordinary):
        frequency_difference = np.abs(left_frequency - right_frequency)
        frequency_allowed = _HARMONIC_TOLERANCES[
            "same_convention_frequency_atol_thz"
        ] + _HARMONIC_TOLERANCES["same_convention_frequency_rtol"] * np.maximum(
            np.abs(left_frequency), np.abs(right_frequency)
        )
        ordinary_passed = bool(
            np.all(frequency_difference[ordinary] <= frequency_allowed[ordinary])
        )
        ordinary_max = float(np.max(frequency_difference[ordinary]))
    else:
        ordinary_passed = True
        ordinary_max = 0.0
    overall_abs, overall_relative = _pair_metrics(
        left_frequency,
        right_frequency,
        atol=_HARMONIC_TOLERANCES["same_convention_frequency_atol_thz"],
    )
    metrics = {
        "frequency_max_abs_thz": overall_abs,
        "frequency_max_relative": overall_relative,
        "acoustic_crossover_thz": float(crossover),
        "acoustic_squared_frequency_comparison_count": int(np.count_nonzero(acoustic)),
        "acoustic_omega2_max_abs_solver": (
            float(np.max(lambda_difference[acoustic])) if np.any(acoustic) else 0.0
        ),
        "above_crossover_frequency_max_abs_thz": ordinary_max,
    }
    return metrics, bool(acoustic_passed and ordinary_passed)


def compare_layered_harmonic_models(
    left: Any,
    right: Any,
    *,
    comparison: Literal["full_compact", "omega0_invariance"],
) -> LayeredHarmonicComparisonReport:
    """Compare two Phase-A models under one preregistered relation."""

    from .layered import LayeredHarmonicModel

    if not isinstance(left, LayeredHarmonicModel) or not isinstance(right, LayeredHarmonicModel):
        raise TypeError("left and right must be LayeredHarmonicModel instances.")
    if comparison not in {"full_compact", "omega0_invariance"}:
        raise ValueError("comparison must be 'full_compact' or 'omega0_invariance'.")
    if left.primitive_crystal_content_sha256 != right.primitive_crystal_content_sha256:
        raise ValueError("Models do not share the same primitive crystal identity.")
    if left.layered_geometry_sha256 != right.layered_geometry_sha256:
        raise ValueError("Models do not share the same layered geometry identity.")

    failures: list[str] = []
    frequency_left = _frequencies_thz(left)
    frequency_right = _frequencies_thz(right)
    frequency_abs, frequency_rel = _pair_metrics(frequency_left, frequency_right, atol=_HARMONIC_TOLERANCES["same_convention_frequency_atol_thz"])
    if comparison == "full_compact":
        if {left.fc2_layout, right.fc2_layout} != {"full", "compact"}:
            raise ValueError("full_compact requires one full and one compact FC2 model.")
        support_exact = bool(np.array_equal(left.mapping.longitudinal_fc2_support, right.mapping.longitudinal_fc2_support))
        metrics: dict[str, Any] = {"support_exact": support_exact}
        full_model = left if left.fc2_layout == "full" else right
        compact_model = left if left.fc2_layout == "compact" else right
        full_numerical = {
            tuple(int(value) for value in row)
            for row in full_model.mapping.longitudinal_fc2_exact_numerical_support
        }
        compact_numerical = {
            tuple(int(value) for value in row)
            for row in compact_model.mapping.longitudinal_fc2_exact_numerical_support
        }
        metrics.update(
            {
                "full_exact_numerical_support_count": len(full_numerical),
                "compact_exact_numerical_support_count": len(compact_numerical),
                "common_exact_numerical_support_count": len(
                    full_numerical & compact_numerical
                ),
                "union_exact_numerical_support_count": len(
                    full_numerical | compact_numerical
                ),
                "full_only_exact_numerical_support_count": len(
                    full_numerical - compact_numerical
                ),
                "compact_only_exact_numerical_support_count": len(
                    compact_numerical - full_numerical
                ),
            }
        )
        if not support_exact:
            failures.append("support")
        for name in ("d00", "d01", "d10"):
            absolute, relative = _pair_metrics(getattr(left, name), getattr(right, name), atol=_HARMONIC_TOLERANCES["operator_atol_solver"])
            metrics[f"{name}_max_abs_solver"] = absolute
            metrics[f"{name}_max_relative"] = relative
            if not _within(getattr(left, name), getattr(right, name), rtol=_HARMONIC_TOLERANCES["operator_rtol"], atol=_HARMONIC_TOLERANCES["operator_atol_solver"]):
                failures.append(name)
        reciprocal_abs, reciprocal_rel, reciprocal_passed = _cross_reciprocal_metrics(
            left, right
        )
        metrics["reciprocal_matrix_max_abs_source"] = reciprocal_abs
        metrics["reciprocal_matrix_max_relative"] = reciprocal_rel
        if not reciprocal_passed:
            failures.append("reciprocal_matrix")
        frequency_metrics, frequency_passed = _full_compact_frequency_metrics(
            left, right
        )
        metrics.update(frequency_metrics)
        if not frequency_passed:
            failures.append("frequency")
        full_metrics, omega_metrics = metrics, None
        qset = _qset_sha256("primitive_reciprocal_v1")
    else:
        if left.fc2_content_sha256 != right.fc2_content_sha256:
            raise ValueError("omega0_invariance requires identical FC2 content.")
        if left.scale.omega0 == right.scale.omega0:
            raise ValueError("omega0_invariance requires distinct reference frequencies.")
        metrics = {}
        for name in ("d00", "d01", "d10"):
            a = getattr(left, name) * left.scale.omega0**2
            b = getattr(right, name) * right.scale.omega0**2
            absolute, relative = _pair_metrics(a, b, atol=_HARMONIC_TOLERANCES["omega0_invariance_atol_physical_s_minus2"])
            metrics[f"physical_{name}_max_abs_s_minus2"] = absolute
            metrics[f"physical_{name}_max_relative"] = relative
            if not _within(a, b, rtol=_HARMONIC_TOLERANCES["omega0_invariance_rtol"], atol=_HARMONIC_TOLERANCES["omega0_invariance_atol_physical_s_minus2"]):
                failures.append(name)
        metrics["frequency_max_abs_thz"] = frequency_abs
        metrics["frequency_max_relative"] = frequency_rel
        if not _within(frequency_left, frequency_right, rtol=_HARMONIC_TOLERANCES["omega0_invariance_rtol"], atol=_HARMONIC_TOLERANCES["same_convention_frequency_atol_thz"]):
            failures.append("frequency")
        full_metrics, omega_metrics = None, metrics
        qset = _qset_sha256("longitudinal_folding_v1")
    failure_codes = tuple(sorted(set(failures)))
    payload = {
        "schema_version": _COMPARISON_SCHEMA,
        "comparison": comparison,
        "left_model_physics_sha256": left.model_physics_sha256,
        "right_model_physics_sha256": right.model_physics_sha256,
        "common_primitive_crystal_content_sha256": left.primitive_crystal_content_sha256,
        "common_layered_geometry_sha256": left.layered_geometry_sha256,
        "left_fc2_content_sha256": left.fc2_content_sha256,
        "right_fc2_content_sha256": right.fc2_content_sha256,
        "qset_sha256": qset,
        "full_compact_metrics_or_null": full_metrics,
        "omega0_metrics_or_null": omega_metrics,
        "tolerances": dict(_COMPARISON_TOLERANCES),
        "failure_codes": failure_codes,
        "warning_codes": (),
        "certification_passed": not failure_codes,
    }
    return LayeredHarmonicComparisonReport._from_payload(payload)


__all__ = [
    "LayeredGeometryValidationReport",
    "LayeredHarmonicComparisonReport",
    "LayeredHarmonicValidationReport",
    "compare_layered_harmonic_models",
    "validate_layered_device_mapping",
    "validate_layered_harmonic_model",
]
