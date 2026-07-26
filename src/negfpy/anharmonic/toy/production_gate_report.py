"""Strict report for the production toy-validation gate.

This module joins two independently produced pieces of evidence without
running either solver:

* contact-dressed, open three-mode oracle/one-shot-Born comparisons; and
* the three-layer, nine-DOF multidimensional SCBA release report.

The builder deliberately consumes the underlying
:class:`OpenThreeModeOracleResult` and
:class:`~negfpy.anharmonic.types.AnharmonicResult` objects.  Pole values,
errors, scaling laws, passivity, refinement changes, and array hashes are
derived here; callers cannot substitute scalar claims for those objects.

``validation_tier="pull_request"`` is an irrevocably non-certifying smoke
tier.  The release tier additionally requires the locked temperature and
coupling matrix, both signs of every nonzero coupling, one-control-at-a-time
grid/cutoff/eta refinements, and a release-certified multidimensional report.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

import numpy as np

from .._provenance import configuration_hash, problem_input_hashes
from ..types import AnharmonicResult, KeldyshSpectrum
from .multidimensional_release import MultidimensionalReleaseReport
from .open_three_mode import (
    OpenThreeModeFixture,
    OpenThreeModeOracleResult,
    measure_mode_poles,
)


Array = np.ndarray
ValidationTier = Literal["pull_request", "release"]

_SCHEMA = "negfpy.anharmonic.production_toy_gate"
_SCHEMA_VERSION = 4
_LOCKED_TEMPERATURES = (0.0, 0.7)
_LOCKED_ABSOLUTE_COUPLINGS = (0.01, 0.02)
_LOCKED_SIGNED_COUPLINGS = (-0.02, -0.01, 0.0, 0.01, 0.02)
_LOCKED_MODE_FREQUENCIES = (1.5, 0.9, 0.6)
_LOCKED_LEAD_COUPLING = 0.1
_PARENT_MODE = 0
_RELATIVE_TOLERANCE = 5.0e-3
_WEAK_COUPLING_LIMIT = 0.1
_PASSIVITY_RELATIVE_FLOOR = 1.0e-10
# A dispersive shift can cross zero under cutoff/eta refinement.  Comparing
# only relative errors there is ill-defined, so the release contract fixes an
# absolute tolerance as this fraction of the parent harmonic frequency.
_NEAR_ZERO_SHIFT_ABSOLUTE_RATIO = 1.0e-8
_LINEWIDTH_ABSOLUTE_RATIO = 5.0e-10
_SPONTANEOUS_LINEWIDTH_RATIO = 1.0e-12
_REQUIRED_INPUT_HASHES = frozenset(
    {
        "device",
        "left_reservoir",
        "right_reservoir",
        "cubic_vertex",
        "frequency_grid",
        "solver_scale",
        "problem",
        "configuration",
        "initial_state",
        "calculation",
    }
)


_LOCKED_MULTIDIMENSIONAL_COMPARISON_POLICY: dict[str, Any] = {
    "field_rtol": 1.0e-8,
    "field_atol": 1.0e-12,
    "fixed_point_field_rtol": 1.0e-8,
    "fixed_point_field_atol": 1.0e-12,
    "fixed_point_current_scope": "symmetrized_transport",
    "fixed_point_current_rtol": 1.0e-8,
    "fixed_point_current_atol": 1.0e-12,
    "backend_field_rtol": 1.0e-10,
    "backend_field_atol": 1.0e-12,
    "backend_current_scope": "full_conservation_vector",
    "backend_current_rtol": 1.0e-10,
    "backend_current_atol": 1.0e-12,
    "refinement_current_scope": "symmetrized_transport",
    "refinement_current_rtol": 5.0e-3,
    "refinement_current_atol": 1.0e-12,
    "refinement_field_policy": {
        "compute_when_grid_identical": True,
        "gate_only_when_spacing_cutoff_eta_unchanged": True,
    },
    "current_rtol": 1.0e-10,
    "current_atol": 1.0e-12,
    "legacy_current_scope": "full_conservation_vector",
    "legacy_current_policy": "backend_only",
    "refinement_rtol": 5.0e-3,
    "line_shape_rtol": 5.0e-3,
    "linewidth_atol": 1.0e-12,
    "shift_atol": 1.0e-8,
    "line_shape_observable": {
        "dof": "center_layer.parent_x",
        "target_angular_frequency": 1.5,
        "linewidth": "-Im[Sigma_pp^r_xx(Omega_p)]/Omega_p",
        "shift": "Re[Sigma_pp^r_xx(Omega_p)]/(2*Omega_p)",
        "interpolation": "piecewise-linear on the signed-frequency grid",
    },
}


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{name} must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _close(left: float, right: float) -> bool:
    scale = max(1.0, abs(float(left)), abs(float(right)))
    return bool(
        np.isclose(
            float(left),
            float(right),
            rtol=0.0,
            atol=64.0 * np.finfo(np.float64).eps * scale,
        )
    )


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _strict_json_value(value: Any, *, path: str = "$") -> Any:
    """Return JSON-native data and reject non-finite or unsupported values."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _strict_json_value(value.item(), path=path)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float.")
        return value
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} contains a non-finite complex number.")
        return {"__complex__": [float(value.real), float(value.imag)]}
    if isinstance(value, np.ndarray):
        return _strict_json_value(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} contains a non-string key.")
            converted[key] = _strict_json_value(
                item,
                path=f"{path}.{key}",
            )
        return {key: converted[key] for key in sorted(converted)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _strict_json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value of type {type(value).__name__}."
    )


def _canonical_json(document: Mapping[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(
        _strict_json_value(document),
        allow_nan=False,
        ensure_ascii=False,
        indent=indent,
        separators=(",", ":") if indent is None else None,
        sort_keys=True,
    )


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return tuple(_deep_freeze(item) for item in value)
    return value


def _array_sha256(values: Any) -> str:
    """Hash dtype, shape, and the exact contiguous raw array bytes."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(item) for item in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _spectrum_hashes(spectrum: KeldyshSpectrum) -> dict[str, str]:
    return {
        "greater": _array_sha256(spectrum.greater),
        "lesser": _array_sha256(spectrum.lesser),
        "retarded": _array_sha256(spectrum.retarded),
    }


def _require_finite_spectrum(spectrum: KeldyshSpectrum, *, name: str) -> None:
    for field_name in ("retarded", "lesser", "greater"):
        if not np.all(np.isfinite(getattr(spectrum, field_name))):
            raise ValueError(f"{name}.{field_name} contains non-finite values.")


@dataclass(frozen=True, slots=True, eq=False)
class OpenThreeModeBornRun:
    """One timed oracle/production-Born comparison using the same fixture."""

    name: str
    fixture: OpenThreeModeFixture
    oracle: OpenThreeModeOracleResult
    production: AnharmonicResult
    oracle_wall_seconds: float
    production_wall_seconds: float
    environment_lock_sha256: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        name = self.name.strip()
        if not isinstance(self.fixture, OpenThreeModeFixture):
            raise TypeError("fixture must be an OpenThreeModeFixture.")
        if not isinstance(self.oracle, OpenThreeModeOracleResult):
            raise TypeError("oracle must be an OpenThreeModeOracleResult.")
        if self.oracle.fixture is not self.fixture:
            raise ValueError(
                "oracle.fixture must be the exact fixture stored by the run."
            )
        if not isinstance(self.production, AnharmonicResult):
            raise TypeError("production must be an AnharmonicResult.")
        if self.production.method != "born":
            raise ValueError("production must be a one-shot Born result.")
        if self.production.green.ndof != 3:
            raise ValueError("the production result must contain three DOFs.")
        if (
            self.production.grid.n_positive != self.fixture.grid.n_positive
            or not _close(self.production.grid.step, self.fixture.grid.step)
            or not np.array_equal(
                self.production.grid.values,
                self.fixture.grid.values,
            )
        ):
            raise ValueError("production and fixture frequency grids differ.")

        metadata = self.production.metadata
        raw_hashes = metadata.get("input_hashes")
        if not isinstance(raw_hashes, Mapping):
            raise ValueError(
                "production metadata must contain input_hashes provenance."
            )
        expected_problem_hashes = problem_input_hashes(self.fixture.problem)
        for key, expected in expected_problem_hashes.items():
            if raw_hashes.get(key) != expected:
                raise ValueError(
                    f"production input hash {key!r} does not match the fixture."
                )
        expected_configuration = configuration_hash(self.fixture.config)
        if raw_hashes.get("configuration") != expected_configuration:
            raise ValueError(
                "production configuration hash does not match fixture.config."
            )
        for key in _REQUIRED_INPUT_HASHES:
            if not _is_sha256(raw_hashes.get(key)):
                raise ValueError(
                    f"production input hash {key!r} is absent or invalid."
                )
        for key, expected in (
            ("eta_lead", self.fixture.eta_lead),
            ("temperature_left", self.fixture.temperature),
            ("temperature_right", self.fixture.temperature),
            ("grid_step", self.fixture.grid.step),
            ("grid_cutoff", self.fixture.grid.cutoff),
            ("vertex_max_abs", abs(self.fixture.g)),
        ):
            value = metadata.get(key)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(float(value))
                or not _close(float(value), expected)
            ):
                raise ValueError(
                    f"production metadata {key!r} does not match the fixture."
                )

        _require_finite_spectrum(self.oracle.green_zero, name="oracle.green_zero")
        _require_finite_spectrum(self.oracle.sigma_pp, name="oracle.sigma_pp")
        _require_finite_spectrum(self.production.green, name="production.green")
        _require_finite_spectrum(
            self.production.sigma_pp,
            name="production.sigma_pp",
        )
        if (
            self.environment_lock_sha256 is not None
            and not _is_sha256(self.environment_lock_sha256)
        ):
            raise ValueError(
                "environment_lock_sha256 must be None or a lowercase "
                "64-character SHA256 digest."
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(
            self,
            "oracle_wall_seconds",
            _finite_nonnegative(
                self.oracle_wall_seconds,
                name="oracle_wall_seconds",
            ),
        )
        object.__setattr__(
            self,
            "production_wall_seconds",
            _finite_nonnegative(
                self.production_wall_seconds,
                name="production_wall_seconds",
            ),
        )


@dataclass(frozen=True, slots=True, eq=False)
class ProductionToyGateReport:
    """Immutable strict-JSON wrapper for the combined production toy gate."""

    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.document, Mapping):
            raise TypeError("document must be a mapping.")
        normalized = _strict_json_value(self.document)
        reported_digest = normalized.get("report_sha256")
        if not _is_sha256(reported_digest):
            raise ValueError("document must contain a lowercase SHA256 digest.")
        core_document = {
            key: value
            for key, value in normalized.items()
            if key != "report_sha256"
        }
        expected_digest = hashlib.sha256(
            _canonical_json(core_document).encode("utf-8")
        ).hexdigest()
        if reported_digest != expected_digest:
            raise ValueError("report_sha256 does not match the report body.")
        _canonical_json(normalized)
        object.__setattr__(self, "document", _deep_freeze(normalized))

    @property
    def release_certified(self) -> bool:
        return bool(self.document["summary"]["release_certified"])

    @property
    def oracle_gate_passed(self) -> bool:
        return bool(self.document["summary"]["oracle_gate_passed"])

    @property
    def sha256(self) -> str:
        return str(self.document["report_sha256"])

    def as_dict(self) -> dict[str, Any]:
        return json.loads(self.to_json())

    def to_json(self, *, indent: int | None = None) -> str:
        if indent is not None:
            if isinstance(indent, bool) or not isinstance(indent, int):
                raise TypeError("indent must be an integer or None.")
            if indent < 0:
                raise ValueError("indent cannot be negative.")
        return _canonical_json(self.document, indent=indent)

    def write_json(
        self,
        path: str | Path,
        *,
        indent: int | None = 2,
    ) -> Path:
        destination = Path(path).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = (
            destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
        )
        payload = (self.to_json(indent=indent) + "\n").encode("utf-8")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
            directory_fd = os.open(destination.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
        return destination


def _parent_scale(fixture: OpenThreeModeFixture) -> float:
    return float(fixture.mode_frequencies[_PARENT_MODE])


def _error_gate(
    candidate: float,
    reference: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float = _RELATIVE_TOLERANCE,
) -> dict[str, Any]:
    difference = abs(float(candidate) - float(reference))
    comparison_scale = max(
        abs(float(candidate)),
        abs(float(reference)),
        absolute_tolerance / max(relative_tolerance, np.finfo(float).eps),
    )
    allowed = max(absolute_tolerance, relative_tolerance * comparison_scale)
    return {
        "absolute_difference": difference,
        "absolute_tolerance": absolute_tolerance,
        "allowed_difference": allowed,
        "comparison_scale": comparison_scale,
        "normalized_error": difference / comparison_scale,
        "passed": bool(difference <= allowed),
        "relative_tolerance": relative_tolerance,
    }


def _scaled_pair_gate(
    lower: float,
    upper: float,
    *,
    expected_upper_over_lower: float,
    absolute_tolerance: float,
) -> dict[str, Any]:
    expected = expected_upper_over_lower * float(lower)
    gate = _error_gate(
        float(upper),
        expected,
        absolute_tolerance=absolute_tolerance,
    )
    gate.update(
        {
            "expected_upper": expected,
            "expected_upper_over_lower": expected_upper_over_lower,
            "lower": float(lower),
            "upper": float(upper),
        }
    )
    return gate


def _spectral_passivity(sigma_retarded: Array, omega: Array) -> dict[str, Any]:
    retarded = np.asarray(sigma_retarded, dtype=np.complex128)
    frequencies = np.asarray(omega, dtype=np.float64)
    minimum = math.inf
    spectral_scale = 0.0
    for index, frequency in enumerate(frequencies):
        if frequency == 0.0:
            continue
        advanced = retarded[index].conj().T
        broadening = 1.0j * (retarded[index] - advanced)
        signed = np.sign(frequency) * 0.5 * (broadening + broadening.conj().T)
        eigenvalues = np.linalg.eigvalsh(signed).real
        minimum = min(minimum, float(np.min(eigenvalues)))
        spectral_scale = max(spectral_scale, float(np.max(np.abs(eigenvalues))))
    if math.isinf(minimum):
        minimum = 0.0
    allowed_floor = -_PASSIVITY_RELATIVE_FLOOR * max(1.0, spectral_scale)
    return {
        "allowed_eigenvalue_floor": allowed_floor,
        "minimum_signed_broadening_eigenvalue": minimum,
        "passed": bool(minimum >= allowed_floor),
        "relative_floor": _PASSIVITY_RELATIVE_FLOOR,
        "spectral_scale": spectral_scale,
    }


def _code_provenance(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    raw_hashes = metadata.get("input_hashes")
    input_hashes = (
        {str(key): str(value) for key, value in raw_hashes.items()}
        if isinstance(raw_hashes, Mapping)
        else {}
    )
    input_complete = bool(
        _REQUIRED_INPUT_HASHES.issubset(input_hashes)
        and all(_is_sha256(input_hashes[key]) for key in _REQUIRED_INPUT_HASHES)
    )
    source_hash = metadata.get("source_tree_sha256")
    revision = metadata.get("git_revision")
    revision_valid = bool(
        isinstance(revision, str)
        and len(revision) in {40, 64}
        and all(character in "0123456789abcdef" for character in revision)
    )
    raw_blas = metadata.get("blas_runtime")
    blas_valid = bool(
        isinstance(raw_blas, (tuple, list))
        and all(isinstance(value, str) and value for value in raw_blas)
    )
    identity = {
        "declared_package_version": metadata.get(
            "declared_package_version"
        ),
        "installed_package_version": metadata.get(
            "installed_package_version"
        ),
        "git_revision": revision,
        "source_tree_sha256": source_hash,
        "dtype": metadata.get("dtype"),
        "numpy_version": metadata.get("numpy_version"),
        "scipy_version": metadata.get("scipy_version"),
        "python": metadata.get("python"),
        "platform": metadata.get("platform"),
        "blas_runtime": list(raw_blas) if blas_valid else None,
    }
    environment_complete = bool(
        identity["dtype"] == "complex128"
        and blas_valid
        and all(
            isinstance(identity[key], str) and bool(identity[key])
            for key in ("numpy_version", "scipy_version", "python", "platform")
        )
    )
    clean = metadata.get("git_dirty") is False
    code_complete = bool(
        _is_sha256(source_hash)
        and revision_valid
        and clean
        and metadata.get("package_version_consistent") is True
        and isinstance(metadata.get("declared_package_version"), str)
        and bool(metadata.get("declared_package_version"))
        and isinstance(metadata.get("installed_package_version"), str)
        and bool(metadata.get("installed_package_version"))
    )
    return (
        {
            **identity,
            "clean": clean,
            "git_dirty": metadata.get("git_dirty"),
            "identity": identity,
            "input_hashes": input_hashes,
            "installed_package_version": metadata.get(
                "installed_package_version"
            ),
            "package_version_consistent": metadata.get(
                "package_version_consistent"
            ),
        },
        bool(input_complete and code_complete and environment_complete),
    )


def _run_document(run: OpenThreeModeBornRun) -> dict[str, Any]:
    fixture = run.fixture
    oracle_shifts = np.asarray(run.oracle.frequency_shifts, dtype=np.float64)
    oracle_linewidths = np.asarray(run.oracle.linewidths, dtype=np.float64)
    production_shifts, production_linewidths = measure_mode_poles(
        fixture.grid,
        fixture.mode_frequencies,
        run.production.sigma_pp.retarded,
    )
    parent_frequency = _parent_scale(fixture)
    shift_atol = _NEAR_ZERO_SHIFT_ABSOLUTE_RATIO * parent_frequency
    linewidth_atol = _LINEWIDTH_ABSOLUTE_RATIO * parent_frequency
    shift_gate = _error_gate(
        float(production_shifts[_PARENT_MODE]),
        float(oracle_shifts[_PARENT_MODE]),
        absolute_tolerance=shift_atol,
    )
    linewidth_gate = _error_gate(
        float(production_linewidths[_PARENT_MODE]),
        float(oracle_linewidths[_PARENT_MODE]),
        absolute_tolerance=linewidth_atol,
    )
    oracle_passivity = _spectral_passivity(
        run.oracle.sigma_pp.retarded,
        fixture.grid.values,
    )
    production_passivity = _spectral_passivity(
        run.production.sigma_pp.retarded,
        fixture.grid.values,
    )
    production_pole_passive = bool(
        np.min(production_linewidths)
        >= -_PASSIVITY_RELATIVE_FLOOR
        * max(1.0, float(np.max(np.abs(production_linewidths))))
    )
    oracle_weak = bool(
        run.oracle.weak_coupling_passed
        and run.oracle.weak_coupling_limit <= _WEAK_COUPLING_LIMIT
    )
    production_weak = bool(
        production_pole_passive
        and np.all(
            np.abs(production_shifts)
            / np.asarray(fixture.mode_frequencies)
            < _WEAK_COUPLING_LIMIT
        )
        and np.all(
            np.abs(production_linewidths)
            / np.asarray(fixture.mode_frequencies)
            < _WEAK_COUPLING_LIMIT
        )
    )
    integration_errors = np.asarray(
        run.oracle.integration_error_estimates,
        dtype=np.float64,
    )
    parent_shift_error_estimate = float(
        (
            integration_errors[_PARENT_MODE, 0]
            + integration_errors[_PARENT_MODE, 2]
        )
        / (2.0 * parent_frequency)
    )
    parent_linewidth_error_estimate = float(
        integration_errors[_PARENT_MODE, 1] / parent_frequency
    )
    shift_error_allowed = max(
        shift_atol,
        _RELATIVE_TOLERANCE
        * abs(float(oracle_shifts[_PARENT_MODE])),
    )
    linewidth_error_allowed = max(
        linewidth_atol,
        _RELATIVE_TOLERANCE
        * abs(float(oracle_linewidths[_PARENT_MODE])),
    )
    method_metadata = run.oracle.method_metadata
    adaptive_eta = method_metadata.get("eta_used_for_adaptive_poles")
    adaptive_eta_is_zero = bool(
        isinstance(adaptive_eta, (int, float, np.number))
        and not isinstance(adaptive_eta, (bool, np.bool_))
        and np.isfinite(float(adaptive_eta))
        and _close(float(adaptive_eta), 0.0)
    )
    oracle_method_passed = bool(
        method_metadata.get("pole_method")
        == "adaptive_eta_free_continuum_on_shell"
        and adaptive_eta_is_zero
        and parent_shift_error_estimate <= shift_error_allowed
        and parent_linewidth_error_estimate <= linewidth_error_allowed
    )
    provenance, provenance_complete = _code_provenance(
        run.production.metadata
    )
    independently_derived_hashes = problem_input_hashes(fixture.problem)
    independently_derived_hashes["configuration"] = configuration_hash(
        fixture.config
    )
    return {
        "controls": {
            "cutoff": fixture.grid.cutoff,
            "eta_lead": fixture.eta_lead,
            "g": fixture.g,
            "grid_n_points": fixture.grid.n_points,
            "grid_n_positive": fixture.grid.n_positive,
            "grid_step": fixture.grid.step,
            "hilbert_backend": fixture.config.hilbert_backend,
            "lead_coupling": fixture.lead_coupling,
            "mode_frequencies": list(fixture.mode_frequencies),
            "production_backend": fixture.config.backend,
            "surface_gf_method": fixture.config.surface_gf_method,
            "temperature": fixture.temperature,
        },
        "gates": {
            "oracle_production_agreement": {
                "linewidth": linewidth_gate,
                "passed": bool(
                    linewidth_gate["passed"] and shift_gate["passed"]
                ),
                "shift": shift_gate,
            },
            "oracle_method": {
                "linewidth_error_allowed": linewidth_error_allowed,
                "parent_linewidth_error_estimate": (
                    parent_linewidth_error_estimate
                ),
                "parent_shift_error_estimate": (
                    parent_shift_error_estimate
                ),
                "passed": oracle_method_passed,
                "pole_method": method_metadata.get("pole_method"),
                "eta_used_for_adaptive_poles": adaptive_eta,
                "shift_error_allowed": shift_error_allowed,
            },
            "passivity": {
                "oracle_pole_passive": run.oracle.passive,
                "oracle_spectrum": oracle_passivity,
                "passed": bool(
                    run.oracle.passive
                    and production_pole_passive
                    and oracle_passivity["passed"]
                    and production_passivity["passed"]
                ),
                "production_pole_passive": production_pole_passive,
                "production_spectrum": production_passivity,
            },
            "weak_coupling": {
                "limit": _WEAK_COUPLING_LIMIT,
                "oracle_passed": oracle_weak,
                "passed": bool(oracle_weak and production_weak),
                "production_passed": production_weak,
            },
            "production_completed": bool(
                run.production.converged
                and run.production.diagnostics.passed
            ),
        },
        "input_hashes": {
            "independently_derived": independently_derived_hashes,
            "production_recorded": dict(
                run.production.metadata["input_hashes"]
            ),
        },
        "environment_lock_sha256": run.environment_lock_sha256,
        "name": run.name,
        "oracle": {
            "frequency_shifts": oracle_shifts.tolist(),
            "grid_frequency_shifts": (
                np.asarray(run.oracle.grid_frequency_shifts).tolist()
            ),
            "grid_linewidths": (
                np.asarray(run.oracle.grid_linewidths).tolist()
            ),
            "integration_error_estimates": (
                np.asarray(run.oracle.integration_error_estimates).tolist()
            ),
            "linewidths": oracle_linewidths.tolist(),
            "method_metadata": dict(method_metadata),
            "parent_frequency_shift": float(oracle_shifts[_PARENT_MODE]),
            "parent_linewidth": float(oracle_linewidths[_PARENT_MODE]),
            "raw_array_sha256": {
                "frequency_shifts": _array_sha256(oracle_shifts),
                "green_zero": _spectrum_hashes(run.oracle.green_zero),
                "grid_frequency_shifts": _array_sha256(
                    run.oracle.grid_frequency_shifts
                ),
                "grid_linewidths": _array_sha256(
                    run.oracle.grid_linewidths
                ),
                "integration_error_estimates": _array_sha256(
                    run.oracle.integration_error_estimates
                ),
                "linewidths": _array_sha256(oracle_linewidths),
                "self_energy_on_shell": _array_sha256(
                    run.oracle.self_energy_on_shell
                ),
                "sigma_pp": _spectrum_hashes(run.oracle.sigma_pp),
            },
            "self_energy_on_shell": (
                np.asarray(run.oracle.self_energy_on_shell).tolist()
            ),
        },
        "production_born": {
            "converged": run.production.converged,
            "diagnostics_passed": run.production.diagnostics.passed,
            "frequency_shifts": production_shifts.tolist(),
            "linewidths": production_linewidths.tolist(),
            "parent_frequency_shift": float(
                production_shifts[_PARENT_MODE]
            ),
            "parent_linewidth": float(
                production_linewidths[_PARENT_MODE]
            ),
            "raw_array_sha256": {
                "green": _spectrum_hashes(run.production.green),
                "sigma_left": _spectrum_hashes(run.production.sigma_left),
                "sigma_pp": _spectrum_hashes(run.production.sigma_pp),
                "sigma_right": _spectrum_hashes(run.production.sigma_right),
            },
            "release_certified": run.production.release_certified,
        },
        "provenance": provenance,
        "provenance_complete": provenance_complete,
        "timings": {
            "oracle_wall_seconds": run.oracle_wall_seconds,
            "production_born_wall_seconds": run.production_wall_seconds,
        },
    }


def _signature(run: OpenThreeModeBornRun) -> tuple[float, int, float]:
    return (
        run.fixture.grid.step,
        run.fixture.grid.n_positive,
        run.fixture.eta_lead,
    )


def _same_controls(
    run: OpenThreeModeBornRun,
    signature: tuple[float, int, float],
) -> bool:
    step, n_positive, eta = signature
    return bool(
        _close(run.fixture.grid.step, step)
        and run.fixture.grid.n_positive == n_positive
        and _close(run.fixture.eta_lead, eta)
    )


def _same_state(
    run: OpenThreeModeBornRun,
    *,
    temperature: float,
    g: float,
) -> bool:
    return bool(
        _close(run.fixture.temperature, temperature)
        and _close(run.fixture.g, g)
    )


def _find_unique_run(
    runs: Sequence[OpenThreeModeBornRun],
    *,
    signature: tuple[float, int, float],
    temperature: float,
    g: float,
) -> OpenThreeModeBornRun | None:
    matches = [
        run
        for run in runs
        if _same_controls(run, signature)
        and _same_state(run, temperature=temperature, g=g)
    ]
    if len(matches) > 1:
        raise ValueError(
            "duplicate open-three-mode runs have the same physical state and "
            "numerical controls."
        )
    return matches[0] if matches else None


def _refinement_candidates(
    runs: Sequence[OpenThreeModeBornRun],
    *,
    base: OpenThreeModeBornRun,
    control: str,
) -> list[OpenThreeModeBornRun]:
    selected: list[OpenThreeModeBornRun] = []
    for candidate in runs:
        if candidate is base:
            continue
        if not _same_state(
            candidate,
            temperature=base.fixture.temperature,
            g=base.fixture.g,
        ):
            continue
        if not all(
            _close(left, right)
            for left, right in zip(
                candidate.fixture.mode_frequencies,
                base.fixture.mode_frequencies,
            )
        ) or not _close(
            candidate.fixture.lead_coupling,
            base.fixture.lead_coupling,
        ):
            continue
        spacing_changed = not _close(
            candidate.fixture.grid.step,
            base.fixture.grid.step,
        )
        cutoff_changed = not _close(
            candidate.fixture.grid.cutoff,
            base.fixture.grid.cutoff,
        )
        eta_changed = not _close(
            candidate.fixture.eta_lead,
            base.fixture.eta_lead,
        )
        changed = {
            name
            for name, value in (
                ("spacing", spacing_changed),
                ("cutoff", cutoff_changed),
                ("eta_lead", eta_changed),
            )
            if value
        }
        if changed != {control}:
            continue
        refined_direction = {
            "spacing": (
                candidate.fixture.grid.step < base.fixture.grid.step
            ),
            "cutoff": (
                candidate.fixture.grid.cutoff > base.fixture.grid.cutoff
            ),
            "eta_lead": (
                candidate.fixture.eta_lead < base.fixture.eta_lead
            ),
        }[control]
        if refined_direction:
            selected.append(candidate)
    if control == "spacing":
        selected.sort(key=lambda run: (run.fixture.grid.step, run.name))
    elif control == "cutoff":
        selected.sort(
            key=lambda run: (-run.fixture.grid.cutoff, run.name)
        )
    else:
        selected.sort(key=lambda run: (run.fixture.eta_lead, run.name))
    return selected


def _select_baseline(
    runs: Sequence[OpenThreeModeBornRun],
) -> tuple[
    tuple[float, int, float],
    dict[tuple[float, float], OpenThreeModeBornRun | None],
    dict[str, OpenThreeModeBornRun | None],
]:
    signatures = sorted(
        {_signature(run) for run in runs},
        key=lambda item: (-item[0], item[0] * item[1], -item[2], item[1]),
    )
    best: tuple[
        int,
        int,
        tuple[float, int, float],
        dict[tuple[float, float], OpenThreeModeBornRun | None],
        dict[str, OpenThreeModeBornRun | None],
    ] | None = None
    for signature in signatures:
        coverage = {
            (temperature, g): _find_unique_run(
                runs,
                signature=signature,
                temperature=temperature,
                g=g,
            )
            for temperature in _LOCKED_TEMPERATURES
            for g in _LOCKED_SIGNED_COUPLINGS
        }
        anchor = coverage[(0.7, 0.02)]
        refinements: dict[str, OpenThreeModeBornRun | None] = {
            "spacing": None,
            "cutoff": None,
            "eta_lead": None,
        }
        if anchor is not None:
            for control in refinements:
                candidates = _refinement_candidates(
                    runs,
                    base=anchor,
                    control=control,
                )
                refinements[control] = candidates[0] if candidates else None
        coverage_count = sum(item is not None for item in coverage.values())
        refinement_count = sum(
            item is not None for item in refinements.values()
        )
        score = (coverage_count, refinement_count)
        if best is None or score > best[:2]:
            best = (
                coverage_count,
                refinement_count,
                signature,
                coverage,
                refinements,
            )
    if best is None:
        raise ValueError("at least one open-three-mode run is required.")
    return best[2], best[3], best[4]


def _observable_pair_document(
    lower: Mapping[str, Any],
    upper: Mapping[str, Any],
    *,
    expected_ratio: float,
    parent_frequency: float,
) -> dict[str, Any]:
    shift_atol = _NEAR_ZERO_SHIFT_ABSOLUTE_RATIO * parent_frequency
    linewidth_atol = _LINEWIDTH_ABSOLUTE_RATIO * parent_frequency
    document: dict[str, Any] = {}
    for source, source_key in (
        ("oracle", "oracle"),
        ("production_born", "production_born"),
    ):
        shift_gate = _scaled_pair_gate(
            float(lower[source_key]["parent_frequency_shift"]),
            float(upper[source_key]["parent_frequency_shift"]),
            expected_upper_over_lower=expected_ratio,
            absolute_tolerance=shift_atol,
        )
        linewidth_gate = _scaled_pair_gate(
            float(lower[source_key]["parent_linewidth"]),
            float(upper[source_key]["parent_linewidth"]),
            expected_upper_over_lower=expected_ratio,
            absolute_tolerance=linewidth_atol,
        )
        document[source] = {
            "linewidth": linewidth_gate,
            "passed": bool(
                shift_gate["passed"] and linewidth_gate["passed"]
            ),
            "shift": shift_gate,
        }
    document["passed"] = bool(
        document["oracle"]["passed"]
        and document["production_born"]["passed"]
    )
    return document


def _refinement_document(
    base: OpenThreeModeBornRun,
    candidate: OpenThreeModeBornRun,
    documents: Mapping[str, Mapping[str, Any]],
    *,
    control: str,
) -> dict[str, Any]:
    base_document = documents[base.name]
    candidate_document = documents[candidate.name]
    comparison = _observable_pair_document(
        base_document,
        candidate_document,
        expected_ratio=1.0,
        parent_frequency=_parent_scale(base.fixture),
    )
    return {
        "base": base.name,
        "candidate": candidate.name,
        "control": control,
        "direction": {
            "base": {
                "cutoff": base.fixture.grid.cutoff,
                "eta_lead": base.fixture.eta_lead,
                "spacing": base.fixture.grid.step,
            },
            "candidate": {
                "cutoff": candidate.fixture.grid.cutoff,
                "eta_lead": candidate.fixture.eta_lead,
                "spacing": candidate.fixture.grid.step,
            },
        },
        "one_control_changed": True,
        **comparison,
    }


def _collect_timing_values(value: Any, *, path: str = "$") -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        for key in sorted(value):
            item = value[key]
            child_path = f"{path}.{key}"
            lowered = str(key).lower()
            if (
                isinstance(item, (int, float, np.number))
                and not isinstance(item, (bool, np.bool_))
                and np.isfinite(float(item))
                and float(item) >= 0.0
                and ("second" in lowered or "timing" in lowered)
            ):
                collected.append(
                    {"path": child_path, "wall_seconds": float(item)}
                )
            collected.extend(_collect_timing_values(item, path=child_path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            collected.extend(
                _collect_timing_values(item, path=f"{path}[{index}]")
            )
    return collected


def _timing_summary(values: Sequence[float]) -> dict[str, Any]:
    raw = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if raw.size == 0:
        return {
            "count": 0,
            "maximum_wall_seconds": None,
            "median_wall_seconds": None,
            "minimum_wall_seconds": None,
            "total_wall_seconds": 0.0,
        }
    return {
        "count": int(raw.size),
        "maximum_wall_seconds": float(np.max(raw)),
        "median_wall_seconds": float(np.median(raw)),
        "minimum_wall_seconds": float(np.min(raw)),
        "total_wall_seconds": float(np.sum(raw)),
    }


def _tuple_close(values: Sequence[float], expected: Sequence[float]) -> bool:
    return bool(
        len(values) == len(expected)
        and all(_close(left, right) for left, right in zip(values, expected))
    )


def _multidimensional_contract_status(
    report: MultidimensionalReleaseReport,
    document: Mapping[str, Any],
) -> tuple[bool, bool]:
    """Return independent static-configuration and runtime-evidence gates."""

    config = report.config
    spec = config.spec
    numerical = config.numerical
    raw_refinement_coverage = document.get("required_refinement_coverage")
    required_directions = {
        "half_spacing",
        "larger_cutoff",
        "half_eta",
        "changed_mixing",
    }
    refinement_directions_passed = bool(
        isinstance(raw_refinement_coverage, Mapping)
        and set(raw_refinement_coverage) == required_directions
        and all(
            isinstance(raw_refinement_coverage[direction], Mapping)
            and raw_refinement_coverage[direction].get("executed") is True
            and raw_refinement_coverage[direction].get(
                "certification_critical"
            ) is True
            and raw_refinement_coverage[direction].get("passed") is True
            for direction in required_directions
        )
    )
    archive = document.get("numerical_refinement_archive")
    all_six_archived = bool(
        isinstance(archive, Mapping)
        and archive.get("all_six_archived") is True
        and len(archive.get("expected_comparison_names", ())) == 6
        and archive.get("expected_comparison_names")
        == archive.get("observed_comparison_names")
    )
    config_document = document.get("config")
    comparison_document = (
        config_document.get("comparison")
        if isinstance(config_document, Mapping)
        else None
    )
    serialized_comparison_policy_passed = bool(
        isinstance(comparison_document, Mapping)
        and dict(comparison_document)
        == _LOCKED_MULTIDIMENSIONAL_COMPARISON_POLICY
    )
    checkpoint_evidence = document.get("checkpoint_evidence", {})
    checkpoint_environment_sha256 = (
        checkpoint_evidence.get("environment_sha256")
        if isinstance(checkpoint_evidence, Mapping)
        else None
    )
    multidimensional_provenance_document = document.get("provenance", {})
    multidimensional_identity = (
        multidimensional_provenance_document.get("identity")
        if isinstance(multidimensional_provenance_document, Mapping)
        else None
    )
    multidimensional_revision = (
        multidimensional_identity.get("git_revision")
        if isinstance(multidimensional_identity, Mapping)
        else None
    )
    exact_release_revision = bool(
        isinstance(multidimensional_revision, str)
        and len(multidimensional_revision) == 40
        and all(
            character in "0123456789abcdef"
            for character in multidimensional_revision
        )
    )
    environment_evidence_valid = bool(
        _is_sha256(checkpoint_environment_sha256)
        and isinstance(checkpoint_evidence, Mapping)
        and checkpoint_evidence.get("passed") is True
    )
    harmonic_preflight = document.get("harmonic_preflight")
    harmonic_points = (
        harmonic_preflight.get("points", ())
        if isinstance(harmonic_preflight, Mapping)
        else ()
    )
    harmonic_comparisons = (
        harmonic_preflight.get("comparisons", ())
        if isinstance(harmonic_preflight, Mapping)
        else ()
    )
    harmonic_preflight_passed = bool(
        isinstance(harmonic_preflight, Mapping)
        and harmonic_preflight.get("schema_version") == 2
        and harmonic_preflight.get("environment_sha256")
        == checkpoint_environment_sha256
        and harmonic_preflight.get("protocol")
        == "multidimensional_toy_gate_v2"
        and harmonic_preflight.get("scope")
        == "base_plus_six_preregistered_numerical_points"
        and harmonic_preflight.get("point_count") == 7
        and harmonic_preflight.get("all_seven_archived") is True
        and len(harmonic_points) == 7
        and all(
            isinstance(point, Mapping)
            and point.get("passed") is True
            and point.get("environment_sha256")
            == checkpoint_environment_sha256
            and isinstance(point.get("field_sha256"), Mapping)
            and len(point["field_sha256"]) == 9
            and all(
                _is_sha256(value)
                for value in point["field_sha256"].values()
            )
            and isinstance(point.get("input_hashes"), Mapping)
            and _REQUIRED_INPUT_HASHES.issubset(point["input_hashes"])
            and all(
                _is_sha256(point["input_hashes"][name])
                for name in _REQUIRED_INPUT_HASHES
            )
            and isinstance(point.get("provenance"), Mapping)
            and point["provenance"].get("complete") is True
            and point["provenance"].get("clean") is True
            and point["provenance"].get("identity")
            == multidimensional_identity
            for point in harmonic_points
        )
        and len(harmonic_comparisons) == 6
        and sum(
            isinstance(item, Mapping)
            and item.get("certification_critical") is True
            for item in harmonic_comparisons
        )
        == 4
        and all(
            item.get("passed") is True
            for item in harmonic_comparisons
            if isinstance(item, Mapping)
            and item.get("certification_critical") is True
        )
        and isinstance(harmonic_preflight.get("provenance"), Mapping)
        and harmonic_preflight["provenance"].get("complete_and_clean")
        is True
        and harmonic_preflight["provenance"].get("consistent") is True
        and harmonic_preflight["provenance"].get("identity")
        == multidimensional_identity
        and harmonic_preflight.get("passed") is True
    )
    backend_validation = document.get("backend_validation")
    backend_status = (
        backend_validation.get("backend_status", {})
        if isinstance(backend_validation, Mapping)
        else {}
    )
    backend_comparisons = (
        backend_validation.get("comparisons", ())
        if isinstance(backend_validation, Mapping)
        else ()
    )
    backend_validation_passed = bool(
        isinstance(backend_validation, Mapping)
        and backend_validation.get("environment_sha256")
        == checkpoint_environment_sha256
        and backend_validation.get("protocol")
        == "multidimensional_toy_gate_v2"
        and backend_validation.get("scope")
        == "bounded_converged_scba_backend_oracle"
        and backend_validation.get(
            "full_grid_serial_threaded_comparison_executed"
        )
        is False
        and backend_validation.get("backends_executed")
        == ["serial", "threaded", "fft"]
        and isinstance(backend_status, Mapping)
        and set(backend_status) == {"serial", "threaded", "fft"}
        and all(
            isinstance(status, Mapping)
            and isinstance(status.get("provenance"), Mapping)
            and status.get("method") == "scba"
            and status.get("converged") is True
            and isinstance(status.get("iterations"), int)
            and status.get("iterations", 0) >= 3
            and isinstance(status.get("elapsed_seconds"), (float, int))
            and math.isfinite(float(status["elapsed_seconds"]))
            and float(status["elapsed_seconds"]) >= 0.0
            and status.get("environment_sha256")
            == checkpoint_environment_sha256
            and isinstance(status.get("field_sha256"), Mapping)
            and len(status["field_sha256"]) == 6
            and all(
                _is_sha256(value)
                for value in status["field_sha256"].values()
            )
            and isinstance(status.get("input_hashes"), Mapping)
            and _REQUIRED_INPUT_HASHES.issubset(status["input_hashes"])
            and all(
                _is_sha256(status["input_hashes"][name])
                for name in _REQUIRED_INPUT_HASHES
            )
            for status in backend_status.values()
        )
        and len(backend_comparisons) == 2
        and all(
            isinstance(item, Mapping) and item.get("passed") is True
            for item in backend_comparisons
        )
        and isinstance(backend_validation.get("provenance"), Mapping)
        and backend_validation["provenance"].get("complete_and_clean")
        is True
        and backend_validation["provenance"].get("consistent") is True
        and all(
            status["provenance"].get("identity")
            == multidimensional_identity
            for status in backend_status.values()
        )
        and backend_validation.get("passed") is True
    )
    topology = document.get("topology_validation")
    observed_topology_rows = (
        topology.get("observed_outcomes", ())
        if isinstance(topology, Mapping)
        else ()
    )
    observed_topology_rows_valid = bool(
        isinstance(observed_topology_rows, Sequence)
        and not isinstance(observed_topology_rows, (str, bytes))
        and len(observed_topology_rows) == 13
        and all(isinstance(row, Mapping) for row in observed_topology_rows)
    )
    observed_topology_ids = (
        {str(row.get("run_id")) for row in observed_topology_rows}
        if observed_topology_rows_valid
        else set()
    )
    outcome_certification = document.get("outcome_certification")
    required_outcome_ids = outcome_certification.get(
        "required_run_ids",
        (),
    ) if isinstance(outcome_certification, Mapping) else ()
    diagnostic_outcome_ids = outcome_certification.get(
        "diagnostic_only_run_ids",
        (),
    ) if isinstance(outcome_certification, Mapping) else ()
    outcome_records = outcome_certification.get(
        "records",
        (),
    ) if isinstance(outcome_certification, Mapping) else ()
    outcome_certification_passed = bool(
        isinstance(outcome_certification, Mapping)
        and outcome_certification.get("applicable") is True
        and outcome_certification.get("coverage") is True
        and outcome_certification.get("passed") is True
        and isinstance(required_outcome_ids, Sequence)
        and not isinstance(required_outcome_ids, (str, bytes))
        and len(required_outcome_ids) == 11
        and len(set(required_outcome_ids)) == 11
        and isinstance(diagnostic_outcome_ids, Sequence)
        and not isinstance(diagnostic_outcome_ids, (str, bytes))
        and len(diagnostic_outcome_ids) == 2
        and len(set(diagnostic_outcome_ids)) == 2
        and set(required_outcome_ids).isdisjoint(diagnostic_outcome_ids)
        and set(required_outcome_ids) | set(diagnostic_outcome_ids)
        == observed_topology_ids
        and isinstance(outcome_records, Sequence)
        and not isinstance(outcome_records, (str, bytes))
        and len(outcome_records) == 11
        and all(isinstance(record, Mapping) for record in outcome_records)
        and {record.get("run_id") for record in outcome_records}
        == set(required_outcome_ids)
        and all(
            record.get("result_available") is True
            and record.get("converged") is True
            and record.get("release_certified") is True
            for record in outcome_records
        )
    )
    fft_topology_passed = bool(
        isinstance(topology, Mapping)
        and topology.get("passed") is True
        and topology.get("expected_outcome_count") == 13
        and topology.get("observed_outcome_count") == 13
        and topology.get("full_release_count") is True
        and observed_topology_rows_valid
        and len(observed_topology_ids) == 13
        and all(
            row.get("backend") == "fft"
            for row in observed_topology_rows
        )
        and not any(
            comparison.kind == "serial_vs_threaded"
            for comparison in report.comparisons
        )
    )
    static_configuration_passed = bool(
        document.get("schema_version") == 4
        and document.get("protocol") == "multidimensional_toy_gate_v2"
        and document.get("protocol_version") == 2
        and isinstance(config_document, Mapping)
        and config_document.get("protocol")
        == "multidimensional_toy_gate_v2"
        and config_document.get("protocol_version") == 2
        and serialized_comparison_policy_passed
        and config.profile == "full_preregistered"
        and config.certification_eligible
        and config.full_physical_ladder_selected
        and config.run_preregistered_refinements
        and config.run_bounded_backend_validation
        and config.production_backend == "fft"
        and config.production_hilbert_backend == "fft"
        and config.surface_gf_method == "analytic_diagonal"
        and _tuple_close(
            spec.polarization_frequencies,
            (1.5, 0.9, 0.6),
        )
        and _tuple_close(spec.masses, (1.0, 1.0, 1.0))
        and _tuple_close(spec.springs, (0.1, 0.1, 0.1))
        and _tuple_close(spec.pinnings, (2.05, 0.61, 0.16))
        and _tuple_close(spec.g_values, (0.0, 0.025, 0.05))
        and _close(spec.mean_temperature, 0.7)
        and _tuple_close(spec.bias_values, (0.0, 0.1, 0.2))
        and _tuple_close(
            spec.spacings,
            (0.0015625, 0.00078125, 0.000390625),
        )
        and _tuple_close(spec.cutoffs, (5.0, 6.0))
        and _tuple_close(
            spec.etas,
            (0.003125, 0.0015625, 0.00078125),
        )
        and _tuple_close(spec.mixings, (0.2, 0.1))
        and _close(numerical.spacing, 0.00078125)
        and _close(numerical.cutoff, 5.0)
        and _close(numerical.eta_lead, 0.0015625)
        and _close(numerical.mixing, 0.2)
        and config.max_iterations >= 200
        and config.min_iterations >= 3
        and config.consecutive_passes >= 3
        and config.sigma_rtol <= 1.0e-8
        and config.sigma_atol <= 1.0e-12
        and config.green_rtol <= 1.0e-8
        and config.green_atol <= 1.0e-12
        and config.identity_rtol <= 1.0e-8
        and config.current_rtol <= 1.0e-5
        and config.current_atol <= 1.0e-10
        and config.tail_rtol <= 1.0e-6
        and config.mixing_floor <= 0.01
        and config.field_rtol <= 1.0e-8
        and config.field_atol <= 1.0e-12
        and config.backend_field_rtol <= 1.0e-10
        and config.backend_field_atol <= 1.0e-12
        and config.current_comparison_rtol <= 1.0e-10
        and config.current_comparison_atol <= 1.0e-12
        and config.refinement_rtol <= 5.0e-3
        and config.line_shape_rtol <= 5.0e-3
        and config.linewidth_atol <= 1.0e-12
        and config.shift_atol <= 1.0e-8
    )
    runtime_evidence_passed = bool(
        environment_evidence_valid
        and exact_release_revision
        and harmonic_preflight_passed
        and backend_validation_passed
        and outcome_certification_passed
        and fft_topology_passed
        and refinement_directions_passed
        and all_six_archived
    )
    return static_configuration_passed, runtime_evidence_passed


def _multidimensional_contract_passed(
    report: MultidimensionalReleaseReport,
    document: Mapping[str, Any],
) -> bool:
    """Check the complete locked contract independently of stored scalars."""

    static_configuration, runtime_evidence = (
        _multidimensional_contract_status(report, document)
    )
    return bool(static_configuration and runtime_evidence)


def _cost_accuracy_table(
    runs: Sequence[OpenThreeModeBornRun],
    multidimensional_document: Mapping[str, Any],
) -> list[dict[str, Any]]:
    multidimensional_timings = _collect_timing_values(
        multidimensional_document
    )
    top_level_total = multidimensional_document.get("total_elapsed_seconds")
    if (
        isinstance(top_level_total, (int, float, np.number))
        and not isinstance(top_level_total, (bool, np.bool_))
        and np.isfinite(float(top_level_total))
        and float(top_level_total) >= 0.0
    ):
        scba_totals = (float(top_level_total),)
        scba_timing_basis = "multidimensional_report.total_elapsed_seconds"
    else:
        scba_totals = tuple(
            float(item["wall_seconds"])
            for item in multidimensional_timings
        )
        scba_timing_basis = "all discovered timing records"
    return [
        {
            "accuracy_role": "exact elastic baseline for the supplied FC2",
            "asymptotic_dense_time": "O(N_omega*D^3)",
            "conserving": True,
            "measured": {
                "note": (
                    "No separate harmonic timing object is part of this "
                    "combined report."
                ),
                **_timing_summary(()),
            },
            "method": "harmonic_negf",
            "release_use": "harmonic baseline",
            "self_consistent": False,
        },
        {
            "accuracy_role": (
                "O(Psi^2) screening, pole oracle comparison, and SCBA "
                "initialization"
            ),
            "asymptotic_dense_time": "O(N_omega^2*D^4 + N_omega*D^3)",
            "conserving": False,
            "measured": _timing_summary(
                tuple(run.production_wall_seconds for run in runs)
            ),
            "method": "one_shot_born",
            "release_use": "screening_only_never_conductance_certified",
            "self_consistent": False,
        },
        {
            "accuracy_role": (
                "nonlinear conserving cubic-bubble fixed point within SCBA"
            ),
            "asymptotic_dense_time": (
                "O(N_iteration*(N_omega^2*D^4 + N_omega*D^3))"
            ),
            "conserving": True,
            "measured": {
                "aggregation_basis": scba_timing_basis,
                "note": (
                    "records include nested outcome/seed timings and can "
                    "overlap; aggregate statistics use the authoritative "
                    "top-level total when it is available."
                ),
                "records": multidimensional_timings,
                **_timing_summary(scba_totals),
            },
            "method": "self_consistent_born",
            "release_use": "required_interacting_conductance_tier",
            "self_consistent": True,
        },
    ]



def _environment_lock_summary(
    run_documents: Mapping[str, Mapping[str, Any]],
    multidimensional_document: Mapping[str, Any],
) -> dict[str, Any]:
    """Cross-check one explicit lock digest across all production evidence."""

    open_values = tuple(
        document.get("environment_lock_sha256")
        for document in run_documents.values()
    )
    open_complete = bool(
        open_values and all(_is_sha256(value) for value in open_values)
    )
    unique_open_values = set(open_values) if open_complete else set()
    open_consistent = bool(open_complete and len(unique_open_values) == 1)
    open_sha256 = (
        next(iter(unique_open_values)) if open_consistent else None
    )
    checkpoint_evidence = multidimensional_document.get(
        "checkpoint_evidence",
        {},
    )
    multidimensional_passed = bool(
        isinstance(checkpoint_evidence, Mapping)
        and checkpoint_evidence.get("passed") is True
    )
    multidimensional_sha256 = (
        checkpoint_evidence.get("environment_sha256")
        if isinstance(checkpoint_evidence, Mapping)
        else None
    )
    multidimensional_digest_valid = _is_sha256(multidimensional_sha256)
    matches = bool(
        open_consistent
        and multidimensional_passed
        and multidimensional_digest_valid
        and open_sha256 == multidimensional_sha256
    )
    return {
        "passed": matches,
        "open_run_count": len(open_values),
        "open_complete": open_complete,
        "open_consistent": open_consistent,
        "open_environment_sha256": open_sha256,
        "multidimensional_checkpoint_evidence_passed": (
            multidimensional_passed
        ),
        "multidimensional_environment_sha256": multidimensional_sha256,
        "multidimensional_environment_sha256_valid": (
            multidimensional_digest_valid
        ),
    }

def build_production_toy_gate_report(
    runs: Sequence[OpenThreeModeBornRun],
    multidimensional_report: MultidimensionalReleaseReport,
    *,
    validation_tier: ValidationTier = "pull_request",
) -> ProductionToyGateReport:
    """Build the combined report from completed, immutable solver objects.

    No solver or oracle is evaluated by this function.  The full release
    matrix is intentionally inferred from fixture values and array-backed
    results rather than from caller-supplied pass/fail flags.
    """

    if validation_tier not in {"pull_request", "release"}:
        raise ValueError(
            "validation_tier must be either 'pull_request' or 'release'."
        )
    if isinstance(runs, (str, bytes)) or not isinstance(runs, Sequence):
        raise TypeError("runs must be a sequence of OpenThreeModeBornRun.")
    selected_runs = tuple(runs)
    if not selected_runs:
        raise ValueError("runs must contain at least one comparison.")
    if any(not isinstance(run, OpenThreeModeBornRun) for run in selected_runs):
        raise TypeError("runs must contain only OpenThreeModeBornRun objects.")
    names = tuple(run.name for run in selected_runs)
    if len(set(names)) != len(names):
        raise ValueError("open-three-mode run names must be unique.")
    if not isinstance(multidimensional_report, MultidimensionalReleaseReport):
        raise TypeError(
            "multidimensional_report must be a MultidimensionalReleaseReport."
        )

    run_documents = {
        run.name: _run_document(run)
        for run in sorted(selected_runs, key=lambda item: item.name)
    }
    base_signature, coverage, refinements = _select_baseline(selected_runs)
    base_step, base_n_positive, base_eta = base_signature
    base_cutoff = base_step * base_n_positive
    base_coverage = {
        f"T={temperature:g},g={g:g}": (
            run.name if run is not None else None
        )
        for (temperature, g), run in coverage.items()
    }
    full_base_coverage = all(run is not None for run in coverage.values())

    fixture_contract_passed = bool(
        all(
            all(
                _close(actual, expected)
                for actual, expected in zip(
                    run.fixture.mode_frequencies,
                    _LOCKED_MODE_FREQUENCIES,
                )
            )
            and _close(
                run.fixture.lead_coupling,
                _LOCKED_LEAD_COUPLING,
            )
            for run in selected_runs
        )
    )

    sign_documents: list[dict[str, Any]] = []
    scaling_documents: list[dict[str, Any]] = []
    spontaneous_documents: list[dict[str, Any]] = []
    locked_base_runs = [
        run for run in coverage.values() if run is not None
    ]
    for temperature in _LOCKED_TEMPERATURES:
        for magnitude in _LOCKED_ABSOLUTE_COUPLINGS:
            positive = coverage[(temperature, magnitude)]
            negative = coverage[(temperature, -magnitude)]
            if positive is not None and negative is not None:
                comparison = _observable_pair_document(
                    run_documents[positive.name],
                    run_documents[negative.name],
                    expected_ratio=1.0,
                    parent_frequency=_parent_scale(positive.fixture),
                )
                sign_documents.append(
                    {
                        "g_magnitude": magnitude,
                        "negative_run": negative.name,
                        "positive_run": positive.name,
                        "temperature": temperature,
                        **comparison,
                    }
                )
        for sign in (-1.0, 1.0):
            lower = coverage[(temperature, sign * 0.01)]
            upper = coverage[(temperature, sign * 0.02)]
            if lower is not None and upper is not None:
                comparison = _observable_pair_document(
                    run_documents[lower.name],
                    run_documents[upper.name],
                    expected_ratio=4.0,
                    parent_frequency=_parent_scale(lower.fixture),
                )
                scaling_documents.append(
                    {
                        "lower_run": lower.name,
                        "sign": int(sign),
                        "temperature": temperature,
                        "upper_run": upper.name,
                        **comparison,
                    }
                )

    for g in (-0.02, -0.01, 0.01, 0.02):
        run = coverage[(0.0, g)]
        if run is None:
            continue
        document = run_documents[run.name]
        parent_frequency = _parent_scale(run.fixture)
        minimum = _SPONTANEOUS_LINEWIDTH_RATIO * parent_frequency
        oracle_linewidth = float(document["oracle"]["parent_linewidth"])
        production_linewidth = float(
            document["production_born"]["parent_linewidth"]
        )
        spontaneous_documents.append(
            {
                "g": g,
                "minimum_linewidth": minimum,
                "oracle_parent_linewidth": oracle_linewidth,
                "passed": bool(
                    oracle_linewidth > minimum
                    and production_linewidth > minimum
                ),
                "production_parent_linewidth": production_linewidth,
                "run": run.name,
            }
        )

    anchor = coverage[(0.7, 0.02)]
    refinement_documents: dict[str, Any] = {}
    if anchor is not None:
        for control in ("spacing", "cutoff", "eta_lead"):
            candidate = refinements[control]
            refinement_documents[control] = (
                _refinement_document(
                    anchor,
                    candidate,
                    run_documents,
                    control=control,
                )
                if candidate is not None
                else None
            )
    else:
        refinement_documents = {
            "spacing": None,
            "cutoff": None,
            "eta_lead": None,
        }

    locked_evidence_names = {
        run.name for run in locked_base_runs
    } | {
        run.name for run in refinements.values() if run is not None
    }
    agreement_passed = bool(
        locked_evidence_names
        and all(
            run_documents[name]["gates"][
                "oracle_production_agreement"
            ]["passed"]
            for name in locked_evidence_names
        )
    )
    oracle_method_passed = bool(
        locked_evidence_names
        and all(
            run_documents[name]["gates"]["oracle_method"]["passed"]
            for name in locked_evidence_names
        )
    )
    production_completed = bool(
        locked_evidence_names
        and all(
            run_documents[name]["gates"]["production_completed"]
            for name in locked_evidence_names
        )
    )
    passivity_passed = bool(
        locked_evidence_names
        and all(
            run_documents[name]["gates"]["passivity"]["passed"]
            for name in locked_evidence_names
        )
    )
    weak_coupling_passed = bool(
        locked_evidence_names
        and all(
            run_documents[name]["gates"]["weak_coupling"]["passed"]
            for name in locked_evidence_names
        )
    )
    provenance_complete = bool(
        locked_evidence_names
        and all(
            run_documents[name]["provenance_complete"]
            for name in locked_evidence_names
        )
    )
    identities = {
        _canonical_json(run_documents[name]["provenance"]["identity"])
        for name in locked_evidence_names
    }
    code_identity_consistent = bool(
        locked_evidence_names and len(identities) == 1
    )
    open_identity = (
        run_documents[next(iter(locked_evidence_names))]["provenance"][
            "identity"
        ]
        if code_identity_consistent
        else None
    )
    sign_coverage = len(sign_documents) == (
        len(_LOCKED_TEMPERATURES) * len(_LOCKED_ABSOLUTE_COUPLINGS)
    )
    sign_passed = bool(
        sign_coverage
        and all(item["passed"] for item in sign_documents)
    )
    scaling_coverage = len(scaling_documents) == (
        2 * len(_LOCKED_TEMPERATURES)
    )
    scaling_passed = bool(
        scaling_coverage
        and all(item["passed"] for item in scaling_documents)
    )
    spontaneous_coverage = len(spontaneous_documents) == 4
    spontaneous_passed = bool(
        spontaneous_coverage
        and all(item["passed"] for item in spontaneous_documents)
    )
    refinement_coverage = all(
        refinement_documents.get(control) is not None
        for control in ("spacing", "cutoff", "eta_lead")
    )
    refinement_passed = bool(
        refinement_coverage
        and all(
            refinement_documents[control]["passed"]
            for control in ("spacing", "cutoff", "eta_lead")
        )
    )
    oracle_gate_passed = bool(
        fixture_contract_passed
        and full_base_coverage
        and oracle_method_passed
        and agreement_passed
        and production_completed
        and passivity_passed
        and weak_coupling_passed
        and sign_passed
        and scaling_passed
        and spontaneous_passed
        and refinement_passed
        and provenance_complete
        and code_identity_consistent
    )

    multidimensional_document = multidimensional_report.as_dict()
    environment_lock = _environment_lock_summary(
        run_documents,
        multidimensional_document,
    )
    (
        multidimensional_static_configuration_passed,
        multidimensional_runtime_evidence_passed,
    ) = _multidimensional_contract_status(
        multidimensional_report,
        multidimensional_document,
    )
    multidimensional_contract_passed = bool(
        multidimensional_static_configuration_passed
        and multidimensional_runtime_evidence_passed
    )
    multidimensional_provenance = multidimensional_document.get(
        "provenance",
        {},
    )
    multidimensional_provenance_passed = bool(
        isinstance(multidimensional_provenance, Mapping)
        and multidimensional_provenance.get("complete") is True
        and multidimensional_provenance.get("clean") is True
        and multidimensional_provenance.get("consistent") is True
        and multidimensional_provenance.get("passed") is True
    )
    multidimensional_weak_coupling_passed = bool(
        isinstance(multidimensional_document.get("weak_coupling"), Mapping)
        and multidimensional_document["weak_coupling"].get("coverage") is True
        and multidimensional_document["weak_coupling"].get("passed") is True
        and _close(
            multidimensional_document["weak_coupling"].get("limit", -1.0),
            _WEAK_COUPLING_LIMIT,
        )
    )
    open_revision = (
        open_identity.get("git_revision")
        if isinstance(open_identity, Mapping)
        else None
    )
    exact_release_revision_passed = bool(
        validation_tier != "release"
        or (
            isinstance(open_revision, str)
            and len(open_revision) == 40
            and all(
                character in "0123456789abcdef"
                for character in open_revision
            )
            and isinstance(multidimensional_provenance, Mapping)
            and isinstance(
                multidimensional_provenance.get("identity"),
                Mapping,
            )
            and multidimensional_provenance["identity"].get(
                "git_revision"
            )
            == open_revision
        )
    )
    cross_platform_identity_consistent = bool(
        provenance_complete
        and code_identity_consistent
        and multidimensional_provenance_passed
        and environment_lock["passed"]
        and open_identity == multidimensional_provenance.get("identity")
    )
    harmonic_preflight_document = multidimensional_document.get(
        "harmonic_preflight",
        {},
    )
    multidimensional_harmonic_preflight_passed = bool(
        isinstance(harmonic_preflight_document, Mapping)
        and harmonic_preflight_document.get("passed") is True
        and harmonic_preflight_document.get("all_seven_archived") is True
    )
    backend_validation_document = multidimensional_document.get(
        "backend_validation",
        {},
    )
    multidimensional_backend_validation_passed = bool(
        isinstance(backend_validation_document, Mapping)
        and backend_validation_document.get("passed") is True
        and backend_validation_document.get(
            "full_grid_serial_threaded_comparison_executed"
        )
        is False
        and backend_validation_document.get("scope")
        == "bounded_converged_scba_backend_oracle"
    )
    outcome_certification_document = multidimensional_document.get(
        "outcome_certification",
        {},
    )
    multidimensional_outcome_certification_passed = bool(
        isinstance(outcome_certification_document, Mapping)
        and outcome_certification_document.get("passed") is True
        and outcome_certification_document.get("coverage") is True
    )
    multidimensional_comparison_kinds = {
        comparison.kind
        for comparison in multidimensional_report.comparisons
    }
    multidimensional_comparison_coverage = (
        multidimensional_comparison_kinds
        == {
            "zero_vs_born",
            "direct_vs_continuation",
            "numerical_refinement",
        }
    )
    multidimensional_underlying_evidence = bool(
        multidimensional_contract_passed
        and multidimensional_report.outcomes
        and all(
            outcome.successful
            and outcome.result is not None
            and outcome.result.method == "scba"
            and outcome.result.diagnostics.passed
            for outcome in multidimensional_report.outcomes
        )
        and multidimensional_report.comparisons
        and multidimensional_comparison_coverage
        and multidimensional_harmonic_preflight_passed
        and multidimensional_backend_validation_passed
        and multidimensional_outcome_certification_passed
        and all(
            comparison.passed or not comparison.certification_critical
            for comparison in multidimensional_report.comparisons
        )
        and multidimensional_provenance_passed
        and multidimensional_weak_coupling_passed
        and multidimensional_report.conservation_checks
        and all(
            check.passed
            for check in multidimensional_report.conservation_checks
        )
    )
    multidimensional_certified = bool(
        multidimensional_report.release_certified
        and multidimensional_underlying_evidence
    )
    release_certified = bool(
        validation_tier == "release"
        and oracle_gate_passed
        and multidimensional_certified
        and environment_lock["passed"]
        and cross_platform_identity_consistent
        and exact_release_revision_passed
    )
    failures: list[str] = []
    if validation_tier != "release":
        failures.append(
            "pull-request validation data are non-certifying by definition"
        )
    for condition, message in (
        (
            fixture_contract_passed,
            "one or more runs do not use the locked frequencies/coupling",
        ),
        (full_base_coverage, "locked T/g/sign base coverage is incomplete"),
        (
            oracle_method_passed,
            "adaptive eta-free oracle method/error gate failed",
        ),
        (agreement_passed, "production Born/oracle agreement gate failed"),
        (
            production_completed,
            "one or more locked one-shot Born calculations are incomplete",
        ),
        (passivity_passed, "open-three-mode passivity gate failed"),
        (weak_coupling_passed, "open-three-mode weak-coupling gate failed"),
        (sign_passed, "g-to-minus-g invariance coverage or gate failed"),
        (scaling_passed, "g-squared scaling coverage or gate failed"),
        (
            spontaneous_passed,
            "zero-temperature spontaneous-decay coverage or gate failed",
        ),
        (
            refinement_passed,
            "spacing/cutoff/eta one-control refinement gate failed",
        ),
        (
            provenance_complete,
            "open-three-mode code/input provenance is incomplete",
        ),
        (
            code_identity_consistent,
            "open-three-mode runs do not share one code identity",
        ),
        (
            multidimensional_static_configuration_passed,
            "multidimensional SCBA configuration does not match the locked "
            "release contract",
        ),
        (
            multidimensional_runtime_evidence_passed,
            "multidimensional SCBA runtime evidence does not satisfy the "
            "locked release contract",
        ),
        (
            multidimensional_comparison_coverage,
            "multidimensional SCBA comparison coverage is incomplete",
        ),
        (
            multidimensional_harmonic_preflight_passed,
            "seven-point harmonic multidimensional preflight failed",
        ),
        (
            multidimensional_backend_validation_passed,
            "bounded converged-SCBA backend validation failed",
        ),
        (
            multidimensional_outcome_certification_passed,
            "required multidimensional outcomes are not release-certified",
        ),
        (
            multidimensional_provenance_passed,
            "multidimensional source/input/environment provenance failed",
        ),
        (
            environment_lock["passed"],
            (
                "every open-system run must share the explicit environment "
                "lock SHA256 recorded by valid multidimensional checkpoints"
            ),
        ),
        (
            multidimensional_weak_coupling_passed,
            "multidimensional nine-DOF weak-coupling gate failed",
        ),
        (
            exact_release_revision_passed,
            "release evidence does not share one exact 40-character commit",
        ),
        (
            cross_platform_identity_consistent,
            (
                "open and multidimensional runs do not share one clean "
                "code/environment identity"
            ),
        ),
        (
            multidimensional_certified,
            "multidimensional SCBA report is not release-certified",
        ),
    ):
        if not condition:
            failures.append(message)

    core_document: dict[str, Any] = {
        "acceptance": {
            "linewidth_absolute_ratio": _LINEWIDTH_ABSOLUTE_RATIO,
            "near_zero_shift_absolute_ratio": (
                _NEAR_ZERO_SHIFT_ABSOLUTE_RATIO
            ),
            "passivity_relative_floor": _PASSIVITY_RELATIVE_FLOOR,
            "relative_tolerance": _RELATIVE_TOLERANCE,
            "spontaneous_linewidth_ratio": (
                _SPONTANEOUS_LINEWIDTH_RATIO
            ),
            "weak_coupling_limit": _WEAK_COUPLING_LIMIT,
        },
        "cost_accuracy": _cost_accuracy_table(
            selected_runs,
            multidimensional_document,
        ),
        "environment_lock": environment_lock,
        "multidimensional_scba": multidimensional_document,
        "open_three_mode": {
            "baseline_controls": {
                "cutoff": base_cutoff,
                "eta_lead": base_eta,
                "grid_n_positive": base_n_positive,
                "grid_step": base_step,
            },
            "base_coverage": base_coverage,
            "refinements": refinement_documents,
            "runs": [
                run_documents[name] for name in sorted(run_documents)
            ],
            "scaling_checks": scaling_documents,
            "sign_checks": sign_documents,
            "spontaneous_decay_checks": spontaneous_documents,
        },
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "summary": {
            "code_identity_consistent": code_identity_consistent,
            "exact_release_revision_passed": (
                exact_release_revision_passed
            ),
            "cross_platform_identity_consistent": (
                cross_platform_identity_consistent
            ),
            "environment_lock_consistent": environment_lock["passed"],
            "failures": failures,
            "fixture_contract_passed": fixture_contract_passed,
            "full_locked_base_coverage": full_base_coverage,
            "multidimensional_release_certified": (
                multidimensional_certified
            ),
            "multidimensional_harmonic_preflight_passed": (
                multidimensional_harmonic_preflight_passed
            ),
            "multidimensional_backend_validation_passed": (
                multidimensional_backend_validation_passed
            ),
            "multidimensional_outcome_certification_passed": (
                multidimensional_outcome_certification_passed
            ),
            "multidimensional_contract_passed": (
                multidimensional_contract_passed
            ),
            "multidimensional_static_configuration_passed": (
                multidimensional_static_configuration_passed
            ),
            "multidimensional_runtime_evidence_passed": (
                multidimensional_runtime_evidence_passed
            ),
            "multidimensional_comparison_coverage": (
                multidimensional_comparison_coverage
            ),
            "multidimensional_underlying_evidence_complete": (
                multidimensional_underlying_evidence
            ),
            "multidimensional_provenance_passed": (
                multidimensional_provenance_passed
            ),
            "multidimensional_weak_coupling_passed": (
                multidimensional_weak_coupling_passed
            ),
            "open_three_mode_agreement_passed": agreement_passed,
            "open_three_mode_oracle_method_passed": oracle_method_passed,
            "open_three_mode_passivity_passed": passivity_passed,
            "open_three_mode_production_completed": production_completed,
            "open_three_mode_provenance_complete": provenance_complete,
            "open_three_mode_weak_coupling_passed": weak_coupling_passed,
            "oracle_gate_passed": oracle_gate_passed,
            "refinement_coverage": refinement_coverage,
            "refinement_passed": refinement_passed,
            "release_certified": release_certified,
            "scaling_coverage": scaling_coverage,
            "scaling_passed": scaling_passed,
            "sign_coverage": sign_coverage,
            "sign_invariance_passed": sign_passed,
            "spontaneous_decay_coverage": spontaneous_coverage,
            "spontaneous_decay_passed": spontaneous_passed,
            "validation_tier": validation_tier,
        },
    }
    report_sha256 = hashlib.sha256(
        _canonical_json(core_document).encode("utf-8")
    ).hexdigest()
    return ProductionToyGateReport(
        document={**core_document, "report_sha256": report_sha256}
    )


__all__ = [
    "OpenThreeModeBornRun",
    "ProductionToyGateReport",
    "build_production_toy_gate_report",
]
