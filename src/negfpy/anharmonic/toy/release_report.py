"""Machine-readable dense-v0.1 refinement and release-gate reports.

The report builder consumes an already completed :class:`RefinementStudy`; it
never reruns or modifies a calculation.  Numerical refinement and release
certification are intentionally separate:

* ``refinement_passed`` records convergence, current, selected-peak, and
  cutoff-tail checks against explicit thresholds.
* ``release_certified`` additionally requires directional coverage of all
  four locked refinements, internally derived peak evidence, nonzero cubic
  vertices, complete input/source/version provenance, and independent
  release certification of every stored solver result.

Consequently, a coarse toy study may demonstrate a stable current without
being mislabeled as a release-certified calculation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from ..diagnostics import keldysh_tail_ratio
from .refinement import (
    RefinementObservation,
    RefinementStudy,
)


_SCHEMA = "negfpy.anharmonic.dense_v0_1_refinement"
_SCHEMA_VERSION = 2
_REQUIRED_CONTROLS = ("spacing", "cutoff", "eta_lead", "mixing")
_CERTIFIED_MAX_CURRENT_CHANGE = 5.0e-3
_CERTIFIED_MAX_PEAK_CHANGE = 5.0e-3
_CERTIFIED_MAX_TAIL_RATIO = 1.0e-6
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


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


@dataclass(frozen=True, slots=True)
class DenseV01Acceptance:
    """Explicit acceptance thresholds for a dense-v0.1 refinement report.

    Defaults reproduce the numerical-refinement gates in the platform plan.
    ``require_selected_peak_changes`` remains enabled because current
    convergence alone does not establish spectral convergence.
    """

    max_current_relative_change: float = 5.0e-3
    max_selected_peak_relative_change: float = 5.0e-3
    max_sigma_tail_ratio: float = 1.0e-6
    require_converged: bool = True
    require_selected_peak_changes: bool = True

    def __post_init__(self) -> None:
        for name in (
            "max_current_relative_change",
            "max_selected_peak_relative_change",
            "max_sigma_tail_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        for name in (
            "require_converged",
            "require_selected_peak_changes",
        ):
            value = getattr(self, name)
            if not isinstance(value, (bool, np.bool_)):
                raise TypeError(f"{name} must be boolean.")
            object.__setattr__(self, name, bool(value))


@dataclass(frozen=True, slots=True)
class DenseV01PeakMeasurement:
    """Request one release-safe peak observable derived from stored results.

    Values, units, comparison scales, and source hashes are all computed by
    the report builder.  Callers cannot inject peak values or an arbitrary
    normalization floor.
    """

    observable: str = "left_contact_spectrum_peak_height"

    def __post_init__(self) -> None:
        if not isinstance(self.observable, str):
            raise TypeError("observable must be a string.")
        observable = self.observable.strip()
        allowed = {
            "left_contact_spectrum_peak_height",
            "right_contact_spectrum_peak_height",
            "collision_spectrum_peak_height",
            "sigma_pp_broadening_peak_frequency",
        }
        if observable not in allowed:
            raise ValueError(
                "observable must be one of: " + ", ".join(sorted(allowed))
            )
        object.__setattr__(self, "observable", observable)


def _json_safe(value: Any, *, path: str = "$") -> Any:
    """Convert supported scientific-Python values to strict JSON data.

    Non-finite floating-point values are encoded as strings rather than the
    non-standard JSON tokens accepted by Python's permissive default encoder.
    Unsupported objects fail loudly so a report cannot silently depend on an
    unstable ``repr``.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path=path)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
        return value
    if isinstance(value, complex):
        return {
            "__complex__": [
                _json_safe(float(value.real), path=f"{path}.real"),
                _json_safe(float(value.imag), path=f"{path}.imag"),
            ]
        }
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path=path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        converted: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has a non-string mapping key: {key!r}.")
            converted[key] = _json_safe(item, path=f"{path}.{key}")
        return {key: converted[key] for key in sorted(converted)}
    if isinstance(value, (set, frozenset)):
        converted = [_json_safe(item, path=f"{path}[]") for item in value]
        return sorted(
            converted,
            key=lambda item: json.dumps(
                item,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported non-JSON value of type "
        f"{type(value).__name__}."
    )


def _canonical_json(document: Mapping[str, Any], *, indent: int | None = None) -> str:
    return json.dumps(
        _json_safe(document),
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


@dataclass(frozen=True, slots=True, eq=False)
class DenseV01RefinementReport:
    """Immutable wrapper around a strict, deterministically serialized report."""

    document: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.document, Mapping):
            raise TypeError("document must be a mapping.")
        normalized = _json_safe(self.document)
        # This also checks that no permissive NaN/Infinity token remains.
        _canonical_json(normalized)
        object.__setattr__(self, "document", _deep_freeze(normalized))

    @property
    def refinement_passed(self) -> bool:
        return bool(self.document["summary"]["refinement_passed"])

    @property
    def release_certified(self) -> bool:
        return bool(self.document["summary"]["release_certified"])

    @property
    def sha256(self) -> str:
        return str(self.document["report_sha256"])

    def as_dict(self) -> dict[str, Any]:
        """Return a detached JSON-native dictionary."""

        return json.loads(self.to_json())

    def to_json(self, *, indent: int | None = None) -> str:
        """Return strict JSON with deterministic key ordering."""

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
        """Write the immutable report and return the exact destination path."""

        destination = Path(path)
        destination.write_text(
            self.to_json(indent=indent) + "\n",
            encoding="utf-8",
        )
        return destination


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _input_hashes(observation: RefinementObservation) -> tuple[dict[str, str], bool]:
    raw = observation.result.metadata.get("input_hashes", {})
    if not isinstance(raw, Mapping):
        return {}, False
    hashes = {
        str(key): str(value)
        for key, value in raw.items()
        if isinstance(key, str)
    }
    complete = (
        _REQUIRED_INPUT_HASHES.issubset(hashes)
        and all(_is_sha256(hashes[name]) for name in _REQUIRED_INPUT_HASHES)
    )
    return hashes, complete


def _code_provenance(
    observation: RefinementObservation,
) -> tuple[dict[str, Any], bool]:
    metadata = observation.result.metadata
    source_hash = metadata.get("source_tree_sha256")
    revision = metadata.get("git_revision")
    declared = metadata.get("declared_package_version")
    installed = metadata.get("installed_package_version")
    consistent = metadata.get("package_version_consistent")
    complete = bool(
        _is_sha256(source_hash)
        and isinstance(revision, str)
        and len(revision) in {40, 64}
        and all(character in "0123456789abcdef" for character in revision)
        and isinstance(declared, str)
        and bool(declared)
        and declared != "unknown"
        and isinstance(installed, str)
        and bool(installed)
        and installed != "unknown"
        and consistent is True
    )
    return (
        {
            "declared_package_version": declared,
            "git_dirty": metadata.get("git_dirty"),
            "git_revision": revision,
            "installed_package_version": installed,
            "package_version_consistent": consistent,
            "source_tree_sha256": source_hash,
        },
        complete,
    )


def _gate(
    *,
    value: Any,
    threshold: Any = None,
    comparison: str,
    passed: bool,
    evaluated: bool = True,
) -> dict[str, Any]:
    return {
        "comparison": comparison,
        "evaluated": bool(evaluated),
        "passed": bool(passed),
        "threshold": threshold,
        "value": value,
    }


def _control(name: str, reference: float, candidate: float) -> dict[str, Any]:
    ratio = candidate / reference
    changed = not math.isclose(
        candidate,
        reference,
        rel_tol=0.0,
        abs_tol=0.0,
    )
    if name == "spacing":
        requirement = "candidate/reference == 0.5"
        qualifies = math.isclose(
            ratio,
            0.5,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        )
    elif name == "cutoff":
        requirement = "candidate/reference > 1"
        qualifies = ratio > 1.0
    elif name == "eta_lead":
        requirement = "candidate/reference <= 0.5"
        qualifies = ratio <= 0.5
    elif name == "mixing":
        requirement = "candidate/reference != 1"
        qualifies = changed
    else:  # pragma: no cover - private fixed call set
        raise ValueError(f"unknown refinement control {name!r}.")
    return {
        "candidate": candidate,
        "changed": changed,
        "qualifies_for_release_coverage": qualifies,
        "release_requirement": requirement,
        "ratio": ratio,
        "reference": reference,
    }


def _observation_document(
    observation: RefinementObservation,
) -> tuple[dict[str, Any], bool]:
    hashes, input_provenance_complete = _input_hashes(observation)
    code_provenance, code_provenance_complete = _code_provenance(observation)
    provenance_complete = bool(
        input_provenance_complete and code_provenance_complete
    )
    result = observation.result
    trusted_tail = keldysh_tail_ratio(result.sigma_pp)
    trusted_eta = float(result.metadata["eta_lead"])
    trusted_mixing = float(result.metadata["mixing_initial"])
    summary_consistent = bool(
        observation.grid_step == result.grid.step
        and observation.grid_cutoff == result.grid.cutoff
        and observation.n_frequency == result.grid.n_points
        and observation.eta_lead == trusted_eta
        and observation.mixing == trusted_mixing
        and observation.left_current == result.currents.left
        and observation.right_current == result.currents.right
        and observation.sigma_tail_ratio == trusted_tail
        and observation.converged == result.converged
        and observation.release_certified == result.release_certified
        and observation.iterations == len(result.history)
    )
    return (
        {
            "code_provenance": code_provenance,
            "code_provenance_complete": code_provenance_complete,
            "converged": result.converged,
            "currents": {
                "collision_power": result.currents.collision_power,
                "contact_balance": result.currents.contact_balance,
                "full_balance": result.currents.full_balance,
                "left": result.currents.left,
                "right": result.currents.right,
            },
            "diagnostics": {
                "failures": list(result.diagnostics.failures),
                "metrics": dict(result.diagnostics.metrics),
                "passed": result.diagnostics.passed,
                "release_certified": result.diagnostics.release_certified,
            },
            "grid": {
                "cutoff": result.grid.cutoff,
                "n_frequency": result.grid.n_points,
                "step": result.grid.step,
            },
            "input_hashes": hashes,
            "input_provenance_complete": input_provenance_complete,
            "iterations": len(result.history),
            "method": result.method,
            "name": observation.name,
            "provenance_complete": provenance_complete,
            "result_metadata": dict(result.metadata),
            "release_certified": result.release_certified,
            "uses_nonzero_cubic_vertex": bool(
                result.metadata.get("vertex_nonzero", False)
            ),
            "solver_controls": {
                "eta_lead": trusted_eta,
                "mixing": trusted_mixing,
            },
            "sigma_tail_ratio": trusted_tail,
            "study_summary_consistent": summary_consistent,
        },
        provenance_complete,
    )


def _structured_peak_measurements(
    reference: RefinementObservation,
    candidate: RefinementObservation,
    name: str,
    selected_peak_measurements: Mapping[
        str,
        DenseV01PeakMeasurement | Sequence[DenseV01PeakMeasurement],
    ]
    | None,
) -> tuple[list[dict[str, Any]], float | None, bool]:
    if selected_peak_measurements is None or name not in selected_peak_measurements:
        return [], None, False
    raw = selected_peak_measurements[name]
    if isinstance(raw, DenseV01PeakMeasurement):
        measurements = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        measurements = tuple(raw)
    else:
        raise TypeError(
            f"selected_peak_measurements[{name!r}] must be a "
            "DenseV01PeakMeasurement or a sequence of them."
        )
    if not measurements:
        raise ValueError(
            f"selected_peak_measurements[{name!r}] cannot be empty."
        )
    if any(
        not isinstance(item, DenseV01PeakMeasurement)
        for item in measurements
    ):
        raise TypeError(
            f"selected_peak_measurements[{name!r}] contains a non-measurement."
        )

    def derive(
        observation: RefinementObservation,
        observable: str,
    ) -> tuple[float, str, str]:
        currents = observation.result.currents
        if observable in {
            "left_contact_spectrum_peak_height",
            "right_contact_spectrum_peak_height",
        }:
            spectra = currents.contact_spectra
            if spectra is None or isinstance(spectra, Mapping):
                raise ValueError(
                    f"{observation.name!r} does not store array-valued "
                    "contact spectra."
                )
            array = np.asarray(spectra, dtype=np.float64)
            if array.ndim != 2 or array.shape[0] != 2 or array.shape[1] == 0:
                raise ValueError(
                    f"{observation.name!r} contact spectra must have shape "
                    "(2, n_positive)."
                )
            row = 0 if observable.startswith("left_") else 1
            source = array[row]
            value = float(np.max(np.abs(source)))
            units = "physical current per solver angular frequency"
        elif observable == "collision_spectrum_peak_height":
            spectrum = currents.collision_spectrum
            if spectrum is None:
                raise ValueError(
                    f"{observation.name!r} does not store a collision spectrum."
                )
            source = np.asarray(spectrum, dtype=np.float64).reshape(-1)
            if source.size == 0:
                raise ValueError(
                    f"{observation.name!r} collision spectrum is empty."
                )
            value = float(np.max(np.abs(source)))
            units = "physical current per solver angular frequency"
        else:
            positive = observation.result.grid.positive_indices
            retarded = observation.result.sigma_pp.retarded[positive]
            advanced = np.swapaxes(retarded.conj(), -1, -2)
            broadening = 1j * (retarded - advanced)
            source = np.linalg.norm(
                broadening,
                ord="fro",
                axis=(-2, -1),
            )
            if source.size < 3 or not np.any(source > 0.0):
                raise ValueError(
                    f"{observation.name!r} does not contain a resolved "
                    "positive-frequency interaction-broadening peak."
                )
            peak_index = int(np.argmax(source))
            if peak_index == 0 or peak_index == source.size - 1:
                raise ValueError(
                    f"{observation.name!r} interaction-broadening peak lies "
                    "at the frequency-grid boundary."
                )
            left_value, center_value, right_value = source[
                peak_index - 1 : peak_index + 2
            ]
            curvature = left_value - 2.0 * center_value + right_value
            if not np.isfinite(curvature) or curvature >= 0.0:
                raise ValueError(
                    f"{observation.name!r} interaction-broadening maximum "
                    "does not admit a concave local quadratic fit."
                )
            offset = 0.5 * (left_value - right_value) / curvature
            if not np.isfinite(offset) or abs(offset) > 1.0:
                raise ValueError(
                    f"{observation.name!r} interaction-broadening quadratic "
                    "vertex lies outside its three-point bracket."
                )
            frequency = observation.result.grid.values[
                positive[peak_index]
            ]
            value = float(
                frequency + offset * observation.result.grid.step
            )
            if value <= 0.0:
                raise ValueError(
                    f"{observation.name!r} fitted interaction-broadening "
                    "peak frequency must be positive."
                )
            units = "solver angular frequency"
        digest = hashlib.sha256()
        for values in (observation.result.grid.values, source):
            contiguous = np.ascontiguousarray(values)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return (
            value,
            units,
            digest.hexdigest(),
        )

    documents: list[dict[str, Any]] = []
    changes: list[float] = []
    for measurement in measurements:
        reference_value, units, reference_hash = derive(
            reference,
            measurement.observable,
        )
        candidate_value, candidate_units, candidate_hash = derive(
            candidate,
            measurement.observable,
        )
        if candidate_units != units:  # pragma: no cover - fixed extractor
            raise RuntimeError("internally derived peak units disagree.")
        comparison_scale = max(abs(reference_value), abs(candidate_value))
        relative_change = (
            abs(candidate_value - reference_value) / comparison_scale
            if comparison_scale > 0.0
            else float("inf")
        )
        changes.append(relative_change)
        documents.append(
            {
                "candidate_source_sha256": candidate_hash,
                "candidate_value": candidate_value,
                "comparison_scale": comparison_scale,
                "observable": measurement.observable,
                "reference_source_sha256": reference_hash,
                "reference_value": reference_value,
                "relative_change": relative_change,
                "units": units,
            }
        )
    return (
        documents,
        max(changes),
        True,
    )


def _legacy_selected_peak_value(
    name: str,
    selected_peak_relative_changes: Mapping[str, float] | None,
) -> tuple[float | None, bool]:
    if (
        selected_peak_relative_changes is None
        or name not in selected_peak_relative_changes
    ):
        return None, False
    return (
        _finite_nonnegative(
            selected_peak_relative_changes[name],
            name=f"selected_peak_relative_changes[{name!r}]",
        ),
        True,
    )


def _comparison_document(
    reference: RefinementObservation,
    candidate: RefinementObservation,
    *,
    acceptance: DenseV01Acceptance,
    selected_peak_measurements: Mapping[
        str,
        DenseV01PeakMeasurement | Sequence[DenseV01PeakMeasurement],
    ]
    | None,
    selected_peak_relative_changes: Mapping[str, float] | None,
    reference_provenance_complete: bool,
    candidate_provenance_complete: bool,
) -> dict[str, Any]:
    reference_result = reference.result
    candidate_result = candidate.result
    left_change = (
        candidate_result.currents.left - reference_result.currents.left
    )
    right_change = (
        candidate_result.currents.right - reference_result.currents.right
    )
    maximum_absolute = max(abs(left_change), abs(right_change))
    current_floor = max(
        float(reference_result.metadata["current_atol_physical"]),
        float(candidate_result.metadata["current_atol_physical"]),
    )
    current_scale = max(
        abs(reference_result.currents.left),
        abs(reference_result.currents.right),
        abs(candidate_result.currents.left),
        abs(candidate_result.currents.right),
        current_floor,
    )
    maximum_relative = maximum_absolute / current_scale
    reference_tail = keldysh_tail_ratio(reference_result.sigma_pp)
    candidate_tail = keldysh_tail_ratio(candidate_result.sigma_pp)
    peak_measurement_documents, peak_change, peak_verified = (
        _structured_peak_measurements(
            reference,
            candidate,
            candidate.name,
            selected_peak_measurements,
        )
    )
    legacy_peak_change, legacy_peak_evaluated = _legacy_selected_peak_value(
        candidate.name,
        selected_peak_relative_changes,
    )
    if peak_verified and legacy_peak_evaluated:
        raise ValueError(
            f"candidate {candidate.name!r} has both structured and legacy "
            "selected-peak evidence."
        )
    if not peak_verified:
        peak_change = legacy_peak_change
    peak_evaluated = bool(peak_verified or legacy_peak_evaluated)
    current_pass = (
        maximum_relative
        <= acceptance.max_current_relative_change
    )
    tail_value = max(reference_tail, candidate_tail)
    tail_pass = tail_value <= acceptance.max_sigma_tail_ratio
    convergence_pass = (
        (reference_result.converged and candidate_result.converged)
        if acceptance.require_converged
        else True
    )
    peak_pass = (
        peak_evaluated
        and peak_change is not None
        and peak_change <= acceptance.max_selected_peak_relative_change
    )
    if not acceptance.require_selected_peak_changes and not peak_evaluated:
        peak_pass = True
    provenance_pass = (
        reference_provenance_complete and candidate_provenance_complete
    )
    numerical_pass = bool(
        current_pass
        and tail_pass
        and convergence_pass
        and peak_pass
    )
    result_certification_pass = (
        reference_result.release_certified
        and candidate_result.release_certified
    )

    return {
        "candidate": candidate.name,
        "controls": {
            "cutoff": _control(
                "cutoff",
                reference_result.grid.cutoff,
                candidate_result.grid.cutoff,
            ),
            "eta_lead": _control(
                "eta_lead",
                float(reference_result.metadata["eta_lead"]),
                float(candidate_result.metadata["eta_lead"]),
            ),
            "mixing": _control(
                "mixing",
                float(reference_result.metadata["mixing_initial"]),
                float(candidate_result.metadata["mixing_initial"]),
            ),
            "spacing": _control(
                "spacing",
                reference_result.grid.step,
                candidate_result.grid.step,
            ),
        },
        "deltas": {
            "convergence_changed": (
                candidate_result.converged != reference_result.converged
            ),
            "iteration_change": (
                len(candidate_result.history) - len(reference_result.history)
            ),
            "left_current_change": left_change,
            "max_current_absolute_change": maximum_absolute,
            "max_current_relative_change": maximum_relative,
            "right_current_change": right_change,
            "sigma_tail_absolute_change": abs(
                candidate_tail - reference_tail
            ),
            "sigma_tail_change": candidate_tail - reference_tail,
        },
        "gates": {
            "convergence": _gate(
                value={
                    "candidate": candidate_result.converged,
                    "reference": reference_result.converged,
                },
                comparison="both true when required",
                passed=convergence_pass,
            ),
            "current_refinement": _gate(
                value=maximum_relative,
                threshold=acceptance.max_current_relative_change,
                comparison="<=",
                passed=current_pass,
            ),
            "input_provenance": _gate(
                value={
                    "candidate": candidate_provenance_complete,
                    "reference": reference_provenance_complete,
                },
                comparison="both complete SHA256 sets",
                passed=provenance_pass,
            ),
            "result_release_certification": _gate(
                value={
                    "candidate": candidate_result.release_certified,
                    "reference": reference_result.release_certified,
                },
                comparison="both true when required",
                passed=result_certification_pass,
            ),
            "selected_peak_refinement": _gate(
                value=peak_change,
                threshold=acceptance.max_selected_peak_relative_change,
                comparison="<=",
                passed=peak_pass,
                evaluated=peak_evaluated,
            )
            | {
                "evidence_verified": peak_verified,
                "legacy_relative_delta_only": legacy_peak_evaluated,
                "measurements": peak_measurement_documents,
            },
            "self_energy_tail": _gate(
                value=tail_value,
                threshold=acceptance.max_sigma_tail_ratio,
                comparison="<=",
                passed=tail_pass,
            ),
        },
        "refinement_passed": numerical_pass,
        "release_certified": bool(
            numerical_pass
            and result_certification_pass
            and peak_verified
        ),
    }


def build_dense_v01_refinement_report(
    study: RefinementStudy,
    *,
    acceptance: DenseV01Acceptance | None = None,
    selected_peak_measurements: Mapping[
        str,
        DenseV01PeakMeasurement | Sequence[DenseV01PeakMeasurement],
    ]
    | None = None,
    selected_peak_relative_changes: Mapping[str, float] | None = None,
) -> DenseV01RefinementReport:
    """Build a deterministic, JSON-safe refinement/release report.

    ``selected_peak_measurements`` stores the actual reference and candidate
    values and lets this builder compute the relative changes.  The legacy
    ``selected_peak_relative_changes`` argument is retained for diagnostic
    reports, but its caller-computed deltas are never accepted as release
    evidence.
    """

    if not isinstance(study, RefinementStudy):
        raise TypeError("study must be a RefinementStudy.")
    if acceptance is None:
        acceptance = DenseV01Acceptance()
    if not isinstance(acceptance, DenseV01Acceptance):
        raise TypeError("acceptance must be a DenseV01Acceptance.")
    for argument_name, values in (
        ("selected_peak_measurements", selected_peak_measurements),
        ("selected_peak_relative_changes", selected_peak_relative_changes),
    ):
        if values is None:
            continue
        if not isinstance(values, Mapping):
            raise TypeError(f"{argument_name} must be a mapping.")
        unknown = set(values) - {
            item.name for item in study.candidates
        }
        if unknown:
            raise ValueError(
                f"{argument_name} contains unknown candidates: "
                + ", ".join(sorted(str(value) for value in unknown))
            )
    if selected_peak_measurements is not None and (
        set(selected_peak_measurements)
        & set(selected_peak_relative_changes or {})
    ):
        overlap = set(selected_peak_measurements) & set(
            selected_peak_relative_changes or {}
        )
        raise ValueError(
            "structured and legacy selected-peak evidence overlap for: "
            + ", ".join(sorted(overlap))
        )

    reference_document, reference_provenance = _observation_document(
        study.reference
    )
    candidate_documents: list[dict[str, Any]] = []
    candidate_provenance: dict[str, bool] = {}
    for candidate in study.candidates:
        document, complete = _observation_document(candidate)
        candidate_documents.append(document)
        candidate_provenance[candidate.name] = complete

    comparisons = [
        _comparison_document(
            study.reference,
            candidate,
            acceptance=acceptance,
            selected_peak_measurements=selected_peak_measurements,
            selected_peak_relative_changes=selected_peak_relative_changes,
            reference_provenance_complete=reference_provenance,
            candidate_provenance_complete=candidate_provenance[candidate.name],
        )
        for candidate in study.candidates
    ]
    has_comparisons = bool(comparisons)
    control_coverage = {
        name: any(
            bool(
                item["controls"][name][
                    "qualifies_for_release_coverage"
                ]
            )
            and {
                control_name
                for control_name in _REQUIRED_CONTROLS
                if item["controls"][control_name]["changed"]
            }
            == {name}
            for item in comparisons
        )
        for name in _REQUIRED_CONTROLS
    }
    missing_controls = tuple(
        name for name, covered in control_coverage.items() if not covered
    )
    full_control_coverage = not missing_controls
    verified_peak_evidence = bool(
        comparisons
        and all(
            item["gates"]["selected_peak_refinement"]["evidence_verified"]
            for item in comparisons
        )
    )
    certified_acceptance = bool(
        acceptance.max_current_relative_change
        <= _CERTIFIED_MAX_CURRENT_CHANGE
        and acceptance.max_selected_peak_relative_change
        <= _CERTIFIED_MAX_PEAK_CHANGE
        and acceptance.max_sigma_tail_ratio <= _CERTIFIED_MAX_TAIL_RATIO
        and acceptance.require_converged
        and acceptance.require_selected_peak_changes
    )
    refinement_passed = bool(
        has_comparisons
        and full_control_coverage
        and all(item["refinement_passed"] for item in comparisons)
    )
    all_results_certified = bool(
        study.reference.result.release_certified
        and study.candidates
        and all(item.result.release_certified for item in study.candidates)
    )
    all_provenance_complete = bool(
        reference_provenance
        and study.candidates
        and all(candidate_provenance.values())
    )
    all_input_provenance_complete = bool(
        reference_document["input_provenance_complete"]
        and candidate_documents
        and all(
            item["input_provenance_complete"]
            for item in candidate_documents
        )
    )
    all_code_provenance_complete = bool(
        reference_document["code_provenance_complete"]
        and candidate_documents
        and all(
            item["code_provenance_complete"]
            for item in candidate_documents
        )
    )
    all_study_summaries_consistent = bool(
        reference_document["study_summary_consistent"]
        and candidate_documents
        and all(
            item["study_summary_consistent"]
            for item in candidate_documents
        )
    )
    all_observation_documents = [
        reference_document,
        *candidate_documents,
    ]
    code_identity_keys = (
        "declared_package_version",
        "git_revision",
        "installed_package_version",
        "source_tree_sha256",
    )
    reference_code_identity = tuple(
        reference_document["code_provenance"].get(key)
        for key in code_identity_keys
    )
    all_code_identity_consistent = all(
        tuple(
            item["code_provenance"].get(key)
            for key in code_identity_keys
        )
        == reference_code_identity
        for item in all_observation_documents
    )
    runtime_identity_keys = (
        "blas_runtime",
        "dtype",
        "numpy_version",
        "platform",
        "python",
        "scipy_version",
    )
    reference_runtime_identity = tuple(
        reference_document["result_metadata"].get(key)
        for key in runtime_identity_keys
    )
    all_runtime_identity_consistent = all(
        tuple(
            item["result_metadata"].get(key)
            for key in runtime_identity_keys
        )
        == reference_runtime_identity
        for item in all_observation_documents
    )
    invariant_input_keys = (
        "device",
        "left_reservoir",
        "right_reservoir",
        "cubic_vertex",
        "solver_scale",
    )
    reference_physics_identity = tuple(
        reference_document["input_hashes"].get(key)
        for key in invariant_input_keys
    )
    all_physical_inputs_consistent = all(
        tuple(item["input_hashes"].get(key) for key in invariant_input_keys)
        == reference_physics_identity
        for item in all_observation_documents
    )
    all_runs_interacting = bool(
        study.reference.result.metadata.get("vertex_nonzero", False)
        and study.candidates
        and all(
            item.result.metadata.get("vertex_nonzero", False)
            for item in study.candidates
        )
    )
    release_certified = bool(
        refinement_passed
        and all_provenance_complete
        and all_results_certified
        and all_runs_interacting
        and verified_peak_evidence
        and certified_acceptance
        and all_study_summaries_consistent
        and all_code_identity_consistent
        and all_runtime_identity_consistent
        and all_physical_inputs_consistent
    )

    failures: list[str] = []
    if not has_comparisons:
        failures.append("at least one refinement comparison is required")
    if missing_controls:
        failures.append(
            "missing required refinement controls: "
            + ", ".join(missing_controls)
        )
    for comparison in comparisons:
        for gate_name, gate in comparison["gates"].items():
            if not gate["passed"]:
                failures.append(
                    f"{comparison['candidate']}: {gate_name} gate failed"
                )
    if not all_results_certified:
        failures.append(
            "reference and every candidate must be independently "
            "release-certified"
        )
    if not all_provenance_complete:
        failures.append(
            "reference and every candidate must have complete input and code "
            "provenance"
        )
    if not all_runs_interacting:
        failures.append(
            "dense-v0.1 anharmonic release evidence requires a nonzero cubic "
            "vertex in every run"
        )
    if not verified_peak_evidence:
        failures.append(
            "every comparison requires structured selected-peak measurements"
        )
    if not certified_acceptance:
        failures.append(
            "acceptance settings are looser than the dense-v0.1 release "
            "contract or disable a mandatory gate"
        )
    if not all_study_summaries_consistent:
        failures.append(
            "stored refinement summaries disagree with their immutable solver "
            "results"
        )
    if not all_code_identity_consistent:
        failures.append(
            "reference and candidates do not share one code identity"
        )
    if not all_runtime_identity_consistent:
        failures.append(
            "reference and candidates do not share one numerical runtime "
            "identity"
        )
    if not all_physical_inputs_consistent:
        failures.append(
            "reference and candidates do not share invariant device, "
            "reservoir, vertex, and scale inputs"
        )

    core_document: dict[str, Any] = {
        "acceptance": {
            "max_current_relative_change": (
                acceptance.max_current_relative_change
            ),
            "max_selected_peak_relative_change": (
                acceptance.max_selected_peak_relative_change
            ),
            "max_sigma_tail_ratio": acceptance.max_sigma_tail_ratio,
            "require_converged": acceptance.require_converged,
            "require_selected_peak_changes": (
                acceptance.require_selected_peak_changes
            ),
            "release_contract_eligible": certified_acceptance,
        },
        "candidates": candidate_documents,
        "comparisons": comparisons,
        "nonfinite_float_encoding": "NaN|Infinity|-Infinity strings",
        "reference": reference_document,
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "summary": {
            "all_code_provenance_complete": all_code_provenance_complete,
            "all_code_identity_consistent": all_code_identity_consistent,
            "all_input_provenance_complete": all_input_provenance_complete,
            "all_physical_inputs_consistent": (
                all_physical_inputs_consistent
            ),
            "all_provenance_complete": all_provenance_complete,
            "all_results_release_certified": all_results_certified,
            "all_runtime_identity_consistent": (
                all_runtime_identity_consistent
            ),
            "all_runs_use_nonzero_cubic_vertex": all_runs_interacting,
            "all_study_summaries_consistent": (
                all_study_summaries_consistent
            ),
            "comparison_count": len(comparisons),
            "control_coverage": control_coverage,
            "failures": failures,
            "full_control_coverage": full_control_coverage,
            "refinement_passed": refinement_passed,
            "release_certified": release_certified,
            "verified_selected_peak_evidence": verified_peak_evidence,
        },
    }
    report_sha256 = hashlib.sha256(
        _canonical_json(core_document).encode("utf-8")
    ).hexdigest()
    return DenseV01RefinementReport(
        document={**core_document, "report_sha256": report_sha256}
    )


__all__ = [
    "DenseV01Acceptance",
    "DenseV01PeakMeasurement",
    "DenseV01RefinementReport",
    "build_dense_v01_refinement_report",
]
