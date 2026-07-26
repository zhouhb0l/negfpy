"""Workflow-facing serialization for immutable harmonic IFC audits."""

from __future__ import annotations

import math
from typing import Any

from negfpy.modeling.validators import IFCValidationReport


def _json_number(value: float) -> float | str:
    """Return a strict-JSON scalar while retaining non-finite diagnostics."""

    number = float(value)
    if math.isfinite(number):
        return number
    if math.isnan(number):
        return "NaN"
    return "Infinity" if number > 0.0 else "-Infinity"


def ifc_validation_report_to_dict(report: IFCValidationReport) -> dict[str, Any]:
    """Convert an IFC audit report into deterministic strict-JSON primitives."""

    if not isinstance(report, IFCValidationReport):
        raise TypeError("report must be an IFCValidationReport instance.")

    minimum_qpoint = report.minimum_eigenvalue_qpoint
    return {
        "passed": bool(report.passed),
        "schema_valid": bool(report.schema_valid),
        "mapping_valid": bool(report.mapping_valid),
        "reciprocal_valid": bool(report.reciprocal_valid),
        "asr_valid": bool(report.asr_valid),
        "rotational_valid": bool(report.rotational_valid),
        "rotational_applicable": bool(report.rotational_applicable),
        "rotational_geometry_complete": bool(
            report.rotational_geometry_complete
        ),
        "hermiticity_valid": bool(report.hermiticity_valid),
        "stable": bool(report.stable),
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "metrics": {
            str(name): _json_number(value)
            for name, value in sorted(report.metrics.items())
        },
        "qpoints": [list(qpoint) for qpoint in report.qpoints],
        "duplicate_translations": [
            list(translation) for translation in report.duplicate_translations
        ],
        "missing_reciprocal_translations": [
            list(translation)
            for translation in report.missing_reciprocal_translations
        ],
        "minimum_eigenvalue_qpoint": (
            None if minimum_qpoint is None else list(minimum_qpoint)
        ),
    }


def ifc_audit_payload(
    *,
    raw: IFCValidationReport,
    final: IFCValidationReport,
    release_preflight: bool,
) -> dict[str, Any]:
    """Build the raw/final audit payload used by the bulk workflow report."""

    requested = bool(release_preflight)
    return {
        "raw": ifc_validation_report_to_dict(raw),
        "final": ifc_validation_report_to_dict(final),
        "release_preflight_requested": requested,
        "release_preflight_passed": bool(final.passed) if requested else None,
        "certification_eligible": bool(final.passed),
    }


__all__ = ["ifc_audit_payload", "ifc_validation_report_to_dict"]
