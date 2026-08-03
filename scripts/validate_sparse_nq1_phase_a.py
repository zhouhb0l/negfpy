#!/usr/bin/env python3
"""Run one bounded, restartable Phase-A material-harmonic validation case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
from typing import Any

import numpy as np

from negfpy.anharmonic import (
    SolverScale,
    build_layered_device_mapping,
    build_layered_harmonic_model,
    compare_layered_harmonic_models,
    read_phono3py_crystal_mapping,
    read_phono3py_fc2,
    validate_layered_device_mapping,
    validate_layered_harmonic_model,
)
from negfpy.anharmonic._canonical import (
    canonical_encode,
    enumerate_direct_reciprocal_cosets,
)
from negfpy.anharmonic.phase_a_fixture import verify_phase_a_fixture


SCHEMA = "negfpy.sparse-nq1-phase-a-case.v1"
TRANSPORT_TRANSFORM = np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int64
)
NONSYMMETRIC_TRANSFORM = np.array(
    [[2, 1, 0], [0, 2, 1], [0, 0, 1]], dtype=np.int64
)
HBAR_J_S = 1.0545718176461565e-34
OMEGA0_A = 2.0 * np.pi * 1.0e13
OMEGA0_B = 4.0 * np.pi * 1.0e13


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(canonical_encode(np.asarray(values))).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "canonical_sha256": _array_sha256(value),
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    if isinstance(value, dict) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _report_summary(report: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": report.schema_version,
        "report_sha256": report.report_sha256,
        "certification_passed": bool(report.certification_passed),
        "failure_codes": list(report.failure_codes),
        "warning_codes": list(report.warning_codes),
    }
    for name in ("checks", "counts", "residuals", "tolerances"):
        if hasattr(report, name):
            result[name] = _plain(getattr(report, name))
    for name in (
        "comparison",
        "qset_sha256",
        "full_compact_metrics_or_null",
        "omega0_metrics_or_null",
    ):
        if hasattr(report, name):
            result[name] = _plain(getattr(report, name))
    return result


def _scale(omega0: float) -> SolverScale:
    return SolverScale(
        omega0=omega0,
        hbar=HBAR_J_S,
        metadata={
            "source_units": {
                "fc2": "eV/angstrom^2",
                "mass": "amu",
                "length": "angstrom",
            },
            "phase": "sparse_nq1_phase_a_harmonic_only",
        },
    )


def _load_model(fixture: Path, *, layout: str, omega0: float) -> Any:
    yaml_path = fixture / "phono3py_disp.yaml"
    mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc2")
    filename = "fc2.compact.hdf5" if layout == "compact" else "fc2.full.hdf5"
    fc2 = read_phono3py_fc2(
        fixture / filename,
        mapping.supercell.masses,
        p2s_map=mapping.p2s,
        s2p_map=mapping.s2p,
        compact=(layout == "compact"),
    )
    device_mapping = build_layered_device_mapping(
        mapping,
        fc2,
        transport_supercell_matrix=TRANSPORT_TRANSFORM,
    )
    return build_layered_harmonic_model(device_mapping, _scale(omega0))


def _model_summary(model: Any) -> dict[str, Any]:
    geometry = validate_layered_device_mapping(model.mapping)
    harmonic = validate_layered_harmonic_model(model)
    result = {
        "fc2_layout": model.fc2_layout,
        "omega0_rad_s": model.scale.omega0,
        "n_transport_basis_atoms": model.mapping.n_transport_basis_atoms,
        "principal_cells": model.mapping.principal_cells,
        "n_principal_layer_dofs": model.mapping.n_principal_layer_dofs,
        "layer_offsets": model.mapping.layer_offsets.tolist(),
        "d00_sha256": _array_sha256(model.d00),
        "d01_sha256": _array_sha256(model.d01),
        "d10_sha256": _array_sha256(model.d10),
        "fc2_content_sha256": model.fc2_content_sha256,
        "primitive_crystal_content_sha256": model.primitive_crystal_content_sha256,
        "fc2_crystal_mapping_content_sha256": model.fc2_crystal_mapping_content_sha256,
        "layered_geometry_sha256": model.layered_geometry_sha256,
        "model_physics_sha256": model.model_physics_sha256,
        "geometry_validation": _report_summary(geometry),
        "harmonic_validation": _report_summary(harmonic),
    }
    if hasattr(model.mapping, "longitudinal_fc2_exact_numerical_support"):
        result["exact_numerical_support_sha256"] = _array_sha256(
            model.mapping.longitudinal_fc2_exact_numerical_support
        )
        result["exact_numerical_support_count"] = int(
            len(model.mapping.longitudinal_fc2_exact_numerical_support)
        )
    return result


def _case_payload(case: str, fixture: Path) -> dict[str, Any]:
    if case == "cosets":
        direct, reciprocal = enumerate_direct_reciprocal_cosets(
            NONSYMMETRIC_TRANSFORM
        )
        passed = bool(
            len(direct.keys) == 4
            and len(reciprocal.keys) == 4
            and not np.array_equal(direct.representatives, reciprocal.representatives)
        )
        return {
            "case": case,
            "direct_sha256": direct.sha256,
            "reciprocal_sha256": reciprocal.sha256,
            "direct_representatives": direct.representatives.tolist(),
            "reciprocal_representatives": reciprocal.representatives.tolist(),
            "certification_passed": passed,
        }
    if case in {"compact", "full"}:
        model = _load_model(fixture, layout=case, omega0=OMEGA0_A)
        summary = _model_summary(model)
        passed = bool(
            summary["geometry_validation"]["certification_passed"]
            and summary["harmonic_validation"]["certification_passed"]
            and summary["n_transport_basis_atoms"] == 8
            and summary["principal_cells"] == 1
            and summary["n_principal_layer_dofs"] == 24
        )
        return {"case": case, "model": summary, "certification_passed": passed}
    if case == "full-compact":
        compact = _load_model(fixture, layout="compact", omega0=OMEGA0_A)
        full = _load_model(fixture, layout="full", omega0=OMEGA0_A)
        report = compare_layered_harmonic_models(
            full, compact, comparison="full_compact"
        )
        return {
            "case": case,
            "full_model_physics_sha256": full.model_physics_sha256,
            "compact_model_physics_sha256": compact.model_physics_sha256,
            "comparison": _report_summary(report),
            "certification_passed": bool(report.certification_passed),
        }
    if case == "omega0":
        first = _load_model(fixture, layout="full", omega0=OMEGA0_A)
        second = _load_model(fixture, layout="full", omega0=OMEGA0_B)
        report = compare_layered_harmonic_models(
            first, second, comparison="omega0_invariance"
        )
        return {
            "case": case,
            "omega0_rad_s": [OMEGA0_A, OMEGA0_B],
            "comparison": _report_summary(report),
            "certification_passed": bool(report.certification_passed),
        }
    raise ValueError(f"Unsupported case {case!r}.")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite completed outcome: {path}")
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=("cosets", "compact", "full", "full-compact", "omega0"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    fixture = arguments.fixture.resolve()
    fixture_report = verify_phase_a_fixture(fixture)
    if arguments.preflight_only:
        if arguments.case is not None or arguments.output is not None:
            parser.error("--preflight-only accepts neither --case nor --output.")
        print(
            json.dumps(
                {
                    "schema": "negfpy.sparse-nq1-phase-a-preflight.v1",
                    "fixture": fixture_report,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "passed": True,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 0
    if arguments.case is None or arguments.output is None:
        parser.error("scientific execution requires --case and --output.")
    scientific = _case_payload(arguments.case, fixture)
    payload = {
        "schema": SCHEMA,
        "case": arguments.case,
        "fixture_manifest_sha256": fixture_report["fixture_manifest_sha256"],
        "fixture_sha256sums_sha256": fixture_report["fixture_sha256sums_sha256"],
        "scientific": scientific,
        "certification_passed": bool(scientific["certification_passed"]),
    }
    _atomic_json(arguments.output, payload)
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "sha256": _sha256_file(arguments.output),
                "certification_passed": payload["certification_passed"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0 if payload["certification_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
