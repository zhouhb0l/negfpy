#!/usr/bin/env python3
"""Render the NEGFPy bulk-phonon/SMRTA validation report.

The renderer consumes, but never produces or modifies, scientific results.  A
report can use release-certified wording only when ``--release-mode`` is
requested and the frozen commit-last completion receipt, provisional
JSON/evidence, HDF5 archive, and checksum manifest satisfy the complete
reporting contract.  Development inputs always produce a conspicuously marked
``DRAFT / UNCERTIFIED`` document.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPORT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SCHEMA = "negfpy.bulk_rta_release_validation.v2"
LEGACY_VALIDATION_SCHEMA = "negfpy.bulk_rta_release_validation.v1"
FINAL_HDF5_SCHEMA = "negfpy.bulk_rta_release_results.hdf5.v1"
FINAL_EVIDENCE_SCHEMA = "negfpy.bulk_rta_release_evidence.v1"
LEGACY_HDF5_SCHEMA = LEGACY_VALIDATION_SCHEMA
FINAL_RECEIPT_SCHEMA = "negfpy.bulk_rta_release_completion_receipt.v1"
EXPECTED_MESHES = ((3, 3, 3), (5, 5, 5), (11, 11, 11))
FROZEN_REPORT_KEYS = frozenset(
    {
        "schema",
        "mode",
        "scientific_passed",
        "candidate_release_gates_passed",
        "release_certified",
        "certification_authority",
        "physical_mesh_convergence_certified",
        "physical_mesh_convergence_note",
        "approximation",
        "excluded_physics",
        "campaign_manifest",
        "campaign_sha256",
        "evidence_document",
        "evidence_sha256",
        "results_hdf5",
        "results_hdf5_sha256",
        "runtime_before",
        "runtime_after",
        "runtime_unchanged",
        "inputs",
        "settings",
        "mesh_evidence",
        "meshes",
    }
)
FROZEN_EVIDENCE_KEYS = frozenset(
    {
        "schema",
        "campaign_sha256",
        "scientific_passed",
        "candidate_release_gates_passed",
        "release_certified",
        "certification_authority",
        "settings",
        "source_sha256",
        "archive_sha256",
        "meshes",
    }
)
FROZEN_RECEIPT_KEYS = frozenset(
    {
        "schema",
        "certification_authority",
        "campaign_sha256",
        "scientific_passed",
        "candidate_release_gates_passed",
        "provisional_release_certified",
        "release_certified",
        "report_sha256",
        "evidence_sha256",
        "results_hdf5_sha256",
        "sha256sums_sha256",
        "evidence_scientific_passed",
        "evidence_candidate_release_gates_passed",
    }
)
FROZEN_BOOLEAN_RECEIPT_FIELDS = (
    "scientific_passed",
    "candidate_release_gates_passed",
    "provisional_release_certified",
    "release_certified",
    "evidence_scientific_passed",
    "evidence_candidate_release_gates_passed",
)
FROZEN_INPUT_KEYS = frozenset(
    {"source_paths", "source_sha256", "archive_paths", "archive_sha256"}
)
CHECKSUM_EXCLUSIONS = frozenset(
    {"SHA256SUMS.json", "release-completion-receipt.json"}
)
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
REQUIRED_FIELDS = (
    "frequencies_thz",
    "group_velocities_thz_angstrom",
    "heat_capacities_ev_per_k",
    "gamma_thz_hwhm",
    "population_lifetime_ps",
    "mean_free_path_angstrom",
    "mode_kappa_w_mk",
    "kappa_w_mk",
)
FIGURE_DPI = 220
SOURCE_DATE_EPOCH = "946684800"
MFP_DISPLAY_FLOOR_NM = 1.0e-6


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file.")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError(f"{label} does not exist.") from exc
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file.")
    return resolved


def _finite_json(value: Any, *, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _finite_json(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for position, item in enumerate(value):
            _finite_json(item, path=f"{path}[{position}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite JSON number at {path}.")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid validation JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - malformed file content
            "Validation JSON root must be an object."
        )
    _finite_json(value)
    return value


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None


def _checksum_manifest_payload(root: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Release artifact tree contains symlink {path}.")
        relative = str(path.relative_to(root))
        if path.is_dir():
            key = relative + "/"
            payload[key] = hashlib.sha256(
                b"negfpy.release-directory.v1\0" + relative.encode("utf-8")
            ).hexdigest()
        elif path.is_file():
            if relative not in CHECKSUM_EXCLUSIONS:
                payload[relative] = _sha256(path)
        else:
            raise ValueError(f"Release artifact tree contains special file {path}.")
    return {name: payload[name] for name in sorted(payload)}


def _validate_checksum_manifest(root: Path) -> Path:
    checksum_path = _regular_file(
        root / "SHA256SUMS.json",
        label="checksum manifest",
    )
    checksums = _load_json(checksum_path)
    if any(
        not isinstance(name, str) or not _is_sha256(digest)
        for name, digest in checksums.items()
    ):
        raise ValueError("Checksum manifest has an invalid entry schema.")
    if checksums != _checksum_manifest_payload(root):
        raise ValueError("Checksum manifest differs from the exact artifact tree.")
    return checksum_path


def _hdf5_schema(path: Path) -> str:
    with h5py.File(path, "r") as handle:
        return str(handle.attrs.get("schema", ""))


def _load_release_evidence(
    report: dict[str, Any],
    *,
    json_path: Path,
    hdf5_path: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, Path]:
    """Authenticate the frozen commit-last publication and its sole authority."""

    schema = _hdf5_schema(hdf5_path)
    if schema == LEGACY_HDF5_SCHEMA:
        checksum_path = _validate_checksum_manifest(json_path.parent)
        return None, None, checksum_path
    if schema != FINAL_HDF5_SCHEMA:
        raise ValueError(f"Unsupported results HDF5 schema {schema!r}.")
    if json_path.name != "bulk-rta-validation.json":
        raise ValueError("Frozen validation JSON must use its canonical filename.")
    if hdf5_path.name != "bulk-rta-results.hdf5":
        raise ValueError("Frozen results HDF5 must use its canonical filename.")
    evidence_path = _regular_file(
        json_path.parent / "release-evidence.json",
        label="release evidence",
    )
    campaign_path = _regular_file(
        json_path.parent / "campaign-manifest.json",
        label="campaign manifest",
    )
    receipt_path = _regular_file(
        json_path.parent / "release-completion-receipt.json",
        label="release completion receipt",
    )
    checksum_path = _validate_checksum_manifest(json_path.parent)
    evidence = _load_json(evidence_path)
    receipt = _load_json(receipt_path)
    evidence_sha256 = _sha256(evidence_path)
    report_sha256 = _sha256(json_path)
    hdf5_sha256 = _sha256(hdf5_path)
    checksum_sha256 = _sha256(checksum_path)
    candidate_report = report.get("candidate_release_gates_passed")
    candidate_evidence = evidence.get("candidate_release_gates_passed")
    checks = {
        "evidence fields": set(evidence) == FROZEN_EVIDENCE_KEYS,
        "evidence schema": evidence.get("schema") == FINAL_EVIDENCE_SCHEMA,
        "evidence boolean types": all(
            isinstance(evidence.get(name), bool)
            for name in (
                "scientific_passed",
                "candidate_release_gates_passed",
                "release_certified",
            )
        ),
        "report provisional": report.get("release_certified") is False,
        "evidence provisional": evidence.get("release_certified") is False,
        "provisional authority": report.get("certification_authority")
        == evidence.get("certification_authority")
        == "release-completion-receipt.json",
        "report evidence SHA256": report.get("evidence_sha256") == evidence_sha256,
        "report HDF5 SHA256": report.get("results_hdf5_sha256")
        == hdf5_sha256,
        "campaign identity": report.get("campaign_sha256")
        == evidence.get("campaign_sha256"),
        "campaign file identity": report.get("campaign_sha256")
        == _sha256(campaign_path),
        "mesh evidence": report.get("mesh_evidence") == evidence.get("meshes"),
        "scientific decision": report.get("scientific_passed")
        == evidence.get("scientific_passed"),
        "candidate decision": isinstance(candidate_report, bool)
        and isinstance(candidate_evidence, bool)
        and candidate_report == candidate_evidence,
        "candidate requires science": candidate_report is False
        or report.get("scientific_passed") is True,
        "settings": report.get("settings") == evidence.get("settings"),
    }
    inputs = report.get("inputs")
    checks["source hashes"] = bool(
        isinstance(inputs, dict)
        and inputs.get("source_sha256") == evidence.get("source_sha256")
    )
    checks["archive hashes"] = bool(
        isinstance(inputs, dict)
        and inputs.get("archive_sha256") == evidence.get("archive_sha256")
    )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Release JSON/evidence binding failed: " + ", ".join(failed)
        )
    receipt_checks = {
        "receipt fields": set(receipt) == FROZEN_RECEIPT_KEYS,
        "receipt schema": receipt.get("schema") == FINAL_RECEIPT_SCHEMA,
        "receipt authority": receipt.get("certification_authority") == "sole",
        "receipt boolean types": all(
            isinstance(receipt.get(name), bool)
            for name in FROZEN_BOOLEAN_RECEIPT_FIELDS
        ),
        "receipt campaign": receipt.get("campaign_sha256")
        == report.get("campaign_sha256")
        == evidence.get("campaign_sha256"),
        "receipt scientific": receipt.get("scientific_passed")
        == receipt.get("evidence_scientific_passed")
        == report.get("scientific_passed")
        == evidence.get("scientific_passed"),
        "receipt candidate": receipt.get("candidate_release_gates_passed")
        == receipt.get("evidence_candidate_release_gates_passed")
        == candidate_report
        == candidate_evidence,
        "receipt provisional": receipt.get("provisional_release_certified") is False,
        "receipt decision": receipt.get("release_certified")
        == receipt.get("candidate_release_gates_passed"),
        "receipt report hash": receipt.get("report_sha256") == report_sha256,
        "receipt evidence hash": receipt.get("evidence_sha256") == evidence_sha256,
        "receipt HDF5 hash": receipt.get("results_hdf5_sha256") == hdf5_sha256,
        "receipt checksum hash": receipt.get("sha256sums_sha256")
        == checksum_sha256,
    }
    failed = sorted(name for name, passed in receipt_checks.items() if not passed)
    if failed:
        raise ValueError("Completion receipt binding failed: " + ", ".join(failed))
    return evidence, receipt, checksum_path


def _mesh_key(mesh: tuple[int, int, int]) -> str:
    return f"mesh-{mesh[0]}x{mesh[1]}x{mesh[2]}"


def _validate_json(report: dict[str, Any]) -> list[dict[str, Any]]:
    if report.get("schema") not in {EXPECTED_SCHEMA, LEGACY_VALIDATION_SCHEMA}:
        raise ValueError(
            f"Unsupported report schema {report.get('schema')!r}; "
            f"expected {EXPECTED_SCHEMA!r}."
        )
    required_booleans = (
        "scientific_passed",
        "release_certified",
        "physical_mesh_convergence_certified",
    )
    if report.get("schema") == EXPECTED_SCHEMA:
        required_booleans += ("candidate_release_gates_passed",)
        if set(report) != FROZEN_REPORT_KEYS:
            raise ValueError("Frozen validation report fields are not exact.")
        inputs = report.get("inputs")
        if not isinstance(inputs, dict) or set(inputs) != FROZEN_INPUT_KEYS:
            raise ValueError("Frozen validation report input fields are not exact.")
        if report.get("release_certified") is not False:
            raise ValueError(
                "The frozen validation report must remain provisional; only the "
                "completion receipt may certify a release."
            )
        if (
            report.get("certification_authority")
            != "release-completion-receipt.json"
        ):
            raise ValueError("Frozen validation report names the wrong authority.")
    for name in required_booleans:
        if not isinstance(report.get(name), bool):
            raise ValueError(  # noqa: TRY004 - malformed file content
                f"Validation field {name!r} must be boolean."
            )
    if report["physical_mesh_convergence_certified"]:
        raise ValueError(
            "This implementation-validation report must not claim physical "
            "mesh convergence."
        )
    if report.get("approximation") != "intrinsic_three_phonon_smrta":
        raise ValueError("The report approximation must be intrinsic SMRTA.")
    locked_schema = isinstance(report.get("mesh_evidence"), list)
    records = report.get("mesh_evidence") if locked_schema else report.get("meshes")
    if not isinstance(records, list) or len(records) != len(EXPECTED_MESHES):
        raise ValueError("Validation report must contain exactly 3^3, 5^3, 11^3.")
    by_mesh: dict[tuple[int, int, int], dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(  # noqa: TRY004 - malformed file content
                "Each mesh record must be an object."
            )
        raw_mesh = record.get("mesh")
        if not isinstance(raw_mesh, (list, tuple)) or len(raw_mesh) != 3:
            raise ValueError("Every mesh record must contain three dimensions.")
        mesh = tuple(int(item) for item in raw_mesh)
        if mesh in by_mesh:
            raise ValueError(f"Duplicate mesh record {mesh}.")
        if int(record.get("n_qpoints", -1)) != int(np.prod(mesh)):
            raise ValueError(f"Mesh {mesh} has an inconsistent q-point count.")
        if not isinstance(record.get("passed"), bool):
            raise ValueError(  # noqa: TRY004 - malformed file content
                f"Mesh {mesh} lacks a boolean pass decision."
            )
        normalized = dict(record)
        normalized["mesh"] = list(mesh)
        normalized["_evidence_record"] = record
        if locked_schema:
            harmonic = record.get("harmonic_metrics")
            if not isinstance(harmonic, dict):
                raise ValueError(f"Mesh {mesh} lacks harmonic_metrics.")
            normalized["maximum_group_velocity_absolute_error_thz_angstrom"] = (
                harmonic.get("group_velocity_max_abs_error_thz_angstrom")
            )
            required = (
                "maximum_heat_capacity_absolute_error_ev_per_k",
                "negfpy_kappa_voigt_w_mk",
                "reference_kappa_voigt_w_mk",
            )
            if any(name not in normalized for name in required):
                raise ValueError(f"Mesh {mesh} direct metrics are incomplete.")
            if record.get("end_to_end_required") is True:
                passed = record.get("end_to_end_passed")
                if not isinstance(passed, bool):
                    raise ValueError(
                        f"Mesh {mesh} lacks its required end-to-end decision."
                    )
                normalized["end_to_end"] = {"passed": passed}
            else:
                normalized["end_to_end"] = None
        by_mesh[mesh] = normalized
    if tuple(sorted(by_mesh)) != tuple(sorted(EXPECTED_MESHES)):
        raise ValueError("Mesh set is not exactly 3^3, 5^3, and 11^3.")
    return [by_mesh[mesh] for mesh in EXPECTED_MESHES]


def _dataset(handle: h5py.File, path: str) -> np.ndarray:
    if path not in handle or not isinstance(handle[path], h5py.Dataset):
        raise ValueError(f"Missing HDF5 dataset {path!r}.")
    if not isinstance(handle.get(path, getlink=True), h5py.HardLink):
        raise ValueError(  # noqa: TRY004 - malformed archive content
            f"HDF5 dataset {path!r} must be an internal hard link."
        )
    values = handle[path][...]
    if values.dtype.kind not in "fiu" or not np.all(np.isfinite(values)):
        raise ValueError(f"HDF5 dataset {path!r} must be finite numeric data.")
    return np.asarray(values)


def _load_scientific_arrays(
    hdf5_path: Path,
    report: dict[str, Any],
    evidence: dict[str, Any] | None,
    report_sha256: str,
    records: list[dict[str, Any]],
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    arrays: dict[int, dict[str, np.ndarray]] = {}
    optional: dict[str, Any] = {}
    with h5py.File(hdf5_path, "r") as handle:
        schema = str(handle.attrs.get("schema", ""))
        locked = schema == FINAL_HDF5_SCHEMA
        if not locked and schema != LEGACY_HDF5_SCHEMA:
            raise ValueError("HDF5 schema does not match a supported report schema.")
        if locked:
            if evidence is None:
                raise ValueError("Locked HDF5 results require release evidence.")
            if {str(name) for name in handle.attrs} != {
                "schema",
                "campaign_sha256",
                "evidence_schema",
                "evidence_sha256",
            }:
                raise ValueError("Locked HDF5 root attributes are not exact.")
            if str(handle.attrs["campaign_sha256"]) != report.get("campaign_sha256"):
                raise ValueError("HDF5 campaign identity does not match the report.")
            if str(handle.attrs["evidence_schema"]) != FINAL_EVIDENCE_SCHEMA:
                raise ValueError("HDF5 evidence schema is not the locked schema.")
            if str(handle.attrs["evidence_sha256"]) != report.get("evidence_sha256"):
                raise ValueError("HDF5 evidence SHA256 does not match the report.")
            if {str(name) for name in handle} != {"meshes"}:
                raise ValueError("Locked HDF5 root group schema is not exact.")
        elif str(handle.attrs.get("report_sha256", "")) != report_sha256:
            raise ValueError("Legacy HDF5 report_sha256 does not bind the JSON.")
        for record in records:
            mesh = tuple(int(item) for item in record["mesh"])
            n_qpoints = int(np.prod(mesh))
            group = (
                f"meshes/{_mesh_key(mesh)}/rta"
                if locked
                else _mesh_key(mesh)
            )
            if locked:
                mesh_group = handle[f"meshes/{_mesh_key(mesh)}"]
                expected_attributes = {
                    "n_qpoints",
                    "passed",
                    "source_rta_sha256",
                    "source_harmonic_sha256",
                    "metrics_json",
                }
                if {
                    str(name) for name in mesh_group.attrs
                } != expected_attributes:
                    raise ValueError(f"Mesh {mesh} HDF5 attributes are not exact.")
                raw_evidence = record["_evidence_record"]
                if int(mesh_group.attrs["n_qpoints"]) != n_qpoints:
                    raise ValueError(f"Mesh {mesh} HDF5 q-point count changed.")
                if bool(mesh_group.attrs["passed"]) != record["passed"]:
                    raise ValueError(f"Mesh {mesh} HDF5 pass decision changed.")
                if str(mesh_group.attrs["source_rta_sha256"]) != raw_evidence.get(
                    "rta_sha256"
                ):
                    raise ValueError(f"Mesh {mesh} RTA checkpoint identity changed.")
                if str(mesh_group.attrs["source_harmonic_sha256"]) != raw_evidence.get(
                    "harmonic_sha256"
                ):
                    raise ValueError(
                        f"Mesh {mesh} harmonic checkpoint identity changed."
                    )
                try:
                    metrics = json.loads(str(mesh_group.attrs["metrics_json"]))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Mesh {mesh} metrics_json is invalid.") from exc
                if metrics != raw_evidence:
                    raise ValueError(f"Mesh {mesh} HDF5 metrics changed.")
            fields = {
                name: _dataset(handle, f"{group}/{name}")
                for name in REQUIRED_FIELDS
            }
            frequencies = fields["frequencies_thz"]
            if frequencies.ndim != 2 or frequencies.shape[0] != n_qpoints:
                raise ValueError(f"{group} frequencies have an invalid shape.")
            n_bands = frequencies.shape[1]
            expected_qb = (n_qpoints, n_bands)
            if fields["group_velocities_thz_angstrom"].shape != expected_qb + (3,):
                raise ValueError(f"{group} group velocities have an invalid shape.")
            for name in (
                "heat_capacities_ev_per_k",
                "gamma_thz_hwhm",
                "population_lifetime_ps",
                "mean_free_path_angstrom",
            ):
                if fields[name].ndim != 3 or fields[name].shape[1:] != expected_qb:
                    raise ValueError(f"{group}/{name} has an invalid shape.")
            mode = fields["mode_kappa_w_mk"]
            if mode.ndim != 5 or mode.shape[1:3] != expected_qb or mode.shape[-2:] != (3, 3):
                raise ValueError(f"{group} mode conductivity has an invalid shape.")
            kappa = fields["kappa_w_mk"]
            if kappa.shape != (mode.shape[0], 3, 3):
                raise ValueError(f"{group} conductivity tensor has an invalid shape.")
            if not np.allclose(
                np.sum(mode, axis=(1, 2)),
                kappa,
                rtol=1.0e-10,
                atol=1.0e-11,
            ):
                raise ValueError(f"{group} mode conductivities do not sum to kappa.")
            arrays[mesh[0]] = fields

        if locked:
            harmonic_group = "meshes/mesh-11x11x11/harmonic"
            distance = _dataset(
                handle, f"{harmonic_group}/band/distances_inv_angstrom"
            )
            frequencies = _dataset(
                handle, f"{harmonic_group}/band/frequencies_thz"
            )
            vertices = _dataset(
                handle, f"{harmonic_group}/band/vertex_indices"
            ).astype(np.int64)
            if distance.ndim != 1 or frequencies.ndim != 2:
                raise ValueError("Band path datasets have invalid dimensions.")
            if frequencies.shape[0] != distance.size or np.any(np.diff(distance) < 0.0):
                raise ValueError("Band path and frequency datasets are inconsistent.")
            if (
                vertices.ndim != 1
                or vertices.size != 4
                or np.any((vertices < 0) | (vertices >= distance.size))
            ):
                raise ValueError("Locked silicon band vertices are invalid.")
            optional["band"] = {
                "distance": distance,
                "frequencies": frequencies,
                "labels": ["Γ", "X", "L", "Γ"],
                "positions": distance[vertices],
            }
            frequency = _dataset(
                handle, f"{harmonic_group}/dos/frequency_points_thz"
            )
            values = _dataset(
                handle, f"{harmonic_group}/dos/total_states_per_thz_cell"
            )
            if frequency.ndim != 1 or values.shape != frequency.shape:
                raise ValueError("DOS datasets have inconsistent shapes.")
            optional["dos"] = {
                "frequency": frequency,
                "values": values,
            }
            harmonic_frequencies = _dataset(
                handle, f"{harmonic_group}/mesh/frequencies_thz"
            )
            harmonic_velocities = _dataset(
                handle,
                f"{harmonic_group}/mesh/group_velocities_thz_angstrom",
            )
            if not np.array_equal(
                harmonic_frequencies, arrays[11]["frequencies_thz"]
            ):
                raise ValueError("Harmonic and RTA 11^3 frequencies differ.")
            if not np.array_equal(
                harmonic_velocities,
                arrays[11]["group_velocities_thz_angstrom"],
            ):
                raise ValueError("Harmonic and RTA 11^3 velocities differ.")
        else:
            band_group = None
            for candidate in ("band_structure", "band-structure"):
                if candidate in handle and isinstance(handle[candidate], h5py.Group):
                    band_group = candidate
                    break
            if band_group is not None:
                distance = _dataset(handle, f"{band_group}/path_distance")
                frequencies = _dataset(handle, f"{band_group}/frequencies_thz")
                if distance.ndim != 1 or frequencies.ndim != 2:
                    raise ValueError("Band path datasets have invalid dimensions.")
                if frequencies.shape[0] != distance.size or np.any(
                    np.diff(distance) < 0.0
                ):
                    raise ValueError(
                        "Band path and frequency datasets are inconsistent."
                    )
                labels_json = str(handle[band_group].attrs.get("labels_json", "[]"))
                labels = json.loads(labels_json)
                positions = _dataset(handle, f"{band_group}/label_positions")
                if not isinstance(labels, list) or len(labels) != positions.size:
                    raise ValueError("Band labels and positions are inconsistent.")
                optional["band"] = {
                    "distance": distance,
                    "frequencies": frequencies,
                    "labels": [str(value) for value in labels],
                    "positions": positions,
                }
            dos_group = None
            for candidate in ("density_of_states", "density-of-states"):
                if candidate in handle and isinstance(handle[candidate], h5py.Group):
                    dos_group = candidate
                    break
            if dos_group is not None:
                frequency = _dataset(handle, f"{dos_group}/frequency_thz")
                values = _dataset(
                    handle, f"{dos_group}/total_dos_states_per_thz_cell"
                )
                reference = _dataset(
                    handle,
                    f"{dos_group}/reference_total_dos_states_per_thz_cell",
                )
                if (
                    frequency.ndim != 1
                    or values.shape != frequency.shape
                    or reference.shape != frequency.shape
                ):
                    raise ValueError("DOS datasets have inconsistent shapes.")
                optional["dos"] = {
                    "frequency": frequency,
                    "values": values,
                    "reference": reference,
                }
    return arrays, optional


def _harmonic_table_metrics(
    report: dict[str, Any], records: list[dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if isinstance(report.get("mesh_evidence"), list):
        harmonic = records[-1].get("harmonic_metrics")
        if not isinstance(harmonic, dict):
            return {}, {}
        joint_pass = harmonic.get("passed") is True
        return (
            {
                "passed": joint_pass,
                "maximum_frequency_absolute_error_thz": harmonic.get(
                    "band_frequency_max_abs_error_thz"
                ),
            },
            {
                "passed": joint_pass
                and harmonic.get("dos_normalization_certified") is True,
                "maximum_absolute_error_states_per_thz_cell": harmonic.get(
                    "dos_max_abs_error_states_per_thz_cell"
                ),
            },
        )
    observables = report.get("harmonic_observables")
    if not isinstance(observables, dict):
        return {}, {}
    band = observables.get("band_structure")
    dos = observables.get("density_of_states")
    return (
        band if isinstance(band, dict) else {},
        dos if isinstance(dos, dict) else {},
    )


def _harmonic_reporting_complete(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    optional: dict[str, Any],
) -> bool:
    band, dos = _harmonic_table_metrics(report, records)
    return bool(
        band.get("passed") is True
        and isinstance(band.get("maximum_frequency_absolute_error_thz"), (int, float))
        and dos.get("passed") is True
        and isinstance(dos.get("maximum_absolute_error_states_per_thz_cell"), (int, float))
        and "band" in optional
        and "dos" in optional
    )


def _stable_runtime_identity(identity: dict[str, Any]) -> dict[str, Any]:
    """Mirror the frozen driver's normalization of lazy runtime state."""

    stable = json.loads(json.dumps(identity, sort_keys=True, allow_nan=False))
    thread_limits = stable.get("thread_limits")
    if isinstance(thread_limits, dict):
        thread_limits["threadpools"] = "validated_separately"
    ownership = stable.get("import_code_ownership")
    if isinstance(ownership, dict):
        ownership["generated_pyc_path_count"] = "validated_separately"
        ownership["generated_pyc_paths_sha256"] = "validated_separately"
    return stable


def _release_gaps(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    harmonic_complete: bool,
    receipt: dict[str, Any] | None,
) -> list[str]:
    """Return every unmet prerequisite for certified report wording."""

    gaps: list[str] = []
    if report.get("scientific_passed") is not True:
        gaps.append("scientific_passed=true")
    if (
        receipt is None
        or receipt.get("release_certified") is not True
        or receipt.get("candidate_release_gates_passed") is not True
    ):
        gaps.append("release-completion-receipt.json certifies the candidate")
    if not harmonic_complete:
        gaps.extend(
            (
                "externally compared band_structure JSON and HDF5 fields",
                "externally compared density_of_states JSON and HDF5 fields",
            )
        )
    locked = isinstance(report.get("mesh_evidence"), list)
    if locked:
        before = report.get("runtime_before")
        after = report.get("runtime_after")
        if report.get("mode") != "release":
            gaps.append("mode=release")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or report.get("runtime_unchanged") is not True
            or _stable_runtime_identity(before) != _stable_runtime_identity(after)
        ):
            gaps.append("stable authenticated runtime before and after")
            runtime: dict[str, Any] = {}
        else:
            runtime = after
        if runtime.get("clean") is not True:
            gaps.append("clean exact source provenance")
        for name in ("commit", "tree"):
            identity = runtime.get(name)
            try:
                if not isinstance(identity, str) or len(identity) != 40:
                    raise ValueError
                int(identity, 16)
            except ValueError:
                gaps.append(f"40-hex Git {name}")
        if runtime.get("negfpy_version") != "1.0.0":
            gaps.append("NEGFPy runtime version 1.0.0")
        package_versions = runtime.get("package_versions")
        if (
            not isinstance(package_versions, dict)
            or package_versions.get("phono3py") != "4.4.0"
        ):
            gaps.append("phono3py runtime version 4.4.0")
        if runtime.get("negfpy_import_tied_to_repository_payload") is not True:
            gaps.append("NEGFPy import tied to the authenticated source payload")
        provenance = runtime.get("code_provenance")
        if not isinstance(provenance, dict) or not bool(
            provenance.get("package_version_consistent") is True
            and provenance.get("installed_package_version") == "1.0.0"
            and provenance.get("git_dirty") is False
            and provenance.get("git_revision") == runtime.get("commit")
        ):
            gaps.append("consistent embedded package provenance")
        inputs = report.get("inputs")
        if not isinstance(inputs, dict) or not isinstance(
            inputs.get("source_sha256"), dict
        ):
            gaps.append("authenticated source SHA256 map")
        if not isinstance(inputs, dict) or not isinstance(
            inputs.get("archive_sha256"), dict
        ):
            gaps.append("authenticated archive SHA256 map")
    else:
        git = report.get("git")
        if not isinstance(git, dict) or git.get("clean") is not True:
            gaps.append("clean exact source provenance")
        versions = report.get("versions")
        if not isinstance(versions, dict) or versions.get("negfpy") != "1.0.0":
            gaps.append("NEGFPy runtime version 1.0.0")
        if not isinstance(versions, dict) or versions.get("phono3py") != "4.4.0":
            gaps.append("phono3py runtime version 4.4.0")
    settings = report.get("settings")
    try:
        sigma = float(settings["sigma_thz"])
    except (KeyError, TypeError, ValueError):
        sigma = float("nan")
    if not math.isclose(sigma, 0.1, rel_tol=0.0, abs_tol=1.0e-15):
        gaps.append("fixed release sigma_thz=0.1")
    if locked and isinstance(settings, dict):
        if settings.get("meshes") != [3, 5, 11]:
            gaps.append("exact release mesh ladder")
        if settings.get("end_to_end_meshes") != [3, 5]:
            gaps.append("exact 3^3 and 5^3 end-to-end ladder")
        if settings.get("temperatures_k") != [300.0]:
            gaps.append("exact 300 K release temperature")
    for record in records:
        mesh = int(record["mesh"][0])
        if record.get("passed") is not True:
            gaps.append(f"passing {mesh}^3 mesh decision")
        if mesh in (3, 5):
            end_to_end = record.get("end_to_end")
            if not isinstance(end_to_end, dict) or end_to_end.get("passed") is not True:
                gaps.append(f"passing {mesh}^3 end-to-end phono3py decision")
    return gaps


def _latex(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _latex_hash(value: Any) -> str:
    text = str(value)
    if _is_sha256(text):
        chunks = [text[position : position + 8] for position in range(0, 64, 8)]
        joined = r"\allowbreak{}".join(chunks)
        return rf"\texttt{{{joined}}}"
    return rf"\texttt{{{_latex(text)}}}"


def _format_scientific(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not recorded"
    if not math.isfinite(number):
        return "not recorded"
    if number == 0.0:
        return "0"
    exponent = math.floor(math.log10(abs(number)))
    if exponent >= 3 or exponent <= -2:
        coefficient = number / (10.0**exponent)
        return rf"${coefficient:.{digits}g}\times 10^{{{exponent}}}$"
    return f"{number:.{digits}g}"


def _voigt_xx(record: dict[str, Any], name: str) -> float:
    values = np.asarray(record[name], dtype=np.float64)
    return float(values.reshape(-1, 6)[0, 0])


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_generated_files(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    receipt: dict[str, Any] | None,
    report_certified: bool,
    harmonic_complete: bool,
    release_gaps: list[str],
    json_sha256: str,
    hdf5_sha256: str,
    receipt_sha256: str | None,
    checksum_sha256: str,
) -> None:
    if isinstance(report.get("runtime_after"), dict):
        git = report["runtime_after"]
    else:
        git = report.get("git") if isinstance(report.get("git"), dict) else {}
    commit = str(git.get("commit", "not-recorded"))
    tree = str(git.get("tree", "not-recorded"))
    generated = str(
        report.get("completed_at_utc", report.get("generated_at_utc", "not recorded"))
    )
    if report_certified:
        status_title = "RELEASE-CERTIFIED INPUT"
        status_message = (
            "The sole completion receipt certifies the exact candidate and all "
            "bound artifacts. Physical mesh convergence remains explicitly false."
        )
        status_color = "statusgreen"
        status_background = "statusgreenlight"
    else:
        status_title = "DRAFT / UNCERTIFIED"
        status_message = (
            "This rendering uses development or incomplete reporting evidence. "
            "It does not authorize a release tag or a silicon property claim."
        )
        status_color = "statusred"
        status_background = "statusredlight"
    macros = [
        r"\newif\ifreportcertified",
        r"\reportcertifiedtrue" if report_certified else r"\reportcertifiedfalse",
        rf"\newcommand{{\EvidenceStatusTitle}}{{{_latex(status_title)}}}",
        rf"\newcommand{{\EvidenceStatusMessage}}{{{_latex(status_message)}}}",
        rf"\newcommand{{\EvidenceStatusColor}}{{{status_color}}}",
        rf"\newcommand{{\EvidenceStatusBackground}}{{{status_background}}}",
        rf"\newcommand{{\CandidateCommit}}{{\texttt{{{_latex(commit)}}}}}",
        rf"\newcommand{{\CandidateShortCommit}}{{\texttt{{{_latex(commit[:12])}}}}}",
        rf"\newcommand{{\CandidateTree}}{{\texttt{{{_latex(tree)}}}}}",
        rf"\newcommand{{\ReportGeneratedUTC}}{{{_latex(generated)}}}",
        rf"\newcommand{{\ValidationJSONSHA}}{{{_latex_hash(json_sha256)}}}",
        rf"\newcommand{{\ResultsHDFSHA}}{{{_latex_hash(hdf5_sha256)}}}",
        rf"\newcommand{{\ReleaseEvidenceSHA}}{{{_latex_hash(report.get('evidence_sha256', 'not-recorded'))}}}",
        rf"\newcommand{{\CompletionReceiptSHA}}{{{_latex_hash(receipt_sha256 or 'not-recorded')}}}",
        rf"\newcommand{{\ChecksumManifestSHA}}{{{_latex_hash(checksum_sha256)}}}",
        rf"\newcommand{{\CandidateDecisionStatus}}{{{'pass' if report.get('candidate_release_gates_passed') is True else 'fail'}}}",
        rf"\newcommand{{\AuthoritativeDecisionStatus}}{{{'certified' if receipt is not None and receipt.get('release_certified') is True else 'not certified'}}}",
        rf"\newcommand{{\HarmonicDataStatus}}{{{'complete' if harmonic_complete else 'incomplete'}}}",
        r"\newcommand{\PhysicalConvergenceStatus}{false}",
    ]
    _write_text(REPORT_ROOT / "generated/evidence_macros.tex", "\n".join(macros) + "\n")

    rows = []
    for record in records:
        mesh = int(record["mesh"][0])
        end_to_end = record.get("end_to_end")
        end_text = "pass" if isinstance(end_to_end, dict) and end_to_end.get("passed") else "not run"
        rows.append(
            " & ".join(
                (
                    rf"${mesh}^3$",
                    str(record["n_qpoints"]),
                    _format_scientific(record.get("maximum_group_velocity_absolute_error_thz_angstrom")),
                    _format_scientific(record.get("maximum_heat_capacity_absolute_error_ev_per_k")),
                    f"{_voigt_xx(record, 'negfpy_kappa_voigt_w_mk'):.6g}",
                    f"{_voigt_xx(record, 'reference_kappa_voigt_w_mk'):.6g}",
                    end_text,
                )
            )
            + r" \\"
        )
    _write_text(
        REPORT_ROOT / "generated/mesh_table.tex",
        "\\newcommand{\\MeshValidationRows}{%\n"
        + "\n".join(rows)
        + "\n}\n",
    )

    band, dos = _harmonic_table_metrics(report, records)
    harmonic_rows = [
        "Band-path frequencies & "
        + ("pass" if band.get("passed") is True else "missing from current artifact")
        + " & "
        + _format_scientific(band.get("maximum_frequency_absolute_error_thz"))
        + r" THz \\",
        "Total phonon DOS & "
        + ("pass" if dos.get("passed") is True else "missing from current artifact")
        + " & "
        + _format_scientific(dos.get("maximum_absolute_error_states_per_thz_cell"))
        + r" states THz$^{-1}$ cell$^{-1}$ \\",
    ]
    _write_text(
        REPORT_ROOT / "generated/harmonic_table.tex",
        "\\newcommand{\\HarmonicValidationRows}{%\n"
        + "\n".join(harmonic_rows)
        + "\n}\n",
    )

    state = {
        "schema": "negfpy.bulk_rta_report_state.v2",
        "report_certified": report_certified,
        "certification_authority": "release-completion-receipt.json",
        "authoritative_release_certified": bool(
            receipt is not None and receipt.get("release_certified") is True
        ),
        "candidate_release_gates_passed": bool(
            report.get("candidate_release_gates_passed")
        ),
        "provisional_report_release_certified": bool(
            report.get("release_certified")
        ),
        "scientific_input_passed": bool(report.get("scientific_passed")),
        "harmonic_reporting_complete": harmonic_complete,
        "physical_mesh_convergence_certified": False,
        "validation_json_sha256": json_sha256,
        "results_hdf5_sha256": hdf5_sha256,
        "release_evidence_sha256": report.get("evidence_sha256"),
        "release_completion_receipt_sha256": receipt_sha256,
        "sha256sums_sha256": checksum_sha256,
        "missing_final_data": release_gaps,
    }
    _write_text(
        REPORT_ROOT / "generated/normalized_report_state.json",
        json.dumps(state, indent=2, sort_keys=True) + "\n",
    )


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(figure: plt.Figure, name: str) -> None:
    figure.tight_layout(pad=0.9)
    pdf = REPORT_ROOT / "figures" / f"{name}.pdf"
    png = REPORT_ROOT / "figures" / f"{name}.png"
    metadata = {
        "Creator": "NEGFPy deterministic report generator",
        "Producer": "Matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    figure.savefig(pdf, bbox_inches="tight", metadata=metadata)
    figure.savefig(png, dpi=FIGURE_DPI, bbox_inches="tight", metadata={"Software": "NEGFPy"})
    plt.close(figure)


def _gaussian_dos(frequencies: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(frequencies, dtype=np.float64).ravel()
    lower = min(0.0, float(np.min(flat)) - 5.0 * sigma)
    upper = float(np.max(flat)) + 5.0 * sigma
    grid = np.linspace(lower, upper, 900)
    values = np.zeros_like(grid)
    prefactor = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    for block in np.array_split(flat, max(1, math.ceil(flat.size / 512))):
        difference = (grid[:, None] - block[None, :]) / sigma
        values += prefactor * np.sum(np.exp(-0.5 * difference**2), axis=1)
    values /= frequencies.shape[0]
    return grid, values


def _plot_band_and_dos(
    arrays: dict[int, dict[str, np.ndarray]],
    optional: dict[str, Any],
    sigma: float,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.35, 1.0]})
    band = optional.get("band")
    if band is None:
        axes[0].set_axis_off()
        axes[0].text(
            0.5,
            0.55,
            "High-symmetry band-path data\nnot present in provisional artifact",
            ha="center",
            va="center",
            color="#B5443C",
            weight="bold",
            transform=axes[0].transAxes,
        )
        axes[0].text(
            0.5,
            0.34,
            "No curve has been inferred from\nuniform-mesh samples.",
            ha="center",
            va="center",
            color="#67727E",
            transform=axes[0].transAxes,
        )
    else:
        for branch in np.asarray(band["frequencies"]).T:
            axes[0].plot(band["distance"], branch, color="#176B87", lw=1.0)
        for position in band["positions"]:
            axes[0].axvline(position, color="#D9DEE3", lw=0.7, zorder=0)
        axes[0].set_xticks(band["positions"], band["labels"])
        axes[0].set_xlim(float(np.min(band["distance"])), float(np.max(band["distance"])))
        axes[0].set_ylabel("Frequency (THz)")
        axes[0].set_title("Silicon phonon band structure")

    if "dos" in optional:
        dos = optional["dos"]
        axes[1].plot(dos["values"], dos["frequency"], color="#168A85", lw=1.4, label="NEGFPy")
        if "reference" in dos:
            axes[1].plot(
                dos["reference"],
                dos["frequency"],
                color="#D67A2F",
                lw=1.0,
                ls="--",
                label="reference",
            )
            axes[1].legend(frameon=False)
        title = "Validated silicon DOS"
    else:
        grid, values = _gaussian_dos(arrays[11]["frequencies_thz"], sigma)
        axes[1].plot(values, grid, color="#168A85", lw=1.4)
        title = rf"$11^3$ uniform-mesh DOS ($\sigma={sigma:g}$ THz)"
    axes[1].set_xlabel(r"DOS (states THz$^{-1}$ cell$^{-1}$)")
    axes[1].set_ylabel("Frequency (THz)")
    axes[1].set_title(title)
    axes[1].grid(axis="y", color="#E8ECEF", lw=0.6)
    _save_figure(figure, "band_and_dos")


def _plot_rates_and_mfp(arrays: dict[int, dict[str, np.ndarray]], cutoff: float) -> None:
    data = arrays[11]
    frequency = data["frequencies_thz"].ravel()
    gamma = data["gamma_thz_hwhm"][0].ravel()
    lifetime = data["population_lifetime_ps"][0].ravel()
    mean_free_path = data["mean_free_path_angstrom"][0].ravel()
    valid_tau = (frequency > cutoff) & (gamma > 0.0) & (lifetime > 0.0)
    mean_free_path_nm = mean_free_path / 10.0
    valid_mfp = valid_tau & (mean_free_path_nm >= MFP_DISPLAY_FLOOR_NM)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    axes[0].scatter(frequency[valid_tau], lifetime[valid_tau], s=4.0, alpha=0.32, color="#176B87", linewidths=0)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Frequency (THz)")
    axes[0].set_ylabel(r"Population lifetime $\tau$ (ps)")
    axes[0].set_title(r"Three-phonon $\tau=(4\pi\gamma)^{-1}$")
    axes[1].scatter(
        frequency[valid_mfp],
        mean_free_path_nm[valid_mfp],
        s=4.0,
        alpha=0.32,
        color="#C46C32",
        linewidths=0,
    )
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Frequency (THz)")
    axes[1].set_ylabel(r"Mean free path $\Lambda$ (nm)")
    axes[1].set_title(r"Mode mean free path $|\mathbf{v}|\tau$")
    for axis in axes:
        axis.grid(color="#E8ECEF", lw=0.55)
    _save_figure(figure, "rates_and_mfp")


def _plot_conductivity(
    records: list[dict[str, Any]], arrays: dict[int, dict[str, np.ndarray]], cutoff: float
) -> None:
    meshes = np.array([record["mesh"][0] for record in records])
    values = np.array(
        [
            np.asarray(record["negfpy_kappa_voigt_w_mk"], dtype=np.float64)
            .reshape(-1, 6)[0, :3]
            for record in records
        ]
    )
    data = arrays[11]
    frequency = data["frequencies_thz"].ravel()
    mfp_nm = data["mean_free_path_angstrom"][0].ravel() / 10.0
    mode = data["mode_kappa_w_mk"][0].reshape(-1, 3, 3)
    contribution = np.trace(mode, axis1=1, axis2=2) / 3.0
    valid = (frequency > cutoff) & (mfp_nm >= 0.0) & (contribution >= 0.0)
    displayed_mfp = np.maximum(mfp_nm[valid], MFP_DISPLAY_FLOOR_NM)
    order = np.argsort(displayed_mfp, kind="stable")
    x = displayed_mfp[order]
    cumulative = np.cumsum(contribution[valid][order])
    if cumulative.size and cumulative[-1] > 0.0:
        cumulative /= cumulative[-1]
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = ("#176B87", "#168A85", "#C46C32")
    for position, label in enumerate((r"$\kappa_{xx}$", r"$\kappa_{yy}$", r"$\kappa_{zz}$")):
        axes[0].plot(meshes, values[:, position], marker="o", lw=1.2, color=colors[position], label=label)
    axes[0].set_yscale("log")
    axes[0].set_xticks(meshes, [rf"${mesh}^3$" for mesh in meshes])
    axes[0].set_xlabel("Uniform mesh")
    axes[0].set_ylabel(r"SMRTA $\kappa$ (W m$^{-1}$ K$^{-1}$)")
    axes[0].set_title("Implementation-validation values")
    axes[0].legend(frameon=False, ncol=3, loc="best")
    axes[0].grid(color="#E8ECEF", lw=0.55)
    if x.size:
        axes[1].plot(x, cumulative, color="#7B5EA7", lw=1.5)
        axes[1].set_xscale("log")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].set_xlabel(r"Mean-free-path threshold (nm)")
    axes[1].set_ylabel(r"Cumulative $\kappa/\kappa_{\mathrm{total}}$")
    axes[1].set_title(r"$11^3$ cumulative conductivity")
    axes[1].grid(color="#E8ECEF", lw=0.55)
    _save_figure(figure, "conductivity_and_cumulative")


def _render_figures(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    arrays: dict[int, dict[str, np.ndarray]],
    optional: dict[str, Any],
) -> None:
    _configure_plots()
    settings = report.get("settings", {})
    sigma = float(settings.get("sigma_thz", 0.1))
    cutoff = float(settings.get("cutoff_frequency_thz", 1.0e-4))
    _plot_band_and_dos(arrays, optional, sigma)
    _plot_rates_and_mfp(arrays, cutoff)
    _plot_conductivity(records, arrays, cutoff)


def _normalize_pdf_trailer_id(path: Path, document_id: str) -> None:
    """Replace pdfTeX's time-varying trailer ID without changing offsets."""

    if len(document_id) != 32 or any(
        character not in "0123456789ABCDEF" for character in document_id
    ):
        raise ValueError("document_id must contain exactly 32 uppercase hex digits.")
    payload = path.read_bytes()
    pattern = re.compile(
        rb"/ID \[<[0-9A-Fa-f]{32}> <[0-9A-Fa-f]{32}>\]"
    )
    replacement = (
        f"/ID [<{document_id}> <{document_id}>]".encode("ascii")
    )
    normalized, count = pattern.subn(replacement, payload)
    if count != 1 or len(normalized) != len(payload):
        raise RuntimeError("Could not normalize the unique fixed-width PDF trailer ID.")
    temporary = path.with_name(f".{path.name}.normalized.tmp")
    temporary.write_bytes(normalized)
    os.replace(temporary, path)


def _compile_pdf(
    *,
    report_certified: bool,
    render_png: bool,
    document_id: str,
) -> Path:
    pdflatex = shutil.which("pdflatex")
    if pdflatex is None:
        raise RuntimeError("pdflatex is required to compile the report.")
    destination_name = (
        "NEGFPy_bulk_RTA_1.0.0_validation_report.pdf"
        if report_certified
        else "NEGFPy_bulk_RTA_1.0.0_DRAFT_UNCERTIFIED.pdf"
    )
    destination = REPORT_ROOT / "output/pdf" / destination_name
    environment = os.environ.copy()
    environment.update(
        {
            "SOURCE_DATE_EPOCH": SOURCE_DATE_EPOCH,
            "FORCE_SOURCE_DATE": "1",
            "TZ": "UTC",
        }
    )
    with tempfile.TemporaryDirectory(prefix="negfpy-rta-report-") as name:
        build = Path(name)
        log_parts = []
        for _ in range(3):
            completed = subprocess.run(
                [
                    pdflatex,
                    "-no-shell-escape",
                    "-halt-on-error",
                    "-interaction=nonstopmode",
                    f"-output-directory={build}",
                    str(REPORT_ROOT / "report.tex"),
                ],
                cwd=REPORT_ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_parts.append(completed.stdout)
            if completed.returncode != 0:
                _write_text(REPORT_ROOT / "output/pdf/build-console.log", "\n".join(log_parts))
                raise RuntimeError("pdflatex failed; see output/pdf/build-console.log.")
        _write_text(REPORT_ROOT / "output/pdf/build-console.log", "\n".join(log_parts))
        shutil.copyfile(build / "report.pdf", destination)
    _normalize_pdf_trailer_id(destination, document_id)
    if render_png:
        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm is None:
            raise RuntimeError("pdftoppm is required for report page rendering.")
        prefix = REPORT_ROOT / "tmp/pdfs/page"
        for stale in (REPORT_ROOT / "tmp/pdfs").glob("page-*.png"):
            stale.unlink()
        subprocess.run(
            [pdftoppm, "-png", "-r", "150", str(destination), str(prefix)],
            check=True,
            capture_output=True,
            text=True,
        )
    checksums = {destination.name: _sha256(destination)}
    for figure in sorted((REPORT_ROOT / "figures").glob("*.pdf")):
        checksums[str(figure.relative_to(REPORT_ROOT))] = _sha256(figure)
    _write_text(
        REPORT_ROOT / "output/pdf/SHA256SUMS.json",
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--results-hdf5", type=Path, required=True)
    parser.add_argument(
        "--release-mode",
        action="store_true",
        help="Permit certified wording only after every strict reporting gate passes.",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--render-pages", action="store_true")
    arguments = parser.parse_args()

    json_path = _regular_file(arguments.validation_json, label="validation JSON")
    hdf5_path = _regular_file(arguments.results_hdf5, label="results HDF5")
    json_sha256 = _sha256(json_path)
    hdf5_sha256 = _sha256(hdf5_path)
    report = _load_json(json_path)
    records = _validate_json(report)
    evidence, receipt, checksum_path = _load_release_evidence(
        report,
        json_path=json_path,
        hdf5_path=hdf5_path,
    )
    receipt_sha256 = (
        None
        if receipt is None
        else _sha256(json_path.parent / "release-completion-receipt.json")
    )
    checksum_sha256 = _sha256(checksum_path)
    arrays, optional = _load_scientific_arrays(
        hdf5_path,
        report,
        evidence,
        json_sha256,
        records,
    )
    harmonic_complete = _harmonic_reporting_complete(report, records, optional)
    release_gaps = _release_gaps(
        report,
        records,
        harmonic_complete=harmonic_complete,
        receipt=receipt,
    )
    report_certified = bool(
        arguments.release_mode
        and not release_gaps
    )
    if arguments.release_mode and not report_certified:
        raise ValueError(
            "Release-mode rendering was requested, but the strict scientific "
            "and reporting contract did not pass: " + "; ".join(release_gaps)
        )
    _write_generated_files(
        report,
        records,
        receipt=receipt,
        report_certified=report_certified,
        harmonic_complete=harmonic_complete,
        release_gaps=release_gaps,
        json_sha256=json_sha256,
        hdf5_sha256=hdf5_sha256,
        receipt_sha256=receipt_sha256,
        checksum_sha256=checksum_sha256,
    )
    _render_figures(report, records, arrays, optional)
    pdf = None
    if arguments.compile:
        pdf = _compile_pdf(
            report_certified=report_certified,
            render_png=arguments.render_pages,
            document_id=hashlib.sha256(
                ":".join(
                    (
                        json_sha256,
                        str(report.get("evidence_sha256", "not-recorded")),
                        hdf5_sha256,
                        checksum_sha256,
                        receipt_sha256 or "not-recorded",
                    )
                ).encode("ascii")
            ).hexdigest()[:32].upper(),
        )
    summary = {
        "report_certified": report_certified,
        "harmonic_reporting_complete": harmonic_complete,
        "release_gaps": release_gaps,
        "physical_mesh_convergence_certified": False,
        "pdf": None if pdf is None else str(pdf),
        "validation_json_sha256": json_sha256,
        "release_evidence_sha256": report.get("evidence_sha256"),
        "results_hdf5_sha256": hdf5_sha256,
        "sha256sums_sha256": checksum_sha256,
        "release_completion_receipt_sha256": receipt_sha256,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
