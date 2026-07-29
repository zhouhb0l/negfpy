"""Strict final evidence aggregator for the NEGFPy 0.2.0 silicon release.

This module authenticates, independently recomputes, and combines:

* one passed 11x11x11 external phono3py comparison; and
* fresh, full-selection 3x3x3 and 5x5x5 serial/MPI field-equality campaigns.

The final decision does not trust nested pass flags.  It reconstructs the
silicon problem, reruns the authoritative serial 3x3x3 and 5x5x5 linewidth
kernels, independently re-merges target-q artifacts, and re-evaluates the
11x11x11 external comparison.  Consequently, aggregation is itself a
substantial scientific calculation and must run in the pinned release
environment.

The three reports and explicit per-mesh V3 MPI evidence must identify the same
clean source revision, installed NEGFPy 0.2.0 payload, environment lock, IFCs,
crystal mapping, and physical controls.  Every referenced task, execution,
merge, comparison, HDF5, qualification, and target-generation artifact is
re-hashed before and after aggregation.  Only this final layer may set
``release_certified=true``.
"""

from __future__ import annotations

import copy
import hashlib
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ._bulk_result_archive import (
    BulkSerialResultArchiveError,
    read_bulk_serial_result_archive,
)
from ._bulk_target_q_evidence import (
    BulkTargetQEvidenceError,
    verify_bulk_target_q_release_evidence,
)
from ._bulk_target_q_mpi import (
    BulkTargetQControls,
    build_bulk_target_q_campaign,
    bulk_three_phonon_problem_sha256,
    compare_bulk_target_q_with_serial,
)
from ._reference_mpi_provenance import (
    ReferenceMPIProvenanceError,
    validate_reference_mpi_evidence,
)
from ._silicon_bulk_problem import (
    SiliconBulkProblemError,
    load_silicon_bulk_three_phonon_problem,
)
from ._v3_sample_manifest_mpi import (
    V3SampleMPIArtifactError,
    current_v3_sampling_source_identity,
    validate_v3_mpi_orchestration_evidence,
)
from .bulk_harmonic import PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ
from .bulk_three_phonon import compute_three_phonon_linewidths
from .phono3py_reference import read_phono3py_silicon_reference
from .silicon_mpi_oracle_validation import (
    SILICON_MPI_ORACLE_REPORT_SCHEMA,
    SILICON_TARGET_Q_CLI_SCHEMA,
    SILICON_TARGET_Q_MERGE_SCHEMA,
    SiliconMPIOracleValidationConfig,
    SiliconMPIOracleValidationError,
    _canonical_sha256,
    _evaluate_silicon_mpi_oracle_validation,
    _mapping,
    _read_json,
    _regular_file,
    _require_sha256,
    _sha256,
    _verify_envelope,
    _verify_file_record,
    _verify_report_digest,
)
from .silicon_oracle_validation import (
    PINNED_SILICON_SOURCE_COMMIT,
    SILICON_ORACLE_REPORT_SCHEMA,
    _code_identity,
    _repository_identity,
    _runtime_identity,
    write_json_atomic_exclusive,
)
from .translational_fc3 import PHONO3PY_4_4_SYMPREC

SILICON_RELEASE_REPORT_SCHEMA = "negfpy.silicon_0_2_0_release_evidence.v1"
SILICON_RELEASE_VERSION = "0.2.0"
_COMPARISON_SCHEMA = "negfpy.bulk_target_q_comparison_report"
_EMPTY_STATUS_SHA256 = hashlib.sha256(b"").hexdigest()
_CANONICAL_EXACT_COMPARISON_FIELDS = frozenset(
    {
        "linewidths/grid_points",
        "linewidths/band_indices",
        "interactions/grid_points",
        "interactions/band_indices",
        "interactions/q1_indices",
        "interactions/q2_indices",
        "interactions/reciprocal_vectors",
        "interactions/normal_mask",
        "phonons/hermitianized",
        "metadata/linewidths_json",
        "metadata/interactions_json",
    }
)
_CANONICAL_NUMERICAL_COMPARISON_FIELDS = frozenset(
    {
        "linewidths/temperatures",
        "linewidths/gamma_thz_hwhm",
        "linewidths/gamma_normal_thz_hwhm",
        "linewidths/gamma_umklapp_thz_hwhm",
        "linewidths/gamma_angular_fwhm",
        "linewidths/decay_phase_space",
        "linewidths/difference_phase_space",
        "interactions/matrix_elements",
        "interactions/interaction_strengths",
        "phonons/qpoints",
        "phonons/dynamical_matrices",
        "phonons/frequencies",
        "phonons/eigenvectors",
        "phonons/frequency_factor_to_thz",
    }
)
_SHARED_TARGET_ROLE_MAP = {
    "environment_lock": "target_environment_lock",
    "fc2_compact_hdf5": "target_bundle_fc2.compact.hdf5",
    "fc2_full_hdf5": "target_bundle_fc2.full.hdf5",
    "fc3_compact_hdf5": "target_bundle_fc3.compact.hdf5",
    "fc3_full_hdf5": "target_bundle_fc3.full.hdf5",
    "input_manifest": "target_bundle_input-manifest.normalized.json",
    "phono3py_yaml": "target_phono3py_yaml",
}
_REFERENCE_MPI_TARGET_ROLES = {
    "orchestration": "target_bundle_reference-mpi-orchestration.json",
    "evidence_archive": "target_bundle_reference-mpi-evidence.tar",
}
_EXTERNAL_DYNAMIC_REPORT_FIELDS = frozenset(
    {
        "started_at_utc",
        "completed_at_utc",
        "wall_time_seconds",
        "stage_times_seconds",
    }
)


class SiliconReleaseValidationError(RuntimeError):
    """Raised when final 0.2.0 evidence is incomplete or inconsistent."""


def _external_scientific_view(
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the deterministic, schema-pinned scientific decision.

    Live RAM and disk observations are legitimate preflight evidence but
    cannot be byte-identical when the complete validator is rerun.  Their
    pass/fail decisions and all deterministic estimates remain in this view;
    only the instantaneous observations and the digest that covers one such
    observation are removed.
    """

    view = {
        str(key): value
        for key, value in report.items()
        if key not in _EXTERNAL_DYNAMIC_REPORT_FIELDS
    }
    view = copy.deepcopy(view)
    merged_result = _mapping(
        view.get("merged_result"), name="external merged_result"
    )
    load_preflight = _mapping(
        merged_result.get("load_preflight"),
        name="external merged_result.load_preflight",
    )
    if load_preflight.get("passed") is not True:
        raise SiliconReleaseValidationError(
            "External merged-result memory/disk preflight did not pass."
        )
    for key in (
        "mem_available_bytes",
        "limit_bytes",
        "output_filesystem_free_bytes",
    ):
        if key not in load_preflight:
            raise SiliconReleaseValidationError(
                f"External load preflight lacks required field {key!r}."
            )
        load_preflight.pop(key)

    campaign = _mapping(
        view.get("campaign_evidence"),
        name="external campaign_evidence",
    )
    task_remerge = _mapping(
        campaign.get("independent_task_remerge"),
        name="external independent_task_remerge",
    )
    if task_remerge.get("passed") is not True:
        raise SiliconReleaseValidationError(
            "External independent target-q re-merge did not pass."
        )
    disk_preflight = _mapping(
        task_remerge.get("disk_preflight"),
        name="external target-q disk_preflight",
    )
    if disk_preflight.get("passed") is not True:
        raise SiliconReleaseValidationError(
            "External target-q re-merge disk preflight did not pass."
        )
    if "observed_free_bytes" not in disk_preflight:
        raise SiliconReleaseValidationError(
            "External target-q disk preflight lacks observed_free_bytes."
        )
    disk_preflight.pop("observed_free_bytes")
    if "evidence_sha256" not in task_remerge:
        raise SiliconReleaseValidationError(
            "External target-q re-merge lacks its evidence digest."
        )
    task_remerge.pop("evidence_sha256")
    return view


def _revalidate_external_science(
    *,
    release_config: SiliconReleaseValidationConfig,
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Repeat the complete external oracle evaluation from underlying bytes."""

    target_inputs = _mapping(
        report.get("target_inputs"), name="external target_inputs"
    )
    _mapping(
        report.get("campaign_evidence"), name="external campaign_evidence"
    )
    qualification = _mapping(
        report.get("qualification"), name="external qualification"
    )
    target_reference = _mapping(
        report.get("target_reference"), name="external target_reference"
    )
    sample = _mapping(
        report.get("v3_sample_manifest"),
        name="external v3_sample_manifest",
    )
    v3_mpi = _mapping(
        report.get("v3_mpi_orchestration"),
        name="external v3_mpi_orchestration",
    )
    post = _mapping(
        report.get("post_comparison_immutability"),
        name="external post_comparison_immutability",
    )
    critical = _mapping(
        post.get("critical_inputs"),
        name="external critical_inputs",
    )
    expected_critical = {
        "campaign_summary",
        "merge_report",
        "merged_result",
        "qualification_report",
    }
    if set(critical) != expected_critical:
        raise SiliconReleaseValidationError(
            "External report does not bind the exact critical input set "
            "needed for independent scientific revalidation."
        )

    def record_path(
        values: Mapping[str, Any], role: str, *, context: str
    ) -> Path:
        record = _mapping(values.get(role), name=f"{context}.{role}")
        return Path(str(record.get("path")))

    probe = release_config.external_oracle_report_path.with_name(
        f".{release_config.external_oracle_report_path.name}."
        f"{_sha256(release_config.external_oracle_report_path)[:16]}."
        "external-revalidation-probe"
    )
    try:
        config = SiliconMPIOracleValidationConfig(
            qualification_report_path=Path(str(qualification.get("path"))),
            campaign_summary_path=record_path(
                critical,
                "campaign_summary",
                context="external critical_inputs",
            ),
            merge_report_path=record_path(
                critical,
                "merge_report",
                context="external critical_inputs",
            ),
            merged_result_path=record_path(
                critical,
                "merged_result",
                context="external critical_inputs",
            ),
            reference_path=Path(str(target_reference.get("path"))),
            fc2_compact_path=record_path(
                target_inputs,
                "fc2_compact_hdf5",
                context="external target_inputs",
            ),
            fc2_full_path=record_path(
                target_inputs,
                "fc2_full_hdf5",
                context="external target_inputs",
            ),
            fc3_compact_path=record_path(
                target_inputs,
                "fc3_compact_hdf5",
                context="external target_inputs",
            ),
            fc3_full_path=record_path(
                target_inputs,
                "fc3_full_hdf5",
                context="external target_inputs",
            ),
            phono3py_yaml_path=record_path(
                target_inputs,
                "phono3py_yaml",
                context="external target_inputs",
            ),
            input_manifest_path=record_path(
                target_inputs,
                "input_manifest",
                context="external target_inputs",
            ),
            checksums_path=record_path(
                target_inputs,
                "sha256sums",
                context="external target_inputs",
            ),
            environment_lock_path=record_path(
                target_inputs,
                "environment_lock",
                context="external target_inputs",
            ),
            interaction_samples_path=Path(str(sample.get("path"))),
            interaction_samples_orchestration_path=Path(
                str(v3_mpi.get("path"))
            ),
            interaction_samples_run_directory_path=Path(
                str(v3_mpi.get("run_directory"))
            ),
            repository_path=release_config.repository_path,
            output_path=probe,
            require_clean_source=True,
        )
        rebuilt = _evaluate_silicon_mpi_oracle_validation(config)
    except (
        OSError,
        TypeError,
        ValueError,
        SiliconMPIOracleValidationError,
    ) as exc:
        raise SiliconReleaseValidationError(
            "Independent external scientific revalidation failed."
        ) from exc
    if probe.exists() or probe.is_symlink():
        raise SiliconReleaseValidationError(
            "The nonpublishing external evaluator unexpectedly created its "
            "probe report path."
        )
    expected_view = _external_scientific_view(report)
    rebuilt_view = _external_scientific_view(rebuilt)
    if rebuilt_view != expected_view:
        raise SiliconReleaseValidationError(
            "The independently recomputed external scientific decision "
            "differs from the supplied report."
        )
    return {
        "passed": True,
        "method": "complete_nonpublishing_external_oracle_re_evaluation",
        "report_view_sha256": _canonical_sha256(rebuilt_view),
        "comparison_sha256": _canonical_sha256(rebuilt["comparison"]),
        "certification_gates_sha256": _canonical_sha256(
            rebuilt["certification_gates"]
        ),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _directory(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise SiliconReleaseValidationError(
            f"{name} must not be a symbolic link."
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise SiliconReleaseValidationError(
            f"{name} does not identify a readable directory."
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise SiliconReleaseValidationError(
            f"{name} must be a regular non-symlink directory."
        )
    return resolved


@dataclass(frozen=True, slots=True)
class SiliconReleaseValidationConfig:
    """Immutable inputs to the final silicon release-evidence decision."""

    external_oracle_report_path: Path
    equality_3x3_summary_path: Path
    equality_5x5_summary_path: Path
    v3_3x3_manifest_path: Path
    v3_3x3_orchestration_path: Path
    v3_3x3_run_directory_path: Path
    v3_5x5_manifest_path: Path
    v3_5x5_orchestration_path: Path
    v3_5x5_run_directory_path: Path
    repository_path: Path
    output_path: Path

    def __post_init__(self) -> None:
        for name in (
            "external_oracle_report_path",
            "equality_3x3_summary_path",
            "equality_5x5_summary_path",
            "v3_3x3_manifest_path",
            "v3_3x3_orchestration_path",
            "v3_5x5_manifest_path",
            "v3_5x5_orchestration_path",
        ):
            object.__setattr__(
                self,
                name,
                _regular_file(getattr(self, name), name=name),
            )
        if (
            len(
                {
                    self.external_oracle_report_path,
                    self.equality_3x3_summary_path,
                    self.equality_5x5_summary_path,
                    self.v3_3x3_manifest_path,
                    self.v3_3x3_orchestration_path,
                    self.v3_5x5_manifest_path,
                    self.v3_5x5_orchestration_path,
                }
            )
            != 7
        ):
            raise ValueError(
                "Release reports and per-mesh V3 evidence files must be distinct."
            )
        for name in (
            "v3_3x3_run_directory_path",
            "v3_5x5_run_directory_path",
        ):
            object.__setattr__(
                self,
                name,
                _directory(getattr(self, name), name=name),
            )
        if self.v3_3x3_run_directory_path == self.v3_5x5_run_directory_path:
            raise ValueError(
                "The 3x3x3 and 5x5x5 V3 run directories must be distinct."
            )
        object.__setattr__(
            self,
            "repository_path",
            _directory(self.repository_path, name="repository_path"),
        )
        output = Path(self.output_path).expanduser().resolve()
        if output.exists():
            raise FileExistsError(
                "output_path already exists; release reports are immutable."
            )
        if output.is_relative_to(self.repository_path):
            raise ValueError(
                "output_path must be outside the clean source repository."
            )
        object.__setattr__(self, "output_path", output)


class _EvidenceTracker:
    """Track exact file identities and fail if any byte/stat changes."""

    def __init__(self) -> None:
        self._records: dict[Path, dict[str, Any]] = {}

    def file_record(
        self,
        value: Any,
        *,
        name: str,
        expected_path: Path | None = None,
        extra_fields: Sequence[str] = (),
    ) -> dict[str, Any]:
        record = _verify_file_record(
            value,
            name=name,
            expected_path=expected_path,
            extra_fields=extra_fields,
        )
        path = Path(str(record["path"]))
        identity = {
            key: record[key]
            for key in (
                "path",
                "sha256",
                "size_bytes",
                "device",
                "inode",
                "mtime_ns",
            )
        }
        previous = self._records.get(path)
        if previous is not None and previous != identity:
            raise SiliconReleaseValidationError(
                f"Conflicting identities were supplied for {path}."
            )
        self._records[path] = identity
        return record

    def path_sha256(
        self,
        path_value: Any,
        sha256_value: Any,
        *,
        name: str,
    ) -> dict[str, Any]:
        path = _regular_file(str(path_value), name=f"{name}.path")
        digest = _require_sha256(sha256_value, name=f"{name}.sha256")
        stat = path.stat()
        record = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
        if record["sha256"] != digest:
            raise SiliconReleaseValidationError(
                f"{name} SHA256 differs from current bytes."
            )
        previous = self._records.get(path)
        if previous is not None and previous != record:
            raise SiliconReleaseValidationError(
                f"Conflicting identities were supplied for {path}."
            )
        self._records[path] = record
        return record

    def snapshot_path(self, path: Path, *, name: str) -> dict[str, Any]:
        resolved = _regular_file(path, name=name)
        stat = resolved.stat()
        return self.path_sha256(
            resolved,
            _sha256(resolved),
            name=name,
        ) | {
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }

    def recheck(self) -> None:
        for path, expected in self._records.items():
            actual = _verify_file_record(
                expected,
                name=f"final evidence recheck {path}",
                expected_path=path,
            )
            if actual != expected:
                raise SiliconReleaseValidationError(
                    f"Evidence changed during aggregation: {path}."
                )

    def summary(self) -> dict[str, Any]:
        return {
            "file_count": len(self._records),
            "files": [
                self._records[path]
                for path in sorted(self._records, key=lambda item: str(item))
            ],
        }


def _require_release_runtime(
    repository: Mapping[str, Any],
    runtime: Mapping[str, Any],
    imported_code: Mapping[str, Any],
) -> None:
    packages = _mapping(runtime.get("packages"), name="runtime.packages")
    record = _mapping(imported_code.get("record"), name="imported_code.record")
    if (
        repository.get("clean") is not True
        or repository.get("status_sha256") != _EMPTY_STATUS_SHA256
        or imported_code.get("passed") is not True
        or packages.get("negfpy") != SILICON_RELEASE_VERSION
        or record.get("installed_package_version") != SILICON_RELEASE_VERSION
        or record.get("declared_package_version") != SILICON_RELEASE_VERSION
        or record.get("package_version") != SILICON_RELEASE_VERSION
        or record.get("package_version_consistent") is not True
    ):
        raise SiliconReleaseValidationError(
            "Final aggregation requires the exact clean installed NEGFPy "
            "0.2.0 source payload."
        )


def _source_matches_current(
    source_repository: Any,
    imported: Any,
    runtime_record: Any,
    *,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    runtime: Mapping[str, Any],
    name: str,
) -> None:
    source = _mapping(source_repository, name=f"{name}.source_repository")
    code = _mapping(imported, name=f"{name}.imported_code")
    report_runtime = _mapping(runtime_record, name=f"{name}.runtime")
    if (
        source.get("clean") is not True
        or source.get("head_commit") != repository.get("head_commit")
        or source.get("tree_hash") != repository.get("tree_hash")
        or code.get("passed") is not True
        or code.get("record") != imported_code.get("record")
        or dict(report_runtime) != dict(runtime)
    ):
        raise SiliconReleaseValidationError(
            f"{name} does not identify the current clean 0.2.0 payload."
        )


def _safe_relative_path(value: Any, *, name: str) -> Path:
    if not isinstance(value, str):
        raise SiliconReleaseValidationError(
            f"{name} must be a relative POSIX path."
        )
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or "\\" in value
        or path.as_posix() != value
    ):
        raise SiliconReleaseValidationError(
            f"{name} is not a safe relative POSIX path."
        )
    return path


def _contained_assignment_path(
    run_root: Path,
    directory: str,
    relative: Path,
    *,
    name: str,
) -> Path:
    base = run_root / directory
    if base.is_symlink() or not base.is_dir():
        raise SiliconReleaseValidationError(
            f"{name} evidence directory is missing or a symlink."
        )
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise SiliconReleaseValidationError(
                f"{name} traverses a symbolic link."
            )
    path = _regular_file(cursor, name=name)
    if not path.is_relative_to(base.resolve(strict=True)):
        raise SiliconReleaseValidationError(
            f"{name} escapes its campaign evidence directory."
        )
    return path


def _track_assignment_files(
    execution: Mapping[str, Any],
    *,
    run_root: Path,
    tracker: _EvidenceTracker,
    name: str,
) -> None:
    assignments = execution.get("assignments")
    if not isinstance(assignments, list):
        raise SiliconReleaseValidationError(
            f"{name}.assignments must be an array."
        )
    for position, raw in enumerate(assignments):
        assignment = _mapping(raw, name=f"{name}.assignments[{position}]")
        checkpoint = _safe_relative_path(
            assignment.get("checkpoint_relative_path"),
            name=f"{name}.checkpoint_relative_path",
        )
        checkpoint_path = _contained_assignment_path(
            run_root,
            "task-outcomes",
            checkpoint,
            name=f"{name}.checkpoint[{position}]",
        )
        tracker.path_sha256(
            checkpoint_path,
            assignment.get("checkpoint_file_sha256"),
            name=f"{name}.checkpoint[{position}]",
        )
        artifact_value = assignment.get("artifact_relative_path")
        artifact_sha256 = assignment.get("artifact_sha256")
        if artifact_value is None or artifact_sha256 is None:
            if assignment.get("status") == "completed":
                raise SiliconReleaseValidationError(
                    f"{name} completed task lacks an artifact."
                )
            continue
        artifact = _safe_relative_path(
            artifact_value,
            name=f"{name}.artifact_relative_path",
        )
        artifact_path = _contained_assignment_path(
            run_root,
            "task-artifacts",
            artifact,
            name=f"{name}.artifact[{position}]",
        )
        tracker.path_sha256(
            artifact_path,
            artifact_sha256,
            name=f"{name}.artifact[{position}]",
        )


def _validate_serial_qualification(
    path: Path,
    *,
    tracker: _EvidenceTracker,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    report = _read_json(path, name="serial qualification")
    gates = report.get("certification_gates")
    if (
        report.get("schema") != SILICON_ORACLE_REPORT_SCHEMA
        or report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("validation_passed") is not True
        or report.get("certification_eligible") is not True
        or not isinstance(gates, Mapping)
        or not gates
        or any(value is not True for value in gates.values())
    ):
        raise SiliconReleaseValidationError(
            "Equality campaign qualification is not fully passed."
        )
    _source_matches_current(
        report.get("source_repository"),
        report.get("imported_code_provenance"),
        report.get("runtime"),
        repository=repository,
        imported_code=imported_code,
        runtime=runtime,
        name="serial qualification",
    )
    inputs_raw = _mapping(
        report.get("inputs"), name="serial qualification.inputs"
    )
    inputs = {
        str(role): tracker.file_record(
            value,
            name=f"serial qualification.inputs.{role}",
        )
        for role, value in inputs_raw.items()
    }
    return report, inputs


def _require_qualification_target_match(
    qualification_inputs: Mapping[str, Mapping[str, Any]],
    target_records: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
) -> None:
    mismatches: dict[str, dict[str, Any]] = {}
    for qualification_role, target_role in _SHARED_TARGET_ROLE_MAP.items():
        qualified = qualification_inputs.get(qualification_role)
        target = target_records.get(target_role)
        qualified_sha = None if qualified is None else qualified.get("sha256")
        target_sha = None if target is None else target.get("sha256")
        if qualified is None or target is None or qualified_sha != target_sha:
            mismatches[qualification_role] = {
                "qualification_sha256": qualified_sha,
                "target_role": target_role,
                "target_sha256": target_sha,
            }
    if mismatches:
        raise SiliconReleaseValidationError(
            f"{name} serial qualification and target IFC/YAML/manifest/"
            f"environment identities differ: {mismatches}."
        )


def _validate_equality_reference_mpi(
    *,
    expected_mesh: tuple[int, int, int],
    target_records: Mapping[str, Mapping[str, Any]],
    reference: Any,
) -> dict[str, Any]:
    """Independently authenticate one equality mesh's reference-MPI bundle."""

    if tuple(reference.mesh) != expected_mesh:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} reference-MPI evidence has the wrong mesh."
        )
    missing = sorted(
        role
        for role in _REFERENCE_MPI_TARGET_ROLES.values()
        if role not in target_records
    )
    if missing:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} target bundle lacks mandatory reference-MPI "
            f"evidence: {missing}."
        )
    orchestration_record = dict(
        target_records[_REFERENCE_MPI_TARGET_ROLES["orchestration"]]
    )
    archive_record = dict(
        target_records[_REFERENCE_MPI_TARGET_ROLES["evidence_archive"]]
    )
    orchestration_path = Path(str(orchestration_record["path"]))
    orchestration_document = _read_json(
        orchestration_path,
        name=f"{expected_mesh} reference MPI orchestration",
    )
    try:
        validated = validate_reference_mpi_evidence(
            orchestration_path.parent,
            orchestration_document,
            expected_mesh=reference.mesh,
            expected_grid_points=reference.grid_points_grg.tolist(),
            expected_band_indices=reference.band_indices.tolist(),
            expected_n_qpoints=reference.n_qpoints,
            expected_n_bands=reference.n_bands,
            require_mpi=True,
        )
    except ReferenceMPIProvenanceError as exc:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} reference-MPI evidence is invalid: {exc}"
        ) from exc
    validated_archive = _mapping(
        validated.get("evidence_archive"),
        name=f"{expected_mesh} reference MPI evidence archive",
    )
    merged_reference = _mapping(
        validated.get("merged_reference"),
        name=f"{expected_mesh} reference MPI merged reference",
    )
    reference_record = target_records[
        "target_bundle_si-phono3py-reference.hdf5"
    ]
    if (
        Path(str(validated_archive.get("path"))).resolve()
        != Path(str(archive_record["path"]))
        or validated_archive.get("sha256") != archive_record.get("sha256")
        or validated_archive.get("size_bytes")
        != archive_record.get("size_bytes")
        or merged_reference.get("sha256") != reference_record.get("sha256")
        or merged_reference.get("size_bytes")
        != reference_record.get("size_bytes")
        or validated.get("full_q_selection") is not True
        or validated.get("full_band_selection") is not True
        or validated.get("floating_point_mpi_reductions") is not False
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} reference-MPI evidence differs from the "
            "checksummed target bundle."
        )
    return {
        "campaign_sha256": validated["campaign_sha256"],
        "orchestration": orchestration_record,
        "evidence_archive": archive_record,
        "evidence_members": validated["evidence_members"],
        "task_count": validated["task_count"],
        "execution_count": validated["execution_count"],
        "execution_ids": validated["execution_ids"],
        "restart": validated["restart"],
        "worker_bindings_by_execution": validated[
            "worker_bindings_by_execution"
        ],
        "bundle_identity": validated["bundle_identity"],
        "merged_reference": validated["merged_reference"],
        "canonical_merge": validated["canonical_merge"],
        "full_q_selection": True,
        "full_band_selection": True,
        "floating_point_mpi_reductions": False,
    }


def _v3_mpi_facts(value: Any) -> dict[str, Any]:
    return {
        "orchestration_file_sha256": value.file_sha256,
        "payload_sha256": value.payload_sha256,
        "campaign_sha256": value.campaign_sha256,
        "source_identity_sha256": value.source_identity_sha256,
        "source_authenticated": value.source_authenticated,
        "source_revision": value.source_revision,
        "source_tree_sha256": value.source_tree_sha256,
        "environment_sha256": value.environment_sha256,
        "mesh": list(value.mesh),
        "target_grid_points": list(value.target_grid_points),
        "target_bands": list(value.target_bands),
        "selected_execution_ids": list(value.selected_execution_ids),
        "selected_communicator_sizes": list(value.selected_communicator_sizes),
        "selected_modes": list(value.selected_modes),
        "restart_resume_recorded": value.restart_resume_recorded,
        "physical_core_binding_passed": value.physical_core_binding_passed,
        "mpi_origin_lineage_passed": value.mpi_origin_lineage_passed,
    }


def _validate_equality_v3_mpi(
    *,
    expected_mesh: tuple[int, int, int],
    manifest_path: Path,
    orchestration_path: Path,
    run_directory: Path,
    qualification_inputs: Mapping[str, Mapping[str, Any]],
    reference: Any,
    tracker: _EvidenceTracker,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    environment_sha256: str,
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Authenticate explicit V3-MPI evidence for one equality mesh."""

    qualified_manifest = qualification_inputs.get("interaction_samples")
    if qualified_manifest is None:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} serial qualification lacks interaction_samples."
        )
    qualified_path = Path(str(qualified_manifest["path"]))
    if manifest_path != qualified_path:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} explicit V3 manifest is not the exact manifest "
            "used by the serial qualification."
        )
    manifest_record = tracker.path_sha256(
        manifest_path,
        qualified_manifest.get("sha256"),
        name=f"{expected_mesh} V3 scientific manifest",
    )
    identity_fields = (
        "path",
        "sha256",
        "size_bytes",
        "device",
        "inode",
        "mtime_ns",
    )
    if any(
        manifest_record[field] != qualified_manifest.get(field)
        for field in identity_fields
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} explicit V3 manifest identity differs from the "
            "serial qualification."
        )
    orchestration_record = tracker.snapshot_path(
        orchestration_path,
        name=f"{expected_mesh} V3 MPI orchestration",
    )
    try:
        validated = validate_v3_mpi_orchestration_evidence(
            manifest_path,
            orchestration_path,
            run_directory,
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=str(repository["head_commit"]),
            expected_source_tree_sha256=str(
                imported_code["record"]["source_tree_sha256"]
            ),
            expected_environment_sha256=environment_sha256,
            expected_source_identity=source_identity,
            expected_mesh=reference.mesh,
            expected_target_grid_points=reference.grid_points_grg,
            expected_target_bands=reference.band_indices,
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} V3 MPI evidence is invalid: {exc}"
        ) from exc
    facts = _v3_mpi_facts(validated)
    if (
        validated.path != orchestration_path
        or facts["orchestration_file_sha256"] != orchestration_record["sha256"]
        or facts["source_authenticated"] is not True
        or facts["source_revision"] != repository["head_commit"]
        or facts["source_tree_sha256"]
        != imported_code["record"]["source_tree_sha256"]
        or facts["environment_sha256"] != environment_sha256
        or facts["mesh"] != list(expected_mesh)
        or facts["target_grid_points"] != reference.grid_points_grg.tolist()
        or facts["target_bands"] != reference.band_indices.tolist()
        or any(mode != "mpi" for mode in facts["selected_modes"])
        or facts["physical_core_binding_passed"] is not True
        or facts["mpi_origin_lineage_passed"] is not True
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} V3 MPI evidence differs from the explicit release inputs."
        )
    return {
        "scientific_manifest": manifest_record,
        "orchestration": orchestration_record,
        "run_directory": str(run_directory),
        **facts,
        "floating_point_mpi_reductions": False,
        "canonical_merge_revalidated": True,
    }


def _recheck_equality_v3_mpi(
    evidence: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any],
) -> None:
    """Repeat the strict V3 evidence validation at the publication barrier."""

    try:
        validated = validate_v3_mpi_orchestration_evidence(
            Path(
                str(
                    _mapping(
                        evidence["scientific_manifest"], name="V3 manifest"
                    )["path"]
                )
            ),
            Path(
                str(
                    _mapping(
                        evidence["orchestration"], name="V3 orchestration"
                    )["path"]
                )
            ),
            Path(str(evidence["run_directory"])),
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=str(evidence["source_revision"]),
            expected_source_tree_sha256=str(evidence["source_tree_sha256"]),
            expected_environment_sha256=str(evidence["environment_sha256"]),
            expected_source_identity=source_identity,
            expected_mesh=evidence["mesh"],
            expected_target_grid_points=evidence["target_grid_points"],
            expected_target_bands=evidence["target_bands"],
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconReleaseValidationError(
            f"Equality V3 MPI evidence failed its final recheck: {exc}"
        ) from exc
    if _v3_mpi_facts(validated) != {
        key: evidence[key] for key in _v3_mpi_facts(validated)
    }:
        raise SiliconReleaseValidationError(
            "Equality V3 MPI evidence changed during final aggregation."
        )


def _validate_external_report(
    config: SiliconReleaseValidationConfig,
    *,
    tracker: _EvidenceTracker,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    tracker.snapshot_path(
        config.external_oracle_report_path,
        name="external_oracle_report_path",
    )
    report = _read_json(
        config.external_oracle_report_path,
        name="external MPI oracle report",
    )
    gates = _mapping(
        report.get("certification_gates"),
        name="external certification_gates",
    )
    release_gates = _mapping(
        report.get("release_gates"), name="external release_gates"
    )
    if (
        report.get("schema") != SILICON_MPI_ORACLE_REPORT_SCHEMA
        or report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("validation_passed") is not True
        or report.get("certification_eligible") is not True
        or report.get("release_certified") is not False
        or report.get("release_aggregation_required") is not True
        or not gates
        or any(value is not True for value in gates.values())
        or release_gates.get("external_reference_certification") is not True
        or release_gates.get("fresh_3x3_full_field_serial_mpi_equality")
        is not False
        or release_gates.get("fresh_5x5_full_field_serial_mpi_equality")
        is not False
        or release_gates.get("release_aggregator_complete") is not False
    ):
        raise SiliconReleaseValidationError(
            "The 11x11x11 external report is not eligible for aggregation."
        )
    _source_matches_current(
        report.get("source_repository"),
        report.get("imported_code_provenance"),
        report.get("runtime"),
        repository=repository,
        imported_code=imported_code,
        runtime=runtime,
        name="external MPI oracle",
    )
    execution = _mapping(report.get("execution"), name="external execution")
    if (
        execution.get("monolithic_negfpy_recomputation") is not False
        or execution.get("fresh_mpi_equality_claimed") is not False
        or execution.get("release_level_mpi_equality_claimed") is not False
    ):
        raise SiliconReleaseValidationError(
            "External report overclaims monolithic or MPI equality work."
        )

    target_inputs_raw = _mapping(
        report.get("target_inputs"), name="external target_inputs"
    )
    target_inputs = {
        str(role): tracker.file_record(
            value, name=f"external target_inputs.{role}"
        )
        for role, value in target_inputs_raw.items()
    }
    qualification = _mapping(
        report.get("qualification"), name="external qualification"
    )
    tracker.path_sha256(
        qualification.get("path"),
        qualification.get("sha256"),
        name="external qualification",
    )
    target_reference = _mapping(
        report.get("target_reference"), name="external target_reference"
    )
    tracker.path_sha256(
        target_reference.get("path"),
        target_reference.get("sha256"),
        name="external target_reference",
    )
    if tuple(target_reference.get("mesh", ())) != (11, 11, 11):
        raise SiliconReleaseValidationError(
            "External scientific report must be the complete 11x11x11 mesh."
        )
    campaign = _mapping(
        report.get("campaign_evidence"), name="external campaign_evidence"
    )
    campaign_records: dict[str, dict[str, Any]] = {}
    for role, extra in (
        ("task_manifest", ("campaign_manifest_sha256",)),
        ("execution_manifest", ("execution_sha256",)),
        ("merge_report", ("report_sha256",)),
        ("merged_hdf5", ()),
    ):
        campaign_records[role] = tracker.file_record(
            campaign.get(role),
            name=f"external campaign_evidence.{role}",
            extra_fields=extra,
        )
    task_document = _read_json(
        Path(str(campaign_records["task_manifest"]["path"])),
        name="external task manifest",
    )
    _verify_envelope(task_document, name="external task manifest")
    execution_document = _read_json(
        Path(str(campaign_records["execution_manifest"]["path"])),
        name="external execution manifest",
    )
    execution_payload = _verify_envelope(
        execution_document, name="external execution manifest"
    )
    merge_document = _read_json(
        Path(str(campaign_records["merge_report"]["path"])),
        name="external merge report",
    )
    merge_payload = _verify_report_digest(
        merge_document, name="external merge report"
    )
    if (
        task_document["payload_sha256"]
        != campaign.get("campaign_manifest_sha256")
        or campaign_records["task_manifest"].get("campaign_manifest_sha256")
        != campaign.get("campaign_manifest_sha256")
        or execution_payload.get("manifest_sha256")
        != campaign.get("campaign_manifest_sha256")
        or campaign_records["execution_manifest"].get("execution_sha256")
        != execution_document["payload_sha256"]
        or merge_payload.get("campaign_manifest_sha256")
        != campaign.get("campaign_manifest_sha256")
        or campaign_records["merge_report"].get("report_sha256")
        != merge_payload.get("report_sha256")
        or merge_payload.get("merged_artifact_sha256")
        != campaign_records["merged_hdf5"]["sha256"]
        or Path(str(merge_payload.get("merged_artifact_path"))).resolve()
        != Path(str(campaign_records["merged_hdf5"]["path"]))
    ):
        raise SiliconReleaseValidationError(
            "External task/execution/merge/HDF5 evidence is inconsistent."
        )
    _track_assignment_files(
        execution_payload,
        run_root=Path(str(campaign_records["task_manifest"]["path"])).parent,
        tracker=tracker,
        name="external execution",
    )
    sample = _mapping(
        report.get("v3_sample_manifest"),
        name="external v3_sample_manifest",
    )
    sample_record = tracker.path_sha256(
        sample.get("path"),
        sample.get("sha256"),
        name="external V3 sample manifest",
    )
    v3_mpi = _mapping(
        report.get("v3_mpi_orchestration"),
        name="external v3_mpi_orchestration",
    )
    v3_orchestration_record = tracker.path_sha256(
        v3_mpi.get("path"),
        v3_mpi.get("sha256"),
        name="external V3 MPI orchestration",
    )
    if v3_orchestration_record["size_bytes"] != v3_mpi.get("size_bytes"):
        raise SiliconReleaseValidationError(
            "External V3 MPI orchestration size evidence is inconsistent."
        )
    v3_run_directory = _directory(
        str(v3_mpi.get("run_directory")),
        name="external V3 MPI run_directory",
    )
    current_v3_source_identity = current_v3_sampling_source_identity(
        config.repository_path,
        require_clean=True,
    )
    reference_for_v3 = read_phono3py_silicon_reference(
        Path(str(target_reference["path"]))
    )
    try:
        validated_v3_mpi = validate_v3_mpi_orchestration_evidence(
            Path(str(sample_record["path"])),
            Path(str(v3_orchestration_record["path"])),
            v3_run_directory,
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=repository["head_commit"],
            expected_source_tree_sha256=imported_code["record"][
                "source_tree_sha256"
            ],
            expected_environment_sha256=target_inputs["environment_lock"][
                "sha256"
            ],
            expected_source_identity=current_v3_source_identity,
            expected_mesh=reference_for_v3.mesh,
            expected_target_grid_points=(reference_for_v3.grid_points_grg),
            expected_target_bands=reference_for_v3.band_indices,
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconReleaseValidationError(
            f"External V3 MPI orchestration is invalid: {exc}"
        ) from exc
    if (
        v3_mpi.get("payload_sha256") != validated_v3_mpi.payload_sha256
        or v3_mpi.get("campaign_sha256") != validated_v3_mpi.campaign_sha256
        or v3_mpi.get("source_identity_sha256")
        != validated_v3_mpi.source_identity_sha256
        or v3_mpi.get("source_authenticated")
        is not validated_v3_mpi.source_authenticated
        or v3_mpi.get("source_revision") != validated_v3_mpi.source_revision
        or v3_mpi.get("source_tree_sha256")
        != validated_v3_mpi.source_tree_sha256
        or v3_mpi.get("environment_sha256")
        != validated_v3_mpi.environment_sha256
        or v3_mpi.get("mesh") != list(validated_v3_mpi.mesh)
        or v3_mpi.get("target_grid_points")
        != list(validated_v3_mpi.target_grid_points)
        or v3_mpi.get("target_bands") != list(validated_v3_mpi.target_bands)
        or v3_mpi.get("selected_execution_ids")
        != list(validated_v3_mpi.selected_execution_ids)
        or v3_mpi.get("selected_communicator_sizes")
        != list(validated_v3_mpi.selected_communicator_sizes)
        or v3_mpi.get("selected_modes")
        != list(validated_v3_mpi.selected_modes)
        or v3_mpi.get("restart_resume_recorded")
        is not validated_v3_mpi.restart_resume_recorded
        or v3_mpi.get("physical_core_binding_passed")
        is not validated_v3_mpi.physical_core_binding_passed
        or v3_mpi.get("mpi_origin_lineage_passed")
        is not validated_v3_mpi.mpi_origin_lineage_passed
        or not validated_v3_mpi.physical_core_binding_passed
        or not validated_v3_mpi.mpi_origin_lineage_passed
        or v3_mpi.get("floating_point_mpi_reductions") is not False
        or v3_mpi.get("canonical_merge_revalidated") is not True
    ):
        raise SiliconReleaseValidationError(
            "External V3 MPI report differs from the revalidated task evidence."
        )
    generation = _mapping(
        target_reference.get("generation_environment"),
        name="external generation_environment",
    )
    generation_records = _mapping(
        generation.get("records"),
        name="external generation_environment.records",
    )
    for stage, raw in generation_records.items():
        record = _mapping(raw, name=f"external generation record {stage}")
        tracker.path_sha256(
            record.get("path"),
            record.get("sha256"),
            name=f"external generation record {stage}",
        )
    reference_mpi = _mapping(
        target_reference.get("task_mpi_generation"),
        name="external target_reference.task_mpi_generation",
    )
    if (
        reference_mpi.get("authenticated") is not True
        or reference_mpi.get("full_q_selection") is not True
        or reference_mpi.get("full_band_selection") is not True
        or reference_mpi.get("floating_point_mpi_reductions") is not False
    ):
        raise SiliconReleaseValidationError(
            "External reference-MPI generation is not fully authenticated."
        )
    orchestration_record = tracker.path_sha256(
        _mapping(
            reference_mpi.get("orchestration"),
            name="external reference MPI orchestration",
        ).get("path"),
        _mapping(
            reference_mpi.get("orchestration"),
            name="external reference MPI orchestration",
        ).get("sha256"),
        name="external reference MPI orchestration",
    )
    archive_report = _mapping(
        reference_mpi.get("evidence_archive"),
        name="external reference MPI evidence_archive",
    )
    archive_record = tracker.path_sha256(
        archive_report.get("path"),
        archive_report.get("sha256"),
        name="external reference MPI evidence archive",
    )
    if orchestration_record["size_bytes"] != _mapping(
        reference_mpi.get("orchestration"),
        name="external reference MPI orchestration",
    ).get("size_bytes") or archive_record["size_bytes"] != archive_report.get(
        "size_bytes"
    ):
        raise SiliconReleaseValidationError(
            "External reference-MPI file size evidence is inconsistent."
        )
    orchestration_document = _read_json(
        Path(str(orchestration_record["path"])),
        name="external reference MPI orchestration",
    )
    reference_object = read_phono3py_silicon_reference(
        Path(str(target_reference["path"]))
    )
    try:
        validated_reference_mpi = validate_reference_mpi_evidence(
            Path(str(orchestration_record["path"])).parent,
            orchestration_document,
            expected_mesh=reference_object.mesh,
            expected_grid_points=reference_object.grid_points_grg.tolist(),
            expected_band_indices=reference_object.band_indices.tolist(),
            expected_n_qpoints=reference_object.n_qpoints,
            expected_n_bands=reference_object.n_bands,
            require_mpi=True,
        )
    except ReferenceMPIProvenanceError as exc:
        raise SiliconReleaseValidationError(
            f"External reference-MPI archive is invalid: {exc}"
        ) from exc
    if (
        reference_mpi.get("campaign_sha256")
        != validated_reference_mpi["campaign_sha256"]
        or reference_mpi.get("task_count")
        != validated_reference_mpi["task_count"]
        or reference_mpi.get("execution_count")
        != validated_reference_mpi["execution_count"]
        or reference_mpi.get("execution_ids")
        != validated_reference_mpi["execution_ids"]
        or reference_mpi.get("restart") != validated_reference_mpi["restart"]
        or reference_mpi.get("worker_bindings_by_execution")
        != validated_reference_mpi["worker_bindings_by_execution"]
        or reference_mpi.get("bundle_identity")
        != validated_reference_mpi["bundle_identity"]
        or reference_mpi.get("merged_reference")
        != validated_reference_mpi["merged_reference"]
        or reference_mpi.get("canonical_merge")
        != validated_reference_mpi["canonical_merge"]
        or reference_mpi.get("evidence_members")
        != validated_reference_mpi["evidence_members"]
        or reference_mpi.get("orchestration_sha256")
        != orchestration_record["sha256"]
        or archive_report.get("sha256")
        != validated_reference_mpi["evidence_archive"]["sha256"]
    ):
        raise SiliconReleaseValidationError(
            "External reference-MPI report differs from archived evidence."
        )
    scientific_revalidation = _revalidate_external_science(
        release_config=config,
        report=report,
    )
    return {
        "report_sha256": _sha256(config.external_oracle_report_path),
        "source_commit": repository["head_commit"],
        "source_tree": repository["tree_hash"],
        "mesh": list(target_reference["mesh"]),
        "target_inputs": target_inputs,
        "temperatures_k": list(target_reference["temperatures_k"]),
        "sigma_thz": float(target_reference["sigma_thz"]),
        "cutoff_frequency_thz": float(
            target_reference["cutoff_frequency_thz"]
        ),
        "campaign_manifest_sha256": campaign["campaign_manifest_sha256"],
        "comparison": report["comparison"],
        "independent_scientific_revalidation": scientific_revalidation,
        "v3_mpi_sampling": {
            "scientific_manifest": sample_record,
            "orchestration": v3_orchestration_record,
            "run_directory": str(v3_run_directory),
            "payload_sha256": validated_v3_mpi.payload_sha256,
            "campaign_sha256": validated_v3_mpi.campaign_sha256,
            "source_identity_sha256": (
                validated_v3_mpi.source_identity_sha256
            ),
            "source_authenticated": (validated_v3_mpi.source_authenticated),
            "source_revision": validated_v3_mpi.source_revision,
            "source_tree_sha256": (validated_v3_mpi.source_tree_sha256),
            "environment_sha256": (validated_v3_mpi.environment_sha256),
            "mesh": list(validated_v3_mpi.mesh),
            "target_grid_points": list(validated_v3_mpi.target_grid_points),
            "target_bands": list(validated_v3_mpi.target_bands),
            "selected_execution_ids": list(
                validated_v3_mpi.selected_execution_ids
            ),
            "selected_communicator_sizes": list(
                validated_v3_mpi.selected_communicator_sizes
            ),
            "selected_modes": list(validated_v3_mpi.selected_modes),
            "restart_resume_recorded": (
                validated_v3_mpi.restart_resume_recorded
            ),
            "physical_core_binding_passed": (
                validated_v3_mpi.physical_core_binding_passed
            ),
            "mpi_origin_lineage_passed": (
                validated_v3_mpi.mpi_origin_lineage_passed
            ),
            "floating_point_mpi_reductions": False,
            "canonical_merge_revalidated": True,
        },
        "reference_mpi_generation": {
            "campaign_sha256": validated_reference_mpi["campaign_sha256"],
            "orchestration": orchestration_record,
            "evidence_archive": archive_record,
            "evidence_members": validated_reference_mpi["evidence_members"],
            "task_count": validated_reference_mpi["task_count"],
            "execution_count": validated_reference_mpi["execution_count"],
            "execution_ids": validated_reference_mpi["execution_ids"],
            "restart": validated_reference_mpi["restart"],
            "worker_bindings_by_execution": validated_reference_mpi[
                "worker_bindings_by_execution"
            ],
            "bundle_identity": validated_reference_mpi["bundle_identity"],
            "merged_reference": validated_reference_mpi["merged_reference"],
            "canonical_merge": validated_reference_mpi["canonical_merge"],
            "full_q_selection": True,
            "full_band_selection": True,
            "floating_point_mpi_reductions": False,
        },
    }


def _validate_comparison_report(
    value: Any,
    *,
    merged_artifact_sha256: str,
) -> dict[str, Any]:
    report = dict(_verify_report_digest(value, name="monolithic comparison"))
    exact = _mapping(
        report.get("exact_field_passed"),
        name="comparison.exact_field_passed",
    )
    numerical = _mapping(
        report.get("numerical_field_passed"),
        name="comparison.numerical_field_passed",
    )
    absolute = _mapping(
        report.get("maximum_absolute_error"),
        name="comparison.maximum_absolute_error",
    )
    relative = _mapping(
        report.get("maximum_relative_error"),
        name="comparison.maximum_relative_error",
    )
    if (
        report.get("schema") != _COMPARISON_SCHEMA
        or report.get("schema_version") != 1
        or report.get("passed") is not True
        or float(report.get("rtol", -1.0)) != 1.0e-10
        or float(report.get("atol", -1.0)) != 1.0e-12
        or set(exact) != _CANONICAL_EXACT_COMPARISON_FIELDS
        or any(value is not True for value in exact.values())
        or set(numerical) != _CANONICAL_NUMERICAL_COMPARISON_FIELDS
        or any(value is not True for value in numerical.values())
        or report.get("errors") != []
        or report.get("merged_artifact_sha256") != merged_artifact_sha256
        or set(absolute) != set(numerical)
        or set(relative) != set(numerical)
    ):
        raise SiliconReleaseValidationError(
            "Monolithic comparison is incomplete, failed, mistoleranced, "
            "or not bound to the merged artifact."
        )
    for name in (
        "serial_result_sha256",
        "merged_result_sha256",
        "merged_artifact_sha256",
    ):
        _require_sha256(report.get(name), name=f"comparison.{name}")
    for mapping_name, errors in (
        ("maximum_absolute_error", absolute),
        ("maximum_relative_error", relative),
    ):
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0
            for value in errors.values()
        ):
            raise SiliconReleaseValidationError(
                f"comparison.{mapping_name} has an invalid value."
            )
    return report


def _validate_equality_summary(
    path: Path,
    *,
    expected_mesh: tuple[int, int, int],
    v3_manifest_path: Path,
    v3_orchestration_path: Path,
    v3_run_directory: Path,
    v3_source_identity: Mapping[str, Any],
    tracker: _EvidenceTracker,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    runtime: Mapping[str, Any],
    external: Mapping[str, Any],
) -> dict[str, Any]:
    tracker.snapshot_path(path, name=f"{expected_mesh} equality summary")
    summary = _read_json(path, name=f"{expected_mesh} equality summary")
    certification = _mapping(
        summary.get("scientific_certification"),
        name=f"{expected_mesh}.scientific_certification",
    )
    qualification = _mapping(
        summary.get("execution_qualification"),
        name=f"{expected_mesh}.execution_qualification",
    )
    if (
        summary.get("schema") != SILICON_TARGET_Q_CLI_SCHEMA
        or summary.get("schema_version") != 1
        or summary.get("status") != "passed"
        or summary.get("provenance_eligible") is not True
        or summary.get("external_validation_eligible") is not True
        or summary.get("memory_override_used") is not False
        or summary.get("complete_target_set") is not True
        or summary.get("complete_band_set") is not True
        or summary.get("complete_scientific_selection") is not True
        or summary.get("no_internal_q1_mpi") is not True
        or summary.get("release_certified") is not False
        or certification.get("certified") is not False
        or certification.get("status") != "pending_external_comparison"
        or qualification.get("resumed_task_count") != 0
        or qualification.get("resumed_task_ids") != []
        or qualification.get("orphan_artifact_adopted_task_count") != 0
        or qualification.get("orphan_artifact_adopted_task_ids") != []
        or qualification.get("fresh_complete_task_execution") is not True
        or qualification.get("fresh_complete_mpi_execution") is not True
        or qualification.get("mpi_equality_eligible") is not True
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} equality campaign is not fresh/full/certifiable."
        )
    evidence = _mapping(
        summary.get("evidence"), name=f"{expected_mesh}.evidence"
    )
    qualification_record = tracker.file_record(
        evidence.get("qualification_report"),
        name=f"{expected_mesh}.qualification_report",
    )
    qualification_report, qualification_inputs = (
        _validate_serial_qualification(
            Path(str(qualification_record["path"])),
            tracker=tracker,
            repository=repository,
            imported_code=imported_code,
            runtime=runtime,
        )
    )
    target_records_raw = _mapping(
        evidence.get("target_generation_records"),
        name=f"{expected_mesh}.target_generation_records",
    )
    target_records = {
        str(role): tracker.file_record(
            record,
            name=f"{expected_mesh}.target_generation_records.{role}",
        )
        for role, record in target_records_raw.items()
    }
    _require_qualification_target_match(
        qualification_inputs,
        target_records,
        name=str(expected_mesh),
    )
    for external_role, equality_role in _SHARED_TARGET_ROLE_MAP.items():
        if (
            target_records.get(equality_role, {}).get("sha256")
            != external["target_inputs"][external_role]["sha256"]
        ):
            raise SiliconReleaseValidationError(
                f"{expected_mesh} {equality_role} differs from 11x11x11."
            )

    task_record = tracker.file_record(
        evidence.get("task_manifest"),
        name=f"{expected_mesh}.task_manifest",
        extra_fields=("campaign_manifest_sha256",),
    )
    execution_record = tracker.file_record(
        evidence.get("execution_manifest"),
        name=f"{expected_mesh}.execution_manifest",
        extra_fields=("execution_sha256",),
    )
    merge_record = tracker.file_record(
        evidence.get("merge_report"),
        name=f"{expected_mesh}.merge_report",
        extra_fields=("report_sha256",),
    )
    merged_record = tracker.file_record(
        evidence.get("merged_hdf5"),
        name=f"{expected_mesh}.merged_hdf5",
    )
    comparison_record = tracker.file_record(
        evidence.get("monolithic_comparison_report"),
        name=f"{expected_mesh}.monolithic_comparison_report",
        extra_fields=("report_sha256",),
    )
    serial_archive_record = tracker.file_record(
        evidence.get("serial_result_archive"),
        name=f"{expected_mesh}.serial_result_archive",
        extra_fields=("result_sha256", "dataset_sha256", "provenance"),
    )
    reference_record = tracker.file_record(
        evidence.get("target_reference"),
        name=f"{expected_mesh}.target_reference",
    )

    controls_raw = _mapping(
        summary.get("controls"), name=f"{expected_mesh}.controls"
    )
    controls = BulkTargetQControls(
        temperatures=controls_raw["temperatures"],
        grid_points=controls_raw["grid_points"],
        band_indices=controls_raw["band_indices"],
        n_qpoints=int(controls_raw["n_qpoints"]),
        n_bands=int(controls_raw["n_bands"]),
        problem_sha256=str(controls_raw["problem_sha256"]),
        input_hashes=_mapping(
            controls_raw.get("input_hashes"),
            name=f"{expected_mesh}.controls.input_hashes",
        ),
    )
    if controls.as_dict() != dict(controls_raw):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} controls are not canonical."
        )
    task_document = _read_json(
        Path(str(task_record["path"])),
        name=f"{expected_mesh} task manifest",
    )
    task_payload = _verify_envelope(
        task_document, name=f"{expected_mesh} task manifest"
    )
    user_metadata = _mapping(
        _mapping(
            task_payload.get("metadata"),
            name=f"{expected_mesh}.task metadata",
        ).get("user_metadata"),
        name=f"{expected_mesh}.task metadata.user_metadata",
    )
    campaign = build_bulk_target_q_campaign(
        controls,
        source_revision=str(task_payload["source_revision"]),
        source_tree_sha256=str(task_payload["source_tree_sha256"]),
        environment_sha256=str(task_payload["environment_sha256"]),
        metadata=user_metadata,
    )
    if (
        campaign.manifest.as_dict() != task_document
        or campaign.as_dict()
        != dict(_mapping(summary.get("campaign"), name="campaign"))
        or campaign.campaign_manifest_sha256
        != summary.get("campaign_manifest_sha256")
        or task_record.get("campaign_manifest_sha256")
        != campaign.campaign_manifest_sha256
        or controls.controls_sha256 != summary.get("controls_sha256")
        or controls.problem_sha256 != summary.get("problem_sha256")
        or task_payload.get("source_revision") != repository.get("head_commit")
        or task_payload.get("source_tree_sha256")
        != imported_code.get("record", {}).get("source_tree_sha256")
        or task_payload.get("environment_sha256")
        != external["target_inputs"]["environment_lock"]["sha256"]
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} campaign/source/control identity is invalid."
        )
    expected_input_hashes = {
        "serial_report": qualification_record["sha256"],
        "repository_tree_identity": hashlib.sha256(
            str(repository["tree_hash"]).encode("ascii")
        ).hexdigest(),
        **{
            f"qualification_{role}": record["sha256"]
            for role, record in qualification_inputs.items()
        },
        **{role: record["sha256"] for role, record in target_records.items()},
    }
    if dict(controls.input_hashes) != dict(
        sorted(expected_input_hashes.items())
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} controls do not bind every exact input."
        )

    execution_document = _read_json(
        Path(str(execution_record["path"])),
        name=f"{expected_mesh} execution manifest",
    )
    execution = _verify_envelope(
        execution_document, name=f"{expected_mesh} execution manifest"
    )
    if (
        execution_document != summary.get("execution")
        or execution_record.get("execution_sha256")
        != execution_document["payload_sha256"]
        or execution.get("manifest_sha256")
        != campaign.campaign_manifest_sha256
        or execution.get("mode") != "mpi"
        or int(execution.get("communicator_size", 0)) < 2
        or execution.get("complete") is not True
        or execution.get("thread_limit") != 1
        or execution.get("missing_task_ids") != []
        or execution.get("infrastructure_errors") != []
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} execution manifest is invalid."
        )
    assignments = execution.get("assignments")
    if not isinstance(assignments, list):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} assignments must be an array."
        )
    assignment_ids = [item.get("task_id") for item in assignments]
    if (
        assignment_ids != list(campaign.canonical_task_ids)
        or qualification.get("executed_task_ids") != assignment_ids
        or qualification.get("executed_task_count") != len(assignment_ids)
        or len(assignments) != len(campaign.canonical_task_ids)
        or any(
            item.get("status") != "completed"
            or item.get("resumed") is not False
            or item.get("orphan_artifact_adopted") is not False
            or int(item.get("rank", 0)) <= 0
            or item.get("thread_cap_passed") is not True
            for item in assignments
        )
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} does not prove a fresh MPI task execution."
        )
    _track_assignment_files(
        execution,
        run_root=Path(str(task_record["path"])).parent,
        tracker=tracker,
        name=f"{expected_mesh} execution",
    )

    merge_document = _read_json(
        Path(str(merge_record["path"])),
        name=f"{expected_mesh} merge report",
    )
    merge = _verify_report_digest(
        merge_document, name=f"{expected_mesh} merge report"
    )
    if (
        merge_document != summary.get("merge")
        or merge.get("schema") != SILICON_TARGET_Q_MERGE_SCHEMA
        or merge.get("schema_version") != 1
        or merge_record.get("report_sha256") != merge.get("report_sha256")
        or merge.get("campaign_manifest_sha256")
        != campaign.campaign_manifest_sha256
        or merge.get("merged_artifact_sha256") != merged_record["sha256"]
        or Path(str(merge.get("merged_artifact_path"))).resolve()
        != Path(str(merged_record["path"]))
        or list(merge.get("task_ids", ())) != list(campaign.canonical_task_ids)
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} merge evidence is inconsistent."
        )
    artifact_hashes = list(merge.get("task_artifact_sha256", ()))
    checkpoint_hashes = list(merge.get("task_checkpoint_sha256", ()))
    if len(artifact_hashes) != len(assignments) or len(
        checkpoint_hashes
    ) != len(assignments):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} merge task hashes are incomplete."
        )
    for position, assignment in enumerate(assignments):
        if (
            assignment.get("artifact_sha256") != artifact_hashes[position]
            or assignment.get("checkpoint_file_sha256")
            != checkpoint_hashes[position]
        ):
            raise SiliconReleaseValidationError(
                f"{expected_mesh} execution/merge task hashes differ."
            )
    try:
        target_q_release_evidence = verify_bulk_target_q_release_evidence(
            campaign,
            Path(str(task_record["path"])).parent,
            Path(str(execution_record["path"])),
            Path(str(merge_record["path"])),
            Path(str(merged_record["path"])),
        )
    except BulkTargetQEvidenceError as exc:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} task bytes failed independent canonical "
            f"re-merge validation: {exc}"
        ) from exc

    comparison_document = _read_json(
        Path(str(comparison_record["path"])),
        name=f"{expected_mesh} monolithic comparison",
    )
    comparison = _validate_comparison_report(
        comparison_document,
        merged_artifact_sha256=merged_record["sha256"],
    )
    if comparison_document != summary.get(
        "monolithic_comparison"
    ) or comparison_record.get("report_sha256") != comparison.get(
        "report_sha256"
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} comparison file differs from summary evidence."
        )
    try:
        serial_archive = read_bulk_serial_result_archive(
            Path(str(serial_archive_record["path"]))
        )
    except BulkSerialResultArchiveError as exc:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} authoritative serial result archive is invalid."
        ) from exc
    expected_serial_provenance = {
        "schema": "negfpy.bulk_three_phonon_serial_result_provenance.v1",
        "schema_version": 1,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "controls_sha256": controls.controls_sha256,
        "problem_sha256": controls.problem_sha256,
        "source_revision": task_payload["source_revision"],
        "source_tree_sha256": task_payload["source_tree_sha256"],
        "environment_sha256": task_payload["environment_sha256"],
        "input_hashes": dict(controls.input_hashes),
        "temperatures": controls.temperatures.tolist(),
        "grid_points": controls.grid_points.tolist(),
        "band_indices": controls.band_indices.tolist(),
        "n_qpoints": controls.n_qpoints,
        "n_bands": controls.n_bands,
        "comparison_rtol": 1.0e-10,
        "comparison_atol": 1.0e-12,
    }
    if (
        serial_archive.result_sha256
        != serial_archive_record.get("result_sha256")
        or dict(serial_archive.dataset_sha256)
        != dict(
            _mapping(
                serial_archive_record.get("dataset_sha256"),
                name=f"{expected_mesh}.serial_result_archive.dataset_sha256",
            )
        )
        or dict(serial_archive.provenance)
        != dict(
            _mapping(
                serial_archive_record.get("provenance"),
                name=f"{expected_mesh}.serial_result_archive.provenance",
            )
        )
        or dict(serial_archive.provenance) != expected_serial_provenance
        or serial_archive.result_sha256 != comparison["serial_result_sha256"]
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} serial archive identity/provenance differs "
            "from the campaign or comparison."
        )
    serial_archive_result_sha256 = serial_archive.result_sha256
    del serial_archive

    reference = read_phono3py_silicon_reference(
        Path(str(reference_record["path"])),
        expected_sha256=reference_record["sha256"],
        require_checksum_sidecar=False,
    )
    n_qpoints = int(np.prod(expected_mesh))
    if (
        tuple(reference.mesh) != expected_mesh
        or reference.n_qpoints != n_qpoints
        or tuple(int(value) for value in controls.grid_points)
        != tuple(range(n_qpoints))
        or tuple(int(value) for value in controls.band_indices)
        != tuple(range(controls.n_bands))
        or tuple(int(value) for value in reference.grid_points_grg)
        != tuple(range(n_qpoints))
        or tuple(int(value) for value in reference.band_indices)
        != tuple(range(reference.n_bands))
        or tuple(float(value) for value in controls.temperatures)
        != tuple(float(value) for value in reference.temperatures_k)
        or list(reference.temperatures_k) != external["temperatures_k"]
        or float(reference.sigmas_thz[0]) != external["sigma_thz"]
        or float(reference.cutoff_frequency_thz)
        != external["cutoff_frequency_thz"]
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} reference/full controls differ from release "
            "requirements or the 11x11x11 oracle."
        )
    reference_mpi_generation = _validate_equality_reference_mpi(
        expected_mesh=expected_mesh,
        target_records=target_records,
        reference=reference,
    )
    v3_mpi_sampling = _validate_equality_v3_mpi(
        expected_mesh=expected_mesh,
        manifest_path=v3_manifest_path,
        orchestration_path=v3_orchestration_path,
        run_directory=v3_run_directory,
        qualification_inputs=qualification_inputs,
        reference=reference,
        tracker=tracker,
        repository=repository,
        imported_code=imported_code,
        environment_sha256=external["target_inputs"]["environment_lock"][
            "sha256"
        ],
        source_identity=v3_source_identity,
    )
    qualification_controls = _mapping(
        qualification_report.get("controls"),
        name=f"{expected_mesh} qualification controls",
    )
    sample_manifest = _mapping(
        qualification_report.get("interaction_sample_manifest"),
        name=f"{expected_mesh} qualification interaction_sample_manifest",
    )
    asr_atol = float(qualification_controls.get("asr_atol", np.nan))
    asr_rtol = float(qualification_controls.get("asr_rtol", np.nan))
    degeneracy = float(sample_manifest.get("degeneracy_tolerance_thz", np.nan))
    if (
        asr_atol != 1.0e-10
        or asr_rtol != 1.0e-8
        or not np.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} qualification lacks the fixed serial problem controls."
        )
    problem_metadata = {
        "validation_driver": SILICON_ORACLE_REPORT_SCHEMA,
        "reference_sha256": target_records[
            "target_bundle_si-phono3py-reference.hdf5"
        ]["sha256"],
        "fc2_compact_sha256": target_records["target_bundle_fc2.compact.hdf5"][
            "sha256"
        ],
        "fc2_full_sha256": target_records["target_bundle_fc2.full.hdf5"][
            "sha256"
        ],
        "fc3_compact_sha256": target_records["target_bundle_fc3.compact.hdf5"][
            "sha256"
        ],
        "fc3_full_sha256": target_records["target_bundle_fc3.full.hdf5"][
            "sha256"
        ],
        "source_commit": PINNED_SILICON_SOURCE_COMMIT,
        "no_fitted_prefactors": True,
        "internal_q_parallelism": "none",
    }
    try:
        serial_problem = load_silicon_bulk_three_phonon_problem(
            fc2_compact_path=str(
                target_records["target_bundle_fc2.compact.hdf5"]["path"]
            ),
            fc3_compact_path=str(
                target_records["target_bundle_fc3.compact.hdf5"]["path"]
            ),
            phono3py_yaml_path=str(
                target_records["target_phono3py_yaml"]["path"]
            ),
            mesh=expected_mesh,
            sigma_thz=float(reference.sigmas_thz[0]),
            cutoff_frequency_thz=float(reference.cutoff_frequency_thz),
            frequency_factor_to_thz=(PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ),
            degenerate_tolerance_thz=degeneracy,
            make_r0_average=True,
            symprec=PHONO3PY_4_4_SYMPREC,
            translation_atol=1.0e-12,
            translation_rtol=1.0e-10,
            asr_atol=asr_atol,
            asr_rtol=asr_rtol,
            metadata=problem_metadata,
        )
    except (OSError, TypeError, ValueError, SiliconBulkProblemError) as exc:
        raise SiliconReleaseValidationError(
            f"{expected_mesh} independent serial problem reconstruction failed."
        ) from exc
    if bulk_three_phonon_problem_sha256(serial_problem) != (
        controls.problem_sha256
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} independently reconstructed serial problem "
            "does not match the campaign problem hash."
        )
    independent_serial = compute_three_phonon_linewidths(
        serial_problem,
        temperatures=controls.temperatures,
        grid_points=controls.grid_points,
        band_indices=controls.band_indices,
    )
    independent_comparison = compare_bulk_target_q_with_serial(
        independent_serial,
        Path(str(merged_record["path"])),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if (
        independent_comparison.as_dict() != comparison_document
        or independent_comparison.serial_result_sha256
        != serial_archive_result_sha256
    ):
        raise SiliconReleaseValidationError(
            f"{expected_mesh} independently rerun serial kernel differs "
            "from the archived serial operand or MPI comparison."
        )
    return {
        "summary_path": str(path),
        "summary_sha256": _sha256(path),
        "mesh": list(expected_mesh),
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "execution_manifest_sha256": execution_record["sha256"],
        "merge_report_sha256": merge_record["sha256"],
        "merged_hdf5_sha256": merged_record["sha256"],
        "comparison_report_sha256": comparison_record["sha256"],
        "serial_archive_sha256": serial_archive_record["sha256"],
        "serial_result_sha256": comparison["serial_result_sha256"],
        "merged_result_sha256": comparison["merged_result_sha256"],
        "task_count": len(assignments),
        "communicator_size": execution["communicator_size"],
        "fresh_complete_mpi_execution": True,
        "full_field_equality": True,
        "independent_task_remerge": target_q_release_evidence,
        "independent_serial_comparison_repeated": True,
        "independent_serial_kernel_recomputed": True,
        "reference_mpi_generation": reference_mpi_generation,
        "v3_mpi_sampling": v3_mpi_sampling,
        "qualification_report_sha256": qualification_record["sha256"],
        "qualification_source_commit": qualification_report[
            "source_repository"
        ]["head_commit"],
    }


def run_silicon_release_validation(
    config: SiliconReleaseValidationConfig,
) -> dict[str, Any]:
    """Authenticate all mandatory evidence and publish the final decision."""

    if not isinstance(config, SiliconReleaseValidationConfig):
        raise TypeError("config must be SiliconReleaseValidationConfig.")
    started_utc = _utc_now()
    started = time.perf_counter()
    tracker = _EvidenceTracker()
    tracker.snapshot_path(
        config.equality_3x3_summary_path,
        name="equality_3x3_summary_path",
    )
    tracker.snapshot_path(
        config.equality_5x5_summary_path,
        name="equality_5x5_summary_path",
    )

    repository_before = _repository_identity(
        config.repository_path, require_clean=True
    )
    runtime_before = _runtime_identity()
    imported_before = _code_identity(repository_before, runtime_before)
    _require_release_runtime(
        repository_before, runtime_before, imported_before
    )
    v3_source_identity_before = current_v3_sampling_source_identity(
        config.repository_path,
        require_clean=True,
    )

    external = _validate_external_report(
        config,
        tracker=tracker,
        repository=repository_before,
        imported_code=imported_before,
        runtime=runtime_before,
    )
    equality_3x3 = _validate_equality_summary(
        config.equality_3x3_summary_path,
        expected_mesh=(3, 3, 3),
        v3_manifest_path=config.v3_3x3_manifest_path,
        v3_orchestration_path=config.v3_3x3_orchestration_path,
        v3_run_directory=config.v3_3x3_run_directory_path,
        v3_source_identity=v3_source_identity_before,
        tracker=tracker,
        repository=repository_before,
        imported_code=imported_before,
        runtime=runtime_before,
        external=external,
    )
    equality_5x5 = _validate_equality_summary(
        config.equality_5x5_summary_path,
        expected_mesh=(5, 5, 5),
        v3_manifest_path=config.v3_5x5_manifest_path,
        v3_orchestration_path=config.v3_5x5_orchestration_path,
        v3_run_directory=config.v3_5x5_run_directory_path,
        v3_source_identity=v3_source_identity_before,
        tracker=tracker,
        repository=repository_before,
        imported_code=imported_before,
        runtime=runtime_before,
        external=external,
    )

    tracker.recheck()
    _recheck_equality_v3_mpi(
        _mapping(
            equality_3x3.get("v3_mpi_sampling"),
            name="3x3x3 V3 MPI sampling",
        ),
        source_identity=v3_source_identity_before,
    )
    _recheck_equality_v3_mpi(
        _mapping(
            equality_5x5.get("v3_mpi_sampling"),
            name="5x5x5 V3 MPI sampling",
        ),
        source_identity=v3_source_identity_before,
    )
    v3_mpi = external["v3_mpi_sampling"]
    try:
        v3_mpi_after = validate_v3_mpi_orchestration_evidence(
            Path(v3_mpi["scientific_manifest"]["path"]),
            Path(v3_mpi["orchestration"]["path"]),
            Path(v3_mpi["run_directory"]),
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=repository_before["head_commit"],
            expected_source_tree_sha256=imported_before["record"][
                "source_tree_sha256"
            ],
            expected_environment_sha256=v3_mpi["environment_sha256"],
            expected_source_identity=current_v3_sampling_source_identity(
                config.repository_path,
                require_clean=True,
            ),
            expected_mesh=v3_mpi["mesh"],
            expected_target_grid_points=v3_mpi["target_grid_points"],
            expected_target_bands=v3_mpi["target_bands"],
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconReleaseValidationError(
            f"V3 MPI evidence failed its final release recheck: {exc}"
        ) from exc
    if (
        v3_mpi_after.file_sha256 != v3_mpi["orchestration"]["sha256"]
        or v3_mpi_after.payload_sha256 != v3_mpi["payload_sha256"]
        or v3_mpi_after.campaign_sha256 != v3_mpi["campaign_sha256"]
        or v3_mpi_after.source_identity_sha256
        != v3_mpi["source_identity_sha256"]
        or v3_mpi_after.source_authenticated
        is not v3_mpi["source_authenticated"]
        or v3_mpi_after.source_revision != v3_mpi["source_revision"]
        or v3_mpi_after.source_tree_sha256 != v3_mpi["source_tree_sha256"]
        or v3_mpi_after.environment_sha256 != v3_mpi["environment_sha256"]
        or list(v3_mpi_after.mesh) != v3_mpi["mesh"]
        or list(v3_mpi_after.target_grid_points)
        != v3_mpi["target_grid_points"]
        or list(v3_mpi_after.target_bands) != v3_mpi["target_bands"]
        or list(v3_mpi_after.selected_execution_ids)
        != v3_mpi["selected_execution_ids"]
        or list(v3_mpi_after.selected_communicator_sizes)
        != v3_mpi["selected_communicator_sizes"]
        or list(v3_mpi_after.selected_modes) != v3_mpi["selected_modes"]
        or v3_mpi_after.restart_resume_recorded
        is not v3_mpi["restart_resume_recorded"]
        or v3_mpi_after.physical_core_binding_passed
        is not v3_mpi["physical_core_binding_passed"]
        or v3_mpi_after.mpi_origin_lineage_passed
        is not v3_mpi["mpi_origin_lineage_passed"]
    ):
        raise SiliconReleaseValidationError(
            "V3 MPI evidence changed during final aggregation."
        )
    repository_after = _repository_identity(
        config.repository_path, require_clean=True
    )
    runtime_after = _runtime_identity()
    imported_after = _code_identity(repository_after, runtime_after)
    v3_source_identity_after = current_v3_sampling_source_identity(
        config.repository_path,
        require_clean=True,
    )
    if (
        repository_after != repository_before
        or runtime_after != runtime_before
        or imported_after != imported_before
        or v3_source_identity_after != v3_source_identity_before
    ):
        raise SiliconReleaseValidationError(
            "Source, imported code, or runtime changed during aggregation."
        )

    report: dict[str, Any] = {
        "schema": SILICON_RELEASE_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed",
        "validation_passed": True,
        "certification_eligible": True,
        "release_certified": True,
        "release_version": SILICON_RELEASE_VERSION,
        "certification_gates": {
            "clean_exact_0_2_0_source_and_runtime": True,
            "external_11x11x11_reference_oracle": True,
            "external_science_independently_recomputed": True,
            "reference_mpi_tasks_executions_and_restart_authenticated": True,
            "v3_mpi_tasks_executions_and_restart_authenticated": True,
            "reference_mpi_3x3_generation_authenticated": True,
            "reference_mpi_5x5_generation_authenticated": True,
            "v3_mpi_3x3_sampling_authenticated": True,
            "v3_mpi_5x5_sampling_authenticated": True,
            "fresh_3x3_full_field_serial_mpi_equality": True,
            "fresh_5x5_full_field_serial_mpi_equality": True,
            "target_task_bytes_independently_remerged": True,
            "serial_operands_archived_and_comparisons_repeated": True,
            "full_q_and_band_selections": True,
            "shared_ifc_mapping_environment": True,
            "immutable_evidence_rechecked": True,
            "no_memory_or_scientific_override": True,
        },
        "started_at_utc": started_utc,
        "completed_at_utc": _utc_now(),
        "wall_time_seconds": time.perf_counter() - started,
        "source_repository": repository_before,
        "imported_code_provenance": imported_before,
        "runtime": runtime_before,
        "external_11x11x11": external,
        "equality_3x3": equality_3x3,
        "equality_5x5": equality_5x5,
        "evidence_inventory": tracker.summary(),
        "decision_scope": {
            "release": "NEGFPy 0.2.0 silicon bulk three-phonon oracle",
            "material_device_scba": False,
            "monolithic_11x11x11_negfpy_recomputation": False,
        },
    }
    digest = write_json_atomic_exclusive(config.output_path, report)
    report["report_artifact"] = {
        "path": str(config.output_path),
        "sha256": digest,
    }
    return report


__all__ = [
    "SILICON_RELEASE_REPORT_SCHEMA",
    "SILICON_RELEASE_VERSION",
    "SiliconReleaseValidationConfig",
    "SiliconReleaseValidationError",
    "run_silicon_release_validation",
]
