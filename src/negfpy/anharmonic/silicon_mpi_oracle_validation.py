"""Independent certification of a merged silicon target-q MPI campaign.

The target-q campaign deliberately proves only orchestration correctness.  A
``status="passed"`` campaign summary is therefore not a scientific release
decision.  This module authenticates that campaign and compares its immutable
merged HDF5 result directly with the independently generated phono3py
reference.

No monolithic NEGFPy three-phonon calculation is performed here.  The only
numerical calculation is the inexpensive random-q FC2 spectrum.  Sampled
gauge-invariant ``|V3|^2`` blocks, phase space, and linewidths are read from
the merged target-q result and passed to :func:`compare_silicon_reference`.

Two narrowly scoped private helpers from :mod:`silicon_oracle_validation` are
reused temporarily: one verifies that comparator blocks still match the
predeclared V3 sample manifest, and the others centralize the already released
strict provenance/JSON rules.  Reusing those rules avoids a second, subtly
different certification convention.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ._bulk_target_q_mpi import (
    BulkTargetQControls,
    build_bulk_target_q_campaign,
    load_bulk_target_q_result,
)
from ._bulk_target_q_evidence import (
    BulkTargetQEvidenceError,
    verify_bulk_target_q_release_evidence,
)
from .bulk_harmonic import compute_bulk_phonons
from .io.phono3py_crystal import read_phono3py_crystal_mapping
from .io.phono3py_fc2 import read_phono3py_fc2
from .phono3py_reference import (
    compare_silicon_reference,
    read_phono3py_silicon_reference,
)
from ._reference_mpi_provenance import (
    REFERENCE_MPI_EVIDENCE_ARCHIVE,
    REFERENCE_MPI_ORCHESTRATION_VERSION,
    ReferenceMPIProvenanceError,
    validate_reference_mpi_evidence,
)
from ._v3_sample_manifest_mpi import (
    V3SampleMPIArtifactError,
    current_v3_sampling_source_identity,
    validate_v3_mpi_orchestration_evidence,
)
from .silicon_oracle_validation import (
    PINNED_SILICON_ENVIRONMENT_LOCK_SHA256,
    SILICON_ORACLE_REPORT_SCHEMA,
    _code_identity,
    _generation_environment_identity,
    _input_records,
    _jsonable,
    _manifest_identity,
    _repository_identity,
    _runtime_identity,
    _validate_sample_manifest_results,
    _verify_generated_checksums,
    _verify_inputs_unchanged,
    write_json_atomic_exclusive,
)
from .v3_sample_manifest import (
    read_v3_sample_manifest,
    verify_v3_sample_manifest_identity,
)


SILICON_MPI_ORACLE_REPORT_SCHEMA = (
    "negfpy.silicon_bulk_mpi_oracle_validation.v1"
)
SILICON_TARGET_Q_CLI_SCHEMA = "negfpy.silicon_bulk_target_q_cli.v1"
SILICON_TARGET_Q_MERGE_SCHEMA = "negfpy.bulk_target_q_merge_report"
_QUALIFICATION_INPUT_ROLES = (
    "environment_lock",
    "fc2_compact_hdf5",
    "fc2_full_hdf5",
    "fc3_compact_hdf5",
    "fc3_full_hdf5",
    "input_manifest",
    "phono3py_yaml",
)
_TARGET_RECORD_ROLES = {
    "environment_lock": "target_environment_lock",
    "fc2_compact_hdf5": "target_bundle_fc2.compact.hdf5",
    "fc2_full_hdf5": "target_bundle_fc2.full.hdf5",
    "fc3_compact_hdf5": "target_bundle_fc3.compact.hdf5",
    "fc3_full_hdf5": "target_bundle_fc3.full.hdf5",
    "input_manifest": "target_bundle_input-manifest.normalized.json",
    "phono3py_yaml": "target_phono3py_yaml",
}
_EXPECTED_GENERATION_FILES = frozenset(
    {
        "fc2.compact.hdf5",
        "fc2.full.hdf5",
        "fc3.compact.hdf5",
        "fc3.full.hdf5",
        "force-constants-environment.json",
        "force-constants-inventory.json",
        "force-constants-settings.json",
        "force-constants-stage.json",
        "input-manifest.normalized.json",
        "inventory.json",
        "reference-environment.json",
        "reference-settings.json",
        "reference-stage.json",
        "si-phono3py-reference.hdf5",
        "source-verification.json",
    }
)
_SHA256_HEX = frozenset("0123456789abcdef")
_MEMORY_LIMIT_FRACTION = 0.70
_LOAD_COPY_FACTOR = 4
_LOAD_FIXED_ALLOWANCE_BYTES = 1024**3
_MINIMUM_REPORT_DISK_BYTES = 64 * 1024**2


class SiliconMPIOracleValidationError(RuntimeError):
    """Raised when MPI campaign or reference evidence is not certifiable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise SiliconMPIOracleValidationError(
            f"{name} must be a lowercase 64-character SHA256."
        )
    return value


def _regular_file(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise SiliconMPIOracleValidationError(
            f"{name} must not be a symbolic link."
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise SiliconMPIOracleValidationError(
            f"{name} does not identify a readable file."
        ) from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise SiliconMPIOracleValidationError(
            f"{name} must be a regular non-symlink file."
        )
    return resolved


def _directory(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise SiliconMPIOracleValidationError(
            f"{name} must not be a symbolic link."
        )
    try:
        resolved = unresolved.resolve(strict=True)
    except OSError as exc:
        raise SiliconMPIOracleValidationError(
            f"{name} does not identify a readable directory."
        ) from exc
    if not resolved.is_dir() or resolved.is_symlink():
        raise SiliconMPIOracleValidationError(
            f"{name} must be a non-symlink directory."
        )
    return resolved


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SiliconMPIOracleValidationError(
            f"{name} must be strict UTF-8 JSON."
        ) from exc
    if not isinstance(value, dict):
        raise SiliconMPIOracleValidationError(
            f"{name} must contain a JSON object."
        )
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SiliconMPIOracleValidationError(f"{name} must be an object.")
    return value


def _verify_file_record(
    value: Any,
    *,
    name: str,
    expected_path: Path | None = None,
    extra_fields: Sequence[str] = (),
) -> dict[str, Any]:
    record = _mapping(value, name=name)
    required = {
        "path",
        "sha256",
        "size_bytes",
        "device",
        "inode",
        "mtime_ns",
    }
    expected_fields = required | set(extra_fields)
    if set(record) != expected_fields:
        raise SiliconMPIOracleValidationError(
            f"{name} file identity has an invalid schema."
        )
    path = _regular_file(str(record["path"]), name=f"{name}.path")
    if expected_path is not None and path != expected_path:
        raise SiliconMPIOracleValidationError(
            f"{name} does not identify the explicitly supplied file."
        )
    expected_sha256 = _require_sha256(
        record["sha256"], name=f"{name}.sha256"
    )
    stat = path.stat()
    actual = {
        "path": str(path),
        "sha256": _sha256(path),
        "size_bytes": stat.st_size,
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "mtime_ns": stat.st_mtime_ns,
    }
    recorded_identity = {
        key: record[key] for key in required
    }
    if recorded_identity != actual or actual["sha256"] != expected_sha256:
        raise SiliconMPIOracleValidationError(
            f"{name} does not match the current immutable bytes/stat identity."
        )
    for field in extra_fields:
        actual[field] = record[field]
    return actual


def _verify_envelope(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    envelope = _mapping(value, name=name)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise SiliconMPIOracleValidationError(
            f"{name} envelope schema is invalid."
        )
    payload = _mapping(envelope["payload"], name=f"{name}.payload")
    expected = _require_sha256(
        envelope["payload_sha256"],
        name=f"{name}.payload_sha256",
    )
    if _canonical_sha256(payload) != expected:
        raise SiliconMPIOracleValidationError(
            f"{name} payload SHA256 is invalid."
        )
    return payload


def _verify_report_digest(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    report = _mapping(value, name=name)
    digest = _require_sha256(
        report.get("report_sha256"),
        name=f"{name}.report_sha256",
    )
    payload = {
        str(key): item
        for key, item in report.items()
        if key != "report_sha256"
    }
    if _canonical_sha256(payload) != digest:
        raise SiliconMPIOracleValidationError(
            f"{name} report SHA256 is invalid."
        )
    return report


@dataclass(frozen=True, slots=True)
class SiliconMPIOracleValidationConfig:
    """Exact immutable inputs for one external target-q certification."""

    qualification_report_path: Path
    campaign_summary_path: Path
    merge_report_path: Path
    merged_result_path: Path
    reference_path: Path
    fc2_compact_path: Path
    fc2_full_path: Path
    fc3_compact_path: Path
    fc3_full_path: Path
    phono3py_yaml_path: Path
    input_manifest_path: Path
    checksums_path: Path
    environment_lock_path: Path
    interaction_samples_path: Path
    interaction_samples_orchestration_path: Path
    interaction_samples_run_directory_path: Path
    repository_path: Path
    output_path: Path
    require_clean_source: bool = True

    def __post_init__(self) -> None:
        for name in (
            "qualification_report_path",
            "campaign_summary_path",
            "merge_report_path",
            "merged_result_path",
            "reference_path",
            "fc2_compact_path",
            "fc2_full_path",
            "fc3_compact_path",
            "fc3_full_path",
            "phono3py_yaml_path",
            "input_manifest_path",
            "checksums_path",
            "environment_lock_path",
            "interaction_samples_path",
            "interaction_samples_orchestration_path",
        ):
            object.__setattr__(
                self,
                name,
                _regular_file(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "interaction_samples_run_directory_path",
            _directory(
                self.interaction_samples_run_directory_path,
                name="interaction_samples_run_directory_path",
            ),
        )
        object.__setattr__(
            self,
            "repository_path",
            _directory(self.repository_path, name="repository_path"),
        )
        output = Path(self.output_path).expanduser().resolve()
        if output.exists():
            raise FileExistsError(
                "output_path already exists; validation reports are immutable."
            )
        object.__setattr__(self, "output_path", output)
        if not isinstance(self.require_clean_source, (bool, np.bool_)):
            raise TypeError("require_clean_source must be boolean.")
        if len(
            {
                self.fc2_compact_path,
                self.fc2_full_path,
                self.fc3_compact_path,
                self.fc3_full_path,
            }
        ) != 4:
            raise ValueError(
                "Compact/full FC2/FC3 paths must be four distinct files."
            )


def _passed_qualification_report(
    config: SiliconMPIOracleValidationConfig,
    *,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    runtime: Mapping[str, Any],
    target_inputs: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    report = _read_json(
        config.qualification_report_path,
        name="qualification_report_path",
    )
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
        raise SiliconMPIOracleValidationError(
            "qualification_report_path is not a fully passed strict serial "
            "silicon qualification."
        )
    report_repository = _mapping(
        report.get("source_repository"),
        name="qualification source_repository",
    )
    report_code = _mapping(
        report.get("imported_code_provenance"),
        name="qualification imported_code_provenance",
    )
    report_runtime = _mapping(
        report.get("runtime"),
        name="qualification runtime",
    )
    if (
        report_repository.get("clean") is not True
        or report_repository.get("head_commit")
        != repository.get("head_commit")
        or report_repository.get("tree_hash") != repository.get("tree_hash")
        or report_code.get("passed") is not True
        or report_code.get("record") != imported_code.get("record")
        or report_runtime.get("packages") != runtime.get("packages")
        or report_runtime.get("thread_environment")
        != runtime.get("thread_environment")
    ):
        raise SiliconMPIOracleValidationError(
            "The serial qualification does not identify this exact clean "
            "source tree and pinned runtime."
        )

    raw_inputs = _mapping(
        report.get("inputs"), name="qualification inputs"
    )
    verified: dict[str, Any] = {}
    for role, value in raw_inputs.items():
        verified[str(role)] = _verify_file_record(
            value,
            name=f"qualification inputs.{role}",
        )
    if not set(_QUALIFICATION_INPUT_ROLES).issubset(verified):
        raise SiliconMPIOracleValidationError(
            "The serial qualification lacks an exact IFC/YAML/environment "
            "input identity."
        )
    for role in _QUALIFICATION_INPUT_ROLES:
        if (
            verified[role]["sha256"]
            != target_inputs[role]["sha256"]
        ):
            raise SiliconMPIOracleValidationError(
                f"Target input {role!r} differs from the exact serial "
                "qualification."
            )
    return report, verified


def _generation_checksum_names(path: Path) -> frozenset[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if len(line) < 67 or line[64:66] != "  ":
            raise SiliconMPIOracleValidationError(
                "Target generation SHA256SUMS contains a malformed line."
            )
        _require_sha256(line[:64], name="target generation checksum")
        candidate = Path(line[66:])
        if (
            candidate.is_absolute()
            or len(candidate.parts) != 1
            or candidate.as_posix() != line[66:]
            or candidate.name in names
        ):
            raise SiliconMPIOracleValidationError(
                "Target generation SHA256SUMS contains an unsafe or "
                "duplicate path."
            )
        names.add(candidate.name)
    if not _EXPECTED_GENERATION_FILES.issubset(names):
        raise SiliconMPIOracleValidationError(
            "Target generation SHA256SUMS omits required reference evidence."
        )
    return frozenset(names)


def _validate_reference_mpi_generation_provenance(
    config: SiliconMPIOracleValidationConfig,
    *,
    repository: Mapping[str, Any],
    checksum_names: frozenset[str],
    reference: Any,
) -> Mapping[str, Any] | None:
    """Authenticate an optional phono3py task-MPI generation record."""

    filename = "reference-mpi-orchestration.json"
    root = config.checksums_path.parent
    stage_path = root / "reference-stage.json"
    path_present = (root / filename).exists()
    checksummed = filename in checksum_names
    if not path_present and not checksummed and not stage_path.exists():
        # Synthetic callers may replace the normal generation preflight.
        # A real authenticated bundle always includes reference-stage.json.
        return None
    stage = _read_json(stage_path, name="reference-stage.json")
    runtime = _mapping(stage.get("runtime"), name="reference-stage.runtime")
    runtime_claimed = "mpi_orchestration" in runtime
    if not any((path_present, runtime_claimed, checksummed)):
        return None
    if not all((path_present, runtime_claimed, checksummed)):
        raise SiliconMPIOracleValidationError(
            "MPI reference orchestration is present or claimed but is not a "
            "complete checksummed generation record."
        )
    document = _read_json(root / filename, name=filename)
    generators = _mapping(
        document.get("generators"), name=f"{filename}.generators"
    )
    serial_code = _mapping(
        generators.get("serial_scientific_generator"),
        name=f"{filename}.serial_scientific_generator",
    )
    mpi_code = _mapping(
        generators.get("mpi_task_and_merge_driver"),
        name=f"{filename}.mpi_task_and_merge_driver",
    )
    serial_script = _regular_file(
        config.repository_path
        / "scripts"
        / "generate_phono3py_si_reference.py",
        name="current serial reference generator",
    )
    mpi_script = _regular_file(
        config.repository_path
        / "scripts"
        / "generate_phono3py_si_reference_mpi.py",
        name="current MPI reference generator",
    )
    scheduler = _mapping(
        document.get("scheduler"), name=f"{filename}.scheduler"
    )
    merged = _mapping(
        document.get("merged_reference"),
        name=f"{filename}.merged_reference",
    )
    tasks = document.get("tasks")
    archive = _mapping(
        document.get("evidence_archive"),
        name=f"{filename}.evidence_archive",
    )
    campaign_sha256 = document.get("campaign_sha256")
    runtime_mpi = _mapping(
        runtime.get("mpi_orchestration"),
        name="reference-stage.runtime.mpi_orchestration",
    )
    if (
        document.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_orchestration"
        or document.get("schema_version")
        != REFERENCE_MPI_ORCHESTRATION_VERSION
        or not isinstance(campaign_sha256, str)
        or len(campaign_sha256) != 64
        or any(character not in _SHA256_HEX for character in campaign_sha256)
        or serial_code.get("script_sha256") != _sha256(serial_script)
        or serial_code.get("git_commit") != repository.get("head_commit")
        or serial_code.get("tracked_worktree_status") != ""
        or serial_code.get("untracked_files_omitted") is not False
        or mpi_code.get("script_sha256") != _sha256(mpi_script)
        or mpi_code.get("git_commit") != repository.get("head_commit")
        or mpi_code.get("tracked_worktree_status") != ""
        or mpi_code.get("untracked_files_omitted") is not False
        or scheduler.get("parallel_boundary") != "complete_target_q"
        or scheduler.get("floating_point_mpi_reductions") is not False
        or scheduler.get("canonical_merge_order") != "strictly_increasing_GRG"
        or merged.get("sha256") != _sha256(config.reference_path)
        or archive.get("name") != REFERENCE_MPI_EVIDENCE_ARCHIVE
        or REFERENCE_MPI_EVIDENCE_ARCHIVE not in checksum_names
        or archive.get("sha256")
        != _sha256(root / REFERENCE_MPI_EVIDENCE_ARCHIVE)
        or not isinstance(tasks, list)
        or len(tasks) != int(reference.grid_points_grg.size)
        or [task.get("grid_point_grg") for task in tasks]
        != reference.grid_points_grg.tolist()
        or runtime_mpi.get("campaign_sha256") != campaign_sha256
        or runtime_mpi.get("orchestration_record_sha256")
        != _sha256(root / filename)
        or runtime_mpi.get("task_count") != len(tasks)
        or runtime_mpi.get("floating_point_mpi_reductions") is not False
        or runtime_mpi.get("full_q_selection") is not True
        or runtime_mpi.get("full_band_selection") is not True
        or runtime_mpi.get("evidence_archive_sha256")
        != archive.get("sha256")
    ):
        raise SiliconMPIOracleValidationError(
            "MPI-generated phono3py reference does not bind the exact clean "
            "serial generator, task/merge wrapper, merged reference, and "
            "canonical no-reduction task graph."
        )
    try:
        evidence = validate_reference_mpi_evidence(
            root,
            document,
            expected_mesh=reference.mesh,
            expected_grid_points=reference.grid_points_grg.tolist(),
            expected_band_indices=reference.band_indices.tolist(),
            expected_n_qpoints=reference.n_qpoints,
            expected_n_bands=reference.n_bands,
            require_mpi=True,
        )
    except ReferenceMPIProvenanceError as exc:
        raise SiliconMPIOracleValidationError(
            f"MPI-generated phono3py evidence is invalid: {exc}"
        ) from exc
    return {**document, "validated_evidence": evidence}


def _validate_generation_runtime(
    generation: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Require both generator stages to use the qualification environment."""

    expected_packages = _mapping(
        runtime.get("packages"), name="current runtime packages"
    )
    expected_threads = _mapping(
        runtime.get("thread_environment"),
        name="current thread environment",
    )
    records = _mapping(
        generation.get("records"), name="generation records"
    )
    checked: dict[str, Any] = {}
    for stage, raw_record in records.items():
        record = _mapping(raw_record, name=f"generation {stage}")
        document = _mapping(
            record.get("record"), name=f"generation {stage}.record"
        )
        identity = _mapping(
            document.get("identity"),
            name=f"generation {stage}.record.identity",
        )
        installed_raw = identity.get("installed_packages")
        if not isinstance(installed_raw, list):
            raise SiliconMPIOracleValidationError(
                f"Generation stage {stage!r} has no installed-package list."
            )
        installed: dict[str, str] = {}
        for raw_package in installed_raw:
            package = _mapping(
                raw_package,
                name=f"generation {stage}.installed_packages[]",
            )
            name = package.get("name")
            version = package.get("version")
            if (
                not isinstance(name, str)
                or not isinstance(version, str)
                or name.lower() in installed
            ):
                raise SiliconMPIOracleValidationError(
                    f"Generation stage {stage!r} has a malformed installed "
                    "package inventory."
                )
            installed[name.lower()] = version
        mismatches = {
            str(name): {
                "expected": str(version),
                "generated": installed.get(str(name).lower()),
            }
            for name, version in expected_packages.items()
            if installed.get(str(name).lower()) != version
        }
        threads = _mapping(
            identity.get("thread_environment"),
            name=f"generation {stage}.thread_environment",
        )
        if mismatches or dict(threads) != dict(expected_threads):
            raise SiliconMPIOracleValidationError(
                f"Generation stage {stage!r} differs from the exact "
                "qualification runtime/thread environment."
            )
        checked[str(stage)] = {
            "installed_package_count": len(installed),
            "required_package_versions": dict(expected_packages),
            "thread_environment": dict(threads),
        }
    return checked


def _target_input_records(
    config: SiliconMPIOracleValidationConfig,
) -> dict[str, dict[str, Any]]:
    paths = {
        "reference_hdf5": config.reference_path,
        "fc2_compact_hdf5": config.fc2_compact_path,
        "fc2_full_hdf5": config.fc2_full_path,
        "fc3_compact_hdf5": config.fc3_compact_path,
        "fc3_full_hdf5": config.fc3_full_path,
        "phono3py_yaml": config.phono3py_yaml_path,
        "input_manifest": config.input_manifest_path,
        "sha256sums": config.checksums_path,
        "environment_lock": config.environment_lock_path,
        "interaction_samples": config.interaction_samples_path,
        "interaction_samples_orchestration": (
            config.interaction_samples_orchestration_path
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        stat = path.stat()
        result[role] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
    return result


def _critical_validation_input_records(
    config: SiliconMPIOracleValidationConfig,
) -> dict[str, dict[str, Any]]:
    """Capture immutable orchestration inputs omitted from IFC input records."""

    paths = {
        "campaign_summary": config.campaign_summary_path,
        "merge_report": config.merge_report_path,
        "merged_result": config.merged_result_path,
        "qualification_report": config.qualification_report_path,
    }
    records: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        stat = path.stat()
        records[role] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
    return records


def _require_identity_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
    *,
    context: str,
) -> None:
    if set(before) != set(after):
        raise SiliconMPIOracleValidationError(
            f"{context} identity role set changed during validation."
        )
    for role in before:
        if dict(before[role]) != dict(after[role]):
            raise SiliconMPIOracleValidationError(
                f"{context} {role!r} changed during validation."
            )


def _linux_mem_available_bytes(
    path: Path = Path("/proc/meminfo"),
) -> int:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SiliconMPIOracleValidationError(
            "Cannot read Linux MemAvailable for the mandatory load preflight."
        ) from exc
    for line in lines:
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                value = int(fields[1]) * 1024
                if value > 0:
                    return value
    raise SiliconMPIOracleValidationError(
        "Linux /proc/meminfo has no valid MemAvailable record."
    )


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists():
        if candidate.parent == candidate:
            raise SiliconMPIOracleValidationError(
                "Cannot identify an existing output-filesystem ancestor."
            )
        candidate = candidate.parent
    return candidate


def _merged_load_preflight(
    handle: Any,
    *,
    merged_path: Path,
    output_path: Path,
    mem_available_bytes: int | None = None,
) -> dict[str, Any]:
    if not os.access(merged_path, os.R_OK):
        raise SiliconMPIOracleValidationError(
            "Merged HDF5 is not readable by the validation process."
        )
    logical_dataset_bytes = 0

    def visitor(_name: str, value: Any) -> None:
        nonlocal logical_dataset_bytes
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            elements = int(np.prod(value.shape, dtype=np.int64))
            logical_dataset_bytes += elements * np.dtype(value.dtype).itemsize

    handle.visititems(visitor)
    if logical_dataset_bytes <= 0:
        raise SiliconMPIOracleValidationError(
            "Merged HDF5 contains no numerical dataset payload."
        )
    predicted_peak = (
        _LOAD_COPY_FACTOR * logical_dataset_bytes
        + _LOAD_FIXED_ALLOWANCE_BYTES
    )
    available = (
        _linux_mem_available_bytes()
        if mem_available_bytes is None
        else int(mem_available_bytes)
    )
    if available <= 0:
        raise SiliconMPIOracleValidationError(
            "MemAvailable must be a positive byte count."
        )
    limit = int(_MEMORY_LIMIT_FRACTION * available)
    if predicted_peak > limit:
        raise SiliconMPIOracleValidationError(
            "Predicted external-validator peak memory exceeds 70% of "
            "Linux MemAvailable; there is no certification override."
        )
    output_filesystem = _existing_ancestor(output_path.parent)
    disk = shutil.disk_usage(output_filesystem)
    if disk.free < _MINIMUM_REPORT_DISK_BYTES:
        raise SiliconMPIOracleValidationError(
            "Output filesystem lacks the minimum free space for immutable "
            "validation reports."
        )
    return {
        "passed": True,
        "logical_hdf5_dataset_bytes": logical_dataset_bytes,
        "copy_factor": _LOAD_COPY_FACTOR,
        "fixed_allowance_bytes": _LOAD_FIXED_ALLOWANCE_BYTES,
        "predicted_peak_bytes": predicted_peak,
        "mem_available_bytes": available,
        "limit_fraction": _MEMORY_LIMIT_FRACTION,
        "limit_bytes": limit,
        "output_filesystem": str(output_filesystem),
        "output_filesystem_free_bytes": disk.free,
        "minimum_output_free_bytes": _MINIMUM_REPORT_DISK_BYTES,
        "override_available": False,
    }


def _authenticate_campaign(
    config: SiliconMPIOracleValidationConfig,
    *,
    summary: Mapping[str, Any],
    qualification_sha256: str,
    qualification_inputs: Mapping[str, Mapping[str, Any]],
    reference: Any,
    repository: Mapping[str, Any],
    imported_code: Mapping[str, Any],
    target_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    certification = _mapping(
        summary.get("scientific_certification"),
        name="campaign scientific_certification",
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
    ):
        raise SiliconMPIOracleValidationError(
            "Campaign summary is not eligible for independent certification."
        )
    if summary.get("monolithic_comparison") is not None:
        raise SiliconMPIOracleValidationError(
            "The production target campaign must not recompute a monolithic "
            "NEGFPy three-phonon result."
        )

    evidence = _mapping(summary.get("evidence"), name="campaign evidence")
    qualification_evidence = _verify_file_record(
        evidence.get("qualification_report"),
        name="campaign evidence.qualification_report",
        expected_path=config.qualification_report_path,
    )
    if qualification_evidence["sha256"] != qualification_sha256:
        raise SiliconMPIOracleValidationError(
            "Campaign qualification-report hash differs from supplied bytes."
        )
    _verify_file_record(
        evidence.get("target_reference"),
        name="campaign evidence.target_reference",
        expected_path=config.reference_path,
    )
    _verify_file_record(
        evidence.get("merged_hdf5"),
        name="campaign evidence.merged_hdf5",
        expected_path=config.merged_result_path,
    )
    merge_evidence = _verify_file_record(
        evidence.get("merge_report"),
        name="campaign evidence.merge_report",
        expected_path=config.merge_report_path,
        extra_fields=("report_sha256",),
    )
    task_manifest_evidence = _verify_file_record(
        evidence.get("task_manifest"),
        name="campaign evidence.task_manifest",
        extra_fields=("campaign_manifest_sha256",),
    )
    execution_evidence = _verify_file_record(
        evidence.get("execution_manifest"),
        name="campaign evidence.execution_manifest",
        extra_fields=("execution_sha256",),
    )

    target_records = _mapping(
        evidence.get("target_generation_records"),
        name="campaign target_generation_records",
    )
    verified_target_records = {
        str(role): _verify_file_record(
            record,
            name=f"target_generation_records.{role}",
        )
        for role, record in target_records.items()
    }
    for local_role, campaign_role in _TARGET_RECORD_ROLES.items():
        expected = verified_target_records.get(campaign_role)
        if (
            expected is None
            or expected["sha256"] != target_inputs[local_role]["sha256"]
            or Path(expected["path"]) != Path(target_inputs[local_role]["path"])
        ):
            raise SiliconMPIOracleValidationError(
                f"Campaign target record {campaign_role!r} differs from the "
                "explicit validation input."
            )
    reference_record = verified_target_records.get(
        "target_bundle_si-phono3py-reference.hdf5"
    )
    if (
        reference_record is None
        or reference_record["sha256"] != reference.sha256
    ):
        raise SiliconMPIOracleValidationError(
            "Campaign target reference identity differs from the "
            "authenticated phono3py reference."
        )

    task_manifest_document = _read_json(
        Path(task_manifest_evidence["path"]),
        name="task manifest",
    )
    task_payload = _verify_envelope(
        task_manifest_document, name="task manifest"
    )
    controls_raw = _mapping(summary.get("controls"), name="campaign controls")
    controls = BulkTargetQControls(
        temperatures=controls_raw["temperatures"],
        grid_points=controls_raw["grid_points"],
        band_indices=controls_raw["band_indices"],
        n_qpoints=int(controls_raw["n_qpoints"]),
        n_bands=int(controls_raw["n_bands"]),
        problem_sha256=str(controls_raw["problem_sha256"]),
        input_hashes=_mapping(
            controls_raw.get("input_hashes"),
            name="campaign controls.input_hashes",
        ),
    )
    if controls.as_dict() != dict(controls_raw):
        raise SiliconMPIOracleValidationError(
            "Campaign controls are not in canonical form."
        )
    if (
        controls.controls_sha256
        != _require_sha256(
            summary.get("controls_sha256"),
            name="campaign controls_sha256",
        )
        or controls.problem_sha256
        != _require_sha256(
            summary.get("problem_sha256"),
            name="campaign problem_sha256",
        )
    ):
        raise SiliconMPIOracleValidationError(
            "Campaign controls/problem hashes are inconsistent."
        )
    user_metadata = _mapping(
        _mapping(
            task_payload.get("metadata"), name="task metadata"
        ).get("user_metadata"),
        name="task metadata.user_metadata",
    )
    rebuilt = build_bulk_target_q_campaign(
        controls,
        source_revision=str(task_payload["source_revision"]),
        source_tree_sha256=str(task_payload["source_tree_sha256"]),
        environment_sha256=str(task_payload["environment_sha256"]),
        metadata=user_metadata,
    )
    if (
        rebuilt.manifest.as_dict() != task_manifest_document
        or rebuilt.as_dict()
        != dict(_mapping(summary.get("campaign"), name="campaign"))
        or rebuilt.campaign_manifest_sha256
        != _require_sha256(
            summary.get("campaign_manifest_sha256"),
            name="campaign_manifest_sha256",
        )
        or task_manifest_evidence.get("campaign_manifest_sha256")
        != rebuilt.campaign_manifest_sha256
    ):
        raise SiliconMPIOracleValidationError(
            "Task manifest, campaign descriptor, and summary are not the "
            "same immutable campaign."
        )
    if (
        task_payload.get("source_revision") != repository.get("head_commit")
        or task_payload.get("source_tree_sha256")
        != imported_code.get("record", {}).get("source_tree_sha256")
        or task_payload.get("environment_sha256")
        != target_inputs["environment_lock"]["sha256"]
    ):
        raise SiliconMPIOracleValidationError(
            "Campaign source/environment identity differs from qualification."
        )

    input_hashes = dict(controls.input_hashes)
    expected_input_hashes = {
        "serial_report": qualification_sha256,
        "repository_tree_identity": hashlib.sha256(
            str(repository["tree_hash"]).encode("ascii")
        ).hexdigest(),
        **{
            f"qualification_{role}": str(record["sha256"])
            for role, record in qualification_inputs.items()
        },
        **{
            role: str(record["sha256"])
            for role, record in verified_target_records.items()
        },
    }
    if input_hashes != dict(sorted(expected_input_hashes.items())):
        raise SiliconMPIOracleValidationError(
            "Campaign controls do not bind exactly the qualification, clean "
            "repository tree, and complete target-generation evidence."
        )

    execution_document = _read_json(
        Path(execution_evidence["path"]),
        name="execution manifest",
    )
    execution_payload = _verify_envelope(
        execution_document, name="execution manifest"
    )
    if execution_document != summary.get("execution"):
        raise SiliconMPIOracleValidationError(
            "Summary execution evidence differs from execution manifest."
        )
    if (
        execution_payload.get("manifest_sha256")
        != rebuilt.campaign_manifest_sha256
        or execution_payload.get("complete") is not True
        or execution_payload.get("mode") != "mpi"
        or int(execution_payload.get("communicator_size", 0)) < 2
        or execution_payload.get("thread_limit") != 1
        or execution_payload.get("missing_task_ids") != []
        or execution_payload.get("infrastructure_errors") != []
        or execution_evidence.get("execution_sha256")
        != execution_document["payload_sha256"]
    ):
        raise SiliconMPIOracleValidationError(
            "Execution manifest does not prove a complete task-level MPI run."
        )

    merge_document = _read_json(
        config.merge_report_path, name="merge report"
    )
    merge = _verify_report_digest(merge_document, name="merge report")
    if (
        merge_document != summary.get("merge")
        or merge_evidence.get("report_sha256")
        != merge.get("report_sha256")
        or merge.get("schema") != SILICON_TARGET_Q_MERGE_SCHEMA
        or merge.get("schema_version") != 1
        or merge.get("campaign_manifest_sha256")
        != rebuilt.campaign_manifest_sha256
        or Path(str(merge.get("merged_artifact_path"))).resolve()
        != config.merged_result_path
        or merge.get("merged_artifact_sha256")
        != target_inputs["merged_result"]["sha256"]
    ):
        raise SiliconMPIOracleValidationError(
            "Merge report, summary, and merged HDF5 identity are inconsistent."
        )

    expected_task_ids = list(rebuilt.canonical_task_ids)
    assignments = execution_payload.get("assignments")
    if not isinstance(assignments, list):
        raise SiliconMPIOracleValidationError(
            "Execution assignments must be an array."
        )
    assignments_by_id = {
        str(item.get("task_id")): _mapping(
            item, name="execution assignment"
        )
        for item in assignments
        if isinstance(item, Mapping)
    }
    if (
        len(assignments_by_id) != len(assignments)
        or list(merge.get("task_ids", ())) != expected_task_ids
        or set(assignments_by_id) != set(expected_task_ids)
    ):
        raise SiliconMPIOracleValidationError(
            "Execution/merge tasks differ from the canonical campaign."
        )
    artifact_hashes = list(merge.get("task_artifact_sha256", ()))
    checkpoint_hashes = list(merge.get("task_checkpoint_sha256", ()))
    for position, task_id in enumerate(expected_task_ids):
        assignment = assignments_by_id[task_id]
        if (
            assignment.get("status") != "completed"
            or assignment.get("thread_cap_passed") is not True
            or assignment.get("artifact_sha256") != artifact_hashes[position]
            or assignment.get("checkpoint_file_sha256")
            != checkpoint_hashes[position]
        ):
            raise SiliconMPIOracleValidationError(
                f"Execution assignment {task_id!r} conflicts with merge "
                "evidence."
            )
    executed_task_ids = [
        task_id
        for task_id in expected_task_ids
        if assignments_by_id[task_id].get("resumed") is False
    ]
    resumed_task_ids = [
        task_id
        for task_id in expected_task_ids
        if assignments_by_id[task_id].get("resumed") is True
    ]
    orphan_artifact_adopted_task_ids = [
        task_id
        for task_id in expected_task_ids
        if assignments_by_id[task_id].get(
            "orphan_artifact_adopted"
        )
        is True
    ]
    fresh_complete_task_execution = bool(
        not resumed_task_ids and not orphan_artifact_adopted_task_ids
    )
    fresh_complete_mpi_execution = bool(
        fresh_complete_task_execution
        and all(
            int(assignments_by_id[task_id].get("rank", 0)) > 0
            for task_id in expected_task_ids
        )
    )
    expected_execution_qualification = {
        "executed_task_count": len(executed_task_ids),
        "resumed_task_count": len(resumed_task_ids),
        "orphan_artifact_adopted_task_count": len(
            orphan_artifact_adopted_task_ids
        ),
        "executed_task_ids": executed_task_ids,
        "resumed_task_ids": resumed_task_ids,
        "orphan_artifact_adopted_task_ids": (
            orphan_artifact_adopted_task_ids
        ),
        "fresh_complete_task_execution": fresh_complete_task_execution,
        "fresh_complete_mpi_execution": fresh_complete_mpi_execution,
        # The external large-mesh campaign intentionally has no monolithic
        # equality calculation.  Scientific reference certification and MPI
        # backend equality are distinct claims.
        "mpi_equality_eligible": False,
    }
    if summary.get("execution_qualification") != (
        expected_execution_qualification
    ):
        raise SiliconMPIOracleValidationError(
            "Campaign execution-qualification fields conflict with task "
            "assignment/resume evidence."
        )
    try:
        target_q_release_evidence = verify_bulk_target_q_release_evidence(
            rebuilt,
            Path(task_manifest_evidence["path"]).parent,
            Path(execution_evidence["path"]),
            config.merge_report_path,
            config.merged_result_path,
        )
    except BulkTargetQEvidenceError as exc:
        raise SiliconMPIOracleValidationError(
            "Target-q task bytes failed independent canonical re-merge "
            f"validation: {exc}"
        ) from exc

    expected_targets = tuple(int(value) for value in reference.grid_points_grg)
    expected_bands = tuple(int(value) for value in reference.band_indices)
    if (
        tuple(int(value) for value in controls.grid_points) != expected_targets
        or tuple(int(value) for value in controls.band_indices)
        != expected_bands
        or tuple(float(value) for value in controls.temperatures)
        != tuple(float(value) for value in reference.temperatures_k)
        or tuple(summary.get("selected_grid_points", ())) != expected_targets
        or tuple(summary.get("selected_band_indices", ())) != expected_bands
        or expected_targets != tuple(range(reference.n_qpoints))
        or expected_bands != tuple(range(reference.n_bands))
    ):
        raise SiliconMPIOracleValidationError(
            "The campaign does not cover the complete authenticated target "
            "q/band/temperature selection."
        )
    return {
        "campaign_manifest_sha256": rebuilt.campaign_manifest_sha256,
        "controls_sha256": controls.controls_sha256,
        "problem_sha256": controls.problem_sha256,
        "temperatures": controls.temperatures.tolist(),
        "grid_points": controls.grid_points.tolist(),
        "band_indices": controls.band_indices.tolist(),
        "n_qpoints": controls.n_qpoints,
        "n_bands": controls.n_bands,
        "task_manifest": task_manifest_evidence,
        "execution_manifest": execution_evidence,
        "merge_report": merge_evidence,
        "merged_hdf5": _verify_file_record(
            evidence["merged_hdf5"],
            name="campaign evidence.merged_hdf5",
            expected_path=config.merged_result_path,
        ),
        "summary_sha256": _sha256(config.campaign_summary_path),
        "communicator_size": execution_payload["communicator_size"],
        "task_count": len(expected_task_ids),
        "task_ids": expected_task_ids,
        "task_artifact_sha256": artifact_hashes,
        "dataset_sha256": dict(
            _mapping(
                merge.get("dataset_sha256"),
                name="merge dataset_sha256",
            )
        ),
        "target_generation_record_count": len(verified_target_records),
        "no_monolithic_recomputation": True,
        "execution_qualification": expected_execution_qualification,
        "independent_task_remerge": target_q_release_evidence,
    }


def _merged_hdf5_identity(
    config: SiliconMPIOracleValidationConfig,
    *,
    campaign: Mapping[str, Any],
    target_inputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as exc:
        raise ImportError(
            "Merged silicon validation requires h5py."
        ) from exc
    with h5py.File(config.merged_result_path, "r") as handle:
        load_preflight = _merged_load_preflight(
            handle,
            merged_path=config.merged_result_path,
            output_path=config.output_path,
        )

        def attribute_text(name: str) -> str:
            value = handle.attrs[name]
            return (
                value.decode("ascii")
                if isinstance(value, bytes)
                else str(value)
            )

        attrs = {
            name: (
                value.decode("ascii")
                if isinstance(value := handle.attrs[name], bytes)
                else value.item()
                if isinstance(value, np.generic)
                else value
            )
            for name in (
                "campaign_manifest_sha256",
                "problem_sha256",
                "controls_sha256",
                "floating_point_mpi_reductions",
            )
        }
        task_ids = json.loads(
            attribute_text("canonical_task_ids_json")
        )
        task_artifact_sha256 = json.loads(
            attribute_text("task_artifact_sha256_json")
        )
        dataset_sha256 = json.loads(
            attribute_text("dataset_sha256_json")
        )
    if (
        attrs["campaign_manifest_sha256"]
        != campaign["campaign_manifest_sha256"]
        or attrs["problem_sha256"] != campaign["problem_sha256"]
        or attrs["controls_sha256"] != campaign["controls_sha256"]
        or bool(attrs["floating_point_mpi_reductions"])
        or task_ids != campaign["task_ids"]
        or task_artifact_sha256 != campaign["task_artifact_sha256"]
        or dataset_sha256 != campaign["dataset_sha256"]
    ):
        raise SiliconMPIOracleValidationError(
            "Merged HDF5 root identity conflicts with campaign evidence."
        )
    result = load_bulk_target_q_result(config.merged_result_path)
    if (
        not np.array_equal(
            result.temperatures,
            np.asarray(campaign["temperatures"], dtype=np.float64),
        )
        or not np.array_equal(
            result.grid_points,
            np.asarray(campaign["grid_points"], dtype=np.int64),
        )
        or not np.array_equal(
            result.band_indices,
            np.asarray(campaign["band_indices"], dtype=np.int64),
        )
        or result.interactions.phonons.qpoints.shape
        != (int(campaign["n_qpoints"]), 3)
        or result.interactions.phonons.frequencies.shape
        != (
            int(campaign["n_qpoints"]),
            int(campaign["n_bands"]),
        )
    ):
        raise SiliconMPIOracleValidationError(
            "Merged HDF5 scientific selectors/dimensions differ from the "
            "authenticated campaign."
        )
    return {
        "sha256": target_inputs["merged_result"]["sha256"],
        "size_bytes": target_inputs["merged_result"]["size_bytes"],
        "campaign_manifest_sha256": attrs["campaign_manifest_sha256"],
        "problem_sha256": attrs["problem_sha256"],
        "controls_sha256": attrs["controls_sha256"],
        "floating_point_mpi_reductions": False,
        "load_preflight": load_preflight,
        "result": result,
    }


def _sample_coverage(
    manifest: Any,
    *,
    reference: Any,
    result: Any,
) -> dict[str, Any]:
    target_q = set(int(value) for value in result.grid_points)
    target_bands = set(int(value) for value in result.band_indices)
    reference_target_q = set(
        int(value) for value in reference.grid_points_grg
    )
    reference_target_bands = set(
        int(value) for value in reference.band_indices
    )
    observed_q: set[int] = set()
    observed_bands: set[int] = set()
    observed_processes: set[str] = set()
    observed_strata: set[str] = set()
    for sample in manifest.samples:
        if (
            sample.q0_grg not in target_q
            or sample.q0_grg not in reference_target_q
            or sample.q1_grg >= reference.n_qpoints
            or sample.expected_q2_grg >= reference.n_qpoints
            or sample.b0 not in target_bands
            or sample.b0 not in reference_target_bands
        ):
            raise SiliconMPIOracleValidationError(
                "V3 sample coverage lies outside the merged/reference "
                "selection."
            )
        observed_q.add(sample.q0_grg)
        observed_bands.add(sample.b0)
        observed_processes.add(sample.process)
        observed_strata.add(sample.stratum)
    if (
        manifest.cover_all_target_grid_points
        and observed_q != reference_target_q
    ):
        raise SiliconMPIOracleValidationError(
            "V3 sample manifest claims but does not cover every target q."
        )
    if (
        manifest.cover_all_target_bands
        and observed_bands != reference_target_bands
    ):
        raise SiliconMPIOracleValidationError(
            "V3 sample manifest claims but does not cover every target band."
        )
    if (
        not set(manifest.required_processes).issubset(observed_processes)
        or not set(manifest.required_strata).issubset(observed_strata)
        or set(manifest.empty_strata).intersection(observed_strata)
    ):
        raise SiliconMPIOracleValidationError(
            "V3 sample manifest process/stratum coverage is inconsistent."
        )
    return {
        "passed": True,
        "sample_count": len(manifest.samples),
        "selection_algorithm": manifest.selection_algorithm,
        "selection_algorithm_version": (
            manifest.selection_algorithm_version
        ),
        "cover_all_target_grid_points": (
            manifest.cover_all_target_grid_points
        ),
        "cover_all_target_bands": manifest.cover_all_target_bands,
        "observed_target_grid_point_count": len(observed_q),
        "observed_target_band_count": len(observed_bands),
        "observed_processes": sorted(observed_processes),
        "observed_strata": sorted(observed_strata),
        "required_processes": list(manifest.required_processes),
        "required_strata": list(manifest.required_strata),
        "empty_strata": list(manifest.empty_strata),
    }


def _evaluate_silicon_mpi_oracle_validation(
    config: SiliconMPIOracleValidationConfig,
) -> dict[str, Any]:
    """Authenticate and scientifically evaluate one merged target-q result.

    This private evaluator intentionally does not publish a report.  Keeping
    evaluation separate from publication lets the final release aggregator
    repeat the complete scientific decision directly from the underlying
    immutable evidence instead of trusting booleans in an earlier JSON file.
    """

    if not isinstance(config, SiliconMPIOracleValidationConfig):
        raise TypeError(
            "config must be SiliconMPIOracleValidationConfig."
        )
    started_utc = _utc_now()
    started = time.perf_counter()
    stage_times: dict[str, float] = {}

    stage = time.perf_counter()
    target_inputs = _target_input_records(config)
    target_inputs["merged_result"] = {
        "path": str(config.merged_result_path),
        "sha256": _sha256(config.merged_result_path),
        "size_bytes": config.merged_result_path.stat().st_size,
        "device": config.merged_result_path.stat().st_dev,
        "inode": config.merged_result_path.stat().st_ino,
        "mtime_ns": config.merged_result_path.stat().st_mtime_ns,
    }
    critical_inputs = _critical_validation_input_records(config)
    if (
        target_inputs["environment_lock"]["sha256"]
        != PINNED_SILICON_ENVIRONMENT_LOCK_SHA256
    ):
        raise SiliconMPIOracleValidationError(
            "environment_lock_path is not the pinned silicon MPI lock."
        )
    # These released helpers are reused to keep generation authentication
    # byte-for-byte consistent with the serial silicon validator.
    legacy_records = _input_records(config)
    _verify_generated_checksums(config, legacy_records)
    checksum_names = _generation_checksum_names(config.checksums_path)
    manifest_identity = _manifest_identity(
        config.input_manifest_path,
        target_inputs["phono3py_yaml"]["sha256"],
    )
    repository = _repository_identity(
        config.repository_path,
        require_clean=bool(config.require_clean_source),
    )
    runtime = _runtime_identity()
    imported_code = _code_identity(repository, runtime)
    generation_environment = _generation_environment_identity(
        config, repository
    )
    if not imported_code["passed"] or not generation_environment["passed"]:
        raise SiliconMPIOracleValidationError(
            "Current code or target reference generation provenance failed."
        )
    generation_runtime = _validate_generation_runtime(
        generation_environment, runtime
    )
    qualification, qualification_inputs = _passed_qualification_report(
        config,
        repository=repository,
        imported_code=imported_code,
        runtime=runtime,
        target_inputs=target_inputs,
    )
    stage_times["provenance_preflight"] = time.perf_counter() - stage

    stage = time.perf_counter()
    reference = read_phono3py_silicon_reference(
        config.reference_path,
        expected_sha256=target_inputs["reference_hdf5"]["sha256"],
        require_checksum_sidecar=False,
    )
    if (
        not reference.make_r0_average
        or reference.fc2_random_qpoints is None
        or reference.n_qpoints > 4096
        or reference.grid_points_grg.size != reference.n_qpoints
    ):
        raise SiliconMPIOracleValidationError(
            "Target reference is not the complete bounded production mesh."
        )
    reference_mpi_generation = (
        _validate_reference_mpi_generation_provenance(
            config,
            repository=repository,
            checksum_names=checksum_names,
            reference=reference,
        )
    )
    summary = _read_json(
        config.campaign_summary_path, name="campaign_summary_path"
    )
    campaign = _authenticate_campaign(
        config,
        summary=summary,
        qualification_sha256=_sha256(config.qualification_report_path),
        qualification_inputs=qualification_inputs,
        reference=reference,
        repository=repository,
        imported_code=imported_code,
        target_inputs=target_inputs,
    )
    merged_identity = _merged_hdf5_identity(
        config,
        campaign=campaign,
        target_inputs=target_inputs,
    )
    result = merged_identity.pop("result")
    stage_times["campaign_and_merged_authentication"] = (
        time.perf_counter() - stage
    )

    stage = time.perf_counter()
    fc2_mapping = read_phono3py_crystal_mapping(
        config.phono3py_yaml_path,
        purpose="fc2",
        is_nac=False,
    )
    fc3_mapping = read_phono3py_crystal_mapping(
        config.phono3py_yaml_path,
        purpose="fc3",
        is_nac=False,
    )
    fc2 = read_phono3py_fc2(
        config.fc2_compact_path,
        fc2_mapping.supercell.masses,
        p2s_map=fc2_mapping.p2s,
        s2p_map=fc2_mapping.s2p,
        compact=True,
    )
    fc2_spectrum = compute_bulk_phonons(
        fc2,
        fc2_mapping,
        reference.fc2_random_qpoints,
        hermitianize=bool(reference.fc2_hermitianized),
    )
    manifest = read_v3_sample_manifest(config.interaction_samples_path)
    current_v3_source_identity = current_v3_sampling_source_identity(
        config.repository_path,
        require_clean=True,
    )
    try:
        v3_mpi_evidence = validate_v3_mpi_orchestration_evidence(
            config.interaction_samples_path,
            config.interaction_samples_orchestration_path,
            config.interaction_samples_run_directory_path,
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=repository["head_commit"],
            expected_source_tree_sha256=imported_code["record"][
                "source_tree_sha256"
            ],
            expected_environment_sha256=target_inputs[
                "environment_lock"
            ]["sha256"],
            expected_source_identity=current_v3_source_identity,
            expected_mesh=reference.mesh,
            expected_target_grid_points=reference.grid_points_grg,
            expected_target_bands=reference.band_indices,
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconMPIOracleValidationError(
            f"V3 MPI orchestration evidence is invalid: {exc}"
        ) from exc
    verify_v3_sample_manifest_identity(
        manifest,
        reference,
        fc2_compact_hdf5_path=config.fc2_compact_path,
        fc2_full_hdf5_path=config.fc2_full_path,
        fc3_compact_hdf5_path=config.fc3_compact_path,
        fc3_full_hdf5_path=config.fc3_full_path,
        phono3py_yaml_path=config.phono3py_yaml_path,
        input_manifest_path=config.input_manifest_path,
        fc2_mapping=fc2_mapping,
        fc3_mapping=fc3_mapping,
    )
    coverage = _sample_coverage(
        manifest, reference=reference, result=result
    )
    stage_times["fc2_and_sample_identity"] = time.perf_counter() - stage

    stage = time.perf_counter()
    comparison = compare_silicon_reference(
        reference,
        fc2_spectrum=fc2_spectrum,
        interactions=result.interactions,
        linewidths=result,
        interaction_samples=manifest.comparator_samples,
    )
    _validate_sample_manifest_results(
        manifest,
        interactions=result.interactions,
        comparison=comparison,
    )
    stage_times["external_scientific_comparison"] = (
        time.perf_counter() - stage
    )

    current_target_inputs = _target_input_records(config)
    _verify_inputs_unchanged(
        legacy_records,
        _input_records(config),
    )
    for role in current_target_inputs:
        if (
            current_target_inputs[role]["sha256"]
            != target_inputs[role]["sha256"]
        ):
            raise SiliconMPIOracleValidationError(
                f"Input {role!r} changed during external validation."
            )
    _require_identity_unchanged(
        critical_inputs,
        _critical_validation_input_records(config),
        context="Critical validation input",
    )
    try:
        v3_mpi_evidence_after = validate_v3_mpi_orchestration_evidence(
            config.interaction_samples_path,
            config.interaction_samples_orchestration_path,
            config.interaction_samples_run_directory_path,
            require_authenticated_source=True,
            require_mpi=True,
            expected_source_revision=repository["head_commit"],
            expected_source_tree_sha256=imported_code["record"][
                "source_tree_sha256"
            ],
            expected_environment_sha256=target_inputs[
                "environment_lock"
            ]["sha256"],
            expected_source_identity=current_v3_source_identity,
            expected_mesh=reference.mesh,
            expected_target_grid_points=reference.grid_points_grg,
            expected_target_bands=reference.band_indices,
        )
    except V3SampleMPIArtifactError as exc:
        raise SiliconMPIOracleValidationError(
            "V3 MPI orchestration evidence failed its final recheck: "
            f"{exc}"
        ) from exc
    if v3_mpi_evidence_after != v3_mpi_evidence:
        raise SiliconMPIOracleValidationError(
            "V3 MPI orchestration evidence changed during validation."
        )
    repository_after = _repository_identity(
        config.repository_path,
        require_clean=bool(config.require_clean_source),
    )
    runtime_after = _runtime_identity()
    imported_code_after = _code_identity(repository_after, runtime_after)
    if (
        repository_after != repository
        or runtime_after != runtime
        or imported_code_after != imported_code
    ):
        raise SiliconMPIOracleValidationError(
            "Repository, runtime, or imported-code identity changed during "
            "external validation."
        )

    scientific_passed = bool(comparison.passed)
    provenance_passed = bool(
        repository["clean"]
        and imported_code["passed"]
        and generation_environment["passed"]
    )
    certification_eligible = bool(
        scientific_passed and provenance_passed and coverage["passed"]
    )
    report: dict[str, Any] = {
        "schema": SILICON_MPI_ORACLE_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed" if scientific_passed else "failed",
        "validation_passed": scientific_passed,
        "certification_eligible": certification_eligible,
        # Fresh 3^3 and 5^3 serial/MPI full-field equality evidence is a
        # separate release-level input.  A later immutable aggregator may set
        # release_certified only after authenticating those reports.
        "release_certified": False,
        "release_aggregation_required": True,
        "certification_gates": {
            "serial_qualification_exact": True,
            "clean_source_and_imported_code": provenance_passed,
            "target_reference_generation_authenticated": True,
            "campaign_execution_and_merge_authenticated": True,
            "complete_target_set": True,
            "no_monolithic_recomputation": True,
            "v3_sample_identity_and_coverage": coverage["passed"],
            "v3_mpi_orchestration_authenticated": True,
            "v3_mpi_physical_core_binding": (
                v3_mpi_evidence.physical_core_binding_passed
            ),
            "v3_mpi_origin_lineage": (
                v3_mpi_evidence.mpi_origin_lineage_passed
            ),
            "fc2_random_q": bool(comparison.fc2.passed),
            "sampled_gauge_invariant_v3": bool(
                comparison.interactions.passed
            ),
            "phase_space": bool(comparison.phase_space.passed),
            "linewidths": bool(comparison.linewidths.passed),
        },
        "release_gates": {
            "external_reference_certification": certification_eligible,
            "fresh_3x3_full_field_serial_mpi_equality": False,
            "fresh_5x5_full_field_serial_mpi_equality": False,
            "release_aggregator_complete": False,
        },
        "started_at_utc": started_utc,
        "completed_at_utc": _utc_now(),
        "wall_time_seconds": time.perf_counter() - started,
        "stage_times_seconds": stage_times,
        "execution": {
            "method": (
                "external_reference_comparison_of_canonical_target_q_merge"
            ),
            "target_q_task_parallelism": "mpi_complete_target_q",
            "exact_checkpoint_resume_permitted": True,
            "internal_q1_parallelism": "none",
            "floating_point_mpi_reductions": False,
            "monolithic_negfpy_recomputation": False,
            "fresh_mpi_equality_claimed": False,
            "release_level_mpi_equality_claimed": False,
            "local_numerical_work": "random_q_fc2_only",
        },
        "source_repository": repository,
        "imported_code_provenance": imported_code,
        "runtime": runtime,
        "post_comparison_immutability": {
            "passed": True,
            "critical_inputs": critical_inputs,
            "v3_mpi_evidence_rechecked": True,
            "repository_identity_rechecked": True,
            "runtime_identity_rechecked": True,
            "imported_code_identity_rechecked": True,
        },
        "qualification": {
            "path": str(config.qualification_report_path),
            "sha256": _sha256(config.qualification_report_path),
            "source_commit": qualification["source_repository"][
                "head_commit"
            ],
            "source_tree": qualification["source_repository"]["tree_hash"],
            "input_sha256": {
                role: record["sha256"]
                for role, record in sorted(qualification_inputs.items())
            },
        },
        "target_reference": {
            "path": str(config.reference_path),
            "sha256": reference.sha256,
            "mesh": list(reference.mesh),
            "target_grid_points": reference.grid_points_grg.tolist(),
            "band_indices": reference.band_indices.tolist(),
            "temperatures_k": reference.temperatures_k.tolist(),
            "sigma_thz": float(reference.sigmas_thz[0]),
            "cutoff_frequency_thz": (
                reference.cutoff_frequency_thz
            ),
            "make_r0_average": reference.make_r0_average,
            "generation_checksum_entry_count": len(checksum_names),
            "normalized_input_manifest": manifest_identity,
            "generation_environment": generation_environment,
            "generation_runtime_match": generation_runtime,
            "task_mpi_generation": (
                None
                if reference_mpi_generation is None
                else {
                    "authenticated": True,
                    "campaign_sha256": reference_mpi_generation[
                        "campaign_sha256"
                    ],
                    "orchestration": {
                        "path": str(
                            config.checksums_path.parent
                            / "reference-mpi-orchestration.json"
                        ),
                        "sha256": _sha256(
                            config.checksums_path.parent
                            / "reference-mpi-orchestration.json"
                        ),
                        "size_bytes": (
                            config.checksums_path.parent
                            / "reference-mpi-orchestration.json"
                        ).stat().st_size,
                    },
                    "orchestration_sha256": _sha256(
                        config.checksums_path.parent
                        / "reference-mpi-orchestration.json"
                    ),
                    "task_count": len(
                        reference_mpi_generation["tasks"]
                    ),
                    "execution_count": reference_mpi_generation[
                        "validated_evidence"
                    ]["execution_count"],
                    "execution_ids": reference_mpi_generation[
                        "validated_evidence"
                    ]["execution_ids"],
                    "restart": reference_mpi_generation[
                        "validated_evidence"
                    ]["restart"],
                    "worker_bindings_by_execution": (
                        reference_mpi_generation["validated_evidence"][
                            "worker_bindings_by_execution"
                        ]
                    ),
                    "bundle_identity": reference_mpi_generation[
                        "validated_evidence"
                    ]["bundle_identity"],
                    "merged_reference": reference_mpi_generation[
                        "validated_evidence"
                    ]["merged_reference"],
                    "canonical_merge": reference_mpi_generation[
                        "validated_evidence"
                    ]["canonical_merge"],
                    "evidence_archive": reference_mpi_generation[
                        "validated_evidence"
                    ]["evidence_archive"],
                    "evidence_members": reference_mpi_generation[
                        "validated_evidence"
                    ]["evidence_members"],
                    "full_q_selection": True,
                    "full_band_selection": True,
                    "floating_point_mpi_reductions": False,
                }
            ),
        },
        "target_inputs": target_inputs,
        "campaign_evidence": campaign,
        "merged_result": merged_identity,
        "v3_sample_manifest": {
            "path": str(config.interaction_samples_path),
            "sha256": manifest.sha256,
            "coverage": coverage,
        },
        "v3_mpi_orchestration": {
            "path": str(v3_mpi_evidence.path),
            "sha256": v3_mpi_evidence.file_sha256,
            "size_bytes": v3_mpi_evidence.path.stat().st_size,
            "payload_sha256": v3_mpi_evidence.payload_sha256,
            "campaign_sha256": v3_mpi_evidence.campaign_sha256,
            "source_identity_sha256": (
                v3_mpi_evidence.source_identity_sha256
            ),
            "source_authenticated": v3_mpi_evidence.source_authenticated,
            "source_revision": v3_mpi_evidence.source_revision,
            "source_tree_sha256": (
                v3_mpi_evidence.source_tree_sha256
            ),
            "environment_sha256": (
                v3_mpi_evidence.environment_sha256
            ),
            "mesh": list(v3_mpi_evidence.mesh),
            "target_grid_points": list(
                v3_mpi_evidence.target_grid_points
            ),
            "target_bands": list(v3_mpi_evidence.target_bands),
            "run_directory": str(
                config.interaction_samples_run_directory_path
            ),
            "selected_execution_ids": list(
                v3_mpi_evidence.selected_execution_ids
            ),
            "selected_communicator_sizes": list(
                v3_mpi_evidence.selected_communicator_sizes
            ),
            "selected_modes": list(v3_mpi_evidence.selected_modes),
            "restart_resume_recorded": (
                v3_mpi_evidence.restart_resume_recorded
            ),
            "physical_core_binding_passed": (
                v3_mpi_evidence.physical_core_binding_passed
            ),
            "mpi_origin_lineage_passed": (
                v3_mpi_evidence.mpi_origin_lineage_passed
            ),
            "floating_point_mpi_reductions": False,
            "canonical_merge_revalidated": True,
        },
        "comparison": _jsonable(comparison),
        "linewidth_semantics": {
            "phono3py_gamma": "cyclic-frequency HWHM",
            "negf_linewidth": "angular-frequency FWHM",
            "conversion_factor": "4*pi",
        },
    }
    return report


def run_silicon_mpi_oracle_validation(
    config: SiliconMPIOracleValidationConfig,
) -> dict[str, Any]:
    """Evaluate and atomically publish one external silicon report."""

    report = _evaluate_silicon_mpi_oracle_validation(config)
    report_sha256 = write_json_atomic_exclusive(config.output_path, report)
    report["report_artifact"] = {
        "path": str(config.output_path),
        "sha256": report_sha256,
    }
    return report


__all__ = [
    "SILICON_MPI_ORACLE_REPORT_SCHEMA",
    "SiliconMPIOracleValidationConfig",
    "SiliconMPIOracleValidationError",
    "run_silicon_mpi_oracle_validation",
]
