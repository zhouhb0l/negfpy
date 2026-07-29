#!/usr/bin/env python3
"""Run provenance-strict whole-target-q silicon campaigns.

One task owns one complete target q point.  The numerical kernel retains the
full internal q-prime and branch sums; MPI is used only to distribute complete
target-q tasks.  Rank zero authenticates the exact problem once, broadcasts a
small immutable descriptor, and performs the canonical filesystem merge.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from negfpy.anharmonic._bulk_target_q_mpi import (
    BulkTargetQControls,
    build_bulk_target_q_campaign,
    bulk_three_phonon_problem_sha256,
    compare_bulk_target_q_with_serial,
    execute_bulk_target_q_campaign,
    merge_bulk_target_q_campaign,
    write_bulk_target_q_comparison_report,
    write_bulk_target_q_merge_report,
)
from negfpy.anharmonic._bulk_result_archive import (
    BulkSerialResultArchive,
    write_bulk_serial_result_archive,
)
from negfpy.anharmonic._mpi_tasks import write_task_manifest
from negfpy.anharmonic._provenance import code_provenance
from negfpy.anharmonic._reference_mpi_provenance import (
    REFERENCE_MPI_EVIDENCE_ARCHIVE,
    REFERENCE_MPI_ORCHESTRATION_VERSION,
    ReferenceMPIProvenanceError,
    validate_reference_mpi_evidence,
)
from negfpy.anharmonic.silicon_oracle_validation import (
    PINNED_SILICON_ENVIRONMENT_LOCK_SHA256,
    PINNED_SILICON_DATASET_ID,
    PINNED_SILICON_REPOSITORY,
    PINNED_SILICON_SOURCE_COMMIT,
    SILICON_ORACLE_REPORT_SCHEMA,
)


CAMPAIGN_CLI_SCHEMA = "negfpy.silicon_bulk_target_q_cli.v1"
MEMORY_PREFLIGHT_SCHEMA = "negfpy.silicon_target_q_memory_preflight.v1"
MEMORY_ESTIMATE_VERSION = 1
MEMORY_LIMIT_FRACTION = 0.70
FIXED_PROCESS_OVERHEAD_BYTES = 256 * 1024**2
TRANSLATIONAL_FC3_ATOL = 1.0e-12
TRANSLATIONAL_FC3_RTOL = 1.0e-10
MONOLITHIC_RTOL = 1.0e-10
MONOLITHIC_ATOL = 1.0e-12
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_EXECUTION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_RUNTIME_PACKAGES = (
    "negfpy",
    "numpy",
    "scipy",
    "h5py",
    "phono3py",
    "phonopy",
    "spglib",
    "mpi4py",
)
_TARGET_BUNDLE_REQUIRED = frozenset(
    {
        "fc2.compact.hdf5",
        "fc2.full.hdf5",
        "fc3.compact.hdf5",
        "fc3.full.hdf5",
        "force-constants-SHA256SUMS",
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


class SiliconTargetQCLIError(RuntimeError):
    """Raised when campaign provenance or orchestration is not certifiable."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("JSON values cannot contain NaN or infinity.")
        return number
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, Mapping):
        return {
            str(key): _json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _json_value(getattr(value, item.name))
            for item in fields(value)
        }
    raise TypeError(f"Unsupported strict-JSON value {type(value).__name__}.")


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            _json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _json_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path_without_symlinks(path: str | Path, *, name: str) -> Path:
    expanded = Path(path).expanduser()
    if ".." in expanded.parts:
        raise SiliconTargetQCLIError(f"{name} contains parent traversal.")
    absolute = Path(os.path.abspath(os.fspath(expanded)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise SiliconTargetQCLIError(
                f"{name} traverses symbolic link {current}."
            )
    return absolute


def _regular_file(path: str | Path, *, name: str) -> Path:
    resolved = _path_without_symlinks(path, name=name)
    if not resolved.is_file():
        raise SiliconTargetQCLIError(f"{name} is not a regular file: {resolved}")
    return resolved


def _directory(path: str | Path, *, name: str) -> Path:
    resolved = _path_without_symlinks(path, name=name)
    if not resolved.is_dir():
        raise SiliconTargetQCLIError(f"{name} is not a directory: {resolved}")
    return resolved


def _read_json(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise SiliconTargetQCLIError(f"{name} is not strict JSON.") from exc
    if not isinstance(value, dict):
        raise SiliconTargetQCLIError(f"{name} root must be an object.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> Path:
    target = _path_without_symlinks(path, name="output path")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = _canonical_bytes(value)
    if target.exists():
        if target.is_file() and target.read_bytes() == payload:
            return target
        raise SiliconTargetQCLIError(
            f"Refusing to replace immutable record {target}."
        )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            if target.read_bytes() != payload:
                raise SiliconTargetQCLIError(
                    f"Concurrent writer published different record {target}."
                )
        _fsync_directory(target.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SiliconTargetQCLIError(
            f"Cannot inspect Git repository {repository}."
        ) from exc
    return result.stdout.strip()


def _runtime_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for package in _RUNTIME_PACKAGES:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "not-installed"
    return result


def _clean_code_identity(
    repository: Path,
    serial_report: Mapping[str, Any],
) -> dict[str, Any]:
    head = _git(repository, "rev-parse", "HEAD")
    tree = _git(repository, "rev-parse", "HEAD^{tree}")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if _REVISION_RE.fullmatch(head) is None:
        raise SiliconTargetQCLIError("Current Git revision is not a full SHA.")
    if status:
        raise SiliconTargetQCLIError(
            "Silicon target-q campaigns require a clean repository; "
            "there is no dirty-source bypass."
        )

    provenance = dict(code_provenance())
    source_hash = provenance.get("source_tree_sha256")
    if (
        provenance.get("git_revision") != head
        or provenance.get("git_dirty") is not False
        or provenance.get("package_version_consistent") is not True
        or not isinstance(source_hash, str)
        or _SHA256_RE.fullmatch(source_hash) is None
    ):
        raise SiliconTargetQCLIError(
            "Imported NEGFPy provenance does not identify the clean repository."
        )

    report_repository = serial_report.get("source_repository")
    report_code = serial_report.get("imported_code_provenance")
    if not isinstance(report_repository, Mapping) or not isinstance(
        report_code, Mapping
    ):
        raise SiliconTargetQCLIError(
            "Serial report lacks source/code provenance."
        )
    report_record = report_code.get("record")
    if not isinstance(report_record, Mapping):
        raise SiliconTargetQCLIError(
            "Serial report lacks imported-code provenance record."
        )
    try:
        report_repository_path = Path(
            str(report_repository["repository"])
        ).resolve(strict=True)
    except (KeyError, OSError) as exc:
        raise SiliconTargetQCLIError(
            "Serial report repository path is unavailable."
        ) from exc
    exact_report_match = (
        report_repository.get("head_commit") == head
        and report_repository.get("tree_hash") == tree
        and report_repository.get("clean") is True
        and report_repository_path == repository.resolve()
        and report_code.get("passed") is True
        and report_record.get("git_revision") == head
        and report_record.get("git_dirty") is False
        and report_record.get("source_tree_sha256") == source_hash
    )
    if not exact_report_match:
        raise SiliconTargetQCLIError(
            "The passed serial silicon report is not from this exact clean "
            "candidate revision, Git tree, and imported package payload. "
            "Historical reports cannot authorize a new candidate."
        )

    runtime = _runtime_versions()
    report_runtime = serial_report.get("runtime")
    if not isinstance(report_runtime, Mapping) or not isinstance(
        report_runtime.get("packages"), Mapping
    ):
        raise SiliconTargetQCLIError("Serial report lacks runtime versions.")
    report_packages = report_runtime["packages"]
    mismatches = {
        package: {
            "serial": report_packages.get(package),
            "current": runtime[package],
        }
        for package in _RUNTIME_PACKAGES
        if report_packages.get(package) != runtime[package]
    }
    if mismatches:
        raise SiliconTargetQCLIError(
            f"Runtime package versions differ from the serial report: {mismatches}"
        )

    invalid_threads = {
        name: os.environ.get(name)
        for name in _THREAD_VARIABLES
        if os.environ.get(name) != "1"
    }
    if invalid_threads:
        raise SiliconTargetQCLIError(
            "Every BLAS/OpenMP thread variable must equal '1': "
            f"{invalid_threads}"
        )
    return {
        "repository": repository.resolve().as_posix(),
        "head_commit": head,
        "tree_hash": tree,
        "tree_identity_sha256": hashlib.sha256(
            tree.encode("ascii")
        ).hexdigest(),
        "source_tree_sha256": source_hash,
        "code_provenance": provenance,
        "runtime_packages": runtime,
        "thread_environment": {
            name: os.environ[name] for name in _THREAD_VARIABLES
        },
    }


@dataclass(frozen=True)
class FileIdentity:
    path: str
    sha256: str
    size_bytes: int
    device: int
    inode: int
    mtime_ns: int

    @classmethod
    def from_path(cls, path: Path) -> "FileIdentity":
        stat = path.stat()
        return cls(
            path=path.as_posix(),
            sha256=_sha256_file(path),
            size_bytes=stat.st_size,
            device=stat.st_dev,
            inode=stat.st_ino,
            mtime_ns=stat.st_mtime_ns,
        )

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise SiliconTargetQCLIError(f"{name} is not a lowercase SHA256.")
    return value


def _load_passed_serial_report(path: Path) -> dict[str, Any]:
    report = _read_json(path, name="serial silicon report")
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
        raise SiliconTargetQCLIError(
            "A passed, certification-eligible strict serial silicon report "
            "with every certification gate true is required."
        )
    return report


def _verify_report_inputs(
    report: Mapping[str, Any],
) -> dict[str, FileIdentity]:
    raw_inputs = report.get("inputs")
    if not isinstance(raw_inputs, Mapping) or not raw_inputs:
        raise SiliconTargetQCLIError("Serial report has no input identities.")
    records: dict[str, FileIdentity] = {}
    for role, raw_record in raw_inputs.items():
        if not isinstance(role, str) or not isinstance(raw_record, Mapping):
            raise SiliconTargetQCLIError(
                "Serial report input identities are malformed."
            )
        expected = _require_sha256(
            raw_record.get("sha256"),
            name=f"serial inputs.{role}.sha256",
        )
        source = _regular_file(
            str(raw_record.get("path", "")),
            name=f"serial inputs.{role}.path",
        )
        identity = FileIdentity.from_path(source)
        if identity.sha256 != expected:
            raise SiliconTargetQCLIError(
                f"Serial report input {role!r} has changed."
            )
        records[f"qualification_{role}"] = identity
    required = {
        "qualification_environment_lock",
        "qualification_fc2_compact_hdf5",
        "qualification_fc3_compact_hdf5",
        "qualification_input_manifest",
        "qualification_phono3py_yaml",
        "qualification_reference_hdf5",
    }
    missing = required - records.keys()
    if missing:
        raise SiliconTargetQCLIError(
            "Serial report lacks required qualified inputs: "
            f"{sorted(missing)}"
        )
    if (
        records["qualification_environment_lock"].sha256
        != PINNED_SILICON_ENVIRONMENT_LOCK_SHA256
    ):
        raise SiliconTargetQCLIError(
            "Environment lock is not the pinned silicon/Open-MPI lock."
        )
    return dict(sorted(records.items()))


def _parse_sha256sums(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SiliconTargetQCLIError(
            "target_checksums must be readable UTF-8 text."
        ) from exc
    if not lines:
        raise SiliconTargetQCLIError("target_checksums cannot be empty.")
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise SiliconTargetQCLIError(
                "target_checksums contains a malformed line."
            )
        digest = line[:64]
        name = line[66:]
        candidate = Path(name)
        if (
            _SHA256_RE.fullmatch(digest) is None
            or not name
            or candidate.is_absolute()
            or candidate.as_posix() != name
            or len(candidate.parts) != 1
            or ".." in candidate.parts
            or "\\" in name
            or name in records
        ):
            raise SiliconTargetQCLIError(
                "target_checksums contains an unsafe or duplicate entry."
            )
        records[name] = digest
    return records


def _require_json_mapping(
    value: Any,
    *,
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SiliconTargetQCLIError(f"{name} must be a JSON object.")
    return value


def _validate_target_environment_record(
    document: Mapping[str, Any],
    *,
    name: str,
    code_identity: Mapping[str, Any],
    serial_report: Mapping[str, Any],
    generator_sha256: str,
) -> None:
    if (
        set(document) != {"schema_version", "recorded_at_utc", "identity"}
        or document.get("schema_version") != 1
    ):
        raise SiliconTargetQCLIError(f"{name} has an invalid schema.")
    identity = _require_json_mapping(
        document.get("identity"),
        name=f"{name}.identity",
    )
    code = _require_json_mapping(
        identity.get("code"),
        name=f"{name}.identity.code",
    )
    versions = _require_json_mapping(
        identity.get("runtime_versions"),
        name=f"{name}.identity.runtime_versions",
    )
    threads = _require_json_mapping(
        identity.get("thread_environment"),
        name=f"{name}.identity.thread_environment",
    )
    if (
        code.get("git_commit") != code_identity["head_commit"]
        or code.get("tracked_worktree_status") != ""
        or code.get("untracked_files_omitted") is not False
        or code.get("script_sha256") != generator_sha256
    ):
        raise SiliconTargetQCLIError(
            f"{name} was not generated by this exact clean candidate."
        )
    report_packages = _require_json_mapping(
        _require_json_mapping(
            serial_report.get("runtime"),
            name="serial runtime",
        ).get("packages"),
        name="serial runtime.packages",
    )
    for package in ("phono3py", "phonopy"):
        if versions.get(package) != report_packages.get(package):
            raise SiliconTargetQCLIError(
                f"{name} {package} version differs from qualification."
            )
    installed = identity.get("installed_packages")
    if not isinstance(installed, list):
        raise SiliconTargetQCLIError(
            f"{name}.identity.installed_packages must be an array."
        )
    installed_versions: dict[str, str] = {}
    for item in installed:
        record = _require_json_mapping(
            item,
            name=f"{name}.identity.installed_packages[]",
        )
        package_name = record.get("name")
        version = record.get("version")
        if not isinstance(package_name, str) or not isinstance(version, str):
            raise SiliconTargetQCLIError(
                f"{name} has a malformed installed-package record."
            )
        normalized = package_name.lower()
        if normalized in installed_versions:
            raise SiliconTargetQCLIError(
                f"{name} contains duplicate installed package {normalized!r}."
            )
        installed_versions[normalized] = version
    for package in _RUNTIME_PACKAGES:
        if installed_versions.get(package) != report_packages.get(package):
            raise SiliconTargetQCLIError(
                f"{name} installed {package} version differs from "
                "qualification."
            )
    if any(threads.get(variable) != "1" for variable in _THREAD_VARIABLES):
        raise SiliconTargetQCLIError(
            f"{name} did not fix every numerical thread count to one."
        )


def _validate_reference_mpi_orchestration(
    *,
    root: Path,
    checksums: Mapping[str, str],
    reference: Any,
    stage: Mapping[str, Any],
    repository: Path,
    code_identity: Mapping[str, Any],
    serial_generator_sha256: str,
) -> Mapping[str, Any] | None:
    """Authenticate the optional task-MPI reference generator evidence.

    Serial reference bundles predate this record and remain valid.  Its
    presence, however, makes both the unchanged scientific generator and the
    task/merge wrapper mandatory exact-candidate provenance.
    """

    filename = "reference-mpi-orchestration.json"
    path_present = (root / filename).exists()
    runtime_value = stage.get("runtime")
    runtime_claimed = bool(
        isinstance(runtime_value, Mapping)
        and "mpi_orchestration" in runtime_value
    )
    checksummed = filename in checksums
    if not any((path_present, runtime_claimed, checksummed)):
        return None
    if not all((path_present, runtime_claimed, checksummed)):
        raise SiliconTargetQCLIError(
            "Target MPI reference orchestration is present or claimed but "
            "is not a complete checksummed generation record."
        )
    document = _require_json_mapping(
        _read_json(root / filename, name=f"target {filename}"),
        name=f"target {filename}",
    )
    generators = _require_json_mapping(
        document.get("generators"),
        name=f"target {filename}.generators",
    )
    serial_code = _require_json_mapping(
        generators.get("serial_scientific_generator"),
        name=f"target {filename}.serial_scientific_generator",
    )
    mpi_code = _require_json_mapping(
        generators.get("mpi_task_and_merge_driver"),
        name=f"target {filename}.mpi_task_and_merge_driver",
    )
    wrapper = _regular_file(
        repository / "scripts" / "generate_phono3py_si_reference_mpi.py",
        name="current phono3py reference MPI generator",
    )
    runtime = _require_json_mapping(
        stage.get("runtime"),
        name="target reference-stage.runtime",
    )
    runtime_mpi = _require_json_mapping(
        runtime.get("mpi_orchestration"),
        name="target reference-stage.runtime.mpi_orchestration",
    )
    campaign_sha256 = document.get("campaign_sha256")
    orchestration_sha256 = checksums[filename]
    merged = _require_json_mapping(
        document.get("merged_reference"),
        name=f"target {filename}.merged_reference",
    )
    scheduler = _require_json_mapping(
        document.get("scheduler"),
        name=f"target {filename}.scheduler",
    )
    tasks = document.get("tasks")
    archive = _require_json_mapping(
        document.get("evidence_archive"),
        name=f"target {filename}.evidence_archive",
    )
    if (
        document.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_orchestration"
        or document.get("schema_version")
        != REFERENCE_MPI_ORCHESTRATION_VERSION
        or not isinstance(campaign_sha256, str)
        or _SHA256_RE.fullmatch(campaign_sha256) is None
        or serial_code.get("script_sha256") != serial_generator_sha256
        or serial_code.get("git_commit") != code_identity["head_commit"]
        or serial_code.get("tracked_worktree_status") != ""
        or serial_code.get("untracked_files_omitted") is not False
        or mpi_code.get("script_sha256") != _sha256_file(wrapper)
        or mpi_code.get("git_commit") != code_identity["head_commit"]
        or mpi_code.get("tracked_worktree_status") != ""
        or mpi_code.get("untracked_files_omitted") is not False
        or merged.get("sha256")
        != checksums["si-phono3py-reference.hdf5"]
        or archive.get("name") != REFERENCE_MPI_EVIDENCE_ARCHIVE
        or checksums.get(REFERENCE_MPI_EVIDENCE_ARCHIVE)
        != archive.get("sha256")
        or scheduler.get("parallel_boundary") != "complete_target_q"
        or scheduler.get("floating_point_mpi_reductions") is not False
        or scheduler.get("canonical_merge_order") != "strictly_increasing_GRG"
        or not isinstance(tasks, list)
        or len(tasks) != int(reference.grid_points_grg.size)
        or runtime_mpi.get("campaign_sha256") != campaign_sha256
        or runtime_mpi.get("orchestration_record_sha256")
        != orchestration_sha256
        or runtime_mpi.get("task_count") != len(tasks)
        or runtime_mpi.get("floating_point_mpi_reductions") is not False
        or runtime_mpi.get("full_q_selection") is not True
        or runtime_mpi.get("full_band_selection") is not True
        or runtime_mpi.get("evidence_archive_sha256")
        != archive.get("sha256")
    ):
        raise SiliconTargetQCLIError(
            "Target MPI reference orchestration does not bind the exact "
            "scientific generator, task/merge wrapper, merged HDF5, and "
            "canonical no-reduction execution contract."
        )
    declared_points = [task.get("grid_point_grg") for task in tasks]
    if declared_points != reference.grid_points_grg.tolist():
        raise SiliconTargetQCLIError(
            "Target MPI reference task evidence is not in canonical GRG order."
        )
    try:
        validate_reference_mpi_evidence(
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
        raise SiliconTargetQCLIError(
            f"Target MPI reference evidence is invalid: {exc}"
        ) from exc
    return document


def _validate_target_inventory(
    inventory: Mapping[str, Any],
    *,
    checksums: Mapping[str, str],
    root: Path,
) -> None:
    if inventory.get("schema_version") != 1:
        raise SiliconTargetQCLIError("Target inventory schema is invalid.")
    artifacts = inventory.get("artifacts")
    if not isinstance(artifacts, list):
        raise SiliconTargetQCLIError(
            "Target inventory artifacts must be an array."
        )
    expected_names = set(checksums) - {"inventory.json"}
    seen: set[str] = set()
    for raw_record in artifacts:
        record = _require_json_mapping(
            raw_record,
            name="target inventory artifact",
        )
        name = record.get("path")
        if (
            not isinstance(name, str)
            or name in seen
            or name not in checksums
        ):
            raise SiliconTargetQCLIError(
                "Target inventory contains an invalid or duplicate path."
            )
        seen.add(name)
        if (
            checksums.get(name) != record.get("sha256")
            or (root / name).stat().st_size != record.get("size_bytes")
        ):
            raise SiliconTargetQCLIError(
                f"Target inventory record {name!r} conflicts with bytes."
            )
    if seen != expected_names:
        raise SiliconTargetQCLIError(
            "Target inventory does not enumerate the complete checksummed "
            "artifact set."
        )


def _validate_target_source_records(
    *,
    normalized_manifest: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    serial_report: Mapping[str, Any],
    yaml_identity: FileIdentity,
) -> str:
    report_source = _require_json_mapping(
        serial_report.get("source_manifest"),
        name="serial source_manifest",
    )
    source = _require_json_mapping(
        normalized_manifest.get("source"),
        name="target normalized manifest source",
    )
    versions = normalized_manifest.get("versions")
    files = _require_json_mapping(
        source.get("files"),
        name="target normalized manifest source.files",
    )
    yaml_record = _require_json_mapping(
        files.get("phono3py_yaml"),
        name="target normalized manifest phono3py_yaml",
    )
    if (
        normalized_manifest.get("schema_version") != 1
        or normalized_manifest.get("dataset_id") != PINNED_SILICON_DATASET_ID
        or source.get("repository") != PINNED_SILICON_REPOSITORY
        or source.get("commit") != PINNED_SILICON_SOURCE_COMMIT
        or versions != {"phono3py": "4.4.0", "phonopy": "4.4.0"}
        or normalized_manifest.get("input_manifest_sha256")
        != report_source.get("input_manifest_sha256")
        or yaml_record.get("sha256") != yaml_identity.sha256
        or yaml_record.get("sha256")
        != report_source.get("phono3py_yaml_sha256")
    ):
        raise SiliconTargetQCLIError(
            "Target normalized manifest does not bind the pinned silicon "
            "dataset and qualified source inputs."
        )
    verified_files = _require_json_mapping(
        source_verification.get("files"),
        name="target source-verification files",
    )
    if (
        source_verification.get("schema_version") != 1
        or source_verification.get("dataset_id") != PINNED_SILICON_DATASET_ID
        or source_verification.get("repository") != PINNED_SILICON_REPOSITORY
        or source_verification.get("commit") != PINNED_SILICON_SOURCE_COMMIT
        or set(verified_files) != set(files)
    ):
        raise SiliconTargetQCLIError(
            "Target source-verification identity is invalid."
        )
    for role, raw_expected in files.items():
        expected = _require_json_mapping(
            raw_expected,
            name=f"target manifest source.files.{role}",
        )
        verified = _require_json_mapping(
            verified_files.get(role),
            name=f"target source-verification.files.{role}",
        )
        if (
            verified.get("path_in_source") != expected.get("path")
            or verified.get("sha256") != expected.get("sha256")
            or not isinstance(verified.get("size_bytes"), int)
            or int(verified["size_bytes"]) <= 0
        ):
            raise SiliconTargetQCLIError(
                f"Target source-verification role {role!r} conflicts with "
                "the normalized manifest."
            )
    return str(source["commit"])


def _validate_target_settings(
    *,
    settings: Mapping[str, Any],
    stage: Mapping[str, Any],
    reference: Any,
) -> None:
    expected_bands = reference.band_indices.tolist()
    declared_bands = settings.get("band_indices")
    if declared_bands is None:
        bands_match = expected_bands == list(range(reference.n_bands))
    else:
        bands_match = declared_bands == expected_bands
    if (
        settings.get("schema_version") != 1
        or settings.get("mesh") != list(reference.mesh)
        or settings.get("grid_points") != reference.grid_points_grg.tolist()
        or not bands_match
        or settings.get("temperatures_K")
        != reference.temperatures_k.tolist()
        or settings.get("sigma_THz") != float(reference.sigmas_thz[0])
        or settings.get("cutoff_frequency_THz")
        != reference.cutoff_frequency_thz
        or settings.get("make_r0_average") is not True
        or settings.get("symmetrize_fc3q") is not False
        or settings.get("is_mesh_symmetry") is not False
        or settings.get("is_nac") is not False
        or settings.get("openmp_per_triplets") is not False
    ):
        raise SiliconTargetQCLIError(
            "Target reference-settings conflict with the authenticated HDF5."
        )
    runtime = _require_json_mapping(
        stage.get("runtime"),
        name="target reference-stage.runtime",
    )
    if (
        stage.get("schema_version") != 1
        or stage.get("stage") != "reference"
        or stage.get("status") != "complete"
        or runtime.get("mesh") != list(reference.mesh)
        or runtime.get("grid_points_grg")
        != reference.grid_points_grg.tolist()
        or runtime.get("effective_band_indices") != expected_bands
        or runtime.get("make_r0_average") is not True
        or runtime.get("symmetrize_fc3q") is not False
        or runtime.get("phono3py_version") != "4.4.0"
    ):
        raise SiliconTargetQCLIError(
            "Target reference-stage conflicts with the authenticated HDF5."
        )


def _verify_target_bundle(
    *,
    reference_path: Path,
    fc2_compact_path: Path,
    fc3_compact_path: Path,
    yaml_path: Path,
    environment_lock_path: Path,
    checksums_path: Path,
    normalized_manifest_path: Path,
    report_records: Mapping[str, FileIdentity],
    serial_report: Mapping[str, Any],
    code_identity: Mapping[str, Any],
    repository: Path,
) -> tuple[Any, dict[str, FileIdentity], dict[str, str], str]:
    root = checksums_path.parent
    checksums = _parse_sha256sums(checksums_path)
    missing = _TARGET_BUNDLE_REQUIRED - checksums.keys()
    if missing:
        raise SiliconTargetQCLIError(
            "Target SHA256SUMS lacks required generated artifacts: "
            f"{sorted(missing)}"
        )
    identities: dict[str, FileIdentity] = {
        "target_sha256sums": FileIdentity.from_path(checksums_path),
        "target_environment_lock": FileIdentity.from_path(
            environment_lock_path
        ),
        "target_phono3py_yaml": FileIdentity.from_path(yaml_path),
    }
    for name, expected in sorted(checksums.items()):
        artifact = _regular_file(
            root / name,
            name=f"target checksummed artifact {name}",
        )
        if artifact.parent != root:
            raise SiliconTargetQCLIError(
                "Target checksummed artifacts must be direct children of "
                "the target bundle directory."
            )
        identity = FileIdentity.from_path(artifact)
        if identity.sha256 != expected:
            raise SiliconTargetQCLIError(
                f"Target checksummed artifact {name!r} has changed."
            )
        identities[f"target_bundle_{name}"] = identity
    explicit = {
        "si-phono3py-reference.hdf5": reference_path,
        "fc2.compact.hdf5": fc2_compact_path,
        "fc3.compact.hdf5": fc3_compact_path,
        "input-manifest.normalized.json": normalized_manifest_path,
    }
    for name, path in explicit.items():
        expected_identity = identities[f"target_bundle_{name}"]
        actual_identity = FileIdentity.from_path(path)
        if actual_identity != expected_identity:
            raise SiliconTargetQCLIError(
                f"Explicit target input {name!r} is not the checksummed "
                "bundle artifact."
            )
    for target_role, qualified_role in (
        ("target_environment_lock", "qualification_environment_lock"),
        ("target_phono3py_yaml", "qualification_phono3py_yaml"),
        (
            "target_bundle_fc2.compact.hdf5",
            "qualification_fc2_compact_hdf5",
        ),
        (
            "target_bundle_fc3.compact.hdf5",
            "qualification_fc3_compact_hdf5",
        ),
    ):
        if (
            identities[target_role].sha256
            != report_records[qualified_role].sha256
        ):
            raise SiliconTargetQCLIError(
                f"Target input {target_role!r} differs from the exact "
                "qualified silicon input."
            )
    if (
        identities["target_environment_lock"].sha256
        != PINNED_SILICON_ENVIRONMENT_LOCK_SHA256
    ):
        raise SiliconTargetQCLIError(
            "Target environment lock is not the pinned silicon/Open-MPI lock."
        )

    from negfpy.anharmonic import read_phono3py_silicon_reference

    reference = read_phono3py_silicon_reference(
        reference_path,
        expected_sha256=identities[
            "target_bundle_si-phono3py-reference.hdf5"
        ].sha256,
        require_checksum_sidecar=False,
    )
    if not reference.make_r0_average:
        raise SiliconTargetQCLIError(
            "Target reference must use make_r0_average=True."
        )
    normalized_manifest = _read_json(
        normalized_manifest_path,
        name="target normalized input manifest",
    )
    source_verification = _read_json(
        root / "source-verification.json",
        name="target source-verification",
    )
    source_commit = _validate_target_source_records(
        normalized_manifest=normalized_manifest,
        source_verification=source_verification,
        serial_report=serial_report,
        yaml_identity=identities["target_phono3py_yaml"],
    )
    settings = _read_json(
        root / "reference-settings.json",
        name="target reference-settings",
    )
    stage = _read_json(
        root / "reference-stage.json",
        name="target reference-stage",
    )
    _validate_target_settings(
        settings=settings,
        stage=stage,
        reference=reference,
    )
    inventory = _read_json(
        root / "inventory.json",
        name="target inventory",
    )
    _validate_target_inventory(
        inventory,
        checksums=checksums,
        root=root,
    )
    generator_path = _regular_file(
        repository / "scripts" / "generate_phono3py_si_reference.py",
        name="current silicon generator",
    )
    generator_sha256 = _sha256_file(generator_path)
    for name in (
        "force-constants-environment.json",
        "reference-environment.json",
    ):
        _validate_target_environment_record(
            _read_json(root / name, name=f"target {name}"),
            name=f"target {name}",
            code_identity=code_identity,
            serial_report=serial_report,
            generator_sha256=generator_sha256,
        )
    _validate_reference_mpi_orchestration(
        root=root,
        checksums=checksums,
        reference=reference,
        stage=stage,
        repository=repository,
        code_identity=code_identity,
        serial_generator_sha256=generator_sha256,
    )
    return reference, dict(sorted(identities.items())), checksums, source_commit


def _integer_sequence(
    values: Sequence[int] | None,
    *,
    default: Sequence[int],
    upper: int,
    name: str,
) -> tuple[int, ...]:
    raw = default if values is None else values
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise SiliconTargetQCLIError(f"{name} must contain only integers.")
    selected = tuple(int(value) for value in raw)
    if not selected:
        raise SiliconTargetQCLIError(f"{name} cannot be empty.")
    if len(set(selected)) != len(selected):
        raise SiliconTargetQCLIError(f"{name} cannot contain duplicates.")
    if any(value < 0 or value >= upper for value in selected):
        raise SiliconTargetQCLIError(
            f"{name} contains an index outside [0, {upper})."
        )
    return tuple(sorted(selected))


def _finite_sequence(
    values: Any,
    *,
    name: str,
    nonnegative: bool,
) -> tuple[float, ...]:
    if not isinstance(values, (tuple, list)) or not values:
        raise SiliconTargetQCLIError(f"{name} must be a nonempty array.")
    result = tuple(float(value) for value in values)
    if any(
        not math.isfinite(value) or (nonnegative and value < 0.0)
        for value in result
    ):
        raise SiliconTargetQCLIError(f"{name} contains an invalid value.")
    return result


@dataclass(frozen=True)
class ProblemLoadSpec:
    fc2_compact_path: str
    fc3_compact_path: str
    phono3py_yaml_path: str
    mesh: tuple[int, int, int]
    sigma_thz: float
    cutoff_frequency_thz: float
    frequency_factor_to_thz: float
    degenerate_tolerance_thz: float
    make_r0_average: bool
    symprec: float
    translation_atol: float
    translation_rtol: float
    asr_atol: float
    asr_rtol: float
    metadata: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return _json_value(self)


def _problem_spec(
    report: Mapping[str, Any],
    explicit_paths: Mapping[str, Path],
    input_records: Mapping[str, FileIdentity],
    target_reference: Any,
    target_checksums: Mapping[str, str],
    source_commit: str,
) -> tuple[
    ProblemLoadSpec,
    tuple[float, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    int,
]:
    controls = report.get("controls")
    sample_manifest = report.get("interaction_sample_manifest")
    if not isinstance(controls, Mapping) or not isinstance(
        sample_manifest, Mapping
    ):
        raise SiliconTargetQCLIError(
            "Serial report lacks exact qualified ASR/degeneracy controls."
        )
    mesh = tuple(int(value) for value in target_reference.mesh)
    n_qpoints = int(target_reference.n_qpoints)
    n_bands = int(target_reference.n_bands)
    reference_targets = _integer_sequence(
        target_reference.grid_points_grg.tolist(),
        default=(),
        upper=n_qpoints,
        name="target-reference grid points",
    )
    reference_bands = _integer_sequence(
        target_reference.band_indices.tolist(),
        default=(),
        upper=n_bands,
        name="target-reference band indices",
    )
    temperatures = _finite_sequence(
        target_reference.temperatures_k.tolist(),
        name="target-reference temperatures",
        nonnegative=True,
    )
    sigma = float(target_reference.sigmas_thz[0])
    cutoff = float(target_reference.cutoff_frequency_thz)
    # The generated reference schema does not store the gauge-subspace
    # tolerance.  Reuse the exact value sealed by the passed qualification
    # report rather than introducing a free campaign control.
    degeneracy = float(
        sample_manifest.get("degeneracy_tolerance_thz")
    )
    if (
        not math.isfinite(sigma)
        or sigma <= 0.0
        or not math.isfinite(cutoff)
        or cutoff < 0.0
        or not math.isfinite(degeneracy)
        or degeneracy <= 0.0
    ):
        raise SiliconTargetQCLIError(
            "Target sigma, cutoff, or degeneracy tolerance is invalid."
        )
    if target_reference.make_r0_average is not True:
        raise SiliconTargetQCLIError(
            "Silicon campaign requires target make_r0_average=True."
        )
    asr_atol = float(controls.get("asr_atol"))
    asr_rtol = float(controls.get("asr_rtol"))
    if asr_atol != 1.0e-10 or asr_rtol != 1.0e-8:
        raise SiliconTargetQCLIError(
            "Serial report does not use certification ASR controls."
        )

    from negfpy.anharmonic import (
        PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
        PHONO3PY_4_4_SYMPREC,
    )

    metadata = {
        "validation_driver": SILICON_ORACLE_REPORT_SCHEMA,
        "reference_sha256": input_records[
            "target_bundle_si-phono3py-reference.hdf5"
        ].sha256,
        "fc2_compact_sha256": input_records[
            "target_bundle_fc2.compact.hdf5"
        ].sha256,
        "fc2_full_sha256": target_checksums["fc2.full.hdf5"],
        "fc3_compact_sha256": input_records[
            "target_bundle_fc3.compact.hdf5"
        ].sha256,
        "fc3_full_sha256": target_checksums["fc3.full.hdf5"],
        "source_commit": source_commit,
        "no_fitted_prefactors": True,
        "internal_q_parallelism": "none",
    }
    return (
        ProblemLoadSpec(
            fc2_compact_path=explicit_paths["fc2_compact"].as_posix(),
            fc3_compact_path=explicit_paths["fc3_compact"].as_posix(),
            phono3py_yaml_path=explicit_paths["phono3py_yaml"].as_posix(),
            mesh=mesh,
            sigma_thz=sigma,
            cutoff_frequency_thz=cutoff,
            frequency_factor_to_thz=(
                PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ
            ),
            degenerate_tolerance_thz=degeneracy,
            make_r0_average=True,
            symprec=PHONO3PY_4_4_SYMPREC,
            translation_atol=TRANSLATIONAL_FC3_ATOL,
            translation_rtol=TRANSLATIONAL_FC3_RTOL,
            asr_atol=asr_atol,
            asr_rtol=asr_rtol,
            metadata=MappingProxyType(metadata),
        ),
        temperatures,
        reference_targets,
        reference_bands,
        n_qpoints,
        n_bands,
    )


def _load_problem(spec: ProblemLoadSpec) -> Any:
    """Load exactly the compact problem used by the strict serial validator."""

    from negfpy.anharmonic._silicon_bulk_problem import (
        SiliconBulkProblemError,
        load_silicon_bulk_three_phonon_problem,
    )

    try:
        return load_silicon_bulk_three_phonon_problem(
            fc2_compact_path=spec.fc2_compact_path,
            fc3_compact_path=spec.fc3_compact_path,
            phono3py_yaml_path=spec.phono3py_yaml_path,
            mesh=spec.mesh,
            sigma_thz=spec.sigma_thz,
            cutoff_frequency_thz=spec.cutoff_frequency_thz,
            frequency_factor_to_thz=spec.frequency_factor_to_thz,
            degenerate_tolerance_thz=spec.degenerate_tolerance_thz,
            make_r0_average=spec.make_r0_average,
            symprec=spec.symprec,
            translation_atol=spec.translation_atol,
            translation_rtol=spec.translation_rtol,
            asr_atol=spec.asr_atol,
            asr_rtol=spec.asr_rtol,
            metadata=spec.metadata,
        )
    except SiliconBulkProblemError as exc:
        raise SiliconTargetQCLIError(
            str(exc)
        ) from exc


class _LazyProblemFactory:
    def __init__(self, spec: ProblemLoadSpec, preloaded: Any | None = None):
        self.spec = spec
        self.cached = [] if preloaded is None else [preloaded]

    def __call__(self) -> Any:
        if not self.cached:
            self.cached.append(_load_problem(self.spec))
        return self.cached[0]


def _deep_resident_bytes(value: Any) -> int:
    seen: set[int] = set()

    def visit(item: Any) -> int:
        identity = id(item)
        if identity in seen:
            return 0
        seen.add(identity)
        size = sys.getsizeof(item)
        if isinstance(item, np.ndarray):
            return size + int(item.nbytes)
        if is_dataclass(item) and not isinstance(item, type):
            return size + sum(visit(getattr(item, field.name)) for field in fields(item))
        if isinstance(item, Mapping):
            return size + sum(visit(key) + visit(child) for key, child in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return size + sum(visit(child) for child in item)
        return size

    return visit(value)


def _target_kernel_bytes(
    *,
    n_targets: int,
    n_qpoints: int,
    n_target_bands: int,
    n_bands: int,
    n_temperatures: int,
) -> int:
    fields = n_targets * n_qpoints * n_target_bands * n_bands**2
    interactions = fields * 24 + n_targets * n_qpoints * 41
    phonons = n_qpoints * (
        3 * 8 + n_bands * 8 + 2 * n_bands**2 * 16
    )
    linewidths = (
        4 * n_temperatures * n_targets * n_target_bands * 8
        + 2 * n_targets * n_target_bands * 8
    )
    # Constructors defensively copy arrays, so include source, immutable copy,
    # and one contraction/workspace allowance.
    return int(3 * (interactions + phonons + linewidths))


def _available_memory_bytes() -> int:
    try:
        fields = {
            line.split(":", 1)[0]: line.split(":", 1)[1].strip()
            for line in Path("/proc/meminfo").read_text(
                encoding="ascii"
            ).splitlines()
            if ":" in line
        }
        amount, unit = fields["MemAvailable"].split()
        if unit != "kB":
            raise ValueError("unsupported MemAvailable unit")
        result = int(amount) * 1024
    except (OSError, KeyError, ValueError) as exc:
        raise SiliconTargetQCLIError(
            "Cannot read Linux /proc/meminfo MemAvailable."
        ) from exc
    if result <= 0:
        raise SiliconTargetQCLIError("MemAvailable is not positive.")
    return result


def _memory_preflight(
    problem: Any,
    *,
    n_tasks: int,
    n_qpoints: int,
    n_bands: int,
    n_target_bands: int,
    n_temperatures: int,
    communicator_size: int,
    compare_monolithic: bool,
    available_memory_bytes: int | None = None,
) -> dict[str, Any]:
    problem_bytes = _deep_resident_bytes(problem)
    one_target = _target_kernel_bytes(
        n_targets=1,
        n_qpoints=n_qpoints,
        n_target_bands=n_target_bands,
        n_bands=n_bands,
        n_temperatures=n_temperatures,
    )
    per_worker = problem_bytes + one_target + FIXED_PROCESS_OVERHEAD_BYTES
    if communicator_size == 1:
        active_workers = 1
    else:
        active_workers = min(max(communicator_size - 1, 0), n_tasks)
    # Canonical merge validation materializes every merged numerical field,
    # even without a monolithic comparison.  A comparison additionally holds
    # one complete authoritative serial result.
    merge_kernel = _target_kernel_bytes(
        n_targets=n_tasks,
        n_qpoints=n_qpoints,
        n_target_bands=n_target_bands,
        n_bands=n_bands,
        n_temperatures=n_temperatures,
    )
    if compare_monolithic:
        merge_kernel += _target_kernel_bytes(
            n_targets=n_tasks,
            n_qpoints=n_qpoints,
            n_target_bands=n_target_bands,
            n_bands=n_bands,
            n_temperatures=n_temperatures,
        )
    coordinator = problem_bytes + merge_kernel + FIXED_PROCESS_OVERHEAD_BYTES
    predicted = active_workers * per_worker + coordinator
    available = (
        _available_memory_bytes()
        if available_memory_bytes is None
        else int(available_memory_bytes)
    )
    if available <= 0:
        raise SiliconTargetQCLIError(
            "available_memory_bytes must be positive."
        )
    limit = int(MEMORY_LIMIT_FRACTION * available)
    return {
        "schema": MEMORY_PREFLIGHT_SCHEMA,
        "schema_version": MEMORY_ESTIMATE_VERSION,
        "measurement": "linux_proc_meminfo_MemAvailable",
        "allowed_fraction": MEMORY_LIMIT_FRACTION,
        "available_memory_bytes": available,
        "limit_bytes": limit,
        "communicator_size": communicator_size,
        "active_worker_count": active_workers,
        "problem_resident_bytes": problem_bytes,
        "one_target_kernel_and_artifact_bytes": one_target,
        "fixed_process_overhead_bytes": FIXED_PROCESS_OVERHEAD_BYTES,
        "per_worker_predicted_bytes": per_worker,
        "coordinator_merge_allowance_bytes": coordinator,
        "aggregate_predicted_bytes": predicted,
        "monolithic_comparison_included": compare_monolithic,
        "passed": predicted <= limit,
    }


def _records_unchanged(records: Mapping[str, FileIdentity]) -> None:
    for role, before in records.items():
        after = FileIdentity.from_path(
            _regular_file(before.path, name=f"input {role}")
        )
        if after != before:
            raise SiliconTargetQCLIError(
                f"Input {role!r} changed during the campaign."
            )


def _code_identity_unchanged(
    prepared: Mapping[str, Any],
    records: Mapping[str, FileIdentity],
) -> None:
    """Re-authenticate the repository and imported payload after a run."""

    before = prepared.get("code_identity")
    serial_record = records.get("serial_report")
    if not isinstance(before, Mapping) or serial_record is None:
        raise SiliconTargetQCLIError(
            "Prepared campaign lacks final code-identity evidence."
        )
    repository_value = before.get("repository")
    if not isinstance(repository_value, str):
        raise SiliconTargetQCLIError(
            "Prepared campaign lacks its canonical repository path."
        )
    report = _load_passed_serial_report(
        _regular_file(
            serial_record.path,
            name="serial_report",
        )
    )
    after = _clean_code_identity(
        _directory(repository_value, name="repository"),
        report,
    )
    if _json_value(after) != _json_value(before):
        raise SiliconTargetQCLIError(
            "Git or imported NEGFPy code identity changed during the campaign."
        )


def _execution_id(value: str | None) -> str:
    selected = (
        value
        if value is not None
        else datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        + "-"
        + uuid4().hex[:12]
    )
    if _EXECUTION_ID_RE.fullmatch(selected) is None:
        raise SiliconTargetQCLIError(
            "execution_id may contain only letters, numbers, '.', '_' and '-'."
        )
    return selected


def _rank_zero_prepare(
    arguments: argparse.Namespace,
    *,
    communicator_size: int,
) -> tuple[dict[str, Any], Any]:
    serial_path = _regular_file(
        arguments.serial_report, name="serial_report"
    )
    report = _load_passed_serial_report(serial_path)
    repository = _directory(arguments.repository, name="repository")
    code_identity = _clean_code_identity(repository, report)
    explicit_paths = {
        name: _regular_file(getattr(arguments, name), name=name)
        for name in (
            "reference",
            "fc2_compact",
            "fc3_compact",
            "phono3py_yaml",
            "environment_lock",
            "target_checksums",
            "target_input_manifest",
        )
    }
    report_records = _verify_report_inputs(report)
    (
        target_reference,
        target_records,
        target_checksums,
        source_commit,
    ) = _verify_target_bundle(
        reference_path=explicit_paths["reference"],
        fc2_compact_path=explicit_paths["fc2_compact"],
        fc3_compact_path=explicit_paths["fc3_compact"],
        yaml_path=explicit_paths["phono3py_yaml"],
        environment_lock_path=explicit_paths["environment_lock"],
        checksums_path=explicit_paths["target_checksums"],
        normalized_manifest_path=explicit_paths["target_input_manifest"],
        report_records=report_records,
        serial_report=report,
        code_identity=code_identity,
        repository=repository,
    )
    input_records = {**report_records, **target_records}
    serial_identity = FileIdentity.from_path(serial_path)
    input_records["serial_report"] = serial_identity
    (
        spec,
        temperatures,
        reference_targets,
        reference_bands,
        n_qpoints,
        n_bands,
    ) = _problem_spec(
        report,
        explicit_paths,
        input_records,
        target_reference,
        target_checksums,
        source_commit,
    )
    targets = _integer_sequence(
        arguments.grid_points,
        default=reference_targets,
        upper=n_qpoints,
        name="grid_points",
    )
    bands = _integer_sequence(
        arguments.band_indices,
        default=reference_bands,
        upper=n_bands,
        name="band_indices",
    )
    if not set(targets).issubset(reference_targets):
        raise SiliconTargetQCLIError(
            "grid_points must be a subset of the authenticated target "
            "reference selection."
        )
    if not set(bands).issubset(reference_bands):
        raise SiliconTargetQCLIError(
            "band_indices must be a subset of the authenticated target "
            "reference selection."
        )
    if arguments.compare_monolithic:
        qualifications = _require_json_mapping(
            report.get("selections"),
            name="serial qualification selections",
        )
        qualified_mesh = tuple(
            int(value) for value in qualifications.get("mesh", ())
        )
        same_controls = (
            spec.mesh == qualified_mesh
            and reference_targets
            == tuple(
                int(value)
                for value in qualifications.get("grid_points_grg", ())
            )
            and reference_bands
            == tuple(
                int(value)
                for value in qualifications.get("band_indices", ())
            )
            and temperatures
            == tuple(
                float(value)
                for value in qualifications.get("temperatures_k", ())
            )
            and spec.sigma_thz
            == float(qualifications.get("sigma_thz", float("nan")))
            and spec.cutoff_frequency_thz
            == float(
                qualifications.get(
                    "cutoff_frequency_thz",
                    float("nan"),
                )
            )
        )
        if (
            spec.mesh not in ((3, 3, 3), (5, 5, 5))
            or not same_controls
        ):
            raise SiliconTargetQCLIError(
                "--compare-monolithic requires an exact qualified 3x3x3 "
                "or 5x5x5 target mesh with matching physical controls."
            )

    problem = _load_problem(spec)
    actual_qpoints = int(problem.mesh.n_qpoints)
    actual_bands = int(3 * problem.fc2_mapping.n_primitive_atoms)
    if actual_qpoints != n_qpoints or actual_bands != n_bands:
        raise SiliconTargetQCLIError(
            "Loaded compact problem dimensions differ from the serial report."
        )
    problem_hash = bulk_three_phonon_problem_sha256(problem)
    memory = _memory_preflight(
        problem,
        n_tasks=len(targets),
        n_qpoints=n_qpoints,
        n_bands=n_bands,
        n_target_bands=len(bands),
        n_temperatures=len(temperatures),
        communicator_size=communicator_size,
        compare_monolithic=bool(arguments.compare_monolithic),
    )
    override_used = not memory["passed"] and bool(
        arguments.allow_memory_override
    )
    if not memory["passed"] and not override_used:
        raise SiliconTargetQCLIError(
            "Predicted aggregate target-q memory exceeds 70% of current "
            "MemAvailable. Refusing launch; --allow-memory-override is an "
            "explicit noncertifying override."
        )

    input_hashes = {
        role: record.sha256 for role, record in input_records.items()
    }
    input_hashes["repository_tree_identity"] = code_identity[
        "tree_identity_sha256"
    ]
    controls = BulkTargetQControls(
        temperatures=temperatures,
        grid_points=targets,
        band_indices=bands,
        n_qpoints=n_qpoints,
        n_bands=n_bands,
        problem_sha256=problem_hash,
        input_hashes=input_hashes,
    )
    campaign_metadata = {
        "driver_schema": CAMPAIGN_CLI_SCHEMA,
        "serial_report_sha256": serial_identity.sha256,
        "serial_report_certification_eligible": True,
        "qualification_mesh": report["selections"]["mesh"],
        "target_reference_sha256": target_reference.sha256,
        "target_checksums_sha256": target_records[
            "target_sha256sums"
        ].sha256,
        "target_input_manifest_sha256": target_records[
            "target_bundle_input-manifest.normalized.json"
        ].sha256,
        "mesh": list(spec.mesh),
        "sigma_thz": spec.sigma_thz,
        "cutoff_frequency_thz": spec.cutoff_frequency_thz,
        "degenerate_tolerance_thz": spec.degenerate_tolerance_thz,
        "make_r0_average": spec.make_r0_average,
        "complete_internal_q1_sum_per_task": True,
        "internal_q1_mpi": False,
        "memory_override_used": override_used,
    }
    campaign = build_bulk_target_q_campaign(
        controls,
        source_revision=code_identity["head_commit"],
        source_tree_sha256=code_identity["source_tree_sha256"],
        environment_sha256=(
            input_records["target_environment_lock"].sha256
        ),
        metadata=campaign_metadata,
    )
    prepared = {
        "schema": CAMPAIGN_CLI_SCHEMA,
        "schema_version": 1,
        "code_identity": code_identity,
        "problem_spec": spec.as_dict(),
        "problem_sha256": problem_hash,
        "controls": controls.as_dict(),
        "campaign_metadata": campaign_metadata,
        "campaign": campaign.as_dict(),
        "input_records": {
            role: record.as_dict()
            for role, record in sorted(input_records.items())
        },
        "memory_preflight": memory,
        "memory_override_used": override_used,
    }
    return prepared, problem


def _spec_from_dict(value: Mapping[str, Any]) -> ProblemLoadSpec:
    return ProblemLoadSpec(
        fc2_compact_path=str(value["fc2_compact_path"]),
        fc3_compact_path=str(value["fc3_compact_path"]),
        phono3py_yaml_path=str(value["phono3py_yaml_path"]),
        mesh=tuple(int(item) for item in value["mesh"]),
        sigma_thz=float(value["sigma_thz"]),
        cutoff_frequency_thz=float(value["cutoff_frequency_thz"]),
        frequency_factor_to_thz=float(value["frequency_factor_to_thz"]),
        degenerate_tolerance_thz=float(value["degenerate_tolerance_thz"]),
        make_r0_average=bool(value["make_r0_average"]),
        symprec=float(value["symprec"]),
        translation_atol=float(value["translation_atol"]),
        translation_rtol=float(value["translation_rtol"]),
        asr_atol=float(value["asr_atol"]),
        asr_rtol=float(value["asr_rtol"]),
        metadata=MappingProxyType(dict(value["metadata"])),
    )


def _campaign_from_prepared(prepared: Mapping[str, Any]) -> Any:
    raw_controls = prepared["controls"]
    controls = BulkTargetQControls(
        temperatures=raw_controls["temperatures"],
        grid_points=raw_controls["grid_points"],
        band_indices=raw_controls["band_indices"],
        n_qpoints=int(raw_controls["n_qpoints"]),
        n_bands=int(raw_controls["n_bands"]),
        problem_sha256=str(raw_controls["problem_sha256"]),
        input_hashes=raw_controls["input_hashes"],
    )
    code = prepared["code_identity"]
    campaign = build_bulk_target_q_campaign(
        controls,
        source_revision=str(code["head_commit"]),
        source_tree_sha256=str(code["source_tree_sha256"]),
        environment_sha256=str(
            raw_controls["input_hashes"]["target_environment_lock"]
        ),
        metadata=prepared["campaign_metadata"],
    )
    expected = prepared["campaign"]["campaign_manifest_sha256"]
    if campaign.campaign_manifest_sha256 != expected:
        raise SiliconTargetQCLIError(
            "Broadcast campaign descriptor did not reproduce the rank-zero "
            "manifest."
        )
    return campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one complete-target-q silicon campaign serially or through "
            "task-level MPI. Internal q-prime sums remain serial and complete."
        )
    )
    parser.add_argument("--serial-report", required=True, type=Path)
    parser.add_argument(
        "--reference",
        required=True,
        type=Path,
        help="Authenticated target-mesh phono3py reference HDF5.",
    )
    parser.add_argument("--fc2-compact", required=True, type=Path)
    parser.add_argument("--fc3-compact", required=True, type=Path)
    parser.add_argument("--phono3py-yaml", required=True, type=Path)
    parser.add_argument("--environment-lock", required=True, type=Path)
    parser.add_argument(
        "--target-checksums",
        required=True,
        type=Path,
        help="Target reference generator's immutable SHA256SUMS.",
    )
    parser.add_argument(
        "--target-input-manifest",
        required=True,
        type=Path,
        help="Target generator's input-manifest.normalized.json.",
    )
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--run-directory", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("serial", "mpi"), default="serial"
    )
    parser.add_argument("--execution-id")
    parser.add_argument("--grid-points", nargs="+", type=int)
    parser.add_argument("--band-indices", nargs="+", type=int)
    parser.add_argument("--compare-monolithic", action="store_true")
    parser.add_argument(
        "--allow-memory-override",
        action="store_true",
        help=(
            "Proceed above the 70%% RAM limit and mark the run "
            "noncertifying."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume only exact immutable checkpoints in an existing run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Authenticate and build the manifest without numerical tasks.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        help="Optional immutable task-manifest path used with --dry-run.",
    )
    return parser


def _communicator(mode: str) -> Any | None:
    if mode == "serial":
        return None
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise SiliconTargetQCLIError(
            "--mode mpi requires mpi4py from the pinned environment."
        ) from exc
    return MPI.COMM_WORLD


def _require_parallel_mpi(mode: str, communicator: Any | None) -> None:
    if mode == "mpi":
        assert communicator is not None
        if int(communicator.Get_size()) < 2:
            raise SiliconTargetQCLIError(
                "--mode mpi requires at least two MPI ranks; a one-rank "
                "launch is rejected rather than represented as parallel "
                "certification evidence."
            )


def _broadcast_prepared(
    arguments: argparse.Namespace,
    communicator: Any | None,
) -> tuple[dict[str, Any], Any | None, int, int]:
    rank = 0 if communicator is None else int(communicator.Get_rank())
    size = 1 if communicator is None else int(communicator.Get_size())
    prepared: dict[str, Any] | None = None
    preloaded: Any | None = None
    error: dict[str, str] | None = None
    if rank == 0:
        try:
            prepared, preloaded = _rank_zero_prepare(
                arguments,
                communicator_size=size,
            )
        except Exception as exc:
            error = {
                "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "message": str(exc),
            }
    if communicator is not None:
        envelope = communicator.bcast(
            {"prepared": prepared, "error": error}
            if rank == 0
            else None,
            root=0,
        )
        prepared = envelope["prepared"]
        error = envelope["error"]
    if error is not None:
        raise SiliconTargetQCLIError(
            f"Rank-zero preflight failed: {error['type']}: {error['message']}"
        )
    assert prepared is not None
    return prepared, preloaded, rank, size


def _initialize_run_directory(
    arguments: argparse.Namespace,
    *,
    communicator: Any | None,
    rank: int,
    execution_id: str,
    prepared: Mapping[str, Any],
) -> Path:
    run_directory: Path | None = None
    error: dict[str, str] | None = None
    if rank == 0:
        try:
            run_directory = _path_without_symlinks(
                arguments.run_directory,
                name="run_directory",
            )
            if arguments.resume:
                if not run_directory.is_dir():
                    raise SiliconTargetQCLIError(
                        "--resume requires an existing regular run directory."
                    )
            elif run_directory.exists():
                raise SiliconTargetQCLIError(
                    "Fresh run_directory already exists; use a new path or "
                    "--resume exact immutable checkpoints."
                )
            run_directory.mkdir(
                parents=True,
                exist_ok=arguments.resume,
            )
            _write_immutable_json(
                run_directory / "preflights" / f"{execution_id}.json",
                {
                    "schema": CAMPAIGN_CLI_SCHEMA,
                    "schema_version": 1,
                    "execution_id": execution_id,
                    "campaign": prepared["campaign"],
                    "problem_sha256": prepared["problem_sha256"],
                    "code_identity": prepared["code_identity"],
                    "input_records": prepared["input_records"],
                    "memory_preflight": prepared["memory_preflight"],
                    "memory_override_used": prepared[
                        "memory_override_used"
                    ],
                },
            )
        except Exception as exc:
            error = {
                "type": (
                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                ),
                "message": str(exc),
            }
    if communicator is not None:
        envelope = communicator.bcast(
            {
                "run_directory": (
                    None
                    if run_directory is None
                    else run_directory.as_posix()
                ),
                "error": error,
            }
            if rank == 0
            else None,
            root=0,
        )
        error = envelope["error"]
        path_value = envelope["run_directory"]
        run_directory = (
            None if path_value is None else Path(str(path_value))
        )
    if error is not None:
        raise SiliconTargetQCLIError(
            "Rank-zero run-directory initialization failed: "
            f"{error['type']}: {error['message']}"
        )
    assert run_directory is not None
    return run_directory


def _final_summary(
    *,
    execution: Any,
    merge: Any,
    comparison: Any | None,
    campaign: Any,
    run_directory: Path,
    prepared: Mapping[str, Any],
    execution_id: str,
    serial_archive: BulkSerialResultArchive | None = None,
) -> dict[str, Any]:
    if (comparison is None) is not (serial_archive is None):
        raise SiliconTargetQCLIError(
            "monolithic comparison and serial-result archive must either "
            "both be present or both be absent."
        )
    comparison_passed = comparison is None or comparison.passed
    task_manifest_path = run_directory / "task-manifest.json"
    execution_manifest_path = (
        run_directory / "executions" / f"{execution_id}.json"
    )
    merge_report_path = run_directory / "reports" / "merge.json"
    comparison_path = (
        run_directory / "reports" / "monolithic-comparison.json"
    )
    input_records = prepared["input_records"]
    target_records = {
        role: record
        for role, record in input_records.items()
        if role.startswith("target_")
    }
    selected_grid_points = tuple(
        int(value) for value in campaign.controls.grid_points
    )
    complete_target_set = selected_grid_points == tuple(
        range(campaign.controls.n_qpoints)
    )
    selected_band_indices = tuple(
        int(value) for value in campaign.controls.band_indices
    )
    complete_band_set = selected_band_indices == tuple(
        range(campaign.controls.n_bands)
    )
    complete_scientific_selection = (
        complete_target_set and complete_band_set
    )
    assignments = tuple(execution.assignments)
    executed_task_ids = tuple(
        item.task_id for item in assignments if not item.resumed
    )
    resumed_task_ids = tuple(
        item.task_id for item in assignments if item.resumed
    )
    orphan_artifact_adopted_task_ids = tuple(
        item.task_id
        for item in assignments
        if getattr(item, "orphan_artifact_adopted", False)
    )
    if any(
        getattr(item, "orphan_artifact_adopted", False)
        and not item.resumed
        for item in assignments
    ):
        raise SiliconTargetQCLIError(
            "an adopted preexisting artifact is not marked resumed."
        )
    fresh_complete_task_execution = (
        execution.complete
        and len(assignments) == len(campaign.manifest.tasks)
        and not resumed_task_ids
        and not orphan_artifact_adopted_task_ids
    )
    fresh_complete_mpi_execution = (
        fresh_complete_task_execution
        and execution.mode == "mpi"
        and execution.communicator_size >= 2
        and all(item.rank > 0 for item in assignments)
    )
    mpi_equality_eligible = (
        comparison is not None
        and comparison_passed
        and fresh_complete_mpi_execution
        and complete_scientific_selection
        and not prepared["memory_override_used"]
    )
    return {
        "schema": CAMPAIGN_CLI_SCHEMA,
        "schema_version": 1,
        "status": "passed" if comparison_passed else "failed",
        "campaign_manifest_sha256": prepared["campaign"][
            "campaign_manifest_sha256"
        ],
        "problem_sha256": prepared["problem_sha256"],
        "campaign": prepared["campaign"],
        "controls": prepared["controls"],
        "controls_sha256": campaign.controls.controls_sha256,
        "selected_grid_points": list(selected_grid_points),
        "selected_band_indices": list(selected_band_indices),
        "complete_target_set": complete_target_set,
        "complete_band_set": complete_band_set,
        "complete_scientific_selection": complete_scientific_selection,
        "execution_id": execution_id,
        "execution": execution.as_dict(),
        "execution_qualification": {
            "executed_task_count": len(executed_task_ids),
            "resumed_task_count": len(resumed_task_ids),
            "orphan_artifact_adopted_task_count": len(
                orphan_artifact_adopted_task_ids
            ),
            "executed_task_ids": list(executed_task_ids),
            "resumed_task_ids": list(resumed_task_ids),
            "orphan_artifact_adopted_task_ids": list(
                orphan_artifact_adopted_task_ids
            ),
            "fresh_complete_task_execution": (
                fresh_complete_task_execution
            ),
            "fresh_complete_mpi_execution": (
                fresh_complete_mpi_execution
            ),
            "mpi_equality_eligible": mpi_equality_eligible,
        },
        "merge": merge.as_dict(),
        "monolithic_comparison": (
            None if comparison is None else comparison.as_dict()
        ),
        "memory_preflight": prepared["memory_preflight"],
        "memory_override_used": prepared["memory_override_used"],
        "provenance_eligible": (
            comparison_passed and not prepared["memory_override_used"]
        ),
        "external_validation_eligible": (
            comparison_passed
            and not prepared["memory_override_used"]
            and complete_scientific_selection
        ),
        "scientific_certification": {
            "certified": False,
            "status": "pending_external_comparison",
            "reason": (
                "Campaign completion and optional monolithic equality do not "
                "replace comparison with the independent target phono3py "
                "reference."
            ),
        },
        "release_certified": False,
        "evidence": {
            "task_manifest": {
                **FileIdentity.from_path(task_manifest_path).as_dict(),
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
            },
            "execution_manifest": {
                **FileIdentity.from_path(execution_manifest_path).as_dict(),
                "execution_sha256": execution.execution_sha256,
            },
            "merge_report": {
                **FileIdentity.from_path(merge_report_path).as_dict(),
                "report_sha256": merge.report_sha256,
            },
            "merged_hdf5": FileIdentity.from_path(
                merge.merged_artifact_path
            ).as_dict(),
            "qualification_report": input_records["serial_report"],
            "target_reference": input_records[
                "target_bundle_si-phono3py-reference.hdf5"
            ],
            "target_generation_records": target_records,
            "monolithic_comparison_report": (
                None
                if comparison is None
                else {
                    **FileIdentity.from_path(comparison_path).as_dict(),
                    "report_sha256": comparison.report_sha256,
                }
            ),
            "serial_result_archive": (
                None
                if serial_archive is None
                else {
                    **FileIdentity.from_path(serial_archive.path).as_dict(),
                    "result_sha256": serial_archive.result_sha256,
                    "dataset_sha256": dict(
                        serial_archive.dataset_sha256
                    ),
                    "provenance": dict(serial_archive.provenance),
                }
            ),
        },
        "no_internal_q1_mpi": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    communicator = _communicator(arguments.mode)
    _require_parallel_mpi(arguments.mode, communicator)
    prepared, preloaded, rank, size = _broadcast_prepared(
        arguments, communicator
    )
    campaign = _campaign_from_prepared(prepared)
    execution_id = _execution_id(arguments.execution_id)
    if communicator is not None:
        execution_id = communicator.bcast(
            execution_id if rank == 0 else None,
            root=0,
        )

    if arguments.dry_run:
        if rank == 0:
            if arguments.manifest_output is not None:
                write_task_manifest(
                    arguments.manifest_output,
                    campaign.manifest,
                )
            print(
                json.dumps(
                    {
                        "status": "dry-run",
                        "campaign": campaign.as_dict(),
                        "memory_preflight": prepared["memory_preflight"],
                        "memory_override_used": prepared[
                            "memory_override_used"
                        ],
                    },
                    sort_keys=True,
                )
            )
        return 0
    if arguments.manifest_output is not None:
        raise SiliconTargetQCLIError(
            "--manifest-output is accepted only with --dry-run."
        )

    run_directory = _initialize_run_directory(
        arguments,
        communicator=communicator,
        rank=rank,
        execution_id=execution_id,
        prepared=prepared,
    )

    spec = _spec_from_dict(prepared["problem_spec"])
    # In serial mode reuse the problem already authenticated by rank zero.
    # In MPI mode rank zero is coordinator-only and workers load on first task.
    factory = _LazyProblemFactory(
        spec,
        preloaded=preloaded if size == 1 else None,
    )
    if size > 1 and rank == 0:
        preloaded = None

    execution = execute_bulk_target_q_campaign(
        campaign,
        factory,
        run_directory,
        comm=communicator,
        require_mpi=arguments.mode == "mpi",
        thread_limit=1,
        execution_id=execution_id,
    )
    if rank != 0:
        return 0
    if execution is None or not execution.complete:
        raise SiliconTargetQCLIError(
            "Target-q task execution did not complete."
        )

    merge = merge_bulk_target_q_campaign(campaign, run_directory)
    report_root = run_directory / "reports"
    write_bulk_target_q_merge_report(
        report_root / "merge.json",
        merge,
    )
    comparison = None
    serial_archive = None
    if arguments.compare_monolithic:
        from negfpy.anharmonic import compute_three_phonon_linewidths

        serial_result = compute_three_phonon_linewidths(
            factory(),
            temperatures=campaign.controls.temperatures,
            grid_points=campaign.controls.grid_points,
            band_indices=campaign.controls.band_indices,
        )
        serial_archive = write_bulk_serial_result_archive(
            run_directory
            / "serial-oracle"
            / "bulk-serial-result.hdf5",
            serial_result,
            provenance={
                "schema": (
                    "negfpy.bulk_three_phonon_serial_result_provenance.v1"
                ),
                "schema_version": 1,
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "controls_sha256": campaign.controls.controls_sha256,
                "problem_sha256": campaign.controls.problem_sha256,
                "source_revision": campaign.manifest.source_revision,
                "source_tree_sha256": (
                    campaign.manifest.source_tree_sha256
                ),
                "environment_sha256": (
                    campaign.manifest.environment_sha256
                ),
                "input_hashes": dict(campaign.controls.input_hashes),
                "temperatures": campaign.controls.temperatures.tolist(),
                "grid_points": campaign.controls.grid_points.tolist(),
                "band_indices": campaign.controls.band_indices.tolist(),
                "n_qpoints": campaign.controls.n_qpoints,
                "n_bands": campaign.controls.n_bands,
                "comparison_rtol": MONOLITHIC_RTOL,
                "comparison_atol": MONOLITHIC_ATOL,
            },
        )
        # The archive reader returns the independently reconstructed operand.
        # Drop the in-memory source before loading/comparing the merged result
        # so the preregistered two-complete-result memory estimate remains
        # conservative.
        del serial_result
        comparison = compare_bulk_target_q_with_serial(
            serial_archive.result,
            merge.merged_artifact_path,
            rtol=MONOLITHIC_RTOL,
            atol=MONOLITHIC_ATOL,
        )
        if (
            comparison.serial_result_sha256
            != serial_archive.result_sha256
        ):
            raise SiliconTargetQCLIError(
                "serialized authoritative result differs from the "
                "monolithic comparison operand."
            )
        write_bulk_target_q_comparison_report(
            report_root / "monolithic-comparison.json",
            comparison,
        )

    records = {
        role: FileIdentity(**record)
        for role, record in prepared["input_records"].items()
    }
    _records_unchanged(records)
    _code_identity_unchanged(prepared, records)
    summary = _final_summary(
        execution=execution,
        merge=merge,
        comparison=comparison,
        serial_archive=serial_archive,
        campaign=campaign,
        run_directory=run_directory,
        prepared=prepared,
        execution_id=execution_id,
    )
    summary_path = _write_immutable_json(
        run_directory / "summaries" / f"{execution_id}.json",
        summary,
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "summary": summary_path.as_posix(),
                "summary_sha256": _sha256_file(summary_path),
                "campaign_manifest_sha256": (
                    campaign.campaign_manifest_sha256
                ),
                "merged_artifact": merge.merged_artifact_path.as_posix(),
                "merged_artifact_sha256": merge.merged_artifact_sha256,
                "monolithic_comparison_passed": (
                    None if comparison is None else comparison.passed
                ),
                "provenance_eligible": summary["provenance_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": (
                        f"{type(error).__module__}."
                        f"{type(error).__qualname__}"
                    ),
                    "error": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(2) from error
