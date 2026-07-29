"""Fail-closed release evidence for a bulk target-q MPI campaign.

This private module is intentionally outside the numerical implementation.
It verifies archived task, execution, and merge evidence and then rebuilds the
canonical merged HDF5 file in a temporary directory.  It never trusts a path
after ``Path.resolve`` without first auditing the unresolved filesystem tree.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Any

import numpy as np

from ._bulk_target_q_mpi import (
    BulkTargetQCampaign,
    BulkTargetQMergeReport,
    _merged_artifact_evidence,
    load_bulk_target_q_result,
    make_bulk_target_q_artifact_validator,
    merge_bulk_target_q_campaign,
)
from ._mpi_tasks import (
    MPIExecutionReport,
    TaskAssignment,
    TaskCheckpoint,
    TaskManifestError,
    collect_task_checkpoints,
    task_assignment_physical_core_evidence,
    task_checkpoint_path,
    task_execution_mpi_origin_lineage_evidence,
)


_SCHEMA = "negfpy.bulk_target_q_release_evidence"
_SCHEMA_VERSION = 1
_EXECUTION_SCHEMA = "negfpy_mpi_execution_manifest"
_EXECUTION_SCHEMA_VERSION = 1
_MERGE_REPORT_SCHEMA = "negfpy.bulk_target_q_merge_report"
_MERGE_REPORT_SCHEMA_VERSION = 1
_MINIMUM_DISK_MARGIN_BYTES = 8 * 1024 * 1024
_SHA256_LENGTH = 64
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


class BulkTargetQEvidenceError(RuntimeError):
    """Raised when archived target-q release evidence is not self-consistent."""


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number {value!r} is forbidden")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_value(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, np.generic):
        return _json_value(value.item(), path=path)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"{path} cannot contain NaN or infinity")
        return result
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} keys must be non-empty strings")
            result[key] = _json_value(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _json_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__name__}"
    )


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _absolute_unresolved(path: str | Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _audit_path_components(path: Path) -> os.stat_result:
    """Reject symlinks in every existing component before opening ``path``."""

    absolute = _absolute_unresolved(path)
    parts = absolute.parts
    current = Path(parts[0])
    try:
        root_status = os.lstat(current)
    except OSError as exc:
        raise BulkTargetQEvidenceError(
            f"cannot lstat filesystem root for {absolute}"
        ) from exc
    if stat.S_ISLNK(root_status.st_mode):
        raise BulkTargetQEvidenceError(
            f"symlinked filesystem root is forbidden: {current}"
        )
    status = root_status
    for part in parts[1:]:
        current /= part
        try:
            status = os.lstat(current)
        except OSError as exc:
            raise BulkTargetQEvidenceError(
                f"required evidence path is missing: {current}"
            ) from exc
        if stat.S_ISLNK(status.st_mode):
            raise BulkTargetQEvidenceError(
                f"symlinked evidence path is forbidden: {current}"
            )
    return status


def _audit_directory(path: Path) -> Path:
    absolute = _absolute_unresolved(path)
    status = _audit_path_components(absolute)
    if not stat.S_ISDIR(status.st_mode):
        raise BulkTargetQEvidenceError(
            f"expected an evidence directory: {absolute}"
        )
    return absolute


def _audit_regular_file(
    path: Path,
    *,
    inodes: dict[tuple[int, int], str],
    label: str,
) -> tuple[Path, os.stat_result]:
    absolute = _absolute_unresolved(path)
    status = _audit_path_components(absolute)
    if not stat.S_ISREG(status.st_mode):
        raise BulkTargetQEvidenceError(
            f"{label} is not a regular file: {absolute}"
        )
    if status.st_nlink != 1:
        raise BulkTargetQEvidenceError(
            f"{label} has multiple hard links: {absolute}"
        )
    identity = (int(status.st_dev), int(status.st_ino))
    owner = inodes.get(identity)
    if owner is not None and owner != label:
        raise BulkTargetQEvidenceError(
            f"{label} duplicates the inode already used by {owner}"
        )
    inodes[identity] = label
    return absolute, status


def _relative_inside(path: Path, root: Path, *, label: str) -> str:
    absolute = _absolute_unresolved(path)
    try:
        return absolute.relative_to(root).as_posix()
    except ValueError as exc:
        raise BulkTargetQEvidenceError(
            f"{label} must be stored inside the campaign run directory"
        ) from exc


def _directory_entries(path: Path) -> tuple[os.DirEntry[str], ...]:
    try:
        return tuple(sorted(os.scandir(path), key=lambda item: item.name))
    except OSError as exc:
        raise BulkTargetQEvidenceError(
            f"cannot enumerate evidence directory: {path}"
        ) from exc


def _exact_flat_files(
    directory: Path,
    expected_names: Sequence[str],
    *,
    inodes: dict[tuple[int, int], str],
    label: str,
) -> tuple[Path, ...]:
    expected = tuple(sorted(str(name) for name in expected_names))
    if len(set(expected)) != len(expected):
        raise BulkTargetQEvidenceError(
            f"{label} expected inventory contains duplicates"
        )
    entries = _directory_entries(directory)
    observed = tuple(item.name for item in entries)
    if observed != expected:
        missing = sorted(set(expected).difference(observed))
        extra = sorted(set(observed).difference(expected))
        raise BulkTargetQEvidenceError(
            f"{label} inventory is not exact; missing={missing}, extra={extra}"
        )
    result = []
    for entry in entries:
        if entry.is_symlink():
            raise BulkTargetQEvidenceError(
                f"{label} contains a symlink: {entry.path}"
            )
        path, _ = _audit_regular_file(
            Path(entry.path),
            inodes=inodes,
            label=f"{label}/{entry.name}",
        )
        result.append(path)
    return tuple(result)


def _safe_relative_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise BulkTargetQEvidenceError(f"{label} must be non-empty text")
    relative = Path(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "." in relative.parts
        or relative.as_posix() != value
    ):
        raise BulkTargetQEvidenceError(
            f"{label} is not a canonical traversal-safe relative path"
        )
    return relative


def _recursive_inventory(
    root: Path,
    *,
    inodes: dict[tuple[int, int], str],
    label: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    files: list[str] = []
    directories: list[str] = []

    def visit(directory: Path, prefix: Path) -> None:
        for entry in _directory_entries(directory):
            relative = prefix / entry.name
            if entry.is_symlink():
                raise BulkTargetQEvidenceError(
                    f"{label} contains a symlink: {entry.path}"
                )
            status = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(status.st_mode):
                directories.append(relative.as_posix())
                visit(Path(entry.path), relative)
            elif stat.S_ISREG(status.st_mode):
                path, _ = _audit_regular_file(
                    Path(entry.path),
                    inodes=inodes,
                    label=f"{label}/{relative.as_posix()}",
                )
                files.append(path.relative_to(root).as_posix())
            else:
                raise BulkTargetQEvidenceError(
                    f"{label} contains a special file: {entry.path}"
                )

    visit(root, Path())
    return tuple(sorted(files)), tuple(sorted(directories))


def _load_strict_json(
    path: Path,
    *,
    expect_envelope: bool,
) -> tuple[dict[str, Any], dict[str, Any], str]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
        document = json.loads(
            text,
            parse_constant=_reject_constant,
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise BulkTargetQEvidenceError(
            f"invalid strict JSON evidence: {path}"
        ) from exc
    if not isinstance(document, dict) or raw != _canonical_bytes(document):
        raise BulkTargetQEvidenceError(
            f"JSON evidence is not a canonical object: {path}"
        )
    if expect_envelope:
        if set(document) != {"payload", "payload_sha256"}:
            raise BulkTargetQEvidenceError(
                f"JSON envelope fields are not exact: {path}"
            )
        payload = document["payload"]
        payload_hash = document["payload_sha256"]
        if not isinstance(payload, dict) or not _is_sha256(payload_hash):
            raise BulkTargetQEvidenceError(
                f"JSON envelope is malformed: {path}"
            )
        observed = _sha256_bytes(_canonical_json(payload).encode("ascii"))
        if observed != payload_hash:
            raise BulkTargetQEvidenceError(
                f"JSON envelope payload hash failed: {path}"
            )
    else:
        payload = document
    return document, payload, _sha256_bytes(raw)


def _mapping_payload(value: Any, *, label: str) -> dict[str, Any]:
    if isinstance(value, MPIExecutionReport):
        return value._payload()  # noqa: SLF001
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a report or mapping")
    normalized = _json_value(value)
    assert isinstance(normalized, dict)
    if set(normalized) == {"payload", "payload_sha256"}:
        payload = normalized["payload"]
        expected = normalized["payload_sha256"]
        if (
            not isinstance(payload, dict)
            or not _is_sha256(expected)
            or _sha256_bytes(_canonical_json(payload).encode("ascii"))
            != expected
        ):
            raise BulkTargetQEvidenceError(f"{label} envelope is invalid")
        return payload
    return normalized


def _finite_nonnegative_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BulkTargetQEvidenceError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise BulkTargetQEvidenceError(
            f"{label} must be finite and nonnegative"
        )
    return result


def _positive_integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BulkTargetQEvidenceError(f"{label} must be a positive integer")
    return int(value)


def _parse_assignment(value: Any) -> TaskAssignment:
    expected_keys = {
        "task_id",
        "status",
        "rank",
        "hostname",
        "elapsed_seconds",
        "resumed",
        "cpu_affinity",
        "thread_cap_passed",
        "checkpoint_relative_path",
        "checkpoint_file_sha256",
        "artifact_relative_path",
        "artifact_sha256",
        "orphan_artifact_adopted",
        "cpu_affinity_topology",
        "origin_communicator_size",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise BulkTargetQEvidenceError(
            "execution assignment fields are not exact"
        )
    task_id = value["task_id"]
    status = value["status"]
    hostname = value["hostname"]
    if not all(
        isinstance(item, str) and item and item == item.strip()
        for item in (task_id, status, hostname)
    ):
        raise BulkTargetQEvidenceError("execution assignment text is invalid")
    rank = value["rank"]
    if isinstance(rank, bool) or not isinstance(rank, int) or rank < 0:
        raise BulkTargetQEvidenceError("execution assignment rank is invalid")
    communicator_size = _positive_integer(
        value["origin_communicator_size"],
        label="origin_communicator_size",
    )
    if rank >= communicator_size:
        raise BulkTargetQEvidenceError(
            "execution assignment rank is outside its origin communicator"
        )
    if type(value["resumed"]) is not bool:
        raise BulkTargetQEvidenceError("assignment resumed must be boolean")
    if type(value["thread_cap_passed"]) is not bool:
        raise BulkTargetQEvidenceError(
            "assignment thread_cap_passed must be boolean"
        )
    if type(value["orphan_artifact_adopted"]) is not bool:
        raise BulkTargetQEvidenceError(
            "assignment orphan_artifact_adopted must be boolean"
        )
    checkpoint_path = _safe_relative_path(
        value["checkpoint_relative_path"],
        label="checkpoint_relative_path",
    )
    if len(checkpoint_path.parts) != 1:
        raise BulkTargetQEvidenceError(
            "checkpoint_relative_path must be a basename"
        )
    checkpoint_hash = value["checkpoint_file_sha256"]
    if not _is_sha256(checkpoint_hash):
        raise BulkTargetQEvidenceError(
            "assignment checkpoint SHA256 is invalid"
        )
    artifact_path = value["artifact_relative_path"]
    artifact_hash = value["artifact_sha256"]
    if (artifact_path is None) != (artifact_hash is None):
        raise BulkTargetQEvidenceError(
            "assignment artifact path/hash pair is incomplete"
        )
    if artifact_path is not None:
        _safe_relative_path(artifact_path, label="artifact_relative_path")
        if not _is_sha256(artifact_hash):
            raise BulkTargetQEvidenceError(
                "assignment artifact SHA256 is invalid"
            )
    affinity_raw = value["cpu_affinity"]
    affinity: tuple[int, ...] | None
    if affinity_raw is None:
        affinity = None
    elif (
        not isinstance(affinity_raw, list)
        or not affinity_raw
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in affinity_raw
        )
        or affinity_raw != sorted(set(affinity_raw))
    ):
        raise BulkTargetQEvidenceError(
            "assignment CPU affinity is invalid"
        )
    else:
        affinity = tuple(int(item) for item in affinity_raw)
    topology_raw = value["cpu_affinity_topology"]
    topology: tuple[tuple[int, int, int], ...] | None
    if topology_raw is None:
        topology = None
    elif not isinstance(topology_raw, list) or not topology_raw:
        raise BulkTargetQEvidenceError(
            "assignment CPU-affinity topology is invalid"
        )
    else:
        topology_items: list[tuple[int, int, int]] = []
        for item in topology_raw:
            if (
                not isinstance(item, dict)
                or set(item) != {"logical_cpu", "package_id", "core_id"}
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, int)
                    or component < 0
                    for component in item.values()
                )
            ):
                raise BulkTargetQEvidenceError(
                    "assignment CPU-affinity topology is invalid"
                )
            topology_items.append(
                (
                    int(item["logical_cpu"]),
                    int(item["package_id"]),
                    int(item["core_id"]),
                )
            )
        topology = tuple(topology_items)
        if (
            affinity is None
            or tuple(item[0] for item in topology) != affinity
            or len(set(topology)) != len(topology)
        ):
            raise BulkTargetQEvidenceError(
                "assignment CPU topology conflicts with its affinity"
            )
    assignment = TaskAssignment(
        task_id=task_id,
        status=status,
        rank=int(rank),
        hostname=hostname,
        elapsed_seconds=_finite_nonnegative_float(
            value["elapsed_seconds"],
            label="assignment elapsed_seconds",
        ),
        resumed=value["resumed"],
        cpu_affinity=affinity,
        thread_cap_passed=value["thread_cap_passed"],
        checkpoint_relative_path=checkpoint_path.as_posix(),
        checkpoint_file_sha256=checkpoint_hash,
        artifact_relative_path=artifact_path,
        artifact_sha256=artifact_hash,
        orphan_artifact_adopted=value["orphan_artifact_adopted"],
        cpu_affinity_topology=topology,
        origin_communicator_size=communicator_size,
    )
    if assignment.as_dict() != value:
        raise BulkTargetQEvidenceError(
            "execution assignment does not round-trip canonically"
        )
    return assignment


def _parse_execution(
    payload: dict[str, Any],
    campaign: BulkTargetQCampaign,
) -> MPIExecutionReport:
    expected_keys = {
        "schema_name",
        "schema_version",
        "execution_id",
        "manifest_sha256",
        "mode",
        "communicator_size",
        "coordinator_hostname",
        "started_utc",
        "finished_utc",
        "elapsed_seconds",
        "thread_limit",
        "complete",
        "assignments",
        "missing_task_ids",
        "infrastructure_errors",
    }
    if set(payload) != expected_keys:
        raise BulkTargetQEvidenceError(
            "execution manifest fields are not exact"
        )
    if (
        payload["schema_name"] != _EXECUTION_SCHEMA
        or payload["schema_version"] != _EXECUTION_SCHEMA_VERSION
        or payload["manifest_sha256"]
        != campaign.campaign_manifest_sha256
    ):
        raise BulkTargetQEvidenceError(
            "execution manifest schema or campaign identity is wrong"
        )
    execution_id = payload["execution_id"]
    coordinator = payload["coordinator_hostname"]
    started = payload["started_utc"]
    finished = payload["finished_utc"]
    if not all(
        isinstance(item, str) and item and item == item.strip()
        for item in (execution_id, coordinator, started, finished)
    ):
        raise BulkTargetQEvidenceError(
            "execution identity or timestamps are invalid"
        )
    if "/" in execution_id or "\\" in execution_id:
        raise BulkTargetQEvidenceError("execution identifier is path-unsafe")
    mode = payload["mode"]
    if mode not in {"serial", "mpi"}:
        raise BulkTargetQEvidenceError("execution mode is invalid")
    communicator_size = _positive_integer(
        payload["communicator_size"],
        label="communicator_size",
    )
    if (mode == "serial") != (communicator_size == 1):
        raise BulkTargetQEvidenceError(
            "execution mode conflicts with communicator size"
        )
    thread_limit = _positive_integer(
        payload["thread_limit"],
        label="thread_limit",
    )
    raw_assignments = payload["assignments"]
    if not isinstance(raw_assignments, list):
        raise BulkTargetQEvidenceError(
            "execution assignments must be a list"
        )
    assignments = tuple(
        _parse_assignment(item) for item in raw_assignments
    )
    task_ids = [item.task_id for item in assignments]
    if len(set(task_ids)) != len(task_ids):
        raise BulkTargetQEvidenceError(
            "execution assignments contain duplicate tasks"
        )
    canonical_positions = {
        task_id: index
        for index, task_id in enumerate(campaign.canonical_task_ids)
    }
    if any(task_id not in canonical_positions for task_id in task_ids):
        raise BulkTargetQEvidenceError(
            "execution assignments name tasks outside the campaign"
        )
    if task_ids != sorted(task_ids, key=canonical_positions.__getitem__):
        raise BulkTargetQEvidenceError(
            "execution assignments are not in canonical task order"
        )
    missing = payload["missing_task_ids"]
    errors = payload["infrastructure_errors"]
    if (
        not isinstance(missing, list)
        or any(
            not isinstance(task_id, str)
            or task_id not in canonical_positions
            for task_id in missing
        )
        or len(set(missing)) != len(missing)
        or not isinstance(errors, list)
        or any(not isinstance(item, dict) for item in errors)
    ):
        raise BulkTargetQEvidenceError(
            "execution missing-task or infrastructure evidence is invalid"
        )
    report = MPIExecutionReport(
        execution_id=execution_id,
        manifest_sha256=campaign.campaign_manifest_sha256,
        mode=mode,
        communicator_size=communicator_size,
        coordinator_hostname=coordinator,
        started_utc=started,
        finished_utc=finished,
        elapsed_seconds=_finite_nonnegative_float(
            payload["elapsed_seconds"],
            label="execution elapsed_seconds",
        ),
        thread_limit=thread_limit,
        assignments=assignments,
        missing_task_ids=tuple(missing),
        infrastructure_errors=tuple(errors),
    )
    if type(payload["complete"]) is not bool:
        raise BulkTargetQEvidenceError("execution complete must be boolean")
    if report.complete is not payload["complete"]:
        raise BulkTargetQEvidenceError(
            "execution complete flag conflicts with terminal state"
        )
    if report._payload() != payload:  # noqa: SLF001
        raise BulkTargetQEvidenceError(
            "execution manifest does not round-trip canonically"
        )
    return report


def _merge_report_from_mapping(value: Any) -> BulkTargetQMergeReport:
    if isinstance(value, BulkTargetQMergeReport):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("declared merge report must be a report or mapping")
    raw = _json_value(value)
    assert isinstance(raw, dict)
    expected_keys = {
        "schema",
        "schema_version",
        "campaign_manifest_sha256",
        "merged_artifact_path",
        "merged_artifact_sha256",
        "task_ids",
        "target_grid_points",
        "task_artifact_sha256",
        "task_checkpoint_sha256",
        "dataset_sha256",
        "report_sha256",
    }
    if set(raw) != expected_keys:
        raise BulkTargetQEvidenceError(
            "declared merge report fields are not exact"
        )
    if (
        raw["schema"] != _MERGE_REPORT_SCHEMA
        or raw["schema_version"] != _MERGE_REPORT_SCHEMA_VERSION
    ):
        raise BulkTargetQEvidenceError(
            "declared merge report schema is invalid"
        )
    try:
        report = BulkTargetQMergeReport(
            campaign_manifest_sha256=raw["campaign_manifest_sha256"],
            merged_artifact_path=raw["merged_artifact_path"],
            merged_artifact_sha256=raw["merged_artifact_sha256"],
            task_ids=tuple(raw["task_ids"]),
            target_grid_points=tuple(raw["target_grid_points"]),
            task_artifact_sha256=tuple(raw["task_artifact_sha256"]),
            task_checkpoint_sha256=tuple(
                raw["task_checkpoint_sha256"]
            ),
            dataset_sha256=raw["dataset_sha256"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkTargetQEvidenceError(
            "declared merge report is invalid"
        ) from exc
    if report.as_dict() != raw:
        raise BulkTargetQEvidenceError(
            "declared merge report hash or canonical fields are invalid"
        )
    return report


def _checkpoint_artifact_relative_path(path: Path) -> Path:
    _, payload, _ = _load_strict_json(path, expect_envelope=True)
    artifact = payload.get("artifact")
    if not isinstance(artifact, dict) or set(artifact) != {
        "relative_path",
        "sha256",
    }:
        raise BulkTargetQEvidenceError(
            f"completed checkpoint lacks an exact artifact record: {path}"
        )
    if not _is_sha256(artifact["sha256"]):
        raise BulkTargetQEvidenceError(
            f"checkpoint artifact SHA256 is invalid: {path}"
        )
    return _safe_relative_path(
        artifact["relative_path"],
        label=f"artifact path in {path.name}",
    )


def _checkpoint_inventory(
    campaign: BulkTargetQCampaign,
    root: Path,
    *,
    inodes: dict[tuple[int, int], str],
) -> tuple[tuple[TaskCheckpoint, ...], tuple[dict[str, Any], ...]]:
    checkpoint_root = _audit_directory(root / "task-outcomes")
    expected_names = tuple(
        task_checkpoint_path(
            campaign.manifest,
            task,
            checkpoint_root,
        ).name
        for task in campaign.manifest.tasks
    )
    checkpoint_paths = _exact_flat_files(
        checkpoint_root,
        expected_names,
        inodes=inodes,
        label="task-outcomes",
    )
    artifact_relatives = tuple(
        _checkpoint_artifact_relative_path(path)
        for path in checkpoint_paths
    )
    if len(set(artifact_relatives)) != len(artifact_relatives):
        raise BulkTargetQEvidenceError(
            "multiple checkpoints claim the same artifact path"
        )
    artifact_root = _audit_directory(root / "task-artifacts")
    observed_files, observed_directories = _recursive_inventory(
        artifact_root,
        inodes=inodes,
        label="task-artifacts",
    )
    expected_files = tuple(
        sorted(path.as_posix() for path in artifact_relatives)
    )
    expected_directories = tuple(
        sorted(
            {
                parent.as_posix()
                for relative in artifact_relatives
                for parent in relative.parents
                if parent != Path(".")
            }
        )
    )
    if (
        observed_files != expected_files
        or observed_directories != expected_directories
    ):
        raise BulkTargetQEvidenceError(
            "task-artifacts inventory is not exact; "
            f"expected_files={list(expected_files)}, "
            f"observed_files={list(observed_files)}, "
            f"expected_directories={list(expected_directories)}, "
            f"observed_directories={list(observed_directories)}"
        )
    artifact_hdf5_paths = {
        relative.as_posix(): _audit_hdf5_storage(artifact_root / relative)
        for relative in artifact_relatives
    }
    checkpoints = collect_task_checkpoints(
        campaign.manifest,
        checkpoint_root,
        artifact_root,
        artifact_validator=make_bulk_target_q_artifact_validator(campaign),
        require_complete=True,
    )
    records = []
    for task, checkpoint in zip(
        campaign.manifest.tasks,
        checkpoints,
        strict=True,
    ):
        if checkpoint.task_id != task.task_id or checkpoint.status != "completed":
            raise BulkTargetQEvidenceError(
                "release checkpoint order or status is invalid"
            )
        checkpoint_path = task_checkpoint_path(
            campaign.manifest,
            task,
            checkpoint_root,
        )
        assert checkpoint.artifact_relative_path is not None
        assert checkpoint.artifact_sha256 is not None
        artifact_path = artifact_root / checkpoint.artifact_relative_path
        hdf5_object_paths = artifact_hdf5_paths[
            Path(checkpoint.artifact_relative_path).as_posix()
        ]
        actual_checkpoint_hash = _file_sha256(checkpoint_path)
        actual_artifact_hash = _file_sha256(artifact_path)
        if actual_artifact_hash != checkpoint.artifact_sha256:
            raise BulkTargetQEvidenceError(
                f"artifact hash changed for task {task.task_id!r}"
            )
        records.append(
            {
                "task_id": task.task_id,
                "task_identity_sha256": checkpoint.task_identity_sha256,
                "checkpoint_relative_path": (
                    checkpoint_path.relative_to(root).as_posix()
                ),
                "checkpoint_file_sha256": actual_checkpoint_hash,
                "checkpoint_payload_sha256": checkpoint.checkpoint_sha256,
                "artifact_relative_path": (
                    artifact_path.relative_to(root).as_posix()
                ),
                "artifact_sha256": actual_artifact_hash,
                "hdf5_storage": {
                    "hard_links_only": True,
                    "virtual_datasets": False,
                    "external_storage": False,
                    "object_paths": list(hdf5_object_paths),
                },
                "artifact_validation": _json_value(
                    checkpoint.artifact_validation
                ),
            }
        )
    return checkpoints, tuple(records)


def _checkpoint_assignment_identity(
    checkpoint: TaskCheckpoint,
    *,
    checkpoint_relative_path: str,
    checkpoint_file_sha256: str,
) -> dict[str, Any]:
    execution = checkpoint.execution
    try:
        thread_cap = execution["thread_cap"]
        affinity_raw = thread_cap["cpu_affinity"]
        topology_raw = thread_cap["cpu_affinity_topology"]
        topology = (
            None
            if topology_raw is None
            else [
                {
                    "logical_cpu": int(item["logical_cpu"]),
                    "package_id": int(item["package_id"]),
                    "core_id": int(item["core_id"]),
                }
                for item in topology_raw
            ]
        )
        return {
            "task_id": checkpoint.task_id,
            "status": checkpoint.status,
            "rank": int(execution["rank"]),
            "hostname": str(execution["hostname"]),
            "elapsed_seconds": float(execution["elapsed_seconds"]),
            "cpu_affinity": (
                None
                if affinity_raw is None
                else [int(item) for item in affinity_raw]
            ),
            "thread_cap_passed": bool(
                thread_cap["all_loaded_libraries_capped"]
            ),
            "checkpoint_relative_path": checkpoint_relative_path,
            "checkpoint_file_sha256": checkpoint_file_sha256,
            "artifact_relative_path": checkpoint.artifact_relative_path,
            "artifact_sha256": checkpoint.artifact_sha256,
            "orphan_artifact_adopted": bool(
                execution.get("orphan_artifact_adopted", False)
            ),
            "cpu_affinity_topology": topology,
            "origin_communicator_size": int(
                execution["communicator_size"]
            ),
            "thread_limit": int(execution["thread_limit"]),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise BulkTargetQEvidenceError(
            f"checkpoint execution metadata is incomplete: "
            f"{checkpoint.task_id}"
        ) from exc


def _cross_check_assignment(
    assignment: TaskAssignment,
    report: MPIExecutionReport,
    checkpoint: TaskCheckpoint,
    record: Mapping[str, Any],
) -> None:
    expected = _checkpoint_assignment_identity(
        checkpoint,
        checkpoint_relative_path=Path(
            str(record["checkpoint_relative_path"])
        ).name,
        checkpoint_file_sha256=str(
            record["checkpoint_file_sha256"]
        ),
    )
    observed = assignment.as_dict()
    for name in (
        "task_id",
        "status",
        "rank",
        "hostname",
        "elapsed_seconds",
        "cpu_affinity",
        "thread_cap_passed",
        "checkpoint_relative_path",
        "checkpoint_file_sha256",
        "artifact_relative_path",
        "artifact_sha256",
        "orphan_artifact_adopted",
        "cpu_affinity_topology",
        "origin_communicator_size",
    ):
        if observed[name] != expected[name]:
            raise BulkTargetQEvidenceError(
                f"execution assignment field {name!r} does not match "
                f"checkpoint {checkpoint.task_id!r}"
            )
    if expected["thread_limit"] != report.thread_limit:
        raise BulkTargetQEvidenceError(
            f"execution thread limit does not match checkpoint "
            f"{checkpoint.task_id!r}"
        )
    if assignment.thread_cap_passed is not True:
        raise BulkTargetQEvidenceError(
            f"thread cap did not pass for task {checkpoint.task_id!r}"
        )
    worker_environment = checkpoint.execution["thread_cap"]["environment"]
    expected_environment = {
        name: str(report.thread_limit) for name in _THREAD_ENVIRONMENT
    }
    if dict(worker_environment) != expected_environment:
        raise BulkTargetQEvidenceError(
            f"worker thread environment differs from release policy for "
            f"task {checkpoint.task_id!r}"
        )


def _execution_inventory(
    campaign: BulkTargetQCampaign,
    root: Path,
    selected_path: Path,
    selected_report: MPIExecutionReport | Mapping[str, Any] | None,
    checkpoints: Sequence[TaskCheckpoint],
    checkpoint_records: Sequence[Mapping[str, Any]],
    *,
    inodes: dict[tuple[int, int], str],
) -> tuple[
    MPIExecutionReport,
    tuple[MPIExecutionReport, ...],
    tuple[dict[str, Any], ...],
    str,
]:
    execution_root = _audit_directory(root / "executions")
    selected = _absolute_unresolved(selected_path)
    if selected.parent != execution_root:
        raise BulkTargetQEvidenceError(
            "selected execution must be directly inside run/executions"
        )
    entries = _directory_entries(execution_root)
    if not entries:
        raise BulkTargetQEvidenceError("execution history is empty")
    observed_names = tuple(item.name for item in entries)
    if any(
        not name.endswith(".json")
        or "/" in name
        or "\\" in name
        for name in observed_names
    ):
        raise BulkTargetQEvidenceError(
            "execution directory contains a non-manifest entry"
        )
    execution_paths = _exact_flat_files(
        execution_root,
        observed_names,
        inodes=inodes,
        label="executions",
    )
    if selected not in execution_paths:
        raise BulkTargetQEvidenceError(
            "selected execution is absent from exact execution inventory"
        )
    history: list[MPIExecutionReport] = []
    records: list[dict[str, Any]] = []
    by_id: dict[str, MPIExecutionReport] = {}
    selected_file_hash = ""
    selected_from_file: MPIExecutionReport | None = None
    by_checkpoint = {
        checkpoint.task_id: (checkpoint, record)
        for checkpoint, record in zip(
            checkpoints,
            checkpoint_records,
            strict=True,
        )
    }
    for path in execution_paths:
        _, payload, file_hash = _load_strict_json(
            path,
            expect_envelope=True,
        )
        report = _parse_execution(payload, campaign)
        if path.name != f"{report.execution_id}.json":
            raise BulkTargetQEvidenceError(
                f"execution filename does not match its identifier: {path}"
            )
        if report.execution_id in by_id:
            raise BulkTargetQEvidenceError(
                "execution history contains duplicate identifiers"
            )
        for assignment in report.assignments:
            checkpoint_and_record = by_checkpoint.get(assignment.task_id)
            if checkpoint_and_record is None:
                raise BulkTargetQEvidenceError(
                    "execution assignment has no current checkpoint"
                )
            _cross_check_assignment(
                assignment,
                report,
                *checkpoint_and_record,
            )
        by_id[report.execution_id] = report
        history.append(report)
        records.append(
            {
                "execution_id": report.execution_id,
                "relative_path": path.relative_to(root).as_posix(),
                "file_sha256": file_hash,
                "payload_sha256": report.execution_sha256,
                "mode": report.mode,
                "communicator_size": report.communicator_size,
                "complete": report.complete,
                "task_ids": [
                    assignment.task_id
                    for assignment in report.assignments
                ],
            }
        )
        if path == selected:
            selected_from_file = report
            selected_file_hash = file_hash
    assert selected_from_file is not None
    if selected_report is not None:
        supplied_payload = _mapping_payload(
            selected_report,
            label="selected execution report",
        )
        supplied = _parse_execution(supplied_payload, campaign)
        if supplied._payload() != selected_from_file._payload():  # noqa: SLF001
            raise BulkTargetQEvidenceError(
                "selected execution report differs from its archived file"
            )
    expected_selected_ids = list(campaign.canonical_task_ids)
    if (
        not selected_from_file.complete
        or selected_from_file.mode != "mpi"
        or selected_from_file.communicator_size < 2
        or [
            assignment.task_id
            for assignment in selected_from_file.assignments
        ]
        != expected_selected_ids
        or any(
            assignment.status != "completed"
            for assignment in selected_from_file.assignments
        )
    ):
        raise BulkTargetQEvidenceError(
            "selected execution is not a complete canonical MPI campaign"
        )
    return (
        selected_from_file,
        tuple(history),
        tuple(records),
        selected_file_hash,
    )


def _require_h5py() -> Any:
    try:
        h5py = importlib.import_module("h5py")
    except ImportError as exc:
        raise BulkTargetQEvidenceError(
            "h5py is required to verify target-q release evidence"
        ) from exc
    return h5py


def _hdf5_object_paths(handle: Any) -> tuple[str, ...]:
    paths: list[str] = []

    def visit(group: Any, prefix: str) -> None:
        for name in sorted(group.keys()):
            path = f"{prefix}/{name}" if prefix else name
            link = group.get(name, getlink=True)
            if type(link).__name__ != "HardLink":
                raise BulkTargetQEvidenceError(
                    f"HDF5 non-hard link is forbidden: {path}"
                )
            item = group[name]
            paths.append(path)
            if hasattr(item, "keys"):
                visit(item, path)

    visit(handle, "")
    return tuple(paths)


def _audit_hdf5_storage(path: Path) -> tuple[str, ...]:
    h5py = _require_h5py()
    paths: tuple[str, ...]
    try:
        with h5py.File(path, "r") as handle:
            paths = _hdf5_object_paths(handle)
            addresses: dict[int, str] = {}
            for object_path in paths:
                item = handle[object_path]
                address = int(h5py.h5o.get_info(item.id).addr)
                owner = addresses.get(address)
                if owner is not None:
                    raise BulkTargetQEvidenceError(
                        f"HDF5 object {object_path!r} is a duplicate hard "
                        f"link to {owner!r}"
                    )
                addresses[address] = object_path
                if isinstance(item, h5py.Dataset):
                    if bool(item.is_virtual):
                        raise BulkTargetQEvidenceError(
                            f"virtual HDF5 dataset is forbidden: {object_path}"
                        )
                    creation = item.id.get_create_plist()
                    if creation.get_external_count() != 0:
                        raise BulkTargetQEvidenceError(
                            f"external HDF5 storage is forbidden: {object_path}"
                        )
                    np.asarray(item)
    except BulkTargetQEvidenceError:
        raise
    except Exception as exc:
        raise BulkTargetQEvidenceError(
            f"cannot fully read HDF5 evidence: {path}"
        ) from exc
    return paths


def _attribute_identity(value: Any) -> tuple[Any, ...]:
    array = np.asarray(value)
    if array.dtype.kind == "O":
        return ("object", _canonical_json(_json_value(array.tolist())))
    contiguous = np.ascontiguousarray(array)
    return (
        array.dtype.str,
        tuple(array.shape),
        memoryview(contiguous.reshape(-1)).cast("B").tobytes().hex(),
    )


def _compare_hdf5(
    expected_path: Path,
    actual_path: Path,
) -> dict[str, Any]:
    h5py = _require_h5py()
    expected_objects = _audit_hdf5_storage(expected_path)
    actual_objects = _audit_hdf5_storage(actual_path)
    if expected_objects != actual_objects:
        raise BulkTargetQEvidenceError(
            "supplied and independently merged HDF5 object trees differ"
        )
    try:
        with (
            h5py.File(expected_path, "r") as expected,
            h5py.File(actual_path, "r") as actual,
        ):
            for object_path in ("", *expected_objects):
                expected_item = (
                    expected if not object_path else expected[object_path]
                )
                actual_item = actual if not object_path else actual[object_path]
                expected_attributes = tuple(sorted(expected_item.attrs.keys()))
                actual_attributes = tuple(sorted(actual_item.attrs.keys()))
                if expected_attributes != actual_attributes:
                    raise BulkTargetQEvidenceError(
                        f"HDF5 attributes differ at {object_path or '/'}"
                    )
                for name in expected_attributes:
                    if _attribute_identity(
                        expected_item.attrs[name]
                    ) != _attribute_identity(actual_item.attrs[name]):
                        raise BulkTargetQEvidenceError(
                            f"HDF5 attribute {name!r} differs at "
                            f"{object_path or '/'}"
                        )
                expected_is_dataset = isinstance(
                    expected_item,
                    h5py.Dataset,
                )
                actual_is_dataset = isinstance(actual_item, h5py.Dataset)
                if expected_is_dataset != actual_is_dataset:
                    raise BulkTargetQEvidenceError(
                        f"HDF5 object type differs at {object_path or '/'}"
                    )
                if not expected_is_dataset:
                    continue
                if (
                    expected_item.shape != actual_item.shape
                    or expected_item.dtype != actual_item.dtype
                ):
                    raise BulkTargetQEvidenceError(
                        f"HDF5 dataset shape/dtype differs at {object_path}"
                    )
                if expected_item.ndim == 0:
                    selectors = ((),)
                else:
                    selectors = tuple(
                        (slice(index, index + 1),)
                        + (slice(None),) * (expected_item.ndim - 1)
                        for index in range(expected_item.shape[0])
                    )
                for selector in selectors:
                    left = np.asarray(expected_item[selector])
                    right = np.asarray(actual_item[selector])
                    if not np.array_equal(left, right, equal_nan=True):
                        raise BulkTargetQEvidenceError(
                            f"HDF5 dataset values differ at {object_path}"
                        )
    except BulkTargetQEvidenceError:
        raise
    except Exception as exc:
        raise BulkTargetQEvidenceError(
            "cannot compare supplied and independently merged HDF5 files"
        ) from exc
    return {
        "object_paths": list(expected_objects),
        "object_count": len(expected_objects),
        "all_object_types_equal": True,
        "all_attributes_equal": True,
        "all_dataset_shapes_equal": True,
        "all_dataset_dtypes_equal": True,
        "all_dataset_values_equal": True,
    }


def _scientific_result_identity(result: Any) -> str:
    values = {
        "temperatures": result.temperatures,
        "grid_points": result.grid_points,
        "band_indices": result.band_indices,
        "gamma_thz_hwhm": result.gamma_thz_hwhm,
        "gamma_normal_thz_hwhm": result.gamma_normal_thz_hwhm,
        "gamma_umklapp_thz_hwhm": result.gamma_umklapp_thz_hwhm,
        "gamma_angular_fwhm": result.gamma_angular_fwhm,
        "decay_phase_space": result.decay_phase_space,
        "difference_phase_space": result.difference_phase_space,
        "interaction/grid_points": result.interactions.grid_points,
        "interaction/band_indices": result.interactions.band_indices,
        "interaction/q1_indices": result.interactions.q1_indices,
        "interaction/q2_indices": result.interactions.q2_indices,
        "interaction/reciprocal_vectors": (
            result.interactions.reciprocal_vectors
        ),
        "interaction/normal_mask": result.interactions.normal_mask,
        "interaction/matrix_elements": result.interactions.matrix_elements,
        "interaction/interaction_strengths": (
            result.interactions.interaction_strengths
        ),
        "phonons/qpoints": result.interactions.phonons.qpoints,
        "phonons/dynamical_matrices": (
            result.interactions.phonons.dynamical_matrices
        ),
        "phonons/frequencies": result.interactions.phonons.frequencies,
        "phonons/eigenvectors": result.interactions.phonons.eigenvectors,
    }
    digest = hashlib.sha256()
    for name in sorted(values):
        array = np.ascontiguousarray(np.asarray(values[name]))
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(b"\0")
        digest.update(_canonical_json(list(array.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(memoryview(array.reshape(-1)).cast("B"))
        digest.update(b"\0")
    scalar_metadata = {
        "frequency_factor_to_thz": (
            result.interactions.phonons.frequency_factor_to_thz
        ),
        "hermitianized": result.interactions.phonons.hermitianized,
        "linewidths_metadata": result.metadata,
        "interactions_metadata": result.interactions.metadata,
    }
    digest.update(_canonical_json(scalar_metadata).encode("ascii"))
    return digest.hexdigest()


def _merge_report_identity_fields(
    report: BulkTargetQMergeReport,
) -> dict[str, Any]:
    return {
        "campaign_manifest_sha256": report.campaign_manifest_sha256,
        "task_ids": list(report.task_ids),
        "target_grid_points": list(report.target_grid_points),
        "task_artifact_sha256": list(report.task_artifact_sha256),
        "task_checkpoint_sha256": list(report.task_checkpoint_sha256),
        "dataset_sha256": dict(report.dataset_sha256),
    }


def _disk_preflight(path: Path, merged_size: int) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    margin = max(_MINIMUM_DISK_MARGIN_BYTES, merged_size // 10)
    required = merged_size + margin
    if usage.free < required:
        raise BulkTargetQEvidenceError(
            "insufficient free disk for independent temporary canonical merge: "
            f"required={required}, free={usage.free}"
        )
    return {
        "filesystem_path": path.as_posix(),
        "supplied_merged_size_bytes": merged_size,
        "minimum_margin_bytes": margin,
        "required_free_bytes": required,
        "observed_free_bytes": int(usage.free),
        "passed": True,
    }


def verify_bulk_target_q_release_evidence(
    campaign: BulkTargetQCampaign,
    run_directory: str | Path,
    selected_execution_path: str | Path,
    declared_merge_report_path: str | Path,
    merged_hdf5_path: str | Path,
    *,
    selected_execution_report: (
        MPIExecutionReport | Mapping[str, Any] | None
    ) = None,
    temporary_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Verify and independently rebuild one complete target-q MPI campaign.

    The returned mapping is canonical-JSON serializable.  Temporary merge
    paths are intentionally omitted, and the temporary directory is removed
    before this function returns or raises.
    """

    if not isinstance(campaign, BulkTargetQCampaign):
        raise TypeError("campaign must be BulkTargetQCampaign")
    root = _audit_directory(_absolute_unresolved(run_directory))
    inodes: dict[tuple[int, int], str] = {}

    manifest_path, _ = _audit_regular_file(
        root / "task-manifest.json",
        inodes=inodes,
        label="task-manifest.json",
    )
    manifest_document, _, manifest_file_hash = _load_strict_json(
        manifest_path,
        expect_envelope=True,
    )
    if manifest_document != campaign.manifest.as_dict():
        raise BulkTargetQEvidenceError(
            "archived task manifest differs from the rebuilt campaign"
        )

    checkpoints, checkpoint_records = _checkpoint_inventory(
        campaign,
        root,
        inodes=inodes,
    )
    selected_path = _absolute_unresolved(selected_execution_path)
    (
        selected_execution,
        execution_history,
        execution_records,
        selected_execution_file_hash,
    ) = _execution_inventory(
        campaign,
        root,
        selected_path,
        selected_execution_report,
        checkpoints,
        checkpoint_records,
        inodes=inodes,
    )
    try:
        lineage = task_execution_mpi_origin_lineage_evidence(
            selected_execution,
            execution_history,
            require_mpi_origin=True,
        )
        origin_execution_ids = {
            str(record["task_id"]): str(
                record["origin_execution_id"]
            )
            for record in lineage["lineage"]
        }
        core_bindings = task_assignment_physical_core_evidence(
            selected_execution.assignments,
            require_available=True,
            origin_execution_ids=origin_execution_ids,
        )
    except TaskManifestError as exc:
        raise BulkTargetQEvidenceError(
            "MPI origin or physical-core evidence failed"
        ) from exc

    merge_report_path, _ = _audit_regular_file(
        _absolute_unresolved(declared_merge_report_path),
        inodes=inodes,
        label="declared merge report",
    )
    _, merge_payload, merge_report_file_hash = _load_strict_json(
        merge_report_path,
        expect_envelope=False,
    )
    declared_merge = _merge_report_from_mapping(merge_payload)
    supplied_merged, supplied_status = _audit_regular_file(
        _absolute_unresolved(merged_hdf5_path),
        inodes=inodes,
        label="merged HDF5 artifact",
    )
    merged_relative_path = _relative_inside(
        supplied_merged,
        root,
        label="merged HDF5 artifact",
    )
    if _absolute_unresolved(declared_merge.merged_artifact_path) != (
        supplied_merged
    ):
        raise BulkTargetQEvidenceError(
            "declared merge path does not name the supplied merged HDF5 file"
        )
    supplied_merged_hash = _file_sha256(supplied_merged)
    if declared_merge.merged_artifact_sha256 != supplied_merged_hash:
        raise BulkTargetQEvidenceError(
            "declared merge SHA256 does not match the supplied HDF5 file"
        )
    if (
        declared_merge.campaign_manifest_sha256
        != campaign.campaign_manifest_sha256
        or tuple(declared_merge.task_ids) != campaign.canonical_task_ids
        or tuple(declared_merge.target_grid_points)
        != tuple(int(value) for value in campaign.controls.grid_points)
    ):
        raise BulkTargetQEvidenceError(
            "declared merge report belongs to another campaign"
        )
    supplied_semantic = _merged_artifact_evidence(
        supplied_merged,
        campaign,
    )
    _audit_hdf5_storage(supplied_merged)

    temp_parent = (
        supplied_merged.parent
        if temporary_directory is None
        else _audit_directory(_absolute_unresolved(temporary_directory))
    )
    disk_evidence = _disk_preflight(
        temp_parent,
        int(supplied_status.st_size),
    )
    temporary_prefix = ".negfpy-target-q-remerge-"
    with tempfile.TemporaryDirectory(
        prefix=temporary_prefix,
        dir=temp_parent,
    ) as temporary_name:
        temporary_root = Path(temporary_name)
        rebuilt_path = temporary_root / "canonical-bulk-target-q.h5"
        rebuilt_report = merge_bulk_target_q_campaign(
            campaign,
            root,
            output_path=rebuilt_path,
        )
        rebuilt_semantic = _merged_artifact_evidence(
            rebuilt_path,
            campaign,
        )
        hdf5_comparison = _compare_hdf5(
            rebuilt_path,
            supplied_merged,
        )
        rebuilt_result = load_bulk_target_q_result(
            rebuilt_path,
            campaign=campaign,
        )
        supplied_result = load_bulk_target_q_result(
            supplied_merged,
            campaign=campaign,
        )
        rebuilt_result_hash = _scientific_result_identity(rebuilt_result)
        supplied_result_hash = _scientific_result_identity(supplied_result)
        if rebuilt_result_hash != supplied_result_hash:
            raise BulkTargetQEvidenceError(
                "independently rebuilt and supplied scientific results differ"
            )
        identity_fields = _merge_report_identity_fields(rebuilt_report)
        if _merge_report_identity_fields(declared_merge) != identity_fields:
            raise BulkTargetQEvidenceError(
                "declared merge report differs from independent canonical "
                "task/checkpoint/dataset identities"
            )
        if (
            dict(rebuilt_semantic) != dict(supplied_semantic)
            or tuple(rebuilt_report.task_artifact_sha256)
            != tuple(
                record["artifact_sha256"]
                for record in checkpoint_records
            )
            or tuple(rebuilt_report.task_checkpoint_sha256)
            != tuple(
                record["checkpoint_file_sha256"]
                for record in checkpoint_records
            )
        ):
            raise BulkTargetQEvidenceError(
                "independent canonical merge identity does not match the "
                "validated filesystem inventory"
            )

    leftovers = tuple(
        entry.name
        for entry in _directory_entries(temp_parent)
        if entry.name.startswith(temporary_prefix)
    )
    if leftovers:
        raise BulkTargetQEvidenceError(
            "temporary canonical merge directory was not cleaned up"
        )

    evidence: dict[str, Any] = {
        "schema": _SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "passed": True,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "manifest": {
            "relative_path": "task-manifest.json",
            "file_sha256": manifest_file_hash,
            "payload_sha256": campaign.manifest.manifest_sha256,
        },
        "selected_execution": {
            "execution_id": selected_execution.execution_id,
            "relative_path": _relative_inside(
                selected_path,
                root,
                label="selected execution",
            ),
            "file_sha256": selected_execution_file_hash,
            "payload_sha256": selected_execution.execution_sha256,
            "mode": selected_execution.mode,
            "communicator_size": selected_execution.communicator_size,
            "task_count": len(selected_execution.assignments),
            "complete": selected_execution.complete,
        },
        "execution_history": list(execution_records),
        "mpi_origin_lineage": _json_value(lineage),
        "physical_core_bindings": _json_value(core_bindings),
        "task_inventory": list(checkpoint_records),
        "merge_report": {
            "relative_path": _relative_inside(
                merge_report_path,
                root,
                label="declared merge report",
            ),
            "file_sha256": merge_report_file_hash,
            "report_sha256": declared_merge.report_sha256,
        },
        "merged_artifact": {
            "relative_path": merged_relative_path,
            "file_sha256": supplied_merged_hash,
            "size_bytes": int(supplied_status.st_size),
            "dataset_sha256": dict(declared_merge.dataset_sha256),
            "task_artifact_sha256": list(
                declared_merge.task_artifact_sha256
            ),
            "task_checkpoint_sha256": list(
                declared_merge.task_checkpoint_sha256
            ),
            "scientific_result_sha256": supplied_result_hash,
            "floating_point_mpi_reductions": False,
        },
        "independent_canonical_remerge": {
            "passed": True,
            "campaign_manifest_sha256": (
                rebuilt_report.campaign_manifest_sha256
            ),
            "task_ids": list(rebuilt_report.task_ids),
            "target_grid_points": list(
                rebuilt_report.target_grid_points
            ),
            "task_artifact_sha256": list(
                rebuilt_report.task_artifact_sha256
            ),
            "task_checkpoint_sha256": list(
                rebuilt_report.task_checkpoint_sha256
            ),
            "dataset_sha256": dict(rebuilt_report.dataset_sha256),
            "scientific_result_sha256": rebuilt_result_hash,
            "hdf5_exact_comparison": hdf5_comparison,
            "temporary_artifact_retained": False,
        },
        "disk_preflight": disk_evidence,
    }
    evidence["evidence_sha256"] = _sha256_bytes(
        _canonical_json(evidence).encode("ascii")
    )
    return _json_value(evidence)


__all__ = [
    "BulkTargetQEvidenceError",
    "verify_bulk_target_q_release_evidence",
]
