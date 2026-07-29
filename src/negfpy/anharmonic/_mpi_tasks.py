"""Private, task-level MPI orchestration for reproducible heavy campaigns.

This module deliberately sits *above* the numerical solvers.  A task is an
independent validation case (and, later, a complete target-q calculation), so
MPI never partitions a Dyson solve, a cubic convolution, or an SCBA fixed
point.  MPI messages contain task identifiers and terminal status only; task
artifacts are published atomically on the shared filesystem.

``mpi4py`` is imported lazily.  Importing NEGFPy, constructing manifests, and
running the serial fallback do not require an MPI installation.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import socket
import time
from types import MappingProxyType
from typing import Any, Literal
from uuid import uuid4

from threadpoolctl import threadpool_info, threadpool_limits


_MANIFEST_SCHEMA = "negfpy_task_manifest"
_CHECKPOINT_SCHEMA = "negfpy_task_checkpoint"
_EXECUTION_SCHEMA = "negfpy_mpi_execution_manifest"
_SCHEMA_VERSION = 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SUFFIX_PATTERN = re.compile(r"^\.[A-Za-z0-9][A-Za-z0-9._-]*$")
_TERMINAL_STATUSES = frozenset(
    {"completed", "scientific_failure", "blocked"}
)
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_TASK_TAG = 4101
_RESULT_TAG = 4102
_STOP_TAG = 4103

ArtifactValidator = Callable[[Path, "TaskSpec"], Mapping[str, Any] | None]
TaskWorker = Callable[["TaskSpec", "TaskContext"], "TaskResult"]
OrphanArtifactAdopter = Callable[
    [Path, "TaskSpec", Mapping[str, Any]], "TaskResult"
]


class TaskMPIError(RuntimeError):
    """Base class for task-manifest, checkpoint, and MPI failures."""


class MPIDependencyError(TaskMPIError):
    """Raised when an explicitly requested MPI run lacks ``mpi4py``."""


class TaskManifestError(TaskMPIError):
    """Raised when a task manifest is invalid, corrupt, or mismatched."""


class TaskCheckpointError(TaskMPIError):
    """Base class for task-outcome checkpoint failures."""


class TaskCheckpointMismatchError(TaskCheckpointError):
    """Raised when an outcome belongs to another manifest or task."""


class TaskCheckpointCorruptionError(TaskCheckpointError):
    """Raised when an outcome or its artifact fails integrity checks."""


class DuplicateTaskCheckpointError(TaskCheckpointError):
    """Raised when more than one outcome claims the same task or artifact."""


class MissingTaskCheckpointError(TaskCheckpointError):
    """Raised when a complete merge is requested with missing outcomes."""

    def __init__(self, missing_task_ids: Sequence[str]) -> None:
        self.missing_task_ids = tuple(str(value) for value in missing_task_ids)
        super().__init__(
            "missing terminal task checkpoints: "
            + ", ".join(repr(value) for value in self.missing_task_ids)
        )


class TaskExecutionError(TaskMPIError):
    """Raised on infrastructure failure, carrying the partial run report."""

    def __init__(
        self,
        message: str,
        *,
        report: "MPIExecutionReport | None" = None,
    ) -> None:
        self.report = report
        super().__init__(message)


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def _validate_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if value != value.strip() or any(ord(character) < 32 for character in value):
        raise ValueError(
            f"{name} cannot contain surrounding whitespace or controls."
        )
    return value


def _normalize_json(value: Any, *, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{path} must not contain NaN or infinity.")
        return number
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings.")
            result[key] = _normalize_json(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [
            _normalize_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{path} contains unsupported metadata type {type(value).__name__}."
    )


def _freeze_json(value: Any) -> Any:
    normalized = _normalize_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {str(key): _freeze_json(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize_json(value),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_mapping(
    values: Mapping[str, str],
    *,
    name: str,
    require_nonempty: bool,
) -> Mapping[str, str]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    normalized: dict[str, str] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError(f"{name} keys must be non-empty strings.")
        normalized[key] = _validate_sha256(value, name=f"{name}[{key!r}]")
    if require_nonempty and not normalized:
        raise ValueError(f"{name} cannot be empty.")
    return MappingProxyType(dict(sorted(normalized.items())))


def _envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = _normalize_json(payload)
    return {
        "payload": normalized,
        "payload_sha256": _sha256_bytes(_canonical_json(normalized).encode("ascii")),
    }


def _read_envelope(
    path: Path,
    *,
    error_type: type[TaskMPIError],
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        text = raw.decode("ascii")
        document = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise error_type(f"invalid or truncated strict-JSON record: {path}") from exc
    if not isinstance(document, dict):
        raise error_type(f"record must be a JSON object: {path}")
    try:
        if raw != _canonical_bytes(document):
            raise error_type(f"record is not in canonical JSON form: {path}")
        payload = document["payload"]
        expected = _validate_sha256(
            document["payload_sha256"], name="payload_sha256"
        )
        actual = _sha256_bytes(_canonical_json(payload).encode("ascii"))
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, error_type):
            raise
        raise error_type(f"record envelope is incomplete or invalid: {path}") from exc
    if actual != expected:
        raise error_type(f"record payload failed its SHA256 check: {path}")
    if not isinstance(payload, dict):
        raise error_type(f"record payload must be an object: {path}")
    return payload


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_immutable(
    target: Path,
    payload: bytes,
    *,
    conflict_type: type[TaskMPIError],
) -> Path:
    """Publish bytes atomically without ever replacing an existing record."""

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            existing = target.read_bytes()
        except OSError as exc:
            raise conflict_type(f"cannot read existing record: {target}") from exc
        if existing == payload:
            return target
        raise conflict_type(f"refusing to overwrite a different record: {target}")

    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError:
            existing = target.read_bytes()
            if existing != payload:
                raise conflict_type(
                    f"a different record was concurrently published: {target}"
                )
        _fsync_directory(target.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return target


@dataclass(frozen=True)
class TaskSpec:
    """One indivisible validation or target-q calculation."""

    task_id: str
    identity_hashes: Mapping[str, str]
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "task_id", _validate_identifier(self.task_id, name="task_id")
        )
        object.__setattr__(
            self,
            "identity_hashes",
            _hash_mapping(
                self.identity_hashes,
                name="identity_hashes",
                require_nonempty=True,
            ),
        )
        if not isinstance(self.payload, Mapping):
            raise TypeError("payload must be a mapping.")
        object.__setattr__(self, "payload", _freeze_json(self.payload))

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "identity_hashes": dict(self.identity_hashes),
            "payload": _normalize_json(self.payload),
        }


@dataclass(frozen=True)
class TaskManifest:
    """Immutable, canonically ordered set of exact task identities."""

    source_revision: str
    source_tree_sha256: str
    environment_sha256: str
    tasks: tuple[TaskSpec, ...]
    identity_hashes: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_revision",
            _validate_identifier(self.source_revision, name="source_revision"),
        )
        object.__setattr__(
            self,
            "source_tree_sha256",
            _validate_sha256(
                self.source_tree_sha256, name="source_tree_sha256"
            ),
        )
        object.__setattr__(
            self,
            "environment_sha256",
            _validate_sha256(
                self.environment_sha256, name="environment_sha256"
            ),
        )
        tasks = tuple(self.tasks)
        if not tasks or not all(isinstance(task, TaskSpec) for task in tasks):
            raise ValueError("tasks must contain at least one TaskSpec.")
        identifiers = [task.task_id for task in tasks]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("task_id values must be unique within a manifest.")
        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(
            self,
            "identity_hashes",
            _hash_mapping(
                self.identity_hashes,
                name="manifest identity_hashes",
                require_nonempty=False,
            ),
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @property
    def run_identity_sha256(self) -> str:
        payload = {
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "environment_sha256": self.environment_sha256,
            "identity_hashes": dict(self.identity_hashes),
        }
        return _sha256_bytes(_canonical_json(payload).encode("ascii"))

    def task_identity_sha256(self, task: TaskSpec | str) -> str:
        selected = self.task(task) if isinstance(task, str) else task
        if selected not in self.tasks:
            raise ValueError("task does not belong to this manifest.")
        ordinal = self.tasks.index(selected)
        payload = {
            "run_identity_sha256": self.run_identity_sha256,
            "ordinal": ordinal,
            **selected.as_dict(),
        }
        return _sha256_bytes(_canonical_json(payload).encode("ascii"))

    def task(self, task_id: str) -> TaskSpec:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise KeyError(task_id)

    def ordinal(self, task: TaskSpec | str) -> int:
        selected = self.task(task) if isinstance(task, str) else task
        try:
            return self.tasks.index(selected)
        except ValueError as exc:
            raise ValueError("task does not belong to this manifest.") from exc

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_name": _MANIFEST_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "environment_sha256": self.environment_sha256,
            "identity_hashes": dict(self.identity_hashes),
            "metadata": _normalize_json(self.metadata),
            "run_identity_sha256": self.run_identity_sha256,
            "tasks": [
                {
                    **task.as_dict(),
                    "task_identity_sha256": self.task_identity_sha256(task),
                }
                for task in self.tasks
            ],
        }

    @property
    def manifest_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload()).encode("ascii"))

    def as_dict(self) -> dict[str, Any]:
        return _envelope(self._payload())


def write_task_manifest(path: str | Path, manifest: TaskManifest) -> Path:
    """Atomically publish an immutable task manifest."""

    if not isinstance(manifest, TaskManifest):
        raise TypeError("manifest must be a TaskManifest.")
    return _publish_immutable(
        Path(path).expanduser(),
        _canonical_bytes(manifest.as_dict()),
        conflict_type=TaskManifestError,
    )


def load_task_manifest(path: str | Path) -> TaskManifest:
    """Load a task manifest and verify every derived identity hash."""

    source = Path(path).expanduser()
    payload = _read_envelope(source, error_type=TaskManifestError)
    try:
        if (
            payload["schema_name"] != _MANIFEST_SCHEMA
            or int(payload["schema_version"]) != _SCHEMA_VERSION
        ):
            raise TaskManifestError(f"unsupported task manifest schema: {source}")
        raw_tasks = payload["tasks"]
        if not isinstance(raw_tasks, list):
            raise TypeError("tasks must be a list")
        tasks = tuple(
            TaskSpec(
                task_id=item["task_id"],
                identity_hashes=item["identity_hashes"],
                payload=item["payload"],
            )
            for item in raw_tasks
        )
        manifest = TaskManifest(
            source_revision=payload["source_revision"],
            source_tree_sha256=payload["source_tree_sha256"],
            environment_sha256=payload["environment_sha256"],
            identity_hashes=payload["identity_hashes"],
            metadata=payload["metadata"],
            tasks=tasks,
        )
        if payload["run_identity_sha256"] != manifest.run_identity_sha256:
            raise TaskManifestError("manifest run identity failed its SHA256 check.")
        raw_task_hashes = [item["task_identity_sha256"] for item in raw_tasks]
        expected_task_hashes = [
            manifest.task_identity_sha256(task) for task in manifest.tasks
        ]
        if raw_task_hashes != expected_task_hashes:
            raise TaskManifestError("manifest task identities failed validation.")
        if _sha256_bytes(_canonical_json(payload).encode("ascii")) != manifest.manifest_sha256:
            raise TaskManifestError("manifest canonical identity failed validation.")
    except TaskManifestError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskManifestError(f"invalid task manifest payload: {source}") from exc
    return manifest


@dataclass(frozen=True)
class TaskResult:
    """Terminal worker result; numerical fields stay in the artifact file."""

    status: Literal["completed", "scientific_failure", "blocked"]
    artifact_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _TERMINAL_STATUSES:
            raise ValueError(f"unsupported terminal task status: {self.status!r}")
        if self.status == "completed" and self.artifact_path is None:
            raise ValueError("completed tasks require an artifact_path.")
        if self.artifact_path is not None:
            object.__setattr__(self, "artifact_path", Path(self.artifact_path))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        object.__setattr__(self, "metadata", _freeze_json(self.metadata))

    @classmethod
    def completed(
        cls,
        artifact_path: str | Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TaskResult":
        return cls(
            status="completed",
            artifact_path=Path(artifact_path),
            metadata={} if metadata is None else metadata,
        )


@dataclass(frozen=True)
class TaskContext:
    """Filesystem and thread-limit context supplied to one whole-task worker."""

    run_directory: Path
    artifact_root: Path
    ordinal: int
    task_identity_sha256: str
    rank: int
    communicator_size: int
    hostname: str
    thread_limit: int
    thread_cap_metadata: Mapping[str, Any]

    def artifact_path(self, suffix: str = ".h5") -> Path:
        if not isinstance(suffix, str) or _SUFFIX_PATTERN.fullmatch(suffix) is None:
            raise ValueError(
                "artifact suffix must begin with '.' and contain only safe "
                "filename characters."
            )
        name = (
            f"{self.ordinal:06d}-"
            f"{self.task_identity_sha256[:16]}{suffix}"
        )
        return self.artifact_root / name

    def write_artifact_bytes(
        self,
        payload: bytes,
        *,
        suffix: str = ".bin",
    ) -> Path:
        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes.")
        return _publish_immutable(
            self.artifact_path(suffix),
            payload,
            conflict_type=DuplicateTaskCheckpointError,
        )


@dataclass(frozen=True)
class TaskCheckpoint:
    """Validated terminal sidecar for one task artifact."""

    manifest_sha256: str
    task_id: str
    task_identity_sha256: str
    status: str
    artifact_relative_path: str | None
    artifact_sha256: str | None
    artifact_validation: Mapping[str, Any]
    metadata: Mapping[str, Any]
    execution: Mapping[str, Any]

    def _payload(self) -> dict[str, Any]:
        artifact: dict[str, str] | None
        if self.artifact_relative_path is None:
            artifact = None
        else:
            assert self.artifact_sha256 is not None
            artifact = {
                "relative_path": self.artifact_relative_path,
                "sha256": self.artifact_sha256,
            }
        return {
            "schema_name": _CHECKPOINT_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "manifest_sha256": self.manifest_sha256,
            "task_id": self.task_id,
            "task_identity_sha256": self.task_identity_sha256,
            "status": self.status,
            "artifact": artifact,
            "artifact_validation": _normalize_json(self.artifact_validation),
            "metadata": _normalize_json(self.metadata),
            "execution": _normalize_json(self.execution),
        }

    @property
    def checkpoint_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload()).encode("ascii"))

    def as_dict(self) -> dict[str, Any]:
        return _envelope(self._payload())


def task_checkpoint_path(
    manifest: TaskManifest,
    task: TaskSpec | str,
    checkpoint_directory: str | Path,
) -> Path:
    selected = manifest.task(task) if isinstance(task, str) else task
    ordinal = manifest.ordinal(selected)
    identity = manifest.task_identity_sha256(selected)
    return Path(checkpoint_directory) / (
        f"{ordinal:06d}-{identity[:16]}.task.json"
    )


def _resolve_artifact(
    artifact_root: Path,
    relative_path: str,
) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise TaskCheckpointCorruptionError("artifact path must be non-empty text.")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise TaskCheckpointCorruptionError(
            "artifact path must be relative and traversal-safe."
        )
    root = artifact_root.expanduser().resolve()
    target = (root / relative).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise TaskCheckpointCorruptionError(
            "artifact resolves outside the declared artifact root."
        ) from exc
    return target


def _validate_artifact(
    path: Path,
    task: TaskSpec,
    validator: ArtifactValidator | None,
) -> Mapping[str, Any]:
    if not path.is_file():
        raise TaskCheckpointCorruptionError(f"task artifact is missing: {path}")
    if validator is None:
        return MappingProxyType({})
    try:
        evidence = validator(path, task)
    except TaskMPIError:
        raise
    except Exception as exc:
        raise TaskCheckpointCorruptionError(
            f"task artifact semantic validation failed: {path}"
        ) from exc
    if evidence is None:
        evidence = {}
    if not isinstance(evidence, Mapping):
        raise TaskCheckpointCorruptionError(
            "artifact validator must return a mapping or None."
        )
    return _freeze_json(evidence)


def _checkpoint_from_result(
    manifest: TaskManifest,
    task: TaskSpec,
    result: TaskResult,
    context: TaskContext,
    *,
    elapsed_seconds: float,
    artifact_validator: ArtifactValidator | None,
    orphan_artifact_adopted: bool = False,
) -> TaskCheckpoint:
    relative_path: str | None = None
    artifact_sha256: str | None = None
    validation: Mapping[str, Any] = MappingProxyType({})
    if result.artifact_path is not None:
        root = context.artifact_root.expanduser().resolve()
        artifact = result.artifact_path.expanduser().resolve()
        try:
            relative_path = artifact.relative_to(root).as_posix()
        except ValueError as exc:
            raise TaskCheckpointCorruptionError(
                "worker artifact must be inside the declared artifact root."
            ) from exc
        validation = _validate_artifact(artifact, task, artifact_validator)
        artifact_sha256 = _file_sha256(artifact)
    execution = {
        "rank": context.rank,
        "communicator_size": context.communicator_size,
        "hostname": context.hostname,
        "elapsed_seconds": elapsed_seconds,
        "thread_limit": context.thread_limit,
        "thread_cap": context.thread_cap_metadata,
    }
    if orphan_artifact_adopted:
        execution["orphan_artifact_adopted"] = True
    return TaskCheckpoint(
        manifest_sha256=manifest.manifest_sha256,
        task_id=task.task_id,
        task_identity_sha256=manifest.task_identity_sha256(task),
        status=result.status,
        artifact_relative_path=relative_path,
        artifact_sha256=artifact_sha256,
        artifact_validation=validation,
        metadata=result.metadata,
        execution=_freeze_json(execution),
    )


def write_task_checkpoint(
    path: str | Path,
    checkpoint: TaskCheckpoint,
) -> Path:
    if not isinstance(checkpoint, TaskCheckpoint):
        raise TypeError("checkpoint must be a TaskCheckpoint.")
    return _publish_immutable(
        Path(path).expanduser(),
        _canonical_bytes(checkpoint.as_dict()),
        conflict_type=DuplicateTaskCheckpointError,
    )


def _validate_checkpoint_execution(value: Any) -> Mapping[str, Any]:
    """Validate the exact worker-execution schema without coercion."""

    if not isinstance(value, dict):
        raise TypeError("checkpoint execution must be an object")
    required = {
        "rank",
        "communicator_size",
        "hostname",
        "elapsed_seconds",
        "thread_limit",
        "thread_cap",
    }
    allowed = required | {"orphan_artifact_adopted"}
    if set(value) not in (required, allowed):
        raise ValueError("checkpoint execution has an invalid key schema")

    def exact_integer(item: Any, *, name: str, positive: bool) -> int:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"checkpoint execution {name} must be an integer")
        if item < (1 if positive else 0):
            raise ValueError(f"checkpoint execution {name} is out of range")
        return item

    rank = exact_integer(value["rank"], name="rank", positive=False)
    communicator_size = exact_integer(
        value["communicator_size"],
        name="communicator_size",
        positive=True,
    )
    if rank >= communicator_size:
        raise ValueError("checkpoint execution rank exceeds communicator")
    hostname = value["hostname"]
    if not isinstance(hostname, str) or not hostname:
        raise TypeError("checkpoint execution hostname must be non-empty text")
    elapsed = value["elapsed_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise TypeError(
            "checkpoint execution elapsed_seconds must be finite and "
            "nonnegative"
        )
    thread_limit = exact_integer(
        value["thread_limit"], name="thread_limit", positive=True
    )
    if "orphan_artifact_adopted" in value and (
        value["orphan_artifact_adopted"] is not True
    ):
        raise TypeError(
            "checkpoint orphan_artifact_adopted, when present, must be true"
        )

    thread_cap = value["thread_cap"]
    expected_cap_keys = {
        "requested_limit",
        "environment",
        "libraries",
        "all_loaded_libraries_capped",
        "cpu_affinity",
        "cpu_affinity_topology",
    }
    if not isinstance(thread_cap, dict) or set(thread_cap) != expected_cap_keys:
        raise TypeError("checkpoint thread_cap has an invalid key schema")
    requested = exact_integer(
        thread_cap["requested_limit"],
        name="thread_cap.requested_limit",
        positive=True,
    )
    if requested != thread_limit:
        raise ValueError("checkpoint thread limits are inconsistent")
    environment = thread_cap["environment"]
    if (
        not isinstance(environment, dict)
        or set(environment) != set(_THREAD_ENVIRONMENT)
        or any(
            item is not None and not isinstance(item, str)
            for item in environment.values()
        )
    ):
        raise TypeError("checkpoint thread environment is invalid")
    if not isinstance(thread_cap["all_loaded_libraries_capped"], bool):
        raise TypeError("checkpoint thread-cap decision must be boolean")
    libraries = thread_cap["libraries"]
    library_keys = {
        "user_api",
        "internal_api",
        "prefix",
        "filepath",
        "version",
        "architecture",
        "num_threads",
    }
    if not isinstance(libraries, list):
        raise TypeError("checkpoint thread libraries must be an array")
    thread_counts: list[int | None] = []
    for library in libraries:
        if not isinstance(library, dict) or set(library) != library_keys:
            raise TypeError("checkpoint thread library schema is invalid")
        if any(
            library[name] is not None
            and not isinstance(library[name], str)
            for name in library_keys - {"num_threads"}
        ):
            raise TypeError("checkpoint thread library metadata is invalid")
        if library["num_threads"] is None:
            thread_counts.append(None)
        else:
            thread_counts.append(
                exact_integer(
                    library["num_threads"],
                    name="thread_cap.libraries[].num_threads",
                    positive=False,
                )
            )
    stored_cap = thread_cap["all_loaded_libraries_capped"]
    if stored_cap is not all(
        count is not None and count <= requested for count in thread_counts
    ):
        raise ValueError(
            "checkpoint thread-cap decision contradicts library thread counts"
        )

    affinity = thread_cap["cpu_affinity"]
    topology = thread_cap["cpu_affinity_topology"]
    if affinity is None:
        if topology is not None:
            raise ValueError(
                "checkpoint topology requires an available CPU affinity"
            )
    else:
        if (
            not isinstance(affinity, list)
            or not affinity
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0
                for item in affinity
            )
            or affinity != sorted(set(affinity))
        ):
            raise TypeError("checkpoint CPU affinity is invalid")
        if not isinstance(topology, list) or len(topology) != len(affinity):
            raise TypeError("checkpoint CPU topology is invalid")
        logical_cpus = []
        for item in topology:
            if (
                not isinstance(item, dict)
                or set(item) != {"logical_cpu", "package_id", "core_id"}
            ):
                raise TypeError("checkpoint CPU topology entry is invalid")
            logical = exact_integer(
                item["logical_cpu"],
                name="thread_cap.topology.logical_cpu",
                positive=False,
            )
            exact_integer(
                item["package_id"],
                name="thread_cap.topology.package_id",
                positive=False,
            )
            exact_integer(
                item["core_id"],
                name="thread_cap.topology.core_id",
                positive=False,
            )
            logical_cpus.append(logical)
        if logical_cpus != affinity:
            raise ValueError(
                "checkpoint CPU topology differs from CPU affinity"
            )
    return _freeze_json(value)


def load_task_checkpoint(
    path: str | Path,
    *,
    manifest: TaskManifest,
    artifact_root: str | Path,
    artifact_validator: ArtifactValidator | None = None,
) -> TaskCheckpoint:
    """Load one sidecar and revalidate its identity and artifact content."""

    source = Path(path).expanduser()
    payload = _read_envelope(
        source, error_type=TaskCheckpointCorruptionError
    )
    try:
        expected_payload_keys = {
            "schema_name",
            "schema_version",
            "manifest_sha256",
            "task_id",
            "task_identity_sha256",
            "status",
            "artifact",
            "artifact_validation",
            "metadata",
            "execution",
        }
        if (
            set(payload) != expected_payload_keys
            or payload["schema_name"] != _CHECKPOINT_SCHEMA
            or isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["schema_version"] != _SCHEMA_VERSION
        ):
            raise TaskCheckpointCorruptionError(
                f"unsupported task checkpoint schema: {source}"
            )
        if payload["manifest_sha256"] != manifest.manifest_sha256:
            raise TaskCheckpointMismatchError(
                "task checkpoint belongs to a different manifest."
            )
        task_id = _validate_identifier(payload["task_id"], name="task_id")
        try:
            task = manifest.task(task_id)
        except KeyError as exc:
            raise TaskCheckpointMismatchError(
                f"task checkpoint names unknown task {task_id!r}."
            ) from exc
        task_identity = _validate_sha256(
            payload["task_identity_sha256"], name="task_identity_sha256"
        )
        if task_identity != manifest.task_identity_sha256(task):
            raise TaskCheckpointMismatchError(
                f"task checkpoint identity does not match {task_id!r}."
            )
        status = payload["status"]
        if status not in _TERMINAL_STATUSES:
            raise TaskCheckpointCorruptionError(
                f"invalid terminal task status {status!r}."
            )
        artifact_record = payload["artifact"]
        artifact_path: str | None
        artifact_sha256: str | None
        actual_validation: Mapping[str, Any]
        if artifact_record is None:
            if status == "completed":
                raise TaskCheckpointCorruptionError(
                    "completed task checkpoint lacks an artifact."
                )
            artifact_path = None
            artifact_sha256 = None
            actual_validation = MappingProxyType({})
        else:
            if (
                not isinstance(artifact_record, dict)
                or set(artifact_record) != {"relative_path", "sha256"}
            ):
                raise TaskCheckpointCorruptionError(
                    "checkpoint artifact record must be an object or null."
                )
            artifact_path = artifact_record["relative_path"]
            artifact_sha256 = _validate_sha256(
                artifact_record["sha256"], name="artifact.sha256"
            )
            resolved = _resolve_artifact(Path(artifact_root), artifact_path)
            if not resolved.is_file() or _file_sha256(resolved) != artifact_sha256:
                raise TaskCheckpointCorruptionError(
                    f"task artifact failed its SHA256 check: {resolved}"
                )
            actual_validation = _validate_artifact(
                resolved, task, artifact_validator
            )
        stored_validation = _freeze_json(payload["artifact_validation"])
        if actual_validation != stored_validation:
            raise TaskCheckpointCorruptionError(
                "task artifact semantic validation evidence changed."
            )
        metadata = _freeze_json(payload["metadata"])
        execution = _validate_checkpoint_execution(payload["execution"])
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
    except TaskCheckpointError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise TaskCheckpointCorruptionError(
            f"invalid task checkpoint payload: {source}"
        ) from exc
    return TaskCheckpoint(
        manifest_sha256=manifest.manifest_sha256,
        task_id=task_id,
        task_identity_sha256=task_identity,
        status=status,
        artifact_relative_path=artifact_path,
        artifact_sha256=artifact_sha256,
        artifact_validation=stored_validation,
        metadata=metadata,
        execution=execution,
    )


def collect_task_checkpoints(
    manifest: TaskManifest,
    checkpoint_directory: str | Path,
    artifact_root: str | Path,
    *,
    artifact_validator: ArtifactValidator | None = None,
    require_complete: bool = True,
) -> tuple[TaskCheckpoint, ...]:
    """Validate and merge outcomes in manifest order, never finish order."""

    if not isinstance(manifest, TaskManifest):
        raise TypeError("manifest must be a TaskManifest.")
    directory = Path(checkpoint_directory).expanduser()
    found: dict[str, TaskCheckpoint] = {}
    artifact_owners: dict[str, str] = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*.task.json")):
            checkpoint = load_task_checkpoint(
                path,
                manifest=manifest,
                artifact_root=artifact_root,
                artifact_validator=artifact_validator,
            )
            if checkpoint.task_id in found:
                raise DuplicateTaskCheckpointError(
                    f"duplicate checkpoints claim task {checkpoint.task_id!r}."
                )
            expected_path = task_checkpoint_path(
                manifest, checkpoint.task_id, directory
            )
            if path.resolve() != expected_path.resolve():
                raise TaskCheckpointMismatchError(
                    "task checkpoint is not stored at its canonical manifest "
                    f"path: {path} != {expected_path}"
                )
            if checkpoint.artifact_relative_path is not None:
                owner = artifact_owners.get(checkpoint.artifact_relative_path)
                if owner is not None:
                    raise DuplicateTaskCheckpointError(
                        "two task checkpoints claim artifact "
                        f"{checkpoint.artifact_relative_path!r}: "
                        f"{owner!r} and {checkpoint.task_id!r}."
                    )
                artifact_owners[checkpoint.artifact_relative_path] = (
                    checkpoint.task_id
                )
            found[checkpoint.task_id] = checkpoint
    missing = [task.task_id for task in manifest.tasks if task.task_id not in found]
    if require_complete and missing:
        raise MissingTaskCheckpointError(missing)
    return tuple(found[task.task_id] for task in manifest.tasks if task.task_id in found)


def _cpu_affinity_topology(
    cpu_affinity: Sequence[int] | None,
) -> list[dict[str, int]] | None:
    """Resolve Linux logical CPUs to stable physical package/core IDs."""

    if cpu_affinity is None or not cpu_affinity:
        return None
    records: list[dict[str, int]] = []
    try:
        for logical_cpu in cpu_affinity:
            topology = (
                Path("/sys/devices/system/cpu")
                / f"cpu{int(logical_cpu)}"
                / "topology"
            )
            package_text = (
                topology / "physical_package_id"
            ).read_text(encoding="ascii").strip()
            core_text = (
                topology / "core_id"
            ).read_text(encoding="ascii").strip()
            package_id = int(package_text)
            core_id = int(core_text)
            if (
                package_id < 0
                or core_id < 0
                or str(package_id) != package_text
                or str(core_id) != core_text
            ):
                return None
            records.append(
                {
                    "logical_cpu": int(logical_cpu),
                    "package_id": package_id,
                    "core_id": core_id,
                }
            )
    except (OSError, UnicodeError, ValueError):
        return None
    return records


def _thread_cap_snapshot(limit: int) -> Mapping[str, Any]:
    libraries: list[dict[str, Any]] = []
    for item in threadpool_info():
        libraries.append(
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "filepath": item.get("filepath"),
                "version": item.get("version"),
                "architecture": item.get("architecture"),
                "num_threads": item.get("num_threads"),
            }
        )
    libraries.sort(
        key=lambda value: (
            str(value.get("internal_api")),
            str(value.get("prefix")),
            str(value.get("filepath")),
        )
    )
    try:
        cpu_affinity: list[int] | None = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_affinity = None
    cpu_affinity_topology = _cpu_affinity_topology(cpu_affinity)
    capped = all(
        isinstance(item["num_threads"], int)
        and int(item["num_threads"]) <= limit
        for item in libraries
    )
    return _freeze_json(
        {
            "requested_limit": limit,
            "environment": {
                name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
            },
            "libraries": libraries,
            "all_loaded_libraries_capped": capped,
            "cpu_affinity": cpu_affinity,
            "cpu_affinity_topology": cpu_affinity_topology,
        }
    )


class _SerialCommunicator:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1


def serial_task_communicator() -> Any:
    """Return the explicit one-rank communicator used by unit tests."""

    return _SerialCommunicator()


def _load_mpi(*, required: bool) -> Any | None:
    try:
        return importlib.import_module("mpi4py.MPI")
    except (ImportError, OSError) as exc:
        if required:
            raise MPIDependencyError(
                "task-level MPI requires the optional 'mpi4py' package; "
                "install it with `pip install 'negfpy[mpi]'` and launch with "
                "that environment's mpiexec."
            ) from exc
        return None


def mpi_available() -> bool:
    """Return whether ``mpi4py.MPI`` can be imported, without caching it."""

    return _load_mpi(required=False) is not None


def _resolve_communicator(comm: Any | None, *, require_mpi: bool) -> Any:
    if comm is None:
        mpi = _load_mpi(required=require_mpi)
        if mpi is None:
            return _SerialCommunicator()
        comm = mpi.COMM_WORLD
    if not callable(getattr(comm, "Get_rank", None)) or not callable(
        getattr(comm, "Get_size", None)
    ):
        raise TypeError("comm must provide MPI-style Get_rank/Get_size methods.")
    return comm


@dataclass(frozen=True)
class TaskAssignment:
    task_id: str
    status: str
    rank: int
    hostname: str
    elapsed_seconds: float
    resumed: bool
    cpu_affinity: tuple[int, ...] | None
    thread_cap_passed: bool
    checkpoint_relative_path: str
    checkpoint_file_sha256: str
    artifact_relative_path: str | None
    artifact_sha256: str | None
    orphan_artifact_adopted: bool = False
    cpu_affinity_topology: (
        tuple[tuple[int, int, int], ...] | None
    ) = None
    origin_communicator_size: int = 1

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "rank": self.rank,
            "hostname": self.hostname,
            "elapsed_seconds": self.elapsed_seconds,
            "resumed": self.resumed,
            "cpu_affinity": (
                None if self.cpu_affinity is None else list(self.cpu_affinity)
            ),
            "thread_cap_passed": self.thread_cap_passed,
            "checkpoint_relative_path": self.checkpoint_relative_path,
            "checkpoint_file_sha256": self.checkpoint_file_sha256,
            "artifact_relative_path": self.artifact_relative_path,
            "artifact_sha256": self.artifact_sha256,
            "orphan_artifact_adopted": self.orphan_artifact_adopted,
            "cpu_affinity_topology": (
                None
                if self.cpu_affinity_topology is None
                else [
                    {
                        "logical_cpu": logical_cpu,
                        "package_id": package_id,
                        "core_id": core_id,
                    }
                    for (
                        logical_cpu,
                        package_id,
                        core_id,
                    ) in self.cpu_affinity_topology
                ]
            ),
            "origin_communicator_size": self.origin_communicator_size,
        }


def task_assignment_physical_core_evidence(
    assignments: Sequence[TaskAssignment | Mapping[str, Any]],
    *,
    require_available: bool = False,
    origin_execution_ids: Mapping[str, str] | None = None,
) -> Mapping[str, Any]:
    """Validate physical-core binding for archived numerical assignments.

    Restart reports can contain checkpoints produced by several executions.
    Rank numbers and physical-core occupancy are meaningful only within one
    execution, so restart validators must provide the task-to-origin mapping
    reconstructed by :func:`task_execution_mpi_origin_lineage_evidence`.
    """

    bindings: dict[
        tuple[str | None, str, int],
        dict[str, Any],
    ] = {}
    if origin_execution_ids is not None and not isinstance(
        origin_execution_ids, Mapping
    ):
        raise TaskManifestError(
            "origin_execution_ids must be a task-to-execution mapping."
        )
    normalized_origins: dict[str, str] | None = None
    if origin_execution_ids is not None:
        normalized_origins = {}
        for task_id, execution_id in origin_execution_ids.items():
            if (
                not isinstance(task_id, str)
                or not task_id
                or not isinstance(execution_id, str)
                or not execution_id
            ):
                raise TaskManifestError(
                    "origin_execution_ids contains an invalid identity."
                )
            normalized_origins[task_id] = execution_id
    unavailable: list[str] = []
    seen_task_ids: set[str] = set()
    one_core = True
    consistent = True
    for raw in assignments:
        value = raw.as_dict() if isinstance(raw, TaskAssignment) else raw
        if not isinstance(value, Mapping):
            raise TaskManifestError(
                "task assignment physical-core evidence is not a mapping."
            )
        task_id = value.get("task_id")
        hostname = value.get("hostname")
        rank = value.get("rank")
        affinity = value.get("cpu_affinity")
        topology = value.get("cpu_affinity_topology")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen_task_ids
            or not isinstance(hostname, str)
            or not hostname
            or isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank < 0
        ):
            raise TaskManifestError(
                "task assignment identity is invalid for CPU binding."
            )
        seen_task_ids.add(task_id)
        origin_execution_id = (
            None
            if normalized_origins is None
            else normalized_origins.get(task_id)
        )
        if normalized_origins is not None and origin_execution_id is None:
            raise TaskManifestError(
                "origin_execution_ids does not cover every assignment."
            )
        if affinity is None or topology is None:
            unavailable.append(task_id)
            continue
        if (
            not isinstance(affinity, (list, tuple))
            or not affinity
            or any(
                isinstance(cpu, bool)
                or not isinstance(cpu, int)
                or cpu < 0
                for cpu in affinity
            )
            or len(set(affinity)) != len(affinity)
            or list(affinity) != sorted(affinity)
            or not isinstance(topology, (list, tuple))
            or len(topology) != len(affinity)
        ):
            raise TaskManifestError(
                "task assignment logical CPU affinity is invalid."
            )
        normalized: list[tuple[int, int, int]] = []
        for item in topology:
            if (
                not isinstance(item, Mapping)
                or set(item)
                != {"logical_cpu", "package_id", "core_id"}
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, int)
                    or component < 0
                    for component in item.values()
                )
            ):
                raise TaskManifestError(
                    "task assignment CPU topology record is invalid."
                )
            normalized.append(
                (
                    int(item["logical_cpu"]),
                    int(item["package_id"]),
                    int(item["core_id"]),
                )
            )
        if [item[0] for item in normalized] != list(affinity):
            raise TaskManifestError(
                "task assignment CPU topology does not cover its affinity."
            )
        physical = {(item[1], item[2]) for item in normalized}
        if len(physical) != 1:
            one_core = False
            continue
        physical_core = next(iter(physical))
        key = (origin_execution_id, hostname, rank)
        record = bindings.get(key)
        binding_identity = (
            tuple(int(cpu) for cpu in affinity),
            tuple(normalized),
            physical_core,
        )
        if record is None:
            bindings[key] = {
                "identity": binding_identity,
                "task_ids": [task_id],
            }
        elif record["identity"] != binding_identity:
            consistent = False
        else:
            record["task_ids"].append(task_id)

    if normalized_origins is not None and set(normalized_origins) != (
        seen_task_ids
    ):
        raise TaskManifestError(
            "origin_execution_ids must cover exactly the selected tasks."
        )
    by_execution_host: dict[
        tuple[str | None, str],
        dict[tuple[int, int], int],
    ] = {}
    disjoint = True
    for (origin_execution_id, hostname, rank), record in bindings.items():
        physical = record["identity"][2]
        observed = by_execution_host.setdefault(
            (origin_execution_id, hostname),
            {},
        )
        if physical in observed and observed[physical] != rank:
            disjoint = False
        observed[physical] = rank
    rank_bindings = []
    for (
        origin_execution_id,
        hostname,
        rank,
    ), record in sorted(
        bindings.items(),
        key=lambda item: (
            "" if item[0][0] is None else item[0][0],
            item[0][1],
            item[0][2],
        ),
    ):
        logical, _, physical = record["identity"]
        rank_bindings.append(
            {
                "origin_execution_id": origin_execution_id,
                "hostname": hostname,
                "rank": rank,
                "logical_cpus": list(logical),
                "physical_core": {
                    "package_id": physical[0],
                    "core_id": physical[1],
                },
                "task_ids": list(record["task_ids"]),
            }
        )
    available = (
        bool(assignments)
        and not unavailable
        and one_core
        and consistent
        and disjoint
    )
    evidence = {
        "schema": "negfpy_task_physical_core_bindings",
        "schema_version": 2,
        "origin_execution_scoped": normalized_origins is not None,
        "available": available,
        "one_physical_core_per_assignment": one_core,
        "consistent_affinity_per_rank": consistent,
        "disjoint_physical_cores_across_ranks": disjoint,
        "unavailable_task_ids": unavailable,
        "rank_bindings": rank_bindings,
    }
    if require_available and not available:
        raise TaskManifestError(
            "release task execution lacks valid, consistent, disjoint "
            "physical-core binding evidence."
        )
    return _freeze_json(evidence)


@dataclass(frozen=True)
class MPIExecutionReport:
    """One orchestration attempt; scientific ordering follows the manifest."""

    execution_id: str
    manifest_sha256: str
    mode: Literal["serial", "mpi"]
    communicator_size: int
    coordinator_hostname: str
    started_utc: str
    finished_utc: str
    elapsed_seconds: float
    thread_limit: int
    assignments: tuple[TaskAssignment, ...]
    missing_task_ids: tuple[str, ...]
    infrastructure_errors: tuple[Mapping[str, Any], ...]

    @property
    def complete(self) -> bool:
        return not self.missing_task_ids and not self.infrastructure_errors

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_name": _EXECUTION_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "execution_id": self.execution_id,
            "manifest_sha256": self.manifest_sha256,
            "mode": self.mode,
            "communicator_size": self.communicator_size,
            "coordinator_hostname": self.coordinator_hostname,
            "started_utc": self.started_utc,
            "finished_utc": self.finished_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "thread_limit": self.thread_limit,
            "complete": self.complete,
            "assignments": [item.as_dict() for item in self.assignments],
            "missing_task_ids": list(self.missing_task_ids),
            "infrastructure_errors": [
                _normalize_json(item) for item in self.infrastructure_errors
            ],
        }

    @property
    def execution_sha256(self) -> str:
        return _sha256_bytes(_canonical_json(self._payload()).encode("ascii"))

    def as_dict(self) -> dict[str, Any]:
        return _envelope(self._payload())


def task_execution_mpi_origin_lineage_evidence(
    selected_execution: MPIExecutionReport | Mapping[str, Any],
    execution_history: Sequence[
        MPIExecutionReport | Mapping[str, Any]
    ],
    *,
    require_mpi_origin: bool = False,
) -> Mapping[str, Any]:
    """Prove every selected task artifact originated in fresh worker MPI."""

    def payload(
        value: MPIExecutionReport | Mapping[str, Any],
    ) -> dict[str, Any]:
        raw = (
            value._payload()  # noqa: SLF001
            if isinstance(value, MPIExecutionReport)
            else value
        )
        normalized = _normalize_json(raw)
        if not isinstance(normalized, dict):
            raise TaskManifestError(
                "MPI origin-lineage execution must be a mapping."
            )
        return normalized

    selected = payload(selected_execution)
    history = [payload(value) for value in execution_history]
    selected_id = selected.get("execution_id")
    selected_assignments = selected.get("assignments")
    if (
        not isinstance(selected_id, str)
        or not selected_id
        or not isinstance(selected_assignments, list)
        or selected.get("complete") is not True
    ):
        raise TaskManifestError(
            "selected MPI origin-lineage execution is incomplete."
        )
    by_execution: dict[str, dict[str, Any]] = {}
    execution_order: dict[str, int] = {}
    for ordinal, execution in enumerate(history):
        execution_id = execution.get("execution_id")
        assignments = execution.get("assignments")
        communicator_size = execution.get("communicator_size")
        complete = execution.get("complete")
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or execution_id in by_execution
            or not isinstance(assignments, list)
            or execution.get("mode") not in {"serial", "mpi"}
            or isinstance(communicator_size, bool)
            or not isinstance(communicator_size, int)
            or communicator_size <= 0
            or (
                execution.get("mode") == "serial"
                and communicator_size != 1
            )
            or (
                execution.get("mode") == "mpi"
                and communicator_size < 2
            )
            or type(complete) is not bool
        ):
            raise TaskManifestError(
                "MPI origin-lineage history is invalid or duplicated."
            )
        by_execution[execution_id] = execution
        execution_order[execution_id] = ordinal
    if selected_id not in by_execution:
        by_execution[selected_id] = selected
        execution_order[selected_id] = len(execution_order)
    elif by_execution[selected_id] != selected:
        raise TaskManifestError(
            "selected MPI execution differs from its history record."
        )

    lineage: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    seen_tasks: set[str] = set()
    for assignment in selected_assignments:
        if not isinstance(assignment, dict):
            raise TaskManifestError(
                "MPI origin-lineage assignment must be a mapping."
            )
        task_id = assignment.get("task_id")
        if (
            not isinstance(task_id, str)
            or not task_id
            or task_id in seen_tasks
            or assignment.get("status") != "completed"
            or type(assignment.get("resumed")) is not bool
            or type(assignment.get("orphan_artifact_adopted")) is not bool
        ):
            raise TaskManifestError(
                "MPI origin-lineage assignment identity is invalid."
            )
        seen_tasks.add(task_id)
        origin_rank = assignment.get("rank")
        origin_size = assignment.get("origin_communicator_size")
        mpi_origin = (
            isinstance(origin_rank, int)
            and not isinstance(origin_rank, bool)
            and isinstance(origin_size, int)
            and not isinstance(origin_size, bool)
            and origin_rank > 0
            and origin_size >= 2
            and origin_rank < origin_size
            and assignment["orphan_artifact_adopted"] is False
        )
        origin_execution_id: str | None = None
        if not assignment["resumed"]:
            if (
                selected.get("mode") == "mpi"
                and selected.get("communicator_size") == origin_size
                and mpi_origin
            ):
                origin_execution_id = selected_id
        elif mpi_origin:
            matches: list[str] = []
            for execution_id, execution in by_execution.items():
                if (
                    execution_id == selected_id
                    or execution.get("mode") != "mpi"
                    or execution_order[execution_id]
                    >= execution_order[selected_id]
                    or execution.get("communicator_size") != origin_size
                ):
                    continue
                for candidate in execution.get("assignments", ()):
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("task_id") == task_id
                        and candidate.get("status") == "completed"
                        and candidate.get("resumed") is False
                        and candidate.get("orphan_artifact_adopted")
                        is False
                        and candidate.get("rank") == origin_rank
                        and candidate.get("origin_communicator_size")
                        == origin_size
                        and candidate.get("checkpoint_file_sha256")
                        == assignment.get("checkpoint_file_sha256")
                        and candidate.get("artifact_sha256")
                        == assignment.get("artifact_sha256")
                    ):
                        matches.append(execution_id)
            if matches:
                origin_execution_id = sorted(set(matches))[0]
        if origin_execution_id is None:
            failures.append(
                {
                    "task_id": task_id,
                    "reason": (
                        "missing_fresh_nonorphan_worker_mpi_origin"
                    ),
                }
            )
            continue
        lineage.append(
            {
                "task_id": task_id,
                "selected_resumed": assignment["resumed"],
                "origin_execution_id": origin_execution_id,
                "origin_rank": int(origin_rank),
                "origin_communicator_size": int(origin_size),
                "checkpoint_file_sha256": assignment.get(
                    "checkpoint_file_sha256"
                ),
                "artifact_sha256": assignment.get("artifact_sha256"),
            }
        )
    passed = (
        not failures
        and len(lineage) == len(selected_assignments)
        and selected.get("mode") == "mpi"
        and isinstance(selected.get("communicator_size"), int)
        and not isinstance(selected.get("communicator_size"), bool)
        and selected["communicator_size"] >= 2
    )
    evidence = {
        "schema": "negfpy_task_mpi_origin_lineage",
        "schema_version": 1,
        "selected_execution_id": selected_id,
        "passed": passed,
        "task_count": len(selected_assignments),
        "lineage": lineage,
        "failures": failures,
    }
    if require_mpi_origin and not passed:
        raise TaskManifestError(
            "release task execution lacks fresh, nonorphan worker-MPI "
            "artifact origin lineage."
        )
    return _freeze_json(evidence)


def write_execution_manifest(
    path: str | Path,
    report: MPIExecutionReport,
) -> Path:
    if not isinstance(report, MPIExecutionReport):
        raise TypeError("report must be an MPIExecutionReport.")
    return _publish_immutable(
        Path(path).expanduser(),
        _canonical_bytes(report.as_dict()),
        conflict_type=TaskManifestError,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _new_execution_id() -> str:
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}-{uuid4().hex[:12]}"


def _execute_one(
    manifest: TaskManifest,
    task: TaskSpec,
    worker: TaskWorker,
    *,
    run_directory: Path,
    artifact_root: Path,
    checkpoint_directory: Path,
    rank: int,
    communicator_size: int,
    thread_limit: int,
    artifact_validator: ArtifactValidator | None,
    orphan_artifact_suffix: str | None,
    orphan_artifact_adopter: OrphanArtifactAdopter | None,
) -> TaskCheckpoint:
    started = time.perf_counter()
    orphan_adopted = False
    with threadpool_limits(limits=thread_limit):
        cap = _thread_cap_snapshot(thread_limit)
        if cap["all_loaded_libraries_capped"] is not True:
            raise TaskExecutionError(
                "threadpoolctl could not enforce the requested BLAS thread cap."
            )
        context = TaskContext(
            run_directory=run_directory,
            artifact_root=artifact_root,
            ordinal=manifest.ordinal(task),
            task_identity_sha256=manifest.task_identity_sha256(task),
            rank=rank,
            communicator_size=communicator_size,
            hostname=socket.gethostname(),
            thread_limit=thread_limit,
            thread_cap_metadata=cap,
        )
        orphan_path = (
            None
            if orphan_artifact_suffix is None
            else context.artifact_path(orphan_artifact_suffix)
        )
        if orphan_path is not None and orphan_path.exists():
            assert orphan_artifact_adopter is not None
            evidence = _validate_artifact(
                orphan_path, task, artifact_validator
            )
            result = orphan_artifact_adopter(orphan_path, task, evidence)
            orphan_adopted = True
        else:
            result = worker(task, context)
    if not isinstance(result, TaskResult):
        raise TypeError("task worker must return a TaskResult.")
    if orphan_adopted and result.artifact_path != orphan_path:
        raise TaskCheckpointMismatchError(
            "orphan adopter must return the validated immutable artifact."
        )
    elapsed = time.perf_counter() - started
    checkpoint = _checkpoint_from_result(
        manifest,
        task,
        result,
        context,
        elapsed_seconds=elapsed,
        artifact_validator=artifact_validator,
        orphan_artifact_adopted=orphan_adopted,
    )
    target = task_checkpoint_path(manifest, task, checkpoint_directory)
    write_task_checkpoint(target, checkpoint)
    return load_task_checkpoint(
        target,
        manifest=manifest,
        artifact_root=artifact_root,
        artifact_validator=artifact_validator,
    )


def _assignments_from_checkpoints(
    manifest: TaskManifest,
    checkpoints: Sequence[TaskCheckpoint],
    checkpoint_directory: Path,
    *,
    initially_complete: set[str],
) -> tuple[TaskAssignment, ...]:
    by_id = {checkpoint.task_id: checkpoint for checkpoint in checkpoints}
    assignments: list[TaskAssignment] = []
    for task in manifest.tasks:
        checkpoint = by_id.get(task.task_id)
        if checkpoint is None:
            continue
        target = task_checkpoint_path(manifest, task, checkpoint_directory)
        execution = checkpoint.execution
        thread_cap = execution["thread_cap"]
        raw_affinity = thread_cap["cpu_affinity"]
        raw_topology = thread_cap.get("cpu_affinity_topology")
        topology: tuple[tuple[int, int, int], ...] | None
        if raw_topology is None:
            topology = None
        else:
            if (
                not isinstance(raw_topology, (tuple, list))
                or not raw_topology
                or any(
                    not isinstance(item, Mapping)
                    or set(item)
                    != {"logical_cpu", "package_id", "core_id"}
                    for item in raw_topology
                )
                or any(
                    type(value) is not int or value < 0
                    for item in raw_topology
                    for value in item.values()
                )
            ):
                raise TaskCheckpointCorruptionError(
                    "checkpoint CPU-affinity topology is invalid."
                )
            topology = tuple(
                (
                    int(item["logical_cpu"]),
                    int(item["package_id"]),
                    int(item["core_id"]),
                )
                for item in raw_topology
            )
            if (
                tuple(value[0] for value in topology)
                != (
                    ()
                    if raw_affinity is None
                    else tuple(int(value) for value in raw_affinity)
                )
                or len(set(topology)) != len(topology)
            ):
                raise TaskCheckpointCorruptionError(
                    "checkpoint CPU-affinity topology conflicts with "
                    "logical affinity."
                )
        orphan_adopted = execution.get(
            "orphan_artifact_adopted",
            False,
        )
        if type(orphan_adopted) is not bool:
            raise TaskCheckpointCorruptionError(
                "checkpoint orphan_artifact_adopted must be boolean."
            )
        origin_size = execution.get("communicator_size")
        origin_rank = execution.get("rank")
        if (
            isinstance(origin_size, bool)
            or not isinstance(origin_size, int)
            or origin_size <= 0
            or isinstance(origin_rank, bool)
            or not isinstance(origin_rank, int)
            or not 0 <= origin_rank < origin_size
        ):
            raise TaskCheckpointCorruptionError(
                "checkpoint origin rank/communicator size is invalid."
            )
        assignments.append(
            TaskAssignment(
                task_id=task.task_id,
                status=checkpoint.status,
                rank=int(execution["rank"]),
                hostname=str(execution["hostname"]),
                elapsed_seconds=float(execution["elapsed_seconds"]),
                resumed=(
                    task.task_id in initially_complete or orphan_adopted
                ),
                cpu_affinity=(
                    None
                    if raw_affinity is None
                    else tuple(int(value) for value in raw_affinity)
                ),
                thread_cap_passed=bool(
                    thread_cap["all_loaded_libraries_capped"]
                ),
                checkpoint_relative_path=target.name,
                checkpoint_file_sha256=_file_sha256(target),
                artifact_relative_path=checkpoint.artifact_relative_path,
                artifact_sha256=checkpoint.artifact_sha256,
                orphan_artifact_adopted=orphan_adopted,
                cpu_affinity_topology=topology,
                origin_communicator_size=origin_size,
            )
        )
    return tuple(assignments)


def _build_report(
    manifest: TaskManifest,
    *,
    execution_id: str,
    communicator_size: int,
    coordinator_hostname: str,
    started_utc: str,
    started_clock: float,
    thread_limit: int,
    checkpoints: Sequence[TaskCheckpoint],
    checkpoint_directory: Path,
    initially_complete: set[str],
    infrastructure_errors: Sequence[Mapping[str, Any]],
) -> MPIExecutionReport:
    completed = {checkpoint.task_id for checkpoint in checkpoints}
    missing = tuple(
        task.task_id for task in manifest.tasks if task.task_id not in completed
    )
    return MPIExecutionReport(
        execution_id=execution_id,
        manifest_sha256=manifest.manifest_sha256,
        mode="serial" if communicator_size == 1 else "mpi",
        communicator_size=communicator_size,
        coordinator_hostname=coordinator_hostname,
        started_utc=started_utc,
        finished_utc=_utc_now(),
        elapsed_seconds=time.perf_counter() - started_clock,
        thread_limit=thread_limit,
        assignments=_assignments_from_checkpoints(
            manifest,
            checkpoints,
            checkpoint_directory,
            initially_complete=initially_complete,
        ),
        missing_task_ids=missing,
        infrastructure_errors=tuple(
            _freeze_json(item) for item in infrastructure_errors
        ),
    )


def execute_task_manifest(
    manifest: TaskManifest,
    worker: TaskWorker,
    run_directory: str | Path,
    *,
    comm: Any | None = None,
    require_mpi: bool = False,
    thread_limit: int = 1,
    artifact_validator: ArtifactValidator | None = None,
    execution_id: str | None = None,
    orphan_artifact_suffix: str | None = None,
    orphan_artifact_adopter: OrphanArtifactAdopter | None = None,
) -> MPIExecutionReport | None:
    """Execute whole tasks serially or with a rank-zero MPI coordinator.

    Every rank must call this function with the same manifest and worker.  Rank
    zero returns the canonical execution report; worker ranks return ``None``.
    Existing valid terminal checkpoints are resumed without invoking ``worker``.
    """

    if not isinstance(manifest, TaskManifest):
        raise TypeError("manifest must be a TaskManifest.")
    if not callable(worker):
        raise TypeError("worker must be callable.")
    if isinstance(thread_limit, bool) or not isinstance(thread_limit, int):
        raise TypeError("thread_limit must be an integer.")
    if thread_limit < 1:
        raise ValueError("thread_limit must be positive.")
    if execution_id is not None:
        execution_id = _validate_identifier(execution_id, name="execution_id")
        if "/" in execution_id or "\\" in execution_id:
            raise ValueError("execution_id cannot contain path separators.")
    if (orphan_artifact_suffix is None) != (
        orphan_artifact_adopter is None
    ):
        raise ValueError(
            "orphan suffix and adopter must be supplied together."
        )
    if orphan_artifact_suffix is not None and (
        _SUFFIX_PATTERN.fullmatch(orphan_artifact_suffix) is None
    ):
        raise ValueError("orphan artifact suffix is invalid.")

    communicator = _resolve_communicator(comm, require_mpi=require_mpi)
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    if size < 1 or rank < 0 or rank >= size:
        raise TaskMPIError("communicator returned an invalid rank or size.")
    if size > 1:
        hashes = tuple(communicator.allgather(manifest.manifest_sha256))
        if len(hashes) != size or any(
            value != manifest.manifest_sha256 for value in hashes
        ):
            raise TaskManifestError(
                "MPI ranks constructed different task manifests."
            )

    root = Path(run_directory).expanduser().resolve()
    artifact_root = root / "task-artifacts"
    checkpoint_directory = root / "task-outcomes"
    manifest_path = root / "task-manifest.json"
    started_clock = time.perf_counter()
    started_utc = _utc_now()
    coordinator_hostname = socket.gethostname() if rank == 0 else ""
    startup_error: Exception | None = None
    initial_checkpoints: tuple[TaskCheckpoint, ...] = ()
    selected_execution_id = execution_id
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            artifact_root.mkdir(parents=True, exist_ok=True)
            checkpoint_directory.mkdir(parents=True, exist_ok=True)
            write_task_manifest(manifest_path, manifest)
            loaded_manifest = load_task_manifest(manifest_path)
            if loaded_manifest.manifest_sha256 != manifest.manifest_sha256:
                raise TaskManifestError(
                    "published task manifest does not match this invocation."
                )
            initial_checkpoints = collect_task_checkpoints(
                manifest,
                checkpoint_directory,
                artifact_root,
                artifact_validator=artifact_validator,
                require_complete=False,
            )
            if selected_execution_id is None:
                selected_execution_id = _new_execution_id()
        except Exception as exc:
            startup_error = exc

    if size > 1:
        startup = communicator.bcast(
            {
                "ok": startup_error is None,
                "execution_id": selected_execution_id,
                "error_type": None
                if startup_error is None
                else type(startup_error).__name__,
                "error_message": None
                if startup_error is None
                else str(startup_error),
                "coordinator_hostname": coordinator_hostname,
                "started_utc": started_utc,
            }
            if rank == 0
            else None,
            root=0,
        )
        if not startup["ok"]:
            if rank == 0 and startup_error is not None:
                raise startup_error
            raise TaskExecutionError(
                "rank-zero startup failed: "
                f"{startup['error_type']}: {startup['error_message']}"
            )
        selected_execution_id = startup["execution_id"]
        coordinator_hostname = startup["coordinator_hostname"]
        started_utc = startup["started_utc"]
    elif startup_error is not None:
        raise startup_error
    assert selected_execution_id is not None

    initial_ids = {item.task_id for item in initial_checkpoints}
    pending = [task for task in manifest.tasks if task.task_id not in initial_ids]
    infrastructure_errors: list[Mapping[str, Any]] = []

    if size == 1:
        for task in pending:
            try:
                _execute_one(
                    manifest,
                    task,
                    worker,
                    run_directory=root,
                    artifact_root=artifact_root,
                    checkpoint_directory=checkpoint_directory,
                    rank=0,
                    communicator_size=1,
                    thread_limit=thread_limit,
                    artifact_validator=artifact_validator,
                    orphan_artifact_suffix=orphan_artifact_suffix,
                    orphan_artifact_adopter=(
                        orphan_artifact_adopter
                    ),
                )
            except Exception as exc:
                infrastructure_errors.append(
                    {
                        "task_id": task.task_id,
                        "rank": 0,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )
                break
    elif rank != 0:
        mpi = _load_mpi(required=True)
        protocol_failed = False
        while True:
            status = mpi.Status()
            message = communicator.recv(
                source=0,
                tag=mpi.ANY_TAG,
                status=status,
            )
            tag = int(status.Get_tag())
            if tag == _STOP_TAG:
                break
            if protocol_failed:
                continue
            if tag != _TASK_TAG:
                communicator.send(
                    (
                        "",
                        "infrastructure_failure",
                        "UnexpectedMPITag",
                        f"worker rank {rank} received MPI tag {tag}.",
                    ),
                    dest=0,
                    tag=_RESULT_TAG,
                )
                protocol_failed = True
                continue
            task_id = str(message)
            try:
                task = manifest.task(task_id)
                _execute_one(
                    manifest,
                    task,
                    worker,
                    run_directory=root,
                    artifact_root=artifact_root,
                    checkpoint_directory=checkpoint_directory,
                    rank=rank,
                    communicator_size=size,
                    thread_limit=thread_limit,
                    artifact_validator=artifact_validator,
                    orphan_artifact_suffix=orphan_artifact_suffix,
                    orphan_artifact_adopter=(
                        orphan_artifact_adopter
                    ),
                )
            except Exception as exc:
                communicator.send(
                    (
                        task_id,
                        "infrastructure_failure",
                        type(exc).__name__,
                        str(exc),
                    ),
                    dest=0,
                    tag=_RESULT_TAG,
                )
            else:
                communicator.send(
                    (task_id, "checkpoint_written", "", ""),
                    dest=0,
                    tag=_RESULT_TAG,
                )
    else:
        mpi = _load_mpi(required=True)
        queue = deque(task.task_id for task in pending)
        active: dict[int, str] = {}
        stopped: set[int] = set()
        abort_scheduling = False

        def stop_worker(worker_rank: int) -> None:
            if worker_rank not in stopped:
                communicator.send(
                    None, dest=worker_rank, tag=_STOP_TAG
                )
                stopped.add(worker_rank)

        for worker_rank in range(1, size):
            if queue:
                task_id = queue.popleft()
                communicator.send(task_id, dest=worker_rank, tag=_TASK_TAG)
                active[worker_rank] = task_id
            else:
                stop_worker(worker_rank)

        while active:
            status = mpi.Status()
            message = communicator.recv(
                source=mpi.ANY_SOURCE,
                tag=mpi.ANY_TAG,
                status=status,
            )
            worker_rank = int(status.Get_source())
            expected_task = active.pop(worker_rank, None)
            if expected_task is None:
                infrastructure_errors.append(
                    {
                        "task_id": "<protocol>",
                        "rank": worker_rank,
                        "error_type": "UnexpectedWorkerMessage",
                        "error_message": (
                            "received a message from a worker with no "
                            "active assignment"
                        ),
                    }
                )
                abort_scheduling = True
                stop_worker(worker_rank)
                continue
            valid_message = (
                status.Get_tag() == _RESULT_TAG
                and isinstance(message, (tuple, list))
                and len(message) == 4
            )
            if valid_message:
                task_id, state, error_type, error_message = message
            else:
                task_id = expected_task
                state = "infrastructure_failure"
                error_type = "MalformedWorkerMessage"
                error_message = repr(message)
            if task_id != expected_task:
                state = "infrastructure_failure"
                error_type = "TaskIdentityMismatch"
                error_message = (
                    f"rank {worker_rank} returned {task_id!r}; "
                    f"expected {expected_task!r}."
                )
            if state == "checkpoint_written":
                try:
                    load_task_checkpoint(
                        task_checkpoint_path(
                            manifest, expected_task, checkpoint_directory
                        ),
                        manifest=manifest,
                        artifact_root=artifact_root,
                        artifact_validator=artifact_validator,
                    )
                except Exception as exc:
                    state = "infrastructure_failure"
                    error_type = type(exc).__name__
                    error_message = str(exc)
            if state != "checkpoint_written":
                infrastructure_errors.append(
                    {
                        "task_id": expected_task,
                        "rank": worker_rank,
                        "error_type": str(error_type),
                        "error_message": str(error_message),
                    }
                )
                abort_scheduling = True

            if queue and not abort_scheduling:
                next_task = queue.popleft()
                communicator.send(next_task, dest=worker_rank, tag=_TASK_TAG)
                active[worker_rank] = next_task
            else:
                stop_worker(worker_rank)

        for worker_rank in range(1, size):
            stop_worker(worker_rank)

    report: MPIExecutionReport | None = None
    terminal_error: Exception | None = None
    if rank == 0:
        try:
            checkpoints = collect_task_checkpoints(
                manifest,
                checkpoint_directory,
                artifact_root,
                artifact_validator=artifact_validator,
                require_complete=False,
            )
            report = _build_report(
                manifest,
                execution_id=selected_execution_id,
                communicator_size=size,
                coordinator_hostname=coordinator_hostname,
                started_utc=started_utc,
                started_clock=started_clock,
                thread_limit=thread_limit,
                checkpoints=checkpoints,
                checkpoint_directory=checkpoint_directory,
                initially_complete=initial_ids,
                infrastructure_errors=infrastructure_errors,
            )
            execution_path = (
                root / "executions" / f"{selected_execution_id}.json"
            )
            write_execution_manifest(execution_path, report)
            if not report.complete:
                raise TaskExecutionError(
                    "task execution stopped before every manifest task "
                    "produced a validated terminal checkpoint.",
                    report=report,
                )
            # A final strict merge is the success boundary.
            collect_task_checkpoints(
                manifest,
                checkpoint_directory,
                artifact_root,
                artifact_validator=artifact_validator,
                require_complete=True,
            )
        except Exception as exc:
            terminal_error = exc

    terminal = {
        "ok": terminal_error is None,
        "error_type": (
            None
            if terminal_error is None
            else type(terminal_error).__name__
        ),
        "error_message": (
            None if terminal_error is None else str(terminal_error)
        ),
    }
    if size > 1:
        terminal = communicator.bcast(
            terminal if rank == 0 else None,
            root=0,
        )
    if not terminal["ok"]:
        if rank == 0 and terminal_error is not None:
            raise terminal_error
        raise TaskExecutionError(
            "rank-zero task-phase finalization failed: "
            f"{terminal['error_type']}: {terminal['error_message']}"
        )
    return report if rank == 0 else None


__all__ = [
    "ArtifactValidator",
    "DuplicateTaskCheckpointError",
    "MPIExecutionReport",
    "MPIDependencyError",
    "MissingTaskCheckpointError",
    "TaskAssignment",
    "OrphanArtifactAdopter",
    "TaskCheckpoint",
    "TaskCheckpointCorruptionError",
    "TaskCheckpointError",
    "TaskCheckpointMismatchError",
    "TaskContext",
    "TaskExecutionError",
    "TaskMPIError",
    "TaskManifest",
    "TaskManifestError",
    "TaskResult",
    "TaskSpec",
    "TaskWorker",
    "collect_task_checkpoints",
    "execute_task_manifest",
    "load_task_checkpoint",
    "load_task_manifest",
    "mpi_available",
    "serial_task_communicator",
    "task_checkpoint_path",
    "task_assignment_physical_core_evidence",
    "task_execution_mpi_origin_lineage_evidence",
    "write_execution_manifest",
    "write_task_checkpoint",
    "write_task_manifest",
]
