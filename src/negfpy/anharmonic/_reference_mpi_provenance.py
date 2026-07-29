"""Strict, read-only validation of phono3py reference-MPI evidence.

The independent phono3py generator deliberately lives outside the NEGFPy
package.  This module is the downstream verifier for its self-contained
release evidence archive.  Every member is required to be a regular file with
a normalized POSIX path. JSON is validated in-stream; each HDF5 task is copied
by content (never by archive path extraction) into a private temporary
directory, hashed, scientifically inspected, and removed before the next task.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import stat
import tarfile
import tempfile
from typing import Any, Mapping, Sequence
import unicodedata


REFERENCE_MPI_ORCHESTRATION_SCHEMA = (
    "negfpy_phono3py_si_reference_mpi_orchestration"
)
REFERENCE_MPI_ORCHESTRATION_VERSION = 3
REFERENCE_MPI_EVIDENCE_ARCHIVE = "reference-mpi-evidence.tar"
_CAMPAIGN_SCHEMA = "negfpy_phono3py_si_reference_mpi_manifest"
_EXECUTION_SCHEMA = "negfpy_phono3py_si_reference_mpi_execution"
_TASK_SCHEMA = "negfpy_phono3py_si_reference_mpi_task_record"
_ORIGIN_ASSIGNMENT_SCHEMA = (
    "negfpy_phono3py_si_reference_mpi_origin_assignment"
)
_SCRATCH_CLEANUP_SCHEMA = (
    "negfpy_phono3py_si_reference_mpi_scratch_cleanup"
)
_SCRATCH_CLEANUP_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "campaign_sha256",
        "execution_id",
        "status",
        "removed",
        "scratch_empty_after_cleanup",
        "completed_at_utc",
    }
)
_SCRATCH_REMOVED_KEYS = frozenset({"directory", "files"})
_SCRATCH_FILE_KEYS = frozenset(
    {"relative_path", "size_bytes", "sha256"}
)
_SCRATCH_DIRECTORY_PATTERN = re.compile(
    r"^\.target-grg-[0-9]{8}-rank[0-9]+-[A-Za-z0-9_-]+$"
)
_PRIVATE_TEMPORARY_PATTERN = re.compile(
    r"^\.(?P<target>.+)\.(?:[a-z0-9_]{8}|[0-9a-f]{32})\.tmp$"
)
_PRIVATE_TEMPORARY_TARGETS = {
    ".": re.compile(r"^reference-mpi-manifest\.json$"),
    "tasks": re.compile(r"^target-grg-[0-9]{8}\.json$"),
    "assignments": re.compile(
        r"^target-grg-[0-9]{8}\.[A-Za-z0-9._-]{1,64}"
        r"\.assignment\.json$"
    ),
    "failures": re.compile(r"^target-grg-[0-9]{8}\.json$"),
    "executions": re.compile(r"^[A-Za-z0-9._-]{1,64}\.json$"),
    "scratch-cleanups": re.compile(r"^[A-Za-z0-9._-]{1,64}\.json$"),
}
_PRIVATE_TEMPORARY_SCOPE_ORDER = tuple(_PRIVATE_TEMPORARY_TARGETS)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_THREAD_NAME_SET = frozenset(_THREAD_NAMES)
_PHYSICAL_CORE_BINDING_POLICY = (
    "one_physical_core_per_numerical_worker_disjoint_per_host"
)
_TASK_RECORD_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "campaign_sha256",
        "grid_point_grg",
        "status",
        "disposition",
        "origin_assignment",
        "artifact",
        "scientific",
        "execution",
        "runtime",
    }
)
_TASK_ARTIFACT_KEYS = frozenset({"name", "size_bytes", "sha256"})
_TASK_SCIENTIFIC_KEYS = frozenset(
    {
        "grid_point_grg",
        "grid_point_bzg",
        "bands",
        "n_qpoints",
        "n_bands",
        "gamma_shape",
        "target_frequency_shape",
        "common_reference_sha256",
    }
)
_TASK_EXECUTION_KEYS = frozenset(
    {
        "rank",
        "hostname",
        "started_at_utc",
        "completed_at_utc",
        "wall_seconds",
        "thread_environment",
    }
)
_ORIGIN_ASSIGNMENT_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "campaign_sha256",
        "grid_point_grg",
        "execution_id",
        "status",
        "mode",
        "rank",
        "communicator_size",
        "hostname",
        "assigned_at_utc",
        "worker_binding",
        "thread_environment",
    }
)
_EXECUTION_RECORD_KEYS = frozenset(
    {
        "schema_name",
        "schema_version",
        "execution_id",
        "campaign_sha256",
        "started_at_utc",
        "completed_at_utc",
        "wall_seconds",
        "communicator_size",
        "coordinator_rank",
        "rank_zero_numerical_tasks",
        "floating_point_mpi_reductions",
        "worker_binding_policy",
        "worker_binding_evidence",
        "canonical_assignment_order",
        "task_grid_points_grg",
        "assignment_grid_points_grg",
        "resumed_grid_points_grg",
        "computed_or_adopted_grid_points_grg",
        "exact_checkpoint_resume",
        "hostname",
        "python",
        "thread_environment",
        "scratch_cleanup",
        "assignments",
        "complete",
    }
)
_EXECUTION_PYTHON_KEYS = frozenset({"version", "executable"})
_RESUMED_ASSIGNMENT_KEYS = frozenset(
    {
        "grid_point_grg",
        "rank",
        "status",
        "wall_seconds",
        "origin_assignment",
    }
)
_SERIAL_NUMERICAL_ASSIGNMENT_KEYS = frozenset(
    {
        "grid_point_grg",
        "rank",
        "status",
        "wall_seconds",
        "artifact",
        "origin_assignment",
    }
)
_MPI_NUMERICAL_ASSIGNMENT_KEYS = frozenset(
    {
        "grid_point_grg",
        "rank",
        "status",
        "wall_seconds",
        "worker_binding",
        "artifact",
        "origin_assignment",
    }
)
_AUTHENTICATED_COMPUTE_DISPOSITIONS = frozenset(
    {
        "computed",
        "computed_after_interrupted_record_publication",
    }
)
_ADOPTED_DISPOSITION = "adopted_after_interrupted_record_publication"


class ReferenceMPIProvenanceError(RuntimeError):
    """Raised when reference-MPI orchestration evidence is not certifiable."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ReferenceMPIProvenanceError(f"{name} must be a JSON object.")
    return value


def _strict_thread_environment(value: Any, *, name: str) -> dict[str, str]:
    environment = dict(_mapping(value, name=name))
    if (
        set(environment) != _THREAD_NAME_SET
        or any(environment[variable] != "1" for variable in _THREAD_NAMES)
    ):
        raise ReferenceMPIProvenanceError(
            f"{name} must contain exactly the six pinned thread variables "
            "with value '1'."
        )
    return environment


def _private_temporary_target(scope: str, name: str) -> str | None:
    pattern = _PRIVATE_TEMPORARY_TARGETS.get(scope)
    match = _PRIVATE_TEMPORARY_PATTERN.fullmatch(name)
    if pattern is None or match is None:
        return None
    target = str(match.group("target"))
    if pattern.fullmatch(target) is None:
        return None
    if scope in {"executions", "scratch-cleanups"}:
        identifier = target.removesuffix(".json")
        if identifier in {"", ".", ".."}:
            return None
    return target


def _strict_final_output_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "\x00" in value
        or "/" in value
        or "\\" in value
        or PurePosixPath(value).name != value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ReferenceMPIProvenanceError(
            "campaign.final_output_name must be a traversal-safe basename."
        )
    return value


def _strict_scratch_cleanup_removed(
    value: Any,
    *,
    execution_id: str,
    final_output_name: str,
) -> None:
    if not isinstance(value, list):
        raise ReferenceMPIProvenanceError(
            f"execution {execution_id!r} scratch cleanup removed must be "
            "an array."
        )
    seen_directories: set[str] = set()
    last_worker_directory: str | None = None
    last_private_scope = -1
    last_staging_directory: str | None = None
    directory_phase = 0
    staging_pattern = re.compile(
        rf"^output-parent/\.{re.escape(final_output_name)}\."
        r"[0-9a-f]{32}\.tmp$"
    )
    for ordinal, item in enumerate(value):
        if (
            not isinstance(item, Mapping)
            or set(item) != _SCRATCH_REMOVED_KEYS
            or not isinstance(item.get("directory"), str)
            or not isinstance(item.get("files"), list)
        ):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} scratch cleanup "
                f"removed[{ordinal}] has an ambiguous schema."
            )
        directory = str(item["directory"])
        is_worker_scratch = (
            _SCRATCH_DIRECTORY_PATTERN.fullmatch(directory) is not None
        )
        is_private_scope = directory in _PRIVATE_TEMPORARY_TARGETS
        is_final_staging = staging_pattern.fullmatch(directory) is not None
        if (
            directory in seen_directories
            or not (
                is_worker_scratch
                or is_private_scope
                or is_final_staging
            )
        ):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} scratch cleanup "
                f"removed[{ordinal}] has an invalid directory."
            )
        if is_worker_scratch:
            if (
                directory_phase != 0
                or (
                    last_worker_directory is not None
                    and directory <= last_worker_directory
                )
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} scratch cleanup directories "
                    "are not in generator order."
                )
            last_worker_directory = directory
        elif is_private_scope:
            if directory_phase > 1:
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} scratch cleanup private "
                    "directories are not in generator order."
                )
            directory_phase = 1
            private_scope = _PRIVATE_TEMPORARY_SCOPE_ORDER.index(directory)
            if (
                private_scope <= last_private_scope
                or not item["files"]
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} scratch cleanup private "
                    "directories are not in generator order."
                )
            last_private_scope = private_scope
        else:
            directory_phase = 2
            if (
                last_staging_directory is not None
                and directory <= last_staging_directory
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} final-staging cleanup "
                    "directories are not in generator order."
                )
            last_staging_directory = directory
        seen_directories.add(directory)

        seen_files: set[str] = set()
        last_path_parts: tuple[str, ...] | None = None
        for file_ordinal, file_value in enumerate(item["files"]):
            if (
                not isinstance(file_value, Mapping)
                or set(file_value) != _SCRATCH_FILE_KEYS
                or not isinstance(file_value.get("relative_path"), str)
                or isinstance(file_value.get("size_bytes"), bool)
                or not isinstance(file_value.get("size_bytes"), int)
                or int(file_value["size_bytes"]) < 0
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} scratch cleanup file "
                    "identity has an ambiguous schema at "
                    f"removed[{ordinal}].files[{file_ordinal}]."
                )
            relative_path = str(file_value["relative_path"])
            canonical_path = PurePosixPath(relative_path)
            path_parts = canonical_path.parts
            if (
                not relative_path
                or "\x00" in relative_path
                or canonical_path.is_absolute()
                or canonical_path.as_posix() != relative_path
                or any(part in {"", ".", ".."} for part in path_parts)
                or relative_path in seen_files
                or (
                    last_path_parts is not None
                    and path_parts <= last_path_parts
                )
                or (
                    is_private_scope
                    and (
                        len(path_parts) != 1
                        or _private_temporary_target(
                            directory,
                            relative_path,
                        )
                        is None
                    )
                )
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} scratch cleanup file "
                    "identity is noncanonical at "
                    f"removed[{ordinal}].files[{file_ordinal}]."
                )
            _sha256_value(
                file_value.get("sha256"),
                name=(
                    f"execution {execution_id!r} scratch cleanup "
                    f"removed[{ordinal}].files[{file_ordinal}].sha256"
                ),
            )
            seen_files.add(relative_path)
            last_path_parts = path_parts


def _immutable_regular_file(path: Path, *, name: str) -> Any:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise ReferenceMPIProvenanceError(
            f"{name} must be an immutable regular file."
        ) from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ReferenceMPIProvenanceError(
            f"{name} must be a regular non-symlink file."
        )
    if metadata.st_nlink != 1:
        raise ReferenceMPIProvenanceError(
            f"{name} must not be hard-linked."
        )
    return metadata


def _sha256_value(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ReferenceMPIProvenanceError(
            f"{name} must be a lowercase SHA256."
        )
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_payload_sha256(value: Mapping[str, Any]) -> str:
    payload = (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _strict_json_bytes(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReferenceMPIProvenanceError(
                    f"{name} contains duplicate key {key!r}."
                )
            result[key] = value
        return result

    def reject_nonfinite(value: str) -> Any:
        raise ReferenceMPIProvenanceError(
            f"{name} contains nonfinite value {value!r}."
        )

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceMPIProvenanceError(
            f"{name} is not strict UTF-8 JSON."
        ) from exc
    return _mapping(value, name=name)


def _safe_member_name(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ReferenceMPIProvenanceError(
            f"{name} must be a normalized relative POSIX path."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise ReferenceMPIProvenanceError(
            f"{name} must be a normalized relative POSIX path."
        )
    return value


def _member_record(value: Any, *, name: str) -> dict[str, Any]:
    record = dict(_mapping(value, name=name))
    if set(record) != {"archive_member", "size_bytes", "sha256"}:
        raise ReferenceMPIProvenanceError(
            f"{name} has an incomplete or ambiguous member identity."
        )
    member = _safe_member_name(
        record["archive_member"], name=f"{name}.archive_member"
    )
    size = record["size_bytes"]
    if (
        not isinstance(size, int)
        or isinstance(size, bool)
        or size <= 0
    ):
        raise ReferenceMPIProvenanceError(
            f"{name}.size_bytes must be a positive integer."
        )
    return {
        "archive_member": member,
        "size_bytes": size,
        "sha256": _sha256_value(record["sha256"], name=f"{name}.sha256"),
    }


def _archive_record(root: Path, value: Any) -> tuple[Path, dict[str, Any]]:
    record = dict(_mapping(value, name="evidence_archive"))
    if set(record) != {"name", "size_bytes", "sha256"}:
        raise ReferenceMPIProvenanceError(
            "evidence_archive has an incomplete or ambiguous identity."
        )
    if record.get("name") != REFERENCE_MPI_EVIDENCE_ARCHIVE:
        raise ReferenceMPIProvenanceError(
            "evidence_archive has a noncanonical filename."
        )
    path = root / REFERENCE_MPI_EVIDENCE_ARCHIVE
    metadata = _immutable_regular_file(
        path,
        name="reference MPI evidence archive",
    )
    expected_size = record.get("size_bytes")
    expected_sha256 = _sha256_value(
        record.get("sha256"), name="evidence_archive.sha256"
    )
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or metadata.st_size != expected_size
        or _sha256_file(path) != expected_sha256
    ):
        raise ReferenceMPIProvenanceError(
            "reference MPI evidence archive identity changed."
        )
    return path, {
        "name": REFERENCE_MPI_EVIDENCE_ARCHIVE,
        "size_bytes": expected_size,
        "sha256": expected_sha256,
    }


def _read_archive(
    archive_path: Path,
    declared: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if len(names) != len(set(names)):
                raise ReferenceMPIProvenanceError(
                    "reference MPI evidence archive contains duplicate members."
                )
            if set(names) != set(declared):
                raise ReferenceMPIProvenanceError(
                    "reference MPI evidence archive membership differs from "
                    "the orchestration record."
                )
            for member in members:
                name = _safe_member_name(
                    member.name, name="evidence archive member"
                )
                if (
                    not member.isreg()
                    or member.issym()
                    or member.islnk()
                    or bool(member.linkname)
                ):
                    raise ReferenceMPIProvenanceError(
                        f"evidence archive member {name!r} is not a regular file."
                    )
                record = declared[name]
                if member.size != record["size_bytes"]:
                    raise ReferenceMPIProvenanceError(
                        f"evidence archive member {name!r} changed size."
                    )
                if name.endswith(".hdf5"):
                    # HDF5 members are streamed, hashed, and scientifically
                    # inspected by the independent canonical-merge pass below.
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    raise ReferenceMPIProvenanceError(
                        f"cannot read evidence archive member {name!r}."
                    )
                digest = hashlib.sha256()
                chunks: list[bytes] | None = [] if name.endswith(".json") else None
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    if chunks is not None:
                        chunks.append(block)
                if digest.hexdigest() != record["sha256"]:
                    raise ReferenceMPIProvenanceError(
                        f"evidence archive member {name!r} failed SHA256."
                    )
                if chunks is not None:
                    payloads[name] = b"".join(chunks)
    except (tarfile.TarError, OSError) as exc:
        raise ReferenceMPIProvenanceError(
            "cannot read reference MPI evidence archive."
        ) from exc
    return payloads


def _regular_bundle_file(root: Path, name: str) -> Path:
    path = root / name
    _immutable_regular_file(path, name=f"reference bundle file {name!r}")
    return path


def _file_identity(path: Path) -> dict[str, Any]:
    metadata = _immutable_regular_file(path, name=f"evidence file {path.name!r}")
    return {
        "name": path.name,
        "size_bytes": metadata.st_size,
        "sha256": _sha256_file(path),
    }


def _validate_campaign_bundle(
    root: Path,
    document: Mapping[str, Any],
    campaign: Mapping[str, Any],
    *,
    campaign_sha256: str,
    grid_points: Sequence[int],
) -> dict[str, Any]:
    """Bind the archived campaign to every actual generation input record."""

    expected_campaign_keys = {
        "schema_version",
        "algorithm",
        "floating_point_mpi_reductions",
        "serial_generator",
        "mpi_driver",
        "source_manifest",
        "source_verification",
        "force_stage",
        "reference_settings",
        "runtime_environment",
        "thread_environment",
        "task_grid_points_grg",
        "final_output_name",
    }
    generators = _mapping(document.get("generators"), name="generators")
    if (
        set(campaign) != expected_campaign_keys
        or campaign.get("schema_version")
        != REFERENCE_MPI_ORCHESTRATION_VERSION
        or campaign.get("algorithm")
        != "one_complete_phono3py_target_q_per_worker_canonical_rank0_merge"
        or campaign.get("floating_point_mpi_reductions") is not False
        or campaign.get("task_grid_points_grg") != list(grid_points)
        or generators.get("serial_scientific_generator")
        != campaign.get("serial_generator")
        or generators.get("mpi_task_and_merge_driver")
        != campaign.get("mpi_driver")
    ):
        raise ReferenceMPIProvenanceError(
            "archived campaign generators or task settings are inconsistent."
        )
    for name, value in (
        ("serial_generator", campaign["serial_generator"]),
        ("mpi_driver", campaign["mpi_driver"]),
    ):
        code = _mapping(value, name=f"campaign.{name}")
        if (
            _SHA256.fullmatch(str(code.get("script_sha256"))) is None
            or not isinstance(code.get("git_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", str(code.get("git_commit")))
            is None
            or code.get("tracked_worktree_status") != ""
            or code.get("untracked_files_omitted") is not False
        ):
            raise ReferenceMPIProvenanceError(
                f"campaign.{name} is not an exact clean generator identity."
            )

    actual_manifest = _strict_json_bytes(
        _regular_bundle_file(
            root, "input-manifest.normalized.json"
        ).read_bytes(),
        name="input-manifest.normalized.json",
    )
    actual_source = _strict_json_bytes(
        _regular_bundle_file(root, "source-verification.json").read_bytes(),
        name="source-verification.json",
    )
    actual_settings = _strict_json_bytes(
        _regular_bundle_file(root, "reference-settings.json").read_bytes(),
        name="reference-settings.json",
    )
    environment_record = _strict_json_bytes(
        _regular_bundle_file(root, "reference-environment.json").read_bytes(),
        name="reference-environment.json",
    )
    if (
        dict(actual_manifest) != campaign.get("source_manifest")
        or dict(actual_source) != campaign.get("source_verification")
        or dict(actual_settings) != campaign.get("reference_settings")
        or environment_record.get("identity")
        != campaign.get("runtime_environment")
    ):
        raise ReferenceMPIProvenanceError(
            "archived campaign source, settings, or runtime differs from the "
            "actual reference bundle."
        )
    threads = _strict_thread_environment(
        campaign.get("thread_environment"),
        name="campaign.thread_environment",
    )
    runtime_threads = _strict_thread_environment(
        _mapping(
            campaign.get("runtime_environment"),
            name="campaign.runtime_environment",
        ).get("thread_environment"),
        name="campaign.runtime_environment.thread_environment",
    )
    if (
        dict(threads) != dict(runtime_threads)
    ):
        raise ReferenceMPIProvenanceError(
            "archived campaign numerical thread environment is inconsistent."
        )

    force_stage = _mapping(campaign.get("force_stage"), name="force_stage")
    if not force_stage:
        raise ReferenceMPIProvenanceError(
            "archived campaign force-stage evidence is empty."
        )
    actual_force: dict[str, dict[str, Any]] = {}
    for filename, value in sorted(force_stage.items()):
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or filename in {".", ".."}
        ):
            raise ReferenceMPIProvenanceError(
                "campaign force-stage filename is unsafe."
            )
        expected = dict(_mapping(value, name=f"force_stage.{filename}"))
        if set(expected) != {"name", "size_bytes", "sha256"}:
            raise ReferenceMPIProvenanceError(
                f"force_stage.{filename} has an ambiguous identity."
            )
        actual = _file_identity(_regular_bundle_file(root, filename))
        if expected != actual:
            raise ReferenceMPIProvenanceError(
                f"force-stage file {filename!r} differs from the campaign."
            )
        actual_force[filename] = actual

    orchestration_path = _regular_bundle_file(
        root, "reference-mpi-orchestration.json"
    )
    on_disk_orchestration = _strict_json_bytes(
        orchestration_path.read_bytes(),
        name="reference-mpi-orchestration.json",
    )
    if dict(on_disk_orchestration) != dict(document):
        raise ReferenceMPIProvenanceError(
            "validated orchestration differs from its immutable bundle file."
        )
    stage = _strict_json_bytes(
        _regular_bundle_file(root, "reference-stage.json").read_bytes(),
        name="reference-stage.json",
    )
    runtime = _mapping(stage.get("runtime"), name="reference-stage.runtime")
    mpi_runtime = _mapping(
        runtime.get("mpi_orchestration"),
        name="reference-stage.runtime.mpi_orchestration",
    )
    archive = _mapping(document.get("evidence_archive"), name="evidence_archive")
    restart = _mapping(document.get("restart"), name="restart")
    if (
        stage.get("status") != "complete"
        or stage.get("stage") != "reference"
        or mpi_runtime.get("campaign_sha256") != campaign_sha256
        or mpi_runtime.get("orchestration_record_sha256")
        != _sha256_file(orchestration_path)
        or mpi_runtime.get("task_count") != len(grid_points)
        or mpi_runtime.get("execution_count")
        != restart.get("execution_count")
        or mpi_runtime.get("floating_point_mpi_reductions") is not False
        or mpi_runtime.get("full_q_selection") is not True
        or mpi_runtime.get("full_band_selection") is not True
        or mpi_runtime.get("evidence_archive_sha256")
        != archive.get("sha256")
        or mpi_runtime.get("restart") != restart
    ):
        raise ReferenceMPIProvenanceError(
            "reference-stage MPI runtime does not bind the exact campaign."
        )
    return {
        "input_manifest_sha256": _sha256_file(
            root / "input-manifest.normalized.json"
        ),
        "source_verification_sha256": _sha256_file(
            root / "source-verification.json"
        ),
        "reference_settings_sha256": _sha256_file(
            root / "reference-settings.json"
        ),
        "reference_environment_sha256": _sha256_file(
            root / "reference-environment.json"
        ),
        "reference_stage_sha256": _sha256_file(root / "reference-stage.json"),
        "generator_sha256": {
            "serial": campaign["serial_generator"]["script_sha256"],
            "mpi": campaign["mpi_driver"]["script_sha256"],
        },
        "force_stage": actual_force,
    }


def _attribute_values_equal(left: Any, right: Any) -> bool:
    try:
        import numpy as np

        return bool(
            np.array_equal(
                np.asarray(left),
                np.asarray(right),
                equal_nan=True,
            )
        )
    except TypeError:
        return bool(np.array_equal(np.asarray(left), np.asarray(right)))


def _hdf_attribute_json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_hdf_attribute_json_value(item) for item in value]
    if isinstance(value, list):
        return [_hdf_attribute_json_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _update_hdf_attributes_digest(digest: Any, attrs: Any) -> None:
    payload = {
        str(name): _hdf_attribute_json_value(attrs[name])
        for name in sorted(attrs.keys())
        if not str(name).startswith("mpi_reference_")
    }
    digest.update(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            + "\n"
        ).encode("utf-8")
    )


def _update_hdf_dataset_digest(digest: Any, dataset: Any) -> None:
    import numpy as np

    digest.update(str(dataset.name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(dataset.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(value)) for value in dataset.shape).encode("ascii")
    )
    digest.update(b"\0")
    _update_hdf_attributes_digest(digest, dataset.attrs)
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes(order="C"))
    elif dataset.shape[0] != 0:
        for index in range(int(dataset.shape[0])):
            digest.update(
                np.ascontiguousarray(dataset[index]).tobytes(order="C")
            )


def _update_hdf_node_digest(digest: Any, node: Any) -> None:
    import h5py

    digest.update(str(node.name).encode("utf-8"))
    digest.update(b"\0")
    _update_hdf_attributes_digest(digest, node.attrs)
    if isinstance(node, h5py.Dataset):
        _update_hdf_dataset_digest(digest, node)
    else:
        for name in sorted(node.keys()):
            _update_hdf_node_digest(digest, node[name])


def _common_reference_digest(h5: Any) -> str:
    digest = hashlib.sha256()
    _update_hdf_attributes_digest(digest, h5.attrs)
    for name in ("crystal", "grid", "fc2_random_q"):
        _update_hdf_node_digest(digest, h5[name])
    settings = h5["settings"]
    _update_hdf_attributes_digest(digest, settings.attrs)
    for name in ("mesh", "band_indices", "temperatures", "sigmas"):
        _update_hdf_node_digest(digest, settings[name])
    phonons = h5["phonons"]
    _update_hdf_attributes_digest(digest, phonons.attrs)
    for name in ("frequencies", "eigenvectors", "bz_grid_addresses"):
        _update_hdf_node_digest(digest, phonons[name])
    gamma = h5["gamma"]
    _update_hdf_attributes_digest(digest, gamma.attrs)
    _update_hdf_attributes_digest(digest, gamma["values"].attrs)
    return digest.hexdigest()


def _compare_hdf_attributes(
    left: Any,
    right: Any,
    *,
    name: str,
    ignore_left_prefix: str | None = None,
    ignored: frozenset[str] = frozenset(),
) -> None:
    left_names = {
        str(value)
        for value in left.attrs
        if value not in ignored
        and (
            ignore_left_prefix is None
            or not str(value).startswith(ignore_left_prefix)
        )
    }
    right_names = {str(value) for value in right.attrs if value not in ignored}
    if left_names != right_names:
        raise ReferenceMPIProvenanceError(
            f"{name} HDF5 attribute names differ."
        )
    for attribute in sorted(left_names):
        if not _attribute_values_equal(
            left.attrs[attribute], right.attrs[attribute]
        ):
            raise ReferenceMPIProvenanceError(
                f"{name}@{attribute} HDF5 attribute differs."
            )


def _compare_hdf_node(
    left: Any,
    right: Any,
    *,
    name: str,
    ignored_attributes: frozenset[str] = frozenset(),
) -> None:
    import h5py
    import numpy as np

    if isinstance(left, h5py.Dataset) != isinstance(right, h5py.Dataset):
        raise ReferenceMPIProvenanceError(f"{name} HDF5 node type differs.")
    _compare_hdf_attributes(
        left,
        right,
        name=name,
        ignored=ignored_attributes,
    )
    if isinstance(left, h5py.Dataset):
        if (
            left.shape != right.shape
            or left.dtype != right.dtype
            or not np.array_equal(left[...], right[...], equal_nan=True)
        ):
            raise ReferenceMPIProvenanceError(
                f"{name} HDF5 dataset differs."
            )
        return
    if set(left.keys()) != set(right.keys()):
        raise ReferenceMPIProvenanceError(
            f"{name} HDF5 child names differ."
        )
    for child in sorted(left.keys()):
        if not isinstance(left.get(child, getlink=True), h5py.HardLink) or not isinstance(
            right.get(child, getlink=True), h5py.HardLink
        ):
            raise ReferenceMPIProvenanceError(
                f"{name}/{child} must use direct HDF5 hard links."
            )
        _compare_hdf_node(
            left[child],
            right[child],
            name=f"{name}/{child}",
        )


def _copy_archive_member(
    archive: tarfile.TarFile,
    member_name: str,
    destination: Path,
    *,
    expected_sha256: str,
) -> None:
    member = archive.getmember(member_name)
    stream = archive.extractfile(member)
    if stream is None:
        raise ReferenceMPIProvenanceError(
            f"cannot read archived HDF5 member {member_name!r}."
        )
    digest = hashlib.sha256()
    with destination.open("wb") as target:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            target.write(block)
    if digest.hexdigest() != expected_sha256:
        raise ReferenceMPIProvenanceError(
            f"archived HDF5 member {member_name!r} failed SHA256."
        )


def _validate_canonical_hdf5_merge(
    archive_path: Path,
    root: Path,
    tasks: Sequence[Mapping[str, Any]],
    *,
    campaign_sha256: str,
    grid_points: Sequence[int],
    bands: Sequence[int],
    n_qpoints: int,
    n_bands: int,
) -> dict[str, Any]:
    """Independently reconstruct every merged target field from task HDF5."""

    import h5py
    import numpy as np

    merged_path = _regular_bundle_file(root, "si-phono3py-reference.hdf5")
    frequency_points: list[np.ndarray] = []
    frequency_point_attrs: dict[str, Any] | None = None
    frequency_points_present: list[bool] = []
    with h5py.File(merged_path, "r") as merged, tarfile.open(
        archive_path, mode="r:"
    ) as archive, tempfile.TemporaryDirectory(
        prefix="negfpy-reference-mpi-verify-"
    ) as temporary:
        temporary_root = Path(temporary)
        if (
            np.asarray(merged["settings/mesh"][...], dtype=np.int64).tolist()
            != list(map(int, merged["settings/mesh"][...]))
        ):
            raise ReferenceMPIProvenanceError(
                "merged reference mesh has an invalid integer representation."
            )
        if (
            tuple(int(value) for value in merged["settings/mesh"][...])
            != tuple(
                int(value)
                for value in _mapping(
                    _strict_json_bytes(
                        (root / "reference-settings.json").read_bytes(),
                        name="reference-settings.json",
                    ),
                    name="reference-settings.json",
                )["mesh"]
            )
            or np.asarray(
                merged["settings/grid_points_grg"][...], dtype=np.int64
            ).tolist()
            != list(grid_points)
            or np.asarray(
                merged["settings/band_indices"][...], dtype=np.int64
            ).tolist()
            != list(bands)
            or merged["phonons/frequencies"].shape
            != (n_qpoints, n_bands)
            or set(merged["target_q"].keys())
            != {f"grg_{point:08d}" for point in grid_points}
        ):
            raise ReferenceMPIProvenanceError(
                "merged reference HDF5 selectors or dimensions are inconsistent."
            )

        task_common_hashes: set[str] = set()
        bzg_points: list[int] = []
        for ordinal, task in enumerate(tasks):
            point = int(task["grid_point_grg"])
            member_record = _mapping(task["hdf5"], name=f"task {point}.hdf5")
            member_name = str(member_record["archive_member"])
            task_path = temporary_root / f"task-{point:08d}.hdf5"
            _copy_archive_member(
                archive,
                member_name,
                task_path,
                expected_sha256=str(member_record["sha256"]),
            )
            with h5py.File(task_path, "r") as source:
                if (
                    source.attrs.get("schema_name")
                    != "negfpy_phono3py_si_reference"
                    or int(source.attrs.get("schema_version", -1)) != 1
                    or source.attrs.get("mpi_reference_task_schema")
                    != "negfpy_phono3py_si_reference_mpi_task"
                    or int(
                        source.attrs.get(
                            "mpi_reference_task_schema_version", -1
                        )
                    )
                    != 1
                    or source.attrs.get("mpi_reference_campaign_sha256")
                    != campaign_sha256
                    or int(
                        source.attrs.get(
                            "mpi_reference_grid_point_grg", -1
                        )
                    )
                    != point
                ):
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} has invalid provenance."
                    )
                expected_root_groups = {
                    "crystal",
                    "grid",
                    "fc2_random_q",
                    "settings",
                    "phonons",
                    "gamma",
                    "target_q",
                }
                if (
                    set(source.keys()) != expected_root_groups
                    or set(merged.keys()) != expected_root_groups
                    or any(
                        not isinstance(
                            source.get(name, getlink=True),
                            h5py.HardLink,
                        )
                        or not isinstance(
                            merged.get(name, getlink=True),
                            h5py.HardLink,
                        )
                        for name in expected_root_groups
                    )
                ):
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} has an ambiguous root tree."
                    )
                runtime_text = source.attrs.get(
                    "mpi_reference_runtime_json"
                )
                if not isinstance(runtime_text, str):
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} lacks runtime provenance."
                    )
                runtime = _strict_json_bytes(
                    runtime_text.encode("utf-8"),
                    name=f"archived task HDF5 {point} runtime",
                )
                if dict(runtime) != dict(
                    _mapping(
                        task["runtime"],
                        name=f"task {point}.runtime",
                    )
                ):
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} runtime differs from its "
                        "sidecar."
                    )
                origin_text = source.attrs.get(
                    "mpi_reference_origin_assignment_json"
                )
                if not isinstance(origin_text, str) or not origin_text:
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} lacks origin provenance."
                    )
                origin_identity = _strict_json_bytes(
                    origin_text.encode("utf-8"),
                    name=f"archived task HDF5 {point} origin",
                )
                origin_member = _mapping(
                    task["origin_assignment"],
                    name=f"task {point}.origin_assignment",
                )
                if dict(origin_identity) != {
                    "relative_path": origin_member["archive_member"],
                    "size_bytes": origin_member["size_bytes"],
                    "sha256": origin_member["sha256"],
                }:
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} origin differs from "
                        "its sidecar."
                    )
                _compare_hdf_attributes(
                    source,
                    merged,
                    name=f"task {point} root",
                    ignore_left_prefix="mpi_reference_",
                )
                for group in ("crystal", "grid", "fc2_random_q"):
                    _compare_hdf_node(
                        source[group],
                        merged[group],
                        name=f"task {point}/{group}",
                    )
                _compare_hdf_attributes(
                    source["settings"],
                    merged["settings"],
                    name=f"task {point}/settings",
                )
                for dataset in ("mesh", "band_indices", "temperatures", "sigmas"):
                    _compare_hdf_node(
                        source[f"settings/{dataset}"],
                        merged[f"settings/{dataset}"],
                        name=f"task {point}/settings/{dataset}",
                    )
                source_grg = np.asarray(
                    source["settings/grid_points_grg"][...], dtype=np.int64
                )
                source_bzg = np.asarray(
                    source["settings/grid_points_bzg"][...], dtype=np.int64
                )
                if source_grg.tolist() != [point] or source_bzg.shape != (1,):
                    raise ReferenceMPIProvenanceError(
                        f"archived task HDF5 {point} target selectors differ."
                    )
                bzg_points.append(int(source_bzg[0]))

                _compare_hdf_attributes(
                    source["phonons"],
                    merged["phonons"],
                    name=f"task {point}/phonons",
                )
                for dataset in (
                    "frequencies",
                    "eigenvectors",
                    "bz_grid_addresses",
                ):
                    _compare_hdf_node(
                        source[f"phonons/{dataset}"],
                        merged[f"phonons/{dataset}"],
                        name=f"task {point}/phonons/{dataset}",
                    )
                source_target = source["phonons/target_band_frequencies"]
                merged_target = merged["phonons/target_band_frequencies"]
                _compare_hdf_attributes(
                    source_target,
                    merged_target,
                    name=f"task {point}/target_band_frequencies",
                )
                if (
                    source_target.shape != (1, len(bands))
                    or source_target.dtype != merged_target.dtype
                    or not np.array_equal(
                        source_target[0],
                        merged_target[ordinal],
                        equal_nan=True,
                    )
                ):
                    raise ReferenceMPIProvenanceError(
                        f"task {point} target frequencies do not canonically merge."
                    )

                _compare_hdf_attributes(
                    source["gamma"],
                    merged["gamma"],
                    name=f"task {point}/gamma",
                )
                source_gamma = source["gamma/values"]
                merged_gamma = merged["gamma/values"]
                _compare_hdf_attributes(
                    source_gamma,
                    merged_gamma,
                    name=f"task {point}/gamma/values",
                )
                if (
                    source_gamma.shape[:2] != merged_gamma.shape[:2]
                    or source_gamma.shape[2] != 1
                    or source_gamma.shape[3:] != merged_gamma.shape[3:]
                    or source_gamma.dtype != merged_gamma.dtype
                    or source_gamma.shape[0]
                    != source["settings/sigmas"].shape[0]
                    or source_gamma.shape[1]
                    != source["settings/temperatures"].shape[0]
                    or source_gamma.shape[3] != len(bands)
                    or not np.array_equal(
                        source_gamma[:, :, 0, :],
                        merged_gamma[:, :, ordinal, :],
                        equal_nan=True,
                    )
                ):
                    raise ReferenceMPIProvenanceError(
                        f"task {point} gamma fields do not canonically merge."
                    )
                present = "frequency_points" in source["gamma"]
                frequency_points_present.append(present)
                if present:
                    dataset = source["gamma/frequency_points"]
                    frequency_points.append(np.asarray(dataset[...]))
                    attrs = {
                        str(name): dataset.attrs[name] for name in dataset.attrs
                    }
                    if frequency_point_attrs is None:
                        frequency_point_attrs = attrs
                    elif any(
                        key not in attrs
                        or not _attribute_values_equal(value, attrs[key])
                        for key, value in frequency_point_attrs.items()
                    ) or set(attrs) != set(frequency_point_attrs):
                        raise ReferenceMPIProvenanceError(
                            "task gamma frequency-point attributes differ."
                        )

                group_name = f"grg_{point:08d}"
                if set(source["target_q"].keys()) != {group_name}:
                    raise ReferenceMPIProvenanceError(
                        f"task {point} has an ambiguous target-q group."
                    )
                source_group = source[f"target_q/{group_name}"]
                merged_group = merged[f"target_q/{group_name}"]
                if (
                    int(source_group.attrs.get("target_ordinal", -1)) != 0
                    or int(source_group.attrs.get("grid_point_grg", -1))
                    != point
                    or int(source_group.attrs.get("grid_point_bzg", -1))
                    != int(source_bzg[0])
                    or int(merged_group.attrs.get("target_ordinal", -1))
                    != ordinal
                ):
                    raise ReferenceMPIProvenanceError(
                        f"task {point} target ordinal does not canonically merge."
                    )
                _compare_hdf_node(
                    source_group,
                    merged_group,
                    name=f"task {point}/target_q/{group_name}",
                    ignored_attributes=frozenset({"target_ordinal"}),
                )
                scientific = _mapping(
                    task["scientific"], name=f"task {point}.scientific"
                )
                common_digest = _common_reference_digest(source)
                expected_scientific = {
                    "grid_point_grg": point,
                    "grid_point_bzg": int(source_bzg[0]),
                    "bands": list(bands),
                    "n_qpoints": int(
                        source["phonons/frequencies"].shape[0]
                    ),
                    "n_bands": int(
                        source["phonons/frequencies"].shape[1]
                    ),
                    "gamma_shape": list(source_gamma.shape),
                    "target_frequency_shape": list(source_target.shape),
                    "common_reference_sha256": common_digest,
                }
                if dict(scientific) != expected_scientific:
                    raise ReferenceMPIProvenanceError(
                        f"task {point} scientific sidecar differs from its HDF5."
                    )
                task_common_hashes.add(common_digest)
            task_path.unlink()

        if np.asarray(
            merged["settings/grid_points_bzg"][...], dtype=np.int64
        ).tolist() != bzg_points:
            raise ReferenceMPIProvenanceError(
                "merged BZG target order differs from archived tasks."
            )
        if any(frequency_points_present):
            if not all(frequency_points_present):
                raise ReferenceMPIProvenanceError(
                    "task gamma frequency-point presence is inconsistent."
                )
            first = frequency_points[0]
            if all(
                np.array_equal(value, first, equal_nan=True)
                for value in frequency_points[1:]
            ):
                expected_frequency_points = first
            elif first.ndim >= 1 and first.shape[0] == 1:
                expected_frequency_points = np.concatenate(
                    frequency_points, axis=0
                )
            elif first.ndim >= 3 and first.shape[-2] == 1:
                expected_frequency_points = np.concatenate(
                    frequency_points, axis=-2
                )
            else:
                raise ReferenceMPIProvenanceError(
                    "task gamma frequency points have unsupported merge layout."
                )
            if "frequency_points" not in merged["gamma"]:
                raise ReferenceMPIProvenanceError(
                    "merged gamma omits task frequency points."
                )
            merged_frequency_points = merged["gamma/frequency_points"]
            if (
                merged_frequency_points.dtype
                != expected_frequency_points.dtype
                or not np.array_equal(
                    merged_frequency_points[...],
                    expected_frequency_points,
                    equal_nan=True,
                )
            ):
                raise ReferenceMPIProvenanceError(
                    "merged gamma frequency points differ from archived tasks."
                )
            assert frequency_point_attrs is not None
            if set(merged_frequency_points.attrs) != set(
                frequency_point_attrs
            ) or any(
                not _attribute_values_equal(
                    merged_frequency_points.attrs[name],
                    frequency_point_attrs[name],
                )
                for name in frequency_point_attrs
            ):
                raise ReferenceMPIProvenanceError(
                    "merged gamma frequency-point attributes differ."
                )
        elif "frequency_points" in merged["gamma"]:
            raise ReferenceMPIProvenanceError(
                "merged gamma has frequency points absent from all tasks."
            )

        if len(task_common_hashes) != 1:
            raise ReferenceMPIProvenanceError(
                "archived task common-reference identities differ."
            )
        return {
            "common_reference_sha256": next(iter(task_common_hashes)),
            "gamma_shape": list(merged["gamma/values"].shape),
            "target_band_frequency_shape": list(
                merged["phonons/target_band_frequencies"].shape
            ),
            "grid_points_grg": list(grid_points),
            "grid_points_bzg": bzg_points,
            "effective_band_indices": list(bands),
            "n_qpoints": n_qpoints,
            "n_bands": n_bands,
        }


def _normalize_worker_binding(
    value: Any,
    *,
    expected_rank: int,
    name: str,
) -> dict[str, Any]:
    binding = _mapping(value, name=name)
    if set(binding) != {
        "rank",
        "hostname",
        "cpu_affinity",
        "cpu_affinity_topology",
        "physical_core",
    }:
        raise ReferenceMPIProvenanceError(f"{name} has an ambiguous schema.")
    rank = binding.get("rank")
    hostname = binding.get("hostname")
    affinity = binding.get("cpu_affinity")
    topology = binding.get("cpu_affinity_topology")
    physical = _mapping(
        binding.get("physical_core"), name=f"{name}.physical_core"
    )
    if (
        rank != expected_rank
        or not isinstance(hostname, str)
        or not hostname
        or not isinstance(affinity, list)
        or not affinity
        or any(
            not isinstance(cpu, int) or isinstance(cpu, bool) or cpu < 0
            for cpu in affinity
        )
        or affinity != sorted(set(affinity))
        or not isinstance(topology, list)
        or len(topology) != len(affinity)
        or set(physical) != {"package_id", "core_id"}
    ):
        raise ReferenceMPIProvenanceError(f"{name} is invalid.")
    normalized_topology: list[dict[str, int]] = []
    for cpu, item in zip(affinity, topology, strict=True):
        topology_item = _mapping(item, name=f"{name}.cpu_affinity_topology")
        if set(topology_item) != {
            "logical_cpu",
            "package_id",
            "core_id",
        }:
            raise ReferenceMPIProvenanceError(
                f"{name}.cpu_affinity_topology has an ambiguous schema."
            )
        logical_cpu = topology_item.get("logical_cpu")
        package_id = topology_item.get("package_id")
        core_id = topology_item.get("core_id")
        if (
            logical_cpu != cpu
            or not isinstance(package_id, int)
            or isinstance(package_id, bool)
            or package_id < 0
            or not isinstance(core_id, int)
            or isinstance(core_id, bool)
            or core_id < 0
        ):
            raise ReferenceMPIProvenanceError(
                f"{name}.cpu_affinity_topology is invalid."
            )
        normalized_topology.append(
            {
                "logical_cpu": logical_cpu,
                "package_id": package_id,
                "core_id": core_id,
            }
        )
    physical_cores = {
        (item["package_id"], item["core_id"])
        for item in normalized_topology
    }
    if len(physical_cores) != 1:
        raise ReferenceMPIProvenanceError(
            f"{name} does not map to exactly one physical core."
        )
    package_id, core_id = next(iter(physical_cores))
    if dict(physical) != {
        "package_id": package_id,
        "core_id": core_id,
    }:
        raise ReferenceMPIProvenanceError(
            f"{name}.physical_core is inconsistent."
        )
    return {
        "rank": rank,
        "hostname": hostname,
        "cpu_affinity": list(affinity),
        "cpu_affinity_topology": normalized_topology,
        "physical_core": {
            "package_id": package_id,
            "core_id": core_id,
        },
    }


def _execution_worker_bindings(
    assignments: Sequence[Mapping[str, Any]],
    *,
    communicator_size: int,
    execution_id: str,
) -> dict[str, Any]:
    by_worker: dict[tuple[str, int], dict[str, Any]] = {}
    occupied: dict[tuple[str, int, int], int] = {}
    for ordinal, assignment in enumerate(assignments):
        status = assignment.get("status")
        rank = assignment.get("rank")
        binding = assignment.get("worker_binding")
        if status == "resumed":
            if rank is not None or binding is not None:
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} resumed assignment claims "
                    "numerical-worker binding."
                )
            continue
        if status not in {"computed", "adopted"}:
            continue
        if communicator_size == 1:
            if binding is not None:
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} serial assignment claims MPI "
                    "worker binding."
                )
            continue
        if not isinstance(rank, int) or isinstance(rank, bool) or rank == 0:
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} numerical assignment has no "
                "nonzero worker rank."
            )
        normalized = _normalize_worker_binding(
            binding,
            expected_rank=rank,
            name=(
                f"execution {execution_id!r}.assignments[{ordinal}]"
                ".worker_binding"
            ),
        )
        worker = (str(normalized["hostname"]), rank)
        previous = by_worker.setdefault(worker, normalized)
        if previous != normalized:
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} worker affinity is inconsistent."
            )
        physical = normalized["physical_core"]
        occupied_key = (
            str(normalized["hostname"]),
            int(physical["package_id"]),
            int(physical["core_id"]),
        )
        previous_rank = occupied.setdefault(occupied_key, rank)
        if previous_rank != rank:
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} assigns distinct ranks to one "
                "physical core."
            )
    rank_bindings: list[dict[str, Any]] = []
    for key in sorted(by_worker, key=lambda item: (item[0], item[1])):
        binding = by_worker[key]
        task_ids = [
            f"target-grg-{int(assignment['grid_point_grg']):08d}"
            for assignment in assignments
            if assignment.get("status") in {"computed", "adopted"}
            and assignment.get("rank") == binding["rank"]
            and isinstance(assignment.get("worker_binding"), Mapping)
            and assignment["worker_binding"].get("hostname")
            == binding["hostname"]
        ]
        rank_bindings.append(
            {
                "hostname": binding["hostname"],
                "rank": binding["rank"],
                "logical_cpus": list(binding["cpu_affinity"]),
                "physical_core": dict(binding["physical_core"]),
                "task_ids": task_ids,
            }
        )
    return {
        "schema": "negfpy_task_physical_core_bindings",
        "schema_version": 1,
        "available": bool(rank_bindings),
        "one_physical_core_per_assignment": True,
        "consistent_affinity_per_rank": True,
        "disjoint_physical_cores_across_ranks": True,
        "unavailable_task_ids": [],
        "rank_bindings": rank_bindings,
    }


def validate_reference_mpi_evidence(
    root: str | Path,
    document: Mapping[str, Any],
    *,
    expected_mesh: Sequence[int],
    expected_grid_points: Sequence[int],
    expected_band_indices: Sequence[int],
    expected_n_qpoints: int,
    expected_n_bands: int,
    require_mpi: bool = True,
) -> dict[str, Any]:
    """Validate all archived task, execution, binding, and merge evidence."""

    unresolved_root = Path(root).expanduser()
    if unresolved_root.is_symlink():
        raise ReferenceMPIProvenanceError(
            "reference bundle root must not be a symbolic link."
        )
    base = unresolved_root.resolve(strict=True)
    if not base.is_dir():
        raise ReferenceMPIProvenanceError(
            "reference bundle root must be a regular directory."
        )
    record = dict(_mapping(document, name="reference MPI orchestration"))
    expected_keys = {
        "schema_name",
        "schema_version",
        "campaign_sha256",
        "generators",
        "campaign_manifest",
        "scheduler",
        "selection",
        "evidence_archive",
        "tasks",
        "executions",
        "restart",
        "merged_reference",
        "merge",
    }
    if (
        set(record) != expected_keys
        or record.get("schema_name") != REFERENCE_MPI_ORCHESTRATION_SCHEMA
        or record.get("schema_version")
        != REFERENCE_MPI_ORCHESTRATION_VERSION
    ):
        raise ReferenceMPIProvenanceError(
            "reference MPI orchestration schema is incomplete or unsupported."
        )
    campaign_sha256 = _sha256_value(
        record.get("campaign_sha256"), name="campaign_sha256"
    )
    scheduler = _mapping(record.get("scheduler"), name="scheduler")
    if (
        scheduler.get("parallel_boundary") != "complete_target_q"
        or scheduler.get("rank_zero_numerical_tasks") is not False
        or scheduler.get("floating_point_mpi_reductions") is not False
        or scheduler.get("canonical_merge_order")
        != "strictly_increasing_GRG"
        or scheduler.get("worker_thread_limit") != 1
        or scheduler.get("worker_binding_policy")
        != _PHYSICAL_CORE_BINDING_POLICY
    ):
        raise ReferenceMPIProvenanceError(
            "reference MPI scheduler contract is not certifiable."
        )

    mesh = [int(value) for value in expected_mesh]
    grid_points = [int(value) for value in expected_grid_points]
    bands = [int(value) for value in expected_band_indices]
    if (
        len(mesh) != 3
        or expected_n_qpoints != math.prod(mesh)
        or grid_points != list(range(expected_n_qpoints))
        or bands != list(range(expected_n_bands))
    ):
        raise ReferenceMPIProvenanceError(
            "release reference expectation is not a complete full mesh/band set."
        )
    selection = _mapping(record.get("selection"), name="selection")
    if (
        selection.get("mesh") != mesh
        or selection.get("n_qpoints") != expected_n_qpoints
        or selection.get("n_bands") != expected_n_bands
        or selection.get("grid_points_grg") != grid_points
        or selection.get("band_indices") != bands
        or selection.get("full_q_selection") is not True
        or selection.get("full_band_selection") is not True
    ):
        raise ReferenceMPIProvenanceError(
            "reference MPI selection is not exactly full-q/full-band."
        )

    archive_path, archive_identity = _archive_record(
        base, record.get("evidence_archive")
    )
    campaign_manifest = _member_record(
        record.get("campaign_manifest"), name="campaign_manifest"
    )
    if campaign_manifest["archive_member"] != "reference-mpi-manifest.json":
        raise ReferenceMPIProvenanceError(
            "campaign manifest has a noncanonical archive path."
        )

    tasks_raw = record.get("tasks")
    if not isinstance(tasks_raw, list) or len(tasks_raw) != len(grid_points):
        raise ReferenceMPIProvenanceError(
            "reference MPI task evidence is incomplete."
        )
    tasks: list[dict[str, Any]] = []
    declared_members: dict[str, Mapping[str, Any]] = {
        campaign_manifest["archive_member"]: campaign_manifest
    }
    for ordinal, value in enumerate(tasks_raw):
        task = _mapping(value, name=f"tasks[{ordinal}]")
        if set(task) != {
            "grid_point_grg",
            "hdf5",
            "record",
            "origin_assignment",
        }:
            raise ReferenceMPIProvenanceError(
                f"tasks[{ordinal}] has an ambiguous schema."
            )
        point = task.get("grid_point_grg")
        if point != grid_points[ordinal]:
            raise ReferenceMPIProvenanceError(
                "reference MPI tasks are not in exact canonical GRG order."
            )
        hdf5 = _member_record(task.get("hdf5"), name=f"tasks[{ordinal}].hdf5")
        sidecar = _member_record(
            task.get("record"), name=f"tasks[{ordinal}].record"
        )
        origin_assignment = _member_record(
            task.get("origin_assignment"),
            name=f"tasks[{ordinal}].origin_assignment",
        )
        expected_stem = f"target-grg-{point:08d}"
        if (
            hdf5["archive_member"] != f"tasks/{expected_stem}.hdf5"
            or sidecar["archive_member"] != f"tasks/{expected_stem}.json"
            or not origin_assignment["archive_member"].startswith(
                f"assignments/{expected_stem}."
            )
            or not origin_assignment["archive_member"].endswith(
                ".assignment.json"
            )
        ):
            raise ReferenceMPIProvenanceError(
                f"task {point} uses noncanonical evidence member paths."
            )
        for member_record in (hdf5, sidecar, origin_assignment):
            member = member_record["archive_member"]
            if member in declared_members:
                raise ReferenceMPIProvenanceError(
                    f"duplicate declared evidence member {member!r}."
                )
            declared_members[member] = member_record
        tasks.append(
            {
                "grid_point_grg": point,
                "hdf5": hdf5,
                "record": sidecar,
                "origin_assignment": origin_assignment,
            }
        )

    executions_raw = record.get("executions")
    if not isinstance(executions_raw, list) or not executions_raw:
        raise ReferenceMPIProvenanceError(
            "reference MPI evidence has no execution records."
        )
    executions: list[dict[str, Any]] = []
    execution_ids: list[str] = []
    for ordinal, value in enumerate(executions_raw):
        execution = _mapping(value, name=f"executions[{ordinal}]")
        if set(execution) != {
            "execution_id",
            "record",
            "scratch_cleanup",
        }:
            raise ReferenceMPIProvenanceError(
                f"executions[{ordinal}] has an ambiguous schema."
            )
        execution_id = execution.get("execution_id")
        if (
            not isinstance(execution_id, str)
            or not execution_id
            or len(execution_id) > 64
            or any(
                not (character.isalnum() or character in "._-")
                for character in execution_id
            )
            or execution_id in {".", ".."}
        ):
            raise ReferenceMPIProvenanceError(
                "reference MPI execution_id is unsafe."
            )
        sidecar = _member_record(
            execution.get("record"),
            name=f"executions[{ordinal}].record",
        )
        scratch_cleanup = _member_record(
            execution.get("scratch_cleanup"),
            name=f"executions[{ordinal}].scratch_cleanup",
        )
        if (
            sidecar["archive_member"] != f"executions/{execution_id}.json"
            or scratch_cleanup["archive_member"]
            != f"scratch-cleanups/{execution_id}.json"
        ):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} has a noncanonical archive path."
            )
        for member_record in (sidecar, scratch_cleanup):
            member = member_record["archive_member"]
            if member in declared_members:
                raise ReferenceMPIProvenanceError(
                    f"duplicate declared evidence member {member!r}."
                )
            declared_members[member] = member_record
        execution_ids.append(execution_id)
        executions.append(
            {
                "execution_id": execution_id,
                "record": sidecar,
                "scratch_cleanup": scratch_cleanup,
            }
        )
    if execution_ids != sorted(set(execution_ids)):
        raise ReferenceMPIProvenanceError(
            "reference MPI executions are not unique canonical identifiers."
        )

    json_payloads = _read_archive(archive_path, declared_members)
    manifest = _strict_json_bytes(
        json_payloads[campaign_manifest["archive_member"]],
        name="archived campaign manifest",
    )
    campaign = _mapping(
        manifest.get("campaign"), name="archived campaign manifest.campaign"
    )
    if (
        set(manifest)
        != {"schema_name", "schema_version", "campaign_sha256", "campaign"}
        or manifest.get("schema_name") != _CAMPAIGN_SCHEMA
        or manifest.get("schema_version")
        != REFERENCE_MPI_ORCHESTRATION_VERSION
        or manifest.get("campaign_sha256") != campaign_sha256
        or _canonical_payload_sha256(campaign) != campaign_sha256
    ):
        raise ReferenceMPIProvenanceError(
            "archived campaign manifest does not bind the exact campaign."
        )
    final_output_name = _strict_final_output_name(
        campaign.get("final_output_name")
    )

    for task in tasks:
        point = int(task["grid_point_grg"])
        origin_member = task["origin_assignment"]
        origin = _strict_json_bytes(
            json_payloads[origin_member["archive_member"]],
            name=f"archived task origin assignment {point}",
        )
        _strict_thread_environment(
            origin.get("thread_environment"),
            name=f"task {point}.origin.thread_environment",
        )
        origin_rank = origin.get("rank")
        origin_size = origin.get("communicator_size")
        origin_mode = origin.get("mode")
        origin_execution_id = origin.get("execution_id")
        if (
            set(origin) != _ORIGIN_ASSIGNMENT_KEYS
            or origin.get("schema_name") != _ORIGIN_ASSIGNMENT_SCHEMA
            or origin.get("schema_version") != 1
            or origin.get("campaign_sha256") != campaign_sha256
            or origin.get("grid_point_grg") != point
            or origin.get("status") != "assigned_before_numerical_work"
            or not isinstance(origin_execution_id, str)
            or not origin_execution_id
            or len(origin_execution_id) > 64
            or any(
                not (character.isalnum() or character in "._-")
                for character in origin_execution_id
            )
            or origin_execution_id in {".", ".."}
            or origin_mode not in {"serial", "mpi"}
            or isinstance(origin_rank, bool)
            or not isinstance(origin_rank, int)
            or isinstance(origin_size, bool)
            or not isinstance(origin_size, int)
            or origin_size <= 0
            or origin_rank < 0
            or origin_rank >= origin_size
            or (
                origin_mode == "serial"
                and (origin_size != 1 or origin_rank != 0)
            )
            or (
                origin_mode == "mpi"
                and (origin_size < 2 or origin_rank == 0)
            )
            or not isinstance(origin.get("hostname"), str)
            or not origin.get("hostname")
            or not isinstance(origin.get("assigned_at_utc"), str)
            or not origin.get("assigned_at_utc")
        ):
            raise ReferenceMPIProvenanceError(
                f"archived task origin assignment {point} is inconsistent."
            )
        if origin_mode == "serial":
            if origin.get("worker_binding") is not None:
                raise ReferenceMPIProvenanceError(
                    f"serial task origin assignment {point} claims binding."
                )
        else:
            normalized_origin_binding = _normalize_worker_binding(
                origin.get("worker_binding"),
                expected_rank=origin_rank,
                name=f"task {point}.origin.worker_binding",
            )
            if (
                dict(
                    _mapping(
                        origin.get("worker_binding"),
                        name=f"task {point}.origin.worker_binding",
                    )
                )
                != normalized_origin_binding
                or normalized_origin_binding["hostname"]
                != origin["hostname"]
            ):
                raise ReferenceMPIProvenanceError(
                    f"task origin assignment {point} binding is inconsistent."
                )
        sidecar = task["record"]
        payload = _strict_json_bytes(
            json_payloads[sidecar["archive_member"]],
            name=f"archived task record {point}",
        )
        artifact = _mapping(
            payload.get("artifact"), name=f"task {point}.artifact"
        )
        scientific = _mapping(
            payload.get("scientific"), name=f"task {point}.scientific"
        )
        runtime = _mapping(
            payload.get("runtime"), name=f"task {point}.runtime"
        )
        task_execution = _mapping(
            payload.get("execution"), name=f"task {point}.execution"
        )
        _strict_thread_environment(
            task_execution.get("thread_environment"),
            name=f"task {point}.execution.thread_environment",
        )
        hdf5 = task["hdf5"]
        if (
            set(payload) != _TASK_RECORD_KEYS
            or set(artifact) != _TASK_ARTIFACT_KEYS
            or set(scientific) != _TASK_SCIENTIFIC_KEYS
            or set(task_execution) != _TASK_EXECUTION_KEYS
            or not isinstance(task_execution.get("hostname"), str)
            or not task_execution.get("hostname")
            or payload.get("schema_name") != _TASK_SCHEMA
            or payload.get("schema_version") != 1
            or payload.get("campaign_sha256") != campaign_sha256
            or payload.get("grid_point_grg") != point
            or payload.get("status") != "complete"
            or payload.get("disposition")
            not in (
                _AUTHENTICATED_COMPUTE_DISPOSITIONS
                | {_ADOPTED_DISPOSITION}
            )
            or payload.get("origin_assignment")
            != {
                "relative_path": origin_member["archive_member"],
                "size_bytes": origin_member["size_bytes"],
                "sha256": origin_member["sha256"],
            }
            or artifact.get("name") != Path(hdf5["archive_member"]).name
            or artifact.get("size_bytes") != hdf5["size_bytes"]
            or artifact.get("sha256") != hdf5["sha256"]
            or scientific.get("grid_point_grg") != point
            or scientific.get("bands") != bands
            or scientific.get("n_qpoints") != expected_n_qpoints
            or scientific.get("n_bands") != expected_n_bands
            or task_execution.get("rank") != origin_rank
            or task_execution.get("hostname") != origin["hostname"]
            or task_execution.get("started_at_utc")
            != origin["assigned_at_utc"]
        ):
            raise ReferenceMPIProvenanceError(
                f"archived task record {point} is inconsistent."
            )
        task["scientific"] = dict(scientific)
        task["runtime"] = dict(runtime)
        task["task_execution"] = dict(task_execution)
        task["disposition"] = payload["disposition"]
        task["origin"] = dict(origin)

    resumed_counts: dict[str, int] = {}
    resumed_execution_ids: list[str] = []
    worker_bindings_by_execution: dict[str, dict[str, Any]] = {}
    numerically_generated_points: set[int] = set()
    tasks_by_point = {
        int(task["grid_point_grg"]): task for task in tasks
    }
    origin_assignments_by_execution: dict[
        str,
        list[dict[str, Any]],
    ] = {}
    for point, task in tasks_by_point.items():
        origin = _mapping(task["origin"], name=f"task {point}.origin")
        disposition = task.get("disposition")
        if require_mpi and disposition == _ADOPTED_DISPOSITION:
            raise ReferenceMPIProvenanceError(
                f"task {point} is an adopted legacy artifact and does not "
                "prove fresh worker-MPI computation."
            )
        if require_mpi and (
            origin.get("mode") != "mpi"
            or not isinstance(origin.get("rank"), int)
            or isinstance(origin.get("rank"), bool)
            or int(origin["rank"]) <= 0
            or not isinstance(origin.get("communicator_size"), int)
            or isinstance(origin.get("communicator_size"), bool)
            or int(origin["communicator_size"]) < 2
        ):
            raise ReferenceMPIProvenanceError(
                f"task {point} lacks fresh worker-MPI origin."
            )
        if (
            origin.get("mode") == "mpi"
            and disposition in _AUTHENTICATED_COMPUTE_DISPOSITIONS
        ):
            numerically_generated_points.add(point)
            origin_assignments_by_execution.setdefault(
                str(origin["execution_id"]),
                [],
            ).append(
                {
                    "grid_point_grg": point,
                    "rank": int(origin["rank"]),
                    "status": "computed",
                    "worker_binding": origin["worker_binding"],
                }
            )
    for execution_id, origin_assignments in (
        origin_assignments_by_execution.items()
    ):
        sizes = {
            int(tasks_by_point[int(item["grid_point_grg"])]["origin"][
                "communicator_size"
            ])
            for item in origin_assignments
        }
        if len(sizes) != 1:
            raise ReferenceMPIProvenanceError(
                f"origin execution {execution_id!r} communicator changed."
            )
        worker_bindings_by_execution[execution_id] = (
            _execution_worker_bindings(
                origin_assignments,
                communicator_size=next(iter(sizes)),
                execution_id=execution_id,
            )
        )
    for execution in executions:
        execution_id = str(execution["execution_id"])
        payload = _strict_json_bytes(
            json_payloads[execution["record"]["archive_member"]],
            name=f"archived execution {execution_id}",
        )
        execution_python = _mapping(
            payload.get("python"),
            name=f"execution {execution_id!r}.python",
        )
        _strict_thread_environment(
            payload.get("thread_environment"),
            name=f"execution {execution_id!r}.thread_environment",
        )
        if (
            set(payload) != _EXECUTION_RECORD_KEYS
            or set(execution_python) != _EXECUTION_PYTHON_KEYS
            or not isinstance(execution_python.get("version"), str)
            or not execution_python.get("version")
            or not isinstance(execution_python.get("executable"), str)
            or not execution_python.get("executable")
        ):
            raise ReferenceMPIProvenanceError(
                f"archived execution {execution_id!r} has an ambiguous schema."
            )
        cleanup_member = execution["scratch_cleanup"]
        cleanup = _strict_json_bytes(
            json_payloads[cleanup_member["archive_member"]],
            name=f"archived scratch cleanup {execution_id}",
        )
        if (
            set(cleanup) != _SCRATCH_CLEANUP_KEYS
            or cleanup.get("schema_name")
            != _SCRATCH_CLEANUP_SCHEMA
            or cleanup.get("schema_version")
            != REFERENCE_MPI_ORCHESTRATION_VERSION
            or cleanup.get("campaign_sha256") != campaign_sha256
            or cleanup.get("execution_id") != execution_id
            or cleanup.get("status") != "complete"
            or not isinstance(cleanup.get("removed"), list)
            or cleanup.get("scratch_empty_after_cleanup") is not True
            or payload.get("scratch_cleanup")
            != {
                "name": Path(
                    cleanup_member["archive_member"]
                ).name,
                "size_bytes": cleanup_member["size_bytes"],
                "sha256": cleanup_member["sha256"],
            }
        ):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} scratch cleanup is inconsistent."
            )
        _strict_scratch_cleanup_removed(
            cleanup["removed"],
            execution_id=execution_id,
            final_output_name=final_output_name,
        )
        assignments = payload.get("assignments")
        if not isinstance(assignments, list):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} assignments must be an array."
            )
        assignment_points: list[Any] = []
        statuses: list[Any] = []
        size = payload.get("communicator_size")
        for item in assignments:
            assignment = _mapping(
                item, name=f"execution {execution_id!r} assignment"
            )
            status = assignment.get("status")
            if status == "resumed":
                expected_assignment_keys = _RESUMED_ASSIGNMENT_KEYS
            elif status in {"computed", "adopted"}:
                expected_assignment_keys = (
                    _SERIAL_NUMERICAL_ASSIGNMENT_KEYS
                    if size == 1
                    else _MPI_NUMERICAL_ASSIGNMENT_KEYS
                )
            else:
                expected_assignment_keys = frozenset()
            wall_seconds = assignment.get("wall_seconds")
            if (
                set(assignment) != expected_assignment_keys
                or isinstance(wall_seconds, bool)
                or not isinstance(wall_seconds, (int, float))
                or not math.isfinite(float(wall_seconds))
                or float(wall_seconds) < 0.0
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} assignment has an "
                    "ambiguous schema."
                )
            assignment_points.append(assignment.get("grid_point_grg"))
            statuses.append(status)
            rank = assignment.get("rank")
            point = assignment.get("grid_point_grg")
            if (
                isinstance(point, int)
                and not isinstance(point, bool)
                and point in tasks_by_point
                and assignment.get("origin_assignment")
                != {
                    "relative_path": tasks_by_point[point][
                        "origin_assignment"
                    ]["archive_member"],
                    "size_bytes": tasks_by_point[point][
                        "origin_assignment"
                    ]["size_bytes"],
                    "sha256": tasks_by_point[point][
                        "origin_assignment"
                    ]["sha256"],
                }
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} task {point} origin differs "
                    "from archived provenance."
                )
            if rank is not None and (
                not isinstance(rank, int)
                or isinstance(rank, bool)
                or not isinstance(size, int)
                or rank < 0
                or rank >= size
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} has an invalid worker rank."
                )
            if (
                isinstance(size, int)
                and size > 1
                and assignment.get("status") != "resumed"
                and rank == 0
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} assigned numerical work to "
                    "rank zero."
                )
            if assignment.get("status") in {"computed", "adopted"}:
                if (
                    not isinstance(point, int)
                    or isinstance(point, bool)
                    or point not in tasks_by_point
                ):
                    raise ReferenceMPIProvenanceError(
                        f"execution {execution_id!r} has an invalid task point."
                    )
                hdf5 = tasks_by_point[point]["hdf5"]
                expected_artifact = {
                    "name": Path(hdf5["archive_member"]).name,
                    "size_bytes": hdf5["size_bytes"],
                    "sha256": hdf5["sha256"],
                }
                if assignment.get("artifact") != expected_artifact:
                    raise ReferenceMPIProvenanceError(
                        f"execution {execution_id!r} assignment artifact does "
                        "not match the archived task HDF5."
                    )
                task_execution = _mapping(
                    tasks_by_point[point]["task_execution"],
                    name=f"task {point}.execution",
                )
                worker_binding = _mapping(
                    assignment.get("worker_binding"),
                    name=(
                        f"execution {execution_id!r} task {point} "
                        "worker_binding"
                    ),
                )
                if assignment.get("status") == "computed" and (
                    tasks_by_point[point].get("disposition") != "computed"
                    or task_execution.get("rank") != rank
                    or task_execution.get("hostname")
                    != worker_binding.get("hostname")
                    or tasks_by_point[point]["origin"].get(
                        "execution_id"
                    )
                    != execution_id
                    or tasks_by_point[point]["origin"].get("rank")
                    != rank
                    or tasks_by_point[point]["origin"].get(
                        "communicator_size"
                    )
                    != size
                ):
                    raise ReferenceMPIProvenanceError(
                        f"execution {execution_id!r} fresh task {point} lineage "
                        "differs from its task sidecar."
                    )
                if assignment.get("status") == "adopted" and (
                    tasks_by_point[point].get("disposition")
                    != "adopted_after_interrupted_record_publication"
                ):
                    raise ReferenceMPIProvenanceError(
                        f"execution {execution_id!r} task {point} adoption "
                        "lineage differs from its task sidecar."
                    )
            elif (
                assignment.get("status") == "resumed"
                and "artifact" in assignment
            ):
                raise ReferenceMPIProvenanceError(
                    f"execution {execution_id!r} resumed assignment claims a "
                    "fresh artifact."
                )
        resumed = [
            point
            for point, status in zip(
                assignment_points, statuses, strict=True
            )
            if status == "resumed"
        ]
        computed = [
            point
            for point, status in zip(
                assignment_points, statuses, strict=True
            )
            if status in {"computed", "adopted"}
        ]
        if isinstance(size, int) and not isinstance(size, bool) and size >= 1:
            worker_binding_evidence = _execution_worker_bindings(
                assignments,
                communicator_size=size,
                execution_id=execution_id,
            )
        else:
            worker_binding_evidence = {}
        if (
            payload.get("schema_name") != _EXECUTION_SCHEMA
            or payload.get("schema_version")
            != REFERENCE_MPI_ORCHESTRATION_VERSION
            or payload.get("execution_id") != execution_id
            or payload.get("campaign_sha256") != campaign_sha256
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < (2 if require_mpi else 1)
            or payload.get("coordinator_rank") != 0
            or payload.get("rank_zero_numerical_tasks") is not (size == 1)
            or payload.get("floating_point_mpi_reductions") is not False
            or payload.get("worker_binding_policy")
            != _PHYSICAL_CORE_BINDING_POLICY
            or payload.get("worker_binding_evidence")
            != worker_binding_evidence
            or payload.get("canonical_assignment_order")
            != "strictly_increasing_GRG"
            or payload.get("task_grid_points_grg") != grid_points
            or payload.get("assignment_grid_points_grg") != grid_points
            or assignment_points != grid_points
            or any(
                status not in {"computed", "adopted", "resumed"}
                for status in statuses
            )
            or payload.get("resumed_grid_points_grg") != resumed
            or payload.get("computed_or_adopted_grid_points_grg")
            != computed
            or payload.get("exact_checkpoint_resume") is not bool(resumed)
            or payload.get("complete") is not True
        ):
            raise ReferenceMPIProvenanceError(
                f"archived execution {execution_id!r} is inconsistent."
            )
        resumed_counts[execution_id] = len(resumed)
        previous_binding = worker_bindings_by_execution.get(execution_id)
        if (
            previous_binding is not None
            and previous_binding != worker_binding_evidence
        ):
            raise ReferenceMPIProvenanceError(
                f"execution {execution_id!r} worker binding differs from "
                "durable task origins."
            )
        worker_bindings_by_execution[execution_id] = worker_binding_evidence
        if resumed:
            resumed_execution_ids.append(execution_id)

    if require_mpi and numerically_generated_points != set(grid_points):
        raise ReferenceMPIProvenanceError(
            "archived executions do not prove a fresh, bound MPI computation "
            "of every target q-point."
        )

    restart = _mapping(record.get("restart"), name="restart")
    expected_restart = {
        "exact_checkpoint_resume_supported": True,
        "execution_count": len(executions),
        "execution_ids": execution_ids,
        "executions_with_resumed_tasks": resumed_execution_ids,
        "resumed_task_count_by_execution": resumed_counts,
        "total_resumed_task_assignments": sum(resumed_counts.values()),
    }
    if dict(restart) != expected_restart:
        raise ReferenceMPIProvenanceError(
            "reference MPI restart facts differ from archived executions."
        )

    bundle_identity = _validate_campaign_bundle(
        base,
        record,
        campaign,
        campaign_sha256=campaign_sha256,
        grid_points=grid_points,
    )
    canonical_merge = _validate_canonical_hdf5_merge(
        archive_path,
        base,
        tasks,
        campaign_sha256=campaign_sha256,
        grid_points=grid_points,
        bands=bands,
        n_qpoints=expected_n_qpoints,
        n_bands=expected_n_bands,
    )
    merged_reference = dict(
        _mapping(record.get("merged_reference"), name="merged_reference")
    )
    actual_merged_reference = _file_identity(
        _regular_bundle_file(base, "si-phono3py-reference.hdf5")
    )
    if (
        set(merged_reference) != {"name", "size_bytes", "sha256"}
        or merged_reference != actual_merged_reference
    ):
        raise ReferenceMPIProvenanceError(
            "merged reference identity differs from the actual HDF5."
        )
    merge = dict(_mapping(record.get("merge"), name="merge"))
    if merge != canonical_merge:
        raise ReferenceMPIProvenanceError(
            "declared merge summary differs from the independently reconstructed "
            "task merge."
        )

    members = [
        dict(declared_members[name]) for name in sorted(declared_members)
    ]
    return {
        "campaign_sha256": campaign_sha256,
        "evidence_archive": {
            "path": str(archive_path),
            **archive_identity,
        },
        "evidence_members": members,
        "task_count": len(tasks),
        "execution_count": len(executions),
        "execution_ids": execution_ids,
        "restart": expected_restart,
        "worker_bindings_by_execution": worker_bindings_by_execution,
        "bundle_identity": bundle_identity,
        "merged_reference": actual_merged_reference,
        "canonical_merge": canonical_merge,
        "full_q_selection": True,
        "full_band_selection": True,
        "floating_point_mpi_reductions": False,
    }


__all__ = [
    "REFERENCE_MPI_EVIDENCE_ARCHIVE",
    "REFERENCE_MPI_ORCHESTRATION_SCHEMA",
    "REFERENCE_MPI_ORCHESTRATION_VERSION",
    "ReferenceMPIProvenanceError",
    "validate_reference_mpi_evidence",
]
