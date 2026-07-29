#!/usr/bin/env python3
"""Generate the independent phono3py Si reference with task-level MPI.

This is an orchestration layer around
``generate_phono3py_si_reference.py``.  It deliberately imports no NEGFPy
module and performs no NEGFPy interaction or linewidth calculation.  Each
worker asks the pinned phono3py runtime to calculate one complete target
GRG point, including its full internal q-prime and branch sums.  Rank zero
then merges the immutable one-target files in canonical GRG order without
an MPI floating-point reduction.

The existing serial generator and its output schema remain unchanged.  The
MPI generator emits the same authenticated reference HDF5 and generation
records, plus one checksummed orchestration record.  A run directory holds
the immutable per-target artifacts and permits an exact restart with a
different MPI worker count.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import hashlib
import json
import math
import os
import platform
import re
import secrets
import shutil
import socket
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

try:
    from scripts import generate_phono3py_si_reference as serial
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    import generate_phono3py_si_reference as serial  # type: ignore[no-redef]


MPI_REFERENCE_SCHEMA_VERSION = 3
MPI_MANIFEST_FILENAME = "reference-mpi-manifest.json"
MPI_ORCHESTRATION_FILENAME = "reference-mpi-orchestration.json"
MPI_EVIDENCE_ARCHIVE_FILENAME = "reference-mpi-evidence.tar"
_TASK_SCHEMA_NAME = "negfpy_phono3py_si_reference_mpi_task"
_TASK_SCHEMA_VERSION = 1
_ORIGIN_ASSIGNMENT_SCHEMA = (
    "negfpy_phono3py_si_reference_mpi_origin_assignment"
)
_ORIGIN_ASSIGNMENT_SUFFIX = ".assignment.json"
_SCRATCH_CLEANUP_SCHEMA = (
    "negfpy_phono3py_si_reference_mpi_scratch_cleanup"
)
_SCRATCH_DIRECTORY_PATTERN = re.compile(
    r"^\.target-grg-[0-9]{8}-rank[0-9]+-[A-Za-z0-9_-]+$"
)
_CAMPAIGN_EXECUTION_LOCK_FILENAME = "campaign-execution.lock"
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
_EXECUTION_RECORD_KEYS = {
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
_COMPUTED_ASSIGNMENT_KEYS = {
    "grid_point_grg",
    "rank",
    "status",
    "wall_seconds",
    "artifact",
    "origin_assignment",
}
_RESUMED_ASSIGNMENT_KEYS = {
    "grid_point_grg",
    "rank",
    "status",
    "wall_seconds",
    "origin_assignment",
}
_TASK_PREFIX = "target-grg-"
_TASK_SUFFIX = ".hdf5"
_TASK_RECORD_SUFFIX = ".json"
_THREAD_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SHA256_HEX = frozenset("0123456789abcdef")


class ReferenceMPIError(RuntimeError):
    """Raised when an MPI reference campaign cannot be authenticated."""


class _Communicator(Protocol):
    """The small mpi4py communicator surface used by the scheduler."""

    def Get_rank(self) -> int:
        ...

    def Get_size(self) -> int:
        ...

    def bcast(self, value: Any, root: int = 0) -> Any:
        ...

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        ...

    def recv(
        self,
        source: int = -1,
        tag: int = -1,
        status: Any | None = None,
    ) -> Any:
        ...

    def Barrier(self) -> None:
        ...


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _validate_final_output_name(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or PurePosixPath(value).name != value
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise ReferenceMPIError(
            f"{name} must be a single safe basename without separators or "
            "control characters."
        )
    return value


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
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
        raise ReferenceMPIError(f"{name} must be a lowercase SHA256.")
    return value


def _regular_file(path: Path, *, name: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ReferenceMPIError(f"{name} must be a regular, non-symlink file.")
    return path


def _regular_single_link_file(path: Path, *, name: str) -> Path:
    """Require an immutable evidence file with no alternate hard-link name."""

    try:
        details = os.lstat(path)
    except OSError as exc:
        raise ReferenceMPIError(
            f"{name} must be a regular, single-link non-symlink file."
        ) from exc
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1:
        raise ReferenceMPIError(
            f"{name} must be a regular, single-link non-symlink file."
        )
    return path


def _require_no_symlink_path_components(path: Path, *, name: str) -> Path:
    """Reject every existing symlink component without resolving the path."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            details = os.lstat(current)
        except FileNotFoundError:
            break
        except OSError as exc:
            raise ReferenceMPIError(
                f"Unable to authenticate {name} path component {current}."
            ) from exc
        if stat.S_ISLNK(details.st_mode):
            raise ReferenceMPIError(
                f"{name} path components must not be symbolic links: {current}."
            )
        if current != absolute and not stat.S_ISDIR(details.st_mode):
            raise ReferenceMPIError(
                f"{name} has a non-directory parent component: {current}."
            )
    return absolute


def _open_directory_nofollow(path: Path, *, name: str) -> int:
    absolute = _require_no_symlink_path_components(path, name=name)
    required = ("O_DIRECTORY", "O_NOFOLLOW")
    if any(not hasattr(os, flag) for flag in required):
        raise ReferenceMPIError(
            "Certified reference generation requires Linux no-follow "
            "directory descriptors."
        )
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(absolute, flags)
    except OSError as exc:
        raise ReferenceMPIError(
            f"{name} must be an existing non-symlink directory."
        ) from exc
    opened = os.fstat(descriptor)
    try:
        visible = os.lstat(absolute)
    except OSError:
        os.close(descriptor)
        raise
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(visible.st_mode)
        or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        os.close(descriptor)
        raise ReferenceMPIError(f"{name} changed while it was opened.")
    return descriptor


def _verify_directory_descriptor(
    path: Path,
    descriptor: int,
    *,
    name: str,
) -> None:
    _require_no_symlink_path_components(path, name=name)
    opened = os.fstat(descriptor)
    try:
        visible = os.lstat(path)
    except OSError as exc:
        raise ReferenceMPIError(f"{name} changed while in use.") from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or not stat.S_ISDIR(visible.st_mode)
        or (opened.st_dev, opened.st_ino) != (visible.st_dev, visible.st_ino)
    ):
        raise ReferenceMPIError(f"{name} changed while in use.")


class _CampaignExecutionLock:
    """Fail-fast advisory lock guarding one campaign's mutable run graph."""

    def __init__(
        self,
        *,
        run_dir: Path,
        run_descriptor: int,
        lock_descriptor: int,
        run_identity: tuple[int, int],
        lock_identity: tuple[int, int],
    ) -> None:
        self.run_dir = Path(os.path.abspath(os.fspath(run_dir)))
        self._run_descriptor = run_descriptor
        self._lock_descriptor = lock_descriptor
        self._run_identity = run_identity
        self._lock_identity = lock_identity
        self._closed = False

    @classmethod
    def acquire(cls, run_dir: Path) -> _CampaignExecutionLock:
        run_descriptor = _open_directory_nofollow(
            run_dir,
            name="run_dir",
        )
        flags = os.O_RDWR | os.O_CREAT | os.O_CLOEXEC
        if not hasattr(os, "O_NOFOLLOW"):
            os.close(run_descriptor)
            raise ReferenceMPIError(
                "Certified campaign locking requires Linux O_NOFOLLOW."
            )
        flags |= os.O_NOFOLLOW
        try:
            lock_descriptor = os.open(
                _CAMPAIGN_EXECUTION_LOCK_FILENAME,
                flags,
                0o600,
                dir_fd=run_descriptor,
            )
        except OSError as exc:
            os.close(run_descriptor)
            raise ReferenceMPIError(
                "Campaign execution lock must be a regular, single-link "
                "non-symlink file."
            ) from exc
        try:
            lock_details = os.fstat(lock_descriptor)
            if (
                not stat.S_ISREG(lock_details.st_mode)
                or lock_details.st_nlink != 1
            ):
                raise ReferenceMPIError(
                    "Campaign execution lock must be a regular, single-link "
                    "non-symlink file."
                )
            try:
                fcntl.flock(
                    lock_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
            except OSError as exc:
                if exc.errno in {errno.EACCES, errno.EAGAIN}:
                    raise ReferenceMPIError(
                        "Another campaign execution is already active."
                    ) from exc
                raise ReferenceMPIError(
                    "Campaign execution lock acquisition failed."
                ) from exc
            run_details = os.fstat(run_descriptor)
            result = cls(
                run_dir=run_dir,
                run_descriptor=run_descriptor,
                lock_descriptor=lock_descriptor,
                run_identity=(run_details.st_dev, run_details.st_ino),
                lock_identity=(lock_details.st_dev, lock_details.st_ino),
            )
            result.verify()
            return result
        except Exception:
            os.close(lock_descriptor)
            os.close(run_descriptor)
            raise

    @property
    def run_descriptor(self) -> int:
        self.verify()
        return self._run_descriptor

    def verify(self) -> None:
        if self._closed:
            raise ReferenceMPIError("Campaign execution lock is closed.")
        _require_no_symlink_path_components(self.run_dir, name="run_dir")
        run_details = os.fstat(self._run_descriptor)
        lock_details = os.fstat(self._lock_descriptor)
        try:
            visible_run = os.lstat(self.run_dir)
            visible_lock = os.stat(
                _CAMPAIGN_EXECUTION_LOCK_FILENAME,
                dir_fd=self._run_descriptor,
                follow_symlinks=False,
            )
        except OSError as exc:
            raise ReferenceMPIError(
                "Campaign execution lock path changed while held."
            ) from exc
        if (
            not stat.S_ISDIR(visible_run.st_mode)
            or (visible_run.st_dev, visible_run.st_ino) != self._run_identity
            or (run_details.st_dev, run_details.st_ino) != self._run_identity
            or not stat.S_ISREG(lock_details.st_mode)
            or lock_details.st_nlink != 1
            or not stat.S_ISREG(visible_lock.st_mode)
            or visible_lock.st_nlink != 1
            or (lock_details.st_dev, lock_details.st_ino)
            != self._lock_identity
            or (visible_lock.st_dev, visible_lock.st_ino)
            != self._lock_identity
        ):
            raise ReferenceMPIError(
                "Campaign execution lock or run directory changed while held."
            )

    def close(self) -> None:
        if self._closed:
            return
        try:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self._lock_descriptor)
            os.close(self._run_descriptor)
            self._closed = True

    def __enter__(self) -> _CampaignExecutionLock:  # noqa: PYI034
        self.verify()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


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


def _open_owned_run_subdirectory(
    lock: _CampaignExecutionLock,
    scope: str,
) -> int:
    lock.verify()
    if scope == ".":
        return os.dup(lock.run_descriptor)
    if scope not in _PRIVATE_TEMPORARY_TARGETS:
        raise ReferenceMPIError(f"Unknown owned run directory {scope!r}.")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        descriptor = os.open(
            scope,
            flags,
            dir_fd=lock.run_descriptor,
        )
        visible = os.stat(
            scope,
            dir_fd=lock.run_descriptor,
            follow_symlinks=False,
        )
    except OSError as exc:
        raise ReferenceMPIError(
            f"Owned run directory {scope!r} is unavailable or unsafe."
        ) from exc
    opened = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(visible.st_mode)
        or not stat.S_ISDIR(opened.st_mode)
        or (visible.st_dev, visible.st_ino)
        != (opened.st_dev, opened.st_ino)
    ):
        os.close(descriptor)
        raise ReferenceMPIError(
            f"Owned run directory {scope!r} changed while opened."
        )
    return descriptor


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        block = os.read(descriptor, 1024 * 1024)
        if not block:
            break
        digest.update(block)
    return digest.hexdigest()


def _cleanup_private_atomic_temporaries(
    lock: _CampaignExecutionLock,
    *,
    scopes: Sequence[str],
) -> list[dict[str, Any]]:
    """Remove only exact generator-owned atomic-publication residues."""

    removed: list[dict[str, Any]] = []
    for scope in scopes:
        descriptor = _open_owned_run_subdirectory(lock, scope)
        files: list[dict[str, Any]] = []
        try:
            for entry in sorted(
                os.scandir(descriptor),
                key=lambda item: item.name,
            ):
                target = _private_temporary_target(scope, entry.name)
                looks_private = entry.name.startswith(".") and entry.name.endswith(
                    ".tmp"
                )
                if target is None:
                    if looks_private:
                        raise ReferenceMPIError(
                            "Suspicious atomic-publication residue in "
                            f"{scope!r}: {entry.name!r}."
                        )
                    continue
                details = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                ):
                    raise ReferenceMPIError(
                        "Atomic-publication residue must be a regular, "
                        f"single-link non-symlink file: {scope}/{entry.name}."
                    )
                flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
                temporary_descriptor = os.open(
                    entry.name,
                    flags,
                    dir_fd=descriptor,
                )
                try:
                    opened = os.fstat(temporary_descriptor)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_nlink != 1
                        or (opened.st_dev, opened.st_ino)
                        != (details.st_dev, details.st_ino)
                    ):
                        raise ReferenceMPIError(
                            "Atomic-publication residue changed during cleanup."
                        )
                    files.append(
                        {
                            "relative_path": entry.name,
                            "size_bytes": opened.st_size,
                            "sha256": _sha256_descriptor(
                                temporary_descriptor
                            ),
                        }
                    )
                finally:
                    os.close(temporary_descriptor)
                current = os.stat(
                    entry.name,
                    dir_fd=descriptor,
                    follow_symlinks=False,
                )
                if (
                    not stat.S_ISREG(current.st_mode)
                    or current.st_nlink != 1
                    or (current.st_dev, current.st_ino)
                    != (details.st_dev, details.st_ino)
                ):
                    raise ReferenceMPIError(
                        "Atomic-publication residue changed before unlink."
                    )
                os.unlink(entry.name, dir_fd=descriptor)
            if files:
                removed.append({"directory": scope, "files": files})
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    lock.verify()
    return removed


def _inspect_scratch_tree(
    descriptor: int,
    *,
    prefix: Path = Path(),
    require_single_link_files: bool = False,
) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for entry in sorted(os.scandir(descriptor), key=lambda item: item.name):
        details = entry.stat(follow_symlinks=False)
        relative = prefix / entry.name
        if stat.S_ISLNK(details.st_mode):
            raise ReferenceMPIError(
                f"Unsafe symbolic link in worker scratch: {relative}."
            )
        if stat.S_ISDIR(details.st_mode):
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            child_descriptor = os.open(
                entry.name,
                flags,
                dir_fd=descriptor,
            )
            try:
                opened = os.fstat(child_descriptor)
                if (opened.st_dev, opened.st_ino) != (
                    details.st_dev,
                    details.st_ino,
                ):
                    raise ReferenceMPIError(
                        "Worker scratch directory changed during inspection."
                    )
                files.extend(
                    _inspect_scratch_tree(
                        child_descriptor,
                        prefix=relative,
                        require_single_link_files=require_single_link_files,
                    )
                )
            finally:
                os.close(child_descriptor)
            continue
        if not stat.S_ISREG(details.st_mode):
            raise ReferenceMPIError(
                f"Unsafe non-regular worker scratch entry: {relative}."
            )
        if require_single_link_files and details.st_nlink != 1:
            raise ReferenceMPIError(
                f"Unsafe hard-linked owned cleanup entry: {relative}."
            )
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
        child_descriptor = os.open(entry.name, flags, dir_fd=descriptor)
        try:
            opened = os.fstat(child_descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (
                    require_single_link_files
                    and opened.st_nlink != 1
                )
                or (opened.st_dev, opened.st_ino)
                != (details.st_dev, details.st_ino)
            ):
                raise ReferenceMPIError(
                    "Worker scratch file changed during inspection."
                )
            files.append(
                {
                    "relative_path": relative.as_posix(),
                    "size_bytes": opened.st_size,
                    "sha256": _sha256_descriptor(child_descriptor),
                }
            )
        finally:
            os.close(child_descriptor)
    return files


def _final_staging_pattern(campaign: PreparedCampaign) -> re.Pattern[str]:
    final_output_name = _authenticated_final_output_name(campaign)
    return re.compile(
        rf"^\.{re.escape(final_output_name)}\."
        r"[0-9a-f]{32}\.tmp$"
    )


def _final_staging_evidence_directory(
    campaign: PreparedCampaign,
    staging_name: str,
) -> str | None:
    if _final_staging_pattern(campaign).fullmatch(staging_name) is None:
        return None
    return f"output-parent/{staging_name}"


def _cleanup_abandoned_final_staging(
    campaign: PreparedCampaign,
    lock: _CampaignExecutionLock,
) -> list[dict[str, Any]]:
    """Remove only canonical abandoned staging owned by this output target."""

    lock.verify()
    final_output_name = _authenticated_final_output_name(campaign)
    parent = _require_no_symlink_path_components(
        campaign.output_dir.parent,
        name="output_dir parent",
    )
    if not parent.exists():
        return []
    parent_descriptor = _open_directory_nofollow(
        parent,
        name="output_dir parent",
    )
    pattern = _final_staging_pattern(campaign)
    prefix = f".{final_output_name}."
    removed: list[dict[str, Any]] = []
    identities: list[tuple[str, tuple[int, int]]] = []
    try:
        for entry in sorted(
            os.scandir(parent_descriptor),
            key=lambda item: item.name,
        ):
            canonical = pattern.fullmatch(entry.name) is not None
            looks_owned = (
                entry.name.startswith(prefix)
                and entry.name.endswith(".tmp")
            )
            if not canonical:
                if looks_owned:
                    raise ReferenceMPIError(
                        "Suspicious abandoned final-staging near-match: "
                        f"{entry.name!r}."
                    )
                continue
            details = entry.stat(follow_symlinks=False)
            if not stat.S_ISDIR(details.st_mode):
                raise ReferenceMPIError(
                    "Abandoned final staging must be a non-symlink directory: "
                    f"{entry.name!r}."
                )
            flags = (
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
            )
            descriptor = os.open(
                entry.name,
                flags,
                dir_fd=parent_descriptor,
            )
            try:
                opened = os.fstat(descriptor)
                identity = (opened.st_dev, opened.st_ino)
                if identity != (details.st_dev, details.st_ino):
                    raise ReferenceMPIError(
                        "Abandoned final staging changed during preflight."
                    )
                files = _inspect_scratch_tree(
                    descriptor,
                    require_single_link_files=True,
                )
            finally:
                os.close(descriptor)
            evidence_directory = _final_staging_evidence_directory(
                campaign,
                entry.name,
            )
            assert evidence_directory is not None
            removed.append(
                {"directory": evidence_directory, "files": files}
            )
            identities.append((entry.name, identity))
        for name, identity in identities:
            _remove_directory_tree_at(
                parent_descriptor,
                name,
                expected_identity=identity,
            )
        os.fsync(parent_descriptor)
        _verify_directory_descriptor(
            parent,
            parent_descriptor,
            name="output_dir parent",
        )
    finally:
        os.close(parent_descriptor)
    lock.verify()
    return removed


def _remove_directory_tree_at(
    parent_descriptor: int,
    name: str,
    *,
    expected_identity: tuple[int, int],
) -> None:
    """Unlink one owned directory tree without following path components."""

    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != expected_identity:
            raise ReferenceMPIError(
                "Owned directory changed before recursive cleanup."
            )
        for entry in list(os.scandir(descriptor)):
            details = entry.stat(follow_symlinks=False)
            if stat.S_ISDIR(details.st_mode):
                _remove_directory_tree_at(
                    descriptor,
                    entry.name,
                    expected_identity=(details.st_dev, details.st_ino),
                )
            else:
                os.unlink(entry.name, dir_fd=descriptor)
    finally:
        os.close(descriptor)
    current = os.stat(name, dir_fd=parent_descriptor, follow_symlinks=False)
    if (
        not stat.S_ISDIR(current.st_mode)
        or (current.st_dev, current.st_ino) != expected_identity
    ):
        raise ReferenceMPIError(
            "Owned directory changed before final removal."
        )
    os.rmdir(name, dir_fd=parent_descriptor)


def _remove_owned_worker_scratch(
    run_dir: Path,
    scratch: Path,
    *,
    expected_identity: tuple[int, int],
) -> None:
    if _SCRATCH_DIRECTORY_PATTERN.fullmatch(scratch.name) is None:
        raise ReferenceMPIError("Refusing to remove noncanonical worker scratch.")
    run_descriptor = _open_directory_nofollow(run_dir, name="run_dir")
    try:
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        scratch_root_descriptor = os.open(
            "scratch",
            flags,
            dir_fd=run_descriptor,
        )
        try:
            try:
                current = os.stat(
                    scratch.name,
                    dir_fd=scratch_root_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            if (
                not stat.S_ISDIR(current.st_mode)
                or (current.st_dev, current.st_ino) != expected_identity
            ):
                raise ReferenceMPIError(
                    "Worker scratch path changed before final cleanup."
                )
            _remove_directory_tree_at(
                scratch_root_descriptor,
                scratch.name,
                expected_identity=expected_identity,
            )
            os.fsync(scratch_root_descriptor)
        finally:
            os.close(scratch_root_descriptor)
    finally:
        os.close(run_descriptor)


def _validate_thread_environment(value: Any, *, name: str) -> dict[str, str]:
    expected = {thread_name: "1" for thread_name in _THREAD_NAMES}
    if not isinstance(value, Mapping) or dict(value) != expected:
        raise ReferenceMPIError(
            f"{name} must contain exactly the six pinned thread controls."
        )
    return expected


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_file_exclusive(source: Path, target: Path) -> None:
    """Publish a same-filesystem temporary file without replacement."""

    _regular_file(source, name="temporary artifact")
    if source.stat().st_size == 0:
        raise ReferenceMPIError("Temporary artifact is empty.")
    target.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, target)
    _fsync_directory(target.parent)


def _write_json_exclusive(path: Path, value: Any) -> None:
    serial._publish_bytes_exclusive(path, _pretty_json_bytes(value))


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    _regular_file(path, name=name)

    def reject_nonfinite(value: str) -> Any:
        raise ReferenceMPIError(
            f"{name} contains nonfinite JSON value {value!r}."
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=serial._reject_duplicate_keys,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReferenceMPIError(f"{name} is not strict UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ReferenceMPIError(f"{name} must contain a JSON object.")
    return value


def _git_output(arguments: Sequence[str], *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _orchestration_code_identity() -> dict[str, Any]:
    script = Path(__file__).resolve()
    root_text = _git_output(
        ("rev-parse", "--show-toplevel"),
        cwd=script.parent,
    )
    result: dict[str, Any] = {
        "script_path": str(script),
        "script_sha256": _sha256_file(script),
    }
    if root_text is None:
        result.update(
            {
                "git_root": None,
                "git_commit": None,
                "tracked_worktree_status": None,
                "untracked_files_omitted": False,
            }
        )
        return result
    root = Path(root_text).resolve()
    result.update(
        {
            "git_root": str(root),
            "git_commit": _git_output(("rev-parse", "HEAD"), cwd=root),
            "git_tree": _git_output(("rev-parse", "HEAD^{tree}"), cwd=root),
            "tracked_worktree_status": _git_output(
                ("status", "--porcelain=v1", "--untracked-files=all"),
                cwd=root,
            ),
            "untracked_files_omitted": False,
        }
    )
    return result


def _require_threads_one() -> dict[str, str]:
    values = {name: os.environ.get(name) for name in _THREAD_NAMES}
    wrong = {name: value for name, value in values.items() if value != "1"}
    if wrong:
        raise ReferenceMPIError(
            "Every BLAS/OpenMP thread variable must equal '1': "
            f"{wrong!r}."
        )
    return {name: "1" for name in _THREAD_NAMES}


_PHYSICAL_CORE_BINDING_POLICY = (
    "one_physical_core_per_numerical_worker_disjoint_per_host"
)


def _physical_core_binding(rank: int) -> dict[str, Any]:
    """Capture fail-closed Linux affinity and physical-core topology evidence."""

    try:
        affinity = sorted(int(value) for value in os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise ReferenceMPIError(
            "Certified MPI numerical work requires readable CPU affinity."
        ) from exc
    if not affinity:
        raise ReferenceMPIError(
            "Certified MPI numerical work has an empty CPU affinity."
        )
    topology: list[dict[str, int]] = []
    for logical_cpu in affinity:
        topology_root = Path(
            f"/sys/devices/system/cpu/cpu{logical_cpu}/topology"
        )
        try:
            package_id = int(
                (topology_root / "physical_package_id").read_text(
                    encoding="ascii"
                )
            )
            core_id = int(
                (topology_root / "core_id").read_text(encoding="ascii")
            )
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise ReferenceMPIError(
                "Certified MPI numerical work requires readable physical CPU "
                f"topology for logical CPU {logical_cpu}."
            ) from exc
        if package_id < 0 or core_id < 0:
            raise ReferenceMPIError(
                "Physical CPU package/core identifiers must be nonnegative."
            )
        topology.append(
            {
                "logical_cpu": logical_cpu,
                "package_id": package_id,
                "core_id": core_id,
            }
        )
    physical_cores = {
        (item["package_id"], item["core_id"]) for item in topology
    }
    if len(physical_cores) != 1:
        raise ReferenceMPIError(
            "Each numerical MPI worker must be bound to exactly one physical "
            "core (SMT siblings on that core are allowed)."
        )
    package_id, core_id = next(iter(physical_cores))
    return {
        "rank": int(rank),
        "hostname": socket.gethostname(),
        "cpu_affinity": affinity,
        "cpu_affinity_topology": topology,
        "physical_core": {
            "package_id": package_id,
            "core_id": core_id,
        },
    }


def _normalize_worker_binding(
    value: Any,
    *,
    expected_rank: int,
    name: str,
) -> dict[str, Any]:
    """Validate and normalize one archived numerical-worker binding."""

    if not isinstance(value, Mapping):
        raise ReferenceMPIError(f"{name} must be a JSON object.")
    if set(value) != {
        "rank",
        "hostname",
        "cpu_affinity",
        "cpu_affinity_topology",
        "physical_core",
    }:
        raise ReferenceMPIError(f"{name} has an ambiguous schema.")
    rank = value.get("rank")
    hostname = value.get("hostname")
    affinity = value.get("cpu_affinity")
    topology = value.get("cpu_affinity_topology")
    physical = value.get("physical_core")
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
        or not isinstance(physical, Mapping)
        or set(physical) != {"package_id", "core_id"}
    ):
        raise ReferenceMPIError(f"{name} is invalid.")
    normalized_topology: list[dict[str, int]] = []
    for cpu, item in zip(affinity, topology, strict=True):
        if not isinstance(item, Mapping) or set(item) != {
            "logical_cpu",
            "package_id",
            "core_id",
        }:
            raise ReferenceMPIError(f"{name} topology is invalid.")
        logical_cpu = item.get("logical_cpu")
        package_id = item.get("package_id")
        core_id = item.get("core_id")
        if (
            logical_cpu != cpu
            or not isinstance(package_id, int)
            or isinstance(package_id, bool)
            or package_id < 0
            or not isinstance(core_id, int)
            or isinstance(core_id, bool)
            or core_id < 0
        ):
            raise ReferenceMPIError(f"{name} topology is invalid.")
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
        raise ReferenceMPIError(
            f"{name} is not bound to exactly one physical core."
        )
    package_id, core_id = next(iter(physical_cores))
    if dict(physical) != {
        "package_id": package_id,
        "core_id": core_id,
    }:
        raise ReferenceMPIError(f"{name} physical-core summary is invalid.")
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


def _normalize_execution_worker_bindings(
    assignments: Sequence[Mapping[str, Any]],
    *,
    communicator_size: int,
) -> dict[str, Any]:
    """Require consistent, disjoint physical-core binding for MPI workers."""

    by_worker: dict[tuple[str, int], dict[str, Any]] = {}
    occupied: dict[tuple[str, int, int], int] = {}
    for ordinal, assignment in enumerate(assignments):
        status = assignment.get("status")
        rank = assignment.get("rank")
        binding = assignment.get("worker_binding")
        if status == "resumed":
            if rank is not None or binding is not None:
                raise ReferenceMPIError(
                    "Resumed assignments must not claim numerical-worker binding."
                )
            continue
        if status not in {"computed", "adopted"}:
            continue
        if communicator_size == 1:
            if binding is not None:
                raise ReferenceMPIError(
                    "Serial assignments must not claim certified MPI binding."
                )
            continue
        if not isinstance(rank, int) or isinstance(rank, bool) or rank == 0:
            raise ReferenceMPIError(
                "MPI numerical assignments require a nonzero worker rank."
            )
        normalized = _normalize_worker_binding(
            binding,
            expected_rank=rank,
            name=f"assignments[{ordinal}].worker_binding",
        )
        worker = (str(normalized["hostname"]), rank)
        previous = by_worker.setdefault(worker, normalized)
        if previous != normalized:
            raise ReferenceMPIError(
                "One MPI worker reported inconsistent affinity/topology."
            )
        physical = normalized["physical_core"]
        occupied_key = (
            str(normalized["hostname"]),
            int(physical["package_id"]),
            int(physical["core_id"]),
        )
        previous_rank = occupied.setdefault(occupied_key, rank)
        if previous_rank != rank:
            raise ReferenceMPIError(
                "Distinct MPI worker ranks share one physical core."
            )
    rank_bindings: list[dict[str, Any]] = []
    for key in sorted(by_worker, key=lambda item: (item[0], item[1])):
        binding = by_worker[key]
        task_ids = [
            f"target-grg-{int(assignment['grid_point_grg']):08d}"
            for assignment in assignments
            if assignment.get("status") in {"computed", "adopted"}
            and assignment.get("rank") == binding["rank"]
            and _mapping_hostname(assignment.get("worker_binding"))
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
    available = bool(rank_bindings)
    return {
        "schema": "negfpy_task_physical_core_bindings",
        "schema_version": 1,
        "available": available,
        "one_physical_core_per_assignment": True,
        "consistent_affinity_per_rank": True,
        "disjoint_physical_cores_across_ranks": True,
        "unavailable_task_ids": [],
        "rank_bindings": rank_bindings,
    }


def _mapping_hostname(value: Any) -> str | None:
    if isinstance(value, Mapping) and isinstance(value.get("hostname"), str):
        return str(value["hostname"])
    return None


def _file_identity(path: Path) -> dict[str, Any]:
    target = _regular_file(path, name=str(path))
    stat = target.stat()
    return {
        "name": target.name,
        "size_bytes": stat.st_size,
        "sha256": _sha256_file(target),
    }


def _force_stage_identities(force_stage_dir: Path) -> dict[str, Any]:
    names = sorted(path.name for path in force_stage_dir.iterdir())
    if frozenset(names) != serial._FORCE_STAGE_FILENAMES:
        raise ReferenceMPIError(
            "force_stage_dir does not have the exact completed serial "
            "force-constant schema."
        )
    return {
        name: _file_identity(force_stage_dir / name)
        for name in names
    }


@dataclass(frozen=True)
class CampaignIdentity:
    """Immutable scientific and provenance identity of an MPI campaign."""

    payload: Mapping[str, Any]
    sha256: str

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> CampaignIdentity:
        normalized = dict(payload)
        _validate_final_output_name(
            normalized.get("final_output_name"),
            name="campaign.final_output_name",
        )
        return cls(
            payload=normalized,
            sha256=_sha256_bytes(_canonical_json_bytes(normalized)),
        )


@dataclass(frozen=True)
class PreparedCampaign:
    """Rank-zero preflight result broadcast to all workers."""

    identity: CampaignIdentity
    config: serial.GeneratorConfig
    source_files: Mapping[str, Path]
    source_verification: Mapping[str, Any]
    environment_identity: Mapping[str, Any]
    force_stage_dir: Path
    run_dir: Path
    output_dir: Path


def _authenticated_final_output_name(campaign: PreparedCampaign) -> str:
    authenticated = _validate_final_output_name(
        campaign.identity.payload.get("final_output_name"),
        name="campaign.final_output_name",
    )
    visible = _validate_final_output_name(
        campaign.output_dir.name,
        name="output_dir.name",
    )
    if visible != authenticated:
        raise ReferenceMPIError(
            "output_dir basename does not match authenticated "
            "campaign.final_output_name."
        )
    return authenticated


def _manifest_document(identity: CampaignIdentity) -> dict[str, Any]:
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_manifest",
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "campaign_sha256": identity.sha256,
        "campaign": dict(identity.payload),
    }


def _initialize_run_directory(
    run_dir: Path,
    *,
    identity: CampaignIdentity,
) -> None:
    run_dir = _require_no_symlink_path_components(
        run_dir,
        name="run_dir",
    )
    document = _manifest_document(identity)
    if run_dir.exists() and not run_dir.is_dir():
        raise ReferenceMPIError("run_dir must be a non-symlink directory.")
    if not run_dir.exists():
        run_dir.mkdir(parents=True)
        _require_no_symlink_path_components(run_dir, name="run_dir")
    with _CampaignExecutionLock.acquire(run_dir) as execution_lock:
        manifest_path = run_dir / MPI_MANIFEST_FILENAME
        if not manifest_path.exists():
            _cleanup_private_atomic_temporaries(
                execution_lock,
                scopes=(".",),
            )
            non_lock_entries = {
                item.name
                for item in run_dir.iterdir()
                if item.name != _CAMPAIGN_EXECUTION_LOCK_FILENAME
            }
            if non_lock_entries:
                raise ReferenceMPIError(
                    "Existing run_dir lacks the immutable campaign manifest."
                )
            _write_json_exclusive(manifest_path, document)
        elif _read_json_object(
            manifest_path,
            name=MPI_MANIFEST_FILENAME,
        ) != document:
            raise ReferenceMPIError(
                "Existing run_dir belongs to a different campaign."
            )
        allowed = {
            MPI_MANIFEST_FILENAME,
            _CAMPAIGN_EXECUTION_LOCK_FILENAME,
            "tasks",
            "assignments",
            "failures",
            "executions",
            "scratch",
            "scratch-cleanups",
        }
        entries = {item.name for item in run_dir.iterdir()}
        suspicious = {
            name
            for name in entries
            if name.startswith(".")
            and name.endswith(".tmp")
            and _private_temporary_target(".", name) is None
        }
        if suspicious:
            raise ReferenceMPIError(
                "run_dir contains suspicious atomic-publication residues: "
                f"{sorted(suspicious)!r}."
            )
        unknown = {
            name
            for name in entries - allowed
            if _private_temporary_target(".", name) is None
        }
        if unknown:
            raise ReferenceMPIError(
                f"run_dir contains unknown entries: {sorted(unknown)!r}."
            )
        for name in (
            "tasks",
            "assignments",
            "failures",
            "executions",
            "scratch",
            "scratch-cleanups",
        ):
            path = run_dir / name
            try:
                details = os.lstat(path)
            except FileNotFoundError:
                path.mkdir()
                details = os.lstat(path)
            if not stat.S_ISDIR(details.st_mode):
                raise ReferenceMPIError(f"run_dir/{name} has an invalid type.")
        execution_lock.verify()


def prepare_campaign(
    *,
    manifest_path: Path,
    source_dir: Path,
    force_stage_dir: Path,
    run_dir: Path,
    output_dir: Path,
    config: serial.GeneratorConfig,
    runtime: Any | None = None,
    commit_resolver: Callable[[Path], str] | None = None,
    require_clean_source: bool = True,
) -> PreparedCampaign:
    """Authenticate inputs and create or resume an immutable run graph."""

    run_dir = _require_no_symlink_path_components(
        run_dir,
        name="run_dir",
    )
    output_dir = _require_no_symlink_path_components(
        output_dir,
        name="output_dir",
    )
    final_output_name = _validate_final_output_name(
        output_dir.name,
        name="output_dir.name",
    )
    if not isinstance(config, serial.GeneratorConfig):
        raise TypeError("config must be the serial GeneratorConfig type.")
    if config.force_constants_only or config.mesh is None:
        raise ValueError("MPI reference generation requires a reference config.")
    _require_threads_one()
    if output_dir.exists():
        raise FileExistsError(
            "output_dir already exists; final artifacts are never overwritten."
        )
    manifest = serial.load_input_manifest(manifest_path)
    source_files, source_verification = serial._verify_source(
        manifest,
        source_dir,
        commit_resolver=commit_resolver,
    )
    active_runtime = runtime or serial._Phono3pyRuntime()
    versions = dict(
        active_runtime.verify_versions(dict(manifest.expected_versions))
    )
    if versions != dict(manifest.expected_versions):
        raise ReferenceMPIError("Runtime returned unexpected pinned versions.")
    environment = serial._environment_identity(
        runtime_versions=versions,
        backend=config.backend,
    )
    serial._validate_force_stage(
        force_stage_dir,
        normalized_manifest=manifest.normalized_dict(),
        source_verification=source_verification,
        force_settings=config.force_constants_settings(),
        environment_identity=environment,
    )
    serial_code = environment.get("code")
    mpi_code = _orchestration_code_identity()
    for label, code in (("serial generator", serial_code), ("MPI driver", mpi_code)):
        if not isinstance(code, Mapping):
            raise ReferenceMPIError(f"{label} has no code identity.")
        if require_clean_source and (
            code.get("tracked_worktree_status") != ""
            or code.get("untracked_files_omitted") is not False
        ):
            raise ReferenceMPIError(
                f"{label} requires an exactly clean worktree including "
                "untracked files."
            )
        if require_clean_source and not code.get("git_commit"):
            raise ReferenceMPIError(
                f"{label} requires an exact Git commit."
            )
    force_identities = _force_stage_identities(force_stage_dir)
    payload = {
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "algorithm": (
            "one_complete_phono3py_target_q_per_worker_canonical_rank0_merge"
        ),
        "floating_point_mpi_reductions": False,
        "serial_generator": dict(serial_code),
        "mpi_driver": mpi_code,
        "source_manifest": manifest.normalized_dict(),
        "source_verification": source_verification,
        "force_stage": force_identities,
        "reference_settings": config.reference_settings(),
        "runtime_environment": environment,
        "thread_environment": _require_threads_one(),
        "task_grid_points_grg": list(config.grid_points),
        "final_output_name": final_output_name,
    }
    identity = CampaignIdentity.from_payload(payload)
    _initialize_run_directory(run_dir, identity=identity)
    return PreparedCampaign(
        identity=identity,
        config=config,
        source_files=source_files,
        source_verification=source_verification,
        environment_identity=environment,
        force_stage_dir=force_stage_dir,
        run_dir=run_dir,
        output_dir=output_dir,
    )


def _task_stem(grid_point: int) -> str:
    return f"{_TASK_PREFIX}{grid_point:08d}"


def _task_paths(run_dir: Path, grid_point: int) -> tuple[Path, Path, Path]:
    stem = _task_stem(grid_point)
    return (
        run_dir / "tasks" / f"{stem}{_TASK_SUFFIX}",
        run_dir / "tasks" / f"{stem}{_TASK_RECORD_SUFFIX}",
        run_dir / "failures" / f"{stem}{_TASK_RECORD_SUFFIX}",
    )


def _origin_assignment_path(
    run_dir: Path,
    grid_point: int,
    execution_id: str,
) -> Path:
    _execution_path(run_dir, execution_id)
    return (
        run_dir
        / "assignments"
        / (
            f"{_task_stem(grid_point)}.{execution_id}"
            f"{_ORIGIN_ASSIGNMENT_SUFFIX}"
        )
    )


def _origin_assignment_identity(
    run_dir: Path,
    path: Path,
) -> dict[str, Any]:
    relative = path.resolve().relative_to(run_dir.resolve()).as_posix()
    return {
        "relative_path": relative,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _publish_origin_assignment(
    campaign: PreparedCampaign,
    grid_point: int,
    *,
    execution_id: str,
    rank: int,
    communicator_size: int,
    worker_binding: Mapping[str, Any] | None,
) -> tuple[Path, dict[str, Any]]:
    """Publish immutable task provenance before numerical work begins."""

    mode = "serial" if communicator_size == 1 else "mpi"
    if (
        isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(communicator_size, bool)
        or not isinstance(communicator_size, int)
        or communicator_size <= 0
        or rank < 0
        or rank >= communicator_size
        or (mode == "serial" and rank != 0)
        or (mode == "mpi" and rank == 0)
    ):
        raise ReferenceMPIError(
            "Origin assignment rank and communicator are invalid."
        )
    normalized_binding: dict[str, Any] | None
    if mode == "serial":
        if worker_binding is not None:
            raise ReferenceMPIError(
                "Serial origin assignment cannot claim MPI core binding."
            )
        normalized_binding = None
    else:
        normalized_binding = _normalize_worker_binding(
            worker_binding,
            expected_rank=rank,
            name="pre-work origin assignment worker_binding",
        )
    document = {
        "schema_name": _ORIGIN_ASSIGNMENT_SCHEMA,
        "schema_version": _TASK_SCHEMA_VERSION,
        "campaign_sha256": campaign.identity.sha256,
        "grid_point_grg": grid_point,
        "execution_id": execution_id,
        "status": "assigned_before_numerical_work",
        "mode": mode,
        "rank": rank,
        "communicator_size": communicator_size,
        "hostname": socket.gethostname(),
        "assigned_at_utc": _utc_now(),
        "worker_binding": normalized_binding,
        "thread_environment": _require_threads_one(),
    }
    path = _origin_assignment_path(
        campaign.run_dir,
        grid_point,
        execution_id,
    )
    _write_json_exclusive(path, document)
    validated = _validate_origin_assignment(
        path,
        run_dir=campaign.run_dir,
        campaign_sha256=campaign.identity.sha256,
        grid_point=grid_point,
    )
    return path, validated


def _validate_origin_assignment(
    path: Path,
    *,
    run_dir: Path,
    campaign_sha256: str,
    grid_point: int,
) -> dict[str, Any]:
    document = _read_json_object(path, name="task origin assignment")
    keys = {
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
    execution_id = document.get("execution_id")
    rank = document.get("rank")
    size = document.get("communicator_size")
    mode = document.get("mode")
    if (
        set(document) != keys
        or document.get("schema_name") != _ORIGIN_ASSIGNMENT_SCHEMA
        or document.get("schema_version") != _TASK_SCHEMA_VERSION
        or document.get("campaign_sha256") != campaign_sha256
        or document.get("grid_point_grg") != grid_point
        or document.get("status") != "assigned_before_numerical_work"
        or not isinstance(execution_id, str)
        or not execution_id
        or _origin_assignment_path(
            run_dir,
            grid_point,
            execution_id,
        )
        != path
        or mode not in {"serial", "mpi"}
        or isinstance(rank, bool)
        or not isinstance(rank, int)
        or isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or rank < 0
        or rank >= size
        or (mode == "serial" and (size != 1 or rank != 0))
        or (mode == "mpi" and (size < 2 or rank == 0))
        or not isinstance(document.get("hostname"), str)
        or not document.get("hostname")
        or not isinstance(document.get("assigned_at_utc"), str)
        or not document.get("assigned_at_utc")
    ):
        raise ReferenceMPIError(
            f"Origin assignment for GRG {grid_point} is inconsistent."
        )
    _validate_thread_environment(
        document.get("thread_environment"),
        name=f"Origin assignment for GRG {grid_point}.thread_environment",
    )
    if mode == "serial":
        if document.get("worker_binding") is not None:
            raise ReferenceMPIError(
                "Serial origin assignment claims worker binding."
            )
    else:
        normalized = _normalize_worker_binding(
            document.get("worker_binding"),
            expected_rank=rank,
            name=f"origin assignment for GRG {grid_point}",
        )
        if document.get("worker_binding") != normalized:
            raise ReferenceMPIError(
                "Origin assignment worker binding is noncanonical."
            )
        if normalized["hostname"] != document["hostname"]:
            raise ReferenceMPIError(
                "Origin assignment hostname differs from worker binding."
            )
    return document


def _cleanup_interrupted_scratch(
    campaign: PreparedCampaign,
    *,
    execution_id: str,
    execution_lock: _CampaignExecutionLock | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Remove only validated disposable worker scratch and record the action."""

    _execution_path(campaign.run_dir, execution_id)
    owned_lock = execution_lock
    acquired_here = owned_lock is None
    if owned_lock is None:
        owned_lock = _CampaignExecutionLock.acquire(campaign.run_dir)
    try:
        owned_lock.verify()
        private_removed = _cleanup_private_atomic_temporaries(
            owned_lock,
            scopes=tuple(_PRIVATE_TEMPORARY_TARGETS),
        )
        staging_removed = _cleanup_abandoned_final_staging(
            campaign,
            owned_lock,
        )
        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        scratch_descriptor = os.open(
            "scratch",
            flags,
            dir_fd=owned_lock.run_descriptor,
        )
        removed: list[dict[str, Any]] = []
        directory_identities: list[tuple[str, tuple[int, int]]] = []
        try:
            for entry in sorted(
                os.scandir(scratch_descriptor),
                key=lambda item: item.name,
            ):
                details = entry.stat(follow_symlinks=False)
                if (
                    not stat.S_ISDIR(details.st_mode)
                    or _SCRATCH_DIRECTORY_PATTERN.fullmatch(entry.name) is None
                ):
                    raise ReferenceMPIError(
                        f"Suspicious worker scratch residue: {entry.name!r}."
                    )
                child_descriptor = os.open(
                    entry.name,
                    flags,
                    dir_fd=scratch_descriptor,
                )
                try:
                    opened = os.fstat(child_descriptor)
                    identity = (opened.st_dev, opened.st_ino)
                    if identity != (details.st_dev, details.st_ino):
                        raise ReferenceMPIError(
                            "Worker scratch changed during cleanup preflight."
                        )
                    files = _inspect_scratch_tree(child_descriptor)
                finally:
                    os.close(child_descriptor)
                removed.append({"directory": entry.name, "files": files})
                directory_identities.append((entry.name, identity))
            for name, identity in directory_identities:
                _remove_directory_tree_at(
                    scratch_descriptor,
                    name,
                    expected_identity=identity,
                )
            if any(os.scandir(scratch_descriptor)):
                raise ReferenceMPIError(
                    "Worker scratch cleanup was incomplete."
                )
            os.fsync(scratch_descriptor)
        finally:
            os.close(scratch_descriptor)
        removed.extend(private_removed)
        removed.extend(staging_removed)
        document = {
            "schema_name": _SCRATCH_CLEANUP_SCHEMA,
            "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
            "campaign_sha256": campaign.identity.sha256,
            "execution_id": execution_id,
            "status": "complete",
            "removed": removed,
            "scratch_empty_after_cleanup": True,
            "completed_at_utc": _utc_now(),
        }
        path = (
            campaign.run_dir
            / "scratch-cleanups"
            / f"{execution_id}.json"
        )
        _write_json_exclusive(path, document)
        owned_lock.verify()
        return path, document
    finally:
        if acquired_here:
            owned_lock.close()


def _validate_scratch_cleanup_record(
    campaign: PreparedCampaign,
    identity: Any,
    *,
    execution_id: str,
) -> tuple[Path, dict[str, Any]]:
    _authenticated_final_output_name(campaign)
    if not isinstance(identity, Mapping):
        raise ReferenceMPIError("Execution lacks scratch-cleanup identity.")
    path = campaign.run_dir / "scratch-cleanups" / f"{execution_id}.json"
    _regular_single_link_file(path, name="scratch-cleanup record")
    if (
        _file_identity(path) != dict(identity)
    ):
        raise ReferenceMPIError("Scratch-cleanup record changed.")
    document = _read_json_object(path, name="scratch cleanup")
    if (
        set(document)
        != {
            "schema_name",
            "schema_version",
            "campaign_sha256",
            "execution_id",
            "status",
            "removed",
            "scratch_empty_after_cleanup",
            "completed_at_utc",
        }
        or document.get("schema_name") != _SCRATCH_CLEANUP_SCHEMA
        or document.get("schema_version")
        != MPI_REFERENCE_SCHEMA_VERSION
        or document.get("campaign_sha256") != campaign.identity.sha256
        or document.get("execution_id") != execution_id
        or document.get("status") != "complete"
        or not isinstance(document.get("removed"), list)
        or document.get("scratch_empty_after_cleanup") is not True
        or not isinstance(document.get("completed_at_utc"), str)
        or not document.get("completed_at_utc")
    ):
        raise ReferenceMPIError("Scratch-cleanup record is inconsistent.")
    seen_directories: set[str] = set()
    for ordinal, item in enumerate(document["removed"]):
        if (
            not isinstance(item, Mapping)
            or set(item) != {"directory", "files"}
            or not isinstance(item.get("directory"), str)
            or not isinstance(item.get("files"), list)
        ):
            raise ReferenceMPIError(
                f"Scratch-cleanup removed[{ordinal}] has an ambiguous schema."
            )
        directory = str(item["directory"])
        staging_name = (
            directory.removeprefix("output-parent/")
            if directory.startswith("output-parent/")
            else ""
        )
        valid_staging_directory = (
            bool(staging_name)
            and _final_staging_evidence_directory(
                campaign,
                staging_name,
            )
            == directory
        )
        if (
            directory in seen_directories
            or (
                _SCRATCH_DIRECTORY_PATTERN.fullmatch(directory) is None
                and directory not in _PRIVATE_TEMPORARY_TARGETS
                and not valid_staging_directory
            )
        ):
            raise ReferenceMPIError(
                f"Scratch-cleanup removed[{ordinal}] has an invalid directory."
            )
        seen_directories.add(directory)
        seen_files: set[str] = set()
        for file_ordinal, value in enumerate(item["files"]):
            if (
                not isinstance(value, Mapping)
                or set(value) != {"relative_path", "size_bytes", "sha256"}
                or not isinstance(value.get("relative_path"), str)
                or not value.get("relative_path")
                or Path(str(value["relative_path"])).is_absolute()
                or ".." in Path(str(value["relative_path"])).parts
                or str(value["relative_path"]) in seen_files
                or isinstance(value.get("size_bytes"), bool)
                or not isinstance(value.get("size_bytes"), int)
                or int(value["size_bytes"]) < 0
            ):
                raise ReferenceMPIError(
                    "Scratch-cleanup file identity has an ambiguous schema at "
                    f"removed[{ordinal}].files[{file_ordinal}]."
                )
            relative_path = str(value["relative_path"])
            _require_sha256(
                value.get("sha256"),
                name=(
                    "scratch-cleanup "
                    f"removed[{ordinal}].files[{file_ordinal}].sha256"
                ),
            )
            if (
                directory in _PRIVATE_TEMPORARY_TARGETS
                and _private_temporary_target(directory, relative_path) is None
            ):
                raise ReferenceMPIError(
                    "Scratch-cleanup private temporary evidence is invalid."
                )
            seen_files.add(relative_path)
    return path, document


def _h5_attr_value(value: Any) -> Any:
    """Normalize HDF5 attribute scalars into canonical JSON values."""

    if isinstance(value, bytes):
        return {"bytes_hex": value.hex()}
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        return [_h5_attr_value(item) for item in value]
    if isinstance(value, list):
        return [_h5_attr_value(item) for item in value]
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _update_attrs_digest(digest: Any, attrs: Any) -> None:
    payload = {
        str(name): _h5_attr_value(attrs[name])
        for name in sorted(attrs.keys())
        if not str(name).startswith("mpi_reference_")
    }
    digest.update(_canonical_json_bytes(payload))


def _update_dataset_digest(digest: Any, dataset: Any) -> None:
    import numpy as np

    digest.update(str(dataset.name).encode("utf-8"))
    digest.update(b"\0")
    digest.update(dataset.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(value)) for value in dataset.shape).encode("ascii")
    )
    digest.update(b"\0")
    _update_attrs_digest(digest, dataset.attrs)
    if dataset.ndim == 0:
        digest.update(np.asarray(dataset[()]).tobytes(order="C"))
        return
    if dataset.shape[0] == 0:
        return
    for index in range(int(dataset.shape[0])):
        digest.update(
            np.ascontiguousarray(dataset[index]).tobytes(order="C")
        )


def _update_node_digest(digest: Any, node: Any) -> None:
    import h5py

    digest.update(str(node.name).encode("utf-8"))
    digest.update(b"\0")
    _update_attrs_digest(digest, node.attrs)
    if isinstance(node, h5py.Dataset):
        _update_dataset_digest(digest, node)
        return
    for name in sorted(node.keys()):
        _update_node_digest(digest, node[name])


def _common_reference_digest(h5: Any) -> str:
    """Hash every field that must be identical in all one-q artifacts."""

    digest = hashlib.sha256()
    _update_attrs_digest(digest, h5.attrs)
    for name in ("crystal", "grid", "fc2_random_q"):
        _update_node_digest(digest, h5[name])
    settings = h5["settings"]
    _update_attrs_digest(digest, settings.attrs)
    for name in ("mesh", "band_indices", "temperatures", "sigmas"):
        _update_node_digest(digest, settings[name])
    phonons = h5["phonons"]
    _update_attrs_digest(digest, phonons.attrs)
    for name in ("frequencies", "eigenvectors", "bz_grid_addresses"):
        _update_node_digest(digest, phonons[name])
    gamma = h5["gamma"]
    _update_attrs_digest(digest, gamma.attrs)
    _update_attrs_digest(digest, gamma["values"].attrs)
    return digest.hexdigest()


def _validate_task_hdf5(
    path: Path,
    *,
    campaign_sha256: str,
    config: serial.GeneratorConfig,
    grid_point: int,
) -> dict[str, Any]:
    import h5py
    import numpy as np

    _regular_file(path, name="task HDF5")
    with h5py.File(path, "r") as h5:
        if (
            h5.attrs.get("schema_name")
            != "negfpy_phono3py_si_reference"
            or int(h5.attrs.get("schema_version", -1))
            != serial.REFERENCE_SCHEMA_VERSION
            or h5.attrs.get("mpi_reference_task_schema") != _TASK_SCHEMA_NAME
            or int(h5.attrs.get("mpi_reference_task_schema_version", -1))
            != _TASK_SCHEMA_VERSION
            or h5.attrs.get("mpi_reference_campaign_sha256")
            != campaign_sha256
            or int(h5.attrs.get("mpi_reference_grid_point_grg", -1))
            != grid_point
        ):
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} has invalid provenance."
            )
        settings = h5["settings"]
        mesh = tuple(int(value) for value in settings["mesh"][...])
        grg = np.asarray(settings["grid_points_grg"][...], dtype=np.int64)
        bzg = np.asarray(settings["grid_points_bzg"][...], dtype=np.int64)
        bands = np.asarray(settings["band_indices"][...], dtype=np.int64)
        temperatures = np.asarray(settings["temperatures"][...], dtype=float)
        sigmas = np.asarray(settings["sigmas"][...], dtype=float)
        expected_bands = (
            None
            if config.band_indices is None
            else np.asarray(config.band_indices, dtype=np.int64)
        )
        if (
            mesh != tuple(config.mesh or ())
            or grg.tolist() != [grid_point]
            or bzg.shape != (1,)
            or (
                expected_bands is not None
                and not np.array_equal(bands, expected_bands)
            )
            or not np.array_equal(
                temperatures,
                np.asarray(config.temperatures, dtype=float),
            )
            or not np.array_equal(sigmas, np.asarray([config.sigma], dtype=float))
        ):
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} conflicts with campaign settings."
            )
        group_name = f"grg_{grid_point:08d}"
        if set(h5["target_q"].keys()) != {group_name}:
            raise ReferenceMPIError(
                f"Task HDF5 for GRG {grid_point} has an ambiguous target group."
            )
        target = h5["target_q"][group_name]
        if (
            int(target.attrs.get("grid_point_grg", -1)) != grid_point
            or int(target.attrs.get("grid_point_bzg", -1)) != int(bzg[0])
            or int(target.attrs.get("target_ordinal", -1)) != 0
        ):
            raise ReferenceMPIError(
                f"Task target metadata is invalid for GRG {grid_point}."
            )
        gamma_shape = tuple(int(value) for value in h5["gamma/values"].shape)
        phonon_shape = tuple(
            int(value) for value in h5["phonons/frequencies"].shape
        )
        phonon_frequencies = np.asarray(
            h5["phonons/frequencies"][...]
        )
        eigenvector_shape = tuple(
            int(value) for value in h5["phonons/eigenvectors"].shape
        )
        grid = h5["grid"]
        d_diag_raw = np.asarray(grid["D_diag"][...])
        grg2bzg_raw = np.asarray(grid["grg2bzg"][...])
        bzg2grg_raw = np.asarray(grid["bzg2grg"][...])
        grid_addresses_raw = np.asarray(grid["addresses"][...])
        phonon_addresses_raw = np.asarray(
            h5["phonons/bz_grid_addresses"][...]
        )
        n_qpoints = math.prod(mesh)
        mapping_dtypes_are_integer = all(
            array.dtype.kind in "iu"
            for array in (
                d_diag_raw,
                grg2bzg_raw,
                bzg2grg_raw,
                grid_addresses_raw,
                phonon_addresses_raw,
            )
        )
        d_diag = np.asarray(d_diag_raw, dtype=np.int64)
        grg2bzg = np.asarray(grg2bzg_raw, dtype=np.int64)
        bzg2grg = np.asarray(bzg2grg_raw, dtype=np.int64)
        grid_addresses = np.asarray(grid_addresses_raw, dtype=np.int64)
        phonon_addresses = np.asarray(
            phonon_addresses_raw,
            dtype=np.int64,
        )
        n_bzg = int(bzg2grg.size)
        if (
            not mapping_dtypes_are_integer
            or d_diag.shape != (3,)
            or tuple(int(value) for value in d_diag) != mesh
            or grg2bzg.shape != (n_qpoints,)
            or bzg2grg.ndim != 1
            or n_bzg <= 0
            or grid_addresses.shape != (n_bzg, 3)
            or phonon_addresses.shape != (n_bzg, 3)
            or not np.array_equal(grid_addresses, phonon_addresses)
            or np.any((grg2bzg < 0) | (grg2bzg >= n_bzg))
            or np.any((bzg2grg < 0) | (bzg2grg >= n_qpoints))
            or not np.array_equal(
                bzg2grg[grg2bzg],
                np.arange(n_qpoints, dtype=np.int64),
            )
            or int(bzg[0]) != int(grg2bzg[grid_point])
        ):
            raise ReferenceMPIError(
                f"Task BZG/GRG mapping is invalid for GRG {grid_point}."
            )
        target_frequency_shape = tuple(
            int(value)
            for value in h5["phonons/target_band_frequencies"].shape
        )
        target_frequencies = np.asarray(
            h5["phonons/target_band_frequencies"][...]
        )
        if (
            len(gamma_shape) != 4
            or gamma_shape[2] != 1
            or len(phonon_shape) != 2
            or phonon_shape[0] != n_bzg
            or phonon_shape[1] <= 0
            or eigenvector_shape
            != (n_bzg, phonon_shape[1], phonon_shape[1])
            or np.any(bands < 0)
            or np.any(bands >= phonon_shape[1])
            or target_frequency_shape != (1, len(bands))
            or not np.array_equal(
                target_frequencies,
                phonon_frequencies[
                    int(bzg[0]),
                    bands,
                ][None, :],
                equal_nan=True,
            )
        ):
            raise ReferenceMPIError(
                f"Task target-axis shape is invalid for GRG {grid_point}."
            )
        runtime_text = h5.attrs.get("mpi_reference_runtime_json")
        if not isinstance(runtime_text, str):
            raise ReferenceMPIError("Task HDF5 lacks runtime metadata.")
        try:
            runtime = json.loads(
                runtime_text,
                object_pairs_hook=serial._reject_duplicate_keys,
            )
        except json.JSONDecodeError as exc:
            raise ReferenceMPIError("Task runtime metadata is invalid.") from exc
        if not isinstance(runtime, dict):
            raise ReferenceMPIError("Task runtime metadata must be an object.")
        runtime_targets = runtime.get("targets")
        runtime_target = (
            runtime_targets[0]
            if isinstance(runtime_targets, list)
            and len(runtime_targets) == 1
            and isinstance(runtime_targets[0], Mapping)
            else None
        )
        if (
            runtime.get("mesh") != list(mesh)
            or runtime.get("grid_points_grg") != [grid_point]
            or runtime.get("grid_points_bzg") != [int(bzg[0])]
            or runtime.get("band_indices") != [
                int(value) for value in bands
            ]
            or runtime.get("gamma_shape") != list(gamma_shape)
            or runtime.get("phonon_frequency_shape") != list(phonon_shape)
            or runtime.get("phonon_eigenvector_shape")
            != list(eigenvector_shape)
            or runtime.get("number_of_mesh_points") != n_qpoints
            or runtime.get("number_of_bz_grid_points") != n_bzg
            or runtime.get("effective_band_indices")
            != [int(value) for value in bands]
            or runtime_target is None
            or runtime_target.get("grid_point_grg") != grid_point
            or runtime_target.get("grid_point_bzg") != int(bzg[0])
            or runtime_target.get("triplets")
            != int(target["triplets_bzg"].shape[0])
            or runtime_target.get("triplet_weight_sum")
            != int(np.sum(target["triplet_weights"][...]))
            or runtime_target.get("interaction_strength_shape")
            != list(target["interaction_strength"].shape)
            or runtime_target.get("integration_weights_shape")
            != list(target["integration_weights"].shape)
        ):
            raise ReferenceMPIError(
                f"Task runtime metadata conflicts with HDF5 for GRG "
                f"{grid_point}."
            )
        origin_text = h5.attrs.get(
            "mpi_reference_origin_assignment_json",
            "",
        )
        if not isinstance(origin_text, str):
            raise ReferenceMPIError(
                "Task HDF5 origin-assignment metadata is invalid."
            )
        if origin_text:
            try:
                origin_assignment = json.loads(
                    origin_text,
                    object_pairs_hook=serial._reject_duplicate_keys,
                )
            except json.JSONDecodeError as exc:
                raise ReferenceMPIError(
                    "Task HDF5 origin-assignment metadata is invalid."
                ) from exc
            if (
                not isinstance(origin_assignment, dict)
                or set(origin_assignment)
                != {"relative_path", "size_bytes", "sha256"}
                or not isinstance(
                    origin_assignment.get("relative_path"), str
                )
                or Path(origin_assignment["relative_path"]).is_absolute()
                or ".." in Path(origin_assignment["relative_path"]).parts
                or isinstance(
                    origin_assignment.get("size_bytes"), bool
                )
                or not isinstance(
                    origin_assignment.get("size_bytes"), int
                )
                or origin_assignment["size_bytes"] <= 0
                or not isinstance(origin_assignment.get("sha256"), str)
                or len(origin_assignment["sha256"]) != 64
                or any(
                    character not in _SHA256_HEX
                    for character in origin_assignment["sha256"]
                )
            ):
                raise ReferenceMPIError(
                    "Task HDF5 origin-assignment identity is invalid."
                )
        else:
            origin_assignment = None
        return {
            "grid_point_grg": grid_point,
            "grid_point_bzg": int(bzg[0]),
            "bands": [int(value) for value in bands],
            "n_qpoints": int(n_qpoints),
            "n_bands": int(phonon_shape[1]),
            "gamma_shape": list(gamma_shape),
            "target_frequency_shape": list(target_frequency_shape),
            "common_reference_sha256": _common_reference_digest(h5),
            "runtime": runtime,
            "origin_assignment": origin_assignment,
        }


def _task_record(
    *,
    campaign_sha256: str,
    grid_point: int,
    hdf5_path: Path,
    validation: Mapping[str, Any],
    rank: int | None,
    hostname: str,
    started_utc: str | None,
    completed_utc: str,
    wall_seconds: float | None,
    disposition: str,
    origin_assignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_task_record",
        "schema_version": _TASK_SCHEMA_VERSION,
        "campaign_sha256": campaign_sha256,
        "grid_point_grg": grid_point,
        "status": "complete",
        "disposition": disposition,
        "origin_assignment": (
            None
            if origin_assignment is None
            else dict(origin_assignment)
        ),
        "artifact": _file_identity(hdf5_path),
        "scientific": {
            key: validation[key]
            for key in (
                "grid_point_grg",
                "grid_point_bzg",
                "bands",
                "n_qpoints",
                "n_bands",
                "gamma_shape",
                "target_frequency_shape",
                "common_reference_sha256",
            )
        },
        "execution": {
            "rank": rank,
            "hostname": hostname,
            "started_at_utc": started_utc,
            "completed_at_utc": completed_utc,
            "wall_seconds": wall_seconds,
            "thread_environment": _require_threads_one(),
        },
        "runtime": validation["runtime"],
    }


def _validate_task_record(
    record_path: Path,
    hdf5_path: Path,
    *,
    campaign_sha256: str,
    config: serial.GeneratorConfig,
    grid_point: int,
) -> dict[str, Any]:
    record = _read_json_object(record_path, name="task record")
    expected_keys = {
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
    if (
        set(record) != expected_keys
        or
        record.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_task_record"
        or record.get("schema_version") != _TASK_SCHEMA_VERSION
        or record.get("campaign_sha256") != campaign_sha256
        or record.get("grid_point_grg") != grid_point
        or record.get("status") != "complete"
    ):
        raise ReferenceMPIError(
            f"Task record for GRG {grid_point} has invalid identity."
        )
    artifact = record.get("artifact")
    if not isinstance(artifact, Mapping):
        raise ReferenceMPIError("Task record has no artifact identity.")
    actual = _file_identity(hdf5_path)
    if dict(artifact) != actual:
        raise ReferenceMPIError(
            f"Task artifact for GRG {grid_point} changed after publication."
        )
    validation = _validate_task_hdf5(
        hdf5_path,
        campaign_sha256=campaign_sha256,
        config=config,
        grid_point=grid_point,
    )
    declared = record.get("scientific")
    expected = {
        key: validation[key]
        for key in (
            "grid_point_grg",
            "grid_point_bzg",
            "bands",
            "n_qpoints",
            "n_bands",
            "gamma_shape",
            "target_frequency_shape",
            "common_reference_sha256",
        )
    }
    if (
        declared != expected
        or record.get("runtime") != validation["runtime"]
        or record.get("origin_assignment")
        != validation["origin_assignment"]
    ):
        raise ReferenceMPIError(
            f"Task record for GRG {grid_point} conflicts with HDF5 fields."
        )
    disposition = record.get("disposition")
    task_execution = record.get("execution")
    origin_identity = record.get("origin_assignment")
    if (
        not isinstance(task_execution, Mapping)
        or set(task_execution)
        != {
            "rank",
            "hostname",
            "started_at_utc",
            "completed_at_utc",
            "wall_seconds",
            "thread_environment",
        }
        or disposition
        not in {
            "computed",
            "computed_after_interrupted_record_publication",
            "adopted_after_interrupted_record_publication",
        }
    ):
        raise ReferenceMPIError("Task record execution is invalid.")
    _validate_thread_environment(
        task_execution.get("thread_environment"),
        name=f"Task record for GRG {grid_point}.execution.thread_environment",
    )
    if disposition in {
        "computed",
        "computed_after_interrupted_record_publication",
    }:
        if origin_identity is None:
            return record
        if (
            not isinstance(origin_identity, Mapping)
            or set(origin_identity)
            != {"relative_path", "size_bytes", "sha256"}
            or not isinstance(origin_identity.get("relative_path"), str)
            or Path(origin_identity["relative_path"]).is_absolute()
            or ".." in Path(origin_identity["relative_path"]).parts
        ):
            raise ReferenceMPIError(
                f"Computed task GRG {grid_point} lacks origin assignment."
            )
        origin_path = campaign_path = record_path.parents[1]
        origin_path = campaign_path / origin_identity["relative_path"]
        if (
            not origin_path.is_file()
            or _origin_assignment_identity(campaign_path, origin_path)
            != dict(origin_identity)
        ):
            raise ReferenceMPIError(
                f"Task origin assignment for GRG {grid_point} changed."
            )
        origin = _validate_origin_assignment(
            origin_path,
            run_dir=campaign_path,
            campaign_sha256=campaign_sha256,
            grid_point=grid_point,
        )
        if (
            task_execution.get("rank") != origin["rank"]
            or task_execution.get("hostname") != origin["hostname"]
            or task_execution.get("started_at_utc")
            != origin["assigned_at_utc"]
        ):
            raise ReferenceMPIError(
                f"Task GRG {grid_point} differs from its origin assignment."
            )
    elif (
        disposition == "adopted_after_interrupted_record_publication"
        and origin_identity is not None
    ):
        raise ReferenceMPIError(
            f"Adopted task GRG {grid_point} claims computed origin."
        )
    return record


def _adopt_or_validate_task(
    campaign: PreparedCampaign,
    grid_point: int,
) -> dict[str, Any] | None:
    hdf5_path, record_path, failure_path = _task_paths(
        campaign.run_dir,
        grid_point,
    )
    if failure_path.exists():
        failure = _read_json_object(failure_path, name="task failure")
        if (
            failure.get("campaign_sha256") != campaign.identity.sha256
            or failure.get("grid_point_grg") != grid_point
        ):
            raise ReferenceMPIError(
                f"Failure record for GRG {grid_point} has invalid identity."
            )
        raise ReferenceMPIError(
            f"GRG {grid_point} has an archived terminal failure."
        )
    if record_path.exists():
        if not hdf5_path.exists():
            raise ReferenceMPIError(
                f"Task record for GRG {grid_point} has no artifact."
            )
        return _validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=grid_point,
        )
    if not hdf5_path.exists():
        return None
    validation = _validate_task_hdf5(
        hdf5_path,
        campaign_sha256=campaign.identity.sha256,
        config=campaign.config,
        grid_point=grid_point,
    )
    origin_identity = validation["origin_assignment"]
    if origin_identity is not None:
        origin_path = campaign.run_dir / origin_identity["relative_path"]
        if (
            not origin_path.is_file()
            or _origin_assignment_identity(
                campaign.run_dir,
                origin_path,
            )
            != origin_identity
        ):
            raise ReferenceMPIError(
                f"Interrupted task GRG {grid_point} has invalid origin."
            )
        origin = _validate_origin_assignment(
            origin_path,
            run_dir=campaign.run_dir,
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
        )
        record = _task_record(
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
            hdf5_path=hdf5_path,
            validation=validation,
            rank=int(origin["rank"]),
            hostname=str(origin["hostname"]),
            started_utc=str(origin["assigned_at_utc"]),
            completed_utc=_utc_now(),
            wall_seconds=None,
            disposition=(
                "computed_after_interrupted_record_publication"
            ),
            origin_assignment=origin_identity,
        )
        _write_json_exclusive(record_path, record)
        return record
    record = _task_record(
        campaign_sha256=campaign.identity.sha256,
        grid_point=grid_point,
        hdf5_path=hdf5_path,
        validation=validation,
        rank=None,
        hostname=socket.gethostname(),
        started_utc=None,
        completed_utc=_utc_now(),
        wall_seconds=None,
        disposition="adopted_after_interrupted_record_publication",
    )
    _write_json_exclusive(record_path, record)
    return record


def _publish_failure(
    campaign: PreparedCampaign,
    grid_point: int,
    *,
    rank: int,
    started_utc: str,
    started: float,
    error: BaseException,
) -> None:
    _, _, failure_path = _task_paths(campaign.run_dir, grid_point)
    document = {
        "schema_name": "negfpy_phono3py_si_reference_mpi_task_failure",
        "schema_version": _TASK_SCHEMA_VERSION,
        "campaign_sha256": campaign.identity.sha256,
        "grid_point_grg": grid_point,
        "status": "failed",
        "rank": rank,
        "hostname": socket.gethostname(),
        "started_at_utc": started_utc,
        "failed_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback": traceback.format_exc(),
        "thread_environment": _require_threads_one(),
    }
    if failure_path.exists():
        raise ReferenceMPIError(
            f"Failure record already exists for GRG {grid_point}."
        )
    _write_json_exclusive(failure_path, document)


def _generate_one_task(
    campaign: PreparedCampaign,
    grid_point: int,
    *,
    rank: int,
    runtime_factory: Callable[[], Any],
    origin_assignment_path: Path | None = None,
) -> dict[str, Any]:
    existing = _adopt_or_validate_task(campaign, grid_point)
    if existing is not None:
        return {"grid_point_grg": grid_point, "status": "resumed"}
    hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
    origin: dict[str, Any] | None = None
    if origin_assignment_path is not None:
        origin = _validate_origin_assignment(
            origin_assignment_path,
            run_dir=campaign.run_dir,
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
        )
        if origin["rank"] != rank:
            raise ReferenceMPIError(
                "Task rank differs from its pre-work origin assignment."
            )
    started_utc = (
        _utc_now() if origin is None else str(origin["assigned_at_utc"])
    )
    started = time.perf_counter()
    scratch = Path(
        tempfile.mkdtemp(
            prefix=f".{_task_stem(grid_point)}-rank{rank}-",
            dir=campaign.run_dir / "scratch",
        )
    )
    scratch_details = os.lstat(scratch)
    scratch_identity = (scratch_details.st_dev, scratch_details.st_ino)
    try:
        for name in ("fc2.compact.hdf5", "fc3.compact.hdf5"):
            source = _regular_file(
                campaign.force_stage_dir / name,
                name=f"force-stage {name}",
            )
            os.link(source, scratch / name)
        config = serial.GeneratorConfig(
            mesh=campaign.config.mesh,
            grid_points=(grid_point,),
            band_indices=campaign.config.band_indices,
            temperatures=campaign.config.temperatures,
            sigma=campaign.config.sigma,
            sigma_cutoff=campaign.config.sigma_cutoff,
            backend=campaign.config.backend,
            make_r0_average=campaign.config.make_r0_average,
            symmetrize_fc=campaign.config.symmetrize_fc,
            symmetrize_fc3q=campaign.config.symmetrize_fc3q,
        )
        runtime = runtime_factory()
        versions = dict(
            runtime.verify_versions(
                dict(
                    campaign.identity.payload["source_manifest"]["versions"]
                )
            )
        )
        if versions != dict(
            campaign.identity.payload["source_manifest"]["versions"]
        ):
            raise ReferenceMPIError("Worker runtime version drifted.")
        metadata = dict(
            runtime.generate_reference(
                output_dir=scratch,
                source_files=campaign.source_files,
                config=config,
            )
        )
        temporary_hdf5 = scratch / serial.REFERENCE_FILENAME
        import h5py

        with h5py.File(temporary_hdf5, "r+") as h5:
            h5.attrs["mpi_reference_task_schema"] = _TASK_SCHEMA_NAME
            h5.attrs[
                "mpi_reference_task_schema_version"
            ] = _TASK_SCHEMA_VERSION
            h5.attrs[
                "mpi_reference_campaign_sha256"
            ] = campaign.identity.sha256
            h5.attrs["mpi_reference_grid_point_grg"] = grid_point
            h5.attrs["mpi_reference_runtime_json"] = (
                _canonical_json_bytes(metadata).decode("ascii").rstrip("\n")
            )
            h5.attrs["mpi_reference_origin_assignment_json"] = (
                ""
                if origin_assignment_path is None
                else _canonical_json_bytes(
                    _origin_assignment_identity(
                        campaign.run_dir,
                        origin_assignment_path,
                    )
                )
                .decode("ascii")
                .rstrip("\n")
            )
            h5.flush()
        descriptor = os.open(temporary_hdf5, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        validation = _validate_task_hdf5(
            temporary_hdf5,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=grid_point,
        )
        _publish_file_exclusive(temporary_hdf5, hdf5_path)
        record = _task_record(
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
            hdf5_path=hdf5_path,
            validation=validation,
            rank=rank,
            hostname=socket.gethostname(),
            started_utc=started_utc,
            completed_utc=_utc_now(),
            wall_seconds=time.perf_counter() - started,
            disposition="computed",
            origin_assignment=(
                None
                if origin_assignment_path is None
                else _origin_assignment_identity(
                    campaign.run_dir,
                    origin_assignment_path,
                )
            ),
        )
        _write_json_exclusive(record_path, record)
        return {"grid_point_grg": grid_point, "status": "computed"}
    except Exception as exc:
        if hdf5_path.exists() and not record_path.exists():
            # A valid artifact published just before interruption is adoptable.
            try:
                recovered = _adopt_or_validate_task(campaign, grid_point)
            except Exception:
                _publish_failure(
                    campaign,
                    grid_point,
                    rank=rank,
                    started_utc=started_utc,
                    started=started,
                    error=exc,
                )
                raise
            assert recovered is not None
            status = (
                "computed"
                if str(recovered.get("disposition", "")).startswith(
                    "computed"
                )
                else "adopted"
            )
            return {"grid_point_grg": grid_point, "status": status}
        _publish_failure(
            campaign,
            grid_point,
            rank=rank,
            started_utc=started_utc,
            started=started,
            error=exc,
        )
        raise
    finally:
        _remove_owned_worker_scratch(
            campaign.run_dir,
            scratch,
            expected_identity=scratch_identity,
        )


def _copy_attrs(source: Any, destination: Any, *, omit_mpi: bool = False) -> None:
    for name in source.attrs:
        if omit_mpi and str(name).startswith("mpi_reference_"):
            continue
        destination.attrs[name] = source.attrs[name]


def _merge_frequency_points(
    task_paths: Sequence[Path],
    destination: Any,
) -> None:
    import h5py
    import numpy as np

    present: list[bool] = []
    arrays: list[Any] = []
    first_dataset_attrs: dict[str, Any] | None = None
    for path in task_paths:
        with h5py.File(path, "r") as source:
            has_values = "frequency_points" in source["gamma"]
            present.append(has_values)
            if has_values:
                dataset = source["gamma/frequency_points"]
                arrays.append(np.asarray(dataset[...]))
                if first_dataset_attrs is None:
                    first_dataset_attrs = {
                        str(name): dataset.attrs[name] for name in dataset.attrs
                    }
    if not any(present):
        return
    if not all(present):
        raise ReferenceMPIError(
            "Task gamma frequency_points presence is inconsistent."
        )
    first = arrays[0]
    # Pinned phono3py may return either a target-leading array or a common
    # target-band vector.  Preserve a common value; otherwise concatenate the
    # unique target axis exactly as emitted.
    if all(np.array_equal(array, first) for array in arrays[1:]):
        merged = first
    elif first.ndim >= 1 and first.shape[0] == 1:
        merged = np.concatenate(arrays, axis=0)
    elif first.ndim >= 3 and first.shape[-2] == 1:
        merged = np.concatenate(arrays, axis=-2)
    else:
        raise ReferenceMPIError(
            "Unsupported target-dependent gamma frequency_points layout."
        )
    dataset = destination.create_dataset(
        "frequency_points",
        data=merged,
        compression="gzip" if merged.size >= 32 else None,
        shuffle=bool(merged.size >= 32),
    )
    assert first_dataset_attrs is not None
    for name, value in first_dataset_attrs.items():
        dataset.attrs[name] = value


def merge_task_references(
    campaign: PreparedCampaign,
    destination: Path,
) -> dict[str, Any]:
    """Merge authenticated one-q HDF5 files into the serial reference schema."""

    import h5py
    import numpy as np

    if destination.exists():
        raise FileExistsError("Merged reference destination already exists.")
    task_records: list[dict[str, Any]] = []
    task_paths: list[Path] = []
    for grid_point in campaign.config.grid_points:
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
        task_records.append(
            _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=grid_point,
            )
        )
        task_paths.append(hdf5_path)
    common_hashes = {
        record["scientific"]["common_reference_sha256"]
        for record in task_records
    }
    n_qpoints = {record["scientific"]["n_qpoints"] for record in task_records}
    n_bands = {record["scientific"]["n_bands"] for record in task_records}
    if len(common_hashes) != 1:
        raise ReferenceMPIError(
            "One-q artifacts disagree in common phonon/crystal/grid fields."
        )
    if n_qpoints != {math.prod(campaign.config.mesh or ())} or len(n_bands) != 1:
        raise ReferenceMPIError(
            "One-q artifacts disagree in full-mesh phonon dimensions."
        )

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp.hdf5",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        with h5py.File(task_paths[0], "r") as first, h5py.File(
            temporary,
            "w",
        ) as target:
            _copy_attrs(first, target, omit_mpi=True)
            for name in ("crystal", "grid", "fc2_random_q"):
                first.copy(name, target)

            settings = target.create_group("settings")
            _copy_attrs(first["settings"], settings)
            for name in ("mesh", "band_indices", "temperatures", "sigmas"):
                first.copy(first[f"settings/{name}"], settings, name=name)
            grg_values = np.asarray(campaign.config.grid_points, dtype=np.int64)
            bzg_values = np.asarray(
                [
                    int(record["scientific"]["grid_point_bzg"])
                    for record in task_records
                ],
                dtype=np.int64,
            )
            settings.create_dataset("grid_points_grg", data=grg_values)
            settings.create_dataset("grid_points_bzg", data=bzg_values)

            phonons = target.create_group("phonons")
            _copy_attrs(first["phonons"], phonons)
            for name in ("frequencies", "eigenvectors", "bz_grid_addresses"):
                first.copy(first[f"phonons/{name}"], phonons, name=name)
            first_target_frequencies = np.asarray(
                first["phonons/target_band_frequencies"][...]
            )
            target_frequencies = np.empty(
                (
                    len(task_paths),
                    first_target_frequencies.shape[1],
                ),
                dtype=first_target_frequencies.dtype,
            )
            first_gamma = np.asarray(first["gamma/values"][...])
            gamma_values = np.empty(
                (
                    first_gamma.shape[0],
                    first_gamma.shape[1],
                    len(task_paths),
                    first_gamma.shape[3],
                ),
                dtype=first_gamma.dtype,
            )
            targets = target.create_group("target_q")
            for ordinal, (grid_point, path) in enumerate(
                zip(
                    campaign.config.grid_points,
                    task_paths,
                    strict=True,
                )
            ):
                with h5py.File(path, "r") as source:
                    target_slice = np.asarray(
                        source["phonons/target_band_frequencies"][...]
                    )
                    gamma_slice = np.asarray(source["gamma/values"][...])
                    if (
                        target_slice.shape
                        != (1, target_frequencies.shape[1])
                        or gamma_slice.shape
                        != (
                            gamma_values.shape[0],
                            gamma_values.shape[1],
                            1,
                            gamma_values.shape[3],
                        )
                    ):
                        raise ReferenceMPIError(
                            f"Target-axis shape drift at GRG {grid_point}."
                        )
                    target_frequencies[ordinal] = target_slice[0]
                    gamma_values[:, :, ordinal, :] = gamma_slice[:, :, 0, :]
                    group_name = f"grg_{grid_point:08d}"
                    source.copy(
                        source[f"target_q/{group_name}"],
                        targets,
                        name=group_name,
                    )
                    targets[group_name].attrs["target_ordinal"] = ordinal
            target_frequency_dataset = phonons.create_dataset(
                "target_band_frequencies",
                data=target_frequencies,
                compression=(
                    "gzip" if target_frequencies.size >= 32 else None
                ),
                shuffle=bool(target_frequencies.size >= 32),
            )
            _copy_attrs(
                first["phonons/target_band_frequencies"],
                target_frequency_dataset,
            )

            gamma = target.create_group("gamma")
            _copy_attrs(first["gamma"], gamma)
            gamma_dataset = gamma.create_dataset(
                "values",
                data=gamma_values,
                compression="gzip" if gamma_values.size >= 32 else None,
                shuffle=bool(gamma_values.size >= 32),
            )
            _copy_attrs(first["gamma/values"], gamma_dataset)
            _merge_frequency_points(task_paths, gamma)
            target.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _publish_file_exclusive(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "common_reference_sha256": next(iter(common_hashes)),
        "gamma_shape": list(gamma_values.shape),
        "target_band_frequency_shape": list(target_frequencies.shape),
        "grid_points_grg": list(campaign.config.grid_points),
        "grid_points_bzg": bzg_values.tolist(),
        "effective_band_indices": task_records[0]["scientific"]["bands"],
        "n_qpoints": next(iter(n_qpoints)),
        "n_bands": next(iter(n_bands)),
        "tasks": [
            {
                "grid_point_grg": grid_point,
                "artifact": record["artifact"],
                "record_sha256": _sha256_file(record_path),
                "runtime": record["runtime"],
            }
            for grid_point, record, (_, record_path, _) in zip(
                campaign.config.grid_points,
                task_records,
                (
                    _task_paths(campaign.run_dir, point)
                    for point in campaign.config.grid_points
                ),
                strict=True,
            )
        ],
    }


def _copy_force_stage(source: Path, destination: Path) -> None:
    destination.mkdir()
    for name in sorted(serial._FORCE_STAGE_FILENAMES):
        shutil.copyfile(source / name, destination / name)
        descriptor = os.open(destination / name, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(destination)


def _archive_member_identity(path: Path, member: str) -> dict[str, Any]:
    """Return the immutable identity used for one regular tar member."""

    _require_no_symlink_path_components(
        path,
        name=f"MPI evidence source {member}",
    )
    source = _regular_single_link_file(
        path,
        name=f"MPI evidence source {member}",
    )
    if (
        not member
        or member.startswith("/")
        or "\\" in member
        or any(part in {"", ".", ".."} for part in Path(member).parts)
    ):
        raise ReferenceMPIError(f"Unsafe MPI evidence member path: {member!r}.")
    return {
        "archive_member": member,
        "size_bytes": source.stat().st_size,
        "sha256": _sha256_file(source),
    }


def _validate_execution_record(
    campaign: PreparedCampaign,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one immutable execution before it enters release evidence."""

    normalized = dict(record)
    if set(normalized) != _EXECUTION_RECORD_KEYS:
        raise ReferenceMPIError(
            "MPI execution record has an ambiguous top-level schema."
        )
    execution_id = normalized.get("execution_id")
    if not isinstance(execution_id, str):
        raise ReferenceMPIError("MPI execution record lacks execution_id.")
    _execution_path(campaign.run_dir, execution_id)
    _validate_scratch_cleanup_record(
        campaign,
        normalized.get("scratch_cleanup"),
        execution_id=execution_id,
    )
    assignments = normalized.get("assignments")
    if not isinstance(assignments, list):
        raise ReferenceMPIError("MPI execution assignments must be an array.")
    expected_points = list(campaign.config.grid_points)
    points = [
        item.get("grid_point_grg")
        for item in assignments
        if isinstance(item, Mapping)
    ]
    statuses = [
        item.get("status") for item in assignments if isinstance(item, Mapping)
    ]
    resumed = [
        point
        for point, status in zip(points, statuses, strict=True)
        if status == "resumed"
    ]
    computed = [
        point
        for point, status in zip(points, statuses, strict=True)
        if status in {"computed", "adopted"}
    ]
    size = normalized.get("communicator_size")
    threads = normalized.get("thread_environment")
    _validate_thread_environment(
        threads,
        name=f"Execution {execution_id!r}.thread_environment",
    )
    python = normalized.get("python")
    if (
        not isinstance(python, Mapping)
        or set(python) != {"version", "executable"}
        or not isinstance(python.get("version"), str)
        or not python.get("version")
        or not isinstance(python.get("executable"), str)
        or not python.get("executable")
        or not Path(str(python["executable"])).is_absolute()
    ):
        raise ReferenceMPIError(
            f"Execution {execution_id!r}.python has an ambiguous schema."
        )
    worker_binding_evidence = (
        _normalize_execution_worker_bindings(
            assignments,
            communicator_size=size,
        )
        if isinstance(size, int) and not isinstance(size, bool) and size >= 1
        else []
    )
    if (
        normalized.get("schema_name")
        != "negfpy_phono3py_si_reference_mpi_execution"
        or normalized.get("schema_version") != MPI_REFERENCE_SCHEMA_VERSION
        or normalized.get("campaign_sha256") != campaign.identity.sha256
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 1
        or normalized.get("coordinator_rank") != 0
        or normalized.get("rank_zero_numerical_tasks") is not (size == 1)
        or normalized.get("floating_point_mpi_reductions") is not False
        or normalized.get("worker_binding_policy")
        != _PHYSICAL_CORE_BINDING_POLICY
        or normalized.get("worker_binding_evidence")
        != worker_binding_evidence
        or normalized.get("canonical_assignment_order")
        != "strictly_increasing_GRG"
        or normalized.get("task_grid_points_grg") != expected_points
        or normalized.get("assignment_grid_points_grg") != expected_points
        or points != expected_points
        or any(status not in {"computed", "adopted", "resumed"} for status in statuses)
        or normalized.get("resumed_grid_points_grg") != resumed
        or normalized.get("computed_or_adopted_grid_points_grg") != computed
        or normalized.get("exact_checkpoint_resume") is not bool(resumed)
        or normalized.get("complete") is not True
        or not isinstance(normalized.get("hostname"), str)
        or not normalized.get("hostname")
        or not isinstance(normalized.get("started_at_utc"), str)
        or not normalized.get("started_at_utc")
        or not isinstance(normalized.get("completed_at_utc"), str)
        or not normalized.get("completed_at_utc")
        or isinstance(normalized.get("wall_seconds"), bool)
        or not isinstance(normalized.get("wall_seconds"), (int, float))
        or not math.isfinite(float(normalized["wall_seconds"]))
        or float(normalized["wall_seconds"]) < 0.0
    ):
        raise ReferenceMPIError(
            f"Execution {execution_id!r} is incomplete or inconsistent."
        )
    for item in assignments:
        if not isinstance(item, Mapping):
            raise ReferenceMPIError("MPI execution assignment is not an object.")
        status = item.get("status")
        expected_assignment_keys = (
            _RESUMED_ASSIGNMENT_KEYS
            if status == "resumed"
            else _COMPUTED_ASSIGNMENT_KEYS
            | ({"worker_binding"} if size > 1 else set())
        )
        if set(item) != expected_assignment_keys:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} assignment has an ambiguous schema."
            )
        wall_seconds = item.get("wall_seconds")
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not math.isfinite(float(wall_seconds))
            or float(wall_seconds) < 0.0
        ):
            raise ReferenceMPIError(
                f"Execution {execution_id!r} assignment wall_seconds "
                "must be finite and nonnegative."
            )
        rank = item.get("rank")
        if rank is not None and (
            not isinstance(rank, int)
            or isinstance(rank, bool)
            or rank < 0
            or rank >= size
        ):
            raise ReferenceMPIError(
                f"Execution {execution_id!r} has an invalid worker rank."
            )
        if size > 1 and status != "resumed" and rank == 0:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} assigned numerical work to rank zero."
            )
        point = item.get("grid_point_grg")
        if isinstance(point, int) and not isinstance(point, bool):
            hdf5_path, record_path, _ = _task_paths(
                campaign.run_dir,
                point,
            )
            task_record = _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=point,
            )
            if item.get("origin_assignment") != task_record.get(
                "origin_assignment"
            ):
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} task origin differs from "
                    "its immutable task record."
                )
        if status in {"computed", "adopted"}:
            if not isinstance(point, int) or isinstance(point, bool):
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} has an invalid task point."
                )
            if item.get("artifact") != task_record["artifact"]:
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} assignment artifact differs "
                    "from its immutable task."
                )
            if (
                status == "adopted"
                and task_record.get("disposition")
                != "adopted_after_interrupted_record_publication"
            ):
                raise ReferenceMPIError(
                    f"Execution {execution_id!r} adoption lineage differs "
                    "from its immutable task."
                )
            if status == "computed":
                if task_record.get("disposition") != "computed":
                    raise ReferenceMPIError(
                        f"Execution {execution_id!r} computed lineage differs "
                        "from its immutable task."
                    )
                origin_identity = task_record.get("origin_assignment")
                assert isinstance(origin_identity, Mapping)
                origin = _validate_origin_assignment(
                    campaign.run_dir
                    / str(origin_identity["relative_path"]),
                    run_dir=campaign.run_dir,
                    campaign_sha256=campaign.identity.sha256,
                    grid_point=point,
                )
                if (
                    origin["execution_id"] != execution_id
                    or origin["rank"] != rank
                    or origin["communicator_size"] != size
                    or origin["worker_binding"]
                    != item.get("worker_binding")
                ):
                    raise ReferenceMPIError(
                        f"Execution {execution_id!r} fresh task origin is "
                        "mismatched."
                    )
        elif status == "resumed" and "artifact" in item:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} resumed assignment claims a fresh "
                "artifact."
            )
    return normalized


def _write_evidence_archive(
    campaign: PreparedCampaign,
    destination: Path,
    *,
    execution_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Publish a deterministic, self-contained archive of MPI task evidence."""

    if destination.exists():
        raise FileExistsError("MPI evidence archive already exists.")
    sources: list[tuple[Path, dict[str, Any]]] = []
    manifest_path = campaign.run_dir / MPI_MANIFEST_FILENAME
    campaign_manifest = _archive_member_identity(
        manifest_path,
        MPI_MANIFEST_FILENAME,
    )
    sources.append((manifest_path, campaign_manifest))

    tasks: list[dict[str, Any]] = []
    fresh_origin_points: set[int] = set()
    for grid_point in campaign.config.grid_points:
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, grid_point)
        task_record = _validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=grid_point,
        )
        if task_record.get("disposition") not in {
            "computed",
            "computed_after_interrupted_record_publication",
        }:
            raise ReferenceMPIError(
                "Release evidence requires computed tasks; legacy adoption "
                f"without an authenticated numerical origin is ineligible "
                f"for GRG {grid_point}."
            )
        origin_identity = task_record.get("origin_assignment")
        if not isinstance(origin_identity, Mapping):
            raise ReferenceMPIError(
                "Release evidence requires a durable pre-work origin for "
                f"GRG {grid_point}."
            )
        origin_path = campaign.run_dir / str(
            origin_identity["relative_path"]
        )
        origin = _validate_origin_assignment(
            origin_path,
            run_dir=campaign.run_dir,
            campaign_sha256=campaign.identity.sha256,
            grid_point=grid_point,
        )
        if (
            _origin_assignment_identity(campaign.run_dir, origin_path)
            != dict(origin_identity)
        ):
            raise ReferenceMPIError(
                f"Origin assignment for GRG {grid_point} changed."
            )
        if (
            origin["mode"] == "mpi"
            and int(origin["rank"]) > 0
            and int(origin["communicator_size"]) >= 2
        ):
            fresh_origin_points.add(grid_point)
        hdf5 = _archive_member_identity(
            hdf5_path,
            f"tasks/{hdf5_path.name}",
        )
        record = _archive_member_identity(
            record_path,
            f"tasks/{record_path.name}",
        )
        sources.extend(((hdf5_path, hdf5), (record_path, record)))
        origin_record = _archive_member_identity(
            origin_path,
            f"assignments/{origin_path.name}",
        )
        sources.append((origin_path, origin_record))
        tasks.append(
            {
                "grid_point_grg": grid_point,
                "hdf5": hdf5,
                "record": record,
                "origin_assignment": origin_record,
            }
        )

    executions: list[dict[str, Any]] = []
    normalized_executions = sorted(
        (
            _validate_execution_record(campaign, value)
            for value in execution_records
        ),
        key=lambda item: str(item["execution_id"]),
    )
    if fresh_origin_points != set(campaign.config.grid_points):
        raise ReferenceMPIError(
            "Release evidence requires one archived, task-bound pre-work "
            "MPI origin assignment for every target q-point."
        )
    for execution in normalized_executions:
        execution_id = str(execution["execution_id"])
        path = _execution_path(campaign.run_dir, execution_id)
        on_disk = _read_json_object(
            path,
            name=f"MPI execution {execution_id}",
        )
        if on_disk != execution:
            raise ReferenceMPIError(
                f"Execution {execution_id!r} differs from its immutable file."
            )
        evidence = _archive_member_identity(
            path,
            f"executions/{path.name}",
        )
        sources.append((path, evidence))
        cleanup_path, _ = _validate_scratch_cleanup_record(
            campaign,
            execution.get("scratch_cleanup"),
            execution_id=execution_id,
        )
        cleanup_evidence = _archive_member_identity(
            cleanup_path,
            f"scratch-cleanups/{cleanup_path.name}",
        )
        sources.append((cleanup_path, cleanup_evidence))
        executions.append(
            {
                "execution_id": execution_id,
                "record": evidence,
                "scratch_cleanup": cleanup_evidence,
            }
        )
    if not executions:
        raise ReferenceMPIError("MPI reference evidence has no execution record.")

    with tarfile.open(
        destination,
        mode="w",
        format=tarfile.USTAR_FORMAT,
    ) as archive:
        for source, identity in sources:
            info = tarfile.TarInfo(str(identity["archive_member"]))
            info.size = int(identity["size_bytes"])
            info.mode = 0o444
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            _require_no_symlink_path_components(
                source,
                name=f"MPI evidence source {info.name}",
            )
            descriptor = os.open(
                source,
                os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC,
            )
            try:
                details = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(details.st_mode)
                    or details.st_nlink != 1
                    or details.st_size != info.size
                    or _sha256_descriptor(descriptor) != identity["sha256"]
                ):
                    raise ReferenceMPIError(
                        f"MPI evidence source {info.name} changed before archive."
                    )
                os.lseek(descriptor, 0, os.SEEK_SET)
                with os.fdopen(os.dup(descriptor), "rb") as stream:
                    archive.addfile(info, stream)
            finally:
                os.close(descriptor)
    descriptor = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

    resumed_ids = [
        str(record["execution_id"])
        for record in normalized_executions
        if record["resumed_grid_points_grg"]
    ]
    resumed_counts = {
        str(record["execution_id"]): len(record["resumed_grid_points_grg"])
        for record in normalized_executions
    }
    return {
        "archive": _file_identity(destination),
        "campaign_manifest": campaign_manifest,
        "tasks": tasks,
        "executions": executions,
        "restart": {
            "exact_checkpoint_resume_supported": True,
            "execution_count": len(executions),
            "execution_ids": [
                str(record["execution_id"])
                for record in normalized_executions
            ],
            "executions_with_resumed_tasks": resumed_ids,
            "resumed_task_count_by_execution": resumed_counts,
            "total_resumed_task_assignments": sum(resumed_counts.values()),
        },
    }


def _reference_runtime_metadata(
    campaign: PreparedCampaign,
    merge: Mapping[str, Any],
    orchestration: Mapping[str, Any],
    orchestration_sha256: str,
) -> dict[str, Any]:
    first_runtime = merge["tasks"][0]["runtime"]
    if not isinstance(first_runtime, Mapping):
        raise ReferenceMPIError("Task runtime metadata is invalid.")
    targets: list[dict[str, Any]] = []
    for item in merge["tasks"]:
        runtime = item["runtime"]
        if not isinstance(runtime, Mapping):
            raise ReferenceMPIError("Task runtime metadata is invalid.")
        summaries = runtime.get("targets")
        if not isinstance(summaries, list) or len(summaries) != 1:
            raise ReferenceMPIError("Task runtime lacks one target summary.")
        targets.append(dict(summaries[0]))
    metadata = dict(first_runtime)
    metadata.update(
        {
            "implementation": "phono3py_4_4_0_task_level_mpi",
            "mesh": list(campaign.config.mesh or ()),
            "grid_points_grg": list(campaign.config.grid_points),
            "grid_points_bzg": list(merge["grid_points_bzg"]),
            "band_indices": list(merge["effective_band_indices"]),
            "gamma_shape": list(merge["gamma_shape"]),
            "targets": targets,
            "mpi_orchestration": {
                "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
                "campaign_sha256": campaign.identity.sha256,
                "orchestration_record_sha256": orchestration_sha256,
                "task_count": len(targets),
                "execution_count": len(orchestration["executions"]),
                "one_complete_target_q_per_worker": True,
                "floating_point_mpi_reductions": False,
                "canonical_merge_order": "strictly_increasing_GRG",
                "full_q_selection": orchestration["selection"][
                    "full_q_selection"
                ],
                "full_band_selection": orchestration["selection"][
                    "full_band_selection"
                ],
                "evidence_archive_sha256": orchestration[
                    "evidence_archive"
                ]["sha256"],
                "restart": orchestration["restart"],
            },
        }
    )
    return metadata


def _finalize_reference_bundle_locked(
    campaign: PreparedCampaign,
    *,
    execution_records: Sequence[Mapping[str, Any]],
    execution_lock: _CampaignExecutionLock,
) -> Path:
    execution_lock.verify()
    final_output_name = _authenticated_final_output_name(campaign)
    _require_no_symlink_path_components(
        campaign.output_dir,
        name="output_dir",
    )
    if campaign.output_dir.exists():
        raise FileExistsError("Final output directory already exists.")
    parent = campaign.output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent = _require_no_symlink_path_components(
        parent,
        name="output_dir parent",
    )
    parent_descriptor = _open_directory_nofollow(
        parent,
        name="output_dir parent",
    )
    staging_name: str | None = None
    for _ in range(100):
        candidate = (
            f".{final_output_name}.{secrets.token_hex(16)}.tmp"
        )
        try:
            os.mkdir(candidate, mode=0o700, dir_fd=parent_descriptor)
        except FileExistsError:
            continue
        staging_name = candidate
        break
    if staging_name is None:
        os.close(parent_descriptor)
        raise ReferenceMPIError("Unable to allocate a private staging directory.")
    staging = parent / staging_name
    staging_details = os.stat(
        staging_name,
        dir_fd=parent_descriptor,
        follow_symlinks=False,
    )
    staging_identity = (staging_details.st_dev, staging_details.st_ino)
    renamed = False
    try:
        _verify_directory_descriptor(
            parent,
            parent_descriptor,
            name="output_dir parent",
        )
        for name in sorted(serial._FORCE_STAGE_FILENAMES):
            shutil.copyfile(
                campaign.force_stage_dir / name,
                staging / name,
            )
        merged_path = staging / serial.REFERENCE_FILENAME
        merge = merge_task_references(campaign, merged_path)
        evidence = _write_evidence_archive(
            campaign,
            staging / MPI_EVIDENCE_ARCHIVE_FILENAME,
            execution_records=execution_records,
        )
        full_q_selection = list(campaign.config.grid_points) == list(
            range(int(merge["n_qpoints"]))
        )
        full_band_selection = list(merge["effective_band_indices"]) == list(
            range(int(merge["n_bands"]))
        )
        orchestration = {
            "schema_name": "negfpy_phono3py_si_reference_mpi_orchestration",
            "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
            "campaign_sha256": campaign.identity.sha256,
            "generators": {
                "serial_scientific_generator": dict(
                    campaign.identity.payload["serial_generator"]
                ),
                "mpi_task_and_merge_driver": dict(
                    campaign.identity.payload["mpi_driver"]
                ),
            },
            "campaign_manifest": evidence["campaign_manifest"],
            "scheduler": {
                "parallel_boundary": "complete_target_q",
                "rank_zero_numerical_tasks": False,
                "floating_point_mpi_reductions": False,
                "canonical_merge_order": "strictly_increasing_GRG",
                "worker_thread_limit": 1,
                "worker_binding_policy": _PHYSICAL_CORE_BINDING_POLICY,
            },
            "selection": {
                "mesh": list(campaign.config.mesh or ()),
                "n_qpoints": int(merge["n_qpoints"]),
                "n_bands": int(merge["n_bands"]),
                "grid_points_grg": list(campaign.config.grid_points),
                "band_indices": list(merge["effective_band_indices"]),
                "full_q_selection": full_q_selection,
                "full_band_selection": full_band_selection,
            },
            "evidence_archive": evidence["archive"],
            "tasks": evidence["tasks"],
            "executions": evidence["executions"],
            "restart": evidence["restart"],
            "merged_reference": _file_identity(merged_path),
            "merge": {
                key: value for key, value in merge.items() if key != "tasks"
            },
        }
        _write_json_exclusive(
            staging / MPI_ORCHESTRATION_FILENAME,
            orchestration,
        )
        orchestration_sha256 = _sha256_file(
            staging / MPI_ORCHESTRATION_FILENAME
        )
        _write_json_exclusive(
            staging / "reference-settings.json",
            campaign.config.reference_settings(),
        )
        _write_json_exclusive(
            staging / "reference-environment.json",
            serial._environment_record(campaign.environment_identity),
        )
        runtime_metadata = _reference_runtime_metadata(
            campaign,
            merge,
            orchestration,
            orchestration_sha256,
        )
        _write_json_exclusive(
            staging / "reference-stage.json",
            {
                "schema_version": serial.REFERENCE_SCHEMA_VERSION,
                "stage": "reference",
                "completed_at_utc": _utc_now(),
                "status": "complete",
                "runtime": runtime_metadata,
            },
        )
        _write_json_exclusive(
            staging / "inventory.json",
            serial._inventory(
                staging,
                exclude={"inventory.json", "SHA256SUMS"},
            ),
        )
        names = sorted(
            path.name for path in staging.iterdir() if path.name != "SHA256SUMS"
        )
        serial._write_sha256sums_exclusive(
            staging / "SHA256SUMS",
            root=staging,
            names=names,
        )
        if not serial.verify_sha256sums(staging / "SHA256SUMS", root=staging):
            raise ReferenceMPIError("Final bundle checksum self-check failed.")
        _fsync_directory(staging)
        _verify_directory_descriptor(
            parent,
            parent_descriptor,
            name="output_dir parent",
        )
        try:
            os.stat(
                final_output_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError("Final output directory already exists.")
        os.rename(
            staging_name,
            final_output_name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        renamed = True
        published = os.stat(
            final_output_name,
            dir_fd=parent_descriptor,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(published.st_mode)
            or (published.st_dev, published.st_ino) != staging_identity
        ):
            raise ReferenceMPIError(
                "Final output directory changed during publication."
            )
        _verify_directory_descriptor(
            parent,
            parent_descriptor,
            name="output_dir parent",
        )
        os.fsync(parent_descriptor)
    finally:
        if not renamed:
            try:
                remaining = os.stat(
                    staging_name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            else:
                if (
                    not stat.S_ISDIR(remaining.st_mode)
                    or (remaining.st_dev, remaining.st_ino)
                    != staging_identity
                ):
                    os.close(parent_descriptor)
                    raise ReferenceMPIError(
                        "Final staging directory changed before cleanup."
                    )
                _remove_directory_tree_at(
                    parent_descriptor,
                    staging_name,
                    expected_identity=staging_identity,
                )
        os.close(parent_descriptor)
    execution_lock.verify()
    return campaign.output_dir


def _finalize_reference_bundle(
    campaign: PreparedCampaign,
    *,
    execution_records: Sequence[Mapping[str, Any]],
    execution_lock: _CampaignExecutionLock | None = None,
) -> Path:
    """Finalize while holding the campaign-wide mutation lock."""

    owned_lock = execution_lock
    acquired_here = owned_lock is None
    if owned_lock is None:
        owned_lock = _CampaignExecutionLock.acquire(campaign.run_dir)
    try:
        return _finalize_reference_bundle_locked(
            campaign,
            execution_records=execution_records,
            execution_lock=owned_lock,
        )
    finally:
        if acquired_here:
            owned_lock.close()


def _execution_path(run_dir: Path, execution_id: str) -> Path:
    if (
        not execution_id
        or len(execution_id) > 64
        or execution_id in {".", ".."}
        or "/" in execution_id
        or "\\" in execution_id
        or any(
            not (character.isalnum() or character in "._-")
            for character in execution_id
        )
    ):
        raise ValueError(
            "execution_id must be a safe identifier of at most 64 characters."
        )
    return run_dir / "executions" / f"{execution_id}.json"


def _execution_record(
    *,
    campaign: PreparedCampaign,
    execution_id: str,
    size: int,
    started_utc: str,
    started: float,
    assignments: Sequence[Mapping[str, Any]],
    complete: bool,
    scratch_cleanup: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_assignments = [dict(item) for item in assignments]
    points = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if isinstance(item.get("grid_point_grg"), int)
    ]
    resumed = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if item.get("status") == "resumed"
    ]
    computed = [
        int(item["grid_point_grg"])
        for item in normalized_assignments
        if item.get("status") in {"computed", "adopted"}
    ]
    worker_binding_evidence = _normalize_execution_worker_bindings(
        normalized_assignments,
        communicator_size=size,
    )
    return {
        "schema_name": "negfpy_phono3py_si_reference_mpi_execution",
        "schema_version": MPI_REFERENCE_SCHEMA_VERSION,
        "execution_id": execution_id,
        "campaign_sha256": campaign.identity.sha256,
        "started_at_utc": started_utc,
        "completed_at_utc": _utc_now(),
        "wall_seconds": time.perf_counter() - started,
        "communicator_size": size,
        "coordinator_rank": 0,
        "rank_zero_numerical_tasks": size == 1,
        "floating_point_mpi_reductions": False,
        "worker_binding_policy": _PHYSICAL_CORE_BINDING_POLICY,
        "worker_binding_evidence": worker_binding_evidence,
        "canonical_assignment_order": "strictly_increasing_GRG",
        "task_grid_points_grg": list(campaign.config.grid_points),
        "assignment_grid_points_grg": points,
        "resumed_grid_points_grg": resumed,
        "computed_or_adopted_grid_points_grg": computed,
        "exact_checkpoint_resume": bool(resumed),
        "hostname": socket.gethostname(),
        "python": {
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "thread_environment": _require_threads_one(),
        "scratch_cleanup": dict(scratch_cleanup),
        "assignments": normalized_assignments,
        "complete": complete,
    }


_TAG_TASK = 3101
_TAG_RESULT = 3102
_TAG_STOP = 3103


def _serial_schedule(
    campaign: PreparedCampaign,
    *,
    runtime_factory: Callable[[], Any],
    execution_id: str,
) -> list[dict[str, Any]]:
    assignments = []
    for point in campaign.config.grid_points:
        started = time.perf_counter()
        existing = _adopt_or_validate_task(campaign, point)
        if existing is not None:
            assignments.append(
                {
                    "grid_point_grg": point,
                    "rank": None,
                    "status": "resumed",
                    "wall_seconds": time.perf_counter() - started,
                    "origin_assignment": existing.get(
                        "origin_assignment"
                    ),
                }
            )
            continue
        origin_path, _ = _publish_origin_assignment(
            campaign,
            point,
            execution_id=execution_id,
            rank=0,
            communicator_size=1,
            worker_binding=None,
        )
        result = _generate_one_task(
            campaign,
            point,
            rank=0,
            runtime_factory=runtime_factory,
            origin_assignment_path=origin_path,
        )
        hdf5_path, record_path, _ = _task_paths(campaign.run_dir, point)
        task_record = _validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=point,
        )
        assignments.append(
            {
                "grid_point_grg": point,
                "rank": 0,
                "status": result["status"],
                "wall_seconds": time.perf_counter() - started,
                "artifact": dict(task_record["artifact"]),
                "origin_assignment": dict(
                    task_record["origin_assignment"]
                ),
            }
        )
    return assignments


def _worker_loop(
    campaign: PreparedCampaign,
    comm: _Communicator,
    *,
    rank: int,
    runtime_factory: Callable[[], Any],
) -> None:
    while True:
        message = comm.recv(source=0)
        if not isinstance(message, Mapping):
            raise ReferenceMPIError("Worker received a malformed command.")
        command = message.get("command")
        if command == "stop":
            return
        if command != "task" or not isinstance(
            message.get("grid_point_grg"),
            int,
        ):
            raise ReferenceMPIError("Worker received an unknown command.")
        point = int(message["grid_point_grg"])
        execution_id = message.get("execution_id")
        communicator_size = message.get("communicator_size")
        started = time.perf_counter()
        try:
            worker_binding = _physical_core_binding(rank)
            if (
                not isinstance(execution_id, str)
                or not isinstance(communicator_size, int)
                or isinstance(communicator_size, bool)
                or communicator_size != comm.Get_size()
            ):
                raise ReferenceMPIError(
                    "Worker task command lacks execution provenance."
                )
            origin_path, _ = _publish_origin_assignment(
                campaign,
                point,
                execution_id=execution_id,
                rank=rank,
                communicator_size=communicator_size,
                worker_binding=worker_binding,
            )
            result = _generate_one_task(
                campaign,
                point,
                rank=rank,
                runtime_factory=runtime_factory,
                origin_assignment_path=origin_path,
            )
            hdf5_path, record_path, _ = _task_paths(
                campaign.run_dir,
                point,
            )
            task_record = _validate_task_record(
                record_path,
                hdf5_path,
                campaign_sha256=campaign.identity.sha256,
                config=campaign.config,
                grid_point=point,
            )
            response = {
                "grid_point_grg": point,
                "status": result["status"],
                "wall_seconds": time.perf_counter() - started,
                "worker_binding": worker_binding,
                "artifact": dict(task_record["artifact"]),
                "origin_assignment": dict(
                    task_record["origin_assignment"]
                ),
            }
        except Exception as exc:  # noqa: BLE001 - worker reports all failures
            response = {
                "grid_point_grg": point,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "wall_seconds": time.perf_counter() - started,
            }
        comm.send(response, dest=0, tag=_TAG_RESULT)


def _mpi_schedule(
    campaign: PreparedCampaign,
    comm: _Communicator,
    *,
    runtime_factory: Callable[[], Any],
    execution_id: str,
) -> list[dict[str, Any]] | None:
    from mpi4py import MPI

    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank != 0:
        _worker_loop(
            campaign,
            comm,
            rank=rank,
            runtime_factory=runtime_factory,
        )
        return None

    pending: list[int] = []
    assignments: list[dict[str, Any]] = []
    for point in campaign.config.grid_points:
        try:
            existing = _adopt_or_validate_task(campaign, point)
        except ReferenceMPIError as exc:
            assignments.append(
                {
                    "grid_point_grg": point,
                    "rank": None,
                    "status": "failed",
                    "wall_seconds": 0.0,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            continue
        if existing is None:
            pending.append(point)
        else:
            assignments.append(
                {
                    "grid_point_grg": point,
                    "rank": None,
                    "status": "resumed",
                    "wall_seconds": 0.0,
                    "origin_assignment": existing.get(
                        "origin_assignment"
                    ),
                }
            )
    workers = list(range(1, size))
    in_flight: dict[int, int] = {}
    for worker in workers:
        if not pending:
            break
        point = pending.pop(0)
        comm.send(
            {
                "command": "task",
                "grid_point_grg": point,
                "execution_id": execution_id,
                "communicator_size": size,
            },
            dest=worker,
            tag=_TAG_TASK,
        )
        in_flight[worker] = point
    failures: list[dict[str, Any]] = []
    while in_flight:
        status = MPI.Status()
        response = comm.recv(
            source=MPI.ANY_SOURCE,
            tag=_TAG_RESULT,
            status=status,
        )
        worker = int(status.Get_source())
        expected = in_flight.pop(worker)
        if (
            not isinstance(response, Mapping)
            or response.get("grid_point_grg") != expected
        ):
            failures.append(
                {
                    "grid_point_grg": expected,
                    "rank": worker,
                    "status": "protocol_failure",
                    "wall_seconds": None,
                }
            )
        else:
            record = {
                "grid_point_grg": expected,
                "rank": worker,
                "status": response.get("status"),
                "wall_seconds": response.get("wall_seconds"),
                "worker_binding": response.get("worker_binding"),
                "artifact": response.get("artifact"),
                "origin_assignment": response.get("origin_assignment"),
            }
            assignments.append(record)
            if response.get("status") == "failed":
                failures.append({**record, **dict(response)})
        if pending:
            point = pending.pop(0)
            comm.send(
                {
                    "command": "task",
                    "grid_point_grg": point,
                    "execution_id": execution_id,
                    "communicator_size": size,
                },
                dest=worker,
                tag=_TAG_TASK,
            )
            in_flight[worker] = point
    for worker in workers:
        comm.send({"command": "stop"}, dest=worker, tag=_TAG_STOP)
    return sorted(
        assignments,
        key=lambda item: int(item["grid_point_grg"]),
    )


def _execute_campaign_under_lock(
    campaign: PreparedCampaign,
    *,
    comm: _Communicator,
    runtime_factory: Callable[[], Any],
    execution_id: str,
    execution_lock: _CampaignExecutionLock | None,
) -> Path | None:
    rank = comm.Get_rank()
    size = comm.Get_size()
    execution_path = _execution_path(campaign.run_dir, execution_id)
    execution_preflight_error = (
        (
            "Execution record already exists: "
            f"{execution_path.name}."
        )
        if rank == 0 and execution_path.exists()
        else None
    )
    execution_preflight_error = comm.bcast(
        execution_preflight_error,
        root=0,
    )
    if execution_preflight_error is not None:
        raise FileExistsError(execution_preflight_error)
    scratch_cleanup_path: Path | None = None
    scratch_cleanup_error: str | None = None
    if rank == 0:
        try:
            scratch_cleanup_path, _ = _cleanup_interrupted_scratch(
                campaign,
                execution_id=execution_id,
                execution_lock=execution_lock,
            )
        except Exception as exc:  # noqa: BLE001 - broadcast rank-zero failure
            scratch_cleanup_error = f"{type(exc).__name__}: {exc}"
    scratch_cleanup_error = comm.bcast(
        scratch_cleanup_error,
        root=0,
    )
    if scratch_cleanup_error is not None:
        raise ReferenceMPIError(
            f"Scratch cleanup preflight failed: {scratch_cleanup_error}"
        )
    started_utc = _utc_now()
    started = time.perf_counter()
    assignments: list[dict[str, Any]] | None
    if size == 1:
        assignments = _serial_schedule(
            campaign,
            runtime_factory=runtime_factory,
            execution_id=execution_id,
        )
    else:
        assignments = _mpi_schedule(
            campaign,
            comm,
            runtime_factory=runtime_factory,
            execution_id=execution_id,
        )
    comm.Barrier()
    if rank != 0:
        return None
    assert assignments is not None
    if any((campaign.run_dir / "scratch").iterdir()):
        raise ReferenceMPIError(
            "Completed campaign retained worker scratch."
        )
    validation_failures: list[int] = []
    for point in campaign.config.grid_points:
        try:
            if _adopt_or_validate_task(campaign, point) is None:
                validation_failures.append(point)
        except ReferenceMPIError:
            validation_failures.append(point)
    complete_statuses = {"computed", "adopted", "resumed"}
    assignment_points = [
        item.get("grid_point_grg") for item in assignments
    ]
    complete = bool(
        not validation_failures
        and assignment_points == list(campaign.config.grid_points)
        and all(
            item.get("status") in complete_statuses for item in assignments
        )
    )
    assert scratch_cleanup_path is not None
    record = _execution_record(
        campaign=campaign,
        execution_id=execution_id,
        size=size,
        started_utc=started_utc,
        started=started,
        assignments=assignments,
        complete=complete,
        scratch_cleanup=_file_identity(scratch_cleanup_path),
    )
    _write_json_exclusive(execution_path, record)
    if not complete:
        raise ReferenceMPIError(
            "Campaign ended without all authenticated target artifacts: "
            f"{validation_failures!r}."
        )
    execution_records = [
        _read_json_object(path, name=f"execution record {path.name}")
        for path in sorted((campaign.run_dir / "executions").glob("*.json"))
    ]
    return _finalize_reference_bundle(
        campaign,
        execution_records=execution_records,
        execution_lock=execution_lock,
    )


def execute_campaign(
    campaign: PreparedCampaign,
    *,
    comm: _Communicator | None = None,
    runtime_factory: Callable[[], Any] = serial._Phono3pyRuntime,
    execution_id: str,
    require_mpi: bool = False,
) -> Path | None:
    """Execute under one fail-fast lock from cleanup through finalization."""

    if comm is None:
        if require_mpi:
            try:
                from mpi4py import MPI
            except ImportError as exc:
                raise ReferenceMPIError(
                    "MPI execution requires the optional mpi4py dependency."
                ) from exc
            comm = MPI.COMM_WORLD
        else:
            comm = _SerialCommunicator()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if require_mpi and size < 2:
        raise ReferenceMPIError(
            "Certified MPI reference generation requires at least two ranks."
        )
    _require_threads_one()
    execution_lock: _CampaignExecutionLock | None = None
    lock_error: str | None = None
    if rank == 0:
        try:
            execution_lock = _CampaignExecutionLock.acquire(campaign.run_dir)
        except Exception as exc:  # noqa: BLE001 - broadcast rank-zero failure
            lock_error = f"{type(exc).__name__}: {exc}"
    lock_error = comm.bcast(lock_error, root=0)
    if lock_error is not None:
        raise ReferenceMPIError(
            f"Campaign execution lock acquisition failed: {lock_error}"
        )
    try:
        return _execute_campaign_under_lock(
            campaign,
            comm=comm,
            runtime_factory=runtime_factory,
            execution_id=execution_id,
            execution_lock=execution_lock,
        )
    finally:
        if execution_lock is not None:
            execution_lock.close()


class _SerialCommunicator:
    def Get_rank(self) -> int:
        return 0

    def Get_size(self) -> int:
        return 1

    def bcast(self, value: Any, root: int = 0) -> Any:
        return value

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        raise RuntimeError("Serial communicator cannot send.")

    def recv(
        self,
        source: int = -1,
        tag: int = -1,
        status: Any | None = None,
    ) -> Any:
        raise RuntimeError("Serial communicator cannot receive.")

    def Barrier(self) -> None:
        return None


def _broadcast_prepared(
    comm: _Communicator,
    *,
    arguments: argparse.Namespace,
    config: serial.GeneratorConfig,
) -> PreparedCampaign:
    rank = comm.Get_rank()
    if rank == 0:
        try:
            prepared = prepare_campaign(
                manifest_path=arguments.manifest,
                source_dir=arguments.source_dir,
                force_stage_dir=arguments.force_stage_dir,
                run_dir=arguments.run_dir,
                output_dir=arguments.output_dir,
                config=config,
            )
            descriptor = {
                "ok": True,
                "campaign_sha256": prepared.identity.sha256,
                "payload": prepared.identity.payload,
                "source_files": {
                    name: str(path)
                    for name, path in prepared.source_files.items()
                },
                "source_verification": prepared.source_verification,
                "environment_identity": prepared.environment_identity,
            }
        except Exception as exc:  # noqa: BLE001 - broadcast rank-zero failure
            descriptor = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
    else:
        descriptor = None
    descriptor = comm.bcast(descriptor, root=0)
    if not descriptor.get("ok"):
        raise ReferenceMPIError(
            "Rank-zero campaign preflight failed: "
            f"{descriptor.get('error_type')}: {descriptor.get('error_message')}"
        )
    identity = CampaignIdentity.from_payload(descriptor["payload"])
    if identity.sha256 != descriptor["campaign_sha256"]:
        raise ReferenceMPIError("Broadcast campaign hash is invalid.")
    prepared = PreparedCampaign(
        identity=identity,
        config=config,
        source_files={
            name: Path(path)
            for name, path in descriptor["source_files"].items()
        },
        source_verification=descriptor["source_verification"],
        environment_identity=descriptor["environment_identity"],
        force_stage_dir=arguments.force_stage_dir,
        run_dir=arguments.run_dir,
        output_dir=arguments.output_dir,
    )
    _authenticated_final_output_name(prepared)
    return prepared


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an authenticated phono3py Si reference with one complete "
            "target q-point per MPI worker."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--force-stage-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--execution-id", required=True)
    parser.add_argument(
        "--mesh",
        nargs=3,
        required=True,
        type=int,
        metavar=("NX", "NY", "NZ"),
    )
    target_selection = parser.add_mutually_exclusive_group(required=True)
    target_selection.add_argument(
        "--grid-points",
        nargs="+",
        type=int,
        metavar="GRG_INDEX",
    )
    target_selection.add_argument(
        "--all-grid-points",
        action="store_true",
        help="Generate every GRG target in the configured full mesh.",
    )
    parser.add_argument("--band-indices", nargs="+", type=int, default=None)
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[300.0],
        metavar="K",
    )
    parser.add_argument("--sigma", type=float, default=0.1, metavar="THZ")
    parser.add_argument("--sigma-cutoff", type=float, default=None)
    parser.add_argument("--backend", choices=("C", "Rust"), default="Rust")
    parser.add_argument(
        "--make-r0-average",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc3q",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = serial.GeneratorConfig(
        mesh=arguments.mesh,
        grid_points=(
            tuple(range(math.prod(arguments.mesh)))
            if arguments.all_grid_points
            else arguments.grid_points
        ),
        band_indices=arguments.band_indices,
        temperatures=arguments.temperatures,
        sigma=arguments.sigma,
        sigma_cutoff=arguments.sigma_cutoff,
        backend=arguments.backend,
        make_r0_average=arguments.make_r0_average,
        symmetrize_fc=arguments.symmetrize_fc,
        symmetrize_fc3q=arguments.symmetrize_fc3q,
    )
    try:
        from mpi4py import MPI
    except ImportError as exc:
        parser.error(
            "This MPI generator requires the optional mpi4py dependency: "
            f"{exc}"
        )
    campaign = _broadcast_prepared(
        MPI.COMM_WORLD,
        arguments=arguments,
        config=config,
    )
    output = execute_campaign(
        campaign,
        comm=MPI.COMM_WORLD,
        execution_id=arguments.execution_id,
        require_mpi=True,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert output is not None
        print(
            json.dumps(
                {
                    "campaign_sha256": campaign.identity.sha256,
                    "communicator_size": MPI.COMM_WORLD.Get_size(),
                    "output_dir": str(output.resolve()),
                    "reference_hdf5_sha256": _sha256_file(
                        output / serial.REFERENCE_FILENAME
                    ),
                    "sha256sums": str((output / "SHA256SUMS").resolve()),
                    "status": "complete",
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
