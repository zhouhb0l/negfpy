#!/usr/bin/env python3
"""Key-only deployment and execution of NEGFPy heavy validation jobs.

Only the preconfigured ``negf-heavy`` SSH alias is supported. Every network
operation forces batch public-key authentication and strict host-key checking.
No hostname, username, identity, or credential can be supplied to this tool.
"""

from __future__ import annotations

import base64
import argparse
import dataclasses
import datetime as dt
import hashlib
import hmac
import importlib.metadata
import json
import os
import pathlib
import platform
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile

from remote_heavy_security import validate_known_hosts_fingerprint
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SSH_ALIAS = "negf-heavy"
EXPECTED_HOST = "172.20.86.67"
EXPECTED_USER = "hangbo"
EXPECTED_IDENTITY_BASENAME = "id_ed25519_aspire2a"
EXPECTED_IDENTITY_FILE = (
    pathlib.Path.home() / ".ssh" / EXPECTED_IDENTITY_BASENAME
).resolve(strict=False)
EXPECTED_KNOWN_HOSTS_FILE = (
    pathlib.Path.home() / ".ssh" / "known_hosts_negf_heavy"
).resolve(strict=False)
EXPECTED_ED25519_FINGERPRINT = (
    "SHA256:sJide2AQhTe8oH2XWtayraDdUsde/LV2HOfUeMeWd70"
)
EXPECTED_ENV_LOCK_SHA256 = (
    "aca1a3f7eb7675adb5982d571005cc5ae65ed442b4972ad835f2d4fdf6fa33c1"
)
REMOTE_ROOT_RELATIVE = "negfpy-validation"
REMOTE_TOOLS_BIN = "/home/hangbo/micromamba/envs/negf-tools/bin"
REMOTE_MICROMAMBA = "/home/hangbo/.local/bin/micromamba"
REMOTE_GIT = f"{REMOTE_TOOLS_BIN}/git"
REMOTE_TMUX = f"{REMOTE_TOOLS_BIN}/tmux"
REMOTE_RSYNC = f"{REMOTE_TOOLS_BIN}/rsync"
REQUIREMENTS_DISTRIBUTIONS = (
    "mpi4py",
    "numpy",
    "scipy",
    "threadpoolctl",
    "h5py",
    "pytest",
)
EXPECTED_MPI_VENDOR = "Open MPI"
EXPECTED_MPI_VERSION = (5, 0, 10)
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
SSH_OPTIONS = (
    "-oBatchMode=yes",
    "-oStrictHostKeyChecking=yes",
    "-oIdentitiesOnly=yes",
    f"-oIdentityFile={EXPECTED_IDENTITY_FILE}",
    f"-oUserKnownHostsFile={EXPECTED_KNOWN_HOSTS_FILE}",
    "-oGlobalKnownHostsFile=/dev/null",
    "-oHostKeyAlgorithms=ssh-ed25519",
    "-oPubkeyAuthentication=yes",
    "-oPreferredAuthentications=publickey",
    "-oPasswordAuthentication=no",
    "-oKbdInteractiveAuthentication=no",
    "-oGSSAPIAuthentication=no",
    "-oHostbasedAuthentication=no",
    f"-oHostName={EXPECTED_HOST}",
    f"-oUser={EXPECTED_USER}",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SESSION_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


class RemoteValidationError(RuntimeError):
    """Raised when a deployment invariant is not satisfied."""


@dataclasses.dataclass(frozen=True)
class Completed:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


class CommandRunner:
    """Execute without a local shell, or print exact dry-run commands."""

    def __init__(self, *, dry_run: bool = False) -> None:
        self.dry_run = bool(dry_run)
        self.commands: list[tuple[str, ...]] = []

    def run(
        self,
        argv: Sequence[str],
        *,
        cwd: pathlib.Path | None = None,
        check: bool = True,
        env: Mapping[str, str] | None = None,
    ) -> Completed:
        command = tuple(str(item) for item in argv)
        self.commands.append(command)
        if self.dry_run:
            print("+ " + shlex.join(command))
            return Completed(command, 0, "", "")
        process = subprocess.run(
            command,
            cwd=None if cwd is None else str(cwd),
            check=False,
            capture_output=True,
            text=True,
            env=None if env is None else dict(env),
        )
        result = Completed(
            command,
            process.returncode,
            process.stdout or "",
            process.stderr or "",
        )
        if check and result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise RemoteValidationError(
                f"command failed ({result.returncode}): "
                f"{shlex.join(command)}\n{detail}"
            )
        return result


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fsync_directory(path: pathlib.Path) -> None:
    """Make a preceding directory-entry update durable."""

    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_copy_file(source: pathlib.Path, destination: pathlib.Path) -> None:
    """Copy a regular file and durably publish it under ``destination``."""

    if not source.is_file() or source.is_symlink():
        raise RemoteValidationError(
            f"identity source is not a regular file: {source}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: pathlib.Path | None = None
    try:
        with source.open("rb") as input_stream, tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_stream:
            temporary = pathlib.Path(output_stream.name)
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_write_json(path: pathlib.Path, value: Mapping[str, Any]) -> None:
    _atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_commit(commit: str) -> str:
    candidate = commit.strip().lower()
    if not _COMMIT_RE.fullmatch(candidate):
        raise RemoteValidationError(
            f"expected a full 40-character lowercase Git commit, got {commit!r}"
        )
    return candidate


def _validate_session(session: str) -> str:
    if not _SESSION_RE.fullmatch(session):
        raise RemoteValidationError(
            "tmux session may contain only letters, digits, '.', '_' and '-'"
        )
    return session


def _remote_relative_dir(commit: str) -> str:
    return f"{REMOTE_ROOT_RELATIVE}/{_validate_commit(commit)}"


def _remote_shell_dir(commit: str) -> str:
    return f'"$HOME/{_remote_relative_dir(commit)}"'


def ssh_argv(remote_command: str | None = None) -> tuple[str, ...]:
    """Return an invariant key-only SSH invocation."""

    base = ("ssh", *SSH_OPTIONS, SSH_ALIAS)
    return base if remote_command is None else (*base, remote_command)


def scp_argv(*paths: str) -> tuple[str, ...]:
    """Return an invariant key-only SCP invocation."""

    return ("scp", *SSH_OPTIONS, *paths)


def _parse_ssh_g(output: str) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    for raw_line in output.splitlines():
        stripped = raw_line.strip()
        if stripped and " " in stripped:
            key, value = stripped.split(None, 1)
            values.setdefault(key.lower(), []).append(value.strip())
    return values


def validate_ssh_alias_config(output: str) -> dict[str, Any]:
    """Validate the effective ``ssh -G negf-heavy`` configuration."""

    values = _parse_ssh_g(output)

    def first(key: str) -> str:
        entries = values.get(key, [])
        if not entries:
            raise RemoteValidationError(
                f"SSH alias {SSH_ALIAS!r} does not define {key}"
            )
        return entries[0]

    problems: list[str] = []
    if first("hostname") != EXPECTED_HOST:
        problems.append(f"HostName must be {EXPECTED_HOST}")
    if first("user") != EXPECTED_USER:
        problems.append(f"User must be {EXPECTED_USER}")
    if first("identitiesonly").lower() not in {"yes", "true"}:
        problems.append("IdentitiesOnly must be yes")
    if first("batchmode").lower() not in {"yes", "true"}:
        problems.append("BatchMode must be yes")
    if first("stricthostkeychecking").lower() not in {"yes", "true"}:
        problems.append("StrictHostKeyChecking must be yes")
    if first("passwordauthentication").lower() not in {"no", "false"}:
        problems.append("PasswordAuthentication must be no")
    if first("kbdinteractiveauthentication").lower() not in {"no", "false"}:
        problems.append("KbdInteractiveAuthentication must be no")
    if first("pubkeyauthentication").lower() not in {"yes", "true"}:
        problems.append("PubkeyAuthentication must be yes")
    preferred = {
        value.strip().lower()
        for value in first("preferredauthentications").split(",")
        if value.strip()
    }
    if preferred != {"publickey"}:
        problems.append("PreferredAuthentications must equal publickey")
    identities = values.get("identityfile", [])
    resolved_identities = {
        pathlib.Path(item).expanduser().resolve(strict=False)
        for item in identities
    }
    if resolved_identities != {EXPECTED_IDENTITY_FILE}:
        problems.append(
            f"IdentityFile set must equal {{{EXPECTED_IDENTITY_FILE}}}"
        )
    known_hosts = {
        pathlib.Path(item).expanduser().resolve(strict=False)
        for raw in values.get("userknownhostsfile", [])
        for item in shlex.split(raw)
        if item.lower() != "none"
    }
    if known_hosts != {EXPECTED_KNOWN_HOSTS_FILE}:
        problems.append(
            "UserKnownHostsFile set must equal "
            f"{{{EXPECTED_KNOWN_HOSTS_FILE}}}"
        )
    try:
        keepalive = int(first("serveraliveinterval"))
        keepalive_count = int(first("serveralivecountmax"))
    except ValueError:
        keepalive = keepalive_count = 0
        problems.append("keepalive settings must be integers")
    if keepalive <= 0 or keepalive_count <= 0:
        problems.append("keepalive settings must be positive")
    if problems:
        raise RemoteValidationError(
            f"invalid SSH alias {SSH_ALIAS!r}: " + "; ".join(problems)
        )
    return {
        "alias": SSH_ALIAS,
        "host": EXPECTED_HOST,
        "user": EXPECTED_USER,
        "identity_basename": EXPECTED_IDENTITY_BASENAME,
        "strict_host_key_checking": True,
        "expected_ed25519_fingerprint": EXPECTED_ED25519_FINGERPRINT,
        "server_alive_interval": keepalive,
        "server_alive_count_max": keepalive_count,
    }


def check_ssh_alias(runner: CommandRunner) -> dict[str, Any]:
    result = runner.run(("ssh", "-G", SSH_ALIAS))
    if runner.dry_run:
        return {"alias": SSH_ALIAS, "dry_run": True}
    report = validate_ssh_alias_config(result.stdout)
    try:
        known_hosts = validate_known_hosts_fingerprint(
            result.stdout, expected=EXPECTED_ED25519_FINGERPRINT,
            expected_host=EXPECTED_HOST,
            expected_path=EXPECTED_KNOWN_HOSTS_FILE,
        )
    except ValueError as error:
        raise RemoteValidationError(str(error)) from error
    report["known_hosts_file"] = str(known_hosts)
    return report


def _git(
    runner: CommandRunner,
    repo_root: pathlib.Path,
    *arguments: str,
) -> Completed:
    return runner.run(("git", *arguments), cwd=repo_root)


def _require_clean_repository(
    runner: CommandRunner,
    repo_root: pathlib.Path,
) -> None:
    result = _git(
        runner,
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if not runner.dry_run and result.stdout.strip():
        preview = "\n".join(result.stdout.splitlines()[:20])
        raise RemoteValidationError(
            "release deployment requires a clean repository; commit the "
            f"candidate first. Outstanding paths:\n{preview}"
        )


def _environment_lock() -> tuple[str, dict[str, Any]]:
    versions: dict[str, str] = {}
    for distribution in REQUIREMENTS_DISTRIBUTIONS:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as error:
            raise RemoteValidationError(
                f"required distribution is not installed: {distribution}"
            ) from error
    text = "\n".join(
        [
            "# Exact candidate-environment versions; do not edit.",
            *(f"{name}=={version}" for name, version in versions.items()),
            "",
        ]
    )
    manifest = {
        "schema": "negfpy.remote_validation.environment.v1",
        "created_utc": _utc_now(),
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "distributions": versions,
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
    }
    return text, manifest


def mpi_environment_identity() -> dict[str, Any]:
    """Return and validate the lock environment's MPI implementation.

    The launcher and the Python binding must come from the same environment.
    This rejects combining an unrelated launcher with an ABI-incompatible
    ``mpi4py`` build.
    """

    try:
        from mpi4py import MPI
    except ImportError as error:
        raise RemoteValidationError(
            "the remote validation environment requires mpi4py"
        ) from error

    vendor, version = MPI.get_vendor()
    normalized_version = tuple(int(item) for item in version)
    if vendor != EXPECTED_MPI_VENDOR or normalized_version != EXPECTED_MPI_VERSION:
        raise RemoteValidationError(
            "mpi4py MPI vendor mismatch: "
            f"{vendor} {normalized_version} != "
            f"{EXPECTED_MPI_VENDOR} {EXPECTED_MPI_VERSION}"
        )

    environment_bin = pathlib.Path(sys.executable).resolve().parent
    launcher = environment_bin / "mpiexec"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise RemoteValidationError(
            f"environment-local mpiexec is missing or not executable: {launcher}"
        )
    resolved_launcher = launcher.resolve(strict=True)
    if resolved_launcher.parent != environment_bin:
        raise RemoteValidationError(
            "environment-local mpiexec resolves outside the environment bin"
        )
    completed = subprocess.run(
        (str(launcher), "--version"),
        check=False,
        capture_output=True,
        text=True,
    )
    launcher_version = (completed.stdout or completed.stderr).strip()
    expected_text = ".".join(str(item) for item in EXPECTED_MPI_VERSION)
    if (
        completed.returncode
        or "Open MPI" not in launcher_version
        or expected_text not in launcher_version
    ):
        raise RemoteValidationError(
            "environment-local mpiexec does not report the pinned Open MPI "
            f"{expected_text}: {launcher_version!r}"
        )
    library_version = MPI.Get_library_version().rstrip("\x00\n")
    if "Open MPI" not in library_version or expected_text not in library_version:
        raise RemoteValidationError(
            "mpi4py library identity does not match the pinned launcher"
        )
    return {
        "vendor": vendor,
        "version": list(normalized_version),
        "library_version": library_version,
        "launcher": str(launcher),
        "launcher_target": str(resolved_launcher),
        "launcher_sha256": _sha256(resolved_launcher),
        "launcher_version": launcher_version,
    }


def _write_sha256sums(
    root: pathlib.Path,
    relative_paths: Iterable[pathlib.PurePath],
) -> pathlib.Path:
    entries: list[str] = []
    for relative in sorted(
        (pathlib.PurePath(item) for item in relative_paths),
        key=lambda item: item.as_posix(),
    ):
        if relative.is_absolute() or ".." in relative.parts:
            raise RemoteValidationError(f"unsafe checksum path: {relative}")
        path = root.joinpath(*relative.parts)
        if not path.is_file() or path.is_symlink():
            raise RemoteValidationError(f"not a regular checksum target: {path}")
        entries.append(f"{_sha256(path)}  {relative.as_posix()}")
    output = root / "SHA256SUMS"
    _atomic_write_text(output, "\n".join(entries) + "\n")
    return output


def verify_sha256sums(root: pathlib.Path) -> tuple[str, ...]:
    """Strictly verify SHA256SUMS without traversal or symlinks."""

    manifest = root / "SHA256SUMS"
    if not manifest.is_file() or manifest.is_symlink():
        raise RemoteValidationError(f"missing regular checksum manifest: {manifest}")
    verified: list[str] = []
    for line_number, raw_line in enumerate(
        manifest.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", raw_line)
        if match is None:
            raise RemoteValidationError(
                f"invalid checksum entry at {manifest}:{line_number}"
            )
        expected, relative_text = match.groups()
        relative = pathlib.PurePosixPath(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative.name == "SHA256SUMS":
            raise RemoteValidationError(f"unsafe checksum path: {relative_text}")
        target = root.joinpath(*relative.parts)
        if not target.is_file() or target.is_symlink():
            raise RemoteValidationError(f"not a regular checksum target: {target}")
        actual = _sha256(target)
        if not hmac.compare_digest(actual, expected):
            raise RemoteValidationError(f"SHA256 mismatch for {relative_text}")
        verified.append(relative_text)
    if not verified:
        raise RemoteValidationError(f"empty checksum manifest: {manifest}")
    return tuple(verified)
