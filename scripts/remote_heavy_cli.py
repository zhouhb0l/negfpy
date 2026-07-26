#!/usr/bin/env python3
"""Command-line entry point for NEGFPy's key-only remote validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import pathlib
import platform
import shlex
import shutil
import socket
import stat
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import remote_heavy_runtime as operations
import remote_heavy_identity
import remote_heavy_explicit as explicit
import remote_heavy_validation as core
import remote_mpi_scaling
import remote_toy_mpi_runner


ORCHESTRATION_SCRIPT_PATHS = (
    "scripts/remote_heavy_cli.py",
    "scripts/remote_heavy_explicit.py",
    "scripts/remote_heavy_identity.py",
    "scripts/remote_heavy_lock.py",
    "scripts/remote_heavy_runtime.py",
    "scripts/remote_heavy_security.py",
    "scripts/remote_heavy_validation.py",
    "scripts/remote_mpi_scaling.py",
    "scripts/remote_toy_mpi_runner.py",
)


def _remote_runtime_path(remote: str) -> str:
    return ":".join(
        (
            core.REMOTE_TOOLS_BIN,
            f"{remote}/environment/bin",
            "/usr/bin",
            "/bin",
        )
    )


def _remote_bootstrap_command(commit: str) -> str:
    remote = core._remote_shell_dir(commit)
    runtime_path = _remote_runtime_path(remote)
    return " && ".join(
        (
            f"mkdir -p {remote}/incoming {remote}/artifacts",
            f"cd {remote}/incoming",
            "sha256sum -c SHA256SUMS",
            f"test \"$(sha256sum environment-explicit.lock | cut -d' ' -f1)\" = "
            f"{core.EXPECTED_ENV_LOCK_SHA256}",
            f"test ! -e {remote}/worktree",
            f"{core.REMOTE_GIT} clone --no-checkout negfpy.bundle {remote}/worktree",
            f"{core.REMOTE_GIT} -C {remote}/worktree checkout --detach {commit}",
            f'test "$({core.REMOTE_GIT} -C {remote}/worktree rev-parse HEAD)" = {commit}',
            f'test -z "$({core.REMOTE_GIT} -C {remote}/worktree status --porcelain)"',
            f"test ! -e {remote}/environment",
            f"{core.REMOTE_MICROMAMBA} create -y -p {remote}/environment "
            f"--file {remote}/incoming/environment-explicit.lock",
            f"{remote}/environment/bin/python -m pip install --no-deps "
            f"--no-build-isolation -e {remote}/worktree",
            f"env PYTHONDONTWRITEBYTECODE=1 PATH={runtime_path} "
            f"PYTHONPATH={remote}/worktree/src {remote}/environment/bin/python "
            f"{remote}/worktree/scripts/remote_heavy_cli.py "
            f"_record-remote-environment --deployment-dir {remote} --commit {commit}",
        )
    )


def deploy(
    *,
    staging: pathlib.Path,
    runner: core.CommandRunner,
) -> str:
    manifest = None
    if not runner.dry_run:
        core.verify_sha256sums(staging)
        manifest = json.loads(
            (staging / "deployment-manifest.json").read_text(encoding="utf-8")
        )
        if manifest.get("schema") != "negfpy.remote_validation.deployment.v2":
            raise core.RemoteValidationError("unsupported deployment manifest schema")
        if manifest["environment"]["explicit_lock_sha256"] != core.EXPECTED_ENV_LOCK_SHA256:
            raise core.RemoteValidationError("deployment explicit-lock hash mismatch")
    commit = (
        core._validate_commit(manifest["source"]["commit"])
        if manifest is not None
        else core._validate_commit(staging.name)
    )
    core.check_ssh_alias(runner)
    remote = core._remote_relative_dir(commit)
    runner.run(core.ssh_argv(f'mkdir -p "$HOME/{remote}/incoming"'))
    files = (
        "SHA256SUMS",
        "deployment-manifest.json",
        "environment-explicit.lock",
        "environment-manifest.json",
        "negfpy.bundle",
    )
    runner.run(
        core.scp_argv(
            *(str(staging.resolve() / name) for name in files),
            f"{core.SSH_ALIAS}:{remote}/incoming/",
        )
    )
    runner.run(core.ssh_argv(_remote_bootstrap_command(commit)))
    return commit


def _launch_command(commit: str, session: str) -> str:
    remote = core._remote_shell_dir(commit)
    runtime_path = _remote_runtime_path(remote)
    session = core._validate_session(session)
    limits = " ".join(
        f"{key}=1" for key in sorted(core.THREAD_LIMIT_ENV)
    )
    worker = (
        f"env PYTHONDONTWRITEBYTECODE=1 PATH={runtime_path} "
        f"PYTHONPATH={remote}/worktree/src {limits} "
        f"{remote}/environment/bin/python "
        f"{remote}/worktree/scripts/remote_heavy_cli.py _worker "
        f"--deployment-dir {remote} --commit {commit}"
    )
    lock = f"{remote}/artifacts/job.lock"
    stale = f"{remote}/artifacts/attempts/stale-lock-$(date +%s)-$$"
    return (
        f"test -x {remote}/environment/bin/python && "
        f"if test -d {lock}; then old=$(cat {lock}/session 2>/dev/null || true); "
        f"if test -n \"$old\" && {core.REMOTE_TMUX} has-session -t \"$old\" 2>/dev/null; then exit 73; fi; "
        f"mkdir -p {remote}/artifacts/attempts; mv {lock} {stale} || exit 74; fi && "
        f"mkdir {lock} && printf '%s\\n' {session} > {lock}/session && "
        f"{core.REMOTE_TMUX} new-session -d -s {session} "
        + shlex.quote("exec " + worker)
    )


def launch(
    *,
    commit: str,
    session: str | None,
    runner: core.CommandRunner,
) -> str:
    commit = core._validate_commit(commit)
    core.check_ssh_alias(runner)
    selected = core._validate_session(
        session or operations._default_session(commit)
    )
    runner.run(core.ssh_argv(_launch_command(commit, selected)))
    print(
        json.dumps(
            {
                "commit": commit,
                "session": selected,
                "state": "launch-requested" if runner.dry_run else "launched",
            },
            sort_keys=True,
        )
    )
    return selected


def _read_environment_lock(deployment_dir: pathlib.Path) -> dict[str, Any]:
    lock = deployment_dir / "incoming" / "environment-manifest.json"
    try:
        value = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise core.RemoteValidationError(f"invalid environment lock: {lock}") from error
    if value.get("schema") != "negfpy.remote_validation.environment.v2":
        raise core.RemoteValidationError("unsupported environment manifest schema")
    return value


def _record_remote_environment(
    *,
    deployment_dir: pathlib.Path,
    commit: str,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Verify exact source/environment identity and archive the machine."""

    root = deployment_dir.resolve()
    commit = core._validate_commit(commit)
    worktree = root / "worktree"
    actual_commit = subprocess.run(
        (core.REMOTE_GIT, "-C", str(worktree), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        (core.REMOTE_GIT, "-C", str(worktree), "status", "--porcelain"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_commit != commit or dirty:
        raise core.RemoteValidationError(
            "remote source is not the requested clean detached commit"
        )
    locked = _read_environment_lock(root)
    lock_sha = core._sha256(root / "incoming" / "environment-explicit.lock")
    if lock_sha != core.EXPECTED_ENV_LOCK_SHA256:
        raise core.RemoteValidationError("remote explicit-lock SHA256 mismatch")
    if locked["explicit_lock_sha256"] != lock_sha:
        raise core.RemoteValidationError("environment manifest/lock mismatch")
    versions = {
        name: importlib.metadata.version(name)
        for name in core.REQUIREMENTS_DISTRIBUTIONS
    }
    versions["negfpy"] = importlib.metadata.version("negfpy")
    import negfpy
    from negfpy.anharmonic._provenance import code_provenance

    package_source = str(pathlib.Path(negfpy.__file__).resolve())
    expected_source = (worktree / "src" / "negfpy").resolve()
    if not pathlib.Path(package_source).is_relative_to(expected_source):
        raise core.RemoteValidationError(
            f"NEGFPy source is outside detached worktree: {package_source}"
        )
    disk = shutil.disk_usage(root)
    tree = subprocess.run(
        (core.REMOTE_GIT, "-C", str(worktree), "rev-parse", "HEAD^{tree}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    package_provenance = code_provenance()
    if (
        package_provenance.get("git_revision") != commit
        or package_provenance.get("git_dirty") is not False
    ):
        raise core.RemoteValidationError(
            "deployed NEGFPy runtime provenance is not the requested clean commit"
        )
    source_tree_sha256 = package_provenance.get("source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
    ):
        raise core.RemoteValidationError(
            "could not compute the deployed NEGFPy source-tree SHA256"
        )
    execution_source_identity = _execution_source_identity(
        worktree=worktree,
        commit=commit,
        git_tree=tree,
        package_source_sha256=source_tree_sha256,
    )
    environment_prefix_manifest = _environment_prefix_manifest(
        root / "environment"
    )
    cpu_model = platform.processor()
    cpuinfo = pathlib.Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    memory_total = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    memory_available = None
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.is_file():
        memory_values = dict(
            line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line
        )
        available_kib = memory_values.get("MemAvailable")
        if available_kib:
            memory_available = int(available_kib.split()[0]) * 1024
    manifest = {
            "schema": "negfpy.remote_validation.remote_environment.v1",
            "created_utc": core._utc_now(),
            "commit": commit,
            "source_tree_sha256": source_tree_sha256,
            "execution_source_identity": execution_source_identity,
            "environment_prefix_manifest": environment_prefix_manifest,
            "explicit_lock_sha256": lock_sha,
            "tree": tree,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **core.machine_cpu_topology(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model,
            "memory_total_bytes": memory_total,
            "package_source": package_source,
            "memory_available_bytes": memory_available,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "distributions": versions,
            "mpi": core.mpi_environment_identity(),
            "deployment_disk_total_bytes": disk.total,
            "deployment_disk_free_bytes": disk.free,
            "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
    }
    if write_manifest:
        core._atomic_write_json(
            root / "artifacts" / "identity" / "remote-environment.json",
            manifest,
        )

    return manifest


def _strict_json(path: pathlib.Path) -> dict[str, Any]:
    """Load strict JSON, rejecting duplicate keys and non-finite tokens."""

    if not path.is_file() or path.is_symlink():
        raise core.RemoteValidationError(
            f"JSON evidence is not a regular file: {path}"
        )

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise core.RemoteValidationError(
                    f"duplicate JSON key {key!r} in {path}"
                )
            result[key] = value
        return result

    def invalid_constant(value: str) -> None:
        raise core.RemoteValidationError(
            f"non-finite JSON token {value!r} in {path}"
        )

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs,
            parse_constant=invalid_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise core.RemoteValidationError(f"invalid JSON document: {path}") from error
    if not isinstance(value, dict):
        raise core.RemoteValidationError(f"JSON root must be an object: {path}")
    return value


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise core.RemoteValidationError(f"{name} must be an object")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise core.RemoteValidationError(message)


def _is_sha256(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_digest(
    document: Mapping[str, Any],
    *,
    excluded_key: str,
    ensure_ascii: bool,
) -> str:
    body = {
        key: value for key, value in document.items() if key != excluded_key
    }
    payload = json.dumps(
        body,
        allow_nan=False,
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _canonical_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(document)
    body["manifest_sha256"] = _canonical_digest(
        body,
        excluded_key="manifest_sha256",
        ensure_ascii=True,
    )
    return body


def _absolute_directory_without_symlinks(
    path: pathlib.Path,
    *,
    name: str,
) -> pathlib.Path:
    absolute = pathlib.Path(os.path.abspath(os.fspath(path.expanduser())))
    current = pathlib.Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise core.RemoteValidationError(f"{name} traverses a symlink: {current}")
    if not absolute.is_dir():
        raise core.RemoteValidationError(f"{name} is not a directory: {absolute}")
    return absolute


def _environment_prefix_manifest(prefix: pathlib.Path) -> dict[str, Any]:
    """Hash every regular file and safely describe every prefix symlink."""

    root = _absolute_directory_without_symlinks(
        prefix,
        name="executed environment prefix",
    )
    regular_files: list[dict[str, Any]] = []
    symlinks: list[dict[str, Any]] = []
    directory_count = 0
    pending = [root]
    try:
        while pending:
            directory = pending.pop()
            directory_count += 1
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda item: item.name)
            for entry in entries:
                candidate = pathlib.Path(entry.path)
                relative = candidate.relative_to(root).as_posix()
                metadata = entry.stat(follow_symlinks=False)
                mode = metadata.st_mode
                if stat.S_ISLNK(mode):
                    target_text = os.readlink(candidate)
                    if pathlib.PurePath(target_text).is_absolute():
                        raise core.RemoteValidationError(
                            "environment prefix contains an absolute symlink: "
                            f"{relative} -> {target_text}"
                        )
                    resolved = candidate.resolve(strict=True)
                    if not resolved.is_relative_to(root):
                        raise core.RemoteValidationError(
                            "environment prefix symlink escapes its prefix: "
                            f"{relative} -> {target_text}"
                        )
                    if resolved.is_file():
                        target_kind = "regular_file"
                    elif resolved.is_dir():
                        target_kind = "directory"
                    else:
                        raise core.RemoteValidationError(
                            f"environment symlink has a special target: {relative}"
                        )
                    symlinks.append(
                        {
                            "path": relative,
                            "target": target_text,
                            "resolved_path": resolved.relative_to(root).as_posix(),
                            "target_kind": target_kind,
                            "link_size": int(metadata.st_size),
                        }
                    )
                elif stat.S_ISDIR(mode):
                    pending.append(candidate)
                elif stat.S_ISREG(mode):
                    regular_files.append(
                        {
                            "path": relative,
                            "size": int(metadata.st_size),
                            "sha256": core._sha256(candidate),
                        }
                    )
                else:
                    raise core.RemoteValidationError(
                        f"environment prefix contains a special file: {relative}"
                    )
    except (OSError, RuntimeError) as error:
        if isinstance(error, core.RemoteValidationError):
            raise
        raise core.RemoteValidationError(
            f"cannot inventory executed environment prefix {root}: {error}"
        ) from error
    regular_files.sort(key=lambda item: item["path"])
    symlinks.sort(key=lambda item: item["path"])
    return _canonical_manifest(
        {
            "schema": "negfpy.remote_validation.environment_prefix.v1",
            "prefix": str(root),
            "directory_count": directory_count,
            "regular_file_count": len(regular_files),
            "symlink_count": len(symlinks),
            "regular_files": regular_files,
            "symlinks": symlinks,
        }
    )


def _execution_source_identity(
    *,
    worktree: pathlib.Path,
    commit: str,
    git_tree: str,
    package_source_sha256: str,
) -> dict[str, Any]:
    """Bind the clean worktree and every imported orchestration script byte."""

    root = _absolute_directory_without_symlinks(worktree, name="worktree")
    revision = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "rev-parse", "HEAD^{tree}"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != commit or tree != git_tree or dirty:
        raise core.RemoteValidationError(
            "executed worktree is not the requested clean Git tree"
        )
    listed = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "ls-files", "-z"),
        check=True,
        capture_output=True,
    ).stdout
    tracked_paths = {
        pathlib.PurePosixPath(raw.decode("utf-8"))
        for raw in listed.split(b"\0")
        if raw
    }
    untracked = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard"),
        check=True,
        capture_output=True,
    ).stdout
    ignored = subprocess.run(
        (core.REMOTE_GIT, "-C", str(root), "ls-files", "-z", "--others", "--ignored", "--exclude-standard"),
        check=True,
        capture_output=True,
    ).stdout
    extra_paths = {
        pathlib.PurePosixPath(raw.decode("utf-8"))
        for raw in untracked.split(b"\0") + ignored.split(b"\0")
        if raw
    }
    worktree_paths = sorted(tracked_paths | extra_paths, key=lambda value: value.as_posix())
    records: list[dict[str, Any]] = []
    for relative in worktree_paths:
        if relative.is_absolute() or ".." in relative.parts:
            raise core.RemoteValidationError(f"unsafe worktree source path: {relative}")
        candidate = root.joinpath(*relative.parts)
        metadata = candidate.lstat()
        if stat.S_ISREG(metadata.st_mode):
            record = {
                "path": relative.as_posix(),
                "git_tracked": relative in tracked_paths,
                "kind": "regular_file",
                "size": int(metadata.st_size),
                "mode": stat.S_IMODE(metadata.st_mode),
                "sha256": core._sha256(candidate),
            }
        elif stat.S_ISLNK(metadata.st_mode):
            target = os.readlink(candidate)
            resolved = candidate.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise core.RemoteValidationError(
                    f"tracked source symlink escapes worktree: {relative}"
                )
            record = {
                "path": relative.as_posix(),
                "git_tracked": relative in tracked_paths,
                "kind": "symlink",
                "target": target,
                "mode": stat.S_IMODE(metadata.st_mode),
            }
        else:
            raise core.RemoteValidationError(
                f"tracked source is not a regular file or symlink: {relative}"
            )
        records.append(record)
    records.sort(key=lambda item: item["path"])
    by_path = {str(item["path"]): item for item in records}
    observed_orchestration = {
        path
        for path in by_path
        if path.startswith("scripts/remote_") and path.endswith(".py")
    }
    expected_orchestration = set(ORCHESTRATION_SCRIPT_PATHS)
    if observed_orchestration != expected_orchestration:
        raise core.RemoteValidationError(
            "orchestration source set differs from the locked imported scripts: "
            f"{sorted(observed_orchestration ^ expected_orchestration)}"
        )
    orchestration = [dict(by_path[path]) for path in ORCHESTRATION_SCRIPT_PATHS]
    if any(item.get("kind") != "regular_file" or item.get("git_tracked") is not True for item in orchestration):
        raise core.RemoteValidationError("orchestration scripts must be regular files")
    tracked_records = [item for item in records if item["git_tracked"]]
    tracked_manifest = _canonical_manifest(
        {
            "schema": "negfpy.remote_validation.tracked_worktree.v1",
            "records": tracked_records,
        }
    )
    worktree_manifest = _canonical_manifest(
        {
            "schema": "negfpy.remote_validation.full_worktree.v1",
            "records": records,
        }
    )
    orchestration_manifest = _canonical_manifest(
        {
            "schema": "negfpy.remote_validation.orchestration_sources.v1",
            "records": orchestration,
        }
    )
    return _canonical_manifest(
        {
            "schema": "negfpy.remote_validation.execution_source.v1",
            "git_revision": revision,
            "git_tree": tree,
            "package_source_sha256": package_source_sha256,
            "tracked_file_count": len(tracked_records),
            "tracked_files_manifest_sha256": tracked_manifest["manifest_sha256"],
            "tracked_files": tracked_records,
            "worktree_file_count": len(records),
            "worktree_files_manifest_sha256": worktree_manifest["manifest_sha256"],
            "worktree_files": records,
            "orchestration_manifest_sha256": orchestration_manifest["manifest_sha256"],
            "orchestration_scripts": orchestration,
        }
    )


def _scientific_launch_environment(
    *,
    root: pathlib.Path,
    attempt_root: pathlib.Path,
    environment_prefix: pathlib.Path,
) -> dict[str, str]:
    """Return a scrubbed, receipt-bound environment for scientific children."""

    cache = attempt_root / "runtime-cache"
    environment = {
        "HOME": f"/home/{core.EXPECTED_USER}",
        "USER": core.EXPECTED_USER,
        "LOGNAME": core.EXPECTED_USER,
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PATH": os.pathsep.join((str(environment_prefix / "bin"), core.REMOTE_TOOLS_BIN, "/usr/bin", "/bin")),
        "PYTHONPATH": str(root / "worktree" / "src"),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": str(cache / "pycache"),
        "TMPDIR": str(attempt_root / "runtime-tmp"),
        "XDG_CACHE_HOME": str(cache),
        "MPLCONFIGDIR": str(attempt_root / "runtime-config" / "matplotlib"),
        **core.THREAD_LIMIT_ENV,
    }
    return dict(sorted(environment.items()))


def _launch_environment_sha256(environment: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(
            dict(sorted(environment.items())),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _validate_code_identity(
    identity: Any,
    *,
    commit: str,
    source_tree_sha256: str,
    name: str,
) -> None:
    record = _mapping(identity, name=name)
    _require(
        record.get("git_revision") == commit,
        f"{name} does not match deployed commit {commit}",
    )
    _require(
        record.get("source_tree_sha256") == source_tree_sha256,
        f"{name} does not match deployed source-tree SHA256",
    )


def _archive_identity_bundle(
    *,
    root: pathlib.Path,
    commit: str,
    stored_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify deployment inputs and copy immutable identity evidence."""

    incoming = root / "incoming"
    expected_inputs = {
        "deployment-manifest.json",
        "environment-explicit.lock",
        "environment-manifest.json",
        "negfpy.bundle",
    }
    verified = set(core.verify_sha256sums(incoming))
    _require(
        verified == expected_inputs,
        "incoming deployment SHA256SUMS does not cover the exact input set",
    )
    deployment = _strict_json(incoming / "deployment-manifest.json")
    environment = _strict_json(incoming / "environment-manifest.json")
    _require(
        deployment.get("schema") == "negfpy.remote_validation.deployment.v2",
        "unsupported archived deployment manifest schema",
    )
    _require(
        environment.get("schema") == "negfpy.remote_validation.environment.v2",
        "unsupported archived environment manifest schema",
    )
    source = _mapping(deployment.get("source"), name="deployment.source")
    deployed_environment = _mapping(
        deployment.get("environment"), name="deployment.environment"
    )
    tree = source.get("tree")
    lock_sha = stored_identity.get("explicit_lock_sha256")
    _require(source.get("commit") == commit, "deployment commit mismatch")
    _require(source.get("dirty") is False, "deployment source is dirty")
    _require(
        isinstance(tree, str) and len(tree) == 40,
        "deployment Git tree identity is invalid",
    )
    _require(_is_sha256(lock_sha), "remote environment lock SHA256 is invalid")
    _require(
        deployed_environment.get("explicit_lock_sha256") == lock_sha
        and environment.get("explicit_lock_sha256") == lock_sha
        and core._sha256(incoming / "environment-explicit.lock") == lock_sha,
        "deployment/environment explicit-lock identities disagree",
    )
    _require(stored_identity.get("tree") == tree, "remote Git tree mismatch")
    _require(
        stored_identity.get("commit") == commit,
        "remote environment commit mismatch",
    )
    source_sha = stored_identity.get("source_tree_sha256")
    _require(_is_sha256(source_sha), "remote source-tree SHA256 is invalid")

    identity_root = root / "artifacts" / "identity"
    identity_root.mkdir(parents=True, exist_ok=True)
    copies = {
        "deployment-manifest.json": incoming / "deployment-manifest.json",
        "environment-manifest.json": incoming / "environment-manifest.json",
        "environment-explicit.lock": incoming / "environment-explicit.lock",
        "incoming-SHA256SUMS": incoming / "SHA256SUMS",
    }
    for destination_name, source_path in copies.items():
        destination = identity_root / destination_name
        if destination.exists() and core._sha256(destination) != core._sha256(
            source_path
        ):
            raise core.RemoteValidationError(
                f"immutable artifact identity drift: {destination}"
            )
        core._atomic_copy_file(source_path, destination)

    lock_record = {
        "schema": "negfpy.remote_validation.lock_digest.v1",
        "commit": commit,
        "git_tree": tree,
        "source_tree_sha256": source_sha,
        "explicit_lock": "environment-explicit.lock",
        "explicit_lock_sha256": lock_sha,
    }
    core._atomic_write_json(identity_root / "lock-digest.json", lock_record)
    bundle = {
        **lock_record,
        "schema": "negfpy.remote_validation.identity_bundle.v1",
        "remote_environment": "remote-environment.json",
        "remote_environment_sha256": core._sha256(
            identity_root / "remote-environment.json"
        ),
        "verified_incoming_files": sorted(verified),
        "archived_files": {
            name: core._sha256(identity_root / name) for name in sorted(copies)
        },
    }
    core._atomic_write_json(identity_root / "identity-manifest.json", bundle)
    return bundle


def _strict_envelope(path: pathlib.Path, *, name: str) -> dict[str, Any]:
    """Load a canonical payload envelope and verify its semantic SHA256."""

    _require(
        path.is_file() and not path.is_symlink(),
        f"{name} is not a regular file: {path}",
    )
    document = _strict_json(path)
    _require(
        set(document) == {"payload", "payload_sha256"},
        f"{name} has the wrong payload envelope",
    )
    payload = _mapping(document.get("payload"), name=f"{name}.payload")
    digest = document.get("payload_sha256")
    canonical_payload = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    _require(
        _is_sha256(digest)
        and hashlib.sha256(canonical_payload).hexdigest() == digest,
        f"{name} payload SHA256 is invalid",
    )
    canonical_document = (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")
    _require(
        path.read_bytes() == canonical_document,
        f"{name} is not canonical strict JSON",
    )
    return document


def _validate_scaling_selection(
    *,
    selection_path: pathlib.Path,
    source_revision: str,
    environment_prefix: pathlib.Path,
    source_tree_sha256: str,
    environment_sha256: str,
    physical_core_count: int,
    socket_count: int,
) -> dict[str, Any]:
    """Rebuild and validate the immutable five-rank scaling decision."""

    try:
        selection = remote_toy_mpi_runner.read_hashed_manifest(selection_path)
    except Exception as error:
        raise core.RemoteValidationError(
            f"invalid immutable MPI scaling selection: {error}"
        ) from error
    _require(
        selection_path.is_file() and not selection_path.is_symlink(),
        "MPI scaling selection is not a regular file",
    )
    scaling_root = selection_path.parent.resolve()
    prefix = environment_prefix.resolve()
    expected_launcher = (prefix / "bin" / "mpiexec").resolve()
    expected_python = (prefix / "bin" / "python").resolve()
    rank_manifests: list[tuple[pathlib.Path, Mapping[str, Any]]] = []
    for rank_count in remote_mpi_scaling.RANK_LADDER:
        rank_path = scaling_root / f"rank-{rank_count:03d}-manifest.json"
        _require(
            rank_path.is_file() and not rank_path.is_symlink(),
            f"rank-{rank_count} scaling manifest is missing or unsafe",
        )
        try:
            rank_document = remote_toy_mpi_runner.read_hashed_manifest(rank_path)
        except Exception as error:
            raise core.RemoteValidationError(
                f"invalid rank-{rank_count} scaling manifest: {error}"
            ) from error
        rank_payload = _mapping(
            rank_document.get("payload"),
            name=f"rank-{rank_count} scaling payload",
        )
        launcher = _mapping(
            rank_payload.get("launcher"),
            name=f"rank-{rank_count} MPI launcher",
        )
        python = _mapping(
            rank_payload.get("python"),
            name=f"rank-{rank_count} Python",
        )
        argv = rank_payload.get("argv")
        _require(
            pathlib.Path(str(launcher.get("path"))).resolve()
            == expected_launcher
            and pathlib.Path(str(python.get("path"))).resolve()
            == expected_python
            and isinstance(argv, list)
            and bool(argv)
            and pathlib.Path(str(argv[0])).resolve() == expected_launcher
            and str(expected_python) in argv,
            f"rank-{rank_count} did not use the deployed MPI/Python environment",
        )
        rank_manifests.append((rank_path, rank_document))
    try:
        rebuilt = remote_toy_mpi_runner.build_scaling_selection_payload(
            rank_manifests,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
        )
    except Exception as error:
        raise core.RemoteValidationError(
            f"MPI scaling evidence failed revalidation: {error}"
        ) from error
    payload = _mapping(selection.get("payload"), name="scaling selection payload")
    supplied_without_time = {
        key: value for key, value in payload.items() if key != "created_utc"
    }
    rebuilt_without_time = {
        key: value for key, value in rebuilt.items() if key != "created_utc"
    }
    _require(
        supplied_without_time == rebuilt_without_time,
        "immutable MPI scaling selection differs from rebuilt rank evidence",
    )
    selection_record = _mapping(payload.get("selection"), name="scaling selection")
    toy = _mapping(payload.get("toy_execution"), name="toy rank selection")
    silicon = _mapping(
        payload.get("silicon_execution"), name="silicon rank selection"
    )
    _require(
        payload.get("schema")
        == "negfpy.remote_toy_mpi.scaling_selection.v1"
        and payload.get("source_revision") == source_revision
        and payload.get("source_tree_sha256") == source_tree_sha256
        and payload.get("environment_sha256") == environment_sha256
        and payload.get("physics_equal_across_all_ranks") is True
        and payload.get("fresh_isolated_repetitions") is True,
        "MPI scaling selection identity or equality gate failed",
    )
    selected_rank = toy.get("rank_count")
    toy_mpi_enabled = selection_record.get("toy_mpi_enabled")
    silicon_rank = selection_record.get("silicon_rank_count")
    _require(
        isinstance(selected_rank, int)
        and not isinstance(selected_rank, bool)
        and selected_rank in remote_mpi_scaling.RANK_LADDER
        and selected_rank == selection_record.get("toy_rank_count"),
        "MPI scaling selection contains an invalid toy rank",
    )
    _require(
        isinstance(toy_mpi_enabled, bool)
        and (selected_rank > 1) is toy_mpi_enabled
        and toy.get("mode")
        == ("task_mpi" if toy_mpi_enabled else "serial_fft")
        and toy.get("minimum_speedup_satisfied") is toy_mpi_enabled
        and isinstance(silicon_rank, int)
        and not isinstance(silicon_rank, bool)
        and silicon_rank in remote_mpi_scaling.RANK_LADDER
        and silicon.get("rank_count") == silicon_rank
        and silicon.get("mode") == "task_mpi",
        "MPI scaling execution policy is inconsistent",
    )
    raw_rank_evidence = payload.get("rank_evidence")
    _require(
        isinstance(raw_rank_evidence, list)
        and len(raw_rank_evidence) == len(remote_mpi_scaling.RANK_LADDER),
        "MPI scaling rank evidence is incomplete",
    )
    rank_evidence: list[dict[str, Any]] = []
    for expected_rank, raw_record in zip(
        remote_mpi_scaling.RANK_LADDER,
        raw_rank_evidence,
        strict=True,
    ):
        record = _mapping(
            raw_record, name=f"rank-{expected_rank} scaling evidence"
        )
        launcher_memory = _mapping(
            record.get("launcher_memory_preflight"),
            name=f"rank-{expected_rank} launcher memory preflight",
        )
        worker_memory = _mapping(
            record.get("worker_memory_preflight"),
            name=f"rank-{expected_rank} worker memory preflight",
        )
        harmonic = _mapping(
            record.get("harmonic_preflight"),
            name=f"rank-{expected_rank} harmonic preflight",
        )
        runtime = record.get("rank_runtime")
        required_hashes = (
            "manifest_file_sha256",
            "manifest_payload_sha256",
            "worker_result_file_sha256",
            "campaign_manifest_file_sha256",
            "execution_manifest_file_sha256",
            "strict_report_file_sha256",
        )
        _require(
            record.get("rank_count") == expected_rank
            and all(_is_sha256(record.get(key)) for key in required_hashes)
            and launcher_memory.get("passed") is True
            and launcher_memory.get("rank_count") == expected_rank
            and worker_memory.get("passed") is True
            and worker_memory.get("rank_count") == expected_rank
            and harmonic.get("passed") is True
            and harmonic.get("real_physics_executed") is True
            and isinstance(runtime, list)
            and len(runtime) == expected_rank,
            f"rank-{expected_rank} scaling runtime evidence is incomplete",
        )
        rank_evidence.append(
            {
                "rank_count": expected_rank,
                **{key: record[key] for key in required_hashes},
                "launcher_memory_preflight_passed": True,
                "worker_memory_preflight_passed": True,
                "harmonic_preflight_passed": True,
                "rank_runtime_count": len(runtime),
            }
        )
    return {
        "schema": "negfpy.remote_validation.scaling_facts.v1",
        "selection_relative_path": selection_path.name,
        "selection_file_sha256": core._sha256(selection_path),
        "selection_payload_sha256": selection["payload_sha256"],
        "production_toy_physics_sha256": payload.get(
            "production_toy_physics_sha256"
        ),
        "physics_equal_across_all_ranks": True,
        "toy_rank_count": selected_rank,
        "toy_mpi_enabled": toy_mpi_enabled,
        "silicon_rank_count": silicon_rank,
        "candidate_speedup": selection_record.get("candidate_speedup"),
        "rank_ladder": list(remote_mpi_scaling.RANK_LADDER),
        "rank_evidence": rank_evidence,
    }



def _prior_scaling_selection(
    *,
    attempts: pathlib.Path,
    current_attempt_root: pathlib.Path,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    environment_prefix: pathlib.Path,
    physical_core_count: int,
    socket_count: int,
) -> tuple[pathlib.Path, dict[str, Any]] | None:
    """Reuse only a completely revalidated prior ladder on exact retries."""

    candidates = sorted(
        (
            path
            for path in attempts.glob("attempt-[0-9][0-9][0-9]")
            if path != current_attempt_root
        ),
        reverse=True,
    )
    for attempt_path in candidates:
        selection_path = attempt_path / "mpi-scaling" / "scaling-selection.json"
        if not selection_path.exists():
            continue
        facts = _validate_scaling_selection(
            selection_path=selection_path,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            environment_prefix=environment_prefix,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
        )
        return selection_path, {
            **facts,
            "reused_for_exact_retry": True,
            "source_attempt": attempt_path.name,
        }
    return None

def _recorded_artifact_path(
    *,
    root: pathlib.Path,
    record: Any,
    name: str,
) -> pathlib.Path:
    evidence = _mapping(record, name=name)
    relative_text = evidence.get("relative_path")
    _require(
        isinstance(relative_text, str) and bool(relative_text),
        f"{name} lacks a relative path",
    )
    relative = pathlib.PurePosixPath(relative_text)
    _require(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.as_posix() == relative_text,
        f"{name} path is unsafe",
    )
    target = root.joinpath(*relative.parts)
    _require(
        target.is_file() and not target.is_symlink(),
        f"{name} is missing or not a regular file",
    )
    resolved_root = root.resolve()
    resolved = target.resolve(strict=True)
    _require(resolved.is_relative_to(resolved_root), f"{name} escaped its run root")
    expected_sha = evidence.get("sha256")
    _require(
        _is_sha256(expected_sha) and core._sha256(resolved) == expected_sha,
        f"{name} file SHA256 is invalid",
    )
    return resolved


def _executable_identity(path: pathlib.Path) -> dict[str, str]:
    target = path.resolve(strict=True)
    _require(
        path.is_file() and os.access(path, os.X_OK),
        f"executable is unavailable: {path}",
    )
    return {
        "path": str(path),
        "resolved_path": str(target),
        "sha256": core._sha256(path),
    }


def _attempt_artifact_record(
    root: pathlib.Path,
    path: pathlib.Path,
) -> dict[str, str]:
    resolved_root = root.resolve()
    resolved = path.resolve(strict=True)
    _require(
        resolved.is_relative_to(resolved_root)
        and resolved.is_file()
        and not path.is_symlink(),
        f"release launch artifact is unsafe: {path}",
    )
    return {
        "relative_path": resolved.relative_to(resolved_root).as_posix(),
        "sha256": core._sha256(resolved),
    }


def _regular_file_identity(path: pathlib.Path, *, name: str) -> dict[str, Any]:
    try:
        metadata = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise core.RemoteValidationError(f"cannot identify {name}: {path}") from error
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        f"{name} is not a non-symlink regular file: {path}",
    )
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "size": int(metadata.st_size),
        "sha256": core._sha256(path),
    }


def _verified_manifest_digest(value: Any, *, name: str) -> str:
    manifest = _mapping(value, name=name)
    digest = manifest.get("manifest_sha256")
    _require(
        _is_sha256(digest)
        and _canonical_digest(
            manifest,
            excluded_key="manifest_sha256",
            ensure_ascii=True,
        )
        == digest,
        f"{name} canonical SHA256 is invalid",
    )
    return str(digest)


def _release_launch_binding(
    *,
    environment_prefix: pathlib.Path,
    mpi_runner: pathlib.Path,
    argv: Sequence[str],
    binding_arguments: Sequence[str],
    selected_rank: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    execution_id: str,
    execution_source_identity: Mapping[str, Any],
    environment_prefix_manifest: Mapping[str, Any],
    launch_environment: Mapping[str, str],
) -> dict[str, Any]:
    source_manifest_sha256 = _verified_manifest_digest(
        execution_source_identity,
        name="execution source identity",
    )
    prefix_manifest_sha256 = _verified_manifest_digest(
        environment_prefix_manifest,
        name="environment prefix manifest",
    )
    orchestration = execution_source_identity.get("orchestration_scripts")
    _require(
        isinstance(orchestration, list)
        and [item.get("path") for item in orchestration if isinstance(item, Mapping)]
        == list(ORCHESTRATION_SCRIPT_PATHS),
        "execution source identity lacks the exact orchestration source set",
    )
    runner = _regular_file_identity(mpi_runner, name="MPI release runner")
    runner_records = [
        item
        for item in orchestration
        if isinstance(item, Mapping)
        and item.get("path") == "scripts/remote_toy_mpi_runner.py"
    ]
    _require(
        len(runner_records) == 1
        and runner_records[0].get("kind") == "regular_file"
        and runner_records[0].get("size") == runner["size"]
        and runner_records[0].get("sha256") == runner["sha256"],
        "executed MPI runner bytes differ from the orchestration manifest",
    )
    normalized_environment = dict(sorted(launch_environment.items()))
    _require(
        all(isinstance(key, str) and isinstance(value, str) for key, value in normalized_environment.items()),
        "scientific launch environment must contain only string pairs",
    )
    return {
        "schema": "negfpy.remote_validation.release_launch_binding.v1",
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "execution_id": execution_id,
        "selected_rank_count": selected_rank,
        "argv": [str(value) for value in argv],
        "binding_arguments": [str(value) for value in binding_arguments],
        "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
        "launcher": _executable_identity(environment_prefix / "bin" / "mpiexec"),
        "python": _executable_identity(environment_prefix / "bin" / "python"),
        "runner": runner,
        "execution_source_identity_sha256": source_manifest_sha256,
        "tracked_files_manifest_sha256": execution_source_identity.get("tracked_files_manifest_sha256"),
        "worktree_files_manifest_sha256": execution_source_identity.get("worktree_files_manifest_sha256"),
        "orchestration_manifest_sha256": execution_source_identity.get("orchestration_manifest_sha256"),
        "orchestration_scripts": [dict(item) for item in orchestration],
        "environment_prefix_manifest_sha256": prefix_manifest_sha256,
        "environment_regular_file_count": environment_prefix_manifest.get("regular_file_count"),
        "environment_symlink_count": environment_prefix_manifest.get("symlink_count"),
        "launch_environment": normalized_environment,
        "launch_environment_sha256": _launch_environment_sha256(normalized_environment),
    }


def _write_release_launch_intent(
    *,
    path: pathlib.Path,
    attempt_root: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    **binding_arguments: Any,
) -> dict[str, Any]:
    for candidate in (stdout_path, stderr_path):
        _require(
            candidate.parent.resolve() == attempt_root.resolve()
            and not candidate.exists(),
            f"release launch log path is not fresh and canonical: {candidate}",
        )
    payload = {
        "schema": "negfpy.remote_validation.release_launch_intent.v1",
        "launch_binding": _release_launch_binding(**binding_arguments),
        "stdout_relative_path": stdout_path.relative_to(attempt_root).as_posix(),
        "stderr_relative_path": stderr_path.relative_to(attempt_root).as_posix(),
    }
    try:
        return remote_toy_mpi_runner.write_hashed_manifest(path, payload)
    except Exception as error:
        raise core.RemoteValidationError(
            f"cannot publish immutable MPI release launch intent: {error}"
        ) from error


def _validate_release_launch_intent(
    *,
    path: pathlib.Path,
    attempt_root: pathlib.Path,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
    **binding_arguments: Any,
) -> dict[str, Any]:
    try:
        document = remote_toy_mpi_runner.read_hashed_manifest(path)
    except Exception as error:
        raise core.RemoteValidationError(
            f"invalid immutable MPI release launch intent: {error}"
        ) from error
    _require(
        path.resolve() == (attempt_root / "mpi-release-launch-intent.json").resolve()
        and path.is_file()
        and not path.is_symlink(),
        "MPI release launch intent path is not canonical",
    )
    expected = {
        "schema": "negfpy.remote_validation.release_launch_intent.v1",
        "launch_binding": _release_launch_binding(**binding_arguments),
        "stdout_relative_path": stdout_path.relative_to(attempt_root).as_posix(),
        "stderr_relative_path": stderr_path.relative_to(attempt_root).as_posix(),
    }
    _require(
        document.get("payload") == expected,
        "MPI release launch intent differs from the exact launch",
    )
    return document


def _write_release_launch_receipt(
    *,
    path: pathlib.Path,
    intent_path: pathlib.Path,
    attempt_root: pathlib.Path,
    environment_prefix: pathlib.Path,
    mpi_runner: pathlib.Path,
    argv: Sequence[str],
    binding_arguments: Sequence[str],
    selected_rank: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    execution_id: str,
    execution_source_identity: Mapping[str, Any],
    environment_prefix_manifest: Mapping[str, Any],
    launch_environment: Mapping[str, str],
    post_run_verification: Mapping[str, Any],
    started_utc: str,
    finished_utc: str,
    returncode: int,
    stdout_path: pathlib.Path,
    stderr_path: pathlib.Path,
) -> dict[str, Any]:
    intent = remote_toy_mpi_runner.read_hashed_manifest(intent_path)
    intent_payload = _mapping(intent.get("payload"), name="release launch intent")
    launch_binding = dict(
        _mapping(intent_payload.get("launch_binding"), name="release launch binding")
    )
    payload = {
        "schema": "negfpy.remote_validation.release_launch.v2",
        "launch_binding": launch_binding,
        "intent": _attempt_artifact_record(attempt_root, intent_path),
        "intent_payload_sha256": intent["payload_sha256"],
        "started_utc": started_utc,
        "finished_utc": finished_utc,
        "returncode": returncode,
        "post_run_verification": dict(post_run_verification),
        "stdout": _attempt_artifact_record(attempt_root, stdout_path),
        "stderr": _attempt_artifact_record(attempt_root, stderr_path),
    }
    try:
        return remote_toy_mpi_runner.write_hashed_manifest(path, payload)
    except Exception as error:
        raise core.RemoteValidationError(
            f"cannot publish immutable MPI release launch receipt: {error}"
        ) from error


def _validate_release_launch_receipt(
    *,
    path: pathlib.Path,
    intent_path: pathlib.Path,
    attempt_root: pathlib.Path,
    environment_prefix: pathlib.Path,
    selected_rank: int,
    physical_core_count: int,
    socket_count: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    execution_id: str,
    release_root: pathlib.Path,
    mpi_runner: pathlib.Path,
    execution_source_identity: Mapping[str, Any],
    environment_prefix_manifest: Mapping[str, Any],
    launch_environment: Mapping[str, str],
) -> dict[str, Any]:
    try:
        document = remote_toy_mpi_runner.read_hashed_manifest(path)
    except Exception as error:
        raise core.RemoteValidationError(
            f"invalid immutable MPI release launch receipt: {error}"
        ) from error
    _require(
        path.resolve() == (attempt_root / "mpi-release-launch.json").resolve()
        and path.is_file()
        and not path.is_symlink(),
        "MPI release launch receipt path is not canonical",
    )
    payload = _mapping(document.get("payload"), name="MPI release launch payload")
    python = environment_prefix / "bin" / "python"
    launcher = environment_prefix / "bin" / "mpiexec"
    program = (
        str(python),
        str(mpi_runner),
        "release",
        "--run-directory",
        str(release_root),
        "--source-revision",
        source_revision,
        "--source-tree-sha256",
        source_tree_sha256,
        "--environment-sha256",
        environment_sha256,
        "--execution-id",
        execution_id,
        "--expected-ranks",
        str(selected_rank),
    )
    expected_binding = list(
        remote_mpi_scaling.openmpi_binding_arguments(
            selected_rank,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
        )
    )
    expected_argv = list(
        remote_mpi_scaling.environment_mpi_argv(
            environment_prefix,
            rank_count=selected_rank,
            physical_core_count=physical_core_count,
            socket_count=socket_count,
            program=program,
        )
    )
    launch_binding = _release_launch_binding(
        environment_prefix=environment_prefix,
        mpi_runner=mpi_runner,
        argv=expected_argv,
        binding_arguments=expected_binding,
        selected_rank=selected_rank,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        execution_id=execution_id,
        execution_source_identity=execution_source_identity,
        environment_prefix_manifest=environment_prefix_manifest,
        launch_environment=launch_environment,
    )
    expected = {
        "schema": "negfpy.remote_validation.release_launch.v2",
        "launch_binding": launch_binding,
        "returncode": 0,
    }
    changed = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    _require(not changed, f"MPI release launch receipt changed: {changed}")
    _require(
        isinstance(payload.get("started_utc"), str)
        and isinstance(payload.get("finished_utc"), str)
        and payload["started_utc"] <= payload["finished_utc"],
        "MPI release launch timestamps are invalid",
    )
    intent_document = _validate_release_launch_intent(
        path=intent_path,
        attempt_root=attempt_root,
        stdout_path=attempt_root / "mpi-release.stdout.log",
        stderr_path=attempt_root / "mpi-release.stderr.log",
        environment_prefix=environment_prefix,
        mpi_runner=mpi_runner,
        argv=expected_argv,
        binding_arguments=expected_binding,
        selected_rank=selected_rank,
        source_revision=source_revision,
        source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        execution_id=execution_id,
        execution_source_identity=execution_source_identity,
        environment_prefix_manifest=environment_prefix_manifest,
        launch_environment=launch_environment,
    )
    recorded_intent = _recorded_artifact_path(
        root=attempt_root,
        record=payload.get("intent"),
        name="MPI release launch intent",
    )
    launcher_record = _mapping(
        launch_binding.get("launcher"), name="MPI release launcher identity"
    )
    python_record = _mapping(
        launch_binding.get("python"), name="MPI release Python identity"
    )
    _require(
        recorded_intent == intent_path.resolve()
        and payload.get("intent_payload_sha256") == intent_document["payload_sha256"],
        "terminal launch receipt is not bound to its immutable intent",
    )
    post_run = _mapping(
        payload.get("post_run_verification"),
        name="post-run source/environment verification",
    )
    _require(
        post_run.get("passed") is True
        and post_run.get("execution_source_identity_sha256")
        == launch_binding["execution_source_identity_sha256"]
        and post_run.get("environment_prefix_manifest_sha256")
        == launch_binding["environment_prefix_manifest_sha256"],
        "post-run source/environment revalidation did not pass",
    )
    stdout = _recorded_artifact_path(
        root=attempt_root,
        record=payload.get("stdout"),
        name="MPI release stdout",
    )
    stderr = _recorded_artifact_path(
        root=attempt_root,
        record=payload.get("stderr"),
        name="MPI release stderr",
    )
    return {
        "schema": "negfpy.remote_validation.release_launch_facts.v1",
        "receipt_relative_path": path.relative_to(attempt_root).as_posix(),
        "receipt_file_sha256": core._sha256(path),
        "receipt_payload_sha256": document["payload_sha256"],
        "intent_relative_path": intent_path.relative_to(attempt_root).as_posix(),
        "intent_file_sha256": core._sha256(intent_path),
        "intent_payload_sha256": intent_document["payload_sha256"],
        "launcher_file_sha256": launcher_record["sha256"],
        "python_file_sha256": python_record["sha256"],
        "runner_file_sha256": launch_binding["runner"]["sha256"],
        "execution_source_identity_sha256": launch_binding["execution_source_identity_sha256"],
        "tracked_files_manifest_sha256": launch_binding["tracked_files_manifest_sha256"],
        "worktree_files_manifest_sha256": launch_binding["worktree_files_manifest_sha256"],
        "orchestration_manifest_sha256": launch_binding["orchestration_manifest_sha256"],
        "environment_prefix_manifest_sha256": launch_binding["environment_prefix_manifest_sha256"],
        "launch_environment_sha256": launch_binding["launch_environment_sha256"],
        "stdout_relative_path": stdout.relative_to(attempt_root).as_posix(),
        "stdout_file_sha256": core._sha256(stdout),
        "stderr_relative_path": stderr.relative_to(attempt_root).as_posix(),
        "stderr_file_sha256": core._sha256(stderr),
        "selected_rank_count": selected_rank,
        "binding_arguments": expected_binding,
        "post_run_verified": True,
        "returncode": 0,
    }


def _validate_release_runner_result(
    *,
    result_path: pathlib.Path,
    release_root: pathlib.Path,
    execution_id: str,
    selected_rank: int,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    socket_count: int,
    environment_prefix: pathlib.Path,
    launch_facts: Mapping[str, Any],
) -> tuple[dict[str, Any], pathlib.Path, pathlib.Path]:
    """Validate the MPI runner, campaign, execution, and strict-report chain."""

    try:
        result = remote_toy_mpi_runner.read_hashed_manifest(result_path)
    except Exception as error:
        raise core.RemoteValidationError(
            f"invalid MPI release runner result: {error}"
        ) from error
    _require(
        result_path.is_file() and not result_path.is_symlink(),
        "MPI release runner result is not a regular file",
    )
    payload = _mapping(result.get("payload"), name="MPI release runner payload")
    expected_mode = "serial" if selected_rank == 1 else "mpi"
    expected = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "release",
        "certification_eligible": True,
        "release_certified": True,
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "expected_rank_count": selected_rank,
        "recorded_rank_count": selected_rank,
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
    }
    changed = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    _require(not changed, f"MPI release runner result changed: {changed}")
    _require(
        payload.get("directory_mode") in {"fresh", "exact_resume_candidate"},
        "MPI release runner recorded an invalid directory mode",
    )
    _require(
        payload.get("thread_limits")
        == dict(sorted(core.THREAD_LIMIT_ENV.items())),
        "MPI release runner thread limits changed",
    )
    try:
        deep_result = remote_toy_mpi_runner.validate_profile_worker_result(
            result_path,
            profile="release",
            rank_count=selected_rank,
            socket_count=socket_count,
            source_revision=source_revision,
            source_tree_sha256=source_tree_sha256,
            environment_sha256=environment_sha256,
            expected_execution_id=execution_id,
            expected_directory_mode=str(payload.get("directory_mode")),
        )
    except Exception as error:
        raise core.RemoteValidationError(
            "MPI release artifact chain failed independent revalidation: "
            f"{error}"
        ) from error
    deep_evidence = _mapping(
        deep_result.get("_validated_evidence"),
        name="MPI release artifact-chain evidence",
    )
    runtime = deep_evidence.get("rank_runtime")
    expected_python = (environment_prefix.resolve() / "bin" / "python").resolve()
    _require(
        isinstance(runtime, list)
        and len(runtime) == selected_rank
        and all(
            isinstance(item, Mapping)
            and pathlib.Path(
                str(_mapping(item.get("python"), name="rank Python").get("path"))
            ).resolve()
            == expected_python
            for item in runtime
        ),
        "MPI release ranks did not use the deployed Python environment",
    )
    _require(
        launch_facts.get("selected_rank_count") == selected_rank
        and launch_facts.get("python_file_sha256")
        == deep_evidence.get("python_file_sha256"),
        "outer MPI launch receipt and rank runtime are not identity-bound",
    )
    for key in (
        "campaign_manifest_sha256",
        "execution_scientific_sha256",
        "production_toy_physics_sha256",
        "configuration_sha256",
    ):
        _require(_is_sha256(payload.get(key)), f"MPI runner {key} is invalid")

    root = release_root.resolve()
    campaign_path = _recorded_artifact_path(
        root=root,
        record=payload.get("campaign_manifest"),
        name="MPI campaign manifest",
    )
    execution_path = _recorded_artifact_path(
        root=root,
        record=payload.get("execution_manifest"),
        name="MPI execution manifest",
    )
    report_path = _recorded_artifact_path(
        root=root,
        record=payload.get("strict_report"),
        name="MPI strict production report",
    )
    _require(
        campaign_path == (root / "campaign-manifest.json").resolve()
        and execution_path
        == (root / "campaign-executions" / f"{execution_id}.json").resolve()
        and report_path
        == (root / "production-reports" / f"{execution_id}.json").resolve(),
        "MPI runner artifact paths are not canonical",
    )

    campaign = _strict_envelope(campaign_path, name="MPI campaign manifest")
    campaign_payload = _mapping(campaign["payload"], name="MPI campaign payload")
    campaign_metadata = _mapping(
        campaign_payload.get("metadata"), name="MPI campaign metadata"
    )
    campaign_tasks = campaign_payload.get("tasks")
    if isinstance(campaign_tasks, list):
        task_ids = [
            task.get("task_id") if isinstance(task, Mapping) else None
            for task in campaign_tasks
        ]
    else:
        task_ids = []
    _require(
        campaign.get("payload_sha256") == payload.get("campaign_manifest_sha256")
        and campaign_payload.get("schema_name") == "negfpy_task_manifest"
        and campaign_payload.get("schema_version") == 1
        and campaign_payload.get("source_revision") == source_revision
        and campaign_payload.get("source_tree_sha256") == source_tree_sha256
        and campaign_payload.get("environment_sha256") == environment_sha256
        and campaign_metadata.get("schema")
        == "negfpy_protocol_v2_toy_mpi_campaign"
        and campaign_metadata.get("schema_version") == 1
        and campaign_metadata.get("protocol")
        == "multidimensional_toy_gate_v2"
        and campaign_metadata.get("certification_eligible") is True
        and campaign_metadata.get("multidimensional_outcome_count") == 13
        and campaign_metadata.get("open_system_case_count") == 13
        and campaign_metadata.get("continuation_path_count") == 5
        and isinstance(campaign_tasks, list)
        and len(campaign_tasks) == 28
        and all(isinstance(task_id, str) and task_id for task_id in task_ids)
        and len(set(task_ids)) == 28,
        "MPI campaign manifest identity or topology is invalid",
    )
    execution = _strict_envelope(execution_path, name="MPI execution manifest")
    execution_payload = _mapping(
        execution["payload"], name="MPI execution payload"
    )
    outcomes = execution_payload.get("outcomes")
    outcome_ids = (
        [
            outcome.get("task_id") if isinstance(outcome, Mapping) else None
            for outcome in outcomes
        ]
        if isinstance(outcomes, list)
        else []
    )
    _require(
        execution_payload.get("schema_name")
        == "negfpy_protocol_v2_toy_mpi_execution"
        and execution_payload.get("schema_version") == 1
        and execution_payload.get("execution_id") == execution_id
        and execution_payload.get("campaign_manifest_sha256")
        == payload.get("campaign_manifest_sha256")
        and execution_payload.get("mode") == expected_mode
        and execution_payload.get("communicator_size") == selected_rank
        and execution_payload.get("complete") is True
        and execution_payload.get("scientific_sha256")
        == payload.get("execution_scientific_sha256")
        and isinstance(outcomes, list)
        and len(outcomes) == 28
        and all(
            isinstance(outcome, Mapping)
            and outcome.get("status") == "completed"
            for outcome in outcomes
        )
        and all(isinstance(task_id, str) and task_id for task_id in outcome_ids)
        and len(set(outcome_ids)) == 28
        and set(outcome_ids) == set(task_ids),
        "MPI execution manifest is incomplete or identity-mismatched",
    )
    facts = {
        "schema": "negfpy.remote_validation.mpi_execution_facts.v1",
        "runner_result_file_sha256": core._sha256(result_path),
        "runner_result_payload_sha256": result["payload_sha256"],
        "campaign_manifest_file_sha256": core._sha256(campaign_path),
        "campaign_manifest_payload_sha256": campaign["payload_sha256"],
        "execution_manifest_file_sha256": core._sha256(execution_path),
        "execution_manifest_payload_sha256": execution["payload_sha256"],
        "execution_scientific_sha256": payload["execution_scientific_sha256"],
        "production_toy_physics_sha256": payload[
            "production_toy_physics_sha256"
        ],
        "execution_id": execution_id,
        "rank_runtime": runtime,
        "python_file_sha256": deep_evidence.get("python_file_sha256"),
        "harmonic_preflight": deep_evidence.get("harmonic_preflight"),
        "memory_preflight": deep_evidence.get("memory_preflight"),
        "phase_evidence": deep_evidence.get("phase_evidence"),
        "communicator_size": selected_rank,
        "mode": expected_mode,
        "directory_mode": payload["directory_mode"],
        "task_count": 28,
        "release_certified": True,
        "launch_receipt": dict(launch_facts),
    }
    return facts, report_path, release_root / "scientific-checkpoints"


def _checkpoint_index(
    *,
    artifacts: pathlib.Path,
    multidimensional: Mapping[str, Any],
    checkpoint_root: pathlib.Path | None = None,
) -> list[dict[str, Any]]:
    checkpoints = (
        artifacts / "checkpoints" if checkpoint_root is None else checkpoint_root
    ).resolve()
    _require(checkpoints.is_relative_to(artifacts.resolve()), "unsafe checkpoint root")
    evidence = _mapping(
        multidimensional.get("checkpoint_evidence"),
        name="multidimensional.checkpoint_evidence",
    )
    records = evidence.get("records")
    _require(isinstance(records, list), "checkpoint records must be a list")
    _require(
        evidence.get("passed") is True
        and evidence.get("completed_outcome_count") == 13
        and evidence.get("total_outcome_count") == 13
        and len(records) == 13,
        "checkpoint evidence must cover all 13 completed outcomes",
    )
    indexed: list[dict[str, Any]] = []
    seen: set[pathlib.Path] = set()
    for number, raw_record in enumerate(records):
        record = _mapping(raw_record, name=f"checkpoint record {number}")
        declared = record.get("path")
        _require(
            isinstance(declared, str) and bool(declared),
            f"checkpoint record {number} lacks a path",
        )
        candidate = pathlib.Path(declared).expanduser()
        if not candidate.is_absolute():
            candidate = (
                artifacts / candidate
                if candidate.parts and candidate.parts[0] == "checkpoints"
                else checkpoints / candidate
            )
        _require(
            not candidate.is_symlink(),
            f"checkpoint path is a symlink: {declared}",
        )
        resolved = candidate.resolve(strict=True)
        _require(
            resolved.is_relative_to(checkpoints),
            f"checkpoint escapes artifact checkpoint root: {declared}",
        )
        _require(
            resolved.is_file() and not resolved.is_symlink(),
            f"checkpoint is not a regular file: {resolved}",
        )
        expected_sha = record.get("file_sha256")
        _require(
            _is_sha256(expected_sha)
            and record.get("actual_file_sha256") == expected_sha
            and record.get("valid") is True
            and core._sha256(resolved) == expected_sha,
            f"checkpoint hash/evidence mismatch: {resolved}",
        )
        _require(resolved not in seen, "checkpoint path is reused by outcomes")
        seen.add(resolved)
        indexed.append(
            {
                "run_id": record.get("run_id"),
                "report_declared_path": declared,
                "artifact_relative_path": resolved.relative_to(
                    artifacts.resolve()
                ).as_posix(),
                "file_sha256": expected_sha,
            }
        )
    return indexed


def _validate_production_report(
    *,
    path: pathlib.Path,
    artifacts: pathlib.Path,
    commit: str,
    identity: Mapping[str, Any],
    checkpoint_root: pathlib.Path,
    mpi_facts: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate current-attempt protocol-v2 evidence before publication."""

    report = _strict_json(path)
    _require(
        report.get("schema") == "negfpy.anharmonic.production_toy_gate"
        and report.get("schema_version") == 4,
        "production toy report must use schema v4",
    )
    reported_digest = report.get("report_sha256")
    _require(
        _is_sha256(reported_digest)
        and reported_digest
        == _canonical_digest(
            report, excluded_key="report_sha256", ensure_ascii=False
        ),
        "production toy report body SHA256 is invalid",
    )
    summary = _mapping(report.get("summary"), name="report.summary")
    for key in (
        "release_certified",
        "oracle_gate_passed",
        "multidimensional_release_certified",
        "multidimensional_contract_passed",
        "multidimensional_static_configuration_passed",
        "multidimensional_runtime_evidence_passed",
        "multidimensional_underlying_evidence_complete",
        "cross_platform_identity_consistent",
        "environment_lock_consistent",
    ):
        _require(summary.get(key) is True, f"report summary gate failed: {key}")
    _require(
        summary.get("validation_tier") == "release",
        "report is not release-tier evidence",
    )

    source_sha = identity.get("source_tree_sha256")
    lock_sha = identity.get("explicit_lock_sha256")
    _require(_is_sha256(source_sha), "execution source-tree SHA256 is invalid")
    _require(_is_sha256(lock_sha), "execution environment SHA256 is invalid")
    multidimensional = _mapping(
        report.get("multidimensional_scba"),
        name="report.multidimensional_scba",
    )
    _require(
        multidimensional.get("schema")
        == "negfpy.anharmonic.multidimensional_dense_toy_release"
        and multidimensional.get("schema_version") == 4
        and multidimensional.get("protocol") == "multidimensional_toy_gate_v2"
        and multidimensional.get("protocol_version") == 2,
        "multidimensional evidence is not protocol v2/schema v4",
    )
    inner_digest = multidimensional.get("report_sha256")
    _require(
        _is_sha256(inner_digest)
        and inner_digest
        == _canonical_digest(
            multidimensional,
            excluded_key="report_sha256",
            ensure_ascii=True,
        ),
        "multidimensional report body SHA256 is invalid",
    )
    _require(
        multidimensional.get("release_certified") is True
        and multidimensional.get("execution_passed") is True
        and multidimensional.get("coarse") is False,
        "multidimensional evidence is not a successful release calculation",
    )
    config = _mapping(multidimensional.get("config"), name="multidimensional.config")
    solver = _mapping(config.get("solver"), name="multidimensional.config.solver")
    _require(
        config.get("protocol") == "multidimensional_toy_gate_v2"
        and config.get("protocol_version") == 2
        and config.get("profile") == "full_preregistered"
        and config.get("certification_eligible") is True
        and config.get("full_physical_ladder_selected") is True
        and config.get("run_preregistered_refinements") is True
        and config.get("run_bounded_backend_validation") is True,
        "multidimensional configuration is not the locked release profile",
    )
    _require(
        solver.get("production_backend") == "fft"
        and solver.get("production_hilbert_backend") == "fft"
        and solver.get("surface_gf_method") == "analytic_diagonal",
        "release outcomes must use FFT/FFT with analytic diagonal contacts",
    )

    topology = _mapping(
        multidimensional.get("topology_validation"),
        name="multidimensional.topology_validation",
    )
    outcomes = multidimensional.get("outcomes")
    _require(isinstance(outcomes, list), "multidimensional outcomes must be a list")
    _require(
        len(outcomes) == 13
        and topology.get("passed") is True
        and topology.get("full_release_count") is True
        and topology.get("expected_outcome_count") == 13
        and topology.get("observed_outcome_count") == 13
        and topology.get("outcomes_exact") is True,
        "multidimensional evidence does not contain the exact 13-outcome graph",
    )
    for number, raw_outcome in enumerate(outcomes):
        outcome = _mapping(raw_outcome, name=f"outcome {number}")
        result = _mapping(outcome.get("result"), name=f"outcome {number}.result")
        _require(
            outcome.get("backend") == "fft"
            and outcome.get("status") == "converged"
            and result.get("method") == "scba"
            and result.get("converged") is True
            and result.get("diagnostics_passed") is True,
            f"outcome {number} is not a converged FFT SCBA result",
        )
        provenance = _mapping(
            result.get("provenance"), name=f"outcome {number}.provenance"
        )
        _validate_code_identity(
            provenance.get("identity"),
            commit=commit,
            source_tree_sha256=str(source_sha),
            name=f"outcome {number} code identity",
        )

    harmonic = _mapping(
        multidimensional.get("harmonic_preflight"),
        name="multidimensional.harmonic_preflight",
    )
    harmonic_points = harmonic.get("points")
    _require(
        harmonic.get("schema")
        == "negfpy.anharmonic.harmonic_same_grid_preflight"
        and harmonic.get("schema_version") == 2
        and harmonic.get("protocol") == "multidimensional_toy_gate_v2"
        and harmonic.get("passed") is True
        and harmonic.get("point_count") == 7
        and harmonic.get("all_seven_archived") is True
        and harmonic.get("required_directional_comparison_count") == 4
        and harmonic.get("critical_comparisons_passed") is True
        and harmonic.get("backend") == "fft"
        and harmonic.get("hilbert_backend") == "fft"
        and harmonic.get("surface_gf_method") == "analytic_diagonal"
        and harmonic.get("environment_sha256") == lock_sha
        and isinstance(harmonic_points, list)
        and len(harmonic_points) == 7,
        "seven-point harmonic preflight evidence is incomplete",
    )
    for number, raw_point in enumerate(harmonic_points):
        point = _mapping(raw_point, name=f"harmonic point {number}")
        provenance = _mapping(
            point.get("provenance"), name=f"harmonic point {number}.provenance"
        )
        _validate_code_identity(
            provenance.get("identity"),
            commit=commit,
            source_tree_sha256=str(source_sha),
            name=f"harmonic point {number} code identity",
        )

    backend = _mapping(
        multidimensional.get("backend_validation"),
        name="multidimensional.backend_validation",
    )
    backend_status = _mapping(
        backend.get("backend_status"), name="backend_validation.backend_status"
    )
    _require(
        backend.get("schema") == "negfpy.anharmonic.bounded_backend_validation"
        and backend.get("schema_version") == 1
        and backend.get("protocol") == "multidimensional_toy_gate_v2"
        and backend.get("scope") == "bounded_converged_scba_backend_oracle"
        and backend.get("passed") is True
        and backend.get("full_grid_serial_threaded_comparison_executed") is False
        and backend.get("backends_executed") == ["serial", "threaded", "fft"]
        and backend.get("environment_sha256") == lock_sha
        and set(backend_status) == {"serial", "threaded", "fft"},
        "bounded serial/threaded/FFT backend oracle is incomplete",
    )
    expected_backend_metadata = {
        "serial": ("serial", "direct"),
        "threaded": ("threaded", "direct"),
        "fft": ("fft", "fft"),
    }
    for name, (effective, hilbert) in expected_backend_metadata.items():
        record = _mapping(backend_status[name], name=f"bounded backend {name}")
        _require(
            record.get("method") == "scba"
            and record.get("converged") is True
            and record.get("diagnostics_passed") is True
            and record.get("effective_backend") == effective
            and record.get("hilbert_backend") == hilbert
            and record.get("environment_sha256") == lock_sha,
            f"bounded backend evidence is invalid for {name}",
        )
        provenance = _mapping(
            record.get("provenance"), name=f"bounded backend {name}.provenance"
        )
        _validate_code_identity(
            provenance.get("identity"),
            commit=commit,
            source_tree_sha256=str(source_sha),
            name=f"bounded backend {name} code identity",
        )

    multidimensional_provenance = _mapping(
        multidimensional.get("provenance"), name="multidimensional.provenance"
    )
    _require(
        multidimensional_provenance.get("passed") is True,
        "multidimensional provenance summary failed",
    )
    _validate_code_identity(
        multidimensional_provenance.get("identity"),
        commit=commit,
        source_tree_sha256=str(source_sha),
        name="multidimensional code identity",
    )
    checkpoint_evidence = _mapping(
        multidimensional.get("checkpoint_evidence"),
        name="multidimensional.checkpoint_evidence",
    )
    _require(
        checkpoint_evidence.get("environment_sha256") == lock_sha,
        "multidimensional checkpoints use a different environment lock",
    )

    environment_lock = _mapping(
        report.get("environment_lock"), name="report.environment_lock"
    )
    _require(
        environment_lock.get("passed") is True
        and environment_lock.get("open_environment_sha256") == lock_sha
        and environment_lock.get("multidimensional_environment_sha256")
        == lock_sha,
        "combined report environment-lock identity is inconsistent",
    )
    open_section = _mapping(report.get("open_three_mode"), name="open_three_mode")
    open_runs = open_section.get("runs")
    _require(
        isinstance(open_runs, list) and len(open_runs) == 13,
        "complete 13-run open-system matrix is absent",
    )
    for number, raw_run in enumerate(open_runs):
        run = _mapping(raw_run, name=f"open run {number}")
        _require(
            run.get("environment_lock_sha256") == lock_sha,
            f"open run {number} environment lock differs",
        )
        provenance = _mapping(
            run.get("provenance"), name=f"open run {number}.provenance"
        )
        _validate_code_identity(
            provenance.get("identity"),
            commit=commit,
            source_tree_sha256=str(source_sha),
            name=f"open run {number} code identity",
        )

    orchestration = _mapping(
        report.get("orchestration_provenance"),
        name="report.orchestration_provenance",
    )
    phase_hashes = orchestration.get("phase_execution_sha256")
    _require(
        orchestration.get("schema") == "negfpy.task_level_mpi_provenance"
        and orchestration.get("schema_version") == 1
        and orchestration.get("campaign_manifest_sha256")
        == mpi_facts.get("campaign_manifest_payload_sha256")
        and orchestration.get("campaign_manifest_file_sha256")
        == mpi_facts.get("campaign_manifest_file_sha256")
        and orchestration.get("execution_manifest_file_sha256")
        == mpi_facts.get("execution_manifest_file_sha256")
        and orchestration.get("execution_payload_sha256")
        == mpi_facts.get("execution_manifest_payload_sha256")
        and orchestration.get("execution_scientific_sha256")
        == mpi_facts.get("execution_scientific_sha256")
        and orchestration.get("execution_id") == mpi_facts.get("execution_id")
        and orchestration.get("mode") == mpi_facts.get("mode")
        and orchestration.get("communicator_size")
        == mpi_facts.get("communicator_size")
        and orchestration.get("complete") is True
        and orchestration.get("solver_and_gate_contract_unchanged") is True
        and isinstance(phase_hashes, list)
        and bool(phase_hashes)
        and all(_is_sha256(value) for value in phase_hashes),
        "strict report MPI orchestration provenance is incomplete or mismatched",
    )
    from negfpy.anharmonic.toy._mpi_production_gate_physics import (
        production_toy_physics_sha256,
    )

    _require(
        production_toy_physics_sha256(report)
        == mpi_facts.get("production_toy_physics_sha256"),
        "strict report canonical physics SHA256 differs from MPI runner evidence",
    )
    checkpoint_records = _checkpoint_index(
        artifacts=artifacts,
        multidimensional=multidimensional,
        checkpoint_root=checkpoint_root,
    )
    facts = {
        "schema": "negfpy.remote_validation.report_facts.v1",
        "report_relative_path": path.relative_to(artifacts).as_posix(),
        "report_file_sha256": core._sha256(path),
        "report_body_sha256": reported_digest,
        "schema_version": 4,
        "protocol": "multidimensional_toy_gate_v2",
        "protocol_version": 2,
        "release_certified": True,
        "multidimensional_outcome_count": 13,
        "multidimensional_backend": "fft",
        "harmonic_preflight_point_count": 7,
        "bounded_backends": ["serial", "threaded", "fft"],
        "open_system_run_count": 13,
        "checkpoint_count": len(checkpoint_records),
        "commit": commit,
        "git_tree": identity.get("tree"),
        "source_tree_sha256": source_sha,
        "environment_sha256": lock_sha,
        "mpi_execution": dict(mpi_facts),
    }
    return facts, checkpoint_records


def _next_attempt(artifacts: pathlib.Path) -> int:
    numbers = [0]
    status = artifacts / "job-status.json"
    if status.is_file():
        try:
            numbers.append(int(_strict_json(status).get("attempt", 0)))
        except (ValueError, TypeError, core.RemoteValidationError):
            pass
    attempts = artifacts / "attempts"
    for path in attempts.glob("attempt-[0-9][0-9][0-9]"):
        try:
            numbers.append(int(path.name.removeprefix("attempt-")))
        except ValueError:
            continue
    return max(numbers) + 1


def _invalidate_previous_publication(
    *,
    artifacts: pathlib.Path,
    attempt_root: pathlib.Path,
) -> None:
    for name, archived_name in (
        ("SHA256SUMS", "previous-SHA256SUMS"),
        ("current-attempt.json", "previous-current-attempt.json"),
        ("production_toy_gate.json", "legacy-production_toy_gate.json"),
        ("production_toy_gate.log", "legacy-production_toy_gate.log"),
        (
            "multidimensional_toy_gate_failure.json",
            "previous-multidimensional_toy_gate_failure.json",
        ),
    ):
        source = artifacts / name
        if source.exists():
            if not source.is_file() or source.is_symlink():
                raise core.RemoteValidationError(
                    f"unsafe prior publication artifact: {source}"
                )
            os.replace(source, attempt_root / archived_name)
    core._fsync_directory(artifacts)
    core._fsync_directory(attempt_root)


def _collect_attempt_sidecars(
    *,
    artifacts: pathlib.Path,
    attempt_root: pathlib.Path,
) -> None:
    """Move runner sidecars out of the shared artifact root for this attempt."""

    source = artifacts / "multidimensional_toy_gate_failure.json"
    if not source.exists():
        return
    if not source.is_file() or source.is_symlink():
        raise core.RemoteValidationError(f"unsafe runner failure sidecar: {source}")
    destination = attempt_root / "multidimensional_toy_gate_failure.json"
    if destination.exists():
        raise core.RemoteValidationError(
            f"attempt failure sidecar already exists: {destination}"
        )
    os.replace(source, destination)
    core._fsync_directory(artifacts)
    core._fsync_directory(attempt_root)


def _artifact_files(artifacts: pathlib.Path) -> list[pathlib.PurePath]:
    files: list[pathlib.PurePath] = []
    for path in artifacts.rglob("*"):
        relative = path.relative_to(artifacts)
        if relative.parts and relative.parts[0] == "job.lock":
            continue
        if path.is_symlink():
            raise core.RemoteValidationError(f"artifact symlink is forbidden: {path}")
        if path.is_file() and path.name == "SHA256SUMS" and relative.as_posix() != "SHA256SUMS":
            raise core.RemoteValidationError(
                f"nested SHA256SUMS is forbidden: {relative.as_posix()}"
            )
        if path.is_file() and relative.as_posix() != "SHA256SUMS":
            files.append(relative)
    return files


def _worker(*, deployment_dir: pathlib.Path, commit: str) -> int:
    """Run one attempt and publish only strictly validated terminal evidence."""

    root = deployment_dir.resolve()
    commit = core._validate_commit(commit)
    artifacts = root / "artifacts"
    attempts = artifacts / "attempts"
    lock_path = artifacts / "job.lock"
    artifacts.mkdir(parents=True, exist_ok=True)
    attempts.mkdir(exist_ok=True)
    if not lock_path.is_dir() or lock_path.is_symlink():
        raise core.RemoteValidationError(
            "worker requires the launch-owned regular job.lock directory"
        )

    attempt = _next_attempt(artifacts)
    attempt_root = attempts / f"attempt-{attempt:03d}"
    attempt_root.mkdir()
    _invalidate_previous_publication(
        artifacts=artifacts,
        attempt_root=attempt_root,
    )
    status_path = artifacts / "job-status.json"
    attempt_status_path = attempt_root / "status.json"
    report_path = attempt_root / "production_toy_gate.json"
    log_path = attempt_root / "production_toy_gate.log"
    checkpoint_index_path = attempt_root / "checkpoint-index.json"
    report_facts_path = attempt_root / "report-facts.json"
    scaling_root = attempt_root / "mpi-scaling"
    release_root = artifacts / "toy-release"
    execution_id = f"release-attempt-{attempt:03d}"
    archived_runner_path = attempt_root / "mpi-release-runner-result.json"
    archived_execution_path = attempt_root / "mpi-execution-manifest.json"
    archived_campaign_path = attempt_root / "mpi-campaign-manifest.json"
    release_intent_path = attempt_root / "mpi-release-launch-intent.json"
    release_receipt_path = attempt_root / "mpi-release-launch.json"
    release_stdout_path = attempt_root / "mpi-release.stdout.log"
    release_stderr_path = attempt_root / "mpi-release.stderr.log"
    started = core._utc_now()
    running = {
        "schema": "negfpy.remote_validation.job_status.v2",
        "state": "running",
        "commit": commit,
        "attempt": attempt,
        "attempt_path": attempt_root.relative_to(artifacts).as_posix(),
        "started_utc": started,
        "pid": os.getpid(),
    }
    core._atomic_write_json(status_path, running)
    core._atomic_write_json(attempt_status_path, running)
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"attempt={attempt}\n")
        log.write(f"commit={commit}\nstarted_utc={started}\n")
        log.flush()
        os.fsync(log.fileno())

    state = "failed"
    returncode = 1
    report_facts: dict[str, Any] | None = None
    scaling_facts: dict[str, Any] | None = None
    selection_path: pathlib.Path | None = None
    mpi_facts: dict[str, Any] | None = None
    identity_bundle: dict[str, Any] | None = None
    try:
        stored_path = artifacts / "identity" / "remote-environment.json"
        stored_identity = _strict_json(stored_path)
        current_identity = _record_remote_environment(
            deployment_dir=root,
            commit=commit,
            write_manifest=False,
        )
        try:
            remote_heavy_identity.compare_environment_identity(
                stored_identity, current_identity
            )
        except ValueError as error:
            raise core.RemoteValidationError(str(error)) from error
        identity_bundle = _archive_identity_bundle(
            root=root,
            commit=commit,
            stored_identity=stored_identity,
        )
        core._atomic_write_json(
            attempt_root / "execution-identity.json",
            dict(current_identity),
        )

        bad_threads = {
            name: os.environ.get(name)
            for name in core.THREAD_LIMIT_ENV
            if os.environ.get(name) != "1"
        }
        if bad_threads:
            detail = ", ".join(
                f"{name}={value!r}"
                for name, value in sorted(bad_threads.items())
            )
            raise core.RemoteValidationError(
                f"worker thread limits were not inherited as 1: {detail}"
            )

        lock_sha = str(stored_identity["explicit_lock_sha256"])
        environment_prefix = root / "environment"
        environment = _scientific_launch_environment(
            root=root,
            attempt_root=attempt_root,
            environment_prefix=environment_prefix,
        )
        for directory in (
            pathlib.Path(environment["TMPDIR"]),
            pathlib.Path(environment["XDG_CACHE_HOME"]),
            pathlib.Path(environment["MPLCONFIGDIR"]),
            pathlib.Path(environment["PYTHONPYCACHEPREFIX"]),
        ):
            directory.mkdir(parents=True, exist_ok=True)
        source_sha = str(stored_identity["source_tree_sha256"])
        physical_core_count = stored_identity.get("physical_core_count")
        socket_count = stored_identity.get("socket_count")
        _require(
            isinstance(physical_core_count, int)
            and not isinstance(physical_core_count, bool)
            and physical_core_count >= max(remote_mpi_scaling.RANK_LADDER),
            "remote identity cannot support the locked 16-rank scaling endpoint",
        )
        _require(
            isinstance(socket_count, int)
            and not isinstance(socket_count, bool)
            and socket_count > 0,
            "remote identity has an invalid socket count",
        )
        python = environment_prefix / "bin" / "python"
        mpi_runner = root / "worktree" / "scripts" / "remote_toy_mpi_runner.py"
        scale_argv = (
            str(python),
            str(mpi_runner),
            "scale",
            "--environment-prefix",
            str(environment_prefix),
            "--run-directory",
            str(scaling_root),
            "--source-revision",
            commit,
            "--source-tree-sha256",
            source_sha,
            "--environment-sha256",
            lock_sha,
            "--physical-core-count",
            str(physical_core_count),
            "--socket-count",
            str(socket_count),
        )
        prior_scaling = _prior_scaling_selection(
            attempts=attempts,
            current_attempt_root=attempt_root,
            source_revision=commit,
            source_tree_sha256=source_sha,
            environment_sha256=lock_sha,
            physical_core_count=physical_core_count,
            environment_prefix=environment_prefix,
            socket_count=socket_count,
        )
        with log_path.open("a", encoding="utf-8") as log:
            for key in sorted(core.THREAD_LIMIT_ENV):
                log.write(f"{key}=1\n")
            log.flush()
            os.fsync(log.fileno())
            try:
                if prior_scaling is None:
                    log.write(f"scaling_argv={shlex.join(scale_argv)}\n")
                    log.flush()
                    os.fsync(log.fileno())
                    completed_scaling = subprocess.run(
                        scale_argv,
                        cwd=root / "worktree",
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=False,
                        text=True,
                    )
                    returncode = int(completed_scaling.returncode)
                    if returncode != 0:
                        raise core.RemoteValidationError(
                            f"MPI scaling ladder returned {returncode}"
                        )
                    selection_path = scaling_root / "scaling-selection.json"
                    scaling_facts = _validate_scaling_selection(
                        selection_path=selection_path,
                        source_revision=commit,
                        source_tree_sha256=source_sha,
                        environment_sha256=lock_sha,
                        physical_core_count=physical_core_count,
                        environment_prefix=environment_prefix,
                        socket_count=socket_count,
                    )
                    scaling_facts["reused_for_exact_retry"] = False
                    scaling_facts["source_attempt"] = attempt_root.name
                else:
                    selection_path, scaling_facts = prior_scaling
                    log.write(
                        "scaling_selection_reused="
                        f"{selection_path.relative_to(artifacts).as_posix()}\n"
                    )
                scaling_facts["selection_relative_path"] = (
                    selection_path.relative_to(artifacts).as_posix()
                )
                selected_rank = int(scaling_facts["toy_rank_count"])
                release_program = (
                    str(python),
                    str(mpi_runner),
                    "release",
                    "--run-directory",
                    str(release_root),
                    "--source-revision",
                    commit,
                    "--source-tree-sha256",
                    source_sha,
                    "--environment-sha256",
                    lock_sha,
                    "--execution-id",
                    execution_id,
                    "--expected-ranks",
                    str(selected_rank),
                )
                release_binding_arguments = tuple(
                    remote_mpi_scaling.openmpi_binding_arguments(
                        selected_rank,
                        physical_core_count=physical_core_count,
                        socket_count=socket_count,
                    )
                )
                release_argv = remote_mpi_scaling.environment_mpi_argv(
                    environment_prefix,
                    rank_count=selected_rank,
                    physical_core_count=physical_core_count,
                    socket_count=socket_count,
                    program=release_program,
                )
                prelaunch_identity = _record_remote_environment(
                    deployment_dir=root,
                    commit=commit,
                    write_manifest=False,
                )
                try:
                    remote_heavy_identity.compare_environment_identity(
                        stored_identity,
                        prelaunch_identity,
                    )
                except ValueError as error:
                    raise core.RemoteValidationError(
                        f"pre-launch identity changed: {error}"
                    ) from error
                execution_source_identity = dict(
                    _mapping(
                        prelaunch_identity.get("execution_source_identity"),
                        name="pre-launch execution source identity",
                    )
                )
                environment_prefix_manifest = dict(
                    _mapping(
                        prelaunch_identity.get("environment_prefix_manifest"),
                        name="pre-launch environment prefix manifest",
                    )
                )
                intent_arguments = {
                    "environment_prefix": environment_prefix,
                    "mpi_runner": mpi_runner,
                    "argv": release_argv,
                    "binding_arguments": release_binding_arguments,
                    "selected_rank": selected_rank,
                    "source_revision": commit,
                    "source_tree_sha256": source_sha,
                    "environment_sha256": lock_sha,
                    "execution_id": execution_id,
                    "execution_source_identity": execution_source_identity,
                    "environment_prefix_manifest": environment_prefix_manifest,
                    "launch_environment": environment,
                }
                _write_release_launch_intent(
                    path=release_intent_path,
                    attempt_root=attempt_root,
                    stdout_path=release_stdout_path,
                    stderr_path=release_stderr_path,
                    **intent_arguments,
                )
                _validate_release_launch_intent(
                    path=release_intent_path,
                    attempt_root=attempt_root,
                    stdout_path=release_stdout_path,
                    stderr_path=release_stderr_path,
                    **intent_arguments,
                )
                log.write(f"release_argv={shlex.join(release_argv)}\n")
                log.write(
                    "release_receipt="
                    f"{release_receipt_path.relative_to(artifacts).as_posix()}\n"
                )
                log.flush()
                os.fsync(log.fileno())
                release_started = core._utc_now()
                with (
                    release_stdout_path.open("x", encoding="utf-8") as stdout,
                    release_stderr_path.open("x", encoding="utf-8") as stderr,
                ):
                    completed_release = subprocess.run(
                        release_argv,
                        cwd=root / "worktree",
                        env=environment,
                        stdout=stdout,
                        stderr=stderr,
                        check=False,
                        text=True,
                    )
                    for stream in (stdout, stderr):
                        stream.flush()
                        os.fsync(stream.fileno())
                release_finished = core._utc_now()
                returncode = int(completed_release.returncode)
                try:
                    post_identity = _record_remote_environment(
                        deployment_dir=root,
                        commit=commit,
                        write_manifest=False,
                    )
                    remote_heavy_identity.compare_environment_identity(
                        stored_identity,
                        post_identity,
                    )
                    post_source = _mapping(
                        post_identity.get("execution_source_identity"),
                        name="post-run execution source identity",
                    )
                    post_prefix = _mapping(
                        post_identity.get("environment_prefix_manifest"),
                        name="post-run environment prefix manifest",
                    )
                    post_run_verification = {
                        "passed": True,
                        "checked_utc": core._utc_now(),
                        "execution_source_identity_sha256": _verified_manifest_digest(
                            post_source,
                            name="post-run execution source identity",
                        ),
                        "environment_prefix_manifest_sha256": _verified_manifest_digest(
                            post_prefix,
                            name="post-run environment prefix manifest",
                        ),
                    }
                except Exception as error:
                    post_run_verification = {
                        "passed": False,
                        "checked_utc": core._utc_now(),
                        "error_type": type(error).__name__,
                        "error": str(error),
                    }
                _write_release_launch_receipt(
                    path=release_receipt_path,
                    intent_path=release_intent_path,
                    attempt_root=attempt_root,
                    environment_prefix=environment_prefix,
                    mpi_runner=mpi_runner,
                    argv=release_argv,
                    binding_arguments=release_binding_arguments,
                    selected_rank=selected_rank,
                    source_revision=commit,
                    source_tree_sha256=source_sha,
                    environment_sha256=lock_sha,
                    execution_id=execution_id,
                    execution_source_identity=execution_source_identity,
                    environment_prefix_manifest=environment_prefix_manifest,
                    launch_environment=environment,
                    post_run_verification=post_run_verification,
                    started_utc=release_started,
                    finished_utc=release_finished,
                    returncode=returncode,
                    stdout_path=release_stdout_path,
                    stderr_path=release_stderr_path,
                )
                if post_run_verification.get("passed") is not True:
                    raise core.RemoteValidationError(
                        "post-run source/environment identity changed"
                    )
                if returncode != 0:
                    raise core.RemoteValidationError(
                        f"locked MPI toy release returned {returncode}"
                    )
            finally:
                _collect_attempt_sidecars(
                    artifacts=artifacts,
                    attempt_root=attempt_root,
                )
        launch_facts = _validate_release_launch_receipt(
            path=release_receipt_path,
            intent_path=release_intent_path,
            attempt_root=attempt_root,
            environment_prefix=environment_prefix,
            selected_rank=int(scaling_facts["toy_rank_count"]),
            physical_core_count=physical_core_count,
            socket_count=socket_count,
            source_revision=commit,
            source_tree_sha256=source_sha,
            environment_sha256=lock_sha,
            execution_id=execution_id,
            release_root=release_root,
            mpi_runner=mpi_runner,
            execution_source_identity=execution_source_identity,
            environment_prefix_manifest=environment_prefix_manifest,
            launch_environment=environment,
        )
        runner_result_path = (
            release_root / "runner-results" / f"{execution_id}.json"
        )
        mpi_facts, strict_report_source, checkpoint_root = (
            _validate_release_runner_result(
                result_path=runner_result_path,
                release_root=release_root,
                execution_id=execution_id,
                selected_rank=int(scaling_facts["toy_rank_count"]),
                source_revision=commit,
                source_tree_sha256=source_sha,
                launch_facts=launch_facts,
                environment_sha256=lock_sha,
                socket_count=socket_count,
                environment_prefix=environment_prefix,
            )
        )
        core._atomic_copy_file(strict_report_source, report_path)
        core._atomic_copy_file(runner_result_path, archived_runner_path)
        core._atomic_copy_file(
            release_root / "campaign-manifest.json",
            archived_campaign_path,
        )
        core._atomic_copy_file(
            release_root / "campaign-executions" / f"{execution_id}.json",
            archived_execution_path,
        )
        report_facts, checkpoint_records = _validate_production_report(
            path=report_path,
            artifacts=artifacts,
            commit=commit,
            identity=stored_identity,
            checkpoint_root=checkpoint_root,
            mpi_facts=mpi_facts,
        )
        report_facts["mpi_scaling"] = scaling_facts
        core._atomic_write_json(
            checkpoint_index_path,
            {
                "schema": "negfpy.remote_validation.checkpoint_index.v1",
                "attempt": attempt,
                "checkpoint_root": checkpoint_root.relative_to(
                    artifacts
                ).as_posix(),
                "count": len(checkpoint_records),
                "records": checkpoint_records,
            },
        )
        core._atomic_write_json(report_facts_path, report_facts)
        state = "completed"
    except BaseException as error:
        if returncode == 0:
            returncode = 2
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"worker_exception={type(error).__name__}: {error}\n"
            )
            log.flush()
            os.fsync(log.fileno())

    finished = core._utc_now()
    terminal: dict[str, Any] = {
        "schema": "negfpy.remote_validation.job_status.v2",
        "state": state,
        "commit": commit,
        "attempt": attempt,
        "attempt_path": attempt_root.relative_to(artifacts).as_posix(),
        "started_utc": started,
        "finished_utc": finished,
        "returncode": returncode,
        "identity_bundle": identity_bundle,
        "report_facts": report_facts,
        "scaling_facts": scaling_facts,
        "mpi_execution_facts": mpi_facts,
    }
    publication_path = attempt_root / "publication.json"
    current_path = artifacts / "current-attempt.json"
    try:
        core._atomic_write_json(attempt_status_path, terminal)
        core._atomic_write_json(status_path, terminal)
        publication = {
            "schema": "negfpy.remote_validation.attempt_publication.v1",
            "attempt": attempt,
            "state": state,
            "commit": commit,
            "attempt_path": attempt_root.relative_to(artifacts).as_posix(),
            "status": attempt_status_path.relative_to(artifacts).as_posix(),
            "status_sha256": core._sha256(attempt_status_path),
            "log": log_path.relative_to(artifacts).as_posix(),
            "log_sha256": core._sha256(log_path),
            "report": (
                report_path.relative_to(artifacts).as_posix()
                if report_path.is_file()
                else None
            ),
            "report_sha256": (
                core._sha256(report_path) if report_path.is_file() else None
            ),
            "report_facts": report_facts,
            "scaling_selection": (
                selection_path.relative_to(artifacts).as_posix()
                if selection_path is not None and selection_path.is_file()
                else None
            ),
            "scaling_selection_sha256": (
                core._sha256(selection_path)
                if selection_path is not None and selection_path.is_file()
                else None
            ),
            "mpi_release_launch_intent": (
                release_intent_path.relative_to(artifacts).as_posix()
                if release_intent_path.is_file()
                else None
            ),
            "mpi_release_launch_intent_sha256": (
                core._sha256(release_intent_path)
                if release_intent_path.is_file()
                else None
            ),
            "mpi_release_launch": (
                release_receipt_path.relative_to(artifacts).as_posix()
                if release_receipt_path.is_file()
                else None
            ),
            "mpi_release_launch_sha256": (
                core._sha256(release_receipt_path)
                if release_receipt_path.is_file()
                else None
            ),
            "mpi_release_stdout": (
                release_stdout_path.relative_to(artifacts).as_posix()
                if release_stdout_path.is_file()
                else None
            ),
            "mpi_release_stdout_sha256": (
                core._sha256(release_stdout_path)
                if release_stdout_path.is_file()
                else None
            ),
            "mpi_release_stderr": (
                release_stderr_path.relative_to(artifacts).as_posix()
                if release_stderr_path.is_file()
                else None
            ),
            "mpi_release_stderr_sha256": (
                core._sha256(release_stderr_path)
                if release_stderr_path.is_file()
                else None
            ),
            "mpi_runner_result": (
                archived_runner_path.relative_to(artifacts).as_posix()
                if archived_runner_path.is_file()
                else None
            ),
            "mpi_runner_result_sha256": (
                core._sha256(archived_runner_path)
                if archived_runner_path.is_file()
                else None
            ),
            "mpi_execution_manifest": (
                archived_execution_path.relative_to(artifacts).as_posix()
                if archived_execution_path.is_file()
                else None
            ),
            "mpi_execution_manifest_sha256": (
                core._sha256(archived_execution_path)
                if archived_execution_path.is_file()
                else None
            ),
            "mpi_campaign_manifest": (
                archived_campaign_path.relative_to(artifacts).as_posix()
                if archived_campaign_path.is_file()
                else None
            ),
            "mpi_campaign_manifest_sha256": (
                core._sha256(archived_campaign_path)
                if archived_campaign_path.is_file()
                else None
            ),
        }
        core._atomic_write_json(publication_path, publication)
        core._atomic_write_json(current_path, publication)
        core._write_sha256sums(artifacts, _artifact_files(artifacts))
        shutil.rmtree(lock_path)
        core._fsync_directory(artifacts)
    except BaseException as error:
        returncode = 1
        state = "finalization_failed"
        failed_terminal = {
            **terminal,
            "state": state,
            "returncode": returncode,
            "finalization_error": f"{type(error).__name__}: {error}",
        }
        for path in (attempt_status_path, status_path):
            try:
                core._atomic_write_json(path, failed_terminal)
            except BaseException:
                pass
        try:
            core._atomic_write_json(
                publication_path,
                {
                    "schema": "negfpy.remote_validation.attempt_publication.v1",
                    "attempt": attempt,
                    "state": state,
                    "commit": commit,
                    "finalization_error": failed_terminal[
                        "finalization_error"
                    ],
                },
            )
            core._atomic_write_json(
                current_path,
                _strict_json(publication_path),
            )
        except BaseException:
            pass
        checksum_path = artifacts / "SHA256SUMS"
        if checksum_path.exists():
            try:
                checksum_path.unlink()
                core._fsync_directory(artifacts)
            except OSError:
                pass
        if not lock_path.exists():
            try:
                lock_path.mkdir()
                core._fsync_directory(artifacts)
            except OSError:
                pass
        # Keep job.lock: status reports this interrupted publication as stale.
    return returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Key-only remote heavy-validation orchestration."
    )
    commands = parser.add_subparsers(dest="operation", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--repo-root", type=pathlib.Path, default=pathlib.Path.cwd())
    prepare.add_argument(
        "--environment-lock",
        type=pathlib.Path,
        default=explicit.DEFAULT_LOCK,
    )
    prepare.add_argument(
        "--staging-root",
        type=pathlib.Path,
        default=pathlib.Path("validation-artifacts/deployments"),
    )
    prepare.add_argument("--dry-run", action="store_true")
    preflight = commands.add_parser("preflight")
    preflight.add_argument("--output", type=pathlib.Path)
    preflight.add_argument("--dry-run", action="store_true")
    deploy = commands.add_parser("deploy")
    deploy.add_argument("staging", type=pathlib.Path)
    deploy.add_argument("--dry-run", action="store_true")
    launch = commands.add_parser("launch")
    launch.add_argument("commit")
    launch.add_argument("--session")
    launch.add_argument("--dry-run", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("commit")
    status.add_argument("--session")
    status.add_argument("--dry-run", action="store_true")
    fetch = commands.add_parser("fetch")
    fetch.add_argument("commit")
    fetch.add_argument(
        "--destination-root",
        type=pathlib.Path,
        default=pathlib.Path("validation-artifacts/remote"),
    )
    fetch.add_argument("--dry-run", action="store_true")
    record = commands.add_parser("_record-remote-environment", help=argparse.SUPPRESS)
    record.add_argument("--deployment-dir", type=pathlib.Path, required=True)
    record.add_argument("--commit", required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--deployment-dir", type=pathlib.Path, required=True)
    worker.add_argument("--commit", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "_record-remote-environment":
            _record_remote_environment(
                deployment_dir=args.deployment_dir, commit=args.commit
            )
            return 0
        if args.operation == "_worker":
            return _worker(deployment_dir=args.deployment_dir, commit=args.commit)
        runner = core.CommandRunner(dry_run=args.dry_run)
        if args.operation == "prepare":
            value = explicit.prepare(
                repo_root=args.repo_root,
                staging_root=args.staging_root,
                environment_lock=args.environment_lock,
                runner=runner,
            )
        elif args.operation == "preflight":
            value = explicit.preflight(runner=runner, output=args.output)
        elif args.operation == "deploy":
            value = deploy(staging=args.staging, runner=runner)
        elif args.operation == "launch":
            value = launch(commit=args.commit, session=args.session, runner=runner)
        elif args.operation == "status":
            value = operations.status(
                commit=args.commit, session=args.session, runner=runner
            )
        elif args.operation == "fetch":
            value = operations.fetch(
                commit=args.commit,
                destination_root=args.destination_root,
                runner=runner,
            )
        else:  # pragma: no cover
            raise AssertionError(args.operation)
        if isinstance(value, Mapping):
            print(json.dumps(value, indent=2, sort_keys=True))
        elif value is not None:
            print(value)
    except core.RemoteValidationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
