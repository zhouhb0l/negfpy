#!/usr/bin/env python3
"""Command-line entry point for NEGFPy's key-only remote validation."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import pathlib
import platform
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

import remote_heavy_runtime as operations
import remote_heavy_identity
import remote_heavy_explicit as explicit
import remote_heavy_validation as core


def _remote_bootstrap_command(commit: str) -> str:
    remote = core._remote_shell_dir(commit)
    return " && ".join(
        (
            f"mkdir -p {remote}/incoming {remote}/artifacts/checkpoints",
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
            f"env PYTHONPATH={remote}/worktree/src {remote}/environment/bin/python "
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
    session = core._validate_session(session)
    limits = " ".join(
        f"{key}=1" for key in sorted(core.THREAD_LIMIT_ENV)
    )
    worker = (
        f"env PYTHONPATH={remote}/worktree/src {limits} "
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
            "explicit_lock_sha256": lock_sha,
            "tree": tree,
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
            "cpu_model": cpu_model,
            "memory_total_bytes": memory_total,
            "package_source": package_source,
            "memory_available_bytes": memory_available,
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "distributions": versions,
            "deployment_disk_total_bytes": disk.total,
            "deployment_disk_free_bytes": disk.free,
            "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
    }
    if write_manifest:
        core._atomic_write_json(
            root / "artifacts" / "remote-environment.json", manifest
        )

    return manifest

def _worker(*, deployment_dir: pathlib.Path, commit: str) -> int:
    """Remote-only release worker run in tmux with durable terminal evidence."""

    root = deployment_dir.resolve()
    commit = core._validate_commit(commit)
    artifacts = root / "artifacts"
    checkpoints = artifacts / "checkpoints"
    attempt_dir = artifacts / "attempts"
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    attempt_dir.mkdir(exist_ok=True)

    status_path = artifacts / "job-status.json"
    report_path = artifacts / "production_toy_gate.json"
    log_path = artifacts / "production_toy_gate.log"
    started = core._utc_now()
    attempt = 1
    if status_path.is_file():
        try:
            previous = json.loads(status_path.read_text(encoding="utf-8"))
            attempt = int(previous.get("attempt", 0)) + 1
        except (OSError, ValueError, json.JSONDecodeError):
            attempt = 1

    core._atomic_write_json(
        status_path,
        {
            "schema": "negfpy.remote_validation.job_status.v1",
            "state": "running",
            "commit": commit,
            "attempt": attempt,
            "started_utc": started,
            "pid": os.getpid(),
        },
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== attempt {attempt} ===\n")
        log.write(f"commit={commit}\nstarted_utc={started}\n")
        log.flush()

    argv = (
        str(root / "environment" / "bin" / "python"),
        "-m",
        "pytest",
        "-q",
        "tests/test_anharmonic_production_gate_runner.py",
        "-m",
        "nightly",
    )
    state = "failed"
    returncode = 1
    try:
        stored_path = artifacts / "remote-environment.json"
        try:
            stored_identity = json.loads(
                stored_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise core.RemoteValidationError(
                f"missing immutable deploy-time environment: {stored_path}"
            ) from error
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

        lock_sha = core._sha256(
            root / "incoming" / "environment-explicit.lock"
        )
        environment = os.environ.copy()
        environment.update(core.THREAD_LIMIT_ENV)
        environment["PATH"] = (
            core.REMOTE_TOOLS_BIN
            + os.pathsep
            + environment.get("PATH", "")
        )
        environment.update(
            {
                "PYTHONPATH": str(root / "worktree" / "src"),
                "PYTHONHASHSEED": "0",
                "NEGFPY_RUN_FULL_ANHARMONIC_TOY_GATE": "1",
                "NEGFPY_TOY_GATE_REPORT": str(report_path),
                "NEGFPY_TOY_GATE_CHECKPOINT_DIR": str(checkpoints),
                "NEGFPY_VALIDATION_ENV_SHA256": lock_sha,
                "NEGFPY_TOY_GATE_RESUME": "1",
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"argv={shlex.join(argv)}\n")
            for key in sorted(core.THREAD_LIMIT_ENV):
                log.write(f"{key}=1\n")
            log.flush()
            completed = subprocess.run(
                argv,
                cwd=root / "worktree",
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                text=True,
            )
        returncode = completed.returncode
        state = "completed" if returncode == 0 else "failed"
    except BaseException as error:
        with log_path.open("a", encoding="utf-8") as log:
            log.write(
                f"worker_exception={type(error).__name__}: {error}\n"
            )
            log.flush()
    finally:
        terminal = {
            "schema": "negfpy.remote_validation.job_status.v1",
            "state": state,
            "commit": commit,
            "attempt": attempt,
            "started_utc": started,
            "finished_utc": core._utc_now(),
            "returncode": returncode,
        }
        core._atomic_write_json(status_path, terminal)
        core._atomic_write_json(
            attempt_dir / f"attempt-{attempt:03d}-status.json",
            terminal,
        )
        try:
            files: list[pathlib.PurePath] = []
            for path in artifacts.rglob("*"):
                relative = path.relative_to(artifacts)
                if relative.parts and relative.parts[0] == "job.lock":
                    continue
                if path.is_symlink():
                    raise core.RemoteValidationError(
                        f"artifact symlink is forbidden: {path}"
                    )
                if path.is_file() and path.name != "SHA256SUMS":
                    files.append(relative)
            core._write_sha256sums(artifacts, files)
        finally:
            # SHA256SUMS is the publication barrier; release the lock last.
            shutil.rmtree(artifacts / "job.lock", ignore_errors=True)
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
