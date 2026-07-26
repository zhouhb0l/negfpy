"""Explicit-micromamba-lock preparation and preflight helpers."""

from __future__ import annotations

import json
import pathlib
import platform
import shlex
import shutil
from typing import Any

import remote_heavy_lock
import remote_heavy_validation as core


DEFAULT_LOCK = pathlib.Path("validation/locks/negf-toy-gate-linux-64.lock")


def _git(
    runner: core.CommandRunner,
    root: pathlib.Path,
    *arguments: str,
) -> core.Completed:
    return runner.run(("git", *arguments), cwd=root)


def prepare(
    *,
    repo_root: pathlib.Path,
    staging_root: pathlib.Path,
    environment_lock: pathlib.Path,
    runner: core.CommandRunner,
) -> pathlib.Path:
    """Bundle a clean HEAD with the pinned explicit micromamba lock."""

    root = repo_root.resolve()
    lock = (
        environment_lock
        if environment_lock.is_absolute()
        else root / environment_lock
    ).resolve()
    core._require_clean_repository(runner, root)
    result = _git(runner, root, "rev-parse", "--verify", "HEAD")
    commit = "0" * 40 if runner.dry_run else core._validate_commit(result.stdout)
    staging = staging_root.resolve() / commit
    if runner.dry_run:
        print(f"would verify {lock}")
        print(f"would prepare {staging}")
        return staging
    try:
        lock_sha = remote_heavy_lock.validate_explicit_lock(
            lock, expected_sha256=core.EXPECTED_ENV_LOCK_SHA256
        )
    except ValueError as error:
        raise core.RemoteValidationError(str(error)) from error
    if staging.exists():
        raise core.RemoteValidationError(
            f"staging directory already exists: {staging}"
        )
    staging.mkdir(parents=True)
    bundle = staging / "negfpy.bundle"
    _git(runner, root, "bundle", "create", str(bundle), "HEAD")
    _git(runner, root, "bundle", "verify", str(bundle))
    tree = _git(runner, root, "rev-parse", "HEAD^{tree}").stdout.strip()
    shutil.copyfile(lock, staging / "environment-explicit.lock")
    core._atomic_write_json(
        staging / "environment-manifest.json",
        {
            "schema": "negfpy.remote_validation.environment.v2",
            "created_utc": core._utc_now(),
            "explicit_lock_sha256": lock_sha,
            "source_lock": str(environment_lock),
            "preparation_python": platform.python_version(),
        },
    )
    core._atomic_write_json(
        staging / "deployment-manifest.json",
        {
            "schema": "negfpy.remote_validation.deployment.v2",
            "created_utc": core._utc_now(),
            "source": {"commit": commit, "tree": tree, "dirty": False},
            "environment": {
                "explicit_lock": "environment-explicit.lock",
                "explicit_lock_sha256": lock_sha,
                "remote_prefix": f"~/{core._remote_relative_dir(commit)}/environment",
            },
            "remote": {
                "alias": core.SSH_ALIAS,
                "host": core.EXPECTED_HOST,
                "user": core.EXPECTED_USER,
                "root": f"~/{core._remote_relative_dir(commit)}",
                "expected_ed25519_fingerprint": (
                    core.EXPECTED_ED25519_FINGERPRINT
                ),
                "micromamba": core.REMOTE_MICROMAMBA,
                "tools_bin": core.REMOTE_TOOLS_BIN,
            },
            "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
        },
    )
    core._write_sha256sums(
        staging,
        map(
            pathlib.PurePath,
            (
                "deployment-manifest.json",
                "environment-explicit.lock",
                "environment-manifest.json",
                "negfpy.bundle",
            ),
        ),
    )
    core.verify_sha256sums(staging)
    return staging


_MACHINE_PROGRAM = """\
import json, os, pathlib, platform, shutil, socket, sys
disk = shutil.disk_usage(pathlib.Path.home())
try:
    memory = int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
except (AttributeError, OSError, ValueError):
    memory = None
cpu_model = platform.processor()
cpuinfo = pathlib.Path("/proc/cpuinfo")
if cpuinfo.is_file():
    for line in cpuinfo.read_text().splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
memory_available = None
meminfo = pathlib.Path("/proc/meminfo")
if meminfo.is_file():
    values = dict(line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line)
    memory_available = int(values["MemAvailable"].split()[0]) * 1024
allowed = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else set(range(os.cpu_count() or 0))
physical_cores = set()
sockets = set()
for cpu in sorted(allowed):
    topology = pathlib.Path("/sys/devices/system/cpu") / f"cpu{cpu}" / "topology"
    try:
        package_id = int((topology / "physical_package_id").read_text().strip())
        core_id = int((topology / "core_id").read_text().strip())
    except (OSError, ValueError):
        continue
    sockets.add(package_id)
    physical_cores.add((package_id, core_id))
if not physical_cores:
    physical_cores = {(0, cpu) for cpu in allowed}
if not sockets:
    sockets = {0}
print(json.dumps({
    "schema": "negfpy.remote_validation.machine.v1",
    "hostname": socket.gethostname(), "platform": platform.platform(),
    "machine": platform.machine(), "cpu_count": os.cpu_count(),
    "affinity_logical_cpu_count": len(allowed),
    "physical_core_count": len(physical_cores),
    "socket_count": len(sockets),
    "cpu_model": cpu_model, "memory_total_bytes": memory,
    "memory_available_bytes": memory_available, "home_disk_total_bytes": disk.total,
    "home_disk_free_bytes": disk.free, "python_executable": sys.executable,
    "python_version": platform.python_version(),
}, sort_keys=True))
"""


def preflight(
    *,
    runner: core.CommandRunner,
    output: pathlib.Path | None,
) -> dict[str, Any]:
    """Verify key-only access and all fixed workstation executables."""

    alias = core.check_ssh_alias(runner)
    required = (
        core.REMOTE_MICROMAMBA,
        core.REMOTE_GIT,
        core.REMOTE_TMUX,
        core.REMOTE_RSYNC,
        "sha256sum",
        "python3",
    )
    checks = " && ".join(
        f"command -v {shlex.quote(item)} >/dev/null"
        if "/" not in item
        else f"test -x {shlex.quote(item)}"
        for item in required
    )
    command = (
        checks
        + " && "
        + "python3"
        + " -c "
        + shlex.quote(_MACHINE_PROGRAM)
    )
    result = runner.run(core.ssh_argv(command))
    if runner.dry_run:
        return {"alias": alias, "dry_run": True}
    try:
        machine = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise core.RemoteValidationError(
            "remote preflight did not return JSON"
        ) from error
    report = {
        "schema": "negfpy.remote_validation.preflight.v2",
        "created_utc": core._utc_now(),
        "ssh": alias,
        "machine": machine,
        "required_executables": list(required),
        "key_only_batch_login": True,
    }
    if output is not None:
        core._atomic_write_json(output.resolve(), report)
    return report
