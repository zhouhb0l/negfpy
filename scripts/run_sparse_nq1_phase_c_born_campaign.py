#!/usr/bin/env python3
"""Seal and supervise one remote Nq=1, Nw=17 Born timing gate.

The coordinator closes provenance and infrastructure gates before launching
the scientific subprocess.  A two-hour wall watchdog and 8-GiB resident-set
watchdog terminate the process group with TERM, then KILL after 60 seconds.
Timeouts and infrastructure failures are terminal archived outcomes and are
never retried automatically.  Scientific failures likewise retain the exact
fixed controls and are never retuned.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import negfpy

import validate_sparse_nq1_phase_c_born as phase_c


SCHEMA = "negfpy.sparse-nq1-phase-c-born-campaign.v1"
STATUS_SCHEMA = "negfpy.sparse-nq1-phase-c-born-status.v1"
REPORT_SCHEMA = "negfpy.sparse-nq1-phase-c-born-report.v1"
MATERIAL_LOCK_SHA256 = (
    "6d8e017b6f1ead8be3ec9fb8abb1c8c1d02fe0de852aad9cbfc69d08ce341bfc"
)
WALL_LIMIT_SECONDS = 2 * 60 * 60
TERM_GRACE_SECONDS = 60
# The one-shot Born timing gate and the material SCBA solve have different
# memory envelopes.  Keep the Born contract at its original 4-GiB bound;
# the SCBA wrapper passes its own explicit 8-GiB limit to the supervisor.
RSS_LIMIT_BYTES = 4 * 1024**3
RSS_POLL_SECONDS = 2.0
EXPECTED_DISTRIBUTIONS = {
    "h5py": "3.16.0",
    "numpy": "2.5.1",
    "phonopy": "4.4.0",
    "phono3py": "4.4.0",
    "scipy": "1.18.0",
    "spglib": "2.7.0",
    "threadpoolctl": "3.6.0",
}
EXPECTED_PYTHON_VERSION = "3.12.13"


class CampaignError(RuntimeError):
    """A fail-closed Phase-C campaign error."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise CampaignError(f"immutable campaign artifact already exists: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise CampaignError(f"immutable campaign artifact raced: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_exact(path: Path, content: bytes) -> None:
    """Publish once, or accept an identical finalization artifact."""

    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise CampaignError(f"existing finalization artifact differs: {path}")
        return
    _publish_once(path, content)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise CampaignError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _source_identity(repo: Path, expected_commit: str) -> dict[str, Any]:
    expected = expected_commit.strip().lower()
    if (
        len(expected) != 40
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise CampaignError("--expected-commit must be a full lowercase SHA-1.")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != expected:
        raise CampaignError(f"source commit mismatch: {commit} != {expected}")
    if _git(repo, "status", "--porcelain=v1", "--untracked-files=all"):
        raise CampaignError("remote scientific worktree is not clean.")
    package_path = Path(negfpy.__file__).resolve()
    expected_package = (repo / "src" / "negfpy").resolve()
    try:
        package_path.relative_to(expected_package)
    except ValueError as error:
        raise CampaignError(
            f"negfpy imported from {package_path}, not {expected_package}"
        ) from error
    return {
        "commit": commit,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "package_file": str(package_path),
    }


def _validate_lock(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"material lock is not a regular file: {path}")
    first = next(
        (
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if first != "@EXPLICIT":
        raise CampaignError("material environment lock is not explicit.")
    digest = _sha256_file(path)
    if digest != MATERIAL_LOCK_SHA256:
        raise CampaignError(
            f"material lock SHA256 mismatch: {digest} != {MATERIAL_LOCK_SHA256}"
        )
    return {"path": str(path), "sha256": digest, "format": "@EXPLICIT"}


def _versions() -> dict[str, str]:
    names = (
        "h5py",
        "numpy",
        "phonopy",
        "phono3py",
        "scipy",
        "spglib",
        "threadpoolctl",
    )
    try:
        result = {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise CampaignError(f"missing locked distribution: {error.name}") from error
    if result != EXPECTED_DISTRIBUTIONS:
        raise CampaignError(
            f"installed distributions differ from the explicit lock: {result!r}."
        )
    if platform.python_version() != EXPECTED_PYTHON_VERSION:
        raise CampaignError(
            "Python differs from the explicit lock: "
            f"{platform.python_version()} != {EXPECTED_PYTHON_VERSION}."
        )
    return result


def _stable_preflight_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Drop only instantaneous resource values from an authenticated preflight."""

    memory = dict(value["memory"])
    memory.pop("available_bytes", None)
    return {
        "fixture_path": value["fixture_path"],
        "fixture": value["fixture"],
        "fc3_full": value["fc3_full"],
        "phase_b": value["phase_b"],
        "thread_limits": value["thread_limits"],
        "controls": value["controls"],
        "problem_input_hashes": value["problem_input_hashes"],
        "configuration_sha256": value["configuration_sha256"],
        "assembled_vertex": value["assembled_vertex"],
        "dimensions": value["dimensions"],
        "sparse_schedule": value["sparse_schedule"],
        "memory": memory,
    }


def build_manifest(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifacts = artifacts.resolve()
    output_json = artifacts / "outcomes" / "phase-c-born.json"
    output_hdf5 = artifacts / "outcomes" / "phase-c-born-fields.h5"
    source = _source_identity(repo, expected_commit)
    lock = _validate_lock(environment_lock)
    preflight = phase_c.preflight_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
        output_json=output_json,
        output_hdf5=output_hdf5,
    )
    driver = (repo / "scripts" / "validate_sparse_nq1_phase_c_born.py").resolve()
    if not driver.is_file():
        raise CampaignError(f"missing Phase-C driver: {driver}")
    return {
        "schema": SCHEMA,
        "scientific_scope": (
            "one fixed Nq=1,Nw=17 contact-only Born bubble and one updated "
            "solve; timing/memory gate only"
        ),
        "source": source,
        "environment_lock": lock,
        "driver": {
            "relative_path": "scripts/validate_sparse_nq1_phase_c_born.py",
            "sha256": _sha256_file(driver),
        },
        "preflight": preflight,
        "fixed_controls_no_retuning": True,
        "watchdog": {
            "wall_limit_seconds": WALL_LIMIT_SECONDS,
            "rss_limit_bytes": RSS_LIMIT_BYTES,
            "term_grace_seconds": TERM_GRACE_SECONDS,
            "rss_poll_seconds": RSS_POLL_SECONDS,
            "automatic_retry": False,
        },
        "outputs": {"json": str(output_json), "hdf5": str(output_hdf5)},
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "distributions": _versions(),
        "platform": platform.platform(),
        "thread_limits": dict(sorted(phase_c.THREAD_LIMIT_ENV.items())),
    }


def _load_canonical(path: Path, *, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"{name} is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict) or path.read_bytes() != _canonical_json(value):
        raise CampaignError(f"{name} is not canonical JSON.")
    return value


def _rss_bytes(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _terminate_group(process: subprocess.Popen[Any]) -> bool:
    """TERM a process group, then KILL after the fixed grace period."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return False
    deadline = time.monotonic() + TERM_GRACE_SECONDS
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        return True
    return False


def _supervised_subprocess(
    command: tuple[str, ...],
    *,
    repo: Path,
    stdout_path: Path,
    stderr_path: Path,
    wall_limit_seconds: int | None = None,
    rss_limit_bytes: int | None = None,
    launch_environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    selected_wall_limit = (
        WALL_LIMIT_SECONDS if wall_limit_seconds is None else wall_limit_seconds
    )
    selected_rss_limit = (
        RSS_LIMIT_BYTES if rss_limit_bytes is None else rss_limit_bytes
    )
    for name, value in {
        "wall_limit_seconds": selected_wall_limit,
        "rss_limit_bytes": selected_rss_limit,
    }.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if launch_environment is None:
        environment = os.environ.copy()
        environment.update(phase_c.THREAD_LIMIT_ENV)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = str((repo / "src").resolve())
    else:
        from negfpy.anharmonic._launch_environment import qualification_environment
        supplied = dict(launch_environment)
        marker = supplied.pop("NEGFPY_QUALIFICATION_RUNTIME_SHA256", None)
        environment = qualification_environment({
            "executable": command[0], "launch_environment": supplied,
            "runtime_sha256": marker,
        }, repo=repo)
        if environment != dict(launch_environment):
            raise ValueError("SCBA launch environment differs from qualified map")
    start = time.monotonic()
    peak_rss = 0
    trigger: str | None = None
    killed_after_grace = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        while process.poll() is None:
            elapsed = time.monotonic() - start
            rss = _rss_bytes(process.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
            if rss is not None and rss > selected_rss_limit:
                trigger = "rss-limit"
            elif elapsed > selected_wall_limit:
                trigger = "wall-timeout"
            if trigger is not None:
                killed_after_grace = _terminate_group(process)
                break
            time.sleep(RSS_POLL_SECONDS)
        returncode = process.wait()
    return {
        "returncode": returncode,
        "elapsed_seconds": time.monotonic() - start,
        "peak_observed_rss_bytes": peak_rss,
        "watchdog_trigger": trigger,
        "killed_after_term_grace": killed_after_grace,
        "wall_limit_seconds": selected_wall_limit,
        "rss_limit_bytes": selected_rss_limit,
    }


def _load_outcome(
    *,
    output_json: Path,
    output_hdf5: Path,
    manifest_sha256: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    if output_json.exists() != output_hdf5.exists():
        raise CampaignError(
            "partial Phase-C scientific outcome is preserved; automatic rerun forbidden."
        )
    outcome = _load_canonical(output_json, name="Phase-C outcome")
    if (
        outcome.get("schema") != phase_c.SCHEMA
        or outcome.get("campaign_manifest_sha256") != manifest_sha256
    ):
        raise CampaignError("Phase-C outcome manifest binding is invalid.")
    archive = phase_c.verify_phase_c_hdf5(output_hdf5)
    if outcome.get("hdf5", {}).get("sha256") != archive["sha256"]:
        raise CampaignError("Phase-C JSON/HDF5 digest mismatch.")
    if bool(outcome.get("one_shot_born_executed")) != bool(
        archive["one_shot_born_executed"]
    ):
        raise CampaignError("Phase-C JSON/HDF5 execution mismatch.")
    if outcome.get("release_certified") or outcome.get("scba_executed"):
        raise CampaignError("Phase-C outcome overclaims SCBA or release status.")
    outcome_preflight = outcome.get("preflight")
    manifest_preflight = manifest.get("preflight")
    if not isinstance(outcome_preflight, Mapping) or not isinstance(
        manifest_preflight, Mapping
    ):
        raise CampaignError("Phase-C outcome lacks its sealed preflight identity.")
    if _stable_preflight_identity(outcome_preflight) != _stable_preflight_identity(
        manifest_preflight
    ):
        raise CampaignError("Phase-C outcome preflight differs from its manifest.")
    return outcome


def _write_checksums(artifacts: Path, paths: tuple[Path, ...]) -> bytes:
    content = "".join(
        f"{_sha256_file(path)}  {path.relative_to(artifacts).as_posix()}\n"
        for path in sorted(paths, key=lambda item: item.relative_to(artifacts).as_posix())
    ).encode("ascii")
    _ensure_exact(artifacts / "SHA256SUMS", content)
    return content


def _finalize_scientific_outcome(
    *,
    artifacts: Path,
    manifest: Mapping[str, Any],
    manifest_path: Path,
    running_path: Path,
    output_json: Path,
    output_hdf5: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Publish report/checksums/status without ever invoking the solver."""

    manifest_sha256 = _sha256_file(manifest_path)
    log = _load_canonical(log_path, name="Phase-C subprocess log")
    supervision = log.get("supervision")
    if not isinstance(supervision, Mapping):
        raise CampaignError("Phase-C subprocess log lacks supervision metadata.")
    if supervision.get("watchdog_trigger") is not None or supervision.get(
        "returncode"
    ) not in {0, 2}:
        raise CampaignError(
            "finalization-only scientific path cannot finalize an infrastructure failure."
        )
    outcome = _load_outcome(
        output_json=output_json,
        output_hdf5=output_hdf5,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
    )
    passed = bool(outcome["phase_c_gate_passed"])
    state = "passed" if passed else "scientific-failure"
    report = {
        "schema": REPORT_SCHEMA,
        "campaign_manifest_sha256": manifest_sha256,
        "state": state,
        "outcome_json_sha256": _sha256_file(output_json),
        "fields_hdf5_sha256": _sha256_file(output_hdf5),
        "log_sha256": _sha256_file(log_path),
        "supervision": dict(supervision),
        "phase_c_gate_passed": passed,
        "one_shot_born_executed": bool(outcome["one_shot_born_executed"]),
        "automatic_retry": False,
        "controls_changed": False,
        "scba_executed": False,
        "release_certified": False,
        "next_gate": (
            "design the checkpointed Nq=1 SCBA campaign"
            if passed
            else "inspect the archived fixed-control scientific failure"
        ),
    }
    report_path = artifacts / "phase-c-report.json"
    _ensure_exact(report_path, _canonical_json(report))
    sums = _write_checksums(
        artifacts,
        (manifest_path, running_path, output_json, output_hdf5, log_path, report_path),
    )
    status = {
        "schema": STATUS_SCHEMA,
        "campaign_manifest_sha256": manifest_sha256,
        "state": state,
        "phase_c_gate_passed": passed,
        "one_shot_born_executed": bool(outcome["one_shot_born_executed"]),
        "automatic_retry": False,
        "report_sha256": _sha256_file(report_path),
        "sha256sums_sha256": _sha256_bytes(sums),
        "scba_executed": False,
        "release_certified": False,
    }
    _ensure_exact(artifacts / "status.json", _canonical_json(status))
    return status


def finalize_existing_campaign(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    """Recover post-solve publication only; this function cannot run science."""

    repo = repo.resolve()
    artifacts = artifacts.resolve()
    manifest_path = artifacts / "campaign-manifest.json"
    running_path = artifacts / "status-running.json"
    status_path = artifacts / "status.json"
    if status_path.exists():
        return _load_canonical(status_path, name="terminal Phase-C status")
    manifest = _load_canonical(manifest_path, name="Phase-C campaign manifest")
    if manifest.get("schema") != SCHEMA:
        raise CampaignError("Phase-C manifest schema mismatch.")
    if manifest.get("source") != _source_identity(repo, expected_commit):
        raise CampaignError("Phase-C source identity changed before finalization.")
    if manifest.get("environment_lock") != _validate_lock(environment_lock):
        raise CampaignError("Phase-C environment lock changed before finalization.")
    driver = repo / "scripts" / "validate_sparse_nq1_phase_c_born.py"
    if manifest.get("driver", {}).get("sha256") != _sha256_file(driver):
        raise CampaignError("Phase-C driver changed before finalization.")
    if manifest.get("distributions") != _versions():
        raise CampaignError("Phase-C installed environment changed before finalization.")
    current, _problem, _config = phase_c.authenticate_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
    )
    if _stable_preflight_identity(current) != _stable_preflight_identity(
        manifest["preflight"]
    ):
        raise CampaignError("Phase-C scientific inputs changed before finalization.")
    running = _load_canonical(running_path, name="Phase-C running status")
    if running.get("campaign_manifest_sha256") != _sha256_file(manifest_path):
        raise CampaignError("Phase-C running status is not bound to its manifest.")
    return _finalize_scientific_outcome(
        artifacts=artifacts,
        manifest=manifest,
        manifest_path=manifest_path,
        running_path=running_path,
        output_json=artifacts / "outcomes" / "phase-c-born.json",
        output_hdf5=artifacts / "outcomes" / "phase-c-born-fields.h5",
        log_path=artifacts / "logs" / "phase-c-born.json",
    )


def _terminal_infrastructure_failure(
    *,
    artifacts: Path,
    manifest_path: Path,
    manifest_sha256: str,
    log_path: Path,
    supervision: Mapping[str, Any],
    detail: str,
) -> dict[str, Any]:
    failure = {
        "schema": "negfpy.sparse-nq1-phase-c-infrastructure-failure.v1",
        "campaign_manifest_sha256": manifest_sha256,
        "classification": "infrastructure-failure",
        "detail": detail,
        "supervision": dict(supervision),
        "automatic_retry": False,
        "controls_changed": False,
        "scba_executed": False,
        "release_certified": False,
    }
    failure_path = artifacts / "infrastructure-failure.json"
    _publish_once(failure_path, _canonical_json(failure))
    sums = _write_checksums(
        artifacts, (manifest_path, log_path, failure_path)
    )
    status = {
        "schema": STATUS_SCHEMA,
        "campaign_manifest_sha256": manifest_sha256,
        "state": "infrastructure-failure",
        "automatic_retry": False,
        "failure_sha256": _sha256_file(failure_path),
        "sha256sums_sha256": _sha256_bytes(sums),
        "phase_c_gate_passed": False,
        "scba_executed": False,
        "release_certified": False,
    }
    _publish_once(artifacts / "status.json", _canonical_json(status))
    return status


def run_campaign(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    fixture = fixture.resolve()
    fc3_full = fc3_full.resolve()
    phase_b_artifacts = phase_b_artifacts.resolve()
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest_path = artifacts / "campaign-manifest.json"
    status_path = artifacts / "status.json"
    running_path = artifacts / "status-running.json"
    output_json = artifacts / "outcomes" / "phase-c-born.json"
    output_hdf5 = artifacts / "outcomes" / "phase-c-born-fields.h5"
    log_path = artifacts / "logs" / "phase-c-born.json"

    if manifest_path.exists():
        if not status_path.exists():
            raise CampaignError(
                "an incomplete Phase-C attempt is preserved; automatic rerun forbidden."
            )
        return _load_canonical(status_path, name="terminal Phase-C status")
    unexpected = [path for path in artifacts.rglob("*") if path.is_file()]
    if unexpected:
        raise CampaignError(
            "artifact directory is nonempty without a manifest; refusing overwrite: "
            f"{unexpected!r}"
        )
    manifest = build_manifest(
        repo=repo,
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
        artifacts=artifacts,
        expected_commit=expected_commit,
        environment_lock=environment_lock,
    )
    _publish_once(manifest_path, _canonical_json(manifest))
    manifest_sha256 = _sha256_file(manifest_path)
    _publish_once(
        running_path,
        _canonical_json(
            {
                "schema": STATUS_SCHEMA,
                "campaign_manifest_sha256": manifest_sha256,
                "state": "running",
                "automatic_retry": False,
                "scba_executed": False,
                "release_certified": False,
            }
        ),
    )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor_out, temporary_out_name = tempfile.mkstemp(
        prefix=".phase-c-stdout.", dir=log_path.parent
    )
    descriptor_err, temporary_err_name = tempfile.mkstemp(
        prefix=".phase-c-stderr.", dir=log_path.parent
    )
    os.close(descriptor_out)
    os.close(descriptor_err)
    temporary_out = Path(temporary_out_name)
    temporary_err = Path(temporary_err_name)
    command = (
        sys.executable,
        str(repo / "scripts" / "validate_sparse_nq1_phase_c_born.py"),
        "--fixture",
        str(fixture),
        "--fc3-full",
        str(fc3_full),
        "--phase-b-artifacts",
        str(phase_b_artifacts),
        "--output-json",
        str(output_json),
        "--output-hdf5",
        str(output_hdf5),
        "--campaign-manifest-sha256",
        manifest_sha256,
    )
    try:
        supervision = _supervised_subprocess(
            command,
            repo=repo,
            stdout_path=temporary_out,
            stderr_path=temporary_err,
        )
        log = {
            "command": list(command),
            "supervision": supervision,
            "stdout": temporary_out.read_text(encoding="utf-8", errors="replace"),
            "stderr": temporary_err.read_text(encoding="utf-8", errors="replace"),
        }
        _publish_once(log_path, _canonical_json(log))
    finally:
        temporary_out.unlink(missing_ok=True)
        temporary_err.unlink(missing_ok=True)

    if supervision["watchdog_trigger"] is not None:
        return _terminal_infrastructure_failure(
            artifacts=artifacts,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            log_path=log_path,
            supervision=supervision,
            detail=f"watchdog triggered: {supervision['watchdog_trigger']}",
        )
    if supervision["returncode"] not in {0, 2}:
        detail = f"scientific driver exited with infrastructure code {supervision['returncode']}"
        if output_json.exists() or output_hdf5.exists():
            detail += "; partial output is preserved"
        return _terminal_infrastructure_failure(
            artifacts=artifacts,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            log_path=log_path,
            supervision=supervision,
            detail=detail,
        )
    return _finalize_scientific_outcome(
        artifacts=artifacts,
        manifest=manifest,
        manifest_path=manifest_path,
        running_path=running_path,
        output_json=output_json,
        output_hdf5=output_hdf5,
        log_path=log_path,
    )


def preflight_campaign(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    if artifacts.resolve().exists() and any(artifacts.resolve().iterdir()):
        raise CampaignError("preflight target artifact directory is not empty.")
    return build_manifest(
        repo=repo,
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
        artifacts=artifacts,
        expected_commit=expected_commit,
        environment_lock=environment_lock,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight_only and arguments.finalize_existing:
        parser.error("--preflight-only and --finalize-existing are mutually exclusive.")
    operation = (
        preflight_campaign
        if arguments.preflight_only
        else (finalize_existing_campaign if arguments.finalize_existing else run_campaign)
    )
    try:
        result = operation(
            repo=arguments.repo_root,
            fixture=arguments.fixture,
            fc3_full=arguments.fc3_full,
            phase_b_artifacts=arguments.phase_b_artifacts,
            artifacts=arguments.artifacts,
            expected_commit=arguments.expected_commit,
            environment_lock=arguments.environment_lock,
        )
    except (CampaignError, phase_c.PhaseCError, OSError, MemoryError) as error:
        print(f"phase-c campaign infrastructure failure: {error}", file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    if arguments.preflight_only:
        return 0
    terminal_state = result.get("state")
    return (
        0
        if terminal_state == "passed"
        else (2 if terminal_state == "scientific-failure" else 3)
    )


if __name__ == "__main__":
    raise SystemExit(main())
