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
    source_tree_sha256 = package_provenance.get("source_tree_sha256")
    if (
        not isinstance(source_tree_sha256, str)
        or len(source_tree_sha256) != 64
    ):
        raise core.RemoteValidationError(
            "could not compute the deployed NEGFPy source-tree SHA256"
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


def _checkpoint_index(
    *,
    artifacts: pathlib.Path,
    multidimensional: Mapping[str, Any],
) -> list[dict[str, Any]]:
    checkpoints = (artifacts / "checkpoints").resolve()
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

    checkpoint_records = _checkpoint_index(
        artifacts=artifacts,
        multidimensional=multidimensional,
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
        if path.is_file() and path.name != "SHA256SUMS":
            files.append(relative)
    return files


def _worker(*, deployment_dir: pathlib.Path, commit: str) -> int:
    """Run one attempt and publish only strictly validated terminal evidence."""

    root = deployment_dir.resolve()
    commit = core._validate_commit(commit)
    artifacts = root / "artifacts"
    checkpoints = artifacts / "checkpoints"
    attempts = artifacts / "attempts"
    lock_path = artifacts / "job.lock"
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
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
    report_facts: dict[str, Any] | None = None
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
            }
        )
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"argv={shlex.join(argv)}\n")
            for key in sorted(core.THREAD_LIMIT_ENV):
                log.write(f"{key}=1\n")
            log.flush()
            os.fsync(log.fileno())
            try:
                completed = subprocess.run(
                    argv,
                    cwd=root / "worktree",
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    text=True,
                )
            finally:
                # The runner may create this sidecar immediately before an
                # exception; never leave it at the shared artifact root.
                _collect_attempt_sidecars(
                    artifacts=artifacts,
                    attempt_root=attempt_root,
                )
        returncode = int(completed.returncode)
        if returncode == 0:
            report_facts, checkpoint_records = _validate_production_report(
                path=report_path,
                artifacts=artifacts,
                commit=commit,
                identity=stored_identity,
            )
            core._atomic_write_json(
                checkpoint_index_path,
                {
                    "schema": "negfpy.remote_validation.checkpoint_index.v1",
                    "attempt": attempt,
                    "checkpoint_root": "checkpoints",
                    "count": len(checkpoint_records),
                    "records": checkpoint_records,
                },
            )
            core._atomic_write_json(report_facts_path, report_facts)
            state = "completed"
        else:
            state = "failed"
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
