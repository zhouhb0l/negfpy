from __future__ import annotations

import importlib
import base64
import hashlib
import json
import os
import pathlib
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import _mpi_tasks as task_mpi
from negfpy.anharmonic.toy import _mpi_production_gate as production_mpi
from negfpy.anharmonic.toy._mpi_production_gate import (
    ProductionToyCaseResult,
    ProductionToyMPIExecutionReport,
)
from negfpy.anharmonic.toy._mpi_production_gate_physics import (
    production_toy_physics_sha256,
)
from negfpy.anharmonic.toy.multidimensional_release import (
    _refinement_points,
)


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
core = importlib.import_module("remote_heavy_validation")
operations = importlib.import_module("remote_heavy_runtime")
explicit = importlib.import_module("remote_heavy_explicit")
cli = importlib.import_module("remote_heavy_cli")
lock_validation = importlib.import_module("remote_heavy_lock")
identity = importlib.import_module("remote_heavy_identity")
security = importlib.import_module("remote_heavy_security")
mpi_scaling = importlib.import_module("remote_mpi_scaling")
toy_mpi = importlib.import_module("remote_toy_mpi_runner")

COMMIT = "0123456789abcdef0123456789abcdef01234567"
_REAL_SUBPROCESS_RUN = subprocess.run
GIT_TREE = "a" * 40
SOURCE_SHA256 = "b" * 64


def _valid_ssh_config() -> str:
    return "\n".join(
        (
            "host negf-heavy",
            "hostname 172.20.86.67",
            "user hangbo",
            "identityfile ~/.ssh/id_ed25519_aspire2a",
            f"userknownhostsfile {core.EXPECTED_KNOWN_HOSTS_FILE}",
            "identitiesonly yes",
            "batchmode yes",
            "preferredauthentications publickey",
            "pubkeyauthentication yes",
            "passwordauthentication no",
            "kbdinteractiveauthentication no",
            "stricthostkeychecking yes",
            "serveraliveinterval 30",
            "serveralivecountmax 3",
        )
    )


def test_all_network_commands_force_key_only_batch_mode() -> None:
    for argv in (
        core.ssh_argv("true"),
        core.scp_argv("source", "negf-heavy:target"),
    ):
        joined = " ".join(argv)
        assert "-oBatchMode=yes" in argv
        assert "-oStrictHostKeyChecking=yes" in argv
        assert "-oIdentitiesOnly=yes" in argv
        assert f"-oIdentityFile={core.EXPECTED_IDENTITY_FILE}" in argv
        assert (
            f"-oUserKnownHostsFile={core.EXPECTED_KNOWN_HOSTS_FILE}" in argv
        )
        assert "-oGlobalKnownHostsFile=/dev/null" in argv
        assert "-oHostKeyAlgorithms=ssh-ed25519" in argv
        assert "-oPreferredAuthentications=publickey" in argv
        assert "-oPasswordAuthentication=no" in argv
        assert "-oKbdInteractiveAuthentication=no" in argv
        assert "-oGSSAPIAuthentication=no" in argv
        assert "-oHostbasedAuthentication=no" in argv
        assert f"-oHostName={core.EXPECTED_HOST}" in argv
        assert f"-oUser={core.EXPECTED_USER}" in argv
        assert core.SSH_ALIAS in joined
        assert "sshpass" not in joined.lower()


def test_alias_validation_locks_host_user_identity_and_keepalive() -> None:
    report = core.validate_ssh_alias_config(_valid_ssh_config())
    assert report["host"] == "172.20.86.67"
    assert report["user"] == "hangbo"
    assert report["identity_basename"] == "id_ed25519_aspire2a"
    assert report["strict_host_key_checking"] is True
    assert report["expected_ed25519_fingerprint"].startswith("SHA256:")
    canonical = _valid_ssh_config().replace("stricthostkeychecking yes", "stricthostkeychecking true")
    assert core.validate_ssh_alias_config(canonical)["strict_host_key_checking"]


    with pytest.raises(core.RemoteValidationError, match="HostName"):
        core.validate_ssh_alias_config(
            _valid_ssh_config().replace("172.20.86.67", "example.invalid")
        )
    with pytest.raises(core.RemoteValidationError, match="IdentitiesOnly"):
        core.validate_ssh_alias_config(
            _valid_ssh_config().replace("identitiesonly yes", "identitiesonly no")
        )
    with pytest.raises(core.RemoteValidationError, match="BatchMode"):
        core.validate_ssh_alias_config(
            _valid_ssh_config().replace("batchmode yes", "batchmode no")
        )
    with pytest.raises(core.RemoteValidationError, match="PreferredAuthentications"):
        core.validate_ssh_alias_config(
            _valid_ssh_config().replace(
                "preferredauthentications publickey",
                "preferredauthentications publickey,password",
            )
        )


def test_alias_rejects_same_identity_basename_at_another_path() -> None:
    config = _valid_ssh_config().replace(
        "~/.ssh/id_ed25519_aspire2a",
        "/tmp/id_ed25519_aspire2a",
    )
    with pytest.raises(core.RemoteValidationError, match="IdentityFile set"):
        core.validate_ssh_alias_config(config)

def test_dedicated_known_hosts_file_must_contain_exact_ed25519_fingerprint(
    tmp_path: pathlib.Path,
) -> None:
    blob = b"synthetic-ed25519-public-key-blob"
    encoded = base64.b64encode(blob).decode("ascii")
    expected = "SHA256:" + base64.b64encode(
        hashlib.sha256(blob).digest()
    ).decode("ascii").rstrip("=")
    known_hosts = tmp_path / "known_hosts_negf_heavy"
    known_hosts.write_text(
        f"172.20.86.67 ssh-ed25519 {encoded}\n", encoding="utf-8"
    )
    known_hosts.chmod(0o644)
    config = _valid_ssh_config().replace(
        str(core.EXPECTED_KNOWN_HOSTS_FILE), str(known_hosts)
    )
    assert security.validate_known_hosts_fingerprint(
        config, expected=expected,
        expected_host="172.20.86.67",
        expected_path=known_hosts,
    ) == known_hosts
    with pytest.raises(ValueError, match="exactly one pinned"):
        security.validate_known_hosts_fingerprint(
            config, expected=core.EXPECTED_ED25519_FINGERPRINT,
            expected_host="172.20.86.67",
            expected_path=known_hosts,
        )

    second = base64.b64encode(b"second-ed25519-key").decode("ascii")
    known_hosts.write_text(
        known_hosts.read_text(encoding="utf-8")
        + f"172.20.86.67 ssh-ed25519 {second}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one pinned"):
        security.validate_known_hosts_fingerprint(
            config, expected=expected, expected_host="172.20.86.67",
            expected_path=known_hosts,
        )


def test_dedicated_known_hosts_rejects_other_algorithms_and_writable_file(
    tmp_path: pathlib.Path,
) -> None:
    blob = base64.b64encode(b"synthetic-key").decode("ascii")
    known_hosts = tmp_path / "known_hosts_negf_heavy"
    config = _valid_ssh_config().replace(
        str(core.EXPECTED_KNOWN_HOSTS_FILE), str(known_hosts)
    )
    known_hosts.write_text(
        f"172.20.86.67 ssh-rsa {blob}\n", encoding="utf-8"
    )
    known_hosts.chmod(0o644)
    with pytest.raises(ValueError, match="ssh-ed25519 exclusively"):
        security.validate_known_hosts_fingerprint(
            config,
            expected="SHA256:" + "a" * 43,
            expected_host=core.EXPECTED_HOST,
            expected_path=known_hosts,
        )
    known_hosts.chmod(0o666)
    with pytest.raises(ValueError, match="group/other writable"):
        security.validate_known_hosts_fingerprint(
            config,
            expected="SHA256:" + "a" * 43,
            expected_host=core.EXPECTED_HOST,
            expected_path=known_hosts,
        )



def test_explicit_lock_accepts_standard_comment_header(tmp_path: pathlib.Path) -> None:
    lock = tmp_path / "environment.lock"
    lock.write_text(
        "# platform: linux-64\n# generated by micromamba\n# pinned\n\n"
        "@EXPLICIT\nhttps://example.invalid/package.conda#deadbeef\n",
        encoding="utf-8",
    )
    digest = core._sha256(lock)
    assert lock_validation.validate_explicit_lock(
        lock, expected_sha256=digest
    ) == digest


def test_committed_lock_pins_one_openmpi_and_matching_mpi4py() -> None:
    lock = pathlib.Path(__file__).parents[1] / explicit.DEFAULT_LOCK
    text = lock.read_text(encoding="utf-8")
    assert core._sha256(lock) == core.EXPECTED_ENV_LOCK_SHA256
    assert text.count("/openmpi-5.0.10-") == 1
    assert text.count("/mpi4py-4.1.1-") == 1
    assert "/mpi-1.0.1-openmpi." in text
    assert "mpich" not in text.lower()


def test_remote_machine_probe_is_valid_and_complete_python() -> None:
    compile(explicit._MACHINE_PROGRAM, "<remote-machine-probe>", "exec")
    for field in (
        "affinity_logical_cpu_count",
        "physical_core_count",
        "socket_count",
        "cpu_model", "memory_total_bytes", "memory_available_bytes",
        "home_disk_total_bytes", "home_disk_free_bytes",
    ):
        assert field in explicit._MACHINE_PROGRAM


def test_local_cpu_topology_is_positive_and_not_oversubscribed() -> None:
    topology = core.machine_cpu_topology()
    assert topology["affinity_logical_cpu_count"] >= 1
    assert topology["physical_core_count"] >= 1
    assert topology["socket_count"] >= 1
    assert topology["physical_core_count"] <= topology["affinity_logical_cpu_count"]


def test_commit_and_session_inputs_cannot_inject_remote_commands() -> None:
    assert core._validate_commit(COMMIT) == COMMIT
    assert core._validate_session("negf-toy-0123") == "negf-toy-0123"
    with pytest.raises(core.RemoteValidationError):
        core._validate_commit(COMMIT + "; touch owned")
    with pytest.raises(core.RemoteValidationError):
        core._validate_session("gate; touch owned")


def test_checksum_round_trip_detects_tamper_and_traversal(
    tmp_path: pathlib.Path,
) -> None:
    payload = tmp_path / "artifact.json"
    payload.write_text('{"ok": true}\n', encoding="utf-8")
    core._write_sha256sums(tmp_path, (pathlib.PurePath(payload.name),))
    assert core.verify_sha256sums(tmp_path) == ("artifact.json",)
    payload.write_text('{"ok": false}\n', encoding="utf-8")
    with pytest.raises(core.RemoteValidationError, match="SHA256 mismatch"):
        core.verify_sha256sums(tmp_path)

    (tmp_path / "SHA256SUMS").write_text(
        f"{'0' * 64}  ../escape\n", encoding="utf-8"
    )
    with pytest.raises(core.RemoteValidationError, match="unsafe checksum"):
        core.verify_sha256sums(tmp_path)


def test_environment_prefix_manifest_detects_dependency_mutation_and_unsafe_symlink(
    tmp_path: pathlib.Path,
) -> None:
    prefix = tmp_path / "environment"
    library = prefix / "lib" / "dependency.bin"
    library.parent.mkdir(parents=True)
    library.write_bytes(b"dependency-v1\n")
    executable = prefix / "bin" / "dependency-link"
    executable.parent.mkdir()
    executable.symlink_to("../lib/dependency.bin")

    before = cli._environment_prefix_manifest(prefix)
    assert before["regular_file_count"] == 1
    assert before["symlink_count"] == 1
    assert before["symlinks"][0]["resolved_path"] == "lib/dependency.bin"
    library.write_bytes(b"dependency-v2\n")
    after = cli._environment_prefix_manifest(prefix)
    assert after["manifest_sha256"] != before["manifest_sha256"]
    assert after["regular_files"][0]["sha256"] != before["regular_files"][0]["sha256"]

    executable.unlink()
    executable.symlink_to("/etc/passwd")
    with pytest.raises(core.RemoteValidationError, match="absolute symlink"):
        cli._environment_prefix_manifest(prefix)


def test_artifact_inventory_rejects_nested_sha256sums(
    tmp_path: pathlib.Path,
) -> None:
    artifacts = tmp_path / "artifacts"
    nested = artifacts / "nested"
    nested.mkdir(parents=True)
    (nested / "SHA256SUMS").write_text("nested\n", encoding="ascii")
    with pytest.raises(core.RemoteValidationError, match="nested SHA256SUMS"):
        cli._artifact_files(artifacts)


def test_prepare_dry_run_is_nonmutating_and_uses_exact_commit_path(
    tmp_path: pathlib.Path,
) -> None:
    runner = core.CommandRunner(dry_run=True)
    staging = explicit.prepare(
        repo_root=tmp_path,
        staging_root=tmp_path / "deployments",
        environment_lock=pathlib.Path("validation/locks/negf-toy-gate-linux-64.lock"),
        runner=runner,
    )
    assert staging.name == "0" * 40
    assert not staging.exists()
    assert runner.commands[0][:3] == ("git", "status", "--porcelain=v1")
    assert runner.commands[1] == ("git", "rev-parse", "--verify", "HEAD")


def test_deploy_launch_status_and_fetch_dry_runs_use_only_alias(
    tmp_path: pathlib.Path,
) -> None:
    staging = tmp_path / COMMIT
    runner = core.CommandRunner(dry_run=True)
    assert cli.deploy(staging=staging, runner=runner) == COMMIT
    cli.launch(commit=COMMIT, session=None, runner=runner)
    operations.status(commit=COMMIT, session=None, runner=runner)
    destination = operations.fetch(
        commit=COMMIT,
        destination_root=tmp_path / "fetched",
        runner=runner,
    )
    assert destination.name == COMMIT
    assert not destination.exists()
    network = [
        argv
        for argv in runner.commands
        if argv and argv[0] in {"ssh", "scp"}
    ]
    assert network
    for argv in network:
        joined = " ".join(argv)
        assert core.SSH_ALIAS in joined
        if argv[:2] == ("ssh", "-G"):
            continue
        assert "-oBatchMode=yes" in argv
        assert f"-oHostName={core.EXPECTED_HOST}" in argv
        assert f"-oUser={core.EXPECTED_USER}" in argv
    launch_command = "\n".join(" ".join(argv) for argv in network)
    for name in core.THREAD_LIMIT_ENV:
        assert f"{name}=1" in launch_command
    assert "/environment/bin/python" in launch_command
    assert core.REMOTE_MICROMAMBA in launch_command
    assert "/home/hangbo/micromamba/envs/negf-tools/bin/tmux" in launch_command


def _synthetic_source_identity() -> dict[str, object]:
    records: list[dict[str, object]] = []
    for relative in cli.ORCHESTRATION_SCRIPT_PATHS:
        source = pathlib.Path(__file__).parents[1] / relative
        records.append(
            {
                "path": relative,
                "git_tracked": True,
                "kind": "regular_file",
                "size": source.stat().st_size,
                "mode": source.stat().st_mode & 0o777,
                "sha256": core._sha256(source),
            }
        )
    tracked = cli._canonical_manifest(
        {"schema": "negfpy.remote_validation.tracked_worktree.v1", "records": records}
    )
    worktree = cli._canonical_manifest(
        {"schema": "negfpy.remote_validation.full_worktree.v1", "records": records}
    )
    orchestration = cli._canonical_manifest(
        {"schema": "negfpy.remote_validation.orchestration_sources.v1", "records": records}
    )
    return cli._canonical_manifest(
        {
            "schema": "negfpy.remote_validation.execution_source.v1",
            "git_revision": COMMIT,
            "git_tree": GIT_TREE,
            "package_source_sha256": SOURCE_SHA256,
            "tracked_file_count": len(records),
            "tracked_files_manifest_sha256": tracked["manifest_sha256"],
            "tracked_files": records,
            "worktree_file_count": len(records),
            "worktree_files_manifest_sha256": worktree["manifest_sha256"],
            "worktree_files": records,
            "orchestration_manifest_sha256": orchestration["manifest_sha256"],
            "orchestration_scripts": records,
        }
    )


def _synthetic_prefix_manifest() -> dict[str, object]:
    return cli._canonical_manifest(
        {
            "schema": "negfpy.remote_validation.environment_prefix.v1",
            "prefix": "/validation/environment",
            "directory_count": 1,
            "regular_file_count": 0,
            "symlink_count": 0,
            "regular_files": [],
            "symlinks": [],
        }
    )


def _worker_identity(lock_sha: str) -> dict[str, object]:
    return {
        "schema": "negfpy.remote_validation.remote_environment.v1",
        "commit": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
        "execution_source_identity": _synthetic_source_identity(),
        "environment_prefix_manifest": _synthetic_prefix_manifest(),
        "explicit_lock_sha256": lock_sha,
        "tree": GIT_TREE,
        "hostname": "validation-host",
        "platform": "Linux-test",
        "machine": "x86_64",
        "affinity_logical_cpu_count": 152,
        "physical_core_count": 76,
        "socket_count": 2,
        "python_executable": "/validation/environment/bin/python",
        "python_version": "3.10.0",
        "distributions": {"numpy": "1.0", "negfpy": "0.1"},
        "mpi": {
            "vendor": "Open MPI",
            "version": [5, 0, 10],
            "library_version": "Open MPI v5.0.10",
            "launcher": "/validation/environment/bin/mpiexec",
            "launcher_target": "/validation/environment/bin/mpirun",
            "launcher_sha256": "c" * 64,
            "launcher_version": "mpiexec (Open MPI) 5.0.10",
        },
        "package_source": "/validation/worktree/src/negfpy/__init__.py",
        "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
    }


def _prepare_worker_root(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str]:
    root = tmp_path / COMMIT
    (root / "worktree").mkdir(parents=True)
    runner = root / "worktree" / "scripts" / "remote_toy_mpi_runner.py"
    runner.parent.mkdir()
    shutil.copy2(pathlib.Path(toy_mpi.__file__).resolve(), runner)
    incoming = root / "incoming"
    incoming.mkdir(parents=True)
    lock = incoming / "environment-explicit.lock"
    lock.write_text("@EXPLICIT\nsynthetic\n", encoding="utf-8")
    lock_sha = core._sha256(lock)
    core._atomic_write_json(
        incoming / "deployment-manifest.json",
        {
            "schema": "negfpy.remote_validation.deployment.v2",
            "source": {"commit": COMMIT, "tree": GIT_TREE, "dirty": False},
            "environment": {"explicit_lock_sha256": lock_sha},
        },
    )
    core._atomic_write_json(
        incoming / "environment-manifest.json",
        {
            "schema": "negfpy.remote_validation.environment.v2",
            "explicit_lock_sha256": lock_sha,
        },
    )
    (incoming / "negfpy.bundle").write_bytes(b"synthetic git bundle\n")
    core._write_sha256sums(
        incoming,
        tuple(
            pathlib.PurePath(name)
            for name in (
                "deployment-manifest.json",
                "environment-explicit.lock",
                "environment-manifest.json",
                "negfpy.bundle",
            )
        ),
    )
    artifacts = root / "artifacts"
    (artifacts / "job.lock").mkdir(parents=True)
    (artifacts / "job.lock" / "session").write_text(
        "test-session\n", encoding="utf-8"
    )
    core._atomic_write_json(
        artifacts / "identity" / "remote-environment.json",
        _worker_identity(lock_sha),
    )
    return root, lock_sha


def _report_digest(document: dict[str, object], *, ensure_ascii: bool) -> str:
    return hashlib.sha256(
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=ensure_ascii,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def _write_valid_worker_report(
    *,
    report_path: pathlib.Path,
    root: pathlib.Path,
    lock_sha: str,
    checkpoint_root: pathlib.Path | None = None,
    orchestration: dict[str, object] | None = None,
) -> dict[str, object]:
    artifacts = root / "artifacts"
    checkpoints = artifacts / "checkpoints" if checkpoint_root is None else checkpoint_root
    checkpoints.mkdir(parents=True, exist_ok=True)
    identity = {
        "git_revision": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
    }
    provenance = {"identity": identity, "complete": True, "clean": True}
    outcomes: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    for index in range(13):
        run_id = f"run-{index:02d}"
        checkpoint = checkpoints / f"{run_id}.h5"
        checkpoint.write_bytes(f"checkpoint {run_id}\n".encode("ascii"))
        checkpoint_sha = core._sha256(checkpoint)
        checkpoint_records.append(
            {
                "run_id": run_id,
                "path": str(checkpoint.resolve()),
                "file_sha256": checkpoint_sha,
                "actual_file_sha256": checkpoint_sha,
                "environment_sha256": lock_sha,
                "valid": True,
            }
        )
        outcomes.append(
            {
                "run_id": run_id,
                "backend": "fft",
                "status": "converged",
                "result": {
                    "method": "scba",
                    "converged": True,
                    "diagnostics_passed": True,
                    "provenance": provenance,
                },
            }
        )

    harmonic_points = [
        {
            "label": "base" if index == 0 else f"refinement-{index}",
            "provenance": provenance,
        }
        for index in range(7)
    ]
    backend_status = {
        "serial": {
            "method": "scba",
            "converged": True,
            "diagnostics_passed": True,
            "effective_backend": "serial",
            "hilbert_backend": "direct",
            "environment_sha256": lock_sha,
            "provenance": provenance,
        },
        "threaded": {
            "method": "scba",
            "converged": True,
            "diagnostics_passed": True,
            "effective_backend": "threaded",
            "hilbert_backend": "direct",
            "environment_sha256": lock_sha,
            "provenance": provenance,
        },
        "fft": {
            "method": "scba",
            "converged": True,
            "diagnostics_passed": True,
            "effective_backend": "fft",
            "hilbert_backend": "fft",
            "environment_sha256": lock_sha,
            "provenance": provenance,
        },
    }
    multidimensional: dict[str, object] = {
        "schema": "negfpy.anharmonic.multidimensional_dense_toy_release",
        "schema_version": 4,
        "protocol": "multidimensional_toy_gate_v2",
        "protocol_version": 2,
        "config": {
            "protocol": "multidimensional_toy_gate_v2",
            "protocol_version": 2,
            "profile": "full_preregistered",
            "certification_eligible": True,
            "full_physical_ladder_selected": True,
            "run_preregistered_refinements": True,
            "run_bounded_backend_validation": True,
            "solver": {
                "production_backend": "fft",
                "production_hilbert_backend": "fft",
                "surface_gf_method": "analytic_diagonal",
            },
        },
        "release_certified": True,
        "execution_passed": True,
        "coarse": False,
        "topology_validation": {
            "passed": True,
            "full_release_count": True,
            "expected_outcome_count": 13,
            "observed_outcome_count": 13,
            "outcomes_exact": True,
        },
        "outcomes": outcomes,
        "harmonic_preflight": {
            "schema": "negfpy.anharmonic.harmonic_same_grid_preflight",
            "schema_version": 2,
            "protocol": "multidimensional_toy_gate_v2",
            "passed": True,
            "point_count": 7,
            "all_seven_archived": True,
            "required_directional_comparison_count": 4,
            "critical_comparisons_passed": True,
            "backend": "fft",
            "hilbert_backend": "fft",
            "surface_gf_method": "analytic_diagonal",
            "environment_sha256": lock_sha,
            "points": harmonic_points,
        },
        "backend_validation": {
            "schema": "negfpy.anharmonic.bounded_backend_validation",
            "schema_version": 1,
            "protocol": "multidimensional_toy_gate_v2",
            "scope": "bounded_converged_scba_backend_oracle",
            "passed": True,
            "full_grid_serial_threaded_comparison_executed": False,
            "backends_executed": ["serial", "threaded", "fft"],
            "environment_sha256": lock_sha,
            "backend_status": backend_status,
        },
        "provenance": {**provenance, "passed": True},
        "checkpoint_evidence": {
            "passed": True,
            "completed_outcome_count": 13,
            "total_outcome_count": 13,
            "environment_sha256": lock_sha,
            "records": checkpoint_records,
        },
    }
    multidimensional["report_sha256"] = _report_digest(
        multidimensional, ensure_ascii=True
    )
    report: dict[str, object] = {
        "schema": "negfpy.anharmonic.production_toy_gate",
        "schema_version": 4,
        "summary": {
            "release_certified": True,
            "oracle_gate_passed": True,
            "multidimensional_release_certified": True,
            "multidimensional_contract_passed": True,
            "multidimensional_static_configuration_passed": True,
            "multidimensional_runtime_evidence_passed": True,
            "multidimensional_underlying_evidence_complete": True,
            "cross_platform_identity_consistent": True,
            "environment_lock_consistent": True,
            "validation_tier": "release",
        },
        "environment_lock": {
            "passed": True,
            "open_environment_sha256": lock_sha,
            "multidimensional_environment_sha256": lock_sha,
        },
        "multidimensional_scba": multidimensional,
        "open_three_mode": {
            "runs": [
                {
                    "name": f"open-{index:02d}",
                    "environment_lock_sha256": lock_sha,
                    "provenance": provenance,
                }
                for index in range(13)
            ]
        },
    }
    if orchestration is not None:
        report["orchestration_provenance"] = orchestration
    report["report_sha256"] = _report_digest(report, ensure_ascii=False)
    core._atomic_write_json(report_path, report)
    return report


def _fake_mpi_environment(
    prefix: pathlib.Path,
) -> tuple[dict[str, str], dict[str, str]]:
    binary = prefix / "bin"
    binary.mkdir(parents=True, exist_ok=True)
    for name in ("mpiexec", "python"):
        path = binary / name
        if not path.exists():
            path.write_text(
                f"#!/bin/sh\n# synthetic {name}\nexit 0\n",
                encoding="ascii",
            )
            path.chmod(0o755)
    return (
        toy_mpi._launcher_record(binary / "mpiexec"),
        toy_mpi._launcher_record(binary / "python"),
    )


def _memory_preflight(profile: str, rank: int) -> dict[str, object]:
    return toy_mpi.campaign_dense_memory_preflight(
        profile=profile,
        rank_count=rank,
        available_memory_bytes=10**15,
    )


def _rank_runtime(
    rank_count: int,
    python_record: dict[str, str],
    *,
    profile: str,
    root: pathlib.Path,
    execution_id: str,
    environment_sha256: str,
    socket_count: int = 2,
) -> list[dict[str, object]]:
    by_core: dict[tuple[int, int], int] = {}
    for cpu_id in sorted(os.sched_getaffinity(0)):
        topology = toy_mpi._logical_cpu_topology(cpu_id)
        key = (
            topology["physical_package_id"],
            topology["core_id"],
        )
        by_core.setdefault(key, cpu_id)
    by_package: dict[int, list[tuple[int, int]]] = {}
    for package_id, core_id in sorted(by_core):
        by_package.setdefault(package_id, []).append(
            (core_id, by_core[(package_id, core_id)])
        )
    packages = sorted(by_package)
    assert len(packages) >= socket_count
    selected: list[tuple[int, int, int]] = []
    if rank_count >= socket_count and rank_count % socket_count == 0:
        per_socket = rank_count // socket_count
        for package_id in packages[:socket_count]:
            assert len(by_package[package_id]) >= per_socket
            selected.extend(
                (package_id, core_id, cpu_id)
                for core_id, cpu_id in by_package[package_id][:per_socket]
            )
    else:
        package_id = packages[0]
        selected.extend(
            (package_id, core_id, cpu_id)
            for core_id, cpu_id in by_package[package_id][:rank_count]
        )
    assert len(selected) == rank_count
    python_argv = toy_mpi._expected_rank_python_argv(
        profile=profile,
        root=root.resolve(),
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=environment_sha256,
        execution_id=execution_id,
        rank_count=rank_count,
    )
    records: list[dict[str, object]] = []
    for rank, (package_id, core_id, cpu_id) in enumerate(selected):
        records.append(
            {
                "rank": rank,
                "hostname": "validation-host",
                "logical_cpu_ids": [cpu_id],
                "logical_cpu_topology": [
                    {
                        "logical_cpu_id": cpu_id,
                        "physical_package_id": package_id,
                        "core_id": core_id,
                    }
                ],
                "physical_cores": [
                    {
                        "physical_package_id": package_id,
                        "core_id": core_id,
                    }
                ],
                "one_physical_core": True,
                "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
                "python": dict(python_record),
                "python_argv": python_argv,
            }
        )
    return records




def _harmonic_preflight_evidence(config) -> dict[str, object]:
    labels = ("base",) + tuple(label for label, _ in _refinement_points(config))
    assert len(labels) == 7
    points = [
        {
            "label": label,
            "passed": True,
            "field_sha256": {
                "green_retarded": hashlib.sha256(label.encode("ascii")).hexdigest()
            },
        }
        for label in labels
    ]
    return {
        "schema": "negfpy.anharmonic.harmonic_same_grid_preflight",
        "schema_version": 2,
        "protocol": "multidimensional_toy_gate_v2",
        "applicable": True,
        "passed": True,
        "scope": "base_plus_six_preregistered_numerical_points",
        "point_count": 7,
        "all_seven_archived": True,
        "points": points,
        "comparisons": [
            {"left_label": "base", "right_label": label, "passed": True}
            for label in labels[1:]
        ],
    }


def _write_canonical_worker_chain(
    *,
    root: pathlib.Path,
    profile: str,
    rank_count: int,
    execution_id: str,
    python_record: dict[str, str],
    environment_sha256: str,
    report_owner_root: pathlib.Path,
    report_mutator=None,
) -> tuple[pathlib.Path, str]:
    lock_sha = environment_sha256
    campaign, config = toy_mpi._expected_campaign(
        profile=profile,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
    )
    root.mkdir(parents=True, exist_ok=True)
    task_mpi.write_task_manifest(root / "campaign-manifest.json", campaign.manifest)
    runtime = _rank_runtime(
        rank_count,
        python_record,
        profile=profile,
        root=root,
        environment_sha256=lock_sha,
        execution_id=execution_id,
    )
    runtime_by_rank = {int(item["rank"]): item for item in runtime}
    validator = production_mpi._artifact_validator(campaign)
    phase_sha256: list[str] = []
    worker_ranks = (0,) if rank_count == 1 else tuple(range(1, rank_count))
    for wave_index, task_ids in enumerate(campaign.waves):
        phase = production_mpi._phase_manifest(campaign, wave_index)
        phase_root = root / "phases" / production_mpi._phase_slug(
            wave_index,
            task_ids,
        )
        artifact_root = phase_root / "task-artifacts"
        checkpoint_root = phase_root / "task-outcomes"
        task_mpi.write_task_manifest(phase_root / "task-manifest.json", phase)
        assignments: list[task_mpi.TaskAssignment] = []
        for task_index, task_id in enumerate(task_ids):
            task = phase.task(task_id)
            rank = worker_ranks[task_index % len(worker_ranks)]
            runtime_record = runtime_by_rank[rank]
            cpu_affinity = tuple(int(value) for value in runtime_record["logical_cpu_ids"])
            cpu_topology = tuple(
                (
                    int(item["logical_cpu_id"]),
                    int(item["physical_package_id"]),
                    int(item["core_id"]),
                )
                for item in runtime_record["logical_cpu_topology"]
            )
            cap = {
                "requested_limit": 1,
                "environment": {
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "BLIS_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                },
                "libraries": [],
                "cpu_affinity": list(cpu_affinity),
                "cpu_affinity_topology": [
                    {
                        "logical_cpu": logical_cpu,
                        "package_id": package_id,
                        "core_id": core_id,
                    }
                    for logical_cpu, package_id, core_id in cpu_topology
                ],
                "all_loaded_libraries_capped": True,
            }
            context = task_mpi.TaskContext(
                run_directory=phase_root,
                artifact_root=artifact_root,
                ordinal=phase.ordinal(task),
                task_identity_sha256=phase.task_identity_sha256(task),
                rank=rank,
                communicator_size=rank_count,
                hostname=str(runtime_record["hostname"]),
                thread_limit=1,
                thread_cap_metadata=cap,
            )
            evidence = (
                _harmonic_preflight_evidence(config)
                if task_id == "preflight.harmonic-same-grid"
                else {"task_id": task_id, "passed": True}
            )
            arrays = (
                {}
                if task_id == "preflight.harmonic-same-grid"
                else {
                    "synthetic_field": np.asarray(
                        [campaign.manifest.ordinal(task_id)],
                        dtype=np.float64,
                    )
                }
            )
            case_result = ProductionToyCaseResult.completed(
                evidence=evidence,
                arrays=arrays,
            )
            artifact = production_mpi._write_case_artifact(
                campaign,
                task,
                context,
                case_result,
            )
            task_result = task_mpi.TaskResult.completed(
                artifact,
                metadata={
                    "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
                    "campaign_task_identity_sha256": (
                        campaign.manifest.task_identity_sha256(task_id)
                    ),
                    "case_status": "completed",
                },
            )
            checkpoint = task_mpi._checkpoint_from_result(
                phase,
                task,
                task_result,
                context,
                elapsed_seconds=0.01,
                artifact_validator=validator,
            )
            checkpoint_path = task_mpi.task_checkpoint_path(
                phase,
                task,
                checkpoint_root,
            )
            task_mpi.write_task_checkpoint(checkpoint_path, checkpoint)
            assignments.append(
                task_mpi.TaskAssignment(
                    task_id=task_id,
                    status="completed",
                    rank=rank,
                    hostname=str(runtime_record["hostname"]),
                    elapsed_seconds=0.01,
                    resumed=False,
                    cpu_affinity=cpu_affinity,
                    thread_cap_passed=True,
                    checkpoint_relative_path=checkpoint_path.name,
                    checkpoint_file_sha256=core._sha256(checkpoint_path),
                    artifact_relative_path=checkpoint.artifact_relative_path,
                    artifact_sha256=checkpoint.artifact_sha256,
                    cpu_affinity_topology=cpu_topology,
                    origin_communicator_size=rank_count,
                )
            )
        phase_id = f"{execution_id}.phase-{wave_index:03d}"
        phase_report = task_mpi.MPIExecutionReport(
            execution_id=phase_id,
            manifest_sha256=phase.manifest_sha256,
            mode="serial" if rank_count == 1 else "mpi",
            communicator_size=rank_count,
            coordinator_hostname="validation-host",
            started_utc="2026-07-26T00:00:00Z",
            finished_utc="2026-07-26T00:00:01Z",
            elapsed_seconds=1.0,
            thread_limit=1,
            assignments=tuple(assignments),
            missing_task_ids=(),
            infrastructure_errors=(),
        )
        task_mpi.write_execution_manifest(
            phase_root / "executions" / f"{phase_id}.json",
            phase_report,
        )
        phase_sha256.append(phase_report.execution_sha256)

    outcomes = production_mpi.collect_production_toy_campaign_outcomes(
        campaign,
        root,
    )
    execution = ProductionToyMPIExecutionReport(
        execution_id=execution_id,
        campaign_manifest_sha256=campaign.campaign_manifest_sha256,
        mode="serial" if rank_count == 1 else "mpi",
        communicator_size=rank_count,
        coordinator_hostname="validation-host",
        started_utc="2026-07-26T00:00:00Z",
        finished_utc="2026-07-26T00:00:02Z",
        elapsed_seconds=2.0,
        phase_execution_sha256=tuple(phase_sha256),
        outcomes=outcomes,
    )
    execution_path = root / "campaign-executions" / f"{execution_id}.json"
    execution_path.parent.mkdir(parents=True, exist_ok=True)
    toy_mpi.write_hashed_manifest(execution_path, execution._payload())
    campaign_path = root / "campaign-manifest.json"
    campaign_document = toy_mpi.read_hashed_manifest(campaign_path)
    execution_document = toy_mpi.read_hashed_manifest(execution_path)
    report_path = root / "production-reports" / f"{execution_id}.json"
    orchestration = {
        "schema": "negfpy.task_level_mpi_provenance",
        "schema_version": 1,
        "campaign_manifest_sha256": campaign.campaign_manifest_sha256,
        "campaign_manifest_file_sha256": core._sha256(campaign_path),
        "execution_manifest_file_sha256": core._sha256(execution_path),
        "execution_payload_sha256": execution_document["payload_sha256"],
        "execution_scientific_sha256": execution.scientific_sha256,
        "execution_id": execution_id,
        "mode": execution.mode,
        "communicator_size": rank_count,
        "phase_execution_sha256": phase_sha256,
        "complete": True,
        "solver_and_gate_contract_unchanged": True,
    }
    report = _write_valid_worker_report(
        report_path=report_path,
        root=report_owner_root,
        lock_sha=lock_sha,
        checkpoint_root=root / "scientific-checkpoints",
        orchestration=orchestration,
    )
    if profile == "scaling":
        report["summary"]["release_certified"] = False
        report["summary"]["validation_tier"] = "pull_request"
    if report_mutator is not None:
        report_mutator(report)
    report["report_sha256"] = _report_digest(
        {key: value for key, value in report.items() if key != "report_sha256"},
        ensure_ascii=False,
    )
    core._atomic_write_json(report_path, report)
    digest = production_toy_physics_sha256(report)
    assert campaign_document["payload_sha256"] == campaign.campaign_manifest_sha256
    return report_path, digest


def _scaling_worker_payload(
    rank: int,
    lock_sha: str,
    digest: str,
    python_record: dict[str, str],
    *,
    root: pathlib.Path,
) -> dict[str, object]:
    execution_id = f"scaling-rank-{rank:03d}"
    campaign_path = root / "campaign-manifest.json"
    execution_path = root / "campaign-executions" / f"{execution_id}.json"
    report_path = root / "production-reports" / f"{execution_id}.json"
    campaign = toy_mpi.read_hashed_manifest(campaign_path)
    execution = toy_mpi.read_hashed_manifest(execution_path)
    config, _fixtures = toy_mpi._locked_profile_inputs("scaling")
    return {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "scaling",
        "certification_eligible": False,
        "release_certified": False,
        "source_revision": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
        "environment_sha256": lock_sha,
        "expected_rank_count": rank,
        "recorded_rank_count": rank,
        "execution_id": execution_id,
        "execution_mode": "serial" if rank == 1 else "mpi",
        "directory_mode": "fresh",
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
        "campaign_manifest_sha256": campaign["payload_sha256"],
        "execution_scientific_sha256": execution["payload"][
            "scientific_sha256"
        ],
        "production_toy_physics_sha256": digest,
        "configuration_sha256": toy_mpi._payload_sha256(config.as_dict()),
        "campaign_manifest": _artifact_record(root, campaign_path),
        "execution_manifest": _artifact_record(root, execution_path),
        "strict_report": _artifact_record(root, report_path),
        "memory_preflight": _memory_preflight("scaling", rank),
        "rank_runtime": _rank_runtime(
            rank,
            python_record,
            profile="scaling",
            root=root,
            execution_id=execution_id,
            environment_sha256=lock_sha,
        ),
        "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
    }


def _write_scaling_selection(
    root: pathlib.Path,
    lock_sha: str,
    *,
    times: tuple[float, ...] = (100.0, 60.0, 31.0, 30.0, 29.8),
    environment_prefix: pathlib.Path | None = None,
) -> pathlib.Path:
    root.mkdir(parents=True)
    prefix = (
        root.parent / "environment"
        if environment_prefix is None
        else environment_prefix
    ).resolve()
    launcher_record, python_record = _fake_mpi_environment(prefix)
    records = []
    for rank, wall_seconds in zip(mpi_scaling.RANK_LADDER, times, strict=True):
        repetition = root / f"rank-{rank:03d}"
        execution_id = f"scaling-rank-{rank:03d}"
        _report_path, digest = _write_canonical_worker_chain(
            root=repetition,
            profile="scaling",
            rank_count=rank,
            execution_id=execution_id,
            python_record=python_record,
            environment_sha256=lock_sha,
            report_owner_root=root,
        )
        worker_path = repetition / "runner-results" / f"{execution_id}.json"
        worker_path.parent.mkdir(parents=True, exist_ok=True)
        worker = toy_mpi.write_hashed_manifest(
            worker_path,
            _scaling_worker_payload(
                rank,
                lock_sha,
                digest,
                python_record,
                root=repetition,
            ),
        )
        stdout_path = root / "logs" / f"rank-{rank:03d}.stdout.log"
        stderr_path = root / "logs" / f"rank-{rank:03d}.stderr.log"
        stdout_path.parent.mkdir(exist_ok=True)
        stdout_path.write_text("synthetic stdout\n", encoding="ascii")
        stderr_path.write_text("", encoding="ascii")
        program = (
            python_record["path"],
            str(pathlib.Path(toy_mpi.__file__).resolve()),
            "_rank-worker",
            "--profile",
            "scaling",
            "--source-revision",
            COMMIT,
            "--source-tree-sha256",
            SOURCE_SHA256,
            "--environment-sha256",
            lock_sha,
            "--run-directory",
            str(repetition),
            "--execution-id",
            execution_id,
            "--expected-ranks",
            str(rank),
        )
        rank_path = root / f"rank-{rank:03d}-manifest.json"
        rank_payload = {
            "schema": "negfpy.remote_toy_mpi.scaling_rank.v1",
            "rank_count": rank,
            "source_revision": COMMIT,
            "source_tree_sha256": SOURCE_SHA256,
            "environment_sha256": lock_sha,
            "fresh_isolated_repetition": True,
            "resume_used": False,
            "outer_wall_seconds": wall_seconds,
            "returncode": 0,
            "completed": True,
            "failure": None,
            "argv": list(
                mpi_scaling.environment_mpi_argv(
                    prefix,
                    rank_count=rank,
                    physical_core_count=76,
                    socket_count=2,
                    program=program,
                )
            ),
            "binding_arguments": list(
                mpi_scaling.openmpi_binding_arguments(
                    rank,
                    physical_core_count=76,
                    socket_count=2,
                )
            ),
            "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
            "launcher": launcher_record,
            "python": python_record,
            "launcher_memory_preflight": _memory_preflight("scaling", rank),
            "stdout": _artifact_record(root, stdout_path),
            "stderr": _artifact_record(root, stderr_path),
            "worker_result_relative_path": worker_path.relative_to(root).as_posix(),
            "worker_result_file_sha256": toy_mpi._file_sha256(worker_path),
            "worker_result_payload_sha256": worker["payload_sha256"],
            "production_toy_physics_sha256": digest,
        }
        rank_document = toy_mpi.write_hashed_manifest(rank_path, rank_payload)
        records.append((rank_path, rank_document))
    selection = toy_mpi.build_scaling_selection_payload(
        records,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        physical_core_count=76,
        socket_count=2,
    )
    target = root / "scaling-selection.json"
    toy_mpi.write_hashed_manifest(target, selection)
    return target


def _artifact_record(root: pathlib.Path, path: pathlib.Path) -> dict[str, str]:
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": core._sha256(path),
    }


def _write_release_evidence(
    *,
    root: pathlib.Path,
    lock_sha: str,
    execution_id: str,
    rank_count: int,
    report_mutator=None,
    environment_prefix: pathlib.Path | None = None,
) -> pathlib.Path:
    root.mkdir(parents=True, exist_ok=True)
    prefix = (
        root.parent / "environment"
        if environment_prefix is None
        else environment_prefix
    ).resolve()
    _launcher_record, python_record = _fake_mpi_environment(prefix)
    report_path, digest = _write_canonical_worker_chain(
        root=root,
        profile="release",
        rank_count=rank_count,
        execution_id=execution_id,
        python_record=python_record,
        environment_sha256=lock_sha,
        report_owner_root=prefix.parent,
        report_mutator=report_mutator,
    )
    campaign_path = root / "campaign-manifest.json"
    campaign = toy_mpi.read_hashed_manifest(campaign_path)
    execution_path = root / "campaign-executions" / f"{execution_id}.json"
    execution = toy_mpi.read_hashed_manifest(execution_path)
    config, _fixtures = toy_mpi._locked_profile_inputs("release")
    runner_payload = {
        "schema": "negfpy.remote_toy_mpi.physics_result.v1",
        "profile": "release",
        "certification_eligible": True,
        "release_certified": True,
        "source_revision": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
        "environment_sha256": lock_sha,
        "directory_mode": "fresh",
        "expected_rank_count": rank_count,
        "recorded_rank_count": rank_count,
        "execution_mode": "serial" if rank_count == 1 else "mpi",
        "task_count": 28,
        "multidimensional_outcome_count": 13,
        "execution_id": execution_id,
        "open_system_case_count": 13,
        "continuation_path_count": 5,
        "all_outcomes_completed": True,
        "campaign_manifest_sha256": campaign["payload_sha256"],
        "execution_scientific_sha256": execution["payload"][
            "scientific_sha256"
        ],
        "production_toy_physics_sha256": digest,
        "configuration_sha256": toy_mpi._payload_sha256(config.as_dict()),
        "campaign_manifest": _artifact_record(root, campaign_path),
        "execution_manifest": _artifact_record(root, execution_path),
        "strict_report": _artifact_record(root, report_path),
        "thread_limits": dict(sorted(core.THREAD_LIMIT_ENV.items())),
        "memory_preflight": _memory_preflight("release", rank_count),
        "rank_runtime": _rank_runtime(
            rank_count,
            python_record,
            profile="release",
            root=root,
            execution_id=execution_id,
            environment_sha256=lock_sha,
        ),
    }
    result_path = root / "runner-results" / f"{execution_id}.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    toy_mpi.write_hashed_manifest(result_path, runner_payload)
    return result_path


def _write_release_launch_facts(
    *,
    deployment: pathlib.Path,
    release_root: pathlib.Path,
    execution_id: str,
    rank_count: int,
    lock_sha: str,
) -> tuple[pathlib.Path, pathlib.Path, dict[str, object]]:
    attempt_root = deployment / "artifacts" / "attempts" / "attempt-001"
    attempt_root.mkdir(parents=True, exist_ok=True)
    stdout_path = attempt_root / "mpi-release.stdout.log"
    stderr_path = attempt_root / "mpi-release.stderr.log"
    prefix = deployment / "environment"
    _launcher, _python = _fake_mpi_environment(prefix)
    mpi_runner = pathlib.Path(toy_mpi.__file__).resolve()
    program = (
        str(prefix / "bin" / "python"),
        str(mpi_runner),
        "release",
        "--run-directory",
        str(release_root),
        "--source-revision",
        COMMIT,
        "--source-tree-sha256",
        SOURCE_SHA256,
        "--environment-sha256",
        lock_sha,
        "--execution-id",
        execution_id,
        "--expected-ranks",
        str(rank_count),
    )
    argv = mpi_scaling.environment_mpi_argv(
        prefix,
        rank_count=rank_count,
        physical_core_count=76,
        socket_count=2,
        program=program,
    )
    binding_arguments = mpi_scaling.openmpi_binding_arguments(
        rank_count,
        physical_core_count=76,
        socket_count=2,
    )
    identity = _worker_identity(lock_sha)
    source_identity = identity["execution_source_identity"]
    prefix_manifest = identity["environment_prefix_manifest"]
    launch_environment = cli._scientific_launch_environment(
        root=deployment,
        attempt_root=attempt_root,
        environment_prefix=prefix,
    )
    intent_arguments = {
        "environment_prefix": prefix,
        "mpi_runner": mpi_runner,
        "argv": argv,
        "binding_arguments": binding_arguments,
        "selected_rank": rank_count,
        "source_revision": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
        "environment_sha256": lock_sha,
        "execution_id": execution_id,
        "execution_source_identity": source_identity,
        "environment_prefix_manifest": prefix_manifest,
        "launch_environment": launch_environment,
    }
    intent_path = attempt_root / "mpi-release-launch-intent.json"
    cli._write_release_launch_intent(
        path=intent_path,
        attempt_root=attempt_root,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        **intent_arguments,
    )
    stdout_path.write_text("synthetic release stdout\n", encoding="utf-8")
    stderr_path.write_text("", encoding="utf-8")
    receipt_path = attempt_root / "mpi-release-launch.json"
    cli._write_release_launch_receipt(
        path=receipt_path,
        intent_path=intent_path,
        attempt_root=attempt_root,
        environment_prefix=prefix,
        mpi_runner=mpi_runner,
        argv=argv,
        binding_arguments=binding_arguments,
        selected_rank=rank_count,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        execution_id=execution_id,
        execution_source_identity=source_identity,
        environment_prefix_manifest=prefix_manifest,
        launch_environment=launch_environment,
        post_run_verification={
            "passed": True,
            "checked_utc": "2026-07-26T00:00:01Z",
            "execution_source_identity_sha256": source_identity["manifest_sha256"],
            "environment_prefix_manifest_sha256": prefix_manifest["manifest_sha256"],
        },
        started_utc="2026-07-26T00:00:00Z",
        finished_utc="2026-07-26T00:00:01Z",
        returncode=0,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )
    facts = cli._validate_release_launch_receipt(
        path=receipt_path,
        intent_path=intent_path,
        attempt_root=attempt_root,
        environment_prefix=prefix,
        selected_rank=rank_count,
        physical_core_count=76,
        socket_count=2,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        execution_id=execution_id,
        release_root=release_root,
        mpi_runner=mpi_runner,
        execution_source_identity=source_identity,
        environment_prefix_manifest=prefix_manifest,
        launch_environment=launch_environment,
    )
    return receipt_path, stdout_path, facts


def _mock_successful_mpi_commands(
    *,
    root: pathlib.Path,
    lock_sha: str,
    captured: list[tuple[tuple[str, ...], dict[str, str]]] | None = None,
    rank_times: tuple[float, ...] = (100.0, 60.0, 31.0, 30.0, 29.8),
    report_mutator=None,
):
    def fake_run(argv, **kwargs):
        command = tuple(str(value) for value in argv)
        if "env" not in kwargs:
            return _REAL_SUBPROCESS_RUN(argv, **kwargs)
        environment = dict(kwargs["env"])
        if captured is not None:
            captured.append((command, environment))
        assert all(environment[name] == "1" for name in core.THREAD_LIMIT_ENV)
        if "scale" in command:
            run_directory = pathlib.Path(
                command[command.index("--run-directory") + 1]
            )
            _write_scaling_selection(
                run_directory,
                lock_sha,
                times=rank_times,
                environment_prefix=root / "environment",
            )
            kwargs["stdout"].write("synthetic scaling success\n")
        else:
            assert command[0] == str(root / "environment" / "bin" / "mpiexec")
            rank_count = int(command[command.index("-n") + 1])
            run_directory = pathlib.Path(
                command[command.index("--run-directory") + 1]
            )
            execution_id = command[command.index("--execution-id") + 1]
            _write_release_evidence(
                root=run_directory,
                lock_sha=lock_sha,
                execution_id=execution_id,
                rank_count=rank_count,
                report_mutator=report_mutator,
                environment_prefix=root / "environment",
            )
            kwargs["stdout"].write("synthetic release success\n")
        return SimpleNamespace(returncode=0)

    return fake_run


def test_worker_validates_current_attempt_and_publishes_self_contained_evidence(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("LD_PRELOAD", "/tmp/injected.so")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/tmp/injected-libraries")
    monkeypatch.setenv("OMPI_MCA_btl", "self,tcp")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )
    captured: list[tuple[tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root, lock_sha=lock_sha, captured=captured
    ))
    clock_tick = 0

    def ordered_utc_now() -> str:
        nonlocal clock_tick
        value = f"2026-07-26T00:00:00.{clock_tick:06d}+00:00"
        clock_tick += 1
        return value

    monkeypatch.setattr(core, "_utc_now", ordered_utc_now)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 0
    assert len(captured) == 2
    scale_argv, scale_environment = captured[0]
    release_argv, release_environment = captured[1]
    expected_environment = cli._scientific_launch_environment(
        root=root,
        attempt_root=root / "artifacts" / "attempts" / "attempt-001",
        environment_prefix=root / "environment",
    )
    for environment in (scale_environment, release_environment):
        assert environment == expected_environment
        assert environment["PATH"].split(os.pathsep)[0] == str(
            root / "environment" / "bin"
        )
        assert "LD_PRELOAD" not in environment
        assert "LD_LIBRARY_PATH" not in environment
        assert "OMPI_MCA_btl" not in environment
        assert not any(name.startswith("NEGFPY_TOY_GATE") for name in environment)
    assert scale_argv[0] == str(root / "environment" / "bin" / "python")
    assert "scale" in scale_argv
    assert release_argv[0] == str(root / "environment" / "bin" / "mpiexec")
    assert tuple(release_argv[1:7]) == mpi_scaling.openmpi_binding_arguments(
        4, physical_core_count=76, socket_count=2
    )
    assert release_argv[release_argv.index("-n") + 1] == "4"
    assert "release" in release_argv
    status = json.loads(
        (root / "artifacts" / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "completed"
    assert status["commit"] == COMMIT
    assert status["report_facts"]["schema_version"] == 4
    assert status["report_facts"]["protocol_version"] == 2
    assert status["report_facts"]["multidimensional_outcome_count"] == 13
    assert status["report_facts"]["checkpoint_count"] == 13
    assert status["report_facts"]["source_tree_sha256"] == SOURCE_SHA256
    assert status["scaling_facts"]["toy_rank_count"] == 4
    assert status["scaling_facts"]["physics_equal_across_all_ranks"] is True
    assert status["mpi_execution_facts"]["mode"] == "mpi"
    assert status["mpi_execution_facts"]["communicator_size"] == 4
    assert status["mpi_execution_facts"]["release_certified"] is True
    launch_facts = status["mpi_execution_facts"]["launch_receipt"]
    assert launch_facts["post_run_verified"] is True
    assert launch_facts["runner_file_sha256"] == core._sha256(
        root / "worktree" / "scripts" / "remote_toy_mpi_runner.py"
    )
    assert launch_facts["launch_environment_sha256"] == cli._launch_environment_sha256(
        expected_environment
    )
    attempt_root = root / "artifacts" / "attempts" / "attempt-001"
    receipt = toy_mpi.read_hashed_manifest(
        attempt_root / "mpi-release-launch.json"
    )["payload"]
    assert (
        receipt["started_utc"]
        <= receipt["post_run_verification"]["checked_utc"]
        <= receipt["finished_utc"]
    )
    intent = toy_mpi.read_hashed_manifest(
        attempt_root / "mpi-release-launch-intent.json"
    )
    binding = intent["payload"]["launch_binding"]
    assert binding["schema"] == "negfpy.remote_validation.release_launch_binding.v1"
    assert binding["launch_environment"] == expected_environment
    assert binding["worktree_files_manifest_sha256"] == (
        _worker_identity(lock_sha)["execution_source_identity"][
            "worktree_files_manifest_sha256"
        ]
    )
    publication = json.loads(
        (root / "artifacts" / "current-attempt.json").read_text(encoding="utf-8")
    )
    assert publication["mpi_release_launch_intent"] == (
        "attempts/attempt-001/mpi-release-launch-intent.json"
    )
    assert publication["mpi_release_launch_intent_sha256"] == core._sha256(
        attempt_root / "mpi-release-launch-intent.json"
    )
    assert not (root / "artifacts" / "job.lock").exists()
    verified = core.verify_sha256sums(root / "artifacts")
    assert "job-status.json" in verified
    assert "current-attempt.json" in verified
    assert "attempts/attempt-001/status.json" in verified
    assert "attempts/attempt-001/production_toy_gate.log" in verified
    assert "attempts/attempt-001/production_toy_gate.json" in verified
    assert "attempts/attempt-001/report-facts.json" in verified
    assert "attempts/attempt-001/checkpoint-index.json" in verified
    assert "attempts/attempt-001/mpi-release-runner-result.json" in verified
    assert "attempts/attempt-001/mpi-execution-manifest.json" in verified
    assert "attempts/attempt-001/mpi-campaign-manifest.json" in verified
    assert "attempts/attempt-001/mpi-scaling/scaling-selection.json" in verified
    assert "identity/deployment-manifest.json" in verified
    assert "identity/environment-manifest.json" in verified
    assert "identity/environment-explicit.lock" in verified
    assert "identity/identity-manifest.json" in verified
    assert all(not name.startswith("job.lock/") for name in verified)
    checkpoint_index = json.loads(
        (root / "artifacts" / "attempts" / "attempt-001" / "checkpoint-index.json").read_text(
            encoding="utf-8"
        )
    )
    assert checkpoint_index["count"] == 13
    assert all(
        item["artifact_relative_path"].startswith("toy-release/scientific-checkpoints/")
        for item in checkpoint_index["records"]
    )



@pytest.mark.parametrize("mutation_stage", ("pre_launch", "post_run"))
def test_worker_rejects_source_identity_mutation_around_release(
    mutation_stage: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    baseline = _worker_identity(lock_sha)
    changed = json.loads(json.dumps(baseline))
    changed_source = dict(changed["execution_source_identity"])
    changed_source["worktree_file_count"] = int(
        changed_source["worktree_file_count"]
    ) + 1
    changed["execution_source_identity"] = cli._canonical_manifest(
        {
            key: value
            for key, value in changed_source.items()
            if key != "manifest_sha256"
        }
    )
    calls = 0

    def identity_at_boundary(**_kwargs):
        nonlocal calls
        calls += 1
        if mutation_stage == "pre_launch" and calls >= 2:
            return changed
        if mutation_stage == "post_run" and calls >= 3:
            return changed
        return baseline

    monkeypatch.setattr(cli, "_record_remote_environment", identity_at_boundary)
    captured: list[tuple[tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        _mock_successful_mpi_commands(
            root=root,
            lock_sha=lock_sha,
            captured=captured,
        ),
    )
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 2
    attempt = root / "artifacts" / "attempts" / "attempt-001"
    status = json.loads(
        (root / "artifacts" / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    if mutation_stage == "pre_launch":
        assert len(captured) == 1
        assert not (attempt / "mpi-release-launch-intent.json").exists()
        assert not (attempt / "mpi-release-launch.json").exists()
    else:
        assert len(captured) == 2
        assert (attempt / "mpi-release-launch-intent.json").is_file()
        receipt = toy_mpi.read_hashed_manifest(
            attempt / "mpi-release-launch.json"
        )
        assert receipt["payload"]["post_run_verification"]["passed"] is False


def test_worker_uses_rank_one_serial_fallback_but_retains_silicon_mpi_choice(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )
    captured: list[tuple[tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root,
        lock_sha=lock_sha,
        captured=captured,
        rank_times=(100.0, 95.0, 92.0, 90.0, 89.0),
    ))

    assert cli._worker(deployment_dir=root, commit=COMMIT) == 0
    status = json.loads(
        (root / "artifacts" / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["scaling_facts"]["toy_rank_count"] == 1
    assert status["scaling_facts"]["toy_mpi_enabled"] is False
    assert status["scaling_facts"]["silicon_rank_count"] == 4
    assert status["mpi_execution_facts"]["mode"] == "serial"
    assert status["mpi_execution_facts"]["communicator_size"] == 1
    release_argv = captured[1][0]
    assert release_argv[release_argv.index("-n") + 1] == "1"
    assert tuple(release_argv[1:7]) == mpi_scaling.openmpi_binding_arguments(
        1, physical_core_count=76, socket_count=2
    )


def test_lifecycle_rebuilds_scaling_selection_and_rejects_worker_tampering(
    tmp_path: pathlib.Path,
) -> None:
    lock_sha = "7" * 64
    selection = _write_scaling_selection(tmp_path / "scaling", lock_sha)
    facts = cli._validate_scaling_selection(
        selection_path=selection,
        environment_prefix=tmp_path / "environment",
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        physical_core_count=76,
        socket_count=2,
    )
    assert facts["toy_rank_count"] == 4
    worker = (
        selection.parent
        / "rank-004"
        / "runner-results"
        / "scaling-rank-004.json"
    )
    worker.write_bytes(worker.read_bytes() + b" ")
    with pytest.raises(core.RemoteValidationError, match="revalidation"):
        cli._validate_scaling_selection(
            selection_path=selection,
            environment_prefix=tmp_path / "environment",
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            physical_core_count=76,
            socket_count=2,
        )


def test_scaling_validation_accepts_canonical_environment_entry_symlinks(
    tmp_path: pathlib.Path,
) -> None:
    lock_sha = "7" * 64
    prefix = tmp_path / "environment"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    for entry_name, target_name in (
        ("mpiexec", "mpirun"),
        ("python", "python3.10"),
    ):
        target = binary / target_name
        target.write_text(
            f"#!/bin/sh\n# synthetic {target_name}\nexit 0\n",
            encoding="ascii",
        )
        target.chmod(0o755)
        (binary / entry_name).symlink_to(target.name)

    selection = _write_scaling_selection(
        tmp_path / "scaling",
        lock_sha,
        environment_prefix=prefix,
    )
    facts = cli._validate_scaling_selection(
        selection_path=selection,
        environment_prefix=prefix,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        physical_core_count=76,
        socket_count=2,
    )
    assert facts["toy_rank_count"] == 4


def test_scaling_validation_rejects_noncanonical_executable_aliases(
    tmp_path: pathlib.Path,
) -> None:
    lock_sha = "7" * 64
    canonical_prefix = tmp_path / "environment"
    canonical_launcher, canonical_python = _fake_mpi_environment(canonical_prefix)
    alias_prefix = tmp_path / "alias-environment"
    alias_binary = alias_prefix / "bin"
    alias_binary.mkdir(parents=True)
    (alias_binary / "mpiexec").symlink_to(canonical_launcher["resolved_path"])
    (alias_binary / "python").symlink_to(canonical_python["resolved_path"])
    selection = _write_scaling_selection(
        tmp_path / "alias-scaling",
        lock_sha,
        environment_prefix=alias_prefix,
    )

    with pytest.raises(
        core.RemoteValidationError,
        match="did not use the deployed MPI/Python environment",
    ):
        cli._validate_scaling_selection(
            selection_path=selection,
            environment_prefix=canonical_prefix,
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            physical_core_count=76,
            socket_count=2,
        )


@pytest.mark.parametrize(
    ("entry_name", "target_name"),
    (("mpiexec", "mpirun"), ("python", "python3.10")),
)
def test_scaling_validation_rejects_executable_target_outside_prefix(
    entry_name: str,
    target_name: str,
    tmp_path: pathlib.Path,
) -> None:
    lock_sha = "7" * 64
    prefix = tmp_path / "environment"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    for name in ("mpiexec", "python"):
        if name == entry_name:
            continue
        path = binary / name
        path.write_text(
            f"#!/bin/sh\n# synthetic {name}\nexit 0\n",
            encoding="ascii",
        )
        path.chmod(0o755)
    outside = tmp_path / "outside" / target_name
    outside.parent.mkdir()
    outside.write_text(
        f"#!/bin/sh\n# outside {target_name}\nexit 0\n",
        encoding="ascii",
    )
    outside.chmod(0o755)
    (binary / entry_name).symlink_to(
        pathlib.Path("../..") / outside.parent.name / outside.name
    )
    selection = _write_scaling_selection(
        tmp_path / f"scaling-{entry_name}",
        lock_sha,
        environment_prefix=prefix,
    )

    with pytest.raises(
        core.RemoteValidationError,
        match="executable escaped its environment prefix",
    ):
        cli._validate_scaling_selection(
            selection_path=selection,
            environment_prefix=prefix,
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            physical_core_count=76,
            socket_count=2,
        )


def test_lifecycle_rejects_hash_mismatched_mpi_execution_manifest(
    tmp_path: pathlib.Path,
) -> None:
    deployment = tmp_path / COMMIT
    release_root = deployment / "artifacts" / "toy-release"
    lock_sha = "7" * 64
    execution_id = "release-attempt-001"
    result = _write_release_evidence(
        root=release_root,
        lock_sha=lock_sha,
        execution_id=execution_id,
        rank_count=4,
        environment_prefix=deployment / "environment",
    )
    _receipt, _stdout, launch_facts = _write_release_launch_facts(
        deployment=deployment,
        release_root=release_root,
        execution_id=execution_id,
        rank_count=4,
        lock_sha=lock_sha,
    )
    facts, _report, _checkpoints = cli._validate_release_runner_result(
        result_path=result,
        release_root=release_root,
        execution_id=execution_id,
        selected_rank=4,
        source_revision=COMMIT,
        source_tree_sha256=SOURCE_SHA256,
        environment_sha256=lock_sha,
        socket_count=2,
        environment_prefix=deployment / "environment",
        launch_facts=launch_facts,
    )
    assert facts["mode"] == "mpi"
    execution = release_root / "campaign-executions" / f"{execution_id}.json"
    execution.write_bytes(execution.read_bytes() + b" ")
    with pytest.raises(core.RemoteValidationError, match="hash-mismatched"):
        cli._validate_release_runner_result(
            result_path=result,
            release_root=release_root,
            execution_id=execution_id,
            selected_rank=4,
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            socket_count=2,
            environment_prefix=deployment / "environment",
            launch_facts=launch_facts,
        )


@pytest.mark.parametrize("tamper_target", ("receipt", "stdout"))
def test_release_launch_receipt_rejects_tampering(
    tamper_target: str,
    tmp_path: pathlib.Path,
) -> None:
    deployment = tmp_path / COMMIT
    release_root = deployment / "artifacts" / "toy-release"
    execution_id = "release-attempt-001"
    lock_sha = "7" * 64
    receipt, stdout, _facts = _write_release_launch_facts(
        deployment=deployment,
        release_root=release_root,
        execution_id=execution_id,
        rank_count=4,
        lock_sha=lock_sha,
    )
    target = receipt if tamper_target == "receipt" else stdout
    target.write_bytes(target.read_bytes() + b"tamper\n")
    identity = _worker_identity(lock_sha)
    launch_environment = cli._scientific_launch_environment(
        root=deployment,
        attempt_root=receipt.parent,
        environment_prefix=deployment / "environment",
    )
    with pytest.raises(core.RemoteValidationError):
        cli._validate_release_launch_receipt(
            path=receipt,
            intent_path=receipt.parent / "mpi-release-launch-intent.json",
            attempt_root=receipt.parent,
            environment_prefix=deployment / "environment",
            selected_rank=4,
            physical_core_count=76,
            socket_count=2,
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            execution_id=execution_id,
            release_root=release_root,
            mpi_runner=pathlib.Path(toy_mpi.__file__).resolve(),
            execution_source_identity=identity["execution_source_identity"],
            environment_prefix_manifest=identity["environment_prefix_manifest"],
            launch_environment=launch_environment,
        )


def test_release_launch_receipt_rejects_post_run_check_after_finish(
    tmp_path: pathlib.Path,
) -> None:
    deployment = tmp_path / COMMIT
    release_root = deployment / "artifacts" / "toy-release"
    execution_id = "release-attempt-001"
    lock_sha = "7" * 64
    receipt, _stdout, _facts = _write_release_launch_facts(
        deployment=deployment,
        release_root=release_root,
        execution_id=execution_id,
        rank_count=4,
        lock_sha=lock_sha,
    )
    document = toy_mpi.read_hashed_manifest(receipt)
    payload = json.loads(json.dumps(document["payload"]))
    payload["post_run_verification"]["checked_utc"] = "2026-07-26T00:00:02Z"
    receipt.unlink()
    toy_mpi.write_hashed_manifest(receipt, payload)
    identity = _worker_identity(lock_sha)
    launch_environment = cli._scientific_launch_environment(
        root=deployment,
        attempt_root=receipt.parent,
        environment_prefix=deployment / "environment",
    )
    with pytest.raises(
        core.RemoteValidationError,
        match="post-run source/environment revalidation did not pass",
    ):
        cli._validate_release_launch_receipt(
            path=receipt,
            intent_path=receipt.parent / "mpi-release-launch-intent.json",
            attempt_root=receipt.parent,
            environment_prefix=deployment / "environment",
            selected_rank=4,
            physical_core_count=76,
            socket_count=2,
            source_revision=COMMIT,
            source_tree_sha256=SOURCE_SHA256,
            environment_sha256=lock_sha,
            execution_id=execution_id,
            release_root=release_root,
            mpi_runner=pathlib.Path(toy_mpi.__file__).resolve(),
            execution_source_identity=identity["execution_source_identity"],
            environment_prefix_manifest=identity["environment_prefix_manifest"],
            launch_environment=launch_environment,
        )


def test_worker_exact_retry_reuses_only_revalidated_scaling_evidence(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )
    first_commands: list[tuple[str, ...]] = []

    def interrupted_release(argv, **kwargs):
        command = tuple(str(value) for value in argv)
        if "env" not in kwargs:
            return _REAL_SUBPROCESS_RUN(argv, **kwargs)
        first_commands.append(command)
        if "scale" in command:
            directory = pathlib.Path(
                command[command.index("--run-directory") + 1]
            )
            _write_scaling_selection(
                directory,
                lock_sha,
                environment_prefix=root / "environment",
            )
            return SimpleNamespace(returncode=0)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(cli.subprocess, "run", interrupted_release)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 1
    assert len(first_commands) == 2
    lock = root / "artifacts" / "job.lock"
    lock.mkdir()
    (lock / "session").write_text("retry-session\n", encoding="ascii")

    retry_commands: list[tuple[tuple[str, ...], dict[str, str]]] = []
    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root,
        lock_sha=lock_sha,
        captured=retry_commands,
    ))
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 0
    assert len(retry_commands) == 1
    assert "release" in retry_commands[0][0]
    assert "scale" not in retry_commands[0][0]
    status = json.loads(
        (root / "artifacts" / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["attempt"] == 2
    assert status["state"] == "completed"
    assert status["scaling_facts"]["reused_for_exact_retry"] is True
    assert status["scaling_facts"]["source_attempt"] == "attempt-001"
    assert status["scaling_facts"]["selection_relative_path"] == (
        "attempts/attempt-001/mpi-scaling/scaling-selection.json"
    )

def test_worker_identity_drift_is_terminal_and_releases_lock(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    changed = _worker_identity(lock_sha)
    changed["python_version"] = "3.11.0"
    monkeypatch.setattr(
        cli, "_record_remote_environment", lambda **_kwargs: changed
    )

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("pytest must not run after identity drift")

    monkeypatch.setattr(cli.subprocess, "run", must_not_run)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 1
    artifacts = root / "artifacts"
    status = json.loads(
        (artifacts / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["returncode"] == 1
    assert not (artifacts / "job.lock").exists()
    assert "attempt environment differs" in (
        artifacts / "attempts" / "attempt-001" / "production_toy_gate.log"
    ).read_text(encoding="utf-8")
    verified = core.verify_sha256sums(artifacts)
    assert "attempts/attempt-001/status.json" in verified


def test_mpi_identity_is_an_immutable_deployment_field() -> None:
    baseline = _worker_identity("d" * 64)
    changed = json.loads(json.dumps(baseline))
    changed["mpi"]["version"] = [5, 0, 9]
    with pytest.raises(ValueError, match="mpi"):
        identity.compare_environment_identity(baseline, changed)


def test_worker_rejects_hash_valid_but_uncertified_current_report(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )

    def mutate_report(report):
        report["summary"]["release_certified"] = False
    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root, lock_sha=lock_sha, report_mutator=mutate_report
    ))
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 2
    artifacts = root / "artifacts"
    status = json.loads(
        (artifacts / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["report_facts"] is None
    assert not (artifacts / "job.lock").exists()
    assert "strict report certification state changed" in (
        artifacts / "attempts" / "attempt-001" / "production_toy_gate.log"
    ).read_text(encoding="utf-8")
    core.verify_sha256sums(artifacts)



@pytest.mark.parametrize(
    "gate_key",
    (
        "multidimensional_static_configuration_passed",
        "multidimensional_runtime_evidence_passed",
    ),
)
def test_worker_requires_explicit_multidimensional_contract_subgates(
    gate_key: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )

    def mutate_report(report):
        assert report["summary"]["multidimensional_contract_passed"] is True
        report["summary"][gate_key] = False
    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root, lock_sha=lock_sha, report_mutator=mutate_report
    ))
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 2
    artifacts = root / "artifacts"
    status = json.loads(
        (artifacts / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["report_facts"] is None
    assert not (artifacts / "job.lock").exists()
    assert f"report summary gate failed: {gate_key}" in (
        artifacts
        / "attempts"
        / "attempt-001"
        / "production_toy_gate.log"
    ).read_text(encoding="utf-8")
    core.verify_sha256sums(artifacts)


def test_worker_invalidates_stale_root_report_and_checksum_before_attempt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    artifacts = root / "artifacts"
    (artifacts / "production_toy_gate.json").write_text(
        '{"stale":true}\n', encoding="utf-8"
    )
    (artifacts / "production_toy_gate.log").write_text(
        "stale success\n", encoding="utf-8"
    )
    (artifacts / "multidimensional_toy_gate_failure.json").write_text(
        '{"stale_failure":true}\n', encoding="utf-8"
    )
    core._write_sha256sums(
        artifacts,
        (
            pathlib.PurePath("production_toy_gate.json"),
            pathlib.PurePath("production_toy_gate.log"),
            pathlib.PurePath("multidimensional_toy_gate_failure.json"),
        ),
    )
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )

    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root, lock_sha=lock_sha
    ))
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 0
    attempt = artifacts / "attempts" / "attempt-001"
    assert (attempt / "previous-SHA256SUMS").is_file()
    assert (attempt / "legacy-production_toy_gate.json").is_file()
    assert (attempt / "legacy-production_toy_gate.log").is_file()
    assert (
        attempt / "previous-multidimensional_toy_gate_failure.json"
    ).is_file()
    assert not (artifacts / "production_toy_gate.json").exists()
    current = json.loads(
        (artifacts / "current-attempt.json").read_text(encoding="utf-8")
    )
    assert current["report"] == (
        "attempts/attempt-001/production_toy_gate.json"
    )


def test_worker_moves_current_runner_failure_sidecar_into_attempt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )

    def fake_run(_argv, **kwargs):
        (root / "artifacts" / "multidimensional_toy_gate_failure.json").write_text(
            '{"current_failure":true}\n', encoding="utf-8"
        )
        kwargs["stdout"].write("synthetic failure\n")
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 1
    artifacts = root / "artifacts"
    assert not (artifacts / "multidimensional_toy_gate_failure.json").exists()
    assert (
        artifacts
        / "attempts"
        / "attempt-001"
        / "multidimensional_toy_gate_failure.json"
    ).is_file()


def test_worker_keeps_lock_and_never_completes_when_checksum_publication_fails(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, lock_sha = _prepare_worker_root(tmp_path)
    for name in core.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setattr(
        cli,
        "_record_remote_environment",
        lambda **_kwargs: _worker_identity(lock_sha),
    )

    monkeypatch.setattr(cli.subprocess, "run", _mock_successful_mpi_commands(
        root=root, lock_sha=lock_sha
    ))
    monkeypatch.setattr(
        core,
        "_write_sha256sums",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 1
    artifacts = root / "artifacts"
    assert (artifacts / "job.lock").is_dir()
    assert not (artifacts / "SHA256SUMS").exists()
    status = json.loads(
        (artifacts / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "finalization_failed"
    assert "disk full" in status["finalization_error"]


def test_status_reports_dead_tmux_lock_as_stale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(core, "check_ssh_alias", lambda _runner: {})

    class Runner:
        dry_run = False

        def __init__(self) -> None:
            self.command = ""

        def run(self, argv, **_kwargs):
            self.command = argv[-1]
            return core.Completed(tuple(argv), 0, '{"state":"stale_lock"}\n', "")

    runner = Runner()
    result = operations.status(commit=COMMIT, session=None, runner=runner)
    assert result["state"] == "stale_lock"
    assert "has-session" in runner.command
    assert "elif test -d" in runner.command



class _ArtifactCopyRunner:
    dry_run = False

    def __init__(self, source: pathlib.Path) -> None:
        self.source = source

    def run(self, argv, **_kwargs):
        destination = pathlib.Path(argv[-1])
        shutil.copytree(self.source, destination, dirs_exist_ok=True)
        return core.Completed(tuple(argv), 0, "", "")


def _terminal_artifacts(tmp_path: pathlib.Path) -> pathlib.Path:
    source = tmp_path / "remote-artifacts"
    source.mkdir()
    core._atomic_write_json(
        source / "job-status.json",
        {
            "schema": "negfpy.remote_validation.job_status.v1",
            "state": "failed",
            "commit": COMMIT,
            "attempt": 1,
            "returncode": 1,
        },
    )
    (source / "production_toy_gate.log").write_text(
        "failed as archived\n", encoding="utf-8"
    )
    core._write_sha256sums(
        source,
        (pathlib.PurePath("job-status.json"), pathlib.PurePath("production_toy_gate.log")),
    )
    return source


@pytest.mark.parametrize("state", ["running", "finalizing", "missing"])
def test_fetch_rejects_nonterminal_state(
    state: str,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {"state": state, "commit": COMMIT},
    )
    with pytest.raises(core.RemoteValidationError, match="not terminal"):
        operations.fetch(
            commit=COMMIT,
            destination_root=tmp_path / "fetched",
            runner=_ArtifactCopyRunner(tmp_path),
        )


def test_fetch_allows_archived_failure_with_complete_checksums(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "failed", "commit": COMMIT, "returncode": 1
        },
    )
    destination = operations.fetch(
        commit=COMMIT,
        destination_root=tmp_path / "fetched",
        runner=_ArtifactCopyRunner(source),
    )
    assert (destination / "fetch-receipt.json").is_file()
    assert core.verify_sha256sums(destination) == (
        "job-status.json",
        "production_toy_gate.log",
    )


def test_fetch_prunes_unhashed_empty_directories_to_file_projection(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    (source / "attempts" / "attempt-001" / "runtime-tmp").mkdir(
        parents=True
    )
    (
        source
        / "attempts"
        / "attempt-001"
        / "runtime-config"
        / "matplotlib"
    ).mkdir(parents=True)
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "failed", "commit": COMMIT, "returncode": 1
        },
    )
    destination = operations.fetch(
        commit=COMMIT,
        destination_root=tmp_path / "fetched",
        runner=_ArtifactCopyRunner(source),
    )
    observed_directories = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("*")
        if path.is_dir()
    }
    assert observed_directories == set()
    assert core.verify_sha256sums(destination) == (
        "job-status.json",
        "production_toy_gate.log",
    )


def test_fetch_rejects_unprunable_empty_directory_without_receipt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    (source / "runtime-tmp").mkdir()
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "failed", "commit": COMMIT, "returncode": 1
        },
    )
    original_rmdir = pathlib.Path.rmdir

    def fail_runtime_rmdir(path: pathlib.Path) -> None:
        if path.name == "runtime-tmp":
            raise PermissionError("fault-injected rmdir failure")
        original_rmdir(path)

    monkeypatch.setattr(pathlib.Path, "rmdir", fail_runtime_rmdir)
    destination = tmp_path / "fetched" / COMMIT
    with pytest.raises(
        core.RemoteValidationError,
        match="empty or unreferenced directory",
    ):
        operations.fetch(
            commit=COMMIT,
            destination_root=tmp_path / "fetched",
            runner=_ArtifactCopyRunner(source),
        )
    assert destination.is_dir()
    assert not (destination / "fetch-receipt.json").exists()


def test_fetch_rejects_special_entry_without_receipt(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "failed", "commit": COMMIT, "returncode": 1
        },
    )

    class SpecialEntryRunner(_ArtifactCopyRunner):
        def run(self, argv, **kwargs):
            completed = super().run(argv, **kwargs)
            os.mkfifo(pathlib.Path(argv[-1]) / "unlisted.pipe")
            return completed

    destination = tmp_path / "fetched" / COMMIT
    with pytest.raises(
        core.RemoteValidationError,
        match="special filesystem entries",
    ):
        operations.fetch(
            commit=COMMIT,
            destination_root=tmp_path / "fetched",
            runner=SpecialEntryRunner(source),
        )
    assert destination.is_dir()
    assert not (destination / "fetch-receipt.json").exists()


def test_fetch_rejects_incomplete_checksum_coverage(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    (source / "unlisted.txt").write_text("not covered\n", encoding="utf-8")
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "completed", "commit": COMMIT, "returncode": 0
        },
    )
    with pytest.raises(core.RemoteValidationError, match="does not cover"):
        operations.fetch(
            commit=COMMIT,
            destination_root=tmp_path / "fetched",
            runner=_ArtifactCopyRunner(source),
        )

def test_fetch_rejects_existing_destination_symlink_without_writing_target(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    destination_root = tmp_path / "fetched"
    destination_root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (destination_root / COMMIT).symlink_to(external, target_is_directory=True)
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {
            "state": "completed",
            "commit": COMMIT,
            "returncode": 0,
        },
    )
    with pytest.raises(core.RemoteValidationError, match="already exists"):
        operations.fetch(
            commit=COMMIT,
            destination_root=destination_root,
            runner=_ArtifactCopyRunner(source),
        )
    assert list(external.iterdir()) == []


def test_fetch_rejects_nested_sha256sums_even_when_root_manifest_is_complete(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _terminal_artifacts(tmp_path)
    nested = source / "nested"
    nested.mkdir()
    (nested / "SHA256SUMS").write_text("nested manifest\n", encoding="ascii")
    core._write_sha256sums(
        source,
        (
            pathlib.PurePath("job-status.json"),
            pathlib.PurePath("production_toy_gate.log"),
            pathlib.PurePath("nested/SHA256SUMS"),
        ),
    )
    monkeypatch.setattr(
        operations,
        "status",
        lambda **_kwargs: {"state": "failed", "commit": COMMIT, "returncode": 1},
    )
    with pytest.raises(core.RemoteValidationError, match="nested SHA256SUMS"):
        operations.fetch(
            commit=COMMIT,
            destination_root=tmp_path / "fetched",
            runner=_ArtifactCopyRunner(source),
        )


def test_cli_has_no_credential_or_host_override_options() -> None:
    help_text = subprocess.run(
        (sys.executable, str(SCRIPTS / "remote_heavy_cli.py"), "--help"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()
    assert "password" not in help_text
    assert "--host" not in help_text
    assert "--user" not in help_text
    assert "--identity" not in help_text


def test_hosted_workflows_do_not_ssh_or_enable_the_heavy_gate() -> None:
    workflow_root = pathlib.Path(__file__).parents[1] / ".github" / "workflows"
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted(workflow_root.glob("*.yml"))
    ).lower()
    assert "negfpy_run_full_anharmonic_toy_gate" not in text
    assert "remote_heavy" not in text
    assert " ssh " not in text
    assert "\nssh " not in text


def test_remote_python_entrypoints_cannot_dirty_a_clean_worktree_with_pycache(
    tmp_path: pathlib.Path,
) -> None:
    bootstrap = cli._remote_bootstrap_command(COMMIT)
    launch = cli._launch_command(COMMIT, "negf-toy-pycache-regression")
    assert "env PYTHONDONTWRITEBYTECODE=1 " in bootstrap
    remote = core._remote_shell_dir(COMMIT)
    assert (
        f"PATH={core.REMOTE_TOOLS_BIN}:{remote}/environment/bin:/usr/bin:/bin "
        in bootstrap
    )
    assert (
        f"PATH={core.REMOTE_TOOLS_BIN}:{remote}/environment/bin:/usr/bin:/bin "
        in launch
    )
    assert "env PYTHONDONTWRITEBYTECODE=1 " in launch

    repository = pathlib.Path(__file__).parents[1].resolve()
    worktree = tmp_path / "clean-worktree"
    _REAL_SUBPROCESS_RUN(
        (
            "git", "clone", "--quiet", "--no-hardlinks", "--no-checkout",
            str(repository), str(worktree),
        ),
        check=True,
    )
    _REAL_SUBPROCESS_RUN(
        ("git", "-C", str(worktree), "checkout", "--quiet", "--detach", "HEAD"),
        check=True,
    )
    local_git = shutil.which("git")
    assert local_git is not None
    remote_tools = tmp_path / "remote-tools-bin"
    remote_tools.mkdir()
    (remote_tools / "git").symlink_to(pathlib.Path(local_git).resolve())
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": str(remote_tools),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(worktree / "src"),
        }
    )
    _REAL_SUBPROCESS_RUN(
        (
            sys.executable,
            "-c",
            (
                "import negfpy; "
                "from negfpy.anharmonic._provenance import code_provenance; "
                "record = code_provenance(); "
                "assert record[\"git_revision\"] != \"unknown\"; "
                "assert record[\"git_dirty\"] is False; "
                "from negfpy.anharmonic.toy._mpi_production_gate_physics "
                "import production_toy_physics_sha256; "
                "from negfpy.anharmonic.toy._mpi_production_gate "
                "import collect_production_toy_campaign_outcomes; "
                "from negfpy.anharmonic.toy.production_gate_report "
                "import ProductionToyGateReport"
            ),
        ),
        cwd=worktree,
        check=True,
        env=environment,
    )
    ignored = _REAL_SUBPROCESS_RUN(
        (
            "git", "-C", str(worktree), "ls-files", "--others", "--ignored",
            "--exclude-standard",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    status = _REAL_SUBPROCESS_RUN(
        (
            "git", "-C", str(worktree), "status", "--porcelain=v1",
            "--untracked-files=all",
        ),
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert ignored == ""
    assert status == ""
