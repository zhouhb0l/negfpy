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

import pytest


SCRIPTS = pathlib.Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
core = importlib.import_module("remote_heavy_validation")
operations = importlib.import_module("remote_heavy_runtime")
explicit = importlib.import_module("remote_heavy_explicit")
cli = importlib.import_module("remote_heavy_cli")
lock_validation = importlib.import_module("remote_heavy_lock")
identity = importlib.import_module("remote_heavy_identity")
security = importlib.import_module("remote_heavy_security")

COMMIT = "0123456789abcdef0123456789abcdef01234567"
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


def _worker_identity(lock_sha: str) -> dict[str, object]:
    return {
        "schema": "negfpy.remote_validation.remote_environment.v1",
        "commit": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
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
) -> dict[str, object]:
    artifacts = root / "artifacts"
    identity = {
        "git_revision": COMMIT,
        "source_tree_sha256": SOURCE_SHA256,
    }
    provenance = {"identity": identity, "complete": True, "clean": True}
    outcomes: list[dict[str, object]] = []
    checkpoint_records: list[dict[str, object]] = []
    for index in range(13):
        run_id = f"run-{index:02d}"
        checkpoint = artifacts / "checkpoints" / f"{run_id}.h5"
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
    report["report_sha256"] = _report_digest(report, ensure_ascii=False)
    core._atomic_write_json(report_path, report)
    return report


def test_worker_validates_current_attempt_and_publishes_self_contained_evidence(
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
    captured: dict[str, object] = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        _write_valid_worker_report(
            report_path=pathlib.Path(kwargs["env"]["NEGFPY_TOY_GATE_REPORT"]),
            root=root,
            lock_sha=lock_sha,
        )
        kwargs["stdout"].write("synthetic success\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 0
    environment = captured["env"]
    assert isinstance(environment, dict)
    for name in core.THREAD_LIMIT_ENV:
        assert environment[name] == "1"
    assert environment["PATH"].split(os.pathsep)[0] == core.REMOTE_TOOLS_BIN
    assert "NEGFPY_TOY_GATE_RESUME" not in environment
    assert environment["NEGFPY_VALIDATION_ENV_SHA256"] == lock_sha
    assert pathlib.Path(
        environment["NEGFPY_TOY_GATE_CHECKPOINT_DIR"]
    ).name == "checkpoints"
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
    assert not (root / "artifacts" / "job.lock").exists()
    verified = core.verify_sha256sums(root / "artifacts")
    assert "job-status.json" in verified
    assert "current-attempt.json" in verified
    assert "attempts/attempt-001/status.json" in verified
    assert "attempts/attempt-001/production_toy_gate.log" in verified
    assert "attempts/attempt-001/production_toy_gate.json" in verified
    assert "attempts/attempt-001/report-facts.json" in verified
    assert "attempts/attempt-001/checkpoint-index.json" in verified
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
        item["artifact_relative_path"].startswith("checkpoints/")
        for item in checkpoint_index["records"]
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

    def fake_run(_argv, **kwargs):
        path = pathlib.Path(kwargs["env"]["NEGFPY_TOY_GATE_REPORT"])
        report = _write_valid_worker_report(
            report_path=path, root=root, lock_sha=lock_sha
        )
        report["summary"]["release_certified"] = False
        report["report_sha256"] = _report_digest(
            {key: value for key, value in report.items() if key != "report_sha256"},
            ensure_ascii=False,
        )
        core._atomic_write_json(path, report)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
    assert cli._worker(deployment_dir=root, commit=COMMIT) == 2
    artifacts = root / "artifacts"
    status = json.loads(
        (artifacts / "job-status.json").read_text(encoding="utf-8")
    )
    assert status["state"] == "failed"
    assert status["report_facts"] is None
    assert not (artifacts / "job.lock").exists()
    assert "report summary gate failed: release_certified" in (
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

    def fake_run(_argv, **kwargs):
        path = pathlib.Path(kwargs["env"]["NEGFPY_TOY_GATE_REPORT"])
        report = _write_valid_worker_report(
            report_path=path,
            root=root,
            lock_sha=lock_sha,
        )
        assert report["summary"]["multidimensional_contract_passed"] is True
        report["summary"][gate_key] = False
        report["report_sha256"] = _report_digest(
            {
                key: value
                for key, value in report.items()
                if key != "report_sha256"
            },
            ensure_ascii=False,
        )
        core._atomic_write_json(path, report)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
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

    def fake_run(_argv, **kwargs):
        _write_valid_worker_report(
            report_path=pathlib.Path(kwargs["env"]["NEGFPY_TOY_GATE_REPORT"]),
            root=root,
            lock_sha=lock_sha,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
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

    def fake_run(_argv, **kwargs):
        _write_valid_worker_report(
            report_path=pathlib.Path(kwargs["env"]["NEGFPY_TOY_GATE_REPORT"]),
            root=root,
            lock_sha=lock_sha,
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)
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
