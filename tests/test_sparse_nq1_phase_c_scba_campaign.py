from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
campaign = importlib.import_module("run_sparse_nq1_phase_c_scba_campaign")


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(campaign._canonical(value))


def test_scba_watchdog_is_independent_of_born_gate() -> None:
    from negfpy.anharmonic import _material_nq1_pilot as pilot

    assert campaign.SCBA_WALL_LIMIT_SECONDS == 23 * 60 * 60
    assert campaign.SCBA_RSS_LIMIT_BYTES == 430 * 1024**3
    # The supervisor producer and pilot receipt consumer must agree exactly.
    assert campaign.SCBA_WALL_LIMIT_SECONDS == pilot.PILOT_WALL_LIMIT_SECONDS
    assert campaign.SCBA_RSS_LIMIT_BYTES == pilot.PILOT_RSS_LIMIT_BYTES
    assert campaign.common.WALL_LIMIT_SECONDS == 2 * 60 * 60
    assert campaign.common.RSS_LIMIT_BYTES == 4 * 1024**3


def _segmented_manifest(artifacts: Path) -> dict[str, object]:
    checkpoint = artifacts / "checkpoint" / "sigma-pp.h5"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-1")
    return {
        "schema": campaign.SCHEMA,
        "run_id": "segmented-run",
        "source_tree_sha256": "a" * 64,
        "environment_lock": {"sha256": "b" * 64},
        "preflight": {"checkpoint_identity_sha256": "c" * 64},
        "outputs": {"checkpoint": str(checkpoint)},
        "segmentation": {
            "enabled": True,
            "max_segments": campaign.SCBA_MAX_SEGMENTS,
            "segment_wall_limit_seconds": (
                campaign.SCBA_SEGMENT_WALL_LIMIT_SECONDS
            ),
            "planned_stop_trigger": "wall-timeout",
            "automatic_advance": False,
        },
    }


def _segment_checkpoint(iterations: int) -> SimpleNamespace:
    return SimpleNamespace(
        terminal=False,
        status="in_progress",
        history=tuple(range(iterations)),
        identity=SimpleNamespace(
            run_id="segmented-run",
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
            identity_sha256="c" * 64,
        ),
    )


def _segment_supervision() -> dict[str, object]:
    return {
        "returncode": -15,
        "elapsed_seconds": campaign.SCBA_SEGMENT_WALL_LIMIT_SECONDS + 0.1,
        "peak_observed_rss_bytes": 1024,
        "watchdog_trigger": "wall-timeout",
        "killed_after_term_grace": False,
        "wall_limit_seconds": campaign.SCBA_SEGMENT_WALL_LIMIT_SECONDS,
        "rss_limit_bytes": campaign.SCBA_RSS_LIMIT_BYTES,
    }


def test_planned_segment_stop_publishes_chained_resumable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    manifest = _segmented_manifest(artifacts)
    manifest_path = artifacts / "campaign-manifest.json"
    _write(manifest_path, manifest)
    log1 = artifacts / "logs" / "scba.json"
    _write(log1, {"attempt": 1})
    monkeypatch.setattr(
        campaign, "_load_segment_checkpoint", lambda path: _segment_checkpoint(17)
    )

    status1 = campaign._publish_segment_stop(
        artifacts,
        manifest,
        manifest_path,
        log1,
        attempt=1,
        resume=False,
        supervision=_segment_supervision(),
    )
    assert status1["state"] == "segment-exhausted"
    assert status1["segment_index"] == 1
    record1_path = artifacts / "segments" / "segment-001.json"
    record1 = campaign._load_json(record1_path, "test segment 1")
    assert record1["previous_segment"] is None
    assert record1["checkpoint"]["completed_iterations"] == 17
    assert record1["automatic_advance"] is False

    # A later scheduler job must start from the exact checkpoint captured by
    # the immediately preceding receipt.
    assert campaign._validate_segment_chain(
        artifacts, manifest, manifest_path
    ) == (record1_path,)

    log2 = artifacts / "logs" / "scba-resume-001.json"
    _write(log2, {"attempt": 2})
    (artifacts / "checkpoint" / "sigma-pp.h5").write_bytes(b"checkpoint-2")
    monkeypatch.setattr(
        campaign, "_load_segment_checkpoint", lambda path: _segment_checkpoint(31)
    )
    status2 = campaign._publish_segment_stop(
        artifacts,
        manifest,
        manifest_path,
        log2,
        attempt=2,
        resume=True,
        supervision=_segment_supervision(),
    )
    record2_path = artifacts / "segments" / "segment-002.json"
    record2 = campaign._load_json(record2_path, "test segment 2")
    assert status2["segment_index"] == 2
    assert record2["previous_segment"] == {
        "path": "segments/segment-001.json",
        "sha256": campaign._sha256(record1_path),
    }
    assert campaign._validate_segment_chain(
        artifacts, manifest, manifest_path
    ) == (record1_path, record2_path)


def test_planned_segment_chain_rejects_checkpoint_or_receipt_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    manifest = _segmented_manifest(artifacts)
    manifest_path = artifacts / "campaign-manifest.json"
    _write(manifest_path, manifest)
    log = artifacts / "logs" / "scba.json"
    _write(log, {"attempt": 1})
    monkeypatch.setattr(
        campaign, "_load_segment_checkpoint", lambda path: _segment_checkpoint(8)
    )
    campaign._publish_segment_stop(
        artifacts,
        manifest,
        manifest_path,
        log,
        attempt=1,
        resume=False,
        supervision=_segment_supervision(),
    )

    (artifacts / "checkpoint" / "sigma-pp.h5").write_bytes(b"changed")
    with pytest.raises(campaign.common.CampaignError, match="checkpoint changed"):
        campaign._validate_segment_chain(artifacts, manifest, manifest_path)


def test_last_planned_segment_cannot_authorize_another_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    manifest = _segmented_manifest(artifacts)
    manifest_path = artifacts / "campaign-manifest.json"
    _write(manifest_path, manifest)
    log = artifacts / "logs" / "scba-resume-003.json"
    _write(log, {"attempt": campaign.SCBA_MAX_SEGMENTS})
    monkeypatch.setattr(
        campaign, "_load_segment_checkpoint", lambda path: _segment_checkpoint(99)
    )
    with pytest.raises(campaign.common.CampaignError, match="budget exhausted"):
        campaign._publish_segment_stop(
            artifacts,
            manifest,
            manifest_path,
            log,
            attempt=campaign.SCBA_MAX_SEGMENTS,
            resume=True,
            supervision=_segment_supervision(),
        )


def test_scba_supervisor_receives_manifest_watchdog_limits(
    tmp_path: Path, monkeypatch,
) -> None:
    observed: dict[str, object] = {}

    def supervise(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return {"returncode": 0}

    monkeypatch.setattr(campaign.common, "_supervised_subprocess", supervise)
    result = campaign._supervise_scba(
        ("python", "driver.py"),
        repo=tmp_path,
        stdout_path=tmp_path / "stdout",
        stderr_path=tmp_path / "stderr",
    )
    assert result == {"returncode": 0}
    assert observed["wall_limit_seconds"] == campaign.SCBA_WALL_LIMIT_SECONDS
    assert observed["rss_limit_bytes"] == campaign.SCBA_RSS_LIMIT_BYTES


def test_infrastructure_resume_uses_append_only_attempt_artifacts(
    tmp_path: Path,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    manifest = artifacts / "campaign-manifest.json"
    _write(manifest, {"schema": campaign.SCHEMA})

    attempt1, log1, failure1, sums1 = campaign._attempt_paths(artifacts)
    assert attempt1 == 1
    _write(log1, {"attempt": 1})
    first = campaign._infrastructure_failure(
        artifacts,
        manifest,
        log1,
        failure1,
        sums1,
        "first interruption",
        {"returncode": -15},
    )
    assert first["state"] == "infrastructure-interrupted"

    attempt2, log2, failure2, sums2 = campaign._attempt_paths(artifacts)
    assert attempt2 == 2
    assert log2.name == "scba-resume-001.json"
    _write(log2, {"attempt": 2})
    second = campaign._infrastructure_failure(
        artifacts,
        manifest,
        log2,
        failure2,
        sums2,
        "second interruption",
        {"returncode": -15},
    )
    assert second["state"] == "infrastructure-interrupted"

    assert log1.is_file() and log2.is_file()
    assert failure1.is_file() and failure2.is_file()
    assert sums1.is_file() and sums2.is_file()
    assert len(campaign._attempt_logs(artifacts)) == 2
    assert len(campaign._failure_records(artifacts)) == 2
    assert campaign._attempt_paths(artifacts)[0] == 3


@pytest.mark.parametrize("research_scout", [False, True])
def test_manifest_and_execution_command_preserve_predictor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, research_scout: bool,
) -> None:
    artifacts = tmp_path / "campaign"
    output = {
        "checkpoint": str(artifacts / "checkpoint/sigma-pp.h5"),
        "json": str(artifacts / "outcomes/scba.json"),
        "hdf5": str(artifacts / "outcomes/scba-fields.h5"),
    }
    parent = {
        "artifacts": str(tmp_path / "parent"),
        "initialization": {"predictor": "coupling-square"},
    }
    manifest = {
        "schema": campaign.SCHEMA,
        "source": {"commit": "candidate"},
        "source_tree_sha256": "a" * 64,
        "environment_lock": {"sha256": "b" * 64},
        "driver": {"sha256": "c" * 64},
        "preflight": {"continuation_parent": parent},
        "run_id": "child",
        "coupling_scale": 0.02,
        "continuation_parent": parent,
        "continuation_axis": "coupling",
        "continuation_predictor": "coupling-square",
        "born_initial": None,
        "numerical_controls": {
            "grid_step": 0.1,
            "grid_n_positive": 75,
            "grid_cutoff": 7.5,
            "eta_lead": 0.025,
            "mixing": 0.2,
            "mixing_floor": 0.001,
            "temperature_bias": 0.0,
        },
        "outputs": output,
    }
    monkeypatch.setattr(campaign, "_manifest", lambda **kwargs: manifest)
    observed: dict[str, object] = {}

    def supervise(command, **kwargs):
        observed["command"] = command
        observed["wall_limit_seconds"] = kwargs["wall_limit_seconds"]
        kwargs["stdout_path"].write_text("")
        kwargs["stderr_path"].write_text("")
        return {"returncode": 2, "watchdog_trigger": None}

    monkeypatch.setattr(campaign, "_supervise_scba", supervise)
    monkeypatch.setattr(
        campaign, "_publish_report", lambda *args: {"state": "passed"}
    )
    status = campaign.run_campaign(
        repo=tmp_path,
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3",
        phase_b=tmp_path / "phase-b",
        artifacts=artifacts,
        expected_commit="candidate",
        environment_lock=tmp_path / "lock",
        run_id="child",
        coupling_scale=0.02,
        continuation_parent_artifacts=tmp_path / "parent",
        continuation_axis="coupling",
        continuation_predictor="coupling-square",
        research_scout=research_scout,
    )
    assert status == {"state": "passed"}
    command = list(observed["command"])
    index = command.index("--continuation-predictor")
    assert command[index + 1] == "coupling-square"
    assert "--no-resume" in command

    assert ("--research-scout" in command) is research_scout
    assert observed["wall_limit_seconds"] == (10800 if research_scout else 82800)


def test_resume_rejects_changed_predictor_before_checkpoint_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    source = {"commit": "candidate"}
    lock = {"sha256": "b" * 64}
    _write(artifacts / "campaign-manifest.json", {
        "schema": campaign.SCHEMA,
        "source": source,
        "environment_lock": lock,
        "driver": {"sha256": "c" * 64},
        "coupling_scale": 0.02,
        "continuation_axis": "coupling",
        "continuation_predictor": "identity",
    })
    recorded = campaign._load_json(
        artifacts / "campaign-manifest.json", "test manifest"
    )
    expected = dict(recorded)
    expected["continuation_predictor"] = "coupling-square"
    monkeypatch.setattr(campaign, "_manifest", lambda **kwargs: expected)
    with pytest.raises(
        campaign.common.CampaignError, match="exact requested identity"
    ):
        campaign.run_campaign(
            repo=tmp_path,
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3",
            phase_b=tmp_path / "phase-b",
            artifacts=artifacts,
            expected_commit="candidate",
            environment_lock=tmp_path / "lock",
            run_id="child",
            coupling_scale=0.02,
            continuation_parent_artifacts=tmp_path / "parent",
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
            resume=True,
        )


@pytest.mark.parametrize("changed", ["run_id", "source", "input"])
def test_terminal_status_never_bypasses_exact_request_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    changed: str,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    expected = {
        "schema": campaign.SCHEMA,
        "source": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "dirty": False,
        },
        "run_id": "requested-run",
        "preflight": {
            "source": {
                "fixture_path": "/inputs/fixture",
                "problem_input_hashes": {"problem": "3" * 64},
            },
            "run_id": "requested-run",
        },
    }
    recorded = deepcopy(expected)
    if changed == "run_id":
        recorded["run_id"] = "other-run"
    elif changed == "source":
        recorded["source"]["tree"] = "4" * 40
    else:
        recorded["preflight"]["source"]["fixture_path"] = "/other/fixture"
    _write(artifacts / "campaign-manifest.json", recorded)
    _write(artifacts / "status.json", {"state": "passed"})
    calls: list[bool] = []

    def requested_manifest(**kwargs):
        calls.append(bool(kwargs.get("allow_existing_outputs")))
        return expected

    monkeypatch.setattr(campaign, "_manifest", requested_manifest)
    monkeypatch.setattr(
        campaign,
        "_authenticated_terminal_status",
        lambda *args: pytest.fail("terminal status used before request authentication"),
    )
    with pytest.raises(
        campaign.common.CampaignError, match="exact requested identity"
    ):
        campaign.run_campaign(
            repo=tmp_path,
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3",
            phase_b=tmp_path / "phase-b",
            artifacts=artifacts,
            expected_commit="1" * 40,
            environment_lock=tmp_path / "lock",
            run_id="requested-run",
            coupling_scale=0.02,
        )
    assert calls == [True]


def test_existing_nonterminal_manifest_requires_explicit_resume_even_without_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = tmp_path / "campaign"
    artifacts.mkdir()
    manifest = {
        "schema": campaign.SCHEMA,
        "run_id": "abrupt-run",
        "preflight": {"run_id": "abrupt-run"},
    }
    _write(artifacts / "campaign-manifest.json", manifest)
    checkpoint = artifacts / "checkpoint/sigma-pp.h5"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"abrupt checkpoint\n")
    monkeypatch.setattr(campaign, "_manifest", lambda **kwargs: manifest)
    monkeypatch.setattr(
        campaign,
        "_supervise_scba",
        lambda *args, **kwargs: pytest.fail("nonexplicit resume ran the child"),
    )
    with pytest.raises(
        campaign.common.CampaignError, match="requires explicit --resume"
    ):
        campaign.run_campaign(
            repo=tmp_path,
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3",
            phase_b=tmp_path / "phase-b",
            artifacts=artifacts,
            expected_commit="1" * 40,
            environment_lock=tmp_path / "lock",
            run_id="abrupt-run",
            coupling_scale=0.02,
        )


def test_qualified_scba_supervisor_preserves_complete_launch_environment(tmp_path, monkeypatch):
    import json
    from negfpy.anharmonic._launch_environment import build_launch_environment

    executable = Path(sys.executable).resolve()
    environment = build_launch_environment(
        repo=tmp_path, executable=executable, home=tmp_path,
        tmpdir=tmp_path, git_bin=Path('/usr/bin'),
    ) | {'NEGFPY_QUALIFICATION_RUNTIME_SHA256': 'a' * 64}
    monkeypatch.setenv('UNRELATED_SECRET', 'must-not-reach-child')
    monkeypatch.setattr(campaign.common, 'RSS_POLL_SECONDS', 0.01)
    stdout = tmp_path / 'stdout'
    result = campaign._supervise_scba(
        (str(executable), '-c', 'import json,os; print(json.dumps(dict(os.environ),sort_keys=True))'),
        repo=tmp_path, stdout_path=stdout, stderr_path=tmp_path / 'stderr',
        launch_environment=environment,
    )
    assert result['returncode'] == 0
    assert json.loads(stdout.read_text()) == environment


def test_qualified_scba_supervisor_rejects_environment_drift_before_launch(tmp_path):
    from negfpy.anharmonic._launch_environment import build_launch_environment

    executable = Path(sys.executable).resolve()
    environment = build_launch_environment(
        repo=tmp_path, executable=executable, home=tmp_path,
        tmpdir=tmp_path, git_bin=Path('/usr/bin'),
    ) | {'NEGFPY_QUALIFICATION_RUNTIME_SHA256': 'a' * 64, 'LD_PRELOAD': '/unexpected.so'}
    stdout = tmp_path / 'stdout'
    with pytest.raises(ValueError, match='sanitized policy'):
        campaign._supervise_scba(
            (str(executable), '-c', 'raise SystemExit(99)'),
            repo=tmp_path, stdout_path=stdout, stderr_path=tmp_path / 'stderr',
            launch_environment=environment,
        )
    assert not stdout.exists()


def _scout_manifest(artifacts):
    manifest = _segmented_manifest(artifacts)
    manifest.pop("segmentation")
    manifest["research_scout"] = {
        "enabled": True, "max_iterations": 100, "max_wall_seconds": 10800,
        "resume_allowed": False, "automatic_advance": False,
    }
    manifest["preflight"].update({
        "configuration_sha256": "d" * 64,
        "controls": {"mixing_method": "anderson", "max_iterations": 100},
    })
    manifest["outputs"].update({"json": str(artifacts / "outcomes/scba.json"),
                                 "hdf5": str(artifacts / "outcomes/scba-fields.h5")})
    return manifest


def _scout_checkpoint(count=17):
    checkpoint = _segment_checkpoint(count)
    checkpoint.identity.configuration_sha256 = "d" * 64
    checkpoint.history = tuple({"iteration": i + 1, "sigma_residual": 0.1}
                               for i in range(count))
    return checkpoint


def test_scout_timeout_is_closed_nonresumable_diagnostic_not_success(tmp_path, monkeypatch):
    manifest = _scout_manifest(tmp_path)
    manifest_path = tmp_path / "campaign-manifest.json"
    _write(manifest_path, manifest)
    _write(tmp_path / "status-running.json", {"state": "running"})
    log = tmp_path / "logs/scba.json"
    supervision = _segment_supervision() | {"wall_limit_seconds": 10800, "elapsed_seconds": 10800.1}
    _write(log, {"supervision": supervision})
    monkeypatch.setattr(campaign, "_load_segment_checkpoint", lambda _: _scout_checkpoint())
    status = campaign._publish_scout_timeout(tmp_path, manifest, manifest_path, log, supervision)
    assert status["state"] == "scout-exhausted"
    assert status["stop_reason"] == "wall-timeout"
    assert status["scba_converged"] is False
    assert status["resume_allowed"] is False
    assert status["release_certified"] is False
    assert (tmp_path / "status.json").is_file()
    report = campaign._load_json(tmp_path / "scba-report.json", "report")
    assert report["iteration_summary"]["completed_iterations"] == 17
    assert report["checkpoint_sha256"] == campaign._sha256(tmp_path / "checkpoint/sigma-pp.h5")


@pytest.mark.parametrize("fault", ["config", "identity", "budget", "wall", "reason"])
def test_scout_stop_rejects_wrong_identity_budget_or_reason(tmp_path, monkeypatch, fault):
    manifest = _scout_manifest(tmp_path)
    checkpoint = _scout_checkpoint()
    supervision = _segment_supervision() | {"wall_limit_seconds": 10800}
    if fault == "config": checkpoint.identity.configuration_sha256 = "e" * 64
    if fault == "identity": checkpoint.identity.run_id = "other"
    if fault == "budget": checkpoint = _scout_checkpoint(101)
    if fault == "wall": supervision["wall_limit_seconds"] = 72000
    if fault == "reason": supervision["watchdog_trigger"] = "rss-limit"
    monkeypatch.setattr(campaign, "_load_segment_checkpoint", lambda _: checkpoint)
    with pytest.raises(campaign.common.CampaignError, match="research scout"):
        campaign._publish_scout_timeout(tmp_path, manifest, tmp_path / "manifest", tmp_path / "log", supervision)
    assert not (tmp_path / "status.json").exists()


@pytest.mark.parametrize("resume,segmented,existing", [(True, False, False), (False, True, False), (False, False, True)])
def test_scout_rejects_resume_segmentation_or_consumed_output_before_io(tmp_path, resume, segmented, existing):
    artifacts = tmp_path / "scout"
    if existing:
        artifacts.mkdir()
        (artifacts / "status.json").write_text("{}")
    with pytest.raises(campaign.common.CampaignError, match="research scout"):
        campaign.run_campaign(repo=tmp_path, fixture=tmp_path / "f", fc3_full=tmp_path / "fc3",
                              phase_b=tmp_path / "phase-b", artifacts=artifacts,
                              expected_commit="candidate", environment_lock=tmp_path / "lock",
                              run_id="scout", coupling_scale=0.0175,
                              research_scout=True, resume=resume, segmented=segmented)


def test_scout_iteration_cap_reports_exhausted_not_scientific_pass(tmp_path, monkeypatch):
    manifest = _scout_manifest(tmp_path)
    manifest_path = tmp_path / "campaign-manifest.json"
    running = tmp_path / "status-running.json"
    log = tmp_path / "logs/scba.json"
    _write(manifest_path, manifest)
    _write(running, {"state": "running"})
    _write(log, {"supervision": {"returncode": 2, "watchdog_trigger": None}})
    _write(Path(manifest["outputs"]["json"]), {
        "campaign_manifest_sha256": campaign._sha256(manifest_path), "scba_executed": True,
        "result": {"converged": False, "scientific_error": {"type": "SCBAConvergenceError"}},
    })
    Path(manifest["outputs"]["hdf5"]).write_bytes(b"fields")
    checkpoint = _scout_checkpoint(100)
    checkpoint.status = "scientific_failure"
    checkpoint.terminal = True
    monkeypatch.setattr(campaign, "_load_segment_checkpoint", lambda _: checkpoint)
    result = campaign._publish_report(tmp_path, manifest_path, running, log)
    assert result["state"] == "scout-exhausted"
    assert result["stop_reason"] == "iteration-limit"
    assert result["scba_converged"] is False
    assert result["resume_allowed"] is False
