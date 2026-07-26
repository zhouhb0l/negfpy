from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from negfpy.anharmonic import _mpi_tasks as mpi_tasks
from negfpy.anharmonic._mpi_tasks import (
    DuplicateTaskCheckpointError,
    MissingTaskCheckpointError,
    TaskCheckpointCorruptionError,
    TaskCheckpointMismatchError,
    TaskManifest,
    TaskSpec,
    serial_task_communicator,
)
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy import _mpi_production_gate as mpi_campaign
from negfpy.anharmonic.toy._mpi_production_gate import (
    ProductionToyCaseResult,
    ProductionToyMPICampaign,
    build_production_toy_mpi_campaign,
    collect_production_toy_campaign_outcomes,
    execute_production_toy_mpi_campaign,
)
from negfpy.anharmonic.toy.multidimensional_release import (
    pr_multidimensional_release_config,
)
from negfpy.anharmonic.toy.open_three_mode import OpenThreeModeFixture


def _sha(index: int) -> str:
    return f"{index:064x}"


def _task(
    task_id: str,
    index: int,
    dependencies: tuple[str, ...],
    *,
    parent: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        identity_hashes={
            "problem_sha256": _sha(index + 1),
            "configuration_sha256": _sha(index + 20),
            "grid_sha256": _sha(index + 40),
        },
        payload={
            "kind": "fake_case",
            "dependencies": list(dependencies),
            "continuation_parent_task_id": parent,
            "ordinal": index,
        },
    )


def _campaign() -> ProductionToyMPICampaign:
    harmonic = "preflight.harmonic"
    backend = "preflight.backend"
    gate = "direct-zero"
    path0 = "multidimensional.path_0"
    path1 = "multidimensional.path_1"
    born = "born"
    refinement = "refinement"
    open_case = "open"
    tasks = (
        _task(harmonic, 0, ()),
        _task(backend, 1, ()),
        _task(gate, 2, (harmonic, backend)),
        _task(path0, 3, (gate,)),
        _task(path1, 4, (gate, path0), parent=path0),
        _task(born, 5, (gate,)),
        _task(refinement, 6, (gate,)),
        _task(open_case, 7, (gate,)),
    )
    manifest = TaskManifest(
        source_revision="0123456789abcdef",
        source_tree_sha256=_sha(100),
        environment_sha256=_sha(101),
        identity_hashes={"protocol_sha256": _sha(102)},
        metadata={"protocol": "fake_protocol_v2"},
        tasks=tasks,
    )
    return ProductionToyMPICampaign(
        manifest=manifest,
        preflight_task_ids=(harmonic, backend),
        gate_task_id=gate,
        waves=(
            (harmonic, backend),
            (gate,),
            (path0,),
            (path1,),
            (born, refinement, open_case),
        ),
    )


def _successful_executor(calls: list[str]):
    def execute(task, context):
        calls.append(task.task_id)
        parent = task.payload.get("continuation_parent_task_id")
        evidence: dict[str, object] = {"task_id": task.task_id}
        if parent is not None:
            assert context.continuation_parent_artifact is not None
            state = context.read_dependency_array(parent, "state")
            evidence["parent_state"] = int(state[0])
        return ProductionToyCaseResult.completed(
            evidence=evidence,
            arrays={"state": np.asarray([task.payload["ordinal"]])},
        )

    return execute


def test_static_dag_order_exact_parent_and_canonical_merge(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    calls: list[str] = []
    report = execute_production_toy_mpi_campaign(
        campaign,
        _successful_executor(calls),
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="first",
    )

    assert report is not None and report.complete
    assert calls == list(campaign.canonical_task_ids)
    assert [outcome.task_id for outcome in report.outcomes] == list(
        campaign.canonical_task_ids
    )
    assert all(outcome.status == "completed" for outcome in report.outcomes)
    assert all(outcome.artifact_relative_path for outcome in report.outcomes)
    assert (tmp_path / "campaign-manifest.json").is_file()
    assert (tmp_path / "campaign-executions" / "first.json").is_file()

    merged = collect_production_toy_campaign_outcomes(campaign, tmp_path)
    assert [outcome.task_id for outcome in merged] == list(
        campaign.canonical_task_ids
    )
    path1 = next(
        outcome
        for outcome in report.outcomes
        if outcome.task_id == "multidimensional.path_1"
    )
    assert path1.artifact_validation["array_sha256"]["state"]


def test_failed_direct_endpoint_blocks_every_remaining_case(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    calls: list[str] = []

    def executor(task, context):
        calls.append(task.task_id)
        if task.task_id == campaign.gate_task_id:
            return ProductionToyCaseResult.scientific_failure(
                evidence={"reason": "fixed-point gate failed"},
                arrays={"state": np.asarray([2])},
            )
        return ProductionToyCaseResult.completed(
            evidence={"task_id": task.task_id},
            arrays={"state": np.asarray([task.payload["ordinal"]])},
        )

    report = execute_production_toy_mpi_campaign(
        campaign,
        executor,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="blocked-by-gate",
    )
    assert report is not None and report.complete
    assert calls == [*campaign.preflight_task_ids, campaign.gate_task_id]
    statuses = {outcome.task_id: outcome.status for outcome in report.outcomes}
    assert statuses[campaign.gate_task_id] == "scientific_failure"
    assert all(
        statuses[task_id] == "blocked"
        for task_id in campaign.canonical_task_ids[3:]
    )
    assert all(
        outcome.artifact_relative_path is None
        for outcome in report.outcomes[3:]
    )


def test_continuation_failure_blocks_only_descendants(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    calls: list[str] = []

    def executor(task, context):
        calls.append(task.task_id)
        if task.task_id == "multidimensional.path_0":
            return ProductionToyCaseResult.scientific_failure(
                evidence={"reason": "continuation parent did not converge"},
                arrays={"state": np.asarray([3])},
            )
        return ProductionToyCaseResult.completed(
            evidence={"task_id": task.task_id},
            arrays={"state": np.asarray([task.payload["ordinal"]])},
        )

    report = execute_production_toy_mpi_campaign(
        campaign,
        executor,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="continuation-failure",
    )
    assert report is not None
    statuses = {outcome.task_id: outcome.status for outcome in report.outcomes}
    assert statuses["multidimensional.path_0"] == "scientific_failure"
    assert statuses["multidimensional.path_1"] == "blocked"
    assert statuses["born"] == "completed"
    assert statuses["refinement"] == "completed"
    assert statuses["open"] == "completed"
    assert "multidimensional.path_1" not in calls


def test_preflight_failure_blocks_gate_and_graph(tmp_path: Path) -> None:
    campaign = _campaign()
    calls: list[str] = []

    def executor(task, context):
        calls.append(task.task_id)
        if task.task_id == campaign.preflight_task_ids[1]:
            return ProductionToyCaseResult.scientific_failure(
                evidence={"reason": "backend equality failed"}
            )
        return ProductionToyCaseResult.completed(
            evidence={"task_id": task.task_id},
            arrays={"state": np.asarray([task.payload["ordinal"]])},
        )

    report = execute_production_toy_mpi_campaign(
        campaign,
        executor,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="preflight-failure",
    )
    assert report is not None
    assert calls == list(campaign.preflight_task_ids)
    assert report.outcomes[1].status == "scientific_failure"
    assert all(outcome.status == "blocked" for outcome in report.outcomes[2:])


def test_exact_resume_skips_cases_and_preserves_scientific_digest(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    calls: list[str] = []
    first = execute_production_toy_mpi_campaign(
        campaign,
        _successful_executor(calls),
        tmp_path,
        comm=serial_task_communicator(),
        thread_limit=1,
        execution_id="initial",
    )
    assert first is not None

    def forbidden(task, context):
        raise AssertionError(f"resumed task reran: {task.task_id}")

    resumed = execute_production_toy_mpi_campaign(
        campaign,
        forbidden,
        tmp_path,
        comm=serial_task_communicator(),
        thread_limit=2,
        execution_id="resumed-different-workers",
    )
    assert resumed is not None
    assert resumed.scientific_sha256 == first.scientific_sha256
    assert all(outcome.resumed for outcome in resumed.outcomes)
    assert calls == list(campaign.canonical_task_ids)


def test_missing_duplicate_corrupt_and_mismatched_records_are_rejected(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    report = execute_production_toy_mpi_campaign(
        campaign,
        _successful_executor([]),
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="complete",
    )
    assert report is not None
    first = report.outcomes[0]
    sidecar = tmp_path / first.checkpoint_relative_path
    original_sidecar = sidecar.read_bytes()

    sidecar.unlink()
    with pytest.raises(MissingTaskCheckpointError):
        collect_production_toy_campaign_outcomes(campaign, tmp_path)
    sidecar.write_bytes(original_sidecar)

    duplicate = sidecar.parent / "zz-duplicate.task.json"
    shutil.copyfile(sidecar, duplicate)
    with pytest.raises(DuplicateTaskCheckpointError):
        collect_production_toy_campaign_outcomes(campaign, tmp_path)
    duplicate.unlink()

    assert first.artifact_relative_path is not None
    artifact = tmp_path / first.artifact_relative_path
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(TaskCheckpointCorruptionError):
        collect_production_toy_campaign_outcomes(campaign, tmp_path)

    other_manifest = TaskManifest(
        source_revision=campaign.manifest.source_revision,
        source_tree_sha256=campaign.manifest.source_tree_sha256,
        environment_sha256=_sha(200),
        identity_hashes=campaign.manifest.identity_hashes,
        metadata=campaign.manifest.metadata,
        tasks=campaign.manifest.tasks,
    )
    other = ProductionToyMPICampaign(
        manifest=other_manifest,
        preflight_task_ids=campaign.preflight_task_ids,
        gate_task_id=campaign.gate_task_id,
        waves=campaign.waves,
    )
    with pytest.raises(
        (TaskCheckpointMismatchError, TaskCheckpointCorruptionError)
    ):
        collect_production_toy_campaign_outcomes(other, tmp_path)


def test_campaign_rejects_nonancestral_continuation_dependency() -> None:
    campaign = _campaign()
    invalid_waves = (
        campaign.waves[0],
        campaign.waves[1],
        (
            "multidimensional.path_0",
            "multidimensional.path_1",
        ),
        campaign.waves[-1],
    )
    with pytest.raises(ValueError, match="non-ancestral"):
        ProductionToyMPICampaign(
            manifest=campaign.manifest,
            preflight_task_ids=campaign.preflight_task_ids,
            gate_task_id=campaign.gate_task_id,
            waves=invalid_waves,
        )


def test_serial_fallback_without_mpi_and_bounded_protocol_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mpi_tasks, "_load_mpi", lambda *, required: None)
    campaign = _campaign()
    report = execute_production_toy_mpi_campaign(
        campaign,
        _successful_executor([]),
        tmp_path / "fallback",
        execution_id="fallback",
    )
    assert report is not None and report.mode == "serial"

    fixture = OpenThreeModeFixture(
        grid=SignedFrequencyGrid(step=0.5, n_positive=10),
        temperature=0.7,
        g=0.0,
        eta_lead=0.05,
    )
    bounded = build_production_toy_mpi_campaign(
        source_revision="0123456789abcdef",
        source_tree_sha256=_sha(300),
        environment_sha256=_sha(301),
        config=pr_multidimensional_release_config(),
        open_fixtures=(("bounded", fixture),),
    )
    assert bounded.manifest.metadata["multidimensional_outcome_count"] == 6
    assert bounded.manifest.metadata["open_system_case_count"] == 1
    assert bounded.waves[0] == bounded.preflight_task_ids
    assert bounded.waves[1] == (bounded.gate_task_id,)


def test_atomic_case_artifact_orphan_is_adopted_without_rerun(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign = _campaign()
    original_writer = mpi_campaign._write_case_artifact
    crash_task = campaign.preflight_task_ids[0]
    crashed = False

    def crash_after_publish(campaign_, task, context, result):
        nonlocal crashed
        path = original_writer(campaign_, task, context, result)
        if task.task_id == crash_task and not crashed:
            crashed = True
            raise RuntimeError(
                "deterministic crash after artifact publication"
            )
        return path

    monkeypatch.setattr(
        mpi_campaign,
        "_write_case_artifact",
        crash_after_publish,
    )
    with pytest.raises(
        mpi_tasks.TaskExecutionError,
        match="validated terminal checkpoint",
    ):
        execute_production_toy_mpi_campaign(
            campaign,
            _successful_executor([]),
            tmp_path,
            comm=serial_task_communicator(),
            execution_id="crash-window",
        )

    orphan_paths = tuple(tmp_path.glob("phases/*/task-artifacts/*.h5"))
    assert len(orphan_paths) == 1
    orphan = orphan_paths[0]
    original_bytes = orphan.read_bytes()
    original_mtime_ns = orphan.stat().st_mtime_ns
    assert not tuple(tmp_path.glob("phases/*/task-outcomes/*.task.json"))

    monkeypatch.setattr(
        mpi_campaign,
        "_write_case_artifact",
        original_writer,
    )
    resumed_calls: list[str] = []

    def resume_executor(task, context):
        if task.task_id == crash_task:
            raise AssertionError("validated orphan task was recomputed")
        return _successful_executor(resumed_calls)(task, context)

    report = execute_production_toy_mpi_campaign(
        campaign,
        resume_executor,
        tmp_path,
        comm=serial_task_communicator(),
        execution_id="resume-orphan",
    )
    assert report is not None and report.complete
    assert crash_task not in resumed_calls
    assert orphan.read_bytes() == original_bytes
    assert orphan.stat().st_mtime_ns == original_mtime_ns

    outcome = next(
        item for item in report.outcomes if item.task_id == crash_task
    )
    checkpoint = json.loads(
        (tmp_path / outcome.checkpoint_relative_path).read_text(
            encoding="ascii"
        )
    )
    assert checkpoint["payload"]["execution"][
        "orphan_artifact_adopted"
    ] is True
