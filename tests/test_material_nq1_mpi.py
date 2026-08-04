from __future__ import annotations

import json
from pathlib import Path

from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
    execute_material_nq1_mpi_campaign,
    task_dependencies,
)
from negfpy.anharmonic._mpi_tasks import TaskResult, serial_task_communicator


def _campaign():
    return build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )


def _validator(path, task):
    value = json.loads(path.read_text())
    assert value["task_id"] == task.task_id
    return {"task_id": task.task_id, "state": value["state"]}


def test_material_mpi_waves_preserve_dependencies_and_fanout() -> None:
    campaign = _campaign()
    assert len(campaign.scientific_tasks) == 18
    assert len(campaign.waves) == 10
    assert campaign.waves[:8] == tuple(
        (f"coupling-g{token}",)
        for token in ("001", "0025", "005", "010", "025", "050", "075", "100")
    )
    assert set(campaign.waves[8]) == {
        "bias-positive-half", "bias-negative-half", "endpoint-zero-start",
        "endpoint-born-start", "refinement-half-spacing",
        "refinement-larger-cutoff", "refinement-half-eta",
        "refinement-mixing-010",
    }
    assert campaign.waves[9] == ("bias-positive-full", "bias-negative-full")
    assert task_dependencies(campaign.task("endpoint-zero-start")) == (
        "coupling-g100",
    )


def test_serial_material_task_campaign_merges_in_canonical_order(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    seen: list[str] = []

    def executor(scientific, spec, context, dependencies):
        seen.append(scientific.task_id)
        assert set(dependencies) == set(task_dependencies(scientific))
        artifact = context.write_artifact_bytes(
            (json.dumps({"task_id": spec.task_id, "state": "completed"}, sort_keys=True) + "\n").encode(),
            suffix=".json",
        )
        return TaskResult.completed(artifact)

    report = execute_material_nq1_mpi_campaign(
        campaign,
        executor,
        tmp_path / "run",
        artifact_validator=_validator,
        comm=serial_task_communicator(),
        execution_id="serial-test",
    )
    assert report is not None and report.complete
    assert seen == [task.task_id for task in campaign.scientific_tasks[:8]] + list(
        campaign.waves[8]
    ) + list(campaign.waves[9])
    assert list(report.statuses) == [
        task.task_id for task in campaign.scientific_tasks
    ]


def test_scientific_failure_blocks_all_descendants_without_execution(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    executed: list[str] = []

    def executor(scientific, spec, context, dependencies):
        executed.append(scientific.task_id)
        state = (
            "scientific_failure"
            if scientific.task_id == "coupling-g025"
            else "completed"
        )
        artifact = context.write_artifact_bytes(
            (json.dumps({"task_id": spec.task_id, "state": state}, sort_keys=True) + "\n").encode(),
            suffix=".json",
        )
        return TaskResult(status=state, artifact_path=artifact)

    report = execute_material_nq1_mpi_campaign(
        campaign,
        executor,
        tmp_path / "run",
        artifact_validator=_validator,
        comm=serial_task_communicator(),
        execution_id="failure-test",
    )
    assert report is not None and not report.complete
    assert report.statuses["coupling-g025"] == "scientific_failure"
    assert report.statuses["coupling-g050"] == "blocked"
    assert report.statuses["coupling-g100"] == "blocked"
    assert report.statuses["endpoint-zero-start"] == "blocked"
    assert "coupling-g050" not in executed
