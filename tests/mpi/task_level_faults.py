"""Bounded real-MPI fault and changed-worker-count fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from mpi4py import MPI
import numpy as np

from negfpy.anharmonic import _mpi_tasks as mpi_tasks
from negfpy.anharmonic._mpi_tasks import (
    TaskContext,
    TaskManifest,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
)
from negfpy.anharmonic.toy import _mpi_production_gate as campaign_tasks
from negfpy.anharmonic.toy._mpi_production_gate import (
    ProductionToyCaseResult,
    ProductionToyMPICampaign,
    execute_production_toy_mpi_campaign,
)


def _sha(character: str) -> str:
    return character * 64


def _manifest() -> TaskManifest:
    return TaskManifest(
        source_revision="mpi-fault-fixture-v1",
        source_tree_sha256=_sha("a"),
        environment_sha256=_sha("b"),
        identity_hashes={"campaign_sha256": _sha("c")},
        tasks=tuple(
            TaskSpec(
                task_id=f"q-point-{index}",
                identity_hashes={
                    "problem_sha256": _sha(str(index + 1)),
                    "grid_sha256": _sha(str(index + 3)),
                },
                payload={"grid_point": index},
            )
            for index in range(6)
        ),
    )



def _toy_campaign() -> ProductionToyMPICampaign:
    preflight_a = TaskSpec(
        "preflight-a",
        {"problem_sha256": _sha("1")},
        {
            "kind": "fixture",
            "dependencies": [],
            "continuation_parent_task_id": None,
            "ordinal": 0,
        },
    )
    preflight_b = TaskSpec(
        "preflight-b",
        {"problem_sha256": _sha("2")},
        {
            "kind": "fixture",
            "dependencies": [],
            "continuation_parent_task_id": None,
            "ordinal": 1,
        },
    )
    gate = TaskSpec(
        "direct-zero",
        {"problem_sha256": _sha("3")},
        {
            "kind": "fixture",
            "dependencies": ["preflight-a", "preflight-b"],
            "continuation_parent_task_id": None,
            "ordinal": 2,
        },
    )
    final = TaskSpec(
        "final-case",
        {"problem_sha256": _sha("4")},
        {
            "kind": "fixture",
            "dependencies": ["direct-zero"],
            "continuation_parent_task_id": None,
            "ordinal": 3,
        },
    )
    manifest = TaskManifest(
        source_revision="mpi-campaign-fault-fixture-v1",
        source_tree_sha256=_sha("d"),
        environment_sha256=_sha("e"),
        identity_hashes={"campaign_sha256": _sha("f")},
        tasks=(preflight_a, preflight_b, gate, final),
    )
    return ProductionToyMPICampaign(
        manifest=manifest,
        preflight_task_ids=("preflight-a", "preflight-b"),
        gate_task_id="direct-zero",
        waves=(
            ("preflight-a", "preflight-b"),

            ("direct-zero",),
            ("final-case",),
        ),
    )

def _validator(path: Path, task: TaskSpec) -> dict[str, object]:
    if (
        MODE == "root-validation-error"
        and MPI.COMM_WORLD.Get_rank() == 0
    ):
        raise RuntimeError("injected rank-zero artifact validation error")
    expected = (task.task_id + "\n").encode("ascii")
    if path.read_bytes() != expected:
        raise ValueError("artifact content mismatch")
    return {"content_sha256": hashlib.sha256(expected).hexdigest()}


def _worker(task: TaskSpec, context: TaskContext) -> TaskResult:
    if MODE == "worker-error" and task.task_id == "q-point-0":
        raise RuntimeError("injected worker infrastructure error")
    if MODE == "partial" and task.task_id == "q-point-2":
        raise RuntimeError("deterministic partial-campaign interruption")

    marker_root = RUN_DIRECTORY / "worker-invocations"
    marker_root.mkdir(parents=True, exist_ok=True)
    partial_marker = marker_root / f"{task.task_id}.partial"
    if MODE == "resume" and partial_marker.exists():
        raise AssertionError(
            f"completed partial task reran: {task.task_id}"
        )
    marker = marker_root / f"{task.task_id}.{MODE}"
    marker.write_text(f"rank={context.rank}\n", encoding="ascii")
    artifact = context.write_artifact_bytes(
        (task.task_id + "\n").encode("ascii"),
        suffix=".txt",
    )
    return TaskResult.completed(artifact)



def _campaign_executor(task, context) -> ProductionToyCaseResult:
    return ProductionToyCaseResult.completed(
        evidence={"task_id": task.task_id},
        arrays={"state": np.asarray([task.payload["ordinal"]])},
    )


def _run_campaign_fault() -> int:
    campaign = _toy_campaign()
    rank = MPI.COMM_WORLD.Get_rank()
    original_collect = campaign_tasks.collect_task_checkpoints
    original_final = campaign_tasks._collect_campaign_outcomes
    if MODE == "campaign-phase-validation-error" and rank == 0:

        def fail_phase_collect(*args, **kwargs):
            raise RuntimeError(
                "injected rank-zero campaign phase validation error"
            )

        campaign_tasks.collect_task_checkpoints = fail_phase_collect
    if MODE == "campaign-final-build-error" and rank == 0:

        def fail_final_collect(*args, **kwargs):
            raise RuntimeError(
                "injected rank-zero campaign final build error"
            )

        campaign_tasks._collect_campaign_outcomes = fail_final_collect

    caught: str | None = None
    try:
        execute_production_toy_mpi_campaign(
            campaign,
            _campaign_executor,
            RUN_DIRECTORY,
            comm=MPI.COMM_WORLD,
            require_mpi=True,
            execution_id=f"{MODE}-{MPI.COMM_WORLD.Get_size()}-ranks",
        )
    except Exception as exc:
        caught = type(exc).__name__
    finally:
        if rank == 0:
            campaign_tasks.collect_task_checkpoints = original_collect
            campaign_tasks._collect_campaign_outcomes = original_final

    caught_by_rank = MPI.COMM_WORLD.allgather(caught)
    if any(value is None for value in caught_by_rank):
        raise AssertionError(
            "not every rank observed campaign terminal failure: "
            f"{caught_by_rank}"
        )
    _emit(
        {
            "caught_by_rank": caught_by_rank,
            "completed": [],
            "mode": MODE,
        }
    )
    return 0

def _physics_sha256(manifest: TaskManifest) -> str:
    checkpoints = collect_task_checkpoints(
        manifest,
        RUN_DIRECTORY / "task-outcomes",
        RUN_DIRECTORY / "task-artifacts",
        artifact_validator=_validator,
        require_complete=True,
    )
    document = [
        {
            "task_id": checkpoint.task_id,
            "status": checkpoint.status,
            "artifact_validation": dict(checkpoint.artifact_validation),
        }
        for checkpoint in checkpoints
    ]
    payload = json.dumps(
        document,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _emit(document: dict[str, object]) -> None:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(json.dumps(document, sort_keys=True), flush=True)


def main() -> int:
    if MODE in {
        "campaign-phase-validation-error",
        "campaign-final-build-error",
    }:
        return _run_campaign_fault()

    manifest = _manifest()
    if MODE == "unexpected-worker-message":
        if MPI.COMM_WORLD.Get_size() != 2:
            raise AssertionError("unexpected-message fixture requires 2 ranks")
        if MPI.COMM_WORLD.Get_rank() == 0:
            MPI.COMM_WORLD.send("injected", dest=1, tag=4999)

    if MODE == "root-build-error" and MPI.COMM_WORLD.Get_rank() == 0:
        original_build = mpi_tasks._build_report

        def fail_build(*args, **kwargs):
            raise RuntimeError("injected rank-zero report build error")

        mpi_tasks._build_report = fail_build
    else:
        original_build = None

    caught: str | None = None
    report = None
    try:
        report = execute_task_manifest(
            manifest,
            _worker,
            RUN_DIRECTORY,
            comm=MPI.COMM_WORLD,
            require_mpi=True,
            artifact_validator=_validator,
            execution_id=f"{MODE}-{MPI.COMM_WORLD.Get_size()}-ranks",
        )
    except Exception as exc:
        caught = type(exc).__name__
    finally:
        if original_build is not None:
            mpi_tasks._build_report = original_build

    caught_by_rank = MPI.COMM_WORLD.allgather(caught)
    fault_modes = {
        "worker-error",
        "unexpected-worker-message",
        "root-validation-error",
        "root-build-error",
        "partial",
    }
    if MODE in fault_modes:
        if any(value is None for value in caught_by_rank):
            raise AssertionError(
                f"not every rank observed terminal failure: {caught_by_rank}"
            )
        completed: list[str] = []
        if MPI.COMM_WORLD.Get_rank() == 0 and MODE == "partial":
            completed = [
                item.task_id
                for item in collect_task_checkpoints(
                    manifest,
                    RUN_DIRECTORY / "task-outcomes",
                    RUN_DIRECTORY / "task-artifacts",
                    artifact_validator=_validator,
                    require_complete=False,
                )
            ]
            if completed != ["q-point-0", "q-point-1"]:
                raise AssertionError(
                    f"unexpected partial checkpoint set: {completed}"
                )
        _emit(
            {
                "caught_by_rank": caught_by_rank,
                "completed": completed,
                "mode": MODE,
            }
        )
        return 0

    if caught is not None or (MPI.COMM_WORLD.Get_rank() == 0 and report is None):
        raise AssertionError(f"unexpected execution failure: {caught_by_rank}")
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert report is not None
        resumed = [
            item.task_id for item in report.assignments if item.resumed
        ]
        if MODE == "resume" and resumed != ["q-point-0", "q-point-1"]:
            raise AssertionError(f"wrong resumed task set: {resumed}")
        _emit(
            {
                "mode": MODE,
                "physics_sha256": _physics_sha256(manifest),
                "resumed": resumed,
            }
        )
    return 0


if len(sys.argv) != 3:
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("usage: task_level_faults.py MODE RUN_DIRECTORY", file=sys.stderr)
    raise SystemExit(2)

MODE = sys.argv[1]
RUN_DIRECTORY = Path(sys.argv[2]).expanduser().resolve()

if __name__ == "__main__":
    raise SystemExit(main())
