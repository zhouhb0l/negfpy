"""Two-rank smoke test for the private task-level MPI coordinator.

Run from an environment containing the optional MPI dependency:

    mpiexec -n 2 python tests/mpi/task_level_smoke.py /shared/smoke/run
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

from mpi4py import MPI

from negfpy.anharmonic._mpi_tasks import (
    TaskContext,
    TaskManifest,
    TaskResult,
    TaskSpec,
    execute_task_manifest,
)


def _sha(character: str) -> str:
    return character * 64


def _manifest() -> TaskManifest:
    return TaskManifest(
        source_revision="mpi-smoke-fixture",
        source_tree_sha256=_sha("a"),
        environment_sha256=_sha("b"),
        identity_hashes={"campaign_sha256": _sha("c")},
        tasks=tuple(
            TaskSpec(
                f"q-point-{index}",
                {
                    "problem_sha256": _sha(str(index + 1)),
                    "grid_sha256": _sha(str(index + 4)),
                },
                {"grid_point": index},
            )
            for index in range(3)
        ),
    )


def _worker(task: TaskSpec, context: TaskContext) -> TaskResult:
    artifact = context.write_artifact_bytes(
        (task.task_id + "\n").encode("ascii"),
        suffix=".txt",
    )
    return TaskResult.completed(
        artifact,
        metadata={"rank": context.rank},
    )


def _validator(path: Path, task: TaskSpec) -> dict[str, object]:
    expected = (task.task_id + "\n").encode("ascii")
    if path.read_bytes() != expected:
        raise ValueError("smoke artifact content mismatch")
    return {"content_matches": True, "byte_count": len(expected)}


def main() -> int:
    if len(sys.argv) != 2:
        if MPI.COMM_WORLD.Get_rank() == 0:
            print("usage: task_level_smoke.py RUN_DIRECTORY", file=sys.stderr)
        return 2
    run_directory = Path(sys.argv[1]).expanduser().resolve()
    report = execute_task_manifest(
        _manifest(),
        _worker,
        run_directory,
        comm=MPI.COMM_WORLD,
        require_mpi=True,
        artifact_validator=_validator,
        execution_id="two-rank-smoke",
    )
    MPI.COMM_WORLD.Barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        if report is None or not report.complete:
            raise AssertionError("rank zero did not receive a complete report")
        if report.mode != "mpi" or report.communicator_size != 2:
            raise AssertionError("smoke did not use the two-rank MPI path")
        if [item.task_id for item in report.assignments] != [
            "q-point-0",
            "q-point-1",
            "q-point-2",
        ]:
            raise AssertionError("outcomes were not merged canonically")
        if {item.rank for item in report.assignments} != {1}:
            raise AssertionError("rank zero executed numerical worker tasks")
        print(
            json.dumps(
                {
                    "complete": report.complete,
                    "manifest_sha256": report.manifest_sha256,
                    "tasks": len(report.assignments),
                    "worker_ranks": sorted(
                        {item.rank for item in report.assignments}
                    ),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
