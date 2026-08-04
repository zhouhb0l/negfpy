#!/usr/bin/env python3
"""Small multi-rank smoke test for the material Nq=1 task graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mpi4py import MPI

from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
    execute_material_nq1_mpi_campaign,
    task_dependencies,
)
from negfpy.anharmonic._mpi_tasks import TaskResult


def _validator(path: Path, spec):
    payload = json.loads(path.read_text(encoding="ascii"))
    if payload != {"state": "completed", "task_id": spec.task_id}:
        raise ValueError("invalid smoke-test artifact")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--execution-id", required=True)
    args = parser.parse_args()
    campaign = build_material_nq1_mpi_campaign(
        source_revision="mpi-smoke",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )

    def executor(scientific, spec, context, dependencies):
        if set(dependencies) != set(task_dependencies(scientific)):
            raise RuntimeError("dependency artifacts do not match the DAG")
        payload = (
            json.dumps(
                {"state": "completed", "task_id": spec.task_id},
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("ascii")
        artifact = context.write_artifact_bytes(payload, suffix=".json")
        return TaskResult.completed(artifact)

    report = execute_material_nq1_mpi_campaign(
        campaign,
        executor,
        args.run_directory,
        artifact_validator=_validator,
        comm=MPI.COMM_WORLD,
        require_mpi=True,
        execution_id=args.execution_id,
    )
    if MPI.COMM_WORLD.Get_rank() == 0:
        if report is None or not report.complete or len(report.statuses) != 18:
            raise RuntimeError("MPI material smoke campaign is incomplete")
        print(json.dumps(dict(report.statuses), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
