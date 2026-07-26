"""Bounded real-physics MPI bridge and changed-worker resume test.

Run from the environment that owns both ``mpiexec`` and ``mpi4py``::

    mpiexec -n 2 python tests/mpi/production_toy_physics_smoke.py RUN initial
    mpiexec -n 4 python tests/mpi/production_toy_physics_smoke.py RUN resume
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

from mpi4py import MPI

from negfpy.anharmonic._mpi_tasks import serial_task_communicator
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy._mpi_production_gate_physics import (
    run_production_toy_mpi_physics,
)
from negfpy.anharmonic.toy.multidimensional_release import (
    pr_multidimensional_release_config,
)
from negfpy.anharmonic.toy.open_three_mode import OpenThreeModeFixture


_SOURCE_REVISION = "mpi-physics-smoke-v1"
_SOURCE_TREE_SHA256 = "a" * 64
_ENVIRONMENT_SHA256 = "b" * 64
_TIMING_KEYS = {
    "asymptotic_dense_time",
    "elapsed_ratio_right_over_left",
    "elapsed_seconds",
    "maximum_wall_seconds",
    "median_wall_seconds",
    "minimum_wall_seconds",
    "oracle_wall_seconds",
    "production_born_wall_seconds",
    "production_wall_seconds",
    "reconstruction_seconds",
    "seed_elapsed_seconds",
    "total_elapsed_seconds",
    "total_wall_seconds",
    "wall_seconds",
}
_ORCHESTRATION_KEYS = {
    "checkpoint",
    "checkpoint_evidence",
    "orchestration_provenance",
    "report_sha256",
    "resumed",
}


def _config():
    return replace(
        pr_multidimensional_release_config(),
        executed_g_values=(0.0,),
        executed_bias_values=(0.0,),
        production_backend="fft",
        production_hilbert_backend="fft",
        surface_gf_method="analytic_diagonal",
    )


def _fixtures():
    return (
        (
            "bounded_T0p7_g0",
            OpenThreeModeFixture(
                grid=SignedFrequencyGrid(step=0.5, n_positive=10),
                temperature=0.7,
                g=0.0,
                eta_lead=0.05,
                backend="fft",
                hilbert_backend="fft",
                surface_gf_method="analytic_diagonal",
            ),
        ),
    )


def _physics_document(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _physics_document(item)
            for key, item in sorted(value.items())
            if key not in _TIMING_KEYS and key not in _ORCHESTRATION_KEYS
        }
    if isinstance(value, list):
        return [_physics_document(item) for item in value]
    return value


def _digest(document: Any) -> str:
    payload = json.dumps(
        _physics_document(document),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _run(root: Path, *, comm: Any, execution_id: str):
    return run_production_toy_mpi_physics(
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        run_directory=root,
        config=_config(),
        open_fixtures=_fixtures(),
        validation_tier="pull_request",
        comm=comm,
        require_mpi=comm is MPI.COMM_WORLD,
        execution_id=execution_id,
    )


def main() -> int:
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if len(sys.argv) != 3 or sys.argv[2] not in {"initial", "resume"}:
        if rank == 0:
            print(
                "usage: production_toy_physics_smoke.py RUN initial|resume",
                file=sys.stderr,
            )
        return 2
    root = Path(sys.argv[1]).expanduser().resolve()
    mode = sys.argv[2]
    serial_root = root / "serial"
    mpi_root = root / "mpi"

    serial = None
    if mode == "initial" and rank == 0:
        serial = _run(
            serial_root,
            comm=serial_task_communicator(),
            execution_id="serial-baseline",
        )
        if serial is None:
            raise AssertionError("serial coordinator did not return a result")
    MPI.COMM_WORLD.Barrier()

    execution_id = (
        "two-rank-initial" if mode == "initial" else f"{size}-rank-resume"
    )
    result = _run(
        mpi_root,
        comm=MPI.COMM_WORLD,
        execution_id=execution_id,
    )
    MPI.COMM_WORLD.Barrier()
    if rank != 0:
        if result is not None:
            raise AssertionError("worker rank unexpectedly received a result")
        return 0
    if result is None or not result.execution.complete:
        raise AssertionError("rank zero did not receive a complete result")
    if result.execution.mode != "mpi":
        raise AssertionError("real MPI communicator used the serial path")
    if result.execution.communicator_size != size:
        raise AssertionError("communicator size was not recorded exactly")

    digest = _digest(result.report.as_dict())
    baseline_path = root / "bounded-physics-baseline.json"
    if mode == "initial":
        if size != 2:
            raise AssertionError("initial smoke must use exactly two ranks")
        assert serial is not None
        serial_digest = _digest(serial.report.as_dict())
        if digest != serial_digest:
            raise AssertionError("serial and two-rank physics reports differ")
        baseline_path.write_text(
            json.dumps(
                {
                    "physics_sha256": digest,
                    "mpi_scientific_sha256": (
                        result.execution.scientific_sha256
                    ),
                },
                allow_nan=False,
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )
        output = {
            "all_resumed": False,
            "equal": True,
            "physics_sha256": digest,
            "ranks": size,
        }
    else:
        baseline = json.loads(baseline_path.read_text(encoding="ascii"))
        if digest != baseline["physics_sha256"]:
            raise AssertionError("changed-worker resume changed physics report")
        if result.execution.scientific_sha256 != (
            baseline["mpi_scientific_sha256"]
        ):
            raise AssertionError("changed-worker resume changed artifacts")
        if not all(item.resumed for item in result.execution.outcomes):
            raise AssertionError("a completed physics task was rerun")
        output = {
            "all_resumed": True,
            "equal": True,
            "physics_sha256": digest,
            "ranks": size,
        }
    print(json.dumps(output, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
