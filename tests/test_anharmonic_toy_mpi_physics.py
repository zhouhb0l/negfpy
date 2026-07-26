from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import h5py
import pytest

from negfpy.anharmonic._mpi_tasks import serial_task_communicator
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.toy._mpi_production_gate_physics import (
    production_toy_physics_sha256,
    run_production_toy_mpi_physics,
)
from negfpy.anharmonic.toy.multidimensional_release import (
    pr_multidimensional_release_config,
)
from negfpy.anharmonic.toy.open_three_mode import OpenThreeModeFixture
from negfpy.anharmonic.toy.production_gate_report import (
    build_production_toy_gate_report,
)


_SOURCE_REVISION = "mpi-physics-bridge-test-v1"
_SOURCE_TREE_SHA256 = "a" * 64
_ENVIRONMENT_SHA256 = "b" * 64


def _bounded_config():
    return replace(
        pr_multidimensional_release_config(),
        executed_bias_values=(0.0,),
        production_backend="fft",
        production_hilbert_backend="fft",
        surface_gf_method="analytic_diagonal",
    )


def _bounded_open_fixtures():
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


def _run_serial(root: Path, execution_id: str):
    result = run_production_toy_mpi_physics(
        source_revision=_SOURCE_REVISION,
        source_tree_sha256=_SOURCE_TREE_SHA256,
        environment_sha256=_ENVIRONMENT_SHA256,
        run_directory=root,
        config=_bounded_config(),
        open_fixtures=_bounded_open_fixtures(),
        validation_tier="pull_request",
        comm=serial_task_communicator(),
        execution_id=execution_id,
    )
    assert result is not None
    return result


def _without_runtime_timing(value):
    if isinstance(value, dict):
        return {
            key: _without_runtime_timing(item)
            for key, item in value.items()
            if "second" not in key.lower()
            and "timing" not in key.lower()
            and key != "elapsed_ratio_right_over_left"
            and key != "report_sha256"
        }
    if isinstance(value, list):
        return [_without_runtime_timing(item) for item in value]
    return value


def _without_mpi_extension(document):
    result = dict(document)
    result.pop("report_sha256")
    result.pop("orchestration_provenance", None)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def test_serial_bridge_reconstructs_existing_gate_without_contract_changes(
    tmp_path: Path,
) -> None:
    first = _run_serial(tmp_path, "serial-first")

    assert first.execution.complete
    assert first.execution.mode == "serial"
    assert first.execution.communicator_size == 1
    assert len(first.execution.outcomes) == 7
    assert all(item.status == "completed" for item in first.execution.outcomes)
    assert not any(item.resumed for item in first.execution.outcomes)

    multidimensional = first.multidimensional.outcomes
    assert len(multidimensional) == 4
    assert all(item.successful for item in multidimensional)
    assert all(item.resumed_from_checkpoint for item in multidimensional)
    continuation = next(
        item
        for item in multidimensional
        if item.initialization == "continuation"
    )
    assert continuation.continuation_parent_id is not None
    assert continuation.continuation_parent_id.startswith("path_g0_bias0_")
    scientific = tmp_path / "scientific-checkpoints" / "multidimensional"
    assert len(tuple(scientific.glob("*.h5"))) == 4

    open_outcome = next(
        item
        for item in first.execution.outcomes
        if item.task_id.startswith("open_three_mode.")
    )
    assert open_outcome.artifact_relative_path is not None
    open_artifact = tmp_path / open_outcome.artifact_relative_path
    with h5py.File(open_artifact, "r") as handle:
        assert set(handle) == {"arrays", "evidence_json"}
        assert len(handle["arrays"]) == 15
        assert all(
            not handle["arrays"][name].dtype.hasobject
            for name in handle["arrays"]
        )
        evidence = json.loads(
            bytes(handle["evidence_json"][:]).decode("ascii")
        )
    assert evidence["artifact_kind"] == (
        "open_three_mode_explicit_arrays_v1"
    )
    assert "__complex__" in json.dumps(evidence["run_document"])

    base = build_production_toy_gate_report(
        first.open_runs,
        first.multidimensional,
        validation_tier="pull_request",
    )
    assert _without_mpi_extension(first.report.as_dict()) == (
        _without_mpi_extension(base.as_dict())
    )
    strict = first.report.as_dict()
    provenance = strict["orchestration_provenance"]
    assert provenance["solver_and_gate_contract_unchanged"] is True
    assert provenance["execution_manifest_file_sha256"] == _sha256(
        first.execution_manifest_path
    )
    assert provenance["execution_scientific_sha256"] == (
        first.execution.scientific_sha256
    )
    assert strict["summary"] == base.as_dict()["summary"]

    resumed = _run_serial(tmp_path, "serial-resume")
    assert all(item.resumed for item in resumed.execution.outcomes)
    assert resumed.execution.scientific_sha256 == (
        first.execution.scientific_sha256
    )
    assert _without_runtime_timing(
        _without_mpi_extension(resumed.report.as_dict())
    ) == _without_runtime_timing(
        _without_mpi_extension(first.report.as_dict())
    )
    assert production_toy_physics_sha256(resumed.report) == (
        production_toy_physics_sha256(first.report)
    )


def _environment_local_mpiexec() -> Path | None:
    candidate = Path(sys.executable).resolve().parent / "mpiexec"
    if candidate.is_file() and importlib.util.find_spec("mpi4py") is not None:
        return candidate
    return None


@pytest.mark.nightly
@pytest.mark.skipif(
    os.environ.get("NEGFPY_RUN_MPI_INTEGRATION") != "1",
    reason="set NEGFPY_RUN_MPI_INTEGRATION=1 for real MPI integration",
)
def test_real_two_rank_bridge_and_four_rank_exact_resume(
    tmp_path: Path,
) -> None:
    mpiexec = _environment_local_mpiexec()
    if mpiexec is None:
        pytest.skip("the active Python environment lacks local MPI/mpi4py")
    script = Path(__file__).parent / "mpi" / "production_toy_physics_smoke.py"
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (
        "" if not current_pythonpath else os.pathsep + current_pythonpath
    )
    initial = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "2",
            sys.executable,
            str(script),
            str(tmp_path),
            "initial",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr
    resumed = subprocess.run(
        [
            str(mpiexec),
            "-n",
            "4",
            sys.executable,
            str(script),
            str(tmp_path),
            "resume",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=180,
    )
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert json.loads(initial.stdout.splitlines()[-1])["equal"] is True
    assert json.loads(resumed.stdout.splitlines()[-1])["all_resumed"] is True
