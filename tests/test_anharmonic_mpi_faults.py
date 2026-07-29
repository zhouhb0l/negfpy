from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "tests" / "mpi" / "task_level_faults.py"


def _environment_local_mpi() -> tuple[Path, Path]:
    binary_root = Path(sys.executable).resolve().parent
    mpiexec = binary_root / "mpiexec"
    python = binary_root / "python"
    if not mpiexec.is_file() or not python.is_file():
        pytest.skip("active test environment lacks its own mpiexec/python")
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("active test environment lacks mpi4py")
    return mpiexec, python


def _run_mpi(
    ranks: int,
    mode: str,
    run_directory: Path,
    *,
    timeout: float = 30.0,
) -> dict[str, object]:
    mpiexec, python = _environment_local_mpi()
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
            "PYTHONPATH": str(_ROOT / "src"),
        }
    )
    completed = subprocess.run(
        [
            str(mpiexec),
            "--bind-to",
            "core",
            "--map-by",
            "core",
            "-n",
            str(ranks),
            str(python),
            str(_SCRIPT),
            mode,
            str(run_directory),
        ],
        cwd=_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    assert lines, completed.stderr
    return json.loads(lines[-1])


@pytest.mark.parametrize(
    "mode",
    (
        "worker-error",
        "unexpected-worker-message",
        "root-validation-error",
        "root-build-error",
        "campaign-phase-validation-error",
        "campaign-final-build-error",
    ),
)
def test_real_two_rank_faults_terminate_on_every_rank_without_hang(
    tmp_path: Path,
    mode: str,
) -> None:
    result = _run_mpi(2, mode, tmp_path / mode)
    assert result["mode"] == mode
    caught = result["caught_by_rank"]
    assert isinstance(caught, list) and len(caught) == 2
    assert all(isinstance(value, str) and value for value in caught)


def test_partial_two_rank_campaign_resumes_exactly_at_four_ranks(
    tmp_path: Path,
) -> None:
    interrupted = _run_mpi(2, "partial", tmp_path / "changed-workers")
    assert interrupted["completed"] == ["q-point-0", "q-point-1"]

    resumed = _run_mpi(4, "resume", tmp_path / "changed-workers")
    clean = _run_mpi(4, "clean", tmp_path / "clean-four-ranks")
    assert resumed["resumed"] == ["q-point-0", "q-point-1"]
    assert resumed["physics_sha256"] == clean["physics_sha256"]


def test_real_mpi_fault_subprocess_timeout_is_bounded() -> None:
    """Keep the no-hang contract explicit in the test source."""

    assert _run_mpi.__kwdefaults__ == {"timeout": 30.0}
