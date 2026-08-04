from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
qualification = importlib.import_module("qualify_material_nq1_candidate")


def _environment() -> dict[str, str]:
    return {
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }


def test_passed_qualification_check_is_reused_without_rerun(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('stable qualification')"]
    first = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "stable.log",
        label="stable check",
    )
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "stable.log",
        label="stable check",
    )
    assert not first["reused"]
    assert second["reused"]
    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert len(list(tmp_path.glob("stable*.log"))) == 1


def test_failed_qualification_attempts_are_append_only(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    for _ in range(2):
        with pytest.raises(qualification.QualificationError, match="failed"):
            qualification._run(
                command,
                cwd=tmp_path,
                environment=_environment(),
                log=tmp_path / "failure.log",
                label="failing check",
            )
    assert (tmp_path / "failure.log").is_file()
    assert (tmp_path / "failure-attempt-002.log").is_file()
    assert len(list(tmp_path.glob("failure*.receipt.json"))) == 2


def test_passed_receipt_rejects_runtime_package_identity_drift(
    tmp_path: Path,
) -> None:
    command = [sys.executable, "-c", "print('runtime-bound')"]
    first_environment = _environment() | {
        "NEGFPY_QUALIFICATION_RUNTIME_SHA256": "a" * 64
    }
    qualification._run(
        command,
        cwd=tmp_path,
        environment=first_environment,
        log=tmp_path / "runtime.log",
        label="runtime check",
    )
    changed_environment = _environment() | {
        "NEGFPY_QUALIFICATION_RUNTIME_SHA256": "b" * 64
    }
    with pytest.raises(
        qualification.QualificationError, match="different identity"
    ):
        qualification._run(
            command,
            cwd=tmp_path,
            environment=changed_environment,
            log=tmp_path / "runtime.log",
            label="runtime check",
        )


def test_passed_receipt_rejects_tool_path_identity_drift(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('tool-path-bound')"]
    first_environment = _environment() | {"PATH": "/validated/tools:/usr/bin"}
    qualification._run(
        command,
        cwd=tmp_path,
        environment=first_environment,
        log=tmp_path / "tool-path.log",
        label="tool path check",
    )
    changed_environment = _environment() | {"PATH": "/usr/bin"}
    with pytest.raises(
        qualification.QualificationError, match="different identity"
    ):
        qualification._run(
            command,
            cwd=tmp_path,
            environment=changed_environment,
            log=tmp_path / "tool-path.log",
            label="tool path check",
        )
