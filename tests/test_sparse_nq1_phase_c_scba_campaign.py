from __future__ import annotations

import importlib
from pathlib import Path
import sys


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
campaign = importlib.import_module("run_sparse_nq1_phase_c_scba_campaign")


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(campaign._canonical(value))


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
