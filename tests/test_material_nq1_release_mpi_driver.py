from __future__ import annotations

import importlib
from pathlib import Path
import sys

import pytest

from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
)


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("run_material_nq1_release_mpi")


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(driver._canonical(value))


def _fixture(tmp_path: Path):
    campaign = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )
    scientific = campaign.task("coupling-g0001")
    spec = campaign.manifest.task(scientific.task_id)
    receipt_path = tmp_path / "000000-0123456789abcdef.json"
    data = receipt_path.with_suffix(".data")
    scba = data / "scba"
    checkpoint = scba / "checkpoint" / "sigma-pp.h5"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authenticated checkpoint fixture\n")
    controls = {
        "grid_step": scientific.grid_step,
        "grid_n_positive": scientific.grid_n_positive,
        "grid_cutoff": scientific.grid_cutoff,
        "eta_lead": scientific.eta_lead,
        "mixing": scientific.mixing,
        "temperature_bias": scientific.temperature_bias_solver,
    }
    manifest = {
        "run_id": scientific.task_id,
        "coupling_scale": scientific.coupling_scale,
        "numerical_controls": controls,
    }
    _write(scba / "campaign-manifest.json", manifest)
    manifest_sha = driver._sha256(scba / "campaign-manifest.json")
    report = {
        "campaign_manifest_sha256": manifest_sha,
        "state": "passed",
        "scba_converged": True,
        "checkpoint_sha256": driver._sha256(checkpoint),
    }
    _write(scba / "scba-report.json", report)
    status = {
        "campaign_manifest_sha256": manifest_sha,
        "state": "passed",
        "scba_converged": True,
        "report_sha256": driver._sha256(scba / "scba-report.json"),
    }
    _write(scba / "status.json", status)
    records = driver._tree_records(data)
    receipt = {
        "schema": driver.RECEIPT_SCHEMA,
        "campaign_manifest_sha256": campaign.manifest.manifest_sha256,
        "task_id": scientific.task_id,
        "task_identity_sha256": "e" * 64,
        "scientific_task_sha256": scientific.task_sha256,
        "state": "passed",
        "task_directory": data.name,
        "task_tree_records": records,
        "task_tree_sha256": driver._tree_sha256(records),
        "scba_status_sha256": driver._sha256(scba / "status.json"),
        "scba_report_sha256": driver._sha256(scba / "scba-report.json"),
        "checkpoint_sha256": driver._sha256(checkpoint),
        "born_initial": None,
        "continuation_parent_task_id": None,
        "release_certified": False,
    }
    _write(receipt_path, receipt)
    return receipt_path, spec, scba, checkpoint


def test_material_receipt_authenticates_entire_terminal_tree(tmp_path: Path) -> None:
    receipt, spec, _scba, _checkpoint = _fixture(tmp_path)
    evidence = driver._receipt_evidence(receipt, spec)
    assert evidence["task_id"] == "coupling-g0001"
    assert evidence["state"] == "passed"


def test_material_receipt_rejects_checkpoint_mutation(tmp_path: Path) -> None:
    receipt, spec, _scba, checkpoint = _fixture(tmp_path)
    checkpoint.write_bytes(b"changed\n")
    with pytest.raises(driver.MaterialCampaignError, match="task tree"):
        driver._receipt_evidence(receipt, spec)


def test_material_receipt_rejects_changed_numerical_controls(
    tmp_path: Path,
) -> None:
    receipt, spec, scba, _checkpoint = _fixture(tmp_path)
    manifest = driver._load_canonical(scba / "campaign-manifest.json", "fixture")
    manifest["numerical_controls"]["mixing"] = 0.1
    _write(scba / "campaign-manifest.json", manifest)
    manifest_sha = driver._sha256(scba / "campaign-manifest.json")
    report = driver._load_canonical(scba / "scba-report.json", "fixture report")
    report["campaign_manifest_sha256"] = manifest_sha
    _write(scba / "scba-report.json", report)
    status = driver._load_canonical(scba / "status.json", "fixture status")
    status["campaign_manifest_sha256"] = manifest_sha
    status["report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(scba / "status.json", status)
    records = driver._tree_records(receipt.with_suffix(".data"))
    payload = driver._load_canonical(receipt, "fixture receipt")
    payload["task_tree_records"] = records
    payload["task_tree_sha256"] = driver._tree_sha256(records)
    payload["scba_status_sha256"] = driver._sha256(scba / "status.json")
    payload["scba_report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(receipt, payload)
    with pytest.raises(driver.MaterialCampaignError, match="numerical controls"):
        driver._receipt_evidence(receipt, spec)


def test_candidate_qualification_is_exact_and_fail_closed(tmp_path: Path) -> None:
    source = {"commit": "candidate", "tree": "tree", "dirty": False}
    report = tmp_path / "qualification.json"
    payload = {
        "schema": driver.QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "checks": {"full_tests": True, "mpi_smoke": True},
        "evidence": {"logs": ["tests.log", "mpi.log"]},
        "release_preflight_passed": True,
    }
    _write(report, payload)
    record = driver._validate_qualification(
        report,
        source=source,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
    )
    assert record["sha256"] == driver._sha256(report)
    payload["checks"]["mpi_smoke"] = False
    _write(report, payload)
    with pytest.raises(driver.MaterialCampaignError, match="does not authorize"):
        driver._validate_qualification(
            report,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )
