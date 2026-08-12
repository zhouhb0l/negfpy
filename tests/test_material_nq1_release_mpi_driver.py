from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import pytest

from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
)
from negfpy.anharmonic._mpi_tasks import TaskSpec


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
    phase_module = importlib.import_module("negfpy.anharmonic._material_nq1_mpi")
    phase = phase_module._phase_manifest(
        campaign, campaign.wave_index(scientific.task_id)
    )
    task_identity = phase.task_identity_sha256(spec)
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
        "mixing_floor": scientific.mixing_floor,
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
        "task_identity_sha256": task_identity,
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
    bindings = {
        "expected_campaign_manifest_sha256": campaign.manifest.manifest_sha256,
        "expected_task_identity_sha256": task_identity,
    }
    return receipt_path, spec, scba, checkpoint, bindings


def test_material_receipt_authenticates_entire_terminal_tree(tmp_path: Path) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    evidence = driver._receipt_evidence(receipt, spec, **bindings)
    assert evidence["task_id"] == "coupling-g0001"
    assert evidence["state"] == "passed"


def test_material_receipt_rejects_checkpoint_mutation(tmp_path: Path) -> None:
    receipt, spec, _scba, checkpoint, bindings = _fixture(tmp_path)
    checkpoint.write_bytes(b"changed\n")
    with pytest.raises(driver.MaterialCampaignError, match="task tree"):
        driver._receipt_evidence(receipt, spec, **bindings)


@pytest.mark.parametrize(
    ("control", "changed_value"),
    (("mixing", 0.1), ("mixing_floor", 0.005)),
)
def test_material_receipt_rejects_changed_numerical_controls(
    tmp_path: Path, control: str, changed_value: float,
) -> None:
    receipt, spec, scba, _checkpoint, bindings = _fixture(tmp_path)
    manifest = driver._load_canonical(scba / "campaign-manifest.json", "fixture")
    manifest["numerical_controls"][control] = changed_value
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
        driver._receipt_evidence(receipt, spec, **bindings)


@pytest.mark.parametrize(
    "binding",
    ("expected_campaign_manifest_sha256", "expected_task_identity_sha256"),
)
def test_material_receipt_rejects_cross_campaign_identity(
    tmp_path: Path, binding: str,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    bindings[binding] = "f" * 64
    with pytest.raises(driver.MaterialCampaignError, match="campaign|identity"):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_v2_material_receipt_requires_serialized_mixing_floor(
    tmp_path: Path,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    payload = {
        "schema": spec.payload["schema"],
        "dependencies": list(spec.payload["dependencies"]),
        "scientific_task": dict(spec.payload["scientific_task"]),
    }
    payload["scientific_task"].pop("mixing_floor")
    malformed = TaskSpec(
        task_id=spec.task_id,
        identity_hashes=spec.identity_hashes,
        payload=payload,
    )
    with pytest.raises(driver.MaterialCampaignError, match="lacks mixing_floor"):
        driver._receipt_evidence(receipt, malformed, **bindings)


def test_task_input_authentication_forwards_serialized_mixing_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    ).task("coupling-g100")
    captured: dict[str, object] = {}

    def authenticate(**kwargs):
        captured.update(kwargs)
        return "identity", "problem", "config"

    monkeypatch.setattr(driver.case, "authenticate_scba_inputs", authenticate)
    result = driver._authenticate_task_inputs(
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.full.hdf5",
        phase_b_artifacts=tmp_path / "phase-b",
        task=task,
    )
    assert result == ("identity", "problem", "config")
    assert captured["mixing"] == task.mixing
    assert captured["mixing_floor"] == task.mixing_floor
    assert captured["coupling_scale"] == task.coupling_scale

    with pytest.raises(driver.MaterialCampaignError, match="mean temperature"):
        driver._authenticate_task_inputs(
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3.full.hdf5",
            phase_b_artifacts=tmp_path / "phase-b",
            task=replace(task, mean_temperature_kelvin=301.0),
        )


def test_candidate_qualification_is_exact_and_fail_closed(tmp_path: Path) -> None:
    source = {"commit": "candidate", "tree": "tree", "dirty": False}
    report = tmp_path / "qualification.json"
    payload = {
        "schema": driver.QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "scientific_controls": driver._scientific_control_record(),
        "checks": {
            name: True for name in driver.EXPECTED_QUALIFICATION_CHECKS
        },
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
    assert record["scientific_task_set_sha256"] == payload[
        "scientific_controls"
    ]["task_set_sha256"]
    assert payload["scientific_controls"]["acceptance"][
        "fixed_point_rtol"
    ] == 1.0e-8
    assert "bulk-film prediction" in payload["scientific_controls"][
        "excluded_claims"
    ]
    payload["scientific_controls"]["tasks"][0]["mixing_floor"] = 0.005
    _write(report, payload)
    with pytest.raises(driver.MaterialCampaignError, match="does not authorize"):
        driver._validate_qualification(
            report,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )
    payload["scientific_controls"] = driver._scientific_control_record()
    payload["checks"]["mpi_two_rank_smoke"] = False
    _write(report, payload)
    with pytest.raises(driver.MaterialCampaignError, match="does not authorize"):
        driver._validate_qualification(
            report,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


@pytest.mark.parametrize("mutation", ("missing", "extra"))
def test_candidate_qualification_requires_exact_check_set(
    tmp_path: Path, mutation: str,
) -> None:
    source = {"commit": "candidate", "tree": "tree", "dirty": False}
    checks = {name: True for name in driver.EXPECTED_QUALIFICATION_CHECKS}
    if mutation == "missing":
        checks.pop("source_clean_and_unchanged")
    else:
        checks["informal_extra_check"] = True
    report = tmp_path / "qualification.json"
    _write(report, {
        "schema": driver.QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "scientific_controls": driver._scientific_control_record(),
        "checks": checks,
        "evidence": {},
        "release_preflight_passed": True,
    })
    with pytest.raises(driver.MaterialCampaignError, match="does not authorize"):
        driver._validate_qualification(
            report,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )
