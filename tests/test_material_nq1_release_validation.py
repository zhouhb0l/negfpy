from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest

from negfpy.anharmonic._material_nq1_mpi import (
    _phase_manifest,
    build_material_nq1_mpi_campaign,
)

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
validation = importlib.import_module("validate_material_nq1_release")


def _campaign():
    return build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )


def test_strict_receipt_loader_rejects_correct_name_in_wrong_wave(
    tmp_path: Path,
) -> None:
    campaign = _campaign()
    task = campaign.task("coupling-g0001")
    spec = campaign.manifest.task(task.task_id)
    phase = _phase_manifest(campaign, campaign.wave_index(task.task_id))
    name = (
        f"{phase.ordinal(spec):06d}-"
        f"{phase.task_identity_sha256(spec)[:16]}.json"
    )
    misplaced = tmp_path / "waves/999/task-artifacts" / name
    misplaced.parent.mkdir(parents=True)
    misplaced.write_bytes(validation.mpi_driver._canonical({
        "task_id": task.task_id,
    }))
    with pytest.raises(
        validation.MaterialReleaseValidationError,
        match="path is not canonical",
    ):
        validation._load_receipts(tmp_path, campaign)


def _fields(path: Path, perturbation: float = 0.0) -> None:
    with h5py.File(path, "w") as handle:
        for name in validation.FIELD_DATASETS:
            handle.create_dataset(
                name,
                data=np.ones((3, 2, 2), dtype=np.complex128) + perturbation,
            )


def test_streamed_field_comparison_uses_scale_aware_tolerance(
    tmp_path: Path,
) -> None:
    left = tmp_path / "left.h5"
    right = tmp_path / "right.h5"
    _fields(left)
    _fields(right, 5.0e-11)
    assert validation._field_comparison(
        left, right, rtol=1.0e-8, atol=1.0e-12
    )["passed"]
    _fields(right, 5.0e-7)
    assert not validation._field_comparison(
        left, right, rtol=1.0e-8, atol=1.0e-12
    )["passed"]


def test_pole_refinement_checks_linewidth_and_shift_separately() -> None:
    baseline = ({
        "ordinal": 0, "frequency": 1.5,
        "linewidths": [0.02], "shifts": [0.01],
    },)
    accepted = ({
        "ordinal": 0, "frequency": 1.5,
        "linewidths": [0.02005], "shifts": [0.01002],
    },)
    rejected = ({
        "ordinal": 0, "frequency": 1.5,
        "linewidths": [0.021], "shifts": [0.01],
    },)
    assert validation._pole_comparison(
        baseline, accepted, rtol=5.0e-3
    )["passed"]
    assert not validation._pole_comparison(
        baseline, rejected, rtol=5.0e-3
    )["passed"]


def test_current_change_has_explicit_physical_floor() -> None:
    assert validation._scaled_change(0.0, 5.0e-12, 1.0e-10) == pytest.approx(0.05)


def test_strict_predictor_authentication_binds_expected_parent_and_factor() -> None:
    campaign = _campaign()
    task = campaign.task("coupling-g001")
    parent = campaign.task("coupling-g0001")
    initialization = {
        "predictor": "coupling-square",
        "parent_coupling_scale": parent.coupling_scale,
        "child_coupling_scale": task.coupling_scale,
        "factor": (task.coupling_scale / parent.coupling_scale) ** 2,
    }
    parent_record = {
        "run_id": parent.task_id,
        "coupling_scale": parent.coupling_scale,
        "continuation_axis": "coupling",
        "initialization": initialization,
    }
    record = {
        "scba_manifest": {"continuation_parent": parent_record},
        "outcome": {"continuation_parent": parent_record},
        "archive_metadata": {"continuation_parent": parent_record},
        "evidence": {"continuation_initialization": initialization},
    }
    authenticated = validation._continuation_authentication(
        task, record, campaign
    )
    assert authenticated["passed"]
    assert authenticated["factor"] == 100.0

    changed = dict(parent_record)
    changed["coupling_scale"] = 0.002
    record["outcome"] = {"continuation_parent": changed}
    rejected = validation._continuation_authentication(task, record, campaign)
    assert not rejected["passed"]
    assert not rejected["checks"]["manifest_outcome_equal"]


def test_material_preflight_requires_every_phase_b_and_memory_gate() -> None:
    source = {
        "phase_b": {
            "phase_b_certified": True,
            "archive": {"metadata": {
                "phase_b_certified": True,
                "checks": {"mapping": True, "range": True},
                "reports": {
                    "raw_fc3_validation": {
                        "errors": [], "permutation_valid": True,
                    },
                    "device_mapping": {"certification_passed": True},
                    "acoustic_lead_classification": {"passed": True},
                },
            }},
        },
        "dimensions": {"n_frequency": 151, "ndof": 216, "nnz": 406080},
        "memory": {"guard_passed": True},
        "sparse_schedule": {"active_dofs": 120, "fft_padding": 320},
    }
    assert validation._material_preflight_passed(source)
    source["memory"]["guard_passed"] = False
    assert not validation._material_preflight_passed(source)


def test_strict_validation_calls_paired_pilot_authorization_before_campaign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repo"
    campaign = tmp_path / "campaign"
    repo.mkdir()
    campaign.mkdir()
    source = {"commit": "1" * 40, "tree": "2" * 40, "dirty": False}
    monkeypatch.setattr(
        validation.common, "_source_identity", lambda *_args: source
    )
    monkeypatch.setattr(
        validation.common,
        "_validate_lock",
        lambda _path: {"sha256": "b" * 64},
    )
    monkeypatch.setattr(
        validation, "code_provenance", lambda: {"source_tree_sha256": "a" * 64}
    )
    monkeypatch.setattr(
        validation.mpi_driver,
        "_authenticate_task_inputs",
        lambda **_kwargs: ({"authenticated": True}, object(), object()),
    )
    inputs = {"phase_a_fixture": "c" * 64}
    monkeypatch.setattr(
        validation.mpi_driver, "_pilot_input_hashes", lambda *_args: inputs
    )
    captured: dict[str, object] = {}

    def paired_gate(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        raise validation.mpi_driver.MaterialCampaignError("paired pilot gate")

    monkeypatch.setattr(
        validation.mpi_driver, "_validate_release_authorization", paired_gate
    )
    args = argparse.Namespace(
        repo_root=repo,
        campaign=campaign,
        expected_commit=source["commit"],
        environment_lock=tmp_path / "environment.lock",
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.h5",
        phase_b_artifacts=tmp_path / "phase-b",
        qualification_report=tmp_path / "qualification.json",
        pilot_audit=tmp_path / "pilot-audit.json",
        output=tmp_path / "strict.json",
    )
    with pytest.raises(
        validation.mpi_driver.MaterialCampaignError, match="paired pilot gate"
    ):
        validation.validate(args)
    assert captured["qualification_path"] == args.qualification_report
    assert captured["pilot_audit_path"] == args.pilot_audit
    assert captured["source"] == source
    assert captured["scientific_input_hashes"] == inputs


def test_strict_validation_rejects_symlinked_campaign_ancestry(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "outside-campaign"
    outside.mkdir()
    alias = tmp_path / "campaign-alias"
    alias.symlink_to(outside, target_is_directory=True)
    args = argparse.Namespace(repo_root=repo, campaign=alias)
    with pytest.raises(
        validation.mpi_driver.MaterialCampaignError, match="symlink"
    ):
        validation.validate(args)


def test_strict_evidence_loader_rejects_symlinked_top_level_record(
    tmp_path: Path,
) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_bytes(validation.mpi_driver._canonical({"passed": True}))
    record = campaign / "observation-modes.json"
    record.symlink_to(outside)
    with pytest.raises(
        validation.mpi_driver.MaterialCampaignError, match="symlink"
    ):
        validation.mpi_driver._load_canonical(record, "observation modes")


@pytest.mark.parametrize("symlink_kind", ("final", "ancestor"))
def test_strict_cli_does_not_resolve_through_symlinked_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    symlink_kind: str,
) -> None:
    report = {"release_certified": False}
    monkeypatch.setattr(validation, "validate", lambda _args: report)
    outside = tmp_path / "outside"
    outside.mkdir()
    if symlink_kind == "final":
        target = outside / "strict.json"
        target.write_bytes(validation.mpi_driver._canonical(report))
        output = tmp_path / "strict.json"
        output.symlink_to(target)
    else:
        alias = tmp_path / "output-alias"
        alias.symlink_to(outside, target_is_directory=True)
        output = alias / "strict.json"
        target = outside / "strict.json"
    monkeypatch.setattr(sys, "argv", [
        "validate_material_nq1_release.py",
        "--repo-root", "repo",
        "--fixture", "fixture",
        "--fc3-full", "fc3",
        "--phase-b-artifacts", "phase-b",
        "--campaign", "campaign",
        "--expected-commit", "1" * 40,
        "--environment-lock", "lock",
        "--qualification-report", "qualification",
        "--pilot-audit", "pilot-audit",
        "--output", str(output),
    ])
    assert validation.main() == 3
    if symlink_kind == "final":
        assert target.read_bytes() == validation.mpi_driver._canonical(report)
    else:
        assert not target.exists()


def test_strict_cli_requires_pilot_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "validate_material_nq1_release.py",
        "--repo-root", "repo",
        "--fixture", "fixture",
        "--fc3-full", "fc3",
        "--phase-b-artifacts", "phase-b",
        "--campaign", "campaign",
        "--expected-commit", "1" * 40,
        "--environment-lock", "lock",
        "--qualification-report", "qualification",
        "--output", "report",
    ])
    with pytest.raises(SystemExit) as error:
        validation.main()
    assert error.value.code == 2
