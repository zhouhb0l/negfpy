from __future__ import annotations

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
