from __future__ import annotations

import importlib
from pathlib import Path
import sys

import h5py
import numpy as np
import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
validation = importlib.import_module("validate_material_nq1_release")


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
