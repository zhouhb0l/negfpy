from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic._material_nq1_validation import (
    build_observation_mode_set,
    measure_observation_mode_poles,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SignedFrequencyGrid,
)
from negfpy.core.types import Device1D, LeadBlocks


def _problem(step: float = 0.05) -> CubicSCBAProblem:
    diagonal = np.diag([0.0, 0.25, 1.0, 4.0]).astype(np.complex128)
    device = Device1D([diagonal], [])
    lead = LeadBlocks(diagonal, np.zeros_like(diagonal))
    indices = np.arange(4, dtype=np.int64)
    reservoir = HarmonicReservoir(
        lead=lead,
        temperature=0.5,
        device_to_lead=np.zeros_like(diagonal),
        device_indices=indices,
    )
    vertex = SparseCubicVertex(
        ndof=4,
        indices=np.empty((0, 3), dtype=np.int64),
        coefficients=np.empty(0, dtype=np.float64),
    )
    return CubicSCBAProblem(
        device=device,
        left=reservoir,
        right=reservoir,
        vertex=vertex,
        grid=SignedFrequencyGrid(step=step, n_positive=60),
    )


def test_observation_modes_are_frozen_by_resolution_before_scba() -> None:
    problem = _problem()
    modes = build_observation_mode_set(
        problem,
        observation_layers=(0,),
        grid_step=0.05,
        eta_lead=0.01,
    )
    assert modes.resolution_threshold == pytest.approx(0.5)
    assert [item.frequency for item in modes.subspaces] == pytest.approx(
        [0.5, 1.0, 2.0]
    )
    assert modes.as_dict()["mode_set_sha256"]


def test_observation_modes_reject_empty_coarse_resolution() -> None:
    with pytest.raises(ValueError, match="resolves no observation modes"):
        build_observation_mode_set(
            _problem(step=0.3),
            observation_layers=(0,),
            grid_step=0.3,
            eta_lead=0.01,
        )


def test_mode_poles_project_complete_degenerate_subspace() -> None:
    problem = _problem()
    modes = build_observation_mode_set(
        problem,
        observation_layers=(0,),
        grid_step=0.05,
        eta_lead=0.01,
    )
    sigma = np.zeros((problem.grid.n_points, 4, 4), dtype=np.complex128)
    sigma[:, 1, 1] = 0.02 - 0.01j
    sigma[:, 2, 2] = 0.04 - 0.02j
    sigma[:, 3, 3] = 0.08 - 0.04j
    measured = measure_observation_mode_poles(problem.grid.values, sigma, modes)
    assert measured[0].shifts == pytest.approx((0.02,))
    assert measured[0].linewidths == pytest.approx((0.02,))
    assert measured[-1].shifts == pytest.approx((0.02,))
    assert measured[-1].linewidths == pytest.approx((0.02,))
