from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from negfpy.anharmonic.toy.cubic_chain import pinned_scalar_chain
from negfpy.anharmonic.toy.refinement import (
    RefinementSpec,
    run_refinement_study,
)
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)


def _problem(*, cubic: float) -> CubicSCBAProblem:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=cubic,
    )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=15),
    )


def _config() -> SCBAConfig:
    return SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=70,
        min_iterations=3,
        consecutive_passes=2,
        mixing=0.3,
        sigma_rtol=3.0e-6,
        green_rtol=3.0e-6,
        identity_rtol=1.0e-7,
        current_rtol=1.0e-4,
        current_atol=1.0e-9,
    )


def test_harmonic_study_applies_named_grid_eta_and_mixing_changes() -> None:
    study = run_refinement_study(
        _problem(cubic=0.0),
        _config(),
        (
            RefinementSpec("half_spacing", step_factor=0.5),
            RefinementSpec("larger_cutoff", cutoff_factor=4.0 / 3.0),
            RefinementSpec("smaller_eta", eta_factor=0.5),
            RefinementSpec("smaller_mixing", mixing=0.15),
        ),
        max_frequency_points=100,
    )

    assert study.reference.name == "reference"
    assert study.reference.grid_step == pytest.approx(0.2)
    assert study.reference.grid_cutoff == pytest.approx(3.0)

    half_spacing = study.observation("half_spacing")
    assert half_spacing.grid_step == pytest.approx(0.1)
    assert half_spacing.grid_cutoff == pytest.approx(3.0)
    assert half_spacing.n_frequency == 61

    larger_cutoff = study.observation("larger_cutoff")
    assert larger_cutoff.grid_step == pytest.approx(0.2)
    assert larger_cutoff.grid_cutoff == pytest.approx(4.0)
    assert larger_cutoff.n_frequency == 41
    assert study.observation("smaller_eta").eta_lead == pytest.approx(5.0e-5)
    assert study.observation("smaller_mixing").mixing == pytest.approx(0.15)

    # With Psi=0, changing only the mixing cannot change the fixed point.
    mixing_delta = study.delta("smaller_mixing")
    assert mixing_delta.max_current_absolute_change == 0.0
    assert mixing_delta.sigma_tail_absolute_change == 0.0
    assert not mixing_delta.convergence_changed

    # A refinement report is descriptive and does not turn a deliberately
    # coarse grid into a certified calculation.
    assert not study.reference.release_certified
    assert all(not item.release_certified for item in study.candidates)
    assert all(not item.candidate_release_certified for item in study.deltas)
    with pytest.raises(FrozenInstanceError):
        mixing_delta.iteration_change = 10  # type: ignore[misc]


def test_weak_cubic_cutoff_refinement_reports_tail_and_current_deltas() -> None:
    study = run_refinement_study(
        _problem(cubic=1.0e-3),
        _config(),
        (RefinementSpec("larger_cutoff", cutoff_factor=4.0 / 3.0),),
        max_frequency_points=100,
    )

    candidate = study.observation("larger_cutoff")
    delta = study.delta("larger_cutoff")
    assert study.reference.converged
    assert candidate.converged
    assert candidate.sigma_tail_ratio < study.reference.sigma_tail_ratio
    assert delta.sigma_tail_change == pytest.approx(
        candidate.sigma_tail_ratio - study.reference.sigma_tail_ratio
    )
    assert delta.sigma_tail_absolute_change == pytest.approx(
        abs(delta.sigma_tail_change)
    )
    assert delta.max_current_absolute_change == pytest.approx(
        max(
            abs(candidate.left_current - study.reference.left_current),
            abs(candidate.right_current - study.reference.right_current),
        )
    )
    assert np.isfinite(delta.max_current_relative_change)
    assert delta.max_current_relative_change < 5.0e-3
    assert not candidate.release_certified


def test_refinement_study_validates_names_and_preflights_frequency_size() -> None:
    problem = _problem(cubic=0.0)
    config = _config()

    with pytest.raises(ValueError, match="unique"):
        run_refinement_study(
            problem,
            config,
            (RefinementSpec("same"), RefinementSpec("same")),
        )
    with pytest.raises(ValueError, match="reserved"):
        run_refinement_study(
            problem,
            config,
            (RefinementSpec("reference"),),
        )
    with pytest.raises(ValueError, match="max_frequency_points"):
        run_refinement_study(
            problem,
            config,
            (RefinementSpec("too_fine", step_factor=0.1),),
            max_frequency_points=100,
        )
    with pytest.raises(KeyError):
        run_refinement_study(problem, config, ()).delta("missing")
