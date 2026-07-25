from __future__ import annotations

import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)
from negfpy.anharmonic.toy import (
    DenseV01PeakMeasurement,
    RefinementSpec,
    build_dense_v01_refinement_report,
    pinned_scalar_chain,
    run_refinement_study,
)


@pytest.mark.nightly
def test_interacting_full_control_matrix_is_grid_and_eta_converged() -> None:
    """Resolve eta/2 on a fixed grid before accepting the full matrix."""

    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=2.0e-3,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.03125, n_positive=256),
    )
    config = SCBAConfig(
        eta_lead=0.015,
        max_iterations=120,
        min_iterations=3,
        consecutive_passes=3,
    )
    study = run_refinement_study(
        problem,
        config,
        (
            RefinementSpec("half_spacing", step_factor=0.5),
            RefinementSpec("larger_cutoff", cutoff_factor=1.125),
            RefinementSpec("smaller_eta", eta_factor=0.5),
            RefinementSpec("changed_mixing", mixing=0.3),
        ),
        max_frequency_points=1100,
    )

    peak_measurements: dict[str, DenseV01PeakMeasurement] = {}
    for candidate in study.candidates:
        peak_measurements[candidate.name] = DenseV01PeakMeasurement(
            observable="sigma_pp_broadening_peak_frequency",
        )

    report = build_dense_v01_refinement_report(
        study,
        selected_peak_measurements=peak_measurements,
    )

    eta_observation = study.observation("smaller_eta")
    eta_comparison = next(
        item
        for item in report.as_dict()["comparisons"]
        if item["candidate"] == "smaller_eta"
    )
    spacing_comparison = next(
        item
        for item in report.as_dict()["comparisons"]
        if item["candidate"] == "half_spacing"
    )

    assert study.reference.release_certified
    assert eta_comparison["controls"]["eta_lead"][
        "qualifies_for_release_coverage"
    ]
    assert (
        eta_comparison["deltas"]["max_current_relative_change"] < 5.0e-3
    )
    assert eta_comparison["gates"]["current_refinement"]["passed"]
    assert eta_observation.release_certified
    assert spacing_comparison["controls"]["spacing"][
        "qualifies_for_release_coverage"
    ]
    assert spacing_comparison["gates"]["current_refinement"]["passed"]
    assert (
        study.reference.result.diagnostics.metrics[
            "green_high_frequency_residual"
        ]
        > 5.0e-3
    )
    assert (
        study.reference.result.diagnostics.metrics[
            "green_high_frequency_extrapolated_residual"
        ]
        < 5.0e-3
    )
    assert report.as_dict()["summary"]["full_control_coverage"]
    assert all(item.release_certified for item in study.candidates)
    assert report.refinement_passed
    assert report.release_certified
