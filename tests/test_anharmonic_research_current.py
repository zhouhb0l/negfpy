"""Reusable final-map current quadrature and solver routing regressions."""

from dataclasses import replace

import numpy as np
import pytest

import negfpy.anharmonic.research_current as module
from negfpy.anharmonic import ResearchCurrentDiagnostic, solve_scba
from negfpy.anharmonic.research_moment import ResearchMomentDiagnostic
from negfpy.anharmonic.types import CurrentReport
from test_anharmonic_research_solver import Driver, config, problem


def biased_problem(*, cubic=0.0):
    value = problem(cubic=cubic)
    return replace(
        value,
        left=replace(value.left, temperature=0.9),
        right=replace(value.right, temperature=0.7),
    )


def test_vector_quadrature_retains_reconstructible_final_leaves():
    def polynomial(omega):
        return np.array([omega, omega**2, -omega - omega**2])

    report, arrays = module.run_positive_current_quadrature(
        [0.0, 0.5, 1.0],
        polynomial,
        max_workers=1,
        rtol=1e-12,
        atol=1e-14,
    )
    assert report["complete_partition"] and report["sampled_tolerance_met"]
    assert report["stop_reason"] == "sampled-tolerance"
    assert report["point_evaluations"] == 48
    np.testing.assert_allclose(
        arrays["leaf_high"].sum(axis=0), [0.5, 1.0 / 3.0, -5.0 / 6.0]
    )
    np.testing.assert_array_equal(
        arrays["leaf_error_abs"], np.abs(arrays["leaf_high"] - arrays["leaf_low"])
    )
    np.testing.assert_array_equal(arrays["leaf_parent_indices"], [0, 1])


def test_budget_and_leaf_stops_are_explicit_numerical_outcomes():
    report, arrays = module.run_positive_current_quadrature(
        [0.0, 1.0],
        lambda omega: np.ones(3),
        max_points=23,
        max_workers=1,
    )
    assert not report["complete_partition"]
    assert not report["sampled_tolerance_met"]
    assert report["stop_reason"] == "point-budget"
    assert report["point_evaluations"] == 0
    assert arrays["leaf_high"].shape == (0, 3)

    report, arrays = module.run_positive_current_quadrature(
        [0.0, 1.0],
        lambda omega: np.array([np.sin(31 * omega), 0.0, 0.0]),
        max_workers=1,
        max_leaves=1,
        rtol=0.0,
        atol=0.0,
    )
    assert report["complete_partition"]
    assert not report["sampled_tolerance_met"]
    assert report["stop_reason"] == "leaf-budget"
    assert arrays["leaf_high"].shape == (1, 3)


@pytest.mark.parametrize(
    "value,match",
    [
        (lambda omega: np.ones(2), "wrong shape"),
        (lambda omega: np.array([np.nan, 0.0, 0.0]), "nonfinite"),
        (lambda omega: np.array([1j, 0.0, 0.0]), "real-valued"),
    ],
)
def test_quadrature_rejects_invalid_integrands(value, match):
    with pytest.raises(ValueError, match=match):
        module.run_positive_current_quadrature([0.0, 1.0], value, max_workers=1)


def test_opt_in_budget_outcome_preserves_solver_result_and_hashes_input():
    p, cfg = biased_problem(), config()
    baseline = solve_scba(p, cfg, research_map=Driver())
    assert "research_current" not in baseline.metadata
    check = ResearchCurrentDiagnostic(max_points=23, max_workers=1)
    driver = Driver()
    result = solve_scba(p, cfg, research_map=driver, research_current=check)
    assert result.converged == baseline.converged is True
    assert result.history == baseline.history
    assert dict(result.diagnostics.metrics) == dict(baseline.diagnostics.metrics)
    assert result.diagnostics.failures == baseline.diagnostics.failures
    assert result.release_certified == baseline.release_certified is False
    for name in ("lesser", "greater", "retarded"):
        np.testing.assert_array_equal(
            getattr(result.sigma_pp, name), getattr(baseline.sigma_pp, name)
        )
    report = result.metadata["research_current"]
    assert report["evaluated"] and not report["passed"]
    assert report["outcome"] == "incomplete-point-budget"
    assert report["quadrature"]["stop_reason"] == "point-budget"
    assert report["input_field_unchanged"]
    assert report["input_field_sha256"] == module._field_hashes(driver.calls[-1][1][1])
    assert report["resolved_atol"] == cfg.current_atol * p.scale.current_scale
    assert check.arrays["leaf_high"].shape == (0, 3)


def test_real_biased_pointwise_integration_and_exact_node_binding():
    p, cfg = biased_problem(), config()
    check = ResearchCurrentDiagnostic(
        max_workers=1, rtol=0.05, coarse_agreement_rtol=0.5
    )
    result = solve_scba(p, cfg, research_map=Driver(), research_current=check)
    report = check.report
    assert result.converged
    assert report["evaluated"] and report["passed"]
    assert report["outcome"] == "sampled-tolerance-met"
    assert report["quadrature"]["complete_partition"]
    assert report["interpretation"]["integration_passed"]
    assert report["interpretation"]["heat_flows_hot_to_cold"]
    assert not report["global_frequency_accuracy_established"]
    np.testing.assert_allclose(
        check.arrays["leaf_high"].sum(axis=0),
        report["quadrature"]["statistics"]["integrals"],
    )

    integrand = module._CurrentIntegrand(p, cfg, result.sigma_pp)
    expected = np.column_stack(
        (
            np.asarray(result.currents.contact_spectra).T,
            result.currents.collision_spectrum,
        )
    ) / (2.0 * np.pi)
    actual = np.array(
        [integrand(omega) for omega in p.grid.values[p.grid.positive_indices]]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)


def test_coarse_comparison_and_conservation_remain_separate_flags():
    p, cfg = biased_problem(), config(current_rtol=1e-5, current_atol=1e-10)
    coarse = CurrentReport(left=1.0, right=-1.0, collision_power=0.0)
    quadrature = {
        "complete_partition": True,
        "sampled_tolerance_met": True,
        "statistics": {
            "integrals": [1.02, -1.02, 0.0],
            "component_errors": [1e-6, 1e-6, 1e-6],
        },
    }
    interpretation = module._interpret(
        p, cfg, quadrature, coarse, coarse_agreement_rtol=0.01
    )
    assert interpretation["integration_passed"]
    assert not interpretation["coarse_contact_agreement"]
    assert interpretation["integrated_balance_screen_passed"]

    quadrature["statistics"]["integrals"] = [1.0, -0.8, 0.0]
    interpretation = module._interpret(
        p, cfg, quadrature, coarse, coarse_agreement_rtol=0.5
    )
    assert interpretation["integration_passed"]
    assert interpretation["coarse_contact_agreement"]
    assert not interpretation["integrated_balance_screen_passed"]


def test_missing_fresh_final_map_skips_and_clears_previous_evidence(monkeypatch):
    check = ResearchCurrentDiagnostic()
    check.report, check.arrays = {"old": True}, {"old": np.ones(1)}

    def forbidden(*args, **kwargs):
        pytest.fail("A stopped final map must not start current integration")

    monkeypatch.setattr(check, "evaluate", forbidden)
    result = solve_scba(
        biased_problem(),
        config(),
        research_map=Driver(final_stop=True),
        research_current=check,
    )
    report = result.metadata["research_current"]
    assert not result.converged
    assert not result.metadata["research_final_check_completed"]
    assert check.arrays is None
    assert report["outcome"] == "skipped"
    assert report["reason"] == "fresh_final_map_incomplete"
    assert not report["evaluated"] and not report["passed"]


def test_rejected_trial_uses_restored_retained_input():
    driver = Driver(4, rejected_at=3)
    check = ResearchCurrentDiagnostic(max_points=0, max_workers=1)
    result = solve_scba(
        biased_problem(cubic=0.005),
        config(),
        research_map=driver,
        research_current=check,
    )
    assert result.metadata["anderson"]["rejected"] == 1
    reference = driver.calls[1][1][1]
    assert check.report["input_field_sha256"] == module._field_hashes(reference)


def test_current_and_moment_share_the_completed_final_input():
    current = ResearchCurrentDiagnostic(max_points=0, max_workers=1)
    moment = ResearchMomentDiagnostic(
        max_points=0, max_workers=1, interpolation_policy="linear-keldysh"
    )
    driver = Driver()
    result = solve_scba(
        biased_problem(),
        config(),
        research_map=driver,
        research_moment=moment,
        research_current=current,
    )
    assert "research_moment" in result.metadata
    assert "research_current" in result.metadata
    assert current.report["input_field_sha256"] == moment.report["input_field_sha256"]


def test_validation_and_pointwise_failure_clear_stale_evidence(monkeypatch):
    with pytest.raises(ValueError, match="explicit research_map"):
        solve_scba(
            biased_problem(), config(), research_current=ResearchCurrentDiagnostic()
        )
    with pytest.raises(TypeError, match="ResearchCurrentDiagnostic"):
        solve_scba(
            biased_problem(), config(), research_map=Driver(), research_current=object()
        )
    with pytest.raises(ValueError, match="max_leaves"):
        solve_scba(
            biased_problem(),
            config(),
            research_map=Driver(),
            research_current=ResearchCurrentDiagnostic(max_leaves=2),
        )
    with pytest.raises(ValueError, match="coarse_agreement_rtol"):
        ResearchCurrentDiagnostic(coarse_agreement_rtol=np.nan)

    check = ResearchCurrentDiagnostic()
    check.report, check.arrays = {"old": True}, {"old": np.ones(1)}

    class FailedIntegrand:
        def __init__(self, *args, **kwargs):
            raise FloatingPointError("pointwise failed")

    monkeypatch.setattr(module, "_CurrentIntegrand", FailedIntegrand)
    with pytest.raises(FloatingPointError, match="pointwise failed"):
        solve_scba(
            biased_problem(),
            config(),
            research_map=Driver(),
            research_current=check,
        )
    assert check.report is None and check.arrays is None


def test_exported_from_anharmonic_namespace():
    from negfpy import anharmonic

    assert anharmonic.ResearchCurrentDiagnostic is ResearchCurrentDiagnostic
    assert "ResearchCurrentDiagnostic" in anharmonic.__all__
