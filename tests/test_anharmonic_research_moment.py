"""Final-map routing and separate moment evidence, without altering SCBA gates."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import negfpy.anharmonic.research_moment as module
from negfpy.anharmonic import solve_scba
from negfpy.anharmonic.research_moment import ResearchMomentDiagnostic
from test_anharmonic_research_solver import Driver, config, problem


def test_opt_in_budget_outcome_preserves_history_fields_and_old_diagnostics():
    p, cfg = problem(cubic=0), config()
    baseline = solve_scba(p, cfg, research_map=Driver())
    assert 'research_moment' not in baseline.metadata
    check = ResearchMomentDiagnostic(max_points=23, max_workers=1)
    driver = Driver()
    result = solve_scba(p, cfg, research_map=driver, research_moment=check)
    assert result.converged == baseline.converged is True
    assert result.history == baseline.history
    assert dict(result.diagnostics.metrics) == dict(baseline.diagnostics.metrics)
    assert result.diagnostics.failures == baseline.diagnostics.failures
    assert result.diagnostics.passed == baseline.diagnostics.passed
    assert result.diagnostics.release_certified == baseline.diagnostics.release_certified
    assert result.release_certified == baseline.release_certified is False
    for name in ('lesser', 'greater', 'retarded'):
        np.testing.assert_array_equal(getattr(result.sigma_pp, name), getattr(baseline.sigma_pp, name))
    assert len(driver.calls) == len(driver.commits) == len(result.history) + 1
    assert driver.calls[-1][0] == 'final_diagnostics'
    report = result.metadata['research_moment']
    assert report['evaluated'] and not report['passed']
    assert not report['quadrature']['complete_partition']
    assert report['quadrature']['stop_reason'] == 'point-budget'
    assert report['moment_screen']['residual'] is None
    assert report['coarse_spectral_moment_residual'] == result.diagnostics.metrics['green_spectral_moment_residual']
    assert 'full_domain_moment' not in check.arrays
    assert report['input_field_sha256'] == module._field_hashes(driver.calls[-1][1][1])


def test_real_pointwise_moment_after_fresh_map_has_reconstructible_evidence():
    p, cfg = problem(cubic=0), config()
    check = ResearchMomentDiagnostic(max_workers=1)
    driver = Driver()
    result = solve_scba(p, cfg, research_map=driver, research_moment=check)
    report = check.report
    assert result.converged and report['passed']
    assert report['input_field_unchanged']
    assert report['quadrature']['complete_partition']
    assert report['quadrature']['sampled_tolerance_met']
    assert report['moment_screen']['passed']
    assert report['input_field_sha256'] == module._field_hashes(result.sigma_pp)
    assert not report['global_frequency_accuracy_established']
    np.testing.assert_allclose(check.arrays['leaf_high'].sum(axis=0), check.arrays['full_domain_moment'])
    assert report['quadrature']['point_evaluations'] == 24 * 60 + 48 * (report['quadrature']['leaf_count'] - 60)


def test_missing_fresh_final_map_skips_and_clears_previous_evidence(monkeypatch):
    check = ResearchMomentDiagnostic()
    check.report, check.arrays = {'old': True}, {'old': np.ones(1)}
    def forbidden(*args, **kwargs):
        pytest.fail('A stopped final map must not start a moment evaluation')
    monkeypatch.setattr(check, 'evaluate', forbidden)
    result = solve_scba(problem(cubic=0), config(), research_map=Driver(final_stop=True), research_moment=check)
    assert not result.converged and not result.metadata['research_final_check_completed']
    assert check.arrays is None
    assert result.metadata['research_moment']['reason'] == 'fresh_final_map_incomplete'
    assert not result.metadata['research_moment']['evaluated']
    assert not result.metadata['research_moment']['passed']


def test_moment_screen_failure_does_not_revoke_scba_convergence(monkeypatch):
    class ZeroSpectral:
        def __init__(self, *args, **kwargs):
            pass
        def evaluate(self, omega):
            return SimpleNamespace(retarded=np.zeros((1, 1), complex))
    monkeypatch.setattr(module, 'PointwiseInteracting', ZeroSpectral)
    check = ResearchMomentDiagnostic(max_workers=1)
    result = solve_scba(problem(cubic=0), config(), research_map=Driver(), research_moment=check)
    assert result.converged and result.diagnostics.passed
    assert check.report['quadrature']['sampled_tolerance_met']
    assert check.report['moment_screen']['residual'] == 1.
    assert not check.report['moment_screen']['passed'] and not check.report['passed']


def test_rejected_trial_final_diagnostic_receives_restored_retained_input(monkeypatch):
    driver = Driver(4, rejected_at=3)
    check = ResearchMomentDiagnostic(max_points=0, max_workers=1)
    result = solve_scba(problem(), config(), research_map=driver, research_moment=check)
    assert result.metadata['anderson']['rejected'] == 1
    reference = driver.calls[1][1][1]
    assert check.report['input_field_sha256'] == module._field_hashes(reference)
    assert not result.converged
    assert check.report['evaluated']


def test_diagnostic_requires_research_map_before_computation():
    check = ResearchMomentDiagnostic()
    with pytest.raises(ValueError, match='explicit research_map'):
        solve_scba(problem(), config(), research_moment=check)
    with pytest.raises(TypeError, match='ResearchMomentDiagnostic'):
        solve_scba(problem(), config(), research_map=Driver(), research_moment=object())


def test_validate_controls_and_policy_before_entering_solver():
    p = problem()
    with pytest.raises(ValueError, match='fit within max_leaves'):
        solve_scba(p, config(), research_map=Driver(), research_moment=ResearchMomentDiagnostic(max_leaves=2))
    with pytest.raises(ValueError, match='equal positive temperatures'):
        solve_scba(replace(p, right=replace(p.right, temperature=1.)), config(),
                   research_map=Driver(), research_moment=ResearchMomentDiagnostic())
    with pytest.raises(ValueError, match='eta_device'):
        solve_scba(p, config(eta_device=.01), research_map=Driver(), research_moment=ResearchMomentDiagnostic())
    with pytest.raises(ValueError, match='moment_rtol'):
        ResearchMomentDiagnostic(moment_rtol=np.nan)
    with pytest.raises(ValueError, match='Unknown'):
        ResearchMomentDiagnostic(interpolation_policy='magic')


def test_pointwise_error_propagates_without_stale_evidence(monkeypatch):
    check = ResearchMomentDiagnostic()
    check.report, check.arrays = {'old': 1}, {'old': 1}
    def failed(*args, **kwargs):
        raise FloatingPointError('pointwise failed')
    monkeypatch.setattr(module, 'PointwiseInteracting', failed)
    with pytest.raises(FloatingPointError, match='pointwise failed'):
        solve_scba(problem(cubic=0), config(), research_map=Driver(), research_moment=check)
    assert check.report is None and check.arrays is None
