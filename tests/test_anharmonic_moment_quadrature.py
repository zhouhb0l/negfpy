"""Focused full-domain numerical, budget, and retained-evidence checks."""
import math

import numpy as np
import pytest

import negfpy.anharmonic._moment_quadrature as helper


def matrix(value):
    return np.array([[value, .2j * value], [-.2j * value, 2. * value]], complex)


def execute(function, grid=None, **kwargs):
    if grid is None:
        grid = np.linspace(-1., 1., 9)
    calls, events = [], []
    def observed(frequency):
        assert grid[0] < frequency < grid[-1]
        assert frequency != 0.
        calls.append(frequency)
        return function(frequency)
    report, arrays = helper.run_full_domain_quadrature(
        grid, observed, 2, emit=lambda **row: events.append(row), **kwargs)
    assert report['point_evaluations'] == len(calls)
    return report, arrays, events


def peak(frequency):
    width = .003
    return matrix(sum(width ** 2 / ((frequency - sign * .31) ** 2 + width ** 2)
                      for sign in (-1., 1.)))


def test_polynomial_full_domain_and_matrix_norm_reconstruction():
    report, arrays, events = execute(lambda frequency: matrix(1. + frequency ** 14))
    expected = matrix(2. + 2. / 15.)
    assert report['complete_initial_partition'] and report['complete_partition']
    assert report['sampled_tolerance_met'] and report['stop_reason'] == 'sampled-tolerance'
    assert report['initial_intervals'] == report['initial_completed_intervals'] == 8
    assert report['point_evaluations'] == 8 * 24 and report['leaf_count'] == 8
    assert report['adaptive_orders'] == [8, 16]
    assert not report['globally_certified']
    for name in ('initial_low_moment', 'initial_high_moment', 'full_domain_moment'):
        np.testing.assert_allclose(arrays[name], expected, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(arrays['leaf_parent_indices'], np.arange(8))
    np.testing.assert_array_equal(arrays['leaf_error_abs'], abs(arrays['leaf_high'] - arrays['leaf_low']))
    np.testing.assert_allclose(arrays['leaf_high'].sum(axis=0), arrays['full_domain_moment'])
    errors = report['sampled_errors']
    assert errors['max_error'] == pytest.approx(np.max(arrays['leaf_error_abs'], axis=(1, 2)).sum())
    assert errors['frobenius_error'] == pytest.approx(
        np.linalg.norm(arrays['leaf_high'] - arrays['leaf_low'], axis=(1, 2)).sum())
    assert errors['max_tolerance'] == pytest.approx(1e-8 + .002 * np.max(abs(expected)))
    assert errors['frobenius_tolerance'] == pytest.approx(1e-8 + .002 * np.linalg.norm(expected))
    assert events[0]['event'] == 'full_domain_quadrature_start'
    assert events[-1]['event'] == 'full_domain_quadrature_summary'


def test_initial_partial_point_budget_has_no_domain_moment_or_acceptance():
    report, arrays, _ = execute(lambda frequency: matrix(1.), max_points=47)
    assert report['point_evaluations'] == 24
    assert report['leaf_count'] == report['initial_completed_intervals'] == 1
    assert not report['complete_initial_partition'] and not report['complete_partition']
    assert not report['sampled_tolerance_met'] and report['sampled_errors'] is None
    assert report['stop_reason'] == 'point-budget' and report['moments'] == {}
    assert not any(name.endswith('_moment') for name in arrays)
    np.testing.assert_array_equal(arrays['leaf_bounds'], [[-1., -.75]])


def test_insufficient_first_panel_budget_keeps_empty_shape_consistent_evidence():
    report, arrays, _ = execute(lambda frequency: matrix(1.), max_points=23)
    assert report['point_evaluations'] == report['leaf_count'] == 0
    assert arrays['leaf_bounds'].shape == (0, 2)
    assert arrays['leaf_low'].shape == arrays['leaf_high'].shape == arrays['leaf_error_abs'].shape == (0, 2, 2)
    assert report['sampled_errors'] is None and report['moments'] == {}


def test_late_initial_panel_is_counted_but_discarded():
    state = dict(points=0)
    def observed(frequency):
        state['points'] += 1
        return matrix(1.)
    report, arrays = helper.run_full_domain_quadrature(
        np.linspace(-1., 1., 5), observed, 2, max_seconds=1.,
        clock=lambda: 2. if state['points'] >= 24 else 0.)
    assert report['point_evaluations'] == state['points'] == 24
    assert report['stop_reason'] == 'wall-budget' and report['leaf_count'] == 0
    assert not report['complete_partition'] and not report['sampled_tolerance_met']
    assert not any(name.endswith('_moment') for name in arrays)


def test_partial_initial_time_budget_retains_only_accepted_whole_intervals():
    state = dict(points=0)
    def observed(frequency):
        state['points'] += 1
        return matrix(1.)
    report, arrays = helper.run_full_domain_quadrature(
        np.linspace(-1., 1., 5), observed, 2, max_seconds=1.,
        clock=lambda: 2. if state['points'] > 24 else 0.)
    assert report['point_evaluations'] == state['points'] == 48
    assert report['stop_reason'] == 'wall-budget'
    assert report['initial_completed_intervals'] == report['leaf_count'] == 1
    np.testing.assert_array_equal(arrays['leaf_bounds'], [[-1., -.5]])
    assert report['sampled_errors'] is None and report['moments'] == {}


def test_failed_split_time_budget_keeps_complete_parent_and_counts_children():
    state = dict(points=0)
    def observed(frequency):
        state['points'] += 1
        return peak(frequency)
    grid = np.linspace(-1., 1., 5)
    report, arrays = helper.run_full_domain_quadrature(
        grid, observed, 2, max_seconds=1., rtol=1e-12, atol=0.,
        clock=lambda: 2. if state['points'] > 96 else 0.)
    assert report['point_evaluations'] == state['points'] == 144
    assert report['stop_reason'] == 'wall-budget' and report['leaf_count'] == 4
    assert report['complete_partition'] and not report['sampled_tolerance_met']
    np.testing.assert_array_equal(arrays['leaf_bounds'], np.column_stack((grid[:-1], grid[1:])))
    np.testing.assert_array_equal(arrays['full_domain_moment'], arrays['initial_high_moment'])


def test_failed_split_point_budget_does_not_submit_or_replace_half_parent():
    grid = np.linspace(-1., 1., 5)
    report, arrays, _ = execute(peak, grid, max_points=120, rtol=1e-12, atol=0.)
    assert report['point_evaluations'] == 96 and report['leaf_count'] == 4
    assert report['stop_reason'] == 'point-budget' and report['complete_partition']
    assert not report['sampled_tolerance_met']
    np.testing.assert_array_equal(arrays['full_domain_moment'], arrays['initial_high_moment'])


def test_leaf_limit_preserves_complete_partition_without_false_tolerance_pass():
    report, arrays, _ = execute(peak, max_leaves=8, rtol=1e-12, atol=0.)
    assert report['complete_initial_partition'] and report['complete_partition']
    assert report['stop_reason'] == 'leaf-budget' and not report['sampled_tolerance_met']
    assert report['leaf_count'] == 8 and report['point_evaluations'] == 192
    assert 'full_domain_moment' in arrays
    assert not report['sampled_errors']['passed']


def test_narrow_peaks_adapt_across_full_domain_and_agree_with_analytic_integral():
    report, arrays, _ = execute(peak, max_leaves=100)
    assert report['complete_partition'] and report['sampled_tolerance_met']
    assert report['leaf_count'] > report['initial_intervals']
    assert report['point_evaluations'] == 24 * 8 + 48 * (report['leaf_count'] - 8)
    width = .003
    total = sum(width * (math.atan((1. - sign * .31) / width)
                         - math.atan((-1. - sign * .31) / width)) for sign in (-1., 1.))
    np.testing.assert_allclose(arrays['full_domain_moment'], matrix(total), rtol=.002, atol=1e-8)
    for parent in range(8):
        bounds = arrays['leaf_bounds'][arrays['leaf_parent_indices'] == parent]
        assert bounds[0, 0] == arrays['grid_values'][parent]
        assert bounds[-1, 1] == arrays['grid_values'][parent + 1]
        np.testing.assert_array_equal(bounds[1:, 0], bounds[:-1, 1])


def test_cached_error_scores_preserve_both_norm_criteria_and_do_not_cancel():
    first = helper._leaf(0, -1., 0., np.zeros((2, 2)), np.diag([1., 0.]))
    second = helper._leaf(1, 0., 1., np.diag([2., 0.]), np.diag([1., 0.]))
    moment = first.high + second.high
    actual = helper._cached_errors([first, second], moment, .01, 0.)
    expected = helper._sampled_errors([first, second], .01, 0.)
    assert actual == expected and not actual['passed']
    assert actual['max_error'] == actual['frobenius_error'] == 2.
    # A distributed difference can pass the maximum criterion while failing
    # Frobenius. Both remain mandatory, including near the stopping threshold.
    leaf = helper._leaf(0, -1., 0., np.array([[1., -.009], [-.009, -.009]]),
                        np.diag([1., 0.]))
    errors = helper._cached_errors([leaf], leaf.high, .01, 0.)
    assert errors['max_error_ratio'] < 1. < errors['frobenius_error_ratio']
    assert not errors['passed']


@pytest.mark.parametrize('grid', [
    [-1., 0., np.nan], [-1., -.5, 0., .5], [-1., 0., 0.],
    [-1., -.5, 1e-15, .5, 1.], [-1., -.4, 0., .5, 1.],
])
def test_invalid_grid_rejected_before_evaluations(grid):
    with pytest.raises(ValueError, match='Grid'):
        helper.run_full_domain_quadrature(grid, lambda frequency: pytest.fail('must not evaluate'), 2)


@pytest.mark.parametrize('keyword,value', [
    ('ndof', 0), ('ndof', True), ('max_points', -1), ('max_points', 2.5),
    ('max_workers', 0), ('max_leaves', 1), ('max_seconds', np.inf),
    ('max_seconds', 0.), ('rtol', -1.), ('atol', np.nan),
])
def test_invalid_limits_rejected_before_evaluations(keyword, value):
    kwargs = dict(ndof=2)
    kwargs[keyword] = value
    with pytest.raises(ValueError):
        helper.run_full_domain_quadrature([-1., 0., 1.],
                                          lambda frequency: pytest.fail('must not evaluate'), **kwargs)


@pytest.mark.parametrize('result', [np.full((2, 2), np.nan), np.ones((3, 3))])
def test_invalid_pointwise_matrices_fail_closed(result):
    with pytest.raises(ValueError, match='nonfinite matrix or wrong shape'):
        helper.run_full_domain_quadrature([-1., 0., 1.], lambda frequency: result, 2)


def test_pointwise_exception_propagates_after_worker_cleanup():
    calls = []
    def failed(frequency):
        calls.append(frequency)
        raise RuntimeError('pointwise failure')
    with pytest.raises(RuntimeError, match='pointwise failure'):
        helper.run_full_domain_quadrature([-1., 0., 1.], failed, 2)
    assert 1 <= len(calls) <= 24


def test_symmetric_nonuniform_domain_is_supported_without_querying_zero():
    report, arrays, _ = execute(lambda frequency: matrix(1. + frequency ** 2),
                                np.array([-1., -.1, 0., .1, 1.]))
    assert report['sampled_tolerance_met']
    np.testing.assert_allclose(arrays['full_domain_moment'], matrix(8. / 3.), rtol=2e-15)
