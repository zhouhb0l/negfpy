"""Analytic curves and failure cases for saved-integrand memory diagnostics."""

import numpy as np
import pytest
from negfpy.anharmonic import diagnose_current_memory


def diagnose(t, y, **kwargs):
    settings = {'current_scale': 1., 'harmonic_current': 10., 'reference_current': 10.,
                    'window_start': 0.}
    settings.update(kwargs)
    return diagnose_current_memory(t, y, **settings)


def test_nonuniform_linear_integrand_has_exact_primitive_and_units():
    t = np.array([0., .3, 1., 2., 4.])
    y = 2 * t
    result = diagnose(t, y, current_scale=3., window_start=1., reference_current=-10.)
    np.testing.assert_allclose(result['connected_current'], 3 * t**2)
    np.testing.assert_allclose(result['total_current'], 10 + 3 * t**2)
    summary = result['summary']
    assert summary['peak_to_peak_current'] == pytest.approx(45.)
    assert summary['peak_to_peak_over_abs_reference'] == pytest.approx(4.5)
    assert summary['window_points'] == 3
    assert summary['minimum_at_lag'] == 1.
    assert not summary['changes_solver_acceptance']


def test_equal_window_endpoints_do_not_hide_interior_oscillation():
    # Integral endpoints at t=4 and t=8 are both zero, yet the stored
    # intermediate integrals reach +/-0.5. A two-endpoint test misses this.
    result = diagnose(np.arange(9.), [1, 0, -1, 0, 1, 0, -1, 0, 1], window_start=4.)
    s = result['summary']
    assert s['endpoint_change'] == 0.
    assert s['peak_to_peak_current'] == 1.
    assert s['maximum_at_lag'] == 5.
    assert s['minimum_at_lag'] == 7.
    assert s['range_over_abs_final_connected'] is None
    assert not s['convergence_certified'] and not s['exterior_tail_estimated']


def test_large_harmonic_offset_does_not_erase_small_component_variation():
    r = diagnose(np.arange(3.), [1., 1., 1.], harmonic_current=1e30, reference_current=0.)
    assert r['summary']['peak_to_peak_current'] == 2.
    assert r['summary']['peak_to_peak_over_abs_reference'] is None


def test_inputs_remain_writable_and_returned_curves_are_independent():
    t, y = np.arange(3.), np.ones(3)
    r = diagnose(t, y)
    t[1], y[1] = .5, 100
    np.testing.assert_array_equal(r['lag_times'], [0, 1, 2])
    np.testing.assert_array_equal(r['connected_current'], [0, 1, 2])
    assert all(not r[k].flags.writeable for k in ('lag_times', 'connected_current', 'total_current'))


@pytest.mark.parametrize('settings', [
    {'current_scale': 0}, {'current_scale': True}, {'window_start': .5},
    {'window_start': 2}, {'frequency_step': -1}, {'reference_current': np.nan},
    {'harmonic_current': 1j}, {'frequency_step': np.pi / 2},
])
def test_invalid_units_window_and_recurrence_rejected(settings):
    with pytest.raises(ValueError):
        diagnose([0, 1, 2], [1, 1, 1], **settings)


@pytest.mark.parametrize('t,y', [
    ([0, 1, 1], [1, 1, 1]), ([1, 2], [1, 1]),
    ([0, 1], [1, np.inf]), ([0, 1], [1, 1j]), ([0, 1], [1]),
])
def test_invalid_saved_arrays_rejected(t, y):
    with pytest.raises(ValueError):
        diagnose(t, y)


def test_recurrence_check_and_overflow():
    result = diagnose([0, 1, 2], [0, 0, 0], frequency_step=1.)
    assert result['summary']['half_recurrence_limit'] == np.pi
    assert result['summary']['recurrence_checked']
    with pytest.raises(ValueError, match='overflowed'):
        diagnose([0, 1, 2], [1e308, 1e308, 1e308])
    with pytest.raises(ValueError, match='Normalized current variation overflowed'):
        diagnose([0, 1, 2], [1, 1, 1], reference_current=1e-320)
