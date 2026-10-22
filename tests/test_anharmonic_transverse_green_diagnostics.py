"""New tiny fixtures for bounded, non-accepting Green failure diagnostics."""

from __future__ import annotations

import json

import numpy as np
import pytest

from negfpy.anharmonic import transverse_material_map as material_map
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_green_diagnostics import (
    FullGreenDiagnosticCollector,
    green_diagnostic_workspace_bytes,
)
from negfpy.anharmonic.transverse_material_map import (
    PackedBornMapEvaluation,
    PackedGreenDiagnostic,
    evaluate_packed_born_map,
    plan_material_born_map,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseSpectrum


def _small_fixture():
    mesh = UniformBZMesh((1, 1, 1))
    grid = SignedFrequencyGrid(step=0.23, n_positive=3)
    vertex = PackedTransverseCubicVertex(mesh, np.zeros((1, 1, 1, 1, 1), complex),
                                        np.array([0]), 1)
    contacts = []
    for side, damping, temperature in (('left', 0.17, 0.75), ('right', 0.11, 0.62)):
        omega = grid.values
        occupation_times_omega = np.full(len(omega), temperature)
        nonzero = omega != 0
        occupation_times_omega[nonzero] = omega[nonzero]/np.expm1(omega[nonzero]/temperature)
        retarded = (-1j*damping*omega).reshape(1, -1, 1, 1)
        lesser = (-2j*damping*occupation_times_omega).reshape(1, -1, 1, 1)
        greater = lesser+retarded-retarded.conj()
        contact = TransverseContact(
            mesh, grid, TransverseSpectrum(retarded, lesser, greater),
            np.array([[[2*damping]]], complex), ({'fixture': 'diagnostic-scalar'},), {},
            side, 0.02, temperature,
        )
        contacts.append(CompactTransverseContact(contact, 1, (0,)))
    return grid, vertex, np.array([[[1.7]]], complex), *contacts


def _finish(collector, metrics):
    return collector.finish(metrics, full_green_validation_error=None,
                            active_green_validation_error=None, validation_checks={})


def _metric(retarded, lesser, greater):
    first, second = greater-lesser, retarded-retarded.conj().T
    return float(np.max(np.abs(first-second))), max(float(np.max(np.abs(first))),
                                                   float(np.max(np.abs(second))))


def _assert_scalar_json(value):
    json.dumps(value, allow_nan=False)
    if isinstance(value, dict):
        for child in value.values():
            _assert_scalar_json(child)
    elif isinstance(value, list):
        for child in value:
            _assert_scalar_json(child)
    else:
        assert value is None or type(value) in (str, float, int, bool)


def test_global_error_scale_and_local_ratio_have_independent_locations_and_bounded_points():
    collector = FullGreenDiagnosticCollector(n_q=1, n_frequency=3, rtol=1e-4, atol=1e-12)
    maxima = [0.0, 0.0]
    for w, (spectral, error) in enumerate(((100.0, 0.001), (0.01, 0.00005), (400.0, 0.0005))):
        retarded = np.array([[-0.5j*spectral]])
        lesser = np.array([[-2j]])
        greater = lesser-1j*(spectral+error)
        collector.observe(0, w, float(w), (0, 0, 0), np.array([[1j]]), retarded,
                          np.array([[-1j]]), np.array([[-3j]]), lesser, greater)
        maxima = [max(old, new) for old, new in zip(maxima, _metric(retarded, lesser, greater))]
        assert len(collector._snapshots) <= 2
    result = _finish(collector, {'keldysh_identity': tuple(maxima)})
    global_check = result['full_green_keldysh']
    assert global_check['error_location']['frequency_index'] == 0
    assert global_check['scale_location']['frequency_index'] == 2
    assert global_check['maximum_pointwise_ratio_location']['frequency_index'] == 1
    assert global_check['ratio'] == pytest.approx(maxima[0]/(1e-12+1e-4*maxima[1]))
    assert global_check['ratio'] < 1 < global_check['maximum_pointwise_ratio']
    assert result['retained_point_count'] == 2
    assert {point['frequency_index'] for point in result['extrema_contexts']} == {0, 1}
    assert result['all_points_observed'] is True
    _assert_scalar_json(result)


def test_source_decomposition_and_inverse_backward_error_use_existing_inverse(monkeypatch):
    operator = np.array([[1.2+0.3j, 0.1], [0.1, 1.8+0.4j]])
    retarded = np.linalg.inv(operator)
    source_less = -0.2j*np.eye(2)
    injected = np.diag([-2e-6j, -1e-6j])
    source_greater = source_less+operator.conj().T-operator+injected
    lesser = retarded@source_less@retarded.conj().T
    greater = retarded@source_greater@retarded.conj().T
    collector = FullGreenDiagnosticCollector(n_q=1, n_frequency=1, rtol=1e-8, atol=1e-11)
    collector.observe(0, 0, 0.3, (0, 0, 0), operator, retarded,
                      source_less, source_greater, lesser, greater)
    monkeypatch.setattr(np.linalg, 'inv', lambda *a, **k: pytest.fail('Diagnostic reinverted A'))
    context = _finish(collector, {'keldysh_identity': _metric(retarded, lesser, greater)})[
        'extrema_contexts'][0]
    conditioning = np.linalg.norm(operator, ord=np.inf)*np.linalg.norm(retarded, ord=np.inf)
    assert context['inverse']['condition_estimate'] == pytest.approx(conditioning)
    left_error = np.linalg.norm(np.eye(2)-operator@retarded, ord=np.inf)
    right_error = np.linalg.norm(np.eye(2)-retarded@operator, ord=np.inf)
    assert context['inverse']['left_absolute_residual'] == pytest.approx(left_error, abs=1e-30)
    assert context['inverse']['right_absolute_residual'] == pytest.approx(right_error, abs=1e-30)
    assert context['source_identity']['error'] == pytest.approx(2e-6)
    propagated = retarded@injected@retarded.conj().T
    assert context['decomposition']['propagated_source_defect']['max_abs'] == pytest.approx(
        np.max(np.abs(propagated)), rel=1e-9)
    assert context['decomposition']['remainder']['max_abs'] < 3e-16
    assert context['decomposition']['separate_products_minus_difference_product']['max_abs'] < 3e-16
    assert context['selection_reasons'] == ['maximum_absolute_error', 'maximum_pointwise_ratio']


def test_cancellation_records_large_terms_and_safe_unbounded_ratio():
    collector = FullGreenDiagnosticCollector(n_q=1, n_frequency=1, rtol=1e-8, atol=1e-11)
    operator = retarded = np.eye(1, dtype=complex)
    lesser = greater = np.array([[-1e15j]])
    collector.observe(0, 0, 0, (0, 0, 0), operator, retarded, lesser, greater, lesser, greater)
    result = _finish(collector, {'keldysh_identity': (0.0, 0.0)})
    cancellation = result['extrema_contexts'][0]['cancellation']
    assert cancellation['lesser_max_abs'] == 1e15
    assert cancellation['greater_minus_lesser_max_abs'] == 0
    assert cancellation['global_magnitude_ratio'] is None
    assert any('cancellation.global_magnitude_ratio' in path for path in result['nonfinite_fields'])
    _assert_scalar_json(result)


def test_extrema_snapshot_is_detached_from_reused_solver_buffers():
    collector = FullGreenDiagnosticCollector(n_q=1, n_frequency=1, rtol=1e-8, atol=1e-11)
    operator = np.array([[1+0.3j]])
    retarded = np.linalg.inv(operator)
    source_less, source_greater = np.array([[-0.1j]]), np.array([[-0.7j]])
    lesser, greater = retarded@source_less@retarded.conj().T, retarded@source_greater@retarded.conj().T
    metric = _metric(retarded, lesser, greater)
    collector.observe(0, 0, 0.2, (0, 0, 0), operator, retarded, source_less,
                      source_greater, lesser, greater)
    for array in (operator, retarded, source_less, source_greater, lesser, greater):
        array[:] = 999
    result = _finish(collector, {'keldysh_identity': metric})
    assert result['extrema_contexts'][0]['inverse']['operator_norm'] == pytest.approx(abs(1+0.3j))


def test_diagnostic_pass_returns_before_born_with_full_grid_coverage(monkeypatch):
    arguments = _small_fixture()
    inverse = np.linalg.inv
    calls = []
    def counted_inverse(operator):
        calls.append(operator.copy())
        return inverse(operator)
    monkeypatch.setattr(np.linalg, 'inv', counted_inverse)
    monkeypatch.setattr(material_map, 'cubic_born_self_energy_q_packed_fft',
                        lambda *a, **k: pytest.fail('Diagnostic reached Born'))
    actual = evaluate_packed_born_map(*arguments, g=0.03, green_diagnostic_only=True)
    assert isinstance(actual, PackedGreenDiagnostic)
    assert actual.completed is actual.green_passed is True
    assert len(calls) == arguments[0].n_points
    assert actual.diagnostics['completed_points'] == actual.plan.dyson_evaluations
    assert actual.diagnostics['full_green_validation_error'] is None
    assert actual.diagnostics['active_green_validation_error'] is None
    assert not hasattr(actual, 'map_sigma') and not hasattr(actual, 'input_currents')
    assert actual.diagnostic_peak_owned_bytes == actual.plan.peak_owned_bytes+actual.diagnostic_workspace_bytes
    _assert_scalar_json(actual.diagnostics)


def test_full_green_failure_is_reported_without_bypass_or_second_inverse(monkeypatch):
    arguments = _small_fixture()
    inverse = np.linalg.inv
    calls = []
    def perturbed_inverse(operator):
        calls.append(None)
        return (1+1e-4)*inverse(operator)
    monkeypatch.setattr(np.linalg, 'inv', perturbed_inverse)
    monkeypatch.setattr(material_map, 'cubic_born_self_energy_q_packed_fft',
                        lambda *a, **k: pytest.fail('Failed diagnostic reached Born'))
    actual = evaluate_packed_born_map(*arguments, green_diagnostic_only=True)
    assert actual.completed is True and actual.green_passed is False
    assert len(calls) == arguments[0].n_points
    assert 'Invalid full Green' in actual.diagnostics['full_green_validation_error']
    assert actual.diagnostics['full_green_keldysh']['ratio'] > 1
    assert any(point['inverse']['left_absolute_residual'] > 1e-5
               for point in actual.diagnostics['extrema_contexts'])
    assert actual.diagnostics['all_points_observed'] is True
    _assert_scalar_json(actual.diagnostics)


def test_default_mode_retains_existing_guard_failure(monkeypatch):
    inverse = np.linalg.inv
    monkeypatch.setattr(np.linalg, 'inv', lambda operator: (1+1e-4)*inverse(operator))
    with pytest.raises(ValueError, match='Invalid full Green'):
        evaluate_packed_born_map(*_small_fixture())


def test_default_mode_does_not_create_diagnostic_collector(monkeypatch):
    monkeypatch.setattr(material_map, 'FullGreenDiagnosticCollector',
                        lambda **k: pytest.fail('Default mode enabled diagnostic'))
    result = evaluate_packed_born_map(*_small_fixture(), g=0)
    assert isinstance(result, PackedBornMapEvaluation)


def test_diagnostic_extra_workspace_is_gated_before_dyson(monkeypatch):
    arguments = _small_fixture()
    plan = plan_material_born_map(1, arguments[0].n_points, 1, 1, 1, 1)
    monkeypatch.setattr(np.linalg, 'inv', lambda *a, **k: pytest.fail('Resource gate reached Dyson'))
    with pytest.raises(ValueError, match='max_bytes'):
        evaluate_packed_born_map(*arguments, green_diagnostic_only=True, max_bytes=plan.peak_owned_bytes)


@pytest.mark.parametrize('option', [1, None, np.bool_(True)])
def test_diagnostic_mode_requires_explicit_bool(option):
    with pytest.raises(TypeError, match='green_diagnostic_only'):
        evaluate_packed_born_map(*_small_fixture(), green_diagnostic_only=option)


def test_diagnostic_mode_keeps_input_validation_before_solving(monkeypatch):
    grid, vertex, dynamical, left, right = _small_fixture()
    dynamical[:] = -1
    monkeypatch.setattr(np.linalg, 'inv', lambda *a, **k: pytest.fail('Invalid input reached Dyson'))
    with pytest.raises(ValueError, match='stable'):
        evaluate_packed_born_map(grid, vertex, dynamical, left, right, green_diagnostic_only=True)


def test_singular_dyson_remains_an_error_without_complete_diagnostic(monkeypatch):
    def singular(operator):
        raise np.linalg.LinAlgError('synthetic singular matrix')
    monkeypatch.setattr(np.linalg, 'inv', singular)
    with pytest.raises(ValueError, match='Singular full-device Dyson solve at q=0, w=0'):
        evaluate_packed_born_map(*_small_fixture(), green_diagnostic_only=True)


@pytest.mark.parametrize('dimension', [0, -1, True, 1.5])
def test_workspace_bound_rejects_bad_dimension(dimension):
    with pytest.raises((ValueError, TypeError)):
        green_diagnostic_workspace_bytes(dimension)


def test_workspace_bound_covers_two_six_matrix_points_and_scales_quadratically():
    assert green_diagnostic_workspace_bytes(216) == 16*64*216**2
    assert green_diagnostic_workspace_bytes(216) > 2*6*16*216**2
    assert green_diagnostic_workspace_bytes(4) == 4*green_diagnostic_workspace_bytes(2)
