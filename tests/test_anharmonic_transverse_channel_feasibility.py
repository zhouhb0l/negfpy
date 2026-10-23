"""Proposal feasibility with tiny native arrays; no material-data replay."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import negfpy.anharmonic.transverse_channel_accel as accel
import negfpy.anharmonic.transverse_channel_feasibility as feasible
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    ChannelSpectrum,
    _field_checks,
    mix_channel_spectra,
)
from negfpy.anharmonic.transverse_solver import TransverseCurrents

SHAPE = (1, 3, 1, 1)
MESH = SimpleNamespace(n_qpoints=1, negative_index=lambda q: q)
GRID = SimpleNamespace(n_points=3)
POLICY = feasible.ChannelProposalFeasibility(mode='psd-backtracking')
KEYS = ('retarded', 'statistical', 'spectral')


def spectrum(c=1., d=.2):
    spectral = np.asarray([-1j*d, 0., 1j*d], complex).reshape(SHAPE)
    fields = [spectral / 2, np.full(SHAPE, -1j*c), spectral]
    for field in fields:
        field.setflags(write=False)
    return ChannelSpectrum(*fields)


def check(anchor, candidate, *, image=None, kind='anderson1', policy=POLICY, remaining=None):
    return feasible.feasible_channel_proposal(
        MESH, GRID, anchor, spectrum(.8) if image is None else image, candidate,
        original_kind=kind, mixing=.05, policy=policy, rtol=1e-8, atol=1e-11,
        time_remaining=remaining)


@pytest.mark.parametrize('kwargs', [
    {'mode': 'project'}, {'factor': 1.}, {'factor': 0.}, {'factor': True},
    {'factor': float('nan')}, {'max_halvings': 9}, {'max_halvings': -1},
    {'max_halvings': True}, {'max_halvings': 1.5},
])
def test_policy_rejects_unbounded_or_ambiguous_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        feasible.ChannelProposalFeasibility(**kwargs)


def test_screen_matches_original_full_input_field_guard_values():
    candidate = spectrum(.7)
    assert feasible.proposal_field_checks(MESH, GRID, candidate, 1e-8, 1e-11) == (
        _field_checks(MESH, GRID, candidate, 1e-8, 1e-11, 'input Sigma'))
    bad = spectrum(-3)
    checks, exports = feasible.proposal_field_checks(MESH, GRID, bad, 1e-8, 1e-11)
    assert {key for key, row in checks.items() if row['ratio'] > 1} == {'lesser_psd', 'greater_psd'}
    assert all(row['ratio'] <= 1 for row in exports.values())
    with pytest.raises(ValueError, match='Invalid input Sigma'):
        _field_checks(MESH, GRID, bad, 1e-8, 1e-11, 'input Sigma')


def test_feasible_original_is_returned_byte_exact_and_read_only():
    anchor, candidate = spectrum(), spectrum(.8)
    before = {key: getattr(candidate, key).tobytes() for key in KEYS}
    result, report = check(anchor, candidate)
    assert result is candidate and report['passed'] and report['chosen_factor'] == 1.
    assert report['screen_count'] == 1 and not report['fallback_used']
    for key in KEYS:
        assert getattr(result, key).tobytes() == before[key]
        assert not getattr(result, key).flags.writeable


def test_one_common_damping_preserves_independent_fields_without_projection():
    anchor, candidate = spectrum(), spectrum(-3)
    originals = [{key: getattr(value, key).tobytes() for key in KEYS}
                 for value in (anchor, candidate)]
    result, report = check(anchor, candidate)
    assert report['chosen_factor'] == .125 and report['selected_kind'] == 'anderson1'
    assert [row['factor'] for row in report['screens']] == [1., .5, .25, .125]
    assert [row['passed'] for row in report['screens']] == [False, False, False, True]
    for row in report['screens'][:-1]:
        assert set(row['failed_keys']) == {'lesser_psd', 'greater_psd'}
        assert row['error'] is None
    expected = mix_channel_spectra(anchor, candidate, .125)
    for key in KEYS:
        np.testing.assert_array_equal(getattr(result, key), getattr(expected, key))
        assert not getattr(result, key).flags.writeable
    np.testing.assert_allclose(result.spectral, result.retarded - result.retarded.swapaxes(-1, -2).conj())
    assert originals == [{key: getattr(value, key).tobytes() for key in KEYS}
                         for value in (anchor, candidate)]


def test_psd_exhaustion_has_one_explicit_fresh_linear_fallback():
    anchor, candidate, image = spectrum(), spectrum(-10), spectrum(.8)
    policy = replace(POLICY, max_halvings=1)
    result, report = check(anchor, candidate, image=image, policy=policy)
    assert report['fallback_used'] and report['selected_kind'] == 'linear_feasibility_fallback'
    assert report['chosen_factor'] is None and report['screen_count'] == report['maximum_screen_count'] == 3
    assert [row['kind'] for row in report['screens']] == ['original', 'damped', 'linear_fallback']
    assert [row['factor'] for row in report['screens']] == [1., .5, None]
    expected = mix_channel_spectra(anchor, image, .05)
    for key in KEYS:
        np.testing.assert_array_equal(getattr(result, key), getattr(expected, key))


def test_failed_single_linear_fallback_stops_with_all_screen_evidence():
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), spectrum(-10), image=spectrum(-100),
              policy=replace(POLICY, max_halvings=1))
    report = exc.value.report
    assert report['stop_reason'] == 'linear_fallback_failure'
    assert report['fallback_used'] and not report['passed'] and report['screen_count'] == 3
    assert not report['map_admitted'] and all(not row['passed'] for row in report['screens'])


def test_full_eight_halving_bound_has_nine_screens_then_only_one_fallback():
    _, report = check(spectrum(), spectrum(-1e9))
    assert report['fallback_used'] and report['screen_count'] == report['maximum_screen_count'] == 10
    assert [row['factor'] for row in report['screens'][:-1]] == [2.**(-n) for n in range(9)]
    assert report['screens'][-1]['kind'] == 'linear_fallback'
    assert report['screens'][-1]['factor'] is None


@pytest.mark.parametrize('kind', ['linear_warmup', 'linear_fallback', 'linear_proposal_fallback'])
def test_linear_candidates_are_never_silently_changed_or_retried(kind):
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), spectrum(-3), kind=kind)
    assert exc.value.report['stop_reason'] == 'linear_input_failure'
    assert exc.value.report['screen_count'] == 1 and not exc.value.report['fallback_used']


def test_non_psd_identity_failure_stops_without_backtracking():
    candidate = spectrum(-3)
    broken = ChannelSpectrum(candidate.retarded + .1j, candidate.statistical, candidate.spectral)
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), broken)
    report = exc.value.report
    assert report['stop_reason'] == 'non_psd_guard_failure' and report['screen_count'] == 1
    assert 'keldysh_identity' in report['screens'][0]['failed_keys']


def test_nonfinite_candidate_is_hard_failure_with_no_repair():
    candidate = spectrum()
    broken = ChannelSpectrum(candidate.retarded, candidate.statistical * np.nan, candidate.spectral)
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), broken)
    report = exc.value.report
    assert report['stop_reason'] == 'nonfinite_or_validation_failure'
    assert report['screen_count'] == 1 and report['screens'][0]['error']
    assert report['screens'][0]['checks'] == {} and not report['fallback_used']


def test_nonfinite_metric_cannot_be_treated_as_a_psd_only_rejection(monkeypatch):
    monkeypatch.setattr(feasible, 'proposal_field_checks', lambda *args: (
        {'lesser_psd': {'error': 1., 'scale': 1., 'allowance': 1e-320, 'ratio': float('inf')}}, {}))
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), spectrum(-3))
    assert exc.value.report['stop_reason'] == 'nonfinite_or_validation_failure'
    assert exc.value.report['screen_count'] == 1
    assert 'Nonfinite' in exc.value.report['error']


def test_screening_time_budget_stops_before_any_input_scan():
    with pytest.raises(feasible.ChannelProposalFailure) as exc:
        check(spectrum(), spectrum(-3), remaining=lambda: 0.)
    assert exc.value.time_budget and exc.value.report['stop_reason'] == 'time_budget'
    assert exc.value.report['screens'] == [] and not exc.value.report['map_admitted']


def run_fixture(monkeypatch, *, proposal=None, max_evaluations=2):
    vertex = SimpleNamespace(mesh=MESH, full_ndof=1, ndof=1)
    contact = SimpleNamespace(device_indices=(0,))
    calls = []
    callbacks = []

    def evaluate(*args, sigma_input, **kwargs):
        calls.append(sigma_input)
        # The map's original validation still happens after the screening.
        _field_checks(MESH, GRID, sigma_input, kwargs['identity_rtol'], kwargs['identity_atol'], 'input Sigma')
        return SimpleNamespace(map_sigma=spectrum(.9), green=spectrum(),
                               input_currents=TransverseCurrents(0., 0., 0.))

    monkeypatch.setattr(accel, 'evaluate_channel_born_map', evaluate)
    if proposal is not None:
        monkeypatch.setattr(accel, 'propose_channel_anderson1', lambda *args, **kwargs:
                            (proposal, {'scales': [1., 1., 1.], 'reference_merit': 1.}))
    config = replace(ChannelSCBAConfig(), max_evaluations=max_evaluations, mixing=.05,
                     max_seconds=60, time_reserve_seconds=1,
                     initial_map_estimate_seconds=.001)
    kwargs = {'seed_input': spectrum(), 'seed_mapped': spectrum(.8), 'seed_index': 19,
              'config': config, 'feasibility': POLICY,
              'on_proposal': lambda idx, batch: callbacks.append(('screen', idx, batch)),
              'on_start': lambda idx, supplied, details: callbacks.append(('start', idx, details)),
              'on_evaluation': lambda state, decision: callbacks.append(('complete', state.index, decision))}
    return (GRID, vertex, np.zeros((1, 1, 1)), contact, contact), kwargs, calls, callbacks


def test_screens_precede_each_actual_map_and_do_not_consume_fresh_map_budget(monkeypatch):
    args, kwargs, calls, callbacks = run_fixture(monkeypatch, proposal=spectrum(-3))
    result = accel.run_channel_accel(*args, **kwargs)
    assert len(calls) == len(result.history) == 2
    assert [row[:2] for row in callbacks] == [('screen', 20), ('start', 20), ('complete', 20),
                                              ('screen', 21), ('start', 21), ('complete', 21)]
    assert len(result.proposal_screens) == 2 and result.proposal_screens[1]['screen_count'] > 1
    assert all(row['map_admitted'] for row in result.proposal_screens)
    assert result.proposal_screens[1] == callbacks[4][2]['feasibility']
    assert callbacks[4][2]['kind'] == 'anderson1'
    assert callbacks[4][2]['feasibility']['chosen_factor'] == .125
    assert all(_field_checks(MESH, GRID, supplied, 1e-8, 1e-11, 'input Sigma')
               for supplied in calls)
    np.testing.assert_array_equal(calls[0].statistical, mix_channel_spectra(spectrum(), spectrum(.8), .05).statistical)


def test_hard_screen_failure_is_saved_before_no_new_map_attempt(monkeypatch):
    candidate = spectrum(-3)
    broken = ChannelSpectrum(candidate.retarded + .1j, candidate.statistical, candidate.spectral)
    args, kwargs, calls, callbacks = run_fixture(monkeypatch, proposal=broken)
    result = accel.run_channel_accel(*args, **kwargs)
    assert result.stop_reason == 'proposal_error' and 'keldysh_identity' in result.failure_reason
    assert len(calls) == len(result.history) == 1 and result.state.index == 20
    assert [row[:2] for row in callbacks][-1] == ('screen', 21)
    assert result.proposal_screens[-1]['index'] == 21 and not result.proposal_screens[-1]['map_admitted']
    assert result.proposal_screens[-1]['anchor_index'] == 20


def test_feasibility_adds_one_spectrum_and_block_workspace_only_when_enabled():
    plain = accel.plan_channel_accel(4, 601, 216, 120, 24, 24)
    enabled = accel.plan_channel_accel(4, 601, 216, 120, 24, 24, feasibility=POLICY)
    assert enabled.additional_retained_bytes - plain.additional_retained_bytes == plain.fixed_plan.input_sigma_bytes
    assert enabled.block_workspace_bytes - plain.block_workspace_bytes == 16*16*120**2
    assert enabled.peak_owned_bytes < 32*1024**3
    assert enabled.nominal_caller_bytes == plain.nominal_caller_bytes
    assert enabled.maximum_contraction_work == plain.maximum_contraction_work
    assert enabled.maximum_dyson_evaluations == plain.maximum_dyson_evaluations


def test_postscreen_map_admission_keeps_extra_unadmitted_batch(monkeypatch):
    args, kwargs, calls, callbacks = run_fixture(monkeypatch)
    clock = [0.]
    monkeypatch.setattr(accel, 'monotonic', lambda: clock[0])
    original = accel.feasible_channel_proposal
    def screen_then_timeout(*args, **kw):
        value, report = original(*args, **kw)
        clock[0] = 60.
        return value, report
    monkeypatch.setattr(accel, 'feasible_channel_proposal', screen_then_timeout)
    result = accel.run_channel_accel(*args, **kwargs)
    assert result.stop_reason == 'max_seconds' and result.failure_reason is None
    assert not calls and not result.history
    assert [row[:2] for row in callbacks] == [('screen', 20)]
    assert result.proposal_screens[0]['passed'] and not result.proposal_screens[0]['map_admitted']
    assert result.proposal_screens[0]['stop_reason'] == 'map_time_budget'
