"""New native-controller algebra and fresh-trial lifecycle; synthetic data only."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import negfpy.anharmonic.transverse_channel_accel as accel
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    ChannelSpectrum,
)
from negfpy.anharmonic.transverse_solver import TransverseCurrents

KEYS = ('retarded', 'statistical', 'spectral')
SHAPE = (1, 3, 1, 1)


def spectrum(values):
    if np.isscalar(values):
        values = (values,) * 3
    arrays = [np.broadcast_to(np.asarray(value, complex), SHAPE).copy() for value in values]
    for array in arrays:
        array.setflags(write=False)
    return ChannelSpectrum(*arrays)


def scale_spectrum(value, factors):
    return spectrum(tuple(getattr(value, key) * factor
                          for key, factor in zip(KEYS, factors, strict=True)))


def dense_vector(value, scales):
    return np.concatenate([part for key, scale in zip(KEYS, scales, strict=True)
                           for part in (getattr(value, key).real.ravel() / scale,
                                        getattr(value, key).imag.ravel() / scale)])


def combine(first, second, a=1., b=1.):
    return spectrum(tuple(a * getattr(first, key) + b * getattr(second, key) for key in KEYS))


def fixture(monkeypatch, map_function=None):
    grid = SimpleNamespace(n_points=3)
    vertex = SimpleNamespace(mesh=SimpleNamespace(n_qpoints=1), full_ndof=1, ndof=1)
    contact = SimpleNamespace(device_indices=(0,))
    calls = []

    def evaluate(*args, sigma_input, **kwargs):
        calls.append(sigma_input)
        if map_function is None:
            mapped = spectrum(tuple(.5 * getattr(sigma_input, key) + 1 for key in KEYS))
        else:
            mapped = map_function(sigma_input, len(calls))
        return SimpleNamespace(map_sigma=mapped, green=spectrum(0),
                               input_currents=TransverseCurrents(0., 0., 0.))

    monkeypatch.setattr(accel, 'evaluate_channel_born_map', evaluate)
    config = replace(ChannelSCBAConfig(), max_evaluations=6, mixing=.05,
                     max_seconds=60, time_reserve_seconds=1,
                     initial_map_estimate_seconds=.001)
    return (grid, vertex, np.zeros((1, 1, 1)), contact, contact), config, calls


@pytest.mark.parametrize('kwargs', [{'mode': 'unknown'}, {'trust_radius': 0},
                                   {'rank_rtol': 1}, {'coefficient_limit': True},
                                   {'coefficient_limit': float('inf')}])
def test_new_controls_reject_invalid_values(kwargs):
    with pytest.raises((TypeError, ValueError)):
        accel.ChannelAccelerationConfig(**kwargs)


def test_streamed_proposal_matches_independent_dense_real_secant():
    rng = np.random.default_rng(590)
    fields = [spectrum(tuple(rng.normal(size=SHAPE) + 1j * rng.normal(size=SHAPE)
                             for _ in KEYS)) for _ in range(4)]
    previous, old_image, current, image = fields
    alpha = .05
    proposal, details = accel.propose_channel_anderson1(
        *fields, alpha, acceleration=accel.ChannelAccelerationConfig(trust_radius=10))
    scales = tuple(max(np.abs(getattr(current, key)).max(),
                       np.abs(getattr(image, key)).max(), 1e-10) for key in KEYS)
    f = combine(image, current, b=-1)
    old_f = combine(old_image, previous, b=-1)
    df = combine(f, old_f, b=-1)
    vector_f, vector_df = dense_vector(f, scales), dense_vector(df, scales)
    raw_gamma = vector_df @ vector_f / (vector_df @ vector_df)
    gamma = np.clip(raw_gamma, -5., 5.)
    assert details['raw_coefficient'] == pytest.approx(raw_gamma, rel=2e-14)
    assert details['coefficient'] == pytest.approx(gamma)
    assert details['clip_factor'] == 1
    for key in KEYS:
        expected = (getattr(current, key) + alpha * getattr(f, key)
                    - gamma * (getattr(current, key) - getattr(previous, key)
                               + alpha * getattr(df, key)))
        np.testing.assert_allclose(getattr(proposal, key), expected, rtol=2e-14, atol=2e-14)
        assert not getattr(proposal, key).flags.writeable
        assert not np.shares_memory(getattr(proposal, key), getattr(current, key))


def test_fit_is_invariant_to_twelve_orders_of_statistical_units():
    fields = (spectrum((1., -2j, .5j)), spectrum((1.5, -3j, .8j)),
              spectrum((1.1, -2.2j, .55j)), spectrum((1.4, -2.8j, .7j)))
    policy = accel.ChannelAccelerationConfig(trust_radius=.25)
    first, one = accel.propose_channel_anderson1(*fields, .05, acceleration=policy)
    factors = (1e-3, 1e12, 1e3)
    second, two = accel.propose_channel_anderson1(
        *(scale_spectrum(value, factors) for value in fields), .05, acceleration=policy)
    assert one['raw_coefficient'] == pytest.approx(two['raw_coefficient'], rel=5e-14)
    assert one['clip_factor'] == pytest.approx(two['clip_factor'], rel=5e-14)
    for key, factor in zip(KEYS, factors, strict=True):
        np.testing.assert_allclose(getattr(first, key), getattr(second, key) / factor, rtol=5e-14)


def test_common_coefficient_and_uniform_clip_preserve_linear_channel_relation():
    fields = [spectrum((2 * x, -1e12j * x, x)) for x in (1., 2., 1.05, 2.025)]
    proposal, details = accel.propose_channel_anderson1(
        *fields, .05, acceleration=accel.ChannelAccelerationConfig(trust_radius=.01))
    assert details['coefficient_clipped'] and details['raw_coefficient'] < -5
    assert details['bounded_coefficient'] == -5
    assert 0 < details['clip_factor'] < 1
    assert details['scaled_step_norm'] <= details['scaled_step_bound'] * (1 + 1e-14)
    np.testing.assert_allclose(proposal.retarded, 2 * proposal.spectral, rtol=1e-15)
    np.testing.assert_allclose(proposal.statistical, -1e12j * proposal.spectral, rtol=1e-15)


def test_flat_residual_history_has_explicit_cheap_fallback():
    proposal, details = accel.propose_channel_anderson1(
        spectrum(1), spectrum(2), spectrum(3), spectrum(4), .05)
    assert proposal is None and details['reason'] == 'degenerate_history'


def test_plan_reserves_seed_history_rollback_and_stops_before_map(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    plan = accel.plan_channel_accel(4, 601, 216, 120, 24, 24, config=config)
    assert plan.additional_retained_bytes == 6 * 3 * 4 * 601 * 120**2 * 16
    assert plan.peak_owned_bytes == (plan.fixed_plan.peak_owned_bytes
                                     + plan.additional_retained_bytes + plan.block_workspace_bytes)
    assert plan.peak_owned_bytes < 32 * 1024**3
    with pytest.raises(ValueError, match='allocation plan'):
        accel.run_channel_accel(*args, seed_input=spectrum(0), seed_mapped=spectrum(1),
                                config=replace(config, max_bytes=1))
    assert not calls


def test_linear_control_is_exact_matched_six_map_trajectory(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    original_input, original_map = spectrum(0), spectrum(1)
    events = []
    result = accel.run_channel_accel(
        *args, seed_input=original_input, seed_mapped=original_map, config=config,
        acceleration=accel.ChannelAccelerationConfig(mode='linear-control'),
        on_evaluation=lambda state, decision: events.append((state.index, decision)))
    expected = .05
    for supplied in calls:
        np.testing.assert_array_equal(supplied.retarded, np.full(SHAPE, expected, complex))
        expected = .95 * expected + .05 * (.5 * expected + 1)
    assert [row[0] for row in events] == list(range(16, 22))
    assert all(row[1]['accepted_as_anchor'] for row in events)
    assert result.state is result.last_evaluated_state and result.state.index == 21
    assert len(result.history) == len(result.decisions) == len(calls) == 6
    assert not result.converged and result.failure_reason is None
    np.testing.assert_array_equal(original_input.retarded, 0)
    np.testing.assert_array_equal(original_map.retarded, 1)


def test_bounded_partial_secant_advances_affine_contraction_without_starvation(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    proposals = []
    result = accel.run_channel_accel(
        *args, seed_input=spectrum(0), seed_mapped=spectrum(1), config=config,
        on_start=lambda index, sigma, metadata: proposals.append(metadata))
    assert len(calls) == 6 and result.state.index == 21
    assert proposals[1]['kind'] == 'anderson1'
    assert proposals[1]['raw_coefficient'] == pytest.approx(-39)
    assert proposals[1]['bounded_coefficient'] == -5
    assert all(row['accepted_as_anchor'] for row in result.decisions)
    assert all(row['kind'] == 'anderson1' for row in result.decisions[1:])
    assert (result.history[-1].sigma_residuals['retarded'].error_frobenius
            < .6 * result.history[0].sigma_residuals['retarded'].error_frobenius)


def _bad_proposal(previous, old_image, current, image, mixing, **kwargs):
    scales = accel.channel_scales(current, image, 1e-10)
    return combine(current, spectrum(100)), {
        'scales': list(scales), 'reference_merit': accel.channel_merit(current, image, scales),
        'coefficient': 1., 'reason': 'proposed'}


def test_rejected_trial_is_saved_but_fallback_uses_retained_anchor(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    monkeypatch.setattr(accel, 'propose_channel_anderson1', _bad_proposal)
    completed = []
    result = accel.run_channel_accel(
        *args, seed_input=spectrum(0), seed_mapped=spectrum(1),
        config=replace(config, max_evaluations=3),
        on_evaluation=lambda state, decision: completed.append((state, decision)))
    assert [state.index for state, _ in completed] == [16, 17, 18]
    assert completed[1][1]['accepted_as_anchor'] is False
    assert completed[1][1]['selected_anchor_index'] == 16
    assert completed[2][1]['kind'] == 'linear_fallback'
    np.testing.assert_array_equal(calls[2].retarded, np.full(SHAPE, .09875, complex))
    assert result.state.index == result.last_evaluated_state.index == 18


def test_final_rejected_trial_is_separate_from_selected_anchor(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    monkeypatch.setattr(accel, 'propose_channel_anderson1', _bad_proposal)
    result = accel.run_channel_accel(*args, seed_input=spectrum(0), seed_mapped=spectrum(1),
                                     config=replace(config, max_evaluations=2))
    assert len(calls) == 2 and len(result.history) == 2
    assert result.state.index == 16 and result.last_evaluated_state.index == 17
    assert result.decisions[-1]['selected_anchor_index'] == 16


def test_original_retarded_frobenius_vetoes_better_aggregate_merit(monkeypatch):
    def mapped(sigma, number):
        return (spectrum(1) if number == 1 else
                spectrum((sigma.retarded + 1.1, sigma.statistical, sigma.spectral)))

    args, config, calls = fixture(monkeypatch, mapped)
    monkeypatch.setattr(accel, 'propose_channel_anderson1', _bad_proposal)
    result = accel.run_channel_accel(*args, seed_input=spectrum(0), seed_mapped=spectrum(1),
                                     config=replace(config, max_evaluations=2))
    decision = result.decisions[-1]
    assert decision['trial_merit'] < decision['reference_merit']
    assert not decision['unmixed_frobenius_nonincrease']['retarded']
    assert not decision['accepted_as_anchor'] and result.state.index == 16
    assert len(calls) == 2


def test_hard_map_failure_stops_without_retry_or_replacing_anchor(monkeypatch):
    def mapped(sigma, number):
        if number == 2:
            raise ValueError('raw source hard gate')
        return spectrum(1)

    args, config, calls = fixture(monkeypatch, mapped)
    result = accel.run_channel_accel(*args, seed_input=spectrum(0), seed_mapped=spectrum(1),
                                     config=config)
    assert len(calls) == 2 and len(result.history) == 1
    assert result.state.index == result.last_evaluated_state.index == 16
    assert result.stop_reason == 'evaluation_error' and 'raw source hard gate' in result.failure_reason


def test_initial_candidate_copied_byte_exactly_and_never_remixed(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    initial = spectrum((-0.0 - 0.0j, 2j, 3j))
    captured = []
    accel.run_channel_accel(
        *args, seed_input=spectrum(100), seed_mapped=spectrum(200), sigma_initial=initial,
        config=replace(config, max_evaluations=1),
        on_start=lambda index, supplied, metadata: captured.append(supplied))
    assert len(calls) == 1
    for key in KEYS:
        assert getattr(captured[0], key).tobytes() == getattr(initial, key).tobytes()
        assert not np.shares_memory(getattr(captured[0], key), getattr(initial, key))


def test_time_admission_can_stop_before_any_fresh_map(monkeypatch):
    args, config, calls = fixture(monkeypatch)
    result = accel.run_channel_accel(
        *args, seed_input=spectrum(0), seed_mapped=spectrum(1),
        config=replace(config, max_seconds=2, initial_map_estimate_seconds=2))
    assert not calls and result.state is result.last_evaluated_state is None
    assert result.stop_reason == 'max_seconds' and not result.history


def test_selection_merit_failure_preserves_last_published_trial_and_reason(monkeypatch):
    args, config, calls = fixture(monkeypatch)

    def broken_merit(*args):
        raise ValueError('selection arithmetic failure')

    monkeypatch.setattr(accel, 'channel_merit', broken_merit)
    completed = []
    result = accel.run_channel_accel(
        *args, seed_input=spectrum(0), seed_mapped=spectrum(1), config=config,
        on_evaluation=lambda state, decision: completed.append(state.index))
    assert len(calls) == 2 and completed == [16]
    assert result.state.index == result.last_evaluated_state.index == 16
    assert len(result.history) == len(result.decisions) == 1
    assert result.stop_reason == 'evaluation_error'
    assert result.failure_reason == 'ValueError: selection arithmetic failure'


def test_original_six_screen_convergence_bypasses_auxiliary_merit_failure(monkeypatch):
    def mapped(sigma, number):
        if number == 1:
            return spectrum((sigma.retarded, sigma.statistical + 1, sigma.spectral))
        return spectrum((sigma.retarded + 1e-11, sigma.statistical, sigma.spectral))

    args, config, calls = fixture(monkeypatch, mapped)

    def broken_merit(*args):
        raise ValueError('auxiliary merit must not run on a fully converged state')

    monkeypatch.setattr(accel, 'channel_merit', broken_merit)
    result = accel.run_channel_accel(*args, seed_input=spectrum(0), seed_mapped=spectrum(1),
                                     config=config)
    decision = result.decisions[-1]
    assert len(calls) == 2 and result.state.index == result.last_evaluated_state.index == 17
    assert result.converged and result.stop_reason == 'fixed_point' and result.failure_reason is None
    assert decision['fully_converged'] and decision['accepted_as_anchor']
    assert not decision['unmixed_frobenius_nonincrease']['retarded']
    assert decision['reason'] == 'fully_converged_trial' and decision['trial_merit'] is None
    assert not decision['merit_evaluated']
