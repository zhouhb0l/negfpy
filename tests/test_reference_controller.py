"""Tiny synthetic controller checks; no material or historical array loads."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from negfpp.reference import controller as c
from negfpy.anharmonic.complete_energy import SignedFrequencyGrid, UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSCBAConfig, ChannelSpectrum


def spectrum(value):
    shape = (1, 3, 1, 1)
    r = np.full(shape, value, dtype=np.complex128)
    statistical = np.full(shape, -1j * value, dtype=np.complex128)
    d = np.zeros(shape, dtype=np.complex128)
    return ChannelSpectrum(r, statistical, d)


def linear_combination(first, second, a, b):
    return ChannelSpectrum(*(a * getattr(first, key) + b * getattr(second, key)
                             for key in c._COMPONENTS))


@pytest.fixture
def model(monkeypatch):
    grid = SignedFrequencyGrid(.1, 1)
    mesh = UniformBZMesh((1, 1, 1))
    vertex = SimpleNamespace(mesh=mesh, ndof=1, full_ndof=1)
    contact = SimpleNamespace(device_indices=(0,))
    plan = SimpleNamespace(peak_owned_bytes=100,
                           fixed_plan=SimpleNamespace(map_plan=SimpleNamespace(
                               contraction_work=1, dyson_evaluations=3)))
    monkeypatch.setattr(c, 'plan_channel_accel', lambda *args, **kwargs: plan)
    calls, tangents = [], []

    def evaluate(*args, sigma_input=None, **kwargs):
        calls.append((sigma_input, kwargs))
        supplied = spectrum(0.) if sigma_input is None else sigma_input
        mapped = linear_combination(supplied, spectrum(1.), .5, 1.)
        currents = SimpleNamespace(left=1e-9, right=-1e-9, collision_power=0.,
                                   contact_balance=0., full_balance=0.)
        value = SimpleNamespace(green=supplied, map_sigma=mapped, input_currents=currents,
                                checks={}, input_sigma_is_zero=sigma_input is None)
        return value, {'reference_path': sigma_input is not None}

    def tangent(*args, direction, **kwargs):
        tangents.append((direction, kwargs))
        return linear_combination(direction, spectrum(0.), .5, 0.), {'analytic': True}

    monkeypatch.setattr(c, 'evaluate_reference_map', evaluate)
    monkeypatch.setattr(c, 'tangent_map', tangent)
    config = ChannelSCBAConfig(max_evaluations=3, max_seconds=100.,
                               initial_map_estimate_seconds=.01,
                               time_reserve_seconds=0., next_map_safety_factor=1.,
                               max_bytes=1000, max_contraction_work=100,
                               max_dyson_evaluations=100, mixing=.05)
    return SimpleNamespace(grid=grid, vertex=vertex, contact=contact, calls=calls,
                           tangents=tangents, config=config, evaluate=evaluate)


def solve(model, **kwargs):
    return c.run_reference_feedback(model.grid, model.vertex, np.zeros((1, 1, 1)),
                                    model.contact, model.contact,
                                    config=kwargs.pop('config', model.config),
                                    feasibility=kwargs.pop('feasibility', c.ChannelProposalFeasibility()),
                                    **kwargs)


def test_real_analytic_proposal_and_original_map_record(model):
    started, evaluated = [], []
    answer = solve(model, on_start=lambda i, x, p: started.append((i, x, p)),
                   on_evaluation=lambda state, decision: evaluated.append((state, decision)))
    assert len(model.calls) == 3 and len(model.tangents) == 1
    assert answer.tangent_count == 1 and len(answer.history) == 3
    assert [row[2]['kind'] for row in started] == ['fresh_zero', 'linear_warmup', 'analytic-feedback1']
    assert answer.decisions[-1]['accepted_as_anchor']
    assert answer.state.index == answer.last_evaluated_state.index == 3
    proposal = started[-1][2]
    assert proposal['raw_coefficient'] == pytest.approx(2.)
    assert proposal['clip_factor'] < 1
    assert proposal['scaled_step_norm'] <= proposal['scaled_step_bound']
    assert evaluated[-1][1]['map_metadata']['reference_path']
    assert model.calls[2][1]['reference_green'] is model.calls[1][0]
    assert [row['phase'] for row in answer.tangent_records] == ['start', 'complete']


def test_rejection_retains_anchor_and_next_linear_fallback(model, monkeypatch):
    original = model.evaluate

    def evaluate(*args, sigma_input=None, **kwargs):
        value, metadata = original(*args, sigma_input=sigma_input, **kwargs)
        if len(model.calls) == 3:
            value.map_sigma = linear_combination(sigma_input, spectrum(100.), 1., 1.)
        return value, metadata

    monkeypatch.setattr(c, 'evaluate_reference_map', evaluate)
    started = []
    answer = solve(model, config=replace(model.config, max_evaluations=4),
                   on_start=lambda i, x, p: started.append((i, x, p)))
    assert answer.decisions[2]['accepted_as_anchor'] is False
    assert answer.decisions[2]['selected_anchor_index'] == 2
    assert started[3][2]['kind'] == 'linear_fallback'
    expected = c.mix_channel_spectra(model.calls[1][0], spectrum(1.025), .05)
    np.testing.assert_allclose(model.calls[3][0].retarded, expected.retarded)
    assert answer.tangent_count == 1
    assert len(answer.history) == 4
    assert answer.state.index == 4


def test_hard_map_failure_stops_without_retry(model, monkeypatch):
    original = model.evaluate

    def evaluate(*args, **kwargs):
        if len(model.calls) == 2:
            model.calls.append(('failed', {}))
            raise ValueError('original source guard failed')
        return original(*args, **kwargs)

    monkeypatch.setattr(c, 'evaluate_reference_map', evaluate)
    answer = solve(model)
    assert answer.stop_reason == 'evaluation_error'
    assert answer.state.index == answer.last_evaluated_state.index == 2
    assert len(answer.history) == 2 and len(model.calls) == 3
    assert 'source guard failed' in answer.failure_reason


def test_hard_tangent_failure_retains_actual_unmatched_start(model, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('signed tangent overflow')

    monkeypatch.setattr(c, 'tangent_map', fail)
    answer = solve(model)
    assert answer.stop_reason == 'proposal_error'
    assert answer.tangent_count == 1 and len(model.calls) == 2
    assert answer.state.index == answer.last_evaluated_state.index == 2
    assert [row['phase'] for row in answer.tangent_records] == ['start']


def test_tangent_requires_time_for_tangent_and_next_full_map(model, monkeypatch):
    now = [0.]
    monkeypatch.setattr(c, 'monotonic', lambda: now[0])
    original = model.evaluate

    def evaluate(*args, **kwargs):
        answer = original(*args, **kwargs)
        now[0] += .2
        return answer

    monkeypatch.setattr(c, 'evaluate_reference_map', evaluate)
    answer = solve(model, config=replace(model.config, max_seconds=1.),
                   methodconfig={'initial_tangent_estimate_seconds': 1.})
    assert answer.tangent_count == 0 and len(model.tangents) == 0
    assert answer.tangent_records[0]['phase'] == 'skipped'
    assert answer.tangent_records[0]['reason'] == 'tangent_and_next_map_time_budget'
    assert len(answer.history) == 3


def test_all_six_screens_override_auxiliary_selection_merit(model, monkeypatch):
    original = model.evaluate

    def evaluate(*args, sigma_input=None, **kwargs):
        value, metadata = original(*args, sigma_input=sigma_input, **kwargs)
        if len(model.calls) == 3:
            value.map_sigma = sigma_input
        return value, metadata

    monkeypatch.setattr(c, 'evaluate_reference_map', evaluate)
    monkeypatch.setattr(c, 'channel_merit', lambda *args: (_ for _ in ()).throw(
        AssertionError('A converged trial must bypass merit evaluation.')))
    answer = solve(model)
    assert answer.converged and answer.stop_reason == 'fixed_point'
    assert answer.decisions[-1]['reason'] == 'fully_converged_trial'
    assert not answer.decisions[-1]['merit_evaluated']


def test_analytic_kind_uses_original_psd_backtracking_without_spoofing(model):
    candidate = spectrum(-1.)
    candidate = ChannelSpectrum(spectrum(1.).retarded, candidate.statistical, candidate.spectral)
    value, batch = c.feasible_reference_proposal(
        model.vertex.mesh, model.grid, spectrum(1.), spectrum(1.), candidate,
        original_kind='analytic-feedback1', mixing=.05,
        policy=c.ChannelProposalFeasibility(mode='psd-backtracking'), rtol=1e-8, atol=1e-11)
    assert batch['original_kind'] == batch['selected_kind'] == 'analytic-feedback1'
    assert batch['chosen_factor'] == .5 and batch['screen_count'] == 2
    assert batch['screens'][0]['failed_keys'] == ['lesser_psd', 'greater_psd']
    assert not batch['fallback_used']
    np.testing.assert_allclose(value.statistical, 0.)


def test_linear_candidate_does_not_get_analytic_halving(model):
    with pytest.raises(c.ChannelProposalFailure) as error:
        c.feasible_reference_proposal(
            model.vertex.mesh, model.grid, spectrum(1.), spectrum(1.), spectrum(-1.),
            original_kind='linear_fallback', mixing=.05,
            policy=c.ChannelProposalFeasibility(mode='psd-backtracking'), rtol=1e-8, atol=1e-11)
    assert error.value.report['screen_count'] == 1
    assert error.value.report['stop_reason'] == 'linear_input_failure'


def test_first_map_time_admission_does_not_start_any_work(model):
    answer = solve(model, config=replace(model.config, max_seconds=.005))
    assert answer.stop_reason == 'max_seconds' and answer.state is None
    assert len(model.calls) == answer.tangent_count == 0


def test_method_budget_rejects_unbounded_directions(model):
    with pytest.raises(ValueError, match='<=6'):
        solve(model, methodconfig={'max_tangent_directions': 7})
