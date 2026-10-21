"""New bounded material iteration checks; no archived calculation is replayed."""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
import pytest
from test_anharmonic_transverse_material_map_reference import _causal_input, _fixture

import negfpy.anharmonic.transverse_material_scba as scba
from negfpy.anharmonic.transverse_material_map import (
    PackedBornMapEvaluation,
    PackedMapSpectrum,
    plan_material_born_map,
)
from negfpy.anharmonic.transverse_solver import TransverseCurrents
from negfpy.anharmonic.types import SolverScale

COMPONENTS = ("retarded", "lesser", "greater")


def _config(**overrides):
    return replace(scba.MaterialSCBAConfig(
        max_evaluations=2, workers=1, external_tile=1, bin_tile=2,
        max_seconds=60, time_reserve_seconds=1, initial_map_estimate_seconds=0.01,
    ), **overrides)


def _constant(value=0.0):
    return PackedMapSpectrum(*(np.full((1, 9, 2, 2), value, complex) for _ in COMPONENTS))


def _fake(monkeypatch, mapped=None, currents=None, fail_at=None, advance=None):
    """Deliberately algebra-free evaluator for control/state tests only."""
    calls = []
    mapped = _constant(1) if mapped is None else mapped
    currents = TransverseCurrents(1, -1, 0) if currents is None else currents

    def evaluate(grid, vertex, dynamical, left, right, **kwargs):
        calls.append(kwargs["sigma_input"])
        if advance is not None:
            advance()
        if len(calls) == fail_at:
            raise ValueError("injected numerical failure")
        plan = plan_material_born_map(1, 9, 5, 2, 2, 2, workers=1, external_tile=1, bin_tile=2)
        return PackedBornMapEvaluation(_constant(len(calls)), mapped, currents, {}, plan,
                                       kwargs["sigma_input"] is None, kwargs["g"])

    monkeypatch.setattr(scba, "evaluate_packed_born_map", evaluate)
    return calls


def test_real_map_zero_coupling_evaluates_initial_candidate_before_accepting():
    args = _fixture(1)
    initial = _causal_input(args[0], args[1])
    before = [getattr(initial, name).copy() for name in COMPONENTS]
    completed = []
    result = scba.run_material_scba(
        *args, sigma_initial=initial, g=0, scale=SolverScale(omega0=1.9, hbar=0.7),
        config=_config(mixing=1, current_atol_w=1e-10), on_evaluation=completed.append,
    )
    assert result.converged and result.stop_reason == "fixed_point"
    assert result.failure_reason is None and len(result.history) == 2
    assert not completed[0].record.sigma_passed
    assert abs(completed[0].evaluation.input_currents.collision_power) > 1e-8
    assert completed[1].record.sigma_passed and completed[1].record.current_passed
    assert completed[1].evaluation.input_currents.collision_power == 0
    for name, original in zip(COMPONENTS, before, strict=True):
        np.testing.assert_array_equal(getattr(initial, name), original)
        assert getattr(initial, name).flags.writeable
        assert not getattr(completed[0].sigma_input, name).flags.writeable
        assert np.count_nonzero(getattr(completed[1].sigma_input, name)) == 0
    assert result.state is completed[-1]


def test_real_complex_four_q_maps_keep_current_green_and_sigma_on_same_state():
    args = _fixture(4)
    states = []
    starts = []
    result = scba.run_material_scba(
        *args, g=0.013, config=_config(sigma_atol=1e-20, sigma_rtol=1e-12),
        on_evaluation=states.append, on_start=lambda index, sigma: starts.append((index, sigma)),
    )
    assert result.failure_reason is None and result.stop_reason == "max_evaluations"
    assert not result.converged and len(states) == 2
    assert starts[0] == (1, None)
    assert starts[1][1] is states[1].sigma_input
    for name in COMPONENTS:
        expected = 0.1 * getattr(states[0].evaluation.map_sigma, name)
        np.testing.assert_array_equal(getattr(states[1].sigma_input, name), expected)
        residual = states[1].record.sigma_residuals[name]
        difference = getattr(states[1].evaluation.map_sigma, name) - expected
        assert residual.error == pytest.approx(float(np.max(np.abs(difference))))
        assert residual.error_frobenius == pytest.approx(float(np.linalg.norm(difference)))
    assert states[0].evaluation.input_sigma_is_zero
    assert not states[1].evaluation.input_sigma_is_zero
    assert states[1].evaluation.g == 0.013
    assert result.state.evaluation.input_currents is states[1].evaluation.input_currents
    assert all(isinstance(asdict(record)["index"], int) for record in result.history)


@pytest.mark.parametrize("component", COMPONENTS)
def test_each_unmixed_component_must_pass(monkeypatch, component):
    mapped = _constant(0)
    getattr(mapped, component)[0, 0, 0, 0] = 2
    _fake(monkeypatch, mapped=mapped)
    result = scba.run_material_scba(*_fixture(1), sigma_initial=_constant(),
                                    config=_config(max_evaluations=1))
    assert not result.converged and not result.state.record.sigma_passed
    assert result.state.record.current_passed
    assert result.state.record.sigma_residuals[component].error == 2
    assert result.state.record.sigma_residuals[component].ratio > 1


def test_tiny_mixed_step_cannot_certify_fixed_point(monkeypatch):
    calls = _fake(monkeypatch)
    result = scba.run_material_scba(*_fixture(1), config=_config(mixing=1e-14))
    assert len(calls) == 2 and not result.converged
    assert np.max(np.abs(calls[1].lesser)) == pytest.approx(1e-14)
    assert result.state.record.sigma_residuals["lesser"].error > 0.99
    assert result.state.record.sigma_residuals["lesser"].ratio > 1000


def test_full_dyson_balance_alone_cannot_certify_scba(monkeypatch):
    _fake(monkeypatch, mapped=_constant(), currents=TransverseCurrents(1, -0.75, -0.25))
    result = scba.run_material_scba(*_fixture(1), config=_config(max_evaluations=1))
    record = result.state.record
    assert record.sigma_passed and not record.current_passed and not result.converged
    assert record.current_residuals["full_balance"].ratio == 0
    assert record.current_residuals["contact_balance"].ratio > 1
    assert record.current_residuals["collision_power"].ratio > 1


def test_absolute_watt_allowance_is_applied_to_evaluated_currents(monkeypatch):
    _fake(monkeypatch, mapped=_constant(), currents=TransverseCurrents(0, 0, 5e-15))
    result = scba.run_material_scba(*_fixture(1), config=_config(current_rtol=0, current_atol_w=1e-14))
    assert result.converged
    assert result.state.record.current_residuals["collision_power"].ratio == 0.5


@pytest.mark.parametrize("fail_at", [1, 2])
def test_numerical_failure_keeps_last_evaluated_state_and_checkpoint(monkeypatch, fail_at):
    calls = _fake(monkeypatch, fail_at=fail_at)
    checkpointed = []
    result = scba.run_material_scba(*_fixture(1), config=_config(),
                                    on_evaluation=checkpointed.append)
    assert len(calls) == fail_at
    assert result.stop_reason == "evaluation_error" and not result.converged
    assert "injected numerical failure" in result.failure_reason
    assert len(result.history) == len(checkpointed) == fail_at - 1
    assert result.state is (None if fail_at == 1 else checkpointed[0])
    if fail_at == 2:
        assert result.state.sigma_input is None
        assert result.state.evaluation.green.retarded[0, 0, 0, 0] == 1


def test_walltime_stops_before_next_map_using_duration_and_callback_cost(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(scba, "monotonic", lambda: now[0])
    calls = _fake(monkeypatch, advance=lambda: now.__setitem__(0, now[0]+8))
    result = scba.run_material_scba(
        *_fixture(1), config=_config(max_seconds=20, time_reserve_seconds=3),
        on_evaluation=lambda state: now.__setitem__(0, now[0]+2),
    )
    assert len(calls) == 1 and result.state.index == 1
    assert result.stop_reason == "max_seconds" and result.failure_reason is None
    assert result.history[0].map_seconds == 8


def test_initial_estimate_can_decline_unsafe_first_map(monkeypatch):
    calls = _fake(monkeypatch)
    result = scba.run_material_scba(
        *_fixture(1), config=_config(max_seconds=10, initial_map_estimate_seconds=8),
    )
    assert calls == [] and result.state is None
    assert result.stop_reason == "max_seconds" and result.failure_reason is None


@pytest.mark.parametrize("hook", ["on_start", "on_evaluation"])
def test_checkpoint_or_start_callback_failure_is_never_swallowed(monkeypatch, hook):
    _fake(monkeypatch)

    def fail(*args):
        raise ValueError("checkpoint failed")

    with pytest.raises(ValueError, match="checkpoint failed"):
        scba.run_material_scba(*_fixture(1), config=_config(), **{hook: fail})


@pytest.mark.parametrize("resource", ["max_bytes", "max_contraction_work", "max_dyson_evaluations"])
def test_resources_reject_before_evaluator(monkeypatch, resource):
    calls = _fake(monkeypatch)
    with pytest.raises(ValueError, match=resource):
        scba.run_material_scba(*_fixture(1), config=_config(**{resource: 1}))
    assert calls == []


def test_plan_counts_previous_evaluated_state_and_next_sigma_separately():
    config = _config(max_evaluations=12)
    plan = scba.plan_material_scba(4, 601, 216, 120, 24, 24, config=config)
    sigma = 3 * 16 * 4 * 601 * 120**2
    assert plan.input_sigma_bytes == sigma
    assert plan.previous_evaluated_state_bytes == 3 * sigma
    assert plan.peak_owned_bytes == plan.map_plan.peak_owned_bytes + 4*sigma + 64*120**2
    assert plan.nominal_caller_bytes == plan.map_plan.nominal_caller_bytes + sigma
    assert plan.maximum_dyson_evaluations == 12*2404
    assert plan.maximum_contraction_work == 12*plan.map_plan.contraction_work


def test_mixing_preserves_input_arrays_and_empty_active_support():
    initial, mapped = _constant(2), _constant(7)
    result = scba.mix_packed_spectra(initial, mapped, 0.25)
    for name in COMPONENTS:
        np.testing.assert_array_equal(getattr(result, name), 3.25)
        np.testing.assert_array_equal(getattr(initial, name), 2)
        np.testing.assert_array_equal(getattr(mapped, name), 7)
        assert not getattr(result, name).flags.writeable
    empty = PackedMapSpectrum(*(np.empty((4, 9, 0, 0), complex) for _ in COMPONENTS))
    empty_result = scba.mix_packed_spectra(None, empty, 0.1)
    assert empty_result.retarded.shape == (4, 9, 0, 0)


@pytest.mark.parametrize("bad", [0, -0.2, 1.1, True, np.nan])
def test_invalid_mixing_rejected(bad):
    with pytest.raises((ValueError, TypeError)):
        _config(mixing=bad)


@pytest.mark.parametrize("overrides", [
    {"max_evaluations": 0}, {"max_evaluations": True},
    {"sigma_rtol": 0, "sigma_atol": 0}, {"current_rtol": 0, "current_atol_w": 0},
    {"max_seconds": 1}, {"next_map_safety_factor": 0.9},
    {"initial_map_estimate_seconds": 0}, {"identity_rtol": np.inf},
])
def test_invalid_config_rejected(overrides):
    with pytest.raises((ValueError, TypeError)):
        _config(**overrides)
