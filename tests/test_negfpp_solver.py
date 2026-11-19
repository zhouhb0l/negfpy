"""Controller accounting and handoff, with no numerical kernel execution."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import negfpp.solver as solver


@pytest.fixture
def inputs():
    return (
        SimpleNamespace(n_points=5),
        SimpleNamespace(mesh=SimpleNamespace(n_qpoints=1), full_ndof=3, ndof=2),
        object(),
        SimpleNamespace(device_indices=(0,)),
        SimpleNamespace(device_indices=(2,)),
    )


@pytest.fixture
def config():
    return solver.ChannelSCBAConfig(
        max_evaluations=5, workers=1, max_seconds=60,
        time_reserve_seconds=1, initial_map_estimate_seconds=0.01,
    )


def _state(index):
    record = solver.ChannelSCBARecord(index, 0.01, 0.01, {}, {}, {}, False, False, False)
    return SimpleNamespace(
        index=index, sigma_input=object(),
        evaluation=SimpleNamespace(map_sigma=object()), record=record,
    )


@pytest.fixture
def engines(monkeypatch):
    calls = []
    states = {}

    def fixed(*args, **kwargs):
        calls.append(("linear", kwargs))
        history = []
        for index in range(1, kwargs["config"].max_evaluations + 1):
            state = _state(index)
            states[index] = state
            kwargs["on_start"](index, None if index == 1 else state.sigma_input)
            kwargs["on_evaluation"](state)
            history.append(state.record)
        return solver.ChannelSCBAResult(
            state, tuple(history), False, "max_evaluations", None, object(),
        )

    def accelerated(*args, **kwargs):
        calls.append(("anderson1", kwargs))
        history = []
        for offset in range(kwargs["config"].max_evaluations):
            state = _state(kwargs["seed_index"] + offset + 1)
            kwargs["on_start"](state.index, state.sigma_input, {"kind": "anderson1"})
            kwargs["on_evaluation"](state, {"accepted_as_anchor": offset != 0})
            history.append(state.record)
        return SimpleNamespace(
            state=state, history=tuple(history), converged=False,
            failure_reason=None, stop_reason="max_evaluations",
        )

    monkeypatch.setattr(solver, "run_channel_scba", fixed)
    monkeypatch.setattr(solver, "run_channel_accel", accelerated)
    return calls, states


def test_two_map_warmup_handoff_and_one_total_budget(inputs, config, engines):
    calls, states = engines
    starts = []
    evaluations = []
    outcome = solver.solve_from_zero(
        *inputs, scale=object(), g=0.1, config=config,
        on_start=lambda index, sigma, proposal: starts.append((index, proposal)),
        on_evaluation=lambda state, decision: evaluations.append((state.index, decision)),
    )
    assert [name for name, _ in calls] == ["linear", "anderson1"]
    warmup, continuation = (call[1] for call in calls)
    assert warmup["sigma_initial"] is None and warmup["first_index"] == 1
    assert warmup["config"].max_evaluations == 2
    assert continuation["config"].max_evaluations == 3
    assert continuation["seed_input"] is states[2].sigma_input
    assert continuation["seed_mapped"] is states[2].evaluation.map_sigma
    assert continuation["seed_index"] == 2
    assert continuation["acceleration"] == solver.ChannelAccelerationConfig()
    assert continuation["selection"] == solver.ChannelSelectionPolicy()
    assert continuation["feasibility"].mode == "psd-backtracking"
    assert (outcome.attempted, outcome.completed, outcome.selected, outcome.rejected) == (5, 5, 4, 1)
    assert [record.index for record in outcome.records] == [1, 2, 3, 4, 5]
    assert [index for index, _ in starts] == [1, 2, 3, 4, 5]
    assert starts[0][1] is None and evaluations[0][1] is None
    assert all(call[1]["config"].max_seconds <= config.max_seconds for call in calls)


@pytest.mark.parametrize(("method", "limit"), [("anderson1", 1), ("anderson1", 2), ("linear", 3)])
def test_short_or_linear_budget_does_not_enter_acceleration(inputs, config, engines, method, limit):
    calls, _ = engines
    outcome = solver.solve_from_zero(
        *inputs, scale=object(), g=0.1, config=replace(config, max_evaluations=limit), method=method,
    )
    assert [name for name, _ in calls] == ["linear"]
    assert outcome.attempted == outcome.completed == limit


def test_expired_input_construction_deadline_prevents_numerical_calls(inputs, config, engines, monkeypatch):
    calls, _ = engines
    monkeypatch.setattr(solver, "monotonic", lambda: 100.0)
    monkeypatch.setattr(solver, "plan_channel_scba", lambda *args, **kwargs: object())
    outcome = solver.solve_from_zero(
        *inputs, scale=object(), g=0.1, config=config, deadline_monotonic=99.0,
    )
    assert not calls
    assert outcome.attempted == outcome.completed == 0
    assert outcome.result.state is None and outcome.result.stop_reason == "max_seconds"


def test_shared_deadline_stops_before_continuation_and_keeps_last_warmup(inputs, config, engines, monkeypatch):
    calls, states = engines
    times = iter([0.0, 0.0, 100.0])
    monkeypatch.setattr(solver, "monotonic", lambda: next(times))
    outcome = solver.solve_from_zero(*inputs, scale=object(), g=0.1, config=config)
    assert [name for name, _ in calls] == ["linear"]
    assert outcome.completed == 2
    assert outcome.result.state is states[2]
    assert not outcome.result.converged and outcome.result.stop_reason == "max_seconds"


def test_acceleration_admission_stop_preserves_only_honest_warmup_scalars(inputs, config, engines, monkeypatch):
    def stopped(*args, **kwargs):
        return SimpleNamespace(state=None, converged=False, stop_reason="max_seconds", failure_reason=None)
    monkeypatch.setattr(solver, "run_channel_accel", stopped)
    outcome = solver.solve_from_zero(*inputs, scale=object(), g=0.1, config=config)
    assert outcome.result.state is None and not outcome.result.converged
    assert outcome.completed == 2 and len(outcome.records) == 2
    assert all(isinstance(record, solver.ChannelSCBARecord) for record in outcome.records)


def test_failed_map_consumes_attempt_without_inventing_completed_state(inputs, config, engines, monkeypatch):
    def failed(*args, **kwargs):
        kwargs["on_start"](1, None)
        return SimpleNamespace(
            state=None, history=(), converged=False,
            failure_reason="ValueError: physical guard failed", stop_reason="evaluation_error",
        )
    monkeypatch.setattr(solver, "run_channel_scba", failed)
    outcome = solver.solve_from_zero(*inputs, scale=object(), g=0.1, config=config)
    assert (outcome.attempted, outcome.completed, outcome.selected, outcome.rejected) == (1, 0, 0, 0)
    assert outcome.result.failure_reason and outcome.result.stop_reason == "evaluation_error"


def test_callback_failure_propagates_without_retry(inputs, config, engines):
    calls, _ = engines
    def fail(*args):
        raise RuntimeError("output callback failed")
    with pytest.raises(RuntimeError, match="output callback failed"):
        solver.solve_from_zero(*inputs, scale=object(), g=0.1, config=config, on_evaluation=fail)
    assert [name for name, _ in calls] == ["linear"]
