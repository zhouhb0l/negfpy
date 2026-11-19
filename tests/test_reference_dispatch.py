"""Public method dispatch and bounded reference diagnostics, without a solve."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import negfpp.reference as reference
import negfpp.solver as solver


def _config(**changes):
    return replace(solver.ChannelSCBAConfig(
        max_evaluations=5, max_seconds=60, time_reserve_seconds=1,
        initial_map_estimate_seconds=0.01, workers=1,
    ), **changes)


def _inputs():
    return (
        SimpleNamespace(n_points=5),
        SimpleNamespace(mesh=SimpleNamespace(n_qpoints=1), full_ndof=3, ndof=2),
        object(), SimpleNamespace(device_indices=(0,)), SimpleNamespace(device_indices=(2,)),
    )


def test_reference_dispatch_uses_same_run_controller_and_keeps_tangent_accounting(monkeypatch):
    calls = []
    states = [SimpleNamespace(index=index, record=object()) for index in (1, 2)]
    tangent_events = ({"phase": "start", "tangent_number": 1},)
    map_records = ({"index": 1, "arithmetic": "direct_baseline"},)
    decisions = ({"accepted_as_anchor": True}, {"accepted_as_anchor": False})
    def existing(*args, **kwargs):
        calls.append(kwargs)
        for state, decision in zip(states, decisions):
            kwargs["on_start"](state.index, None if state.index == 1 else object(), {"kind": "fresh_zero"})
            kwargs["on_evaluation"](state, decision)
        return reference.ReferenceFeedbackResult(
            state=states[0], last_evaluated_state=states[1], history=tuple(s.record for s in states),
            decisions=decisions, converged=False, stop_reason="proposal_error",
            failure_reason="ValueError: signed tangent failed", plan=object(),
            tangent_count=1, tangent_records=tangent_events, map_records=map_records,
        )
    monkeypatch.setattr(reference, "run_reference_feedback", existing)
    settings = {"warmup_evaluations": 2, "tangent_interval": 3, "max_tangent_directions": 6}
    outcome = solver.solve_from_zero(
        *_inputs(), scale=object(), g=0.1, config=_config(),
        method="reference-seeded", reference_config=settings,
    )
    assert len(calls) == 1
    call = calls[0]
    assert "sigma_initial" not in call and call["methodconfig"] == settings
    assert call["acceleration"] == solver.ChannelAccelerationConfig()
    assert call["selection"] == solver.ChannelSelectionPolicy()
    assert call["feasibility"].mode == "psd-backtracking"
    assert (outcome.attempted, outcome.completed, outcome.selected, outcome.rejected) == (2, 2, 1, 1)
    assert outcome.diagnostics["reference_kind"] == "same_run_selected_anchor"
    assert outcome.diagnostics["tangent_count"] == 1
    assert outcome.diagnostics["tangent_records"] == tangent_events
    assert outcome.diagnostics["map_records"] == map_records
    assert outcome.result.state is states[0] and outcome.result.last_evaluated_state is states[1]


def test_reference_full_map_budget_is_rejected_before_dispatch(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("The controller must not run")
    monkeypatch.setattr(reference, "run_reference_feedback", unexpected)
    with pytest.raises(ValueError, match="at most 18"):
        solver.solve_from_zero(*_inputs(), scale=object(), g=0.1, config=_config(max_evaluations=19), method="reference-seeded")


def test_reference_expired_shared_deadline_skips_controller(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("The controller must not run")
    monkeypatch.setattr(reference, "run_reference_feedback", unexpected)
    monkeypatch.setattr(solver, "monotonic", lambda: 100.0)
    monkeypatch.setattr(solver, "plan_channel_scba", lambda *args, **kwargs: object())
    outcome = solver.solve_from_zero(
        *_inputs(), scale=object(), g=0.1, config=_config(),
        method="reference-seeded", deadline_monotonic=99.0,
    )
    assert outcome.attempted == outcome.completed == 0
    assert outcome.result.stop_reason == "max_seconds" and not outcome.result.converged
    assert outcome.diagnostics["tangent_count"] == 0


def test_reference_tangent_cap_remains_the_existing_six_attempt_limit():
    with pytest.raises(ValueError, match="<=6 signed directions"):
        solver.solve_from_zero(
            *_inputs(), scale=object(), g=0.1, config=_config(),
            method="reference-seeded", reference_config={"max_tangent_directions": 7},
        )
