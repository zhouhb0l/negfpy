"""New native-channel continuation references; no archived state is replayed."""

from __future__ import annotations

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

# These are literal finite-sum synthetic reference helpers, imported without
# collecting or replaying any previous test case.
from test_anharmonic_transverse_channel_feedback import (
    CHANNELS,
    _constant,
    _fixture,
    _reference,
    _sigma,
)

import negfpy.anharmonic.transverse_channel_feedback as feedback
from negfpy.anharmonic.transverse_solver import TransverseCurrents
from negfpy.anharmonic.types import SolverScale


def _config(**overrides):
    return replace(feedback.ChannelSCBAConfig(
        max_evaluations=6, workers=1, external_tile=1, bin_tile=2,
        max_seconds=60, time_reserve_seconds=1, initial_map_estimate_seconds=.01,
        sigma_atol=1e-30, sigma_rtol=1e-14,
    ), **overrides)


def _candidate_from_synthetic_state3(args, scale, g):
    """Create candidate4 by independent native mixing, without production mix."""
    input3 = _sigma(*args[:2])
    _, mapped3, _, _ = _reference(*args, input3, scale, g)
    candidate4 = feedback.ChannelSpectrum(*[
        .9 * getattr(input3, name) + .1 * mapped3[name] for name in CHANNELS
    ])
    return candidate4


def _native_component(spectrum, name):
    if name in CHANNELS:
        return getattr(spectrum, name)
    return spectrum.statistical + (-.5 if name == "lesser" else .5) * spectrum.spectral


def _fake(monkeypatch, *, mapped=None, fail_at=None):
    calls = []
    mapped = _constant(1) if mapped is None else mapped
    def evaluate(grid, vertex, dynamical, left, right, **kwargs):
        calls.append(kwargs["sigma_input"])
        if len(calls) == fail_at:
            raise ValueError("injected continuation candidate failure")
        plan = feedback.plan_channel_born_map(3, 5, 3, 2, 2, 2,
                                              workers=1, external_tile=1, bin_tile=2)
        return feedback.ChannelBornMapEvaluation(
            _constant(len(calls)), mapped, TransverseCurrents(1, -1, 0), {}, plan,
            False, kwargs["g"],
        )
    monkeypatch.setattr(feedback, "evaluate_channel_born_map", evaluate)
    return calls


def test_six_native_continuation_states_match_literal_nonzero_chain():
    args, states, starts = _fixture(), [], []
    scale, coupling = SolverScale(omega0=1.7, hbar=.6), .017
    candidate = _candidate_from_synthetic_state3(args, scale, coupling)
    original = {name: getattr(candidate, name).copy() for name in CHANNELS}
    result = feedback.run_channel_scba(
        *args, sigma_initial=candidate, first_index=4, g=coupling, scale=scale,
        config=_config(), on_evaluation=states.append,
        on_start=lambda index, sigma: starts.append((index, sigma)),
    )
    assert result.failure_reason is None and result.stop_reason == "max_evaluations"
    assert not result.converged and result.state is states[-1]
    assert [state.index for state in states] == list(range(4, 10))
    assert [record.index for record in result.history] == list(range(4, 10))
    assert result.plan.maximum_dyson_evaluations == 6 * 3 * 5
    assert result.plan.maximum_contraction_work == 6 * result.plan.map_plan.contraction_work
    for name in CHANNELS:
        np.testing.assert_array_equal(getattr(candidate, name), original[name])
        assert getattr(candidate, name).flags.writeable
        np.testing.assert_array_equal(getattr(states[0].sigma_input, name), original[name])
        assert not np.shares_memory(getattr(states[0].sigma_input, name), getattr(candidate, name))
        assert not getattr(states[0].sigma_input, name).flags.writeable
    for offset, state in enumerate(states):
        assert starts[offset] == (offset + 4, state.sigma_input)
        assert not state.evaluation.input_sigma_is_zero
        expected_green, expected_map, expected_currents, _ = _reference(
            *args, candidate, scale, coupling,
        )
        for name in CHANNELS:
            np.testing.assert_allclose(getattr(state.evaluation.green, name), expected_green[name],
                                       rtol=4e-12, atol=2e-14)
            np.testing.assert_allclose(getattr(state.evaluation.map_sigma, name), expected_map[name],
                                       rtol=5e-12, atol=2e-15)
        for name, expected in zip(("left", "right", "collision_power"),
                                  expected_currents, strict=True):
            assert getattr(state.evaluation.input_currents, name) == pytest.approx(
                expected, rel=2e-9, abs=2e-15,
            )
        candidate = feedback.ChannelSpectrum(*[
            .9 * getattr(candidate, name) + .1 * expected_map[name] for name in CHANNELS
        ])
    json.dumps([asdict(record) for record in result.history], allow_nan=False)


@pytest.mark.parametrize("fail_at", [1, 2])
def test_resume_failure_preserves_only_completed_new_state(monkeypatch, fail_at):
    calls = _fake(monkeypatch, fail_at=fail_at)
    states, starts = [], []
    initial = _constant(.5)
    result = feedback.run_channel_scba(
        *_fixture(), sigma_initial=initial, first_index=4, config=_config(),
        on_evaluation=states.append,
        on_start=lambda index, sigma: starts.append((index, sigma)),
    )
    assert len(calls) == fail_at and [entry[0] for entry in starts] == list(range(4, 4+fail_at))
    assert [state.index for state in states] == list(range(4, 3+fail_at))
    assert result.stop_reason == "evaluation_error" and not result.converged
    assert "injected continuation candidate failure" in result.failure_reason
    assert result.state is (None if fail_at == 1 else states[0])
    if fail_at == 2:
        assert result.state.index == 4
        for name in CHANNELS:
            np.testing.assert_array_equal(getattr(result.state.sigma_input, name), .5)
            # First candidate was used directly; the one allowed later mix is .9*.5+.1*1.
            np.testing.assert_allclose(getattr(calls[1], name), .55, rtol=2e-16)
    for name in CHANNELS:
        np.testing.assert_array_equal(getattr(initial, name), .5)
        assert getattr(initial, name).flags.writeable


def test_native_residual_locations_match_independent_full_array_argmax(monkeypatch):
    initial, mapped = _constant(0), _constant(0)
    for name, first_location, scale, error_location, error in (
        ("retarded", (0, 1, 0, 1), 10., (2, 4, 1, 0), 3.),
        ("statistical", (1, 3, 1, 0), 7., (2, 0, 0, 1), 2.),
        ("spectral", (0, 2, 0, 1), 4., (2, 4, 1, 1), 1.5),
    ):
        getattr(initial, name)[first_location] = scale
        getattr(mapped, name)[first_location] = scale - .1
        getattr(mapped, name)[error_location] = error
    _fake(monkeypatch, mapped=mapped)
    result = feedback.run_channel_scba(*_fixture(), sigma_initial=initial, first_index=4,
                                       config=_config(max_evaluations=1))
    record = result.state.record
    residuals = dict(record.sigma_residuals, **record.channel_residuals)
    assert set(residuals) == {"retarded", "lesser", "greater", "statistical", "spectral"}
    for name, residual in residuals.items():
        first, second = _native_component(mapped, name), _native_component(initial, name)
        absolute_difference = np.abs(first - second)
        scale_values = np.maximum(np.abs(first), np.abs(second))
        expected_error_location = tuple(int(v) for v in np.unravel_index(
            int(np.argmax(absolute_difference)), absolute_difference.shape,
        ))
        expected_scale_location = tuple(int(v) for v in np.unravel_index(
            int(np.argmax(scale_values)), scale_values.shape,
        ))
        assert tuple(residual.error_location) == expected_error_location
        assert tuple(residual.scale_location) == expected_scale_location
        assert residual.error == pytest.approx(float(absolute_difference.max()))
        assert residual.scale == pytest.approx(float(scale_values.max()))
        assert residual.error_frobenius == pytest.approx(float(np.linalg.norm(first-second)))
        allowance = _config().sigma_atol + _config().sigma_rtol * scale_values.max()
        assert residual.ratio == pytest.approx(absolute_difference.max() / allowance)
    json.dumps(asdict(record), allow_nan=False)


@pytest.mark.parametrize("sigma_initial,first_index", [(None, 4), ("native", 1)])
def test_native_candidate_and_continuation_index_must_be_paired(monkeypatch, sigma_initial, first_index):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid continuation must be rejected before evaluation")
    monkeypatch.setattr(feedback, "evaluate_channel_born_map", forbidden)
    supplied = _constant(0) if sigma_initial == "native" else None
    with pytest.raises((ValueError, TypeError), match="(initial|index|continu)"):
        feedback.run_channel_scba(*_fixture(), sigma_initial=supplied, first_index=first_index,
                                   config=_config())


@pytest.mark.parametrize("huge", [False, True])
def test_gamma_zero_context_matches_independent_full_matrix_products(huge):
    args = _fixture()
    grid, vertex, dynamical, left, right = args
    supplied = _sigma(grid, vertex, huge=huge)
    result = feedback.evaluate_channel_born_map(
        *args, sigma_input=supplied, g=0, workers=1, external_tile=1, bin_tile=2,
    )
    context = result.zero_frequency_context
    assert context["diagnostic_only"] is True and context["available"] is True
    assert (context["q_index"], context["frequency_index"], context["omega"]) == (0, 2, 0)
    assert context["q_address"] == [0, 0, 0]
    _, _, _, total = _reference(*args, supplied, SolverScale(), 0)
    q, w = 0, grid.zero_index
    operator = -dynamical[q] - total["retarded"][q, w]
    r = np.linalg.inv(operator)
    advanced = r.conj().T
    cs, ds = total["statistical"][q, w], total["spectral"][q, w]
    c, d = r @ cs @ advanced, r @ ds @ advanced
    source_delta = ds - (operator.conj().T - operator)
    propagated = r @ source_delta @ advanced
    primary_delta = d - (r - advanced)
    greater_product, lesser_product = (r @ (cs + sign * ds / 2) @ advanced
                                      for sign in (1, -1))
    reference_pairs = {
        "raw_source_identity": (ds, operator.conj().T - operator),
        "primary_spectral_identity": (d, r-advanced),
        "propagated_source_identity": (propagated, np.zeros_like(propagated)),
        "identity_product_remainder": (primary_delta, propagated),
        "source_spectral_roundtrip": ((cs+ds/2)-(cs-ds/2), ds),
        "exported_spectral_roundtrip": ((c+d/2)-(c-d/2), d),
        "spectral_product_vs_contour_products": (d, greater_product-lesser_product),
    }
    assert set(context["metrics"]) == set(reference_pairs)
    for name, (first, second) in reference_pairs.items():
        difference = np.abs(first - second)
        error = float(difference.max())
        scale = max(float(np.abs(first).max()), float(np.abs(second).max()))
        diagnostic = context["metrics"][name]
        assert diagnostic["error"] == pytest.approx(error, rel=3e-12, abs=2e-18)
        assert diagnostic["scale"] == pytest.approx(scale, rel=3e-12, abs=2e-18)
        assert diagnostic["allowance"] == pytest.approx(1e-11 + 1e-8 * scale)
        assert diagnostic["ratio"] == pytest.approx(error / (1e-11 + 1e-8 * scale),
                                                     rel=3e-12, abs=2e-7)
        expected_entry = [int(v) for v in np.unravel_index(int(np.argmax(difference)), difference.shape)]
        assert diagnostic["error_location"] == [q, w, *expected_entry]
    source_components = {
        "harmonic": (dynamical[q], dynamical[q].conj().T),
        "interaction": (supplied.spectral[q, w],
                        supplied.retarded[q, w]-supplied.retarded[q, w].conj().T),
    }
    for name, contact in (("left", left), ("right", right)):
        source_components[name] = (
            contact.spectrum.greater[q, w]-contact.spectrum.lesser[q, w],
            contact.spectrum.retarded[q, w]-contact.spectrum.retarded[q, w].conj().T,
        )
    for name, (first, second) in source_components.items():
        assert context["source_component_metrics"][name]["error"] == pytest.approx(
            float(np.abs(first-second).max()), rel=2e-13, abs=2e-18,
        )
    denominator = max(np.linalg.norm(operator, ord=np.inf) * np.linalg.norm(r, ord=np.inf), 1.)
    for name, difference in (("left", operator@r-np.eye(3)),
                              ("right", r@operator-np.eye(3))):
        assert context["inverse_backward_residuals"][name] == pytest.approx(
            float(np.linalg.norm(difference, ord=np.inf) / denominator), rel=3e-12, abs=2e-18,
        )
    for name, statistical, spectral in (("source", cs, ds), ("green", c, d)):
        expected = float(np.abs(statistical).max()) / float(np.abs(spectral).max())
        assert context["statistical_to_spectral_scale_ratios"][name] == pytest.approx(expected, rel=3e-12)
        if huge:
            assert expected > 1e20
    assert abs(d[2, 0].real) > 1e-7
    assert result.checks["full_green_local"]["keldysh_identity"]["ratio"] < 1
    assert result.checks["active_green"]["keldysh_identity"]["ratio"] < 1
    json.dumps(context, allow_nan=False)
