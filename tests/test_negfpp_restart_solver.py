"""A loaded guess consumes fresh maps and cannot import solver history."""
from types import SimpleNamespace

import numpy as np
import pytest

import negfpp.solver as solver
from test_reference_controller import model, solve, spectrum


def test_external_fixed_point_still_gets_a_fresh_reference_map(model):
    seed = spectrum(2.)
    originals = [getattr(seed, key).tobytes() for key in ("retarded", "statistical", "spectral")]
    answer = solve(model, sigma_initial=seed)
    assert answer.converged and len(answer.history) == len(model.calls) == 1
    assert answer.state.index == 2 and answer.tangent_count == 0
    assert model.calls[0][1]["reference_sigma"] is None
    assert model.calls[0][1]["reference_green"] is None
    assert model.calls[0][1]["reference_mapped"] is None
    assert model.calls[0][0] is not seed
    for key, expected in zip(("retarded", "statistical", "spectral"), originals):
        actual = getattr(model.calls[0][0], key)
        assert not actual.flags.writeable and actual.tobytes() == expected


def test_seeded_reference_keeps_two_new_warmups_and_total_map_budget(model):
    starts = []
    answer = solve(model, sigma_initial=spectrum(.25),
                   on_start=lambda i, x, p: starts.append((i, p["kind"])))
    assert [i for i, _ in starts] == [2, 3, 4]
    assert [kind for _, kind in starts] == ["external_initial_guess", "linear_warmup", "analytic-feedback1"]
    assert len(answer.history) == 3 and answer.tangent_count == 1


def test_external_reference_field_dtype_is_not_silently_converted(model):
    seed = spectrum(2.)
    from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
    seed = ChannelSpectrum(seed.retarded.astype(np.complex64), seed.statistical, seed.spectral)
    with pytest.raises(ValueError, match="complex128"):
        solve(model, sigma_initial=seed)
    assert not model.calls


@pytest.mark.parametrize("method", ["linear", "anderson1"])
def test_seeded_solver_starts_at_exact_loaded_input_with_fresh_budget(monkeypatch, method):
    inputs = (SimpleNamespace(n_points=3),
              SimpleNamespace(mesh=SimpleNamespace(n_qpoints=1), full_ndof=1, ndof=1),
              object(), SimpleNamespace(device_indices=(0,)), SimpleNamespace(device_indices=(0,)))
    config = solver.ChannelSCBAConfig(max_evaluations=1, max_seconds=100,
                                      initial_map_estimate_seconds=.01, time_reserve_seconds=0)
    seed = spectrum(.25)
    calls = []
    def fixed(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["sigma_initial"] is seed and kwargs["first_index"] == 2
        assert kwargs["config"].max_evaluations == 1
        record = solver.ChannelSCBARecord(2, .01, .01, {}, {}, {}, False, False, False)
        state = SimpleNamespace(index=2, sigma_input=seed, record=record)
        kwargs["on_start"](2, seed)
        kwargs["on_evaluation"](state)
        return solver.ChannelSCBAResult(state, (record,), False, "max_evaluations", None, object())
    monkeypatch.setattr(solver, "run_channel_scba", fixed)
    outcome = solver.solve_from_self_energy(*inputs, sigma_initial=seed, scale=object(),
                                            g=1., config=config, method=method)
    assert len(calls) == 1 and outcome.attempted == outcome.completed == 1
    assert not outcome.result.converged


def test_zero_start_api_cannot_accidentally_accept_foreign_fields():
    with pytest.raises(TypeError, match="solve_from_self_energy"):
        solver.solve_from_zero(sigma_initial=spectrum(1.))
    with pytest.raises(ValueError, match="complete R/C/D"):
        solver.solve_from_self_energy(sigma_initial=None)
