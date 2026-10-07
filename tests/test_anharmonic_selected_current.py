"""Selected spectral history against the existing full-spectrum current path.

Random complex lines check indexing and signs without presuming a physical
state. The separate dressed reference checks the specified centered toy model.
"""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import KeldyshSpectrum, SignedFrequencyGrid, SolverScale
from negfpy.anharmonic.contracted_cubic_current import build_contracted_cubic_cut
from negfpy.anharmonic.selected_current import collect_selected_cut_spectra
import negfpy.anharmonic.selected_current as selected_module
from test_anharmonic_cubic_cut_current import _problem, _tensor


POLICY = "centered-fixed-field-diagnostic"


def _case():
    # Response-only vertices have columns outside either requested cut's rows.
    values = _tensor(8, [((0, 0, 2), .7), ((2, 2, 4), -.4), ((6, 6, 7), .9)])
    problem = replace(
        _problem(values, npl=2, sparse_vertex=True, scale=SolverScale(hbar=2, omega0=3)),
        grid=SignedFrequencyGrid(step=.17, n_positive=5),
    )
    plans = tuple(build_contracted_cubic_cut(problem, cut) for cut in (1, 2))
    rng = np.random.default_rng(718903)
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    green = KeldyshSpectrum(retarded=np.zeros(shape, complex), lesser=lesser, greater=greater)
    return problem, plans, green


def _evaluator(problem, green, seen=None):
    def evaluate(omega):
        if seen is not None:
            seen.append(omega)
        position = np.flatnonzero(problem.grid.values == omega)
        assert len(position) == 1
        index = position.item()
        return SimpleNamespace(lesser=green.lesser[index], greater=green.greater[index])
    return evaluate


def _collect(problem, plans, green, **kwargs):
    options = dict(max_evaluations=problem.grid.n_points, max_seconds=60)
    options.update(kwargs)
    return collect_selected_cut_spectra(plans, problem.grid, _evaluator(problem, green), **options)


def test_selected_two_cuts_match_full_green_and_preserve_inputs():
    problem, plans, green = _case()
    lesser_before, greater_before = green.lesser.copy(), green.greater.copy()
    seen = []
    spectra = collect_selected_cut_spectra(
        plans, problem.grid, _evaluator(problem, green, seen),
        max_evaluations=problem.grid.n_points, max_seconds=60,
    )
    np.testing.assert_array_equal(seen, problem.grid.values)
    assert seen.count(0.0) == 1
    np.testing.assert_array_equal(spectra.rows, [0, 2, 4])
    np.testing.assert_array_equal(spectra.active_indices, [0, 2, 4, 6, 7])
    assert spectra.greater.shape == (problem.grid.n_points, 3, 5)
    np.testing.assert_array_equal(spectra.greater, green.greater[:, spectra.rows[:, None], spectra.active_indices])
    np.testing.assert_array_equal(green.lesser, lesser_before)
    np.testing.assert_array_equal(green.greater, greater_before)
    for value in (spectra.rows, spectra.active_indices, spectra.greater, spectra.harmonic_spectra):
        assert not value.flags.writeable
    times = np.linspace(0, 2.8, 39)
    for plan in plans:
        expected_cubic = plan.integrate_from_green(problem.grid, green, times, block_size=4)
        expected_harmonic = float(plan.operator.harmonic_coefficients @ (
            plan.operator.harmonic_moments_from_green(problem.grid, green)
        ))
        result = spectra.integrate(plan.operator.cut, times, mean_policy=POLICY, block_size=7)
        np.testing.assert_allclose(result["integrand"], expected_cubic["integrand"], rtol=3e-13, atol=2e-17)
        np.testing.assert_allclose(result["connected_cubic_solver"], expected_cubic["connected_cubic_solver"],
                                   rtol=3e-13, atol=2e-17)
        np.testing.assert_allclose(result["harmonic_solver"], expected_harmonic, rtol=3e-13, atol=2e-17)
        expected_total = expected_harmonic + expected_cubic["connected_cubic_solver"]
        np.testing.assert_allclose(result["centered_total_solver"], expected_total, atol=2e-17)
        np.testing.assert_allclose(result["centered_total"], 18 * expected_total, atol=2e-16)
        np.testing.assert_allclose(result["harmonic"], 18 * expected_harmonic, atol=2e-16)
        assert result["mean_policy"] == POLICY
        assert result["harmonic_current_included"]
        for label in ("physical_means_solved", "static_feedback_included", "counterterm_added",
                      "full_material_current_validated", "input_centering_verified",
                      "spatial_validation_passed", "changes_solver_acceptance", "disconnected_means_included"):
            assert result[label] is False
    assert not spectra.metadata["frequency_accuracy_verified"]
    assert not spectra.metadata["full_green_frequency_history_retained"]
    assert spectra.metadata["zero_frequency_evaluated_explicitly"]
    assert spectra.metadata["evaluations"] == problem.grid.n_points
    assert spectra.metadata["output_array_bytes"] == spectra.greater.nbytes + spectra.harmonic_spectra.nbytes


def test_zero_node_is_retained_and_affects_cubic_current():
    problem, plans, green = _case()
    spectra = _collect(problem, plans, green)
    zero = problem.grid.n_positive
    np.testing.assert_array_equal(spectra.greater[zero], green.greater[zero][spectra.rows[:, None], spectra.active_indices])
    greater_without_zero = green.greater.copy()
    greater_without_zero[zero] = 0
    wrong = KeldyshSpectrum(retarded=green.retarded, lesser=green.lesser, greater=greater_without_zero)
    times = np.linspace(0, 2.8, 39)
    correct = spectra.integrate(1, times, mean_policy=POLICY)
    omitted = plans[0].integrate_from_green(problem.grid, wrong, times)
    assert abs(correct["connected_cubic_solver"] - omitted["connected_cubic_solver"]) > 1e-7


def test_memory_budget_counts_shared_row_union_once_and_fails_before_evaluation():
    problem, plans, green = _case()
    required = problem.grid.n_points * (16 * 3 * 5 + 8 * 2)
    seen = []
    with pytest.raises(ValueError, match="max_spectrum_bytes"):
        collect_selected_cut_spectra(plans, problem.grid, _evaluator(problem, green, seen),
                                     max_evaluations=problem.grid.n_points, max_seconds=60,
                                     max_spectrum_bytes=required - 1)
    assert seen == []
    spectra = _collect(problem, plans, green, max_spectrum_bytes=required)
    assert spectra.metadata["output_array_bytes"] == required


@pytest.mark.parametrize("field, value", [
    ("max_evaluations", 0), ("max_evaluations", True), ("max_evaluations", 1.5),
    ("max_seconds", 0), ("max_seconds", -1), ("max_seconds", np.nan),
    ("max_seconds", np.inf), ("max_seconds", True),
    ("max_spectrum_bytes", 0), ("max_spectrum_bytes", True),
])
def test_invalid_limits_rejected_before_evaluator(field, value):
    problem, plans, _ = _case()
    options = dict(max_evaluations=problem.grid.n_points, max_seconds=60, max_spectrum_bytes=100000)
    options[field] = value
    with pytest.raises((ValueError, TypeError)):
        collect_selected_cut_spectra(plans, problem.grid, lambda _: pytest.fail("invalid limit evaluated"), **options)


def test_incomplete_grid_budget_is_rejected_before_evaluator():
    problem, plans, _ = _case()
    with pytest.raises(ValueError, match="complete grid"):
        collect_selected_cut_spectra(plans, problem.grid, lambda _: pytest.fail("partial grid evaluated"),
                                     max_evaluations=problem.grid.n_points - 1, max_seconds=60)


@pytest.mark.parametrize("plans_kind", ["empty", "wrong_type", "duplicate", "different_ndof", "different_scale"])
def test_invalid_plan_collections_rejected(plans_kind):
    problem, plans, _ = _case()
    if plans_kind == "empty":
        plans = ()
    elif plans_kind == "wrong_type":
        plans = (object(),)
    elif plans_kind == "duplicate":
        plans = (plans[0], plans[0])
    elif plans_kind == "different_ndof":
        other = build_contracted_cubic_cut(_problem(_tensor(3, [((0, 1, 2), .2)])), 2)
        plans = (plans[0], other)
    else:
        other_problem = replace(problem, scale=SolverScale(hbar=1, omega0=1))
        plans = (plans[0], build_contracted_cubic_cut(other_problem, 2))
    with pytest.raises(ValueError):
        collect_selected_cut_spectra(plans, problem.grid, lambda _: pytest.fail("invalid plan evaluated"),
                                     max_evaluations=problem.grid.n_points, max_seconds=60)


@pytest.mark.parametrize("failure", ["lesser_shape", "greater_shape", "lesser_nan", "greater_inf"])
def test_invalid_evaluator_output_is_rejected(failure):
    problem, plans, green = _case()
    good = _evaluator(problem, green)

    def evaluate(omega):
        result = good(omega)
        if failure == "lesser_shape":
            result.lesser = result.lesser[:-1]
        elif failure == "greater_shape":
            result.greater = result.greater[:, :-1]
        elif failure == "lesser_nan":
            result.lesser = result.lesser.copy()
            result.lesser[0, 0] = np.nan
        else:
            result.greater = result.greater.copy()
            result.greater[0, 0] = np.inf
        return result

    with pytest.raises(ValueError):
        collect_selected_cut_spectra(plans, problem.grid, evaluate,
                                     max_evaluations=problem.grid.n_points, max_seconds=60)


def test_mid_collection_time_budget_failure_does_not_return_partial_grid(monkeypatch):
    problem, plans, green = _case()
    seen = []
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(selected_module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="after 1 frequencies"):
        collect_selected_cut_spectra(plans, problem.grid, _evaluator(problem, green, seen),
                                     max_evaluations=problem.grid.n_points, max_seconds=1)
    assert len(seen) == 1


def test_final_evaluator_call_cannot_overrun_time_budget_silently(monkeypatch):
    problem, plans, green = _case()
    seen = []
    ticks = iter([0.0] * (problem.grid.n_points + 1) + [2.0])
    monkeypatch.setattr(selected_module.time, "monotonic", lambda: next(ticks))
    with pytest.raises(TimeoutError, match="final frequency"):
        collect_selected_cut_spectra(plans, problem.grid, _evaluator(problem, green, seen),
                                     max_evaluations=problem.grid.n_points, max_seconds=1)
    assert len(seen) == problem.grid.n_points


@pytest.mark.parametrize("policy", [None, "", "equilibrium", "centered", "physical-means"])
def test_mean_policy_must_be_explicit_diagnostic(policy):
    problem, plans, green = _case()
    spectra = _collect(problem, plans, green)
    with pytest.raises(ValueError, match="mean_policy"):
        spectra.integrate(1, [0, .1], mean_policy=policy)


@pytest.mark.parametrize("cut", [True, 1.0, 0, 3])
def test_unknown_or_invalid_cuts_rejected(cut):
    problem, plans, green = _case()
    spectra = _collect(problem, plans, green)
    with pytest.raises(ValueError, match="cut"):
        spectra.integrate(cut, [0, .1], mean_policy=POLICY)


def test_material_memory_alias_guard_remains_in_selected_path():
    problem, plans, green = _case()
    spectra = _collect(problem, plans, green)
    with pytest.raises(ValueError, match="recurrence"):
        spectra.integrate(1, [0, np.pi / problem.grid.step], mean_policy=POLICY)


def test_selected_dressed_toy_total_matches_full_current_and_contacts():
    from test_anharmonic_steady_quantum_reference import reference

    state = reference.solve_reference()
    problem, green = state["problem"], state["green"]
    plans = tuple(build_contracted_cubic_cut(problem, cut) for cut in (1, 2))
    spectra = _collect(problem, plans, green)
    expected = reference.measure_currents(state)
    times = np.linspace(0, 150, 3001)
    left = state["metadata"]["contact_currents"][0]
    for old in expected["cuts"]:
        result = spectra.integrate(old["cut"], times, mean_policy=POLICY, block_size=16)
        np.testing.assert_allclose(result["connected_cubic"], old["cubic"], rtol=5e-12, atol=2e-15)
        np.testing.assert_allclose(result["harmonic"], old["harmonic"], rtol=5e-12, atol=2e-15)
        assert abs(result["centered_total"] - left) / left < 1e-3
        assert not result["full_material_current_validated"]
