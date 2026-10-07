"""Fixed-field frequency reconstruction; these tests do not iterate SCBA."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json

import numpy as np
import pytest

from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.fixed_field_green import FixedFieldGreenEvaluator
from negfpy.anharmonic.interacting_pointwise import PointwiseInteracting
from negfpy.anharmonic.types import KeldyshSpectrum, SCBAConfig
from negfpy.core.types import LeadBlocks
from test_anharmonic_harmonic_pointwise import _problem
from test_anharmonic_interacting_pointwise import _sigma


@pytest.mark.parametrize("acoustic", [False, True])
@pytest.mark.parametrize("temperatures", [(0.7, 0.7), (0.2, 1.1), (0.0, 0.7)])
def test_all_signed_nodes_and_zero_match_production_equations(acoustic, temperatures):
    # Gapped fixture has nonsymmetric oriented leads, complex contact
    # couplings and reversed explicit device indices.
    problem = _problem(acoustic=acoustic, temperatures=temperatures)
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof, complex_matrix=not acoustic)
    left, right = build_contact_spectra(problem, config.eta_lead)
    retarded = solve_retarded(problem, left.retarded, right.retarded, sigma.retarded)
    expected = solve_keldysh_green(retarded, left.lesser + right.lesser + sigma.lesser,
                                  left.greater + right.greater + sigma.greater)
    evaluator = FixedFieldGreenEvaluator(problem, config, sigma, selected_entries=[(0, 1)])
    for index, omega in enumerate(problem.grid.values):
        actual = evaluator.evaluate(omega, diagnostic=index == problem.grid.zero_index)
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name)[index],
                                       rtol=5e-12, atol=2e-12)
    zero = evaluator.evaluate(0.0, diagnostic=True)
    assert zero.diagnostics["dyson_backward_error"] < 1e-14
    assert zero.diagnostics["device_operator_singular_min"] > 0
    assert zero.diagnostics["eta_device"] == 0
    assert zero.diagnostics["self_energy_interpolation"]["zero_node_preserved"]
    assert zero.diagnostics["zero_contact_limits"]["left"]["temperature"] == temperatures[0]
    assert zero.diagnostics["zero_contact_limits"]["right"]["temperature"] == temperatures[1]
    json.dumps(zero.diagnostics, allow_nan=False)
    for actual, expected_field in zip(evaluator(0.0), (zero.lesser, zero.greater)):
        np.testing.assert_array_equal(actual, expected_field)


def test_acoustic_zero_is_nonzero_and_near_zero_continuous():
    problem = _problem(acoustic=True, temperatures=(0.2, 1.1))
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof, complex_matrix=False)
    evaluator = FixedFieldGreenEvaluator(problem, config, sigma)
    zero = evaluator.evaluate(0.0)
    assert np.max(np.abs(zero.greater)) > 1
    for omega in (-1e-7, 1e-7):
        near = evaluator.evaluate(omega)
        np.testing.assert_allclose(near.greater, zero.greater, rtol=3e-5, atol=1e-7)
        np.testing.assert_allclose(near.lesser, zero.lesser, rtol=3e-5, atol=1e-7)
    for omega in (-0.713, 0.713):
        original = PointwiseInteracting(problem, config, sigma).evaluate(omega)
        actual = evaluator.evaluate(omega)
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_array_equal(getattr(actual, name), getattr(original, name))


def test_supplied_zero_node_is_preserved_and_results_are_independent():
    problem = _problem(acoustic=True)
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof, complex_matrix=False)
    fields = {name: getattr(sigma, name).copy() for name in ("retarded", "lesser", "greater")}
    zero = problem.grid.zero_index
    # Independent Hermitian static shift is deliberately different from the
    # transform of this Keldysh pair; a fixed-field evaluator must preserve it.
    fields["retarded"][zero] += 0.013 * np.eye(problem.ndof)
    fields["lesser"][zero] *= 1.3
    fields["greater"][zero] *= 1.3
    supplied = KeldyshSpectrum(**fields)
    evaluator = FixedFieldGreenEvaluator(problem, config, supplied)
    first = evaluator.evaluate(0.0)
    for name in fields:
        np.testing.assert_array_equal(getattr(first.self_energy, name), fields[name][zero])
    first.greater[:] = 0
    first.self_energy.greater[:] = 0
    first.diagnostics["zero_contact_limits"]["left"]["temperature"] = -1
    second = evaluator.evaluate(0.0)
    assert np.max(np.abs(second.greater)) > 0
    assert second.diagnostics["zero_contact_limits"]["left"]["temperature"] == problem.left.temperature
    np.testing.assert_array_equal(second.self_energy.greater, fields["greater"][zero])


def test_zero_calls_are_thread_safe():
    problem = _problem(acoustic=True, temperatures=(0.2, 1.1))
    evaluator = FixedFieldGreenEvaluator(problem, SCBAConfig(eta_lead=0.025),
                                        _sigma(problem.grid, problem.ndof, complex_matrix=False))
    frequencies = [0.0, -0.01, 0.0, 0.01]
    serial = [evaluator.evaluate(omega) for omega in frequencies]
    with ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(evaluator.evaluate, frequencies))
    for first, second in zip(serial, parallel):
        for name in ("retarded", "lesser", "greater"):
            np.testing.assert_array_equal(getattr(first, name), getattr(second, name))


def test_rejects_wrong_policy_device_eta_and_outside_queries():
    problem = _problem(acoustic=True)
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof)
    with pytest.raises(ValueError, match="linear-keldysh"):
        FixedFieldGreenEvaluator(problem, config, sigma, policy="equilibrium-broadening")
    with pytest.raises(ValueError, match="eta_device"):
        FixedFieldGreenEvaluator(problem, replace(config, eta_device=1e-5), sigma)
    evaluator = FixedFieldGreenEvaluator(problem, config, sigma)
    for omega in [np.nan, np.inf, problem.grid.cutoff + 0.001]:
        with pytest.raises(ValueError):
            evaluator.evaluate(omega)


def test_mixed_acoustic_and_gapped_contacts_retain_separate_zero_policies():
    problem = _problem(acoustic=True, temperatures=(0.2, 1.1))
    right = replace(problem.right, lead=LeadBlocks(np.array([[2.2]]), np.array([[-1.0]])))
    problem = replace(problem, right=right)
    config = SCBAConfig(eta_lead=0.025)
    sigma = _sigma(problem.grid, problem.ndof, complex_matrix=False)
    left, right = build_contact_spectra(problem, config.eta_lead)
    retarded = solve_retarded(problem, left.retarded, right.retarded, sigma.retarded)
    expected = solve_keldysh_green(retarded, left.lesser + right.lesser + sigma.lesser,
                                  left.greater + right.greater + sigma.greater)
    actual = FixedFieldGreenEvaluator(problem, config, sigma).evaluate(0.0)
    for name in ("retarded", "lesser", "greater"):
        np.testing.assert_allclose(getattr(actual, name), getattr(expected, name)[problem.grid.zero_index],
                                   rtol=1e-12, atol=1e-12)
    assert actual.diagnostics["zero_contact_limits"]["left"]["classification"] == "acoustic"
    assert actual.diagnostics["zero_contact_limits"]["right"]["classification"] == "gapped"


def test_zero_failure_does_not_use_pseudoinverse(monkeypatch):
    problem = _problem(acoustic=True)
    evaluator = FixedFieldGreenEvaluator(problem, SCBAConfig(eta_lead=0.025),
                                        _sigma(problem.grid, problem.ndof, complex_matrix=False))

    def fail(*_args, **_kwargs):
        raise np.linalg.LinAlgError("singular exact-zero operator")

    monkeypatch.setattr(np.linalg, "solve", fail)
    with pytest.raises(np.linalg.LinAlgError, match="singular exact-zero"):
        evaluator.evaluate(0.0)
