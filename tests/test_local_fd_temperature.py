"""Selected generated-network thermal closure and guard oracles."""
from dataclasses import replace
import time
import numpy as np
import pytest
from negfpp.local_fd.temperature_solver import (
    SolverOptions, solve_linear_response, solve_probe_temperatures,
)


def dense_directed_network(probes):
    """Balanced directed ring plus arbitrary long-range reciprocal edges."""
    size = probes+2
    transmission = np.zeros((size, size))
    path = [0, *range(2, size), 1]
    for a, b in zip(path, path[1:]+path[:1]):
        transmission[a, b] += .7e-10
    rng = np.random.default_rng(616+probes)
    for a in range(size):
        for b in range(a+1, size):
            weight = (.1+rng.random())*1e-11/(1+abs(a-b))
            transmission[a, b] += weight
            transmission[b, a] += weight
    # Independent current operator for outgoing-minus-incoming convention.
    return np.diag(transmission.sum(axis=1))-transmission.T



def bose_reduced_temperature(temperatures):
    temperatures = np.asarray(temperatures)
    theta = np.array([20., 100., 470.])[:, None]
    x = theta/temperatures[None, :]
    return np.sum(theta/np.expm1(x)-theta/np.expm1(theta/300.), axis=0)



def bose_derivative(temperatures):
    temperatures = np.asarray(temperatures)
    theta = np.array([20., 100., 470.])[:, None]
    x = theta/temperatures[None, :]
    return np.sum(x*x*np.exp(-x)/(-np.expm1(-x))**2, axis=0)



def chain_action(probes, edge_conductance=1e-10):
    path = np.array([0, *range(2, probes+2), 1])
    def action(temperatures, vector):
        chain = vector[path]
        flux = edge_conductance*(chain[:-1]-chain[1:])
        ordered = np.zeros(probes+2)
        ordered[:-1] += flux
        ordered[1:] -= flux
        output = np.empty(probes+2)
        output[path] = ordered
        return output
    def currents(temperatures):
        return action(temperatures, temperatures-300.)
    return currents, action



@pytest.mark.parametrize("probes", [1, 9, 23])
def test_lr_exact_directed_long_range_network_matches_dense(probes):
    matrix = dense_directed_network(probes)
    def currents(temperatures):
        return matrix@(temperatures-300.)
    def action(temperatures, vector):
        return matrix@vector
    solution = solve_linear_response(currents, action, 300., probes,
                                     local_conductances=1e-10)
    expected = np.zeros(probes+2)
    expected[:2] = [.5, -.5]
    expected[2:] = np.linalg.solve(matrix[2:, 2:], -matrix[2:, :2]@expected[:2])
    assert solution.success, solution.scalar_summary()
    np.testing.assert_allclose(solution.temperature_shifts_kelvin, expected, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(solution.currents_w, matrix@expected, atol=1e-22, rtol=2e-11)
    assert solution.diagnostics["uniform_ward_passed"]
    assert solution.conductance_w_per_kelvin > 0



def test_nonlinear_bose_network_uses_complete_callback_and_true_flux():
    matrix = dense_directed_network(17)
    seen = []
    def currents(temperatures):
        seen.append(temperatures.copy())
        return matrix@bose_reduced_temperature(temperatures)
    def action(temperatures, vector):
        return matrix@(bose_derivative(temperatures)*vector)
    initial = np.full(19, 300.)
    result = solve_probe_temperatures(currents, action, initial, local_conductances=3e-10)
    assert result.success, result.scalar_summary()
    assert result.newton_iterations >= 2
    assert result.max_probe_flux_w <= result.flux_allowance_w
    # Independent dense elimination in the nonlinear Bose occupation variable.
    expected_occupations = np.zeros(19)
    expected_occupations[:2] = bose_reduced_temperature(np.array([300.5, 299.5]))
    expected_occupations[2:] = np.linalg.solve(matrix[2:, 2:], -matrix[2:, :2]@expected_occupations[:2])
    np.testing.assert_allclose(bose_reduced_temperature(result.temperatures_kelvin),
                               expected_occupations, rtol=1e-10, atol=3e-12)
    assert all(np.all((state >= 299.5) & (state <= 300.5)) for state in seen)
    assert result.diagnostics["preconditioner_only_approximation"]



def test_preconditioner_changes_no_physical_result():
    matrix = dense_directed_network(13)
    currents = lambda temperatures: matrix@(temperatures-300.)
    action = lambda temperatures, vector: matrix@vector
    scalar = solve_linear_response(currents, action, 300., 13, local_conductances=1e-10)
    varied = solve_linear_response(currents, action, 300., 13,
                                   local_conductances=np.linspace(.3e-10, 2e-10, 14))
    assert scalar.success and varied.success
    np.testing.assert_allclose(scalar.temperature_shifts_kelvin,
                               varied.temperature_shifts_kelvin, rtol=1e-11, atol=1e-12)
    for scale in (1e-25, 1., 1e15):
        alternate = solve_linear_response(currents, action, 300., 13,
                                          local_conductances=scale)
        assert alternate.success, alternate.scalar_summary()
        np.testing.assert_allclose(scalar.temperature_shifts_kelvin,
                                   alternate.temperature_shifts_kelvin, rtol=1e-11, atol=1e-12)



def test_uniform_temperature_is_checked_without_calling_zero_by_definition():
    currents, action = chain_action(4)
    result = solve_probe_temperatures(currents, action, np.full(6, 300.),
                                      contact_temperatures=(300., 300.))
    assert result.success and result.current_calls == 1 and result.jacobian_calls == 0
    corrupted = solve_probe_temperatures(lambda t: currents(t)+np.array([0, 0, 1e-20, 0, 0, 0]),
                                         action, np.full(6, 300.),
                                         contact_temperatures=(300., 300.))
    assert not corrupted.success
    assert corrupted.message == "Uniform-temperature current identity failed"



def test_full_uniform_ward_and_equilibrium_gate_detect_defects():
    currents, action = chain_action(4)
    def bad_ward(t, direction):
        return action(t, direction)+1e-13*np.mean(direction)*np.array([1., -1., 0, 0, 0, 0])
    result = solve_linear_response(currents, bad_ward, 300., 4, local_conductances=1e-10)
    assert not result.success and not result.diagnostics["uniform_ward_passed"]
    result = solve_linear_response(lambda t: currents(t)+np.array([1e-15, -1e-15, 0, 0, 0, 0]),
                                   action, 300., 4, local_conductances=1e-10)
    assert not result.success and not result.diagnostics["equilibrium_current_passed"]



def test_conservation_is_a_separate_gate_after_probe_solution():
    currents, action = chain_action(4)
    def nonconserving(t):
        values = currents(t)
        values[0] += 1e-15
        return values
    result = solve_probe_temperatures(nonconserving, action, np.full(6, 300.),
                                      local_conductances=1e-10)
    assert not result.success and result.diagnostics["numerical_residual_passed"]
    assert not result.diagnostics["total_balance_passed"]



def test_work_caps_stop_without_dense_fallback_or_success():
    matrix = dense_directed_network(9)
    currents = lambda t: matrix@(t-300.)
    action = lambda t, v: matrix@v
    options = replace(SolverOptions(), max_krylov_iterations=1)
    result = solve_linear_response(currents, action, 300., 9, options=options)
    assert not result.success and "cap" in result.message
    assert result.krylov_iterations == 1
    options = replace(SolverOptions(), max_jacobian_calls=1)
    result = solve_linear_response(currents, action, 300., 9, options=options)
    assert not result.success and result.jacobian_calls == 1
    result = solve_probe_temperatures(currents, action, np.full(11, 300.),
                                      deadline=time.monotonic()-1)
    assert not result.success and result.current_calls == 0
