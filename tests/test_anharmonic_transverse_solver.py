"""Small physical and failure-path checks for the coupled-q toy solver."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_q import TransverseCubicVertex
from negfpy.anharmonic.transverse_solver import (
    TransverseSCBAConfig,
    TransverseSCBAProblem,
    TransverseSpectrum,
    reconstruct_transverse_spectrum,
    solve_transverse_scba,
    transverse_heat_current,
    validate_transverse_spectrum,
)
from negfpy.anharmonic.types import SolverScale


def _ohmic(grid, damping, temperature):
    """Analytic Ohmic bath, including its finite Bose product at zero."""
    w = grid.values
    product = np.zeros_like(w)
    if temperature:
        nonzero = w != 0
        product[nonzero] = w[nonzero] / np.expm1(w[nonzero] / temperature)
        product[grid.zero_index] = temperature
    else:
        product[w < 0] = -w[w < 0]
    retarded = -1j * w[None, :, None, None] * damping[:, None]
    lesser = -2j * product[None, :, None, None] * damping[:, None]
    greater = lesser + retarded - retarded.conj().swapaxes(-1, -2)
    return TransverseSpectrum(retarded, lesser, greater)


def _problem(*, nq=1, d=1, coupling=0.04, temperatures=(0.8, 0.6)):
    grid = SignedFrequencyGrid(0.4, 4)
    mesh = UniformBZMesh((nq, 1, 1))
    dynamical = np.tile(np.eye(d) * 1.4, (nq, 1, 1)).astype(complex)
    damping = np.tile(np.eye(d) * 0.3, (nq, 1, 1)).astype(complex)
    if d == 2:
        for q in range(nq):
            phase = np.exp(2j * np.pi * q / nq)
            dynamical[q, 0, 1] = -0.1 - 0.04 * phase
            dynamical[q, 1, 0] = dynamical[q, 0, 1].conjugate()
            damping[q, 0, 1] = 0.04 * phase
            damping[q, 1, 0] = damping[q, 0, 1].conjugate()
    coefficients = np.zeros((nq, nq, d, d, d))
    for a in range(d):
        coefficients[:, :, a, a, a] = coupling
    return TransverseSCBAProblem(
        grid, TransverseCubicVertex(mesh, coefficients), dynamical,
        _ohmic(grid, damping, temperatures[0]),
        _ohmic(grid, damping * 0.7, temperatures[1]),
    )


def test_zero_vertex_complex_four_q_matches_harmonic_landauer():
    problem = _problem(nq=4, d=2, coupling=0)
    result = solve_transverse_scba(problem)
    assert result.converged and len(result.history) == 1
    np.testing.assert_array_equal(result.sigma_pp.lesser, 0)
    grid = problem.grid
    w = grid.values[grid.positive_indices]
    left = problem.left.retarded[:, grid.positive_indices]
    right = problem.right.retarded[:, grid.positive_indices]
    harmonic = np.linalg.inv(
        w[None, :, None, None] ** 2 * np.eye(2)
        - problem.dynamical[:, None] - left - right,
    )
    gamma_l = 1j * (left - left.conj().swapaxes(-1, -2))
    gamma_r = 1j * (right - right.conj().swapaxes(-1, -2))
    transmission = np.trace(
        gamma_l @ harmonic @ gamma_r @ harmonic.conj().swapaxes(-1, -2),
        axis1=-2, axis2=-1,
    ).real
    occupation_difference = 1 / np.expm1(w / 0.8) - 1 / np.expm1(w / 0.6)
    expected = grid.integration_weight * np.sum(w * transmission * occupation_difference) / 4
    assert result.currents.left == pytest.approx(expected, rel=2e-13)
    assert result.currents.right == pytest.approx(-expected, rel=2e-13)
    assert result.currents.collision_power == 0
    scaled = transverse_heat_current(grid, problem.left, result.green, SolverScale(3, 2))
    assert scaled == pytest.approx(18 * expected, rel=2e-13)


def test_joint_reflection_allows_complex_zero_frequency_q_block():
    problem = _problem(nq=4, d=2)
    field = reconstruct_transverse_spectrum(
        problem.grid, problem.left.lesser, problem.left.greater,
    )
    metrics = validate_transverse_spectrum(problem.vertex.mesh, problem.grid, field)
    assert max(metrics.values()) <= 1
    zero = problem.grid.zero_index
    # These violations of same-q rules are legitimate for q=+/-pi/2.
    assert np.max(np.abs(field.retarded[1, zero].imag)) > 0.01
    assert np.max(np.abs(field.lesser[1, zero] - field.greater[1, zero].T)) > 0.01


def test_broken_opposite_q_pair_is_rejected_without_repair():
    problem = _problem(nq=4, d=2)
    components = {}
    for name in ("retarded", "lesser", "greater"):
        values = getattr(problem.left, name).copy()
        values[1] *= 1.2
        components[name] = values
    bad = TransverseSpectrum(**components)
    with pytest.raises(ValueError, match="reflection|reality"):
        validate_transverse_spectrum(problem.vertex.mesh, problem.grid, bad)
    np.testing.assert_array_equal(bad.lesser, components["lesser"])


def test_negative_correlation_eigenvalue_is_rejected():
    problem = _problem()
    contact = problem.left
    bad = TransverseSpectrum(
        contact.retarded, contact.lesser + 10j, contact.greater + 10j,
    )
    with pytest.raises(ValueError, match="psd"):
        validate_transverse_spectrum(problem.vertex.mesh, problem.grid, bad)


def test_unmixed_residual_prevents_false_small_mixing_convergence():
    result = solve_transverse_scba(
        _problem(), TransverseSCBAConfig(max_iterations=2, mixing=1e-12),
    )
    assert not result.converged and result.reason == "max_iterations"
    assert result.history[-1].sigma_residual > 100
    assert np.max(np.abs(result.sigma_pp.lesser)) < 1e-12


def test_iteration_limit_returns_evaluated_input_and_its_green():
    problem = _problem()
    result = solve_transverse_scba(problem, TransverseSCBAConfig(max_iterations=1))
    assert not result.converged
    np.testing.assert_array_equal(result.sigma_pp.retarded, 0)
    expected = np.linalg.inv(
        problem.grid.values[None, :, None, None] ** 2
        - problem.dynamical[:, None] - problem.left.retarded - problem.right.retarded,
    )
    np.testing.assert_allclose(result.green.retarded, expected, rtol=0, atol=0)
    # Returning the fresh raw output here would break this exact input check.
    assert result.history[-1].sigma_residual > 1


def test_equilibrium_interacting_fixed_point_has_zero_heat_flow():
    problem = _problem(temperatures=(0.8, 0.8))
    snapshots = [
        values.copy() for values in
        (problem.dynamical, problem.vertex.coefficients, problem.left.lesser)
    ]
    result = solve_transverse_scba(
        problem, TransverseSCBAConfig(mixing=0.8, max_iterations=30),
    )
    assert result.converged
    assert np.max(np.abs(result.sigma_pp.lesser)) > 1e-5
    assert abs(result.currents.left) < 1e-12
    assert abs(result.currents.right) < 1e-12
    assert abs(result.currents.collision_power) < 1e-12
    for actual, before in zip(
        (problem.dynamical, problem.vertex.coefficients, problem.left.lesser),
        snapshots, strict=True,
    ):
        np.testing.assert_array_equal(actual, before)
        assert not actual.flags.writeable
    assert not result.sigma_pp.lesser.flags.writeable
    assert not result.green.retarded.flags.writeable


def test_zero_temperature_zero_vertex_with_relative_only_current_gate():
    result = solve_transverse_scba(
        _problem(coupling=0, temperatures=(0, 0)),
        TransverseSCBAConfig(current_atol=0),
    )
    assert result.converged and result.history[-1].current_residual == 0


def test_noncausal_initial_retarded_field_is_rejected():
    problem = _problem()
    zeros = np.zeros_like(problem.left.retarded)
    initial = TransverseSpectrum(zeros + 0.1, zeros, zeros)
    with pytest.raises(ValueError, match="causal reconstruction"):
        solve_transverse_scba(problem, initial_sigma=initial)


def test_work_budget_rejects_before_first_map():
    problem = _problem()
    with pytest.raises(ValueError, match="max_pair_evaluations"):
        solve_transverse_scba(problem, TransverseSCBAConfig(max_pair_evaluations=1))


def test_incompatible_harmonic_reality_is_rejected():
    problem = _problem(nq=4, d=2)
    dynamical = problem.dynamical.copy()
    dynamical[1, 0, 0] += 0.1
    with pytest.raises(ValueError, match="D\\(-q\\)"):
        replace(problem, dynamical=dynamical)


@pytest.mark.parametrize("kwargs", [
    {"mixing": 0}, {"mixing": 1.1}, {"max_iterations": 0},
    {"max_iterations": True}, {"sigma_atol": -1},
    {"sigma_atol": 0, "sigma_rtol": 0}, {"identity_rtol": np.nan},
])
def test_invalid_iteration_configuration(kwargs):
    with pytest.raises((TypeError, ValueError)):
        TransverseSCBAConfig(**kwargs)
