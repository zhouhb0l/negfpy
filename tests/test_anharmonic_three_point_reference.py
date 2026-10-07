"""Independent finite-oscillator references for the connected cubic response.

The reference differentiates an actual Hamiltonian quench and evaluates matrix
commutators. It never constructs its answer from the implementation's Wick
contractions. A rotated two-mode vacuum is exactly Gaussian and stationary;
cutoff six is sufficient for the degree-six matrix products used below.
"""

from itertools import permutations, product

import numpy as np
import pytest

from negfpy.anharmonic.cubic_cut_current import build_cubic_cut_current_operator
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.three_point import combine_cubic_moments, connected_cubic_moments
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
)
from negfpy.core.types import Device1D, LeadBlocks


def _integrate(values, times):
    """Explicit composite trapezoid, independent of the production integrator."""
    return np.sum((values[1:] + values[:-1]) * np.diff(times)[:, None] / 2, axis=0)


def _model(cutoff=6):
    frequencies = np.array([0.9, 1.4])
    annihilation = np.diag(np.sqrt(np.arange(1, cutoff)), 1)
    identity = np.eye(cutoff)
    annihilators = [np.kron(annihilation, identity), np.kron(identity, annihilation)]
    q = [(a + a.T) / np.sqrt(2 * w) for a, w in zip(annihilators, frequencies)]
    p = [-1j * np.sqrt(w / 2) * (a - a.T) for a, w in zip(annihilators, frequencies)]
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    u = np.einsum("ia,amn->imn", rotation, q)
    v = np.einsum("ia,amn->imn", rotation, p)
    energies = np.array([frequencies @ (np.array(occupations) + 0.5)
                         for occupations in product(range(cutoff), repeat=2)])
    tensor = np.zeros((2, 2, 2))
    for indices, value in [((0, 0, 0), 0.47), ((0, 0, 1), -0.61),
                           ((0, 1, 1), 0.33), ((1, 1, 1), -0.29)]:
        for index in set(permutations(indices)):
            tensor[index] = value
    potential = np.zeros_like(u[0], dtype=complex)
    for a, b, c in product(range(2), repeat=3):
        potential += tensor[a, b, c] * (u[a] @ u[b] @ u[c]) / 6
    indices = np.array(list(product(range(2), repeat=3)), dtype=int)
    operators = np.array([(v[i] @ u[j] @ u[k] + u[j] @ u[k] @ v[i]) / 2
                          for i, j, k in indices])
    return energies, u, v, potential, tensor, indices, operators


def _at_time(operator, energies, time):
    phase = np.exp(1j * energies * time)
    return phase[:, None] * operator * phase.conj()[None, :]


def _linear_response(operator, potential, energies, times):
    """-i integral <[O(T), V(s)]>_vac, by finite oscillator matrices."""
    endpoint = _at_time(operator, energies, times[-1])
    integrand = []
    for time in times:
        perturbation = _at_time(potential, energies, time)
        # Matrix element of the actual commutator, without Wick expansion.
        integrand.append((-1j * (endpoint[0] @ perturbation[:, 0]
                                - perturbation[0] @ endpoint[:, 0])).real)
    return _integrate(np.asarray(integrand)[:, None], times)[0]


def _linear_response_exact(operator, potential, energies, duration):
    """Integrate finite-matrix interaction-picture phases analytically."""
    endpoint = _at_time(operator, energies, duration)
    gaps = energies - energies[0]
    weights = np.full(gaps.shape, duration, dtype=complex)
    nonzero = gaps != 0
    weights[nonzero] = np.expm1(1j * gaps[nonzero] * duration) / (1j * gaps[nonzero])
    return 2 * np.imag(np.sum(endpoint[0] * potential[:, 0] * weights))


def _correlations(energies, u, v, times):
    displacement = np.empty((len(times), 2, 2), dtype=complex)
    velocity = np.empty_like(displacement)
    for n, time in enumerate(times):
        for i, a in product(range(2), repeat=2):
            displacement[n, i, a] = _at_time(u[i], energies, time)[0] @ u[a][:, 0]
            velocity[n, i, a] = _at_time(v[i], energies, time)[0] @ u[a][:, 0]
    return displacement, velocity


def _disconnected_response(indices, u, v, mean_u_response, mean_p_response):
    result = []
    for i, j, k in indices:
        qq = (u[j] @ u[k])[0, 0].real
        pq_ik = ((v[i] @ u[k] + u[k] @ v[i]) / 2)[0, 0].real
        pq_ij = ((v[i] @ u[j] + u[j] @ v[i]) / 2)[0, 0].real
        result.append(mean_p_response[i] * qq + mean_u_response[j] * pq_ik
                      + mean_u_response[k] * pq_ij)
    return np.array(result)


def build_reference():
    """Build reusable independent small-matrix reference data for tests/reports."""
    energies, u, v, potential, tensor, indices, operators = _model()
    times = np.linspace(0, 0.63, 501)
    displacement, velocity = _correlations(energies, u, v, times)
    raw = np.array([_linear_response(operator, potential, energies, times)
                    for operator in operators])
    mean_u = np.array([_linear_response(operator, potential, energies, times) for operator in u])
    mean_p = np.array([_linear_response(operator, potential, energies, times) for operator in v])
    disconnected = _disconnected_response(indices, u, v, mean_u, mean_p)
    exact_raw = np.array([_linear_response_exact(op, potential, energies, times[-1])
                          for op in operators])
    exact_u = np.array([_linear_response_exact(op, potential, energies, times[-1]) for op in u])
    exact_p = np.array([_linear_response_exact(op, potential, energies, times[-1]) for op in v])
    exact = exact_raw - _disconnected_response(indices, u, v, exact_u, exact_p)
    return {
        "energies": energies, "u": u, "v": v, "potential": potential,
        "tensor": tensor, "indices": indices, "operators": operators,
        "times": times, "displacement": displacement, "velocity": velocity,
        "raw": raw, "disconnected": disconnected, "connected": raw - disconnected,
        "exact_connected": exact, "mean_u_response": mean_u, "mean_p_response": mean_p,
    }


@pytest.fixture(scope="module")
def quench_reference():
    return build_reference()


def test_connected_response_matches_independent_quantum_commutator(quench_reference):
    ref = quench_reference
    result = connected_cubic_moments(
        DenseCubicVertex(ref["tensor"]), ref["indices"], ref["times"],
        ref["displacement"], ref["velocity"],
    )
    np.testing.assert_allclose(result["connected_moments"], ref["connected"],
                               atol=8e-15, rtol=2e-13)
    assert np.max(np.abs(ref["connected"])) > 0.02
    assert np.max(np.abs(ref["disconnected"])) > 0.02
    # A raw three-point expectation cannot be silently labelled connected.
    assert np.max(np.abs(result["connected_moments"] - ref["raw"])) > 0.02


def test_quantum_term_is_required_even_for_vacuum(quench_reference):
    ref = quench_reference
    c, p = ref["displacement"], ref["velocity"]
    semiclassical = np.zeros((len(ref["times"]), len(ref["indices"])))
    for n, (i, j, k) in enumerate(ref["indices"]):
        for a, b, d in product(range(2), repeat=3):
            x, y, z = p[:, i, a], c[:, j, b], c[:, k, d]
            # Deliberately omit -Im(x)*Im(y)*Im(z), the quantum correction.
            semiclassical[:, n] += 2 * ref["tensor"][a, b, d] * (
                x.imag * y.real * z.real + x.real * y.imag * z.real
                + x.real * y.real * z.imag
            )
    wrong = _integrate(semiclassical, ref["times"])
    assert np.max(np.abs(wrong - ref["connected"])) > 0.009


def _exact_quench_moments(ref, strength):
    hamiltonian = np.diag(ref["energies"]) + strength * ref["potential"]
    energies, vectors = np.linalg.eigh(hamiltonian)
    state = vectors @ (np.exp(-1j * energies * ref["times"][-1]) * vectors[0].conj())
    mean_u = np.array([np.vdot(state, operator @ state).real for operator in ref["u"]])
    mean_p = np.array([np.vdot(state, operator @ state).real for operator in ref["v"]])
    raw = np.array([np.vdot(state, operator @ state).real for operator in ref["operators"]])
    disconnected = []
    for i, j, k in ref["indices"]:
        # Central third cumulant uses full, perturbed second moments and means.
        qq = np.vdot(state, ref["u"][j] @ ref["u"][k] @ state).real
        pq_ik = np.vdot(state, (ref["v"][i] @ ref["u"][k]
                                + ref["u"][k] @ ref["v"][i]) @ state).real / 2
        pq_ij = np.vdot(state, (ref["v"][i] @ ref["u"][j]
                                + ref["u"][j] @ ref["v"][i]) @ state).real / 2
        disconnected.append(mean_p[i] * qq + mean_u[j] * pq_ik + mean_u[k] * pq_ij
                            - 2 * mean_p[i] * mean_u[j] * mean_u[k])
    return raw - np.array(disconnected)


def test_exact_finite_coupling_quench_derivative_matches_connected_response(quench_reference):
    ref = quench_reference
    errors = []
    for strength in (0.02, 0.01):
        derivative = (_exact_quench_moments(ref, strength)
                      - _exact_quench_moments(ref, -strength)) / (2 * strength)
        errors.append(np.max(np.abs(derivative - ref["exact_connected"])))
    # The matrix phase integrals provide an independent exact-in-time answer.
    assert np.max(np.abs(ref["connected"] - ref["exact_connected"])) < 1e-7
    # Centered finite-coupling derivative converges at O(strength**2).
    assert errors[1] < 0.32 * errors[0]
    assert errors[1] < 8e-6


def test_cubic_cut_current_with_explicit_means_matches_hamiltonian_response(quench_reference):
    """The coefficient of g**2 in cubic current, including response means.

    J3 itself is linear in g, while its leading three-point state moment is
    linear in g. This checks that component only; the harmonic-current state
    response is a separate O(g**2) contribution to the total current.
    """
    ref = quench_reference
    angle = 0.37
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]])
    dynamical = rotation @ np.diag(np.array([0.9, 1.4]) ** 2) @ rotation.T
    device = Device1D([np.array([[dynamical[i, i]]]) for i in range(2)],
                      [np.array([[dynamical[0, 1]]])])
    lead = LeadBlocks(np.array([[2.0]]), np.array([[-0.2]]))
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.0),
        right=HarmonicReservoir(lead, temperature=0.0),
        vertex=DenseCubicVertex(ref["tensor"]),
        grid=SignedFrequencyGrid(step=0.1, n_positive=3),
    )
    operator = build_cubic_cut_current_operator(problem, cut=1)
    connected = connected_cubic_moments(
        problem.vertex, operator.cubic_indices, ref["times"],
        ref["displacement"], ref["velocity"],
    )["connected_moments"]
    u, p = ref["u"], ref["v"]
    covariance = np.array([[(u[j] @ u[k])[0, 0].real for k in range(2)] for j in range(2)])
    mixed = np.array([[((p[i] @ u[j] + u[j] @ p[i]) / 2)[0, 0].real
                       for j in range(2)] for i in range(2)])
    # Feed actual small-g first-order means through the public assembly API.
    # Remove its explicit g**3 product before extracting the linear coefficient:
    # it belongs to higher order and cannot be compared with a one-vertex result.
    strength = 0.01
    mean_u = strength * ref["mean_u_response"]
    mean_p = strength * ref["mean_p_response"]
    assembled = combine_cubic_moments(
        operator.cubic_indices, strength * connected,
        displacement_mean=mean_u, velocity_mean=mean_p,
        displacement_covariance=covariance, velocity_displacement_covariance=mixed,
    )
    i, j, k = operator.cubic_indices.T
    moments_linear = (assembled["total_moments"] - mean_p[i] * mean_u[j] * mean_u[k]) / strength
    result = operator.evaluate_moments(np.zeros(len(operator.harmonic_indices)), moments_linear)

    # Independently assemble the cubic part of -i[H,H_A] from the Hamiltonian
    # and equal-leg local energy. No compiled cut-current coefficients enter.
    kinetic = sum(momentum @ momentum / 2 for momentum in p)
    local_kinetic = p[0] @ p[0] / 2
    local_cubic = np.zeros_like(ref["potential"])
    membership = np.array([1, 0])
    for a, b, c in product(range(2), repeat=3):
        monomial_energy = ref["tensor"][a, b, c] * (u[a] @ u[b] @ u[c]) / 6
        local_cubic += monomial_energy * (membership[a] + membership[b] + membership[c]) / 3
    cubic_current = -1j * (kinetic @ local_cubic - local_cubic @ kinetic
                           + ref["potential"] @ local_kinetic
                           - local_kinetic @ ref["potential"])
    expected = _linear_response(cubic_current, ref["potential"], ref["energies"], ref["times"])
    np.testing.assert_allclose(result["cubic_solver"], expected, atol=2e-14, rtol=3e-13)
    assert abs(expected) > 1e-3
    connected_only = operator.cubic_coefficients @ connected
    assert abs(connected_only - expected) > 1e-3
    assert result["spatial_validation_passed"] is False
