"""Hamiltonian references for the opt-in equal-leg cubic cut operator.

The classical reference differentiates the partitioned potential, and the
quantum reference commutes independently assembled finite oscillator matrices.
Neither reference reconstructs the implementation's current coefficients.
"""

from dataclasses import replace
from itertools import permutations, product

import numpy as np
import pytest
from scipy import sparse

from negfpy.anharmonic._harmonic import solve_keldysh_green, solve_retarded
from negfpy.anharmonic.contacts import build_contact_spectra
from negfpy.anharmonic.cubic_cut_current import build_cubic_cut_current_operator
from negfpy.anharmonic.current import bond_heat_currents
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import fpu_alpha_chain
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SolverScale,
)
from negfpy.core.types import Device1D, LeadBlocks


def _tensor(ndof, terms):
    values = np.zeros((ndof, ndof, ndof))
    for indices, coefficient in terms:
        for ordered in set(permutations(indices)):
            values[ordered] = coefficient
    return values


def _problem(values, *, npl=1, sparse_vertex=False, scale=None):
    ndof = values.shape[0]
    n_layers = ndof // npl
    onsite = [np.diag(np.linspace(2.2, 2.6, npl)) for _ in range(n_layers)]
    couplings = [
        -0.3 * np.eye(npl) + 0.025 * np.ones((npl, npl))
        for _ in range(n_layers - 1)
    ]
    device = Device1D(onsite, couplings)
    lead = LeadBlocks(np.eye(npl) * 2.5, -np.eye(npl) * 0.3)
    vertex = DenseCubicVertex(values)
    if sparse_vertex:
        vertex = SparseCubicVertex.from_dense(vertex)
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.8),
        right=HarmonicReservoir(lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=3),
        scale=SolverScale() if scale is None else scale,
    )


def _dynamical(problem):
    """Independent assembly, only for tiny reference devices."""
    npl = problem.device.dof_per_layer
    values = np.zeros((problem.ndof, problem.ndof))
    for layer, onsite in enumerate(problem.device.onsite_blocks):
        region = slice(layer * npl, (layer + 1) * npl)
        values[region, region] = onsite
    for layer, coupling in enumerate(problem.device.coupling_blocks):
        left = slice(layer * npl, (layer + 1) * npl)
        right = slice((layer + 1) * npl, (layer + 2) * npl)
        values[left, right] = coupling
        values[right, left] = coupling.T
    return values


def _potential(q, dynamical, cubic, membership):
    """V and V_A from equal shares of each coordinate leg, before differentiation."""
    total = 0.0
    local = 0.0
    for a, b in product(range(q.size), repeat=2):
        energy = dynamical[a, b] * q[a] * q[b] / 2.0
        total += energy
        local += energy * (membership[a] + membership[b]) / 2.0
    for a, b, c in product(range(q.size), repeat=3):
        energy = cubic[a, b, c] * q[a] * q[b] * q[c] / 6.0
        total += energy
        local += energy * (membership[a] + membership[b] + membership[c]) / 3.0
    return total, local


def _classical_reference(q, p, dynamical, cubic, membership):
    """Hamilton equations and complex-step gradients, without a current formula."""
    gradients = np.empty((2, q.size))
    step = 1e-25
    for coordinate in range(q.size):
        displaced = q.astype(complex)
        displaced[coordinate] += 1j * step
        gradients[:, coordinate] = np.imag(
            _potential(displaced, dynamical, cubic, membership)
        ) / step
    total_force = -gradients[0]
    # dH_A/dt = dT_A/dp dot dp/dt + dV_A/dq dot dq/dt.
    return -float(np.dot(membership * p, total_force) + np.dot(gradients[1], p))


def _assert_scope(operator, result, evaluation):
    assert operator.metadata["partition"] == "equal-coordinate-leg"
    assert operator.metadata["orientation"] == "left-to-right"
    assert result["evaluation"] == evaluation
    assert result["full_operator_included"] is True
    assert result["state_moments_verified"] is False
    assert result["spatial_validation_passed"] is False
    assert result["changes_solver_acceptance"] is False


@pytest.mark.parametrize("cut", [1, 2])
@pytest.mark.parametrize("sparse_vertex", [False, True])
def test_classical_current_matches_partitioned_hamiltonian_gradient(cut, sparse_vertex):
    # Includes repeated indices, three-layer terms, and intralayer cubic terms.
    values = _tensor(6, [((0, 0, 3), 0.7), ((1, 2, 5), -0.4), ((4, 4, 5), 0.9)])
    problem = _problem(values, npl=2, sparse_vertex=sparse_vertex)
    operator = build_cubic_cut_current_operator(problem, cut)
    dynamical = _dynamical(problem)
    membership = (np.arange(6) < 2 * cut).astype(float)
    rng = np.random.default_rng(97002)
    for _ in range(5):
        q, p = rng.normal(size=(2, 6))
        result = operator.evaluate_phase_space(q, p)
        expected = _classical_reference(q, p, dynamical, values, membership)
        np.testing.assert_allclose(result["total_solver"], expected, rtol=1e-12, atol=1e-13)
        harmonic = _classical_reference(q, p, dynamical, np.zeros_like(values), membership)
        np.testing.assert_allclose(result["harmonic_solver"], harmonic, atol=1e-13)
        np.testing.assert_allclose(result["cubic_solver"], expected - harmonic, atol=1e-13)
        # Complement ownership reverses orientation without changing the state.
        reverse = _classical_reference(q, p, dynamical, values, 1.0 - membership)
        np.testing.assert_allclose(expected, -reverse, atol=1e-13)
    _assert_scope(operator, result, "classical-phase-space")


def _oscillators(n_modes=3, cutoff=7):
    annihilation = sparse.diags(np.sqrt(np.arange(1, cutoff)), 1, format="csr")
    local_q = (annihilation + annihilation.T) / np.sqrt(2.0)
    local_p = -1j * (annihilation - annihilation.T) / np.sqrt(2.0)
    q, p = [], []
    for mode in range(n_modes):
        q_operator = sparse.csr_matrix([[1.0]])
        p_operator = sparse.csr_matrix([[1.0]])
        for place in range(n_modes):
            identity = sparse.eye(cutoff, format="csr")
            q_operator = sparse.kron(q_operator, local_q if place == mode else identity, format="csr")
            p_operator = sparse.kron(p_operator, local_p if place == mode else identity, format="csr")
        q.append(q_operator)
        p.append(p_operator)
    return q, p


def _quantum_hamiltonians(q, p, dynamical, cubic, membership):
    dimension = q[0].shape[0]
    hamiltonian = sparse.csr_matrix((dimension, dimension), dtype=complex)
    local = hamiltonian.copy()
    for a in range(len(q)):
        kinetic = (p[a] @ p[a]) / 2.0
        hamiltonian += kinetic
        local += membership[a] * kinetic
    for a, b in product(range(len(q)), repeat=2):
        potential = (q[a] @ q[b]) * (dynamical[a, b] / 2.0)
        hamiltonian += potential
        local += potential * (membership[a] + membership[b]) / 2.0
    for a, b, c in product(range(len(q)), repeat=3):
        potential = (q[a] @ q[b] @ q[c]) * (cubic[a, b, c] / 6.0)
        hamiltonian += potential
        local += potential * (membership[a] + membership[b] + membership[c]) / 3.0
    return hamiltonian, local


@pytest.mark.parametrize("cut", [1, 2])
def test_quantum_operator_matches_independent_hamiltonian_commutator(cut):
    values = _tensor(3, [((0, 0, 1), 0.7), ((0, 1, 2), -0.4), ((2, 2, 2), 0.9)])
    problem = _problem(values)
    operator = build_cubic_cut_current_operator(problem, cut)
    q, p = _oscillators()
    membership = (np.arange(3) < cut).astype(float)
    hamiltonian, local = _quantum_hamiltonians(q, p, _dynamical(problem), values, membership)
    exact = -1j * (hamiltonian @ local - local @ hamiltonian)
    moments2 = [(p[a] @ q[b] + q[b] @ p[a]) / 2.0 for a, b in operator.harmonic_indices]
    moments3 = [
        (p[a] @ q[b] @ q[c] + q[b] @ q[c] @ p[a]) / 2.0
        for a, b, c in operator.cubic_indices
    ]
    reconstructed = sparse.csr_matrix(exact.shape, dtype=complex)
    harmonic = reconstructed.copy()
    for coefficient, moment in zip(operator.harmonic_coefficients, moments2):
        harmonic += coefficient * moment
    reconstructed += harmonic
    for coefficient, moment in zip(operator.cubic_coefficients, moments3):
        reconstructed += coefficient * moment

    # [q,p]=i fails on the top truncated oscillator state. Only compare the
    # occupation <=2 block, with four empty levels above every occupied mode.
    low = np.array([np.ravel_multi_index(state, (7, 7, 7)) for state in product(range(3), repeat=3)])
    exact_low = exact[low][:, low].toarray()
    result_low = reconstructed[low][:, low].toarray()
    np.testing.assert_allclose(result_low, exact_low, rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(result_low, result_low.conj().T, atol=1e-14)
    assert np.max(np.abs(exact_low - harmonic[low][:, low].toarray())) > 0.05
    assert np.max(np.abs(result_low)) > 0.05

    rng = np.random.default_rng(7359)
    state = np.zeros(exact.shape[0], dtype=complex)
    state[low] = rng.normal(size=low.size) + 1j * rng.normal(size=low.size)
    state /= np.linalg.norm(state)
    expected = np.vdot(state, exact @ state)
    assert abs(expected.imag) < 1e-13
    result = operator.evaluate_moments(
        np.array([np.vdot(state, moment @ state).real for moment in moments2]),
        np.array([np.vdot(state, moment @ state).real for moment in moments3]),
    )
    np.testing.assert_allclose(result["total_solver"], expected.real, atol=1e-13)
    _assert_scope(operator, result, "symmetrized-moments")


def test_fpu_pair_has_equal_leg_current_with_correct_sign_and_physical_scale():
    alpha = 0.6
    device, lead, vertex = fpu_alpha_chain(2, alpha=alpha)
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, temperature=0.8),
        right=HarmonicReservoir(lead, temperature=0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.1, n_positive=3),
        scale=SolverScale(hbar=2.0, omega0=3.0),
    )
    operator = build_cubic_cut_current_operator(problem, 1)
    q, p = np.array([0.2, 0.8]), np.array([0.7, -0.3])
    result = operator.evaluate_phase_space(q, p)
    # H_A3=-alpha*q0*(q1-q0)^2/6; J3 follows its Hamiltonian time derivative.
    expected = alpha * (q[1] - q[0]) * (q[0] * p[1] - q[1] * p[0]) / 3.0
    np.testing.assert_allclose(result["cubic_solver"], expected, atol=1e-14)
    assert result["cubic_solver"] < 0.0
    for component in ("harmonic", "cubic", "total"):
        np.testing.assert_allclose(result[component], result[component + "_solver"] * 18.0)
    reversed_result = operator.evaluate_phase_space(q, -p)
    np.testing.assert_allclose(reversed_result["total"], -result["total"])


@pytest.mark.parametrize("intralayer", [False, True])
def test_zero_or_intralayer_vertex_has_no_cubic_cut_current(intralayer):
    terms = [((0, 0, 1), 0.7), ((2, 3, 3), -0.3)] if intralayer else []
    problem = _problem(_tensor(4, terms), npl=2)
    operator = build_cubic_cut_current_operator(problem, 1)
    assert operator.cubic_indices.shape == (0, 3)
    assert operator.cubic_coefficients.shape == (0,)
    q, p = np.array([0.2, -0.5, 0.4, 0.8]), np.array([0.6, 0.1, -0.7, 0.3])
    result = operator.evaluate_phase_space(q, p)
    assert result["cubic_solver"] == 0.0
    expected = _classical_reference(q, p, _dynamical(problem), np.zeros((4, 4, 4)), np.array([1., 1., 0., 0.]))
    np.testing.assert_allclose(result["total_solver"], expected, atol=1e-14)


def test_sparse_dense_parity_without_dense_conversion_and_arrays_are_readonly(monkeypatch):
    values = _tensor(3, [((0, 0, 2), 0.7), ((0, 1, 2), -0.4)])
    dense_problem = _problem(values)
    sparse_problem = _problem(values, sparse_vertex=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("cut operator must not densify the sparse cubic vertex")

    monkeypatch.setattr(SparseCubicVertex, "to_dense", forbidden)
    for cut in (1, 2):
        dense_operator = build_cubic_cut_current_operator(dense_problem, cut)
        sparse_operator = build_cubic_cut_current_operator(sparse_problem, cut)
        for field in ("harmonic_indices", "harmonic_coefficients", "cubic_indices", "cubic_coefficients"):
            expected = getattr(dense_operator, field)
            actual = getattr(sparse_operator, field)
            np.testing.assert_array_equal(actual, expected)
            assert not actual.flags.writeable
            with pytest.raises(ValueError):
                actual.flat[0] = 0


def test_every_ordered_coefficient_contributes_without_mutating_vertex():
    values = _tensor(3, [((0, 1, 2), 0.7)])
    # Allowed numerical permutation differences must enter the Hamiltonian,
    # rather than being replaced by a selected representative coefficient.
    values[0, 1, 2] += 5e-11
    problem = _problem(values, sparse_vertex=True)
    original = problem.vertex.coefficients.copy()
    operator = build_cubic_cut_current_operator(problem, 1)
    q, p = np.array([0.4, -0.7, 0.8]), np.array([0.9, 0.5, -0.3])
    actual = operator.evaluate_phase_space(q, p)["total_solver"]
    expected = _classical_reference(q, p, _dynamical(problem), values, np.array([1., 0., 0.]))
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-16)
    np.testing.assert_array_equal(problem.vertex.coefficients, original)


@pytest.mark.parametrize("kind", ["missing", "wrong-shape", "nan", "inf", "complex"])
def test_missing_malformed_or_complex_moments_are_rejected(kind):
    operator = build_cubic_cut_current_operator(_problem(_tensor(3, [((0, 1, 2), 0.4)])), 1)
    harmonic = np.zeros(operator.harmonic_coefficients.size)
    cubic = np.zeros(operator.cubic_coefficients.size)

    def invalid(reference):
        if kind == "missing":
            return None
        if kind == "wrong-shape":
            return np.zeros(reference.size + 1)
        if kind == "complex":
            return reference.astype(complex)
        return np.full(reference.shape, np.nan if kind == "nan" else np.inf)

    with pytest.raises((TypeError, ValueError)):
        operator.evaluate_moments(invalid(harmonic), cubic)
    with pytest.raises((TypeError, ValueError)):
        operator.evaluate_moments(harmonic, invalid(cubic))


def test_harmonic_green_moments_reproduce_existing_bond_current():
    problem = replace(
        _problem(np.zeros((3, 3, 3)), scale=SolverScale(hbar=2.0, omega0=3.0)),
        grid=SignedFrequencyGrid(step=0.05, n_positive=50),
    )
    left, right = build_contact_spectra(problem, eta_lead=0.02)
    retarded = solve_retarded(problem, left.retarded, right.retarded)
    green = solve_keldysh_green(
        retarded, left.lesser + right.lesser, left.greater + right.greater,
    )
    reference = bond_heat_currents(problem, green)
    assert np.min(reference) > 0.0
    for cut in (1, 2):
        operator = build_cubic_cut_current_operator(problem, cut)
        two = operator.harmonic_moments_from_green(problem.grid, green)
        result = operator.evaluate_moments(two, np.empty(0))
        np.testing.assert_allclose(result["total"], reference[cut - 1], rtol=1e-12, atol=1e-13)
        assert result["cubic"] == 0.0
        _assert_scope(operator, result, "symmetrized-moments")


@pytest.mark.parametrize("cut", [0, 3, -1, True, 1.5])
def test_invalid_cut_is_rejected(cut):
    with pytest.raises((TypeError, ValueError)):
        build_cubic_cut_current_operator(_problem(np.zeros((3, 3, 3))), cut)


def test_complex_harmonic_matrix_requires_a_different_canonical_coordinate_model():
    problem = _problem(np.zeros((3, 3, 3)))
    couplings = [coupling.astype(complex) for coupling in problem.device.coupling_blocks]
    couplings[0][0, 0] += 1e-30j
    problem = replace(problem, device=replace(problem.device, coupling_blocks=couplings))
    with pytest.raises((TypeError, ValueError)):
        build_cubic_cut_current_operator(problem, 1)
