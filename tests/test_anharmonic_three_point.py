"""Small independent Fourier, static-force and moment-assembly checks."""

from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic import (
    DenseCubicVertex, KeldyshSpectrum, SignedFrequencyGrid, SparseCubicVertex,
    combine_cubic_moments, connected_cubic_moments, correlations_from_green,
    static_cubic_shift,
)


def _vertex():
    values = np.zeros((2, 2, 2))
    values[0, 0, 0] = 0.2
    for key in set(permutations((0, 1, 1))):
        values[key] = -0.4
    return DenseCubicVertex(values)


def _lines(lag):
    basis = np.array([[0.8, -0.6], [0.6, 0.8]])
    frequency = np.array([0.9, 1.5])
    occupation = np.array([0.2, 0.5])
    minus = np.exp(-1j * lag[:, None] * frequency)
    plus = minus.conj()
    c = ((occupation + 1) * minus + occupation * plus) / (2 * frequency)
    p = -0.5j * ((occupation + 1) * minus - occupation * plus)
    return (np.einsum("ia,ta,ja->tij", basis, c, basis),
            np.einsum("ia,ta,ja->tij", basis, p, basis))


def test_fourier_reconstruction_of_discrete_harmonic_modes():
    grid = SignedFrequencyGrid(step=0.3, n_positive=5)
    basis = np.array([[0.8, -0.6], [0.6, 0.8]])
    frequency = [0.9, 1.5]
    occupation = [0.2, 0.5]
    greater = np.zeros((grid.n_points, 2, 2), complex)
    for mode, offset in enumerate((3, 5)):
        projector = np.outer(basis[:, mode], basis[:, mode])
        greater[grid.zero_index + offset] = -1j * (occupation[mode] + 1) * projector / (
            2 * frequency[mode] * grid.integration_weight)
        greater[grid.zero_index - offset] = -1j * occupation[mode] * projector / (
            2 * frequency[mode] * grid.integration_weight)
    green = KeldyshSpectrum(retarded=np.zeros_like(greater), lesser=greater[::-1], greater=greater)
    lag = np.linspace(0, 2.1, 95)
    report = correlations_from_green(grid, green, lag)
    expected, derivative = _lines(lag)
    np.testing.assert_allclose(report["displacement_correlation"], expected, atol=4e-16)
    np.testing.assert_allclose(report["velocity_displacement_correlation"], derivative, atol=4e-16)
    assert not report["frequency_accuracy_verified"]
    with pytest.raises(ValueError, match="recurrence"):
        correlations_from_green(grid, green, [0, np.pi / grid.step])


def test_sparse_path_preserves_all_ordered_coefficients_without_dense(monkeypatch):
    vertex = _vertex()
    values = vertex.values.copy()
    values[0, 1, 1] += 1e-14
    dense = DenseCubicVertex(values)
    sparse = SparseCubicVertex.from_dense(dense)
    monkeypatch.setattr(SparseCubicVertex, "to_dense", lambda self: pytest.fail("densified"))
    lag = np.linspace(0, 1.7, 401)
    c, p = _lines(lag)
    indices = np.array([[0, 1, 1], [1, 0, 1], [0, 0, 0]])
    expected = []
    for i, j, k in indices:
        kernel = np.zeros(len(lag), complex)
        for a, b, d in np.ndindex(values.shape):
            for u, v, w in permutations((a, b, d)):
                kernel += values[a, b, d] / 6 * p[:, i, u] * c[:, j, v] * c[:, k, w]
        expected.append(np.trapezoid(2 * kernel.imag, lag))
    for v in (dense, sparse):
        result = connected_cubic_moments(v, indices, lag, c, p)
        np.testing.assert_allclose(result["connected_moments"], expected, atol=2e-16)
        assert not result["spatial_validation_passed"]
        assert not result["disconnected_means_included"]
    np.testing.assert_array_equal(dense.values, values)


def test_stationary_derivative_endpoint_identity_and_time_refinement():
    lag = np.linspace(0, 1.7, 1601)
    c, p = _lines(lag)
    indices = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    vertex = _vertex()
    fine = connected_cubic_moments(vertex, indices, lag, c, p)["connected_moments"]
    coarse = connected_cubic_moments(vertex, indices, lag[::2], c[::2], p[::2])["connected_moments"]
    def endpoint(t):
        return 2 * np.einsum("abc,a,b,c", vertex.values, c[t, 0], c[t, 1], c[t, 1]).imag
    target = endpoint(-1) - endpoint(0)
    assert abs(np.sum(fine) - target) < 3e-7
    assert abs(np.sum(fine) - target) < abs(np.sum(coarse) - target) / 3.9


def test_disconnected_terms_include_velocity_and_repeated_displacements():
    mu = np.array([0.3, -0.2])
    pi = np.array([0.05, 0.08])
    f = np.array([[1.2, 0.4], [0.4, 0.7]])
    q = np.array([[0.1, 0.2], [-0.2, 0.3]])
    indices = np.array([[0, 1, 1], [1, 0, 1]])
    result = combine_cubic_moments(indices, [0.2, -0.1], displacement_mean=mu,
                                  velocity_mean=pi, displacement_covariance=f,
                                  velocity_displacement_covariance=q)
    expected = np.array([0.05 * 0.7 + 2 * -0.2 * 0.2 + 0.05 * 0.04,
                         0.08 * 0.4 + 0.3 * 0.3 + -0.2 * -0.2 + 0.08 * -0.06])
    np.testing.assert_allclose(result["disconnected_moments"], expected)
    np.testing.assert_allclose(result["total_moments"], [0.2, -0.1] + expected)
    with pytest.raises(TypeError):
        combine_cubic_moments(indices, [0.2, -0.1])


def test_static_shift_matches_independent_gaussian_potential_derivative():
    vertex = _vertex()
    f = np.array([[1.1, 0.3], [0.3, 0.8]])
    stiffness = np.array([[1.8, -0.2], [-0.2, 1.3]])
    # <H3(mu+x)> is directly evaluated from Gaussian Wick algebra; differentiate
    # the polynomial, not the helper's force coefficients.
    def energy(mu):
        value = 0j
        for a, b, c in np.ndindex(vertex.values.shape):
            value += vertex.values[a, b, c] / 6 * (
                mu[a] * mu[b] * mu[c] + mu[a] * f[b, c]
                + mu[b] * f[a, c] + mu[c] * f[a, b])
        return value
    expected = np.array([-energy(1e-25j * np.eye(2)[i]).imag / 1e-25 for i in range(2)])
    result = static_cubic_shift(vertex, f, stiffness)
    np.testing.assert_allclose(result["mean_force"], expected, atol=2e-17)
    np.testing.assert_allclose(stiffness @ result["displacement_mean"], expected, atol=2e-17)
    np.testing.assert_allclose(result["induced_stiffness"],
                               np.einsum("ijk,k->ij", vertex.values, result["displacement_mean"]),
                               atol=2e-17)
    assert not result["harmonic_feedback_included"]
    for bad in (np.zeros((2, 2)), np.diag([1.0, -0.1])):
        with pytest.raises(ValueError, match="positive definite"):
            static_cubic_shift(vertex, f, bad)


@pytest.mark.parametrize("lag", ([0.1, 0.2], [0, 0], [0, -1], [0, np.nan], [0]))
def test_rejects_invalid_lag_windows(lag):
    c = np.zeros((len(lag), 2, 2), complex)
    with pytest.raises(ValueError):
        connected_cubic_moments(_vertex(), np.array([[0, 1, 1]]), lag, c, c)


def test_empty_vertex_and_empty_targets():
    lag = np.array([0, 1])
    c, p = _lines(lag)
    zero = DenseCubicVertex(np.zeros((2, 2, 2)))
    result = connected_cubic_moments(zero, np.array([[0, 1, 1]]), lag, c, p)
    np.testing.assert_array_equal(result["connected_moments"], [0])
    result = connected_cubic_moments(_vertex(), np.empty((0, 3), dtype=int), lag, c, p)
    assert result["connected_moments"].shape == (0,)
