from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.retarded import (
    principal_value_cauchy,
    retarded_from_keldysh,
)
from negfpy.anharmonic.types import DenseCubicVertex


def _brute_force_bubble(
    psi: np.ndarray,
    green: np.ndarray,
    grid: SignedFrequencyGrid,
) -> np.ndarray:
    """Independent, intentionally literal implementation of Guo Eq. (8)."""

    n_frequency, ndof, _ = green.shape
    result = np.zeros((n_frequency, ndof, ndof), dtype=np.complex128)
    for output_index in range(n_frequency):
        for input_index in range(n_frequency):
            difference_index = (
                output_index - input_index + grid.zero_index
            )
            if difference_index < 0 or difference_index >= n_frequency:
                continue
            for a in range(ndof):
                for b in range(ndof):
                    value = 0.0j
                    for c in range(ndof):
                        for d in range(ndof):
                            for e in range(ndof):
                                for f in range(ndof):
                                    value += (
                                        psi[a, c, d]
                                        * psi[b, e, f]
                                        * green[input_index, c, e]
                                        * green[difference_index, d, f]
                                    )
                    result[output_index, a, b] += value
    return 0.5j * grid.integration_weight * result


def _symmetric_vertex(rng: np.random.Generator, ndof: int) -> np.ndarray:
    raw = rng.normal(size=(ndof, ndof, ndof))
    return sum(
        raw.transpose(permutation)
        for permutation in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    ) / 6.0


def test_dense_cubic_bubble_matches_literal_six_index_oracle() -> None:
    rng = np.random.default_rng(29184)
    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    psi = _symmetric_vertex(rng, ndof=2)
    vertex = DenseCubicVertex(psi)
    green_lesser = (
        rng.normal(size=(grid.n_points, 2, 2))
        + 1j * rng.normal(size=(grid.n_points, 2, 2))
    )
    green_greater = (
        rng.normal(size=(grid.n_points, 2, 2))
        + 1j * rng.normal(size=(grid.n_points, 2, 2))
    )

    actual_lesser, actual_greater = cubic_lesser_greater(
        vertex,
        green_lesser,
        green_greater,
        grid,
    )

    expected_lesser = _brute_force_bubble(
        psi,
        green_lesser,
        grid,
    )
    expected_greater = _brute_force_bubble(
        psi,
        green_greater,
        grid,
    )
    np.testing.assert_allclose(
        actual_lesser,
        expected_lesser,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        actual_greater,
        expected_greater,
        rtol=1e-12,
        atol=1e-12,
    )
    assert actual_lesser.dtype == np.complex128
    assert actual_greater.dtype == np.complex128


def test_cubic_bubble_has_fixed_quadratic_vertex_scaling_and_sign() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=1)
    green = np.arange(1, grid.n_points + 1, dtype=float)[:, None, None]
    vertex = DenseCubicVertex(np.array([[[0.4]]]))
    scaled = DenseCubicVertex(np.array([[[-1.2]]]))

    sigma, _ = cubic_lesser_greater(vertex, green, green, grid)
    sigma_scaled, _ = cubic_lesser_greater(
        scaled,
        green,
        green,
        grid,
    )

    np.testing.assert_allclose(sigma_scaled, 9.0 * sigma)


def test_cubic_bubble_applies_guo_i_over_two_and_rectangle_weight() -> None:
    grid = SignedFrequencyGrid(step=0.5, n_positive=1)
    vertex = DenseCubicVertex(np.array([[[3.0]]]))
    green = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    green[grid.zero_index, 0, 0] = 2.0

    sigma, _ = cubic_lesser_greater(vertex, green, green, grid)

    expected = np.zeros_like(sigma)
    expected[grid.zero_index, 0, 0] = (
        0.5j * (grid.step / (2.0 * np.pi)) * 3.0**2 * 2.0**2
    )
    np.testing.assert_allclose(sigma, expected, rtol=0.0, atol=1e-15)


def test_cubic_convolution_is_zero_extended_not_circular() -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=1)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    green[-1, 0, 0] = 1.0

    sigma, _ = cubic_lesser_greater(vertex, green, green, grid)

    # The only nonzero sum is +1 + +1 = +2, which lies beyond the grid.
    # A circular implementation would spuriously wrap it into an in-grid bin.
    np.testing.assert_array_equal(sigma, np.zeros_like(sigma))


def test_cubic_bubble_rejects_wrong_green_shape() -> None:
    grid = SignedFrequencyGrid(step=0.1, n_positive=2)
    vertex = DenseCubicVertex(np.ones((2, 2, 2)))
    wrong = np.ones((grid.n_points, 1, 1))

    with pytest.raises(ValueError, match="green_lesser must have shape"):
        cubic_lesser_greater(vertex, wrong, wrong, grid)


def test_principal_value_cauchy_matches_literal_discrete_sum() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=3)
    values = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    values[1, 0, 0] = 2.0 - 0.5j
    values[5, 0, 0] = -0.4 + 1.2j

    actual = principal_value_cauchy(values, grid)

    expected = np.zeros_like(values)
    for i, omega in enumerate(grid.values):
        for j, omega_prime in enumerate(grid.values):
            if i != j:
                expected[i] += (
                    grid.integration_weight
                    * values[j]
                    / (omega - omega_prime)
                )
    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_principal_value_cauchy_matches_analytic_harmonic_number_pair() -> None:
    grid = SignedFrequencyGrid(step=0.17, n_positive=8)
    values = np.ones((grid.n_points, 1, 1), dtype=np.complex128)

    actual = principal_value_cauchy(values, grid)[:, 0, 0]
    harmonic = np.zeros(grid.n_points)
    harmonic[1:] = np.cumsum(1.0 / np.arange(1, grid.n_points))
    expected = np.array(
        [
            (harmonic[index] - harmonic[grid.n_points - 1 - index])
            / (2.0 * np.pi)
            for index in range(grid.n_points)
        ]
    )

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-15)


def test_cubic_frequency_convolution_matches_zero_padded_fft_product() -> None:
    rng = np.random.default_rng(8841)
    grid = SignedFrequencyGrid(step=0.13, n_positive=5)
    green = (
        rng.normal(size=grid.n_points)
        + 1j * rng.normal(size=grid.n_points)
    )
    vertex = DenseCubicVertex(np.array([[[0.7]]]))

    sigma, _ = cubic_lesser_greater(
        vertex,
        green[:, None, None],
        green[:, None, None],
        grid,
    )
    padded = 2 * grid.n_points - 1
    convolution = np.fft.ifft(
        np.fft.fft(green, padded) ** 2
    )[:padded]
    expected = (
        0.5j
        * grid.integration_weight
        * vertex.values[0, 0, 0] ** 2
        * convolution[
            grid.zero_index : grid.zero_index + grid.n_points
        ]
    )

    np.testing.assert_allclose(
        sigma[:, 0, 0],
        expected,
        rtol=2.0e-13,
        atol=2.0e-13,
    )


def test_causal_retarded_reconstruction_includes_discrete_pv_term() -> None:
    grid = SignedFrequencyGrid(step=0.4, n_positive=2)
    lesser = np.zeros((grid.n_points, 2, 2), dtype=np.complex128)
    greater = np.zeros_like(lesser)
    greater[1] = np.array([[0.0 + 2.0j, 0.3], [-0.3, 0.0 + 1.0j]])
    greater[3] = np.array([[0.0 - 0.5j, -0.2j], [-0.2j, 0.0 + 0.8j]])
    difference = greater - lesser

    expected = (
        0.5 * difference
        + 1j * principal_value_cauchy(difference, grid)
    )
    actual = retarded_from_keldysh(
        lesser,
        greater,
        grid,
        strategy="causal_hilbert",
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-15, atol=1e-15)


def test_spectral_only_warns_that_it_is_incomplete() -> None:
    grid = SignedFrequencyGrid(step=0.5, n_positive=1)
    lesser = np.full((grid.n_points, 1, 1), -1.0j)
    greater = np.full((grid.n_points, 1, 1), 2.0j)

    with pytest.warns(RuntimeWarning, match="Kramers-Kronig"):
        actual = retarded_from_keldysh(
            lesser,
            greater,
            grid,
            strategy="spectral_only",
        )

    np.testing.assert_array_equal(actual, 0.5 * (greater - lesser))


def test_retarded_reconstruction_rejects_unknown_strategy() -> None:
    grid = SignedFrequencyGrid(step=0.5, n_positive=1)
    values = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)

    with pytest.raises(ValueError, match="strategy must be"):
        retarded_from_keldysh(
            values,
            values,
            grid,
            strategy="fft",  # type: ignore[arg-type]
        )
