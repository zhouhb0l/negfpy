"""Equivalence tests for the optional padded-FFT Hilbert transform."""

from __future__ import annotations

import numpy as np

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.retarded import (
    principal_value_cauchy,
    principal_value_cauchy_fft,
    retarded_from_keldysh,
)


def test_fft_principal_value_matches_direct_complex_matrix_spectrum() -> None:
    rng = np.random.default_rng(20260724)
    grid = SignedFrequencyGrid(step=0.037, n_positive=19)
    values = (
        rng.normal(size=(grid.n_points, 3, 3))
        + 1j * rng.normal(size=(grid.n_points, 3, 3))
    )

    direct = principal_value_cauchy(values, grid)
    accelerated = principal_value_cauchy_fft(values, grid)

    np.testing.assert_allclose(
        accelerated,
        direct,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_fft_principal_value_has_no_edge_impulse_wraparound() -> None:
    grid = SignedFrequencyGrid(step=0.29, n_positive=8)
    amplitude = 0.7 - 1.3j
    values = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    values[-1, 0, 0] = amplitude

    accelerated = principal_value_cauchy_fft(values, grid)[:, 0, 0]
    expected = np.zeros(grid.n_points, dtype=np.complex128)
    for output_index in range(grid.n_points - 1):
        lag = output_index - (grid.n_points - 1)
        expected[output_index] = amplitude / (2.0 * np.pi * lag)

    np.testing.assert_allclose(
        accelerated,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert abs(accelerated[-1]) < 1.0e-12


def test_fft_principal_value_matches_direct_at_513_frequencies() -> None:
    """Exercise the first grid size at which FFT profiling is permitted."""

    rng = np.random.default_rng(513)
    grid = SignedFrequencyGrid(step=0.02, n_positive=256)
    values = (
        rng.normal(size=(grid.n_points, 2, 2))
        + 1j * rng.normal(size=(grid.n_points, 2, 2))
    )

    direct = principal_value_cauchy(values, grid)
    accelerated = principal_value_cauchy_fft(values, grid)

    np.testing.assert_allclose(
        accelerated,
        direct,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_retarded_reconstruction_selects_exact_fft_backend() -> None:
    rng = np.random.default_rng(907)
    grid = SignedFrequencyGrid(step=0.11, n_positive=13)
    shape = (grid.n_points, 2, 2)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    direct = retarded_from_keldysh(
        lesser,
        greater,
        grid,
        hilbert_backend="direct",
    )
    accelerated = retarded_from_keldysh(
        lesser,
        greater,
        grid,
        hilbert_backend="fft",
    )
    np.testing.assert_allclose(
        accelerated,
        direct,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
