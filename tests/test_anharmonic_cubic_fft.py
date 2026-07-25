from __future__ import annotations

import itertools

import numpy as np
import pytest

from negfpy.anharmonic.cubic import (
    FFT_ACTIVATION_MINIMUM_SPEEDUP,
    cubic_lesser_greater,
    resolved_worker_count,
)
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.types import DenseCubicVertex


def _symmetric_vertex(
    rng: np.random.Generator,
    ndof: int,
) -> DenseCubicVertex:
    raw = rng.normal(size=(ndof, ndof, ndof))
    values = np.zeros_like(raw)
    for permutation in itertools.permutations(range(3)):
        values += raw.transpose(permutation)
    return DenseCubicVertex(values / 6.0)


@pytest.mark.parametrize("ndof", [1, 2, 3])
def test_fft_cubic_kernel_matches_direct_zero_extended_oracle(
    ndof: int,
) -> None:
    rng = np.random.default_rng(22190 + ndof)
    grid = SignedFrequencyGrid(step=0.17, n_positive=7)
    vertex = _symmetric_vertex(rng, ndof)
    shape = (grid.n_points, ndof, ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    direct = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="serial",
    )
    fft = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="fft",
    )

    for fft_field, direct_field in zip(fft, direct, strict=True):
        np.testing.assert_allclose(
            fft_field,
            direct_field,
            rtol=1.0e-10,
            atol=1.0e-12,
        )


@pytest.mark.parametrize("edge_index", [0, -1])
def test_fft_cubic_kernel_does_not_wrap_cutoff_impulse(
    edge_index: int,
) -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=4)
    vertex = DenseCubicVertex(np.ones((1, 1, 1)))
    green = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    green[edge_index, 0, 0] = 1.0

    fft, _ = cubic_lesser_greater(
        vertex,
        green,
        green,
        grid,
        backend="fft",
    )

    # The only pair lies at +/-2*cutoff, outside the returned signed grid.
    # Circular convolution would put a finite signal back inside this array.
    assert np.max(np.abs(fft)) < 1.0e-14


@pytest.mark.nightly
def test_fft_activation_scale_matches_direct_fields() -> None:
    """Exercise the Nw>=513 scale used for the recorded activation profile.

    Timing is intentionally not asserted in CI.  The implementation was
    enabled after a controlled Nw=513, D=3 profile exceeded the required 2x
    speedup; this test keeps the corresponding field-accuracy gate.
    """

    rng = np.random.default_rng(20260724)
    grid = SignedFrequencyGrid(step=0.02, n_positive=256)
    vertex = _symmetric_vertex(rng, ndof=3)
    shape = (grid.n_points, vertex.ndof, vertex.ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    direct = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="serial",
    )
    fft = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend="fft",
    )

    assert FFT_ACTIVATION_MINIMUM_SPEEDUP == 2.0
    assert resolved_worker_count("fft", None) == 1
    for fft_field, direct_field in zip(fft, direct, strict=True):
        np.testing.assert_allclose(
            fft_field,
            direct_field,
            rtol=1.0e-10,
            atol=1.0e-12,
        )
