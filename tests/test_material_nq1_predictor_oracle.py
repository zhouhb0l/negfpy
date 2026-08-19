from __future__ import annotations

import itertools

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import DenseCubicVertex


def _dense_vertex() -> DenseCubicVertex:
    values = np.zeros((3, 3, 3), dtype=np.float64)
    for index in set(itertools.permutations((0, 1, 2))):
        values[index] = 0.375
    values[1, 1, 1] = -0.2
    return DenseCubicVertex(values)


def _scaled_vertex(
    vertex: DenseCubicVertex | SparseCubicVertex,
    factor: float,
) -> DenseCubicVertex | SparseCubicVertex:
    if isinstance(vertex, DenseCubicVertex):
        return DenseCubicVertex(vertex.values * factor)
    return SparseCubicVertex(
        vertex.ndof,
        vertex.indices,
        vertex.coefficients * factor,
    )


@pytest.mark.parametrize(
    ("representation", "backend", "rtol"),
    (
        ("dense", "serial", 1.0e-12),
        ("dense", "fft", 1.0e-11),
        ("sparse", "serial", 1.0e-12),
        ("sparse", "fft", 1.0e-11),
    ),
)
def test_cubic_bubble_predictor_has_exact_quadratic_coupling_oracle(
    representation: str,
    backend: str,
    rtol: float,
) -> None:
    """The pilot predictor follows the production bubble's two vertices."""

    grid = SignedFrequencyGrid(step=0.125, n_positive=4)
    dense = _dense_vertex()
    vertex: DenseCubicVertex | SparseCubicVertex = (
        dense
        if representation == "dense"
        else SparseCubicVertex.from_dense(dense)
    )
    rng = np.random.default_rng(20260819)
    shape = (grid.n_points, vertex.ndof, vertex.ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    coupling_ratio = 2.5

    baseline = cubic_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
        backend=backend,
    )
    scaled = cubic_lesser_greater(
        _scaled_vertex(vertex, coupling_ratio),
        lesser,
        greater,
        grid,
        backend=backend,
    )

    expected_factor = coupling_ratio**2
    for observed, reference in zip(scaled, baseline, strict=True):
        np.testing.assert_allclose(
            observed,
            expected_factor * reference,
            rtol=rtol,
            atol=1.0e-12,
        )


def test_quadratic_predictor_factor_is_positive_for_signed_vertex_scaling() -> None:
    """A vertex sign change must not change the cubic-bubble self-energy."""

    grid = SignedFrequencyGrid(step=0.25, n_positive=2)
    vertex = _dense_vertex()
    rng = np.random.default_rng(311)
    shape = (grid.n_points, vertex.ndof, vertex.ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)

    positive = cubic_lesser_greater(
        _scaled_vertex(vertex, 3.0),
        lesser,
        greater,
        grid,
        backend="serial",
    )
    negative = cubic_lesser_greater(
        _scaled_vertex(vertex, -3.0),
        lesser,
        greater,
        grid,
        backend="serial",
    )
    for left, right in zip(positive, negative, strict=True):
        np.testing.assert_allclose(left, right, rtol=1.0e-12, atol=1.0e-12)
