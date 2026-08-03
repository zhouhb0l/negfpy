from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.sparse_cubic import (
    SparseCubicVertex,
    cubic_sparse_lesser_greater,
    cubic_vertex_physics_sha256,
)
from negfpy.anharmonic.types import DenseCubicVertex


def _vertex() -> DenseCubicVertex:
    values = np.zeros((4, 4, 4))
    for index in set(__import__("itertools").permutations((0, 1, 2))):
        values[index] = 0.25
    values[3, 3, 3] = -0.1
    return DenseCubicVertex(values)


def _green(grid: SignedFrequencyGrid, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(grid.n_points, 4, 4)) + 1j * rng.normal(
        size=(grid.n_points, 4, 4)
    )


def test_sparse_vertex_roundtrip_and_physics_identity() -> None:
    dense = _vertex()
    sparse = SparseCubicVertex.from_dense(dense)
    assert sparse.nnz == 7
    assert sparse.ndof == 4
    assert sparse.support_density == 7 / 64
    assert sparse.max_abs == 0.25
    assert not sparse.is_zero
    assert cubic_vertex_physics_sha256(dense) == sparse.cubic_physics_sha256
    rebuilt = sparse.to_dense(max_bytes=10_000_000)
    np.testing.assert_array_equal(rebuilt.values, dense.values)


def test_sparse_vertex_rejects_missing_or_unequal_permutation() -> None:
    with pytest.raises(ValueError, match="missing"):
        SparseCubicVertex(3, [[0, 1, 2]], [1.0])
    indices = np.array(sorted(set(__import__("itertools").permutations((0, 1, 2)))))
    values = np.ones(6)
    values[-1] = np.nextafter(1.0, 2.0)
    with pytest.raises(ValueError, match="bitwise"):
        SparseCubicVertex(3, indices, values)


def test_sparse_dense_direct_and_fft_fields_agree() -> None:
    grid = SignedFrequencyGrid(step=0.125, n_positive=4)
    dense = _vertex()
    sparse = SparseCubicVertex.from_dense(dense)
    lesser = _green(grid, 11)
    greater = _green(grid, 12)
    expected = cubic_lesser_greater(dense, lesser, greater, grid, backend="serial")
    direct = cubic_sparse_lesser_greater(sparse, lesser, greater, grid, backend="direct")
    fft = cubic_sparse_lesser_greater(sparse, lesser, greater, grid, backend="fft")
    for reference, observed_direct, observed_fft in zip(expected, direct, fft, strict=True):
        np.testing.assert_allclose(observed_direct, reference, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(observed_fft, reference, rtol=1e-10, atol=1e-12)


def test_dense_memory_guard_precedes_allocation() -> None:
    sparse = SparseCubicVertex.from_dense(_vertex())
    with pytest.raises(MemoryError, match="max_bytes"):
        sparse.to_dense(max_bytes=1)
    with pytest.raises(TypeError, match="max_bytes"):
        sparse.to_dense(max_bytes=True)
