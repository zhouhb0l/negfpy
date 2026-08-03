from __future__ import annotations

import itertools

import numpy as np
import pytest

import negfpy.anharmonic.sparse_cubic as sparse_cubic_module
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
    for index in set(itertools.permutations((0, 1, 2))):
        values[index] = 0.25
    values[3, 3, 3] = -0.1
    return DenseCubicVertex(values)


def _green(grid: SignedFrequencyGrid, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(grid.n_points, 4, 4)) + 1j * rng.normal(
        size=(grid.n_points, 4, 4)
    )


def _sparse_orbit_vertex(
    ndof: int,
    orbits: list[tuple[tuple[int, int, int], float]],
) -> SparseCubicVertex:
    entries: dict[tuple[int, int, int], float] = {}
    for representative, coefficient in orbits:
        for index in set(itertools.permutations(representative)):
            entries[index] = coefficient
    ordered = sorted(entries.items())
    indices = np.asarray([index for index, _ in ordered], dtype=np.int64)
    coefficients = np.asarray(
        [coefficient for _, coefficient in ordered],
        dtype=np.float64,
    )
    return SparseCubicVertex(ndof, indices, coefficients)


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
    indices = np.array(sorted(set(itertools.permutations((0, 1, 2)))))
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


def test_sparse_active_basis_matches_dense_with_inactive_device_dofs() -> None:
    grid = SignedFrequencyGrid(step=0.2, n_positive=3)
    sparse = _sparse_orbit_vertex(6, [((1, 3, 5), 0.375)])
    dense = sparse.to_dense(max_bytes=10_000_000)
    lesser = np.random.default_rng(718).normal(
        size=(grid.n_points, 6, 6)
    ) + 1j * np.random.default_rng(719).normal(
        size=(grid.n_points, 6, 6)
    )
    greater = np.random.default_rng(720).normal(
        size=(grid.n_points, 6, 6)
    ) + 1j * np.random.default_rng(721).normal(
        size=(grid.n_points, 6, 6)
    )

    reference = cubic_lesser_greater(
        dense,
        lesser,
        greater,
        grid,
        backend="serial",
    )
    schedule = sparse_cubic_module._compiled_sparse_schedule(sparse)
    assert schedule.active.tolist() == [1, 3, 5]
    assert schedule.flattened_first_legs.shape == (3, 9)

    for backend, rtol in (("direct", 1.0e-12), ("fft", 1.0e-10)):
        observed = cubic_sparse_lesser_greater(
            sparse,
            lesser,
            greater,
            grid,
            backend=backend,
        )
        for actual, expected in zip(observed, reference, strict=True):
            np.testing.assert_allclose(actual, expected, rtol=rtol, atol=1.0e-12)
            np.testing.assert_array_equal(actual[:, (0, 2, 4), :], 0.0)
            np.testing.assert_array_equal(actual[:, :, (0, 2, 4)], 0.0)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_sparse_signed_cutoff_convolution_has_no_circular_wrap(
    backend: str,
) -> None:
    grid = SignedFrequencyGrid(step=1.0, n_positive=2)
    sparse = _sparse_orbit_vertex(1, [((0, 0, 0), 1.0)])
    green = np.zeros((grid.n_points, 1, 1), dtype=np.complex128)
    negative = 1.0 + 2.0j
    zero = -0.75 + 0.5j
    positive = 0.25 - 1.5j
    green[0, 0, 0] = negative
    green[grid.zero_index, 0, 0] = zero
    green[-1, 0, 0] = positive

    expected = np.zeros(grid.n_points, dtype=np.complex128)
    expected[0] = 2.0 * negative * zero
    expected[grid.zero_index] = zero**2 + 2.0 * negative * positive
    expected[-1] = 2.0 * zero * positive
    expected *= 0.5j * grid.integration_weight

    lesser, greater = cubic_sparse_lesser_greater(
        sparse,
        green,
        green,
        grid,
        backend=backend,
    )
    np.testing.assert_allclose(lesser[:, 0, 0], expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(greater[:, 0, 0], expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.parametrize("backend", ["direct", "fft"])
def test_sparse_backend_is_bitwise_deterministic(backend: str) -> None:
    grid = SignedFrequencyGrid(step=0.125, n_positive=3)
    sparse = SparseCubicVertex.from_dense(_vertex())
    lesser = _green(grid, 28)
    greater = _green(grid, 29)

    first = cubic_sparse_lesser_greater(
        sparse,
        lesser,
        greater,
        grid,
        backend=backend,
    )
    second = cubic_sparse_lesser_greater(
        sparse,
        lesser,
        greater,
        grid,
        backend=backend,
    )
    for first_field, second_field in zip(first, second, strict=True):
        np.testing.assert_array_equal(first_field, second_field)


@pytest.mark.parametrize(
    ("backend", "expected_pair_calls"),
    [("direct", 38), ("fft", 32)],
)
def test_schedule_compiles_once_and_frequency_work_matches_model(
    backend: str,
    expected_pair_calls: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=2)
    sparse = _sparse_orbit_vertex(1, [((0, 0, 0), 0.5)])
    lesser = np.ones((grid.n_points, 1, 1), dtype=np.complex128)
    greater = (1.0j * lesser).copy()
    original_compile = sparse_cubic_module._compiled_sparse_schedule
    original_pair = sparse_cubic_module._sparse_pair_contraction
    compile_calls = 0
    pair_schedule_ids: set[int] = set()
    pair_calls = 0

    def counted_compile(vertex: SparseCubicVertex):
        nonlocal compile_calls
        compile_calls += 1
        return original_compile(vertex)

    def counted_pair(vertex, schedule, first, second):
        nonlocal pair_calls
        pair_calls += 1
        pair_schedule_ids.add(id(schedule))
        return original_pair(vertex, schedule, first, second)

    monkeypatch.setattr(
        sparse_cubic_module,
        "_compiled_sparse_schedule",
        counted_compile,
    )
    monkeypatch.setattr(
        sparse_cubic_module,
        "_sparse_pair_contraction",
        counted_pair,
    )

    cubic_sparse_lesser_greater(
        sparse,
        lesser,
        greater,
        grid,
        backend=backend,
    )

    # Nw=5 direct: 3+4+5+4+3=19 pairs per field.  FFT: the
    # zero-padded length is next_power_of_two(2*5-1)=16 per field.
    assert compile_calls == 1
    assert len(pair_schedule_ids) == 1
    assert pair_calls == expected_pair_calls


def test_schedule_memory_and_pair_work_are_sparse_payload_models() -> None:
    # A light D=120 surrogate: four disjoint local three-DOF orbits give the
    # same dimensional pressure as a future device without loading real Si.
    sparse = _sparse_orbit_vertex(
        120,
        [
            ((3 * group, 3 * group + 1, 3 * group + 2), group + 1.0)
            for group in range(4)
        ],
    )
    schedule = sparse_cubic_module._compiled_sparse_schedule(sparse)
    csr = schedule.flattened_first_legs

    assert sparse.nnz == 24
    assert len(schedule.active) == 12
    assert csr.shape == (12, 12**2)
    assert {local.shape for local in schedule.local_matrices} == {(2, 2)}
    assert schedule.reconstructed_first_leg_bytes == 12 * 120**2 * 8
    assert schedule.pair_multiply_add_count == 4_320

    accounted_payload = sum(
        array.nbytes
        for array in (
            schedule.active,
            *schedule.row_indices,
            *schedule.column_indices,
            *schedule.local_matrices,
            csr.data,
            csr.indices,
            csr.indptr,
        )
    )
    assert schedule.numeric_storage_bytes == accounted_payload
    assert schedule.numeric_storage_bytes < schedule.reconstructed_first_leg_bytes // 500
    for array in (schedule.active, csr.data, csr.indices, csr.indptr):
        assert not array.flags.writeable


def test_dense_memory_guard_precedes_allocation() -> None:
    sparse = SparseCubicVertex.from_dense(_vertex())
    with pytest.raises(MemoryError, match="max_bytes"):
        sparse.to_dense(max_bytes=1)
    with pytest.raises(TypeError, match="max_bytes"):
        sparse.to_dense(max_bytes=True)
