"""New FFT coupled-q bubbles checked against direct finite-grid contractions."""

from __future__ import annotations

import numpy as np
import pytest
from test_anharmonic_transverse_q_reference import (
    _cells,
    _fourier_vertex,
    _unitary_basis,
)

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
)


def _polynomial_vertex(shape):
    """A fresh translation-invariant real cubic polynomial with two coordinates."""

    cells = _cells(shape)
    lookup = {cell: number for number, cell in enumerate(cells)}
    dimension = 2 * len(cells)
    real = np.zeros((dimension,) * 3)

    def index(cell, shift, coordinate):
        translated = tuple((a + b) % n for a, b, n in zip(cell, shift, shape))
        return 2 * lookup[translated] + coordinate

    for cell in cells:
        first = np.zeros(dimension)
        first[index(cell, (0, 0, 0), 0)] += 0.8
        first[index(cell, (1, 0, 0), 1)] -= 0.5
        first[index(cell, (0, 1, 0), 1)] += 0.17
        second = np.zeros(dimension)
        second[index(cell, (0, 0, 0), 1)] += 0.6
        second[index(cell, (-1, 0, 0), 0)] += 0.31
        second[index(cell, (0, 1, 0), 0)] -= 0.19
        real += 0.23 * np.einsum("i,j,k->ijk", first, first, first)
        real -= 0.11 * np.einsum("i,j,k->ijk", second, second, second)
    mesh = UniformBZMesh(shape)
    coefficients = _fourier_vertex(shape, real, _unitary_basis(shape, 2))
    return TransverseCubicVertex(mesh, coefficients)


def _greens(nq, nw, dimension, *, seed):
    rng = np.random.default_rng(seed)
    shape = (nq, nw, dimension, dimension)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = 0.7 * rng.normal(size=shape) + 1.3j * rng.normal(size=shape)
    return lesser, greater


def _local_scalar_vertex(nq, coefficient=0.43):
    return TransverseCubicVertex(
        UniformBZMesh((nq, 1, 1)),
        np.full((nq, nq, 1, 1, 1), coefficient),
    )


@pytest.mark.parametrize("shape", [(1, 1, 1), (4, 1, 1), (2, 2, 1)])
def test_fft_matches_direct_on_new_complex_polynomial_and_unequal_fields(shape):
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    vertex = _polynomial_vertex(shape)
    grid = SignedFrequencyGrid(step=0.19, n_positive=4)
    greens = _greens(vertex.mesh.n_qpoints, grid.n_points, 2, seed=9127 + shape[1])
    before = [field.copy() for field in greens]
    vertex_before = vertex.coefficients.copy()
    for field in greens:
        field.setflags(write=False)
    direct = cubic_born_self_energy_q(grid, vertex, *greens)
    fft = cubic_born_self_energy_q_fft(grid, vertex, *greens)
    for actual, expected in zip(fft, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
    assert not np.allclose(fft[0], fft[1])
    if shape == (4, 1, 1):
        assert np.max(np.abs(vertex.coefficients.imag)) > 1e-3
    for field, original in zip(greens, before, strict=True):
        np.testing.assert_array_equal(field, original)
    np.testing.assert_array_equal(vertex.coefficients, vertex_before)


@pytest.mark.parametrize("edge", [0, -1])
def test_fft_edge_impulses_do_not_wrap_back_into_signed_band(edge):
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    grid = SignedFrequencyGrid(step=0.31, n_positive=3)
    lesser = np.zeros((4, grid.n_points, 1, 1), complex)
    greater = np.zeros_like(lesser)
    lesser[1, edge, 0, 0] = 1 + 0.2j
    greater[2, edge, 0, 0] = -0.6 + 0.5j
    for field in cubic_born_self_energy_q_fft(
        grid, _local_scalar_vertex(4), lesser, greater
    ):
        np.testing.assert_allclose(field, 0.0, rtol=0.0, atol=1e-14)


def test_opposite_edge_impulses_have_exact_mesh_weight_and_frequency_alignment():
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    grid = SignedFrequencyGrid(step=0.31, n_positive=3)
    vertex = _local_scalar_vertex(4)
    lesser = np.zeros((4, grid.n_points, 1, 1), complex)
    greater = np.zeros_like(lesser)
    first, second = 2 + 0.7j, -0.4 + 0.5j
    lesser[1, -1, 0, 0] = first
    lesser[3, 0, 0, 0] = second
    greater[2, -1, 0, 0] = -0.8 + 0.3j
    actual_lesser, actual_greater = cubic_born_self_energy_q_fft(
        grid, vertex, lesser, greater
    )
    expected = np.zeros_like(lesser)
    expected[0, grid.zero_index, 0, 0] = (
        0.5j * grid.integration_weight / 4 * 0.43**2 * 2 * first * second
    )
    np.testing.assert_allclose(actual_lesser, expected, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(actual_greater, 0.0, rtol=0.0, atol=1e-14)


def test_fft_vertex_scale_is_quadratic_for_complex_q_couplings():
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    vertex = _polynomial_vertex((4, 1, 1))
    grid = SignedFrequencyGrid(step=0.21, n_positive=1)
    greens = _greens(4, grid.n_points, 2, seed=9981)
    strength = -0.37
    scaled = TransverseCubicVertex(vertex.mesh, strength * vertex.coefficients)
    result = cubic_born_self_energy_q_fft(grid, vertex, *greens)
    result_scaled = cubic_born_self_energy_q_fft(grid, scaled, *greens)
    for actual, unscaled in zip(result_scaled, result, strict=True):
        np.testing.assert_allclose(
            actual, strength**2 * unscaled, rtol=3e-12, atol=1e-15
        )


def test_active_compression_preserves_nonordered_embedding_and_exact_zero_rows():
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    reduced = _polynomial_vertex((4, 1, 1))
    active = np.array([4, 1])
    raw = np.zeros((4, 4, 5, 5, 5), complex)
    raw[np.ix_(np.arange(4), np.arange(4), active, active, active)] = (
        reduced.coefficients
    )
    vertex = TransverseCubicVertex(reduced.mesh, raw)
    grid = SignedFrequencyGrid(step=0.23, n_positive=2)
    greens = _greens(4, grid.n_points, 5, seed=18842)
    direct = cubic_born_self_energy_q(grid, vertex, *greens)
    # Full D=5 FFT scratch exceeds this cap; the exact two-coordinate support
    # fits, as does each complete embedded output (8000 bytes).
    fft = cubic_born_self_energy_q_fft(grid, vertex, *greens, max_bytes=32 * 1024)
    inactive = np.array([0, 2, 3])
    for actual, expected in zip(fft, direct, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=3e-14)
        np.testing.assert_array_equal(actual[:, :, inactive, :], 0.0)
        np.testing.assert_array_equal(actual[:, :, :, inactive], 0.0)


def _forbid_fft_and_output_allocation(monkeypatch):
    import negfpy.anharmonic.transverse_fft as module

    def forbidden_fft(*args, **kwargs):
        pytest.fail("An FFT ran before input/resource rejection.")

    def guard(original):
        def instrumented(shape, *args, **kwargs):
            if isinstance(shape, (tuple, list)) and len(shape) == 4:
                pytest.fail(
                    "Output allocation occurred before input/resource rejection."
                )
            return original(shape, *args, **kwargs)

        return instrumented

    monkeypatch.setattr(module.np.fft, "fft", forbidden_fft)
    monkeypatch.setattr(module.np.fft, "ifft", forbidden_fft)
    monkeypatch.setattr(module.np, "zeros", guard(module.np.zeros))
    monkeypatch.setattr(module.np, "empty", guard(module.np.empty))
    return module


@pytest.mark.parametrize("budget", [{"max_pair_evaluations": 1}, {"max_bytes": 1}])
def test_fft_resource_guards_run_before_fft_or_output_allocation(budget, monkeypatch):
    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    vertex = _polynomial_vertex((4, 1, 1))
    greens = _greens(4, grid.n_points, 2, seed=8391)
    module = _forbid_fft_and_output_allocation(monkeypatch)
    with pytest.raises(ValueError, match="max_pair_evaluations|max_bytes"):
        module.cubic_born_self_energy_q_fft(grid, vertex, *greens, **budget)


@pytest.mark.parametrize("invalid", ["shape", "nonfinite"])
def test_fft_rejects_invalid_green_before_transform(invalid, monkeypatch):
    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    vertex = _polynomial_vertex((4, 1, 1))
    lesser, greater = _greens(4, grid.n_points, 2, seed=817)
    if invalid == "shape":
        lesser = lesser[0]
    else:
        greater[1, 2, 0, 1] = np.nan
    module = _forbid_fft_and_output_allocation(monkeypatch)
    with pytest.raises(ValueError, match="shape|finite"):
        module.cubic_born_self_energy_q_fft(grid, vertex, lesser, greater)


def test_all_zero_vertex_returns_exact_zeros_without_fft(monkeypatch):
    import negfpy.anharmonic.transverse_fft as module

    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    vertex = TransverseCubicVertex(UniformBZMesh((4, 1, 1)), np.zeros((4, 4, 2, 2, 2)))
    greens = _greens(4, grid.n_points, 2, seed=12371)

    def forbidden_fft(*args, **kwargs):
        pytest.fail("A zero vertex must not invoke an FFT.")

    monkeypatch.setattr(module.np.fft, "fft", forbidden_fft)
    monkeypatch.setattr(module.np.fft, "ifft", forbidden_fft)
    for field in module.cubic_born_self_energy_q_fft(
        grid, vertex, *greens, max_pair_evaluations=1
    ):
        np.testing.assert_array_equal(field, np.zeros_like(greens[0]))


def test_active_compression_never_thresholds_a_small_nonzero_coefficient():
    from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft

    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    values = np.zeros((1, 1, 3, 3, 3))
    values[0, 0, 2, 2, 2] = 1e-40
    vertex = TransverseCubicVertex(UniformBZMesh((1, 1, 1)), values)
    green = np.ones((1, grid.n_points, 3, 3), complex)
    expected = 0.5j * grid.integration_weight * 1e-80 * np.array([2.0, 3.0, 2.0])
    lesser, greater = cubic_born_self_energy_q_fft(grid, vertex, green, 2 * green)
    np.testing.assert_allclose(lesser[0, :, 2, 2], expected, rtol=3e-13, atol=0.0)
    np.testing.assert_allclose(greater[0, :, 2, 2], 4 * expected, rtol=3e-13, atol=0.0)
    for field in (lesser, greater):
        np.testing.assert_array_equal(field[:, :, :2], 0.0)
        np.testing.assert_array_equal(field[:, :, :, :2], 0.0)
