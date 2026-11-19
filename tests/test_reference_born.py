"""Small synthetic checks only; no material or saved checkpoint arrays.

The literal oracle independently uses X-fast componentwise q residues, finite
frequency sums and an explicit four-index vertex contraction. Finite
differences below check the *development* tangent; execution never uses them.
"""

from itertools import product

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channels import (
    StatisticalSpectral,
    cubic_born_channels_packed_fft,
    plan_born_channels_fft_dimensions,
    retarded_from_spectral,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex

from negfpp.reference.increment_born import (
    anchored_born,
    born_bilinear,
    born_increment,
    difference_born,
    plan_born_expansion_dimensions,
    tangent_born,
)


def _vertex(shape=(3, 2, 1), size=2):
    """Real-space rank potential, complex on both nontrivial mesh axes."""
    nx, ny, _ = shape
    nq = nx * ny
    factors = np.empty((2, nq, size), complex)
    for rank, q, a in product(range(2), range(nq), range(size)):
        x, y = q % nx, q // nx
        factors[rank, q, a] = (
            (rank + a + 2) / 9
            + (a - rank + 1) / 17 * np.exp(2j * np.pi * x / nx)
            + (rank + 2) / 21 * np.exp(2j * np.pi * y / ny)
            + (a + 1) / 31 * np.exp(2j * np.pi * (x / nx + y / ny))
        )
    coefficients = np.zeros((nq, nq, size, size, size), complex)
    for q, p, a, c, d in product(range(nq), range(nq), range(size), range(size), range(size)):
        qx, qy, px, py = q % nx, q // nx, p % nx, p // nx
        negq = (-qx) % nx + nx * ((-qy) % ny)
        partner = (qx - px) % nx + nx * ((qy - py) % ny)
        for rank, strength in enumerate((0.7, -0.23)):
            coefficients[q, p, a, c, d] += (
                strength * factors[rank, negq, a]
                * factors[rank, p, c] * factors[rank, partner, d]
            )
    return PackedTransverseCubicVertex(
        UniformBZMesh(shape), coefficients, np.arange(size), max(1, size),
    )


def _raw(vertex, grid, seed):
    rng = np.random.default_rng(seed)
    shape = (vertex.mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
    return StatisticalSpectral(*[
        rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2)
    ])


def _values(channels):
    return channels.statistical, channels.spectral


def _sum(first, second, factor=1):
    return StatisticalSpectral(*[a + factor * b for a, b in zip(_values(first), _values(second))])


def _assert_close(first, second, rtol=4e-12, atol=2e-14):
    for a, b in zip(_values(first), _values(second)):
        np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)


def _literal_bilinear(grid, vertex, first, second):
    nx, ny, _ = vertex.mesh.shape
    nq, nw, a, _ = first.statistical.shape
    c, d = (np.zeros_like(first.statistical) for _ in range(2))
    weight = 0.5j * grid.step / (2 * np.pi * nq)
    for q, p, w, internal in product(range(nq), range(nq), range(nw), range(nw)):
        other = w - internal + grid.zero_index
        if not 0 <= other < nw:
            continue
        partner = (q % nx - p % nx) % nx + nx * ((q // nx - p // nx) % ny)
        v = vertex.coefficients[q, p]
        def contract(left, right):
            return np.einsum("acd,ce,df,bef->ab", v, left, right, v.conj(), optimize=False)
        fc, fd = first.statistical[p, internal], first.spectral[p, internal]
        sc, sd = second.statistical[partner, other], second.spectral[partner, other]
        c[q, w] += weight * (contract(fc, sc) + 0.25 * contract(fd, sd))
        d[q, w] += weight * (contract(fc, sd) + contract(fd, sc))
    return StatisticalSpectral(c, d)


@pytest.mark.parametrize("meshshape,workers", [((3, 2, 1), 1), ((2, 3, 1), 2)])
def test_bilinear_matches_independent_two_axis_oracle_and_canonical(meshshape, workers):
    grid, vertex = SignedFrequencyGrid(0.173, 2), _vertex(meshshape)
    first, second = _raw(vertex, grid, 17), _raw(vertex, grid, 31)
    originals = [a.copy() for x in (first, second) for a in _values(x)]
    for x in (first, second):
        for a in _values(x):
            a.setflags(write=False)
    actual = born_bilinear(grid, vertex, first, second, workers=workers, bin_tile=3, external_tile=1)
    _assert_close(actual, _literal_bilinear(grid, vertex, first, second))
    square = born_bilinear(grid, vertex, first, first, workers=workers, external_tile=1)
    _assert_close(square, cubic_born_channels_packed_fft(grid, vertex, first, workers=workers, external_tile=1))
    assert np.max(abs(vertex.coefficients.imag)) > 1e-4
    assert np.max(abs(actual.spectral + actual.spectral.conj().swapaxes(-1, -2))) > 1e-4
    for original, a in zip(originals, [a for x in (first, second) for a in _values(x)]):
        np.testing.assert_array_equal(original, a)


@pytest.mark.parametrize("scale", [0.001, 0.7])
def test_exact_expansion_retains_all_quadratic_increment_terms(scale):
    grid, vertex = SignedFrequencyGrid(0.13, 1), _vertex()
    reference = _raw(vertex, grid, 318)
    raw = _raw(vertex, grid, 711)
    increment = StatisticalSpectral(*[scale * a for a in _values(raw)])
    expected = cubic_born_channels_packed_fft(grid, vertex, _sum(reference, increment))
    baseline = cubic_born_channels_packed_fft(grid, vertex, reference)
    baseline_before = [a.copy() for a in _values(baseline)]
    _assert_close(anchored_born(grid, vertex, reference, increment), expected)
    _assert_close(anchored_born(grid, vertex, reference, increment, reference_born=baseline), expected)
    delta = born_increment(grid, vertex, reference, increment)
    _assert_close(difference_born(grid, vertex, reference, increment), delta)
    tangent = tangent_born(grid, vertex, reference, increment)
    quadratic = born_bilinear(grid, vertex, increment, increment)
    _assert_close(delta, _sum(tangent, quadratic))
    _assert_close(_sum(baseline, delta), expected)
    assert max(np.max(abs(x)) for x in _values(quadratic)) > 1e-6 * scale**2
    for before, after in zip(baseline_before, _values(baseline)):
        np.testing.assert_array_equal(before, after)
    # Retarded reconstruction is linear in the independent spectral output.
    expected_r = retarded_from_spectral(expected.spectral, grid)
    expansion_r = (
        retarded_from_spectral(baseline.spectral, grid)
        + retarded_from_spectral(delta.spectral, grid)
    )
    np.testing.assert_allclose(expansion_r, expected_r, rtol=4e-12, atol=2e-14)


def test_tangent_is_analytic_derivative_of_full_canonical_quadratic_map():
    grid, vertex = SignedFrequencyGrid(0.11, 2), _vertex()
    base, direction = _raw(vertex, grid, 413), _raw(vertex, grid, 491)
    analytic = tangent_born(grid, vertex, base, direction)
    for epsilon in (1e-3, 1e-4):
        plus = cubic_born_channels_packed_fft(grid, vertex, _sum(base, direction, epsilon))
        minus = cubic_born_channels_packed_fft(grid, vertex, _sum(base, direction, -epsilon))
        centered = StatisticalSpectral(*[(p - m) / (2 * epsilon) for p, m in zip(_values(plus), _values(minus))])
        _assert_close(analytic, centered, rtol=2e-9, atol=2e-11)
    # No use of a physical conjugation/projection can hide an independent C/D term.
    spectral_only = StatisticalSpectral(np.zeros_like(direction.statistical), direction.spectral)
    assert np.linalg.norm(tangent_born(grid, vertex, base, spectral_only).statistical) > 1e-5


def test_small_increment_survives_cancellation_of_two_large_born_results():
    grid = SignedFrequencyGrid(0.2, 1)
    vertex = PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)), np.ones((1, 1, 1, 1, 1)), [0], 1)
    c = np.zeros((1, 3, 1, 1), complex)
    c[0, 1, 0, 0] = 2**56
    h = np.zeros_like(c)
    h[0, 1, 0, 0] = 1
    base = StatisticalSpectral(c, np.zeros_like(c))
    step = StatisticalSpectral(h, np.zeros_like(c))
    np.testing.assert_array_equal(c + h, c)  # Information lost in full-sum representation.
    before = cubic_born_channels_packed_fft(grid, vertex, base)
    after = cubic_born_channels_packed_fft(grid, vertex, _sum(base, step))
    np.testing.assert_array_equal(after.statistical - before.statistical, 0)
    exact_delta = born_increment(grid, vertex, base, step)
    expected = np.zeros_like(c)
    expected[0, 1, 0, 0] = 0.5j * grid.integration_weight * (2 * 2**56 + 1)
    np.testing.assert_allclose(exact_delta.statistical, expected, atol=0.1, rtol=3e-15)
    assert abs(exact_delta.statistical[0, 1, 0, 0]) > 1e15
    _assert_close(difference_born(grid, vertex, base, step), exact_delta, rtol=3e-15, atol=0.1)


def test_second_axis_wrap_and_finite_frequency_edges():
    grid, vertex = SignedFrequencyGrid(0.21, 2), _vertex((3, 2, 1), 1)
    c = np.zeros((6, 5, 1, 1), complex)
    d = np.zeros_like(c)
    c[5, -1, 0, 0] = 2  # residue (2,1), frequency +2
    d[4, 0, 0, 0] = 3   # residue (1,1), frequency -2
    first = StatisticalSpectral(c, np.zeros_like(c))
    second = StatisticalSpectral(np.zeros_like(c), d)
    result = born_bilinear(grid, vertex, first, second)
    expected = np.zeros_like(c)
    # (2,1)+(1,1) wraps independently to Gamma on a 3x2 grid.
    v = vertex.coefficients[0, 5, 0, 0, 0]
    expected[0, 2, 0, 0] = 0.5j * grid.integration_weight / 6 * v * v.conjugate() * 6
    np.testing.assert_allclose(result.spectral, expected, atol=2e-17, rtol=3e-14)
    np.testing.assert_allclose(result.statistical, 0, atol=2e-17)


@pytest.mark.parametrize("function", [born_bilinear, tangent_born, born_increment, difference_born, anchored_born])
@pytest.mark.parametrize("keyword", ["max_bytes", "max_contraction_work"])
def test_resource_failure_precedes_any_fft(monkeypatch, function, keyword):
    grid, vertex = SignedFrequencyGrid(0.1, 1), _vertex((1, 1, 1), 1)
    source = _raw(vertex, grid, 1)
    def forbidden(*args, **kwargs):
        raise AssertionError("No FFT is allowed before input/resource gates.")
    monkeypatch.setattr(np.fft, "fft", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        function(grid, vertex, source, source, **{keyword: 1})


def test_count_plan_bounds_four_transforms_and_aggregate_work():
    base = plan_born_channels_fft_dimensions(8, 601, 132, workers=64)
    for operation, multiple in (("bilinear", 1), ("tangent", 2), ("increment", 3), ("polarized_increment", 2), ("anchored", 4)):
        plan = plan_born_expansion_dimensions(8, 601, 132, operation=operation, workers=64)
        assert plan.n_pairs == multiple
        assert plan.contraction_work == multiple * base.contraction_work
        assert plan.transformed_channel_bytes == 2 * base.transformed_channel_bytes
        assert plan.peak_owned_bytes == base.peak_owned_bytes + base.transformed_channel_bytes
    cached = plan_born_expansion_dimensions(8, 601, 132, operation="anchored", cached_reference=True)
    assert cached.n_pairs == 3
    with pytest.raises(ValueError, match="only valid"):
        plan_born_expansion_dimensions(8, 601, 132, cached_reference=True)


def test_nonfinite_and_wrong_shape_are_rejected_without_mutation():
    grid, vertex = SignedFrequencyGrid(0.1, 1), _vertex((1, 1, 1), 1)
    base, wrong = _raw(vertex, grid, 71), _raw(vertex, grid, 23)
    wrong.spectral[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        born_increment(grid, vertex, base, wrong)
    with pytest.raises(ValueError, match="shape"):
        born_bilinear(grid, vertex, base, StatisticalSpectral(base.statistical[..., :0], base.spectral))


def test_empty_vertex_support_returns_two_independent_empty_outputs():
    grid = SignedFrequencyGrid(0.2, 1)
    vertex = PackedTransverseCubicVertex(UniformBZMesh((2, 2, 1)), np.empty((4, 4, 0, 0, 0)), [], 3)
    value = StatisticalSpectral(np.empty((4, 3, 0, 0), complex), np.empty((4, 3, 0, 0), complex))
    result = anchored_born(grid, vertex, value, value)
    assert result.statistical.shape == (4, 3, 0, 0)
    assert result.statistical is not result.spectral
