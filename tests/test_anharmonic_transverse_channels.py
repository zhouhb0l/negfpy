"""Independent discrete references for raw statistical/spectral arithmetic."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channels import (
    StatisticalSpectral,
    cubic_born_channels_packed_fft,
    mix_channels,
    plan_born_channels_fft_dimensions,
    retarded_from_spectral,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex


def _vertex(nq=3, dimension=2):
    """A real-space rank-two potential gives complex valid reduced vertices."""
    mesh = UniformBZMesh((nq, 1, 1))
    factors = np.empty((2, nq, dimension), complex)
    for rank, q, a in product(range(2), range(nq), range(dimension)):
        factors[rank, q, a] = (
            (rank + 1) * (a + 2) / 7
            + (a - rank + 1) / 11 * np.exp(2j * np.pi * q / nq)
        )
    coefficients = np.zeros((nq, nq, dimension, dimension, dimension), complex)
    for q, p, a, c, d in product(
        range(nq), range(nq), range(dimension), range(dimension), range(dimension)
    ):
        for rank, strength in enumerate((0.7, -0.23)):
            coefficients[q, p, a, c, d] += (
                strength * factors[rank, (-q) % nq, a]
                * factors[rank, p, c] * factors[rank, (q - p) % nq, d]
            )
    return PackedTransverseCubicVertex(
        mesh, coefficients, np.arange(dimension), max(1, dimension)
    )


def _direct_contours(grid, vertex, lesser, greater):
    """Literal finite sum, with no production routing or contraction helper."""
    nq, nw, size, _ = lesser.shape
    output = [np.zeros_like(lesser), np.zeros_like(greater)]
    prefactor = 0.5j * grid.step / (2 * np.pi * nq)
    for q, p, w, internal in product(range(nq), range(nq), range(nw), range(nw)):
        partner_frequency = w - internal + grid.zero_index
        if not 0 <= partner_frequency < nw:
            continue
        partner_q = (q - p) % nq
        for a, b, c, d, e, f in product(range(size), repeat=6):
            coefficient = (
                prefactor * vertex.coefficients[q, p, a, c, d]
                * vertex.coefficients[q, p, b, e, f].conjugate()
            )
            for source, target in zip((lesser, greater), output, strict=True):
                target[q, w, a, b] += (
                    coefficient * source[p, internal, c, e]
                    * source[partner_q, partner_frequency, d, f]
                )
    return output


def _raw_channels(nq, nw, size, seed=19271):
    rng = np.random.default_rng(seed)
    shape = (nq, nw, size, size)
    return StatisticalSpectral(*[
        rng.normal(size=shape) + 1j * rng.normal(size=shape) for _ in range(2)
    ])


@pytest.mark.parametrize("nq,positive,workers", [(3, 1, 1), (4, 2, 2)])
def test_arbitrary_raw_channels_match_literal_contour_sum(nq, positive, workers):
    grid, vertex = SignedFrequencyGrid(0.173, positive), _vertex(nq)
    channels = _raw_channels(nq, grid.n_points, 2)
    before = [channels.statistical.copy(), channels.spectral.copy()]
    for value in (channels.statistical, channels.spectral):
        value.setflags(write=False)
    lesser = channels.statistical - 0.5 * channels.spectral
    greater = channels.statistical + 0.5 * channels.spectral
    reference_less, reference_greater = _direct_contours(grid, vertex, lesser, greater)
    result = cubic_born_channels_packed_fft(
        grid, vertex, channels, workers=workers, external_tile=1, bin_tile=3
    )
    np.testing.assert_allclose(
        result.statistical, (reference_greater + reference_less) / 2,
        rtol=3e-12, atol=2e-14,
    )
    np.testing.assert_allclose(
        result.spectral, reference_greater - reference_less, rtol=3e-12, atol=2e-14
    )
    # These deliberately nonphysical inputs must not acquire perfect structure.
    assert np.max(np.abs(result.spectral + result.spectral.conj().swapaxes(-1, -2))) > 1e-5
    for actual, original in zip((channels.statistical, channels.spectral), before, strict=True):
        np.testing.assert_array_equal(actual, original)


def test_frequency_edge_zero_extension_and_q_nyquist_route():
    grid = SignedFrequencyGrid(0.21, 2)
    vertex = PackedTransverseCubicVertex(
        UniformBZMesh((4, 1, 1)), np.ones((4, 4, 1, 1, 1)), [0], 1
    )
    c = np.zeros((4, 5, 1, 1), complex)
    d = np.zeros_like(c)
    c[2, -1, 0, 0] = 2
    d[3, 0, 0, 0] = 3
    result = cubic_born_channels_packed_fft(grid, vertex, StatisticalSpectral(c, d))
    expected = np.zeros_like(c)
    # Frequencies +2 and -2 add to zero; q=2+3 wraps to q=1.
    expected[1, 2, 0, 0] = 0.5j * grid.integration_weight / 4 * 12
    np.testing.assert_allclose(result.spectral, expected, atol=2e-17, rtol=1e-14)
    # CC and DD lie at +/-4, outside the supplied interval, without wraparound.
    np.testing.assert_allclose(result.statistical, 0, atol=2e-17)


def _physical_channels(nq, nw):
    rng = np.random.default_rng(4121)
    c, d = (np.empty((nq, nw, 2, 2), complex) for _ in range(2))
    consumed = set()
    for q, w in product(range(nq), range(nw)):
        if (q, w) in consumed:
            continue
        opposite = ((-q) % nq, nw - 1 - w)
        if opposite == (q, w):
            c[q, w] = 1j * np.array([[2.0, 0.4], [0.4, 3.0]])
            d[q, w] = np.array([[0.0, 0.7], [-0.7, 0.0]])
        else:
            raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            c[q, w] = raw - raw.conj().T
            raw = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            d[q, w] = raw - raw.conj().T
            c[opposite] = c[q, w].T
            d[opposite] = -d[q, w].T
        consumed.update(((q, w), opposite))
    return StatisticalSpectral(c, d)


def test_physical_conjugate_relations_and_nonzero_gamma_zero_spectral_input():
    grid, vertex = SignedFrequencyGrid(0.18, 2), _vertex(4)
    channels = _physical_channels(4, 5)
    assert channels.spectral[0, grid.zero_index, 0, 1] == 0.7
    result = cubic_born_channels_packed_fft(grid, vertex, channels, workers=2)
    for value in (result.statistical, result.spectral):
        np.testing.assert_allclose(value, -value.conj().swapaxes(-1, -2), atol=1e-14)
    for q, w in product(range(4), range(5)):
        partner = ((-q) % 4, 4 - w)
        np.testing.assert_allclose(result.statistical[q, w], result.statistical[partner].T, atol=1e-14)
        np.testing.assert_allclose(result.spectral[q, w], -result.spectral[partner].T, atol=1e-14)


def test_independent_spectral_channel_survives_lost_exported_contour_difference():
    grid = SignedFrequencyGrid(0.125, 1)
    vertex = PackedTransverseCubicVertex(
        UniformBZMesh((1, 1, 1)), np.ones((1, 1, 1, 1, 1)), [0], 1
    )
    c = np.zeros((1, 3, 1, 1), complex)
    d = np.zeros_like(c)
    c[0, 1, 0, 0], d[0, 1, 0, 0] = -1j * 2**56, -3j
    np.testing.assert_array_equal(c + d / 2, c - d / 2)
    result = cubic_born_channels_packed_fft(grid, vertex, StatisticalSpectral(c, d))
    # Integer arithmetic fixes the exact bilinear numerator before the scalar weight.
    expected_d = 0.5j * grid.integration_weight * (-6 * 2**56)
    assert result.spectral[0, 1, 0, 0] == pytest.approx(expected_d, rel=2e-15)
    assert abs(result.spectral[0, 1, 0, 0]) > 1e15


def test_parallel_bins_match_serial_without_modifying_input():
    grid, vertex = SignedFrequencyGrid(0.11, 2), _vertex(3)
    channels = _raw_channels(3, 5, 2, seed=97)
    serial = cubic_born_channels_packed_fft(grid, vertex, channels, workers=1, bin_tile=2)
    parallel = cubic_born_channels_packed_fft(grid, vertex, channels, workers=3, bin_tile=1)
    np.testing.assert_array_equal(serial.statistical, parallel.statistical)
    np.testing.assert_array_equal(serial.spectral, parallel.spectral)


@pytest.mark.parametrize("alpha", [0.0, 0.1, 1.0])
def test_mixing_keeps_channels_independent_and_inputs_immutable(alpha):
    old, new = _raw_channels(2, 3, 2, 17), _raw_channels(2, 3, 2, 18)
    old.statistical[...] *= 1e20
    new.statistical[...] *= 1e20
    originals = [v.copy() for v in (old.statistical, old.spectral, new.statistical, new.spectral)]
    result = mix_channels(old, new, alpha)
    for actual, first, second in zip(
        (result.statistical, result.spectral),
        (old.statistical, old.spectral), (new.statistical, new.spectral), strict=True,
    ):
        np.testing.assert_array_equal(actual, (1 - alpha) * first + alpha * second)
        assert not np.shares_memory(actual, first)
        assert not np.shares_memory(actual, second)
    assert np.max(np.abs(result.spectral)) > 0.1
    for value, before in zip(
        (old.statistical, old.spectral, new.statistical, new.spectral), originals, strict=True
    ):
        np.testing.assert_array_equal(value, before)


@pytest.mark.parametrize("backend,with_q", [("direct", False), ("fft", False), ("fft", True)])
def test_retarded_reconstruction_matches_independent_finite_pv(backend, with_q):
    grid = SignedFrequencyGrid(0.13, 2)
    spectrum = _raw_channels(2, 5, 2, 403).spectral
    if not with_q:
        spectrum = spectrum[0]
    original = spectrum.copy()
    expected = 0.5 * spectrum
    source_q = spectrum if with_q else spectrum[None]
    expected_q = expected if with_q else expected[None]
    for q, w, j, a, b in product(range(len(source_q)), range(5), range(5), range(2), range(2)):
        if w != j:
            expected_q[q, w, a, b] += 1j * source_q[q, j, a, b] / (2 * np.pi * (w - j))
    actual = retarded_from_spectral(spectrum, grid, hilbert_backend=backend)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-15)
    np.testing.assert_array_equal(spectrum, original)
    # Generic raw spectral data are not projected to make the identity pass.
    assert np.max(np.abs(actual - actual.conj().swapaxes(-1, -2) - spectrum)) > 0.1


def test_count_plan_accounts_for_four_terms_and_two_transforms():
    plan = plan_born_channels_fft_dimensions(4, 601, 120, workers=8, external_tile=16, bin_tile=1)
    assert plan.n_fft == 2048
    assert plan.contraction_work == 12 * 4**2 * 2048 * 120**4
    assert plan.transformed_channel_bytes == 2 * 16 * 4 * 2048 * 120**2
    assert plan.output_bytes == 2 * 16 * 4 * 601 * 120**2
    assert plan.peak_owned_bytes > plan.transformed_channel_bytes + plan.output_bytes


@pytest.mark.parametrize("keyword", ["max_bytes", "max_contraction_work"])
def test_born_budget_failure_precedes_fft(monkeypatch, keyword):
    grid, vertex = SignedFrequencyGrid(0.13, 1), _vertex(1)
    channels = _raw_channels(1, 3, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("FFT must not run before the resource gate")
    monkeypatch.setattr(np.fft, "fft", forbidden)
    with pytest.raises(ValueError, match="exceeds"):
        cubic_born_channels_packed_fft(grid, vertex, channels, **{keyword: 1})


def test_bad_shape_and_nonfinite_raw_input_fail_before_fft(monkeypatch):
    grid, vertex = SignedFrequencyGrid(0.13, 1), _vertex(1)
    channels = _raw_channels(1, 3, 2)
    def forbidden(*args, **kwargs):
        raise AssertionError("FFT must not run for invalid input")
    monkeypatch.setattr(np.fft, "fft", forbidden)
    with pytest.raises(ValueError, match="shape"):
        cubic_born_channels_packed_fft(grid, vertex, StatisticalSpectral(channels.statistical, channels.spectral[:, :2]))
    channels.spectral[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cubic_born_channels_packed_fft(grid, vertex, channels)


def test_mixing_and_retarded_bounds_do_not_mutate_input():
    grid, channels = SignedFrequencyGrid(0.13, 1), _raw_channels(1, 3, 2)
    original = channels.spectral.copy()
    with pytest.raises(ValueError, match="exceed"):
        mix_channels(channels, channels, 0.1, max_bytes=1)
    with pytest.raises(ValueError, match="exceed"):
        retarded_from_spectral(channels.spectral, grid, max_bytes=1)
    with pytest.raises(ValueError, match="mixing"):
        mix_channels(channels, channels, 1.1)
    np.testing.assert_array_equal(channels.spectral, original)


def test_empty_active_support_has_zero_work_and_empty_output():
    grid = SignedFrequencyGrid(0.14, 1)
    vertex = PackedTransverseCubicVertex(
        UniformBZMesh((2, 1, 1)), np.empty((2, 2, 0, 0, 0)), [], 3
    )
    empty = np.empty((2, 3, 0, 0), complex)
    result = cubic_born_channels_packed_fft(grid, vertex, StatisticalSpectral(empty, empty))
    assert result.statistical.shape == result.spectral.shape == empty.shape
    assert plan_born_channels_fft_dimensions(2, 3, 0).contraction_work == 0
    assert retarded_from_spectral(empty, grid).shape == empty.shape
