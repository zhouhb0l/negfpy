"""Independent second-direction momentum and native-channel Born audit.

Only tiny synthetic fields are used. The reference enumerates reciprocal
fractions and conserves momentum by testing all candidate partners; it never
calls the production address, partner, contraction, or FFT helpers. Non-Nyquist
second-direction points prevent q -> -q sign errors from hiding at a 2-point
mesh. This is an arithmetic/routing audit, not a material SCBA stability test.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channels import (
    StatisticalSpectral,
    cubic_born_channels_packed_fft,
    retarded_from_spectral,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_q import difference_q_indices


def _labels(shape):
    """Raw residue representatives, explicitly X-fast, with exact fractions."""
    nx, ny, nz = shape
    return [
        (Fraction(x, nx), Fraction(y, ny), Fraction(z, nz))
        for z in range(nz) for y in range(ny) for x in range(nx)
    ]


def _partner(labels, q, p):
    """Find q-p by independently searching all reciprocal equivalence classes."""
    matches = [
        r for r, label in enumerate(labels)
        if all((a-b-c).denominator == 1 for a, b, c in zip(labels[q], labels[p], label))
    ]
    assert len(matches) == 1
    return matches[0]


def _vertex(shape, dimension=2):
    """An anisotropic real-space rank-two cubic energy, complex in q space."""
    labels = _labels(shape)
    # Real coefficients and two different direction-dependent neighborhoods.
    sites = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, -1, 0))
    weights = np.array([
        [[0.72, -0.13], [0.21, 0.37], [-0.19, 0.48], [0.11, -0.24]],
        [[-0.17, 0.61], [0.33, -0.16], [0.42, 0.08], [-0.29, 0.23]],
    ])[:, :, :dimension]

    def factor(rank, q, coordinate):
        return sum(
            weights[rank, s, coordinate] * np.exp(
                2j*np.pi*float(sum(k*r for k, r in zip(q, site)))
            )
            for s, site in enumerate(sites)
        )

    nq = len(labels)
    coefficients = np.zeros((nq, nq, dimension, dimension, dimension), complex)
    for q, p, a, c, d in product(range(nq), range(nq), *[range(dimension)]*3):
        r = _partner(labels, q, p)
        for rank, strength in enumerate((0.83, -0.31)):
            coefficients[q, p, a, c, d] += (
                strength * factor(rank, tuple(-k for k in labels[q]), a)
                * factor(rank, labels[p], c) * factor(rank, labels[r], d)
            )
    return PackedTransverseCubicVertex(
        UniformBZMesh(shape), coefficients, np.arange(dimension), dimension,
    )


def _channels(nq, nw, size, seed=830214):
    rng = np.random.default_rng(seed)
    shape = (nq, nw, size, size)
    return StatisticalSpectral(*[
        rng.normal(size=shape)+1j*rng.normal(size=shape) for _ in range(2)
    ])


def _literal_four_terms(grid, vertex, channels):
    """Dense scalar loops with separately retained CC, DD/4, CD and DC terms."""
    labels = _labels(vertex.mesh.shape)
    nq, nw, size, _ = channels.statistical.shape
    result = np.zeros((4, nq, nw, size, size), complex)
    c, d = channels.statistical, channels.spectral
    terms = ((c, c, 1.0), (d, d, 0.25), (c, d, 1.0), (d, c, 1.0))
    signed = list(range(-grid.zero_index, grid.zero_index+1))
    # The rectangle frequency measure appears once and the full-q average once.
    weight = 0.5j*grid.step/(2*np.pi*len(labels))
    for q, p, w, left_w, right_w in product(range(nq), range(nq), *[range(nw)]*3):
        if signed[left_w]+signed[right_w] != signed[w]:
            continue
        r = _partner(labels, q, p)
        for a, b, i, j, k, ell in product(range(size), repeat=6):
            coupling = (weight*vertex.coefficients[q, p, a, i, j]
                        * vertex.coefficients[q, p, b, k, ell].conjugate())
            for term, (left, right, factor) in enumerate(terms):
                result[term, q, w, a, b] += (
                    coupling*factor*left[p, left_w, i, k]*right[r, right_w, j, ell]
                )
    return result


@pytest.mark.parametrize("shape", [(2, 2, 1), (2, 3, 1), (4, 2, 1), (1, 5, 1)])
def test_second_q_routes_match_independent_reciprocal_conservation(shape):
    labels = _labels(shape)
    actual = difference_q_indices(UniformBZMesh(shape))
    expected = np.array([
        [_partner(labels, q, p) for p in range(len(labels))] for q in range(len(labels))
    ])
    np.testing.assert_array_equal(actual, expected)
    for row in actual:
        assert sorted(row) == list(range(len(labels)))


@pytest.mark.parametrize("shape,workers", [((2, 2, 1), 1), ((2, 3, 1), 2), ((4, 2, 1), 1)])
def test_anisotropic_two_dimensional_native_born_matches_literal_sum(shape, workers):
    grid, vertex = SignedFrequencyGrid(0.173, 2), _vertex(shape)
    channels = _channels(len(_labels(shape)), grid.n_points, 2)
    saved = [value.copy() for value in (channels.statistical, channels.spectral)]
    terms = _literal_four_terms(grid, vertex, channels)
    actual = cubic_born_channels_packed_fft(
        grid, vertex, channels, workers=workers, external_tile=1, bin_tile=3,
    )
    np.testing.assert_allclose(actual.statistical, terms[0]+terms[1], rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(actual.spectral, terms[2]+terms[3], rtol=3e-12, atol=3e-14)
    for original, value in zip(saved, (channels.statistical, channels.spectral), strict=True):
        np.testing.assert_array_equal(original, value)
    # R is reconstructed separately at every q; compare a literal finite PV sum.
    d = terms[2]+terms[3]
    expected_r = d/2
    for w, internal in product(range(grid.n_points), repeat=2):
        if w != internal:
            expected_r[:, w] += 1j*d[:, internal]/(2*np.pi*(w-internal))
    actual_r = retarded_from_spectral(actual.spectral, grid)
    np.testing.assert_allclose(actual_r, expected_r, rtol=4e-12, atol=4e-14)


@pytest.mark.parametrize("isolated", ["CC", "DD", "CD+DC"])
def test_four_native_terms_keep_their_factors_with_second_q(isolated):
    grid, vertex = SignedFrequencyGrid(0.219, 1), _vertex((2, 3, 1))
    channels = _channels(6, 3, 2, seed=417)
    if isolated == "CC":
        channels.spectral.fill(0)
    elif isolated == "DD":
        channels.statistical.fill(0)
    terms = _literal_four_terms(grid, vertex, channels)
    result = cubic_born_channels_packed_fft(grid, vertex, channels, external_tile=1)
    np.testing.assert_allclose(result.statistical, terms[0]+terms[1], rtol=3e-12, atol=3e-14)
    np.testing.assert_allclose(result.spectral, terms[2]+terms[3], rtol=3e-12, atol=3e-14)
    assert np.linalg.norm(terms[0]+terms[1]) > 1e-5
    if isolated == "CD+DC":
        assert min(np.linalg.norm(terms[2]), np.linalg.norm(terms[3])) > 1e-5


@pytest.mark.parametrize("shape", [(2, 3, 1), (1, 3, 1), (1, 4, 1)])
def test_swapping_both_transverse_axes_is_covariant(shape):
    grid, vertex = SignedFrequencyGrid(0.127, 1), _vertex(shape)
    channels = _channels(len(_labels(shape)), 3, 2)
    swapped_shape = (shape[1], shape[0], 1)
    labels, swapped = _labels(shape), _labels(swapped_shape)
    # Each new index takes the old point with its two transverse coordinates swapped.
    permutation = np.array([labels.index((q[1], q[0], q[2])) for q in swapped])
    relabeled = PackedTransverseCubicVertex(
        UniformBZMesh(swapped_shape),
        vertex.coefficients[permutation][:, permutation], [0, 1], 2,
    )
    relabeled_channels = StatisticalSpectral(
        channels.statistical[permutation], channels.spectral[permutation],
    )
    original = cubic_born_channels_packed_fft(grid, vertex, channels, workers=1)
    transformed = cubic_born_channels_packed_fft(grid, relabeled, relabeled_channels, workers=2)
    for value, expected in zip(
        (transformed.statistical, transformed.spectral),
        (original.statistical[permutation], original.spectral[permutation]), strict=True,
    ):
        np.testing.assert_allclose(value, expected, rtol=3e-12, atol=3e-14)


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 1), (2, 3, 1), (4, 2, 1)])
def test_constant_q_input_is_averaged_once_over_entire_mesh(shape):
    grid = SignedFrequencyGrid(0.19, 1)
    seed = _channels(1, 3, 1)
    nq = len(_labels(shape))
    vertex = PackedTransverseCubicVertex(
        UniformBZMesh(shape), np.ones((nq, nq, 1, 1, 1)), [0], 1,
    )
    channels = StatisticalSpectral(
        np.repeat(seed.statistical, nq, axis=0), np.repeat(seed.spectral, nq, axis=0),
    )
    reference_vertex = PackedTransverseCubicVertex(
        UniformBZMesh((1, 1, 1)), np.ones((1, 1, 1, 1, 1)), [0], 1,
    )
    reference = _literal_four_terms(grid, reference_vertex, seed)
    result = cubic_born_channels_packed_fft(grid, vertex, channels)
    np.testing.assert_allclose(
        result.statistical, np.repeat(reference[0]+reference[1], nq, axis=0), atol=3e-16,
    )
    np.testing.assert_allclose(
        result.spectral, np.repeat(reference[2]+reference[3], nq, axis=0), atol=3e-16,
    )


def test_second_direction_non_nyquist_impulse_wraps_momentum_not_frequency():
    shape, grid = (2, 3, 1), SignedFrequencyGrid(0.21, 2)
    labels, nq = _labels(shape), 6
    vertex = PackedTransverseCubicVertex(
        UniformBZMesh(shape), np.ones((nq, nq, 1, 1, 1)), [0], 1,
    )
    c, d = (np.zeros((nq, 5, 1, 1), complex) for _ in range(2))
    p = labels.index((Fraction(1, 2), Fraction(2, 3), Fraction(0)))
    r = labels.index((Fraction(1, 2), Fraction(2, 3), Fraction(0)))
    q = labels.index((Fraction(0), Fraction(1, 3), Fraction(0)))
    c[p, -1, 0, 0], d[r, 0, 0, 0] = 2, 3
    result = cubic_born_channels_packed_fft(grid, vertex, StatisticalSpectral(c, d))
    expected = np.zeros_like(c)
    # Two ordered cross terms, +2 + (-2) = 0, x: 1/2+1/2 = 0, y: 2/3+2/3 = 1/3.
    # Flattened modulo-Nq would incorrectly carry the x wrap into the y index.
    expected[q, grid.zero_index, 0, 0] = 0.5j*grid.step/(2*np.pi*nq)*12
    np.testing.assert_allclose(result.spectral, expected, atol=3e-17, rtol=1e-14)
    # CC and DD frequencies are outside [-2,2]; no circular wrap is permitted.
    np.testing.assert_allclose(result.statistical, 0, atol=3e-17)
