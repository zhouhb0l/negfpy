"""New packed/tiled execution checked against independent discrete references."""

from __future__ import annotations

import numpy as np
import pytest
from test_anharmonic_transverse_fft import _polynomial_vertex

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_fft import cubic_born_self_energy_q_fft
from negfpy.anharmonic.transverse_packed import (
    PackedTransverseCubicVertex,
    contract_packed_fft_bins,
    cubic_born_self_energy_q_packed_fft,
    embed_packed_self_energy,
    plan_packed_born_fft_dimensions,
    project_transverse_green,
)
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
)


def _embedded(shape):
    reduced = _polynomial_vertex(shape)
    nq = reduced.mesh.n_qpoints
    indices = np.array([5, 1])
    full = np.zeros((nq, nq, 6, 6, 6), dtype=complex)
    full[np.ix_(np.arange(nq), np.arange(nq), indices, indices, indices)] = reduced.coefficients
    return reduced, indices, TransverseCubicVertex(reduced.mesh, full)


@pytest.mark.parametrize("shape,workers", [((1, 1, 1), 1), ((4, 1, 1), 2), ((2, 2, 1), 3)])
def test_new_packed_tiled_fields_match_full_direct_and_fft(shape, workers):
    reduced, indices, dense = _embedded(shape)
    packed = PackedTransverseCubicVertex(reduced.mesh, reduced.coefficients, indices, 6)
    grid = SignedFrequencyGrid(step=0.173, n_positive=3)
    rng = np.random.default_rng(21948 + workers)
    full_shape = (reduced.mesh.n_qpoints, grid.n_points, 6, 6)
    greens = [rng.normal(size=full_shape) + 1j*rng.normal(size=full_shape) for _ in range(2)]
    before = [v.copy() for v in greens]
    for value in greens:
        value.setflags(write=False)
    projected = [project_transverse_green(packed, value) for value in greens]
    direct = cubic_born_self_energy_q(grid, dense, *greens)
    old_fft = cubic_born_self_energy_q_fft(grid, dense, *greens)
    new = cubic_born_self_energy_q_packed_fft(
        grid, packed, *projected, workers=workers, external_tile=1, bin_tile=3,
    )
    inactive = [0, 2, 3, 4]
    for actual, reference, fft in zip(new, direct, old_fft, strict=True):
        embedded = embed_packed_self_energy(packed, actual)
        np.testing.assert_allclose(embedded, reference, rtol=5e-12, atol=4e-14)
        np.testing.assert_allclose(embedded, fft, rtol=5e-12, atol=4e-14)
        np.testing.assert_array_equal(embedded[:, :, inactive, :], 0)
        np.testing.assert_array_equal(embedded[:, :, :, inactive], 0)
    for value, original in zip(greens, before, strict=True):
        np.testing.assert_array_equal(value, original)


def test_exact_dense_packing_handles_nonordered_embedding_and_tiny_support():
    reduced, _, dense = _embedded((4, 1, 1))
    raw = dense.coefficients.copy()
    raw[:, :, 3, 3, 3] = 1e-40
    dense = TransverseCubicVertex(reduced.mesh, raw)
    packed = PackedTransverseCubicVertex.from_dense(dense)
    np.testing.assert_array_equal(packed.active_dofs, [1, 3, 5])
    np.testing.assert_array_equal(packed.coefficients[:, :, 1, 1, 1], 1e-40)
    indices = np.asarray(packed.active_dofs)
    np.testing.assert_array_equal(
        packed.coefficients,
        raw[np.ix_(np.arange(4), np.arange(4), indices, indices, indices)],
    )
    assert not packed.coefficients.flags.writeable


@pytest.mark.parametrize("workers,external_tile,bin_tile", [(1, 1, 2), (3, 3, 2)])
def test_complex_slice_matches_independent_factorized_polynomial_reference(workers, external_tile, bin_tile):
    rng = np.random.default_rng(39071)
    dimension, rank, bins = 5, 3, 5
    factors = rng.normal(size=(rank, dimension)) + 0.23j*rng.normal(size=(rank, dimension))
    strengths = rng.normal(size=rank) + 0.31j*rng.normal(size=rank)
    vertex = np.einsum("r,ra,rc,rd->acd", strengths, factors, factors, factors)
    left = rng.normal(size=(bins, dimension, dimension)) + 1j*rng.normal(size=(bins, dimension, dimension))
    right = rng.normal(size=left.shape) + 1j*rng.normal(size=left.shape)
    originals = [v.copy() for v in [vertex, left, right]]
    expected = np.zeros_like(left)
    for k in range(bins):
        for r in range(rank):
            for s in range(rank):
                amplitude = strengths[r] * strengths[s].conjugate()
                amplitude *= factors[r] @ left[k] @ factors[s].conj()
                amplitude *= factors[r] @ right[k] @ factors[s].conj()
                expected[k] += amplitude*np.outer(factors[r], factors[s].conj())
    actual = contract_packed_fft_bins(
        vertex, left, right, workers=workers, external_tile=external_tile, bin_tile=bin_tile,
    )
    np.testing.assert_allclose(actual, expected, rtol=4e-12, atol=2e-10)
    for value, original in zip([vertex, left, right], originals, strict=True):
        np.testing.assert_array_equal(value, original)


def test_packed_frequency_endpoints_preserve_zero_extension_and_mesh_weight():
    grid = SignedFrequencyGrid(step=0.127, n_positive=4)
    packed = PackedTransverseCubicVertex(
        UniformBZMesh((4, 1, 1)), np.full((4,4,1,1,1), -0.29), [7], 9,
    )
    lesser = np.zeros((4, grid.n_points, 1, 1), complex)
    greater = np.zeros_like(lesser)
    first, second = 0.71+0.2j, -0.42+0.17j
    lesser[1, 0, 0, 0], lesser[3, -1, 0, 0] = first, second
    greater[2, -1, 0, 0] = 0.23-0.7j
    actual = cubic_born_self_energy_q_packed_fft(grid, packed, lesser, greater, workers=2, bin_tile=3)
    expected = np.zeros_like(lesser)
    expected[0, grid.zero_index, 0, 0] = 0.5j*grid.integration_weight/4*(-0.29)**2*2*first*second
    np.testing.assert_allclose(actual[0], expected, rtol=2e-12, atol=1e-16)
    np.testing.assert_allclose(actual[1], 0, rtol=0, atol=1e-16)


def test_count_only_material_plan_retains_dimension_cost_and_packed_payload():
    plan = plan_packed_born_fft_dimensions(4, 601, 120, external_tile=16, workers=8, bin_tile=1)
    assert plan.n_fft == 2048
    assert plan.vertex_bytes == 16*4**2*120**3 + 8*120
    assert plan.output_bytes == 2*16*4*601*120**2
    assert plan.contraction_work == 6*4**2*2048*120**4
    assert plan.peak_owned_bytes >= plan.output_bytes + plan.transformed_green_bytes
    assert plan.vertex_bytes < 16*4**2*216**3
