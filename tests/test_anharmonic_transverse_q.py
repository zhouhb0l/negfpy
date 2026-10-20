from itertools import permutations

import numpy as np
import pytest

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_q import (
    TransverseCubicVertex,
    cubic_born_self_energy_q,
    difference_q_indices,
)
from negfpy.anharmonic.types import DenseCubicVertex


def local_vertex(n_q, value=0.3):
    mesh = UniformBZMesh((n_q, 1, 1))
    return TransverseCubicVertex(mesh, np.full((n_q, n_q, 1, 1, 1), value))


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 1), (4, 1, 1), (2, 2, 1)])
def test_exact_difference_routes_and_input_mesh_are_unchanged(shape):
    mesh = UniformBZMesh(shape)
    routes = difference_q_indices(mesh)
    for q, address in enumerate(mesh.addresses):
        for p, internal in enumerate(mesh.addresses):
            # Independent residue enumeration, including centered Nyquist indices.
            delta = tuple((a - b) % m for a, b, m in zip(address, internal, shape))
            expected = delta[0] + shape[0] * (delta[1] + shape[1] * delta[2])
            assert routes[q, p] == expected
            triplet = mesh.triplets_at_q(mesh.negative_index(q))[p]
            assert triplet.q1_index == p and triplet.q2_index == expected
    assert mesh.shape == shape
    assert not routes.flags.writeable


@pytest.mark.parametrize("n_q", [1, 2, 4])
def test_mesh_normalization_recovers_same_single_cell_answer(n_q):
    rng = np.random.default_rng(772)
    grid = SignedFrequencyGrid(step=0.2, n_positive=2)
    shape = (grid.n_points, 1, 1)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = 0.6 * lesser + 0.4j
    expected = cubic_lesser_greater(DenseCubicVertex(np.full((1, 1, 1), 0.3)),
                                   lesser, greater, grid, backend="serial")
    pair = [np.repeat(g[None], n_q, axis=0) for g in (lesser, greater)]
    result = cubic_born_self_energy_q(grid, local_vertex(n_q), *pair)
    for field, reference in zip(result, expected):
        for q in range(n_q):
            np.testing.assert_allclose(field[q], reference, rtol=1e-12, atol=1e-14)


def test_q_scattering_uses_other_q_values_and_known_frequency_convolution():
    grid = SignedFrequencyGrid(step=0.3, n_positive=2)
    vertex = local_vertex(2, value=1.)
    green = np.zeros((2, 5, 1, 1), complex)
    green[1, :, 0, 0] = [1, 2, 3, 5, 7]
    sigma, _ = cubic_born_self_energy_q(grid, vertex, green, green)
    # qout=0 receives p=1, q-p=1, although G(qout=0) vanishes.
    expected = 0.5j * grid.integration_weight / 2 * np.convolve(
        green[1, :, 0, 0], green[1, :, 0, 0]
    )[2:7]
    np.testing.assert_allclose(sigma[0, :, 0, 0], expected, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(sigma[1], 0.)


@pytest.mark.parametrize("edge", [0, -1])
def test_frequency_cutoff_is_zero_extended_not_wrapped(edge):
    grid = SignedFrequencyGrid(step=1., n_positive=2)
    green = np.zeros((4, 5, 1, 1), complex)
    green[1, edge, 0, 0] = 1.
    for result in cubic_born_self_energy_q(grid, local_vertex(4), green, green):
        np.testing.assert_array_equal(result, 0.)


def test_vertex_scaling_is_quadratic_and_inputs_are_unchanged():
    grid = SignedFrequencyGrid(step=0.2, n_positive=1)
    green = np.arange(6).reshape(2, 3, 1, 1).astype(complex) * (1 + 1j)
    before = green.copy()
    raw = np.full((2, 2, 1, 1, 1), 0.3)
    vertex = TransverseCubicVertex(UniformBZMesh((2, 1, 1)), raw)
    raw[:] = 100.  # The vertex owns its copy.
    before_vertex = vertex.coefficients.copy()
    a = cubic_born_self_energy_q(grid, vertex, green, green)
    b = cubic_born_self_energy_q(grid, local_vertex(2, value=-0.6), green, green)
    for first, second in zip(a, b):
        np.testing.assert_allclose(second, 4 * first, rtol=1e-13, atol=1e-14)
    np.testing.assert_array_equal(green, before)
    np.testing.assert_array_equal(vertex.coefficients, before_vertex)
    assert not vertex.coefficients.flags.writeable


def test_bosonic_reflection_pairs_opposite_q_and_preserves_antihermiticity_psd():
    mesh = UniformBZMesh((4, 1, 1))
    grid = SignedFrequencyGrid(step=0.3, n_positive=1)
    rng = np.random.default_rng(608)
    raw = rng.normal(size=(2, 2, 2))
    base = sum(raw.transpose(p) for p in permutations(range(3))) / 6
    origins = np.array([0, 1])
    values = np.empty((4, 4, 2, 2, 2), complex)
    routes = difference_q_indices(mesh)
    angles = 2 * np.pi * np.arange(4) / 4
    for q in range(4):
        for p in range(4):
            phase = (angles[q] * origins[:, None, None]
                     - angles[p] * origins[None, :, None]
                     - angles[routes[q, p]] * origins[None, None, :])
            values[q, p] = base * np.exp(1j * phase)
    vertex = TransverseCubicVertex(mesh, values)
    random = rng.normal(size=(4, 3, 2, 2)) + 1j * rng.normal(size=(4, 3, 2, 2))
    greater = -1j * (random @ random.conj().swapaxes(-1, -2))
    lesser = np.array([greater[mesh.negative_index(q), ::-1].swapaxes(-1, -2)
                       for q in range(4)])
    assert not np.allclose(lesser[1, ::-1], greater[1].swapaxes(-1, -2))
    sl, sg = cubic_born_self_energy_q(grid, vertex, lesser, greater)
    for q in range(4):
        np.testing.assert_allclose(sl[q, ::-1], sg[mesh.negative_index(q)].swapaxes(-1, -2),
                                   rtol=1e-12, atol=1e-13)
    for field in (sl, sg):
        np.testing.assert_allclose(field + field.conj().swapaxes(-1, -2), 0, atol=1e-13)
        assert np.linalg.eigvalsh(1j * field).min() > -1e-12


@pytest.mark.parametrize("shape", [(2, 1, 1, 1, 1), (2, 2, 2, 1, 1), (2, 2, 0, 0, 0)])
def test_rejects_bad_vertex_shape(shape):
    with pytest.raises(ValueError, match="shape"):
        TransverseCubicVertex(UniformBZMesh((2, 1, 1)), np.zeros(shape))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j])
def test_rejects_nonfinite_or_nonreal_realspace_vertex(value):
    with pytest.raises(ValueError, match="finite|reality"):
        TransverseCubicVertex(UniformBZMesh((1, 1, 1)), np.full((1, 1, 1, 1, 1), value))


def test_rejects_permutation_asymmetry_without_repair():
    raw = np.zeros((1, 1, 2, 2, 2))
    raw[0, 0, 0, 1, 0] = 1.
    before = raw.copy()
    with pytest.raises(ValueError, match="leg exchange"):
        TransverseCubicVertex(UniformBZMesh((1, 1, 1)), raw)
    np.testing.assert_array_equal(raw, before)


def test_limits_reject_work_and_allocations_before_contractions():
    grid = SignedFrequencyGrid(step=1., n_positive=1)
    vertex = local_vertex(2)
    with pytest.raises(ValueError, match="max_pair_evaluations"):
        cubic_born_self_energy_q(grid, vertex, None, None, max_pair_evaluations=27)
    with pytest.raises(ValueError, match="max_output_bytes"):
        cubic_born_self_energy_q(grid, vertex, None, None, max_output_bytes=1)
    with pytest.raises(ValueError, match="max_bytes"):
        difference_q_indices(vertex.mesh, max_bytes=1)
    with pytest.raises(ValueError, match="max_bytes"):
        TransverseCubicVertex(vertex.mesh, vertex.coefficients, max_bytes=1)


@pytest.mark.parametrize("bad", [True, 1.5, 0, -1])
def test_rejects_invalid_work_limits(bad):
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        cubic_born_self_energy_q(SignedFrequencyGrid(1., 1), local_vertex(1),
                                 None, None, max_pair_evaluations=bad)


def test_rejects_wrong_green_shape_and_nonfinite_values():
    grid = SignedFrequencyGrid(1., 1)
    vertex = local_vertex(2)
    good = np.ones((2, 3, 1, 1), complex)
    with pytest.raises(ValueError, match="shape"):
        cubic_born_self_energy_q(grid, vertex, good[0], good)
    bad = good.copy()
    bad[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        cubic_born_self_energy_q(grid, vertex, good, bad)


def test_transverse_axes_and_mesh_type_are_explicit():
    with pytest.raises(TypeError, match="UniformBZMesh"):
        difference_q_indices((2, 1, 1))
    with pytest.raises(ValueError, match="shape"):
        difference_q_indices(UniformBZMesh((1, 1, 2)))
