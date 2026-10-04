"""Algebraic and physical-invariance checks for the opt-in boundary model."""

import itertools
import json

import numpy as np
import pytest

from negfpy.anharmonic.research_boundary import complete_acoustic_boundary
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex


def sparse(tensor):
    indices = np.argwhere(tensor != 0)
    return SparseCubicVertex(len(tensor), indices, tensor[tuple(indices.T)])


def dense(vertex):
    result = np.zeros((vertex.ndof,) * 3)
    result[tuple(vertex.indices.T)] = vertex.coefficients
    return result


def cube(vector):
    return np.einsum("i,j,k->ijk", vector, vector, vector)


def bond(ndof, first, second, axis=0, masses=None):
    masses = np.ones(ndof // 3) if masses is None else masses
    vector = np.zeros(ndof)
    vector[3 * first + axis] = 1 / np.sqrt(masses[first])
    vector[3 * second + axis] = -1 / np.sqrt(masses[second])
    return cube(vector)


def complete(vertex, masses=None, **kwargs):
    return complete_acoustic_boundary(
        vertex, atom_masses=np.ones(vertex.ndof // 3) if masses is None else masses,
        dof_per_layer=3, boundary_layers=(0, vertex.ndof // 3 - 1), **kwargs)


@pytest.mark.parametrize("masses", [np.ones(3), np.asarray([1., 4., 9.])])
def test_consistent_local_bonds_preserved_with_energy_and_mass_translation(masses):
    tensor = bond(9, 0, 1, masses=masses) + .37 * bond(9, 1, 2, axis=2, masses=masses)
    old = sparse(tensor)
    old_bytes = old.coefficients.tobytes()
    result = complete(old, masses=masses)
    np.testing.assert_allclose(dense(result.vertex), tensor, atol=3e-15, rtol=3e-15)
    assert result.report["completed_translation"]["relative_max"] < 2e-15
    assert result.report["correction"]["total"]["relative_frobenius"] < 3e-15
    assert old.coefficients.tobytes() == old_bytes
    assert result.report["bounded_trial_screen_passed"]
    rng = np.random.default_rng(150)
    q = rng.normal(size=9)
    displacement = np.repeat(np.sqrt(masses), 3) * np.tile([.3, -.7, 1.1], 3)
    energy = lambda values, field: np.einsum("ijk,i,j,k", values, field, field, field) / 6
    assert energy(tensor, q) == pytest.approx(energy(dense(result.vertex), q), rel=1e-13)
    assert energy(tensor, q) == pytest.approx(energy(dense(result.vertex), q + displacement), rel=1e-13)
    assert not result.vertex.metadata["release_certified"]
    json.dumps(result.report, allow_nan=False)


def test_pair_completion_is_independent_dense_minimum_norm_solution():
    tensor = bond(6, 0, 1)
    tensor[0, 0, 0] += .1
    original = sparse(tensor)
    result = complete(original)
    translations = np.zeros((6, 3))
    for axis in range(3):
        translations[axis::3, axis] = 1 / np.sqrt(2.)
    projector = np.eye(6) - translations @ translations.T
    expected = np.einsum("ia,jb,kc,ijk->abc", projector, projector, projector, tensor)
    np.testing.assert_allclose(dense(result.vertex), expected, atol=3e-15)
    np.testing.assert_allclose(dense(result.vertex), 1.0125 * bond(6, 0, 1), atol=3e-15)
    assert result.report["source_translation"]["relative_max"] > .04
    assert result.report["completed_translation"]["relative_max"] < 1e-14
    assert result.report["correction"]["total"]["correction_frobenius"] == pytest.approx(.1 * np.sqrt(7 / 8))


def test_local_support_permutation_and_readonly_input():
    tensor = bond(12, 0, 1) + .8 * bond(12, 1, 2, axis=1) + .3 * bond(12, 2, 3, axis=2)
    tensor[0, 0, 0] += .03
    tensor[11, 11, 11] -= .02
    original = sparse(tensor)
    result = complete(original)
    blocks = lambda vertex: {tuple(sorted(row // 3)) for row in vertex.indices}
    assert blocks(result.vertex) <= blocks(original)
    corrected = dense(result.vertex)
    for axes in itertools.permutations(range(3)):
        np.testing.assert_array_equal(corrected, corrected.transpose(axes))
    assert not np.any(corrected[:3, 9:, :])
    np.testing.assert_array_equal(original.coefficients, tensor[tuple(original.indices.T)])
    assert not result.vertex.coefficients.flags.writeable
    assert result.report["added_atom_triplet_blocks"] == 0


def test_interior_correction_and_cost_report_match_independent_dense_calculation():
    tensor = bond(12, 0, 1) + .8 * bond(12, 1, 2, axis=1) + .3 * bond(12, 2, 3, axis=2)
    tensor[0, 0, 0] += .1
    result = complete(sparse(tensor))
    corrected = dense(result.vertex)
    internal = np.s_[3:9, 3:9, 3:9]
    assert result.report["correction"]["interior_only"]["correction_frobenius"] == pytest.approx(np.linalg.norm((corrected - tensor)[internal]))
    assert result.report["correction"]["total"]["relative_frobenius"] == pytest.approx(np.linalg.norm(corrected - tensor) / np.linalg.norm(tensor))
    assert result.report["nnz_ratio"] == result.vertex.nnz / sparse(tensor).nnz


def test_large_change_cannot_pass_trial_screen():
    tensor = bond(6, 0, 1)
    tensor[0, 0, 0] += 4.
    result = complete(sparse(tensor))
    assert not result.report["bounded_trial_screen_passed"]
    assert not result.report["bounded_trial_screen"]["total_change_at_most_25_percent"]
    assert result.report["completed_translation"]["relative_max"] < 1e-12


def test_relative_masses_common_scale_same_hamiltonian_and_identity():
    tensor = bond(6, 0, 1, masses=[1., 2.])
    tensor[0, 0, 0] += .01
    first = complete(sparse(tensor), masses=[1., 2.])
    second = complete(sparse(tensor), masses=[2., 4.])
    assert first.vertex.cubic_physics_sha256 == second.vertex.cubic_physics_sha256
    assert first.report["model_identity_sha256"] == second.report["model_identity_sha256"]


def test_new_identity_tracks_source_and_report_parameters():
    tensor = bond(6, 0, 1)
    first = complete(sparse(tensor))
    second = complete(sparse(tensor), asr_tolerance=1e-11)
    assert first.vertex.cubic_physics_sha256 == second.vertex.cubic_physics_sha256
    assert first.report["model_identity_sha256"] != second.report["model_identity_sha256"]
    assert first.vertex.metadata["boundary_model"] != "uncorrected-material"
    assert "device_mapped_fc3_sha256" not in first.vertex.metadata


@pytest.mark.parametrize("masses", [[1], [1, 0], [1, -1], [1, np.inf], [1, 1j]])
def test_invalid_mass_contract_rejected(masses):
    with pytest.raises((TypeError, ValueError)):
        complete(sparse(bond(6, 0, 1)), masses=masses)


@pytest.mark.parametrize("kwargs", [{"asr_tolerance": 1.}, {"max_variables": 1}, {"max_iterations": True}])
def test_invalid_or_unbounded_controls_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        complete(sparse(bond(6, 0, 1)), **kwargs)


def test_zero_tensor_stays_zero():
    original = SparseCubicVertex(6, np.empty((0, 3), dtype=np.int64), np.empty(0))
    result = complete(original)
    assert result.vertex.nnz == 0
    assert result.report["completed_translation"]["relative_max"] == 0
    assert result.report["bounded_trial_screen_passed"]
    json.dumps(result.report, allow_nan=False)
