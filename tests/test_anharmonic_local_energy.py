from itertools import permutations
from time import monotonic

import numpy as np
import pytest

from negfpy.anharmonic import local_energy as local


def _cubic_product(n, vectors, coefficient=1.):
    """Third derivative of coefficient*prod(v_i.u)/6, including equal factors."""
    tensor = np.zeros((n*3,)*3)
    unique = {tuple(order) for order in permutations(range(3))}
    # Test cases use different formal factors even if their atom support repeats.
    for order in unique:
        tensor += coefficient*np.einsum("i,j,k->ijk", *(vectors[index].ravel() for index in order))/6
    return tensor.reshape(n, 3, n, 3, n, 3).transpose(0, 2, 4, 1, 3, 5)


def _difference(n, first, second, axis):
    result = np.zeros((n, 3))
    result[second, axis], result[first, axis] = 1., -1.
    return result


def _assert_conservative(tensor):
    for leg in range(3):
        np.testing.assert_allclose(tensor.sum(axis=leg), 0., atol=1e-13)
    for order in permutations(range(3)):
        np.testing.assert_allclose(tensor, tensor.transpose(tuple(order)+tuple(axis+3 for axis in order)), atol=1e-13)


def test_unequal_cartesian_bond_energy_exact_and_source_unchanged():
    n = 2
    factors = [_difference(n, 0, 1, axis) for axis in (0, 1, 1)]
    source = _cubic_product(n, factors, 2.7)
    before = source.copy()
    result = local.study_local_energy_fit(source, [[.1, 0, 0], [.2, 0, 0]], np.eye(3)*10, [7, 9], neighbors=1)
    assert result["fit"]["source_reconstruction_passed"]
    assert result["residual"]["relative_frobenius"] < 1e-13
    assert result["source"]["unchanged"]
    np.testing.assert_array_equal(source, before)
    assert result["scope"]["physical_model_adopted"] is False
    rng = np.random.default_rng(20261026)
    for probe in result["invariants"]["bounded_energy_probes"]:
        displacement = rng.uniform(-1e-5, 1e-5, size=(n, 3))
        difference = displacement[1]-displacement[0]
        expected_energy = 2.7/6*difference[0]*difference[1]**2
        np.testing.assert_allclose(probe["raw_energy"], expected_energy, rtol=1e-13, atol=1e-30)
        assert probe["remainder_included"]
        assert probe["closure_abs_error"] < 1e-27


def test_whole_center_cluster_cut_remains_conservative():
    # Retain one complete center-star term and remove every other center term.
    # This is an energy-cluster cut, not a tensor-entry crop.
    n = 4
    vectors = [_difference(n, 0, neighbor, axis) for neighbor, axis in ((1, 0), (2, 1), (3, 2))]
    source = _cubic_product(n, vectors, 1.9)
    _assert_conservative(source)
    result = local.study_local_energy_fit(source,
        [[.5, .5, .5], [.6, .5, .5], [.5, .6, .5], [.5, .5, .6]],
        np.eye(3)*10, [0, 1, 2, 3], neighbors=3)
    assert result["fit"]["source_reconstruction_passed"]
    assert max(row["max_abs"] for row in result["invariants"]["site_star"]["acoustic_sum_rule_by_leg"]) < 1e-13
    assert result["invariants"]["site_star"]["simultaneous_atom_cartesian_permutation_max_abs_error"] < 1e-13


def test_periodic_class_replication_uses_coherent_image_tags():
    n = 4
    source = sum((_cubic_product(n, [_difference(n, a, b, 0)]*3, .8) for a, b in ((0, 1), (2, 3))), np.zeros((n, n, n, 3, 3, 3)))
    positions = [[.95, .2, .2], [.05, .2, .2], [.45, .2, .2], [.55, .2, .2]]
    result = local.study_local_energy_fit(source, positions, np.eye(3)*10, [11, 17, 11, 17], neighbors=1)
    assert result["fit"]["parameter_count"] == 20
    assert result["fit"]["source_reconstruction_passed"]
    assert result["geometry"]["stars"][0]["neighbor_image_integers"] == [[1, 0, 0]]
    assert result["geometry"]["class_environments_consistent"]
    assert result["basis"]["arbitrary_wavevector_image_resolved_reconstruction_established"] is False


def test_nonlocal_or_asr_breaking_source_remainder_is_retained():
    n = 4
    source = _cubic_product(n, [_difference(n, 0, 2, 0)]*3, .9)
    source[0, 0, 0, 2, 2, 2] += .7  # Deliberately break ASR; never repair it.
    original = source.copy()
    result = local.study_local_energy_fit(source,
        [[.1, 0, 0], [.2, 0, 0], [.6, 0, 0], [.7, 0, 0]], np.eye(3)*10,
        [0, 1, 0, 1], neighbors=1)
    assert not result["fit"]["source_reconstruction_passed"]
    assert result["residual"]["outside_coverage_irreducible_floor"]["frobenius"] > 0
    assert result["residual"]["retained_explicitly"]
    assert result["residual"]["sum_closure"]["passed"]
    assert result["invariants"]["raw"]["acoustic_sum_rule_by_leg"][0]["max_abs"] >= .7
    np.testing.assert_array_equal(source, original)


def test_admission_and_ambiguous_geometry_fail_before_fitting():
    plan = local.plan_local_energy_fit(64, 2)
    assert plan["parameter_count"] == 728
    assert plan["peak_owned_bytes"] < 2*1024**3
    source = np.zeros((2, 2, 2, 3, 3, 3))
    args = (source, [[0, 0, 0], [.1, 0, 0]], np.eye(3)*10, [0, 1])
    with pytest.raises(ValueError, match="memory"):
        local.study_local_energy_fit(*args, neighbors=1, max_bytes=1)
    with pytest.raises(TimeoutError):
        local.study_local_energy_fit(*args, neighbors=1, deadline_monotonic=monotonic()-1)
    with pytest.raises(ValueError, match="ambiguous minimum images"):
        local.study_local_energy_fit(source, [[0, 0, 0], [.5, 0, 0]], np.eye(3)*10, [0, 1], neighbors=1)
