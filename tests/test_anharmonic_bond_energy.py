"""Tiny complete-energy examples, with no material-array replay."""

from itertools import permutations

import numpy as np
import pytest

import negfpy.anharmonic.bond_energy as bond


def _chain_source(n=4):
    rng = np.random.default_rng(83)
    m = n-1
    coefficients = rng.normal(size=(3*m,)*3)
    coefficients = sum(coefficients.transpose(p) for p in permutations(range(3)))/6
    indices = np.indices(coefficients.shape)//3
    coefficients[np.max(indices, axis=0)-np.min(indices, axis=0) > 1] = 0.
    difference = np.zeros((3*m, 3*n))
    for edge in range(m):
        for direction in range(3):
            difference[3*edge+direction, 3*edge+direction] = -1.
            difference[3*edge+direction, 3*(edge+1)+direction] = 1.
    source = np.einsum("ea,fb,gc,efg->abc", difference, difference, difference,
                       coefficients, optimize=True)
    atom_source = source.reshape(n, 3, n, 3, n, 3).transpose(0, 2, 4, 1, 3, 5).copy()
    atom_coefficients = coefficients.reshape(m, 3, m, 3, m, 3).transpose(0, 2, 4, 1, 3, 5).copy()
    parent = np.r_[0, np.arange(n-1)]
    return atom_source, atom_coefficients, parent, list(range(n))


def _geometry(n=4):
    positions = np.zeros((n, 3))
    positions[:, 0] = np.arange(n)/(n+2)
    return positions, np.eye(3)*10


def test_cartesian_chain_energy_reconstructs_without_fitting():
    source, expected, parent, order = _chain_source()
    coefficients, children = bond._bond_coefficients(source, parent, order)
    relative = bond._relative_tensor(coefficients, parent, children)
    np.testing.assert_allclose(coefficients, expected, rtol=2e-14, atol=2e-14)
    np.testing.assert_allclose(relative, source, rtol=2e-14, atol=2e-14)
    # Mixed Cartesian components are essential; this is not three unrelated
    # scalar displacement chains.
    assert np.linalg.norm(expected[..., 0, 1, 2]) > 0
    displacement = np.arange(12).reshape(4, 3)/71
    differences = np.diff(displacement, axis=0)
    literal = np.einsum("ijkabc,ia,jb,kc->", expected, differences, differences, differences)/6
    assert bond._energy(source, displacement) == pytest.approx(literal, rel=3e-13, abs=1e-15)


def test_complete_cluster_cut_preserves_asr_and_permutation():
    source, _, parent, order = _chain_source()
    coefficients, children = bond._bond_coefficients(source, parent, order)
    # Keep an entire physical bond-energy cluster, all27Cartesian components,
    # rather than cutting force-constant entries atom by atom.
    mask = np.zeros(coefficients.shape[:3], dtype=bool)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                mask[i, j, k] = 1 in (i, j, k)
    cut = bond._relative_tensor(coefficients*mask[..., None, None, None], parent, children)
    assert max(row["max_abs"] for row in bond._asr(cut)["legs"]) < 3e-14
    assert bond._permutation(cut)["max_abs"] < 3e-14
    displacement = np.arange(12).reshape(4, 3)/71
    shift = np.array([.4, -.2, .7])
    assert bond._energy(cut, displacement+shift) == pytest.approx(bond._energy(cut, displacement), abs=2e-14)


def test_asr_breaking_remainder_is_retained_and_source_unchanged():
    source, _, _, _ = _chain_source()
    source[0, 0, 0, 1, 1, 1] += .3
    before = source.copy()
    source.setflags(write=False)
    result = bond.study_bulk_bond_energy(source, *_geometry())
    np.testing.assert_array_equal(source, before)
    assert result["source_unchanged"]
    assert result["source_ASR"]["legs"][0]["max_abs"] > .29
    assert result["remainder_discarded"] is False
    assert result["relative_part_adopted_as_model"] is False
    assert all(row["remainder"]["frobenius"] > .29 for row in result["gauges"])
    for row in result["gauges"]:
        assert row["relative_plus_remainder_closure"]["passed"]
        assert all(probe["closure"]["passed"] for probe in row["energy_probes"]["rows"])
        assert row["energy_probes"]["count"] == 4


def test_periodic_geometry_unwraps_same_tree_for_both_gauges():
    positions = np.array([[.95, 0, 0], [.05, 0, 0], [.15, 0, 0], [.25, 0, 0]])
    adjacency, periodic, unwrapped, nearest, metadata = bond._geometry(
        positions, np.eye(3)*10, 4, lambda: None)
    assert nearest == pytest.approx(1.)
    assert periodic[0, 1] == pytest.approx(1.)
    assert unwrapped[0, 3] == pytest.approx(3.)
    assert metadata["second_root"] == 3
    assert metadata["image_bound_uses_smallest_lattice_singular_value"]
    _, _, _, first = bond._root_tree(adjacency, 0)
    _, _, _, second = bond._root_tree(adjacency, 3)
    np.testing.assert_allclose(first-first[0], second-second[0], atol=1e-14)
    source, _, _, _ = _chain_source()
    result = bond.study_bulk_bond_energy(source, positions, np.eye(3)*10)
    assert [x["root_atom"] for x in result["gauges"]] == [0, 3]
    for row in result["gauges"]:
        assert row["remainder"]["frobenius"] < 4e-14
        locality = row["unwrapped_physical_locality"]
        assert locality["norm_is_basis_and_gauge_dependent"]
        assert locality["physical_probability_or_scattering_fraction"] is False
        assert sum(x["fraction_of_this_gauge_bond_coefficient_norm_squared"] for x in locality["bins"]) == pytest.approx(1.)


def test_admission_precedes_source_reads_and_rejects_invalid_geometry(monkeypatch):
    source, _, _, _ = _chain_source()
    monkeypatch.setattr(bond, "_finite", lambda *_: pytest.fail("No source read admitted"))
    with pytest.raises(ValueError, match="max_bytes"):
        bond.study_bulk_bond_energy(source, *_geometry(), max_bytes=1)
    with pytest.raises(TimeoutError):
        bond.study_bulk_bond_energy(source, *_geometry(), deadline_monotonic=0)
    monkeypatch.undo()
    with pytest.raises(ValueError, match="nonsingular"):
        bond.study_bulk_bond_energy(source, _geometry()[0], np.zeros((3, 3)))
    with pytest.raises(ValueError, match="coincide"):
        bond.study_bulk_bond_energy(source, np.zeros((4, 3)), np.eye(3))
    assert bond.plan_bulk_bond_energy(64)["peak_owned_bytes"] < 2*1024**3
