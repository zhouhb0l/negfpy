from __future__ import annotations

from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic.crystal import (
    CrystalCell,
    CrystalMapping,
    compare_primitive_mappings,
    validate_crystal_mapping,
)


def _primitive_cell(*, position_shift: float = 0.0) -> CrystalCell:
    return CrystalCell(
        lattice=np.diag([4.0, 5.0, 6.0]),
        scaled_positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.25 + position_shift, 0.5, 0.75],
            ]
        ),
        masses=np.array([28.085, 72.630]),
        symbols=["Si", "Ge"],
    )


def _mapping(
    purpose: str,
    *,
    n_supercell_atoms: int,
    position_shift: float = 0.0,
) -> CrystalMapping:
    if n_supercell_atoms == 2:
        p2s = np.array([0, 1], dtype=np.int64)
        s2p = np.array([0, 1], dtype=np.int64)
        p2p = np.array([0, 1], dtype=np.int64)
        supercell_matrix = np.eye(3, dtype=np.int64)
    elif n_supercell_atoms == 4:
        p2s = np.array([0, 2], dtype=np.int64)
        s2p = np.array([0, 0, 2, 2], dtype=np.int64)
        p2p = np.array([0, -1, 1, -1], dtype=np.int64)
        supercell_matrix = np.diag([2, 1, 1])
    else:
        raise ValueError("The test fixture supports two or four supercell atoms.")

    supercell = CrystalCell(
        lattice=supercell_matrix @ np.diag([4.0, 5.0, 6.0]),
        scaled_positions=np.zeros((n_supercell_atoms, 3)),
        masses=np.resize(np.array([28.085, 72.630]), n_supercell_atoms),
        symbols=np.resize(np.array(["Si", "Ge"]), n_supercell_atoms).tolist(),
    )
    multiplicity = np.empty((n_supercell_atoms, 2, 2), dtype=np.int64)
    multiplicity[..., 0] = 1
    multiplicity[..., 1] = np.arange(
        n_supercell_atoms * 2, dtype=np.int64
    ).reshape(n_supercell_atoms, 2)
    shortest_vectors = np.arange(
        n_supercell_atoms * 2 * 3, dtype=np.float64
    ).reshape(n_supercell_atoms * 2, 3)

    return CrystalMapping(
        primitive_matrix=np.array(
            [
                [0.0, 0.5, 0.5],
                [0.5, 0.0, 0.5],
                [0.5, 0.5, 0.0],
            ]
        ),
        supercell_matrix=supercell_matrix,
        primitive=_primitive_cell(position_shift=position_shift),
        supercell=supercell,
        p2s=p2s,
        s2p=s2p,
        p2p=p2p,
        shortest_vectors=shortest_vectors,
        multiplicity=multiplicity,
        atomic_permutations=np.arange(
            n_supercell_atoms, dtype=np.int64
        )[None, :],
        purpose=purpose,
        translation_permutations=(
            np.arange(n_supercell_atoms, dtype=np.int64)[None, :]
            if n_supercell_atoms == 2
            else np.array(
                [
                    [0, 1, 2, 3],
                    [1, 0, 3, 2],
                ],
                dtype=np.int64,
            )
        ),
        provenance={
            "reader": {
                "version": "4.4.0",
                "source_indices": np.array([1, 3], dtype=np.int64),
            }
        },
    )


def test_crystal_cell_detaches_freezes_and_derives_cartesian_positions() -> None:
    lattice = np.diag([2.0, 3.0, 4.0])
    scaled = np.array([[0.5, 0.5, 0.5]])
    masses = np.array([12.0])
    symbols = ["C"]

    cell = CrystalCell(
        lattice=lattice,
        scaled_positions=scaled,
        masses=masses,
        symbols=symbols,
    )
    lattice[...] = 9.0
    scaled[...] = 9.0
    masses[...] = 9.0
    symbols[0] = "X"

    np.testing.assert_array_equal(cell.lattice, np.diag([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(cell.scaled_positions, [[0.5, 0.5, 0.5]])
    np.testing.assert_array_equal(cell.cartesian_positions, [[1.0, 1.5, 2.0]])
    np.testing.assert_array_equal(cell.masses, [12.0])
    assert cell.symbols == ("C",)
    assert cell.n_atoms == 1
    for values in (
        cell.lattice,
        cell.scaled_positions,
        cell.cartesian_positions,
        cell.masses,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0


def test_crystal_cell_rejects_inconsistent_cartesian_positions() -> None:
    with pytest.raises(ValueError, match="cartesian_positions"):
        CrystalCell(
            lattice=np.eye(3),
            scaled_positions=[[0.25, 0.5, 0.75]],
            cartesian_positions=[[0.25, 0.5, 0.70]],
            masses=[1.0],
            symbols=["H"],
        )


def test_crystal_mapping_detaches_and_freezes_every_mapping_input() -> None:
    p2s = np.array([0, 2], dtype=np.int64)
    s2p = np.array([0, 0, 2, 2], dtype=np.int64)
    p2p = np.array([0, -1, 1, -1], dtype=np.int64)
    multiplicity = np.empty((4, 2, 2), dtype=np.int64)
    multiplicity[..., 0] = 1
    multiplicity[..., 1] = np.arange(8).reshape(4, 2)
    shortest_vectors = np.arange(24, dtype=float).reshape(8, 3)
    permutations = np.arange(4, dtype=np.int64)[None, :]
    translations = np.array(
        [[0, 1, 2, 3], [1, 0, 3, 2]],
        dtype=np.int64,
    )
    nested_provenance = np.array([1, 3], dtype=np.int64)

    mapping = CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([2, 1, 1]),
        primitive=_primitive_cell(),
        supercell=CrystalCell(
            lattice=np.diag([8.0, 5.0, 6.0]),
            scaled_positions=np.zeros((4, 3)),
            masses=[28.085, 72.630, 28.085, 72.630],
            symbols=["Si", "Ge", "Si", "Ge"],
        ),
        p2s=p2s,
        s2p=s2p,
        p2p=p2p,
        shortest_vectors=shortest_vectors,
        multiplicity=multiplicity,
        atomic_permutations=permutations,
        purpose="fc3",
        translation_permutations=translations,
        provenance={"reader": {"source_indices": nested_provenance}},
    )

    p2s[...] = 1
    s2p[...] = 1
    p2p[...] = 1
    multiplicity[...] = 0
    shortest_vectors[...] = -1
    permutations[...] = -1
    translations[...] = -1
    nested_provenance[...] = -1

    np.testing.assert_array_equal(mapping.p2s, [0, 2])
    np.testing.assert_array_equal(mapping.s2p, [0, 0, 2, 2])
    np.testing.assert_array_equal(mapping.p2p, [0, -1, 1, -1])
    np.testing.assert_array_equal(mapping.multiplicity[..., 0], 1)
    np.testing.assert_array_equal(mapping.atomic_permutations, [[0, 1, 2, 3]])
    np.testing.assert_array_equal(
        mapping.translation_permutations,
        [[0, 1, 2, 3], [1, 0, 3, 2]],
    )
    np.testing.assert_array_equal(
        mapping.provenance["reader"]["source_indices"], [1, 3]
    )
    assert isinstance(mapping.provenance, MappingProxyType)
    assert isinstance(mapping.provenance["reader"], MappingProxyType)
    assert mapping.purpose == "fc3"
    assert mapping.n_primitive_atoms == 2
    assert mapping.n_supercell_atoms == 4

    for values in (
        mapping.primitive_matrix,
        mapping.supercell_matrix,
        mapping.p2s,
        mapping.s2p,
        mapping.p2p,
        mapping.shortest_vectors,
        mapping.multiplicity,
        mapping.atomic_permutations,
        mapping.translation_permutations,
    ):
        assert not values.flags.writeable
        with pytest.raises(ValueError):
            values.flat[0] = 0
    with pytest.raises(TypeError):
        mapping.provenance["new"] = "value"

    report = validate_crystal_mapping(mapping)
    assert report.passed
    assert report.primitive_compatible is None
    assert report.metrics["shortest_vector_count"] == 8
    assert isinstance(report.metrics, MappingProxyType)


def test_mapping_rejects_s2p_values_outside_p2s() -> None:
    kwargs = _mapping("fc3", n_supercell_atoms=4)
    with pytest.raises(ValueError, match="s2p values"):
        CrystalMapping(
            primitive_matrix=kwargs.primitive_matrix,
            supercell_matrix=kwargs.supercell_matrix,
            primitive=kwargs.primitive,
            supercell=kwargs.supercell,
            p2s=kwargs.p2s,
            s2p=[0, 1, 2, 2],
            p2p=kwargs.p2p,
            shortest_vectors=kwargs.shortest_vectors,
            multiplicity=kwargs.multiplicity,
            atomic_permutations=kwargs.atomic_permutations,
            purpose="fc3",
        )


@pytest.mark.parametrize(
    ("s2p", "p2p", "message"),
    [
        ([0, 0, 0, 2], [0, -1, 1, -1], r"s2p\[p2s"),
        ([0, 0, 2, 2], [0, -1, 0, -1], r"p2p\[p2s"),
    ],
)
def test_mapping_rejects_representative_identity_mismatch(
    s2p: list[int],
    p2p: list[int],
    message: str,
) -> None:
    kwargs = _mapping("fc3", n_supercell_atoms=4)
    with pytest.raises(ValueError, match=message):
        CrystalMapping(
            primitive_matrix=kwargs.primitive_matrix,
            supercell_matrix=kwargs.supercell_matrix,
            primitive=kwargs.primitive,
            supercell=kwargs.supercell,
            p2s=kwargs.p2s,
            s2p=s2p,
            p2p=p2p,
            shortest_vectors=kwargs.shortest_vectors,
            multiplicity=kwargs.multiplicity,
            atomic_permutations=kwargs.atomic_permutations,
            purpose="fc3",
        )


@pytest.mark.parametrize("bad_offset", [0, 9])
def test_mapping_rejects_overlap_or_gap_in_shortest_vector_partition(
    bad_offset: int,
) -> None:
    kwargs = _mapping("fc3", n_supercell_atoms=4)
    multiplicity = kwargs.multiplicity.copy()
    multiplicity[0, 1, 1] = bad_offset

    with pytest.raises(ValueError, match="partition"):
        CrystalMapping(
            primitive_matrix=kwargs.primitive_matrix,
            supercell_matrix=kwargs.supercell_matrix,
            primitive=kwargs.primitive,
            supercell=kwargs.supercell,
            p2s=kwargs.p2s,
            s2p=kwargs.s2p,
            p2p=kwargs.p2p,
            shortest_vectors=kwargs.shortest_vectors,
            multiplicity=multiplicity,
            atomic_permutations=kwargs.atomic_permutations,
            purpose="fc3",
        )


def test_mapping_rejects_nonpermutation_atomic_permutation_row() -> None:
    kwargs = _mapping("fc3", n_supercell_atoms=4)
    with pytest.raises(ValueError, match="atomic_permutations"):
        CrystalMapping(
            primitive_matrix=kwargs.primitive_matrix,
            supercell_matrix=kwargs.supercell_matrix,
            primitive=kwargs.primitive,
            supercell=kwargs.supercell,
            p2s=kwargs.p2s,
            s2p=kwargs.s2p,
            p2p=kwargs.p2p,
            shortest_vectors=kwargs.shortest_vectors,
            multiplicity=kwargs.multiplicity,
            atomic_permutations=[[0, 1, 1, 3]],
            purpose="fc3",
        )


def test_fc2_fc3_primitive_comparison_ignores_different_supercells() -> None:
    fc2 = _mapping("fc2", n_supercell_atoms=2)
    fc3 = _mapping("fc3", n_supercell_atoms=4)

    report = compare_primitive_mappings(fc2, fc3)

    assert report.passed
    assert report.primitive_compatible is True
    assert report.metrics["primitive_lattice_max_abs"] == 0.0
    assert report.metrics["primitive_scaled_position_modulo_max_abs"] == 0.0


def test_fc2_fc3_primitive_comparison_rejects_incompatible_primitive() -> None:
    fc2 = _mapping("fc2", n_supercell_atoms=2)
    fc3 = _mapping("fc3", n_supercell_atoms=4, position_shift=0.01)
    original_positions = fc3.primitive.scaled_positions.copy()

    report = compare_primitive_mappings(fc2, fc3, atol=1.0e-10)

    assert not report.passed
    assert report.primitive_compatible is False
    assert any("scaled positions" in message for message in report.errors)
    assert report.metrics["primitive_scaled_position_modulo_max_abs"] == pytest.approx(
        0.01
    )
    np.testing.assert_array_equal(fc3.primitive.scaled_positions, original_positions)
