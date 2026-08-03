from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic._canonical import (
    crystal_mapping_content_sha256,
    fc3_content_sha256,
    primitive_crystal_content_sha256,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc3 import FC3Data, validate_fc3
from negfpy.anharmonic.translational_fc3 import (
    PHONO3PY_4_4_SYMPREC,
    PHONO3PY_PHASE_CONVENTION,
    PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
    FC3Term,
    TranslationalFC3,
    convert_fc3_to_translational,
    derive_phono3py_all_shortest,
)


def _mapping(
    *,
    purpose: str = "fc3",
    include_translation: bool = True,
) -> CrystalMapping:
    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.25, 0.0, 0.0]],
        masses=[1.0],
        symbols=["X"],
    )
    supercell = CrystalCell(
        lattice=np.diag([2.0, 1.0, 1.0]),
        scaled_positions=[
            [0.125, 0.0, 0.0],
            [0.625, 0.0, 0.0],
        ],
        masses=[1.0, 1.0],
        symbols=["X", "X"],
    )
    permutations = (
        np.array([[0, 1], [1, 0]], dtype=np.int64)
        if include_translation
        else np.array([[0, 1]], dtype=np.int64)
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([2, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=np.array([0], dtype=np.int64),
        s2p=np.array([0, 0], dtype=np.int64),
        p2p=np.array([0, -1], dtype=np.int64),
        shortest_vectors=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
            ]
        ),
        multiplicity=np.array(
            [
                [[2, 0]],
                [[1, 2]],
            ],
            dtype=np.int64,
        ),
        atomic_permutations=permutations,
        translation_permutations=permutations if include_translation else None,
        purpose=purpose,
        provenance={"fixture": "two-cell-one-atom"},
    )


def _compact_fc3() -> FC3Data:
    scalar = np.array([[1.0, 2.0], [2.0, 2.0]])
    values = scalar[None, :, :, None, None, None] * np.ones((3, 3, 3))
    data = FC3Data(
        values=values,
        masses=[1.0, 1.0],
        compact=True,
        p2s_map=np.array([0], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
        fc3_nonzero_indices=np.zeros((1, 2, 2), dtype=np.byte),
        provenance={"fixture": "compact"},
    )
    assert validate_fc3(data).passed
    return data


def _full_fc3() -> FC3Data:
    compact = _compact_fc3()
    values = np.empty((2, 2, 2, 3, 3, 3))
    values[0] = compact.values[0]
    translation = np.array([1, 0])
    values[1] = compact.values[0][translation][:, translation]
    data = FC3Data(
        values=values,
        masses=[1.0, 1.0],
        compact=False,
        p2s_map=np.array([0], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
        fc3_nonzero_indices=np.zeros((2, 2, 2), dtype=np.byte),
        provenance={"fixture": "full"},
    )
    assert validate_fc3(data).passed
    return data


def _anisotropic_compact_fc3() -> FC3Data:
    values = np.zeros((1, 2, 2, 3, 3, 3), dtype=np.float64)
    raw = np.arange(27.0, dtype=np.float64).reshape(3, 3, 3)
    values[0, 1, 1] = (raw + raw.transpose(0, 2, 1)) / 2.0
    data = FC3Data(
        values=values,
        masses=[1.0, 1.0],
        compact=True,
        p2s_map=np.array([0], dtype=np.int64),
        s2p_map=np.array([0, 0], dtype=np.int64),
        provenance={"fixture": "anisotropic-compact"},
    )
    assert validate_fc3(data).passed
    return data


def _single_atom_mapping() -> CrystalMapping:
    cell = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[1.0],
        symbols=["X"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=cell,
        supercell=cell,
        p2s=np.array([0], dtype=np.int64),
        s2p=np.array([0], dtype=np.int64),
        p2p=np.array([0], dtype=np.int64),
        shortest_vectors=np.zeros((1, 3), dtype=np.float64),
        multiplicity=np.array([[[1, 0]]], dtype=np.int64),
        atomic_permutations=np.array([[0]], dtype=np.int64),
        translation_permutations=np.array([[0]], dtype=np.int64),
        purpose="fc3",
        provenance={"fixture": "single-atom"},
    )


def _primitive_index(mapping: CrystalMapping, satom: int) -> int:
    return int(mapping.p2p[int(mapping.s2p[satom])])


def _independent_r0_leg(
    data: FC3Data,
    mapping: CrystalMapping,
    q_triplet: np.ndarray,
    primitive_indices: tuple[int, int, int],
    q_indices: tuple[int, int],
    leg_index: int,
    all_shortest: np.ndarray,
) -> np.ndarray:
    """Raw-array reference independent of TranslationalFC3 terms."""

    anchor, second_primitive, third_primitive = primitive_indices
    source_first = anchor if data.compact else int(mapping.p2s[anchor])
    block = np.zeros((3, 3, 3), dtype=np.complex128)
    for source_second in range(mapping.n_supercell_atoms):
        if _primitive_index(mapping, source_second) != second_primitive:
            continue
        second_count, second_start = mapping.multiplicity[
            source_second, anchor
        ]
        second_offsets = mapping.shortest_vectors[
            second_start : second_start + second_count
        ]
        second_phase = np.mean(
            np.exp(
                2j
                * np.pi
                * (second_offsets @ q_triplet[q_indices[0]])
            )
        )
        for source_third in range(mapping.n_supercell_atoms):
            if _primitive_index(mapping, source_third) != third_primitive:
                continue
            flag = int(
                all_shortest[anchor, source_second, source_third]
            )
            if leg_index > 1 and flag:
                continue
            factor = 3.0 if leg_index == 1 and flag else 1.0
            third_count, third_start = mapping.multiplicity[
                source_third, anchor
            ]
            third_offsets = mapping.shortest_vectors[
                third_start : third_start + third_count
            ]
            third_phase = np.mean(
                np.exp(
                    2j
                    * np.pi
                    * (third_offsets @ q_triplet[q_indices[1]])
                )
            )
            block += (
                factor
                * data.values[source_first, source_second, source_third]
                * second_phase
                * third_phase
            )
    return block


def _independent_r0_fourier(
    data: FC3Data,
    mapping: CrystalMapping,
    q_triplet: np.ndarray,
    all_shortest: np.ndarray,
) -> np.ndarray:
    n_primitive = mapping.n_primitive_atoms
    result = np.zeros(
        (n_primitive, n_primitive, n_primitive, 3, 3, 3),
        dtype=np.complex128,
    )
    q_sum = np.sum(q_triplet, axis=0)
    for first in range(n_primitive):
        for second in range(n_primitive):
            for third in range(n_primitive):
                first_phase = np.exp(
                    2j
                    * np.pi
                    * np.dot(
                        q_sum,
                        mapping.primitive.scaled_positions[first],
                    )
                )
                second_phase = np.exp(
                    2j
                    * np.pi
                    * np.dot(
                        q_sum,
                        mapping.primitive.scaled_positions[second],
                    )
                )
                third_phase = np.exp(
                    2j
                    * np.pi
                    * np.dot(
                        q_sum,
                        mapping.primitive.scaled_positions[third],
                    )
                )
                leg_1 = _independent_r0_leg(
                    data,
                    mapping,
                    q_triplet,
                    (first, second, third),
                    (1, 2),
                    1,
                    all_shortest,
                )
                leg_2 = _independent_r0_leg(
                    data,
                    mapping,
                    q_triplet,
                    (second, first, third),
                    (0, 2),
                    2,
                    all_shortest,
                ).transpose(1, 0, 2)
                leg_3 = _independent_r0_leg(
                    data,
                    mapping,
                    q_triplet,
                    (third, second, first),
                    (1, 0),
                    3,
                    all_shortest,
                ).transpose(2, 1, 0)
                result[first, second, third] = (
                    leg_1 * first_phase
                    + leg_2 * second_phase
                    + leg_3 * third_phase
                ) / 3.0
    return result


def _term_fingerprint(term: FC3Term) -> tuple:
    return (
        term.primitive_indices,
        term.source_indices,
        term.translated_supercell_indices,
        tuple(term.anchor_position),
        tuple(term.second_offset),
        tuple(term.third_offset),
        term.second_vector_index,
        term.third_vector_index,
        term.second_multiplicity,
        term.third_multiplicity,
        term.anchor_weight,
        term.multiplicity_weight,
        tuple(term.values.reshape(-1)),
    )


def test_fc3_term_and_container_detach_and_freeze_inputs() -> None:
    values = np.arange(27.0).reshape(3, 3, 3)
    anchor = np.array([0.25, 0.0, 0.0])
    second = np.array([0.5, 0.0, 0.0])
    third = np.array([1.0, 0.0, 0.0])
    term = FC3Term(
        primitive_indices=(0, 0, 0),
        source_indices=(0, 1, 0),
        translated_supercell_indices=(0, 1, 0),
        anchor_position=anchor,
        second_offset=second,
        third_offset=third,
        second_vector_index=2,
        third_vector_index=1,
        second_multiplicity=1,
        third_multiplicity=2,
        anchor_weight=0.5,
        multiplicity_weight=0.5,
        values=values,
    )
    provenance_array = np.array([1, 2])
    converted = TranslationalFC3(
        terms=(term,),
        n_primitive_atoms=1,
        n_supercell_atoms=2,
        units="eV/angstrom^3",
        mass_units="amu",
        source_layout="full",
        provenance={"nested": {"array": provenance_array}},
    )

    values[...] = -1.0
    anchor[...] = -1.0
    second[...] = -1.0
    third[...] = -1.0
    provenance_array[...] = -1

    np.testing.assert_array_equal(term.values, np.arange(27.0).reshape(3, 3, 3))
    np.testing.assert_array_equal(term.anchor_position, [0.25, 0.0, 0.0])
    np.testing.assert_array_equal(term.second_offset, [0.5, 0.0, 0.0])
    np.testing.assert_array_equal(term.third_offset, [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        converted.provenance["nested"]["array"],
        [1, 2],
    )
    assert term.weight == pytest.approx(0.25)
    assert converted.phase_convention == PHONO3PY_PHASE_CONVENTION
    assert "make_r0_average=False" in converted.anchor_convention
    assert isinstance(converted.provenance, MappingProxyType)
    for array in (
        term.values,
        term.anchor_position,
        term.second_offset,
        term.third_offset,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 0.0
    with pytest.raises(FrozenInstanceError):
        term.anchor_weight = 1.0  # type: ignore[misc]


def test_compact_conversion_preserves_each_raw_and_shortest_vector_term() -> None:
    data = _compact_fc3()
    mapping = _mapping()
    raw_before = data.values.copy()
    vectors_before = mapping.shortest_vectors.copy()

    converted = convert_fc3_to_translational(data, mapping)

    # Counts are (2, 1), hence sum_{j,k} m_j m_k = 9 terms.
    assert len(converted.terms) == 9
    assert converted.source_layout == "compact"
    assert converted.make_r0_average is False
    assert converted.provenance["fc3_nonzero_indices_applied"] is False
    assert converted.provenance["raw_atom_triplet_count"] == 4
    assert converted.provenance["expanded_term_count"] == 9
    assert converted.provenance["source_fc3_content_sha256"] == (
        fc3_content_sha256(data)
    )
    assert converted.provenance["primitive_crystal_content_sha256"] == (
        primitive_crystal_content_sha256(mapping)
    )
    assert converted.provenance[
        "fc3_crystal_mapping_content_sha256"
    ] == crystal_mapping_content_sha256(mapping)
    assert {term.source_indices for term in converted.terms} == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    }
    assert {term.translated_supercell_indices for term in converted.terms} == {
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, 0),
        (0, 1, 1),
    }
    assert {
        term.multiplicity_weight for term in converted.terms
    } == {0.25, 0.5, 1.0}
    assert all(term.anchor_weight == 1.0 for term in converted.terms)
    np.testing.assert_array_equal(data.values, raw_before)
    np.testing.assert_array_equal(mapping.shortest_vectors, vectors_before)


def test_full_and_compact_conversion_have_identical_fourier_transform() -> None:
    mapping = _mapping()
    compact = convert_fc3_to_translational(_compact_fc3(), mapping)
    full = convert_fc3_to_translational(_full_fc3(), mapping)
    q_triplet = np.array(
        [
            [0.13, 0.0, 0.0],
            [-0.31, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )

    assert len(compact.terms) == 9
    assert len(full.terms) == 9
    assert full.source_layout == "full"
    assert all(term.anchor_weight == 1.0 for term in full.terms)
    assert full.provenance["redundant_translation_audit"]["passed"]
    assert full.provenance["redundant_translation_audit"][
        "maximum_absolute_residual"
    ] == 0.0
    np.testing.assert_allclose(
        full.fourier(q_triplet),
        compact.fourier(q_triplet),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_r0_full_and_compact_conversion_are_identical() -> None:
    mapping = _mapping()
    compact = convert_fc3_to_translational(
        _compact_fc3(),
        mapping,
        make_r0_average=True,
    )
    full = convert_fc3_to_translational(
        _full_fc3(),
        mapping,
        make_r0_average=True,
    )
    q_triplet = np.array(
        [
            [0.13, 0.0, 0.0],
            [-0.31, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(full.all_shortest, compact.all_shortest)
    np.testing.assert_allclose(
        full.fourier(q_triplet),
        compact.fourier(q_triplet),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_fourier_reconstruction_matches_explicit_phono3py_phase_sum() -> None:
    converted = convert_fc3_to_translational(_compact_fc3(), _mapping())
    q_triplet = np.array(
        [
            [0.13, 0.0, 0.0],
            [-0.31, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )
    scalar = np.array([[1.0, 2.0], [2.0, 2.0]])
    vectors = (
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([[0.5, 0.0, 0.0]]),
    )

    phase_second = [
        np.mean(np.exp(2j * np.pi * (offsets @ q_triplet[1])))
        for offsets in vectors
    ]
    phase_third = [
        np.mean(np.exp(2j * np.pi * (offsets @ q_triplet[2])))
        for offsets in vectors
    ]
    anchor_phase = np.exp(
        2j
        * np.pi
        * np.dot(
            [0.25, 0.0, 0.0],
            np.sum(q_triplet, axis=0),
        )
    )
    scalar_result = anchor_phase * sum(
        scalar[j, k] * phase_second[j] * phase_third[k]
        for j in range(2)
        for k in range(2)
    )
    expected = scalar_result * np.ones((3, 3, 3))

    reciprocal = converted.fourier(q_triplet)

    assert reciprocal.shape == (1, 1, 1, 3, 3, 3)
    np.testing.assert_allclose(
        reciprocal[0, 0, 0],
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        converted.fourier(-q_triplet),
        reciprocal.conj(),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_term_order_is_canonical_and_reproducible() -> None:
    left = convert_fc3_to_translational(_full_fc3(), _mapping())
    right = convert_fc3_to_translational(_full_fc3(), _mapping())

    left_keys = tuple(_term_fingerprint(term) for term in left.terms)
    right_keys = tuple(_term_fingerprint(term) for term in right.terms)

    assert left_keys == right_keys
    assert left_keys == tuple(sorted(left_keys))


def test_inconsistent_redundant_full_row_is_rejected_not_averaged() -> None:
    source = _full_fc3()
    values = source.values.copy()
    values[1, 1, 1] += 0.5
    inconsistent = FC3Data(
        values=values,
        masses=source.masses,
        compact=False,
        p2s_map=source.p2s_map,
        s2p_map=source.s2p_map,
    )
    assert validate_fc3(inconsistent).passed

    with pytest.raises(ValueError, match="redundant translation audit failed"):
        convert_fc3_to_translational(inconsistent, _mapping())


def test_full_redundancy_tolerance_is_explicit_and_never_changes_terms() -> None:
    source = _full_fc3()
    values = source.values.copy()
    values[1, 1, 1] += 1.0e-11
    within_default_tolerance = FC3Data(
        values=values,
        masses=source.masses,
        compact=False,
        p2s_map=source.p2s_map,
        s2p_map=source.s2p_map,
    )
    assert validate_fc3(within_default_tolerance).passed

    reference = convert_fc3_to_translational(source, _mapping())
    accepted = convert_fc3_to_translational(
        within_default_tolerance,
        _mapping(),
    )
    residual = accepted.provenance["redundant_translation_audit"][
        "maximum_absolute_residual"
    ]

    assert residual == pytest.approx(1.0e-11)
    assert tuple(term.source_indices for term in accepted.terms) == tuple(
        term.source_indices for term in reference.terms
    )
    for accepted_term, reference_term in zip(
        accepted.terms,
        reference.terms,
        strict=True,
    ):
        np.testing.assert_array_equal(accepted_term.values, reference_term.values)
    with pytest.raises(ValueError, match="redundant translation audit failed"):
        convert_fc3_to_translational(
            within_default_tolerance,
            _mapping(),
            translation_atol=0.0,
            translation_rtol=0.0,
        )


def test_all_shortest_is_derived_exactly_and_frozen() -> None:
    no_unique_triangle = derive_phono3py_all_shortest(_mapping())
    unique_triangle = derive_phono3py_all_shortest(_single_atom_mapping())

    assert no_unique_triangle.dtype == np.byte
    np.testing.assert_array_equal(no_unique_triangle, np.zeros((1, 2, 2)))
    np.testing.assert_array_equal(unique_triangle, np.ones((1, 1, 1)))
    assert not no_unique_triangle.flags.writeable
    assert not unique_triangle.flags.writeable
    with pytest.raises(ValueError):
        unique_triangle[0, 0, 0] = 0


def test_all_shortest_requires_pure_translations_and_positive_symprec() -> None:
    with pytest.raises(
        NotImplementedError,
        match="translation_permutations",
    ):
        derive_phono3py_all_shortest(_mapping(include_translation=False))
    with pytest.raises(ValueError, match="symprec.*positive"):
        derive_phono3py_all_shortest(_mapping(), symprec=0.0)
    with pytest.raises(TypeError, match="symprec.*real"):
        derive_phono3py_all_shortest(_mapping(), symprec=True)


def test_r0_average_matches_independent_raw_three_anchor_reference() -> None:
    data = _anisotropic_compact_fc3()
    mapping = _mapping()
    values_before = data.values.copy()
    vectors_before = mapping.shortest_vectors.copy()
    translations_before = mapping.translation_permutations.copy()
    converted = convert_fc3_to_translational(
        data,
        mapping,
        make_r0_average=True,
    )
    q_triplet = np.array(
        [
            [0.13, 0.0, 0.0],
            [-0.31, 0.0, 0.0],
            [0.09, 0.0, 0.0],
        ]
    )
    expected_flags = derive_phono3py_all_shortest(mapping)
    expected = _independent_r0_fourier(
        data,
        mapping,
        q_triplet,
        expected_flags,
    )

    assert converted.make_r0_average is True
    assert (
        converted.anchor_convention
        == PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION
    )
    assert converted.symprec == PHONO3PY_4_4_SYMPREC
    np.testing.assert_array_equal(converted.all_shortest, expected_flags)
    assert converted.all_shortest is not None
    assert not converted.all_shortest.flags.writeable
    np.testing.assert_allclose(
        converted.fourier(q_triplet),
        expected,
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_allclose(
        converted.fourier(-q_triplet),
        expected.conj(),
        rtol=0.0,
        atol=1.0e-14,
    )
    np.testing.assert_array_equal(data.values, values_before)
    np.testing.assert_array_equal(mapping.shortest_vectors, vectors_before)
    np.testing.assert_array_equal(
        mapping.translation_permutations,
        translations_before,
    )
    assert converted.provenance["symmetrized"] is False
    assert converted.provenance["corrected"] is False
    assert converted.provenance["r0_policy"].startswith(
        "three-anchor Fourier representation only"
    )
    assert len(converted.provenance["all_shortest_sha256"]) == 64


def test_r0_unique_triangle_factor_and_skip_equal_legacy_result() -> None:
    mapping = _single_atom_mapping()
    values = np.arange(27.0, dtype=np.float64).reshape(
        1, 1, 1, 3, 3, 3
    )
    values = sum(
        np.transpose(values, axes + tuple(3 + axis for axis in axes))
        for axes in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    ) / 6.0
    data = FC3Data(
        values=values,
        masses=[1.0],
        compact=True,
        p2s_map=[0],
        s2p_map=[0],
    )
    assert validate_fc3(data).passed
    q_triplet = np.array(
        [[0.2, 0.1, 0.0], [-0.4, 0.0, 0.1], [0.2, -0.1, -0.1]]
    )
    legacy = convert_fc3_to_translational(data, mapping)
    averaged = convert_fc3_to_translational(
        data,
        mapping,
        make_r0_average=True,
    )

    np.testing.assert_array_equal(averaged.all_shortest, [[[1]]])
    np.testing.assert_allclose(
        averaged.fourier(q_triplet),
        legacy.fourier(q_triplet),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_r0_container_rejects_inconsistent_or_mutable_semantics() -> None:
    legacy = convert_fc3_to_translational(_compact_fc3(), _mapping())
    with pytest.raises(ValueError, match="all_shortest.*required"):
        TranslationalFC3(
            terms=legacy.terms,
            n_primitive_atoms=1,
            n_supercell_atoms=2,
            units=legacy.units,
            mass_units=legacy.mass_units,
            source_layout=legacy.source_layout,
            anchor_convention=PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
            make_r0_average=True,
        )
    with pytest.raises(ValueError, match="legacy.*all_shortest"):
        TranslationalFC3(
            terms=legacy.terms,
            n_primitive_atoms=1,
            n_supercell_atoms=2,
            units=legacy.units,
            mass_units=legacy.mass_units,
            source_layout=legacy.source_layout,
            all_shortest=np.zeros((1, 2, 2), dtype=np.byte),
        )


def test_pinned_phono3py_4_4_direct_real_to_reciprocal_match(
    tmp_path,
) -> None:
    phono3py = pytest.importorskip("phono3py")
    if phono3py.__version__ != "4.4.0":
        pytest.skip("Exact direct oracle is pinned to phono3py 4.4.0.")

    from phono3py import Phono3py
    from phono3py.phonon3.interaction import Interaction
    from phono3py.phonon3.real_to_reciprocal import RealToReciprocal
    from phonopy.structure.atoms import PhonopyAtoms

    from negfpy.anharmonic.io.phono3py_crystal import (
        read_phono3py_crystal_mapping,
    )

    unitcell = PhonopyAtoms(
        symbols=["Si", "Ge"],
        cell=np.eye(3),
        scaled_positions=[
            [0.0, 0.0, 0.0],
            [0.25, 0.25, 0.25],
        ],
        masses=[28.0855, 72.630],
    )
    source = Phono3py(
        unitcell,
        supercell_matrix=np.diag([2, 1, 1]),
        primitive_matrix=np.eye(3),
        symprec=PHONO3PY_4_4_SYMPREC,
    )
    source.mesh_numbers = [5, 5, 5]
    yaml_path = tmp_path / "synthetic-phono3py.yaml"
    yaml_path.write_text("synthetic direct-oracle fixture\n", encoding="utf-8")
    mapping = read_phono3py_crystal_mapping(
        yaml_path,
        loader=lambda **_kwargs: source,
    )

    values = np.ones((2, 4, 4, 3, 3, 3), dtype=np.float64)
    data = FC3Data(
        values=values,
        masses=mapping.supercell.masses,
        compact=True,
        p2s_map=mapping.p2s,
        s2p_map=mapping.s2p,
    )
    assert validate_fc3(data).passed

    interaction = Interaction(
        source.primitive,
        source.grid,
        source.primitive_symmetry,
        fc3=values,
        make_r0_average=True,
        lang="Python",
    )
    all_shortest = derive_phono3py_all_shortest(mapping)
    np.testing.assert_array_equal(all_shortest, interaction.all_shortest)
    assert 0 < np.count_nonzero(all_shortest) < all_shortest.size

    mesh = np.array([5, 5, 5], dtype=np.int64)
    triplet = np.array(
        [[2, 1, 1], [2, -1, -1], [1, 0, 0]],
        dtype=np.int64,
    )
    q_triplet = triplet / mesh
    for make_r0_average in (False, True):
        converted = convert_fc3_to_translational(
            data,
            mapping,
            make_r0_average=make_r0_average,
        )
        reference_transform = RealToReciprocal(
            values,
            source.primitive,
            mesh,
            make_r0_average=make_r0_average,
            all_shortest=(
                interaction.all_shortest if make_r0_average else None
            ),
        )
        reference_transform.run(triplet)
        reference = reference_transform.get_fc3_reciprocal()
        assert reference is not None
        np.testing.assert_allclose(
            converted.fourier(q_triplet),
            reference,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_conversion_rejects_invalid_or_incompatible_inputs() -> None:
    invalid_values = _compact_fc3().values.copy()
    invalid_values[0, 0, 1, 0, 0, 0] += 0.5
    invalid = FC3Data(
        values=invalid_values,
        masses=[1.0, 1.0],
        compact=True,
        p2s_map=[0],
        s2p_map=[0, 0],
    )
    assert not validate_fc3(invalid).passed

    with pytest.raises(ValueError, match="FC3 audit failed"):
        convert_fc3_to_translational(invalid, _mapping())
    with pytest.raises(ValueError, match="purpose.*fc3"):
        convert_fc3_to_translational(_compact_fc3(), _mapping(purpose="fc2"))
    with pytest.raises(NotImplementedError, match="pure translation.*compact FC3"):
        convert_fc3_to_translational(
            _full_fc3(),
            _mapping(include_translation=False),
        )
