from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType

import numpy as np
import pytest

from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc3 import FC3Data
from negfpy.anharmonic.translational_fc3 import (
    PHONO3PY_LEGACY_ANCHOR_CONVENTION,
    PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
    FC3Term,
    TranslationalFC3,
    convert_fc3_to_translational,
)
from negfpy.anharmonic.translational_fc3_validation import (
    EXACT_NONZERO_SUPPORT_POLICY,
    PAIRWISE_ATOM_AXES,
    TranslationalFC3ValidationReport,
    validate_translational_fc3,
)


def _two_atom_mapping() -> CrystalMapping:
    cell = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
        masses=[1.0, 1.0],
        symbols=["X", "Y"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=cell,
        supercell=cell,
        p2s=[0, 1],
        s2p=[0, 1],
        p2p=[0, 1],
        shortest_vectors=[
            [0.0, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [-0.25, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        multiplicity=[
            [[1, 0], [1, 2]],
            [[1, 1], [1, 3]],
        ],
        atomic_permutations=[[0, 1]],
        translation_permutations=[[0, 1]],
        purpose="fc3",
        provenance={"fixture": "two-atom-identity"},
    )


def _vertex_from_gamma(
    gamma: np.ndarray,
    *,
    make_r0_average: bool = False,
) -> TranslationalFC3:
    mapping = _two_atom_mapping()
    terms: list[FC3Term] = []
    for first in range(2):
        for second in range(2):
            for third in range(2):
                second_index = int(
                    mapping.multiplicity[second, first, 1]
                )
                third_index = int(
                    mapping.multiplicity[third, first, 1]
                )
                terms.append(
                    FC3Term(
                        primitive_indices=(first, second, third),
                        source_indices=(first, second, third),
                        translated_supercell_indices=(
                            first,
                            second,
                            third,
                        ),
                        anchor_position=(
                            mapping.primitive.scaled_positions[first]
                        ),
                        second_offset=mapping.shortest_vectors[
                            second_index
                        ],
                        third_offset=mapping.shortest_vectors[third_index],
                        second_vector_index=second_index,
                        third_vector_index=third_index,
                        second_multiplicity=1,
                        third_multiplicity=1,
                        anchor_weight=1.0,
                        multiplicity_weight=1.0,
                        values=gamma[first, second, third],
                    )
                )
    return TranslationalFC3(
        terms=tuple(terms),
        n_primitive_atoms=2,
        n_supercell_atoms=2,
        units="eV/angstrom^3",
        mass_units="amu",
        source_layout="compact",
        anchor_convention=(
            PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION
            if make_r0_average
            else PHONO3PY_LEGACY_ANCHOR_CONVENTION
        ),
        make_r0_average=make_r0_average,
        all_shortest=(
            np.zeros((2, 2, 2), dtype=np.byte)
            if make_r0_average
            else None
        ),
        symprec=1.0e-5 if make_r0_average else None,
    )


def _asr_gamma() -> np.ndarray:
    signs = np.array([1.0, -1.0])
    atom_tensor = np.einsum("i,j,k->ijk", signs, signs, signs)
    cartesian = np.zeros((3, 3, 3), dtype=np.float64)
    cartesian[0, 0, 0] = 1.0
    return atom_tensor[..., None, None, None] * cartesian


def _support_mapping() -> CrystalMapping:
    primitive = CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0]],
        masses=[1.0],
        symbols=["X"],
    )
    supercell = CrystalCell(
        lattice=np.diag([2.0, 1.0, 1.0]),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        masses=[1.0, 1.0],
        symbols=["X", "X"],
    )
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.diag([2, 1, 1]),
        primitive=primitive,
        supercell=supercell,
        p2s=[0],
        s2p=[0, 0],
        p2p=[0, -1],
        shortest_vectors=[
            [0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        multiplicity=[[[1, 0]], [[2, 1]]],
        atomic_permutations=[[0, 1], [1, 0]],
        translation_permutations=[[0, 1], [1, 0]],
        purpose="fc3",
        provenance={"fixture": "pairwise-diameter"},
    )


def _support_vertex(*, drop_last: bool = False) -> TranslationalFC3:
    mapping = _support_mapping()
    cartesian = np.zeros((3, 3, 3), dtype=np.float64)
    cartesian[0, 1, 2] = 2.0
    terms: list[FC3Term] = []
    for second in range(2):
        second_count, second_start = mapping.multiplicity[second, 0]
        for third in range(2):
            third_count, third_start = mapping.multiplicity[third, 0]
            values = (
                np.zeros((3, 3, 3), dtype=np.float64)
                if (second, third) == (0, 0)
                else cartesian
            )
            for second_index in range(
                second_start,
                second_start + second_count,
            ):
                for third_index in range(
                    third_start,
                    third_start + third_count,
                ):
                    terms.append(
                        FC3Term(
                            primitive_indices=(0, 0, 0),
                            source_indices=(0, second, third),
                            translated_supercell_indices=(
                                0,
                                second,
                                third,
                            ),
                            anchor_position=[0.0, 0.0, 0.0],
                            second_offset=mapping.shortest_vectors[
                                second_index
                            ],
                            third_offset=mapping.shortest_vectors[
                                third_index
                            ],
                            second_vector_index=int(second_index),
                            third_vector_index=int(third_index),
                            second_multiplicity=int(second_count),
                            third_multiplicity=int(third_count),
                            anchor_weight=1.0,
                            multiplicity_weight=1.0
                            / float(second_count * third_count),
                            values=values,
                        )
                    )
    if drop_last:
        terms.pop()
    return TranslationalFC3(
        terms=tuple(terms),
        n_primitive_atoms=1,
        n_supercell_atoms=2,
        units="eV/angstrom^3",
        mass_units="amu",
        source_layout="compact",
    )


@pytest.mark.parametrize("make_r0_average", [False, True])
def test_three_gamma_asr_axes_pass_and_report_is_immutable(
    make_r0_average: bool,
) -> None:
    mapping = _two_atom_mapping()
    vertex = _vertex_from_gamma(
        _asr_gamma(),
        make_r0_average=make_r0_average,
    )
    values_before = tuple(term.values.copy() for term in vertex.terms)
    lattice_before = mapping.primitive.lattice.copy()

    report = validate_translational_fc3(vertex, mapping)

    assert isinstance(report, TranslationalFC3ValidationReport)
    assert report.passed
    assert report.mapping_valid
    assert report.schedule_complete
    assert report.gamma_finite
    assert report.asr_valid
    assert report.make_r0_average is make_r0_average
    assert report.asr_valid_by_axis == (True, True, True)
    np.testing.assert_array_equal(report.gamma_asr_max_abs, [0.0, 0.0, 0.0])
    np.testing.assert_array_equal(report.gamma_asr_relative, [0.0, 0.0, 0.0])
    assert len(report.gamma_asr_residuals) == 3
    assert all(
        residual.shape == (2, 2, 3, 3, 3)
        for residual in report.gamma_asr_residuals
    )
    assert isinstance(report.metrics, MappingProxyType)
    assert report.metrics["gamma_asr_axis_0_max_abs"] == 0.0
    assert report.metrics["gamma_asr_axis_1_max_abs"] == 0.0
    assert report.metrics["gamma_asr_axis_2_max_abs"] == 0.0
    for actual, expected in zip(vertex.terms, values_before, strict=True):
        np.testing.assert_array_equal(actual.values, expected)
    np.testing.assert_array_equal(mapping.primitive.lattice, lattice_before)
    for array in (
        *report.gamma_asr_residuals,
        report.gamma_asr_max_abs,
        report.gamma_asr_relative,
        report.pairwise_minimum_distances,
        report.pairwise_maximum_distances,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.flat[0] = 1.0
    with pytest.raises(FrozenInstanceError):
        report.support_valid = False  # type: ignore[misc]


@pytest.mark.parametrize("failed_axis", [0, 1, 2])
def test_each_gamma_atom_axis_is_checked_independently(
    failed_axis: int,
) -> None:
    signs = np.array([1.0, -1.0])
    factors = [signs.copy(), signs.copy(), signs.copy()]
    factors[failed_axis] = np.array([1.0, 0.0])
    perturbation = np.einsum(
        "i,j,k->ijk",
        factors[0],
        factors[1],
        factors[2],
    )
    cartesian = np.zeros((3, 3, 3), dtype=np.float64)
    cartesian[0, 0, 0] = 0.25
    gamma = _asr_gamma() + perturbation[..., None, None, None] * cartesian

    report = validate_translational_fc3(
        _vertex_from_gamma(gamma),
        _two_atom_mapping(),
        asr_atol=1.0e-14,
        asr_rtol=0.0,
    )

    expected = [True, True, True]
    expected[failed_axis] = False
    assert report.asr_valid_by_axis == tuple(expected)
    assert not report.asr_valid
    assert not report.passed
    assert report.gamma_asr_max_abs[failed_axis] == pytest.approx(0.25)
    assert all(
        report.gamma_asr_max_abs[axis] == pytest.approx(0.0)
        for axis in range(3)
        if axis != failed_axis
    )
    assert any(
        f"atom axis {failed_axis}" in message for message in report.errors
    )


def test_complete_pairwise_cartesian_range_includes_nonanchor_pair() -> None:
    report = validate_translational_fc3(
        _support_vertex(),
        _support_mapping(),
    )

    assert report.support_valid
    assert report.support_policy == EXACT_NONZERO_SUPPORT_POLICY
    assert report.pairwise_atom_axes == PAIRWISE_ATOM_AXES
    np.testing.assert_allclose(
        report.pairwise_minimum_distances,
        [0.0, 0.0, 0.0],
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        report.pairwise_maximum_distances,
        [1.0, 1.0, 2.0],
        rtol=0.0,
        atol=0.0,
    )
    assert report.maximum_pairwise_distance == pytest.approx(2.0)
    assert report.nonzero_expanded_term_count == 8
    assert report.exact_zero_expanded_term_count == 1
    assert report.nonzero_raw_block_count == 3
    assert report.exact_zero_raw_block_count == 1
    assert report.metrics["support_pair_1_2_maximum"] == pytest.approx(2.0)
    assert report.metrics["support_maximum_pairwise_distance"] == pytest.approx(
        2.0
    )
    # The two anchor radii are only 1; the complete cluster diameter is 2.
    assert report.maximum_pairwise_distance > np.max(
        report.pairwise_maximum_distances[:2]
    )


@pytest.mark.parametrize("make_r0_average", [False, True])
def test_converted_dense_vertex_has_complete_auditable_schedule(
    make_r0_average: bool,
) -> None:
    mapping = _support_mapping()
    scalar = np.array([[1.0, 2.0], [2.0, 2.0]])
    values = scalar[None, :, :, None, None, None] * np.ones((3, 3, 3))
    data = FC3Data(
        values=values,
        masses=[1.0, 1.0],
        compact=True,
        p2s_map=[0],
        s2p_map=[0, 0],
    )
    vertex = convert_fc3_to_translational(
        data,
        mapping,
        make_r0_average=make_r0_average,
    )

    report = validate_translational_fc3(vertex, mapping)

    assert report.mapping_valid
    assert report.schedule_complete
    assert report.support_valid
    assert report.nonzero_expanded_term_count == 9
    assert report.exact_zero_expanded_term_count == 0
    np.testing.assert_array_equal(
        report.pairwise_maximum_distances,
        [1.0, 1.0, 2.0],
    )


def test_incomplete_expanded_schedule_is_reported_not_repaired() -> None:
    vertex = _support_vertex(drop_last=True)
    before = tuple(term.values.copy() for term in vertex.terms)

    report = validate_translational_fc3(vertex, _support_mapping())

    assert report.mapping_valid
    assert not report.schedule_complete
    assert not report.support_valid
    assert not report.passed
    assert any("expanded term schedule" in message for message in report.errors)
    assert len(vertex.terms) == 8
    for term, expected in zip(vertex.terms, before, strict=True):
        np.testing.assert_array_equal(term.values, expected)


def test_validation_rejects_invalid_call_contract() -> None:
    vertex = _support_vertex()
    mapping = _support_mapping()
    with pytest.raises(TypeError, match="vertex.*TranslationalFC3"):
        validate_translational_fc3(object(), mapping)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="mapping.*CrystalMapping"):
        validate_translational_fc3(vertex, object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="asr_atol.*non-negative"):
        validate_translational_fc3(vertex, mapping, asr_atol=-1.0)
    with pytest.raises(TypeError, match="asr_rtol.*real"):
        validate_translational_fc3(vertex, mapping, asr_rtol=True)
