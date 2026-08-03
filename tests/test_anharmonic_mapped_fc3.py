from __future__ import annotations

from collections import defaultdict
import itertools
import math

import numpy as np
import pytest

from negfpy.anharmonic._canonical import (
    fc3_content_sha256,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.fc3 import FC3Data, FC3ValidationReport, validate_fc3
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import (
    DeviceMappedFC3,
    MassNormalizedTranslationalFC3,
    _accumulate_compact,
    _compact_accumulation_preflight,
    _permutation_audit,
    map_fc3_to_device,
    mass_normalize_translational_fc3,
    scale_device_fc3,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.translational_fc3 import (
    FC3Term,
    PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
    TranslationalFC3,
    convert_fc3_to_translational,
)
from negfpy.anharmonic.types import SolverScale


def _cell() -> CrystalCell:
    return CrystalCell(
        lattice=np.eye(3),
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        masses=[1.0, 1.0],
        symbols=("A", "B"),
    )


def _mapping(
    purpose: str,
    *,
    duplicate_images: bool = False,
    cross_cell: bool = False,
    two_cell: bool = False,
) -> CrystalMapping:
    cell = _cell()
    second_from_first = 2.5 if two_cell else (1.5 if cross_cell else 0.5)
    first_from_second = -2.5 if two_cell else (-1.5 if cross_cell else -0.5)
    base = (
        np.array([0.0, 0.0, 0.0]),
        np.array([first_from_second, 0.0, 0.0]),
        np.array([second_from_first, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    count = 2 if duplicate_images else 1
    vectors: list[np.ndarray] = []
    multiplicity = np.empty((2, 2, 2), dtype=np.int64)
    # CrystalMapping partitions vectors as [supercell atom, primitive anchor].
    for atom, anchor in itertools.product(range(2), repeat=2):
        vector = base[2 * atom + anchor]
        offset = len(vectors)
        vectors.extend(np.array(vector, copy=True) for _ in range(count))
        multiplicity[atom, anchor] = (count, offset)
    return CrystalMapping(
        primitive_matrix=np.eye(3),
        supercell_matrix=np.eye(3, dtype=np.int64),
        primitive=cell,
        supercell=cell,
        p2s=[0, 1],
        s2p=[0, 1],
        p2p=[0, 1],
        shortest_vectors=vectors,
        multiplicity=multiplicity,
        atomic_permutations=[[0, 1]],
        translation_permutations=[[0, 1]],
        purpose=purpose,
    )


def _layered_mapping(*, transform: np.ndarray | None = None):
    fc2 = FC2Data(
        values=np.zeros((2, 2, 3, 3)),
        masses=[1.0, 1.0],
        compact=False,
        p2s_map=[0, 1],
        s2p_map=[0, 1],
        units="J/m^2",
        mass_units="kg",
    )
    return build_layered_device_mapping(
        _mapping("fc2"),
        fc2,
        transport_supercell_matrix=(
            np.eye(3, dtype=np.int64) if transform is None else transform
        ),
    )


def _raw_fc3(*, g: float = 2.0) -> FC3Data:
    values = np.zeros((2, 2, 2, 3, 3, 3))
    signs = (1.0, -1.0)
    for first, second, third in itertools.product(range(2), repeat=3):
        values[first, second, third, 0, 0, 0] = (
            g * signs[first] * signs[second] * signs[third]
        )
    return FC3Data(
        values=values,
        masses=[1.0, 1.0],
        compact=False,
        p2s_map=[0, 1],
        s2p_map=[0, 1],
        units="J/m^3",
        mass_units="kg",
    )


def _translational(
    *,
    duplicate_images: bool = False,
    cross_cell: bool = False,
    two_cell: bool = False,
) -> tuple[TranslationalFC3, CrystalMapping, FC3ValidationReport]:
    raw = _raw_fc3()
    certification = validate_fc3(raw)
    assert certification.passed
    assert certification.permutation_complete
    crystal = _mapping(
        "fc3",
        duplicate_images=duplicate_images,
        cross_cell=cross_cell,
        two_cell=two_cell,
    )
    translated = convert_fc3_to_translational(raw, crystal)
    return translated, crystal, certification


def _normalized(
    *,
    duplicate_images: bool = False,
    cross_cell: bool = False,
    two_cell: bool = False,
) -> MassNormalizedTranslationalFC3:
    translated, crystal, certification = _translational(
        duplicate_images=duplicate_images,
        cross_cell=cross_cell,
        two_cell=two_cell,
    )
    return mass_normalize_translational_fc3(
        translated,
        crystal,
        source_certification=certification,
    )


def test_mass_normalization_keeps_image_weights_separate_and_immutable() -> None:
    normalized = _normalized(duplicate_images=True)
    assert normalized.output_units == "J m^-3 kg^-3/2"
    assert normalized.source_certification_kind == "full_validation"
    assert normalized.source_fc3_content_sha256 == fc3_content_sha256(_raw_fc3())
    assert len(normalized.terms) == 2 * 2 * 2 * 2 * 2
    assert {term.multiplicity_weight for term in normalized.terms} == {0.25}
    assert {abs(float(term.values[0, 0, 0])) for term in normalized.terms} == {
        2.0
    }
    assert all(term.anchor_weight == 1.0 for term in normalized.terms)
    with pytest.raises(ValueError):
        normalized.terms[0].values[0, 0, 0] = 7.0
    with pytest.raises(TypeError):
        MassNormalizedTranslationalFC3()  # type: ignore[call-arg]


def test_all_cosets_and_longitudinal_anchors_are_mapped_without_prefactor() -> None:
    mapping = _layered_mapping(transform=np.diag([2, 1, 1]))
    normalized = _normalized()
    mapped = map_fc3_to_device(
        normalized,
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert isinstance(mapped, DeviceMappedFC3)
    assert mapped.report.certification_passed
    assert mapping.n_transport_basis_atoms == 4
    # 2 primitive cells/cosets per transport cell, 8 ordered atom triplets
    # for (u_A-u_B)^3, and 3 complete interaction layers.
    assert mapped.nnz == 2 * 8 * 3
    assert mapped.report.final_nonzero_count == mapped.nnz
    assert mapped.report.exact_permutation_equality
    assert mapped.report.permutation_max_abs == 0.0
    assert set(np.abs(mapped.values)) == {2.0}
    layer_dofs = mapping.n_principal_layer_dofs
    first_indices = mapped.indices[:, 0]
    for layer in (1, 2, 3):
        local_first = np.unique(
            first_indices[
                (first_indices >= layer * layer_dofs)
                & (first_indices < (layer + 1) * layer_dofs)
            ]
            - layer * layer_dofs
        )
        # Every physical A/B atom in both cosets occurs as a first leg.
        np.testing.assert_array_equal(local_first, [0, 3, 6, 9])
    np.testing.assert_array_equal(
        mapped.indices,
        np.array(sorted(map(tuple, mapped.indices)), dtype=np.int64),
    )
    assert not mapped.report.symmetrized
    assert not mapped.report.truncated
    assert not mapped.report.thresholded


def test_nonsymmetric_transport_cosets_use_row_coordinate_mapping() -> None:
    transform = np.array([[2, 1, 0], [0, 1, 0], [0, 0, 1]])
    mapping = _layered_mapping(transform=transform)
    mapped = map_fc3_to_device(
        _normalized(),
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert mapping.signed_determinant == 2
    assert mapping.direct_cosets_sha256 != mapping.reciprocal_cosets_sha256
    assert mapping.n_transport_basis_atoms == 4
    assert mapped.nnz == 2 * 8 * 3
    assert mapped.report.certification_passed


def test_duplicate_shortest_images_use_fsum_and_apply_weight_once() -> None:
    mapping = _layered_mapping(transform=np.diag([2, 1, 1]))
    single = map_fc3_to_device(
        _normalized(),
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    duplicated = map_fc3_to_device(
        _normalized(duplicate_images=True),
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    np.testing.assert_array_equal(duplicated.indices, single.indices)
    np.testing.assert_array_equal(duplicated.values, single.values)
    assert duplicated.cubic_physics_sha256 == single.cubic_physics_sha256
    assert duplicated.report.duplicate_accumulation_count > 0
    assert duplicated.report.expanded_image_count == 4 * single.report.expanded_image_count


def test_finite_crop_and_padding_use_complete_cross_cell_extent() -> None:
    mapping = _layered_mapping()
    normalized = _normalized(cross_cell=True)
    mapped = map_fc3_to_device(
        normalized,
        mapping,
        n_device_layers=9,
        interaction_layers=(2, 3, 4, 5, 6),
        observation_layers=(3, 4, 5),
    )
    assert mapped.report.required_left_halo_layers == 1
    assert mapped.report.required_right_halo_layers == 1
    np.testing.assert_array_equal(
        mapped.report.pairwise_max_principal_layers,
        np.ones(3),
    )
    assert mapped.report.harmonic_buffer_layers == ((0, 1), (7, 8))
    assert mapped.report.certification_passed
    layer_dofs = mapping.n_principal_layer_dofs
    assert int(np.min(mapped.indices)) >= 2 * layer_dofs
    assert int(np.max(mapped.indices)) < 7 * layer_dofs

    insufficient = map_fc3_to_device(
        normalized,
        mapping,
        n_device_layers=9,
        interaction_layers=(2, 3, 4, 5, 6),
        observation_layers=(2, 3, 4, 5),
    )
    assert not insufficient.report.certification_passed
    assert "observation_left_padding" in insufficient.report.failure_codes
    with pytest.raises(ValueError, match="mapping report failed"):
        scale_device_fc3(insufficient, SolverScale())


def test_contact_layers_are_the_required_five_layer_harmonic_halo() -> None:
    mapped = map_fc3_to_device(
        _normalized(cross_cell=True),
        _layered_mapping(),
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert mapped.report.required_left_halo_layers == 1
    assert mapped.report.required_right_halo_layers == 1
    assert mapped.report.harmonic_buffer_layers == ((0,), (4,))
    assert mapped.report.failure_codes == ()
    assert mapped.report.certification_passed
    assert mapped.report.exact_permutation_equality


def test_two_layer_range_requires_the_minimum_nine_layer_crop() -> None:
    mapped = map_fc3_to_device(
        _normalized(two_cell=True),
        _layered_mapping(),
        n_device_layers=9,
        interaction_layers=(2, 3, 4, 5, 6),
        observation_layers=(4,),
    )
    assert mapped.report.required_left_halo_layers == 2
    assert mapped.report.required_right_halo_layers == 2
    np.testing.assert_array_equal(
        mapped.report.pairwise_max_principal_layers,
        np.full(3, 2, dtype=np.int64),
    )
    assert mapped.report.harmonic_buffer_layers == ((0, 1), (7, 8))
    assert mapped.report.failure_codes == ()
    assert mapped.report.certification_passed

    five_layer = map_fc3_to_device(
        _normalized(two_cell=True),
        _layered_mapping(),
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert not five_layer.report.certification_passed
    assert {
        "left_harmonic_buffer",
        "right_harmonic_buffer",
        "observation_left_padding",
        "observation_right_padding",
    }.issubset(five_layer.report.failure_codes)


def test_roundoff_bounded_permutation_is_certified_without_modification() -> None:
    indices = np.asarray(sorted(itertools.permutations((0, 1, 2))), dtype=np.int64)
    values = np.ones(len(indices), dtype=np.float64)
    values[0] = np.nextafter(1.0, 2.0)
    original = np.array(values, copy=True)
    contribution_l1 = np.full(len(indices), 8.0, dtype=np.float64)
    maximum, missing, passed, exact, ratio, forward_ratio = _permutation_audit(
        3,
        indices,
        values,
        contribution_l1,
        np.zeros_like(values),
    )
    assert maximum == np.spacing(1.0)
    assert missing == 0
    assert passed
    assert not exact
    assert ratio < 1.0
    assert forward_ratio < 1.0
    np.testing.assert_array_equal(values, original)

    asymmetric = np.array(values, copy=True)
    asymmetric[0] = 1.01
    _maximum, missing, passed, exact, ratio, forward_ratio = _permutation_audit(
        3,
        indices,
        asymmetric,
        contribution_l1,
        np.zeros_like(values),
    )
    assert missing == 0
    assert not passed
    assert not exact
    assert ratio > 32.0
    assert forward_ratio > 1.0


def test_translation_uncertainty_is_separate_from_arithmetic_roundoff() -> None:
    indices = np.asarray(sorted(itertools.permutations((0, 1, 2))), dtype=np.int64)
    values = np.ones(len(indices), dtype=np.float64)
    values[0] += 1.0e-12
    contribution_l1 = np.ones(len(indices), dtype=np.float64)
    translation_bounds = np.full(len(indices), 6.0e-13, dtype=np.float64)

    _maximum, missing, passed, exact, arithmetic_ratio, forward_ratio = (
        _permutation_audit(
            3,
            indices,
            values,
            contribution_l1,
            translation_bounds,
        )
    )
    assert missing == 0
    assert not exact
    assert arithmetic_ratio > 32.0
    assert forward_ratio < 1.0
    assert passed

    insufficient = np.full(len(indices), 1.0e-14, dtype=np.float64)
    _maximum, missing, passed, _exact, arithmetic_ratio, forward_ratio = (
        _permutation_audit(
            3,
            indices,
            values,
            contribution_l1,
            insufficient,
        )
    )
    assert missing == 0
    assert arithmetic_ratio > 32.0
    assert forward_ratio > 1.0
    assert not passed


def test_measured_translation_residual_is_mass_normalized_not_tolerance_tuned() -> None:
    translated, crystal, certification = _translational()
    provenance = dict(translated.provenance)
    audit = dict(provenance["redundant_translation_audit"])
    audit["maximum_absolute_residual"] = 2.0e-14
    provenance["redundant_translation_audit"] = audit
    with_residual = TranslationalFC3(
        terms=translated.terms,
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout=translated.source_layout,
        phase_convention=translated.phase_convention,
        anchor_convention=translated.anchor_convention,
        make_r0_average=translated.make_r0_average,
        all_shortest=translated.all_shortest,
        symprec=translated.symprec,
        provenance=provenance,
    )
    normalized = mass_normalize_translational_fc3(
        with_residual,
        crystal,
        source_certification=certification,
    )

    assert normalized.source_translation_max_abs == 2.0e-14
    assert all(
        term.source_translation_abs_bound > 2.0e-14
        for term in normalized.terms
    )
    assert all(
        term.source_translation_abs_bound < normalized.translation_atol
        for term in normalized.terms
    )


def test_compact_accumulator_is_bitwise_equal_to_legacy_record_lists() -> None:
    ndof = 4
    records = (
        ((0, 0, 0), (0, 0), 1.0e16),
        ((0, 1, 0), (0, 1), 4.0),
        ((1, 1, 1), (0, 2), 1.0),
        ((0, 0, 0), (1, 0), 1.0),
        ((0, 1, 0), (1, 1), -1.0),
        ((1, 1, 1), (1, 2), -1.0),
        ((0, 0, 0), (2, 0), -1.0e16),
        ((0, 1, 0), (2, 1), 2.0),
        ((2, 3, 1), (2, 2), -7.5),
    )
    # The mapper emits this global record-suffix order before its stable
    # output-key grouping.  This is precisely what the former Python record
    # tuples represented.
    ordered = tuple(sorted(records, key=lambda record: record[1]))
    keys = np.asarray(
        [
            (output[0] * ndof + output[1]) * ndof + output[2]
            for output, _suffix, _value in ordered
        ],
        dtype=np.uint64,
    )
    contributions = np.asarray(
        [value for _output, _suffix, value in ordered],
        dtype=np.float64,
    )
    (
        indices,
        values,
        contribution_l1,
        translation_bounds,
        duplicate_count,
        cancellation_count,
    ) = (
        _accumulate_compact(
            keys,
            contributions,
            np.zeros_like(contributions),
            ndof=ndof,
        )
    )

    legacy: dict[tuple[int, int, int], list[tuple[tuple[int, ...], float]]]
    legacy = defaultdict(list)
    for output, suffix, value in records:
        legacy[output].append(((*output, *suffix), value))
    expected = {
        output: math.fsum(
            value for _key, value in sorted(legacy[output])
        )
        for output in sorted(legacy)
    }
    expected = {key: value for key, value in expected.items() if value != 0.0}
    np.testing.assert_array_equal(
        indices,
        np.asarray(sorted(expected), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        values,
        np.asarray([expected[tuple(index)] for index in indices]),
    )
    assert duplicate_count == sum(len(group) - 1 for group in legacy.values())
    assert cancellation_count == 1
    expected_l1 = {
        output: math.fsum(abs(value) for _key, value in legacy[output])
        for output in expected
    }
    np.testing.assert_array_equal(
        contribution_l1,
        np.asarray([expected_l1[tuple(index)] for index in indices]),
    )
    np.testing.assert_array_equal(translation_bounds, np.zeros(len(indices)))


def test_compact_accumulation_preflight_is_linear_and_allocation_free() -> None:
    # A 120-DOF, ten-million-contribution Si-shaped crop is modeled without
    # allocating its arrays.  The compact workspace is hundreds of MB, not
    # the multi-GB Python tuple/list footprint of the former accumulator.
    preflight = _compact_accumulation_preflight(10_000_000, ndof=120)
    assert preflight.compact_payload_bytes == 240_000_000
    assert preflight.stable_order_bytes == 80_000_000
    assert preflight.stable_sort_scratch_bytes == 80_000_000
    assert preflight.maximum_final_entry_count == 120**3
    assert preflight.estimated_peak_bytes == 400_000_000
    assert preflight.geometry_pass_count == 2
    assert preflight.retained_periodic_record_count == 0
    assert preflight.retained_python_contribution_record_count == 0


def test_r0_three_leg_schedule_equals_legacy_for_symmetric_source() -> None:
    translated, crystal, certification = _translational()
    all_shortest = np.zeros((2, 2, 2), dtype=np.byte)
    r0 = TranslationalFC3(
        terms=translated.terms,
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout=translated.source_layout,
        anchor_convention=PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION,
        make_r0_average=True,
        all_shortest=all_shortest,
        symprec=1.0e-5,
        provenance=translated.provenance,
    )
    normalized_r0 = mass_normalize_translational_fc3(
        r0, crystal, source_certification=certification
    )
    mapping = _layered_mapping(transform=np.diag([2, 1, 1]))
    legacy_mapped = map_fc3_to_device(
        mass_normalize_translational_fc3(
            translated, crystal, source_certification=certification
        ),
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    r0_mapped = map_fc3_to_device(
        normalized_r0,
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    np.testing.assert_array_equal(r0_mapped.indices, legacy_mapped.indices)
    np.testing.assert_allclose(r0_mapped.values, legacy_mapped.values, rtol=0, atol=1e-15)
    assert r0_mapped.report.effective_term_count == 3 * len(r0.terms)


def test_permutation_audit_is_independent_and_never_symmetrizes() -> None:
    translated, crystal, _ = _translational()
    asymmetric_block = np.zeros((3, 3, 3))
    asymmetric_block[0, 1, 2] = 1.0
    terms = []
    for term in translated.terms:
        sign = float(np.sign(term.values[0, 0, 0]))
        terms.append(
            FC3Term(
                primitive_indices=term.primitive_indices,
                source_indices=term.source_indices,
                translated_supercell_indices=term.translated_supercell_indices,
                anchor_position=term.anchor_position,
                second_offset=term.second_offset,
                third_offset=term.third_offset,
                second_vector_index=term.second_vector_index,
                third_vector_index=term.third_vector_index,
                second_multiplicity=term.second_multiplicity,
                third_multiplicity=term.third_multiplicity,
                anchor_weight=term.anchor_weight,
                multiplicity_weight=term.multiplicity_weight,
                values=sign * asymmetric_block,
            )
        )
    asymmetric = TranslationalFC3(
        terms=tuple(terms),
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout="full",
        provenance=translated.provenance,
    )
    # This deliberately synthetic bridge report lets the mapper's independent
    # audit be tested without weakening the production raw-reader path.
    certification = FC3ValidationReport(
        dimension_valid=True,
        finite_valid=True,
        masses_valid=True,
        units_valid=True,
        mapping_valid=True,
        permutation_valid=True,
        permutation_complete=True,
        source_fc3_content_sha256=translated.provenance[
            "source_fc3_content_sha256"
        ],
    )
    normalized = mass_normalize_translational_fc3(
        asymmetric, crystal, source_certification=certification
    )
    mapped = map_fc3_to_device(
        normalized,
        _layered_mapping(),
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert not mapped.report.permutation_tolerance_passed
    assert not mapped.report.exact_permutation_equality
    assert not mapped.report.certification_passed
    assert not mapped.report.symmetrized
    assert mapped.values.size > 0


def test_compact_and_missing_source_identity_fail_closed() -> None:
    translated, crystal, certification = _translational()
    missing_identity = TranslationalFC3(
        terms=translated.terms,
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout="full",
    )
    with pytest.raises(ValueError, match="source_fc3_content_sha256"):
        mass_normalize_translational_fc3(
            missing_identity, crystal, source_certification=certification
        )

    compact = TranslationalFC3(
        terms=translated.terms,
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout="compact",
        provenance=translated.provenance,
    )
    with pytest.raises(NotImplementedError, match="paired-certification"):
        mass_normalize_translational_fc3(
            compact, crystal, source_certification=certification
        )

    wrong_mapping_identity = TranslationalFC3(
        terms=translated.terms,
        n_primitive_atoms=translated.n_primitive_atoms,
        n_supercell_atoms=translated.n_supercell_atoms,
        units=translated.units,
        mass_units=translated.mass_units,
        source_layout="full",
        provenance={
            **dict(translated.provenance),
            "fc3_crystal_mapping_content_sha256": "22" * 32,
        },
    )
    with pytest.raises(ValueError, match="mapping identity conflicts"):
        mass_normalize_translational_fc3(
            wrong_mapping_identity,
            crystal,
            source_certification=certification,
        )


def test_full_certification_is_bound_to_the_exact_raw_fc3_content() -> None:
    translated, crystal, _certification = _translational()
    stale_certification = validate_fc3(_raw_fc3(g=3.0))
    assert stale_certification.passed

    with pytest.raises(ValueError, match="does not authenticate"):
        mass_normalize_translational_fc3(
            translated,
            crystal,
            source_certification=stale_certification,
        )


def test_mapped_arrays_are_detached_readonly_and_hashes_are_deterministic() -> None:
    mapping = _layered_mapping()
    normalized = _normalized()
    left = map_fc3_to_device(
        normalized,
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    right = map_fc3_to_device(
        normalized,
        mapping,
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    assert left.content_sha256 == right.content_sha256
    assert left.report.report_sha256 == right.report.report_sha256
    assert left.cubic_physics_sha256 == right.cubic_physics_sha256
    with pytest.raises(ValueError):
        left.indices[0, 0] = 0
    with pytest.raises(ValueError):
        left.values[0] = 0.0
    with pytest.raises(TypeError):
        DeviceMappedFC3()  # type: ignore[call-arg]


def test_solver_scaling_is_separate_and_joins_sparse_metadata() -> None:
    mapped = map_fc3_to_device(
        _normalized(),
        _layered_mapping(),
        n_device_layers=5,
        interaction_layers=(1, 2, 3),
        observation_layers=(2,),
    )
    scale = SolverScale(
        omega0=4.0,
        hbar=9.0,
        metadata={"source_units": {"fc3": "J/m^3", "mass": "kg"}},
    )
    sparse = scale_device_fc3(mapped, scale)
    assert isinstance(sparse, SparseCubicVertex)
    expected_factor = np.sqrt(scale.hbar) / scale.omega0**2.5
    np.testing.assert_array_equal(sparse.indices, mapped.indices)
    np.testing.assert_allclose(
        sparse.coefficients,
        mapped.values * expected_factor,
        rtol=0,
        atol=0,
    )
    assert sparse.metadata["device_mapped_fc3_sha256"] == mapped.content_sha256
    assert (
        sparse.metadata["source_certification_sha256"]
        == mapped.report.source_certification_sha256
    )
    assert sparse.metadata["solver_scaled"] is True
