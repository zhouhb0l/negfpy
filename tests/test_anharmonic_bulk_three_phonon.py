from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.bulk_three_phonon import (
    BulkThreePhononProblem,
    _average_selected_degenerate,
    _mode_space_matrix_elements,
    compute_three_phonon_interactions,
    compute_three_phonon_linewidths,
    phono3py_imag_self_energy_unit_factor,
    phono3py_interaction_unit_factor,
)
from negfpy.anharmonic.crystal import CrystalCell, CrystalMapping
from negfpy.anharmonic.fc2 import FC2Data
from negfpy.anharmonic.fc3 import FC3Data
from negfpy.anharmonic.momentum import (
    PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR,
    PHONO3PY_4_4_NIGGLI_TOLERANCE,
    UniformBZMesh,
)
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational


def _mapping(purpose: str) -> CrystalMapping:
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
        p2s=[0],
        s2p=[0],
        p2p=[0],
        shortest_vectors=[[0.0, 0.0, 0.0]],
        multiplicity=[[[1, 0]]],
        atomic_permutations=[[0]],
        translation_permutations=[[0]],
        purpose=purpose,
    )


def _problem(
    *,
    coupling: float = 0.2,
    sigma: float = 0.05,
    acoustic_mode: bool = False,
    mesh: tuple[int, int, int] = (1, 1, 1),
) -> BulkThreePhononProblem:
    fc2 = np.zeros((1, 1, 3, 3))
    fc2[0, 0] = np.diag([0.0 if acoustic_mode else 1.0, 4.0, 9.0])
    fc2_data = FC2Data(
        values=fc2,
        masses=[1.0],
        p2s_map=[0],
        s2p_map=[0],
    )
    fc3 = np.zeros((1, 1, 1, 3, 3, 3))
    for axes in (
        (0, 1, 2),
        (0, 2, 1),
        (1, 0, 2),
        (1, 2, 0),
        (2, 0, 1),
        (2, 1, 0),
    ):
        fc3[(0, 0, 0) + axes] = coupling
    fc3_data = FC3Data(
        values=fc3,
        masses=[1.0],
        p2s_map=[0],
        s2p_map=[0],
    )
    mapping3 = _mapping("fc3")
    return BulkThreePhononProblem(
        fc2=fc2_data,
        fc2_mapping=_mapping("fc2"),
        fc3=convert_fc3_to_translational(fc3_data, mapping3),
        fc3_mapping=mapping3,
        mesh=UniformBZMesh(mesh),
        sigma_thz=sigma,
        bz_reduced_reciprocal_lattice=np.eye(3),
        cutoff_frequency_thz=1.0e-8,
        frequency_factor_to_thz=1.0,
    )


def test_pinned_unit_factors_match_phono3py_4_4_values() -> None:
    assert phono3py_interaction_unit_factor(27) == pytest.approx(
        0.0001327959698104446,
        rel=1.0e-15,
    )
    assert phono3py_imag_self_energy_unit_factor() == pytest.approx(
        3306215.697005045,
        rel=1.0e-15,
    )


def test_three_mode_matrix_element_and_fixed_normalizations() -> None:
    coupling = 0.2
    result = compute_three_phonon_interactions(
        _problem(coupling=coupling),
        grid_points=[0],
        band_indices=[2],
    )

    expected = coupling / np.sqrt(3.0 * 1.0 * 2.0)
    assert result.matrix_elements.shape == (1, 1, 1, 3, 3)
    assert result.matrix_elements[0, 0, 0, 0, 1] == pytest.approx(expected)
    assert result.matrix_elements[0, 0, 0, 1, 0] == pytest.approx(expected)
    assert result.interaction_strengths[0, 0, 0, 0, 1] == pytest.approx(
        expected**2 * phono3py_interaction_unit_factor(1)
    )
    assert result.metadata["factorial_divisor"] == 36.0
    assert result.metadata["mode_normalization_divisor"] == 8.0
    assert result.metadata["full_mesh_normalization"] == 1.0
    assert result.metadata["bz_grid_tolerance_factor"] == (
        PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR
    )
    assert result.metadata["bz_niggli_tolerance"] == (
        PHONO3PY_4_4_NIGGLI_TOLERANCE
    )
    assert result.metadata["bz_reduction_source"] == (
        "problem_explicit_unimodular_equivalence_only"
    )
    assert not result.metadata["bz_reduction_verified_niggli"]
    assert result.metadata["bz_reduced_reciprocal_basis_convention"] == (
        "3x3 column vectors in Cartesian reciprocal coordinates"
    )
    np.testing.assert_array_equal(
        result.metadata["bz_reduced_reciprocal_lattice"],
        np.eye(3),
    )
    assert not result.metadata[
        "bz_reduced_reciprocal_lattice"
    ].flags.writeable
    assert result.normal_mask[0, 0]
    assert not result.matrix_elements.flags.writeable
    assert not result.interaction_strengths.flags.writeable


def test_complex_modes_use_phono3py_nonconjugated_eigenvectors() -> None:
    masses = np.array([4.0])
    frequencies = (
        np.array([1.0, 3.0, 5.0]),
        np.array([1.5, 2.5, 4.0]),
        np.array([2.0, 3.5, 4.5]),
    )
    phases = (
        np.exp(1j * np.array([0.2, 0.7, -0.4])),
        np.exp(1j * np.array([-0.3, 0.5, 1.1])),
        np.exp(1j * np.array([0.9, -0.8, 0.6])),
    )
    eigenvectors = tuple(np.diag(values) for values in phases)
    reciprocal_fc3 = np.zeros((1, 1, 1, 3, 3, 3), dtype=np.complex128)
    reciprocal_fc3[0, 0, 0, 1, 2, 0] = 2.0 + 3.0j

    result = _mode_space_matrix_elements(
        reciprocal_fc3,
        frequencies,
        eigenvectors,
        masses,
        np.array([1], dtype=np.int64),
        0.1,
    )
    expected = (
        phases[0][1]
        * phases[1][2]
        * phases[2][0]
        * (2.0 + 3.0j)
        / np.sqrt(masses[0] ** 3)
        / np.sqrt(frequencies[0][1] * frequencies[1][2] * frequencies[2][0])
    )

    assert np.count_nonzero(result) == 1
    assert result[0, 2, 0] == pytest.approx(expected, rel=1.0e-14)
    assert result[0, 2, 0] != pytest.approx(expected.conjugate())


def test_degenerate_subspace_strength_sum_is_gauge_invariant() -> None:
    rng = np.random.default_rng(7129)
    reciprocal_fc3 = rng.normal(size=(1, 1, 1, 3, 3, 3)) + 1j * rng.normal(
        size=(1, 1, 1, 3, 3, 3)
    )
    frequencies = (np.ones(3), np.ones(3), np.ones(3))
    identity = tuple(np.eye(3, dtype=np.complex128) for _ in range(3))
    rotations = []
    for _ in range(3):
        raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        unitary, _ = np.linalg.qr(raw)
        rotations.append(unitary)

    reference = _mode_space_matrix_elements(
        reciprocal_fc3,
        frequencies,
        identity,
        np.array([2.0]),
        np.arange(3, dtype=np.int64),
        0.1,
    )
    rotated = _mode_space_matrix_elements(
        reciprocal_fc3,
        frequencies,
        tuple(rotations),
        np.array([2.0]),
        np.arange(3, dtype=np.int64),
        0.1,
    )

    assert np.sum(np.abs(rotated) ** 2) == pytest.approx(
        np.sum(np.abs(reference) ** 2),
        rel=2.0e-15,
    )


def test_degeneracy_average_matches_selected_subspace_semantics() -> None:
    frequencies = np.array([1.0, 1.00005, 1.00012, 2.0])
    selected = np.array([0, 2, 3], dtype=np.int64)
    averaged = _average_selected_degenerate(
        np.array([2.0, 8.0, 20.0]),
        frequencies,
        selected,
        1.0e-4,
    )
    # Bands 0--2 form one transitively connected degenerate set. As in
    # phono3py, averaging is over the members present in band_indices.
    np.testing.assert_array_equal(averaged, [5.0, 5.0, 20.0])

    partial = _average_selected_degenerate(
        np.array([2.0, 20.0]),
        frequencies,
        np.array([0, 3], dtype=np.int64),
        1.0e-4,
    )
    np.testing.assert_array_equal(partial, [2.0, 20.0])


def test_full_mesh_interaction_normalization_is_applied_exactly_once() -> None:
    one_q = compute_three_phonon_interactions(
        _problem(mesh=(1, 1, 1)),
        grid_points=[0],
        band_indices=[2],
    )
    three_q = compute_three_phonon_interactions(
        _problem(mesh=(3, 1, 1)),
        grid_points=[0],
        band_indices=[2],
    )

    np.testing.assert_allclose(
        three_q.interaction_strengths[0],
        np.repeat(
            one_q.interaction_strengths[0, 0][None, ...] / 3.0,
            3,
            axis=0,
        ),
        rtol=1.0e-15,
        atol=0.0,
    )
    one_gamma = compute_three_phonon_linewidths(
        _problem(mesh=(1, 1, 1)),
        temperatures=[0.0, 300.0],
        grid_points=[0],
        band_indices=[2],
    )
    three_gamma = compute_three_phonon_linewidths(
        _problem(mesh=(3, 1, 1)),
        temperatures=[0.0, 300.0],
        grid_points=[0],
        band_indices=[2],
    )
    np.testing.assert_allclose(
        three_gamma.gamma_thz_hwhm,
        one_gamma.gamma_thz_hwhm,
        rtol=2.0e-15,
        atol=0.0,
    )


def test_zero_temperature_spontaneous_decay_matches_explicit_sum() -> None:
    coupling = 0.2
    sigma = 0.05
    result = compute_three_phonon_linewidths(
        _problem(coupling=coupling, sigma=sigma),
        temperatures=[0.0],
        grid_points=[0],
        band_indices=[2],
    )
    gaussian_zero = 1.0 / np.sqrt(2.0 * np.pi) / sigma
    element_squared = coupling**2 / (3.0 * 1.0 * 2.0)
    expected = (
        2.0
        * gaussian_zero
        * element_squared
        * phono3py_interaction_unit_factor(1)
        * phono3py_imag_self_energy_unit_factor()
    )

    assert result.gamma_thz_hwhm[0, 0, 0] == pytest.approx(expected)
    assert result.gamma_normal_thz_hwhm[0, 0, 0] == pytest.approx(expected)
    assert result.gamma_umklapp_thz_hwhm[0, 0, 0] == 0.0
    assert result.gamma_angular_fwhm[0, 0, 0] == pytest.approx(
        4.0 * np.pi * expected
    )
    assert result.gamma_thz_hwhm[0, 0, 0] > 0.0


def test_finite_temperature_difference_channels_match_explicit_gaussians() -> None:
    coupling = 0.2
    sigma = 0.35
    temperature = 300.0
    result = compute_three_phonon_linewidths(
        _problem(coupling=coupling, sigma=sigma),
        temperatures=[temperature],
        grid_points=[0],
        band_indices=[0],
    )

    def gaussian(value: float) -> float:
        return (
            np.exp(-(value**2) / (2.0 * sigma**2))
            / np.sqrt(2.0 * np.pi)
            / sigma
        )

    def occupation(frequency: float) -> float:
        return 1.0 / np.expm1(
            0.00413566733
            * frequency
            / (8.617338256808316e-5 * temperature)
        )

    n2 = occupation(2.0)
    n3 = occupation(3.0)
    channel_23 = (
        (n2 + n3 + 1.0) * gaussian(1.0 - 2.0 - 3.0)
        + (n2 - n3)
        * (
            gaussian(1.0 + 2.0 - 3.0)
            - gaussian(1.0 - 2.0 + 3.0)
        )
    )
    # The (3, 2) internal ordering gives the same contribution independently.
    expected = (
        2.0
        * channel_23
        * coupling**2
        / (1.0 * 2.0 * 3.0)
        * phono3py_interaction_unit_factor(1)
        * phono3py_imag_self_energy_unit_factor()
    )
    assert result.gamma_thz_hwhm[0, 0, 0] == pytest.approx(
        expected,
        rel=2.0e-15,
    )


def test_finite_temperature_masks_acoustic_modes_before_bose_evaluation() -> None:
    with np.errstate(
        divide="raise",
        invalid="raise",
        over="raise",
        under="ignore",
    ):
        result = compute_three_phonon_linewidths(
            _problem(acoustic_mode=True),
            temperatures=[300.0],
            grid_points=[0],
            band_indices=[1, 2],
        )

    assert np.all(np.isfinite(result.gamma_thz_hwhm))
    # Every nonzero cubic component in this fixture contains the acoustic
    # Cartesian mode. Phono3py excludes that mode at the interaction cutoff,
    # so no residual 0 * n_B(0) contribution is permitted.
    np.testing.assert_array_equal(result.gamma_thz_hwhm, 0.0)


def test_interactions_are_even_in_fc3_sign_and_quadratic_in_scale() -> None:
    positive = compute_three_phonon_interactions(
        _problem(coupling=0.2),
        grid_points=[0],
        band_indices=[2],
    )
    negative = compute_three_phonon_interactions(
        _problem(coupling=-0.2),
        grid_points=[0],
        band_indices=[2],
    )
    half = compute_three_phonon_interactions(
        _problem(coupling=0.1),
        grid_points=[0],
        band_indices=[2],
    )

    np.testing.assert_allclose(
        positive.interaction_strengths,
        negative.interaction_strengths,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        half.interaction_strengths,
        positive.interaction_strengths / 4.0,
        rtol=1.0e-15,
        atol=0.0,
    )


def test_selectors_are_canonical_and_temperature_validation_is_strict() -> None:
    problem = _problem()
    result = compute_three_phonon_interactions(
        problem,
        grid_points=[0],
        band_indices=[2, 0],
    )
    np.testing.assert_array_equal(result.band_indices, [0, 2])

    with pytest.raises(ValueError, match="duplicate"):
        compute_three_phonon_interactions(
            problem,
            grid_points=[0, 0],
        )
    with pytest.raises(ValueError, match="nonnegative"):
        compute_three_phonon_linewidths(
            problem,
            temperatures=[-1.0],
            grid_points=[0],
        )


def test_selector_subsets_equal_canonical_slices_of_full_mesh_results() -> None:
    problem = _problem(mesh=(3, 1, 1))
    full_interactions = compute_three_phonon_interactions(problem)
    subset_interactions = compute_three_phonon_interactions(
        problem,
        grid_points=[2, 0],
        band_indices=[2, 0],
    )

    np.testing.assert_array_equal(subset_interactions.grid_points, [0, 2])
    np.testing.assert_array_equal(subset_interactions.band_indices, [0, 2])
    expected_strengths = full_interactions.interaction_strengths[[0, 2]][
        :, :, [0, 2]
    ]
    expected_elements = full_interactions.matrix_elements[[0, 2]][
        :, :, [0, 2]
    ]
    np.testing.assert_array_equal(
        subset_interactions.interaction_strengths,
        expected_strengths,
    )
    np.testing.assert_array_equal(
        subset_interactions.matrix_elements,
        expected_elements,
    )

    full_linewidths = compute_three_phonon_linewidths(
        problem,
        temperatures=[0.0, 300.0],
    )
    subset_linewidths = compute_three_phonon_linewidths(
        problem,
        temperatures=[0.0, 300.0],
        grid_points=[2, 0],
        band_indices=[2, 0],
    )
    expected_gamma = full_linewidths.gamma_thz_hwhm[:, [0, 2]][
        :, :, [0, 2]
    ]
    np.testing.assert_array_equal(
        subset_linewidths.gamma_thz_hwhm,
        expected_gamma,
    )
