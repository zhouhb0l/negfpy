from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from negfpy.anharmonic.bulk_harmonic import BulkPhononSpectrum
from negfpy.anharmonic.bulk_three_phonon import (
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
)
from negfpy.anharmonic.phono3py_reference import (
    InteractionStrengthSample,
    Phono3pyReferenceSchemaError,
    compare_fc2_random_q,
    compare_interaction_strengths,
    compare_linewidths,
    compare_phase_space,
    compare_silicon_reference,
    read_phono3py_silicon_reference,
)


_RANDOM_QPOINTS = np.asarray(
    [
        [0.125, 0.25, -0.375],
        [0.3141592653589793, -0.2718281828459045, 0.1618033988749895],
        [-0.4375, 0.09375, 0.28125],
        [0.03125, -0.21875, 0.40625],
        [-0.1924500897298753, 1.0 / 3.0, -0.1178511301977579],
        [0.24609375, 0.37109375, -0.46484375],
        [-0.35546875, -0.14453125, 0.21484375],
        [0.419921875, -0.337890625, -0.072265625],
    ],
    dtype=float,
)
_PHASE_VALUES = np.asarray(
    [
        [0.75, 1.5, 2.25],
        [-0.125, 0.25, -0.5],
    ],
    dtype=float,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _refresh_checksum(path: Path) -> None:
    (path.parent / "SHA256SUMS").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )


def _write_reference(
    tmp_path: Path,
    *,
    include_fc2_random_q: bool = False,
    phase_compatible: bool = False,
    reference_gamma: np.ndarray | None = None,
    frequencies: np.ndarray | None = None,
    selected_bands: tuple[int, ...] = (0, 1, 2),
) -> Path:
    path = tmp_path / "si-phono3py-reference.hdf5"
    grg2bzg = np.array([1, 0], dtype=np.int64)
    # BZG index 2 is a non-primary surface image of GRG index 1.
    bzg2grg = np.array([1, 0, 1], dtype=np.int64)
    frequencies_grg = (
        np.array(
            [
                [1.0, 1.0, 2.0],
                [1.5, 1.5, 3.0],
            ]
        )
        if frequencies is None
        else np.asarray(frequencies, dtype=float)
    )
    frequencies_bzg = frequencies_grg[bzg2grg]
    gamma = (
        np.array(
            [
                [
                    [[0.5, 0.5, 0.0]],
                    [[2.5, 2.5, 0.2]],
                ]
            ],
            dtype=float,
        )
        if reference_gamma is None
        else np.asarray(reference_gamma, dtype=float)
    )
    gamma = gamma[..., list(selected_bands)]

    # phono3py route order is intentionally q1=1, then q1=0. NEGFPy's
    # synthetic candidate below stores q1=0, then q1=1.
    strength = np.zeros((2, 3, 3, 3), dtype=float)
    strength[0, 0, 0, 0] = 16.0
    strength[1, 0, 0, 0] = 8.0

    with h5py.File(path, "w") as h5:
        h5.attrs["schema_name"] = "negfpy_phono3py_si_reference"
        h5.attrs["schema_version"] = 1
        h5.attrs["phono3py_version"] = "4.4.0"
        h5.attrs["phonopy_version"] = "4.4.0"
        h5.attrs["frequency_units"] = "cyclic_THz"
        h5.attrs["gamma_units"] = "cyclic_THz_HWHM"
        h5.attrs["interaction_strength_units"] = "eV^2"
        h5.attrs["is_mesh_symmetry"] = False
        h5.attrs["is_nac"] = False
        h5.attrs["make_r0_average"] = True
        h5.attrs["cutoff_frequency_THz"] = 1.0e-4
        h5.attrs["frequency_unit_conversion_factor"] = 15.633302300230191
        h5.attrs["symmetrize_fc3q"] = False

        settings = h5.create_group("settings")
        settings.create_dataset("mesh", data=[2, 1, 1])
        settings.create_dataset("grid_points_grg", data=[0])
        settings.create_dataset("grid_points_bzg", data=[1])
        settings.create_dataset("band_indices", data=selected_bands)
        settings.create_dataset("temperatures", data=[0.0, 300.0])
        settings.create_dataset("sigmas", data=[0.1])
        settings.attrs["sigma_cutoff"] = "None"

        grid = h5.create_group("grid")
        grid.attrs["GRG_index_order"] = "x_fast"
        grid.attrs["target_grid_point_input_space"] = "GRG"
        grid.attrs["triplet_grid_point_space"] = "BZG"
        grid.create_dataset("D_diag", data=[2, 1, 1])
        grid.create_dataset("Q", data=np.eye(3, dtype=np.int64))
        grid.create_dataset("PS", data=np.zeros(3, dtype=np.int64))
        grid.create_dataset("grg2bzg", data=grg2bzg)
        grid.create_dataset("bzg2grg", data=bzg2grg)
        grid.create_dataset(
            "addresses",
            data=[[1, 0, 0], [0, 0, 0], [-1, 0, 0]],
        )

        phonons = h5.create_group("phonons")
        phonons.create_dataset("frequencies", data=frequencies_bzg)
        phonons.create_dataset(
            "eigenvectors",
            data=np.repeat(np.eye(3)[None], len(bzg2grg), axis=0),
        )
        phonons.create_dataset(
            "bz_grid_addresses",
            data=[[1, 0, 0], [0, 0, 0], [-1, 0, 0]],
        )
        phonons.create_dataset(
            "target_band_frequencies",
            data=frequencies_grg[[0]][:, list(selected_bands)],
        )

        gamma_group = h5.create_group("gamma")
        gamma_dataset = gamma_group.create_dataset("values", data=gamma)
        gamma_dataset.attrs["units"] = "cyclic_THz_HWHM"
        gamma_dataset.attrs["axis_order"] = (
            "sigma,temperature,target_grid_point,target_band"
        )

        targets = h5.create_group("target_q")
        target = targets.create_group("grg_00000000")
        target.attrs["grid_point_grg"] = 0
        target.attrs["grid_point_bzg"] = 1
        target.attrs["Nq"] = 2
        target.create_dataset(
            "triplets_bzg",
            data=[
                [1, 2, 2],  # GRG (0, 1, 1), non-primary BZG images
                [1, 1, 1],  # GRG (0, 0, 0)
            ],
        )
        target.create_dataset(
            "triplet_addresses",
            data=np.zeros((2, 3, 3), dtype=np.int64),
        )
        target.create_dataset("triplet_weights", data=[1, 1])
        target.create_dataset("triplets_map_grg", data=[0, 1])
        target.create_dataset("ir_map_grg", data=[0, 1])
        strength_dataset = target.create_dataset(
            "interaction_strength",
            data=strength[:, list(selected_bands)],
        )
        strength_dataset.attrs["units"] = "eV^2"
        strength_dataset.attrs["axis_order"] = (
            "triplet,target_band,band1,band2"
        )
        integration = target.create_dataset(
            "integration_weights",
            data=np.zeros((2, 2, len(selected_bands), 3, 3)),
        )
        integration.attrs["units"] = "1/cyclic_THz"
        integration.attrs["axis_order"] = (
            "channel,triplet,target_frequency,band1,band2"
        )
        integration.attrs["channel_0"] = (
            "Gaussian delta(omega-f1-f2): decay/sum channel"
        )
        integration.attrs["channel_1"] = (
            "Gaussian delta(omega+f1-f2) minus "
            "delta(omega-f1+f2): signed difference channel"
        )
        phase = target.create_dataset(
            "derived_phase_space_channel_sums",
            data=_PHASE_VALUES[:, list(selected_bands)],
        )
        phase.attrs["units"] = "1/cyclic_THz"
        phase.attrs["axis_order"] = "channel,target_frequency"
        phase.attrs["definition"] = (
            "sum_triplet,band1,band2("
            "triplet_weight*integration_weight)/Nq"
        )
        if phase_compatible:
            phase.attrs["cutoff_frequency_THz"] = 1.0e-4
            phase.attrs["cutoff_rule"] = (
                "strict_all_three_frequencies_above_cutoff"
            )

        if include_fc2_random_q:
            random_q = h5.create_group("fc2_random_q")
            random_q.attrs["qpoint_coordinates"] = (
                "primitive_reciprocal_fractional"
            )
            random_q.attrs["dynamical_matrix_units"] = (
                "eV/(angstrom^2*amu)"
            )
            random_q.attrs["gauge"] = "phonopy_shortest_vector"
            random_q.attrs["hermitianized"] = True
            random_q.attrs["point_selection"] = (
                "committed_literal_non_mesh_qpoints"
            )
            qpoint_dataset = random_q.create_dataset(
                "qpoints",
                data=_RANDOM_QPOINTS,
            )
            qpoint_dataset.attrs["units"] = (
                "primitive_reciprocal_fractional"
            )
            matrices = np.stack(
                [
                    np.diag(
                        [
                            1.0 + 0.1 * index,
                            4.0 + 0.1 * index,
                            9.0 + 0.1 * index,
                        ]
                    )
                    for index in range(len(_RANDOM_QPOINTS))
                ]
            ).astype(np.complex128)
            matrix_dataset = random_q.create_dataset(
                "dynamical_matrices",
                data=matrices,
            )
            matrix_dataset.attrs["units"] = "eV/(angstrom^2*amu)"

    _refresh_checksum(path)
    return path


def _candidate(
    *,
    gamma: np.ndarray | None = None,
    strength_offset: float = 0.0,
    frequencies: np.ndarray | None = None,
) -> tuple[
    BulkPhononSpectrum,
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
]:
    frequency_values = (
        np.array(
            [
                [1.0, 1.0, 2.0],
                [1.5, 1.5, 3.0],
            ]
        )
        if frequencies is None
        else np.asarray(frequencies, dtype=float)
    )
    phonons = BulkPhononSpectrum(
        qpoints=[[0.0, 0.0, 0.0], [-0.5, 0.0, 0.0]],
        dynamical_matrices=np.zeros((2, 3, 3), dtype=np.complex128),
        frequencies=frequency_values,
        eigenvectors=np.repeat(
            np.eye(3, dtype=np.complex128)[None],
            2,
            axis=0,
        ),
        frequency_factor_to_thz=1.0,
        hermitianized=True,
    )
    strengths = np.zeros((1, 2, 3, 3, 3), dtype=float)
    strengths[0, 0, :2, :2, :2] = 1.0
    strengths[0, 1, :2, :2, :2] = 2.0
    strengths[0, 0, 0, 0, 0] += strength_offset
    interactions = BulkThreePhononInteractions(
        grid_points=[0],
        band_indices=[0, 1, 2],
        q1_indices=[[0, 1]],
        q2_indices=[[0, 1]],
        reciprocal_vectors=np.zeros((1, 2, 3), dtype=np.int64),
        normal_mask=np.ones((1, 2), dtype=bool),
        matrix_elements=np.sqrt(strengths).astype(np.complex128),
        interaction_strengths=strengths,
        phonons=phonons,
        metadata={
            "mesh": (2, 1, 1),
            "make_r0_average": True,
            "interaction_strength_units": "eV^2",
            "factorial_divisor": 36.0,
            "mode_normalization_divisor": 8.0,
            "full_mesh_normalization": 0.5,
        },
    )
    gamma_values = (
        np.array(
            [
                [[0.4, 0.6, 0.5e-6]],
                [[2.0, 3.0, 0.2]],
            ]
        )
        if gamma is None
        else np.asarray(gamma, dtype=float)
    )
    linewidths = BulkThreePhononLinewidths(
        temperatures=[0.0, 300.0],
        grid_points=[0],
        band_indices=[0, 1, 2],
        gamma_thz_hwhm=gamma_values,
        gamma_normal_thz_hwhm=gamma_values,
        gamma_umklapp_thz_hwhm=np.zeros_like(gamma_values),
        gamma_angular_fwhm=4.0 * np.pi * gamma_values,
        decay_phase_space=_PHASE_VALUES[[0]],
        difference_phase_space=_PHASE_VALUES[[1]],
        interactions=interactions,
        metadata={
            "sigma_thz": 0.1,
            "cutoff_frequency_thz": 1.0e-4,
            "degenerate_tolerance_thz": 1.0e-4,
            "gamma_thz_semantics": "cyclic-frequency HWHM",
            "gamma_angular_fwhm_conversion": "4*pi*gamma_thz_hwhm",
            "phase_space_units": "1/cyclic_THz",
            "phase_space_definition": (
                "sum_triplet,band1,band2("
                "triplet_weight*integration_weight)/Nq"
            ),
            "phase_space_cutoff_rule": (
                "strict_all_three_frequencies_above_cutoff"
            ),
            "phase_space_channel_0": (
                "Gaussian delta(omega-f1-f2): decay/sum channel"
            ),
            "phase_space_channel_1": (
                "Gaussian delta(omega+f1-f2) minus "
                "delta(omega-f1+f2): signed difference channel"
            ),
            "phase_space_normalization": "complete_full_mesh_1_over_Nq",
            "sigma_cutoff": None,
        },
    )
    return phonons, interactions, linewidths


def _fc2_candidate(reference) -> BulkPhononSpectrum:
    assert reference.fc2_random_qpoints is not None
    assert reference.fc2_dynamical_matrices is not None
    eigenvalues, eigenvectors = np.linalg.eigh(
        reference.fc2_dynamical_matrices
    )
    return BulkPhononSpectrum(
        qpoints=reference.fc2_random_qpoints,
        dynamical_matrices=reference.fc2_dynamical_matrices,
        frequencies=np.sqrt(np.abs(eigenvalues)),
        eigenvectors=eigenvectors,
        frequency_factor_to_thz=1.0,
        hermitianized=True,
    )


def _select_candidate_band_zero(
    interactions: BulkThreePhononInteractions,
    linewidths: BulkThreePhononLinewidths,
) -> tuple[BulkThreePhononInteractions, BulkThreePhononLinewidths]:
    selected_interactions = BulkThreePhononInteractions(
        grid_points=interactions.grid_points,
        band_indices=[0],
        q1_indices=interactions.q1_indices,
        q2_indices=interactions.q2_indices,
        reciprocal_vectors=interactions.reciprocal_vectors,
        normal_mask=interactions.normal_mask,
        matrix_elements=interactions.matrix_elements[:, :, [0]],
        interaction_strengths=interactions.interaction_strengths[:, :, [0]],
        phonons=interactions.phonons,
        metadata=interactions.metadata,
    )
    selected_linewidths = BulkThreePhononLinewidths(
        temperatures=linewidths.temperatures,
        grid_points=linewidths.grid_points,
        band_indices=[0],
        gamma_thz_hwhm=linewidths.gamma_thz_hwhm[:, :, [0]],
        gamma_normal_thz_hwhm=linewidths.gamma_normal_thz_hwhm[:, :, [0]],
        gamma_umklapp_thz_hwhm=linewidths.gamma_umklapp_thz_hwhm[:, :, [0]],
        gamma_angular_fwhm=linewidths.gamma_angular_fwhm[:, :, [0]],
        decay_phase_space=linewidths.decay_phase_space[:, [0]],
        difference_phase_space=linewidths.difference_phase_space[:, [0]],
        interactions=selected_interactions,
        metadata=linewidths.metadata,
    )
    return selected_interactions, selected_linewidths


def test_reader_requires_checksum_and_reports_current_schema_gaps(tmp_path) -> None:
    path = _write_reference(tmp_path)

    reference = read_phono3py_silicon_reference(path)

    np.testing.assert_array_equal(reference.grg2bzg, [1, 0])
    np.testing.assert_array_equal(reference.bzg2grg, [1, 0, 1])
    assert reference.make_r0_average is True
    codes = {issue.code for issue in reference.schema_issues}
    assert "missing_fc2_random_q" in codes
    assert "phase_space_cutoff_semantics_missing" in codes
    assert "phase_space_cutoff_rule_missing" in codes

    (tmp_path / "SHA256SUMS").write_text(
        f"{'0' * 64}  {path.name}\n",
        encoding="utf-8",
    )
    with pytest.raises(Phono3pyReferenceSchemaError, match="SHA256SUMS"):
        read_phono3py_silicon_reference(path)


def test_fc2_random_q_gate_passes_and_detects_matrix_error(tmp_path) -> None:
    path = _write_reference(tmp_path, include_fc2_random_q=True)
    reference = read_phono3py_silicon_reference(path)
    candidate = _fc2_candidate(reference)

    passing = compare_fc2_random_q(reference, candidate)

    assert passing.available
    assert passing.passed
    assert passing.max_absolute_error == 0.0

    changed = np.array(candidate.dynamical_matrices, copy=True)
    changed[1, 0, 0] += 2.0e-9
    failing_candidate = BulkPhononSpectrum(
        qpoints=candidate.qpoints,
        dynamical_matrices=changed,
        frequencies=candidate.frequencies,
        eigenvectors=candidate.eigenvectors,
        frequency_factor_to_thz=1.0,
        hermitianized=True,
    )
    failing = compare_fc2_random_q(reference, failing_candidate)
    assert not failing.passed
    assert failing.max_absolute_error == pytest.approx(2.0e-9)


def test_fc2_random_q_gate_is_absolute_not_scale_relative(tmp_path) -> None:
    path = _write_reference(tmp_path, include_fc2_random_q=True)
    reference = read_phono3py_silicon_reference(path)
    candidate = _fc2_candidate(reference)
    changed = np.array(candidate.dynamical_matrices, copy=True)
    changed[0, 2, 2] += 5.0e-10
    changed_candidate = BulkPhononSpectrum(
        qpoints=candidate.qpoints,
        dynamical_matrices=changed,
        frequencies=candidate.frequencies,
        eigenvectors=candidate.eigenvectors,
        frequency_factor_to_thz=1.0,
        hermitianized=True,
    )

    result = compare_fc2_random_q(reference, changed_candidate)

    assert result.max_absolute_error == pytest.approx(5.0e-10)
    assert result.max_scaled_error is not None
    assert result.max_scaled_error < result.tolerance
    assert not result.passed


def test_fc2_reader_rejects_uncommitted_random_q_selection(tmp_path) -> None:
    path = _write_reference(tmp_path, include_fc2_random_q=True)
    with h5py.File(path, "r+") as h5:
        h5["fc2_random_q/qpoints"][0, 0] += 1.0e-3
    _refresh_checksum(path)

    with pytest.raises(
        Phono3pyReferenceSchemaError,
        match="committed independent FC2",
    ):
        read_phono3py_silicon_reference(path)


def test_missing_fc2_random_q_is_unavailable_not_reconstructed_from_modes(
    tmp_path,
) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))

    result = compare_fc2_random_q(reference, None)

    assert not result.available
    assert not result.passed
    assert [issue.code for issue in result.issues] == ["missing_fc2_random_q"]


def test_comparison_rejects_reference_changed_after_loading(tmp_path) -> None:
    path = _write_reference(tmp_path)
    reference = read_phono3py_silicon_reference(path)
    with path.open("ab") as stream:
        stream.write(b"changed")

    with pytest.raises(Phono3pyReferenceSchemaError, match="changed"):
        compare_fc2_random_q(reference, None)


def test_interaction_comparison_maps_grg_to_bzg_and_sums_degenerate_blocks(
    tmp_path,
) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, interactions, _ = _candidate()
    assert reference.bzg2grg.size > reference.n_qpoints
    assert reference.bzg2grg[2] == 1

    result = compare_interaction_strengths(
        reference,
        interactions,
        [
            InteractionStrengthSample(0, 0, 0, 0, 0),
            InteractionStrengthSample(0, 1, 1, 1, 1),
        ],
    )

    assert result.available
    assert result.complete
    assert result.passed
    assert [block.q1_grid_point for block in result.blocks] == [0, 1]
    assert [block.q2_grid_point for block in result.blocks] == [0, 1]
    assert [block.target_bands for block in result.blocks] == [
        (0, 1),
        (0, 1),
    ]
    assert [block.reference_sum_ev2 for block in result.blocks] == [8.0, 16.0]
    assert [block.candidate_sum_ev2 for block in result.blocks] == [8.0, 16.0]


def test_frequency_matching_canonicalizes_subcutoff_acoustic_noise(
    tmp_path,
) -> None:
    reference_frequencies = np.array(
        [
            [-0.75e-4, 0.10e-4, 2.0],
            [1.5, 1.5, 3.0],
        ]
    )
    candidate_frequencies = np.array(
        [
            [0.75e-4, -0.10e-4, 2.0],
            [1.5, 1.5, 3.0],
        ]
    )
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, frequencies=reference_frequencies)
    )
    _, interactions, linewidths = _candidate(
        frequencies=candidate_frequencies
    )

    interaction_result = compare_interaction_strengths(
        reference,
        interactions,
        [(0, 0, 0, 0, 0)],
    )
    linewidth_result = compare_linewidths(reference, linewidths)

    assert interaction_result.passed
    assert linewidth_result.passed


def test_frequency_matching_rejects_one_mode_crossing_cutoff(tmp_path) -> None:
    cutoff = 1.0e-4
    reference_frequencies = np.array(
        [
            [cutoff, 1.0, 2.0],
            [1.5, 1.5, 3.0],
        ]
    )
    candidate_frequencies = np.array(
        [
            [np.nextafter(cutoff, np.inf), 1.0, 2.0],
            [1.5, 1.5, 3.0],
        ]
    )
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, frequencies=reference_frequencies)
    )
    _, interactions, linewidths = _candidate(
        frequencies=candidate_frequencies
    )

    with pytest.raises(ValueError, match="frequencies differ"):
        compare_interaction_strengths(
            reference,
            interactions,
            [(0, 0, 0, 0, 0)],
        )
    with pytest.raises(ValueError, match="frequencies differ"):
        compare_linewidths(reference, linewidths)


def test_frequency_matching_rejects_active_mode_beyond_tolerance(
    tmp_path,
) -> None:
    reference_frequencies = np.array(
        [
            [-1.0, 1.0, 2.0],
            [1.5, 1.5, 3.0],
        ]
    )
    candidate_frequencies = reference_frequencies.copy()
    candidate_frequencies[0, 0] -= 2.0e-8
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, frequencies=reference_frequencies)
    )
    _, interactions, linewidths = _candidate(
        frequencies=candidate_frequencies
    )

    with pytest.raises(ValueError, match="frequencies differ"):
        compare_interaction_strengths(
            reference,
            interactions,
            [(0, 0, 0, 0, 0)],
        )
    with pytest.raises(ValueError, match="frequencies differ"):
        compare_linewidths(reference, linewidths)


def test_interaction_relative_gate_detects_subspace_sum_error(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, interactions, _ = _candidate(strength_offset=1.0e-4)

    result = compare_interaction_strengths(
        reference,
        interactions,
        [(0, 0, 0, 0, 0)],
    )

    assert not result.passed
    assert result.max_relative_error == pytest.approx(1.0e-4 / 8.0)


def test_partial_degenerate_target_selection_is_a_schema_gap(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, selected_bands=(0,))
    )
    _, full_interactions, full_linewidths = _candidate()
    interactions, linewidths = _select_candidate_band_zero(
        full_interactions,
        full_linewidths,
    )

    interaction_result = compare_interaction_strengths(
        reference,
        interactions,
        [(0, 0, 0, 0, 0)],
    )
    linewidth_result = compare_linewidths(reference, linewidths)

    assert not interaction_result.complete
    assert not interaction_result.passed
    assert [issue.code for issue in interaction_result.issues] == [
        "incomplete_reference_target_subspace"
    ]
    assert not linewidth_result.complete
    assert not linewidth_result.passed
    assert [issue.code for issue in linewidth_result.issues] == [
        "incomplete_linewidth_target_subspace"
    ]


def test_linewidth_gate_uses_degenerate_trace_and_small_absolute_tolerance(
    tmp_path,
) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, _, linewidths = _candidate()

    result = compare_linewidths(reference, linewidths)

    assert result.available
    assert result.complete
    assert result.passed
    assert result.median_relative_error == 0.0
    assert result.maximum_relative_error == 0.0
    assert result.maximum_small_absolute_error_thz == pytest.approx(0.5e-6)
    degenerate = [
        item
        for item in result.subspaces
        if item.temperature_k == 0.0 and item.bands == (0, 1)
    ]
    assert len(degenerate) == 1
    assert degenerate[0].reference_sum_thz_hwhm == 1.0
    assert degenerate[0].candidate_sum_thz_hwhm == 1.0


def test_linewidth_relative_and_absolute_gates_fail_independently(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, _, relative_bad = _candidate(
        gamma=np.array(
            [
                [[0.4, 0.7, 0.0]],
                [[2.0, 3.0, 0.2]],
            ]
        )
    )
    relative_result = compare_linewidths(reference, relative_bad)
    assert not relative_result.passed
    assert relative_result.maximum_relative_error == pytest.approx(0.1)

    _, _, absolute_bad = _candidate(
        gamma=np.array(
            [
                [[0.4, 0.6, 1.1e-6]],
                [[2.0, 3.0, 0.2]],
            ]
        )
    )
    absolute_result = compare_linewidths(reference, absolute_bad)
    assert not absolute_result.passed
    assert absolute_result.maximum_small_absolute_error_thz == pytest.approx(
        1.1e-6
    )


def test_linewidth_median_gate_is_independent_of_maximum_gate(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, _, linewidths = _candidate(
        gamma=np.array(
            [
                [[0.51, 0.51, 0.0]],
                [[2.55, 2.55, 0.204]],
            ]
        )
    )

    result = compare_linewidths(reference, linewidths)

    assert not result.passed
    assert result.median_relative_error == pytest.approx(0.02)
    assert result.maximum_relative_error == pytest.approx(0.02)
    assert result.maximum_relative_error < result.maximum_relative_limit
    assert result.median_relative_error > result.median_relative_limit


def test_small_degenerate_modes_use_per_mode_absolute_gate(tmp_path) -> None:
    reference_gamma = np.array(
        [
            [
                [[0.6e-6, 0.6e-6, 0.0]],
                [[2.5, 2.5, 0.2]],
            ]
        ]
    )
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, reference_gamma=reference_gamma)
    )
    _, _, linewidths = _candidate(
        gamma=np.array(
            [
                [[1.4e-6, 1.4e-6, 0.5e-6]],
                [[2.5, 2.5, 0.2]],
            ]
        )
    )

    result = compare_linewidths(reference, linewidths)

    assert result.passed
    small_pair = next(
        item
        for item in result.subspaces
        if item.temperature_k == 0.0 and item.bands == (0, 1)
    )
    assert not small_pair.used_relative_gate
    assert small_pair.reference_sum_thz_hwhm == pytest.approx(1.2e-6)
    assert small_pair.reference_mean_thz_hwhm == pytest.approx(0.6e-6)
    assert small_pair.mean_absolute_error_thz == pytest.approx(0.8e-6)


def test_linewidth_gate_verifies_angular_fwhm_conversion(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, interactions, linewidths = _candidate()
    invalid = BulkThreePhononLinewidths(
        temperatures=linewidths.temperatures,
        grid_points=linewidths.grid_points,
        band_indices=linewidths.band_indices,
        gamma_thz_hwhm=linewidths.gamma_thz_hwhm,
        gamma_normal_thz_hwhm=linewidths.gamma_normal_thz_hwhm,
        gamma_umklapp_thz_hwhm=linewidths.gamma_umklapp_thz_hwhm,
        gamma_angular_fwhm=2.0 * np.pi * linewidths.gamma_thz_hwhm,
        decay_phase_space=linewidths.decay_phase_space,
        difference_phase_space=linewidths.difference_phase_space,
        interactions=interactions,
        metadata=linewidths.metadata,
    )

    with pytest.raises(ValueError, match="4\\*pi"):
        compare_linewidths(reference, invalid)


def test_phase_space_is_refused_without_matching_cutoff_definition(
    tmp_path,
) -> None:
    reference = read_phono3py_silicon_reference(_write_reference(tmp_path))
    _, _, linewidths = _candidate()

    result = compare_phase_space(reference, linewidths)

    assert not result.available
    assert not result.passed
    assert {
        issue.code for issue in result.issues
    } == {
        "phase_space_cutoff_semantics_missing",
        "phase_space_cutoff_rule_missing",
    }


def test_phase_space_compares_when_all_definitions_are_explicitly_equal(
    tmp_path,
) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, phase_compatible=True)
    )
    _, _, linewidths = _candidate()

    result = compare_phase_space(reference, linewidths)

    assert result.available
    assert result.passed
    assert result.max_absolute_error == 0.0


def test_phase_space_gate_detects_numerical_disagreement(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, phase_compatible=True)
    )
    _, interactions, linewidths = _candidate()
    changed = BulkThreePhononLinewidths(
        temperatures=linewidths.temperatures,
        grid_points=linewidths.grid_points,
        band_indices=linewidths.band_indices,
        gamma_thz_hwhm=linewidths.gamma_thz_hwhm,
        gamma_normal_thz_hwhm=linewidths.gamma_normal_thz_hwhm,
        gamma_umklapp_thz_hwhm=linewidths.gamma_umklapp_thz_hwhm,
        gamma_angular_fwhm=linewidths.gamma_angular_fwhm,
        decay_phase_space=linewidths.decay_phase_space + 2.0e-10,
        difference_phase_space=linewidths.difference_phase_space,
        interactions=interactions,
        metadata=linewidths.metadata,
    )

    result = compare_phase_space(reference, changed)

    assert result.available
    assert not result.passed
    assert result.max_absolute_error == pytest.approx(2.0e-10)


def test_phase_space_gate_authenticates_candidate_definition(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, phase_compatible=True)
    )
    _, interactions, linewidths = _candidate()
    metadata = dict(linewidths.metadata)
    metadata["phase_space_normalization"] = "missing_1_over_Nq"
    invalid = BulkThreePhononLinewidths(
        temperatures=linewidths.temperatures,
        grid_points=linewidths.grid_points,
        band_indices=linewidths.band_indices,
        gamma_thz_hwhm=linewidths.gamma_thz_hwhm,
        gamma_normal_thz_hwhm=linewidths.gamma_normal_thz_hwhm,
        gamma_umklapp_thz_hwhm=linewidths.gamma_umklapp_thz_hwhm,
        gamma_angular_fwhm=linewidths.gamma_angular_fwhm,
        decay_phase_space=linewidths.decay_phase_space,
        difference_phase_space=linewidths.difference_phase_space,
        interactions=interactions,
        metadata=metadata,
    )

    with pytest.raises(ValueError, match="phase_space_normalization"):
        compare_phase_space(reference, invalid)


def test_combined_gate_requires_definition_matched_phase_space(tmp_path) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path, include_fc2_random_q=True)
    )
    fc2 = _fc2_candidate(reference)
    _, interactions, linewidths = _candidate()

    result = compare_silicon_reference(
        reference,
        fc2_spectrum=fc2,
        interactions=interactions,
        linewidths=linewidths,
        interaction_samples=[(0, 0, 0, 0, 0)],
    )

    assert result.fc2.passed
    assert result.interactions.passed
    assert result.linewidths.passed
    assert not result.phase_space.available
    assert not result.passed

    compatible_dir = tmp_path / "compatible"
    compatible_dir.mkdir()
    compatible_reference = read_phono3py_silicon_reference(
        _write_reference(
            compatible_dir,
            include_fc2_random_q=True,
            phase_compatible=True,
        )
    )
    compatible_fc2 = _fc2_candidate(compatible_reference)
    compatible = compare_silicon_reference(
        compatible_reference,
        fc2_spectrum=compatible_fc2,
        interactions=interactions,
        linewidths=linewidths,
        interaction_samples=[(0, 0, 0, 0, 0)],
    )
    assert compatible.phase_space.passed
    assert compatible.passed
