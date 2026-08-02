from __future__ import annotations

import os
import stat
from dataclasses import replace

import numpy as np
import pytest

import negfpy.bulk.archive as rta_archive_module
import negfpy.bulk.rate_archive as rate_archive_module
from negfpy.anharmonic.bulk_harmonic import (
    BulkGroupVelocities,
    BulkPhononSpectrum,
)
from negfpy.anharmonic.bulk_observables import (
    compute_phonon_dos,
    interpolate_band_path,
)
from negfpy.anharmonic.bulk_rta import (
    PHONO3PY_4_4_RTA_CONVERSION,
    BulkModeHeatCapacities,
    BulkRelaxationTimes,
    BulkThreePhononRateSpectrum,
    assemble_rta_thermal_conductivity,
    compute_cumulative_thermal_conductivity,
    compute_mean_free_paths,
    compute_mode_heat_capacities,
    compute_three_phonon_lifetimes,
)
from negfpy.anharmonic.bulk_three_phonon import (
    PHONO3PY_4_4_EV_J,
    PHONO3PY_4_4_KB_EV_K,
    PHONO3PY_4_4_THZ_TO_EV,
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
)
from negfpy.bulk.archive import read_rta_result_hdf5, write_rta_result_hdf5
from negfpy.bulk.rate_archive import (
    read_rate_spectrum_hdf5,
    write_rate_spectrum_hdf5,
)


def _spectrum(
    frequencies: np.ndarray,
    eigenvectors: np.ndarray | None = None,
) -> BulkPhononSpectrum:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    n_q, n_band = frequencies.shape
    if eigenvectors is None:
        eigenvectors = np.repeat(
            np.eye(n_band, dtype=np.complex128)[None, :, :],
            n_q,
            axis=0,
        )
    qpoints = np.zeros((n_q, 3))
    qpoints[:, 0] = np.arange(n_q, dtype=np.float64) / n_q
    return BulkPhononSpectrum(
        qpoints=qpoints,
        dynamical_matrices=np.repeat(
            np.diag(frequencies[0] ** 2)[None, :, :],
            n_q,
            axis=0,
        ),
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        frequency_factor_to_thz=1.0,
        hermitianized=True,
        sampling_kind="uniform_full_mesh",
        mesh_shape=(n_q, 1, 1),
    )


def _linewidth_result(gamma: float) -> BulkThreePhononLinewidths:
    spectrum = _spectrum(np.array([[1.0]]))
    interactions = BulkThreePhononInteractions(
        grid_points=[0],
        band_indices=[0],
        q1_indices=[[0]],
        q2_indices=[[0]],
        reciprocal_vectors=[[[0, 0, 0]]],
        normal_mask=[[True]],
        matrix_elements=np.zeros((1, 1, 1, 1, 1), dtype=np.complex128),
        interaction_strengths=np.zeros((1, 1, 1, 1, 1)),
        phonons=spectrum,
        metadata={},
    )
    return BulkThreePhononLinewidths(
        temperatures=[300.0],
        grid_points=[0],
        band_indices=[0],
        gamma_thz_hwhm=[[[gamma]]],
        gamma_normal_thz_hwhm=[[[gamma]]],
        gamma_umklapp_thz_hwhm=[[[0.0]]],
        gamma_angular_fwhm=[[[4.0 * np.pi * gamma]]],
        decay_phase_space=[[0.0]],
        difference_phase_space=[[0.0]],
        interactions=interactions,
        metadata={"cutoff_frequency_thz": 1.0e-4},
    )


def test_linewidth_result_enforces_keldysh_style_field_identities() -> None:
    reference = _linewidth_result(0.01)
    with pytest.raises(ValueError, match="normal plus Umklapp"):
        replace(reference, gamma_normal_thz_hwhm=[[[0.009]]])
    with pytest.raises(ValueError, match=r"4\*pi"):
        replace(reference, gamma_angular_fwhm=[[[0.1]]])
    with pytest.raises(ValueError, match="nonnegative"):
        replace(reference, gamma_thz_hwhm=[[[-0.01]]])
    with pytest.raises(ValueError, match="grid_points"):
        replace(reference, grid_points=[1])


def test_band_path_interpolation_has_no_duplicate_joints_and_exact_distance() -> None:
    path = interpolate_band_path(
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [0.5, 0.5, 0.0]],
        np.eye(3),
        points_per_segment=[3, 4],
        labels=["G", "X", "M"],
    )
    assert path.qpoints.shape == (6, 3)
    np.testing.assert_array_equal(path.vertex_indices, [0, 2, 5])
    np.testing.assert_allclose(path.vertex_distances_inv_angstrom, [0.0, 0.5, 1.0])
    assert path.labels == ("G", "X", "M")


def test_phonon_dos_and_cartesian_projections_have_standard_normalization() -> None:
    result = compute_phonon_dos(
        _spectrum(np.array([[1.0, 2.0, 3.0]])),
        sigma_thz=0.05,
        n_frequency_points=8001,
    )
    assert result.expected_states_per_cell == 3
    assert result.integral_states_per_cell == pytest.approx(3.0, abs=2.0e-6)
    np.testing.assert_allclose(
        np.sum(result.projected_states_per_thz_cell, axis=(0, 1)),
        result.total_states_per_thz_cell,
        rtol=1.0e-14,
        atol=1.0e-14,
    )
    projected_integrals = np.trapezoid(
        result.projected_states_per_thz_cell,
        result.frequency_points_thz,
        axis=2,
    )
    np.testing.assert_allclose(projected_integrals, np.ones((1, 3)), atol=7.0e-7)


def test_phonon_dos_rejects_unstable_spectrum() -> None:
    with pytest.raises(ValueError, match="dynamically unstable"):
        compute_phonon_dos(_spectrum(np.array([[-0.1, 1.0, 2.0]])))


def test_phonon_dos_rejects_band_paths_and_truncated_windows() -> None:
    base = _spectrum(np.array([[1.0, 2.0, 3.0]]))
    band_path = BulkPhononSpectrum(
        qpoints=base.qpoints,
        dynamical_matrices=base.dynamical_matrices,
        frequencies=base.frequencies,
        eigenvectors=base.eigenvectors,
        frequency_factor_to_thz=base.frequency_factor_to_thz,
        hermitianized=True,
        sampling_kind="band_path",
    )
    with pytest.raises(ValueError, match="band path"):
        compute_phonon_dos(band_path)
    with pytest.raises(ValueError, match="state-count normalization"):
        compute_phonon_dos(
            base,
            frequency_points_thz=[1.95, 2.0, 2.05],
            sigma_thz=0.05,
        )
    truncated = compute_phonon_dos(
        base,
        frequency_points_thz=[1.95, 2.0, 2.05],
        sigma_thz=0.05,
        allow_truncated_window=True,
    )
    assert not truncated.normalization_certified


def test_mode_heat_capacity_is_stable_in_known_limits() -> None:
    frequencies = np.array([[1.0e-8, 1.0, 1000.0]])
    result = compute_mode_heat_capacities(
        frequencies,
        [0.0, 300.0],
        cutoff_frequency_thz=0.0,
    )
    np.testing.assert_array_equal(result.values_ev_per_k[0], 0.0)
    assert result.values_ev_per_k[1, 0, 0] == pytest.approx(
        PHONO3PY_4_4_KB_EV_K,
        rel=1.0e-14,
    )
    assert result.values_ev_per_k[1, 0, 2] < 1.0e-60
    np.testing.assert_allclose(
        result.values_j_per_k,
        result.values_ev_per_k * PHONO3PY_4_4_EV_J,
    )


def test_mode_heat_capacity_handles_positive_subnormal_temperature() -> None:
    subnormal = np.nextafter(np.float64(0.0), np.float64(1.0))
    with np.errstate(all="raise"):
        result = compute_mode_heat_capacities(
            [[subnormal, 1.0]],
            [subnormal],
            cutoff_frequency_thz=0.0,
        )
    assert np.all(np.isfinite(result.values_ev_per_k))
    x = PHONO3PY_4_4_THZ_TO_EV / PHONO3PY_4_4_KB_EV_K
    expected = (
        PHONO3PY_4_4_KB_EV_K
        * x**2
        * np.exp(-x)
        / (-np.expm1(-x)) ** 2
    )
    assert result.values_ev_per_k[0, 0, 0] == pytest.approx(
        expected,
        rel=2.0e-13,
    )
    assert result.values_ev_per_k[0, 0, 1] == 0.0


def _track_durable_publication(monkeypatch, module) -> list[str]:
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def tracked_fsync(descriptor: int) -> None:
        kind = "directory" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file"
        events.append(f"fsync-{kind}")
        real_fsync(descriptor)

    def tracked_replace(source, destination) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(module.os, "replace", tracked_replace)
    return events


def test_linewidth_lifetime_conversion_contains_exact_four_pi() -> None:
    gamma = 0.01
    result = compute_three_phonon_lifetimes(_linewidth_result(gamma))
    assert result.linewidth_angular_fwhm_ps_inv[0, 0, 0] == pytest.approx(
        4.0 * np.pi * gamma,
        rel=1.0e-15,
    )
    assert result.population_lifetime_ps[0, 0, 0] == pytest.approx(
        1.0 / (4.0 * np.pi * gamma),
        rel=1.0e-15,
    )
    assert result.resolved_mask[0, 0, 0]


def test_mean_free_path_native_unit_identity() -> None:
    spectrum = _spectrum(np.array([[1.0]]))
    velocities = BulkGroupVelocities(
        qpoints=spectrum.qpoints,
        frequencies=spectrum.frequencies,
        values_thz_angstrom=[[[1.0, 2.0, 2.0]]],
        values_m_per_s=[[[100.0, 200.0, 200.0]]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-4,
        degenerate_tolerance_thz=1.0e-4,
        perturbation_direction=np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    relaxation = compute_three_phonon_lifetimes(
        _linewidth_result(1.0 / (8.0 * np.pi))
    )
    result = compute_mean_free_paths(velocities, relaxation)
    np.testing.assert_allclose(result.vector_angstrom[0, 0, 0], [2.0, 4.0, 4.0])
    assert result.magnitude_angstrom[0, 0, 0] == pytest.approx(6.0)
    assert result.magnitude_m[0, 0, 0] == pytest.approx(6.0e-10)


def test_one_mode_rta_fixture_catches_all_unit_and_linewidth_factors() -> None:
    spectrum = _spectrum(np.array([[1.0]]))
    velocities = BulkGroupVelocities(
        qpoints=spectrum.qpoints,
        frequencies=spectrum.frequencies,
        values_thz_angstrom=[[[1.0, 2.0, 3.0]]],
        values_m_per_s=[[[100.0, 200.0, 300.0]]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=0.0,
        degenerate_tolerance_thz=1.0e-4,
        perturbation_direction=np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    heat_capacity = BulkModeHeatCapacities(
        temperatures=[300.0],
        frequencies_thz=spectrum.frequencies,
        values_ev_per_k=[[[1.0]]],
        values_j_per_k=[[[PHONO3PY_4_4_EV_J]]],
        included_mask=[[True]],
        cutoff_frequency_thz=0.0,
    )
    gamma = 1.0 / (4.0 * np.pi)
    relaxation = BulkRelaxationTimes(
        temperatures=[300.0],
        frequencies_thz=spectrum.frequencies,
        gamma_thz_hwhm=[[[gamma]]],
        linewidth_angular_fwhm_ps_inv=[[[1.0]]],
        population_lifetime_ps=[[[1.0]]],
        included_mask=[[True]],
        resolved_mask=[[[True]]],
        cutoff_frequency_thz=0.0,
        gamma_floor_thz=0.0,
    )
    result = assemble_rta_thermal_conductivity(
        spectrum,
        velocities,
        heat_capacity,
        relaxation,
        primitive_volume_angstrom3=1.0,
    )
    expected = PHONO3PY_4_4_RTA_CONVERSION * np.outer([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result.kappa_w_mk[0], expected, rtol=1.0e-14)
    np.testing.assert_array_equal(result.qpoints, spectrum.qpoints)
    np.testing.assert_allclose(result.kappa_voigt_w_mk[0], expected[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]])
    assert result.mean_free_path_angstrom[0, 0, 0] == pytest.approx(np.sqrt(14.0))

    with pytest.raises(ValueError, match="nonnegative vector"):
        replace(result, temperatures=[-300.0])
    with pytest.raises(ValueError, match="heat capacities"):
        replace(result, heat_capacities_ev_per_k=-result.heat_capacities_ev_per_k)
    with pytest.raises(ValueError, match="positive semidefinite|diagonal"):
        replace(
            result,
            mode_kappa_w_mk=-result.mode_kappa_w_mk,
            band_index_kappa_w_mk=-result.band_index_kappa_w_mk,
            kappa_w_mk=-result.kappa_w_mk,
        )

    cumulative = compute_cumulative_thermal_conductivity(
        result,
        coordinate="frequency_thz",
        thresholds=[0.5, 1.0],
    )
    np.testing.assert_array_equal(cumulative.cumulative_kappa_w_mk[0, 0], 0.0)
    np.testing.assert_allclose(cumulative.cumulative_kappa_w_mk[0, -1], expected)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_cumulative_thermal_conductivity(
            result,
            coordinate="frequency_thz",
            thresholds=[-1.0, 1.0],
        )
    with pytest.raises(ValueError, match="nondecreasing"):
        replace(
            cumulative,
            cumulative_kappa_w_mk=cumulative.cumulative_kappa_w_mk[:, ::-1],
        )


def test_rta_hdf5_archive_round_trip_is_schema_checked(
    tmp_path,
    monkeypatch,
) -> None:
    spectrum = _spectrum(np.array([[1.0]]))
    velocities = BulkGroupVelocities(
        qpoints=spectrum.qpoints,
        frequencies=spectrum.frequencies,
        values_thz_angstrom=[[[1.0, 2.0, 3.0]]],
        values_m_per_s=[[[100.0, 200.0, 300.0]]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-4,
        degenerate_tolerance_thz=1.0e-4,
        perturbation_direction=np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    heat_capacity = compute_mode_heat_capacities(
        spectrum.frequencies,
        [300.0],
        cutoff_frequency_thz=1.0e-4,
    )
    relaxation = compute_three_phonon_lifetimes(_linewidth_result(0.01))
    result = assemble_rta_thermal_conductivity(
        spectrum,
        velocities,
        heat_capacity,
        relaxation,
        primitive_volume_angstrom3=10.0,
        metadata={"source_sha256": {"fc2": "a" * 64}},
    )
    events = _track_durable_publication(monkeypatch, rta_archive_module)
    path = write_rta_result_hdf5(tmp_path / "rta.h5", result)
    assert events == ["fsync-file", "replace", "fsync-directory"]
    restored = read_rta_result_hdf5(path)
    for name in (
        "qpoints",
        "frequencies_thz",
        "gamma_thz_hwhm",
        "population_lifetime_ps",
        "mode_kappa_w_mk",
        "kappa_w_mk",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(result, name))
    assert dict(restored.metadata) == dict(result.metadata)
    with pytest.raises(TypeError):
        restored.metadata["source_sha256"]["fc2"] = "b" * 64
    with pytest.raises(FileExistsError):
        write_rta_result_hdf5(path, result)

    import h5py

    with h5py.File(path, "r+") as handle:
        handle["gamma_thz_hwhm"][0, 0, 0] *= 2.0
    with pytest.raises(ValueError, match="payload SHA256"):
        read_rta_result_hdf5(path)

    external_path = tmp_path / "external.h5"
    linked_path = write_rta_result_hdf5(tmp_path / "linked.h5", result)
    with h5py.File(external_path, "w") as external:
        external.create_dataset(
            "gamma",
            data=result.gamma_thz_hwhm,
        )
    with h5py.File(linked_path, "r+") as handle:
        del handle["gamma_thz_hwhm"]
        handle["gamma_thz_hwhm"] = h5py.ExternalLink(
            external_path.name,
            "/gamma",
        )
    with pytest.raises(ValueError, match="internal hard link"):
        read_rta_result_hdf5(linked_path)


def test_lightweight_rate_archive_round_trip_and_invariants(
    tmp_path,
    monkeypatch,
) -> None:
    rates = BulkThreePhononRateSpectrum(
        temperatures=[300.0],
        qpoints=[[0.0, 0.0, 0.0]],
        frequencies_thz=[[1.0]],
        grid_points=[0],
        band_indices=[0],
        gamma_thz_hwhm=[[[0.01]]],
        gamma_normal_thz_hwhm=[[[0.006]]],
        gamma_umklapp_thz_hwhm=[[[0.004]]],
        gamma_angular_fwhm=[[[4.0 * np.pi * 0.01]]],
        decay_phase_space=[[0.2]],
        difference_phase_space=[[-0.1]],
        cutoff_frequency_thz=1.0e-4,
        frequency_factor_to_thz=1.0,
        physics_input_sha256="a" * 64,
        metadata={"source_archive_sha256": "b" * 64},
    )
    events = _track_durable_publication(monkeypatch, rate_archive_module)
    path = write_rate_spectrum_hdf5(tmp_path / "rates.h5", rates)
    assert events == ["fsync-file", "replace", "fsync-directory"]
    restored = read_rate_spectrum_hdf5(path)
    for name in (
        "qpoints",
        "frequencies_thz",
        "gamma_thz_hwhm",
        "gamma_normal_thz_hwhm",
        "gamma_umklapp_thz_hwhm",
    ):
        np.testing.assert_array_equal(getattr(restored, name), getattr(rates, name))
    assert restored.physics_input_sha256 == "a" * 64
    assert restored.metadata["source_archive_sha256"] == "b" * 64

    import h5py

    with h5py.File(path, "r+") as handle:
        handle["gamma_thz_hwhm"][0, 0, 0] *= 2.0
    with pytest.raises(ValueError, match="payload SHA256"):
        read_rate_spectrum_hdf5(path)


def test_strict_rta_assembly_rejects_unresolved_included_mode() -> None:
    spectrum = _spectrum(np.array([[1.0]]))
    velocities = BulkGroupVelocities(
        qpoints=spectrum.qpoints,
        frequencies=spectrum.frequencies,
        values_thz_angstrom=[[[1.0, 0.0, 0.0]]],
        values_m_per_s=[[[100.0, 0.0, 0.0]]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-4,
        degenerate_tolerance_thz=1.0e-4,
        perturbation_direction=np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    heat_capacity = compute_mode_heat_capacities(
        spectrum.frequencies,
        [300.0],
        cutoff_frequency_thz=1.0e-4,
    )
    relaxation = compute_three_phonon_lifetimes(_linewidth_result(0.0))
    with pytest.raises(ValueError, match="unresolved"):
        assemble_rta_thermal_conductivity(
            spectrum,
            velocities,
            heat_capacity,
            relaxation,
            primitive_volume_angstrom3=1.0,
        )


def test_relaxation_result_rejects_inconsistent_lifetime() -> None:
    gamma = 1.0
    with pytest.raises(ValueError, match="must equal 1/angular"):
        BulkRelaxationTimes(
            temperatures=[300.0],
            frequencies_thz=[[1.0]],
            gamma_thz_hwhm=[[[gamma]]],
            linewidth_angular_fwhm_ps_inv=[[[4.0 * np.pi * gamma]]],
            population_lifetime_ps=[[[123.0]]],
            included_mask=[[True]],
            resolved_mask=[[[True]]],
            cutoff_frequency_thz=0.0,
            gamma_floor_thz=0.0,
        )


def test_rta_assembly_rejects_reordered_or_different_qpoints() -> None:
    spectrum = _spectrum(np.array([[1.0]]))
    velocities = BulkGroupVelocities(
        qpoints=[[0.5, 0.0, 0.0]],
        frequencies=spectrum.frequencies,
        values_thz_angstrom=[[[1.0, 0.0, 0.0]]],
        values_m_per_s=[[[100.0, 0.0, 0.0]]],
        frequency_factor_to_thz=1.0,
        cutoff_frequency_thz=1.0e-4,
        degenerate_tolerance_thz=1.0e-4,
        perturbation_direction=np.array([1.0, 2.0, 3.0]) / np.sqrt(14.0),
    )
    heat_capacity = compute_mode_heat_capacities(spectrum.frequencies, [300.0])
    relaxation = compute_three_phonon_lifetimes(_linewidth_result(0.01))
    with pytest.raises(ValueError, match="q-points do not match"):
        assemble_rta_thermal_conductivity(
            spectrum,
            velocities,
            heat_capacity,
            relaxation,
            primitive_volume_angstrom3=1.0,
        )


def test_public_bulk_namespace_contains_release_operations() -> None:
    from negfpy import bulk

    assert bulk.compute_phonon_dos is compute_phonon_dos
    assert bulk.compute_mode_heat_capacities is compute_mode_heat_capacities
    assert callable(bulk.compute_rta_thermal_conductivity)
    assert callable(bulk.summarize_three_phonon_linewidths)
    assert callable(bulk.read_rate_spectrum_hdf5)
    assert callable(bulk.write_rate_spectrum_hdf5)
