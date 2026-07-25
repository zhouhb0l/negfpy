from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic import SignedFrequencyGrid, solve_one_shot_born
from negfpy.anharmonic.toy.open_three_mode import (
    OpenThreeModeFixture,
    _zero_extended_scalar_convolution,
    measure_mode_poles,
    open_three_mode_oracle,
)


def test_open_three_mode_fixture_is_matched_resonant_and_release_sized() -> None:
    fixture = OpenThreeModeFixture()

    assert fixture.mode_frequencies == (1.5, 0.9, 0.6)
    assert fixture.mode_frequencies[0] == pytest.approx(
        fixture.mode_frequencies[1] + fixture.mode_frequencies[2]
    )
    assert fixture.temperature == 0.7
    assert fixture.masses == (1.0, 1.0, 1.0)
    assert fixture.lead_coupling == 0.1
    assert fixture.grid.step == 0.015
    assert fixture.grid.n_positive == 334

    lead = fixture.problem.left.lead
    expected_onsite = np.diag(np.square(fixture.mode_frequencies))
    np.testing.assert_allclose(lead.d00, expected_onsite)
    np.testing.assert_allclose(lead.d01, -0.1 * np.eye(3))
    np.testing.assert_allclose(
        fixture.problem.device.onsite_blocks[0],
        lead.d00,
    )
    assert fixture.grid.cutoff >= 3.0 * np.sqrt(
        np.max(np.diag(lead.d00)) + 2.0 * fixture.lead_coupling
    )

    vertex = fixture.problem.vertex.values
    assert np.count_nonzero(vertex) == 6
    assert all(
        vertex[indices] == fixture.g
        for indices in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    )


def test_independent_grid_spectrum_matches_production_one_shot_born() -> None:
    fixture = OpenThreeModeFixture()
    oracle = open_three_mode_oracle(fixture)
    production = solve_one_shot_born(fixture.problem, fixture.config)

    for component in ("lesser", "greater", "retarded"):
        np.testing.assert_allclose(
            getattr(oracle.sigma_pp, component),
            getattr(production.sigma_pp, component),
            rtol=5.0e-12,
            atol=5.0e-13,
        )

    production_shifts, production_linewidths = measure_mode_poles(
        fixture.grid,
        fixture.mode_frequencies,
        production.sigma_pp.retarded,
    )
    np.testing.assert_allclose(
        production_shifts,
        oracle.grid_frequency_shifts,
        rtol=5.0e-12,
        atol=5.0e-13,
    )
    np.testing.assert_allclose(
        production_linewidths,
        oracle.grid_linewidths,
        rtol=5.0e-12,
        atol=5.0e-13,
    )
    assert oracle.method_metadata["pole_method"] == (
        "adaptive_eta_free_continuum_on_shell"
    )
    assert oracle.method_metadata["grid_spectra_role"].startswith("diagnostic")


def test_open_oracle_is_even_in_g_and_scales_as_g_squared() -> None:
    positive = open_three_mode_oracle(OpenThreeModeFixture(g=0.01))
    negative = open_three_mode_oracle(OpenThreeModeFixture(g=-0.01))
    doubled = open_three_mode_oracle(OpenThreeModeFixture(g=0.02))

    for component in ("lesser", "greater", "retarded"):
        positive_values = getattr(positive.sigma_pp, component)
        np.testing.assert_allclose(
            getattr(negative.sigma_pp, component),
            positive_values,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            getattr(doubled.sigma_pp, component),
            4.0 * positive_values,
            rtol=2.0e-14,
            atol=2.0e-14,
        )

    np.testing.assert_allclose(
        negative.self_energy_on_shell,
        positive.self_energy_on_shell,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        doubled.self_energy_on_shell,
        4.0 * positive.self_energy_on_shell,
        rtol=5.0e-13,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        doubled.frequency_shifts,
        4.0 * positive.frequency_shifts,
        rtol=5.0e-13,
        atol=5.0e-14,
    )
    np.testing.assert_allclose(
        doubled.linewidths,
        4.0 * positive.linewidths,
        rtol=5.0e-13,
        atol=5.0e-14,
    )


def test_zero_temperature_retains_spontaneous_parent_decay() -> None:
    fixture = OpenThreeModeFixture(temperature=0.0)
    oracle = open_three_mode_oracle(fixture)
    parent_index = fixture.grid.zero_index + int(
        round(fixture.mode_frequencies[0] / fixture.grid.step)
    )

    assert oracle.sigma_pp.lesser[parent_index, 0, 0] == pytest.approx(
        0.0j,
        abs=1.0e-14,
    )
    assert oracle.sigma_pp.greater[parent_index, 0, 0].imag < 0.0
    assert oracle.self_energy_on_shell[0].imag < 0.0
    assert oracle.linewidths[0] > 5.0e-4
    assert oracle.linewidths[1] == pytest.approx(0.0, abs=1.0e-12)
    assert oracle.linewidths[2] == pytest.approx(0.0, abs=1.0e-12)
    assert oracle.passive


def test_zero_temperature_parent_linewidth_matches_independent_quadrature() -> None:
    fixture = OpenThreeModeFixture(temperature=0.0)
    oracle = open_three_mode_oracle(fixture)
    omega_x, omega_y, omega_z = fixture.mode_frequencies
    coupling = fixture.lead_coupling

    # At T=0 only x -> y + z contributes on the positive-frequency parent
    # shell.  For a matched scalar chain,
    #
    #   A_j(w) = 2 / sqrt(4 k**2 - (w**2 - Omega_j**2)**2),
    #
    # and w_y(q)**2 = Omega_y**2 - 2 k cos(q) gives
    # A_y(w_y) dw_y = dq / w_y.  This independently reduces the Born
    # linewidth to a smooth one-dimensional integral.  A sin(theta)**2
    # remapping also regularizes an endpoint if the z band limits the overlap.
    y_low = np.sqrt(omega_y**2 - 2.0 * coupling)
    y_high = np.sqrt(omega_y**2 + 2.0 * coupling)
    z_low = np.sqrt(omega_z**2 - 2.0 * coupling)
    z_high = np.sqrt(omega_z**2 + 2.0 * coupling)
    overlap_low = max(y_low, omega_x - z_high)
    overlap_high = min(y_high, omega_x - z_low)
    assert overlap_low < overlap_high

    def wavevector(frequency: float) -> float:
        cosine = (omega_y**2 - frequency**2) / (2.0 * coupling)
        return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    q_low = wavevector(overlap_low)
    q_high = wavevector(overlap_high)
    nodes, weights = np.polynomial.legendre.leggauss(96)
    theta = 0.25 * np.pi * (nodes + 1.0)
    theta_weights = 0.25 * np.pi * weights
    q_span = q_high - q_low
    q = q_low + q_span * np.sin(theta) ** 2
    dq_dtheta = 2.0 * q_span * np.sin(theta) * np.cos(theta)
    frequency_y = np.sqrt(omega_y**2 - 2.0 * coupling * np.cos(q))
    frequency_z = omega_x - frequency_y
    spectral_z = 2.0 / np.sqrt(
        4.0 * coupling**2
        - (frequency_z**2 - omega_z**2) ** 2
    )
    bubble_spectral_weight = fixture.g**2 / (2.0 * np.pi) * np.sum(
        theta_weights * dq_dtheta * spectral_z / frequency_y
    )
    expected_linewidth = bubble_spectral_weight / (2.0 * omega_x)

    assert expected_linewidth > 0.0
    assert oracle.linewidths[0] == pytest.approx(
        expected_linewidth,
        rel=5.0e-7,
        abs=1.0e-12,
    )


def test_weak_coupling_gate_rejects_strong_cubic_input() -> None:
    weak = open_three_mode_oracle(OpenThreeModeFixture(g=0.02))
    strong = open_three_mode_oracle(OpenThreeModeFixture(g=0.2))

    assert weak.passive
    assert weak.weak_coupling_passed
    assert np.all(weak.relative_shifts < 0.1)
    assert np.all(weak.relative_linewidths < 0.1)
    assert strong.passive
    assert strong.relative_linewidths[0] > 0.1
    assert not strong.weak_coupling_passed


def test_adaptive_poles_are_grid_eta_and_tolerance_refinable_in_pr() -> None:
    coarse = open_three_mode_oracle(
        OpenThreeModeFixture(
            grid=SignedFrequencyGrid(step=0.03, n_positive=167),
            eta_lead=2.0e-3,
        ),
        integration_epsabs=1.0e-8,
        integration_epsrel=1.0e-6,
    )
    reference = open_three_mode_oracle(OpenThreeModeFixture())
    tight = open_three_mode_oracle(
        OpenThreeModeFixture(
            grid=SignedFrequencyGrid(step=0.01, n_positive=501),
            eta_lead=5.0e-4,
        ),
        integration_epsabs=2.5e-10,
        integration_epsrel=2.5e-8,
    )

    # Adaptive continuum acceptance values do not inherit the diagnostic grid
    # or its lead regulator.  Use the declared absolute shift tolerance so a
    # future nearly-zero shift cannot make a relative comparison meaningless.
    for candidate in (coarse, tight):
        np.testing.assert_allclose(
            candidate.frequency_shifts,
            reference.frequency_shifts,
            rtol=2.0e-6,
            atol=reference.frequency_shift_atol,
        )
        np.testing.assert_allclose(
            candidate.linewidths,
            reference.linewidths,
            rtol=2.0e-6,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            candidate.self_energy_on_shell,
            reference.self_energy_on_shell,
            rtol=2.0e-6,
            atol=2.0e-10,
        )

    assert np.max(reference.integration_error_estimates[:, :2]) < 2.0e-9
    assert np.max(tight.integration_error_estimates[:, :2]) <= (
        1.05 * np.max(coarse.integration_error_estimates[:, :2])
    )
    assert reference.method_metadata["eta_used_for_adaptive_poles"] == 0.0
    assert reference.method_metadata["frequency_shift_comparison_atol"] == (
        reference.frequency_shift_atol
    )


def test_open_fixture_rejects_nonresonant_or_underresolved_domain() -> None:
    with pytest.raises(ValueError, match="Omega_x=Omega_y\\+Omega_z"):
        OpenThreeModeFixture(mode_frequencies=(1.5, 0.8, 0.6))
    with pytest.raises(ValueError, match="cutoff"):
        OpenThreeModeFixture(
            grid=SignedFrequencyGrid(step=0.1, n_positive=15),
        )


def test_cutoff_six_surface_oracle_uses_stable_quadratic_root() -> None:
    """The preregistered cutoff refinement must not lose root precision."""

    fixture = OpenThreeModeFixture(
        grid=SignedFrequencyGrid(step=0.015, n_positive=400),
        g=0.0,
    )
    oracle = open_three_mode_oracle(fixture)

    assert fixture.grid.cutoff == pytest.approx(6.0)
    for component in ("retarded", "lesser", "greater"):
        assert np.all(np.isfinite(getattr(oracle.green_zero, component)))
        assert np.all(np.isfinite(getattr(oracle.sigma_pp, component)))


def test_vectorized_analytic_diagonal_contacts_match_sancho_rubio() -> None:
    controls = {
        "grid": SignedFrequencyGrid(step=0.1, n_positive=51),
        "g": 0.01,
        "eta_lead": 0.01,
    }
    reference_fixture = OpenThreeModeFixture(**controls)
    analytic_fixture = OpenThreeModeFixture(
        **controls,
        surface_gf_method="analytic_diagonal",
    )
    reference = solve_one_shot_born(
        reference_fixture.problem,
        reference_fixture.config,
    )
    analytic = solve_one_shot_born(
        analytic_fixture.problem,
        analytic_fixture.config,
    )

    for spectrum_name in ("sigma_left", "sigma_right", "sigma_pp", "green"):
        reference_spectrum = getattr(reference, spectrum_name)
        analytic_spectrum = getattr(analytic, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(analytic_spectrum, component),
                getattr(reference_spectrum, component),
                rtol=1.0e-10,
                atol=1.0e-12,
            )


def test_release_sized_independent_fft_convolution_is_linear_not_circular() -> None:
    rng = np.random.default_rng(773)
    grid = SignedFrequencyGrid(step=0.003, n_positive=1024)
    first = rng.normal(size=grid.n_points) + 1.0j * rng.normal(
        size=grid.n_points
    )
    second = rng.normal(size=grid.n_points) + 1.0j * rng.normal(
        size=grid.n_points
    )

    accelerated = _zero_extended_scalar_convolution(first, second, grid)
    full = np.convolve(first, second, mode="full")
    expected = (
        grid.integration_weight
        * full[grid.zero_index : grid.zero_index + grid.n_points]
    )
    np.testing.assert_allclose(
        accelerated,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
