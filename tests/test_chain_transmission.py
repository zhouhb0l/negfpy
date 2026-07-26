import numpy as np

from negfpy.core import transmission, transmission_kavg_adaptive
from negfpy.models import (
    ChainParams,
    analytic_band_max,
    device_mass_defect,
    device_perfect_chain,
    lead_blocks,
)


def test_transmission_plateau_matched_chain() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=20, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(0.2 * wmax, 0.8 * wmax, 9)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-10) for w in omegas])

    assert np.all(np.isfinite(vals))
    np.testing.assert_allclose(vals, 1.0, rtol=1.0e-10, atol=1.0e-12)


def test_transmission_nonnegative() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=12, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(0.05 * wmax, 1.2 * wmax, 30)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

    assert np.min(vals) > -1e-8


def test_out_of_band_goes_to_zero() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=16, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(1.1 * wmax, 1.7 * wmax, 10)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

    assert np.max(np.abs(vals)) < 1e-4


def test_single_mass_defect_matches_analytic_scattering() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    defect_mass = 2.5
    lead = lead_blocks(params)
    device = device_mass_defect(
        n_layers=21,
        params=params,
        defect_index=10,
        defect_mass=defect_mass,
    )

    for omega in (0.3, 0.8, 1.3, 1.8):
        cos_q = 1.0 - params.mass * omega**2 / (2.0 * params.spring)
        sin_q_sq = 1.0 - cos_q**2
        expected = (4.0 * params.spring**2 * sin_q_sq) / (
            4.0 * params.spring**2 * sin_q_sq
            + (defect_mass - params.mass) ** 2 * omega**4
        )
        actual = transmission(
            omega=omega,
            device=device,
            lead_left=lead,
            lead_right=lead,
            eta=1.0e-12,
        )
        assert np.isclose(actual, expected, rtol=1.0e-10, atol=1.0e-11)


def test_default_device_regulator_is_exactly_zero() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=20, params=params)

    default = transmission(1.0, device, lead, lead, eta=1.0e-3)
    explicit_zero = transmission(
        1.0, device, lead, lead, eta=1.0e-3, eta_device=0.0
    )
    absorbing_device = transmission(
        1.0, device, lead, lead, eta=1.0e-3, eta_device=1.0e-3
    )

    assert default == explicit_zero
    assert not np.isclose(default, absorbing_device, rtol=1.0e-8, atol=1.0e-10)


def test_harmonic_omega_scaling_preserves_default_and_explicit_contacts() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=16, params=params)

    for omega in (0.4, 1.0, 1.8):
        reference = transmission(
            omega, device, lead, lead, eta=1.0e-8, eta_device=0.0
        )
        scaled_default = transmission(
            omega,
            device,
            lead,
            lead,
            eta=1.0e-8,
            eta_device=0.0,
            omega_scale=2.0,
        )
        scaled_explicit = transmission(
            omega,
            device,
            lead,
            lead,
            eta=1.0e-8,
            eta_device=0.0,
            omega_scale=2.0,
            device_to_lead_left=lead.d01.conj().T,
            device_to_lead_right=lead.d01,
        )

        assert np.isclose(
            scaled_default, reference, rtol=1.0e-10, atol=1.0e-11
        )
        assert np.isclose(
            scaled_explicit, reference, rtol=1.0e-10, atol=1.0e-11
        )


def test_transmission_kavg_adaptive_stats_matched_chain() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=14, params=params)

    omega = 0.5 * analytic_band_max(params)
    tavg, info = transmission_kavg_adaptive(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        kpoints=[(), (), ()],
        eta_values=(1e-8, 1e-7),
        min_success_fraction=1.0,
    )

    assert np.isfinite(tavg)
    assert tavg > 0.9
    assert info["n_total"] == 3
    assert info["n_success"] == 3
    assert info["n_failed"] == 0
    assert abs(float(info["success_fraction"]) - 1.0) < 1e-12
