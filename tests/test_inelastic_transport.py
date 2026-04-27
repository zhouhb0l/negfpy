import numpy as np
import pytest

from negfpy.core import transmission
from negfpy.inelastic import PowerLawPPSelfEnergy, transmission_inelastic
from negfpy.models import ChainParams, analytic_band_max, device_perfect_chain, lead_blocks


def _chain_setup() -> tuple[float, object, object]:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=14, params=params)
    omega = 1.0
    return omega, device, lead


def test_inelastic_zero_scattering_matches_ballistic() -> None:
    omega, device, lead = _chain_setup()
    model = PowerLawPPSelfEnergy(gamma0=0.0, power=2.0, omega_ref=1.0)

    t_ballistic = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    t_inelastic, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=model,
        max_iter=20,
        mixing=0.5,
        tol=1e-10,
    )

    assert np.isfinite(t_inelastic)
    assert abs(t_inelastic - t_ballistic) < 1e-8
    assert bool(info["converged"])
    assert int(info["iterations"]) == 1
    assert float(info["sigma_pp_norm"]) == 0.0


def test_inelastic_zero_scattering_matches_ballistic_spectrum() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=18, params=params)
    model = PowerLawPPSelfEnergy(gamma0=0.0, power=2.0, omega_ref=1.0)

    omegas = np.linspace(0.15 * analytic_band_max(params), 0.85 * analytic_band_max(params), 11)
    ballistic = np.array(
        [transmission(omega=float(w), device=device, lead_left=lead, lead_right=lead, eta=1e-8) for w in omegas]
    )
    inelastic = np.array(
        [
            transmission_inelastic(
                omega=float(w),
                device=device,
                lead_left=lead,
                lead_right=lead,
                eta=1e-8,
                pp_self_energy=model,
                max_iter=20,
                mixing=0.5,
                tol=1e-10,
            )[0]
            for w in omegas
        ]
    )

    assert np.all(np.isfinite(ballistic))
    assert np.all(np.isfinite(inelastic))
    assert np.max(np.abs(inelastic - ballistic)) < 1e-8


def test_inelastic_finite_scattering_reduces_transmission() -> None:
    omega, device, lead = _chain_setup()
    model = PowerLawPPSelfEnergy(gamma0=0.08, power=2.0, omega_ref=1.0)

    t_ballistic = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    t_inelastic, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=model,
        max_iter=20,
        mixing=1.0,
        tol=1e-10,
    )

    assert np.isfinite(t_inelastic)
    assert 0.0 <= t_inelastic < t_ballistic
    assert bool(info["converged"])
    assert float(info["sigma_pp_norm"]) > 0.0


def test_inelastic_custom_model_converges_with_mixing() -> None:
    omega, device, lead = _chain_setup()

    def constant_self_energy(omega_local: float, g: np.ndarray, iteration: int) -> np.ndarray:
        del omega_local, iteration
        return -1j * 0.03 * np.eye(g.shape[0], dtype=np.complex128)

    t_inelastic, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=constant_self_energy,
        max_iter=30,
        mixing=0.4,
        tol=1e-6,
    )

    assert np.isfinite(t_inelastic)
    assert bool(info["converged"])
    assert int(info["iterations"]) > 1
    assert float(info["residual"]) <= 1e-6


def test_inelastic_nonconvergence_can_raise() -> None:
    omega, device, lead = _chain_setup()

    def oscillating_self_energy(omega_local: float, g: np.ndarray, iteration: int) -> np.ndarray:
        del omega_local
        sign = 1.0 if iteration % 2 == 0 else -1.0
        return sign * 1j * 1e-4 * np.eye(g.shape[0], dtype=np.complex128)

    with pytest.raises(RuntimeError):
        transmission_inelastic(
            omega=omega,
            device=device,
            lead_left=lead,
            lead_right=lead,
            eta=1e-8,
            pp_self_energy=oscillating_self_energy,
            max_iter=4,
            mixing=1.0,
            tol=1e-12,
            raise_on_nonconvergence=True,
        )
