import numpy as np

from negfpy.core import device_green_function, transmission
from negfpy.inelastic import Phi4Params, phi4_scba_model, transmission_inelastic
from negfpy.models import analytic_band_max, device_perfect_chain, lead_blocks


def test_phi4_scba_zero_lambda_gives_zero_self_energy() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.0)
    model = phi4_scba_model(
        n_layers=8,
        params=params,
        temperature=0.3,
        broadening=0.02,
    )
    sigma = model(omega=0.8, green_function=np.eye(8, dtype=np.complex128), iteration=0)
    assert np.max(np.abs(sigma)) == 0.0


def test_phi4_scba_self_energy_is_finite_for_ballistic_green_function() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.08)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=8, params=params.harmonic_params)
    omega = 0.8 * analytic_band_max(params.harmonic_params)
    g_ballistic, _, _ = device_green_function(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
    )
    model = phi4_scba_model(
        n_layers=8,
        params=params,
        temperature=0.3,
        broadening=0.02,
    )
    sigma = model(omega=omega, green_function=g_ballistic, iteration=0)
    assert np.all(np.isfinite(sigma))
    assert np.linalg.norm(sigma) > 0.0


def test_phi4_scba_zero_lambda_matches_ballistic_transmission() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.0)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=12, params=params.harmonic_params)
    omega = 0.8 * analytic_band_max(params.harmonic_params)
    t_ballistic = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    model = phi4_scba_model(
        n_layers=12,
        params=params,
        temperature=0.35,
        broadening=0.03,
    )
    t_scba, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=model,
        max_iter=6,
        mixing=0.5,
        tol=1e-10,
    )
    assert np.isfinite(t_scba)
    assert abs(t_scba - t_ballistic) < 1e-8
    assert bool(info["converged"])


def test_phi4_scba_reduces_toy_chain_transmission() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.08)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=12, params=params.harmonic_params)
    omega = 0.85 * analytic_band_max(params.harmonic_params)
    t_ballistic = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    model = phi4_scba_model(
        n_layers=12,
        params=params,
        temperature=0.35,
        broadening=0.03,
    )
    t_scba, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=model,
        max_iter=20,
        mixing=0.5,
        tol=1e-5,
    )
    assert np.isfinite(t_scba)
    assert 0.0 <= t_scba < t_ballistic
    assert bool(info["converged"])
    assert float(info["sigma_pp_norm"]) > 0.0
