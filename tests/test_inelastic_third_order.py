import numpy as np

from negfpy.core import transmission
from negfpy.inelastic import (
    FPUAlphaParams,
    fpu_alpha_lowest_order_model,
    fpu_alpha_third_order_interaction,
    transmission_inelastic,
)
from negfpy.models import analytic_band_max, device_perfect_chain, lead_blocks


def test_fpu_alpha_third_order_interaction_zero_alpha_is_zero() -> None:
    params = FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.0)
    interaction = fpu_alpha_third_order_interaction(n_layers=6, params=params)
    assert np.max(np.abs(interaction.phi3)) == 0.0


def test_fpu_alpha_third_order_interaction_scales_linearly_with_alpha() -> None:
    interaction_1 = fpu_alpha_third_order_interaction(
        n_layers=5,
        params=FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.1),
    )
    interaction_2 = fpu_alpha_third_order_interaction(
        n_layers=5,
        params=FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.2),
    )
    norm_1 = float(np.linalg.norm(interaction_1.phi3))
    norm_2 = float(np.linalg.norm(interaction_2.phi3))
    assert norm_1 > 0.0
    assert abs(norm_2 / norm_1 - 2.0) < 1e-12


def test_fpu_alpha_lowest_order_model_zero_alpha_gives_zero_self_energy() -> None:
    params = FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.0)
    model = fpu_alpha_lowest_order_model(
        n_layers=8,
        params=params,
        temperature=0.3,
        broadening=0.02,
    )
    sigma = model(omega=0.8, green_function=np.eye(8, dtype=np.complex128), iteration=0)
    assert np.max(np.abs(sigma)) == 0.0


def test_fpu_alpha_lowest_order_self_energy_scales_quadratically_with_alpha() -> None:
    model_1 = fpu_alpha_lowest_order_model(
        n_layers=7,
        params=FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.05),
        temperature=0.25,
        broadening=0.03,
    )
    model_2 = fpu_alpha_lowest_order_model(
        n_layers=7,
        params=FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.10),
        temperature=0.25,
        broadening=0.03,
    )
    sigma_1 = model_1(omega=0.9, green_function=np.eye(7, dtype=np.complex128), iteration=0)
    sigma_2 = model_2(omega=0.9, green_function=np.eye(7, dtype=np.complex128), iteration=0)
    norm_1 = float(np.linalg.norm(sigma_1))
    norm_2 = float(np.linalg.norm(sigma_2))
    assert norm_1 > 0.0
    assert abs(norm_2 / norm_1 - 4.0) < 5e-2


def test_fpu_alpha_lowest_order_scattering_reduces_toy_chain_transmission() -> None:
    params = FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.08)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=12, params=params.harmonic_params)
    model = fpu_alpha_lowest_order_model(
        n_layers=12,
        params=params,
        temperature=0.35,
        broadening=0.03,
    )

    omega = 0.8 * analytic_band_max(params.harmonic_params)
    t_ballistic = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    t_inelastic, info = transmission_inelastic(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        pp_self_energy=model,
        max_iter=2,
        mixing=1.0,
        tol=1e-12,
    )

    assert np.isfinite(t_inelastic)
    assert 0.0 <= t_inelastic < t_ballistic
    assert bool(info["converged"])
