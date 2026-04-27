import numpy as np

from negfpy.inelastic import (
    FourthOrderInteraction,
    Phi4Params,
    ThirdOrderInteraction,
    fourth_order_lowest_order_model_from_covariance,
    fourth_order_lowest_order_self_energy,
    fourth_order_mean_field_model_from_covariance,
    fourth_order_mean_field_self_energy,
    fourth_order_scba_self_energy,
    phi4_fourth_order_interaction,
    phi4_lowest_order_model,
    phi4_mean_field_model,
    third_order_mean_field_model_from_displacement,
    third_order_mean_field_self_energy,
    third_order_scba_self_energy,
)


def test_third_order_mean_field_zero_displacement_gives_zero_self_energy() -> None:
    phi3 = np.ones((3, 3, 3), dtype=np.complex128)
    interaction = ThirdOrderInteraction(phi3=phi3)
    sigma = third_order_mean_field_self_energy(
        interaction=interaction,
        mean_displacement=np.zeros(3, dtype=np.complex128),
    )
    assert np.max(np.abs(sigma)) == 0.0


def test_third_order_mean_field_model_matches_direct_evaluation() -> None:
    phi3 = np.zeros((2, 2, 2), dtype=np.complex128)
    phi3[0, 1, 1] = 2.0
    phi3[1, 0, 1] = 2.0
    phi3[1, 1, 0] = 2.0
    interaction = ThirdOrderInteraction(phi3=phi3)
    displacement = np.array([0.3, -0.2], dtype=np.complex128)
    model = third_order_mean_field_model_from_displacement(
        interaction=interaction,
        mean_displacement=displacement,
    )
    sigma_direct = third_order_mean_field_self_energy(
        interaction=interaction,
        mean_displacement=displacement,
    )
    sigma_model = model(omega=0.4, green_function=np.eye(2, dtype=np.complex128), iteration=0)
    assert np.max(np.abs(sigma_direct - sigma_model)) < 1e-12


def test_fourth_order_mean_field_zero_covariance_gives_zero_self_energy() -> None:
    interaction = FourthOrderInteraction(phi4=np.ones((2, 2, 2, 2), dtype=np.complex128))
    sigma = fourth_order_mean_field_self_energy(
        interaction=interaction,
        covariance=np.zeros((2, 2), dtype=np.complex128),
    )
    assert np.max(np.abs(sigma)) == 0.0


def test_fourth_order_mean_field_scales_linearly_with_covariance() -> None:
    interaction = FourthOrderInteraction(phi4=np.zeros((2, 2, 2, 2), dtype=np.complex128))
    interaction.phi4[0, 0, 0, 0] = 1.5
    cov1 = np.eye(2, dtype=np.complex128)
    cov2 = 2.0 * cov1
    sigma1 = fourth_order_mean_field_self_energy(interaction=interaction, covariance=cov1)
    sigma2 = fourth_order_mean_field_self_energy(interaction=interaction, covariance=cov2)
    assert np.linalg.norm(sigma1) > 0.0
    assert abs(np.linalg.norm(sigma2) / np.linalg.norm(sigma1) - 2.0) < 1e-12


def test_phi4_interaction_and_mean_field_model_are_finite() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.2)
    interaction = phi4_fourth_order_interaction(n_layers=4, params=params)
    assert interaction.phi4.shape == (4, 4, 4, 4)

    model = phi4_mean_field_model(
        n_layers=4,
        params=params,
        temperature=0.5,
    )
    sigma = model(omega=0.7, green_function=np.eye(4, dtype=np.complex128), iteration=0)
    assert np.all(np.isfinite(sigma))
    assert np.allclose(sigma, sigma.conj().T)
    assert model.converged
    assert model.iterations >= 1
    assert float(model.residual) >= 0.0
    assert np.linalg.norm(sigma) > 0.0


def test_phi4_zero_lambda_gives_zero_self_consistent_mean_field() -> None:
    model = phi4_mean_field_model(
        n_layers=5,
        params=Phi4Params(mass=1.0, spring=1.0, lambda4=0.0),
        temperature=0.4,
    )
    sigma = model(omega=0.6, green_function=np.eye(5, dtype=np.complex128), iteration=0)
    assert model.converged
    assert np.max(np.abs(sigma)) == 0.0


def test_phi4_lowest_order_model_is_finite() -> None:
    model = phi4_lowest_order_model(
        n_layers=5,
        params=Phi4Params(mass=1.0, spring=1.0, lambda4=0.15),
        temperature=0.4,
    )
    sigma = model(omega=0.6, green_function=np.eye(5, dtype=np.complex128), iteration=0)
    assert np.all(np.isfinite(sigma))
    assert np.allclose(sigma, sigma.conj().T)
    assert np.linalg.norm(sigma) > 0.0


def test_fourth_order_mean_field_model_from_covariance_matches_direct() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.1)
    interaction = phi4_fourth_order_interaction(n_layers=3, params=params)
    covariance = 0.4 * np.eye(3, dtype=np.complex128)
    model = fourth_order_mean_field_model_from_covariance(
        interaction=interaction,
        covariance=covariance,
    )
    sigma_model = model(omega=0.5, green_function=np.eye(3, dtype=np.complex128), iteration=0)
    sigma_direct = fourth_order_mean_field_self_energy(interaction=interaction, covariance=covariance)
    assert np.max(np.abs(sigma_model - sigma_direct)) < 1e-12


def test_fourth_order_lowest_order_model_from_covariance_matches_direct() -> None:
    params = Phi4Params(mass=1.0, spring=1.0, lambda4=0.1)
    interaction = phi4_fourth_order_interaction(n_layers=3, params=params)
    covariance = 0.4 * np.eye(3, dtype=np.complex128)
    model = fourth_order_lowest_order_model_from_covariance(
        interaction=interaction,
        covariance=covariance,
    )
    sigma_model = model(omega=0.5, green_function=np.eye(3, dtype=np.complex128), iteration=0)
    sigma_direct = fourth_order_lowest_order_self_energy(
        omega=0.5,
        interaction=interaction,
        covariance=covariance,
    )
    assert np.max(np.abs(sigma_model - sigma_direct)) < 1e-12


def test_named_scba_and_fourth_order_lowest_order_hooks_are_present() -> None:
    interaction3 = ThirdOrderInteraction(phi3=np.zeros((2, 2, 2), dtype=np.complex128))
    interaction4 = FourthOrderInteraction(phi4=np.zeros((2, 2, 2, 2), dtype=np.complex128))

    sigma3 = third_order_scba_self_energy(
        omega=0.5,
        interaction=interaction3,
        green_function=np.eye(2, dtype=np.complex128),
        temperature=0.3,
        mode_frequencies=np.ones(2, dtype=float),
        mode_vectors=np.eye(2, dtype=np.complex128),
    )
    assert np.max(np.abs(sigma3)) == 0.0

    sigma4 = fourth_order_lowest_order_self_energy(
        omega=0.5,
        interaction=interaction4,
        covariance=np.eye(2, dtype=np.complex128),
    )
    assert np.max(np.abs(sigma4)) == 0.0

    sigma4_scba = fourth_order_scba_self_energy(
        omega=0.5,
        interaction=interaction4,
        green_function=np.eye(2, dtype=np.complex128),
        temperature=0.3,
        mode_frequencies=np.ones(2, dtype=float),
        mode_vectors=np.eye(2, dtype=np.complex128),
    )
    assert np.max(np.abs(sigma4_scba)) == 0.0
