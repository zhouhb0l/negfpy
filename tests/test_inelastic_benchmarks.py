import numpy as np

from negfpy.inelastic import (
    FPUAlphaParams,
    Phi4Params,
    fpu_alpha_lowest_order_conductance_sweep,
    phi4_compare_conductance_sweeps,
    phi4_conductance_sweep,
)


def test_fpu_alpha_lowest_order_conductance_sweep_returns_finite_arrays() -> None:
    result = fpu_alpha_lowest_order_conductance_sweep(
        temperatures=[100.0, 200.0, 300.0],
        n_layers=8,
        params=FPUAlphaParams(mass=1.0, spring=1.0, alpha=0.05),
        n_omega=40,
        broadening=0.03,
    )

    assert result["temperature_avg"].shape == (3,)
    assert result["temperature_left"].shape == (3,)
    assert result["temperature_right"].shape == (3,)
    assert result["heat_current"].shape == (3,)
    assert result["conductance"].shape == (3,)
    assert result["transmission_map"].shape == (3, 40)
    assert np.all(np.isfinite(result["heat_current"]))
    assert np.all(np.isfinite(result["conductance"]))
    assert np.all(result["conductance"] >= 0.0)


def test_phi4_mean_field_conductance_sweep_returns_finite_arrays() -> None:
    result = phi4_conductance_sweep(
        temperatures=[100.0, 200.0, 300.0],
        n_layers=6,
        params=Phi4Params(mass=1.0, spring=1.0, lambda4=0.05),
        approximation="mean_field",
        n_omega=32,
        broadening=0.02,
    )

    assert result["approximation"] == "mean_field"
    assert result["temperature_avg"].shape == (3,)
    assert result["conductance"].shape == (3,)
    assert result["transmission_map"].shape == (3, 32)
    assert result["converged_all"].shape == (3,)
    assert np.all(np.isfinite(result["heat_current"]))
    assert np.all(np.isfinite(result["conductance"]))
    assert np.all(result["conductance"] >= 0.0)


def test_phi4_compare_conductance_sweeps_includes_scba() -> None:
    result = phi4_compare_conductance_sweeps(
        temperatures=[150.0, 250.0],
        approximations=("lowest_order", "scba"),
        n_layers=5,
        params=Phi4Params(mass=1.0, spring=1.0, lambda4=0.03),
        n_omega=16,
        broadening=0.02,
        transport_max_iter=8,
        transport_tol=1e-4,
    )

    assert set(result.keys()) == {"lowest_order", "scba"}
    assert result["scba"]["conductance"].shape == (2,)
    assert np.all(np.isfinite(result["scba"]["conductance"]))
    assert np.all(result["scba"]["conductance"] >= 0.0)
