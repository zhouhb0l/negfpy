import numpy as np

from negfpy.core.fcs import (
    HeatCurrentCumulants,
    heat_current_cumulants_from_k_moments,
    heat_current_cumulants_from_spectrum,
    heat_current_uncertainty,
)


def test_fcs_equilibrium_zero_mean_current() -> None:
    omega = np.linspace(1.0e11, 2.0e14, 400)
    t = np.full_like(omega, 0.5)
    c = heat_current_cumulants_from_spectrum(
        omega_rad_s=omega,
        transmission_vals=t,
        temp_left_k=300.0,
        temp_right_k=300.0,
    )
    assert abs(c.c1_j_per_s) < 1.0e-18
    assert c.c2_j2_per_s >= 0.0


def test_fcs_k_moment_reduces_to_spectrum_when_t2_equals_t_squared() -> None:
    omega = np.linspace(1.0e11, 1.0e14, 300)
    t = 0.2 + 0.1 * np.sin(np.linspace(0.0, 3.0, omega.size))
    c_ref = heat_current_cumulants_from_spectrum(
        omega_rad_s=omega,
        transmission_vals=t,
        temp_left_k=310.0,
        temp_right_k=295.0,
    )
    c_k = heat_current_cumulants_from_k_moments(
        omega_rad_s=omega,
        t_mean_vs_omega=t,
        t2_mean_vs_omega=t * t,
        temp_left_k=310.0,
        temp_right_k=295.0,
    )
    assert np.isclose(c_ref.c1_j_per_s, c_k.c1_j_per_s, rtol=1e-12, atol=1e-20)
    assert c_k.c2_j2_per_s >= c_ref.c2_j2_per_s


def test_fcs_uncertainty_scales_with_time() -> None:
    c = HeatCurrentCumulants(c1_j_per_s=1.0, c2_j2_per_s=4.0)
    u1 = heat_current_uncertainty(c, measurement_time_s=1.0)
    u4 = heat_current_uncertainty(c, measurement_time_s=4.0)
    assert np.isclose(u1.std_current_j_per_s, 2.0)
    assert np.isclose(u4.std_current_j_per_s, 1.0)
    assert np.isclose(u4.std_energy_j, 4.0)

