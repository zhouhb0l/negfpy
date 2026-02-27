"""Full counting statistics utilities for coherent phonon heat transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


Array = np.ndarray

# SI constants.
HBAR = 1.054_571_817e-34  # J*s
KB = 1.380_649e-23  # J/K


@dataclass(frozen=True)
class HeatCurrentCumulants:
    """First and second cumulants of heat current (per unit time)."""

    c1_j_per_s: float
    c2_j2_per_s: float


@dataclass(frozen=True)
class HeatCurrentUncertainty:
    """Finite-time uncertainty derived from the second cumulant."""

    measurement_time_s: float
    std_current_j_per_s: float
    variance_energy_j2: float
    std_energy_j: float


def _to_1d_array(values: Iterable[float], name: str) -> Array:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _phi1(x: Array) -> Array:
    # x / expm1(x), stable near x=0 and at large x.
    out = np.zeros_like(x, dtype=float)
    ax = np.abs(x)
    small = ax < 1e-6
    mid = (ax >= 1e-6) & (x < 700.0)
    xs = x[small]
    out[small] = 1.0 - 0.5 * xs + (xs * xs) / 12.0 - (xs**4) / 720.0
    out[mid] = x[mid] / np.expm1(x[mid])
    return out


def _phi2(x: Array) -> Array:
    # x^2 * exp(x) / expm1(x)^2, stable near x=0 and at large x.
    out = np.zeros_like(x, dtype=float)
    ax = np.abs(x)
    small = ax < 1e-6
    mid = (ax >= 1e-6) & (x < 700.0)
    xs = x[small]
    out[small] = 1.0 - (xs * xs) / 12.0 + (xs**4) / 240.0
    em1 = np.expm1(x[mid])
    out[mid] = (x[mid] * x[mid]) * (em1 + 1.0) / (em1 * em1)
    return out


def _a_bose(omega_rad_s: Array, temperature_k: float) -> Array:
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive in Kelvin.")
    x = (HBAR * omega_rad_s) / (KB * temperature_k)
    return KB * temperature_k * _phi1(x)


def _b_bose(omega_rad_s: Array, temperature_k: float) -> Array:
    if temperature_k <= 0.0:
        raise ValueError("Temperature must be positive in Kelvin.")
    x = (HBAR * omega_rad_s) / (KB * temperature_k)
    s = KB * temperature_k
    return (s * s) * _phi2(x)


def heat_current_cumulants_from_spectrum(
    omega_rad_s: Iterable[float],
    transmission_vals: Iterable[float],
    temp_left_k: float,
    temp_right_k: float,
) -> HeatCurrentCumulants:
    """Return C1 and C2 for coherent phonon transport.

    Uses the Levitov-style bosonic formulas in terms of transmission and
    Bose functions. Output units are:
    - C1: J/s
    - C2: J^2/s
    """

    omega = _to_1d_array(omega_rad_s, "omega_rad_s")
    t = np.asarray(list(transmission_vals), dtype=float)
    if t.shape != omega.shape:
        raise ValueError("transmission_vals must have the same length as omega_rad_s.")
    if np.any(t < 0.0):
        raise ValueError("transmission_vals must be non-negative.")

    a_l = _a_bose(omega, temp_left_k)
    a_r = _a_bose(omega, temp_right_k)
    b_l = _b_bose(omega, temp_left_k)
    b_r = _b_bose(omega, temp_right_k)

    ad = a_l - a_r
    pref = 1.0 / (2.0 * np.pi)
    c1 = pref * np.trapezoid(t * ad, omega)
    c2 = pref * np.trapezoid(t * (b_l + b_r) + t * (1.0 + t) * (ad * ad), omega)
    return HeatCurrentCumulants(c1_j_per_s=float(c1), c2_j2_per_s=float(c2))


def heat_current_cumulants_from_k_moments(
    omega_rad_s: Iterable[float],
    t_mean_vs_omega: Iterable[float],
    t2_mean_vs_omega: Iterable[float],
    temp_left_k: float,
    temp_right_k: float,
) -> HeatCurrentCumulants:
    """Return exact C1 and C2 from k-resolved transmission moments.

    C2 uses <T> and <T^2> through:
    C2 ~ <T>(B_L+B_R) + (<T>+<T^2>)(A_L-A_R)^2.
    """

    omega = _to_1d_array(omega_rad_s, "omega_rad_s")
    t1 = np.asarray(list(t_mean_vs_omega), dtype=float)
    t2 = np.asarray(list(t2_mean_vs_omega), dtype=float)
    if t1.shape != omega.shape or t2.shape != omega.shape:
        raise ValueError("t_mean_vs_omega and t2_mean_vs_omega must match omega_rad_s shape.")
    if np.any(t1 < 0.0) or np.any(t2 < 0.0):
        raise ValueError("Transmission moments must be non-negative.")

    a_l = _a_bose(omega, temp_left_k)
    a_r = _a_bose(omega, temp_right_k)
    b_l = _b_bose(omega, temp_left_k)
    b_r = _b_bose(omega, temp_right_k)

    ad = a_l - a_r
    pref = 1.0 / (2.0 * np.pi)
    c1 = pref * np.trapezoid(t1 * ad, omega)
    c2 = pref * np.trapezoid(t1 * (b_l + b_r) + (t1 + t2) * (ad * ad), omega)
    return HeatCurrentCumulants(c1_j_per_s=float(c1), c2_j2_per_s=float(c2))


def heat_current_uncertainty(
    cumulants: HeatCurrentCumulants,
    *,
    measurement_time_s: float,
) -> HeatCurrentUncertainty:
    """Return finite-time uncertainty metrics from C2."""

    tau = float(measurement_time_s)
    if tau <= 0.0:
        raise ValueError("measurement_time_s must be positive.")
    var_e = float(cumulants.c2_j2_per_s * tau)
    std_e = float(np.sqrt(max(var_e, 0.0)))
    std_j = float(np.sqrt(max(cumulants.c2_j2_per_s / tau, 0.0)))
    return HeatCurrentUncertainty(
        measurement_time_s=tau,
        std_current_j_per_s=std_j,
        variance_energy_j2=var_e,
        std_energy_j=std_e,
    )
