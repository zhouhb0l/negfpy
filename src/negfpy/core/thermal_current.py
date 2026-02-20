"""Thermal current utilities from coherent phonon transmission spectra."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .negf import DeviceLike, LeadLike, transmission, transmission_kavg


Array = np.ndarray

# SI constants.
HBAR = 1.054_571_817e-34  # J*s
KB = 1.380_649e-23  # J/K


def _to_1d_array(values: Iterable[float], name: str) -> Array:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1D.")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two points.")
    if np.any(np.diff(arr) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return arr


def _spectral_bose_weight(omegas: Array, temp_left: float, temp_right: float) -> Array:
    if temp_left <= 0.0 or temp_right <= 0.0:
        raise ValueError("Temperatures must be positive in Kelvin.")

    omega = np.asarray(omegas, dtype=float)
    x_l = HBAR * omega / (KB * temp_left)
    x_r = HBAR * omega / (KB * temp_right)

    # Use x/expm1(x): finite at x=0 with limit 1.
    def _phi(x: Array) -> Array:
        out = np.zeros_like(x)
        small = x < 1e-6
        mid = (x >= 1e-6) & (x < 700.0)
        out[small] = 1.0 - 0.5 * x[small] + (x[small] ** 2) / 12.0
        out[mid] = x[mid] / np.expm1(x[mid])
        return out

    return KB * temp_left * _phi(x_l) - KB * temp_right * _phi(x_r)


def heat_current_from_spectrum(
    omegas: Iterable[float],
    transmission_vals: Iterable[float],
    temp_left: float,
    temp_right: float,
) -> float:
    """Return phonon heat current from T(omega) via the Landauer formula.

    The input angular frequencies must be in rad/s, and the output is in Watts.
    """

    omega = _to_1d_array(omegas, "omegas")
    tvals = np.asarray(list(transmission_vals), dtype=float)
    if tvals.shape != omega.shape:
        raise ValueError("transmission_vals must have the same length as omegas.")

    spectral = _spectral_bose_weight(omega, temp_left=temp_left, temp_right=temp_right) * tvals
    return float((1.0 / (2.0 * np.pi)) * np.trapz(spectral, omega))


def heat_current_1d(
    omegas: Iterable[float],
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    temp_left: float,
    temp_right: float,
    eta: float = 1e-8,
) -> float:
    """Return 1D phonon heat current (W)."""

    omega = _to_1d_array(omegas, "omegas")
    tvals = np.array(
        [
            transmission(
                omega=float(w),
                device=device,
                lead_left=lead_left,
                lead_right=lead_right,
                eta=eta,
            )
            for w in omega
        ],
        dtype=float,
    )
    return heat_current_from_spectrum(
        omegas=omega,
        transmission_vals=tvals,
        temp_left=temp_left,
        temp_right=temp_right,
    )


def heat_current_per_length_2d(
    omegas: Iterable[float],
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    transverse_length: float,
    temp_left: float,
    temp_right: float,
    eta: float = 1e-8,
) -> float:
    """Return 2D phonon heat current density per transverse length (W/m)."""

    if transverse_length <= 0.0:
        raise ValueError("transverse_length must be positive.")
    omega = _to_1d_array(omegas, "omegas")
    tavg = np.array(
        [
            transmission_kavg(
                omega=float(w),
                device=device,
                lead_left=lead_left,
                lead_right=lead_right,
                kpoints=kpoints,
                eta=eta,
            )
            for w in omega
        ],
        dtype=float,
    )
    j = heat_current_from_spectrum(
        omegas=omega,
        transmission_vals=tavg,
        temp_left=temp_left,
        temp_right=temp_right,
    )
    return float(j / transverse_length)


def heat_current_density_3d(
    omegas: Iterable[float],
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    transverse_area: float,
    temp_left: float,
    temp_right: float,
    eta: float = 1e-8,
) -> float:
    """Return 3D phonon heat current density per transverse area (W/m^2)."""

    if transverse_area <= 0.0:
        raise ValueError("transverse_area must be positive.")
    omega = _to_1d_array(omegas, "omegas")
    tavg = np.array(
        [
            transmission_kavg(
                omega=float(w),
                device=device,
                lead_left=lead_left,
                lead_right=lead_right,
                kpoints=kpoints,
                eta=eta,
            )
            for w in omega
        ],
        dtype=float,
    )
    j = heat_current_from_spectrum(
        omegas=omega,
        transmission_vals=tavg,
        temp_left=temp_left,
        temp_right=temp_right,
    )
    return float(j / transverse_area)


def transverse_length_from_vector(
    transverse_vec: Iterable[float],
    transport_dir: Iterable[float],
) -> float:
    """Return the transverse periodic length orthogonal to transport."""

    t = np.asarray(list(transverse_vec), dtype=float)
    e = np.asarray(list(transport_dir), dtype=float)
    if t.ndim != 1 or e.ndim != 1 or t.shape != e.shape:
        raise ValueError("transverse_vec and transport_dir must be 1D with the same size.")
    en = np.linalg.norm(e)
    if en == 0.0:
        raise ValueError("transport_dir must be non-zero.")
    ehat = e / en
    t_perp = t - np.dot(t, ehat) * ehat
    length = float(np.linalg.norm(t_perp))
    if length == 0.0:
        raise ValueError("transverse_vec has zero component orthogonal to transport_dir.")
    return length


def transverse_area_from_vectors(
    transverse_vec_1: Iterable[float],
    transverse_vec_2: Iterable[float],
    transport_dir: Iterable[float],
) -> float:
    """Return projected transverse cell area normal to transport."""

    t1 = np.asarray(list(transverse_vec_1), dtype=float)
    t2 = np.asarray(list(transverse_vec_2), dtype=float)
    e = np.asarray(list(transport_dir), dtype=float)
    if t1.shape != (3,) or t2.shape != (3,) or e.shape != (3,):
        raise ValueError("All vectors must be 3D.")
    en = np.linalg.norm(e)
    if en == 0.0:
        raise ValueError("transport_dir must be non-zero.")
    ehat = e / en
    return float(abs(np.dot(np.cross(t1, t2), ehat)))
