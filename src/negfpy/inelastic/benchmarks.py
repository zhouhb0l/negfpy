"""Benchmark helpers for inelastic toy-model studies."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from negfpy.core import heat_current_from_spectrum
from negfpy.models import analytic_band_max, device_perfect_chain, lead_blocks

from .fpu_alpha import FPUAlphaParams, fpu_alpha_lowest_order_model
from .phi4 import Phi4Params, phi4_lowest_order_model, phi4_mean_field_model, phi4_scba_model
from .solver import transmission_inelastic


Array = np.ndarray


def _to_1d_temperature_array(values: Iterable[float]) -> Array:
    temps = np.asarray(list(values), dtype=float)
    if temps.ndim != 1 or temps.size == 0:
        raise ValueError("temperatures must be a non-empty 1D array.")
    if np.any(temps <= 0.0):
        raise ValueError("temperatures must be positive.")
    return temps


def _normalize_approximation(name: str) -> str:
    key = str(name).strip().lower()
    aliases = {
        "lo": "lowest_order",
        "lowest_order": "lowest_order",
        "lowest-order": "lowest_order",
        "mf": "mean_field",
        "mean_field": "mean_field",
        "mean-field": "mean_field",
        "scba": "scba",
    }
    if key not in aliases:
        raise ValueError(
            "approximation must be one of "
            "'lowest_order'/'lo', 'mean_field'/'mf', or 'scba'."
        )
    return aliases[key]


def _review_style_temperature_bias(
    temperatures: Array,
    *,
    bias_fraction: float,
) -> tuple[Array, Array]:
    if not (0.0 < bias_fraction < 1.0):
        raise ValueError("bias_fraction must lie in (0, 1).")
    temp_left = (1.0 + float(bias_fraction)) * temperatures
    temp_right = (1.0 - float(bias_fraction)) * temperatures
    return temp_left, temp_right


def _sweep_transmission_spectrum(
    omegas: Array,
    *,
    device: object,
    lead: object,
    eta: float,
    pp_self_energy: object,
    max_iter: int,
    mixing: float,
    tol: float,
) -> tuple[Array, dict[str, object]]:
    transmission_vals = np.zeros_like(omegas, dtype=float)
    converged_flags = np.zeros_like(omegas, dtype=bool)
    iterations = np.zeros_like(omegas, dtype=int)
    residuals = np.zeros_like(omegas, dtype=float)
    sigma_norms = np.zeros_like(omegas, dtype=float)

    for i, omega in enumerate(omegas):
        tval, info = transmission_inelastic(
            omega=float(omega),
            device=device,
            lead_left=lead,
            lead_right=lead,
            eta=float(eta),
            pp_self_energy=pp_self_energy,
            max_iter=int(max_iter),
            mixing=float(mixing),
            tol=float(tol),
        )
        transmission_vals[i] = max(float(tval), 0.0)
        converged_flags[i] = bool(info["converged"])
        iterations[i] = int(info["iterations"])
        residuals[i] = float(info["residual"])
        sigma_norms[i] = float(info["sigma_pp_norm"])

    summary = {
        "converged_all": bool(np.all(converged_flags)),
        "converged_fraction": float(np.mean(converged_flags)),
        "iterations_mean": float(np.mean(iterations)),
        "iterations_max": int(np.max(iterations)),
        "residual_max": float(np.max(residuals)),
        "sigma_pp_norm_max": float(np.max(sigma_norms)),
    }
    return transmission_vals, summary


def fpu_alpha_lowest_order_conductance_sweep(
    temperatures: Iterable[float],
    *,
    n_layers: int,
    params: FPUAlphaParams,
    n_omega: int = 200,
    omega_max_factor: float = 1.05,
    eta: float = 1e-8,
    broadening: float = 0.03,
    bias_fraction: float = 0.25,
) -> dict[str, Array]:
    """Return a review-style conductance-vs-temperature benchmark sweep.

    The bath temperatures follow the symmetric review-style convention
    ``T_L = (1 + bias_fraction) T_avg`` and ``T_R = (1 - bias_fraction) T_avg``.
    The anharmonic self-energy model is evaluated at the average temperature.
    """

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if n_omega < 2:
        raise ValueError("n_omega must be at least 2.")
    if omega_max_factor <= 0.0:
        raise ValueError("omega_max_factor must be positive.")
    if eta <= 0.0:
        raise ValueError("eta must be positive.")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive.")
    temps = _to_1d_temperature_array(temperatures)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    wmax = analytic_band_max(params.harmonic_params)
    omegas = np.linspace(0.02 * wmax, float(omega_max_factor) * wmax, int(n_omega))

    temp_left, temp_right = _review_style_temperature_bias(temps, bias_fraction=bias_fraction)
    heat_current = np.zeros_like(temps)
    conductance = np.zeros_like(temps)
    transmission_map = np.zeros((temps.size, omegas.size), dtype=float)

    for i, temp_avg in enumerate(temps):
        model = fpu_alpha_lowest_order_model(
            n_layers=n_layers,
            params=params,
            temperature=float(temp_avg),
            broadening=broadening,
        )
        transmission_vals = np.array(
            [
                transmission_inelastic(
                    omega=float(omega),
                    device=device,
                    lead_left=lead,
                    lead_right=lead,
                    eta=float(eta),
                    pp_self_energy=model,
                    max_iter=1,
                    mixing=1.0,
                    tol=1e-12,
                )[0]
                for omega in omegas
            ],
            dtype=float,
        )
        transmission_vals = np.clip(transmission_vals, a_min=0.0, a_max=None)
        transmission_map[i, :] = transmission_vals

        current = heat_current_from_spectrum(
            omegas=omegas,
            transmission_vals=transmission_vals,
            temp_left=float(temp_left[i]),
            temp_right=float(temp_right[i]),
        )
        heat_current[i] = current
        conductance[i] = current / float(temp_left[i] - temp_right[i])

    return {
        "temperature_avg": temps,
        "temperature_left": temp_left,
        "temperature_right": temp_right,
        "omegas": omegas,
        "transmission_map": transmission_map,
        "heat_current": heat_current,
        "conductance": conductance,
    }


def phi4_conductance_sweep(
    temperatures: Iterable[float],
    *,
    n_layers: int,
    params: Phi4Params,
    approximation: str,
    n_omega: int = 200,
    omega_max_factor: float = 1.05,
    eta: float = 1e-8,
    bias_fraction: float = 0.25,
    broadening: float = 0.03,
    frequency_floor: float = 1e-8,
    mean_field_max_iter: int = 100,
    mean_field_mixing: float = 0.5,
    mean_field_tol: float = 1e-8,
    transport_max_iter: int | None = None,
    transport_mixing: float | None = None,
    transport_tol: float | None = None,
) -> dict[str, object]:
    """Return a review-style quartic conductance-vs-temperature sweep.

    The benchmark uses the Wang-style symmetric temperature protocol
    ``T_L = (1 + bias_fraction) T_avg`` and ``T_R = (1 - bias_fraction) T_avg``.
    The current toy-model implementation still uses the project's harmonic chain
    leads rather than the exact Lorentz-Drude baths from the literature, so this
    helper is best viewed as a clean approximation-comparison benchmark inside
    the current codebase.
    """

    approx = _normalize_approximation(approximation)
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if n_omega < 2:
        raise ValueError("n_omega must be at least 2.")
    if omega_max_factor <= 0.0:
        raise ValueError("omega_max_factor must be positive.")
    if eta <= 0.0:
        raise ValueError("eta must be positive.")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    temps = _to_1d_temperature_array(temperatures)
    temp_left, temp_right = _review_style_temperature_bias(temps, bias_fraction=bias_fraction)
    lead = lead_blocks(params.harmonic_params)
    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    wmax = analytic_band_max(params.harmonic_params)
    omegas = np.linspace(0.02 * wmax, float(omega_max_factor) * wmax, int(n_omega))

    if approx == "scba":
        sweep_max_iter = 20 if transport_max_iter is None else int(transport_max_iter)
        sweep_mixing = 0.5 if transport_mixing is None else float(transport_mixing)
        sweep_tol = 1e-5 if transport_tol is None else float(transport_tol)
    else:
        sweep_max_iter = 1 if transport_max_iter is None else int(transport_max_iter)
        sweep_mixing = 1.0 if transport_mixing is None else float(transport_mixing)
        sweep_tol = 1e-12 if transport_tol is None else float(transport_tol)

    heat_current = np.zeros_like(temps)
    conductance = np.zeros_like(temps)
    transmission_map = np.zeros((temps.size, omegas.size), dtype=float)
    converged_all = np.zeros_like(temps, dtype=bool)
    converged_fraction = np.zeros_like(temps)
    iterations_mean = np.zeros_like(temps)
    iterations_max = np.zeros_like(temps, dtype=int)
    residual_max = np.zeros_like(temps)
    sigma_pp_norm_max = np.zeros_like(temps)

    for i, temp_avg in enumerate(temps):
        if approx == "lowest_order":
            model = phi4_lowest_order_model(
                n_layers=n_layers,
                params=params,
                temperature=float(temp_avg),
                frequency_floor=frequency_floor,
            )
        elif approx == "mean_field":
            model = phi4_mean_field_model(
                n_layers=n_layers,
                params=params,
                temperature=float(temp_avg),
                max_iter=mean_field_max_iter,
                mixing=mean_field_mixing,
                tol=mean_field_tol,
                frequency_floor=frequency_floor,
            )
        else:
            model = phi4_scba_model(
                n_layers=n_layers,
                params=params,
                temperature=float(temp_avg),
                broadening=broadening,
                frequency_floor=frequency_floor,
            )

        transmission_vals, summary = _sweep_transmission_spectrum(
            omegas,
            device=device,
            lead=lead,
            eta=eta,
            pp_self_energy=model,
            max_iter=sweep_max_iter,
            mixing=sweep_mixing,
            tol=sweep_tol,
        )
        transmission_map[i, :] = transmission_vals
        converged_all[i] = bool(summary["converged_all"])
        converged_fraction[i] = float(summary["converged_fraction"])
        iterations_mean[i] = float(summary["iterations_mean"])
        iterations_max[i] = int(summary["iterations_max"])
        residual_max[i] = float(summary["residual_max"])
        sigma_pp_norm_max[i] = float(summary["sigma_pp_norm_max"])

        current = heat_current_from_spectrum(
            omegas=omegas,
            transmission_vals=transmission_vals,
            temp_left=float(temp_left[i]),
            temp_right=float(temp_right[i]),
        )
        heat_current[i] = current
        conductance[i] = current / float(temp_left[i] - temp_right[i])

    return {
        "approximation": approx,
        "temperature_avg": temps,
        "temperature_left": temp_left,
        "temperature_right": temp_right,
        "omegas": omegas,
        "transmission_map": transmission_map,
        "heat_current": heat_current,
        "conductance": conductance,
        "converged_all": converged_all,
        "converged_fraction": converged_fraction,
        "iterations_mean": iterations_mean,
        "iterations_max": iterations_max,
        "residual_max": residual_max,
        "sigma_pp_norm_max": sigma_pp_norm_max,
    }


def phi4_compare_conductance_sweeps(
    temperatures: Iterable[float],
    *,
    approximations: Iterable[str] = ("lowest_order", "mean_field", "scba"),
    n_layers: int,
    params: Phi4Params,
    n_omega: int = 200,
    omega_max_factor: float = 1.05,
    eta: float = 1e-8,
    bias_fraction: float = 0.25,
    broadening: float = 0.03,
    frequency_floor: float = 1e-8,
    mean_field_max_iter: int = 100,
    mean_field_mixing: float = 0.5,
    mean_field_tol: float = 1e-8,
    transport_max_iter: int | None = None,
    transport_mixing: float | None = None,
    transport_tol: float | None = None,
) -> dict[str, dict[str, object]]:
    """Return a bundle of quartic benchmark sweeps for selected approximations."""

    results: dict[str, dict[str, object]] = {}
    for name in approximations:
        approx = _normalize_approximation(name)
        results[approx] = phi4_conductance_sweep(
            temperatures=temperatures,
            n_layers=n_layers,
            params=params,
            approximation=approx,
            n_omega=n_omega,
            omega_max_factor=omega_max_factor,
            eta=eta,
            bias_fraction=bias_fraction,
            broadening=broadening,
            frequency_floor=frequency_floor,
            mean_field_max_iter=mean_field_max_iter,
            mean_field_mixing=mean_field_mixing,
            mean_field_tol=mean_field_tol,
            transport_max_iter=transport_max_iter,
            transport_mixing=transport_mixing,
            transport_tol=transport_tol,
        )
    return results
