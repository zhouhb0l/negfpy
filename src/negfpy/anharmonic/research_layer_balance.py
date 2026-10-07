"""Opt-in layer equation-of-motion diagnostics for arbitrary cubic topology.

These identities diagnose the supplied Dyson/Keldysh fields. They do not
evaluate the missing cubic Hamiltonian current across a layer cut, prove a
self-consistent fixed point, or change any solver acceptance decision.
"""

from __future__ import annotations

import numpy as np

from .current import (
    _harmonic_bond_current_is_valid,
    _validate_problem_pair,
    harmonic_cut_current_contribution_spectra,
    projected_collision_power_spectra,
    projected_eom_exchange_spectra,
)


def _maximum(*arrays):
    return max(float(np.max(np.abs(array), initial=0.0)) for array in arrays)


def evaluate_layer_energy_balance(
    problem,
    sigma_left,
    sigma_right,
    sigma_pp,
    green,
    *,
    rtol=1.0e-5,
    atol=None,
    eta_device=0.0,
):
    r"""Return a summary and spectra of the local Dyson equation balance.

    All fields must describe the same signed grid and device. For each layer
    the reported residual is ``div(J_h) - C_left - C_right - C_pp``, where
    ``C`` is the projected equation-of-motion exchange. Projecting contacts
    individually supports contacts attached to arbitrary device layers.
    ``div(J_h)`` is the difference of consecutive entries of
    ``[0, J_h[0], ..., J_h[n_layers-2], 0]``. The layer collision spectrum is
    also returned; it generally differs from ``C_pp`` for nonlocal fields.

    The original positive grid and its rectangle weight are used. Spectral
    and integrated residuals have separate screens with allowance
    ``atol + rtol * max(abs(each EOM or harmonic-divergence term))``. The
    same numerical absolute floor applies in physical current per solver
    frequency and physical current units, respectively; its default is
    ``1e-10 * problem.scale.current_scale``. This checks algebraic balance,
    not quadrature accuracy. Both screens must pass for ``eom_balance_passed``.

    Only ``eta_device=0`` is supported. A nonzero artificial device damping
    would add a separate source term and is rejected rather than omitted.
    The caller must supply the broadening used to obtain these fields.

    This opt-in operation computes no SCBA map and changes no input. It returns
    no full spatial-current pass, including when its EOM residual vanishes.
    """
    for sigma in (sigma_left, sigma_right, sigma_pp):
        _validate_problem_pair(problem, sigma, green)
    if not np.isfinite(eta_device) or eta_device != 0.0:
        raise ValueError("Layer energy balance requires eta_device=0.")
    if atol is None:
        atol = 1.0e-10 * problem.scale.current_scale
    if not np.isfinite(rtol) or rtol < 0.0:
        raise ValueError("rtol must be finite and non-negative.")
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be finite and non-negative.")

    harmonic = harmonic_cut_current_contribution_spectra(problem, green)
    contacts = np.stack([
        projected_eom_exchange_spectra(problem, sigma, green)
        for sigma in (sigma_left, sigma_right)
    ])
    interaction = projected_eom_exchange_spectra(problem, sigma_pp, green)
    collision = projected_collision_power_spectra(problem, sigma_pp, green)
    boundary_zero = np.zeros((1, problem.grid.n_positive), dtype=float)
    divergence = np.diff(
        np.concatenate((boundary_zero, harmonic, boundary_zero)), axis=0
    )
    residual = divergence - contacts.sum(axis=0) - interaction
    weight = problem.grid.integration_weight
    integrated_contacts = weight * contacts.sum(axis=-1)
    integrated_interaction = weight * interaction.sum(axis=-1)
    integrated_divergence = weight * divergence.sum(axis=-1)
    integrated_residual = weight * residual.sum(axis=-1)
    spectral_scale = _maximum(divergence, contacts, interaction)
    integrated_scale = _maximum(
        integrated_divergence, integrated_contacts, integrated_interaction
    )
    spectral_allowed = float(atol + rtol * spectral_scale)
    integrated_allowed = float(atol + rtol * integrated_scale)
    if not np.isfinite(spectral_allowed) or not np.isfinite(integrated_allowed):
        raise ValueError("Layer energy balance tolerances overflowed.")
    spectral_max = _maximum(residual)
    integrated_max = _maximum(integrated_residual)
    spectral_passed = spectral_max <= spectral_allowed
    integrated_passed = integrated_max <= integrated_allowed
    report = {
        "schema": "negfpy.research.layer-energy-balance.v1",
        "diagnostic_available": True,
        "equation_balance_only": True,
        "eom_balance_passed": bool(spectral_passed and integrated_passed),
        "spectral_balance_passed": bool(spectral_passed),
        "integrated_balance_passed": bool(integrated_passed),
        "spectral_max_residual": spectral_max,
        "spectral_scale": spectral_scale,
        "spectral_allowed": spectral_allowed,
        "integrated_max_residual": integrated_max,
        "integrated_scale": integrated_scale,
        "integrated_allowed": integrated_allowed,
        "rtol": float(rtol),
        "atol_physical": float(atol),
        "eta_device": 0.0,
        "quadrature": "original-positive-grid-rectangle",
        "harmonic_bond_operator_supported": _harmonic_bond_current_is_valid(problem),
        "full_spatial_current_available": False,
        "spatial_validation_passed": False,
        "changes_solver_acceptance": False,
        "global_frequency_accuracy_established": False,
    }
    arrays = {
        "positive_frequencies": problem.grid.values[problem.grid.positive_indices].copy(),
        "harmonic_cut_spectra": harmonic,
        "harmonic_cut_currents": weight * harmonic.sum(axis=-1),
        "harmonic_divergence_spectra": divergence,
        "contact_eom_spectra": contacts,
        "contact_eom_layer_exchange": integrated_contacts,
        "interaction_eom_spectra": interaction,
        "interaction_eom_layer_exchange": integrated_interaction,
        "layer_collision_spectra": collision,
        "layer_collision_power": weight * collision.sum(axis=-1),
        "eom_residual_spectra": residual,
        "eom_layer_residual": integrated_residual,
    }
    if not all(np.all(np.isfinite(array)) for array in arrays.values()):
        raise ValueError("Layer energy balance produced non-finite values.")
    return report, arrays


__all__ = ["evaluate_layer_energy_balance"]
