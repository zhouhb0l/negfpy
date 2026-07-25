"""Interacting phonon heat-current and collision-power integrals."""

from __future__ import annotations

import numpy as np

from .grid import SignedFrequencyGrid
from .types import (
    CubicSCBAProblem,
    CurrentReport,
    KeldyshSpectrum,
    SolverScale,
)


Array = np.ndarray


def _validate_pair(
    self_energy: KeldyshSpectrum,
    green: KeldyshSpectrum,
    grid: SignedFrequencyGrid,
) -> None:
    expected_nw = grid.n_points
    if self_energy.lesser.shape != green.lesser.shape:
        raise ValueError("Self-energy and Green-function spectra must have matching shapes.")
    if self_energy.lesser.shape[0] != expected_nw:
        raise ValueError("Spectrum frequency dimension does not match the signed grid.")


def energy_exchange_spectrum(
    grid: SignedFrequencyGrid,
    self_energy: KeldyshSpectrum,
    green: KeldyshSpectrum,
) -> Array:
    """Return the dimensionless frequency-resolved energy exchange.

    The sign is positive when energy enters the device from the object
    represented by ``self_energy``.
    """

    _validate_pair(self_energy, green, grid)
    exchange = (
        self_energy.greater @ green.lesser
        - self_energy.lesser @ green.greater
    )
    trace = np.trace(exchange, axis1=-2, axis2=-1)
    return np.asarray(grid.values * np.real(trace), dtype=float)


def contact_heat_current(
    grid: SignedFrequencyGrid,
    sigma_contact: KeldyshSpectrum,
    green: KeldyshSpectrum,
    scale: SolverScale,
) -> float:
    """Integrate contact heat current, positive into the device.

    Only strictly positive frequencies are included and the signed grid's
    rectangle weight ``Delta omega / (2 pi)`` is used.
    """

    spectrum = energy_exchange_spectrum(grid, sigma_contact, green)
    dimensionless = float(
        grid.integration_weight * np.sum(spectrum[grid.positive_indices])
    )
    return scale.to_physical_current(dimensionless)


def collision_power(
    grid: SignedFrequencyGrid,
    sigma_pp: KeldyshSpectrum,
    green: KeldyshSpectrum,
    scale: SolverScale,
) -> float:
    """Return integrated energy transfer from the interaction self-energy."""

    return contact_heat_current(grid, sigma_pp, green, scale)


def _harmonic_bond_current_is_valid(problem: CubicSCBAProblem) -> bool:
    """Return whether no retained cubic vertex crosses a layer boundary."""

    explicit = problem.vertex.metadata.get(
        "harmonic_bond_current_supported",
        None,
    )
    npl = problem.device.dof_per_layer
    support = np.argwhere(np.abs(problem.vertex.values) > 0.0)
    for first, second, third in support:
        layers = {int(first) // npl, int(second) // npl, int(third) // npl}
        if len(layers) != 1:
            return False
    # Metadata may conservatively disable the operator for a topology with
    # additional interaction-current terms, but it may never override support
    # that demonstrably crosses a device-layer cut.
    return explicit is not False


def bond_heat_current_spectra(
    problem: CubicSCBAProblem,
    green: KeldyshSpectrum,
) -> Array:
    """Return harmonic energy-current spectra across each device cut.

    The result has shape ``(n_layers - 1, n_positive)`` and is expressed in
    physical current units per dimensionless angular-frequency unit.  It is a
    complete current operator only when the cubic interaction is onsite or
    strictly intralayer.  A bond-cubic vertex crossing a cut also contributes
    an interaction-current term and is therefore rejected.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(green, KeldyshSpectrum):
        raise TypeError("green must be a KeldyshSpectrum.")
    expected = (problem.grid.n_points, problem.ndof, problem.ndof)
    if green.shape != expected:
        raise ValueError(f"green must have shape {expected}; got {green.shape}.")
    if not _harmonic_bond_current_is_valid(problem):
        raise NotImplementedError(
            "The harmonic bond-current operator is incomplete because the "
            "cubic vertex crosses a layer boundary."
        )

    n_layers = problem.device.n_layers
    npl = problem.device.dof_per_layer
    positive = problem.grid.positive_indices
    spectra = np.zeros((max(0, n_layers - 1), positive.size), dtype=float)
    for layer, coupling_raw in enumerate(problem.device.coupling_blocks):
        coupling = np.asarray(coupling_raw, dtype=np.complex128)
        left = slice(layer * npl, (layer + 1) * npl)
        right = slice((layer + 1) * npl, (layer + 2) * npl)
        reverse = coupling.conj().T
        for column, iw in enumerate(positive):
            lesser = green.lesser[iw]
            bracket = np.trace(
                coupling @ lesser[right, left]
                - lesser[left, right] @ reverse
            )
            spectra[layer, column] = (
                -problem.scale.current_scale
                * problem.grid.values[iw]
                * float(np.real(bracket))
            )
    return spectra


def bond_heat_currents(
    problem: CubicSCBAProblem,
    green: KeldyshSpectrum,
) -> Array:
    """Integrate the valid harmonic bond-current spectra over frequency."""

    spectra = bond_heat_current_spectra(problem, green)
    return problem.grid.integration_weight * np.sum(spectra, axis=1)


def _evaluate_current_report(
    problem: CubicSCBAProblem,
    sigma_left: KeldyshSpectrum,
    sigma_right: KeldyshSpectrum,
    sigma_pp: KeldyshSpectrum,
    green: KeldyshSpectrum,
) -> CurrentReport:
    """Recompute the complete current report from stored field spectra."""

    current_left = contact_heat_current(
        problem.grid,
        sigma_left,
        green,
        problem.scale,
    )
    current_right = contact_heat_current(
        problem.grid,
        sigma_right,
        green,
        problem.scale,
    )
    power_pp = collision_power(
        problem.grid,
        sigma_pp,
        green,
        problem.scale,
    )
    positive = problem.grid.positive_indices
    spectrum_scale = problem.scale.current_scale
    contact_spectra = np.stack(
        (
            spectrum_scale
            * energy_exchange_spectrum(problem.grid, sigma_left, green)[positive],
            spectrum_scale
            * energy_exchange_spectrum(problem.grid, sigma_right, green)[positive],
        ),
        axis=0,
    )
    collision_spectrum = (
        spectrum_scale
        * energy_exchange_spectrum(problem.grid, sigma_pp, green)[positive]
    )
    try:
        bond_spectra = bond_heat_current_spectra(problem, green)
        bond_currents = (
            problem.grid.integration_weight * np.sum(bond_spectra, axis=1)
        )
    except NotImplementedError:
        bond_spectra = None
        bond_currents = None
    return CurrentReport(
        left=current_left,
        right=current_right,
        collision_power=power_pp,
        contact_spectra=contact_spectra,
        collision_spectrum=collision_spectrum,
        bond_currents=bond_currents,
        bond_spectra=bond_spectra,
    )


__all__ = [
    "bond_heat_current_spectra",
    "bond_heat_currents",
    "collision_power",
    "contact_heat_current",
    "energy_exchange_spectrum",
]
