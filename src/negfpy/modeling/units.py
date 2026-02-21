"""Unit conversion helpers for model frequencies."""

from __future__ import annotations

import numpy as np


# CODATA 2018 constants (SI).
RYDBERG_J = 2.1798723611035e-18
BOHR_M = 5.29177210903e-11
ELECTRON_MASS_KG = 9.1093837015e-31
QE_RY_MASS_UNIT_KG = 2.0 * ELECTRON_MASS_KG
SPEED_OF_LIGHT_CM_S = 2.99792458e10


def qe_omega_to_rad_s(omega: np.ndarray | float) -> np.ndarray | float:
    """Convert internal QE-q2r omega to angular frequency [rad/s].

    Assumes IFC blocks are in Ry/Bohr^2 and masses are in QE Ry-mass units
    (``2 * electron_mass``).
    """

    factor = np.sqrt(RYDBERG_J / (QE_RY_MASS_UNIT_KG * BOHR_M * BOHR_M))
    return np.asarray(omega) * factor


def qe_omega_to_thz(omega: np.ndarray | float) -> np.ndarray | float:
    """Convert internal QE-q2r omega to frequency [THz]."""

    w = np.asarray(qe_omega_to_rad_s(omega), dtype=float)
    return w / (2.0 * np.pi * 1.0e12)


def qe_omega_to_cm1(omega: np.ndarray | float) -> np.ndarray | float:
    """Convert internal QE-q2r omega to wavenumber [cm^-1]."""

    w = np.asarray(qe_omega_to_rad_s(omega), dtype=float)
    return w / (2.0 * np.pi * SPEED_OF_LIGHT_CM_S)
