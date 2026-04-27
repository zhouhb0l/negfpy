"""Simple phenomenological inelastic self-energies for toy-model validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


def _resolve_projector(projector: Array | None, dim: int) -> Array:
    if projector is None:
        return np.eye(dim, dtype=np.complex128)

    arr = np.asarray(projector, dtype=np.complex128)
    if arr.ndim == 1:
        if arr.size != dim:
            raise ValueError(
                "1D projector length must match device dimension "
                f"(got {arr.size}, expected {dim})."
            )
        if np.any(np.abs(arr.imag) > 1e-14):
            raise ValueError("1D projector weights must be real.")
        if np.any(arr.real < 0.0):
            raise ValueError("Projector weights must be non-negative.")
        return np.diag(arr)

    if arr.ndim == 2 and arr.shape == (dim, dim):
        return arr
    raise ValueError("projector must be None, 1D weights, or a (dim, dim) matrix.")


@dataclass(frozen=True)
class PowerLawPPSelfEnergy:
    """Phenomenological diagonal phonon-phonon retarded self-energy."""

    gamma0: float
    omega_ref: float = 1.0
    power: float = 2.0
    projector: Array | None = None
    min_gamma: float = 0.0
    max_gamma: float | None = None

    def __post_init__(self) -> None:
        if self.gamma0 < 0.0:
            raise ValueError("gamma0 must be non-negative.")
        if self.omega_ref <= 0.0:
            raise ValueError("omega_ref must be positive.")
        if self.min_gamma < 0.0:
            raise ValueError("min_gamma must be non-negative.")
        if self.max_gamma is not None and self.max_gamma <= 0.0:
            raise ValueError("max_gamma must be positive when provided.")
        if self.max_gamma is not None and self.max_gamma < self.min_gamma:
            raise ValueError("max_gamma must be >= min_gamma.")

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del iteration
        dim = green_function.shape[0]
        proj = _resolve_projector(self.projector, dim)

        ratio = abs(float(omega)) / float(self.omega_ref)
        gamma = float(self.gamma0) * (ratio**float(self.power))
        gamma = max(float(self.min_gamma), gamma)
        if self.max_gamma is not None:
            gamma = min(gamma, float(self.max_gamma))

        return -2j * abs(float(omega)) * gamma * proj
