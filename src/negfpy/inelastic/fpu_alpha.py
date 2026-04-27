"""Toy-model utilities for the FPU-alpha chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.models import ChainParams, device_perfect_chain

from .third_order import (
    ThirdOrderInteraction,
    ThirdOrderLowestOrderModel,
    ThirdOrderSCBAModel,
    third_order_lowest_order_model_from_device,
    third_order_scba_model_from_device,
)


Array = np.ndarray


@dataclass(frozen=True)
class FPUAlphaParams:
    """Monoatomic FPU-alpha chain parameters.

    Potential per bond:
        V(x) = 0.5 * spring * x^2 + (alpha / 3) * x^3
    with x = u_{i+1} - u_i.
    """

    mass: float
    spring: float
    alpha: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.spring <= 0.0:
            raise ValueError("spring must be positive.")

    @property
    def harmonic_params(self) -> ChainParams:
        return ChainParams(mass=float(self.mass), spring=float(self.spring))


def fpu_alpha_third_order_interaction(n_layers: int, params: FPUAlphaParams) -> ThirdOrderInteraction:
    """Return the mass-normalized cubic interaction tensor for a finite chain."""

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    dim = int(n_layers)
    phi3 = np.zeros((dim, dim, dim), dtype=np.complex128)
    coeff = 2.0 * float(params.alpha) / (float(params.mass) ** 1.5)

    for left in range(n_layers - 1):
        right = left + 1
        indices = (left, right)
        signs = (-1.0, 1.0)
        for ia, a in enumerate(indices):
            for ib, b in enumerate(indices):
                for ic, c in enumerate(indices):
                    phi3[a, b, c] += coeff * signs[ia] * signs[ib] * signs[ic]

    return ThirdOrderInteraction(phi3=phi3)


def fpu_alpha_lowest_order_model(
    n_layers: int,
    params: FPUAlphaParams,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> ThirdOrderLowestOrderModel:
    """Return a lowest-order cubic self-energy model for a finite FPU-alpha chain."""

    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    interaction = fpu_alpha_third_order_interaction(n_layers=n_layers, params=params)
    return third_order_lowest_order_model_from_device(
        device=device,
        interaction=interaction,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )


def fpu_alpha_scba_model(
    n_layers: int,
    params: FPUAlphaParams,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> ThirdOrderSCBAModel:
    """Return a cubic SCBA-like self-energy model for a finite FPU-alpha chain."""

    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    interaction = fpu_alpha_third_order_interaction(n_layers=n_layers, params=params)
    return third_order_scba_model_from_device(
        device=device,
        interaction=interaction,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )
