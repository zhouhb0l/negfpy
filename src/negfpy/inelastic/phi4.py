"""Toy-model utilities for an onsite quartic (phi^4) chain."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.models import ChainParams, device_perfect_chain

from .fourth_order import (
    FourthOrderInteraction,
    FourthOrderLowestOrderModel,
    FourthOrderMeanFieldModel,
    FourthOrderSCBAModel,
    fourth_order_lowest_order_model_from_device,
    fourth_order_mean_field_model_from_device,
    fourth_order_scba_model_from_device,
)


Array = np.ndarray


@dataclass(frozen=True)
class Phi4Params:
    """Monoatomic onsite quartic chain parameters.

    Potential per site:
        V(u_i) = (lambda4 / 4) * x_i^4
    with x_i = u_i / sqrt(mass).
    """

    mass: float
    spring: float
    lambda4: float

    def __post_init__(self) -> None:
        if self.mass <= 0.0:
            raise ValueError("mass must be positive.")
        if self.spring <= 0.0:
            raise ValueError("spring must be positive.")
        if self.lambda4 < 0.0:
            raise ValueError("lambda4 must be non-negative.")

    @property
    def harmonic_params(self) -> ChainParams:
        return ChainParams(mass=float(self.mass), spring=float(self.spring))


def phi4_fourth_order_interaction(n_layers: int, params: Phi4Params) -> FourthOrderInteraction:
    """Return the mass-normalized quartic interaction tensor for a finite chain."""

    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")

    dim = int(n_layers)
    phi4 = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    coeff = float(params.lambda4) / (float(params.mass) ** 2)
    for i in range(dim):
        phi4[i, i, i, i] = coeff
    return FourthOrderInteraction(phi4=phi4)


def phi4_mean_field_model(
    n_layers: int,
    params: Phi4Params,
    *,
    temperature: float,
    max_iter: int = 100,
    mixing: float = 0.5,
    tol: float = 1e-8,
    frequency_floor: float = 1e-8,
) -> FourthOrderMeanFieldModel:
    """Return a quartic mean-field model for a finite onsite-phi^4 chain."""

    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    interaction = phi4_fourth_order_interaction(n_layers=n_layers, params=params)
    return fourth_order_mean_field_model_from_device(
        device=device,
        interaction=interaction,
        temperature=temperature,
        max_iter=max_iter,
        mixing=mixing,
        tol=tol,
        frequency_floor=frequency_floor,
    )


def phi4_lowest_order_model(
    n_layers: int,
    params: Phi4Params,
    *,
    temperature: float,
    frequency_floor: float = 1e-8,
) -> FourthOrderLowestOrderModel:
    """Return a quartic lowest-order model for a finite onsite-phi^4 chain."""

    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    interaction = phi4_fourth_order_interaction(n_layers=n_layers, params=params)
    return fourth_order_lowest_order_model_from_device(
        device=device,
        interaction=interaction,
        temperature=temperature,
        frequency_floor=frequency_floor,
    )


def phi4_scba_model(
    n_layers: int,
    params: Phi4Params,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> FourthOrderSCBAModel:
    """Return a quartic SCBA-like model for a finite onsite-phi^4 chain."""

    device = device_perfect_chain(n_layers=n_layers, params=params.harmonic_params)
    interaction = phi4_fourth_order_interaction(n_layers=n_layers, params=params)
    return fourth_order_scba_model_from_device(
        device=device,
        interaction=interaction,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )
