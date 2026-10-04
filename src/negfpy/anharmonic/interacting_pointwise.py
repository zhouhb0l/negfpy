"""Opt-in finite-grid interacting Green functions at nonzero frequencies.

Interpolation preserves the supplied grid nodes and the algebraic Keldysh
identity. It is an approximation between nodes, not a continuous Hilbert
transform or proof of causality/accuracy. ``linear-keldysh`` uses common convex
weights for both independent Keldysh fields; it preserves their positivity if
the input matrices are positive, but does not preserve off-grid equilibrium
FDT. ``equilibrium-broadening`` instead interpolates the Hermitian shift and
broadening and constructs the Bose products at the query frequency. It is
restricted to a verified equilibrium input at a strictly positive temperature.
Neither policy clips eigenvalues or projects input matrices. Exact zero needs
separate acoustic-limit machinery and is deliberately excluded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.surface_gf import surface_gf

from .acoustic_contacts import _stable_bose_broadening_products
from .grid import SignedFrequencyGrid
from .harmonic_pointwise import (
    PointwiseHarmonic, _complex_pair, _readonly_copy, surface_equation_metrics,
)
from .types import KeldyshSpectrum


def _adjoint(array):
    return np.swapaxes(array.conj(), -1, -2)


def _check_close(actual, expected, *, name, rtol, atol):
    difference = float(np.max(np.abs(actual - expected)))
    scale = max(float(np.max(np.abs(actual))), float(np.max(np.abs(expected))))
    if difference > atol + rtol * scale:
        raise ValueError(f"Input self-energy violates {name}: absolute error {difference:.6g}.")
    return difference / max(scale, np.finfo(float).tiny)


@dataclass(frozen=True, slots=True)
class PointwiseSelfEnergy:
    retarded: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    diagnostics: dict


class GridSelfEnergyInterpolator:
    """Immutable, bounded linear interpolation of one grid self-energy.

    Validation checks every node with a per-node maximum-matrix scale. Input
    positivity is not screened here; common convex weights preserve it when
    present. No endpoint extrapolation, frequency wrapping or zero query is
    allowed. Exact floating-point grid nodes return copies of original fields.
    """

    def __init__(self, grid: SignedFrequencyGrid, sigma: KeldyshSpectrum, *,
                 policy="linear-keldysh", temperature=None, rtol=1e-8, atol=1e-12):
        if not isinstance(grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid.")
        if not isinstance(sigma, KeldyshSpectrum):
            raise TypeError("sigma must be a KeldyshSpectrum.")
        if sigma.n_frequency != grid.n_points:
            raise ValueError("Self-energy frequency count does not match the grid.")
        if policy not in {"linear-keldysh", "equilibrium-broadening"}:
            raise ValueError("Unknown self-energy interpolation policy.")
        if not np.isfinite(rtol) or not np.isfinite(atol) or rtol < 0 or atol < 0:
            raise ValueError("Validation tolerances must be finite and non-negative.")
        if policy == "equilibrium-broadening":
            if temperature is None or not np.isfinite(temperature) or temperature <= 0:
                raise ValueError("Equilibrium interpolation requires a strictly positive temperature.")
            temperature = float(temperature)
        self.grid = grid
        self.policy = policy
        self.temperature = temperature
        self.ndof = sigma.ndof
        self.retarded = _readonly_copy(sigma.retarded)
        self.lesser = _readonly_copy(sigma.lesser)
        self.greater = _readonly_copy(sigma.greater)
        if not all(np.all(np.isfinite(field)) for field in (self.retarded, self.lesser, self.greater)):
            raise ValueError("Self-energy must contain only finite values.")
        self.shift = _readonly_copy(0.5 * (self.retarded + _adjoint(self.retarded)))
        self.gamma = _readonly_copy(1j * (self.greater - self.lesser))
        metrics = {"lesser_antihermiticity": 0.0, "greater_antihermiticity": 0.0,
                   "keldysh_identity": 0.0}
        if policy == "equilibrium-broadening":
            metrics["equilibrium_fdt"] = 0.0
        for index, omega in enumerate(grid.values):
            lesser, greater, retarded = self.lesser[index], self.greater[index], self.retarded[index]
            for name, actual, expected in (
                ("lesser_antihermiticity", lesser, -lesser.conj().T),
                ("greater_antihermiticity", greater, -greater.conj().T),
                ("keldysh_identity", retarded - retarded.conj().T, greater - lesser),
            ):
                metrics[name] = max(metrics[name], _check_close(
                    actual, expected, name=name, rtol=rtol, atol=atol))
            if policy == "equilibrium-broadening":
                if omega == 0.0:
                    # At zero the finite Bose-product limit is independent
                    # data; only Gamma(0)=0 is required by this nonzero policy.
                    metrics["equilibrium_fdt"] = max(metrics["equilibrium_fdt"], _check_close(
                        greater, lesser, name="equilibrium zero-frequency identity", rtol=rtol, atol=atol))
                else:
                    n_gamma, np1_gamma = _stable_bose_broadening_products(
                        float(omega), temperature, self.gamma[index])
                    for actual, expected in ((lesser, -1j * n_gamma), (greater, -1j * np1_gamma)):
                        metrics["equilibrium_fdt"] = max(metrics["equilibrium_fdt"], _check_close(
                            actual, expected, name="equilibrium FDT", rtol=rtol, atol=atol))
        if policy == "equilibrium-broadening":
            # Gamma(0) is analytically zero at equilibrium. Roundoff left in
            # greater-lesser would create an artificial constant / omega term
            # arbitrarily close to zero. Its removal is explicit and bounded
            # by the zero-node validation above; original nodal fields remain
            # untouched and exact zero is still not an evaluable query.
            gamma = self.gamma.copy()
            metrics["zero_gamma_roundoff_removed_max"] = float(np.max(np.abs(gamma[grid.zero_index])))
            gamma[grid.zero_index] = 0.0
            self.gamma = _readonly_copy(gamma)
        self.validation_metrics = metrics

    def __call__(self, omega: float) -> PointwiseSelfEnergy:
        return self.evaluate(omega)

    def evaluate(self, omega: float) -> PointwiseSelfEnergy:
        omega = float(omega)
        if not np.isfinite(omega) or omega == 0.0:
            raise ValueError("Pointwise interacting frequencies must be finite and nonzero.")
        values = self.grid.values
        if omega < values[0] or omega > values[-1]:
            raise ValueError("Frequency lies outside the self-energy grid cutoff.")
        right = int(np.searchsorted(values, omega, side="left"))
        exact = right < len(values) and omega == values[right]
        if exact:
            left = right
            weight = 0.0
            left_weight = 1.0
            retarded, lesser, greater = (field[right].copy() for field in (
                self.retarded, self.lesser, self.greater))
        else:
            left = right - 1
            width = float(values[right] - values[left])
            # Compute both weights directly: 1-right_weight loses the small
            # left weight for negative frequencies very close to zero.
            left_weight = float((values[right] - omega) / width)
            weight = float((omega - values[left]) / width)
            shift = left_weight * self.shift[left] + weight * self.shift[right]
            if self.policy == "linear-keldysh":
                lesser = left_weight * self.lesser[left] + weight * self.lesser[right]
                greater = left_weight * self.greater[left] + weight * self.greater[right]
            else:
                gamma = left_weight * self.gamma[left] + weight * self.gamma[right]
                n_gamma, np1_gamma = _stable_bose_broadening_products(omega, self.temperature, gamma)
                lesser, greater = -1j * n_gamma, -1j * np1_gamma
            retarded = shift + 0.5 * (greater - lesser)
        return PointwiseSelfEnergy(retarded, lesser, greater, {
            "policy": self.policy, "exact_grid_node": bool(exact),
            "left_index": left, "right_index": right,
            "left_weight": left_weight, "right_weight": weight,
        })


@dataclass(frozen=True, slots=True)
class PointwiseInteractingResult:
    lesser: np.ndarray
    greater: np.ndarray
    retarded: np.ndarray
    diagnostics: dict
    self_energy: PointwiseSelfEnergy


class PointwiseInteracting(PointwiseHarmonic):
    """Dyson/Keldysh solve with freshly evaluated contacts and interpolated pp.

    Contacts retain their separate temperatures and orientations. Retarded,
    lesser and greater interaction terms all enter their corresponding equation;
    greater Green functions are never inferred from an equilibrium shortcut.
    Like the harmonic evaluator, all calls use local scratch and are thread safe.
    """

    def __init__(self, problem, config, sigma, *, policy="linear-keldysh",
                 selected_entries=(), rtol=1e-8, atol=1e-12):
        super().__init__(problem, config, selected_entries=selected_entries)
        if not isinstance(sigma, KeldyshSpectrum):
            raise TypeError("sigma must be a KeldyshSpectrum.")
        if sigma.ndof != problem.ndof:
            raise ValueError("Self-energy matrix dimension does not match the device.")
        temperature = None
        if policy == "equilibrium-broadening":
            temperature = float(problem.left.temperature)
            if temperature <= 0 or temperature != float(problem.right.temperature):
                raise ValueError("Equilibrium interpolation requires equal strictly positive reservoir temperatures.")
        self.interpolator = GridSelfEnergyInterpolator(
            problem.grid, sigma, policy=policy, temperature=temperature, rtol=rtol, atol=atol)

    def evaluate(self, omega: float, diagnostic: bool = False) -> PointwiseInteractingResult:
        omega = float(omega)
        pp = self.interpolator.evaluate(omega)
        sigma_r, sigma_lesser, sigma_greater = (
            field.copy() for field in (pp.retarded, pp.lesser, pp.greater))
        contact_metrics = {}
        for contact in self.contacts:
            surface = surface_gf(
                omega=omega, d00=contact.d00, d01=contact.forward, d10=contact.reverse,
                eta=self.eta_lead, method=self.surface_gf_method,
            )
            block = contact.coupling @ surface @ contact.coupling.conj().T
            gamma = 1j * (block - block.conj().T)
            n_gamma, np1_gamma = _stable_bose_broadening_products(omega, contact.temperature, gamma)
            indices = np.ix_(contact.indices, contact.indices)
            sigma_r[indices] += block
            sigma_lesser[indices] += -1j * n_gamma
            sigma_greater[indices] += -1j * np1_gamma
            if diagnostic:
                contact_metrics[contact.side] = surface_equation_metrics(
                    omega, contact.d00, contact.forward, contact.reverse, self.eta_lead, surface)
        operator = omega ** 2 * self.identity - self.dynamical - sigma_r
        retarded = np.linalg.solve(operator, self.identity)
        advanced = retarded.conj().T
        lesser = retarded @ sigma_lesser @ advanced
        greater = retarded @ sigma_greater @ advanced
        if not all(np.all(np.isfinite(field)) for field in (retarded, lesser, greater)):
            raise FloatingPointError("Non-finite pointwise interacting Green function.")
        record = {
            "omega": omega, "self_energy_interpolation": pp.diagnostics,
            "selected_entries": [
                {"row": int(row), "column": int(column),
                 "lesser": _complex_pair(lesser[row, column]),
                 "greater": _complex_pair(greater[row, column]),
                 "retarded": _complex_pair(retarded[row, column])}
                for row, column in self.selected_entries],
        }
        if diagnostic:
            singular = np.linalg.svd(operator, compute_uv=False)
            residual = float(np.linalg.norm(operator @ retarded - self.identity, ord="fro"))
            denominator = max(
                float(np.linalg.norm(operator, ord="fro")) * float(np.linalg.norm(retarded, ord="fro"))
                + float(np.linalg.norm(self.identity, ord="fro")), np.finfo(float).tiny)
            difference = greater - lesser
            spectral = retarded - advanced
            record.update({
                "dyson_backward_error": residual / denominator,
                "green_keldysh_identity_relative": float(np.max(np.abs(difference - spectral))) / max(
                    float(np.max(np.abs(difference))), float(np.max(np.abs(spectral))), np.finfo(float).tiny),
                "device_operator_singular_min": float(singular[-1]),
                "device_operator_singular_max": float(singular[0]),
                "surface_equations": contact_metrics,
                "eta_lead": self.eta_lead, "eta_device": 0.0,
            })
        return PointwiseInteractingResult(lesser, greater, retarded, record, pp)


__all__ = ["GridSelfEnergyInterpolator", "PointwiseSelfEnergy",
           "PointwiseInteracting", "PointwiseInteractingResult"]
