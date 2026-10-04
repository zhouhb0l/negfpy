"""Research harmonic Green functions at nonzero, arbitrary signed frequencies.

This evaluator keeps the production harmonic equations, contact orientations,
reservoir temperatures and lead broadening. It supplies independent lesser and
greater fields for targeted bubble quadrature. It does not interpolate a grid,
change the Hamiltonian, or provide an interacting SCBA Green function. Exact
zero needs the production acoustic-limit machinery and is explicitly excluded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.surface_gf import surface_gf

from ._harmonic import assemble_device_dynamical_matrix
from .acoustic_contacts import _stable_bose_broadening_products
from .types import CubicSCBAProblem, SCBAConfig


def _readonly_copy(value, dtype=np.complex128):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _complex_pair(value):
    return [float(np.real(value)), float(np.imag(value))]


def surface_equation_metrics(omega, d00, forward, reverse, eta, surface):
    """Check the surface Dyson equation with couplings oriented into the lead."""
    eye = np.eye(surface.shape[0], dtype=np.complex128)
    operator = (float(omega) + 1j * float(eta)) ** 2 * eye - d00 - forward @ surface @ reverse
    product = operator @ surface
    residual = float(np.linalg.norm(product - eye, ord="fro"))
    tiny = np.finfo(float).tiny
    return {
        "equation_residual": residual / max(
            float(np.linalg.norm(product, ord="fro")) + float(np.linalg.norm(eye, ord="fro")), tiny
        ),
        "backward_error": residual / max(
            float(np.linalg.norm(operator, ord="fro")) * float(np.linalg.norm(surface, ord="fro"))
            + float(np.linalg.norm(eye, ord="fro")), tiny
        ),
    }


@dataclass(frozen=True, slots=True)
class PointwiseHarmonicResult:
    """One frequency's independent harmonic fields and JSON-ready diagnostics."""

    lesser: np.ndarray
    greater: np.ndarray
    retarded: np.ndarray
    diagnostics: dict


@dataclass(frozen=True, slots=True)
class _Contact:
    side: str
    d00: np.ndarray
    forward: np.ndarray
    reverse: np.ndarray
    coupling: np.ndarray
    indices: np.ndarray
    temperature: float


class PointwiseHarmonic:
    """Evaluate full harmonic fields without retaining a frequency cache.

    ``evaluate(omega, diagnostic=True)`` also computes singular values and
    equation residuals. ``evaluator(omega)`` returns ``(lesser, greater)`` for
    bubble-quadrature callbacks. Inputs are copied once into read-only arrays;
    calls use only local scratch arrays and may execute concurrently. All
    frequencies, temperatures, matrices and eta values use solver units.
    """

    def __init__(self, problem: CubicSCBAProblem, config: SCBAConfig, selected_entries=()):
        if float(config.eta_device) != 0.0:
            raise ValueError("Pointwise harmonic evaluation requires eta_device=0.")
        eta_lead = float(config.eta_lead)
        if not np.isfinite(eta_lead) or eta_lead <= 0.0:
            raise ValueError("eta_lead must be finite and strictly positive.")
        method = config.surface_gf_method.strip().lower()
        if method in {"analytic_diagonal", "analytic-diagonal"}:
            raise ValueError("Pointwise evaluation requires an explicit surface_gf solver.")
        self.eta_lead = eta_lead
        self.surface_gf_method = config.surface_gf_method
        self.dynamical = _readonly_copy(assemble_device_dynamical_matrix(problem.device))
        self.identity = _readonly_copy(np.eye(problem.ndof))
        entries = tuple(tuple(entry) for entry in selected_entries)
        for entry in entries:
            if len(entry) != 2 or any(
                isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer))
                or not 0 <= index < problem.ndof
                for index in entry
            ):
                raise ValueError("Each selected entry must be two valid integer indices.")
        self.selected_entries = entries
        contacts = []
        for side, reservoir in (("left", problem.left), ("right", problem.right)):
            lead = reservoir.lead
            reverse = lead.d10 if lead.d10 is not None else lead.d01.conj().T
            forward, backward = ((reverse, lead.d01) if side == "left" else (lead.d01, reverse))
            contacts.append(_Contact(
                side=side,
                d00=_readonly_copy(lead.d00),
                forward=_readonly_copy(forward),
                reverse=_readonly_copy(backward),
                coupling=_readonly_copy(reservoir.resolved_device_to_lead(side)),
                indices=_readonly_copy(
                    reservoir.resolved_device_indices(dim=problem.ndof, side=side), np.int64
                ),
                temperature=float(reservoir.temperature),
            ))
        self.contacts = tuple(contacts)

    def __call__(self, omega: float) -> tuple[np.ndarray, np.ndarray]:
        result = self.evaluate(omega)
        return result.lesser, result.greater

    def evaluate(self, omega: float, diagnostic: bool = False) -> PointwiseHarmonicResult:
        omega = float(omega)
        if not np.isfinite(omega) or omega == 0.0:
            raise ValueError("Pointwise harmonic frequencies must be finite and nonzero.")
        sigma_r = np.zeros_like(self.identity)
        sigma_lesser = np.zeros_like(self.identity)
        sigma_greater = np.zeros_like(self.identity)
        contact_metrics = {}
        for contact in self.contacts:
            surface = surface_gf(
                omega=omega, d00=contact.d00, d01=contact.forward, d10=contact.reverse,
                eta=self.eta_lead, method=self.surface_gf_method,
            )
            block = contact.coupling @ surface @ contact.coupling.conj().T
            gamma = 1j * (block - block.conj().T)
            # Compute both stable reservoir products independently, including
            # signed-frequency and zero-temperature continuations. In particular
            # greater is not inferred from an equilibrium FDT relation.
            n_gamma, np1_gamma = _stable_bose_broadening_products(omega, contact.temperature, gamma)
            indices = np.ix_(contact.indices, contact.indices)
            sigma_r[indices] += block
            sigma_lesser[indices] += -1j * n_gamma
            sigma_greater[indices] += -1j * np1_gamma
            if diagnostic:
                contact_metrics[contact.side] = surface_equation_metrics(
                    omega, contact.d00, contact.forward, contact.reverse, self.eta_lead, surface
                )
        operator = omega ** 2 * self.identity - self.dynamical - sigma_r
        retarded = np.linalg.solve(operator, self.identity)
        advanced = retarded.conj().T
        lesser = retarded @ sigma_lesser @ advanced
        greater = retarded @ sigma_greater @ advanced
        if not all(np.all(np.isfinite(field)) for field in (retarded, lesser, greater)):
            raise FloatingPointError("Non-finite pointwise harmonic Green function.")
        record = {
            "omega": omega,
            "selected_entries": [
                {"row": int(row), "column": int(column),
                 "lesser": _complex_pair(lesser[row, column]),
                 "greater": _complex_pair(greater[row, column]),
                 "retarded": _complex_pair(retarded[row, column])}
                for row, column in self.selected_entries
            ],
        }
        if diagnostic:
            singular = np.linalg.svd(operator, compute_uv=False)
            residual = float(np.linalg.norm(operator @ retarded - self.identity, ord="fro"))
            denominator = max(
                float(np.linalg.norm(operator, ord="fro")) * float(np.linalg.norm(retarded, ord="fro"))
                + float(np.linalg.norm(self.identity, ord="fro")), np.finfo(float).tiny,
            )
            record.update({
                "dyson_backward_error": residual / denominator,
                "device_operator_singular_min": float(singular[-1]),
                "device_operator_singular_max": float(singular[0]),
                "surface_equations": contact_metrics,
                "eta_lead": self.eta_lead,
                "eta_device": 0.0,
            })
        return PointwiseHarmonicResult(lesser, greater, retarded, record)
