"""Opt-in fixed-field Green evaluation including the existing acoustic zero.

Nonzero frequencies use PointwiseInteracting unchanged. Exact zero takes the
supplied self-energy node and the production contact prescription: the audited
finite-eta acoustic Bose-broadening limit, or zero Keldysh contact fields for a
gapped lead. No mean delta, pinning, pseudoinverse or device eta is introduced.
This evaluates a specified numerical field; it does not certify its frequency
accuracy, a stationary mean equation, or an infinite-system acoustic limit.
"""

from __future__ import annotations

import numpy as np

from ._harmonic import retarded_contact_self_energy
from .acoustic_contacts import _build_acoustic_contact_descriptor, _classify_lead_spectrum
from .harmonic_pointwise import _complex_pair, _readonly_copy
from .interacting_pointwise import (
    PointwiseInteracting, PointwiseInteractingResult, PointwiseSelfEnergy,
)


class FixedFieldGreenEvaluator(PointwiseInteracting):
    """Dyson/Keldysh fields at arbitrary signed frequencies, including zero.

The self-energy is copied and validated once by the parent interpolator.
Only ``linear-keldysh`` is supported so separate reservoir temperatures and
the archived biased field remain explicit. The zero node is never interpolated
or reconstructed from neighboring nodes. Calls use local scratch and return
independent arrays. Exact zero remains a finite-lead-eta numerical convention.
"""

    def __init__(self, problem, config, sigma, *, policy="linear-keldysh",
                 selected_entries=(), rtol=1e-8, atol=1e-12):
        if policy != "linear-keldysh":
            raise ValueError("Fixed-field evaluation requires linear-keldysh interpolation.")
        super().__init__(problem, config, sigma, policy=policy,
                         selected_entries=selected_entries, rtol=rtol, atol=atol)
        contact_r = np.zeros_like(self.identity)
        contact_k = np.zeros_like(self.identity)
        records = {}
        for side, reservoir in (("left", problem.left), ("right", problem.right)):
            classification = _classify_lead_spectrum(reservoir.lead)
            if classification.classification == "acoustic" and classification.passed:
                descriptor = _build_acoustic_contact_descriptor(
                    reservoir, side=side, ndof=problem.ndof, eta_lead=self.eta_lead,
                    surface_gf_method=self.surface_gf_method,
                    spectral_classification=classification)
                contact_r += descriptor.embedded_zero_retarded
                contact_k += -1j * reservoir.temperature * descriptor.embedded_limit
                records[side] = {
                    "classification": "acoustic", "policy": "audited-finite-eta-limit",
                    "temperature": float(reservoir.temperature),
                    "descriptor_sha256": descriptor.descriptor_sha256,
                    "surface_equation_residual": descriptor.surface_equation_residual,
                    "derivative_residual": descriptor.derivative_residual,
                }
            elif classification.classification == "gapped" and classification.passed:
                retarded = retarded_contact_self_energy(
                    0.0, reservoir, problem.ndof, self.eta_lead, side=side,
                    surface_gf_method=self.surface_gf_method)
                gamma = 1j * (retarded - retarded.conj().T)
                if np.max(np.abs(gamma)) > 1e-10 * max(1.0, float(np.max(np.abs(retarded)))):
                    raise ValueError("Gapped exact-zero contact has nonzero broadening.")
                contact_r += retarded
                records[side] = {
                    "classification": "gapped", "policy": "production-gapped-zero",
                    "temperature": float(reservoir.temperature),
                }
            else:
                raise ValueError(f"{side.capitalize()} lead has invalid classification "
                                 f"{classification.classification}.")
        self._zero_contact_retarded = _readonly_copy(contact_r)
        self._zero_contact_keldysh = _readonly_copy(contact_k)
        self._zero_contact_records = records

    def evaluate(self, omega: float, diagnostic: bool = False) -> PointwiseInteractingResult:
        omega = float(omega)
        if omega != 0.0:
            return super().evaluate(omega, diagnostic=diagnostic)
        index = self.interpolator.grid.zero_index
        pp = PointwiseSelfEnergy(
            *(field[index].copy() for field in (self.interpolator.retarded,
                                               self.interpolator.lesser,
                                               self.interpolator.greater)),
            diagnostics={"policy": self.interpolator.policy, "exact_grid_node": True,
                         "left_index": index, "right_index": index,
                         "left_weight": 1.0, "right_weight": 0.0,
                         "zero_node_preserved": True})
        operator = -self.dynamical - self._zero_contact_retarded - pp.retarded
        retarded = np.linalg.solve(operator, self.identity)
        advanced = retarded.conj().T
        lesser = retarded @ (self._zero_contact_keldysh + pp.lesser) @ advanced
        greater = retarded @ (self._zero_contact_keldysh + pp.greater) @ advanced
        if not all(np.all(np.isfinite(field)) for field in (retarded, lesser, greater)):
            raise FloatingPointError("Non-finite exact-zero fixed-field Green function.")
        record = {
            "omega": 0.0, "self_energy_interpolation": pp.diagnostics,
            "zero_frequency_policy": "production-contact-limit-with-supplied-pp-node",
            "zero_contact_limits": {side: dict(values) for side, values in self._zero_contact_records.items()},
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
                "eta_lead": self.eta_lead, "eta_device": 0.0,
            })
        return PointwiseInteractingResult(lesser, greater, retarded, record, pp)


__all__ = ["FixedFieldGreenEvaluator"]
