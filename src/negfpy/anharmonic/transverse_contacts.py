"""Finite-eta thermal contacts with joint transverse q/-q validation.

Every stable sector uses its continuous Bose-times-broadening limit at zero,
including gapped sectors. This is a separate convention from the historical
gapped zero assignment; it does not change existing material calculations.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from negfpy.core.types import LeadBlocks

from ._harmonic import normalize_contact_indices, retarded_contact_self_energy
from .acoustic_contacts import (
    _build_fixed_eta_contact_descriptor,
    _classify_lead_spectrum,
    _lead_arrays,
    _reservoir_keldysh_from_retarded_acoustic,
    _validate_q_samples,
)
from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .transverse_mapping import _integer, _kpoints, _real
from .transverse_solver import TransverseSpectrum, validate_transverse_spectrum
from .types import HarmonicReservoir, KeldyshSpectrum

_MAX_BYTES = 64 * 1024**2
_MAX_REQUEST_BYTES = 4 * 1024**3
_COMPONENTS = ("retarded", "lesser", "greater")
ReservoirBuilder = Callable[[tuple[float, float]], HarmonicReservoir]


@dataclass(frozen=True, slots=True, eq=False)
class TransverseContact:
    """One reservoir on a full transverse mesh, in dimensionless solver units."""

    mesh: UniformBZMesh
    grid: SignedFrequencyGrid
    spectrum: TransverseSpectrum
    zero_limits: np.ndarray
    q_reports: tuple[Mapping[str, Any], ...]
    checks: Mapping[str, float]
    side: str
    eta_lead: float
    temperature: float

    def __post_init__(self) -> None:
        limits = np.array(self.zero_limits, dtype=complex, copy=True)
        limits.setflags(write=False)
        object.__setattr__(self, "zero_limits", limits)
        object.__setattr__(self, "q_reports", tuple(
            MappingProxyType(dict(report)) for report in self.q_reports
        ))
        object.__setattr__(self, "checks", MappingProxyType(dict(self.checks)))

    def at_q(self, kpar: tuple[float, float]) -> KeldyshSpectrum:
        """Serve an already computed mesh point to ``build_transverse_problem``.

        Coordinates are radians. Equivalent negative/translated mesh points
        are accepted; arbitrary interpolation or contact recomputation is not.
        """
        values = np.asarray(kpar)
        if values.shape != (2,) or np.iscomplexobj(values) or not np.isfinite(values).all():
            raise ValueError("kpar must contain two finite real radian coordinates.")
        address = values * np.asarray(self.mesh.shape[:2]) / (2 * np.pi)
        nearest = np.rint(address)
        if np.max(np.abs(address)) >= 2**40 or np.max(np.abs(address - nearest)) > 1e-9:
            raise ValueError("Requested kpar is not a resolved point of this contact mesh.")
        index = self.mesh.index_from_address((int(nearest[0]), int(nearest[1]), 0))
        return KeldyshSpectrum(**{
            name: getattr(self.spectrum, name)[index] for name in _COMPONENTS
        })


def build_transverse_contact(
    mesh: UniformBZMesh,
    grid: SignedFrequencyGrid,
    reservoirs: ReservoirBuilder,
    ndof: int,
    eta_lead: float,
    *,
    side: str,
    surface_gf_method: str = "sancho_rubio",
    q_samples: int = 257,
    max_bytes: int = _MAX_BYTES,
    max_base_evaluations: int = 100_000,
) -> TransverseContact:
    """Build signed thermal spectra for one physical, transverse-periodic lead.

    The callback supplies solver-unit lead/coupling blocks at radian kpar.
    It must use one temperature, one embedding and D(-q)=D(q)*. Each sector
    is classified over longitudinal momentum before its surface is evaluated.
    Unstable or unresolved soft sectors are rejected, never pinned or clipped.

    At omega=0, lesser=greater=-i*T*L_eta, with L_eta=Gamma'(0), in every
    stable sector. L_eta may be complex Hermitian. Its reality/reflection
    relation is checked jointly at opposite transverse q, not within one q.

    The default payload budget stays 64 MiB. Callers refining the q/frequency
    mesh may explicitly admit up to 4 GiB; this changes allocation admission
    only and does not change the contact equations or their physical checks.
    """
    kpoints = _kpoints(mesh)
    if not isinstance(grid, SignedFrequencyGrid) or not callable(reservoirs):
        raise TypeError("Supply a SignedFrequencyGrid and reservoir callback.")
    dimension = _integer(ndof, "ndof", minimum=1)
    eta = _real(eta_lead, "eta_lead")
    if eta <= 0 or side not in ("left", "right"):
        raise ValueError("Positive eta_lead and side='left' or 'right' are required.")
    samples = _validate_q_samples(q_samples)
    limit = _integer(max_bytes, "max_bytes", minimum=1)
    if limit > _MAX_REQUEST_BYTES:
        raise ValueError("Explicit contact max_bytes exceeds the 4 GiB request ceiling.")
    work = _integer(max_base_evaluations, "max_base_evaluations", minimum=1)
    nq = mesh.n_qpoints
    if (3 * grid.n_points + 1) * nq * dimension**2 * 16 > limit:
        raise ValueError("Contact spectra and zero limits exceed max_bytes.")
    if nq * (grid.n_points + 2 * samples) > work:
        raise ValueError("Contact construction exceeds max_base_evaluations.")
    if not isinstance(surface_gf_method, str):
        raise TypeError("surface_gf_method must be a string.")

    sources: list[HarmonicReservoir] = []
    lead_payload = 0
    for point in kpoints:
        raw = reservoirs(tuple(float(v) for v in point))
        if not isinstance(raw, HarmonicReservoir):
            raise TypeError("Each callback result must be a HarmonicReservoir.")
        size = raw.lead.d00.shape[0]
        if size < 1 or 16 * size**4 > limit:
            raise ValueError("Surface derivative operator exceeds max_bytes.")
        d00, d01, d10, _, _ = _lead_arrays(raw.lead)
        coupling = raw.resolved_device_to_lead(side)
        indices = normalize_contact_indices(raw, dimension, side)
        lead_payload += 16 * (3 * size**2 + coupling.size)
        if lead_payload > limit:
            raise ValueError("Input lead blocks exceed max_bytes.")
        lead = LeadBlocks(*(np.array(a, dtype=complex, copy=True) for a in (d00, d01, d10)))
        sources.append(HarmonicReservoir(
            lead, raw.temperature, device_to_lead=coupling, device_indices=tuple(indices),
        ))
    temperature = sources[0].temperature
    for q, source in enumerate(sources):
        partner = sources[mesh.negative_index(q)]
        if source.temperature != temperature:
            raise ValueError("One physical reservoir must have one q-independent temperature.")
        if source.device_indices != sources[0].device_indices:
            raise ValueError("Reservoir device embedding must be identical across q.")
        for name, a, b in (
            ("d00", source.lead.d00, partner.lead.d00),
            ("d01", source.lead.d01, partner.lead.d01),
            ("d10", source.lead.d10, partner.lead.d10),
            ("coupling", source.device_to_lead, partner.device_to_lead),
        ):
            if a.shape != b.shape or not np.allclose(a.conj(), b, rtol=1e-10, atol=1e-12):
                raise ValueError(f"Reservoir {name} violates opposite-q conjugacy at q={q}.")
        if source.lead.d00.shape != sources[0].lead.d00.shape:
            raise ValueError("Lead principal-layer dimension changes across q.")

    fields, limits, reports = [], [], []
    for q, source in enumerate(sources):
        classification = _classify_lead_spectrum(source.lead, q_samples=samples)
        if not classification.passed or classification.classification not in {"acoustic", "gapped"}:
            raise ValueError(
                f"Lead sector q={q} is {classification.classification}: "
                f"{classification.failure_reasons}"
            )
        descriptor = _build_fixed_eta_contact_descriptor(
            source, side=side, ndof=dimension, eta_lead=eta,
            surface_gf_method=surface_gf_method, spectral_classification=classification,
        )
        # The descriptor already solved the exact-zero surface. Every other
        # signed node is evaluated directly, without a same-q real reflection.
        retarded = np.stack([
            descriptor.embedded_zero_retarded if index == grid.zero_index else
            retarded_contact_self_energy(
                omega, source, dimension, eta, side=side,
                surface_gf_method=surface_gf_method,
            )
            for index, omega in enumerate(grid.values)
        ])
        fields.append(_reservoir_keldysh_from_retarded_acoustic(
            grid, retarded, temperature, descriptor,
        ))
        limits.append(descriptor.embedded_limit)
        reports.append({
            "q_index": q,
            "classification": classification.classification,
            "lead_physics_sha256": descriptor.lead_physics_sha256,
            "descriptor_sha256": descriptor.descriptor_sha256,
            "surface_equation_residual": descriptor.surface_equation_residual,
            "derivative_residual": descriptor.derivative_residual,
            "derivative_condition": descriptor.derivative_condition,
            "zero_retarded_hermiticity": descriptor.analytic_zero_residual,
            "limit_minimum_eigenvalue": descriptor.passivity_minimum_eigenvalue,
            "zero_convention": "continuous-fixed-eta-bose-broadening",
        })
    spectrum = TransverseSpectrum(**{
        name: np.stack([getattr(field, name) for field in fields]) for name in _COMPONENTS
    })
    zero_limits = np.stack(limits)
    negative = [mesh.negative_index(q) for q in range(nq)]
    if not np.allclose(zero_limits[negative], zero_limits.conj(), rtol=1e-8, atol=1e-11):
        raise ValueError("Fixed-eta zero limits violate opposite-q conjugacy.")
    checks = validate_transverse_spectrum(mesh, grid, spectrum)
    return TransverseContact(
        mesh, grid, spectrum, zero_limits, tuple(reports), checks, side, eta, temperature,
    )


__all__ = ["TransverseContact", "build_transverse_contact"]
