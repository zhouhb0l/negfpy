"""Fail-closed acoustic harmonic-contact construction.

The dense anharmonic solver historically accepted only spectrally gapped
reservoirs.  This module supplies the one additional ingredient needed by an
acoustic material lead: the finite-contact-regulator limit of
``Gamma(omega) / omega``.  It deliberately does not alter ``LeadBlocks`` or
``HarmonicReservoir`` and does not regularize the finite device.

The public lead audit consumes a :class:`LayeredHarmonicModel`, because only
that object knows the mass-weighted rigid-translation subspace.  The private
spectral audit used during generic contact construction can recognize scalar
and synthetic acoustic leads, but it is not a replacement for that material
audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.optimize import minimize_scalar

from negfpy.core.surface_gf import surface_gf
from negfpy.core.types import LeadBlocks

from ._canonical import digest_bytes, domain_digest
from ._harmonic import (
    _device_to_lead_coupling,
    embed_contact_block,
    normalize_contact_indices,
)
from .grid import SignedFrequencyGrid
from .types import HarmonicReservoir, KeldyshSpectrum


if TYPE_CHECKING:
    from .layered import LayeredHarmonicModel


Array = np.ndarray
LeadKind = Literal["gapped", "acoustic", "unstable", "invalid-soft"]
ContactSide = Literal["left", "right"]

_EPS = np.finfo(np.float64).eps
_TINY = np.finfo(np.float64).tiny
_SERIES_THRESHOLD = 1.0e-5


def _readonly(values: Array, *, dtype: np.dtype) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("Acoustic-contact report arrays must be finite.")
    result.setflags(write=False)
    return result


def _lead_arrays(lead: LeadBlocks) -> tuple[Array, Array, Array, float, float]:
    if not isinstance(lead, LeadBlocks):
        raise TypeError("lead must be a LeadBlocks instance.")
    d00 = np.asarray(lead.d00, dtype=np.complex128)
    d01 = np.asarray(lead.d01, dtype=np.complex128)
    d10 = np.asarray(
        lead.d10 if lead.d10 is not None else d01.conj().T,
        dtype=np.complex128,
    )
    if d00.ndim != 2 or d00.shape[0] != d00.shape[1]:
        raise ValueError("lead d00 must be square.")
    if d01.shape != d00.shape or d10.shape != d00.shape:
        raise ValueError("lead blocks must have identical square shapes.")
    if not all(np.all(np.isfinite(value)) for value in (d00, d01, d10)):
        raise ValueError("lead blocks must contain only finite values.")
    gap_scale = max(
        float(np.linalg.norm(d00, ord=2)),
        float(np.linalg.norm(d01, ord=2)),
        float(np.linalg.norm(d10, ord=2)),
        _TINY,
    )
    reciprocal_tolerance = 100.0 * _EPS * gap_scale
    if float(np.max(np.abs(d00 - d00.conj().T))) > reciprocal_tolerance:
        raise ValueError("lead d00 must be Hermitian for spectral classification.")
    if float(np.max(np.abs(d10 - d01.conj().T))) > reciprocal_tolerance:
        raise ValueError("lead classification requires d10 = d01^dagger.")
    full_scale = max(
        float(np.linalg.norm(d00, ord=2))
        + float(np.linalg.norm(d01, ord=2))
        + float(np.linalg.norm(d10, ord=2)),
        _TINY,
    )
    return d00, d01, d10, gap_scale, full_scale


def harmonic_lead_physics_sha256(lead: LeadBlocks) -> str:
    """Return the canonical identity of one increasing-layer lead."""

    d00, d01, d10, _, _ = _lead_arrays(lead)
    return domain_digest(
        "negfpy-harmonic-lead-physics-v1",
        {
            "d00": np.array(d00, dtype=np.complex128, order="C"),
            "d01": np.array(d01, dtype=np.complex128, order="C"),
            "d10": np.array(d10, dtype=np.complex128, order="C"),
            "orientation": "increasing-layer",
        },
    )


def _bulk_dynamical(d00: Array, d01: Array, d10: Array, qpoint: float) -> Array:
    phase = np.exp(1j * float(qpoint))
    result = d00 + d01 * phase + d10 * phase.conjugate()
    scale = max(float(np.linalg.norm(result, ord=2)), 1.0)
    if float(np.max(np.abs(result - result.conj().T))) > 100.0 * _EPS * scale:
        raise ValueError("lead dynamical matrix is not Hermitian.")
    return result


def _wrapped_abs_q(qpoint: float) -> float:
    return abs((float(qpoint) + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True, slots=True, eq=False)
class _LeadSpectralClassification:
    classification: LeadKind
    passed: bool
    lead_physics_sha256: str
    q_samples: int
    gap_scale: float
    full_scale: float
    gap_threshold: float
    eigenvalue_tolerance: float
    stability_tolerance: float
    sampled_minimum_frequency_squared: float
    refined_minimum_frequency_squared: float
    gamma_eigenvalues: Array
    gamma_nullity: int
    gamma_nullspace: Array
    normalized_asr_residual: float
    curvature_eigenvalues_h: Array
    curvature_eigenvalues_half_h: Array
    curvature_consistent: bool
    soft_minima_q: Array
    failure_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gamma_eigenvalues",
            _readonly(self.gamma_eigenvalues, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "gamma_nullspace",
            _readonly(self.gamma_nullspace, dtype=np.dtype(np.complex128)),
        )
        object.__setattr__(
            self,
            "curvature_eigenvalues_h",
            _readonly(self.curvature_eigenvalues_h, dtype=np.dtype(np.float64)),
        )
        object.__setattr__(
            self,
            "curvature_eigenvalues_half_h",
            _readonly(
                self.curvature_eigenvalues_half_h,
                dtype=np.dtype(np.float64),
            ),
        )
        object.__setattr__(
            self,
            "soft_minima_q",
            _readonly(self.soft_minima_q, dtype=np.dtype(np.float64)),
        )


def _validate_q_samples(q_samples: int) -> int:
    if isinstance(q_samples, (bool, np.bool_)) or not isinstance(
        q_samples, (int, np.integer)
    ):
        raise TypeError("q_samples must be an integer.")
    count = int(q_samples)
    if count < 9 or count % 2 == 0:
        raise ValueError("q_samples must be an odd integer of at least nine.")
    return count


def _classify_lead_spectrum(
    lead: LeadBlocks,
    *,
    q_samples: int = 257,
    _asr_residual: float | None = None,
) -> _LeadSpectralClassification:
    """Classify a reciprocal lead without using atomic mass information."""

    count = _validate_q_samples(q_samples)
    d00, d01, d10, gap_scale, full_scale = _lead_arrays(lead)
    lead_hash = harmonic_lead_physics_sha256(lead)
    gap_threshold = 100.0 * _EPS * gap_scale

    # This is the original shifted v0.1 grid, with a separate Gamma value.
    shifted_step = 2.0 * np.pi / count
    shifted_q = -np.pi + shifted_step * np.arange(count, dtype=np.float64)

    def branch_value(qpoint: float, branch: int) -> float:
        return float(np.linalg.eigvalsh(_bulk_dynamical(d00, d01, d10, qpoint))[branch])

    shifted_low = np.asarray([branch_value(q, 0) for q in shifted_q])
    shifted_minimum = min(float(np.min(shifted_low)), branch_value(0.0, 0))
    shifted_refined: list[tuple[float, float]] = []
    shifted_candidates = [
        index
        for index in range(count)
        if shifted_low[index] <= shifted_low[(index - 1) % count]
        and shifted_low[index] <= shifted_low[(index + 1) % count]
    ]
    for index in shifted_candidates or [int(np.argmin(shifted_low))]:
        center = float(shifted_q[index])
        result = minimize_scalar(
            lambda q: branch_value(q, 0),
            bounds=(center - shifted_step, center + shifted_step),
            method="bounded",
            options={"xatol": 32.0 * _EPS},
        )
        if not result.success or not np.isfinite(result.fun):
            raise ValueError("could not refine the harmonic lead band minimum.")
        shifted_minimum = min(shifted_minimum, float(result.fun))
        shifted_refined.append((float(result.x), float(result.fun)))

    empty = np.empty((0,), dtype=np.float64)
    empty_modes = np.empty((d00.shape[0], 0), dtype=np.complex128)
    if shifted_minimum > gap_threshold:
        gamma_values = np.linalg.eigvalsh(_bulk_dynamical(d00, d01, d10, 0.0))
        return _LeadSpectralClassification(
            classification="gapped",
            passed=True,
            lead_physics_sha256=lead_hash,
            q_samples=count,
            gap_scale=gap_scale,
            full_scale=full_scale,
            gap_threshold=gap_threshold,
            eigenvalue_tolerance=0.0,
            stability_tolerance=0.0,
            sampled_minimum_frequency_squared=float(np.min(shifted_low)),
            refined_minimum_frequency_squared=shifted_minimum,
            gamma_eigenvalues=gamma_values,
            gamma_nullity=0,
            gamma_nullspace=empty_modes,
            normalized_asr_residual=0.0,
            curvature_eigenvalues_h=empty,
            curvature_eigenvalues_half_h=empty,
            curvature_consistent=True,
            soft_minima_q=empty,
            failure_reasons=(),
        )

    gamma = _bulk_dynamical(d00, d01, d10, 0.0)
    gamma_values, gamma_vectors = np.linalg.eigh(gamma)
    # The private audit has no external rigid subspace.  Its ASR evidence is
    # therefore the residual on the numerically identified Gamma nullspace.
    preliminary_tol = max(1.0e-12 * full_scale, 0.0)
    preliminary = np.abs(gamma_values) <= preliminary_tol
    preliminary_nullspace = gamma_vectors[:, preliminary]
    if _asr_residual is None:
        asr_residual = (
            float(np.linalg.norm(gamma @ preliminary_nullspace, ord=2))
            if preliminary_nullspace.shape[1]
            # There is no candidate translation subspace when Gamma has no
            # numerical null vector.  Treating the smallest (possibly deeply
            # negative) eigenvalue as an ASR residual would inflate the
            # stability floor and could misclassify an unstable lead as soft.
            else 0.0
        )
    else:
        asr_residual = float(_asr_residual)
        if not np.isfinite(asr_residual) or asr_residual < 0.0:
            raise ValueError("_asr_residual must be finite and non-negative.")
    eigenvalue_tolerance = max(1.0e-12 * full_scale, 10.0 * asr_residual)
    stability_tolerance = eigenvalue_tolerance
    null = np.abs(gamma_values) <= eigenvalue_tolerance
    nullspace = gamma_vectors[:, null]
    nullity = int(np.count_nonzero(null))

    centered_indices = np.arange(-(count // 2), count // 2 + 1, dtype=np.int64)
    centered_q = 2.0 * np.pi * centered_indices.astype(np.float64) / count
    spectra = np.asarray(
        [np.linalg.eigvalsh(_bulk_dynamical(d00, d01, d10, q)) for q in centered_q],
        dtype=np.float64,
    )
    sampled_minimum = float(np.min(spectra))
    refined_minimum = min(sampled_minimum, float(np.min(gamma_values)))
    soft_locations: list[float] = [
        qpoint
        for qpoint, value in shifted_refined
        if value <= eigenvalue_tolerance
    ]
    for branch in range(d00.shape[0]):
        values = spectra[:, branch]
        for index in range(count):
            if values[index] > 100.0 * eigenvalue_tolerance:
                continue
            if not (
                values[index] <= values[(index - 1) % count]
                and values[index] <= values[(index + 1) % count]
            ):
                continue
            center = float(centered_q[index])
            result = minimize_scalar(
                lambda q, b=branch: branch_value(q, b),
                bounds=(center - shifted_step, center + shifted_step),
                method="bounded",
                options={"xatol": 32.0 * _EPS},
            )
            if not result.success or not np.isfinite(result.fun):
                raise ValueError("could not refine a soft harmonic-lead basin.")
            refined_minimum = min(refined_minimum, float(result.fun))
            if float(result.fun) <= eigenvalue_tolerance:
                soft_locations.append(float(result.x))

    h = 2.0 * np.pi / 1024.0
    curvature_values: list[Array] = []
    if nullity:
        for spacing in (h, h / 2.0):
            curvature = nullspace.conj().T @ (
                _bulk_dynamical(d00, d01, d10, spacing)
                + _bulk_dynamical(d00, d01, d10, -spacing)
                - 2.0 * gamma
            ) @ nullspace / (2.0 * spacing**2)
            curvature_values.append(np.linalg.eigvalsh(curvature))
    else:
        curvature_values = [empty, empty]
    curvature_consistent = bool(
        nullity
        and np.all(curvature_values[0] > 1.0e-10 * full_scale)
        and np.all(curvature_values[1] > 1.0e-10 * full_scale)
        and np.allclose(
            curvature_values[0],
            curvature_values[1],
            rtol=5.0e-3,
            atol=1.0e-12 * full_scale,
        )
    )

    failures: list[str] = []
    if refined_minimum < -stability_tolerance:
        classification: LeadKind = "unstable"
        failures.append("a refined lead eigenvalue is below the stability floor")
    else:
        if nullity < 1:
            failures.append("Gamma has no acoustic nullspace")
        if asr_residual / full_scale > 1.0e-10:
            failures.append("Gamma-nullspace residual exceeds the ASR tolerance")
        if any(_wrapped_abs_q(q) > 1.0e-7 for q in soft_locations):
            failures.append("a soft local minimum is not at Gamma")
        if not curvature_consistent:
            failures.append("acoustic curvature is nonpositive or unresolved")
        classification = "acoustic" if not failures else "invalid-soft"

    return _LeadSpectralClassification(
        classification=classification,
        passed=classification == "acoustic",
        lead_physics_sha256=lead_hash,
        q_samples=count,
        gap_scale=gap_scale,
        full_scale=full_scale,
        gap_threshold=gap_threshold,
        eigenvalue_tolerance=eigenvalue_tolerance,
        stability_tolerance=stability_tolerance,
        sampled_minimum_frequency_squared=sampled_minimum,
        refined_minimum_frequency_squared=refined_minimum,
        gamma_eigenvalues=gamma_values,
        gamma_nullity=nullity,
        gamma_nullspace=nullspace,
        normalized_asr_residual=asr_residual / full_scale,
        curvature_eigenvalues_h=curvature_values[0],
        curvature_eigenvalues_half_h=curvature_values[1],
        curvature_consistent=curvature_consistent,
        soft_minima_q=np.asarray(soft_locations, dtype=np.float64),
        failure_reasons=tuple(failures),
    )


@dataclass(frozen=True, slots=True, eq=False)
class HarmonicLeadClassification:
    """Immutable material-lead acoustic/gap/stability audit."""

    classification: LeadKind
    passed: bool
    lead_physics_sha256: str
    fc2_content_sha256: str
    layered_geometry_sha256: str
    scale_sha256: str
    q_samples: int
    gap_threshold: float
    eigenvalue_tolerance: float
    stability_tolerance: float
    sampled_minimum_frequency_squared: float
    refined_minimum_frequency_squared: float
    gamma_eigenvalues: Array
    gamma_nullity: int
    rigid_subspace_projector_error: float
    normalized_asr_residual: float
    curvature_eigenvalues_h: Array
    curvature_eigenvalues_half_h: Array
    soft_minima_q: Array
    failure_reasons: tuple[str, ...]
    classification_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "gamma_eigenvalues",
            "curvature_eigenvalues_h",
            "curvature_eigenvalues_half_h",
            "soft_minima_q",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), dtype=np.dtype(np.float64)),
            )


def classify_harmonic_lead(
    model: LayeredHarmonicModel,
    *,
    q_samples: int = 257,
) -> HarmonicLeadClassification:
    """Audit a layered material lead including its rigid translations."""

    # Local import avoids making generic acoustic contacts depend on material
    # ingestion during ordinary package import.
    from .layered import LayeredHarmonicModel

    if not isinstance(model, LayeredHarmonicModel):
        raise TypeError("model must be a LayeredHarmonicModel instance.")
    rigid = np.asarray(model.rigid_translation_vectors, dtype=np.complex128)
    if (
        rigid.ndim != 2
        or rigid.shape[0] != model.d00.shape[0]
        or rigid.shape[1] != 3
    ):
        raise ValueError("model must contain exactly three rigid-translation vectors.")
    gram_error = float(np.linalg.norm(rigid.conj().T @ rigid - np.eye(3), ord=2))
    gamma = _bulk_dynamical(
        np.asarray(model.d00, dtype=np.complex128),
        np.asarray(model.d01, dtype=np.complex128),
        np.asarray(model.d10, dtype=np.complex128),
        0.0,
    )
    rigid_residual = float(np.linalg.norm(gamma @ rigid, ord=2))
    spectral = _classify_lead_spectrum(
        model.lead,
        q_samples=q_samples,
        _asr_residual=rigid_residual,
    )
    normalized_asr = rigid_residual / spectral.full_scale
    numerical_projector = (
        spectral.gamma_nullspace @ spectral.gamma_nullspace.conj().T
        if spectral.gamma_nullity
        else np.zeros_like(gamma)
    )
    rigid_projector = rigid @ rigid.conj().T
    projector_error = float(np.linalg.norm(numerical_projector - rigid_projector, ord=2))

    h = 2.0 * np.pi / 1024.0
    curvatures: list[Array] = []
    d00 = np.asarray(model.d00, dtype=np.complex128)
    d01 = np.asarray(model.d01, dtype=np.complex128)
    d10 = np.asarray(model.d10, dtype=np.complex128)
    for spacing in (h, h / 2.0):
        matrix = rigid.conj().T @ (
            _bulk_dynamical(d00, d01, d10, spacing)
            + _bulk_dynamical(d00, d01, d10, -spacing)
            - 2.0 * gamma
        ) @ rigid / (2.0 * spacing**2)
        curvatures.append(np.linalg.eigvalsh(matrix))
    curvature_ok = bool(
        np.all(curvatures[0] > 1.0e-10 * spectral.full_scale)
        and np.all(curvatures[1] > 1.0e-10 * spectral.full_scale)
        and np.allclose(
            curvatures[0],
            curvatures[1],
            rtol=5.0e-3,
            atol=1.0e-12 * spectral.full_scale,
        )
    )

    failures = list(spectral.failure_reasons)
    classification = spectral.classification
    if classification != "gapped" and classification != "unstable":
        if spectral.gamma_nullity != 3:
            failures.append("Gamma nullity is not exactly three")
        if gram_error > 1.0e-10:
            failures.append("stored rigid translations are not orthonormal")
        if projector_error > 1.0e-8:
            failures.append("Gamma nullspace does not match rigid translations")
        if normalized_asr > 1.0e-10:
            failures.append("rigid-translation ASR residual exceeds tolerance")
        if not curvature_ok:
            failures.append("rigid-subspace curvature gate failed")
        classification = "acoustic" if not failures else "invalid-soft"
    passed = classification in {"gapped", "acoustic"}

    scale_sha = domain_digest(
        "negfpy-solver-scale-v1",
        {"omega0": float(model.scale.omega0), "hbar": float(model.scale.hbar)},
    )
    payload = {
        "classification": classification,
        "lead_physics_sha256": digest_bytes(spectral.lead_physics_sha256),
        "fc2_content_sha256": digest_bytes(model.fc2_content_sha256),
        "layered_geometry_sha256": digest_bytes(model.layered_geometry_sha256),
        "scale_sha256": digest_bytes(scale_sha),
        "q_samples": int(spectral.q_samples),
        "gap_threshold": float(spectral.gap_threshold),
        "eigenvalue_tolerance": float(spectral.eigenvalue_tolerance),
        "stability_tolerance": float(spectral.stability_tolerance),
        "sampled_minimum_frequency_squared": float(
            spectral.sampled_minimum_frequency_squared
        ),
        "refined_minimum_frequency_squared": float(
            spectral.refined_minimum_frequency_squared
        ),
        "gamma_eigenvalues": np.asarray(spectral.gamma_eigenvalues, dtype=np.float64),
        "gamma_nullity": int(spectral.gamma_nullity),
        "rigid_subspace_projector_error": float(projector_error),
        "normalized_asr_residual": float(normalized_asr),
        "curvature_eigenvalues_h": np.asarray(curvatures[0], dtype=np.float64),
        "curvature_eigenvalues_half_h": np.asarray(curvatures[1], dtype=np.float64),
        "soft_minima_q": np.asarray(spectral.soft_minima_q, dtype=np.float64),
        "failure_reasons": tuple(failures),
    }
    report_hash = domain_digest("negfpy-harmonic-lead-classification-v1", payload)
    return HarmonicLeadClassification(
        classification=classification,
        passed=passed,
        lead_physics_sha256=spectral.lead_physics_sha256,
        fc2_content_sha256=model.fc2_content_sha256,
        layered_geometry_sha256=model.layered_geometry_sha256,
        scale_sha256=scale_sha,
        q_samples=spectral.q_samples,
        gap_threshold=spectral.gap_threshold,
        eigenvalue_tolerance=spectral.eigenvalue_tolerance,
        stability_tolerance=spectral.stability_tolerance,
        sampled_minimum_frequency_squared=spectral.sampled_minimum_frequency_squared,
        refined_minimum_frequency_squared=spectral.refined_minimum_frequency_squared,
        gamma_eigenvalues=spectral.gamma_eigenvalues,
        gamma_nullity=spectral.gamma_nullity,
        rigid_subspace_projector_error=projector_error,
        normalized_asr_residual=normalized_asr,
        curvature_eigenvalues_h=curvatures[0],
        curvature_eigenvalues_half_h=curvatures[1],
        soft_minima_q=spectral.soft_minima_q,
        failure_reasons=tuple(failures),
        classification_sha256=report_hash,
    )


def _surface_orientation(lead: LeadBlocks, side: ContactSide) -> tuple[Array, Array]:
    d01 = np.asarray(lead.d01, dtype=np.complex128)
    d10 = np.asarray(
        lead.d10 if lead.d10 is not None else d01.conj().T,
        dtype=np.complex128,
    )
    if side == "left":
        return d10, d01
    if side == "right":
        return d01, d10
    raise ValueError("side must be 'left' or 'right'.")


def _analytic_diagonal_surface_zero(lead: LeadBlocks, eta_lead: float) -> Array:
    d00, d01, d10, gap_scale, _ = _lead_arrays(lead)
    tolerance = 128.0 * _EPS * max(1.0, gap_scale)
    for values in (d00, d01, d10):
        if float(np.max(np.abs(values - np.diag(np.diag(values))))) > tolerance:
            raise ValueError("analytic_diagonal requires diagonal lead blocks.")
    onsite = np.diag(d00)
    product = np.diag(d01) * np.diag(d10)
    if float(np.max(np.abs(onsite.imag))) > tolerance:
        raise ValueError("analytic_diagonal requires real onsite entries.")
    if float(np.max(np.abs(product.imag))) > tolerance or np.any(product.real <= 0.0):
        raise ValueError("analytic_diagonal requires positive reciprocal hopping products.")
    z = -float(eta_lead) ** 2
    delta = z - onsite.real
    coefficient = product.real
    discriminant = np.sqrt(delta**2 - 4.0 * coefficient + 0.0j)
    roots = np.stack(
        ((delta + discriminant) / (2.0 * coefficient),
         (delta - discriminant) / (2.0 * coefficient)),
        axis=0,
    )
    # Below a stable spectrum the physical root has the smaller magnitude.
    selected = np.where(np.abs(roots[0]) <= np.abs(roots[1]), roots[0], roots[1])
    return np.diag(selected.astype(np.complex128))


@dataclass(frozen=True, slots=True, eq=False)
class _AcousticContactDescriptor:
    side: ContactSide
    orientation: str
    eta_lead: float
    surface_gf_method: str
    lead_physics_sha256: str
    coupling_sha256: str
    device_indices_sha256: str
    spectral_classification: _LeadSpectralClassification
    g0: Array
    surface_derivative: Array
    embedded_zero_retarded: Array
    contact_limit: Array
    embedded_limit: Array
    surface_equation_residual: float
    derivative_residual: float
    derivative_condition: float
    hermiticity_residual: float
    real_symmetry_residual: float
    passivity_minimum_eigenvalue: float
    analytic_zero_residual: float
    descriptor_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "g0",
            "surface_derivative",
            "embedded_zero_retarded",
            "contact_limit",
            "embedded_limit",
        ):
            object.__setattr__(
                self,
                name,
                _readonly(getattr(self, name), dtype=np.dtype(np.complex128)),
            )


def _build_acoustic_contact_descriptor(
    reservoir: HarmonicReservoir,
    *,
    side: ContactSide,
    ndof: int,
    eta_lead: float,
    surface_gf_method: str,
    spectral_classification: _LeadSpectralClassification | None = None,
) -> _AcousticContactDescriptor:
    if not np.isfinite(eta_lead) or eta_lead <= 0.0:
        raise ValueError("eta_lead must be finite and strictly positive.")
    spectral = spectral_classification or _classify_lead_spectrum(reservoir.lead)
    if spectral.classification != "acoustic" or not spectral.passed:
        raise ValueError(
            "an acoustic descriptor requires a lead classified as acoustic; "
            f"got {spectral.classification}."
        )
    method = surface_gf_method.strip().lower()
    a, b = _surface_orientation(reservoir.lead, side)
    if method in {"analytic_diagonal", "analytic-diagonal"}:
        g0 = _analytic_diagonal_surface_zero(reservoir.lead, eta_lead)
    else:
        g0 = surface_gf(
            omega=0.0,
            d00=np.asarray(reservoir.lead.d00, dtype=np.complex128),
            d01=a,
            d10=b,
            eta=float(eta_lead),
            method=surface_gf_method,
        )
    nlead = g0.shape[0]
    surface_operator = (
        -float(eta_lead) ** 2 * np.eye(nlead, dtype=np.complex128)
        - np.asarray(reservoir.lead.d00, dtype=np.complex128)
        - a @ g0 @ b
    )
    surface_error = surface_operator @ g0 - np.eye(
        nlead,
        dtype=np.complex128,
    )
    surface_denominator = max(
        float(np.linalg.norm(surface_operator @ g0, ord="fro"))
        + float(np.sqrt(nlead)),
        _TINY,
    )
    surface_residual = (
        float(np.linalg.norm(surface_error, ord="fro")) / surface_denominator
    )
    if surface_residual > 1.0e-10:
        raise ValueError("acoustic surface Green function fails its fixed-point equation.")

    left_multiplier = g0 @ a
    right_multiplier = b @ g0
    operator = np.eye(nlead * nlead, dtype=np.complex128) - np.kron(
        right_multiplier.T,
        left_multiplier,
    )
    rhs = (-g0 @ g0).reshape(-1, order="F")
    condition = float(np.linalg.cond(operator))
    if not np.isfinite(condition) or condition >= 1.0 / _EPS:
        raise ValueError("acoustic surface-derivative equation is singular or unresolved.")
    derivative = np.linalg.solve(operator, rhs).reshape((nlead, nlead), order="F")
    lhs_correction = g0 @ a @ derivative @ b @ g0
    equation_error = derivative - lhs_correction + g0 @ g0
    denominator = max(
        float(np.linalg.norm(derivative, ord="fro"))
        + float(np.linalg.norm(lhs_correction, ord="fro"))
        + float(np.linalg.norm(g0 @ g0, ord="fro")),
        _TINY,
    )
    derivative_residual = float(np.linalg.norm(equation_error, ord="fro")) / denominator
    if derivative_residual > 1.0e-10 or condition * derivative_residual > 1.0e-8:
        raise ValueError("acoustic surface-derivative residual exceeds tolerance.")

    coupling = _device_to_lead_coupling(reservoir, side)
    sigma_prime = coupling @ (2.0j * eta_lead * derivative) @ coupling.conj().T
    contact_limit = 1.0j * (sigma_prime - sigma_prime.conj().T)
    indices = normalize_contact_indices(reservoir, ndof, side)
    embedded = embed_contact_block(contact_limit, indices, ndof)
    limit_scale = max(float(np.linalg.norm(contact_limit, ord=2)), 1.0)
    audit_tolerance = 1.0e-10 * limit_scale + 1.0e-12
    hermiticity = float(np.max(np.abs(contact_limit - contact_limit.conj().T)))
    real_symmetry = max(
        float(np.max(np.abs(contact_limit.imag))),
        float(np.max(np.abs(contact_limit - contact_limit.T))),
    )
    minimum_eigenvalue = float(
        np.min(np.linalg.eigvalsh((contact_limit + contact_limit.conj().T) / 2.0))
    )
    if hermiticity > audit_tolerance:
        raise ValueError("acoustic Gamma/omega limit is not Hermitian.")
    # Nq=1 material operators are real.  Failing rather than projecting here
    # preserves the exact-zero negative-frequency transpose identity.
    lead_is_real = all(
        float(np.max(np.abs(np.asarray(value).imag)))
        <= 100.0 * _EPS * max(1.0, spectral.full_scale)
        for value in (
            np.asarray(reservoir.lead.d00, dtype=np.complex128),
            np.asarray(reservoir.lead.d01, dtype=np.complex128),
            np.asarray(
                reservoir.lead.d10
                if reservoir.lead.d10 is not None
                else reservoir.lead.d01.conj().T,
                dtype=np.complex128,
            ),
            coupling,
        )
    )
    if lead_is_real and real_symmetry > audit_tolerance:
        raise ValueError("real Nq=1 acoustic Gamma/omega limit is not real symmetric.")
    if minimum_eigenvalue < -audit_tolerance:
        raise ValueError("acoustic Gamma/omega limit is not positive semidefinite.")

    zero_sigma = coupling @ g0 @ coupling.conj().T
    embedded_zero_sigma = embed_contact_block(zero_sigma, indices, ndof)
    zero_gamma = 1.0j * (zero_sigma - zero_sigma.conj().T)
    zero_scale = max(float(np.linalg.norm(zero_sigma, ord=2)), 1.0)
    analytic_zero_residual = float(np.linalg.norm(zero_gamma, ord=2)) / zero_scale
    if analytic_zero_residual > 1.0e-10:
        raise ValueError("fixed-eta zero-frequency retarded contact is not Hermitian.")

    coupling_sha = domain_digest(
        "negfpy-contact-coupling-v1",
        np.array(coupling, dtype=np.complex128, order="C"),
    )
    indices_sha = domain_digest(
        "negfpy-contact-indices-v1",
        np.array(indices, dtype=np.int64, order="C"),
    )
    payload = {
        "side": side,
        "orientation": "outward-left" if side == "left" else "increasing-right",
        "eta_lead": float(eta_lead),
        "surface_gf_method": method,
        "lead_physics_sha256": digest_bytes(spectral.lead_physics_sha256),
        "coupling_sha256": digest_bytes(coupling_sha),
        "device_indices_sha256": digest_bytes(indices_sha),
        "g0": np.array(g0, dtype=np.complex128, order="C"),
        "surface_derivative": np.array(derivative, dtype=np.complex128, order="C"),
        "embedded_zero_retarded": np.array(
            embedded_zero_sigma,
            dtype=np.complex128,
            order="C",
        ),
        "contact_limit": np.array(contact_limit, dtype=np.complex128, order="C"),
        "embedded_limit": np.array(embedded, dtype=np.complex128, order="C"),
        "surface_equation_residual": surface_residual,
        "derivative_residual": derivative_residual,
        "derivative_condition": condition,
        "hermiticity_residual": hermiticity,
        "real_symmetry_residual": real_symmetry,
        "passivity_minimum_eigenvalue": minimum_eigenvalue,
        "analytic_zero_residual": analytic_zero_residual,
    }
    descriptor_sha = domain_digest("negfpy-acoustic-contact-descriptor-v1", payload)
    return _AcousticContactDescriptor(
        side=side,
        orientation=payload["orientation"],
        eta_lead=float(eta_lead),
        surface_gf_method=method,
        lead_physics_sha256=spectral.lead_physics_sha256,
        coupling_sha256=coupling_sha,
        device_indices_sha256=indices_sha,
        spectral_classification=spectral,
        g0=g0,
        surface_derivative=derivative,
        embedded_zero_retarded=embedded_zero_sigma,
        contact_limit=contact_limit,
        embedded_limit=embedded,
        surface_equation_residual=surface_residual,
        derivative_residual=derivative_residual,
        derivative_condition=condition,
        hermiticity_residual=hermiticity,
        real_symmetry_residual=real_symmetry,
        passivity_minimum_eigenvalue=minimum_eigenvalue,
        analytic_zero_residual=analytic_zero_residual,
        descriptor_sha256=descriptor_sha,
    )


def _stable_bose_broadening_products(
    omega: float,
    temperature: float,
    gamma: Array,
) -> tuple[Array, Array]:
    """Return ``n Gamma`` and ``(1+n) Gamma`` away from exact zero."""

    if omega == 0.0:
        raise ValueError("the exact-zero Bose-broadening product needs L_eta.")
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative.")
    matrix = np.asarray(gamma, dtype=np.complex128)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("broadening must contain only finite values.")
    if temperature == 0.0:
        if omega > 0.0:
            return np.zeros_like(matrix), matrix.copy()
        return -matrix, np.zeros_like(matrix)

    x = float(omega) / float(temperature)
    if abs(x) < _SERIES_THRESHOLD:
        ratio = matrix / float(omega)
        common = (
            float(temperature) * ratio
            + float(omega) * matrix / (12.0 * float(temperature))
            - float(omega) ** 3 * matrix / (720.0 * float(temperature) ** 3)
        )
        return common - 0.5 * matrix, common + 0.5 * matrix
    if x >= 700.0:
        occupation = 0.0
    elif x <= -700.0:
        occupation = -1.0
    else:
        occupation = 1.0 / np.expm1(x)
    return occupation * matrix, (1.0 + occupation) * matrix


def _reservoir_keldysh_from_retarded_acoustic(
    grid: SignedFrequencyGrid,
    retarded: Array,
    temperature: float,
    descriptor: _AcousticContactDescriptor,
) -> KeldyshSpectrum:
    """Build an acoustic contact spectrum using one audited ``L_eta``."""

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative.")
    sigma_r = np.asarray(retarded, dtype=np.complex128)
    expected = (grid.n_points,) + descriptor.embedded_limit.shape
    if sigma_r.shape != expected:
        raise ValueError(f"retarded must have shape {expected}.")
    if not np.all(np.isfinite(sigma_r)):
        raise ValueError("retarded must contain only finite values.")
    gamma = 1.0j * (sigma_r - np.swapaxes(sigma_r.conj(), -1, -2))
    zero = grid.zero_index
    zero_scale = max(float(np.max(np.abs(sigma_r))), 1.0)
    if not np.allclose(
        sigma_r[zero],
        descriptor.embedded_zero_retarded,
        rtol=1.0e-10,
        atol=1.0e-12,
    ):
        raise ValueError(
            "retarded acoustic contact does not match the descriptor at zero."
        )
    if float(np.max(np.abs(gamma[zero]))) > 1.0e-10 * zero_scale + 1.0e-12:
        raise ValueError("retarded acoustic contact has nonzero exact-zero broadening.")

    lesser = np.empty_like(sigma_r)
    greater = np.empty_like(sigma_r)
    for index, omega in enumerate(grid.values):
        if index == zero:
            value = (
                np.zeros_like(descriptor.embedded_limit)
                if temperature == 0.0
                else -1.0j * float(temperature) * descriptor.embedded_limit
            )
            lesser[index] = value
            greater[index] = value
        else:
            n_gamma, one_plus_n_gamma = _stable_bose_broadening_products(
                float(omega),
                float(temperature),
                gamma[index],
            )
            lesser[index] = -1.0j * n_gamma
            greater[index] = -1.0j * one_plus_n_gamma
    return KeldyshSpectrum(retarded=sigma_r, lesser=lesser, greater=greater)


def scalar_acoustic_contact_limit(t: float, eta_lead: float) -> float:
    """Exact finite-eta ``L_eta`` for the matched scalar acoustic chain."""

    value = float(t)
    eta = float(eta_lead)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("t must be finite and positive.")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta_lead must be finite and positive.")
    return 2.0 * ((eta**2 + 2.0 * value) / np.sqrt(eta**2 + 4.0 * value) - eta)


__all__ = [
    "HarmonicLeadClassification",
    "classify_harmonic_lead",
    "harmonic_lead_physics_sha256",
    "scalar_acoustic_contact_limit",
]
