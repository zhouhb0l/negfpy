"""Harmonic reservoir self-energies on a signed frequency grid."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from negfpy.core.types import LeadBlocks
from ._harmonic import build_retarded_contacts
from .grid import SignedFrequencyGrid
from .types import CubicSCBAProblem, KeldyshSpectrum


Array = np.ndarray


def minimum_lead_frequency_squared(
    lead: LeadBlocks,
    *,
    q_samples: int = 257,
) -> float:
    """Return the minimum bulk eigenvalue of a reciprocal 1D lead.

    The Brillouin zone is sampled including Gamma, and every sampled local
    basin of the lowest branch is refined.  V1 uses this as a fail-closed
    check that the Bose--broadening product has a genuine gap at zero.
    """

    if not isinstance(lead, LeadBlocks):
        raise TypeError("lead must be a LeadBlocks instance.")
    if (
        isinstance(q_samples, (bool, np.bool_))
        or not isinstance(q_samples, (int, np.integer))
    ):
        raise TypeError("q_samples must be an integer.")
    q_samples = int(q_samples)
    if q_samples < 9:
        raise ValueError("q_samples must be at least nine.")

    d00 = np.asarray(lead.d00, dtype=np.complex128)
    d01 = np.asarray(lead.d01, dtype=np.complex128)
    d10 = (
        np.asarray(lead.d10, dtype=np.complex128)
        if lead.d10 is not None
        else d01.conj().T
    )
    if not (
        np.all(np.isfinite(d00))
        and np.all(np.isfinite(d01))
        and np.all(np.isfinite(d10))
    ):
        raise ValueError("Lead blocks must contain only finite values.")
    scale = max(
        float(np.linalg.norm(d00, ord=2)),
        float(np.linalg.norm(d01, ord=2)),
        float(np.linalg.norm(d10, ord=2)),
        np.finfo(float).tiny,
    )
    reciprocal_tolerance = 100.0 * np.finfo(float).eps * scale
    if float(np.max(np.abs(d00 - d00.conj().T))) > reciprocal_tolerance:
        raise ValueError("V1 gap certification requires Hermitian lead d00.")
    if float(np.max(np.abs(d10 - d01.conj().T))) > reciprocal_tolerance:
        raise ValueError(
            "V1 gap certification requires reciprocal d10 = d01^dagger."
        )

    def lowest(qpoint: float) -> float:
        phase = np.exp(1j * float(qpoint))
        dynamical = d00 + d01 * phase + d10 * phase.conjugate()
        hermiticity = float(
            np.max(np.abs(dynamical - dynamical.conj().T))
        )
        if hermiticity > reciprocal_tolerance:
            raise ValueError(
                "Lead dynamical matrix is not Hermitian during gap certification."
            )
        return float(np.linalg.eigvalsh(dynamical)[0])

    step = 2.0 * np.pi / q_samples
    qpoints = -np.pi + step * np.arange(q_samples, dtype=float)
    sampled = np.asarray([lowest(value) for value in qpoints])
    candidates: list[int] = []
    for index in range(q_samples):
        if (
            sampled[index] <= sampled[(index - 1) % q_samples]
            and sampled[index] <= sampled[(index + 1) % q_samples]
        ):
            candidates.append(index)
    if not candidates:
        candidates = [int(np.argmin(sampled))]

    minimum = min(float(np.min(sampled)), lowest(0.0))
    for index in candidates:
        center = float(qpoints[index])
        optimized = minimize_scalar(
            lowest,
            bounds=(center - step, center + step),
            method="bounded",
            options={"xatol": 32.0 * np.finfo(float).eps},
        )
        if not optimized.success or not np.isfinite(optimized.fun):
            raise ValueError(
                "Could not certify the harmonic lead gap by BZ minimization."
            )
        minimum = min(minimum, float(optimized.fun))
    return minimum


def _require_resolved_lead_gap(lead: LeadBlocks) -> float:
    minimum = minimum_lead_frequency_squared(lead)
    d00 = np.asarray(lead.d00, dtype=np.complex128)
    d01 = np.asarray(lead.d01, dtype=np.complex128)
    d10 = (
        np.asarray(lead.d10, dtype=np.complex128)
        if lead.d10 is not None
        else d01.conj().T
    )
    scale = max(
        float(np.linalg.norm(d00, ord=2)),
        float(np.linalg.norm(d01, ord=2)),
        float(np.linalg.norm(d10, ord=2)),
        np.finfo(float).tiny,
    )
    threshold = 100.0 * np.finfo(float).eps * scale
    if minimum <= threshold:
        raise ValueError(
            "V1 requires a spectrally gapped harmonic reservoir; "
            f"minimum omega^2={minimum:.6e} is not above the numerical "
            f"gap threshold {threshold:.6e}. Acoustic/soft leads require "
            "the analytic n_B*Gamma zero-frequency limit."
        )
    return minimum


def _positive_bose_occupation(omega: Array, temperature: float) -> Array:
    """Evaluate ``1/expm1(omega / temperature)`` without overflow."""

    values = np.asarray(omega, dtype=float)
    if np.any(values <= 0.0):
        raise ValueError("Positive-frequency Bose evaluation requires omega > 0.")
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative.")
    if temperature == 0.0:
        return np.zeros_like(values)

    x = values / float(temperature)
    result = np.zeros_like(x)
    small = x < 1.0e-5
    regular = (x >= 1.0e-5) & (x < 700.0)
    # Bernoulli expansion avoids cancellation in expm1 for very small x.
    xs = x[small]
    result[small] = 1.0 / xs - 0.5 + xs / 12.0 - xs**3 / 720.0
    result[regular] = 1.0 / np.expm1(x[regular])
    return result


def bose_occupation_signed(omega: Array | float, temperature: float) -> Array | float:
    """Return the signed-frequency Bose function away from zero.

    The exact bosonic continuation ``n(-omega) = -[1+n(omega)]`` is used.
    The value at zero is singular and is returned as ``+inf``; numerical
    contact products handle that point separately through the gapped-contact
    condition.
    """

    scalar = np.isscalar(omega)
    values = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("omega must contain only finite values.")
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative.")

    result = np.empty_like(values)
    positive = values > 0.0
    negative = values < 0.0
    zero = ~(positive | negative)
    if np.any(positive):
        result[positive] = _positive_bose_occupation(values[positive], temperature)
    if np.any(negative):
        result[negative] = -1.0 - _positive_bose_occupation(-values[negative], temperature)
    result[zero] = np.inf
    if scalar:
        return float(result)
    return result


def broadening(retarded: Array) -> Array:
    """Return ``Gamma = i (Sigma^r - Sigma^a)`` frequency by frequency."""

    sigma = np.asarray(retarded, dtype=np.complex128)
    if sigma.ndim != 3 or sigma.shape[1] != sigma.shape[2]:
        raise ValueError("retarded must have shape (n_frequency, ndof, ndof).")
    return 1j * (sigma - np.swapaxes(sigma.conj(), -1, -2))


def reservoir_keldysh_from_retarded(
    grid: SignedFrequencyGrid,
    retarded: Array,
    temperature: float,
    *,
    zero_gamma_atol: float = 1.0e-10,
) -> KeldyshSpectrum:
    """Build equilibrium reservoir ``r``, ``<``, and ``>`` self-energies.

    V1 supports a gapped contact at zero frequency.  Its exact zero-frequency
    broadening vanishes, so the otherwise singular Bose product has the
    unambiguous value zero.  A nonzero broadening is rejected rather than
    silently assigning an acoustic limiting prescription.
    """

    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and non-negative.")
    if zero_gamma_atol < 0.0 or not np.isfinite(zero_gamma_atol):
        raise ValueError("zero_gamma_atol must be finite and non-negative.")

    sigma_r = np.asarray(retarded, dtype=np.complex128)
    if sigma_r.ndim != 3 or sigma_r.shape[1] != sigma_r.shape[2]:
        raise ValueError("retarded must have shape (n_frequency, ndof, ndof).")
    if sigma_r.shape[0] != grid.n_points:
        raise ValueError(
            "retarded frequency dimension does not match the signed grid "
            f"(got {sigma_r.shape[0]}, expected {grid.n_points})."
        )
    if not np.all(np.isfinite(sigma_r)):
        raise ValueError("retarded must contain only finite values.")

    gamma = broadening(sigma_r)
    zero_gamma = gamma[grid.zero_index]
    gamma_scale = max(1.0, float(np.max(np.abs(gamma))))
    if float(np.max(np.abs(zero_gamma))) > zero_gamma_atol * gamma_scale:
        raise ValueError(
            "V1 requires a gapped reservoir with zero broadening at omega=0; "
            "an acoustic Bose-Gamma limiting prescription is not implemented."
        )

    lesser = np.zeros_like(sigma_r)
    greater = np.zeros_like(sigma_r)
    nonzero = np.arange(grid.n_points) != grid.zero_index
    occupation = np.asarray(
        bose_occupation_signed(grid.values[nonzero], temperature),
        dtype=float,
    )
    lesser[nonzero] = -1j * occupation[:, None, None] * gamma[nonzero]
    greater[nonzero] = -1j * (1.0 + occupation)[:, None, None] * gamma[nonzero]

    return KeldyshSpectrum(retarded=sigma_r, lesser=lesser, greater=greater)


def build_contact_spectra(
    problem: CubicSCBAProblem,
    eta_lead: float,
    *,
    surface_gf_method: str = "sancho_rubio",
    zero_gamma_atol: float = 1.0e-10,
) -> tuple[KeldyshSpectrum, KeldyshSpectrum]:
    """Return signed-frequency Keldysh self-energies for both contacts."""

    _require_resolved_lead_gap(problem.left.lead)
    _require_resolved_lead_gap(problem.right.lead)
    sigma_left_r, sigma_right_r = build_retarded_contacts(
        problem,
        eta_lead,
        surface_gf_method=surface_gf_method,
    )
    left = reservoir_keldysh_from_retarded(
        problem.grid,
        sigma_left_r,
        problem.left.temperature,
        zero_gamma_atol=zero_gamma_atol,
    )
    right = reservoir_keldysh_from_retarded(
        problem.grid,
        sigma_right_r,
        problem.right.temperature,
        zero_gamma_atol=zero_gamma_atol,
    )
    return left, right


__all__ = [
    "bose_occupation_signed",
    "broadening",
    "build_contact_spectra",
    "minimum_lead_frequency_squared",
    "reservoir_keldysh_from_retarded",
]
