"""Dense harmonic building blocks used by the anharmonic reference solver.

This module deliberately mirrors the public harmonic NEGF conventions without
calling private functions from :mod:`negfpy.core.negf`.  The independent dense
implementation is useful as a correctness oracle and, unlike
``device_green_function``, permits an additional device self-energy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from negfpy.core.surface_gf import surface_gf
from negfpy.core.types import Device1D

from .types import CubicSCBAProblem, HarmonicReservoir, KeldyshSpectrum


Array = np.ndarray
ContactSide = Literal["left", "right"]


def assemble_device_dynamical_matrix(device: Device1D) -> Array:
    """Return the dense dynamical matrix represented by ``device``."""

    npl = device.dof_per_layer
    ndof = device.n_layers * npl
    matrix = np.zeros((ndof, ndof), dtype=np.complex128)

    for layer, block in enumerate(device.onsite_blocks):
        sl = slice(layer * npl, (layer + 1) * npl)
        matrix[sl, sl] = np.asarray(block, dtype=np.complex128)

    for layer, block in enumerate(device.coupling_blocks):
        left = slice(layer * npl, (layer + 1) * npl)
        right = slice((layer + 1) * npl, (layer + 2) * npl)
        coupling = np.asarray(block, dtype=np.complex128)
        matrix[left, right] = coupling
        matrix[right, left] = coupling.conj().T

    if not np.all(np.isfinite(matrix)):
        raise ValueError("The device dynamical matrix must contain only finite values.")
    if not np.allclose(matrix, matrix.conj().T, rtol=1e-12, atol=1e-12):
        raise ValueError("The assembled device dynamical matrix must be Hermitian.")
    return matrix


def harmonic_frequency_upper_bound(problem: CubicSCBAProblem) -> float:
    """Return a conservative upper bound for every retained harmonic mode.

    The finite-device contribution uses its spectral norm.  Each periodic
    lead uses the spectral-norm triangle bound for
    ``D(q) = d00 + d01 exp(iq) + d10 exp(-iq)``.  The result is therefore
    enlarged by both contact-coupling norms, so it remains safe for arbitrary
    explicit contacts without pretending to be a sampled band maximum.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    device = assemble_device_dynamical_matrix(problem.device)
    maximum_square = float(np.linalg.norm(device, ord=2))
    for reservoir in (problem.left, problem.right):
        lead = reservoir.lead
        reverse = (
            lead.d10
            if lead.d10 is not None
            else np.asarray(lead.d01).conj().T
        )
        lead_square_bound = (
            float(np.linalg.norm(lead.d00, ord=2))
            + float(np.linalg.norm(lead.d01, ord=2))
            + float(np.linalg.norm(reverse, ord=2))
        )
        maximum_square = max(maximum_square, lead_square_bound)
    # Coupling the otherwise block-diagonal device/leads adds Hermitian
    # off-diagonal operators whose norms equal the rectangular coupling
    # norms.  The triangle inequality therefore gives a safe bound even for
    # the arbitrary explicit contacts allowed by HarmonicReservoir.
    maximum_square += float(
        np.linalg.norm(
            problem.left.resolved_device_to_lead("left"),
            ord=2,
        )
    )
    maximum_square += float(
        np.linalg.norm(
            problem.right.resolved_device_to_lead("right"),
            ord=2,
        )
    )
    return float(np.sqrt(maximum_square))


def _device_to_lead_coupling(
    reservoir: HarmonicReservoir,
    side: ContactSide,
) -> Array:
    """Resolve the rectangular device-to-lead coupling for one contact."""

    if side not in {"left", "right"}:
        raise ValueError("side must be either 'left' or 'right'.")

    if reservoir.device_to_lead is not None:
        coupling = np.asarray(reservoir.device_to_lead, dtype=np.complex128)
    elif side == "left":
        lead = reservoir.lead
        coupling = lead.d10 if lead.d10 is not None else lead.d01.conj().T
        coupling = np.asarray(coupling, dtype=np.complex128)
    else:
        coupling = np.asarray(reservoir.lead.d01, dtype=np.complex128)

    if coupling.ndim != 2:
        raise ValueError("device_to_lead must be a two-dimensional array.")
    nlead = reservoir.lead.d00.shape[0]
    if coupling.shape[1] != nlead:
        raise ValueError(
            "device_to_lead has an incompatible lead dimension "
            f"(got {coupling.shape[1]}, expected {nlead})."
        )
    if coupling.shape[0] == 0:
        raise ValueError("device_to_lead must couple at least one device degree of freedom.")
    if not np.all(np.isfinite(coupling)):
        raise ValueError("device_to_lead must contain only finite values.")
    return coupling


def normalize_contact_indices(
    reservoir: HarmonicReservoir,
    ndof: int,
    side: ContactSide,
) -> Array:
    """Return validated device indices for one reservoir."""

    if ndof <= 0:
        raise ValueError("ndof must be positive.")
    coupling = _device_to_lead_coupling(reservoir, side)
    ncontact = coupling.shape[0]
    indices = reservoir.device_indices

    if indices is None:
        if side == "left":
            result = np.arange(ncontact, dtype=int)
        elif side == "right":
            result = np.arange(ndof - ncontact, ndof, dtype=int)
        else:
            raise ValueError("side must be either 'left' or 'right'.")
    elif isinstance(indices, slice):
        result = np.arange(ndof, dtype=int)[indices]
    else:
        result = np.asarray(indices, dtype=int).ravel()

    if result.size != ncontact:
        raise ValueError(
            f"{side.capitalize()} contact index count must equal the number of "
            f"device coupling rows (got {result.size}, expected {ncontact})."
        )
    if np.any(result < 0) or np.any(result >= ndof):
        raise ValueError(f"{side.capitalize()} contact indices are outside the device.")
    if np.unique(result).size != result.size:
        raise ValueError(f"{side.capitalize()} contact indices must be unique.")
    return result


def embed_contact_block(block: Array, indices: Array, ndof: int) -> Array:
    """Embed a contact-sized square matrix in the full device space."""

    values = np.asarray(block, dtype=np.complex128)
    idx = np.asarray(indices, dtype=int).ravel()
    if values.shape != (idx.size, idx.size):
        raise ValueError(
            "Contact block shape must match the number of contact indices "
            f"(got {values.shape}, expected {(idx.size, idx.size)})."
        )
    result = np.zeros((ndof, ndof), dtype=np.complex128)
    result[np.ix_(idx, idx)] = values
    return result


def retarded_contact_self_energy(
    omega: float,
    reservoir: HarmonicReservoir,
    ndof: int,
    eta_lead: float,
    *,
    side: ContactSide,
    surface_gf_method: str = "sancho_rubio",
) -> Array:
    """Return one retarded reservoir self-energy embedded in device space."""

    if not np.isfinite(omega):
        raise ValueError("omega must be finite.")
    if not np.isfinite(eta_lead) or eta_lead <= 0.0:
        raise ValueError("eta_lead must be finite and strictly positive.")

    coupling = _device_to_lead_coupling(reservoir, side)
    indices = normalize_contact_indices(reservoir, ndof, side)
    lead = reservoir.lead
    reverse = lead.d10 if lead.d10 is not None else lead.d01.conj().T
    if side == "left":
        surface_d01 = reverse
        surface_d10 = lead.d01
    else:
        surface_d01 = lead.d01
        surface_d10 = reverse
    surface = surface_gf(
        omega=float(omega),
        d00=lead.d00,
        d01=surface_d01,
        d10=surface_d10,
        eta=float(eta_lead),
        method=surface_gf_method,
    )
    block = coupling @ surface @ coupling.conj().T
    return embed_contact_block(block, indices, ndof)


def _analytic_diagonal_contact_spectrum(
    problem: CubicSCBAProblem,
    reservoir: HarmonicReservoir,
    eta_lead: float,
    *,
    side: ContactSide,
) -> Array:
    """Return an exact vectorized contact for reciprocal diagonal leads."""

    lead = reservoir.lead
    d00 = np.asarray(lead.d00, dtype=np.complex128)
    d01 = np.asarray(lead.d01, dtype=np.complex128)
    d10 = np.asarray(
        lead.d10 if lead.d10 is not None else lead.d01.conj().T,
        dtype=np.complex128,
    )
    scale = max(
        1.0,
        float(np.max(np.abs(d00))),
        float(np.max(np.abs(d01))),
        float(np.max(np.abs(d10))),
    )
    tolerance = 128.0 * np.finfo(float).eps * scale
    for values, name in ((d00, "d00"), (d01, "d01"), (d10, "d10")):
        diagonal = np.diag(np.diag(values))
        if float(np.max(np.abs(values - diagonal))) > tolerance:
            raise ValueError(
                "surface_gf_method='analytic_diagonal' requires diagonal "
                f"lead {name}."
            )
    if float(np.max(np.abs(d10 - d01.conj().T))) > tolerance:
        raise ValueError(
            "surface_gf_method='analytic_diagonal' requires reciprocal "
            "d10=d01^dagger."
        )
    onsite = np.diag(d00)
    if float(np.max(np.abs(onsite.imag))) > tolerance:
        raise ValueError(
            "surface_gf_method='analytic_diagonal' requires real onsite "
            "diagonal entries."
        )
    hopping_product = np.diag(d01) * np.diag(d10)
    if (
        float(np.max(np.abs(hopping_product.imag))) > tolerance
        or np.any(hopping_product.real <= tolerance)
    ):
        raise ValueError(
            "surface_gf_method='analytic_diagonal' requires nonzero "
            "reciprocal diagonal lead couplings."
        )

    frequencies = problem.grid.values
    z = (frequencies[:, None] + 1.0j * eta_lead) ** 2
    delta = z - onsite.real[None, :]
    coefficient = hopping_product.real[None, :]
    discriminant = np.sqrt(delta**2 - 4.0 * coefficient)
    denominator_plus = delta + discriminant
    denominator_minus = delta - discriminant
    stable_denominator = np.where(
        np.abs(denominator_plus) >= np.abs(denominator_minus),
        denominator_plus,
        denominator_minus,
    )
    small = 2.0 / stable_denominator
    other = 1.0 / (coefficient * small)
    small_magnitude = np.abs(small)
    other_magnitude = np.abs(other)
    tied = np.abs(small_magnitude - other_magnitude) <= (
        64.0
        * np.finfo(float).eps
        * np.maximum(1.0, np.maximum(small_magnitude, other_magnitude))
    )
    positive = frequencies > 0.0
    negative = frequencies < 0.0
    causal_small = np.ones(small.shape, dtype=bool)
    causal_small[positive] = (
        small[positive].imag <= other[positive].imag
    )
    causal_small[negative] = (
        small[negative].imag >= other[negative].imag
    )
    surface_diagonal = np.where(tied & ~causal_small, other, small)

    residual = (
        coefficient * surface_diagonal**2
        - delta * surface_diagonal
        + 1.0
    )
    residual_scale = np.maximum(
        1.0,
        np.abs(coefficient * surface_diagonal**2)
        + np.abs(delta * surface_diagonal),
    )
    if float(np.max(np.abs(residual) / residual_scale)) > 1.0e-11:
        raise RuntimeError(
            "analytic diagonal surface root failed its quadratic."
        )

    coupling = _device_to_lead_coupling(reservoir, side)
    block = np.einsum(
        "ai,wi,bi->wab",
        coupling,
        surface_diagonal,
        coupling.conj(),
        optimize=True,
    )
    indices = normalize_contact_indices(reservoir, problem.ndof, side)
    result = np.zeros(
        (problem.grid.n_points, problem.ndof, problem.ndof),
        dtype=np.complex128,
    )
    result[:, indices[:, None], indices[None, :]] = block
    return result


def build_retarded_contacts(
    problem: CubicSCBAProblem,
    eta_lead: float,
    *,
    surface_gf_method: str = "sancho_rubio",
) -> tuple[Array, Array]:
    """Build left and right retarded contact self-energies on the signed grid."""

    if surface_gf_method.strip().lower() in {
        "analytic_diagonal",
        "analytic-diagonal",
    }:
        return (
            _analytic_diagonal_contact_spectrum(
                problem,
                problem.left,
                eta_lead,
                side="left",
            ),
            _analytic_diagonal_contact_spectrum(
                problem,
                problem.right,
                eta_lead,
                side="right",
            ),
        )

    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    sigma_left = np.empty(shape, dtype=np.complex128)
    sigma_right = np.empty(shape, dtype=np.complex128)

    for iw, omega in enumerate(problem.grid.values):
        sigma_left[iw] = retarded_contact_self_energy(
            float(omega),
            problem.left,
            problem.ndof,
            eta_lead,
            side="left",
            surface_gf_method=surface_gf_method,
        )
        sigma_right[iw] = retarded_contact_self_energy(
            float(omega),
            problem.right,
            problem.ndof,
            eta_lead,
            side="right",
            surface_gf_method=surface_gf_method,
        )

    return sigma_left, sigma_right


def _validate_frequency_matrices(
    values: Array,
    *,
    name: str,
    n_frequency: int,
    ndof: int,
) -> Array:
    result = np.asarray(values, dtype=np.complex128)
    expected = (n_frequency, ndof, ndof)
    if result.shape != expected:
        raise ValueError(f"{name} must have shape {expected}, got {result.shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def solve_retarded(
    problem: CubicSCBAProblem,
    sigma_left_retarded: Array,
    sigma_right_retarded: Array,
    sigma_pp_retarded: Array | None = None,
) -> Array:
    """Solve the dense retarded Dyson equation with exactly zero device eta."""

    nw = problem.grid.n_points
    ndof = problem.ndof
    sigma_left = _validate_frequency_matrices(
        sigma_left_retarded,
        name="sigma_left_retarded",
        n_frequency=nw,
        ndof=ndof,
    )
    sigma_right = _validate_frequency_matrices(
        sigma_right_retarded,
        name="sigma_right_retarded",
        n_frequency=nw,
        ndof=ndof,
    )
    if sigma_pp_retarded is None:
        sigma_pp = np.zeros_like(sigma_left)
    else:
        sigma_pp = _validate_frequency_matrices(
            sigma_pp_retarded,
            name="sigma_pp_retarded",
            n_frequency=nw,
            ndof=ndof,
        )

    dynamical = assemble_device_dynamical_matrix(problem.device)
    identity = np.eye(ndof, dtype=np.complex128)
    green = np.empty_like(sigma_left)
    for iw, omega in enumerate(problem.grid.values):
        operator = (
            float(omega) ** 2 * identity
            - dynamical
            - sigma_left[iw]
            - sigma_right[iw]
            - sigma_pp[iw]
        )
        green[iw] = np.linalg.solve(operator, identity)
    return green


def solve_keldysh_green(
    retarded: Array,
    sigma_lesser: Array,
    sigma_greater: Array,
) -> KeldyshSpectrum:
    """Solve the dense Keldysh equation for supplied total self-energies."""

    gr = np.asarray(retarded, dtype=np.complex128)
    lesser = np.asarray(sigma_lesser, dtype=np.complex128)
    greater = np.asarray(sigma_greater, dtype=np.complex128)
    if gr.ndim != 3 or gr.shape[1] != gr.shape[2]:
        raise ValueError("retarded must have shape (n_frequency, ndof, ndof).")
    if lesser.shape != gr.shape or greater.shape != gr.shape:
        raise ValueError("All retarded, lesser, and greater arrays must have the same shape.")
    if not (
        np.all(np.isfinite(gr))
        and np.all(np.isfinite(lesser))
        and np.all(np.isfinite(greater))
    ):
        raise ValueError("Green functions and self-energies must contain only finite values.")

    ga = np.swapaxes(gr.conj(), -1, -2)
    green_lesser = gr @ lesser @ ga
    green_greater = gr @ greater @ ga
    return KeldyshSpectrum(retarded=gr, lesser=green_lesser, greater=green_greater)
