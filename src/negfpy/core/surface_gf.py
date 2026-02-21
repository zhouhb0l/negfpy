"""Surface Green's function solvers for periodic leads."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig


Array = np.ndarray


def _resolve_d10(d01: Array, d10: Array | None) -> Array:
    if d10 is None:
        return d01.conj().T.astype(np.complex128, copy=True)
    return d10.astype(np.complex128, copy=True)


def surface_gf_sancho_rubio(
    omega: float,
    d00: Array,
    d01: Array,
    d10: Array | None = None,
    eta: float = 1e-8,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> Array:
    """Return the retarded surface Green's function for a semi-infinite lead.

    The lead is assumed block-tridiagonal and periodic with onsite block ``d00``
    and nearest-neighbor coupling ``d01``.
    """

    z = (omega + 1j * eta) ** 2
    eye = np.eye(d00.shape[0], dtype=np.complex128)

    alpha = d01.astype(np.complex128, copy=True)
    beta = _resolve_d10(d01=d01, d10=d10)
    eps = d00.astype(np.complex128, copy=True)
    eps_s = eps.copy()

    for _ in range(maxiter):
        g = np.linalg.inv(z * eye - eps)
        a_g_b = alpha @ g @ beta
        b_g_a = beta @ g @ alpha

        eps_s_new = eps_s + a_g_b
        eps_new = eps + a_g_b + b_g_a
        alpha_new = alpha @ g @ alpha
        beta_new = beta @ g @ beta

        err = max(np.linalg.norm(alpha_new), np.linalg.norm(beta_new))

        eps_s = eps_s_new
        eps = eps_new
        alpha = alpha_new
        beta = beta_new

        if err < tol:
            break
    else:
        raise RuntimeError("Sancho-Rubio decimation did not converge within maxiter.")

    return np.linalg.inv(z * eye - eps_s)


def surface_gf_generalized_eigen(
    omega: float,
    d00: Array,
    d01: Array,
    d10: Array | None = None,
    eta: float = 1e-8,
    unit_circle_tol: float = 1e-10,
) -> Array:
    """Return retarded surface GF using a generalized-eigenmode formulation.

    Solves the quadratic eigenvalue problem of the periodic block-tridiagonal
    lead and builds the decaying transfer matrix from modes with |lambda| < 1.
    """

    n = int(d00.shape[0])
    if d00.shape != (n, n) or d01.shape != (n, n):
        raise ValueError("d00 and d01 must be square matrices of the same shape.")

    z = (omega + 1j * eta) ** 2
    eye = np.eye(n, dtype=np.complex128)
    a = d00.astype(np.complex128, copy=False) - z * eye
    b = _resolve_d10(d01=d01, d10=d10)
    c = d01.astype(np.complex128, copy=False)

    # Linearization of QEP: (c*lambda^2 + a*lambda + b) phi = 0.
    mat_a = np.block([[-a, -b], [eye, np.zeros((n, n), dtype=np.complex128)]])
    mat_b = np.block([[c, np.zeros((n, n), dtype=np.complex128)], [np.zeros((n, n), dtype=np.complex128), eye]])
    evals, evecs = eig(mat_a, mat_b, right=True)
    finite = np.isfinite(evals)
    evals = evals[finite]
    evecs = evecs[:, finite]
    if evals.size < n:
        raise RuntimeError("Generalized-eigen surface GF failed: insufficient finite eigenmodes.")

    absvals = np.abs(evals)
    inside = np.where(absvals < (1.0 - float(unit_circle_tol)))[0]
    if inside.size >= n:
        pick = inside[np.argsort(absvals[inside])[:n]]
    else:
        # Robust fallback: retarded eta nudges the decaying set to smaller |lambda|.
        pick = np.argsort(absvals)[:n]

    lam = evals[pick]
    phi = evecs[n:, pick]
    if np.linalg.matrix_rank(phi) < n:
        raise RuntimeError("Generalized-eigen surface GF failed: selected eigenvectors are rank-deficient.")

    transfer = phi @ np.diag(lam) @ np.linalg.inv(phi)
    sigma = c @ transfer
    return np.linalg.inv(z * eye - d00.astype(np.complex128, copy=False) - sigma)


def surface_gf(
    omega: float,
    d00: Array,
    d01: Array,
    d10: Array | None = None,
    eta: float = 1e-8,
    *,
    method: str = "sancho_rubio",
    tol: float = 1e-12,
    maxiter: int = 200,
    unit_circle_tol: float = 1e-10,
) -> Array:
    """Return retarded surface GF using the selected solver."""

    m = method.strip().lower()
    if m in {"sancho_rubio", "sancho-rubio", "sr"}:
        return surface_gf_sancho_rubio(
            omega=omega,
            d00=d00,
            d01=d01,
            d10=d10,
            eta=eta,
            tol=tol,
            maxiter=maxiter,
        )
    if m in {"generalized_eigen", "generalized-eigen", "eig", "gep"}:
        return surface_gf_generalized_eigen(
            omega=omega,
            d00=d00,
            d01=d01,
            d10=d10,
            eta=eta,
            unit_circle_tol=unit_circle_tol,
        )
    raise ValueError(
        "Unknown surface GF method. Use one of: "
        "'sancho_rubio', 'generalized_eigen'."
    )
