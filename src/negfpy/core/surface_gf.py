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
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                g = np.linalg.inv(z * eye - eps)
                a_g_b = alpha @ g @ beta
                b_g_a = beta @ g @ alpha

                eps_s_new = eps_s + a_g_b
                eps_new = eps + a_g_b + b_g_a
                alpha_new = alpha @ g @ alpha
                beta_new = beta @ g @ beta
        except FloatingPointError as exc:
            raise RuntimeError(
                "Sancho-Rubio decimation became numerically unstable. "
                "Try a larger eta or a different surface_gf method."
            ) from exc

        if not (
            np.all(np.isfinite(eps_s_new))
            and np.all(np.isfinite(eps_new))
            and np.all(np.isfinite(alpha_new))
            and np.all(np.isfinite(beta_new))
        ):
            raise RuntimeError(
                "Sancho-Rubio decimation produced non-finite values. "
                "Try a larger eta or a different surface_gf method."
            )

        err = max(float(np.max(np.abs(alpha_new))), float(np.max(np.abs(beta_new))))

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


def surface_gf_generalized_eigen_svd(
    omega: float,
    d00: Array,
    d01: Array,
    d10: Array | None = None,
    eta: float = 1e-8,
    unit_circle_tol: float = 1e-10,
    svd_rcond: float = 1e-12,
) -> Array:
    """Return retarded surface GF via generalized-eigen + SVD pseudo-inverse.

    This follows the same QEP linearization as ``surface_gf_generalized_eigen``
    but uses an SVD pseudo-inverse for the mode matrix to improve robustness
    near degenerate/ill-conditioned mode subspaces.
    """

    n = int(d00.shape[0])
    if d00.shape != (n, n) or d01.shape != (n, n):
        raise ValueError("d00 and d01 must be square matrices of the same shape.")
    if svd_rcond <= 0.0:
        raise ValueError("svd_rcond must be positive.")

    z = (omega + 1j * eta) ** 2
    eye = np.eye(n, dtype=np.complex128)
    a = d00.astype(np.complex128, copy=False) - z * eye
    b = _resolve_d10(d01=d01, d10=d10)
    c = d01.astype(np.complex128, copy=False)

    mat_a = np.block([[-a, -b], [eye, np.zeros((n, n), dtype=np.complex128)]])
    mat_b = np.block([[c, np.zeros((n, n), dtype=np.complex128)], [np.zeros((n, n), dtype=np.complex128), eye]])
    evals, evecs = eig(mat_a, mat_b, right=True)
    finite = np.isfinite(evals)
    evals = evals[finite]
    evecs = evecs[:, finite]
    if evals.size < n:
        raise RuntimeError("Generalized-eigen-SVD surface GF failed: insufficient finite eigenmodes.")

    absvals = np.abs(evals)
    inside = np.where(absvals < (1.0 - float(unit_circle_tol)))[0]
    if inside.size >= n:
        pick = inside[np.argsort(absvals[inside])[:n]]
    else:
        pick = np.argsort(absvals)[:n]

    lam = evals[pick]
    phi = evecs[n:, pick]
    phi_pinv = np.linalg.pinv(phi, rcond=float(svd_rcond))
    transfer = phi @ np.diag(lam) @ phi_pinv
    sigma = c @ transfer
    return np.linalg.inv(z * eye - d00.astype(np.complex128, copy=False) - sigma)


def _pinv_legacy_svd(mat: Array) -> Array:
    """Pseudo-inverse with legacy SVD cutoff max(tiny, smax*eps)."""

    u, s, vh = np.linalg.svd(mat, full_matrices=True)
    if s.size == 0:
        return np.zeros((mat.shape[1], mat.shape[0]), dtype=np.complex128)
    cutoff = max(np.finfo(float).tiny, float(s[0]) * np.finfo(float).eps)
    sinv = np.zeros_like(s, dtype=np.float64)
    keep = s >= cutoff
    sinv[keep] = 1.0 / s[keep]
    return (vh.conj().T[:, : s.size] * sinv) @ u[:, : s.size].conj().T


def surface_gf_legacy_eigen_svd(
    omega: float,
    d00: Array,
    d01: Array,
    d10: Array | None = None,
    eta: float = 1e-8,
    unit_circle_tol: float = 0.0,
) -> Array:
    """Return retarded surface GF using a legacy generalized-eigen SVD path.

    This mirrors old Fortran ``sfg_solve`` style:
    1) Keep all finite eigenmodes with |lambda| < 1 - unit_circle_tol.
    2) Build rectangular mode matrix and SVD pseudo-inverse with machine cutoff.
    3) Compute g11 then update to g00 via Dyson-like correction.
    """

    n = int(d00.shape[0])
    if d00.shape != (n, n) or d01.shape != (n, n):
        raise ValueError("d00 and d01 must be square matrices of the same shape.")

    z = (omega + 1j * eta) ** 2
    eye = np.eye(n, dtype=np.complex128)
    h01 = d01.astype(np.complex128, copy=False)
    h10 = _resolve_d10(d01=d01, d10=d10)
    th00 = z * eye - d00.astype(np.complex128, copy=False)
    th11 = th00.copy()

    mat_a = np.block([[th11, -eye], [h10, np.zeros((n, n), dtype=np.complex128)]])
    mat_b = np.block([[h01, np.zeros((n, n), dtype=np.complex128)], [np.zeros((n, n), dtype=np.complex128), eye]])
    evals, evecs = eig(mat_a, mat_b, right=True)

    finite = np.isfinite(evals)
    evals = evals[finite]
    evecs = evecs[:, finite]
    if evals.size == 0:
        raise RuntimeError("Legacy-eigen-SVD surface GF failed: no finite eigenmodes.")

    decaying = np.where(np.abs(evals) < (1.0 - float(unit_circle_tol)))[0]
    if decaying.size == 0:
        raise RuntimeError("Legacy-eigen-SVD surface GF failed: no decaying eigenmodes.")

    lam = evals[decaying]
    # For this linearization, upper block matches old transE selection.
    trans_e = evecs[:n, decaying]
    trans_e_inv = _pinv_legacy_svd(trans_e)
    gam = np.diag(lam)

    g11_inv = th11 - h01 @ trans_e @ gam @ trans_e_inv
    g11 = np.linalg.inv(g11_inv)
    g00_inv = th00 - h01 @ g11 @ h10
    return np.linalg.inv(g00_inv)


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
    svd_rcond: float = 1e-12,
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
    if m in {"generalized_eigen_svd", "generalized-eigen-svd", "ge_svd", "gep_svd"}:
        return surface_gf_generalized_eigen_svd(
            omega=omega,
            d00=d00,
            d01=d01,
            d10=d10,
            eta=eta,
            unit_circle_tol=unit_circle_tol,
            svd_rcond=svd_rcond,
        )
    if m in {"legacy_eigen_svd", "legacy-eigen-svd", "legacy", "old_svd"}:
        return surface_gf_legacy_eigen_svd(
            omega=omega,
            d00=d00,
            d01=d01,
            d10=d10,
            eta=eta,
            unit_circle_tol=unit_circle_tol,
        )
    raise ValueError(
        "Unknown surface GF method. Use one of: "
        "'sancho_rubio', 'generalized_eigen', 'generalized_eigen_svd', 'legacy_eigen_svd'."
    )
