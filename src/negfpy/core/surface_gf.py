"""Surface Green's function solvers for periodic leads."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def surface_gf_sancho_rubio(
    omega: float,
    d00: Array,
    d01: Array,
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
    beta = d01.conj().T.astype(np.complex128, copy=True)
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
