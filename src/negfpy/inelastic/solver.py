"""Generic inelastic solver hooks built on top of the ballistic core assembly."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from negfpy.core.negf import (
    ContactIndices,
    DeviceLike,
    DeviceToLeadLike,
    KPar,
    LeadLike,
    _broadening,
    _build_system_matrix_and_contact_sigmas,
    _embed_self_energies,
    _resolve_device,
    _resolve_device_to_lead_coupling,
)

from .base import InelasticSolveInfo, PPSelfEnergyModel


Array = np.ndarray


def _validate_sigma_shape(sigma: Array, dim: int) -> Array:
    out = np.asarray(sigma, dtype=np.complex128)
    if out.shape != (dim, dim):
        raise ValueError(
            "Phonon-phonon self-energy must have shape (dim, dim), "
            f"got {out.shape} for dim={dim}."
        )
    return out


def device_green_function_inelastic(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float = 1e-8,
    eta_device: float | None = None,
    kpar: KPar = None,
    device_to_lead_left: DeviceToLeadLike | None = None,
    device_to_lead_right: DeviceToLeadLike | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
    pp_self_energy: PPSelfEnergyModel | None = None,
    max_iter: int = 50,
    mixing: float = 0.5,
    tol: float = 1e-8,
    raise_on_nonconvergence: bool = False,
) -> tuple[Array, Array, Array, Array, InelasticSolveInfo]:
    """Return inelastic ``(G, Sigma_L, Sigma_R, Sigma_pp, info)``."""

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must be in (0, 1].")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")

    dev = _resolve_device(device=device, kpar=kpar)
    vdl_left = _resolve_device_to_lead_coupling(device_to_lead_left, kpar=kpar)
    vdl_right = _resolve_device_to_lead_coupling(device_to_lead_right, kpar=kpar)
    dim = dev.n_layers * dev.dof_per_layer

    a, sigma_l_block, sigma_r_block, idx_l, idx_r = _build_system_matrix_and_contact_sigmas(
        omega=omega,
        device=dev,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        eta_device=eta_device,
        kpar=kpar,
        device_to_lead_left=vdl_left,
        device_to_lead_right=vdl_right,
        contact_left_indices=contact_left_indices,
        contact_right_indices=contact_right_indices,
        surface_gf_method=surface_gf_method,
        omega_scale=omega_scale,
    )
    sigma_l, sigma_r = _embed_self_energies(
        sigma_l_block=sigma_l_block,
        sigma_r_block=sigma_r_block,
        dim=dim,
        idx_l=idx_l,
        idx_r=idx_r,
    )

    sigma_pp = np.zeros((dim, dim), dtype=np.complex128)
    eye_dense = np.eye(dim, dtype=np.complex128)

    if pp_self_energy is None:
        g = splu(a).solve(eye_dense)
        info = InelasticSolveInfo(converged=True, iterations=1, residual=0.0)
        return g, sigma_l, sigma_r, sigma_pp, info

    converged = False
    residual = float("inf")
    g = np.zeros((dim, dim), dtype=np.complex128)
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        a_eff = (a - csc_matrix(sigma_pp)).tocsc()
        g = splu(a_eff).solve(eye_dense)

        sigma_new = _validate_sigma_shape(pp_self_energy(float(omega), g, it), dim)
        sigma_next = mixing * sigma_new + (1.0 - mixing) * sigma_pp
        residual = float(np.linalg.norm(sigma_next - sigma_pp) / max(np.linalg.norm(sigma_next), 1e-30))
        sigma_pp = sigma_next

        if residual <= tol:
            converged = True
            a_eff = (a - csc_matrix(sigma_pp)).tocsc()
            g = splu(a_eff).solve(eye_dense)
            break

    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "Inelastic self-consistent solve did not converge within max_iter "
            f"(iterations={iterations}, residual={residual:.3e}, tol={tol:.3e})."
        )
    if not converged:
        a_eff = (a - csc_matrix(sigma_pp)).tocsc()
        g = splu(a_eff).solve(eye_dense)

    info = InelasticSolveInfo(converged=converged, iterations=iterations, residual=residual)
    return g, sigma_l, sigma_r, sigma_pp, info


def transmission_inelastic(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float = 1e-8,
    eta_device: float | None = None,
    kpar: KPar = None,
    device_to_lead_left: DeviceToLeadLike | None = None,
    device_to_lead_right: DeviceToLeadLike | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
    pp_self_energy: PPSelfEnergyModel | None = None,
    max_iter: int = 50,
    mixing: float = 0.5,
    tol: float = 1e-8,
    raise_on_nonconvergence: bool = False,
) -> tuple[float, dict[str, object]]:
    """Return inelastic transmission and SCF metadata."""

    g, sigma_l, sigma_r, sigma_pp, info = device_green_function_inelastic(
        omega=omega,
        device=device,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        eta_device=eta_device,
        kpar=kpar,
        device_to_lead_left=device_to_lead_left,
        device_to_lead_right=device_to_lead_right,
        contact_left_indices=contact_left_indices,
        contact_right_indices=contact_right_indices,
        surface_gf_method=surface_gf_method,
        omega_scale=omega_scale,
        pp_self_energy=pp_self_energy,
        max_iter=max_iter,
        mixing=mixing,
        tol=tol,
        raise_on_nonconvergence=raise_on_nonconvergence,
    )

    gamma_l = _broadening(sigma_l)
    gamma_r = _broadening(sigma_r)
    tval = np.trace(gamma_l @ g @ gamma_r @ g.conj().T)
    out = float(np.real_if_close(tval).real)
    meta = info.as_dict()
    meta["sigma_pp_norm"] = float(np.linalg.norm(sigma_pp))
    return out, meta
