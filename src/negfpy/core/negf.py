"""Block-based phonon NEGF calculations."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, eye, lil_matrix
from scipy.sparse.linalg import splu

from .surface_gf import surface_gf_sancho_rubio
from .types import Device1D, DeviceKSpace, KPar, LeadBlocks, LeadKSpace


Array = np.ndarray
LeadLike = LeadBlocks | LeadKSpace
DeviceLike = Device1D | DeviceKSpace
ContactIndices = slice | list[int] | tuple[int, ...] | Array | None


def _resolve_lead_blocks(lead: LeadLike, kpar: KPar) -> LeadBlocks:
    if isinstance(lead, LeadBlocks):
        return lead
    d00, d01 = lead.blocks(kpar=kpar)
    return LeadBlocks(d00=d00, d01=d01)


def _resolve_device(device: DeviceLike, kpar: KPar) -> Device1D:
    if isinstance(device, Device1D):
        return device
    return device.device(kpar=kpar)


def _assemble_device_matrix_sparse(device: Device1D) -> csc_matrix:
    n = device.n_layers
    npl = device.dof_per_layer
    dim = n * npl
    dmat = lil_matrix((dim, dim), dtype=np.complex128)

    for i, block in enumerate(device.onsite_blocks):
        sl = slice(i * npl, (i + 1) * npl)
        dmat[sl, sl] = block

    for i, block in enumerate(device.coupling_blocks):
        sli = slice(i * npl, (i + 1) * npl)
        slj = slice((i + 1) * npl, (i + 2) * npl)
        dmat[sli, slj] = block
        dmat[slj, sli] = block.conj().T

    return dmat.tocsc()


def _self_energy_contact(
    omega: float,
    lead: LeadBlocks,
    device_to_lead: Array,
    eta: float,
) -> Array:
    gsurf = surface_gf_sancho_rubio(omega=omega, d00=lead.d00, d01=lead.d01, eta=eta)
    vdl = np.asarray(device_to_lead, dtype=np.complex128)
    if vdl.ndim != 2:
        raise ValueError("device_to_lead coupling must be a 2D array.")
    nlead = lead.d00.shape[0]
    if vdl.shape[1] != nlead:
        raise ValueError(
            "device_to_lead coupling has incompatible lead dimension "
            f"(got {vdl.shape[1]}, expected {nlead})."
        )
    return vdl @ gsurf @ vdl.conj().T


def _embed_self_energies(
    sigma_l_block: Array,
    sigma_r_block: Array,
    dim: int,
    idx_l: Array,
    idx_r: Array,
) -> tuple[Array, Array]:
    sigma_l = np.zeros((dim, dim), dtype=np.complex128)
    sigma_r = np.zeros((dim, dim), dtype=np.complex128)
    sigma_l[np.ix_(idx_l, idx_l)] = sigma_l_block
    sigma_r[np.ix_(idx_r, idx_r)] = sigma_r_block

    return sigma_l, sigma_r


def _broadening(sigma: Array) -> Array:
    return 1j * (sigma - sigma.conj().T)


def _normalize_contact_indices(
    indices: ContactIndices,
    *,
    dim: int,
    default: Array,
) -> Array:
    if indices is None:
        idx = np.asarray(default, dtype=int).ravel()
    elif isinstance(indices, slice):
        idx = np.arange(dim, dtype=int)[indices]
    else:
        idx = np.asarray(indices, dtype=int).ravel()

    if idx.size == 0:
        raise ValueError("Contact indices must be non-empty.")
    if np.any(idx < 0) or np.any(idx >= dim):
        raise ValueError("Contact indices out of device range.")
    if np.unique(idx).size != idx.size:
        raise ValueError("Contact indices must be unique.")
    return idx


def _build_system_matrix_and_contact_sigmas(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
) -> tuple[csc_matrix, Array, Array, Array, Array]:
    dev = _resolve_device(device=device, kpar=kpar)
    left = _resolve_lead_blocks(lead=lead_left, kpar=kpar)
    right = _resolve_lead_blocks(lead=lead_right, kpar=kpar)

    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    if device_to_lead_left is None:
        # Backward-compatible default for periodic matched contacts.
        vdl_left = left.d01.conj().T
    else:
        vdl_left = np.asarray(device_to_lead_left, dtype=np.complex128)
    if device_to_lead_right is None:
        # Backward-compatible default for periodic matched contacts.
        vdl_right = right.d01
    else:
        vdl_right = np.asarray(device_to_lead_right, dtype=np.complex128)

    ncl = vdl_left.shape[0]
    ncr = vdl_right.shape[0]
    idx_l = _normalize_contact_indices(
        contact_left_indices,
        dim=dim,
        default=np.arange(ncl, dtype=int),
    )
    idx_r = _normalize_contact_indices(
        contact_right_indices,
        dim=dim,
        default=np.arange(dim - ncr, dim, dtype=int),
    )
    if idx_l.size != ncl:
        raise ValueError(
            "Left contact index count must match device_to_lead_left rows "
            f"(got {idx_l.size}, expected {ncl})."
        )
    if idx_r.size != ncr:
        raise ValueError(
            "Right contact index count must match device_to_lead_right rows "
            f"(got {idx_r.size}, expected {ncr})."
        )

    dmat = _assemble_device_matrix_sparse(dev)
    sigma_l_block = _self_energy_contact(omega=omega, lead=left, device_to_lead=vdl_left, eta=eta)
    sigma_r_block = _self_energy_contact(omega=omega, lead=right, device_to_lead=vdl_right, eta=eta)

    z = (omega + 1j * eta) ** 2
    a = (z * eye(dim, dtype=np.complex128, format="csc") - dmat).tolil()
    a[np.ix_(idx_l, idx_l)] = a[np.ix_(idx_l, idx_l)] - sigma_l_block
    a[np.ix_(idx_r, idx_r)] = a[np.ix_(idx_r, idx_r)] - sigma_r_block

    return a.tocsc(), sigma_l_block, sigma_r_block, idx_l, idx_r


def device_green_function(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float = 1e-8,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
) -> tuple[Array, Array, Array]:
    """Return (G, Sigma_L, Sigma_R) for the finite device."""

    dev = _resolve_device(device=device, kpar=kpar)
    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    a, sigma_l_block, sigma_r_block, idx_l, idx_r = _build_system_matrix_and_contact_sigmas(
        omega=omega,
        device=dev,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        kpar=kpar,
        device_to_lead_left=device_to_lead_left,
        device_to_lead_right=device_to_lead_right,
        contact_left_indices=contact_left_indices,
        contact_right_indices=contact_right_indices,
    )
    sigma_l, sigma_r = _embed_self_energies(
        sigma_l_block=sigma_l_block,
        sigma_r_block=sigma_r_block,
        dim=dim,
        idx_l=idx_l,
        idx_r=idx_r,
    )

    # Kept for API compatibility; transmission() avoids constructing full G.
    lu = splu(a)
    g = lu.solve(np.eye(dim, dtype=np.complex128))
    return g, sigma_l, sigma_r


def transmission(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float = 1e-8,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
) -> float:
    """Return coherent phonon transmission T(omega)."""

    dev = _resolve_device(device=device, kpar=kpar)
    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    a, sigma_l_block, sigma_r_block, idx_l, idx_r = _build_system_matrix_and_contact_sigmas(
        omega=omega,
        device=dev,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        kpar=kpar,
        device_to_lead_left=device_to_lead_left,
        device_to_lead_right=device_to_lead_right,
        contact_left_indices=contact_left_indices,
        contact_right_indices=contact_right_indices,
    )

    gamma_l_block = _broadening(sigma_l_block)
    gamma_r_block = _broadening(sigma_r_block)
    ncr = idx_r.size

    # Solve for right-contact columns only: X = G[:, R].
    rhs_r = np.zeros((dim, ncr), dtype=np.complex128)
    rhs_r[idx_r, :] = np.eye(ncr, dtype=np.complex128)
    g_cols_r = splu(a).solve(rhs_r)
    g_lr = g_cols_r[idx_l, :]

    tval = np.trace(gamma_l_block @ g_lr @ gamma_r_block @ g_lr.conj().T)
    return float(np.real_if_close(tval).real)


def transmission_kavg(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta: float = 1e-8,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
) -> float:
    """Return k_parallel-averaged transmission over supplied k-point list."""

    if len(kpoints) == 0:
        raise ValueError("kpoints must contain at least one k-point.")
    vals = [
        transmission(
            omega=omega,
            device=device,
            lead_left=lead_left,
            lead_right=lead_right,
            eta=eta,
            kpar=kpar,
            device_to_lead_left=device_to_lead_left,
            device_to_lead_right=device_to_lead_right,
            contact_left_indices=contact_left_indices,
            contact_right_indices=contact_right_indices,
        )
        for kpar in kpoints
    ]
    return float(np.mean(vals))
