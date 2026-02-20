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


def _self_energy_left(omega: float, lead: LeadBlocks, eta: float) -> Array:
    gsurf = surface_gf_sancho_rubio(omega=omega, d00=lead.d00, d01=lead.d01, eta=eta)
    tau = lead.d01
    return tau.conj().T @ gsurf @ tau


def _self_energy_right(omega: float, lead: LeadBlocks, eta: float) -> Array:
    gsurf = surface_gf_sancho_rubio(omega=omega, d00=lead.d00, d01=lead.d01, eta=eta)
    tau = lead.d01
    return tau @ gsurf @ tau.conj().T


def _embed_self_energies(
    sigma_l_block: Array,
    sigma_r_block: Array,
    n_layers: int,
    npl: int,
) -> tuple[Array, Array]:
    dim = n_layers * npl
    sigma_l = np.zeros((dim, dim), dtype=np.complex128)
    sigma_r = np.zeros((dim, dim), dtype=np.complex128)

    first = slice(0, npl)
    last = slice((n_layers - 1) * npl, n_layers * npl)
    sigma_l[first, first] = sigma_l_block
    sigma_r[last, last] = sigma_r_block

    return sigma_l, sigma_r


def _broadening(sigma: Array) -> Array:
    return 1j * (sigma - sigma.conj().T)


def _contact_slices(n_layers: int, npl: int) -> tuple[slice, slice]:
    first = slice(0, npl)
    last = slice((n_layers - 1) * npl, n_layers * npl)
    return first, last


def _build_system_matrix_and_contact_sigmas(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float,
    kpar: KPar = None,
) -> tuple[csc_matrix, Array, Array]:
    dev = _resolve_device(device=device, kpar=kpar)
    left = _resolve_lead_blocks(lead=lead_left, kpar=kpar)
    right = _resolve_lead_blocks(lead=lead_right, kpar=kpar)

    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    dmat = _assemble_device_matrix_sparse(dev)
    sigma_l_block = _self_energy_left(omega=omega, lead=left, eta=eta)
    sigma_r_block = _self_energy_right(omega=omega, lead=right, eta=eta)

    z = (omega + 1j * eta) ** 2
    a = (z * eye(dim, dtype=np.complex128, format="csc") - dmat).tolil()
    first, last = _contact_slices(n_layers=n_layers, npl=npl)
    a[first, first] = a[first, first] - sigma_l_block
    a[last, last] = a[last, last] - sigma_r_block

    return a.tocsc(), sigma_l_block, sigma_r_block


def device_green_function(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    eta: float = 1e-8,
    kpar: KPar = None,
) -> tuple[Array, Array, Array]:
    """Return (G, Sigma_L, Sigma_R) for the finite device."""

    dev = _resolve_device(device=device, kpar=kpar)
    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    a, sigma_l_block, sigma_r_block = _build_system_matrix_and_contact_sigmas(
        omega=omega,
        device=dev,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        kpar=kpar,
    )
    sigma_l, sigma_r = _embed_self_energies(
        sigma_l_block=sigma_l_block,
        sigma_r_block=sigma_r_block,
        n_layers=n_layers,
        npl=npl,
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
) -> float:
    """Return coherent phonon transmission T(omega)."""

    dev = _resolve_device(device=device, kpar=kpar)
    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl

    a, sigma_l_block, sigma_r_block = _build_system_matrix_and_contact_sigmas(
        omega=omega,
        device=dev,
        lead_left=lead_left,
        lead_right=lead_right,
        eta=eta,
        kpar=kpar,
    )

    gamma_l_block = _broadening(sigma_l_block)
    gamma_r_block = _broadening(sigma_r_block)
    first, last = _contact_slices(n_layers=n_layers, npl=npl)

    # Solve for right-contact columns only: X = G[:, R].
    rhs_r = np.zeros((dim, npl), dtype=np.complex128)
    rhs_r[last, :] = np.eye(npl, dtype=np.complex128)
    g_cols_r = splu(a).solve(rhs_r)
    g_lr = g_cols_r[first, :]

    tval = np.trace(gamma_l_block @ g_lr @ gamma_r_block @ g_lr.conj().T)
    return float(np.real_if_close(tval).real)


def transmission_kavg(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta: float = 1e-8,
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
        )
        for kpar in kpoints
    ]
    return float(np.mean(vals))
