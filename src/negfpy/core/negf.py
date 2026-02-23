"""Block-based phonon NEGF calculations."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, eye, lil_matrix
from scipy.sparse.linalg import splu

from .surface_gf import surface_gf
from .types import Device1D, DeviceKSpace, KPar, LeadBlocks, LeadKSpace


Array = np.ndarray
LeadLike = LeadBlocks | LeadKSpace
DeviceLike = Device1D | DeviceKSpace
ContactIndices = slice | list[int] | tuple[int, ...] | Array | None


def _resolve_lead_blocks(lead: LeadLike, kpar: KPar) -> LeadBlocks:
    if isinstance(lead, LeadBlocks):
        return lead
    blocks = lead.blocks(kpar=kpar)
    if len(blocks) == 2:
        d00, d01 = blocks
        return LeadBlocks(d00=d00, d01=d01)
    if len(blocks) == 3:
        d00, d01, d10 = blocks
        return LeadBlocks(d00=d00, d01=d01, d10=d10)
    raise ValueError("LeadKSpace blocks_builder must return (d00,d01) or (d00,d01,d10).")


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
    *,
    surface_gf_method: str,
) -> Array:
    gsurf = surface_gf(
        omega=omega,
        d00=lead.d00,
        d01=lead.d01,
        d10=lead.d10,
        eta=eta,
        method=surface_gf_method,
    )
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
    eta_device: float | None = None,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
) -> tuple[csc_matrix, Array, Array, Array, Array]:
    dev = _resolve_device(device=device, kpar=kpar)
    left = _resolve_lead_blocks(lead=lead_left, kpar=kpar)
    right = _resolve_lead_blocks(lead=lead_right, kpar=kpar)

    npl = dev.dof_per_layer
    n_layers = dev.n_layers
    dim = n_layers * npl
    eta_dev = float(eta if eta_device is None else eta_device)
    if eta_dev < 0.0:
        raise ValueError("eta_device must be non-negative.")
    eta_lead = float(eta)
    omega_eff = float(omega)

    if omega_scale is not None:
        scale = float(omega_scale)
        if scale <= 0.0:
            raise ValueError("omega_scale must be positive when provided.")
        scale2 = scale * scale
        omega_eff = omega_eff / scale
        eta_lead = eta_lead / scale
        eta_dev = eta_dev / scale
        dev = Device1D(
            onsite_blocks=[np.asarray(b, dtype=np.complex128) / scale2 for b in dev.onsite_blocks],
            coupling_blocks=[np.asarray(b, dtype=np.complex128) / scale2 for b in dev.coupling_blocks],
        )
        left = LeadBlocks(
            d00=np.asarray(left.d00, dtype=np.complex128) / scale2,
            d01=np.asarray(left.d01, dtype=np.complex128) / scale2,
            d10=None if left.d10 is None else np.asarray(left.d10, dtype=np.complex128) / scale2,
        )
        right = LeadBlocks(
            d00=np.asarray(right.d00, dtype=np.complex128) / scale2,
            d01=np.asarray(right.d01, dtype=np.complex128) / scale2,
            d10=None if right.d10 is None else np.asarray(right.d10, dtype=np.complex128) / scale2,
        )

    if device_to_lead_left is None:
        # Backward-compatible default for periodic matched contacts.
        vdl_left = left.d10 if left.d10 is not None else left.d01.conj().T
    else:
        vdl_left = np.asarray(device_to_lead_left, dtype=np.complex128)
    if device_to_lead_right is None:
        # Backward-compatible default for periodic matched contacts.
        vdl_right = right.d01
    else:
        vdl_right = np.asarray(device_to_lead_right, dtype=np.complex128)
    if omega_scale is not None:
        vdl_left = vdl_left / (float(omega_scale) * float(omega_scale))
        vdl_right = vdl_right / (float(omega_scale) * float(omega_scale))

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
    sigma_l_block = _self_energy_contact(
        omega=omega_eff,
        lead=left,
        device_to_lead=vdl_left,
        eta=eta_lead,
        surface_gf_method=surface_gf_method,
    )
    sigma_r_block = _self_energy_contact(
        omega=omega_eff,
        lead=right,
        device_to_lead=vdl_right,
        eta=eta_lead,
        surface_gf_method=surface_gf_method,
    )

    z = (omega_eff + 1j * eta_dev) ** 2
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
    eta_device: float | None = None,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
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
        eta_device=eta_device,
        kpar=kpar,
        device_to_lead_left=device_to_lead_left,
        device_to_lead_right=device_to_lead_right,
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
    eta_device: float | None = None,
    kpar: KPar = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
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
        eta_device=eta_device,
        kpar=kpar,
        device_to_lead_left=device_to_lead_left,
        device_to_lead_right=device_to_lead_right,
        contact_left_indices=contact_left_indices,
        contact_right_indices=contact_right_indices,
        surface_gf_method=surface_gf_method,
        omega_scale=omega_scale,
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
    eta_device: float | None = None,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
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
            eta_device=eta_device,
            kpar=kpar,
            device_to_lead_left=device_to_lead_left,
            device_to_lead_right=device_to_lead_right,
            contact_left_indices=contact_left_indices,
            contact_right_indices=contact_right_indices,
            surface_gf_method=surface_gf_method,
            omega_scale=omega_scale,
        )
        for kpar in kpoints
    ]
    return float(np.mean(vals))


def transmission_kavg_adaptive(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta_values: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5),
    eta_device: float | None = None,
    min_success_fraction: float = 0.0,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
    nonnegative_tolerance: float = 0.0,
    max_channel_factor: float | None = None,
    collect_rejected: bool = False,
) -> tuple[float, dict[str, object]]:
    """Return adaptive k-averaged transmission and convergence statistics.

    For each k-point, tries ``eta_values`` in order until transmission converges.
    """

    if len(kpoints) == 0:
        raise ValueError("kpoints must contain at least one k-point.")
    if len(eta_values) == 0:
        raise ValueError("eta_values must contain at least one eta value.")
    if not (0.0 <= min_success_fraction <= 1.0):
        raise ValueError("min_success_fraction must be in [0, 1].")
    if nonnegative_tolerance < 0.0:
        raise ValueError("nonnegative_tolerance must be non-negative.")
    if max_channel_factor is not None and max_channel_factor <= 0.0:
        raise ValueError("max_channel_factor must be positive when provided.")

    vals: list[float] = []
    used_etas: list[float] = []
    failed_kpoints = 0
    rejected: list[dict[str, object]] = []

    for kpar in kpoints:
        max_t_allowed: float | None = None
        if max_channel_factor is not None:
            lead_blk = _resolve_lead_blocks(lead=lead_left, kpar=kpar)
            max_t_allowed = float(max_channel_factor) * float(lead_blk.d00.shape[0])
        converged = False
        for eta in eta_values:
            try:
                tval = transmission(
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
                )
            except Exception:
                if collect_rejected:
                    rejected.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": None,
                            "reason": "exception",
                        }
                    )
                continue
            if np.isfinite(tval):
                if tval < -float(nonnegative_tolerance):
                    if collect_rejected:
                        rejected.append(
                            {
                                "omega": float(omega),
                                "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                                "eta": float(eta),
                                "t": float(tval),
                                "reason": "negative",
                            }
                        )
                    continue
                if max_t_allowed is not None and tval > max_t_allowed:
                    if collect_rejected:
                        rejected.append(
                            {
                                "omega": float(omega),
                                "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                                "eta": float(eta),
                                "t": float(tval),
                                "reason": "over_channel_limit",
                            }
                        )
                    continue
                vals.append(float(max(tval, 0.0)))
                used_etas.append(float(eta))
                converged = True
                break
            else:
                if collect_rejected:
                    rejected.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": float(tval),
                            "reason": "non_finite",
                        }
                    )
        if not converged:
            failed_kpoints += 1

    n_total = len(kpoints)
    n_success = len(vals)
    success_fraction = float(n_success / n_total)
    if n_success == 0:
        raise RuntimeError("No k-point converged for adaptive k-averaged transmission.")
    if success_fraction < min_success_fraction:
        raise RuntimeError(
            "Adaptive k-averaged transmission failed success-fraction threshold "
            f"(got {success_fraction:.3f}, required >= {min_success_fraction:.3f})."
        )

    eta_histogram: dict[float, int] = {}
    for eta in eta_values:
        eta_histogram[float(eta)] = int(sum(1 for e in used_etas if e == float(eta)))

    info: dict[str, object] = {
        "n_total": int(n_total),
        "n_success": int(n_success),
        "n_failed": int(failed_kpoints),
        "success_fraction": success_fraction,
        "eta_histogram": eta_histogram,
    }
    if collect_rejected:
        info["n_rejected"] = int(len(rejected))
        info["rejected"] = rejected
    return float(np.mean(vals)), info


def transmission_kavg_adaptive_global_eta(
    omega: float,
    device: DeviceLike,
    lead_left: LeadLike,
    lead_right: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta_values: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5),
    eta_device: float | None = None,
    min_success_fraction: float = 0.0,
    device_to_lead_left: Array | None = None,
    device_to_lead_right: Array | None = None,
    contact_left_indices: ContactIndices = None,
    contact_right_indices: ContactIndices = None,
    surface_gf_method: str = "sancho_rubio",
    omega_scale: float | None = None,
    nonnegative_tolerance: float = 0.0,
    max_channel_factor: float | None = None,
    collect_rejected: bool = False,
) -> tuple[float, dict[str, object]]:
    """Return adaptive k-averaged transmission using one eta per omega.

    Tries eta values in order; for each eta, computes transmissions on all
    k-points and checks quality filters. The first eta meeting the required
    success fraction is accepted, so all accepted k-points for this omega
    share the same broadening.
    """

    if len(kpoints) == 0:
        raise ValueError("kpoints must contain at least one k-point.")
    if len(eta_values) == 0:
        raise ValueError("eta_values must contain at least one eta value.")
    if not (0.0 <= min_success_fraction <= 1.0):
        raise ValueError("min_success_fraction must be in [0, 1].")
    if nonnegative_tolerance < 0.0:
        raise ValueError("nonnegative_tolerance must be non-negative.")
    if max_channel_factor is not None and max_channel_factor <= 0.0:
        raise ValueError("max_channel_factor must be positive when provided.")

    n_total = len(kpoints)
    rejected_all: list[dict[str, object]] = []

    for eta in eta_values:
        vals: list[float] = []
        rejected_eta: list[dict[str, object]] = []

        for kpar in kpoints:
            max_t_allowed: float | None = None
            if max_channel_factor is not None:
                lead_blk = _resolve_lead_blocks(lead=lead_left, kpar=kpar)
                max_t_allowed = float(max_channel_factor) * float(lead_blk.d00.shape[0])
            try:
                tval = transmission(
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
                )
            except Exception:
                if collect_rejected:
                    rejected_eta.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": None,
                            "reason": "exception",
                        }
                    )
                continue

            if not np.isfinite(tval):
                if collect_rejected:
                    rejected_eta.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": float(tval),
                            "reason": "non_finite",
                        }
                    )
                continue

            if tval < -float(nonnegative_tolerance):
                if collect_rejected:
                    rejected_eta.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": float(tval),
                            "reason": "negative",
                        }
                    )
                continue

            if max_t_allowed is not None and tval > max_t_allowed:
                if collect_rejected:
                    rejected_eta.append(
                        {
                            "omega": float(omega),
                            "kpar": tuple(float(x) for x in kpar) if kpar is not None else (),
                            "eta": float(eta),
                            "t": float(tval),
                            "reason": "over_channel_limit",
                        }
                    )
                continue

            vals.append(float(max(tval, 0.0)))

        n_success = len(vals)
        success_fraction = float(n_success / n_total)
        if collect_rejected:
            rejected_all.extend(rejected_eta)

        if n_success > 0 and success_fraction >= min_success_fraction:
            info: dict[str, object] = {
                "n_total": int(n_total),
                "n_success": int(n_success),
                "n_failed": int(n_total - n_success),
                "success_fraction": success_fraction,
                "eta_selected": float(eta),
                "eta_histogram": {float(e): (int(n_success) if float(e) == float(eta) else 0) for e in eta_values},
            }
            if collect_rejected:
                info["n_rejected"] = int(len(rejected_all))
                info["rejected"] = rejected_all
            return float(np.mean(vals)), info

    raise RuntimeError(
        "No eta value satisfied global adaptive k-averaging quality threshold "
        f"(required success_fraction >= {min_success_fraction:.3f})."
    )
