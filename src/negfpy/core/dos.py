"""Phonon density-of-states utilities for periodic leads."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .dispersion import lead_dynamical_matrix
from .surface_gf import surface_gf
from .types import KPar, LeadBlocks, LeadKSpace


Array = np.ndarray
LeadLike = LeadBlocks | LeadKSpace


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


def lead_surface_dos(
    omega: float,
    lead: LeadLike,
    eta: float = 1e-8,
    kpar: KPar = None,
    *,
    normalize_per_mode: bool = False,
    surface_gf_method: str = "sancho_rubio",
) -> float:
    """Return surface DOS of a semi-infinite lead at (omega, k_parallel).

    Uses the retarded surface Green's function with
    ``DOS = (omega / pi) * Tr[A]``, where ``A = i(G - G^dagger)``.
    """

    if omega < 0.0:
        raise ValueError("omega must be non-negative.")
    blocks = _resolve_lead_blocks(lead=lead, kpar=kpar)
    gsurf = surface_gf(
        omega=omega,
        d00=blocks.d00,
        d01=blocks.d01,
        d10=blocks.d10,
        eta=eta,
        method=surface_gf_method,
    )
    spec = 1j * (gsurf - gsurf.conj().T)
    dos = float((omega / np.pi) * np.trace(spec).real)
    if normalize_per_mode:
        dos /= blocks.d00.shape[0]
    return dos


def lead_surface_dos_kavg(
    omega: float,
    lead: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta: float = 1e-8,
    *,
    normalize_per_mode: bool = False,
    surface_gf_method: str = "sancho_rubio",
) -> float:
    """Return k_parallel-averaged lead surface DOS."""

    if len(kpoints) == 0:
        raise ValueError("kpoints must contain at least one k-point.")
    vals = [
        lead_surface_dos(
            omega=omega,
            lead=lead,
            eta=eta,
            kpar=kpar,
            normalize_per_mode=normalize_per_mode,
            surface_gf_method=surface_gf_method,
        )
        for kpar in kpoints
    ]
    return float(np.mean(vals))


def lead_surface_dos_kavg_adaptive(
    omega: float,
    lead: LeadLike,
    kpoints: list[tuple[float, ...]],
    eta_values: tuple[float, ...] = (1e-8, 1e-7, 1e-6, 1e-5),
    min_success_fraction: float = 0.0,
    *,
    normalize_per_mode: bool = False,
    surface_gf_method: str = "sancho_rubio",
) -> tuple[float, dict[str, object]]:
    """Return adaptive k-averaged lead surface DOS and convergence statistics."""

    if len(kpoints) == 0:
        raise ValueError("kpoints must contain at least one k-point.")
    if len(eta_values) == 0:
        raise ValueError("eta_values must contain at least one eta value.")
    if not (0.0 <= min_success_fraction <= 1.0):
        raise ValueError("min_success_fraction must be in [0, 1].")

    vals: list[float] = []
    used_etas: list[float] = []
    failed_kpoints = 0

    for kpar in kpoints:
        converged = False
        for eta in eta_values:
            try:
                dval = lead_surface_dos(
                    omega=omega,
                    lead=lead,
                    eta=eta,
                    kpar=kpar,
                    normalize_per_mode=normalize_per_mode,
                    surface_gf_method=surface_gf_method,
                )
            except Exception:
                continue
            if np.isfinite(dval):
                vals.append(float(dval))
                used_etas.append(float(eta))
                converged = True
                break
        if not converged:
            failed_kpoints += 1

    n_total = len(kpoints)
    n_success = len(vals)
    success_fraction = float(n_success / n_total)
    if n_success == 0:
        raise RuntimeError("No k-point converged for adaptive k-averaged DOS.")
    if success_fraction < min_success_fraction:
        raise RuntimeError(
            "Adaptive k-averaged DOS failed success-fraction threshold "
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
    return float(np.mean(vals)), info


def lead_surface_dos_spectrum(
    omegas: Iterable[float],
    lead: LeadLike,
    *,
    kpoints: list[tuple[float, ...]] | None = None,
    eta: float = 1e-8,
    eta_values: tuple[float, ...] | None = None,
    min_success_fraction: float = 0.0,
    normalize_per_mode: bool = False,
    surface_gf_method: str = "sancho_rubio",
) -> Array:
    """Return DOS(omega) spectrum for one lead."""

    omega_arr = np.asarray(list(omegas), dtype=float)
    if omega_arr.ndim != 1 or omega_arr.size == 0:
        raise ValueError("omegas must be a non-empty 1D iterable.")

    vals = np.zeros_like(omega_arr, dtype=float)
    if kpoints is None:
        for i, w in enumerate(omega_arr):
            vals[i] = lead_surface_dos(
                omega=float(w),
                lead=lead,
                eta=eta,
                kpar=None,
                normalize_per_mode=normalize_per_mode,
                surface_gf_method=surface_gf_method,
            )
        return vals

    if eta_values is None:
        for i, w in enumerate(omega_arr):
            vals[i] = lead_surface_dos_kavg(
                omega=float(w),
                lead=lead,
                kpoints=kpoints,
                eta=eta,
                normalize_per_mode=normalize_per_mode,
                surface_gf_method=surface_gf_method,
            )
        return vals

    for i, w in enumerate(omega_arr):
        vals[i], _ = lead_surface_dos_kavg_adaptive(
            omega=float(w),
            lead=lead,
            kpoints=kpoints,
            eta_values=eta_values,
            min_success_fraction=min_success_fraction,
            normalize_per_mode=normalize_per_mode,
            surface_gf_method=surface_gf_method,
        )
    return vals


def leads_surface_dos_spectrum(
    omegas: Iterable[float],
    lead_left: LeadLike,
    lead_right: LeadLike,
    *,
    kpoints: list[tuple[float, ...]] | None = None,
    eta: float = 1e-8,
    eta_values: tuple[float, ...] | None = None,
    min_success_fraction: float = 0.0,
    normalize_per_mode: bool = False,
    surface_gf_method: str = "sancho_rubio",
) -> dict[str, Array]:
    """Return DOS spectra for left and right leads."""

    return {
        "left": lead_surface_dos_spectrum(
            omegas=omegas,
            lead=lead_left,
            kpoints=kpoints,
            eta=eta,
            eta_values=eta_values,
            min_success_fraction=min_success_fraction,
            normalize_per_mode=normalize_per_mode,
            surface_gf_method=surface_gf_method,
        ),
        "right": lead_surface_dos_spectrum(
            omegas=omegas,
            lead=lead_right,
            kpoints=kpoints,
            eta=eta,
            eta_values=eta_values,
            min_success_fraction=min_success_fraction,
            normalize_per_mode=normalize_per_mode,
            surface_gf_method=surface_gf_method,
        ),
    }


def _default_kgrid(nk: int, centered: bool) -> Array:
    if nk <= 0:
        raise ValueError("nk must be positive.")
    if centered:
        step = 2.0 * np.pi / nk
        idx = np.arange(nk, dtype=float) - (nk // 2)
        return idx * step
    return -np.pi + (np.arange(nk, dtype=float) + 0.5) * (2.0 * np.pi / nk)


def bulk_phonon_dos_3d(
    lead: LeadLike,
    omegas: Iterable[float],
    *,
    nkx: int,
    nky: int,
    nkz: int,
    sigma: float,
    centered_grid: bool = False,
    normalize_per_mode: bool = True,
    chunk_size: int = 20000,
) -> Array:
    """Return bulk phonon DOS from full 3D periodic dynamical matrix sampling.

    The x/y/z directions are all treated as translationally periodic.
    """

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    omega = np.asarray(list(omegas), dtype=float)
    if omega.ndim != 1 or omega.size == 0:
        raise ValueError("omegas must be a non-empty 1D iterable.")

    kx = _default_kgrid(nkx, centered=centered_grid)
    ky = _default_kgrid(nky, centered=centered_grid)
    kz = _default_kgrid(nkz, centered=centered_grid)

    ref = _resolve_lead_blocks(lead=lead, kpar=(float(ky[0]), float(kz[0])))
    nmode = ref.d00.shape[0]
    eigs: list[float] = []
    for kyv in ky:
        for kzv in kz:
            for kxv in kx:
                dmat = lead_dynamical_matrix(lead=lead, kx=float(kxv), kpar=(float(kyv), float(kzv)), hermitize=True)
                w2 = np.linalg.eigvalsh(dmat)
                w = np.sqrt(np.clip(w2.real, 0.0, None))
                eigs.extend(w.tolist())

    eig = np.asarray(eigs, dtype=float)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    # Gaussian broadening approximation of delta(omega - omega_nk).
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    dos = np.zeros_like(omega, dtype=float)
    n_modes_total = eig.size
    for start in range(0, n_modes_total, chunk_size):
        stop = min(start + chunk_size, n_modes_total)
        e_chunk = eig[start:stop]
        x = (omega[:, None] - e_chunk[None, :]) / sigma
        dos += pref * np.sum(np.exp(-0.5 * x * x), axis=1)
    dos /= float(n_modes_total)
    if normalize_per_mode:
        dos = dos / float(nmode)
    return np.asarray(dos, dtype=float)


def bulk_phonon_dos_3d_from_ifc_terms(
    omegas: Iterable[float],
    masses: Array,
    dof_per_atom: int,
    terms: Iterable[tuple[int, int, int, Array]],
    *,
    nkx: int,
    nky: int,
    nkz: int,
    sigma: float,
    centered_grid: bool = False,
    normalize_per_mode: bool = True,
    chunk_size: int = 20000,
) -> Array:
    """Return bulk phonon DOS from direct 3D IFC Fourier summation."""

    if sigma <= 0.0:
        raise ValueError("sigma must be positive.")
    omega = np.asarray(list(omegas), dtype=float)
    if omega.ndim != 1 or omega.size == 0:
        raise ValueError("omegas must be a non-empty 1D iterable.")
    if dof_per_atom <= 0:
        raise ValueError("dof_per_atom must be positive.")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    masses_arr = np.asarray(masses, dtype=float).ravel()
    if masses_arr.size == 0 or np.any(masses_arr <= 0.0):
        raise ValueError("masses must be a non-empty array with positive entries.")
    ndof = masses_arr.size * int(dof_per_atom)
    minv = np.diag(1.0 / np.sqrt(np.repeat(masses_arr, int(dof_per_atom))))

    term_list = [(int(dx), int(dy), int(dz), np.asarray(block, dtype=np.complex128)) for dx, dy, dz, block in terms]
    if len(term_list) == 0:
        raise ValueError("terms must be non-empty.")
    for _, _, _, block in term_list:
        if block.shape != (ndof, ndof):
            raise ValueError("Each IFC block must have shape (n_atoms*dof_per_atom, n_atoms*dof_per_atom).")

    kx = _default_kgrid(nkx, centered=centered_grid)
    ky = _default_kgrid(nky, centered=centered_grid)
    kz = _default_kgrid(nkz, centered=centered_grid)

    eigs: list[float] = []
    for kyv in ky:
        for kzv in kz:
            for kxv in kx:
                phi = np.zeros((ndof, ndof), dtype=np.complex128)
                for dx, dy, dz, block in term_list:
                    phi += block * np.exp(1j * (kxv * dx + kyv * dy + kzv * dz))
                dmat = minv @ phi @ minv
                dmat = 0.5 * (dmat + dmat.conj().T)
                w2 = np.linalg.eigvalsh(dmat)
                eigs.extend(np.sqrt(np.clip(w2.real, 0.0, None)).tolist())

    eig = np.asarray(eigs, dtype=float)
    pref = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    dos = np.zeros_like(omega, dtype=float)
    n_modes_total = eig.size
    for start in range(0, n_modes_total, chunk_size):
        stop = min(start + chunk_size, n_modes_total)
        e_chunk = eig[start:stop]
        x = (omega[:, None] - e_chunk[None, :]) / sigma
        dos += pref * np.sum(np.exp(-0.5 * x * x), axis=1)
    dos /= float(n_modes_total)
    if normalize_per_mode:
        dos = dos / float(ndof)
    return np.asarray(dos, dtype=float)
