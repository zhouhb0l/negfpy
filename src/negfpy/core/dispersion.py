"""Phonon dispersion utilities for periodic leads."""

from __future__ import annotations

from typing import Iterable

import numpy as np

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


def _to_1d_array(values: Iterable[float], name: str) -> Array:
    arr = np.asarray(list(values), dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty 1D iterable.")
    return arr


def lead_dynamical_matrix(
    lead: LeadLike,
    kx: float,
    kpar: KPar = None,
    *,
    hermitize: bool = True,
) -> Array:
    """Return D(kx, k_parallel) for a periodic lead."""

    blocks = _resolve_lead_blocks(lead=lead, kpar=kpar)
    phase = np.exp(1j * float(kx))
    d10 = blocks.d10 if blocks.d10 is not None else blocks.d01.conj().T
    dmat = blocks.d00 + blocks.d01 * phase + d10 * phase.conjugate()
    if hermitize:
        dmat = 0.5 * (dmat + dmat.conj().T)
    return dmat


def lead_phonon_dispersion_3d(
    lead: LeadLike,
    kx_points: Iterable[float],
    ky_points: Iterable[float],
    kz_points: Iterable[float],
    *,
    negative_tolerance: float = 1e-10,
    allow_unstable: bool = False,
) -> Array:
    """Return omega(kx, ky, kz, mode) for an x-transport 3D periodic lead."""

    if negative_tolerance < 0.0:
        raise ValueError("negative_tolerance must be non-negative.")

    kx = _to_1d_array(kx_points, "kx_points")
    ky = _to_1d_array(ky_points, "ky_points")
    kz = _to_1d_array(kz_points, "kz_points")

    ref = _resolve_lead_blocks(lead=lead, kpar=(float(ky[0]), float(kz[0])))
    nmode = int(ref.d00.shape[0])
    omega = np.zeros((kx.size, ky.size, kz.size, nmode), dtype=float)

    min_omega2 = np.inf
    for iy, ky_val in enumerate(ky):
        for iz, kz_val in enumerate(kz):
            blocks = _resolve_lead_blocks(lead=lead, kpar=(float(ky_val), float(kz_val)))
            d00 = 0.5 * (blocks.d00 + blocks.d00.conj().T)
            d01 = np.asarray(blocks.d01, dtype=np.complex128)
            d10 = (
                np.asarray(blocks.d10, dtype=np.complex128)
                if blocks.d10 is not None
                else d01.conj().T
            )
            for ix, kx_val in enumerate(kx):
                phase = np.exp(1j * float(kx_val))
                dmat = d00 + d01 * phase + d10 * phase.conjugate()
                vals = np.linalg.eigvalsh(dmat)
                min_omega2 = min(min_omega2, float(np.min(vals)))
                vals = np.where((vals < 0.0) & (vals >= -negative_tolerance), 0.0, vals)
                omega[ix, iy, iz, :] = np.sqrt(np.clip(vals.real, 0.0, None))

    if min_omega2 < -negative_tolerance and not allow_unstable:
        raise ValueError(
            "Lead dispersion contains unstable modes with omega^2 < 0. "
            f"Minimum omega^2 = {min_omega2:.6e}. "
            "Pass allow_unstable=True to inspect clipped frequencies."
        )
    return omega


def leads_phonon_dispersion_3d(
    lead_left: LeadLike,
    lead_right: LeadLike,
    kx_points: Iterable[float],
    ky_points: Iterable[float],
    kz_points: Iterable[float],
    *,
    negative_tolerance: float = 1e-10,
    allow_unstable: bool = False,
) -> dict[str, Array]:
    """Return 3D phonon dispersion for left and right leads."""

    return {
        "left": lead_phonon_dispersion_3d(
            lead=lead_left,
            kx_points=kx_points,
            ky_points=ky_points,
            kz_points=kz_points,
            negative_tolerance=negative_tolerance,
            allow_unstable=allow_unstable,
        ),
        "right": lead_phonon_dispersion_3d(
            lead=lead_right,
            kx_points=kx_points,
            ky_points=ky_points,
            kz_points=kz_points,
            negative_tolerance=negative_tolerance,
            allow_unstable=allow_unstable,
        ),
    }
