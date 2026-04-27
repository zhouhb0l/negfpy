"""Paper-specific cubic onsite LO benchmark utilities for Wang 0701164 Fig. 5.

This module is intentionally separate from the reusable inelastic solver layer.
It implements the published cubic-onsite toy model used in Fig. 5 of

    J.-S. Wang, J. Wang, N. Zeng,
    "Nonequilibrium Green's function method for thermal transport in junctions"

The current implementation follows the paper's model and leading-order cubic
structure closely, while evaluating conductance through a small finite
temperature difference. This keeps the benchmark path isolated and lets us
verify model/parameter consistency before carrying ideas back into more general
material-ready code.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


Array = np.ndarray
EV_J = 1.602_176_634e-19
HBAR_SI = 1.054_571_817e-34
KB_SI = 1.380_649e-23
AMU_KG = 1.660_539_066_60e-27
ANGSTROM_M = 1.0e-10
EV_CUBIC_TO_SI = EV_J / ((ANGSTROM_M**3) * (AMU_KG ** 1.5))
PAPER_DYNAMIC_SCALE_SI = EV_J / ((ANGSTROM_M**2) * AMU_KG)


def _stable_bose(omega: Array, temperature: float, *, kb_effective: float) -> Array:
    w = np.asarray(omega, dtype=float)
    if temperature <= 0.0:
        return np.zeros_like(w)
    x = w / (float(kb_effective) * float(temperature))
    out = np.zeros_like(x)
    small = np.abs(x) < 1e-6
    regular = (~small) & (x < 700.0)
    out[small] = 1.0 / np.maximum(x[small], 1e-12)
    out[regular] = 1.0 / np.expm1(x[regular])
    return out


def _stable_bose_dfdt(omega: Array, temperature: float, *, kb_effective: float) -> Array:
    w = np.asarray(omega, dtype=float)
    if temperature <= 0.0:
        return np.zeros_like(w)
    x = w / (float(kb_effective) * float(temperature))
    occ = _stable_bose(w, temperature, kb_effective=kb_effective)
    out = np.zeros_like(w)
    mask = np.abs(x) < 700.0
    out[mask] = occ[mask] * (occ[mask] + 1.0) * x[mask] / float(temperature)
    return out


def wang0701164_kb_effective_si() -> float:
    return KB_SI / HBAR_SI


def wang0701164_paper_unit_factors() -> dict[str, float]:
    """Return physical conversion factors for Wang's native Fig. 5 numeric units.

    The paper quotes harmonic constants in eV/(A^2 u) and cubic couplings in
    eV/(A^3 u^(3/2)), while the transport formulas still use physical Bose
    factors and physical heat current units. If we keep the benchmark dynamics
    in the paper's native numeric units, then

        omega_phys = alpha * omega_paper,

    with alpha = sqrt(eV / (A^2 u)) in SI. This gives the effective prefactors

        kB_eff_paper = k_B / (hbar * alpha),
        hbar_eff_paper = hbar * alpha^2.
    """

    alpha = float(np.sqrt(PAPER_DYNAMIC_SCALE_SI))
    return {
        "omega_scale_si": alpha,
        "kb_effective_paper": KB_SI / (HBAR_SI * alpha),
        "hbar_effective_paper": HBAR_SI * (alpha ** 2),
    }


@dataclass(frozen=True)
class CubicOnsiteChainParams:
    """One-dimensional cubic onsite chain in Wang 0701164 notation."""

    spring: float
    onsite_spring: float
    cubic: float
    n_layers: int

    def __post_init__(self) -> None:
        if self.spring <= 0.0:
            raise ValueError("spring must be positive.")
        if self.onsite_spring < 0.0:
            raise ValueError("onsite_spring must be non-negative.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")


def wang0701164_fig5_spec(*, unit_system: str = "paper") -> dict[str, object]:
    """Return the published Fig. 5 benchmark specification."""

    if unit_system not in {"paper", "si"}:
        raise ValueError("unit_system must be 'paper' or 'si'.")
    return {
        "paper": "0701164v1",
        "target_figure": "Fig. 5",
        "model": "one-dimensional cubic onsite model",
        "approximation": "leading order cubic",
        "parameters": {
            "spring": 0.625,
            "onsite_spring": 0.0625,
            "n_layers": 5,
            "cubic_values": [0.0, 0.2, 0.5, 0.7, 1.0, 2.0],
            "spring_units": "eV/(A^2 u)" if unit_system == "paper" else "s^-2",
            "cubic_units": "eV/(A^3 u^(3/2))" if unit_system == "paper" else "s^-2/(m sqrt(kg))",
        },
        "published_axis": "conductance (10^-9 W/K)",
    }


def _to_dynamic_si_force(value_ev_per_a2_u: float) -> float:
    return float(value_ev_per_a2_u) * EV_J / ((ANGSTROM_M**2) * AMU_KG)


def _to_dynamic_si_cubic(value_ev_per_a3_u32: float) -> float:
    return float(value_ev_per_a3_u32) * EV_CUBIC_TO_SI


def wang0701164_fig5_params(*, cubic: float, unit_system: str = "paper") -> CubicOnsiteChainParams:
    """Return Fig. 5 parameters for one cubic coupling."""

    if unit_system == "paper":
        return CubicOnsiteChainParams(spring=0.625, onsite_spring=0.0625, cubic=float(cubic), n_layers=5)
    if unit_system == "si":
        return CubicOnsiteChainParams(
            spring=_to_dynamic_si_force(0.625),
            onsite_spring=_to_dynamic_si_force(0.0625),
            cubic=_to_dynamic_si_cubic(float(cubic)),
            n_layers=5,
        )
    raise ValueError("unit_system must be 'paper' or 'si'.")


def _lambda_roots(omega: float, *, spring: float, onsite_spring: float, eta: float) -> tuple[complex, complex]:
    om = (float(omega) + 1j * float(eta)) ** 2 - 2.0 * float(spring) - float(onsite_spring)
    disc = np.sqrt(om * om - 4.0 * (float(spring) ** 2) + 0j)
    lam_a = (-om + disc) / (2.0 * float(spring))
    lam_b = (-om - disc) / (2.0 * float(spring))
    if abs(lam_a) <= abs(lam_b):
        lam1, lam2 = lam_a, lam_b
    else:
        lam1, lam2 = lam_b, lam_a
    return lam1, lam2


def _theta_band(omega: float, *, spring: float, onsite_spring: float) -> float:
    w2 = float(omega) ** 2
    return 1.0 if (float(onsite_spring) < w2 < (4.0 * float(spring) + float(onsite_spring))) else 0.0


def _band_edge(params: CubicOnsiteChainParams) -> float:
    return float(np.sqrt(4.0 * float(params.spring) + float(params.onsite_spring)))


def cubic_onsite_g0_retarded(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    eta: float,
) -> Array:
    """Return analytic retarded G0 for the uniform harmonic chain restricted to the center."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    out = np.zeros((ws.size, n, n), dtype=np.complex128)
    for iw, omega in enumerate(ws):
        lam1, lam2 = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
        denom = (lam1 - lam2) * float(params.spring)
        for j in range(n):
            for l in range(n):
                out[iw, j, l] = lam1 ** abs(j - l) / denom
    return out


def cubic_onsite_g0_lesser(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
) -> Array:
    """Return analytic lesser G0 from the paper's Eq. (76)."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    out = np.zeros((ws.size, n, n), dtype=np.complex128)
    f_l = _stable_bose(ws, temp_left, kb_effective=kb_effective)
    f_r = _stable_bose(ws, temp_right, kb_effective=kb_effective)
    for iw, omega in enumerate(ws):
        lam1, lam2 = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
        denom = (lam1 - lam2) * float(params.spring)
        theta = _theta_band(omega, spring=params.spring, onsite_spring=params.onsite_spring)
        for j in range(n):
            for l in range(n):
                out[iw, j, l] = theta * (
                    f_l[iw] * (lam1 ** (j - l)) + f_r[iw] * (lam1 ** (l - j))
                ) / denom
    return out


def cubic_onsite_lead_self_energies(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
) -> tuple[Array, Array, Array, Array]:
    """Return analytic lead retarded/lesser self-energies for the center sites."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    sigma_l_r = np.zeros((ws.size, n, n), dtype=np.complex128)
    sigma_r_r = np.zeros_like(sigma_l_r)
    sigma_l_less = np.zeros_like(sigma_l_r)
    sigma_r_less = np.zeros_like(sigma_l_r)
    f_l = _stable_bose(ws, temp_left, kb_effective=kb_effective)
    f_r = _stable_bose(ws, temp_right, kb_effective=kb_effective)

    for iw, omega in enumerate(ws):
        lam1, _, = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
        g0 = -lam1 / float(params.spring)
        sigma_l_r[iw, 0, 0] = (float(params.spring) ** 2) * g0
        sigma_r_r[iw, -1, -1] = (float(params.spring) ** 2) * g0

        imag_lam1 = float(np.imag(lam1))
        sigma_l_less[iw, 0, 0] = -2.0j * f_l[iw] * float(params.spring) * imag_lam1
        sigma_r_less[iw, -1, -1] = -2.0j * f_r[iw] * float(params.spring) * imag_lam1

    return sigma_l_r, sigma_r_r, sigma_l_less, sigma_r_less


def cubic_onsite_delta_g0_lesser(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
) -> Array:
    """Return the paper Eq. (88) variation of G0^< at equilibrium."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    out = np.zeros((ws.size, n, n), dtype=np.complex128)
    dfdt = _stable_bose_dfdt(ws, float(temperature), kb_effective=kb_effective)
    for iw, omega in enumerate(ws):
        if float(omega) <= 0.0:
            continue
        lam1, lam2 = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
        denom = 2.0 * (lam1 - lam2) * float(params.spring)
        for j in range(n):
            for l in range(n):
                out[iw, j, l] = (lam1 ** (j - l) - lam1 ** (l - j)) * dfdt[iw] / denom
    return out


def cubic_onsite_delta_lead_lesser_self_energies(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
) -> Array:
    """Return d(Sigma_L^< + Sigma_R^<)/d(Delta T) at equilibrium."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    out = np.zeros((ws.size, n, n), dtype=np.complex128)
    dfdt = _stable_bose_dfdt(ws, float(temperature), kb_effective=kb_effective)
    for iw, omega in enumerate(ws):
        lam1, _ = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
        imag_lam1 = float(np.imag(lam1))
        out[iw, 0, 0] = -1.0j * float(params.spring) * imag_lam1 * dfdt[iw]
        out[iw, -1, -1] = +1.0j * float(params.spring) * imag_lam1 * dfdt[iw]
    return out


def cubic_onsite_delta_self_energies_paper(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
    include_second_graph: bool = False,
) -> tuple[Array, Array, Array]:
    """Return paper Eq. (89) derivatives dSigma_n^{r,<}/d(Delta T)."""

    if include_second_graph:
        raise NotImplementedError("paper derivative mode currently supports only the first LO graph.")

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=eta)
    g0_less = cubic_onsite_g0_lesser(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=eta,
        kb_effective=kb_effective,
    )
    delta_g0_less = cubic_onsite_delta_g0_lesser(
        ws,
        params=params,
        temperature=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )

    delta_sigma_r = np.zeros((ws.size, n, n), dtype=np.complex128)
    delta_sigma_less = np.zeros_like(delta_sigma_r)
    diff_grid = (ws[:, None] - ws[None, :]).reshape(-1)
    g_shift_r = cubic_onsite_g0_retarded(diff_grid, params=params, eta=eta).reshape(ws.size, ws.size, n, n)
    g_shift_l = cubic_onsite_g0_lesser(
        diff_grid,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=eta,
        kb_effective=kb_effective,
    ).reshape(ws.size, ws.size, n, n)

    for j in range(n):
        for l in range(n):
            delta_sigma_r[:, j, l] = (
                4.0j
                * (float(params.cubic) ** 2)
                * np.trapezoid(g_shift_r[:, :, j, l] * delta_g0_less[None, :, j, l], ws, axis=1)
                / (2.0 * np.pi)
            )
            delta_sigma_less[:, j, l] = (
                4.0j
                * (float(params.cubic) ** 2)
                * np.trapezoid(g_shift_l[:, :, j, l] * delta_g0_less[None, :, j, l], ws, axis=1)
                / (2.0 * np.pi)
            )

    return delta_g0_less, delta_sigma_r, delta_sigma_less


def _bar_g_r_j(omega: float, j: int, *, params: CubicOnsiteChainParams, eta: float) -> complex:
    lam1, lam2 = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
    n = int(params.n_layers)
    num = 1.0 + lam1 - (lam1 ** (j + 1)) - (lam1 ** (n - j))
    den = (1.0 - lam1) * (lam1 - lam2) * float(params.spring)
    return num / den


def _bar_g_t(omega: float, *, params: CubicOnsiteChainParams, temp_left: float, temp_right: float, eta: float, kb_effective: float) -> complex:
    lam1, lam2 = _lambda_roots(omega, spring=params.spring, onsite_spring=params.onsite_spring, eta=eta)
    theta = _theta_band(omega, spring=params.spring, onsite_spring=params.onsite_spring)
    f_l = float(_stable_bose(np.array([omega]), temp_left, kb_effective=kb_effective)[0])
    f_r = float(_stable_bose(np.array([omega]), temp_right, kb_effective=kb_effective)[0])
    return (1.0 + (f_l + f_r) * theta) / ((lam1 - lam2) * float(params.spring))


def _bar_g_t_zero(
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    omega_max: float,
    n_conv: int,
    eta: float,
    kb_effective: float,
) -> complex:
    omega_pos = np.linspace(0.0, float(omega_max), int(n_conv))
    omega_neg = -omega_pos[::-1]
    omega_all = np.concatenate([omega_neg[:-1], omega_pos])
    vals = np.array(
        [
            _bar_g_t(
                float(w),
                params=params,
                temp_left=float(temp_left),
                temp_right=float(temp_right),
                eta=float(eta),
                kb_effective=float(kb_effective),
            )
            for w in omega_all
        ],
        dtype=np.complex128,
    )
    return np.trapezoid(vals, omega_all) / (2.0 * np.pi)


def cubic_onsite_lowest_order_self_energies(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
) -> tuple[Array, Array]:
    """Return paper-specific LO cubic self-energies for the onsite model."""

    ws = np.asarray(omegas, dtype=float)
    n = int(params.n_layers)
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=eta)
    g0_less = cubic_onsite_g0_lesser(
        ws,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=eta,
        kb_effective=kb_effective,
    )

    sigma_r = np.zeros((ws.size, n, n), dtype=np.complex128)
    sigma_less = np.zeros_like(sigma_r)
    diff_grid = (ws[:, None] - ws[None, :]).reshape(-1)
    g_shift_r = cubic_onsite_g0_retarded(diff_grid, params=params, eta=eta).reshape(ws.size, ws.size, n, n)
    g_shift_l = cubic_onsite_g0_lesser(
        diff_grid,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=eta,
        kb_effective=kb_effective,
    ).reshape(ws.size, ws.size, n, n)

    for j in range(n):
        for l in range(n):
            sigma_r[:, j, l] = (
                2.0j
                * (float(params.cubic) ** 2)
                * np.trapezoid(g_shift_r[:, :, j, l] * g0_less[None, :, j, l], ws, axis=1)
                / (2.0 * np.pi)
            )
            sigma_less[:, j, l] = (
                2.0j
                * (float(params.cubic) ** 2)
                * np.trapezoid(g_shift_l[:, :, j, l] * g0_less[None, :, j, l], ws, axis=1)
                / (2.0 * np.pi)
            )

    bar_g_t_zero = _bar_g_t_zero(
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        omega_max=max(float(ws[-1]), 1.0),
        n_conv=max(int(ws.size), 64),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    for j in range(n):
        sigma_r[:, j, j] += 2.0j * (float(params.cubic) ** 2) * _bar_g_r_j(0.0, j, params=params, eta=eta) * bar_g_t_zero

    return sigma_r, sigma_less


def cubic_onsite_current(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
    hbar_effective: float = 1.0,
) -> dict[str, object]:
    """Return LO current data for the cubic onsite model."""

    ws = np.asarray(omegas, dtype=float)
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=eta)
    sigma_n_r, sigma_n_less = cubic_onsite_lowest_order_self_energies(
        ws,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    sigma_l_r, sigma_r_r, sigma_l_less, sigma_r_less = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )

    n = int(params.n_layers)
    g_r = np.zeros_like(g0_r)
    g_less = np.zeros_like(g0_r)
    t_eff = np.zeros(ws.size, dtype=float)
    current_integrand = np.zeros(ws.size, dtype=float)

    for iw, _ in enumerate(ws):
        g_r[iw] = np.linalg.inv(np.linalg.inv(g0_r[iw]) - sigma_n_r[iw])
        g_a = g_r[iw].conj().T
        sigma_less_total = sigma_l_less[iw] + sigma_r_less[iw] + sigma_n_less[iw]
        g_less[iw] = g_r[iw] @ sigma_less_total @ g_a

        gamma_l = 1j * (sigma_l_r[iw] - sigma_l_r[iw].conj().T)
        gamma_r = 1j * (sigma_r_r[iw] - sigma_r_r[iw].conj().T)
        term = (
            (g_r[iw] - g_a) @ (sigma_r_less[iw] - sigma_l_less[iw])
            + 1j * g_less[iw] @ (gamma_r - gamma_l)
        )
        t_eff[iw] = float(np.real_if_close(np.trace(term) / (2.0 * ( _stable_bose(np.array([ws[iw]]), temp_left, kb_effective=kb_effective)[0] - _stable_bose(np.array([ws[iw]]), temp_right, kb_effective=kb_effective)[0] + 1e-30))).real)
        current_integrand[iw] = float(
            np.real_if_close(
                ws[iw]
                * np.trace((g_r[iw] - g_a) @ (sigma_r_less[iw] - sigma_l_less[iw]) + 1j * g_less[iw] @ (gamma_r - gamma_l))
                / (4.0 * np.pi)
            ).real
        )

    current = float(float(hbar_effective) * np.trapezoid(current_integrand, ws))
    return {
        "omegas": ws,
        "g_retarded": g_r,
        "g_lesser": g_less,
        "sigma_n_retarded": sigma_n_r,
        "sigma_n_lesser": sigma_n_less,
        "effective_transmission": t_eff,
        "current_internal": current,
    }


def _to_time_domain_uniform(omegas: Array, values_w: Array) -> tuple[Array, Array]:
    ws = np.asarray(omegas, dtype=float)
    if ws.ndim != 1 or ws.size < 8:
        raise ValueError("omegas must be a 1D array with at least 8 points.")
    if not np.allclose(np.diff(ws), ws[1] - ws[0], rtol=1e-10, atol=1e-12):
        raise ValueError("omegas must be uniformly spaced for the time-domain LO transform.")
    dw = float(ws[1] - ws[0])
    dt = 2.0 * np.pi / (float(ws.size) * dw)
    ts = dt * (np.arange(ws.size) - ws.size // 2)
    phase = np.exp(-1j * np.outer(ts, ws))
    vals_t = np.tensordot(phase, np.asarray(values_w, dtype=np.complex128), axes=(1, 0)) * dw / (2.0 * np.pi)
    return ts, vals_t


def _to_frequency_domain_uniform(omegas: Array, times: Array, values_t: Array) -> Array:
    ws = np.asarray(omegas, dtype=float)
    ts = np.asarray(times, dtype=float)
    if ts.ndim != 1 or ts.size != ws.size:
        raise ValueError("times must be a 1D array with the same length as omegas.")
    dt = float(ts[1] - ts[0]) if ts.size >= 2 else 1.0
    phase = np.exp(1j * np.outer(ws, ts))
    return np.tensordot(phase, np.asarray(values_t, dtype=np.complex128), axes=(1, 0)) * dt


def _extended_first_graph_grid(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    internal_oversample: int = 1,
) -> tuple[Array, Array]:
    """Return a transform grid wide enough for the Eq. (77) self-convolution.

    The first cubic graph is a frequency-space self-convolution of the
    harmonic greater/lesser functions. Since the harmonic spectral support is
    limited to the phonon band edge ``sqrt(4K + K0)``, the first graph carries
    support up to roughly twice that value. If we evaluate Eq. (77) on the same
    truncated window as the requested output grid, high-frequency self-energy
    weight aliases back into the physical window and distorts the strong-coupling
    conductance. We therefore evaluate the time-domain product on an extended
    uniform grid and sample the result back onto the requested points.
    """

    ws = np.asarray(omegas, dtype=float)
    if ws.ndim != 1 or ws.size < 8:
        raise ValueError("omegas must be a 1D array with at least 8 points.")
    if not np.allclose(np.diff(ws), ws[1] - ws[0], rtol=1e-10, atol=1e-12):
        raise ValueError("omegas must be uniformly spaced for the time-domain LO transform.")

    if int(internal_oversample) <= 0:
        raise ValueError("internal_oversample must be positive.")

    dw = float(ws[1] - ws[0])
    dw_ext = dw / float(int(internal_oversample))
    band_edge = _band_edge(params)
    support = max(abs(float(ws[0])), abs(float(ws[-1])), 2.0 * float(band_edge))
    n_pos = int(np.ceil(support / dw_ext))
    w_pos = dw_ext * np.arange(n_pos + 1)
    ws_ext = np.concatenate([-w_pos[:0:-1], w_pos])
    sample_idx = np.rint((ws - float(ws_ext[0])) / dw_ext).astype(int)
    if np.any(sample_idx < 0) or np.any(sample_idx >= ws_ext.size):
        raise RuntimeError("Failed to embed the requested omega grid inside the extended transform grid.")
    if not np.allclose(ws_ext[sample_idx], ws, rtol=1e-10, atol=1e-12):
        raise RuntimeError("Failed to map the requested omega grid onto the extended transform grid.")
    return ws_ext, sample_idx


def cubic_onsite_lowest_order_self_energies_exact(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
    include_second_graph: bool = False,
    internal_oversample: int = 1,
) -> tuple[Array, Array]:
    """Return the cubic onsite LO self-energies using the exact time-domain first graph.

    The first diagram in Wang 0701164 Eq. (77) is evaluated in the time domain,
    where it is simply proportional to the square of the contour Green's
    function. The constant second graph Eq. (78)-(79) can optionally be added.
    For the Fig. 5 benchmark path we keep it optional because the published
    conductance is reproduced much more faithfully after tadpole subtraction.
    """

    ws = np.asarray(omegas, dtype=float)
    if ws[0] >= 0.0 or ws[-1] <= 0.0:
        raise ValueError("omegas must span both negative and positive values.")

    ws_ext, sample_idx = _extended_first_graph_grid(
        ws,
        params=params,
        internal_oversample=int(internal_oversample),
    )

    g0_r_ext = cubic_onsite_g0_retarded(ws_ext, params=params, eta=eta)
    g0_l_ext = cubic_onsite_g0_lesser(
        ws_ext,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    g0_a_ext = np.conjugate(np.swapaxes(g0_r_ext, 1, 2))
    g0_g_ext = g0_l_ext + (g0_r_ext - g0_a_ext)

    times_ext, g0_l_t = _to_time_domain_uniform(ws_ext, g0_l_ext)
    _, g0_g_t = _to_time_domain_uniform(ws_ext, g0_g_ext)
    sigma_l_t = 2.0j * (float(params.cubic) ** 2) * (g0_l_t**2)
    sigma_g_t = 2.0j * (float(params.cubic) ** 2) * (g0_g_t**2)
    theta = (times_ext >= 0.0).astype(float)[:, None, None]
    sigma_r_ext = _to_frequency_domain_uniform(ws_ext, times_ext, theta * (sigma_g_t - sigma_l_t))
    sigma_lesser_ext = _to_frequency_domain_uniform(ws_ext, times_ext, sigma_l_t)
    sigma_r = sigma_r_ext[sample_idx]
    sigma_lesser = sigma_lesser_ext[sample_idx]

    if include_second_graph:
        bar_g_t_zero = _bar_g_t_zero(
            params=params,
            temp_left=float(temp_left),
            temp_right=float(temp_right),
            omega_max=max(abs(float(ws_ext[0])), abs(float(ws_ext[-1]))),
            n_conv=max(4 * int(ws_ext.size), 401),
            eta=float(eta),
            kb_effective=float(kb_effective),
        )
        for j in range(params.n_layers):
            sigma_r[:, j, j] += 2.0j * (float(params.cubic) ** 2) * _bar_g_r_j(
                0.0, j, params=params, eta=eta
            ) * bar_g_t_zero

    return sigma_r, sigma_lesser


def cubic_onsite_effective_transmission(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    delta_t: float,
    eta: float,
    kb_effective: float,
    include_second_graph: bool = False,
    internal_oversample: int = 1,
) -> dict[str, object]:
    """Return the Fig. 5 effective transmission from the LO cubic onsite model."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")

    ws = np.asarray(omegas, dtype=float)
    temp_left = float(temperature) + 0.5 * float(delta_t)
    temp_right = float(temperature) - 0.5 * float(delta_t)

    sigma_n_r, sigma_n_lesser = cubic_onsite_lowest_order_self_energies_exact(
        ws,
        params=params,
        temp_left=temp_left,
        temp_right=temp_right,
        eta=float(eta),
        kb_effective=float(kb_effective),
        include_second_graph=include_second_graph,
        internal_oversample=int(internal_oversample),
    )
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=float(eta))
    sigma_l_r, sigma_r_r, sigma_l_less, sigma_r_less = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=temp_left,
        temp_right=temp_right,
        eta=float(eta),
        kb_effective=float(kb_effective),
    )

    pos = ws > 0.0
    wp = ws[pos]
    g_r = np.zeros((wp.size, params.n_layers, params.n_layers), dtype=np.complex128)
    g_less = np.zeros_like(g_r)
    t_eff = np.zeros(wp.size, dtype=float)

    pos_idx = np.where(pos)[0]
    for iw, idx in enumerate(pos_idx):
        gr = np.linalg.inv(np.linalg.inv(g0_r[idx]) - sigma_n_r[idx])
        ga = gr.conj().T
        gless = gr @ (sigma_l_less[idx] + sigma_r_less[idx] + sigma_n_lesser[idx]) @ ga
        gamma_l = 1j * (sigma_l_r[idx] - sigma_l_r[idx].conj().T)
        gamma_r = 1j * (sigma_r_r[idx] - sigma_r_r[idx].conj().T)
        fl = float(_stable_bose(np.array([wp[iw]]), temp_left, kb_effective=kb_effective)[0])
        fr = float(_stable_bose(np.array([wp[iw]]), temp_right, kb_effective=kb_effective)[0])
        numerator = 0.5 * np.trace((gr - ga) @ (sigma_r_less[idx] - sigma_l_less[idx]) + 1j * gless @ (gamma_r - gamma_l))
        t_eff[iw] = float(np.real_if_close(numerator / max(fl - fr, 1e-30)).real)
        g_r[iw] = gr
        g_less[iw] = gless

    return {
        "omegas_positive": wp,
        "g_retarded_positive": g_r,
        "g_lesser_positive": g_less,
        "effective_transmission": t_eff,
        "sigma_n_retarded": sigma_n_r,
        "sigma_n_lesser": sigma_n_lesser,
        "temp_left": temp_left,
        "temp_right": temp_right,
        "include_second_graph": bool(include_second_graph),
    }


def cubic_onsite_current_exact(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temp_left: float,
    temp_right: float,
    eta: float,
    kb_effective: float,
    hbar_effective: float = 1.0,
    include_second_graph: bool = False,
    internal_oversample: int = 1,
) -> dict[str, object]:
    """Return the LO current using the exact time-domain first graph."""

    ws = np.asarray(omegas, dtype=float)
    sigma_n_r, sigma_n_lesser = cubic_onsite_lowest_order_self_energies_exact(
        ws,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=float(eta),
        kb_effective=float(kb_effective),
        include_second_graph=include_second_graph,
        internal_oversample=int(internal_oversample),
    )
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=float(eta))
    sigma_l_r, sigma_r_r, sigma_l_less, sigma_r_less = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temp_left),
        temp_right=float(temp_right),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )

    pos = ws > 0.0
    wp = ws[pos]
    spectral_current = np.zeros(wp.size, dtype=float)
    g_r = np.zeros((wp.size, params.n_layers, params.n_layers), dtype=np.complex128)
    g_less = np.zeros_like(g_r)

    pos_idx = np.where(pos)[0]
    for iw, idx in enumerate(pos_idx):
        gr = np.linalg.inv(np.linalg.inv(g0_r[idx]) - sigma_n_r[idx])
        ga = gr.conj().T
        gless = gr @ (sigma_l_less[idx] + sigma_r_less[idx] + sigma_n_lesser[idx]) @ ga
        gamma_l = 1j * (sigma_l_r[idx] - sigma_l_r[idx].conj().T)
        gamma_r = 1j * (sigma_r_r[idx] - sigma_r_r[idx].conj().T)
        numerator = 0.5 * np.trace(
            (gr - ga) @ (sigma_r_less[idx] - sigma_l_less[idx])
            + 1j * gless @ (gamma_r - gamma_l)
        )
        spectral_current[iw] = float(np.real_if_close(wp[iw] * numerator / (2.0 * np.pi)).real)
        g_r[iw] = gr
        g_less[iw] = gless

    current = float(float(hbar_effective) * np.trapezoid(spectral_current, wp))
    return {
        "omegas_positive": wp,
        "g_retarded_positive": g_r,
        "g_lesser_positive": g_less,
        "sigma_n_retarded": sigma_n_r,
        "sigma_n_lesser": sigma_n_lesser,
        "spectral_current_positive": spectral_current,
        "current": current,
        "temp_left": float(temp_left),
        "temp_right": float(temp_right),
        "include_second_graph": bool(include_second_graph),
    }


def cubic_onsite_effective_transmission_paper_derivative(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
    include_second_graph: bool = False,
) -> dict[str, object]:
    """Return the paper Eq. (84)-(89) effective transmission at equilibrium."""

    if include_second_graph:
        raise NotImplementedError("paper derivative mode currently supports only the first LO graph.")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")

    ws = np.asarray(omegas, dtype=float)
    sigma_n_r, sigma_n_lesser = cubic_onsite_lowest_order_self_energies_exact(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
        include_second_graph=False,
    )
    g0_r = cubic_onsite_g0_retarded(ws, params=params, eta=float(eta))
    sigma_l_r, sigma_r_r, sigma_l_less, sigma_r_less = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    delta_g0_less, delta_sigma_n_r, delta_sigma_n_less = cubic_onsite_delta_self_energies_paper(
        ws,
        params=params,
        temperature=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
        include_second_graph=False,
    )
    g0_less_eq = cubic_onsite_g0_lesser(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    pos = ws > 0.0
    wp = ws[pos]
    dfdt = _stable_bose_dfdt(wp, float(temperature), kb_effective=kb_effective)
    f_eq = _stable_bose(wp, float(temperature), kb_effective=kb_effective)

    g_r = np.zeros((wp.size, params.n_layers, params.n_layers), dtype=np.complex128)
    g_less = np.zeros_like(g_r)
    delta_g_r = np.zeros_like(g_r)
    delta_g_less = np.zeros_like(g_r)
    t_eff = np.zeros(wp.size, dtype=float)

    pos_idx = np.where(pos)[0]
    for iw, idx in enumerate(pos_idx):
        gr = np.linalg.inv(np.linalg.inv(g0_r[idx]) - sigma_n_r[idx])
        ga = gr.conj().T
        sigma_less_total = sigma_l_less[idx] + sigma_r_less[idx] + sigma_n_lesser[idx]
        gless = gr @ sigma_less_total @ ga

        dgr = gr @ delta_sigma_n_r[idx] @ gr
        dga = dgr.conj().T
        sigma_n_a = sigma_n_r[idx].conj().T
        delta_sigma_n_a = delta_sigma_n_r[idx].conj().T
        # Eq. (87): bar{G0^<} = G0^< (I + Sigma_n^a G^a)
        g0_bar_less = g0_less_eq[idx] @ (np.eye(params.n_layers, dtype=np.complex128) + sigma_n_a @ ga)
        term = gr @ (
            sigma_n_r[idx] @ delta_g0_less[idx]
            + delta_sigma_n_r[idx]
            @ (g0_bar_less + gr @ (sigma_n_lesser[idx] @ ga + sigma_n_r[idx] @ g0_bar_less))
        )
        dgless = (
            term
            - term.conj().T
            + delta_g0_less[idx]
            + gr
            @ (delta_sigma_n_less[idx] + sigma_n_r[idx] @ delta_g0_less[idx] @ sigma_n_a)
            @ ga
        )

        g_r[iw] = gr
        g_less[iw] = gless
        delta_g_r[iw] = dgr
        delta_g_less[iw] = dgless

        if abs(dfdt[iw]) < 1e-30:
            continue

        spectral = gr - ga
        delta_spectral = dgr - dga
        common = (f_eq[iw] * delta_spectral - dgless) / dfdt[iw]
        f_left = 0.5 * spectral + common
        f_right = -0.5 * spectral + common
        lam1, _ = _lambda_roots(float(wp[iw]), spring=params.spring, onsite_spring=params.onsite_spring, eta=float(eta))
        pref = 1.0j * float(params.spring) * float(np.imag(lam1))
        t_eff[iw] = float(np.real_if_close(pref * (f_left[0, 0] - f_right[-1, -1])).real)

    return {
        "omegas_positive": wp,
        "g_retarded_positive": g_r,
        "g_lesser_positive": g_less,
        "delta_g_retarded_positive": delta_g_r,
        "delta_g_lesser_positive": delta_g_less,
        "effective_transmission": t_eff,
        "sigma_n_retarded": sigma_n_r,
        "sigma_n_lesser": sigma_n_lesser,
        "delta_g0_lesser": delta_g0_less,
        "delta_sigma_n_retarded": delta_sigma_n_r,
        "delta_sigma_n_lesser": delta_sigma_n_less,
        "include_second_graph": False,
    }


def cubic_onsite_effective_transmission_paper_derivative_numeric(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
    delta_t: float = 0.02,
    include_second_graph: bool = False,
    hbar_effective: float = 1.0,
    internal_oversample: int = 1,
) -> dict[str, object]:
    """Return Eq. (85) transmission using finite-difference derivatives of the exact LO closure.

    This benchmark branch keeps the paper's derivative-based observable while
    differentiating the equilibrium Green's functions numerically from the same
    exact time-domain first-graph closure used for the best current benchmark.
    In practice this is substantially more consistent than mixing the exact
    first-graph equilibrium self-energy with the analytic Eq. (89) derivative
    branch, and it gives a closer match to Fig. 5.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")

    ws = np.asarray(omegas, dtype=float)
    eq = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )
    plus = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature) + 0.5 * float(delta_t),
        temp_right=float(temperature) - 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )
    minus = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature) - 0.5 * float(delta_t),
        temp_right=float(temperature) + 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )

    wp = np.asarray(eq["omegas_positive"], dtype=float)
    g_r = np.asarray(eq["g_retarded_positive"], dtype=np.complex128)
    g_less = np.asarray(eq["g_lesser_positive"], dtype=np.complex128)
    central_scale = 2.0 * float(delta_t)
    delta_g_r = (
        np.asarray(plus["g_retarded_positive"], dtype=np.complex128)
        - np.asarray(minus["g_retarded_positive"], dtype=np.complex128)
    ) / central_scale
    delta_g_less = (
        np.asarray(plus["g_lesser_positive"], dtype=np.complex128)
        - np.asarray(minus["g_lesser_positive"], dtype=np.complex128)
    ) / central_scale

    g_a = np.conjugate(np.swapaxes(g_r, 1, 2))
    delta_g_a = np.conjugate(np.swapaxes(delta_g_r, 1, 2))
    spectral = g_r - g_a
    delta_spectral = delta_g_r - delta_g_a
    f_eq = _stable_bose(wp, float(temperature), kb_effective=float(kb_effective))
    dfdt = _stable_bose_dfdt(wp, float(temperature), kb_effective=float(kb_effective))
    t_eff = np.zeros(wp.size, dtype=float)

    for iw, omega in enumerate(wp):
        if abs(dfdt[iw]) < 1e-30:
            continue
        common = (f_eq[iw] * delta_spectral[iw] - delta_g_less[iw]) / dfdt[iw]
        f_left = 0.5 * spectral[iw] + common
        f_right = -0.5 * spectral[iw] + common
        lam1, _ = _lambda_roots(float(omega), spring=params.spring, onsite_spring=params.onsite_spring, eta=float(eta))
        pref = 1.0j * float(params.spring) * float(np.imag(lam1))
        t_eff[iw] = float(np.real_if_close(pref * (f_left[0, 0] - f_right[-1, -1])).real)

    return {
        "omegas_positive": wp,
        "g_retarded_positive": g_r,
        "g_lesser_positive": g_less,
        "delta_g_retarded_positive": delta_g_r,
        "delta_g_lesser_positive": delta_g_less,
        "effective_transmission": t_eff,
        "sigma_n_retarded": np.asarray(eq["sigma_n_retarded"], dtype=np.complex128),
        "sigma_n_lesser": np.asarray(eq["sigma_n_lesser"], dtype=np.complex128),
        "include_second_graph": bool(include_second_graph),
        "delta_t": float(delta_t),
    }


def cubic_onsite_eq84_numeric_decomposition(
    omegas: Array,
    *,
    params: CubicOnsiteChainParams,
    temperature: float,
    eta: float,
    kb_effective: float,
    delta_t: float = 0.02,
    include_second_graph: bool = False,
    hbar_effective: float = 1.0,
    internal_oversample: int = 1,
) -> dict[str, object]:
    """Decompose the Eq. (84) linear-response numerator into finite-difference terms.

    This helper is diagnostic rather than a benchmark path. It evaluates the
    equilibrium interacting Green's functions with the corrected first-graph
    closure, then constructs finite-difference derivatives with respect to the
    lead temperature bias ``Delta T``.

    The three returned frequency-resolved terms are:

    - ``delta_spectral_sigma_eq``: ``0.5 Tr[ delta(G^r-G^a) (Sigma_R^< - Sigma_L^<)_eq ]``
    - ``spectral_delta_sigma``: ``0.5 Tr[ (G^r-G^a)_eq delta(Sigma_R^< - Sigma_L^<) ]``
    - ``lesser_gamma_eq``: ``0.5 Tr[ i delta G^< (Gamma_R - Gamma_L)_eq ]``

    Their sum reproduces the direct finite-difference derivative of the Eq. (84)
    numerator in the current formulation used here. This is useful for auditing
    which piece of the linear-response observable is responsible for a mismatch.
    """

    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")

    ws = np.asarray(omegas, dtype=float)
    eq = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )
    plus = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature) + 0.5 * float(delta_t),
        temp_right=float(temperature) - 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )
    minus = cubic_onsite_current_exact(
        ws,
        params=params,
        temp_left=float(temperature) - 0.5 * float(delta_t),
        temp_right=float(temperature) + 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
        hbar_effective=float(hbar_effective),
        include_second_graph=bool(include_second_graph),
        internal_oversample=int(internal_oversample),
    )

    wp = np.asarray(eq["omegas_positive"], dtype=float)
    g_r_eq = np.asarray(eq["g_retarded_positive"], dtype=np.complex128)
    g_a_eq = np.conjugate(np.swapaxes(g_r_eq, 1, 2))
    central_scale = 2.0 * float(delta_t)
    delta_g_r = (
        np.asarray(plus["g_retarded_positive"], dtype=np.complex128)
        - np.asarray(minus["g_retarded_positive"], dtype=np.complex128)
    ) / central_scale
    delta_g_a = np.conjugate(np.swapaxes(delta_g_r, 1, 2))
    delta_g_less = (
        np.asarray(plus["g_lesser_positive"], dtype=np.complex128)
        - np.asarray(minus["g_lesser_positive"], dtype=np.complex128)
    ) / central_scale

    sigma_l_r_eq, sigma_r_r_eq, sigma_l_less_eq, sigma_r_less_eq = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temperature),
        temp_right=float(temperature),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    sigma_l_r_p, sigma_r_r_p, sigma_l_less_p, sigma_r_less_p = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temperature) + 0.5 * float(delta_t),
        temp_right=float(temperature) - 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )
    sigma_l_r_m, sigma_r_r_m, sigma_l_less_m, sigma_r_less_m = cubic_onsite_lead_self_energies(
        ws,
        params=params,
        temp_left=float(temperature) - 0.5 * float(delta_t),
        temp_right=float(temperature) + 0.5 * float(delta_t),
        eta=float(eta),
        kb_effective=float(kb_effective),
    )

    pos = ws > 0.0
    sigma_diff_eq = sigma_r_less_eq[pos] - sigma_l_less_eq[pos]
    delta_sigma_diff = (
        (sigma_r_less_p[pos] - sigma_l_less_p[pos]) - (sigma_r_less_m[pos] - sigma_l_less_m[pos])
    ) / central_scale
    gamma_diff_eq = 1j * (
        (sigma_r_r_eq[pos] - np.conjugate(np.swapaxes(sigma_r_r_eq[pos], 1, 2)))
        - (sigma_l_r_eq[pos] - np.conjugate(np.swapaxes(sigma_l_r_eq[pos], 1, 2)))
    )

    term_delta_spectral_sigma_eq = np.zeros(wp.size, dtype=float)
    term_spectral_delta_sigma = np.zeros(wp.size, dtype=float)
    term_lesser_gamma_eq = np.zeros(wp.size, dtype=float)

    for iw in range(wp.size):
        term_delta_spectral_sigma_eq[iw] = float(
            np.real_if_close(
                0.5 * np.trace((delta_g_r[iw] - delta_g_a[iw]) @ sigma_diff_eq[iw])
            ).real
        )
        term_spectral_delta_sigma[iw] = float(
            np.real_if_close(
                0.5 * np.trace((g_r_eq[iw] - g_a_eq[iw]) @ delta_sigma_diff[iw])
            ).real
        )
        term_lesser_gamma_eq[iw] = float(
            np.real_if_close(
                0.5 * np.trace(1j * delta_g_less[iw] @ gamma_diff_eq[iw])
            ).real
        )

    numerator_total = term_delta_spectral_sigma_eq + term_spectral_delta_sigma + term_lesser_gamma_eq
    dfdt = _stable_bose_dfdt(wp, float(temperature), kb_effective=float(kb_effective))
    effective_transmission = np.zeros_like(wp)
    valid = np.abs(dfdt) > 1e-30
    effective_transmission[valid] = numerator_total[valid] / dfdt[valid]

    kappa_delta_spectral_sigma_eq = float(
        float(hbar_effective) * np.trapezoid(wp * term_delta_spectral_sigma_eq, wp) / (2.0 * np.pi)
    )
    kappa_spectral_delta_sigma = float(
        float(hbar_effective) * np.trapezoid(wp * term_spectral_delta_sigma, wp) / (2.0 * np.pi)
    )
    kappa_lesser_gamma_eq = float(
        float(hbar_effective) * np.trapezoid(wp * term_lesser_gamma_eq, wp) / (2.0 * np.pi)
    )
    kappa_total = float(
        float(hbar_effective) * np.trapezoid(wp * numerator_total, wp) / (2.0 * np.pi)
    )

    return {
        "omegas_positive": wp,
        "dfdt": dfdt,
        "delta_g_retarded_positive": delta_g_r,
        "delta_g_lesser_positive": delta_g_less,
        "delta_spectral_sigma_eq": term_delta_spectral_sigma_eq,
        "spectral_delta_sigma": term_spectral_delta_sigma,
        "lesser_gamma_eq": term_lesser_gamma_eq,
        "numerator_total": numerator_total,
        "effective_transmission": effective_transmission,
        "kappa_delta_spectral_sigma_eq": kappa_delta_spectral_sigma_eq,
        "kappa_spectral_delta_sigma": kappa_spectral_delta_sigma,
        "kappa_lesser_gamma_eq": kappa_lesser_gamma_eq,
        "kappa_total": kappa_total,
        "delta_t": float(delta_t),
        "include_second_graph": bool(include_second_graph),
    }


def wang0701164_fig5_lowest_order_conductance_sweep(
    temperatures: Array,
    *,
    cubic_values: Array | None = None,
    unit_system: str = "paper",
    omega_max: float | None = None,
    n_omega: int = 61,
    delta_t: float = 0.1,
    eta: float = 1e-4,
    kb_effective: float | None = None,
    hbar_effective: float = 1.0,
    include_second_graph: bool = False,
    conductance_mode: str = "effective_transmission",
    internal_oversample: int = 1,
) -> dict[str, object]:
    """Return a Fig. 5-style conductance sweep for the cubic onsite model.

    The default benchmark path follows the paper's native numeric dynamics with
    physical prefactors, and uses the exact time-domain first graph. The
    constant second graph can be enabled for analysis, but it is disabled by
    default because tadpole subtraction reproduces the published Fig. 5 curves
    far more faithfully. The small imaginary broadening defaults to ``1e-4`` so
    the function matches the literature example runner and saved benchmark
    metadata out of the box.

    ``conductance_mode`` controls how the final conductance is extracted:
    ``"effective_transmission"`` uses the current benchmark path based on the
    LO effective transmission, ``"current_over_delta_t"`` computes the LO
    current from the same exact self-energies and divides by ``delta_t``, and
    ``"paper_derivative"`` follows the current analytic Eq. (84)-(89)
    derivative route, while ``"paper_derivative_numeric"`` uses the same
    derivative-based observable but evaluates the derivatives of ``G^r`` and
    ``G^<`` numerically from the exact first-graph closure. The numeric branch
    is currently the closer Fig. 5 benchmark.

    Important note:
    ``omega_max`` is used as the symmetric frequency window for the time-domain
    transform. The conductance integral itself is always restricted to the
    physical harmonic band edge ``sqrt(4K + K0)``.
    """

    temps = np.asarray(temperatures, dtype=float)
    if temps.ndim != 1 or temps.size == 0 or np.any(temps <= 0.0):
        raise ValueError("temperatures must be a non-empty 1D array of positive values.")
    if n_omega < 32:
        raise ValueError("n_omega must be at least 32.")
    if int(internal_oversample) <= 0:
        raise ValueError("internal_oversample must be positive.")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")
    if conductance_mode not in {"effective_transmission", "current_over_delta_t", "paper_derivative", "paper_derivative_numeric"}:
        raise ValueError(
            "conductance_mode must be 'effective_transmission', 'current_over_delta_t', 'paper_derivative', "
            "or 'paper_derivative_numeric'."
        )

    if unit_system == "si":
        kb_eff = wang0701164_kb_effective_si() if kb_effective is None else float(kb_effective)
        hbar_eff = HBAR_SI if float(hbar_effective) == 1.0 else float(hbar_effective)
        base_params = wang0701164_fig5_params(cubic=0.0, unit_system="si")
        alpha = wang0701164_paper_unit_factors()["omega_scale_si"]
        omega_cut = float(omega_max) if omega_max is not None else 2.1 * alpha
    elif unit_system == "paper":
        factors = wang0701164_paper_unit_factors()
        kb_eff = factors["kb_effective_paper"] if kb_effective is None else float(kb_effective)
        hbar_eff = factors["hbar_effective_paper"] if float(hbar_effective) == 1.0 else float(hbar_effective)
        if omega_max is not None:
            omega_cut = float(omega_max)
        elif conductance_mode == "paper_derivative_numeric":
            # The numeric Eq. (85) benchmark converges better with a slightly
            # wider transform window than the older 2.1 default.
            omega_cut = 2.3
        else:
            omega_cut = 2.1
    else:
        raise ValueError("unit_system must be 'paper' or 'si'.")

    tvals = np.asarray([0.0, 0.2, 0.5, 0.7, 1.0, 2.0] if cubic_values is None else cubic_values, dtype=float)
    omegas_nonneg = np.linspace(0.0, omega_cut, int(n_omega))
    omegas = np.concatenate([-omegas_nonneg[:0:-1], omegas_nonneg])
    omegas_pos = omegas_nonneg[1:]
    results: dict[str, object] = {
        "temperatures": temps,
        "omegas": omegas,
        "omegas_positive": omegas_pos,
        "unit_system": unit_system,
        "delta_t": float(delta_t),
        "include_second_graph": bool(include_second_graph),
        "conductance_mode": str(conductance_mode),
        "internal_oversample": int(internal_oversample),
        "curves": {},
    }

    for cubic in tvals:
        params = wang0701164_fig5_params(cubic=float(cubic), unit_system=unit_system)
        conductance = np.zeros_like(temps)
        transmission_map = np.zeros((temps.size, omegas_pos.size), dtype=float)
        omega_physical_max = min(_band_edge(params), float(omegas_pos[-1]))
        for i, temp in enumerate(temps):
            if conductance_mode == "effective_transmission":
                run = cubic_onsite_effective_transmission(
                    omegas,
                    params=params,
                    temperature=float(temp),
                    delta_t=float(delta_t),
                    eta=float(eta),
                    kb_effective=float(kb_eff),
                    include_second_graph=bool(include_second_graph),
                    internal_oversample=int(internal_oversample),
                )
                transmission_map[i] = np.asarray(run["effective_transmission"], dtype=float)
                x = omegas_pos / (float(kb_eff) * float(temp))
                occ = np.zeros_like(x)
                mask = x < 700.0
                occ[mask] = 1.0 / np.expm1(x[mask])
                dfdt = occ * (occ + 1.0) * x / float(temp)
                band_mask = omegas_pos <= omega_physical_max
                conductance_val = float(
                    float(hbar_eff)
                    * np.trapezoid(
                        omegas_pos[band_mask] * transmission_map[i][band_mask] * dfdt[band_mask],
                        omegas_pos[band_mask],
                    )
                    / (2.0 * np.pi)
                )
            elif conductance_mode == "current_over_delta_t":
                temp_left = float(temp) + 0.5 * float(delta_t)
                temp_right = float(temp) - 0.5 * float(delta_t)
                run = cubic_onsite_current_exact(
                    omegas,
                    params=params,
                    temp_left=temp_left,
                    temp_right=temp_right,
                    eta=float(eta),
                    kb_effective=float(kb_eff),
                    hbar_effective=float(hbar_eff),
                    include_second_graph=bool(include_second_graph),
                    internal_oversample=int(internal_oversample),
                )
                conductance_val = float(run["current"]) / float(delta_t)
            elif conductance_mode == "paper_derivative":
                run = cubic_onsite_effective_transmission_paper_derivative(
                    omegas,
                    params=params,
                    temperature=float(temp),
                    eta=float(eta),
                    kb_effective=float(kb_eff),
                    include_second_graph=bool(include_second_graph),
                )
                transmission_map[i] = np.asarray(run["effective_transmission"], dtype=float)
                x = omegas_pos / (float(kb_eff) * float(temp))
                dfdt = _stable_bose_dfdt(omegas_pos, float(temp), kb_effective=float(kb_eff))
                band_mask = omegas_pos <= omega_physical_max
                conductance_val = float(
                    float(hbar_eff)
                    * np.trapezoid(
                        omegas_pos[band_mask] * transmission_map[i][band_mask] * dfdt[band_mask],
                        omegas_pos[band_mask],
                    )
                    / (2.0 * np.pi)
                )
            else:
                run = cubic_onsite_effective_transmission_paper_derivative_numeric(
                    omegas,
                    params=params,
                    temperature=float(temp),
                    eta=float(eta),
                    kb_effective=float(kb_eff),
                    delta_t=min(float(delta_t), 0.02),
                    include_second_graph=bool(include_second_graph),
                    hbar_effective=float(hbar_eff),
                    internal_oversample=int(internal_oversample),
                )
                transmission_map[i] = np.asarray(run["effective_transmission"], dtype=float)
                dfdt = _stable_bose_dfdt(omegas_pos, float(temp), kb_effective=float(kb_eff))
                band_mask = omegas_pos <= omega_physical_max
                conductance_val = float(
                    float(hbar_eff)
                    * np.trapezoid(
                        omegas_pos[band_mask] * transmission_map[i][band_mask] * dfdt[band_mask],
                        omegas_pos[band_mask],
                    )
                    / (2.0 * np.pi)
                )
            conductance[i] = max(conductance_val, 0.0)
        label = f"{float(cubic):.1f}"
        results["curves"][label] = {
            "cubic": float(cubic),
            "conductance": conductance,
            "effective_transmission_map": transmission_map,
            "omega_physical_max": omega_physical_max,
        }

    return results


def wang0701164_load_digitized_fig5_curves(path: str | Path) -> dict[str, Array]:
    """Load digitized Fig. 5 curves from a TSV file."""

    ref = Path(path)
    with ref.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = [{k: float(v) for k, v in row.items()} for row in reader]

    temps = np.array([row["temperature_K"] for row in rows], dtype=float)
    curves = {
        "0.0": 1.0e-9 * np.array([row["kappa_t0_1e-9_W_per_K"] for row in rows], dtype=float),
        "0.2": 1.0e-9 * np.array([row["kappa_t0p2_1e-9_W_per_K"] for row in rows], dtype=float),
        "0.5": 1.0e-9 * np.array([row["kappa_t0p5_1e-9_W_per_K"] for row in rows], dtype=float),
        "0.7": 1.0e-9 * np.array([row["kappa_t0p7_1e-9_W_per_K"] for row in rows], dtype=float),
        "1.0": 1.0e-9 * np.array([row["kappa_t1p0_1e-9_W_per_K"] for row in rows], dtype=float),
        "2.0": 1.0e-9 * np.array([row["kappa_t2p0_1e-9_W_per_K"] for row in rows], dtype=float),
    }
    return {
        "path": str(ref),
        "temperatures": temps,
        "curves": curves,
    }


def _wang0701164_fig5_vector_reference_from_svg_text(svg_text: str) -> dict[str, object]:
    """Parse the left-panel Fig. 5 vector curves from an SVG page dump."""

    path_pattern = re.compile(
        r'<path fill="none" stroke-width="3\.19021"[^>]*stroke="([^"]+)"[^>]*d="([^"]+)"'
    )
    number_pattern = re.compile(r"[-+]?(?:\d*\.\d+|\d+)")
    extracted: list[tuple[str, Array, Array]] = []
    for match in path_pattern.finditer(svg_text):
        stroke = match.group(1)
        values = [float(token) for token in number_pattern.findall(match.group(2))]
        xs = np.asarray(values[0::2], dtype=float)
        ys = np.asarray(values[1::2], dtype=float)
        if xs.size < 50:
            continue
        if xs.min() >= 790.0 and xs.max() <= 2930.0 and ys.min() >= 5880.0 and ys.max() <= 7375.0:
            extracted.append((stroke, xs, ys))

    if len(extracted) != 6:
        raise ValueError(f"Expected 6 Fig. 5 left-panel curves in SVG, found {len(extracted)}.")

    extracted.sort(key=lambda item: float(np.max(item[2])), reverse=True)
    labels = ["0.0", "0.2", "0.5", "0.7", "1.0", "2.0"]
    x0 = 799.921875
    x1 = 2926.757812
    y0 = 5882.890625
    y1 = 7371.679688
    curves: dict[str, Array] = {}
    curve_temperatures: dict[str, Array] = {}
    colors: dict[str, str] = {}
    for label, (stroke, xs, ys) in zip(labels, extracted):
        temps = 2000.0 * (xs - x0) / (x1 - x0)
        kappas = 1.0e-9 * 0.3 * (ys - y0) / (y1 - y0)
        curves[label] = kappas
        curve_temperatures[label] = temps
        colors[label] = stroke
    return {
        "temperatures": curve_temperatures[labels[0]],
        "curve_temperatures": curve_temperatures,
        "curves": curves,
        "colors": colors,
        "axis_mapping": {
            "x0": x0,
            "x1": x1,
            "temperature_max_K": 2000.0,
            "y0": y0,
            "y1": y1,
            "conductance_max_1e-9_W_per_K": 0.3,
        },
    }


def wang0701164_load_vector_fig5_curves(svg_path: str | Path) -> dict[str, object]:
    """Load Fig. 5 reference curves from a vector SVG dump of the paper page."""

    ref = Path(svg_path)
    parsed = _wang0701164_fig5_vector_reference_from_svg_text(ref.read_text(encoding="utf-8"))
    parsed["path"] = str(ref)
    parsed["source"] = "svg"
    return parsed


def wang0701164_extract_fig5_vector_curves(pdf_path: str | Path, *, page: int = 10) -> dict[str, object]:
    """Extract Fig. 5 reference curves directly from the paper PDF via SVG."""

    ref = Path(pdf_path)
    if not ref.exists():
        raise FileNotFoundError(ref)
    with tempfile.TemporaryDirectory(prefix="wang0701164_fig5_") as tmpdir:
        base = Path(tmpdir) / f"page{int(page)}"
        subprocess.run(
            [
                "pdftocairo",
                "-f",
                str(int(page)),
                "-l",
                str(int(page)),
                "-svg",
                str(ref),
                str(base),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        svg_path = base
        parsed = _wang0701164_fig5_vector_reference_from_svg_text(svg_path.read_text(encoding="utf-8"))
    parsed["path"] = str(ref)
    parsed["page"] = int(page)
    parsed["source"] = "pdf-vector"
    return parsed


def _wang0701164_compare_sweep_to_reference(
    result: dict[str, object],
    reference: dict[str, object],
    *,
    reference_path: str,
) -> dict[str, object]:
    t_model = np.asarray(result["temperatures"], dtype=float)
    ref_curves = reference["curves"]
    ref_curve_temperatures = reference.get("curve_temperatures")
    metrics: dict[str, object] = {
        "reference": reference_path,
        "score": 0.0,
        "curves": {},
        "sweep_parameters": {
            "conductance_mode": result.get("conductance_mode"),
            "delta_t": result.get("delta_t"),
            "include_second_graph": result.get("include_second_graph"),
        },
    }

    total_score = 0.0
    for label, curve in result["curves"].items():
        y_model = np.asarray(curve["conductance"], dtype=float)
        y_ref = np.asarray(ref_curves[label], dtype=float)
        if isinstance(ref_curve_temperatures, dict) and label in ref_curve_temperatures:
            t_ref = np.asarray(ref_curve_temperatures[label], dtype=float)
        else:
            t_ref = np.asarray(reference["temperatures"], dtype=float)
        y_interp = np.interp(t_ref, t_model, y_model)
        rmse = float(np.sqrt(np.mean((y_interp - y_ref) ** 2)))
        peak_model = float(np.max(y_interp) * 1.0e9)
        peak_ref = float(np.max(y_ref) * 1.0e9)
        total_score += rmse / max(float(np.max(y_ref)), 1e-30)
        metrics["curves"][label] = {
            "rmse_W_per_K": rmse,
            "peak_model_1e-9_W_per_K": peak_model,
            "peak_digitized_1e-9_W_per_K": peak_ref,
        }

    metrics["score"] = total_score
    return metrics


def wang0701164_compare_sweep_to_digitized(
    result: dict[str, object],
    digitized_path: str | Path,
) -> dict[str, object]:
    """Compare a Wang Fig. 5 sweep against digitized literature curves."""

    digitized = wang0701164_load_digitized_fig5_curves(digitized_path)
    return _wang0701164_compare_sweep_to_reference(
        result,
        digitized,
        reference_path=str(Path(digitized_path)),
    )


def wang0701164_compare_sweep_to_vector_pdf(
    result: dict[str, object],
    pdf_path: str | Path,
    *,
    page: int = 10,
) -> dict[str, object]:
    """Compare a Wang Fig. 5 sweep against vector-extracted curves from the PDF."""

    reference = wang0701164_extract_fig5_vector_curves(pdf_path, page=page)
    return _wang0701164_compare_sweep_to_reference(
        result,
        reference,
        reference_path=f"{Path(pdf_path)}#page={int(page)}",
    )
