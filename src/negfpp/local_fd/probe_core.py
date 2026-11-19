"""Frozen-spectrum thermal probe currents and Bose emission weights.

Frequencies are dimensionless angular frequencies omega_physical / omega0.
The caller supplies positive-frequency quadrature weights including 1/(2*pi)
and averages transmissions over transverse q exactly once.  For the existing
SignedFrequencyGrid rectangle convention, omit zero and give every strictly
positive node step/(2*pi); zero is not given a half trapezoid weight here.

All terminal currents are positive from that terminal into the device.  The
retarded field and transmissions stay fixed while probe temperatures change.
No imaginary device eta, PSD projection, reciprocity repair, or Born map is
introduced.  A broadening block retains all its off-diagonal coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

HBAR_J_S = 1.0545718176461565e-34
KB_J_K = 1.380649e-23


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _units(omega0, hbar, kb):
    return tuple(_positive(value, name) for value, name in
                 ((omega0, "omega0"), (hbar, "hbar"), (kb, "kb")))


def bose_energy_weight(omega, temperature_kelvin, *, omega0, hbar=HBAR_J_S,
                       kb=KB_J_K):
    """Return omega*n_B in solver units, including its continuous zero limit.

Inputs follow NumPy broadcasting.  Pass omega[:,None] and T[None,:] for a
frequency by terminal table.  At omega=0 the value is k_B*T/(hbar*omega0),
which avoids an infinite Bose occupation multiplied by a stored zero.
"""
    omega0, hbar, kb = _units(omega0, hbar, kb)
    w, t = np.broadcast_arrays(np.asarray(omega, dtype=float),
                               np.asarray(temperature_kelvin, dtype=float))
    if not np.isfinite(w).all() or not np.isfinite(t).all() or np.any(w < 0) or np.any(t < 0):
        raise ValueError("Frequencies and Kelvin temperatures must be finite and nonnegative.")
    theta = t * (kb / (hbar * omega0))
    result = np.zeros(w.shape, dtype=float)
    zero = w == 0
    result[zero] = theta[zero]
    active = (w > 0) & (t > 0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="raise"):
        x = w[active] / theta[active]
        result[active] = w[active] * np.exp(-x) / (-np.expm1(-x))
    return float(result) if result.ndim == 0 else result


def bose_energy_temperature_derivative(omega, temperature_kelvin, *, omega0,
                                        hbar=HBAR_J_S, kb=KB_J_K):
    """Kelvin derivative of :func:`bose_energy_weight`, with analytic zero."""
    omega0, hbar, kb = _units(omega0, hbar, kb)
    w, t = np.broadcast_arrays(np.asarray(omega, dtype=float),
                               np.asarray(temperature_kelvin, dtype=float))
    if not np.isfinite(w).all() or not np.isfinite(t).all() or np.any(w < 0) or np.any(t < 0):
        raise ValueError("Frequencies and Kelvin temperatures must be finite and nonnegative.")
    conversion = kb / (hbar * omega0)
    result = np.zeros(w.shape, dtype=float)
    result[w == 0] = conversion
    active = (w > 0) & (t > 0)
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        x = w[active] / (conversion * t[active])
        ratio = np.ones(x.shape)
        regular = x > 1e-5
        moderate = regular & (x < 700)
        ratio[regular] = 0.
        ratio[moderate] = (x[moderate] / (-np.expm1(-x[moderate])))**2 * np.exp(-x[moderate])
        small = ~regular
        ratio[small] = 1 - x[small]**2 / 12 + x[small]**4 / 240
        result[active] = conversion * ratio
    return float(result) if result.ndim == 0 else result


def spectral_completeness(retarded, gamma_blocks, supports=None):
    """Relative optical-identity error; zero means every damping bath is counted.

Returns max|i(GR-GA)-GR*sum(Gamma)*GA| divided by the larger maximum
absolute matrix entry on the two sides.  A zero identity has zero error.
No transmission reciprocity assumption is used.
    """
    green = np.asarray(retarded, dtype=np.complex128)
    if green.ndim != 2 or green.shape[0] != green.shape[1] or not green.size or not np.isfinite(green).all():
        raise ValueError("retarded must be a finite nonempty square matrix.")
    blocks = tuple(np.asarray(block, dtype=np.complex128) for block in gamma_blocks)
    if supports is None:
        supports = tuple(np.arange(len(green)) for _ in blocks)
    if len(supports) != len(blocks):
        raise ValueError("One support is required for each Gamma block.")
    total = np.zeros_like(green)
    for block, support in zip(blocks, supports, strict=True):
        index = np.asarray(support)
        if (index.ndim != 1 or index.dtype.kind not in "iu" or not len(index)
                or len(np.unique(index)) != len(index) or np.any(index < 0)
                or np.any(index >= len(green)) or block.shape != (len(index), len(index))
                or not np.isfinite(block).all()):
            raise ValueError("Invalid Gamma block or support.")
        total[np.ix_(index, index)] += block
    spectral = 1j*(green-green.conj().T)
    propagated = green @ total @ green.conj().T
    scale = max(float(np.max(np.abs(spectral))), float(np.max(np.abs(propagated))))
    error = float(np.max(np.abs(spectral-propagated)))
    return error/scale if scale else (0. if error == 0 else float("inf"))


def transmission_matrix(retarded, gamma_blocks: Sequence[np.ndarray],
                        supports: Sequence[Sequence[int]] | None = None, *,
                        validate_passivity=True, validate_spectral_identity=True,
                        rtol=1e-8, atol=1e-11):
    """Return T_ab=Tr[Gamma_a G^R Gamma_b G^A] for one q/frequency.

Blocks are embedded on their ordered ``supports``.  Supports may overlap.
When omitted, each Gamma must already be a full-device matrix.  Passivity is
validated without clipping eigenvalues.  At an individual transverse q the
    matrix need not be reciprocal; average q and -q before judging reciprocity.
The default spectral-completeness check requires that the supplied terminal
blocks include every anti-Hermitian term in the Dyson operator.  It detects
an omitted device-eta sink even when transmissions remain reciprocal.
    """
    green = np.asarray(retarded, dtype=np.complex128)
    if green.ndim != 2 or green.shape[0] != green.shape[1] or not green.size or not np.isfinite(green).all():
        raise ValueError("retarded must be a finite nonempty square matrix.")
    n = len(green)
    blocks = tuple(np.asarray(block, dtype=np.complex128) for block in gamma_blocks)
    if not blocks:
        raise ValueError("At least one terminal is required.")
    if supports is None:
        supports = tuple(np.arange(n) for _ in blocks)
    if len(supports) != len(blocks):
        raise ValueError("One support is required for each Gamma block.")
    indices = []
    for block, support in zip(blocks, supports, strict=True):
        index = np.asarray(support)
        if (index.ndim != 1 or index.dtype.kind not in "iu" or not len(index)
                or len(np.unique(index)) != len(index) or np.any(index < 0) or np.any(index >= n)
                or block.shape != (len(index), len(index)) or not np.isfinite(block).all()):
            raise ValueError("Gamma blocks need finite square matrices on unique in-range supports.")
        magnitude = float(np.max(np.abs(block)))
        if np.max(np.abs(block-block.conj().T)) > atol + rtol*magnitude:
            raise ValueError("Gamma must be Hermitian.")
        if validate_passivity:
            eigenvalues = np.linalg.eigvalsh(block)
            if eigenvalues.min() < -(atol + rtol*max(float(np.abs(eigenvalues).max()), magnitude)):
                raise ValueError("Gamma is not passive; no PSD repair is applied.")
        indices.append(index)
    if validate_spectral_identity:
        total_gamma = np.zeros_like(green)
        for block, index in zip(blocks, indices, strict=True):
            total_gamma[np.ix_(index, index)] += block
        spectral = 1j*(green-green.conj().T)
        propagated = green @ total_gamma @ green.conj().T
        scale = max(float(np.max(np.abs(spectral))), float(np.max(np.abs(propagated))))
        if np.max(np.abs(spectral-propagated)) > atol + rtol*scale:
            raise ValueError("Incomplete terminal spectral partition: an unrepresented sink or Dyson defect exists.")
    result = np.zeros((len(blocks), len(blocks)), dtype=float)
    for a, (gamma_a, ia) in enumerate(zip(blocks, indices, strict=True)):
        for b, (gamma_b, ib) in enumerate(zip(blocks, indices, strict=True)):
            if a == b:
                continue
            between = green[np.ix_(ia, ib)]
            value = np.trace(gamma_a @ between @ gamma_b @ between.conj().T)
            if abs(value.imag) > atol + rtol*abs(value.real):
                raise ValueError("Transmission has a material imaginary residue.")
            result[a, b] = value.real
    if result.min() < -(atol + rtol*max(float(np.max(np.abs(result))), 1.)):
        raise ValueError("Negative terminal transmission; no clipping is applied.")
    return result


def _transport_inputs(omega, weights, transmissions, temperatures):
    w, quad = np.asarray(omega, dtype=float), np.asarray(weights, dtype=float)
    trans, temp = np.asarray(transmissions, dtype=float), np.asarray(temperatures, dtype=float)
    if (w.ndim != 1 or not len(w) or quad.shape != w.shape or trans.ndim != 3
            or trans.shape[0] != len(w) or trans.shape[1] != trans.shape[2]
            or temp.shape != (trans.shape[1],)):
        raise ValueError("Expected omega/weights (nw,), transmissions (nw,nt,nt), and temperatures (nt,).")
    if (not all(np.isfinite(a).all() for a in (w, quad, trans, temp))
            or np.any(w < 0) or np.any(quad < 0) or np.any(temp < 0)):
        raise ValueError("Transport data must be finite, with nonnegative frequencies, weights, and temperatures.")
    tolerance = 1e-12 + 1e-9*max(float(np.max(np.abs(trans))), 1.)
    if trans.min() < -tolerance:
        raise ValueError("Transmissions cannot be negative.")
    return w, quad, trans, temp


def terminal_currents(omega, weights, transmissions, temperatures_kelvin, *,
                      omega0, hbar=HBAR_J_S, kb=KB_J_K):
    """Frozen-spectrum terminal currents in W, positive into the device.

    This is sum_b T_ab*(n_a-n_b), valid for a lossless terminal partition whose
row and column sums agree; :func:`solve_probe_temperatures` checks this
identity.  The transmission row/column test alone cannot reveal every omitted
sink; use the spectral identity checked by :func:`transmission_matrix` too.
    """
    omega0, hbar, kb = _units(omega0, hbar, kb)
    w, quad, trans, temp = _transport_inputs(omega, weights, transmissions, temperatures_kelvin)
    energy = bose_energy_weight(w[:, None], temp[None, :], omega0=omega0, hbar=hbar, kb=kb)
    spectral = trans.sum(axis=2)*energy - np.einsum("wab,wb->wa", trans, energy)
    return hbar*omega0**2 * np.einsum("w,wa->a", quad, spectral)


def current_temperature_jacobian(omega, weights, transmissions, temperatures_kelvin,
                                 *, omega0, hbar=HBAR_J_S, kb=KB_J_K):
    """Analytic derivative d I_a / d T_b in W/K for frozen transmissions."""
    omega0, hbar, kb = _units(omega0, hbar, kb)
    w, quad, trans, temp = _transport_inputs(omega, weights, transmissions, temperatures_kelvin)
    derivative = bose_energy_temperature_derivative(w[:, None], temp[None, :],
                                                    omega0=omega0, hbar=hbar, kb=kb)
    jacobian = -np.einsum("w,wab,wb->ab", quad, trans, derivative)
    diagonal = np.einsum("w,wa,wa->a", quad, trans.sum(axis=2), derivative)
    jacobian[np.diag_indices_from(jacobian)] += diagonal
    return hbar*omega0**2 * jacobian


def transmission_diagnostics(transmissions):
    """Scalar reciprocity and lossless row/column partition defects."""
    trans = np.asarray(transmissions, dtype=float)
    if trans.ndim != 3 or trans.shape[1] != trans.shape[2] or not np.isfinite(trans).all():
        raise ValueError("transmissions must be finite with shape (nw,nt,nt).")
    scale = max(float(np.max(np.abs(trans))), np.finfo(float).tiny)
    return {"maximum_transmission": scale,
            "minimum_transmission": float(trans.min()),
            "reciprocity_max_abs": float(np.max(np.abs(trans-trans.swapaxes(1, 2)))),
            "row_column_max_abs": float(np.max(np.abs(trans.sum(axis=2)-trans.sum(axis=1)))),
            "row_column_relative": float(np.max(np.abs(trans.sum(axis=2)-trans.sum(axis=1)))) / scale}


def _partition(transmissions, weights, fixed_temperatures):
    nt = transmissions.shape[1]
    if not isinstance(fixed_temperatures, Mapping) or not fixed_temperatures:
        raise ValueError("fixed_temperatures must map boundary terminal indices to temperatures.")
    boundary = np.asarray(sorted(fixed_temperatures), dtype=int)
    if (any(isinstance(i, bool) or not isinstance(i, (int, np.integer)) for i in fixed_temperatures)
            or np.any(boundary < 0) or np.any(boundary >= nt)):
        raise ValueError("Boundary indices must be in-range integers.")
    probes = np.asarray([i for i in range(nt) if i not in fixed_temperatures], dtype=int)
    diagnostic = transmission_diagnostics(transmissions)
    if diagnostic["row_column_max_abs"] > 1e-11 + 1e-8*diagnostic["maximum_transmission"]:
        raise ValueError("Terminal transmission partition is lossy or inconsistent (row/column sum defect).")
    adjacency = np.einsum("w,wab->ab", weights, transmissions)
    reachable = set(int(i) for i in boundary)
    while True:
        new = {int(j) for i in reachable for j in np.flatnonzero((adjacency[i]+adjacency[:, i]) > 0)}
        if new.issubset(reachable):
            break
        reachable.update(new)
    if any(int(i) not in reachable for i in probes):
        raise ValueError("A probe component is disconnected from every boundary terminal.")
    return boundary, probes, diagnostic


@dataclass(frozen=True)
class ProbeSolution:
    temperatures_kelvin: np.ndarray
    currents_w: np.ndarray
    boundary_indices: np.ndarray
    probe_indices: np.ndarray
    max_probe_flux_w: float
    contact_balance_w: float
    total_balance_w: float
    success: bool
    message: str
    nfev: int
    transmission_checks: dict


def solve_probe_temperatures(omega, weights, transmissions, fixed_temperatures,
                             *, omega0, hbar=HBAR_J_S, kb=KB_J_K,
                             initial_temperatures_kelvin=None, max_nfev=200,
                             flux_rtol=1e-10, flux_atol_w=1e-25):
    """Solve integrated zero probe heat currents with bounded Kelvin T.

``fixed_temperatures={0:300.5, 1:299.5}`` leaves all later terminals as probes.
Each probe is bounded by the minimum/maximum boundary temperature, as required
by the passive thermal-network maximum principle.  Equal boundaries give the
equilibrium solution directly.  Numerical convergence is reported separately
from the final physical flux tolerance; failed solutions are not certified.
    """
    nt = np.asarray(transmissions).shape[-1]
    provisional = np.full(nt, 300., dtype=float)
    w, quad, trans, _ = _transport_inputs(omega, weights, transmissions, provisional)
    boundary, probes, diagnostic = _partition(trans, quad, fixed_temperatures)
    fixed = np.asarray([fixed_temperatures[int(i)] for i in boundary], dtype=float)
    if not np.isfinite(fixed).all() or np.any(fixed < 0):
        raise ValueError("Boundary Kelvin temperatures must be finite and nonnegative.")
    lower, upper = float(fixed.min()), float(fixed.max())
    temperatures = np.full(nt, float(fixed.mean()))
    temperatures[boundary] = fixed
    kwargs = dict(omega0=omega0, hbar=hbar, kb=kb)
    nfev, numerical_success, message = 0, True, "No probe solve required."
    if len(probes) and upper > lower:
        if initial_temperatures_kelvin is not None:
            initial = np.asarray(initial_temperatures_kelvin, dtype=float)
            if initial.shape != (len(probes),) or not np.isfinite(initial).all() or np.any(initial < lower) or np.any(initial > upper):
                raise ValueError("Initial probe temperatures must lie within the boundary range.")
            temperatures[probes] = initial
        reference = current_temperature_jacobian(w, quad, trans, np.full(nt, fixed.mean()), **kwargs)
        flux_scale = max(float(np.max(np.abs(reference)))*(upper-lower), flux_atol_w, np.finfo(float).tiny)
        def residual(values):
            trial = temperatures.copy()
            trial[probes] = values
            return terminal_currents(w, quad, trans, trial, **kwargs)[probes] / flux_scale
        def jacobian(values):
            trial = temperatures.copy()
            trial[probes] = values
            return current_temperature_jacobian(w, quad, trans, trial, **kwargs)[np.ix_(probes, probes)] / flux_scale
        result = least_squares(residual, temperatures[probes], jac=jacobian,
                               bounds=(lower, upper), xtol=1e-13, ftol=1e-13,
                               gtol=1e-13, max_nfev=int(max_nfev))
        temperatures[probes] = result.x
        nfev, numerical_success, message = int(result.nfev), bool(result.success), str(result.message)
    currents = terminal_currents(w, quad, trans, temperatures, **kwargs)
    maximum = float(np.max(np.abs(currents[probes]))) if len(probes) else 0.
    flux_allowance = flux_atol_w + flux_rtol*max(float(np.max(np.abs(currents[boundary]))), flux_atol_w)
    return ProbeSolution(temperatures, currents, boundary, probes, maximum,
                         float(currents[boundary].sum()), float(currents.sum()),
                         bool(numerical_success and maximum <= flux_allowance), message,
                         nfev, diagnostic)


@dataclass(frozen=True)
class LinearResponseSolution:
    temperature_shifts_kelvin: np.ndarray
    currents_w: np.ndarray
    laplacian_w_per_kelvin: np.ndarray
    effective_boundary_laplacian_w_per_kelvin: np.ndarray
    boundary_indices: np.ndarray
    probe_indices: np.ndarray


def solve_linear_response(omega, weights, transmissions, fixed_temperature_shifts,
                           *, reference_temperature_kelvin=300., omega0,
                           hbar=HBAR_J_S, kb=KB_J_K):
    """Linear probe closure and boundary Schur conductance at a common T."""
    nt = np.asarray(transmissions).shape[-1]
    reference_temperature_kelvin = _positive(reference_temperature_kelvin, "reference_temperature_kelvin")
    w, quad, trans, temp = _transport_inputs(omega, weights, transmissions,
                                           np.full(nt, reference_temperature_kelvin))
    boundary, probes, _ = _partition(trans, quad, fixed_temperature_shifts)
    shifts = np.zeros(nt)
    shifts[boundary] = [float(fixed_temperature_shifts[int(i)]) for i in boundary]
    if not np.isfinite(shifts).all():
        raise ValueError("Boundary temperature shifts must be finite.")
    laplacian = current_temperature_jacobian(w, quad, trans, temp,
                                            omega0=omega0, hbar=hbar, kb=kb)
    effective = laplacian[np.ix_(boundary, boundary)].copy()
    if len(probes):
        pp, pb = laplacian[np.ix_(probes, probes)], laplacian[np.ix_(probes, boundary)]
        response = np.linalg.solve(pp, pb)
        shifts[probes] = -response @ shifts[boundary]
        effective -= laplacian[np.ix_(boundary, probes)] @ response
    return LinearResponseSolution(shifts, laplacian @ shifts, laplacian,
                                  effective, boundary, probes)
