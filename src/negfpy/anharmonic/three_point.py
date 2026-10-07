"""Opt-in one-vertex connected cubic moments in real canonical solver units.

The causal Kubo/Wick contraction retains the quantum three-commutator term.
It accepts Gaussian or dressed two-point lines, but does not certify a dressed
SCBA closure, a stationary infinite-time limit, or transport conservation.
Static means and their disconnected contributions are explicit separate steps.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np

from .cubic_cut_current import _collect, _real
from .grid import SignedFrequencyGrid
from .sparse_cubic import SparseCubicVertex
from .types import DenseCubicVertex, KeldyshSpectrum


def _potential(vertex):
    if isinstance(vertex, SparseCubicVertex):
        indices, coefficients = vertex.indices, vertex.coefficients
    elif isinstance(vertex, DenseCubicVertex):
        indices = np.argwhere(vertex.values != 0.0)
        coefficients = vertex.values[tuple(indices.T)]
    else:
        raise TypeError("vertex must be a dense or sparse cubic vertex.")
    indices, coefficients = _collect(np.sort(indices, axis=1), coefficients / 6.0, 3)
    return vertex.ndof, indices, coefficients


def _lags(values):
    raw = np.asarray(values)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("lag_times must be a vector with at least two points.")
    result = _real(raw, raw.shape, "lag_times")
    if result[0] != 0.0 or not np.all(np.diff(result) > 0.0):
        raise ValueError("lag_times must start at zero and strictly increase.")
    return result


def _indices(values, ndof):
    result = np.asarray(values)
    if (result.ndim != 2 or result.shape[1] != 3 or result.dtype.kind not in "iu"
            or np.any(result < 0) or np.any(result >= ndof)):
        raise ValueError("moment_indices must be integer rows (momentum, displacement, displacement).")
    return result.astype(np.int64)


def _complex(values, shape, name):
    raw = np.asarray(values)
    if raw.shape != shape or raw.dtype.kind not in "fiuc":
        raise ValueError(f"{name} must be a numeric array of shape {shape}.")
    with np.errstate(over="ignore"):
        result = np.asarray(raw, dtype=np.complex128)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite in complex128.")
    return result


def connected_cubic_moments(vertex, moment_indices, lag_times,
                            displacement_correlation, velocity_displacement_correlation):
    r"""Integrate connected <W(p_i*u_j*u_k)> at one cubic vertex.

    C[t,i,a]=<x_i(t)x_a(0)> and P[t,i,a]=<p_i(t)x_a(0)>=d_t C
    are *centered*, unsymmetrized correlations. For symmetric Psi the kernel
    is 2 Im sum_abc Psi_abc P_ia C_jb C_kc. Every ordered input coefficient
    enters through all six Wick pairings, including repeated-index factors.
    Only the supplied finite lag interval is integrated, using trapezoids.
    The result is a finite-history response. Any disconnected means must be
    drawn from the same history; static means require a separate stationary limit.
    No damping, tail estimate, finite-time pass, or zero mean is inferred.
    """
    ndof, monomials, coefficients = _potential(vertex)
    indices = _indices(moment_indices, ndof)
    lag = _lags(lag_times)
    shape = (len(lag), ndof, ndof)
    correlation = _complex(displacement_correlation, shape, "displacement_correlation")
    derivative = _complex(velocity_displacement_correlation, shape,
                          "velocity_displacement_correlation")
    moments = np.zeros(len(indices))
    endpoint = np.zeros(len(indices))
    # O(N_time * tile) temporary memory, no dense rank-three vertex or moment tensor.
    for target, (i, j, k) in enumerate(indices):
        kernel = np.zeros(len(lag), dtype=np.complex128)
        for start in range(0, len(monomials), 256):
            terms = monomials[start:start + 256]
            weights = coefficients[start:start + 256]
            for order in permutations(range(3)):
                a, b, c = terms[:, order].T
                kernel += np.sum(derivative[:, i, a] * correlation[:, j, b]
                                 * correlation[:, k, c] * weights, axis=1)
        integrand = 2.0 * kernel.imag
        moments[target] = np.trapezoid(integrand, lag)
        endpoint[target] = integrand[-1]
    if not np.all(np.isfinite(moments)) or not np.all(np.isfinite(endpoint)):
        raise ValueError("Connected cubic moment contraction overflowed.")
    return {
        "connected_moments": moments,
        "endpoint_integrand": endpoint,
        "approximation": "connected-one-cubic-vertex-with-supplied-two-point-lines",
        "quadrature": "finite-lag-trapezoid",
        "lag_end": float(lag[-1]),
        "lag_points": len(lag),
        "quantum_three_commutator_term_included": True,
        "disconnected_means_included": False,
        "tail_error_estimated": False,
        "quadrature_convergence_verified": False,
        "scba_self_consistency_verified": False,
        "spatial_validation_passed": False,
        "changes_solver_acceptance": False,
    }


def correlations_from_green(grid, green, lag_times):
    r"""Finite signed-grid Fourier sums for centered C=iG> and d_t C.

    This is a direct rectangle sum, not an FFT or an infinite-time reconstruction.
    The user must supply connected two-point functions (no zero-frequency mean
    delta). Uniform discrete frequencies recur at 2*pi/step: lag_end must be
    below half this period to avoid interpreting a full recurrence as decay.
    This guard does not establish adequate decay or spectral accuracy.
    """
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(green, KeldyshSpectrum):
        raise TypeError("grid and green must be signed-grid and Keldysh spectrum objects.")
    if green.shape[0] != grid.n_points:
        raise ValueError("Green function and grid sizes differ.")
    lag = _lags(lag_times)
    recurrence = 2.0 * np.pi / grid.step
    if lag[-1] >= recurrence / 2.0:
        raise ValueError("lag window must be shorter than half the discrete-frequency recurrence.")
    shape = (len(lag), *green.shape[1:])
    correlation = np.empty(shape, dtype=np.complex128)
    derivative = np.empty(shape, dtype=np.complex128)
    weighted = grid.integration_weight * green.greater
    for start in range(0, len(lag), 32):
        stop = min(start + 32, len(lag))
        phase = np.exp(-1j * np.outer(lag[start:stop], grid.values))
        correlation[start:stop] = 1j * np.einsum("tw,wij->tij", phase, weighted)
        derivative[start:stop] = np.einsum("tw,w,wij->tij", phase, grid.values, weighted)
    if not np.all(np.isfinite(correlation)) or not np.all(np.isfinite(derivative)):
        raise ValueError("Fourier correlation reconstruction overflowed.")
    return {
        "displacement_correlation": correlation,
        "velocity_displacement_correlation": derivative,
        "frequency_quadrature": "stored-signed-grid-rectangle",
        "recurrence_period": float(recurrence),
        "input_centering_verified": False,
        "frequency_accuracy_verified": False,
    }


def combine_cubic_moments(moment_indices, connected_moments, *, displacement_mean,
                          velocity_mean, displacement_covariance,
                          velocity_displacement_covariance):
    r"""Add explicit disconnected means to a centered third Weyl cumulant.

    Total = connected + pi_i F_jk + mu_j Q_ik + mu_k Q_ij + pi_i mu_j mu_k,
    where F and Q are centered symmetrized uu and pu covariances. Means are
    mandatory; explicit zeros describe a centered model, not solved tadpoles.
    For nonzero velocity means harmonic pu moments also need pi_i*mu_j.
    """
    raw = np.asarray(displacement_mean)
    if raw.ndim != 1 or len(raw) == 0:
        raise ValueError("displacement_mean must be a nonempty vector.")
    ndof = len(raw)
    mu = _real(raw, (ndof,), "displacement_mean")
    pi = _real(velocity_mean, (ndof,), "velocity_mean")
    covariance = _real(displacement_covariance, (ndof, ndof), "displacement_covariance")
    mixed = _real(velocity_displacement_covariance, (ndof, ndof),
                  "velocity_displacement_covariance")
    indices = _indices(moment_indices, ndof)
    connected = _real(connected_moments, (len(indices),), "connected_moments")
    if not np.allclose(covariance, covariance.T, rtol=1e-12, atol=1e-14):
        raise ValueError("displacement_covariance must be symmetric.")
    i, j, k = indices.T
    disconnected = (pi[i] * covariance[j, k] + mu[j] * mixed[i, k]
                    + mu[k] * mixed[i, j] + pi[i] * mu[j] * mu[k])
    total = connected + disconnected
    if not np.all(np.isfinite(total)) or not np.all(np.isfinite(disconnected)):
        raise ValueError("Disconnected moment assembly overflowed.")
    return {"connected_moments": connected.copy(), "disconnected_moments": disconnected,
            "total_moments": total, "mean_policy": "explicit-caller-supplied",
            "static_mean_equation_verified": False, "spatial_validation_passed": False}


def static_cubic_shift(vertex, displacement_covariance, static_stiffness):
    r"""Leading static force/shift diagnostic: K_static*mu=-<grad H3>_Gaussian.

    K_static is supplied explicitly: include the static contact Schur complement
    for relaxed harmonic leads, or declare clamped boundaries. Translational
    zero modes must be constrained by the caller, never pseudoinverted here.
    This diagnostic does not change the SCBA background or add its O(Psi^2)
    harmonic feedback. Therefore adding its mean alone is not full mean-field
    consistency at the order of the cubic current.
    """
    ndof, monomials, coefficients = _potential(vertex)
    covariance = _real(displacement_covariance, (ndof, ndof), "displacement_covariance")
    stiffness = _real(static_stiffness, (ndof, ndof), "static_stiffness")
    for name, matrix in (("displacement_covariance", covariance), ("static_stiffness", stiffness)):
        if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-14):
            raise ValueError(f"{name} must be symmetric.")
    # Require a stable constrained static reference; no silent acoustic pseudoinverse.
    if np.linalg.eigvalsh(stiffness)[0] <= 0.0:
        raise ValueError("static_stiffness must be positive definite after explicit constraints.")
    force = np.zeros(ndof)
    for leg in range(3):
        other = [slot for slot in range(3) if slot != leg]
        a, b = monomials[:, other].T
        np.add.at(force, monomials[:, leg], -coefficients * covariance[a, b])
    mean = np.linalg.solve(stiffness, force)
    residual = stiffness @ mean - force
    feedback = np.zeros((ndof, ndof))
    for order in permutations(range(3)):
        a, b, c = monomials[:, order].T
        np.add.at(feedback, (a, b), coefficients * mean[c])
    if (not np.all(np.isfinite(mean)) or not np.all(np.isfinite(residual))
            or not np.all(np.isfinite(feedback))):
        raise ValueError("Static shift evaluation overflowed.")
    return {"mean_force": force, "displacement_mean": mean,
            "induced_stiffness": feedback,
            "force_balance_residual": residual,
            "stiffness_condition_number": float(np.linalg.cond(stiffness)),
            "approximation": "leading-cubic-static-shift",
            "harmonic_feedback_included": False,
            "changes_solver_acceptance": False}


__all__ = ["connected_cubic_moments", "correlations_from_green",
           "combine_cubic_moments", "static_cubic_shift"]
