"""Pointwise exact Dyson/Keldysh increments and their analytic tangent.

This arithmetic module does not import or modify the underlying solver.  A reference is the
full device operator K0 and two independent sources SC0 and SD0.  If
K = K0 - dSigmaR, then R = K**-1 and Y = R SY R**H.  The routines below retain
all full-device matrix entries; active blocks are embedded before the solve.

The independent spectral field is never reconstructed as R-R**H, and no field
is symmetrized, projected, clipped, or repaired.  Finiteness and shape checks
are arithmetic requirements only.  The caller must apply the unchanged
physical checks and convergence tests to every evaluated state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def adjoint(value):
    return value.conj().T


def _matrix(value, n, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (n, n):
        raise ValueError(f"{name} must have shape {(n, n)}.")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite.")
    return result


def _readonly(value):
    result = np.array(value, dtype=np.complex128, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("The Green arithmetic produced a nonfinite matrix.")
    result.setflags(write=False)
    return result


def _sum(*values):
    """Compensated addition; never obtain an increment by subtraction.

    Knuth's two-sum acts componentwise on the real and imaginary doubles.
    Compensation reduces assembly loss but cannot make an ill-conditioned
    Dyson problem well-conditioned or recover prior product-rounding errors.
    """
    total = np.zeros_like(values[0], dtype=np.complex128)
    correction = np.zeros_like(total)
    for value in values:
        combined = total + value
        virtual_value = combined - total
        correction += ((total - (combined - virtual_value))
                       + (value - virtual_value))
        total = combined
    return total + correction


def _embedded(value, n, active_dofs, name):
    if active_dofs is None:
        return _matrix(value, n, name)
    active = np.asarray(active_dofs)
    if (active.ndim != 1 or active.dtype.kind not in "iu"
            or len(np.unique(active)) != len(active)
            or np.any(active < 0) or np.any(active >= n)):
        raise ValueError("active_dofs must be unique in-range integer indices.")
    block = _matrix(value, len(active), name)
    result = np.zeros((n, n), dtype=np.complex128)
    result[np.ix_(active, active)] = block
    return result


@dataclass(frozen=True, eq=False)
class ReferencePoint:
    operator: np.ndarray
    source_c: np.ndarray
    source_d: np.ndarray
    retarded: np.ndarray
    statistical: np.ndarray
    spectral: np.ndarray


@dataclass(frozen=True, eq=False)
class IncrementPoint:
    operator: np.ndarray
    source_c: np.ndarray
    source_d: np.ndarray
    retarded: np.ndarray
    statistical: np.ndarray
    spectral: np.ndarray
    delta_retarded: np.ndarray
    delta_statistical: np.ndarray
    delta_spectral: np.ndarray
    diagnostics: dict


@dataclass(frozen=True, eq=False)
class TangentPoint:
    retarded: np.ndarray
    statistical: np.ndarray
    spectral: np.ndarray


def prepare_reference(operator, source_c, source_d, *, retarded=None):
    """Evaluate or accept the full-device reference at one (q, frequency).

    An optional ``retarded`` is an already evaluated full reference inverse,
    useful when the map also needs that same reference for its current.  No
    material or checkpoint arrays are loaded by this module.
    """
    raw = np.asarray(operator)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1] or not raw.shape[0]:
        raise ValueError("operator must be a nonempty square matrix.")
    n = raw.shape[0]
    k0, sc0, sd0 = (_matrix(value, n, name) for value, name in
                     ((operator, "operator"), (source_c, "source_c"),
                      (source_d, "source_d")))
    try:
        r0 = (np.linalg.solve(k0, np.eye(n, dtype=np.complex128))
              if retarded is None else _matrix(retarded, n, "retarded"))
    except np.linalg.LinAlgError as exc:
        raise ValueError("The reference Dyson operator is singular.") from exc
    a0 = adjoint(r0)
    with np.errstate(over="raise", invalid="raise"):
        c0, d0 = r0 @ sc0 @ a0, r0 @ sd0 @ a0
    return ReferencePoint(*(_readonly(value) for value in
                            (k0, sc0, sd0, r0, c0, d0)))


def evaluate_increment(reference, delta_r, delta_c, delta_d, *, active_dofs=None):
    """Evaluate an exact increment about a *fixed* reference point.

    ``delta_r/c/d`` are changes in the self-energy/source, not changes in
    Green functions.  They may be full-device matrices or the ordered active
    block specified by ``active_dofs``.  All returns are full-device matrices.

    R is solved from (I-R0 dSigmaR) R=R0.  Its increment is formed as
    dR=R0 dSigmaR R, avoiding the subtraction R-R0.  For either Y=C or D,
    P=R dSigmaR and Q=P**H give

        dY=P Y0 + Y0 Q + P Y0 Q + R dSY R**H.

    The total is assembled directly from the reference and these exact terms.
    Reference subtraction by itself does not change fixed-point eigenvalues.
    """
    if not isinstance(reference, ReferencePoint):
        raise TypeError("reference must be ReferencePoint.")
    n = reference.retarded.shape[0]
    drs, dcs, dds = (_embedded(value, n, active_dofs, name)
                     for value, name in ((delta_r, "delta_r"),
                                         (delta_c, "delta_c"),
                                         (delta_d, "delta_d")))
    r0 = reference.retarded
    identity = np.eye(n, dtype=np.complex128)
    with np.errstate(over="raise", invalid="raise"):
        anchored = identity - r0 @ drs
        try:
            solved_r = np.linalg.solve(anchored, r0)
        except np.linalg.LinAlgError as exc:
            raise ValueError("The anchored current Dyson operator is singular.") from exc
        delta_green_r = r0 @ drs @ solved_r
        r = _sum(r0, delta_green_r)
        a = adjoint(r)
        p, q = r @ drs, adjoint(r @ drs)
        increments, totals = [], []
        for reference_y, delta_source in (
                (reference.statistical, dcs), (reference.spectral, dds)):
            terms = (p @ reference_y, reference_y @ q,
                     p @ reference_y @ q, r @ delta_source @ a)
            increments.append(_sum(*terms))
            totals.append(_sum(reference_y, *terms))
        operator = _sum(reference.operator, -drs)
        sc, sd = (_sum(reference.source_c, dcs), _sum(reference.source_d, dds))
        denominator = max(float(np.linalg.norm(operator, np.inf)
                                * np.linalg.norm(r, np.inf)), 1.0)
        diagnostics = {
            "dyson_left_backward_residual": float(
                np.linalg.norm(operator @ r - identity, np.inf) / denominator),
            "dyson_right_backward_residual": float(
                np.linalg.norm(r @ operator - identity, np.inf) / denominator),
            "anchored_assembly_relative_gap": float(
                np.linalg.norm(r - solved_r, np.inf)
                / max(float(np.linalg.norm(solved_r, np.inf)), 1.0)),
        }
    return IncrementPoint(*(_readonly(value) for value in
                            (operator, sc, sd, r, *totals,
                             delta_green_r, *increments)), diagnostics)


def tangent_green(current, direction_r, direction_c, direction_d, *, active_dofs=None):
    """Exact analytic Green derivative in one joint R/C/D direction.

    dR=R dSigmaR R and
    dY=R dSigmaR Y + Y dSigmaR**H R**H + R dSY R**H.
    Both C and D derivatives use their independently propagated current Y.
    A single shared *real* coefficient should combine directions in an SCBA
    proposal; this routine imposes no proposal/acceptance policy on the caller.
    """
    if not isinstance(current, (ReferencePoint, IncrementPoint)):
        raise TypeError("current must be ReferencePoint or IncrementPoint.")
    r = current.retarded
    n = r.shape[0]
    drs, dcs, dds = (_embedded(value, n, active_dofs, name)
                     for value, name in ((direction_r, "direction_r"),
                                         (direction_c, "direction_c"),
                                         (direction_d, "direction_d")))
    with np.errstate(over="raise", invalid="raise"):
        a, p = adjoint(r), r @ drs
        q = adjoint(p)
        dr = p @ r
        derivatives = [_sum(p @ y, y @ q, r @ source @ a)
                       for y, source in ((current.statistical, dcs),
                                         (current.spectral, dds))]
    return TangentPoint(*(_readonly(value) for value in (dr, *derivatives)))
