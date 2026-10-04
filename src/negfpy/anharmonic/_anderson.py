"""Bounded, real-coefficient Anderson proposals for a Keldysh pair.

Only proposals are formed here. The solver accepts a proposal after evaluating
its actual fixed-point residual, and falls back to linear mixing on rejection.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from .diagnostics import antihermitian_residual, correlation_positivity_violation
from .grid import SignedFrequencyGrid

Pair = tuple[np.ndarray, np.ndarray]


def _dot(left: Pair, right: Pair) -> float:
    return float(sum(np.vdot(a, b).real for a, b in zip(left, right)))


def residual_scales(state: Pair, raw: Pair, absolute_floor: float) -> tuple[float, float]:
    """One scale per correlation field, frozen while assessing a trial."""
    return tuple(
        max(float(np.max(np.abs(a))), float(np.max(np.abs(b))),
            absolute_floor, np.finfo(float).tiny)
        for a, b in zip(state, raw)
    )


def _scaled(pair: Pair, scales: tuple[float, float]) -> Pair:
    normalization = np.sqrt(sum(a.size for a in pair))
    return tuple(a / scale / normalization for a, scale in zip(pair, scales))


def scaled_residual_merit(state: Pair, raw: Pair, scales: tuple[float, float]) -> float:
    """RMS defect in the exact metric used by the scaled Anderson fit."""
    residual = _scaled(tuple(b - a for a, b in zip(state, raw)), scales)
    return float(np.sqrt(_dot(residual, residual)))


class AndersonHistory:
    """Damped type-II Anderson with a regularized small real Gram system."""

    def __init__(self, depth: int, regularization: float, max_step_ratio: float,
                 *, policy: str = "legacy", trust_radius: float = 1.0,
                 rank_rtol: float = 1e-10, absolute_floor: float = 1e-12):
        self.states: deque[Pair] = deque(maxlen=depth + 1)
        self.residuals: deque[Pair] = deque(maxlen=depth + 1)
        self.regularization = regularization
        self.max_step_ratio = max_step_ratio
        self.policy = policy
        self.max_radius = trust_radius
        self.radius = trust_radius
        self.rank_rtol = rank_rtol
        self.absolute_floor = absolute_floor
        self.details: dict = {}

    def feedback(self, accepted: bool) -> None:
        if self.policy == "scaled-trust-v1":
            self.radius = (min(self.max_radius, 1.5 * self.radius) if accepted
                           else max(np.finfo(float).eps, 0.5 * self.radius))

    def _scaled_coefficients(self, df: list[Pair], residual: Pair,
                             scales: tuple[float, float]) -> np.ndarray | None:
        # Two-pass modified Gram-Schmidt in the real complex-vector inner
        # product avoids forming D^T D (which squares the condition number).
        # Only the depth-by-depth R matrix enters the SVD; no giant tall
        # concatenated real/imaginary matrix is materialized.
        basis: list[Pair] = []
        r = np.zeros((len(df), len(df)))
        for column, delta in enumerate(df):
            vector = _scaled(delta, scales)
            for _ in range(2):
                for row, unit in enumerate(basis):
                    coefficient = _dot(unit, vector)
                    r[row, column] += coefficient
                    vector = tuple(a - coefficient * b for a, b in zip(vector, unit))
            norm = float(np.sqrt(max(0.0, _dot(vector, vector))))
            if norm > np.finfo(float).tiny:
                r[len(basis), column] = norm
                basis.append(tuple(a / norm for a in vector))
        if not basis:
            return None
        r = r[:len(basis)]
        u, singular, vt = np.linalg.svd(r, full_matrices=False)
        keep = singular > self.rank_rtol * singular[0]
        self.details["retained_rank"] = int(np.count_nonzero(keep))
        if not np.any(keep):
            return None
        scaled_residual = _scaled(residual, scales)
        rhs = np.array([_dot(q, scaled_residual) for q in basis])
        ridge = self.regularization * singular[0] ** 2
        weights = singular[keep] / (singular[keep] ** 2 + ridge)
        return vt[keep].T @ (weights * (u[:, keep].T @ rhs))

    def clear(self) -> None:
        self.states.clear()
        self.residuals.clear()

    def propose(self, state: Pair, raw: Pair, mixing: float) -> tuple[Pair | None, str]:
        self.details = {}
        residual = tuple(b - a for a, b in zip(state, raw))
        self.states.append(tuple(np.array(a, copy=True) for a in state))
        self.residuals.append(residual)
        if len(self.states) < 2:
            return None, "insufficient_history"
        dx = [
            tuple(b - a for a, b in zip(x0, x1))
            for x0, x1 in zip(list(self.states)[:-1], list(self.states)[1:])
        ]
        df = [
            tuple(b - a for a, b in zip(f0, f1))
            for f0, f1 in zip(list(self.residuals)[:-1], list(self.residuals)[1:])
        ]
        try:
            if self.policy == "scaled-trust-v1":
                scales = residual_scales(state, raw, self.absolute_floor)
                self.details.update(scales=list(scales), trust_radius=self.radius)
                coefficients = self._scaled_coefficients(df, residual, scales)
                if coefficients is None:
                    return None, "degenerate_history"
            else:
                gram = np.array([[_dot(a, b) for b in df] for a in df])
                scale = float(np.max(np.diag(gram)))
                if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
                    return None, "degenerate_history"
                rhs = np.array([_dot(a, residual) for a in df]) / scale
                gram /= scale
                gram.flat[:: len(df) + 1] += self.regularization
                coefficients = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return None, "singular_history"
        if not np.all(np.isfinite(coefficients)):
            return None, "nonfinite_coefficients"
        step = tuple(mixing * a for a in residual)
        for coefficient, delta_x, delta_f in zip(coefficients, dx, df):
            for index in range(2):
                step[index][...] -= coefficient * (
                    delta_x[index] + mixing * delta_f[index]
                )
        if any(not np.all(np.isfinite(a)) for a in step):
            return None, "nonfinite_proposal"
        if self.policy == "scaled-trust-v1":
            scaled_step = _scaled(step, scales)
            norm = float(np.sqrt(_dot(scaled_step, scaled_step)))
            # Bound against the reference field magnitude, not alpha*defect.
            # Using the same scaled norm also makes the bound independent of
            # padding by inactive zero entries in a larger device matrix.
            reference_norm = max(
                np.sqrt(_dot(value, value))
                for value in (_scaled(state, scales), _scaled(raw, scales))
            )
            bound = self.radius * reference_norm
            factor = min(1.0, bound / norm) if norm else 1.0
            self.details.update(unbounded_step_norm=norm, clip_factor=factor,
                                reference_field_norm=float(reference_norm),
                                scaled_step_bound=float(bound))
            step = tuple(factor * a for a in step)
        else:
            linear_norm_squared = mixing**2 * _dot(residual, residual)
            if _dot(step, step) > self.max_step_ratio**2 * linear_norm_squared:
                return None, "step_bound"
        proposal = tuple(a + b for a, b in zip(state, step))
        return proposal, "proposed"


def physical_pair_admissible(pair: Pair, grid: SignedFrequencyGrid) -> bool:
    """Require finite bosonic algebra and PSD correlation matrices.

    No spectral passivity is imposed on their difference: the individual
    nonequilibrium correlations are PSD even when their difference is not.
    This guard neither projects nor repairs a rejected extrapolation.
    """

    lesser, greater = pair
    if any(not np.all(np.isfinite(a)) for a in pair):
        return False
    tolerance = 1e-10
    if any(antihermitian_residual(a) > tolerance for a in pair):
        return False
    scale = max(float(np.max(np.abs(a))) for a in pair)
    mismatch = np.max(np.abs(lesser[grid.negative_partners] - greater.swapaxes(-1, -2)))
    if mismatch > tolerance * max(scale, np.finfo(float).tiny):
        return False
    # The checked signed-frequency transpose relation makes the second PSD
    # eigensolve redundant; it has the same eigenvalues at partner frequencies.
    return correlation_positivity_violation(lesser) <= tolerance
