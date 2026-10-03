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


class AndersonHistory:
    """Damped type-II Anderson with a regularized small real Gram system."""

    def __init__(self, depth: int, regularization: float, max_step_ratio: float):
        self.states: deque[Pair] = deque(maxlen=depth + 1)
        self.residuals: deque[Pair] = deque(maxlen=depth + 1)
        self.regularization = regularization
        self.max_step_ratio = max_step_ratio

    def clear(self) -> None:
        self.states.clear()
        self.residuals.clear()

    def propose(self, state: Pair, raw: Pair, mixing: float) -> tuple[Pair | None, str]:
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
        gram = np.array([[_dot(a, b) for b in df] for a in df])
        scale = float(np.max(np.diag(gram)))
        if not np.isfinite(scale) or scale <= np.finfo(float).tiny:
            return None, "degenerate_history"
        rhs = np.array([_dot(a, residual) for a in df]) / scale
        gram /= scale
        gram.flat[:: len(df) + 1] += self.regularization
        try:
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
