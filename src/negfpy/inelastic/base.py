"""Shared types for inelastic phonon transport."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Array = np.ndarray
PPSelfEnergyModel = Callable[[float, Array, int], Array]


@dataclass(frozen=True)
class InelasticSolveInfo:
    """Convergence metadata for self-consistent inelastic solves."""

    converged: bool
    iterations: int
    residual: float

    def as_dict(self) -> dict[str, object]:
        return {
            "converged": bool(self.converged),
            "iterations": int(self.iterations),
            "residual": float(self.residual),
        }
