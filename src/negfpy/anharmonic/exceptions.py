"""Exceptions raised by the anharmonic NEGF solvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .types import AnharmonicResult


class SCBAConvergenceError(RuntimeError):
    """SCBA failed while retaining its last complete partial result."""

    def __init__(
        self,
        message: str,
        result: AnharmonicResult | None = None,
    ) -> None:
        super().__init__(message)
        self.result = result
        self.partial_result = result
