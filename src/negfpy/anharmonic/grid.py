"""Uniform signed-frequency grids for anharmonic NEGF calculations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


Array = np.ndarray


def _readonly(values: Array) -> Array:
    array = np.array(values, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True, eq=False)
class SignedFrequencyGrid:
    """Odd, uniform frequency grid centered exactly on zero.

    Frequencies are angular frequencies in solver units.  If ``n_positive`` is
    ``N``, the grid contains ``2*N + 1`` points

    ``step * [-N, ..., -1, 0, 1, ..., N]``.

    All exposed arrays are read-only so the index/frequency correspondence
    cannot be changed after construction.
    """

    step: float
    n_positive: int
    _values: Array = field(init=False, repr=False, compare=False)
    _weights: Array = field(init=False, repr=False, compare=False)
    _positive_indices: Array = field(init=False, repr=False, compare=False)
    _nonnegative_indices: Array = field(init=False, repr=False, compare=False)
    _negative_indices: Array = field(init=False, repr=False, compare=False)
    _negative_partners: Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.step, (bool, np.bool_)):
            raise TypeError("step must be a positive finite real number.")
        step = float(self.step)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be a positive finite real number.")

        if isinstance(self.n_positive, (bool, np.bool_)) or not isinstance(
            self.n_positive, (int, np.integer)
        ):
            raise TypeError("n_positive must be an integer.")
        n_positive = int(self.n_positive)
        if n_positive < 1:
            raise ValueError("n_positive must be at least one.")

        indices = np.arange(-n_positive, n_positive + 1, dtype=np.int64)
        values = step * indices
        zero_index = n_positive
        n_points = 2 * n_positive + 1

        object.__setattr__(self, "step", step)
        object.__setattr__(self, "n_positive", n_positive)
        object.__setattr__(self, "_values", _readonly(values.astype(np.float64)))
        object.__setattr__(
            self,
            "_weights",
            _readonly(np.full(n_points, step / (2.0 * np.pi), dtype=np.float64)),
        )
        object.__setattr__(
            self,
            "_positive_indices",
            _readonly(np.arange(zero_index + 1, n_points, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "_nonnegative_indices",
            _readonly(np.arange(zero_index, n_points, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "_negative_indices",
            _readonly(np.arange(0, zero_index, dtype=np.int64)),
        )
        object.__setattr__(
            self,
            "_negative_partners",
            _readonly(np.arange(n_points - 1, -1, -1, dtype=np.int64)),
        )

    @property
    def values(self) -> Array:
        """Signed angular-frequency values in ascending order."""

        return self._values

    @property
    def omega(self) -> Array:
        """Alias for :attr:`values` used in numerical formulas."""

        return self._values

    @property
    def weights(self) -> Array:
        """Rectangle-rule integration weights, including ``1/(2*pi)``."""

        return self._weights

    @property
    def n_points(self) -> int:
        return 2 * self.n_positive + 1

    @property
    def zero_index(self) -> int:
        return self.n_positive

    @property
    def cutoff(self) -> float:
        return self.step * self.n_positive

    @property
    def integration_weight(self) -> float:
        """Uniform integration weight ``step / (2*pi)``."""

        return self.step / (2.0 * np.pi)

    @property
    def positive_indices(self) -> Array:
        """Indices of the strictly positive frequencies."""

        return self._positive_indices

    @property
    def nonnegative_indices(self) -> Array:
        """Indices of zero and all positive frequencies."""

        return self._nonnegative_indices

    @property
    def negative_indices(self) -> Array:
        """Indices of the strictly negative frequencies."""

        return self._negative_indices

    @property
    def negative_partners(self) -> Array:
        """Map every grid index to the index at the opposite frequency."""

        return self._negative_partners

    def partner_index(self, index: int) -> int:
        """Return the array index corresponding to ``-omega[index]``."""

        if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
            raise TypeError("index must be an integer.")
        normalized = int(index)
        if normalized < 0:
            normalized += self.n_points
        if normalized < 0 or normalized >= self.n_points:
            raise IndexError("frequency-grid index out of range.")
        return int(self._negative_partners[normalized])
