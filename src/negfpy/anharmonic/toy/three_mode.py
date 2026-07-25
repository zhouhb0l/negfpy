r"""Discrete isolated-mode fixture for validating three-phonon kinematics.

This module is deliberately a validation oracle, not a model of a broadened
reservoir.  Each isolated harmonic mode lies exactly on a
:class:`~negfpy.anharmonic.grid.SignedFrequencyGrid` bin.  With the Green
function convention used by :mod:`negfpy.anharmonic`, a mode of frequency
``Omega`` and Bose occupation ``n`` has

.. math::

   G^<(\omega) = -\frac{i\pi}{\Omega}
       [n\,\delta(\omega-\Omega)
        +(n+1)\,\delta(\omega+\Omega)],

.. math::

   G^>(\omega) = -\frac{i\pi}{\Omega}
       [(n+1)\,\delta(\omega-\Omega)
        +n\,\delta(\omega+\Omega)].

The discrete delta is represented by ``1 / grid.step`` at the matching bin.
Consequently, rectangle integration with ``grid.step / (2*pi)`` gives the
correct equal-time factor ``1 / (2*Omega)``.

The three-mode fixture uses ``Omega_0 = Omega_1 + Omega_2`` and the sole
cubic interaction

.. math::

   H_3 = \psi q_0 q_1 q_2.

Equivalently, all six permutations of ``Psi[0, 1, 2]`` equal ``psi`` in the
package convention ``H3 = sum(Psi_abc*q_a*q_b*q_c) / 3!``.  This construction
isolates the sum-frequency decay line and the two difference-frequency
combination lines while retaining the exact production diagrammatic
prefactor.

The spectral lines are distributional grid fixtures.  They are intended for
kernel and occupation-factor tests only; they are not finite-lifetime Green
functions and must not be used as an SCBA convergence benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from typing import Any

import numpy as np

from ..grid import SignedFrequencyGrid
from ..types import DenseCubicVertex


Array = np.ndarray


def _readonly_complex(values: Any) -> Array:
    result = np.array(values, dtype=np.complex128, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError("spectral lines must contain only finite values.")
    result.setflags(write=False)
    return result


def _positive_bin(value: Any, *, name: str, grid: SignedFrequencyGrid) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1 or result > grid.n_positive:
        raise ValueError(
            f"{name} must lie in [1, {grid.n_positive}] for the supplied grid."
        )
    return result


def _nonnegative_temperature(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError("temperature must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError("temperature must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError("temperature must be finite and non-negative.")
    return result


def _finite_coupling(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError("coupling must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError("coupling must be real.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError("coupling must be finite.")
    return result


def bose_occupation(frequency: float, temperature: float) -> float:
    """Return ``1 / expm1(frequency / temperature)`` safely.

    Both arguments are dimensionless solver quantities, and ``frequency``
    must be strictly positive.  The zero-temperature limit is returned
    analytically, including the spontaneous ``n + 1`` contribution retained
    by :func:`isolated_mode_keldysh_lines`.
    """

    frequency_value = float(frequency)
    if not np.isfinite(frequency_value) or frequency_value <= 0.0:
        raise ValueError("frequency must be finite and strictly positive.")
    temperature_value = _nonnegative_temperature(temperature)
    if temperature_value == 0.0:
        return 0.0

    exponent = frequency_value / temperature_value
    # Above this value exp(exponent) would overflow in float64, while the
    # occupation is already below the smallest representable normal number.
    if exponent > np.log(np.finfo(np.float64).max):
        return 0.0
    return float(1.0 / np.expm1(exponent))


@dataclass(frozen=True, slots=True, eq=False)
class IsolatedModeKeldyshLines:
    """Immutable lesser/greater spectral lines for one isolated mode."""

    frequency_bin: int
    frequency: float
    temperature: float
    occupation: float
    lesser: Array = field(repr=False, compare=False)
    greater: Array = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "lesser", _readonly_complex(self.lesser))
        object.__setattr__(self, "greater", _readonly_complex(self.greater))


def isolated_mode_keldysh_lines(
    grid: SignedFrequencyGrid,
    frequency_bin: int,
    temperature: float,
) -> IsolatedModeKeldyshLines:
    """Construct exact-bin equilibrium spectral lines for one harmonic mode."""

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    mode_bin = _positive_bin(frequency_bin, name="frequency_bin", grid=grid)
    temperature_value = _nonnegative_temperature(temperature)
    frequency = mode_bin * grid.step
    occupation = bose_occupation(frequency, temperature_value)

    lesser = np.zeros(grid.n_points, dtype=np.complex128)
    greater = np.zeros_like(lesser)
    positive_index = grid.zero_index + mode_bin
    negative_index = grid.zero_index - mode_bin
    line_scale = -1.0j * np.pi / (frequency * grid.step)

    lesser[positive_index] = line_scale * occupation
    lesser[negative_index] = line_scale * (occupation + 1.0)
    greater[positive_index] = line_scale * (occupation + 1.0)
    greater[negative_index] = line_scale * occupation

    return IsolatedModeKeldyshLines(
        frequency_bin=mode_bin,
        frequency=frequency,
        temperature=temperature_value,
        occupation=occupation,
        lesser=lesser,
        greater=greater,
    )


@dataclass(frozen=True, slots=True, eq=False)
class ThreeModeDecayFixture:
    """Immutable resonant three-mode input for the production cubic kernel."""

    grid: SignedFrequencyGrid
    mode_bins: tuple[int, int, int]
    mode_frequencies: tuple[float, float, float]
    occupations: tuple[float, float, float]
    temperature: float
    coupling: float
    vertex: DenseCubicVertex
    green_lesser: Array = field(repr=False, compare=False)
    green_greater: Array = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "green_lesser",
            _readonly_complex(self.green_lesser),
        )
        object.__setattr__(
            self,
            "green_greater",
            _readonly_complex(self.green_greater),
        )


def equilibrium_three_mode_decay_fixture(
    grid: SignedFrequencyGrid,
    daughter_bins: tuple[int, int],
    *,
    temperature: float,
    coupling: float,
) -> ThreeModeDecayFixture:
    r"""Return an exact-bin ``Omega_0 = Omega_1 + Omega_2`` decay fixture.

    ``daughter_bins`` gives the positive grid-bin numbers of modes 1 and 2.
    They must be distinct so the two combination lines remain separately
    identifiable.  Mode 0 occupies their sum-frequency bin.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if (
        not isinstance(daughter_bins, tuple)
        or len(daughter_bins) != 2
    ):
        raise TypeError("daughter_bins must be a pair of integer grid bins.")
    first_bin = _positive_bin(
        daughter_bins[0],
        name="daughter_bins[0]",
        grid=grid,
    )
    second_bin = _positive_bin(
        daughter_bins[1],
        name="daughter_bins[1]",
        grid=grid,
    )
    if first_bin == second_bin:
        raise ValueError(
            "daughter_bins must be distinct so combination lines do not merge."
        )
    parent_bin = first_bin + second_bin
    if parent_bin > grid.n_positive:
        raise ValueError(
            "the daughter-bin sum must lie inside the positive grid cutoff."
        )

    temperature_value = _nonnegative_temperature(temperature)
    coupling_value = _finite_coupling(coupling)
    mode_bins = (parent_bin, first_bin, second_bin)
    lines = tuple(
        isolated_mode_keldysh_lines(grid, mode_bin, temperature_value)
        for mode_bin in mode_bins
    )

    green_lesser = np.zeros((grid.n_points, 3, 3), dtype=np.complex128)
    green_greater = np.zeros_like(green_lesser)
    for mode, mode_lines in enumerate(lines):
        green_lesser[:, mode, mode] = mode_lines.lesser
        green_greater[:, mode, mode] = mode_lines.greater

    values = np.zeros((3, 3, 3), dtype=np.float64)
    for indices in permutations((0, 1, 2)):
        values[indices] = coupling_value
    vertex = DenseCubicVertex(
        values,
        metadata={
            "model": "isolated_resonant_three_mode_decay",
            "interaction": "coupling*q0*q1*q2",
            "distributional_grid_fixture": True,
            "mass_weighted": True,
        },
    )

    return ThreeModeDecayFixture(
        grid=grid,
        mode_bins=mode_bins,
        mode_frequencies=tuple(item.frequency for item in lines),
        occupations=tuple(item.occupation for item in lines),
        temperature=temperature_value,
        coupling=coupling_value,
        vertex=vertex,
        green_lesser=green_lesser,
        green_greater=green_greater,
    )


__all__ = [
    "IsolatedModeKeldyshLines",
    "ThreeModeDecayFixture",
    "bose_occupation",
    "equilibrium_three_mode_decay_fixture",
    "isolated_mode_keldysh_lines",
]
