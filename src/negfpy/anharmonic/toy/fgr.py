r"""Independent mode-space Born/FGR oracle for finite toy models.

This module does not call the production cubic contraction or retarded
reconstruction. Instead, it evaluates the diagonal mode-space bubble by
explicitly enumerating the four signed-frequency channels of every ordered
internal-mode pair. It is therefore a prefactor, occupation-factor, and
Kramers--Kronig oracle for the dense solver.

For harmonic modes ``Omega_u`` and a real mode-space cubic vertex ``V_suv``,
one ordered internal pair contributes

.. math::

   \Sigma^{x}_{ss}(\omega) =
   -i\pi\frac{V_{suv}^2}{4\Omega_u\Omega_v}
   \sum_{\sigma_u,\sigma_v=\pm1}
   W^x_{\sigma_u\sigma_v}
   \delta(\omega-\sigma_u\Omega_u-\sigma_v\Omega_v),

where ``x`` is lesser or greater. The occupation factors are

.. math::

   W^<_{\sigma_u\sigma_v}
   = \left[n_u+\frac{1-\sigma_u}{2}\right]
     \left[n_v+\frac{1-\sigma_v}{2}\right],

.. math::

   W^>_{\sigma_u\sigma_v}
   = \left[n_u+\frac{1+\sigma_u}{2}\right]
     \left[n_v+\frac{1+\sigma_v}{2}\right].

The fixed factor ``1/4`` follows from the production bubble's ``i/2``
diagrammatic prefactor and the two harmonic spectral-line normalizations.
There is no adjustable prefactor.

The delta functions are represented by ``1 / grid.step`` at exact grid
bins. If ``K`` multiplies ``-i*pi*delta(omega-E)`` in
``Sigma_greater-Sigma_lesser``, its independently reconstructed retarded
contribution is

.. math::

   \frac{K}{2}\left[
     \mathcal P\frac{1}{\omega-E}
     -i\pi\delta(\omega-E)
   \right].

This is exactly the rectangle-grid distribution used by the direct
principal-value implementation. It is not a continuum thermodynamic-limit
linewidth: its on-resonance imaginary part depends on ``grid.step``. A
released continuum linewidth still requires a converged bath/momentum
continuum or a separately specified and refined smearing prescription.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..grid import SignedFrequencyGrid
from ..types import DenseCubicVertex


Array = np.ndarray


def _readonly(values: Any, *, dtype: np.dtype[Any], name: str) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _finite_nonnegative(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{name} must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative.")
    return result


def _positive_fraction(value: Any, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _mode_index(value: Any, *, nmodes: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError("target_mode must be an integer.")
    result = int(value)
    if result < 0 or result >= nmodes:
        raise ValueError(f"target_mode must lie in [0, {nmodes - 1}].")
    return result


def _bose_occupations(frequencies: Array, temperature: float) -> Array:
    if temperature == 0.0:
        return np.zeros_like(frequencies)
    exponents = frequencies / temperature
    result = np.zeros_like(frequencies)
    finite_exponential = exponents <= np.log(np.finfo(np.float64).max)
    result[finite_exponential] = 1.0 / np.expm1(exponents[finite_exponential])
    return result


def _exact_frequency_bins(
    frequencies: Any,
    grid: SignedFrequencyGrid,
) -> tuple[Array, Array]:
    raw = np.asarray(frequencies)
    if np.iscomplexobj(raw):
        raise TypeError("mode_frequencies must be real.")
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("mode_frequencies must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(
            "mode_frequencies must be finite and strictly positive."
        )

    rounded = np.rint(values / grid.step)
    bins = rounded.astype(np.int64)
    reconstructed = bins.astype(np.float64) * grid.step
    tolerance = 32.0 * np.finfo(np.float64).eps * np.maximum(
        1.0,
        np.abs(values),
    )
    if np.any(np.abs(values - reconstructed) > tolerance):
        raise ValueError(
            "every mode frequency must lie exactly on a positive grid bin."
        )
    if np.any(bins < 1) or np.any(bins > grid.n_positive):
        raise ValueError(
            "every mode frequency must lie inside the positive grid cutoff."
        )
    return values, bins


@dataclass(frozen=True, slots=True, eq=False)
class HarmonicModeBasis:
    """Positive normal modes and a cubic tensor projected into that basis.

    The eigenvectors are stored column-wise. Mode-resolved observables inside
    an exactly degenerate subspace depend on the basis selected by the
    eigensolver; only subspace-summed quantities are invariant there.
    """

    frequencies: Array = field(repr=False, compare=False)
    eigenvectors: Array = field(repr=False, compare=False)
    vertex: DenseCubicVertex

    def __post_init__(self) -> None:
        frequencies = _readonly(
            self.frequencies,
            dtype=np.dtype(np.float64),
            name="frequencies",
        )
        eigenvectors = _readonly(
            self.eigenvectors,
            dtype=np.dtype(np.float64),
            name="eigenvectors",
        )
        if frequencies.ndim != 1 or np.any(frequencies <= 0.0):
            raise ValueError("frequencies must be a positive one-dimensional array.")
        if eigenvectors.shape != (frequencies.size, frequencies.size):
            raise ValueError(
                "eigenvectors must be square with one column per frequency."
            )
        if not isinstance(self.vertex, DenseCubicVertex):
            raise TypeError("vertex must be a DenseCubicVertex.")
        if self.vertex.ndof != frequencies.size:
            raise ValueError("vertex dimension must equal the number of modes.")
        object.__setattr__(self, "frequencies", frequencies)
        object.__setattr__(self, "eigenvectors", eigenvectors)


def harmonic_mode_basis(
    dynamical_matrix: Any,
    vertex: DenseCubicVertex,
    *,
    stability_atol: float = 1e-12,
) -> HarmonicModeBasis:
    """Diagonalize a real pinned toy model and project its cubic vertex.

    ``dynamical_matrix`` is the mass-weighted finite-device matrix. This
    helper deliberately rejects acoustic/zero and unstable modes because the
    exact-bin oracle contains explicit ``1 / Omega`` factors.
    """

    if not isinstance(vertex, DenseCubicVertex):
        raise TypeError("vertex must be a DenseCubicVertex.")
    stability = _finite_nonnegative(stability_atol, name="stability_atol")

    raw = np.asarray(dynamical_matrix)
    if raw.ndim != 2 or raw.shape[0] != raw.shape[1]:
        raise ValueError("dynamical_matrix must be square.")
    if raw.shape[0] != vertex.ndof:
        raise ValueError(
            "dynamical_matrix dimension must equal the vertex dimension."
        )
    if not np.all(np.isfinite(raw)):
        raise ValueError("dynamical_matrix must contain only finite values.")
    if np.iscomplexobj(raw):
        imaginary_scale = float(np.max(np.abs(raw.imag), initial=0.0))
        if imaginary_scale > 1e-13:
            raise ValueError(
                "the v0.1 mode-space oracle accepts only real-space toy matrices."
            )
        raw = raw.real
    matrix = np.asarray(raw, dtype=np.float64)
    if not np.allclose(matrix, matrix.T, rtol=1e-12, atol=1e-12):
        raise ValueError("dynamical_matrix must be real symmetric.")

    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    if np.any(eigenvalues <= stability):
        raise ValueError(
            "dynamical_matrix must be pinned and strictly stable above "
            "stability_atol."
        )
    frequencies = np.sqrt(eigenvalues)
    projected = np.einsum(
        "abc,ai,bj,ck->ijk",
        vertex.values,
        eigenvectors,
        eigenvectors,
        eigenvectors,
        optimize=True,
    )
    mode_vertex = DenseCubicVertex(
        projected,
        metadata={
            **dict(vertex.metadata),
            "basis": "finite_device_harmonic_normal_modes",
            "source_basis": "mass_weighted_real_space",
            "eigenvectors_column_major": True,
        },
    )
    return HarmonicModeBasis(
        frequencies=frequencies,
        eigenvectors=eigenvectors,
        vertex=mode_vertex,
    )


@dataclass(frozen=True, slots=True)
class FGRChannel:
    """One ordered-pair, signed-frequency contribution to the bubble."""

    internal_modes: tuple[int, int]
    signs: tuple[int, int]
    frequency_bin: int
    lesser_coefficient: float
    greater_coefficient: float
    spectral_coefficient: float

    def __post_init__(self) -> None:
        if (
            not isinstance(self.internal_modes, tuple)
            or len(self.internal_modes) != 2
            or any(
                isinstance(mode, (bool, np.bool_))
                or not isinstance(mode, (int, np.integer))
                or int(mode) < 0
                for mode in self.internal_modes
            )
        ):
            raise ValueError(
                "internal_modes must be a pair of non-negative integers."
            )
        if self.signs not in {(1, 1), (1, -1), (-1, 1), (-1, -1)}:
            raise ValueError("signs must be a pair whose entries are +1 or -1.")
        if isinstance(self.frequency_bin, (bool, np.bool_)) or not isinstance(
            self.frequency_bin,
            (int, np.integer),
        ):
            raise TypeError("frequency_bin must be an integer.")
        coefficients = np.asarray(
            (
                self.lesser_coefficient,
                self.greater_coefficient,
                self.spectral_coefficient,
            ),
            dtype=np.float64,
        )
        if not np.all(np.isfinite(coefficients)):
            raise ValueError("channel coefficients must be finite.")
        expected_spectral = (
            float(self.greater_coefficient)
            - float(self.lesser_coefficient)
        )
        tolerance = 32.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(expected_spectral),
        )
        if abs(float(self.spectral_coefficient) - expected_spectral) > tolerance:
            raise ValueError(
                "spectral_coefficient must equal greater_coefficient "
                "minus lesser_coefficient."
            )

    @property
    def process(self) -> str:
        if self.signs == (1, 1):
            return "sum_positive"
        if self.signs == (-1, -1):
            return "sum_negative"
        return "difference"


@dataclass(frozen=True, slots=True, eq=False)
class ModeSpaceFGRSpectrum:
    """Diagonal discrete Born spectrum and its on-shell quasiparticle data.

    ``linewidth`` is the positive-pole spectral FWHM (also the population
    decay rate in the pole approximation),

    ``linewidth = -Im Sigma^r(Omega) / Omega``.

    The corresponding spectral HWHM/amplitude decay rate is
    ``linewidth / 2``. The dynamic bubble shift excludes the cubic tadpole:

    ``frequency_shift = Re Sigma^r(Omega) / (2*Omega)``.
    """

    grid: SignedFrequencyGrid
    target_mode: int
    target_frequency_bin: int
    mode_frequencies: Array = field(repr=False, compare=False)
    occupations: Array = field(repr=False, compare=False)
    lesser: Array = field(repr=False, compare=False)
    greater: Array = field(repr=False, compare=False)
    retarded: Array = field(repr=False, compare=False)
    channels: tuple[FGRChannel, ...]
    weak_coupling_limit: float = 0.1

    def __post_init__(self) -> None:
        if not isinstance(self.grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid.")
        frequencies = _readonly(
            self.mode_frequencies,
            dtype=np.dtype(np.float64),
            name="mode_frequencies",
        )
        occupations = _readonly(
            self.occupations,
            dtype=np.dtype(np.float64),
            name="occupations",
        )
        expected = (self.grid.n_points,)
        lesser = _readonly(
            self.lesser,
            dtype=np.dtype(np.complex128),
            name="lesser",
        )
        greater = _readonly(
            self.greater,
            dtype=np.dtype(np.complex128),
            name="greater",
        )
        retarded = _readonly(
            self.retarded,
            dtype=np.dtype(np.complex128),
            name="retarded",
        )
        if frequencies.ndim != 1 or occupations.shape != frequencies.shape:
            raise ValueError(
                "mode_frequencies and occupations must be matching vectors."
            )
        if np.any(occupations < 0.0):
            raise ValueError("occupations must be non-negative.")
        if (
            lesser.shape != expected
            or greater.shape != expected
            or retarded.shape != expected
        ):
            raise ValueError(f"spectra must all have shape {expected}.")
        _, frequency_bins = _exact_frequency_bins(frequencies, self.grid)
        target_mode = _mode_index(
            self.target_mode,
            nmodes=frequencies.size,
        )
        if (
            isinstance(self.target_frequency_bin, (bool, np.bool_))
            or not isinstance(self.target_frequency_bin, (int, np.integer))
        ):
            raise TypeError("target_frequency_bin must be an integer.")
        target_frequency_bin = int(self.target_frequency_bin)
        if target_frequency_bin != int(frequency_bins[target_mode]):
            raise ValueError(
                "target_frequency_bin must match the selected mode frequency."
            )
        if not isinstance(self.channels, tuple) or not all(
            isinstance(channel, FGRChannel) for channel in self.channels
        ):
            raise TypeError("channels must be a tuple of FGRChannel objects.")
        if any(
            channel.internal_modes[0] >= frequencies.size
            or channel.internal_modes[1] >= frequencies.size
            or abs(channel.frequency_bin) > self.grid.n_positive
            for channel in self.channels
        ):
            raise ValueError(
                "channel mode indices and frequency bins must lie inside "
                "the supplied mode/grid domains."
            )
        object.__setattr__(self, "target_mode", target_mode)
        object.__setattr__(
            self,
            "target_frequency_bin",
            target_frequency_bin,
        )
        object.__setattr__(self, "mode_frequencies", frequencies)
        object.__setattr__(self, "occupations", occupations)
        object.__setattr__(self, "lesser", lesser)
        object.__setattr__(self, "greater", greater)
        object.__setattr__(self, "retarded", retarded)
        object.__setattr__(
            self,
            "weak_coupling_limit",
            _positive_fraction(
                self.weak_coupling_limit,
                name="weak_coupling_limit",
            ),
        )

    @property
    def target_frequency(self) -> float:
        return float(self.mode_frequencies[self.target_mode])

    @property
    def on_shell_index(self) -> int:
        return self.grid.zero_index + self.target_frequency_bin

    @property
    def self_energy_on_shell(self) -> complex:
        return complex(self.retarded[self.on_shell_index])

    @property
    def frequency_shift(self) -> float:
        return float(
            self.self_energy_on_shell.real / (2.0 * self.target_frequency)
        )

    @property
    def spectral_half_width(self) -> float:
        return float(
            -self.self_energy_on_shell.imag / (2.0 * self.target_frequency)
        )

    @property
    def linewidth(self) -> float:
        """Positive-pole spectral FWHM/population decay rate."""

        return 2.0 * self.spectral_half_width

    @property
    def relative_shift(self) -> float:
        return abs(self.frequency_shift) / self.target_frequency

    @property
    def relative_linewidth(self) -> float:
        return abs(self.linewidth) / self.target_frequency

    @property
    def passive(self) -> bool:
        tolerance = 64.0 * np.finfo(np.float64).eps * max(
            1.0,
            abs(self.self_energy_on_shell),
        )
        return self.spectral_half_width >= -tolerance

    @property
    def weak_coupling_passed(self) -> bool:
        return bool(
            self.passive
            and self.relative_shift < self.weak_coupling_limit
            and self.relative_linewidth < self.weak_coupling_limit
        )


def discrete_mode_space_fgr(
    grid: SignedFrequencyGrid,
    mode_frequencies: Any,
    mode_vertex: DenseCubicVertex,
    *,
    temperature: float,
    target_mode: int,
    weak_coupling_limit: float = 0.1,
) -> ModeSpaceFGRSpectrum:
    """Evaluate the diagonal exact-bin Born/FGR spectrum of one mode.

    Internal mode pairs are ordered, exactly as in the two independent
    internal sums of the production cubic bubble. Therefore distinct modes
    appear twice and identical modes once; no ad-hoc degeneracy or symmetry
    factor is applied.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(mode_vertex, DenseCubicVertex):
        raise TypeError("mode_vertex must be a DenseCubicVertex.")
    frequencies, frequency_bins = _exact_frequency_bins(
        mode_frequencies,
        grid,
    )
    if mode_vertex.ndof != frequencies.size:
        raise ValueError(
            "mode_vertex dimension must equal the number of mode frequencies."
        )
    selected_mode = _mode_index(target_mode, nmodes=frequencies.size)
    temperature_value = _finite_nonnegative(temperature, name="temperature")
    limit = _positive_fraction(
        weak_coupling_limit,
        name="weak_coupling_limit",
    )
    occupations = _bose_occupations(frequencies, temperature_value)

    lesser = np.zeros(grid.n_points, dtype=np.complex128)
    greater = np.zeros_like(lesser)
    retarded = np.zeros_like(lesser)
    channels: list[FGRChannel] = []
    omega = grid.values

    for first_mode in range(frequencies.size):
        first_frequency = frequencies[first_mode]
        first_occupation = occupations[first_mode]
        first_bin = int(frequency_bins[first_mode])
        for second_mode in range(frequencies.size):
            coupling = float(
                mode_vertex.values[selected_mode, first_mode, second_mode]
            )
            if coupling == 0.0:
                continue
            second_frequency = frequencies[second_mode]
            second_occupation = occupations[second_mode]
            second_bin = int(frequency_bins[second_mode])
            base = coupling**2 / (
                4.0 * first_frequency * second_frequency
            )

            for first_sign, second_sign in (
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ):
                event_bin = first_sign * first_bin + second_sign * second_bin
                # This exactly mirrors zero extension of the production
                # convolution: out-of-cutoff spectral lines do not enter the
                # finite-grid Hilbert reconstruction.
                if abs(event_bin) > grid.n_positive:
                    continue

                lesser_weight = (
                    first_occupation + 0.5 * (1 - first_sign)
                ) * (
                    second_occupation + 0.5 * (1 - second_sign)
                )
                greater_weight = (
                    first_occupation + 0.5 * (1 + first_sign)
                ) * (
                    second_occupation + 0.5 * (1 + second_sign)
                )
                lesser_coefficient = base * lesser_weight
                greater_coefficient = base * greater_weight
                spectral_coefficient = (
                    greater_coefficient - lesser_coefficient
                )
                event_index = grid.zero_index + event_bin
                lesser[event_index] += (
                    -1.0j
                    * np.pi
                    * lesser_coefficient
                    / grid.step
                )
                greater[event_index] += (
                    -1.0j
                    * np.pi
                    * greater_coefficient
                    / grid.step
                )

                if spectral_coefficient != 0.0:
                    event_frequency = event_bin * grid.step
                    other = np.arange(grid.n_points) != event_index
                    retarded[other] += (
                        0.5
                        * spectral_coefficient
                        / (omega[other] - event_frequency)
                    )
                    retarded[event_index] += (
                        -0.5j
                        * np.pi
                        * spectral_coefficient
                        / grid.step
                    )

                channels.append(
                    FGRChannel(
                        internal_modes=(first_mode, second_mode),
                        signs=(first_sign, second_sign),
                        frequency_bin=event_bin,
                        lesser_coefficient=float(lesser_coefficient),
                        greater_coefficient=float(greater_coefficient),
                        spectral_coefficient=float(spectral_coefficient),
                    )
                )

    return ModeSpaceFGRSpectrum(
        grid=grid,
        target_mode=selected_mode,
        target_frequency_bin=int(frequency_bins[selected_mode]),
        mode_frequencies=frequencies,
        occupations=occupations,
        lesser=lesser,
        greater=greater,
        retarded=retarded,
        channels=tuple(channels),
        weak_coupling_limit=limit,
    )


__all__ = [
    "FGRChannel",
    "HarmonicModeBasis",
    "ModeSpaceFGRSpectrum",
    "discrete_mode_space_fgr",
    "harmonic_mode_basis",
]
