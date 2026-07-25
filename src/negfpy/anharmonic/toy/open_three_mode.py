r"""Independent open-system oracle for a resonant three-mode cubic vertex.

The fixture in this module is deliberately small enough to evaluate without
calling the production cubic contraction or retarded reconstruction.  A
single three-DOF device layer is matched to identical diagonal left and right
leads.  The uncoupled band centres are

.. math::

   (\Omega_x,\Omega_y,\Omega_z)=(1.5,0.9,0.6),
   \qquad \Omega_x=\Omega_y+\Omega_z,

and every scalar lead has nearest-layer dynamical-matrix coupling ``-0.1``.
The unit-mass cubic interaction is

.. math::

   H_3=g\,u_xu_yu_z.

Thus all six permutations of :math:`\Psi_{xyz}` equal ``g`` in the NEGFPy
``H3 = sum(Psi*u*u*u)/3!`` convention.

Unlike the isolated exact-bin fixture, every mode here is broadened by an
explicit harmonic continuum.  For pointwise regression against the
production Born solver, the oracle constructs a diagnostic signed-grid
spectrum by evaluating

* each scalar surface Green function from its analytic quadratic equation;
* the contact-dressed harmonic lesser and greater Green functions directly;
* each zero-extended scalar frequency convolution explicitly (or with an
  independently coded exact padded FFT on release-sized grids); and
* the retarded self-energy with the same finite principal-value sum, evaluated
  directly on reference grids or by a nonperiodic Toeplitz FFT embedding.

Those finite-grid values are not the acceptance poles.  The reported
``self_energy_on_shell``, frequency shifts, and linewidths instead use the
eta-free matched-chain continuum spectral density, explicit Bose scattering
channels, adaptive angle quadrature, and adaptive Cauchy principal-value
quadrature.  Consequently their convergence is controlled by integration
tolerances rather than by the diagnostic grid or ``eta_lead``.

It never imports or calls ``cubic_lesser_greater`` or
``retarded_from_keldysh``.  The resulting Born self-energy is therefore an
independent open-device check of the production prefactor, occupation
factors, convolution indexing, and dispersive reconstruction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import permutations
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
from scipy.fft import fft, ifft, next_fast_len
from scipy.integrate import quad

from negfpy.core.types import Device1D, LeadBlocks

from ..grid import SignedFrequencyGrid
from ..types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
)


Array = np.ndarray
_DEFAULT_FREQUENCIES = (1.5, 0.9, 0.6)
_GRID_FFT_THRESHOLD = 2049


def _readonly(values: Any, *, dtype: np.dtype[Any], name: str) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    result.setflags(write=False)
    return result


def _finite_real(
    value: Any,
    *,
    name: str,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{name} must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _validated_frequencies(values: Any) -> tuple[float, float, float]:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise TypeError("mode_frequencies must be real.")
    frequencies = np.asarray(raw, dtype=np.float64)
    if frequencies.shape != (3,):
        raise ValueError("mode_frequencies must contain exactly three values.")
    if not np.all(np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError(
            "mode_frequencies must be finite and strictly positive."
        )
    resonance_scale = max(1.0, float(np.max(frequencies)))
    if not np.isclose(
        frequencies[0],
        frequencies[1] + frequencies[2],
        rtol=0.0,
        atol=64.0 * np.finfo(float).eps * resonance_scale,
    ):
        raise ValueError(
            "the open three-mode fixture requires Omega_x=Omega_y+Omega_z."
        )
    return tuple(float(value) for value in frequencies)


def _signed_bose(omega: Array, temperature: float) -> Array:
    """Evaluate the signed Bose function without using contact helpers."""

    values = np.asarray(omega, dtype=np.float64)
    result = np.empty_like(values)
    positive = values > 0.0
    negative = values < 0.0
    zero = ~(positive | negative)

    def positive_values(frequencies: Array) -> Array:
        if temperature == 0.0:
            return np.zeros_like(frequencies)
        exponent = frequencies / temperature
        occupation = np.zeros_like(exponent)
        regular = exponent < np.log(np.finfo(np.float64).max)
        occupation[regular] = 1.0 / np.expm1(exponent[regular])
        return occupation

    result[positive] = positive_values(values[positive])
    result[negative] = -1.0 - positive_values(-values[negative])
    result[zero] = np.inf
    return result


def _positive_bose(frequency: float, temperature: float) -> float:
    if temperature == 0.0:
        return 0.0
    exponent = float(frequency) / temperature
    if exponent >= np.log(np.finfo(np.float64).max):
        return 0.0
    return float(1.0 / np.expm1(exponent))


def _analytic_scalar_surface(
    omega: Array,
    *,
    onsite: float,
    hopping: float,
    eta: float,
) -> Array:
    r"""Return the causal scalar surface Green function analytically.

    The physical root of

    ``hopping**2 * g**2 - (z-onsite)*g + 1 = 0``

    has the smaller magnitude for finite retarded ``eta``.  A causal-sign
    fallback resolves the limiting equal-magnitude propagating roots.
    """

    frequencies = np.asarray(omega, dtype=np.float64)
    z = (frequencies + 1.0j * eta) ** 2
    delta = z - onsite
    hopping_squared = hopping**2
    discriminant = np.sqrt(delta**2 - 4.0 * hopping_squared)
    denominator_plus = delta + discriminant
    denominator_minus = delta - discriminant
    # The small-magnitude physical root is evaluated through the reciprocal
    # quadratic formula.  Directly subtracting ``delta-discriminant`` loses
    # all significant digits far outside the lead band (the cutoff-refinement
    # grid reaches precisely that regime).
    use_plus = np.abs(denominator_plus) >= np.abs(denominator_minus)
    stable_denominator = np.where(
        use_plus,
        denominator_plus,
        denominator_minus,
    )
    small = 2.0 / stable_denominator
    # The product of the two roots is exactly 1/hopping**2.  Recovering the
    # other root this way avoids using the cancellation-prone denominator.
    other = 1.0 / (hopping_squared * small)

    small_magnitude = np.abs(small)
    other_magnitude = np.abs(other)
    magnitude_scale = np.maximum(
        1.0,
        np.maximum(small_magnitude, other_magnitude),
    )
    tied = np.abs(small_magnitude - other_magnitude) <= (
        64.0 * np.finfo(float).eps * magnitude_scale
    )

    positive = frequencies > 0.0
    negative = frequencies < 0.0
    causal_small = np.ones(frequencies.shape, dtype=bool)
    causal_small[positive] = (
        np.imag(small[positive]) <= np.imag(other[positive])
    )
    causal_small[negative] = (
        np.imag(small[negative]) >= np.imag(other[negative])
    )
    causal_small[~(positive | negative)] = (
        small_magnitude[~(positive | negative)]
        <= other_magnitude[~(positive | negative)]
    )
    surface = np.where(tied & ~causal_small, other, small)

    residual = (
        hopping_squared * surface**2 - delta * surface + 1.0
    )
    residual_scale = np.maximum(
        1.0,
        np.abs(hopping_squared * surface**2)
        + np.abs(delta * surface),
    )
    if np.max(np.abs(residual) / residual_scale) > 1.0e-11:
        raise RuntimeError("analytic scalar surface root failed its quadratic.")
    return surface


def _positive_band_edges(
    mode_frequency: float,
    lead_coupling: float,
) -> tuple[float, float]:
    lower_squared = mode_frequency**2 - 2.0 * lead_coupling
    upper_squared = mode_frequency**2 + 2.0 * lead_coupling
    if lower_squared <= 0.0:
        raise ValueError(
            "the open oracle requires a pinned band with positive lower edge."
        )
    return float(np.sqrt(lower_squared)), float(np.sqrt(upper_squared))


def _positive_bulk_spectral(
    frequency: float,
    *,
    mode_frequency: float,
    lead_coupling: float,
) -> float:
    """Return the eta-free positive-frequency local spectral function."""

    value = float(frequency)
    if value <= 0.0:
        return 0.0
    offset = value**2 - mode_frequency**2
    radicand = 4.0 * lead_coupling**2 - offset**2
    scale = max(1.0, 4.0 * lead_coupling**2, offset**2)
    if radicand <= 64.0 * np.finfo(float).eps * scale:
        # Exact band-edge values have measure zero.  Returning zero keeps the
        # QUADPACK endpoint callback finite; nearby quadrature nodes retain
        # the integrable inverse-square-root singularity.
        return 0.0
    return float(2.0 / np.sqrt(radicand))


def _frequency_from_angle(
    angle: float,
    *,
    mode_frequency: float,
    lead_coupling: float,
) -> float:
    return float(
        np.sqrt(
            mode_frequency**2
            - 2.0 * lead_coupling * np.cos(float(angle))
        )
    )


def _angle_from_frequency(
    frequency: float,
    *,
    mode_frequency: float,
    lead_coupling: float,
) -> float | None:
    if float(frequency) <= 0.0:
        return None
    cosine = (
        mode_frequency**2 - float(frequency) ** 2
    ) / (2.0 * lead_coupling)
    tolerance = 128.0 * np.finfo(float).eps
    if cosine < -1.0 - tolerance or cosine > 1.0 + tolerance:
        return None
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _channel_spectral_weight(
    first_frequency: float,
    second_frequency: float,
    *,
    first_sign: int,
    second_sign: int,
    temperature: float,
) -> float:
    """Return the explicit greater-minus-lesser Bose channel weight."""

    first_occupation = _positive_bose(first_frequency, temperature)
    second_occupation = _positive_bose(second_frequency, temperature)
    greater = (
        first_occupation + 0.5 * (1 + first_sign)
    ) * (
        second_occupation + 0.5 * (1 + second_sign)
    )
    lesser = (
        first_occupation + 0.5 * (1 - first_sign)
    ) * (
        second_occupation + 0.5 * (1 - second_sign)
    )
    return float(greater - lesser)


def _outer_breakpoints(
    output_frequency: float,
    *,
    first_mode_frequency: float,
    second_mode_frequency: float,
    lead_coupling: float,
    first_sign: int,
    second_sign: int | None = None,
) -> tuple[float, ...]:
    """Return angle points where the second frequency meets a band edge."""

    second_edges = _positive_band_edges(
        second_mode_frequency,
        lead_coupling,
    )
    signs = (-1, 1) if second_sign is None else (second_sign,)
    points: list[float] = []
    for sign in signs:
        for edge in second_edges:
            first_frequency = (
                output_frequency - sign * edge
            ) / first_sign
            if first_frequency <= 0.0:
                continue
            angle = _angle_from_frequency(
                first_frequency,
                mode_frequency=first_mode_frequency,
                lead_coupling=lead_coupling,
            )
            if angle is not None and 0.0 < angle < np.pi:
                points.append(angle)
    return tuple(sorted(set(points)))


def _adaptive_spectral_difference(
    output_frequency: float,
    *,
    first_mode_frequency: float,
    second_mode_frequency: float,
    lead_coupling: float,
    temperature: float,
    coupling: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float]:
    r"""Return ``B`` and its error estimate for ``Sigma^>-Sigma^<=-i B``."""

    total = 0.0
    error = 0.0
    second_low, second_high = _positive_band_edges(
        second_mode_frequency,
        lead_coupling,
    )
    for first_sign in (-1, 1):
        breakpoints = _outer_breakpoints(
            output_frequency,
            first_mode_frequency=first_mode_frequency,
            second_mode_frequency=second_mode_frequency,
            lead_coupling=lead_coupling,
            first_sign=first_sign,
        )

        def integrand(angle: float) -> float:
            first_frequency = _frequency_from_angle(
                angle,
                mode_frequency=first_mode_frequency,
                lead_coupling=lead_coupling,
            )
            signed_second = output_frequency - first_sign * first_frequency
            if signed_second == 0.0:
                return 0.0
            second_sign = 1 if signed_second > 0.0 else -1
            second_frequency = abs(signed_second)
            if (
                second_frequency <= second_low
                or second_frequency >= second_high
            ):
                return 0.0
            second_spectral = _positive_bulk_spectral(
                second_frequency,
                mode_frequency=second_mode_frequency,
                lead_coupling=lead_coupling,
            )
            weight = _channel_spectral_weight(
                first_frequency,
                second_frequency,
                first_sign=first_sign,
                second_sign=second_sign,
                temperature=temperature,
            )
            # A_first dnu = d(angle)/nu for the matched infinite chain.
            return second_spectral * weight / first_frequency

        value, estimate = quad(
            integrand,
            0.0,
            np.pi,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
            points=breakpoints or None,
        )
        total += value
        error += estimate
    scale = coupling**2 / (2.0 * np.pi)
    return float(scale * total), float(abs(scale) * error)


def _adaptive_pv_real(
    output_frequency: float,
    *,
    first_mode_frequency: float,
    second_mode_frequency: float,
    lead_coupling: float,
    temperature: float,
    coupling: float,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[float, float, float]:
    """Return the on-shell real bubble from nested adaptive PV quadrature."""

    total = 0.0
    outer_error = 0.0
    maximum_inner_error = 0.0
    prefactor = coupling**2 / (2.0 * np.pi) ** 2

    for first_sign in (-1, 1):
        for second_sign in (-1, 1):
            breakpoints = _outer_breakpoints(
                output_frequency,
                first_mode_frequency=first_mode_frequency,
                second_mode_frequency=second_mode_frequency,
                lead_coupling=lead_coupling,
                first_sign=first_sign,
                second_sign=second_sign,
            )

            def outer_integrand(first_angle: float) -> float:
                nonlocal maximum_inner_error
                first_frequency = _frequency_from_angle(
                    first_angle,
                    mode_frequency=first_mode_frequency,
                    lead_coupling=lead_coupling,
                )
                pole_frequency = second_sign * (
                    output_frequency - first_sign * first_frequency
                )
                pole_angle = _angle_from_frequency(
                    pole_frequency,
                    mode_frequency=second_mode_frequency,
                    lead_coupling=lead_coupling,
                )

                def numerator(second_angle: float) -> float:
                    second_frequency = _frequency_from_angle(
                        second_angle,
                        mode_frequency=second_mode_frequency,
                        lead_coupling=lead_coupling,
                    )
                    weight = _channel_spectral_weight(
                        first_frequency,
                        second_frequency,
                        first_sign=first_sign,
                        second_sign=second_sign,
                        temperature=temperature,
                    )
                    return weight / second_frequency

                if (
                    pole_angle is not None
                    and 0.0 < pole_angle < np.pi
                ):
                    derivative = (
                        lead_coupling
                        * np.sin(pole_angle)
                        / pole_frequency
                    )

                    def cauchy_numerator(second_angle: float) -> float:
                        distance = second_angle - pole_angle
                        if abs(distance) <= 1.0e-7:
                            quotient = derivative
                        else:
                            second_frequency = _frequency_from_angle(
                                second_angle,
                                mode_frequency=second_mode_frequency,
                                lead_coupling=lead_coupling,
                            )
                            quotient = (
                                second_frequency - pole_frequency
                            ) / distance
                        return (
                            -second_sign
                            * numerator(second_angle)
                            / quotient
                        )

                    inner, inner_error = quad(
                        cauchy_numerator,
                        0.0,
                        np.pi,
                        weight="cauchy",
                        wvar=pole_angle,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )
                else:

                    def regular(second_angle: float) -> float:
                        second_frequency = _frequency_from_angle(
                            second_angle,
                            mode_frequency=second_mode_frequency,
                            lead_coupling=lead_coupling,
                        )
                        denominator = (
                            output_frequency
                            - first_sign * first_frequency
                            - second_sign * second_frequency
                        )
                        return numerator(second_angle) / denominator

                    inner, inner_error = quad(
                        regular,
                        0.0,
                        np.pi,
                        epsabs=epsabs,
                        epsrel=epsrel,
                        limit=limit,
                    )
                maximum_inner_error = max(
                    maximum_inner_error,
                    float(inner_error),
                )
                return inner / first_frequency

            channel, channel_error = quad(
                outer_integrand,
                0.0,
                np.pi,
                epsabs=epsabs,
                epsrel=epsrel,
                limit=limit,
                points=breakpoints or None,
            )
            total += channel
            outer_error += channel_error

    return (
        float(prefactor * total),
        float(abs(prefactor) * outer_error),
        float(abs(prefactor) * maximum_inner_error),
    )


def _adaptive_on_shell_self_energy(
    fixture: "OpenThreeModeFixture",
    *,
    epsabs: float,
    epsrel: float,
    limit: int,
) -> tuple[Array, Array]:
    """Evaluate all on-shell retarded components in the eta-free continuum."""

    frequencies = fixture.mode_frequencies
    values = np.empty(3, dtype=np.complex128)
    errors = np.empty((3, 3), dtype=np.float64)
    for target_mode, output_frequency in enumerate(frequencies):
        internal = tuple(mode for mode in range(3) if mode != target_mode)
        common = {
            "output_frequency": output_frequency,
            "first_mode_frequency": frequencies[internal[0]],
            "second_mode_frequency": frequencies[internal[1]],
            "lead_coupling": fixture.lead_coupling,
            "temperature": fixture.temperature,
            "coupling": fixture.g,
            "epsabs": epsabs,
            "epsrel": epsrel,
            "limit": limit,
        }
        spectral, spectral_error = _adaptive_spectral_difference(**common)
        real, outer_error, inner_error = _adaptive_pv_real(**common)
        values[target_mode] = real - 0.5j * spectral
        errors[target_mode] = (
            outer_error,
            0.5 * spectral_error,
            inner_error,
        )
    return values, errors


def _diagonal_spectrum(values: Array) -> Array:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("diagonal values must have shape (n_frequency, 3).")
    result = np.zeros((array.shape[0], 3, 3), dtype=np.complex128)
    indices = np.arange(3)
    result[:, indices, indices] = array
    return result


def _zero_extended_scalar_convolution(
    first: Array,
    second: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    """Evaluate ``integral first(w') second(w-w') dw'/(2*pi)``."""

    left = np.asarray(first, dtype=np.complex128)
    right = np.asarray(second, dtype=np.complex128)
    if left.shape != (grid.n_points,) or right.shape != (grid.n_points,):
        raise ValueError(
            "scalar convolution inputs must match the signed frequency grid."
        )

    if grid.n_points >= _GRID_FFT_THRESHOLD:
        padded = next_fast_len(2 * grid.n_points - 1)
        full = ifft(
            fft(left, n=padded, workers=1)
            * fft(right, n=padded, workers=1),
            workers=1,
        )
        start = grid.zero_index
        return np.asarray(
            grid.integration_weight
            * full[start : start + grid.n_points],
            dtype=np.complex128,
        )

    result = np.zeros(grid.n_points, dtype=np.complex128)
    zero = grid.zero_index
    for output_index in range(grid.n_points):
        # second_index = output_index - first_index + zero
        first_start = max(0, output_index + zero - (grid.n_points - 1))
        first_stop = min(grid.n_points, output_index + zero + 1)
        first_indices = np.arange(first_start, first_stop, dtype=np.int64)
        second_indices = output_index - first_indices + zero
        result[output_index] = grid.integration_weight * np.dot(
            left[first_indices],
            right[second_indices],
        )
    return result


def _grid_retarded(
    lesser: Array,
    greater: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    """Reconstruct diagonal retarded spectra from the finite-grid PV sum."""

    sigma_lesser = np.asarray(lesser, dtype=np.complex128)
    sigma_greater = np.asarray(greater, dtype=np.complex128)
    expected = (grid.n_points, 3)
    if sigma_lesser.shape != expected or sigma_greater.shape != expected:
        raise ValueError(f"self-energy diagonals must have shape {expected}.")

    difference = sigma_greater - sigma_lesser
    retarded = 0.5 * difference
    if grid.n_points >= _GRID_FFT_THRESHOLD:
        padded = next_fast_len(2 * grid.n_points - 1)
        offsets = np.arange(1, grid.n_points, dtype=np.float64)
        kernel = np.zeros(padded, dtype=np.float64)
        weights = grid.integration_weight / (grid.step * offsets)
        kernel[1 : grid.n_points] = weights
        kernel[padded - np.arange(1, grid.n_points)] = -weights
        transform = fft(kernel, workers=1)[:, None]
        principal_value = ifft(
            transform
            * fft(
                difference,
                n=padded,
                axis=0,
                workers=1,
            ),
            axis=0,
            workers=1,
        )[: grid.n_points]
        return np.asarray(
            retarded + 1.0j * principal_value,
            dtype=np.complex128,
        )

    all_indices = np.arange(grid.n_points, dtype=np.int64)
    for output_index, output_frequency in enumerate(grid.values):
        keep = all_indices != output_index
        weights = (
            grid.integration_weight
            / (output_frequency - grid.values[keep])
        )
        retarded[output_index] += 1.0j * np.einsum(
            "j,jm->m",
            weights,
            difference[keep],
            optimize=True,
        )
    return retarded


def measure_mode_poles(
    grid: SignedFrequencyGrid,
    mode_frequencies: Any,
    sigma_retarded: Array,
) -> tuple[Array, Array]:
    r"""Return Born frequency shifts and linewidths for all three modes.

    The returned pair is ``(frequency_shifts, linewidths)`` with

    .. math::

       \Delta\Omega_s =
       \frac{\operatorname{Re}\Sigma^r_{ss}(\Omega_s)}{2\Omega_s},
       \qquad
       \gamma_s =
       -\frac{\operatorname{Im}\Sigma^r_{ss}(\Omega_s)}{\Omega_s}.

    Linear interpolation permits grid-refinement studies whose mode
    frequencies are not exact grid bins.  These are weak-coupling on-shell
    pole estimates, not nonlinear roots of the fully dressed Dyson equation.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    frequencies = np.asarray(
        _validated_frequencies(mode_frequencies),
        dtype=np.float64,
    )
    raw = np.asarray(sigma_retarded, dtype=np.complex128)
    if raw.shape == (grid.n_points, 3, 3):
        diagonal = np.diagonal(raw, axis1=1, axis2=2)
    elif raw.shape == (grid.n_points, 3):
        diagonal = raw
    else:
        raise ValueError(
            "sigma_retarded must have shape (n_frequency,3) or "
            "(n_frequency,3,3)."
        )
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("sigma_retarded must contain only finite values.")
    if np.any(frequencies >= grid.cutoff):
        raise ValueError("every mode frequency must lie below the grid cutoff.")

    on_shell = np.empty(3, dtype=np.complex128)
    for mode, frequency in enumerate(frequencies):
        on_shell[mode] = (
            np.interp(frequency, grid.values, diagonal[:, mode].real)
            + 1.0j
            * np.interp(frequency, grid.values, diagonal[:, mode].imag)
        )
    shifts = on_shell.real / (2.0 * frequencies)
    linewidths = -on_shell.imag / frequencies
    return (
        _readonly(shifts, dtype=np.dtype(np.float64), name="frequency_shifts"),
        _readonly(linewidths, dtype=np.dtype(np.float64), name="linewidths"),
    )


@dataclass(frozen=True, slots=True, eq=False)
class OpenThreeModeFixture:
    """Matched open three-mode problem in reduced, unit-mass coordinates."""

    grid: SignedFrequencyGrid = field(
        default_factory=lambda: SignedFrequencyGrid(
            step=0.015,
            n_positive=334,
        )
    )
    temperature: float = 0.7
    g: float = 0.02
    eta_lead: float = 1.0e-3
    mode_frequencies: tuple[float, float, float] = _DEFAULT_FREQUENCIES
    lead_coupling: float = 0.1
    backend: Literal["serial", "threaded", "fft"] = "serial"
    hilbert_backend: Literal["direct", "fft"] = "direct"
    surface_gf_method: str = "sancho_rubio"
    problem: CubicSCBAProblem = field(init=False, repr=False, compare=False)
    config: SCBAConfig = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid.")
        temperature = _finite_real(
            self.temperature,
            name="temperature",
            nonnegative=True,
        )
        cubic = _finite_real(self.g, name="g")
        eta = _finite_real(self.eta_lead, name="eta_lead", positive=True)
        lead_coupling = _finite_real(
            self.lead_coupling,
            name="lead_coupling",
            positive=True,
        )
        frequencies = _validated_frequencies(self.mode_frequencies)
        if self.backend not in {"serial", "threaded", "fft"}:
            raise ValueError(
                "backend must be 'serial', 'threaded', or 'fft'."
            )
        if self.hilbert_backend not in {"direct", "fft"}:
            raise ValueError("hilbert_backend must be 'direct' or 'fft'.")
        if (
            not isinstance(self.surface_gf_method, str)
            or not self.surface_gf_method
        ):
            raise ValueError("surface_gf_method must be a non-empty string.")
        if self.grid.cutoff <= max(frequencies):
            raise ValueError(
                "the signed frequency cutoff must exceed every mode frequency."
            )

        onsite = np.diag(np.square(frequencies)).astype(np.float64)
        hopping = -lead_coupling * np.eye(3, dtype=np.float64)
        lead = LeadBlocks(d00=onsite.copy(), d01=hopping.copy())
        device = Device1D(
            onsite_blocks=[onsite.copy()],
            coupling_blocks=[],
        )

        vertex_values = np.zeros((3, 3, 3), dtype=np.float64)
        for indices in permutations((0, 1, 2)):
            vertex_values[indices] = cubic
        vertex = DenseCubicVertex(
            vertex_values,
            metadata={
                "model": "matched_open_three_mode",
                "interaction": "g*u_x*u_y*u_z",
                "mass_weighted": True,
                "masses": (1.0, 1.0, 1.0),
                "mode_frequencies": frequencies,
                "lead_coupling": lead_coupling,
                "harmonic_bond_current_supported": True,
            },
        )
        problem = CubicSCBAProblem(
            device=device,
            left=HarmonicReservoir(lead, temperature),
            right=HarmonicReservoir(lead, temperature),
            vertex=vertex,
            grid=self.grid,
        )
        config = SCBAConfig(
            eta_lead=eta,
            retarded_strategy="causal_hilbert",
            backend=self.backend,
            hilbert_backend=self.hilbert_backend,
            surface_gf_method=self.surface_gf_method,
        )

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "g", cubic)
        object.__setattr__(self, "eta_lead", eta)
        object.__setattr__(self, "mode_frequencies", frequencies)
        object.__setattr__(self, "lead_coupling", lead_coupling)
        object.__setattr__(self, "problem", problem)
        object.__setattr__(self, "config", config)

    @property
    def masses(self) -> tuple[float, float, float]:
        return (1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True, eq=False)
class OpenThreeModeOracleResult:
    """Independent harmonic-continuum Born spectrum and pole estimates."""

    fixture: OpenThreeModeFixture
    green_zero: KeldyshSpectrum
    sigma_pp: KeldyshSpectrum
    self_energy_on_shell: Array = field(repr=False, compare=False)
    frequency_shifts: Array = field(repr=False, compare=False)
    linewidths: Array = field(repr=False, compare=False)
    grid_frequency_shifts: Array = field(repr=False, compare=False)
    grid_linewidths: Array = field(repr=False, compare=False)
    integration_error_estimates: Array = field(repr=False, compare=False)
    weak_coupling_limit: float = 0.1
    frequency_shift_atol: float = 1.0e-8
    method_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fixture, OpenThreeModeFixture):
            raise TypeError("fixture must be an OpenThreeModeFixture.")
        expected = (self.fixture.grid.n_points, 3, 3)
        if self.green_zero.shape != expected or self.sigma_pp.shape != expected:
            raise ValueError(
                f"Green functions and self-energies must have shape {expected}."
            )
        on_shell = _readonly(
            self.self_energy_on_shell,
            dtype=np.dtype(np.complex128),
            name="self_energy_on_shell",
        )
        shifts = _readonly(
            self.frequency_shifts,
            dtype=np.dtype(np.float64),
            name="frequency_shifts",
        )
        linewidths = _readonly(
            self.linewidths,
            dtype=np.dtype(np.float64),
            name="linewidths",
        )
        grid_shifts = _readonly(
            self.grid_frequency_shifts,
            dtype=np.dtype(np.float64),
            name="grid_frequency_shifts",
        )
        grid_linewidths = _readonly(
            self.grid_linewidths,
            dtype=np.dtype(np.float64),
            name="grid_linewidths",
        )
        errors = _readonly(
            self.integration_error_estimates,
            dtype=np.dtype(np.float64),
            name="integration_error_estimates",
        )
        if (
            on_shell.shape != (3,)
            or shifts.shape != (3,)
            or linewidths.shape != (3,)
            or grid_shifts.shape != (3,)
            or grid_linewidths.shape != (3,)
        ):
            raise ValueError(
                "on-shell self-energies and all pole arrays must have shape (3,)."
            )
        if errors.shape != (3, 3) or np.any(errors < 0.0):
            raise ValueError(
                "integration_error_estimates must be a non-negative (3,3) array."
            )
        limit = _finite_real(
            self.weak_coupling_limit,
            name="weak_coupling_limit",
            positive=True,
        )
        shift_atol = _finite_real(
            self.frequency_shift_atol,
            name="frequency_shift_atol",
            positive=True,
        )
        if not isinstance(self.method_metadata, Mapping):
            raise TypeError("method_metadata must be a mapping.")
        metadata = MappingProxyType(
            {str(key): value for key, value in self.method_metadata.items()}
        )
        object.__setattr__(self, "self_energy_on_shell", on_shell)
        object.__setattr__(self, "frequency_shifts", shifts)
        object.__setattr__(self, "linewidths", linewidths)
        object.__setattr__(self, "grid_frequency_shifts", grid_shifts)
        object.__setattr__(self, "grid_linewidths", grid_linewidths)
        object.__setattr__(self, "integration_error_estimates", errors)
        object.__setattr__(self, "weak_coupling_limit", limit)
        object.__setattr__(self, "frequency_shift_atol", shift_atol)
        object.__setattr__(self, "method_metadata", metadata)

    @property
    def pole_frequencies(self) -> Array:
        return _readonly(
            np.asarray(self.fixture.mode_frequencies) + self.frequency_shifts,
            dtype=np.dtype(np.float64),
            name="pole_frequencies",
        )

    @property
    def relative_shifts(self) -> Array:
        return np.abs(self.frequency_shifts) / np.asarray(
            self.fixture.mode_frequencies
        )

    @property
    def relative_linewidths(self) -> Array:
        return np.abs(self.linewidths) / np.asarray(
            self.fixture.mode_frequencies
        )

    @property
    def passive(self) -> bool:
        scale = max(1.0, float(np.max(np.abs(self.linewidths))))
        tolerance = 128.0 * np.finfo(float).eps * scale
        return bool(np.all(self.linewidths >= -tolerance))

    @property
    def weak_coupling_passed(self) -> bool:
        return bool(
            self.passive
            and np.all(self.relative_shifts < self.weak_coupling_limit)
            and np.all(self.relative_linewidths < self.weak_coupling_limit)
        )


def open_three_mode_oracle(
    fixture: OpenThreeModeFixture,
    *,
    weak_coupling_limit: float = 0.1,
    integration_epsabs: float = 1.0e-9,
    integration_epsrel: float = 1.0e-7,
    integration_limit: int = 200,
    frequency_shift_atol: float = 1.0e-8,
) -> OpenThreeModeOracleResult:
    """Evaluate the independent contact-dressed Born/FGR oracle."""

    if not isinstance(fixture, OpenThreeModeFixture):
        raise TypeError("fixture must be an OpenThreeModeFixture.")
    limit = _finite_real(
        weak_coupling_limit,
        name="weak_coupling_limit",
        positive=True,
    )
    epsabs = _finite_real(
        integration_epsabs,
        name="integration_epsabs",
        positive=True,
    )
    epsrel = _finite_real(
        integration_epsrel,
        name="integration_epsrel",
        positive=True,
    )
    shift_atol = _finite_real(
        frequency_shift_atol,
        name="frequency_shift_atol",
        positive=True,
    )
    if (
        isinstance(integration_limit, (bool, np.bool_))
        or not isinstance(integration_limit, (int, np.integer))
    ):
        raise TypeError("integration_limit must be an integer.")
    integration_limit = int(integration_limit)
    if integration_limit < 50:
        raise ValueError("integration_limit must be at least 50.")
    grid = fixture.grid
    omega = grid.values
    frequencies = np.asarray(fixture.mode_frequencies, dtype=np.float64)

    surface = np.empty((grid.n_points, 3), dtype=np.complex128)
    for mode, frequency in enumerate(frequencies):
        surface[:, mode] = _analytic_scalar_surface(
            omega,
            onsite=frequency**2,
            hopping=fixture.lead_coupling,
            eta=fixture.eta_lead,
        )
    contact_retarded = fixture.lead_coupling**2 * surface
    gamma = -2.0 * np.imag(contact_retarded)
    occupation = _signed_bose(omega, fixture.temperature)
    contact_lesser = np.zeros_like(contact_retarded)
    contact_greater = np.zeros_like(contact_retarded)
    nonzero = np.arange(grid.n_points) != grid.zero_index
    contact_lesser[nonzero] = (
        -1.0j * occupation[nonzero, None] * gamma[nonzero]
    )
    contact_greater[nonzero] = (
        -1.0j * (1.0 + occupation[nonzero, None]) * gamma[nonzero]
    )

    total_contact_retarded = 2.0 * contact_retarded
    total_contact_lesser = 2.0 * contact_lesser
    total_contact_greater = 2.0 * contact_greater
    green_retarded = 1.0 / (
        omega[:, None] ** 2
        - frequencies[None, :] ** 2
        - total_contact_retarded
    )
    green_lesser = (
        green_retarded
        * total_contact_lesser
        * green_retarded.conj()
    )
    green_greater = (
        green_retarded
        * total_contact_greater
        * green_retarded.conj()
    )

    sigma_lesser = np.empty_like(green_lesser)
    sigma_greater = np.empty_like(green_greater)
    for target_mode in range(3):
        internal = tuple(mode for mode in range(3) if mode != target_mode)
        # The production i/2 bubble contains the two ordered internal pairs.
        # Their equal scalar convolutions combine to the fixed coefficient i.
        sigma_lesser[:, target_mode] = (
            1.0j
            * fixture.g**2
            * _zero_extended_scalar_convolution(
                green_lesser[:, internal[0]],
                green_lesser[:, internal[1]],
                grid,
            )
        )
        sigma_greater[:, target_mode] = (
            1.0j
            * fixture.g**2
            * _zero_extended_scalar_convolution(
                green_greater[:, internal[0]],
                green_greater[:, internal[1]],
                grid,
            )
        )
    sigma_retarded = _grid_retarded(
        sigma_lesser,
        sigma_greater,
        grid,
    )

    green = KeldyshSpectrum(
        retarded=_diagonal_spectrum(green_retarded),
        lesser=_diagonal_spectrum(green_lesser),
        greater=_diagonal_spectrum(green_greater),
    )
    sigma = KeldyshSpectrum(
        retarded=_diagonal_spectrum(sigma_retarded),
        lesser=_diagonal_spectrum(sigma_lesser),
        greater=_diagonal_spectrum(sigma_greater),
    )
    grid_shifts, grid_linewidths = measure_mode_poles(
        grid,
        frequencies,
        sigma.retarded,
    )
    self_energy_on_shell, integration_errors = (
        _adaptive_on_shell_self_energy(
            fixture,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=integration_limit,
        )
    )
    shifts = self_energy_on_shell.real / (2.0 * frequencies)
    linewidths = -self_energy_on_shell.imag / frequencies
    return OpenThreeModeOracleResult(
        fixture=fixture,
        green_zero=green,
        sigma_pp=sigma,
        self_energy_on_shell=self_energy_on_shell,
        frequency_shifts=shifts,
        linewidths=linewidths,
        grid_frequency_shifts=grid_shifts,
        grid_linewidths=grid_linewidths,
        integration_error_estimates=integration_errors,
        weak_coupling_limit=limit,
        frequency_shift_atol=shift_atol,
        method_metadata={
            "pole_method": "adaptive_eta_free_continuum_on_shell",
            "surface_spectrum": "analytic_matched_scalar_lead",
            "convolution": (
                "scipy.integrate.quad over positive-band angle with "
                "explicit signed Bose channels"
            ),
            "principal_value": (
                "nested adaptive quadrature with scipy cauchy weight "
                "at internal on-shell poles"
            ),
            "grid_spectra_role": (
                "diagnostic production-regression only; not pole acceptance"
            ),
            "grid_convolution_backend": (
                "independent_zero_padded_fft"
                if grid.n_points >= _GRID_FFT_THRESHOLD
                else "independent_direct_sum"
            ),
            "grid_hilbert_backend": (
                "independent_toeplitz_fft"
                if grid.n_points >= _GRID_FFT_THRESHOLD
                else "independent_direct_sum"
            ),
            "integration_epsabs": epsabs,
            "integration_epsrel": epsrel,
            "integration_limit": integration_limit,
            "integration_error_columns": (
                "pv_outer_absolute",
                "imaginary_half_spectral_absolute",
                "pv_max_inner_absolute",
            ),
            "frequency_shift_comparison_atol": shift_atol,
            "eta_used_for_adaptive_poles": 0.0,
            "eta_used_for_grid_diagnostics": fixture.eta_lead,
        },
    )


__all__ = [
    "OpenThreeModeFixture",
    "OpenThreeModeOracleResult",
    "measure_mode_poles",
    "open_three_mode_oracle",
]
