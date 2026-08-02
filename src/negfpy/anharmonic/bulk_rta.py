"""Bulk three-phonon single-mode relaxation-time transport.

The implementation consumes the independently validated phono3py-compatible
linewidth result.  It is a bulk phonon-BTE approximation and is deliberately
kept separate from the open-device anharmonic NEGF solver.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .bulk_harmonic import (
    BulkGroupVelocities,
    BulkPhononSpectrum,
    compute_bulk_group_velocities,
    compute_bulk_phonons,
)
from .bulk_three_phonon import (
    PHONO3PY_4_4_EV_J,
    PHONO3PY_4_4_KB_EV_K,
    PHONO3PY_4_4_THZ_TO_EV,
    BulkThreePhononLinewidths,
    BulkThreePhononProblem,
    compute_three_phonon_linewidths,
)

Array = np.ndarray
PHONO3PY_4_4_RTA_CONVERSION = PHONO3PY_4_4_EV_J * 1.0e22
VOIGT_INDEX_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _readonly(values: Any, *, dtype: Any | None = None) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _freeze_metadata(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_metadata(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _temperatures(values: Any) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind == "c" or raw.ndim != 1:
        raise TypeError("temperatures must be a one-dimensional real array.")
    result = np.array(values, dtype=np.float64, order="C", copy=True)
    if result.size < 1 or not np.all(np.isfinite(result)) or np.any(result < 0.0):
        raise ValueError("temperatures must be nonempty, finite, and nonnegative.")
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BulkModeHeatCapacities:
    """Quantum harmonic heat capacity of every bulk phonon mode."""

    temperatures: Array
    frequencies_thz: Array
    values_ev_per_k: Array
    values_j_per_k: Array
    included_mask: Array
    cutoff_frequency_thz: float

    def __post_init__(self) -> None:
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        frequencies = _readonly(self.frequencies_thz, dtype=np.float64)
        ev = _readonly(self.values_ev_per_k, dtype=np.float64)
        joule = _readonly(self.values_j_per_k, dtype=np.float64)
        included = _readonly(self.included_mask, dtype=np.bool_)
        if temperatures.ndim != 1 or temperatures.size < 1:
            raise ValueError("temperatures must be a nonempty vector.")
        if frequencies.ndim != 2:
            raise ValueError("frequencies_thz must have shape (n_q, n_band).")
        if ev.shape != (temperatures.size,) + frequencies.shape:
            raise ValueError("Heat capacities have an inconsistent shape.")
        if joule.shape != ev.shape or included.shape != frequencies.shape:
            raise ValueError("Heat-capacity fields have inconsistent shapes.")
        if not all(np.all(np.isfinite(v)) for v in (temperatures, frequencies, ev, joule)):
            raise ValueError("Heat-capacity fields must be finite.")
        if np.any(ev < 0.0) or np.any(joule < 0.0):
            raise ValueError("Heat capacities must be nonnegative.")
        if not np.allclose(
            joule,
            ev * PHONO3PY_4_4_EV_J,
            rtol=1.0e-14,
            atol=0.0,
        ):
            raise ValueError("J/K heat capacities conflict with eV/K values.")
        cutoff = float(self.cutoff_frequency_thz)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff_frequency_thz must be nonnegative.")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "frequencies_thz", frequencies)
        object.__setattr__(self, "values_ev_per_k", ev)
        object.__setattr__(self, "values_j_per_k", joule)
        object.__setattr__(self, "included_mask", included)
        object.__setattr__(self, "cutoff_frequency_thz", cutoff)


def compute_mode_heat_capacities(
    frequencies_thz: Any,
    temperatures: Any,
    *,
    cutoff_frequency_thz: float = 1.0e-4,
) -> BulkModeHeatCapacities:
    r"""Return stable quantum mode heat capacities.

    The evaluated expression is
    :math:`C=k_B x^2 e^x/(e^x-1)^2`, where
    :math:`x=h\nu/(k_B T)`.  Small and large ``x`` use stable equivalent
    forms, and the zero-temperature result is evaluated analytically.
    """

    raw_frequencies = np.asarray(frequencies_thz)
    if raw_frequencies.dtype.kind == "c" or raw_frequencies.ndim != 2:
        raise TypeError("frequencies_thz must be a two-dimensional real array.")
    frequencies = np.array(frequencies_thz, dtype=np.float64, order="C", copy=True)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequencies_thz must be finite.")
    temperature_array = _temperatures(temperatures)
    cutoff = float(cutoff_frequency_thz)
    if not np.isfinite(cutoff) or cutoff < 0.0:
        raise ValueError("cutoff_frequency_thz must be nonnegative.")
    included = frequencies > cutoff
    values = np.zeros((temperature_array.size,) + frequencies.shape, dtype=np.float64)
    log_frequency_temperature_scale = np.log(
        PHONO3PY_4_4_THZ_TO_EV / PHONO3PY_4_4_KB_EV_K
    )
    log_small_x = np.log(1.0e-4)
    log_largest_float = np.log(np.finfo(np.float64).max)
    log_smallest_normal = np.log(np.finfo(np.float64).tiny)
    log_kb = np.log(PHONO3PY_4_4_KB_EV_K)
    for position, temperature in enumerate(temperature_array):
        if temperature == 0.0:
            continue
        # Form log(x) directly.  Multiplying h*nu and k_B*T separately can
        # underflow for positive subnormal inputs, turning a finite ratio into
        # 0/0 or infinity before the stable heat-capacity formula is reached.
        log_x = (
            np.log(frequencies[included])
            + log_frequency_temperature_scale
            - np.log(temperature)
        )
        included_values = np.zeros(log_x.shape, dtype=np.float64)
        small = log_x < log_small_x
        with np.errstate(under="ignore"):
            x_small = np.exp(log_x[small])
            included_values[small] = PHONO3PY_4_4_KB_EV_K * (
                1.0 - x_small**2 / 12.0 + x_small**4 / 240.0
            )

        # For finite representable x, evaluate the regular branch in log
        # space.  Values smaller than the least positive normal float are set
        # to their analytic large-x limit, zero, rather than manufacturing a
        # subnormal result that can raise an underflow exception downstream.
        regular = ~small & (log_x <= log_largest_float)
        x_regular = np.exp(log_x[regular])
        log_capacity = (
            log_kb
            + 2.0 * log_x[regular]
            - x_regular
            - 2.0 * np.log(-np.expm1(-x_regular))
        )
        representable = log_capacity >= log_smallest_normal
        regular_values = np.zeros(log_capacity.shape, dtype=np.float64)
        regular_values[representable] = np.exp(log_capacity[representable])
        included_values[regular] = regular_values
        values[position][included] = included_values
    return BulkModeHeatCapacities(
        temperatures=temperature_array,
        frequencies_thz=frequencies,
        values_ev_per_k=values,
        values_j_per_k=values * PHONO3PY_4_4_EV_J,
        included_mask=included,
        cutoff_frequency_thz=cutoff,
    )


@dataclass(frozen=True, slots=True, eq=False)
class BulkThreePhononRateSpectrum:
    """Lightweight, portable three-phonon rates for bulk RTA post-processing.

    Unlike :class:`BulkThreePhononLinewidths`, this object deliberately omits
    mode-space FC3 interaction tensors.  It retains only the full harmonic
    mesh and the selected on-shell rate fields, reducing an 11x11x11 silicon
    post-processing input from several gigabytes to well below one megabyte.
    ``physics_input_sha256`` binds the rates to portable physical inputs rather
    than absolute source paths.
    """

    temperatures: Array
    qpoints: Array
    frequencies_thz: Array
    grid_points: Array
    band_indices: Array
    gamma_thz_hwhm: Array
    gamma_normal_thz_hwhm: Array
    gamma_umklapp_thz_hwhm: Array
    gamma_angular_fwhm: Array
    decay_phase_space: Array
    difference_phase_space: Array
    cutoff_frequency_thz: float
    frequency_factor_to_thz: float
    physics_input_sha256: str
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        qpoints = _readonly(self.qpoints, dtype=np.float64)
        frequencies = _readonly(self.frequencies_thz, dtype=np.float64)
        grid_points = _readonly(self.grid_points, dtype=np.int64)
        bands = _readonly(self.band_indices, dtype=np.int64)
        gamma = _readonly(self.gamma_thz_hwhm, dtype=np.float64)
        gamma_n = _readonly(self.gamma_normal_thz_hwhm, dtype=np.float64)
        gamma_u = _readonly(self.gamma_umklapp_thz_hwhm, dtype=np.float64)
        angular = _readonly(self.gamma_angular_fwhm, dtype=np.float64)
        decay = _readonly(self.decay_phase_space, dtype=np.float64)
        difference = _readonly(self.difference_phase_space, dtype=np.float64)
        if (
            temperatures.ndim != 1
            or temperatures.size < 1
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise ValueError(
                "temperatures must be a nonempty finite nonnegative vector."
            )
        if qpoints.ndim != 2 or qpoints.shape[1:] != (3,):
            raise ValueError("qpoints must have shape (n_qpoints, 3).")
        if frequencies.ndim != 2 or frequencies.shape[0] != qpoints.shape[0]:
            raise ValueError(
                "frequencies_thz must have shape (n_qpoints, n_bands)."
            )
        if not np.all(np.isfinite(qpoints)) or not np.all(np.isfinite(frequencies)):
            raise ValueError("qpoints and frequencies must be finite.")
        if grid_points.ndim != 1 or grid_points.size < 1:
            raise ValueError("grid_points must be a nonempty vector.")
        if bands.ndim != 1 or bands.size < 1:
            raise ValueError("band_indices must be a nonempty vector.")
        if (
            np.unique(grid_points).size != grid_points.size
            or np.any((grid_points < 0) | (grid_points >= qpoints.shape[0]))
        ):
            raise ValueError("grid_points must be unique valid mesh indices.")
        if (
            np.unique(bands).size != bands.size
            or np.any((bands < 0) | (bands >= frequencies.shape[1]))
        ):
            raise ValueError("band_indices must be unique valid band indices.")
        expected = (temperatures.size, grid_points.size, bands.size)
        for name, values in (
            ("gamma_thz_hwhm", gamma),
            ("gamma_normal_thz_hwhm", gamma_n),
            ("gamma_umklapp_thz_hwhm", gamma_u),
            ("gamma_angular_fwhm", angular),
        ):
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape {expected}.")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be nonnegative.")
        if not np.allclose(gamma, gamma_n + gamma_u, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError("Total gamma must equal normal plus Umklapp gamma.")
        if not np.allclose(angular, 4.0 * np.pi * gamma, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError("Angular FWHM must equal 4*pi*cyclic-THz HWHM.")
        phase_shape = (grid_points.size, bands.size)
        if decay.shape != phase_shape or difference.shape != phase_shape:
            raise ValueError("Phase-space arrays have inconsistent shape.")
        if not np.all(np.isfinite(decay)) or not np.all(np.isfinite(difference)):
            raise ValueError("Phase-space arrays must be finite.")
        if np.any(decay < 0.0):
            raise ValueError("Decay phase space must be nonnegative.")
        cutoff = float(self.cutoff_frequency_thz)
        factor = float(self.frequency_factor_to_thz)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff_frequency_thz must be nonnegative.")
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("frequency_factor_to_thz must be positive.")
        identity = str(self.physics_input_sha256)
        if len(identity) != 64:
            raise ValueError("physics_input_sha256 must contain 64 hex digits.")
        try:
            int(identity, 16)
        except ValueError as exc:
            raise ValueError(
                "physics_input_sha256 must contain 64 hex digits."
            ) from exc
        identity = identity.lower()
        metadata = _freeze_metadata({} if self.metadata is None else self.metadata)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "frequencies_thz", frequencies)
        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "band_indices", bands)
        object.__setattr__(self, "gamma_thz_hwhm", gamma)
        object.__setattr__(self, "gamma_normal_thz_hwhm", gamma_n)
        object.__setattr__(self, "gamma_umklapp_thz_hwhm", gamma_u)
        object.__setattr__(self, "gamma_angular_fwhm", angular)
        object.__setattr__(self, "decay_phase_space", decay)
        object.__setattr__(self, "difference_phase_space", difference)
        object.__setattr__(self, "cutoff_frequency_thz", cutoff)
        object.__setattr__(self, "frequency_factor_to_thz", factor)
        object.__setattr__(self, "physics_input_sha256", identity)
        object.__setattr__(self, "metadata", metadata)


def summarize_three_phonon_linewidths(
    linewidths: BulkThreePhononLinewidths,
) -> BulkThreePhononRateSpectrum:
    """Discard FC3 interaction tensors after preserving validated rate fields."""

    if not isinstance(linewidths, BulkThreePhononLinewidths):
        raise TypeError("linewidths must be a BulkThreePhononLinewidths result.")
    identity = linewidths.metadata.get("physics_input_sha256")
    if not isinstance(identity, str):
        raise ValueError(  # noqa: TRY004 - invalid result state
            "linewidths do not carry a portable physical identity."
        )
    phonons = linewidths.interactions.phonons
    return BulkThreePhononRateSpectrum(
        temperatures=linewidths.temperatures,
        qpoints=phonons.qpoints,
        frequencies_thz=phonons.frequencies,
        grid_points=linewidths.grid_points,
        band_indices=linewidths.band_indices,
        gamma_thz_hwhm=linewidths.gamma_thz_hwhm,
        gamma_normal_thz_hwhm=linewidths.gamma_normal_thz_hwhm,
        gamma_umklapp_thz_hwhm=linewidths.gamma_umklapp_thz_hwhm,
        gamma_angular_fwhm=linewidths.gamma_angular_fwhm,
        decay_phase_space=linewidths.decay_phase_space,
        difference_phase_space=linewidths.difference_phase_space,
        cutoff_frequency_thz=float(
            linewidths.metadata["cutoff_frequency_thz"]
        ),
        frequency_factor_to_thz=phonons.frequency_factor_to_thz,
        physics_input_sha256=identity,
        metadata={
            "schema": "negfpy.bulk_three_phonon_rate_spectrum.v1",
            "source_linewidth_metadata": linewidths.metadata,
        },
    )


@dataclass(frozen=True, slots=True, eq=False)
class BulkRelaxationTimes:
    """Three-phonon population lifetimes with explicit linewidth semantics."""

    temperatures: Array
    frequencies_thz: Array
    gamma_thz_hwhm: Array
    linewidth_angular_fwhm_ps_inv: Array
    population_lifetime_ps: Array
    included_mask: Array
    resolved_mask: Array
    cutoff_frequency_thz: float
    gamma_floor_thz: float

    def __post_init__(self) -> None:
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        frequencies = _readonly(self.frequencies_thz, dtype=np.float64)
        gamma = _readonly(self.gamma_thz_hwhm, dtype=np.float64)
        angular = _readonly(self.linewidth_angular_fwhm_ps_inv, dtype=np.float64)
        lifetime = _readonly(self.population_lifetime_ps, dtype=np.float64)
        included = _readonly(self.included_mask, dtype=np.bool_)
        resolved = _readonly(self.resolved_mask, dtype=np.bool_)
        expected = (temperatures.size,) + frequencies.shape
        if frequencies.ndim != 2 or gamma.shape != expected:
            raise ValueError("Relaxation-time fields have inconsistent shapes.")
        if angular.shape != expected or lifetime.shape != expected:
            raise ValueError("Relaxation-time fields have inconsistent shapes.")
        if included.shape != frequencies.shape or resolved.shape != expected:
            raise ValueError("Relaxation masks have inconsistent shapes.")
        if not all(
            np.all(np.isfinite(values))
            for values in (temperatures, frequencies, gamma, angular)
        ):
            raise ValueError("Relaxation temperatures, frequencies, and linewidths must be finite.")
        if np.any(temperatures < 0.0):
            raise ValueError("Relaxation temperatures must be nonnegative.")
        if np.any(gamma < 0.0) or np.any(angular < 0.0):
            raise ValueError("Linewidths must be nonnegative.")
        if np.any(np.isnan(lifetime)) or np.any(lifetime < 0.0):
            raise ValueError("Lifetimes must be nonnegative or positive infinity.")
        if not np.allclose(angular, 4.0 * np.pi * gamma, rtol=1.0e-14, atol=0.0):
            raise ValueError("Angular FWHM must equal 4*pi*cyclic-THz HWHM.")
        if np.any(resolved & ~included[None, :, :]):
            raise ValueError("A below-cutoff mode cannot be resolved.")
        if np.any(resolved & ~np.isfinite(lifetime)):
            raise ValueError("Resolved modes must have finite lifetimes.")
        cutoff = float(self.cutoff_frequency_thz)
        floor = float(self.gamma_floor_thz)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff_frequency_thz must be nonnegative.")
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("gamma_floor_thz must be nonnegative.")
        expected_included = frequencies > cutoff
        if not np.array_equal(included, expected_included):
            raise ValueError(
                "included_mask must select exactly frequencies above the cutoff."
            )
        expected_resolved = included[None, :, :] & (gamma > floor)
        if not np.array_equal(resolved, expected_resolved):
            raise ValueError(
                "resolved_mask must select included linewidths above gamma_floor."
            )
        if not np.allclose(
            lifetime[resolved],
            1.0 / angular[resolved],
            rtol=1.0e-14,
            atol=0.0,
        ):
            raise ValueError("Resolved lifetime must equal 1/angular FWHM.")
        if np.any(lifetime[:, ~included] != 0.0):
            raise ValueError("Below-cutoff lifetimes must be exactly zero.")
        unresolved = included[None, :, :] & ~resolved
        if np.any(~np.isinf(lifetime[unresolved])):
            raise ValueError("Unresolved included lifetimes must be positive infinity.")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "frequencies_thz", frequencies)
        object.__setattr__(self, "gamma_thz_hwhm", gamma)
        object.__setattr__(self, "linewidth_angular_fwhm_ps_inv", angular)
        object.__setattr__(self, "population_lifetime_ps", lifetime)
        object.__setattr__(self, "included_mask", included)
        object.__setattr__(self, "resolved_mask", resolved)
        object.__setattr__(self, "cutoff_frequency_thz", cutoff)
        object.__setattr__(self, "gamma_floor_thz", floor)

    @property
    def unresolved_included_mask(self) -> Array:
        result = self.included_mask[None, :, :] & ~self.resolved_mask
        result.setflags(write=False)
        return result


def compute_three_phonon_lifetimes(
    linewidths: BulkThreePhononLinewidths | BulkThreePhononRateSpectrum,
    *,
    gamma_floor_thz: float = 0.0,
    negative_tolerance_thz: float = 1.0e-15,
) -> BulkRelaxationTimes:
    """Convert cyclic-THz HWHM linewidths to population lifetimes in ps."""

    if not isinstance(
        linewidths,
        (BulkThreePhononLinewidths, BulkThreePhononRateSpectrum),
    ):
        raise TypeError(
            "linewidths must be a BulkThreePhononLinewidths or "
            "BulkThreePhononRateSpectrum result."
        )
    floor = float(gamma_floor_thz)
    negative_tolerance = float(negative_tolerance_thz)
    if not np.isfinite(floor) or floor < 0.0:
        raise ValueError("gamma_floor_thz must be nonnegative.")
    if not np.isfinite(negative_tolerance) or negative_tolerance < 0.0:
        raise ValueError("negative_tolerance_thz must be nonnegative.")
    gamma = np.array(linewidths.gamma_thz_hwhm, dtype=np.float64, copy=True)
    if np.any(gamma < -negative_tolerance):
        raise ValueError("A three-phonon linewidth is significantly negative.")
    gamma[(gamma < 0.0) & (gamma >= -negative_tolerance)] = 0.0
    if isinstance(linewidths, BulkThreePhononLinewidths):
        all_frequencies = linewidths.interactions.phonons.frequencies
        cutoff = float(linewidths.metadata["cutoff_frequency_thz"])
    else:
        all_frequencies = linewidths.frequencies_thz
        cutoff = linewidths.cutoff_frequency_thz
    frequencies = all_frequencies[
        np.asarray(linewidths.grid_points, dtype=np.int64)
    ][:, np.asarray(linewidths.band_indices, dtype=np.int64)]
    included = frequencies > cutoff
    resolved = included[None, :, :] & (gamma > floor)
    angular = 4.0 * np.pi * gamma
    lifetime = np.zeros_like(gamma)
    lifetime[resolved] = 1.0 / angular[resolved]
    lifetime[included[None, :, :] & ~resolved] = np.inf
    return BulkRelaxationTimes(
        temperatures=linewidths.temperatures,
        frequencies_thz=frequencies,
        gamma_thz_hwhm=gamma,
        linewidth_angular_fwhm_ps_inv=angular,
        population_lifetime_ps=lifetime,
        included_mask=included,
        resolved_mask=resolved,
        cutoff_frequency_thz=cutoff,
        gamma_floor_thz=floor,
    )


@dataclass(frozen=True, slots=True, eq=False)
class BulkMeanFreePaths:
    """Mode-resolved vector and magnitude mean free paths."""

    temperatures: Array
    vector_angstrom: Array
    magnitude_angstrom: Array
    vector_m: Array
    magnitude_m: Array
    resolved_mask: Array

    def __post_init__(self) -> None:
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        vector_a = _readonly(self.vector_angstrom, dtype=np.float64)
        magnitude_a = _readonly(self.magnitude_angstrom, dtype=np.float64)
        vector_m = _readonly(self.vector_m, dtype=np.float64)
        magnitude_m = _readonly(self.magnitude_m, dtype=np.float64)
        resolved = _readonly(self.resolved_mask, dtype=np.bool_)
        if vector_a.ndim != 4 or vector_a.shape[-1] != 3:
            raise ValueError("vector_angstrom must have shape (n_t,n_q,n_b,3).")
        if magnitude_a.shape != vector_a.shape[:-1] or resolved.shape != magnitude_a.shape:
            raise ValueError("Mean-free-path fields have inconsistent shapes.")
        if vector_m.shape != vector_a.shape or magnitude_m.shape != magnitude_a.shape:
            raise ValueError("Mean-free-path SI fields have inconsistent shapes.")
        if np.any(np.isnan(vector_a)) or np.any(np.isnan(magnitude_a)):
            raise ValueError("Mean free paths must not contain NaN.")
        if not np.allclose(vector_m, vector_a * 1.0e-10, equal_nan=False):
            raise ValueError("vector_m conflicts with vector_angstrom.")
        if not np.allclose(magnitude_m, magnitude_a * 1.0e-10, equal_nan=False):
            raise ValueError("magnitude_m conflicts with magnitude_angstrom.")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "vector_angstrom", vector_a)
        object.__setattr__(self, "magnitude_angstrom", magnitude_a)
        object.__setattr__(self, "vector_m", vector_m)
        object.__setattr__(self, "magnitude_m", magnitude_m)
        object.__setattr__(self, "resolved_mask", resolved)


def compute_mean_free_paths(
    velocities: BulkGroupVelocities,
    relaxation_times: BulkRelaxationTimes,
) -> BulkMeanFreePaths:
    """Combine ``v[THz angstrom]`` and ``tau[ps]`` into MFPs."""

    if not isinstance(velocities, BulkGroupVelocities):
        raise TypeError("velocities must be a BulkGroupVelocities result.")
    if not isinstance(relaxation_times, BulkRelaxationTimes):
        raise TypeError("relaxation_times must be a BulkRelaxationTimes result.")
    if not np.allclose(
        velocities.frequencies,
        relaxation_times.frequencies_thz,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("Velocity and relaxation frequencies do not match.")
    shape = relaxation_times.population_lifetime_ps.shape + (3,)
    vector = np.zeros(shape, dtype=np.float64)
    resolved = relaxation_times.resolved_mask
    velocity_broadcast = np.broadcast_to(
        velocities.values_thz_angstrom[None, :, :, :],
        shape,
    )
    vector[resolved] = (
        velocity_broadcast[resolved]
        * relaxation_times.population_lifetime_ps[resolved][:, None]
    )
    unresolved = relaxation_times.unresolved_included_mask
    for axis in range(3):
        component = vector[..., axis]
        source = velocity_broadcast[..., axis]
        component[unresolved & (source != 0.0)] = np.copysign(
            np.inf,
            source[unresolved & (source != 0.0)],
        )
    magnitude = np.linalg.norm(vector, axis=-1)
    magnitude[unresolved] = np.inf
    return BulkMeanFreePaths(
        temperatures=relaxation_times.temperatures,
        vector_angstrom=vector,
        magnitude_angstrom=magnitude,
        vector_m=vector * 1.0e-10,
        magnitude_m=magnitude * 1.0e-10,
        resolved_mask=resolved,
    )


@dataclass(frozen=True, slots=True, eq=False)
class RTAThermalConductivity:
    """Intrinsic three-phonon SMRTA thermal-conductivity result."""

    temperatures: Array
    qpoints: Array
    frequencies_thz: Array
    group_velocities_thz_angstrom: Array
    heat_capacities_ev_per_k: Array
    gamma_thz_hwhm: Array
    population_lifetime_ps: Array
    mean_free_path_angstrom: Array
    mode_kappa_w_mk: Array
    band_index_kappa_w_mk: Array
    kappa_w_mk: Array
    qpoint_weights: Array
    primitive_volume_angstrom3: float
    cutoff_frequency_thz: float
    included_mask: Array
    approximation: str = "three_phonon_smrta"
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        qpoints = _readonly(self.qpoints, dtype=np.float64)
        frequencies = _readonly(self.frequencies_thz, dtype=np.float64)
        velocities = _readonly(self.group_velocities_thz_angstrom, dtype=np.float64)
        cv = _readonly(self.heat_capacities_ev_per_k, dtype=np.float64)
        gamma = _readonly(self.gamma_thz_hwhm, dtype=np.float64)
        lifetime = _readonly(self.population_lifetime_ps, dtype=np.float64)
        mfp = _readonly(self.mean_free_path_angstrom, dtype=np.float64)
        mode = _readonly(self.mode_kappa_w_mk, dtype=np.float64)
        band_index = _readonly(self.band_index_kappa_w_mk, dtype=np.float64)
        kappa = _readonly(self.kappa_w_mk, dtype=np.float64)
        weights = _readonly(self.qpoint_weights, dtype=np.float64)
        included = _readonly(self.included_mask, dtype=np.bool_)
        n_t = temperatures.size
        if (
            temperatures.ndim != 1
            or n_t < 1
            or np.any(temperatures < 0.0)
        ):
            raise ValueError(
                "temperatures must be a nonempty nonnegative vector."
            )
        n_q, n_b = frequencies.shape
        if qpoints.shape != (n_q, 3):
            raise ValueError("qpoints must have shape (n_q, 3).")
        if velocities.shape != (n_q, n_b, 3):
            raise ValueError("Velocity shape is inconsistent.")
        expected_mode = (n_t, n_q, n_b)
        if any(values.shape != expected_mode for values in (cv, gamma, lifetime, mfp)):
            raise ValueError("Mode-property shapes are inconsistent.")
        if mode.shape != expected_mode + (3, 3):
            raise ValueError("mode_kappa_w_mk has an inconsistent shape.")
        if band_index.shape != (n_t, n_b, 3, 3) or kappa.shape != (n_t, 3, 3):
            raise ValueError("Conductivity tensor shapes are inconsistent.")
        if weights.shape != (n_q,) or included.shape != (n_q, n_b):
            raise ValueError("Weight or inclusion-mask shape is inconsistent.")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                temperatures,
                qpoints,
                frequencies,
                velocities,
                cv,
                gamma,
                lifetime,
                mfp,
                mode,
                band_index,
                kappa,
                weights,
            )
        ):
            raise ValueError("A certified RTA result must be finite.")
        if np.any(cv < 0.0):
            raise ValueError("Mode heat capacities must be nonnegative.")
        spectral_scale = max(float(np.max(np.abs(mode))), 1.0)
        positivity_tolerance = 1.0e-12 * spectral_scale
        if np.any(np.diagonal(mode, axis1=-2, axis2=-1) < -positivity_tolerance):
            raise ValueError("Every mode-conductivity diagonal must be nonnegative.")
        kappa_eigenvalues = np.linalg.eigvalsh(
            (kappa + kappa.transpose(0, 2, 1)) / 2.0
        )
        if np.any(kappa_eigenvalues < -positivity_tolerance):
            raise ValueError("Thermal-conductivity tensors must be positive semidefinite.")
        if not np.allclose(
            np.sum(mode, axis=(1, 2)),
            kappa,
            rtol=1.0e-12,
            atol=1.0e-10,
        ):
            raise ValueError("Mode conductivities do not sum to total kappa.")
        if not np.allclose(
            np.sum(mode, axis=1),
            band_index,
            rtol=1.0e-12,
            atol=1.0e-10,
        ):
            raise ValueError("Mode conductivities do not sum by band index.")
        if not np.allclose(kappa, kappa.transpose(0, 2, 1), rtol=0.0, atol=1.0e-12):
            raise ValueError("Thermal conductivity must be symmetric.")
        if np.any(weights < 0.0) or not np.isclose(np.sum(weights), 1.0, atol=1.0e-14):
            raise ValueError("qpoint_weights must be normalized.")
        volume = float(self.primitive_volume_angstrom3)
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("primitive_volume_angstrom3 must be positive.")
        cutoff = float(self.cutoff_frequency_thz)
        if not np.isfinite(cutoff) or cutoff < 0.0:
            raise ValueError("cutoff_frequency_thz must be nonnegative.")
        expected_included = frequencies > cutoff
        if not np.array_equal(included, expected_included):
            raise ValueError(
                "included_mask must select exactly frequencies above the cutoff."
            )
        if np.any(frequencies < -max(1.0e-8, cutoff)):
            raise ValueError("RTA result contains a dynamically unstable mode.")
        included_3d = np.broadcast_to(included[None, :, :], gamma.shape)
        if np.any(gamma[included_3d] <= 0.0):
            raise ValueError("Every included RTA mode requires positive gamma.")
        if np.any(gamma[:, ~included] != 0.0):
            raise ValueError("Below-cutoff RTA gamma must be exactly zero.")
        expected_lifetime = np.zeros_like(lifetime)
        expected_lifetime[included_3d] = 1.0 / (
            4.0 * np.pi * gamma[included_3d]
        )
        if not np.allclose(
            lifetime,
            expected_lifetime,
            rtol=1.0e-13,
            atol=0.0,
        ):
            raise ValueError("RTA lifetime must equal 1/(4*pi*gamma).")
        expected_mfp = (
            np.linalg.norm(velocities, axis=2)[None, :, :] * lifetime
        )
        if not np.allclose(mfp, expected_mfp, rtol=1.0e-13, atol=1.0e-13):
            raise ValueError("RTA mean free path must equal |v|*lifetime.")
        expected_mode = (
            cv[:, :, :, None, None]
            * np.einsum("qbi,qbj->qbij", velocities, velocities)[
                None, :, :, :, :
            ]
            * lifetime[:, :, :, None, None]
            * weights[None, :, None, None, None]
            * PHONO3PY_4_4_RTA_CONVERSION
            / volume
        )
        expected_mode *= included[None, :, :, None, None]
        if not np.allclose(mode, expected_mode, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError("RTA mode conductivity fields are inconsistent.")
        if self.approximation != "three_phonon_smrta":
            raise ValueError("The v1 result must be explicitly three_phonon_smrta.")
        if self.metadata is not None and not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping or None.")
        metadata = _freeze_metadata(
            {} if self.metadata is None else self.metadata
        )
        if "cutoff_frequency_thz" in metadata and not np.isclose(
            float(metadata["cutoff_frequency_thz"]),
            cutoff,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("RTA metadata cutoff conflicts with the result.")
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "frequencies_thz", frequencies)
        object.__setattr__(self, "group_velocities_thz_angstrom", velocities)
        object.__setattr__(self, "heat_capacities_ev_per_k", cv)
        object.__setattr__(self, "gamma_thz_hwhm", gamma)
        object.__setattr__(self, "population_lifetime_ps", lifetime)
        object.__setattr__(self, "mean_free_path_angstrom", mfp)
        object.__setattr__(self, "mode_kappa_w_mk", mode)
        object.__setattr__(self, "band_index_kappa_w_mk", band_index)
        object.__setattr__(self, "kappa_w_mk", kappa)
        object.__setattr__(self, "qpoint_weights", weights)
        object.__setattr__(self, "primitive_volume_angstrom3", volume)
        object.__setattr__(self, "cutoff_frequency_thz", cutoff)
        object.__setattr__(self, "included_mask", included)
        object.__setattr__(self, "metadata", metadata)

    @property
    def kappa_voigt_w_mk(self) -> Array:
        result = np.stack(
            [self.kappa_w_mk[:, first, second] for first, second in VOIGT_INDEX_PAIRS],
            axis=1,
        )
        result.setflags(write=False)
        return result


def assemble_rta_thermal_conductivity(
    spectrum: BulkPhononSpectrum,
    velocities: BulkGroupVelocities,
    heat_capacities: BulkModeHeatCapacities,
    relaxation_times: BulkRelaxationTimes,
    *,
    primitive_volume_angstrom3: float,
    qpoint_weights: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> RTAThermalConductivity:
    """Assemble strict intrinsic three-phonon SMRTA conductivity."""

    if not isinstance(spectrum, BulkPhononSpectrum):
        raise TypeError("spectrum must be a BulkPhononSpectrum.")
    if not isinstance(velocities, BulkGroupVelocities):
        raise TypeError("velocities must be a BulkGroupVelocities result.")
    if not isinstance(heat_capacities, BulkModeHeatCapacities):
        raise TypeError("heat_capacities must be a BulkModeHeatCapacities result.")
    if not isinstance(relaxation_times, BulkRelaxationTimes):
        raise TypeError("relaxation_times must be a BulkRelaxationTimes result.")
    frequencies = spectrum.frequencies
    qpoint_difference = velocities.qpoints - spectrum.qpoints
    qpoint_difference -= np.rint(qpoint_difference)
    if not np.allclose(qpoint_difference, 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("Velocity and spectrum q-points do not match in order.")
    for name, values in (
        ("velocities", velocities.frequencies),
        ("heat_capacities", heat_capacities.frequencies_thz),
        ("relaxation_times", relaxation_times.frequencies_thz),
    ):
        if not np.allclose(values, frequencies, rtol=1.0e-12, atol=1.0e-12):
            raise ValueError(f"{name} frequencies do not match the spectrum.")
    if not np.array_equal(heat_capacities.temperatures, relaxation_times.temperatures):
        raise ValueError("Heat-capacity and linewidth temperatures must match.")
    if not np.isclose(
        velocities.frequency_factor_to_thz,
        spectrum.frequency_factor_to_thz,
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Velocity and spectrum frequency factors do not match.")
    if not (
        velocities.cutoff_frequency_thz
        == heat_capacities.cutoff_frequency_thz
        == relaxation_times.cutoff_frequency_thz
    ):
        raise ValueError("Velocity, heat-capacity, and lifetime cutoffs must match.")
    if not np.array_equal(
        heat_capacities.included_mask,
        relaxation_times.included_mask,
    ):
        raise ValueError("Heat-capacity and lifetime inclusion masks do not match.")
    stability_tolerance = max(
        1.0e-8,
        relaxation_times.cutoff_frequency_thz,
    )
    if np.any(frequencies < -stability_tolerance):
        raise ValueError("RTA conductivity rejects dynamically unstable FC2 input.")
    unresolved = relaxation_times.unresolved_included_mask
    if np.any(unresolved):
        count = int(np.count_nonzero(unresolved))
        raise ValueError(
            f"RTA conductivity has {count} included modes with unresolved "
            "nonpositive linewidth; refine the mesh/smearing or add an explicit "
            "scattering model."
        )
    if qpoint_weights is None:
        weights = np.full(spectrum.n_qpoints, 1.0 / spectrum.n_qpoints)
    else:
        raw_weights = np.asarray(qpoint_weights)
        if raw_weights.dtype.kind == "c":
            raise TypeError("qpoint_weights must be real-valued.")
        weights = np.array(qpoint_weights, dtype=np.float64, copy=True)
        if (
            weights.shape != (spectrum.n_qpoints,)
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or np.sum(weights) <= 0.0
        ):
            raise ValueError("qpoint_weights must be finite and nonnegative.")
        weights /= np.sum(weights)
    volume = float(primitive_volume_angstrom3)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("primitive_volume_angstrom3 must be positive.")
    mfp = compute_mean_free_paths(velocities, relaxation_times)
    outer_velocity = np.einsum(
        "qbi,qbj->qbij",
        velocities.values_thz_angstrom,
        velocities.values_thz_angstrom,
    )
    mode = (
        heat_capacities.values_ev_per_k[:, :, :, None, None]
        * outer_velocity[None, :, :, :, :]
        * relaxation_times.population_lifetime_ps[:, :, :, None, None]
        * weights[None, :, None, None, None]
        * PHONO3PY_4_4_RTA_CONVERSION
        / volume
    )
    mode *= relaxation_times.included_mask[None, :, :, None, None]
    band_index = np.sum(mode, axis=1)
    kappa = np.sum(band_index, axis=1)
    return RTAThermalConductivity(
        temperatures=heat_capacities.temperatures,
        qpoints=spectrum.qpoints,
        frequencies_thz=frequencies,
        group_velocities_thz_angstrom=velocities.values_thz_angstrom,
        heat_capacities_ev_per_k=heat_capacities.values_ev_per_k,
        gamma_thz_hwhm=relaxation_times.gamma_thz_hwhm,
        population_lifetime_ps=relaxation_times.population_lifetime_ps,
        mean_free_path_angstrom=mfp.magnitude_angstrom,
        mode_kappa_w_mk=mode,
        band_index_kappa_w_mk=band_index,
        kappa_w_mk=kappa,
        qpoint_weights=weights,
        primitive_volume_angstrom3=volume,
        cutoff_frequency_thz=relaxation_times.cutoff_frequency_thz,
        included_mask=relaxation_times.included_mask,
        metadata=metadata,
    )


def compute_rta_thermal_conductivity(
    problem: Any,
    *,
    temperatures: Sequence[float] | Array,
    linewidths: (
        BulkThreePhononLinewidths | BulkThreePhononRateSpectrum | None
    ) = None,
    gamma_floor_thz: float = 0.0,
) -> RTAThermalConductivity:
    """Run or reuse full-mesh linewidths and return intrinsic SMRTA kappa."""

    wrapper = None
    if not isinstance(problem, BulkThreePhononProblem):
        from negfpy.bulk.problem import BulkTransportProblem

        if isinstance(problem, BulkTransportProblem):
            wrapper = problem
            wrapper.verify_source_and_object_binding()
            problem = wrapper.three_phonon
        else:
            raise TypeError(
                "problem must be a BulkThreePhononProblem or "
                "BulkTransportProblem."
            )
    temperature_array = _temperatures(temperatures)
    if linewidths is None:
        linewidths = compute_three_phonon_linewidths(
            problem,
            temperatures=temperature_array,
        )
    elif not isinstance(
        linewidths,
        (BulkThreePhononLinewidths, BulkThreePhononRateSpectrum),
    ):
        raise TypeError(
            "linewidths must be a BulkThreePhononLinewidths or "
            "BulkThreePhononRateSpectrum result."
        )
    if not np.array_equal(linewidths.temperatures, temperature_array):
        raise ValueError("Supplied linewidth temperatures do not match.")
    expected_grid = np.arange(problem.mesh.n_qpoints, dtype=np.int64)
    expected_bands = np.arange(3 * problem.fc2_mapping.n_primitive_atoms, dtype=np.int64)
    if not np.array_equal(linewidths.grid_points, expected_grid):
        raise ValueError("Total RTA conductivity requires the complete q mesh.")
    if not np.array_equal(linewidths.band_indices, expected_bands):
        raise ValueError("Total RTA conductivity requires every phonon band.")
    from ._bulk_identity import (
        bulk_three_phonon_physics_sha256,
        bulk_three_phonon_problem_sha256,
    )

    expected_problem_sha256 = bulk_three_phonon_problem_sha256(problem)
    expected_physics_sha256 = bulk_three_phonon_physics_sha256(problem)
    supplied_physics_sha256 = (
        linewidths.physics_input_sha256
        if isinstance(linewidths, BulkThreePhononRateSpectrum)
        else linewidths.metadata.get("physics_input_sha256")
    )
    if supplied_physics_sha256 != expected_physics_sha256:
        raise ValueError(
            "Supplied linewidths were not computed from the same FC2/FC3, "
            "mapping, mesh, smearing, cutoff, and physical conventions."
        )
    expected_qpoints = np.asarray(
        problem.mesh.addresses,
        dtype=np.float64,
    ) / np.asarray(problem.mesh.shape, dtype=np.float64)[None, :]
    expected_spectrum = compute_bulk_phonons(
        problem.fc2,
        problem.fc2_mapping,
        expected_qpoints,
        frequency_factor_to_thz=problem.frequency_factor_to_thz,
        hermitianize=True,
        sampling_kind="uniform_full_mesh",
        mesh_shape=problem.mesh.shape,
    )
    if isinstance(linewidths, BulkThreePhononRateSpectrum):
        if not np.isclose(
            linewidths.cutoff_frequency_thz,
            problem.cutoff_frequency_thz,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Rate-spectrum cutoff conflicts with the problem.")
        if not np.isclose(
            linewidths.frequency_factor_to_thz,
            problem.frequency_factor_to_thz,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Rate-spectrum frequency factor conflicts with the problem."
            )
        supplied_qpoints = linewidths.qpoints
        supplied_frequencies = linewidths.frequencies_thz
    else:
        supplied_qpoints = linewidths.interactions.phonons.qpoints
        supplied_frequencies = linewidths.interactions.phonons.frequencies
        supplied_factor = linewidths.interactions.phonons.frequency_factor_to_thz
        if not np.isclose(
            supplied_factor,
            problem.frequency_factor_to_thz,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "Linewidth-spectrum frequency factor conflicts with the problem."
            )
        if not np.isclose(
            float(linewidths.metadata["cutoff_frequency_thz"]),
            problem.cutoff_frequency_thz,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("Linewidth-spectrum cutoff conflicts with the problem.")
    qpoint_difference = supplied_qpoints - expected_qpoints
    qpoint_difference -= np.rint(qpoint_difference)
    if not np.allclose(
        qpoint_difference,
        0.0,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            "Linewidth q-points do not match the canonical problem mesh."
        )
    if not np.allclose(
        supplied_frequencies,
        expected_spectrum.frequencies,
        rtol=1.0e-12,
        atol=1.0e-10,
    ):
        raise ValueError("Linewidth frequencies do not match the problem FC2.")
    spectrum = expected_spectrum
    velocities = compute_bulk_group_velocities(
        problem.fc2,
        problem.fc2_mapping,
        spectrum.qpoints,
        frequency_factor_to_thz=problem.frequency_factor_to_thz,
        cutoff_frequency_thz=problem.cutoff_frequency_thz,
        degenerate_tolerance_thz=problem.degenerate_tolerance_thz,
        symmetrize=True,
    )
    heat_capacities = compute_mode_heat_capacities(
        spectrum.frequencies,
        temperature_array,
        cutoff_frequency_thz=problem.cutoff_frequency_thz,
    )
    relaxation = compute_three_phonon_lifetimes(
        linewidths,
        gamma_floor_thz=gamma_floor_thz,
    )
    from ._provenance import code_provenance

    code_identity = code_provenance()
    provenance_complete = bool(
        wrapper is not None
        and wrapper.fc2_audit.passed
        and wrapper.fc2_physical_audit.passed
        and wrapper.fc3_audit.passed
        and set(wrapper.source_sha256) == {"crystal", "fc2", "fc3"}
        and all(len(value) == 64 for value in wrapper.source_sha256.values())
    )
    code_input_eligible = bool(
        provenance_complete
        and code_identity.get("package_version_consistent") is True
        and code_identity.get("git_dirty") is False
        and code_identity.get("git_revision") not in {None, "unknown"}
    )
    metadata: dict[str, Any] = {
        "schema": "negfpy.rta_thermal_conductivity.v1",
        "mesh": tuple(int(value) for value in problem.mesh.shape),
        "sigma_thz": problem.sigma_thz,
        "cutoff_frequency_thz": problem.cutoff_frequency_thz,
        "degenerate_tolerance_thz": problem.degenerate_tolerance_thz,
        "frequency_factor_to_thz": problem.frequency_factor_to_thz,
        "linewidth_semantics": "cyclic_frequency_hwhm",
        "lifetime_conversion": "tau_ps=1/(4*pi*gamma_thz_hwhm)",
        "qpoint_normalization": "full_uniform_mesh_equal_weights",
        "length_unit": "angstrom",
        "gamma_floor_thz": float(gamma_floor_thz),
        "group_velocity_symmetrized": velocities.symmetrized,
        "group_velocity_symmetry_operation_count": (
            velocities.symmetry_operation_count
        ),
        "group_velocity_perturbation_direction": (
            velocities.perturbation_direction
        ),
        "bulk_three_phonon_problem_sha256": expected_problem_sha256,
        "bulk_three_phonon_physics_sha256": expected_physics_sha256,
        "provenance_complete": provenance_complete,
        "code_and_input_provenance_eligible": code_input_eligible,
        "release_certification_eligible": False,
        "release_certification_authority": (
            "strict_external_release_driver_only"
        ),
        "code_provenance": code_identity,
    }
    if wrapper is not None:
        metadata["input_format"] = getattr(wrapper, "input_format", None)
        metadata["source_paths"] = dict(getattr(wrapper, "source_paths", {}))
        metadata["source_sha256"] = dict(getattr(wrapper, "source_sha256", {}))
    return assemble_rta_thermal_conductivity(
        spectrum,
        velocities,
        heat_capacities,
        relaxation,
        primitive_volume_angstrom3=abs(
            float(np.linalg.det(problem.fc2_mapping.primitive.lattice))
        ),
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True, eq=False)
class CumulativeThermalConductivity:
    """Cumulative conductivity tensor versus frequency or MFP threshold."""

    coordinate: Literal["frequency_thz", "mean_free_path_angstrom"]
    thresholds: Array
    temperatures: Array
    cumulative_kappa_w_mk: Array

    def __post_init__(self) -> None:
        if self.coordinate not in {"frequency_thz", "mean_free_path_angstrom"}:
            raise ValueError("Unsupported cumulative coordinate.")
        thresholds = _readonly(self.thresholds, dtype=np.float64)
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        cumulative = _readonly(self.cumulative_kappa_w_mk, dtype=np.float64)
        if (
            thresholds.ndim != 1
            or thresholds.size < 1
            or not np.all(np.isfinite(thresholds))
            or np.any(thresholds < 0.0)
            or np.any(np.diff(thresholds) < 0.0)
        ):
            raise ValueError(
                "thresholds must be a finite, nonnegative, ordered vector."
            )
        if (
            temperatures.ndim != 1
            or temperatures.size < 1
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise ValueError(
                "temperatures must be a nonempty finite nonnegative vector."
            )
        if cumulative.shape != (temperatures.size, thresholds.size, 3, 3):
            raise ValueError("Cumulative conductivity has an inconsistent shape.")
        if not np.all(np.isfinite(cumulative)):
            raise ValueError("Cumulative conductivity must be finite.")
        if not np.allclose(
            cumulative,
            cumulative.transpose(0, 1, 3, 2),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("Cumulative conductivity tensors must be symmetric.")
        scale = max(float(np.max(np.abs(cumulative))), 1.0)
        tolerance = 1.0e-12 * scale
        eigenvalues = np.linalg.eigvalsh(
            (cumulative + cumulative.transpose(0, 1, 3, 2)) / 2.0
        )
        if np.any(eigenvalues < -tolerance):
            raise ValueError(
                "Cumulative conductivity tensors must be positive semidefinite."
            )
        diagonals = np.diagonal(cumulative, axis1=-2, axis2=-1)
        if np.any(np.diff(diagonals, axis=1) < -tolerance):
            raise ValueError(
                "Cumulative conductivity diagonals must be nondecreasing."
            )
        object.__setattr__(self, "thresholds", thresholds)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "cumulative_kappa_w_mk", cumulative)


def compute_cumulative_thermal_conductivity(
    result: RTAThermalConductivity,
    *,
    coordinate: Literal["frequency_thz", "mean_free_path_angstrom"],
    thresholds: Any | None = None,
    n_thresholds: int = 201,
) -> CumulativeThermalConductivity:
    """Accumulate mode conductivity below frequency or MFP thresholds."""

    if not isinstance(result, RTAThermalConductivity):
        raise TypeError("result must be an RTAThermalConductivity.")
    if coordinate == "frequency_thz":
        coordinates = np.broadcast_to(
            result.frequencies_thz[None, :, :],
            result.population_lifetime_ps.shape,
        )
    elif coordinate == "mean_free_path_angstrom":
        coordinates = result.mean_free_path_angstrom
    else:
        raise ValueError("coordinate must be frequency_thz or mean_free_path_angstrom.")
    finite_coordinates = coordinates[np.isfinite(coordinates)]
    maximum = float(np.max(finite_coordinates)) if finite_coordinates.size else 0.0
    if thresholds is None:
        if isinstance(n_thresholds, (bool, np.bool_)) or not isinstance(n_thresholds, (int, np.integer)):
            raise TypeError("n_thresholds must be an integer.")
        if int(n_thresholds) < 2:
            raise ValueError("n_thresholds must be at least two.")
        threshold_array = np.linspace(0.0, maximum, int(n_thresholds))
    else:
        raw = np.asarray(thresholds)
        if raw.dtype.kind == "c":
            raise TypeError("thresholds must be real-valued.")
        threshold_array = np.array(thresholds, dtype=np.float64, copy=True)
        if (
            threshold_array.ndim != 1
            or threshold_array.size < 1
            or not np.all(np.isfinite(threshold_array))
            or np.any(threshold_array < 0.0)
            or np.any(np.diff(threshold_array) < 0.0)
        ):
            raise ValueError(
                "thresholds must be finite, nonnegative, and ordered."
            )
    cumulative = np.zeros(
        (result.temperatures.size, threshold_array.size, 3, 3),
        dtype=np.float64,
    )
    for temperature in range(result.temperatures.size):
        for position, threshold in enumerate(threshold_array):
            mask = coordinates[temperature] <= threshold
            cumulative[temperature, position] = np.sum(
                result.mode_kappa_w_mk[temperature][mask],
                axis=0,
            )
    return CumulativeThermalConductivity(
        coordinate=coordinate,
        thresholds=threshold_array,
        temperatures=result.temperatures,
        cumulative_kappa_w_mk=cumulative,
    )


__all__ = [
    "PHONO3PY_4_4_RTA_CONVERSION",
    "BulkMeanFreePaths",
    "BulkModeHeatCapacities",
    "BulkRelaxationTimes",
    "BulkThreePhononRateSpectrum",
    "CumulativeThermalConductivity",
    "RTAThermalConductivity",
    "assemble_rta_thermal_conductivity",
    "compute_cumulative_thermal_conductivity",
    "compute_mean_free_paths",
    "compute_mode_heat_capacities",
    "compute_rta_thermal_conductivity",
    "compute_three_phonon_lifetimes",
    "summarize_three_phonon_linewidths",
]
