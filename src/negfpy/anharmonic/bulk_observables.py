"""Audited bulk phonon band-structure and density-of-states observables."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .bulk_harmonic import (
    DEFAULT_FREQUENCY_FACTOR_TO_THZ,
    BulkPhononSpectrum,
    compute_bulk_phonons,
)
from .crystal import CrystalMapping
from .fc2 import FC2Data

Array = np.ndarray


def _readonly(values: Any, *, dtype: Any | None = None) -> Array:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class PhononBandPath:
    """Interpolated reciprocal-space path with explicit vertex positions."""

    qpoints: Array
    distances_inv_angstrom: Array
    vertex_indices: Array
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        qpoints = _readonly(self.qpoints, dtype=np.float64)
        distances = _readonly(self.distances_inv_angstrom, dtype=np.float64)
        vertices = _readonly(self.vertex_indices, dtype=np.int64)
        labels = tuple(str(value) for value in self.labels)
        if qpoints.ndim != 2 or qpoints.shape[1:] != (3,):
            raise ValueError("qpoints must have shape (n_qpoints, 3).")
        if distances.shape != (qpoints.shape[0],):
            raise ValueError("distances must contain one value per q-point.")
        if vertices.ndim != 1 or vertices.size < 2:
            raise ValueError("vertex_indices must contain at least two entries.")
        if len(labels) != vertices.size:
            raise ValueError("labels must contain one string per vertex.")
        if (
            vertices[0] != 0
            or vertices[-1] != qpoints.shape[0] - 1
            or np.any(np.diff(vertices) <= 0)
        ):
            raise ValueError("vertex_indices must be strictly ordered endpoints.")
        if (
            not np.all(np.isfinite(qpoints))
            or not np.all(np.isfinite(distances))
            or np.any(np.diff(distances) < 0.0)
        ):
            raise ValueError("Band-path coordinates must be finite and ordered.")
        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "distances_inv_angstrom", distances)
        object.__setattr__(self, "vertex_indices", vertices)
        object.__setattr__(self, "labels", labels)

    @property
    def vertex_distances_inv_angstrom(self) -> Array:
        result = self.distances_inv_angstrom[self.vertex_indices].copy()
        result.setflags(write=False)
        return result


@dataclass(frozen=True, slots=True, eq=False)
class PhononBandStructure:
    """Bulk phonon eigenproblem evaluated on a labelled band path."""

    path: PhononBandPath
    spectrum: BulkPhononSpectrum

    def __post_init__(self) -> None:
        if not isinstance(self.path, PhononBandPath):
            raise TypeError("path must be a PhononBandPath.")
        if not isinstance(self.spectrum, BulkPhononSpectrum):
            raise TypeError("spectrum must be a BulkPhononSpectrum.")
        if not np.array_equal(self.path.qpoints, self.spectrum.qpoints):
            raise ValueError("Band path and spectrum q-points must match exactly.")

    @property
    def frequencies_thz(self) -> Array:
        return self.spectrum.frequencies


def interpolate_band_path(
    vertices: Any,
    primitive_lattice_angstrom: Any,
    *,
    points_per_segment: int | Sequence[int] = 51,
    labels: Sequence[str] | None = None,
) -> PhononBandPath:
    """Interpolate fractional reciprocal vertices without duplicate joints."""

    raw_vertices = np.asarray(vertices)
    raw_lattice = np.asarray(primitive_lattice_angstrom)
    if raw_vertices.dtype.kind == "c" or raw_lattice.dtype.kind == "c":
        raise TypeError("Band vertices and lattice must be real-valued.")
    vertex_array = np.array(vertices, dtype=np.float64, order="C", copy=True)
    lattice = np.array(
        primitive_lattice_angstrom,
        dtype=np.float64,
        order="C",
        copy=True,
    )
    if (
        vertex_array.ndim != 2
        or vertex_array.shape[0] < 2
        or vertex_array.shape[1:] != (3,)
        or not np.all(np.isfinite(vertex_array))
    ):
        raise ValueError("vertices must have finite shape (n_vertices>=2, 3).")
    if (
        lattice.shape != (3, 3)
        or not np.all(np.isfinite(lattice))
        or np.linalg.matrix_rank(lattice) != 3
    ):
        raise ValueError("primitive_lattice_angstrom must be nonsingular (3, 3).")
    n_segments = vertex_array.shape[0] - 1
    if isinstance(points_per_segment, (int, np.integer)) and not isinstance(
        points_per_segment,
        (bool, np.bool_),
    ):
        counts = np.full(n_segments, int(points_per_segment), dtype=np.int64)
    else:
        raw_counts = np.asarray(points_per_segment)
        if raw_counts.dtype.kind not in "iu" or raw_counts.ndim != 1:
            raise TypeError("points_per_segment must contain integers.")
        counts = np.array(raw_counts, dtype=np.int64, copy=True)
        if counts.shape != (n_segments,):
            raise ValueError("points_per_segment must have one value per segment.")
    if np.any(counts < 2):
        raise ValueError("Every band-path segment requires at least two points.")
    if labels is None:
        label_tuple = tuple("" for _ in range(vertex_array.shape[0]))
    else:
        label_tuple = tuple(str(value) for value in labels)
        if len(label_tuple) != vertex_array.shape[0]:
            raise ValueError("labels must contain one entry per vertex.")

    qpoint_parts: list[Array] = []
    vertex_indices = [0]
    cursor = 0
    for segment, count in enumerate(counts):
        parameter = np.linspace(0.0, 1.0, int(count))
        part = (
            vertex_array[segment][None, :] * (1.0 - parameter[:, None])
            + vertex_array[segment + 1][None, :] * parameter[:, None]
        )
        if segment:
            part = part[1:]
        qpoint_parts.append(part)
        cursor += int(count) - (1 if segment else 0)
        vertex_indices.append(cursor - 1)
    qpoints = np.concatenate(qpoint_parts, axis=0)
    reciprocal_lattice = np.linalg.inv(lattice).T
    cartesian_steps = np.diff(qpoints, axis=0) @ reciprocal_lattice
    distances = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(cartesian_steps, axis=1)))
    )
    return PhononBandPath(
        qpoints=qpoints,
        distances_inv_angstrom=distances,
        vertex_indices=vertex_indices,
        labels=label_tuple,
    )


def compute_phonon_band_structure(
    fc2: FC2Data,
    mapping: CrystalMapping,
    vertices: Any,
    *,
    points_per_segment: int | Sequence[int] = 51,
    labels: Sequence[str] | None = None,
    frequency_factor_to_thz: float = DEFAULT_FREQUENCY_FACTOR_TO_THZ,
) -> PhononBandStructure:
    """Compute a labelled non-NAC bulk phonon band structure from FC2."""

    path = interpolate_band_path(
        vertices,
        mapping.primitive.lattice,
        points_per_segment=points_per_segment,
        labels=labels,
    )
    spectrum = compute_bulk_phonons(
        fc2,
        mapping,
        path.qpoints,
        frequency_factor_to_thz=frequency_factor_to_thz,
        hermitianize=True,
        sampling_kind="band_path",
    )
    return PhononBandStructure(path=path, spectrum=spectrum)


@dataclass(frozen=True, slots=True, eq=False)
class PhononDOS:
    """Gaussian bulk DOS normalized to states per primitive cell per THz."""

    frequency_points_thz: Array
    total_states_per_thz_cell: Array
    projected_states_per_thz_cell: Array
    sigma_thz: float
    qpoint_weights: Array
    integral_states_per_cell: float
    expected_states_per_cell: int
    stability_tolerance_thz: float
    sampling_description: str
    normalization_absolute_error: float
    normalization_tolerance: float
    normalization_certified: bool

    def __post_init__(self) -> None:
        points = _readonly(self.frequency_points_thz, dtype=np.float64)
        total = _readonly(self.total_states_per_thz_cell, dtype=np.float64)
        projected = _readonly(
            self.projected_states_per_thz_cell,
            dtype=np.float64,
        )
        weights = _readonly(self.qpoint_weights, dtype=np.float64)
        if points.ndim != 1 or points.size < 2 or np.any(np.diff(points) <= 0.0):
            raise ValueError("frequency_points_thz must be strictly increasing.")
        if total.shape != points.shape:
            raise ValueError("total DOS must match the frequency grid.")
        if projected.ndim != 3 or projected.shape[2] != points.size:
            raise ValueError("projected DOS must have shape (n_atom, 3, n_freq).")
        if weights.ndim != 1 or weights.size < 1:
            raise ValueError("qpoint_weights must be one-dimensional.")
        if not all(
            np.all(np.isfinite(values))
            for values in (points, total, projected, weights)
        ):
            raise ValueError("DOS arrays must be finite.")
        if np.any(total < 0.0) or np.any(projected < 0.0):
            raise ValueError("DOS values must be nonnegative.")
        if np.any(weights < 0.0) or not np.isclose(
            np.sum(weights), 1.0, rtol=0.0, atol=1.0e-14
        ):
            raise ValueError("qpoint_weights must be nonnegative and sum to one.")
        if not np.allclose(
            np.sum(projected, axis=(0, 1)),
            total,
            rtol=1.0e-12,
            atol=1.0e-14,
        ):
            raise ValueError("Projected DOS components must sum to total DOS.")
        sigma = float(self.sigma_thz)
        integral = float(self.integral_states_per_cell)
        tolerance = float(self.stability_tolerance_thz)
        expected = int(self.expected_states_per_cell)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_thz must be positive.")
        if not np.isfinite(integral) or integral < 0.0:
            raise ValueError("integral_states_per_cell must be nonnegative.")
        if expected != 3 * projected.shape[0]:
            raise ValueError("expected state count must equal 3*n_primitive_atoms.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("stability_tolerance_thz must be nonnegative.")
        description = str(self.sampling_description).strip()
        if not description:
            raise ValueError("sampling_description must be nonempty.")
        normalization_error = float(self.normalization_absolute_error)
        normalization_tolerance = float(self.normalization_tolerance)
        if not np.isfinite(normalization_error) or normalization_error < 0.0:
            raise ValueError("normalization_absolute_error must be nonnegative.")
        if (
            not np.isfinite(normalization_tolerance)
            or normalization_tolerance < 0.0
        ):
            raise ValueError("normalization_tolerance must be nonnegative.")
        expected_certified = normalization_error <= normalization_tolerance
        if bool(self.normalization_certified) != expected_certified:
            raise ValueError("normalization_certified conflicts with the DOS integral.")
        object.__setattr__(self, "frequency_points_thz", points)
        object.__setattr__(self, "total_states_per_thz_cell", total)
        object.__setattr__(self, "projected_states_per_thz_cell", projected)
        object.__setattr__(self, "qpoint_weights", weights)
        object.__setattr__(self, "sigma_thz", sigma)
        object.__setattr__(self, "integral_states_per_cell", integral)
        object.__setattr__(self, "expected_states_per_cell", expected)
        object.__setattr__(self, "stability_tolerance_thz", tolerance)
        object.__setattr__(self, "sampling_description", description)
        object.__setattr__(
            self,
            "normalization_absolute_error",
            normalization_error,
        )
        object.__setattr__(self, "normalization_tolerance", normalization_tolerance)
        object.__setattr__(self, "normalization_certified", expected_certified)


def compute_phonon_dos(
    spectrum: BulkPhononSpectrum,
    *,
    frequency_points_thz: Any | None = None,
    sigma_thz: float = 0.1,
    qpoint_weights: Any | None = None,
    n_frequency_points: int = 2001,
    stability_tolerance_thz: float = 1.0e-4,
    sampling_description: str | None = None,
    normalization_tolerance: float = 5.0e-4,
    allow_truncated_window: bool = False,
) -> PhononDOS:
    """Compute total and atom/Cartesian-projected Gaussian phonon DOS.

    The default grid extends five Gaussian widths below the smallest stable
    frequency and above the maximum.  This retains the normalization of
    acoustic modes centered at zero instead of silently discarding half of
    their Gaussian weight.
    """

    if not isinstance(spectrum, BulkPhononSpectrum):
        raise TypeError("spectrum must be a BulkPhononSpectrum.")
    if spectrum.sampling_kind == "band_path":
        raise ValueError(
            "A one-dimensional band path cannot be used as a bulk DOS mesh."
        )
    if spectrum.sampling_kind == "uniform_full_mesh":
        description = (
            "uniform_full_mesh:"
            + "x".join(str(value) for value in spectrum.mesh_shape or ())
        )
        if sampling_description is not None:
            description = str(sampling_description).strip()
    else:
        if qpoint_weights is None or sampling_description is None:
            raise ValueError(
                "An arbitrary spectrum requires explicit qpoint_weights and a "
                "nonempty sampling_description for bulk DOS integration."
            )
        description = str(sampling_description).strip()
    if not description:
        raise ValueError("sampling_description must be nonempty.")
    normalization_tolerance = float(normalization_tolerance)
    if not np.isfinite(normalization_tolerance) or normalization_tolerance < 0.0:
        raise ValueError("normalization_tolerance must be nonnegative.")
    if not isinstance(allow_truncated_window, (bool, np.bool_)):
        raise TypeError("allow_truncated_window must be a boolean.")
    sigma = float(sigma_thz)
    tolerance = float(stability_tolerance_thz)
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma_thz must be positive.")
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("stability_tolerance_thz must be nonnegative.")
    frequencies = np.array(spectrum.frequencies, dtype=np.float64, copy=True)
    if np.any(frequencies < -tolerance):
        minimum = float(np.min(frequencies))
        raise ValueError(
            "RTA-release DOS rejects dynamically unstable spectra; "
            f"minimum frequency is {minimum:.16g} THz."
        )
    frequencies[(frequencies < 0.0) & (frequencies >= -tolerance)] = 0.0
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
    if frequency_points_thz is None:
        if isinstance(n_frequency_points, (bool, np.bool_)) or not isinstance(
            n_frequency_points,
            (int, np.integer),
        ):
            raise TypeError("n_frequency_points must be an integer.")
        if int(n_frequency_points) < 2:
            raise ValueError("n_frequency_points must be at least two.")
        lower = float(np.min(frequencies) - 5.0 * sigma)
        upper = float(np.max(frequencies) + 5.0 * sigma)
        points = np.linspace(lower, upper, int(n_frequency_points))
    else:
        raw_points = np.asarray(frequency_points_thz)
        if raw_points.dtype.kind == "c":
            raise TypeError("frequency_points_thz must be real-valued.")
        points = np.array(frequency_points_thz, dtype=np.float64, copy=True)
        if (
            points.ndim != 1
            or points.size < 2
            or not np.all(np.isfinite(points))
            or np.any(np.diff(points) <= 0.0)
        ):
            raise ValueError("frequency_points_thz must be strictly increasing.")

    n_atoms = spectrum.n_bands // 3
    projected = np.zeros((n_atoms, 3, points.size), dtype=np.float64)
    normalization = 1.0 / (np.sqrt(2.0 * np.pi) * sigma)
    for q_position in range(spectrum.n_qpoints):
        eigenvectors = spectrum.eigenvectors[q_position].reshape(
            n_atoms,
            3,
            spectrum.n_bands,
        )
        projections = np.abs(eigenvectors) ** 2
        for band in range(spectrum.n_bands):
            kernel = normalization * np.exp(
                -0.5
                * ((points - frequencies[q_position, band]) / sigma) ** 2
            )
            projected += (
                weights[q_position]
                * projections[:, :, band, None]
                * kernel[None, None, :]
            )
    total = np.sum(projected, axis=(0, 1))
    integral = float(np.trapezoid(total, points))
    normalization_error = abs(integral - spectrum.n_bands)
    normalization_certified = normalization_error <= normalization_tolerance
    if not normalization_certified and not allow_truncated_window:
        raise ValueError(
            "DOS frequency window fails state-count normalization: "
            f"integral={integral:.16g}, expected={spectrum.n_bands}, "
            f"absolute error={normalization_error:.6e}. Extend the window or "
            "set allow_truncated_window=True for an explicitly uncertified DOS."
        )
    return PhononDOS(
        frequency_points_thz=points,
        total_states_per_thz_cell=total,
        projected_states_per_thz_cell=projected,
        sigma_thz=sigma,
        qpoint_weights=weights,
        integral_states_per_cell=integral,
        expected_states_per_cell=spectrum.n_bands,
        stability_tolerance_thz=tolerance,
        sampling_description=description,
        normalization_absolute_error=normalization_error,
        normalization_tolerance=normalization_tolerance,
        normalization_certified=normalization_certified,
    )


__all__ = [
    "PhononBandPath",
    "PhononBandStructure",
    "PhononDOS",
    "compute_phonon_band_structure",
    "compute_phonon_dos",
    "interpolate_band_path",
]
