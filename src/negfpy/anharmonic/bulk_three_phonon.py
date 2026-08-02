"""Deterministic serial bulk three-phonon interaction and linewidth oracle.

The implementation mirrors the explicitly pinned phono3py 4.4.0 numerical
convention but does not call phono3py computational kernels.  It uses the full
unreduced mesh, complete target-q cases, canonical summation order, Gaussian
energy conservation, and cyclic-THz HWHM output.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .bulk_harmonic import (
    PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    BulkPhononSpectrum,
    compute_bulk_phonons,
)
from .crystal import (
    CrystalMapping,
    compare_primitive_mappings,
    validate_crystal_mapping,
)
from .fc2 import FC2Data
from .fc3 import (
    ANGSTROM_M,
    ATOMIC_MASS_UNIT_KG,
    ELECTRON_VOLT_J,
    fc3_force_constant_to_si_factor,
    mass_to_si_factor,
)
from .momentum import (
    PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR,
    PHONO3PY_4_4_NIGGLI_TOLERANCE,
    UniformBZMesh,
    minimum_bz_reciprocal_vector,
    niggli_reduced_reciprocal_lattice,
    shortest_bz_address_images,
)
from .translational_fc3 import TranslationalFC3

Array = np.ndarray
Metadata = Mapping[str, Any]

# These are the constants embedded in phonopy/phono3py 4.4.0 rather than
# modern CODATA values. They are pinned because the oracle is a reproduction
# target, not a general constants service.
PHONO3PY_4_4_HBAR_EV_S = 6.582118985531608e-16
PHONO3PY_4_4_EV_J = 1.60217733e-19
PHONO3PY_4_4_AMU_KG = 1.6605402e-27
PHONO3PY_4_4_ANGSTROM_M = 1.0e-10
PHONO3PY_4_4_THZ_HZ = 1.0e12
PHONO3PY_4_4_THZ_TO_EV = 0.00413566733
PHONO3PY_4_4_KB_EV_K = 8.617338256808316e-05

PHONO3PY_INTERACTION_FACTORIAL = 36.0
PHONO3PY_INTERACTION_MODE_FACTOR = 8.0
PHONO3PY_ISE_NUMERATOR = 18.0 * np.pi


def _readonly(values: Any, *, dtype: Any | None = None) -> Array:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    return value


def _freeze_mapping(values: Metadata | None) -> Metadata:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("metadata must be a mapping.")
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in values.items()}
    )


def phono3py_interaction_unit_factor(n_qpoints: int) -> float:
    """Return phono3py 4.4's fixed conversion of ``|fc3_normal|^2``.

    The factor includes ``1 / 3!**2 = 1 / 36``, ``1 / 2**3 = 1 / 8``,
    and the full-mesh normalization ``1 / N_q``.
    """

    if isinstance(n_qpoints, (bool, np.bool_)) or not isinstance(
        n_qpoints,
        (int, np.integer),
    ):
        raise TypeError("n_qpoints must be a positive integer.")
    n_qpoints = int(n_qpoints)
    if n_qpoints <= 0:
        raise ValueError("n_qpoints must be a positive integer.")
    return float(
        (PHONO3PY_4_4_HBAR_EV_S * PHONO3PY_4_4_EV_J) ** 3
        / PHONO3PY_INTERACTION_FACTORIAL
        / PHONO3PY_INTERACTION_MODE_FACTOR
        * PHONO3PY_4_4_EV_J**2
        / PHONO3PY_4_4_ANGSTROM_M**6
        / (2.0 * np.pi * PHONO3PY_4_4_THZ_HZ) ** 3
        / PHONO3PY_4_4_AMU_KG**3
        / n_qpoints
        / PHONO3PY_4_4_EV_J**2
    )


def phono3py_imag_self_energy_unit_factor() -> float:
    """Return phono3py 4.4's conversion from accumulated strength to THz."""

    return float(
        PHONO3PY_ISE_NUMERATOR
        / (PHONO3PY_4_4_HBAR_EV_S * PHONO3PY_4_4_EV_J) ** 2
        / (2.0 * np.pi * PHONO3PY_4_4_THZ_HZ) ** 2
        * PHONO3PY_4_4_EV_J**2
    )


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite positive scalar.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite positive scalar.") from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a finite positive scalar.")
    return number


def _nonnegative_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite nonnegative scalar.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a finite nonnegative scalar.") from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative scalar.")
    return number


@dataclass(frozen=True, slots=True, eq=False)
class BulkThreePhononProblem:
    """Audited bulk FC2/FC3 problem on a full uniform mesh."""

    fc2: FC2Data
    fc2_mapping: CrystalMapping
    fc3: TranslationalFC3
    fc3_mapping: CrystalMapping
    mesh: UniformBZMesh
    sigma_thz: float
    bz_reduced_reciprocal_lattice: Array | None = None
    cutoff_frequency_thz: float = 1.0e-4
    frequency_factor_to_thz: float = PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ
    degenerate_tolerance_thz: float = 1.0e-4
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.fc2, FC2Data):
            raise TypeError("fc2 must be an FC2Data instance.")
        if not isinstance(self.fc2_mapping, CrystalMapping):
            raise TypeError("fc2_mapping must be a CrystalMapping.")
        if not isinstance(self.fc3, TranslationalFC3):
            raise TypeError("fc3 must be a TranslationalFC3.")
        if not isinstance(self.fc3_mapping, CrystalMapping):
            raise TypeError("fc3_mapping must be a CrystalMapping.")
        if not isinstance(self.mesh, UniformBZMesh):
            raise TypeError("mesh must be a UniformBZMesh.")
        if self.fc2_mapping.purpose != "fc2":
            raise ValueError("fc2_mapping purpose must be 'fc2'.")
        if self.fc3_mapping.purpose != "fc3":
            raise ValueError("fc3_mapping purpose must be 'fc3'.")
        for mapping in (self.fc2_mapping, self.fc3_mapping):
            report = validate_crystal_mapping(mapping)
            if not report.passed:
                raise ValueError(
                    "A crystal mapping failed validation: "
                    + " ".join(report.errors)
                )
        comparison = compare_primitive_mappings(
            self.fc2_mapping,
            self.fc3_mapping,
        )
        if not comparison.passed:
            raise ValueError(
                "FC2 and FC3 mappings do not share the same ordered primitive "
                "cell: "
                + " ".join(comparison.errors)
            )
        if self.fc3.n_primitive_atoms != self.fc3_mapping.n_primitive_atoms:
            raise ValueError(
                "TranslationalFC3 primitive size conflicts with fc3_mapping."
            )
        if self.fc3.n_supercell_atoms != self.fc3_mapping.n_supercell_atoms:
            raise ValueError(
                "TranslationalFC3 supercell size conflicts with fc3_mapping."
            )
        if not np.isclose(
            fc3_force_constant_to_si_factor(self.fc3.units),
            ELECTRON_VOLT_J / ANGSTROM_M**3,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "The phono3py bulk oracle requires FC3 in eV/angstrom^3."
            )
        if not np.isclose(
            mass_to_si_factor(self.fc3.mass_units),
            ATOMIC_MASS_UNIT_KG,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("The phono3py bulk oracle requires masses in amu.")

        sigma = _positive_finite(self.sigma_thz, name="sigma_thz")
        cutoff = _nonnegative_finite(
            self.cutoff_frequency_thz,
            name="cutoff_frequency_thz",
        )
        factor = _positive_finite(
            self.frequency_factor_to_thz,
            name="frequency_factor_to_thz",
        )
        degeneracy = _positive_finite(
            self.degenerate_tolerance_thz,
            name="degenerate_tolerance_thz",
        )
        reduced_reciprocal = self.bz_reduced_reciprocal_lattice
        if reduced_reciprocal is not None:
            raw_reduced = np.asarray(reduced_reciprocal)
            if raw_reduced.dtype.kind == "c":
                raise TypeError(
                    "bz_reduced_reciprocal_lattice must be real-valued."
                )
            reduced_reciprocal = _readonly(
                raw_reduced,
                dtype=np.float64,
            )
            if (
                reduced_reciprocal.shape != (3, 3)
                or not np.all(np.isfinite(reduced_reciprocal))
                or np.linalg.matrix_rank(reduced_reciprocal) != 3
            ):
                raise ValueError(
                    "bz_reduced_reciprocal_lattice must be a finite "
                    "nonsingular matrix with shape (3, 3)."
                )
        object.__setattr__(self, "sigma_thz", sigma)
        object.__setattr__(self, "cutoff_frequency_thz", cutoff)
        object.__setattr__(self, "frequency_factor_to_thz", factor)
        object.__setattr__(self, "degenerate_tolerance_thz", degeneracy)
        object.__setattr__(
            self,
            "bz_reduced_reciprocal_lattice",
            reduced_reciprocal,
        )
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _selector(
    values: Sequence[int] | Array | None,
    *,
    upper: int,
    name: str,
) -> Array:
    if values is None:
        result = np.arange(upper, dtype=np.int64)
    else:
        raw = np.asarray(values)
        if raw.ndim != 1 or raw.dtype.kind not in "iu":
            raise TypeError(f"{name} must be a one-dimensional integer array.")
        if raw.dtype.kind == "b":
            raise TypeError(f"{name} must not contain booleans.")
        result = np.array(raw, dtype=np.int64, order="C", copy=True)
        if result.size == 0:
            raise ValueError(f"{name} cannot be empty.")
        if np.any((result < 0) | (result >= upper)):
            raise IndexError(f"{name} contains an index outside [0, {upper}).")
        if np.unique(result).size != result.size:
            raise ValueError(f"{name} cannot contain duplicate indices.")
        result.sort()
    result.setflags(write=False)
    return result


def _mesh_qpoints(mesh: UniformBZMesh) -> Array:
    return np.array(
        [
            [
                float(component) / period
                for component, period in zip(
                    address,
                    mesh.shape,
                    strict=True,
                )
            ]
            for address in mesh.addresses
        ],
        dtype=np.float64,
    )


def _estimate_interaction_result_bytes(
    n_targets: int,
    n_qpoints: int,
    n_target_bands: int,
    n_bands: int,
) -> int:
    fields = n_targets * n_qpoints * n_target_bands * n_bands * n_bands
    # complex matrix element + real strength, plus routing arrays.
    return int(fields * 24 + n_targets * n_qpoints * (8 + 8 + 3 * 8 + 1))


@dataclass(frozen=True, slots=True, eq=False)
class BulkThreePhononInteractions:
    """Immutable target-q interaction fields in canonical full-mesh order."""

    grid_points: Array
    band_indices: Array
    q1_indices: Array
    q2_indices: Array
    reciprocal_vectors: Array
    normal_mask: Array
    matrix_elements: Array
    interaction_strengths: Array
    phonons: BulkPhononSpectrum
    metadata: Metadata

    def __post_init__(self) -> None:
        grid_points = _readonly(self.grid_points, dtype=np.int64)
        band_indices = _readonly(self.band_indices, dtype=np.int64)
        q1 = _readonly(self.q1_indices, dtype=np.int64)
        q2 = _readonly(self.q2_indices, dtype=np.int64)
        reciprocal = _readonly(self.reciprocal_vectors, dtype=np.int64)
        normal = _readonly(self.normal_mask, dtype=np.bool_)
        matrix_elements = _readonly(
            self.matrix_elements,
            dtype=np.complex128,
        )
        strengths = _readonly(self.interaction_strengths, dtype=np.float64)

        n_targets = grid_points.size
        if grid_points.ndim != 1 or band_indices.ndim != 1:
            raise ValueError("grid_points and band_indices must be 1D.")
        if q1.ndim != 2 or q1.shape[0] != n_targets:
            raise ValueError("q1_indices must have shape (n_targets, n_qpoints).")
        if q2.shape != q1.shape or normal.shape != q1.shape:
            raise ValueError("q2_indices and normal_mask must match q1_indices.")
        if reciprocal.shape != q1.shape + (3,):
            raise ValueError(
                "reciprocal_vectors must have shape "
                "(n_targets, n_qpoints, 3)."
            )
        expected = (
            n_targets,
            q1.shape[1],
            band_indices.size,
            self.phonons.n_bands,
            self.phonons.n_bands,
        )
        if matrix_elements.shape != expected or strengths.shape != expected:
            raise ValueError(
                "Interaction fields have an inconsistent target/q/band shape."
            )
        if not (
            np.all(np.isfinite(matrix_elements))
            and np.all(np.isfinite(strengths))
            and np.all(strengths >= 0.0)
        ):
            raise ValueError("Interaction fields must be finite and nonnegative.")

        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "band_indices", band_indices)
        object.__setattr__(self, "q1_indices", q1)
        object.__setattr__(self, "q2_indices", q2)
        object.__setattr__(self, "reciprocal_vectors", reciprocal)
        object.__setattr__(self, "normal_mask", normal)
        object.__setattr__(self, "matrix_elements", matrix_elements)
        object.__setattr__(self, "interaction_strengths", strengths)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))


def _mode_space_matrix_elements(
    reciprocal_fc3: Array,
    frequencies: tuple[Array, Array, Array],
    eigenvectors: tuple[Array, Array, Array],
    masses: Array,
    selected_bands: Array,
    cutoff: float,
) -> Array:
    n_primitive = masses.size
    n_bands = 3 * n_primitive
    scaled_vectors = []
    for vectors in eigenvectors:
        reshaped = vectors.reshape(n_primitive, 3, n_bands)
        scaled_vectors.append(reshaped / np.sqrt(masses)[:, None, None])
    contracted = np.einsum(
        "ial,jbm,kcn,ijkabc->lmn",
        scaled_vectors[0][:, :, selected_bands],
        scaled_vectors[1],
        scaled_vectors[2],
        reciprocal_fc3,
        optimize=True,
    )

    f0 = frequencies[0][selected_bands]
    f1 = frequencies[1]
    f2 = frequencies[2]
    valid = (
        (f0[:, None, None] > cutoff)
        & (f1[None, :, None] > cutoff)
        & (f2[None, None, :] > cutoff)
    )
    denominator = np.sqrt(
        np.where(
            valid,
            f0[:, None, None] * f1[None, :, None] * f2[None, None, :],
            1.0,
        )
    )
    return np.where(valid, contracted / denominator, 0.0j)


def compute_three_phonon_interactions(
    problem: BulkThreePhononProblem,
    *,
    grid_points: Sequence[int] | Array | None = None,
    band_indices: Sequence[int] | Array | None = None,
) -> BulkThreePhononInteractions:
    """Compute independent target-q mode-space matrix elements serially."""

    if not isinstance(problem, BulkThreePhononProblem):
        raise TypeError("problem must be a BulkThreePhononProblem.")
    targets = _selector(
        grid_points,
        upper=problem.mesh.n_qpoints,
        name="grid_points",
    )
    n_bands = 3 * problem.fc2_mapping.n_primitive_atoms
    bands = _selector(
        band_indices,
        upper=n_bands,
        name="band_indices",
    )
    estimate = _estimate_interaction_result_bytes(
        targets.size,
        problem.mesh.n_qpoints,
        bands.size,
        n_bands,
    )

    qpoints = _mesh_qpoints(problem.mesh)
    phonons = compute_bulk_phonons(
        problem.fc2,
        problem.fc2_mapping,
        qpoints,
        frequency_factor_to_thz=problem.frequency_factor_to_thz,
        hermitianize=True,
        sampling_kind="uniform_full_mesh",
        mesh_shape=problem.mesh.shape,
    )
    shape = (
        targets.size,
        problem.mesh.n_qpoints,
        bands.size,
        n_bands,
        n_bands,
    )
    matrix_elements = np.empty(shape, dtype=np.complex128)
    strengths = np.empty(shape, dtype=np.float64)
    q1_indices = np.empty(
        (targets.size, problem.mesh.n_qpoints),
        dtype=np.int64,
    )
    q2_indices = np.empty_like(q1_indices)
    reciprocal_vectors = np.empty(q1_indices.shape + (3,), dtype=np.int64)
    normal_mask = np.empty(q1_indices.shape, dtype=np.bool_)
    conversion = phono3py_interaction_unit_factor(problem.mesh.n_qpoints)
    masses = problem.fc3_mapping.primitive.masses
    primitive_lattice = problem.fc3_mapping.primitive.lattice
    if problem.bz_reduced_reciprocal_lattice is None:
        reduced_reciprocal_lattice = niggli_reduced_reciprocal_lattice(
            primitive_lattice,
            tolerance=PHONO3PY_4_4_NIGGLI_TOLERANCE,
        )
        bz_reduction_source = "spglib.niggli_reduce"
        bz_reduction_verified = True
    else:
        reduced_reciprocal_lattice = (
            problem.bz_reduced_reciprocal_lattice
        )
        bz_reduction_source = (
            "problem_explicit_unimodular_equivalence_only"
        )
        bz_reduction_verified = False
    bz_address_images = shortest_bz_address_images(
        problem.mesh,
        primitive_lattice,
        reduced_reciprocal_lattice=reduced_reciprocal_lattice,
    )

    for target_position, target in enumerate(targets):
        triplets = problem.mesh.triplets_at_q(int(target))
        for route_position, triplet in enumerate(triplets):
            q0, q1, q2 = triplet.indices
            reciprocal_fc3 = problem.fc3.fourier(
                qpoints[[q0, q1, q2]]
            )
            elements = _mode_space_matrix_elements(
                reciprocal_fc3,
                (
                    phonons.frequencies[q0],
                    phonons.frequencies[q1],
                    phonons.frequencies[q2],
                ),
                (
                    phonons.eigenvectors[q0],
                    phonons.eigenvectors[q1],
                    phonons.eigenvectors[q2],
                ),
                masses,
                bands,
                problem.cutoff_frequency_thz,
            )
            matrix_elements[target_position, route_position] = elements
            strengths[target_position, route_position] = (
                np.abs(elements) ** 2 * conversion
            )
            q1_indices[target_position, route_position] = q1
            q2_indices[target_position, route_position] = q2
            reciprocal = minimum_bz_reciprocal_vector(
                triplet,
                bz_address_images,
                primitive_lattice,
                reduced_reciprocal_lattice=reduced_reciprocal_lattice,
            )
            reciprocal_vectors[target_position, route_position] = reciprocal
            normal_mask[target_position, route_position] = (
                reciprocal == (0, 0, 0)
            )

    from ._bulk_identity import (
        bulk_three_phonon_physics_sha256,
        bulk_three_phonon_problem_sha256,
    )

    metadata = {
        "schema": "negfpy.bulk_three_phonon_interactions.v1",
        "method": "deterministic_serial_full_mesh",
        "triplet_convention": "q0+q1+q2=G",
        "bz_surface_policy": (
            "minimum Cartesian |G| over phono3py-4.4 tolerance-retained "
            "BZ images of all three legs"
        ),
        "bz_reduction_source": bz_reduction_source,
        "bz_reduction_verified_niggli": bz_reduction_verified,
        "bz_niggli_tolerance": PHONO3PY_4_4_NIGGLI_TOLERANCE,
        "bz_grid_tolerance_factor": (
            PHONO3PY_4_4_BZ_GRID_TOLERANCE_FACTOR
        ),
        "bz_reduced_reciprocal_basis_convention": (
            "3x3 column vectors in Cartesian reciprocal coordinates"
        ),
        "bz_reduced_reciprocal_lattice": reduced_reciprocal_lattice,
        "phase_convention": problem.fc3.phase_convention,
        "make_r0_average": problem.fc3.make_r0_average,
        "mesh": problem.mesh.shape,
        "n_qpoints": problem.mesh.n_qpoints,
        "interaction_unit_factor": conversion,
        "factorial_divisor": PHONO3PY_INTERACTION_FACTORIAL,
        "mode_normalization_divisor": PHONO3PY_INTERACTION_MODE_FACTOR,
        "full_mesh_normalization": 1.0 / problem.mesh.n_qpoints,
        "frequency_units": "cyclic THz",
        "matrix_element_units": (
            "eV/angstrom^3/(amu^(3/2)*THz^(3/2))"
        ),
        "interaction_strength_units": "eV^2",
        "estimated_result_bytes": estimate,
        "problem_sha256": bulk_three_phonon_problem_sha256(problem),
        "physics_input_sha256": bulk_three_phonon_physics_sha256(problem),
        "problem_metadata": problem.metadata,
    }
    return BulkThreePhononInteractions(
        grid_points=targets,
        band_indices=bands,
        q1_indices=q1_indices,
        q2_indices=q2_indices,
        reciprocal_vectors=reciprocal_vectors,
        normal_mask=normal_mask,
        matrix_elements=matrix_elements,
        interaction_strengths=strengths,
        phonons=phonons,
        metadata=metadata,
    )


def _gaussian(values: Array, sigma: float) -> Array:
    return (
        np.exp(-(values**2) / (2.0 * sigma**2))
        / np.sqrt(2.0 * np.pi)
        / sigma
    )


def _bose_cyclic_thz(
    frequencies: Array,
    temperature: float,
    cutoff_frequency_thz: float,
) -> Array:
    if temperature == 0.0:
        return np.zeros_like(frequencies, dtype=np.float64)
    valid = frequencies > cutoff_frequency_thz
    exponent = np.zeros_like(frequencies, dtype=np.float64)
    exponent[valid] = (
        PHONO3PY_4_4_THZ_TO_EV
        * frequencies[valid]
        / (PHONO3PY_4_4_KB_EV_K * temperature)
    )
    result = np.zeros_like(exponent, dtype=np.float64)
    safe = valid & (exponent < 700.0)
    result[safe] = 1.0 / np.expm1(exponent[safe])
    return result


def _degenerate_groups(frequencies: Array, tolerance: float) -> tuple[Array, ...]:
    groups: list[Array] = []
    used: set[int] = set()
    for first in range(frequencies.size):
        if first in used:
            continue
        group = [first]
        used.add(first)
        for candidate in range(first + 1, frequencies.size):
            if any(
                abs(frequencies[index] - frequencies[candidate]) < tolerance
                for index in group
            ):
                group.append(candidate)
                used.add(candidate)
        groups.append(np.array(group, dtype=np.int64))
    return tuple(groups)


def _average_selected_degenerate(
    values: Array,
    frequencies: Array,
    selected_bands: Array,
    tolerance: float,
) -> Array:
    result = np.array(values, dtype=np.float64, copy=True)
    position = {int(band): index for index, band in enumerate(selected_bands)}
    for group in _degenerate_groups(frequencies, tolerance):
        selected_positions = [
            position[int(band)]
            for band in group
            if int(band) in position
        ]
        if not selected_positions:
            continue
        mean = np.mean(result[..., selected_positions], axis=-1)
        result[..., selected_positions] = mean[..., None]
    return result


@dataclass(frozen=True, slots=True, eq=False)
class BulkThreePhononLinewidths:
    """On-shell Gaussian three-phonon linewidths."""

    temperatures: Array
    grid_points: Array
    band_indices: Array
    gamma_thz_hwhm: Array
    gamma_normal_thz_hwhm: Array
    gamma_umklapp_thz_hwhm: Array
    gamma_angular_fwhm: Array
    decay_phase_space: Array
    difference_phase_space: Array
    interactions: BulkThreePhononInteractions
    metadata: Metadata

    def __post_init__(self) -> None:
        if not isinstance(self.interactions, BulkThreePhononInteractions):
            raise TypeError(
                "interactions must be a BulkThreePhononInteractions result."
            )
        temperatures = _readonly(self.temperatures, dtype=np.float64)
        grid_points = _readonly(self.grid_points, dtype=np.int64)
        bands = _readonly(self.band_indices, dtype=np.int64)
        if (
            temperatures.ndim != 1
            or temperatures.size < 1
            or not np.all(np.isfinite(temperatures))
            or np.any(temperatures < 0.0)
        ):
            raise ValueError(
                "temperatures must be a nonempty finite nonnegative vector."
            )
        if grid_points.ndim != 1 or grid_points.size < 1:
            raise ValueError("grid_points must be a nonempty vector.")
        if bands.ndim != 1 or bands.size < 1:
            raise ValueError("band_indices must be a nonempty vector.")
        if not np.array_equal(grid_points, self.interactions.grid_points):
            raise ValueError(
                "linewidth grid_points must equal interaction grid_points."
            )
        if not np.array_equal(bands, self.interactions.band_indices):
            raise ValueError(
                "linewidth band_indices must equal interaction band_indices."
            )
        expected = (temperatures.size, grid_points.size, bands.size)
        arrays = {
            "gamma_thz_hwhm": _readonly(
                self.gamma_thz_hwhm, dtype=np.float64
            ),
            "gamma_normal_thz_hwhm": _readonly(
                self.gamma_normal_thz_hwhm, dtype=np.float64
            ),
            "gamma_umklapp_thz_hwhm": _readonly(
                self.gamma_umklapp_thz_hwhm, dtype=np.float64
            ),
            "gamma_angular_fwhm": _readonly(
                self.gamma_angular_fwhm, dtype=np.float64
            ),
        }
        for name, values in arrays.items():
            if values.shape != expected or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must have finite shape {expected}.")
            if np.any(values < 0.0):
                raise ValueError(f"{name} must be nonnegative.")
        if not np.allclose(
            arrays["gamma_thz_hwhm"],
            arrays["gamma_normal_thz_hwhm"]
            + arrays["gamma_umklapp_thz_hwhm"],
            rtol=1.0e-13,
            atol=1.0e-15,
        ):
            raise ValueError(
                "Total linewidth must equal normal plus Umklapp linewidth."
            )
        if not np.allclose(
            arrays["gamma_angular_fwhm"],
            4.0 * np.pi * arrays["gamma_thz_hwhm"],
            rtol=1.0e-13,
            atol=1.0e-15,
        ):
            raise ValueError(
                "Angular FWHM linewidth must equal 4*pi*cyclic-THz HWHM."
            )
        phase_shape = (grid_points.size, bands.size)
        decay = _readonly(self.decay_phase_space, dtype=np.float64)
        difference = _readonly(self.difference_phase_space, dtype=np.float64)
        if decay.shape != phase_shape or difference.shape != phase_shape:
            raise ValueError("Phase-space arrays have inconsistent shape.")
        if not np.all(np.isfinite(decay)) or not np.all(np.isfinite(difference)):
            raise ValueError("Phase-space arrays must be finite.")
        if np.any(decay < 0.0):
            raise ValueError("Decay phase space must be nonnegative.")

        metadata = _freeze_mapping(self.metadata)
        for name in ("problem_sha256", "physics_input_sha256"):
            interaction_identity = self.interactions.metadata.get(name)
            linewidth_identity = metadata.get(name)
            if interaction_identity is not None and (
                linewidth_identity != interaction_identity
            ):
                raise ValueError(
                    f"Linewidth {name} conflicts with interaction metadata."
                )

        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "band_indices", bands)
        for name, values in arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "decay_phase_space", decay)
        object.__setattr__(self, "difference_phase_space", difference)
        object.__setattr__(self, "metadata", metadata)


def compute_three_phonon_linewidths(
    problem: BulkThreePhononProblem,
    *,
    temperatures: Sequence[float] | Array,
    grid_points: Sequence[int] | Array | None = None,
    band_indices: Sequence[int] | Array | None = None,
) -> BulkThreePhononLinewidths:
    """Compute cyclic-THz HWHM and angular-frequency FWHM linewidths."""

    raw_temperatures = np.asarray(temperatures)
    if raw_temperatures.ndim != 1 or raw_temperatures.dtype.kind == "c":
        raise TypeError("temperatures must be a one-dimensional real array.")
    try:
        temperature_array = np.array(
            raw_temperatures,
            dtype=np.float64,
            order="C",
            copy=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "temperatures must be a one-dimensional real array."
        ) from exc
    if (
        temperature_array.size == 0
        or not np.all(np.isfinite(temperature_array))
        or np.any(temperature_array < 0.0)
    ):
        raise ValueError("temperatures must be nonempty, finite, and nonnegative.")

    interactions = compute_three_phonon_interactions(
        problem,
        grid_points=grid_points,
        band_indices=band_indices,
    )
    n_temperatures = temperature_array.size
    n_targets = interactions.grid_points.size
    n_selected = interactions.band_indices.size
    gamma_normal = np.zeros(
        (n_temperatures, n_targets, n_selected),
        dtype=np.float64,
    )
    gamma_umklapp = np.zeros_like(gamma_normal)
    decay_phase_space = np.zeros((n_targets, n_selected), dtype=np.float64)
    difference_phase_space = np.zeros_like(decay_phase_space)
    conversion = phono3py_imag_self_energy_unit_factor()
    sigma = problem.sigma_thz
    frequencies = interactions.phonons.frequencies

    for target_position, target in enumerate(interactions.grid_points):
        f0 = frequencies[int(target), interactions.band_indices]
        for route_position in range(problem.mesh.n_qpoints):
            q1 = int(interactions.q1_indices[target_position, route_position])
            q2 = int(interactions.q2_indices[target_position, route_position])
            f1 = frequencies[q1]
            f2 = frequencies[q2]
            valid_internal = (
                (f1[:, None] > problem.cutoff_frequency_thz)
                & (f2[None, :] > problem.cutoff_frequency_thz)
            )
            g_decay = _gaussian(
                f0[:, None, None] - f1[None, :, None] - f2[None, None, :],
                sigma,
            )
            g_plus = _gaussian(
                f0[:, None, None] + f1[None, :, None] - f2[None, None, :],
                sigma,
            )
            g_minus = _gaussian(
                f0[:, None, None] - f1[None, :, None] + f2[None, None, :],
                sigma,
            )
            valid = (
                (f0[:, None, None] > problem.cutoff_frequency_thz)
                & valid_internal[None, :, :]
            )
            g_decay = np.where(valid, g_decay, 0.0)
            g_difference = np.where(valid, g_plus - g_minus, 0.0)
            decay_phase_space[target_position] += np.sum(
                g_decay,
                axis=(1, 2),
            ) / problem.mesh.n_qpoints
            difference_phase_space[target_position] += np.sum(
                g_difference,
                axis=(1, 2),
            ) / problem.mesh.n_qpoints

            strength = interactions.interaction_strengths[
                target_position,
                route_position,
            ]
            for temperature_position, temperature in enumerate(
                temperature_array
            ):
                occupations1 = _bose_cyclic_thz(
                    f1,
                    float(temperature),
                    problem.cutoff_frequency_thz,
                )
                occupations2 = _bose_cyclic_thz(
                    f2,
                    float(temperature),
                    problem.cutoff_frequency_thz,
                )
                channel = (
                    (
                        occupations1[None, :, None]
                        + occupations2[None, None, :]
                        + 1.0
                    )
                    * g_decay
                    + (
                        occupations1[None, :, None]
                        - occupations2[None, None, :]
                    )
                    * g_difference
                )
                contribution = np.sum(
                    channel * strength,
                    axis=(1, 2),
                ) * conversion
                if interactions.normal_mask[
                    target_position,
                    route_position,
                ]:
                    gamma_normal[
                        temperature_position,
                        target_position,
                    ] += contribution
                else:
                    gamma_umklapp[
                        temperature_position,
                        target_position,
                    ] += contribution

        for temperature_position in range(n_temperatures):
            gamma_normal[temperature_position, target_position] = (
                _average_selected_degenerate(
                    gamma_normal[temperature_position, target_position],
                    frequencies[int(target)],
                    interactions.band_indices,
                    problem.degenerate_tolerance_thz,
                )
            )
            gamma_umklapp[temperature_position, target_position] = (
                _average_selected_degenerate(
                    gamma_umklapp[temperature_position, target_position],
                    frequencies[int(target)],
                    interactions.band_indices,
                    problem.degenerate_tolerance_thz,
                )
            )

    gamma = gamma_normal + gamma_umklapp
    gamma_angular_fwhm = 4.0 * np.pi * gamma
    metadata = {
        "schema": "negfpy.bulk_three_phonon_linewidths.v1",
        "method": "gaussian_on_shell_full_mesh",
        "sigma_thz": sigma,
        "cutoff_frequency_thz": problem.cutoff_frequency_thz,
        "degenerate_tolerance_thz": problem.degenerate_tolerance_thz,
        "imag_self_energy_unit_factor": conversion,
        "gamma_thz_semantics": "cyclic-frequency HWHM",
        "gamma_angular_fwhm_conversion": "4*pi*gamma_thz_hwhm",
        "phase_space_units": "1/cyclic_THz",
        "phase_space_definition": (
            "sum_triplet,band1,band2("
            "triplet_weight*integration_weight)/Nq"
        ),
        "phase_space_cutoff_rule": (
            "strict_all_three_frequencies_above_cutoff"
        ),
        "phase_space_channel_0": (
            "Gaussian delta(omega-f1-f2): decay/sum channel"
        ),
        "phase_space_channel_1": (
            "Gaussian delta(omega+f1-f2) minus "
            "delta(omega-f1+f2): signed difference channel"
        ),
        "phase_space_normalization": "complete_full_mesh_1_over_Nq",
        "sigma_cutoff": None,
        "temperature_units": "K",
        "problem_sha256": interactions.metadata["problem_sha256"],
        "physics_input_sha256": interactions.metadata[
            "physics_input_sha256"
        ],
        "interaction_metadata": interactions.metadata,
    }
    return BulkThreePhononLinewidths(
        temperatures=temperature_array,
        grid_points=interactions.grid_points,
        band_indices=interactions.band_indices,
        gamma_thz_hwhm=gamma,
        gamma_normal_thz_hwhm=gamma_normal,
        gamma_umklapp_thz_hwhm=gamma_umklapp,
        gamma_angular_fwhm=gamma_angular_fwhm,
        decay_phase_space=decay_phase_space,
        difference_phase_space=difference_phase_space,
        interactions=interactions,
        metadata=metadata,
    )


__all__ = [
    "BulkThreePhononInteractions",
    "BulkThreePhononLinewidths",
    "BulkThreePhononProblem",
    "compute_three_phonon_interactions",
    "compute_three_phonon_linewidths",
    "phono3py_imag_self_energy_unit_factor",
    "phono3py_interaction_unit_factor",
]
