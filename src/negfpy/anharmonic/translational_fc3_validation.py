"""Read-only validation of explicit translational third-order IFCs.

The audit in this module never changes, symmetrizes, thresholds, truncates,
or completes a :class:`~negfpy.anharmonic.translational_fc3.TranslationalFC3`.
It checks the three independent atom-index acoustic sum rules at the
all-Gamma reciprocal tensor and reports the full Cartesian diameter of every
exactly nonzero expanded FC3 image.  The latter includes the second--third
atom distance, which can exceed both distances measured from the anchor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np

from .crystal import CrystalMapping, validate_crystal_mapping
from .translational_fc3 import FC3Term, TranslationalFC3


Array = np.ndarray
Metrics = Mapping[str, float]

PAIRWISE_ATOM_AXES = ((0, 1), (0, 2), (1, 2))
EXACT_NONZERO_SUPPORT_POLICY = (
    "an expanded term is supported iff at least one raw Cartesian FC3 "
    "component is numerically unequal to zero; no threshold is applied"
)
_PAIRWISE_NAMES = ("0_1", "0_2", "1_2")


def _readonly_array(
    values: Any,
    *,
    name: str,
    dtype: np.dtype[Any] | type,
    shape: tuple[int, ...] | None = None,
) -> Array:
    try:
        array = np.array(values, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numerical array.") from exc
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a non-negative integer.")
    integer = int(value)
    if integer < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return integer


def _nonnegative_tolerance(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative finite real number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a non-negative finite real number."
        ) from exc
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite real number.")
    return number


@dataclass(frozen=True, slots=True)
class TranslationalFC3ValidationReport:
    """Immutable validation result for :class:`TranslationalFC3`.

    Distances are expressed in the same length unit as the primitive lattice
    stored in the supplied :class:`CrystalMapping`.  Support is defined by
    exact numerical nonzero blocks; no numerical cutoff is applied.
    """

    mapping_valid: bool
    schedule_complete: bool
    gamma_finite: bool
    asr_valid_by_axis: tuple[bool, bool, bool]
    support_valid: bool
    gamma_asr_residuals: tuple[Array, Array, Array]
    gamma_asr_max_abs: Array
    gamma_asr_relative: Array
    gamma_scale: float
    asr_threshold: float
    pairwise_atom_axes: tuple[tuple[int, int], ...]
    pairwise_minimum_distances: Array
    pairwise_maximum_distances: Array
    maximum_pairwise_distance: float
    nonzero_expanded_term_count: int
    exact_zero_expanded_term_count: int
    nonzero_raw_block_count: int
    exact_zero_raw_block_count: int
    make_r0_average: bool
    phase_convention: str
    anchor_convention: str
    support_policy: str = EXACT_NONZERO_SUPPORT_POLICY
    distance_unit: str = "primitive_lattice_length_unit"
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Metrics = field(default_factory=dict)

    def __post_init__(self) -> None:
        axes = tuple(
            tuple(int(value) for value in pair)
            for pair in self.pairwise_atom_axes
        )
        if axes != PAIRWISE_ATOM_AXES:
            raise ValueError(
                f"pairwise_atom_axes must be exactly {PAIRWISE_ATOM_AXES}."
            )
        try:
            axis_valid = tuple(bool(value) for value in self.asr_valid_by_axis)
        except TypeError as exc:
            raise TypeError(
                "asr_valid_by_axis must contain three booleans."
            ) from exc
        if len(axis_valid) != 3:
            raise ValueError("asr_valid_by_axis must contain three booleans.")

        try:
            residual_inputs = tuple(self.gamma_asr_residuals)
        except TypeError as exc:
            raise TypeError(
                "gamma_asr_residuals must contain three arrays."
            ) from exc
        if len(residual_inputs) != 3:
            raise ValueError(
                "gamma_asr_residuals must contain three arrays."
            )
        residuals = tuple(
            _readonly_array(
                values,
                name=f"gamma_asr_residuals[{axis}]",
                dtype=np.complex128,
            )
            for axis, values in enumerate(residual_inputs)
        )
        if any(
            residual.ndim != 5 or residual.shape[-3:] != (3, 3, 3)
            for residual in residuals
        ):
            raise ValueError(
                "Each Gamma ASR residual must have shape "
                "(Np, Np, 3, 3, 3)."
            )
        if any(
            residual.shape != residuals[0].shape
            for residual in residuals[1:]
        ):
            raise ValueError("All Gamma ASR residual shapes must agree.")

        maxima = _readonly_array(
            self.gamma_asr_max_abs,
            name="gamma_asr_max_abs",
            dtype=np.float64,
            shape=(3,),
        )
        relative = _readonly_array(
            self.gamma_asr_relative,
            name="gamma_asr_relative",
            dtype=np.float64,
            shape=(3,),
        )
        pairwise_minimum = _readonly_array(
            self.pairwise_minimum_distances,
            name="pairwise_minimum_distances",
            dtype=np.float64,
            shape=(3,),
        )
        pairwise_maximum = _readonly_array(
            self.pairwise_maximum_distances,
            name="pairwise_maximum_distances",
            dtype=np.float64,
            shape=(3,),
        )
        if np.any(pairwise_minimum < 0.0) or np.any(pairwise_maximum < 0.0):
            raise ValueError("Pairwise support distances must be non-negative.")
        if np.any(pairwise_minimum > pairwise_maximum):
            raise ValueError(
                "Pairwise minimum distances cannot exceed their maxima."
            )

        gamma_scale = float(self.gamma_scale)
        threshold = float(self.asr_threshold)
        maximum_distance = float(self.maximum_pairwise_distance)
        if (
            not np.isfinite(gamma_scale)
            or gamma_scale < 0.0
            or not np.isfinite(threshold)
            or threshold < 0.0
            or not np.isfinite(maximum_distance)
            or maximum_distance < 0.0
        ):
            raise ValueError(
                "Scales, tolerances, and support distances must be finite "
                "and non-negative."
            )
        if not np.isclose(
            maximum_distance,
            float(np.max(pairwise_maximum)),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                "maximum_pairwise_distance must equal the largest pairwise "
                "maximum."
            )
        if not isinstance(self.distance_unit, str) or not self.distance_unit:
            raise ValueError("distance_unit must be a non-empty string.")
        if not isinstance(self.phase_convention, str) or not self.phase_convention:
            raise ValueError("phase_convention must be a non-empty string.")
        if not isinstance(self.anchor_convention, str) or not self.anchor_convention:
            raise ValueError("anchor_convention must be a non-empty string.")
        if self.support_policy != EXACT_NONZERO_SUPPORT_POLICY:
            raise ValueError(
                "support_policy must retain the exact-nonzero, no-threshold "
                "validation contract."
            )

        errors = tuple(str(value) for value in self.errors)
        warnings = tuple(str(value) for value in self.warnings)
        if any(not value for value in (*errors, *warnings)):
            raise ValueError("Validation messages must be non-empty strings.")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "Validation metric names must be non-empty strings."
                )
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(
                    f"Validation metric {name!r} must be finite."
                )
            metrics[name] = number

        object.__setattr__(self, "mapping_valid", bool(self.mapping_valid))
        object.__setattr__(
            self,
            "schedule_complete",
            bool(self.schedule_complete),
        )
        object.__setattr__(self, "gamma_finite", bool(self.gamma_finite))
        object.__setattr__(self, "asr_valid_by_axis", axis_valid)
        object.__setattr__(self, "support_valid", bool(self.support_valid))
        object.__setattr__(self, "gamma_asr_residuals", residuals)
        object.__setattr__(self, "gamma_asr_max_abs", maxima)
        object.__setattr__(self, "gamma_asr_relative", relative)
        object.__setattr__(self, "gamma_scale", gamma_scale)
        object.__setattr__(self, "asr_threshold", threshold)
        object.__setattr__(self, "pairwise_atom_axes", axes)
        object.__setattr__(
            self,
            "pairwise_minimum_distances",
            pairwise_minimum,
        )
        object.__setattr__(
            self,
            "pairwise_maximum_distances",
            pairwise_maximum,
        )
        object.__setattr__(
            self,
            "maximum_pairwise_distance",
            maximum_distance,
        )
        for name in (
            "nonzero_expanded_term_count",
            "exact_zero_expanded_term_count",
            "nonzero_raw_block_count",
            "exact_zero_raw_block_count",
        ):
            object.__setattr__(
                self,
                name,
                _nonnegative_integer(getattr(self, name), name=name),
            )
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "make_r0_average",
            bool(self.make_r0_average),
        )

    @property
    def asr_valid(self) -> bool:
        return bool(all(self.asr_valid_by_axis))

    @property
    def passed(self) -> bool:
        return bool(
            self.mapping_valid
            and self.schedule_complete
            and self.gamma_finite
            and self.asr_valid
            and self.support_valid
            and not self.errors
        )

    @property
    def valid(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.errors


def _primitive_index(mapping: CrystalMapping, satom: int) -> int:
    representative = int(mapping.s2p[satom])
    primitive = int(mapping.p2p[representative])
    if primitive < 0:
        raise ValueError(
            "A supercell atom could not be reduced to a primitive ordinal."
        )
    return primitive


def _schedule_audit(
    vertex: TranslationalFC3,
    mapping: CrystalMapping,
) -> tuple[bool, bool, list[str]]:
    """Check term-to-mapping compatibility and complete image expansion."""

    errors: list[str] = []
    mapping_valid = True
    schedule_complete = True
    n_primitive = mapping.n_primitive_atoms
    n_supercell = mapping.n_supercell_atoms
    if vertex.n_primitive_atoms != n_primitive:
        errors.append(
            "TranslationalFC3 primitive atom count conflicts with "
            "CrystalMapping."
        )
        mapping_valid = False
    if vertex.n_supercell_atoms != n_supercell:
        errors.append(
            "TranslationalFC3 supercell atom count conflicts with "
            "CrystalMapping."
        )
        mapping_valid = False
    if not mapping_valid:
        return mapping_valid, False, errors

    groups: dict[tuple[int, int, int], list[FC3Term]] = {}
    for term_position, term in enumerate(vertex.terms):
        anchor = term.anchor
        source_first, source_second, source_third = term.source_indices
        translated_first, translated_second, translated_third = (
            term.translated_supercell_indices
        )
        group_key = (anchor, source_second, source_third)
        groups.setdefault(group_key, []).append(term)
        expected_source_first = (
            anchor
            if vertex.source_layout == "compact"
            else int(mapping.p2s[anchor])
        )
        compatible = True
        if source_first != expected_source_first:
            compatible = False
        if translated_first != int(mapping.p2s[anchor]):
            compatible = False
        if not np.array_equal(
            term.anchor_position,
            mapping.primitive.scaled_positions[anchor],
        ):
            compatible = False
        if term.primitive_indices[1] != _primitive_index(
            mapping,
            translated_second,
        ):
            compatible = False
        if term.primitive_indices[2] != _primitive_index(
            mapping,
            translated_third,
        ):
            compatible = False

        second_count, second_start = mapping.multiplicity[
            translated_second,
            anchor,
        ]
        third_count, third_start = mapping.multiplicity[
            translated_third,
            anchor,
        ]
        if term.second_multiplicity != int(second_count):
            compatible = False
        if term.third_multiplicity != int(third_count):
            compatible = False
        if not (
            int(second_start)
            <= term.second_vector_index
            < int(second_start + second_count)
        ):
            compatible = False
        elif not np.array_equal(
            term.second_offset,
            mapping.shortest_vectors[term.second_vector_index],
        ):
            compatible = False
        if not (
            int(third_start)
            <= term.third_vector_index
            < int(third_start + third_count)
        ):
            compatible = False
        elif not np.array_equal(
            term.third_offset,
            mapping.shortest_vectors[term.third_vector_index],
        ):
            compatible = False
        if not compatible:
            errors.append(
                "TranslationalFC3 term "
                f"{term_position} conflicts with CrystalMapping indices, "
                "positions, multiplicities, or shortest vectors."
            )
            mapping_valid = False

    expected_groups = {
        (anchor, second, third)
        for anchor in range(n_primitive)
        for second in range(n_supercell)
        for third in range(n_supercell)
    }
    actual_groups = set(groups)
    missing = expected_groups - actual_groups
    extra = actual_groups - expected_groups
    if missing or extra:
        errors.append(
            "The expanded term schedule is incomplete or contains unexpected "
            f"raw groups: {len(missing)} missing, {len(extra)} unexpected."
        )
        schedule_complete = False

    for key in sorted(expected_groups & actual_groups):
        group = groups[key]
        first = group[0]
        translated_second = first.translated_supercell_indices[1]
        translated_third = first.translated_supercell_indices[2]
        second_count, second_start = mapping.multiplicity[
            translated_second,
            first.anchor,
        ]
        third_count, third_start = mapping.multiplicity[
            translated_third,
            first.anchor,
        ]
        expected_pairs = {
            (second_index, third_index)
            for second_index in range(
                int(second_start),
                int(second_start + second_count),
            )
            for third_index in range(
                int(third_start),
                int(third_start + third_count),
            )
        }
        actual_pairs = {
            (term.second_vector_index, term.third_vector_index)
            for term in group
        }
        group_consistent = all(
            term.translated_supercell_indices
            == first.translated_supercell_indices
            and term.second_multiplicity == first.second_multiplicity
            and term.third_multiplicity == first.third_multiplicity
            and term.anchor_weight == first.anchor_weight
            and term.multiplicity_weight == first.multiplicity_weight
            and np.array_equal(term.values, first.values)
            for term in group
        )
        if (
            len(group) != len(expected_pairs)
            or actual_pairs != expected_pairs
            or not group_consistent
        ):
            errors.append(
                "The expanded term schedule for raw group "
                f"{key} does not contain exactly one consistent term for "
                "each shortest-vector image pair."
            )
            schedule_complete = False

    if not mapping_valid:
        schedule_complete = False
    return mapping_valid, schedule_complete, errors


def _pairwise_support(
    vertex: TranslationalFC3,
    primitive_lattice: Array,
) -> tuple[Array, Array, int, int, int, int]:
    distances: list[tuple[float, float, float]] = []
    raw_nonzero: dict[tuple[int, int, int], bool] = {}
    nonzero_terms = 0
    zero_terms = 0
    for term in vertex.terms:
        nonzero = bool(np.any(term.values != 0))
        previous = raw_nonzero.get(term.source_indices, False)
        raw_nonzero[term.source_indices] = previous or nonzero
        if not nonzero:
            zero_terms += 1
            continue
        nonzero_terms += 1
        second = term.second_offset @ primitive_lattice
        third = term.third_offset @ primitive_lattice
        distances.append(
            (
                float(np.linalg.norm(second)),
                float(np.linalg.norm(third)),
                float(np.linalg.norm(third - second)),
            )
        )

    if distances:
        array = np.asarray(distances, dtype=np.float64)
        minimum = np.min(array, axis=0)
        maximum = np.max(array, axis=0)
    else:
        minimum = np.zeros(3, dtype=np.float64)
        maximum = np.zeros(3, dtype=np.float64)
    nonzero_raw = sum(raw_nonzero.values())
    zero_raw = len(raw_nonzero) - nonzero_raw
    return (
        minimum,
        maximum,
        nonzero_terms,
        zero_terms,
        nonzero_raw,
        zero_raw,
    )


def validate_translational_fc3(
    vertex: TranslationalFC3,
    mapping: CrystalMapping,
    *,
    asr_atol: float = 1.0e-10,
    asr_rtol: float = 1.0e-8,
) -> TranslationalFC3ValidationReport:
    """Audit Gamma translational ASR and complete Cartesian FC3 support.

    The three ASR tensors are

    ``sum_i Phi_ijk(Gamma)``, ``sum_j Phi_ijk(Gamma)``, and
    ``sum_k Phi_ijk(Gamma)``.

    The common acceptance threshold is
    ``asr_atol + asr_rtol * max(abs(Phi(Gamma)))``.  Support uses exact
    numerical nonzero blocks and reports distances for atom pairs ``(0, 1)``,
    ``(0, 2)``, and ``(1, 2)`` separately.
    """

    if not isinstance(vertex, TranslationalFC3):
        raise TypeError("vertex must be a TranslationalFC3 instance.")
    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping instance.")
    asr_atol = _nonnegative_tolerance(asr_atol, name="asr_atol")
    asr_rtol = _nonnegative_tolerance(asr_rtol, name="asr_rtol")

    errors: list[str] = []
    warnings: list[str] = []
    mapping_report = validate_crystal_mapping(mapping)
    mapping_valid = mapping_report.passed and mapping.purpose == "fc3"
    if mapping.purpose != "fc3":
        errors.append("CrystalMapping purpose must be 'fc3'.")
    if not mapping_report.passed:
        errors.extend(mapping_report.errors)

    term_mapping_valid, schedule_complete, schedule_errors = _schedule_audit(
        vertex,
        mapping,
    )
    mapping_valid = mapping_valid and term_mapping_valid
    errors.extend(schedule_errors)

    gamma = vertex.fourier(np.zeros((3, 3), dtype=np.float64))
    gamma_finite = bool(np.all(np.isfinite(gamma)))
    if not gamma_finite:
        errors.append("The all-Gamma FC3 tensor contains nonfinite values.")
    residuals = tuple(
        np.sum(gamma, axis=axis, dtype=np.complex128)
        for axis in range(3)
    )
    maxima = np.array(
        [
            float(np.max(np.abs(residual)))
            for residual in residuals
        ],
        dtype=np.float64,
    )
    gamma_scale = float(np.max(np.abs(gamma)))
    denominator = max(gamma_scale, np.finfo(np.float64).tiny)
    relative = maxima / denominator
    threshold = asr_atol + asr_rtol * gamma_scale
    axis_valid = tuple(bool(value <= threshold) for value in maxima)
    for axis, valid in enumerate(axis_valid):
        if not valid:
            errors.append(
                "FC3 translational acoustic sum rule failed on atom axis "
                f"{axis}: maximum Gamma residual {maxima[axis]:.6e}, "
                f"allowed {threshold:.6e}."
            )

    (
        pairwise_minimum,
        pairwise_maximum,
        nonzero_terms,
        zero_terms,
        nonzero_raw,
        zero_raw,
    ) = _pairwise_support(vertex, mapping.primitive.lattice)
    distances_finite = bool(
        np.all(np.isfinite(pairwise_minimum))
        and np.all(np.isfinite(pairwise_maximum))
    )
    support_valid = bool(
        mapping_valid and schedule_complete and distances_finite
    )
    if not distances_finite:
        errors.append("Pairwise Cartesian support distances are nonfinite.")
    if nonzero_terms == 0:
        warnings.append(
            "TranslationalFC3 contains no exactly nonzero expanded terms; "
            "its reported Cartesian support is zero."
        )

    maximum_distance = float(np.max(pairwise_maximum))
    metrics: dict[str, float] = {
        "asr_atol": asr_atol,
        "asr_rtol": asr_rtol,
        "gamma_fc3_scale": gamma_scale,
        "gamma_asr_threshold": threshold,
        "expanded_term_count": float(len(vertex.terms)),
        "nonzero_expanded_term_count": float(nonzero_terms),
        "exact_zero_expanded_term_count": float(zero_terms),
        "nonzero_raw_block_count": float(nonzero_raw),
        "exact_zero_raw_block_count": float(zero_raw),
        "support_maximum_pairwise_distance": maximum_distance,
    }
    for axis in range(3):
        metrics[f"gamma_asr_axis_{axis}_max_abs"] = float(maxima[axis])
        metrics[f"gamma_asr_axis_{axis}_relative"] = float(relative[axis])
    for pair_name, minimum, maximum in zip(
        _PAIRWISE_NAMES,
        pairwise_minimum,
        pairwise_maximum,
        strict=True,
    ):
        metrics[f"support_pair_{pair_name}_minimum"] = float(minimum)
        metrics[f"support_pair_{pair_name}_maximum"] = float(maximum)

    return TranslationalFC3ValidationReport(
        mapping_valid=mapping_valid,
        schedule_complete=schedule_complete,
        gamma_finite=gamma_finite,
        asr_valid_by_axis=axis_valid,
        support_valid=support_valid,
        gamma_asr_residuals=residuals,
        gamma_asr_max_abs=maxima,
        gamma_asr_relative=relative,
        gamma_scale=gamma_scale,
        asr_threshold=threshold,
        pairwise_atom_axes=PAIRWISE_ATOM_AXES,
        pairwise_minimum_distances=pairwise_minimum,
        pairwise_maximum_distances=pairwise_maximum,
        maximum_pairwise_distance=maximum_distance,
        nonzero_expanded_term_count=nonzero_terms,
        exact_zero_expanded_term_count=zero_terms,
        nonzero_raw_block_count=nonzero_raw,
        exact_zero_raw_block_count=zero_raw,
        make_r0_average=vertex.make_r0_average,
        phase_convention=vertex.phase_convention,
        anchor_convention=vertex.anchor_convention,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )


__all__ = [
    "EXACT_NONZERO_SUPPORT_POLICY",
    "PAIRWISE_ATOM_AXES",
    "TranslationalFC3ValidationReport",
    "validate_translational_fc3",
]
