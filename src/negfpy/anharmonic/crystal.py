"""Immutable crystal and phono3py-style mapping primitives.

The objects in this module are detached numerical records.  They do not retain
live phonopy or phono3py objects, and they do not infer, repair, reorder, or
symmetrize mappings.  In particular, ``s2p`` follows phonopy's convention:
its values are representative *supercell* indices contained in ``p2s``, not
primitive-cell ordinals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


Array = np.ndarray
Metadata = Mapping[str, Any]
MappingPurpose = Literal["fc2", "fc3"]

_CARTESIAN_ATOL = 1.0e-10


def _readonly_float_array(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind == "c":
        raise TypeError(f"{name} must be real-valued.")
    try:
        array = np.array(values, dtype=np.float64, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numerical array.") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    array.setflags(write=False)
    return array


def _readonly_int_array(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu":
        raise TypeError(f"{name} must contain integers.")
    array = np.array(raw, dtype=np.int64, order="C", copy=True)
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
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
        raise TypeError("provenance must be a mapping.")
    return MappingProxyType(
        {str(key): _freeze_value(value) for key, value in values.items()}
    )


def _normalize_symbols(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("symbols must be a sequence of atomic-symbol strings.")
    try:
        symbols = tuple(values)
    except TypeError as exc:
        raise TypeError(
            "symbols must be a sequence of atomic-symbol strings."
        ) from exc
    if any(not isinstance(value, str) or not value.strip() for value in symbols):
        raise ValueError("symbols must contain non-empty strings.")
    return symbols


def _matrix_is_nonsingular(values: Array) -> bool:
    return bool(np.linalg.matrix_rank(values) == 3)


@dataclass(frozen=True, slots=True, eq=False)
class CrystalCell:
    """Detached primitive- or supercell geometry.

    Lattice vectors are stored row-wise.  Cartesian positions therefore obey
    ``cartesian_positions = scaled_positions @ lattice``.  If Cartesian
    positions are supplied explicitly, they must agree with this convention.
    """

    lattice: Array
    scaled_positions: Array
    masses: Array
    symbols: tuple[str, ...]
    cartesian_positions: Array | None = None

    def __post_init__(self) -> None:
        lattice = _readonly_float_array(self.lattice, name="lattice")
        scaled = _readonly_float_array(
            self.scaled_positions, name="scaled_positions"
        )
        masses = _readonly_float_array(self.masses, name="masses")
        symbols = _normalize_symbols(self.symbols)

        if lattice.shape != (3, 3):
            raise ValueError("lattice must have shape (3, 3).")
        if not _matrix_is_nonsingular(lattice):
            raise ValueError("lattice must be nonsingular.")
        if scaled.ndim != 2 or scaled.shape[1:] != (3,):
            raise ValueError("scaled_positions must have shape (n_atoms, 3).")
        n_atoms = int(scaled.shape[0])
        if n_atoms < 1:
            raise ValueError("A crystal cell must contain at least one atom.")
        if masses.shape != (n_atoms,):
            raise ValueError("masses must contain one value per atom.")
        if np.any(masses <= 0.0):
            raise ValueError("masses must be strictly positive.")
        if len(symbols) != n_atoms:
            raise ValueError("symbols must contain one value per atom.")

        derived_cartesian = np.array(
            scaled @ lattice,
            dtype=np.float64,
            order="C",
            copy=True,
        )
        if self.cartesian_positions is not None:
            supplied_cartesian = _readonly_float_array(
                self.cartesian_positions,
                name="cartesian_positions",
            )
            if supplied_cartesian.shape != (n_atoms, 3):
                raise ValueError(
                    "cartesian_positions must have shape (n_atoms, 3)."
                )
            if not np.allclose(
                supplied_cartesian,
                derived_cartesian,
                rtol=0.0,
                atol=_CARTESIAN_ATOL,
            ):
                raise ValueError(
                    "cartesian_positions are inconsistent with "
                    "scaled_positions @ lattice."
                )
        derived_cartesian.setflags(write=False)

        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "scaled_positions", scaled)
        object.__setattr__(self, "cartesian_positions", derived_cartesian)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "symbols", symbols)

    @property
    def n_atoms(self) -> int:
        return int(self.scaled_positions.shape[0])


@dataclass(frozen=True, slots=True)
class CrystalMappingValidationReport:
    """Immutable outcome of a mapping audit or primitive comparison."""

    purpose_valid: bool
    matrices_valid: bool
    cells_valid: bool
    mapping_valid: bool
    shortest_vectors_valid: bool
    atomic_permutations_valid: bool
    translation_permutations_valid: bool = True
    primitive_compatible: bool | None = None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
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

        compatible = self.primitive_compatible
        if compatible is not None:
            compatible = bool(compatible)

        object.__setattr__(self, "purpose_valid", bool(self.purpose_valid))
        object.__setattr__(self, "matrices_valid", bool(self.matrices_valid))
        object.__setattr__(self, "cells_valid", bool(self.cells_valid))
        object.__setattr__(self, "mapping_valid", bool(self.mapping_valid))
        object.__setattr__(
            self,
            "shortest_vectors_valid",
            bool(self.shortest_vectors_valid),
        )
        object.__setattr__(
            self,
            "atomic_permutations_valid",
            bool(self.atomic_permutations_valid),
        )
        object.__setattr__(
            self,
            "translation_permutations_valid",
            bool(self.translation_permutations_valid),
        )
        object.__setattr__(self, "primitive_compatible", compatible)
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))

    @property
    def passed(self) -> bool:
        compatible = self.primitive_compatible
        return bool(
            self.purpose_valid
            and self.matrices_valid
            and self.cells_valid
            and self.mapping_valid
            and self.shortest_vectors_valid
            and self.atomic_permutations_valid
            and self.translation_permutations_valid
            and compatible is not False
            and not self.errors
        )

    @property
    def valid(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.errors


def _mapping_errors(
    *,
    primitive_matrix: Array,
    supercell_matrix: Array,
    primitive: CrystalCell,
    supercell: CrystalCell,
    p2s: Array,
    s2p: Array,
    p2p: Array,
    shortest_vectors: Array,
    multiplicity: Array,
    atomic_permutations: Array,
    translation_permutations: Array | None,
    purpose: str,
) -> tuple[list[str], dict[str, bool], dict[str, float]]:
    errors: list[str] = []
    status = {
        "purpose_valid": True,
        "matrices_valid": True,
        "cells_valid": True,
        "mapping_valid": True,
        "shortest_vectors_valid": True,
        "atomic_permutations_valid": True,
        "translation_permutations_valid": True,
    }

    if purpose not in {"fc2", "fc3"}:
        status["purpose_valid"] = False
        errors.append("purpose must be either 'fc2' or 'fc3'.")

    if primitive_matrix.shape != (3, 3) or not _matrix_is_nonsingular(
        primitive_matrix
    ):
        status["matrices_valid"] = False
        errors.append("primitive_matrix must be a nonsingular (3, 3) matrix.")
    if supercell_matrix.shape != (3, 3) or not _matrix_is_nonsingular(
        supercell_matrix
    ):
        status["matrices_valid"] = False
        errors.append("supercell_matrix must be a nonsingular (3, 3) matrix.")

    n_primitive = primitive.n_atoms
    n_supercell = supercell.n_atoms
    if p2s.shape != (n_primitive,):
        status["mapping_valid"] = False
        errors.append("p2s must contain one representative per primitive atom.")
    if s2p.shape != (n_supercell,):
        status["mapping_valid"] = False
        errors.append("s2p must contain one representative per supercell atom.")
    if p2p.shape != (n_supercell,):
        status["mapping_valid"] = False
        errors.append(
            "p2p must be a dense supercell-index array with -1 outside "
            "representative indices."
        )

    p2s_indices_valid = bool(
        p2s.shape == (n_primitive,)
        and np.all((p2s >= 0) & (p2s < n_supercell))
        and np.unique(p2s).size == n_primitive
    )
    if not p2s_indices_valid:
        status["mapping_valid"] = False
        errors.append("p2s must contain unique in-range supercell indices.")

    if s2p.shape == (n_supercell,) and p2s_indices_valid:
        representatives = set(int(value) for value in p2s)
        if any(int(value) not in representatives for value in s2p):
            status["mapping_valid"] = False
            errors.append(
                "s2p values must all be representative supercell indices "
                "contained in p2s."
            )
        elif not np.array_equal(s2p[p2s], p2s):
            status["mapping_valid"] = False
            errors.append("The official identity s2p[p2s[p]] == p2s[p] failed.")

    if p2p.shape == (n_supercell,) and p2s_indices_valid:
        expected_p2p = np.full(n_supercell, -1, dtype=np.int64)
        expected_p2p[p2s] = np.arange(n_primitive, dtype=np.int64)
        if not np.array_equal(p2p, expected_p2p):
            status["mapping_valid"] = False
            errors.append(
                "The official identity p2p[p2s[p]] == p failed, or p2p "
                "contains values outside representative indices."
            )

    if shortest_vectors.ndim != 2 or shortest_vectors.shape[1:] != (3,):
        status["shortest_vectors_valid"] = False
        errors.append("shortest_vectors must have shape (n_vectors, 3).")
    if multiplicity.shape != (n_supercell, n_primitive, 2):
        status["shortest_vectors_valid"] = False
        errors.append(
            "multiplicity must have shape (n_supercell, n_primitive, 2)."
        )
    elif shortest_vectors.ndim == 2 and shortest_vectors.shape[1:] == (3,):
        counts = multiplicity[..., 0]
        offsets = multiplicity[..., 1]
        if np.any(counts <= 0) or np.any(offsets < 0):
            status["shortest_vectors_valid"] = False
            errors.append(
                "Shortest-vector multiplicities must be positive and offsets "
                "must be nonnegative."
            )
        else:
            intervals = sorted(
                (
                    int(offset),
                    int(offset + count),
                )
                for count, offset in zip(
                    counts.ravel(),
                    offsets.ravel(),
                    strict=True,
                )
            )
            cursor = 0
            partition_valid = True
            for start, stop in intervals:
                if start != cursor or stop <= start:
                    partition_valid = False
                    break
                cursor = stop
            if cursor != int(shortest_vectors.shape[0]):
                partition_valid = False
            if not partition_valid:
                status["shortest_vectors_valid"] = False
                errors.append(
                    "Multiplicity offsets and counts must exactly partition "
                    "the flattened shortest-vector storage without gaps or "
                    "overlaps."
                )

    if (
        atomic_permutations.ndim != 2
        or atomic_permutations.shape[0] < 1
        or atomic_permutations.shape[1] != n_supercell
    ):
        status["atomic_permutations_valid"] = False
        errors.append(
            "atomic_permutations must have shape "
            "(n_permutations, n_supercell_atoms)."
        )
    else:
        expected = np.arange(n_supercell, dtype=np.int64)
        if any(
            not np.array_equal(np.sort(row), expected)
            for row in atomic_permutations
        ):
            status["atomic_permutations_valid"] = False
            errors.append(
                "Every atomic_permutations row must be a permutation of all "
                "supercell atom indices."
            )

    if translation_permutations is not None:
        expected_translation_count = (
            n_supercell // n_primitive
            if n_primitive > 0 and n_supercell % n_primitive == 0
            else -1
        )
        if (
            translation_permutations.ndim != 2
            or translation_permutations.shape
            != (expected_translation_count, n_supercell)
        ):
            status["translation_permutations_valid"] = False
            errors.append(
                "translation_permutations must have shape "
                "(n_supercell/n_primitive, n_supercell)."
            )
        else:
            expected = np.arange(n_supercell, dtype=np.int64)
            rows_are_permutations = not any(
                not np.array_equal(np.sort(row), expected)
                for row in translation_permutations
            )
            if not rows_are_permutations:
                status["translation_permutations_valid"] = False
                errors.append(
                    "Every translation_permutations row must be a permutation "
                    "of all supercell atom indices."
                )
            if rows_are_permutations and s2p.shape == (n_supercell,):
                for row in translation_permutations:
                    if not np.array_equal(s2p[row], s2p):
                        status["translation_permutations_valid"] = False
                        errors.append(
                            "translation_permutations must preserve every "
                            "primitive atom class."
                        )
                        break
            if (
                rows_are_permutations
                and p2s_indices_valid
                and s2p.shape == (n_supercell,)
            ):
                for source in range(n_supercell):
                    representative = int(s2p[source])
                    hit_count = int(
                        np.count_nonzero(
                            translation_permutations[:, source]
                            == representative
                        )
                    )
                    if hit_count != 1:
                        status["translation_permutations_valid"] = False
                        errors.append(
                            "Exactly one pure translation must map each "
                            "supercell atom to its p2s representative."
                        )
                        break

    metrics = {
        "primitive_atom_count": float(n_primitive),
        "supercell_atom_count": float(n_supercell),
        "shortest_vector_count": float(
            shortest_vectors.shape[0] if shortest_vectors.ndim == 2 else 0
        ),
        "atomic_permutation_count": float(
            atomic_permutations.shape[0]
            if atomic_permutations.ndim == 2
            else 0
        ),
        "translation_permutation_count": float(
            0
            if translation_permutations is None
            else translation_permutations.shape[0]
        ),
    }
    return errors, status, metrics


@dataclass(frozen=True, slots=True, eq=False)
class CrystalMapping:
    """Immutable primitive/supercell mapping detached from phono3py.

    ``p2p`` is stored densely over supercell indices.  It equals the primitive
    ordinal at representative indices in ``p2s`` and ``-1`` elsewhere.  This
    is the array equivalent of phonopy's sparse ``p2p_map`` dictionary.

    ``multiplicity[..., 0]`` stores shortest-vector counts and
    ``multiplicity[..., 1]`` stores offsets into flattened
    ``shortest_vectors``.
    """

    primitive_matrix: Array
    supercell_matrix: Array
    primitive: CrystalCell
    supercell: CrystalCell
    p2s: Array
    s2p: Array
    p2p: Array
    shortest_vectors: Array
    multiplicity: Array
    atomic_permutations: Array
    purpose: MappingPurpose
    translation_permutations: Array | None = None
    provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, CrystalCell):
            raise TypeError("primitive must be a CrystalCell.")
        if not isinstance(self.supercell, CrystalCell):
            raise TypeError("supercell must be a CrystalCell.")
        if not isinstance(self.purpose, str):
            raise TypeError("purpose must be either 'fc2' or 'fc3'.")

        primitive_matrix = _readonly_float_array(
            self.primitive_matrix,
            name="primitive_matrix",
        )
        supercell_matrix = _readonly_int_array(
            self.supercell_matrix,
            name="supercell_matrix",
        )
        p2s = _readonly_int_array(self.p2s, name="p2s")
        s2p = _readonly_int_array(self.s2p, name="s2p")
        p2p = _readonly_int_array(self.p2p, name="p2p")
        shortest_vectors = _readonly_float_array(
            self.shortest_vectors,
            name="shortest_vectors",
        )
        multiplicity = _readonly_int_array(
            self.multiplicity,
            name="multiplicity",
        )
        atomic_permutations = _readonly_int_array(
            self.atomic_permutations,
            name="atomic_permutations",
        )
        translation_permutations = (
            None
            if self.translation_permutations is None
            else _readonly_int_array(
                self.translation_permutations,
                name="translation_permutations",
            )
        )
        provenance = _freeze_mapping(self.provenance)

        errors, _, _ = _mapping_errors(
            primitive_matrix=primitive_matrix,
            supercell_matrix=supercell_matrix,
            primitive=self.primitive,
            supercell=self.supercell,
            p2s=p2s,
            s2p=s2p,
            p2p=p2p,
            shortest_vectors=shortest_vectors,
            multiplicity=multiplicity,
            atomic_permutations=atomic_permutations,
            translation_permutations=translation_permutations,
            purpose=self.purpose,
        )
        if errors:
            raise ValueError("Invalid crystal mapping: " + " ".join(errors))

        object.__setattr__(self, "primitive_matrix", primitive_matrix)
        object.__setattr__(self, "supercell_matrix", supercell_matrix)
        object.__setattr__(self, "p2s", p2s)
        object.__setattr__(self, "s2p", s2p)
        object.__setattr__(self, "p2p", p2p)
        object.__setattr__(self, "shortest_vectors", shortest_vectors)
        object.__setattr__(self, "multiplicity", multiplicity)
        object.__setattr__(
            self,
            "atomic_permutations",
            atomic_permutations,
        )
        object.__setattr__(
            self,
            "translation_permutations",
            translation_permutations,
        )
        object.__setattr__(self, "purpose", self.purpose)
        object.__setattr__(self, "provenance", provenance)

    @property
    def n_primitive_atoms(self) -> int:
        return self.primitive.n_atoms

    @property
    def n_supercell_atoms(self) -> int:
        return self.supercell.n_atoms


def validate_crystal_mapping(
    mapping: CrystalMapping,
) -> CrystalMappingValidationReport:
    """Re-audit an immutable mapping without modifying it."""

    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping.")
    errors, status, metrics = _mapping_errors(
        primitive_matrix=mapping.primitive_matrix,
        supercell_matrix=mapping.supercell_matrix,
        primitive=mapping.primitive,
        supercell=mapping.supercell,
        p2s=mapping.p2s,
        s2p=mapping.s2p,
        p2p=mapping.p2p,
        shortest_vectors=mapping.shortest_vectors,
        multiplicity=mapping.multiplicity,
        atomic_permutations=mapping.atomic_permutations,
        translation_permutations=mapping.translation_permutations,
        purpose=mapping.purpose,
    )
    return CrystalMappingValidationReport(
        purpose_valid=status["purpose_valid"],
        matrices_valid=status["matrices_valid"],
        cells_valid=status["cells_valid"],
        mapping_valid=status["mapping_valid"],
        shortest_vectors_valid=status["shortest_vectors_valid"],
        atomic_permutations_valid=status["atomic_permutations_valid"],
        translation_permutations_valid=status[
            "translation_permutations_valid"
        ],
        errors=tuple(errors),
        metrics=metrics,
    )


def _max_abs(values: Array) -> float:
    if values.size == 0:
        return 0.0
    return float(np.max(np.abs(values)))


def compare_primitive_mappings(
    left: CrystalMapping,
    right: CrystalMapping,
    *,
    atol: float = 1.0e-10,
) -> CrystalMappingValidationReport:
    """Compare the shared primitive represented by two crystal mappings.

    Supercell matrices, supercell atom counts, and their associated mapping
    arrays are intentionally not compared.  FC2 and FC3 may use different
    supercells while still referring to the same ordered primitive basis.
    Scaled positions are compared modulo lattice translations; atom
    reordering is never inferred.
    """

    if not isinstance(left, CrystalMapping) or not isinstance(
        right, CrystalMapping
    ):
        raise TypeError("left and right must be CrystalMapping instances.")
    if isinstance(atol, (bool, np.bool_)):
        raise TypeError("atol must be a nonnegative finite real number.")
    atol = float(atol)
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a nonnegative finite real number.")

    left_report = validate_crystal_mapping(left)
    right_report = validate_crystal_mapping(right)
    errors = [
        *(f"Left mapping: {message}" for message in left_report.errors),
        *(f"Right mapping: {message}" for message in right_report.errors),
    ]
    metrics: dict[str, float] = {}

    primitive_matrix_delta = left.primitive_matrix - right.primitive_matrix
    primitive_matrix_max_abs = _max_abs(primitive_matrix_delta)
    metrics["primitive_matrix_max_abs"] = primitive_matrix_max_abs
    if primitive_matrix_max_abs > atol:
        errors.append("Primitive matrices differ.")

    lattice_delta = left.primitive.lattice - right.primitive.lattice
    lattice_max_abs = _max_abs(lattice_delta)
    metrics["primitive_lattice_max_abs"] = lattice_max_abs
    if lattice_max_abs > atol:
        errors.append("Primitive lattices differ.")

    left_count = left.n_primitive_atoms
    right_count = right.n_primitive_atoms
    metrics["left_primitive_atom_count"] = float(left_count)
    metrics["right_primitive_atom_count"] = float(right_count)
    if left_count != right_count:
        errors.append("Primitive atom counts differ.")
    else:
        position_delta = (
            left.primitive.scaled_positions
            - right.primitive.scaled_positions
        )
        position_delta -= np.rint(position_delta)
        position_max_abs = _max_abs(position_delta)
        metrics["primitive_scaled_position_modulo_max_abs"] = position_max_abs
        if position_max_abs > atol:
            errors.append(
                "Primitive scaled positions differ modulo lattice translations."
            )

        mass_max_abs = _max_abs(left.primitive.masses - right.primitive.masses)
        metrics["primitive_mass_max_abs"] = mass_max_abs
        if mass_max_abs > atol:
            errors.append("Primitive masses differ.")

        if left.primitive.symbols != right.primitive.symbols:
            errors.append("Primitive symbol order differs.")

    compatible = bool(
        left_report.passed and right_report.passed and not errors
    )
    return CrystalMappingValidationReport(
        purpose_valid=left_report.purpose_valid and right_report.purpose_valid,
        matrices_valid=left_report.matrices_valid
        and right_report.matrices_valid,
        cells_valid=left_report.cells_valid and right_report.cells_valid,
        mapping_valid=left_report.mapping_valid and right_report.mapping_valid,
        shortest_vectors_valid=left_report.shortest_vectors_valid
        and right_report.shortest_vectors_valid,
        atomic_permutations_valid=left_report.atomic_permutations_valid
        and right_report.atomic_permutations_valid,
        translation_permutations_valid=(
            left_report.translation_permutations_valid
            and right_report.translation_permutations_valid
        ),
        primitive_compatible=compatible,
        errors=tuple(errors),
        metrics=metrics,
    )


__all__ = [
    "CrystalCell",
    "CrystalMapping",
    "CrystalMappingValidationReport",
    "compare_primitive_mappings",
    "validate_crystal_mapping",
]
