"""Non-mutating scientific audit of harmonic real-space IFC data.

The existing :func:`validate_ifc_data` intentionally performs only lightweight
shape checks.  This module adds a stricter, report-oriented audit without
changing that established behavior and without modifying or symmetrizing the
input data.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from itertools import product
from types import MappingProxyType
from typing import Any

import numpy as np

from negfpy.modeling.schema import IFCData, IFCTerm


Array = np.ndarray
Translation = tuple[int, int, int]
QPoint = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class IFCValidationReport:
    """Immutable outcome of :func:`audit_ifc_data`.

    Duplicate translations are summed internally for all physical checks and
    are listed explicitly in ``duplicate_translations``.  The merged data are
    never written back to the input :class:`IFCData`.
    """

    schema_valid: bool
    mapping_valid: bool
    reciprocal_valid: bool
    asr_valid: bool
    hermiticity_valid: bool
    stable: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    qpoints: tuple[QPoint, ...] = ()
    duplicate_translations: tuple[Translation, ...] = ()
    missing_reciprocal_translations: tuple[Translation, ...] = ()
    minimum_eigenvalue_qpoint: QPoint | None = None
    rotational_valid: bool = True
    rotational_applicable: bool = False
    rotational_geometry_complete: bool = False

    def __post_init__(self) -> None:
        errors = tuple(str(value) for value in self.errors)
        warnings = tuple(str(value) for value in self.warnings)
        if any(not value for value in (*errors, *warnings)):
            raise ValueError("Audit messages must be non-empty strings.")

        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Audit metric names must be non-empty strings.")
            number = float(value)
            if np.isnan(number):
                raise ValueError(f"Audit metric {name!r} cannot be NaN.")
            metrics[name] = number

        qpoints = tuple(_normalize_qpoint(value, name="report qpoint") for value in self.qpoints)
        duplicates = tuple(_normalize_translation(value) for value in self.duplicate_translations)
        missing = tuple(
            _normalize_translation(value)
            for value in self.missing_reciprocal_translations
        )
        minimum_q = self.minimum_eigenvalue_qpoint
        if minimum_q is not None:
            minimum_q = _normalize_qpoint(minimum_q, name="minimum eigenvalue qpoint")

        object.__setattr__(self, "schema_valid", bool(self.schema_valid))
        object.__setattr__(self, "mapping_valid", bool(self.mapping_valid))
        object.__setattr__(self, "reciprocal_valid", bool(self.reciprocal_valid))
        object.__setattr__(self, "asr_valid", bool(self.asr_valid))
        object.__setattr__(self, "hermiticity_valid", bool(self.hermiticity_valid))
        object.__setattr__(self, "stable", bool(self.stable))
        object.__setattr__(self, "rotational_valid", bool(self.rotational_valid))
        object.__setattr__(
            self,
            "rotational_applicable",
            bool(self.rotational_applicable),
        )
        object.__setattr__(
            self,
            "rotational_geometry_complete",
            bool(self.rotational_geometry_complete),
        )
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "qpoints", qpoints)
        object.__setattr__(self, "duplicate_translations", duplicates)
        object.__setattr__(self, "missing_reciprocal_translations", missing)
        object.__setattr__(self, "minimum_eigenvalue_qpoint", minimum_q)

    @property
    def passed(self) -> bool:
        return bool(
            self.schema_valid
            and self.mapping_valid
            and self.reciprocal_valid
            and self.asr_valid
            and self.rotational_valid
            and self.hermiticity_valid
            and self.stable
            and not self.errors
        )

    @property
    def valid(self) -> bool:
        """Alias for :attr:`passed`."""

        return self.passed

    @property
    def failures(self) -> tuple[str, ...]:
        """Alias used by other NEGFPy diagnostic reports."""

        return self.errors


def _normalize_translation(values: Any) -> Translation:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise TypeError("translation must contain three integers.") from exc
    if len(raw) != 3:
        raise ValueError("translation must contain exactly three integers.")
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        for value in raw
    ):
        raise TypeError("translation components must be integers.")
    return int(raw[0]), int(raw[1]), int(raw[2])


def _normalize_qpoint(values: Any, *, name: str) -> QPoint:
    try:
        qpoint = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain three real numbers.") from exc
    if qpoint.shape != (3,):
        raise ValueError(f"{name} must contain exactly three components.")
    if not np.all(np.isfinite(qpoint)):
        raise ValueError(f"{name} must contain only finite values.")
    return float(qpoint[0]), float(qpoint[1]), float(qpoint[2])


def _nonnegative_tolerance(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a non-negative finite real number.")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a non-negative finite real number.")
    return number


def _qpoint_samples(
    qpoints: Iterable[Iterable[float]] | None,
    *,
    random_q_count: int,
    random_seed: int,
) -> tuple[QPoint, ...]:
    if isinstance(random_q_count, (bool, np.bool_)) or not isinstance(
        random_q_count,
        (int, np.integer),
    ):
        raise TypeError("random_q_count must be an integer.")
    if int(random_q_count) < 0:
        raise ValueError("random_q_count must be non-negative.")
    if isinstance(random_seed, (bool, np.bool_)) or not isinstance(
        random_seed,
        (int, np.integer),
    ):
        raise TypeError("random_seed must be an integer.")

    # These samples include Gamma, BZ faces, and a generic incommensurate
    # point.  q is the phase coordinate used in exp(i q dot R), in radians.
    values: list[QPoint] = [
        (0.0, 0.0, 0.0),
        (float(np.pi), 0.0, 0.0),
        (0.0, float(np.pi), 0.0),
        (0.0, 0.0, float(np.pi)),
        (float(np.pi / 2.0), float(np.pi / 3.0), float(np.pi / 5.0)),
    ]
    if qpoints is not None:
        if isinstance(qpoints, (str, bytes)):
            raise TypeError("qpoints must be an iterable of three-component points.")
        values.extend(_normalize_qpoint(value, name="qpoint") for value in qpoints)

    rng = np.random.default_rng(int(random_seed))
    if int(random_q_count):
        random_values = rng.uniform(-np.pi, np.pi, size=(int(random_q_count), 3))
        values.extend(
            (float(value[0]), float(value[1]), float(value[2]))
            for value in random_values
        )

    # Preserve deterministic order while removing exact duplicate samples.
    return tuple(dict.fromkeys(values))


def _audit_mapping(
    ifc: IFCData,
    n_atoms: int | None,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, float],
) -> bool:
    lattice = ifc.lattice_vectors
    positions = ifc.atom_positions
    symbols = ifc.atom_symbols
    supplied = (lattice is not None, positions is not None, symbols is not None)
    valid = True

    if not isinstance(ifc.index_convention, str) or not ifc.index_convention.strip():
        errors.append("index_convention must be a non-empty string.")
        valid = False

    if not any(supplied):
        warnings.append(
            "Lattice vectors, atom positions, and symbols are absent; "
            "geometry-to-IFC mapping could not be independently checked."
        )
        metrics["geometry_metadata_complete"] = 0.0
        return valid
    if not all(supplied):
        errors.append(
            "Geometry metadata is incomplete: lattice_vectors, atom_positions, "
            "and atom_symbols must be supplied together."
        )
        metrics["geometry_metadata_complete"] = 0.0
        return False

    metrics["geometry_metadata_complete"] = 1.0

    try:
        lattice_raw = np.asarray(lattice)
        if np.iscomplexobj(lattice_raw) and np.any(np.abs(lattice_raw.imag) > 0.0):
            raise ValueError
        lattice_array = np.asarray(lattice_raw.real, dtype=float)
    except (TypeError, ValueError):
        lattice_array = np.empty((0, 0), dtype=float)
    if lattice_array.shape != (3, 3) or not np.all(np.isfinite(lattice_array)):
        errors.append("lattice_vectors must be a finite 3x3 array.")
        valid = False
    else:
        determinant = float(np.linalg.det(lattice_array))
        metrics["lattice_determinant_abs"] = abs(determinant)
        scale = max(1.0, float(np.linalg.norm(lattice_array, ord=2)) ** 3)
        if abs(determinant) <= 1.0e-12 * scale:
            errors.append("lattice_vectors are singular or numerically rank deficient.")
            valid = False

    try:
        position_raw = np.asarray(positions)
        if np.iscomplexobj(position_raw) and np.any(np.abs(position_raw.imag) > 0.0):
            raise ValueError
        position_array = np.asarray(position_raw.real, dtype=float)
    except (TypeError, ValueError):
        position_array = np.empty((0, 0), dtype=float)
    expected_positions = None if n_atoms is None else (n_atoms, 3)
    if (
        position_array.ndim != 2
        or position_array.shape[1:] != (3,)
        or not np.all(np.isfinite(position_array))
    ):
        errors.append("atom_positions must be a finite array with shape (n_atoms, 3).")
        valid = False
    elif expected_positions is not None and position_array.shape != expected_positions:
        errors.append(
            "atom_positions atom count does not match IFCData.masses "
            f"(got {position_array.shape[0]}, expected {n_atoms})."
        )
        valid = False

    try:
        if isinstance(symbols, (str, bytes)):
            raise TypeError
        symbol_values = tuple(symbols) if symbols is not None else ()
    except TypeError:
        symbol_values = ()
    if n_atoms is not None and len(symbol_values) != n_atoms:
        errors.append(
            "atom_symbols count does not match IFCData.masses "
            f"(got {len(symbol_values)}, expected {n_atoms})."
        )
        valid = False
    if any(not isinstance(value, str) or not value.strip() for value in symbol_values):
        errors.append("atom_symbols entries must be non-empty strings.")
        valid = False

    return valid


def _schema_and_merged_terms(
    ifc: IFCData,
    *,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, float],
) -> tuple[
    bool,
    Array | None,
    int | None,
    dict[Translation, Array],
    tuple[Translation, ...],
]:
    valid = True
    masses: Array | None = None
    n_atoms: int | None = None

    try:
        raw_masses = np.asarray(ifc.masses)
    except (TypeError, ValueError):
        raw_masses = np.empty(0, dtype=float)
    if np.iscomplexobj(raw_masses):
        errors.append("IFCData.masses must be real-valued.")
        valid = False
    try:
        candidate_masses = np.asarray(raw_masses.real, dtype=float)
    except (TypeError, ValueError):
        candidate_masses = np.empty(0, dtype=float)
    if candidate_masses.ndim != 1 or candidate_masses.size == 0:
        errors.append("IFCData.masses must be a non-empty one-dimensional array.")
        valid = False
    elif not np.all(np.isfinite(candidate_masses)):
        errors.append("IFCData.masses must contain only finite values.")
        valid = False
    elif np.any(candidate_masses <= 0.0):
        errors.append("All IFCData masses must be strictly positive.")
        valid = False
    else:
        masses = candidate_masses
        n_atoms = int(candidate_masses.size)
        metrics["n_atoms"] = float(n_atoms)
        metrics["minimum_mass"] = float(np.min(candidate_masses))

    dof = ifc.dof_per_atom
    if (
        isinstance(dof, (bool, np.bool_))
        or not isinstance(dof, (int, np.integer))
        or int(dof) <= 0
    ):
        errors.append("IFCData.dof_per_atom must be a strictly positive integer.")
        valid = False
        ndof = None
    else:
        dof = int(dof)
        ndof = None if n_atoms is None else n_atoms * dof
        metrics["dof_per_atom"] = float(dof)
        if ndof is not None:
            metrics["ndof"] = float(ndof)

    if not isinstance(ifc.units, str) or not ifc.units.strip():
        errors.append("IFCData.units must be a non-empty string.")
        valid = False
    elif ifc.units.strip().lower() == "unknown":
        warnings.append("IFCData.units is 'unknown'; physical unit provenance is unavailable.")
    if not isinstance(ifc.metadata, Mapping):
        errors.append("IFCData.metadata must be a mapping.")
        valid = False

    terms = ifc.terms if isinstance(ifc.terms, tuple) else ()
    if not isinstance(ifc.terms, tuple) or len(terms) == 0:
        errors.append("IFCData.terms must be a non-empty tuple of IFCTerm objects.")
        valid = False

    merged: dict[Translation, Array] = {}
    counts: dict[Translation, int] = {}
    for index, term in enumerate(terms):
        if not isinstance(term, IFCTerm):
            errors.append(f"IFCData.terms[{index}] is not an IFCTerm.")
            valid = False
            continue
        try:
            key = _normalize_translation((term.dx, term.dy, term.dz))
        except (TypeError, ValueError) as exc:
            errors.append(f"IFCData.terms[{index}] has an invalid translation: {exc}")
            valid = False
            continue
        try:
            block = np.asarray(term.block, dtype=np.complex128)
        except (TypeError, ValueError):
            errors.append(f"IFCData.terms[{index}].block is not a numeric array.")
            valid = False
            continue
        if ndof is None or block.shape != (ndof, ndof):
            expected = "(n_atoms*dof, n_atoms*dof)" if ndof is None else str((ndof, ndof))
            errors.append(
                f"IFCData.terms[{index}].block has shape {block.shape}, expected {expected}."
            )
            valid = False
            continue
        if not np.all(np.isfinite(block.real)) or not np.all(np.isfinite(block.imag)):
            errors.append(f"IFCData.terms[{index}].block contains non-finite values.")
            valid = False
            continue
        if key not in merged:
            merged[key] = np.zeros((ndof, ndof), dtype=np.complex128)
            counts[key] = 0
        merged[key] += block
        counts[key] += 1

    duplicates = tuple(sorted(key for key, count in counts.items() if count > 1))
    if duplicates:
        warnings.append(
            f"Merged {sum(counts[key] - 1 for key in duplicates)} duplicate IFC "
            f"term(s) across {len(duplicates)} translation(s) for audit calculations."
        )
    metrics["input_term_count"] = float(len(terms))
    metrics["merged_translation_count"] = float(len(merged))
    metrics["duplicate_translation_count"] = float(len(duplicates))

    if any(not np.all(np.isfinite(block)) for block in merged.values()):
        errors.append("Summing duplicate translations produced non-finite IFC values.")
        valid = False
    return valid, masses, n_atoms, merged, duplicates


def _audit_explicit_nyquist_grid(
    ifc: IFCData,
    merged: Mapping[Translation, Array],
    *,
    errors: list[str],
    metrics: dict[str, float],
) -> bool:
    """Validate a reader-declared explicit q2r reciprocal grid."""

    metadata = ifc.metadata if isinstance(ifc.metadata, Mapping) else {}
    expansion = metadata.get("nyquist_expansion")
    declaration_present = bool(
        expansion is not None
        or metadata.get("explicit_nyquist_signs") is not None
        or metadata.get("reciprocal_grid_complete") is not None
    )
    metrics["explicit_nyquist_grid_declared"] = float(declaration_present)
    metrics["explicit_nyquist_grid_complete"] = 0.0
    if not declaration_present:
        return True
    if not isinstance(expansion, Mapping):
        errors.append("Explicit Nyquist grid declaration lacks expansion metadata.")
        return False
    nr = metadata.get("nr")
    if not isinstance(nr, (list, tuple)) or len(nr) != 3:
        errors.append("Explicit Nyquist grid declaration requires three positive nr values.")
        return False
    try:
        repeats = tuple(int(value) for value in nr)
    except (TypeError, ValueError):
        errors.append("Explicit Nyquist grid declaration contains invalid nr values.")
        return False
    if any(value <= 0 for value in repeats):
        errors.append("Explicit Nyquist grid declaration contains non-positive nr values.")
        return False

    valid = True
    if expansion.get("schema") != "negfpy.qe_q2r_nyquist_expansion.v1":
        errors.append("Explicit Nyquist grid has an unsupported expansion schema.")
        valid = False
    if expansion.get("strategy") != "all_sign_variants_equal_weight":
        errors.append("Explicit Nyquist grid has an unsupported weighting strategy.")
        valid = False
    for name, value in (
        ("complete_reciprocal_grid", expansion.get("complete_reciprocal_grid")),
        ("source_mesh_equivalent", expansion.get("source_mesh_equivalent")),
        ("explicit_nyquist_signs", metadata.get("explicit_nyquist_signs")),
        ("reciprocal_grid_complete", metadata.get("reciprocal_grid_complete")),
    ):
        if value is not True:
            errors.append(f"Explicit Nyquist grid requires {name}=true.")
            valid = False

    expected_axes: list[tuple[int, ...]] = []
    for repeat in repeats:
        if repeat == 1:
            expected_axes.append((0,))
        else:
            expected_axes.append(tuple(range(-(repeat // 2), repeat // 2 + 1)))
    expected = set(product(*expected_axes))
    present = set(merged)
    metrics["explicit_nyquist_expected_support_count"] = float(len(expected))
    metrics["explicit_nyquist_present_support_count"] = float(len(present))
    if present != expected:
        errors.append(
            "Explicit Nyquist grid support is incomplete or contains undeclared translations."
        )
        valid = False
    try:
        declared_count = int(expansion.get("expanded_support_count"))
    except (TypeError, ValueError):
        declared_count = -1
    if declared_count != len(expected):
        errors.append("Explicit Nyquist grid expanded_support_count is inconsistent.")
        valid = False

    try:
        mesh_residual = float(expansion.get("source_mesh_max_abs"))
        mesh_threshold = float(expansion.get("source_mesh_threshold"))
    except (TypeError, ValueError):
        mesh_residual = float("inf")
        mesh_threshold = 0.0
    metrics["explicit_nyquist_source_mesh_max_abs"] = mesh_residual
    metrics["explicit_nyquist_source_mesh_threshold"] = mesh_threshold
    if (
        not np.isfinite(mesh_residual)
        or not np.isfinite(mesh_threshold)
        or mesh_threshold < 0.0
        or mesh_residual > mesh_threshold
    ):
        errors.append("Explicit Nyquist source-mesh equivalence evidence is invalid.")
        valid = False
    metrics["explicit_nyquist_grid_complete"] = float(valid)
    return valid


_LEVI_CIVITA = np.asarray(
    (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)),
        ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        ((0.0, 1.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
    ),
    dtype=np.float64,
)


def _rotational_geometry(
    ifc: IFCData,
    *,
    n_atoms: int,
    warnings: list[str],
    metrics: dict[str, float],
) -> tuple[bool, bool, Array | None, Array | None]:
    """Return applicability, geometry completeness, lattice, and positions.

    ``IFCData.atom_positions`` are fractional coordinates and lattice vectors
    are rows.  Cartesian positions are consequently ``fractional @ lattice``.
    Invalid geometry is already reported by :func:`_audit_mapping`; here it
    makes the rotational check inapplicable instead of creating a second
    independent failure.
    """

    lattice_array: Array | None = None
    fractional_positions: Array | None = None
    if ifc.lattice_vectors is not None and ifc.atom_positions is not None:
        try:
            lattice_raw = np.asarray(ifc.lattice_vectors)
            positions_raw = np.asarray(ifc.atom_positions)
            lattice_has_imaginary = bool(
                np.iscomplexobj(lattice_raw)
                and np.any(np.abs(lattice_raw.imag) > 0.0)
            )
            positions_have_imaginary = bool(
                np.iscomplexobj(positions_raw)
                and np.any(np.abs(positions_raw.imag) > 0.0)
            )
            if lattice_has_imaginary or positions_have_imaginary:
                raise ValueError
            lattice_candidate = np.asarray(lattice_raw.real, dtype=np.float64)
            positions_candidate = np.asarray(
                positions_raw.real,
                dtype=np.float64,
            )
        except (TypeError, ValueError):
            lattice_candidate = np.empty((0, 0), dtype=np.float64)
            positions_candidate = np.empty((0, 0), dtype=np.float64)

        if (
            lattice_candidate.shape == (3, 3)
            and positions_candidate.shape == (n_atoms, 3)
            and np.all(np.isfinite(lattice_candidate))
            and np.all(np.isfinite(positions_candidate))
        ):
            determinant = float(np.linalg.det(lattice_candidate))
            determinant_scale = max(
                1.0,
                float(np.linalg.norm(lattice_candidate, ord=2)) ** 3,
            )
            if abs(determinant) > 1.0e-12 * determinant_scale:
                lattice_array = lattice_candidate
                fractional_positions = positions_candidate

    geometry_complete = (
        lattice_array is not None and fractional_positions is not None
    )
    dof_is_cartesian = (
        isinstance(ifc.dof_per_atom, (int, np.integer))
        and not isinstance(ifc.dof_per_atom, (bool, np.bool_))
        and int(ifc.dof_per_atom) == 3
    )
    applicable = bool(dof_is_cartesian and geometry_complete)
    metrics["rotational_geometry_complete"] = float(geometry_complete)
    metrics["rotational_applicable"] = float(applicable)

    if not applicable:
        reasons: list[str] = []
        if not dof_is_cartesian:
            reasons.append("dof_per_atom is not 3 Cartesian components")
        if not geometry_complete:
            reasons.append(
                "complete finite fractional atom positions and a nonsingular "
                "row-vector lattice are unavailable"
            )
        warnings.append(
            "Rotational invariance check is not applicable: "
            + "; ".join(reasons)
            + "."
        )
        return False, geometry_complete, None, None

    assert lattice_array is not None
    assert fractional_positions is not None
    cartesian_positions = fractional_positions @ lattice_array
    return True, True, lattice_array, cartesian_positions


def _audit_rotational_invariance(
    ifc: IFCData,
    *,
    n_atoms: int,
    merged: Mapping[Translation, Array],
    rotational_atol: float,
    rotational_rtol: float,
    errors: list[str],
    warnings: list[str],
    metrics: dict[str, float],
) -> tuple[bool, bool, bool]:
    r"""Evaluate the harmonic rotational sum rule when geometry permits.

    The reported residual is

    .. math::

       T_{i\alpha\gamma} =
       \sum_{Rj\beta\delta}
       \Phi_{i\alpha,j\beta}(R)
       \epsilon_{\beta\gamma\delta}
       r_{ijR,\delta},

    with ``r_ijR = pos_j + R @ lattice - pos_i`` in Cartesian coordinates.
    The relative residual is normalized by the maximum componentwise sum of
    absolute contributions, so it is invariant under a common IFC or length
    rescaling.
    """

    metrics["rotational_max_abs"] = 0.0
    metrics["rotational_relative"] = 0.0
    metrics["rotational_scale"] = 0.0
    metrics["rotational_threshold"] = rotational_atol

    applicable, geometry_complete, lattice, positions = _rotational_geometry(
        ifc,
        n_atoms=n_atoms,
        warnings=warnings,
        metrics=metrics,
    )
    if not applicable:
        # Not applicable is a passing gate.  The separate applicability flag
        # prevents the zero sentinel metrics from being mistaken for a check.
        return True, False, geometry_complete

    assert lattice is not None
    assert positions is not None
    residual = np.zeros((n_atoms, 3, 3), dtype=np.complex128)
    absolute_scale = np.zeros((n_atoms, 3, 3), dtype=np.float64)
    for translation, block in merged.items():
        translation_cartesian = (
            np.asarray(translation, dtype=np.float64) @ lattice
        )
        displacement = (
            positions[None, :, :]
            + translation_cartesian[None, None, :]
            - positions[:, None, :]
        )
        phi = np.asarray(block, dtype=np.complex128).reshape(
            n_atoms,
            3,
            n_atoms,
            3,
        )
        residual += np.einsum(
            "iajb,bgd,ijd->iag",
            phi,
            _LEVI_CIVITA,
            displacement,
            optimize=True,
        )
        absolute_scale += np.einsum(
            "iajb,bgd,ijd->iag",
            np.abs(phi),
            np.abs(_LEVI_CIVITA),
            np.abs(displacement),
            optimize=True,
        )

    maximum = float(np.max(np.abs(residual)))
    scale = float(np.max(absolute_scale))
    relative = 0.0 if scale == 0.0 else maximum / scale
    threshold = rotational_atol + rotational_rtol * scale
    valid = maximum <= threshold
    metrics["rotational_max_abs"] = maximum
    metrics["rotational_relative"] = relative
    metrics["rotational_scale"] = scale
    metrics["rotational_threshold"] = threshold
    if not valid:
        errors.append(
            "Harmonic rotational invariance failed: "
            f"max torque residual {maximum:.6e}, allowed {threshold:.6e} "
            f"(relative residual {relative:.6e})."
        )
    return valid, True, True


def audit_ifc_data(
    ifc: IFCData,
    *,
    reciprocal_atol: float = 1.0e-10,
    reciprocal_rtol: float = 1.0e-8,
    asr_atol: float = 1.0e-8,
    rotational_atol: float = 1.0e-8,
    rotational_rtol: float = 1.0e-8,
    stability_atol: float = 1.0e-8,
    hermiticity_atol: float | None = None,
    qpoints: Iterable[Iterable[float]] | None = None,
    random_q_count: int = 4,
    random_seed: int = 0,
) -> IFCValidationReport:
    """Audit a harmonic IFC dataset without mutating or correcting it.

    ``qpoints`` are dimensionless phase coordinates in radians, consistent
    with ``exp(i * q dot R)``.  Deterministic Gamma/face/generic points are
    always included; optional seeded random points and caller-supplied points
    are appended.  The rotational gate uses fractional ``atom_positions`` and
    row-vector ``lattice_vectors``.  It is evaluated only for three Cartesian
    degrees of freedom per atom and valid complete geometry; otherwise the
    report marks it not applicable and emits a non-failing warning.
    """

    if not isinstance(ifc, IFCData):
        raise TypeError("ifc must be an IFCData instance.")
    reciprocal_atol = _nonnegative_tolerance(reciprocal_atol, name="reciprocal_atol")
    reciprocal_rtol = _nonnegative_tolerance(reciprocal_rtol, name="reciprocal_rtol")
    asr_atol = _nonnegative_tolerance(asr_atol, name="asr_atol")
    rotational_atol = _nonnegative_tolerance(
        rotational_atol,
        name="rotational_atol",
    )
    rotational_rtol = _nonnegative_tolerance(
        rotational_rtol,
        name="rotational_rtol",
    )
    stability_atol = _nonnegative_tolerance(stability_atol, name="stability_atol")
    if hermiticity_atol is None:
        hermiticity_atol = reciprocal_atol
    hermiticity_atol = _nonnegative_tolerance(
        hermiticity_atol,
        name="hermiticity_atol",
    )
    sampled_qpoints = _qpoint_samples(
        qpoints,
        random_q_count=random_q_count,
        random_seed=random_seed,
    )

    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {
        "reciprocal_atol": reciprocal_atol,
        "reciprocal_rtol": reciprocal_rtol,
        "asr_atol": asr_atol,
        "rotational_atol": rotational_atol,
        "rotational_rtol": rotational_rtol,
        "rotational_geometry_complete": 0.0,
        "rotational_applicable": 0.0,
        "rotational_max_abs": 0.0,
        "rotational_relative": 0.0,
        "rotational_scale": 0.0,
        "rotational_threshold": rotational_atol,
        "stability_atol": stability_atol,
        "hermiticity_atol": hermiticity_atol,
        "qpoint_count": float(len(sampled_qpoints)),
    }
    schema_valid, masses, n_atoms, merged, duplicates = _schema_and_merged_terms(
        ifc,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
    if schema_valid and merged:
        schema_valid = bool(
            _audit_explicit_nyquist_grid(
                ifc,
                merged,
                errors=errors,
                metrics=metrics,
            )
            and schema_valid
        )
    else:
        metrics["explicit_nyquist_grid_declared"] = 0.0
        metrics["explicit_nyquist_grid_complete"] = 0.0
    mapping_valid = _audit_mapping(
        ifc,
        n_atoms,
        errors=errors,
        warnings=warnings,
        metrics=metrics,
    )
    # A malformed block/mass schema makes all numerical physics checks
    # undefined.  Return one coherent report rather than raising or emitting a
    # cascade of linear-algebra errors.
    if not schema_valid or masses is None or not merged:
        errors.append(
            "Reciprocity, ASR, rotational invariance, and q-space checks "
            "were skipped due to schema errors."
        )
        return IFCValidationReport(
            schema_valid=False,
            mapping_valid=mapping_valid,
            reciprocal_valid=False,
            asr_valid=False,
            hermiticity_valid=False,
            stable=False,
            errors=tuple(errors),
            warnings=tuple(warnings),
            metrics=metrics,
            qpoints=sampled_qpoints,
            duplicate_translations=duplicates,
            rotational_valid=True,
            rotational_applicable=False,
            rotational_geometry_complete=False,
        )

    block_scale = max(float(np.max(np.abs(block))) for block in merged.values())
    block_denominator = max(block_scale, np.finfo(float).tiny)
    reciprocal_threshold = reciprocal_atol + reciprocal_rtol * block_scale
    reciprocal_max = 0.0
    missing: list[Translation] = []
    visited: set[Translation] = set()
    for key in sorted(merged):
        if key in visited:
            continue
        partner = (-key[0], -key[1], -key[2])
        visited.add(key)
        visited.add(partner)
        if partner not in merged:
            residual = float(np.max(np.abs(merged[key])))
            reciprocal_max = max(reciprocal_max, residual)
            if residual > reciprocal_threshold:
                missing.append(key)
            continue
        residual = float(
            np.max(np.abs(merged[key] - merged[partner].conj().T))
        )
        reciprocal_max = max(reciprocal_max, residual)

    reciprocal_valid = reciprocal_max <= reciprocal_threshold and not missing
    metrics["ifc_block_scale"] = block_scale
    metrics["reciprocal_max_abs"] = reciprocal_max
    metrics["reciprocal_relative"] = reciprocal_max / block_denominator
    metrics["missing_reciprocal_count"] = float(len(missing))
    if not reciprocal_valid:
        errors.append(
            "IFC reciprocal relation Phi(R)=Phi(-R)^dagger failed: "
            f"max residual {reciprocal_max:.6e}, allowed {reciprocal_threshold:.6e}."
        )
        if missing:
            errors.append(
                f"{len(missing)} nonzero translation(s) lack a reciprocal partner."
            )

    ndof = masses.size * int(ifc.dof_per_atom)
    phi_sum = np.zeros((ndof, ndof), dtype=np.complex128)
    for block in merged.values():
        phi_sum += block
    asr = np.zeros((ndof, int(ifc.dof_per_atom)), dtype=np.complex128)
    for component in range(int(ifc.dof_per_atom)):
        columns = np.arange(component, ndof, int(ifc.dof_per_atom))
        asr[:, component] = np.sum(phi_sum[:, columns], axis=1)
    asr_max = float(np.max(np.abs(asr)))
    asr_valid = asr_max <= asr_atol
    metrics["asr_max_abs"] = asr_max
    metrics["asr_relative"] = asr_max / block_denominator
    metrics["phi_sum_matrix_max_abs"] = float(np.max(np.abs(phi_sum)))
    if not asr_valid:
        errors.append(
            "Translational acoustic sum rule failed: "
            f"max residual {asr_max:.6e}, allowed {asr_atol:.6e}."
        )

    rotational_valid, rotational_applicable, rotational_geometry_complete = (
        _audit_rotational_invariance(
            ifc,
            n_atoms=int(n_atoms),
            merged=merged,
            rotational_atol=rotational_atol,
            rotational_rtol=rotational_rtol,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
        )
    )

    repeated_masses = np.repeat(masses, int(ifc.dof_per_atom))
    inverse_sqrt_mass = 1.0 / np.sqrt(repeated_masses)
    hermiticity_max = 0.0
    dynamical_scale = 1.0
    minimum_eigenvalue = float("inf")
    maximum_eigenvalue = float("-inf")
    minimum_qpoint: QPoint | None = None
    unstable_count = 0
    for qpoint in sampled_qpoints:
        phi_q = np.zeros((ndof, ndof), dtype=np.complex128)
        for translation, block in merged.items():
            phase = np.exp(1j * np.dot(qpoint, translation))
            phi_q += block * phase
        dynamical = (
            inverse_sqrt_mass[:, None]
            * phi_q
            * inverse_sqrt_mass[None, :]
        )
        scale_q = max(1.0, float(np.max(np.abs(dynamical))))
        dynamical_scale = max(dynamical_scale, scale_q)
        residual = float(np.max(np.abs(dynamical - dynamical.conj().T)))
        hermiticity_max = max(hermiticity_max, residual)

        # Stability is meaningful only for a Hermitian dynamical matrix.  Its
        # Hermitian part provides a deterministic diagnostic even when the
        # separate Hermiticity check fails.
        hermitian_part = 0.5 * (dynamical + dynamical.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian_part)
        local_min = float(eigenvalues[0])
        local_max = float(eigenvalues[-1])
        if local_min < minimum_eigenvalue:
            minimum_eigenvalue = local_min
            minimum_qpoint = qpoint
        maximum_eigenvalue = max(maximum_eigenvalue, local_max)
        if local_min < -stability_atol:
            unstable_count += 1

    hermiticity_valid = hermiticity_max <= hermiticity_atol
    stable = unstable_count == 0
    metrics["dynamical_matrix_scale"] = dynamical_scale
    metrics["dynamical_hermiticity_max_abs"] = hermiticity_max
    metrics["dynamical_hermiticity_relative"] = hermiticity_max / dynamical_scale
    metrics["minimum_eigenvalue"] = minimum_eigenvalue
    metrics["maximum_eigenvalue"] = maximum_eigenvalue
    metrics["unstable_qpoint_count"] = float(unstable_count)
    if not hermiticity_valid:
        errors.append(
            "D(q) Hermiticity failed on the sampled q points: "
            f"max residual {hermiticity_max:.6e}, allowed {hermiticity_atol:.6e}."
        )
    if not stable:
        errors.append(
            "D(q) has unstable sampled modes: "
            f"minimum eigenvalue {minimum_eigenvalue:.6e}, "
            f"allowed floor {-stability_atol:.6e}."
        )

    return IFCValidationReport(
        schema_valid=schema_valid,
        mapping_valid=mapping_valid,
        reciprocal_valid=reciprocal_valid,
        asr_valid=asr_valid,
        hermiticity_valid=hermiticity_valid,
        stable=stable,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
        qpoints=sampled_qpoints,
        duplicate_translations=duplicates,
        missing_reciprocal_translations=tuple(missing),
        minimum_eigenvalue_qpoint=minimum_qpoint,
        rotational_valid=rotational_valid,
        rotational_applicable=rotational_applicable,
        rotational_geometry_complete=rotational_geometry_complete,
    )


__all__ = ["IFCValidationReport", "audit_ifc_data"]
