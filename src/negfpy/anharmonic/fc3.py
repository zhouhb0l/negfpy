"""Immutable third-order force-constant data and validation.

This module deliberately separates four operations that are often mixed
together in anharmonic transport codes:

* reading raw third-order force constants,
* auditing their schema and permutation symmetry,
* converting physical units, and
* mass normalizing the tensor.

No operation in this module symmetrizes or otherwise repairs an input tensor.
Phono3py's full and compact array layouts are both retained without expanding
the compact representation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import permutations
from types import MappingProxyType
from typing import Any, Literal

import numpy as np


Array = np.ndarray
Metadata = Mapping[str, Any]

# CODATA 2018 exact/conventional constants used elsewhere in NEGFPy.
ELECTRON_VOLT_J = 1.602176634e-19
ANGSTROM_M = 1.0e-10
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
EV_ANGSTROM3_TO_J_M3 = ELECTRON_VOLT_J / ANGSTROM_M**3

_EV_ANGSTROM3_ALIASES = frozenset(
    {
        "ev/angstrom^3",
        "ev/angstrom3",
        "ev/a^3",
        "ev/a3",
        "ev/å^3",
        "ev/å3",
    }
)
_SI_FC3_ALIASES = frozenset(
    {
        "j/m^3",
        "j/m3",
        "j m^-3",
        "j*m^-3",
    }
)
_AMU_ALIASES = frozenset({"amu", "u", "dalton", "da"})
_KG_ALIASES = frozenset({"kg", "kilogram", "kilograms"})


def _readonly_copy(values: Any) -> Array:
    array = np.array(values, copy=True, order="C")
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
    """Recursively detach and freeze provenance values."""

    if isinstance(value, np.ndarray):
        return _readonly_copy(value)
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


def _canonical_unit_text(units: str) -> str:
    return (
        units.strip()
        .lower()
        .replace("ångström", "angstrom")
        .replace("ångstrom", "angstrom")
        .replace("angström", "angstrom")
        .replace(" ", "")
    )


def fc3_force_constant_to_si_factor(units: str) -> float:
    """Return the multiplier that converts an FC3 value to ``J/m^3``.

    Phono3py writes ``fc3.hdf5`` in eV/Angstrom^3 independently of the
    calculator used to generate the forces. SI-valued input is also accepted
    so that conversion can remain explicit but idempotent.
    """

    if not isinstance(units, str) or not units.strip():
        raise ValueError("FC3 units must be a non-empty string.")
    normalized = _canonical_unit_text(units)
    if normalized in {value.replace(" ", "") for value in _EV_ANGSTROM3_ALIASES}:
        return EV_ANGSTROM3_TO_J_M3
    if normalized in {value.replace(" ", "") for value in _SI_FC3_ALIASES}:
        return 1.0
    raise ValueError(
        f"Unsupported FC3 units {units!r}; supported units are "
        "'eV/angstrom^3' and 'J/m^3'."
    )


def mass_to_si_factor(units: str) -> float:
    """Return the multiplier that converts an atomic mass to kilograms."""

    if not isinstance(units, str) or not units.strip():
        raise ValueError("Mass units must be a non-empty string.")
    normalized = _canonical_unit_text(units)
    if normalized in _AMU_ALIASES:
        return ATOMIC_MASS_UNIT_KG
    if normalized in _KG_ALIASES:
        return 1.0
    raise ValueError(
        f"Unsupported mass units {units!r}; supported units are 'amu' and 'kg'."
    )


def convert_fc3_values_to_si(values: Any, units: str) -> Array:
    """Convert third-order force constants to a new float64 SI array."""

    array = np.asarray(values, dtype=np.float64)
    return np.array(
        array * fc3_force_constant_to_si_factor(units),
        dtype=np.float64,
        order="C",
        copy=True,
    )


@dataclass(frozen=True, slots=True, eq=False)
class FC3Data:
    """Raw full or compact phono3py-style third-order force constants.

    Parameters
    ----------
    values
        Full ``(Ns, Ns, Ns, 3, 3, 3)`` or compact
        ``(Np, Ns, Ns, 3, 3, 3)`` tensor. The input is copied without
        symmetrization or dtype coercion.
    masses
        One mass per supercell atom. Phono3py workflows normally use amu.
    p2s_map
        Primitive atom to representative supercell atom map. It is required
        for a compact tensor.
    s2p_map
        Optional phonopy-style map from every supercell atom to a
        representative supercell atom (i.e. its values belong to
        ``p2s_map``).
    """

    values: Array
    masses: Array
    units: str = "eV/angstrom^3"
    compact: bool = False
    p2s_map: Array | None = None
    s2p_map: Array | None = None
    provenance: Metadata = field(default_factory=dict)
    mass_units: str = "amu"

    def __post_init__(self) -> None:
        if not isinstance(self.compact, (bool, np.bool_)):
            raise TypeError("compact must be a boolean.")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string.")
        if not isinstance(self.mass_units, str) or not self.mass_units.strip():
            raise ValueError("mass_units must be a non-empty string.")

        object.__setattr__(self, "values", _readonly_copy(self.values))
        object.__setattr__(self, "masses", _readonly_copy(self.masses))
        object.__setattr__(self, "compact", bool(self.compact))
        object.__setattr__(
            self,
            "p2s_map",
            None if self.p2s_map is None else _readonly_copy(self.p2s_map),
        )
        object.__setattr__(
            self,
            "s2p_map",
            None if self.s2p_map is None else _readonly_copy(self.s2p_map),
        )
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def fc3(self) -> Array:
        """Alias matching phono3py's name for the tensor."""

        return self.values

    @property
    def n_supercell_atoms(self) -> int | None:
        if self.values.ndim != 6:
            return None
        return int(self.values.shape[1])

    @property
    def n_primitive_atoms(self) -> int | None:
        if self.values.ndim != 6:
            return None
        return int(self.values.shape[0] if self.compact else self.values.shape[1])


@dataclass(frozen=True, slots=True, eq=False)
class MassNormalizedFC3:
    """A distinct container for a mass-normalized cubic tensor."""

    values: Array
    masses: Array
    units: str
    mass_units: str
    compact: bool
    p2s_map: Array | None = None
    s2p_map: Array | None = None
    provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.array(self.values, dtype=np.float64, order="C", copy=True)
        masses = np.array(self.masses, dtype=np.float64, order="C", copy=True)
        if not np.all(np.isfinite(values)):
            raise ValueError("Mass-normalized FC3 values must be finite.")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("Mass-normalization masses must be finite and positive.")
        values.setflags(write=False)
        masses.setflags(write=False)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "compact", bool(self.compact))
        object.__setattr__(
            self,
            "p2s_map",
            None if self.p2s_map is None else _readonly_copy(self.p2s_map),
        )
        object.__setattr__(
            self,
            "s2p_map",
            None if self.s2p_map is None else _readonly_copy(self.s2p_map),
        )
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def fc3(self) -> Array:
        return self.values


@dataclass(frozen=True, slots=True)
class FC3ValidationReport:
    """Non-mutating validation result for :class:`FC3Data`."""

    dimension_valid: bool
    finite_valid: bool
    masses_valid: bool
    units_valid: bool
    mapping_valid: bool
    permutation_valid: bool
    permutation_complete: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    checked_permutations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        errors = tuple(str(value) for value in self.errors)
        warnings = tuple(str(value) for value in self.warnings)
        if any(not message for message in (*errors, *warnings)):
            raise ValueError("Validation messages must be non-empty strings.")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("Validation metric names must be non-empty strings.")
            number = float(value)
            if np.isnan(number):
                raise ValueError(f"Validation metric {name!r} cannot be NaN.")
            metrics[name] = number
        object.__setattr__(self, "errors", errors)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "checked_permutations",
            tuple(str(value) for value in self.checked_permutations),
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.dimension_valid
            and self.finite_valid
            and self.masses_valid
            and self.units_valid
            and self.mapping_valid
            and self.permutation_valid
            and not self.errors
        )

    @property
    def valid(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.errors

    @property
    def mass_valid(self) -> bool:
        return self.masses_valid

    @property
    def map_valid(self) -> bool:
        return self.mapping_valid


def _finite_numeric(array: Array) -> bool:
    if array.dtype.kind not in "biufc":
        return False
    return bool(np.all(np.isfinite(array)))


def _validate_map_array(
    values: Array | None,
    *,
    name: str,
    expected_length: int | None,
    upper_bound: int | None,
    errors: list[str],
) -> tuple[bool, Array | None]:
    if values is None:
        return True, None
    array = np.asarray(values)
    valid = True
    if array.ndim != 1:
        errors.append(f"{name} must be a one-dimensional integer array.")
        valid = False
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        errors.append(f"{name} must contain integers without implicit conversion.")
        valid = False
    if expected_length is not None and (
        array.ndim != 1 or array.size != expected_length
    ):
        errors.append(f"{name} must contain exactly {expected_length} entries.")
        valid = False
    if (
        valid
        and upper_bound is not None
        and (np.any(array < 0) or np.any(array >= upper_bound))
    ):
        errors.append(f"{name} contains an atom index outside [0, {upper_bound}).")
        valid = False
    return valid, array if valid else None


def _permutation_residual(
    values: Array,
    *,
    compact: bool,
    p2s_map: Array | None,
) -> tuple[float, int, float, bool, tuple[str, ...]]:
    """Return maximum residual and accessible compact-check coverage."""

    maximum = 0.0
    compared = 0
    checked: list[str] = []

    def compare(left: Array, right: Array, name: str) -> None:
        nonlocal maximum, compared
        maximum = max(maximum, float(np.max(np.abs(left - right))))
        compared += int(left.size)
        checked.append(name)

    if not compact:
        for axes in permutations(range(3)):
            if axes == (0, 1, 2):
                continue
            permutation_axes = axes + tuple(3 + axis for axis in axes)
            compare(
                values,
                np.transpose(values, permutation_axes),
                "atoms_" + "".join(str(value) for value in axes),
            )
        return maximum, compared, 1.0, True, tuple(checked)

    # Swapping the two supercell-sized axes is fully represented.
    compare(
        values,
        np.transpose(values, (0, 2, 1, 3, 5, 4)),
        "swap_atoms_1_2",
    )

    assert p2s_map is not None
    representatives = np.asarray(p2s_map, dtype=np.intp)

    # A compact array can directly test a permutation that moves an atom onto
    # the first axis only when that atom is one of the p2s representatives.
    subset_01 = values[:, representatives, :, :, :, :]
    compare(
        subset_01,
        np.transpose(subset_01, (1, 0, 2, 4, 3, 5)),
        "swap_atoms_0_1_representatives",
    )
    subset_02 = values[:, :, representatives, :, :, :]
    compare(
        subset_02,
        np.transpose(subset_02, (2, 1, 0, 5, 4, 3)),
        "swap_atoms_0_2_representatives",
    )

    baseline = max(3 * values.size, 1)
    coverage = min(float(compared / baseline), 1.0)
    ns = values.shape[1]
    complete = bool(
        representatives.size == ns
        and np.array_equal(np.sort(representatives), np.arange(ns))
    )
    return maximum, compared, coverage, complete, tuple(checked)


def validate_fc3(
    data: FC3Data,
    *,
    permutation_atol: float = 1.0e-10,
    permutation_rtol: float = 1.0e-8,
) -> FC3ValidationReport:
    """Audit raw FC3 data without changing or symmetrizing it."""

    if not isinstance(data, FC3Data):
        raise TypeError("data must be an FC3Data instance.")
    for name, tolerance in (
        ("permutation_atol", permutation_atol),
        ("permutation_rtol", permutation_rtol),
    ):
        if (
            isinstance(tolerance, (bool, np.bool_))
            or not np.isfinite(float(tolerance))
            or float(tolerance) < 0.0
        ):
            raise ValueError(f"{name} must be a non-negative finite real number.")

    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {
        "permutation_atol": float(permutation_atol),
        "permutation_rtol": float(permutation_rtol),
    }
    values = np.asarray(data.values)

    dimension_valid = True
    ns: int | None = None
    npatom: int | None = None
    if values.ndim != 6:
        errors.append(
            "FC3 values must have six dimensions: three atom and three Cartesian axes."
        )
        dimension_valid = False
    else:
        npatom = int(values.shape[0])
        ns = int(values.shape[1])
        metrics["n_primitive_atoms"] = float(npatom if data.compact else ns)
        metrics["n_supercell_atoms"] = float(ns)
        metrics["element_count"] = float(values.size)
        if min(values.shape) < 1:
            errors.append("FC3 values cannot contain an empty dimension.")
            dimension_valid = False
        if values.shape[-3:] != (3, 3, 3):
            errors.append("The final three FC3 dimensions must each have size three.")
            dimension_valid = False
        if values.shape[1] != values.shape[2]:
            errors.append("The second and third atom dimensions must have equal size.")
            dimension_valid = False
        if data.compact:
            if npatom > ns:
                errors.append(
                    "A compact FC3 first atom dimension cannot exceed the supercell size."
                )
                dimension_valid = False
        elif values.shape[:3] != (ns, ns, ns):
            errors.append(
                "A full FC3 tensor must have shape (Ns, Ns, Ns, 3, 3, 3)."
            )
            dimension_valid = False

    finite_valid = _finite_numeric(values) and not np.iscomplexobj(values)
    if not finite_valid:
        if np.iscomplexobj(values):
            errors.append("FC3 values must be real.")
        else:
            errors.append("FC3 values must be numeric and finite.")

    masses = np.asarray(data.masses)
    masses_valid = True
    if masses.ndim != 1:
        errors.append("masses must be a one-dimensional array.")
        masses_valid = False

    units_valid = True
    try:
        force_factor = fc3_force_constant_to_si_factor(data.units)
    except ValueError as exc:
        errors.append(str(exc))
        units_valid = False
    else:
        metrics["force_constant_to_si_factor"] = force_factor
    try:
        mass_factor = mass_to_si_factor(data.mass_units)
    except ValueError as exc:
        errors.append(str(exc))
        units_valid = False
    else:
        metrics["mass_to_si_factor"] = mass_factor
    if masses.dtype.kind not in "iuf" or masses.dtype.kind == "b":
        errors.append("masses must be real numeric values.")
        masses_valid = False
    elif not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        errors.append("masses must contain only finite, strictly positive values.")
        masses_valid = False
    if ns is not None and (masses.ndim != 1 or masses.size != ns):
        errors.append(f"masses must contain one value for each of the {ns} supercell atoms.")
        masses_valid = False

    mapping_valid = True
    p2s_expected = npatom if data.compact and dimension_valid else None
    p2s_ok, p2s = _validate_map_array(
        data.p2s_map,
        name="p2s_map",
        expected_length=p2s_expected,
        upper_bound=ns,
        errors=errors,
    )
    mapping_valid &= p2s_ok
    if data.compact and data.p2s_map is None:
        errors.append("p2s_map is required for compact FC3 data.")
        mapping_valid = False
    if p2s is not None and np.unique(p2s).size != p2s.size:
        errors.append("p2s_map must contain unique representative supercell atoms.")
        mapping_valid = False

    s2p_ok, s2p = _validate_map_array(
        data.s2p_map,
        name="s2p_map",
        expected_length=ns if data.s2p_map is not None else None,
        upper_bound=ns,
        errors=errors,
    )
    mapping_valid &= s2p_ok
    if s2p is not None:
        if p2s is None:
            errors.append("s2p_map cannot be validated without p2s_map.")
            mapping_valid = False
        else:
            representatives = set(int(value) for value in p2s)
            if any(int(value) not in representatives for value in s2p):
                errors.append(
                    "Every s2p_map value must be a representative supercell "
                    "index present in p2s_map."
                )
                mapping_valid = False
            elif any(int(s2p[index]) != int(index) for index in p2s):
                errors.append("Every p2s representative must map to itself in s2p_map.")
                mapping_valid = False
    elif data.compact:
        warnings.append(
            "s2p_map is absent; this is valid for fc3.hdf5 ingestion, but "
            "later translational device mapping will require crystal mapping data."
        )

    permutation_valid = False
    permutation_complete = False
    checked_permutations: tuple[str, ...] = ()
    if dimension_valid and finite_valid and (not data.compact or mapping_valid):
        scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
        (
            permutation_max,
            comparison_count,
            coverage,
            permutation_complete,
            checked_permutations,
        ) = _permutation_residual(
            np.asarray(values, dtype=np.float64),
            compact=data.compact,
            p2s_map=p2s,
        )
        threshold = float(permutation_atol) + float(permutation_rtol) * scale
        permutation_valid = permutation_max <= threshold
        metrics.update(
            {
                "fc3_scale": scale,
                "permutation_max_abs": permutation_max,
                "permutation_relative": permutation_max / scale,
                "permutation_threshold": threshold,
                "permutation_comparison_count": float(comparison_count),
                "permutation_coverage": coverage,
            }
        )
        if not permutation_valid:
            errors.append(
                "FC3 permutation symmetry failed: maximum residual "
                f"{permutation_max:.6e}, allowed {threshold:.6e}."
            )
        if data.compact and not permutation_complete:
            warnings.append(
                "Compact FC3 permutation validation is partial: exchanges "
                "requiring a non-representative atom on the first axis need "
                "lattice-translation metadata and were not inferred."
            )
    else:
        warnings.append(
            "Permutation checks were skipped because FC3 dimensions, values, "
            "or compact mappings are invalid."
        )

    return FC3ValidationReport(
        dimension_valid=dimension_valid,
        finite_valid=finite_valid,
        masses_valid=masses_valid,
        units_valid=units_valid,
        mapping_valid=mapping_valid,
        permutation_valid=permutation_valid,
        permutation_complete=permutation_complete,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
        checked_permutations=checked_permutations,
    )


def mass_normalize_fc3(
    data: FC3Data,
    *,
    output_units: Literal["si", "source"] = "si",
) -> MassNormalizedFC3:
    """Divide FC3 by ``sqrt(M_i M_j M_k)`` without modifying the source.

    In compact layout, first-axis atom ``i`` uses
    ``masses[p2s_map[i]]`` while the other two axes use their direct
    supercell-atom indices.
    """

    if not isinstance(data, FC3Data):
        raise TypeError("data must be an FC3Data instance.")
    if output_units not in {"si", "source"}:
        raise ValueError("output_units must be either 'si' or 'source'.")

    report = validate_fc3(data)
    required = (
        report.dimension_valid,
        report.finite_valid,
        report.masses_valid,
        report.units_valid,
        report.mapping_valid,
    )
    if not all(required):
        details = "; ".join(report.errors) or "invalid FC3 schema"
        raise ValueError(f"Cannot mass normalize invalid FC3 data: {details}")

    values = np.asarray(data.values, dtype=np.float64)
    masses = np.asarray(data.masses, dtype=np.float64)
    force_factor = 1.0
    mass_factor = 1.0
    if output_units == "si":
        force_factor = fc3_force_constant_to_si_factor(data.units)
        mass_factor = mass_to_si_factor(data.mass_units)
        output_unit_text = "J/m^3/kg^(3/2)"
        output_mass_units = "kg"
    else:
        output_unit_text = f"({data.units})/({data.mass_units})^(3/2)"
        output_mass_units = data.mass_units

    converted_values = values * force_factor
    converted_masses = masses * mass_factor
    if data.compact:
        assert data.p2s_map is not None
        first_masses = converted_masses[np.asarray(data.p2s_map, dtype=np.intp)]
    else:
        first_masses = converted_masses
    denominator = np.sqrt(
        first_masses[:, None, None]
        * converted_masses[None, :, None]
        * converted_masses[None, None, :]
    )
    normalized = converted_values / denominator[..., None, None, None]

    provenance = dict(data.provenance)
    provenance["mass_normalization"] = {
        "formula": "Phi_ijk/sqrt(M_i*M_j*M_k)",
        "compact_first_axis_uses_p2s_map": bool(data.compact),
        "source_fc3_units": data.units,
        "source_mass_units": data.mass_units,
        "force_constant_to_si_factor": float(force_factor),
        "mass_to_si_factor": float(mass_factor),
        "output_units": output_unit_text,
        "source_permutation_valid": report.permutation_valid,
        "source_permutation_complete": report.permutation_complete,
    }
    return MassNormalizedFC3(
        values=normalized,
        masses=converted_masses,
        units=output_unit_text,
        mass_units=output_mass_units,
        compact=data.compact,
        p2s_map=data.p2s_map,
        s2p_map=data.s2p_map,
        provenance=provenance,
    )


__all__ = [
    "ANGSTROM_M",
    "ATOMIC_MASS_UNIT_KG",
    "ELECTRON_VOLT_J",
    "EV_ANGSTROM3_TO_J_M3",
    "FC3Data",
    "FC3ValidationReport",
    "MassNormalizedFC3",
    "convert_fc3_values_to_si",
    "fc3_force_constant_to_si_factor",
    "mass_normalize_fc3",
    "mass_to_si_factor",
    "validate_fc3",
]
