"""Immutable second-order force-constant ingestion data.

The types in this module represent raw phono3py-style FC2 arrays.  They do
not symmetrize, remap, convert, or otherwise repair force constants.  Schema
validation is deliberately separate and non-mutating so that source data and
all validation findings remain auditable.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


Array = np.ndarray
Metadata = Mapping[str, Any]

ELECTRON_VOLT_J = 1.602176634e-19
ANGSTROM_M = 1.0e-10
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
EV_ANGSTROM2_TO_J_M2 = ELECTRON_VOLT_J / ANGSTROM_M**2

_EV_ANGSTROM2_ALIASES = frozenset(
    {
        "ev/angstrom^2",
        "ev/angstrom2",
        "ev/a^2",
        "ev/a2",
        "ev/å^2",
        "ev/å2",
    }
)
_SI_FC2_ALIASES = frozenset(
    {
        "j/m^2",
        "j/m2",
        "j m^-2",
        "j*m^-2",
    }
)
_AMU_ALIASES = frozenset({"amu", "u", "dalton", "da"})
_KG_ALIASES = frozenset({"kg", "kilogram", "kilograms"})


def _readonly_copy(values: Any) -> Array:
    array = np.array(values, copy=True, order="C")
    array.setflags(write=False)
    return array


def _freeze_value(value: Any) -> Any:
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


def fc2_force_constant_to_si_factor(units: str) -> float:
    """Return the multiplier converting FC2 values to ``J/m^2``."""

    if not isinstance(units, str) or not units.strip():
        raise ValueError("FC2 units must be a non-empty string.")
    normalized = _canonical_unit_text(units)
    supported_ev = {
        value.replace(" ", "") for value in _EV_ANGSTROM2_ALIASES
    }
    supported_si = {value.replace(" ", "") for value in _SI_FC2_ALIASES}
    if normalized in supported_ev:
        return EV_ANGSTROM2_TO_J_M2
    if normalized in supported_si:
        return 1.0
    raise ValueError(
        f"Unsupported FC2 units {units!r}; supported units are "
        "'eV/angstrom^2' and 'J/m^2'."
    )


def mass_to_si_factor(units: str) -> float:
    """Return the multiplier converting atomic masses to kilograms."""

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


@dataclass(frozen=True, slots=True, eq=False)
class FC2Data:
    """Raw full or compact phono3py-style second-order force constants.

    ``values`` is retained exactly as supplied, including its dtype.  A full
    tensor has shape ``(Ns, Ns, 3, 3)`` and a compact tensor has shape
    ``(Np, Ns, 3, 3)`` with ``Np < Ns``.  Equal first and second atom
    dimensions denote the full/equivalent representation.
    """

    values: Array
    masses: Array
    units: str = "eV/angstrom^2"
    compact: bool = False
    p2s_map: Array | None = None
    s2p_map: Array | None = None
    version: str | None = None
    cutoff: float | None = None
    provenance: Metadata = field(default_factory=dict)
    mass_units: str = "amu"

    def __post_init__(self) -> None:
        if not isinstance(self.compact, (bool, np.bool_)):
            raise TypeError("compact must be a boolean.")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string.")
        if not isinstance(self.mass_units, str) or not self.mass_units.strip():
            raise ValueError("mass_units must be a non-empty string.")
        if self.version is not None and not isinstance(self.version, str):
            raise TypeError("version must be a string or None.")

        cutoff = self.cutoff
        if cutoff is not None:
            if isinstance(cutoff, (bool, np.bool_)):
                raise TypeError("cutoff must be a non-negative finite real number.")
            try:
                cutoff = float(cutoff)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "cutoff must be a non-negative finite real number."
                ) from exc
            if not np.isfinite(cutoff) or cutoff < 0.0:
                raise ValueError(
                    "cutoff must be a non-negative finite real number."
                )

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
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    @property
    def fc2(self) -> Array:
        """Alias matching phono3py's common name for the FC2 tensor."""

        return self.values

    @property
    def n_supercell_atoms(self) -> int | None:
        if self.values.ndim != 4:
            return None
        return int(self.values.shape[1])

    @property
    def n_primitive_atoms(self) -> int | None:
        if self.p2s_map is not None and self.p2s_map.ndim == 1:
            return int(self.p2s_map.size)
        if self.compact and self.values.ndim == 4:
            return int(self.values.shape[0])
        return None


@dataclass(frozen=True, slots=True)
class FC2ValidationReport:
    """Non-mutating schema validation result for :class:`FC2Data`."""

    dimension_valid: bool
    finite_valid: bool
    masses_valid: bool
    units_valid: bool
    mapping_valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)

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

    @property
    def passed(self) -> bool:
        return bool(
            self.dimension_valid
            and self.finite_valid
            and self.masses_valid
            and self.units_valid
            and self.mapping_valid
            and not self.errors
        )

    @property
    def valid(self) -> bool:
        return self.passed

    @property
    def failures(self) -> tuple[str, ...]:
        return self.errors

    @property
    def layout_valid(self) -> bool:
        return self.dimension_valid

    @property
    def mass_valid(self) -> bool:
        return self.masses_valid

    @property
    def map_valid(self) -> bool:
        return self.mapping_valid


def _finite_real_numeric(array: Array) -> bool:
    return bool(
        array.dtype.kind in "iuf"
        and array.dtype.kind != "b"
        and np.all(np.isfinite(array))
    )


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
    if array.ndim == 1 and array.size == 0:
        errors.append(f"{name} cannot be empty.")
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


def validate_fc2(data: FC2Data) -> FC2ValidationReport:
    """Audit a raw FC2 object without changing, remapping, or correcting it."""

    if not isinstance(data, FC2Data):
        raise TypeError("data must be an FC2Data instance.")

    errors: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, float] = {}
    values = np.asarray(data.values)

    dimension_valid = True
    n_first: int | None = None
    n_supercell: int | None = None
    if values.ndim != 4:
        errors.append(
            "FC2 values must have four dimensions: two atom and two "
            "Cartesian axes."
        )
        dimension_valid = False
    else:
        n_first = int(values.shape[0])
        n_supercell = int(values.shape[1])
        metrics["n_first_axis_atoms"] = float(n_first)
        metrics["n_supercell_atoms"] = float(n_supercell)
        metrics["element_count"] = float(values.size)
        if n_first < 1 or n_supercell < 1:
            errors.append("FC2 atom dimensions cannot be empty.")
            dimension_valid = False
        if values.shape[2:] != (3, 3):
            errors.append("The final two FC2 dimensions must each have size three.")
            dimension_valid = False
        if n_first > n_supercell:
            errors.append(
                "The first FC2 atom dimension cannot exceed the supercell size."
            )
            dimension_valid = False
        if data.compact:
            if n_first >= n_supercell:
                errors.append(
                    "A compact FC2 tensor must have shape (Np, Ns, 3, 3) "
                    "with Np < Ns."
                )
                dimension_valid = False
        elif n_first != n_supercell:
            errors.append(
                "A full FC2 tensor must have shape (Ns, Ns, 3, 3)."
            )
            dimension_valid = False

    finite_valid = _finite_real_numeric(values)
    if not finite_valid:
        if np.iscomplexobj(values):
            errors.append("FC2 values must be real.")
        else:
            errors.append("FC2 values must be real numeric and finite.")

    masses = np.asarray(data.masses)
    masses_valid = True
    if masses.ndim != 1:
        errors.append("masses must be a one-dimensional array.")
        masses_valid = False
    if not _finite_real_numeric(masses):
        errors.append("masses must contain only finite real numeric values.")
        masses_valid = False
    elif np.any(masses <= 0.0):
        errors.append("masses must contain only strictly positive values.")
        masses_valid = False
    if n_supercell is not None and (
        masses.ndim != 1 or masses.size != n_supercell
    ):
        errors.append(
            "masses must contain one value for each of the "
            f"{n_supercell} supercell atoms."
        )
        masses_valid = False

    units_valid = True
    try:
        metrics["force_constant_to_si_factor"] = (
            fc2_force_constant_to_si_factor(data.units)
        )
    except ValueError as exc:
        errors.append(str(exc))
        units_valid = False
    try:
        metrics["mass_to_si_factor"] = mass_to_si_factor(data.mass_units)
    except ValueError as exc:
        errors.append(str(exc))
        units_valid = False

    mapping_valid = True
    expected_p2s = (
        n_first if data.compact and dimension_valid and n_first is not None else None
    )
    p2s_ok, p2s = _validate_map_array(
        data.p2s_map,
        name="p2s_map",
        expected_length=expected_p2s,
        upper_bound=n_supercell,
        errors=errors,
    )
    mapping_valid &= p2s_ok
    if data.compact and data.p2s_map is None:
        errors.append("p2s_map is required for compact FC2 data.")
        mapping_valid = False
    if p2s is not None and np.unique(p2s).size != p2s.size:
        errors.append("p2s_map must contain unique representative supercell atoms.")
        mapping_valid = False

    s2p_ok, s2p = _validate_map_array(
        data.s2p_map,
        name="s2p_map",
        expected_length=n_supercell if data.s2p_map is not None else None,
        upper_bound=n_supercell,
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
            "s2p_map is absent; FC2 ingestion is valid, but later "
            "translational mapping requires complete crystal mapping data."
        )

    if p2s is not None:
        metrics["n_primitive_atoms"] = float(p2s.size)

    return FC2ValidationReport(
        dimension_valid=dimension_valid,
        finite_valid=finite_valid,
        masses_valid=masses_valid,
        units_valid=units_valid,
        mapping_valid=mapping_valid,
        errors=tuple(errors),
        warnings=tuple(warnings),
        metrics=metrics,
    )


__all__ = [
    "ANGSTROM_M",
    "ATOMIC_MASS_UNIT_KG",
    "ELECTRON_VOLT_J",
    "EV_ANGSTROM2_TO_J_M2",
    "FC2Data",
    "FC2ValidationReport",
    "fc2_force_constant_to_si_factor",
    "mass_to_si_factor",
    "validate_fc2",
]
