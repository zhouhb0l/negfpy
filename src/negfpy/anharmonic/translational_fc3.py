"""Explicit translational terms derived from audited real-space FC3.

This correctness-oriented representation implements both phono3py's legacy
first-atom-reference real-to-reciprocal convention
(``make_r0_average=False``) and its explicitly requested three-anchor
``r0`` average.  Each shortest-vector image is represented by a separate
immutable term with its exact multiplicity weight.  Physical
primitive-anchor blocks, including blocks whose optional nonzero mask is
zero, are never corrected, symmetrized, or truncated.  For full FC3 input,
redundant first-axis rows are audited against the primitive representative
rows using explicit pure translations and are never averaged into them.

For fractional reciprocal coordinates ``q0``, ``q1``, and ``q2``, a term is
transformed with

``exp(+2π i [(q0+q1+q2)·r_anchor + q1·R1 + q2·R2])``.

This is the positive-sign phase convention used by phono3py's
``RealToReciprocal`` implementation.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from ._canonical import (
    crystal_mapping_content_sha256,
    fc3_content_sha256,
    primitive_crystal_content_sha256,
    validate_sha256,
)
from .crystal import CrystalMapping, validate_crystal_mapping
from .fc3 import FC3Data, validate_fc3


Array = np.ndarray
Metadata = Mapping[str, Any]
SourceLayout = Literal["full", "compact"]

PHONO3PY_PHASE_CONVENTION = "exp(+2π i q·r)"
PHONO3PY_4_4_SYMPREC = 1.0e-5
PHONO3PY_LEGACY_ANCHOR_CONVENTION = (
    "phono3py first-primitive-atom reference, make_r0_average=False; "
    "(q0+q1+q2) acts on the primitive anchor position and q1/q2 act "
    "on the two shortest-vector offsets"
)
PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION = (
    "phono3py 4.4 three-anchor average, make_r0_average=True; "
    "legs (i,j,k), (j,i,k), and (k,j,i) use q-offset pairs (q1,q2), "
    "(q0,q2), and (q1,q0), Cartesian transposes (identity), (1,0,2), "
    "and (2,1,0), then divide by three"
)


def _readonly_array(
    values: Any,
    *,
    name: str,
    shape: tuple[int, ...],
    float_dtype: bool,
) -> Array:
    raw = np.asarray(values)
    if raw.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if float_dtype:
        if raw.dtype.kind == "c":
            raise TypeError(f"{name} must be real-valued.")
        try:
            array = np.array(raw, dtype=np.float64, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must be a real numerical array.") from exc
    else:
        if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
            raise TypeError(f"{name} must be a real numerical array.")
        array = np.array(raw, order="C", copy=True)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
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


def _index_triplet(values: Sequence[int], *, name: str) -> tuple[int, int, int]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain three integer indices.")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain three integer indices.") from exc
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three indices.")
    normalized: list[int] = []
    for value in result:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"{name} must contain integer indices.")
        integer = int(value)
        if integer < 0:
            raise ValueError(f"{name} indices must be non-negative.")
        normalized.append(integer)
    return normalized[0], normalized[1], normalized[2]


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer.")
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return integer


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


def _positive_weight(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a positive finite real number."
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return number


def _positive_tolerance(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be a positive finite real number."
        ) from exc
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return number


def _readonly_all_shortest(
    values: Any,
    *,
    n_primitive: int,
    n_supercell: int,
) -> Array:
    raw = np.asarray(values)
    expected_shape = (n_primitive, n_supercell, n_supercell)
    if raw.shape != expected_shape:
        raise ValueError(
            f"all_shortest must have shape {expected_shape}."
        )
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError("all_shortest must contain integer flags.")
    if np.any((raw != 0) & (raw != 1)):
        raise ValueError("all_shortest may contain only zero and one.")
    flags = np.array(raw, dtype=np.byte, order="C", copy=True)
    flags.setflags(write=False)
    return flags


@dataclass(frozen=True, slots=True, eq=False)
class FC3Term:
    """One raw FC3 block at one pair of shortest-vector images.

    ``values`` is the unmodified source ``(3, 3, 3)`` block.
    ``anchor_weight`` is an explicit normalization factor (the current
    conversion uses one).  ``multiplicity_weight`` is exactly
    ``1 / (m_second * m_third)`` for the two shortest-vector degeneracies.
    """

    primitive_indices: tuple[int, int, int]
    source_indices: tuple[int, int, int]
    translated_supercell_indices: tuple[int, int, int]
    anchor_position: Array
    second_offset: Array
    third_offset: Array
    second_vector_index: int
    third_vector_index: int
    second_multiplicity: int
    third_multiplicity: int
    anchor_weight: float
    multiplicity_weight: float
    values: Array

    def __post_init__(self) -> None:
        primitive_indices = _index_triplet(
            self.primitive_indices,
            name="primitive_indices",
        )
        source_indices = _index_triplet(
            self.source_indices,
            name="source_indices",
        )
        translated_supercell_indices = _index_triplet(
            self.translated_supercell_indices,
            name="translated_supercell_indices",
        )
        anchor_position = _readonly_array(
            self.anchor_position,
            name="anchor_position",
            shape=(3,),
            float_dtype=True,
        )
        second_offset = _readonly_array(
            self.second_offset,
            name="second_offset",
            shape=(3,),
            float_dtype=True,
        )
        third_offset = _readonly_array(
            self.third_offset,
            name="third_offset",
            shape=(3,),
            float_dtype=True,
        )
        values = _readonly_array(
            self.values,
            name="values",
            shape=(3, 3, 3),
            float_dtype=False,
        )

        object.__setattr__(self, "primitive_indices", primitive_indices)
        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(
            self,
            "translated_supercell_indices",
            translated_supercell_indices,
        )
        object.__setattr__(self, "anchor_position", anchor_position)
        object.__setattr__(self, "second_offset", second_offset)
        object.__setattr__(self, "third_offset", third_offset)
        object.__setattr__(
            self,
            "second_vector_index",
            _nonnegative_integer(
                self.second_vector_index,
                name="second_vector_index",
            ),
        )
        object.__setattr__(
            self,
            "third_vector_index",
            _nonnegative_integer(
                self.third_vector_index,
                name="third_vector_index",
            ),
        )
        second_multiplicity = _positive_integer(
            self.second_multiplicity,
            name="second_multiplicity",
        )
        third_multiplicity = _positive_integer(
            self.third_multiplicity,
            name="third_multiplicity",
        )
        object.__setattr__(
            self,
            "second_multiplicity",
            second_multiplicity,
        )
        object.__setattr__(
            self,
            "third_multiplicity",
            third_multiplicity,
        )
        object.__setattr__(
            self,
            "anchor_weight",
            _positive_weight(self.anchor_weight, name="anchor_weight"),
        )
        multiplicity_weight = _positive_weight(
            self.multiplicity_weight,
            name="multiplicity_weight",
        )
        expected_weight = 1.0 / float(
            second_multiplicity * third_multiplicity
        )
        if multiplicity_weight != expected_weight:
            raise ValueError(
                "multiplicity_weight must equal exactly "
                "1 / (second_multiplicity * third_multiplicity)."
            )
        object.__setattr__(
            self,
            "multiplicity_weight",
            multiplicity_weight,
        )
        object.__setattr__(self, "values", values)

    @property
    def anchor(self) -> int:
        return self.primitive_indices[0]

    @property
    def weight(self) -> float:
        return self.anchor_weight * self.multiplicity_weight


def _term_sort_key(term: FC3Term) -> tuple[Any, ...]:
    return (
        term.primitive_indices,
        term.source_indices,
        term.translated_supercell_indices,
        tuple(float(value) for value in term.anchor_position),
        tuple(float(value) for value in term.second_offset),
        tuple(float(value) for value in term.third_offset),
        term.second_vector_index,
        term.third_vector_index,
        term.second_multiplicity,
        term.third_multiplicity,
        term.anchor_weight,
        term.multiplicity_weight,
        tuple(float(value) for value in term.values.reshape(-1)),
    )


@dataclass(frozen=True, slots=True, eq=False)
class TranslationalFC3:
    """Deterministic immutable sequence of explicit translational FC3 terms."""

    terms: tuple[FC3Term, ...]
    n_primitive_atoms: int
    n_supercell_atoms: int
    units: str
    mass_units: str
    source_layout: SourceLayout
    phase_convention: str = PHONO3PY_PHASE_CONVENTION
    anchor_convention: str = PHONO3PY_LEGACY_ANCHOR_CONVENTION
    make_r0_average: bool = False
    all_shortest: Array | None = None
    symprec: float | None = None
    provenance: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            terms = tuple(self.terms)
        except TypeError as exc:
            raise TypeError("terms must be an iterable of FC3Term objects.") from exc
        if any(not isinstance(term, FC3Term) for term in terms):
            raise TypeError("terms must contain only FC3Term objects.")
        terms = tuple(sorted(terms, key=_term_sort_key))

        n_primitive = _positive_integer(
            self.n_primitive_atoms,
            name="n_primitive_atoms",
        )
        n_supercell = _positive_integer(
            self.n_supercell_atoms,
            name="n_supercell_atoms",
        )
        if n_primitive > n_supercell:
            raise ValueError(
                "n_primitive_atoms cannot exceed n_supercell_atoms."
            )
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("units must be a non-empty string.")
        if not isinstance(self.mass_units, str) or not self.mass_units.strip():
            raise ValueError("mass_units must be a non-empty string.")
        if self.source_layout not in {"full", "compact"}:
            raise ValueError("source_layout must be either 'full' or 'compact'.")
        if self.phase_convention != PHONO3PY_PHASE_CONVENTION:
            raise ValueError(
                "Only the phono3py positive phase convention "
                f"{PHONO3PY_PHASE_CONVENTION!r} is supported."
            )
        if not isinstance(self.make_r0_average, (bool, np.bool_)):
            raise TypeError("make_r0_average must be a boolean.")
        make_r0_average = bool(self.make_r0_average)
        if make_r0_average:
            if (
                self.anchor_convention
                != PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION
            ):
                raise ValueError(
                    "make_r0_average=True requires the phono3py 4.4 "
                    "three-anchor convention."
                )
            if self.all_shortest is None:
                raise ValueError(
                    "all_shortest is required when make_r0_average=True."
                )
            all_shortest = _readonly_all_shortest(
                self.all_shortest,
                n_primitive=n_primitive,
                n_supercell=n_supercell,
            )
            if self.symprec is None:
                raise ValueError(
                    "symprec is required when make_r0_average=True."
                )
            symprec: float | None = _positive_tolerance(
                self.symprec,
                name="symprec",
            )
        else:
            if self.anchor_convention != PHONO3PY_LEGACY_ANCHOR_CONVENTION:
                raise ValueError(
                    "make_r0_average=False requires the phono3py legacy "
                    "first-anchor convention."
                )
            if self.all_shortest is not None:
                raise ValueError(
                    "The legacy convention must not carry all_shortest; "
                    "set make_r0_average=True explicitly."
                )
            if self.symprec is not None:
                raise ValueError(
                    "The legacy convention must not carry symprec because "
                    "it does not derive or apply all_shortest."
                )
            all_shortest = None
            symprec = None

        first_source_bound = (
            n_primitive if self.source_layout == "compact" else n_supercell
        )
        for term in terms:
            if any(index >= n_primitive for index in term.primitive_indices):
                raise ValueError(
                    "Term primitive indices exceed n_primitive_atoms."
                )
            source_first, source_second, source_third = term.source_indices
            if (
                source_first >= first_source_bound
                or source_second >= n_supercell
                or source_third >= n_supercell
            ):
                raise ValueError(
                    "Term source indices are inconsistent with source_layout "
                    "and atom counts."
                )
            if any(
                index >= n_supercell
                for index in term.translated_supercell_indices
            ):
                raise ValueError(
                    "Term translated supercell indices exceed "
                    "n_supercell_atoms."
                )

        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "n_primitive_atoms", n_primitive)
        object.__setattr__(self, "n_supercell_atoms", n_supercell)
        object.__setattr__(self, "make_r0_average", make_r0_average)
        object.__setattr__(self, "all_shortest", all_shortest)
        object.__setattr__(self, "symprec", symprec)
        object.__setattr__(self, "provenance", _freeze_mapping(self.provenance))

    def fourier(self, q_triplet: Any) -> Array:
        """Reconstruct reciprocal FC3 for three fractional q vectors.

        The returned atom-first array has shape
        ``(Np, Np, Np, 3, 3, 3)`` and complex128 dtype.  Terms are summed in
        canonical order, making the serial result deterministic.
        """

        raw_q = np.asarray(q_triplet)
        if raw_q.dtype.kind == "c":
            raise TypeError("q_triplet must be real-valued.")
        try:
            q = np.array(raw_q, dtype=np.float64, order="C", copy=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("q_triplet must be a real numerical array.") from exc
        if q.shape != (3, 3):
            raise ValueError("q_triplet must have shape (3, 3).")
        if not np.all(np.isfinite(q)):
            raise ValueError("q_triplet must contain only finite values.")

        result = np.zeros(
            (
                self.n_primitive_atoms,
                self.n_primitive_atoms,
                self.n_primitive_atoms,
                3,
                3,
                3,
            ),
            dtype=np.complex128,
        )
        if self.make_r0_average:
            return self._fourier_r0_average(q, result)

        q_sum = np.sum(q, axis=0)
        for term in self.terms:
            phase_argument = (
                np.dot(q_sum, term.anchor_position)
                + np.dot(q[1], term.second_offset)
                + np.dot(q[2], term.third_offset)
            )
            phase = np.exp(2j * np.pi * phase_argument)
            result[term.primitive_indices] += (
                term.weight * term.values * phase
            )
        return result

    def _fourier_r0_average(self, q: Array, result: Array) -> Array:
        """Apply phono3py 4.4's exact three-anchor r0 algorithm."""

        assert self.all_shortest is not None
        grouped: dict[tuple[int, int, int], list[FC3Term]] = {}
        for term in self.terms:
            grouped.setdefault(term.primitive_indices, []).append(term)
        q_sum = np.sum(q, axis=0)

        def leg(
            primitive_indices: tuple[int, int, int],
            *,
            q_indices: tuple[int, int],
            leg_index: int,
        ) -> Array:
            block = np.zeros((3, 3, 3), dtype=np.complex128)
            for term in grouped.get(primitive_indices, ()):
                flag = int(
                    self.all_shortest[
                        term.anchor,
                        term.source_indices[1],
                        term.source_indices[2],
                    ]
                )
                if leg_index > 1 and flag:
                    continue
                factor = 3.0 if leg_index == 1 and flag else 1.0
                phase_argument = (
                    np.dot(q_sum, term.anchor_position)
                    + np.dot(q[q_indices[0]], term.second_offset)
                    + np.dot(q[q_indices[1]], term.third_offset)
                )
                phase = np.exp(2j * np.pi * phase_argument)
                block += factor * term.weight * term.values * phase
            return block

        for first in range(self.n_primitive_atoms):
            for second in range(self.n_primitive_atoms):
                for third in range(self.n_primitive_atoms):
                    leg_1 = leg(
                        (first, second, third),
                        q_indices=(1, 2),
                        leg_index=1,
                    )
                    leg_2 = leg(
                        (second, first, third),
                        q_indices=(0, 2),
                        leg_index=2,
                    ).transpose(1, 0, 2)
                    leg_3 = leg(
                        (third, second, first),
                        q_indices=(1, 0),
                        leg_index=3,
                    ).transpose(2, 1, 0)
                    result[first, second, third] = (
                        leg_1 + leg_2 + leg_3
                    ) / 3.0
        return result

    def fourier_transform(self, q_triplet: Any) -> Array:
        """Alias for :meth:`fourier`."""

        return self.fourier(q_triplet)


def _mapping_primitive_index(mapping: CrystalMapping, satom: int) -> int:
    representative = int(mapping.s2p[satom])
    primitive = int(mapping.p2p[representative])
    if primitive < 0:
        raise ValueError(
            "CrystalMapping cannot reduce a representative supercell atom "
            "to a primitive ordinal."
        )
    return primitive


def derive_phono3py_all_shortest(
    mapping: CrystalMapping,
    *,
    symprec: float = PHONO3PY_4_4_SYMPREC,
) -> Array:
    """Derive phono3py 4.4's immutable three-anchor optimization flags.

    This is a direct implementation of
    ``Interaction._get_all_shortest``.  It does not modify FC3, shortest
    vectors, or crystal mappings.  A flag is one only when all three relevant
    shortest vectors are unique and the translated pair distance agrees
    within the explicitly supplied ``symprec``.
    """

    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping instance.")
    symprec = _positive_tolerance(symprec, name="symprec")
    if mapping.purpose != "fc3":
        raise ValueError("CrystalMapping purpose must be 'fc3'.")
    report = validate_crystal_mapping(mapping)
    if not report.passed:
        details = "; ".join(report.errors) or "invalid crystal mapping"
        raise ValueError(f"CrystalMapping audit failed: {details}")

    translations = mapping.translation_permutations
    if translations is None:
        raise NotImplementedError(
            "phono3py all_shortest derivation requires explicit "
            "CrystalMapping.translation_permutations."
        )

    n_primitive = mapping.n_primitive_atoms
    n_supercell = mapping.n_supercell_atoms
    multiplicity = mapping.multiplicity
    shortest_vectors = mapping.shortest_vectors
    lattice = mapping.primitive.lattice
    all_shortest = np.zeros(
        (n_primitive, n_supercell, n_supercell),
        dtype=np.byte,
    )

    for primitive_anchor in range(n_primitive):
        for source_second in range(n_supercell):
            if int(multiplicity[source_second, primitive_anchor, 0]) > 1:
                continue
            second_primitive = _mapping_primitive_index(
                mapping,
                source_second,
            )
            representative = int(mapping.p2s[second_primitive])
            candidates = np.flatnonzero(
                translations[:, source_second] == representative
            )
            if candidates.size != 1:
                raise ValueError(
                    "Exactly one pure translation must map supercell atom "
                    f"{source_second} to primitive representative "
                    f"{representative}; found {candidates.size}."
                )
            translation = translations[int(candidates[0])]
            second_start = int(
                multiplicity[source_second, primitive_anchor, 1]
            )

            for source_third in range(n_supercell):
                if int(
                    multiplicity[source_third, primitive_anchor, 0]
                ) > 1:
                    continue
                translated_third = int(translation[source_third])
                if int(
                    multiplicity[
                        translated_third,
                        second_primitive,
                        0,
                    ]
                ) > 1:
                    continue
                third_start = int(
                    multiplicity[source_third, primitive_anchor, 1]
                )
                translated_start = int(
                    multiplicity[
                        translated_third,
                        second_primitive,
                        1,
                    ]
                )
                second_to_third = (
                    shortest_vectors[third_start]
                    - shortest_vectors[second_start]
                )
                distance = float(
                    np.linalg.norm(second_to_third @ lattice)
                )
                translated_distance = float(
                    np.linalg.norm(
                        shortest_vectors[translated_start] @ lattice
                    )
                )
                if abs(translated_distance - distance) < symprec:
                    all_shortest[
                        primitive_anchor,
                        source_second,
                        source_third,
                    ] = 1

    all_shortest.setflags(write=False)
    return all_shortest


def _translation_to_representative(
    mapping: CrystalMapping,
    *,
    source_first: int,
    primitive_anchor: int,
) -> Array:
    translations = mapping.translation_permutations
    if translations is None:
        raise NotImplementedError(
            "General full FC3 conversion requires explicit primitive pure "
            "translation permutations in CrystalMapping; use compact FC3 "
            "until translation_permutations are available."
        )
    representative = int(mapping.p2s[primitive_anchor])
    candidates: dict[tuple[int, ...], Array] = {}
    for row in translations:
        if int(row[source_first]) != representative:
            continue
        key = tuple(int(value) for value in row)
        candidates[key] = row

    if not candidates:
        raise ValueError(
            "No translation permutation maps full FC3 first-axis atom "
            f"{source_first} to primitive representative {representative}."
        )
    if len(candidates) != 1:
        raise ValueError(
            "The translation permutation for full FC3 first-axis atom "
            f"{source_first} is ambiguous; refusing to choose one."
        )
    permutation = next(iter(candidates.values()))

    # CrystalMapping validates this table as pure primitive translations; the
    # class-preservation invariant is repeated at this scientific boundary.
    for source in range(mapping.n_supercell_atoms):
        target = int(permutation[source])
        if int(mapping.s2p[target]) != int(mapping.s2p[source]):
            raise ValueError(
                "CrystalMapping translation_permutations include an operation "
                "that changes primitive atom classes; the full-FC3 mapping is "
                "not a pure translation."
            )
    return permutation


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


def _audit_full_translation_redundancy(
    values: Array,
    mapping: CrystalMapping,
    *,
    atol: float,
    rtol: float,
) -> dict[str, Any]:
    """Audit, but never average, redundant full-FC3 first-axis rows."""

    representatives = set(int(value) for value in mapping.p2s)
    redundant = [
        source
        for source in range(mapping.n_supercell_atoms)
        if source not in representatives
    ]
    if redundant and mapping.translation_permutations is None:
        raise NotImplementedError(
            "General full FC3 conversion requires explicit primitive pure "
            "translation permutations in CrystalMapping; use compact FC3 "
            "until translation_permutations are available."
        )

    maximum_absolute = 0.0
    maximum_relative = 0.0
    for source_first in redundant:
        primitive_anchor = _mapping_primitive_index(mapping, source_first)
        representative = int(mapping.p2s[primitive_anchor])
        translation = _translation_to_representative(
            mapping,
            source_first=source_first,
            primitive_anchor=primitive_anchor,
        )
        expected = values[representative][translation][:, translation]
        actual = values[source_first]
        residual = float(np.max(np.abs(actual - expected)))
        scale = max(
            float(np.max(np.abs(actual))),
            float(np.max(np.abs(expected))),
            np.finfo(float).tiny,
        )
        relative = residual / scale
        maximum_absolute = max(maximum_absolute, residual)
        maximum_relative = max(maximum_relative, relative)
        threshold = atol + rtol * scale
        if residual > threshold:
            raise ValueError(
                "Full FC3 redundant translation audit failed for first-axis "
                f"atom {source_first} against representative "
                f"{representative}: residual {residual:.6e}, allowed "
                f"{threshold:.6e}. The redundant row was not averaged or "
                "used to replace the representative row."
            )

    return {
        "passed": True,
        "checked_first_axis_rows": len(redundant),
        "maximum_absolute_residual": maximum_absolute,
        "maximum_relative_residual": maximum_relative,
        "atol": atol,
        "rtol": rtol,
        "policy": (
            "validate redundant full first-axis rows through explicit pure "
            "translations; construct terms only from p2s representative rows"
        ),
    }


def _verify_data_mapping(data: FC3Data, mapping: CrystalMapping) -> None:
    if mapping.purpose != "fc3":
        raise ValueError("CrystalMapping purpose must be 'fc3'.")
    mapping_report = validate_crystal_mapping(mapping)
    if not mapping_report.passed:
        details = "; ".join(mapping_report.errors) or "invalid crystal mapping"
        raise ValueError(f"CrystalMapping audit failed: {details}")

    report = validate_fc3(data)
    if not report.passed:
        details = "; ".join(report.errors) or "invalid FC3 data"
        raise ValueError(f"FC3 audit failed: {details}")

    n_supercell = mapping.n_supercell_atoms
    n_primitive = mapping.n_primitive_atoms
    if data.values.shape[1:3] != (n_supercell, n_supercell):
        raise ValueError(
            "FC3 supercell atom dimensions conflict with CrystalMapping."
        )
    expected_first = n_primitive if data.compact else n_supercell
    if data.values.shape[0] != expected_first:
        raise ValueError(
            "FC3 first atom dimension conflicts with its layout and "
            "CrystalMapping."
        )
    if data.p2s_map is not None and not np.array_equal(
        data.p2s_map,
        mapping.p2s,
    ):
        raise ValueError("FC3 p2s_map conflicts with CrystalMapping.p2s.")
    if data.s2p_map is not None and not np.array_equal(
        data.s2p_map,
        mapping.s2p,
    ):
        raise ValueError("FC3 s2p_map conflicts with CrystalMapping.s2p.")
    if not np.allclose(
        np.asarray(data.masses, dtype=np.float64),
        mapping.supercell.masses,
        rtol=1.0e-12,
        atol=0.0,
    ):
        raise ValueError(
            "FC3 masses conflict with the ordered CrystalMapping supercell "
            "masses."
        )


def _shortest_vector_partition(
    mapping: CrystalMapping,
    *,
    satom: int,
    anchor: int,
) -> tuple[int, int]:
    count = int(mapping.multiplicity[satom, anchor, 0])
    offset = int(mapping.multiplicity[satom, anchor, 1])
    if count <= 0 or offset < 0:
        # CrystalMapping already validates this, but keep this local invariant
        # explicit at the conversion boundary.
        raise ValueError(
            "CrystalMapping contains an invalid shortest-vector partition."
        )
    return count, offset


def convert_fc3_to_translational(
    data: FC3Data,
    mapping: CrystalMapping,
    *,
    make_r0_average: bool = False,
    symprec: float = PHONO3PY_4_4_SYMPREC,
    translation_atol: float = 1.0e-12,
    translation_rtol: float = 1.0e-10,
) -> TranslationalFC3:
    """Convert audited FC3 and crystal mappings into explicit terms.

    Compact input uses one primitive first-axis anchor directly.  Full input
    follows phono3py: physical terms are constructed only from
    ``fc3[p2s[anchor], :, :]``.  Every other full first-axis row is first
    checked against that representative through a unique explicit pure
    translation.  An inconsistent row is rejected and is never averaged.

    When ``make_r0_average=True``, phono3py 4.4's ``all_shortest`` flags are
    derived from the audited pure-translation permutations and shortest-vector
    tables at the explicit ``symprec``.  This selects the three-anchor Fourier
    representation; it never averages or corrects the input FC3 tensor.
    """

    if not isinstance(data, FC3Data):
        raise TypeError("data must be an FC3Data instance.")
    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping instance.")
    if not isinstance(make_r0_average, (bool, np.bool_)):
        raise TypeError("make_r0_average must be a boolean.")
    make_r0_average = bool(make_r0_average)
    if make_r0_average:
        symprec_value: float | None = _positive_tolerance(
            symprec,
            name="symprec",
        )
    else:
        # The legacy result does not depend on symprec.  Retain historical
        # behavior by neither validating nor recording an unused tolerance.
        symprec_value = None

    translation_atol = _nonnegative_tolerance(
        translation_atol,
        name="translation_atol",
    )
    translation_rtol = _nonnegative_tolerance(
        translation_rtol,
        name="translation_rtol",
    )
    _verify_data_mapping(data, mapping)
    source_fc3_sha256 = fc3_content_sha256(data)
    declared_source_fc3_sha256 = data.provenance.get(
        "source_fc3_content_sha256"
    )
    if declared_source_fc3_sha256 is None:
        raise ValueError("FC3Data is missing source_fc3_content_sha256.")
    if validate_sha256(
        declared_source_fc3_sha256,
        name="source_fc3_content_sha256",
    ) != source_fc3_sha256:
        raise ValueError(
            "FC3Data source_fc3_content_sha256 conflicts with its content."
        )
    primitive_crystal_sha256 = primitive_crystal_content_sha256(mapping)
    fc3_mapping_sha256 = crystal_mapping_content_sha256(mapping)
    all_shortest = (
        derive_phono3py_all_shortest(
            mapping,
            symprec=symprec_value,
        )
        if make_r0_average
        else None
    )

    values = np.asarray(data.values)
    n_primitive = mapping.n_primitive_atoms
    n_supercell = mapping.n_supercell_atoms

    class_counts = np.zeros(n_primitive, dtype=np.int64)
    for satom in range(n_supercell):
        class_counts[_mapping_primitive_index(mapping, satom)] += 1
    if np.any(class_counts <= 0) or not np.all(class_counts == class_counts[0]):
        raise ValueError(
            "CrystalMapping supercell atoms do not form equally sized "
            "primitive translation classes."
        )

    if data.compact:
        redundant_translation_audit: dict[str, Any] = {
            "passed": True,
            "checked_first_axis_rows": 0,
            "maximum_absolute_residual": 0.0,
            "maximum_relative_residual": 0.0,
            "atol": translation_atol,
            "rtol": translation_rtol,
            "policy": "not applicable to compact FC3",
        }
    else:
        redundant_translation_audit = _audit_full_translation_redundancy(
            values,
            mapping,
            atol=translation_atol,
            rtol=translation_rtol,
        )

    identity = np.arange(n_supercell, dtype=np.int64)
    source_first_records = [
        (
            (
                primitive_anchor
                if data.compact
                else int(mapping.p2s[primitive_anchor])
            ),
            primitive_anchor,
            identity,
            1.0,
        )
        for primitive_anchor in range(n_primitive)
    ]

    terms: list[FC3Term] = []
    for (
        source_first,
        primitive_anchor,
        translation,
        anchor_weight,
    ) in source_first_records:
        anchor_position = mapping.primitive.scaled_positions[primitive_anchor]
        for source_second in range(n_supercell):
            translated_second = int(translation[source_second])
            primitive_second = _mapping_primitive_index(
                mapping,
                translated_second,
            )
            second_count, second_start = _shortest_vector_partition(
                mapping,
                satom=translated_second,
                anchor=primitive_anchor,
            )
            for source_third in range(n_supercell):
                translated_third = int(translation[source_third])
                primitive_third = _mapping_primitive_index(
                    mapping,
                    translated_third,
                )
                third_count, third_start = _shortest_vector_partition(
                    mapping,
                    satom=translated_third,
                    anchor=primitive_anchor,
                )
                multiplicity_weight = 1.0 / float(
                    second_count * third_count
                )
                raw_block = values[
                    source_first,
                    source_second,
                    source_third,
                ]
                for second_vector_index in range(
                    second_start,
                    second_start + second_count,
                ):
                    second_offset = mapping.shortest_vectors[
                        second_vector_index
                    ]
                    for third_vector_index in range(
                        third_start,
                        third_start + third_count,
                    ):
                        third_offset = mapping.shortest_vectors[
                            third_vector_index
                        ]
                        terms.append(
                            FC3Term(
                                primitive_indices=(
                                    primitive_anchor,
                                    primitive_second,
                                    primitive_third,
                                ),
                                source_indices=(
                                    source_first,
                                    source_second,
                                    source_third,
                                ),
                                translated_supercell_indices=(
                                    int(mapping.p2s[primitive_anchor]),
                                    translated_second,
                                    translated_third,
                                ),
                                anchor_position=anchor_position,
                                second_offset=second_offset,
                                third_offset=third_offset,
                                second_vector_index=second_vector_index,
                                third_vector_index=third_vector_index,
                                second_multiplicity=second_count,
                                third_multiplicity=third_count,
                                anchor_weight=anchor_weight,
                                multiplicity_weight=multiplicity_weight,
                                values=raw_block,
                            )
                        )

    layout: SourceLayout = "compact" if data.compact else "full"
    anchor_convention = (
        PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION
        if make_r0_average
        else PHONO3PY_LEGACY_ANCHOR_CONVENTION
    )
    provenance = {
        "format": "negfpy_translational_fc3_v1",
        "source_layout": layout,
        "phase_convention": PHONO3PY_PHASE_CONVENTION,
        "anchor_convention": anchor_convention,
        "make_r0_average": make_r0_average,
        "all_shortest_algorithm": (
            "phono3py 4.4 Interaction._get_all_shortest exact port"
            if make_r0_average
            else "not_applied"
        ),
        "all_shortest_symprec": symprec_value,
        "all_shortest_true_count": (
            int(np.count_nonzero(all_shortest))
            if all_shortest is not None
            else 0
        ),
        "all_shortest_sha256": (
            hashlib.sha256(all_shortest.tobytes(order="C")).hexdigest()
            if all_shortest is not None
            else None
        ),
        "r0_policy": (
            "three-anchor Fourier representation only; raw FC3 is not "
            "averaged, corrected, symmetrized, or replaced"
            if make_r0_average
            else "legacy single-anchor Fourier representation"
        ),
        "full_first_axis_policy": (
            "not_applicable"
            if data.compact
            else "phono3py p2s representative rows only after all redundant "
            "rows pass the pure-translation consistency audit"
        ),
        "translation_permutation_direction": (
            "translation_permutations[row, source] gives translated target; "
            "selected row maps source first atom to p2s representative"
        ),
        "raw_atom_triplet_count": int(
            values.shape[0] * n_supercell * n_supercell
        ),
        "physical_atom_triplet_count": int(
            n_primitive * n_supercell * n_supercell
        ),
        "redundant_atom_triplet_count": int(
            (values.shape[0] - n_primitive)
            * n_supercell
            * n_supercell
        ),
        "expanded_term_count": len(terms),
        "redundant_translation_audit": redundant_translation_audit,
        "fc3_nonzero_indices_present": data.fc3_nonzero_indices is not None,
        "fc3_nonzero_indices_applied": False,
        "symmetrized": False,
        "corrected": False,
        "truncated": False,
        "source_fc3_content_sha256": source_fc3_sha256,
        "primitive_crystal_content_sha256": primitive_crystal_sha256,
        "fc3_crystal_mapping_content_sha256": fc3_mapping_sha256,
        "source_fc3_provenance": data.provenance,
        "source_mapping_provenance": mapping.provenance,
    }
    return TranslationalFC3(
        terms=tuple(terms),
        n_primitive_atoms=n_primitive,
        n_supercell_atoms=n_supercell,
        units=data.units,
        mass_units=data.mass_units,
        source_layout=layout,
        anchor_convention=anchor_convention,
        make_r0_average=make_r0_average,
        all_shortest=all_shortest,
        symprec=symprec_value,
        provenance=provenance,
    )


to_translational_fc3 = convert_fc3_to_translational


__all__ = [
    "FC3Term",
    "PHONO3PY_4_4_SYMPREC",
    "PHONO3PY_LEGACY_ANCHOR_CONVENTION",
    "PHONO3PY_PHASE_CONVENTION",
    "PHONO3PY_R0_AVERAGE_ANCHOR_CONVENTION",
    "TranslationalFC3",
    "convert_fc3_to_translational",
    "derive_phono3py_all_shortest",
    "to_translational_fc3",
]
