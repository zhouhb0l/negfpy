"""Canonical byte identities used by the material-NEGF implementation.

This module is deliberately private.  It contains the small, auditable byte
grammar frozen by the sparse :math:`N_q=1` design contract, rather than using
JSON, pickle, native-endian array bytes, or Python object hashes.  Physics
objects use the helpers here to fail closed when their content identities do
not reproduce.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import math
import re
import struct
import unicodedata
from typing import Any, Literal

import numpy as np

from .crystal import CrystalMapping, validate_crystal_mapping
from .fc2 import FC2Data, validate_fc2
from .fc3 import FC3Data


Array = np.ndarray
CosetRole = Literal["D", "R"]

_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U64_MAX = (1 << 64) - 1
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


def _u16(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("u16 values must be integers.")
    if not 0 <= value <= 0xFFFF:
        raise ValueError("u16 value is out of range.")
    return struct.pack("<H", value)


def _u64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("u64 values must be integers.")
    if not 0 <= value <= _U64_MAX:
        raise ValueError("u64 value is out of range.")
    return struct.pack("<Q", value)


def _i64(value: int) -> bytes:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("i64 values must be integers.")
    if not _I64_MIN <= value <= _I64_MAX:
        raise ValueError("i64 value is out of range.")
    return struct.pack("<q", value)


def _f64(value: float, *, name: str = "real") -> bytes:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if number == 0.0:
        number = 0.0
    return struct.pack("<d", number)


def _normalized_text(value: str, *, name: str = "string") -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    return unicodedata.normalize("NFC", value)


def _canonical_array(values: Array) -> tuple[int, Array]:
    """Return ``(dtype_code, little-endian C array)`` for ``E(ndarray)``."""

    if not isinstance(values, np.ndarray):
        raise TypeError("Canonical array encoding requires an ndarray.")
    dtype = values.dtype
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError("Object, structured, and subarray dtypes are unsupported.")

    if dtype.kind == "b" and dtype.itemsize == 1:
        code = 0x01
        target = np.dtype(np.bool_)
    elif dtype.kind == "u" and dtype.itemsize == 1:
        code = 0x01
        target = np.dtype(np.uint8)
    elif dtype.kind == "i" and dtype.itemsize == 8:
        code = 0x02
        target = np.dtype("<i8")
    elif dtype.kind == "u" and dtype.itemsize == 8:
        code = 0x03
        target = np.dtype("<u8")
    elif dtype.kind == "f" and dtype.itemsize == 8:
        code = 0x04
        target = np.dtype("<f8")
    elif dtype.kind == "c" and dtype.itemsize == 16:
        code = 0x05
        target = np.dtype("<c16")
    else:
        raise TypeError(
            "Canonical arrays must have dtype bool/u1, int64, uint64, "
            "float64, or complex128."
        )

    array = np.array(values, dtype=target, order="C", copy=True)
    if dtype.kind in "fc":
        if not np.all(np.isfinite(array)):
            raise ValueError("Canonical arrays must contain only finite values.")
        if dtype.kind == "f":
            array[array == 0.0] = 0.0
        else:
            components = array.view(np.dtype("<f8")).reshape(array.shape + (2,))
            components[components == 0.0] = 0.0
    return code, array


def canonical_encode(value: Any) -> bytes:
    """Encode a value with the frozen tagged ``E(value)`` grammar.

    Only the explicitly declared scalar, sequence, mapping, and ndarray
    types are accepted.  In particular, NumPy scalars, sets, object arrays,
    and platform-width numerical arrays are rejected instead of being
    coerced implicitly.
    """

    if value is None:
        return b"\x00"
    if isinstance(value, np.generic):
        raise TypeError(
            "NumPy scalar values are unsupported; convert them explicitly "
            "to a declared Python scalar or canonical ndarray."
        )
    if value is False:
        return b"\x01"
    if value is True:
        return b"\x02"
    if isinstance(value, int):
        return b"\x03" + _i64(value)
    if isinstance(value, float):
        return b"\x04" + _f64(value)
    if isinstance(value, complex):
        return b"\x05" + _f64(value.real, name="complex real component") + _f64(
            value.imag,
            name="complex imaginary component",
        )
    if isinstance(value, str):
        payload = _normalized_text(value).encode("utf-8")
        return b"\x06" + _u64(len(payload)) + payload
    if isinstance(value, bytes):
        return b"\x07" + _u64(len(value)) + value
    if isinstance(value, (list, tuple)):
        return b"\x08" + _u64(len(value)) + b"".join(
            canonical_encode(item) for item in value
        )
    if isinstance(value, Mapping):
        normalized: dict[bytes, tuple[str, Any]] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Canonical mapping keys must be strings.")
            key_text = _normalized_text(key, name="mapping key")
            key_bytes = key_text.encode("utf-8")
            if key_bytes in normalized:
                raise ValueError(
                    "Canonical mapping contains duplicate keys after NFC "
                    "normalization."
                )
            normalized[key_bytes] = (key_text, item)
        encoded = [b"\x09", _u64(len(normalized))]
        for key_bytes in sorted(normalized):
            key_text, item = normalized[key_bytes]
            encoded.extend((canonical_encode(key_text), canonical_encode(item)))
        return b"".join(encoded)
    if isinstance(value, np.ndarray):
        code, array = _canonical_array(value)
        data = array.tobytes(order="C")
        encoded = [b"\x0a", bytes((code,)), _u64(array.ndim)]
        encoded.extend(_u64(int(dimension)) for dimension in array.shape)
        encoded.extend((_u64(len(data)), data))
        return b"".join(encoded)
    raise TypeError(f"Unsupported canonical value type: {type(value).__name__}.")


def domain_digest(
    domain: str | bytes,
    payload: Any,
    *,
    version: int = 1,
) -> str:
    """Return the common domain-separated digest for a canonical payload."""

    if isinstance(domain, str):
        try:
            domain_bytes = domain.encode("ascii")
        except UnicodeEncodeError as exc:
            raise ValueError("Digest domains must contain ASCII only.") from exc
    elif isinstance(domain, bytes):
        domain_bytes = domain
        try:
            domain_bytes.decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("Digest domains must contain ASCII only.") from exc
    else:
        raise TypeError("Digest domain must be a string or bytes.")
    if not domain_bytes or b"\x00" in domain_bytes:
        raise ValueError("Digest domain must be non-empty and cannot contain NUL.")
    preimage = domain_bytes + b"\x00" + _u16(version) + canonical_encode(payload)
    return hashlib.sha256(preimage).hexdigest()


def validate_sha256(value: str, *, name: str = "digest") -> str:
    """Validate and return one lowercase, 64-character SHA256 string."""

    if not isinstance(name, str) or not name:
        raise ValueError("Digest validation name must be a non-empty string.")
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def digest_bytes(value: str, *, name: str = "digest") -> bytes:
    """Decode a validated public SHA256 representation to its 32 raw bytes."""

    return bytes.fromhex(validate_sha256(value, name=name))


def normalize_positive_zero(
    values: Any,
    *,
    dtype: Any | None = None,
    name: str = "values",
) -> Array:
    """Return a finite C-order array with floating signed zeros normalized.

    ``dtype`` is explicit at schema boundaries.  If it is omitted, the input
    dtype is retained.  This helper supports numerical construction as well
    as byte encoding, so it accepts the five canonical ndarray dtypes only.
    """

    try:
        raw = np.asarray(values)
        selected = raw.dtype if dtype is None else np.dtype(dtype)
        array = np.array(raw, dtype=selected, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} cannot be represented by the declared dtype.") from exc
    _, canonical = _canonical_array(array)
    canonical.setflags(write=False)
    return canonical


def _exact_float64_array(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must be a real numerical array.")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values.")
    converted = np.array(raw, dtype=np.float64, order="C", copy=True)
    try:
        roundtrip = converted.astype(raw.dtype, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot be converted exactly to float64.") from exc
    if not np.array_equal(roundtrip, raw):
        raise ValueError(f"{name} conversion to float64 changes a numerical value.")
    converted[converted == 0.0] = 0.0
    converted = converted.astype("<f8", copy=False)
    converted.setflags(write=False)
    return converted


def _exact_int64_array(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError(f"{name} must contain integers.")
    if raw.size:
        minimum = int(np.min(raw))
        maximum = int(np.max(raw))
        if minimum < _I64_MIN or maximum > _I64_MAX:
            raise ValueError(f"{name} contains an integer outside int64 range.")
    converted = np.array(raw, dtype="<i8", order="C", copy=True)
    if not np.array_equal(converted, raw):
        raise ValueError(f"{name} conversion to int64 changes a numerical value.")
    converted.setflags(write=False)
    return converted


def primitive_crystal_content_sha256(mapping: CrystalMapping) -> str:
    """Hash the common primitive crystal shared by FC2 and FC3 mappings."""

    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping.")
    if not validate_crystal_mapping(mapping).passed:
        raise ValueError("mapping must pass crystal-mapping validation.")

    positions = _exact_float64_array(
        mapping.primitive.scaled_positions,
        name="primitive scaled positions",
    ).copy()
    positions = np.mod(positions, 1.0)
    snap = 1.0e-10
    positions[(np.abs(positions) <= snap) | (np.abs(positions - 1.0) <= snap)] = 0.0
    positions[positions == 0.0] = 0.0
    positions = np.array(positions, dtype="<f8", order="C", copy=True)

    payload = {
        "schema_version": "negfpy.primitive-crystal-content.v1",
        "length_unit": "angstrom",
        "mass_unit": "amu",
        "primitive_matrix": _exact_float64_array(
            mapping.primitive_matrix,
            name="primitive matrix",
        ),
        "lattice": _exact_float64_array(
            mapping.primitive.lattice,
            name="primitive lattice",
        ),
        "scaled_positions_modulo_one": positions,
        "masses": _exact_float64_array(
            mapping.primitive.masses,
            name="primitive masses",
        ),
        "symbols": tuple(mapping.primitive.symbols),
    }
    return domain_digest("NEGFPY-PRIMITIVE-CRYSTAL-CONTENT", payload)


def crystal_mapping_content_sha256(mapping: CrystalMapping) -> str:
    """Hash one purpose-specific immutable crystal mapping."""

    if not isinstance(mapping, CrystalMapping):
        raise TypeError("mapping must be a CrystalMapping.")
    if not validate_crystal_mapping(mapping).passed:
        raise ValueError("mapping must pass crystal-mapping validation.")
    primitive_hash = primitive_crystal_content_sha256(mapping)
    payload = {
        "schema_version": "negfpy.crystal-mapping-content.v1",
        "purpose": mapping.purpose,
        "primitive_crystal_content_sha256": digest_bytes(
            primitive_hash,
            name="primitive_crystal_content_sha256",
        ),
        "supercell_matrix": _exact_int64_array(
            mapping.supercell_matrix,
            name="supercell matrix",
        ),
        "supercell": {
            "lattice": _exact_float64_array(
                mapping.supercell.lattice,
                name="supercell lattice",
            ),
            "scaled_positions": _exact_float64_array(
                mapping.supercell.scaled_positions,
                name="supercell scaled positions",
            ),
            "masses": _exact_float64_array(
                mapping.supercell.masses,
                name="supercell masses",
            ),
            "symbols": tuple(mapping.supercell.symbols),
        },
        "p2s": _exact_int64_array(mapping.p2s, name="p2s"),
        "s2p": _exact_int64_array(mapping.s2p, name="s2p"),
        "p2p": _exact_int64_array(mapping.p2p, name="p2p"),
        "shortest_vectors": _exact_float64_array(
            mapping.shortest_vectors,
            name="shortest vectors",
        ),
        "multiplicity": _exact_int64_array(
            mapping.multiplicity,
            name="multiplicity",
        ),
        "atomic_permutations": _exact_int64_array(
            mapping.atomic_permutations,
            name="atomic permutations",
        ),
        "translation_permutations": (
            None
            if mapping.translation_permutations is None
            else _exact_int64_array(
                mapping.translation_permutations,
                name="translation permutations",
            )
        ),
    }
    return domain_digest("NEGFPY-CRYSTAL-MAPPING-CONTENT", payload)


def fc2_content_sha256(fc2: FC2Data) -> str:
    """Return the authoritative canonical FC2 physics-content identity."""

    if not isinstance(fc2, FC2Data):
        raise TypeError("fc2 must be an FC2Data instance.")
    report = validate_fc2(fc2)
    if not report.passed:
        raise ValueError("fc2 must pass FC2 schema validation.")
    values = _exact_float64_array(fc2.values, name="FC2 values")
    masses = _exact_float64_array(fc2.masses, name="FC2 masses")
    p2s = (
        None
        if fc2.p2s_map is None
        else _exact_int64_array(fc2.p2s_map, name="FC2 p2s map")
    )
    s2p = (
        None
        if fc2.s2p_map is None
        else _exact_int64_array(fc2.s2p_map, name="FC2 s2p map")
    )
    preimage = b"".join(
        (
            b"NEGFPY-FC2-CONTENT\x00",
            _u16(1),
            canonical_encode(values),
            canonical_encode(masses),
            canonical_encode(bool(fc2.compact)),
            canonical_encode(p2s),
            canonical_encode(s2p),
            canonical_encode(fc2.units),
            canonical_encode(fc2.mass_units),
            canonical_encode(fc2.version),
            canonical_encode(fc2.cutoff),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def fc3_content_sha256(fc3: FC3Data) -> str:
    """Return the authoritative canonical raw-FC3 content identity.

    The identity covers every first-class field that can affect FC3
    interpretation or a later physical conversion.  Filesystem paths and
    descriptive HDF5 provenance are deliberately excluded; the source
    container hash remains a separate identity.
    """

    if not isinstance(fc3, FC3Data):
        raise TypeError("fc3 must be an FC3Data instance.")
    values = _exact_float64_array(fc3.values, name="FC3 values")
    masses = _exact_float64_array(fc3.masses, name="FC3 masses")
    p2s = (
        None
        if fc3.p2s_map is None
        else _exact_int64_array(fc3.p2s_map, name="FC3 p2s map")
    )
    s2p = (
        None
        if fc3.s2p_map is None
        else _exact_int64_array(fc3.s2p_map, name="FC3 s2p map")
    )
    nonzero_indices = (
        None
        if fc3.fc3_nonzero_indices is None
        else _exact_int64_array(
            fc3.fc3_nonzero_indices,
            name="FC3 nonzero indices",
        )
    )
    preimage = b"".join(
        (
            b"NEGFPY-FC3-CONTENT\x00",
            _u16(1),
            canonical_encode(values),
            canonical_encode(masses),
            canonical_encode(bool(fc3.compact)),
            canonical_encode(p2s),
            canonical_encode(s2p),
            canonical_encode(fc3.units),
            canonical_encode(fc3.mass_units),
            canonical_encode(fc3.version),
            canonical_encode(nonzero_indices),
            canonical_encode(fc3.fc3_cutoff),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


_FC2_PROVENANCE_KEYS = (
    "format",
    "source_sha256",
    "source_dataset",
    "source_dataset_shape",
    "source_dataset_dtype",
    "source_dataset_attrs",
    "source_file_attrs",
    "source_auxiliary_dataset_attrs",
    "source_dataset_inventory",
    "source_physical_unit",
    "source_version",
    "source_cutoff",
    "units_source",
    "version_source",
    "cutoff_source",
    "layout_source",
    "p2s_map_source",
    "s2p_map_source",
)


def fc2_provenance_sha256(fc2: FC2Data) -> str:
    """Hash the portable projection of verified FC2 reader provenance."""

    if not isinstance(fc2, FC2Data):
        raise TypeError("fc2 must be an FC2Data instance.")
    provenance = fc2.provenance
    missing = tuple(key for key in _FC2_PROVENANCE_KEYS if key not in provenance)
    if missing:
        raise ValueError(f"FC2 provenance is missing required keys: {missing!r}.")
    source_hash = validate_sha256(
        provenance["source_sha256"],
        name="FC2 provenance source_sha256",
    )
    if "sha256" not in provenance or provenance["sha256"] != source_hash:
        raise ValueError("FC2 provenance sha256 does not equal source_sha256.")
    if "dataset_attrs" not in provenance:
        raise ValueError("FC2 provenance is missing dataset_attrs.")
    if canonical_encode(provenance["dataset_attrs"]) != canonical_encode(
        provenance["source_dataset_attrs"]
    ):
        raise ValueError(
            "FC2 provenance dataset_attrs does not equal source_dataset_attrs."
        )

    payload = {key: provenance[key] for key in _FC2_PROVENANCE_KEYS}
    payload["schema_version"] = "negfpy.fc2-provenance.v1"
    payload["source_sha256"] = digest_bytes(
        source_hash,
        name="FC2 provenance source_sha256",
    )
    return domain_digest("NEGFPY-FC2-PROVENANCE", payload)


def _integer_matrix(
    values: Any,
    *,
    name: str = "matrix",
) -> tuple[Array, list[list[int]]]:
    array = _exact_int64_array(values, name=name)
    if array.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3).")
    python_values = [[int(array[i, j]) for j in range(3)] for i in range(3)]
    return array, python_values


def _determinant_and_adjugate(
    matrix: Sequence[Sequence[int]],
) -> tuple[int, list[list[int]]]:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (
        d * h - e * g
    )
    adjugate = [
        [e * i - f * h, c * h - b * i, b * f - c * e],
        [f * g - d * i, a * i - c * g, c * d - a * f],
        [d * h - e * g, b * g - a * h, a * e - b * d],
    ]
    return determinant, adjugate


def _coset_key(
    representative: Sequence[int],
    adjugate: Sequence[Sequence[int]],
    modulus: int,
) -> tuple[int, int, int]:
    return tuple(
        sum(int(representative[row]) * int(adjugate[row][column]) for row in range(3))
        % modulus
        for column in range(3)
    )  # type: ignore[return-value]


def _enumerate_integer_cosets(matrix: Any) -> tuple[Array, Array]:
    """Enumerate canonical representatives of ``Z^3 / (Z^3 matrix)``.

    All determinant, adjugate, key, and equivalence operations use Python
    arbitrary-precision integers.  The returned records are lexicographically
    sorted by exact quotient key and stored as read-only little-endian int64.
    """

    _, exact = _integer_matrix(matrix)
    determinant, adjugate = _determinant_and_adjugate(exact)
    if determinant == 0:
        raise ValueError("Coset matrix must be nonsingular.")
    if determinant < _I64_MIN or determinant > _I64_MAX:
        raise ValueError("Coset determinant cannot be represented as int64.")
    modulus = abs(determinant)
    records: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    safety_bound = max(modulus, 1)

    for shell in range(safety_bound + 1):
        candidates = (
            candidate
            for candidate in itertools.product(range(-shell, shell + 1), repeat=3)
            if max(abs(value) for value in candidate) == shell
        )
        for representative in sorted(
            candidates,
            key=lambda item: (
                sum(abs(value) for value in item),
                item[0],
                item[1],
                item[2],
            ),
        ):
            key = _coset_key(representative, adjugate, modulus)
            records.setdefault(key, representative)
        if len(records) == modulus:
            break
        if shell == safety_bound:
            raise RuntimeError(
                "Integer-coset enumeration reached its deterministic safety "
                "bound without complete cardinality."
            )
    if len(records) != modulus:
        raise RuntimeError("Integer-coset enumeration has incomplete cardinality.")

    ordered = sorted(records.items())
    keys = np.array([key for key, _ in ordered], dtype="<i8", order="C")
    representatives = np.array(
        [representative for _, representative in ordered],
        dtype="<i8",
        order="C",
    )
    if keys.shape != (modulus, 3) or representatives.shape != (modulus, 3):
        raise RuntimeError("Integer-coset enumeration produced an invalid shape.")
    if not any(np.array_equal(key, np.zeros(3, dtype=np.int64)) for key in keys):
        raise RuntimeError("Integer-coset enumeration omitted the origin class.")
    origin = int(np.flatnonzero(np.all(keys == 0, axis=1))[0])
    if not bool(np.all(representatives[origin] == 0)):
        raise RuntimeError("The canonical origin class is not represented by zero.")
    for key, representative in zip(keys, representatives, strict=True):
        if tuple(int(value) for value in key) != _coset_key(
            tuple(int(value) for value in representative),
            adjugate,
            modulus,
        ):
            raise RuntimeError("A retained representative does not reproduce its key.")
    keys.setflags(write=False)
    representatives.setflags(write=False)
    return keys, representatives


def _coset_records(
    matrix: Any,
    keys: Any,
    representatives: Any,
) -> tuple[Array, int, Array]:
    matrix_array, exact = _integer_matrix(matrix)
    determinant, adjugate = _determinant_and_adjugate(exact)
    if determinant == 0:
        raise ValueError("Coset matrix must be nonsingular.")
    if not _I64_MIN <= determinant <= _I64_MAX:
        raise ValueError("Coset determinant cannot be represented as int64.")
    modulus = abs(determinant)
    key_array = _exact_int64_array(keys, name="coset keys")
    representative_array = _exact_int64_array(
        representatives,
        name="coset representatives",
    )
    if key_array.shape != (modulus, 3):
        raise ValueError("coset keys must have shape (abs(det(matrix)), 3).")
    if representative_array.shape != (modulus, 3):
        raise ValueError(
            "coset representatives must have shape (abs(det(matrix)), 3)."
        )

    normalized_records: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw_key, raw_representative in zip(
        key_array,
        representative_array,
        strict=True,
    ):
        key = tuple(int(value) for value in raw_key)
        representative = tuple(int(value) for value in raw_representative)
        if any(value < 0 or value >= modulus for value in key):
            raise ValueError("Coset key component is outside its Euclidean range.")
        if key in seen:
            raise ValueError("Coset keys are not pairwise inequivalent.")
        seen.add(key)
        if _coset_key(representative, adjugate, modulus) != key:
            raise ValueError("Coset representative does not reproduce its stored key.")
        normalized_records.append((key, representative))
    if len(seen) != modulus or (0, 0, 0) not in seen:
        raise ValueError("Coset records do not contain every quotient class.")
    normalized_records.sort(key=lambda record: record[0])
    records = np.array(
        [key + representative for key, representative in normalized_records],
        dtype="<i8",
        order="C",
    )
    records.setflags(write=False)
    return matrix_array, determinant, records


def row_cosets_sha256(
    matrix: Any,
    keys: Any,
    representatives: Any,
    *,
    role: CosetRole,
) -> str:
    """Hash a canonical direct (``D``) or reciprocal (``R``) row-coset table.

    The custom v1 stream is the domain and role bytes, the two-dimensional
    matrix shape and bytes, the signed determinant, then the two-dimensional
    sorted record-table shape and interleaved ``(key, representative)`` bytes.
    All shape fields are little-endian uint64 and all integer data are
    little-endian signed int64.
    """

    if role not in {"D", "R"}:
        raise ValueError("Coset role must be exactly 'D' or 'R'.")
    matrix_array, determinant, records = _coset_records(
        matrix,
        keys,
        representatives,
    )
    preimage = b"".join(
        (
            b"negfpy-row-cosets-v1\x00",
            role.encode("ascii"),
            _u64(int(matrix_array.shape[0])),
            _u64(int(matrix_array.shape[1])),
            matrix_array.astype("<i8", copy=False).tobytes(order="C"),
            _i64(determinant),
            _u64(int(records.shape[0])),
            _u64(int(records.shape[1])),
            records.tobytes(order="C"),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


@dataclass(frozen=True, slots=True, eq=False)
class _IntegerCosetTable:
    """Immutable exact row-quotient table with its custom stream identity."""

    matrix: Array
    signed_determinant: int
    role: CosetRole
    keys: Array
    representatives: Array
    sha256: str

    def __post_init__(self) -> None:
        matrix, determinant, records = _coset_records(
            self.matrix,
            self.keys,
            self.representatives,
        )
        if self.signed_determinant != determinant:
            raise ValueError("signed_determinant does not match matrix.")
        observed = row_cosets_sha256(
            matrix,
            records[:, :3],
            records[:, 3:],
            role=self.role,
        )
        if validate_sha256(self.sha256, name="coset table sha256") != observed:
            raise ValueError("Coset-table digest does not reproduce.")
        keys = np.array(records[:, :3], dtype="<i8", order="C", copy=True)
        representatives = np.array(
            records[:, 3:],
            dtype="<i8",
            order="C",
            copy=True,
        )
        matrix = np.array(matrix, dtype="<i8", order="C", copy=True)
        for value in (matrix, keys, representatives):
            value.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "signed_determinant", determinant)
        object.__setattr__(self, "keys", keys)
        object.__setattr__(self, "representatives", representatives)
        object.__setattr__(self, "sha256", observed)


def build_row_coset_table(matrix: Any, *, role: CosetRole) -> _IntegerCosetTable:
    """Build and self-validate one exact row-coset table."""

    matrix_array, exact = _integer_matrix(matrix)
    determinant, _ = _determinant_and_adjugate(exact)
    keys, representatives = _enumerate_integer_cosets(matrix_array)
    digest = row_cosets_sha256(
        matrix_array,
        keys,
        representatives,
        role=role,
    )
    return _IntegerCosetTable(
        matrix=matrix_array,
        signed_determinant=determinant,
        role=role,
        keys=keys,
        representatives=representatives,
        sha256=digest,
    )


def enumerate_direct_reciprocal_cosets(
    transport_supercell_matrix: Any,
) -> tuple[_IntegerCosetTable, _IntegerCosetTable]:
    """Build direct ``S`` and independently enumerated reciprocal ``S.T`` tables."""

    matrix, _ = _integer_matrix(
        transport_supercell_matrix,
        name="transport supercell matrix",
    )
    direct = build_row_coset_table(matrix, role="D")
    reciprocal = build_row_coset_table(matrix.T, role="R")
    return direct, reciprocal


__all__ = [
    "_IntegerCosetTable",
    "_enumerate_integer_cosets",
    "build_row_coset_table",
    "canonical_encode",
    "crystal_mapping_content_sha256",
    "digest_bytes",
    "domain_digest",
    "enumerate_direct_reciprocal_cosets",
    "fc2_content_sha256",
    "fc2_provenance_sha256",
    "fc3_content_sha256",
    "normalize_positive_zero",
    "primitive_crystal_content_sha256",
    "row_cosets_sha256",
    "validate_sha256",
]
