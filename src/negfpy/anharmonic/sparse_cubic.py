"""Exact sparse cubic vertices and correctness-reference contractions.

The sparse object stores the finite cubic Hamiltonian without a support
threshold.  It deliberately has no ``values`` property that could silently
materialize a dense :math:`D^3` tensor.  Conversion and contraction are
explicit, memory-guarded operations.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import math
import os
from itertools import permutations
import struct
import sys
from types import MappingProxyType
from typing import Any

import numpy as np

from ._canonical import canonical_encode, digest_bytes, domain_digest
from .grid import SignedFrequencyGrid
from .types import DenseCubicVertex


Array = np.ndarray


def _readonly(values: Any, *, dtype: Any, name: str) -> Array:
    try:
        result = np.array(values, dtype=dtype, order="C", copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} cannot be represented with the required dtype.") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    if result.dtype.kind in "fc":
        result[result == 0] = 0
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True, order="C")
        if result.dtype.kind in "fc":
            result[result == 0] = 0
        result.setflags(write=False)
        return result
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _available_memory_bytes() -> int | None:
    try:
        pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        size = int(os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, ValueError):
        return None
    return pages * size if pages > 0 and size > 0 else None


def _physics_sha256(ndof: int, indices: Array, values: Array) -> str:
    payload = [
        b"negfpy-cubic-physics-v1\x00",
        struct.pack("<Q", ndof),
        struct.pack("<Q", len(values)),
    ]
    for index, value in zip(indices, values, strict=True):
        payload.append(np.asarray(index, dtype="<i8").tobytes())
        number = 0.0 if float(value) == 0.0 else float(value)
        payload.append(struct.pack("<d", number))
    return hashlib.sha256(b"".join(payload)).hexdigest()


def _storage_sha256(
    ndof: int,
    indices: Array,
    values: Array,
    *,
    physics_sha256: str,
    metadata_sha256: str,
) -> str:
    preimage = b"".join(
        (
            b"NEGFPY-CUBIC-STORAGE\x00",
            struct.pack("<H", 1),
            struct.pack("<Q", ndof),
            canonical_encode("sparse"),
            digest_bytes(physics_sha256),
            canonical_encode(np.asarray(indices, dtype="<i8")),
            canonical_encode(np.asarray(values, dtype="<f8")),
            digest_bytes(metadata_sha256),
        )
    )
    return hashlib.sha256(preimage).hexdigest()


def _validate_exact_permutation_symmetry(indices: Array, values: Array) -> None:
    lookup = {
        tuple(int(component) for component in index): np.float64(value).tobytes()
        for index, value in zip(indices, values, strict=True)
    }
    for index, value_bytes in lookup.items():
        for order in permutations(index):
            if order not in lookup:
                raise ValueError(
                    "Every distinct ordered permutation of a sparse cubic "
                    f"component must be present; missing {order}."
                )
            if lookup[order] != value_bytes:
                raise ValueError(
                    "Permutation-orbit coefficients must be bitwise equal; "
                    "implicit symmetrization is forbidden."
                )


@dataclass(frozen=True, slots=True, eq=False)
class SparseCubicVertex:
    """Unique lexicographically sorted cubic components in solver units."""

    ndof: int
    indices: Array
    coefficients: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)
    cubic_physics_sha256: str = field(init=False)
    metadata_sha256: str = field(init=False)
    vertex_storage_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.ndof, (bool, np.bool_)) or not isinstance(self.ndof, (int, np.integer)):
            raise TypeError("ndof must be an integer.")
        ndof = int(self.ndof)
        if ndof < 1:
            raise ValueError("ndof must be at least one.")
        raw_indices = np.asarray(self.indices)
        if raw_indices.dtype.kind not in "iu" or raw_indices.dtype.kind == "b":
            raise TypeError("indices must contain integers without implicit conversion.")
        indices = _readonly(raw_indices, dtype=np.dtype("<i8"), name="indices")
        coefficients = _readonly(self.coefficients, dtype=np.dtype("<f8"), name="coefficients")
        if indices.ndim != 2 or indices.shape[1:] != (3,):
            raise ValueError("indices must have shape (nnz, 3).")
        if coefficients.shape != (indices.shape[0],):
            raise ValueError("coefficients must have shape (nnz,).")
        if np.any(indices < 0) or np.any(indices >= ndof):
            raise ValueError("indices contain a degree of freedom outside [0, ndof).")
        if np.any(coefficients == 0.0):
            raise ValueError("Stored sparse coefficients must be exactly nonzero.")
        if len(indices):
            order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0]))
            if not np.array_equal(order, np.arange(len(indices))):
                raise ValueError("indices must be lexicographically sorted.")
            if len(indices) > 1 and np.any(np.all(indices[1:] == indices[:-1], axis=1)):
                raise ValueError("indices must be unique.")
        _validate_exact_permutation_symmetry(indices, coefficients)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        metadata = _freeze(self.metadata)
        metadata_digest = domain_digest("NEGFPY-METADATA", metadata)
        physics_digest = _physics_sha256(ndof, indices, coefficients)
        storage_digest = _storage_sha256(
            ndof,
            indices,
            coefficients,
            physics_sha256=physics_digest,
            metadata_sha256=metadata_digest,
        )
        object.__setattr__(self, "ndof", ndof)
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "coefficients", coefficients)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "cubic_physics_sha256", physics_digest)
        object.__setattr__(self, "metadata_sha256", metadata_digest)
        object.__setattr__(self, "vertex_storage_sha256", storage_digest)

    @property
    def nnz(self) -> int:
        return int(len(self.coefficients))

    @property
    def support_density(self) -> float:
        return float(self.nnz / self.ndof**3)

    @property
    def max_abs(self) -> float:
        return float(np.max(np.abs(self.coefficients))) if self.nnz else 0.0

    @property
    def is_zero(self) -> bool:
        return self.nnz == 0

    def to_dense(
        self,
        *,
        max_bytes: int,
        allow_memory_overcommit: bool = False,
    ) -> DenseCubicVertex:
        """Materialize a dense oracle only after both memory guards pass."""

        if isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, (int, np.integer)):
            raise TypeError("max_bytes must be an integer.")
        if int(max_bytes) < 1:
            raise ValueError("max_bytes must be at least one.")
        if not isinstance(allow_memory_overcommit, bool):
            raise TypeError("allow_memory_overcommit must be a literal boolean.")
        element_count = self.ndof**3
        payload_bytes = 8 * element_count
        if element_count > (1 << 64) - 1 or payload_bytes > min((1 << 64) - 1, sys.maxsize):
            raise OverflowError("Dense cubic payload exceeds addressable size.")
        required_peak_bytes = 3 * payload_bytes
        if required_peak_bytes > int(max_bytes):
            raise MemoryError(
                f"Dense conversion requires {required_peak_bytes} peak bytes, "
                f"exceeding max_bytes={int(max_bytes)}."
            )
        available = _available_memory_bytes()
        if available is None and not allow_memory_overcommit:
            raise MemoryError("Available RAM cannot be determined; refusing dense conversion.")
        if available is not None and required_peak_bytes > 0.70 * available and not allow_memory_overcommit:
            raise MemoryError("Dense conversion exceeds 70% of currently available RAM.")
        dense = np.zeros((self.ndof, self.ndof, self.ndof), dtype=np.float64)
        if self.nnz:
            dense[tuple(self.indices.T)] = self.coefficients
        metadata = dict(self.metadata)
        metadata["sparse_conversion"] = {
            "source_vertex_storage_sha256": self.vertex_storage_sha256,
            "cubic_physics_sha256": self.cubic_physics_sha256,
            "max_bytes": int(max_bytes),
            "required_peak_bytes": required_peak_bytes,
            "allow_memory_overcommit": allow_memory_overcommit,
        }
        result = DenseCubicVertex(values=dense, metadata=metadata)
        return result

    @classmethod
    def from_dense(cls, vertex: DenseCubicVertex) -> "SparseCubicVertex":
        """Create an exact sparse oracle from a dense vertex without thresholding."""

        if not isinstance(vertex, DenseCubicVertex):
            raise TypeError("vertex must be a DenseCubicVertex.")
        indices = np.argwhere(vertex.values != 0.0).astype(np.int64)
        values = vertex.values[tuple(indices.T)] if len(indices) else np.empty(0, dtype=np.float64)
        metadata = dict(vertex.metadata)
        metadata["conversion"] = "exact_dense_nonzero_support"
        return cls(vertex.ndof, indices, values, metadata)


def cubic_vertex_is_zero(vertex: DenseCubicVertex | SparseCubicVertex) -> bool:
    if isinstance(vertex, SparseCubicVertex):
        return vertex.is_zero
    if isinstance(vertex, DenseCubicVertex):
        return not bool(np.any(vertex.values))
    raise TypeError("vertex must be DenseCubicVertex or SparseCubicVertex.")


def cubic_vertex_max_abs(vertex: DenseCubicVertex | SparseCubicVertex) -> float:
    if isinstance(vertex, SparseCubicVertex):
        return vertex.max_abs
    if isinstance(vertex, DenseCubicVertex):
        return float(np.max(np.abs(vertex.values)))
    raise TypeError("vertex must be DenseCubicVertex or SparseCubicVertex.")


def cubic_vertex_physics_sha256(vertex: DenseCubicVertex | SparseCubicVertex) -> str:
    if isinstance(vertex, SparseCubicVertex):
        return vertex.cubic_physics_sha256
    if not isinstance(vertex, DenseCubicVertex):
        raise TypeError("vertex must be DenseCubicVertex or SparseCubicVertex.")
    indices = np.argwhere(vertex.values != 0.0).astype(np.int64)
    values = vertex.values[tuple(indices.T)] if len(indices) else np.empty(0, dtype=np.float64)
    return _physics_sha256(vertex.ndof, indices, np.asarray(values, dtype=np.float64))


def _validated_green(values: Any, *, grid: SignedFrequencyGrid, ndof: int, name: str) -> Array:
    array = np.asarray(values, dtype=np.complex128)
    expected = (grid.n_points, ndof, ndof)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    return array


def _first_leg_matrices(vertex: SparseCubicVertex) -> tuple[Array, Array]:
    active = np.unique(vertex.indices[:, 0]) if vertex.nnz else np.empty(0, dtype=np.int64)
    matrices = np.zeros((len(active), vertex.ndof, vertex.ndof), dtype=np.float64)
    lookup = {int(index): position for position, index in enumerate(active)}
    for (a, c, d), coefficient in zip(vertex.indices, vertex.coefficients, strict=True):
        matrices[lookup[int(a)], int(c), int(d)] = coefficient
    return active, matrices


def _sparse_pair_contraction(vertex: SparseCubicVertex, first: Array, second: Array) -> Array:
    """Evaluate one frequency pair using active first-leg slices.

    This grouped reference avoids a rank-six tensor and keeps inactive output
    rows exactly zero.  It is intended as the storage-correct oracle; a later
    profiled schedule may optimize the same algebra without changing support.
    """

    active, matrices = _first_leg_matrices(vertex)
    result = np.zeros((vertex.ndof, vertex.ndof), dtype=np.complex128)
    if not len(active):
        return result
    transformed = np.einsum("ce,acd,df->aef", first, matrices, second, optimize=True)
    block = np.einsum("aef,bef->ab", transformed, matrices, optimize=True)
    result[np.ix_(active, active)] = block
    return result


def _sparse_bubble_direct(vertex: SparseCubicVertex, green: Array, grid: SignedFrequencyGrid) -> Array:
    result = np.zeros((grid.n_points, vertex.ndof, vertex.ndof), dtype=np.complex128)
    zero = grid.zero_index
    for output in range(grid.n_points):
        start = max(0, output + zero - (grid.n_points - 1))
        stop = min(grid.n_points - 1, output + zero)
        accumulated = np.zeros((vertex.ndof, vertex.ndof), dtype=np.complex128)
        for first_index in range(start, stop + 1):
            second_index = output - first_index + zero
            accumulated += _sparse_pair_contraction(vertex, green[first_index], green[second_index])
        result[output] = 0.5j * grid.integration_weight * accumulated
    return result


def _sparse_bubble_fft(vertex: SparseCubicVertex, green: Array, grid: SignedFrequencyGrid) -> Array:
    n_frequency = grid.n_points
    padded = 1 << (2 * n_frequency - 2).bit_length()
    transformed_green = np.fft.fft(green, n=padded, axis=0)
    transformed_bubble = np.empty((padded, vertex.ndof, vertex.ndof), dtype=np.complex128)
    for index in range(padded):
        transformed_bubble[index] = _sparse_pair_contraction(
            vertex,
            transformed_green[index],
            transformed_green[index],
        )
    full = np.fft.ifft(transformed_bubble, axis=0)
    start = grid.zero_index
    return 0.5j * grid.integration_weight * full[start : start + n_frequency]


def cubic_sparse_lesser_greater(
    vertex: SparseCubicVertex,
    green_lesser: Array,
    green_greater: Array,
    grid: SignedFrequencyGrid,
    *,
    backend: str = "direct",
) -> tuple[Array, Array]:
    """Return sparse cubic lesser/greater self-energies.

    ``direct`` is the deterministic oracle. ``fft`` is the exactly
    zero-padded linear-convolution implementation and never wraps the cutoff.
    """

    if not isinstance(vertex, SparseCubicVertex):
        raise TypeError("vertex must be a SparseCubicVertex.")
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if backend not in {"direct", "fft"}:
        raise ValueError("backend must be 'direct' or 'fft'.")
    lesser = _validated_green(green_lesser, grid=grid, ndof=vertex.ndof, name="green_lesser")
    greater = _validated_green(green_greater, grid=grid, ndof=vertex.ndof, name="green_greater")
    operation = _sparse_bubble_direct if backend == "direct" else _sparse_bubble_fft
    return operation(vertex, lesser, grid), operation(vertex, greater, grid)


__all__ = [
    "SparseCubicVertex",
    "cubic_sparse_lesser_greater",
    "cubic_vertex_is_zero",
    "cubic_vertex_max_abs",
    "cubic_vertex_physics_sha256",
]
