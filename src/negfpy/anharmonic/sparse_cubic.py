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
from typing import Any, ClassVar

import numpy as np
from scipy import sparse as scipy_sparse

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
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) >= 2 and fields[0] == "MemAvailable:":
                    kilobytes = int(fields[1])
                    if kilobytes > 0:
                        return kilobytes * 1024
                    break
    except (OSError, ValueError):
        pass
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


def _permutation_symmetry_audit(
    indices: Array,
    values: Array,
    *,
    rtol: float,
    atol: float,
    cubic_physics_sha256: str,
) -> Mapping[str, Any]:
    """Validate complete support and numerical symmetry without correction.

    The comparison is the finite-value form of ``numpy.isclose`` used by
    :class:`DenseCubicVertex`: ``abs(a-b) <= atol + rtol*abs(b)``.  Both
    directions are checked because every stored ordering is visited.  Exact
    equality remains an audit result, never a reason to rewrite a coefficient.
    """

    lookup = {
        tuple(int(component) for component in index): position
        for position, index in enumerate(indices)
    }
    maximum = 0.0
    exact = True
    comparison_count = 0
    first_failure: (
        tuple[tuple[int, int, int], tuple[int, int, int], float, float]
        | None
    ) = None
    for index, position in lookup.items():
        reference = float(values[position])
        reference_bytes = np.float64(reference).tobytes()
        for raw_order in sorted(set(permutations(index))):
            order = tuple(int(component) for component in raw_order)
            observed_position = lookup.get(order)
            if observed_position is None:
                raise ValueError(
                    "Every distinct ordered permutation of a sparse cubic "
                    f"component must be present; missing {order}."
                )
            observed = float(values[observed_position])
            difference = abs(reference - observed)
            allowed = atol + rtol * abs(observed)
            maximum = max(maximum, difference)
            comparison_count += 1
            if np.float64(observed).tobytes() != reference_bytes:
                exact = False
            if difference > allowed and first_failure is None:
                first_failure = (index, order, difference, allowed)

    if first_failure is not None:
        index, order, difference, allowed = first_failure
        raise ValueError(
            "Permutation-orbit coefficients must be numerically permutation "
            "symmetric without correction; "
            f"{index} versus {order} differs by {difference:.17g}, exceeding "
            f"{allowed:.17g} for rtol={rtol:.17g}, atol={atol:.17g}."
        )

    scale = float(np.max(np.abs(values), initial=0.0))
    relative = maximum / scale if scale else 0.0
    return {
        "schema_version": "negfpy.sparse-cubic-permutation-audit.v1",
        "cubic_physics_sha256": cubic_physics_sha256,
        "support_complete": True,
        "permutation_tolerance_passed": True,
        "exact_permutation_equality": exact,
        "symmetry_rtol": rtol,
        "symmetry_atol": atol,
        "permutation_max_abs": maximum,
        "coefficient_scale": scale,
        "permutation_relative_to_global_scale": relative,
        "checked_ordered_comparison_count": comparison_count,
        "comparison_rule": "abs(a-b) <= atol + rtol*abs(b), both directions",
        "symmetrized": False,
        "corrected": False,
        "coefficients_modified": False,
    }


@dataclass(frozen=True, slots=True, eq=False)
class SparseCubicVertex:
    """Unique sparse cubic components in solver units, never corrected.

    Every ordered permutation must be stored.  Coefficients use the same
    scale-aware numerical symmetry contract as :class:`DenseCubicVertex`;
    last-bit differences are retained and reported rather than averaged.
    """

    ndof: int
    indices: Array
    coefficients: Array
    metadata: Mapping[str, Any] = field(default_factory=dict)
    permutation_symmetry: Mapping[str, Any] = field(init=False)
    cubic_physics_sha256: str = field(init=False)
    metadata_sha256: str = field(init=False)
    vertex_storage_sha256: str = field(init=False)

    symmetry_rtol: ClassVar[float] = DenseCubicVertex.symmetry_rtol
    symmetry_atol: ClassVar[float] = DenseCubicVertex.symmetry_atol

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
        physics_digest = _physics_sha256(ndof, indices, coefficients)
        symmetry_audit = _permutation_symmetry_audit(
            indices,
            coefficients,
            rtol=self.symmetry_rtol,
            atol=self.symmetry_atol,
            cubic_physics_sha256=physics_digest,
        )
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        metadata = _freeze(self.metadata)
        declared_audit = metadata.get("permutation_symmetry")
        if declared_audit is not None:
            if not isinstance(declared_audit, Mapping):
                raise TypeError(
                    "metadata['permutation_symmetry'] must be a mapping."
                )
            declared_physics = declared_audit.get("cubic_physics_sha256")
            if (
                declared_physics == physics_digest
                and canonical_encode(declared_audit)
                != canonical_encode(symmetry_audit)
            ):
                raise ValueError(
                    "metadata['permutation_symmetry'] does not match the "
                    "coefficient audit."
                )
        enriched_metadata = dict(metadata)
        enriched_metadata["permutation_symmetry"] = symmetry_audit
        metadata = _freeze(enriched_metadata)
        frozen_audit = metadata["permutation_symmetry"]
        metadata_digest = domain_digest("NEGFPY-METADATA", metadata)
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
        object.__setattr__(self, "permutation_symmetry", frozen_audit)
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


@dataclass(frozen=True, slots=True)
class _SparseContractionSchedule:
    """Compiled active-basis slices used throughout one cubic evaluation.

    ``active`` maps the compact schedule basis back to device degrees of
    freedom.  Exact permutation symmetry guarantees that every second- and
    third-leg index is in this same set, so neither the local contractions nor
    the flattened CSR operator need an inactive ``D`` dimension.
    """

    ndof: int
    active: Array
    row_indices: tuple[Array, ...]
    column_indices: tuple[Array, ...]
    local_matrices: tuple[Array, ...]
    flattened_first_legs: scipy_sparse.csr_matrix

    @property
    def numeric_storage_bytes(self) -> int:
        """Return exact numeric payload bytes, excluding Python headers."""

        dense_arrays = (
            self.active,
            *self.row_indices,
            *self.column_indices,
            *self.local_matrices,
        )
        csr = self.flattened_first_legs
        return int(
            sum(array.nbytes for array in dense_arrays)
            + csr.data.nbytes
            + csr.indices.nbytes
            + csr.indptr.nbytes
        )

    @property
    def reconstructed_first_leg_bytes(self) -> int:
        """Payload formerly rebuilt as ``n_active x D x D`` per pair."""

        return int(
            len(self.active)
            * self.ndof**2
            * np.dtype(np.float64).itemsize
        )

    @property
    def pair_multiply_add_count(self) -> int:
        """Deterministic multiply-add model for one frequency pair.

        The count models the two dense local matrix products and the repeated
        CSR matvecs.  It intentionally excludes indexing, allocation, and FFT
        work, and counts dense zeros inside a local rectangular slice because
        BLAS evaluates those entries too.
        """

        active_count = len(self.active)
        local_products = sum(
            active_count * len(rows) * len(columns)
            + active_count**2 * min(len(rows), len(columns))
            for rows, columns in zip(
                self.row_indices,
                self.column_indices,
                strict=True,
            )
        )
        csr_products = active_count * self.flattened_first_legs.nnz
        return int(local_products + csr_products)


def _compiled_sparse_schedule(
    vertex: SparseCubicVertex,
) -> _SparseContractionSchedule:
    """Compile exact local slices once, without materializing ``D^3``.

    For one active first leg ``a``, only rows ``C_a`` and columns ``D_a`` of
    ``Psi[a, :, :]`` are stored.  The second contraction uses a CSR matrix of
    all first-leg slices in the compact active basis.  This preserves the
    exact algebra while avoiding the previous reconstruction of an
    ``n_active x D x D`` array for every frequency pair.
    """

    active = (
        np.unique(vertex.indices[:, 0])
        if vertex.nnz
        else np.empty(0, dtype=np.int64)
    )
    active = np.asarray(active, dtype=np.int64)
    active.setflags(write=False)
    row_indices: list[Array] = []
    column_indices: list[Array] = []
    local_matrices: list[Array] = []

    if vertex.nnz:
        # Permutation validation in ``SparseCubicVertex`` guarantees that all
        # three legs occur as first legs.  Search in the sorted active array is
        # therefore an exact, vectorized global-to-local mapping.
        compact_indices = np.searchsorted(active, vertex.indices).astype(
            np.int64,
            copy=False,
        )
        first_counts = np.bincount(
            compact_indices[:, 0],
            minlength=len(active),
        )
        first_offsets = np.empty(len(active) + 1, dtype=np.int64)
        first_offsets[0] = 0
        np.cumsum(first_counts, out=first_offsets[1:])
        flattened_columns = (
            compact_indices[:, 1] * len(active) + compact_indices[:, 2]
        )
        flattened = scipy_sparse.csr_matrix(
            (
                np.array(vertex.coefficients, dtype=np.float64, copy=True),
                flattened_columns,
                first_offsets,
            ),
            shape=(len(active), len(active) ** 2),
            dtype=np.float64,
            copy=True,
        )
        flattened.sort_indices()
    else:
        compact_indices = np.empty((0, 3), dtype=np.int64)
        first_offsets = np.zeros(1, dtype=np.int64)
        flattened = scipy_sparse.csr_matrix(
            (0, 0), dtype=np.float64
        )

    for first_position in range(len(active)):
        start = int(first_offsets[first_position])
        stop = int(first_offsets[first_position + 1])
        pairs = compact_indices[start:stop, 1:]
        coefficients = vertex.coefficients[start:stop]
        rows = np.unique(pairs[:, 0])
        columns = np.unique(pairs[:, 1])
        rows = np.asarray(rows, dtype=np.int64)
        columns = np.asarray(columns, dtype=np.int64)
        local = np.zeros((len(rows), len(columns)), dtype=np.float64)
        local[
            np.searchsorted(rows, pairs[:, 0]),
            np.searchsorted(columns, pairs[:, 1]),
        ] = coefficients
        for value in (rows, columns, local):
            value.setflags(write=False)
        row_indices.append(rows)
        column_indices.append(columns)
        local_matrices.append(local)

    # A frozen schedule must not retain mutable sparse payload arrays.
    for value in (flattened.data, flattened.indices, flattened.indptr):
        value.setflags(write=False)

    return _SparseContractionSchedule(
        ndof=vertex.ndof,
        active=active,
        row_indices=tuple(row_indices),
        column_indices=tuple(column_indices),
        local_matrices=tuple(local_matrices),
        flattened_first_legs=flattened,
    )


def _sparse_pair_contraction(
    vertex: SparseCubicVertex,
    schedule: _SparseContractionSchedule,
    first: Array,
    second: Array,
) -> Array:
    """Evaluate one frequency pair in the compact active-DOF basis.

    This grouped reference avoids a rank-six tensor.  Inactive output rows are
    inserted exactly once by the calling bubble operation.
    """

    if not len(schedule.active):
        return np.zeros((0, 0), dtype=np.complex128)
    block = np.empty(
        (len(schedule.active), len(schedule.active)), dtype=np.complex128
    )
    for output_row, (rows, columns, local) in enumerate(
        zip(
            schedule.row_indices,
            schedule.column_indices,
            schedule.local_matrices,
            strict=True,
        )
    ):
        # T_a[e,f] = sum_cd G1[c,e] Psi[a,c,d] G2[d,f].
        # Choose the smaller intermediate without changing operand order.
        if len(rows) <= len(columns):
            transformed = (
                first[rows, :].T @ (local @ second[columns, :])
            )
        else:
            transformed = (
                (first[rows, :].T @ local) @ second[columns, :]
            )
        block[output_row] = np.asarray(
            schedule.flattened_first_legs @ transformed.reshape(-1),
            dtype=np.complex128,
        )
    return block


def _sparse_bubble_direct(
    vertex: SparseCubicVertex,
    schedule: _SparseContractionSchedule,
    green: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    result = np.zeros((grid.n_points, vertex.ndof, vertex.ndof), dtype=np.complex128)
    if not len(schedule.active):
        return result
    projected_green = green[
        np.ix_(
            np.arange(grid.n_points, dtype=np.int64),
            schedule.active,
            schedule.active,
        )
    ]
    zero = grid.zero_index
    for output in range(grid.n_points):
        start = max(0, output + zero - (grid.n_points - 1))
        stop = min(grid.n_points - 1, output + zero)
        accumulated = np.zeros(
            (len(schedule.active), len(schedule.active)),
            dtype=np.complex128,
        )
        for first_index in range(start, stop + 1):
            second_index = output - first_index + zero
            accumulated += _sparse_pair_contraction(
                vertex,
                schedule,
                projected_green[first_index],
                projected_green[second_index],
            )
        result[
            output,
            schedule.active[:, None],
            schedule.active,
        ] = 0.5j * grid.integration_weight * accumulated
    return result


def _sparse_bubble_fft(
    vertex: SparseCubicVertex,
    schedule: _SparseContractionSchedule,
    green: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    n_frequency = grid.n_points
    result = np.zeros(
        (n_frequency, vertex.ndof, vertex.ndof),
        dtype=np.complex128,
    )
    if not len(schedule.active):
        return result
    padded = 1 << (2 * n_frequency - 2).bit_length()
    projected_green = green[
        np.ix_(
            np.arange(n_frequency, dtype=np.int64),
            schedule.active,
            schedule.active,
        )
    ]
    transformed_green = np.fft.fft(projected_green, n=padded, axis=0)
    del projected_green
    transformed_bubble = np.empty(
        (padded, len(schedule.active), len(schedule.active)),
        dtype=np.complex128,
    )
    for index in range(padded):
        transformed_bubble[index] = _sparse_pair_contraction(
            vertex,
            schedule,
            transformed_green[index],
            transformed_green[index],
        )
    del transformed_green
    full = np.fft.ifft(transformed_bubble, axis=0)
    start = grid.zero_index
    result[
        np.ix_(
            np.arange(n_frequency, dtype=np.int64),
            schedule.active,
            schedule.active,
        )
    ] = (
        0.5j
        * grid.integration_weight
        * full[start : start + n_frequency]
    )
    return result


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
    schedule = _compiled_sparse_schedule(vertex)
    operation = (
        _sparse_bubble_direct if backend == "direct" else _sparse_bubble_fft
    )
    return (
        operation(vertex, schedule, lesser, grid),
        operation(vertex, schedule, greater, grid),
    )


__all__ = [
    "SparseCubicVertex",
    "cubic_sparse_lesser_greater",
    "cubic_vertex_is_zero",
    "cubic_vertex_max_abs",
    "cubic_vertex_physics_sha256",
]
