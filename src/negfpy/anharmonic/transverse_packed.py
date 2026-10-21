"""Exact-support cubic storage and bounded, tiled frequency-FFT Born maps.

The harmonic device is not truncated: only cubic coordinates with exact zero
support are omitted. Packing uses no magnitude threshold. The executor returns
packed self-energies; callers explicitly embed them into their harmonic device.
These routines do not change the existing prototype solver or its limits.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .transverse_q import TransverseCubicVertex, difference_q_indices

_DEFAULT_BYTES = 64 * 1024**2
_DEFAULT_WORK = 100_000_000
_C = np.dtype(np.complex128).itemsize


def _integer(value: int, name: str, *, zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'}.")
    return int(value)


def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    # No whole-input conversion or hidden materialization of Python sequences.
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numeric numpy.ndarray.")
    if value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {value.shape}.")
    if value.dtype.kind not in "iufc":
        raise TypeError(f"{name} must contain numeric values.")
    return value


def _finite(value: np.ndarray, name: str) -> None:
    # At most a matrix-size boolean mask, including noncontiguous input views.
    for index in np.ndindex(value.shape[:-2]):
        if not np.isfinite(value[index]).all():
            raise ValueError(f"{name} must contain only finite values.")


def _packed(vertex: PackedTransverseCubicVertex) -> None:
    if not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError("vertex must be a PackedTransverseCubicVertex.")


@dataclass(frozen=True, slots=True, eq=False)
class PackedTransverseCubicVertex:
    """Owned immutable cubic coefficients and their full-device addresses.

    Coordinates may be in any unique integer order. For a nonempty support the
    existing vertex's reality and leg-exchange checks are reused unchanged.
    ``max_bytes`` bounds the owned coefficient payload, as in that constructor;
    construction validation and the caller's arrays require additional memory.
    Direct construction permits a future mapper to avoid a dense full vertex.
    """

    mesh: UniformBZMesh
    coefficients: np.ndarray = field(repr=False)
    active_dofs: np.ndarray = field(repr=False)
    full_ndof: int
    max_bytes: int = field(default=_DEFAULT_BYTES, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.mesh, UniformBZMesh):
            raise TypeError("mesh must be a UniformBZMesh.")
        if self.mesh.shape[2] != 1:
            raise ValueError("The transverse mesh must have third dimension one.")
        limit = _integer(self.max_bytes, "max_bytes")
        full = _integer(self.full_ndof, "full_ndof")
        raw_indices = np.asarray(self.active_dofs)
        if raw_indices.ndim != 1 or (raw_indices.size and raw_indices.dtype.kind not in "iu"):
            raise TypeError("active_dofs must be a one-dimensional integer sequence.")
        indices = tuple(int(x) for x in raw_indices)
        if len(set(indices)) != len(indices) or any(x < 0 or x >= full for x in indices):
            raise ValueError("active_dofs must be unique and within the full device.")
        a, nq = len(indices), self.mesh.n_qpoints
        raw = _array(self.coefficients, (nq, nq, a, a, a), "coefficients")
        if _C * raw.size + 8 * a > limit:
            raise ValueError("Packed coefficient and address payload exceeds max_bytes.")
        if a:
            owned = TransverseCubicVertex(self.mesh, raw, max_bytes=limit).coefficients
        else:
            owned = np.empty(raw.shape, dtype=np.complex128)
            owned.setflags(write=False)
        addresses = np.array(indices, dtype=np.int64)
        addresses.setflags(write=False)
        object.__setattr__(self, "coefficients", owned)
        object.__setattr__(self, "active_dofs", addresses)
        object.__setattr__(self, "full_ndof", full)
        object.__setattr__(self, "max_bytes", limit)

    @property
    def ndof(self) -> int:
        """Number of active cubic coordinates."""
        return len(self.active_dofs)

    @classmethod
    def from_dense(
        cls, vertex: TransverseCubicVertex, *, max_bytes: int = _DEFAULT_BYTES,
    ) -> PackedTransverseCubicVertex:
        """Discover exact union support; the caller's dense vertex remains live.

        The temporary packed projection and the owned validated copy coexist
        during construction. This utility is not a streaming material mapper.
        """
        if not isinstance(vertex, TransverseCubicVertex):
            raise TypeError("vertex must be a TransverseCubicVertex.")
        limit = _integer(max_bytes, "max_bytes")
        d = vertex.ndof
        if d**2 + 9 * d > limit:
            raise ValueError("Support discovery workspace exceeds max_bytes.")
        support = np.zeros(d, dtype=bool)
        for q in vertex.coefficients:
            for block in q:
                for i, matrix in enumerate(block):
                    mask = matrix != 0
                    if mask.any():
                        support[i] = True
                        support |= mask.any(axis=0)
                        support |= mask.any(axis=1)
                    del mask
        active = np.flatnonzero(support)
        a, nq = len(active), vertex.mesh.n_qpoints
        if _C * nq**2 * a**3 + 8 * a > limit:
            raise ValueError("Packed coefficient and address payload exceeds max_bytes.")
        values = np.empty((nq, nq, a, a, a), dtype=np.complex128)
        for q in range(nq):
            for p in range(nq):
                for i, source_i in enumerate(active):
                    for j, source_j in enumerate(active):
                        values[q, p, i, j] = vertex.coefficients[q, p, source_i, source_j, active]
        return cls(vertex.mesh, values, active, d, max_bytes=limit)


def _spectrum_shape(vertex: PackedTransverseCubicVertex, value: np.ndarray, d: int) -> tuple[int, ...]:
    if not isinstance(value, np.ndarray) or value.ndim != 4 or value.shape[1] < 1:
        raise ValueError("Spectrum must have shape (Nq, Nomega, D, D), Nomega >= 1.")
    return (vertex.mesh.n_qpoints, value.shape[1], d, d)


def project_transverse_green(
    vertex: PackedTransverseCubicVertex, full_green: np.ndarray, *, max_bytes: int = _DEFAULT_BYTES,
) -> np.ndarray:
    """Copy the exact active block without a full complex dtype conversion."""
    _packed(vertex)
    limit = _integer(max_bytes, "max_bytes")
    shape = _spectrum_shape(vertex, full_green, vertex.full_ndof)
    raw = _array(full_green, shape, "full_green")
    nq, nw, d, _ = shape
    a = vertex.ndof
    if _C * nq * nw * a**2 + max(_C, raw.dtype.itemsize) * a + d**2 > limit:
        raise ValueError("Projected Green payload and bounded workspace exceed max_bytes.")
    _finite(raw, "full_green")
    output = np.empty((nq, nw, a, a), dtype=np.complex128)
    for q in range(nq):
        for w in range(nw):
            for i, source_i in enumerate(vertex.active_dofs):
                output[q, w, i] = raw[q, w, source_i, vertex.active_dofs]
    _finite(output, "projected Green")
    return output


def embed_packed_self_energy(
    vertex: PackedTransverseCubicVertex, packed_array: np.ndarray, *, max_bytes: int = _DEFAULT_BYTES,
) -> np.ndarray:
    """Embed a packed spectrum, leaving every inactive entry exactly zero."""
    _packed(vertex)
    limit = _integer(max_bytes, "max_bytes")
    shape = _spectrum_shape(vertex, packed_array, vertex.ndof)
    raw = _array(packed_array, shape, "packed_array")
    nq, nw, _, _ = shape
    d = vertex.full_ndof
    if _C * nq * nw * d**2 + d**2 > limit:
        raise ValueError("Embedded output and bounded workspace exceed max_bytes.")
    _finite(raw, "packed_array")
    output = np.zeros((nq, nw, d, d), dtype=np.complex128)
    for q in range(nq):
        for w in range(nw):
            for i, target_i in enumerate(vertex.active_dofs):
                output[q, w, target_i, vertex.active_dofs] = raw[q, w, i]
    _finite(output, "embedded self-energy")
    return output


@dataclass(frozen=True, slots=True)
class PackedBornFFTPlan:
    """Explicit numeric payload and complex multiply-accumulate work counts.

    ``peak_owned_bytes`` excludes caller-owned input arrays and persistent
    vertex storage, reported separately. FFT/BLAS internal scratch and Python
    object overhead (including the bounded futures) are excluded: this is not
    an RSS guarantee. No rank-four coordinate array is allocated. Vertex bytes
    include its int64 addresses. Input Green bytes are a complex128 logical
    payload for both components; caller dtypes may have different actual sizes.
    """

    n_q: int
    n_w: int
    n_fft: int
    n_active: int
    external_tile: int
    workers: int
    bin_tile: int
    effective_workers: int
    max_inflight_futures: int
    vertex_bytes: int
    input_green_bytes: int
    output_bytes: int
    transformed_green_bytes: int
    q_transform_bytes: int
    conjugate_vertex_bytes: int
    worker_workspace_bytes: int
    auxiliary_bytes: int
    peak_owned_bytes: int
    per_component_contraction_work: int
    contraction_work: int


def plan_packed_born_fft_dimensions(
    n_q: int, n_w: int, n_active: int, *, external_tile: int = 16,
    workers: int = 1, bin_tile: int = 8,
) -> PackedBornFFTPlan:
    """Plan by dimensions only; no material arrays are read or allocated."""
    nq = _integer(n_q, "n_q")
    nw = _integer(n_w, "n_w")
    a = _integer(n_active, "n_active", zero=True)
    if nw % 2 != 1:
        raise ValueError("n_w must be odd.")
    tile = _integer(external_tile, "external_tile")
    workers = _integer(workers, "workers")
    bins = _integer(bin_tile, "bin_tile")
    nfft = 1 << (2 * nw - 2).bit_length()
    effective = min(workers, (nfft + bins - 1) // bins) if a else 0
    t = min(tile, a)
    output = 2 * _C * nq * nw * a**2
    transformed = _C * nq * nfft * a**2
    qbytes = _C * nfft * a**2
    conjugate = _C * a**3
    worker = effective * _C * (2 * t * a**2 + t * a)
    auxiliary = (8 * nq**2 + a**2) if a else 0
    # Transform: all-q storage + one returned q FFT. Contraction: one q Sigma
    # plus shared conjugated vertex and workers. IFFT: q Sigma + returned q IFFT.
    peak = output + transformed + max(qbytes + conjugate + worker, 2 * qbytes) + auxiliary
    work = 3 * nq**2 * nfft * a**4
    return PackedBornFFTPlan(
        nq, nw, nfft, a, tile, workers, bins, effective, effective,
        _C * nq**2 * a**3 + 8 * a, output, output, transformed, qbytes,
        conjugate, worker, auxiliary, peak, work, 2 * work,
    )


def plan_packed_born_fft(
    grid: SignedFrequencyGrid, vertex: PackedTransverseCubicVertex, *,
    external_tile: int = 16, workers: int = 1, bin_tile: int = 8,
) -> PackedBornFFTPlan:
    """Plan both contour components, processed sequentially with shared routing."""
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    _packed(vertex)
    return plan_packed_born_fft_dimensions(
        vertex.mesh.n_qpoints, grid.n_points, vertex.ndof,
        external_tile=external_tile, workers=workers, bin_tile=bin_tile,
    )


def _contract_into(
    first: np.ndarray, green_left: np.ndarray, green_right: np.ndarray,
    output: np.ndarray, tile: int, workers: int, bin_tile: int,
) -> None:
    """Accumulate into disjoint bins; the caller serializes the q' loop."""
    nbin, a = green_left.shape[:2]
    tile = min(tile, a)
    conjugate = np.empty((a, a, a), dtype=np.complex128)
    np.conjugate(first, out=conjugate)
    conjugate_matrix = conjugate.reshape(a, a * a).T

    def chunk(start: int, stop: int) -> None:
        left = np.empty((tile, a, a), dtype=np.complex128)
        both = np.empty_like(left)
        final = np.empty((tile, a), dtype=np.complex128)
        for index in range(start, stop):
            for begin in range(0, a, tile):
                end = min(begin + tile, a)
                width = end - begin
                # (e,c) @ (tile,c,d) -> (tile,e,d), then (tile,e,d) @ (d,f).
                np.matmul(green_left[index].T, first[begin:end], out=left[:width])
                np.matmul(left[:width], green_right[index], out=both[:width])
                np.matmul(both[:width].reshape(width, a * a), conjugate_matrix, out=final[:width])
                output[index, begin:end] += final[:width]

    effective = min(workers, (nbin + bin_tile - 1) // bin_tile)
    if effective == 1:
        for start in range(0, nbin, bin_tile):
            chunk(start, min(start + bin_tile, nbin))
    else:
        # At most effective futures; no result arrays or unbounded work queue.
        with ThreadPoolExecutor(max_workers=effective) as executor:
            pending = deque()
            for start in range(0, nbin, bin_tile):
                if len(pending) == effective:
                    pending.popleft().result()
                pending.append(executor.submit(chunk, start, min(start + bin_tile, nbin)))
            for future in pending:
                future.result()


def contract_packed_fft_bins(
    first_vertex: np.ndarray, green_left: np.ndarray, green_right: np.ndarray, *,
    external_tile: int = 16, workers: int = 1, bin_tile: int = 8,
    max_bytes: int = _DEFAULT_BYTES, max_contraction_work: int = _DEFAULT_WORK,
) -> np.ndarray:
    """Unweighted ``V G_left G_right conj(V)`` for a bounded set of FFT bins.

    Accept arbitrary finite cubic coefficients, without physical vertex checks.
    Work is ``3 * B * A**4`` complex multiply-accumulate terms. ``max_bytes``
    bounds the output, conjugated vertex, worker arrays and finite mask; caller
    input arrays and library/Python scratch are separate. BLAS uses one thread
    inside a restoring context; independent bin chunks use bounded workers.
    """
    tile = _integer(external_tile, "external_tile")
    workers = _integer(workers, "workers")
    bins = _integer(bin_tile, "bin_tile")
    limit = _integer(max_bytes, "max_bytes")
    work_limit = _integer(max_contraction_work, "max_contraction_work")
    if not isinstance(first_vertex, np.ndarray) or first_vertex.ndim != 3:
        raise ValueError("first_vertex must have shape (A, A, A).")
    a = first_vertex.shape[0]
    first = _array(first_vertex, (a, a, a), "first_vertex")
    if not isinstance(green_left, np.ndarray) or green_left.ndim != 3 or green_left.shape[0] < 1:
        raise ValueError("green_left must have shape (B, A, A), B >= 1.")
    nbin = green_left.shape[0]
    left = _array(green_left, (nbin, a, a), "green_left")
    right = _array(green_right, (nbin, a, a), "green_right")
    effective = min(workers, (nbin + bins - 1) // bins) if a else 0
    t = min(tile, a)
    owned = _C * (nbin * a**2 + a**3 + effective * (2 * t * a**2 + t * a)) + a**2
    if owned > limit:
        raise ValueError("Packed contraction output and workspace exceed max_bytes.")
    if 3 * nbin * a**4 > work_limit:
        raise ValueError("Packed contraction exceeds max_contraction_work.")
    for value, name in ((first, "first_vertex"), (left, "green_left"), (right, "green_right")):
        _finite(value, name)
    output = np.zeros((nbin, a, a), dtype=np.complex128)
    if a:
        with threadpool_limits(limits=1, user_api="blas"):
            _contract_into(first, left, right, output, tile, workers, bins)
        _finite(output, "contraction output")
    return output


def cubic_born_self_energy_q_packed_fft(
    grid: SignedFrequencyGrid, vertex: PackedTransverseCubicVertex,
    g_lesser_packed: np.ndarray, g_greater_packed: np.ndarray, *,
    external_tile: int = 16, workers: int = 1, bin_tile: int = 8,
    max_bytes: int = _DEFAULT_BYTES, max_contraction_work: int = _DEFAULT_WORK,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exact zero-extended frequency sum on packed coordinates.

    Returns two ``(Nq, Nomega, A, A)`` arrays. Their simultaneous payload is
    included in ``max_bytes``, alongside the plan's additional owned arrays.
    Caller inputs/persistent vertex and library/Python scratch are separate.
    Limits and finite/type checks precede outputs and FFT execution. Contour
    components and q' accumulation are serial; only independent bins run in
    parallel. No Green function is conjugated and no physical cutoff is added.
    """
    plan = plan_packed_born_fft(grid, vertex, external_tile=external_tile, workers=workers, bin_tile=bin_tile)
    limit = _integer(max_bytes, "max_bytes")
    work_limit = _integer(max_contraction_work, "max_contraction_work")
    if plan.peak_owned_bytes > limit:
        raise ValueError("Packed FFT peak owned payload exceeds max_bytes.")
    if plan.contraction_work > work_limit:
        raise ValueError("Packed FFT exceeds max_contraction_work.")
    shape = (plan.n_q, plan.n_w, plan.n_active, plan.n_active)
    greens = (
        _array(g_lesser_packed, shape, "g_lesser_packed"),
        _array(g_greater_packed, shape, "g_greater_packed"),
    )
    for value in greens:
        _finite(value, "packed Green")
    outputs = (np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128))
    if not plan.n_active:
        return outputs
    routes = difference_q_indices(vertex.mesh, max_bytes=limit)
    prefactor = 0.5j * grid.integration_weight / plan.n_q
    with threadpool_limits(limits=1, user_api="blas"):
        for green, output in zip(greens, outputs, strict=True):
            transformed = np.zeros((plan.n_q, plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
            transformed[:, :plan.n_w] = green
            for q in range(plan.n_q):
                transformed[q] = np.fft.fft(transformed[q], axis=0)
            for q in range(plan.n_q):
                sigma = np.zeros((plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
                for p in range(plan.n_q):
                    _contract_into(vertex.coefficients[q, p], transformed[p], transformed[routes[q, p]],
                                   sigma, plan.external_tile, plan.workers, plan.bin_tile)
                convolution = np.fft.ifft(sigma, axis=0)
                del sigma
                selected = convolution[grid.zero_index:grid.zero_index + plan.n_w]
                np.multiply(selected, prefactor, out=output[q])
                del selected, convolution
            del transformed
            _finite(output, "packed FFT output")
    return outputs


__all__ = [
    "PackedBornFFTPlan", "PackedTransverseCubicVertex", "contract_packed_fft_bins",
    "cubic_born_self_energy_q_packed_fft", "embed_packed_self_energy",
    "plan_packed_born_fft", "plan_packed_born_fft_dimensions", "project_transverse_green",
]
