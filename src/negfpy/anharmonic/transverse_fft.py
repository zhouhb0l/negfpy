"""Bounded frequency-FFT implementation of the reduced transverse-q bubble.

This evaluates the same zero-extended, rectangular frequency sum as
``transverse_q.cubic_born_self_energy_q``. Only the frequency convolution is
transformed: the two internal q sectors are still routed and summed explicitly.
Exactly inactive cubic coordinates are compressed without a numerical cutoff.
"""

from __future__ import annotations

import numpy as np

from .grid import SignedFrequencyGrid
from .transverse_q import TransverseCubicVertex, difference_q_indices

_MAX_ARRAY_BYTES = 64 * 1024**2
_COMPLEX_BYTES = np.dtype(np.complex128).itemsize


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _green(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    # Keep the caller's dtype here; projection below converts one matrix at a
    # time, avoiding an unbudgeted full complex copy of a real input spectrum.
    array = np.asarray(values)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values.")
    for q_block in array:
        if not np.isfinite(q_block).all():
            raise ValueError(f"{name} must contain only finite values.")
    return array


def _active_coordinates(coefficients: np.ndarray) -> np.ndarray:
    """Union exact support on all three legs, using only matrix-size masks."""
    ndof = coefficients.shape[2]
    active = np.zeros(ndof, dtype=bool)
    for q_block in coefficients:
        for vertex in q_block:
            for a, matrix in enumerate(vertex):
                mask = matrix != 0
                if np.any(mask):
                    active[a] = True
                    active |= np.any(mask, axis=0)
                    active |= np.any(mask, axis=1)
    return np.flatnonzero(active)


def _workspace_bytes(n_q: int, n_w: int, n_fft: int, ndof: int, active: int) -> int:
    """Conservative explicit-array payload, excluding outputs and input arrays."""
    complex_entries = (
        2 * n_q * n_fft * active**2
        + n_q * n_w * active**2
        + 4 * active**3
        + 2 * active**2
    )
    # Routing, active addresses and bounded support/projection index masks.
    auxiliary_bytes = 8 * n_q**2 + 8 * active + ndof**2 + ndof
    return _COMPLEX_BYTES * complex_entries + auxiliary_bytes


def _component(
    green: np.ndarray,
    output: np.ndarray,
    vertex: TransverseCubicVertex,
    routes: np.ndarray,
    active: np.ndarray,
    n_fft: int,
    grid: SignedFrequencyGrid,
) -> None:
    n_q, n_w = green.shape[:2]
    n_active = len(active)
    active_pair = np.ix_(active, active)
    active_triple = np.ix_(active, active, active)
    projected = np.empty((n_q, n_w, n_active, n_active), dtype=np.complex128)
    for q in range(n_q):
        for omega in range(n_w):
            projected[q, omega] = green[q, omega][active_pair]
    transformed_green = np.fft.fft(projected, n=n_fft, axis=1)
    del projected
    transformed_sigma = np.zeros_like(transformed_green)
    for q in range(n_q):
        for p in range(n_q):
            partner = routes[q, p]
            first = vertex.coefficients[q, p][active_triple]
            second = first.conj()
            for frequency_bin in range(n_fft):
                left = np.einsum(
                    "acd,ce->aed",
                    first,
                    transformed_green[p, frequency_bin],
                    optimize=True,
                )
                both = np.einsum(
                    "aed,df->aef",
                    left,
                    transformed_green[partner, frequency_bin],
                    optimize=True,
                )
                del left
                transformed_sigma[q, frequency_bin] += np.einsum(
                    "aef,bef->ab", both, second, optimize=True
                )
                del both
            del first, second
    del transformed_green
    convolution = np.fft.ifft(transformed_sigma, axis=1)
    del transformed_sigma
    start = grid.zero_index
    selected = convolution[:, start : start + n_w]
    selected *= 0.5j * grid.integration_weight / n_q
    # Advanced indexing writes just the active block; all other entries retain
    # the exact zeros of the full-size output allocation.
    output[:, :, active[:, None], active] = selected


def cubic_born_self_energy_q_fft(
    grid: SignedFrequencyGrid,
    vertex: TransverseCubicVertex,
    g_lesser: np.ndarray,
    g_greater: np.ndarray,
    *,
    max_pair_evaluations: int = 100_000,
    max_bytes: int = _MAX_ARRAY_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate lesser/greater bubbles on ``(Nq, Nomega, D, D)``.

    The second vertex is conjugated; neither Green matrix is conjugated or
    transposed. The factor is ``0.5j * grid.integration_weight / Nq``. Zero
    padding to the next power of two at least ``2*Nomega-1`` prevents frequency
    wraparound. The result is algebraically equal to the direct finite-grid
    sum, up to floating-point FFT and summation differences.

    Exact nonzero coefficients determine a union of active coordinates on all
    three cubic legs. All coordinates outside this support return exact zeros;
    arbitrarily small nonzero coefficients are retained. Coupling scaling, if
    required, must already be included in ``vertex``.

    ``max_pair_evaluations`` bounds ``Nq**2 * Nfft`` three-stage contractions
    per contour component (zero for an identically zero vertex). ``max_bytes``
    independently bounds each full output and the planned explicit FFT
    workspace. The two outputs coexist; input arrays, library-internal FFT/BLAS
    scratch, and Python overhead are not part of that workspace or a process
    RSS guarantee. No rank-four coordinate tensor is formed. Resource and input
    checks precede output allocation and transforms. Inputs are not modified;
    physical identities are not imposed on arbitrary algebraic Green fixtures.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(vertex, TransverseCubicVertex):
        raise TypeError("vertex must be a TransverseCubicVertex.")
    pair_limit = _positive_integer(max_pair_evaluations, "max_pair_evaluations")
    byte_limit = _positive_integer(max_bytes, "max_bytes")
    n_q, n_w, ndof = vertex.mesh.n_qpoints, grid.n_points, vertex.ndof
    shape = (n_q, n_w, ndof, ndof)
    if _COMPLEX_BYTES * n_q * n_w * ndof**2 > byte_limit:
        raise ValueError("Each full coupled-q output payload must fit max_bytes.")
    active = _active_coordinates(vertex.coefficients)
    n_fft = 1 << (2 * n_w - 2).bit_length()
    pair_evaluations = n_q**2 * n_fft if len(active) else 0
    if pair_evaluations > pair_limit:
        raise ValueError("Coupled-q FFT contraction exceeds max_pair_evaluations.")
    if _workspace_bytes(n_q, n_w, n_fft, ndof, len(active)) > byte_limit:
        raise ValueError("Planned coupled-q FFT workspace exceeds max_bytes.")
    greens = (
        _green(g_lesser, shape, "g_lesser"),
        _green(g_greater, shape, "g_greater"),
    )
    outputs = (
        np.zeros(shape, dtype=np.complex128),
        np.zeros(shape, dtype=np.complex128),
    )
    if not len(active):
        return outputs
    routes = difference_q_indices(vertex.mesh, max_bytes=byte_limit)
    for green, output in zip(greens, outputs, strict=True):
        _component(green, output, vertex, routes, active, n_fft, grid)
        if not np.isfinite(output).all():
            raise ValueError("Coupled-q FFT contraction produced nonfinite output.")
    return outputs


__all__ = ["cubic_born_self_energy_q_fft"]
