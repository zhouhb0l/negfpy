r"""Cubic Born self-energy on a signed, uniform frequency grid.

The implementation follows Eq. (8) of Guo et al.,
Phys. Rev. B 102, 195412 (2020), using the convention

.. math::

   H_3 = \frac{1}{3!}\sum_{abc}\Psi_{abc}u_a u_b u_c.

In the dimensionless solver units used by :mod:`negfpy.anharmonic`, the
lesser/greater bubble is

.. math::

   \Sigma^{</>}_{ab}(\omega) =
   \frac{i}{2}\sum_{cdef}\Psi_{acd}\Psi_{bef}
   \int\frac{d\omega'}{2\pi}
   G^{</>}_{ce}(\omega')G^{</>}_{df}(\omega-\omega').

Frequencies outside the supplied grid are exactly zero.  In particular, this
module never performs a circular convolution.  The default ``"serial"`` and
``"threaded"`` implementations retain the direct frequency sum used as the
correctness oracle.  The optional ``"fft"`` backend evaluates the same
discrete sum as a zero-padded linear convolution; its transform length is
always at least ``2 * N_omega - 1``.  The activation profile on the reference
development environment used ``N_omega=513`` and ``D=3``: the complete
lesser/greater call took 2.384 s directly and 0.01679 s by FFT (142x), with a
maximum absolute field difference of ``7.78e-16``.  Timing is
hardware-dependent; the direct implementation remains the numerical oracle.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from functools import partial
import os
from typing import TYPE_CHECKING, Literal, cast

import numpy as np

from .grid import SignedFrequencyGrid
from .types import DenseCubicVertex


if TYPE_CHECKING:
    from .sparse_cubic import SparseCubicVertex


Array = np.ndarray
Backend = Literal["serial", "threaded", "fft"]

# Keep each rank-four complex contraction intermediate near this size.  The
# un-tiled batched contraction scales as O(N_omega D^3) scratch *per worker*
# and can exceed the solver's O(N_omega D^2) state estimate long before the
# stored Green functions do.
_TARGET_INTERMEDIATE_BYTES = 8 * 2**20
_MAX_INTERNAL_FREQUENCY_TILE = 128

# The FFT path was activated only after an N_omega=513, D=3 benchmark exceeded
# the project's required 2x cubic-kernel speedup while satisfying
# rtol=1e-10, atol=1e-12 against the direct oracle.  Keep the threshold here as
# provenance for the optimization decision, not as a runtime timing test.
FFT_ACTIVATION_MINIMUM_SPEEDUP = 2.0


def _validated_green(
    values: Array,
    *,
    name: str,
    n_frequency: int,
    ndof: int,
) -> Array:
    """Return a finite complex128 Green-function spectrum."""

    array = np.asarray(values, dtype=np.complex128)
    expected = (n_frequency, ndof, ndof)
    if array.shape != expected:
        raise ValueError(f"{name} must have shape {expected}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _contract_cubic_pair(
    vertex: Array,
    green_first: Array,
    green_second: Array,
) -> Array:
    """Contract one frequency pair without forming a rank-six tensor.

    This is algebraically identical to

    ``einsum("acd,bef,ce,df->ab", vertex, vertex, g1, g2)``

    but applies the two Green functions to one vertex first.  Each step has at
    most three free indices, which gives an exact dense contraction with a
    rank-three intermediate.
    """

    transformed_c = np.einsum(
        "acd,ce->aed",
        vertex,
        green_first,
        optimize=True,
    )
    transformed_cd = np.einsum(
        "aed,df->aef",
        transformed_c,
        green_second,
        optimize=True,
    )
    return np.einsum(
        "aef,bef->ab",
        transformed_cd,
        vertex,
        optimize=True,
    )


def _cubic_bubble_at_index(
    vertex: Array,
    green: Array,
    grid: SignedFrequencyGrid,
    output_index: int,
) -> tuple[int, Array]:
    """Evaluate one output-frequency slice with deterministic summation."""

    # Array index i represents integer frequency i-zero.  For a chosen output
    # i and integration index j, omega_i - omega_j therefore has array index
    # i-j+zero.  Restricting that index to the grid implements zero extension.
    n_frequency = grid.n_points
    zero = grid.zero_index
    input_start = max(0, output_index + zero - (n_frequency - 1))
    input_stop = min(n_frequency - 1, output_index + zero)
    input_indices = np.arange(input_start, input_stop + 1, dtype=np.int64)
    difference_indices = output_index - input_indices + zero
    ndof = int(vertex.shape[0])
    bytes_per_frequency = np.dtype(np.complex128).itemsize * ndof**3
    tile_size = max(
        1,
        min(
            _MAX_INTERNAL_FREQUENCY_TILE,
            _TARGET_INTERMEDIATE_BYTES // max(1, bytes_per_frequency),
        ),
    )
    accumulated = np.zeros((ndof, ndof), dtype=np.complex128)

    # Batch a bounded interval of internal frequencies at a time.  ``m`` is
    # the internal-frequency index.  Tiling retains zero extension and the
    # deterministic summation order while preventing a hidden
    # O(N_omega D^3) allocation in every thread.
    for start in range(0, input_indices.size, tile_size):
        stop = min(input_indices.size, start + tile_size)
        selected = input_indices[start:stop]
        selected_difference = difference_indices[start:stop]
        transformed_c = np.einsum(
            "acd,mce->maed",
            vertex,
            green[selected],
            optimize=True,
        )
        transformed_cd = np.einsum(
            "maed,mdf->maef",
            transformed_c,
            green[selected_difference],
            optimize=True,
        )
        accumulated += np.einsum(
            "maef,bef->ab",
            transformed_cd,
            vertex,
            optimize=True,
        )
    return output_index, 0.5j * grid.integration_weight * accumulated


def _validated_execution(
    backend: Backend | str,
    max_workers: int | None,
) -> tuple[Backend, int | None]:
    """Validate deterministic dense-kernel execution controls."""

    if backend not in {"serial", "threaded", "fft"}:
        raise ValueError(
            "backend must be 'serial', 'threaded', or 'fft'."
        )
    if max_workers is not None:
        if isinstance(max_workers, (bool, np.bool_)) or not isinstance(
            max_workers,
            (int, np.integer),
        ):
            raise TypeError("max_workers must be an integer or None.")
        max_workers = int(max_workers)
        if max_workers < 1:
            raise ValueError("max_workers must be at least one.")
    return cast(Backend, backend), max_workers


def resolved_worker_count(
    backend: Backend | str,
    max_workers: int | None,
) -> int:
    """Return the worker count used by the deterministic dense backend."""

    selected, workers = _validated_execution(backend, max_workers)
    if selected != "threaded":
        return 1
    if workers is not None:
        return workers
    return min(32, (os.cpu_count() or 1) + 4)


def cubic_worker_scratch_bytes(ndof: int) -> int:
    """Upper bound for the two tiled rank-four intermediates per worker."""

    if (
        isinstance(ndof, (bool, np.bool_))
        or not isinstance(ndof, (int, np.integer))
        or int(ndof) < 1
    ):
        raise ValueError("ndof must be a positive integer.")
    dimension = int(ndof)
    bytes_per_frequency = np.dtype(np.complex128).itemsize * dimension**3
    tile_size = max(
        1,
        min(
            _MAX_INTERNAL_FREQUENCY_TILE,
            _TARGET_INTERMEDIATE_BYTES // max(1, bytes_per_frequency),
        ),
    )
    return int(2 * tile_size * bytes_per_frequency)


def cubic_fft_scratch_bytes(n_frequency: int, ndof: int) -> int:
    """Conservative peak scratch estimate for one FFT bubble evaluation."""

    if (
        isinstance(n_frequency, (bool, np.bool_))
        or not isinstance(n_frequency, (int, np.integer))
        or int(n_frequency) < 1
    ):
        raise ValueError("n_frequency must be a positive integer.")
    if (
        isinstance(ndof, (bool, np.bool_))
        or not isinstance(ndof, (int, np.integer))
        or int(ndof) < 1
    ):
        raise ValueError("ndof must be a positive integer.")
    count = int(n_frequency)
    dimension = int(ndof)
    padded_length = 1 << (2 * count - 2).bit_length()
    transformed_spectra = (
        4
        * padded_length
        * dimension**2
        * np.dtype(np.complex128).itemsize
    )
    return int(
        transformed_spectra + cubic_worker_scratch_bytes(dimension)
    )


def _cubic_bubble_one(
    vertex: Array,
    green: Array,
    grid: SignedFrequencyGrid,
    *,
    backend: Backend,
    max_workers: int | None,
) -> Array:
    """Evaluate one lesser/greater bubble using the selected backend."""

    n_frequency = grid.n_points
    if backend == "fft":
        return _cubic_bubble_fft(vertex, green, grid)
    if vertex.shape[0] == 1:
        full = np.convolve(
            green[:, 0, 0],
            green[:, 0, 0],
            mode="full",
        )
        result = np.empty((n_frequency, 1, 1), dtype=np.complex128)
        start = grid.zero_index
        result[:, 0, 0] = (
            0.5j
            * grid.integration_weight
            * vertex[0, 0, 0] ** 2
            * full[start : start + n_frequency]
        )
        return result

    result = np.empty(
        (n_frequency, vertex.shape[0], vertex.shape[0]),
        dtype=np.complex128,
    )
    evaluate = partial(_cubic_bubble_at_index, vertex, green, grid)

    if backend == "serial":
        slices = map(evaluate, range(n_frequency))
        for output_index, values in slices:
            result[output_index] = values
        return result

    # Each worker owns one complete output-frequency slice.  The inner
    # frequency sum has exactly the same order as in serial execution, and
    # executor.map returns slices in input order, giving deterministic output.
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:  # pragma: no cover - dependency guard for source trees
        blas_limit = nullcontext()
    else:
        blas_limit = threadpool_limits(limits=1, user_api="blas")
    workers = resolved_worker_count(backend, max_workers)
    with blas_limit:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            slices = executor.map(evaluate, range(n_frequency))
            for output_index, values in slices:
                result[output_index] = values

    return result


def _cubic_bubble_fft(
    vertex: Array,
    green: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    """Evaluate one bubble as an exact zero-padded linear convolution.

    For every Fourier bin, the two transformed Green functions are contracted
    with the cubic vertices before the inverse transform.  This avoids
    materializing an ``O(N_omega D^4)`` table of element-wise convolutions.
    Frequency bins are tiled so that the two rank-four intermediates remain
    bounded for larger dense vertices.
    """

    n_frequency = grid.n_points
    linear_length = 2 * n_frequency - 1
    # A power-of-two transform is deterministic for a fixed NumPy build and
    # never shorter than the length required for a linear convolution.
    padded_length = 1 << (linear_length - 1).bit_length()
    transformed_green = np.fft.fft(
        green,
        n=padded_length,
        axis=0,
    )

    ndof = int(vertex.shape[0])
    bytes_per_frequency = np.dtype(np.complex128).itemsize * ndof**3
    tile_size = max(
        1,
        min(
            _MAX_INTERNAL_FREQUENCY_TILE,
            _TARGET_INTERMEDIATE_BYTES // max(1, bytes_per_frequency),
        ),
    )
    transformed_bubble = np.empty(
        (padded_length, ndof, ndof),
        dtype=np.complex128,
    )
    for start in range(0, padded_length, tile_size):
        stop = min(padded_length, start + tile_size)
        selected = transformed_green[start:stop]
        transformed_c = np.einsum(
            "acd,kce->kaed",
            vertex,
            selected,
            optimize=True,
        )
        transformed_cd = np.einsum(
            "kaed,kdf->kaef",
            transformed_c,
            selected,
            optimize=True,
        )
        transformed_bubble[start:stop] = np.einsum(
            "kaef,bef->kab",
            transformed_cd,
            vertex,
            optimize=True,
        )

    full_convolution = np.fft.ifft(
        transformed_bubble,
        axis=0,
    )
    start = grid.zero_index
    return (
        0.5j
        * grid.integration_weight
        * full_convolution[start : start + n_frequency]
    )


def cubic_vertex_support(
    vertex: DenseCubicVertex | SparseCubicVertex,
) -> Array:
    """Return exact ordered nonzero cubic indices without densifying.

    Dense vertices retain their historical row-major ``argwhere`` ordering;
    sparse vertices are already required to use the same lexicographic order.
    The returned sparse view is read-only, while the dense result is a new
    array as before.
    """

    from .sparse_cubic import SparseCubicVertex

    if isinstance(vertex, SparseCubicVertex):
        return vertex.indices
    if isinstance(vertex, DenseCubicVertex):
        return np.argwhere(np.abs(vertex.values) > 0.0)
    raise TypeError("vertex must be a DenseCubicVertex or SparseCubicVertex.")


def cubic_lesser_greater(
    vertex: DenseCubicVertex | SparseCubicVertex,
    green_lesser: Array,
    green_greater: Array,
    grid: SignedFrequencyGrid,
    *,
    backend: Backend = "serial",
    max_workers: int | None = None,
) -> tuple[Array, Array]:
    """Return the cubic ``(Sigma_lesser, Sigma_greater)`` pair.

    The rectangle-rule weight ``Delta omega / (2 pi)`` and the fixed ``i/2``
    diagrammatic prefactor are applied internally.  There is deliberately no
    user-adjustable prefactor.  ``backend="threaded"`` parallelizes independent
    output-frequency slices while preserving the serial summation order within
    every slice.  ``backend="fft"`` uses a zero-padded linear convolution and
    is tested against the direct serial oracle.  The default remains
    ``"serial"``.  Exact sparse vertices dispatch to their direct kernel for
    ``"serial"`` and ``"threaded"`` (threaded sparse contraction is not yet
    enabled), and to their zero-padded implementation for ``"fft"``. Sparse
    FFT accepts an explicit ``max_workers`` of 2--8 for independent slices;
    the default None and 1 retain its serial path.
    """

    # The sparse module imports ``DenseCubicVertex`` for explicit conversion,
    # so import its runtime type and implementation only at this dispatch
    # boundary.
    from .sparse_cubic import SparseCubicVertex, cubic_sparse_lesser_greater

    if not isinstance(vertex, (DenseCubicVertex, SparseCubicVertex)):
        raise TypeError("vertex must be a DenseCubicVertex or SparseCubicVertex.")
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    selected_backend, selected_workers = _validated_execution(
        backend,
        max_workers,
    )

    if isinstance(vertex, SparseCubicVertex):
        sparse_backend = "fft" if selected_backend == "fft" else "direct"
        return cubic_sparse_lesser_greater(
            vertex,
            green_lesser,
            green_greater,
            grid,
            backend=sparse_backend,
            max_workers=selected_workers if sparse_backend == "fft" else None,
        )

    lesser = _validated_green(
        green_lesser,
        name="green_lesser",
        n_frequency=grid.n_points,
        ndof=vertex.ndof,
    )
    greater = _validated_green(
        green_greater,
        name="green_greater",
        n_frequency=grid.n_points,
        ndof=vertex.ndof,
    )
    if not np.any(vertex.values):
        shape = (grid.n_points, vertex.ndof, vertex.ndof)
        zeros = np.zeros(shape, dtype=np.complex128)
        return zeros, zeros.copy()

    sigma_lesser = _cubic_bubble_one(
        vertex.values,
        lesser,
        grid,
        backend=selected_backend,
        max_workers=selected_workers,
    )
    sigma_greater = _cubic_bubble_one(
        vertex.values,
        greater,
        grid,
        backend=selected_backend,
        max_workers=selected_workers,
    )
    return sigma_lesser, sigma_greater


__all__ = [
    "FFT_ACTIVATION_MINIMUM_SPEEDUP",
    "cubic_fft_scratch_bytes",
    "cubic_lesser_greater",
    "cubic_vertex_support",
    "cubic_worker_scratch_bytes",
    "resolved_worker_count",
]
