"""Independent statistical/spectral channels for bounded cubic Born arithmetic.

The channels are ``C = (G_greater + G_lesser)/2`` and
``D = G_greater - G_lesser``.  They remain independent arrays through the Born
contraction, mixing and retarded reconstruction.  This avoids subtracting two
large calculated contour outputs to recover the smaller spectral output.  It
cannot recover information already lost in supplied arrays and does not repair
inconsistent input fields.  All raw matrix entries are retained: no projection,
triangle copying, conjugate completion or identity forcing is performed.

Payload plans exclude caller arrays, persistent vertex storage, Python objects
and FFT/BLAS internal scratch.  They are numeric allocation counts, not RSS
guarantees.  No existing material map or solver is switched to these routines.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from threadpoolctl import threadpool_limits

from .grid import SignedFrequencyGrid
from .transverse_packed import (
    PackedTransverseCubicVertex,
    _array,
    _contract_into,
    _finite,
    _integer,
    _packed,
    plan_packed_born_fft_dimensions,
)
from .transverse_q import difference_q_indices

_DEFAULT_BYTES = 64 * 1024**2
_DEFAULT_WORK = 100_000_000
_C = np.dtype(np.complex128).itemsize


@dataclass(frozen=True, slots=True, eq=False)
class StatisticalSpectral:
    """Borrowed raw statistical and spectral arrays; no implicit conversion.

    Operations validate shape, finiteness and their complete payload bounds
    before allocation.  This container alone neither validates nor copies its
    arrays.  Callers must retain independent ``spectral`` data, rather than
    reconstruct it by subtracting exported lesser/greater fields.
    """

    statistical: np.ndarray = field(repr=False)
    spectral: np.ndarray = field(repr=False)


@dataclass(frozen=True, slots=True)
class BornChannelsFFTPlan:
    """Counts for two input transforms and four bilinear contractions.

    Both channel transforms coexist.  Each output channel is accumulated in
    one q-sized transform buffer, so the four terms do not require four full
    output spectra.  Work counts include all four ``V X Y conj(V)`` terms.
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
    input_channel_bytes: int
    output_bytes: int
    transformed_channel_bytes: int
    q_transform_bytes: int
    conjugate_vertex_bytes: int
    worker_workspace_bytes: int
    auxiliary_bytes: int
    peak_owned_bytes: int
    per_bilinear_contraction_work: int
    contraction_work: int


def plan_born_channels_fft_dimensions(
    n_q: int, n_w: int, n_active: int, *, external_tile: int = 16,
    workers: int = 8, bin_tile: int = 1,
) -> BornChannelsFFTPlan:
    """Count dimensions only; do not access or allocate numerical fields."""
    base = plan_packed_born_fft_dimensions(
        n_q, n_w, n_active, external_tile=external_tile,
        workers=workers, bin_tile=bin_tile,
    )
    transformed = 2 * base.transformed_green_bytes
    # Output buffers + both transformed inputs + one q contraction/IFFT stage.
    # The q-sized temporary returned while forming an input FFT fits this peak.
    peak = base.peak_owned_bytes + base.transformed_green_bytes
    return BornChannelsFFTPlan(
        base.n_q, base.n_w, base.n_fft, base.n_active, base.external_tile,
        base.workers, base.bin_tile, base.effective_workers,
        base.max_inflight_futures, base.vertex_bytes, base.input_green_bytes,
        base.output_bytes, transformed, base.q_transform_bytes,
        base.conjugate_vertex_bytes, base.worker_workspace_bytes,
        base.auxiliary_bytes, peak, base.per_component_contraction_work,
        4 * base.per_component_contraction_work,
    )


def plan_born_channels_fft(
    grid: SignedFrequencyGrid, vertex: PackedTransverseCubicVertex, *,
    external_tile: int = 16, workers: int = 8, bin_tile: int = 1,
) -> BornChannelsFFTPlan:
    """Plan independent C and D on the existing finite signed grid."""
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    _packed(vertex)
    return plan_born_channels_fft_dimensions(
        vertex.mesh.n_qpoints, grid.n_points, vertex.ndof,
        external_tile=external_tile, workers=workers, bin_tile=bin_tile,
    )


def _channel_arrays(
    channels: StatisticalSpectral, shape: tuple[int, ...], name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(channels, StatisticalSpectral):
        raise TypeError(f"{name} must be StatisticalSpectral.")
    return (
        _array(channels.statistical, shape, f"{name}.statistical"),
        _array(channels.spectral, shape, f"{name}.spectral"),
    )


def mix_channels(
    old: StatisticalSpectral, new: StatisticalSpectral, mixing: float, *,
    max_bytes: int = _DEFAULT_BYTES,
) -> StatisticalSpectral:
    """Return ``(1-mixing)*old + mixing*new`` independently for C and D.

    Inputs have shape ``(Nq, Nomega, A, A)`` and are never modified.  The
    bound includes both complex128 outputs and one matrix-sized scratch array.
    Endpoints zero and one are allowed; no damping is selected automatically.
    """
    limit = _integer(max_bytes, "max_bytes")
    if isinstance(mixing, (bool, np.bool_)) or not isinstance(
        mixing, (int, float, np.integer, np.floating)
    ):
        raise TypeError("mixing must be a finite real number in [0, 1].")
    alpha = float(mixing)
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("mixing must be a finite real number in [0, 1].")
    if not isinstance(old, StatisticalSpectral):
        raise TypeError("old must be StatisticalSpectral.")
    raw = old.statistical
    if (
        not isinstance(raw, np.ndarray) or raw.ndim != 4
        or raw.shape[0] < 1 or raw.shape[1] < 1 or raw.shape[2] != raw.shape[3]
    ):
        raise ValueError("Channels must have shape (Nq, Nomega, A, A).")
    shape = raw.shape
    sources = (_channel_arrays(old, shape, "old"), _channel_arrays(new, shape, "new"))
    a = shape[-1]
    owned = 2 * _C * raw.size + (_C + 1) * a**2
    if owned > limit:
        raise ValueError("Channel mixing output and workspace exceed max_bytes.")
    for pair in sources:
        for value in pair:
            _finite(value, "channel mixing input")
    outputs = (np.empty(shape, dtype=np.complex128), np.empty(shape, dtype=np.complex128))
    scratch = np.empty((a, a), dtype=np.complex128)
    for before, after, output in zip(*sources, outputs, strict=True):
        for index in np.ndindex(shape[:-2]):
            np.multiply(before[index], 1.0 - alpha, out=output[index])
            np.multiply(after[index], alpha, out=scratch)
            np.add(output[index], scratch, out=output[index])
        _finite(output, "mixed channel")
    return StatisticalSpectral(*outputs)


def cubic_born_channels_packed_fft(
    grid: SignedFrequencyGrid, vertex: PackedTransverseCubicVertex,
    channels: StatisticalSpectral, *, external_tile: int = 16,
    workers: int = 8, bin_tile: int = 1,
    max_bytes: int = _DEFAULT_BYTES, max_contraction_work: int = _DEFAULT_WORK,
) -> StatisticalSpectral:
    """Evaluate independent statistical and spectral cubic Born channels.

    With ``B(X,Y) = V X Y conj(V)`` including the finite, zero-extended
    frequency convolution and the existing q routing, the outputs are
    ``k*(B(C,C) + B(D,D)/4)`` and ``k*(B(C,D) + B(D,C))``, where
    ``k = 0.5j * integration_weight / Nq``.  No Green factor is conjugated.
    Generic finite raw matrices are accepted without silently imposing their
    expected physical symmetries.  Every term is evaluated independently.

    Resource and input checks precede outputs and FFTs.  Only independent FFT
    bin chunks run in parallel; q routes and bilinear terms remain serial.
    """
    plan = plan_born_channels_fft(
        grid, vertex, external_tile=external_tile, workers=workers, bin_tile=bin_tile,
    )
    limit = _integer(max_bytes, "max_bytes")
    work_limit = _integer(max_contraction_work, "max_contraction_work")
    if plan.peak_owned_bytes > limit:
        raise ValueError("Channel FFT peak owned payload exceeds max_bytes.")
    if plan.contraction_work > work_limit:
        raise ValueError("Channel FFT exceeds max_contraction_work.")
    shape = (plan.n_q, plan.n_w, plan.n_active, plan.n_active)
    inputs = _channel_arrays(channels, shape, "channels")
    for value in inputs:
        _finite(value, "channel FFT input")
    outputs = (np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128))
    if not plan.n_active:
        return StatisticalSpectral(*outputs)
    routes = difference_q_indices(vertex.mesh, max_bytes=limit)
    prefactor = 0.5j * grid.integration_weight / plan.n_q
    transform_shape = (plan.n_q, plan.n_fft, plan.n_active, plan.n_active)
    transformed = (
        np.zeros(transform_shape, dtype=np.complex128),
        np.zeros(transform_shape, dtype=np.complex128),
    )
    with threadpool_limits(limits=1, user_api="blas"):
        for source, target in zip(inputs, transformed, strict=True):
            target[:, :plan.n_w] = source
            for q in range(plan.n_q):
                target[q] = np.fft.fft(target[q], axis=0)
            _finite(target, "transformed channel")
        c_fft, d_fft = transformed
        for channel_index, output in enumerate(outputs):
            for q in range(plan.n_q):
                sigma = np.zeros((plan.n_fft, plan.n_active, plan.n_active), dtype=np.complex128)
                if channel_index == 0:
                    # Accumulate DD first so its factor 1/4 is applied in place
                    # without another q-sized buffer or changes to C or D.
                    terms = ((d_fft, d_fft), (c_fft, c_fft))
                else:
                    terms = ((c_fft, d_fft), (d_fft, c_fft))
                for term_index, (left, right) in enumerate(terms):
                    for p in range(plan.n_q):
                        _contract_into(
                            vertex.coefficients[q, p], left[p], right[routes[q, p]],
                            sigma, plan.external_tile, plan.workers, plan.bin_tile,
                        )
                    if channel_index == 0 and term_index == 0:
                        sigma *= 0.25
                convolution = np.fft.ifft(sigma, axis=0)
                del sigma
                selected = convolution[grid.zero_index:grid.zero_index + plan.n_w]
                np.multiply(selected, prefactor, out=output[q])
                del selected, convolution
            _finite(output, "Born channel output")
    return StatisticalSpectral(*outputs)


def retarded_from_spectral(
    spectral: np.ndarray, grid: SignedFrequencyGrid, *,
    hilbert_backend: str = "fft", max_bytes: int = _DEFAULT_BYTES,
) -> np.ndarray:
    """Return ``D/2 + i*PV(D)`` directly from the independent raw D array.

    Accept shape ``(Nomega,A,A)`` or ``(Nq,Nomega,A,A)`` and return the same
    shape.  The same-grid singular term is omitted and frequencies outside the
    supplied grid are zero.  FFT mode uses a padded finite Toeplitz product;
    direct mode uses a bounded matrix scratch buffer.  This reconstruction
    does not make inconsistent D anti-Hermitian or force ``R-R^dagger = D``.
    Both modes retain independent raw-field consistency checks for the caller.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    limit = _integer(max_bytes, "max_bytes")
    if hilbert_backend not in ("fft", "direct"):
        raise ValueError("hilbert_backend must be 'fft' or 'direct'.")
    if (
        not isinstance(spectral, np.ndarray) or spectral.ndim not in (3, 4)
        or spectral.shape[-3] != grid.n_points
        or spectral.shape[-2] != spectral.shape[-1]
        or (spectral.ndim == 4 and spectral.shape[0] < 1)
    ):
        raise ValueError("spectral must have shape (Nomega,A,A) or (Nq,Nomega,A,A).")
    raw = _array(spectral, spectral.shape, "spectral")
    a = raw.shape[-1]
    nfft = 1 << (2 * grid.n_points - 2).bit_length()
    scratch = (_C + 1) * a**2
    if hilbert_backend == "fft" and a:
        # Two q-sized spectra can coexist during an FFT/IFFT.  Two kernel
        # arrays conservatively cover the kernel FFT's source and result.
        scratch += 2 * _C * nfft * (a**2 + 1)
    owned = _C * raw.size + scratch
    if owned > limit:
        raise ValueError("Retarded reconstruction output and workspace exceed max_bytes.")
    _finite(raw, "spectral")
    output = np.empty(raw.shape, dtype=np.complex128)
    if not a:
        return output
    source_q = raw[np.newaxis] if raw.ndim == 3 else raw
    output_q = output[np.newaxis] if output.ndim == 3 else output
    matrix = np.empty((a, a), dtype=np.complex128)
    if hilbert_backend == "direct":
        for source, target in zip(source_q, output_q, strict=True):
            for i in range(grid.n_points):
                np.multiply(source[i], 0.5, out=target[i])
                for j in range(grid.n_points):
                    if i != j:
                        weight = grid.integration_weight / (grid.values[i] - grid.values[j])
                        np.multiply(source[j], 1j * weight, out=matrix)
                        np.add(target[i], matrix, out=target[i])
    else:
        kernel = np.zeros(nfft, dtype=np.complex128)
        for lag in range(1, grid.n_points):
            weight = grid.integration_weight / (grid.step * lag)
            kernel[lag] = weight
            kernel[nfft - lag] = -weight
        kernel = np.fft.fft(kernel)
        for source, target in zip(source_q, output_q, strict=True):
            transformed = np.zeros((nfft, a, a), dtype=np.complex128)
            transformed[:grid.n_points] = source
            transformed = np.fft.fft(transformed, axis=0)
            transformed *= kernel[:, None, None]
            convolution = np.fft.ifft(transformed, axis=0)
            for w in range(grid.n_points):
                np.multiply(convolution[w], 1j, out=target[w])
                np.multiply(source[w], 0.5, out=matrix)
                np.add(target[w], matrix, out=target[w])
            del transformed, convolution
    _finite(output, "retarded reconstruction output")
    return output


__all__ = [
    "BornChannelsFFTPlan", "StatisticalSpectral", "cubic_born_channels_packed_fft",
    "mix_channels", "plan_born_channels_fft", "plan_born_channels_fft_dimensions",
    "retarded_from_spectral",
]
