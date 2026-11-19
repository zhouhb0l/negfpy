"""Unscaled, overlapping transfer of one complete-center-layer retarded kernel.

This is an explicit frozen-bath approximation, not a long-device Born map.
Every copy retains all 36 endpoint coordinates of the original 72-DOF device.
Copies share the same immutable spectrum; only their coordinate injections
differ. No PSD clipping, acoustic projection, fit, or coverage normalization
is performed. This module reads no files and starts no calculations.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


SOURCE_LAYER_DOFS = 24
SOURCE_LAYERS = 3
SOURCE_FULL_NDOF = SOURCE_LAYER_DOFS * SOURCE_LAYERS
SOURCE_ENDPOINT_ATOMS = (0, 2, 8, 9, 10, 11, 12, 13, 14, 15, 17, 19)
SOURCE_ACTIVE_DOFS = tuple(3 * atom + axis for atom in SOURCE_ENDPOINT_ATOMS for axis in range(3))
DEFAULT_RTOL = 1.0e-8
DEFAULT_ATOL = 1.0e-11


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    if value < 1:
        raise ValueError(f"{name} must be positive.")
    return int(value)


def _tolerances(rtol: float, atol: float) -> tuple[float, float]:
    if not np.isfinite([rtol, atol]).all() or rtol < 0 or atol <= 0:
        raise ValueError("Require finite rtol >= 0 and atol > 0.")
    return float(rtol), float(atol)


def _readonly(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    result = np.array(values, dtype=dtype, order="C", copy=True)
    result.setflags(write=False)
    return result


def _active_indices(value: np.ndarray) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != 1 or raw.dtype.kind not in "iu":
        raise TypeError("active_dofs must be a one-dimensional integer array.")
    if len(raw) != 36 or set(int(x) for x in raw) != set(SOURCE_ACTIVE_DOFS):
        raise ValueError("Require exactly the original 36 complete endpoint DOFs on atoms 0,2,8..15,17,19.")
    return _readonly(raw, np.dtype(np.int64))


def _frequencies(value: np.ndarray, n_w: int) -> np.ndarray:
    raw = np.asarray(value, dtype=np.float64)
    if raw.shape != (n_w,) or not np.isfinite(raw).all():
        raise ValueError("frequencies must be a finite vector matching the spectrum.")
    if n_w < 3 or n_w % 2 != 1 or np.any(np.diff(raw) <= 0):
        raise ValueError("Require an increasing odd signed grid with positive frequencies.")
    scale = max(float(np.max(np.abs(raw))), 1.0)
    if raw[n_w // 2] != 0.0 or not np.allclose(raw, -raw[::-1], rtol=0.0, atol=32 * np.finfo(float).eps * scale):
        raise ValueError("The signed grid must be symmetric and contain an exact central zero.")
    return _readonly(raw, np.dtype(np.float64))


def _negative_q_indices(value: np.ndarray | None, n_q: int) -> np.ndarray:
    if value is None:
        if n_q != 1:
            raise ValueError("Supply negative_q_indices explicitly for a multi-q spectrum.")
        value = np.array([0], dtype=np.int64)
    raw = np.asarray(value)
    if raw.shape != (n_q,) or raw.dtype.kind not in "iu":
        raise ValueError("negative_q_indices must be an integer vector of length Nq.")
    if set(int(q) for q in raw) != set(range(n_q)) or not np.array_equal(raw[raw], np.arange(n_q)):
        raise ValueError("negative_q_indices must be an involutive permutation.")
    return _readonly(raw, np.dtype(np.int64))


def broadening(retarded: np.ndarray) -> np.ndarray:
    """Return i(R - R-dagger), with frequency-squared solver units unchanged."""
    raw = np.asarray(retarded)
    return 1.0j * (raw - np.swapaxes(raw.conj(), -1, -2))


def psd_statistics(retarded: np.ndarray, frequencies: np.ndarray, *, rtol: float = DEFAULT_RTOL,
                   atol: float = DEFAULT_ATOL) -> dict:
    """Gate every positive-frequency broadening, without altering any matrix.

    For each matrix the permitted negative eigenvalue is atol + rtol*||Gamma||2.
    Tolerances are diagnostics; they never authorize clipping the input.
    """
    rtol, atol = _tolerances(rtol, atol)
    raw = np.asarray(retarded)
    omega = np.asarray(frequencies)
    if raw.ndim != 4 or raw.shape[-2:] != (36, 36) or omega.shape != (raw.shape[1],):
        raise ValueError("Require packed (Nq, Nomega, 36, 36) fields and matching frequencies.")
    if not np.isfinite(raw).all():
        raise ValueError("Retarded kernel must contain only finite values.")
    positive = np.flatnonzero(omega > 0.0)
    if not len(positive):
        raise ValueError("At least one positive frequency is required.")
    worst_ratio, minimum, worst = 0.0, float("inf"), None
    largest_scale = 0.0
    for q in range(raw.shape[0]):
        for w in positive:
            eigenvalues = np.linalg.eigvalsh(broadening(raw[q, w]))
            lo = float(eigenvalues[0])
            scale = float(np.max(np.abs(eigenvalues)))
            ratio = max(0.0, -lo) / (atol + rtol * scale)
            minimum = min(minimum, lo)
            largest_scale = max(largest_scale, scale)
            if worst is None or ratio > worst_ratio:
                worst_ratio = ratio
                worst = dict(q=int(q), frequency_index=int(w), omega=float(omega[w]),
                             minimum_eigenvalue=lo, spectral_scale=scale,
                             allowed_negative_eigenvalue=atol + rtol * scale)
    result = dict(passed=bool(worst_ratio <= 1.0), positive_frequency_count=int(len(positive)),
                  matrices_checked=int(raw.shape[0] * len(positive)),
                  minimum_eigenvalue=minimum, largest_spectral_scale=largest_scale,
                  worst_ratio=worst_ratio, worst=worst, rtol=rtol, atol=atol,
                  clipping_performed=False)
    if not result["passed"]:
        raise ValueError(f"Frozen retarded kernel is not passive: worst PSD ratio {worst_ratio:.9g}, location {worst}.")
    return result


def _reality_statistics(raw: np.ndarray, negative_q: np.ndarray, rtol: float, atol: float) -> dict:
    worst = 0.0
    maximum_absolute = 0.0
    for q, negative in enumerate(negative_q):
        for w in range(raw.shape[1]):
            matrix = raw[q, w]
            expected = raw[int(negative), raw.shape[1] - 1 - w].conj()
            error = float(np.max(np.abs(matrix - expected)))
            scale = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(expected))))
            maximum_absolute = max(maximum_absolute, error)
            worst = max(worst, error / (atol + rtol * scale))
    result = dict(passed=bool(worst <= 1.0), worst_ratio=worst,
                  maximum_absolute_residual=maximum_absolute, rtol=rtol, atol=atol,
                  relation="R[-q,-omega] = R[q,omega].conj(); no transpose", repair_performed=False)
    if not result["passed"]:
        raise ValueError(f"Retarded signed q-frequency reality failed: ratio {worst:.9g}.")
    return result


def _asr_statistics(raw: np.ndarray, translations: np.ndarray | None, active: np.ndarray,
                    gamma_q_index: int, rtol: float, atol: float) -> dict:
    if translations is None:
        return dict(checked=False, passed=None, projection_performed=False)
    vector = np.asarray(translations)
    if vector.shape == (SOURCE_FULL_NDOF, 3):
        vector = vector[active]
    if vector.shape != (36, 3) or not np.isfinite(vector).all():
        raise ValueError("source_translation_vectors must be finite with shape (36,3) or (72,3).")
    if not 0 <= gamma_q_index < raw.shape[0]:
        raise ValueError("gamma_q_index is outside the q grid.")
    vector_scale = float(np.linalg.norm(vector))
    if vector_scale <= 0.0:
        raise ValueError("Translation vectors must have nonzero norm.")
    worst = maximum = 0.0
    for matrix in raw[gamma_q_index]:
        error = max(float(np.linalg.norm(matrix @ vector)), float(np.linalg.norm(vector.conj().T @ matrix)))
        scale = float(np.linalg.norm(matrix)) * vector_scale
        maximum = max(maximum, error)
        worst = max(worst, error / (atol + rtol * scale))
    result = dict(checked=True, passed=bool(worst <= 1.0), worst_ratio=worst,
                  maximum_frobenius_residual=maximum, rtol=rtol, atol=atol,
                  projection_performed=False, gamma_q_index=int(gamma_q_index))
    if not result["passed"]:
        raise ValueError(f"Frozen source kernel violates supplied acoustic translations: ratio {worst:.9g}.")
    return result


@dataclass(frozen=True, slots=True)
class FrozenRetardedKernel:
    retarded: np.ndarray
    frequencies: np.ndarray
    active_dofs: np.ndarray
    negative_q_indices: np.ndarray
    validation: dict

    @property
    def n_positive(self) -> int:
        return int(np.count_nonzero(self.frequencies > 0.0))

    @property
    def npositive(self) -> int:
        return self.n_positive

    def gamma(self, q: int, frequency_index: int) -> np.ndarray:
        return broadening(self.retarded[q, frequency_index])


def make_frozen_kernel(sigma_retarded: np.ndarray, frequencies: np.ndarray, active_dofs: np.ndarray,
                       *, negative_q_indices: np.ndarray | None = None, gamma_q_index: int = 0,
                       source_translation_vectors: np.ndarray | None = None,
                       rtol: float = DEFAULT_RTOL, atol: float = DEFAULT_ATOL) -> FrozenRetardedKernel:
    """Validate and own an unchanged packed short-device retarded kernel.

    ``active_dofs`` is supplied by ``supported.vertex.active_dofs``. Its order
    is respected. Full72 input is accepted only when inactive entries are
    exactly zero, so extraction cannot hide discarded self-energy support.
    For multi-q input the caller must supply the actual q -> -q permutation.
    Acoustic ASR is gated only when physical translation vectors are supplied.
    """
    rtol, atol = _tolerances(rtol, atol)
    active = _active_indices(active_dofs)
    raw = np.asarray(sigma_retarded)
    if raw.ndim != 4 or raw.shape[0] < 1 or raw.shape[-2:] not in ((36, 36), (72, 72)):
        raise ValueError("sigma_retarded must have shape (Nq, Nomega,36,36) or (Nq,Nomega,72,72).")
    if raw.dtype.kind not in "fci" or not np.isfinite(raw).all():
        raise ValueError("sigma_retarded must contain finite numeric matrices.")
    omega = _frequencies(frequencies, raw.shape[1])
    negative_q = _negative_q_indices(negative_q_indices, raw.shape[0])
    if raw.shape[-1] == SOURCE_FULL_NDOF:
        inactive = np.setdiff1d(np.arange(SOURCE_FULL_NDOF), active)
        for q in range(raw.shape[0]):
            for w in range(raw.shape[1]):
                matrix = raw[q, w]
                if np.any(matrix[inactive] != 0) or np.any(matrix[:, inactive] != 0):
                    raise ValueError("Full source contains nonzero self-energy outside its exact active endpoint support.")
        raw = raw[..., active[:, None], active[None, :]]
    psd = psd_statistics(raw, omega, rtol=rtol, atol=atol)
    reality = _reality_statistics(raw, negative_q, rtol, atol)
    asr = _asr_statistics(raw, source_translation_vectors, active, gamma_q_index, rtol, atol)
    owned = _readonly(raw, np.dtype(np.complex128))
    validation = dict(psd=psd, retarded_reality=reality, acoustic_asr=asr,
                      source_full_ndof=SOURCE_FULL_NDOF, source_active_ndof=36,
                      source_center_atoms=8, source_endpoint_atoms=12,
                      source_atoms_by_layer=[2, 8, 2], coupling_rescaled=False,
                      coverage_normalized=False, field_projected=False)
    return FrozenRetardedKernel(owned, omega, active, negative_q, validation)


@dataclass(frozen=True, slots=True)
class TiledFrozenKernel:
    kernel: FrozenRetardedKernel
    n_center_layers: int
    target_ndof: int
    probe_coordinates: tuple[np.ndarray, ...]

    @property
    def supports(self) -> tuple[np.ndarray, ...]:
        return self.probe_coordinates

    def iter_probes(self, q: int, frequency_index: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Yield (coordinates, full36 R block, full36 Gamma block), sharing R."""
        retarded = self.kernel.retarded[q, frequency_index]
        gamma = self.kernel.gamma(q, frequency_index)
        for coordinates in self.probe_coordinates:
            yield coordinates, retarded, gamma

    def total_retarded(self, q: int, frequency_index: int) -> np.ndarray:
        """Allocate one target matrix and sum each overlapping injection once."""
        result = np.zeros((self.target_ndof, self.target_ndof), dtype=np.complex128)
        block = self.kernel.retarded[q, frequency_index]
        for coordinates in self.probe_coordinates:
            result[np.ix_(coordinates, coordinates)] += block
        return result

    def total_gamma(self, q: int, frequency_index: int) -> np.ndarray:
        result = np.zeros((self.target_ndof, self.target_ndof), dtype=np.complex128)
        block = self.kernel.gamma(q, frequency_index)
        for coordinates in self.probe_coordinates:
            result[np.ix_(coordinates, coordinates)] += block
        return result

    def neighborhood_statistics(self) -> dict:
        coverage = np.zeros(self.target_ndof, dtype=np.int64)
        for coordinates in self.probe_coordinates:
            coverage[coordinates] += 1
        values, counts = np.unique(coverage, return_counts=True)
        return dict(n_center_layers=self.n_center_layers, n_energy_centers=8 * self.n_center_layers,
                    device_layers=self.n_center_layers + 2, target_ndof=self.target_ndof,
                    n_probes=len(self.probe_coordinates), endpoint_dofs_per_probe=36,
                    endpoint_atoms_per_probe=12, full_kernel_copies=True, endpoint_overlap_allowed=True,
                    coordinate_coverage_histogram={str(int(v)): int(c) for v, c in zip(values, counts)},
                    overlapping_coordinates=int(np.count_nonzero(coverage > 1)),
                    unique_endpoint_coordinates=int(np.count_nonzero(coverage)),
                    total_injected_coordinates=int(np.sum(coverage)),
                    target_center_layers=list(range(1, self.n_center_layers + 1)),
                    probe_layer_spans=[[s, s + 2] for s in range(self.n_center_layers)],
                    coupling_rescaled=False, coverage_normalized=False,
                    approximation="Independent frozen center-layer baths retain within-source-layer vertex pairs; cross-layer Born vertex pairs are omitted.")


def tile_frozen_kernel(kernel: FrozenRetardedKernel, n_center_layers: int) -> TiledFrozenKernel:
    if not isinstance(kernel, FrozenRetardedKernel):
        raise TypeError("kernel must be made by make_frozen_kernel.")
    count = _positive_integer(n_center_layers, "n_center_layers")
    target_ndof = SOURCE_LAYER_DOFS * (count + 2)
    supports = tuple(_readonly(kernel.active_dofs + SOURCE_LAYER_DOFS * shift, np.dtype(np.int64))
                     for shift in range(count))
    return TiledFrozenKernel(kernel, count, target_ndof, supports)


__all__ = ["FrozenRetardedKernel", "TiledFrozenKernel", "SOURCE_ACTIVE_DOFS", "broadening",
           "make_frozen_kernel", "psd_statistics", "tile_frozen_kernel"]
