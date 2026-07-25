"""Retarded self-energy reconstruction from Keldysh components."""

from __future__ import annotations

from typing import Literal
import warnings

import numpy as np
from scipy.fft import fft, ifft, next_fast_len

from .grid import SignedFrequencyGrid


Array = np.ndarray
RetardedStrategy = Literal["causal_hilbert", "spectral_only"]
HilbertBackend = Literal["direct", "fft"]

_SPECTRAL_ONLY_WARNING = (
    "retarded_strategy='spectral_only' omits the principal-value "
    "Kramers-Kronig term and is not a release-certified approximation"
)


def _validated_spectrum(
    values: Array,
    *,
    name: str,
    n_frequency: int,
) -> Array:
    """Return a finite square matrix spectrum as complex128."""

    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 3 or array.shape[0] != n_frequency:
        raise ValueError(
            f"{name} must have shape (n_frequency, ndof, ndof) "
            f"with n_frequency={n_frequency}; got {array.shape}."
        )
    if array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} matrices must be square; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def principal_value_cauchy(
    values: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    """Evaluate ``PV integral dω'/(2π) values(ω')/(ω-ω')`` directly.

    The singular same-grid-point contribution is excluded, as required by the
    Cauchy principal value.  No endpoint extrapolation or periodic wrapping is
    applied.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    spectrum = _validated_spectrum(
        values,
        name="values",
        n_frequency=grid.n_points,
    )

    omega = grid.values
    result = np.empty_like(spectrum)
    all_indices = np.arange(grid.n_points)
    for output_index, output_frequency in enumerate(omega):
        keep = all_indices != output_index
        weights = (
            grid.integration_weight
            / (output_frequency - omega[keep])
        )
        result[output_index] = np.einsum(
            "j,jab->ab",
            weights,
            spectrum[keep],
            optimize=True,
        )
    return result


def principal_value_cauchy_fft(
    values: Array,
    grid: SignedFrequencyGrid,
) -> Array:
    """Evaluate the finite-grid principal-value sum by a padded FFT.

    This is an exact FFT acceleration of :func:`principal_value_cauchy` up to
    floating-point roundoff; it does not introduce a periodic Hilbert
    transform.  On the uniform grid,

    ``integration_weight / (omega[i] - omega[j])
       = 1 / (2*pi*(i-j))``.

    The resulting finite Toeplitz product is embedded in a circulant array
    whose length is at least ``2*n_frequency - 1``.  Positive and negative
    lags therefore occupy disjoint entries, the zero lag is explicitly zero,
    and values outside the supplied frequency interval remain zero.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    spectrum = _validated_spectrum(
        values,
        name="values",
        n_frequency=grid.n_points,
    )

    n_frequency = grid.n_points
    padded_length = next_fast_len(2 * n_frequency - 1)
    offsets = np.arange(1, n_frequency, dtype=np.float64)
    kernel = np.zeros(padded_length, dtype=np.float64)
    lag_weights = (
        grid.integration_weight
        / (grid.step * offsets)
    )
    kernel[1:n_frequency] = lag_weights
    kernel[padded_length - np.arange(1, n_frequency)] = -lag_weights

    kernel_transform = fft(kernel, workers=1)
    spectrum_transform = fft(
        spectrum,
        n=padded_length,
        axis=0,
        workers=1,
    )
    embedded = ifft(
        kernel_transform[:, None, None] * spectrum_transform,
        axis=0,
        workers=1,
    )
    return np.asarray(embedded[:n_frequency], dtype=np.complex128)


def principal_value_cauchy_fft_scratch_bytes(
    n_frequency: int,
    ndof: int,
) -> int:
    """Conservative peak scratch estimate for the padded FFT PV transform."""

    for value, name in (
        (n_frequency, "n_frequency"),
        (ndof, "ndof"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
        ):
            raise ValueError(f"{name} must be a positive integer.")
    count = int(n_frequency)
    dimension = int(ndof)
    padded_length = next_fast_len(2 * count - 1)
    # Kernel transform plus input, product, and inverse-transform spectra.
    return int(
        np.dtype(np.complex128).itemsize
        * padded_length
        * (1 + 3 * dimension**2)
    )


def retarded_from_keldysh(
    sigma_lesser: Array,
    sigma_greater: Array,
    grid: SignedFrequencyGrid,
    strategy: RetardedStrategy = "causal_hilbert",
    *,
    hilbert_backend: HilbertBackend = "direct",
) -> Array:
    """Reconstruct a retarded self-energy from lesser/greater components.

    ``causal_hilbert`` implements Eq. (7) of Guo et al.,

    ``Sigma_r = 0.5 (Sigma_greater - Sigma_lesser)
               + i PV integral dω'/(2π)
                   (Sigma_greater-Sigma_lesser)/(ω-ω')``.

    ``spectral_only`` retains only the first term.  It is exposed solely for
    controlled diagnostics and emits a warning because it omits the dispersive
    Kramers-Kronig contribution.

    ``hilbert_backend`` changes only how the same finite-grid
    principal-value sum is evaluated. ``"direct"`` remains the reference;
    ``"fft"`` uses the tested zero-padded Toeplitz embedding.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    lesser = _validated_spectrum(
        sigma_lesser,
        name="sigma_lesser",
        n_frequency=grid.n_points,
    )
    greater = _validated_spectrum(
        sigma_greater,
        name="sigma_greater",
        n_frequency=grid.n_points,
    )
    if lesser.shape != greater.shape:
        raise ValueError(
            "sigma_lesser and sigma_greater must have the same shape; "
            f"got {lesser.shape} and {greater.shape}."
        )
    if strategy not in {"causal_hilbert", "spectral_only"}:
        raise ValueError(
            "strategy must be 'causal_hilbert' or 'spectral_only'; "
            f"got {strategy!r}."
        )
    if hilbert_backend not in {"direct", "fft"}:
        raise ValueError(
            "hilbert_backend must be 'direct' or 'fft'; "
            f"got {hilbert_backend!r}."
        )

    difference = greater - lesser
    spectral_part = 0.5 * difference

    if strategy == "spectral_only":
        warnings.warn(
            _SPECTRAL_ONLY_WARNING,
            RuntimeWarning,
            stacklevel=2,
        )
        return spectral_part
    if not np.any(difference):
        return np.zeros_like(difference)
    transform = (
        principal_value_cauchy
        if hilbert_backend == "direct"
        else principal_value_cauchy_fft
    )
    return spectral_part + 1j * transform(difference, grid)


__all__ = [
    "HilbertBackend",
    "RetardedStrategy",
    "principal_value_cauchy",
    "principal_value_cauchy_fft",
    "principal_value_cauchy_fft_scratch_bytes",
    "retarded_from_keldysh",
]
