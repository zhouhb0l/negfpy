"""Third-order anharmonic interaction tools.

Development tracks exposed here:
- lowest order (implemented)
- mean field (implemented as a displacement-renormalized static correction)
- SCBA (implemented as a self-consistent quasiparticle Born closure in a
  harmonic mode basis)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from negfpy.core.types import Device1D


Array = np.ndarray
KB_EFFECTIVE = 1.0  # toy-model convention: hbar = k_B = 1


@dataclass(frozen=True)
class ThirdOrderInteraction:
    """Cubic interaction tensor in the device-coordinate basis."""

    phi3: Array

    def __post_init__(self) -> None:
        phi3 = np.asarray(self.phi3, dtype=np.complex128)
        if phi3.ndim != 3:
            raise ValueError("phi3 must be a rank-3 tensor.")
        if phi3.shape[0] != phi3.shape[1] or phi3.shape[1] != phi3.shape[2]:
            raise ValueError("phi3 must be a cubic tensor with shape (dim, dim, dim).")


def _symmetrize_matrix(matrix: Array) -> Array:
    arr = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (arr + arr.conj().T)


def _assemble_device_matrix_dense(device: Device1D) -> Array:
    dim = device.n_layers * device.dof_per_layer
    dmat = np.zeros((dim, dim), dtype=np.complex128)
    npl = device.dof_per_layer

    for i, block in enumerate(device.onsite_blocks):
        sl = slice(i * npl, (i + 1) * npl)
        dmat[sl, sl] = np.asarray(block, dtype=np.complex128)

    for i, block in enumerate(device.coupling_blocks):
        sli = slice(i * npl, (i + 1) * npl)
        slj = slice((i + 1) * npl, (i + 2) * npl)
        blk = np.asarray(block, dtype=np.complex128)
        dmat[sli, slj] = blk
        dmat[slj, sli] = blk.conj().T

    return dmat


def _mode_data_from_device(device: Device1D, *, frequency_floor: float) -> tuple[Array, Array]:
    dmat = _assemble_device_matrix_dense(device)
    evals, evecs = np.linalg.eigh(dmat)
    freqs = np.sqrt(np.clip(evals.real, a_min=frequency_floor * frequency_floor, a_max=None))
    return freqs, np.asarray(evecs, dtype=np.complex128)


def _bose_occupation(freqs: Array, temperature: float) -> Array:
    if temperature <= 0.0:
        return np.zeros_like(freqs, dtype=float)
    x = np.asarray(freqs, dtype=float) / (KB_EFFECTIVE * float(temperature))
    out = np.zeros_like(x)
    small = x < 1e-6
    regular = (x >= 1e-6) & (x < 700.0)
    out[small] = 1.0 / np.maximum(x[small], 1e-12)
    out[regular] = 1.0 / np.expm1(x[regular])
    return out


def _mode_couplings(phi3: Array, mode_vectors: Array) -> Array:
    return np.einsum("ijk,ia,jb,kc->abc", phi3, mode_vectors, mode_vectors, mode_vectors, optimize=True)


def _project_green_function_to_modes(green_function: Array, mode_vectors: Array) -> Array:
    g = np.asarray(green_function, dtype=np.complex128)
    vecs = np.asarray(mode_vectors, dtype=np.complex128)
    return vecs.conj().T @ g @ vecs


def _effective_mode_data_from_green_function(
    omega: float,
    green_function_modes: Array,
    harmonic_frequencies: Array,
    *,
    frequency_floor: float,
    broadening_floor: float,
) -> tuple[Array, Array]:
    diag_g = np.diag(np.asarray(green_function_modes, dtype=np.complex128))
    freqs0 = np.asarray(harmonic_frequencies, dtype=float)
    n_modes = freqs0.size

    dressed_freqs = np.zeros(n_modes, dtype=float)
    dressed_broadening = np.zeros(n_modes, dtype=float)
    omega_ref = max(abs(float(omega)), float(frequency_floor))

    for q in range(n_modes):
        gq = diag_g[q]
        if abs(gq) < 1e-30:
            sigma_est = 0.0j
        else:
            sigma_est = (float(omega) ** 2) - (float(freqs0[q]) ** 2) - (1.0 / gq)

        dressed_w2 = max(float(frequency_floor) ** 2, float(freqs0[q]) ** 2 + float(np.real(sigma_est)))
        dressed_freqs[q] = np.sqrt(dressed_w2)

        gamma_q = -float(np.imag(sigma_est)) / max(2.0 * max(dressed_freqs[q], omega_ref), float(frequency_floor))
        dressed_broadening[q] = max(float(broadening_floor), gamma_q)

    return dressed_freqs, dressed_broadening


@dataclass
class ThirdOrderLowestOrderModel:
    """Callable cubic lowest-order self-energy model for toy transport runs.

    This implementation follows a mode-space quasiparticle-style lowest-order
    retarded self-energy using broadened energy denominators. It is intended as
    a careful first development step for FPU-alpha chain benchmarks.
    """

    interaction: ThirdOrderInteraction
    mode_frequencies: Array
    mode_vectors: Array
    temperature: float
    broadening: float = 1e-3
    frequency_floor: float = 1e-8
    couplings: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.broadening <= 0.0:
            raise ValueError("broadening must be positive.")
        if self.frequency_floor <= 0.0:
            raise ValueError("frequency_floor must be positive.")

        freqs = np.asarray(self.mode_frequencies, dtype=float)
        vecs = np.asarray(self.mode_vectors, dtype=np.complex128)
        dim = freqs.size
        if vecs.shape != (dim, dim):
            raise ValueError("mode_vectors must have shape (n_modes, n_modes).")
        if np.asarray(self.interaction.phi3).shape != (dim, dim, dim):
            raise ValueError("interaction tensor shape must match the harmonic mode basis dimension.")

        object.__setattr__(self, "mode_frequencies", freqs)
        object.__setattr__(self, "mode_vectors", vecs)
        object.__setattr__(self, "couplings", _mode_couplings(np.asarray(self.interaction.phi3), vecs))

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del green_function, iteration
        return third_order_lowest_order_self_energy(
            omega=omega,
            interaction=self.interaction,
            temperature=self.temperature,
            broadening=self.broadening,
            frequency_floor=self.frequency_floor,
            mode_frequencies=self.mode_frequencies,
            mode_vectors=self.mode_vectors,
            mode_couplings=self.couplings,
        )


@dataclass
class ThirdOrderMeanFieldModel:
    """Static mean-field cubic self-energy from a supplied average displacement."""

    interaction: ThirdOrderInteraction
    mean_displacement: Array

    def __post_init__(self) -> None:
        disp = np.asarray(self.mean_displacement, dtype=np.complex128).ravel()
        dim = np.asarray(self.interaction.phi3).shape[0]
        if disp.shape != (dim,):
            raise ValueError("mean_displacement must have shape (dim,).")
        object.__setattr__(self, "mean_displacement", disp)

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del omega, green_function, iteration
        return third_order_mean_field_self_energy(
            interaction=self.interaction,
            mean_displacement=self.mean_displacement,
        )


@dataclass
class ThirdOrderSCBAModel:
    """Cubic SCBA-like model using dressed modal lines in a fixed harmonic basis.

    This is a general self-consistent quasiparticle closure: the current
    retarded Green function is projected onto the harmonic modes, which are then
    used to update dressed mode frequencies and broadenings inside the cubic
    Born self-energy.
    """

    interaction: ThirdOrderInteraction
    mode_frequencies: Array
    mode_vectors: Array
    temperature: float
    broadening: float = 1e-3
    frequency_floor: float = 1e-8
    couplings: Array = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if self.broadening <= 0.0:
            raise ValueError("broadening must be positive.")
        if self.frequency_floor <= 0.0:
            raise ValueError("frequency_floor must be positive.")

        freqs = np.asarray(self.mode_frequencies, dtype=float)
        vecs = np.asarray(self.mode_vectors, dtype=np.complex128)
        dim = freqs.size
        if vecs.shape != (dim, dim):
            raise ValueError("mode_vectors must have shape (n_modes, n_modes).")
        if np.asarray(self.interaction.phi3).shape != (dim, dim, dim):
            raise ValueError("interaction tensor shape must match the harmonic mode basis dimension.")

        object.__setattr__(self, "mode_frequencies", freqs)
        object.__setattr__(self, "mode_vectors", vecs)
        object.__setattr__(self, "couplings", _mode_couplings(np.asarray(self.interaction.phi3), vecs))

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del iteration
        return third_order_scba_self_energy(
            omega=omega,
            interaction=self.interaction,
            green_function=green_function,
            temperature=self.temperature,
            broadening=self.broadening,
            frequency_floor=self.frequency_floor,
            mode_frequencies=self.mode_frequencies,
            mode_vectors=self.mode_vectors,
            mode_couplings=self.couplings,
        )


def third_order_lowest_order_model_from_device(
    device: Device1D,
    interaction: ThirdOrderInteraction,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> ThirdOrderLowestOrderModel:
    """Build a lowest-order cubic self-energy model from a harmonic device."""

    freqs, vecs = _mode_data_from_device(device, frequency_floor=frequency_floor)
    return ThirdOrderLowestOrderModel(
        interaction=interaction,
        mode_frequencies=freqs,
        mode_vectors=vecs,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )


def third_order_mean_field_model_from_displacement(
    interaction: ThirdOrderInteraction,
    *,
    mean_displacement: Array,
) -> ThirdOrderMeanFieldModel:
    """Build a static cubic mean-field model from an average displacement."""

    return ThirdOrderMeanFieldModel(
        interaction=interaction,
        mean_displacement=np.asarray(mean_displacement, dtype=np.complex128),
    )


def third_order_scba_model_from_device(
    device: Device1D,
    interaction: ThirdOrderInteraction,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> ThirdOrderSCBAModel:
    """Build a cubic SCBA-like self-energy model from a harmonic device."""

    freqs, vecs = _mode_data_from_device(device, frequency_floor=frequency_floor)
    return ThirdOrderSCBAModel(
        interaction=interaction,
        mode_frequencies=freqs,
        mode_vectors=vecs,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )


def third_order_lowest_order_self_energy(
    omega: float,
    interaction: ThirdOrderInteraction,
    *,
    temperature: float | None = None,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
    mode_frequencies: Array,
    mode_vectors: Array,
    mode_couplings: Array | None = None,
) -> Array:
    """Return a mode-space cubic lowest-order retarded self-energy.

    The current implementation uses broadened denominators in a quasiparticle
    mode basis. This is the intended first benchmarkable step before SCBA.
    """

    if temperature is None:
        raise ValueError("temperature must be provided for third-order lowest-order self-energy.")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    freqs = np.asarray(mode_frequencies, dtype=float)
    vecs = np.asarray(mode_vectors, dtype=np.complex128)
    dim = freqs.size
    if vecs.shape != (dim, dim):
        raise ValueError("mode_vectors must have shape (n_modes, n_modes).")
    if np.asarray(interaction.phi3).shape != (dim, dim, dim):
        raise ValueError("interaction tensor shape must match mode-space dimension.")

    couplings = _mode_couplings(np.asarray(interaction.phi3), vecs) if mode_couplings is None else np.asarray(
        mode_couplings, dtype=np.complex128
    )
    occ = _bose_occupation(freqs, float(temperature))
    zeta = 1j * float(broadening)

    sigma_modes = np.zeros(dim, dtype=np.complex128)
    for q in range(dim):
        wq = max(freqs[q], float(frequency_floor))
        accum = 0.0j
        for r in range(dim):
            wr = max(freqs[r], float(frequency_floor))
            nr = occ[r]
            for s in range(dim):
                ws = max(freqs[s], float(frequency_floor))
                ns = occ[s]
                gqrs = couplings[q, r, s]
                pref = (abs(gqrs) ** 2) / (8.0 * wq * wr * ws)
                decay = (nr + ns + 1.0) * (
                    1.0 / (omega - wr - ws + zeta) - 1.0 / (omega + wr + ws + zeta)
                )
                collision = 2.0 * (nr - ns) * (
                    1.0 / (omega - wr + ws + zeta) - 1.0 / (omega + wr - ws + zeta)
                )
                accum = accum + pref * (decay + collision)
        sigma_modes[q] = accum

    return vecs @ np.diag(sigma_modes) @ vecs.conj().T


def third_order_mean_field_self_energy(
    interaction: ThirdOrderInteraction,
    *,
    mean_displacement: Array,
) -> Array:
    """Return the cubic mean-field self-energy from ``<u>``.

    For a cubic interaction, the simplest mean-field decoupling renormalizes the
    quadratic problem through the average displacement.
    """

    phi3 = np.asarray(interaction.phi3, dtype=np.complex128)
    disp = np.asarray(mean_displacement, dtype=np.complex128).ravel()
    dim = phi3.shape[0]
    if disp.shape != (dim,):
        raise ValueError("mean_displacement must have shape (dim,).")

    sigma = np.einsum("ijk,k->ij", phi3, disp, optimize=True)
    return _symmetrize_matrix(sigma)


def third_order_scba_self_energy(
    omega: float,
    interaction: ThirdOrderInteraction,
    green_function: Array,
    *,
    temperature: float | None = None,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
    mode_frequencies: Array,
    mode_vectors: Array,
    mode_couplings: Array | None = None,
) -> Array:
    """Return a cubic SCBA-like retarded self-energy.

    This is a self-consistent quasiparticle Born closure in a fixed harmonic
    mode basis. The current retarded Green function provides dressed mode
    frequencies and linewidths, which are then used inside the cubic Born
    expression. It is a practical material-ready stepping stone toward fuller
    frequency-convolution SCBA.
    """

    if temperature is None:
        raise ValueError("temperature must be provided for third-order SCBA self-energy.")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    freqs0 = np.asarray(mode_frequencies, dtype=float)
    vecs = np.asarray(mode_vectors, dtype=np.complex128)
    dim = freqs0.size
    if np.asarray(green_function, dtype=np.complex128).shape != (dim, dim):
        raise ValueError("green_function must have shape (dim, dim).")
    if vecs.shape != (dim, dim):
        raise ValueError("mode_vectors must have shape (n_modes, n_modes).")
    if np.asarray(interaction.phi3).shape != (dim, dim, dim):
        raise ValueError("interaction tensor shape must match mode-space dimension.")

    couplings = _mode_couplings(np.asarray(interaction.phi3), vecs) if mode_couplings is None else np.asarray(
        mode_couplings, dtype=np.complex128
    )
    g_modes = _project_green_function_to_modes(green_function=green_function, mode_vectors=vecs)
    dressed_freqs, dressed_broadening = _effective_mode_data_from_green_function(
        omega=float(omega),
        green_function_modes=g_modes,
        harmonic_frequencies=freqs0,
        frequency_floor=float(frequency_floor),
        broadening_floor=float(broadening),
    )
    occ = _bose_occupation(dressed_freqs, float(temperature))

    sigma_modes = np.zeros(dim, dtype=np.complex128)
    for q in range(dim):
        wq = max(dressed_freqs[q], float(frequency_floor))
        accum = 0.0j
        for r in range(dim):
            wr = max(dressed_freqs[r], float(frequency_floor))
            nr = occ[r]
            for s in range(dim):
                ws = max(dressed_freqs[s], float(frequency_floor))
                ns = occ[s]
                gqrs = couplings[q, r, s]
                pref = (abs(gqrs) ** 2) / (8.0 * wq * wr * ws)
                z_rs = 1j * (dressed_broadening[r] + dressed_broadening[s])
                decay = (nr + ns + 1.0) * (
                    1.0 / (float(omega) - wr - ws + z_rs) - 1.0 / (float(omega) + wr + ws + z_rs)
                )
                collision = 2.0 * (nr - ns) * (
                    1.0 / (float(omega) - wr + ws + z_rs) - 1.0 / (float(omega) + wr - ws + z_rs)
                )
                accum = accum + pref * (decay + collision)
        sigma_modes[q] = accum

    return vecs @ np.diag(sigma_modes) @ vecs.conj().T
