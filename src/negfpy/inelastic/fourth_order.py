"""Fourth-order anharmonic interaction tools.

Development tracks exposed here:
- lowest order (implemented as a one-shot harmonic-covariance closure)
- mean field (implemented as a self-consistent covariance-renormalized correction)
- SCBA (implemented as a self-consistent quasiparticle quartic closure in a
  harmonic mode basis)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from negfpy.core.types import Device1D

from .third_order import (
    _assemble_device_matrix_dense,
    _bose_occupation,
    _effective_mode_data_from_green_function,
    _mode_data_from_device,
    _project_green_function_to_modes,
)

Array = np.ndarray


@dataclass(frozen=True)
class FourthOrderInteraction:
    """Quartic interaction tensor in the device-coordinate basis."""

    phi4: Array

    def __post_init__(self) -> None:
        phi4 = np.asarray(self.phi4, dtype=np.complex128)
        if phi4.ndim != 4:
            raise ValueError("phi4 must be a rank-4 tensor.")
        if len(set(phi4.shape)) != 1:
            raise ValueError("phi4 must have shape (dim, dim, dim, dim).")


def _symmetrize_matrix(matrix: Array) -> Array:
    arr = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (arr + arr.conj().T)


def _harmonic_covariance_from_modes(
    mode_frequencies: Array,
    mode_vectors: Array,
    *,
    temperature: float,
    frequency_floor: float,
) -> Array:
    freqs = np.asarray(mode_frequencies, dtype=float)
    vecs = np.asarray(mode_vectors, dtype=np.complex128)
    occ = _bose_occupation(freqs, float(temperature))
    weights = (occ + 0.5) / np.maximum(freqs, float(frequency_floor))
    return np.einsum("iq,q,jq->ij", vecs, weights, vecs.conj(), optimize=True)


def _mode_couplings(phi4: Array, mode_vectors: Array) -> Array:
    return np.einsum(
        "ijkl,ia,jb,kc,ld->abcd",
        phi4,
        mode_vectors,
        mode_vectors,
        mode_vectors,
        mode_vectors,
        optimize=True,
    )


@dataclass
class FourthOrderLowestOrderModel:
    """One-shot quartic lowest-order self-energy from a harmonic covariance."""

    interaction: FourthOrderInteraction
    covariance: Array
    sigma_static: Array | None = None

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=np.complex128)
        dim = np.asarray(self.interaction.phi4).shape[0]
        if cov.shape != (dim, dim):
            raise ValueError("covariance must have shape (dim, dim).")
        object.__setattr__(self, "covariance", _symmetrize_matrix(cov))
        sigma = self.sigma_static
        if sigma is None:
            sigma = fourth_order_lowest_order_self_energy(
                omega=0.0,
                interaction=self.interaction,
                covariance=self.covariance,
            )
        sigma_arr = np.asarray(sigma, dtype=np.complex128)
        if sigma_arr.shape != (dim, dim):
            raise ValueError("sigma_static must have shape (dim, dim).")
        object.__setattr__(self, "sigma_static", _symmetrize_matrix(sigma_arr))

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del omega, green_function, iteration
        return np.asarray(self.sigma_static, dtype=np.complex128)


@dataclass
class FourthOrderMeanFieldModel:
    """Static quartic mean-field self-energy with optional SCF metadata."""

    interaction: FourthOrderInteraction
    covariance: Array
    sigma_static: Array | None = None
    converged: bool = True
    iterations: int = 0
    residual: float = 0.0

    def __post_init__(self) -> None:
        cov = np.asarray(self.covariance, dtype=np.complex128)
        dim = np.asarray(self.interaction.phi4).shape[0]
        if cov.shape != (dim, dim):
            raise ValueError("covariance must have shape (dim, dim).")
        object.__setattr__(self, "covariance", _symmetrize_matrix(cov))
        sigma = self.sigma_static
        if sigma is None:
            sigma = fourth_order_mean_field_self_energy(
                interaction=self.interaction,
                covariance=self.covariance,
            )
        sigma_arr = np.asarray(sigma, dtype=np.complex128)
        if sigma_arr.shape != (dim, dim):
            raise ValueError("sigma_static must have shape (dim, dim).")
        object.__setattr__(self, "sigma_static", _symmetrize_matrix(sigma_arr))

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del omega, green_function, iteration
        return np.asarray(self.sigma_static, dtype=np.complex128)


@dataclass
class FourthOrderSCBAModel:
    """Quartic SCBA-like model with dressed Hartree shift and modal damping.

    The exact quartic Keldysh SCBA requires full frequency convolutions. For the
    current development stage we keep the same philosophy as the cubic SCBA
    implementation: use the current retarded Green function to extract dressed
    quasiparticle frequencies and linewidths in a fixed harmonic basis, then
    update a quartic self-energy containing:
    - a dressed Hartree shift from the current modal covariance
    - a positive semidefinite modal damping closure from quartic mode couplings
    """

    interaction: FourthOrderInteraction
    mode_frequencies: Array
    mode_vectors: Array
    temperature: float
    broadening: float = 1e-3
    frequency_floor: float = 1e-8
    couplings: Array | None = None

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
        if np.asarray(self.interaction.phi4).shape != (dim, dim, dim, dim):
            raise ValueError("interaction tensor shape must match the harmonic mode basis dimension.")

        object.__setattr__(self, "mode_frequencies", freqs)
        object.__setattr__(self, "mode_vectors", vecs)
        couplings = self.couplings
        if couplings is None:
            couplings = _mode_couplings(np.asarray(self.interaction.phi4), vecs)
        couplings_arr = np.asarray(couplings, dtype=np.complex128)
        if couplings_arr.shape != (dim, dim, dim, dim):
            raise ValueError("mode-space quartic couplings must have shape (dim, dim, dim, dim).")
        object.__setattr__(self, "couplings", couplings_arr)

    def __call__(self, omega: float, green_function: Array, iteration: int) -> Array:
        del iteration
        return fourth_order_scba_self_energy(
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


def fourth_order_lowest_order_model_from_covariance(
    interaction: FourthOrderInteraction,
    *,
    covariance: Array,
) -> FourthOrderLowestOrderModel:
    """Build a quartic lowest-order model from a harmonic covariance matrix."""

    return FourthOrderLowestOrderModel(
        interaction=interaction,
        covariance=np.asarray(covariance, dtype=np.complex128),
    )


def fourth_order_lowest_order_model_from_device(
    device: Device1D,
    interaction: FourthOrderInteraction,
    *,
    temperature: float,
    frequency_floor: float = 1e-8,
) -> FourthOrderLowestOrderModel:
    """Build a quartic lowest-order model from the harmonic device modes."""

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    dmat0 = _assemble_device_matrix_dense(device)
    dim = dmat0.shape[0]
    phi4 = np.asarray(interaction.phi4, dtype=np.complex128)
    if phi4.shape != (dim, dim, dim, dim):
        raise ValueError("interaction tensor shape must match the device dimension.")

    freqs0, vecs0 = _mode_data_from_device(device, frequency_floor=frequency_floor)
    covariance = _harmonic_covariance_from_modes(
        mode_frequencies=freqs0,
        mode_vectors=vecs0,
        temperature=float(temperature),
        frequency_floor=float(frequency_floor),
    )
    sigma = fourth_order_lowest_order_self_energy(
        omega=0.0,
        interaction=interaction,
        covariance=covariance,
    )
    return FourthOrderLowestOrderModel(
        interaction=interaction,
        covariance=covariance,
        sigma_static=sigma,
    )


def fourth_order_mean_field_model_from_covariance(
    interaction: FourthOrderInteraction,
    *,
    covariance: Array,
) -> FourthOrderMeanFieldModel:
    """Build a quartic mean-field model from a displacement covariance matrix."""

    return FourthOrderMeanFieldModel(
        interaction=interaction,
        covariance=np.asarray(covariance, dtype=np.complex128),
    )


def fourth_order_scba_model_from_device(
    device: Device1D,
    interaction: FourthOrderInteraction,
    *,
    temperature: float,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
) -> FourthOrderSCBAModel:
    """Build a quartic SCBA-like self-energy model from a harmonic device."""

    freqs, vecs = _mode_data_from_device(device, frequency_floor=frequency_floor)
    return FourthOrderSCBAModel(
        interaction=interaction,
        mode_frequencies=freqs,
        mode_vectors=vecs,
        temperature=temperature,
        broadening=broadening,
        frequency_floor=frequency_floor,
    )


def fourth_order_mean_field_model_from_device(
    device: Device1D,
    interaction: FourthOrderInteraction,
    *,
    temperature: float,
    max_iter: int = 100,
    mixing: float = 0.5,
    tol: float = 1e-8,
    frequency_floor: float = 1e-8,
    raise_on_nonconvergence: bool = False,
) -> FourthOrderMeanFieldModel:
    """Build a self-consistent quartic mean-field model from a harmonic device."""

    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must be in (0, 1].")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    dmat0 = _assemble_device_matrix_dense(device)
    dim = dmat0.shape[0]
    phi4 = np.asarray(interaction.phi4, dtype=np.complex128)
    if phi4.shape != (dim, dim, dim, dim):
        raise ValueError("interaction tensor shape must match the device dimension.")

    freqs0, vecs0 = _mode_data_from_device(device, frequency_floor=frequency_floor)
    covariance = _harmonic_covariance_from_modes(
        mode_frequencies=freqs0,
        mode_vectors=vecs0,
        temperature=float(temperature),
        frequency_floor=float(frequency_floor),
    )
    converged = False
    residual = float("inf")
    sigma = fourth_order_mean_field_self_energy(interaction=interaction, covariance=covariance)
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        dmat_eff = _symmetrize_matrix(dmat0 + sigma)
        evals, evecs = np.linalg.eigh(dmat_eff)
        freqs_eff = np.sqrt(np.clip(evals.real, a_min=frequency_floor * frequency_floor, a_max=None))
        covariance_new = _harmonic_covariance_from_modes(
            mode_frequencies=freqs_eff,
            mode_vectors=np.asarray(evecs, dtype=np.complex128),
            temperature=float(temperature),
            frequency_floor=float(frequency_floor),
        )
        covariance_next = mixing * covariance_new + (1.0 - mixing) * covariance
        residual = float(
            np.linalg.norm(covariance_next - covariance) / max(np.linalg.norm(covariance_next), 1e-30)
        )
        covariance = _symmetrize_matrix(covariance_next)
        sigma = fourth_order_mean_field_self_energy(interaction=interaction, covariance=covariance)
        if residual <= tol:
            converged = True
            break

    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "Fourth-order mean-field self-consistent solve did not converge within max_iter "
            f"(iterations={iterations}, residual={residual:.3e}, tol={tol:.3e})."
        )

    return FourthOrderMeanFieldModel(
        interaction=interaction,
        covariance=covariance,
        sigma_static=sigma,
        converged=converged,
        iterations=iterations,
        residual=residual,
    )


def fourth_order_lowest_order_self_energy(
    omega: float,
    interaction: FourthOrderInteraction,
    *,
    temperature: float | None = None,
    covariance: Array | None = None,
    mode_frequencies: Array | None = None,
    mode_vectors: Array | None = None,
    frequency_floor: float = 1e-8,
) -> Array:
    """Return the quartic lowest-order retarded self-energy.

    In this codebase, the quartic lowest-order closure is the one-shot Hartree /
    tadpole correction evaluated from the harmonic covariance, without the
    self-consistency loop used by ``fourth_order_mean_field_*``.
    """

    del omega
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    if covariance is None:
        if temperature is None:
            raise ValueError("temperature must be provided when covariance is not supplied.")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")
        if mode_frequencies is None or mode_vectors is None:
            raise ValueError("mode_frequencies and mode_vectors are required when covariance is not supplied.")
        covariance = _harmonic_covariance_from_modes(
            mode_frequencies=np.asarray(mode_frequencies, dtype=float),
            mode_vectors=np.asarray(mode_vectors, dtype=np.complex128),
            temperature=float(temperature),
            frequency_floor=float(frequency_floor),
        )

    return fourth_order_mean_field_self_energy(
        interaction=interaction,
        covariance=np.asarray(covariance, dtype=np.complex128),
    )


def fourth_order_mean_field_self_energy(
    interaction: FourthOrderInteraction,
    *,
    covariance: Array,
) -> Array:
    """Return the quartic mean-field self-energy from ``<u u^T>``.

    This is the Hartree-style static renormalization that makes quartic
    interactions the most natural first target for SCMF benchmarking.
    """

    phi4 = np.asarray(interaction.phi4, dtype=np.complex128)
    cov = np.asarray(covariance, dtype=np.complex128)
    dim = phi4.shape[0]
    if cov.shape != (dim, dim):
        raise ValueError("covariance must have shape (dim, dim).")

    sigma = 3.0 * np.einsum("ijkl,kl->ij", phi4, cov, optimize=True)
    return _symmetrize_matrix(sigma)


def fourth_order_scba_self_energy(
    omega: float,
    interaction: FourthOrderInteraction,
    green_function: Array,
    *,
    temperature: float | None = None,
    broadening: float = 1e-3,
    frequency_floor: float = 1e-8,
    mode_frequencies: Array,
    mode_vectors: Array,
    mode_couplings: Array | None = None,
) -> Array:
    """Return a quartic SCBA-like retarded self-energy.

    This is a practical self-consistent quasiparticle quartic closure in a
    fixed harmonic basis. It combines:
    - a dressed Hartree shift evaluated from the current modal covariance
    - a dynamic modal damping term built from quartic mode couplings and the
      current dressed frequencies/linewidths

    It is intentionally lighter than a full frequency-convolution quartic SCBA,
    but keeps the self-consistent feedback that we want for transport studies
    and future material-derived interaction tensors.
    """

    if temperature is None:
        raise ValueError("temperature must be provided for fourth-order SCBA self-energy.")
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if broadening <= 0.0:
        raise ValueError("broadening must be positive.")
    if frequency_floor <= 0.0:
        raise ValueError("frequency_floor must be positive.")

    freqs0 = np.asarray(mode_frequencies, dtype=float)
    vecs = np.asarray(mode_vectors, dtype=np.complex128)
    dim = freqs0.size
    g = np.asarray(green_function, dtype=np.complex128)
    if g.shape != (dim, dim):
        raise ValueError("green_function must have shape (dim, dim).")
    if vecs.shape != (dim, dim):
        raise ValueError("mode_vectors must have shape (n_modes, n_modes).")
    if np.asarray(interaction.phi4).shape != (dim, dim, dim, dim):
        raise ValueError("interaction tensor shape must match mode-space dimension.")

    couplings = _mode_couplings(np.asarray(interaction.phi4), vecs) if mode_couplings is None else np.asarray(
        mode_couplings, dtype=np.complex128
    )
    if couplings.shape != (dim, dim, dim, dim):
        raise ValueError("mode_couplings must have shape (dim, dim, dim, dim).")

    g_modes = _project_green_function_to_modes(green_function=g, mode_vectors=vecs)
    dressed_freqs, dressed_broadening = _effective_mode_data_from_green_function(
        omega=float(omega),
        green_function_modes=g_modes,
        harmonic_frequencies=freqs0,
        frequency_floor=float(frequency_floor),
        broadening_floor=float(broadening),
    )
    occ = _bose_occupation(dressed_freqs, float(temperature))

    covariance = _harmonic_covariance_from_modes(
        mode_frequencies=dressed_freqs,
        mode_vectors=vecs,
        temperature=float(temperature),
        frequency_floor=float(frequency_floor),
    )
    sigma_hartree = fourth_order_mean_field_self_energy(
        interaction=interaction,
        covariance=covariance,
    )

    sigma_modes_dyn = np.zeros(dim, dtype=np.complex128)
    for q in range(dim):
        wq = max(dressed_freqs[q], float(frequency_floor))
        gamma_q = 0.0
        for r in range(dim):
            wr = max(dressed_freqs[r], float(frequency_floor))
            nr = occ[r]
            for s in range(dim):
                ws = max(dressed_freqs[s], float(frequency_floor))
                ns = occ[s]
                for t in range(dim):
                    wt = max(dressed_freqs[t], float(frequency_floor))
                    nt = occ[t]
                    gqrst = couplings[q, r, s, t]
                    pref = (abs(gqrst) ** 2) / (16.0 * wq * wr * ws * wt)
                    gamma_rst = float(broadening) + dressed_broadening[r] + dressed_broadening[s] + dressed_broadening[t]
                    thermal = (2.0 * nr + 1.0) * (2.0 * ns + 1.0) * (2.0 * nt + 1.0)

                    kernel_13 = gamma_rst / (((float(omega) - wr - ws - wt) ** 2) + gamma_rst**2)
                    kernel_13 += gamma_rst / (((float(omega) + wr + ws + wt) ** 2) + gamma_rst**2)

                    kernel_22 = gamma_rst / (((float(omega) + wr - ws - wt) ** 2) + gamma_rst**2)
                    kernel_22 += gamma_rst / (((float(omega) - wr + ws - wt) ** 2) + gamma_rst**2)
                    kernel_22 += gamma_rst / (((float(omega) - wr - ws + wt) ** 2) + gamma_rst**2)

                    gamma_q += pref * thermal * (kernel_13 + 0.5 * kernel_22)

        sigma_modes_dyn[q] = -1j * max(gamma_q, 0.0)

    sigma_dynamic = vecs @ np.diag(sigma_modes_dyn) @ vecs.conj().T
    return _symmetrize_matrix(sigma_hartree + sigma_dynamic)
