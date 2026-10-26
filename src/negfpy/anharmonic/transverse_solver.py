"""Bounded dense coupled-q SCBA for synthetic, transverse-periodic models.

Contacts are supplied on the full signed grid. All quantities use solver
units; currents are per transverse cell. This bubble-only research solver
does not build material reservoirs/FC3, certify a continuum limit, or include
a static cubic mean-displacement correction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .retarded import retarded_from_keldysh
from .transverse_fft import (
    _active_coordinates,
    _workspace_bytes,
    cubic_born_self_energy_q_fft,
)
from .transverse_q import TransverseCubicVertex, cubic_born_self_energy_q
from .types import SolverScale

_MAX_BYTES = 64 * 1024**2
_MAX_REQUEST_BYTES = 4 * 1024**3
_COMPONENTS = ("retarded", "lesser", "greater")


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.complex128, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("Fields must contain only finite values.")
    result.setflags(write=False)
    return result


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _nonnegative(value: float, name: str) -> None:
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise TypeError(f"{name} must be a finite real number.")
    if np.iscomplexobj(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")


def _adjoint(values: np.ndarray) -> np.ndarray:
    return values.conj().swapaxes(-1, -2)


def _ratio(a: np.ndarray, b: np.ndarray, atol: float, rtol: float) -> float:
    scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
    error = float(np.max(np.abs(a - b)))
    allowance = atol + rtol * scale
    return error / allowance if allowance else (0.0 if error == 0 else float("inf"))


@dataclass(frozen=True, slots=True, eq=False)
class TransverseSpectrum:
    """Immutable full-grid ``r,<,>`` fields of shape ``(Nq,Nw,D,D)``.

    The three detached complex128 payloads share an explicit allocation
    budget, defaulting to 64 MiB and bounded above by 4 GiB. This admission
    limit does not estimate process RSS or change spectrum validation.
    """

    retarded: np.ndarray = field(repr=False)
    lesser: np.ndarray = field(repr=False)
    greater: np.ndarray = field(repr=False)
    max_bytes: int = field(default=_MAX_BYTES, kw_only=True, repr=False)

    def __post_init__(self) -> None:
        _positive_integer(self.max_bytes, "max_bytes")
        if self.max_bytes > _MAX_REQUEST_BYTES:
            raise ValueError("Explicit spectrum max_bytes exceeds the 4 GiB request ceiling.")
        shape = np.shape(self.retarded)
        if len(shape) != 4 or min(shape) < 1 or shape[-1] != shape[-2]:
            raise ValueError("Spectrum must have shape (Nq,Nw,D,D), with D >= 1.")
        if 3 * np.prod(shape, dtype=object) * 16 > self.max_bytes:
            raise ValueError("Spectrum payload exceeds max_bytes (64 MiB by default).")
        object.__setattr__(self, "max_bytes", int(self.max_bytes))
        for name in _COMPONENTS:
            value = getattr(self, name)
            if np.shape(value) != shape:
                raise ValueError("All three spectrum components must have the same shape.")
            object.__setattr__(self, name, _readonly(value))


def validate_transverse_spectrum(
    mesh: UniformBZMesh,
    grid: SignedFrequencyGrid,
    spectrum: TransverseSpectrum,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-11,
) -> Mapping[str, float]:
    """Check identities and correlation PSD without projection or clipping.

    Each returned value is an error divided by ``atol + rtol*scale`` and must
    be <= 1. Reflection couples q to -q; no same-q reality is imposed.
    Spectral passivity is not imposed on a nonequilibrium interaction field.
    """
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise ValueError("A transverse UniformBZMesh with third period one is required.")
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(spectrum, TransverseSpectrum):
        raise TypeError("spectrum must be a TransverseSpectrum.")
    _nonnegative(rtol, "rtol")
    _nonnegative(atol, "atol")
    if rtol + atol == 0:
        raise ValueError("At least one identity tolerance must be positive.")
    if spectrum.retarded.shape[:2] != (mesh.n_qpoints, grid.n_points):
        raise ValueError("Spectrum mesh/frequency dimensions differ.")
    negative = [mesh.negative_index(q) for q in range(mesh.n_qpoints)]
    r, lesser, greater = (getattr(spectrum, name) for name in _COMPONENTS)
    checks = {
        "lesser_antihermitian": _ratio(lesser, -_adjoint(lesser), atol, rtol),
        "greater_antihermitian": _ratio(greater, -_adjoint(greater), atol, rtol),
        "keldysh_identity": _ratio(greater - lesser, r - _adjoint(r), atol, rtol),
        "bosonic_reflection": _ratio(
            lesser[:, ::-1], greater[negative].swapaxes(-1, -2), atol, rtol,
        ),
        "retarded_reality": _ratio(r[:, ::-1], r[negative].conj(), atol, rtol),
    }
    for name, values in (("lesser_psd", lesser), ("greater_psd", greater)):
        correlation = 1j * values
        # Only the diagnostic eigensolve uses a Hermitian part. Stored fields
        # remain unchanged and anti-Hermiticity is independently checked above.
        eigenvalues = np.linalg.eigvalsh(0.5 * (correlation + _adjoint(correlation)))
        error = max(0.0, -float(np.min(eigenvalues)))
        allowance = atol + rtol * float(np.max(np.abs(eigenvalues)))
        checks[name] = error / allowance if allowance else 0.0
    failed = {name: value for name, value in checks.items() if value > 1.0}
    if failed:
        raise ValueError(f"Invalid coupled-q spectrum: {failed}")
    return MappingProxyType(checks)


def reconstruct_transverse_spectrum(
    grid: SignedFrequencyGrid,
    lesser: np.ndarray,
    greater: np.ndarray,
) -> TransverseSpectrum:
    """Apply the existing finite-grid causal PV reconstruction at every q."""
    left = np.asarray(lesser, dtype=np.complex128)
    right = np.asarray(greater, dtype=np.complex128)
    if left.ndim != 4 or left.shape != right.shape or left.shape[0] < 1:
        raise ValueError("Lesser/greater fields must share shape (Nq,Nw,D,D).")
    if 3 * left.size * 16 > _MAX_BYTES:
        raise ValueError("Spectrum payload exceeds the 64 MiB prototype limit.")
    retarded = np.stack([
        retarded_from_keldysh(left[q], right[q], grid)
        for q in range(left.shape[0])
    ])
    return TransverseSpectrum(retarded, left, right)


@dataclass(frozen=True, slots=True, eq=False)
class TransverseSCBAProblem:
    """A small periodic device with explicit contact fields and zero device eta."""

    grid: SignedFrequencyGrid
    vertex: TransverseCubicVertex
    dynamical: np.ndarray = field(repr=False)
    left: TransverseSpectrum
    right: TransverseSpectrum
    scale: SolverScale = field(default_factory=SolverScale)

    def __post_init__(self) -> None:
        if not isinstance(self.grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid.")
        if not isinstance(self.vertex, TransverseCubicVertex):
            raise TypeError("vertex must be a TransverseCubicVertex.")
        if not isinstance(self.scale, SolverScale):
            raise TypeError("scale must be a SolverScale.")
        mesh, d = self.vertex.mesh, self.vertex.ndof
        if np.shape(self.dynamical) != (mesh.n_qpoints, d, d):
            raise ValueError("dynamical must have shape (Nq,D,D) matching the vertex.")
        if mesh.n_qpoints * d * d * 16 > _MAX_BYTES:
            raise ValueError("Dynamical matrix exceeds the prototype payload limit.")
        dynamical = _readonly(self.dynamical)
        negative = [mesh.negative_index(q) for q in range(mesh.n_qpoints)]
        if (_ratio(dynamical, _adjoint(dynamical), 1e-12, 1e-10) > 1
                or _ratio(dynamical[negative], dynamical.conj(), 1e-12, 1e-10) > 1):
            raise ValueError("dynamical must be Hermitian and obey D(-q)=D(q)*.")
        if np.min(np.linalg.eigvalsh(dynamical)) < -1e-12:
            raise ValueError("The harmonic toy device must be stable.")
        for contact in (self.left, self.right):
            validate_transverse_spectrum(mesh, self.grid, contact)
            if contact.retarded.shape[-1] != d:
                raise ValueError("Contact dimension differs from the vertex.")
        object.__setattr__(self, "dynamical", dynamical)


@dataclass(frozen=True, slots=True)
class TransverseSCBAConfig:
    """Bounded linear mixing; convergence tests use the unmixed Born map."""

    max_iterations: int = 80
    mixing: float = 0.5
    sigma_rtol: float = 1e-7
    sigma_atol: float = 1e-10
    current_rtol: float = 1e-5
    current_atol: float = 1e-10
    identity_rtol: float = 1e-8
    identity_atol: float = 1e-11
    max_pair_evaluations: int = 100_000
    born_backend: str = "direct"

    def __post_init__(self) -> None:
        if not isinstance(self.born_backend, str):
            raise TypeError("born_backend must be 'direct' or 'fft'.")
        if self.born_backend not in ("direct", "fft"):
            raise ValueError("born_backend must be 'direct' or 'fft'.")
        _positive_integer(self.max_iterations, "max_iterations")
        _positive_integer(self.max_pair_evaluations, "max_pair_evaluations")
        _nonnegative(self.mixing, "mixing")
        if not 0 < self.mixing <= 1:
            raise ValueError("mixing must lie in (0,1].")
        for prefix in ("sigma", "current", "identity"):
            for suffix in ("rtol", "atol"):
                name = f"{prefix}_{suffix}"
                _nonnegative(getattr(self, name), name)
            if getattr(self, f"{prefix}_rtol") + getattr(self, f"{prefix}_atol") == 0:
                raise ValueError(f"At least one {prefix} tolerance must be positive.")


@dataclass(frozen=True, slots=True)
class TransverseCurrents:
    """Heat currents per transverse cell, positive into the device."""

    left: float
    right: float
    collision_power: float

    @property
    def contact_balance(self) -> float:
        return self.left + self.right

    @property
    def full_balance(self) -> float:
        return self.contact_balance + self.collision_power


@dataclass(frozen=True, slots=True)
class TransverseIteration:
    iteration: int
    sigma_residual: float
    current_residual: float
    identity_residual: float


@dataclass(frozen=True, slots=True, eq=False)
class TransverseSCBAResult:
    converged: bool
    reason: str
    green: TransverseSpectrum
    sigma_pp: TransverseSpectrum
    currents: TransverseCurrents
    history: tuple[TransverseIteration, ...]
    born_backend: str = "direct"


def transverse_heat_current(
    grid: SignedFrequencyGrid,
    contact: TransverseSpectrum,
    green: TransverseSpectrum,
    scale: SolverScale | None = None,
) -> float:
    """Positive-frequency rectangle current, averaged over the complete q mesh."""
    if scale is None:
        scale = SolverScale()
    if contact.retarded.shape != green.retarded.shape:
        raise ValueError("Contact and Green shapes must match.")
    if green.retarded.shape[1] != grid.n_points:
        raise ValueError("Spectrum/grid frequency dimensions differ.")
    exchange = contact.greater @ green.lesser - contact.lesser @ green.greater
    trace = np.trace(exchange, axis1=-2, axis2=-1).real
    positive = grid.positive_indices
    value = grid.integration_weight * np.sum(
        trace[:, positive] * grid.values[positive],
    ) / trace.shape[0]
    return float(scale.to_physical_current(float(value)))


def _dyson(problem: TransverseSCBAProblem, sigma: TransverseSpectrum) -> TransverseSpectrum:
    d = problem.vertex.ndof
    operator = (
        problem.grid.values[None, :, None, None] ** 2 * np.eye(d)
        - problem.dynamical[:, None]
        - problem.left.retarded - problem.right.retarded - sigma.retarded
    )
    try:
        retarded = np.linalg.inv(operator)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Singular coupled-q Dyson solve; check unbroadened/bound modes.") from exc
    advanced = _adjoint(retarded)
    lesser = retarded @ (problem.left.lesser + problem.right.lesser + sigma.lesser) @ advanced
    greater = retarded @ (problem.left.greater + problem.right.greater + sigma.greater) @ advanced
    return TransverseSpectrum(retarded, lesser, greater)


def solve_transverse_scba(
    problem: TransverseSCBAProblem,
    config: TransverseSCBAConfig | None = None,
    *,
    initial_sigma: TransverseSpectrum | None = None,
) -> TransverseSCBAResult:
    """Solve a bounded toy fixed point, retaining the last *evaluated* state.

    Success requires the raw map-minus-input residual (r,<,>) and q-averaged
    contact/collision conservation screens to pass. Identities/PSD are checked
    on contacts, every evaluated Green function, input, and raw output.
    A budget stop returns ``converged=False`` with mutually consistent fields.
    No residual projection, clipping, hidden retry, or post-return map occurs.
    """
    if not isinstance(problem, TransverseSCBAProblem):
        raise TypeError("problem must be a TransverseSCBAProblem.")
    if config is None:
        config = TransverseSCBAConfig()
    if not isinstance(config, TransverseSCBAConfig):
        raise TypeError("config must be a TransverseSCBAConfig.")
    mesh, grid = problem.vertex.mesh, problem.grid
    # Guard a complete map before doing any Dyson or PV work.
    if config.born_backend == "direct":
        pairs = mesh.n_qpoints**2 * (3 * grid.n_positive**2 + 3 * grid.n_positive + 1)
        born_map = cubic_born_self_energy_q
    else:
        # Reuse the kernel's exact-support planner; a full-D estimate would
        # incorrectly reject valid sparse-support FFT problems.
        active = len(_active_coordinates(problem.vertex.coefficients))
        n_fft = 1 << (2 * grid.n_points - 2).bit_length()
        pairs = mesh.n_qpoints**2 * n_fft if active else 0
        workspace = _workspace_bytes(
            mesh.n_qpoints, grid.n_points, n_fft, problem.vertex.ndof, active,
        )
        if workspace > _MAX_BYTES:
            raise ValueError("Planned coupled-q FFT workspace exceeds the 64 MiB prototype limit.")
        born_map = cubic_born_self_energy_q_fft
    if pairs > config.max_pair_evaluations:
        raise ValueError("Coupled-q map exceeds max_pair_evaluations.")

    def validate(spectrum: TransverseSpectrum) -> float:
        return max(validate_transverse_spectrum(
            mesh, grid, spectrum, rtol=config.identity_rtol, atol=config.identity_atol,
        ).values())

    validate(problem.left)
    validate(problem.right)
    shape = problem.left.retarded.shape
    if initial_sigma is None:
        zeros = np.zeros(shape, dtype=np.complex128)
        sigma = TransverseSpectrum(zeros, zeros, zeros)
    else:
        validate(initial_sigma)
        if initial_sigma.retarded.shape != shape:
            raise ValueError("Initial self-energy shape differs from the problem.")
        sigma = reconstruct_transverse_spectrum(grid, initial_sigma.lesser, initial_sigma.greater)
        if _ratio(sigma.retarded, initial_sigma.retarded,
                  config.identity_atol, config.identity_rtol) > 1:
            raise ValueError("Initial retarded self-energy differs from causal reconstruction.")
    history = []
    for iteration in range(1, config.max_iterations + 1):
        identity = validate(sigma)
        green = _dyson(problem, sigma)
        identity = max(identity, validate(green))
        lesser, greater = born_map(
            grid, problem.vertex, green.lesser, green.greater,
            max_pair_evaluations=config.max_pair_evaluations,
        )
        raw = reconstruct_transverse_spectrum(grid, lesser, greater)
        identity = max(identity, validate(raw))
        residual = max(_ratio(
            getattr(raw, name), getattr(sigma, name), config.sigma_atol, config.sigma_rtol,
        ) for name in _COMPONENTS)
        currents = TransverseCurrents(*(
            transverse_heat_current(grid, contact, green, problem.scale)
            for contact in (problem.left, problem.right, sigma)
        ))
        allowance = config.current_atol + config.current_rtol * max(
            abs(currents.left), abs(currents.right),
        )
        current_error = max(
            abs(currents.contact_balance), abs(currents.collision_power),
            abs(currents.full_balance),
        )
        current_residual = (
            current_error / allowance if allowance
            else (0.0 if current_error == 0 else float("inf"))
        )
        if not np.isfinite([residual, current_residual, identity]).all():
            raise ValueError("Nonfinite iteration diagnostics.")
        history.append(TransverseIteration(iteration, residual, current_residual, identity))
        converged = residual <= 1 and current_residual <= 1
        if converged or iteration == config.max_iterations:
            return TransverseSCBAResult(
                converged, "fixed_point" if converged else "max_iterations",
                green, sigma, currents, tuple(history), config.born_backend,
            )
        alpha = config.mixing
        sigma = reconstruct_transverse_spectrum(
            grid, (1 - alpha) * sigma.lesser + alpha * raw.lesser,
            (1 - alpha) * sigma.greater + alpha * raw.greater,
        )
    raise AssertionError("Positive iteration budget required.")


__all__ = [
    "TransverseCurrents",
    "TransverseIteration",
    "TransverseSCBAConfig",
    "TransverseSCBAProblem",
    "TransverseSCBAResult",
    "TransverseSpectrum",
    "reconstruct_transverse_spectrum",
    "solve_transverse_scba",
    "transverse_heat_current",
    "validate_transverse_spectrum",
]
