"""Checkpoint-friendly SCBA over the unchanged bounded material Born evaluator.

Only the last evaluated state and the candidate being evaluated are retained.
All returned currents and Green functions refer to ``state.sigma_input``;
``state.evaluation.map_sigma`` is an unevaluated Born image of that state.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import hypot
from time import monotonic

import numpy as np

from .grid import SignedFrequencyGrid
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_material_map import (
    MaterialMapPlan,
    PackedBornMapEvaluation,
    PackedMapSpectrum,
    _real,
    _spectrum,
    evaluate_packed_born_map,
    plan_material_born_map,
)
from .transverse_packed import PackedTransverseCubicVertex, _finite, _integer

_COMPONENTS = ("retarded", "lesser", "greater")


@dataclass(frozen=True, slots=True)
class MaterialSCBAConfig:
    """Finite trajectory and fixed convex mixing; no hidden retries or damping."""

    max_evaluations: int = 12
    mixing: float = 0.1
    sigma_rtol: float = 1e-4
    sigma_atol: float = 1e-10
    current_rtol: float = 1e-3
    current_atol_w: float = 1e-14
    max_seconds: float = 79200.0
    time_reserve_seconds: float = 300.0
    next_map_safety_factor: float = 1.25
    initial_map_estimate_seconds: float = 1200.0
    max_bytes: int = 16 * 1024**3
    workers: int = 8
    external_tile: int = 16
    bin_tile: int = 1
    max_contraction_work: int = 50_000_000_000_000
    max_dyson_evaluations: int = 2404
    identity_rtol: float = 1e-8
    identity_atol: float = 1e-11

    def __post_init__(self):
        for name in ("max_evaluations", "max_bytes", "workers", "external_tile", "bin_tile",
                     "max_contraction_work", "max_dyson_evaluations"):
            _integer(getattr(self, name), name)
        for name in ("mixing", "sigma_rtol", "sigma_atol", "current_rtol", "current_atol_w",
                     "max_seconds", "time_reserve_seconds", "next_map_safety_factor",
                     "initial_map_estimate_seconds", "identity_rtol", "identity_atol"):
            _real(getattr(self, name), name)
        if not 0 < self.mixing <= 1:
            raise ValueError("mixing must lie in (0, 1].")
        for first, second in ((self.sigma_rtol, self.sigma_atol),
                              (self.current_rtol, self.current_atol_w),
                              (self.identity_rtol, self.identity_atol)):
            if first + second == 0:
                raise ValueError("Each tolerance pair requires a positive allowance.")
        if self.max_seconds <= self.time_reserve_seconds:
            raise ValueError("max_seconds must exceed time_reserve_seconds.")
        if self.next_map_safety_factor < 1 or self.initial_map_estimate_seconds <= 0:
            raise ValueError("Map safety factor must be >= 1 and initial estimate > 0.")


@dataclass(frozen=True, slots=True)
class MaterialSCBAPlan:
    map_plan: MaterialMapPlan
    input_sigma_bytes: int
    previous_evaluated_state_bytes: int
    residual_mixing_workspace_bytes: int
    peak_owned_bytes: int
    nominal_caller_bytes: int
    maximum_contraction_work: int
    maximum_dyson_evaluations: int


def plan_material_scba(
    n_q, n_frequency, full_ndof, active_ndof, left_ndof, right_ndof, *,
    config=None, has_initial_sigma=True,
) -> MaterialSCBAPlan:
    """Numeric-byte upper bound, excluding caller data and callback retention.

    Includes the previous evaluated input/Green/map while the next candidate
    is evaluated, so a failed evaluation cannot discard the last good state.
    Python metadata and FFT/BLAS/LAPACK internal scratch are separate.
    """
    config = MaterialSCBAConfig() if config is None else config
    if not isinstance(config, MaterialSCBAConfig):
        raise TypeError("config must be MaterialSCBAConfig.")
    if type(has_initial_sigma) is not bool:
        raise TypeError("has_initial_sigma must be bool.")
    plan = plan_material_born_map(
        n_q, n_frequency, full_ndof, active_ndof, left_ndof, right_ndof,
        has_input_sigma=False, workers=config.workers,
        external_tile=config.external_tile, bin_tile=config.bin_tile,
    )
    sigma = plan.mapped_sigma_bytes
    prior = 2 * sigma + plan.retained_green_bytes
    scratch = 64 * max(plan.active_ndof, 1)**2
    return MaterialSCBAPlan(
        plan, sigma, prior, scratch, plan.peak_owned_bytes + sigma + prior + scratch,
        plan.nominal_caller_bytes + (sigma if has_initial_sigma else 0),
        config.max_evaluations * plan.contraction_work,
        config.max_evaluations * plan.dyson_evaluations,
    )


@dataclass(frozen=True, slots=True)
class MaterialSigmaResidual:
    error: float
    scale: float
    allowance: float
    ratio: float
    error_frobenius: float
    scale_frobenius: float
    relative_frobenius: float


@dataclass(frozen=True, slots=True)
class MaterialCurrentResidual:
    error_w: float
    scale_w: float
    allowance_w: float
    ratio: float


@dataclass(frozen=True, slots=True)
class MaterialSCBARecord:
    index: int
    map_seconds: float
    elapsed_seconds: float
    sigma_residuals: dict[str, MaterialSigmaResidual]
    current_residuals: dict[str, MaterialCurrentResidual]
    sigma_passed: bool
    current_passed: bool
    converged: bool


@dataclass(frozen=True, slots=True, eq=False)
class MaterialSCBAState:
    index: int
    sigma_input: PackedMapSpectrum | None
    evaluation: PackedBornMapEvaluation
    record: MaterialSCBARecord


@dataclass(frozen=True, slots=True, eq=False)
class MaterialSCBAResult:
    state: MaterialSCBAState | None
    history: tuple[MaterialSCBARecord, ...]
    converged: bool
    stop_reason: str
    failure_reason: str | None
    plan: MaterialSCBAPlan


def _ratio(error, allowance):
    return error / allowance if allowance else (0.0 if error == 0 else float("inf"))


def _shape(spectrum):
    if not isinstance(spectrum, PackedMapSpectrum):
        raise TypeError("Supply PackedMapSpectrum fields.")
    shape = np.shape(spectrum.retarded)
    if (len(shape) != 4 or min(shape[:2]) < 1 or shape[-1] != shape[-2]):
        raise ValueError("Spectrum must have shape (Nq, Nw, A, A).")
    _spectrum(spectrum, shape, "spectrum")
    return shape


def mix_packed_spectra(sigma_input, mapped_sigma, mixing) -> PackedMapSpectrum:
    """Return owned read-only ``(1-alpha)*input + alpha*map``; None means zero.

    This constructs only an unevaluated candidate, never an accepted state.
    At most one matrix-block temporary is needed beyond the output spectrum.
    """
    alpha = _real(mixing, "mixing")
    if not 0 < alpha <= 1:
        raise ValueError("mixing must lie in (0, 1].")
    shape = _shape(mapped_sigma)
    if sigma_input is not None:
        _spectrum(sigma_input, shape, "sigma_input")
    arrays = []
    with np.errstate(over="raise", invalid="raise"):
        for name in _COMPONENTS:
            mapped = getattr(mapped_sigma, name)
            output = np.empty(shape, dtype=np.complex128)
            for q in range(shape[0]):
                for w in range(shape[1]):
                    np.multiply(mapped[q, w], alpha, out=output[q, w])
                    if sigma_input is not None:
                        output[q, w] += (1-alpha) * getattr(sigma_input, name)[q, w]
            _finite(output, "mixed candidate")
            output.setflags(write=False)
            arrays.append(output)
    return PackedMapSpectrum(*arrays)


def _sigma_residual(mapped, supplied, config):
    error = scale = error_fro = mapped_fro = supplied_fro = 0.0
    for q in range(mapped.shape[0]):
        for w in range(mapped.shape[1]):
            first = mapped[q, w]
            second = supplied[q, w] if supplied is not None else 0.0
            if first.size == 0:
                continue
            difference = first - second
            error = max(error, float(np.max(np.abs(difference))))
            scale = max(scale, float(np.max(np.abs(first))), float(np.max(np.abs(second))))
            error_fro = hypot(error_fro, float(np.linalg.norm(difference)))
            mapped_fro = hypot(mapped_fro, float(np.linalg.norm(first)))
            supplied_fro = hypot(supplied_fro, float(np.linalg.norm(second)))
    allowance = config.sigma_atol + config.sigma_rtol * scale
    scale_fro = max(mapped_fro, supplied_fro)
    values = (error, scale, allowance, error_fro, scale_fro)
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite unmixed Sigma residual diagnostic.")
    return MaterialSigmaResidual(error, scale, allowance, _ratio(error, allowance),
                                 error_fro, scale_fro, _ratio(error_fro, scale_fro))


def _record(index, sigma, evaluation, map_seconds, elapsed_seconds, config):
    residuals = {
        name: _sigma_residual(getattr(evaluation.map_sigma, name),
                              getattr(sigma, name) if sigma is not None else None, config)
        for name in _COMPONENTS
    }
    currents = evaluation.input_currents
    scale = max(abs(currents.left), abs(currents.right))
    allowance = config.current_atol_w + config.current_rtol * scale
    current_values = (currents.left, currents.right, currents.collision_power,
                      currents.contact_balance, currents.full_balance, allowance)
    if not np.isfinite(current_values).all():
        raise ValueError("Nonfinite evaluated-state current diagnostic.")
    current_residuals = {
        name: MaterialCurrentResidual(abs(getattr(currents, name)), scale, allowance,
                                      _ratio(abs(getattr(currents, name)), allowance))
        for name in ("contact_balance", "collision_power", "full_balance")
    }
    sigma_passed = all(value.ratio <= 1 for value in residuals.values())
    current_passed = all(value.ratio <= 1 for value in current_residuals.values())
    return MaterialSCBARecord(index, map_seconds, elapsed_seconds, residuals, current_residuals,
                              sigma_passed, current_passed, sigma_passed and current_passed)


def run_material_scba(
    grid, vertex, dynamical, left, right, *, sigma_initial=None, scale=None, g=0.03,
    config=None, on_evaluation: Callable[[MaterialSCBAState], None] | None = None,
    on_start: Callable[[int, PackedMapSpectrum | None], None] | None = None,
) -> MaterialSCBAResult:
    """Evaluate a bounded SCBA trajectory with callbacks on completed states.

    Initial arrays are copied, then evaluated before any convergence decision.
    Unmixed component defects and three evaluated-current balances must all
    pass. Failed later maps return the last evaluated state with failure_reason;
    a first-map failure returns state=None. Callback errors propagate, so a
    failed checkpoint can never be silently reported as successful execution.

    Before every map, remaining time must exceed reserve plus the safety-scaled
    maximum of the initial estimate and completed map durations. This is a
    scheduling heuristic, not an interruption guarantee for an individual map.
    on_start(index, sigma_input) is called immediately before each evaluated
    map and can record candidate hashes. Callbacks must save synchronously
    without retaining historical array states; their errors propagate.
    """
    started = monotonic()
    config = MaterialSCBAConfig() if config is None else config
    if not isinstance(config, MaterialSCBAConfig):
        raise TypeError("config must be MaterialSCBAConfig.")
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(vertex, PackedTransverseCubicVertex):
        raise TypeError("Supply a signed grid and packed vertex.")
    if not all(isinstance(contact, CompactTransverseContact) for contact in (left, right)):
        raise TypeError("Supply compact contacts.")
    if any(callback is not None and not callable(callback) for callback in (on_evaluation, on_start)):
        raise TypeError("Callbacks must be callable.")
    plan = plan_material_scba(
        vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof, vertex.ndof,
        len(left.device_indices), len(right.device_indices), config=config,
        has_initial_sigma=sigma_initial is not None,
    )
    if plan.peak_owned_bytes > config.max_bytes:
        raise ValueError("Material SCBA allocation plan exceeds max_bytes.")
    if plan.map_plan.contraction_work > config.max_contraction_work:
        raise ValueError("Material SCBA map exceeds max_contraction_work.")
    if plan.map_plan.dyson_evaluations > config.max_dyson_evaluations:
        raise ValueError("Material SCBA map exceeds max_dyson_evaluations.")
    sigma = None if sigma_initial is None else mix_packed_spectra(None, sigma_initial, 1.0)
    state = None
    history = []
    estimate = config.initial_map_estimate_seconds
    for index in range(1, config.max_evaluations + 1):
        remaining = config.max_seconds - (monotonic() - started)
        if remaining <= config.time_reserve_seconds + config.next_map_safety_factor * estimate:
            return MaterialSCBAResult(state, tuple(history), False, "max_seconds", None, plan)
        try:
            if state is not None:
                sigma = mix_packed_spectra(state.sigma_input, state.evaluation.map_sigma, config.mixing)
        except (ValueError, FloatingPointError) as exc:
            return MaterialSCBAResult(state, tuple(history), False, "evaluation_error",
                                      f"{type(exc).__name__}: {exc}", plan)
        if on_start is not None:
            on_start(index, sigma)
        try:
            map_started = monotonic()
            evaluation = evaluate_packed_born_map(
                grid, vertex, dynamical, left, right, sigma_input=sigma, scale=scale, g=g,
                workers=config.workers, external_tile=config.external_tile, bin_tile=config.bin_tile,
                max_bytes=plan.map_plan.peak_owned_bytes,
                max_contraction_work=config.max_contraction_work,
                max_dyson_evaluations=config.max_dyson_evaluations,
                identity_rtol=config.identity_rtol, identity_atol=config.identity_atol,
            )
            map_seconds = monotonic() - map_started
            record = _record(index, sigma, evaluation, map_seconds, monotonic()-started, config)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            return MaterialSCBAResult(state, tuple(history), False, "evaluation_error",
                                      f"{type(exc).__name__}: {exc}", plan)
        state = MaterialSCBAState(index, sigma, evaluation, record)
        history.append(record)
        estimate = max(estimate, map_seconds)
        if on_evaluation is not None:
            on_evaluation(state)
        if record.converged:
            return MaterialSCBAResult(state, tuple(history), True, "fixed_point", None, plan)
    return MaterialSCBAResult(state, tuple(history), False, "max_evaluations", None, plan)


__all__ = [
    "MaterialCurrentResidual",
    "MaterialSCBAConfig",
    "MaterialSCBAPlan",
    "MaterialSCBARecord",
    "MaterialSCBAResult",
    "MaterialSCBAState",
    "MaterialSigmaResidual",
    "mix_packed_spectra",
    "plan_material_scba",
    "run_material_scba",
]
