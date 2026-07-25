"""Small, deterministic refinement studies for dense cubic-SCBA toys.

This module deliberately provides a comparison harness, not a certification
shortcut.  A study runs the unmodified reference problem first and then
changes one or more explicitly named numerical controls.  It reports physical
contact-current changes, the interaction self-energy cutoff-tail change, and
the convergence-history change.  Whether an individual solve satisfies the
solver's release gates remains available as data and is never inferred from a
small current delta alone.

The default limits keep this helper within the dense-v0.1 validation regime:
at most eight candidate calculations and at most 1025 signed-frequency points
per calculation.  Callers must opt in explicitly to larger studies.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from ..diagnostics import keldysh_tail_ratio
from ..exceptions import SCBAConvergenceError
from ..grid import SignedFrequencyGrid
from ..solver import solve_scba
from ..types import (
    AnharmonicResult,
    CubicSCBAProblem,
    SCBAConfig,
)


def _positive_finite(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and strictly positive.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one.")
    return result


@dataclass(frozen=True, slots=True)
class RefinementSpec:
    """One named numerical change relative to a reference calculation.

    ``step_factor`` and ``cutoff_factor`` multiply the reference grid spacing
    and cutoff.  The new positive-grid count is rounded upward, so a requested
    cutoff is never silently shortened.  ``eta_factor`` multiplies the lead
    regulator.  ``mixing`` is an absolute replacement; ``None`` retains the
    reference value.
    """

    name: str
    step_factor: float = 1.0
    cutoff_factor: float = 1.0
    eta_factor: float = 1.0
    mixing: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string.")
        object.__setattr__(self, "name", self.name.strip())
        for field_name in ("step_factor", "cutoff_factor", "eta_factor"):
            object.__setattr__(
                self,
                field_name,
                _positive_finite(getattr(self, field_name), name=field_name),
            )
        if self.mixing is not None:
            mixing = _positive_finite(self.mixing, name="mixing")
            if mixing > 1.0:
                raise ValueError("mixing cannot exceed one.")
            object.__setattr__(self, "mixing", mixing)


@dataclass(frozen=True, slots=True)
class RefinementObservation:
    """Immutable numerical summary of one SCBA calculation."""

    name: str
    grid_step: float
    grid_cutoff: float
    n_frequency: int
    eta_lead: float
    mixing: float
    left_current: float
    right_current: float
    sigma_tail_ratio: float
    converged: bool
    release_certified: bool
    iterations: int
    result: AnharmonicResult


@dataclass(frozen=True, slots=True)
class RefinementDelta:
    """Change of one candidate relative to the reference calculation."""

    name: str
    left_current_change: float
    right_current_change: float
    max_current_absolute_change: float
    max_current_relative_change: float
    sigma_tail_change: float
    sigma_tail_absolute_change: float
    iteration_change: int
    convergence_changed: bool
    candidate_converged: bool
    candidate_release_certified: bool


@dataclass(frozen=True, slots=True)
class RefinementStudy:
    """Reference run, candidate runs, and their like-named deltas."""

    reference: RefinementObservation
    candidates: tuple[RefinementObservation, ...]
    deltas: tuple[RefinementDelta, ...]

    def __post_init__(self) -> None:
        candidate_names = tuple(item.name for item in self.candidates)
        delta_names = tuple(item.name for item in self.deltas)
        if candidate_names != delta_names:
            raise ValueError("candidate and delta names must match in order.")
        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError("candidate names must be unique.")

    def observation(self, name: str) -> RefinementObservation:
        """Return a candidate observation by its exact name."""

        for item in self.candidates:
            if item.name == name:
                return item
        raise KeyError(name)

    def delta(self, name: str) -> RefinementDelta:
        """Return a candidate delta by its exact name."""

        for item in self.deltas:
            if item.name == name:
                return item
        raise KeyError(name)


def _refined_inputs(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    spec: RefinementSpec,
) -> tuple[CubicSCBAProblem, SCBAConfig]:
    step = problem.grid.step * spec.step_factor
    requested_cutoff = problem.grid.cutoff * spec.cutoff_factor
    ratio = requested_cutoff / step
    # Moving one representable value toward -infinity avoids an accidental
    # extra point when an exact integer ratio has only roundoff-level excess.
    n_positive = int(np.ceil(np.nextafter(ratio, -np.inf)))
    n_positive = max(1, n_positive)
    grid = SignedFrequencyGrid(step=step, n_positive=n_positive)

    mixing = config.mixing if spec.mixing is None else spec.mixing
    candidate_problem = replace(problem, grid=grid)
    candidate_config = replace(
        config,
        eta_lead=config.eta_lead * spec.eta_factor,
        mixing=mixing,
    )
    return candidate_problem, candidate_config


def _run_retaining_partial(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> AnharmonicResult:
    try:
        return solve_scba(problem, config)
    except SCBAConvergenceError as error:
        if error.partial_result is None:
            raise
        return error.partial_result


def _observe(
    name: str,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> RefinementObservation:
    result = _run_retaining_partial(problem, config)
    return RefinementObservation(
        name=name,
        grid_step=problem.grid.step,
        grid_cutoff=problem.grid.cutoff,
        n_frequency=problem.grid.n_points,
        eta_lead=config.eta_lead,
        mixing=config.mixing,
        left_current=result.currents.left,
        right_current=result.currents.right,
        sigma_tail_ratio=keldysh_tail_ratio(result.sigma_pp),
        converged=result.converged,
        release_certified=result.release_certified,
        iterations=len(result.history),
        result=result,
    )


def _compare(
    reference: RefinementObservation,
    candidate: RefinementObservation,
    *,
    current_floor: float,
) -> RefinementDelta:
    left_change = candidate.left_current - reference.left_current
    right_change = candidate.right_current - reference.right_current
    maximum_absolute = max(abs(left_change), abs(right_change))
    current_scale = max(
        abs(reference.left_current),
        abs(reference.right_current),
        abs(candidate.left_current),
        abs(candidate.right_current),
        current_floor,
    )
    tail_change = candidate.sigma_tail_ratio - reference.sigma_tail_ratio
    return RefinementDelta(
        name=candidate.name,
        left_current_change=left_change,
        right_current_change=right_change,
        max_current_absolute_change=maximum_absolute,
        max_current_relative_change=maximum_absolute / current_scale,
        sigma_tail_change=tail_change,
        sigma_tail_absolute_change=abs(tail_change),
        iteration_change=candidate.iterations - reference.iterations,
        convergence_changed=(candidate.converged != reference.converged),
        candidate_converged=candidate.converged,
        candidate_release_certified=candidate.release_certified,
    )


def run_refinement_study(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    refinements: Sequence[RefinementSpec],
    *,
    max_candidates: int = 8,
    max_frequency_points: int = 1025,
) -> RefinementStudy:
    """Run a bounded reference/candidate SCBA refinement study.

    Nonconverged calculations are retained through the solver's complete
    partial result and are marked explicitly in the observations and deltas.
    This makes a failed refinement visible without discarding the other
    candidates.  A small delta is descriptive only: callers must still inspect
    ``converged``, ``release_certified``, and the full result diagnostics.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")
    if isinstance(refinements, (str, bytes)) or not isinstance(
        refinements, Sequence
    ):
        raise TypeError("refinements must be a sequence of RefinementSpec objects.")

    max_candidates = _positive_int(max_candidates, name="max_candidates")
    max_frequency_points = _positive_int(
        max_frequency_points,
        name="max_frequency_points",
    )
    specifications = tuple(refinements)
    if len(specifications) > max_candidates:
        raise ValueError(
            f"study has {len(specifications)} candidates, exceeding "
            f"max_candidates={max_candidates}."
        )
    if any(not isinstance(item, RefinementSpec) for item in specifications):
        raise TypeError("refinements must contain only RefinementSpec objects.")
    names = tuple(item.name for item in specifications)
    if len(set(names)) != len(names):
        raise ValueError("refinement names must be unique.")
    if "reference" in names:
        raise ValueError("'reference' is reserved for the unmodified calculation.")

    prepared: list[tuple[RefinementSpec, CubicSCBAProblem, SCBAConfig]] = []
    if problem.grid.n_points > max_frequency_points:
        raise ValueError(
            f"reference grid has {problem.grid.n_points} points, exceeding "
            f"max_frequency_points={max_frequency_points}."
        )
    for spec in specifications:
        candidate_problem, candidate_config = _refined_inputs(
            problem,
            config,
            spec,
        )
        if candidate_problem.grid.n_points > max_frequency_points:
            raise ValueError(
                f"refinement {spec.name!r} has "
                f"{candidate_problem.grid.n_points} points, exceeding "
                f"max_frequency_points={max_frequency_points}."
            )
        prepared.append((spec, candidate_problem, candidate_config))

    reference = _observe("reference", problem, config)
    candidates = tuple(
        _observe(spec.name, candidate_problem, candidate_config)
        for spec, candidate_problem, candidate_config in prepared
    )
    current_floor = config.current_atol * problem.scale.current_scale
    deltas = tuple(
        _compare(reference, candidate, current_floor=current_floor)
        for candidate in candidates
    )
    return RefinementStudy(
        reference=reference,
        candidates=candidates,
        deltas=deltas,
    )


__all__ = [
    "RefinementDelta",
    "RefinementObservation",
    "RefinementSpec",
    "RefinementStudy",
    "run_refinement_study",
]
