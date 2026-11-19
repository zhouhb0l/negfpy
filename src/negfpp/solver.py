"""Bounded fresh-zero SCBA using the established native-channel solvers.

The default trajectory is the existing two-map linear warm-up followed by
depth-one Anderson acceleration with strict selection and PSD backtracking.
Every attempted full map, including rejected trials, consumes the same budget.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from time import monotonic
from typing import TYPE_CHECKING, Callable

from negfpy.anharmonic.complete_energy import (
    ChannelAccelerationConfig,
    ChannelAccelerationResult,
    ChannelProposalFeasibility,
    ChannelSCBAConfig,
    ChannelSCBARecord,
    ChannelSCBAResult,
    ChannelSelectionPolicy,
    plan_channel_scba,
    run_channel_accel,
    run_channel_scba,
)

if TYPE_CHECKING:
    from .reference import ReferenceFeedbackResult


@dataclass(frozen=True, slots=True)
class SolveOutcome:
    """Last solver result and scalar accounting across the whole trajectory.

    ``records`` contains scalar convergence records only. Green functions and
    self-energy arrays from earlier evaluations are not retained here.
    If acceleration stops before its first completed evaluation, its result
    may have no state; warm-up records and emitted scalar callbacks survive.
    """

    result: ChannelSCBAResult | ChannelAccelerationResult | ReferenceFeedbackResult
    attempted: int
    completed: int
    selected: int
    rejected: int
    records: tuple[ChannelSCBARecord, ...]
    diagnostics: dict | None = None


def _solve(
    grid,
    vertex,
    dynamical,
    left,
    right,
    *,
    scale,
    g: float,
    config: ChannelSCBAConfig,
    method: str = "anderson1",
    reference_config: dict | None = None,
    deadline_monotonic: float | None = None,
    on_start: Callable | None = None,
    on_evaluation: Callable | None = None,
    sigma_initial=None,
) -> SolveOutcome:
    """Solve with fresh history within one map and time-admission budget.

    ``anderson1`` preserves the established two-map warm-up and continuation
    policy. ``linear`` uses the existing fixed-mixing solver throughout.
    ``reference-seeded`` uses the existing exact reference/increment controller
    with same-run selected anchors and bounded analytic feedback. Its first
    map evaluates the supplied self-energy or the default exact-zero state;
    reference and acceleration history always start fresh.
    A supplied absolute monotonic deadline can include input construction.
    Time admission does not interrupt an already running full map.

    ``on_start(index, sigma, proposal)`` and
    ``on_evaluation(state, decision)`` receive the existing solver objects;
    warm-up/fixed-mixing callbacks receive ``None`` for proposal/decision.
    Callback failures propagate without retry.
    """

    if not isinstance(config, ChannelSCBAConfig):
        raise TypeError("config must be ChannelSCBAConfig.")
    if method not in {"anderson1", "linear", "reference-seeded"}:
        raise ValueError("method must be 'anderson1', 'linear', or 'reference-seeded'.")
    if reference_config is not None and not isinstance(reference_config, dict):
        raise TypeError("reference_config must be a dict when supplied.")
    if reference_config and method != "reference-seeded":
        raise ValueError("reference_config requires method='reference-seeded'.")
    if method == "reference-seeded":
        if config.max_evaluations > 18:
            raise ValueError("reference-seeded permits at most 18 nonlinear map attempts.")
        allowed = {"warmup_evaluations", "tangent_interval", "max_tangent_directions",
                   "initial_tangent_estimate_seconds"}
        if reference_config is not None and not set(reference_config) <= allowed:
            raise ValueError("Unknown reference_config fields.")
    if any(callback is not None and not callable(callback)
           for callback in (on_start, on_evaluation)):
        raise TypeError("Callbacks must be callable.")
    started = monotonic()
    first_index = 1 if sigma_initial is None else 2
    deadline = started + config.max_seconds
    if deadline_monotonic is not None:
        if isinstance(deadline_monotonic, bool) or not isfinite(deadline_monotonic):
            raise ValueError("deadline_monotonic must be finite.")
        deadline = min(deadline, deadline_monotonic)
    attempted = completed = selected = rejected = 0
    records = []

    def start(index, sigma, proposal=None):
        nonlocal attempted
        if attempted >= config.max_evaluations:
            raise RuntimeError("Total SCBA map-attempt budget exhausted.")
        attempted += 1
        if on_start is not None:
            on_start(index, sigma, proposal)

    def evaluated(state, decision=None):
        nonlocal completed, selected, rejected
        accepted = decision is None or decision["accepted_as_anchor"]
        completed += 1
        selected += int(bool(accepted))
        rejected += int(not accepted)
        records.append(state.record)
        if on_evaluation is not None:
            on_evaluation(state, decision)

    def remaining(count):
        seconds = deadline - monotonic()
        if seconds <= config.time_reserve_seconds:
            return None
        return replace(config, max_evaluations=count, max_seconds=seconds)

    def outcome(result, diagnostics=None):
        if not 0 <= completed <= attempted <= config.max_evaluations:
            raise RuntimeError("SCBA map accounting is inconsistent.")
        if selected + rejected != completed:
            raise RuntimeError("SCBA trial-selection accounting is inconsistent.")
        return SolveOutcome(result, attempted, completed, selected, rejected, tuple(records), diagnostics)

    if method == "reference-seeded":
        from .reference import run_reference_feedback
        reference = remaining(config.max_evaluations)
        if reference is None:
            plan = plan_channel_scba(
                vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof, vertex.ndof,
                len(left.device_indices), len(right.device_indices), config=config,
                has_initial_sigma=sigma_initial is not None,
            )
            return outcome(
                ChannelSCBAResult(None, (), False, "max_seconds", None, plan),
                {"reference_kind": "same_run_selected_anchor", "tangent_count": 0,
                 "tangent_records": (), "map_records": (), "decisions": (), "proposal_screens": ()},
            )
        initial_options = {} if sigma_initial is None else {"sigma_initial": sigma_initial}
        result = run_reference_feedback(
            grid, vertex, dynamical, left, right,
            scale=scale, g=g, config=reference,
            acceleration=ChannelAccelerationConfig(),
            selection=ChannelSelectionPolicy(),
            feasibility=ChannelProposalFeasibility(mode="psd-backtracking"),
            methodconfig=reference_config, on_start=start, on_evaluation=evaluated,
            **initial_options,
        )
        return outcome(result, {
            "reference_kind": "same_run_selected_anchor",
            "tangent_count": result.tangent_count,
            "tangent_records": result.tangent_records,
            "map_records": result.map_records,
            "decisions": result.decisions,
            "proposal_screens": result.proposal_screens,
        })

    first_count = config.max_evaluations if method == "linear" else min(2, config.max_evaluations)
    first_config = remaining(first_count)
    if first_config is None:
        plan = plan_channel_scba(
            vertex.mesh.n_qpoints, grid.n_points, vertex.full_ndof, vertex.ndof,
            len(left.device_indices), len(right.device_indices), config=config,
            has_initial_sigma=sigma_initial is not None,
        )
        return outcome(ChannelSCBAResult(None, (), False, "max_seconds", None, plan))

    result = run_channel_scba(
        grid, vertex, dynamical, left, right,
        sigma_initial=sigma_initial, first_index=first_index, scale=scale, g=g,
        config=first_config, on_start=start, on_evaluation=evaluated,
    )
    if (
        method == "linear" or result.converged or result.failure_reason is not None
        or result.state is None or len(result.history) != 2
        or attempted >= config.max_evaluations
    ):
        return outcome(result)

    next_config = remaining(config.max_evaluations - attempted)
    if next_config is None:
        return outcome(replace(result, converged=False, stop_reason="max_seconds"))
    seed_input = result.state.sigma_input
    seed_mapped = result.state.evaluation.map_sigma
    seed_index = result.state.index
    # The original controller releases the warm-up Green fields before the
    # next solve; only its evaluated input and unevaluated image are seeds.
    del result
    result = run_channel_accel(
        grid, vertex, dynamical, left, right,
        seed_input=seed_input, seed_mapped=seed_mapped, seed_index=seed_index,
        scale=scale, g=g, config=next_config,
        acceleration=ChannelAccelerationConfig(),
        selection=ChannelSelectionPolicy(),
        feasibility=ChannelProposalFeasibility(mode="psd-backtracking"),
        on_start=start, on_evaluation=evaluated,
    )
    return outcome(result)


def solve_from_zero(*args, **kwargs):
    """Preserve the original exact-zero initialization and fresh history."""
    if "sigma_initial" in kwargs:
        raise TypeError("Use solve_from_self_energy for an external initial field")
    return _solve(*args, sigma_initial=None, **kwargs)


def solve_from_self_energy(*args, sigma_initial, **kwargs):
    """Freshly evaluate a supplied R/C/D guess; never reuse saved history."""
    if sigma_initial is None:
        raise ValueError("A complete R/C/D initial self-energy is required")
    return _solve(*args, sigma_initial=sigma_initial, **kwargs)


__all__ = ["SolveOutcome", "solve_from_zero", "solve_from_self_energy"]
