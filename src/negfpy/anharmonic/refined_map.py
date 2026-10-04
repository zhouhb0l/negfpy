"""Opt-in research map with whole midpoint-cell frequency replacements.

This is deliberately separate from ``solve_scba`` and checkpoint machinery.
The grid Dyson solution supplies the exact coarse terms being removed; the
interacting pointwise evaluator supplies only the replacement integrals. Each
changed Keldysh grid is followed by a full-grid retarded reconstruction. A
completed evaluation is a map evaluation, never an accepted SCBA solution.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Sequence

import numpy as np

from .cubic import _contract_cubic_pair, cubic_lesser_greater
from .frequency_refinement import (
    GaussLegendreWindowPlan, WindowBubbleResult, _integer,
    cubic_window_lesser_greater,
)
from .diagnostics import relative_residual
from .retarded import retarded_from_keldysh
from .solver import _current_report, _solve_green
from .sparse_cubic import (
    SparseCubicVertex, _compiled_sparse_schedule, _sparse_pair_contraction,
)
from .types import CurrentReport, KeldyshSpectrum


@dataclass(frozen=True, slots=True)
class ResearchRefinementSpec:
    """One output index and its complete, nonoverlapping midpoint cells.

    Supply one plan per contiguous cell group, in increasing order. Plans may
    contain extra panel boundaries but must retain every cell boundary. Indices
    are actual nonnegative grid indices, not signed frequency offsets.
    """

    output_index: int
    cell_indices: tuple[int, ...]
    plans: tuple[GaussLegendreWindowPlan, ...]

    def __post_init__(self) -> None:
        output = _integer(self.output_index, "output_index")
        cells = tuple(_integer(value, "cell index") for value in self.cell_indices)
        plans = tuple(self.plans)
        if not cells or tuple(sorted(set(cells))) != cells:
            raise ValueError("cell_indices must be nonempty, unique and increasing.")
        if not plans or any(not isinstance(plan, GaussLegendreWindowPlan) for plan in plans):
            raise ValueError("plans must contain GaussLegendreWindowPlan instances.")
        if any(plan.order not in (8, 16) for plan in plans):
            raise ValueError("replacement quadrature requires GL8 or GL16.")
        object.__setattr__(self, "output_index", output)
        object.__setattr__(self, "cell_indices", cells)
        object.__setattr__(self, "plans", plans)


@dataclass(frozen=True, slots=True)
class RefinedWindowEvaluation:
    output_index: int
    cell_indices: tuple[int, ...]
    coarse_lesser: np.ndarray
    coarse_greater: np.ndarray
    quadrature: WindowBubbleResult
    applied: bool


@dataclass(frozen=True, slots=True)
class RefinedMapEvaluation:
    """Research evidence; ``completed`` must hold before using the raw map.

    When a budget expires, a partial window is retained as diagnostic evidence
    but is never applied. Earlier complete windows may already be replaced;
    ``residual`` is None and these raw arrays must not be advanced or accepted.
    Currents describe the input interaction and its Dyson solution. At thermal
    equilibrium their cancellation alone does not test biased conservation.
    """

    green: KeldyshSpectrum
    sigma: KeldyshSpectrum
    raw_lesser: np.ndarray
    raw_greater: np.ndarray
    raw_retarded: np.ndarray
    residual: float | None
    coarse_residual: float
    currents: CurrentReport
    windows: tuple[RefinedWindowEvaluation, ...]
    completed: bool
    elapsed_seconds: float


def _cell_groups(cells: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    groups = [[cells[0]]]
    for index in cells[1:]:
        if index == groups[-1][-1] + 1:
            groups[-1].append(index)
        else:
            groups.append([index])
    return tuple(tuple(group) for group in groups)


def _validate_specs(grid, refinements):
    refinements = tuple(refinements)
    if not refinements:
        raise ValueError("At least one explicit refinement is required.")
    outputs = set()
    validated = []
    # Decimal physical grids can differ by one ulp at an adjacent cell edge.
    edge_tolerance = 8 * np.finfo(float).eps * max(1.0, grid.cutoff)
    for spec in refinements:
        if not isinstance(spec, ResearchRefinementSpec):
            raise TypeError("refinements must contain ResearchRefinementSpec instances.")
        if not 0 <= spec.output_index < grid.n_points:
            raise ValueError("output_index must be an exact valid grid index.")
        if spec.output_index in outputs:
            raise ValueError("Duplicate output_index; combine its cells into one spec.")
        outputs.add(spec.output_index)
        for cell in spec.cell_indices:
            partner = spec.output_index - cell + grid.zero_index
            if not 0 < cell < grid.n_points - 1 or not 0 < partner < grid.n_points - 1:
                raise ValueError("Refined cells and reflected partner cells must be wholly inside cutoff.")
        groups = _cell_groups(spec.cell_indices)
        if len(groups) != len(spec.plans):
            raise ValueError("Supply exactly one quadrature plan per contiguous cell group.")
        omega = float(grid.values[spec.output_index])
        for group, plan in zip(groups, spec.plans):
            boundaries = [float(grid.values[group[0]] - grid.step / 2)]
            boundaries.extend(float(grid.values[index] + grid.step / 2) for index in group)
            if (abs(plan.breakpoints[0] - boundaries[0]) > edge_tolerance
                    or abs(plan.breakpoints[-1] - boundaries[-1]) > edge_tolerance):
                raise ValueError("Plan endpoints must match the complete midpoint-cell group.")
            for boundary in boundaries[1:-1]:
                if not any(abs(x - boundary) <= edge_tolerance for x in plan.breakpoints):
                    raise ValueError("Plans must retain every declared midpoint-cell boundary.")
            for singular in (0.0, omega):
                if (plan.breakpoints[0] < singular < plan.breakpoints[-1]
                        and singular not in plan.breakpoints):
                    raise ValueError("Split panels at zero and output frequency for pointwise evaluation.")
                if np.any(plan.nodes == singular):
                    raise ValueError("Quadrature nodes cannot sample exact zero in either Green argument.")
            validated.append((spec.output_index, group, plan))
    return tuple(validated)


def discrete_cell_contribution(vertex, green, grid, output_index, cell_indices):
    """Exact coarse discrete terms for these cells, using original grid fields.

    The contraction has no conjugation or Keldysh cross term. The factor is
    exactly the production rectangle weight times i/2. Sparse schedules are
    compiled once and only compact matrices are retained during accumulation.
    """
    sparse = isinstance(vertex, SparseCubicVertex)
    schedule = _compiled_sparse_schedule(vertex) if sparse else None
    active = schedule.active if sparse else np.arange(vertex.ndof)
    compact = np.ix_(active, active)
    values = np.zeros((2, len(active), len(active)), dtype=np.complex128)
    correction = np.zeros_like(values)
    for index in cell_indices:
        partner = output_index - index + grid.zero_index
        if not 0 <= index < grid.n_points or not 0 <= partner < grid.n_points:
            raise ValueError("Discrete cell pair is outside the signed grid.")
        for component, field in enumerate((green.lesser, green.greater)):
            first, second = field[index][compact], field[partner][compact]
            contraction = (_sparse_pair_contraction(vertex, schedule, first, second)
                           if sparse else _contract_cubic_pair(vertex.values, first, second))
            adjusted = contraction - correction[component]
            updated = values[component] + adjusted
            correction[component] = (updated - values[component]) - adjusted
            values[component] = updated
    full = np.zeros((2, vertex.ndof, vertex.ndof), dtype=np.complex128)
    for component in (0, 1):
        full[component][compact] = (0.5j * grid.integration_weight) * values[component]
    return full[0], full[1]


def _residual(raw_lesser, raw_greater, raw_retarded, sigma, config):
    floor = (config.sigma_atol / config.sigma_rtol if config.sigma_rtol > 0
             else np.finfo(float).tiny)
    return max(relative_residual(raw_lesser, sigma.lesser, floor=floor),
               relative_residual(raw_greater, sigma.greater, floor=floor),
               relative_residual(raw_retarded, sigma.retarded, floor=floor))


def refined_fixed_point_evaluation(
    problem, left, right, sigma_lesser, sigma_greater, config,
    refinements: Sequence[ResearchRefinementSpec], *,
    interpolation_policy: str = "equilibrium-broadening",
    max_seconds: float | None = None,
    max_seconds_per_window: float | None = None,
    max_nodes_per_window: int = 100_000,
    progress: Callable[[dict], None] | None = None,
) -> RefinedMapEvaluation:
    """Evaluate one bounded, opt-in full-grid Dyson/bubble research map.

    Budgets are cooperative: no new quadrature batch is started after expiry;
    a running solve/contraction cannot be interrupted by this helper. The caller
    must also impose its process cap. No SCBA mixing or checkpoint is performed.
    ``max_seconds`` includes coarse preparation; per-window budgets start with
    each declared window. Completion says only that the declared rule was run.
    """
    started = time.monotonic()
    groups = _validate_specs(problem.grid, refinements)
    for value, name in ((max_seconds, "max_seconds"),
                        (max_seconds_per_window, "max_seconds_per_window")):
        if value is not None and (not math.isfinite(float(value)) or float(value) <= 0):
            raise ValueError(f"{name} must be finite and positive.")
    max_nodes_per_window = _integer(max_nodes_per_window, "max_nodes_per_window")
    if max_nodes_per_window <= 0 or any(plan.n_points > max_nodes_per_window for _, _, plan in groups):
        raise ValueError("Replacement plan exceeds max_nodes_per_window.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable.")
    from .interacting_pointwise import PointwiseInteracting

    green, sigma = _solve_green(problem, left, right, sigma_lesser, sigma_greater, config)
    # Construct first so unsupported/nonthermal interpolation fails before the
    # costly bubble or before any research replacement is attempted.
    pointwise = PointwiseInteracting(problem, config, sigma, policy=interpolation_policy)
    raw_lesser, raw_greater = cubic_lesser_greater(
        problem.vertex, green.lesser, green.greater, problem.grid,
        backend=config.backend, max_workers=config.max_workers,
    )
    reconstruct = dict(strategy=config.retarded_strategy, hilbert_backend=config.hilbert_backend)
    raw_retarded = retarded_from_keldysh(raw_lesser, raw_greater, problem.grid, **reconstruct)
    coarse_residual = _residual(raw_lesser, raw_greater, raw_retarded, sigma, config)
    currents = _current_report(problem, left, right, sigma, green)
    windows = []
    completed = True
    for window_index, (output, cells, plan) in enumerate(groups):
        remaining = None if max_seconds is None else float(max_seconds) - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            completed = False
            break
        coarse = discrete_cell_contribution(problem.vertex, green, problem.grid, output, cells)
        # Coarse extraction can itself be expensive; its duration must not
        # leave the following quadrature with a stale global allowance.
        remaining = None if max_seconds is None else float(max_seconds) - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            completed = False
            break
        budget = max_seconds_per_window
        if remaining is not None:
            budget = remaining if budget is None else min(float(budget), remaining)

        def report(record):
            if progress is not None:
                progress({"window_index": window_index, "output_index": output,
                          "output_frequency": float(problem.grid.values[output]),
                          "cell_indices": list(cells), **record})

        quadrature = cubic_window_lesser_greater(
            problem.vertex, pointwise, float(problem.grid.values[output]), plan,
            cutoff=problem.grid.cutoff, max_workers=config.max_workers or 1,
            max_seconds=budget, max_nodes=max_nodes_per_window, progress=report,
        )
        windows.append(RefinedWindowEvaluation(output, cells, *coarse, quadrature, quadrature.completed))
        if not quadrature.completed:
            completed = False
            break
        raw_lesser[output] += quadrature.lesser - coarse[0]
        raw_greater[output] += quadrature.greater - coarse[1]

    # Changes to the dispersive shift extend beyond the modified output slices.
    # Reconstruct from the complete resulting Keldysh grids, never patch just
    # the same retarded slices or interpolate a correction factor.
    raw_retarded = retarded_from_keldysh(raw_lesser, raw_greater, problem.grid, **reconstruct)
    residual = _residual(raw_lesser, raw_greater, raw_retarded, sigma, config) if completed else None
    return RefinedMapEvaluation(green, sigma, raw_lesser, raw_greater, raw_retarded,
                               residual, coarse_residual, currents, tuple(windows),
                               completed, time.monotonic() - started)
