"""Opt-in full-matrix cubic Born integrals over a frequency window.

This module evaluates a declared composite Gauss--Legendre rule, not an adaptive
error estimate.  It does not replace entries in a uniform-grid self-energy or
participate in SCBA.  Its lesser and greater results each use the production
convention ``0.5j/(2*pi) * integral Psi[a,c,d] Psi[b,e,f] G[c,e] G[d,f]``;
there is no conjugation and no lesser/greater cross term.

The supplied callback must return both finite Green matrices at any requested
signed frequency.  The two Green arguments are zero outside the declared
symmetric cutoff.  Panels are clipped to their common nonzero support before
quadrature, rather than sampling across an artificial cutoff discontinuity.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import math
import operator
import time
from types import MappingProxyType
from typing import Callable, Mapping

import numpy as np
from threadpoolctl import threadpool_limits

from .cubic import _contract_cubic_pair
from .sparse_cubic import (
    SparseCubicVertex,
    _compiled_sparse_schedule,
    _sparse_pair_contraction,
)
from .types import DenseCubicVertex


GreenEvaluator = Callable[[float], tuple[np.ndarray, np.ndarray]]


def _immutable_array(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    # Bytes-backed arrays cannot regain write access via setflags(write=True).
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc


@dataclass(frozen=True, slots=True, eq=False)
class GaussLegendreWindowPlan:
    """Immutable positive-weight GL1, GL8 or GL16 panels in increasing order.

    ``breakpoints`` defines one contiguous window.  Every original interior
    breakpoint is retained when clipping to the Green-function support.
    ``weights`` integrate d(nu); the Born prefactor is applied by the evaluator.
    GL1 is the explicit midpoint rectangle, useful for exact whole-cell
    subtraction from a coarse discrete sum.  This plan supplies no error
    estimate or accuracy guarantee by itself.
    """

    breakpoints: tuple[float, ...]
    order: int = 16
    nodes: np.ndarray = field(init=False, repr=False)
    weights: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        breakpoints = tuple(float(value) for value in self.breakpoints)
        if len(breakpoints) < 2 or not all(math.isfinite(x) for x in breakpoints):
            raise ValueError("breakpoints must contain at least two finite values.")
        if any(b <= a for a, b in zip(breakpoints, breakpoints[1:])):
            raise ValueError("breakpoints must be strictly increasing.")
        order = _integer(self.order, "order")
        if order not in (1, 8, 16):
            raise ValueError("order must be 1, 8 or 16.")
        base_nodes, base_weights = np.polynomial.legendre.leggauss(order)
        nodes, weights = [], []
        for left, right in zip(breakpoints, breakpoints[1:]):
            midpoint = left / 2.0 + right / 2.0
            half_width = right / 2.0 - left / 2.0
            nodes.append(midpoint + half_width * base_nodes)
            weights.append(half_width * base_weights)
        node_array = np.concatenate(nodes)
        weight_array = np.concatenate(weights)
        if (not np.all(np.isfinite(node_array))
                or not np.all(np.isfinite(weight_array))
                or not np.all(weight_array > 0)
                or not np.all(np.diff(node_array) > 0)):
            raise ValueError("panels are not representable by distinct finite nodes and positive weights.")
        object.__setattr__(self, "breakpoints", breakpoints)
        object.__setattr__(self, "order", order)
        object.__setattr__(self, "nodes", _immutable_array(node_array, np.float64))
        object.__setattr__(self, "weights", _immutable_array(weight_array, np.float64))

    @property
    def n_points(self) -> int:
        return len(self.nodes)

    def clipped(self, lower: float, upper: float) -> GaussLegendreWindowPlan | None:
        """Intersect panels with finite bounds, returning None for empty support."""
        lower, upper = float(lower), float(upper)
        if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
            raise ValueError("clipping bounds must be finite and ordered.")
        left = max(lower, self.breakpoints[0])
        right = min(upper, self.breakpoints[-1])
        if right <= left:
            return None
        if left == self.breakpoints[0] and right == self.breakpoints[-1]:
            return self
        return GaussLegendreWindowPlan(
            (left, *(x for x in self.breakpoints[1:-1] if left < x < right), right),
            self.order,
        )


@dataclass(frozen=True, slots=True, eq=False)
class WindowContribution:
    """Immutable lesser/greater contribution to one tracked range."""

    lesser: np.ndarray
    greater: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "lesser", _immutable_array(self.lesser, np.complex128))
        object.__setattr__(self, "greater", _immutable_array(self.greater, np.complex128))


@dataclass(frozen=True, slots=True, eq=False)
class WindowBubbleResult:
    """Window contribution only; no claim of global or adaptive convergence."""

    lesser: np.ndarray
    greater: np.ndarray
    output_frequency: float
    cutoff: float
    effective_breakpoints: tuple[float, ...]
    order: int
    pair_evaluations: int
    green_evaluations: int
    elapsed_seconds: float
    completed: bool
    planned_pair_evaluations: int
    ranges: Mapping[str, WindowContribution]

    def __post_init__(self) -> None:
        object.__setattr__(self, "lesser", _immutable_array(self.lesser, np.complex128))
        object.__setattr__(self, "greater", _immutable_array(self.greater, np.complex128))
        object.__setattr__(self, "ranges", MappingProxyType(dict(self.ranges)))


def cubic_window_lesser_greater(
    vertex: DenseCubicVertex | SparseCubicVertex,
    evaluator: GreenEvaluator,
    output_frequency: float,
    plan: GaussLegendreWindowPlan,
    *,
    cutoff: float,
    max_workers: int = 1,
    tracked_ranges: Mapping[str, tuple[float, float]] | None = None,
    max_seconds: float | None = None,
    max_nodes: int = 100_000,
    progress: Callable[[dict], None] | None = None,
) -> WindowBubbleResult:
    """Integrate both full Born matrices in a bounded, deterministic stream.

    ``evaluator(nu)`` returns ``(G_lesser(nu), G_greater(nu))``, each with shape
    ``(vertex.ndof, vertex.ndof)``.  The callback must be thread-safe when
    ``max_workers > 1``.  Its arrays are copied before a second callback is
    issued, allowing a serial callback to reuse internal scratch buffers.

    At most eight frequency pairs are in flight; no full Green-frequency stack
    is built or retained.  BLAS threads are limited to one while workers are
    active.  Pair results are accumulated in increasing node order with
    compensated summation, identically for serial and threaded execution.
    Sparse contraction structure is compiled exactly once per call.

    The returned quadrature is restricted to the declared window and the
    intersection ``[-cutoff, cutoff] & [omega-cutoff, omega+cutoff]``.  It must
    not be used as a multiplicative correction for other entries/frequencies.

    ``tracked_ranges`` accumulates additional intervals without reevaluating
    nodes.  Their endpoints must be exact original plan breakpoints.  A wall
    budget is checked before every bounded batch; it cannot interrupt a running
    callback.  On budget exhaustion ``completed`` is False and every matrix is
    only the processed-node sum, without extrapolation.  ``progress`` receives
    compact counts about every 64 nodes and at the end, never Green arrays.
    """
    started = time.monotonic()
    if not isinstance(vertex, (DenseCubicVertex, SparseCubicVertex)):
        raise TypeError("vertex must be DenseCubicVertex or SparseCubicVertex.")
    if not isinstance(plan, GaussLegendreWindowPlan):
        raise TypeError("plan must be GaussLegendreWindowPlan.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    output_frequency, cutoff = float(output_frequency), float(cutoff)
    if not math.isfinite(output_frequency):
        raise ValueError("output_frequency must be finite.")
    if not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be positive and finite.")
    max_workers = _integer(max_workers, "max_workers")
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")

    max_nodes = _integer(max_nodes, "max_nodes")
    if max_nodes <= 0 or plan.n_points > max_nodes:
        raise ValueError("plan exceeds the positive max_nodes budget.")
    if max_seconds is not None:
        max_seconds = float(max_seconds)
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be positive and finite.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable.")
    declared_ranges = {}
    for name, bounds in (tracked_ranges or {}).items():
        if not isinstance(name, str) or not name:
            raise ValueError("tracked range names must be nonempty strings.")
        if len(bounds) != 2:
            raise ValueError("tracked ranges must contain two endpoints.")
        left, right = (float(x) for x in bounds)
        if left >= right or left not in plan.breakpoints or right not in plan.breakpoints:
            raise ValueError("tracked range endpoints must be ordered exact plan breakpoints.")
        declared_ranges[name] = (left, right)

    # Use comparisons before subtraction to avoid overflow for disjoint domains.
    if abs(output_frequency) / 2.0 >= cutoff:
        effective_plan = None
    else:
        effective_plan = plan.clipped(
            max(-cutoff, output_frequency - cutoff),
            min(cutoff, output_frequency + cutoff),
        )
    sparse = isinstance(vertex, SparseCubicVertex)
    schedule = _compiled_sparse_schedule(vertex) if sparse else None
    active = schedule.active if sparse else np.arange(vertex.ndof)
    compact_shape = (len(active), len(active))
    accumulator = np.zeros((1 + len(declared_ranges), 2, *compact_shape), dtype=np.complex128)
    compensation = np.zeros_like(accumulator)

    def fields(frequency: float) -> tuple[np.ndarray, np.ndarray]:
        # Every evaluation lies inside both support bounds by construction.
        values = evaluator(float(frequency))
        if not isinstance(values, (tuple, list)) or len(values) != 2:
            raise ValueError("evaluator must return (lesser, greater) matrices.")
        checked = []
        for name, value in zip(("lesser", "greater"), values):
            array = np.asarray(value, dtype=np.complex128)
            if array.shape != (vertex.ndof, vertex.ndof):
                raise ValueError(f"evaluator {name} must have shape {(vertex.ndof, vertex.ndof)}.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"evaluator {name} must be finite.")
            checked.append(np.array(array[np.ix_(active, active)], copy=True))
        return checked[0], checked[1]

    def pair(index: int) -> tuple[np.ndarray, int]:
        first_frequency = float(effective_plan.nodes[index])
        second_frequency = output_frequency - first_frequency
        first = fields(first_frequency)
        if first_frequency == second_frequency:
            second, evaluations = first, 1
        else:
            second, evaluations = fields(second_frequency), 2
        if sparse:
            values = [_sparse_pair_contraction(vertex, schedule, g1, g2)
                      for g1, g2 in zip(first, second)]
        else:
            values = [_contract_cubic_pair(vertex.values, g1, g2)
                      for g1, g2 in zip(first, second)]
        weighted = effective_plan.weights[index] * np.asarray(values)
        if not np.all(np.isfinite(weighted)):
            raise ValueError("weighted Born contraction must be finite.")
        return weighted, evaluations

    pair_evaluations = 0
    green_evaluations = 0

    def accumulate(index: int, result: tuple[np.ndarray, int]) -> None:
        nonlocal green_evaluations, pair_evaluations, accumulator, compensation
        weighted, count = result
        frequency = float(effective_plan.nodes[index])
        rows = [0] + [position for position, (left, right) in
                      enumerate(declared_ranges.values(), 1) if left < frequency < right]
        for row in rows:
            adjusted = weighted - compensation[row]
            updated = accumulator[row] + adjusted
            compensation[row] = (updated - accumulator[row]) - adjusted
            accumulator[row] = updated
        green_evaluations += count
        pair_evaluations += 1

    planned = effective_plan.n_points if effective_plan is not None else 0
    last_progress = 0

    def budget_available() -> bool:
        return max_seconds is None or time.monotonic() - started < max_seconds

    def report(*, final: bool = False) -> None:
        nonlocal last_progress
        if progress is not None and (final or pair_evaluations - last_progress >= 64):
            progress({"pair_evaluations": pair_evaluations,
                      "planned_pair_evaluations": planned,
                      "green_evaluations": green_evaluations,
                      "elapsed_seconds": time.monotonic() - started,
                      "completed": (pair_evaluations == planned or not len(active)) if final else False,
                      "final": final})
            last_progress = pair_evaluations

    if effective_plan is not None and len(active):
        with threadpool_limits(limits=1):
            if max_workers == 1:
                for index in range(effective_plan.n_points):
                    if not budget_available():
                        break
                    accumulate(index, pair(index))
                    report()
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as pool:
                    # Do not use executor.map over the complete plan: Python
                    # 3.10--3.12 would eagerly queue every sample and can retain
                    # a large number of completed output matrices.
                    for start in range(0, effective_plan.n_points, max_workers):
                        if not budget_available():
                            break
                        indices = range(start, min(start + max_workers,
                                                   effective_plan.n_points))
                        futures = [pool.submit(pair, index) for index in indices]
                        for index, future in zip(indices, futures):
                            accumulate(index, future.result())
                        report()
    complete = np.zeros((1 + len(declared_ranges), 2, vertex.ndof, vertex.ndof), dtype=np.complex128)
    complete[:, :, active[:, None], active] = (0.5j / (2.0 * np.pi)) * accumulator
    if not np.all(np.isfinite(complete)):
        raise ValueError("integrated Born matrices must be finite.")
    report(final=True)
    return WindowBubbleResult(
        lesser=complete[0, 0], greater=complete[0, 1],
        output_frequency=output_frequency, cutoff=cutoff,
        effective_breakpoints=effective_plan.breakpoints if effective_plan is not None else (),
        order=plan.order, pair_evaluations=pair_evaluations,
        green_evaluations=green_evaluations,
        elapsed_seconds=time.monotonic() - started,
        completed=(pair_evaluations == planned or not len(active)),
        planned_pair_evaluations=planned,
        ranges={name: WindowContribution(complete[index, 0], complete[index, 1])
                for index, name in enumerate(declared_ranges, 1)},
    )
