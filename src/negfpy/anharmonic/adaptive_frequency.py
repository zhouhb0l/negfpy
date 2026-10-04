"""Bounded research quadrature for both full cubic Keldysh matrices.

Each panel compares independent Gauss--Legendre 8 and 16 point rules. These
sampled differences are estimators, not guaranteed error bounds. Error norms
are summed over panels without cancellation, separately for lesser/greater and
maximum/Frobenius norms. A second partition remains a useful independent check.
No SCBA state, production default, Green interpolation or checkpoint is changed.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time
from typing import Callable

import numpy as np
from threadpoolctl import threadpool_limits

from .cubic import _contract_cubic_pair
from .frequency_refinement import _immutable_array, _integer
from .sparse_cubic import (
    SparseCubicVertex, _compiled_sparse_schedule, _sparse_pair_contraction,
)
from .types import DenseCubicVertex


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveMatrixLeaf:
    """One complete panel, with both rules in the compact active basis."""

    left: float
    right: float
    low: np.ndarray
    high: np.ndarray

    def __post_init__(self):
        for name in ("low", "high"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), np.complex128))

    @property
    def bounds(self):
        return (self.left, self.right)

    @property
    def error_abs(self):
        return np.abs(self.high - self.low)

    @property
    def max_errors(self):
        return np.max(self.error_abs, axis=(1, 2), initial=0.0)

    @property
    def frobenius_errors(self):
        return np.linalg.norm(self.high - self.low, axis=(1, 2))


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveMatrixResult:
    """Integral and independently reconstructible final partition evidence.

    ``complete_partition`` means every part of the requested domain has a
    complete GL8/16 panel. ``tolerance_met`` additionally requires all four
    sampled norm criteria. Neither flag establishes physical grid convergence.
    An incomplete initial partition returns only its complete panels, and must
    not be used as a whole-window integral. Interrupted refinement retains the
    complete parent until both proposed children finish.
    """

    high: np.ndarray
    low: np.ndarray
    leaves: tuple[AdaptiveMatrixLeaf, ...]
    effective_breakpoints: tuple[float, ...]
    summed_error_abs: np.ndarray
    max_errors: np.ndarray
    frobenius_errors: np.ndarray
    max_tolerances: np.ndarray
    frobenius_tolerances: np.ndarray
    pair_evaluations: int
    green_evaluations: int
    complete_partition: bool
    tolerance_met: bool
    stop_reason: str
    elapsed_seconds: float

    def __post_init__(self):
        for name in ("high", "low"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), np.complex128))
        for name in ("summed_error_abs", "max_errors", "frobenius_errors",
                     "max_tolerances", "frobenius_tolerances"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), np.float64))

    @property
    def max_scales(self):
        return np.max(np.abs(self.high), axis=(1, 2), initial=0.0)

    @property
    def frobenius_scales(self):
        return np.linalg.norm(self.high, axis=(1, 2))

    @property
    def max_error_ratios(self):
        return np.divide(self.max_errors, self.max_tolerances,
                         out=np.where(self.max_errors > 0, np.inf, 0.0),
                         where=self.max_tolerances > 0)

    @property
    def frobenius_error_ratios(self):
        return np.divide(self.frobenius_errors, self.frobenius_tolerances,
                         out=np.where(self.frobenius_errors > 0, np.inf, 0.0),
                         where=self.frobenius_tolerances > 0)


@dataclass(frozen=True, slots=True, eq=False)
class AdaptiveWindowResult(AdaptiveMatrixResult):
    """Full output matrices plus compact per-leaf cubic contraction evidence."""

    lesser: np.ndarray
    greater: np.ndarray
    active_indices: np.ndarray
    output_frequency: float
    cutoff: float

    def __post_init__(self):
        super(AdaptiveWindowResult, self).__post_init__()
        for name in ("lesser", "greater"):
            object.__setattr__(self, name, _immutable_array(getattr(self, name), np.complex128))
        object.__setattr__(self, "active_indices", _immutable_array(self.active_indices, np.int64))


def _breakpoints(values):
    result = tuple(float(value) for value in values)
    if len(result) < 2 or not all(math.isfinite(value) for value in result):
        raise ValueError("breakpoints must contain at least two finite values.")
    if any(right <= left for left, right in zip(result, result[1:])):
        raise ValueError("breakpoints must be strictly increasing.")
    return result


def _controls(rtol, atol, max_pair_evaluations, max_seconds, max_panels, max_workers, progress):
    rtol, atol = float(rtol), float(atol)
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0 or rtol + atol == 0:
        raise ValueError("rtol and atol must be finite and nonnegative, with at least one positive.")
    max_pair_evaluations = _integer(max_pair_evaluations, "max_pair_evaluations")
    max_panels = _integer(max_panels, "max_panels")
    max_workers = _integer(max_workers, "max_workers")
    if max_pair_evaluations <= 0 or max_panels <= 0:
        raise ValueError("Evaluation and panel budgets must be positive integers.")
    if not 1 <= max_workers <= 8:
        raise ValueError("max_workers must be between 1 and 8.")
    if max_seconds is not None:
        max_seconds = float(max_seconds)
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive.")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable.")
    return rtol, atol, max_pair_evaluations, max_seconds, max_panels, max_workers


def _sum_arrays(values, shape, dtype):
    total = np.zeros(shape, dtype=dtype)
    compensation = np.zeros_like(total)
    for value in values:
        corrected = value - compensation
        updated = total + corrected
        compensation = (updated - total) - corrected
        total = updated
    return total


def _metrics(leaves, dimension, rtol, atol):
    shape = (2, dimension, dimension)
    high = _sum_arrays((leaf.high for leaf in leaves), shape, np.complex128)
    low = _sum_arrays((leaf.low for leaf in leaves), shape, np.complex128)
    error_abs = _sum_arrays((leaf.error_abs for leaf in leaves), shape, np.float64)
    maximum = _sum_arrays((leaf.max_errors for leaf in leaves), (2,), np.float64)
    frobenius = _sum_arrays((leaf.frobenius_errors for leaf in leaves), (2,), np.float64)
    max_tolerance = atol + rtol*np.max(np.abs(high), axis=(1, 2), initial=0.0)
    frobenius_tolerance = atol + rtol*np.linalg.norm(high, axis=(1, 2))
    finite = all(np.all(np.isfinite(value)) for value in
                 (high, low, error_abs, maximum, frobenius, max_tolerance, frobenius_tolerance))
    if not finite:
        raise FloatingPointError("Adaptive integral or error accumulation is non-finite.")
    return high, low, error_abs, maximum, frobenius, max_tolerance, frobenius_tolerance


def _adaptive_core(evaluate, breakpoints, dimension, *, rtol, atol,
                   max_pair_evaluations, max_seconds, max_panels, max_workers,
                   progress, started):
    controls = _controls(rtol, atol, max_pair_evaluations, max_seconds, max_panels, max_workers, progress)
    rtol, atol, max_pair_evaluations, max_seconds, max_panels, max_workers = controls
    if len(breakpoints) - 1 > max_panels:
        raise ValueError("Initial partition exceeds max_panels.")
    shapes = (2, dimension, dimension)
    rules = [np.polynomial.legendre.leggauss(order) for order in (8, 16)]
    pairs = greens = 0
    leaves = []

    def budget_reason(required_pairs):
        if max_seconds is not None and time.monotonic() - started >= max_seconds:
            return "wall-budget"
        if pairs + required_pairs > max_pair_evaluations:
            return "evaluation-budget"
        return None

    def panel(left, right, pool):
        nonlocal pairs, greens
        reason = budget_reason(24)
        if reason is not None:
            return None, reason
        midpoint, half_width = left/2 + right/2, right/2 - left/2
        values = []
        for base_nodes, base_weights in rules:
            nodes, weights = midpoint + half_width*base_nodes, half_width*base_weights
            if (not np.all(np.isfinite(nodes)) or not np.all(np.diff(nodes) > 0)
                    or not np.all((nodes > left) & (nodes < right))
                    or not np.all(np.isfinite(weights)) or not np.all(weights > 0)):
                return None, "roundoff-limited"
            total = np.zeros(shapes, complex)
            compensation = np.zeros_like(total)
            for start in range(0, len(nodes), max_workers):
                count = min(max_workers, len(nodes)-start)
                reason = budget_reason(count)
                if reason is not None:
                    return None, reason
                indices = range(start, start+count)
                results = ([evaluate(float(nodes[index])) for index in indices] if pool is None
                           else [future.result() for future in
                                 [pool.submit(evaluate, float(nodes[index])) for index in indices]])
                for index, (sample, green_count) in zip(indices, results):
                    sample = np.asarray(sample, dtype=np.complex128)
                    if sample.shape != shapes or not np.all(np.isfinite(sample)):
                        raise ValueError(f"Integrand must return finite matrices of shape {shapes}.")
                    weighted = weights[index]*sample
                    if not np.all(np.isfinite(weighted)):
                        raise FloatingPointError("Weighted adaptive integrand is non-finite.")
                    adjusted = weighted - compensation
                    updated = total + adjusted
                    compensation = (updated-total)-adjusted
                    total = updated
                    pairs += 1
                    greens += green_count
            values.append(total)
        return AdaptiveMatrixLeaf(left, right, values[0], values[1]), None

    def report(phase, final=False):
        if progress is not None:
            metrics = _metrics(leaves, dimension, rtol, atol)
            progress(dict(phase=phase,final=final,panels=len(leaves),pair_evaluations=pairs,
                          green_evaluations=greens,max_errors=metrics[3].tolist(),
                          frobenius_errors=metrics[4].tolist(),max_tolerances=metrics[5].tolist(),
                          frobenius_tolerances=metrics[6].tolist(),elapsed_seconds=time.monotonic()-started))

    complete = False
    met = False
    reason = "no-support" if not breakpoints else None
    with threadpool_limits(limits=1):
        pool = ThreadPoolExecutor(max_workers=max_workers) if max_workers > 1 else None
        try:
            for left, right in zip(breakpoints, breakpoints[1:]):
                leaf, reason = panel(left, right, pool)
                if leaf is None:
                    break
                leaves.append(leaf)
                report("initial")
            else:
                complete = True
            while complete:
                metrics = _metrics(leaves, dimension, rtol, atol)
                met = bool(np.all(metrics[3] <= metrics[5]) and np.all(metrics[4] <= metrics[6]))
                if met:
                    reason = "tolerance-met" if breakpoints else "no-support"
                    break
                reason = budget_reason(48)
                if reason is not None:
                    break
                if len(leaves) >= max_panels:
                    reason = "panel-budget"
                    break
                def score(leaf):
                    return max(*(float(error/tolerance) if tolerance > 0 else
                                 (math.inf if error > 0 else 0.0)
                                 for error,tolerance in zip(leaf.max_errors, metrics[5])),
                               *(float(error/tolerance) if tolerance > 0 else
                                 (math.inf if error > 0 else 0.0)
                                 for error,tolerance in zip(leaf.frobenius_errors, metrics[6])))
                index = max(range(len(leaves)), key=lambda item: score(leaves[item]))
                parent = leaves[index]
                midpoint = parent.left/2 + parent.right/2
                if not parent.left < midpoint < parent.right:
                    reason = "roundoff-limited"
                    break
                first, reason = panel(parent.left, midpoint, pool)
                if first is None:
                    break
                second, reason = panel(midpoint, parent.right, pool)
                if second is None:
                    break
                # Atomic partition update: no partial child or lone sibling
                # can replace the parent's complete integral.
                leaves[index:index+1] = [first, second]
                report("refinement")
        finally:
            if pool is not None:
                pool.shutdown(wait=True)
    report("finished", final=True)
    metrics = _metrics(leaves, dimension, rtol, atol)
    return AdaptiveMatrixResult(metrics[0], metrics[1], tuple(leaves), tuple(breakpoints),
                                *metrics[2:], pairs, greens, complete, met, str(reason),
                                time.monotonic()-started)


def adaptive_matrix_integral(evaluator: Callable[[float], np.ndarray], breakpoints, *,
                             matrix_dimension: int, rtol=.001, atol=1e-12,
                             max_pair_evaluations=24000, max_seconds=900,
                             max_panels=2048, max_workers=1, progress=None):
    """Integrate a callback returning two full matrices, without cubic factors.

    The evaluator must be thread-safe for multiple workers. Arrays are copied
    at callback return so a serial evaluator can reuse scratch buffers. No
    matrix samples are retained beyond one worker batch.
    """
    started = time.monotonic()
    dimension = _integer(matrix_dimension, "matrix_dimension")
    if dimension <= 0 or not callable(evaluator):
        raise ValueError("matrix_dimension must be positive and evaluator callable.")
    points = _breakpoints(breakpoints)
    def evaluate(nu):
        return np.array(evaluator(nu), dtype=np.complex128, copy=True), 0
    return _adaptive_core(evaluate, points, dimension, rtol=rtol, atol=atol,
                          max_pair_evaluations=max_pair_evaluations, max_seconds=max_seconds,
                          max_panels=max_panels, max_workers=max_workers, progress=progress,
                          started=started)


def cubic_adaptive_lesser_greater(vertex, evaluator, output_frequency, breakpoints, *,
                                  cutoff, rtol=.001, atol=1e-12,
                                  max_pair_evaluations=24000, max_seconds=900,
                                  max_panels=2048, max_workers=1, progress=None):
    """Adaptively integrate both cubic matrices over one declared window.

    Uses the exact ``0.5j/(2*pi)`` prefactor, independent lesser/greater pair
    contractions and no conjugation. Both Green arguments are zero outside the
    cutoff; panels are clipped to their common support. Exact zero in either
    argument becomes a panel endpoint, never a pointwise Green evaluation.
    Sparse structure is compiled once, and no full frequency Green cache exists.
    All final per-leaf matrices use the compact ``active_indices`` basis.

    Wall limits are cooperative between bounded batches; an in-flight callback
    cannot be interrupted here. Callers should also enforce a process cap.
    """
    started = time.monotonic()
    if not isinstance(vertex, (DenseCubicVertex, SparseCubicVertex)):
        raise TypeError("vertex must be DenseCubicVertex or SparseCubicVertex.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    points = _breakpoints(breakpoints)
    omega, cutoff = float(output_frequency), float(cutoff)
    if not math.isfinite(omega) or not math.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("output frequency must be finite and cutoff finite and positive.")
    _controls(rtol,atol,max_pair_evaluations,max_seconds,max_panels,max_workers,progress)
    lower, upper = max(-cutoff, omega-cutoff, points[0]), min(cutoff, omega+cutoff, points[-1])
    if abs(omega)/2 >= cutoff or lower >= upper:
        effective = ()
    else:
        effective = tuple(sorted({lower,upper,*(x for x in points if lower < x < upper),
                                  *(x for x in (0.0,omega) if lower < x < upper)}))
    if len(effective)-1 > max_panels:
        raise ValueError("Initial partition exceeds max_panels.")
    sparse = isinstance(vertex, SparseCubicVertex)
    schedule = _compiled_sparse_schedule(vertex) if sparse else None
    active = schedule.active if sparse else np.arange(vertex.ndof)
    compact = np.ix_(active,active)
    def fields(nu):
        values = evaluator(float(nu))
        if not isinstance(values,(tuple,list)) or len(values)!=2:
            raise ValueError("evaluator must return lesser and greater matrices.")
        matrices=[]
        for value in values:
            array=np.asarray(value,dtype=np.complex128)
            if array.shape!=(vertex.ndof,vertex.ndof) or not np.all(np.isfinite(array)):
                raise ValueError("Green evaluator returned invalid shape or non-finite values.")
            matrices.append(np.array(array[compact],copy=True))
        return matrices
    def evaluate(nu):
        first=fields(nu)
        partner=omega-nu
        second,count=(first,1) if partner==nu else (fields(partner),2)
        result=[(_sparse_pair_contraction(vertex,schedule,g1,g2) if sparse else
                 _contract_cubic_pair(vertex.values,g1,g2)) for g1,g2 in zip(first,second)]
        return .5j/(2*np.pi)*np.asarray(result),count
    if len(active)==0:
        # Zero sparse vertices have no pairs to evaluate; retain domain bounds
        # as complete zero panels, with exact vanishing error.
        dimension=0
        zero=np.zeros((2,0,0),complex)
        leaves=tuple(AdaptiveMatrixLeaf(a,b,zero,zero) for a,b in zip(effective,effective[1:]))
        metrics=_metrics(leaves,dimension,float(rtol),float(atol))
        result=AdaptiveMatrixResult(metrics[0],metrics[1],leaves,effective,*metrics[2:],0,0,
                                    True,True,"zero-vertex",time.monotonic()-started)
    else:
        result=_adaptive_core(evaluate,effective,len(active),rtol=rtol,atol=atol,
                              max_pair_evaluations=max_pair_evaluations,max_seconds=max_seconds,
                              max_panels=max_panels,max_workers=max_workers,progress=progress,
                              started=started)
    full=np.zeros((2,vertex.ndof,vertex.ndof),complex)
    for component in (0,1): full[component][compact]=result.high[component]
    fields_dict={name:getattr(result,name) for name in result.__dataclass_fields__}
    return AdaptiveWindowResult(**fields_dict,lesser=full[0],greater=full[1],
                                active_indices=active,output_frequency=omega,cutoff=cutoff)
