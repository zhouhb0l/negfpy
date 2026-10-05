"""Bounded full-matrix GL8/16 research quadrature.

Extracted unchanged from the authenticated full-domain moment diagnostic
25688099.pbs101. All final leaves are retained; sampled error agreement is
not a global accuracy bound. Callers must provide a separate process timeout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import numpy as np

def _maximum(value):
    return float(np.max(np.abs(value), initial=0.))

def moment_metrics(value):
    value = np.asarray(value)
    identity = np.eye(value.shape[0])
    delta = value - identity
    return dict(relative_max_residual=_maximum(delta) / max(1., _maximum(value)),
                identity_normalized_frobenius=float(np.linalg.norm(delta)) / math.sqrt(len(value)),
                maximum_magnitude=_maximum(value),
                hermitian_relative=_maximum(value - value.conj().T) / max(_maximum(value), 1e-300))

class _Budget(Exception):
    pass

@dataclass
class _Leaf:
    parent: int
    left: float
    right: float
    low: np.ndarray
    high: np.ndarray

    @property
    def error_abs(self):
        return np.abs(self.high - self.low)

class _Evaluator:
    def __init__(self, evaluate, shape, max_points, max_seconds, max_workers, clock):
        self.evaluate = evaluate
        self.shape = shape
        self.max_points, self.max_seconds = max_points, max_seconds
        self.clock, self.started = clock, clock()
        self.points = 0
        self.pool = ThreadPoolExecutor(max_workers=max_workers)
        self.reason = None

    def check(self, required):
        if self.clock() - self.started >= self.max_seconds:
            self.reason = 'wall-budget'
            raise _Budget()
        if self.points + required > self.max_points:
            self.reason = 'point-budget'
            raise _Budget()

    def panels(self, intervals, orders):
        plans, frequencies = [], []
        for left, right in intervals:
            panel = []
            for order in orders:
                nodes, weights = np.polynomial.legendre.leggauss(order)
                omega = .5 * (left + right) + .5 * (right - left) * nodes
                if np.any(omega == 0.) or np.any(omega <= left) or np.any(omega >= right):
                    raise ValueError('GL nodes must be nonzero and strictly inside each interval.')
                panel.append((len(frequencies), .5 * (right - left) * weights))
                frequencies.extend(float(w) for w in omega)
            plans.append(panel)
        self.check(len(frequencies))
        self.points += len(frequencies)  # Every submitted evaluation counts, including a failure.
        results = list(self.pool.map(self.evaluate, frequencies))
        matrices = []
        for result in results:
            value = np.asarray(result, dtype=complex)
            if value.shape != self.shape or not np.all(np.isfinite(value)):
                raise ValueError('Pointwise integrand returned a nonfinite matrix or wrong shape.')
            matrices.append(value)
        integrated = []
        for panel in plans:
            integrated.append(tuple(sum((weight * matrices[start + i]
                                          for i, weight in enumerate(weights)),
                                         np.zeros(self.shape, complex))
                                    for start, weights in panel))
        # A panel that finishes after its cooperative deadline stays evidence,
        # but cannot be accepted into the numerical partition.
        if self.clock() - self.started >= self.max_seconds:
            self.reason = 'wall-budget'
            raise _Budget()
        return integrated

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)

def _sampled_errors(leaves, rtol, atol):
    moment = sum((leaf.high for leaf in leaves), np.zeros_like(leaves[0].high))
    maximum = sum(_maximum(leaf.error_abs) for leaf in leaves)
    frobenius = sum(float(np.linalg.norm(leaf.high - leaf.low)) for leaf in leaves)
    max_tolerance = atol + rtol * _maximum(moment)
    frob_tolerance = atol + rtol * float(np.linalg.norm(moment))
    return dict(max_error=maximum, frobenius_error=frobenius,
                max_tolerance=max_tolerance, frobenius_tolerance=frob_tolerance,
                max_error_ratio=maximum / max(max_tolerance, 1e-300),
                frobenius_error_ratio=frobenius / max(frob_tolerance, 1e-300),
                rtol=rtol, atol=atol,
                passed=bool(maximum <= max_tolerance and frobenius <= frob_tolerance))

def _validate(grid_values, ndof, evaluate, emit, max_points, max_seconds,
              max_workers, max_leaves, rtol, atol, clock):
    grid = np.asarray(grid_values, dtype=float)
    if (grid.ndim != 1 or len(grid) < 3 or len(grid) % 2 != 1
            or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
            or not np.all(np.isfinite(np.diff(grid)))):
        raise ValueError('Grid must be finite, increasing, odd, and contain at least three nodes.')
    if grid[len(grid) // 2] != 0. or not np.allclose(grid, -grid[::-1], rtol=0., atol=1e-13):
        raise ValueError('Grid must be reflection symmetric with exact central zero.')
    for name, number, minimum in [('ndof', ndof, 1), ('max_points', max_points, 0),
                                  ('max_workers', max_workers, 1), ('max_leaves', max_leaves, 1)]:
        if isinstance(number, bool) or not isinstance(number, (int, np.integer)) or number < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}.')
    if max_leaves < len(grid) - 1:
        raise ValueError('All original intervals must fit within max_leaves.')
    if (not math.isfinite(max_seconds) or max_seconds <= 0
            or not all(math.isfinite(value) and value >= 0 for value in (rtol, atol))):
        raise ValueError('Budgets and tolerances must be finite and valid.')
    if not all(callable(function) for function in (evaluate, emit, clock)):
        raise TypeError('evaluate, emit, and clock must be callable.')
    return grid

def _leaf(parent, left, right, low, high):
    if not (np.all(np.isfinite(low)) and np.all(np.isfinite(high))):
        raise ValueError('Integrated GL panel is nonfinite.')
    result = _Leaf(int(parent), float(left), float(right), low, high)
    # Cache scalar errors once per newly accepted panel. Repeated matrix norms
    # across hundreds of unchanged leaves would dominate this cheap diagnostic.
    difference = high - low
    result.max_error = float(np.max(np.abs(difference), initial=0.))
    result.frobenius_error = float(np.linalg.norm(difference))
    if not (math.isfinite(result.max_error) and math.isfinite(result.frobenius_error)):
        raise ValueError('Integrated GL panel error is nonfinite.')
    return result

def _cached_errors(leaves, moment, rtol, atol):
    maximum = math.fsum(leaf.max_error for leaf in leaves)
    frobenius = math.fsum(leaf.frobenius_error for leaf in leaves)
    max_tolerance = atol + rtol * float(np.max(np.abs(moment), initial=0.))
    frob_tolerance = atol + rtol * float(np.linalg.norm(moment))
    return dict(max_error=maximum, frobenius_error=frobenius,
                max_tolerance=max_tolerance, frobenius_tolerance=frob_tolerance,
                max_error_ratio=maximum / max(max_tolerance, 1e-300),
                frobenius_error_ratio=frobenius / max(frob_tolerance, 1e-300),
                rtol=rtol, atol=atol,
                passed=bool(maximum <= max_tolerance and frobenius <= frob_tolerance))

def run_full_domain_quadrature(grid_values, evaluate, ndof, *, emit=lambda **row: None,
                               max_points=40000, max_seconds=1800., max_workers=8,
                               max_leaves=1024, rtol=.002, atol=1e-8,
                               clock=time.monotonic):
    """Return a JSON-safe report and complete final-leaf evidence.

    Each starting interval costs 24 point calls; an atomic split costs 48.
    Work that finishes after the cooperative deadline still counts but does
    not replace any accepted interval. Pointwise failures propagate after the
    worker pool closes. No incomplete initial partition can publish a domain
    moment or pass a sampled tolerance. Complete partitions can fail tolerance.

    Leaf parent indices refer to original grid intervals. Final leaves retain
    both GL integrals and elementwise absolute differences for independent
    reconstruction. Initial moment sums exist only after complete initial
    coverage; their constituent initial leaves may later have been replaced.
    """
    grid = _validate(grid_values, ndof, evaluate, emit, max_points, max_seconds,
                     max_workers, max_leaves, rtol, atol, clock)
    bounds = np.column_stack((grid[:-1], grid[1:]))
    shape = (int(ndof), int(ndof))
    arrays = dict(grid_values=grid.copy())
    leaves = []
    complete_initial = complete_partition = tolerance_met = False
    initial_completed = 0
    stop_reason, errors = None, None
    limits = dict(max_points=int(max_points), max_seconds=float(max_seconds),
                  max_workers=int(max_workers), max_leaves=int(max_leaves))
    runner = _Evaluator(evaluate, shape, max_points, max_seconds, max_workers, clock)
    emit(event='full_domain_quadrature_start', initial_intervals=len(bounds),
         adaptive_orders=[8, 16], limits=limits, rtol=rtol, atol=atol)
    try:
        for parent, (left, right) in enumerate(bounds):
            low, high = runner.panels([(left, right)], (8, 16))[0]
            leaves.append(_leaf(parent, left, right, low, high))
            initial_completed += 1
            if initial_completed % 64 == 0 or initial_completed == len(bounds):
                emit(event='full_domain_initial_progress',
                     completed_intervals=initial_completed, total_intervals=len(bounds),
                     point_evaluations=runner.points,
                     elapsed_seconds=runner.clock() - runner.started)
        complete_initial = complete_partition = True
        arrays['initial_low_moment'] = sum((leaf.low for leaf in leaves), np.zeros(shape, complex))
        arrays['initial_high_moment'] = sum((leaf.high for leaf in leaves), np.zeros(shape, complex))
        moment = arrays['initial_high_moment'].copy()
        splits = 0
        while True:
            errors = _cached_errors(leaves, moment, rtol, atol)
            # Reconstruct directly before accepting tolerance; incremental
            # moment updates cannot make a roundoff-sensitive pass decision.
            if errors['passed']:
                errors = _sampled_errors(leaves, rtol, atol)
                moment = sum((leaf.high for leaf in leaves), np.zeros(shape, complex))
            if splits % 8 == 0 or errors['passed']:
                emit(event='full_domain_adaptive_progress', leaves=len(leaves),
                     sampled_errors=errors, point_evaluations=runner.points,
                     elapsed_seconds=runner.clock() - runner.started)
            if errors['passed']:
                tolerance_met, stop_reason = True, 'sampled-tolerance'
                break
            if len(leaves) >= max_leaves:
                stop_reason = 'leaf-budget'
                break
            def score(leaf):
                return max(leaf.max_error / max(errors['max_tolerance'], 1e-300),
                           leaf.frobenius_error / max(errors['frobenius_tolerance'], 1e-300))
            index = max(range(len(leaves)), key=lambda i: (score(leaves[i]), -leaves[i].left))
            old = leaves[index]
            midpoint = .5 * (old.left + old.right)
            if midpoint == old.left or midpoint == old.right:
                stop_reason = 'floating-point-resolution'
                break
            child_bounds = [(old.left, midpoint), (midpoint, old.right)]
            # Both children finish within budget before either replaces the
            # complete old parent. Any late or incomplete split is discarded.
            result = runner.panels(child_bounds, (8, 16))
            children = [_leaf(old.parent, left, right, low, high)
                        for (left, right), (low, high) in zip(child_bounds, result)]
            moment += children[0].high + children[1].high - old.high
            leaves[index:index + 1] = children
            splits += 1
    except _Budget:
        stop_reason = runner.reason
    finally:
        runner.close()
    leaves.sort(key=lambda leaf: (leaf.parent, leaf.left))
    arrays.update(leaf_bounds=np.array([(leaf.left, leaf.right) for leaf in leaves]).reshape(-1, 2),
                  leaf_parent_indices=np.array([leaf.parent for leaf in leaves], dtype=np.int64),
                  leaf_low=np.stack([leaf.low for leaf in leaves]) if leaves else np.empty((0, *shape), complex),
                  leaf_high=np.stack([leaf.high for leaf in leaves]) if leaves else np.empty((0, *shape), complex),
                  leaf_error_abs=np.stack([leaf.error_abs for leaf in leaves]) if leaves else np.empty((0, *shape), float))
    if complete_partition:
        arrays['full_domain_moment'] = arrays['leaf_high'].sum(axis=0)
        errors = _sampled_errors(leaves, rtol, atol)
        # A consumed budget never becomes tolerance acceptance after stopping.
        tolerance_met = bool(tolerance_met and errors['passed'])
    report = dict(complete_initial_partition=complete_initial,
                  complete_partition=complete_partition,
                  sampled_tolerance_met=tolerance_met, stop_reason=stop_reason,
                  point_evaluations=runner.points, leaf_count=len(leaves),
                  initial_intervals=len(bounds), initial_completed_intervals=initial_completed,
                  limits=limits, rtol=rtol, atol=atol, adaptive_orders=[8, 16],
                  sampled_errors=errors,
                  moments={name: moment_metrics(value) for name, value in arrays.items()
                           if name.endswith('_moment')},
                  elapsed_seconds=runner.clock() - runner.started,
                  globally_certified=False,
                  evidence_scope='Every final full-matrix GL8/16 leaf retained; pointwise Green matrices not retained.')
    emit(event='full_domain_quadrature_summary', **report)
    return report, arrays
