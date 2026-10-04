from __future__ import annotations

from dataclasses import FrozenInstanceError
import itertools
import threading
import time

import numpy as np
import pytest

import negfpy.anharmonic.frequency_refinement as refinement
from negfpy.anharmonic.frequency_refinement import (
    GaussLegendreWindowPlan,
    cubic_window_lesser_greater,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import DenseCubicVertex


def vertex() -> DenseCubicVertex:
    values = np.zeros((4, 4, 4))
    for index in set(itertools.permutations((0, 1, 3))):
        values[index] = -0.41
    values[0, 0, 0] = 0.12
    values[3, 3, 3] = -0.21
    return DenseCubicVertex(values)


def fields(nu: float) -> tuple[np.ndarray, np.ndarray]:
    a = np.arange(16).reshape(4, 4) / 23
    # Non-Hermitian, complex off-diagonal values expose conjugations and
    # transpositions; the components intentionally have different dependence.
    lesser = (1.2 + 0.4j) * a + (0.03 - 0.08j) * nu * a.T + 0.2j
    greater = (0.7 - 0.6j) * a.T + (-0.01 + 0.2j) * nu**2 * a + 0.3
    return lesser, greater


def oracle(v, evaluate, output, plan, cutoff):
    clipped = plan.clipped(max(-cutoff, output-cutoff), min(cutoff, output+cutoff))
    results = np.zeros((2, v.ndof, v.ndof), complex)
    if clipped is None:
        return results
    for nu, weight in zip(clipped.nodes, clipped.weights):
        first, second = evaluate(nu), evaluate(output-nu)
        for component in (0, 1):
            results[component] += weight * np.einsum(
                'acd,bef,ce,df->ab', v.values, v.values,
                first[component], second[component], optimize=False)
    return 0.5j / (2*np.pi) * results


@pytest.mark.parametrize('output', [-1.17, 0.0, 0.83])
@pytest.mark.parametrize('order', [8, 16])
def test_dense_complex_full_matrices_match_independent_einsum(output, order):
    plan = GaussLegendreWindowPlan((-1.2, -0.91, -0.09, 0.34, 1.12), order)
    v = vertex()
    result = cubic_window_lesser_greater(v, fields, output, plan, cutoff=2)
    expected = oracle(v, fields, output, plan, cutoff=2)
    np.testing.assert_allclose(result.lesser, expected[0], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result.greater, expected[1], rtol=1e-13, atol=1e-14)
    assert not np.allclose(result.lesser, result.greater)
    assert result.completed
    assert result.green_evaluations == result.pair_evaluations * 2


def test_sparse_matches_dense_and_threaded_is_bitwise_deterministic(monkeypatch):
    plan = GaussLegendreWindowPlan((-1.4, -1.38, -1.2), 16)
    v = vertex()
    sparse = SparseCubicVertex.from_dense(v)
    calls = []
    original = refinement._compiled_sparse_schedule
    monkeypatch.setattr(refinement, '_compiled_sparse_schedule', lambda v: (calls.append(v), original(v))[1])
    dense = cubic_window_lesser_greater(v, fields, -2.6, plan, cutoff=4)
    serial = cubic_window_lesser_greater(sparse, fields, -2.6, plan, cutoff=4)
    parallel = cubic_window_lesser_greater(sparse, fields, -2.6, plan, cutoff=4, max_workers=8)
    assert len(calls) == 2  # Exactly once per sparse call, not per pair.
    for name in ('lesser', 'greater'):
        np.testing.assert_allclose(getattr(serial, name), getattr(dense, name), rtol=1e-13, atol=1e-14)
        np.testing.assert_array_equal(getattr(serial, name), getattr(parallel, name))
        np.testing.assert_array_equal(getattr(serial, name)[2, :], 0)
        np.testing.assert_array_equal(getattr(serial, name)[:, 2], 0)


@pytest.mark.parametrize('output,expected_span', [(1.1, (0.1, 1.0)), (-1.1, (-1.0, -0.1)), (0.0, (-1.0, 1.0))])
def test_zero_extension_clips_panels_before_sampling(output, expected_span):
    plan = GaussLegendreWindowPlan((-4.0, -0.3, 0.5, 4.0), 8)
    frequencies = []
    def evaluate(nu):
        assert -1 <= nu <= 1
        frequencies.append(nu)
        return np.ones((1, 1)), 2j*np.ones((1, 1))
    result = cubic_window_lesser_greater(DenseCubicVertex(np.ones((1, 1, 1))), evaluate,
                                        output, plan, cutoff=1)
    expected = 0.5j/(2*np.pi)*(expected_span[1]-expected_span[0])
    np.testing.assert_allclose(result.lesser, [[expected]], rtol=1e-14)
    np.testing.assert_allclose(result.greater, [[-4*expected]], rtol=1e-14)
    assert len(frequencies) == result.green_evaluations
    np.testing.assert_allclose([result.effective_breakpoints[0], result.effective_breakpoints[-1]], expected_span)


@pytest.mark.parametrize('output', [-3.0, -2.0, 2.0, 3.0])
def test_disjoint_support_never_calls_evaluator(output):
    def reject(nu):
        raise AssertionError('outside support callback')
    result = cubic_window_lesser_greater(vertex(), reject, output,
        GaussLegendreWindowPlan((-4., 4.)), cutoff=1)
    np.testing.assert_array_equal(result.lesser, 0)
    np.testing.assert_array_equal(result.greater, 0)
    assert result.completed and result.pair_evaluations == result.green_evaluations == 0
    assert result.effective_breakpoints == ()


def test_zero_sparse_vertex_never_calls_evaluator():
    v = SparseCubicVertex.from_dense(DenseCubicVertex(np.zeros((2, 2, 2))))
    def reject(nu):
        raise AssertionError('zero vertex callback')
    result = cubic_window_lesser_greater(v, reject, 0.2,
        GaussLegendreWindowPlan((-1., 1.)), cutoff=2)
    np.testing.assert_array_equal(result.lesser, 0)
    assert result.completed and result.pair_evaluations == 0


def test_tracked_ranges_use_same_nodes_and_do_not_recompute_fields():
    plan = GaussLegendreWindowPlan((-1.4, -1.3875, -1.38, -1.3625, -1.3), 16)
    calls = []
    def evaluate(nu):
        calls.append(nu)
        return fields(nu)
    result = cubic_window_lesser_greater(vertex(), evaluate, -2.75, plan,
        cutoff=5, max_workers=4, tracked_ranges={'core':(-1.3875, -1.3625)})
    expected = oracle(vertex(), fields, -2.75,
        GaussLegendreWindowPlan((-1.3875, -1.38, -1.3625), 16), 5)
    np.testing.assert_allclose(result.ranges['core'].lesser, expected[0], rtol=1e-13, atol=1e-14)
    np.testing.assert_allclose(result.ranges['core'].greater, expected[1], rtol=1e-13, atol=1e-14)
    assert len(calls) == result.green_evaluations == 2*plan.n_points


def test_serial_reused_callback_scratch_is_copied():
    buffers = [np.empty((4, 4), complex), np.empty((4, 4), complex)]
    def reused(nu):
        buffers[0][:], buffers[1][:] = fields(nu)
        return tuple(buffers)
    plan = GaussLegendreWindowPlan((-1.2, -.8), 8)
    expected = cubic_window_lesser_greater(vertex(), fields, -1.9, plan, cutoff=3)
    result = cubic_window_lesser_greater(vertex(), reused, -1.9, plan, cutoff=3)
    np.testing.assert_array_equal(result.lesser, expected.lesser)
    np.testing.assert_array_equal(result.greater, expected.greater)


def test_plan_and_result_are_immutable_copies():
    points = [-1.0, 0.0, 1.0]
    plan = GaussLegendreWindowPlan(points, 8)
    points[0] = -2
    assert plan.breakpoints == (-1.0, 0.0, 1.0)
    assert sum(plan.weights) == pytest.approx(2.0)
    with pytest.raises(FrozenInstanceError):
        plan.order = 16
    for array in (plan.nodes, plan.weights):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    result = cubic_window_lesser_greater(vertex(), fields, .1, plan, cutoff=2,
                                         tracked_ranges={'left':(-1., 0.)})
    with pytest.raises(TypeError):
        result.ranges['bad'] = result.ranges['left']
    for array in (result.lesser, result.greater, result.ranges['left'].lesser):
        with pytest.raises(ValueError):
            array.setflags(write=True)


@pytest.mark.parametrize('breakpoints,order', [((1, 1),8), ((2,1),8), ((0,np.nan),8), ((0,np.inf),8), ((0,),8), ((0,1),4), ((0,1),True), ((0,1),8.0)])
def test_invalid_plans_fail(breakpoints, order):
    with pytest.raises(ValueError):
        GaussLegendreWindowPlan(breakpoints, order)


@pytest.mark.parametrize('kwargs', [dict(cutoff=0), dict(cutoff=np.inf), dict(cutoff=np.nan), dict(max_workers=0), dict(max_workers=9), dict(max_workers=True), dict(max_nodes=7), dict(max_seconds=0), dict(max_seconds=np.nan), dict(tracked_ranges={'core':(-.5,.5)})])
def test_invalid_execution_controls_fail_before_callback(kwargs):
    def reject(nu):
        raise AssertionError('unexpected evaluation')
    arguments = dict(cutoff=2)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        cubic_window_lesser_greater(vertex(), reject, .1,
                                   GaussLegendreWindowPlan((-1.,1.),8), **arguments)


@pytest.mark.parametrize('value', [(np.zeros((3,3)),np.zeros((4,4))), (np.zeros((4,4)),np.full((4,4),np.nan)), np.zeros((2,4,4))])
def test_bad_callback_fields_are_rejected(value):
    with pytest.raises(ValueError, match='evaluator'):
        cubic_window_lesser_greater(vertex(), lambda nu:value, .1,
                                   GaussLegendreWindowPlan((-1.,1.),8), cutoff=2)


def test_budget_exhaustion_returns_explicit_partial_without_extrapolation(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(refinement.time, 'monotonic', lambda:clock[0])
    def evaluate(nu):
        clock[0] += .1
        return fields(nu)
    plan = GaussLegendreWindowPlan((-1.,0.,1.),8)
    progress = []
    result = cubic_window_lesser_greater(vertex(), evaluate, .1, plan, cutoff=2,
        max_seconds=.31, tracked_ranges={'left':(-1.,0.)}, progress=progress.append)
    assert not result.completed
    assert result.pair_evaluations == 2
    assert result.planned_pair_evaluations == 16
    assert result.green_evaluations == 4
    assert progress[-1]['final'] and not progress[-1]['completed']
    expected = np.zeros((4,4),complex)
    for nu, weight in zip(plan.nodes[:2],plan.weights[:2]):
        expected += weight*np.einsum('acd,bef,ce,df->ab', vertex().values,
            vertex().values, fields(nu)[0], fields(.1-nu)[0])
    expected *= .5j/(2*np.pi)
    np.testing.assert_allclose(result.lesser, expected, atol=1e-14)
    np.testing.assert_array_equal(result.ranges['left'].lesser,result.lesser)


def test_bounded_workers_and_compact_progress():
    lock = threading.Lock()
    state = {'active':0,'maximum':0}
    def evaluate(nu):
        with lock:
            state['active'] += 1
            state['maximum'] = max(state['maximum'],state['active'])
        time.sleep(.001)
        result = fields(nu)
        with lock:
            state['active'] -= 1
        return result
    progress = []
    result = cubic_window_lesser_greater(vertex(), evaluate, .1,
        GaussLegendreWindowPlan(tuple(np.linspace(-1.,1.,11)),8), cutoff=2,
        max_workers=4, progress=progress.append)
    assert 1 < state['maximum'] <= 4
    assert [entry['pair_evaluations'] for entry in progress] == [64,80]
    assert progress[-1]['completed'] and result.completed
    assert all(not isinstance(value,np.ndarray) for entry in progress for value in entry.values())


def test_order_one_is_exact_complex_central_rectangle_and_reuses_pair():
    plan = GaussLegendreWindowPlan((-1.3875, -1.3625), 1)
    assert plan.nodes.tolist() == [-1.375]
    assert plan.weights[0] == pytest.approx(.025)
    result = cubic_window_lesser_greater(vertex(), fields, -2.75, plan, cutoff=7.5)
    expected = oracle(vertex(), fields, -2.75, plan, 7.5)
    np.testing.assert_allclose(result.lesser, expected[0], atol=1e-14)
    np.testing.assert_allclose(result.greater, expected[1], atol=1e-14)
    assert result.pair_evaluations == result.green_evaluations == 1
