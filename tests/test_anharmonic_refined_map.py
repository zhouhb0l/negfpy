"""Independent whole-cell stitching, retarded propagation and budget checks."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import negfpy.anharmonic.refined_map as module
import negfpy.anharmonic.interacting_pointwise as pointwise_module
from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.frequency_refinement import GaussLegendreWindowPlan
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.refined_map import (
    ResearchRefinementSpec, discrete_cell_contribution,
    refined_fixed_point_evaluation,
)
from negfpy.anharmonic.retarded import retarded_from_keldysh
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import (
    CurrentReport, DenseCubicVertex, KeldyshSpectrum, SCBAConfig,
)


def _fields(nu):
    first = np.array([[1 + .2j, .7 - .3j], [.2 + .1j, -.3j]])
    second = np.array([[.5 - .1j, -.2 + .6j], [.8 - .3j, 1.1]])
    return first * (1 + .3*nu**2), second * (2 + .2*nu + .1*nu**2)


def _vertex():
    # Symmetric real tensor with two nontrivial output rows.
    return DenseCubicVertex(np.array([[[.4, .2], [.2, -.3]], [[.2, -.3], [-.3, .7]]]))


@pytest.fixture
def rig(monkeypatch):
    grid = SignedFrequencyGrid(.25, 12)
    problem = SimpleNamespace(grid=grid, vertex=_vertex(), ndof=2)
    config = SCBAConfig(eta_lead=.025, backend="fft", hilbert_backend="fft", max_workers=2)
    fields = np.array([_fields(nu) for nu in grid.values])
    # Deliberate nodal offset: subtraction must use grid Green arrays, never
    # trust the pointwise callback to reproduce them.
    green = KeldyshSpectrum(fields[:, 0] + .3, fields[:, 0] + .2j, fields[:, 1] - .1j)
    sigma = KeldyshSpectrum.zeros(n_frequency=grid.n_points, ndof=2)
    currents = CurrentReport(.2, -.19, -.01)
    monkeypatch.setattr(module, "_solve_green", lambda *args: (green, sigma))
    monkeypatch.setattr(module, "_current_report", lambda *args: currents)
    class Evaluator:
        def __init__(self, *args, **kwargs):
            pass
        def __call__(self, nu):
            assert nu != 0
            return _fields(nu)
    monkeypatch.setattr(pointwise_module, "PointwiseInteracting", Evaluator)
    return SimpleNamespace(problem=problem, config=config, green=green,
                           sigma=sigma, currents=currents)


def _spec(grid, output_offset=-8, cell_offsets=(-4,), order=8):
    cells = tuple(grid.zero_index + offset for offset in cell_offsets)
    groups = module._cell_groups(cells)
    plans = []
    for group in groups:
        bounds = [grid.values[group[0]] - grid.step/2]
        bounds.extend(grid.values[index] + grid.step/2 for index in group)
        plans.append(GaussLegendreWindowPlan(tuple(bounds), order))
    return ResearchRefinementSpec(grid.zero_index + output_offset, cells, tuple(plans))


def _run(rig, specs, **kwargs):
    return refined_fixed_point_evaluation(
        rig.problem, rig.sigma, rig.sigma, rig.sigma.lesser, rig.sigma.greater,
        rig.config, specs, **kwargs)


def _oracle_pair(vertex, first, second):
    return np.einsum("acd,bef,ce,df->ab", vertex.values, vertex.values,
                     first, second, optimize=False)


@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("output,cells", [(4, (2, 3)), (12, (8, 16)), (20, (16, 17))])
def test_discrete_cells_match_independent_complex_dense_sum(rig, sparse, output, cells):
    grid, dense = rig.problem.grid, rig.problem.vertex
    vertex = SparseCubicVertex.from_dense(dense) if sparse else dense
    actual = discrete_cell_contribution(vertex, rig.green, grid, output, cells)
    for component, field in enumerate((rig.green.lesser, rig.green.greater)):
        expected = sum(_oracle_pair(dense, field[index], field[output-index+grid.zero_index])
                       for index in cells) * .5j * grid.integration_weight
        np.testing.assert_allclose(actual[component], expected, rtol=3e-14, atol=1e-14)


@pytest.mark.parametrize("sparse", [False, True])
def test_whole_matrix_stitch_uses_grid_terms_and_propagates_retarded_full_grid(rig, sparse):
    dense = rig.problem.vertex
    if sparse:
        rig.problem.vertex = SparseCubicVertex.from_dense(dense)
    grid = rig.problem.grid
    specs = [_spec(grid, -8, (-4,)), _spec(grid, 0, (-4, 4)), _spec(grid, 8, (4,))]
    result = _run(rig, specs)
    expected = list(cubic_lesser_greater(dense, rig.green.lesser, rig.green.greater, grid, backend="serial"))
    coarse_retarded = retarded_from_keldysh(*expected, grid)
    for spec in specs:
        for cells, plan in zip(module._cell_groups(spec.cell_indices), spec.plans):
            output = spec.output_index
            omega = grid.values[output]
            for component, field in enumerate((rig.green.lesser, rig.green.greater)):
                coarse = sum(_oracle_pair(dense, field[index], field[output-index+grid.zero_index])
                             for index in cells) * .5j * grid.integration_weight
                integral = sum(weight * _oracle_pair(dense, _fields(nu)[component], _fields(omega-nu)[component])
                               for nu, weight in zip(plan.nodes, plan.weights)) * .5j/(2*np.pi)
                expected[component][output] += integral - coarse
    np.testing.assert_allclose(result.raw_lesser, expected[0], rtol=3e-13, atol=1e-14)
    np.testing.assert_allclose(result.raw_greater, expected[1], rtol=3e-13, atol=1e-14)
    expected_retarded = retarded_from_keldysh(*expected, grid)
    np.testing.assert_allclose(result.raw_retarded, expected_retarded, rtol=3e-13, atol=1e-14)
    unchanged_output = grid.zero_index + 1
    assert np.max(abs(result.raw_retarded[unchanged_output] - coarse_retarded[unchanged_output])) > 1e-4
    assert result.completed and len(result.windows) == 4
    assert result.currents is rig.currents
    assert result.sigma is rig.sigma and result.green is rig.green
    assert np.isfinite(result.residual) and np.isfinite(result.coarse_residual)
    assert all(window.applied and window.quadrature.completed for window in result.windows)
    # Offdiagonal and greater changes are computed independently, no scalar
    # normalization of a selected entry can reproduce these ratios.
    ratio = result.raw_lesser[4] / result.raw_greater[4]
    assert np.ptp(abs(ratio)) > .01


@pytest.mark.parametrize("spec_factory,match", [
    (lambda grid: [_spec(grid), _spec(grid)], "Duplicate output"),
    (lambda grid: [_spec(grid, -13)], "output_index"),
    (lambda grid: [_spec(grid, -8, (-12,))], "inside cutoff"),
    (lambda grid: [_spec(grid, -8, (4,))], "inside cutoff"),
    (lambda grid: [replace(_spec(grid), plans=(GaussLegendreWindowPlan((-1.125, -.9)),))], "endpoints"),
    (lambda grid: [replace(_spec(grid, -8, (-4, -3)), plans=(GaussLegendreWindowPlan((-1.125, -.625)),))], "every declared"),
    (lambda grid: [replace(_spec(grid, 0, (-4, 4)), plans=(GaussLegendreWindowPlan((-1.125, 1.125)),))], "one quadrature"),
    (lambda grid: [_spec(grid, 0, (0,))], "Split panels"),
])
def test_invalid_cell_plans_fail_before_dyson(rig, monkeypatch, spec_factory, match):
    monkeypatch.setattr(module, "_solve_green", lambda *args: pytest.fail("must validate first"))
    with pytest.raises(ValueError, match=match):
        _run(rig, spec_factory(rig.problem.grid))


@pytest.mark.parametrize("kwargs,match", [
    ({"output_index": True}, "integer"),
    ({"cell_indices": (8, 8)}, "unique"),
    ({"cell_indices": (9, 8)}, "increasing"),
    ({"cell_indices": (8.0,)}, "integer"),
    ({"plans": (GaussLegendreWindowPlan((-1.125, -.875), 1),)}, "GL8"),
])
def test_spec_rejects_ambiguous_indices_and_midpoint_replacement(kwargs, match):
    args = dict(output_index=4, cell_indices=(8,), plans=(GaussLegendreWindowPlan((-1.125, -.875)),))
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        ResearchRefinementSpec(**args)


def test_expired_global_budget_prevents_any_quadrature(rig, monkeypatch):
    ticks = iter([0., 2., 3.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(module, "cubic_window_lesser_greater", lambda *args, **kwargs: pytest.fail("budget expired"))
    result = _run(rig, [_spec(rig.problem.grid)], max_seconds=1)
    expected = cubic_lesser_greater(rig.problem.vertex, rig.green.lesser, rig.green.greater,
                                   rig.problem.grid, backend="fft", max_workers=2)
    assert not result.completed and result.residual is None and not result.windows
    np.testing.assert_array_equal(result.raw_lesser, expected[0])
    np.testing.assert_array_equal(result.raw_greater, expected[1])


def test_partial_window_is_evidence_only_never_applied(rig, monkeypatch):
    original = module.cubic_window_lesser_greater
    calls = []
    def partial(*args, **kwargs):
        calls.append(kwargs)
        result = original(*args, **kwargs)
        return replace(result, completed=False, pair_evaluations=3,
                       lesser=result.lesser*100, greater=result.greater*100)
    monkeypatch.setattr(module, "cubic_window_lesser_greater", partial)
    result = _run(rig, [_spec(rig.problem.grid), _spec(rig.problem.grid, 8, (4,))],
                  max_seconds_per_window=10)
    expected = cubic_lesser_greater(rig.problem.vertex, rig.green.lesser, rig.green.greater,
                                   rig.problem.grid, backend="fft", max_workers=2)
    assert len(calls) == 1 and calls[0]["max_seconds"] == 10
    assert not result.completed and result.residual is None
    assert len(result.windows) == 1 and not result.windows[0].applied
    np.testing.assert_array_equal(result.raw_lesser, expected[0])
    np.testing.assert_array_equal(result.raw_greater, expected[1])


def test_coarse_extraction_cannot_start_quadrature_after_global_expiry(rig, monkeypatch):
    ticks = iter([0., .1, 2., 3.])
    monkeypatch.setattr(module.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(module, "cubic_window_lesser_greater", lambda *args, **kwargs: pytest.fail("global expiry"))
    result = _run(rig, [_spec(rig.problem.grid)], max_seconds=1)
    assert not result.completed and result.residual is None and not result.windows


def test_reflection_at_zero_uses_two_complete_cells_without_double_counting(rig):
    grid = rig.problem.grid
    result = _run(rig, [_spec(grid, 0, (-4, 4))])
    assert result.completed
    assert [w.cell_indices for w in result.windows] == [(8,), (16,)]
    assert all(w.quadrature.output_frequency == 0 for w in result.windows)
    assert sum(w.quadrature.pair_evaluations for w in result.windows) == 16


@pytest.mark.parametrize("kwargs", [{"max_seconds": 0}, {"max_seconds_per_window": float("nan")},
                                    {"max_nodes_per_window": 7}])
def test_bad_budgets_rejected_before_computation(rig, monkeypatch, kwargs):
    monkeypatch.setattr(module, "_solve_green", lambda *args: pytest.fail("must validate first"))
    with pytest.raises(ValueError):
        _run(rig, [_spec(rig.problem.grid)], **kwargs)


def test_progress_records_bind_window_output_and_cells(rig):
    records = []
    result = _run(rig, [_spec(rig.problem.grid, 0, (-4, 4))], progress=records.append)
    assert result.completed
    assert [record["window_index"] for record in records] == [0, 1]
    assert [record["cell_indices"] for record in records] == [[8], [16]]
    assert all(record["completed"] and record["final"] and record["output_frequency"] == 0 for record in records)


def test_real_two_map_equilibrium_path_preserves_relations_and_nodal_dyson():
    """Exercise contacts, interacting interpolation and two actual map calls."""
    from negfpy.anharmonic.contacts import build_contact_spectra
    from negfpy.anharmonic.interacting_pointwise import PointwiseInteracting
    from negfpy.anharmonic.types import CubicSCBAProblem, HarmonicReservoir
    from negfpy.core.types import Device1D, LeadBlocks
    grid = SignedFrequencyGrid(.125, 32)
    lead = LeadBlocks(d00=np.array([[4.]]), d01=np.array([[-.6]]))
    device = Device1D(onsite_blocks=[np.array([[4.]]), np.array([[4.1]])],
                      coupling_blocks=[np.array([[-.4]])])
    problem = CubicSCBAProblem(device=device, left=HarmonicReservoir(lead, .7),
                              right=HarmonicReservoir(lead, .7),
                              vertex=DenseCubicVertex(.03*_vertex().values), grid=grid)
    config = SCBAConfig(eta_lead=.025, backend="fft", hilbert_backend="fft", max_workers=2)
    left, right = build_contact_spectra(problem, config.eta_lead)
    zero = np.zeros_like(left.lesser)
    specs = [_spec(grid, -22, (-11,)), _spec(grid, 0, (-11, 11)), _spec(grid, 22, (11,))]
    first = refined_fixed_point_evaluation(problem, left, right, zero, zero, config, specs)
    second = refined_fixed_point_evaluation(problem, left, right, .2*first.raw_lesser,
                                            .2*first.raw_greater, config, specs)
    for result in (first, second):
        assert result.completed and np.isfinite(result.residual)
        np.testing.assert_allclose(result.raw_greater-result.raw_lesser,
                                   result.raw_retarded-result.raw_retarded.conj().transpose(0, 2, 1),
                                   rtol=2e-10, atol=1e-12)
        assert abs(result.currents.contact_balance) < 1e-10
        assert abs(result.currents.collision_power) < 1e-10
    assert np.max(abs(second.sigma.retarded)) > 1e-8
    pointwise = PointwiseInteracting(problem, config, second.sigma, policy="equilibrium-broadening")
    for offset in (-11, 11):
        index = grid.zero_index + offset
        sample = pointwise.evaluate(float(grid.values[index]))
        for field in ("lesser", "greater", "retarded"):
            np.testing.assert_allclose(getattr(sample, field), getattr(second.green, field)[index],
                                       rtol=1e-11, atol=1e-12)
