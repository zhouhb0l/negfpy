"""New allocation, failure-path and state-semantics checks for one-map integration."""

from dataclasses import replace

import numpy as np
import pytest
from test_anharmonic_transverse_material_map_reference import _causal_input, _fixture
from threadpoolctl import threadpool_info

from negfpy.anharmonic.transverse_material_map import (
    PackedMapSpectrum,
    evaluate_packed_born_map,
    plan_material_born_map,
)


@pytest.mark.parametrize('budget', ['max_bytes','max_contraction_work','max_dyson_evaluations'])
def test_plan_guards_precede_dyson(monkeypatch, budget):
    args = _fixture(1)
    def forbidden(*args, **kwargs):
        pytest.fail('Dyson reached before resource rejection')
    monkeypatch.setattr(np.linalg, 'inv', forbidden)
    with pytest.raises(ValueError, match=budget):
        evaluate_packed_born_map(*args, **{budget:1})


def test_material_dimension_plan_retains_full_harmonic_and_bounds_live_workspaces():
    plan = plan_material_born_map(4,601,216,120,24,24)
    assert plan.retained_green_bytes == 3*16*4*601*120**2
    assert plan.dyson_evaluations == 2404
    assert plan.contraction_work == 40768634880000
    assert plan.peak_owned_bytes >= plan.retained_green_bytes + plan.born_peak_bytes + plan.dyson_workspace_bytes
    assert plan.peak_owned_bytes >= plan.retained_green_bytes + plan.mapped_sigma_bytes + plan.reconstruction_workspace_bytes + plan.dyson_workspace_bytes
    with_input = plan_material_born_map(4,601,216,120,24,24,has_input_sigma=True)
    assert with_input.nominal_caller_bytes-plan.nominal_caller_bytes == plan.mapped_sigma_bytes
    assert plan.peak_owned_bytes < 16*1024**3


@pytest.mark.parametrize('g', [True, -1, np.nan, 1e308, 1e-300])
def test_bad_coupling_rejected(g):
    with pytest.raises((ValueError,TypeError), match='g'):
        evaluate_packed_born_map(*_fixture(1), g=g)


def test_input_causality_cannot_be_hidden_by_keldysh_identity():
    args = _fixture(1)
    sigma = _causal_input(args[0], args[1])
    bad = replace(sigma, retarded=sigma.retarded+0.1*np.eye(2))
    with pytest.raises(ValueError, match='causality'):
        evaluate_packed_born_map(*args, sigma_input=bad)


def test_wrong_contact_embedding_context_rejected():
    grid,vertex,dynamical,left,right = _fixture(1)
    bad = replace(left, full_ndof=6)
    with pytest.raises(ValueError, match='device/mesh/grid/side'):
        evaluate_packed_born_map(grid,vertex,dynamical,bad,right)


def test_nonarray_and_nonfinite_input_are_rejected_before_solving():
    grid,vertex,dynamical,left,right = _fixture(1)
    shape = (1,grid.n_points,2,2)
    bad = PackedMapSpectrum(np.zeros(shape).tolist(),np.zeros(shape),np.zeros(shape))
    with pytest.raises(TypeError, match='numpy.ndarray'):
        evaluate_packed_born_map(grid,vertex,dynamical,left,right,sigma_input=bad)
    dynamical[0,0,0] = np.inf
    with pytest.raises(ValueError, match='finite'):
        evaluate_packed_born_map(grid,vertex,dynamical,left,right)


def test_singular_failure_restores_blas(monkeypatch):
    before = [(v['prefix'],v['num_threads']) for v in threadpool_info()]
    def singular(*args, **kwargs):
        raise np.linalg.LinAlgError('forced test singularity')
    monkeypatch.setattr(np.linalg, 'inv', singular)
    with pytest.raises(ValueError, match='Singular full-device Dyson'):
        evaluate_packed_born_map(*_fixture(1))
    assert before == [(v['prefix'],v['num_threads']) for v in threadpool_info()]


def test_zero_coupling_has_zero_image_and_evaluated_harmonic_currents():
    actual = evaluate_packed_born_map(*_fixture(1),g=0,identity_atol=0)
    for name in ('retarded','lesser','greater'):
        assert np.count_nonzero(getattr(actual.map_sigma,name)) == 0
    assert actual.input_currents.collision_power == 0
    assert abs(actual.input_currents.contact_balance) < 1e-12
    assert actual.input_sigma_is_zero
