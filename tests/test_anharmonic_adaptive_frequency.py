"""Independent full-matrix quadrature and bounded-partition negative controls."""
import dataclasses

import numpy as np
import pytest

import negfpy.anharmonic.adaptive_frequency as module
from negfpy.anharmonic.adaptive_frequency import (
    adaptive_matrix_integral, cubic_adaptive_lesser_greater,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.types import DenseCubicVertex


def test_complex_polynomial_complete_matrices_and_archived_error_reconstruction():
    base=np.arange(18).reshape(2,3,3)/17 + 1j*np.arange(18,36).reshape(2,3,3)/23
    result=adaptive_matrix_integral(lambda x:base*(1+x+x**4),(-1.,-.3,.7,1.),
                                    matrix_dimension=3,rtol=1e-10,atol=1e-13)
    np.testing.assert_allclose(result.high,base*2.4,rtol=2e-14,atol=1e-14)
    assert result.complete_partition and result.tolerance_met
    assert result.pair_evaluations==72 and result.green_evaluations==0
    np.testing.assert_allclose(sum(leaf.high for leaf in result.leaves),result.high,rtol=2e-14)
    errors=[abs(leaf.high-leaf.low) for leaf in result.leaves]
    np.testing.assert_allclose(sum(errors),result.summed_error_abs,atol=1e-30)
    np.testing.assert_allclose(sum(np.max(error,axis=(1,2)) for error in errors),result.max_errors,atol=1e-30)
    np.testing.assert_allclose(sum(np.linalg.norm(error,axis=(1,2)) for error in errors),result.frobenius_errors,atol=1e-30)
    assert np.all(result.max_error_ratios<=1) and np.all(result.frobenius_error_ratios<=1)
    assert not result.leaves[0].high.flags.writeable
    with pytest.raises(ValueError): result.leaves[0].high.setflags(write=True)


@pytest.mark.parametrize('component',[0,1])
def test_small_offdiagonal_narrow_lorentzian_drives_refinement(component):
    width=.004
    center=.17
    def evaluate(x):
        value=np.zeros((2,3,3),complex)
        value[:,0,0]=10
        value[component,1,2]=(.4+.3j)*width**2/(width*width+(x-center)**2)
        return value
    result=adaptive_matrix_integral(evaluate,(-1.,center,1.),matrix_dimension=3,
                                    rtol=2e-10,atol=1e-13,max_pair_evaluations=6000)
    expected=(.4+.3j)*width*(np.arctan((1-center)/width)-np.arctan((-1-center)/width))
    np.testing.assert_allclose(result.high[component,1,2],expected,rtol=2e-7)
    np.testing.assert_allclose(result.high[:,0,0],20,rtol=1e-14)
    assert result.pair_evaluations>48 and result.tolerance_met
    assert len(result.leaves)>2
    assert min(leaf.right-leaf.left for leaf in result.leaves)<.02


def test_frobenius_criterion_can_require_refinement_when_max_criterion_passes():
    def evaluate(x):
        value=np.full((2,4,4),x**16,dtype=complex)
        value[:,0,0]+=100
        return value
    initial=adaptive_matrix_integral(evaluate,(-1.,1.),matrix_dimension=4,
                                     rtol=0,atol=1e-15,max_pair_evaluations=24)
    threshold=float((initial.max_errors[0]+initial.frobenius_errors[0])/2)
    result=adaptive_matrix_integral(evaluate,(-1.,1.),matrix_dimension=4,
                                    rtol=0,atol=threshold,max_pair_evaluations=24)
    assert np.all(result.max_errors<=result.max_tolerances)
    assert np.all(result.frobenius_errors>result.frobenius_tolerances)
    assert result.complete_partition and not result.tolerance_met
    assert result.stop_reason=='evaluation-budget'


def test_opposite_local_error_cancellation_cannot_fake_acceptance():
    result=adaptive_matrix_integral(lambda x:np.full((2,1,1),x**17),(-1.,0.,1.),
                                    matrix_dimension=1,rtol=.1,atol=1e-12,max_pair_evaluations=48)
    assert np.max(abs(result.high-result.low))<1e-14
    assert np.min(result.max_errors)>1e-9
    assert result.complete_partition and not result.tolerance_met
    assert result.stop_reason=='evaluation-budget'


def test_zero_integral_absolute_floor_accepted_without_division_artifact():
    result=adaptive_matrix_integral(lambda x:np.zeros((2,2,2)),(-1.,1.),
                                    matrix_dimension=2,rtol=1e-5,atol=0)
    assert result.tolerance_met
    np.testing.assert_array_equal(result.max_error_ratios,0)
    np.testing.assert_array_equal(result.frobenius_error_ratios,0)


def _vertex():
    values=np.zeros((3,3,3))
    values[0,0,0]=.2
    values[2,2,2]=-.7
    values[0,0,2]=values[0,2,0]=values[2,0,0]=.4
    return DenseCubicVertex(values)


def _fields(x):
    base=np.array([[1+.4j,.3-.2j,.7+.3j],[.4j,2+.1j,.1-.1j],[-.3+.2j,.8-.2j,1.2]])
    return base*(1+.2*x), base.T*(.5-.1*x+.03*x*x)


@pytest.mark.parametrize('omega',[-1.1,0.,.7])
def test_cubic_dense_sparse_prefactor_complex_full_matrix_and_single_compile(monkeypatch,omega):
    vertex=_vertex()
    calls=[]
    original=module._compiled_sparse_schedule
    monkeypatch.setattr(module,'_compiled_sparse_schedule',lambda v:(calls.append(v),original(v))[1])
    dense=cubic_adaptive_lesser_greater(vertex,_fields,omega,(-1.4,-.2,1.3),
                                       cutoff=1.8,rtol=1e-10,max_workers=1)
    sparse=cubic_adaptive_lesser_greater(SparseCubicVertex.from_dense(vertex),_fields,
                                        omega,(-1.4,-.2,1.3),cutoff=1.8,rtol=1e-10,max_workers=4)
    assert len(calls)==1
    expected=np.zeros((2,3,3),complex)
    nodes,weights=np.polynomial.legendre.leggauss(20)
    for left,right in zip(dense.effective_breakpoints,dense.effective_breakpoints[1:]):
        for node,weight in zip(nodes,weights):
            nu=(left+right)/2+(right-left)/2*node
            first,second=_fields(nu),_fields(omega-nu)
            for component in (0,1):
                expected[component]+=(right-left)/2*weight*np.einsum('acd,bef,ce,df->ab',
                    vertex.values,vertex.values,first[component],second[component],optimize=False)
    expected*=.5j/(2*np.pi)
    for component,name in enumerate(('lesser','greater')):
        np.testing.assert_allclose(getattr(dense,name),expected[component],rtol=2e-13,atol=1e-14)
        np.testing.assert_allclose(getattr(sparse,name),expected[component],rtol=2e-13,atol=1e-14)
        np.testing.assert_array_equal(getattr(sparse,name)[1],0)
    assert dense.complete_partition and dense.tolerance_met and sparse.tolerance_met
    np.testing.assert_array_equal(sparse.active_indices,[0,2])


def test_serial_threaded_adaptive_integrals_leaves_and_accounting_are_identical():
    def evaluate(x):
        value=np.ones((2,2,2),complex)
        value[1,0,1]=(1+.2j)/(.003**2+(x-.13)**2)
        return value
    args=dict(matrix_dimension=2,rtol=1e-7,max_pair_evaluations=6000)
    serial=adaptive_matrix_integral(evaluate,(-1.,.13,1.),max_workers=1,**args)
    threaded=adaptive_matrix_integral(evaluate,(-1.,.13,1.),max_workers=8,**args)
    assert serial.pair_evaluations==threaded.pair_evaluations
    assert [leaf.bounds for leaf in serial.leaves]==[leaf.bounds for leaf in threaded.leaves]
    np.testing.assert_array_equal(serial.high,threaded.high)
    np.testing.assert_array_equal(serial.max_errors,threaded.max_errors)
    for first,second in zip(serial.leaves,threaded.leaves):
        np.testing.assert_array_equal(first.low,second.low)
        np.testing.assert_array_equal(first.high,second.high)


@pytest.mark.parametrize('omega',[-.3,0.,.4])
def test_support_clipping_and_zero_arguments_are_panel_endpoints(omega):
    frequencies=[]
    def evaluate(x):
        assert -1<=x<=1 and x!=0
        frequencies.append(x)
        return np.ones((1,1)),2*np.ones((1,1))
    result=cubic_adaptive_lesser_greater(DenseCubicVertex(np.ones((1,1,1))),evaluate,
                                        omega,(-2.,2.),cutoff=1)
    length=2-abs(omega)
    np.testing.assert_allclose(result.lesser,[[.5j/(2*np.pi)*length]])
    assert 0 in result.effective_breakpoints
    assert omega in result.effective_breakpoints
    assert result.green_evaluations==len(frequencies)


@pytest.mark.parametrize('omega',[-4.,-2.,2.,4.])
def test_empty_support_never_evaluates_green(omega):
    result=cubic_adaptive_lesser_greater(_vertex(),lambda x:pytest.fail('no support'),
                                        omega,(-3.,3.),cutoff=1)
    assert result.complete_partition and result.tolerance_met
    assert result.stop_reason=='no-support' and result.pair_evaluations==0
    assert not result.leaves
    np.testing.assert_array_equal(result.lesser,0)


def test_zero_sparse_vertex_keeps_zero_domain_and_panel_guard():
    vertex=SparseCubicVertex.from_dense(DenseCubicVertex(np.zeros((2,2,2))))
    result=cubic_adaptive_lesser_greater(vertex,lambda x:pytest.fail('zero vertex'),
                                        0,(-1.,1.),cutoff=2)
    assert result.stop_reason=='zero-vertex' and result.tolerance_met
    assert len(result.leaves)==2 and result.pair_evaluations==0
    assert result.leaves[0].high.shape==(2,0,0)
    with pytest.raises(ValueError,match='max_panels'):
        cubic_adaptive_lesser_greater(vertex,lambda x:None,0,(-1.,1.),cutoff=2,max_panels=1)


def test_incomplete_initial_partition_never_claims_window_complete():
    result=adaptive_matrix_integral(lambda x:np.ones((2,1,1)),(-1.,0.,1.),
                                    matrix_dimension=1,max_pair_evaluations=24)
    assert len(result.leaves)==1 and result.leaves[0].bounds==(-1.,0.)
    assert not result.complete_partition and not result.tolerance_met
    assert result.stop_reason=='evaluation-budget'
    np.testing.assert_allclose(result.high,1)


def test_interrupted_second_child_retains_full_parent_and_discards_both_children(monkeypatch):
    calls=[0]
    def evaluate(x):
        calls[0]+=1
        return np.full((2,1,1),x**16)
    # Initial24, firstchild24, four secondchild samples then wall expiry.
    monkeypatch.setattr(module.time,'monotonic',lambda:0. if calls[0]<52 else 2.)
    result=adaptive_matrix_integral(evaluate,(-1.,1.),matrix_dimension=1,
                                    rtol=1e-12,atol=1e-15,max_workers=1,max_seconds=1)
    nodes,weights=np.polynomial.legendre.leggauss(16)
    expected=np.dot(weights,nodes**16)
    assert result.complete_partition and not result.tolerance_met
    assert result.stop_reason=='wall-budget' and result.pair_evaluations==52
    assert len(result.leaves)==1 and result.leaves[0].bounds==(-1.,1.)
    np.testing.assert_allclose(result.high,expected,rtol=1e-14)


def test_panel_budget_keeps_complete_unconverged_partition():
    result=adaptive_matrix_integral(lambda x:np.full((2,1,1),x**16),(-1.,1.),
                                    matrix_dimension=1,rtol=1e-12,atol=1e-15,max_panels=1)
    assert result.complete_partition and not result.tolerance_met
    assert result.stop_reason=='panel-budget' and result.pair_evaluations==24


@pytest.mark.parametrize('kwargs',[
    dict(rtol=-1),dict(atol=float('nan')),dict(rtol=0,atol=0),dict(max_seconds=0),
    dict(max_workers=9),dict(max_workers=True),dict(max_pair_evaluations=0),dict(max_panels=0),
])
def test_bad_controls_rejected_before_callback(kwargs):
    with pytest.raises(ValueError):
        adaptive_matrix_integral(lambda x:pytest.fail('bad controls'),(-1.,1.),matrix_dimension=1,**kwargs)


@pytest.mark.parametrize('breakpoints',[(1.,-1.),(0.,0.),(0.,float('inf')),(0.,)])
def test_invalid_breakpoints_rejected(breakpoints):
    with pytest.raises(ValueError):
        adaptive_matrix_integral(lambda x:None,breakpoints,matrix_dimension=1)


@pytest.mark.parametrize('value',[np.ones((1,1)),np.ones((2,1,2)),np.full((2,1,1),float('nan'))])
def test_invalid_integrand_matrix_rejected(value):
    with pytest.raises(ValueError,match='finite matrices'):
        adaptive_matrix_integral(lambda x:value,(-1.,1.),matrix_dimension=1)


def test_serial_scratch_reuse_is_copied_and_progress_final_counts():
    scratch=np.zeros((2,2,2),complex)
    def evaluate(x):
        scratch[:]=1+x*x
        return scratch
    records=[]
    result=adaptive_matrix_integral(evaluate,(-1.,1.),matrix_dimension=2,progress=records.append)
    np.testing.assert_allclose(result.high,8/3,rtol=1e-14)
    assert records[-1]['final'] and records[-1]['pair_evaluations']==24
