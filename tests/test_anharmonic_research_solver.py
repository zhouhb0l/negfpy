"""Research routing and bounded controller-retained input regressions."""
import numpy as np
import pytest
import negfpy.anharmonic.solver as solver
import negfpy.anharmonic.diagnostics as diagnostics
from negfpy.anharmonic import CubicSCBAProblem, HarmonicReservoir, SCBAConfig, SignedFrequencyGrid
from negfpy.anharmonic.toy import pinned_scalar_chain
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex


def problem(cubic=.005):
    device,lead,vertex=pinned_scalar_chain(1,pinning=1.,cubic=cubic)
    return CubicSCBAProblem(device=device,left=HarmonicReservoir(lead,.8),
        right=HarmonicReservoir(lead,.8),vertex=SparseCubicVertex.from_dense(vertex),
        grid=SignedFrequencyGrid(step=.1,n_positive=30))


def config(**kw):
    return SCBAConfig(**dict(dict(eta_lead=.02,max_iterations=80,min_iterations=3,
        consecutive_passes=3,mixing=.3,sigma_rtol=1e-7,green_rtol=1e-7,
        identity_rtol=1e-7,current_rtol=1e-4,current_atol=1e-9,
        on_nonconvergence='return',mixing_method='anderson',
        anderson_policy='scaled-trust-v1'),**kw))


class Driver:
    identity={'kind':'test-map','version':1}
    def __init__(self,maximum=16,*,incomplete_at=None,final_stop=False,refined=False,rejected_at=None):
        self.maximum,self.incomplete_at,self.final_stop=maximum,incomplete_at,final_stop
        self.refined,self.rejected_at=refined,rejected_at
        self.started=0;self.calls=[];self.commits=[];self.retained_evaluation=None
    def evaluate(self,p,left,right,lesser,greater,cfg,*,purpose):
        if purpose=='final_diagnostics' and self.final_stop:
            raise solver.ResearchMapBudgetExhausted('wall_budget')
        if self.started>=self.maximum-(purpose=='iteration'):
            raise solver.ResearchMapBudgetExhausted('map_budget')
        self.started+=1
        if self.started==self.incomplete_at:
            raise solver.ResearchMapBudgetExhausted('incomplete_window')
        if self.refined:
            from negfpy.anharmonic.refined_map import refined_fixed_point_evaluation,ResearchRefinementSpec
            from negfpy.anharmonic.frequency_refinement import GaussLegendreWindowPlan
            grid=p.grid;cell=grid.zero_index+5
            plan=GaussLegendreWindowPlan((grid.values[cell]-.05,grid.values[cell]+.05),order=16)
            v=refined_fixed_point_evaluation(p,left,right,lesser,greater,cfg,
                (ResearchRefinementSpec(grid.zero_index,(cell,),(plan,)),))
            assert v.completed
            out=v.green,v.sigma,v.raw_lesser,v.raw_greater,v.residual
        else:
            out=solver._fixed_point_evaluation(p,left,right,lesser,greater,cfg)
        if self.started==self.rejected_at: out=(*out[:-1],10.)
        self.calls.append((purpose,out));return out
    def metadata(self):
        return {'started':self.started,'completed':len(self.calls),'maximum':self.maximum}
    def committed(self,evaluation,record,event):
        self.retained_evaluation=evaluation;self.commits.append((evaluation,record,event))


def no_fft(*args,**kwargs):
    raise AssertionError('diagnostics switched to the production bubble')


def test_default_unchanged_and_final_uses_same_map(monkeypatch):
    p,cfg=problem(),config();baseline=solver.solve_scba(p,cfg);driver=Driver(80)
    monkeypatch.setattr(diagnostics,'cubic_lesser_greater',no_fft)
    result=solver.solve_scba(p,cfg,research_map=driver)
    assert baseline.converged and result.converged
    assert baseline.history==result.history
    np.testing.assert_array_equal(baseline.sigma_pp.lesser,result.sigma_pp.lesser)
    assert driver.started==len(result.history)+1
    assert driver.calls[-1][0]=='final_diagnostics'
    assert result.metadata['research_final_check_completed']
    assert result.diagnostics.passed and not result.release_certified
    assert not result.diagnostics.release_certified
    assert driver.commits[-1][2]['step']=='final_diagnostics'
    with pytest.raises(ValueError,match='explicit research raw'):
        diagnostics.check_solution(p,result,cfg)


def test_real_refined_map_drives_final_raw_residual(monkeypatch):
    p,cfg=problem(),config();driver=Driver(3,refined=True)
    monkeypatch.setattr(diagnostics,'cubic_lesser_greater',no_fft)
    result=solver.solve_scba(p,cfg,research_map=driver);final=driver.calls[-1][1]
    assert driver.started==3 and result.metadata['research_final_check_completed']
    np.testing.assert_array_equal(result.sigma_pp.lesser,final[1].lesser)
    report=diagnostics.check_solution(p,result,cfg,research_raw_pair=(final[2],final[3]))
    assert report.metrics==result.diagnostics.metrics
    coarse=solver._fixed_point_evaluation(p,result.sigma_left,result.sigma_right,
        result.sigma_pp.lesser,result.sigma_pp.greater,cfg)
    assert abs(final[2][p.grid.zero_index,0,0]-coarse[2][p.grid.zero_index,0,0])>1e-15


def test_map_cap_reserves_final_and_retains_evaluated_input(monkeypatch):
    p,cfg=problem(),config();driver=Driver(3)
    monkeypatch.setattr(diagnostics,'cubic_lesser_greater',no_fft)
    result=solver.solve_scba(p,cfg,research_map=driver)
    assert not result.converged and len(result.history)==2 and driver.started==3
    assert result.metadata['research_stop_reason']=='map_budget'
    assert result.metadata['research_final_check_completed']
    for value in (result.sigma_pp.lesser,driver.calls[-1][1][1].lesser):
        np.testing.assert_array_equal(value,driver.calls[1][1][1].lesser)
    assert result.history[-1].sigma_residual==driver.calls[1][1][-1]


def test_rejected_trial_counts_and_final_checks_restored_reference(monkeypatch):
    p,cfg=problem(),config();driver=Driver(4,rejected_at=3)
    monkeypatch.setattr(diagnostics,'cubic_lesser_greater',no_fft)
    result=solver.solve_scba(p,cfg,research_map=driver);reference=driver.calls[1][1]
    assert driver.started==4 and len(result.history)==3
    assert result.metadata['anderson']['rejected']==1 and not result.history[-1].passed
    assert driver.commits[2][0] is reference
    assert driver.commits[2][2]['step']=='anderson_rejected'
    for value in (result.sigma_pp.lesser,driver.calls[-1][1][1].lesser):
        np.testing.assert_array_equal(value,reference[1].lesser)
    assert result.history[-1].sigma_residual==reference[-1]


@pytest.mark.parametrize('final_stop',[False,True])
def test_incomplete_trial_keeps_committed_history(final_stop):
    driver=Driver(6,incomplete_at=3,final_stop=final_stop)
    result=solver.solve_scba(problem(),config(),research_map=driver)
    assert len(result.history)==2 and not result.converged
    assert result.metadata['research_stop_reason']=='incomplete_window'
    assert result.metadata['research_final_check_completed'] is (not final_stop)
    np.testing.assert_array_equal(result.sigma_pp.lesser,driver.calls[1][1][1].lesser)
    assert result.history[-1].sigma_residual==driver.calls[1][1][-1]
    assert result.metadata['anderson']['map_evaluations']==2
    assert driver.started==(3 if final_stop else 4)
    if final_stop:
        assert result.metadata['research_final_stop_reason']=='wall_budget'
        assert not result.diagnostics.passed


def test_final_budget_stop_forbids_otherwise_converged_claim(monkeypatch):
    p,cfg=problem(cubic=0),config();driver=Driver(final_stop=True)
    monkeypatch.setattr(diagnostics,'cubic_lesser_greater',no_fft)
    result=solver.solve_scba(p,cfg,research_map=driver)
    assert result.history[-1].consecutive_passes==cfg.consecutive_passes
    assert not result.converged and not result.diagnostics.passed
    assert result.metadata['research_final_check_completed'] is False
    assert result.metadata['research_final_stop_reason']=='wall_budget'


def test_stop_before_first_complete_map_raises_without_fake_result():
    with pytest.raises(solver.ResearchMapBudgetExhausted,match='incomplete_window'):
        solver.solve_scba(problem(),config(),research_map=Driver(incomplete_at=1))


def test_checkpoint_context_rejected_before_begin(monkeypatch):
    import negfpy.anharmonic._checkpoint_v2 as checkpoint
    class Session:
        def begin(self,*args): pytest.fail('research reached checkpoint begin')
    monkeypatch.setattr(checkpoint,'_active_solver_checkpoint_session',lambda:Session())
    with pytest.raises(ValueError,match='resumable checkpoints'):
        solver.solve_scba(problem(),config(),research_map=Driver())


def test_numerical_failure_is_not_reported_as_budget_stop():
    driver=Driver()
    def fail(*args,**kwargs): raise np.linalg.LinAlgError('real numerical failure')
    driver.evaluate=fail
    with pytest.raises(np.linalg.LinAlgError,match='real numerical failure'):
        solver.solve_scba(problem(),config(),research_map=driver)


def test_missing_identity_and_incomplete_return_rejected():
    driver=Driver();driver.identity={}
    with pytest.raises(ValueError,match='nonempty JSON'):
        solver.solve_scba(problem(),config(),research_map=driver)
    driver=Driver();driver.evaluate=lambda *a,**k:None
    with pytest.raises(TypeError,match='complete five-field'):
        solver.solve_scba(problem(),config(),research_map=driver)
