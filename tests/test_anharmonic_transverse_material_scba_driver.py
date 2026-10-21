"""Fresh driver wiring/stop/provenance checks; no silicon input reads."""

from __future__ import annotations

import copy
import json
import runpy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.transverse_material_map import PackedMapSpectrum
from negfpy.anharmonic.transverse_material_scba import (
    MaterialCurrentResidual,
    MaterialSCBAConfig,
    MaterialSCBARecord,
    MaterialSigmaResidual,
)
from negfpy.anharmonic.transverse_solver import TransverseCurrents


@pytest.fixture
def driver():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_transverse_material_scba.py'))


def profile(driver):
    return {'map': copy.deepcopy(driver['PHYSICS']),
            'restart': {'initial_mixing': 0.1, 'checkpoint': None, 'parent_root': '/unused',
                        'parent_result_sha256': 'a'*64, 'parent_profile_sha256': 'b'*64,
                        'parent_source_commit': 'c'*40, 'parent_source_tree': 'd'*40},
            'scba': asdict(MaterialSCBAConfig()), 'source_commit': 'e'*40, 'source_tree': 'f'*40,
            'driver_timeout_seconds': 82800, 'checkpoint_max_bytes': 4*1024**3,
            'parent_max_bytes': 8*1024**3}


@pytest.mark.parametrize('section,key,value', [
    ('map','g',0.04), ('map','eta_lead',0.00078125), ('map','n_positive',200),
    ('scba','max_evaluations',13), ('scba','mixing',0.5),
    ('restart','parent_result_sha256','unbound'),
    ('scba','max_seconds',82700),
])
def test_invalid_profile_stops_before_saved_field_read(driver, monkeypatch, tmp_path, section, key, value):
    run=driver['run'];p=profile(driver);p[section][key]=value
    monkeypatch.setitem(run.__globals__,'load_material_map_fields',
                        lambda *a,**k: pytest.fail('invalid configuration reached inputs'))
    with pytest.raises(ValueError):
        run(p,{},lambda *a:None,tmp_path/'checkpoints',profile_sha256='1'*64)
    assert not list(tmp_path.iterdir())


def tiny_fields():
    shape=(1,3,1,1)
    # Wiring fixtures, not claimed physical spectra; solver is replaced below.
    spectrum=PackedMapSpectrum(*(np.full(shape,complex(i),dtype=complex) for i in (1,2,3)))
    return SimpleNamespace(binding={'artifacts_freshly_authenticated':19},mapped_sigma=spectrum,
                           grid='grid',vertex='vertex',dynamical='D',left='L',right='R')


def record(converged=False):
    residual=MaterialSigmaResidual(1,2,0.1,10,1,2,0.5)
    current=MaterialCurrentResidual(0,1e-9,1e-12,0)
    return MaterialSCBARecord(1,0.01,0.02,{n:residual for n in ('retarded','lesser','greater')},
                              {n:current for n in ('contact_balance','collision_power','full_balance')},
                              False,True,converged)


def setup(driver, monkeypatch, *, fail=False):
    run=driver['run'];fields=tiny_fields();seen={}
    def load(*args,**kwargs):
        seen['load']=kwargs
        return fields
    monkeypatch.setitem(run.__globals__,'load_material_map_fields',load)
    def solve(*args,**kwargs):
        sigma=kwargs['sigma_initial'];seen['sigma']=sigma
        kwargs['on_start'](1,sigma)
        if fail:
            return SimpleNamespace(state=None,converged=False,stop_reason='evaluation_error',
                                   failure_reason='controlled map failure',history=(),plan={})
        evaluated=SimpleNamespace(map_sigma=fields.mapped_sigma,
                                  input_currents=TransverseCurrents(1e-9,-1e-9,0))
        state=SimpleNamespace(index=1,sigma_input=sigma,evaluation=evaluated,record=record())
        kwargs['on_evaluation'](state)
        return SimpleNamespace(state=state,converged=False,stop_reason='max_evaluations',
                               failure_reason=None,history=(state.record,),plan={})
    monkeypatch.setitem(run.__globals__,'run_material_scba',solve)
    return run,fields,seen


def test_first_candidate_checkpoint_currents_and_budget_stop_are_distinct(driver,monkeypatch,tmp_path):
    run,fields,seen=setup(driver,monkeypatch);p=profile(driver);result={};events=[]
    run(p,result,lambda *a:events.append(a),tmp_path/'checkpoints',profile_sha256='1'*64)
    assert result['passed'] and not result['converged']
    assert result['stop_reason']=='max_evaluations' and result['completed_evaluations']==1
    for name in ('retarded','lesser','greater'):
        np.testing.assert_array_equal(getattr(seen['sigma'],name),0.1*getattr(fields.mapped_sigma,name))
    assert seen['load']['expected_settings']==p['map']
    assert seen['load']['expected_source_commit']==p['restart']['parent_source_commit']
    saved=result['checkpoints'][0]
    manifest=json.loads((Path(saved['path'])/'manifest.json').read_text())
    assert manifest['record']['currents']['left']==1e-9
    assert manifest['sigma_input_evaluated'] is True
    assert manifest['mapped_sigma_dyson_evaluated'] is False
    assert manifest['accepted_restart'] is False
    assert manifest['binding']['profile_sha256']=='1'*64
    assert len(manifest['artifacts'])==6
    assert result['attempts'][0]['dyson_evaluated'] is False
    assert [e[0] for e in events]==['parent_authentication_start','parent_authenticated',
        'initial_candidate_ready','scba_evaluation_start','scba_evaluation_complete',
        'checkpoint_saved','scba_stopped']


def test_first_map_failure_retains_attempt_evidence_without_false_checkpoint(driver,monkeypatch,tmp_path):
    run,_,_=setup(driver,monkeypatch,fail=True);result={}
    run(profile(driver),result,lambda *a:None,tmp_path/'checkpoints',profile_sha256='1'*64)
    assert not result['passed'] and not result['converged']
    assert result['failure_reason']=='controlled map failure'
    assert len(result['attempts'])==1 and result['checkpoints']==[]
    assert not (tmp_path/'checkpoints').exists()


def test_checkpoint_failure_cannot_report_success(driver,monkeypatch,tmp_path):
    run,_,_=setup(driver,monkeypatch)
    def broken(*args,**kwargs):
        raise OSError('controlled full filesystem')
    monkeypatch.setitem(run.__globals__,'save_material_checkpoint',broken)
    result={}
    with pytest.raises(OSError,match='full filesystem'):
        run(profile(driver),result,lambda *a:None,tmp_path/'checkpoints',profile_sha256='1'*64)
    assert not result.get('passed',False) and result['checkpoints']==[]


def test_authenticated_checkpoint_resume_uses_saved_input_and_map(driver,monkeypatch,tmp_path):
    run,fields,seen=setup(driver,monkeypatch);p=profile(driver)
    binding={'physics':p['map'],'parent_result_sha256':'a'*64,'parent_profile_sha256':'b'*64}
    p['restart']['checkpoint']={'path':'/unused-resume','manifest_sha256':'2'*64,'binding':binding}
    old=PackedMapSpectrum(*(np.full((1,3,1,1),5+0j) for _ in range(3)))
    saved=SimpleNamespace(sigma_input=old,mapped_sigma=fields.mapped_sigma,
                          binding=binding,manifest_sha256='2'*64,record={'index':7,'converged':False})
    monkeypatch.setitem(run.__globals__,'load_material_checkpoint',lambda *a,**k:saved)
    result={}
    run(p,result,lambda *a:None,tmp_path/'checkpoints',profile_sha256='1'*64)
    np.testing.assert_array_equal(seen['sigma'].retarded,0.9*old.retarded+0.1*fields.mapped_sigma.retarded)
    assert result['checkpoints'][0]['manifest']['binding']['resume_checkpoint_sha256']=='2'*64
    assert result['checkpoints'][0]['manifest']['binding']['resume_evaluated_iteration']==7
    assert result['scope']['initial_state'].startswith('unevaluated mixed successor')
