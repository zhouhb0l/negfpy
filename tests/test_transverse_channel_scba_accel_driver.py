"""Tiny new acceleration-driver contracts; no material calculation or old replay."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSpectrum,
    mix_channel_spectra,
)
from negfpy.anharmonic.transverse_channel_restart import save_channel_checkpoint
from negfpy.anharmonic.transverse_solver import TransverseCurrents

PACKAGE = (Path(__file__).resolve().parents[3]
           / 'reports/phonon-phonon-interaction/multiq-channel-anderson1-20261023')


def driver():
    spec = importlib.util.spec_from_file_location(
        'native_channel_accel_driver',
        Path(__file__).parents[1]/'scripts/run_transverse_channel_scba_accel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(mode='anderson1'):
    value = json.loads((PACKAGE/'control/profile-draft.json').read_text())
    value['source_commit'], value['source_tree'] = '1'*40, '2'*40
    value['acceleration']['mode'] = mode
    return value


POLICY = {'mode': 'bounded-tradeoff', 'component_growth_limit': 1.02, 'merit_ratio_limit': .99}


def spectrum(factor=1.):
    return ChannelSpectrum(*(np.full((1, 3, 1, 1), factor*x, dtype=np.complex128)
                             for x in (1., -1e12j, 1e-12j)))


def record(index):
    residual = {'error': 1., 'scale': 1., 'allowance': .01, 'ratio': 100.,
                'error_frobenius': 1., 'scale_frobenius': 1., 'relative_frobenius': 1.,
                'error_location': [0, 1, 0, 0], 'scale_location': [0, 1, 0, 0]}
    current = {'error_w': 0., 'scale_w': 2e-9, 'allowance_w': 2.01e-12, 'ratio': 0.}
    return {'index': index, 'map_seconds': .1, 'elapsed_seconds': .2,
            'sigma_residuals': {key: dict(residual) for key in ('retarded', 'lesser', 'greater')},
            'channel_residuals': {key: dict(residual) for key in ('statistical', 'spectral')},
            'current_residuals': {key: dict(current) for key in
                                  ('contact_balance', 'collision_power', 'full_balance')},
            'sigma_passed': False, 'current_passed': True, 'converged': False}


def state(index, supplied):
    return SimpleNamespace(index=index, sigma_input=supplied, record=record(index),
        evaluation=SimpleNamespace(map_sigma=spectrum(2.),
            input_currents=TransverseCurrents(2e-9, -2e-9, 0.),
            checks={'fixture': {}}, zero_frequency_context={'available': False}))


def decision(index, selected, accepted=True, mode='anderson1'):
    return {'index': index, 'accepted_as_anchor': accepted, 'selected_anchor_index': selected,
            'kind': 'linear_warmup' if index == 16 else mode,
            'reference_anchor_index': max(15, index-1), 'reason': 'tiny fixture',
            'reference_merit': 1., 'trial_merit': .5, 'merit_scales': [1., 1., 1.],
            'unmixed_frobenius_nonincrease': {key: True for key in ('retarded', 'lesser', 'greater')},
            'current_passed': True, 'fully_converged': False}


def wiring(module, monkeypatch, mode='anderson1'):
    pf, supplied, mapped = profile(mode), spectrum(), spectrum(4.)
    fields = SimpleNamespace(grid='grid', vertex='vertex', dynamical='dynamical',
                             left='left', right='right', binding={'static19': True})
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(module, 'authenticate_resume', lambda resume: (
        SimpleNamespace(sigma_input=supplied, mapped_sigma=mapped, record=record(15)),
        {'path': resume['path'], 'manifest_sha256': resume['manifest_sha256'],
         'iteration': 15, 'binding': copy.deepcopy(resume['binding']),
         'artifacts_freshly_authenticated': 6, 'artifacts': {}, 'accepted_restart': False}))
    return pf, supplied, mapped


@pytest.mark.parametrize('mode', ['linear-control', 'anderson1'])
def test_both_modes_pin_seed15_with_exact_lineage_and_original_six_convergence_screens(mode):
    module, pf = driver(), profile(mode)
    config, acceleration = module.validate_profile(pf)
    assert acceleration.mode == mode and config.mixing == .05 and config.max_evaluations == 6
    assert (config.sigma_rtol, config.sigma_atol) == (1e-4, 1e-10)
    assert (config.current_rtol, config.current_atol_w) == (1e-3, 1e-14)
    assert (config.identity_rtol, config.identity_atol) == (1e-8, 1e-11)
    assert pf['resume']['binding'] == module.seed_binding(pf)
    assert pf['resume']['binding']['resume_parent_iteration'] == 9
    assert pf['resume']['binding']['resume_parent_binding']['resume_parent_iteration'] == 3
    assert pf['resume']['binding']['source_commit'] == module.SEED_BINDING['source_commit']
    assert pf['source_commit'] != pf['resume']['binding']['source_commit']


def test_opt_in_policy_is_explicitly_bound_without_changing_proposals_physics_or_screens():
    module, pf = driver(), profile()
    original_config, original_acceleration = module.validate_profile(pf)
    old_binding = module.checkpoint_binding(pf, 'a'*64)
    assert 'selection_policy' not in old_binding
    assert module.validate_selection_policy(pf).mode == 'component-nonincreasing'
    pf['selection_policy'] = copy.deepcopy(POLICY)
    assert module.validate_profile(pf) == (original_config, original_acceleration)
    policy = module.validate_selection_policy(pf)
    assert (policy.mode, policy.component_growth_limit, policy.merit_ratio_limit) == ('bounded-tradeoff', 1.02, .99)
    binding = module.checkpoint_binding(pf, 'a'*64)
    assert binding.pop('selection_policy') == POLICY and binding == old_binding


@pytest.mark.parametrize('change', [
    {'mode': 'always-accept'}, {'component_growth_limit': 1.03},
    {'merit_ratio_limit': 1.}, {'mode': 'component-nonincreasing'},
])
def test_new_policy_profile_limits_are_pinned_before_field_loading(change):
    module, pf = driver(), profile()
    pf['selection_policy'] = dict(POLICY, **change)
    with pytest.raises(ValueError):
        module.validate_profile(pf)


def test_bounded_policy_cannot_be_applied_to_linear_control():
    module, pf = driver(), profile('linear-control')
    pf['selection_policy'] = copy.deepcopy(POLICY)
    with pytest.raises(ValueError):
        module.validate_profile(pf)


@pytest.mark.parametrize('defect', ['seed', 'manifest', 'lineage', 'alpha', 'tolerance',
                                  'mode', 'coefficient_bound', 'first_index'])
def test_unapproved_experiment_rejected_before_static_or_resume_loading(
        tmp_path, monkeypatch, defect):
    module, pf = driver(), profile()
    if defect == 'seed':
        pf['resume']['iteration'] = 9
    elif defect == 'manifest':
        pf['resume']['manifest_sha256'] = module.STATE9_MANIFEST
    elif defect == 'lineage':
        del pf['resume']['binding']['resume_parent_binding']
    elif defect == 'alpha':
        pf['scba']['mixing'] = .1
    elif defect == 'tolerance':
        pf['scba']['sigma_rtol'] = .01
    elif defect == 'mode':
        pf['acceleration']['mode'] = 'always-accept'
    elif defect == 'coefficient_bound':
        pf['acceleration']['coefficient_limit'] = 10.
    else:
        pf['first_index'] = 15
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No static load'))
    monkeypatch.setattr(module, 'authenticate_resume', lambda *a, **kw: pytest.fail('No resume load'))
    with pytest.raises(ValueError):
        module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


@pytest.mark.parametrize('mode,selection', [('linear-control', None), ('anderson1', None), ('anderson1', POLICY)])
def test_candidate16_once_all_trials_saved_and_selected_anchor_distinct_from_last_trial(
        tmp_path, monkeypatch, mode, selection):
    module = driver()
    pf, supplied15, mapped15 = wiring(module, monkeypatch, mode)
    if selection is not None:
        pf['selection_policy'] = copy.deepcopy(selection)
        pf['selection_experiment'] = {'comparison_baselines': {'metadata_only': True},
                                      'baselines_used_computationally': False}
    original = {(label, name): getattr(value, name).tobytes()
                for label, value in [('input', supplied15), ('map', mapped15)]
                for name in module.COMPONENTS}
    expected = mix_channel_spectra(supplied15, mapped15, .05)
    expected_hashes = {name: hashlib.sha256(getattr(expected, name).tobytes()).hexdigest()
                       for name in module.COMPONENTS}
    mix_count = []
    def mix_once(*args):
        mix_count.append(args)
        return mix_channel_spectra(*args)
    monkeypatch.setattr(module, 'mix_channel_spectra', mix_once)
    def solve(*args, **kwargs):
        assert args == ('grid', 'vertex', 'dynamical', 'left', 'right')
        assert kwargs['seed_index'] == 15 and kwargs['config'].max_evaluations == 6
        assert kwargs['seed_input'] is supplied15 and kwargs['seed_mapped'] is mapped15
        assert kwargs['acceleration'].mode == mode and kwargs['g'] == .03
        assert kwargs['selection'].mode == ('component-nonincreasing' if selection is None else 'bounded-tradeoff')
        supplied = kwargs['sigma_initial']
        history, decisions, states = [], [], []
        for index in range(16, 22):
            kwargs['on_start'](index, supplied, {'kind': mode, 'anchor_index': index-1,
                                                'previous_index': index-2})
            evaluated = state(index, supplied)
            accepted = mode == 'linear-control' or index != 21
            selected = index if accepted else 20
            choice = decision(index, selected, accepted, mode)
            kwargs['on_evaluation'](evaluated, choice)
            history.append(evaluated.record)
            decisions.append(choice)
            states.append(evaluated)
        selected_state = states[-1] if mode == 'linear-control' else states[-2]
        return SimpleNamespace(state=selected_state, last_evaluated_state=states[-1],
                               history=history, decisions=decisions, converged=False,
                               stop_reason='max_evaluations', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result, events = {}, []
    module.run(pf, result, lambda name, details=None: events.append(name), tmp_path/'checkpoints',
               profile_sha256='a'*64)
    assert len(mix_count) == 1 and mix_count[0] == (supplied15, mapped15, .05)
    assert result['initial_candidate']['sigma_sha256'] == expected_hashes
    assert result['attempts'][0]['sigma_sha256'] == result['evaluated_records'][0]['sigma_input_sha256'] == expected_hashes
    assert result['initial_candidate']['construction'] == '.95*evaluated_input15+.05*unevaluated_image15'
    assert result['passed'] and not result['converged'] and result['completed_evaluations'] == 6
    if selection is not None:
        assert result['selection_policy_config'] == selection
        assert result['selection_experiment'] == pf['selection_experiment']
    assert result['scope']['comparison_baselines_used_computationally'] is False
    selected = 21 if mode == 'linear-control' else 20
    assert result['selected_anchor_index'] == selected
    assert result['selected_anchor']['index'] == selected
    assert result['final_evaluated_state']['index'] == selected
    assert result['last_evaluated_trial']['index'] == 21
    assert result['history'][-1]['index'] == 21
    assert len(result['checkpoints']) == len(result['decisions']) == 6
    for saved, rec, choice in zip(result['checkpoints'], result['evaluated_records'], result['decisions'], strict=True):
        manifest, binding = saved['manifest'], saved['manifest']['binding']
        assert manifest['record'] == rec and rec['controller_decision'] == choice
        assert binding == module.checkpoint_binding(pf, 'a'*64)
        assert binding['resume_parent_iteration'] == 15
        assert binding['resume_parent_binding'] == pf['resume']['binding']
        assert binding['resume_parent_binding']['resume_parent_binding']['resume_parent_iteration'] == 3
        assert binding['acceleration'] == pf['acceleration']
        assert manifest['accepted_restart'] is False and manifest['mapped_sigma_dyson_evaluated'] is False
    assert result['checkpoints'][-1]['manifest']['record']['controller_decision']['accepted_as_anchor'] == (mode == 'linear-control')
    assert events == ['parent_authentication_start', 'parent_authenticated',
                      'resume_authentication_start', 'resume_authenticated', 'channel_accel_start']+[
                          name for _ in range(6) for name in
                          ('scba_evaluation_start', 'scba_evaluation_complete', 'checkpoint_saved', 'controller_decision')]+['scba_stopped']
    assert {(label, name): getattr(value, name).tobytes()
            for label, value in [('input', supplied15), ('map', mapped15)]
            for name in module.COMPONENTS} == original
    assert not result['scope']['accepted_restart'] and not result['scope']['accepted_field_update']


def test_first_candidate_rawhash_mismatch_stops_before_any_fresh_trial(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        kwargs['on_start'](16, spectrum(99.), {'kind': 'linear_warmup'})
        pytest.fail('Changed first candidate cannot be evaluated')
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result = {}
    with pytest.raises(ValueError, match='Candidate16 differs'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == result['checkpoints'] == []


@pytest.mark.parametrize('bad_index', [15, 22])
def test_old_or_seventh_trial_cannot_start(tmp_path, monkeypatch, bad_index):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        kwargs['on_start'](bad_index, kwargs['sigma_initial'], {'kind': 'linear_warmup'})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    with pytest.raises(ValueError, match='unexpected'):
        module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


def test_actual_failed_second_map_preserves_first_evaluated_anchor_and_unmatched_start(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        supplied = kwargs['sigma_initial']
        kwargs['on_start'](16, supplied, {'kind': 'linear_warmup', 'anchor_index': 15})
        completed, choice = state(16, supplied), decision(16, 16)
        kwargs['on_evaluation'](completed, choice)
        kwargs['on_start'](17, supplied, {'kind': 'anderson1', 'anchor_index': 16})
        return SimpleNamespace(state=completed, last_evaluated_state=completed,
                               history=[completed.record], decisions=[choice], converged=False,
                               stop_reason='evaluation_error', failure_reason='actual map failed', plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result = {}
    module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert not result['passed'] and result['failure_reason'] == 'actual map failed'
    assert [row['index'] for row in result['attempts']] == [16, 17]
    assert result['selected_anchor_index'] == result['last_evaluated_trial']['index'] == 16
    assert len(result['checkpoints']) == result['completed_evaluations'] == 1


def test_native15_authentication_records_six_stable_files_without_relabelling(tmp_path):
    module, pf = driver(), profile()
    rec = record(15)
    rec.update(currents={'left': 2e-9, 'right': -2e-9, 'collision_power': 0.},
               input_sigma_is_zero=False, sigma_input_evaluated=True,
               mapped_sigma_dyson_evaluated=False, status='evaluated')
    saved = save_channel_checkpoint(tmp_path/'seed', iteration=15, sigma_input=spectrum(),
                                    mapped_sigma=spectrum(4.), record=rec,
                                    binding=pf['resume']['binding'])
    resume = dict(pf['resume'], path=saved['path'], manifest_sha256=saved['manifest_sha256'])
    loaded, binding = module.authenticate_resume(resume)
    assert binding['iteration'] == loaded.record['index'] == 15
    assert binding['binding'] == pf['resume']['binding']
    assert binding['accepted_restart'] is False and binding['artifacts_freshly_authenticated'] == 6
    assert all(row['stat_before'] == row['stat_after'] for row in binding['artifacts'].values())
    with pytest.raises(ValueError, match='must be pinned'):
        module.validate_profile(dict(pf, resume=resume))
