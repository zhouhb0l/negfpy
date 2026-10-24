"""Native25 continuation contracts using tiny synthetic fields only."""
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
           / 'reports/phonon-phonon-interaction/multiq-channel-feasibility-20261023')
POLICY = {'mode': 'bounded-tradeoff', 'component_growth_limit': 1.02, 'merit_ratio_limit': .99}
FEASIBILITY = {'mode': 'psd-backtracking', 'factor': .5, 'max_halvings': 8}


def driver():
    spec = importlib.util.spec_from_file_location(
        'native_channel_matched25_driver',
        Path(__file__).parents[1]/'scripts/run_transverse_channel_scba_matched25.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile():
    value = json.loads((PACKAGE/'feasible/profile.json').read_text())
    prior = json.loads((PACKAGE/'feasible/terminal/runtime/channel-scba-accel-result.json').read_text())
    saved = prior['checkpoints'][-1]
    value.update(schema='negfpy.transverse-channel-scba-matched25.profile.v1', first_index=26,
                 source_commit='1'*40, source_tree='2'*40, proposal_feasibility=copy.deepcopy(FEASIBILITY))
    value['scba']['max_evaluations'] = 12
    value['matched_experiment'] = dict(driver().MATCHED_INVARIANTS, purpose='matched longer continuation at equal fresh-map budget')
    value['resume'].update(iteration=25, manifest_sha256=saved['manifest_sha256'],
                           path=saved['path'], binding=copy.deepcopy(saved['manifest']['binding']))
    return value


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


def choice(index, selected, *, accepted=True):
    return {'index': index, 'accepted_as_anchor': accepted, 'selected_anchor_index': selected,
            'kind': 'linear_warmup' if index == 26 else 'anderson1',
            'reference_anchor_index': max(25, index-1), 'reason': 'tiny fixture',
            'current_passed': True, 'fully_converged': False}


def screen(index, *, passed=True, map_admitted=True):
    return {'index': index, 'anchor_index': max(25, index-1), 'mode': 'psd-backtracking',
            'original_kind': 'linear_warmup' if index == 26 else 'anderson1',
            'selected_kind': 'linear_warmup' if index == 26 else 'anderson1',
            'screens': [{'factor': 1., 'kind': 'linear_warmup' if index == 26 else 'anderson1',
                         'checks': {}, 'export_checks': {}, 'failed_keys': [], 'passed': passed,
                         'elapsed_seconds': .01, 'error': None}],
            'chosen_factor': 1. if passed else None, 'fallback_used': False, 'passed': passed,
            'stop_reason': None if passed else 'hard_failure',
            'error': None if passed else 'input field nonfinite', 'screen_count': 1,
            'maximum_screen_count': 10, 'elapsed_seconds': .01, 'map_admitted': map_admitted}


def wiring(module, monkeypatch):
    pf, supplied, mapped = profile(), spectrum(), spectrum(4.)
    fields = SimpleNamespace(grid='grid', vertex='vertex', dynamical='dynamical',
                             left='left', right='right', binding={'static19': True})
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(module, 'authenticate_resume', lambda resume: (
        SimpleNamespace(sigma_input=supplied, mapped_sigma=mapped, record=record(25)),
        {'path': resume['path'], 'manifest_sha256': resume['manifest_sha256'],
         'iteration': 25, 'binding': copy.deepcopy(resume['binding']),
         'artifacts_freshly_authenticated': 6, 'artifacts': {}, 'accepted_restart': False}))
    return pf, supplied, mapped


def test_native25_seed_binding_exactly_preserves_selected25_to19_to15_to9_to3_and_screens():
    module, pf = driver(), profile()
    config, acceleration = module.validate_profile(pf)
    assert acceleration.mode == 'anderson1' and config.mixing == .05 and config.max_evaluations == 12
    assert (config.sigma_rtol, config.sigma_atol) == (1e-4, 1e-10)
    assert (config.current_rtol, config.current_atol_w) == (1e-3, 1e-14)
    assert (config.identity_rtol, config.identity_atol) == (1e-8, 1e-11)
    assert module.seed_binding(pf) == pf['resume']['binding']
    binding = module.checkpoint_binding(pf, 'a'*64)
    assert binding['resume_parent_iteration'] == 25
    chain = binding
    for expected in (25, 19, 15, 9, 3):
        assert chain['resume_parent_iteration'] == expected
        chain = chain.get('resume_parent_binding', {})
    assert binding['matched_experiment'] == pf['matched_experiment']
    assert binding['proposal_feasibility'] == FEASIBILITY
    assert binding['selection_policy'] == POLICY


@pytest.mark.parametrize('defect', ['seed', 'manifest', 'lineage', 'alpha', 'tolerance',
                                  'first_index', 'factor', 'halvings', 'feasibility_mode', 'selection'])
def test_changed_experiment_bounds_rejected_before_any_field_loading(tmp_path, monkeypatch, defect):
    module, pf = driver(), profile()
    if defect == 'seed': pf['resume']['iteration'] = 15
    elif defect == 'manifest': pf['resume']['manifest_sha256'] = module.STATE15_MANIFEST
    elif defect == 'lineage': del pf['resume']['binding']['resume_parent_binding']
    elif defect == 'alpha': pf['scba']['mixing'] = .1
    elif defect == 'tolerance': pf['scba']['sigma_rtol'] = .01
    elif defect == 'first_index': pf['first_index'] = 25
    elif defect == 'factor': pf['proposal_feasibility']['factor'] = .25
    elif defect == 'halvings': pf['proposal_feasibility']['max_halvings'] = 9
    elif defect == 'feasibility_mode': pf['proposal_feasibility']['mode'] = 'disabled'
    else: pf['selection_policy']['component_growth_limit'] = 1.03
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No static load'))
    monkeypatch.setattr(module, 'authenticate_resume', lambda *a, **kw: pytest.fail('No resume load'))
    with pytest.raises(ValueError):
        module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


@pytest.mark.parametrize('mode', ['anderson1', 'linear-control'])
def test_candidate26_once_screens_precede_maps_and_all_trials_saved_with_last_role(tmp_path, monkeypatch, mode):
    module = driver()
    pf, seed25, image25 = wiring(module, monkeypatch)
    pf['acceleration']['mode'] = mode
    expected = mix_channel_spectra(seed25, image25, .05)
    expected_hashes = {name: hashlib.sha256(getattr(expected, name).tobytes()).hexdigest()
                       for name in module.COMPONENTS}
    mix_count = []
    def mix_once(*args):
        mix_count.append(args)
        return mix_channel_spectra(*args)
    monkeypatch.setattr(module, 'mix_channel_spectra', mix_once)
    def solve(*args, **kwargs):
        assert kwargs['seed_index'] == 25 and kwargs['config'].max_evaluations == 12
        assert kwargs['seed_input'] is seed25 and kwargs['seed_mapped'] is image25
        assert module.plain(kwargs['feasibility']) == FEASIBILITY
        assert module.plain(kwargs['selection']) == POLICY
        supplied = kwargs['sigma_initial'];history, decisions, states, batches = [], [], [], []
        for index in range(26, 38):
            batch = screen(index)
            if mode == 'linear-control' and index > 26:
                batch.update(original_kind='linear_control', selected_kind='linear_control')
                batch['screens'][0]['kind'] = 'linear_control'
            batches.append(batch)
            kwargs['on_proposal'](index, batch)
            proposal = {'kind': batch['selected_kind'], 'anchor_index': index-1,
                        'previous_index': None if index == 26 else index-2, 'feasibility': batch}
            kwargs['on_start'](index, supplied, proposal)
            evaluated = state(index, supplied)
            selected = index if index < 37 or mode == 'linear-control' else 36
            decision = choice(index, selected, accepted=selected == index)
            decision['kind'] = batch['selected_kind']
            kwargs['on_evaluation'](evaluated, decision)
            history.append(evaluated.record);decisions.append(decision);states.append(evaluated)
        return SimpleNamespace(state=states[-2] if mode == 'anderson1' else states[-1], last_evaluated_state=states[-1],
                               history=history, decisions=decisions, proposal_screens=batches,
                               converged=False, stop_reason='max_evaluations', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result, events = {}, []
    module.run(pf, result, lambda name, details=None: events.append((name, details)),
               tmp_path/'checkpoints', profile_sha256='a'*64)
    assert len(mix_count) == 1 and mix_count[0] == (seed25, image25, .05)
    assert result['initial_candidate']['construction'] == '.95*evaluated_input25+.05*unevaluated_image25'
    assert result['initial_candidate']['sigma_sha256'] == expected_hashes
    assert result['attempts'][0]['sigma_sha256'] == result['evaluated_records'][0]['sigma_input_sha256'] == expected_hashes
    assert result['passed'] and not result['converged'] and result['completed_evaluations'] == 12
    assert result['selected_anchor_index'] == result['selected_anchor']['index'] == (36 if mode == 'anderson1' else 37)
    assert result['last_evaluated_trial']['index'] == 37
    assert result['proposal_feasibility_config'] == FEASIBILITY
    assert len(result['proposal_screens']) == len(result['attempts']) == len(result['checkpoints']) == 12
    for saved, rec, decision in zip(result['checkpoints'], result['evaluated_records'], result['decisions'], strict=True):
        assert saved['manifest']['record'] == rec and rec['controller_decision'] == decision
        assert saved['manifest']['binding'] == module.checkpoint_binding(pf, 'a'*64)
        assert saved['manifest']['accepted_restart'] is False
    assert [name for name, _ in events] == [
        'parent_authentication_start', 'parent_authenticated', 'resume_authentication_start',
        'resume_authenticated', 'channel_accel_start']+[
            name for _ in range(12) for name in ('proposal_screened', 'scba_evaluation_start',
                'scba_evaluation_complete', 'checkpoint_saved', 'controller_decision')]+['scba_stopped']
    assert result['scope']['first_candidate26_linear_once']
    assert result['scope']['controller_history_reset'] and result['scope']['seed25_only_history']
    assert result['scope']['previous24_imported'] is False
    assert result['matched_experiment'] == pf['matched_experiment']
    assert not result['scope']['accepted_restart'] and not result['scope']['accepted_field_update']


def test_hard_screen_failure_before_next_map_preserves_batch_and_last_complete_state(tmp_path, monkeypatch):
    module = driver();pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        supplied = kwargs['sigma_initial'];batch26 = screen(26)
        kwargs['on_proposal'](26, batch26)
        kwargs['on_start'](26, supplied, {'kind': 'linear_warmup', 'feasibility': batch26})
        completed, decision = state(26, supplied), choice(26, 26)
        kwargs['on_evaluation'](completed, decision)
        bad = screen(27, passed=False, map_admitted=False);kwargs['on_proposal'](27, bad)
        return SimpleNamespace(state=completed, last_evaluated_state=completed,
            history=[completed.record], decisions=[decision], proposal_screens=[batch26, bad],
            converged=False, stop_reason='proposal_error', failure_reason='bad finite non-PSD guard', plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result, events = {}, []
    module.run(pf, result, lambda event, details=None: events.append(event),
               tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['passed'] is False and result['failure_reason'] == 'bad finite non-PSD guard'
    assert [row['index'] for row in result['attempts']] == [26]
    assert [row['index'] for row in result['proposal_screens']] == [26, 27]
    assert result['selected_anchor_index'] == result['last_evaluated_trial']['index'] == 26
    assert len(result['checkpoints']) == 1
    assert events[-2:] == ['proposal_screened', 'scba_stopped']


@pytest.mark.parametrize('screen_completed', [False, True])
def test_time_admission_can_end_with_one_unadmitted_screen_batch_without_a_map(tmp_path, monkeypatch, screen_completed):
    module = driver();pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        supplied = kwargs['sigma_initial'];batch26 = screen(26)
        kwargs['on_proposal'](26, batch26)
        kwargs['on_start'](26, supplied, {'kind': 'linear_warmup', 'feasibility': batch26})
        completed, decision = state(26, supplied), choice(26, 26)
        kwargs['on_evaluation'](completed, decision)
        pending = screen(27, passed=screen_completed, map_admitted=False)
        pending.update(error=None, stop_reason='map_time_budget' if screen_completed else 'time_budget')
        kwargs['on_proposal'](27, pending)
        return SimpleNamespace(state=completed, last_evaluated_state=completed,
            history=[completed.record], decisions=[decision], proposal_screens=[batch26, pending],
            converged=False, stop_reason='max_seconds', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result = {}
    module.run(pf, result, lambda *args: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['passed'] is True and result['stop_reason'] == 'max_seconds'
    assert len(result['proposal_screens']) == 2 and len(result['attempts']) == 1
    assert result['proposal_screens'][-1]['map_admitted'] is False
    assert result['last_evaluated_trial']['index'] == result['selected_anchor_index'] == 26


@pytest.mark.parametrize('defect', ['missing_screen', 'unadmitted', 'different_batch', 'first_hash'])
def test_a_map_cannot_start_without_matching_admitted_batch_and_exact_first_input(tmp_path, monkeypatch, defect):
    module = driver();pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        batch = screen(26, map_admitted=defect != 'unadmitted')
        if defect != 'missing_screen': kwargs['on_proposal'](26, batch)
        proposal = {'kind': 'linear_warmup', 'feasibility': copy.deepcopy(batch)}
        if defect == 'different_batch': proposal['feasibility']['chosen_factor'] = .5
        sigma = spectrum(99.) if defect == 'first_hash' else kwargs['sigma_initial']
        kwargs['on_start'](26, sigma, proposal)
        pytest.fail('Invalid admission cannot reach a fresh evaluation')
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result = {}
    with pytest.raises(ValueError):
        module.run(pf, result, lambda *args: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == [] and result['checkpoints'] == []


def test_native25_authentication_requires_actual_selected_complete_state(tmp_path):
    module, pf = driver(), profile();rec = record(25)
    rec.update(currents={'left': 2e-9, 'right': -2e-9, 'collision_power': 0.},
               input_sigma_is_zero=False, sigma_input_evaluated=True,
               mapped_sigma_dyson_evaluated=False, status='evaluated',
               controller_decision=choice(25, 25))
    saved = save_channel_checkpoint(tmp_path/'seed', iteration=25, sigma_input=spectrum(),
                                    mapped_sigma=spectrum(4.), record=rec, binding=pf['resume']['binding'])
    resume = dict(pf['resume'], path=saved['path'], manifest_sha256=saved['manifest_sha256'])
    loaded, authenticated = module.authenticate_resume(resume)
    assert loaded.record['index'] == authenticated['iteration'] == 25
    assert authenticated['artifacts_freshly_authenticated'] == 6
    assert all(row['stat_before'] == row['stat_after'] for row in authenticated['artifacts'].values())
    rec['controller_decision'] = choice(25, 18, accepted=False)
    rejected = save_channel_checkpoint(tmp_path/'rejected', iteration=25, sigma_input=spectrum(),
                                       mapped_sigma=spectrum(4.), record=rec, binding=pf['resume']['binding'])
    with pytest.raises(ValueError, match='complete nonconverged'):
        module.authenticate_resume(dict(resume, path=rejected['path'], manifest_sha256=rejected['manifest_sha256']))


@pytest.mark.parametrize('mode', ['anderson1', 'linear-control'])
def test_both_matched_modes_share_seed26_warmup_and_fixed_scientific_gates(mode):
    module, pf = driver(), profile()
    pf['acceleration']['mode'] = mode
    config, acceleration = module.validate_profile(pf)
    assert config.max_evaluations == 12 and acceleration.mode == mode
    assert config.max_seconds == 79200 and config.time_reserve_seconds == 300
    assert module.seed_binding(pf) == pf['resume']['binding']
    assert module.checkpoint_binding(pf, 'a'*64)['matched_experiment'] == pf['matched_experiment']


@pytest.mark.parametrize('key,value', [
    ('controller_history_reset', False), ('common_seed_iteration', 24),
    ('common_first_index', 25), ('max_fresh_attempts', 6),
    ('last_possible_index', 38), ('comparison_arms', ['anderson1']),
    ('comparison_baselines_used_computationally', True),
    ('no_automatic_retry_or_extension', False),
])
def test_matched_metadata_cannot_change_history_budget_or_seed(key, value):
    module, pf = driver(), profile()
    pf['matched_experiment'][key] = value
    with pytest.raises(ValueError, match='Matched continuation'):
        module.validate_profile(pf)


@pytest.mark.parametrize('count', [6, 11, 13])
def test_exact_twelve_map_limit_is_pinned(count):
    module, pf = driver(), profile()
    pf['scba']['max_evaluations'] = count
    with pytest.raises(ValueError, match='Comparison bounds'):
        module.validate_profile(pf)


def test_rejected_trial_is_saved_then_new_fallback_uses_retained_anchor(tmp_path, monkeypatch):
    module = driver();pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        supplied = kwargs['sigma_initial'];history, decisions, states, batches = [], [], [], []
        for index, kind, anchor, accepted in (
                (26, 'linear_warmup', 25, True),
                (27, 'anderson1', 26, False),
                (28, 'linear_fallback', 26, True)):
            batch = screen(index)
            batch.update(anchor_index=anchor, original_kind=kind, selected_kind=kind)
            batch['screens'][0]['kind'] = kind;batches.append(batch)
            kwargs['on_proposal'](index, batch)
            proposal = {'kind': kind, 'anchor_index': anchor, 'previous_index': None,
                        'feasibility': batch}
            kwargs['on_start'](index, supplied, proposal)
            evaluated = state(index, supplied)
            decision = choice(index, index if accepted else anchor, accepted=accepted)
            decision.update(kind=kind, reference_anchor_index=anchor)
            kwargs['on_evaluation'](evaluated, decision)
            history.append(evaluated.record);decisions.append(decision);states.append(evaluated)
        return SimpleNamespace(state=states[-1], last_evaluated_state=states[-1],
            history=history, decisions=decisions, proposal_screens=batches,
            converged=False, stop_reason='max_seconds', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_accel', solve)
    result = {}
    module.run(pf, result, lambda *args: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert [x['index'] for x in result['attempts']] == [26, 27, 28]
    assert len(result['checkpoints']) == 3
    assert result['decisions'][1]['accepted_as_anchor'] is False
    assert result['checkpoints'][1]['manifest']['record']['index'] == 27
    assert result['attempts'][2]['proposal']['kind'] == 'linear_fallback'
    assert result['attempts'][2]['proposal']['anchor_index'] == 26
    assert result['selected_anchor_index'] == result['last_evaluated_trial']['index'] == 28
