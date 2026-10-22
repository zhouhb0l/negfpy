"""New tiny resume authentication and mocked six-state driver integration."""
import copy
import hashlib
import importlib.util
import json
import sys
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

PACKAGE = Path(__file__).resolve().parents[3]/'reports/phonon-phonon-interaction/multiq-channel-resume-20261022'


def driver():
    spec = importlib.util.spec_from_file_location('native_channel_resume_driver',
        Path(__file__).parents[1]/'scripts/run_transverse_channel_scba_resume.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile():
    value = json.loads((PACKAGE/'profile-draft.json').read_text())
    value.update(source_commit='1'*40, source_tree='2'*40)
    return value


def spectrum(numbers=(1, -1e20j, 1e-20j)):
    return ChannelSpectrum(*(np.full((1, 3, 1, 1), value, complex) for value in numbers))


def record(index):
    residual = {'error': 1., 'scale': 1., 'allowance': .01, 'ratio': 100.,
                'error_frobenius': 1., 'scale_frobenius': 1., 'relative_frobenius': 1.,
                'error_location': [0, 1, 0, 0], 'scale_location': [0, 1, 0, 0]}
    current = {'error_w': 0., 'scale_w': 2e-9, 'allowance_w': 2.01e-12, 'ratio': 0.}
    return {'index': index, 'map_seconds': .1, 'elapsed_seconds': .2,
            'sigma_residuals': {key: dict(residual) for key in ('retarded', 'lesser', 'greater')},
            'channel_residuals': {key: dict(residual) for key in ('statistical', 'spectral')},
            'current_residuals': {key: dict(current) for key in ('contact_balance', 'collision_power', 'full_balance')},
            'sigma_passed': False, 'current_passed': True, 'converged': False}


def state(index, supplied):
    return SimpleNamespace(index=index, sigma_input=supplied, record=record(index),
        evaluation=SimpleNamespace(map_sigma=spectrum((2, -2e20j, 2e-20j)),
            input_currents=TransverseCurrents(2e-9, -2e-9, 0.),
            checks={'fixture': {}}, zero_frequency_context={'available': True, 'frequency_index': 1}))


def wiring(module, monkeypatch):
    pf = profile()
    input3, image3 = spectrum(), spectrum((4, -4e20j, 4e-20j))
    old = SimpleNamespace(sigma_input=input3, mapped_sigma=image3)
    fields = SimpleNamespace(grid='grid', vertex='vertex', dynamical='dynamical',
                             left='left', right='right', binding={'static19': True})
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    monkeypatch.setattr(module, 'authenticate_resume', lambda resume: (old, {
        'path': resume['path'], 'manifest_sha256': resume['manifest_sha256'],
        'iteration': 3, 'binding': copy.deepcopy(resume['binding']),
        'artifacts_freshly_authenticated': 6, 'artifacts': {}, 'accepted_restart': False}))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=256*1024**3))
    return pf, input3, image3


@pytest.fixture
def saved_resume(tmp_path):
    module = driver()
    pf = profile()
    rec = record(3)
    rec.update(currents={'left': 2e-9, 'right': -2e-9, 'collision_power': 0.},
               input_sigma_is_zero=False, sigma_input_evaluated=True,
               mapped_sigma_dyson_evaluated=False, status='evaluated')
    saved = save_channel_checkpoint(tmp_path/'old', iteration=3, sigma_input=spectrum(),
                                    mapped_sigma=spectrum((4, -4e20j, 4e-20j)), record=rec,
                                    binding=pf['resume']['binding'])
    resume = dict(pf['resume'], path=saved['path'], manifest_sha256=saved['manifest_sha256'])
    return module, resume, saved


def test_six_new_states_candidate4_and_native_checkpoint_lineage(tmp_path, monkeypatch):
    module = driver()
    pf, input3, image3 = wiring(module, monkeypatch)
    original_input, original_image = input3.spectral.copy(), image3.spectral.copy()
    events = []
    def solve(*args, **kwargs):
        assert args == ('grid', 'vertex', 'dynamical', 'left', 'right')
        assert kwargs['first_index'] == 4 and kwargs['config'].max_evaluations == 6
        assert kwargs['config'].mixing == .1 and kwargs['g'] == .03
        current = kwargs['sigma_initial']
        for name in ('retarded', 'statistical', 'spectral'):
            np.testing.assert_array_equal(getattr(current, name),
                                          .9*getattr(input3, name)+.1*getattr(image3, name))
        history = []
        for index in range(4, 10):
            kwargs['on_start'](index, current)
            evaluated = state(index, current)
            kwargs['on_evaluation'](evaluated)
            history.append(evaluated.record)
            current = mix_channel_spectra(current, evaluated.evaluation.map_sigma, .1)
        return SimpleNamespace(state=evaluated, history=history, converged=False,
                               stop_reason='max_evaluations', failure_reason=None, plan={'tiny': True})
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    module.run(pf, result, lambda name, details=None: events.append(name), tmp_path/'checkpoints',
               profile_sha256='a'*64)
    assert result['passed'] and not result['converged'] and result['completed_evaluations'] == 6
    assert [row['index'] for row in result['attempts']] == list(range(4, 10))
    assert result['initial_candidate']['sigma_sha256'] == result['attempts'][0]['sigma_sha256']
    assert result['initial_candidate']['dyson_evaluated'] is False
    assert all(row['dyson_evaluated'] is False for row in result['attempts'])
    assert result['scope']['native_checkpoint_imported']
    assert not result['scope']['historical_contour_checkpoint_imported']
    assert not result['scope']['cached_green_used'] and not result['scope']['accepted_restart']
    assert result['scope']['trajectory_initial_state'] == 'fresh-zero'
    assert [row['manifest']['iteration'] for row in result['checkpoints']] == list(range(4, 10))
    for saved in result['checkpoints']:
        manifest = saved['manifest']
        assert manifest['binding']['source_commit'] == '1'*40
        assert manifest['binding']['profile_sha256'] == 'a'*64
        assert manifest['binding']['resume_parent_manifest_sha256'] == module.RESUME_MANIFEST
        assert manifest['binding']['resume_parent_iteration'] == 3
        assert manifest['binding']['initial_state'] == 'fresh-zero'
        assert manifest['binding']['source_commit'] != pf['resume']['binding']['source_commit']
        assert manifest['mapped_sigma_dyson_evaluated'] is False
        assert manifest['accepted_restart'] is False and len(manifest['artifacts']) == 6
        assert manifest['record']['zero_frequency_context']['available']
        assert set(manifest['record']['mapped_sigma_sha256']) == set(module.COMPONENTS)
    assert events == ['parent_authentication_start', 'parent_authenticated',
                      'resume_authentication_start', 'resume_authenticated', 'channel_resume_start']+[
                          event for _ in range(6) for event in
                          ('scba_evaluation_start', 'scba_evaluation_complete', 'checkpoint_saved')]+['scba_stopped']
    np.testing.assert_array_equal(input3.spectral, original_input)
    np.testing.assert_array_equal(image3.spectral, original_image)


def test_profile_guards_precede_static_or_resume_loading(tmp_path, monkeypatch):
    module = driver()
    for section, key, value in [('map', 'g', .02), ('map', 'eta_lead', .001),
                               ('scba', 'mixing', .2), ('scba', 'max_evaluations', 7),
                               ('scba', 'identity_rtol', 1e-6), ('root', 'first_index', 3),
                               ('root', 'initial_state', 'fresh-zero'),
                               ('resume', 'manifest_sha256', 'f'*64),
                               ('root', 'free_storage_min_bytes', 1)]:
        pf = profile()
        (pf if section == 'root' else pf[section])[key] = value
        monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No static read'))
        monkeypatch.setattr(module, 'authenticate_resume', lambda *a: pytest.fail('No resume read'))
        with pytest.raises(ValueError):
            module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


def test_old_binding_is_required_separately_from_new_source_profile():
    module = driver()
    pf = profile()
    assert module.validate_profile(pf).max_evaluations == 6
    for key in ('source_commit', 'source_tree', 'profile_sha256', 'representation', 'initial_state'):
        trial = copy.deepcopy(pf)
        trial['resume']['binding'][key] = trial.get(key, 'wrong')
        if trial['resume']['binding'][key] == pf['resume']['binding'][key]:
            trial['resume']['binding'][key] = 'wrong'
        with pytest.raises(ValueError, match='binding differs'):
            module.validate_profile(trial)


def test_six_resume_artifacts_freshly_authenticated_and_unchanged(saved_resume):
    module, resume, saved = saved_resume
    before = {path: path.read_bytes() for path in Path(saved['path']).rglob('*') if path.is_file()}
    checkpoint, binding = module.authenticate_resume(resume)
    assert binding['artifacts_freshly_authenticated'] == 6
    assert binding['manifest_sha256'] == saved['manifest_sha256']
    assert not checkpoint.sigma_input.spectral.flags.writeable
    for row in binding['artifacts'].values():
        assert row['stat_before'] == row['stat_after']
        assert row['shape'] == [1, 3, 1, 1] and row['dtype'] == 'complex128'
        assert row['stat_after']['bytes'] == row['file_bytes']
        assert hashlib.sha256(Path(row['path']).read_bytes()).hexdigest() == row['file_sha256']
    assert {path: path.read_bytes() for path in before} == before


def test_resume_tamper_rejected_before_any_mapping(saved_resume, monkeypatch):
    module, resume, saved = saved_resume
    path = Path(saved['path'])/'arrays'/'mapped_sigma_spectral.npy'
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    monkeypatch.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('No mapping before all six hashes pass'))
    with pytest.raises(ValueError, match='SHA256'):
        module.authenticate_resume(resume)


def test_resume_descriptor_mutation_during_loader_is_rejected(saved_resume, monkeypatch):
    module, resume, saved = saved_resume
    real_load = module.load_channel_checkpoint
    def changed(*args, **kwargs):
        loaded = real_load(*args, **kwargs)
        path = Path(saved['path'])/'arrays'/'sigma_input_spectral.npy'
        path.write_bytes(path.read_bytes()+b'changed')
        return loaded
    monkeypatch.setattr(module, 'load_channel_checkpoint', changed)
    with pytest.raises(ValueError, match='changed during authentication'):
        module.authenticate_resume(resume)


def test_resume_artifact_path_escape_rejected_without_mapping(saved_resume, monkeypatch):
    module, resume, saved = saved_resume
    path = Path(saved['path'])/'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['artifacts']['sigma_input_spectral']['relative_path'] = '../outside.npy'
    path.write_text(json.dumps(manifest))
    resume['manifest_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(np, 'memmap', lambda *a, **kw: pytest.fail('No mapping for invalid path'))
    with pytest.raises(ValueError, match='artifact path'):
        module.authenticate_resume(resume)


def test_insufficient_disk_space_fails_before_input_loading(tmp_path, monkeypatch):
    module = driver()
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=1))
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No read'))
    with pytest.raises(ValueError, match='free storage'):
        module.run(profile(), {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert not (tmp_path/'checkpoints').exists()


def test_unexpected_first_attempt_rejected_before_evaluation_or_checkpoint(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        kwargs['on_start'](3, kwargs['sigma_initial'])
        pytest.fail('No numerical evaluation after wrong index')
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    with pytest.raises(ValueError, match='unexpected'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == [] and result['checkpoints'] == []


def test_later_solver_failure_retains_completed_state_and_unmatched_attempt(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def solve(*args, **kwargs):
        supplied = kwargs['sigma_initial']
        kwargs['on_start'](4, supplied)
        complete = state(4, supplied)
        kwargs['on_evaluation'](complete)
        kwargs['on_start'](5, mix_channel_spectra(supplied, complete.evaluation.map_sigma, .1))
        return SimpleNamespace(state=complete, history=[complete.record], converged=False,
                               stop_reason='evaluation_error', failure_reason='synthetic guard failure', plan={})
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert not result['passed'] and result['failure_reason'] == 'synthetic guard failure'
    assert result['completed_evaluations'] == 1 and len(result['attempts']) == 2
    assert result['final_evaluated_state']['index'] == 4
    assert result['checkpoints'][0]['manifest']['iteration'] == 4
    assert not (tmp_path/'checkpoints'/'iteration-000005').exists()


def test_checkpoint_callback_failure_preserves_completed_record_and_stops(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring(module, monkeypatch)
    def fail(*args, **kwargs):
        raise OSError('synthetic checkpoint failure')
    monkeypatch.setattr(module, 'save_channel_checkpoint', fail)
    def solve(*args, **kwargs):
        kwargs['on_start'](4, kwargs['sigma_initial'])
        kwargs['on_evaluation'](state(4, kwargs['sigma_initial']))
        pytest.fail('No next evaluation after checkpoint failure')
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    with pytest.raises(OSError, match='checkpoint failure'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert len(result['evaluated_records']) == 1 and result['checkpoints'] == []


def test_main_preserves_failure_tail_and_closes_without_accepted_claim(tmp_path, monkeypatch):
    module = driver()
    profile_path = tmp_path/'profile.json'
    output = tmp_path/'result.json'
    profile_path.write_text(json.dumps(profile()))
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(profile_path), '--output', str(output)])
    def fail(pf, result, emit, checkpoint_directory, **kwargs):
        result['attempts'] = [{'index': 4, 'dyson_evaluated': False}]
        raise ValueError('synthetic authentication failure')
    monkeypatch.setattr(module, 'run', fail)
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['schema'] == 'negfpy.transverse-channel-scba-resume.v1'
    assert not result['passed'] and not result['converged']
    assert result['attempts'][0]['index'] == 4
    assert result['failure_reason'] == 'ValueError: synthetic authentication failure'
    events = [json.loads(row)['event'] for row in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events == ['channel_scba_resume_failed', 'channel_scba_resume_closed']
