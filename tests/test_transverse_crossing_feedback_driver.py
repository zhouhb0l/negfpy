"""Small driver wiring checks for the new full-domain feedback pilot."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1]/'scripts/run_transverse_crossing_feedback.py'
    spec = importlib.util.spec_from_file_location('crossing_feedback_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({'schema': 'negfpy.transverse-crossing-feedback.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'runtime_inputs': module.RAW_INPUTS, 'scba': module.SCBA, 'support': module.SUPPORT,
        'initial_state': 'fresh-zero', 'checkpoint_max_bytes': module.CHECKPOINT_MAX_BYTES,
        'min_free_bytes': module.MIN_FREE_BYTES, 'driver_timeout_seconds': 82800,
        'source_commit': 'a'*40, 'source_tree': 'b'*40})


def test_profile_keeps_full_pilot_bounds_and_forbids_old_state():
    module = driver()
    original = profile(module)
    config = module.validate_profile(original)
    assert config.max_evaluations == 2 and config.mixing == .05
    assert config.identity_rtol == 1e-8 and config.current_rtol == 1e-3
    changed = copy.deepcopy(original)
    changed['map']['eta_lead'] = .005
    with pytest.raises(ValueError, match='Frozen map'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['resume'] = 'old-native-checkpoint'
    with pytest.raises(ValueError, match='no old state'):
        module.validate_profile(changed)


def test_pilot_passes_full_domain_zero_start_and_publishes_callbacks(monkeypatch, tmp_path):
    module = driver()
    cfg = profile(module)
    mesh, grid = SimpleNamespace(n_qpoints=4), SimpleNamespace(n_points=601)
    vertex = SimpleNamespace(full_ndof=216, ndof=216, active_dofs=np.arange(216))
    contact = SimpleNamespace(device_indices=tuple(range(24)))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module._static, 'load_static',
        lambda p, d, s: (mesh, grid, 'old-crop-reference', 'dynamical', [contact, contact], {}, {}))
    monkeypatch.setattr(module._material, 'rebuild_material', lambda p: ('fc3', 'geometry', {}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda rows: None)
    monkeypatch.setattr(module._static, 'release', lambda arrays: None)
    monkeypatch.setattr(module, 'build_crossing_feedback_vertex',
        lambda *a, **kw: SimpleNamespace(vertex=vertex, metadata={'model_identity_sha256': 'c'*64}))
    plan = SimpleNamespace(peak_owned_bytes=1024)
    monkeypatch.setattr(module, 'plan_channel_scba', lambda *a, **kw: plan)
    saved = []

    def checkpoint(path, **kwargs):
        saved.append(kwargs)
        return {'path': str(path/'iteration-000001'), 'manifest_sha256': 'd'*64,
                'manifest': {'not_downloaded_into_result': True}}

    monkeypatch.setattr(module, 'save_channel_checkpoint', checkpoint)
    monkeypatch.setattr(module, 'zero_support_context', lambda spectrum, zero_index: {'diagnostic_only': True})
    record = {'index': 1, 'converged': False}
    state = SimpleNamespace(index=1, sigma_input=None, record=record,
        evaluation=SimpleNamespace(input_currents={'left': 1.}, checks={'original': 'unchanged'},
                                   map_sigma='unevaluated-image'))

    def solve(*args, **kwargs):
        assert args[1] is vertex
        assert kwargs['sigma_initial'] is None and kwargs['first_index'] == 1
        assert kwargs['config'].max_evaluations == 2
        assert kwargs['g'] == .03
        kwargs['on_start'](1, None)
        kwargs['on_evaluation'](state)
        return SimpleNamespace(converged=False, stop_reason='max_evaluations', failure_reason=None,
                               state=state, history=[record])

    monkeypatch.setattr(module, 'run_channel_scba', solve)
    # Plans in production are dataclasses; this tiny mock needs a metadata adapter.
    original_plain = module.plain
    monkeypatch.setattr(module, 'plain',
        lambda v: {'peak_owned_bytes': v.peak_owned_bytes} if v is plan else original_plain(v))
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event),
               tmp_path/'checkpoints', profile_sha256='e'*64)
    assert result['passed'] is True and result['converged'] is False
    assert result['completed_evaluations'] == 1
    assert saved[0]['sigma_input'] is None
    assert saved[0]['mapped_sigma'] == 'unevaluated-image'
    assert saved[0]['binding']['feedback_dofs'] == list(range(216))
    assert saved[0]['binding']['model_identity_sha256'] == 'c'*64
    assert saved[0]['record']['mapped_sigma_dyson_evaluated'] is False
    assert set(result['checkpoints'][0]) == {'path', 'manifest_sha256'}
    assert events[-1] == 'scba_stopped'


def test_actual_failure_preserves_completed_state_and_exit_one(monkeypatch, tmp_path):
    module = driver()
    source = tmp_path/'profile.json'
    output = tmp_path/'result.json'
    source.write_text('{}')

    def fail(profile, result, emit, checkpoint_directory, *, profile_sha256):
        result.update(source_commit='a'*40, source_tree='b'*40,
                      profile_sha256=profile_sha256, completed_evaluations=1,
                      checkpoints=[{'path': 'saved-one', 'manifest_sha256': 'c'*64}])
        raise ValueError('original physical guard failed')

    monkeypatch.setattr(module, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['completed_evaluations'] == 1
    assert result['checkpoints'][0]['path'] == 'saved-one'
    assert result['failure_reason'] == 'ValueError: original physical guard failed'
    events = [json.loads(line)['event'] for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events == ['crossing_feedback_failed', 'crossing_feedback_closed']
