"""New state9 pin, recursive provenance and six-new-state driver controls."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest
from test_transverse_channel_scba_resume_driver import (
    driver,
    record,
    spectrum,
    state,
)

from negfpy.anharmonic.transverse_channel_feedback import mix_channel_spectra
from negfpy.anharmonic.transverse_channel_restart import save_channel_checkpoint


def profile9(module):
    settings = copy.deepcopy(module.PHYSICS)
    settings.update(expected_fc3_content_sha256='6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e',
                    expected_geometry_sha256='778afae03a7f83580b6eac8ab5529901cbf25fee7fa3ebacbdded85c50a85ab5')
    parent = {'root': '/scratch/static-parent', 'result_sha256': 'a'*64,
              'profile_sha256': 'b'*64, 'source_commit': 'c'*40, 'source_tree': 'd'*40}
    return {'schema': 'negfpy.transverse-channel-scba-resume.profile.v1',
            'initial_state': 'native-channel-continuation', 'first_index': 10,
            'source_commit': '1'*40, 'source_tree': '2'*40, 'map': settings,
            'scba': module.plain(module.ChannelSCBAConfig(max_evaluations=6)),
            'parent': parent, 'parent_max_bytes': 8*1024**3,
            'checkpoint_max_bytes': 4*1024**3, 'free_storage_min_bytes': 128*1024**3,
            'driver_timeout_seconds': 82800, 'remote_root': '/scratch/native-state9-continuation',
            'resume': {'iteration': 9,
                       'path': '/scratch/closed-native-resume/runtime/checkpoints/iteration-000009',
                       'manifest_sha256': module.STATE9_MANIFEST, 'max_total_bytes': 4*1024**3,
                       'binding': dict(module.STATE9_BINDING, physics=copy.deepcopy(settings),
                                       parent_result_sha256=parent['result_sha256'],
                                       parent_profile_sha256=parent['profile_sha256'])}}


def wiring9(module, monkeypatch):
    supplied, mapped = spectrum(), spectrum((4, -4e20j, 4e-20j))
    pf = profile9(module)
    fields = SimpleNamespace(grid='grid', vertex='vertex', dynamical='dynamical',
                             left='left', right='right', binding={'static19': True})
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(module, 'authenticate_resume', lambda resume: (
        SimpleNamespace(sigma_input=supplied, mapped_sigma=mapped),
        {'path': resume['path'], 'manifest_sha256': resume['manifest_sha256'],
         'iteration': 9, 'binding': copy.deepcopy(resume['binding']),
         'artifacts_freshly_authenticated': 6, 'artifacts': {}, 'accepted_restart': False}))
    return pf, supplied, mapped


def test_approved_state9_pin_has_exact_source_profile_and_parent3_binding():
    module = driver()
    pf = profile9(module)
    assert module.validate_profile(pf).max_evaluations == 6
    assert module.STATE9_MANIFEST == 'cf591c69ea613f59e336cf66e6e8a909be6f0148bb5747103ee2b2c829b1ba5b'
    assert pf['resume']['binding']['source_commit'] == 'b014cc81ec1e7cfb1a7e2502feef3b67dd017de5'
    assert pf['resume']['binding']['source_tree'] == '9099644af64e86179847eeacf26978276dbce9c9'
    assert pf['resume']['binding']['profile_sha256'] == 'f2554a2e168c85db45247e861fb303751e0cc5c2e81d988fb686ad4b8710cd06'
    assert pf['resume']['binding']['resume_parent_iteration'] == 3
    assert pf['resume']['binding']['resume_parent_manifest_sha256'] == module.RESUME_MANIFEST
    assert pf['source_commit'] != pf['resume']['binding']['source_commit']


@pytest.mark.parametrize('defect', ['wrong_manifest', 'unapproved_seed', 'wrong_first_index'])
def test_invalid_state9_seed_tuple_rejected_before_any_material_load(tmp_path, monkeypatch, defect):
    module = driver()
    pf = profile9(module)
    if defect == 'wrong_manifest':
        pf['resume']['manifest_sha256'] = module.RESUME_MANIFEST
    elif defect == 'unapproved_seed':
        pf['resume']['iteration'] = 15
        pf['first_index'] = 16
    else:
        pf['first_index'] = 4
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No load'))
    with pytest.raises(ValueError):
        module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


def test_state9_full_binding_rejects_missing_changed_or_relabelled_lineage():
    module = driver()
    for key, value in [('resume_parent_manifest_sha256', 'f'*64),
                       ('resume_parent_iteration', 9),
                       ('source_commit', module.OLD_BINDING['source_commit']),
                       ('profile_sha256', 'a'*64), ('unexpected_lineage', {})]:
        pf = profile9(module)
        pf['resume']['binding'][key] = value
        with pytest.raises(ValueError, match='binding differs'):
            module.validate_profile(pf)
    pf = profile9(module)
    del pf['resume']['binding']['resume_parent_manifest_sha256']
    with pytest.raises(ValueError, match='binding differs'):
        module.validate_profile(pf)


def test_six_new_states10_through15_preserve_nested_state9_parent3_binding(tmp_path, monkeypatch):
    module = driver()
    pf, input9, image9 = wiring9(module, monkeypatch)
    original_binding = copy.deepcopy(pf['resume']['binding'])
    calls = []
    def solve(*args, **kwargs):
        assert kwargs['first_index'] == 10 and kwargs['config'].max_evaluations == 6
        supplied = kwargs['sigma_initial']
        for component in module.COMPONENTS:
            np.testing.assert_array_equal(getattr(supplied, component),
                                          .9*getattr(input9, component)+.1*getattr(image9, component))
        history = []
        for index in range(10, 16):
            calls.append(index)
            kwargs['on_start'](index, supplied)
            evaluated = state(index, supplied)
            kwargs['on_evaluation'](evaluated)
            history.append(evaluated.record)
            if index < 15:
                supplied = mix_channel_spectra(supplied, evaluated.evaluation.map_sigma, .1)
        return SimpleNamespace(state=evaluated, history=history, converged=False,
                               stop_reason='max_evaluations', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result, events = {}, []
    module.run(pf, result, lambda name, details=None: events.append(name), tmp_path/'checkpoints',
               profile_sha256='a'*64)
    assert calls == list(range(10, 16))
    assert result['passed'] and not result['converged']
    assert result['first_index'] == result['initial_candidate']['index'] == 10
    assert result['initial_candidate']['construction'] == '.9*evaluated_input9+.1*unevaluated_image9'
    assert result['initial_candidate']['dyson_evaluated'] is False
    assert result['initial_candidate']['sigma_sha256'] == result['attempts'][0]['sigma_sha256']
    assert [row['manifest']['iteration'] for row in result['checkpoints']] == list(range(10, 16))
    for saved in result['checkpoints']:
        manifest, binding = saved['manifest'], saved['manifest']['binding']
        assert binding['resume_parent_iteration'] == 9
        assert binding['resume_parent_manifest_sha256'] == module.STATE9_MANIFEST
        assert binding['resume_parent_binding'] == original_binding
        assert binding['resume_parent_binding']['resume_parent_iteration'] == 3
        assert binding['resume_parent_binding']['resume_parent_manifest_sha256'] == module.RESUME_MANIFEST
        assert binding['source_commit'] == pf['source_commit']
        assert binding['profile_sha256'] == 'a'*64
        assert binding['initial_state'] == 'fresh-zero'
        assert not manifest['accepted_restart'] and not manifest['mapped_sigma_dyson_evaluated']
    assert pf['resume']['binding'] == original_binding
    assert len(events) == 24 and events[-1] == 'scba_stopped'


def test_state9_authentication_reports_seed9_without_relabelling(tmp_path):
    module = driver()
    pf = profile9(module)
    rec = record(9)
    rec.update(currents={'left': 2e-9, 'right': -2e-9, 'collision_power': 0.},
               input_sigma_is_zero=False, sigma_input_evaluated=True,
               mapped_sigma_dyson_evaluated=False, status='evaluated')
    saved = save_channel_checkpoint(tmp_path/'old', iteration=9, sigma_input=spectrum(),
                                    mapped_sigma=spectrum((4, -4e20j, 4e-20j)), record=rec,
                                    binding=pf['resume']['binding'])
    resume = dict(pf['resume'], path=saved['path'], manifest_sha256=saved['manifest_sha256'])
    checkpoint, binding = module.authenticate_resume(resume)
    assert checkpoint.record['index'] == binding['iteration'] == 9
    assert binding['binding'] == pf['resume']['binding']
    assert binding['artifacts_freshly_authenticated'] == 6
    assert all(row['stat_before'] == row['stat_after'] for row in binding['artifacts'].values())
    # Profile admission still rejects this synthetic fixture's unapproved hash.
    with pytest.raises(ValueError, match='must be pinned'):
        module.validate_profile(dict(pf, resume=resume))


@pytest.mark.parametrize('bad_index', [4, 16])
def test_state9_callback_rejects_old_or_seventh_candidate_index(tmp_path, monkeypatch, bad_index):
    module = driver()
    pf, _, _ = wiring9(module, monkeypatch)
    def solve(*args, **kwargs):
        kwargs['on_start'](bad_index, kwargs['sigma_initial'])
        pytest.fail('No evaluation may follow an invalid state index')
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    with pytest.raises(ValueError, match='unexpected'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == [] and result['checkpoints'] == []


def test_state9_first_input_hash_must_match_already_mixed_candidate10(tmp_path, monkeypatch):
    module = driver()
    pf, _, _ = wiring9(module, monkeypatch)
    def solve(*args, **kwargs):
        kwargs['on_start'](10, spectrum((0, -1j, 0)))
        pytest.fail('No evaluation may follow changed candidate10 payload')
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    with pytest.raises(ValueError, match='Candidate10 differs'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == []
