"""Same-state9 .05 comparison admission and tiny mocked driver controls."""
import copy
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
from test_transverse_channel_scba_resume9_driver import profile9, wiring9
from test_transverse_channel_scba_resume_driver import driver, state

from negfpy.anharmonic.transverse_channel_feedback import mix_channel_spectra


def baseline():
    return {'job_id': '25919346.pbs101',
            'profile_sha256': 'a26557c1305786bb7e024ade98b634196812aeaba945aa483d0feb953c439ffb',
            'result_sha256': 'f6bb9aca9192dc6908a86688bcc0052c08c3112732581306efe46e421ef87d9f',
            'mixing': .1, 'resume_iteration': 9,
            'resume_manifest_sha256': 'cf591c69ea613f59e336cf66e6e8a909be6f0148bb5747103ee2b2c829b1ba5b',
            'first_index': 10, 'max_evaluations': 6,
            'result_path': '/does/not/exist/result.json', 'use': 'Recorded scalar comparison only.'}


def test_state9_mixing005_admitted_with_original_physics_and_tolerances():
    module = driver()
    pf = profile9(module)
    pf['scba']['mixing'] = .05
    pf['comparison_baseline'] = baseline()
    config = module.validate_profile(pf)
    assert config.mixing == .05 and config.max_evaluations == 6
    assert (config.sigma_rtol, config.sigma_atol) == (1e-4, 1e-10)
    assert (config.current_rtol, config.current_atol_w) == (1e-3, 1e-14)
    assert (config.identity_rtol, config.identity_atol) == (1e-8, 1e-11)
    assert all(pf['map'][key] == value for key, value in module.PHYSICS.items())


def test_mixing005_requires_complete_immutable_same_seed_baseline_before_loading(tmp_path, monkeypatch):
    module = driver()
    pf = profile9(module)
    pf['scba']['mixing'] = .05
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No static load'))
    monkeypatch.setattr(module, 'authenticate_resume', lambda *a, **kw: pytest.fail('No resume load'))
    for value in [None, {}, *[dict(baseline(), **{key: 'changed'})
                            for key in ('job_id', 'profile_sha256', 'result_sha256', 'mixing',
                                        'resume_manifest_sha256', 'resume_iteration', 'first_index',
                                        'max_evaluations')]]:
        if value is None:
            pf.pop('comparison_baseline', None)
        else:
            pf['comparison_baseline'] = value
        with pytest.raises(ValueError, match='comparison baseline'):
            module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


@pytest.mark.parametrize(('seed', 'mixing'), [(3, .05), (9, .075), (9, .2)])
def test_unapproved_seed_mixing_pair_rejected_before_material_loading(
        tmp_path, monkeypatch, seed, mixing):
    module = driver()
    pf = profile9(module)
    if seed == 3:
        pf['first_index'] = 4
        pf['resume'].update(iteration=3, manifest_sha256=module.RESUME_MANIFEST,
                            binding=dict(module.OLD_BINDING, physics=copy.deepcopy(pf['map']),
                                         parent_result_sha256=pf['parent']['result_sha256'],
                                         parent_profile_sha256=pf['parent']['profile_sha256']))
    pf['scba']['mixing'] = mixing
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('No static load'))
    monkeypatch.setattr(module, 'authenticate_resume', lambda *a, **kw: pytest.fail('No resume load'))
    with pytest.raises(ValueError, match='Mixing'):
        module.run(pf, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)


def test_mixing005_single_candidate10_and_six_new_records_keep_exact_lineage(tmp_path, monkeypatch):
    module = driver()
    pf, input9, image9 = wiring9(module, monkeypatch)
    pf['scba']['mixing'] = .05
    # Metadata may identify the immutable .1 comparison result; the driver must
    # not read it or use it to construct an input or image.
    pf['comparison_baseline'] = baseline()
    old_binding = copy.deepcopy(pf['resume']['binding'])
    old_bytes = {(label, name): getattr(value, name).tobytes()
                 for label, value in [('input', input9), ('image', image9)]
                 for name in module.COMPONENTS}
    expected = {name: .95*getattr(input9, name)+.05*getattr(image9, name)
                for name in module.COMPONENTS}
    expected_hashes = {name: hashlib.sha256(value.tobytes(order='C')).hexdigest()
                       for name, value in expected.items()}
    mix_calls = []
    def counted_mix(supplied, mapped, alpha):
        mix_calls.append((supplied, mapped, alpha))
        return mix_channel_spectra(supplied, mapped, alpha)
    monkeypatch.setattr(module, 'mix_channel_spectra', counted_mix)
    calls = []
    def solve(*args, **kwargs):
        assert kwargs['first_index'] == 10 and kwargs['config'].max_evaluations == 6
        assert kwargs['config'].mixing == .05 and kwargs['g'] == .03
        supplied = kwargs['sigma_initial']
        for name in module.COMPONENTS:
            np.testing.assert_array_equal(getattr(supplied, name), expected[name])
        history = []
        for index in range(10, 16):
            calls.append(index)
            kwargs['on_start'](index, supplied)
            evaluated = state(index, supplied)
            kwargs['on_evaluation'](evaluated)
            history.append(evaluated.record)
            if index < 15:
                supplied = mix_channel_spectra(supplied, evaluated.evaluation.map_sigma, .05)
        return SimpleNamespace(state=evaluated, history=history, converged=False,
                               stop_reason='max_evaluations', failure_reason=None, plan={})
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result, events = {}, []
    module.run(pf, result, lambda name, details=None: events.append(name), tmp_path/'checkpoints',
               profile_sha256='a'*64)
    assert len(mix_calls) == 1
    assert mix_calls[0][0] is input9 and mix_calls[0][1] is image9 and mix_calls[0][2] == .05
    assert calls == list(range(10, 16))
    assert result['passed'] and not result['converged'] and result['completed_evaluations'] == 6
    assert result['initial_candidate']['construction'] == '.95*evaluated_input9+.05*unevaluated_image9'
    assert result['initial_candidate']['mixing'] == .05
    assert result['initial_candidate']['index'] == 10
    assert result['initial_candidate']['dyson_evaluated'] is False
    assert result['comparison_baseline'] == pf['comparison_baseline']
    assert result['comparison_baseline'] is not pf['comparison_baseline']
    assert result['scope']['comparison_baseline_used_computationally'] is False
    assert result['initial_candidate']['sigma_sha256'] == expected_hashes
    assert result['attempts'][0]['sigma_sha256'] == expected_hashes
    assert result['evaluated_records'][0]['sigma_input_sha256'] == expected_hashes
    assert [row['index'] for row in result['evaluated_records']] == list(range(10, 16))
    for saved, record in zip(result['checkpoints'], result['evaluated_records'], strict=True):
        manifest, binding = saved['manifest'], saved['manifest']['binding']
        assert manifest['iteration'] == record['index']
        assert manifest['record'] == record
        assert binding['resume_parent_iteration'] == 9
        assert binding['resume_parent_manifest_sha256'] == module.STATE9_MANIFEST
        assert binding['resume_parent_binding'] == old_binding
        assert binding['resume_parent_binding']['resume_parent_iteration'] == 3
        assert binding['source_commit'] == pf['source_commit'] and binding['profile_sha256'] == 'a'*64
        assert binding['initial_state'] == 'fresh-zero' and 'mixing' not in binding
        assert not manifest['accepted_restart'] and not manifest['mapped_sigma_dyson_evaluated']
        assert record['sigma_input_evaluated'] and not record['input_sigma_is_zero']
    assert pf['resume']['binding'] == old_binding
    assert not result['scope']['accepted_restart'] and not result['scope']['accepted_field_update']
    assert not result['scope']['converged_transport']
    assert events == ['parent_authentication_start', 'parent_authenticated',
                      'resume_authentication_start', 'resume_authenticated', 'channel_resume_start']+[
                          event for _ in range(6) for event in
                          ('scba_evaluation_start', 'scba_evaluation_complete', 'checkpoint_saved')]+['scba_stopped']
    assert {(label, name): getattr(value, name).tobytes()
            for label, value in [('input', input9), ('image', image9)]
            for name in module.COMPONENTS} == old_bytes


def test_mixing005_first_attempt_rejects_candidate_built_with_baseline_alpha(tmp_path, monkeypatch):
    module = driver()
    pf, input9, image9 = wiring9(module, monkeypatch)
    pf['scba']['mixing'] = .05
    pf['comparison_baseline'] = baseline()
    def solve(*args, **kwargs):
        kwargs['on_start'](10, mix_channel_spectra(input9, image9, .1))
        pytest.fail('No evaluation may follow the wrong comparison candidate')
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result = {}
    with pytest.raises(ValueError, match='Candidate10 differs'):
        module.run(pf, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='a'*64)
    assert result['attempts'] == [] and result['checkpoints'] == []
