"""Driver authentication/role/budget contracts; no material fields are loaded."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum


def driver():
    spec = importlib.util.spec_from_file_location(
        'native_zero_diagnostic_driver',
        Path(__file__).parents[1]/'scripts/run_transverse_native_zero_diagnostic.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    value = {
        'schema': 'negfpy.transverse-native-zero-diagnostic.profile.v1',
        'source_commit': 'a'*40, 'source_tree': 'b'*40,
        'automatic_resubmission': False, 'automatic_successor': False,
        'driver_timeout_seconds': 82800, 'diagnostic': copy.deepcopy(module.DIAGNOSTIC),
        'map': copy.deepcopy(module._matched.PHYSICS), 'parent_max_bytes': 8*1024**3,
        'parent': {'root': '/static-parent', 'result_sha256': 'c'*64,
                   'profile_sha256': 'd'*64, 'source_commit': 'e'*40, 'source_tree': 'f'*40},
    }
    binding = module.expected_checkpoint_binding(value)
    value['checkpoints'] = [{'iteration': index, 'path': f'/native/iteration-{index:06d}',
        'manifest_sha256': digest, 'binding': copy.deepcopy(binding), 'max_total_bytes': 4*1024**3}
        for index, digest in module.CASE_MANIFESTS.items()]
    value['runtime_inputs'] = [
        {'name': 'parent_profile', 'path': '/static-parent/profile.json', 'sha256': 'd'*64,
         'bytes': 1},
        {'name': 'parent_result', 'path': '/static-parent/runtime/material-map-result.json',
         'sha256': 'c'*64, 'bytes': 2},
        *[{'name': f'checkpoint_{case["iteration"]}_manifest',
           'path': str(Path(case['path'])/'manifest.json'), 'sha256': case['manifest_sha256'],
           'bytes': 3} for case in value['checkpoints']],
    ]
    return value


def record(module, index):
    return {
        'index': index, 'converged': False, 'sigma_passed': False, 'current_passed': True,
        'sigma_input_evaluated': True, 'mapped_sigma_dyson_evaluated': False,
        'input_sigma_is_zero': False,
        'sigma_input_sha256': {key: str(n)*64 for n, key in enumerate(module.COMPONENTS, 1)},
        'mapped_sigma_sha256': {key: str(n)*64 for n, key in enumerate(module.COMPONENTS, 4)},
        'controller_decision': {'accepted_as_anchor': True, 'selected_anchor_index': index},
        'checks': {group: {'original_guard': {'ratio': .5}} for group in module.GLOBAL_GROUPS},
        'current_residuals': {name: {'ratio': .1} for name in
                              ('contact_balance', 'collision_power', 'full_balance')},
        'zero_frequency_context': {'diagnostic_only': True, 'local_ratio': 200.},
    }


def spectrum(factor):
    return ChannelSpectrum(*(np.full((1, 3, 1, 1), factor*v, dtype=np.complex128)
                             for v in (1., -2j, 1j)))


def wiring(module, monkeypatch):
    value = profile(module)
    order, supplied, images = [], {}, {}
    fields = SimpleNamespace(mesh=SimpleNamespace(n_qpoints=4),
        grid=SimpleNamespace(n_points=601), vertex=SimpleNamespace(ndof=120),
        dynamical=np.zeros((1, 2, 216)), left='left', right='right',
        binding={'artifacts_freshly_authenticated': 19, 'artifacts': {}})
    monkeypatch.setattr(module, 'authenticate_json_inputs', lambda rows: [])
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **k: fields)
    monkeypatch.setattr(module, 'verify_unchanged', lambda binding: order.append('stable'))

    def authenticate(case):
        index = case['iteration']
        order.append(('authenticate', index))
        supplied[index], images[index] = spectrum(index), spectrum(-index)
        cp = SimpleNamespace(sigma_input=supplied[index], mapped_sigma=images[index],
                             record=record(module, index), binding=case['binding'])
        return cp, {'iteration': index, 'artifacts': {}, 'input_role': 'fixed evaluated input',
                    'mapped_role': 'comparison total only', 'accepted_restart': False}

    monkeypatch.setattr(module, 'authenticate_checkpoint', authenticate)
    monkeypatch.setattr(module, 'release_checkpoint', lambda cp: order.append(('release', cp.record['index'])))

    def study(grid, vertex, dynamical, left, right, sigma_input, mapped_sigma, **kwargs):
        index = next(index for index, value in supplied.items() if value is sigma_input)
        order.append(('study', index))
        assert mapped_sigma is images[index]
        assert kwargs['g'] == .03 and kwargs['max_dyson_evaluations'] == 20
        assert kwargs['max_refinements'] == 4 and kwargs['half_width'] == 2
        assert kwargs['selected_elements'] == [[9, 102], [96, 102], [102, 102]]
        plan = module.plain(module.plan_native_zero_diagnostic_dimensions(4, 601, 216, 120))
        return {'plan': plan, 'dyson_evaluations': 20, 'local_ratio': 200.,
                'scope': {'diagnostic_only': True}, 'tiny_scalar': np.float64(index)}

    monkeypatch.setattr(module, 'study_native_zero_diagnostic', study)
    return value, order, supplied, images


def test_profile_exact_three_cases_five_pins_and_full_lineage():
    module = driver()
    value = profile(module)
    assert module.validate_profile(value) == module.DIAGNOSTIC
    for case in value['checkpoints']:
        node = case['binding']
        for ancestor in (25, 19, 15, 9, 3):
            assert node['resume_parent_iteration'] == ancestor
            node = node.get('resume_parent_binding', {})
        assert case['binding']['matched_experiment']['controller_history_reset'] is True
        assert case['binding']['acceleration']['mode'] == 'linear-control'
    assert [row['name'] for row in value['runtime_inputs']] == [
        'parent_profile', 'parent_result', 'checkpoint_27_manifest',
        'checkpoint_28_manifest', 'checkpoint_29_manifest']


@pytest.mark.parametrize('defect', ['physics', 'eta', 'guard', 'g', 'halfwidth', 'points',
    'totalpoints', 'work', 'totalwork', 'memory', 'refinements', 'casecount', 'cases',
    'order', 'manifest', 'lineage', 'policy', 'hashpin', 'pinname', 'pinbytes',
    'timeout', 'parentbytes', 'retry', 'successor', 'resume', 'source'])
def test_changed_guards_roles_bounds_or_pins_rejected_before_loading(monkeypatch, defect):
    module = driver()
    value = profile(module)
    if defect == 'physics': value['map']['g'] = .05
    elif defect == 'eta': value['map']['eta_lead'] *= 2
    elif defect == 'guard': value['diagnostic']['identity_rtol'] = 1e-6
    elif defect == 'g': value['diagnostic']['g'] = .05
    elif defect == 'halfwidth': value['diagnostic']['half_width'] = 3
    elif defect == 'points': value['diagnostic']['max_dyson_evaluations'] = 21
    elif defect == 'totalpoints': value['diagnostic']['max_total_dyson_evaluations'] = 61
    elif defect == 'work': value['diagnostic']['max_contraction_work'] += 1
    elif defect == 'totalwork': value['diagnostic']['max_total_contraction_work'] += 1
    elif defect == 'memory': value['diagnostic']['max_bytes'] *= 2
    elif defect == 'refinements': value['diagnostic']['max_refinements'] = 5
    elif defect == 'casecount': value['diagnostic']['max_cases'] = 4
    elif defect == 'cases': value['checkpoints'].append(copy.deepcopy(value['checkpoints'][-1]))
    elif defect == 'order': value['checkpoints'].reverse()
    elif defect == 'manifest': value['checkpoints'][0]['manifest_sha256'] = '0'*64
    elif defect == 'lineage': del value['checkpoints'][0]['binding']['resume_parent_binding']
    elif defect == 'policy': value['checkpoints'][0]['binding']['selection_policy']['merit_ratio_limit'] = 1.
    elif defect == 'hashpin': value['runtime_inputs'][0]['sha256'] = '0'*64
    elif defect == 'pinname': value['runtime_inputs'][0]['name'] = 'not_parent'
    elif defect == 'pinbytes': value['runtime_inputs'][0]['bytes'] = 0
    elif defect == 'timeout': value['driver_timeout_seconds'] = 86400
    elif defect == 'parentbytes': value['parent_max_bytes'] = 16*1024**3
    elif defect == 'retry': value['automatic_resubmission'] = True
    elif defect == 'successor': value['automatic_successor'] = True
    elif defect == 'resume': value['resume'] = {'iteration': 25}
    else: value['source_commit'] = 'bad'
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **k: pytest.fail('no material loading'))
    with pytest.raises(ValueError):
        module.run(value, {}, lambda *a: None, profile_sha256='a'*64)


def test_fixed_evaluated_inputs_sequential_release_scalar_output_and_unchanged_gates(monkeypatch):
    module = driver()
    value, order, supplied, images = wiring(module, monkeypatch)
    result, events = {}, []
    module.run(value, result, lambda name, details=None: events.append((name, details)),
               profile_sha256='a'*64)
    assert result['passed'] is True and result['converged'] is False
    assert result['completed_cases'] == result['started_cases'] == 3
    assert result['admitted_dyson_evaluation_bound'] == 60
    assert result['admitted_contraction_work_bound'] == 149299200000
    for index in (27, 28, 29):
        assert order.index(('authenticate', index)) < order.index(('study', index)) < order.index(('release', index))
        if index < 29:
            assert order.index(('release', index)) < order.index(('authenticate', index+1))
        assert supplied[index] is not images[index]
    assert [event[0] for event in events] == [
        'diagnostic_parent_authentication_start', 'diagnostic_parent_authenticated',
        *[name for _ in (27, 28, 29) for name in ('checkpoint_authentication_start',
          'checkpoint_authenticated', 'diagnostic_case_start', 'diagnostic_case_complete')]]
    assert len(events) == 14  # main appends diagnostic_closed as the fifteenth driver event
    for key in ('scba', 'full_map', 'full_born_map', 'field_output', 'accepted_field_update',
                'accepted_restart', 'current_acceptance', 'checkpoint_mixing_or_update',
                'mapped_sigma_used_as_green_input', 'cached_parent_green_or_born_used_computationally'):
        assert result['scope'][key] is False
    assert result['cases'][1]['diagnostic']['local_ratio'] == 200.
    json.dumps(module.plain(result), allow_nan=False)


@pytest.mark.parametrize('defect', ['pointcount', 'work', 'memory'])
def test_plan_bound_fails_before_any_case_load(monkeypatch, defect):
    module = driver()
    value, order, _, _ = wiring(module, monkeypatch)
    plan = module.plain(module.plan_native_zero_diagnostic_dimensions(4, 601, 216, 120))
    plan[{'pointcount': 'dyson_evaluations', 'work': 'contraction_work', 'memory': 'peak_owned_bytes'}[defect]] = 10**12
    monkeypatch.setattr(module, 'plan_native_zero_diagnostic_dimensions', lambda *a, **k: plan)
    with pytest.raises(ValueError, match='plan exceeds'):
        module.run(value, {}, lambda *a: None, profile_sha256='a'*64)
    assert not order


def test_module_failure_preserves_started_case_stops_and_releases(monkeypatch):
    module = driver()
    value, order, _, _ = wiring(module, monkeypatch)
    def fail(*args, **kwargs):
        raise ValueError('singular full-device operator')
    monkeypatch.setattr(module, 'study_native_zero_diagnostic', fail)
    result, events = {}, []
    with pytest.raises(ValueError, match='singular'):
        module.run(value, result, lambda name, details=None: events.append(name), profile_sha256='a'*64)
    assert result['started_cases'] == 1 and result['completed_cases'] == 0
    assert result['cases'][0]['status'] == 'failed'
    assert result['stop_reason'] == 'case_failed' and result['passed'] is False
    assert ('release', 27) in order and ('authenticate', 28) not in order
    assert events[-1] == 'diagnostic_case_start'


@pytest.mark.parametrize('when', ['before', 'after_authentication'])
def test_time_admission_preserves_actual_unstarted_tail(monkeypatch, when):
    module = driver()
    value, order, _, _ = wiring(module, monkeypatch)
    ticks = iter([0., 78900.] if when == 'before' else [0., 0., 78900.])
    result = {}
    module.run(value, result, lambda *a: None, profile_sha256='a'*64, clock=lambda: next(ticks))
    assert result['passed'] is False and result['started_cases'] == result['completed_cases'] == 0
    assert result['unstarted_case_index'] == 27
    assert result['stop_reason'].startswith('time_admission_')
    if when == 'before':
        assert not any(isinstance(row, tuple) for row in order)
        assert result['cases'] == []
    else:
        assert ('release', 27) in order and result['cases'][0]['status'] == 'not_admitted'


@pytest.mark.parametrize('bad', [np.zeros(1), {'accidental_array': np.zeros(1)}, float('inf'), float('nan'), complex(1, 2)])
def test_scalar_output_contract_rejects_arrays_complex_and_nonfinite(bad):
    with pytest.raises((TypeError, ValueError)):
        driver().plain(bad)


@pytest.mark.parametrize('defect', ['index', 'converged', 'accepted', 'current', 'global', 'payload', 'roles'])
def test_saved_record_original_pass_and_input_mapped_roles_required(defect):
    module = driver()
    value = record(module, 27)
    if defect == 'index': value['index'] = 28
    elif defect == 'converged': value['converged'] = True
    elif defect == 'accepted': value['controller_decision']['accepted_as_anchor'] = False
    elif defect == 'current': value['current_residuals']['contact_balance']['ratio'] = 2.
    elif defect == 'global': value['checks']['raw_source']['original_guard']['ratio'] = 2.
    elif defect == 'payload': value['sigma_input_sha256']['retarded'] = None
    else: value['mapped_sigma_dyson_evaluated'] = True
    with pytest.raises(ValueError):
        module._validate_evaluated_record(value, 27)


def test_large_saved_local_diagnostic_does_not_add_a_gate():
    module = driver()
    value = record(module, 28)
    value['checks']['full_green_exports'] = {'diagnostic': {'ratio': 100.}}
    value['checks']['inverse'] = {'diagnostic': {'ratio': 100.}}
    module._validate_evaluated_record(value, 28)


def test_json_authentication_real_tiny_files_and_stability(tmp_path):
    module = driver()
    path = tmp_path/'tiny.json'
    path.write_text('{"tiny":1}\n')
    row = {'name': 'tiny', 'path': str(path), 'sha256': module.sha(path), 'bytes': path.stat().st_size}
    auth = module.authenticate_json_inputs([row])
    assert auth[0]['stat_before'] == auth[0]['stat_after']
    module.verify_unchanged({'artifacts': {'tiny': auth[0]}})
    path.write_text('{"tiny":2}\n')
    with pytest.raises(ValueError, match='changed'):
        module.verify_unchanged({'artifacts': {'tiny': auth[0]}})


def test_release_closes_all_six_tiny_readonly_memmaps(tmp_path):
    module = driver()
    arrays = []
    for index in range(6):
        path = tmp_path/f'{index}.npy'
        np.save(path, np.zeros((1, 3, 1, 1), dtype=np.complex128))
        arrays.append(np.load(path, mmap_mode='r'))
    checkpoint = SimpleNamespace(sigma_input=ChannelSpectrum(*arrays[:3]),
                                 mapped_sigma=ChannelSpectrum(*arrays[3:]))
    module.release_checkpoint(checkpoint)
    assert all(array._mmap.closed for array in arrays)


def test_six_array_authentication_uses_native_loader_pins_and_preserves_hash_domains(monkeypatch):
    module = driver()
    case = profile(module)['checkpoints'][0]
    artifacts = {label+'_'+name: {
        'relative_path': 'arrays/'+label+'_'+name+'.npy',
        'shape': [4, 601, 120, 120], 'dtype': 'complex128',
        'array_bytes': 553881600, 'file_bytes': 553881728, 'file_sha256': '7'*64}
        for label in ('sigma_input', 'mapped_sigma') for name in module.COMPONENTS}
    checkpoint = SimpleNamespace(sigma_input=spectrum(1.), mapped_sigma=spectrum(2.),
        record=record(module, 27), binding=case['binding'], manifest={'iteration': 27})
    monkeypatch.setattr(module, '_json_verified', lambda *a: {'artifacts': artifacts})
    descriptor = {'device': 1, 'inode': 2, 'bytes': 553881728, 'mtime_ns': 3, 'ctime_ns': 4}
    monkeypatch.setattr(module, '_descriptor', lambda path: dict(descriptor))
    calls = []
    def load(path, **kwargs):
        calls.append((path, kwargs))
        return checkpoint
    monkeypatch.setattr(module, 'load_channel_checkpoint', load)
    loaded, authentication = module.authenticate_checkpoint(case)
    assert loaded is checkpoint and len(calls) == 1
    assert calls[0][1] == {'expected_manifest_sha256': case['manifest_sha256'],
        'expected_binding': case['binding'], 'max_total_bytes': 4*1024**3}
    assert authentication['artifacts_freshly_authenticated'] == 6
    assert all(row['stat_before'] == row['stat_after'] for row in authentication['artifacts'].values())
    assert 'excluding NPY header' in authentication['raw_payload_hash_domain']
    assert 'including header' in authentication['artifact_hash_domain']
    assert checkpoint.record['sigma_input_sha256']['retarded'] != artifacts['sigma_input_retarded']['file_sha256']


@pytest.mark.parametrize('failure', [False, True])
def test_cli_preserves_actual_success_or_failed_tail_and_final_closure(tmp_path, monkeypatch, failure):
    module = driver()
    value, _, _, _ = wiring(module, monkeypatch)
    if failure:
        def fail(*args, **kwargs):
            raise ValueError('synthetic singular operator')
        monkeypatch.setattr(module, 'study_native_zero_diagnostic', fail)
    profile_path, output = tmp_path/'profile.json', tmp_path/'result.json'
    profile_path.write_text(json.dumps(value))
    monkeypatch.setattr('sys.argv', ['driver', '--profile', str(profile_path), '--output', str(output)])
    assert module.main() == int(failure)
    result = json.loads(output.read_text())
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events[-1]['event'] == 'diagnostic_closed'
    assert events[-1]['details']['passed'] is not failure
    assert result['passed'] is not failure and result['converged'] is False
    assert result['profile_sha256'] == module.sha(profile_path)
    if failure:
        assert events[-2]['event'] == 'diagnostic_failed'
        assert result['failure']['type'] == 'ValueError'
        assert result['started_cases'] == 1 and result['completed_cases'] == 0
        assert result['cases'][0]['status'] == 'failed'
        assert len(events) == 7  # two parent, authentication pair, start, failure, closure
    else:
        assert len(events) == 15 and result['completed_cases'] == 3
