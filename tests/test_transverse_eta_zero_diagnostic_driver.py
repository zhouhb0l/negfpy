"""Eta experiment input/contact isolation and failure contracts using tiny mocks."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def driver():
    return load_file('eta_zero_driver', Path(__file__).parents[1]/'scripts/run_transverse_eta_zero_diagnostic.py')


def fixture_module():
    return load_file('old_native_zero_driver_fixture', Path(__file__).with_name('test_transverse_native_zero_diagnostic_driver.py'))


def profile(module):
    value = fixture_module().profile(module._native)
    value.update(schema='negfpy.transverse-eta-zero-diagnostic.profile.v1',
                 diagnostic=copy.deepcopy(module.DIAGNOSTIC),
                 harmonic_inputs=copy.deepcopy(module.HARMONIC_INPUTS))
    value['runtime_inputs'] += copy.deepcopy(module.HARMONIC_INPUTS)
    value['map'].update(transport_matrix=[[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
        expected_geometry_sha256='778afae03a7f83580b6eac8ab5529901cbf25fee7fa3ebacbdded85c50a85ab5')
    for case in value['checkpoints']:
        case['binding'] = module.expected_checkpoint_binding(value)
    return value


def wiring(module, monkeypatch):
    value = profile(module)
    order, supplied, images, contacts = [], {}, {}, {}
    fields = SimpleNamespace(mesh=SimpleNamespace(n_qpoints=4), grid=SimpleNamespace(n_points=601),
        vertex=SimpleNamespace(ndof=120), dynamical=np.zeros((1, 2, 216)),
        left=object(), right=object(), binding={'artifacts_freshly_authenticated': 19, 'artifacts': {}})
    monkeypatch.setattr(module, 'authenticate_inputs', lambda value: [])
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **k: fields)
    monkeypatch.setattr(module, 'verify_unchanged', lambda value: order.append('stable'))
    monkeypatch.setattr(module, 'rebuild_harmonic', lambda *a: ('fresh_lead', np.eye(216, 3),
        {'fresh_from_pinned_fc2': True, 'saved_dynamical_comparison': {'passed': True}}))
    monkeypatch.setattr(module, 'audit_cropped_vertex_acoustic', lambda *a, **k:
        {'diagnostic_only': True, 'maximum_relative_residual': 10.})

    def build(fields_arg, lead, eta, config, record, deadline, clock):
        assert fields_arg is fields and lead == 'fresh_lead'
        order.append(('contacts', eta))
        contacts[eta] = (object(), object())
        record.update(left={'freshly_rebuilt': True}, right={'freshly_rebuilt': True})
        return contacts[eta]
    monkeypatch.setattr(module, 'build_contacts', build)

    def authenticate(case):
        index = case['iteration']
        order.append(('authenticate', index))
        supplied[index], images[index] = object(), object()
        cp = SimpleNamespace(sigma_input=supplied[index], mapped_sigma=images[index],
            record=fixture_module().record(module._native, index))
        return cp, {'iteration': index, 'artifacts': {}, 'input_role': 'fixed input',
                    'mapped_role': 'cross eta reference', 'accepted_restart': False}
    monkeypatch.setattr(module, 'authenticate_checkpoint', authenticate)
    monkeypatch.setattr(module, 'release_checkpoint', lambda cp: order.append(('release', cp.record['index'])))

    def study(grid, vertex, dynamical, left, right, sigma_input, mapped_sigma, **kw):
        index = next(index for index, value in supplied.items() if value is sigma_input)
        eta = kw['eta_lead']
        order.append(('study', index, eta))
        assert (left, right) == contacts[eta]
        assert left is not fields.left and right is not fields.right
        assert mapped_sigma is images[index] and mapped_sigma is not sigma_input
        assert kw['baseline_eta'] == .0015625 and kw['g'] == .03
        return {'plan': module.plain(module.plan_native_zero_diagnostic_dimensions(4, 601, 216, 120)),
                'dyson_evaluations': 20, 'local_ratio': 200.}
    monkeypatch.setattr(module, 'study_eta_zero_diagnostic', study)
    return value, order


def test_exact_human_values_separate_from_immutable_baseline_and_pins():
    module = driver()
    value = profile(module)
    assert module.validate_profile(value) == module.DIAGNOSTIC
    assert value['diagnostic']['eta_lead_values'] == [.001, .005, .01]
    assert value['map']['eta_lead'] == .0015625
    assert value['map']['eta_device'] == 0
    assert len(value['runtime_inputs']) == 7
    assert value['harmonic_inputs'] == value['runtime_inputs'][-2:]
    assert all(row['binding']['physics']['eta_lead'] == .0015625 for row in value['checkpoints'])


@pytest.mark.parametrize('defect', ['eta', 'eta_order', 'baseline', 'device_eta', 'guard',
    'g', 'budget', 'raw_hash', 'raw_path', 'raw_size', 'raw_subset', 'pin_dup', 'pin_missing',
    'ancestry', 'schema'])
def test_changed_experiment_guard_pin_or_lineage_fails_before_loading(monkeypatch, defect):
    module = driver()
    value = profile(module)
    if defect == 'eta': value['diagnostic']['eta_lead_values'][0] = .002
    elif defect == 'eta_order': value['diagnostic']['eta_lead_values'].reverse()
    elif defect == 'baseline': value['diagnostic']['baseline_eta_lead'] = .001
    elif defect == 'device_eta': value['map']['eta_device'] = .001
    elif defect == 'guard': value['diagnostic']['identity_atol'] = 1e-9
    elif defect == 'g': value['map']['g'] = .05
    elif defect == 'budget': value['diagnostic']['max_total_dyson_evaluations'] = 181
    elif defect.startswith('raw_'):
        key = {'raw_hash': 'sha256', 'raw_path': 'path', 'raw_size': 'bytes', 'raw_subset': 'name'}[defect]
        value['harmonic_inputs'][0][key] = 'changed'
    elif defect == 'pin_dup': value['runtime_inputs'][-1] = copy.deepcopy(value['runtime_inputs'][-2])
    elif defect == 'pin_missing': value['runtime_inputs'].pop()
    elif defect == 'ancestry': value['checkpoints'][0]['binding']['resume_parent_iteration'] = 24
    else: value['schema'] = 'old schema'
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **k: pytest.fail('no field loading'))
    with pytest.raises(ValueError):
        module.run(value, {}, lambda *a: None, profile_sha256='a'*64)


def test_nine_cases_share_fixed_inputs_with_three_fresh_contact_pairs_and_once_auth(monkeypatch):
    module = driver()
    value, order = wiring(module, monkeypatch)
    result, events = {}, []
    module.run(value, result, lambda name, details=None: events.append(name), profile_sha256='a'*64)
    assert result['passed'] is True and result['converged'] is False
    assert result['completed_cases'] == result['started_cases'] == 9
    assert result['admitted_dyson_evaluation_bound'] == 180
    assert result['admitted_contraction_work_bound'] == 447897600000
    assert result['admitted_contact_base_evaluation_bound'] == 600000
    assert len(events) == 35  # main adds one diagnostic_closed
    assert events[:5] == ['diagnostic_parent_authentication_start', 'diagnostic_parent_authenticated',
        'harmonic_inputs_authenticated', 'harmonic_reconstructed', 'acoustic_audit_complete']
    assert [(row['index'], row['eta_lead']) for row in result['cases']] == [
        (index, eta) for index in (27, 28, 29) for eta in (.001, .005, .01)]
    for index in (27, 28, 29):
        assert order.count(('authenticate', index)) == 1
        assert order.count(('release', index)) == 1
        assert order.index(('authenticate', index)) < order.index(('study', index, .001))
        assert order.index(('study', index, .01)) < order.index(('release', index))
    assert result['scope']['old_contacts_used_computationally'] is False
    assert result['scope']['physical_guards_unchanged'] is True
    assert result['scope']['baseline_mapped_total_is_cross_eta_reference_only'] is True
    assert result['cropped_vertex_acoustic']['maximum_relative_residual'] == 10.
    assert all(row['diagnostic']['local_ratio'] == 200. for row in result['cases'])


def test_partial_eta_failure_preserves_first_case_and_stops_without_retries(monkeypatch):
    module = driver()
    value, order = wiring(module, monkeypatch)
    original = module.study_eta_zero_diagnostic
    def study(*args, **kw):
        if kw['eta_lead'] == .005:
            raise ValueError('independent spectral Green identity failure')
        return original(*args, **kw)
    monkeypatch.setattr(module, 'study_eta_zero_diagnostic', study)
    result = {}
    with pytest.raises(ValueError, match='identity'):
        module.run(value, result, lambda *a: None, profile_sha256='a'*64)
    assert result['started_cases'] == 2 and result['completed_cases'] == 1
    assert result['cases'][0]['completed'] is True
    assert result['cases'][1]['status'] == 'failed'
    assert result['passed'] is False and result['stop_reason'] == 'case_failed'
    assert ('release', 27) in order and ('authenticate', 28) not in order
    assert ('study', 27, .01) not in order


def test_contact_failure_preserves_partial_pair_and_never_loads_checkpoint(monkeypatch):
    module = driver()
    value, order = wiring(module, monkeypatch)
    def build(fields, lead, eta, config, record, deadline, clock):
        record['left'] = {'freshly_rebuilt': True}
        raise ValueError('right contact causality failure')
    monkeypatch.setattr(module, 'build_contacts', build)
    result = {}
    with pytest.raises(ValueError, match='causality'):
        module.run(value, result, lambda *a: None, profile_sha256='a'*64)
    assert result['contacts'] == [{'eta_lead': .001, 'left': {'freshly_rebuilt': True}, 'right': None}]
    assert result['started_cases'] == 0 and not any(row[0] == 'authenticate' for row in order if isinstance(row, tuple))


def test_time_admission_before_contacts_does_not_start_material_case(monkeypatch):
    module = driver()
    value, order = wiring(module, monkeypatch)
    ticks = iter([0., 0., 78900.])
    result = {}
    module.run(value, result, lambda *a: None, profile_sha256='a'*64, clock=lambda: next(ticks))
    assert result['passed'] is False and result['started_cases'] == 0
    assert result['stop_reason'] == 'time_admission_before_contacts'
    assert result['unbuilt_contact_eta'] == .001
    assert not order


def test_raw_harmonic_authentication_uses_byte_pins_and_stable_descriptors(tmp_path, monkeypatch):
    module = driver()
    inputs = []
    for name, data in (('fc2', b'fake-hdf5\0\xff'), ('crystal', b'crystal: synthetic\n')):
        path = tmp_path/name
        path.write_bytes(data)
        inputs.append({'name': name, 'path': str(path), 'bytes': len(data), 'sha256': module.sha(path)})
    monkeypatch.setattr(module._native, 'authenticate_json_inputs', lambda rows: [])
    value = {'runtime_inputs': inputs, 'harmonic_inputs': inputs}
    rows = module.authenticate_inputs(value)
    assert len(rows) == 2 and all(row['stat_before'] == row['stat_after'] for row in rows)
    value['harmonic_inputs'][0]['sha256'] = 'a'*64
    with pytest.raises(ValueError, match='SHA256'):
        module.authenticate_inputs(value)


def test_reconstruction_uses_actual_massweight_order_and_full_device_normalization(monkeypatch):
    from negfpy.anharmonic.momentum import UniformBZMesh
    module = driver()
    value = profile(module)
    mesh = UniformBZMesh((4, 1, 1))
    fields = SimpleNamespace(mesh=mesh, dynamical=np.zeros((4, 216, 216), dtype=np.complex128))
    crystal = SimpleNamespace(primitive=SimpleNamespace(masses=np.array([1., 4.])),
        supercell=SimpleNamespace(masses=np.array([1., 4.])), p2s=np.arange(2), s2p=np.arange(2))
    geometry = SimpleNamespace(n_principal_layer_dofs=24, principal_cells=1,
        basis_primitive_atoms=np.tile([0, 1], 4),
        layered_geometry_sha256=value['map']['expected_geometry_sha256'],
        fc2_content_sha256='1'*64, primitive_crystal_content_sha256='2'*64)
    monkeypatch.setattr(module, 'read_phono3py_crystal_mapping', lambda *a, **k: crystal)
    monkeypatch.setattr(module, 'read_phono3py_fc2', lambda *a, **k: object())
    monkeypatch.setattr(module, 'build_layered_device_mapping', lambda *a, **k: geometry)
    lead = object()
    monkeypatch.setattr(module, 'material_transverse_harmonic', lambda *a, **k:
        (SimpleNamespace(device=lambda q: q), lead))
    monkeypatch.setattr(module, 'assemble_device_dynamical_matrix', lambda *a:
        np.zeros((216, 216), dtype=np.complex128))
    returned_lead, translations, record = module.rebuild_harmonic(fields, value)
    assert returned_lead is lead
    assert translations.shape == (216, 3)
    assert np.allclose(translations.T@translations, np.eye(3))
    assert translations[3, 0] == 2*translations[0, 0]
    assert np.linalg.norm(translations[48:168, 0]) < 1.
    assert record['mass_minimum_source_units'] == 1. and record['mass_maximum_source_units'] == 4.
    assert record['saved_dynamical_comparison']['passed'] is True
    assert len(record['saved_dynamical_comparison']['rows']) == 4
    fields.dynamical[0, 0, 0] = 1.
    with pytest.raises(ValueError, match='does not match'):
        module.rebuild_harmonic(fields, value)
