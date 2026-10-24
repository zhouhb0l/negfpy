"""Frozen raw-source reconstruction and scalar-only driver wiring with tiny mocks."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1]/'scripts/run_transverse_boundary_cancellation.py'
    spec = importlib.util.spec_from_file_location('boundary_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({'schema': 'negfpy.transverse-boundary-cancellation.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'runtime_inputs': module.RUNTIME_INPUTS,
        'resources': module.RESOURCES, 'diagnostic': module.DIAGNOSTIC,
        'source_commit': 'a'*40, 'source_tree': 'b'*40})


def documents(module):
    p = {'schema': 'negfpy.transverse-material-born-map.profile.v1',
        'map': copy.deepcopy(module.FROZEN_MAP), 'runtime_inputs': copy.deepcopy(module.RUNTIME_INPUTS[:3]),
        'source_commit': module.PARENT['source_commit'], 'source_tree': module.PARENT['source_tree']}
    metadata = dict(module.SOURCE_IDENTITIES,
        source_fc3_sha256=module.FROZEN_MAP['expected_fc3_content_sha256'],
        layered_geometry_sha256=module.FROZEN_MAP['expected_geometry_sha256'],
        make_r0_average=True, weights_applied=True, symmetrized=False, boundary_corrected=False)
    r = {'schema': 'negfpy.transverse-material-born-map.v1', 'passed': True,
        'settings': copy.deepcopy(p['map']), 'source_inputs': copy.deepcopy(p['runtime_inputs']),
        'source_commit': p['source_commit'], 'source_tree': p['source_tree'],
        'cubic': {'full_ndof': 216, 'active_dofs': list(range(48, 168)), 'shape': [4, 4, 120, 120, 120],
            'coupling_applied': False, 'solver_units_before_g': True, 'all_joint_vertex_checks_passed': True,
            'metadata': metadata}, 'artifact_state_semantics': {'accepted_restart': False},
        'artifacts': copy.deepcopy(module.REFERENCE_ARRAYS)}
    return {'parent_profile': p, 'parent_result': r}


def wiring(module, monkeypatch):
    p = profile(module)
    normalized, geometry, reference = object(), object(), object()
    order = []
    source_records = [{'path': 'raw-fc3', 'stat_before': {'bytes': 5}}]
    refs = {'vertex_before_g': {'path': 'vertex', 'stat_before': {'bytes': 7}},
        'active_dofs': {'path': 'active', 'stat_before': {'bytes': 3}}}
    monkeypatch.setattr(module, 'authenticate_inputs',
        lambda *a: (source_records, documents(module)))
    monkeypatch.setattr(module, 'authenticate_reference', lambda *a: (reference, refs))
    monkeypatch.setattr(module, 'rebuild_material',
        lambda *a: (normalized, geometry, {'fresh_raw_fc3': True}))
    monkeypatch.setattr(module, 'verify_unchanged', lambda rows: order.append(('stable', rows)))

    def audit(mesh, actual, mapping, **kwargs):
        assert actual is normalized and mapping is geometry
        assert kwargs['reference_vertex'] is reference
        assert mesh.shape == (4, 1, 1)
        assert kwargs['n_device_layers'] == 9
        assert kwargs['interaction_layers'] == (2, 3, 4, 5, 6)
        assert kwargs['max_geometry_evaluations'] == 2000000
        assert kwargs['max_accumulation_work'] == 1000000000
        assert kwargs['max_workspace_bytes'] == 2147483648
        assert kwargs['scale'].omega0 == module.OMEGA0_RAD_S
        assert kwargs['scale'].hbar == module.HBAR_J_S
        order.append('audit')
        return {'records': [{'complete_residual': 100.} for _ in range(36)],
            'residuals_are_diagnostic_only': True}
    monkeypatch.setattr(module, 'audit_transverse_boundary_cancellation', audit)
    return p, order


@pytest.mark.parametrize('defect', ['schema', 'g', 'eta', 'crop', 'r0', 'guard', 'time',
    'work', 'queue', 'cpus', 'memory', 'parent', 'raw_hash', 'raw_path', 'pin_missing', 'source'])
def test_changed_physics_pin_resource_or_cap_fails_before_file_loading(monkeypatch, defect):
    module = driver()
    p = profile(module)
    if defect == 'schema': p['schema'] = 'old'
    elif defect == 'g': p['map']['g'] = .05
    elif defect == 'eta': p['map']['eta_lead'] = .005
    elif defect == 'crop': p['map']['interaction_layers'] = [1, 2, 3, 4, 5]
    elif defect == 'r0': p['map']['make_r0_average'] = False
    elif defect == 'guard': p['diagnostic']['identity_rtol'] = 1e-7
    elif defect == 'time': p['diagnostic']['max_seconds'] += 1
    elif defect == 'work': p['diagnostic']['max_accumulation_work'] += 1
    elif defect == 'queue': p['resources']['execution_queue'] = 'q2'
    elif defect == 'cpus': p['resources']['cpus'] = 64
    elif defect == 'memory': p['resources']['memory_gb'] = 400
    elif defect == 'parent': p['parent']['result_sha256'] = 'c'*64
    elif defect == 'raw_hash': p['runtime_inputs'][0]['sha256'] = 'c'*64
    elif defect == 'raw_path': p['runtime_inputs'][0]['path'] = '/wrong'
    elif defect == 'pin_missing': p['runtime_inputs'].pop()
    else: p['source_commit'] = 'not a commit'
    monkeypatch.setattr(module, 'authenticate_inputs', lambda *a: pytest.fail('no file loading'))
    with pytest.raises(ValueError):
        module.run(p, {}, lambda *a: None, profile_sha256='c'*64)


@pytest.mark.parametrize('defect', ['setting', 'input', 'source', 'normalized', 'r0',
    'coupling', 'restart', 'reference_hash', 'crop'])
def test_reference_binding_rejects_wrong_material_role_or_identity(defect):
    module = driver()
    docs = documents(module)
    r = docs['parent_result']
    if defect == 'setting': r['settings']['eta_lead'] = .005
    elif defect == 'input': r['source_inputs'][0]['sha256'] = 'd'*64
    elif defect == 'source': r['source_tree'] = 'd'*40
    elif defect == 'normalized': r['cubic']['metadata']['source_normalized_fc3_sha256'] = 'd'*64
    elif defect == 'r0': r['cubic']['metadata']['make_r0_average'] = False
    elif defect == 'coupling': r['cubic']['coupling_applied'] = True
    elif defect == 'restart': r['artifact_state_semantics']['accepted_restart'] = True
    elif defect == 'reference_hash': r['artifacts']['vertex_before_g']['file_sha256'] = 'd'*64
    else: r['cubic']['active_dofs'][0] = 47
    with pytest.raises(ValueError):
        module.validate_parent(profile(module), docs)


def test_actual_material_and_independent_reference_wired_once_no_convergence_claim(monkeypatch):
    module = driver()
    p, order = wiring(module, monkeypatch)
    events, result = [], {}
    module.run(p, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert result['passed'] is True and result['converged'] is False
    assert result['completed_records'] == 36
    assert result['scope']['accepted_restart'] is False
    assert result['scope']['ifc_repair'] is False
    assert result['scope']['green_evaluations'] == result['scope']['born_evaluations'] == 0
    assert result['scope']['saved_reference_arrays_loaded'] == 2
    assert result['scope']['checkpoint_arrays_loaded'] == 0
    assert result['boundary_audit']['records'][0]['complete_residual'] == 100.
    assert events == ['boundary_input_authentication_start', 'boundary_inputs_authenticated',
        'boundary_reference_authenticated', 'boundary_material_rebuilt', 'boundary_audit_start',
        'boundary_audit_complete']
    assert order[0] == 'audit'
    assert [row['path'] for row in order[1][1]] == ['raw-fc3', 'vertex', 'active']


def test_hard_audit_failure_stops_without_retry_and_checks_bound_files(monkeypatch):
    module = driver()
    p, order = wiring(module, monkeypatch)
    def fail(*a, **kw):
        order.append('audit-failed')
        raise ValueError('cropped/reference mismatch')
    monkeypatch.setattr(module, 'audit_transverse_boundary_cancellation', fail)
    result = {}
    with pytest.raises(ValueError, match='cropped/reference'):
        module.run(p, result, lambda *a: None, profile_sha256='c'*64)
    assert order[0] == 'audit-failed' and len(order) == 2
    assert result['passed'] is False and result['started_audit'] is True
    assert result['completed_records'] == 0


def test_expired_admission_loads_no_raw_material_or_audit(monkeypatch):
    module = driver()
    p, order = wiring(module, monkeypatch)
    monkeypatch.setattr(module, 'rebuild_material', lambda *a: pytest.fail('expired material rebuild'))
    clock = iter([0., 80000.])
    result = {}
    module.run(p, result, lambda *a: None, profile_sha256='c'*64, clock=lambda: next(clock))
    assert result['passed'] is False and result['started_audit'] is False
    assert result['stop_reason'] == 'time_admission_before_material_reconstruction'
    assert len(order) == 1 and order[0][0] == 'stable'


def test_fresh_raw_workflow_keeps_certification_r0_mass_units_and_source_identities(monkeypatch):
    module = driver()
    p = profile(module)
    order = []
    c2 = SimpleNamespace(supercell=SimpleNamespace(masses=np.ones(1)), p2s='p2', s2p='s2')
    c3 = SimpleNamespace(supercell=SimpleNamespace(masses=np.ones(1)*2), p2s='p3', s2p='s3')
    geometry = SimpleNamespace(n_principal_layer_dofs=24, principal_cells=1,
        layered_geometry_sha256=p['map']['expected_geometry_sha256'],
        fc2_content_sha256=module.SOURCE_IDENTITIES['source_fc2_sha256'],
        primitive_crystal_content_sha256=module.SOURCE_IDENTITIES['primitive_crystal_sha256'])
    raw, translated = object(), object()
    certification = SimpleNamespace(passed=True, permutation_complete=True,
        source_fc3_content_sha256=p['map']['expected_fc3_content_sha256'])
    normalized = SimpleNamespace(content_sha256=module.SOURCE_IDENTITIES['source_normalized_fc3_sha256'])
    monkeypatch.setattr(module, 'read_phono3py_crystal_mapping',
        lambda path, purpose: c2 if purpose == 'fc2' else c3)
    def fc2(path, masses, **kwargs):
        assert masses is c2.supercell.masses and kwargs == {'p2s_map': 'p2', 's2p_map': 's2', 'compact': False}
        order.append('fc2')
        return 'actual_fc2'
    monkeypatch.setattr(module, 'read_phono3py_fc2', fc2)
    def mapping(crystal, actual, **kwargs):
        assert crystal is c2 and actual == 'actual_fc2'
        assert np.array_equal(kwargs['transport_supercell_matrix'], p['map']['transport_matrix'])
        return geometry
    monkeypatch.setattr(module, 'build_layered_device_mapping', mapping)
    def fc3(path, masses, **kwargs):
        assert masses is c3.supercell.masses and kwargs == {'p2s_map': 'p3', 's2p_map': 's3', 'compact': False}
        order.append('fc3')
        return raw
    monkeypatch.setattr(module, 'read_phono3py_fc3', fc3)
    monkeypatch.setattr(module, 'validate_fc3', lambda actual: certification if actual is raw else None)
    def convert(actual, crystal, **kwargs):
        assert actual is raw and crystal is c3 and kwargs == {'make_r0_average': True}
        order.append('r0')
        return translated
    monkeypatch.setattr(module, 'convert_fc3_to_translational', convert)
    def normalize(actual, crystal, **kwargs):
        assert actual is translated and crystal is c3
        assert kwargs == {'source_certification': certification, 'output_units': 'si'}
        order.append('normalize')
        return normalized
    monkeypatch.setattr(module, 'mass_normalize_translational_fc3', normalize)
    actual, mapping, record = module.rebuild_material(p)
    assert actual is normalized and mapping is geometry
    assert order == ['fc2', 'fc3', 'r0', 'normalize']
    assert record['source_normalized_fc3_sha256'] == normalized.content_sha256
    assert record['coupling_applied'] is False
    normalized.content_sha256 = 'd'*64
    with pytest.raises(ValueError, match='mass-normalized'):
        module.rebuild_material(p)


def test_metadata_output_rejects_arrays_and_nonfinite_scalars():
    module = driver()
    with pytest.raises(TypeError):
        module.plain({'material': np.ones((1, 1))})
    with pytest.raises(ValueError):
        module.plain({'residual': float('nan')})
    assert module.plain({'a': np.float64(2.), 'b': (True, 3)}) == {'a': 2., 'b': [True, 3]}


def test_five_input_authentication_hashes_once_and_parses_only_two_jsons(tmp_path, monkeypatch):
    module = driver()
    p = profile(module)
    rows = []
    for index, name in enumerate(('fc3', 'fc2', 'crystal', 'parent_profile', 'parent_result')):
        payload = json.dumps({'role': name}).encode() if name.startswith('parent_') else b'not-json'
        path = tmp_path/str(index)
        path.write_bytes(payload)
        rows.append({'name': name, 'path': str(path), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()})
    p['runtime_inputs'] = rows
    opened = []
    original = module._open_verified
    def verify(stack, path, digest, size):
        opened.append(str(path))
        return original(stack, path, digest, size)
    monkeypatch.setattr(module, '_open_verified', verify)
    with ExitStack() as stack:
        records, docs = module.authenticate_inputs(p, stack)
        module.verify_unchanged(records)
    assert opened == [row['path'] for row in rows]
    assert len(opened) == len(set(opened)) == 5
    assert docs == {name: {'role': name} for name in ('parent_profile', 'parent_result')}
    assert all(row['stat_before'] == row['stat_after'] == row['stat_after_diagnostic'] for row in records)


def test_reference_authentication_loads_exactly_two_arrays_and_closes_mappings(monkeypatch):
    module = driver()
    p, docs = profile(module), documents(module)
    closed = []
    arrays = {name: SimpleNamespace(_mmap=SimpleNamespace(close=lambda name=name: closed.append(name)))
        for name in module.REFERENCE_ARRAYS}
    arrays['active_dofs'].tolist = lambda: list(range(48, 168))
    calls = []
    def load(stack, root, descriptors, expected, max_bytes, authentication):
        calls.append((root, descriptors, expected, max_bytes))
        assert set(descriptors) == set(expected) == {'vertex_before_g', 'active_dofs'}
        assert max_bytes == 536870912
        authentication.update({name: {'fresh': True} for name in expected})
        return arrays
    monkeypatch.setattr(module, '_arrays_verified', load)
    vertex = object()
    def packed(mesh, coefficients, addresses, ndof, **kwargs):
        assert coefficients is arrays['vertex_before_g'] and addresses is arrays['active_dofs']
        assert ndof == 216 and mesh.shape == (4, 1, 1)
        return vertex
    monkeypatch.setattr(module, 'PackedTransverseCubicVertex', packed)
    with ExitStack() as stack:
        actual, auth = module.authenticate_reference(p, docs, stack)
    assert actual is vertex and set(auth) == set(module.REFERENCE_ARRAYS)
    assert len(calls) == 1 and set(closed) == set(module.REFERENCE_ARRAYS)


def test_main_preserves_actual_failure_partial_scalar_result_and_nonzero_exit(tmp_path, monkeypatch):
    module = driver()
    source, output = tmp_path/'profile.json', tmp_path/'result.json'
    source.write_text(json.dumps(profile(module)))
    def failed(p, result, emit, **kwargs):
        result.update(completed_records=4, scalar_partial={'source': 'actual IFC3'})
        emit('boundary_audit_start', {'maximum_records': 36})
        raise ValueError('streamed crop/reference mismatch')
    monkeypatch.setattr(module, 'run', failed)
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['converged'] is False
    assert result['completed_records'] == 4 and result['scalar_partial']['source'] == 'actual IFC3'
    assert result['failure'] == {'type': 'ValueError', 'message': 'streamed crop/reference mismatch'}
    events = [json.loads(row) for row in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [row['event'] for row in events] == ['boundary_audit_start',
        'boundary_diagnostic_failed', 'boundary_diagnostic_closed']
    assert events[-1]['details']['passed'] is False
