"""Focused operator-only frozen-input driver wiring checks."""
from __future__ import annotations

import copy
import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1]/'scripts/run_transverse_soft_source.py'
    spec = importlib.util.spec_from_file_location('soft_source_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    old = module._acoustic
    binding = {
        'source_commit': old._OLD_SOURCE, 'source_tree': old._OLD_TREE,
        'profile_sha256': old._OLD_PROFILE, 'physics': module.FROZEN_MAP,
        'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
        'parent_profile_sha256': module.PARENT['profile_sha256'],
        'parent_result_sha256': module.PARENT['result_sha256'],
        'support_model': old.SUPPORT['model'], 'model_identity_sha256': old.MODEL_IDENTITY,
        'support_configuration': old.SUPPORT, 'feedback_dofs': list(range(216)),
        'new_hamiltonian_trajectory': True,
    }
    return copy.deepcopy({
        'schema': 'negfpy.transverse-soft-source.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'diagnostic': module.DIAGNOSTIC, 'driver_timeout_seconds': 82800,
        'source_commit': 'a'*40, 'source_tree': 'b'*40,
        'runtime_inputs': module.RAW_INPUTS+[
            {'name': 'native2_manifest', 'path': module.CHECKPOINT_PATH+'/manifest.json',
             'bytes': module.CHECKPOINT_BYTES, 'sha256': module.CHECKPOINT_SHA256}],
        'checkpoint': {'iteration': 2, 'path': module.CHECKPOINT_PATH,
                       'manifest_sha256': module.CHECKPOINT_SHA256, 'binding': binding},
    })


def test_frozen_profile_forbids_fc3_physics_change_and_scba():
    module = driver()
    original = profile(module)
    assert module.validate_profile(original)['max_dyson_evaluations'] == 12
    assert [pin['name'] for pin in original['runtime_inputs']] == [
        'fc2', 'crystal', 'parent_profile', 'parent_result', 'native2_manifest']
    changed = copy.deepcopy(original)
    changed['runtime_inputs'].insert(0, module._acoustic.RAW_INPUTS[0])
    with pytest.raises(ValueError, match='Only harmonic inputs'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['map']['eta_lead'] = .005
    with pytest.raises(ValueError, match='Frozen map'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['scba'] = {'max_evaluations': 2}
    with pytest.raises(ValueError, match='cannot advance'):
        module.validate_profile(changed)


def test_static_loader_excludes_vertex_active_and_cached_fields(monkeypatch):
    module = driver()
    cfg = profile(module)
    names = ['dynamical']
    arrays = {'dynamical': np.zeros((1, 1, 1), dtype=complex)}
    contacts = {}
    for index, side in enumerate(('left', 'right')):
        names.extend(side+'_'+label for label in
                     ('device_indices', 'zero_limits', 'retarded', 'lesser', 'greater'))
        arrays[side+'_device_indices'] = np.array([index])
        for label in ('zero_limits', 'retarded', 'lesser', 'greater'):
            arrays[side+'_'+label] = np.zeros((1, 1, 1), dtype=complex)
        contacts[side] = {'device_indices': [index], 'temperature_kelvin': [305, 295][index],
                          'q_reports': [], 'checks': {}}
    artifacts = {name: {'shape': list(arrays[name].shape), 'dtype': str(arrays[name].dtype)}
                 for name in names}
    artifacts['vertex_before_g'] = {'must_not_be_loaded': True}
    artifacts['input_green_active_retarded'] = {'must_not_be_loaded': True}
    documents = {'parent_result': {'artifacts': artifacts, 'contacts': contacts}}
    monkeypatch.setattr(module._material, 'validate_parent', lambda *args: None)

    def verified(stack, path, descriptors, expected, cap, authentication):
        assert list(descriptors) == names and len(names) == 11
        assert expected['dynamical'] == ((4, 216, 216), 'complex128')
        return arrays

    monkeypatch.setattr(module, '_arrays_verified', verified)
    monkeypatch.setattr(module._static, 'TransverseSpectrum', lambda *a: a)
    monkeypatch.setattr(module._static, 'TransverseContact', lambda *a: a)
    monkeypatch.setattr(module._static, 'CompactTransverseContact', lambda *a: a)
    with ExitStack() as stack:
        mesh, grid, dynamical, loaded_contacts, loaded, _ = module.load_static_operator(
            cfg, documents, stack)
    assert mesh.n_qpoints == 4 and grid.n_points == 601
    assert loaded is arrays and dynamical is arrays['dynamical']
    assert len(loaded_contacts) == 2


def test_harmonic_rebuild_reads_no_fc3_and_verifies_geometry(monkeypatch):
    module = driver()
    cfg = profile(module)
    crystal = SimpleNamespace(supercell=SimpleNamespace(masses=np.array([1.])), p2s=[0], s2p=[0])
    geometry = SimpleNamespace(
        n_principal_layer_dofs=24, principal_cells=1,
        layered_geometry_sha256=module.FROZEN_MAP['expected_geometry_sha256'],
        fc2_content_sha256=module._material.SOURCE_IDENTITIES['source_fc2_sha256'],
        primitive_crystal_content_sha256=module._material.SOURCE_IDENTITIES['primitive_crystal_sha256'])
    calls = []

    def read_crystal(path, *, purpose):
        assert purpose == 'fc2'
        calls.append('crystal')
        return crystal

    monkeypatch.setattr(module._material, 'read_phono3py_crystal_mapping', read_crystal)
    monkeypatch.setattr(module._material, 'read_phono3py_fc2', lambda *a, **kw: calls.append('fc2') or 'fc2')
    monkeypatch.setattr(module._material, 'build_layered_device_mapping', lambda *a, **kw: geometry)
    monkeypatch.setattr(module._material, 'read_phono3py_fc3',
        lambda *a, **kw: pytest.fail('FC3 must not be read in the soft-source diagnostic'))
    actual, record = module.rebuild_harmonic_geometry(cfg)
    assert actual is geometry and calls == ['crystal', 'fc2']
    assert record['fc3_read'] is False and record['vertex_built'] is False
    geometry.fc2_content_sha256 = 'f'*64
    with pytest.raises(ValueError, match='harmonic geometry'):
        module.rebuild_harmonic_geometry(cfg)


def test_run_passes_unmodified_fields_and_bounded_zero_reference(monkeypatch):
    module = driver()
    cfg = profile(module)
    sigma, translations = object(), np.eye(216, 3)
    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module, 'load_static_operator',
        lambda *args: ('mesh', 'grid', 'dynamical', ['left', 'right'], {}, {}))
    monkeypatch.setattr(module, 'rebuild_harmonic_geometry', lambda p: ('geometry', {}))
    monkeypatch.setattr(module, 'rigid_translations', lambda geometry: (translations, {}))
    monkeypatch.setattr(module, 'load_fixed_input', lambda case, stack: (sigma, {}, {}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda records: None)
    monkeypatch.setattr(module._static, 'release', lambda arrays: None)

    def study(*args, **kwargs):
        assert args == ('grid', 'mesh', 'dynamical', 'left', 'right', sigma, translations)
        assert kwargs['include_zero_reference'] is True
        assert kwargs['max_dyson_evaluations'] == kwargs['max_svd_evaluations'] == 12
        assert kwargs['identity_rtol'] == 1e-8 and kwargs['identity_atol'] == 1e-11
        return {'dyson_evaluations': 12, 'svd_evaluations': 12}

    numerical = SimpleNamespace(plan_soft_source_dimensions=lambda *a, **kw: {'peak_owned_bytes': 1},
                                study_soft_source_diagnostic=study)
    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.transverse_soft_source', numerical)
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert result['passed'] is True and result['completed_studies'] == 1
    assert result['scope']['born_evaluations'] == 0 and result['scope']['scba'] is False
    assert result['scope']['operator_or_source_modified'] is False
    assert result['scope']['fc3_read_or_vertex_built'] is False
    assert events[-1] == 'soft_source_study_complete'
