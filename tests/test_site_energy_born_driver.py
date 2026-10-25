"""Focused common-input, one-fit and complete-energy/control driver checks."""
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
    path = Path(__file__).parents[1]/'scripts/run_site_energy_born.py'
    spec = importlib.util.spec_from_file_location('site_energy_born_driver_test', path)
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
        'support_model': module.SUPPORT['model'], 'model_identity_sha256': old.MODEL_IDENTITY,
        'support_configuration': module.SUPPORT, 'feedback_dofs': list(range(216)),
        'new_hamiltonian_trajectory': True,
    }
    return copy.deepcopy({
        'schema': 'negfpy.site-energy-born.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'support': module.SUPPORT, 'site_support': module.SITE_SUPPORT,
        'fit': module.FIT, 'diagnostic': module.DIAGNOSTIC, 'driver_timeout_seconds': 82800,
        'source_commit': 'a'*40, 'source_tree': 'b'*40,
        'runtime_inputs': module.RAW_INPUTS+[
            {'name': 'native2_manifest', 'path': module.CHECKPOINT_PATH+'/manifest.json',
             'bytes': module.CHECKPOINT_BYTES, 'sha256': module.CHECKPOINT_SHA256}],
        'checkpoint': {'iteration': 2, 'path': module.CHECKPOINT_PATH,
                       'manifest_sha256': module.CHECKPOINT_SHA256, 'binding': binding},
    })


def test_profile_preserves_physics_seed_and_full_internal_budget():
    module = driver()
    original = profile(module)
    config = module.validate_profile(original)
    assert config['max_dyson_evaluations'] == 20
    assert config['max_contraction_work'] == 1_600_000_000_000
    assert module.FIT['max_iterations'] == 500
    changed = copy.deepcopy(original)
    changed['map']['eta_lead'] = .005
    with pytest.raises(ValueError, match='Frozen map'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['checkpoint']['binding']['model_identity_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='Hamiltonian binding'):
        module.validate_profile(changed)


def test_material_loader_reads_each_ifc_once_and_preserves_physical_fit_source(monkeypatch):
    module = driver()
    cfg = profile(module)
    masses = np.ones(64)
    crystal = SimpleNamespace(supercell=SimpleNamespace(
        masses=masses, n_atoms=64, scaled_positions=np.zeros((64, 3)), lattice=np.eye(3)),
        p2s=np.array([17, 45]), s2p=np.repeat([17, 45], 32))
    raw_values = SimpleNamespace(shape=(64, 64, 64, 3, 3, 3))
    raw = SimpleNamespace(values=raw_values, masses=masses, compact=False,
                          units='eV/angstrom^3', mass_units='amu')
    certification = SimpleNamespace(passed=True, permutation_complete=True,
        source_fc3_content_sha256=module.FROZEN_MAP['expected_fc3_content_sha256'])
    identities = module._material.SOURCE_IDENTITIES
    geometry = SimpleNamespace(n_principal_layer_dofs=24, principal_cells=1,
        layered_geometry_sha256=cfg['map']['expected_geometry_sha256'],
        fc2_content_sha256=identities['source_fc2_sha256'],
        primitive_crystal_content_sha256=identities['primitive_crystal_sha256'])
    normalized = SimpleNamespace(content_sha256=identities['source_normalized_fc3_sha256'])
    calls = []

    def read_crystal(path, *, purpose):
        calls.append(purpose)
        return crystal

    monkeypatch.setattr(module._material, 'read_phono3py_crystal_mapping', read_crystal)
    monkeypatch.setattr(module._material, 'read_phono3py_fc2',
        lambda *a, **kw: calls.append('read-fc2') or 'fc2')
    monkeypatch.setattr(module._material, 'read_phono3py_fc3',
        lambda *a, **kw: calls.append('read-fc3') or raw)
    monkeypatch.setattr(module._material, 'build_layered_device_mapping', lambda *a, **kw: geometry)
    monkeypatch.setattr(module._material, 'validate_fc3', lambda source: certification)
    monkeypatch.setattr(module._material, 'convert_fc3_to_translational', lambda *a, **kw: 'translated')
    monkeypatch.setattr(module._material, 'mass_normalize_translational_fc3', lambda *a, **kw: normalized)
    norm, actual_geometry, actual_crystal, actual_values, classes, record = module.load_material_for_study(cfg)
    assert norm is normalized and actual_geometry is geometry and actual_crystal is crystal
    assert actual_values is raw_values and classes is crystal.s2p
    assert calls == ['fc2', 'read-fc2', 'fc3', 'read-fc3']
    assert record['force_constant_units'] == 'eV/angstrom^3'
    assert record['raw_fc3_modified'] is False and record['source_asr_repaired'] is False


def test_one_fitted_model_supplies_complete_and_cropped_vertices_with_common_green(monkeypatch):
    module = driver()
    cfg = profile(module)
    mesh, grid, sigma, translations = object(), object(), object(), np.eye(216, 3)
    raw_values, classes = object(), object()
    crystal = SimpleNamespace(supercell=SimpleNamespace(scaled_positions='positions', lattice='lattice'))
    fit = {'parameter_count': 728, 'iterations': 40, 'stop_code': 2, 'bounded_solver_converged': True}
    fitted = SimpleNamespace(fit_report={'fit': fit, 'residual': {'relative_frobenius': .0214}}, geometry={})
    fit_calls, crop_calls, raw_calls = [], [], []
    vertices = {key: object() for key in module.MODEL_KEYS}

    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module._static, 'load_static', lambda *a: (
        mesh, grid, 'crop-reference', 'dynamical', ['left', 'right'], {}, {}))
    monkeypatch.setattr(module, 'load_material_for_study', lambda p: (
        'normalized', 'geometry', crystal, raw_values, classes,
        {'force_constant_units': 'eV/angstrom^3', 'mass_units': 'amu'}))
    monkeypatch.setattr(module, 'rigid_translations', lambda geometry: (translations, {}))
    monkeypatch.setattr(module, 'load_fixed_input', lambda *a: (sigma, {}, {}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda records: None)
    monkeypatch.setattr(module._static, 'release', lambda arrays: None)

    def fit_model(*args, **kwargs):
        assert args == (raw_values, 'positions', 'lattice', classes)
        assert kwargs['neighbors'] == 4 and 'g' not in kwargs
        fit_calls.append(True)
        return fitted

    def raw_vertex(*args, **kwargs):
        assert args == (mesh, 'normalized', 'geometry')
        raw_calls.append(True)
        return SimpleNamespace(vertex=vertices['actual_ifc_reference'],
            metadata={'model_identity_sha256': module._acoustic.MODEL_IDENTITY})

    def fitted_vertex(*args, **kwargs):
        assert args == (mesh, fitted, crystal, 'geometry')
        assert kwargs['force_constant_units'] == 'eV/angstrom^3' and 'g' not in kwargs
        crop = kwargs['crop_derivatives']
        crop_calls.append(crop)
        key = 'fitted_derivative_crop' if crop else 'fitted_complete_energy'
        return SimpleNamespace(vertex=vertices[key], metadata={'crop_derivatives': crop})

    def study(*args, **kwargs):
        assert args[0] is grid and args[1] == vertices
        assert args[2:6] == ('dynamical', 'left', 'right', sigma)
        assert args[6] is translations
        np.testing.assert_array_equal(args[7], np.arange(48, 168))
        assert kwargs['g'] == .03 and kwargs['max_dyson_evaluations'] == 20
        assert kwargs['max_contraction_work'] == 1_600_000_000_000
        return {'dyson_evaluations': 20, 'models': dict.fromkeys(module.MODEL_KEYS, 'same-Green')}

    monkeypatch.setattr(module, 'build_crossing_feedback_vertex', raw_vertex)
    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.site_energy_model',
                        SimpleNamespace(fit_site_energy_model=fit_model))
    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.site_energy_boundary',
                        SimpleNamespace(build_site_energy_vertex=fitted_vertex))
    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.site_energy_born',
                        SimpleNamespace(study_site_energy_born=study))
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert fit_calls == [True] and raw_calls == [True] and crop_calls == [False, True]
    assert result['passed'] is True and result['completed_studies'] == 1
    assert result['scope']['surrogate_vertex_includes_raw_fit_remainder'] is False
    assert result['scope']['raw_actual_ifc_reference_retained'] is True
    assert result['scope']['new_surrogate_dyson_solved'] is False and result['scope']['scba'] is False
    assert events[-1] == 'site_energy_born_complete'


def test_main_preserves_fit_report_when_later_born_fails(monkeypatch, tmp_path):
    module = driver()
    source, output = tmp_path/'profile.json', tmp_path/'result.json'
    source.write_text(json.dumps(profile(module)))

    def fail(profile, result, emit, *, profile_sha256):
        result['fit_report'] = {'residual': {'relative_frobenius': .0214}}
        raise ValueError('original Green guard failed')

    monkeypatch.setattr(module, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['fit_report']['residual']['relative_frobenius'] == .0214
    assert result['failure_reason'] == 'ValueError: original Green guard failed'
    events = [json.loads(line)['event'] for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events == ['site_energy_born_failed', 'site_energy_born_closed']
