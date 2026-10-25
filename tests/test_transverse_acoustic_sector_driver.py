"""Focused frozen-input, mass ordering and diagnostic wiring checks."""
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
    path = Path(__file__).parents[1]/'scripts/run_transverse_acoustic_sector.py'
    spec = importlib.util.spec_from_file_location('acoustic_sector_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    binding = {
        'source_commit': module._OLD_SOURCE, 'source_tree': module._OLD_TREE,
        'profile_sha256': module._OLD_PROFILE, 'physics': module.FROZEN_MAP,
        'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
        'parent_profile_sha256': module.PARENT['profile_sha256'],
        'parent_result_sha256': module.PARENT['result_sha256'],
        'support_model': module.SUPPORT['model'], 'model_identity_sha256': module.MODEL_IDENTITY,
        'support_configuration': module.SUPPORT, 'feedback_dofs': list(range(216)),
        'new_hamiltonian_trajectory': True,
    }
    return copy.deepcopy({
        'schema': 'negfpy.transverse-acoustic-sector.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'support': module.SUPPORT, 'diagnostic': module.DIAGNOSTIC,
        'driver_timeout_seconds': 82800, 'source_commit': 'a'*40, 'source_tree': 'b'*40,
        'runtime_inputs': module.RAW_INPUTS+[
            {'name': 'native2_manifest', 'path': module.CHECKPOINT_PATH+'/manifest.json',
             'bytes': module.CHECKPOINT_BYTES, 'sha256': module.CHECKPOINT_SHA256}],
        'checkpoint': {'iteration': 2, 'path': module.CHECKPOINT_PATH,
                       'manifest_sha256': module.CHECKPOINT_SHA256, 'binding': binding},
    })


def test_fixed_profile_keeps_physics_and_forbids_trajectory_update():
    module = driver()
    original = profile(module)
    assert module.validate_profile(original)['max_dyson_evaluations'] == 20
    changed = copy.deepcopy(original)
    changed['map']['eta_lead'] = .005
    with pytest.raises(ValueError, match='Frozen map'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['checkpoint']['binding']['model_identity_sha256'] = 'f'*64
    with pytest.raises(ValueError, match='Hamiltonian binding'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['resume'] = {'mixing': .05}
    with pytest.raises(ValueError, match='cannot advance'):
        module.validate_profile(changed)


def test_loader_authenticates_only_evaluated_native_arrays(monkeypatch):
    module = driver()
    cfg = profile(module)
    names = ['sigma_input_'+c for c in module.COMPONENTS]
    manifest = {
        'schema': 'negfpy.transverse-channel-scba-checkpoint.v1',
        'representation': 'retarded-statistical-spectral-v1', 'iteration': 2,
        'binding': cfg['checkpoint']['binding'], 'accepted_restart': False,
        'sigma_input_evaluated': True, 'mapped_sigma_dyson_evaluated': False,
        'record': {'sigma_input_evaluated': True},
        'artifacts': {name: {'descriptor': True} for name in names+['mapped_sigma_statistical']},
    }
    monkeypatch.setattr(module, '_json_verified', lambda *args: manifest)
    arrays = {name: np.zeros((1, 1, 1, 1), dtype=complex) for name in names}

    def verified(stack, path, descriptors, expected, cap, authentication):
        assert list(descriptors) == names
        assert all(v == ((4, 601, 216, 216), 'complex128') for v in expected.values())
        assert cap == 8*1024**3
        return arrays

    monkeypatch.setattr(module, '_arrays_verified', verified)
    with ExitStack() as stack:
        sigma, loaded, _ = module.load_fixed_input(cfg['checkpoint'], stack)
    assert loaded is arrays
    assert sigma.statistical is arrays['sigma_input_statistical']
    manifest['mapped_sigma_dyson_evaluated'] = True
    with ExitStack() as stack, pytest.raises(ValueError, match='field roles'):
        module.load_fixed_input(cfg['checkpoint'], stack)


def test_actual_mass_ordering_and_global_normalization():
    module = driver()
    geometry = SimpleNamespace(
        basis_primitive_atoms=np.array([1, 0]), principal_cells=1, n_principal_layer_dofs=6,
        _crystal=SimpleNamespace(primitive=SimpleNamespace(masses=np.array([1., 4.]))))
    translations, metadata = module.rigid_translations(geometry, n_device_layers=3)
    assert translations.shape == (18, 3)
    np.testing.assert_allclose(translations.T@translations, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(translations[::3, 0], np.sqrt(np.tile([4., 1.], 3)/15))
    assert metadata['n_atoms'] == 6 and metadata['normal_mode_eigenbasis'] is False


def test_run_uses_same_model_and_keeps_sector_analysis_out_of_feedback(monkeypatch):
    module = driver()
    cfg = profile(module)
    vertex = SimpleNamespace(full_ndof=216, active_dofs=np.arange(216))
    sigma = object()
    translations = np.eye(216, 3)
    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module._static, 'load_static',
        lambda *args: ('mesh', 'grid', 'crop-reference', 'dynamical', ['left', 'right'], {}, {}))
    monkeypatch.setattr(module._material, 'rebuild_material', lambda p: ('fc3', 'geometry', {}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda records: None)
    monkeypatch.setattr(module._static, 'release', lambda arrays: None)
    monkeypatch.setattr(module, 'rigid_translations', lambda geometry: (translations, {}))
    monkeypatch.setattr(module, 'load_fixed_input', lambda case, stack: (sigma, {}, {}))

    def build(*args, **kwargs):
        assert args == ('mesh', 'fc3', 'geometry')
        assert kwargs['reference_vertex'] == 'crop-reference'
        return SimpleNamespace(vertex=vertex, metadata={'model_identity_sha256': module.MODEL_IDENTITY})

    monkeypatch.setattr(module, 'build_crossing_feedback_vertex', build)

    def study(*args, **kwargs):
        assert args[:7] == ('grid', vertex, 'dynamical', 'left', 'right', sigma, translations)
        np.testing.assert_array_equal(args[7], np.arange(48, 168))
        assert kwargs['g'] == .03 and kwargs['max_dyson_evaluations'] == 20
        assert kwargs['identity_rtol'] == 1e-8 and kwargs['identity_atol'] == 1e-11
        return {'dyson_evaluations': 20, 'all_interference_terms_retained': True}

    numerical = SimpleNamespace(plan_acoustic_sector_dimensions=lambda *a, **kw: {'peak_owned_bytes': 1},
                                study_acoustic_sector_diagnostic=study)
    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.transverse_acoustic_sector', numerical)
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert result['passed'] is True and result['converged'] is False
    assert result['completed_cases'] == 1
    assert result['scope']['scba'] is False and result['scope']['mapped_images_loaded'] is False
    assert result['scope']['field_projection_or_repair'] is False
    assert events[-1] == 'acoustic_case_complete'
