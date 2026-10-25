"""Focused raw class mapping, bounded fit and non-adoption driver checks."""
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
    path = Path(__file__).parents[1]/'scripts/run_local_energy_fit.py'
    spec = importlib.util.spec_from_file_location('local_energy_fit_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({
        'schema': 'negfpy.local-energy-fit.profile.v1', 'map': module.FROZEN_MAP,
        'resources': module.RESOURCES, 'runtime_inputs': module.RAW_INPUTS,
        'diagnostic': module.DIAGNOSTIC,
        'expected_source_fc3_sha256': module.EXPECTED_SOURCE_FC3_SHA256,
        'driver_timeout_seconds': 82800, 'source_commit': 'a'*40, 'source_tree': 'b'*40,
    })


def test_profile_freezes_fit_bounds_and_allows_only_raw_two_inputs():
    module = driver()
    original = profile(module)
    config = module.validate_profile(original)
    assert config['neighbors'] == 4 and config['max_iterations'] == 500
    assert config['fit_rtol'] == 1e-12
    assert [pin['name'] for pin in original['runtime_inputs']] == ['fc3', 'crystal']
    changed = copy.deepcopy(original)
    changed['diagnostic']['neighbors'] = 8
    with pytest.raises(ValueError, match='Frozen diagnostic'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['checkpoint'] = {'path': 'old-state'}
    with pytest.raises(ValueError, match='no trajectory inputs'):
        module.validate_profile(changed)


def test_loader_keeps_actual_representative_labels_and_source_values(monkeypatch):
    module = driver()
    cfg = profile(module)
    values = SimpleNamespace(shape=(64, 64, 64, 3, 3, 3))
    masses, classes = np.ones(64), np.repeat([17, 45], 32)
    positions, lattice = np.zeros((64, 3)), np.eye(3)
    crystal = SimpleNamespace(
        supercell=SimpleNamespace(masses=masses, n_atoms=64,
                                  scaled_positions=positions, lattice=lattice),
        p2s=np.array([17, 45]), s2p=classes)
    raw = SimpleNamespace(values=values, masses=masses, compact=False,
                          units='eV/angstrom^3', mass_units='amu')
    certification = SimpleNamespace(passed=True, permutation_complete=True,
        source_fc3_content_sha256=module.EXPECTED_SOURCE_FC3_SHA256)

    def read_crystal(path, *, purpose):
        assert purpose == 'fc3'
        return crystal

    monkeypatch.setattr(module._material, 'read_phono3py_crystal_mapping', read_crystal)
    monkeypatch.setattr(module._material, 'read_phono3py_fc3', lambda *a, **kw: raw)
    monkeypatch.setattr(module._material, 'validate_fc3', lambda source: certification)
    monkeypatch.setattr(module._material, 'read_phono3py_fc2',
        lambda *a, **kw: pytest.fail('Only rawFC3 and its crystal are computational inputs'))
    actual, actual_positions, actual_lattice, actual_classes, record = module.load_fit_source(cfg)
    assert actual is values and actual_positions is positions and actual_lattice is lattice
    assert actual_classes is classes
    assert record['center_class_labels'] == [17, 45] and record['center_class_counts'] == [32, 32]
    assert record['raw_fc3_modified'] is False and record['source_asr_repaired'] is False
    classes[32] = 17
    with pytest.raises(ValueError, match='two primitive-center classes'):
        module.load_fit_source(cfg)


def test_bounded_solver_stop_and_large_residual_are_reported_without_adoption(monkeypatch):
    module = driver()
    cfg = profile(module)
    values, positions, lattice, classes = object(), object(), object(), object()
    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module._material, 'validate_parent',
        lambda *a, **kw: pytest.fail('No parent material/checkpoint arrays may be loaded'))
    monkeypatch.setattr(module, 'load_fit_source', lambda p: (
        values, positions, lattice, classes, {'center_class_counts': [32, 32]}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda records: None)

    def study(*args, **kwargs):
        assert args == (values, positions, lattice, classes)
        assert kwargs['neighbors'] == 4 and kwargs['max_iterations'] == 500
        assert kwargs['fit_rtol'] == 1e-12
        assert kwargs['identity_rtol'] == 1e-8 and kwargs['identity_atol'] == 1e-11
        return {'fit': {'parameter_count': 728, 'iterations': 500, 'stop_code': 7,
                        'bounded_solver_converged': False, 'source_reconstruction_passed': True},
                'residual': {'total_frobenius': 1000000., 'explicitly_retained': True}}

    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.local_energy',
                        SimpleNamespace(study_local_energy_fit=study))
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert result['passed'] is True and result['completed_studies'] == 1
    assert result['study']['fit']['bounded_solver_converged'] is False
    assert result['scope']['fit_remainder_retained'] is True
    assert result['scope']['fitted_tensor_adopted_as_new_physics'] is False
    assert result['scope']['all_local_four_atom_clusters_spanned'] is False
    assert result['scope']['rotational_constraints_imposed'] is False
    assert result['scope']['green_evaluations'] == result['scope']['born_evaluations'] == 0
    assert events[-1] == 'local_energy_fit_complete'


def test_main_preserves_partial_fit_failure(monkeypatch, tmp_path):
    module = driver()
    source, output = tmp_path/'profile.json', tmp_path/'result.json'
    source.write_text(json.dumps(profile(module)))

    def fail(profile, result, emit, *, profile_sha256):
        result['study'] = {'geometry_complete': True, 'fit_iteration': 10}
        raise TimeoutError('local fit deadline')

    monkeypatch.setattr(module, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['study']['fit_iteration'] == 10
    assert result['failure_reason'] == 'TimeoutError: local fit deadline'
    events = [json.loads(line)['event'] for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events == ['local_energy_fit_failed', 'local_energy_fit_closed']
