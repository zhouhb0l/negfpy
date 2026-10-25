"""Focused raw-source selection and explicit-remainder driver checks."""
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
    path = Path(__file__).parents[1]/'scripts/run_bulk_bond_energy.py'
    spec = importlib.util.spec_from_file_location('bulk_bond_energy_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({
        'schema': 'negfpy.bulk-bond-energy.profile.v1', 'map': module.FROZEN_MAP,
        'resources': module.RESOURCES, 'runtime_inputs': module.RAW_INPUTS,
        'diagnostic': module.DIAGNOSTIC,
        'expected_source_fc3_sha256': module.EXPECTED_SOURCE_FC3_SHA256,
        'driver_timeout_seconds': 82800, 'source_commit': 'a'*40, 'source_tree': 'b'*40,
    })


def test_profile_allows_only_two_raw_inputs_and_keeps_map_as_provenance():
    module = driver()
    original = profile(module)
    assert module.validate_profile(original)['max_atoms'] == 128
    assert [pin['name'] for pin in original['runtime_inputs']] == ['fc3', 'crystal']
    changed = copy.deepcopy(original)
    changed['runtime_inputs'].append(module._material.RUNTIME_INPUTS[1])
    with pytest.raises(ValueError, match='Frozen runtime_inputs'):
        module.validate_profile(changed)
    changed = copy.deepcopy(original)
    changed['checkpoint'] = {'path': 'old-state'}
    with pytest.raises(ValueError, match='no trajectory inputs'):
        module.validate_profile(changed)


def test_raw_loader_retains_physical_values_and_requires_source_certification(monkeypatch):
    module = driver()
    cfg = profile(module)
    values = SimpleNamespace(shape=(64, 64, 64, 3, 3, 3))
    masses = np.arange(64, dtype=float)+1
    positions, lattice = np.zeros((64, 3)), np.eye(3)
    crystal = SimpleNamespace(
        supercell=SimpleNamespace(masses=masses, n_atoms=64,
                                  scaled_positions=positions, lattice=lattice), p2s=[0], s2p=[0])
    raw = SimpleNamespace(values=values, masses=masses, compact=False,
                          units='eV/angstrom^3', mass_units='amu')
    certification = SimpleNamespace(passed=True, permutation_complete=True,
        source_fc3_content_sha256=module.EXPECTED_SOURCE_FC3_SHA256)

    def read_crystal(path, *, purpose):
        assert purpose == 'fc3'
        return crystal

    def read_fc3(path, given_masses, **kwargs):
        assert given_masses is masses and kwargs['compact'] is False
        return raw

    monkeypatch.setattr(module._material, 'read_phono3py_crystal_mapping', read_crystal)
    monkeypatch.setattr(module._material, 'read_phono3py_fc3', read_fc3)
    monkeypatch.setattr(module._material, 'validate_fc3', lambda source: certification)
    monkeypatch.setattr(module._material, 'read_phono3py_fc2',
        lambda *a, **kw: pytest.fail('The raw source diagnostic must not read FC2'))
    actual, actual_positions, actual_lattice, record = module.load_bulk_source(cfg)
    assert actual is values and actual_positions is positions and actual_lattice is lattice
    assert record['raw_fc3_modified'] is False and record['source_asr_repaired'] is False
    assert record['masses'] == masses.tolist()
    certification.source_fc3_content_sha256 = 'f'*64
    with pytest.raises(ValueError, match='source content'):
        module.load_bulk_source(cfg)


def test_large_explicit_remainder_is_retained_without_becoming_new_physics(monkeypatch):
    module = driver()
    cfg = profile(module)
    raw_values, positions, lattice = object(), object(), object()

    def authenticate(given_profile, stack):
        assert [pin['name'] for pin in given_profile['runtime_inputs']] == ['fc3', 'crystal']
        return [], {}

    monkeypatch.setattr(module._material, 'authenticate_inputs', authenticate)
    monkeypatch.setattr(module._material, 'validate_parent',
        lambda *a, **kw: pytest.fail('This diagnostic has no parent array inputs'))
    monkeypatch.setattr(module, 'load_bulk_source', lambda p: (
        raw_values, positions, lattice, {'force_constant_units': 'eV/angstrom^3'}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda records: None)

    def study(*args, **kwargs):
        assert args == (raw_values, positions, lattice)
        assert kwargs['identity_rtol'] == 1e-8 and kwargs['identity_atol'] == 1e-11
        return {'remainder': {'frobenius': 1000000., 'retained': True},
                'complete_reconstruction': True}

    monkeypatch.setitem(sys.modules, 'negfpy.anharmonic.bond_energy',
                        SimpleNamespace(study_bulk_bond_energy=study))
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event), profile_sha256='c'*64)
    assert result['passed'] is True and result['completed_studies'] == 1
    assert result['scope']['remainder_retained'] is True
    assert result['scope']['relative_part_adopted_as_new_physics'] is False
    assert result['scope']['finite_interaction_boundary_defined'] is False
    assert result['scope']['g_or_eta_used_numerically'] is False
    assert result['scope']['green_evaluations'] == result['scope']['born_evaluations'] == 0
    assert events[-1] == 'bond_study_complete'


def test_main_preserves_actual_partial_failure(monkeypatch, tmp_path):
    module = driver()
    source, output = tmp_path/'profile.json', tmp_path/'result.json'
    source.write_text(json.dumps(profile(module)))

    def fail(profile, result, emit, *, profile_sha256):
        result['study'] = {'first_root_complete': True, 'remainder_retained': True}
        raise TimeoutError('bounded decomposition deadline')

    monkeypatch.setattr(module, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['study']['first_root_complete'] is True
    assert result['failure_reason'] == 'TimeoutError: bounded decomposition deadline'
    events = [json.loads(line)['event'] for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert events == ['bond_study_failed', 'bond_diagnostic_closed']
