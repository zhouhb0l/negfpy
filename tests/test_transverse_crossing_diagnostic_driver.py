"""Focused driver checks: frozen physics, evaluated-only inputs, honest failure tail."""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1] / 'scripts/run_transverse_crossing_diagnostic.py'
    spec = importlib.util.spec_from_file_location('crossing_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    cases, pins = [], copy.deepcopy(module.RAW_INPUTS)
    for iteration in (28, 27, 29):
        path = f'/remote/fixed/iteration-{iteration:06d}'
        digest = module.CASE_MANIFESTS[iteration]
        cases.append({'iteration': iteration, 'path': path, 'manifest_sha256': digest,
                      'binding': {'physics': copy.deepcopy(module.FROZEN_MAP),
                                  'representation': 'retarded-statistical-spectral-v1'}})
        pins.append({'name': f'checkpoint{iteration}', 'path': path + '/manifest.json',
                     'bytes': 100, 'sha256': digest})
    return copy.deepcopy({'schema': 'negfpy.transverse-crossing-diagnostic.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'diagnostic': module.DIAGNOSTIC, 'runtime_inputs': pins, 'checkpoints': cases,
        'source_commit': 'a'*40, 'source_tree': 'b'*40})


def test_profile_keeps_eta_and_fixed_case_identity():
    module = driver()
    original = profile(module)
    assert module.validate_profile(original) == module.DIAGNOSTIC
    changed_eta = copy.deepcopy(original)
    changed_eta['map']['eta_lead'] = .005
    with pytest.raises(ValueError, match='Frozen map differs'):
        module.validate_profile(changed_eta)
    changed_case = copy.deepcopy(original)
    changed_case['checkpoints'][0]['manifest_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='fixed native input identity'):
        module.validate_profile(changed_case)


def test_fixed_input_loads_evaluated_channels_never_mapped_images(monkeypatch):
    module = driver()
    case = profile(module)['checkpoints'][0]
    names = ['sigma_input_' + component for component in module.COMPONENTS]
    artifacts = {name: {'role': name} for name in names}
    artifacts.update({'sigma_mapped_' + component: {'forbidden': True}
                      for component in module.COMPONENTS})
    manifest = {'schema': 'negfpy.transverse-channel-scba-checkpoint.v1',
                'representation': 'retarded-statistical-spectral-v1', 'iteration': 28,
                'binding': case['binding'], 'accepted_restart': False,
                'record': {'sigma_input_evaluated': True}, 'artifacts': artifacts}
    arrays = {name: np.zeros((1, 1, 1, 1), dtype=np.complex128) for name in names}
    reads = []

    def json_input(stack, path, filename, digest):
        reads.append((path, filename, digest))
        return manifest

    def array_input(stack, path, descriptors, expected, budget, authentication):
        assert list(descriptors) == names
        assert descriptors == {name: artifacts[name] for name in names}
        assert expected == {name: ((4, 601, 120, 120), 'complex128') for name in names}
        assert path == Path(case['path'])
        authentication['evaluated_only'] = True
        return arrays

    monkeypatch.setattr(module, '_json_verified', json_input)
    monkeypatch.setattr(module, '_arrays_verified', array_input)
    with ExitStack() as stack:
        spectrum, borrowed, authentication = module.load_fixed_input(case, stack)
    assert reads == [(case['path'], 'manifest.json', case['manifest_sha256'])]
    assert borrowed is arrays and authentication == {'evaluated_only': True}
    assert spectrum.retarded is arrays[names[0]]
    assert spectrum.statistical is arrays[names[1]]
    assert spectrum.spectral is arrays[names[2]]


def test_main_retains_completed_case_count_on_actual_failure(monkeypatch, tmp_path):
    module = driver()
    source, output = tmp_path/'profile.json', tmp_path/'result.json'
    source.write_text('{}\n')

    def fail(profile, result, emit, **kwargs):
        result['completed_cases'] = 1
        emit('crossing_case_complete', {'iteration': 28})
        raise ValueError('Original physical guard failure')

    monkeypatch.setattr(module, 'run', fail)
    monkeypatch.setattr(sys, 'argv', ['diagnostic', '--profile', str(source), '--output', str(output)])
    assert module.main() == 1
    result = json.loads(output.read_text())
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert result['passed'] is False and result['completed_cases'] == 1
    assert result['failure_reason'] == 'ValueError: Original physical guard failure'
    assert [row['event'] for row in events] == [
        'crossing_case_complete', 'crossing_diagnostic_failed', 'crossing_diagnostic_closed']
    assert events[-1]['details'] == {'passed': False, 'completed_cases': 1}
