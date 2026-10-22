"""Tiny integration checks for the raw-channel material comparison driver."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_material_map import PackedMapSpectrum
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex


def driver():
    path = Path(__file__).parents[1]/'scripts/run_transverse_channels_comparison.py'
    spec = importlib.util.spec_from_file_location('channels_comparison_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(module):
    profile = {'schema': 'negfpy.transverse-channels-comparison.profile.v1',
               'map': dict(module.PHYSICS), 'parent': {'root': '/unused',
               'profile_sha256': 'a'*64, 'result_sha256': 'b'*64,
               'source_commit': 'c'*40, 'source_tree': 'd'*40},
               'source_commit': 'e'*40, 'source_tree': 'f'*40,
               'parent_max_bytes': 1024**2,
               'comparison': {'kind': 'raw-cached-harmonic-channel-born', 'born_calls': 1,
                'dyson_evaluations': 0, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
                'workers': 8, 'external_tile': 16, 'bin_tile': 1,
                'max_bytes': 16*1024**2, 'max_contraction_work': 1000000}}
    grid = SignedFrequencyGrid(.025, 1)
    mesh = UniformBZMesh((1, 1, 1))
    vertex = PackedTransverseCubicVertex(mesh, np.ones((1, 1, 1, 1, 1)), np.array([0]), 1)
    contour = np.full((1, 3, 1, 1), -2j)
    zero = np.zeros_like(contour)
    # Independent finite convolution: valid input-pair counts are 2,3,2.
    expected = (-2j*grid.integration_weight*.03**2*np.array([2, 3, 2])).reshape(1, 3, 1, 1)
    fields = SimpleNamespace(grid=grid, mesh=mesh, vertex=vertex,
                             input_green=PackedMapSpectrum(zero, contour, contour.copy()),
                             mapped_sigma=PackedMapSpectrum(zero.copy(), expected, expected.copy()),
                             binding={'tiny_fixture': True})
    return profile, fields


def test_tiny_driver_complete_once_preserves_scope_and_coupling(monkeypatch):
    module = driver()
    profile, fields = fixture(module)
    calls = []
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: calls.append(kw) or fields)
    before = fields.input_green.lesser.copy()
    events = []
    result = {}
    module.run(profile, result, lambda name, details=None: events.append(name), profile_sha256='0'*64)
    assert len(calls) == 1 and calls[0]['expected_source_commit'] == 'c'*40
    assert result['passed'] and result['completed'] and not result['converged']
    assert result['born_calls'] == 1 and result['dyson_evaluations'] == 0
    assert not result['scope']['repair_accepted'] and not result['scope']['accepted_restart']
    assert result['scope']['inherited_input_defects_retained']
    assert result['comparison']['exported_lesser_vs_saved']['error'] < 1e-19
    assert result['comparison']['spectral_vs_saved_difference']['error'] == 0
    assert all(not row['acceptance'] for row in result['comparison'].values())
    assert events == ['parent_authentication_start', 'parent_authenticated', 'channels_prepared',
                      'channel_born_start', 'channel_born_complete', 'comparison_complete']
    np.testing.assert_array_equal(before, fields.input_green.lesser)


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('map', 'g', .02), ('map', 'eta_lead', .001),
    ('comparison', 'born_calls', 2), ('comparison', 'dyson_evaluations', 1),
    ('comparison', 'identity_rtol', 1e-6), ('comparison', 'max_bytes', True),
])
def test_profile_rejected_before_authentication(monkeypatch, section, key, value):
    module = driver()
    profile, _ = fixture(module)
    profile[section][key] = value
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('read'))
    with pytest.raises(ValueError):
        module.run(profile, {}, lambda *args: None, profile_sha256='0'*64)


def test_memory_guard_precedes_kernel_and_channel_allocation(monkeypatch):
    module = driver()
    profile, fields = fixture(module)
    profile['comparison']['max_bytes'] = 1
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    monkeypatch.setattr(module, 'StatisticalSpectral', lambda *a: pytest.fail('allocation'))
    monkeypatch.setattr(module, 'cubic_born_channels_packed_fft', lambda *a, **kw: pytest.fail('kernel'))
    with pytest.raises(ValueError, match='resources'):
        module.run(profile, {}, lambda *args: None, profile_sha256='0'*64)


def test_raw_inconsistency_is_reported_not_projected():
    module = driver()
    channels = module.StatisticalSpectral(np.array([[[[1+2j]]]]), np.array([[[[.3+0.1j]]]]))
    checks = module.channel_checks(channels, UniformBZMesh((1, 1, 1)))
    assert checks['statistical_antihermiticity']['error'] == 2
    assert checks['spectral_antihermiticity']['error'] == .6
    assert checks['spectral_reflection']['error'] > .6
    assert all(not row['acceptance'] for row in checks.values())
