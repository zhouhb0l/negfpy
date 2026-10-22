"""Focused selected-reference provenance and scope tests, using tiny mock fields."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_material_map import PackedMapSpectrum
from negfpy.anharmonic.transverse_material_scba import mix_packed_spectra


@pytest.fixture
def driver():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_transverse_zero_reference.py'))


def setup(driver, monkeypatch):
    shape = (4, 601, 1, 1)
    sigma = PackedMapSpectrum(*(np.full(shape, value+.25j) for value in (1, 2, 3)))
    mapped = PackedMapSpectrum(*(np.full(shape, value-.125j) for value in (4, 5, 6)))
    contact = SimpleNamespace(device_indices=(0,), spectrum=sigma)
    fields = SimpleNamespace(
        binding={'artifacts_freshly_authenticated': 19}, grid=SignedFrequencyGrid(.025, 300),
        mesh=UniformBZMesh((4, 1, 1)), vertex=SimpleNamespace(active_dofs=np.array([1])),
        dynamical=np.tile(np.eye(2, dtype=complex), (4, 1, 1)), left=contact, right=contact,
    )
    cfg = copy.deepcopy(driver['PHYSICS'])
    cfg.update(workers=8, external_tile=16, bin_tile=1,
               max_contraction_work=50_000_000_000_000, max_dyson_evaluations=2404)
    binding = {'physics': copy.deepcopy(cfg), 'parent_result_sha256': 'a'*64,
               'parent_profile_sha256': 'b'*64, 'profile_sha256': '3'*64,
               'source_commit': '4'*40, 'source_tree': '5'*40,
               'resume_checkpoint_sha256': '6'*64, 'resume_evaluated_iteration': 12}
    saved = SimpleNamespace(sigma_input=sigma, mapped_sigma=mapped, binding=copy.deepcopy(binding),
                            record={'index': 2, 'converged': False}, manifest_sha256='2'*64)
    candidate = mix_packed_spectra(sigma, mapped, .1)
    p = {
        'map': cfg, 'source_commit': 'e'*40, 'source_tree': 'f'*40,
        'restart': {'initial_mixing': .1, 'parent_root': '/unused-material',
                    'parent_result_sha256': 'a'*64, 'parent_profile_sha256': 'b'*64,
                    'parent_source_commit': 'c'*40, 'parent_source_tree': 'd'*40,
                    'checkpoint': {'path': '/unused-last-complete', 'manifest_sha256': '2'*64,
                                   'binding': binding}},
        'diagnostic': {'kind': 'zero-frequency-reference', 'q_index': 0, 'frequency_index': 300,
                       'max_dyson_cases': 2, 'max_refinements': 4,
                       'identity_rtol': 1e-8, 'identity_atol': 1e-11,
                       'expected_candidate_sha256': driver['spectrum_hashes'](candidate),
                       'resume_local_iteration': 2, 'resume_trajectory': 14,
                       'trajectory_candidate': 15, 'max_bytes': 16*1024**3},
        'parent_max_bytes': 8*1024**3, 'checkpoint_max_bytes': 4*1024**3,
        'driver_timeout_seconds': 82800,
    }
    seen = {'evaluations': 0}

    def parent(*args, **kwargs):
        seen['parent'] = (args, kwargs)
        return fields

    def checkpoint(*args, **kwargs):
        seen['checkpoint'] = (args, kwargs)
        return saved

    def evaluate(*args, **kwargs):
        seen['evaluations'] += 1
        seen['evaluate'] = (args, kwargs)
        return {
            'schema': 'negfpy.transverse-zero-frequency-reference.v1',
            'frequency_index': 300, 'omega_solver': 0.,
            'resources': {'zero_dyson_cases': 2, 'max_zero_dyson_cases': 2,
                          'inverse_refinements_per_case': 4, 'frequency_stack_allocated': False,
                          'owned_numeric_workspace_bound_bytes':
                          driver['zero_reference_workspace_bytes'](2, 1, 601)},
            'original_candidate': {'ordinary_complex128': {'failed_identity_ratio': 12.}},
            'principal_value': {'raw_difference_retained': True},
            'direct_pv_retarded_trial': {'changed_candidate': True, 'accepted': False},
            'nonfinite_fields': [], 'identity_forced': False, 'symmetry_projected': False,
            'born_evaluated': False, 'scba_state_completed': False,
            'field_outputs': False, 'accepted_restart': False,
        }

    run = driver['run']
    monkeypatch.setitem(run.__globals__, 'load_material_map_fields', parent)
    monkeypatch.setitem(run.__globals__, 'load_material_checkpoint', checkpoint)
    monkeypatch.setitem(run.__globals__, 'study_zero_frequency', evaluate)
    return run, p, fields, saved, seen


@pytest.mark.parametrize('section,key,value', [
    ('map', 'eta_lead', .00078125), ('diagnostic', 'identity_rtol', 1e-7),
    ('diagnostic', 'kind', 'full-grid'), ('diagnostic', 'q_index', 1),
    ('diagnostic', 'frequency_index', 299), ('diagnostic', 'max_dyson_cases', 3),
    ('diagnostic', 'max_refinements', 5), ('restart', 'initial_mixing', .2),
])
def test_unapproved_scope_rejected_before_material_reads(driver, monkeypatch, section, key, value):
    run, p, _, _, seen = setup(driver, monkeypatch)
    p[section][key] = value
    with pytest.raises(ValueError):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)
    assert 'parent' not in seen and seen['evaluations'] == 0


@pytest.mark.parametrize('component', ('retarded', 'lesser', 'greater'))
def test_failed_candidate_hash_prevents_study(driver, monkeypatch, component):
    run, p, _, _, seen = setup(driver, monkeypatch)
    p['diagnostic']['expected_candidate_sha256'][component] = '0'*64
    result = {}
    with pytest.raises(ValueError, match='candidate hashes differ'):
        run(p, result, lambda *args: None, profile_sha256='1'*64)
    assert result['candidate_hashes_match'] is False and not result['passed']
    assert seen['evaluations'] == 0


def test_reference_receives_only_selected_views_and_all_old_bindings(driver, monkeypatch):
    run, p, fields, saved, seen = setup(driver, monkeypatch)
    result, events = {}, []
    run(p, result, lambda *args: events.append(args), profile_sha256='1'*64)
    assert result['passed'] is True and result['completed'] is True
    assert result['converged'] is False and seen['evaluations'] == 1
    assert result['selected_points'] == 1 and result['dyson_cases_completed'] == 2
    assert result['selected_point'] == {
        'q_index': 0, 'q_address': [0, 0, 0], 'frequency_index': 300, 'omega': 0.,
    }
    args, kwargs = seen['evaluate']
    assert args[0] is fields.grid
    np.testing.assert_array_equal(args[1], fields.dynamical[0])
    np.testing.assert_array_equal(args[2], fields.vertex.active_dofs)
    assert args[3]['support_dofs'] == fields.left.device_indices
    expected = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, .1)
    for name in ('retarded', 'lesser', 'greater'):
        np.testing.assert_array_equal(args[5][name], getattr(expected, name)[0])
        np.testing.assert_array_equal(args[3][name], getattr(fields.left.spectrum, name)[0, 300])
        assert np.shares_memory(kwargs['sigma_input_spectrum_q'][name], getattr(saved.sigma_input, name))
        assert np.shares_memory(kwargs['mapped_sigma_spectrum_q'][name], getattr(saved.mapped_sigma, name))
    assert kwargs['max_refinements'] == 4 and kwargs['mixing'] == .1
    assert kwargs['max_bytes'] == p['diagnostic']['max_bytes']-result['memory_plan']['candidate_owned_bytes']
    assert result['resume_binding'] == saved.binding
    assert result['resume_binding']['source_commit'] != result['source_commit']
    assert seen['parent'][1]['expected_source_commit'] == p['restart']['parent_source_commit']
    assert seen['checkpoint'][1]['expected_binding'] == p['restart']['checkpoint']['binding']
    assert result['scope']['trial_retarded_slice_changed'] is True
    for key in ('accepted', 'accepted_restart', 'accepted_field_update', 'scba', 'born_evaluated',
                'converged_transport', 'frequency_or_q_convergence', 'production_solver_changed'):
        assert result['scope'][key] is False
    assert not {'checkpoints', 'final_currents', 'history', 'map_sigma', 'green'} & result.keys()
    assert [event[0] for event in events] == [
        'parent_authentication_start', 'parent_authenticated', 'checkpoint_authenticated',
        'reference_candidate_ready', 'zero_reference_start', 'zero_reference_complete',
    ]


def test_candidate_workspace_guard_precedes_mixing(driver, monkeypatch):
    run, p, _, _, seen = setup(driver, monkeypatch)
    p['diagnostic']['max_bytes'] = 1
    monkeypatch.setitem(run.__globals__, 'mix_packed_spectra',
                        lambda *args: pytest.fail('guard did not precede candidate allocation'))
    with pytest.raises(ValueError, match='before mixing'):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)
    assert seen['evaluations'] == 0


@pytest.mark.parametrize('index,converged', [(1, False), (2, True)])
def test_only_last_nonconverged_checkpoint_allowed(driver, monkeypatch, index, converged):
    run, p, _, saved, seen = setup(driver, monkeypatch)
    saved.record = {'index': index, 'converged': converged}
    with pytest.raises(ValueError, match='last nonconverged completed'):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)
    assert seen['evaluations'] == 0


@pytest.mark.parametrize('key,value', [
    ('nonfinite_fields', ['reference.invalid']), ('identity_forced', True),
    ('resources', {'zero_dyson_cases': 1}),
])
def test_invalid_or_incomplete_reference_cannot_pass(driver, monkeypatch, key, value):
    run, p, _, _, seen = setup(driver, monkeypatch)
    original = run.__globals__['study_zero_frequency']

    def invalid(*args, **kwargs):
        captured = original(*args, **kwargs)
        captured[key] = value
        return captured

    monkeypatch.setitem(run.__globals__, 'study_zero_frequency', invalid)
    result = {}
    with pytest.raises(ValueError, match='bounded zero-frequency reference'):
        run(p, result, lambda *args: None, profile_sha256='1'*64)
    assert not result['passed'] and seen['evaluations'] == 1


def test_main_preserves_actual_failure_and_closes(driver, monkeypatch, tmp_path):
    run, p, _, _, seen = setup(driver, monkeypatch)

    def singular(*args, **kwargs):
        seen['evaluations'] += 1
        raise np.linalg.LinAlgError('controlled zero-slice singular operator')

    monkeypatch.setitem(run.__globals__, 'study_zero_frequency', singular)
    source, output = tmp_path/'profile.json', tmp_path/'zero-reference-result.json'
    source.write_text(json.dumps(p))
    monkeypatch.setattr('sys.argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert driver['main']() == 1
    result = json.loads(output.read_text())
    assert result['schema'] == 'negfpy.transverse-zero-reference.v1'
    assert result['passed'] is False and result['completed'] is False
    assert result['failure']['type'] == 'LinAlgError'
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events[-2:]] == ['zero_reference_failed', 'zero_reference_closed']
    assert seen['evaluations'] == 1
