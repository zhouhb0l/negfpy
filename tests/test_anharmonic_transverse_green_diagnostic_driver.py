"""New diagnostic driver wiring checks using tiny mocked, nonmaterial fields."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.transverse_material_map import PackedMapSpectrum
from negfpy.anharmonic.transverse_material_scba import mix_packed_spectra


@pytest.fixture
def driver():
    return runpy.run_path(str(Path(__file__).parents[1] / 'scripts/run_transverse_green_diagnostic.py'))


def fields_and_saved():
    # Deliberately nonphysical values: these test provenance and call wiring only.
    shape = (1, 3, 1, 1)
    sigma = PackedMapSpectrum(*(np.full(shape, value+0.25j) for value in (1, 2, 3)))
    mapped = PackedMapSpectrum(*(np.full(shape, value-0.125j) for value in (4, 5, 6)))
    fields = SimpleNamespace(binding={'artifacts_freshly_authenticated': 19}, grid='grid',
                             vertex='vertex', dynamical='D', left='L', right='R')
    saved = SimpleNamespace(sigma_input=sigma, mapped_sigma=mapped,
                            record={'index': 2, 'converged': False}, manifest_sha256='2'*64)
    return fields, saved


def profile(driver, saved):
    cfg = copy.deepcopy(driver['PHYSICS'])
    cfg.update(workers=8, external_tile=16, bin_tile=1,
               max_contraction_work=50_000_000_000_000, max_dyson_evaluations=2404)
    binding = {'physics': copy.deepcopy(cfg), 'parent_result_sha256': 'a'*64,
               'parent_profile_sha256': 'b'*64, 'profile_sha256': '3'*64,
               'source_commit': '4'*40, 'source_tree': '5'*40,
               'resume_checkpoint_sha256': '6'*64, 'resume_evaluated_iteration': 12}
    saved.binding = copy.deepcopy(binding)
    candidate = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, 0.1)
    return {
        'map': cfg, 'source_commit': 'e'*40, 'source_tree': 'f'*40,
        'restart': {'initial_mixing': 0.1, 'parent_root': '/unused-material',
                    'parent_result_sha256': 'a'*64, 'parent_profile_sha256': 'b'*64,
                    'parent_source_commit': 'c'*40, 'parent_source_tree': 'd'*40,
                    'checkpoint': {'path': '/unused-last-complete', 'manifest_sha256': '2'*64,
                                   'binding': binding}},
        'diagnostic': {'identity_rtol': 1e-8, 'identity_atol': 1e-11,
                       'expected_candidate_sha256': driver['spectrum_hashes'](candidate),
                       'resume_local_iteration': 2, 'resume_trajectory': 14,
                       'trajectory_candidate': 15, 'max_bytes': 16*1024**3,
                       'max_dyson_evaluations': 2404},
        'parent_max_bytes': 8*1024**3, 'checkpoint_max_bytes': 4*1024**3,
        'driver_timeout_seconds': 82800,
    }


def setup(driver, monkeypatch, *, green_passed=False):
    run = driver['run']
    fields, saved = fields_and_saved()
    p = profile(driver, saved)
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
        return SimpleNamespace(
            completed=True, green_passed=green_passed, plan={'n_q': 4, 'n_frequency': 601},
            diagnostic_workspace_bytes=128, diagnostic_peak_owned_bytes=256,
            diagnostics={'all_points_observed': True, 'completed_points': 2404,
                         'expected_points': 2404,
                         'full_green_validation_error': None if green_passed else 'controlled guard failure'},
        )

    monkeypatch.setitem(run.__globals__, 'load_material_map_fields', parent)
    monkeypatch.setitem(run.__globals__, 'load_material_checkpoint', checkpoint)
    monkeypatch.setitem(run.__globals__, 'evaluate_packed_born_map', evaluate)
    return run, p, saved, seen


@pytest.mark.parametrize('section,key,value', [
    ('map', 'eta_lead', 0.00078125), ('map', 'g', 0.02),
    ('diagnostic', 'identity_rtol', 1e-7), ('diagnostic', 'identity_atol', 1e-10),
    ('diagnostic', 'trajectory_candidate', 16), ('diagnostic', 'resume_local_iteration', 1),
    ('diagnostic', 'max_dyson_evaluations', 2405),
    ('diagnostic', 'expected_candidate_sha256', {'retarded': '0'*64}),
    ('restart', 'initial_mixing', 0.2),
])
def test_invalid_profile_precedes_parent_reads(driver, monkeypatch, section, key, value):
    run, p, _, _ = setup(driver, monkeypatch)
    p[section][key] = value
    monkeypatch.setitem(run.__globals__, 'load_material_map_fields',
                        lambda *args, **kwargs: pytest.fail('invalid profile reached saved arrays'))
    with pytest.raises(ValueError):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)


@pytest.mark.parametrize('component', ('retarded', 'lesser', 'greater'))
def test_each_candidate_hash_rejects_before_evaluator(driver, monkeypatch, component):
    run, p, _, seen = setup(driver, monkeypatch)
    p['diagnostic']['expected_candidate_sha256'][component] = '0'*64
    result = {}
    events = []
    with pytest.raises(ValueError, match='candidate hashes differ'):
        run(p, result, lambda *args: events.append(args), profile_sha256='1'*64)
    assert seen['evaluations'] == 0
    assert result['candidate_hashes_match'] is False
    assert not result['passed']
    assert 'green_diagnostic_start' not in [event[0] for event in events]


@pytest.mark.parametrize('green_passed', (False, True))
def test_one_scoped_capture_separates_execution_from_green_result(driver, monkeypatch, green_passed):
    run, p, saved, seen = setup(driver, monkeypatch, green_passed=green_passed)
    result = {}
    events = []
    run(p, result, lambda *args: events.append(args), profile_sha256='1'*64)
    assert result['passed'] is True and result['completed'] is True
    assert result['green_passed'] is green_passed and result['failure_reason'] is None
    assert result['converged'] is False and seen['evaluations'] == 1
    args, kwargs = seen['evaluate']
    assert args == ('grid', 'vertex', 'D', 'L', 'R')
    assert kwargs['green_diagnostic_only'] is True
    assert kwargs['identity_rtol'] == 1e-8 and kwargs['identity_atol'] == 1e-11
    assert kwargs['max_dyson_evaluations'] == 2404 and kwargs['g'] == 0.03
    assert kwargs['max_bytes'] == p['diagnostic']['max_bytes']
    expected = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, 0.1)
    for name in ('retarded', 'lesser', 'greater'):
        np.testing.assert_array_equal(getattr(kwargs['sigma_input'], name), getattr(expected, name))
    assert result['candidate_sha256'] == p['diagnostic']['expected_candidate_sha256']
    assert result['scope']['diagnostic_only'] is True
    for name in ('born_evaluated', 'accepted', 'accepted_restart', 'accepted_field_update',
                 'scba', 'converged_transport', 'frequency_or_q_convergence'):
        assert result['scope'][name] is False
    assert not {'checkpoints', 'final_currents', 'history', 'map_sigma', 'green'} & result.keys()
    assert [event[0] for event in events] == [
        'parent_authentication_start', 'parent_authenticated', 'checkpoint_authenticated',
        'diagnostic_candidate_ready', 'green_diagnostic_start', 'green_diagnostic_complete',
    ]


def test_original_and_checkpoint_provenance_stay_bound_to_old_source(driver, monkeypatch):
    run, p, saved, seen = setup(driver, monkeypatch)
    result = {}
    run(p, result, lambda *args: None, profile_sha256='1'*64)
    assert seen['parent'][1] == {
        'expected_result_sha256': p['restart']['parent_result_sha256'],
        'expected_profile_sha256': p['restart']['parent_profile_sha256'],
        'expected_source_commit': p['restart']['parent_source_commit'],
        'expected_source_tree': p['restart']['parent_source_tree'],
        'expected_settings': p['map'], 'max_total_bytes': p['parent_max_bytes'],
    }
    assert seen['checkpoint'] == (('/unused-last-complete',), {
        'expected_manifest_sha256': '2'*64, 'expected_binding': p['restart']['checkpoint']['binding'],
        'max_total_bytes': p['checkpoint_max_bytes'],
    })
    assert result['resume_binding'] == saved.binding
    assert result['resume_binding']['source_commit'] != result['source_commit']
    assert result['resume_evaluated_iteration'] == 2 and result['resume_trajectory'] == 14
    assert result['trajectory_candidate'] == 15 and result['resume_checkpoint_sha256'] == '2'*64


@pytest.mark.parametrize('index,converged', [(1, False), (2, True)])
def test_wrong_checkpoint_state_rejects_before_evaluation(driver, monkeypatch, index, converged):
    run, p, saved, seen = setup(driver, monkeypatch)
    saved.record = {'index': index, 'converged': converged}
    with pytest.raises(ValueError, match='last nonconverged completed'):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)
    assert seen['evaluations'] == 0


def test_checkpoint_parent_mismatch_rejects_before_read(driver, monkeypatch):
    run, p, _, seen = setup(driver, monkeypatch)
    p['restart']['checkpoint']['binding']['parent_result_sha256'] = '0'*64
    with pytest.raises(ValueError, match='original material parent differs'):
        run(p, {}, lambda *args: None, profile_sha256='1'*64)
    assert 'parent' not in seen and seen['evaluations'] == 0


def test_incomplete_capture_cannot_pass(driver, monkeypatch):
    run, p, _, seen = setup(driver, monkeypatch)
    original = run.__globals__['evaluate_packed_born_map']

    def incomplete(*args, **kwargs):
        value = original(*args, **kwargs)
        value.diagnostics['completed_points'] = 2403
        value.diagnostics['all_points_observed'] = False
        return value

    monkeypatch.setitem(run.__globals__, 'evaluate_packed_born_map', incomplete)
    result = {}
    with pytest.raises(ValueError, match='complete Green diagnostic capture'):
        run(p, result, lambda *args: None, profile_sha256='1'*64)
    assert not result['passed'] and seen['evaluations'] == 1


def test_main_preserves_evaluator_error_and_failed_closure(driver, monkeypatch, tmp_path):
    run, p, _, seen = setup(driver, monkeypatch)

    def singular(*args, **kwargs):
        seen['evaluations'] += 1
        raise np.linalg.LinAlgError('controlled singular operator')

    monkeypatch.setitem(run.__globals__, 'evaluate_packed_born_map', singular)
    source = tmp_path/'profile.json'
    output = tmp_path/'green-diagnostic-result.json'
    source.write_text(json.dumps(p))
    monkeypatch.setattr('sys.argv', ['driver', '--profile', str(source), '--output', str(output)])
    assert driver['main']() == 1
    result = json.loads(output.read_text())
    assert result['passed'] is False and result['completed'] is False
    assert result['failure']['type'] == 'LinAlgError'
    assert 'controlled singular operator' in result['failure_reason']
    events = [json.loads(line) for line in (tmp_path/'events.jsonl').read_text().splitlines()]
    assert [event['event'] for event in events[-2:]] == ['green_diagnostic_failed', 'green_diagnostic_closed']
    assert seen['evaluations'] == 1
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'events.jsonl', 'green-diagnostic-result.json', 'profile.json',
    ]
