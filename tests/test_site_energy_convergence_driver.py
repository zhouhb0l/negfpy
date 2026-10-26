"""Tiny continuation wiring, exact-model and selected/latest role checks."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1]/'scripts/run_site_energy_convergence.py'
    spec = importlib.util.spec_from_file_location('site_convergence_test_driver', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({'schema': 'negfpy.site-energy-convergence.profile.v1',
        'map': module.FROZEN_MAP, 'parent': module.PARENT, 'resources': module.RESOURCES,
        'runtime_inputs': module.RAW_INPUTS, 'fit': module.FIT, 'site_support': module.SITE_SUPPORT,
        'scba': module.SCBA, 'acceleration': module.ACCELERATION,
        'selection_policy': module.SELECTION, 'proposal_feasibility': module.FEASIBILITY,
        'seed': module.SEED, 'checkpoint_max_bytes': module.CHECKPOINT_MAX_BYTES,
        'min_free_bytes': module.MIN_FREE_BYTES, 'initial_state': module.INITIAL_STATE,
        'driver_timeout_seconds': 82800, 'source_commit': 'a'*40, 'source_tree': 'b'*40})


@dataclass
class TinyPlan:
    peak_owned_bytes: int = 1024


@dataclass
class TinyRecord:
    index: int
    sigma_passed: bool = False
    current_passed: bool = True
    converged: bool = False
    sigma_residuals: dict | None = None
    current_residuals: dict | None = None


def setup_run(module, monkeypatch):
    cfg = profile(module)
    active = np.array(module.SEED['binding']['feedback_dofs'])
    vertex = SimpleNamespace(full_ndof=216, ndof=132, active_dofs=active)
    mesh, grid = SimpleNamespace(n_qpoints=4), SimpleNamespace(n_points=601)
    contacts = [SimpleNamespace(device_indices=tuple(range(24))) for _ in range(2)]
    fitted = SimpleNamespace(fit_report={'residual': {'relative_frobenius': .0214147}}, geometry={})
    shape = (4, 601, 132, 132)
    def spectrum(value):
        # Readonly broadcast views of one synthetic scalar, never material-sized allocations.
        a = np.broadcast_to(np.array(value, dtype=np.complex128), shape)
        return SimpleNamespace(retarded=a, statistical=a, spectral=a)
    seed = SimpleNamespace(sigma_input=spectrum(.1), mapped_sigma=spectrum(.8), record={'index': 2})
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=256*1024**3))
    monkeypatch.setattr(module, 'authenticate_inputs', lambda *a: ([], {'model_descriptor': module.SEED['model_parameters']}))
    monkeypatch.setattr(module, 'load_static_operator', lambda *a: (mesh, grid, 'D216', contacts, {}, {}))
    monkeypatch.setattr(module, 'load_seed_model', lambda *a: (fitted, 'geometry', 'crystal',
        {'force_constant_units': 'eV/angstrom^3', 'mass_units': 'amu'}))
    metadata = {key: module.SEED['model_parameters'][key] for key in ('model_identity_sha256', 'energy_identity_sha256')}
    monkeypatch.setattr(module, 'build_site_energy_vertex', lambda *a, **kw: SimpleNamespace(vertex=vertex, metadata=metadata))
    monkeypatch.setattr(module, 'load_channel_checkpoint', lambda *a, **kw: seed)
    monkeypatch.setattr(module, 'plan_channel_accel', lambda *a, **kw: TinyPlan())
    monkeypatch.setattr(module, '_close_seed', lambda value: None)
    monkeypatch.setattr(module._operator._static, 'release', lambda value: None)
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda value: None)
    return cfg, seed, spectrum, metadata


def test_profile_keeps_all_scientific_thresholds_and_exact_seed():
    module = driver();cfg = profile(module)
    config = module.validate_profile(cfg)
    assert config.max_evaluations == 16 and config.mixing == .05
    assert (config.sigma_rtol, config.sigma_atol) == (1e-4, 1e-10)
    assert (config.current_rtol, config.current_atol_w) == (1e-3, 1e-14)
    assert (config.identity_rtol, config.identity_atol) == (1e-8, 1e-11)
    assert cfg['seed']['iteration'] == 2 and cfg['initial_state'] == 'native-channel-continuation'
    assert cfg['selection_policy']['component_growth_limit'] == 1
    changed = copy.deepcopy(cfg);changed['scba']['sigma_rtol'] = 1e-3
    with pytest.raises(ValueError, match='Frozen scba'):
        module.validate_profile(changed)
    changed = copy.deepcopy(cfg);changed['seed']['iteration'] = 3
    with pytest.raises(ValueError, match='Frozen seed'):
        module.validate_profile(changed)


def test_input_authentication_loads_only_small_saved_parameters_and_json(tmp_path):
    module = driver();params = tmp_path/'coefficients.npy';meta = tmp_path/'saved.json'
    coefficients = np.arange(12., dtype=np.float64).reshape(2, 6)
    np.save(params, coefficients);meta.write_text('{"name":"synthetic"}')
    rows = [{'name': name, 'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
             'bytes': path.stat().st_size} for name, path in [('model_coefficients', params), ('seed_result', meta)]]
    with ExitStack() as stack:
        records, documents = module.authenticate_inputs({'runtime_inputs': rows}, stack)
        assert len(records) == 2 and documents['seed_result'] == {'name': 'synthetic'}
        np.testing.assert_array_equal(documents['model_coefficients'], coefficients)
    rows[0]['sha256'] = 'f'*64
    with ExitStack() as stack, pytest.raises(ValueError, match='SHA256'):
        module.authenticate_inputs({'runtime_inputs': rows}, stack)


def test_restoration_never_loads_fc3_or_refits(monkeypatch):
    module = driver();cfg = profile(module);old = {'source_commit': module.SEED['binding']['source_commit'],
        'source_tree': module.SEED['binding']['source_tree'], 'map': module.FROZEN_MAP, 'fit': module.FIT}
    saved = {'passed': True, 'completed_evaluations': 2, 'settings': module.FROZEN_MAP,
        'fit_config': module.FIT, 'source_commit': old['source_commit'], 'source_tree': old['source_tree'],
        'profile_sha256': module.SEED['binding']['profile_sha256'],
        'model_parameters': module.SEED['model_parameters'], 'fit_report': {'historical': True},
        'fit_geometry': {'historical_geometry': True}, 'material': {'force_constant_units': 'eV/angstrom^3'}}
    coefficients = np.arange(728., dtype=float).reshape(2, 364)
    documents = {'seed_profile': old, 'seed_result': saved,
        'seed_manifest': {'binding': module.SEED['binding'], 'iteration': 2},
        'model_descriptor': module.SEED['model_parameters'], 'model_coefficients': coefficients}
    crystal = SimpleNamespace(supercell=SimpleNamespace(n_atoms=64, scaled_positions='positions', lattice='lattice'),
        s2p=np.repeat([0, 32], 32))
    monkeypatch.setattr(module._operator, 'rebuild_harmonic_geometry', lambda p: ('geometry', {}))
    monkeypatch.setattr(module._material, 'read_phono3py_crystal_mapping', lambda *a, **kw: crystal)
    def forbidden(*a, **kw):
        raise AssertionError('Old fit/raw FC3 must not execute')
    monkeypatch.setattr(module._base, 'fit_site_energy_model', forbidden)
    monkeypatch.setattr(module._material, 'read_phono3py_fc3', forbidden)
    def restore(values, *a, **kw):
        assert values is coefficients and kw['fit_report'] == saved['fit_report']
        assert a == ('positions', 'lattice', crystal.s2p)
        return 'restored-model'
    monkeypatch.setattr(module, 'restore_site_energy_model', restore)
    fitted, geometry, _, material = module.load_seed_model(cfg, documents, 10.)
    assert fitted == 'restored-model' and geometry == 'geometry'
    assert material['fitting_repeated'] is False and material['raw_fc3_loaded'] is False


def test_core_uses_exact_seed_pair_and_retains_rejected_latest_checkpoint(tmp_path, monkeypatch):
    module = driver();cfg, seed, spectrum, _ = setup_run(module, monkeypatch)
    saved = []
    def save(directory, **kw):
        saved.append(kw)
        assert kw['record']['controller_decision']['index'] == kw['iteration']
        assert kw['binding']['resume_parent_binding'] == module.SEED['binding']
        assert kw['binding']['model_parameters'] == module.SEED['model_parameters']
        assert kw['max_checkpoints'] == 16
        return {'path': f'fake/{kw["iteration"]}', 'manifest_sha256': 'e'*64}
    monkeypatch.setattr(module, 'save_channel_checkpoint', save)
    def core(*args, **kw):
        assert kw['seed_input'] is seed.sigma_input and kw['seed_mapped'] is seed.mapped_sigma
        assert kw['seed_index'] == 2 and 'sigma_initial' not in kw
        assert kw['config'].max_evaluations == 16 and kw['config'].mixing == .05
        assert kw['g'] == .03 and kw['selection'].mode == 'component-nonincreasing'
        assert kw['feasibility'].mode == 'psd-backtracking'
        states, decisions = [], []
        for index, accepted in ((3, True), (4, False)):
            proposal = {'kind': 'linear_warmup' if index == 3 else 'anderson1', 'anchor_index': 2 if index == 3 else 3}
            kw['on_proposal'](index, {'index': index, 'map_admitted': True})
            kw['on_start'](index, seed.sigma_input, proposal)
            record = TinyRecord(index)
            state = SimpleNamespace(index=index, sigma_input=seed.sigma_input, record=record,
                evaluation=SimpleNamespace(input_currents={}, checks={}, map_sigma=spectrum(.7)))
            decision = {'index': index, 'accepted_as_anchor': accepted, 'selected_anchor_index': 3}
            kw['on_evaluation'](state, decision);states.append(state);decisions.append(decision)
        return SimpleNamespace(converged=False, stop_reason='max_evaluations', failure_reason=None,
            history=tuple(state.record for state in states), state=states[0], last_evaluated_state=states[1],
            decisions=decisions)
    monkeypatch.setattr(module, 'run_channel_accel', core)
    result, events = {}, []
    module.run(cfg, result, lambda name, detail=None: events.append(name), tmp_path/'checkpoints', profile_sha256='f'*64)
    assert result['passed'] and not result['converged'] and len(saved) == 2
    assert result['selected_anchor']['index'] == 3 and result['last_evaluated_trial']['index'] == 4
    assert result['selected_anchor_count'] == 1 and result['rejected_trial_count'] == 1
    assert result['selected_anchor']['sigma_passed'] is False
    assert result['evaluated_records'][1]['controller_decision']['accepted_as_anchor'] is False
    assert events.count('checkpoint_saved') == 2 and events.count('proposal_screened') == 2


def test_model_identity_mismatch_stops_before_any_new_map(tmp_path, monkeypatch):
    module = driver();cfg, _, _, metadata = setup_run(module, monkeypatch)
    metadata['energy_identity_sha256'] = '0'*64
    monkeypatch.setattr(module, 'run_channel_accel', lambda *a, **kw: pytest.fail('No map allowed'))
    with pytest.raises(ValueError, match='energy or exact endpoint'):
        module.run(cfg, {}, lambda *a: None, tmp_path/'checkpoints', profile_sha256='f'*64)


def test_hard_map_failure_remains_failed_without_retry(tmp_path, monkeypatch):
    module = driver();cfg, seed, _, _ = setup_run(module, monkeypatch)
    calls = []
    def core(*a, **kw):
        calls.append(True)
        kw['on_start'](3, seed.sigma_input, {'kind': 'linear_warmup', 'anchor_index': 2})
        return SimpleNamespace(converged=False, stop_reason='evaluation_error', failure_reason='hard guard',
            history=(), state=None, last_evaluated_state=None, decisions=())
    monkeypatch.setattr(module, 'run_channel_accel', core)
    result = {}
    module.run(cfg, result, lambda *a: None, tmp_path/'checkpoints', profile_sha256='f'*64)
    assert len(calls) == 1 and result['passed'] is False and result['failure_reason'] == 'hard guard'
    assert result['completed_evaluations'] == 0 and len(result['attempts']) == 1
