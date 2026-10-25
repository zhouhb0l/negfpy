"""Focused exact-zero, sparse-support and immutable-checkpoint driver checks."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def driver():
    path = Path(__file__).parents[1]/'scripts/run_site_energy_feedback.py'
    spec = importlib.util.spec_from_file_location('site_energy_feedback_driver_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile(module):
    return copy.deepcopy({
        'schema': 'negfpy.site-energy-feedback.profile.v1', 'map': module.FROZEN_MAP,
        'parent': module.PARENT, 'resources': module.RESOURCES, 'runtime_inputs': module.RAW_INPUTS,
        'fit': module.FIT, 'site_support': module.SITE_SUPPORT, 'scba': module.SCBA,
        'initial_state': 'fresh-zero', 'checkpoint_max_bytes': module.CHECKPOINT_MAX_BYTES,
        'min_free_bytes': module.MIN_FREE_BYTES, 'driver_timeout_seconds': 82800,
        'source_commit': 'a'*40, 'source_tree': 'b'*40,
    })


@dataclass
class TinyPlan:
    peak_owned_bytes: int = 1024


def setup_run(module, monkeypatch):
    cfg = profile(module)
    active = np.arange(36, 168)
    vertex = SimpleNamespace(full_ndof=216, ndof=132, active_dofs=active)
    mesh, grid = SimpleNamespace(n_qpoints=4), SimpleNamespace(n_points=601)
    contact = SimpleNamespace(device_indices=tuple(range(24)))
    crystal = SimpleNamespace(supercell=SimpleNamespace(scaled_positions='positions', lattice='lattice'))
    fitted = SimpleNamespace(fit_report={
        'fit': {'parameter_count': 728, 'iterations': 40, 'stop_code': 2, 'bounded_solver_converged': True},
        'residual': {'relative_frobenius': .0214147, 'outside_floor_relative_frobenius': .0199395}},
        geometry={}, coefficients=np.arange(728, dtype=np.float64).reshape(2, 364))
    monkeypatch.setattr(module.shutil, 'disk_usage', lambda path: SimpleNamespace(free=100*1024**3))
    monkeypatch.setattr(module._material, 'authenticate_inputs', lambda p, s: ([], {}))
    monkeypatch.setattr(module, 'load_static_operator', lambda *a: (
        mesh, grid, 'dynamical', [contact, contact], {}, {}))
    monkeypatch.setattr(module, 'load_material_for_study', lambda p: (
        'unused-normalized-reference', 'geometry', crystal, 'raw-physical-fc3', 'classes',
        {'force_constant_units': 'eV/angstrom^3', 'mass_units': 'amu'}))
    monkeypatch.setattr(module._material, 'verify_unchanged', lambda rows: None)
    monkeypatch.setattr(module._operator._static, 'release', lambda arrays: None)

    def fit_model(*args, **kwargs):
        assert args == ('raw-physical-fc3', 'positions', 'lattice', 'classes')
        assert kwargs['neighbors'] == 4 and 'g' not in kwargs
        return fitted

    def build(*args, **kwargs):
        assert args == (mesh, fitted, crystal, 'geometry')
        assert kwargs['crop_derivatives'] is False and 'g' not in kwargs
        return SimpleNamespace(vertex=vertex, metadata={
            'model_identity_sha256': 'c'*64, 'energy_identity_sha256': 'd'*64})

    monkeypatch.setattr(module, 'fit_site_energy_model', fit_model)
    monkeypatch.setattr(module, 'build_site_energy_vertex', build)

    def plan(*args, **kwargs):
        assert args == (4, 601, 216, 132, 24, 24)
        return TinyPlan()

    monkeypatch.setattr(module, 'plan_channel_scba', plan)
    return cfg, vertex, fitted


def test_profile_preserves_six_gates_and_forbids_old_state_import():
    module = driver()
    cfg = profile(module)
    config = module.validate_profile(cfg)
    assert config.max_evaluations == 2 and config.mixing == .05
    assert (config.sigma_rtol, config.sigma_atol) == (1e-4, 1e-10)
    assert (config.current_rtol, config.current_atol_w) == (1e-3, 1e-14)
    assert (config.identity_rtol, config.identity_atol) == (1e-8, 1e-11)
    assert len(cfg['runtime_inputs']) == 5
    changed = copy.deepcopy(cfg)
    changed['checkpoint'] = {'path': 'old-state2'}
    with pytest.raises(ValueError, match='no old checkpoint'):
        module.validate_profile(changed)
    changed = copy.deepcopy(cfg)
    changed['map']['g'] = .05
    with pytest.raises(ValueError, match='Frozen map'):
        module.validate_profile(changed)


def test_zero_context_uses_global_support_addresses_without_full_padding():
    module = driver()
    active = np.array([0, 48, 167, 215])
    matrix = np.diag([1., 2., 3., 4.]).astype(complex)
    matrix[0, 1] = 5.
    values = matrix[None, None]
    spectrum = SimpleNamespace(retarded=values, statistical=values, spectral=values)
    context = module.zero_support_context(spectrum, active, 0)
    assert context['stored_active_dofs_count'] == 4 and context['outside_active_exact_zero'] is True
    assert context['central_dofs_count'] == context['outer_dofs_count'] == 2
    assert context['statistical']['full']['frobenius'] == pytest.approx(np.sqrt(55))
    assert context['statistical']['central']['frobenius'] == pytest.approx(np.sqrt(13))
    assert context['statistical']['outer']['frobenius'] == pytest.approx(np.sqrt(17))
    zero = module.zero_support_context(None, active, 0)
    assert zero['exact_zero_input'] is True and zero['statistical']['full']['frobenius'] == 0


@pytest.mark.parametrize('hard_failure', [False, True])
def test_core_loop_and_checkpoints_keep_exact_zero_and_evaluated_roles(
    monkeypatch, tmp_path, hard_failure,
):
    module = driver()
    cfg, vertex, _ = setup_run(module, monkeypatch)
    saved, calls = [], []
    monkeypatch.setattr(module, 'zero_support_context',
        lambda spectrum, active, zero: {'active_count': len(active), 'exact_zero_input': spectrum is None})

    def checkpoint(path, **kwargs):
        saved.append(kwargs)
        return {'path': str(path/'iteration-000001'), 'manifest_sha256': 'e'*64}

    monkeypatch.setattr(module, 'save_channel_checkpoint', checkpoint)
    record = {'index': 1, 'converged': False}
    state = SimpleNamespace(index=1, sigma_input=None, record=record,
        evaluation=SimpleNamespace(input_currents={'left': 1.}, checks={'original': 'unchanged'},
                                   map_sigma='new-unevaluated-image'))

    def solve(*args, **kwargs):
        calls.append(True)
        assert args[1] is vertex and args[2] == 'dynamical'
        assert kwargs['sigma_initial'] is None and kwargs['first_index'] == 1
        assert kwargs['g'] == .03 and kwargs['config'].max_evaluations == 2
        assert kwargs['config'].mixing == .05
        kwargs['on_start'](1, None)
        kwargs['on_evaluation'](state)
        if hard_failure:
            kwargs['on_start'](2, 'fresh-core-mixed-candidate')
        return SimpleNamespace(state=state, history=[record], converged=False,
            stop_reason='evaluation_error' if hard_failure else 'max_evaluations',
            failure_reason='original guard failed' if hard_failure else None)

    monkeypatch.setattr(module, 'run_channel_scba', solve)
    result, events = {}, []
    module.run(cfg, result, lambda event, details=None: events.append(event),
               tmp_path/'checkpoints', profile_sha256='f'*64)
    assert calls == [True] and result['passed'] is (not hard_failure)
    assert result['completed_evaluations'] == 1 and len(result['checkpoints']) == 1
    assert saved[0]['sigma_input'] is None and saved[0]['mapped_sigma'] == 'new-unevaluated-image'
    assert saved[0]['record']['mapped_sigma_dyson_evaluated'] is False
    binding = saved[0]['binding']
    assert binding['feedback_dofs'] == vertex.active_dofs.tolist()
    assert binding['full_device_dofs'] == 216 and binding['outside_feedback_dofs_exact_zero'] is True
    assert binding['initial_state'] == 'fresh-zero'
    assert binding['bulk_fit_residual']['relative_frobenius'] == .0214147
    assert binding['model_parameters'] == result['model_parameters']
    assert result['scope']['research_energy_parameters_persisted'] is True
    assert binding['production_ifc_adoption'] is False and result['scope']['accepted_restart'] is False
    assert events[-1] == 'scba_stopped'


def test_unconverged_bulk_fit_stops_before_any_full_map(monkeypatch, tmp_path):
    module = driver()
    cfg, _, fitted = setup_run(module, monkeypatch)
    fitted.fit_report['fit']['bounded_solver_converged'] = False
    monkeypatch.setattr(module, 'build_site_energy_vertex',
        lambda *a, **kw: pytest.fail('An unconverged declared bulk fit cannot create a trajectory'))
    monkeypatch.setattr(module, 'run_channel_scba',
        lambda *a, **kw: pytest.fail('No full map may start after bulk solver failure'))
    result = {}
    with pytest.raises(ValueError, match='fit must converge'):
        module.run(cfg, result, lambda *args: None, tmp_path/'checkpoints', profile_sha256='f'*64)
    assert result['passed'] is False and result['completed_evaluations'] == 0
    assert result['fit_report']['fit']['bounded_solver_converged'] is False


def test_exact_research_coefficients_are_preserved_exclusively_with_identity_binding(tmp_path):
    module = driver()
    cfg = profile(module)
    coefficients = np.arange(728, dtype=np.float64).reshape(2, 364)
    coefficients[0, 0] = -0.
    fitted = SimpleNamespace(coefficients=coefficients)
    descriptor = module.save_model_parameters(
        tmp_path, fitted, cfg, profile_sha256='f'*64,
        model_identity_sha256='c'*64, energy_identity_sha256='d'*64)
    path = tmp_path/'model-parameters/coefficients.npy'
    saved = np.load(path, allow_pickle=False)
    assert saved.tobytes() == coefficients.tobytes()
    assert descriptor['shape'] == [2, 364] and descriptor['dtype'] == 'float64'
    assert descriptor['array_bytes'] == 5824 and descriptor['file_bytes'] == path.stat().st_size
    assert descriptor['file_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert descriptor['source_commit'] == cfg['source_commit'] and descriptor['profile_sha256'] == 'f'*64
    assert descriptor['model_identity_sha256'] == 'c'*64 and descriptor['energy_identity_sha256'] == 'd'*64
    assert descriptor['research_trajectory_parameters'] is True and descriptor['actual_ifc_tensor'] is False
    assert json.loads((tmp_path/'model-parameters/descriptor.json').read_text()) == descriptor
    with pytest.raises(FileExistsError):
        module.save_model_parameters(tmp_path, fitted, cfg, profile_sha256='f'*64,
                                     model_identity_sha256='c'*64, energy_identity_sha256='d'*64)
