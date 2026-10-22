"""Mock solver integration verifies fresh pilot and channel checkpoint wiring."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_solver import TransverseCurrents

PACKAGE = Path(__file__).resolve().parents[3]/'reports/phonon-phonon-interaction/multiq-channel-feedback-20261022'


def driver():
    spec = importlib.util.spec_from_file_location('channel_pilot_driver',
        Path(__file__).parents[1]/'scripts/run_transverse_channel_scba.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def profile():
    return json.loads((PACKAGE/'profile-draft.json').read_text())


def test_fresh_driver_checkpoint_wiring(tmp_path, monkeypatch):
    module = driver()
    pf = profile()
    shape = (1, 3, 1, 1)
    mapped = ChannelSpectrum(np.zeros(shape, complex), np.full(shape, -1j), np.zeros(shape, complex))
    fields = SimpleNamespace(grid='grid', vertex='vertex', dynamical='dynamical', left='left',
                             right='right', binding={'test_parent': True})
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: fields)
    residual = {'error': 1., 'scale': 1., 'allowance': .01, 'ratio': 100.,
                'error_frobenius': 1., 'scale_frobenius': 1., 'relative_frobenius': 1.}
    current = {'error_w': 0., 'scale_w': 2e-9, 'allowance_w': 2.01e-12, 'ratio': 0.}
    record = {'index': 1, 'map_seconds': .1, 'elapsed_seconds': .2,
              'sigma_residuals': {k: dict(residual) for k in ('retarded', 'lesser', 'greater')},
              'channel_residuals': {k: dict(residual) for k in ('statistical', 'spectral')},
              'current_residuals': {k: dict(current) for k in ('contact_balance', 'collision_power', 'full_balance')},
              'sigma_passed': False, 'current_passed': True, 'converged': False}
    state = SimpleNamespace(index=1, sigma_input=None, record=record,
        evaluation=SimpleNamespace(map_sigma=mapped, input_currents=TransverseCurrents(2e-9, -2e-9, 0.),
                                   checks={'fixture_checks': {}}))
    called = []

    def solve(*args, **kwargs):
        assert args == ('grid', 'vertex', 'dynamical', 'left', 'right')
        assert 'sigma_initial' not in kwargs and kwargs['g'] == .03
        assert kwargs['config'].max_evaluations == 3 and kwargs['config'].mixing == .1
        called.append(True)
        kwargs['on_start'](1, None)
        kwargs['on_evaluation'](state)
        return SimpleNamespace(state=state, history=(record,), converged=False,
                               stop_reason='max_seconds', failure_reason=None, plan={'tiny': True})
    monkeypatch.setattr(module, 'run_channel_scba', solve)
    events = []
    result = {}
    module.run(pf, result, lambda name, details=None: events.append(name), tmp_path/'checkpoints',
               profile_sha256='a'*64)
    assert called == [True] and result['passed'] and not result['converged']
    assert result['attempts'] == [{'index': 1, 'input_sigma_is_zero': True,
                                  'sigma_sha256': None, 'dyson_evaluated': False}]
    assert result['scope']['fresh_green_evaluations'] and not result['scope']['cached_green_used']
    assert not result['scope']['historical_checkpoint_imported']
    checkpoint = result['checkpoints'][0]
    assert checkpoint['manifest']['representation'] == 'retarded-statistical-spectral-v1'
    assert checkpoint['manifest']['input_sigma_is_zero']
    assert len(checkpoint['manifest']['artifacts']) == 6
    assert checkpoint['manifest']['record']['currents']['left'] == 2e-9
    assert events == ['parent_authentication_start', 'parent_authenticated', 'channel_pilot_start',
                      'scba_evaluation_start', 'scba_evaluation_complete', 'checkpoint_saved', 'scba_stopped']


@pytest.mark.parametrize(('section', 'key', 'value'), [
    ('map', 'eta_lead', .001), ('map', 'g', .02), ('scba', 'mixing', .2),
    ('scba', 'identity_rtol', 1e-6), ('scba', 'max_evaluations', 4),
    ('root', 'initial_state', 'old-contour-checkpoint'),
    ('root', 'checkpoint_max_bytes', 1), ('root', 'restart', {}),
])
def test_invalid_profile_before_material_load(monkeypatch, section, key, value):
    module = driver()
    pf = profile()
    (pf if section == 'root' else pf[section])[key] = value
    monkeypatch.setattr(module, 'load_material_map_fields', lambda *a, **kw: pytest.fail('read'))
    with pytest.raises(ValueError):
        module.run(pf, {}, lambda *a: None, Path('/unused'), profile_sha256='a'*64)


def test_independent_hash_domain_and_exact_zero():
    module = driver()
    assert module.spectrum_hashes(None) is None
    c = np.full((1, 3, 1, 1), -1e20j)
    d = np.full_like(c, 1e-20j)
    hashes = module.spectrum_hashes(ChannelSpectrum(np.zeros_like(c), c, d))
    assert set(hashes) == {'retarded', 'statistical', 'spectral'}
    assert hashes['statistical'] != hashes['spectral']
