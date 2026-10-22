"""Focused continuation admission, detachment and diagnostic-only contracts.

Only synthetic fields are used. The separate resume-reference file exercises
six actual Dyson/Born/current evaluations against independent finite algebra.
"""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pytest
from test_anharmonic_transverse_channel_feedback import (
    _config,
    _constant,
    _fake,
    _fixture,
)

import negfpy.anharmonic.transverse_channel_feedback as feedback


def test_caller_mutation_in_start_callback_cannot_change_copied_candidate(monkeypatch):
    initial = _constant(.2)
    calls = _fake(monkeypatch)
    def on_start(index, candidate):
        assert index == 4
        for name in ('retarded', 'statistical', 'spectral'):
            getattr(initial, name)[:] = 7
            np.testing.assert_array_equal(getattr(candidate, name), .2)
            assert not getattr(candidate, name).flags.writeable
    result = feedback.run_channel_scba(*_fixture(), sigma_initial=initial, first_index=4,
                                      config=_config(max_evaluations=1), on_start=on_start)
    assert len(calls) == 1 and result.state.index == 4
    for name in ('retarded', 'statistical', 'spectral'):
        np.testing.assert_array_equal(getattr(result.state.sigma_input, name), .2)
        assert not np.shares_memory(getattr(initial, name), getattr(result.state.sigma_input, name))


def test_initial_copy_preserves_all_signed_zero_payload_bits(monkeypatch):
    initial = _constant(0)
    # Exercise every real/imaginary signed-zero pairing through a non-C view.
    bits = np.array([0, 1 << 63, 0, 0, 1 << 63, 0, 1 << 63, 1 << 63], dtype=np.uint64)
    arrays = []
    for offset in range(3):
        raw = np.tile(np.roll(bits, 2*offset), 15).view(np.complex128).reshape(3, 5, 2, 2)
        arrays.append(raw[..., ::-1])
    initial = feedback.ChannelSpectrum(*arrays)
    expected = [np.ascontiguousarray(value).tobytes() for value in arrays]
    _fake(monkeypatch)
    result = feedback.run_channel_scba(*_fixture(), sigma_initial=initial, first_index=4,
                                      config=_config(max_evaluations=1))
    for index, name in enumerate(('retarded', 'statistical', 'spectral')):
        copied = getattr(result.state.sigma_input, name)
        assert copied.flags.c_contiguous and not copied.flags.writeable
        assert copied.tobytes() == expected[index]
        assert not np.shares_memory(copied, arrays[index])


def test_resume_start_callback_failure_precedes_new_evaluation(monkeypatch):
    calls = _fake(monkeypatch)
    def failed(index, candidate):
        raise RuntimeError(f'candidate{index} provenance write failed')
    with pytest.raises(RuntimeError, match='candidate4 provenance'):
        feedback.run_channel_scba(*_fixture(), sigma_initial=_constant(.2), first_index=4,
                                 config=_config(max_evaluations=6), on_start=failed)
    assert calls == []


def test_resume_completed_callback_failure_prevents_next_map(monkeypatch):
    calls = _fake(monkeypatch)
    def failed(state):
        assert state.index == 4
        raise RuntimeError('native checkpoint publication failed')
    with pytest.raises(RuntimeError, match='checkpoint publication'):
        feedback.run_channel_scba(*_fixture(), sigma_initial=_constant(.2), first_index=4,
                                 config=_config(max_evaluations=6), on_evaluation=failed)
    assert len(calls) == 1


@pytest.mark.parametrize('bad', ['shape', 'nonfinite'])
def test_initial_channel_validation_precedes_detached_copy(monkeypatch, bad):
    initial = _constant(.2)
    if bad == 'shape':
        initial = feedback.ChannelSpectrum(*(getattr(initial, name)[:, :, :1, :1]
                                             for name in ('retarded', 'statistical', 'spectral')))
    else:
        initial.spectral[0, 0, 0, 0] = np.nan
    def forbidden(*args, **kwargs):
        raise AssertionError('invalid candidate must not be copied/evaluated')
    monkeypatch.setattr(feedback, 'mix_channel_spectra', forbidden)
    monkeypatch.setattr(feedback, 'evaluate_channel_born_map', forbidden)
    with pytest.raises(ValueError, match='shape|finite'):
        feedback.run_channel_scba(*_fixture(), sigma_initial=initial, first_index=4,
                                 config=_config(max_evaluations=6))


@pytest.mark.parametrize('bound', ['max_bytes', 'max_contraction_work', 'max_dyson_evaluations'])
def test_resume_budget_gate_precedes_initial_copy(monkeypatch, bound):
    def forbidden(*args, **kwargs):
        raise AssertionError('budget rejection must precede candidate copy/evaluation')
    monkeypatch.setattr(feedback, 'mix_channel_spectra', forbidden)
    monkeypatch.setattr(feedback, 'evaluate_channel_born_map', forbidden)
    with pytest.raises(ValueError, match=bound):
        feedback.run_channel_scba(*_fixture(), sigma_initial=_constant(.2), first_index=4,
                                 config=_config(max_evaluations=6, **{bound: 1}))


def test_count_only_six_map_plan_accounts_for_context_and_caller_candidate():
    config = feedback.ChannelSCBAConfig(max_evaluations=6)
    fresh = feedback.plan_channel_scba(4, 601, 216, 120, 24, 24, config=config)
    resumed = feedback.plan_channel_scba(4, 601, 216, 120, 24, 24,
                                         config=config, has_initial_sigma=True)
    component = 16 * 4 * 601 * 120**2
    assert resumed.peak_owned_bytes == fresh.peak_owned_bytes == 14_181_250_192
    assert resumed.peak_owned_bytes - 14_172_292_240 == 12 * 16 * 216**2
    assert resumed.nominal_caller_bytes - fresh.nominal_caller_bytes == 3 * component
    assert resumed.maximum_contraction_work == 6 * 81_537_269_760_000
    assert resumed.maximum_dyson_evaluations == 6 * 2404
    with pytest.raises(TypeError, match='bool'):
        feedback.plan_channel_scba(4, 601, 216, 120, 24, 24, has_initial_sigma=1)


def test_nonfinite_context_scalar_is_reported_without_new_guard():
    grid, vertex, dynamical, left, right = _fixture()
    identity = np.eye(3, dtype=complex)
    statistical = -1j * 1e308 * identity
    spectral = 1e-300 * identity
    context = feedback._zero_frequency_context(
        0, grid.zero_index, grid, vertex.mesh, identity, identity,
        statistical, spectral, statistical, spectral, dynamical[0], left, right,
        None, 1e-8, 1e-11,
    )
    assert context['diagnostic_only'] is True and context['available'] is False
    assert 'diagnostic_error' in context
    json.dumps(context, allow_nan=False)


def test_unavailable_context_does_not_prevent_actual_fresh_map(monkeypatch):
    unavailable = {'diagnostic_only': True, 'available': False,
                   'diagnostic_error': 'bounded diagnostic reference unavailable'}
    monkeypatch.setattr(feedback, '_zero_frequency_context', lambda *args: unavailable)
    evaluated = feedback.evaluate_channel_born_map(*_fixture(), g=.023, workers=1,
                                                   external_tile=1, bin_tile=2)
    assert evaluated.zero_frequency_context is unavailable
    assert evaluated.checks['full_green_local']['keldysh_identity']['ratio'] < 1
    assert evaluated.map_sigma.retarded.shape == (3, 5, 2, 2)


def test_residual_locations_choose_first_tied_global_maximum():
    mapped, supplied = _constant(0), _constant(0)
    mapped.spectral[0, 0, 0, 1] = 2
    mapped.spectral[2, 4, 1, 0] = -2
    supplied.spectral[1, 1, 1, 1] = .3
    row = feedback._sigma_residual(mapped, supplied, _config(), 'spectral')
    assert row.error == row.scale == 2
    assert row.error_location == row.scale_location == (0, 0, 0, 1)
    json.dumps(asdict(row), allow_nan=False)
