"""Fresh bounded material SCBA pilot with persistent statistical/spectral fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._material_nq1_campaign import HBAR_J_S, OMEGA0_RAD_S
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    run_channel_scba,
)
from negfpy.anharmonic.transverse_channel_restart import save_channel_checkpoint
from negfpy.anharmonic.transverse_material_restart import load_material_map_fields
from negfpy.anharmonic.types import SolverScale

PHYSICS = {
    'g': .03, 'eta_lead': .0015625, 'eta_device': 0,
    'n_device_layers': 9, 'interaction_layers': [2, 3, 4, 5, 6],
    'mesh': [4, 1, 1], 'grid_step': .025, 'n_positive': 300,
    'temperatures_kelvin': [305, 295], 'make_r0_average': True,
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
}


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spectrum_hashes(spectrum):
    if spectrum is None:
        return None
    records = {}
    for name in ('retarded', 'statistical', 'spectral'):
        field = getattr(spectrum, name)
        digest = hashlib.sha256()
        for q in range(field.shape[0]):
            for w in range(field.shape[1]):
                digest.update(memoryview(np.ascontiguousarray(field[q, w])).cast('B'))
        records[name] = digest.hexdigest()
    return records


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-channel-scba.profile.v1':
        raise ValueError('Wrong channel pilot profile schema.')
    if any(profile['map'].get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('The material physics must remain fixed.')
    if profile['initial_state'] != 'fresh-zero' or 'restart' in profile:
        raise ValueError('This pilot requires a fresh zero self-energy, not imported contours.')
    config = ChannelSCBAConfig(**profile['scba'])
    pinned = {'max_evaluations': 3, 'mixing': .1, 'sigma_rtol': 1e-4, 'sigma_atol': 1e-10,
              'current_rtol': 1e-3, 'current_atol_w': 1e-14,
              'identity_rtol': 1e-8, 'identity_atol': 1e-11,
              'workers': 8, 'external_tile': 16, 'bin_tile': 1,
              'max_contraction_work': 100_000_000_000_000, 'max_dyson_evaluations': 2404}
    if any(getattr(config, k) != v for k, v in pinned.items()):
        raise ValueError('Pilot bounds, mixing, resources or tolerances differ.')
    if config.max_seconds+config.time_reserve_seconds >= profile['driver_timeout_seconds']:
        raise ValueError('SCBA must reserve time for driver closure.')
    if profile['checkpoint_max_bytes'] < 6*(16*4*601*120**2+65536):
        raise ValueError('Checkpoint bound cannot hold six fields.')
    if type(profile['parent_max_bytes']) is not int or profile['parent_max_bytes'] < 1:
        raise ValueError('Positive parent byte limit required.')
    for key in ('source_commit', 'source_tree'):
        value = profile[key]
        if not isinstance(value, str) or len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Explicit source commit/tree bindings required.')
    return config


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config = validate_profile(profile)
    parent = profile['parent']
    result.update(passed=False, converged=False, source_commit=profile['source_commit'],
                  source_tree=profile['source_tree'], profile_sha256=profile_sha256,
                  settings=profile['map'], scba_config=plain(config), checkpoints=[], attempts=[],
                  evaluated_records=[],
                  scope={'scba': True, 'fresh_green_evaluations': True,
                         'independent_spectral_storage': True, 'initial_state': 'fresh-zero',
                         'representation': 'retarded-statistical-spectral-v1',
                         'historical_checkpoint_imported': False, 'cached_green_used': False,
                         'accepted_restart': False, 'accepted_field_update': False,
                         'converged_transport': False, 'frequency_or_q_convergence': False,
                         'exact_conjugate_structure_enforced': False,
                         'primary_identity_uses_independent_spectral': True,
                         'exported_contour_checks_are_diagnostic': True})
    emit('parent_authentication_start')
    fields = load_material_map_fields(
        parent['root'], expected_result_sha256=parent['result_sha256'],
        expected_profile_sha256=parent['profile_sha256'],
        expected_source_commit=parent['source_commit'], expected_source_tree=parent['source_tree'],
        expected_settings=profile['map'], max_total_bytes=profile['parent_max_bytes'])
    result['parent_binding'] = plain(fields.binding)
    emit('parent_authenticated', result['parent_binding'])
    binding = {'parent_result_sha256': parent['result_sha256'],
               'parent_profile_sha256': parent['profile_sha256'],
               'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
               'profile_sha256': profile_sha256, 'physics': profile['map'],
               'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero'}
    emit('channel_pilot_start', {'initial_state': 'fresh-zero', 'max_evaluations': 3,
                                 'mixing': .1, 'historical_checkpoint_imported': False})

    def on_start(index, sigma_input):
        attempt = {'index': index, 'input_sigma_is_zero': sigma_input is None,
                   'sigma_sha256': spectrum_hashes(sigma_input), 'dyson_evaluated': False}
        result['attempts'].append(attempt)
        emit('scba_evaluation_start', attempt)

    def on_evaluation(state):
        record = plain(state.record)
        record.update(currents=plain(state.evaluation.input_currents),
                      input_sigma_is_zero=state.sigma_input is None,
                      sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False,
                      status='converged' if record['converged'] else 'evaluated',
                      checks=plain(state.evaluation.checks))
        result['evaluated_records'].append(record)
        emit('scba_evaluation_complete', record)
        manifest = save_channel_checkpoint(
            checkpoint_directory, iteration=state.index, sigma_input=state.sigma_input,
            mapped_sigma=state.evaluation.map_sigma, record=record, binding=binding,
            max_bytes=profile['checkpoint_max_bytes'], max_checkpoints=config.max_evaluations)
        result['checkpoints'].append(plain(manifest))
        emit('checkpoint_saved', plain(manifest))

    solved = run_channel_scba(
        fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
        scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S), g=profile['map']['g'],
        config=config, on_start=on_start, on_evaluation=on_evaluation)
    result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                  failure_reason=solved.failure_reason, history=plain(solved.history),
                  iteration_plan=plain(solved.plan), completed_evaluations=len(solved.history))
    result['passed'] = solved.failure_reason is None and solved.state is not None
    if solved.state is not None:
        result['final_evaluated_state'] = plain(solved.state.record)
        result['final_currents'] = plain(solved.state.evaluation.input_currents)
    emit('scba_stopped', {'converged': solved.converged, 'stop_reason': solved.stop_reason,
                          'failure_reason': solved.failure_reason,
                          'completed_evaluations': len(solved.history)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'schema': 'negfpy.transverse-channel-scba.v1', 'passed': False, 'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)
    try:
        run(profile, result, emit, args.output.parent/'checkpoints', profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve checkpoints and actual failure tails
        result['failure_reason'] = type(exc).__name__+': '+str(exc)
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
        emit('channel_scba_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('channel_scba_closed', {'passed': result['passed'], 'converged': result['converged'],
                                 'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
