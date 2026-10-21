"""Bounded silicon SCBA continuation from authenticated saved material fields."""

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
from negfpy.anharmonic.transverse_material_restart import (
    load_material_checkpoint,
    load_material_map_fields,
    save_material_checkpoint,
)
from negfpy.anharmonic.transverse_material_scba import (
    MaterialSCBAConfig,
    mix_packed_spectra,
    run_material_scba,
)
from negfpy.anharmonic.types import SolverScale

PHYSICS = {
    'g': 0.03, 'eta_lead': 0.0015625, 'eta_device': 0,
    'n_device_layers': 9, 'interaction_layers': [2, 3, 4, 5, 6],
    'mesh': [4, 1, 1], 'grid_step': 0.025, 'n_positive': 300,
    'temperatures_kelvin': [305, 295], 'make_r0_average': True,
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
}


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spectrum_hashes(spectrum):
    """Hash existing contiguous scalar blocks without whole-spectrum copies."""
    hashes = {}
    for name in ('retarded', 'lesser', 'greater'):
        field = getattr(spectrum, name)
        digest = hashlib.sha256()
        for q in range(field.shape[0]):
            for w in range(field.shape[1]):
                digest.update(memoryview(np.ascontiguousarray(field[q, w])).cast('B'))
        hashes[name] = digest.hexdigest()
    return hashes


def validate_profile(profile):
    cfg = profile['map']
    if any(cfg.get(key) != value for key, value in PHYSICS.items()):
        raise ValueError('The continuation must preserve the fixed material physics.')
    if cfg['make_r0_average'] is not True:
        raise ValueError('The continuation must preserve the IFC normalization.')
    restart = profile['restart']
    if restart['initial_mixing'] != 0.1:
        raise ValueError('The continuation uses fixed 0.1 mixing.')
    checkpoint = restart.get('checkpoint')
    if checkpoint is not None and not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint restart requires explicit manifest and binding.')
    for key in ('parent_result_sha256', 'parent_profile_sha256'):
        value = restart[key]
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Parent hashes must be explicit SHA256 values.')
    config = MaterialSCBAConfig(**profile['scba'])
    if config.max_evaluations > 12 or config.mixing != 0.1:
        raise ValueError('The first pilot is limited to 12 evaluations with mixing 0.1.')
    if config.max_seconds + config.time_reserve_seconds >= profile['driver_timeout_seconds']:
        raise ValueError('SCBA must leave time for checkpoint and driver closure.')
    if profile['checkpoint_max_bytes'] < 6 * (16 * 4 * 601 * 120**2 + 65536):
        raise ValueError('Checkpoint budget cannot hold one evaluated state and its map.')
    return config


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config = validate_profile(profile)
    restart = profile['restart']
    result.update(
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=profile['map'], scba_config=plain(config),
        checkpoints=[], attempts=[],
        scope={'scba': True, 'accepted_field_update': False, 'accepted_restart': False,
               'frequency_or_q_convergence': False, 'converged_transport': False,
               'initial_state': 'unevaluated 0.1 times the parent Born image',
               'coupling_already_applied_to_input': True},
    )
    emit('parent_authentication_start')
    fields = load_material_map_fields(
        restart['parent_root'], expected_result_sha256=restart['parent_result_sha256'],
        expected_profile_sha256=restart['parent_profile_sha256'],
        expected_source_commit=restart['parent_source_commit'],
        expected_source_tree=restart['parent_source_tree'], expected_settings=profile['map'],
        max_total_bytes=profile['parent_max_bytes'],
    )
    result['parent_binding'] = plain(fields.binding)
    emit('parent_authenticated', result['parent_binding'])
    binding = {
        'parent_result_sha256': restart['parent_result_sha256'],
        'parent_profile_sha256': restart['parent_profile_sha256'],
        'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
        'profile_sha256': profile_sha256, 'physics': profile['map'],
    }
    checkpoint = restart.get('checkpoint')
    if checkpoint is None:
        candidate = mix_packed_spectra(None, fields.mapped_sigma, restart['initial_mixing'])
    else:
        expected_binding = checkpoint['binding']
        if (expected_binding['physics'] != profile['map']
                or expected_binding['parent_result_sha256'] != restart['parent_result_sha256']
                or expected_binding['parent_profile_sha256'] != restart['parent_profile_sha256']):
            raise ValueError('Checkpoint physics or original material parent differs.')
        saved = load_material_checkpoint(
            checkpoint['path'], expected_manifest_sha256=checkpoint['manifest_sha256'],
            expected_binding=expected_binding, max_total_bytes=profile['checkpoint_max_bytes'],
        )
        if saved.record['converged']:
            raise ValueError('A completed converged checkpoint must not be iterated automatically.')
        candidate = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, config.mixing)
        binding['resume_checkpoint_sha256'] = saved.manifest_sha256
        binding['resume_evaluated_iteration'] = saved.record['index']
        result['scope']['initial_state'] = 'unevaluated mixed successor of authenticated checkpoint'
        result['resume_binding'] = plain(saved.binding)
        emit('checkpoint_authenticated', {'manifest_sha256': saved.manifest_sha256,
                                          'index': saved.record['index']})
        del saved
    # This is a new candidate. Its currents are unknown until its own evaluation.
    emit('initial_candidate_ready', {'mixing': restart['initial_mixing'],
                                     'dyson_evaluated': False})

    def on_start(index, sigma_input):
        record = {'index': index, 'sigma_sha256': spectrum_hashes(sigma_input),
                  'dyson_evaluated': False}
        result['attempts'].append(record)
        emit('scba_evaluation_start', record)

    def on_evaluation(state):
        record = plain(state.record)
        record.update(currents=plain(state.evaluation.input_currents),
                      sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False,
                      status='converged' if record['converged'] else 'evaluated')
        emit('scba_evaluation_complete', record)
        manifest = save_material_checkpoint(
            checkpoint_directory, iteration=state.index, sigma_input=state.sigma_input,
            mapped_sigma=state.evaluation.map_sigma, record=record, binding=binding,
            max_bytes=profile['checkpoint_max_bytes'], max_checkpoints=config.max_evaluations,
        )
        result['checkpoints'].append(plain(manifest))
        emit('checkpoint_saved', plain(manifest))

    solved = run_material_scba(
        fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
        sigma_initial=candidate, scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S),
        g=profile['map']['g'], config=config, on_start=on_start, on_evaluation=on_evaluation,
    )
    result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                  failure_reason=solved.failure_reason, history=plain(solved.history),
                  iteration_plan=plain(solved.plan), completed_evaluations=len(solved.history))
    result['passed'] = solved.failure_reason is None and solved.state is not None
    if solved.state is not None:
        result['final_evaluated_state'] = plain(solved.state.record)
        result['final_currents'] = plain(solved.state.evaluation.input_currents)
    emit('scba_stopped', {'converged': solved.converged, 'stop_reason': solved.stop_reason,
                          'completed_evaluations': len(solved.history),
                          'failure_reason': solved.failure_reason})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'passed': False, 'converged': False, 'schema': 'negfpy.transverse-material-scba.v1'}

    def emit(event, details=None):
        row = {'event': event, 'time': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic() - started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(text+'\n')
        print(text, flush=True)

    try:
        run(profile, result, emit, args.output.parent / 'checkpoints',
            profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve completed checkpoints and actual failure
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
        emit('material_scba_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic() - started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('material_scba_closed', {'passed': result['passed'], 'converged': result['converged'],
                                 'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
