"""Diagnose one authenticated failed-candidate Green pass, without a Born map."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
import traceback
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._material_nq1_campaign import HBAR_J_S, OMEGA0_RAD_S
from negfpy.anharmonic.transverse_material_map import evaluate_packed_born_map
from negfpy.anharmonic.transverse_material_restart import (
    load_material_checkpoint,
    load_material_map_fields,
)
from negfpy.anharmonic.transverse_material_scba import mix_packed_spectra
from negfpy.anharmonic.types import SolverScale

PHYSICS = {
    'g': 0.03, 'eta_lead': 0.0015625, 'eta_device': 0,
    'n_device_layers': 9, 'interaction_layers': [2, 3, 4, 5, 6],
    'mesh': [4, 1, 1], 'grid_step': 0.025, 'n_positive': 300,
    'temperatures_kelvin': [305, 295], 'make_r0_average': True,
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
}
_COMPONENTS = ('retarded', 'lesser', 'greater')


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def spectrum_hashes(spectrum):
    """Use the failed attempt's q/frequency-ordered raw complex128 byte domain."""
    hashes = {}
    for name in _COMPONENTS:
        field = getattr(spectrum, name)
        digest = hashlib.sha256()
        for q in range(field.shape[0]):
            for w in range(field.shape[1]):
                digest.update(memoryview(np.ascontiguousarray(field[q, w])).cast('B'))
        hashes[name] = digest.hexdigest()
    return hashes


def _digest(value, name, length=64):
    if (not isinstance(value, str) or len(value) != length
            or any(char not in '0123456789abcdef' for char in value)):
        raise ValueError(f'{name} must be an explicit hexadecimal digest.')


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f'{name} must be a positive integer.')


def validate_profile(profile):
    cfg = profile['map']
    if any(cfg.get(key) != value for key, value in PHYSICS.items()):
        raise ValueError('The diagnostic must preserve the fixed material physics.')
    if cfg['make_r0_average'] is not True:
        raise ValueError('The diagnostic must preserve the IFC normalization.')
    for key in ('source_commit', 'source_tree'):
        _digest(profile[key], key, 40)
    restart = profile['restart']
    if restart['initial_mixing'] != 0.1:
        raise ValueError('The failed candidate uses fixed 0.1 mixing.')
    for key in ('parent_result_sha256', 'parent_profile_sha256'):
        _digest(restart[key], key)
    for key in ('parent_source_commit', 'parent_source_tree'):
        _digest(restart[key], key, 40)
    checkpoint = restart['checkpoint']
    if not isinstance(checkpoint, dict) or set(checkpoint) != {'path', 'manifest_sha256', 'binding'}:
        raise ValueError('The diagnostic requires an explicitly bound complete checkpoint.')
    _digest(checkpoint['manifest_sha256'], 'checkpoint manifest')
    if not isinstance(checkpoint['path'], str) or not checkpoint['path']:
        raise ValueError('Checkpoint path must be explicit.')
    binding = checkpoint['binding']
    if (binding['physics'] != cfg
            or binding['parent_result_sha256'] != restart['parent_result_sha256']
            or binding['parent_profile_sha256'] != restart['parent_profile_sha256']):
        raise ValueError('Checkpoint physics or original material parent differs.')
    # The authenticated saved binding belongs to the old scientific source.
    # It must never be rewritten to the new diagnostic driver's source.
    for key in ('source_commit', 'source_tree'):
        _digest(binding[key], 'checkpoint '+key, 40)
    _digest(binding['profile_sha256'], 'checkpoint profile')
    diagnostic = profile['diagnostic']
    if diagnostic['identity_rtol'] != 1e-8 or diagnostic['identity_atol'] != 1e-11:
        raise ValueError('Green identity tolerances must remain unchanged.')
    for key, expected in (('resume_local_iteration', 2), ('resume_trajectory', 14),
                          ('trajectory_candidate', 15)):
        if type(diagnostic[key]) is not int or diagnostic[key] != expected:
            raise ValueError('The diagnostic is bound to failed trajectory candidate 15.')
    expected_hashes = diagnostic['expected_candidate_sha256']
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(_COMPONENTS):
        raise ValueError('All three failed-candidate component hashes must be pinned.')
    for key, value in expected_hashes.items():
        _digest(value, 'failed candidate '+key)
    for key in ('parent_max_bytes', 'checkpoint_max_bytes', 'driver_timeout_seconds'):
        _positive_integer(profile[key], key)
    _positive_integer(diagnostic['max_bytes'], 'diagnostic max_bytes')
    for key in ('workers', 'external_tile', 'bin_tile', 'max_contraction_work',
                'max_dyson_evaluations'):
        _positive_integer(cfg[key], key)
    if (cfg['max_dyson_evaluations'] != 2404
            or type(diagnostic['max_dyson_evaluations']) is not int
            or diagnostic['max_dyson_evaluations'] != 2404):
        raise ValueError('The diagnostic is limited to the same 2404 Dyson points.')
    return diagnostic


def run(profile, result, emit, *, profile_sha256):
    diagnostic = validate_profile(profile)
    _digest(profile_sha256, 'profile')
    restart = profile['restart']
    checkpoint = restart['checkpoint']
    result.update(
        passed=False, completed=False, converged=False, failure_reason=None,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']),
        diagnostic_config=plain(diagnostic),
        scope={'diagnostic_only': True, 'scba': False, 'born_evaluated': False,
               'accepted': False, 'accepted_field_update': False, 'accepted_restart': False,
               'frequency_or_q_convergence': False, 'converged_transport': False,
               'initial_state': 'unevaluated mixed successor of authenticated checkpoint',
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
    saved = load_material_checkpoint(
        checkpoint['path'], expected_manifest_sha256=checkpoint['manifest_sha256'],
        expected_binding=checkpoint['binding'], max_total_bytes=profile['checkpoint_max_bytes'],
    )
    if (saved.record['index'] != diagnostic['resume_local_iteration']
            or saved.record['converged'] is not False):
        raise ValueError('The checkpoint is not the last nonconverged completed local state 2.')
    result.update(
        resume_binding=plain(saved.binding), resume_checkpoint_sha256=saved.manifest_sha256,
        resume_evaluated_iteration=saved.record['index'],
        resume_trajectory=diagnostic['resume_trajectory'],
        trajectory_candidate=diagnostic['trajectory_candidate'],
    )
    emit('checkpoint_authenticated', {
        'manifest_sha256': saved.manifest_sha256, 'index': saved.record['index'],
        'trajectory_index': diagnostic['resume_trajectory'],
    })
    candidate = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, restart['initial_mixing'])
    del saved
    candidate_hashes = spectrum_hashes(candidate)
    result['candidate_sha256'] = candidate_hashes
    result['candidate_hashes_match'] = candidate_hashes == diagnostic['expected_candidate_sha256']
    if not result['candidate_hashes_match']:
        raise ValueError('Mixed candidate hashes differ from the captured failed attempt.')
    emit('diagnostic_candidate_ready', {
        'mixing': restart['initial_mixing'], 'sigma_sha256': candidate_hashes,
        'trajectory_index': diagnostic['trajectory_candidate'], 'dyson_evaluated': False,
    })
    emit('green_diagnostic_start', {
        'trajectory_index': diagnostic['trajectory_candidate'],
        'green_diagnostic_only': True, 'maximum_evaluations': 1,
    })
    cfg = profile['map']
    captured = evaluate_packed_born_map(
        fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
        sigma_input=candidate, scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S),
        g=cfg['g'], workers=cfg['workers'], external_tile=cfg['external_tile'],
        bin_tile=cfg['bin_tile'], max_bytes=diagnostic['max_bytes'],
        max_contraction_work=cfg['max_contraction_work'],
        max_dyson_evaluations=diagnostic['max_dyson_evaluations'],
        identity_rtol=diagnostic['identity_rtol'], identity_atol=diagnostic['identity_atol'],
        green_diagnostic_only=True,
    )
    result.update(
        completed=captured.completed, green_passed=captured.green_passed,
        diagnostics=plain(captured.diagnostics), plan=plain(captured.plan),
        diagnostic_workspace_bytes=captured.diagnostic_workspace_bytes,
        diagnostic_peak_owned_bytes=captured.diagnostic_peak_owned_bytes,
    )
    coverage = result['diagnostics']
    expected_points = math.prod(cfg['mesh']) * (2*cfg['n_positive']+1)
    if (captured.completed is not True or type(captured.green_passed) is not bool
            or coverage.get('all_points_observed') is not True
            or coverage.get('completed_points') != expected_points
            or coverage.get('expected_points') != expected_points):
        raise ValueError('The requested complete Green diagnostic capture was not returned.')
    # A captured identity failure is the diagnostic's result, not execution failure.
    result['passed'] = True
    emit('green_diagnostic_complete', {
        'passed': True, 'completed': True, 'green_passed': captured.green_passed,
        'candidate_hashes_match': True, 'completed_points': expected_points,
        'born_evaluated': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'passed': False, 'completed': False, 'converged': False,
              'schema': 'negfpy.transverse-green-diagnostic.v1'}

    def emit(event, details=None):
        row = {'event': event, 'time': datetime.now(timezone.utc).isoformat(),
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(text+'\n')
        print(text, flush=True)

    try:
        run(profile, result, emit, profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - retain actual failed diagnostic evidence
        result['passed'] = False
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        result['failure_reason'] = type(exc).__name__+': '+str(exc)
        traceback.print_exc()
        emit('green_diagnostic_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('green_diagnostic_closed', {
        'passed': result['passed'], 'completed': result['completed'],
        'green_passed': result.get('green_passed'), 'elapsed_seconds': result['elapsed_seconds'],
    })
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
