"""One raw-channel Born arithmetic comparison using authenticated harmonic fields.

This does not repair inherited input defects, evaluate a Dyson successor, advance
SCBA, or publish numerical fields. Existing contour results are a comparison,
not an independent material oracle or an acceptance threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic.transverse_channels import (
    StatisticalSpectral,
    cubic_born_channels_packed_fft,
    plan_born_channels_fft,
    retarded_from_spectral,
)
from negfpy.anharmonic.transverse_material_restart import load_material_map_fields

PHYSICS = {
    'g': .03, 'eta_lead': .0015625, 'eta_device': 0,
    'n_device_layers': 9, 'interaction_layers': [2, 3, 4, 5, 6],
    'mesh': [4, 1, 1], 'grid_step': .025, 'n_positive': 300,
    'temperatures_kelvin': [305, 295], 'make_r0_average': True,
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def field_hash(field):
    digest = hashlib.sha256()
    for q in range(field.shape[0]):
        for w in range(field.shape[1]):
            digest.update(memoryview(np.ascontiguousarray(field[q, w])).cast('B'))
    return digest.hexdigest()


def maximum(value):
    return float(np.max(np.abs(value), initial=0))


def accumulate(record, name, left, right, q, w):
    difference = left-right
    error = maximum(difference)
    scale = max(maximum(left), maximum(right))
    row = record.setdefault(name, {'error': 0.0, 'scale': 0.0, 'location': None})
    if error > row['error']:
        entry = np.unravel_index(int(np.argmax(np.abs(difference))), difference.shape)
        row['error'] = error
        row['location'] = {'q_index': q, 'frequency_index': w,
                           'matrix_entry': [int(i) for i in entry]}
    row['scale'] = max(row['scale'], scale)


def finish(record):
    for row in record.values():
        if not np.isfinite(row['error']) or not np.isfinite(row['scale']):
            raise ValueError('Nonfinite comparison scalar.')
        row['allowance'] = 1e-11+1e-8*row['scale']
        row['allowance_ratio'] = row['error']/row['allowance']
        row['acceptance'] = False
    return record


def channel_checks(channels, mesh):
    checks = {}
    nq, nw = channels.statistical.shape[:2]
    for q in range(nq):
        opposite = mesh.negative_index(q)
        for w in range(nw):
            for name in ('statistical', 'spectral'):
                field = getattr(channels, name)
                block = field[q, w]
                accumulate(checks, name+'_antihermiticity', block, -block.conj().T, q, w)
                partner = field[opposite, nw-1-w].T
                accumulate(checks, name+'_reflection', block,
                           partner if name == 'statistical' else -partner, q, w)
    return finish(checks)


def comparison(channels, retarded, saved):
    checks = {}
    nq, nw = channels.statistical.shape[:2]
    for q in range(nq):
        for w in range(nw):
            center, delta = channels.statistical[q, w], channels.spectral[q, w]
            old_l, old_g = saved.lesser[q, w], saved.greater[q, w]
            for name, new, old in (
                ('statistical_vs_saved_mean', center, .5*(old_g+old_l)),
                ('spectral_vs_saved_difference', delta, old_g-old_l),
                ('exported_lesser_vs_saved', center-.5*delta, old_l),
                ('exported_greater_vs_saved', center+.5*delta, old_g),
                ('retarded_vs_saved', retarded[q, w], saved.retarded[q, w]),
                ('independent_source_identity', delta,
                 retarded[q, w]-retarded[q, w].conj().T),
                ('saved_source_identity', old_g-old_l,
                 saved.retarded[q, w]-saved.retarded[q, w].conj().T),
                ('exported_contour_roundtrip', (center+.5*delta)-(center-.5*delta), delta),
            ):
                accumulate(checks, name, new, old, q, w)
    return finish(checks)


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-channels-comparison.profile.v1':
        raise ValueError('Incorrect comparison profile schema.')
    if any(profile['map'].get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('Material physics must remain unchanged.')
    config = profile['comparison']
    required = {'kind': 'raw-cached-harmonic-channel-born', 'born_calls': 1,
                'dyson_evaluations': 0, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
                'workers': 8, 'external_tile': 16, 'bin_tile': 1}
    if any(config.get(k) != v for k, v in required.items()):
        raise ValueError('Comparison scope or tolerances differ.')
    for key in ('max_bytes', 'max_contraction_work'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Positive integer resource limits required.')
    if type(profile['parent_max_bytes']) is not int or profile['parent_max_bytes'] < 1:
        raise ValueError('Positive parent byte limit required.')
    for key in ('source_commit', 'source_tree'):
        value = profile[key]
        if len(value) != 40 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Explicit source bindings required.')
    return config


def run(profile, result, emit, *, profile_sha256):
    config = validate_profile(profile)
    parent = profile['parent']
    result.update(passed=False, completed=False, converged=False, failure_reason=None,
                  source_commit=profile['source_commit'], source_tree=profile['source_tree'],
                  profile_sha256=profile_sha256, settings=profile['map'],
                  comparison_config=config,
                  scope={'comparison_only': True, 'inherited_input_defects_retained': True,
                         'input_channels_from_saved_contours': True,
                         'independent_material_oracle': False, 'repair_accepted': False,
                         'dyson_evaluated': False, 'scba': False, 'accepted_restart': False,
                         'accepted_field_update': False, 'converged_transport': False,
                         'frequency_or_q_convergence': False, 'numerical_field_outputs': False,
                         'coupling_applied_once_as_g_squared': True})
    emit('parent_authentication_start')
    fields = load_material_map_fields(
        parent['root'], expected_profile_sha256=parent['profile_sha256'],
        expected_result_sha256=parent['result_sha256'],
        expected_source_commit=parent['source_commit'], expected_source_tree=parent['source_tree'],
        expected_settings=profile['map'], max_total_bytes=profile['parent_max_bytes'])
    result['parent_binding'] = fields.binding
    emit('parent_authenticated', fields.binding)
    plan = plan_born_channels_fft(fields.grid, fields.vertex,
                                 external_tile=config['external_tile'], workers=config['workers'],
                                 bin_tile=config['bin_tile'])
    input_bytes = 2*fields.input_green.lesser.nbytes
    # Retarded reconstruction follows the kernel after its FFT storage is freed.
    # Reserve four extra output-sized arrays for its finite-grid FFT workspace.
    budget = input_bytes + max(plan.peak_owned_bytes, 6*fields.input_green.lesser.nbytes)
    if budget > config['max_bytes'] or plan.contraction_work > config['max_contraction_work']:
        raise ValueError('Comparison resources exceeded before channel allocation.')
    result['memory_plan'] = {'born': asdict(plan), 'input_channel_bytes': input_bytes,
                             'combined_owned_bound_bytes': budget,
                             'caller_parent_fields_python_and_library_scratch_excluded': True}
    raw = fields.input_green
    channels = StatisticalSpectral(.5*(raw.greater+raw.lesser), raw.greater-raw.lesser)
    result['input_channel_sha256'] = {key: field_hash(getattr(channels, key))
                                     for key in ('statistical', 'spectral')}
    result['input_channel_checks'] = channel_checks(channels, fields.mesh)
    emit('channels_prepared', {'input_channel_sha256': result['input_channel_sha256'],
                              'inherited_input_defects_retained': True})
    emit('channel_born_start', {'born_calls': 1, 'contraction_work': plan.contraction_work,
                               'workers': config['workers'], 'dyson_evaluations': 0})
    start = time.monotonic()
    mapped = cubic_born_channels_packed_fft(
        fields.grid, fields.vertex, channels, external_tile=config['external_tile'],
        workers=config['workers'], bin_tile=config['bin_tile'],
        max_bytes=config['max_bytes']-input_bytes,
        max_contraction_work=config['max_contraction_work'])
    for key in ('statistical', 'spectral'):
        getattr(mapped, key)[:] *= profile['map']['g']**2
    result['born_seconds'] = time.monotonic()-start
    emit('channel_born_complete', {'born_calls': 1, 'elapsed_seconds': result['born_seconds'],
                                  'coupling_applied_once_as_g_squared': True})
    retarded = retarded_from_spectral(mapped.spectral, fields.grid,
                                     max_bytes=config['max_bytes']-input_bytes-plan.output_bytes)
    result['output_channel_checks'] = channel_checks(mapped, fields.mesh)
    result['comparison'] = comparison(mapped, retarded, fields.mapped_sigma)
    result['output_raw_sha256'] = {'statistical': field_hash(mapped.statistical),
                                   'spectral': field_hash(mapped.spectral),
                                   'retarded': field_hash(retarded)}
    z = fields.grid.zero_index
    result['gamma_zero'] = {
        'q_index': 0, 'frequency_index': z,
        'independent_spectral_max_abs': maximum(mapped.spectral[0, z]),
        'independent_spectral_antihermiticity': maximum(
            mapped.spectral[0, z]+mapped.spectral[0, z].conj().T),
        'independent_source_defect': maximum(mapped.spectral[0, z]
                                            -(retarded[0, z]-retarded[0, z].conj().T)),
        'saved_source_defect': maximum(fields.mapped_sigma.greater[0, z]
                                      -fields.mapped_sigma.lesser[0, z]
                                      -(fields.mapped_sigma.retarded[0, z]
                                        -fields.mapped_sigma.retarded[0, z].conj().T)),
        'retarded_reality_defect': maximum(retarded[0, z]-retarded[0, z].conj()),
    }
    result.update(passed=True, completed=True, born_calls=1, dyson_evaluations=0)
    emit('comparison_complete', {'passed': True, 'born_calls': 1, 'dyson_evaluations': 0,
                                 'repair_accepted': False})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    start = time.monotonic()
    result = {'schema': 'negfpy.transverse-channels-comparison.v1', 'passed': False,
              'completed': False, 'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'details': details or {}}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)
    try:
        run(profile, result, emit, profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve the actual failed comparison tail
        result['failure_reason'] = type(exc).__name__+': '+str(exc)
        traceback.print_exc()
        emit('comparison_failed', {'failure_reason': result['failure_reason']})
    result['elapsed_seconds'] = time.monotonic()-start
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('comparison_closed', {'passed': result['passed'], 'completed': result['completed'],
                               'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
