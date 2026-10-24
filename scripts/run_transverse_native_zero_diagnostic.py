"""Bounded native-channel zero-frequency audit of three fixed evaluated inputs.

This driver never forms a mixed candidate or runs a full Born/SCBA map.  The
saved mapped fields are comparison totals, not Green-function inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import stat
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic.transverse_channel_restart import load_channel_checkpoint
from negfpy.anharmonic.transverse_material_restart import (
    _json_verified,
    _safe_path,
    load_material_map_fields,
)
from negfpy.anharmonic.transverse_native_zero_diagnostic import (
    plan_native_zero_diagnostic_dimensions,
    study_native_zero_diagnostic,
)

# Reuse only the immutable prior driver's pure provenance constructor.  Its
# solver, mixing and field-output entry points are never invoked here.
_spec = importlib.util.spec_from_file_location(
    '_native_zero_matched25_contract',
    Path(__file__).with_name('run_transverse_channel_scba_matched25.py'))
_matched = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_matched)

COMPONENTS = ('retarded', 'statistical', 'spectral')
CASE_MANIFESTS = {
    27: '6c007c1b1febe175b073eb48a25851ee4f06120bb9eeecc993350d43c6c08c7b',
    28: '83c3d29146e7decdfd89c259b0f12608d1ef1559c8d50203d7efb23b4e95683f',
    29: 'dd96eedb7f24c880da6482743f0ceeb60f0859d72ed323c0641d90eb9a9a9bc6',
}
CONTROL_SOURCE = {
    'source_commit': '3eed849b3477ff2c63b0c68539a2c4eb59f2e3e0',
    'source_tree': '650cdfd1e8926b2c769729daf913c669872463b9',
    'profile_sha256': '13e61491299c8e24768a5693106d34937a13e3c51d97cde6193df97362de34e4',
}
DIAGNOSTIC = {
    'g': .03, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'max_bytes': 2*1024**3, 'max_contraction_work': 100_000_000_000,
    'max_total_contraction_work': 300_000_000_000,
    'max_dyson_evaluations': 20, 'max_total_dyson_evaluations': 60,
    'max_cases': 3, 'max_refinements': 4, 'half_width': 2, 'external_q': 0,
    'selected_elements': [[9, 102], [96, 102], [102, 102]],
    'workers': 8, 'external_tile': 16, 'bin_tile': 1,
    'max_seconds': 79200., 'time_reserve_seconds': 300.,
}
GLOBAL_GROUPS = ('harmonic', 'left', 'right', 'input_sigma', 'input_causality',
                 'raw_source', 'full_green_local', 'active_green', 'mapped_sigma')


def plain(value):
    """Convert metadata only; reject accidental field arrays and nonfinite scalars."""
    if isinstance(value, np.ndarray):
        raise TypeError('Diagnostic output must contain scalars and metadata, not arrays.')
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite diagnostic output scalar.')
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f'Unsupported diagnostic output type: {type(value).__name__}.')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _digest(value, length=64):
    if (not isinstance(value, str) or len(value) != length
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Explicit lowercase source and input digests are required.')


def _descriptor(path):
    value = Path(path).stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Diagnostic artifacts must be regular files without symlinks.')
    return dict(zip(('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'),
                    (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
                     value.st_ctime_ns), strict=True))


def expected_checkpoint_binding(profile):
    """Recreate exact control27..29 binding, including all policies and ancestors."""
    return dict(CONTROL_SOURCE,
        parent_result_sha256=profile['parent']['result_sha256'],
        parent_profile_sha256=profile['parent']['profile_sha256'],
        physics=plain(profile['map']), representation='retarded-statistical-spectral-v1',
        initial_state='fresh-zero', resume_parent_manifest_sha256=_matched.SEED_MANIFEST,
        resume_parent_iteration=25, resume_parent_binding=_matched.seed_binding(profile),
        acceleration={'mode': 'linear-control', 'coefficient_limit': 5.,
                      'trust_radius': .25, 'rank_rtol': 1e-10},
        selection_policy={'mode': 'bounded-tradeoff', 'component_growth_limit': 1.02,
                          'merit_ratio_limit': .99},
        proposal_feasibility={'mode': 'psd-backtracking', 'factor': .5, 'max_halvings': 8},
        matched_experiment=dict(_matched.MATCHED_INVARIANTS,
            purpose='matched longer continuation at equal fresh-map budget'))


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-native-zero-diagnostic.profile.v1':
        raise ValueError('Wrong native zero diagnostic profile schema.')
    if any(profile['map'].get(k) != v for k, v in _matched.PHYSICS.items()):
        raise ValueError('The material physics must remain fixed.')
    if profile.get('diagnostic') != DIAGNOSTIC:
        raise ValueError('The exact bounded diagnostic, guards or resources differ.')
    if (profile.get('driver_timeout_seconds') != 82800
            or profile.get('automatic_resubmission') is not False
            or profile.get('automatic_successor') is not False
            or any(k in profile for k in ('resume', 'restart', 'scba'))):
        raise ValueError('This is a fixed-input diagnostic without continuation or retry.')
    if (type(profile.get('parent_max_bytes')) is not int
            or profile['parent_max_bytes'] != 8*1024**3):
        raise ValueError('The inherited static-parent byte bound differs.')
    for key in ('source_commit', 'source_tree'):
        _digest(profile[key], 40)
    parent = profile['parent']
    for key in ('profile_sha256', 'result_sha256'):
        _digest(parent[key])
    for key in ('source_commit', 'source_tree'):
        _digest(parent[key], 40)
    binding = expected_checkpoint_binding(profile)
    cases = profile.get('checkpoints')
    if (not isinstance(cases, list) or len(cases) != 3
            or [case.get('iteration') for case in cases] != [27, 28, 29]):
        raise ValueError('Exactly the ordered native control27,28,29 cases are required.')
    for case in cases:
        index = case['iteration']
        if (type(index) is not int or case.get('manifest_sha256') != CASE_MANIFESTS[index]
                or case.get('binding') != binding
                or not isinstance(case.get('path'), str) or not case['path']
                or case.get('max_total_bytes') != 4*1024**3):
            raise ValueError('A fixed native case identity, full lineage or bound differs.')
    expected_inputs = {
        'parent_profile': (str(Path(parent['root'])/'profile.json'), parent['profile_sha256']),
        'parent_result': (str(Path(parent['root'])/'runtime/material-map-result.json'),
                          parent['result_sha256']),
        **{f'checkpoint_{case["iteration"]}_manifest':
           (str(Path(case['path'])/'manifest.json'), case['manifest_sha256']) for case in cases},
    }
    inputs = profile.get('runtime_inputs')
    if (not isinstance(inputs, list) or len(inputs) != 5
            or {row.get('name') for row in inputs} != set(expected_inputs)):
        raise ValueError('Exactly five parent/checkpoint JSON pins are required.')
    for row in inputs:
        if ((row.get('path'), row.get('sha256')) != expected_inputs[row['name']]
                or type(row.get('bytes')) is not int or not 0 < row['bytes'] <= 16*1024**2):
            raise ValueError('A diagnostic JSON pin path, hash or size differs.')
    return plain(profile['diagnostic'])


def authenticate_json_inputs(inputs):
    """Authenticate the five small JSON pins; never open field arrays here."""
    records = []
    with ExitStack() as stack:
        for row in inputs:
            path = Path(row['path'])
            before = _descriptor(path)
            if before['bytes'] != row['bytes']:
                raise ValueError('Pinned JSON size differs.')
            _json_verified(stack, path.parent, path.name, row['sha256'])
            after = _descriptor(path)
            if before != after:
                raise ValueError('Pinned JSON changed during authentication.')
            records.append(dict(row, stat_before=before, stat_after=after))
    return records


def _validate_evaluated_record(record, index):
    if (record.get('index') != index or record.get('converged') is not False
            or record.get('sigma_passed') is not False or record.get('current_passed') is not True
            or record.get('sigma_input_evaluated') is not True
            or record.get('mapped_sigma_dyson_evaluated') is not False
            or record.get('input_sigma_is_zero') is not False
            or record.get('controller_decision', {}).get('accepted_as_anchor') is not True
            or record['controller_decision'].get('selected_anchor_index') != index):
        raise ValueError('A case is not the exact evaluated nonconverged control input.')
    for group in GLOBAL_GROUPS:
        rows = record['checks'].get(group)
        if (not isinstance(rows, dict) or not rows
                or any(type(row.get('ratio')) not in (int, float)
                       or not math.isfinite(row['ratio']) or not 0 <= row['ratio'] <= 1
                       for row in rows.values())):
            raise ValueError('The saved original physical/global guards did not pass.')
    rows = record.get('current_residuals', {})
    if (set(rows) != {'contact_balance', 'collision_power', 'full_balance'}
            or any(type(row.get('ratio')) not in (int, float)
                   or not math.isfinite(row['ratio']) or not 0 <= row['ratio'] <= 1
                   for row in rows.values())):
        raise ValueError('The saved original current guards did not pass.')
    for label in ('sigma_input_sha256', 'mapped_sigma_sha256'):
        if set(record.get(label, {})) != set(COMPONENTS):
            raise ValueError('Saved input/mapped raw payload identities are required.')
        for digest in record[label].values():
            _digest(digest)


def authenticate_checkpoint(case):
    """Use the native loader's all-six-wholefile-hashes-before-mapping gate."""
    root, index = Path(case['path']), case['iteration']
    with ExitStack() as stack:
        manifest = _json_verified(stack, root, 'manifest.json', case['manifest_sha256'])
        artifacts = manifest['artifacts']
        expected = {label+'_'+name for label in ('sigma_input', 'mapped_sigma')
                    for name in COMPONENTS}
        if set(artifacts) != expected:
            raise ValueError('Each fixed native case requires exactly six R/C/D artifacts.')
        paths, before = {}, {}
        for name, row in artifacts.items():
            if (row['relative_path'] != 'arrays/'+name+'.npy'
                    or row.get('shape') != [4, 601, 120, 120]
                    or row.get('dtype') != 'complex128'
                    or row.get('array_bytes') != 553881600
                    or row.get('file_bytes') != 553881728):
                raise ValueError('A native material case artifact has unexpected metadata.')
            paths[name] = _safe_path(root, row['relative_path'])
            before[name] = _descriptor(paths[name])
        checkpoint = load_channel_checkpoint(
            root, expected_manifest_sha256=case['manifest_sha256'],
            expected_binding=case['binding'], max_total_bytes=case['max_total_bytes'])
        try:
            if checkpoint.manifest['iteration'] != index:
                raise ValueError('Native checkpoint iteration differs from its fixed case.')
            _validate_evaluated_record(checkpoint.record, index)
            authenticated = {}
            for name, row in artifacts.items():
                after = _descriptor(paths[name])
                if before[name] != after:
                    raise ValueError('Native case artifact changed during authentication.')
                authenticated[name] = dict(row, path=str(paths[name]),
                                           stat_before=before[name], stat_after=after)
        except Exception:
            release_checkpoint(checkpoint)
            raise
    return checkpoint, {
        'path': str(root), 'manifest_sha256': case['manifest_sha256'], 'iteration': index,
        'binding': plain(checkpoint.binding), 'artifacts_freshly_authenticated': 6,
        'artifacts': authenticated, 'accepted_restart': False,
        'input_role': 'fixed evaluated native R/C/D input; no mixing or update',
        'mapped_role': 'saved unevaluated comparison total; never Green input',
        'raw_payload_hash_domain': 'q/frequency-ordered complex128 bytes excluding NPY header',
        'artifact_hash_domain': 'whole NPY file including header',
    }


def verify_unchanged(binding):
    """Compare stable descriptors after computation without rehashing field arrays."""
    for row in binding['artifacts'].values():
        after = _descriptor(row['path'])
        row['stat_after_diagnostic'] = after
        if after != row['stat_before']:
            raise ValueError('Authenticated artifact changed during the diagnostic.')


def release_checkpoint(checkpoint):
    """Release this case's six readonly mappings before the next case is loaded."""
    for spectrum in (checkpoint.sigma_input, checkpoint.mapped_sigma):
        for name in COMPONENTS:
            values = getattr(spectrum, name)
            mmap = getattr(values, '_mmap', None)
            if mmap is not None and not mmap.closed:
                mmap.close()


def run(profile, result, emit, *, profile_sha256, clock=time.monotonic):
    config = validate_profile(profile)
    _digest(profile_sha256)
    started = clock()
    deadline = started+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-native-zero-diagnostic.v1', passed=False,
        converged=False, source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']),
        diagnostic_config=config, cases=[], started_cases=0, completed_cases=0,
        admitted_dyson_evaluation_bound=0, admitted_contraction_work_bound=0,
        stop_reason=None, failure_reason=None,
        scope={'scba': False, 'full_map': False, 'full_born_map': False,
               'scalar_output_only': True, 'field_output': False,
               'accepted_field_update': False, 'accepted_restart': False,
               'current_acceptance': False, 'converged_transport': False,
               'frequency_or_q_convergence': False, 'fixed_evaluated_inputs': True,
               'checkpoint_mixing_or_update': False, 'native_spectral_independent': True,
               'mapped_sigma_used_as_green_input': False,
               'cached_parent_green_or_born_used_computationally': False,
               'local_ratios_are_diagnostic_only': True,
               'inferred_remainder_is_not_direct_evaluation': True})
    emit('diagnostic_parent_authentication_start')
    result['runtime_input_authentication'] = authenticate_json_inputs(profile['runtime_inputs'])
    parent = profile['parent']
    fields = load_material_map_fields(
        parent['root'], expected_result_sha256=parent['result_sha256'],
        expected_profile_sha256=parent['profile_sha256'],
        expected_source_commit=parent['source_commit'], expected_source_tree=parent['source_tree'],
        expected_settings=profile['map'], max_total_bytes=profile['parent_max_bytes'])
    result['parent_binding'] = plain(fields.binding)
    if result['parent_binding'].get('artifacts_freshly_authenticated') != 19:
        raise ValueError('All nineteen static parent artifacts must authenticate freshly.')
    emit('diagnostic_parent_authenticated', {'parent_binding': result['parent_binding'],
        'runtime_inputs': result['runtime_input_authentication']})
    plan = plain(plan_native_zero_diagnostic_dimensions(
        fields.mesh.n_qpoints, fields.grid.n_points, fields.dynamical.shape[-1], fields.vertex.ndof,
        half_width=config['half_width'], external_tile=config['external_tile'],
        workers=config['workers'], bin_tile=config['bin_tile']))
    if (plan['dyson_evaluations'] != 20 or plan['dyson_evaluations'] > config['max_dyson_evaluations']
            or plan['contraction_work'] > config['max_contraction_work']
            or plan['peak_owned_bytes'] > config['max_bytes']):
        raise ValueError('The bounded native diagnostic plan exceeds its frozen per-case limits.')
    result['case_plan'] = plan
    for case in profile['checkpoints']:
        index = case['iteration']
        if clock() >= deadline:
            result['stop_reason'] = 'time_admission_before_case'
            result['unstarted_case_index'] = index
            break
        row = {'index': index, 'status': 'authenticating', 'completed': False,
               'checkpoint_manifest_sha256': case['manifest_sha256']}
        result['cases'].append(row)
        checkpoint = None
        emit('checkpoint_authentication_start', {'index': index, 'path': case['path'],
                                                'manifest_sha256': case['manifest_sha256']})
        try:
            checkpoint, row['checkpoint_binding'] = authenticate_checkpoint(case)
            row['evaluated_record'] = plain(checkpoint.record)
            emit('checkpoint_authenticated', {'index': index, **row['checkpoint_binding']})
            if clock() >= deadline:
                row['status'] = 'not_admitted'
                result['stop_reason'] = 'time_admission_after_checkpoint_authentication'
                result['unstarted_case_index'] = index
                break
            dyson = result['admitted_dyson_evaluation_bound']+plan['dyson_evaluations']
            work = result['admitted_contraction_work_bound']+plan['contraction_work']
            if (dyson > config['max_total_dyson_evaluations']
                    or work > config['max_total_contraction_work']):
                raise ValueError('Cumulative diagnostic resource limit exceeded before a case.')
            result['admitted_dyson_evaluation_bound'] = dyson
            result['admitted_contraction_work_bound'] = work
            result['started_cases'] += 1
            row.update(status='running', plan=plan)
            emit('diagnostic_case_start', {'index': index, 'plan': plan,
                'input_sigma_sha256': checkpoint.record['sigma_input_sha256'],
                'saved_mapped_sigma_sha256': checkpoint.record['mapped_sigma_sha256'],
                'input_role': row['checkpoint_binding']['input_role'],
                'mapped_role': row['checkpoint_binding']['mapped_role']})
            call_keys = ('g', 'max_bytes', 'max_contraction_work', 'max_dyson_evaluations',
                         'max_refinements', 'half_width', 'external_q', 'selected_elements',
                         'workers', 'external_tile', 'bin_tile')
            diagnostic = plain(study_native_zero_diagnostic(
                fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
                checkpoint.sigma_input, checkpoint.mapped_sigma,
                **{key: config[key] for key in call_keys}, deadline_monotonic=deadline))
            if (diagnostic.get('plan') != plan
                    or diagnostic.get('dyson_evaluations') != plan['dyson_evaluations']):
                raise ValueError('The completed diagnostic plan differs from its admitted plan.')
            row.update(status='complete', completed=True, diagnostic=diagnostic)
            verify_unchanged(row['checkpoint_binding'])
            result['completed_cases'] += 1
            emit('diagnostic_case_complete', row)
        except Exception as exc:
            row.update(status='failed', failure={'type': type(exc).__name__, 'message': str(exc)})
            result['failure_reason'] = type(exc).__name__+': '+str(exc)
            result['stop_reason'] = 'case_failed'
            raise
        finally:
            if checkpoint is not None:
                try:
                    verify_unchanged(row['checkpoint_binding'])
                finally:
                    release_checkpoint(checkpoint)
    verify_unchanged(result['parent_binding'])
    for row in result['runtime_input_authentication']:
        row['stat_after_diagnostic'] = _descriptor(row['path'])
        if row['stat_after_diagnostic'] != row['stat_before']:
            raise ValueError('A pinned JSON changed during the diagnostic.')
    if result['completed_cases'] == 3:
        result.update(passed=True, stop_reason='all_fixed_cases_completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'schema': 'negfpy.transverse-native-zero-diagnostic.v1', 'passed': False,
              'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)

    try:
        run(profile, result, emit, profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve actual failed/partial case evidence
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc),
                      failure={'type': type(exc).__name__, 'message': str(exc)})
        traceback.print_exc()
        emit('diagnostic_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('diagnostic_closed', {'passed': result['passed'], 'converged': False,
        'completed_cases': result.get('completed_cases', 0),
        'stop_reason': result.get('stop_reason'), 'failure_reason': result.get('failure_reason'),
        'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
