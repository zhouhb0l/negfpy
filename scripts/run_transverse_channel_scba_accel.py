"""Bounded native-channel acceleration/control comparison from pinned state15."""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import shutil
import stat
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._material_nq1_campaign import HBAR_J_S, OMEGA0_RAD_S
from negfpy.anharmonic.transverse_channel_accel import (
    ChannelAccelerationConfig,
    run_channel_accel,
)
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    mix_channel_spectra,
)
from negfpy.anharmonic.transverse_channel_restart import (
    load_channel_checkpoint,
    save_channel_checkpoint,
)
from negfpy.anharmonic.transverse_material_restart import (
    _json_verified,
    _safe_path,
    load_material_map_fields,
)
from negfpy.anharmonic.types import SolverScale

PHYSICS = {
    'g': .03, 'eta_lead': .0015625, 'eta_device': 0,
    'n_device_layers': 9, 'interaction_layers': [2, 3, 4, 5, 6],
    'mesh': [4, 1, 1], 'grid_step': .025, 'n_positive': 300,
    'temperatures_kelvin': [305, 295], 'make_r0_average': True,
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
}
OLD_BINDING = {
    'source_commit': '09d58392cf3ead026886d06a50efbbe7204a4f19',
    'source_tree': '9b835a9995b22f6d1b71b0a8123aebf32b3d5f75',
    'profile_sha256': 'f46de9ca9e58b923be5c68644af48787116719cd5c26a46abd98340121808898',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
}
RESUME_MANIFEST = '57365e5448bbc8fd3570ef7175afa7257c80a003f7bd1039d669b9d254dc53be'
STATE9_MANIFEST = 'cf591c69ea613f59e336cf66e6e8a909be6f0148bb5747103ee2b2c829b1ba5b'
STATE9_BINDING = {
    'source_commit': 'b014cc81ec1e7cfb1a7e2502feef3b67dd017de5',
    'source_tree': '9099644af64e86179847eeacf26978276dbce9c9',
    'profile_sha256': 'f2554a2e168c85db45247e861fb303751e0cc5c2e81d988fb686ad4b8710cd06',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
    'resume_parent_manifest_sha256': RESUME_MANIFEST, 'resume_parent_iteration': 3,
}
SEED_MANIFEST = 'f605ff69ba1b040ada59cbb26c5d4ff06b9c77c94585a828f3bbbd4a80caa5ef'
SEED_BINDING = {
    'source_commit': '3c4af426d73373437a379bd8e0c15e7802e05046',
    'source_tree': 'caf5698fc3c022d0849a2f0415257f7b0ea8d250',
    'profile_sha256': 'b0e2cb2c3ee0fa14c7d598a49f04ec520c2bad7d08b1f96f26683a0ad9e027a7',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
    'resume_parent_manifest_sha256': STATE9_MANIFEST, 'resume_parent_iteration': 9,
}
COMPONENTS = ('retarded', 'statistical', 'spectral')


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
    for name in COMPONENTS:
        field = getattr(spectrum, name)
        digest = hashlib.sha256()
        for q in range(field.shape[0]):
            for w in range(field.shape[1]):
                digest.update(memoryview(np.ascontiguousarray(field[q, w])).cast('B'))
        records[name] = digest.hexdigest()
    return records


def _digest(value, length=64):
    if (not isinstance(value, str) or len(value) != length
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Explicit lowercase source and input digests are required.')


def seed_binding(profile):
    """Recreate the exact old15 binding, including its native9/ancestor3 lineage."""
    shared = {'physics': plain(profile['map']),
              'parent_result_sha256': profile['parent']['result_sha256'],
              'parent_profile_sha256': profile['parent']['profile_sha256']}
    state9 = dict(STATE9_BINDING, **shared)
    return dict(SEED_BINDING, **shared, resume_parent_binding=state9)


def checkpoint_binding(profile, profile_sha256):
    """Bind every evaluated trial, including rejected ones, to this strategy."""
    return {'parent_result_sha256': profile['parent']['result_sha256'],
            'parent_profile_sha256': profile['parent']['profile_sha256'],
            'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
            'profile_sha256': profile_sha256, 'physics': plain(profile['map']),
            'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
            'resume_parent_manifest_sha256': profile['resume']['manifest_sha256'],
            'resume_parent_iteration': 15,
            'resume_parent_binding': plain(profile['resume']['binding']),
            'acceleration': plain(profile['acceleration'])}


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-channel-scba-accel.profile.v1':
        raise ValueError('Wrong native channel acceleration profile schema.')
    if any(profile['map'].get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('The material physics must remain fixed.')
    if (profile.get('initial_state') != 'native-channel-continuation'
            or profile.get('first_index') != 16 or 'restart' in profile):
        raise ValueError('This comparison starts with new native candidate16.')
    config = ChannelSCBAConfig(**profile['scba'])
    pinned = {'max_evaluations': 6, 'mixing': .05,
              'sigma_rtol': 1e-4, 'sigma_atol': 1e-10,
              'current_rtol': 1e-3, 'current_atol_w': 1e-14,
              'identity_rtol': 1e-8, 'identity_atol': 1e-11,
              'workers': 8, 'external_tile': 16, 'bin_tile': 1,
              'max_contraction_work': 100_000_000_000_000, 'max_dyson_evaluations': 2404}
    if any(getattr(config, k) != v for k, v in pinned.items()):
        raise ValueError('Comparison bounds, fallback mixing, resources or tolerances differ.')
    acceleration = ChannelAccelerationConfig(**profile['acceleration'])
    if (acceleration.mode not in ('linear-control', 'anderson1')
            or acceleration.coefficient_limit != 5.0 or acceleration.trust_radius != .25
            or acceleration.rank_rtol != 1e-10):
        raise ValueError('The explicit bounded acceleration strategy differs.')
    if config.max_seconds+config.time_reserve_seconds >= profile['driver_timeout_seconds']:
        raise ValueError('SCBA must reserve time for driver closure.')
    if profile['checkpoint_max_bytes'] < 6*(16*4*601*120**2+65536):
        raise ValueError('Checkpoint bound cannot hold six native fields.')
    if type(profile['parent_max_bytes']) is not int or profile['parent_max_bytes'] < 1:
        raise ValueError('Positive parent byte limit required.')
    if (type(profile.get('free_storage_min_bytes')) is not int
            or profile['free_storage_min_bytes'] < 128*1024**3):
        raise ValueError('At least 128 GiB of free checkpoint storage must be required.')
    for key in ('source_commit', 'source_tree'):
        _digest(profile[key], 40)
    resume = profile['resume']
    if (type(resume.get('iteration')) is not int or resume['iteration'] != 15
            or resume.get('manifest_sha256') != SEED_MANIFEST
            or not isinstance(resume.get('path'), str) or not resume['path']
            or type(resume.get('max_total_bytes')) is not int
            or resume['max_total_bytes'] < 6*(16*4*601*120**2+65536)):
        raise ValueError('The exact complete native state15 checkpoint must be pinned.')
    if resume['binding'] != seed_binding(profile):
        raise ValueError('Original state15 binding differs, including nested state9/ancestor3 lineage.')
    return config, acceleration


def _descriptor(path):
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Resume artifacts must be regular files without symlinks.')
    return dict(zip(('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'),
                    (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns),
                    strict=True))


def authenticate_resume(resume):
    """Record six descriptors around the loader's all-hashes-before-mapping gate."""
    root = Path(resume['path'])
    seed = resume['iteration']
    if type(seed) is not int or seed != 15:
        raise ValueError('Native resume iteration is not an approved seed.')
    with ExitStack() as stack:
        manifest = _json_verified(stack, root, 'manifest.json', resume['manifest_sha256'])
        artifacts = manifest['artifacts']
        expected = {label+'_'+name for label in ('sigma_input', 'mapped_sigma') for name in COMPONENTS}
        if set(artifacts) != expected:
            raise ValueError('Native resume requires exactly six R/C/D artifacts.')
        before = {}
        paths = {}
        for name, row in artifacts.items():
            if row['relative_path'] != 'arrays/'+name+'.npy':
                raise ValueError('Unexpected native resume artifact path.')
            paths[name] = _safe_path(root, row['relative_path'])
            before[name] = _descriptor(paths[name])
        checkpoint = load_channel_checkpoint(
            root, expected_manifest_sha256=resume['manifest_sha256'],
            expected_binding=resume['binding'], max_total_bytes=resume['max_total_bytes'])
        if (checkpoint.manifest['iteration'] != seed or checkpoint.record['index'] != seed
                or checkpoint.record['converged'] is not False):
            raise ValueError(f'Resume input must be the complete nonconverged native state{seed}.')
        authenticated = {}
        for name, row in artifacts.items():
            after = _descriptor(paths[name])
            if before[name] != after:
                raise ValueError('Native resume artifact changed during authentication.')
            authenticated[name] = dict(row, path=str(paths[name]),
                                       stat_before=before[name], stat_after=after)
    return checkpoint, {
        'path': str(root), 'manifest_sha256': resume['manifest_sha256'],
        'iteration': seed, 'binding': plain(checkpoint.binding),
        'artifacts_freshly_authenticated': 6, 'artifacts': authenticated,
        'accepted_restart': False,
    }


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config, acceleration = validate_profile(profile)
    first_index, last_index = 16, 21
    storage = shutil.disk_usage(Path(checkpoint_directory).parent)
    if storage.free < profile['free_storage_min_bytes']:
        raise ValueError('Insufficient free storage for the bounded native comparison.')
    parent = profile['parent']
    result.update(passed=False, converged=False, source_commit=profile['source_commit'],
                  source_tree=profile['source_tree'], profile_sha256=profile_sha256,
                  settings=plain(profile['map']), scba_config=plain(config),
                  acceleration_config=plain(acceleration), mode=acceleration.mode,
                  checkpoints=[], attempts=[], evaluated_records=[], decisions=[], first_index=first_index,
                  selected_anchor_index=None, selected_anchor=None, last_evaluated_trial=None,
                  free_storage_before={'free_bytes': storage.free,
                                       'required_bytes': profile['free_storage_min_bytes']},
                  scope={'scba': True, 'fresh_green_evaluations': True,
                         'independent_spectral_storage': True,
                         'initial_state': 'native-channel-continuation',
                         'trajectory_initial_state': 'fresh-zero',
                         'representation': 'retarded-statistical-spectral-v1',
                         'native_checkpoint_imported': True,
                         'historical_contour_checkpoint_imported': False,
                         'cached_green_used': False, 'accepted_restart': False,
                         'accepted_field_update': False, 'converged_transport': False,
                         'frequency_or_q_convergence': False,
                         'exact_conjugate_structure_enforced': False,
                         'primary_identity_uses_independent_spectral': True,
                         'exported_contour_checks_are_diagnostic': True,
                         'all_completed_trials_checkpointed': True,
                         'algorithmic_anchor_is_not_accepted_restart': True,
                         'seed_image_used_only_for_candidate_and_controller_history': True})
    emit('parent_authentication_start')
    fields = load_material_map_fields(
        parent['root'], expected_result_sha256=parent['result_sha256'],
        expected_profile_sha256=parent['profile_sha256'],
        expected_source_commit=parent['source_commit'], expected_source_tree=parent['source_tree'],
        expected_settings=profile['map'], max_total_bytes=profile['parent_max_bytes'])
    result['parent_binding'] = plain(fields.binding)
    emit('parent_authenticated', result['parent_binding'])
    emit('resume_authentication_start', {'path': profile['resume']['path'],
                                       'manifest_sha256': profile['resume']['manifest_sha256']})
    checkpoint, result['resume_binding'] = authenticate_resume(profile['resume'])
    result['seed_evaluated_record'] = plain(checkpoint.record)
    emit('resume_authenticated', result['resume_binding'])
    prepared = mix_channel_spectra(checkpoint.sigma_input, checkpoint.mapped_sigma, config.mixing)
    candidate_hashes = spectrum_hashes(prepared)
    result['initial_candidate'] = {
        'index': first_index, 'mixing': config.mixing, 'sigma_sha256': candidate_hashes,
        'dyson_evaluated': False, 'construction': '.95*evaluated_input15+.05*unevaluated_image15',
        'parent_manifest_sha256': profile['resume']['manifest_sha256'],
    }
    binding = checkpoint_binding(profile, profile_sha256)
    emit('channel_accel_start', dict(result['initial_candidate'], max_evaluations=6,
                                    acceleration=plain(acceleration), native_checkpoint_imported=True))

    def on_start(index, sigma_input, proposal):
        if (index != first_index+len(result['attempts'])
                or not first_index <= index <= last_index or sigma_input is None):
            raise ValueError('Comparison attempted an unexpected or exact-zero candidate.')
        hashes = spectrum_hashes(sigma_input)
        if index == first_index and hashes != candidate_hashes:
            raise ValueError('Candidate16 differs from the prepared native mixed candidate.')
        attempt = {'index': index, 'input_sigma_is_zero': False,
                   'sigma_sha256': hashes, 'dyson_evaluated': False, 'proposal': plain(proposal)}
        result['attempts'].append(attempt)
        emit('scba_evaluation_start', attempt)

    def on_evaluation(state, decision):
        if not result['attempts'] or state.index != result['attempts'][-1]['index']:
            raise ValueError('Completed trial has no matching comparison attempt.')
        decision = plain(decision)
        if (decision.get('index') != state.index
                or type(decision.get('accepted_as_anchor')) is not bool
                or type(decision.get('selected_anchor_index')) is not int
                or not first_index <= decision['selected_anchor_index'] <= state.index
                or (decision['accepted_as_anchor'] and decision['selected_anchor_index'] != state.index)):
            raise ValueError('Controller decision does not identify a valid fresh evaluated anchor.')
        record = plain(state.record)
        record.update(currents=plain(state.evaluation.input_currents), input_sigma_is_zero=False,
                      sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False,
                      sigma_input_sha256=result['attempts'][-1]['sigma_sha256'],
                      mapped_sigma_sha256=spectrum_hashes(state.evaluation.map_sigma),
                      zero_frequency_context=plain(state.evaluation.zero_frequency_context),
                      status='converged' if record['converged'] else 'evaluated',
                      checks=plain(state.evaluation.checks), controller_decision=decision)
        result['evaluated_records'].append(record)
        emit('scba_evaluation_complete', record)
        saved = save_channel_checkpoint(
            checkpoint_directory, iteration=state.index, sigma_input=state.sigma_input,
            mapped_sigma=state.evaluation.map_sigma, record=record, binding=binding,
            max_bytes=profile['checkpoint_max_bytes'], max_checkpoints=config.max_evaluations)
        result['checkpoints'].append(plain(saved))
        emit('checkpoint_saved', plain(saved))
        result['decisions'].append(decision)
        emit('controller_decision', decision)

    solved = run_channel_accel(
        fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
        seed_input=checkpoint.sigma_input, seed_mapped=checkpoint.mapped_sigma,
        sigma_initial=prepared, seed_index=15,
        scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S), g=profile['map']['g'],
        config=config, acceleration=acceleration, on_start=on_start, on_evaluation=on_evaluation)
    if plain(solved.decisions) != result['decisions']:
        raise ValueError('Solver decisions differ from completed callback/checkpoint decisions.')
    result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                  failure_reason=solved.failure_reason, history=plain(solved.history),
                  iteration_plan=plain(solved.plan), completed_evaluations=len(solved.history))
    result['passed'] = solved.failure_reason is None and solved.state is not None
    if solved.state is not None:
        selected = next(row for row in result['evaluated_records'] if row['index'] == solved.state.index)
        result['selected_anchor_index'] = solved.state.index
        result['selected_anchor'] = selected
        result['final_evaluated_state'] = plain(solved.state.record)
        result['final_currents'] = plain(solved.state.evaluation.input_currents)
    if solved.last_evaluated_state is not None:
        result['last_evaluated_trial'] = result['evaluated_records'][-1]
        if result['last_evaluated_trial']['index'] != solved.last_evaluated_state.index:
            raise ValueError('Last evaluated trial differs from the preserved callback sequence.')
    emit('scba_stopped', {'converged': solved.converged, 'stop_reason': solved.stop_reason,
                          'failure_reason': solved.failure_reason,
                          'completed_evaluations': len(solved.history),
                          'selected_anchor_index': result['selected_anchor_index'],
                          'last_evaluated_trial_index': (None if solved.last_evaluated_state is None
                                                         else solved.last_evaluated_state.index)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'schema': 'negfpy.transverse-channel-scba-accel.v1', 'passed': False, 'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)
    try:
        run(profile, result, emit, args.output.parent/'checkpoints', profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve complete checkpoints and actual failure tails
        result['failure_reason'] = type(exc).__name__+': '+str(exc)
        result['failure'] = {'type': type(exc).__name__, 'message': str(exc)}
        traceback.print_exc()
        emit('channel_scba_accel_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('channel_scba_accel_closed', {'passed': result['passed'], 'converged': result['converged'],
                                        'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
