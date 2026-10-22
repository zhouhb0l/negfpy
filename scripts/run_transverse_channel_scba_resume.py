"""Six bounded native-channel evaluations from an explicitly approved seed."""
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
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    mix_channel_spectra,
    run_channel_scba,
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
APPROVED_SEEDS = {3: (RESUME_MANIFEST, OLD_BINDING), 9: (STATE9_MANIFEST, STATE9_BINDING)}
COMPARISON_BASELINE = {
    'job_id': '25919346.pbs101',
    'profile_sha256': 'a26557c1305786bb7e024ade98b634196812aeaba945aa483d0feb953c439ffb',
    'result_sha256': 'f6bb9aca9192dc6908a86688bcc0052c08c3112732581306efe46e421ef87d9f',
    'mixing': .1, 'resume_manifest_sha256': STATE9_MANIFEST, 'resume_iteration': 9,
    'first_index': 10, 'max_evaluations': 6,
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


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-channel-scba-resume.profile.v1':
        raise ValueError('Wrong native channel continuation profile schema.')
    if any(profile['map'].get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('The material physics must remain fixed.')
    resume = profile['resume']
    seed = resume.get('iteration')
    if type(seed) is not int or seed not in APPROVED_SEEDS:
        raise ValueError('Only the explicitly pinned native state3 or state9 seed is approved.')
    if (profile.get('initial_state') != 'native-channel-continuation'
            or profile.get('first_index') != seed+1 or 'restart' in profile):
        raise ValueError(f'This continuation starts with new native candidate{seed+1}.')
    config = ChannelSCBAConfig(**profile['scba'])
    if config.mixing != .1 and not (seed == 9 and config.mixing == .05):
        raise ValueError('Mixing .1 is approved for state3/9; .05 is approved only for state9.')
    if config.mixing == .05:
        baseline = profile.get('comparison_baseline')
        if (not isinstance(baseline, Mapping)
                or any(baseline.get(key) != value for key, value in COMPARISON_BASELINE.items())):
            raise ValueError('The immutable same-seed .1 comparison baseline must be pinned.')
    pinned = {'max_evaluations': 6, 'sigma_rtol': 1e-4, 'sigma_atol': 1e-10,
              'current_rtol': 1e-3, 'current_atol_w': 1e-14,
              'identity_rtol': 1e-8, 'identity_atol': 1e-11,
              'workers': 8, 'external_tile': 16, 'bin_tile': 1,
              'max_contraction_work': 100_000_000_000_000, 'max_dyson_evaluations': 2404}
    if any(getattr(config, k) != v for k, v in pinned.items()):
        raise ValueError('Continuation bounds, mixing, resources or tolerances differ.')
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
    manifest_sha256, source_binding = APPROVED_SEEDS[seed]
    if (resume.get('manifest_sha256') != manifest_sha256
            or not isinstance(resume.get('path'), str) or not resume['path']
            or type(resume.get('max_total_bytes')) is not int
            or resume['max_total_bytes'] < 6*(16*4*601*120**2+65536)):
        raise ValueError(f'The exact complete native state{seed} checkpoint must be pinned.')
    expected_binding = dict(source_binding, physics=profile['map'],
                            parent_result_sha256=profile['parent']['result_sha256'],
                            parent_profile_sha256=profile['parent']['profile_sha256'])
    if resume['binding'] != expected_binding:
        raise ValueError('Original checkpoint source/profile, physics or parent binding differs.')
    return config


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
    if type(seed) is not int or seed not in APPROVED_SEEDS:
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
    config = validate_profile(profile)
    seed = profile['resume']['iteration']
    first_index = seed+1
    last_index = first_index+config.max_evaluations-1
    storage = shutil.disk_usage(Path(checkpoint_directory).parent)
    if storage.free < profile['free_storage_min_bytes']:
        raise ValueError('Insufficient free storage for the bounded native continuation.')
    parent = profile['parent']
    result.update(passed=False, converged=False, source_commit=profile['source_commit'],
                  source_tree=profile['source_tree'], profile_sha256=profile_sha256,
                  settings=profile['map'], scba_config=plain(config), checkpoints=[], attempts=[],
                  evaluated_records=[], first_index=first_index,
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
                         'exported_contour_checks_are_diagnostic': True})
    if config.mixing == .05:
        result['comparison_baseline'] = plain(profile['comparison_baseline'])
        result['scope']['comparison_baseline_used_computationally'] = False
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
    emit('resume_authenticated', result['resume_binding'])
    prepared = {'candidate': mix_channel_spectra(checkpoint.sigma_input, checkpoint.mapped_sigma, config.mixing)}
    candidate_hashes = spectrum_hashes(prepared['candidate'])
    result['initial_candidate'] = {
        'index': first_index, 'mixing': config.mixing, 'sigma_sha256': candidate_hashes,
        'dyson_evaluated': False,
        'construction': (f'.95*evaluated_input{seed}+.05*unevaluated_image{seed}'
                         if config.mixing == .05 else
                         f'.9*evaluated_input{seed}+.1*unevaluated_image{seed}'),
        'parent_manifest_sha256': profile['resume']['manifest_sha256'],
    }
    # The candidate owns new fields, so the six resume mmap references need not
    # remain alive during the new solver trajectory.
    del checkpoint
    binding = {'parent_result_sha256': parent['result_sha256'],
               'parent_profile_sha256': parent['profile_sha256'],
               'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
               'profile_sha256': profile_sha256, 'physics': profile['map'],
               'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
               'resume_parent_manifest_sha256': profile['resume']['manifest_sha256'],
               'resume_parent_iteration': seed}
    if seed == 9:
        # Preserve state9's own parent3 provenance, rather than relabeling it
        # with the new source/profile or discarding the inherited binding.
        binding['resume_parent_binding'] = plain(profile['resume']['binding'])
    emit('channel_resume_start', dict(result['initial_candidate'], max_evaluations=6,
                                     native_checkpoint_imported=True))

    def on_start(index, sigma_input):
        if (index != first_index+len(result['attempts'])
                or not first_index <= index <= last_index or sigma_input is None):
            raise ValueError('Continuation attempted an unexpected or exact-zero candidate.')
        hashes = spectrum_hashes(sigma_input)
        if index == first_index and hashes != candidate_hashes:
            raise ValueError(f'Candidate{first_index} differs from the prepared native mixed candidate.')
        attempt = {'index': index, 'input_sigma_is_zero': False,
                   'sigma_sha256': hashes, 'dyson_evaluated': False}
        result['attempts'].append(attempt)
        emit('scba_evaluation_start', attempt)

    def on_evaluation(state):
        if not result['attempts'] or state.index != result['attempts'][-1]['index']:
            raise ValueError('Completed state has no matching continuation attempt.')
        record = plain(state.record)
        record.update(currents=plain(state.evaluation.input_currents), input_sigma_is_zero=False,
                      sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False,
                      sigma_input_sha256=result['attempts'][-1]['sigma_sha256'],
                      mapped_sigma_sha256=spectrum_hashes(state.evaluation.map_sigma),
                      zero_frequency_context=plain(state.evaluation.zero_frequency_context),
                      status='converged' if record['converged'] else 'evaluated',
                      checks=plain(state.evaluation.checks))
        result['evaluated_records'].append(record)
        emit('scba_evaluation_complete', record)
        saved = save_channel_checkpoint(
            checkpoint_directory, iteration=state.index, sigma_input=state.sigma_input,
            mapped_sigma=state.evaluation.map_sigma, record=record, binding=binding,
            max_bytes=profile['checkpoint_max_bytes'], max_checkpoints=config.max_evaluations)
        result['checkpoints'].append(plain(saved))
        emit('checkpoint_saved', plain(saved))

    solved = run_channel_scba(
        fields.grid, fields.vertex, fields.dynamical, fields.left, fields.right,
        sigma_initial=prepared.pop('candidate'), first_index=first_index,
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
    result = {'schema': 'negfpy.transverse-channel-scba-resume.v1', 'passed': False, 'converged': False}

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
        emit('channel_scba_resume_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('channel_scba_resume_closed', {'passed': result['passed'], 'converged': result['converged'],
                                        'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
