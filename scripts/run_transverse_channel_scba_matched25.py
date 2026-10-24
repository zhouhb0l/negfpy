"""Matched twelve-map native-channel continuations from pinned selected25."""
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
    ChannelSelectionPolicy,
    run_channel_accel,
)
from negfpy.anharmonic.transverse_channel_feasibility import ChannelProposalFeasibility
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
STATE15_MANIFEST = 'f605ff69ba1b040ada59cbb26c5d4ff06b9c77c94585a828f3bbbd4a80caa5ef'
STATE19_MANIFEST = '674c9c1f5126d65c24f5e934571995cda1d1aa1ae61fc0d1dd90136c5e8d5205'
STATE15_BINDING = {
    'source_commit': '3c4af426d73373437a379bd8e0c15e7802e05046',
    'source_tree': 'caf5698fc3c022d0849a2f0415257f7b0ea8d250',
    'profile_sha256': 'b0e2cb2c3ee0fa14c7d598a49f04ec520c2bad7d08b1f96f26683a0ad9e027a7',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
    'resume_parent_manifest_sha256': STATE9_MANIFEST, 'resume_parent_iteration': 9,
}
STATE19_BINDING = {
    'source_commit': '77d2f5b95d16be1abd806f68448af1819ce654d6',
    'source_tree': 'b83cdca57e272e3eaa11558cadae286006ac1ac7',
    'profile_sha256': '447ceed21e61235988b76fa23797075e21b2f3f0ac64ee093d9488fdf4ba8092',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
    'resume_parent_manifest_sha256': STATE15_MANIFEST, 'resume_parent_iteration': 15,
    'acceleration': {'mode': 'anderson1', 'coefficient_limit': 5.,
                     'trust_radius': .25, 'rank_rtol': 1e-10},
    'selection_policy': {'mode': 'bounded-tradeoff', 'component_growth_limit': 1.02,
                         'merit_ratio_limit': .99},
}
SEED_MANIFEST = '70c5c8cdc1f56ed1b88ddf115ed8492442aa4c534a3f0e5386ea0d3182844731'
SEED_BINDING = {
    'source_commit': '7d2daf006db4ebf45254abd4f35433b33369a2cd',
    'source_tree': '83d264401aa243cecb33275ef8d9cc5d03f77f30',
    'profile_sha256': '1e937184a6c3b681939f28cc06a7d7a590e441e02b93a36bba333678b069a902',
    'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
    'resume_parent_manifest_sha256': STATE19_MANIFEST, 'resume_parent_iteration': 19,
    'acceleration': {'mode': 'anderson1', 'coefficient_limit': 5.,
                     'trust_radius': .25, 'rank_rtol': 1e-10},
    'selection_policy': {'mode': 'bounded-tradeoff', 'component_growth_limit': 1.02,
                         'merit_ratio_limit': .99},
    'proposal_feasibility': {'mode': 'psd-backtracking', 'factor': .5, 'max_halvings': 8},
}
MATCHED_INVARIANTS = {
    'controller_history_reset': True, 'common_seed_iteration': 25,
    'common_first_index': 26, 'max_fresh_attempts': 12, 'last_possible_index': 37,
    'resume_parent_job': '25939637.pbs101',
    'comparison_arms': ['anderson1', 'linear-control'],
    'comparison_baselines_used_computationally': False,
    'no_automatic_retry_or_extension': True,
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
    """Recreate exact selected25 binding, retaining native19/15/9/3 ancestry."""
    shared = {'physics': plain(profile['map']),
              'parent_result_sha256': profile['parent']['result_sha256'],
              'parent_profile_sha256': profile['parent']['profile_sha256']}
    state9 = dict(STATE9_BINDING, **shared)
    state15 = dict(STATE15_BINDING, **shared, resume_parent_binding=state9)
    state19 = dict(STATE19_BINDING, **shared, resume_parent_binding=state15)
    return dict(SEED_BINDING, **shared, resume_parent_binding=state19)


def checkpoint_binding(profile, profile_sha256):
    """Bind every evaluated trial, including rejected ones, to this strategy."""
    binding = {'parent_result_sha256': profile['parent']['result_sha256'],
            'parent_profile_sha256': profile['parent']['profile_sha256'],
            'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
            'profile_sha256': profile_sha256, 'physics': plain(profile['map']),
            'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
            'resume_parent_manifest_sha256': profile['resume']['manifest_sha256'],
            'resume_parent_iteration': 25,
            'resume_parent_binding': plain(profile['resume']['binding']),
            'acceleration': plain(profile['acceleration'])}
    if 'selection_policy' in profile:
        binding['selection_policy'] = plain(profile['selection_policy'])
    binding['proposal_feasibility'] = plain(profile['proposal_feasibility'])
    binding['matched_experiment'] = plain(profile['matched_experiment'])
    return binding


def validate_selection_policy(profile):
    """Keep the completed bounded-selection study's ranking rule fixed."""
    policy = ChannelSelectionPolicy(**profile['selection_policy'])
    if (policy.mode != 'bounded-tradeoff' or policy.component_growth_limit != 1.02
            or policy.merit_ratio_limit != .99
            or profile['acceleration']['mode'] not in ('anderson1', 'linear-control')):
        raise ValueError('The explicit bounded selection-policy study differs.')
    return policy


def validate_feasibility_profile(profile):
    """Pin the bounded cheap-screen policy separately from physical tolerances."""
    policy = ChannelProposalFeasibility(**profile['proposal_feasibility'])
    if (policy.mode != 'psd-backtracking' or policy.factor != .5
            or policy.max_halvings != 8):
        raise ValueError('The explicit bounded proposal-feasibility policy differs.')
    return policy


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-channel-scba-matched25.profile.v1':
        raise ValueError('Wrong native channel feasibility profile schema.')
    if any(profile['map'].get(k) != v for k, v in PHYSICS.items()):
        raise ValueError('The material physics must remain fixed.')
    if (profile.get('initial_state') != 'native-channel-continuation'
            or profile.get('first_index') != 26 or 'restart' in profile):
        raise ValueError('This continuation starts with new native candidate26.')
    config = ChannelSCBAConfig(**profile['scba'])
    pinned = {'max_evaluations': 12, 'mixing': .05,
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
    validate_selection_policy(profile)
    validate_feasibility_profile(profile)
    experiment = profile.get('matched_experiment')
    if (not isinstance(experiment, dict)
            or any(type(experiment.get(k)) is not type(v) or experiment[k] != v
                   for k, v in MATCHED_INVARIANTS.items())):
        raise ValueError('Matched continuation budget, seed or history-reset metadata differs.')
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
    if (type(resume.get('iteration')) is not int or resume['iteration'] != 25
            or resume.get('manifest_sha256') != SEED_MANIFEST
            or not isinstance(resume.get('path'), str) or not resume['path']
            or type(resume.get('max_total_bytes')) is not int
            or resume['max_total_bytes'] < 6*(16*4*601*120**2+65536)):
        raise ValueError('The exact complete selected native state25 checkpoint must be pinned.')
    if resume['binding'] != seed_binding(profile):
        raise ValueError('Original selected25 binding differs, including native19/15/9/3 lineage.')
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
    if type(seed) is not int or seed != 25:
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
                or checkpoint.record['converged'] is not False
                or checkpoint.record.get('controller_decision', {}).get('accepted_as_anchor') is not True
                or checkpoint.record['controller_decision'].get('selected_anchor_index') != seed):
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
    selection = validate_selection_policy(profile)
    feasibility = validate_feasibility_profile(profile)
    first_index, last_index = 26, 37
    storage = shutil.disk_usage(Path(checkpoint_directory).parent)
    if storage.free < profile['free_storage_min_bytes']:
        raise ValueError('Insufficient free storage for the bounded native comparison.')
    parent = profile['parent']
    result.update(passed=False, converged=False, source_commit=profile['source_commit'],
                  source_tree=profile['source_tree'], profile_sha256=profile_sha256,
                  settings=plain(profile['map']), scba_config=plain(config),
                  acceleration_config=plain(acceleration), mode=acceleration.mode,
                  selection_policy_config=plain(selection),
                  proposal_feasibility_config=plain(feasibility), proposal_screens=[],
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
                         'seed_image_used_only_for_candidate_and_controller_history': True,
                         'comparison_baselines_used_computationally': False,
                         'proposal_feasibility_screening_is_not_fresh_map': True,
                         'proposal_damping_is_not_physical_projection': True,
                         'controller_history_reset': True,
                         'first_candidate26_linear_once': True,
                         'seed25_only_history': True, 'previous24_imported': False})
    result['matched_experiment'] = plain(profile['matched_experiment'])
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
        'dyson_evaluated': False, 'construction': '.95*evaluated_input25+.05*unevaluated_image25',
        'parent_manifest_sha256': profile['resume']['manifest_sha256'],
    }
    binding = checkpoint_binding(profile, profile_sha256)
    emit('channel_accel_start', dict(result['initial_candidate'], max_evaluations=12,
                                    acceleration=plain(acceleration), selection_policy=plain(selection),
                                    proposal_feasibility=plain(feasibility),
                                    native_checkpoint_imported=True))

    def on_proposal(index, batch):
        batch = plain(batch)
        if (index != first_index+len(result['attempts'])
                or not first_index <= index <= last_index
                or len(result['proposal_screens']) != len(result['attempts'])
                or batch.get('index') != index):
            raise ValueError('Proposal screening does not identify the next fresh attempt.')
        result['proposal_screens'].append(batch)
        emit('proposal_screened', batch)

    def on_start(index, sigma_input, proposal):
        if (index != first_index+len(result['attempts'])
                or not first_index <= index <= last_index or sigma_input is None):
            raise ValueError('Comparison attempted an unexpected or exact-zero candidate.')
        if (len(result['proposal_screens']) != len(result['attempts'])+1
                or result['proposal_screens'][-1]['index'] != index
                or result['proposal_screens'][-1].get('passed') is not True
                or result['proposal_screens'][-1].get('map_admitted') is not True
                or plain(proposal.get('feasibility')) != result['proposal_screens'][-1]):
            raise ValueError('Fresh map attempt lacks its admitted feasibility screening batch.')
        hashes = spectrum_hashes(sigma_input)
        if index == first_index and hashes != candidate_hashes:
            raise ValueError('Candidate26 differs from the prepared native mixed candidate.')
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
        sigma_initial=prepared, seed_index=25,
        scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S), g=profile['map']['g'],
        config=config, acceleration=acceleration, selection=selection, feasibility=feasibility,
        on_proposal=on_proposal, on_start=on_start, on_evaluation=on_evaluation)
    if plain(solved.decisions) != result['decisions']:
        raise ValueError('Solver decisions differ from completed callback/checkpoint decisions.')
    if plain(solved.proposal_screens) != result['proposal_screens']:
        raise ValueError('Solver proposal screens differ from emitted screening batches.')
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
                          'proposal_screening_batches': len(result['proposal_screens']),
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
