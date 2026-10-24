"""Two fresh full-domain SCBA states for an actual-IFC boundary-support model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import shutil
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    plan_channel_scba,
    run_channel_scba,
)
from negfpy.anharmonic.transverse_channel_restart import save_channel_checkpoint
from negfpy.anharmonic.transverse_crossing_feedback import (
    MODEL,
    build_crossing_feedback_vertex,
)
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_crossing_feedback_static', Path(__file__).with_name('run_transverse_crossing_diagnostic.py'))
_static = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_static)
_material = _static._material
PARENT, FROZEN_MAP, RESOURCES, RAW_INPUTS = (
    _static.PARENT, _static.FROZEN_MAP, _static.RESOURCES, _static.RAW_INPUTS)
SCBA = {
    'max_evaluations': 2, 'mixing': .05, 'sigma_rtol': 1e-4, 'sigma_atol': 1e-10,
    'current_rtol': 1e-3, 'current_atol_w': 1e-14,
    'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'max_seconds': 79200, 'time_reserve_seconds': 300,
    'next_map_safety_factor': 1.25, 'initial_map_estimate_seconds': 21600,
    'max_bytes': 128*1024**3, 'max_contraction_work': 1_000_000_000_000_000,
    'max_dyson_evaluations': 2404, 'workers': 8, 'external_tile': 16, 'bin_tile': 1,
}
SUPPORT = {'model': 'actual-ifc-any-leg-central-crop-v1',
    'max_bytes': 4*1024**3, 'max_workspace_bytes': 32*1024**3,
    'max_geometry_evaluations': 2_000_000, 'max_phase_evaluations': 1_000_000_000}
CHECKPOINT_MAX_BYTES = 16*1024**3
MIN_FREE_BYTES = 64*1024**3


def plain(value):
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return _material.plain(value)


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-crossing-feedback.profile.v1':
        raise ValueError('Wrong crossing feedback profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT), ('resources', RESOURCES),
                          ('runtime_inputs', RAW_INPUTS), ('scba', SCBA), ('support', SUPPORT),
                          ('checkpoint_max_bytes', CHECKPOINT_MAX_BYTES),
                          ('min_free_bytes', MIN_FREE_BYTES), ('initial_state', 'fresh-zero'),
                          ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    if any(key in profile for key in ('resume', 'restart', 'checkpoints', 'sigma_initial')):
        raise ValueError('This new Hamiltonian starts from exact zero; no old state is imported.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return ChannelSCBAConfig(**SCBA)



def zero_support_context(spectrum, zero_index):
    """Small central/outer block summaries; these are never new acceptance gates."""
    domains = {'full': np.arange(216), 'central': np.arange(48, 168),
               'outer': np.r_[np.arange(48), np.arange(168, 216)]}
    result = {'q_index': 0, 'frequency_index': int(zero_index),
              'diagnostic_only': True, 'exact_zero_input': spectrum is None,
              'domain_semantics': 'full matrix and central/outer diagonal subblocks; norms are not additive'}
    for name in ('retarded', 'statistical', 'spectral'):
        rows = {}
        for label, indices in domains.items():
            if spectrum is None:
                rows[label] = {'frobenius': 0., 'max_abs': 0.}
            else:
                matrix = getattr(spectrum, name)[0, zero_index]
                block = matrix if label == 'full' else matrix[np.ix_(indices, indices)]
                rows[label] = {'frobenius': float(np.linalg.norm(block)),
                               'max_abs': float(np.max(np.abs(block), initial=0.))}
        result[name] = rows
    return result


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config = validate_profile(profile)
    _material._digest(profile_sha256)
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(checkpoint_directory.parent).free
    if free < profile['min_free_bytes']:
        raise ValueError('Insufficient free checkpoint storage for this bounded pilot.')
    result.update(schema='negfpy.transverse-crossing-feedback.v1', passed=False, converged=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), scba_config=plain(config),
        attempts=[], evaluated_records=[], checkpoints=[], completed_evaluations=0,
        failure_reason=None, free_storage_at_admission=free,
        scope={'scba': True, 'initial_state': 'fresh-zero', 'new_trajectory': True,
            'representation': 'retarded-statistical-spectral-v1',
            'historical_checkpoint_imported': False, 'cached_green_or_born_used': False,
            'full_domain_self_energy_and_currents': True, 'raw_ifc_modified': False,
            'support_model': MODEL, 'finite_research_hamiltonian': True,
            'global_acoustic_sum_rule_restored': False, 'unique_physical_interface_claimed': False,
            'original_guards_and_six_convergence_screens_unchanged': True,
            'accepted_restart': False, 'accepted_field_update': False,
            'accepted_transport': False, 'frequency_or_q_convergence': False})
    started = time.monotonic()
    deadline = started+config.max_seconds-config.time_reserve_seconds
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('feedback_inputs_authenticated', {'count': len(pins)})
        mesh, grid, reference, dynamical, contacts, arrays, authentication = _static.load_static(
            profile, documents, stack)
        try:
            normalized, geometry, result['material'] = _material.rebuild_material(profile)
            emit('feedback_vertex_start')
            supported = build_crossing_feedback_vertex(
                mesh, normalized, geometry, n_device_layers=9, interaction_layers=(2, 3, 4, 5, 6),
                scale=SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S),
                reference_vertex=reference, deadline_monotonic=deadline,
                **{key: value for key, value in SUPPORT.items() if key != 'model'})
            vertex = supported.vertex
            if vertex.full_ndof != 216 or vertex.active_dofs.tolist() != list(range(216)):
                raise ValueError('The feedback and current domain must cover all216 coordinates.')
            result['support_model'] = plain(supported.metadata)
            model_identity = result['support_model']['model_identity_sha256']
            _material._digest(model_identity)
            emit('feedback_vertex_complete', result['support_model'])
            plan = plan_channel_scba(mesh.n_qpoints, grid.n_points, 216, vertex.ndof,
                len(contacts[0].device_indices), len(contacts[1].device_indices), config=config)
            result['iteration_plan'] = plain(plan)
            if plan.peak_owned_bytes > config.max_bytes:
                raise ValueError('Full-domain SCBA memory plan exceeds its admission bound.')
            binding = {'parent_result_sha256': PARENT['result_sha256'],
                'parent_profile_sha256': PARENT['profile_sha256'],
                'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
                'profile_sha256': profile_sha256, 'physics': plain(profile['map']),
                'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
                'support_model': MODEL, 'model_identity_sha256': model_identity,
                'support_configuration': plain(profile['support']),
                'feedback_dofs': list(range(216)), 'new_hamiltonian_trajectory': True}

            def on_start(index, sigma_input):
                attempt = {'index': index, 'input_sigma_is_zero': sigma_input is None,
                           'dyson_evaluated': False, 'new_trajectory': True}
                result['attempts'].append(attempt)
                emit('scba_evaluation_start', attempt)

            def on_evaluation(state):
                record = plain(state.record)
                record.update(currents=plain(state.evaluation.input_currents),
                    input_sigma_is_zero=state.sigma_input is None, sigma_input_evaluated=True,
                    mapped_sigma_dyson_evaluated=False,
                    status='converged' if record['converged'] else 'evaluated',
                    checks=plain(state.evaluation.checks),
                    zero_frequency_support={'input': zero_support_context(state.sigma_input, grid.n_points//2),
                        'mapped': zero_support_context(state.evaluation.map_sigma, grid.n_points//2)})
                result['evaluated_records'].append(record)
                result['completed_evaluations'] = len(result['evaluated_records'])
                emit('scba_evaluation_complete', record)
                saved = save_channel_checkpoint(checkpoint_directory, iteration=state.index,
                    sigma_input=state.sigma_input, mapped_sigma=state.evaluation.map_sigma,
                    record=record, binding=binding, max_bytes=profile['checkpoint_max_bytes'],
                    max_checkpoints=config.max_evaluations)
                # Keep concise checkpoint identities; full manifests live beside the fields.
                receipt = {key: saved[key] for key in ('path', 'manifest_sha256')}
                result['checkpoints'].append(receipt)
                emit('checkpoint_saved', receipt)

            emit('crossing_feedback_start', {'max_evaluations': 2, 'mixing': .05,
                'initial_state': 'fresh-zero', 'feedback_ndof': 216})
            # Mapping/authentication consume the same overall runtime allowance.
            effective_config = replace(config, max_seconds=config.max_seconds-(time.monotonic()-started))
            result['effective_scba_config'] = plain(effective_config)
            solved = run_channel_scba(grid, vertex, dynamical, *contacts, sigma_initial=None,
                first_index=1, scale=SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S),
                g=profile['map']['g'], config=effective_config, on_start=on_start, on_evaluation=on_evaluation)
            result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                failure_reason=solved.failure_reason, history=plain(solved.history),
                completed_evaluations=len(solved.history),
                passed=solved.failure_reason is None and solved.state is not None)
            if solved.state is not None:
                result['final_evaluated_state'] = plain(solved.state.record)
                result['final_currents'] = plain(solved.state.evaluation.input_currents)
            _material.verify_unchanged(pins+list(authentication.values()))
            emit('scba_stopped', {'converged': solved.converged,
                'completed_evaluations': len(solved.history), 'stop_reason': solved.stop_reason,
                'failure_reason': solved.failure_reason})
        finally:
            _static.release(arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    result = {'passed': False, 'converged': False, 'failure_reason': None}
    started = time.monotonic()

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)

    try:
        run(profile, result, emit, args.output.parent/'checkpoints', profile_sha256=digest)
    except Exception as exc:  # noqa: BLE001 - retain actual failure and any published checkpoint
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('crossing_feedback_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('crossing_feedback_closed', {'passed': result['passed'], 'converged': result['converged'],
                                     'completed_evaluations': result.get('completed_evaluations', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
