"""Two fresh native-channel SCBA states for a declared complete site-energy surrogate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import resource
import shutil
import time
import traceback
from contextlib import ExitStack
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic.site_energy_boundary import build_site_energy_vertex
from negfpy.anharmonic.site_energy_model import fit_site_energy_model
from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSCBAConfig,
    plan_channel_scba,
    run_channel_scba,
)
from negfpy.anharmonic.transverse_channel_restart import save_channel_checkpoint
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_site_feedback_source', Path(__file__).with_name('run_site_energy_born.py'))
_born = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_born)
_operator_spec = importlib.util.spec_from_file_location(
    '_site_feedback_operator', Path(__file__).with_name('run_transverse_soft_source.py'))
_operator = importlib.util.module_from_spec(_operator_spec)
_operator_spec.loader.exec_module(_operator)
_material = _born._material
PARENT, FROZEN_MAP, RESOURCES, RAW_INPUTS, FIT, SITE_SUPPORT = (
    _born.PARENT, _born.FROZEN_MAP, _born.RESOURCES, _born.RAW_INPUTS, _born.FIT, _born.SITE_SUPPORT)
SCBA = _born._acoustic._feedback.SCBA
CHECKPOINT_MAX_BYTES = 8*1024**3
MIN_FREE_BYTES = 64*1024**3
MODEL = 'fitted-complete-center-star-v1'
plain = _born.plain
load_material_for_study = _born.load_material_for_study
load_static_operator = _operator.load_static_operator


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.site-energy-feedback.profile.v1':
        raise ValueError('Wrong site-energy-feedback profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT), ('resources', RESOURCES),
                          ('runtime_inputs', RAW_INPUTS), ('fit', FIT), ('site_support', SITE_SUPPORT),
                          ('scba', SCBA), ('checkpoint_max_bytes', CHECKPOINT_MAX_BYTES),
                          ('min_free_bytes', MIN_FREE_BYTES), ('initial_state', 'fresh-zero'),
                          ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    if any(key in profile for key in ('resume', 'restart', 'checkpoints', 'checkpoint', 'sigma_initial')):
        raise ValueError('The new fitted Hamiltonian starts from exact zero; no old checkpoint is imported.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return ChannelSCBAConfig(**SCBA)


def zero_support_context(spectrum, active_dofs, zero_index, *, full_ndof=216):
    """Report diagonal blocks in global coordinates without padding stored fields."""
    active = np.asarray(active_dofs)
    if (active.ndim != 1 or active.dtype.kind not in 'iu' or len(np.unique(active)) != len(active)
            or np.any(active < 0) or np.any(active >= full_ndof)):
        raise ValueError('Stored interaction coordinates must be unique full-device addresses.')
    is_central = (active >= 48) & (active < 168)
    domains = {'full': np.arange(len(active)), 'central': np.flatnonzero(is_central),
               'outer': np.flatnonzero(~is_central)}
    result = {'q_index': 0, 'frequency_index': int(zero_index), 'diagnostic_only': True,
        'exact_zero_input': spectrum is None, 'full_device_dofs': full_ndof,
        'stored_active_dofs_count': len(active), 'outside_active_exact_zero': True,
        'central_dofs_count': int(np.count_nonzero(is_central)),
        'outer_dofs_count': int(np.count_nonzero(~is_central)),
        'domain_semantics': ('full zero-embedded interaction matrix and central/outer diagonal '
                             'subblocks in global216 coordinates; norms are not additive')}
    for name in ('retarded', 'statistical', 'spectral'):
        rows = {}
        for label, indices in domains.items():
            if spectrum is None:
                rows[label] = {'frobenius': 0., 'max_abs': 0.}
            else:
                matrix = getattr(spectrum, name)[0, zero_index]
                if matrix.shape != (len(active), len(active)):
                    raise ValueError('Stored channel matrix differs from active coordinate count.')
                block = matrix if label == 'full' else matrix[np.ix_(indices, indices)]
                rows[label] = {'frobenius': float(np.linalg.norm(block)),
                               'max_abs': float(np.max(np.abs(block), initial=0.))}
        result[name] = rows
    return result


def save_model_parameters(runtime_directory, fitted, profile, *, profile_sha256,
                          model_identity_sha256, energy_identity_sha256):
    """Preserve exact research-energy coefficients without emitting an IFC tensor."""
    coefficients = fitted.coefficients
    if (not isinstance(coefficients, np.ndarray) or coefficients.shape != (2, 364)
            or coefficients.dtype != np.dtype('float64') or not np.all(np.isfinite(coefficients))):
        raise ValueError('The research energy requires728 finite float64 coefficients in two classes.')
    for identity in (profile_sha256, model_identity_sha256, energy_identity_sha256):
        _material._digest(identity)
    directory = Path(runtime_directory)/'model-parameters'
    directory.mkdir(exist_ok=False)
    path = directory/'coefficients.npy'
    with path.open('xb') as stream:
        np.save(stream, coefficients, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    descriptor = {
        'schema': 'negfpy.site-energy-parameters.v1',
        'relative_path': 'model-parameters/coefficients.npy',
        'descriptor_relative_path': 'model-parameters/descriptor.json',
        'shape': list(coefficients.shape), 'dtype': str(coefficients.dtype),
        'array_bytes': coefficients.nbytes, 'file_bytes': path.stat().st_size,
        'file_sha256': digest,
        'hash_domain': 'whole NPY file including header',
        'energy_identity_sha256': energy_identity_sha256,
        'model_identity_sha256': model_identity_sha256,
        'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
        'profile_sha256': profile_sha256, 'fit_configuration': plain(profile['fit']),
        'coefficient_semantics': ('two primitive-center classes; ordered cubic site-star monomials '
                                  'and normalization fixed by the bound source and energy identity'),
        'role': 'exact energy parameters used for this declared research trajectory',
        'research_trajectory_parameters': True, 'production_ifc_adoption': False,
        'actual_ifc_tensor': False,
    }
    with (directory/'descriptor.json').open('x') as stream:
        json.dump(descriptor, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    return descriptor


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config = validate_profile(profile)
    _material._digest(profile_sha256)
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(checkpoint_directory.parent).free
    if free < profile['min_free_bytes']:
        raise ValueError('Insufficient free checkpoint storage for the bounded new trajectory.')
    started = time.monotonic()
    deadline = started+config.max_seconds-config.time_reserve_seconds
    result.update(schema='negfpy.site-energy-feedback.v1', passed=False, converged=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), scba_config=plain(config),
        fit_config=plain(profile['fit']), attempts=[], evaluated_records=[], checkpoints=[],
        completed_evaluations=0, failure_reason=None, free_storage_at_admission=free,
        scope={'scba': True, 'initial_state': 'fresh-zero', 'new_hamiltonian_trajectory': True,
            'support_model': MODEL, 'explicit_approximate_research_surrogate': True,
            'surrogate_includes_raw_fit_remainder': False, 'bulk_fit_distortion_reported': True,
            'source_asr_repaired': False, 'raw_ifc_modified': False,
            'complete_center_energies_selected': True, 'all_outer_derivatives_retained': True,
            'derivative_crop': False, 'full216_green_and_contact_currents': True,
            'interaction_exact_zero_outside_active_endpoint_union': True,
            'historical_checkpoint_imported': False, 'cached_green_or_born_used': False,
            'representation': 'retarded-statistical-spectral-v1',
            'original_guards_and_six_convergence_screens_unchanged': True,
            'fit_solver_convergence_required_before_full_maps': True,
            'research_energy_parameters_persisted': False,
            'unique_physical_interface_claimed': False, 'production_ifc_adoption': False,
            'accepted_restart': False, 'accepted_field_update': False,
            'accepted_transport': False, 'frequency_or_q_convergence': False})
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('site_feedback_inputs_authenticated', {'count': len(pins)})
        mesh, grid, dynamical, contacts, static_arrays, static_auth = load_static_operator(
            profile, documents, stack)
        try:
            _, geometry, crystal, raw_values, classes, result['material'] = load_material_for_study(profile)
            emit('site_feedback_fit_start')
            fitted = fit_site_energy_model(
                raw_values, crystal.supercell.scaled_positions, crystal.supercell.lattice,
                classes, deadline_monotonic=deadline, **FIT)
            result['fit_report'] = plain(fitted.fit_report)
            result['fit_geometry'] = plain(fitted.geometry)
            fit = result['fit_report']['fit']
            if fit['parameter_count'] != 728 or fit['bounded_solver_converged'] is not True:
                raise ValueError('The frozen728-parameter bulk fit must converge before new full maps.')
            emit('site_feedback_fit_complete', {key: fit[key] for key in
                ('parameter_count', 'iterations', 'stop_code', 'bounded_solver_converged')})
            scale = SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S)
            supported = build_site_energy_vertex(
                mesh, fitted, crystal, geometry, n_device_layers=9,
                interaction_layers=(2, 3, 4, 5, 6), scale=scale,
                force_constant_units=result['material']['force_constant_units'],
                mass_units=result['material']['mass_units'], crop_derivatives=False,
                deadline_monotonic=deadline, **SITE_SUPPORT)
            vertex = supported.vertex
            active = vertex.active_dofs
            if (vertex.full_ndof != 216 or vertex.ndof != 132
                    or not np.all(np.isin(np.arange(48, 168), active))):
                raise ValueError('The declared complete-energy endpoint union must cover132 of216 DOFs.')
            result['support_model'] = plain(supported.metadata)
            model_identity = result['support_model']['model_identity_sha256']
            energy_identity = result['support_model']['energy_identity_sha256']
            _material._digest(model_identity)
            _material._digest(energy_identity)
            result['model_parameters'] = save_model_parameters(
                checkpoint_directory.parent, fitted, profile, profile_sha256=profile_sha256,
                model_identity_sha256=model_identity, energy_identity_sha256=energy_identity)
            result['scope']['research_energy_parameters_persisted'] = True
            emit('site_feedback_parameters_saved', {
                key: result['model_parameters'][key] for key in ('relative_path', 'file_bytes')})
            emit('site_feedback_vertex_complete', {'model_identity_sha256': model_identity,
                'energy_identity_sha256': energy_identity, 'active_dofs': vertex.ndof, 'full_ndof': 216})
            plan = plan_channel_scba(mesh.n_qpoints, grid.n_points, 216, vertex.ndof,
                len(contacts[0].device_indices), len(contacts[1].device_indices), config=config)
            result['iteration_plan'] = plain(plan)
            if plan.peak_owned_bytes > config.max_bytes:
                raise ValueError('The full-device SCBA allocation plan exceeds max_bytes.')
            binding = {
                'parent_result_sha256': PARENT['result_sha256'],
                'parent_profile_sha256': PARENT['profile_sha256'],
                'source_commit': profile['source_commit'], 'source_tree': profile['source_tree'],
                'profile_sha256': profile_sha256, 'physics': plain(profile['map']),
                'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
                'support_model': MODEL, 'model_identity_sha256': model_identity,
                'energy_identity_sha256': energy_identity, 'fit_configuration': plain(profile['fit']),
                'bulk_fit_residual': plain(result['fit_report']['residual']),
                'model_parameters': plain(result['model_parameters']),
                'site_support_configuration': plain(profile['site_support']),
                'feedback_dofs': active.tolist(), 'full_device_dofs': 216,
                'outside_feedback_dofs_exact_zero': True, 'new_hamiltonian_trajectory': True,
                'raw_fit_remainder_in_surrogate': False, 'production_ifc_adoption': False}

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
                    zero_frequency_support={
                        'input': zero_support_context(state.sigma_input, active, grid.n_points//2),
                        'mapped': zero_support_context(state.evaluation.map_sigma, active, grid.n_points//2)})
                result['evaluated_records'].append(record)
                result['completed_evaluations'] = len(result['evaluated_records'])
                emit('scba_evaluation_complete', record)
                saved = save_channel_checkpoint(checkpoint_directory, iteration=state.index,
                    sigma_input=state.sigma_input, mapped_sigma=state.evaluation.map_sigma,
                    record=record, binding=binding, max_bytes=profile['checkpoint_max_bytes'],
                    max_checkpoints=config.max_evaluations)
                receipt = {key: saved[key] for key in ('path', 'manifest_sha256')}
                result['checkpoints'].append(receipt)
                emit('checkpoint_saved', receipt)

            emit('site_energy_feedback_start', {'max_evaluations': 2, 'mixing': .05,
                'initial_state': 'fresh-zero', 'feedback_ndof': 132, 'full_device_dofs': 216})
            effective_config = replace(config, max_seconds=config.max_seconds-(time.monotonic()-started))
            result['effective_scba_config'] = plain(effective_config)
            solved = run_channel_scba(
                grid, vertex, dynamical, *contacts, sigma_initial=None, first_index=1,
                scale=scale, g=profile['map']['g'], config=effective_config,
                on_start=on_start, on_evaluation=on_evaluation)
            result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                failure_reason=solved.failure_reason, history=plain(solved.history),
                completed_evaluations=len(solved.history),
                passed=solved.failure_reason is None and solved.state is not None)
            if solved.state is not None:
                result['final_evaluated_state'] = plain(solved.state.record)
                result['final_currents'] = plain(solved.state.evaluation.input_currents)
            _material.verify_unchanged(pins+list(static_auth.values()))
            emit('scba_stopped', {'converged': solved.converged,
                'completed_evaluations': len(solved.history), 'stop_reason': solved.stop_reason,
                'failure_reason': solved.failure_reason})
        finally:
            _operator._static.release(static_arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    result = {'passed': False, 'converged': False, 'failure_reason': None,
              'profile_sha256': digest, 'source_commit': profile.get('source_commit'),
              'source_tree': profile.get('source_tree')}
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
    except Exception as exc:  # noqa: BLE001 - preserve actual failed map and any completed checkpoint
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('site_energy_feedback_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('site_energy_feedback_closed', {'passed': result['passed'], 'converged': result['converged'],
                                       'completed_evaluations': result.get('completed_evaluations', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
