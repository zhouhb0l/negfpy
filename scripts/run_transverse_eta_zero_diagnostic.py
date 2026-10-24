"""Fixed-input lead-eta sensitivity with freshly rebuilt thermal contacts.

The unchanged native interaction inputs are evaluated at three explicitly new
reservoir regularizations. This is neither SCBA continuation nor an eta
convergence claim. Old mapped fields are cross-eta references only.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import resource
import time
import traceback
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic._material_nq1_campaign import (
    HBAR_J_S,
    OMEGA0_RAD_S,
    kelvin_to_solver_temperature,
)
from negfpy.anharmonic.io.phono3py_crystal import read_phono3py_crystal_mapping
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.transverse_compact_contacts import (
    build_compact_transverse_contact,
)
from negfpy.anharmonic.transverse_eta_zero_diagnostic import (
    audit_cropped_vertex_acoustic,
    study_eta_zero_diagnostic,
)
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.transverse_material import material_transverse_harmonic
from negfpy.anharmonic.transverse_material_restart import (
    _open_verified,
    load_material_map_fields,
)
from negfpy.anharmonic.transverse_native_zero_diagnostic import (
    plan_native_zero_diagnostic_dimensions,
)
from negfpy.anharmonic.types import HarmonicReservoir, SolverScale
from negfpy.core.types import LeadBlocks

_spec = importlib.util.spec_from_file_location(
    '_eta_zero_fixed_input_contract',
    Path(__file__).with_name('run_transverse_native_zero_diagnostic.py'))
_native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_native)
plain, sha, _digest, _descriptor = (_native.plain, _native.sha,
                                  _native._digest, _native._descriptor)
expected_checkpoint_binding = _native.expected_checkpoint_binding
CASE_MANIFESTS, COMPONENTS = _native.CASE_MANIFESTS, _native.COMPONENTS
GLOBAL_GROUPS = _native.GLOBAL_GROUPS
validate_evaluated_record = _native._validate_evaluated_record
release_checkpoint, verify_unchanged = _native.release_checkpoint, _native.verify_unchanged
ETA_VALUES = (.001, .005, .01)
BASELINE_ETA = .0015625
_RAW_ROOT = '/scratch/users/astar/ihpc/zhouhb/NEGFPy/0.3.0/nq1-g2probe-provenancefast-2d61370/inputs/validation-fixture'
HARMONIC_INPUTS = [
    {'name': 'fc2', 'path': _RAW_ROOT+'/fc2.full.hdf5', 'bytes': 86059,
     'sha256': '6b26d489e6a7fffe61530e77b0c2a907eb4e4f13d5b352abb29d9c0c2c6c74f7'},
    {'name': 'crystal', 'path': _RAW_ROOT+'/phono3py_disp.yaml', 'bytes': 32796,
     'sha256': '1375e3ae19a2c4138a888af8177a3becfcaa3b127ca50755773c405c3e28a2a7'},
]
DIAGNOSTIC = dict(_native.DIAGNOSTIC, eta_lead_values=list(ETA_VALUES),
    baseline_eta_lead=BASELINE_ETA, max_cases=9, max_checkpoints=3,
    max_total_dyson_evaluations=180, max_total_contraction_work=450_000_000_000,
    contact_q_samples=257, contact_max_bytes=64*1024**2,
    contact_max_base_evaluations=100_000, max_contact_pairs=3,
    max_total_contact_base_evaluations=600_000)


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-eta-zero-diagnostic.profile.v1':
        raise ValueError('Wrong eta zero diagnostic profile schema.')
    if profile.get('diagnostic') != DIAGNOSTIC:
        raise ValueError('The exact user eta values, physical guards or bounds differ.')
    harmonic = profile.get('harmonic_inputs')
    if harmonic != HARMONIC_INPUTS:
        raise ValueError('Fresh harmonic construction requires the exact two accepted raw input pins.')
    inputs = profile.get('runtime_inputs')
    if (not isinstance(inputs, list) or len(inputs) != 7
            or len({row.get('name') for row in inputs}) != 7
            or [row for row in inputs if row.get('name') in {'fc2', 'crystal'}] != harmonic):
        raise ValueError('Exactly five JSON and two raw harmonic input pins are required.')
    old = dict(profile, schema='negfpy.transverse-native-zero-diagnostic.profile.v1',
               diagnostic=_native.DIAGNOSTIC,
               runtime_inputs=[row for row in inputs if row.get('name') not in {'fc2', 'crystal'}])
    _native.validate_profile(old)
    return plain(profile['diagnostic'])


def authenticate_inputs(profile):
    """Authenticate all seven files; JSON parsing never touches the binary FC2."""
    json_rows = [row for row in profile['runtime_inputs'] if row['name'] not in {'fc2', 'crystal'}]
    records = _native.authenticate_json_inputs(json_rows)
    with ExitStack() as stack:
        for row in profile['harmonic_inputs']:
            path = Path(row['path'])
            before = _descriptor(path)
            _open_verified(stack, path, row['sha256'], row['bytes'])
            after = _descriptor(path)
            if before != after:
                raise ValueError('A raw harmonic input changed during authentication.')
            records.append(dict(row, stat_before=before, stat_after=after))
    return records


def authenticate_checkpoint(case):
    checkpoint, binding = _native.authenticate_checkpoint(case)
    binding['mapped_role'] = ('saved UNEVALUATED mapped total at baseline eta=.0015625; '
                              'cross-eta reference only, never Green input or new-eta far-bin remainder')
    return checkpoint, binding


def rebuild_harmonic(fields, profile):
    """Reconstruct the accepted harmonic model from freshly authenticated raw FC2."""
    paths = {row['name']: Path(row['path']) for row in profile['harmonic_inputs']}
    cfg = profile['map']
    crystal = read_phono3py_crystal_mapping(paths['crystal'], purpose='fc2')
    fc2 = read_phono3py_fc2(paths['fc2'], crystal.supercell.masses,
                          p2s_map=crystal.p2s, s2p_map=crystal.s2p, compact=False)
    geometry = build_layered_device_mapping(crystal, fc2,
        transport_supercell_matrix=np.asarray(cfg['transport_matrix'], dtype=np.int64))
    if (geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1
            or geometry.layered_geometry_sha256 != cfg['expected_geometry_sha256']):
        raise ValueError('Reconstructed material geometry differs from the accepted harmonic model.')
    scale = SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S)
    device, lead = material_transverse_harmonic(geometry, n_device_layers=9, scale=scale)
    comparison = []
    for q, point in enumerate(_kpoints(fields.mesh)):
        matrix = assemble_device_dynamical_matrix(device.device(tuple(point)))
        saved = fields.dynamical[q]
        if matrix.shape != (216, 216) or saved.shape != matrix.shape:
            raise ValueError('Reconstructed and saved full-device harmonic dimensions differ.')
        error = float(np.max(np.abs(matrix-saved)))
        reference = max(float(np.max(np.abs(matrix))), float(np.max(np.abs(saved))))
        allowance = 1e-11+1e-8*reference
        passed = bool(np.isfinite(matrix).all() and error <= allowance)
        comparison.append({'q': q, 'error': error, 'scale': reference,
                           'allowance': allowance, 'passed': passed})
        if not passed:
            raise ValueError('Fresh reconstructed harmonic device does not match the saved operator.')
    # Ordering matches material_transverse_harmonic: layer, principal-cell, basis,
    # Cartesian component. A common mass-unit factor cancels in normalization.
    principal_masses = np.tile(np.asarray(crystal.primitive.masses)[geometry.basis_primitive_atoms],
                               geometry.principal_cells)
    masses = np.tile(principal_masses, cfg['n_device_layers'])
    if masses.shape != (72,) or not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError('Actual material masses do not define the 216-DOF rigid translations.')
    translations = np.zeros((216, 3), dtype=np.float64)
    for axis in range(3):
        translations[axis::3, axis] = np.sqrt(masses)
        translations[:, axis] /= np.linalg.norm(translations[:, axis])
    record = {'fresh_from_pinned_fc2': True,
        'geometry_sha256': geometry.layered_geometry_sha256,
        'fc2_content_sha256': geometry.fc2_content_sha256,
        'primitive_sha256': geometry.primitive_crystal_content_sha256,
        'saved_dynamical_comparison': {'passed': True, 'rtol': 1e-8, 'atol': 1e-11, 'rows': comparison},
        'full_device_dofs': 216, 'n_atoms': 72,
        'mass_minimum_source_units': float(masses.min()),
        'mass_maximum_source_units': float(masses.max()),
        'translation_normalization': 'unit full-device massweighted rigid vectors; active crop not renormalized',
        'translation_norms': [float(np.linalg.norm(translations[:, axis])) for axis in range(3)]}
    return lead, translations, record


def _array_digest(array):
    return hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast('B')).hexdigest()


def build_contacts(fields, lead, eta, config, record, deadline, clock):
    """Fresh retarded and Keldysh contact fields with matched analytic zero limits."""
    contacts = []
    for side, kelvin in (('left', 305.), ('right', 295.)):
        if clock() >= deadline:
            raise TimeoutError('Diagnostic time admission failed before fresh contact construction.')
        def reservoir(q, kelvin=kelvin):
            return HarmonicReservoir(LeadBlocks(*lead.blocks(q)),
                temperature=kelvin_to_solver_temperature(kelvin))
        contact = build_compact_transverse_contact(fields.mesh, fields.grid, reservoir, 216, eta,
            side=side, q_samples=config['contact_q_samples'], surface_gf_method='sancho_rubio',
            max_bytes=config['contact_max_bytes'],
            max_base_evaluations=config['contact_max_base_evaluations'])
        if contact.contact.eta_lead != eta:
            raise ValueError('Fresh contact eta differs from the explicit experimental value.')
        record[side] = {
            'freshly_rebuilt': True, 'eta_lead': eta, 'eta_device': 0,
            'temperature_kelvin': kelvin,
            'temperature_solver': contact.contact.temperature,
            'checks': dict(contact.contact.checks), 'q_reports': list(contact.contact.q_reports),
            'device_indices': list(contact.device_indices),
            'zero_limit_norm_by_q': [float(np.linalg.norm(v)) for v in contact.zero_limits],
            'zero_limits_sha256': _array_digest(contact.zero_limits),
            'spectrum_sha256': {name: _array_digest(getattr(contact.spectrum, name))
                                for name in ('retarded', 'lesser', 'greater')},
            'zero_limit_convention': 'lesser=greater=-i*T*Gamma_prime(0) at the same finite eta',
        }
        contacts.append(contact)
        if clock() >= deadline:
            raise TimeoutError('Diagnostic time admission failed after fresh contact construction.')
    return tuple(contacts)


def run(profile, result, emit, *, profile_sha256, clock=time.monotonic):
    config = validate_profile(profile)
    _digest(profile_sha256)
    deadline = clock()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-eta-zero-diagnostic.v1', passed=False,
        converged=False, source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), diagnostic_config=config,
        cases=[], checkpoints=[], contacts=[], started_cases=0, completed_cases=0,
        admitted_dyson_evaluation_bound=0, admitted_contraction_work_bound=0,
        admitted_contact_base_evaluation_bound=0, stop_reason=None, failure_reason=None,
        scope={'scba': False, 'full_map': False, 'full_born_map': False,
            'scalar_output_only': True, 'field_output': False, 'accepted_field_update': False,
            'accepted_restart': False, 'current_acceptance': False, 'converged_transport': False,
            'frequency_or_q_convergence': False, 'fixed_evaluated_inputs': True,
            'checkpoint_mixing_or_update': False, 'native_spectral_independent': True,
            'mapped_sigma_used_as_green_input': False,
            'cached_parent_green_or_born_used_computationally': False,
            'old_contacts_used_computationally': False, 'fresh_matched_thermal_contacts': True,
            'local_ratios_are_diagnostic_only': True, 'physical_guards_unchanged': True,
            'convergence_checks_unchanged': True, 'eta_device': 0,
            'baseline_mapped_total_is_cross_eta_reference_only': True,
            'cross_eta_reference_gap_is_not_far_bin_remainder': True})
    emit('diagnostic_parent_authentication_start')
    result['runtime_input_authentication'] = authenticate_inputs(profile)
    parent = profile['parent']
    fields = load_material_map_fields(parent['root'],
        expected_result_sha256=parent['result_sha256'],
        expected_profile_sha256=parent['profile_sha256'],
        expected_source_commit=parent['source_commit'], expected_source_tree=parent['source_tree'],
        expected_settings=profile['map'], max_total_bytes=profile['parent_max_bytes'])
    result['parent_binding'] = plain(fields.binding)
    if result['parent_binding'].get('artifacts_freshly_authenticated') != 19:
        raise ValueError('All nineteen static parent artifacts must authenticate freshly.')
    emit('diagnostic_parent_authenticated', {'parent_binding': result['parent_binding'],
                                            'runtime_inputs': result['runtime_input_authentication']})
    emit('harmonic_inputs_authenticated', {'inputs': [row for row in result['runtime_input_authentication']
                                                    if row['name'] in {'fc2', 'crystal'}]})
    plan = plain(plan_native_zero_diagnostic_dimensions(fields.mesh.n_qpoints, fields.grid.n_points,
        fields.dynamical.shape[-1], fields.vertex.ndof, half_width=config['half_width'],
        external_tile=config['external_tile'], workers=config['workers'], bin_tile=config['bin_tile']))
    if (plan['dyson_evaluations'] != 20 or plan['dyson_evaluations'] > config['max_dyson_evaluations']
            or plan['contraction_work'] > config['max_contraction_work']
            or plan['peak_owned_bytes'] > config['max_bytes']):
        raise ValueError('The bounded eta diagnostic plan exceeds its frozen per-case limits.')
    result['case_plan'] = plan
    if clock() >= deadline:
        result['stop_reason'] = 'time_admission_before_harmonic_reconstruction'
        return
    lead, translations, result['harmonic_reconstruction'] = rebuild_harmonic(fields, profile)
    emit('harmonic_reconstructed', result['harmonic_reconstruction'])
    result['cropped_vertex_acoustic'] = plain(audit_cropped_vertex_acoustic(
        fields.vertex, translations, max_bytes=config['max_bytes'], deadline_monotonic=deadline))
    emit('acoustic_audit_complete', result['cropped_vertex_acoustic'])
    new_contacts = {}
    for eta in ETA_VALUES:
        if clock() >= deadline:
            result.update(stop_reason='time_admission_before_contacts', unbuilt_contact_eta=eta)
            return
        work = result['admitted_contact_base_evaluation_bound']+2*config['contact_max_base_evaluations']
        if work > config['max_total_contact_base_evaluations']:
            raise ValueError('Cumulative contact construction bound exceeded.')
        result['admitted_contact_base_evaluation_bound'] = work
        contact_record = {'eta_lead': eta, 'left': None, 'right': None}
        result['contacts'].append(contact_record)
        emit('eta_contacts_start', {'eta_lead': eta})
        new_contacts[eta] = build_contacts(fields, lead, eta, config, contact_record, deadline, clock)
        emit('eta_contacts_complete', contact_record)
    for case in profile['checkpoints']:
        index = case['iteration']
        if clock() >= deadline:
            result.update(stop_reason='time_admission_before_checkpoint', unstarted_case_index=index)
            break
        checkpoint = None
        binding = None
        emit('checkpoint_authentication_start', {'index': index, 'path': case['path'],
                                                'manifest_sha256': case['manifest_sha256']})
        try:
            checkpoint, binding = authenticate_checkpoint(case)
            result['checkpoints'].append(binding)
            emit('checkpoint_authenticated', {'index': index, **binding})
            for eta in ETA_VALUES:
                if clock() >= deadline:
                    result.update(stop_reason='time_admission_before_case',
                                  unstarted_case_index=index, unstarted_case_eta=eta)
                    break
                row = {'index': index, 'eta_lead': eta, 'status': 'running', 'completed': False,
                    'checkpoint_manifest_sha256': case['manifest_sha256'],
                    'evaluated_record': plain(checkpoint.record), 'plan': plan}
                result['cases'].append(row)
                dyson = result['admitted_dyson_evaluation_bound']+plan['dyson_evaluations']
                work = result['admitted_contraction_work_bound']+plan['contraction_work']
                if dyson > config['max_total_dyson_evaluations'] or work > config['max_total_contraction_work']:
                    raise ValueError('Cumulative fixed-input eta diagnostic bound exceeded.')
                result['admitted_dyson_evaluation_bound'] = dyson
                result['admitted_contraction_work_bound'] = work
                result['started_cases'] += 1
                emit('diagnostic_case_start', {'index': index, 'eta_lead': eta, 'plan': plan,
                    'input_sigma_sha256': checkpoint.record['sigma_input_sha256'],
                    'saved_mapped_sigma_sha256': checkpoint.record['mapped_sigma_sha256'],
                    'input_role': binding['input_role'], 'mapped_role': binding['mapped_role']})
                call_keys = ('g', 'max_bytes', 'max_contraction_work', 'max_dyson_evaluations',
                    'max_refinements', 'half_width', 'external_q', 'selected_elements',
                    'workers', 'external_tile', 'bin_tile')
                diagnostic = plain(study_eta_zero_diagnostic(fields.grid, fields.vertex,
                    fields.dynamical, *new_contacts[eta], checkpoint.sigma_input, checkpoint.mapped_sigma,
                    **{key: config[key] for key in call_keys}, eta_lead=eta,
                    baseline_eta=BASELINE_ETA, deadline_monotonic=deadline))
                if (diagnostic.get('plan') != plan
                        or diagnostic.get('dyson_evaluations') != plan['dyson_evaluations']):
                    raise ValueError('Completed eta diagnostic differs from its admitted plan.')
                row.update(status='complete', completed=True, diagnostic=diagnostic)
                verify_unchanged(binding)
                result['completed_cases'] += 1
                emit('diagnostic_case_complete', row)
        except Exception as exc:
            if result['cases'] and not result['cases'][-1]['completed']:
                result['cases'][-1].update(status='failed',
                    failure={'type': type(exc).__name__, 'message': str(exc)})
            result.update(failure_reason=type(exc).__name__+': '+str(exc), stop_reason='case_failed')
            raise
        finally:
            if checkpoint is not None:
                try:
                    verify_unchanged(binding)
                finally:
                    release_checkpoint(checkpoint)
        if result['stop_reason'] is not None:
            break
    verify_unchanged(result['parent_binding'])
    for row in result['runtime_input_authentication']:
        row['stat_after_diagnostic'] = _descriptor(row['path'])
        if row['stat_after_diagnostic'] != row['stat_before']:
            raise ValueError('A pinned input changed during the eta diagnostic.')
    if result['completed_cases'] == 9:
        result.update(passed=True, stop_reason='all_fixed_eta_cases_completed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'schema': 'negfpy.transverse-eta-zero-diagnostic.v1', 'passed': False, 'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
               'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)
    try:
        run(profile, result, emit, profile_sha256=sha(args.profile))
    except Exception as exc:  # noqa: BLE001 - preserve actual failed/partial experiment evidence
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
