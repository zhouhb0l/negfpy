"""Study one authenticated Gamma/zero slice and a distinct direct-PV trial."""

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

from negfpy.anharmonic.transverse_material_restart import (
    load_material_checkpoint,
    load_material_map_fields,
)
from negfpy.anharmonic.transverse_material_scba import mix_packed_spectra
from negfpy.anharmonic.transverse_zero_reference import (
    study_zero_frequency,
    zero_reference_workspace_bytes,
)

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
    if diagnostic.get('kind') != 'zero-frequency-reference':
        raise ValueError('This driver requires the zero-frequency-reference diagnostic.')
    for key, expected in (('q_index', 0), ('frequency_index', 300),
                          ('max_dyson_cases', 2), ('max_refinements', 4)):
        if type(diagnostic.get(key)) is not int or diagnostic[key] != expected:
            raise ValueError('The reference is bounded to Gamma/zero and two Dyson cases.')
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
               'coupling_already_applied_to_input': True,
               'production_solver_changed': False, 'trial_retarded_slice_changed': True},
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
    q, w = diagnostic['q_index'], diagnostic['frequency_index']
    if (fields.mesh.address_from_index(q).components != (0, 0, 0)
            or fields.grid.zero_index != w or fields.grid.values[w] != 0):
        raise ValueError('Authenticated fields do not identify the requested Gamma/zero slice.')
    selected = {'q_index': q, 'q_address': [0, 0, 0], 'frequency_index': w, 'omega': 0.0}
    result['selected_point'] = selected
    candidate_bytes = sum(getattr(saved.sigma_input, name).nbytes for name in _COMPONENTS)
    workspace = zero_reference_workspace_bytes(
        fields.dynamical.shape[-1], len(fields.vertex.active_dofs), fields.grid.n_points,
    )
    if candidate_bytes + workspace > diagnostic['max_bytes']:
        raise ValueError('Candidate plus reference workspace exceeds max_bytes before mixing.')
    result['memory_plan'] = {
        'candidate_owned_bytes': candidate_bytes, 'reference_owned_bound_bytes': workspace,
        'combined_owned_bound_bytes': candidate_bytes+workspace,
        'parent_checkpoint_fields_python_and_library_scratch_excluded': True,
    }
    candidate = mix_packed_spectra(saved.sigma_input, saved.mapped_sigma, restart['initial_mixing'])
    candidate_hashes = spectrum_hashes(candidate)
    result['candidate_sha256'] = candidate_hashes
    result['candidate_hashes_match'] = candidate_hashes == diagnostic['expected_candidate_sha256']
    if not result['candidate_hashes_match']:
        raise ValueError('Mixed candidate hashes differ from the captured failed attempt.')
    emit('reference_candidate_ready', {
        'mixing': restart['initial_mixing'], 'sigma_sha256': candidate_hashes,
        'trajectory_index': diagnostic['trajectory_candidate'], 'dyson_evaluated': False,
    })
    emit('zero_reference_start', {
        'trajectory_index': diagnostic['trajectory_candidate'], 'selected_point': selected,
        'maximum_dyson_cases': diagnostic['max_dyson_cases'],
        'maximum_refinements_per_case': diagnostic['max_refinements'],
        'second_case_changes_only_interaction_retarded_zero_slice': True,
    })

    def spectrum_q(spectrum):
        return {name: getattr(spectrum, name)[q] for name in _COMPONENTS}

    def contact_point(contact):
        return {'support_dofs': contact.device_indices,
                **{name: getattr(contact.spectrum, name)[q, w] for name in _COMPONENTS}}

    captured = study_zero_frequency(
        fields.grid, fields.dynamical[q], fields.vertex.active_dofs,
        contact_point(fields.left), contact_point(fields.right), spectrum_q(candidate),
        sigma_input_spectrum_q=spectrum_q(saved.sigma_input),
        mapped_sigma_spectrum_q=spectrum_q(saved.mapped_sigma),
        mixing=restart['initial_mixing'], max_bytes=diagnostic['max_bytes']-candidate_bytes,
        max_refinements=diagnostic['max_refinements'],
    )
    result['reference'] = plain(captured)
    reference = result['reference']
    resources = reference.get('resources', {})
    trial = reference.get('direct_pv_retarded_trial', {})
    if (reference.get('schema') != 'negfpy.transverse-zero-frequency-reference.v1'
            or reference.get('frequency_index') != w or reference.get('omega_solver') != 0
            or resources.get('zero_dyson_cases') != 2
            or resources.get('max_zero_dyson_cases') != 2
            or resources.get('inverse_refinements_per_case') != diagnostic['max_refinements']
            or resources.get('frequency_stack_allocated') is not False
            or resources.get('owned_numeric_workspace_bound_bytes') != workspace
            or reference.get('nonfinite_fields') != []
            or any(reference.get(key) is not False for key in (
                'identity_forced', 'symmetry_projected', 'born_evaluated',
                'scba_state_completed', 'field_outputs', 'accepted_restart'))
            or trial.get('changed_candidate') is not True or trial.get('accepted') is not False
            or 'original_candidate' not in reference or 'principal_value' not in reference):
        raise ValueError('The requested bounded zero-frequency reference was not returned.')
    # Serialize now to reject arrays or nonfinite scalar values before marking success.
    json.dumps(reference, allow_nan=False)
    result.update(passed=True, completed=True, selected_points=1, dyson_cases_completed=2)
    emit('zero_reference_complete', {
        'passed': True, 'completed': True, 'candidate_hashes_match': True,
        'selected_points': 1, 'dyson_cases_completed': 2, 'born_evaluated': False,
        'trial_accepted': False,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'passed': False, 'completed': False, 'converged': False,
              'schema': 'negfpy.transverse-zero-reference.v1'}

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
        emit('zero_reference_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('zero_reference_closed', {
        'passed': result['passed'], 'completed': result['completed'],
        'dyson_cases_completed': result.get('dyson_cases_completed'),
        'elapsed_seconds': result['elapsed_seconds'],
    })
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
