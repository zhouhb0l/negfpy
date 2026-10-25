"""Frozen full-domain state2 acoustic-sector diagnostic with unchanged fields."""
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

from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_crossing_feedback import (
    build_crossing_feedback_vertex,
)
from negfpy.anharmonic.transverse_material_restart import (
    _arrays_verified,
    _json_verified,
)
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_acoustic_feedback_source', Path(__file__).with_name('run_transverse_crossing_feedback.py'))
_feedback = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feedback)
_static, _material = _feedback._static, _feedback._material
PARENT, FROZEN_MAP, RESOURCES, RAW_INPUTS, SUPPORT = (
    _feedback.PARENT, _feedback.FROZEN_MAP, _feedback.RESOURCES,
    _feedback.RAW_INPUTS, _feedback.SUPPORT)
DIAGNOSTIC = {
    'g': .03, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'internal_half_width': 2, 'workers': 8, 'external_tile': 16, 'bin_tile': 1,
    'max_bytes': 4*1024**3, 'max_contraction_work': 1_000_000_000_000,
    'max_dyson_evaluations': 20, 'max_seconds': 79200, 'time_reserve_seconds': 300,
    'checkpoint_max_bytes': 8*1024**3,
}
CHECKPOINT_PATH = ('/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/'
                   '20261024-crossing-feedback-001/runtime/checkpoints/iteration-000002')
CHECKPOINT_SHA256 = '4963db684dd922c43cd786f29861ef1a0bc1283bd628c6e9226cf3726135a816'
CHECKPOINT_BYTES = 43655
MODEL_IDENTITY = '29f530677668e803eeaa4f8890bb0d956cddbf6152c81fb1b51cf7c6a6f7fd5b'
_OLD_SOURCE = '992ed0b88c7ad0ae94e5431afd5d899112b34af3'
_OLD_TREE = '047a519cd46611ee0b858d4113625c170049b342'
_OLD_PROFILE = '711206629546aa6686ec502dd89f7fc370e2c114000b3977860b6083bb1c2adc'
COMPONENTS = ('retarded', 'statistical', 'spectral')
plain = _feedback.plain


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-acoustic-sector.profile.v1':
        raise ValueError('Wrong acoustic-sector profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT), ('resources', RESOURCES),
                          ('support', SUPPORT), ('diagnostic', DIAGNOSTIC),
                          ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    pin = {'name': 'native2_manifest', 'path': CHECKPOINT_PATH+'/manifest.json',
           'bytes': CHECKPOINT_BYTES, 'sha256': CHECKPOINT_SHA256}
    if profile.get('runtime_inputs') != RAW_INPUTS+[pin]:
        raise ValueError('Raw material, static parent or fixed state2 pins differ.')
    case = profile['checkpoint']
    if (case['iteration'] != 2 or case['path'] != CHECKPOINT_PATH
            or case['manifest_sha256'] != CHECKPOINT_SHA256):
        raise ValueError('Only the saved evaluated state2 is admitted.')
    binding = case['binding']
    required = {
        'source_commit': _OLD_SOURCE, 'source_tree': _OLD_TREE,
        'profile_sha256': _OLD_PROFILE, 'physics': FROZEN_MAP,
        'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
        'parent_profile_sha256': PARENT['profile_sha256'],
        'parent_result_sha256': PARENT['result_sha256'],
        'support_model': SUPPORT['model'], 'model_identity_sha256': MODEL_IDENTITY,
        'support_configuration': SUPPORT, 'feedback_dofs': list(range(216)),
        'new_hamiltonian_trajectory': True,
    }
    if binding != required:
        raise ValueError('The exact state2 Hamiltonian binding differs.')
    if any(key in profile for key in ('resume', 'restart', 'sigma_initial', 'scba')):
        raise ValueError('This fixed-input diagnostic cannot advance or restart SCBA.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return DIAGNOSTIC


def load_fixed_input(case, stack):
    """Authenticate and map evaluated R/C/D only; never load the mapped image."""
    manifest = _json_verified(stack, case['path'], 'manifest.json', case['manifest_sha256'])
    if (manifest.get('schema') != 'negfpy.transverse-channel-scba-checkpoint.v1'
            or manifest.get('representation') != 'retarded-statistical-spectral-v1'
            or manifest.get('iteration') != 2 or manifest.get('binding') != case['binding']
            or manifest.get('accepted_restart') is not False
            or manifest.get('sigma_input_evaluated') is not True
            or manifest.get('mapped_sigma_dyson_evaluated') is not False
            or manifest['record'].get('sigma_input_evaluated') is not True):
        raise ValueError('Fixed evaluated state2 manifest or field roles differ.')
    names = ['sigma_input_'+name for name in COMPONENTS]
    descriptors = {name: manifest['artifacts'][name] for name in names}
    expected = {name: ((4, 601, 216, 216), 'complex128') for name in names}
    authentication = {}
    arrays = _arrays_verified(stack, Path(case['path']), descriptors, expected,
                              DIAGNOSTIC['checkpoint_max_bytes'], authentication)
    return ChannelSpectrum(*(arrays[name] for name in names)), arrays, authentication


def rigid_translations(geometry, n_device_layers=9):
    """Normalized full-device mass-weighted translations in packed DOF order."""
    basis = np.asarray(geometry.basis_primitive_atoms, dtype=np.int64)
    primitive_masses = np.asarray(geometry._crystal.primitive.masses, dtype=float)
    masses = np.tile(primitive_masses[basis], geometry.principal_cells*n_device_layers)
    if (masses.ndim != 1 or len(masses)*3 != geometry.n_principal_layer_dofs*n_device_layers
            or not np.all(np.isfinite(masses)) or np.any(masses <= 0)):
        raise ValueError('Actual device masses or their packed ordering are invalid.')
    translations = np.zeros((3*len(masses), 3), dtype=float)
    weights = np.sqrt(masses/np.sum(masses))
    for direction in range(3):
        translations[direction::3, direction] = weights
    return translations, {
        'normalization': 'full-device mass-weighted rigid translations; no crop renormalization',
        'mass_source': 'rebuilt geometry primitive masses in packed device ordering',
        'n_atoms': len(masses), 'mass_sum_in_crystal_units': float(np.sum(masses)),
        'gram': (translations.T@translations).tolist(),
        'gamma_only_sector_basis': True, 'normal_mode_eigenbasis': False,
    }


def run(profile, result, emit, *, profile_sha256):
    from negfpy.anharmonic.transverse_acoustic_sector import (
        plan_acoustic_sector_dimensions,
        study_acoustic_sector_diagnostic,
    )

    config = validate_profile(profile)
    _material._digest(profile_sha256)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-acoustic-sector.v1', passed=False,
        converged=False, source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), diagnostic_config=plain(config),
        fixed_checkpoint=plain(profile['checkpoint']), failure_reason=None, completed_cases=0,
        scope={'scalar_outputs_only': True, 'fixed_evaluated_state2': True,
            'native_channels_independent': True, 'mapped_images_loaded': False,
            'cached_green_or_born_used': False, 'raw_ifc_modified': False,
            'field_projection_or_repair': False, 'sector_decomposition_analysis_only': True,
            'all_sector_and_interference_terms_retained': True,
            'eta_changed': False, 'g_changed': False, 'scba': False, 'full_born_map': False,
            'convergence_checks_unchanged': True, 'physical_guards_unchanged': True,
            'accepted_restart': False, 'accepted_field_update': False,
            'accepted_current_or_transport': False, 'q_or_frequency_convergence': False})
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('acoustic_inputs_authenticated', {'count': len(pins)})
        mesh, grid, reference, dynamical, contacts, static_arrays, static_auth = _static.load_static(
            profile, documents, stack)
        try:
            normalized, geometry, result['material'] = _material.rebuild_material(profile)
            translations, result['translation_basis'] = rigid_translations(geometry)
            emit('acoustic_vertex_start')
            supported = build_crossing_feedback_vertex(
                mesh, normalized, geometry, n_device_layers=9, interaction_layers=(2, 3, 4, 5, 6),
                scale=SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S),
                reference_vertex=reference, deadline_monotonic=deadline,
                **{key: value for key, value in SUPPORT.items() if key != 'model'})
            vertex = supported.vertex
            result['support_model'] = plain(supported.metadata)
            if (vertex.full_ndof != 216 or vertex.active_dofs.tolist() != list(range(216))
                    or result['support_model']['model_identity_sha256'] != MODEL_IDENTITY):
                raise ValueError('Rebuilt full216 Hamiltonian differs from saved state2.')
            emit('acoustic_vertex_complete', {'model_identity_sha256': MODEL_IDENTITY})
            plan = plan_acoustic_sector_dimensions(
                4, 601, 216, internal_half_width=2, workers=8, external_tile=16, bin_tile=1)
            result['plan'] = plain(plan)
            with ExitStack() as case_stack:
                sigma, input_arrays, input_auth = load_fixed_input(profile['checkpoint'], case_stack)
                try:
                    emit('acoustic_case_start', {'iteration': 2, 'dyson_budget': 20})
                    result['study'] = plain(study_acoustic_sector_diagnostic(
                        grid, vertex, dynamical, *contacts, sigma, translations, np.arange(48, 168),
                        g=config['g'], internal_half_width=2, workers=8, external_tile=16, bin_tile=1,
                        max_bytes=config['max_bytes'], max_contraction_work=config['max_contraction_work'],
                        max_dyson_evaluations=20, deadline_monotonic=deadline,
                        identity_rtol=config['identity_rtol'], identity_atol=config['identity_atol']))
                    _material.verify_unchanged(list(input_auth.values()))
                    result['input_authentication'] = plain(input_auth)
                    result['completed_cases'] = 1
                    emit('acoustic_case_complete', {'iteration': 2})
                finally:
                    _static.release(input_arrays)
            _material.verify_unchanged(pins+list(static_auth.values()))
            result.update(passed=True, static_authentication=plain(static_auth))
        finally:
            _static.release(static_arrays)


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
        run(profile, result, emit, profile_sha256=digest)
    except Exception as exc:  # noqa: BLE001 - preserve the actual partial diagnostic tail
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('acoustic_diagnostic_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('acoustic_diagnostic_closed', {'passed': result['passed'],
                                      'completed_cases': result.get('completed_cases', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
