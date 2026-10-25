"""Inspect unmodified soft resolvent directions and sources at fixed state2."""
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

from negfpy.anharmonic.transverse_material_restart import _arrays_verified

_spec = importlib.util.spec_from_file_location(
    '_soft_acoustic_source', Path(__file__).with_name('run_transverse_acoustic_sector.py'))
_acoustic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acoustic)
_static, _material = _acoustic._static, _acoustic._material
PARENT, FROZEN_MAP, RESOURCES = _acoustic.PARENT, _acoustic.FROZEN_MAP, _acoustic.RESOURCES
# FC3 is neither read nor rebuilt in this operator/source-only diagnostic.
RAW_INPUTS = _acoustic.RAW_INPUTS[1:5]
DIAGNOSTIC = {
    'n_layers': 9, 'n_soft': 12, 'include_zero_reference': True,
    'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'max_bytes': 1024**3, 'max_dyson_evaluations': 12, 'max_svd_evaluations': 12,
    'max_seconds': 79200, 'time_reserve_seconds': 300, 'checkpoint_max_bytes': 8*1024**3,
}
CHECKPOINT_PATH, CHECKPOINT_SHA256, CHECKPOINT_BYTES = (
    _acoustic.CHECKPOINT_PATH, _acoustic.CHECKPOINT_SHA256, _acoustic.CHECKPOINT_BYTES)
plain = _acoustic.plain
load_fixed_input = _acoustic.load_fixed_input
rigid_translations = _acoustic.rigid_translations


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-soft-source.profile.v1':
        raise ValueError('Wrong soft-source profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT), ('resources', RESOURCES),
                          ('diagnostic', DIAGNOSTIC), ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    pin = {'name': 'native2_manifest', 'path': CHECKPOINT_PATH+'/manifest.json',
           'bytes': CHECKPOINT_BYTES, 'sha256': CHECKPOINT_SHA256}
    if profile.get('runtime_inputs') != RAW_INPUTS+[pin]:
        raise ValueError('Only harmonic inputs, parent metadata and fixed state2 pins are admitted.')
    case = profile['checkpoint']
    if (case['iteration'] != 2 or case['path'] != CHECKPOINT_PATH
            or case['manifest_sha256'] != CHECKPOINT_SHA256):
        raise ValueError('Only the saved evaluated state2 is admitted.')
    required = {
        'source_commit': _acoustic._OLD_SOURCE, 'source_tree': _acoustic._OLD_TREE,
        'profile_sha256': _acoustic._OLD_PROFILE, 'physics': FROZEN_MAP,
        'representation': 'retarded-statistical-spectral-v1', 'initial_state': 'fresh-zero',
        'parent_profile_sha256': PARENT['profile_sha256'],
        'parent_result_sha256': PARENT['result_sha256'],
        'support_model': _acoustic.SUPPORT['model'],
        'model_identity_sha256': _acoustic.MODEL_IDENTITY,
        'support_configuration': _acoustic.SUPPORT, 'feedback_dofs': list(range(216)),
        'new_hamiltonian_trajectory': True,
    }
    if case['binding'] != required:
        raise ValueError('The exact evaluated state2 binding differs.')
    if any(key in profile for key in ('resume', 'restart', 'sigma_initial', 'scba')):
        raise ValueError('This fixed-input diagnostic cannot advance or restart SCBA.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return DIAGNOSTIC


def load_static_operator(profile, documents, stack):
    """Authenticate only dynamical and ten contact arrays; no cubic or cached fields."""
    _material.validate_parent(profile, documents)
    result = documents['parent_result']
    names = ['dynamical']
    for side in ('left', 'right'):
        names.extend(side+'_'+label for label in
                     ('device_indices', 'zero_limits', 'retarded', 'lesser', 'greater'))
    descriptors = {name: result['artifacts'][name] for name in names}
    expected = {name: (tuple(row['shape']), row['dtype']) for name, row in descriptors.items()}
    expected['dynamical'] = ((4, 216, 216), 'complex128')
    authentication = {}
    arrays = _arrays_verified(stack, Path(PARENT['root'])/'runtime', descriptors,
                              expected, 2*1024**3, authentication)
    mesh = _static.UniformBZMesh(tuple(profile['map']['mesh']))
    grid = _static.SignedFrequencyGrid(profile['map']['grid_step'], profile['map']['n_positive'])
    contacts = []
    try:
        for index, side in enumerate(('left', 'right')):
            report = result['contacts'][side]
            indices = arrays[side+'_device_indices'].tolist()
            temperature = profile['map']['temperatures_kelvin'][index]
            if indices != report['device_indices'] or report['temperature_kelvin'] != temperature:
                raise ValueError('Saved contact embedding or temperature differs.')
            spectrum = _static.TransverseSpectrum(
                *(arrays[side+'_'+component] for component in ('retarded', 'lesser', 'greater')))
            contact = _static.TransverseContact(
                mesh, grid, spectrum, arrays[side+'_zero_limits'], tuple(report['q_reports']),
                report['checks'], side, profile['map']['eta_lead'],
                _static.kelvin_to_solver_temperature(temperature))
            contacts.append(_static.CompactTransverseContact(contact, 216, tuple(indices)))
    except BaseException:
        _static.release(arrays)
        raise
    return mesh, grid, arrays['dynamical'], contacts, arrays, authentication


def rebuild_harmonic_geometry(profile):
    """Read FC2/crystal for actual ordered masses; never read FC3 or build a vertex."""
    paths = {row['name']: Path(row['path']) for row in profile['runtime_inputs']}
    crystal = _material.read_phono3py_crystal_mapping(paths['crystal'], purpose='fc2')
    fc2 = _material.read_phono3py_fc2(paths['fc2'], crystal.supercell.masses,
        p2s_map=crystal.p2s, s2p_map=crystal.s2p, compact=False)
    geometry = _material.build_layered_device_mapping(crystal, fc2,
        transport_supercell_matrix=np.asarray(profile['map']['transport_matrix'], dtype=np.int64))
    identities = _material.SOURCE_IDENTITIES
    if (geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1
            or geometry.layered_geometry_sha256 != profile['map']['expected_geometry_sha256']
            or geometry.fc2_content_sha256 != identities['source_fc2_sha256']
            or geometry.primitive_crystal_content_sha256 != identities['primitive_crystal_sha256']):
        raise ValueError('Fresh harmonic geometry or actual masses differ from the accepted mapping.')
    return geometry, {
        'fresh_raw_fc2': True, 'fresh_crystal': True, 'fc3_read': False,
        'vertex_built': False, 'full_device_dofs': 216,
        'source_fc2_sha256': geometry.fc2_content_sha256,
        'primitive_crystal_sha256': geometry.primitive_crystal_content_sha256,
        'layered_geometry_sha256': geometry.layered_geometry_sha256,
        'dynamical_source': 'unchanged authenticated harmonic parent array',
    }


def run(profile, result, emit, *, profile_sha256):
    from negfpy.anharmonic.transverse_soft_source import (
        plan_soft_source_dimensions,
        study_soft_source_diagnostic,
    )

    config = validate_profile(profile)
    _material._digest(profile_sha256)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-soft-source.v1', passed=False, converged=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), diagnostic_config=plain(config),
        fixed_checkpoint=plain(profile['checkpoint']), completed_studies=0, failure_reason=None,
        scope={'scalar_outputs_only': True, 'fixed_evaluated_state2': True,
            'exact_zero_sigma_reference': True, 'native_channels_independent': True,
            'mapped_images_loaded': False, 'cached_green_or_born_used': False,
            'fc3_read_or_vertex_built': False, 'born_evaluations': 0, 'scba': False,
            'operator_or_source_modified': False, 'field_projection_or_repair': False,
            'resolvent_singular_vectors_are_not_normal_modes': True,
            'eta_changed': False, 'g_changed': False, 'physical_guards_unchanged': True,
            'convergence_checks_unchanged': True, 'accepted_restart': False,
            'accepted_field_update': False, 'accepted_current_or_transport': False,
            'q_or_frequency_convergence': False})
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('soft_source_inputs_authenticated', {'count': len(pins)})
        mesh, grid, dynamical, contacts, static_arrays, static_auth = load_static_operator(
            profile, documents, stack)
        try:
            geometry, result['harmonic_geometry'] = rebuild_harmonic_geometry(profile)
            translations, result['translation_basis'] = rigid_translations(geometry)
            emit('soft_source_geometry_complete', {'full_device_dofs': 216})
            result['plan'] = plain(plan_soft_source_dimensions(
                4, 601, 216, n_layers=9, n_soft=12, include_zero_reference=True))
            with ExitStack() as case_stack:
                sigma, input_arrays, input_auth = load_fixed_input(profile['checkpoint'], case_stack)
                try:
                    emit('soft_source_study_start', {'roles': ['exact-zero', 'fixed-state2'],
                                                     'dyson_budget': 12, 'svd_budget': 12})
                    result['study'] = plain(study_soft_source_diagnostic(
                        grid, mesh, dynamical, *contacts, sigma, translations,
                        n_layers=9, n_soft=12, include_zero_reference=True,
                        max_bytes=config['max_bytes'], max_dyson_evaluations=12, max_svd_evaluations=12,
                        deadline_monotonic=deadline, identity_rtol=config['identity_rtol'],
                        identity_atol=config['identity_atol']))
                    _material.verify_unchanged(list(input_auth.values()))
                    result['input_authentication'] = plain(input_auth)
                    result['completed_studies'] = 1
                    emit('soft_source_study_complete')
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
    except Exception as exc:  # noqa: BLE001 - preserve the actual bounded diagnostic failure
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('soft_source_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('soft_source_closed', {'passed': result['passed'],
                                'completed_studies': result.get('completed_studies', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
