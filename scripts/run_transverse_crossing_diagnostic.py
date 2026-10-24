"""One bounded matched Born comparison retaining actual crossing IFC support."""
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

from negfpy.anharmonic._material_nq1_campaign import kelvin_to_solver_temperature
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_crossing_diagnostic import (
    plan_crossing_diagnostic_dimensions,
    study_crossing_support_diagnostic,
)
from negfpy.anharmonic.transverse_crossing_support import build_crossing_support_vertex
from negfpy.anharmonic.transverse_material_restart import (
    _arrays_verified,
    _json_verified,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseSpectrum
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_crossing_material_source', Path(__file__).with_name('run_transverse_boundary_cancellation.py'))
_material = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_material)
PARENT, FROZEN_MAP, RESOURCES = _material.PARENT, _material.FROZEN_MAP, _material.RESOURCES
RAW_INPUTS = _material.RUNTIME_INPUTS
CASE_MANIFESTS = {
    27: '6c007c1b1febe175b073eb48a25851ee4f06120bb9eeecc993350d43c6c08c7b',
    28: '83c3d29146e7decdfd89c259b0f12608d1ef1559c8d50203d7efb23b4e95683f',
    29: 'dd96eedb7f24c880da6482743f0ceeb60f0859d72ed323c0641d90eb9a9a9bc6',
}
COMPONENTS = ('retarded', 'statistical', 'spectral')
DIAGNOSTIC = {
    'g': .03, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'internal_half_width': 2, 'external_offsets': [-1, 0, 1],
    'case_order': [28, 27, 29], 'max_cases': 3, 'max_bytes': 4*1024**3,
    'max_contraction_work': 1_000_000_000_000,
    'max_total_contraction_work': 3_000_000_000_000,
    'max_dyson_evaluations': 28, 'max_total_dyson_evaluations': 84,
    'workers': 8, 'external_tile': 16, 'bin_tile': 1,
    'max_seconds': 79200, 'time_reserve_seconds': 300,
    'vertex_max_bytes': 2*1024**3, 'mapping_workspace_bytes': 4*1024**3,
    'max_geometry_evaluations': 2_000_000, 'max_phase_evaluations': 1_000_000_000,
}
plain = _material.plain


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-crossing-diagnostic.profile.v1':
        raise ValueError('Wrong crossing-support profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT),
                          ('resources', RESOURCES), ('diagnostic', DIAGNOSTIC)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    if profile['runtime_inputs'][:5] != RAW_INPUTS or len(profile['runtime_inputs']) != 8:
        raise ValueError('Raw IFC and parent pins differ.')
    cases = profile['checkpoints']
    if [c['iteration'] for c in cases] != [28, 27, 29]:
        raise ValueError('Fixed cases must be ordered28,27,29.')
    for case, pin in zip(cases, profile['runtime_inputs'][5:], strict=True):
        index = case['iteration']
        if (case['manifest_sha256'] != CASE_MANIFESTS[index]
                or pin['path'] != str(Path(case['path'])/'manifest.json')
                or pin['sha256'] != CASE_MANIFESTS[index]
                or case['binding']['physics'] != FROZEN_MAP
                or case['binding']['representation'] != 'retarded-statistical-spectral-v1'):
            raise ValueError('A fixed native input identity differs.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return DIAGNOSTIC


def release(arrays):
    for value in arrays.values():
        mapping = getattr(value, '_mmap', None)
        if mapping is not None and not mapping.closed:
            mapping.close()


def load_static(profile, documents, stack):
    """Load13 required arrays, excluding six cached Green/Born fields."""
    _material.validate_parent(profile, documents)
    result = documents['parent_result']
    names = ['dynamical', 'vertex_before_g', 'active_dofs']
    for side in ('left', 'right'):
        names.extend(side+'_'+label for label in
                     ('device_indices', 'zero_limits', 'retarded', 'lesser', 'greater'))
    descriptors = {name: result['artifacts'][name] for name in names}
    expected = {name: (tuple(row['shape']), row['dtype']) for name, row in descriptors.items()}
    authentication = {}
    arrays = _arrays_verified(stack, Path(PARENT['root'])/'runtime', descriptors,
                              expected, 2*1024**3, authentication)
    mesh = UniformBZMesh(tuple(profile['map']['mesh']))
    grid = SignedFrequencyGrid(profile['map']['grid_step'], profile['map']['n_positive'])
    if arrays['active_dofs'].tolist() != list(range(48, 168)):
        raise ValueError('The external crop differs.')
    vertex = PackedTransverseCubicVertex(mesh, arrays['vertex_before_g'],
                                       arrays['active_dofs'], 216, max_bytes=536870912)
    contacts = []
    for index, side in enumerate(('left', 'right')):
        report = result['contacts'][side]
        indices = arrays[side+'_device_indices'].tolist()
        if (indices != report['device_indices']
                or report['temperature_kelvin'] != profile['map']['temperatures_kelvin'][index]):
            raise ValueError('Saved contact embedding or temperature differs.')
        spectrum = TransverseSpectrum(*(arrays[side+'_'+c] for c in ('retarded', 'lesser', 'greater')))
        contact = TransverseContact(mesh, grid, spectrum, arrays[side+'_zero_limits'],
                                    tuple(report['q_reports']), report['checks'], side,
                                    profile['map']['eta_lead'],
                                    kelvin_to_solver_temperature(report['temperature_kelvin']))
        contacts.append(CompactTransverseContact(contact, 216, tuple(indices)))
    return mesh, grid, vertex, arrays['dynamical'], contacts, arrays, authentication


def load_fixed_input(case, stack):
    """Authenticate only evaluated R/C/D; mapped images are never loaded."""
    manifest = _json_verified(stack, case['path'], 'manifest.json', case['manifest_sha256'])
    if (manifest['schema'] != 'negfpy.transverse-channel-scba-checkpoint.v1'
            or manifest['representation'] != 'retarded-statistical-spectral-v1'
            or manifest['iteration'] != case['iteration'] or manifest['binding'] != case['binding']
            or manifest.get('accepted_restart') is not False
            or manifest['record']['sigma_input_evaluated'] is not True):
        raise ValueError('A fixed evaluated input manifest differs.')
    names = ['sigma_input_'+c for c in COMPONENTS]
    descriptors = {name: manifest['artifacts'][name] for name in names}
    expected = {name: ((4, 601, 120, 120), 'complex128') for name in names}
    authentication = {}
    arrays = _arrays_verified(stack, Path(case['path']), descriptors, expected,
                              2*1024**3, authentication)
    return ChannelSpectrum(*(arrays[name] for name in names)), arrays, authentication


def run(profile, result, emit, *, profile_sha256):
    config = validate_profile(profile)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-crossing-diagnostic.v1', passed=False,
                  source_commit=profile['source_commit'], source_tree=profile['source_tree'],
                  profile_sha256=profile_sha256, settings=plain(profile['map']), cases=[],
                  completed_cases=0, failure_reason=None,
                  scope={'actual_ifc_crossing_support': True, 'same_green_comparison': True,
                         'raw_ifc_modified': False, 'ifc_projection': False,
                         'eta_changed': False, 'g_changed': False,
                         'cached_green_or_born_used': False, 'mapped_inputs_loaded': False,
                         'scba': False, 'full_born_map': False, 'converged': False,
                         'accepted_restart': False, 'accepted_field_update': False,
                         'scalar_outputs_only': True, 'convergence_checks_unchanged': True})
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('crossing_inputs_authenticated', {'count': len(pins)})
        mesh, grid, vertex, dynamical, contacts, static_arrays, static_auth = load_static(
            profile, documents, stack)
        try:
            normalized, geometry, result['material'] = _material.rebuild_material(profile)
            emit('crossing_vertex_start')
            crossing = build_crossing_support_vertex(
                mesh, normalized, geometry, n_device_layers=9, interaction_layers=(2, 3, 4, 5, 6),
                scale=SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S),
                reference_vertex=vertex, max_bytes=config['vertex_max_bytes'],
                max_workspace_bytes=config['mapping_workspace_bytes'],
                max_geometry_evaluations=config['max_geometry_evaluations'],
                max_phase_evaluations=config['max_phase_evaluations'],
                deadline_monotonic=deadline)
            result['crossing_vertex'] = plain(crossing.metadata)
            emit('crossing_vertex_complete', result['crossing_vertex'])
            plan = plan_crossing_diagnostic_dimensions(
                4, 601, 216, 120, internal_half_width=2, external_offsets=(-1, 0, 1),
                workers=8, external_tile=16, bin_tile=1)
            result['plan_per_case'] = plain(plan)
            total_work, total_dyson = 0, 0
            for case in profile['checkpoints']:
                if time.monotonic() >= deadline:
                    raise TimeoutError('No further fixed case admitted.')
                emit('crossing_case_start', {'iteration': case['iteration']})
                with ExitStack() as case_stack:
                    sigma, arrays, auth = load_fixed_input(case, case_stack)
                    try:
                        study = study_crossing_support_diagnostic(
                            grid, vertex, crossing, dynamical, *contacts, sigma,
                            g=config['g'], internal_half_width=2, external_offsets=(-1, 0, 1),
                            workers=8, external_tile=16, bin_tile=1, max_bytes=config['max_bytes'],
                            max_contraction_work=config['max_contraction_work'],
                            max_dyson_evaluations=28, deadline_monotonic=deadline)
                        total_work += study['plan']['contraction_work']
                        total_dyson += study['dyson_evaluations']
                        if total_work > config['max_total_contraction_work'] or total_dyson > 84:
                            raise ValueError('Total diagnostic budget exceeded.')
                        result['cases'].append({'iteration': case['iteration'],
                            'manifest_sha256': case['manifest_sha256'], 'study': plain(study)})
                        result['completed_cases'] += 1
                        _material.verify_unchanged(list(auth.values()))
                    finally:
                        release(arrays)
                emit('crossing_case_complete', {'iteration': case['iteration']})
            _material.verify_unchanged(pins+list(static_auth.values()))
            result.update(passed=True, total_contraction_work=total_work,
                          total_dyson_evaluations=total_dyson)
        finally:
            release(static_arrays)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    result = {'passed': False, 'failure_reason': None}
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
    except Exception as exc:  # noqa: BLE001 - retain actual bounded failures
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('crossing_diagnostic_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('crossing_diagnostic_closed', {'passed': result['passed'],
                                      'completed_cases': result.get('completed_cases', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
