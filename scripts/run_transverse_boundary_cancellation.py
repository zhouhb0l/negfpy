"""Compare complete source-image acoustic cancellation with the production crop.

This scalar diagnostic rebuilds the accepted actual FC3 source and keeps the two
uncontracted active coordinates fixed. It does not repair IFCs or evaluate a
Green function, Born image, SCBA trajectory, current, or accepted restart.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import stat
import time
import traceback
from collections.abc import Mapping
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._material_nq1_campaign import HBAR_J_S, OMEGA0_RAD_S
from negfpy.anharmonic.fc3 import validate_fc3
from negfpy.anharmonic.io.phono3py import read_phono3py_fc3
from negfpy.anharmonic.io.phono3py_crystal import read_phono3py_crystal_mapping
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import mass_normalize_translational_fc3
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational
from negfpy.anharmonic.transverse_boundary_cancellation import (
    audit_transverse_boundary_cancellation,
)
from negfpy.anharmonic.transverse_material_restart import (
    _arrays_verified,
    _digest,
    _open_verified,
)
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.types import SolverScale

_RAW = '/scratch/users/astar/ihpc/zhouhb/NEGFPy/0.3.0/nq1-g2probe-provenancefast-2d61370/inputs'
_PARENT = '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261021-multiq-material-map-001'
PARENT = {'root': _PARENT,
    'profile_sha256': 'b2a4fba6693e572b4327327e44a5dcc2b58d77758cc7b72579d7f335e8253797',
    'result_sha256': '704c01a502bf71b9d30d3f94acc9eae4b3fb1ed3c20166ad26e19e2af32dcece',
    'source_commit': '9eaf8537a500459e89fa3f9cc2ad4c5223711f55',
    'source_tree': 'a0319c5ec023c9dca8e18464383142abff945aac'}
RUNTIME_INPUTS = [
    {'name': 'fc3', 'path': _RAW+'/fc3.full.hdf5', 'bytes': 17867402,
         'sha256': 'e4e297963b4c2a2a98953b140217baa1a7461b18cb05f12126ae545fe572cd73'},
    {'name': 'fc2', 'path': _RAW+'/validation-fixture/fc2.full.hdf5', 'bytes': 86059,
         'sha256': '6b26d489e6a7fffe61530e77b0c2a907eb4e4f13d5b352abb29d9c0c2c6c74f7'},
    {'name': 'crystal', 'path': _RAW+'/validation-fixture/phono3py_disp.yaml', 'bytes': 32796,
         'sha256': '1375e3ae19a2c4138a888af8177a3becfcaa3b127ca50755773c405c3e28a2a7'},
    {'name': 'parent_profile', 'path': _PARENT+'/profile.json', 'bytes': 3803,
         'sha256': PARENT['profile_sha256']},
    {'name': 'parent_result', 'path': _PARENT+'/runtime/material-map-result.json', 'bytes': 29648,
         'sha256': PARENT['result_sha256']},
]
FROZEN_MAP = {'bin_tile': 1, 'eta_device': 0, 'eta_lead': .0015625,
    'expected_fc3_content_sha256': '6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e',
    'expected_geometry_sha256': '778afae03a7f83580b6eac8ab5529901cbf25fee7fa3ebacbdded85c50a85ab5',
    'external_tile': 16, 'g': .03, 'grid_step': .025, 'interaction_layers': [2, 3, 4, 5, 6],
    'make_r0_average': True, 'map_max_bytes': 17179869184, 'mapping_workspace_bytes': 2147483648,
    'max_contraction_work': 50000000000000, 'max_dyson_evaluations': 2404,
    'max_geometry_evaluations': 2000000, 'max_phase_evaluations': 512000000, 'mesh': [4, 1, 1],
    'n_device_layers': 9, 'n_positive': 300, 'temperatures_kelvin': [305, 295],
    'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
    'vertex_max_bytes': 536870912, 'workers': 8}
RESOURCES = {'blas_threads': 1, 'cpus': 128, 'execution_queue': 'q4', 'memory_gb': 440,
    'numerical_workers': 8, 'rerunnable': False, 'submission_queue': 'normal', 'walltime': '24:00:00'}
DIAGNOSTIC = {'max_seconds': 79200, 'time_reserve_seconds': 300, 'max_records': 36,
    'max_bytes': 2147483648, 'max_geometry_evaluations': 2000000,
    'max_accumulation_work': 1000000000, 'identity_rtol': 1e-8, 'identity_atol': 1e-11}
SOURCE_IDENTITIES = {
    'source_normalized_fc3_sha256': '830ddd8d4a8d2732b17f0639fe7b2b85efdc6bc03b23e5df7c8e0e9f457bec94',
    'source_fc2_sha256': 'd6e7796f41db7d0aad24e24404cfb2ec25b1ad52c5aebc5a47c79cb557b659e7',
    'primitive_crystal_sha256': '14efe6eff8478b04873f8c9a38c01adbfc83d0a6a10d6ffee59d51d073587ec9'}
REFERENCE_ARRAYS = {
    'active_dofs': {'relative_path': 'arrays/active_dofs.npy', 'shape': [120], 'dtype': 'int64',
        'array_bytes': 960, 'file_bytes': 1088,
        'file_sha256': 'f93f589116ed981bd579677c8fd5f2c69af321d536949901939ade5cd3582822'},
    'vertex_before_g': {'relative_path': 'arrays/vertex_before_g.npy',
        'shape': [4, 4, 120, 120, 120], 'dtype': 'complex128', 'array_bytes': 442368000,
        'file_bytes': 442368128,
        'file_sha256': 'ede7945dc70a9c392b06033519810f12ccd41759fc8e305b824050d620f2ff45'}}


def plain(value):
    """Permit finite scalar metadata only; never serialize material arrays."""
    if isinstance(value, np.ndarray):
        raise TypeError('Boundary diagnostic output must not contain arrays.')
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Nonfinite boundary diagnostic scalar.')
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported metadata type: {type(value).__name__}.')


def descriptor(path):
    value = Path(path).stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise ValueError('Pinned input must be a regular non-symlink file.')
    return dict(zip(('device', 'inode', 'bytes', 'mtime_ns', 'ctime_ns'),
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns), strict=True))


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.transverse-boundary-cancellation.profile.v1':
        raise ValueError('Wrong boundary cancellation profile schema.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT),
                          ('runtime_inputs', RUNTIME_INPUTS), ('resources', RESOURCES),
                          ('diagnostic', DIAGNOSTIC)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen boundary diagnostic {key} differs.')
    for key in ('source_commit', 'source_tree'):
        _digest(profile.get(key), size=40)
    return plain(profile['diagnostic'])


def authenticate_inputs(profile, stack):
    """Hash each of five inputs once and parse the two bound JSON descriptors."""
    records, documents = [], {}
    for row in profile['runtime_inputs']:
        before = descriptor(row['path'])
        stream, _ = _open_verified(stack, Path(row['path']), row['sha256'], row['bytes'])
        after = descriptor(row['path'])
        if after != before:
            raise ValueError('Pinned input changed during authentication.')
        records.append(dict(row, stat_before=before, stat_after=after))
        if row['name'] in ('parent_profile', 'parent_result'):
            documents[row['name']] = json.load(stream)
    return records, documents


def validate_parent(profile, documents):
    parent_profile, parent_result = documents['parent_profile'], documents['parent_result']
    if (parent_profile.get('schema') != 'negfpy.transverse-material-born-map.profile.v1'
            or parent_result.get('schema') != 'negfpy.transverse-material-born-map.v1'
            or parent_result.get('passed') is not True
            or parent_profile.get('map') != profile['map']
            or parent_result.get('settings') != profile['map']
            or parent_profile.get('runtime_inputs') != RUNTIME_INPUTS[:3]
            or parent_result.get('source_inputs') != RUNTIME_INPUTS[:3]):
        raise ValueError('Accepted material parent settings or source inputs differ.')
    for key in ('source_commit', 'source_tree'):
        if parent_profile.get(key) != PARENT[key] or parent_result.get(key) != PARENT[key]:
            raise ValueError('Accepted material parent source differs.')
    cubic = parent_result['cubic']
    if (cubic.get('full_ndof') != 216 or cubic.get('active_dofs') != list(range(48, 168))
            or cubic.get('shape') != [4, 4, 120, 120, 120]
            or cubic.get('coupling_applied') is not False
            or cubic.get('solver_units_before_g') is not True
            or cubic.get('all_joint_vertex_checks_passed') is not True
            or parent_result['artifact_state_semantics'].get('accepted_restart') is not False):
        raise ValueError('The saved production reference has a different role or shape.')
    metadata = cubic['metadata']
    for key, expected in SOURCE_IDENTITIES.items():
        if metadata.get(key) != expected:
            raise ValueError('The saved production reference source identity differs.')
    if (metadata.get('source_fc3_sha256') != profile['map']['expected_fc3_content_sha256']
            or metadata.get('layered_geometry_sha256') != profile['map']['expected_geometry_sha256']
            or metadata.get('make_r0_average') is not True
            or metadata.get('weights_applied') is not True
            or metadata.get('symmetrized') is not False
            or metadata.get('boundary_corrected') is not False
            or {key: parent_result['artifacts'].get(key) for key in REFERENCE_ARRAYS}
                != REFERENCE_ARRAYS):
        raise ValueError('The independent saved crop reference differs from the accepted mapping.')


def authenticate_reference(profile, documents, stack):
    """Only two saved arrays are read; contact, Green, Born and checkpoint fields are excluded."""
    validate_parent(profile, documents)
    authentication = {}
    expected = {key: (tuple(row['shape']), row['dtype']) for key, row in REFERENCE_ARRAYS.items()}
    arrays = _arrays_verified(stack, Path(PARENT['root'])/'runtime', REFERENCE_ARRAYS,
        expected, profile['map']['vertex_max_bytes'], authentication)
    try:
        if arrays['active_dofs'].tolist() != list(range(48, 168)):
            raise ValueError('Authenticated reference addresses differ from the exact active crop.')
        vertex = PackedTransverseCubicVertex(UniformBZMesh(tuple(profile['map']['mesh'])),
            arrays['vertex_before_g'], arrays['active_dofs'], 216,
            max_bytes=profile['map']['vertex_max_bytes'])
    finally:
        for array in arrays.values():
            array._mmap.close()
    return vertex, authentication


def rebuild_material(profile):
    """Use the same raw validation, image averaging, and mass conversion as the accepted map."""
    paths = {row['name']: Path(row['path']) for row in profile['runtime_inputs'][:3]}
    cfg = profile['map']
    crystal2 = read_phono3py_crystal_mapping(paths['crystal'], purpose='fc2')
    fc2 = read_phono3py_fc2(paths['fc2'], crystal2.supercell.masses,
        p2s_map=crystal2.p2s, s2p_map=crystal2.s2p, compact=False)
    geometry = build_layered_device_mapping(crystal2, fc2,
        transport_supercell_matrix=np.asarray(cfg['transport_matrix'], dtype=np.int64))
    if (geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1
            or geometry.layered_geometry_sha256 != cfg['expected_geometry_sha256']
            or geometry.fc2_content_sha256 != SOURCE_IDENTITIES['source_fc2_sha256']
            or geometry.primitive_crystal_content_sha256 != SOURCE_IDENTITIES['primitive_crystal_sha256']):
        raise ValueError('Fresh harmonic geometry or actual masses differ from the accepted mapping.')
    crystal3 = read_phono3py_crystal_mapping(paths['crystal'], purpose='fc3')
    raw = read_phono3py_fc3(paths['fc3'], crystal3.supercell.masses,
        p2s_map=crystal3.p2s, s2p_map=crystal3.s2p, compact=False)
    certification = validate_fc3(raw)
    if (not certification.passed or not certification.permutation_complete
            or certification.source_fc3_content_sha256 != cfg['expected_fc3_content_sha256']):
        raise ValueError('Fresh raw material FC3 certification differs or fails.')
    translated = convert_fc3_to_translational(raw, crystal3, make_r0_average=True)
    normalized = mass_normalize_translational_fc3(translated, crystal3,
        source_certification=certification, output_units='si')
    if normalized.content_sha256 != SOURCE_IDENTITIES['source_normalized_fc3_sha256']:
        raise ValueError('Fresh mass-normalized material FC3 identity differs from the accepted source.')
    record = dict(SOURCE_IDENTITIES, raw_fc3_certification_passed=True,
        permutation_complete=True, source_fc3_sha256=certification.source_fc3_content_sha256,
        layered_geometry_sha256=geometry.layered_geometry_sha256, make_r0_average=True,
        output_units='si', fresh_raw_fc3=True, fresh_raw_fc2=True,
        full_device_dofs=216, active_dofs=120, coupling_applied=False)
    return normalized, geometry, record


def verify_unchanged(records):
    for row in records:
        row['stat_after_diagnostic'] = descriptor(row['path'])
        if row['stat_after_diagnostic'] != row['stat_before']:
            raise ValueError('A bound source or saved reference changed during the diagnostic.')


def run(profile, result, emit, *, profile_sha256, clock=time.monotonic):
    config = validate_profile(profile)
    _digest(profile_sha256)
    deadline = clock()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.transverse-boundary-cancellation.v1', passed=False,
        converged=False, source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), diagnostic_config=config,
        parent=plain(profile['parent']), started_audit=False, completed_records=0,
        stop_reason=None, failure_reason=None,
        scope={'scalar_output_only': True, 'actual_material_fc3': True,
            'saved_production_vertex_independent_reference': True,
            'saved_reference_arrays_loaded': 2, 'checkpoint_arrays_loaded': 0,
            'green_evaluations': 0, 'born_evaluations': 0, 'scba': False,
            'physical_guards_unchanged': True, 'convergence_checks_unchanged': True,
            'field_output': False, 'ifc_repair': False, 'projection': False,
            'accepted_field_update': False, 'accepted_restart': False, 'current_acceptance': False,
            'converged_transport': False, 'frequency_or_q_convergence': False,
            'complete_support_residual_is_diagnostic_only': True})
    with ExitStack() as stack:
        emit('boundary_input_authentication_start')
        records, documents = authenticate_inputs(profile, stack)
        result['runtime_input_authentication'] = records
        emit('boundary_inputs_authenticated', {'count': len(records)})
        reference, result['reference_array_authentication'] = authenticate_reference(profile, documents, stack)
        reference_records = list(result['reference_array_authentication'].values())
        emit('boundary_reference_authenticated', {'array_count': len(reference_records)})
        try:
            if clock() >= deadline:
                result['stop_reason'] = 'time_admission_before_material_reconstruction'
                return
            normalized, geometry, result['material_reconstruction'] = rebuild_material(profile)
            emit('boundary_material_rebuilt', result['material_reconstruction'])
            if clock() >= deadline:
                result['stop_reason'] = 'time_admission_before_audit'
                return
            result['started_audit'] = True
            emit('boundary_audit_start', {'maximum_records': config['max_records']})
            audit = plain(audit_transverse_boundary_cancellation(
                UniformBZMesh(tuple(profile['map']['mesh'])), normalized, geometry,
                n_device_layers=9, interaction_layers=tuple(profile['map']['interaction_layers']),
                scale=SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S), reference_vertex=reference,
                max_workspace_bytes=config['max_bytes'],
                max_geometry_evaluations=config['max_geometry_evaluations'],
                max_accumulation_work=config['max_accumulation_work'],
                deadline_monotonic=deadline))
            result['boundary_audit'] = audit
            if len(audit.get('records', [])) != config['max_records']:
                raise ValueError('Boundary audit did not complete its exact 36 scalar contractions.')
            result['completed_records'] = len(audit['records'])
            result.update(passed=True, stop_reason='all_boundary_contractions_completed')
            emit('boundary_audit_complete', {'completed_records': result['completed_records']})
        finally:
            verify_unchanged(records+reference_records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {'schema': 'negfpy.transverse-boundary-cancellation.v1', 'passed': False, 'converged': False}

    def emit(event, details=None):
        row = {'time': datetime.now(timezone.utc).isoformat(), 'event': event,
            'elapsed_seconds': time.monotonic()-started, 'details': plain(details or {})}
        text = json.dumps(row, sort_keys=True, allow_nan=False)
        with (args.output.parent/'events.jsonl').open('a') as stream:
            stream.write(text+'\n')
        print(text, flush=True)

    try:
        with args.profile.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        run(profile, result, emit, profile_sha256=digest)
    except Exception as exc:  # noqa: BLE001 - preserve actual bounded diagnostic failures
        result.update(passed=False, stop_reason='diagnostic_failed',
            failure_reason=type(exc).__name__+': '+str(exc),
            failure={'type': type(exc).__name__, 'message': str(exc)})
        traceback.print_exc()
        emit('boundary_diagnostic_failed', result['failure'])
    result['elapsed_seconds'] = time.monotonic()-started
    result['maximum_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('boundary_diagnostic_closed', {'passed': result['passed'], 'converged': False,
        'completed_records': result.get('completed_records', 0), 'stop_reason': result.get('stop_reason'),
        'failure_reason': result.get('failure_reason'), 'elapsed_seconds': result['elapsed_seconds']})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
