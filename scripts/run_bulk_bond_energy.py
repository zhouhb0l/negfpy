"""Measure a complete relative-coordinate decomposition of the raw bulk cubic energy."""
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

_spec = importlib.util.spec_from_file_location(
    '_bond_material_source', Path(__file__).with_name('run_transverse_boundary_cancellation.py'))
_material = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_material)
RAW_INPUTS = [_material.RUNTIME_INPUTS[0], _material.RUNTIME_INPUTS[2]]
FROZEN_MAP, RESOURCES = _material.FROZEN_MAP, _material.RESOURCES
EXPECTED_SOURCE_FC3_SHA256 = '6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e'
DIAGNOSTIC = {
    'max_atoms': 128, 'max_bytes': 2*1024**3,
    'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'max_seconds': 79200, 'time_reserve_seconds': 300,
}
plain = _material.plain


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.bulk-bond-energy.profile.v1':
        raise ValueError('Wrong bulk bond-energy profile.')
    for key, expected in (('map', FROZEN_MAP), ('resources', RESOURCES),
                          ('runtime_inputs', RAW_INPUTS), ('diagnostic', DIAGNOSTIC),
                          ('expected_source_fc3_sha256', EXPECTED_SOURCE_FC3_SHA256),
                          ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    if any(key in profile for key in ('parent', 'checkpoint', 'checkpoints', 'resume', 'restart', 'scba')):
        raise ValueError('Only raw bulk FC3 and crystal inputs are admitted; no trajectory inputs.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return DIAGNOSTIC


def load_bulk_source(profile):
    """Read the full physical-displacement FC3 source without mass conversion or ASR repair."""
    paths = {row['name']: Path(row['path']) for row in profile['runtime_inputs']}
    crystal = _material.read_phono3py_crystal_mapping(paths['crystal'], purpose='fc3')
    raw = _material.read_phono3py_fc3(paths['fc3'], crystal.supercell.masses,
        p2s_map=crystal.p2s, s2p_map=crystal.s2p, compact=False)
    certification = _material.validate_fc3(raw)
    if (not certification.passed or not certification.permutation_complete
            or certification.source_fc3_content_sha256 != EXPECTED_SOURCE_FC3_SHA256):
        raise ValueError('Raw bulk FC3 certification or source content differs.')
    if (raw.compact or tuple(raw.values.shape) != (64, 64, 64, 3, 3, 3)
            or crystal.supercell.n_atoms != 64 or raw.units != 'eV/angstrom^3'):
        raise ValueError('The full64-atom physical-unit bulk FC3 source is required.')
    if not np.array_equal(raw.masses, crystal.supercell.masses):
        raise ValueError('Raw FC3 and crystal mass ordering differs.')
    record = {
        'source_fc3_content_sha256': certification.source_fc3_content_sha256,
        'raw_fc3_certification_passed': True, 'permutation_complete': True,
        'shape': list(raw.values.shape), 'n_atoms': 64,
        'force_constant_units': raw.units, 'mass_units': raw.mass_units,
        'masses': raw.masses.tolist(), 'lattice_angstrom': crystal.supercell.lattice.tolist(),
        'coordinate_algebra': 'physical Cartesian displacements, without mass normalization',
        'raw_fc3_modified': False, 'source_asr_repaired': False,
    }
    return raw.values, crystal.supercell.scaled_positions, crystal.supercell.lattice, record


def run(profile, result, emit, *, profile_sha256):
    from negfpy.anharmonic.bond_energy import study_bulk_bond_energy

    config = validate_profile(profile)
    _material._digest(profile_sha256)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.bulk-bond-energy.v1', passed=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, diagnostic_config=plain(config),
        map_provenance_only=plain(profile['map']), completed_studies=0, failure_reason=None,
        scope={'scalar_outputs_only': True, 'actual_raw_bulk_fc3': True,
            'complete_source_decomposition': 'T = T_relative + explicit_remainder',
            'remainder_retained': True, 'relative_part_adopted_as_new_physics': False,
            'finite_interaction_boundary_defined': False, 'source_asr_repaired': False,
            'tensor_or_field_output': False, 'mass_normalization': False,
            'g_or_eta_used_numerically': False, 'map_metadata_provenance_only': True,
            'fc2_read': False, 'static_parent_arrays_loaded': False,
            'checkpoint_arrays_loaded': False, 'green_evaluations': 0, 'born_evaluations': 0,
            'current_evaluations': 0, 'scba': False, 'accepted_field_update': False,
            'accepted_restart': False, 'accepted_transport': False,
            'physical_guards_unchanged': True, 'convergence_checks_unchanged': True})
    with ExitStack() as stack:
        pins, _ = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('bond_inputs_authenticated', {'count': len(pins)})
        fc3, positions, lattice, result['source'] = load_bulk_source(profile)
        emit('bond_source_loaded', {'n_atoms': 64, 'units': result['source']['force_constant_units']})
        if time.monotonic() >= deadline:
            raise TimeoutError('No decomposition admitted after the diagnostic deadline.')
        emit('bond_study_start')
        result['study'] = plain(study_bulk_bond_energy(
            fc3, positions, lattice, max_atoms=config['max_atoms'], max_bytes=config['max_bytes'],
            deadline_monotonic=deadline, identity_rtol=config['identity_rtol'],
            identity_atol=config['identity_atol']))
        # A large explicit source remainder is a measured limitation, not an execution veto.
        result['completed_studies'] = 1
        _material.verify_unchanged(pins)
        result['passed'] = True
        emit('bond_study_complete', {'remainder_retained': True})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    digest = hashlib.sha256(args.profile.read_bytes()).hexdigest()
    result = {'passed': False, 'failure_reason': None,
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
    except Exception as exc:  # noqa: BLE001 - preserve actual incomplete diagnostic evidence
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('bond_study_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('bond_diagnostic_closed', {'passed': result['passed'],
                                   'completed_studies': result.get('completed_studies', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
