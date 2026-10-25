"""Fit a declared nearest-neighbour site-star energy to the unchanged raw bulk FC3."""
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
    '_local_energy_bulk_source', Path(__file__).with_name('run_bulk_bond_energy.py'))
_bulk = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bulk)
_material = _bulk._material
RAW_INPUTS, FROZEN_MAP, RESOURCES, EXPECTED_SOURCE_FC3_SHA256 = (
    _bulk.RAW_INPUTS, _bulk.FROZEN_MAP, _bulk.RESOURCES, _bulk.EXPECTED_SOURCE_FC3_SHA256)
DIAGNOSTIC = {
    'neighbors': 4, 'max_atoms': 128, 'max_bytes': 2*1024**3,
    'max_iterations': 500, 'fit_rtol': 1e-12,
    'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'max_seconds': 79200, 'time_reserve_seconds': 300,
}
plain = _bulk.plain


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.local-energy-fit.profile.v1':
        raise ValueError('Wrong local-energy-fit profile.')
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


def load_fit_source(profile):
    """Load physical source values and the actual two primitive-center classes once."""
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
    center_classes = np.asarray(crystal.s2p)
    representatives = np.asarray(crystal.p2s)
    labels, counts = np.unique(center_classes, return_counts=True)
    if (center_classes.dtype.kind not in 'iu' or center_classes.shape != (64,)
            or representatives.shape != (2,) or len(labels) != 2
            or counts.tolist() != [32, 32]
            or set(labels.tolist()) != set(representatives.tolist())):
        raise ValueError('The actual crystal must have two primitive-center classes of32 atoms each.')
    record = {
        'source_fc3_content_sha256': certification.source_fc3_content_sha256,
        'raw_fc3_certification_passed': True, 'permutation_complete': True,
        'shape': list(raw.values.shape), 'n_atoms': 64,
        'force_constant_units': raw.units, 'mass_units': raw.mass_units,
        'masses': raw.masses.tolist(), 'lattice_angstrom': crystal.supercell.lattice.tolist(),
        'center_class_source': 'actual crystal.s2p supercell-representative labels',
        'center_class_labels': labels.tolist(), 'center_class_counts': counts.tolist(),
        'center_class_normalization': 'deterministic normalization inside the fit module',
        'coordinate_algebra': 'physical Cartesian displacement differences, no mass normalization',
        'raw_fc3_modified': False, 'source_asr_repaired': False,
    }
    return (raw.values, crystal.supercell.scaled_positions, crystal.supercell.lattice,
            center_classes, record)


def run(profile, result, emit, *, profile_sha256):
    from negfpy.anharmonic.local_energy import study_local_energy_fit

    config = validate_profile(profile)
    _material._digest(profile_sha256)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.local-energy-fit.v1', passed=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, diagnostic_config=plain(config),
        map_provenance_only=plain(profile['map']), completed_studies=0, failure_reason=None,
        scope={'scalar_outputs_only': True, 'actual_raw_bulk_fc3': True,
            'explicit_approximate_research_hamiltonian': True,
            'energy_basis': 'cubic monomials on each center and its four nearest-neighbour bonds',
            'all_local_four_atom_clusters_spanned': False,
            'rotational_constraints_imposed': False, 'source_asr_repaired': False,
            'fit_remainder_retained': True, 'fitted_tensor_adopted_as_new_physics': False,
            'finite_boundary_transport_evaluated': False, 'tensor_or_field_output': False,
            'mass_normalization': False, 'g_or_eta_used_numerically': False,
            'map_metadata_provenance_only': True, 'fc2_read': False,
            'static_parent_arrays_loaded': False, 'checkpoint_arrays_loaded': False,
            'green_evaluations': 0, 'born_evaluations': 0, 'current_evaluations': 0, 'scba': False,
            'accepted_field_update': False, 'accepted_restart': False, 'accepted_transport': False,
            'physical_guards_unchanged': True, 'convergence_checks_unchanged': True})
    with ExitStack() as stack:
        pins, _ = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('local_energy_inputs_authenticated', {'count': len(pins)})
        fc3, positions, lattice, classes, result['source'] = load_fit_source(profile)
        emit('local_energy_source_loaded', {'n_atoms': 64, 'center_class_counts': [32, 32]})
        if time.monotonic() >= deadline:
            raise TimeoutError('No local energy fit admitted after the diagnostic deadline.')
        emit('local_energy_fit_start', {'neighbors': 4, 'expected_coefficients': 728,
                                      'max_iterations': config['max_iterations']})
        result['study'] = plain(study_local_energy_fit(
            fc3, positions, lattice, classes, neighbors=config['neighbors'],
            max_atoms=config['max_atoms'], max_bytes=config['max_bytes'],
            max_iterations=config['max_iterations'], fit_rtol=config['fit_rtol'],
            identity_rtol=config['identity_rtol'], identity_atol=config['identity_atol'],
            deadline_monotonic=deadline))
        fit = result['study']['fit']
        if fit['parameter_count'] != 728:
            raise ValueError('The frozen two-class/four-neighbour star basis must contain728 coefficients.')
        # Bounded solver convergence and fit quality are reported separately from execution.
        result['completed_studies'] = 1
        _material.verify_unchanged(pins)
        result['passed'] = True
        emit('local_energy_fit_complete', {
            key: fit[key] for key in
            ('parameter_count', 'iterations', 'stop_code', 'bounded_solver_converged')})


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
    except Exception as exc:  # noqa: BLE001 - preserve actual bounded fit failure evidence
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('local_energy_fit_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('local_energy_fit_closed', {'passed': result['passed'],
                                    'completed_studies': result.get('completed_studies', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
