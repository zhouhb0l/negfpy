"""Common-Green partial Born sensitivity to a declared fitted site-energy boundary."""
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

from negfpy.anharmonic.transverse_crossing_feedback import (
    build_crossing_feedback_vertex,
)
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_site_born_acoustic_source', Path(__file__).with_name('run_transverse_acoustic_sector.py'))
_acoustic = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_acoustic)
_static, _material = _acoustic._static, _acoustic._material
PARENT, FROZEN_MAP, RESOURCES, RAW_INPUTS, SUPPORT = (
    _acoustic.PARENT, _acoustic.FROZEN_MAP, _acoustic.RESOURCES,
    _acoustic.RAW_INPUTS, _acoustic.SUPPORT)
FIT = {'neighbors': 4, 'max_atoms': 128, 'max_bytes': 2*1024**3,
       'max_iterations': 500, 'fit_rtol': 1e-12, 'identity_rtol': 1e-8, 'identity_atol': 1e-11}
SITE_SUPPORT = {'max_bytes': 4*1024**3, 'max_workspace_bytes': 32*1024**3}
DIAGNOSTIC = {
    'g': .03, 'identity_rtol': 1e-8, 'identity_atol': 1e-11,
    'internal_half_width': 2, 'workers': 8, 'external_tile': 16, 'bin_tile': 1,
    'max_bytes': 4*1024**3, 'max_contraction_work': 1_600_000_000_000,
    'max_dyson_evaluations': 20, 'max_seconds': 79200, 'time_reserve_seconds': 300,
    'checkpoint_max_bytes': 8*1024**3,
}
CHECKPOINT_PATH, CHECKPOINT_SHA256, CHECKPOINT_BYTES = (
    _acoustic.CHECKPOINT_PATH, _acoustic.CHECKPOINT_SHA256, _acoustic.CHECKPOINT_BYTES)
MODEL_KEYS = ('actual_ifc_reference', 'fitted_complete_energy', 'fitted_derivative_crop')
plain = _acoustic.plain
load_fixed_input, rigid_translations = _acoustic.load_fixed_input, _acoustic.rigid_translations


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.site-energy-born.profile.v1':
        raise ValueError('Wrong site-energy-born profile.')
    for key, expected in (('fit', FIT), ('site_support', SITE_SUPPORT), ('diagnostic', DIAGNOSTIC)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    # Reuse the unchanged physics, six input pins and exact saved state2 binding.
    _acoustic.validate_profile(dict(profile,
        schema='negfpy.transverse-acoustic-sector.profile.v1', diagnostic=_acoustic.DIAGNOSTIC))
    return DIAGNOSTIC


def load_material_for_study(profile):
    """Read each raw IFC once, retaining physical FC3 for the one declared fit."""
    paths = {row['name']: Path(row['path']) for row in profile['runtime_inputs']}
    cfg, identities = profile['map'], _material.SOURCE_IDENTITIES
    crystal2 = _material.read_phono3py_crystal_mapping(paths['crystal'], purpose='fc2')
    fc2 = _material.read_phono3py_fc2(paths['fc2'], crystal2.supercell.masses,
        p2s_map=crystal2.p2s, s2p_map=crystal2.s2p, compact=False)
    geometry = _material.build_layered_device_mapping(crystal2, fc2,
        transport_supercell_matrix=np.asarray(cfg['transport_matrix'], dtype=np.int64))
    if (geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1
            or geometry.layered_geometry_sha256 != cfg['expected_geometry_sha256']
            or geometry.fc2_content_sha256 != identities['source_fc2_sha256']
            or geometry.primitive_crystal_content_sha256 != identities['primitive_crystal_sha256']):
        raise ValueError('Fresh harmonic geometry or actual masses differ from the accepted mapping.')
    crystal = _material.read_phono3py_crystal_mapping(paths['crystal'], purpose='fc3')
    raw = _material.read_phono3py_fc3(paths['fc3'], crystal.supercell.masses,
        p2s_map=crystal.p2s, s2p_map=crystal.s2p, compact=False)
    certification = _material.validate_fc3(raw)
    if (not certification.passed or not certification.permutation_complete
            or certification.source_fc3_content_sha256 != cfg['expected_fc3_content_sha256']
            or raw.compact or tuple(raw.values.shape) != (64, 64, 64, 3, 3, 3)
            or crystal.supercell.n_atoms != 64 or raw.units != 'eV/angstrom^3'
            or not np.array_equal(raw.masses, crystal.supercell.masses)):
        raise ValueError('The unchanged full64-atom physical-unit FC3 source is required.')
    classes = np.asarray(crystal.s2p)
    labels, counts = np.unique(classes, return_counts=True)
    if (classes.shape != (64,) or classes.dtype.kind not in 'iu' or len(labels) != 2
            or counts.tolist() != [32, 32]
            or set(labels.tolist()) != set(np.asarray(crystal.p2s).tolist())):
        raise ValueError('Two actual primitive-center classes of32 atoms each are required.')
    translated = _material.convert_fc3_to_translational(raw, crystal, make_r0_average=True)
    normalized = _material.mass_normalize_translational_fc3(
        translated, crystal, source_certification=certification, output_units='si')
    if normalized.content_sha256 != identities['source_normalized_fc3_sha256']:
        raise ValueError('Normalized actual IFC reference identity differs.')
    record = dict(identities, source_fc3_sha256=certification.source_fc3_content_sha256,
        layered_geometry_sha256=geometry.layered_geometry_sha256,
        force_constant_units=raw.units, mass_units=raw.mass_units,
        raw_fc3_certification_passed=True, permutation_complete=True,
        raw_fc3_modified=False, source_asr_repaired=False,
        center_class_labels=labels.tolist(), center_class_counts=counts.tolist(),
        fit_coordinates='physical Cartesian displacement differences before mass normalization')
    return normalized, geometry, crystal, raw.values, classes, record


def run(profile, result, emit, *, profile_sha256):
    from negfpy.anharmonic.site_energy_born import study_site_energy_born
    from negfpy.anharmonic.site_energy_boundary import build_site_energy_vertex
    from negfpy.anharmonic.site_energy_model import fit_site_energy_model

    config = validate_profile(profile)
    _material._digest(profile_sha256)
    deadline = time.monotonic()+config['max_seconds']-config['time_reserve_seconds']
    result.update(schema='negfpy.site-energy-born.v1', passed=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), diagnostic_config=plain(config),
        fixed_checkpoint=plain(profile['checkpoint']), completed_studies=0, failure_reason=None,
        scope={'scalar_outputs_only': True, 'common_full216_green': True,
            'same_fitted_model_for_complete_energy_and_crop': True,
            'one_fit_only': True, 'explicit_approximate_research_surrogate': True,
            'raw_fit_remainder_measured_and_reported': True,
            'surrogate_vertex_includes_raw_fit_remainder': False,
            'raw_actual_ifc_reference_retained': True, 'raw_ifc_modified': False,
            'fitted_model_adopted_as_production_physics': False,
            'fixed_evaluated_state2_input': True, 'mapped_images_loaded': False,
            'same_native_channels_for_all_models': True,
            'all_required_internal_endpoint_coordinates_retained': True,
            'new_surrogate_dyson_solved': False, 'partial_born_only': True,
            'full_born_map': False, 'current_evaluations': 0, 'scba': False,
            'physical_guards_unchanged': True, 'convergence_checks_unchanged': True,
            'eta_changed': False, 'g_changed': False, 'accepted_restart': False,
            'accepted_field_update': False, 'accepted_transport': False})
    with ExitStack() as stack:
        pins, documents = _material.authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('site_energy_inputs_authenticated', {'count': len(pins)})
        mesh, grid, reference, dynamical, contacts, static_arrays, static_auth = _static.load_static(
            profile, documents, stack)
        try:
            normalized, geometry, crystal, raw_values, classes, result['material'] = load_material_for_study(profile)
            translations, result['translation_basis'] = rigid_translations(geometry)
            emit('site_energy_fit_start', {'expected_coefficients': 728})
            fitted = fit_site_energy_model(
                raw_values, crystal.supercell.scaled_positions, crystal.supercell.lattice,
                classes, deadline_monotonic=deadline, **FIT)
            result['fit_report'] = plain(fitted.fit_report)
            result['fit_geometry'] = plain(fitted.geometry)
            if result['fit_report']['fit']['parameter_count'] != 728:
                raise ValueError('The frozen fitted energy basis must contain728 coefficients.')
            emit('site_energy_fit_complete', {
                key: result['fit_report']['fit'][key] for key in
                ('parameter_count', 'iterations', 'stop_code', 'bounded_solver_converged')})
            scale = SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S)
            emit('site_energy_vertex_start')
            raw_reference = build_crossing_feedback_vertex(
                mesh, normalized, geometry, n_device_layers=9,
                interaction_layers=(2, 3, 4, 5, 6), scale=scale,
                reference_vertex=reference, deadline_monotonic=deadline,
                **{key: value for key, value in SUPPORT.items() if key != 'model'})
            if raw_reference.metadata['model_identity_sha256'] != _acoustic.MODEL_IDENTITY:
                raise ValueError('Rebuilt actual IFC reference differs from the fixed state2 Hamiltonian.')
            built = {'actual_ifc_reference': raw_reference}
            for key, crop in (('fitted_complete_energy', False), ('fitted_derivative_crop', True)):
                built[key] = build_site_energy_vertex(
                    mesh, fitted, crystal, geometry, n_device_layers=9,
                    interaction_layers=(2, 3, 4, 5, 6), scale=scale,
                    force_constant_units=result['material']['force_constant_units'],
                    mass_units=result['material']['mass_units'], crop_derivatives=crop,
                    deadline_monotonic=deadline, **SITE_SUPPORT)
            result['vertex_metadata'] = {key: plain(value.metadata) for key, value in built.items()}
            emit('site_energy_vertex_complete', {'models': list(built)})
            with ExitStack() as case_stack:
                sigma, input_arrays, input_auth = load_fixed_input(profile['checkpoint'], case_stack)
                try:
                    emit('site_energy_born_start', {'iteration': 2, 'dyson_budget': 20})
                    result['study'] = plain(study_site_energy_born(
                        grid, {key: value.vertex for key, value in built.items()}, dynamical,
                        *contacts, sigma, translations, np.arange(48, 168),
                        g=config['g'], internal_half_width=2, workers=8, external_tile=16, bin_tile=1,
                        max_bytes=config['max_bytes'], max_contraction_work=config['max_contraction_work'],
                        max_dyson_evaluations=20, deadline_monotonic=deadline,
                        identity_rtol=config['identity_rtol'], identity_atol=config['identity_atol'], emit=emit))
                    _material.verify_unchanged(list(input_auth.values()))
                    result['input_authentication'] = plain(input_auth)
                    result['completed_studies'] = 1
                    emit('site_energy_born_complete')
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
    except Exception as exc:  # noqa: BLE001 - retain actual incomplete bounded study evidence
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('site_energy_born_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('site_energy_born_closed', {'passed': result['passed'],
                                   'completed_studies': result.get('completed_studies', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
