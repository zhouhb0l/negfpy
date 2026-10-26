"""Bounded native SCBA continuation of the exact saved complete-energy surrogate."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
from negfpy.anharmonic.site_energy_saved import restore_site_energy_model
from negfpy.anharmonic.transverse_channel_accel import (
    ChannelAccelerationConfig,
    ChannelSelectionPolicy,
    plan_channel_accel,
    run_channel_accel,
)
from negfpy.anharmonic.transverse_channel_feasibility import ChannelProposalFeasibility
from negfpy.anharmonic.transverse_channel_feedback import ChannelSCBAConfig
from negfpy.anharmonic.transverse_channel_restart import (
    load_channel_checkpoint,
    save_channel_checkpoint,
)
from negfpy.anharmonic.transverse_material_restart import _open_verified
from negfpy.anharmonic.types import SolverScale

_spec = importlib.util.spec_from_file_location(
    '_site_convergence_feedback', Path(__file__).with_name('run_site_energy_feedback.py'))
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)
_material, _operator = _base._material, _base._operator
PARENT, FROZEN_MAP, RESOURCES, FIT, SITE_SUPPORT = (
    _base.PARENT, _base.FROZEN_MAP, _base.RESOURCES, _base.FIT, _base.SITE_SUPPORT)
SCBA = dict(_base.SCBA, max_evaluations=16)
ACCELERATION = {'mode': 'anderson1', 'coefficient_limit': 5., 'trust_radius': .25, 'rank_rtol': 1e-10}
SELECTION = {'mode': 'component-nonincreasing', 'component_growth_limit': 1., 'merit_ratio_limit': 1.}
FEASIBILITY = {'mode': 'psd-backtracking', 'factor': .5, 'max_halvings': 8}
CHECKPOINT_MAX_BYTES = 8*1024**3
MIN_FREE_BYTES = 128*1024**3
INITIAL_STATE = 'native-channel-continuation'
MODEL = _base.MODEL
plain, zero_support_context = _base.plain, _base.zero_support_context
load_static_operator = _base.load_static_operator

SEED = {'binding': {'bulk_fit_residual': {'covered_fit_remainder': {'frobenius': 8.859213080188733,
                                                             'max_abs': 0.09979940543831134},
                                   'explicitly_reported': True,
                                   'frobenius': 24.289642611912438,
                                   'included_in_surrogate_vertex': False,
                                   'max_abs': 0.09979940543831134,
                                   'outside_coverage_irreducible_floor': {'frobenius': 22.616389672410854,
                                                                          'max_abs': 0.08382997006809358},
                                   'outside_floor_relative_frobenius': 0.019939490240425517,
                                   'raw_covered': {'frobenius': 1134.0256447733343,
                                                   'max_abs': 33.02960294758116},
                                   'raw_source_retained': True,
                                   'relative_frobenius': 0.021414695219655902,
                                   'sum_closure': {'allowance': 3.3030602947581165e-07,
                                                   'error_to_allowance_ratio': 1.0503734846922261e-11,
                                                   'frobenius_error': 2.8846897590451474e-16,
                                                   'max_abs_error': 3.469446951953614e-18,
                                                   'passed': True,
                                                   'scale': 33.02960294758116}},
             'energy_identity_sha256': 'dce0ab054db0999b8923e44ff7f1e85f4a778e91d8f84654f9c4db91636a867a',
             'feedback_dofs': [24,
                               25,
                               26,
                               30,
                               31,
                               32,
                               48,
                               49,
                               50,
                               51,
                               52,
                               53,
                               54,
                               55,
                               56,
                               57,
                               58,
                               59,
                               60,
                               61,
                               62,
                               63,
                               64,
                               65,
                               66,
                               67,
                               68,
                               69,
                               70,
                               71,
                               72,
                               73,
                               74,
                               75,
                               76,
                               77,
                               78,
                               79,
                               80,
                               81,
                               82,
                               83,
                               84,
                               85,
                               86,
                               87,
                               88,
                               89,
                               90,
                               91,
                               92,
                               93,
                               94,
                               95,
                               96,
                               97,
                               98,
                               99,
                               100,
                               101,
                               102,
                               103,
                               104,
                               105,
                               106,
                               107,
                               108,
                               109,
                               110,
                               111,
                               112,
                               113,
                               114,
                               115,
                               116,
                               117,
                               118,
                               119,
                               120,
                               121,
                               122,
                               123,
                               124,
                               125,
                               126,
                               127,
                               128,
                               129,
                               130,
                               131,
                               132,
                               133,
                               134,
                               135,
                               136,
                               137,
                               138,
                               139,
                               140,
                               141,
                               142,
                               143,
                               144,
                               145,
                               146,
                               147,
                               148,
                               149,
                               150,
                               151,
                               152,
                               153,
                               154,
                               155,
                               156,
                               157,
                               158,
                               159,
                               160,
                               161,
                               162,
                               163,
                               164,
                               165,
                               166,
                               167,
                               171,
                               172,
                               173,
                               177,
                               178,
                               179],
             'fit_configuration': {'fit_rtol': 1e-12,
                                   'identity_atol': 1e-11,
                                   'identity_rtol': 1e-08,
                                   'max_atoms': 128,
                                   'max_bytes': 2147483648,
                                   'max_iterations': 500,
                                   'neighbors': 4},
             'full_device_dofs': 216,
             'initial_state': 'fresh-zero',
             'model_identity_sha256': 'ae5d0d7db3b80ddcfc6fc8e81b98fb1310424165a019abc67a6c075268209e65',
             'model_parameters': {'actual_ifc_tensor': False,
                                  'array_bytes': 5824,
                                  'coefficient_semantics': 'two primitive-center classes; ordered cubic '
                                                           'site-star monomials and normalization fixed by '
                                                           'the bound source and energy identity',
                                  'descriptor_relative_path': 'model-parameters/descriptor.json',
                                  'dtype': 'float64',
                                  'energy_identity_sha256': 'dce0ab054db0999b8923e44ff7f1e85f4a778e91d8f84654f9c4db91636a867a',
                                  'file_bytes': 5952,
                                  'file_sha256': '92358d713fc2b622e94e9c977154124b6575f01b057fe2645a20fb85ccd6fad9',
                                  'fit_configuration': {'fit_rtol': 1e-12,
                                                        'identity_atol': 1e-11,
                                                        'identity_rtol': 1e-08,
                                                        'max_atoms': 128,
                                                        'max_bytes': 2147483648,
                                                        'max_iterations': 500,
                                                        'neighbors': 4},
                                  'hash_domain': 'whole NPY file including header',
                                  'model_identity_sha256': 'ae5d0d7db3b80ddcfc6fc8e81b98fb1310424165a019abc67a6c075268209e65',
                                  'production_ifc_adoption': False,
                                  'profile_sha256': 'a18a460ea4a671759bf3558c5b7ef35f5999feba125c681c643b440b03c96495',
                                  'relative_path': 'model-parameters/coefficients.npy',
                                  'research_trajectory_parameters': True,
                                  'role': 'exact energy parameters used for this declared research '
                                          'trajectory',
                                  'schema': 'negfpy.site-energy-parameters.v1',
                                  'shape': [2, 364],
                                  'source_commit': '998ab79d89fde57e0d81f3b123d79c4e66e6f6d0',
                                  'source_tree': 'ab5e3240de656a688d50c8178e94029cde32acf2'},
             'new_hamiltonian_trajectory': True,
             'outside_feedback_dofs_exact_zero': True,
             'parent_profile_sha256': 'b2a4fba6693e572b4327327e44a5dcc2b58d77758cc7b72579d7f335e8253797',
             'parent_result_sha256': '704c01a502bf71b9d30d3f94acc9eae4b3fb1ed3c20166ad26e19e2af32dcece',
             'physics': {'bin_tile': 1,
                         'eta_device': 0,
                         'eta_lead': 0.0015625,
                         'expected_fc3_content_sha256': '6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e',
                         'expected_geometry_sha256': '778afae03a7f83580b6eac8ab5529901cbf25fee7fa3ebacbdded85c50a85ab5',
                         'external_tile': 16,
                         'g': 0.03,
                         'grid_step': 0.025,
                         'interaction_layers': [2, 3, 4, 5, 6],
                         'make_r0_average': True,
                         'map_max_bytes': 17179869184,
                         'mapping_workspace_bytes': 2147483648,
                         'max_contraction_work': 50000000000000,
                         'max_dyson_evaluations': 2404,
                         'max_geometry_evaluations': 2000000,
                         'max_phase_evaluations': 512000000,
                         'mesh': [4, 1, 1],
                         'n_device_layers': 9,
                         'n_positive': 300,
                         'temperatures_kelvin': [305, 295],
                         'transport_matrix': [[-1, 1, 1], [1, -1, 1], [1, 1, -1]],
                         'vertex_max_bytes': 536870912,
                         'workers': 8},
             'production_ifc_adoption': False,
             'profile_sha256': 'a18a460ea4a671759bf3558c5b7ef35f5999feba125c681c643b440b03c96495',
             'raw_fit_remainder_in_surrogate': False,
             'representation': 'retarded-statistical-spectral-v1',
             'site_support_configuration': {'max_bytes': 4294967296, 'max_workspace_bytes': 34359738368},
             'source_commit': '998ab79d89fde57e0d81f3b123d79c4e66e6f6d0',
             'source_tree': 'ab5e3240de656a688d50c8178e94029cde32acf2',
             'support_model': 'fitted-complete-center-star-v1'},
 'iteration': 2,
 'manifest_sha256': 'ea4c9bde9395b88d0acc39d1077939158b6976e2ae17345ac53c38c0704cbbfb',
 'model_parameters': {'actual_ifc_tensor': False,
                      'array_bytes': 5824,
                      'coefficient_semantics': 'two primitive-center classes; ordered cubic site-star '
                                               'monomials and normalization fixed by the bound source and '
                                               'energy identity',
                      'descriptor_relative_path': 'model-parameters/descriptor.json',
                      'dtype': 'float64',
                      'energy_identity_sha256': 'dce0ab054db0999b8923e44ff7f1e85f4a778e91d8f84654f9c4db91636a867a',
                      'file_bytes': 5952,
                      'file_sha256': '92358d713fc2b622e94e9c977154124b6575f01b057fe2645a20fb85ccd6fad9',
                      'fit_configuration': {'fit_rtol': 1e-12,
                                            'identity_atol': 1e-11,
                                            'identity_rtol': 1e-08,
                                            'max_atoms': 128,
                                            'max_bytes': 2147483648,
                                            'max_iterations': 500,
                                            'neighbors': 4},
                      'hash_domain': 'whole NPY file including header',
                      'model_identity_sha256': 'ae5d0d7db3b80ddcfc6fc8e81b98fb1310424165a019abc67a6c075268209e65',
                      'production_ifc_adoption': False,
                      'profile_sha256': 'a18a460ea4a671759bf3558c5b7ef35f5999feba125c681c643b440b03c96495',
                      'relative_path': 'model-parameters/coefficients.npy',
                      'research_trajectory_parameters': True,
                      'role': 'exact energy parameters used for this declared research trajectory',
                      'schema': 'negfpy.site-energy-parameters.v1',
                      'shape': [2, 364],
                      'source_commit': '998ab79d89fde57e0d81f3b123d79c4e66e6f6d0',
                      'source_tree': 'ab5e3240de656a688d50c8178e94029cde32acf2'},
 'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/runtime/checkpoints/iteration-000002'}

SEED_PINS = [{'bytes': 5453,
  'name': 'seed_profile',
  'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/profile.json',
  'sha256': 'a18a460ea4a671759bf3558c5b7ef35f5999feba125c681c643b440b03c96495'},
 {'bytes': 184409,
  'name': 'seed_result',
  'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/runtime/site-energy-feedback-result.json',
  'sha256': 'd9227c06dda6072c6b687da62b9989497898684ec2e7c4e30ca994f54feb457e'},
 {'bytes': 46112,
  'name': 'seed_manifest',
  'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/runtime/checkpoints/iteration-000002/manifest.json',
  'sha256': 'ea4c9bde9395b88d0acc39d1077939158b6976e2ae17345ac53c38c0704cbbfb'},
 {'bytes': 1365,
  'name': 'model_descriptor',
  'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/runtime/model-parameters/descriptor.json',
  'sha256': '5a33aa5dfd8f9e526af490173235e90ea583224285ec0bf91e50513c589e5507'},
 {'bytes': 5952,
  'name': 'model_coefficients',
  'path': '/scratch/users/astar/ihpc/zhouhb/NEGFPy/diagnostics/20261026-site-energy-feedback-001/runtime/model-parameters/coefficients.npy',
  'sha256': '92358d713fc2b622e94e9c977154124b6575f01b057fe2645a20fb85ccd6fad9'}]
RAW_INPUTS = _base.RAW_INPUTS+SEED_PINS


def validate_profile(profile):
    if profile.get('schema') != 'negfpy.site-energy-convergence.profile.v1':
        raise ValueError('Wrong site-energy-convergence profile.')
    for key, expected in (('map', FROZEN_MAP), ('parent', PARENT), ('resources', RESOURCES),
                          ('runtime_inputs', RAW_INPUTS), ('fit', FIT), ('site_support', SITE_SUPPORT),
                          ('scba', SCBA), ('acceleration', ACCELERATION), ('selection_policy', SELECTION),
                          ('proposal_feasibility', FEASIBILITY), ('seed', SEED),
                          ('checkpoint_max_bytes', CHECKPOINT_MAX_BYTES),
                          ('min_free_bytes', MIN_FREE_BYTES), ('initial_state', INITIAL_STATE),
                          ('driver_timeout_seconds', 82800)):
        if profile.get(key) != expected:
            raise ValueError(f'Frozen {key} differs.')
    if any(key in profile for key in ('restart', 'checkpoints', 'sigma_initial')):
        raise ValueError('Only the exact native fitted state2 seed is admitted.')
    for key in ('source_commit', 'source_tree'):
        _material._digest(profile[key], size=40)
    return ChannelSCBAConfig(**SCBA)


def authenticate_inputs(profile, stack):
    """Authenticate bound metadata/material/parameters oncompute; never load raw FC3."""
    records, documents = [], {}
    for row in profile['runtime_inputs']:
        before = _material.descriptor(row['path'])
        stream, _ = _open_verified(stack, Path(row['path']), row['sha256'], row['bytes'])
        after = _material.descriptor(row['path'])
        if before != after:
            raise ValueError('A pinned continuation input changed during authentication.')
        records.append(dict(row, stat_before=before, stat_after=after))
        if Path(row['path']).suffix == '.json':
            documents[row['name']] = json.load(stream)
        elif row['name'] == 'model_coefficients':
            documents[row['name']] = np.load(stream, allow_pickle=False)
    return records, documents


def load_seed_model(profile, documents, deadline):
    """Rebuild deterministic geometry and use exact saved coefficients without a fit."""
    old, saved, manifest = (documents[name] for name in ('seed_profile', 'seed_result', 'seed_manifest'))
    descriptor = documents['model_descriptor']
    if (old['source_commit'] != SEED['binding']['source_commit']
            or old['source_tree'] != SEED['binding']['source_tree']
            or old['map'] != FROZEN_MAP or old['fit'] != FIT
            or saved.get('passed') is not True or saved.get('completed_evaluations') != 2
            or saved['settings'] != FROZEN_MAP or saved['fit_config'] != FIT
            or saved['source_commit'] != old['source_commit']
            or saved['source_tree'] != old['source_tree']
            or saved['profile_sha256'] != SEED['binding']['profile_sha256']
            or descriptor != SEED['model_parameters'] or saved['model_parameters'] != descriptor
            or manifest['binding'] != SEED['binding'] or manifest['iteration'] != 2):
        raise ValueError('The saved fitted model, seed or source binding differs.')
    geometry, harmonic = _operator.rebuild_harmonic_geometry(profile)
    paths = {row['name']: Path(row['path']) for row in profile['runtime_inputs']}
    crystal = _material.read_phono3py_crystal_mapping(paths['crystal'], purpose='fc3')
    classes = np.asarray(crystal.s2p)
    if (crystal.supercell.n_atoms != 64 or classes.shape != (64,) or classes.dtype.kind not in 'iu'
            or np.unique(classes, return_counts=True)[1].tolist() != [32, 32]):
        raise ValueError('The saved two-class64-atom crystal differs.')
    fitted = restore_site_energy_model(documents['model_coefficients'],
        crystal.supercell.scaled_positions, crystal.supercell.lattice, classes,
        fit_report=saved['fit_report'], geometry=saved['fit_geometry'],
        deadline_monotonic=deadline, **{key: FIT[key] for key in
            ('neighbors', 'max_atoms', 'max_bytes', 'identity_rtol', 'identity_atol')})
    material = dict(saved['material'], fresh_harmonic_geometry=plain(harmonic),
                    raw_fc3_loaded=False, fitting_repeated=False, saved_coefficients_reused_exactly=True)
    return fitted, geometry, crystal, material


def _close_seed(seed):
    for spectrum in (seed.sigma_input, seed.mapped_sigma):
        for name in ('retarded', 'statistical', 'spectral'):
            mmap = getattr(getattr(spectrum, name), '_mmap', None)
            if mmap is not None:
                mmap.close()


def run(profile, result, emit, checkpoint_directory, *, profile_sha256):
    config = validate_profile(profile)
    _material._digest(profile_sha256)
    checkpoint_directory = Path(checkpoint_directory)
    checkpoint_directory.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(checkpoint_directory.parent).free
    if free < profile['min_free_bytes']:
        raise ValueError('Insufficient free checkpoint storage for16 bounded fresh attempts.')
    started = time.monotonic()
    deadline = started+config.max_seconds-config.time_reserve_seconds
    result.update(schema='negfpy.site-energy-convergence.v1', passed=False, converged=False,
        source_commit=profile['source_commit'], source_tree=profile['source_tree'],
        profile_sha256=profile_sha256, settings=plain(profile['map']), scba_config=plain(config),
        fit_config=plain(profile['fit']), acceleration=plain(ACCELERATION),
        selection_policy=plain(SELECTION), proposal_feasibility=plain(FEASIBILITY),
        seed=plain(SEED), attempts=[], evaluated_records=[], checkpoints=[], decisions=[],
        proposal_screens=[], completed_evaluations=0, failure_reason=None,
        free_storage_at_admission=free,
        scope={'scba': True, 'initial_state': INITIAL_STATE, 'trajectory_origin': 'fresh-zero',
            'support_model': MODEL, 'explicit_approximate_research_surrogate': True,
            'surrogate_includes_raw_fit_remainder': False, 'bulk_fit_distortion_reported': True,
            'raw_ifc_modified': False, 'source_asr_repaired': False,
            'complete_center_energies_selected': True, 'all_outer_derivatives_retained': True,
            'derivative_crop': False, 'full216_green_and_contact_currents': True,
            'interaction_exact_zero_outside_active_endpoint_union': True,
            'exact_saved_fitted_state2_seed': True, 'seed_input_and_unevaluated_image_distinct': True,
            'controller_history_reset_to_seed_pair_only': True, 'fitting_repeated': False,
            'cached_green_or_born_used': False, 'physical_projection': False,
            'representation': 'retarded-statistical-spectral-v1',
            'original_guards_and_six_convergence_screens_unchanged': True,
            'unique_physical_interface_claimed': False, 'production_ifc_adoption': False,
            'accepted_restart': False, 'accepted_field_update': False,
            'accepted_transport': False, 'frequency_or_q_convergence': False})
    acceleration = ChannelAccelerationConfig(**ACCELERATION)
    selection = ChannelSelectionPolicy(**SELECTION)
    feasibility = ChannelProposalFeasibility(**FEASIBILITY)
    with ExitStack() as stack:
        pins, documents = authenticate_inputs(profile, stack)
        result['input_pins'] = pins
        emit('site_convergence_inputs_authenticated', {'count': len(pins)})
        mesh, grid, dynamical, contacts, static_arrays, static_auth = load_static_operator(profile, documents, stack)
        seed = None
        try:
            fitted, geometry, crystal, result['material'] = load_seed_model(profile, documents, deadline)
            result['fit_report'], result['fit_geometry'] = plain(fitted.fit_report), plain(fitted.geometry)
            result['model_parameters'] = plain(documents['model_descriptor'])
            scale = SolverScale(omega0=_material.OMEGA0_RAD_S, hbar=_material.HBAR_J_S)
            supported = build_site_energy_vertex(mesh, fitted, crystal, geometry,
                n_device_layers=9, interaction_layers=(2, 3, 4, 5, 6), scale=scale,
                force_constant_units=result['material']['force_constant_units'],
                mass_units=result['material']['mass_units'], crop_derivatives=False,
                deadline_monotonic=deadline, **SITE_SUPPORT)
            vertex, active = supported.vertex, supported.vertex.active_dofs
            result['support_model'] = plain(supported.metadata)
            if (vertex.full_ndof != 216 or vertex.ndof != 132
                    or active.tolist() != SEED['binding']['feedback_dofs']
                    or any(supported.metadata[key] != SEED['model_parameters'][key] for key in
                        ('model_identity_sha256', 'energy_identity_sha256'))):
                raise ValueError('The rebuilt complete energy or exact endpoint addresses differ from the seed.')
            emit('site_convergence_model_restored', {
                'model_identity_sha256': supported.metadata['model_identity_sha256'],
                'energy_identity_sha256': supported.metadata['energy_identity_sha256'],
                'fitting_repeated': False, 'active_dofs': vertex.ndof, 'full_ndof': vertex.full_ndof})
            seed = load_channel_checkpoint(SEED['path'], expected_manifest_sha256=SEED['manifest_sha256'],
                expected_binding=SEED['binding'], max_total_bytes=CHECKPOINT_MAX_BYTES)
            expected_shape = (mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
            if any(getattr(spectrum, name).shape != expected_shape for spectrum in
                    (seed.sigma_input, seed.mapped_sigma) for name in ('retarded', 'statistical', 'spectral')):
                raise ValueError('Seed native channels differ from the rebuilt model grid/support.')
            emit('site_convergence_seed_authenticated', {'iteration': seed.record['index'],
                'input_evaluated': True, 'mapped_image_dyson_evaluated': False})
            plan = plan_channel_accel(mesh.n_qpoints, grid.n_points, 216, vertex.ndof,
                len(contacts[0].device_indices), len(contacts[1].device_indices), config=config,
                acceleration=acceleration, feasibility=feasibility)
            result['iteration_plan'] = plain(plan)
            if plan.peak_owned_bytes > config.max_bytes:
                raise ValueError('The continuation allocation plan exceeds max_bytes.')
            binding = dict(SEED['binding'], source_commit=profile['source_commit'],
                source_tree=profile['source_tree'], profile_sha256=profile_sha256,
                initial_state='fresh-zero', resume_parent_iteration=2,
                resume_parent_manifest_sha256=SEED['manifest_sha256'],
                resume_parent_binding=plain(SEED['binding']), acceleration=plain(ACCELERATION),
                selection_policy=plain(SELECTION), proposal_feasibility=plain(FEASIBILITY),
                continuation={'max_fresh_attempts': 16, 'first_index': 3,
                    'controller_history': 'seed-input-and-unevaluated-image-only',
                    'initial_state': INITIAL_STATE, 'same_exact_energy_parameters': True})

            def on_proposal(index, batch):
                result['proposal_screens'].append(plain(batch))
                emit('proposal_screened', plain(batch))

            def on_start(index, sigma_input, proposal):
                attempt = {'index': index, 'input_sigma_is_zero': sigma_input is None,
                    'dyson_evaluated': False, 'proposal': plain(proposal)}
                result['attempts'].append(attempt)
                emit('scba_evaluation_start', attempt)

            def on_evaluation(state, decision):
                record = plain(state.record)
                record.update(currents=plain(state.evaluation.input_currents),
                    input_sigma_is_zero=state.sigma_input is None, sigma_input_evaluated=True,
                    mapped_sigma_dyson_evaluated=False,
                    status='converged' if record['converged'] else 'evaluated',
                    checks=plain(state.evaluation.checks), controller_decision=plain(decision),
                    zero_frequency_support={
                        'input': zero_support_context(state.sigma_input, active, grid.n_points//2),
                        'mapped': zero_support_context(state.evaluation.map_sigma, active, grid.n_points//2)})
                result['evaluated_records'].append(record)
                result['completed_evaluations'] = len(result['evaluated_records'])
                result['decisions'].append(plain(decision))
                emit('scba_evaluation_complete', record)
                saved = save_channel_checkpoint(checkpoint_directory, iteration=state.index,
                    sigma_input=state.sigma_input, mapped_sigma=state.evaluation.map_sigma,
                    record=record, binding=binding, max_bytes=profile['checkpoint_max_bytes'],
                    max_checkpoints=config.max_evaluations)
                receipt = {key: saved[key] for key in ('path', 'manifest_sha256')}
                receipt['index'] = state.index
                result['checkpoints'].append(receipt)
                emit('checkpoint_saved', receipt)
                emit('controller_decision', plain(decision))

            emit('site_energy_convergence_start', {'max_evaluations': 16, 'first_index': 3,
                'mixing': .05, 'initial_state': INITIAL_STATE,
                'feedback_ndof': 132, 'full_device_dofs': 216})
            effective_config = replace(config, max_seconds=config.max_seconds-(time.monotonic()-started))
            result['effective_scba_config'] = plain(effective_config)
            solved = run_channel_accel(grid, vertex, dynamical, *contacts,
                seed_input=seed.sigma_input, seed_mapped=seed.mapped_sigma, seed_index=2,
                scale=scale, g=profile['map']['g'], config=effective_config,
                acceleration=acceleration, selection=selection, feasibility=feasibility,
                on_start=on_start, on_evaluation=on_evaluation, on_proposal=on_proposal)
            result.update(converged=solved.converged, stop_reason=solved.stop_reason,
                failure_reason=solved.failure_reason, history=plain(solved.history),
                completed_evaluations=len(solved.history),
                selected_anchor_index=solved.state.index if solved.state is not None else None,
                last_evaluated_trial_index=solved.last_evaluated_state.index
                    if solved.last_evaluated_state is not None else None,
                selected_anchor_count=sum(row['accepted_as_anchor'] for row in solved.decisions),
                rejected_trial_count=sum(not row['accepted_as_anchor'] for row in solved.decisions),
                passed=solved.failure_reason is None and solved.state is not None)
            for key, state in (('selected_anchor', solved.state), ('last_evaluated_trial', solved.last_evaluated_state)):
                if state is not None:
                    result[key] = dict(plain(state.record),
                        currents=plain(state.evaluation.input_currents),
                        checkpoint=next((row for row in result['checkpoints'] if row['index'] == state.index), None))
            _material.verify_unchanged(pins+list(static_auth.values()))
            emit('scba_stopped', {key: result[key] for key in ('converged', 'completed_evaluations',
                'stop_reason', 'failure_reason', 'selected_anchor_index', 'last_evaluated_trial_index')})
        finally:
            if seed is not None:
                _close_seed(seed)
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
    except Exception as exc:  # noqa: BLE001 - preserve the actual failed proposal/map/checkpoint tail
        result.update(passed=False, failure_reason=type(exc).__name__+': '+str(exc))
        traceback.print_exc()
        emit('site_energy_convergence_failed', {'reason': result['failure_reason']})
    result.update(elapsed_seconds=time.monotonic()-started,
                  maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    with args.output.open('x') as stream:
        json.dump(plain(result), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    emit('site_energy_convergence_closed', {'passed': result['passed'], 'converged': result['converged'],
                                           'completed_evaluations': result.get('completed_evaluations', 0)})
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
