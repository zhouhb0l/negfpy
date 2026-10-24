"""Small independent contraction and actual-support mapping checks."""
from __future__ import annotations

import time

import numpy as np
import pytest
from test_anharmonic_transverse_material_streaming import _inputs

from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_crossing_support import (
    build_crossing_support_vertex,
    contract_rectangular_bins,
)
from negfpy.anharmonic.transverse_material_packed import fourier_material_cubic_packed
from negfpy.anharmonic.transverse_packed import contract_packed_fft_bins
from negfpy.anharmonic.types import SolverScale

_SCALE = SolverScale(omega0=2.,hbar=3.)


def _complex(rng,shape):
    return rng.normal(size=shape)+1j*rng.normal(size=shape)


@pytest.mark.parametrize("workers",[1,3])
def test_rectangular_complex_kernel_matches_explicit_einsum(workers):
    rng = np.random.default_rng(713)
    v,x,y = (_complex(rng,shape) for shape in ((2,3,3),(5,3,3),(5,3,3)))
    expected = np.einsum("acd,bce,bdf,zef->baz",v,x,y,v.conj(),optimize=False)
    actual = contract_rectangular_bins(v,x,y,workers=workers,external_tile=1,bin_tile=2)
    np.testing.assert_allclose(actual,expected,rtol=1e-13,atol=1e-13)


def test_square_case_matches_original_kernel_and_zero_embedded_crop():
    rng = np.random.default_rng(91)
    v,x,y = (_complex(rng,shape) for shape in ((2,2,2),(3,4,4),(3,4,4)))
    full = np.zeros((2,4,4),dtype=np.complex128)
    active = np.array([1,3])
    full[:,active[:,None],active] = v
    actual = contract_rectangular_bins(full,x,y,external_tile=1)
    crop = contract_packed_fft_bins(v,x[:,active[:,None],active],y[:,active[:,None],active],external_tile=1)
    np.testing.assert_allclose(actual,crop,rtol=1e-13,atol=1e-13)
    small = contract_rectangular_bins(v,x[:,:2,:2],y[:,:2,:2])
    old = contract_packed_fft_bins(v,x[:,:2,:2],y[:,:2,:2])
    np.testing.assert_array_equal(small,old)


@pytest.mark.parametrize("complex_source",[False,True])
def test_actual_crossing_support_matches_independent_full_production_vertex(complex_source):
    transform = np.array([[2,1,0],[0,1,0],[0,0,1]]) if complex_source else np.eye(3,dtype=int)
    geometry,source = _inputs(transform,r0=complex_source,duplicate_images=complex_source,mixed_cartesian=complex_source)
    mesh = UniformBZMesh((4,1,1))
    crop = fourier_material_cubic_packed(mesh,source,geometry,n_device_layers=4,interaction_layers=(1,2),scale=_SCALE).vertex
    full = fourier_material_cubic_packed(mesh,source,geometry,n_device_layers=4,interaction_layers=(0,1,2,3),scale=_SCALE).vertex
    actual = build_crossing_support_vertex(mesh,source,geometry,n_device_layers=4,
        interaction_layers=(1,2),scale=_SCALE,reference_vertex=crop)
    index = {int(dof):i for i,dof in enumerate(full.active_dofs)}
    expected = np.zeros_like(actual.coefficients)
    external = [index[int(dof)] for dof in crop.active_dofs]
    for p in range(mesh.n_qpoints):
        expected[p][:,full.active_dofs[:,None],full.active_dofs] = full.coefficients[0,p,external]
    np.testing.assert_allclose(actual.coefficients,expected,rtol=2e-13,atol=2e-15)
    assert all(row['passed'] for row in actual.metadata['production_crop_comparison'])
    assert actual.metadata['support']['crossing_anchor_images'] > 0
    assert actual.metadata['support']['complete_internal_support']
    assert actual.metadata['projection'] is False
    assert actual.metadata['ifc_correction'] is False
    assert actual.metadata['external_internal_full_permutation_tested'] is False
    assert max(row['frobenius'] for row in actual.metadata['internal_gamma_translation_diagnostic_only']) < 1e-14
    if complex_source:
        assert np.max(np.abs(actual.coefficients.imag)) > 1e-3


def test_missing_harmonic_domain_is_rejected_without_clipping():
    geometry,source = _inputs(np.eye(3,dtype=int))
    with pytest.raises(ValueError,match='Crossing support exceeds the harmonic device'):
        build_crossing_support_vertex(UniformBZMesh((4,1,1)),source,geometry,
            n_device_layers=3,interaction_layers=(0,1),scale=_SCALE)


@pytest.mark.parametrize('budget,match',[
    ({'max_contraction_work':1},'work'),({'max_bytes':1},'workspace'),
])
def test_rectangular_budget_failure_precedes_kernel_allocation(budget,match):
    value = np.ones((2,3,3),dtype=np.complex128)
    with pytest.raises(ValueError,match=match):
        contract_rectangular_bins(value,value,value,**budget)


def test_mapping_expired_deadline_precedes_source_work():
    geometry,source = _inputs(np.eye(3,dtype=int))
    with pytest.raises(TimeoutError,match='deadline'):
        build_crossing_support_vertex(UniformBZMesh((4,1,1)),source,geometry,
            n_device_layers=4,interaction_layers=(1,2),scale=_SCALE,
            deadline_monotonic=time.monotonic()-1)
