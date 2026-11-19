"""Independent finite-ring/Bloch checks of the generic cubic interface."""
from dataclasses import asdict
from itertools import product
import json
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import complete_site_vertex as v


def physical_tensor(n,seed):
    rng=np.random.default_rng(seed);c=rng.normal(size=3*n)
    # Cubic energy (c.u)^3/6 with c orthogonal to rigid translations.
    c[:3]=-c[3:].reshape(-1,3).sum(axis=0)
    return np.asarray(np.einsum('i,j,k->ijk',c,c,c),dtype=np.float64)


def fixture(transverse_only=True):
    x=0 if transverse_only else 1
    stars=(v.SiteStar(0,np.array([0,1,0,1]),np.array([[0,0,0],[0,0,0],[x,1,0],[-x,-1,1]]),0),
           v.SiteStar(1,np.array([1,0,1]),np.array([[0,0,0],[0,-1,0],[x,1,-1]]),1))
    shapes={s.tensor_key:(s.cartesian_dimension,)*3 for s in stars}
    material=v.SiteFC3Metadata(np.array([12.,95.95]),np.diag([2.,3.,12.]),
        np.array([[0.,0.,0.],[.3,.2,.5]]),('C','Mo'),stars,shapes)
    tensors={0:physical_tensor(4,123),1:physical_tensor(3,456)}
    return material,tensors


def build(material,tensors,mesh=(3,2),n=1,centers=(0,),**kwargs):
    args=dict(n_device_cells=n,center_layers=centers,max_bytes=64*1024**2,
        max_workspace_bytes=256*1024**2,deadline_monotonic=monotonic()+30.)
    args.update(kwargs)
    return v.build_vertex(material,tensors,mesh,**args)


def test_duplicate_safe_mixed_mass_vertex_against_independent_periodic_ring():
    material,tensors=fixture();actual=build(material,tensors)
    mesh=actual.vertex.mesh;ny,nz=mesh.shape[:2];nq=ny*nz;d=6
    # Literal real-space periodic ring: each complete atomic energy occurs
    # once per transverse cell. Several different coherent site images share
    # the same physical atom; no representative Fourier formula is used here.
    real=np.zeros((nq*d,)*3,float)
    conversion=v.EV_J/v.ANGSTROM_M**3*np.sqrt(v.HBAR_J_S)/v.OMEGA0_RAD_S**2.5
    for z,y in product(range(nz),range(ny)):
        for star in material.stars:
            labels=[];masses=[]
            for basis,image in zip(star.basis_indices,star.cell_offsets,strict=True):
                target_y=(y+int(image[1]))%ny;target_z=(z+int(image[2]))%nz
                labels.extend((target_y+ny*target_z)*d+3*int(basis)+axis for axis in range(3))
                masses.extend([material.masses_amu[int(basis)]*v.AMU_KG]*3)
            raw=tensors[star.tensor_key]
            for i,j,k in product(range(len(labels)),repeat=3):
                real[labels[i],labels[j],labels[k]]+=raw[i,j,k]*conversion/np.sqrt(masses[i]*masses[j]*masses[k])
    addresses=np.asarray(mesh.addresses);points=2*np.pi*addresses[:,:2]/np.array([ny,nz])
    cells=np.array([(y,z) for z in range(nz) for y in range(ny)])
    def wave(point):
        out=np.zeros((nq*d,d),complex)
        for cell,coord in enumerate(cells):
            out[cell*d:(cell+1)*d,:]=np.eye(d)*np.exp(1j*np.dot(point,coord))
        return out
    for q,p in product(range(nq),repeat=2):
        expected=np.einsum('ijk,ia,jb,kc->abc',real,wave(-points[q]),wave(points[p]),
                           wave(points[q]-points[p]),optimize=True)/nq
        assert actual.vertex.coefficients[q,p]==pytest.approx(expected,rel=1e-10,abs=1e-12)
    assert actual.plan.endpoint_dofs==6
    assert actual.plan.coherent_endpoint_images>actual.plan.geometric_endpoint_atoms
    assert actual.metadata['mesh_normalization_applied'] is False
    assert actual.metadata['coupling_g_applied'] is False


def test_all_q_leg_exchange_reality_and_no_hidden_coupling_or_threshold():
    material,tensors=fixture();one=build(material,tensors);mesh=one.vertex.mesh
    arrays=one.vertex.coefficients
    for q,p in product(range(mesh.n_qpoints),repeat=2):
        difference=mesh.index_from_address(tuple(int(a-b) for a,b in zip(mesh.addresses[q],mesh.addresses[p],strict=True)))
        assert arrays[q,p]==pytest.approx(arrays[mesh.negative_index(q),mesh.negative_index(p)].conj())
        assert arrays[q,p]==pytest.approx(arrays[q,difference].transpose(0,2,1))
        assert arrays[q,p]==pytest.approx(arrays[mesh.negative_index(p),mesh.negative_index(q)].transpose(1,0,2))
    two=build(material,{k:2*raw for k,raw in tensors.items()})
    assert two.vertex.coefficients==pytest.approx(2*arrays,abs=1e-12)
    tiny=build(material,{k:1e-15*raw for k,raw in tensors.items()})
    assert np.max(np.abs(tiny.vertex.coefficients))>0
    assert tiny.vertex.coefficients==pytest.approx(1e-15*arrays,rel=1e-10,abs=1e-27)


def test_longitudinal_complete_endpoints_and_multiple_centers_are_not_cropped():
    material,tensors=fixture(transverse_only=False)
    with pytest.raises(ValueError,match='outside the finite device'):
        v.plan_vertex(material,(2,1),n_device_cells=2,center_layers=(0,))
    plan=v.plan_vertex(material,(2,1),n_device_cells=5,center_layers=(1,3))
    assert plan.full_device_dofs==30 and plan.complete_energy_centers==4
    assert min(plan.active_dofs)==3 and max(plan.active_dofs)==29
    one=build(material,tensors,(2,1),n=5,centers=(1,3))
    assert one.vertex.full_ndof==30
    assert tuple(one.vertex.active_dofs)==plan.active_dofs
    assert one.metadata['endpoint_cropping'] is False


def test_geometry_only_exact_payload_plan_and_integer_products():
    material,_=fixture();plan=v.plan_vertex(material,(4,4),n_device_cells=1,center_layers=(0,))
    assert plan.coefficient_bytes==16*16**2*6**3
    assert plan.address_bytes==8*6
    assert plan.peak_numeric_bytes>2*plan.coefficient_bytes+plan.representative_input_bytes
    assert plan.physical_derivative_evaluations==0
    assert plan.scalar_phase_scatter_evaluations==16**2*(12**3+9**3)
    assert json.loads(json.dumps(asdict(plan)))['endpoint_dofs']==6


@pytest.mark.parametrize('limits,exception,match',[
    ({'max_bytes':1},MemoryError,'payload'),
    ({'max_workspace_bytes':1},MemoryError,'workspace'),
    ({'max_phase_evaluations':1},ValueError,'phase/scatter'),
    ({'max_geometry_endpoints':1},ValueError,'geometry'),
    ({'deadline_monotonic':monotonic()-1},TimeoutError,'deadline'),
])
def test_limits_reject_before_any_output_materialization(monkeypatch,limits,exception,match):
    material,tensors=fixture()
    def forbidden(*args,**kwargs):raise AssertionError('An output allocation occurred before cap admission')
    monkeypatch.setattr(v.np,'zeros',forbidden)
    with pytest.raises(exception,match=match):build(material,tensors,**limits)


def test_nonsymmetric_physical_tensor_fails_canonical_exchange_without_averaging():
    material,tensors=fixture();bad={key:raw.copy() for key,raw in tensors.items()}
    bad[0][1,3,7]+=.1
    with pytest.raises(ValueError,match='leg exchange'):
        build(material,bad)


def write_translation_class(path,complete=True):
    with h5py.File(path,'w') as out:
        out.attrs.update(schema=v.SITE_FC3_SCHEMA,complete=complete,potential_sha256='generated',
                         finalmetadata_json=json.dumps({'complete_site_derivatives':True}))
        out['cell_rows_angstrom']=np.diag([2.,5.,6.])
        out['fractional_positions']=[[0,0,0],[.5,0,0]]
        out['masses_amu']=[12.,12.]
        out['physical_pbc']=[True,True,False]
        out.create_dataset('symbols',data=['C','C'],dtype=h5py.string_dtype())
        group=out.create_group('representatives/0')
        group['member_centers']=[0,1]
        group['basis_indices']=[0,1,1]
        group['cell_offsets']=[[0,0,0],[0,0,0],[-1,0,0]]
        group['relative_positions_A']=[[0.,0,0],[1.,0,0],[-1.,0,0]]
        group.create_dataset('species',data=['C','C','C'],dtype=h5py.string_dtype())
        tensor=group.create_dataset('fc3_ev_A3',data=physical_tensor(3,77))
        tensor.attrs['units']='eV/angstrom^3'


def test_hdf_metadata_reads_no_tensor_and_translates_coherent_alias_labels(tmp_path,monkeypatch):
    path=tmp_path/'site-ifc3.hdf5';write_translation_class(path)
    original=h5py.Dataset.__getitem__
    def deny_fc3_read(dataset,key):
        if dataset.name.endswith('/fc3_ev_A3'):raise AssertionError('Metadata planner read physical FC3 payload')
        return original(dataset,key)
    with monkeypatch.context() as patch:
        patch.setattr(h5py.Dataset,'__getitem__',deny_fc3_read)
        material=v.load_site_ifc3_metadata(path)
        plan=v.plan_vertex(material,(2,1),n_device_cells=3,center_layers=(1,))
    assert material.metadata['fc3_payload_read'] is False
    assert len(material.tensor_shapes)==1
    assert material.stars[1].basis_indices.tolist()==[1,0,0]
    assert material.stars[1].cell_offsets.tolist()==[[0,0,0],[1,0,0],[0,0,0]]
    tensors=v.load_site_fc3(material,max_bytes=10000,deadline_monotonic=monotonic()+5)
    assert set(tensors)=={0} and not tensors[0].flags.writeable
    result=build(material,tensors,(2,1),n=3,centers=(1,))
    assert result.plan.active_dofs==plan.active_dofs
    with pytest.raises(MemoryError,match='before loading'):
        v.load_site_fc3(material,max_bytes=1,deadline_monotonic=monotonic()+5)
    with h5py.File(path,'r+') as out:out.attrs['complete']=False
    with pytest.raises(ValueError,match='complete'):
        v.load_site_ifc3_metadata(path)


def test_invalid_repeated_classes_and_changed_translation_are_rejected(tmp_path):
    path=tmp_path/'site-ifc3.hdf5';write_translation_class(path)
    with h5py.File(path,'r+') as out:out['representatives/0/member_centers'][:]=[0,0]
    with pytest.raises(ValueError,match='partition'):
        v.load_site_ifc3_metadata(path)
    path2=tmp_path/'wrong-translation.hdf5';write_translation_class(path2)
    with h5py.File(path2,'r+') as out:out['representatives/0/relative_positions_A'][1,0]=1.01
    with pytest.raises(ValueError,match='Cartesian'):
        v.load_site_ifc3_metadata(path2)


def test_current_producer_metadata_field_and_legacy_fallback_are_preserved(tmp_path):
    path=tmp_path/'site-ifc3.hdf5';write_translation_class(path)
    legacy=v.load_site_ifc3_metadata(path)
    assert legacy.metadata['complete_site_derivatives'] is True
    with h5py.File(path,'r+') as out:
        out.attrs['metadata_json']=json.dumps({'published_producer_field':True,'scientific_guards_passed':True})
    current=v.load_site_ifc3_metadata(path)
    assert current.metadata['published_producer_field'] is True
    assert current.metadata['scientific_guards_passed'] is True
    with h5py.File(path,'r+') as out:
        out['representatives/0/cell_offsets'][1,2]=1
        out['representatives/0/relative_positions_A'][1,2]=6
    with pytest.raises(ValueError,match='nonperiodic'):
        v.load_site_ifc3_metadata(path)
