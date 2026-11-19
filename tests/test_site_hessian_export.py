"""Independent spring-energy controls of lossless original-archive preparation."""
from copy import deepcopy
import hashlib
import json
from time import monotonic
import h5py
import numpy as np
import pytest
from negfpy.anharmonic.complete_site_vertex import SiteFC3Metadata,SiteStar
from negfpy.anharmonic.site_hessian_export import (
    ALGORITHM,_CHECKS,geometry_for_source,assemble_original_hessian,
    verify_original_cubic,native_identities,
)


def proof(value):
    text=json.dumps(value,sort_keys=True)+'\n'
    return {'sha256':hashlib.sha256(text.encode()).hexdigest(),'json_text':text}


def fixture(tmp_path):
    potential='a'*64
    path=tmp_path/'original.h5'
    vector=np.array([-1.,0,0,1.,0,0])
    cubic=np.einsum('i,j,k->ijk',vector,vector,vector)
    hessian=np.kron([[1.,-1.],[-1.,1.]],np.eye(3))
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='negfpy.nep3.direct-site-derivatives.v1',complete=True,potential_sha256=potential)
        g=f.create_group('representative_0')
        g.create_dataset('fc2_richardson',data=hessian).attrs['units']='eV/angstrom^2'
        g.create_dataset('fc3_richardson',data=cubic).attrs['units']='eV/angstrom^3'
    pin={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    qualification=proof({'schema':'negfpy.nep3.material-consistency-qualification.v1','complete':True,'passed':True,'potential_sha256':potential,'site_derivatives_sha256':pin['sha256'],'checks':{k:True for k in _CHECKS},'generated_analytic_fixture':True})
    geometry=proof({'schema':'negfpy.nep.geometry-admission.v1','geometry_only':True,
        'basis_eighths':[[0,0,0]],'lattice_constant_angstrom':2.,'site_radial_and_angular_cutoff_angstrom':2.1,
        'center_classes':[{'representative_basis':0,'member_basis_indices':[0],'sites':[
            {'basis_index':0,'cell_offset':[0,0,0],'relative_eighths':[0,0,0]},
            {'basis_index':0,'cell_offset':[1,0,0],'relative_eighths':[8,0,0]}]}]})
    transform={'schema':'negfpp.original-site-hessian-transform.v1','algorithm':ALGORITHM,
        'source_archive':pin,'qualified_reference_ifc2':{'sha256':'b'*64,'bytes':1},
        'source_qualification':qualification,'source_geometry':geometry}
    site=SiteFC3Metadata(np.array([12.]),np.eye(3)*2,np.array([[0.,0.,0.]]),('C',),
        (SiteStar(0,np.array([0,0]),np.array([[0,0,0],[1,0,0]]),0),),{0:(6,6,6)},
        metadata={'potential_sha256':potential,'reused_qualified_silicon_source':{
            'source_sha256':pin['sha256'],'qualification_sha256':qualification['sha256']}})
    return path,site,transform,{0:cubic}


def test_exact_hessian_image_assembly_matches_independent_pair_spring(tmp_path):
    path,site,transform,_=fixture(tmp_path)
    images,geo=assemble_original_hessian(path,site,transform,deadline=monotonic()+10,max_bytes=1024**2)
    assert list(images)==[(-1,0,0),(0,0,0),(1,0,0)]
    # Literal d² [sum_axis (u_(n+1)-u_n)^2/2] / du_i du_j.
    np.testing.assert_array_equal(images[-1,0,0],-np.eye(3))
    np.testing.assert_array_equal(images[0,0,0],2*np.eye(3))
    np.testing.assert_array_equal(images[1,0,0],-np.eye(3))
    g,e,descriptor=native_identities(geo,site,transform,cells=3,centers=(1,))
    assert descriptor['active_dofs']==list(range(3,9))
    assert len(g)==len(e)==64
    assert descriptor['selected_complete_center_images']==[[1,0,[[1,0,0,0],[2,0,0,0]]]]


def test_original_cubic_permutation_is_exact_and_corruption_fails(tmp_path):
    path,site,transform,tensors=fixture(tmp_path)
    verify_original_cubic(path,site,transform,tensors,deadline=monotonic()+10)
    tensors[0]=tensors[0].copy();tensors[0][0,0,0]+=1e-15
    with pytest.raises(ValueError,match='lossless permutation'):
        verify_original_cubic(path,site,transform,tensors,deadline=monotonic()+10)


def test_new_endpoint_order_is_checked_by_inverse_permutation(tmp_path):
    path,site,transform,tensors=fixture(tmp_path)
    # The source identity requires the center first, but a larger star may
    # reorder any outer endpoints. Add a negative endpoint and exercise all
    # three Cartesian tensor axes independently with labelled synthetic data.
    with h5py.File(path,'r+') as f:
        del f['representative_0/fc3_richardson']
        values=np.arange(9**3,dtype=float).reshape(9,9,9)
        f.create_dataset('representative_0/fc3_richardson',data=values).attrs['units']='eV/angstrom^3'
    geom=json.loads(transform['source_geometry']['json_text'])
    geom['center_classes'][0]['sites'].insert(1,{'basis_index':0,'cell_offset':[-1,0,0],'relative_eighths':[-8,0,0]})
    transform['source_geometry']=proof(geom)
    star=SiteStar(0,np.array([0,0,0]),np.array([[0,0,0],[1,0,0],[-1,0,0]]),0)
    site=SiteFC3Metadata(site.masses_amu,site.cell_rows_angstrom,site.fractional_positions,site.symbols,(star,),{0:(9,9,9)},metadata=site.metadata)
    coords=np.array([0,1,2,6,7,8,3,4,5])
    verify_original_cubic(path,site,transform,{0:values[np.ix_(coords,coords,coords)]},deadline=monotonic()+10)


def test_qualification_and_geometry_hash_fail_closed_before_hessian(tmp_path,monkeypatch):
    path,site,transform,_=fixture(tmp_path)
    bad=deepcopy(transform);bad['source_geometry']['json_text']+=' '
    with pytest.raises(ValueError,match='proof bytes'):
        geometry_for_source(site,bad)
    bad=deepcopy(transform)
    document=json.loads(bad['source_qualification']['json_text']);document['checks']['no_basis_fit']=False
    bad['source_qualification']=proof(document)
    with pytest.raises(ValueError,match='qualification'):
        geometry_for_source(site,bad)


def test_original_representative_geometry_order_cannot_be_relabelled(tmp_path):
    _,site,transform,_=fixture(tmp_path)
    document=json.loads(transform['source_geometry']['json_text'])
    document['center_classes'][0]['sites'][1]['cell_offset']=[-1,0,0]
    transform['source_geometry']=proof(document)
    with pytest.raises(ValueError,match='tensor order'):
        geometry_for_source(site,transform)
