"""Lossless image assembly from qualified saved complete-site Hessians.

This is data preparation: no potential evaluation, differentiation, fitting,
contact, Green function or SCBA map. The original accumulation order and exact
relative-vector translations are retained. Cubic coordinate permutations are
checked against the same original archive before its model identity is reused.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from time import monotonic
import h5py
import numpy as np
from .io._saved_hdf import require_internal_storage

ALGORITHM = 'exact-relative-vector-translation-native-order-sum-v1'
_CHECKS = {'complete_site_derivatives','selected_force_support_and_translation',
    'representative_translation','raw_asr_and_permutation','forward_fold_64',
    'independent_nep_ifcs_64','sparse_512_directional','image_resolved_fc2','no_basis_fit'}


def _check(deadline):
    if monotonic() >= deadline: raise TimeoutError('Saved site-Hessian transform reached its shared deadline')


def authenticated_json(entry):
    if (not isinstance(entry,dict) or set(entry)!={'sha256','json_text'}
            or not isinstance(entry['json_text'],str)
            or hashlib.sha256(entry['json_text'].encode()).hexdigest()!=entry['sha256']):
        raise ValueError('Original transform proof bytes differ from their SHA256')
    def pairs(rows):
        result={}
        for k,v in rows:
            if k in result: raise ValueError('Duplicate transform proof JSON key')
            result[k]=v
        return result
    def reject(v): raise ValueError('Nonfinite transform proof JSON')
    result=json.loads(entry['json_text'],object_pairs_hook=pairs,parse_constant=reject)
    if not isinstance(result,dict): raise ValueError('Transform proof must be a JSON object')
    return result


def geometry_for_source(site,transform):
    """Check all source stars against explicit endpoint labels without tensors."""
    if (transform.get('schema')!='negfpp.original-site-hessian-transform.v1'
            or transform.get('algorithm')!=ALGORITHM):
        raise ValueError('Unsupported original site-Hessian transformation')
    source=authenticated_json(transform['source_geometry'])
    qualification=authenticated_json(transform['source_qualification'])
    rawpin=transform['source_archive']
    potential=site.metadata.get('potential_sha256')
    reuse=site.metadata.get('reused_qualified_silicon_source',{})
    if (qualification.get('schema')!='negfpy.nep3.material-consistency-qualification.v1'
            or qualification.get('complete') is not True or qualification.get('passed') is not True
            or qualification.get('potential_sha256')!=potential
            or qualification.get('site_derivatives_sha256')!=rawpin['sha256']
            or set(qualification.get('checks',{}))!=_CHECKS
            or any(v is not True for v in qualification['checks'].values())
            or reuse.get('source_sha256')!=rawpin['sha256']
            or reuse.get('qualification_sha256')!=transform['source_qualification']['sha256']):
        raise ValueError('Original complete-site source qualification or mapped archive bindings differ')
    if source.get('schema')!='negfpy.nep.geometry-admission.v1' or source.get('geometry_only') is not True:
        raise ValueError('Original geometry admission schema differs')
    basis=np.asarray(source['basis_eighths'])
    nb=len(site.symbols)
    a=source['lattice_constant_angstrom']
    if (basis.shape!=(nb,3) or basis.dtype.kind not in 'iu'
            or not np.allclose(np.asarray(site.fractional_positions)*8,basis,rtol=0,atol=1e-12)
            or not np.allclose(site.cell_rows_angstrom,np.eye(3)*a,rtol=0,atol=1e-12)):
        raise ValueError('Original source lattice or ordered basis differs from complete cubic data')
    classes={};reference={}
    for row in source['center_classes']:
        rep=row['representative_basis']
        if rep in reference: raise ValueError('Source geometry repeats a representative')
        reference[rep]=row['sites']
        for member in row['member_basis_indices']:
            if member in classes: raise ValueError('Source geometry repeats a center class member')
            classes[member]=rep
    if set(classes)!=set(range(nb)) or set(reference)!=set(site.tensor_shapes):
        raise ValueError('Source geometry classes differ from complete cubic representatives')
    stars=[];old_to_source=[];old_to_new=[]
    for star in site.stars:
        center=star.center_basis
        def relative(i):
            return tuple(int(basis[int(star.basis_indices[i]),k]+8*star.cell_offsets[i,k]-basis[center,k]) for k in range(3))
        order=[0]+sorted(range(1,len(star.basis_indices)),key=lambda i:(sum(v*v for v in relative(i)),int(star.basis_indices[i]),tuple(int(x) for x in star.cell_offsets[i])))
        old=[{'basis_index':int(star.basis_indices[i]),'cell_offset':[int(x) for x in star.cell_offsets[i]],'relative_eighths':list(relative(i))} for i in order]
        rep=classes[center]
        lookup={tuple(x['relative_eighths']):i for i,x in enumerate(reference[rep])}
        if len(lookup)!=len(old) or set(lookup)!={tuple(x['relative_eighths']) for x in old}:
            raise ValueError('Complete source/cubic star relative-vector sets differ')
        if center==rep and [{k:x[k] for k in ('basis_index','cell_offset','relative_eighths')} for x in reference[rep]]!=old:
            raise ValueError('Original representative tensor order differs from its explicit geometry')
        stars.append(old)
        old_to_source.append([lookup[tuple(x['relative_eighths'])] for x in old])
        old_to_new.append(order)
    return {'source':source,'qualification':qualification,'stars':stars,'classes':classes,
        'old_to_source':old_to_source,'old_to_new':old_to_new}


def assemble_original_hessian(source_path,site,transform,*,deadline,max_bytes):
    """Reproduce exact representative expansion, coherent sum and image packing."""
    _check(deadline);geo=geometry_for_source(site,transform)
    source_path=Path(source_path)
    if source_path.stat().st_size>max_bytes: raise MemoryError('Original derivative archive exceeds model_max_bytes')
    reps={}
    with h5py.File(source_path,'r') as f:
        require_internal_storage(f)
        if f.attrs.get('schema')!='negfpy.nep3.direct-site-derivatives.v1' or not bool(f.attrs.get('complete',False)):
            raise ValueError('Original derivative archive publication differs')
        if f.attrs.get('potential_sha256')!=site.metadata['potential_sha256']:
            raise ValueError('Original derivative archive potential differs')
        for rep in sorted(site.tensor_shapes):
            _check(deadline)
            ds=f[f'representative_{rep}/fc2_richardson'];d=site.tensor_shapes[rep][0]
            if ds.shape!=(d,d) or ds.dtype!=np.dtype('float64') or ds.attrs.get('units')!='eV/angstrom^2':
                raise ValueError('Original site Hessian shape/dtype/units differ')
            reps[rep]=ds[:]
            if not np.isfinite(reps[rep]).all(): raise ValueError('Original site Hessian is nonfinite')
    coherent={}
    for center,star in enumerate(geo['stars']):
        _check(deadline)
        permutation=np.asarray([3*i+axis for i in geo['old_to_source'][center] for axis in range(3)])
        matrix=reps[geo['classes'][center]][np.ix_(permutation,permutation)]
        for i,first in enumerate(star):
            for j,second in enumerate(star):
                image=tuple(second['cell_offset'][k]-first['cell_offset'][k] for k in range(3))
                key=(first['basis_index'],second['basis_index'],*image)
                if key not in coherent: coherent[key]=np.zeros((3,3))
                coherent[key]+=matrix[3*i:3*i+3,3*j:3*j+3]
    images={};nb=len(site.symbols)
    for key,block in sorted(coherent.items()):
        bi,bj,*image=key;image=tuple(image)
        if image not in images: images[image]=np.zeros((3*nb,3*nb))
        images[image][3*bi:3*bi+3,3*bj:3*bj+3]+=block
    _check(deadline)
    return dict(sorted(images.items())),geo


def verify_original_cubic(source_path,site,transform,tensors,*,deadline):
    """Verify lossless all-axis coordinate permutation of original cubic bytes."""
    geo=geometry_for_source(site,transform)
    with h5py.File(source_path,'r') as f:
        require_internal_storage(f)
        for rep,shape in site.tensor_shapes.items():
            _check(deadline)
            ds=f[f'representative_{rep}/fc3_richardson']
            if ds.shape!=tuple(shape) or ds.dtype!=np.dtype('float64') or ds.attrs.get('units')!='eV/angstrom^3':
                raise ValueError('Original cubic shape/dtype/units differ')
            old=ds[:]
            inverse=np.argsort(geo['old_to_new'][rep])
            coords=np.asarray([3*int(i)+axis for i in inverse for axis in range(3)])
            if not np.array_equal(old[np.ix_(coords,coords,coords)],tensors[rep]):
                raise ValueError('Mapped cubic archive is not a lossless permutation of original source')
            del old
    _check(deadline)


def native_identities(geo,site,transform,*,cells,centers):
    """Compute original physical identities from validated structural fields."""
    selected=[[cell,basis,[[cell+s['cell_offset'][0],s['basis_index'],*s['cell_offset'][1:]] for s in star]] for cell in sorted(centers) for basis,star in enumerate(geo['stars'])]
    if any(not 0<=p[0]<cells for row in selected for p in row[2]):
        raise ValueError('Original complete source endpoint leaves the finite device')
    active=sorted({3*(p[0]*len(site.symbols)+p[1])+axis for row in selected for p in row[2] for axis in range(3)})
    source=geo['source']
    geometry=dict(domain='negfpy.supplied-nep3-silicon-geometry.v1',lattice_constant_angstrom=source['lattice_constant_angstrom'],basis_eighths=source['basis_eighths'],site_cutoff_angstrom=source['site_radial_and_angular_cutoff_angstrom'],conventional_device_cells=cells,selected_complete_center_images=selected,active_dofs=active,transverse_phase='exp(i*[p.(Tj-Ti)+(q-p).(Tk-Ti)])')
    def digest(value):return hashlib.sha256((json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()).hexdigest()
    geometry_sha=digest(geometry)
    energy=dict(domain='negfpy.supplied-nep3-complete-selected-energy.v1',potential_sha256=site.metadata['potential_sha256'],selected_site_derivatives_sha256=transform['source_archive']['sha256'],geometry_sha256=geometry_sha,selected_complete_center_images=selected,coordinate_convention='center-first(site,Cartesian);exact coherent images',local_cartesian_units='eV/angstrom^2,eV/angstrom^3',old_fitted_model_or_fields_used=False)
    return geometry_sha,digest(energy),geometry
