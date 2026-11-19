"""Complete, image-preserving cubic vertices for arbitrary NEP materials.

Local tensors are physical derivatives of individual complete atomic energies,
in center-first (site, Cartesian) order and eV/Angstrom³. Every coherent image
is phased before duplicate physical addresses are summed. No fitting, magnitude
threshold, cropping, coupling g, coverage division or mesh weight is applied.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from math import prod
from pathlib import Path
from time import monotonic
from typing import Mapping

import numpy as np

SITE_FC3_SCHEMA = "negfpy.materialTest.complete-site-ifc3.v1"
EV_J = 1.602176634e-19
AMU_KG = 1.66053906660e-27
ANGSTROM_M = 1e-10
HBAR_J_S = 1.054571817e-34
OMEGA0_RAD_S = 2*np.pi*1e13


def _integer(value,name,minimum=1):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _deadline(deadline,reserve=0.):
    if isinstance(deadline,(bool,np.bool_)) or not isinstance(deadline,(float,int,np.floating)) or not np.isfinite(deadline):
        raise ValueError("A finite explicit deadline_monotonic is required")
    if monotonic()+reserve>=deadline:
        raise TimeoutError("Complete-site cubic mapping reached its shared deadline/reserve")


def _readonly(value,dtype=None):
    out=np.array(value,dtype=dtype,copy=True);out.setflags(write=False);return out


def _text(value):
    return value.decode('utf-8') if isinstance(value,bytes) else str(value)


@dataclass(frozen=True)
class SiteStar:
    center_basis: int
    basis_indices: np.ndarray
    cell_offsets: np.ndarray
    tensor_key: int

    @property
    def cartesian_dimension(self):
        return 3*len(self.basis_indices)


@dataclass(frozen=True)
class SiteFC3Metadata:
    masses_amu: np.ndarray
    cell_rows_angstrom: np.ndarray
    fractional_positions: np.ndarray
    symbols: tuple[str,...]
    stars: tuple[SiteStar,...]
    tensor_shapes: Mapping[int,tuple[int,int,int]]
    source_path: str = ""
    metadata: dict = field(default_factory=dict)


def _validate_metadata(material):
    if not isinstance(material,SiteFC3Metadata):
        raise TypeError("material must be SiteFC3Metadata")
    masses=np.asarray(material.masses_amu)
    nb=len(masses) if masses.ndim==1 else 0
    if not nb or not np.isfinite(masses).all() or np.any(masses<=0):
        raise ValueError("One finite positive mass is required per basis atom")
    cell=np.asarray(material.cell_rows_angstrom);frac=np.asarray(material.fractional_positions)
    if (cell.shape!=(3,3) or frac.shape!=(nb,3) or not np.isfinite(cell).all()
        or not np.isfinite(frac).all() or abs(np.linalg.det(cell))<1e-12
        or len(material.symbols)!=nb):
        raise ValueError("Complete crystal geometry must match the mass basis")
    stars={}
    for star in material.stars:
        if not isinstance(star,SiteStar):raise TypeError("Each star must be SiteStar")
        center=_integer(star.center_basis,'center_basis',0)
        ids=np.asarray(star.basis_indices);images=np.asarray(star.cell_offsets)
        if (center>=nb or center in stars or ids.ndim!=1 or not len(ids)
            or ids.dtype.kind not in 'iu' or np.any(ids<0) or np.any(ids>=nb)
            or images.shape!=(len(ids),3) or images.dtype.kind not in 'iu'
            or ids[0]!=center or np.any(images[0]!=0)
            or len(set(zip(ids.tolist(),map(tuple,images.tolist()))))!=len(ids)):
            raise ValueError("Require unique complete center-first coherent star labels")
        if tuple(material.tensor_shapes.get(star.tensor_key,()))!=(star.cartesian_dimension,)*3:
            raise ValueError("Declared representative FC3 shape differs from the complete site star")
        stars[center]=star
    if set(stars)!=set(range(nb)):
        raise ValueError("Exactly one complete energy star per basis atom is required")
    return stars


def load_site_ifc3_metadata(path,*,max_metadata_bytes=16*1024**2,
                           translation_tolerance=1e-8):
    """Read small geometry/shape metadata only; no FC3 payload is accessed.

    Exact translation classes store one tensor in representative star order.
    Member labels are obtained by the known center translation and a unique
    same-species integer image match, then checked in Cartesian coordinates.
    This does not infer images from a folded force-constant tensor.
    """
    import h5py
    source=Path(path).expanduser().resolve()
    cap=_integer(max_metadata_bytes,'max_metadata_bytes')
    tolerance=float(translation_tolerance)
    if not np.isfinite(tolerance) or tolerance<=0:raise ValueError("translation_tolerance must be positive")
    with h5py.File(source,'r') as handle:
        from .io._saved_hdf import require_internal_storage
        require_internal_storage(handle)
        complete=handle.attrs.get('complete',False)
        if (_text(handle.attrs.get('schema',''))!=SITE_FC3_SCHEMA
            or not isinstance(complete,(bool,np.bool_)) or not bool(complete)):
            raise ValueError("Site FC3 schema or complete publication flag is invalid")
        names=('cell_rows_angstrom','fractional_positions','symbols','masses_amu','physical_pbc')
        if any(name not in handle for name in names) or 'representatives' not in handle:
            raise ValueError("Site FC3 artifact is missing crystal or representative metadata")
        metadata_bytes=sum(int(handle[n].size*handle[n].dtype.itemsize) for n in names)
        for group in handle['representatives'].values():
            for name in ('member_centers','basis_indices','cell_offsets','relative_positions_A','species'):
                if name not in group:raise ValueError("Representative star metadata is incomplete")
                metadata_bytes+=int(group[name].size*group[name].dtype.itemsize)
        if metadata_bytes>cap:raise MemoryError("Site FC3 geometry metadata exceeds max_metadata_bytes")
        cell=np.asarray(handle['cell_rows_angstrom'][:],float)
        frac=np.asarray(handle['fractional_positions'][:],float)
        symbols=tuple(str(x) for x in handle['symbols'].asstr()[:])
        masses=np.asarray(handle['masses_amu'][:],float)
        nb=len(symbols)
        if cell.shape!=(3,3) or frac.shape!=(nb,3) or masses.shape!=(nb,):
            raise ValueError("Site FC3 crystal shapes differ")
        stars=[];shapes={};membership=set();maximum_mapping_residual=0.
        for key,group in handle['representatives'].items():
            rep=int(key)
            ids=np.asarray(group['basis_indices'][:])
            images=np.asarray(group['cell_offsets'][:])
            relative=np.asarray(group['relative_positions_A'][:],float)
            species=tuple(str(x) for x in group['species'].asstr()[:])
            members=np.asarray(group['member_centers'][:])
            if (not 0<=rep<nb or ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
                or images.shape!=(len(ids),3) or images.dtype.kind not in 'iu'
                or relative.shape!=(len(ids),3) or len(species)!=len(ids)
                or members.ndim!=1 or members.dtype.kind not in 'iu'
                or ids[0]!=rep or np.any(images[0]!=0) or not np.allclose(relative[0],0,rtol=0,atol=tolerance)):
                raise ValueError("Invalid representative coherent image metadata")
            if np.any(ids>=nb) or np.any(ids<0) or not np.isfinite(relative).all():
                raise ValueError("Representative site labels/relative positions are invalid")
            if any(species[i]!=symbols[int(b)] for i,b in enumerate(ids)):
                raise ValueError("Representative site species differs from crystal basis")
            if 'fc3_ev_A3' not in group:raise ValueError("Representative FC3 dataset is absent")
            tensor=group['fc3_ev_A3'];shape=(3*len(ids),)*3
            if tensor.shape!=shape or tensor.dtype!=np.dtype('float64') or _text(tensor.attrs.get('units',''))!='eV/angstrom^3':
                raise ValueError("Representative FC3 shape/dtype/physical units differ")
            shapes[rep]=shape
            for member in members:
                member=int(member)
                if member in membership or not 0<=member<nb or symbols[member]!=symbols[rep]:
                    raise ValueError("Translation classes must partition same-species basis centers")
                membership.add(member);translated_ids=[];translated_images=[]
                for endpoint,image,symbol,displacement in zip(ids,images,species,relative,strict=True):
                    desired=frac[int(endpoint)]+image+frac[member]-frac[rep]
                    matches=[]
                    for target,target_symbol in enumerate(symbols):
                        if target_symbol!=symbol:continue
                        delta=desired-frac[target];integer=np.rint(delta)
                        residual=float(np.max(np.abs(delta-integer)))
                        if residual<=tolerance:matches.append((target,integer.astype(np.int64),residual))
                    if len(matches)!=1:raise ValueError("A translated coherent endpoint has no unique same-species integer image")
                    target,target_image,residual=matches[0]
                    actual=(frac[target]+target_image-frac[member])@cell
                    cartesian_error=float(np.max(np.abs(actual-displacement)))
                    if cartesian_error>tolerance*(1+float(np.max(np.abs(displacement)))):
                        raise ValueError("Translation-class endpoint changes its Cartesian site order")
                    maximum_mapping_residual=max(maximum_mapping_residual,residual,cartesian_error)
                    translated_ids.append(target);translated_images.append(target_image)
                stars.append(SiteStar(member,_readonly(translated_ids,np.int64),_readonly(translated_images,np.int64),rep))
        # Current producer publishes metadata_json. Accept the earlier
        # documented finalmetadata_json spelling for preserved artifacts.
        rawmeta=handle.attrs.get('metadata_json',handle.attrs.get('finalmetadata_json','{}'))
        meta=json.loads(_text(rawmeta))
        if not isinstance(meta,dict):raise ValueError("Site FC3 metadata JSON must be an object")
        pbc=np.asarray(handle['physical_pbc'][:])
        if pbc.shape!=(3,) or pbc.dtype.kind!='b':raise ValueError("Site FC3 physical_pbc must contain three Boolean flags")
        for star in stars:
            if any(not bool(pbc[axis]) and np.any(star.cell_offsets[:,axis]!=0) for axis in range(3)):
                raise ValueError("A coherent star image crosses a declared nonperiodic axis")
        meta=dict(meta,potential_sha256=_text(handle.attrs.get('potential_sha256','')),
            physical_pbc=[bool(x) for x in pbc],
            maximum_translation_label_residual=maximum_mapping_residual,
            representative_translation_tolerance=tolerance,
            metadata_bytes_read=metadata_bytes,fc3_payload_read=False)
    material=SiteFC3Metadata(_readonly(masses),_readonly(cell),_readonly(frac),symbols,
        tuple(sorted(stars,key=lambda s:s.center_basis)),shapes,str(source),meta)
    _validate_metadata(material)
    return material


def load_site_fc3(material,*,max_bytes,deadline_monotonic):
    """Read each physical representative once, with mandatory caps/deadline."""
    import h5py
    _deadline(deadline_monotonic);_validate_metadata(material)
    cap=_integer(max_bytes,'max_bytes')
    predicted=sum(8*prod(int(v) for v in shape) for shape in material.tensor_shapes.values())
    if predicted>cap:raise MemoryError("Representative FC3 inputs exceed max_bytes before loading")
    if not material.source_path:raise ValueError("Metadata has no source HDF5 for representative loading")
    result={}
    with h5py.File(material.source_path,'r') as handle:
        from .io._saved_hdf import require_internal_storage
        require_internal_storage(handle)
        published=handle.attrs.get('complete',False)
        if (not isinstance(published,(bool,np.bool_)) or not bool(published)
            or _text(handle.attrs.get('schema',''))!=SITE_FC3_SCHEMA):
            raise ValueError("Site FC3 source changed after metadata planning")
        if _text(handle.attrs.get('potential_sha256',''))!=material.metadata.get('potential_sha256',''):
            raise ValueError("Site FC3 potential identity changed after metadata planning")
        for rep,shape in material.tensor_shapes.items():
            _deadline(deadline_monotonic)
            dataset=handle[f'representatives/{rep}/fc3_ev_A3']
            if (dataset.shape!=tuple(shape) or dataset.dtype!=np.dtype('float64')
                or _text(dataset.attrs.get('units',''))!='eV/angstrom^3'):
                raise ValueError("Site FC3 tensor dimensions changed after planning")
            tensor=np.empty(shape,dtype=np.float64)
            for plane in range(shape[0]):
                _deadline(deadline_monotonic);tensor[plane]=dataset[plane]
                if not np.isfinite(tensor[plane]).all():raise ValueError("Physical site FC3 contains nonfinite values")
            tensor.setflags(write=False);result[int(rep)]=tensor
    _deadline(deadline_monotonic)
    return result


def _mesh(value):
    from negfpy.anharmonic.momentum import UniformBZMesh
    if isinstance(value,UniformBZMesh):mesh=value
    else:
        shape=tuple(value)
        mesh=UniformBZMesh(shape+(1,) if len(shape)==2 else shape)
    if mesh.shape[2]!=1:raise ValueError("Two transverse directions require canonical mesh.shape[2]=1")
    return mesh




@dataclass(frozen=True)
class CompleteSiteVertexPlan:
    mesh_shape: tuple[int,int,int]
    n_q: int
    basis_atoms: int
    full_device_cells: int
    full_device_dofs: int
    center_layers: tuple[int,...]
    complete_energy_centers: int
    geometric_endpoint_atoms: int
    endpoint_dofs: int
    coherent_endpoint_images: int
    active_dofs: tuple[int,...]
    representative_input_bytes: int
    coefficient_bytes: int
    address_bytes: int
    canonical_owned_copy_bytes: int
    canonical_validation_workspace_bytes: int
    scatter_workspace_bytes: int
    auxiliary_bytes: int
    caller_additional_resident_bytes: int
    peak_numeric_bytes: int
    scalar_phase_scatter_evaluations: int
    geometric_endpoint_evaluations: int
    physical_derivative_evaluations: int = 0


def _embedded_stars(material,n_device_cells,center_layers):
    stars=_validate_metadata(material);nb=len(material.masses_amu)
    cells=_integer(n_device_cells,'n_device_cells')
    centers=tuple(_integer(v,'center layer',0) for v in center_layers)
    if not centers or len(set(centers))!=len(centers) or any(v>=cells for v in centers):
        raise ValueError("Supply unique in-device center layers; no layer is silently duplicated or removed")
    active=set();coherent=set();embedded=[]
    for layer in sorted(centers):
        for basis in range(nb):
            star=stars[basis];physical=[];transverse=[]
            for atom,image in zip(star.basis_indices,star.cell_offsets,strict=True):
                target_layer=layer+int(image[0]);atom=int(atom)
                if not 0<=target_layer<cells:
                    raise ValueError("A complete energy center has an endpoint outside the finite device; add harmonic buffers, never crop the star")
                dofs=tuple(3*(target_layer*nb+atom)+cart for cart in range(3))
                physical.extend(dofs);transverse.extend([tuple(int(x) for x in image[1:])]*3)
                active.update(dofs);coherent.add((target_layer,atom,int(image[1]),int(image[2])))
            embedded.append((star,tuple(physical),tuple(transverse)))
    return tuple(sorted(centers)),tuple(sorted(active)),coherent,tuple(embedded)


def plan_vertex(material,mesh,*,n_device_cells,center_layers,
                caller_additional_resident_bytes=0,max_geometry_endpoints=1_000_000):
    """Plan geometry/output before any physical FC3 tensor is accessed."""
    routed=_mesh(mesh);stars=_validate_metadata(material)
    resident=_integer(caller_additional_resident_bytes,'caller_additional_resident_bytes',0)
    cap=_integer(max_geometry_endpoints,'max_geometry_endpoints')
    selected_layers=tuple(center_layers)
    planned_endpoints=len(selected_layers)*sum(len(s.basis_indices) for s in stars.values())
    if planned_endpoints>cap:raise ValueError("Complete-star geometry work exceeds max_geometry_endpoints")
    centers,active,coherent,embedded=_embedded_stars(material,n_device_cells,selected_layers)
    nq,a=routed.n_qpoints,len(active)
    dimensions=[s.cartesian_dimension for s in stars.values()]
    inputs=sum(8*prod(int(v) for v in shape) for shape in material.tensor_shapes.values())
    coefficients=16*nq*nq*a**3;addresses=8*a
    # Canonical constructor owns a coexisting full copy, a whole-array finite
    # mask, routes, and bounded conjugation/allclose temporaries per cubic block.
    validation=(nq*nq+128)*a**3
    largest=max(dimensions)
    scatter=64*largest**2+256*largest
    auxiliary=65536+16*nq*nq+128*nq+256*len(coherent)+64*sum(len(row[1]) for row in embedded)
    peak=resident+inputs+2*coefficients+addresses+max(validation,scatter)+auxiliary
    return CompleteSiteVertexPlan(tuple(routed.shape),nq,len(material.masses_amu),int(n_device_cells),
        3*len(material.masses_amu)*int(n_device_cells),centers,len(embedded),a//3,a,len(coherent),active,
        inputs,coefficients,addresses,coefficients,validation,scatter,auxiliary,resident,peak,
        len(centers)*nq*nq*sum(d**3 for d in dimensions),planned_endpoints)


@dataclass(frozen=True)
class CompleteSiteVertexBuild:
    vertex: object
    plan: CompleteSiteVertexPlan
    metadata: dict


def build_vertex(material,site_fc3,mesh,*,n_device_cells,center_layers,
                 max_bytes,max_workspace_bytes,deadline_monotonic,
                 omega0_rad_s=OMEGA0_RAD_S,hbar_j_s=HBAR_J_S,
                 max_phase_evaluations=100_000_000,max_geometry_endpoints=1_000_000,
                 caller_additional_resident_bytes=0,constructor_reserve_seconds=1.):
    """Build the unchanged packed canonical vertex after explicit admission.

    max_bytes limits owned coefficients+addresses. max_workspace_bytes limits
    the conservative coexisting numeric plan, including caller input tensors
    and the canonical owned copy. Python objects and internal library scratch
    are excluded; this is not an RSS guarantee. Deadline checks are cooperative;
    final canonical validation is noninterruptible and uses a caller reserve.
    """
    from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
    reserve=float(constructor_reserve_seconds)
    if not np.isfinite(reserve) or reserve<0:raise ValueError("constructor_reserve_seconds must be nonnegative")
    _deadline(deadline_monotonic,reserve)
    routed=_mesh(mesh)
    plan=plan_vertex(material,routed,n_device_cells=n_device_cells,center_layers=center_layers,
        caller_additional_resident_bytes=caller_additional_resident_bytes,max_geometry_endpoints=max_geometry_endpoints)
    if plan.coefficient_bytes+plan.address_bytes>_integer(max_bytes,'max_bytes'):
        raise MemoryError("Complete vertex payload exceeds max_bytes before materialization")
    if plan.peak_numeric_bytes>_integer(max_workspace_bytes,'max_workspace_bytes'):
        raise MemoryError("Coexisting vertex/input/workspace plan exceeds max_workspace_bytes before materialization")
    if plan.scalar_phase_scatter_evaluations>_integer(max_phase_evaluations,'max_phase_evaluations'):
        raise ValueError("Complete phase/scatter work exceeds max_phase_evaluations before materialization")
    omega=float(omega0_rad_s);hbar=float(hbar_j_s)
    if not np.isfinite([omega,hbar]).all() or omega<=0 or hbar<=0:
        raise ValueError("Positive physical omega0 and hbar are mandatory")
    factor=(EV_J/ANGSTROM_M**3)*np.sqrt(hbar)/omega**2.5
    masses_kg=np.asarray(material.masses_amu)*AMU_KG
    if not np.isfinite(factor) or factor<=0 or not np.isfinite(masses_kg).all() or np.any(masses_kg<=0):
        raise ValueError("Physical FC3/mass unit conversion is not representable")
    if not isinstance(site_fc3,Mapping) or set(site_fc3)!=set(material.tensor_shapes):
        raise ValueError("Exactly the planned physical representative FC3 tensors are required")
    for key,shape in material.tensor_shapes.items():
        tensor=site_fc3[key]
        if not isinstance(tensor,np.ndarray) or tensor.dtype!=np.dtype('float64') or tensor.shape!=tuple(shape):
            raise ValueError("Physical FC3 must be float64 in the exact planned star order")
        for plane in tensor:
            _deadline(deadline_monotonic,reserve)
            if not np.isfinite(plane).all():raise ValueError("Nonfinite physical FC3 tensor")
    _,active,_,embedded=_embedded_stars(material,n_device_cells,plan.center_layers)
    lookup={dof:i for i,dof in enumerate(active)}
    qpoints=2*np.pi*np.asarray(routed.addresses,dtype=float)[:,:2]/np.asarray(routed.shape[:2],float)
    a=len(active);values=np.zeros((plan.n_q,plan.n_q,a,a,a),dtype=np.complex128)
    for star,physical,images in embedded:
        _deadline(deadline_monotonic,reserve)
        coordinates=np.asarray([lookup[v] for v in physical],dtype=np.int64)
        transverse=np.asarray(images,dtype=np.int64)
        inverse_mass=1/np.sqrt(np.repeat(masses_kg[np.asarray(star.basis_indices,dtype=int)],3))
        raw=site_fc3[star.tensor_key]
        jk=coordinates[:,None]*a+coordinates[None,:]
        for q,point_q in enumerate(qpoints):
            for p,point_p in enumerate(qpoints):
                _deadline(deadline_monotonic,reserve)
                fi=np.exp(-1j*(transverse@point_q))*inverse_mass
                fj=np.exp(1j*(transverse@point_p))*inverse_mass
                fk=np.exp(1j*(transverse@(point_q-point_p)))*inverse_mass
                for i in range(len(coordinates)):
                    _deadline(deadline_monotonic,reserve)
                    contribution=fi[i]*fj[:,None]*fk[None,:]
                    with np.errstate(over='ignore',under='ignore',invalid='ignore'):
                        contribution*=raw[i];contribution*=factor
                    if not np.isfinite(contribution).all() or np.any((raw[i]!=0)&(contribution==0)):
                        raise ValueError("Nonzero FC3 over/underflows mass-weighted Fourier units")
                    targets=coordinates[i]*a*a+jk
                    # Each image is still distinct here. np.add.at retains
                    # all duplicate triples that fancy-index += would lose.
                    np.add.at(values[q,p].reshape(-1),targets.reshape(-1),contribution.reshape(-1))
        del contribution,targets,fi,fj,fk,jk,inverse_mass,coordinates,transverse
    _deadline(deadline_monotonic,reserve)
    vertex=PackedTransverseCubicVertex(routed,values,np.asarray(active,np.int64),plan.full_device_dofs,max_bytes=int(max_bytes))
    _deadline(deadline_monotonic)
    metadata=dict(schema='negfpy.materialTest.complete-site-vertex.v1',source_site_file=material.source_path,
        source_material_metadata=material.metadata,physical_fc3_units='eV/angstrom^3',
        masses_units='amu',omega0_rad_s=omega,hbar_j_s=hbar,
        normalization='FC3_to_SI*sqrt(hbar)/(omega0**2.5*sqrt(mi_kg*mj_kg*mk_kg))',
        fourier='exp(i*[p.(Tj-Ti)+(q-p).(Tk-Ti)])',
        coherent_images_phased_before_duplicate_safe_scatter=True,
        complete_energy_stars=True,magnitude_threshold_applied=False,endpoint_cropping=False,
        coupling_g_applied=False,mesh_normalization_applied=False,coverage_division=False,
        symmetry_averaging_applied_by_mapper=False,canonical_leg_exchange_and_reality_checks=True,
        resources=asdict(plan),numeric_plan_excludes='Python objects,allocator and library scratch;not RSS guarantee',
        deadline_scope='Cooperative;canonical owned-copy validation is noninterruptible;outer timeout remains caller responsibility',
        physical_derivative_evaluations=0,scba_maps=0,transport_qualified=False)
    return CompleteSiteVertexBuild(vertex,plan,metadata)
