"""Portable complete-energy admission and tiny generated physical controls."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import monotonic
from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from negfpp import nep as s
from negfpy.anharmonic.complete_site_vertex import SITE_FC3_SCHEMA

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def document(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")


def fixture(tmp_path, *, consistency=True):
    base = tmp_path / "generated-chain"
    for directory in ("inputs", "outputs", "validation"):
        (base / directory).mkdir(parents=True)
    potential = base / "inputs/nep.txt"
    potential.write_text("Generated analytic model identity only; never evaluated as NEP.\n")
    potential_sha = sha(potential)
    cell = np.diag([2., 3., 15.])
    frac = np.array([[0., 0., .5]])
    symbols, masses = ["C"], [12.]
    config = {"schema": "negfpy.material-test-plan.v1", "material": "generated-chain",
              "potential_sha256": potential_sha, "symbols": symbols,
              "masses_amu": masses, "cell_rows_angstrom": cell.tolist(),
              "fractional_positions": frac.tolist(), "physical_pbc": [True, True, False],
              "lead_interface": {"unit_cells_per_principal_layer": 1},
              "case_definition": {
                  "short": {"device_cells": 3, "first_center_cell": 1, "center_layers": 1,
                            "active_length_nm": .2},
                  "long_direct": {"device_cells": 5, "first_center_cell": 1, "center_layers": 3,
                                  "active_length_nm": .6}}}
    document(base / "material.json", config)
    backend = {"schema": "negfpy.materialTest.nep-backend-qualification.v1",
               "material": "generated-chain", "potential_sha256": potential_sha,
               "complete": True, "passed": True, "test_fixture_not_actual_NEP": True}
    document(base / "validation/nep-backend.json", backend)
    ifc2 = base / "outputs/ifc2.hdf5"
    with h5py.File(ifc2, "w") as out:
        out.attrs.update(schema=s.IFC2_SCHEMA, complete=True, metadata_json="{}")
        out["images"] = [[-1, 0, 0], [0, 0, 0], [1, 0, 0]]
        out["blocks_ev_A2"] = np.array([-np.eye(3), 2*np.eye(3), -np.eye(3)])
        out["cell_rows_angstrom"] = cell
        out["fractional_positions"] = frac
        out["masses_amu"] = masses
        out.create_dataset("symbols", data=symbols, dtype=h5py.string_dtype())
    ifc2_proof = {"schema": "negfpy.material-test-ifc2-qualification.v1",
                  "material": "generated-chain", "complete": True, "passed": True,
                  "potential_sha256": potential_sha, "output_sha256": sha(ifc2),
                  "independent_total_force": {"passed": True}}
    document(base / "validation/ifc2.json", ifc2_proof)
    prerequisites = {name: {"sha256": sha(base / path)} for name, path in (
        ("nep-backend.json", "validation/nep-backend.json"),
        ("ifc2.json", "validation/ifc2.json"), ("ifc2.hdf5", "outputs/ifc2.hdf5"))}
    final = {"complete": True, "passed": True, "potential_sha256": potential_sha,
             "prerequisites": prerequisites}
    # E_i^cubic = (u_(i+1),x-u_i,x)^3/6.  Its literal complete site derivative
    # obeys every Cartesian leg permutation and all rigid-translation sums.
    chain = np.array([-1., 0., 0., 1., 0., 0.])
    tensor = np.einsum("i,j,k->ijk", chain, chain, chain)
    site = base / "outputs/site-ifc3.hdf5"
    with h5py.File(site, "w") as out:
        out.attrs.update(schema=SITE_FC3_SCHEMA, complete=True,
                         potential_sha256=potential_sha, finalmetadata_json=json.dumps(final))
        out["cell_rows_angstrom"] = cell
        out["fractional_positions"] = frac
        out["masses_amu"] = masses
        out["physical_pbc"] = [True, True, False]
        out.create_dataset("symbols", data=symbols, dtype=h5py.string_dtype())
        group = out.create_group("representatives/0")
        group["member_centers"] = [0]
        group["basis_indices"] = [0, 0]
        group["cell_offsets"] = [[0, 0, 0], [1, 0, 0]]
        group["relative_positions_A"] = [[0., 0., 0.], [2., 0., 0.]]
        group.create_dataset("species", data=["C", "C"], dtype=h5py.string_dtype())
        group.create_dataset("fc3_ev_A3", data=tensor).attrs["units"] = "eV/angstrom^3"
    proof = {"schema": "negfpy.materialTest.ifc3-qualification.v1", "material": "generated-chain",
             **final, "output_sha256": sha(site),
             "representatives": [{"representative": 0, "derivative": {"passed_raw_step_gate": True}}]}
    document(base / "validation/ifc3.json", proof)
    if consistency:
        document(base / "validation/ifc3-consistency.json", {
            "schema": "negfpy.materialTest.ifc3-consistency.v1", "material": "generated-chain",
            "potential_sha256": potential_sha, "ifc2_sha256": sha(ifc2), "site_ifc3_sha256": sha(site),
            "complete": True, "passed": True, "generated_analytic_identity": True})
    return base


def portable_fixture(tmp_path):
    base = fixture(tmp_path)
    material = json.loads((base/'material.json').read_text())
    roles = {'ifc2':base/'outputs/ifc2.hdf5','archive':base/'outputs/site-ifc3.hdf5','metadata':base/'model.json'}
    hashes = {key:{'sha256':sha(path),'bytes':path.stat().st_size} for key,path in roles.items() if key!='metadata'}
    qualifications={}
    for key,name in [('backend','nep-backend.json'),('ifc2','ifc2.json'),('ifc3','ifc3.json'),('ifc3_consistency','ifc3-consistency.json')]:
        path=base/'validation'/name
        qualifications[key]={'sha256':sha(path),'json_text':path.read_text()}
    metadata={'schema':s._SCHEMA,'material':{'name':material['material'],**{k:material[k] for k in ('potential_sha256','symbols','masses_amu','cell_rows_angstrom','fractional_positions','physical_pbc')},'unit_cells_per_principal_layer':1},'inputs':deepcopy(hashes),'qualifications':qualifications}
    document(roles['metadata'],metadata)
    hashes['metadata']={'sha256':sha(roles['metadata']),'bytes':roles['metadata'].stat().st_size}
    doc={'model':{'type':'saved_complete_energy','archive':'outputs/site-ifc3.hdf5','metadata':'model.json'},
        'geometry':{'transport_supercell_matrix':[[1,0,0],[0,1,0],[0,0,1]],'n_device_layers':3,'interaction_layers':[1]},
        'grid':{'mesh':[2,1,1],'step':.25,'n_positive':1},
        'reservoirs':{'left_temperature_k':300.5,'right_temperature_k':299.5,'eta_lead':.1,'eta_device':0},
        'scale':{'omega0_rad_s':2*np.pi*1e13,'hbar_j_s':1.054571817e-34},
        'resources':{'model_max_bytes':16*1024**2,'vertex_max_bytes':16*1024**2,'vertex_workspace_bytes':128*1024**2,'harmonic_max_bytes':16*1024**2,'contact_max_bytes':16*1024**2,'q_samples':9,'max_base_evaluations':1000,'max_geometry_evaluations':1000000,'max_phase_evaluations':1000000}}
    return SimpleNamespace(document=doc,files=roles,hashes=hashes),metadata


def update_metadata(config,metadata):
    document(config.files['metadata'],metadata)
    config.hashes['metadata']={'sha256':sha(config.files['metadata']),'bytes':config.files['metadata'].stat().st_size}


def deny_tensors(monkeypatch):
    original=h5py.Dataset.__getitem__
    def read(dataset,key):
        if dataset.name.endswith('fc3_ev_A3') or dataset.name.endswith('blocks_ev_A2'):
            raise AssertionError('Input admission read a physical derivative tensor')
        return original(dataset,key)
    monkeypatch.setattr(h5py.Dataset,'__getitem__',read)


def test_plan_is_geometry_only_and_has_complete_active_endpoints(tmp_path,monkeypatch):
    config,_=portable_fixture(tmp_path)
    deny_tensors(monkeypatch)
    plan=s.plan_nep(config,monotonic()+10)
    assert plan['vertex_plan'].active_dofs==tuple(range(3,9))
    assert plan['vertex_plan'].full_device_dofs==9
    assert plan['vertex_plan'].complete_energy_centers==1
    assert plan['site'].metadata['fc3_payload_read'] is False
    one=(plan['geometry_sha256'],plan['energy_identity_sha256'])
    config.document['grid']['mesh']=[1,1,1]
    plan=s.plan_nep(config,monotonic()+10)
    assert (plan['geometry_sha256'],plan['energy_identity_sha256'])==one


@pytest.mark.parametrize('role',['backend','ifc2','ifc3','ifc3_consistency'])
def test_modified_qualification_cannot_be_relabelled_without_raw_hash(tmp_path,monkeypatch,role):
    config,metadata=portable_fixture(tmp_path)
    metadata['qualifications'][role]['json_text']+=' '
    update_metadata(config,metadata)
    deny_tensors(monkeypatch)
    with pytest.raises(ValueError,match='qualification bytes'):
        s.plan_nep(config,monotonic()+10)


def test_failed_raw_step_receipt_and_crossbound_consistency_fail_before_tensors(tmp_path,monkeypatch):
    config,metadata=portable_fixture(tmp_path)
    proof=json.loads(metadata['qualifications']['ifc3_consistency']['json_text'])
    proof['site_ifc3_sha256']='0'*64
    text=json.dumps(proof)
    metadata['qualifications']['ifc3_consistency']={'json_text':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
    update_metadata(config,metadata)
    deny_tensors(monkeypatch)
    with pytest.raises(ValueError,match='consistency qualification input bindings'):
        s.plan_nep(config,monotonic()+10)


@pytest.mark.parametrize('resource',['vertex_max_bytes','vertex_workspace_bytes','contact_max_bytes','harmonic_max_bytes','max_phase_evaluations'])
def test_resource_caps_fail_before_physical_arrays(tmp_path,monkeypatch,resource):
    config,_=portable_fixture(tmp_path)
    config.document['resources'][resource]=1
    deny_tensors(monkeypatch)
    with pytest.raises((ValueError,MemoryError)):
        s.plan_nep(config,monotonic()+10)


def test_buffers_are_complete_and_basis_transform_is_explicit(tmp_path,monkeypatch):
    config,_=portable_fixture(tmp_path)
    config.document['geometry']['interaction_layers']=[2]
    deny_tensors(monkeypatch)
    with pytest.raises(ValueError,match='endpoint outside'):
        s.plan_nep(config,monotonic()+10)
    config.document['geometry']['interaction_layers']=[1]
    config.document['geometry']['transport_supercell_matrix'][0][0]=2
    with pytest.raises(ValueError,match='recorded cell basis'):
        s.plan_nep(config,monotonic()+10)


def test_small_generated_prepare_retains_thermal_physical_guards_and_fresh_inputs(tmp_path):
    from negfpy.anharmonic import acoustic_contacts
    original=acoustic_contacts._build_fixed_eta_contact_descriptor
    config,_=portable_fixture(tmp_path)
    grid,mesh,dynamic,contacts,supported,scale,info=s.prepare_nep(config,monotonic()+30)
    assert dynamic.shape==(2,9,9)
    assert info['full_atoms']==3 and info['active_atoms']==2 and info['energy_centers']==1
    assert info['fit_residual'] is None and info['fitting_repeated'] is False
    assert info['physical_derivative_evaluations']==0
    assert supported.metadata['all_outer_derivatives_retained']
    assert supported.metadata['cropped_local_derivative_blocks']==0
    assert info['precision_contact_counts']['descriptor_completed']==4
    for contact,kelvin,ids in zip(contacts,(300.5,299.5),((0,1,2),(6,7,8))):
        assert contact.device_indices==ids
        temperature=scale.dimensionless_temperature(kelvin,boltzmann_constant=1.380649e-23)
        np.testing.assert_allclose(contact.spectrum.lesser[:,grid.zero_index],-1j*temperature*contact.zero_limits)
        assert all(row['precise_contact']['finite_frequency_limit']['passed'] for row in contact.contact.q_reports)
    assert acoustic_contacts._build_fixed_eta_contact_descriptor is original
    json.dumps(info,allow_nan=False)


def test_hdf_external_and_virtual_payloads_are_rejected_before_data_access(tmp_path):
    from negfpy.anharmonic.io._saved_hdf import require_internal_storage
    outside=tmp_path/'outside.h5'
    with h5py.File(outside,'w') as f: f['payload']=[1.]
    external=tmp_path/'external.h5'
    with h5py.File(external,'w') as f: f['payload']=h5py.ExternalLink(str(outside),'/payload')
    with h5py.File(external,'r') as f:
        with pytest.raises(ValueError,match='internal hard links'): require_internal_storage(f)
    virtual=tmp_path/'virtual.h5'
    layout=h5py.VirtualLayout(shape=(1,),dtype='f8');layout[:]=h5py.VirtualSource(str(outside),'payload',shape=(1,))
    with h5py.File(virtual,'w') as f: f.create_virtual_dataset('payload',layout)
    with h5py.File(virtual,'r') as f:
        with pytest.raises(ValueError,match='non-virtual'): require_internal_storage(f)

def transformed_fixture(tmp_path):
    from negfpy.anharmonic.site_hessian_export import ALGORITHM,_CHECKS
    config,metadata=portable_fixture(tmp_path)
    material=metadata['material'];material['cell_rows_angstrom']=(2*np.eye(3)).tolist()
    def pin(path):return {'sha256':sha(path),'bytes':path.stat().st_size}
    def receipt(value):
        text=json.dumps(value,sort_keys=True)+'\n'
        return {'sha256':hashlib.sha256(text.encode()).hexdigest(),'json_text':text}
    path=config.files['ifc2']
    with h5py.File(path,'r+') as f:f['cell_rows_angstrom'][:]=np.eye(3)*2
    docs={k:json.loads(v['json_text']) for k,v in metadata['qualifications'].items()}
    reference=pin(path);docs['ifc2']['output_sha256']=reference['sha256']
    metadata['qualifications']['ifc2']=receipt(docs['ifc2'])
    prerequisites=docs['ifc3']['prerequisites']
    prerequisites['ifc2.json']['sha256']=metadata['qualifications']['ifc2']['sha256']
    prerequisites['ifc2.hdf5']['sha256']=reference['sha256']
    raw=path.parent/'original-site.h5';vector=np.array([-1.,0,0,1.,0,0])
    with h5py.File(raw,'w') as f:
        f.attrs.update(schema='negfpy.nep3.direct-site-derivatives.v1',complete=True,potential_sha256=material['potential_sha256'])
        g=f.create_group('representative_0')
        g.create_dataset('fc2_richardson',data=np.kron([[1.,-1.],[-1.,1.]],np.eye(3))).attrs['units']='eV/angstrom^2'
        g.create_dataset('fc3_richardson',data=np.einsum('i,j,k->ijk',vector,vector,vector)).attrs['units']='eV/angstrom^3'
    rawpin=pin(raw)
    source_qualification=receipt({'schema':'negfpy.nep3.material-consistency-qualification.v1','complete':True,'passed':True,'potential_sha256':material['potential_sha256'],'site_derivatives_sha256':rawpin['sha256'],'checks':{k:True for k in _CHECKS},'generated_analytic_fixture':True})
    source_geometry=receipt({'schema':'negfpy.nep.geometry-admission.v1','geometry_only':True,
        'basis_eighths':[[0,0,4]],'lattice_constant_angstrom':2.,'site_radial_and_angular_cutoff_angstrom':2.1,
        'center_classes':[{'representative_basis':0,'member_basis_indices':[0],'sites':[
            {'basis_index':0,'cell_offset':[0,0,0],'relative_eighths':[0,0,0]},
            {'basis_index':0,'cell_offset':[1,0,0],'relative_eighths':[8,0,0]}]}]})
    with h5py.File(config.files['archive'],'r+') as f:
        f['cell_rows_angstrom'][:]=np.eye(3)*2
        final=json.loads(f.attrs['finalmetadata_json']);final['prerequisites']=prerequisites
        final['reused_qualified_silicon_source']={'source_sha256':rawpin['sha256'],'qualification_sha256':source_qualification['sha256']}
        f.attrs['metadata_json']=json.dumps(final)
    docs['ifc3']['output_sha256']=sha(config.files['archive'])
    docs['ifc3']['prerequisites']=prerequisites
    docs['ifc3_consistency']['ifc2_sha256']=reference['sha256']
    docs['ifc3_consistency']['site_ifc3_sha256']=docs['ifc3']['output_sha256']
    metadata['qualifications']={k:receipt(v) for k,v in docs.items()}
    config.files['source_archive']=raw
    for role in ('ifc2','archive','source_archive'):config.hashes[role]=pin(config.files[role])
    metadata['inputs']={k:config.hashes[k] for k in ('ifc2','archive','source_archive')}
    metadata['harmonic_source_transform']={'schema':'negfpp.original-site-hessian-transform.v1','algorithm':ALGORITHM,
        'source_archive':rawpin,'qualified_reference_ifc2':reference,
        'source_qualification':source_qualification,'source_geometry':source_geometry}
    update_metadata(config,metadata)
    return config,metadata


def test_generated_transformed_prepare_verifies_both_payloads_then_exposes_native_identity(tmp_path):
    config,metadata=transformed_fixture(tmp_path)
    grid,mesh,dynamic,contacts,supported,scale,info=s.prepare_nep(config,monotonic()+30)
    assert dynamic.shape==(2,9,9)
    assert info['initialization_physical_input_roles']==['source_archive']
    assert supported.metadata['source_model']=='native-complete-site-derivatives-lossless-export'
    proof=info['original_source_transform']
    assert proof['exported_ifc2_verified_array_equal'] and proof['mapped_fc3_verified_array_equal']
    assert proof['harmonic_unit_conversion']=='native-diagonal-mass'
    assert supported.metadata['selected_site_derivatives_sha256']==config.hashes['source_archive']['sha256']
    plan=s.plan_nep(config,monotonic()+10)
    assert plan['geometry_sha256']==supported.metadata['geometry_sha256']
    assert plan['energy_identity_sha256']==supported.metadata['energy_identity_sha256']
    assert plan['geometry_descriptor']['active_dofs']==list(supported.vertex.active_dofs)


def test_changed_export_cannot_borrow_reference_qualification(tmp_path,monkeypatch):
    config,metadata=transformed_fixture(tmp_path)
    with h5py.File(config.files['ifc2'],'r+') as f:f['blocks_ev_A2'][1,0,0]+=1e-9
    config.hashes['ifc2']={'sha256':sha(config.files['ifc2']),'bytes':config.files['ifc2'].stat().st_size}
    metadata['inputs']['ifc2']=config.hashes['ifc2']
    update_metadata(config,metadata)
    with pytest.raises(ValueError,match='not equal to the authenticated original'):
        s.prepare_nep(config,monotonic()+30)
