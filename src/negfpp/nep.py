"""Prepare saved complete atomic-energy derivatives with no potential evaluation.

The two HDF5 inputs carry physical IFC2 images and complete cubic site stars.
Geometry, units, source qualifications, endpoint support and resource admission
are checked before loading numerical tensors. No fitted coefficients, campaign
imports, historical solver state or runtime namespace replacement is used.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from time import monotonic
from types import MappingProxyType

import h5py
import numpy as np

from negfpy.anharmonic.complete_energy import SignedFrequencyGrid, SolverScale, UniformBZMesh
from negfpy.anharmonic.complete_site_vertex import (
    build_vertex, load_site_fc3, load_site_ifc3_metadata, plan_vertex,
)
from negfpy.anharmonic.image_harmonic import IFC2_SCHEMA, ImageHarmonics, load_ifc2
from negfpy.anharmonic.contact_precision import (
    PreciseContactBudget, PreciseContactCaps, build_precise_compact_contact,
    algorithm_manifest, certificate, refinement,
)
from negfpy.anharmonic.io._saved_hdf import require_internal_storage
from negfpy.anharmonic.site_hessian_export import (
    geometry_for_source, assemble_original_hessian, verify_original_cubic, native_identities,
)
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.types import HarmonicReservoir

from .input import parse_json, read_json, sha256_file

_SCHEMA = "negfpp.saved-complete-energy-model.v1"
_KB = 1.380649e-23


def _admit(deadline):
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not np.isfinite(deadline):
        raise ValueError("Saved complete energy requires a finite monotonic deadline")
    if monotonic() >= deadline:
        raise TimeoutError("Complete-energy preparation reached max_seconds")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _qualified(doc, schema, material):
    if (doc.get("schema") != schema or doc.get("complete") is not True
            or doc.get("passed") is not True or doc.get("material") != material["name"]
            or doc.get("potential_sha256") != material["potential_sha256"]):
        raise ValueError("Complete same-material and same-potential qualification is required")


def _metadata(config, deadline):
    _admit(deadline)
    resource = config.document["resources"]
    if config.files["metadata"].stat().st_size > resource["model_max_bytes"]:
        raise MemoryError("Complete-energy metadata exceeds model_max_bytes")
    model = read_json(config.files["metadata"])
    if (set(model) not in ({"schema", "material", "inputs", "qualifications"},
            {"schema", "material", "inputs", "qualifications", "harmonic_source_transform"})
            or model["schema"] != _SCHEMA):
        raise ValueError("Complete-energy metadata requires " + _SCHEMA)
    material = model["material"]
    if (not isinstance(material, dict) or set(material) != {"name", "potential_sha256", "cell_rows_angstrom",
            "fractional_positions", "symbols", "masses_amu", "physical_pbc", "unit_cells_per_principal_layer"}
            or not isinstance(material["name"], str) or not material["name"]):
        raise ValueError("Complete-energy material metadata differs from the portable schema")
    potential = material["potential_sha256"]
    if not isinstance(potential, str) or len(potential) != 64 or any(c not in "0123456789abcdef" for c in potential):
        raise ValueError("Complete-energy potential SHA256 is invalid")
    transform = model.get("harmonic_source_transform")
    roles = {"ifc2", "archive"} | ({"source_archive"} if transform is not None else set())
    if (("source_archive" in config.files) != (transform is not None)
            or set(model["inputs"]) != roles
            or any(model["inputs"][key] != {k:config.hashes[key][k] for k in ("sha256", "bytes")} for key in roles)):
        raise ValueError("Complete-energy derivative files differ from their saved descriptors")
    reference_ifc2 = config.hashes["ifc2"]
    if transform is not None:
        if (not isinstance(transform,dict) or set(transform)!={"schema","algorithm","source_archive",
                "qualified_reference_ifc2","source_qualification","source_geometry"}
                or transform["source_archive"] != model["inputs"]["source_archive"]):
            raise ValueError("Original harmonic source transformation has invalid input bindings")
        reference_ifc2 = transform["qualified_reference_ifc2"]
        if (not isinstance(reference_ifc2,dict) or set(reference_ifc2)!={"sha256","bytes"}
                or type(reference_ifc2["bytes"]) is not int or reference_ifc2["bytes"]<=0):
            raise ValueError("Original harmonic reference qualification descriptor is invalid")
    proofs = model["qualifications"]
    proof_schemas = {"backend": "negfpy.materialTest.nep-backend-qualification.v1",
             "ifc2": "negfpy.material-test-ifc2-qualification.v1",
             "ifc3": "negfpy.materialTest.ifc3-qualification.v1",
             "ifc3_consistency": "negfpy.materialTest.ifc3-consistency.v1"}
    if not isinstance(proofs, dict) or set(proofs) != set(proof_schemas):
        raise ValueError("All four original complete-energy qualifications are required")
    documents = {}
    for role, schema in proof_schemas.items():
        entry = proofs[role]
        if (not isinstance(entry, dict) or set(entry) != {"sha256", "json_text"}
                or not isinstance(entry["json_text"], str)
                or hashlib.sha256(entry["json_text"].encode("utf-8")).hexdigest() != entry["sha256"]):
            raise ValueError("Original complete-energy qualification bytes differ from their SHA256")
        document = parse_json(entry["json_text"])
        _qualified(document, schema, material)
        documents[role] = document
    for role, artifact in (("ifc2", "ifc2"), ("ifc3", "archive")):
        expected = reference_ifc2["sha256"] if role=="ifc2" else config.hashes[artifact]["sha256"]
        if documents[role].get("output_sha256") != expected:
            raise ValueError("Derivative output differs from its original qualification")
    prerequisites = documents["ifc3"].get("prerequisites", {})
    for name, role in (("nep-backend.json", "backend"), ("ifc2.json", "ifc2")):
        if prerequisites.get(name, {}).get("sha256") != proofs[role]["sha256"]:
            raise ValueError("Original cubic qualification prerequisite receipts differ")
    if prerequisites.get("ifc2.hdf5", {}).get("sha256") != reference_ifc2["sha256"]:
        raise ValueError("Original cubic qualification binds a different harmonic model")
    consistency = documents["ifc3_consistency"]
    if any(consistency.get(name) != (reference_ifc2["sha256"] if role=="ifc2" else config.hashes[role]["sha256"]) for name, role in
           (("ifc2_sha256", "ifc2"), ("site_ifc3_sha256", "archive"))):
        raise ValueError("Independent IFC3 consistency qualification input bindings differ")
    if sum(config.files[role].stat().st_size for role in roles) > resource["model_max_bytes"]:
        raise MemoryError("Saved derivative files exceed model_max_bytes before tensors are loaded")
    for role in sorted(roles | {"metadata"}):
        _admit(deadline)
        if sha256_file(config.files[role]) != config.hashes[role]["sha256"]:
            raise ValueError("Complete-energy input changed after authentication: " + role)
    return model, documents


def _harmonic_header(path, cap):
    with h5py.File(path, "r") as handle:
        require_internal_storage(handle)
        if handle.attrs.get("schema") != IFC2_SCHEMA or not bool(handle.attrs.get("complete", False)):
            raise ValueError("Complete harmonic IFC2 schema/publication differs")
        names = ("images", "blocks_ev_A2", "cell_rows_angstrom", "fractional_positions", "symbols", "masses_amu")
        if any(name not in handle for name in names):
            raise ValueError("Complete harmonic dataset is absent")
        nb = handle["masses_amu"].shape
        images = handle["images"]
        if len(nb) != 1 or not nb[0] or len(images.shape) != 2 or images.shape[1] != 3 or images.dtype.kind not in "iu":
            raise ValueError("Complete harmonic image/mass dimensions differ")
        nb = nb[0]
        blocks = handle["blocks_ev_A2"]
        if blocks.shape != (images.shape[0], 3*nb, 3*nb) or blocks.dtype.kind not in "fi":
            raise ValueError("Complete harmonic tensor dimensions differ")
        numeric_bytes = sum(int(handle[name].size)*int(handle[name].dtype.itemsize) for name in names)
        if numeric_bytes > cap:
            raise MemoryError("Complete harmonic input exceeds harmonic_max_bytes before tensor loading")
        ids = images[:]
        result = {name: handle[name][:] for name in ("cell_rows_angstrom", "fractional_positions", "masses_amu")}
        result.update(symbols=tuple(handle["symbols"].asstr()[:]), images=ids,
                      numeric_payload_bytes=numeric_bytes,
                      principal_cells_upper_bound=int(np.max(np.abs(ids[:, 0]))))
    return result


def plan_nep(config, deadline):
    """Read bounded geometry and shapes; reject infeasible inputs before tensors."""
    model, proofs = _metadata(config, deadline)
    doc, resource = config.document, config.document["resources"]
    geometry, material = doc["geometry"], model["material"]
    site = load_site_ifc3_metadata(config.files["archive"], max_metadata_bytes=resource["model_max_bytes"])
    transform = model.get("harmonic_source_transform")
    source_geometry = None if transform is None else geometry_for_source(site,transform)
    harmonic = _harmonic_header(config.files["ifc2"], resource["harmonic_max_bytes"])
    for name in ("masses_amu", "cell_rows_angstrom", "fractional_positions"):
        a, b, c = np.asarray(material[name]), np.asarray(getattr(site, name)), np.asarray(harmonic[name])
        if a.shape != b.shape or a.shape != c.shape or not np.allclose(a,b,rtol=0,atol=1e-10) or not np.allclose(a,c,rtol=0,atol=1e-10):
            raise ValueError("Complete harmonic/cubic/material " + name + " disagree")
    if tuple(material["symbols"]) != site.symbols or harmonic["symbols"] != site.symbols:
        raise ValueError("Complete harmonic/cubic species or basis order differs")
    pbc = material["physical_pbc"]
    if (not isinstance(pbc,list) or len(pbc)!=3 or any(type(v) is not bool for v in pbc)
            or not pbc[0] or site.metadata.get("physical_pbc") != pbc):
        raise ValueError("Complete harmonic/cubic physical PBC must agree")
    for axis, periodic in enumerate(pbc):
        if not periodic and (np.any(harmonic["images"][:,axis] != 0)
                or any(np.any(star.cell_offsets[:,axis] != 0) for star in site.stars)):
            raise ValueError("Complete derivative image crosses a nonperiodic direction")
    if (site.metadata.get("complete") is not True or site.metadata.get("passed") is not True
            or site.metadata.get("potential_sha256") != material["potential_sha256"]
            or site.metadata.get("prerequisites") != proofs["ifc3"].get("prerequisites")):
        raise ValueError("Complete cubic HDF5 qualification or model bindings differ")
    representatives = proofs["ifc3"].get("representatives", [])
    if (len(representatives) != len(site.tensor_shapes)
            or any(row.get("derivative",{}).get("passed_raw_step_gate") is not True for row in representatives)):
        raise ValueError("Every cubic representative requires its original raw-step qualification")
    cells, centers = geometry["n_device_layers"], tuple(geometry["interaction_layers"])
    r = harmonic["principal_cells_upper_bound"]
    declared = material["unit_cells_per_principal_layer"]
    if type(declared) is not int or not 1 <= r <= declared or cells < r:
        raise ValueError("Complete lead longitudinal range exceeds declared support or device buffers")
    if "transport_supercell_matrix" in geometry and geometry["transport_supercell_matrix"] != [[1,0,0],[0,1,0],[0,0,1]]:
        raise ValueError("Saved complete-energy image labels require their recorded cell basis (identity transport matrix)")
    mesh = UniformBZMesh(tuple(doc["grid"]["mesh"]))
    grid = SignedFrequencyGrid(doc["grid"]["step"], doc["grid"]["n_positive"])
    full = 3*len(site.masses_amu)*cells
    lead = 3*len(site.masses_amu)*r
    contact_payload = 16*(3*grid.n_points+1)*mesh.n_qpoints*lead**2
    derivative_operator = 16*lead**4
    contact_input = 16*mesh.n_qpoints*4*lead**2
    base = mesh.n_qpoints*(grid.n_points+2*resource["q_samples"])
    if max(contact_payload,derivative_operator,contact_input) > resource["contact_max_bytes"]:
        raise MemoryError("Complete contact payload/operator exceeds contact_max_bytes before tensors")
    if base > resource["max_base_evaluations"]:
        raise ValueError("Complete contact work exceeds max_base_evaluations")
    harmonic_resident = 3*harmonic["numeric_payload_bytes"]+16*mesh.n_qpoints*full**2+16*(3*full**2+8*lead**2)
    if transform is not None:
        # Two representative cubes plus permutation/check scratch coexist only
        # before vertex materialization; retaining the allowance is conservative.
        harmonic_resident += 3*config.hashes["source_archive"]["bytes"]
    plan = plan_vertex(site,mesh,n_device_cells=cells,center_layers=centers,
        caller_additional_resident_bytes=harmonic_resident+2*contact_payload,
        max_geometry_endpoints=resource["max_geometry_evaluations"])
    if plan.coefficient_bytes+plan.address_bytes > resource["vertex_max_bytes"]:
        raise MemoryError("Complete vertex payload exceeds vertex_max_bytes before tensors")
    if plan.scalar_phase_scatter_evaluations > resource["max_phase_evaluations"]:
        raise ValueError("Complete phase work exceeds max_phase_evaluations before tensors")
    numeric = refinement.planned_numeric_bytes(lead)
    peak = max(plan.peak_numeric_bytes,harmonic_resident+plan.representative_input_bytes+
        plan.coefficient_bytes+4*contact_payload+6*derivative_operator+8*contact_input+max(0,numeric-6*derivative_operator))
    if peak > resource["vertex_workspace_bytes"]:
        raise MemoryError("Complete coexisting input/contact workspace exceeds vertex_workspace_bytes before tensors")
    work = certificate.planned_work(lead)
    actions = 2*mesh.n_qpoints
    precision_caps = PreciseContactCaps(max_refinement_bytes=resource["contact_max_bytes"],
        max_certificate_work_per_descriptor=work,max_total_certificate_work=actions*work,
        max_descriptor_actions=actions,max_finite_frequency_surfaces=3*actions,
        descriptor_seconds=min(300., max(1., deadline-monotonic())))
    if numeric > precision_caps.max_refinement_bytes:
        raise MemoryError("Refined contact workspace exceeds contact_max_bytes before tensors")
    case_geometry = {"device_cells":cells,"center_layers":centers,
        "active_length_nm":float(format(len(centers)*np.linalg.norm(site.cell_rows_angstrom[0])/10., ".15g"))}
    descriptor = {"cell_rows_angstrom":site.cell_rows_angstrom.tolist(),
        "fractional_positions":site.fractional_positions.tolist(),"symbols":list(site.symbols),
        "masses_amu":site.masses_amu.tolist(),"physical_pbc":pbc,**case_geometry,
        "complete_stars":[{"center_basis":s.center_basis,"basis_indices":s.basis_indices.tolist(),
            "cell_offsets":s.cell_offsets.tolist(),"representative_tensor_key":s.tensor_key} for s in site.stars]}
    geometry_sha = _digest(descriptor)
    energy_sha = _digest({"potential_sha256":material["potential_sha256"],
        "site_ifc3_sha256":config.hashes["archive"]["sha256"],"ifc2_sha256":config.hashes["ifc2"]["sha256"],
        "geometry_sha256":geometry_sha,"selected_energy":"sum of complete selected NEP atomic energies, every endpoint retained"})
    if transform is not None:
        geometry_sha,energy_sha,descriptor = native_identities(source_geometry,site,transform,cells=cells,centers=centers)
        if descriptor["active_dofs"] != list(plan.active_dofs):
            raise ValueError("Original native source and current complete vertex active basis differ")
    _admit(deadline)
    return {"site":site,"vertex_plan":plan,"mesh":mesh,"grid":grid,"precision_caps":precision_caps,
        "model":model,"proofs":proofs,"source_geometry":source_geometry,"geometry_descriptor":descriptor,"geometry_sha256":geometry_sha,
        "energy_identity_sha256":energy_sha,"planned_peak_numeric_bytes":peak,"lead_principal_cells":r}


def mass_weighted_translations(masses, repetitions=1):
    masses = np.asarray(masses,dtype=float)
    if masses.ndim != 1 or not len(masses) or not np.isfinite(masses).all() or np.any(masses<=0):
        raise ValueError("Positive finite harmonic masses are required")
    weights = np.sqrt(np.tile(masses,repetitions))
    weights /= np.linalg.norm(weights)
    result = np.zeros((len(weights)*3,3))
    for axis in range(3): result[axis::3,axis] = weights
    return result


def prepare_nep(config, deadline):
    """Return grid, mesh, full D(q), contacts, complete vertex, scale, diagnostics."""
    from .workflow import plain
    plan = plan_nep(config,deadline)
    doc, resource = config.document, config.document["resources"]
    grid,mesh = plan["grid"],plan["mesh"]
    scale = SolverScale(omega0=doc["scale"]["omega0_rad_s"],hbar=doc["scale"]["hbar_j_s"],
        metadata={"source_units":"SI; IFC2 eV/angstrom^2; cubic eV/angstrom^3; masses amu"})
    data = load_ifc2(config.files["ifc2"],max_bytes=resource["harmonic_max_bytes"])
    if data.source_sha256 != config.hashes["ifc2"]["sha256"]:
        raise ValueError("Complete IFC2 changed after planning")
    transform = plan["model"].get("harmonic_source_transform")
    if transform is not None:
        if sha256_file(config.files["source_archive"]) != config.hashes["source_archive"]["sha256"]:
            raise ValueError("Original derivative archive changed after input planning")
        original_images,_ = assemble_original_hessian(config.files["source_archive"],plan["site"],transform,
            deadline=deadline,max_bytes=resource["model_max_bytes"])
        actual = data.image_blocks
        if set(actual)!=set(original_images) or any(not np.array_equal(actual[k],v) for k,v in original_images.items()):
            raise ValueError("Exported harmonic IFC2 is not equal to the authenticated original site-Hessian assembly")
        del actual,original_images
    harmonic = ImageHarmonics.from_data(data,omega0_rad_s=scale.omega0,reciprocity="reject",
        unit_conversion="native-diagonal-mass" if transform is not None else "material-test",
        reciprocity_rtol=1e-10,reciprocity_atol=1e-12,max_numeric_bytes=resource["harmonic_max_bytes"])
    cells = doc["geometry"]["n_device_layers"]
    dimension = plan["vertex_plan"].full_device_dofs
    dynamical = np.empty((mesh.n_qpoints,dimension,dimension),dtype=np.complex128)
    for q,point in enumerate(_kpoints(mesh)):
        _admit(deadline)
        dynamical[q] = harmonic.at_q(*tuple(point),cells).dynamical
    for q,matrix in enumerate(dynamical):
        if (not np.allclose(matrix,matrix.conj().T,rtol=1e-10,atol=1e-12)
                or not np.allclose(matrix.conj(),dynamical[mesh.negative_index(q)],rtol=1e-10,atol=1e-12)):
            raise ValueError("Complete harmonic Hermiticity or opposite-q identity failed")
    gamma = harmonic.at_q(0.,0.,cells)
    bulk = gamma.lead.d00+gamma.lead.d01+gamma.lead.d10
    translations = mass_weighted_translations(harmonic.masses_amu,harmonic.principal_cells)
    error = float(np.max(np.abs(bulk@translations)))
    allowance = 1e-11+1e-8*float(np.max(np.abs(bulk)))
    if error > allowance:
        raise ValueError("Complete harmonic mass-weighted Gamma translation guard failed")
    stability = harmonic.stability_audit(_kpoints(mesh),longitudinal_points=65,
        negative_tolerance_thz2=1e-4,deadline=deadline)
    _admit(deadline)
    if sha256_file(config.files["archive"]) != config.hashes["archive"]["sha256"]:
        raise ValueError("Complete IFC3 changed after planning")
    tensors = load_site_fc3(plan["site"],max_bytes=resource["model_max_bytes"],deadline_monotonic=deadline)
    if transform is not None:
        verify_original_cubic(config.files["source_archive"],plan["site"],transform,tensors,deadline=deadline)
        if sha256_file(config.files["source_archive"]) != config.hashes["source_archive"]["sha256"]:
            raise ValueError("Original derivative archive changed during lossless input verification")
    built = build_vertex(plan["site"],tensors,mesh,n_device_cells=cells,
        center_layers=tuple(doc["geometry"]["interaction_layers"]),max_bytes=resource["vertex_max_bytes"],
        max_workspace_bytes=resource["vertex_workspace_bytes"],deadline_monotonic=deadline,
        omega0_rad_s=scale.omega0,hbar_j_s=scale.hbar,max_phase_evaluations=resource["max_phase_evaluations"],
        max_geometry_endpoints=resource["max_geometry_evaluations"],
        caller_additional_resident_bytes=plan["vertex_plan"].caller_additional_resident_bytes)
    if (built.vertex.full_ndof != plan["vertex_plan"].full_device_dofs
            or tuple(built.vertex.active_dofs) != plan["vertex_plan"].active_dofs):
        raise ValueError("Complete vertex differs from admitted endpoint geometry")
    support = dict(built.metadata,geometry_sha256=plan["geometry_sha256"],
        energy_identity_sha256=plan["energy_identity_sha256"],potential_sha256=plan["model"]["material"]["potential_sha256"],
        selected_site_derivatives_sha256=config.hashes["source_archive" if transform is not None else "archive"]["sha256"],
        full_ndof=built.vertex.full_ndof,active_dofs=built.vertex.ndof,
        active_device_indices=list(built.vertex.active_dofs),
        selected_center_count=built.plan.complete_energy_centers,all_outer_derivatives_retained=True,
        cropped_local_derivative_blocks=0,fitted_model_used=False)
    if transform is not None:
        support["source_model"] = "native-complete-site-derivatives-lossless-export"
    supported = replace(built,metadata=MappingProxyType(support))
    budget = PreciseContactBudget(plan["precision_caps"],deadline)
    contacts, records = [],{}
    for side in ("left","right"):
        kelvin = doc["reservoirs"][side+"_temperature_k"]
        temperature = scale.dimensionless_temperature(kelvin,boltzmann_constant=_KB)
        def reservoir(point,side=side,temperature=temperature):
            _admit(deadline)
            q = harmonic.at_q(*tuple(point),cells)
            return HarmonicReservoir(q.lead,temperature,
                device_to_lead=q.device_to_lead_left if side=="left" else q.device_to_lead_right,
                device_indices=tuple(int(v) for v in (q.contact_left_indices if side=="left" else q.contact_right_indices)))
        contact = build_precise_compact_contact(mesh,grid,reservoir,built.vertex.full_ndof,
            doc["reservoirs"]["eta_lead"],side=side,precision_caps=plan["precision_caps"],
            deadline_monotonic=deadline,budget=budget,q_samples=resource["q_samples"],
            max_bytes=resource["contact_max_bytes"],max_base_evaluations=resource["max_base_evaluations"])
        contacts.append(contact)
        records[side] = {"temperature_k":kelvin,"device_indices":list(contact.device_indices),
            "checks":plain(contact.contact.checks),"q_reports":plain(contact.contact.q_reports),
            "all_original_contact_physical_guards_retained":True,
            "original_binary64_derivative_screen_replaced_by_interval_machine_residual":True}
    info = {"full_atoms":built.vertex.full_ndof//3,"full_dofs":built.vertex.full_ndof,
        "active_atoms":built.vertex.ndof//3,"active_dofs":built.vertex.ndof,
        "energy_centers":built.plan.complete_energy_centers,"q_points":mesh.n_qpoints,"frequency_nodes":grid.n_points,
        "support":plain(support),"contacts":records,"source_units":{"force_constants":"eV/angstrom^3","mass":"amu"},
        "fit_residual":None,"fitting_repeated":False,"physical_derivative_evaluations":0,"eta_device":0,
        "harmonic_checks":{"hermitian_and_opposite_q":True,"gamma_translation_error":error,
            "gamma_translation_allowance":allowance,"sampled_bulk_stability":stability,"reciprocity":harmonic.reciprocity},
        "geometry_descriptor":plan["geometry_descriptor"],"lead_principal_cells":harmonic.principal_cells,
        "qualification_receipts_sha256":{k:v["sha256"] for k,v in plan["model"]["qualifications"].items()},
        "contact_refinement_algorithm":algorithm_manifest(),"precision_contact_counts":dict(budget.counts),
        "planned_peak_numeric_bytes":plan["planned_peak_numeric_bytes"]}
    if transform is not None:
        info.update(initialization_physical_input_roles=["source_archive"],
            original_source_transform={"schema":transform["schema"],"algorithm":transform["algorithm"],
                "source_archive_sha256":config.hashes["source_archive"]["sha256"],
                "exported_ifc2_verified_array_equal":True,"mapped_fc3_verified_array_equal":True,
                "harmonic_unit_conversion":harmonic.unit_conversion,
                "qualification_reference_ifc2_is_not_target_ifc2":True})
    _admit(deadline)
    return grid,mesh,dynamical,tuple(contacts),supported,scale,info
