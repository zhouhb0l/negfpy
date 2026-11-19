"""Prepare complete-energy inputs and export bounded SCBA results."""
from __future__ import annotations

from collections.abc import Mapping
import csv
import hashlib
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as packages
import json
import math
import platform
from pathlib import Path
from time import monotonic

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic._provenance import code_provenance
from negfpy.anharmonic.complete_energy import (
    ChannelSCBAConfig, SignedFrequencyGrid, SolverScale, UniformBZMesh,
    build_compact_transverse_contact, build_site_energy_vertex, restore_site_energy_model,
)
from negfpy.anharmonic.io.phono3py_crystal import read_phono3py_crystal_mapping
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.transverse_material import material_transverse_harmonic
from negfpy.anharmonic.types import HarmonicReservoir
from negfpy.core.types import LeadBlocks

from . import __version__
from .input import parse_json, read_json, sha256_file
from .solver import solve_from_zero, solve_from_self_energy


def plain(value):
    """Serialize scalar diagnostics without accidentally exporting field arrays."""
    if is_dataclass(value):
        return {f.name:plain(getattr(value,f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [plain(v) for v in value]
    if isinstance(value,np.generic):
        return plain(value.item())
    if isinstance(value,float) and not math.isfinite(value):
        return None
    if isinstance(value,np.ndarray):
        raise TypeError("Field arrays belong in fields.h5, not the scalar report")
    return value


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(plain(value),indent=2,allow_nan=False)+"\n",encoding="utf-8")
    temp.replace(path)


def six_pass(record):
    return bool(record.converged and record.sigma_passed and record.current_passed
        and set(record.sigma_residuals)=={"retarded","lesser","greater"}
        and set(record.current_residuals)=={"contact_balance","collision_power","full_balance"}
        and all(math.isfinite(v.ratio) and v.ratio<=1
                for rows in (record.sigma_residuals,record.current_residuals) for v in rows.values()))


def scalar_state(state):
    currents = plain(state.evaluation.input_currents)
    currents.update(contact_balance=state.evaluation.input_currents.contact_balance,
                    full_balance=state.evaluation.input_currents.full_balance)
    return {"index":state.index,"record":plain(state.record),"currents_w":currents,
            "physical_checks":plain(state.evaluation.checks),
            "input_sigma_evaluated":True,"mapped_image_dyson_evaluated":False}


def prepare(config, deadline):
    """Parameterize the existing silicon input construction without refitting."""
    doc, files = config.document, config.files
    if doc["model"]["type"] == "saved_complete_energy":
        from .nep import prepare_nep
        return prepare_nep(config, deadline)
    resource, cfg = doc["resources"], doc["scba"]
    def admit():
        if monotonic()>=deadline:
            raise TimeoutError("Input preparation reached max_seconds")
    admit()
    model = read_json(files["metadata"])
    if model.get("schema") != "negfpp.saved-site-energy-model.v1":
        raise ValueError("model metadata requires negfpp.saved-site-energy-model.v1")
    descriptor, material = model["model_parameters"], model["material"]
    if (descriptor["file_sha256"] != config.hashes["coefficients"]["sha256"]
            or descriptor["file_bytes"] != config.hashes["coefficients"]["bytes"]):
        raise ValueError("Coefficient file differs from its saved model descriptor")
    coefficients = np.load(files["coefficients"],allow_pickle=False)
    if (coefficients.dtype != np.dtype(descriptor["dtype"])
            or list(coefficients.shape)!=descriptor["shape"]
            or coefficients.nbytes!=descriptor["array_bytes"]):
        raise ValueError("Coefficient shape/dtype differs from the saved model")
    harmonic_crystal = read_phono3py_crystal_mapping(files["crystal"],purpose="fc2")
    fc2 = read_phono3py_fc2(files["fc2"],harmonic_crystal.supercell.masses,
        p2s_map=harmonic_crystal.p2s,s2p_map=harmonic_crystal.s2p,compact=False)
    geometry = build_layered_device_mapping(harmonic_crystal,fc2,
        transport_supercell_matrix=np.asarray(doc["geometry"]["transport_supercell_matrix"],dtype=np.int64))
    # Content-domain hashes bind the fitted model to the supplied harmonic crystal.
    if (geometry.layered_geometry_sha256!=material["layered_geometry_sha256"]
            or geometry.fc2_content_sha256!=material["source_fc2_sha256"]
            or geometry.primitive_crystal_content_sha256!=material["primitive_crystal_sha256"]):
        raise ValueError("Harmonic geometry/FC2 differs from the saved cubic model")
    crystal = read_phono3py_crystal_mapping(files["crystal"],purpose="fc3")
    fitted = restore_site_energy_model(coefficients,crystal.supercell.scaled_positions,
        crystal.supercell.lattice,np.asarray(crystal.s2p),fit_report=model["fit_report"],
        geometry=model["fit_geometry"],neighbors=descriptor["fit_configuration"]["neighbors"],
        max_atoms=descriptor["fit_configuration"]["max_atoms"],max_bytes=resource["model_max_bytes"],
        identity_rtol=cfg["identity_rtol"],identity_atol=cfg["identity_atol"],deadline_monotonic=deadline)
    scale = SolverScale(omega0=doc["scale"]["omega0_rad_s"],hbar=doc["scale"]["hbar_j_s"],
                        metadata={"source_units":"SI; IFC2 eV/angstrom^2; cubic " + material["force_constant_units"]})
    mesh = UniformBZMesh(tuple(doc["grid"]["mesh"]))
    grid = SignedFrequencyGrid(doc["grid"]["step"],doc["grid"]["n_positive"])
    layers = doc["geometry"]["n_device_layers"]
    supported = build_site_energy_vertex(mesh,fitted,crystal,geometry,n_device_layers=layers,
        interaction_layers=tuple(doc["geometry"]["interaction_layers"]),scale=scale,
        force_constant_units=material["force_constant_units"],mass_units=material["mass_units"],
        crop_derivatives=False,max_bytes=resource["vertex_max_bytes"],
        max_workspace_bytes=resource["vertex_workspace_bytes"],
        max_geometry_evaluations=resource["max_geometry_evaluations"],
        max_phase_evaluations=resource["max_phase_evaluations"],deadline_monotonic=deadline)
    support = supported.metadata
    if (support["energy_identity_sha256"]!=descriptor["energy_identity_sha256"]
            or not support["all_outer_derivatives_retained"] or support["cropped_local_derivative_blocks"]!=0):
        raise ValueError("Complete-energy model identity or endpoint support differs")
    device, lead = material_transverse_harmonic(geometry,n_device_layers=layers,
        scale=scale,max_bytes=resource["harmonic_max_bytes"])
    matrices = []
    for point in _kpoints(mesh):
        admit()
        matrices.append(assemble_device_dynamical_matrix(device.device(tuple(point))))
    for index,matrix in enumerate(matrices):
        if not np.allclose(matrix,matrix.conj().T,rtol=1e-10,atol=1e-12):
            raise ValueError("Harmonic matrix is not Hermitian")
        if not np.allclose(matrix.conj(),matrices[mesh.negative_index(index)],rtol=1e-10,atol=1e-12):
            raise ValueError("Harmonic opposite-q identity failed")
    gamma = LeadBlocks(*lead.blocks((0.,0.)))
    reverse = gamma.d01.conj().T if gamma.d10 is None else gamma.d10
    bulk = gamma.d00+gamma.d01+reverse
    # Preserve the original mass-weighted rigid-translation check for general cells.
    masses = np.tile(harmonic_crystal.primitive.masses[geometry.basis_primitive_atoms], geometry.principal_cells)
    translation = np.zeros((len(masses)*3,3))
    for axis in range(3):
        translation[axis::3,axis]=np.sqrt(masses/np.sum(masses))
    error = float(np.max(np.abs(bulk@translation)))
    allowance = cfg["identity_atol"]+cfg["identity_rtol"]*float(np.max(np.abs(bulk)))
    if error>allowance:
        raise ValueError("Gamma rigid-translation harmonic guard failed")
    contacts, contact_info = [], {}
    for side in ("left","right"):
        admit()
        kelvin=doc["reservoirs"][side+"_temperature_k"]
        def reservoir(q, kelvin=kelvin):
            return HarmonicReservoir(LeadBlocks(*lead.blocks(q)),
                temperature=scale.dimensionless_temperature(kelvin,boltzmann_constant=1.380649e-23))
        contact=build_compact_transverse_contact(mesh,grid,reservoir,supported.vertex.full_ndof,
            doc["reservoirs"]["eta_lead"],side=side,q_samples=resource["q_samples"],
            surface_gf_method="sancho_rubio",max_bytes=resource["contact_max_bytes"],
            max_base_evaluations=resource["max_base_evaluations"])
        admit()
        contacts.append(contact)
        contact_info[side]={"temperature_k":kelvin,"checks":plain(contact.contact.checks),
                            "q_reports":plain(contact.contact.q_reports),"device_indices":list(contact.device_indices)}
    info={"full_atoms":supported.vertex.full_ndof//3,"full_dofs":supported.vertex.full_ndof,
          "active_atoms":supported.vertex.ndof//3,"active_dofs":supported.vertex.ndof,
          "energy_centers":support["selected_center_count"],"q_points":mesh.n_qpoints,"frequency_nodes":grid.n_points,
          "support":plain(support),"contacts":contact_info,
          "harmonic_checks":{"hermitian_and_opposite_q":True,"gamma_translation_error":error,"gamma_translation_allowance":allowance},
          "fit_residual":plain(model["fit_report"]["residual"]),"fitting_repeated":False,
          "eta_device":0,"source_units":{"force_constants":material["force_constant_units"],"mass":material["mass_units"]}}
    return grid,mesh,np.stack(matrices),tuple(contacts),supported,scale,info


def save_fields(path, state, grid, mesh, vertex, scale):
    """Save evaluated fields and the distinct unevaluated Born image."""
    with h5py.File(path,"x") as output:
        output.attrs["schema"]="negfpp.fields.v1"
        output.attrs["evaluated_state_index"]=state.index
        output.attrs["mapped_image_dyson_evaluated"]=False
        output.attrs["omega0_rad_s"]=scale.omega0
        output.attrs["hbar_j_s"]=scale.hbar
        output.create_dataset("omega_solver",data=grid.values).attrs["units"]="dimensionless angular frequency"
        output.create_dataset("omega_rad_s",data=grid.values*scale.omega0).attrs["units"]="rad/s"
        phases=np.asarray(_kpoints(mesh))
        output.create_dataset("q_reduced",data=phases/(2*np.pi)).attrs["units"]="reciprocal lattice fractions"
        output.create_dataset("q_phase_rad",data=phases).attrs["units"]="rad"
        output.create_dataset("active_device_indices",data=vertex.active_dofs)
        for role,spectrum in (("green_evaluated",state.evaluation.green),("sigma_input_evaluated",state.sigma_input),
                              ("sigma_born_image_unevaluated",state.evaluation.map_sigma)):
            group=output.create_group(role)
            group.attrs["exact_zero"]=spectrum is None
            group.attrs["units"]="dimensionless solver fields; see SolverScale in input.json"
            if spectrum is not None:
                for name in ("retarded","statistical","spectral"):
                    group.create_dataset(name,data=getattr(spectrum,name),compression="gzip",compression_opts=1)


def run(config):
    if config.document.get("schema") == "negfpp.local-fd.v1":
        from .local_fd_workflow import run_local_fd
        return run_local_fd(config)
    started=monotonic()
    doc=config.document
    output=config.output
    source_bytes = config.source.read_bytes()
    if parse_json(source_bytes) != doc:
        raise ValueError("Input JSON changed after validation")
    output.mkdir(parents=True,exist_ok=False)
    (output/"input.json").write_bytes(source_bytes)
    report={"schema":"negfpp.result.v1","name":doc["run"]["name"],"version":__version__,
            "started_utc":datetime.now(timezone.utc).isoformat(),"status":"preparing","converged":False,
            "stop_reason":"not_started","failure_reason":None,"inputs":config.hashes,
            "counts":{"attempted":0,"completed":0,"selected":0,"rejected":0},
            "source_input_sha256":hashlib.sha256(source_bytes).hexdigest(),"outputs":{},
            "calculation":"finite-device complete-center cubic SCBA","method":doc["scba"]["method"],
            "initialization":{"kind":"fresh_zero" if config.restart is None else "external_self_energy_pending",
                              "external_reference_loaded":False},
            "long_range_extension":False,
            "scientific_release_certified":False}
    write_json(output/"report.json",report)
    anchor=latest=harmonic=None
    try:
        engine=code_provenance()
        report["software"]={"engine":engine,"command_files_sha256":{p.relative_to(Path(__file__).parent).as_posix():sha256_file(p) for p in sorted(Path(__file__).parent.rglob("*.py"))},
            "python":platform.python_version(),"platform":platform.platform(),
            "environment":{name:packages.version(name) for name in ("negfpp","numpy","scipy","h5py","phono3py","phonopy","spglib","threadpoolctl")}}
        cfg=ChannelSCBAConfig(**{key:value for key,value in doc["scba"].items() if key not in {"g","method"}})
        deadline=started+cfg.max_seconds
        with threadpool_limits(limits=1):
            grid,mesh,dynamical,contacts,supported,scale,info=prepare(config,deadline)
            report["model"]=info
            report["preparation_seconds"]=monotonic()-started
            checkpoint_binding = None
            if "checkpoint" in doc:
                from .checkpoint import make_binding
                checkpoint_binding = make_binding(config, info)
            def on_start(index,sigma,proposal):
                report["counts"]["attempted"]+=1
                print(f"Map {index} started",flush=True)
            def on_evaluation(state,decision):
                nonlocal anchor,latest,harmonic
                accepted=decision is None or decision["accepted_as_anchor"]
                report["counts"]["completed"]+=1
                report["counts"]["selected" if accepted else "rejected"]+=1
                row=scalar_state(state)
                row.update(accepted_as_anchor=bool(accepted),decision=plain(decision))
                if harmonic is None and config.restart is None:
                    harmonic=row["currents_w"].copy()
                latest=row
                if accepted:
                    anchor=row
                if checkpoint_binding is not None:
                    from .checkpoint import write_latest
                    receipt = write_latest(output/doc["checkpoint"]["file"], state, decision,
                        binding=checkpoint_binding, source_input_bytes=source_bytes,
                        grid=grid, mesh=mesh, vertex=supported.vertex, scale=scale,
                        completed_ordinal=report["counts"]["completed"],
                        selected_anchor_index=None if anchor is None else anchor["index"],
                        max_bytes=cfg.max_bytes)
                    report["checkpoint"] = receipt
                    report["outputs"]["checkpoint"] = doc["checkpoint"]["file"]
                report["selected_anchor"], report["last_evaluated"] = anchor, latest
                write_json(output/"report.json", report)
                with (output/"iterations.jsonl").open("a",encoding="utf-8") as stream:
                    stream.write(json.dumps(plain(row),allow_nan=False)+"\n")
                print(f"Map {state.index}: Sigma max {max(v.ratio for v in state.record.sigma_residuals.values()):.6g}; current max {max(v.ratio for v in state.record.current_residuals.values()):.6g}; selected {bool(accepted)}",flush=True)
            solver_options={}
            if doc["scba"]["method"]=="reference-seeded":
                solver_options["reference_config"]=doc.get("reference")
            solve = solve_from_zero
            if config.restart is not None:
                from .restart import load_self_energy
                initial_sigma, receipt = load_self_energy(config, grid, mesh, supported.vertex, scale, info)
                report["initialization"] = receipt
                write_json(output/"report.json", report)
                solve = solve_from_self_energy
                solver_options["sigma_initial"] = initial_sigma
            outcome=solve(grid,supported.vertex,dynamical,*contacts,scale=scale,g=doc["scba"]["g"],
                config=cfg,method=doc["scba"]["method"],deadline_monotonic=deadline,
                on_start=on_start,on_evaluation=on_evaluation,**solver_options)
        result=outcome.result
        passed=bool(result.converged and result.failure_reason is None and result.state is not None and six_pass(result.state.record))
        report.update(status="converged" if passed else "not_converged",converged=passed,
            stop_reason=result.stop_reason,failure_reason=result.failure_reason,
            counts={key:getattr(outcome,key) for key in ("attempted","completed","selected","rejected")},
            selected_anchor=anchor,last_evaluated=latest,harmonic_reference_currents_w=harmonic,
            plan=plain(result.plan),scba_config=plain(cfg))
        if getattr(outcome,"diagnostics",None) is not None:
            report["reference_scba"]=plain(outcome.diagnostics)
            write_json(output/"reference.json",outcome.diagnostics)
            report["outputs"]["reference"]="reference.json"
        if result.state is not None:
            report["currents_w"]=scalar_state(result.state)["currents_w"]
            save_fields(output/"fields.h5",result.state,grid,mesh,supported.vertex,scale)
            report["outputs"]["fields"]="fields.h5"
            report["fields_file"]={"sha256":sha256_file(output/"fields.h5"),
                                   "bytes":(output/"fields.h5").stat().st_size}
        with (output/"convergence.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.writer(stream)
            writer.writerow(["index","map_seconds","sigma_retarded_ratio","sigma_lesser_ratio","sigma_greater_ratio","contact_balance_ratio","collision_power_ratio","full_balance_ratio","six_passed"])
            for row in outcome.records:
                writer.writerow([row.index,row.map_seconds,*[row.sigma_residuals[k].ratio for k in ("retarded","lesser","greater")],*[row.current_residuals[k].ratio for k in ("contact_balance","collision_power","full_balance")],six_pass(row)])
        if report.get("currents_w"):
            with (output/"currents.csv").open("x",newline="",encoding="utf-8") as stream:
                writer=csv.writer(stream);writer.writerow(["quantity","watts_per_transverse_cell"])
                writer.writerows(report["currents_w"].items())
            delta=doc["reservoirs"]["left_temperature_k"]-doc["reservoirs"]["right_temperature_k"]
            if passed and delta:
                report["finite_bias_conductance_w_per_k"]=report["currents_w"]["left"]/delta
        report["outputs"].update(report="report.json",input="input.json",convergence="convergence.csv")
        if (output/"iterations.jsonl").exists():
            report["outputs"]["iterations"]="iterations.jsonl"
        if report.get("currents_w"):
            report["outputs"]["currents"]="currents.csv"
    except Exception as exc:
        report.update(status="failed",converged=False,stop_reason="error",failure_reason=f"{type(exc).__name__}: {exc}",selected_anchor=anchor,last_evaluated=latest)
    report["elapsed_seconds"]=monotonic()-started
    write_json(output/"report.json",report)
    files={p.name:sha256_file(p) for p in output.iterdir() if p.is_file() and p.name!="SHA256SUMS"}
    (output/"SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(files.items())))
    return report,output
