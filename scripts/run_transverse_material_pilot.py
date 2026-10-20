"""Bounded real-silicon transverse input pilot; no SCBA or current evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from negfpy.anharmonic._harmonic import assemble_device_dynamical_matrix
from negfpy.anharmonic._material_nq1_campaign import (
    HBAR_J_S,
    OMEGA0_RAD_S,
    kelvin_to_solver_temperature,
)
from negfpy.anharmonic.fc3 import validate_fc3
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.io.phono3py import read_phono3py_fc3
from negfpy.anharmonic.io.phono3py_crystal import read_phono3py_crystal_mapping
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import build_layered_device_mapping
from negfpy.anharmonic.mapped_fc3 import mass_normalize_translational_fc3
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational
from negfpy.anharmonic.transverse_contacts import build_transverse_contact
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.transverse_material import (
    fourier_material_cubic,
    material_transverse_harmonic,
)
from negfpy.anharmonic.types import HarmonicReservoir, SolverScale
from negfpy.core.types import LeadBlocks


def plain(value):
    if isinstance(value, Mapping):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(plain(value), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def array_hash(values):
    return hashlib.sha256(memoryview(np.ascontiguousarray(values)).cast("B")).hexdigest()


def memory_plan(nq, ndof, nlead, n_frequency=601, pilot_nodes=3):
    """Exact array payloads; explicitly not peak process memory."""
    return {
        "vertex_bytes": 16 * nq**2 * ndof**3,
        "full_grid_spectrum_bytes": 48 * nq * n_frequency * ndof**2,
        "pilot_contact_with_limits_bytes": 16 * (3 * pilot_nodes + 1) * nq * ndof**2,
        "surface_derivative_operator_bytes": 16 * nlead**4,
        "dense_born_pairs_per_component": nq**2 * (
            3 * ((n_frequency - 1) // 2)**2 + 3 * ((n_frequency - 1) // 2) + 1
        ),
        "prototype_default_payload_limit": 64 * 1024**2,
        "prototype_default_born_pair_limit": 100_000,
        "includes_copies_workspaces_or_rss": False,
    }


def run(profile, result, emit):
    cfg = profile["pilot"]
    inputs = {row["name"]: Path(row["path"]) for row in profile["runtime_inputs"]}
    if cfg["g"] != 0.03 or cfg["eta_lead"] != 0.0015625 or cfg["eta_device"] != 0:
        raise ValueError("This material pilot preserves the approved coupling and broadening.")
    if cfg["n_device_layers"] != 9 or cfg["interaction_layers"] != [2, 3, 4, 5, 6]:
        raise ValueError("This pilot preserves the accepted nine-layer interaction crop.")
    mesh = UniformBZMesh(tuple(cfg["mesh"]))
    if mesh.shape != (4, 1, 1):
        raise ValueError("The frozen pilot requires Gamma, Nyquist and one complex opposite-q pair.")
    grid = SignedFrequencyGrid(0.025, 1)
    scale = SolverScale(omega0=OMEGA0_RAD_S, hbar=HBAR_J_S)
    result["settings"] = cfg
    result["source_inputs"] = profile["runtime_inputs"]
    result["source_commit"] = profile["source_commit"]
    result["source_tree"] = profile["source_tree"]
    result["scope"] = {
        "actual_material_input_validation": True, "scba": False,
        "currents": False, "frequency_or_q_convergence": False,
        "accepted_field_update": False,
        "contact_frequencies": grid.values.tolist(),
        "cubic_vertex_coupling_applied": False,
        "g_for_future_born_calculation": cfg["g"],
    }

    emit("read_fc2_start")
    crystal2 = read_phono3py_crystal_mapping(inputs["crystal"], purpose="fc2")
    fc2 = read_phono3py_fc2(
        inputs["fc2"], crystal2.supercell.masses,
        p2s_map=crystal2.p2s, s2p_map=crystal2.s2p, compact=False,
    )
    geometry = build_layered_device_mapping(
        crystal2, fc2, transport_supercell_matrix=np.asarray(cfg["transport_matrix"], dtype=np.int64),
    )
    if geometry.n_principal_layer_dofs != 24 or geometry.principal_cells != 1:
        raise ValueError("Unexpected material principal layer.")
    if geometry.layered_geometry_sha256 != cfg["expected_geometry_sha256"]:
        raise ValueError("Material geometry identity differs from the accepted source.")
    ndof = 9 * geometry.n_principal_layer_dofs
    result["geometry"] = {
        "ndof": ndof, "principal_cells": geometry.principal_cells,
        "principal_layer_dofs": geometry.n_principal_layer_dofs,
        "transport_basis_atoms": geometry.n_transport_basis_atoms,
        "geometry_sha256": geometry.layered_geometry_sha256,
        "fc2_content_sha256": geometry.fc2_content_sha256,
        "primitive_sha256": geometry.primitive_crystal_content_sha256,
    }
    result["memory_plan"] = memory_plan(mesh.n_qpoints, ndof, 24)
    emit("geometry_complete", result["geometry"])

    emit("harmonic_map_start")
    device, lead = material_transverse_harmonic(geometry, n_device_layers=9, scale=scale)
    points = _kpoints(mesh)
    devices = [assemble_device_dynamical_matrix(device.device(tuple(q))) for q in points]
    for q, matrix in enumerate(devices):
        if not np.allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-12):
            raise ValueError(f"Device is not Hermitian at q={q}.")
        if not np.allclose(matrix.conj(), devices[mesh.negative_index(q)], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Device opposite-q identity fails at q={q}.")
    gamma = LeadBlocks(*lead.blocks((0.0, 0.0)))
    reverse = gamma.d01.conj().T if gamma.d10 is None else gamma.d10
    gamma_bulk = gamma.d00 + gamma.d01 + reverse
    translations = np.zeros((24, 3))
    # Eight identical Si masses: normalized mass-weighted rigid translations.
    for axis in range(3):
        translations[axis::3, axis] = 1 / np.sqrt(8)
    rigid_error = float(np.max(np.abs(gamma_bulk @ translations)))
    rigid_allowance = 1e-11 + 1e-8 * float(np.max(np.abs(gamma_bulk)))
    if rigid_error > rigid_allowance:
        raise ValueError(f"Gamma rigid-translation residual {rigid_error} exceeds {rigid_allowance}.")
    result["harmonic"] = {
        "device_hashes": [array_hash(v) for v in devices],
        "maximum_imaginary_entry_by_q": [float(np.max(np.abs(v.imag))) for v in devices],
        "gamma_rigid_translation_residual": rigid_error,
        "gamma_rigid_translation_allowance": rigid_allowance,
        "gamma_bulk_eigenvalues": np.linalg.eigvalsh(gamma_bulk).tolist(),
        "all_q_hermitian_and_opposite_q_valid": True,
    }
    emit("harmonic_map_complete", result["harmonic"])

    # Use three existing coarse-grid nodes for contact algebra only. This does
    # not replace the 601-node material frequency grid or integrate transport.
    result["contacts"] = {}
    for side, kelvin in (("left", 305.0), ("right", 295.0)):
        emit("contact_start", {"side": side})

        def reservoir(q, kelvin=kelvin):
            return HarmonicReservoir(
                LeadBlocks(*lead.blocks(q)), temperature=kelvin_to_solver_temperature(kelvin),
            )

        contact = build_transverse_contact(
            mesh, grid, reservoir, ndof, cfg["eta_lead"], side=side,
            q_samples=257, surface_gf_method="sancho_rubio",
        )
        record = {
            "temperature_kelvin": kelvin, "checks": dict(contact.checks),
            "q_reports": list(contact.q_reports),
            "zero_limits_sha256": array_hash(contact.zero_limits),
            "zero_limit_norm_by_q": [float(np.linalg.norm(v)) for v in contact.zero_limits],
            "spectrum_sha256": {name: array_hash(getattr(contact.spectrum, name))
                                for name in ("retarded", "lesser", "greater")},
        }
        result["contacts"][side] = record
        emit("contact_complete", {"side": side, **record})
        del contact

    emit("read_fc3_start")
    crystal3 = read_phono3py_crystal_mapping(inputs["crystal"], purpose="fc3")
    raw = read_phono3py_fc3(
        inputs["fc3"], crystal3.supercell.masses,
        p2s_map=crystal3.p2s, s2p_map=crystal3.s2p, compact=False,
    )
    certification = validate_fc3(raw)
    if not certification.passed or not certification.permutation_complete:
        raise ValueError(f"Raw FC3 audit failed: {certification.errors}.")
    if certification.source_fc3_content_sha256 != cfg["expected_fc3_content_sha256"]:
        raise ValueError("Raw FC3 content differs from the accepted material source.")
    emit("raw_fc3_validated", {"source_sha256": certification.source_fc3_content_sha256,
                               "metrics": certification.metrics})
    translated = convert_fc3_to_translational(raw, crystal3, make_r0_average=True)
    normalized = mass_normalize_translational_fc3(
        translated, crystal3, source_certification=certification, output_units="si",
    )
    emit("normalized_fc3_complete", {"source_image_terms": len(normalized.terms),
                                     "content_sha256": normalized.content_sha256})
    streamed = fourier_material_cubic(
        mesh, normalized, geometry, n_device_layers=9,
        interaction_layers=(2, 3, 4, 5, 6), scale=scale,
        max_bytes=cfg["vertex_max_bytes"],
        max_geometry_evaluations=cfg["max_geometry_evaluations"],
        max_phase_evaluations=cfg["max_phase_evaluations"],
    )
    vertex, metadata = streamed.vertex, streamed.metadata
    block_reports = []
    for q in range(mesh.n_qpoints):
        for p in range(mesh.n_qpoints):
            block = vertex.coefficients[q, p]
            norm = float(np.linalg.norm(block))
            block_reports.append({
                "q": q, "p": p, "sha256": array_hash(block),
                "norm_unscaled": norm, "norm_at_g": cfg["g"] * norm,
                "maximum_imaginary_entry": float(np.max(np.abs(block.imag))),
            })
    result["cubic"] = {
        "metadata": metadata, "shape": vertex.coefficients.shape,
        "blocks": block_reports, "all_joint_vertex_checks_passed": True,
        "coupling_applied": False, "solver_units_before_g": True,
    }
    emit("cubic_fourier_complete", {"metadata": metadata, "shape": vertex.coefficients.shape})
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    started = time.monotonic()
    result = {"passed": False, "schema": "negfpy.transverse-material-input-pilot.v1"}

    def emit(event, details=None):
        row = {"event": event, "time": datetime.now(timezone.utc).isoformat(),
               "elapsed_seconds": time.monotonic() - started, "details": plain(details or {})}
        with (args.output.parent / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps(row, sort_keys=True, allow_nan=False), flush=True)

    try:
        run(profile, result, emit)
    except Exception as exc:  # noqa: BLE001 - preserve a failed pilot's completed stages
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        traceback.print_exc()
        emit("pilot_failed", result["failure"])
    result["elapsed_seconds"] = time.monotonic() - started
    result["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    write_json(args.output, result)
    emit("pilot_closed", {"passed": result["passed"], "elapsed_seconds": result["elapsed_seconds"]})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
