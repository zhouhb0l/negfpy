"""Run the existing frozen-retarded Method 1 with local Bose probe closure."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
from importlib import metadata as packages
import json
from pathlib import Path
import platform
import resource
from time import monotonic

import numpy as np
from threadpoolctl import threadpool_limits

from negfpy.anharmonic._provenance import code_provenance

from . import __version__
from .input import sha256_file
from .local_fd.g_only_engine import GOnlyParallelCurrents
from .local_fd.harmonic_blocks import certify_repeated_stability
from .local_fd.material import prepare_material
from .local_fd.source import load_frozen_source
from .local_fd.temperature_solver import (
    SolverOptions, build_adaptive_preconditioner, solve_linear_response,
    solve_probe_temperatures,
)
from .local_fd_resources import resource_plan
from .workflow import plain, write_json


def _profile(path, centers, period, temperatures, currents, *, linear=False):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["terminal", "role", "cell_origin_nm",
                         "temperature_shift_k_per_k" if linear else "temperature_k",
                         "current_w_per_k" if linear else "current_w"])
        for terminal in range(centers + 2):
            role = "left" if terminal == 0 else "right" if terminal == 1 else "probe"
            cell = 0 if terminal == 0 else centers + 1 if terminal == 1 else terminal - 1
            writer.writerow([terminal, role, cell*period, temperatures[terminal], currents[terminal]])


def run_local_fd(config):
    """Preserve failed diagnostics and require both original probe solve gates."""
    started = monotonic()
    doc, output = config.document, config.output
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "input.json", doc)
    report = {
        "schema": "negfpp.local-fd-result.v1", "calculation_type": "method1-local-fd",
        "name": doc["run"]["name"], "version": __version__,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "status": "preparing", "converged": False, "stop_reason": "not_started",
        "failure_reason": None, "source_input_sha256": sha256_file(config.source),
        "new_scba_maps": 0, "new_born_contractions": 0,
        "retarded_kernel_recomputed": False, "calculation": "frozen retarded kernel; local fluctuation-dissipation/Bose probes",
        "scientific_release_certified": False,
        "model_limitations": ["Frozen short-device retarded approximation", "Cross-center Born vertex pairs omitted",
                              "No frequency/q/eta convergence certification"],
        "outputs": {"input": "input.json", "report": "report.json"},
    }
    write_json(output / "report.json", report)

    def emit(phase, details=None):
        record = {"phase": phase, "elapsed_seconds": monotonic()-started, "details": plain(details or {})}
        write_json(output / "progress.json", record)
        with (output / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        # Keep large per-worker tile records in engine.json only.
        print(f"Local FD: {phase} ({record['elapsed_seconds']:.1f} s)", flush=True)

    try:
        controls = doc["resources"]
        deadline = started + controls["max_seconds"] - controls["exit_reserve_seconds"]
        centers = doc["geometry"]["center_layers"]
        # Reauthenticate after input checking, before any long-device allocation.
        source = load_frozen_source(config.manifest, expected_sha256=doc["frozen_source"]["sha256"])
        plan = resource_plan(centers, source.mesh.n_qpoints, len(source.grid.positive_indices), controls)
        write_json(output / "resource-plan.json", plan)
        report["outputs"]["resource_plan"] = "resource-plan.json"
        write_json(output / "donor-qualification.json", source.qualification)
        report["outputs"]["donor_qualification"] = "donor-qualification.json"
        report["donor"] = source.qualification
        report["software"] = {
            "engine": code_provenance(), "python": platform.python_version(), "platform": platform.platform(),
            "command_files_sha256": {p.relative_to(Path(__file__).parent).as_posix(): sha256_file(p)
                                     for p in sorted(Path(__file__).parent.rglob("*.py"))},
            "environment": {name: packages.version(name) for name in
                            ("negfpp", "numpy", "scipy", "h5py", "phono3py", "phonopy", "spglib", "threadpoolctl")},
        }
        temperatures = doc["reservoirs"]
        left, right, equilibrium = (temperatures[key] for key in
            ("left_temperature_k", "right_temperature_k", "equilibrium_temperature_k"))
        options = SolverOptions(**{k:v for k,v in doc["solver"].items() if k != "preconditioner_seconds"})
        report["solver_options"] = plain(options)
        with threadpool_limits(limits=1):
            emit("reconstructing_authenticated_short_inputs")
            data = prepare_material(source, deadline_monotonic=deadline,
                                    harmonic_max_bytes=controls["harmonic_max_bytes"],
                                    contact_max_bytes=controls["contact_max_bytes"])
            report["material"] = data.metadata
            period = data.metadata["transport_repeat_nm"]
            report["geometry"] = {
                "center_layers": centers, "full_device_layers": centers+2,
                "full_atoms": 8*(centers+2), "full_dofs": 24*(centers+2),
                "transport_repeat_nm": period, "center_slab_length_nm": centers*period,
                "full_device_box_nm": (centers+2)*period, "cross_section_nm2": data.metadata["cross_section_nm2"],
            }
            emit("checking_long_harmonic_stability")
            stability = []
            for q in range(source.mesh.n_qpoints):
                if monotonic() >= deadline:
                    raise TimeoutError("Harmonic stability reached the shared deadline")
                stability.append(certify_repeated_stability(data.harmonic.onsite[q], data.harmonic.forward[q], centers+2))
            report["harmonic_stability"] = plain(stability)
            write_json(output / "report.json", report)
            emit("building_recursive_caches")
            with GOnlyParallelCurrents(data, centers, deadline=deadline,
                                      tile_size=controls["frequency_tile"], max_workers=controls["workers"],
                                      max_cache_bytes=controls["max_cache_bytes"], emit=emit,
                                      left_temperature=left, right_temperature=right,
                                      equilibrium_temperature=equilibrium) as engine:
                report["harmonic_reference_currents_w"] = {"left": float(engine.harmonic_current[0]),
                                                          "right": float(engine.harmonic_current[1])}
                if not np.isfinite(engine.harmonic_current).all():
                    raise ValueError("Nonfinite harmonic reference current")
                if left != right and engine.harmonic_current[0]*(left-right) <= 0:
                    raise ValueError("Harmonic current flows against the temperature bias")
                preconditioner = build_adaptive_preconditioner(
                    engine.jacobian_action, np.full(centers+2, equilibrium),
                    max_seconds=doc["solver"]["preconditioner_seconds"], deadline=deadline)
                report["preconditioner"] = plain(preconditioner.diagnostics)
                emit("solving_local_probe_temperatures")
                initial = np.r_[left, right, np.linspace(left, right, centers+2)[1:-1]]
                thermal = solve_probe_temperatures(
                    engine.currents, engine.jacobian_action, initial,
                    contact_temperatures=(left, right), preconditioner=preconditioner, options=options, deadline=deadline)
                report["nonlinear"] = plain(thermal.scalar_summary())
                report["currents_w"] = {"left": float(thermal.currents_w[0]), "right": float(thermal.currents_w[1])}
                _profile(output / "temperatures.csv", centers, period, thermal.temperatures_kelvin, thermal.currents_w)
                report["outputs"]["temperatures"] = "temperatures.csv"
                if left != right and thermal.success:
                    report["finite_bias_conductance_w_per_k"] = float(thermal.currents_w[0]/(left-right))
                write_json(output / "report.json", report)
                if not thermal.success:
                    if monotonic() < deadline:
                        write_json(output / "engine.json", engine.stats())
                        report["outputs"]["engine"] = "engine.json"
                    raise ValueError("Nonlinear local-FD closure failed: " + thermal.message)
                emit("solving_linear_response")
                linear = solve_linear_response(
                    engine.currents, engine.jacobian_action, equilibrium, centers,
                    bias_direction=(.5, -.5), preconditioner=preconditioner, options=options, deadline=deadline)
                report["linear_response"] = plain(linear.scalar_summary())
                _profile(output / "linear-response.csv", centers, period, linear.temperature_shifts_kelvin, linear.currents_w, linear=True)
                report["outputs"]["linear_response"] = "linear-response.csv"
                write_json(output / "engine.json", engine.stats())
                report["outputs"]["engine"] = "engine.json"
                if not linear.success:
                    raise ValueError("Linear-response local-FD closure failed: " + linear.message)
                if not np.isfinite(linear.conductance_w_per_kelvin) or linear.conductance_w_per_kelvin <= 0:
                    raise ValueError("Nonpositive/nonfinite linear conductance")
        report.update(status="converged", converged=True, stop_reason="local_probe_and_linear_response_gates_passed")
        emit("complete")
    except Exception as exc:
        report.update(status="failed", converged=False, stop_reason="error",
                      failure_reason=f"{type(exc).__name__}: {exc}")
        if hasattr(exc, "failure_record"):
            write_json(output / "worker-failure.json", exc.failure_record)
            report["outputs"]["worker_failure"] = "worker-failure.json"
        emit("failed", {"error": report["failure_reason"]})
    report["elapsed_seconds"] = monotonic()-started
    report["parent_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    report["parent_rss_excludes_workers"] = True
    report["outputs"].update(progress="progress.json", progress_history="progress.jsonl")
    write_json(output / "report.json", report)
    files = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file() and p.name != "SHA256SUMS"}
    (output / "SHA256SUMS").write_text("".join(f"{digest}  {name}\n" for name,digest in sorted(files.items())))
    return report, output
