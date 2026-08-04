#!/usr/bin/env python3
"""Execute one authenticated sparse-material ``Nq=1`` SCBA case.

This driver is deliberately separate from the one-shot Born resource gate.
It consumes the exact Phase-B mapped vertex, writes an atomic v2 restart
checkpoint after every fixed-point boundary, and publishes the derived Green
functions only after the terminal checkpoint has been reconstructed.  It is a
development/material algebra gate, not a release certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

import validate_sparse_nq1_phase_c_born as phase_c
import validate_sparse_nq1_phase_b as phase_b
from negfpy.anharmonic import SCBAConfig, SignedFrequencyGrid
from negfpy.anharmonic._checkpoint_v2 import (
    checkpoint_initial_sigma_v2,
    load_scba_checkpoint_v2_self_validating,
    load_scba_checkpoint_v2,
    solve_sparse_scba_checkpointed,
)
from negfpy.anharmonic._material_nq1_problem import (
    build_material_nq1_problem_from_phase_b,
)
from negfpy.anharmonic._provenance import (
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from negfpy.anharmonic.exceptions import SCBAConvergenceError
from negfpy.anharmonic._checkpoint import CheckpointDependencyError


SCHEMA = "negfpy.sparse-nq1-phase-c-scba-case.v1"
ARCHIVE_SCHEMA = "negfpy.sparse-nq1-phase-c-scba-fields.v1"
PREFLIGHT_SCHEMA = "negfpy.sparse-nq1-phase-c-scba-preflight.v1"
CHECKPOINT_SCHEMA = "negfpy_sparse_material_scba_checkpoint"
SCBA_RELEASE_CONTRACT = "dense_v0.1"
MIN_OUTPUT_FREE_BYTES = 2 * 1024**3


class SCBACaseError(RuntimeError):
    """Fail-closed input, archive, or publication error."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, ensure_ascii=True,
                       separators=(",", ":"), allow_nan=False) + "\n").encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_canonical_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SCBACaseError(f"{label} is not readable strict JSON") from error
    if not isinstance(value, dict) or payload != _canonical(value):
        raise SCBACaseError(f"{label} is not a canonical JSON object")
    return value


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": _array_sha256(value),
        }
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dataclass_fields__"):
        return _plain({name: getattr(value, name) for name in value.__dataclass_fields__})
    raise TypeError(f"unsupported report value {type(value).__name__}")


def _publish_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SCBACaseError(f"immutable artifact already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _config(*, eta_lead: float | None = None, mixing: float | None = None) -> SCBAConfig:
    # These controls are intentionally explicit and immutable for this case.
    return SCBAConfig(
        eta_lead=phase_c.ETA_LEAD if eta_lead is None else float(eta_lead),
        eta_device=0.0,
        retarded_strategy="causal_hilbert",
        hilbert_backend="direct",
        backend="fft",
        max_workers=1,
        max_iterations=200,
        min_iterations=3,
        consecutive_passes=3,
        mixing=0.2 if mixing is None else float(mixing),
        mixing_floor=0.01,
        on_nonconvergence="raise",
        max_memory_fraction=phase_c.MEMORY_FRACTION,
        allow_memory_overcommit=False,
    )


def _source_hashes(
    identity: Mapping[str, Any],
    continuation_parent: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    phase_b = identity["phase_b"]
    result = {
        "phase_a_fixture": identity["fixture"]["fixture_manifest_sha256"],
        "raw_fc3": identity["fc3_full"]["raw_sha256"],
        "phase_b_artifact_set": phase_b["artifact_set_sha256"],
        "phase_b_mapped_hdf5": phase_b["archive"]["sha256"],
        "phase_b_campaign_manifest": phase_b["campaign_manifest_sha256"],
    }
    if continuation_parent is not None:
        result.update({
            "continuation_parent_campaign_manifest": continuation_parent[
                "campaign_manifest_sha256"
            ],
            "continuation_parent_checkpoint": continuation_parent[
                "checkpoint_sha256"
            ],
            "continuation_parent_report": continuation_parent["report_sha256"],
            "continuation_parent_status": continuation_parent["status_sha256"],
        })
    return result


def authenticate_scba_inputs(
    *, fixture: Path, fc3_full: Path, phase_b_artifacts: Path,
    coupling_scale: float = 1.0,
    grid_step: float | None = None,
    grid_n_positive: int | None = None,
    eta_lead: float | None = None,
    mixing: float | None = None,
    temperature_bias: float | None = None,
) -> tuple[dict[str, Any], Any, SCBAConfig]:
    if not np.isfinite(coupling_scale) or coupling_scale <= 0.0 or coupling_scale > 1.0:
        raise SCBACaseError("coupling_scale must be finite and in (0, 1]")
    resolved_step = phase_c.GRID_STEP if grid_step is None else float(grid_step)
    resolved_n_positive = (
        phase_c.GRID_N_POSITIVE if grid_n_positive is None else int(grid_n_positive)
    )
    resolved_eta = phase_c.ETA_LEAD if eta_lead is None else float(eta_lead)
    resolved_mixing = 0.2 if mixing is None else float(mixing)
    resolved_bias = (
        phase_c.TEMPERATURE_BIAS
        if temperature_bias is None else float(temperature_bias)
    )
    if not np.isfinite(resolved_step) or resolved_step <= 0.0:
        raise SCBACaseError("grid_step must be finite and positive")
    if resolved_n_positive < 1:
        raise SCBACaseError("grid_n_positive must be a positive integer")
    if not np.isfinite(resolved_eta) or resolved_eta <= 0.0:
        raise SCBACaseError("eta_lead must be finite and positive")
    if not np.isfinite(resolved_mixing) or not 0.0 < resolved_mixing <= 1.0:
        raise SCBACaseError("mixing must be finite and in (0, 1]")
    if not np.isfinite(resolved_bias):
        raise SCBACaseError("temperature_bias must be finite")
    identity, problem, _born_config = phase_c.authenticate_phase_c_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts
    )
    controls_changed = (
        coupling_scale != 1.0
        or resolved_step != phase_c.GRID_STEP
        or resolved_n_positive != phase_c.GRID_N_POSITIVE
        or resolved_eta != phase_c.ETA_LEAD
        or resolved_bias != phase_c.TEMPERATURE_BIAS
    )
    if controls_changed:
        # Reuse the authenticated Phase-B source and harmonic model, changing
        # only the explicitly recorded continuation factor.  This is a new
        # problem identity, never a mutation of the certified archive.
        model = phase_c._load_harmonic_model(fixture)
        phase_b_set = phase_c.verify_phase_b_artifact_set(phase_b_artifacts)
        vertex = phase_b.load_phase_b_sparse_vertex(Path(phase_b_set["hdf5_path"]))
        problem = build_material_nq1_problem_from_phase_b(
            model,
            vertex,
            phase_b_set["archive"],
            grid=SignedFrequencyGrid(step=resolved_step, n_positive=resolved_n_positive),
            mean_temperature=phase_c.MEAN_TEMPERATURE,
            temperature_bias=resolved_bias,
            coupling_scale=coupling_scale,
        )
        identity = dict(identity)
        identity["problem_input_hashes"] = problem_input_hashes(problem)
    config = _config(eta_lead=resolved_eta, mixing=resolved_mixing)
    # The Phase-C assembly is the authenticated source of the material
    # problem.  Only the SCBA iteration controls differ from the Born gate.
    if problem.ndof != phase_c.N_DEVICE_DOFS:
        raise SCBACaseError("authenticated material dimensions changed")
    return identity, problem, config


def _continuation_initial_state(
    *,
    parent_artifacts: Path | None,
    identity: Mapping[str, Any],
    problem: Any,
    config: SCBAConfig,
    source_tree_sha256: str,
    environment_sha256: str,
    child_coupling_scale: float,
    continuation_axis: str | None,
) -> tuple[Any | None, dict[str, Any] | None]:
    """Authenticate a completed lower-coupling campaign and load its state."""

    if parent_artifacts is None:
        if continuation_axis is not None:
            raise SCBACaseError(
                "continuation_axis requires continuation parent artifacts"
            )
        return None, None
    if continuation_axis not in {"coupling", "temperature_bias"}:
        raise SCBACaseError(
            "a continuation parent requires axis 'coupling' or 'temperature_bias'"
        )
    root = parent_artifacts.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise SCBACaseError("continuation parent artifacts must be a directory")
    manifest_path = root / "campaign-manifest.json"
    status_path = root / "status.json"
    report_path = root / "scba-report.json"
    checkpoint_path = root / "checkpoint" / "sigma-pp.h5"
    for path, label in (
        (manifest_path, "campaign manifest"),
        (status_path, "terminal status"),
        (report_path, "SCBA report"),
        (checkpoint_path, "checkpoint"),
    ):
        if not path.is_file() or path.is_symlink():
            raise SCBACaseError(
                f"continuation parent {label} is missing or is not a regular file"
            )

    manifest = _load_canonical_json(manifest_path, "continuation parent manifest")
    status = _load_canonical_json(status_path, "continuation parent status")
    report = _load_canonical_json(report_path, "continuation parent report")
    manifest_sha = _sha256_file(manifest_path)
    status_sha = _sha256_file(status_path)
    report_sha = _sha256_file(report_path)
    checkpoint_sha_before = _sha256_file(checkpoint_path)

    if manifest.get("schema") != "negfpy.sparse-nq1-phase-c-scba-campaign.v1":
        raise SCBACaseError("continuation parent campaign schema is invalid")
    if status.get("schema") != "negfpy.sparse-nq1-phase-c-scba-status.v1":
        raise SCBACaseError("continuation parent status schema is invalid")
    if report.get("schema") != "negfpy.sparse-nq1-phase-c-scba-report.v1":
        raise SCBACaseError("continuation parent report schema is invalid")
    if (
        status.get("state") != "passed"
        or status.get("scba_converged") is not True
        or report.get("state") != "passed"
        or report.get("scba_converged") is not True
    ):
        raise SCBACaseError(
            "continuation parent is not a completed converged campaign"
        )
    if (
        status.get("campaign_manifest_sha256") != manifest_sha
        or report.get("campaign_manifest_sha256") != manifest_sha
    ):
        raise SCBACaseError("continuation parent records do not bind one manifest")
    if status.get("report_sha256") != report_sha:
        raise SCBACaseError("continuation parent status does not bind its report")
    if report.get("checkpoint_sha256") != checkpoint_sha_before:
        raise SCBACaseError("continuation parent report does not bind its checkpoint")
    declared_checkpoint = manifest.get("outputs", {}).get("checkpoint")
    if not isinstance(declared_checkpoint, str) or Path(
        declared_checkpoint
    ).expanduser().resolve() != checkpoint_path:
        raise SCBACaseError(
            "continuation parent manifest checkpoint path is inconsistent"
        )
    parent_scale = manifest.get("coupling_scale")
    if (
        isinstance(parent_scale, bool)
        or not isinstance(parent_scale, (int, float))
        or not np.isfinite(parent_scale)
        or not 0.0 < float(parent_scale) <= 1.0
    ):
        raise SCBACaseError("continuation parent coupling is invalid")
    parent_controls = manifest.get("numerical_controls")
    if not isinstance(parent_controls, dict):
        raise SCBACaseError("continuation parent numerical controls are missing")
    current_controls = {
        "grid_step": float(problem.grid.step),
        "grid_n_positive": int(problem.grid.n_positive),
        "grid_cutoff": float(problem.grid.cutoff),
        "eta_lead": float(config.eta_lead),
        "mixing": float(config.mixing),
        "temperature_bias": float(
            problem.left.temperature - problem.right.temperature
        ),
    }
    if set(parent_controls) != set(current_controls):
        raise SCBACaseError("continuation parent numerical controls are incomplete")
    if continuation_axis == "coupling":
        if not float(parent_scale) < child_coupling_scale:
            raise SCBACaseError(
                "coupling-continuation parent must be strictly below the child"
            )
        if parent_controls != current_controls:
            raise SCBACaseError(
                "coupling continuation cannot change numerical or temperature controls"
            )
    else:
        if float(parent_scale) != child_coupling_scale:
            raise SCBACaseError(
                "temperature-bias continuation cannot change cubic coupling"
            )
        for name in set(current_controls) - {"temperature_bias"}:
            if parent_controls.get(name) != current_controls[name]:
                raise SCBACaseError(
                    f"temperature-bias continuation changed {name!r}"
                )
        parent_bias = float(parent_controls["temperature_bias"])
        child_bias = current_controls["temperature_bias"]
        if not abs(parent_bias) < abs(child_bias) or (
            parent_bias != 0.0 and np.signbit(parent_bias) != np.signbit(child_bias)
        ):
            raise SCBACaseError(
                "temperature-bias parent must be closer to zero on the same branch"
            )
    if manifest.get("source_tree_sha256") != source_tree_sha256:
        raise SCBACaseError("continuation parent source-tree hash differs")
    if manifest.get("environment_lock", {}).get("sha256") != environment_sha256:
        raise SCBACaseError("continuation parent environment hash differs")

    try:
        parent = load_scba_checkpoint_v2_self_validating(checkpoint_path)
    except Exception as error:
        raise SCBACaseError(
            f"continuation parent checkpoint is invalid: {checkpoint_path}: {error}"
        ) from error
    checkpoint_sha_after = _sha256_file(checkpoint_path)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise SCBACaseError("continuation parent checkpoint changed during authentication")
    if parent.status != "completed":  # also enforced by checkpoint_initial_sigma_v2
        raise SCBACaseError(
            "continuation parents must be completed checkpoints; "
            f"{checkpoint_path} has status {parent.status!r}"
        )
    parent_identity = parent.identity
    current_hashes = problem_input_hashes(problem)
    current_sources = _source_hashes(identity)
    if parent_identity.source_tree_sha256 != source_tree_sha256:
        raise SCBACaseError("continuation parent source-tree hash differs")
    if parent_identity.environment_sha256 != environment_sha256:
        raise SCBACaseError("continuation parent environment hash differs")
    if parent_identity.configuration_sha256 != configuration_hash(config):
        raise SCBACaseError("continuation parent SCBA configuration differs")
    for name, digest in current_sources.items():
        if parent_identity.source_hashes.get(name) != digest:
            raise SCBACaseError(
                f"continuation parent material/source hash differs for {name!r}"
            )
    if parent_identity.grid_sha256 != current_hashes["frequency_grid"]:
        raise SCBACaseError("continuation parent frequency grid differs")
    stable = ["device", "frequency_grid", "solver_scale"]
    if continuation_axis == "coupling":
        stable.extend(("left_reservoir", "right_reservoir"))
    for name in stable:
        if parent_identity.problem_hashes.get(name) != current_hashes.get(name):
            raise SCBACaseError(
                f"continuation parent numerical input differs for {name!r}"
            )
    expected_shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    if parent.shape != expected_shape:
        raise SCBACaseError(
            f"continuation parent shape {parent.shape} does not match {expected_shape}"
        )
    try:
        initial = checkpoint_initial_sigma_v2(parent)
    except CheckpointDependencyError as error:  # defensive; status checked above
        raise SCBACaseError(str(error)) from error
    return initial, {
        "artifacts": str(root),
        "run_id": manifest.get("run_id"),
        "coupling_scale": float(parent_scale),
        "continuation_axis": continuation_axis,
        "campaign_manifest_sha256": manifest_sha,
        "status_sha256": status_sha,
        "report_sha256": report_sha,
        "checkpoint_sha256": checkpoint_sha_after,
        "checkpoint_identity_sha256": parent_identity.identity_sha256,
        "initial_sigma_sha256": initial_self_energy_hash(problem, initial),
    }


def _result_arrays(result: Any) -> dict[str, np.ndarray]:
    arrays = {"grid/omega": np.asarray(result.grid.values)}
    for spectrum_name in ("green", "sigma_pp", "sigma_left", "sigma_right"):
        spectrum = getattr(result, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            arrays[f"{spectrum_name}/{component}"] = np.asarray(getattr(spectrum, component))
    return arrays


def _write_result_hdf5(path: Path, metadata: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> str:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise SCBACaseError("h5py is required for SCBA result publication") from error
    if path.exists():
        raise SCBACaseError(f"immutable result already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs["schema"] = ARCHIVE_SCHEMA
            handle.attrs["scba_executed"] = True
            handle.attrs["release_certified"] = False
            for dataset_name, values in sorted(arrays.items()):
                handle.create_dataset(dataset_name, data=np.asarray(values), track_times=False)
            encoded = _canonical(dict(metadata))
            handle.create_dataset("metadata_json", data=np.frombuffer(encoded, dtype=np.uint8), track_times=False)
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def preflight_scba_inputs(
    *, fixture: Path, fc3_full: Path, phase_b_artifacts: Path,
    checkpoint: Path, output_json: Path, output_hdf5: Path,
    source_tree_sha256: str, environment_sha256: str, run_id: str,
    coupling_scale: float = 1.0,
    continuation_parent_artifacts: Path | None = None,
    continuation_axis: str | None = None,
    grid_step: float | None = None,
    grid_n_positive: int | None = None,
    eta_lead: float | None = None,
    mixing: float | None = None,
    temperature_bias: float | None = None,
) -> dict[str, Any]:
    for path, label in ((checkpoint, "checkpoint"), (output_json, "output JSON"), (output_hdf5, "output HDF5")):
        if path.exists() or path.is_symlink():
            raise SCBACaseError(f"{label} already exists; refusing overwrite")
    identity, problem, config = authenticate_scba_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts,
        coupling_scale=coupling_scale,
        grid_step=grid_step, grid_n_positive=grid_n_positive,
        eta_lead=eta_lead, mixing=mixing, temperature_bias=temperature_bias,
    )
    initial_sigma_pp, parent = _continuation_initial_state(
        parent_artifacts=continuation_parent_artifacts,
        identity=identity, problem=problem,
        config=config, source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        child_coupling_scale=coupling_scale,
        continuation_axis=continuation_axis,
    )
    output_parent = output_hdf5.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if os.statvfs(output_parent).f_bavail * os.statvfs(output_parent).f_frsize < MIN_OUTPUT_FREE_BYTES:
        raise SCBACaseError("insufficient output filesystem space")
    from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
    checkpoint_identity = make_scba_checkpoint_identity_v2(
        run_id, problem, config, environment_sha256=environment_sha256,
        initial_sigma_pp=initial_sigma_pp,
        source_hashes=_source_hashes(identity, parent),
        source_tree_sha256=source_tree_sha256,
    )
    predicted = int(problem.grid.n_points * problem.ndof * problem.ndof * 288)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "source": identity,
        "run_id": run_id,
        "coupling_scale": coupling_scale,
        "continuation_parent": parent,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "configuration_sha256": configuration_hash(config),
        "checkpoint_identity_sha256": checkpoint_identity.identity_sha256,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "outputs": {"json": str(output_json.resolve()), "hdf5": str(output_hdf5.resolve())},
        "dimensions": {"n_frequency": problem.grid.n_points, "ndof": problem.ndof, "nnz": problem.vertex.nnz},
        "predicted_peak_memory_bytes": predicted,
        "numerical_controls": {
            "grid_step": float(problem.grid.step),
            "grid_n_positive": int(problem.grid.n_positive),
            "grid_cutoff": float(problem.grid.cutoff),
            "eta_lead": float(config.eta_lead),
            "mixing": float(config.mixing),
            "temperature_bias": float(
                problem.left.temperature - problem.right.temperature
            ),
        },
        "controls": _plain(config),
        "fixed_controls_no_retuning": True,
        "resume_exact_identity": True,
        "passed": True,
        "scba_executed": False,
        "release_certified": False,
    }


def run_scba_case(
    *, fixture: Path, fc3_full: Path, phase_b_artifacts: Path,
    checkpoint: Path, output_json: Path, output_hdf5: Path,
    campaign_manifest_sha256: str, source_tree_sha256: str, environment_sha256: str,
    run_id: str, resume: bool = True, coupling_scale: float = 1.0,
    continuation_parent_artifacts: Path | None = None,
    continuation_axis: str | None = None,
    grid_step: float | None = None,
    grid_n_positive: int | None = None,
    eta_lead: float | None = None,
    mixing: float | None = None,
    temperature_bias: float | None = None,
) -> dict[str, Any]:
    identity, problem, config = authenticate_scba_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts,
        coupling_scale=coupling_scale,
        grid_step=grid_step, grid_n_positive=grid_n_positive,
        eta_lead=eta_lead, mixing=mixing, temperature_bias=temperature_bias,
    )
    initial_sigma_pp, parent = _continuation_initial_state(
        parent_artifacts=continuation_parent_artifacts,
        identity=identity, problem=problem,
        config=config, source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256,
        child_coupling_scale=coupling_scale,
        continuation_axis=continuation_axis,
    )
    from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
    expected = make_scba_checkpoint_identity_v2(
        run_id, problem, config, environment_sha256=environment_sha256,
        initial_sigma_pp=initial_sigma_pp,
        source_hashes=_source_hashes(identity, parent),
        source_tree_sha256=source_tree_sha256,
    )
    start = time.perf_counter()
    result = None
    scientific_error = None
    try:
        result = solve_sparse_scba_checkpointed(
            problem, config, checkpoint_path=checkpoint, run_id=run_id,
            environment_sha256=environment_sha256,
            source_hashes=_source_hashes(identity, parent),
            source_tree_sha256=source_tree_sha256, resume=resume,
            initial_sigma_pp=initial_sigma_pp,
        )
    except SCBAConvergenceError as error:
        result = error.result
        scientific_error = {"type": type(error).__name__, "message": str(error)}
    if result is None:
        raise SCBACaseError("SCBA returned no result")
    arrays = _result_arrays(result)
    array_hashes = {name: _array_sha256(value) for name, value in sorted(arrays.items())}
    summary = {
        "method": result.method, "converged": bool(result.converged),
        "history_length": len(result.history), "release_certified": False,
        "array_hashes": array_hashes, "currents": _plain(result.currents),
        "diagnostics": _plain(result.diagnostics), "solver_metadata": _plain(result.metadata),
        "scientific_error": scientific_error,
    }
    metadata = {
        "schema": ARCHIVE_SCHEMA, "campaign_manifest_sha256": campaign_manifest_sha256,
        "checkpoint_identity_sha256": expected.identity_sha256,
        "problem_sha256": identity["problem_input_hashes"]["problem"],
        "configuration_sha256": configuration_hash(config),
        "continuation_parent": parent,
        "result": summary,
        "timings_seconds": {"total": time.perf_counter() - start},
        "scba_executed": True, "release_certified": False,
    }
    hdf5_sha = _write_result_hdf5(output_hdf5, metadata, arrays)
    outcome = {
        "schema": SCHEMA, "campaign_manifest_sha256": campaign_manifest_sha256,
        "preflight_source": identity, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint), "checkpoint_status": load_scba_checkpoint_v2(checkpoint, expected_identity=expected).status,
        "continuation_parent": parent,
        "result": summary, "hdf5": {"path": str(output_hdf5.resolve()), "sha256": hdf5_sha, "schema": ARCHIVE_SCHEMA},
        "scba_executed": True, "release_certified": False,
        "not_certified": ["converged material conductance", "0.3.0 release", "transverse SCBA"],
    }
    _publish_once(output_json.resolve(), _canonical(outcome))
    return outcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    for name in ("fixture", "fc3-full", "phase-b-artifacts", "checkpoint", "output-json", "output-hdf5"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--campaign-manifest-sha256")
    parser.add_argument("--source-tree-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--coupling-scale", type=float, default=1.0)
    parser.add_argument("--continuation-parent-artifacts", type=Path)
    parser.add_argument(
        "--continuation-axis", choices=("coupling", "temperature_bias")
    )
    parser.add_argument("--grid-step", type=float)
    parser.add_argument("--grid-n-positive", type=int)
    parser.add_argument("--eta-lead", type=float)
    parser.add_argument("--mixing", type=float)
    parser.add_argument("--temperature-bias", type=float)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.preflight_only:
            result = preflight_scba_inputs(
                fixture=args.fixture, fc3_full=args.fc3_full, phase_b_artifacts=args.phase_b_artifacts,
                checkpoint=args.checkpoint, output_json=args.output_json, output_hdf5=args.output_hdf5,
                source_tree_sha256=args.source_tree_sha256, environment_sha256=args.environment_sha256,
                run_id=args.run_id, coupling_scale=args.coupling_scale,
                continuation_parent_artifacts=args.continuation_parent_artifacts,
                continuation_axis=args.continuation_axis,
                grid_step=args.grid_step, grid_n_positive=args.grid_n_positive,
                eta_lead=args.eta_lead, mixing=args.mixing,
                temperature_bias=args.temperature_bias,
            )
        else:
            if args.campaign_manifest_sha256 is None:
                raise SCBACaseError("--campaign-manifest-sha256 is required for execution")
            result = run_scba_case(
                fixture=args.fixture, fc3_full=args.fc3_full, phase_b_artifacts=args.phase_b_artifacts,
                checkpoint=args.checkpoint, output_json=args.output_json, output_hdf5=args.output_hdf5,
                campaign_manifest_sha256=args.campaign_manifest_sha256,
                source_tree_sha256=args.source_tree_sha256, environment_sha256=args.environment_sha256,
                run_id=args.run_id, resume=not args.no_resume, coupling_scale=args.coupling_scale,
                continuation_parent_artifacts=args.continuation_parent_artifacts,
                continuation_axis=args.continuation_axis,
                grid_step=args.grid_step, grid_n_positive=args.grid_n_positive,
                eta_lead=args.eta_lead, mixing=args.mixing,
                temperature_bias=args.temperature_bias,
            )
        print(json.dumps(_plain(result), sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        return 0 if result.get("result", {}).get("converged", result.get("passed", False)) else 2
    except SCBACaseError as error:
        print(f"SCBA case infrastructure error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
