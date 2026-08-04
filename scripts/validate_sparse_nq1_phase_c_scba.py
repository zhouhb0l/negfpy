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
from negfpy.anharmonic import SCBAConfig
from negfpy.anharmonic._checkpoint_v2 import (
    load_scba_checkpoint_v2,
    solve_sparse_scba_checkpointed,
)
from negfpy.anharmonic._provenance import configuration_hash
from negfpy.anharmonic.exceptions import SCBAConvergenceError


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


def _config() -> SCBAConfig:
    # These controls are intentionally explicit and immutable for this case.
    return SCBAConfig(
        eta_lead=phase_c.ETA_LEAD,
        eta_device=0.0,
        retarded_strategy="causal_hilbert",
        hilbert_backend="direct",
        backend="fft",
        max_workers=1,
        max_iterations=200,
        min_iterations=3,
        consecutive_passes=3,
        mixing=0.2,
        mixing_floor=0.01,
        on_nonconvergence="raise",
        max_memory_fraction=phase_c.MEMORY_FRACTION,
        allow_memory_overcommit=False,
    )


def _source_hashes(identity: Mapping[str, Any]) -> dict[str, str]:
    phase_b = identity["phase_b"]
    return {
        "phase_a_fixture": identity["fixture"]["fixture_manifest_sha256"],
        "raw_fc3": identity["fc3_full"]["raw_sha256"],
        "phase_b_artifact_set": phase_b["artifact_set_sha256"],
        "phase_b_mapped_hdf5": phase_b["archive"]["sha256"],
        "phase_b_campaign_manifest": phase_b["campaign_manifest_sha256"],
    }


def authenticate_scba_inputs(
    *, fixture: Path, fc3_full: Path, phase_b_artifacts: Path
) -> tuple[dict[str, Any], Any, SCBAConfig]:
    identity, problem, _born_config = phase_c.authenticate_phase_c_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts
    )
    config = _config()
    # The Phase-C assembly is the authenticated source of the material
    # problem.  Only the SCBA iteration controls differ from the Born gate.
    if problem.grid.n_points != phase_c.GRID_N_POINTS or problem.ndof != phase_c.N_DEVICE_DOFS:
        raise SCBACaseError("authenticated material dimensions changed")
    return identity, problem, config


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
) -> dict[str, Any]:
    for path, label in ((checkpoint, "checkpoint"), (output_json, "output JSON"), (output_hdf5, "output HDF5")):
        if path.exists() or path.is_symlink():
            raise SCBACaseError(f"{label} already exists; refusing overwrite")
    identity, problem, config = authenticate_scba_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts
    )
    output_parent = output_hdf5.resolve().parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if os.statvfs(output_parent).f_bavail * os.statvfs(output_parent).f_frsize < MIN_OUTPUT_FREE_BYTES:
        raise SCBACaseError("insufficient output filesystem space")
    from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
    checkpoint_identity = make_scba_checkpoint_identity_v2(
        run_id, problem, config, environment_sha256=environment_sha256,
        source_hashes=_source_hashes(identity), source_tree_sha256=source_tree_sha256,
    )
    predicted = int(problem.grid.n_points * problem.ndof * problem.ndof * 288)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "source": identity,
        "run_id": run_id,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "configuration_sha256": configuration_hash(config),
        "checkpoint_identity_sha256": checkpoint_identity.identity_sha256,
        "checkpoint_schema": CHECKPOINT_SCHEMA,
        "checkpoint_path": str(checkpoint.resolve()),
        "outputs": {"json": str(output_json.resolve()), "hdf5": str(output_hdf5.resolve())},
        "dimensions": {"n_frequency": problem.grid.n_points, "ndof": problem.ndof, "nnz": problem.vertex.nnz},
        "predicted_peak_memory_bytes": predicted,
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
    run_id: str, resume: bool = True,
) -> dict[str, Any]:
    identity, problem, config = authenticate_scba_inputs(
        fixture=fixture, fc3_full=fc3_full, phase_b_artifacts=phase_b_artifacts
    )
    from negfpy.anharmonic._checkpoint_v2 import make_scba_checkpoint_identity_v2
    expected = make_scba_checkpoint_identity_v2(
        run_id, problem, config, environment_sha256=environment_sha256,
        source_hashes=_source_hashes(identity), source_tree_sha256=source_tree_sha256,
    )
    start = time.perf_counter()
    result = None
    scientific_error = None
    try:
        result = solve_sparse_scba_checkpointed(
            problem, config, checkpoint_path=checkpoint, run_id=run_id,
            environment_sha256=environment_sha256, source_hashes=_source_hashes(identity),
            source_tree_sha256=source_tree_sha256, resume=resume,
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
        "configuration_sha256": configuration_hash(config), "result": summary,
        "timings_seconds": {"total": time.perf_counter() - start},
        "scba_executed": True, "release_certified": False,
    }
    hdf5_sha = _write_result_hdf5(output_hdf5, metadata, arrays)
    outcome = {
        "schema": SCHEMA, "campaign_manifest_sha256": campaign_manifest_sha256,
        "preflight_source": identity, "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256_file(checkpoint), "checkpoint_status": load_scba_checkpoint_v2(checkpoint, expected_identity=expected).status,
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
                run_id=args.run_id,
            )
        else:
            if args.campaign_manifest_sha256 is None:
                raise SCBACaseError("--campaign-manifest-sha256 is required for execution")
            result = run_scba_case(
                fixture=args.fixture, fc3_full=args.fc3_full, phase_b_artifacts=args.phase_b_artifacts,
                checkpoint=args.checkpoint, output_json=args.output_json, output_hdf5=args.output_hdf5,
                campaign_manifest_sha256=args.campaign_manifest_sha256,
                source_tree_sha256=args.source_tree_sha256, environment_sha256=args.environment_sha256,
                run_id=args.run_id, resume=not args.no_resume,
            )
        print(json.dumps(_plain(result), sort_keys=True, ensure_ascii=True, separators=(",", ":")))
        return 0 if result.get("result", {}).get("converged", result.get("passed", False)) else 2
    except SCBACaseError as error:
        print(f"SCBA case infrastructure error: {error}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
