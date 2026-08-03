#!/usr/bin/env python3
"""Run the sealed silicon Nq=1, Nw=17 one-shot Born resource gate.

This is a private development gate between the certified Phase-B FC3 mapping
and any material SCBA campaign.  It consumes the immutable Phase-B sparse
vertex and never maps FC3 again.  The calculation has exactly one contact-only
Born bubble followed by exactly one updated Dyson/Keldysh solve.  It is a
timing, memory, and execution-semantics gate, not a conserving calculation and
never a release certificate.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from negfpy.anharmonic import (
    SCBAConfig,
    SignedFrequencyGrid,
    SolverScale,
    build_layered_device_mapping,
    build_layered_harmonic_model,
    read_phono3py_crystal_mapping,
    read_phono3py_fc2,
    solve_one_shot_born,
)
from negfpy.anharmonic._canonical import canonical_encode, domain_digest
from negfpy.anharmonic._provenance import (
    configuration_hash,
    problem_input_hashes,
)
from negfpy.anharmonic.phase_a_fixture import (
    RAW_FORCE_CONSTANT_SHA256,
    verify_phase_a_fixture,
)
from negfpy.anharmonic.solver import _guard_dense_memory

import validate_sparse_nq1_phase_b as phase_b


SCHEMA = "negfpy.sparse-nq1-phase-c-born-case.v1"
ARCHIVE_SCHEMA = "negfpy.sparse-nq1-phase-c-born-fields.v1"
PREFLIGHT_SCHEMA = "negfpy.sparse-nq1-phase-c-born-preflight.v1"
PHASE_B_SET_SCHEMA = "negfpy.sparse-nq1-phase-b-input-set.v1"
EXPECTED_PHASE_B_SHA256SUMS_SHA256 = (
    "c0c6c99bfd3b68ed5ce022bb7d70d96dd67117ae765b133ab870635d3fbdab20"
)
EXPECTED_PHASE_B_SOURCE_COMMIT = "cc8f5a7befa4124d48018ca5227af6ae278eed66"
EXPECTED_PHASE_B_SOURCE_TREE = "0868a59744b1d81c96c3ab8827ad8f0aae2db9ab"
EXPECTED_PHASE_B_MAPPED_HDF5_SHA256 = (
    "06b3294ecc81ddad5399f4ebe0ae226eab406c1c06c4fef277c76420a0865101"
)
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
TRANSPORT_TRANSFORM = np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int64
)
HBAR_J_S = 1.0545718176461565e-34
KB_J_K = 1.380649e-23
OMEGA0_RAD_S = 2.0 * np.pi * 1.0e13
TEMPERATURE_K = 300.0
MEAN_TEMPERATURE = KB_J_K * TEMPERATURE_K / (HBAR_J_S * OMEGA0_RAD_S)
TEMPERATURE_BIAS = 0.0
COUPLING_SCALE = 1.0
GRID_STEP = 0.6
GRID_N_POSITIVE = 8
GRID_N_POINTS = 17
N_DEVICE_DOFS = 216
ETA_LEAD = 0.025
MEMORY_FRACTION = 0.70
MIN_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
MIN_OUTPUT_FREE_BYTES = 2 * 1024**3
EXPECTED_PREDICTED_MEMORY_BYTES = 755_011_856
EXPECTED_SPARSE_NNZ = 406_080
EXPECTED_SCHEDULE_ACTIVE_DOFS = 120
EXPECTED_SCHEDULE_STORAGE_BYTES = 11_491_492
EXPECTED_PAIR_MULTIPLY_ADD_COUNT = 284_601_600
HARMONIC_1D_MAX_FREQUENCY = 1.5269764033
HARMONIC_CONSERVATIVE_BOUND = 2.463999
EXPECTED_RAW_FC3_SHA256 = RAW_FORCE_CONSTANT_SHA256["fc3.full.hdf5"]


class PhaseCError(RuntimeError):
    """A fail-closed Phase-C input, archive, or execution error."""


class BornScientificFailure(RuntimeError):
    """A numerical Born failure that is archived as a scientific outcome."""


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: Any) -> str:
    return hashlib.sha256(canonical_encode(np.asarray(values))).hexdigest()


def _validate_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PhaseCError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PhaseCError(f"immutable artifact already exists: {path}")
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PhaseCError(f"immutable artifact raced: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size <= 256:
            return _plain(array.tolist())
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "canonical_sha256": _array_sha256(array),
        }
    if dataclasses.is_dataclass(value):
        return {
            field.name: _plain(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping) or hasattr(value, "items"):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_plain(item) for item in value]
        return sorted(encoded, key=lambda item: _canonical_json({"item": item}))
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0.0 else "-Infinity"
    if isinstance(value, complex):
        return {"real": _plain(value.real), "imag": _plain(value.imag)}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Unsupported Phase-C report value {type(value).__name__}.")


def _regular_file(path: Path, *, name: str) -> Path:
    original = Path(path)
    if original.is_symlink():
        raise PhaseCError(f"{name} must not be a symlink: {original}")
    resolved = original.resolve()
    if not resolved.is_file():
        raise PhaseCError(f"{name} is not a regular non-symlink file: {resolved}")
    return resolved


def _read_canonical_json(path: Path, *, name: str) -> dict[str, Any]:
    regular = _regular_file(path, name=name)
    try:
        value = json.loads(regular.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCError(f"invalid {name}: {error}") from error
    if not isinstance(value, dict) or regular.read_bytes() != _canonical_json(value):
        raise PhaseCError(f"{name} is not a canonical JSON object.")
    return value


def _verify_sha256sums(root: Path) -> dict[str, str]:
    sums_path = _regular_file(root / "SHA256SUMS", name="Phase-B SHA256SUMS")
    records: dict[str, str] = {}
    try:
        lines = sums_path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise PhaseCError(f"cannot read Phase-B SHA256SUMS: {error}") from error
    for line in lines:
        digest, separator, relative = line.partition("  ")
        _validate_sha256(digest, name="Phase-B checksum")
        if separator != "  " or not relative or relative in records:
            raise PhaseCError("Phase-B SHA256SUMS has an invalid or duplicate line.")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise PhaseCError("Phase-B SHA256SUMS contains an unsafe path.")
        target = _regular_file(root / candidate, name=f"Phase-B artifact {relative}")
        try:
            target.relative_to(root)
        except ValueError as error:
            raise PhaseCError("Phase-B checksum path escapes its artifact root.") from error
        if _sha256_file(target) != digest:
            raise PhaseCError(f"Phase-B checksum mismatch for {relative}.")
        records[relative] = digest
    required = {
        "campaign-manifest.json",
        "logs/phase-b-mapping.json",
        "outcomes/phase-b-mapped-fc3.h5",
        "outcomes/phase-b-mapping.json",
        "phase-b-report.json",
    }
    if set(records) != required:
        raise PhaseCError(
            "Phase-B SHA256SUMS inventory is not exact: "
            f"{sorted(records)!r}."
        )
    return records


def verify_phase_b_artifact_set(root: Path) -> dict[str, Any]:
    """Authenticate the complete terminal Phase-B artifact set."""

    original_root = Path(root)
    if original_root.is_symlink():
        raise PhaseCError(f"Phase-B artifact root must not be a symlink: {original_root}")
    root = original_root.resolve()
    if not root.is_dir():
        raise PhaseCError(f"Phase-B artifact root is invalid: {root}")
    records = _verify_sha256sums(root)
    sha256sums_sha256 = _sha256_file(root / "SHA256SUMS")
    if sha256sums_sha256 != EXPECTED_PHASE_B_SHA256SUMS_SHA256:
        raise PhaseCError(
            "Phase-B terminal SHA256SUMS is not the certified artifact set: "
            f"{sha256sums_sha256} != {EXPECTED_PHASE_B_SHA256SUMS_SHA256}."
        )
    manifest = _read_canonical_json(
        root / "campaign-manifest.json", name="Phase-B campaign manifest"
    )
    outcome = _read_canonical_json(
        root / "outcomes/phase-b-mapping.json", name="Phase-B mapping outcome"
    )
    report = _read_canonical_json(root / "phase-b-report.json", name="Phase-B report")
    status = _read_canonical_json(root / "status.json", name="Phase-B status")
    hdf5_path = root / "outcomes/phase-b-mapped-fc3.h5"
    archive = phase_b.verify_phase_b_hdf5(hdf5_path)
    if archive["sha256"] != EXPECTED_PHASE_B_MAPPED_HDF5_SHA256:
        raise PhaseCError(
            "Phase-B mapped HDF5 is not the certified artifact: "
            f"{archive['sha256']} != {EXPECTED_PHASE_B_MAPPED_HDF5_SHA256}."
        )
    source = manifest.get("source", {})
    if (
        source.get("commit") != EXPECTED_PHASE_B_SOURCE_COMMIT
        or source.get("tree") != EXPECTED_PHASE_B_SOURCE_TREE
        or source.get("dirty") is not False
    ):
        raise PhaseCError("Phase-B source commit/tree is not the certified mapping source.")
    manifest_sha256 = records["campaign-manifest.json"]
    if (
        manifest.get("schema") != "negfpy.sparse-nq1-phase-b-campaign.v1"
        or outcome.get("campaign_manifest_sha256") != manifest_sha256
        or report.get("campaign_manifest_sha256") != manifest_sha256
        or status.get("campaign_manifest_sha256") != manifest_sha256
    ):
        raise PhaseCError("Phase-B campaign-manifest binding does not reproduce.")
    if (
        status.get("state") != "passed"
        or not status.get("phase_b_certified")
        or not report.get("phase_b_certified")
        or not outcome.get("phase_b_certified")
        or not archive["phase_b_certified"]
    ):
        raise PhaseCError("Phase-B artifact set is not scientifically certified.")
    if any(
        bool(value.get("release_certified")) or bool(value.get("scba_executed"))
        for value in (status, report, outcome, archive)
    ):
        raise PhaseCError("Phase-B artifact set overclaims SCBA or release status.")
    if (
        records["outcomes/phase-b-mapped-fc3.h5"] != archive["sha256"]
        or outcome.get("hdf5", {}).get("sha256") != archive["sha256"]
        or report.get("mapped_fc3_hdf5_sha256") != archive["sha256"]
    ):
        raise PhaseCError("Phase-B mapped-HDF5 identity does not reproduce.")
    binding = domain_digest(
        "NEGFPY-PHASE-B-TERMINAL-ARTIFACT-SET",
        {
            "records": dict(sorted(records.items())),
            "sha256sums_sha256": sha256sums_sha256,
            "status_sha256": _sha256_file(root / "status.json"),
        },
    )
    return {
        "schema": PHASE_B_SET_SCHEMA,
        "root": str(root),
        "records": dict(sorted(records.items())),
        "sha256sums_sha256": sha256sums_sha256,
        "status_sha256": _sha256_file(root / "status.json"),
        "artifact_set_sha256": binding,
        "campaign_manifest_sha256": manifest_sha256,
        "source": source,
        "archive": archive,
        "hdf5_path": str(hdf5_path.resolve()),
    }


def _thread_runtime() -> list[dict[str, Any]]:
    try:
        from threadpoolctl import threadpool_info
    except ImportError as error:  # pragma: no cover - locked environment gate
        raise PhaseCError("Phase C requires threadpoolctl.") from error
    return [
        {
            "internal_api": item.get("internal_api"),
            "prefix": item.get("prefix"),
            "version": item.get("version"),
            "num_threads": int(item.get("num_threads", 0)),
        }
        for item in threadpool_info()
    ]


def _verify_thread_limits() -> dict[str, Any]:
    environment = {name: os.environ.get(name) for name in THREAD_LIMIT_ENV}
    wrong = {name: value for name, value in environment.items() if value != "1"}
    if wrong:
        raise PhaseCError(f"thread-limit environment is not exact: {wrong!r}")
    pools = _thread_runtime()
    if any(item["num_threads"] != 1 for item in pools):
        raise PhaseCError(f"native thread pool is not single-threaded: {pools!r}")
    return {"environment": environment, "runtime_pools": pools, "passed": True}


def _available_memory_bytes() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _probe_atomic_hdf5(directory: Path) -> None:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - locked environment gate
        raise PhaseCError("Phase C requires h5py.") from error
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".negfpy-phase-c-probe.", suffix=".h5", dir=directory
    )
    os.close(descriptor)
    probe = Path(name)
    published = probe.with_suffix(".published.h5")
    try:
        metadata = {
            "schema": "negfpy.sparse-nq1-phase-c-publication-probe.v1",
            "diagnostic_nonfinite": _plain(float("inf")),
        }
        encoded = _canonical_json(metadata)
        with h5py.File(probe, "w", libver="latest") as handle:
            handle.attrs["schema"] = metadata["schema"]
            handle.attrs["release_certified"] = False
            handle.create_dataset(
                "grid/omega",
                data=np.array([-1.0, 0.0, 1.0], dtype=np.float64),
                track_times=False,
            )
            handle.create_dataset(
                "green/retarded",
                data=np.eye(2, dtype=np.complex128)[None, ...],
                track_times=False,
            )
            handle.create_dataset(
                "metadata_json",
                data=np.frombuffer(encoded, dtype=np.uint8),
                track_times=False,
            )
            handle.flush()
        with probe.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(probe, published)
        except FileExistsError as error:
            raise PhaseCError("HDF5 publication probe path raced.") from error
        _fsync_directory(directory)
        with h5py.File(published, "r") as handle:
            recovered = bytes(
                np.asarray(handle["metadata_json"][...], dtype=np.uint8)
            )
            if (
                str(handle.attrs.get("schema", "")) != metadata["schema"]
                or bool(handle.attrs.get("release_certified", True))
                or not np.array_equal(
                    handle["grid/omega"][...], np.array([-1.0, 0.0, 1.0])
                )
                or handle["green/retarded"].shape != (1, 2, 2)
                or recovered != encoded
                or json.loads(recovered.decode("ascii")) != metadata
            ):
                raise PhaseCError("HDF5 publication probe returned corrupt data.")
    finally:
        published.unlink(missing_ok=True)
        probe.unlink(missing_ok=True)


def _solver_config() -> SCBAConfig:
    return SCBAConfig(
        eta_lead=ETA_LEAD,
        retarded_strategy="causal_hilbert",
        hilbert_backend="direct",
        backend="fft",
        max_workers=1,
        eta_device=0.0,
        max_memory_fraction=MEMORY_FRACTION,
        allow_memory_overcommit=False,
    )


def _load_harmonic_model(fixture: Path) -> Any:
    yaml_path = fixture / "phono3py_disp.yaml"
    mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc2")
    fc2 = read_phono3py_fc2(
        fixture / "fc2.full.hdf5",
        mapping.supercell.masses,
        p2s_map=mapping.p2s,
        s2p_map=mapping.s2p,
        compact=False,
    )
    layered = build_layered_device_mapping(
        mapping,
        fc2,
        transport_supercell_matrix=TRANSPORT_TRANSFORM,
    )
    scale = SolverScale(
        omega0=OMEGA0_RAD_S,
        hbar=HBAR_J_S,
        metadata={
            "phase": "sparse_nq1_phase_b_mapping_only",
            "source_units": {
                "fc2": "eV/angstrom^2",
                "fc3": "eV/angstrom^3",
                "mass": "amu",
            },
        },
    )
    return build_layered_harmonic_model(layered, scale)


def _assemble_problem(fixture: Path, phase_b_set: Mapping[str, Any]) -> Any:
    # Imported lazily so this private driver cannot accidentally acquire a
    # fallback that remaps FC3.  There is exactly one allowed assembly path.
    try:
        from negfpy.anharmonic._material_nq1_problem import (
            build_material_nq1_problem_from_phase_b,
        )
    except ImportError as error:  # pragma: no cover - integration guard
        raise PhaseCError("Phase-B material-problem loader is unavailable.") from error
    model = _load_harmonic_model(fixture)
    archive = phase_b_set["archive"]
    vertex = phase_b.load_phase_b_sparse_vertex(Path(phase_b_set["hdf5_path"]))
    grid = SignedFrequencyGrid(step=GRID_STEP, n_positive=GRID_N_POSITIVE)
    return build_material_nq1_problem_from_phase_b(
        model,
        vertex,
        archive,
        grid=grid,
        mean_temperature=MEAN_TEMPERATURE,
        temperature_bias=TEMPERATURE_BIAS,
        coupling_scale=COUPLING_SCALE,
    )


def authenticate_phase_c_inputs(
    *, fixture: Path, fc3_full: Path, phase_b_artifacts: Path
) -> tuple[dict[str, Any], Any, SCBAConfig]:
    """Authenticate and assemble the fixed problem without evaluating a bubble."""

    fixture = fixture.resolve()
    fc3_full = _regular_file(fc3_full, name="sealed full FC3")
    fixture_report = verify_phase_a_fixture(fixture)
    raw_fc3_sha256 = _sha256_file(fc3_full)
    if raw_fc3_sha256 != EXPECTED_RAW_FC3_SHA256:
        raise PhaseCError(
            f"sealed full FC3 SHA256 mismatch: {raw_fc3_sha256} != "
            f"{EXPECTED_RAW_FC3_SHA256}"
        )
    phase_b_set = verify_phase_b_artifact_set(phase_b_artifacts)
    metadata = phase_b_set["archive"]["metadata"]
    if metadata.get("identities", {}).get("raw_fc3_sha256") != raw_fc3_sha256:
        raise PhaseCError("raw FC3 does not match the certified Phase-B mapping.")
    thread_limits = _verify_thread_limits()
    problem = _assemble_problem(fixture, phase_b_set)
    config = _solver_config()
    if (
        problem.grid.n_points != GRID_N_POINTS
        or problem.ndof != N_DEVICE_DOFS
        or problem.vertex.nnz != EXPECTED_SPARSE_NNZ
    ):
        raise PhaseCError(
            "fixed Phase-C dimensions do not reproduce "
            f"(Nw={problem.grid.n_points}, ndof={problem.ndof}, "
            f"nnz={problem.vertex.nnz})."
        )
    if problem.vertex.vertex_storage_sha256 == metadata["identities"][
        "sparse_vertex_storage_sha256"
    ]:
        raise PhaseCError(
            "Phase-C assembly did not bind its one explicit continuation stage."
        )
    vertex_metadata = problem.vertex.metadata
    if (
        vertex_metadata.get("phase_b_archive_sha256")
        != phase_b_set["archive"]["sha256"]
        or vertex_metadata.get("phase_b_campaign_manifest_sha256")
        != phase_b_set["campaign_manifest_sha256"]
        or vertex_metadata.get("cubic_continuation_factor") != COUPLING_SCALE
        or vertex_metadata.get("cubic_continuation_applied_count") != 1
    ):
        raise PhaseCError("assembled problem lacks exact Phase-B/continuation binding.")
    predicted_memory = _guard_dense_memory(problem, config)
    available = _available_memory_bytes()
    if available is None or available < MIN_AVAILABLE_MEMORY_BYTES:
        raise PhaseCError(
            f"at least {MIN_AVAILABLE_MEMORY_BYTES} available bytes are required."
        )
    if predicted_memory > MEMORY_FRACTION * available:
        raise PhaseCError(
            f"predicted {predicted_memory} bytes exceed {MEMORY_FRACTION:.0%} "
            f"of available memory {available}."
        )
    if predicted_memory != EXPECTED_PREDICTED_MEMORY_BYTES:
        raise PhaseCError(
            "fixed Phase-C memory model changed: "
            f"{predicted_memory} != {EXPECTED_PREDICTED_MEMORY_BYTES}."
        )
    from negfpy.anharmonic.sparse_cubic import _compiled_sparse_schedule

    schedule = _compiled_sparse_schedule(problem.vertex)
    fft_padding = 1 << (2 * problem.grid.n_points - 2).bit_length()
    schedule_identity = (
        int(len(schedule.active)),
        int(schedule.numeric_storage_bytes),
        int(schedule.pair_multiply_add_count),
        int(fft_padding),
    )
    expected_schedule = (
        EXPECTED_SCHEDULE_ACTIVE_DOFS,
        EXPECTED_SCHEDULE_STORAGE_BYTES,
        EXPECTED_PAIR_MULTIPLY_ADD_COUNT,
        64,
    )
    if schedule_identity != expected_schedule:
        raise PhaseCError(
            "fixed sparse schedule cost changed: "
            f"{schedule_identity!r} != {expected_schedule!r}."
        )
    input_hashes = problem_input_hashes(problem)
    controls = {
        "grid_step": GRID_STEP,
        "grid_n_positive": GRID_N_POSITIVE,
        "grid_n_points": GRID_N_POINTS,
        "grid_cutoff": problem.grid.cutoff,
        "temperature_kelvin": TEMPERATURE_K,
        "mean_temperature_solver": MEAN_TEMPERATURE,
        "temperature_bias_solver": TEMPERATURE_BIAS,
        "coupling_scale": COUPLING_SCALE,
        "eta_lead": ETA_LEAD,
        "eta_device": 0.0,
        "backend": "fft",
        "hilbert_backend": "direct",
        "retarded_strategy": "causal_hilbert",
        "memory_fraction": MEMORY_FRACTION,
        "nongating_harmonic_band_evidence": {
            "one_dimensional_max_frequency": HARMONIC_1D_MAX_FREQUENCY,
            "cutoff_to_one_dimensional_max": (
                problem.grid.cutoff / HARMONIC_1D_MAX_FREQUENCY
            ),
            "conservative_operator_bound": HARMONIC_CONSERVATIVE_BOUND,
            "cutoff_to_conservative_bound": (
                problem.grid.cutoff / HARMONIC_CONSERVATIVE_BOUND
            ),
        },
    }
    identity = {
        "fixture_path": str(fixture),
        "fixture": fixture_report,
        "fc3_full": {"path": str(fc3_full), "raw_sha256": raw_fc3_sha256},
        "phase_b": phase_b_set,
        "thread_limits": thread_limits,
        "controls": controls,
        "problem_input_hashes": input_hashes,
        "configuration_sha256": configuration_hash(config),
        "assembled_vertex": {
            "storage_sha256": problem.vertex.vertex_storage_sha256,
            "physics_sha256": problem.vertex.cubic_physics_sha256,
            "metadata_sha256": problem.vertex.metadata_sha256,
        },
        "dimensions": {
            "n_frequency": problem.grid.n_points,
            "ndof": problem.ndof,
            "nnz": problem.vertex.nnz,
        },
        "sparse_schedule": {
            "active_dofs": int(len(schedule.active)),
            "numeric_storage_bytes": int(schedule.numeric_storage_bytes),
            "pair_multiply_add_count": int(schedule.pair_multiply_add_count),
            "fft_padding": int(fft_padding),
            "two_bubble_pair_multiply_add_count": int(
                2 * fft_padding * schedule.pair_multiply_add_count
            ),
        },
        "memory": {
            "predicted_peak_bytes": predicted_memory,
            "available_bytes": available,
            "maximum_fraction": MEMORY_FRACTION,
            "guard_passed": True,
        },
    }
    return identity, problem, config


def preflight_phase_c_inputs(
    *,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    output_json: Path,
    output_hdf5: Path,
) -> dict[str, Any]:
    """Close all non-scientific gates without executing a Born bubble."""

    output_json = output_json.resolve()
    output_hdf5 = output_hdf5.resolve()
    if output_json == output_hdf5:
        raise PhaseCError("JSON and HDF5 outputs must be different paths.")
    if output_json.exists() or output_hdf5.exists():
        raise PhaseCError("Phase-C scientific output exists; refusing overwrite.")
    identity, _problem, _config = authenticate_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
    )
    for parent in {output_json.parent, output_hdf5.parent}:
        parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(parent).free
        if free < MIN_OUTPUT_FREE_BYTES:
            raise PhaseCError(f"output filesystem has only {free} free bytes at {parent}.")
        _probe_atomic_hdf5(parent)
    return {
        "schema": PREFLIGHT_SCHEMA,
        **identity,
        "outputs": {"json": str(output_json), "hdf5": str(output_hdf5)},
        "outputs_absent": True,
        "atomic_hdf5_probe_passed": True,
        "passed": True,
        "one_shot_born_executed": False,
        "scba_executed": False,
        "release_certified": False,
    }


def _result_arrays(result: Any) -> dict[str, np.ndarray]:
    arrays = {"grid/omega": np.asarray(result.grid.values)}
    for spectrum_name in ("green", "sigma_pp", "sigma_left", "sigma_right"):
        spectrum = getattr(result, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            arrays[f"{spectrum_name}/{component}"] = np.asarray(
                getattr(spectrum, component)
            )
    return arrays


def _result_summary(
    result: Any,
    *,
    expected_problem_sha256: str | None = None,
    expected_configuration_sha256: str | None = None,
) -> dict[str, Any]:
    arrays = _result_arrays(result)
    finite = all(np.all(np.isfinite(value)) for value in arrays.values())
    matrix_shape = (GRID_N_POINTS, N_DEVICE_DOFS, N_DEVICE_DOFS)
    exact_shapes = bool(
        arrays["grid/omega"].shape == (GRID_N_POINTS,)
        and all(
            value.shape == matrix_shape
            for name, value in arrays.items()
            if name != "grid/omega"
        )
    )
    checks = {
        "method_is_one_shot_born": result.method == "born",
        "one_updated_solve_semantics": bool(result.converged and not result.history),
        "never_release_certified": not result.release_certified,
        "sparse_fft_kernel": (
            result.metadata.get("cubic_kernel") == "sparse_zero_padded_fft"
            and result.metadata.get("effective_backend") == "fft"
        ),
        "single_worker_execution": (
            result.metadata.get("worker_count") == 1
            and result.metadata.get("requested_worker_count") == 1
        ),
        "predicted_memory_model": (
            result.metadata.get("predicted_peak_memory_bytes")
            == EXPECTED_PREDICTED_MEMORY_BYTES
        ),
        "problem_hash_bound_to_preflight": (
            expected_problem_sha256 is not None
            and result.metadata.get("input_hashes", {}).get("problem")
            == expected_problem_sha256
        ),
        "configuration_hash_bound_to_preflight": (
            expected_configuration_sha256 is not None
            and result.metadata.get("input_hashes", {}).get("configuration")
            == expected_configuration_sha256
        ),
        "fixed_signed_grid": result.grid.n_points == GRID_N_POINTS,
        "exact_field_shapes": exact_shapes,
        "finite_fields": bool(finite),
    }
    observations = {
        # These are deliberately recorded but are not gates: one-shot Born is
        # nonconserving, and Nw=17 is a timing grid rather than a refinement.
        "finite_currents": all(
            math.isfinite(float(getattr(result.currents, name)))
            for name in ("left", "right", "contact_balance", "collision_power")
        ),
        "diagnostics_passed": bool(result.diagnostics.passed),
    }
    return {
        "method": result.method,
        "converged": bool(result.converged),
        "history_length": len(result.history),
        "checks": checks,
        "nongating_observations": observations,
        "gate_passed": all(checks.values()),
        "array_hashes": {
            name: _array_sha256(value) for name, value in sorted(arrays.items())
        },
        "currents": _plain(result.currents),
        "diagnostics": _plain(result.diagnostics),
        "solver_metadata": _plain(result.metadata),
        "release_certified": False,
        "scba_executed": False,
    }


def _write_hdf5_once(
    path: Path,
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray] | None,
) -> str:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - locked environment gate
        raise PhaseCError("Phase C requires h5py.") from error
    if path.exists():
        raise PhaseCError(f"immutable HDF5 artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs["schema"] = ARCHIVE_SCHEMA
            handle.attrs["one_shot_born_executed"] = bool(
                metadata["one_shot_born_executed"]
            )
            handle.attrs["scba_executed"] = False
            handle.attrs["release_certified"] = False
            if arrays is not None:
                for dataset_name, values in sorted(arrays.items()):
                    handle.create_dataset(
                        dataset_name,
                        data=np.asarray(values),
                        track_times=False,
                    )
            encoded = _canonical_json(dict(metadata))
            handle.create_dataset(
                "metadata_json",
                data=np.frombuffer(encoded, dtype=np.uint8),
                track_times=False,
            )
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise PhaseCError(f"immutable HDF5 artifact raced: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def verify_phase_c_hdf5(path: Path) -> dict[str, Any]:
    """Verify a successful or scientifically failed Phase-C archive."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - locked environment gate
        raise PhaseCError("Phase C requires h5py.") from error
    path = _regular_file(path, name="Phase-C HDF5")
    with h5py.File(path, "r") as handle:
        if str(handle.attrs.get("schema", "")) != ARCHIVE_SCHEMA:
            raise PhaseCError("Phase-C HDF5 schema mismatch.")
        if bool(handle.attrs.get("scba_executed", True)) or bool(
            handle.attrs.get("release_certified", True)
        ):
            raise PhaseCError("Phase-C HDF5 overclaims SCBA or release status.")
        executed = bool(handle.attrs.get("one_shot_born_executed", False))
        metadata_bytes = bytes(np.asarray(handle["metadata_json"][...], dtype=np.uint8))
        arrays: dict[str, np.ndarray] = {}
        for group_name in ("grid", "green", "sigma_pp", "sigma_left", "sigma_right"):
            if group_name in handle:
                group = handle[group_name]
                for key in group:
                    arrays[f"{group_name}/{key}"] = np.asarray(group[key][...])
    try:
        metadata = json.loads(metadata_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseCError("Phase-C HDF5 metadata is invalid.") from error
    if metadata_bytes != _canonical_json(metadata):
        raise PhaseCError("Phase-C HDF5 metadata is not canonical.")
    if bool(metadata.get("one_shot_born_executed")) != executed:
        raise PhaseCError("Phase-C execution flag mismatch.")
    result_metadata = metadata.get("result")
    expected = (
        result_metadata.get("array_hashes", {})
        if isinstance(result_metadata, dict)
        else {}
    )
    if executed:
        required = {
            "grid/omega",
            *{
                f"{name}/{component}"
                for name in ("green", "sigma_pp", "sigma_left", "sigma_right")
                for component in ("retarded", "lesser", "greater")
            },
        }
        if set(arrays) != required or set(expected) != required:
            raise PhaseCError("Phase-C field inventory is not exact.")
        for name in sorted(required):
            if not np.all(np.isfinite(arrays[name])):
                raise PhaseCError(f"Phase-C field {name} is non-finite.")
            if _array_sha256(arrays[name]) != expected[name]:
                raise PhaseCError(f"Phase-C field hash mismatch for {name}.")
        matrix_shape = (GRID_N_POINTS, N_DEVICE_DOFS, N_DEVICE_DOFS)
        if arrays["grid/omega"].shape != (GRID_N_POINTS,) or any(
            arrays[name].shape != matrix_shape
            for name in required
            if name != "grid/omega"
        ):
            raise PhaseCError("Phase-C field shape is not exactly (17,216,216).")
    elif arrays:
        raise PhaseCError("scientifically failed Phase-C HDF5 contains partial fields.")
    return {
        "schema": ARCHIVE_SCHEMA,
        "sha256": _sha256_file(path),
        "one_shot_born_executed": executed,
        "scba_executed": False,
        "release_certified": False,
        "metadata": metadata,
    }


def _publish_outcome(
    *,
    output_json: Path,
    output_hdf5: Path,
    preflight: Mapping[str, Any],
    campaign_manifest_sha256: str,
    result: Any | None,
    scientific_error: Mapping[str, Any] | None,
    timings: Mapping[str, float],
) -> dict[str, Any]:
    result_summary = (
        None
        if result is None
        else _result_summary(
            result,
            expected_problem_sha256=preflight["problem_input_hashes"]["problem"],
            expected_configuration_sha256=preflight["configuration_sha256"],
        )
    )
    executed = result is not None
    gate_passed = bool(result_summary and result_summary["gate_passed"])
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "phase_b_artifact_set_sha256": preflight["phase_b"][
            "artifact_set_sha256"
        ],
        "phase_b_mapped_hdf5_sha256": preflight["phase_b"]["archive"]["sha256"],
        "problem_sha256": preflight["problem_input_hashes"]["problem"],
        "configuration_sha256": preflight["configuration_sha256"],
        "scientific_scope": "one contact-only Born bubble and one updated solve",
        "one_shot_born_executed": executed,
        "phase_c_gate_passed": gate_passed,
        "scientific_error": scientific_error,
        "result": result_summary,
        "timings_seconds": dict(timings),
        "scba_executed": False,
        "release_certified": False,
    }
    arrays = None if result is None else _result_arrays(result)
    hdf5_sha256 = _write_hdf5_once(
        output_hdf5.resolve(), metadata=metadata, arrays=arrays
    )
    verified = verify_phase_c_hdf5(output_hdf5)
    if verified["sha256"] != hdf5_sha256:
        raise PhaseCError("Phase-C HDF5 changed immediately after publication.")
    max_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_bytes = max_rss_raw * 1024 if sys.platform.startswith("linux") else max_rss_raw
    outcome = {
        "schema": SCHEMA,
        "campaign_manifest_sha256": campaign_manifest_sha256,
        "scientific_scope": (
            "Nq=1,Nw=17 one-shot Born timing/memory gate only; nonconserving, "
            "not SCBA and not a release certificate"
        ),
        "preflight": preflight,
        "result": result_summary,
        "scientific_error": scientific_error,
        "timings_seconds": dict(timings),
        "resources": {
            "max_rss_raw": max_rss_raw,
            "max_rss_bytes": max_rss_bytes,
            "platform": platform.platform(),
        },
        "hdf5": {
            "path": str(output_hdf5.resolve()),
            "sha256": hdf5_sha256,
            "schema": ARCHIVE_SCHEMA,
        },
        "one_shot_born_executed": executed,
        "phase_c_gate_passed": gate_passed,
        "scba_executed": False,
        "release_certified": False,
        "not_certified": [
            "SCBA fixed point",
            "interacting-lead self-energy",
            "material-device conductance",
            "0.3.0 release",
        ],
    }
    _publish_once(output_json.resolve(), _canonical_json(outcome))
    return outcome


def run_phase_c_born(
    *,
    fixture: Path,
    fc3_full: Path,
    phase_b_artifacts: Path,
    output_json: Path,
    output_hdf5: Path,
    campaign_manifest_sha256: str,
) -> dict[str, Any]:
    """Run exactly one Born calculation and archive a terminal outcome."""

    campaign_hash = _validate_sha256(
        campaign_manifest_sha256, name="campaign_manifest_sha256"
    )
    start = time.perf_counter()
    preflight = preflight_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
        output_json=output_json,
        output_hdf5=output_hdf5,
    )
    preflight_elapsed = time.perf_counter() - start
    # Reassemble from the already authenticated immutable inputs.  This is
    # intentionally cheap relative to the bubble and still performs no FC3 map.
    _identity, problem, config = authenticate_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3_full,
        phase_b_artifacts=phase_b_artifacts,
    )
    solve_start = time.perf_counter()
    result = None
    scientific_error = None
    try:
        result = solve_one_shot_born(problem, config)
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        scientific_error = {
            "classification": "scientific-numerical-failure",
            "type": type(error).__name__,
            "message": str(error),
        }
    solve_elapsed = time.perf_counter() - solve_start
    return _publish_outcome(
        output_json=output_json,
        output_hdf5=output_hdf5,
        preflight=preflight,
        campaign_manifest_sha256=campaign_hash,
        result=result,
        scientific_error=scientific_error,
        timings={
            "preflight": preflight_elapsed,
            "one_shot_born": solve_elapsed,
            "total_before_archive": time.perf_counter() - start,
        },
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--phase-b-artifacts", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-hdf5", type=Path, required=True)
    parser.add_argument("--campaign-manifest-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    try:
        if arguments.preflight_only:
            if arguments.campaign_manifest_sha256 is not None:
                raise PhaseCError(
                    "--preflight-only does not accept --campaign-manifest-sha256."
                )
            result = preflight_phase_c_inputs(
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                phase_b_artifacts=arguments.phase_b_artifacts,
                output_json=arguments.output_json,
                output_hdf5=arguments.output_hdf5,
            )
        else:
            if arguments.campaign_manifest_sha256 is None:
                raise PhaseCError("--campaign-manifest-sha256 is required.")
            result = run_phase_c_born(
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                phase_b_artifacts=arguments.phase_b_artifacts,
                output_json=arguments.output_json,
                output_hdf5=arguments.output_hdf5,
                campaign_manifest_sha256=arguments.campaign_manifest_sha256,
            )
    except (PhaseCError, OSError, MemoryError) as error:
        print(f"phase-c infrastructure failure: {error}", file=sys.stderr)
        return 3
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if bool(result.get("passed", result.get("phase_c_gate_passed"))) else 2


if __name__ == "__main__":
    raise SystemExit(main())
