#!/usr/bin/env python3
"""Run the bounded silicon FC3 finite-device Phase-B preflight.

This driver stops immediately after an authenticated nine-layer cubic mapping,
the real acoustic-lead audit, and conversion to ``SparseCubicVertex``.  It does
not construct a frequency grid and cannot call an SCBA solver.  The resulting
HDF5 file is therefore reusable input evidence for a later, separately sealed
one-bubble timing preflight without pretending that an anharmonic current was
calculated here.
"""

from __future__ import annotations

import argparse
import base64
from collections.abc import Mapping
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

from negfpy.anharmonic._canonical import (
    canonical_encode,
    fc3_content_sha256,
    primitive_crystal_content_sha256,
)
from negfpy.anharmonic.acoustic_contacts import classify_harmonic_lead
from negfpy.anharmonic.fc3 import validate_fc3
from negfpy.anharmonic.io.phono3py import read_phono3py_fc3
from negfpy.anharmonic.io.phono3py_crystal import (
    read_phono3py_crystal_mapping,
)
from negfpy.anharmonic.io.phono3py_fc2 import read_phono3py_fc2
from negfpy.anharmonic.layered import (
    build_layered_device_mapping,
    build_layered_harmonic_model,
)
from negfpy.anharmonic.mapped_fc3 import (
    map_fc3_to_device,
    mass_normalize_translational_fc3,
    scale_device_fc3,
)
from negfpy.anharmonic.phase_a_fixture import (
    RAW_FORCE_CONSTANT_SHA256,
    verify_phase_a_fixture,
)
from negfpy.anharmonic.translational_fc3 import convert_fc3_to_translational
from negfpy.anharmonic.types import SolverScale, estimate_dense_memory_bytes


SCHEMA = "negfpy.sparse-nq1-phase-b-case.v1"
ARCHIVE_SCHEMA = "negfpy.sparse-nq1-phase-b-mapped-fc3.v1"
PREFLIGHT_SCHEMA = "negfpy.sparse-nq1-phase-b-preflight.v1"
EXPECTED_RAW_FC3_SHA256 = RAW_FORCE_CONSTANT_SHA256["fc3.full.hdf5"]
EXPECTED_CANONICAL_FC3_SHA256 = (
    "6368368a4db9dcd324321fbc4d2d65519c2167e702641a098509afaf3de6c32e"
)
TRANSPORT_TRANSFORM = np.array(
    [[-1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int64
)
HBAR_J_S = 1.0545718176461565e-34
OMEGA0_RAD_S = 2.0 * np.pi * 1.0e13
N_DEVICE_LAYERS = 9
N_PRINCIPAL_LAYER_DOFS = 24
N_DEVICE_DOFS = N_DEVICE_LAYERS * N_PRINCIPAL_LAYER_DOFS
INTERACTION_LAYERS = (2, 3, 4, 5, 6)
OBSERVATION_LAYERS = (4,)
CONTACT_LAYERS = (0, 8)
HARMONIC_BUFFER_LAYERS = ((0, 1), (7, 8))
REQUIRED_HALO_LAYERS = 2
SIGNED_GRID_SIZES = (17, 41)
MIN_AVAILABLE_MEMORY_BYTES = 8 * 1024**3
MIN_OUTPUT_FREE_BYTES = 2 * 1024**3


class PhaseBError(RuntimeError):
    """A fail-closed Phase-B input, archive, or orchestration error."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(values: Any) -> str:
    return hashlib.sha256(canonical_encode(np.asarray(values))).hexdigest()


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


def _validate_sha256(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PhaseBError(f"{name} must be a lowercase SHA256 digest.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, content: bytes) -> None:
    """Atomically publish immutable bytes, accepting only identical replay."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise PhaseBError(f"immutable artifact mismatch: {path}")
        return
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise PhaseBError(f"immutable artifact raced: {path}")
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    except ImportError as error:  # pragma: no cover - environment gate
        raise PhaseBError("Phase B requires h5py.") from error
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".negfpy-phase-b-probe.", suffix=".h5", dir=directory
    )
    os.close(descriptor)
    probe = Path(name)
    try:
        with h5py.File(probe, "w") as handle:
            handle.create_dataset("probe", data=np.array([1], dtype=np.uint8))
            handle.flush()
        with probe.open("rb") as stream:
            os.fsync(stream.fileno())
        with h5py.File(probe, "r") as handle:
            if int(handle["probe"][0]) != 1:
                raise PhaseBError("HDF5 write/read probe returned corrupt data.")
    finally:
        probe.unlink(missing_ok=True)


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.size <= 256:
            return array.tolist()
        return {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "canonical_sha256": _array_sha256(array),
        }
    if dataclasses.is_dataclass(value):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        encoded = [_json_value(item) for item in value]
        return sorted(encoded, key=lambda item: _canonical_json({"item": item}))
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "encoding": "base64",
            "data": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _json_value(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"Unsupported Phase-B report value {type(value).__name__}.")


def _encode_sparse_metadata(value: Any) -> Any:
    """Encode frozen SparseCubicVertex metadata without losing Python types."""

    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Sparse vertex metadata cannot contain object arrays.")
        return {
            "__negfpy_type__": "ndarray",
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data_base64": base64.b64encode(array.tobytes(order="C")).decode("ascii"),
        }
    if isinstance(value, np.generic):
        return _encode_sparse_metadata(value.item())
    if isinstance(value, Mapping):
        return {
            "__negfpy_type__": "mapping",
            "items": {
                str(key): _encode_sparse_metadata(item)
                for key, item in value.items()
            },
        }
    if isinstance(value, (tuple, list)):
        return {
            "__negfpy_type__": "tuple",
            "items": [_encode_sparse_metadata(item) for item in value],
        }
    if isinstance(value, (set, frozenset)):
        items = [_encode_sparse_metadata(item) for item in value]
        items.sort(key=lambda item: _canonical_json({"item": item}))
        return {"__negfpy_type__": "frozenset", "items": items}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {
            "__negfpy_type__": "bytes",
            "data_base64": base64.b64encode(bytes(value)).decode("ascii"),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(
        f"Unsupported SparseCubicVertex metadata value {type(value).__name__}."
    )


def _decode_sparse_metadata(value: Any) -> Any:
    if not isinstance(value, dict) or "__negfpy_type__" not in value:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        raise PhaseBError("Sparse vertex metadata contains an untagged compound value.")
    kind = value.get("__negfpy_type__")
    if kind == "ndarray":
        try:
            dtype = np.dtype(value["dtype"])
            shape = tuple(int(item) for item in value["shape"])
            payload = base64.b64decode(value["data_base64"], validate=True)
            expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        except (KeyError, TypeError, ValueError) as error:
            raise PhaseBError("Invalid encoded sparse-metadata array.") from error
        if dtype.hasobject or expected != len(payload):
            raise PhaseBError("Encoded sparse-metadata array byte count is invalid.")
        return np.frombuffer(payload, dtype=dtype).reshape(shape).copy()
    if kind == "mapping":
        items = value.get("items")
        if not isinstance(items, dict):
            raise PhaseBError("Encoded sparse-metadata mapping is invalid.")
        return {str(key): _decode_sparse_metadata(item) for key, item in items.items()}
    if kind == "tuple":
        items = value.get("items")
        if not isinstance(items, list):
            raise PhaseBError("Encoded sparse-metadata tuple is invalid.")
        return tuple(_decode_sparse_metadata(item) for item in items)
    if kind == "frozenset":
        items = value.get("items")
        if not isinstance(items, list):
            raise PhaseBError("Encoded sparse-metadata frozenset is invalid.")
        return frozenset(_decode_sparse_metadata(item) for item in items)
    if kind == "bytes":
        try:
            return base64.b64decode(value["data_base64"], validate=True)
        except (KeyError, TypeError, ValueError) as error:
            raise PhaseBError("Encoded sparse-metadata bytes are invalid.") from error
    raise PhaseBError(f"Unknown sparse-metadata type tag {kind!r}.")


def _timed(timings: dict[str, float], name: str, operation: Any) -> Any:
    started = time.perf_counter()
    try:
        return operation()
    finally:
        timings[name] = time.perf_counter() - started


def _inspect_full_fc3_container(path: Path, *, n_supercell_atoms: int) -> dict[str, Any]:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment gate
        raise PhaseBError("Phase B requires h5py.") from error
    with h5py.File(path, "r") as handle:
        if "fc3" not in handle or not isinstance(handle["fc3"], h5py.Dataset):
            raise PhaseBError("sealed FC3 container has no regular 'fc3' dataset.")
        dataset = handle["fc3"]
        expected = (n_supercell_atoms,) * 3 + (3, 3, 3)
        if tuple(int(item) for item in dataset.shape) != expected:
            raise PhaseBError(
                f"sealed FC3 is not the required full tensor: {dataset.shape} != {expected}"
            )
        if dataset.dtype.kind not in "iuf" or dataset.dtype.kind == "b":
            raise PhaseBError("sealed FC3 dataset must be real numeric data.")
        return {
            "dataset": "fc3",
            "shape": list(expected),
            "dtype": str(dataset.dtype),
            "storage_size_bytes": int(dataset.id.get_storage_size()),
        }


def preflight_phase_b_inputs(
    *,
    fixture: Path,
    fc3_full: Path,
    output_json: Path,
    output_hdf5: Path,
) -> dict[str, Any]:
    """Authenticate inputs and exercise output mechanics before heavy mapping."""

    fixture = fixture.resolve()
    fc3_full = fc3_full.resolve()
    output_json = output_json.resolve()
    output_hdf5 = output_hdf5.resolve()
    if output_json == output_hdf5:
        raise PhaseBError("JSON and HDF5 outputs must be distinct paths.")
    if output_json.exists() or output_hdf5.exists():
        raise PhaseBError("Phase-B scientific outputs already exist; refusing overwrite.")
    fixture_report = verify_phase_a_fixture(fixture)
    if not fc3_full.is_file() or fc3_full.is_symlink():
        raise PhaseBError(f"sealed full FC3 must be a regular non-symlink file: {fc3_full}")
    raw_sha256 = _sha256_file(fc3_full)
    if raw_sha256 != EXPECTED_RAW_FC3_SHA256:
        raise PhaseBError(
            f"sealed full FC3 SHA256 mismatch: {raw_sha256} != {EXPECTED_RAW_FC3_SHA256}"
        )
    yaml_path = fixture / "phono3py_disp.yaml"
    fc2_mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc2")
    fc3_mapping = read_phono3py_crystal_mapping(yaml_path, purpose="fc3")
    primitive_fc2 = primitive_crystal_content_sha256(fc2_mapping)
    primitive_fc3 = primitive_crystal_content_sha256(fc3_mapping)
    if primitive_fc2 != primitive_fc3:
        raise PhaseBError("FC2 and FC3 mappings do not share one primitive crystal.")
    container = _inspect_full_fc3_container(
        fc3_full, n_supercell_atoms=fc3_mapping.n_supercell_atoms
    )
    available = _available_memory_bytes()
    if available is None or available < MIN_AVAILABLE_MEMORY_BYTES:
        raise PhaseBError(
            "available-memory preflight failed; at least "
            f"{MIN_AVAILABLE_MEMORY_BYTES} bytes are required"
        )
    for parent in {output_json.parent, output_hdf5.parent}:
        parent.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(parent).free
        if free < MIN_OUTPUT_FREE_BYTES:
            raise PhaseBError(
                f"output filesystem has only {free} free bytes at {parent}"
            )
        _probe_atomic_hdf5(parent)
    return {
        "schema": PREFLIGHT_SCHEMA,
        "fixture_path": str(fixture),
        "fixture": fixture_report,
        "fc3_full": {
            "path": str(fc3_full),
            "raw_sha256": raw_sha256,
            **container,
        },
        "primitive_crystal_content_sha256": primitive_fc2,
        "available_memory_bytes": available,
        "minimum_available_memory_bytes": MIN_AVAILABLE_MEMORY_BYTES,
        "minimum_output_free_bytes": MIN_OUTPUT_FREE_BYTES,
        "outputs_absent": True,
        "atomic_hdf5_probe_passed": True,
        "passed": True,
    }


def _write_hdf5_once(
    path: Path,
    *,
    indices: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray | None,
    metadata: Mapping[str, Any],
) -> str:
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment gate
        raise PhaseBError("Phase B requires h5py.") from error
    if path.exists():
        raise PhaseBError(f"immutable HDF5 artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs["schema"] = ARCHIVE_SCHEMA
            handle.attrs["phase_b_certified"] = bool(
                metadata["phase_b_certified"]
            )
            handle.attrs["release_certified"] = False
            handle.attrs["scba_executed"] = False
            handle.create_dataset(
                "indices", data=np.asarray(indices, dtype="<i8"), track_times=False
            )
            handle.create_dataset(
                "values", data=np.asarray(values, dtype="<f8"), track_times=False
            )
            if coefficients is not None:
                handle.create_dataset(
                    "solver_coefficients",
                    data=np.asarray(coefficients, dtype="<f8"),
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
            raise PhaseBError(f"immutable HDF5 artifact raced: {path}") from error
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return _sha256_file(path)


def verify_phase_b_hdf5(path: Path) -> dict[str, Any]:
    """Verify and summarize a Phase-B mapped-FC3 archive without SCBA."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment gate
        raise PhaseBError("Phase B requires h5py.") from error
    path = path.resolve()
    if not path.is_file() or path.is_symlink():
        raise PhaseBError(f"Phase-B HDF5 is not a regular file: {path}")
    with h5py.File(path, "r") as handle:
        allowed = {"indices", "values", "solver_coefficients", "metadata_json"}
        if set(handle) not in (
            {"indices", "values", "metadata_json"},
            allowed,
        ):
            raise PhaseBError("Phase-B HDF5 dataset inventory is not exact.")
        if str(handle.attrs.get("schema", "")) != ARCHIVE_SCHEMA:
            raise PhaseBError("Phase-B HDF5 schema mismatch.")
        if bool(handle.attrs.get("release_certified", True)):
            raise PhaseBError("Phase-B mapping evidence cannot certify a release.")
        if bool(handle.attrs.get("scba_executed", True)):
            raise PhaseBError("Phase-B archive falsely claims SCBA execution.")
        indices = np.asarray(handle["indices"][...])
        values = np.asarray(handle["values"][...])
        coefficients = (
            np.asarray(handle["solver_coefficients"][...])
            if "solver_coefficients" in handle
            else None
        )
        metadata_bytes = bytes(np.asarray(handle["metadata_json"][...], dtype=np.uint8))
    try:
        metadata = json.loads(metadata_bytes.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PhaseBError("Phase-B HDF5 metadata is invalid JSON.") from error
    if metadata_bytes != _canonical_json(metadata):
        raise PhaseBError("Phase-B HDF5 metadata is not canonical JSON.")
    if indices.ndim != 2 or indices.shape[1:] != (3,):
        raise PhaseBError("Phase-B indices must have shape (nnz, 3).")
    if values.shape != (len(indices),) or not np.all(np.isfinite(values)):
        raise PhaseBError("Phase-B physical values are invalid.")
    if coefficients is not None and (
        coefficients.shape != values.shape or not np.all(np.isfinite(coefficients))
    ):
        raise PhaseBError("Phase-B solver coefficients are invalid.")
    hashes = metadata.get("array_hashes", {})
    if hashes.get("indices") != _array_sha256(indices):
        raise PhaseBError("Phase-B index hash mismatch.")
    if hashes.get("values") != _array_sha256(values):
        raise PhaseBError("Phase-B physical-value hash mismatch.")
    expected_coefficients = hashes.get("solver_coefficients")
    if coefficients is None:
        if expected_coefficients is not None:
            raise PhaseBError("Phase-B solver coefficients are missing.")
    elif expected_coefficients != _array_sha256(coefficients):
        raise PhaseBError("Phase-B solver-coefficient hash mismatch.")
    if coefficients is not None:
        from negfpy.anharmonic.sparse_cubic import SparseCubicVertex

        encoded_metadata = metadata.get("sparse_vertex_metadata")
        if encoded_metadata is None:
            raise PhaseBError("Phase-B HDF5 lacks lossless sparse vertex metadata.")
        vertex = SparseCubicVertex(
            ndof=int(metadata["dimensions"]["ndof"]),
            indices=indices,
            coefficients=coefficients,
            metadata=_decode_sparse_metadata(encoded_metadata),
        )
        identities = metadata.get("identities", {})
        if vertex.vertex_storage_sha256 != identities.get(
            "sparse_vertex_storage_sha256"
        ):
            raise PhaseBError("Reconstructed sparse vertex storage hash mismatch.")
        if vertex.cubic_physics_sha256 != identities.get(
            "cubic_physics_sha256_solver"
        ):
            raise PhaseBError("Reconstructed sparse vertex physics hash mismatch.")
        if vertex.metadata_sha256 != identities.get("sparse_vertex_metadata_sha256"):
            raise PhaseBError("Reconstructed sparse vertex metadata hash mismatch.")
    return {
        "schema": ARCHIVE_SCHEMA,
        "sha256": _sha256_file(path),
        "ndof": int(metadata["dimensions"]["ndof"]),
        "nnz": int(len(values)),
        "phase_b_certified": bool(metadata["phase_b_certified"]),
        "release_certified": False,
        "scba_executed": False,
        "metadata": metadata,
    }


def load_phase_b_sparse_vertex(path: Path) -> Any:
    """Reconstruct the exact solver vertex from certified Phase-B evidence."""

    archive = verify_phase_b_hdf5(path)
    if not archive["phase_b_certified"]:
        raise PhaseBError("A scientifically failed Phase-B archive has no certified vertex.")
    try:
        import h5py
    except ImportError as error:  # pragma: no cover - environment gate
        raise PhaseBError("Phase B requires h5py.") from error
    from negfpy.anharmonic.sparse_cubic import SparseCubicVertex

    with h5py.File(Path(path).resolve(), "r") as handle:
        indices = np.asarray(handle["indices"][...])
        coefficients = np.asarray(handle["solver_coefficients"][...])
    metadata = archive["metadata"]
    return SparseCubicVertex(
        ndof=int(metadata["dimensions"]["ndof"]),
        indices=indices,
        coefficients=coefficients,
        metadata=_decode_sparse_metadata(metadata["sparse_vertex_metadata"]),
    )


def run_phase_b(
    *,
    fixture: Path,
    fc3_full: Path,
    output_json: Path,
    output_hdf5: Path,
    campaign_manifest_sha256: str,
) -> dict[str, Any]:
    """Execute the authenticated, nine-layer mapping and publish its evidence."""

    campaign_hash = _validate_sha256(
        campaign_manifest_sha256, name="campaign_manifest_sha256"
    )
    timings: dict[str, float] = {}
    total_start = time.perf_counter()
    preflight = _timed(
        timings,
        "input_preflight",
        lambda: preflight_phase_b_inputs(
            fixture=fixture,
            fc3_full=fc3_full,
            output_json=output_json,
            output_hdf5=output_hdf5,
        ),
    )
    yaml_path = fixture.resolve() / "phono3py_disp.yaml"
    mapping_fc3 = _timed(
        timings,
        "read_fc3_crystal_mapping",
        lambda: read_phono3py_crystal_mapping(yaml_path, purpose="fc3"),
    )
    raw_fc3 = _timed(
        timings,
        "read_full_fc3",
        lambda: read_phono3py_fc3(
            fc3_full,
            mapping_fc3.supercell.masses,
            p2s_map=mapping_fc3.p2s,
            s2p_map=mapping_fc3.s2p,
            compact=False,
        ),
    )
    canonical_fc3 = _timed(
        timings, "canonical_fc3_hash", lambda: fc3_content_sha256(raw_fc3)
    )
    if canonical_fc3 != EXPECTED_CANONICAL_FC3_SHA256:
        raise PhaseBError(
            "canonical full FC3 content mismatch: "
            f"{canonical_fc3} != {EXPECTED_CANONICAL_FC3_SHA256}"
        )
    raw_report = _timed(timings, "validate_raw_full_fc3", lambda: validate_fc3(raw_fc3))
    if not raw_report.passed or not raw_report.permutation_complete:
        raise PhaseBError(
            "sealed raw full FC3 did not pass its complete source validation."
        )
    translational = _timed(
        timings,
        "convert_r0_average_translational_fc3",
        lambda: convert_fc3_to_translational(
            raw_fc3, mapping_fc3, make_r0_average=True
        ),
    )
    if not translational.make_r0_average:
        raise PhaseBError("FC3 translation did not retain make_r0_average=True.")
    normalized = _timed(
        timings,
        "mass_normalize_fc3",
        lambda: mass_normalize_translational_fc3(
            translational,
            mapping_fc3,
            source_certification=raw_report,
        ),
    )

    mapping_fc2 = _timed(
        timings,
        "read_fc2_crystal_mapping",
        lambda: read_phono3py_crystal_mapping(yaml_path, purpose="fc2"),
    )
    fc2_path = fixture.resolve() / "fc2.full.hdf5"
    raw_fc2 = _timed(
        timings,
        "read_full_fc2",
        lambda: read_phono3py_fc2(
            fc2_path,
            mapping_fc2.supercell.masses,
            p2s_map=mapping_fc2.p2s,
            s2p_map=mapping_fc2.s2p,
            compact=False,
        ),
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
    layered_mapping = _timed(
        timings,
        "build_layered_fc2_mapping",
        lambda: build_layered_device_mapping(
            mapping_fc2,
            raw_fc2,
            transport_supercell_matrix=TRANSPORT_TRANSFORM,
        ),
    )
    harmonic = _timed(
        timings,
        "build_layered_harmonic_model",
        lambda: build_layered_harmonic_model(layered_mapping, scale),
    )
    mapped = _timed(
        timings,
        "map_r0_fc3_to_nine_layers",
        lambda: map_fc3_to_device(
            normalized,
            layered_mapping,
            n_device_layers=N_DEVICE_LAYERS,
            interaction_layers=INTERACTION_LAYERS,
            observation_layers=OBSERVATION_LAYERS,
        ),
    )
    lead = _timed(
        timings,
        "classify_real_acoustic_lead",
        lambda: classify_harmonic_lead(harmonic),
    )
    sparse = None
    if mapped.report.certification_passed:
        sparse = _timed(
            timings,
            "scale_sparse_cubic_vertex",
            lambda: scale_device_fc3(mapped, scale),
        )

    report = mapped.report
    checks = {
        "raw_file_sha256": preflight["fc3_full"]["raw_sha256"]
        == EXPECTED_RAW_FC3_SHA256,
        "canonical_fc3_sha256": canonical_fc3 == EXPECTED_CANONICAL_FC3_SHA256,
        "raw_fc3_complete_validation": bool(
            raw_report.passed and raw_report.permutation_complete
        ),
        "r0_average_enabled": bool(translational.make_r0_average),
        "nine_device_layers": report.n_device_layers == N_DEVICE_LAYERS,
        "silicon_principal_layer_dofs": (
            int(layered_mapping.n_principal_layer_dofs)
            == N_PRINCIPAL_LAYER_DOFS
        ),
        "silicon_total_device_dofs": int(mapped.ndof) == N_DEVICE_DOFS,
        "contact_layers": tuple(report.contact_layers) == CONTACT_LAYERS,
        "interaction_layers": tuple(report.interaction_layers) == INTERACTION_LAYERS,
        "observation_layers": tuple(report.observation_layers) == OBSERVATION_LAYERS,
        "contact_layers_count_as_harmonic_buffers": tuple(
            tuple(item) for item in report.harmonic_buffer_layers
        )
        == HARMONIC_BUFFER_LAYERS,
        "range_certification": bool(
            report.required_left_halo_layers == REQUIRED_HALO_LAYERS
            and report.required_right_halo_layers == REQUIRED_HALO_LAYERS
            and len(report.harmonic_buffer_layers[0])
            >= report.required_left_halo_layers
            and len(report.harmonic_buffer_layers[1])
            >= report.required_right_halo_layers
            and report.observation_layers[0] - report.interaction_layers[0]
            >= report.required_left_halo_layers
            and report.interaction_layers[-1] - report.observation_layers[-1]
            >= report.required_right_halo_layers
        ),
        "mapping_certification": bool(
            report.certification_passed and not report.failure_codes
        ),
        "permutation_support_complete": report.permutation_missing_count == 0,
        "permutation_roundoff_bound": bool(
            report.permutation_tolerance_passed
        ),
        "no_source_mutation_or_truncation": bool(
            not report.symmetrized
            and not report.corrected
            and not report.truncated
            and not report.thresholded
        ),
        "real_acoustic_lead": bool(
            lead.passed
            and lead.classification == "acoustic"
            and not lead.failure_reasons
        ),
        "sparse_vertex_scaled": bool(
            sparse is not None
            and sparse.ndof == mapped.ndof
            and sparse.nnz == mapped.nnz
        ),
    }
    diagnostics = {
        "permutation": {
            "exact_bitwise_equality": bool(report.exact_permutation_equality),
            "maximum_absolute_difference": float(report.permutation_max_abs),
            "maximum_roundoff_ratio": float(
                report.permutation_max_roundoff_ratio
            ),
            "roundoff_factor": float(report.permutation_roundoff_factor),
        }
    }
    certified = all(checks.values())
    coefficients = None if sparse is None else np.asarray(sparse.coefficients)
    dimensions = {
        "n_device_layers": N_DEVICE_LAYERS,
        "n_principal_layer_dofs": int(layered_mapping.n_principal_layer_dofs),
        "ndof": int(mapped.ndof),
        "nnz": int(mapped.nnz),
        "support_density": float(mapped.support_density),
    }
    array_hashes = {
        "indices": _array_sha256(mapped.indices),
        "values": _array_sha256(mapped.values),
        "solver_coefficients": (
            None if coefficients is None else _array_sha256(coefficients)
        ),
    }
    timings["total_before_archive"] = time.perf_counter() - total_start
    max_rss_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    max_rss_bytes = max_rss_raw * 1024 if sys.platform.startswith("linux") else max_rss_raw
    metadata = {
        "schema": ARCHIVE_SCHEMA,
        "campaign_manifest_sha256": campaign_hash,
        "phase_b_certified": certified,
        "release_certified": False,
        "scba_executed": False,
        "method": "authenticated_r0_fc3_finite_mapping_preflight",
        "configuration": {
            "transport_transform": TRANSPORT_TRANSFORM.tolist(),
            "omega0_rad_s": OMEGA0_RAD_S,
            "hbar_j_s": HBAR_J_S,
            "n_device_layers": N_DEVICE_LAYERS,
            "contact_layers": list(CONTACT_LAYERS),
            "interaction_layers": list(INTERACTION_LAYERS),
            "observation_layers": list(OBSERVATION_LAYERS),
            "make_r0_average": True,
        },
        "dimensions": dimensions,
        "array_hashes": array_hashes,
        "identities": {
            "raw_fc3_sha256": EXPECTED_RAW_FC3_SHA256,
            "canonical_fc3_sha256": canonical_fc3,
            "source_translational_sha256": normalized.source_translational_sha256,
            "mass_normalized_fc3_sha256": normalized.content_sha256,
            "device_mapped_fc3_sha256": mapped.content_sha256,
            "mapping_report_sha256": report.report_sha256,
            "cubic_physics_sha256_physical": mapped.cubic_physics_sha256,
            "sparse_vertex_storage_sha256": (
                None if sparse is None else sparse.vertex_storage_sha256
            ),
            "sparse_vertex_metadata_sha256": (
                None if sparse is None else sparse.metadata_sha256
            ),
            "cubic_physics_sha256_solver": (
                None if sparse is None else sparse.cubic_physics_sha256
            ),
            "fc2_content_sha256": layered_mapping.fc2_content_sha256,
            "primitive_crystal_content_sha256": (
                layered_mapping.primitive_crystal_content_sha256
            ),
            "fc2_crystal_mapping_content_sha256": (
                layered_mapping.fc2_crystal_mapping_content_sha256
            ),
            "fc3_crystal_mapping_content_sha256": (
                normalized.fc3_crystal_mapping_content_sha256
            ),
            "layered_geometry_sha256": layered_mapping.layered_geometry_sha256,
        },
        "sparse_vertex_metadata_encoding": (
            None if sparse is None else "negfpy.lossless-metadata-json.v1"
        ),
        "sparse_vertex_metadata": (
            None if sparse is None else _encode_sparse_metadata(sparse.metadata)
        ),
        "reports": {
            "raw_fc3_validation": _json_value(raw_report),
            "device_mapping": _json_value(report),
            "acoustic_lead_classification": _json_value(lead),
        },
        "checks": checks,
        "diagnostics": diagnostics,
    }
    hdf5_sha256 = _timed(
        timings,
        "publish_hdf5",
        lambda: _write_hdf5_once(
            output_hdf5.resolve(),
            indices=np.asarray(mapped.indices),
            values=np.asarray(mapped.values),
            coefficients=coefficients,
            metadata=metadata,
        ),
    )
    archive_verification = _timed(
        timings,
        "verify_hdf5",
        lambda: verify_phase_b_hdf5(output_hdf5),
    )
    if archive_verification["sha256"] != hdf5_sha256:
        raise PhaseBError("Phase-B HDF5 changed immediately after publication.")
    timings["total"] = time.perf_counter() - total_start
    outcome = {
        "schema": SCHEMA,
        "campaign_manifest_sha256": campaign_hash,
        "scientific_scope": (
            "authenticated full-FC3 r0 finite mapping and acoustic-lead "
            "preflight only; no one-shot Born or SCBA calculation"
        ),
        "inputs": preflight,
        "configuration": metadata["configuration"],
        "dimensions": dimensions,
        "array_hashes": array_hashes,
        "identities": metadata["identities"],
        "raw_fc3_validation": _json_value(raw_report),
        "device_mapping_report": _json_value(report),
        "acoustic_lead_classification": _json_value(lead),
        "dense_scba_memory_estimates_bytes": {
            str(size): estimate_dense_memory_bytes(size, mapped.ndof)
            for size in SIGNED_GRID_SIZES
        },
        "timings_seconds": timings,
        "resources": {
            "max_rss_raw": max_rss_raw,
            "max_rss_bytes": max_rss_bytes,
            "max_rss_semantics": (
                "Linux ru_maxrss KiB converted to bytes"
                if sys.platform.startswith("linux")
                else "platform ru_maxrss bytes"
            ),
            "platform": platform.platform(),
        },
        "hdf5": {
            "path": str(output_hdf5.resolve()),
            "sha256": hdf5_sha256,
            "schema": ARCHIVE_SCHEMA,
        },
        "checks": checks,
        "diagnostics": diagnostics,
        "phase_b_certified": certified,
        "scba_executed": False,
        "release_certified": False,
        "not_certified": [
            "one-shot Born timing preflight",
            "N_q=1 anharmonic SCBA fixed point",
            "material-device heat current",
            "0.3.0 release",
        ],
    }
    _publish_once(output_json.resolve(), _canonical_json(outcome))
    return outcome


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
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
                raise PhaseBError(
                    "--preflight-only does not accept --campaign-manifest-sha256."
                )
            result = preflight_phase_b_inputs(
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                output_json=arguments.output_json,
                output_hdf5=arguments.output_hdf5,
            )
        else:
            if arguments.campaign_manifest_sha256 is None:
                raise PhaseBError(
                    "scientific execution requires --campaign-manifest-sha256."
                )
            result = run_phase_b(
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                output_json=arguments.output_json,
                output_hdf5=arguments.output_hdf5,
                campaign_manifest_sha256=arguments.campaign_manifest_sha256,
            )
    except (PhaseBError, OSError, ValueError, TypeError, MemoryError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    print(_canonical_json(result).decode("ascii"), end="")
    return 0 if bool(result.get("passed", result.get("phase_b_certified"))) else 2


if __name__ == "__main__":
    raise SystemExit(main())
