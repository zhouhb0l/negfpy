"""Immutable evaluated checkpoints retaining independent statistical/spectral data.

This distinct schema cannot load historical contour checkpoints as repaired
channels. Authentication covers all six arrays before any read-only mapping.
The saved Born image has no Dyson evaluation and is never an accepted restart.
"""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .transverse_channel_feedback import ChannelSpectrum
from .transverse_material_restart import (
    _arrays_verified,
    _json_verified,
)
from .transverse_material_restart import (
    _binding as _material_binding,
)
from .transverse_material_restart import (
    _record as _material_record,
)
from .transverse_packed import _finite, _integer

_COMPONENTS = ("retarded", "statistical", "spectral")
_SCHEMA = "negfpy.transverse-channel-scba-checkpoint.v1"
_REPRESENTATION = "retarded-statistical-spectral-v1"


@dataclass(frozen=True, slots=True)
class ChannelCheckpoint:
    sigma_input: ChannelSpectrum
    mapped_sigma: ChannelSpectrum
    record: dict
    binding: dict
    manifest: dict
    manifest_sha256: str


def _binding(binding):
    binding = _material_binding(binding)
    if (binding.get("representation") != _REPRESENTATION
            or binding.get("initial_state") != "fresh-zero"):
        raise ValueError("Channel checkpoint requires its representation and fresh-zero binding.")
    return binding


def _record(record, iteration):
    # Retain the original three contour Sigma and three current screens as the
    # only residual convergence gates; independent C/D ratios are diagnostic.
    record = _material_record(record, iteration)
    residuals = record.get("channel_residuals")
    if not isinstance(residuals, dict) or set(residuals) != {"statistical", "spectral"}:
        raise ValueError("Channel checkpoint requires statistical and spectral residual diagnostics.")
    for row in residuals.values():
        ratio = row.get("ratio") if isinstance(row, dict) else None
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0:
            raise ValueError("Channel residual ratios must be finite nonnegative real numbers.")
    if (type(record.get("input_sigma_is_zero")) is not bool
            or record["input_sigma_is_zero"] != (iteration == 1)):
        raise ValueError("Only fresh harmonic state1 has an exact-zero input flag.")
    return record


def _arrays(spectrum, shape, label):
    if not isinstance(spectrum, ChannelSpectrum):
        raise TypeError("Channel checkpoint fields must be ChannelSpectrum.")
    arrays = {}
    for component in _COMPONENTS:
        values = getattr(spectrum, component)
        if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("complex128")
                or values.ndim != 4 or min(values.shape) < 1
                or values.shape[-1] != values.shape[-2]):
            raise ValueError("Channel fields must be complex128 (Nq,Nw,A,A) arrays.")
        shape = values.shape if shape is None else shape
        if values.shape != shape:
            raise ValueError("Channel checkpoint field shapes differ.")
        _finite(values, label + "." + component)
        arrays[label + "_" + component] = values
    return shape, arrays


def save_channel_checkpoint(
    root, *, iteration, sigma_input, mapped_sigma, record, binding,
    max_bytes=8 * 1024**3, max_checkpoints=3,
):
    """Publish six arrays and manifest last in a new exclusive iteration path.

    ``None`` is the fresh harmonic state's exact-zero interaction, written from
    a scalar broadcast view without allocating three whole-grid zero arrays.
    Incomplete writes remain visible without a manifest; no state is replaced.
    """
    iteration = _integer(iteration, "iteration")
    limit = _integer(max_bytes, "max_bytes")
    count = _integer(max_checkpoints, "max_checkpoints")
    record, binding = _record(record, iteration), _binding(binding)
    if record["input_sigma_is_zero"] != (sigma_input is None):
        raise ValueError("Exact-zero record flag differs from the supplied input state.")
    shape, mapped = _arrays(mapped_sigma, None, "mapped_sigma")
    if sigma_input is None:
        # Every view aliases only this one zero scalar, never an image field.
        zero = np.broadcast_to(np.array(0j, dtype=np.complex128), shape)
        arrays = {"sigma_input_" + name: zero for name in _COMPONENTS}
    else:
        _, arrays = _arrays(sigma_input, shape, "sigma_input")
    arrays.update(mapped)
    if sum(value.nbytes + 65536 for value in arrays.values()) > limit:
        raise ValueError("Channel checkpoint fields exceed max_bytes.")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Channel checkpoint root must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    if sum(path.is_dir() for path in root.glob("iteration-*")) >= count:
        raise ValueError("Channel checkpoint count exceeds max_checkpoints.")
    directory = root / f"iteration-{iteration:06d}"
    directory.mkdir(exist_ok=False)
    (directory / "arrays").mkdir()
    descriptors = {}
    for name, values in arrays.items():
        path = directory / "arrays" / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        descriptors[name] = {
            "relative_path": "arrays/" + path.name, "shape": list(shape),
            "dtype": "complex128", "array_bytes": values.nbytes,
            "file_bytes": path.stat().st_size, "file_sha256": digest,
        }
    manifest = {
        "schema": _SCHEMA, "representation": _REPRESENTATION,
        "iteration": iteration, "binding": binding, "record": record,
        "artifacts": descriptors, "input_sigma_is_zero": sigma_input is None,
        "accepted_restart": False, "sigma_input_evaluated": True,
        "mapped_sigma_dyson_evaluated": False,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = directory / "manifest.json.partial"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, directory / "manifest.json")
    temporary.unlink()
    for target in (directory / "arrays", directory, root):
        descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"path": str(directory), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "manifest": manifest}


def load_channel_checkpoint(
    path, *, expected_manifest_sha256, expected_binding, max_total_bytes=8 * 1024**3,
):
    """Authenticate all six fields before mapping, without promoting the state.

    Source files must remain immutable while returned read-only mappings live.
    Fresh-zero input loads as three actual saved zero channel arrays, not None.
    """
    with ExitStack() as stack:
        manifest = _json_verified(stack, path, "manifest.json", expected_manifest_sha256)
        if (manifest.get("schema") != _SCHEMA
                or manifest.get("representation") != _REPRESENTATION
                or manifest.get("accepted_restart") is not False
                or manifest.get("sigma_input_evaluated") is not True
                or manifest.get("mapped_sigma_dyson_evaluated") is not False):
            raise ValueError("Invalid evaluated channel checkpoint manifest.")
        if _binding(manifest["binding"]) != _binding(expected_binding):
            raise ValueError("Channel checkpoint source, physics or parent binding differs.")
        iteration = _integer(manifest["iteration"], "iteration")
        record = _record(manifest["record"], iteration)
        if (type(manifest.get("input_sigma_is_zero")) is not bool
                or manifest["input_sigma_is_zero"] != record["input_sigma_is_zero"]):
            raise ValueError("Channel checkpoint exact-zero state semantics differ.")
        shape = tuple(manifest["artifacts"]["sigma_input_retarded"]["shape"])
        if (len(shape) != 4 or any(type(v) is not int or v < 1 for v in shape)
                or shape[-1] != shape[-2]):
            raise ValueError("Invalid channel checkpoint field shape.")
        expected = {label + "_" + name: (shape, "complex128")
                    for label in ("sigma_input", "mapped_sigma") for name in _COMPONENTS}
        arrays = _arrays_verified(stack, path, manifest["artifacts"], expected, max_total_bytes)
        for name, values in arrays.items():
            _finite(values, name)
            if manifest["input_sigma_is_zero"] and name.startswith("sigma_input_"):
                for index in np.ndindex(values.shape[:-2]):
                    if np.any(values[index] != 0):
                        raise ValueError("Fresh harmonic checkpoint input is not exact zero.")
        spectra = [ChannelSpectrum(*(arrays[label + "_" + name] for name in _COMPONENTS))
                   for label in ("sigma_input", "mapped_sigma")]
        return ChannelCheckpoint(*spectra, record, manifest["binding"], manifest,
                                 expected_manifest_sha256)


__all__ = ["ChannelCheckpoint", "load_channel_checkpoint", "save_channel_checkpoint"]
