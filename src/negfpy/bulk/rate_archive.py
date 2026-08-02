"""Small, self-contained archives for portable three-phonon rate spectra."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from negfpy.anharmonic.bulk_rta import BulkThreePhononRateSpectrum

RATE_SPECTRUM_HDF5_SCHEMA = "negfpy.bulk_three_phonon_rates.hdf5.v1"

_FIELDS = (
    "temperatures",
    "qpoints",
    "frequencies_thz",
    "grid_points",
    "band_indices",
    "gamma_thz_hwhm",
    "gamma_normal_thz_hwhm",
    "gamma_umklapp_thz_hwhm",
    "gamma_angular_fwhm",
    "decay_phase_space",
    "difference_phase_space",
)
_INTEGER_FIELDS = frozenset({"grid_points", "band_indices"})
_EXPECTED_DTYPES = {
    name: np.dtype(np.int64 if name in _INTEGER_FIELDS else np.float64)
    for name in _FIELDS
}
_ROOT_ATTRIBUTES = frozenset(
    {
        "schema",
        "cutoff_frequency_thz",
        "frequency_factor_to_thz",
        "physics_input_sha256",
        "metadata_json",
        "payload_sha256",
    }
)


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - isolated import test
        raise ImportError("Rate-spectrum archives require h5py.") from exc
    return h5py


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(
        f"Rate-spectrum metadata type {type(value).__name__!r} is not JSON-safe."
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(
    arrays: Mapping[str, np.ndarray],
    *,
    cutoff_frequency_thz: float,
    frequency_factor_to_thz: float,
    physics_input_sha256: str,
    metadata_json: str,
) -> str:
    digest = hashlib.sha256()
    digest.update((RATE_SPECTRUM_HDF5_SCHEMA + "\0").encode("ascii"))
    header = json.dumps(
        {
            "cutoff_frequency_thz": cutoff_frequency_thz,
            "frequency_factor_to_thz": frequency_factor_to_thz,
            "physics_input_sha256": physics_input_sha256,
            "metadata_json": metadata_json,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name in _FIELDS:
        array = np.ascontiguousarray(arrays[name])
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        for framed in (name.encode("ascii"), descriptor, array.tobytes(order="C")):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


def _arrays(result: BulkThreePhononRateSpectrum) -> dict[str, np.ndarray]:
    return {name: np.asarray(getattr(result, name)) for name in _FIELDS}


def write_rate_spectrum_hdf5(
    path: str | Path,
    result: BulkThreePhononRateSpectrum,
    *,
    overwrite: bool = False,
) -> Path:
    """Atomically write a compact rate spectrum with a payload digest."""

    if not isinstance(result, BulkThreePhononRateSpectrum):
        raise TypeError("result must be a BulkThreePhononRateSpectrum.")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    metadata_json = json.dumps(
        _json_value(result.metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    arrays = _arrays(result)
    payload = _payload_sha256(
        arrays,
        cutoff_frequency_thz=result.cutoff_frequency_thz,
        frequency_factor_to_thz=result.frequency_factor_to_thz,
        physics_input_sha256=result.physics_input_sha256,
        metadata_json=metadata_json,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    h5py = _h5py()
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = RATE_SPECTRUM_HDF5_SCHEMA
            handle.attrs["cutoff_frequency_thz"] = result.cutoff_frequency_thz
            handle.attrs["frequency_factor_to_thz"] = result.frequency_factor_to_thz
            handle.attrs["physics_input_sha256"] = result.physics_input_sha256
            handle.attrs["metadata_json"] = metadata_json
            handle.attrs["payload_sha256"] = payload
            for name, values in arrays.items():
                handle.create_dataset(
                    name,
                    data=values,
                    compression="gzip" if values.ndim > 1 else None,
                    shuffle=values.ndim > 1,
                )
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        if destination.exists() and not overwrite:
            raise FileExistsError(destination)
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def read_rate_spectrum_hdf5(path: str | Path) -> BulkThreePhononRateSpectrum:
    """Read an exact, self-contained rate archive and rerun all invariants."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("Rate-spectrum archive must not be a symbolic link.")
    source = candidate.resolve(strict=True)
    if not source.is_file():
        raise ValueError("Rate-spectrum archive must be a regular file.")
    before = _file_sha256(source)
    h5py = _h5py()
    with h5py.File(source, "r") as handle:
        if frozenset(str(name) for name in handle.attrs) != _ROOT_ATTRIBUTES:
            raise ValueError("Rate-spectrum root attributes are not exact.")
        if frozenset(str(name) for name in handle) != frozenset(_FIELDS):
            raise ValueError("Rate-spectrum dataset schema is not exact.")
        if str(handle.attrs["schema"]) != RATE_SPECTRUM_HDF5_SCHEMA:
            raise ValueError("Unsupported rate-spectrum archive schema.")
        arrays: dict[str, np.ndarray] = {}
        for name in _FIELDS:
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise ValueError(  # noqa: TRY004 - malformed archive content
                    f"Rate field {name!r} is not an internal hard link."
                )
            dataset = handle[name]
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(  # noqa: TRY004 - malformed archive content
                    f"Rate field {name!r} is not a dataset."
                )
            if np.dtype(dataset.dtype) != _EXPECTED_DTYPES[name]:
                raise ValueError(f"Rate field {name!r} has the wrong dtype.")
            if {str(key) for key in dataset.attrs}:
                raise ValueError(f"Rate field {name!r} has unexpected attributes.")
            arrays[name] = dataset[...]
        metadata_json = str(handle.attrs["metadata_json"])
        try:
            metadata = json.loads(metadata_json)
        except json.JSONDecodeError as exc:
            raise ValueError("Rate-spectrum metadata_json is invalid.") from exc
        cutoff = float(handle.attrs["cutoff_frequency_thz"])
        factor = float(handle.attrs["frequency_factor_to_thz"])
        identity = str(handle.attrs["physics_input_sha256"])
        stored_payload = str(handle.attrs["payload_sha256"])
    if _file_sha256(source) != before:
        raise RuntimeError("Rate-spectrum archive changed while it was read.")
    actual_payload = _payload_sha256(
        arrays,
        cutoff_frequency_thz=cutoff,
        frequency_factor_to_thz=factor,
        physics_input_sha256=identity,
        metadata_json=metadata_json,
    )
    if not hmac.compare_digest(stored_payload, actual_payload):
        raise ValueError("Rate-spectrum payload SHA256 does not match.")
    return BulkThreePhononRateSpectrum(
        **arrays,
        cutoff_frequency_thz=cutoff,
        frequency_factor_to_thz=factor,
        physics_input_sha256=identity,
        metadata=metadata,
    )


__all__ = [
    "RATE_SPECTRUM_HDF5_SCHEMA",
    "read_rate_spectrum_hdf5",
    "write_rate_spectrum_hdf5",
]
