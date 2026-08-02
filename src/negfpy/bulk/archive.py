"""Atomic, schema-checked archives for bulk SMRTA results."""

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

from negfpy.anharmonic.bulk_rta import RTAThermalConductivity

RTA_RESULT_HDF5_SCHEMA = "negfpy.rta_thermal_conductivity.hdf5.v1"

_ARRAY_FIELDS = (
    "temperatures",
    "qpoints",
    "frequencies_thz",
    "group_velocities_thz_angstrom",
    "heat_capacities_ev_per_k",
    "gamma_thz_hwhm",
    "population_lifetime_ps",
    "mean_free_path_angstrom",
    "mode_kappa_w_mk",
    "band_index_kappa_w_mk",
    "kappa_w_mk",
    "qpoint_weights",
    "included_mask",
)

_EXPECTED_DTYPES = {
    name: np.dtype(np.bool_ if name == "included_mask" else np.float64)
    for name in _ARRAY_FIELDS
}


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
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
        f"RTA metadata value of type {type(value).__name__!r} is not JSON-safe."
    )


def _h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - isolated import test
        raise ImportError(
            "RTA HDF5 archives require the core h5py dependency."
        ) from exc
    return h5py


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload_sha256(
    arrays: Mapping[str, Any],
    *,
    approximation: str,
    primitive_volume_angstrom3: float,
    cutoff_frequency_thz: float,
    metadata_json: str,
) -> str:
    digest = hashlib.sha256()
    digest.update((RTA_RESULT_HDF5_SCHEMA + "\0").encode("utf-8"))
    header = json.dumps(
        {
            "approximation": approximation,
            "cutoff_frequency_thz": cutoff_frequency_thz,
            "metadata_json": metadata_json,
            "primitive_volume_angstrom3": primitive_volume_angstrom3,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    digest.update(len(header).to_bytes(8, "big"))
    digest.update(header)
    for name in _ARRAY_FIELDS:
        array = np.ascontiguousarray(arrays[name])
        name_bytes = name.encode("utf-8")
        descriptor = json.dumps(
            {"dtype": array.dtype.str, "shape": array.shape},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        payload = array.tobytes(order="C")
        for framed in (name_bytes, descriptor, payload):
            digest.update(len(framed).to_bytes(8, "big"))
            digest.update(framed)
    return digest.hexdigest()


def write_rta_result_hdf5(
    path: str | Path,
    result: RTAThermalConductivity,
    *,
    overwrite: bool = False,
) -> Path:
    """Write one result atomically without changing its numerical fields."""

    if not isinstance(result, RTAThermalConductivity):
        raise TypeError("result must be an RTAThermalConductivity.")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise TypeError("overwrite must be a boolean.")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    h5py = _h5py()
    metadata_json = json.dumps(
        _json_value(result.metadata),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    fields = {
        "temperatures": result.temperatures,
        "qpoints": result.qpoints,
        "frequencies_thz": result.frequencies_thz,
        "group_velocities_thz_angstrom": (
            result.group_velocities_thz_angstrom
        ),
        "heat_capacities_ev_per_k": result.heat_capacities_ev_per_k,
        "gamma_thz_hwhm": result.gamma_thz_hwhm,
        "population_lifetime_ps": result.population_lifetime_ps,
        "mean_free_path_angstrom": result.mean_free_path_angstrom,
        "mode_kappa_w_mk": result.mode_kappa_w_mk,
        "band_index_kappa_w_mk": result.band_index_kappa_w_mk,
        "kappa_w_mk": result.kappa_w_mk,
        "qpoint_weights": result.qpoint_weights,
        "included_mask": result.included_mask,
    }
    payload_sha256 = _payload_sha256(
        fields,
        approximation=result.approximation,
        primitive_volume_angstrom3=result.primitive_volume_angstrom3,
        cutoff_frequency_thz=result.cutoff_frequency_thz,
        metadata_json=metadata_json,
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = RTA_RESULT_HDF5_SCHEMA
            handle.attrs["approximation"] = result.approximation
            handle.attrs["primitive_volume_angstrom3"] = (
                result.primitive_volume_angstrom3
            )
            handle.attrs["cutoff_frequency_thz"] = result.cutoff_frequency_thz
            handle.attrs["metadata_json"] = metadata_json
            handle.attrs["payload_sha256"] = payload_sha256
            for name, values in fields.items():
                handle.create_dataset(
                    name,
                    data=values,
                    compression="gzip" if np.asarray(values).ndim > 1 else None,
                    shuffle=np.asarray(values).ndim > 1,
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


def read_rta_result_hdf5(path: str | Path) -> RTAThermalConductivity:
    """Read one complete v1 archive and re-run all result invariants."""

    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise ValueError("RTA archive must be a regular non-symlink file.")
    source = candidate.resolve(strict=True)
    if not source.is_file():
        raise ValueError("RTA archive must be a regular non-symlink file.")
    file_sha256_before = _file_sha256(source)
    h5py = _h5py()
    required = _ARRAY_FIELDS
    with h5py.File(source, "r") as handle:
        expected_attributes = {
            "schema",
            "approximation",
            "primitive_volume_angstrom3",
            "cutoff_frequency_thz",
            "metadata_json",
            "payload_sha256",
        }
        if {str(name) for name in handle.attrs} != expected_attributes:
            raise ValueError("RTA archive root attributes are not exact.")
        if {str(name) for name in handle} != set(required):
            raise ValueError("RTA archive root dataset schema is not exact.")
        schema = str(handle.attrs.get("schema", ""))
        if schema != RTA_RESULT_HDF5_SCHEMA:
            raise ValueError(
                f"Unsupported RTA archive schema {schema!r}; "
                f"expected {RTA_RESULT_HDF5_SCHEMA!r}."
            )
        missing = [name for name in required if name not in handle]
        if missing:
            raise ValueError(
                "RTA archive is incomplete; missing datasets: "
                + ", ".join(missing)
            )
        for name in required:
            link = handle.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError(  # noqa: TRY004 - malformed archive content
                    f"RTA archive dataset {name!r} must be an internal hard link."
                )
            dataset = handle.get(name, getclass=False)
            if not isinstance(dataset, h5py.Dataset):
                raise ValueError(  # noqa: TRY004 - malformed archive content
                    f"RTA archive field {name!r} is not a dataset."
                )
            if np.dtype(dataset.dtype) != _EXPECTED_DTYPES[name]:
                raise ValueError(
                    f"RTA archive dataset {name!r} has the wrong dtype."
                )
            if {str(key) for key in dataset.attrs}:
                raise ValueError(
                    f"RTA archive dataset {name!r} has unexpected attributes."
                )
        try:
            metadata_json = str(handle.attrs["metadata_json"])
            metadata = json.loads(metadata_json)
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("RTA archive metadata_json is missing or invalid.") from exc
        arrays = {name: handle[name][...] for name in required}
        approximation = str(handle.attrs.get("approximation", ""))
        try:
            volume = float(handle.attrs["primitive_volume_angstrom3"])
            cutoff = float(handle.attrs["cutoff_frequency_thz"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "RTA archive primitive volume or cutoff is missing or invalid."
            ) from exc
        stored_sha256 = str(handle.attrs.get("payload_sha256", ""))
    if _file_sha256(source) != file_sha256_before:
        raise RuntimeError("RTA archive changed while it was being read.")
    actual_sha256 = _payload_sha256(
        arrays,
        approximation=approximation,
        primitive_volume_angstrom3=volume,
        cutoff_frequency_thz=cutoff,
        metadata_json=metadata_json,
    )
    if not hmac.compare_digest(stored_sha256, actual_sha256):
        raise ValueError("RTA archive payload SHA256 does not match its contents.")
    return RTAThermalConductivity(
        temperatures=arrays["temperatures"],
        qpoints=arrays["qpoints"],
        frequencies_thz=arrays["frequencies_thz"],
        group_velocities_thz_angstrom=(
            arrays["group_velocities_thz_angstrom"]
        ),
        heat_capacities_ev_per_k=arrays["heat_capacities_ev_per_k"],
        gamma_thz_hwhm=arrays["gamma_thz_hwhm"],
        population_lifetime_ps=arrays["population_lifetime_ps"],
        mean_free_path_angstrom=arrays["mean_free_path_angstrom"],
        mode_kappa_w_mk=arrays["mode_kappa_w_mk"],
        band_index_kappa_w_mk=arrays["band_index_kappa_w_mk"],
        kappa_w_mk=arrays["kappa_w_mk"],
        qpoint_weights=arrays["qpoint_weights"],
        primitive_volume_angstrom3=volume,
        cutoff_frequency_thz=cutoff,
        included_mask=arrays["included_mask"],
        approximation=approximation,
        metadata=metadata,
    )


__all__ = [
    "RTA_RESULT_HDF5_SCHEMA",
    "read_rta_result_hdf5",
    "write_rta_result_hdf5",
]
