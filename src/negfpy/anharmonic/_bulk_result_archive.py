"""Immutable HDF5 archive for a complete serial bulk three-phonon result.

The 3x3x3 and 5x5x5 release gates compare task-MPI fields with the
authoritative serial kernel.  A comparison report alone is not evidence of
the serial operand, so this module stores every serial numerical and metadata
field in a strict, content-addressed HDF5 artifact.  The final release
aggregator reopens this artifact and repeats the comparison.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
from types import MappingProxyType
from typing import Any
from uuid import uuid4

import numpy as np

from ._bulk_target_q_mpi import (
    _ALL_DATASET_PATHS,
    _comparison_result_sha256,
    _dataset_paths,
    _dataset_sha256,
    _expected_dataset_dtypes,
    _read_metadata,
    _require_h5py,
    _result_arrays,
    _text_attribute,
)
from .bulk_harmonic import BulkPhononSpectrum
from .bulk_three_phonon import (
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
)


_SCHEMA = "negfpy.bulk_three_phonon_serial_result"
_SCHEMA_VERSION = 1
_ROOT_ATTRIBUTES = frozenset(
    {
        "schema_name",
        "schema_version",
        "result_sha256",
        "dataset_sha256_json",
        "provenance_json",
    }
)
_GROUPS = frozenset({"interactions", "linewidths", "metadata", "phonons"})


class BulkSerialResultArchiveError(RuntimeError):
    """Raised when a serial result archive is unsafe or inconsistent."""


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant {value!r}")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _strict_json_object(value: str, *, name: str) -> dict[str, Any]:
    try:
        result = json.loads(
            value,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BulkSerialResultArchiveError(
            f"{name} is not strict finite JSON."
        ) from exc
    if not isinstance(result, dict):
        raise BulkSerialResultArchiveError(f"{name} must be a JSON object.")
    if _canonical_json(result) != value:
        raise BulkSerialResultArchiveError(f"{name} is not canonical JSON.")
    return result


def _regular_path_without_symlinks(
    path: str | os.PathLike[str],
    *,
    must_exist: bool,
    name: str,
) -> Path:
    unresolved = Path(path).expanduser()
    if not unresolved.is_absolute():
        unresolved = Path.cwd() / unresolved
    cursor = Path(unresolved.anchor)
    for part in unresolved.parts[1:]:
        cursor = cursor / part
        if cursor.exists() or cursor.is_symlink():
            if cursor.is_symlink():
                raise BulkSerialResultArchiveError(
                    f"{name} may not traverse a symbolic link."
                )
    if must_exist:
        resolved = unresolved.resolve(strict=True)
        if not resolved.is_file() or resolved.is_symlink():
            raise BulkSerialResultArchiveError(
                f"{name} must be a regular non-symlink file."
            )
        if resolved.stat().st_nlink != 1:
            raise BulkSerialResultArchiveError(
                f"{name} must not have multiple hard links."
            )
    else:
        resolved = unresolved.resolve(strict=False)
    return resolved


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _audit_hdf5_storage(handle: Any) -> None:
    """Reject links or dataset storage whose bytes are outside this file."""

    h5py = _require_h5py()
    addresses: dict[int, str] = {}

    def visit(group: Any, prefix: str) -> None:
        for name in sorted(group.keys()):
            path = f"{prefix}/{name}" if prefix else name
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError(f"HDF5 non-hard link is forbidden: {path}")
            item = group[name]
            address = int(h5py.h5o.get_info(item.id).addr)
            owner = addresses.get(address)
            if owner is not None:
                raise ValueError(
                    f"HDF5 object {path!r} is a duplicate hard link to "
                    f"{owner!r}"
                )
            addresses[address] = path
            if isinstance(item, h5py.Dataset):
                if bool(item.is_virtual):
                    raise ValueError(
                        f"virtual HDF5 dataset is forbidden: {path}"
                    )
                if item.id.get_create_plist().get_external_count() != 0:
                    raise ValueError(
                        f"external HDF5 storage is forbidden: {path}"
                    )
            elif isinstance(item, h5py.Group):
                visit(item, path)
            else:
                raise ValueError(f"unsupported HDF5 object: {path}")

    visit(handle, "")


def _reconstruct_result(handle: Any) -> BulkThreePhononLinewidths:
    phonons = BulkPhononSpectrum(
        qpoints=np.asarray(handle["phonons/qpoints"]),
        dynamical_matrices=np.asarray(handle["phonons/dynamical_matrices"]),
        frequencies=np.asarray(handle["phonons/frequencies"]),
        eigenvectors=np.asarray(handle["phonons/eigenvectors"]),
        frequency_factor_to_thz=float(
            np.asarray(handle["phonons/frequency_factor_to_thz"])
        ),
        hermitianized=bool(np.asarray(handle["phonons/hermitianized"])),
    )
    interactions = BulkThreePhononInteractions(
        grid_points=np.asarray(handle["interactions/grid_points"]),
        band_indices=np.asarray(handle["interactions/band_indices"]),
        q1_indices=np.asarray(handle["interactions/q1_indices"]),
        q2_indices=np.asarray(handle["interactions/q2_indices"]),
        reciprocal_vectors=np.asarray(
            handle["interactions/reciprocal_vectors"]
        ),
        normal_mask=np.asarray(handle["interactions/normal_mask"]),
        matrix_elements=np.asarray(
            handle["interactions/matrix_elements"]
        ),
        interaction_strengths=np.asarray(
            handle["interactions/interaction_strengths"]
        ),
        phonons=phonons,
        metadata=_read_metadata(handle, "interactions"),
    )
    return BulkThreePhononLinewidths(
        temperatures=np.asarray(handle["linewidths/temperatures"]),
        grid_points=np.asarray(handle["linewidths/grid_points"]),
        band_indices=np.asarray(handle["linewidths/band_indices"]),
        gamma_thz_hwhm=np.asarray(handle["linewidths/gamma_thz_hwhm"]),
        gamma_normal_thz_hwhm=np.asarray(
            handle["linewidths/gamma_normal_thz_hwhm"]
        ),
        gamma_umklapp_thz_hwhm=np.asarray(
            handle["linewidths/gamma_umklapp_thz_hwhm"]
        ),
        gamma_angular_fwhm=np.asarray(
            handle["linewidths/gamma_angular_fwhm"]
        ),
        decay_phase_space=np.asarray(
            handle["linewidths/decay_phase_space"]
        ),
        difference_phase_space=np.asarray(
            handle["linewidths/difference_phase_space"]
        ),
        interactions=interactions,
        metadata=_read_metadata(handle, "linewidths"),
    )


@dataclass(frozen=True, slots=True)
class BulkSerialResultArchive:
    """Validated serial result plus immutable archive identity."""

    path: Path
    result: BulkThreePhononLinewidths
    result_sha256: str
    dataset_sha256: Mapping[str, str]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path).resolve(strict=True))
        object.__setattr__(
            self,
            "dataset_sha256",
            MappingProxyType(dict(self.dataset_sha256)),
        )
        object.__setattr__(
            self,
            "provenance",
            MappingProxyType(dict(self.provenance)),
        )


def read_bulk_serial_result_archive(
    path: str | os.PathLike[str],
) -> BulkSerialResultArchive:
    """Open, hash, and reconstruct every field in a serial result archive."""

    source = _regular_path_without_symlinks(
        path,
        must_exist=True,
        name="serial result archive",
    )
    h5py = _require_h5py()
    try:
        with h5py.File(source, "r") as handle:
            _audit_hdf5_storage(handle)
            if frozenset(str(name) for name in handle.attrs) != _ROOT_ATTRIBUTES:
                raise ValueError("serial result root attributes are not exact")
            if _text_attribute(handle, "schema_name") != _SCHEMA:
                raise ValueError("wrong serial result archive schema")
            if int(handle.attrs["schema_version"]) != _SCHEMA_VERSION:
                raise ValueError("unsupported serial result archive version")
            if frozenset(str(name) for name in handle) != _GROUPS:
                raise ValueError("serial result group schema is not exact")
            if _dataset_paths(handle) != tuple(sorted(_ALL_DATASET_PATHS)):
                raise ValueError("serial result dataset schema is not exact")
            expected_dtypes = _expected_dataset_dtypes()
            dataset_hashes: dict[str, str] = {}
            for name in _ALL_DATASET_PATHS:
                dataset = handle[name]
                if np.dtype(dataset.dtype) != expected_dtypes[name]:
                    raise ValueError(f"dataset {name!r} has the wrong dtype")
                if frozenset(str(key) for key in dataset.attrs) != {"sha256"}:
                    raise ValueError(
                        f"dataset {name!r} attributes are not exact"
                    )
                digest = _dataset_sha256(dataset)
                if _text_attribute(dataset, "sha256") != digest:
                    raise ValueError(
                        f"dataset {name!r} failed its SHA256 check"
                    )
                dataset_hashes[name] = digest
            if _strict_json_object(
                _text_attribute(handle, "dataset_sha256_json"),
                name="dataset_sha256_json",
            ) != dataset_hashes:
                raise ValueError("dataset SHA256 manifest is inconsistent")
            provenance = _strict_json_object(
                _text_attribute(handle, "provenance_json"),
                name="provenance_json",
            )
            result_sha256 = _text_attribute(handle, "result_sha256")
            if (
                len(result_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in result_sha256
                )
            ):
                raise ValueError("result_sha256 is invalid")
            result = _reconstruct_result(handle)
            if _comparison_result_sha256(result) != result_sha256:
                raise ValueError(
                    "reconstructed serial result SHA256 is inconsistent"
                )
    except BulkSerialResultArchiveError:
        raise
    except Exception as exc:
        raise BulkSerialResultArchiveError(
            f"invalid serial bulk result archive: {source}"
        ) from exc
    return BulkSerialResultArchive(
        path=source,
        result=result,
        result_sha256=result_sha256,
        dataset_sha256=dataset_hashes,
        provenance=provenance,
    )


def write_bulk_serial_result_archive(
    path: str | os.PathLike[str],
    result: BulkThreePhononLinewidths,
    *,
    provenance: Mapping[str, Any],
) -> BulkSerialResultArchive:
    """Publish a new immutable serial-result HDF5 artifact atomically."""

    if not isinstance(result, BulkThreePhononLinewidths):
        raise TypeError("result must be BulkThreePhononLinewidths.")
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping.")
    provenance_json = _canonical_json(dict(provenance))
    _strict_json_object(provenance_json, name="provenance")
    target = _regular_path_without_symlinks(
        path,
        must_exist=False,
        name="serial result archive",
    )
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "serial result archive already exists; artifacts are immutable."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target = _regular_path_without_symlinks(
        target,
        must_exist=False,
        name="serial result archive",
    )
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    h5py = _require_h5py()
    arrays = _result_arrays(result)
    dataset_hashes: dict[str, str] = {}
    try:
        with h5py.File(temporary, "x", track_order=False) as handle:
            handle.attrs["schema_name"] = _SCHEMA
            handle.attrs["schema_version"] = _SCHEMA_VERSION
            handle.attrs["result_sha256"] = _comparison_result_sha256(
                result
            )
            handle.attrs["provenance_json"] = provenance_json
            for group in sorted(_GROUPS):
                handle.create_group(group, track_order=False)
            for name in sorted(_ALL_DATASET_PATHS):
                dataset = handle.create_dataset(name, data=arrays[name])
                digest = _dataset_sha256(dataset)
                dataset.attrs["sha256"] = digest
                dataset_hashes[name] = digest
            handle.attrs["dataset_sha256_json"] = _canonical_json(
                dataset_hashes
            )
            handle.flush()
        descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(temporary, 0o444)
        os.link(temporary, target)
        temporary.unlink()
        _fsync_directory(target.parent)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise
    return read_bulk_serial_result_archive(target)


__all__ = [
    "BulkSerialResultArchive",
    "BulkSerialResultArchiveError",
    "read_bulk_serial_result_archive",
    "write_bulk_serial_result_archive",
]
