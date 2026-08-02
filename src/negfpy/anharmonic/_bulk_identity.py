"""Dependency-light canonical identities for bulk scientific objects."""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .bulk_three_phonon import BulkThreePhononProblem


def _update_scientific_digest(
    digest: Any,
    value: Any,
    *,
    path: str,
) -> None:
    """Stream a deterministic, type-sensitive scientific object identity."""

    if value is None:
        digest.update(b"N;")
        return
    if isinstance(value, np.generic):
        _update_scientific_digest(digest, value.item(), path=path)
        return
    if isinstance(value, bool):
        digest.update(b"B1;" if value else b"B0;")
        return
    if isinstance(value, int):
        digest.update(b"I" + str(value).encode("ascii") + b";")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains NaN or infinity.")
        digest.update(b"F" + struct.pack(">d", float(value)) + b";")
        return
    if isinstance(value, complex):
        if not math.isfinite(value.real) or not math.isfinite(value.imag):
            raise ValueError(f"{path} contains NaN or infinity.")
        digest.update(
            b"C" + struct.pack(">dd", float(value.real), float(value.imag)) + b";"
        )
        return
    if isinstance(value, str):
        encoded = value.encode("utf-8")
        digest.update(b"S" + str(len(encoded)).encode("ascii") + b":" + encoded + b";")
        return
    if isinstance(value, Path):
        _update_scientific_digest(digest, value.as_posix(), path=path)
        return
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{path} cannot contain an object-dtype array.")
        header = json.dumps(
            {"dtype": array.dtype.str, "shape": list(array.shape)},
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(b"A" + str(len(header)).encode("ascii") + b":" + header + b":")
        digest.update(memoryview(array).cast("B"))
        digest.update(b";")
        return
    if is_dataclass(value) and not isinstance(value, type):
        name = f"{type(value).__module__}.{type(value).__qualname__}"
        digest.update(b"D")
        _update_scientific_digest(digest, name, path=f"{path}.__type__")
        for item in fields(value):
            _update_scientific_digest(digest, item.name, path=f"{path}.__field__")
            _update_scientific_digest(
                digest,
                getattr(value, item.name),
                path=f"{path}.{item.name}",
            )
        digest.update(b";")
        return
    if isinstance(value, Mapping):
        digest.update(b"M")
        keys = tuple(value.keys())
        if not all(isinstance(key, str) for key in keys):
            raise TypeError(f"{path} mapping keys must be strings.")
        for key in sorted(keys):
            _update_scientific_digest(digest, key, path=f"{path}.__key__")
            _update_scientific_digest(digest, value[key], path=f"{path}.{key}")
        digest.update(b";")
        return
    if isinstance(value, (tuple, list)):
        digest.update(b"T" if isinstance(value, tuple) else b"L")
        for index, item in enumerate(value):
            _update_scientific_digest(digest, item, path=f"{path}[{index}]")
        digest.update(b";")
        return
    if isinstance(value, (set, frozenset)):
        item_hashes: list[bytes] = []
        for index, item in enumerate(value):
            item_digest = hashlib.sha256()
            _update_scientific_digest(item_digest, item, path=f"{path}[{index}]")
            item_hashes.append(item_digest.digest())
        digest.update(b"E")
        for item_hash in sorted(item_hashes):
            digest.update(item_hash)
        digest.update(b";")
        return
    raise TypeError(
        f"{path} contains unsupported identity type {type(value).__name__}."
    )


def bulk_three_phonon_problem_sha256(problem: BulkThreePhononProblem) -> str:
    """Return the canonical, full scientific identity of a bulk problem."""

    if not isinstance(problem, BulkThreePhononProblem):
        raise TypeError("problem must be a BulkThreePhononProblem.")
    digest = hashlib.sha256()
    digest.update(b"negfpy.bulk_three_phonon_problem.v1\0")
    _update_scientific_digest(digest, problem, path="problem")
    return digest.hexdigest()


def _dataclass_payload_without(
    value: Any,
    *,
    excluded: frozenset[str],
) -> Mapping[str, Any]:
    """Return a typed dataclass payload with non-physical fields removed."""

    if not is_dataclass(value) or isinstance(value, type):
        raise TypeError("value must be a dataclass instance.")
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "fields": {
            item.name: getattr(value, item.name)
            for item in fields(value)
            if item.name not in excluded
        },
    }


def bulk_three_phonon_physics_sha256(problem: BulkThreePhononProblem) -> str:
    """Hash only inputs that can change the three-phonon calculation.

    The full problem identity intentionally includes provenance and user
    metadata for exact-run/restart protection.  That makes it unsuitable for
    deciding whether a linewidth result may be reused after byte-identical IFC
    files are relocated.  This companion identity includes numerical tensors,
    crystal mappings, conventions, mesh, and scientific controls, while
    excluding source paths, reader metadata, and arbitrary user annotations.
    """

    if not isinstance(problem, BulkThreePhononProblem):
        raise TypeError("problem must be a BulkThreePhononProblem.")
    payload = {
        "fc2": _dataclass_payload_without(
            problem.fc2,
            excluded=frozenset({"provenance", "version", "cutoff"}),
        ),
        "fc2_mapping": _dataclass_payload_without(
            problem.fc2_mapping,
            excluded=frozenset({"provenance"}),
        ),
        "fc3": _dataclass_payload_without(
            problem.fc3,
            excluded=frozenset({"provenance"}),
        ),
        "fc3_mapping": _dataclass_payload_without(
            problem.fc3_mapping,
            excluded=frozenset({"provenance"}),
        ),
        "mesh_shape": tuple(int(value) for value in problem.mesh.shape),
        "sigma_thz": problem.sigma_thz,
        "bz_reduced_reciprocal_lattice": (
            problem.bz_reduced_reciprocal_lattice
        ),
        "cutoff_frequency_thz": problem.cutoff_frequency_thz,
        "frequency_factor_to_thz": problem.frequency_factor_to_thz,
        "degenerate_tolerance_thz": problem.degenerate_tolerance_thz,
    }
    digest = hashlib.sha256()
    digest.update(b"negfpy.bulk_three_phonon_physics.v1\0")
    _update_scientific_digest(digest, payload, path="physics")
    return digest.hexdigest()


__all__ = [
    "bulk_three_phonon_physics_sha256",
    "bulk_three_phonon_problem_sha256",
]
