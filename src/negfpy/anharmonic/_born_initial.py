"""Immutable one-shot-Born initial states for material SCBA campaigns."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

import numpy as np

from ._provenance import (
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from .solver import solve_one_shot_born
from .types import CubicSCBAProblem, KeldyshSpectrum, SCBAConfig


SCHEMA = "negfpy_material_born_initial"
SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BornInitialError(RuntimeError):
    """Raised when a Born initial-state artifact is invalid or mismatched."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True,
        separators=(",", ":"), sort_keys=True,
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype="<c16")
    return _sha256_bytes(array.tobytes(order="C"))


def _validated_sources(values: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError("source_hashes must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, digest in values.items():
        if not isinstance(key, str) or not key:
            raise TypeError("source_hashes keys must be nonempty strings")
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"source_hashes[{key!r}] is not a SHA256")
        result[key] = digest
    return dict(sorted(result.items()))


def _identity(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    source_tree_sha256: str,
    environment_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    for name, digest in (
        ("source_tree_sha256", source_tree_sha256),
        ("environment_sha256", environment_sha256),
    ):
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{name} is not a SHA256")
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "method": "one_shot_born_contact_only",
        "problem_hashes": problem_input_hashes(problem),
        "configuration_sha256": configuration_hash(config),
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "source_hashes": _validated_sources(source_hashes),
    }


def write_born_initial_state(
    path: str | Path,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    source_tree_sha256: str,
    environment_sha256: str,
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate one Born bubble and atomically publish its Keldysh pair."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError("Born initial-state artifacts require h5py") from error
    target = Path(path).expanduser()
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Born initial-state artifact already exists: {target}")
    identity = _identity(
        problem, config, source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256, source_hashes=source_hashes,
    )
    result = solve_one_shot_born(problem, config)
    lesser = np.ascontiguousarray(result.sigma_pp.lesser, dtype="<c16")
    greater = np.ascontiguousarray(result.sigma_pp.greater, dtype="<c16")
    expected = (problem.grid.n_points, problem.ndof, problem.ndof)
    if lesser.shape != expected or greater.shape != expected:
        raise BornInitialError("one-shot Born returned an unexpected field shape")
    if not np.all(np.isfinite(lesser)) or not np.all(np.isfinite(greater)):
        raise BornInitialError("one-shot Born initial state is nonfinite")
    spectrum = KeldyshSpectrum(
        retarded=np.zeros(expected, dtype=np.complex128),
        lesser=lesser,
        greater=greater,
    )
    identity = {
        **identity,
        "initial_state_sha256": initial_self_energy_hash(problem, spectrum),
    }
    identity_text = _canonical(identity)
    lesser_sha = _array_sha256(lesser)
    greater_sha = _array_sha256(greater)
    manifest = {
        "identity_sha256": _sha256_bytes(identity_text.encode("ascii")),
        "lesser_sha256": lesser_sha,
        "greater_sha256": greater_sha,
    }
    manifest_sha = _sha256_bytes(_canonical(manifest).encode("ascii"))

    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with h5py.File(temporary, "w", libver="latest") as handle:
            handle.attrs["schema"] = SCHEMA
            handle.attrs["schema_version"] = SCHEMA_VERSION
            handle.attrs["write_state"] = "complete"
            handle.attrs["identity_sha256"] = manifest["identity_sha256"]
            handle.attrs["manifest_sha256"] = manifest_sha
            identity_dataset = handle.create_dataset(
                "identity_json", data=identity_text,
                dtype=h5py.string_dtype("utf-8"), track_times=False,
            )
            identity_dataset.attrs["sha256"] = manifest["identity_sha256"]
            lesser_dataset = handle.create_dataset(
                "sigma_lesser", data=lesser, track_times=False
            )
            greater_dataset = handle.create_dataset(
                "sigma_greater", data=greater, track_times=False
            )
            lesser_dataset.attrs["sha256"] = lesser_sha
            greater_dataset.attrs["sha256"] = greater_sha
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target.resolve()),
        "identity": identity,
        "identity_sha256": manifest["identity_sha256"],
        "manifest_sha256": manifest_sha,
        "initial_state_sha256": identity["initial_state_sha256"],
    }


def load_born_initial_state(
    path: str | Path,
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    source_tree_sha256: str,
    environment_sha256: str,
    source_hashes: Mapping[str, str],
) -> tuple[KeldyshSpectrum, dict[str, Any]]:
    """Load and independently authenticate a Born Keldysh pair."""

    try:
        import h5py
    except ImportError as error:  # pragma: no cover
        raise ImportError("Born initial-state artifacts require h5py") from error
    source = Path(path).expanduser().resolve(strict=True)
    expected_identity = _identity(
        problem, config, source_tree_sha256=source_tree_sha256,
        environment_sha256=environment_sha256, source_hashes=source_hashes,
    )
    try:
        with h5py.File(source, "r") as handle:
            if set(handle) != {"identity_json", "sigma_lesser", "sigma_greater"}:
                raise BornInitialError("Born artifact has unexpected objects")
            if set(handle.attrs) != {
                "schema", "schema_version", "write_state",
                "identity_sha256", "manifest_sha256",
            }:
                raise BornInitialError("Born artifact has unexpected attributes")
            if (
                str(handle.attrs["schema"]) != SCHEMA
                or int(handle.attrs["schema_version"]) != SCHEMA_VERSION
                or str(handle.attrs["write_state"]) != "complete"
            ):
                raise BornInitialError("Born artifact schema is invalid")
            raw_text = handle["identity_json"][()]
            identity_text = raw_text.decode("utf-8") if isinstance(raw_text, bytes) else str(raw_text)
            identity = json.loads(identity_text)
            if _canonical(identity) != identity_text:
                raise BornInitialError("Born identity JSON is not canonical")
            identity_sha = _sha256_bytes(identity_text.encode("ascii"))
            if (
                str(handle.attrs["identity_sha256"]) != identity_sha
                or str(handle["identity_json"].attrs.get("sha256", "")) != identity_sha
            ):
                raise BornInitialError("Born identity hash is invalid")
            lesser = np.asarray(handle["sigma_lesser"][...])
            greater = np.asarray(handle["sigma_greater"][...])
            lesser_sha = _array_sha256(lesser)
            greater_sha = _array_sha256(greater)
            if (
                str(handle["sigma_lesser"].attrs.get("sha256", "")) != lesser_sha
                or str(handle["sigma_greater"].attrs.get("sha256", "")) != greater_sha
            ):
                raise BornInitialError("Born field hash is invalid")
            manifest_sha = _sha256_bytes(_canonical({
                "identity_sha256": identity_sha,
                "lesser_sha256": lesser_sha,
                "greater_sha256": greater_sha,
            }).encode("ascii"))
            if str(handle.attrs["manifest_sha256"]) != manifest_sha:
                raise BornInitialError("Born manifest hash is invalid")
    except BornInitialError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise BornInitialError("Born initial-state artifact is corrupt") from error

    stored_base = dict(identity)
    stored_initial_hash = stored_base.pop("initial_state_sha256", None)
    if stored_base != expected_identity:
        raise BornInitialError("Born initial-state identity does not match the case")
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    if lesser.shape != shape or greater.shape != shape:
        raise BornInitialError("Born initial-state shape does not match the case")
    spectrum = KeldyshSpectrum(
        retarded=np.zeros(shape, dtype=np.complex128),
        lesser=lesser,
        greater=greater,
    )
    if stored_initial_hash != initial_self_energy_hash(problem, spectrum):
        raise BornInitialError("Born consumed-state hash is invalid")
    return spectrum, {
        "path": str(source),
        "file_sha256": _file_sha256(source),
        "identity_sha256": identity_sha,
        "manifest_sha256": manifest_sha,
        "initial_state_sha256": stored_initial_hash,
    }


__all__ = [
    "BornInitialError",
    "load_born_initial_state",
    "write_born_initial_state",
]
