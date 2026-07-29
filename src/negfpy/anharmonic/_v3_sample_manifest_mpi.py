"""Task-level MPI construction of the bounded silicon V3 sample manifest.

Each MPI task owns one complete target-q HDF5 slab.  The first task phase
finds exact local promoted-block maxima.  Rank zero merges those records in
canonical target-q order and broadcasts the exact global maximum.  The second
phase builds fixed-size local reservoirs and integer coverage counts.  Rank
zero merges those artifacts without MPI floating-point reductions and emits
the same scientific JSON payload as the serial bounded-v2 builder.

The task manifests, immutable artifacts, checkpoints, and execution records
are orchestration evidence only.  They are deliberately absent from the final
scientific V3 sample manifest.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
import platform
import re
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, Literal

import numpy as np

from . import _mpi_tasks as mpi_tasks
from ._provenance import code_provenance
from ._mpi_tasks import (
    MPIExecutionReport,
    TaskCheckpointCorruptionError,
    TaskContext,
    TaskManifest,
    TaskManifestError,
    TaskResult,
    TaskSpec,
    collect_task_checkpoints,
    execute_task_manifest,
    load_task_manifest,
)
from .phono3py_reference import Phono3pySiliconReference
from .silicon_oracle_validation import (
    V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM,
    V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION,
    V3_SAMPLE_MANIFEST_SCHEMA,
    V3InteractionSampleManifest,
    array_sha256,
    crystal_mapping_sha256,
)
from .v3_sample_manifest import (
    LARGE_MESH_BOUNDED_SELECTION,
    _BoundedSelectionAccumulator,
    _DEGENERACY_TOLERANCE_THZ,
    _MAXIMUM_PROMOTED_BLOCKS,
    _PromotedBlock,
    _PromotedScanContext,
    _SmallestHashReservoir,
    _canonical_extreme,
    _coordinate_sha256,
    _file_sha256,
    _input_hashes,
    _iter_promoted_blocks,
    _positive_int,
    _prepare_promoted_scan,
    _select_bounded_blocks,
    _vectorized_maximum_promoted_block,
    read_v3_sample_manifest,
    write_v3_sample_manifest,
)


_ARTIFACT_SCHEMA = "negfpy.v3_sample_q0_task"
_CAMPAIGN_SCHEMA = "negfpy.v3_sample_mpi_campaign"
_SCHEMA_VERSION = 1
_TASK_ID_PATTERN = re.compile(r"^v3-q0-([0-9]{8})$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_PHASE_MAXIMUM = "maximum"
_PHASE_RESERVOIRS = "reservoirs"
_SOURCE_IDENTITY_SCHEMA = "negfpy.v3_sample_source_identity"
_ORCHESTRATION_SCHEMA = "negfpy.v3_sample_mpi_orchestration"
_ORCHESTRATION_SUFFIX = ".mpi-orchestration"
_CODE_FILES = (
    "scripts/generate_v3_sample_manifest_mpi.py",
    "src/negfpy/anharmonic/_v3_sample_manifest_mpi.py",
    "src/negfpy/anharmonic/v3_sample_manifest.py",
    "src/negfpy/anharmonic/_mpi_tasks.py",
    "src/negfpy/anharmonic/_provenance.py",
)
_RUNTIME_PACKAGES = (
    "negfpy",
    "numpy",
    "h5py",
    "phono3py",
    "phonopy",
    "spglib",
    "mpi4py",
)


class V3SampleMPICampaignError(RuntimeError):
    """Base class for task-level V3 sampling failures."""


class V3SampleMPIArtifactError(
    TaskCheckpointCorruptionError,
    V3SampleMPICampaignError,
):
    """Raised when an immutable q0 task artifact is invalid."""


class V3SampleMPIMergeError(V3SampleMPICampaignError):
    """Raised when canonical q0 artifacts cannot be merged."""


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase 64-character SHA256.")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_bytes(value: Any) -> bytes:
    return (_canonical_json(value) + "\n").encode("ascii")


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _resolve_nonsymlink_path(
    value: str | Path,
    *,
    name: str,
    must_exist: bool,
    expected_kind: Literal["file", "directory"],
    error_type: type[Exception] = V3SampleMPICampaignError,
) -> Path:
    """Reject symlinks component-by-component before resolving a path."""

    raw = Path(value).expanduser()
    if "\x00" in os.fspath(raw) or ".." in raw.parts:
        raise error_type(f"{name} must be a canonical non-symlink path.")
    lexical = raw if raw.is_absolute() else Path.cwd() / raw
    lexical = Path(os.path.abspath(os.fspath(lexical)))
    cursor = Path(lexical.anchor)
    missing = False
    for part in lexical.parts[1:]:
        cursor = cursor / part
        if missing:
            continue
        if cursor.is_symlink():
            raise error_type(f"{name} may not traverse a symlink.")
        try:
            cursor.lstat()
        except FileNotFoundError:
            missing = True
        except OSError as exc:
            raise error_type(f"cannot inspect {name}: {cursor}") from exc
    if must_exist and missing:
        raise error_type(f"{name} does not exist: {lexical}")
    if not missing:
        if expected_kind == "file" and not lexical.is_file():
            raise error_type(f"{name} must be a regular file.")
        if expected_kind == "directory" and not lexical.is_dir():
            raise error_type(f"{name} must be a directory.")
    try:
        return lexical.resolve(strict=must_exist)
    except OSError as exc:
        raise error_type(f"cannot resolve {name}: {lexical}") from exc


def _git(repository: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise V3SampleMPICampaignError(
            f"cannot inspect Git repository {repository}."
        ) from exc
    return result.stdout.strip()


def current_v3_sampling_source_identity(
    repository: str | Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    """Authenticate Git, imported NEGFPy, runtime, and executed source files."""

    root = _resolve_nonsymlink_path(
        repository,
        name="repository",
        must_exist=True,
        expected_kind="directory",
    )
    top = _resolve_nonsymlink_path(
        _git(root, "rev-parse", "--show-toplevel"),
        name="Git worktree root",
        must_exist=True,
        expected_kind="directory",
    )
    if top != root:
        raise V3SampleMPICampaignError(
            "repository must be the exact Git worktree root."
        )
    head = _git(root, "rev-parse", "HEAD")
    tree = _git(root, "rev-parse", "HEAD^{tree}")
    status = _git(
        root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        raise V3SampleMPICampaignError(
            "repository HEAD is not a full lowercase Git SHA."
        )
    provenance = dict(code_provenance())
    source_hash = provenance.get("source_tree_sha256")
    imported_checks = {
        "git_revision_matches": provenance.get("git_revision") == head,
        "source_tree_sha256_known": (
            isinstance(source_hash, str)
            and _SHA256_PATTERN.fullmatch(source_hash) is not None
        ),
        "package_version_consistent": (
            provenance.get("package_version_consistent") is True
        ),
        "imported_git_clean": provenance.get("git_dirty") is False,
    }
    clean = status == ""
    code_files: dict[str, str] = {}
    for relative in _CODE_FILES:
        path = _resolve_nonsymlink_path(
            root / relative,
            name=f"required V3 sampling code file {relative}",
            must_exist=True,
            expected_kind="file",
        )
        code_files[relative] = _file_sha256(path)
    packages: dict[str, str] = {}
    for name in _RUNTIME_PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not-installed"
    threads = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
    }
    authenticated = (
        clean
        and all(imported_checks.values())
        and all(value == "1" for value in threads.values())
    )
    if require_clean and not authenticated:
        raise V3SampleMPICampaignError(
            "V3 MPI sampling requires a clean Git worktree, matching clean "
            "imported NEGFPy provenance, a consistent installed package, and "
            "all thread controls set to 1."
        )
    return {
        "schema": _SOURCE_IDENTITY_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "repository": str(root),
        "head_commit": head,
        "git_tree": tree,
        "clean": clean,
        "status_sha256": hashlib.sha256(
            status.encode("utf-8")
        ).hexdigest(),
        "dirty_entry_count": 0 if clean else len(status.splitlines()),
        "source_tree_sha256": source_hash,
        "imported_code_provenance": provenance,
        "imported_checks": imported_checks,
        "code_files": code_files,
        "runtime": {
            "python": sys.version,
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "packages": packages,
            "thread_environment": threads,
        },
        "authenticated": authenticated,
    }


def _validated_source_identity(
    value: Any,
    *,
    require_authenticated: bool,
) -> dict[str, Any]:
    try:
        identity = json.loads(_canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise V3SampleMPICampaignError(
            "source identity is not finite canonical JSON."
        ) from exc
    expected_keys = {
        "schema",
        "schema_version",
        "repository",
        "head_commit",
        "git_tree",
        "clean",
        "status_sha256",
        "dirty_entry_count",
        "source_tree_sha256",
        "imported_code_provenance",
        "imported_checks",
        "code_files",
        "runtime",
        "authenticated",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != expected_keys
        or identity["schema"] != _SOURCE_IDENTITY_SCHEMA
        or identity["schema_version"] != _SCHEMA_VERSION
    ):
        raise V3SampleMPICampaignError(
            "source identity schema is invalid."
        )
    repository = identity["repository"]
    if (
        not isinstance(repository, str)
        or not repository
        or "\x00" in repository
        or not Path(repository).is_absolute()
        or not isinstance(identity["head_commit"], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity["head_commit"]) is None
        or not isinstance(identity["git_tree"], str)
        or re.fullmatch(r"[0-9a-f]{40}", identity["git_tree"]) is None
        or type(identity["clean"]) is not bool
        or not isinstance(identity["status_sha256"], str)
        or _SHA256_PATTERN.fullmatch(identity["status_sha256"]) is None
        or isinstance(identity["dirty_entry_count"], bool)
        or not isinstance(identity["dirty_entry_count"], int)
        or identity["dirty_entry_count"] < 0
        or not isinstance(identity["source_tree_sha256"], str)
        or _SHA256_PATTERN.fullmatch(identity["source_tree_sha256"])
        is None
        or type(identity["authenticated"]) is not bool
    ):
        raise V3SampleMPICampaignError(
            "source identity scalar fields are invalid."
        )
    expected_dirty = identity["dirty_entry_count"] == 0
    if identity["clean"] is not expected_dirty:
        raise V3SampleMPICampaignError(
            "source identity clean state and dirty-entry count disagree."
        )
    if identity["clean"] and identity["status_sha256"] != hashlib.sha256(
        b""
    ).hexdigest():
        raise V3SampleMPICampaignError(
            "clean source identity has a nonempty-status SHA256."
        )
    code_files = identity["code_files"]
    if (
        not isinstance(code_files, dict)
        or set(code_files) != set(_CODE_FILES)
        or any(
            not isinstance(digest, str)
            or _SHA256_PATTERN.fullmatch(digest) is None
            for digest in code_files.values()
        )
    ):
        raise V3SampleMPICampaignError(
            "source identity code-file hashes are incomplete."
        )
    provenance = identity["imported_code_provenance"]
    provenance_keys = {
        "package_version",
        "declared_package_version",
        "installed_package_version",
        "package_version_consistent",
        "source_tree_sha256",
        "git_revision",
        "git_dirty",
    }
    if (
        not isinstance(provenance, dict)
        or set(provenance) != provenance_keys
        or any(
            not isinstance(provenance[name], str)
            for name in (
                "package_version",
                "declared_package_version",
                "installed_package_version",
                "source_tree_sha256",
                "git_revision",
            )
        )
        or type(provenance["package_version_consistent"]) is not bool
        or (
            type(provenance["git_dirty"]) is not bool
            and provenance["git_dirty"] != "unknown"
        )
        or provenance["source_tree_sha256"]
        != identity["source_tree_sha256"]
    ):
        raise V3SampleMPICampaignError(
            "imported-code provenance schema is invalid."
        )
    checks = identity["imported_checks"]
    expected_checks = {
        "git_revision_matches": (
            provenance["git_revision"] == identity["head_commit"]
        ),
        "source_tree_sha256_known": (
            _SHA256_PATTERN.fullmatch(
                provenance["source_tree_sha256"]
            )
            is not None
        ),
        "package_version_consistent": (
            provenance["package_version_consistent"] is True
        ),
        "imported_git_clean": provenance["git_dirty"] is False,
    }
    if (
        not isinstance(checks, dict)
        or set(checks) != set(expected_checks)
        or any(type(item) is not bool for item in checks.values())
        or checks != expected_checks
    ):
        raise V3SampleMPICampaignError(
            "source identity imported checks are inconsistent."
        )
    runtime = identity["runtime"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "python_executable",
        "platform",
        "machine",
        "packages",
        "thread_environment",
    }:
        raise V3SampleMPICampaignError(
            "source identity runtime schema is invalid."
        )
    if any(
        not isinstance(runtime[name], str) or not runtime[name]
        for name in (
            "python",
            "python_executable",
            "platform",
            "machine",
        )
    ):
        raise V3SampleMPICampaignError(
            "source identity runtime values are invalid."
        )
    packages = runtime["packages"]
    threads = runtime["thread_environment"]
    if (
        not isinstance(packages, dict)
        or set(packages) != set(_RUNTIME_PACKAGES)
        or any(
            not isinstance(item, str) or not item
            for item in packages.values()
        )
        or not isinstance(threads, dict)
        or set(threads) != set(_THREAD_ENVIRONMENT)
        or any(
            item is not None and not isinstance(item, str)
            for item in threads.values()
        )
    ):
        raise V3SampleMPICampaignError(
            "source identity package or thread runtime is invalid."
        )
    authenticated = (
        identity["clean"]
        and all(checks.values())
        and all(value == "1" for value in threads.values())
    )
    if identity["authenticated"] is not authenticated:
        raise V3SampleMPICampaignError(
            "source identity authentication decision is inconsistent."
        )
    if require_authenticated and not authenticated:
        raise V3SampleMPICampaignError(
            "release V3 sampling requires authenticated clean source."
        )
    return identity


def _strict_json(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        value = json.loads(
            raw.decode("ascii"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise V3SampleMPIArtifactError(
            f"invalid or truncated q0 artifact: {path}"
        ) from exc
    if not isinstance(value, dict) or raw != _canonical_bytes(value):
        raise V3SampleMPIArtifactError(
            f"q0 artifact is not a canonical JSON object: {path}"
        )
    return value


def _read_unique_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON token {token}")
            ),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        raise V3SampleMPIArtifactError(
            f"{name} is not a strict finite JSON object."
        ) from exc
    if not isinstance(value, dict):
        raise V3SampleMPIArtifactError(
            f"{name} must be a JSON object."
        )
    return value


def _block_payload(block: _PromotedBlock) -> dict[str, Any]:
    return {
        "q0": block.q0,
        "q1": block.q1,
        "q2": block.q2,
        "target_subspace": list(block.target_subspace),
        "band1_subspace": list(block.band1_subspace),
        "band2_subspace": list(block.band2_subspace),
        "seed_b0": block.seed_b0,
        "seed_b1": block.seed_b1,
        "seed_b2": block.seed_b2,
        "process": block.process,
        "strength_ev2": block.strength_ev2,
        "stratum": block.stratum,
        "coordinate_sha256": block.coordinate_sha256,
    }


def _index_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise V3SampleMPIArtifactError(f"{name} must be a nonempty list.")
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in value
    ):
        raise V3SampleMPIArtifactError(
            f"{name} must contain nonnegative integers."
        )
    result = tuple(int(item) for item in value)
    if tuple(sorted(set(result))) != result:
        raise V3SampleMPIArtifactError(
            f"{name} must be strictly increasing and unique."
        )
    return result


def _block_from_payload(
    value: Any,
    *,
    expected_q0: int | None = None,
) -> _PromotedBlock:
    keys = {
        "q0",
        "q1",
        "q2",
        "target_subspace",
        "band1_subspace",
        "band2_subspace",
        "seed_b0",
        "seed_b1",
        "seed_b2",
        "process",
        "strength_ev2",
        "stratum",
        "coordinate_sha256",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise V3SampleMPIArtifactError(
            "promoted-block artifact has an invalid schema."
        )
    integer_names = ("q0", "q1", "q2", "seed_b0", "seed_b1", "seed_b2")
    if any(
        isinstance(value[name], bool)
        or not isinstance(value[name], int)
        or value[name] < 0
        for name in integer_names
    ):
        raise V3SampleMPIArtifactError(
            "promoted-block indices must be nonnegative integers."
        )
    target = _index_tuple(value["target_subspace"], name="target_subspace")
    band1 = _index_tuple(value["band1_subspace"], name="band1_subspace")
    band2 = _index_tuple(value["band2_subspace"], name="band2_subspace")
    strength = value["strength_ev2"]
    if (
        isinstance(strength, bool)
        or not isinstance(strength, (int, float))
        or not math.isfinite(float(strength))
        or float(strength) < 0.0
    ):
        raise V3SampleMPIArtifactError(
            "promoted-block strength must be finite and nonnegative."
        )
    q0, q1, q2 = (int(value[name]) for name in ("q0", "q1", "q2"))
    if expected_q0 is not None and q0 != expected_q0:
        raise V3SampleMPIArtifactError(
            "promoted block belongs to another target q point."
        )
    coordinate = _coordinate_sha256(q0, q1, q2, target, band1, band2)
    if value["coordinate_sha256"] != coordinate:
        raise V3SampleMPIArtifactError(
            "promoted-block coordinate SHA256 is inconsistent."
        )
    process = value["process"]
    stratum = value["stratum"]
    if process not in {"normal", "umklapp"} or not isinstance(stratum, str):
        raise V3SampleMPIArtifactError(
            "promoted-block process or stratum is invalid."
        )
    if not stratum.endswith(f".{process}"):
        raise V3SampleMPIArtifactError(
            "promoted-block stratum and process disagree."
        )
    seeds = tuple(int(value[name]) for name in ("seed_b0", "seed_b1", "seed_b2"))
    if (
        seeds[0] not in target
        or seeds[1] not in band1
        or seeds[2] not in band2
    ):
        raise V3SampleMPIArtifactError(
            "promoted-block seed lies outside its subspace."
        )
    return _PromotedBlock(
        q0=q0,
        q1=q1,
        q2=q2,
        target_subspace=target,
        band1_subspace=band1,
        band2_subspace=band2,
        seed_b0=seeds[0],
        seed_b1=seeds[1],
        seed_b2=seeds[2],
        process=process,
        strength_ev2=float(strength),
        stratum=stratum,
        coordinate_sha256=coordinate,
    )


def _reservoir_payload(
    reservoir: _SmallestHashReservoir,
) -> list[dict[str, Any]]:
    return [_block_payload(value) for value in reservoir.values()]


def _accumulator_payload(
    accumulator: _BoundedSelectionAccumulator,
) -> dict[str, Any]:
    return {
        "floor": accumulator.floor,
        "global_maximum": accumulator.global_maximum,
        "maximum_promoted_blocks": accumulator.maximum_promoted_blocks,
        "target_bands": list(accumulator.target_bands),
        "viable_count": accumulator.viable_count,
        "stratum_counts": dict(sorted(accumulator.stratum_counts.items())),
        "global_reservoir": _reservoir_payload(
            accumulator.global_reservoir
        ),
        "stratum_reservoirs": {
            key: _reservoir_payload(value)
            for key, value in sorted(
                accumulator.stratum_reservoirs.items()
            )
        },
        "strength_reservoirs": {
            f"{key[0]}:{key[1]}": _reservoir_payload(value)
            for key, value in sorted(
                accumulator.strength_reservoirs.items()
            )
        },
        "target_band_reservoirs": {
            str(key): _reservoir_payload(value)
            for key, value in sorted(
                accumulator.target_band_reservoirs.items()
            )
        },
        "process_minima": {
            key: _block_payload(value)
            for key, value in sorted(accumulator.process_minima.items())
        },
        "process_maxima": {
            key: _block_payload(value)
            for key, value in sorted(accumulator.process_maxima.items())
        },
        "global_maximum_block": (
            None
            if accumulator.global_maximum_block is None
            else _block_payload(accumulator.global_maximum_block)
        ),
    }


def _reservoir_from_payload(
    values: Any,
    *,
    capacity: int,
    expected_q0: int,
) -> _SmallestHashReservoir:
    if not isinstance(values, list) or len(values) > capacity:
        raise V3SampleMPIArtifactError(
            "reservoir artifact exceeds its fixed capacity."
        )
    reservoir = _SmallestHashReservoir(capacity)
    for value in values:
        reservoir.add(
            _block_from_payload(value, expected_q0=expected_q0)
        )
    if len(reservoir) != len(values):
        raise V3SampleMPIArtifactError(
            "reservoir artifact contains duplicate promoted blocks."
        )
    if _reservoir_payload(reservoir) != values:
        raise V3SampleMPIArtifactError(
            "reservoir artifact is not in canonical hash order."
        )
    return reservoir


def _accumulator_from_payload(
    value: Any,
    *,
    expected_q0: int,
    expected_floor: float,
    expected_global_maximum: float,
    expected_maximum: int,
    expected_target_bands: tuple[int, ...],
) -> _BoundedSelectionAccumulator:
    keys = {
        "floor",
        "global_maximum",
        "maximum_promoted_blocks",
        "target_bands",
        "viable_count",
        "stratum_counts",
        "global_reservoir",
        "stratum_reservoirs",
        "strength_reservoirs",
        "target_band_reservoirs",
        "process_minima",
        "process_maxima",
        "global_maximum_block",
    }
    if not isinstance(value, dict) or set(value) != keys:
        raise V3SampleMPIArtifactError(
            "bounded-accumulator artifact has an invalid schema."
        )
    if (
        value["floor"] != expected_floor
        or value["global_maximum"] != expected_global_maximum
        or value["maximum_promoted_blocks"] != expected_maximum
        or value["target_bands"] != list(expected_target_bands)
    ):
        raise V3SampleMPIArtifactError(
            "bounded-accumulator numerical identity is mismatched."
        )
    viable_count = value["viable_count"]
    if (
        isinstance(viable_count, bool)
        or not isinstance(viable_count, int)
        or viable_count < 0
    ):
        raise V3SampleMPIArtifactError("viable_count is invalid.")
    accumulator = _BoundedSelectionAccumulator(
        floor=expected_floor,
        global_maximum=expected_global_maximum,
        maximum_promoted_blocks=expected_maximum,
        target_bands=expected_target_bands,
    )
    accumulator.viable_count = int(viable_count)
    raw_counts = value["stratum_counts"]
    if not isinstance(raw_counts, dict):
        raise V3SampleMPIArtifactError("stratum_counts must be an object.")
    for name, count in raw_counts.items():
        if (
            not isinstance(name, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
        ):
            raise V3SampleMPIArtifactError("stratum_counts is invalid.")
    accumulator.stratum_counts = {
        str(name): int(count) for name, count in raw_counts.items()
    }
    accumulator.global_reservoir = _reservoir_from_payload(
        value["global_reservoir"],
        capacity=expected_maximum,
        expected_q0=expected_q0,
    )
    raw_strata = value["stratum_reservoirs"]
    if not isinstance(raw_strata, dict):
        raise V3SampleMPIArtifactError(
            "stratum_reservoirs must be an object."
        )
    accumulator.stratum_reservoirs = {
        str(name): _reservoir_from_payload(
            records,
            capacity=2,
            expected_q0=expected_q0,
        )
        for name, records in raw_strata.items()
    }
    raw_strengths = value["strength_reservoirs"]
    if not isinstance(raw_strengths, dict):
        raise V3SampleMPIArtifactError(
            "strength_reservoirs must be an object."
        )
    accumulator.strength_reservoirs = {}
    for text, records in raw_strengths.items():
        if not isinstance(text, str) or ":" not in text:
            raise V3SampleMPIArtifactError(
                "strength reservoir key is invalid."
            )
        process, raw_bin = text.split(":", 1)
        if process not in {"normal", "umklapp"}:
            raise V3SampleMPIArtifactError(
                "strength reservoir process is invalid."
            )
        try:
            bin_index = int(raw_bin)
        except ValueError as exc:
            raise V3SampleMPIArtifactError(
                "strength reservoir bin is invalid."
            ) from exc
        if str(bin_index) != raw_bin or not 0 <= bin_index < 8:
            raise V3SampleMPIArtifactError(
                "strength reservoir bin is invalid."
            )
        accumulator.strength_reservoirs[(process, bin_index)] = (
            _reservoir_from_payload(
                records,
                capacity=2,
                expected_q0=expected_q0,
            )
        )
    raw_bands = value["target_band_reservoirs"]
    if not isinstance(raw_bands, dict) or set(raw_bands) != {
        str(value) for value in expected_target_bands
    }:
        raise V3SampleMPIArtifactError(
            "target-band reservoir schema is invalid."
        )
    accumulator.target_band_reservoirs = {
        band: _reservoir_from_payload(
            raw_bands[str(band)],
            capacity=16,
            expected_q0=expected_q0,
        )
        for band in expected_target_bands
    }

    def extremes(raw: Any, *, name: str) -> dict[str, _PromotedBlock]:
        if not isinstance(raw, dict) or not set(raw).issubset(
            {"normal", "umklapp"}
        ):
            raise V3SampleMPIArtifactError(f"{name} schema is invalid.")
        return {
            process: _block_from_payload(
                block,
                expected_q0=expected_q0,
            )
            for process, block in raw.items()
        }

    accumulator.process_minima = extremes(
        value["process_minima"], name="process_minima"
    )
    accumulator.process_maxima = extremes(
        value["process_maxima"], name="process_maxima"
    )
    raw_global = value["global_maximum_block"]
    accumulator.global_maximum_block = (
        None
        if raw_global is None
        else _block_from_payload(raw_global, expected_q0=expected_q0)
    )
    if accumulator.retained_candidate_slots > accumulator.retention_bound:
        raise V3SampleMPIArtifactError(
            "bounded-accumulator artifact exceeds its retention bound."
        )
    return accumulator


def _task_q0(task: TaskSpec) -> int:
    match = _TASK_ID_PATTERN.fullmatch(task.task_id)
    if match is None:
        raise TaskManifestError("V3 sampling task_id is not canonical.")
    q0 = int(match.group(1))
    if task.payload.get("q0") != q0:
        raise TaskManifestError("V3 sampling task q0 payload is mismatched.")
    return q0


def _artifact_document(
    task: TaskSpec,
    *,
    phase: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    body = {
        "schema": _ARTIFACT_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "phase": phase,
        "task_id": task.task_id,
        "identity_hashes": dict(task.identity_hashes),
        "q0": _task_q0(task),
        "payload": dict(payload),
    }
    return {"payload": body, "payload_sha256": _payload_sha256(body)}


def _read_task_artifact(
    path: Path,
    task: TaskSpec,
    *,
    expected_phase: str,
    floor: float | None = None,
    global_maximum: float | None = None,
    maximum_promoted_blocks: int | None = None,
    target_bands: tuple[int, ...] | None = None,
) -> _PromotedBlock | _BoundedSelectionAccumulator:
    document = _strict_json(path)
    if set(document) != {"payload", "payload_sha256"}:
        raise V3SampleMPIArtifactError(
            "q0 task artifact envelope is invalid."
        )
    body = document["payload"]
    if (
        not isinstance(body, dict)
        or document["payload_sha256"] != _payload_sha256(body)
    ):
        raise V3SampleMPIArtifactError(
            "q0 task artifact payload SHA256 is invalid."
        )
    expected_keys = {
        "schema",
        "schema_version",
        "phase",
        "task_id",
        "identity_hashes",
        "q0",
        "payload",
    }
    if (
        set(body) != expected_keys
        or body["schema"] != _ARTIFACT_SCHEMA
        or body["schema_version"] != _SCHEMA_VERSION
        or body["phase"] != expected_phase
        or body["task_id"] != task.task_id
        or body["identity_hashes"] != dict(task.identity_hashes)
        or body["q0"] != _task_q0(task)
    ):
        raise V3SampleMPIArtifactError(
            "q0 task artifact identity is mismatched."
        )
    q0 = _task_q0(task)
    if expected_phase == _PHASE_MAXIMUM:
        if not isinstance(body["payload"], dict) or set(
            body["payload"]
        ) != {"maximum_block"}:
            raise V3SampleMPIArtifactError(
                "maximum task payload schema is invalid."
            )
        return _block_from_payload(
            body["payload"]["maximum_block"],
            expected_q0=q0,
        )
    if (
        floor is None
        or global_maximum is None
        or maximum_promoted_blocks is None
        or target_bands is None
    ):
        raise ValueError("reservoir artifact validation needs exact controls.")
    if not isinstance(body["payload"], dict) or set(
        body["payload"]
    ) != {"accumulator"}:
        raise V3SampleMPIArtifactError(
            "reservoir task payload schema is invalid."
        )
    return _accumulator_from_payload(
        body["payload"]["accumulator"],
        expected_q0=q0,
        expected_floor=floor,
        expected_global_maximum=global_maximum,
        expected_maximum=maximum_promoted_blocks,
        expected_target_bands=target_bands,
    )


def _reference_descriptor(
    reference: Phono3pySiliconReference,
) -> dict[str, Any]:
    """Return a small picklable descriptor; interaction slabs stay in HDF5."""

    if not isinstance(reference, Phono3pySiliconReference):
        raise TypeError("reference must be a Phono3pySiliconReference.")
    return {
        "path": str(reference.path),
        "sha256": reference.sha256,
        "mesh": tuple(reference.mesh),
        "d_diag": tuple(reference.d_diag),
        "grid_points_grg": np.asarray(reference.grid_points_grg),
        "grid_points_bzg": np.asarray(reference.grid_points_bzg),
        "band_indices": np.asarray(reference.band_indices),
        "temperatures_k": np.asarray(reference.temperatures_k),
        "sigmas_thz": np.asarray(reference.sigmas_thz),
        "grg2bzg": np.asarray(reference.grg2bzg),
        "bzg2grg": np.asarray(reference.bzg2grg),
        "bz_addresses": np.asarray(reference.bz_addresses),
        "frequencies_thz": np.asarray(reference.frequencies_thz),
        "gamma_thz_hwhm": np.asarray(reference.gamma_thz_hwhm),
        "target_groups": dict(reference.target_groups),
        "make_r0_average": reference.make_r0_average,
        "cutoff_frequency_thz": reference.cutoff_frequency_thz,
    }


def _reference_from_descriptor(
    value: Mapping[str, Any],
) -> Phono3pySiliconReference:
    expected = {
        "path",
        "sha256",
        "mesh",
        "d_diag",
        "grid_points_grg",
        "grid_points_bzg",
        "band_indices",
        "temperatures_k",
        "sigmas_thz",
        "grg2bzg",
        "bzg2grg",
        "bz_addresses",
        "frequencies_thz",
        "gamma_thz_hwhm",
        "target_groups",
        "make_r0_average",
        "cutoff_frequency_thz",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError("V3 sampling reference descriptor is invalid.")
    return Phono3pySiliconReference(
        path=Path(str(value["path"])),
        sha256=str(value["sha256"]),
        mesh=tuple(int(item) for item in value["mesh"]),
        d_diag=tuple(int(item) for item in value["d_diag"]),
        grid_points_grg=value["grid_points_grg"],
        grid_points_bzg=value["grid_points_bzg"],
        band_indices=value["band_indices"],
        temperatures_k=value["temperatures_k"],
        sigmas_thz=value["sigmas_thz"],
        grg2bzg=value["grg2bzg"],
        bzg2grg=value["bzg2grg"],
        bz_addresses=value["bz_addresses"],
        frequencies_thz=value["frequencies_thz"],
        gamma_thz_hwhm=value["gamma_thz_hwhm"],
        target_groups=value["target_groups"],
        make_r0_average=bool(value["make_r0_average"]),
        cutoff_frequency_thz=float(value["cutoff_frequency_thz"]),
        fc2_random_qpoints=None,
        fc2_dynamical_matrices=None,
        fc2_hermitianized=None,
        schema_issues=(),
    )


@dataclass(frozen=True, slots=True)
class V3SampleMPICampaign:
    """Exact scientific and provenance inputs for one bounded-v2 campaign."""

    reference: Phono3pySiliconReference
    fc2_compact_hdf5_path: Path
    fc2_full_hdf5_path: Path
    fc3_compact_hdf5_path: Path
    fc3_full_hdf5_path: Path
    phono3py_yaml_path: Path
    input_manifest_path: Path
    fc2_mapping: Any
    fc3_mapping: Any
    source_revision: str
    source_tree_sha256: str
    environment_sha256: str
    source_identity: Mapping[str, Any]
    require_authenticated_source: bool = True
    minimum_promoted_blocks: int = 64
    maximum_promoted_blocks: int = 512
    degeneracy_tolerance_thz: float = _DEGENERACY_TOLERANCE_THZ
    reference_stat_identity: tuple[int, int, int, int] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.reference, Phono3pySiliconReference):
            raise TypeError(
                "reference must be a Phono3pySiliconReference."
            )
        _resolve_nonsymlink_path(
            self.reference.path,
            name="reference",
            must_exist=True,
            expected_kind="file",
            error_type=ValueError,
        )
        for name in (
            "fc2_compact_hdf5_path",
            "fc2_full_hdf5_path",
            "fc3_compact_hdf5_path",
            "fc3_full_hdf5_path",
            "phono3py_yaml_path",
            "input_manifest_path",
        ):
            path = _resolve_nonsymlink_path(
                getattr(self, name),
                name=name,
                must_exist=True,
                expected_kind="file",
                error_type=ValueError,
            )
            object.__setattr__(self, name, path)
        if not isinstance(self.source_revision, str) or not self.source_revision:
            raise ValueError("source_revision must be nonempty text.")
        _require_sha256(
            self.source_tree_sha256, name="source_tree_sha256"
        )
        _require_sha256(
            self.environment_sha256, name="environment_sha256"
        )
        if not isinstance(
            self.require_authenticated_source, (bool, np.bool_)
        ):
            raise TypeError("require_authenticated_source must be boolean.")
        if not isinstance(self.source_identity, Mapping):
            raise TypeError("source_identity must be a mapping.")
        identity = _validated_source_identity(
            self.source_identity,
            require_authenticated=bool(
                self.require_authenticated_source
            ),
        )
        if (
            identity["head_commit"] != self.source_revision
            or identity["source_tree_sha256"]
            != self.source_tree_sha256
        ):
            raise ValueError(
                "source_identity does not identify the declared source."
            )
        object.__setattr__(self, "source_identity", identity)
        object.__setattr__(
            self,
            "require_authenticated_source",
            bool(self.require_authenticated_source),
        )
        minimum = _positive_int(
            self.minimum_promoted_blocks,
            name="minimum_promoted_blocks",
        )
        maximum = _positive_int(
            self.maximum_promoted_blocks,
            name="maximum_promoted_blocks",
        )
        if minimum < 64 or maximum > _MAXIMUM_PROMOTED_BLOCKS:
            raise ValueError(
                "bounded V3 sampling requires 64 <= minimum and maximum <= 512."
            )
        if maximum < minimum:
            raise ValueError(
                "maximum_promoted_blocks must be at least the minimum."
            )
        if self.degeneracy_tolerance_thz != _DEGENERACY_TOLERANCE_THZ:
            raise ValueError(
                "degeneracy_tolerance_thz is pinned to 1e-4 THz."
            )
        object.__setattr__(self, "minimum_promoted_blocks", minimum)
        object.__setattr__(self, "maximum_promoted_blocks", maximum)
        stat = self.reference.path.stat()
        object.__setattr__(
            self,
            "reference_stat_identity",
            (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns),
        )

    @property
    def target_grid_points(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.reference.grid_points_grg)

    @property
    def target_bands(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.reference.band_indices)

    @property
    def input_hashes(self) -> dict[str, str]:
        return _input_hashes(
            reference=self.reference,
            fc2_compact_hdf5_path=self.fc2_compact_hdf5_path,
            fc2_full_hdf5_path=self.fc2_full_hdf5_path,
            fc3_compact_hdf5_path=self.fc3_compact_hdf5_path,
            fc3_full_hdf5_path=self.fc3_full_hdf5_path,
            phono3py_yaml_path=self.phono3py_yaml_path,
            input_manifest_path=self.input_manifest_path,
        )

    @property
    def mapping_hashes(self) -> dict[str, str]:
        return {
            "fc2": crystal_mapping_sha256(self.fc2_mapping),
            "fc3": crystal_mapping_sha256(self.fc3_mapping),
        }

    def identity_payload(self) -> dict[str, Any]:
        """Return the complete canonical campaign identity."""

        return {
            "schema": _CAMPAIGN_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "selection_mode": LARGE_MESH_BOUNDED_SELECTION,
            "selection_algorithm": V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM,
            "selection_algorithm_version": (
                V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
            ),
            "source_revision": self.source_revision,
            "source_tree_sha256": self.source_tree_sha256,
            "environment_sha256": self.environment_sha256,
            "source_identity_sha256": _payload_sha256(
                self.source_identity
            ),
            "source_authenticated": self.source_identity["authenticated"],
            "inputs": self.input_hashes,
            "mapping_sha256": self.mapping_hashes,
            "mesh": list(self.reference.mesh),
            "grg2bzg_sha256": array_sha256(self.reference.grg2bzg),
            "bzg2grg_sha256": array_sha256(self.reference.bzg2grg),
            "target_grid_points": list(self.target_grid_points),
            "target_bands": list(self.target_bands),
            "minimum_promoted_blocks": self.minimum_promoted_blocks,
            "maximum_promoted_blocks": self.maximum_promoted_blocks,
            "degeneracy_tolerance_thz": (
                self.degeneracy_tolerance_thz
            ),
        }

    @property
    def campaign_sha256(self) -> str:
        return _payload_sha256(self.identity_payload())

    def task_manifest(
        self,
        phase: Literal["maximum", "reservoirs"],
        *,
        global_maximum_block: _PromotedBlock | None = None,
    ) -> TaskManifest:
        if phase not in {_PHASE_MAXIMUM, _PHASE_RESERVOIRS}:
            raise ValueError("unsupported V3 sampling task phase.")
        phase_hashes = {"campaign_sha256": self.campaign_sha256}
        phase_hashes["source_identity_sha256"] = _payload_sha256(
            self.source_identity
        )
        metadata: dict[str, Any] = {
            "campaign_schema": _CAMPAIGN_SCHEMA,
            "phase": phase,
            "selection_algorithm_version": 2,
            "source_authenticated": self.source_identity["authenticated"],
        }
        if phase == _PHASE_RESERVOIRS:
            if global_maximum_block is None:
                raise ValueError(
                    "reservoir phase requires the exact global maximum."
                )
            maximum_payload = _block_payload(global_maximum_block)
            phase_hashes["global_maximum_sha256"] = _payload_sha256(
                maximum_payload
            )
            metadata["global_maximum_block"] = maximum_payload
        tasks = tuple(
            TaskSpec(
                task_id=f"v3-q0-{q0:08d}",
                identity_hashes=phase_hashes,
                payload={"phase": phase, "q0": q0},
            )
            for q0 in self.target_grid_points
        )
        return TaskManifest(
            source_revision=self.source_revision,
            source_tree_sha256=self.source_tree_sha256,
            environment_sha256=self.environment_sha256,
            tasks=tasks,
            identity_hashes=phase_hashes,
            metadata=metadata,
        )


def _make_worker(
    campaign: V3SampleMPICampaign,
    *,
    phase: Literal["maximum", "reservoirs"],
    global_maximum_block: _PromotedBlock | None = None,
):
    scan_context: _PromotedScanContext | None = None

    def worker(task: TaskSpec, context: TaskContext) -> TaskResult:
        nonlocal scan_context
        if context.thread_limit != 1:
            raise V3SampleMPICampaignError(
                "V3 sampling workers require thread_limit=1."
            )
        if task.payload.get("phase") != phase:
            raise TaskManifestError("worker received a task from another phase.")
        q0 = _task_q0(task)
        if scan_context is None:
            scan_context = _prepare_promoted_scan(
                campaign.reference,
                degeneracy_tolerance_thz=(
                    campaign.degeneracy_tolerance_thz
                ),
                primitive_lattice=campaign.fc3_mapping.primitive.lattice,
            )
        if phase == _PHASE_MAXIMUM:
            maximum = _vectorized_maximum_promoted_block(
                campaign.reference,
                degeneracy_tolerance_thz=(
                    campaign.degeneracy_tolerance_thz
                ),
                primitive_lattice=campaign.fc3_mapping.primitive.lattice,
                target_grid_points=(q0,),
                scan_context=scan_context,
                verify_reference_hash=False,
            )
            payload = {"maximum_block": _block_payload(maximum)}
        else:
            assert global_maximum_block is not None
            floor = max(
                1.0e-30,
                1.0e-12 * global_maximum_block.strength_ev2,
            )
            accumulator = _BoundedSelectionAccumulator(
                floor=floor,
                global_maximum=global_maximum_block.strength_ev2,
                maximum_promoted_blocks=(
                    campaign.maximum_promoted_blocks
                ),
                target_bands=campaign.target_bands,
            )
            for candidate in _iter_promoted_blocks(
                campaign.reference,
                degeneracy_tolerance_thz=(
                    campaign.degeneracy_tolerance_thz
                ),
                primitive_lattice=campaign.fc3_mapping.primitive.lattice,
                track_duplicates=False,
                target_grid_points=(q0,),
                scan_context=scan_context,
                verify_reference_hash=False,
            ):
                accumulator.add(candidate)
            payload = {"accumulator": _accumulator_payload(accumulator)}
        document = _artifact_document(task, phase=phase, payload=payload)
        artifact = context.write_artifact_bytes(
            _canonical_bytes(document),
            suffix=".json",
        )
        return TaskResult.completed(
            artifact,
            metadata={
                "phase": phase,
                "q0": q0,
                "reference_sha256": campaign.reference.sha256,
            },
        )

    return worker


def _make_validator(
    campaign: V3SampleMPICampaign,
    *,
    phase: Literal["maximum", "reservoirs"],
    global_maximum_block: _PromotedBlock | None = None,
):
    def validator(path: Path, task: TaskSpec) -> Mapping[str, Any]:
        controls: dict[str, Any] = {}
        if phase == _PHASE_RESERVOIRS:
            assert global_maximum_block is not None
            controls = {
                "floor": max(
                    1.0e-30,
                    1.0e-12 * global_maximum_block.strength_ev2,
                ),
                "global_maximum": (
                    global_maximum_block.strength_ev2
                ),
                "maximum_promoted_blocks": (
                    campaign.maximum_promoted_blocks
                ),
                "target_bands": campaign.target_bands,
            }
        result = _read_task_artifact(
            path,
            task,
            expected_phase=phase,
            **controls,
        )
        if isinstance(result, _PromotedBlock):
            retained = 1
        else:
            retained = result.retained_candidate_slots
        return {
            "schema": _ARTIFACT_SCHEMA,
            "phase": phase,
            "q0": _task_q0(task),
            "retained_candidate_slots": retained,
        }

    return validator


def _phase_artifacts(
    campaign: V3SampleMPICampaign,
    manifest: TaskManifest,
    run_directory: Path,
    *,
    phase: Literal["maximum", "reservoirs"],
    global_maximum_block: _PromotedBlock | None = None,
) -> tuple[_PromotedBlock | _BoundedSelectionAccumulator, ...]:
    checkpoints = collect_task_checkpoints(
        manifest,
        run_directory / "task-outcomes",
        run_directory / "task-artifacts",
        artifact_validator=_make_validator(
            campaign,
            phase=phase,
            global_maximum_block=global_maximum_block,
        ),
        require_complete=True,
    )
    by_id = {value.task_id: value for value in checkpoints}
    results: list[_PromotedBlock | _BoundedSelectionAccumulator] = []
    for task in manifest.tasks:
        checkpoint = by_id[task.task_id]
        if checkpoint.status != "completed":
            raise V3SampleMPIMergeError(
                f"task {task.task_id!r} did not complete."
            )
        if checkpoint.artifact_relative_path is None:
            raise V3SampleMPIMergeError(
                f"task {task.task_id!r} has no artifact."
            )
        path = (
            run_directory
            / "task-artifacts"
            / checkpoint.artifact_relative_path
        )
        controls: dict[str, Any] = {}
        if phase == _PHASE_RESERVOIRS:
            assert global_maximum_block is not None
            controls = {
                "floor": max(
                    1.0e-30,
                    1.0e-12 * global_maximum_block.strength_ev2,
                ),
                "global_maximum": (
                    global_maximum_block.strength_ev2
                ),
                "maximum_promoted_blocks": (
                    campaign.maximum_promoted_blocks
                ),
                "target_bands": campaign.target_bands,
            }
        results.append(
            _read_task_artifact(
                path,
                task,
                expected_phase=phase,
                **controls,
            )
        )
    return tuple(results)


def _merge_maxima(
    values: Sequence[_PromotedBlock | _BoundedSelectionAccumulator],
) -> _PromotedBlock:
    maximum: _PromotedBlock | None = None
    for value in values:
        if not isinstance(value, _PromotedBlock):
            raise V3SampleMPIMergeError(
                "maximum phase produced a non-block artifact."
            )
        maximum = _canonical_extreme(maximum, value, maximum=True)
    if maximum is None:
        raise V3SampleMPIMergeError("maximum phase produced no artifacts.")
    return maximum


def _merge_reservoir(
    target: _SmallestHashReservoir,
    source: _SmallestHashReservoir,
) -> None:
    for candidate in source.values():
        target.add(candidate)


def _merge_accumulators(
    campaign: V3SampleMPICampaign,
    values: Sequence[_PromotedBlock | _BoundedSelectionAccumulator],
    *,
    global_maximum_block: _PromotedBlock,
) -> _BoundedSelectionAccumulator:
    floor = max(1.0e-30, 1.0e-12 * global_maximum_block.strength_ev2)
    merged = _BoundedSelectionAccumulator(
        floor=floor,
        global_maximum=global_maximum_block.strength_ev2,
        maximum_promoted_blocks=campaign.maximum_promoted_blocks,
        target_bands=campaign.target_bands,
    )
    for value in values:
        if not isinstance(value, _BoundedSelectionAccumulator):
            raise V3SampleMPIMergeError(
                "reservoir phase produced a non-accumulator artifact."
            )
        merged.viable_count += value.viable_count
        for name, count in value.stratum_counts.items():
            merged.stratum_counts[name] = (
                merged.stratum_counts.get(name, 0) + count
            )
        _merge_reservoir(
            merged.global_reservoir, value.global_reservoir
        )
        for name, source in value.stratum_reservoirs.items():
            target = merged.stratum_reservoirs.get(name)
            if target is None:
                target = _SmallestHashReservoir(2)
                merged.stratum_reservoirs[name] = target
            _merge_reservoir(target, source)
        for key, source in value.strength_reservoirs.items():
            target = merged.strength_reservoirs.get(key)
            if target is None:
                target = _SmallestHashReservoir(2)
                merged.strength_reservoirs[key] = target
            _merge_reservoir(target, source)
        for band, source in value.target_band_reservoirs.items():
            _merge_reservoir(
                merged.target_band_reservoirs[band], source
            )
        for process, candidate in value.process_minima.items():
            merged.process_minima[process] = _canonical_extreme(
                merged.process_minima.get(process),
                candidate,
                maximum=False,
            )
        for process, candidate in value.process_maxima.items():
            merged.process_maxima[process] = _canonical_extreme(
                merged.process_maxima.get(process),
                candidate,
                maximum=True,
            )
        if value.global_maximum_block is not None:
            merged.global_maximum_block = _canonical_extreme(
                merged.global_maximum_block,
                value.global_maximum_block,
                maximum=True,
            )
    if (
        merged.global_maximum_block is None
        or merged.global_maximum_block.key != global_maximum_block.key
        or merged.global_maximum_block.strength_ev2
        != global_maximum_block.strength_ev2
    ):
        raise V3SampleMPIMergeError(
            "reservoir phase did not recover the phase-one global maximum."
        )
    if merged.retained_candidate_slots > merged.retention_bound:
        raise V3SampleMPIMergeError(
            "merged accumulator exceeded its fixed retention bound."
        )
    return merged


def _scientific_payload(
    campaign: V3SampleMPICampaign,
    accumulator: _BoundedSelectionAccumulator,
) -> dict[str, Any]:
    selected, required_strata, empty_strata = _select_bounded_blocks(
        accumulator,
        minimum_promoted_blocks=campaign.minimum_promoted_blocks,
    )
    return {
        "schema": V3_SAMPLE_MANIFEST_SCHEMA,
        "schema_version": 1,
        "inputs": campaign.input_hashes,
        "mapping_sha256": campaign.mapping_hashes,
        "mesh": list(campaign.reference.mesh),
        "grg2bzg_sha256": array_sha256(campaign.reference.grg2bzg),
        "bzg2grg_sha256": array_sha256(campaign.reference.bzg2grg),
        "degeneracy_tolerance_thz": (
            campaign.degeneracy_tolerance_thz
        ),
        "selection_algorithm": V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM,
        "selection_algorithm_version": (
            V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
        ),
        "robust_nonzero_floor_ev2": accumulator.floor,
        "coverage": {
            "cover_all_target_grid_points": False,
            "cover_all_target_bands": True,
            "required_processes": ["normal", "umklapp"],
            "required_strata": list(required_strata),
            "empty_strata": list(empty_strata),
            "minimum_promoted_blocks": (
                campaign.minimum_promoted_blocks
            ),
        },
        "samples": [value.json_record() for value in selected],
    }


def _require_thread_environment_one(communicator: Any) -> None:
    local = {
        name: os.environ.get(name) for name in _THREAD_ENVIRONMENT
    }
    gathered = (
        (local,)
        if int(communicator.Get_size()) == 1
        else tuple(communicator.allgather(local))
    )
    if any(
        any(record.get(name) != "1" for name in _THREAD_ENVIRONMENT)
        for record in gathered
    ):
        raise V3SampleMPICampaignError(
            "Every MPI rank must set OMP, OpenBLAS, MKL, and NumExpr "
            "thread counts exactly to 1."
        )


def _collective_reference_identity_check(
    campaign: V3SampleMPICampaign,
    communicator: Any,
    *,
    full_hash_on_root: bool,
) -> None:
    stat = campaign.reference.path.stat()
    local_stat = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns)
    local_ok = local_stat == campaign.reference_stat_identity
    size = int(communicator.Get_size())
    rank = int(communicator.Get_rank())
    observed = (
        (local_ok,)
        if size == 1
        else tuple(communicator.allgather(local_ok))
    )
    if not all(observed):
        raise V3SampleMPICampaignError(
            "reference HDF5 stat identity changed during the campaign."
        )
    hash_ok = True
    if full_hash_on_root and rank == 0:
        hash_ok = (
            _file_sha256(campaign.reference.path)
            == campaign.reference.sha256
        )
    if size > 1:
        hash_ok = bool(communicator.bcast(hash_ok, root=0))
    if not hash_ok:
        raise V3SampleMPICampaignError(
            "reference HDF5 SHA256 changed during the campaign."
        )


def _collective_source_identity_check(
    campaign: V3SampleMPICampaign,
    communicator: Any,
) -> None:
    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    outcome: dict[str, Any] | None = None
    if rank == 0:
        try:
            current = current_v3_sampling_source_identity(
                campaign.source_identity["repository"],
                require_clean=campaign.require_authenticated_source,
            )
            outcome = {
                "ok": (
                    _canonical_json(current)
                    == _canonical_json(campaign.source_identity)
                ),
                "error": None,
            }
        except Exception as exc:
            outcome = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if size > 1:
        outcome = communicator.bcast(outcome, root=0)
    assert outcome is not None
    if not outcome["ok"]:
        suffix = "" if outcome["error"] is None else f" {outcome['error']}"
        raise V3SampleMPICampaignError(
            "Git, imported code, runtime, or bound source-file identity "
            f"changed during the campaign.{suffix}"
        )


def _publish_source_identity_evidence(
    campaign: V3SampleMPICampaign,
    run_directory: Path,
    communicator: Any,
) -> None:
    """Collectively publish or verify the immutable full source record."""

    rank = int(communicator.Get_rank())
    size = int(communicator.Get_size())
    outcome: dict[str, Any] | None = None
    if rank == 0:
        payload = {
            "schema": _SOURCE_IDENTITY_SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "campaign_sha256": campaign.campaign_sha256,
            "source_identity_sha256": _payload_sha256(
                campaign.source_identity
            ),
            "source_identity": campaign.source_identity,
        }
        try:
            mpi_tasks._publish_immutable(  # noqa: SLF001
                run_directory / "source-identity.json",
                _canonical_bytes(payload),
                conflict_type=TaskManifestError,
            )
            outcome = {"ok": True, "error": None}
        except Exception as exc:
            outcome = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
    if size > 1:
        outcome = communicator.bcast(outcome, root=0)
    assert outcome is not None
    if not outcome["ok"]:
        raise TaskManifestError(
            "cannot publish or verify immutable V3 source identity evidence: "
            f"{outcome['error']}"
        )


def _evidence_file_record(
    root: Path,
    path: Path,
    *,
    role: str,
) -> dict[str, Any]:
    """Hash a regular evidence file and reject symlink traversal."""

    root = _resolve_nonsymlink_path(
        root,
        name="orchestration run directory",
        must_exist=True,
        expected_kind="directory",
        error_type=V3SampleMPIArtifactError,
    )
    candidate = Path(path).expanduser()
    candidate = (
        candidate
        if candidate.is_absolute()
        else Path.cwd() / candidate
    )
    try:
        lexical = candidate.relative_to(root)
    except ValueError as exc:
        raise V3SampleMPIArtifactError(
            "orchestration evidence file lies outside its run directory."
        ) from exc
    if (
        lexical == Path(".")
        or lexical.is_absolute()
        or ".." in lexical.parts
        or any(part in {"", "."} for part in lexical.parts)
    ):
        raise V3SampleMPIArtifactError(
            "orchestration evidence path is not canonical."
        )
    resolved = _resolve_nonsymlink_path(
        candidate,
        name="orchestration evidence",
        must_exist=True,
        expected_kind="file",
        error_type=V3SampleMPIArtifactError,
    )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise V3SampleMPIArtifactError(
            "orchestration evidence resolves outside its run directory."
        ) from exc
    stat = resolved.stat()
    return {
        "role": role,
        "relative_path": lexical.as_posix(),
        "size_bytes": int(stat.st_size),
        "sha256": _file_sha256(resolved),
    }


def _phase_evidence_payload(
    campaign: V3SampleMPICampaign,
    *,
    root: Path,
    phase: Literal["maximum", "reservoirs"],
    manifest: TaskManifest,
    report: MPIExecutionReport,
    directory_name: str,
) -> dict[str, Any]:
    phase_root = root / directory_name
    persisted = load_task_manifest(phase_root / "task-manifest.json")
    if persisted.manifest_sha256 != manifest.manifest_sha256:
        raise TaskManifestError(
            f"persisted {phase} manifest differs from the in-memory manifest."
        )
    artifact_paths = sorted(
        (phase_root / "task-artifacts").glob("*.json"),
        key=lambda path: path.name,
    )
    checkpoint_paths = sorted(
        (phase_root / "task-outcomes").glob("*.task.json"),
        key=lambda path: path.name,
    )
    execution_paths = sorted(
        (phase_root / "executions").glob("*.json"),
        key=lambda path: path.name,
    )
    if (
        len(artifact_paths) != len(manifest.tasks)
        or len(checkpoint_paths) != len(manifest.tasks)
        or not execution_paths
        or {path.name for path in phase_root.iterdir()}
        != {
            "task-manifest.json",
            "executions",
            "task-artifacts",
            "task-outcomes",
        }
        or {path.name for path in (phase_root / "task-artifacts").iterdir()}
        != {path.name for path in artifact_paths}
        or {path.name for path in (phase_root / "task-outcomes").iterdir()}
        != {path.name for path in checkpoint_paths}
        or {path.name for path in (phase_root / "executions").iterdir()}
        != {path.name for path in execution_paths}
    ):
        raise V3SampleMPIArtifactError(
            f"{phase} evidence is incomplete."
        )
    selected_name = f"{report.execution_id}.json"
    if selected_name not in {path.name for path in execution_paths}:
        raise V3SampleMPIArtifactError(
            f"{phase} selected execution record is missing."
        )
    execution_history = [
        _strict_envelope_payload(
            path,
            name=f"{phase} MPI execution",
        )[0]
        for path in execution_paths
    ]
    origin_lineage = mpi_tasks.task_execution_mpi_origin_lineage_evidence(
        report,
        execution_history,
        require_mpi_origin=campaign.require_authenticated_source,
    )
    origin_execution_ids = (
        {
            str(record["task_id"]): str(
                record["origin_execution_id"]
            )
            for record in origin_lineage["lineage"]
        }
        if origin_lineage["passed"]
        else None
    )
    physical_cores = mpi_tasks.task_assignment_physical_core_evidence(
        report.assignments,
        require_available=campaign.require_authenticated_source,
        origin_execution_ids=origin_execution_ids,
    )
    return {
        "phase": phase,
        "directory": directory_name,
        "task_manifest_sha256": manifest.manifest_sha256,
        "selected_execution_id": report.execution_id,
        "selected_execution_sha256": report.execution_sha256,
        "selected_execution_physical_cores": (
            mpi_tasks._normalize_json(  # noqa: SLF001
                physical_cores
            )
        ),
        "selected_execution_origin_lineage": (
            mpi_tasks._normalize_json(  # noqa: SLF001
                origin_lineage
            )
        ),
        "task_ids": [task.task_id for task in manifest.tasks],
        "files": {
            "task_manifest": _evidence_file_record(
                root,
                phase_root / "task-manifest.json",
                role=f"{phase}.task_manifest",
            ),
            "executions": [
                _evidence_file_record(
                    root,
                    path,
                    role=f"{phase}.execution",
                )
                for path in execution_paths
            ],
            "task_artifacts": [
                _evidence_file_record(
                    root,
                    path,
                    role=f"{phase}.task_artifact",
                )
                for path in artifact_paths
            ],
            "task_checkpoints": [
                _evidence_file_record(
                    root,
                    path,
                    role=f"{phase}.task_checkpoint",
                )
                for path in checkpoint_paths
            ],
        },
    }


def _orchestration_payload(
    campaign: V3SampleMPICampaign,
    *,
    root: Path,
    scientific_manifest: V3InteractionSampleManifest,
    maximum_manifest: TaskManifest,
    maximum_report: MPIExecutionReport,
    reservoir_manifest: TaskManifest,
    reservoir_report: MPIExecutionReport,
    global_maximum: _PromotedBlock,
) -> dict[str, Any]:
    maximum_payload = _block_payload(global_maximum)
    return {
        "schema": _ORCHESTRATION_SCHEMA,
        "schema_version": _SCHEMA_VERSION,
        "scientific_manifest": {
            "sha256": scientific_manifest.sha256,
            "schema": V3_SAMPLE_MANIFEST_SCHEMA,
            "selection_algorithm": (
                V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM
            ),
            "selection_algorithm_version": (
                V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
            ),
            "promoted_block_count": len(scientific_manifest.samples),
        },
        "campaign": {
            "identity": campaign.identity_payload(),
            "campaign_sha256": campaign.campaign_sha256,
            "source_identity_sha256": _payload_sha256(
                campaign.source_identity
            ),
            "source_authenticated": campaign.source_identity[
                "authenticated"
            ],
        },
        "execution_contract": {
            "parallel_boundary": (
                "one_complete_target_q_hdf5_slab_per_task"
            ),
            "mpi_messages": (
                "task_identifiers_status_and_interphase_global_maximum_payload"
            ),
            "interphase_broadcast": {
                "sender": "rank_zero",
                "recipients": "all_ranks",
                "payload": "canonical_global_maximum_promoted_block",
                "floating_point_collective_reduction": False,
            },
            "floating_point_mpi_reductions": False,
            "canonical_merge": (
                "rank_zero_q0_manifest_order_integer_counts_and_fixed_"
                "reservoirs"
            ),
            "phase_order": ["maximum", "reservoirs"],
            "worker_thread_limit": 1,
            "scientific_payload_contains_orchestration": False,
        },
        "merge": {
            "global_maximum_block": maximum_payload,
            "global_maximum_block_sha256": _payload_sha256(
                maximum_payload
            ),
            "target_q_order": list(campaign.target_grid_points),
            "floating_point_reduction_operator": "none",
        },
        "source_identity_file": _evidence_file_record(
            root,
            root / "source-identity.json",
            role="source_identity",
        ),
        "phases": [
            _phase_evidence_payload(
                campaign,
                root=root,
                phase=_PHASE_MAXIMUM,
                manifest=maximum_manifest,
                report=maximum_report,
                directory_name="phase-maximum",
            ),
            _phase_evidence_payload(
                campaign,
                root=root,
                phase=_PHASE_RESERVOIRS,
                manifest=reservoir_manifest,
                report=reservoir_report,
                directory_name="phase-reservoirs",
            ),
        ],
    }


def _publish_orchestration_evidence(
    campaign: V3SampleMPICampaign,
    *,
    root: Path,
    scientific_manifest: V3InteractionSampleManifest,
    maximum_manifest: TaskManifest,
    maximum_report: MPIExecutionReport,
    reservoir_manifest: TaskManifest,
    reservoir_report: MPIExecutionReport,
    global_maximum: _PromotedBlock,
) -> tuple[Path, str, str]:
    payload = _orchestration_payload(
        campaign,
        root=root,
        scientific_manifest=scientific_manifest,
        maximum_manifest=maximum_manifest,
        maximum_report=maximum_report,
        reservoir_manifest=reservoir_manifest,
        reservoir_report=reservoir_report,
        global_maximum=global_maximum,
    )
    payload_sha256 = _payload_sha256(payload)
    document = {
        "payload": payload,
        "payload_sha256": payload_sha256,
    }
    target = scientific_manifest.path.with_name(
        f"{scientific_manifest.path.name}{_ORCHESTRATION_SUFFIX}."
        f"{payload_sha256}.json"
    )
    mpi_tasks._publish_immutable(  # noqa: SLF001
        target,
        _canonical_bytes(document),
        conflict_type=TaskManifestError,
    )
    return target, _file_sha256(target), payload_sha256


@dataclass(frozen=True, slots=True)
class V3SampleMPIResult:
    """Rank-zero result and separate orchestration evidence."""

    manifest: V3InteractionSampleManifest
    phase_maximum_execution: MPIExecutionReport
    phase_reservoir_execution: MPIExecutionReport
    campaign_sha256: str
    global_maximum_block_sha256: str
    orchestration_path: Path
    orchestration_file_sha256: str
    orchestration_payload_sha256: str


def run_v3_sample_manifest_mpi(
    campaign: V3SampleMPICampaign,
    *,
    run_directory: str | Path,
    output_path: str | Path,
    comm: Any | None = None,
    require_mpi: bool = False,
) -> V3SampleMPIResult | None:
    """Run or exactly resume a two-phase bounded-v2 sampling campaign."""

    if not isinstance(campaign, V3SampleMPICampaign):
        raise TypeError("campaign must be a V3SampleMPICampaign.")
    communicator = mpi_tasks._resolve_communicator(  # noqa: SLF001
        comm,
        require_mpi=require_mpi,
    )
    rank = int(communicator.Get_rank())
    root = _resolve_nonsymlink_path(
        run_directory,
        name="V3 MPI run directory",
        must_exist=False,
        expected_kind="directory",
    )
    target = _resolve_nonsymlink_path(
        output_path,
        name="V3 scientific manifest output",
        must_exist=False,
        expected_kind="file",
    )
    _require_thread_environment_one(communicator)
    _collective_source_identity_check(campaign, communicator)
    _publish_source_identity_evidence(campaign, root, communicator)
    _collective_reference_identity_check(
        campaign,
        communicator,
        full_hash_on_root=False,
    )

    maximum_manifest = campaign.task_manifest(_PHASE_MAXIMUM)
    maximum_report = execute_task_manifest(
        maximum_manifest,
        _make_worker(campaign, phase=_PHASE_MAXIMUM),
        root / "phase-maximum",
        comm=communicator,
        require_mpi=require_mpi,
        thread_limit=1,
        artifact_validator=_make_validator(
            campaign, phase=_PHASE_MAXIMUM
        ),
    )
    _collective_reference_identity_check(
        campaign,
        communicator,
        full_hash_on_root=False,
    )
    maximum_payload: dict[str, Any] | None = None
    if rank == 0:
        maxima = _phase_artifacts(
            campaign,
            maximum_manifest,
            root / "phase-maximum",
            phase=_PHASE_MAXIMUM,
        )
        maximum_payload = _block_payload(_merge_maxima(maxima))
    if int(communicator.Get_size()) > 1:
        maximum_payload = communicator.bcast(maximum_payload, root=0)
    if not isinstance(maximum_payload, dict):
        raise V3SampleMPIMergeError(
            "rank zero did not broadcast a global maximum."
        )
    global_maximum = _block_from_payload(maximum_payload)

    reservoir_manifest = campaign.task_manifest(
        _PHASE_RESERVOIRS,
        global_maximum_block=global_maximum,
    )
    reservoir_report = execute_task_manifest(
        reservoir_manifest,
        _make_worker(
            campaign,
            phase=_PHASE_RESERVOIRS,
            global_maximum_block=global_maximum,
        ),
        root / "phase-reservoirs",
        comm=communicator,
        require_mpi=require_mpi,
        thread_limit=1,
        artifact_validator=_make_validator(
            campaign,
            phase=_PHASE_RESERVOIRS,
            global_maximum_block=global_maximum,
        ),
    )
    _collective_reference_identity_check(
        campaign,
        communicator,
        full_hash_on_root=True,
    )
    _collective_source_identity_check(campaign, communicator)
    _publish_source_identity_evidence(campaign, root, communicator)
    if rank != 0:
        return None
    assert maximum_report is not None
    assert reservoir_report is not None
    local_accumulators = _phase_artifacts(
        campaign,
        reservoir_manifest,
        root / "phase-reservoirs",
        phase=_PHASE_RESERVOIRS,
        global_maximum_block=global_maximum,
    )
    merged = _merge_accumulators(
        campaign,
        local_accumulators,
        global_maximum_block=global_maximum,
    )
    payload = _scientific_payload(campaign, merged)
    target = _resolve_nonsymlink_path(
        target,
        name="V3 scientific manifest output",
        must_exist=target.exists(),
        expected_kind="file",
    )
    if target.exists():
        existing = json.loads(target.read_text(encoding="utf-8"))
        if existing != payload:
            raise V3SampleMPIMergeError(
                "existing final manifest differs from this exact campaign."
            )
        manifest = read_v3_sample_manifest(target)
    else:
        manifest = write_v3_sample_manifest(target, payload)
    (
        orchestration_path,
        orchestration_file_sha256,
        orchestration_payload_sha256,
    ) = _publish_orchestration_evidence(
        campaign,
        root=root,
        scientific_manifest=manifest,
        maximum_manifest=maximum_manifest,
        maximum_report=maximum_report,
        reservoir_manifest=reservoir_manifest,
        reservoir_report=reservoir_report,
        global_maximum=global_maximum,
    )
    validated_orchestration = validate_v3_mpi_orchestration_evidence(
        manifest.path,
        orchestration_path,
        root,
        require_authenticated_source=(
            campaign.require_authenticated_source
        ),
        require_mpi=int(communicator.Get_size()) > 1,
        expected_source_revision=campaign.source_revision,
        expected_source_tree_sha256=campaign.source_tree_sha256,
        expected_environment_sha256=campaign.environment_sha256,
        expected_source_identity=campaign.source_identity,
        expected_mesh=campaign.reference.mesh,
        expected_target_grid_points=campaign.target_grid_points,
        expected_target_bands=campaign.target_bands,
    )
    if (
        validated_orchestration.file_sha256
        != orchestration_file_sha256
        or validated_orchestration.payload_sha256
        != orchestration_payload_sha256
    ):
        raise V3SampleMPIArtifactError(
            "published V3 orchestration evidence changed during validation."
        )
    return V3SampleMPIResult(
        manifest=manifest,
        phase_maximum_execution=maximum_report,
        phase_reservoir_execution=reservoir_report,
        campaign_sha256=campaign.campaign_sha256,
        global_maximum_block_sha256=_payload_sha256(maximum_payload),
        orchestration_path=orchestration_path,
        orchestration_file_sha256=orchestration_file_sha256,
        orchestration_payload_sha256=orchestration_payload_sha256,
    )


@dataclass(frozen=True, slots=True)
class V3MPIOrchestrationEvidence:
    """Validated link from a scientific manifest to complete MPI evidence."""

    path: Path
    file_sha256: str
    payload_sha256: str
    campaign_sha256: str
    source_identity_sha256: str
    source_authenticated: bool
    selected_execution_ids: tuple[str, str]
    selected_communicator_sizes: tuple[int, int]
    selected_modes: tuple[str, str]
    restart_resume_recorded: bool
    physical_core_binding_passed: bool
    mpi_origin_lineage_passed: bool
    source_revision: str
    source_tree_sha256: str
    environment_sha256: str
    mesh: tuple[int, int, int]
    target_grid_points: tuple[int, ...]
    target_bands: tuple[int, ...]


def _validated_evidence_record(
    root: Path,
    value: Any,
    *,
    expected_role: str,
    expected_prefix: str | None = None,
) -> Path:
    if not isinstance(value, dict) or set(value) != {
        "role",
        "relative_path",
        "size_bytes",
        "sha256",
    }:
        raise V3SampleMPIArtifactError(
            "an orchestration evidence file record has an invalid schema."
        )
    if value["role"] != expected_role:
        raise V3SampleMPIArtifactError(
            "an orchestration evidence file role is mismatched."
        )
    relative = value["relative_path"]
    if not isinstance(relative, str) or "\\" in relative:
        raise V3SampleMPIArtifactError(
            "an orchestration evidence relative path is invalid."
        )
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or not pure.parts
        or any(part in {"", ".", ".."} for part in pure.parts)
        or pure.as_posix() != relative
    ):
        raise V3SampleMPIArtifactError(
            "an orchestration evidence path is noncanonical or traverses."
        )
    if expected_prefix is not None and (
        not relative.startswith(f"{expected_prefix}/")
    ):
        raise V3SampleMPIArtifactError(
            "an orchestration evidence path is outside its declared phase."
        )
    size = value["size_bytes"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or not isinstance(value["sha256"], str)
        or _SHA256_PATTERN.fullmatch(value["sha256"]) is None
    ):
        raise V3SampleMPIArtifactError(
            "an orchestration evidence size or SHA256 is invalid."
        )
    path = root.joinpath(*pure.parts)
    observed = _evidence_file_record(
        root,
        path,
        role=expected_role,
    )
    if observed != value:
        raise V3SampleMPIArtifactError(
            f"orchestration evidence file changed: {relative}"
        )
    return path


def _strict_envelope_payload(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], str]:
    document = _strict_json(path)
    if set(document) != {"payload", "payload_sha256"}:
        raise V3SampleMPIArtifactError(f"{name} envelope is invalid.")
    payload = document["payload"]
    digest = document["payload_sha256"]
    if (
        not isinstance(payload, dict)
        or not isinstance(digest, str)
        or _SHA256_PATTERN.fullmatch(digest) is None
        or digest != _payload_sha256(payload)
    ):
        raise V3SampleMPIArtifactError(
            f"{name} payload failed its SHA256 check."
        )
    return payload, digest


def _validate_phase_file_inventory(
    root: Path,
    phase: Mapping[str, Any],
) -> dict[str, Any]:
    phase_name = phase["phase"]
    directory = phase["directory"]
    expected_directory = (
        "phase-maximum"
        if phase_name == _PHASE_MAXIMUM
        else "phase-reservoirs"
    )
    if directory != expected_directory:
        raise V3SampleMPIArtifactError(
            "orchestration phase directory is not canonical."
        )
    phase_root = root / directory
    if phase_root.is_symlink() or not phase_root.is_dir():
        raise V3SampleMPIArtifactError(
            f"{phase_name} evidence directory is missing or a symlink."
        )
    if {path.name for path in phase_root.iterdir()} != {
        "task-manifest.json",
        "executions",
        "task-artifacts",
        "task-outcomes",
    }:
        raise V3SampleMPIArtifactError(
            f"{phase_name} evidence directory has missing or extra entries."
        )
    files = phase["files"]
    if not isinstance(files, dict) or set(files) != {
        "task_manifest",
        "executions",
        "task_artifacts",
        "task_checkpoints",
    }:
        raise V3SampleMPIArtifactError(
            f"{phase_name} file inventory schema is invalid."
        )
    manifest_path = _validated_evidence_record(
        root,
        files["task_manifest"],
        expected_role=f"{phase_name}.task_manifest",
        expected_prefix=directory,
    )
    collections: dict[str, list[Path]] = {}
    specifications = {
        "executions": ("execution", "executions", ".json"),
        "task_artifacts": ("task_artifact", "task-artifacts", ".json"),
        "task_checkpoints": (
            "task_checkpoint",
            "task-outcomes",
            ".task.json",
        ),
    }
    for key, (role, subdirectory, suffix) in specifications.items():
        raw_records = files[key]
        if not isinstance(raw_records, list) or not raw_records:
            raise V3SampleMPIArtifactError(
                f"{phase_name} {key} inventory must be nonempty."
            )
        paths = [
            _validated_evidence_record(
                root,
                record,
                expected_role=f"{phase_name}.{role}",
                expected_prefix=f"{directory}/{subdirectory}",
            )
            for record in raw_records
        ]
        relative_names = [path.name for path in paths]
        if (
            relative_names != sorted(relative_names)
            or len(set(relative_names)) != len(relative_names)
            or any(not name.endswith(suffix) for name in relative_names)
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} {key} inventory is not canonical."
            )
        actual_directory = phase_root / subdirectory
        if (
            actual_directory.is_symlink()
            or not actual_directory.is_dir()
            or {path.name for path in actual_directory.iterdir()}
            != set(relative_names)
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} {key} inventory omits or adds files."
            )
        collections[key] = paths
    return {
        "manifest_path": manifest_path,
        **collections,
    }


def _execution_payload(
    path: Path,
    *,
    manifest: TaskManifest,
    artifact_records: Mapping[str, str],
    checkpoint_records: Mapping[str, str],
) -> tuple[dict[str, Any], str]:
    payload, digest = _strict_envelope_payload(
        path,
        name="MPI execution",
    )
    expected_keys = {
        "schema_name",
        "schema_version",
        "execution_id",
        "manifest_sha256",
        "mode",
        "communicator_size",
        "coordinator_hostname",
        "started_utc",
        "finished_utc",
        "elapsed_seconds",
        "thread_limit",
        "complete",
        "assignments",
        "missing_task_ids",
        "infrastructure_errors",
    }
    if (
        set(payload) != expected_keys
        or payload["schema_name"] != "negfpy_mpi_execution_manifest"
        or payload["schema_version"] != 1
        or payload["manifest_sha256"] != manifest.manifest_sha256
        or payload["thread_limit"] != 1
        or payload["mode"] not in {"serial", "mpi"}
        or not isinstance(payload["coordinator_hostname"], str)
        or not payload["coordinator_hostname"]
        or not isinstance(payload["started_utc"], str)
        or not payload["started_utc"]
        or not isinstance(payload["finished_utc"], str)
        or not payload["finished_utc"]
        or isinstance(payload["elapsed_seconds"], bool)
        or not isinstance(payload["elapsed_seconds"], (int, float))
        or not math.isfinite(float(payload["elapsed_seconds"]))
        or float(payload["elapsed_seconds"]) < 0.0
        or type(payload["complete"]) is not bool
    ):
        raise V3SampleMPIArtifactError(
            "MPI execution record violates the task-history contract."
        )
    size = payload["communicator_size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size <= 0
        or (payload["mode"] == "serial" and size != 1)
        or (payload["mode"] == "mpi" and size < 2)
    ):
        raise V3SampleMPIArtifactError(
            "MPI execution mode and communicator size disagree."
        )
    assignments = payload["assignments"]
    expected_task_ids = [task.task_id for task in manifest.tasks]
    canonical_positions = {
        task_id: ordinal
        for ordinal, task_id in enumerate(expected_task_ids)
    }
    assignment_keys = {
        "task_id",
        "status",
        "rank",
        "hostname",
        "elapsed_seconds",
        "resumed",
        "cpu_affinity",
        "thread_cap_passed",
        "checkpoint_relative_path",
        "checkpoint_file_sha256",
        "artifact_relative_path",
        "artifact_sha256",
        "orphan_artifact_adopted",
        "cpu_affinity_topology",
        "origin_communicator_size",
    }
    if (
        not isinstance(assignments, list)
        or any(not isinstance(item, dict) for item in assignments)
    ):
        raise V3SampleMPIArtifactError(
            "MPI execution assignments are not an array of objects."
        )
    assignment_task_ids = [item.get("task_id") for item in assignments]
    if (
        any(task_id not in canonical_positions for task_id in assignment_task_ids)
        or len(set(assignment_task_ids)) != len(assignment_task_ids)
        or assignment_task_ids
        != sorted(
            assignment_task_ids,
            key=canonical_positions.__getitem__,
        )
    ):
        raise V3SampleMPIArtifactError(
            "MPI execution assignments are duplicated, unknown, or "
            "noncanonical."
        )
    for assignment in assignments:
        if (
            not isinstance(assignment, dict)
            or set(assignment) != assignment_keys
            or assignment["status"] != "completed"
            or assignment["thread_cap_passed"] is not True
            or type(assignment["resumed"]) is not bool
            or type(assignment["orphan_artifact_adopted"]) is not bool
            or (
                assignment["orphan_artifact_adopted"]
                and not assignment["resumed"]
            )
            or isinstance(assignment["rank"], bool)
            or not isinstance(assignment["rank"], int)
            or assignment["rank"] < 0
            or isinstance(assignment["origin_communicator_size"], bool)
            or not isinstance(
                assignment["origin_communicator_size"], int
            )
            or assignment["origin_communicator_size"] <= assignment["rank"]
            or not isinstance(
                assignment["checkpoint_relative_path"], str
            )
            or "/" in assignment["checkpoint_relative_path"]
            or "\\" in assignment["checkpoint_relative_path"]
            or not isinstance(assignment["artifact_relative_path"], str)
            or "/" in assignment["artifact_relative_path"]
            or "\\" in assignment["artifact_relative_path"]
            or not isinstance(
                assignment["checkpoint_file_sha256"], str
            )
            or _SHA256_PATTERN.fullmatch(
                assignment["checkpoint_file_sha256"]
            )
            is None
            or not isinstance(assignment["artifact_sha256"], str)
            or _SHA256_PATTERN.fullmatch(assignment["artifact_sha256"])
            is None
            or assignment["checkpoint_file_sha256"]
            != checkpoint_records.get(
                assignment["checkpoint_relative_path"]
            )
            or assignment["artifact_sha256"]
            != artifact_records.get(
                assignment["artifact_relative_path"]
            )
        ):
            raise V3SampleMPIArtifactError(
                "MPI execution assignment conflicts with task evidence."
            )
    missing = payload["missing_task_ids"]
    errors = payload["infrastructure_errors"]
    error_keys = {
        "task_id",
        "rank",
        "error_type",
        "error_message",
    }
    if (
        not isinstance(missing, list)
        or any(task_id not in canonical_positions for task_id in missing)
        or len(set(missing)) != len(missing)
        or missing
        != sorted(missing, key=canonical_positions.__getitem__)
        or set(assignment_task_ids).intersection(missing)
        or set(assignment_task_ids).union(missing)
        != set(expected_task_ids)
        or not isinstance(errors, list)
        or any(
            not isinstance(item, dict)
            or set(item) != error_keys
            or not isinstance(item["task_id"], str)
            or not item["task_id"]
            or isinstance(item["rank"], bool)
            or not isinstance(item["rank"], int)
            or item["rank"] < 0
            or not isinstance(item["error_type"], str)
            or not item["error_type"]
            or not isinstance(item["error_message"], str)
            for item in errors
        )
        or payload["complete"] is not (not missing and not errors)
    ):
        raise V3SampleMPIArtifactError(
            "MPI execution terminal state is inconsistent."
        )
    return payload, digest


def validate_v3_mpi_orchestration_evidence(
    scientific_manifest_path: str | Path,
    orchestration_path: str | Path,
    run_directory: str | Path,
    *,
    require_authenticated_source: bool = True,
    require_mpi: bool = True,
    expected_source_revision: str | None = None,
    expected_source_tree_sha256: str | None = None,
    expected_environment_sha256: str | None = None,
    expected_source_identity: Mapping[str, Any] | None = None,
    expected_mesh: Sequence[int] | None = None,
    expected_target_grid_points: Sequence[int] | None = None,
    expected_target_bands: Sequence[int] | None = None,
) -> V3MPIOrchestrationEvidence:
    """Rehash, reparse, and canonically remerge complete V3 MPI evidence.

    Release consumers should call this for bounded-v2 manifests and bind the
    returned ``file_sha256`` and ``payload_sha256`` into their own strict
    report. Historical serial v1 manifests do not use this contract.
    """

    if type(require_authenticated_source) is not bool or type(require_mpi) is not bool:
        raise TypeError(
            "require_authenticated_source and require_mpi must be boolean."
        )
    expected_values = {
        "expected_source_revision": expected_source_revision,
        "expected_source_tree_sha256": expected_source_tree_sha256,
        "expected_environment_sha256": expected_environment_sha256,
        "expected_source_identity": expected_source_identity,
        "expected_mesh": expected_mesh,
        "expected_target_grid_points": expected_target_grid_points,
        "expected_target_bands": expected_target_bands,
    }
    if require_authenticated_source and any(
        value is None for value in expected_values.values()
    ):
        missing = sorted(
            name for name, value in expected_values.items() if value is None
        )
        raise V3SampleMPIArtifactError(
            "release V3 orchestration validation requires the explicit "
            "current candidate identity and complete target: "
            + ", ".join(missing)
        )

    def expected_indices(
        value: Sequence[int] | None,
        *,
        name: str,
    ) -> tuple[int, ...] | None:
        if value is None:
            return None
        if isinstance(value, (str, bytes)):
            raise V3SampleMPIArtifactError(
                f"{name} must be a nonempty integer sequence."
            )
        result = tuple(value)
        if (
            not result
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, np.integer))
                or int(item) < 0
                for item in result
            )
            or len(set(int(item) for item in result)) != len(result)
        ):
            raise V3SampleMPIArtifactError(
                f"{name} must contain unique nonnegative integers."
            )
        return tuple(int(item) for item in result)

    expected_qpoints = expected_indices(
        expected_target_grid_points,
        name="expected_target_grid_points",
    )
    expected_bands = expected_indices(
        expected_target_bands,
        name="expected_target_bands",
    )
    expected_mesh_tuple: tuple[int, int, int] | None = None
    if expected_mesh is not None:
        if isinstance(expected_mesh, (str, bytes)):
            raise V3SampleMPIArtifactError(
                "expected_mesh must contain three positive integers."
            )
        raw_mesh = tuple(expected_mesh)
        if (
            len(raw_mesh) != 3
            or any(
                isinstance(item, bool)
                or not isinstance(item, (int, np.integer))
                or int(item) <= 0
                for item in raw_mesh
            )
        ):
            raise V3SampleMPIArtifactError(
                "expected_mesh must contain three positive integers."
            )
        expected_mesh_tuple = tuple(
            int(item) for item in raw_mesh
        )  # type: ignore[assignment]
    if require_authenticated_source:
        assert expected_mesh_tuple is not None
        assert expected_qpoints is not None
        assert expected_bands is not None
        if expected_qpoints != tuple(
            range(math.prod(expected_mesh_tuple))
        ):
            raise V3SampleMPIArtifactError(
                "release V3 target q points must cover the complete mesh in "
                "canonical grid-point order."
            )
        if expected_bands != tuple(range(len(expected_bands))):
            raise V3SampleMPIArtifactError(
                "release V3 target bands must cover every band in canonical "
                "order."
            )
    if expected_source_revision is not None and (
        not isinstance(expected_source_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", expected_source_revision) is None
    ):
        raise V3SampleMPIArtifactError(
            "expected_source_revision must be a full lowercase Git SHA."
        )
    for name, value in (
        ("expected_source_tree_sha256", expected_source_tree_sha256),
        ("expected_environment_sha256", expected_environment_sha256),
    ):
        if value is not None:
            try:
                _require_sha256(value, name=name)
            except ValueError as exc:
                raise V3SampleMPIArtifactError(str(exc)) from exc
    normalized_expected_source: dict[str, Any] | None = None
    if expected_source_identity is not None:
        try:
            normalized_expected_source = _validated_source_identity(
                expected_source_identity,
                require_authenticated=require_authenticated_source,
            )
        except V3SampleMPICampaignError as exc:
            raise V3SampleMPIArtifactError(
                "expected current candidate source identity is invalid."
            ) from exc

    scientific_path = _resolve_nonsymlink_path(
        scientific_manifest_path,
        name="scientific V3 manifest",
        must_exist=True,
        expected_kind="file",
        error_type=V3SampleMPIArtifactError,
    )
    evidence_path = _resolve_nonsymlink_path(
        orchestration_path,
        name="V3 orchestration sidecar",
        must_exist=True,
        expected_kind="file",
        error_type=V3SampleMPIArtifactError,
    )
    root = _resolve_nonsymlink_path(
        run_directory,
        name="V3 orchestration run directory",
        must_exist=True,
        expected_kind="directory",
        error_type=V3SampleMPIArtifactError,
    )
    if (
        evidence_path.parent != scientific_path.parent
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration inputs must be adjacent regular files and a "
            "regular non-symlink run directory."
        )
    payload, payload_sha256 = _strict_envelope_payload(
        evidence_path,
        name="V3 MPI orchestration",
    )
    expected_name = (
        f"{scientific_path.name}{_ORCHESTRATION_SUFFIX}."
        f"{payload_sha256}.json"
    )
    if evidence_path.name != expected_name:
        raise V3SampleMPIArtifactError(
            "V3 orchestration sidecar name is not content-addressed to its "
            "payload."
        )
    expected_root_keys = {
        "schema",
        "schema_version",
        "scientific_manifest",
        "campaign",
        "execution_contract",
        "merge",
        "source_identity_file",
        "phases",
    }
    if (
        set(payload) != expected_root_keys
        or payload["schema"] != _ORCHESTRATION_SCHEMA
        or payload["schema_version"] != _SCHEMA_VERSION
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration sidecar schema is invalid."
        )

    scientific = payload["scientific_manifest"]
    if not isinstance(scientific, dict) or set(scientific) != {
        "sha256",
        "schema",
        "selection_algorithm",
        "selection_algorithm_version",
        "promoted_block_count",
    }:
        raise V3SampleMPIArtifactError(
            "scientific-manifest orchestration binding is invalid."
        )
    manifest = read_v3_sample_manifest(scientific_path)
    if (
        scientific["sha256"] != manifest.sha256
        or scientific["schema"] != V3_SAMPLE_MANIFEST_SCHEMA
        or scientific["selection_algorithm"]
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM
        or scientific["selection_algorithm_version"]
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
        or manifest.selection_algorithm
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM
        or manifest.selection_algorithm_version
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
        or scientific["promoted_block_count"] != len(manifest.samples)
    ):
        raise V3SampleMPIArtifactError(
            "orchestration sidecar does not bind this bounded-v2 scientific "
            "manifest."
        )
    raw_scientific = _read_unique_json_object(
        scientific_path,
        name="scientific V3 manifest",
    )

    campaign_record = payload["campaign"]
    if not isinstance(campaign_record, dict) or set(campaign_record) != {
        "identity",
        "campaign_sha256",
        "source_identity_sha256",
        "source_authenticated",
    }:
        raise V3SampleMPIArtifactError(
            "V3 orchestration campaign record is invalid."
        )
    identity = campaign_record["identity"]
    campaign_identity_keys = {
        "schema",
        "schema_version",
        "selection_mode",
        "selection_algorithm",
        "selection_algorithm_version",
        "source_revision",
        "source_tree_sha256",
        "environment_sha256",
        "source_identity_sha256",
        "source_authenticated",
        "inputs",
        "mapping_sha256",
        "mesh",
        "grg2bzg_sha256",
        "bzg2grg_sha256",
        "target_grid_points",
        "target_bands",
        "minimum_promoted_blocks",
        "maximum_promoted_blocks",
        "degeneracy_tolerance_thz",
    }
    if (
        not isinstance(identity, dict)
        or set(identity) != campaign_identity_keys
        or identity.get("schema") != _CAMPAIGN_SCHEMA
        or identity.get("schema_version") != _SCHEMA_VERSION
        or campaign_record["campaign_sha256"] != _payload_sha256(identity)
        or identity.get("selection_mode") != LARGE_MESH_BOUNDED_SELECTION
        or identity.get("selection_algorithm")
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM
        or identity.get("selection_algorithm_version")
        != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
        or identity.get("inputs") != raw_scientific.get("inputs")
        or identity.get("mapping_sha256")
        != raw_scientific.get("mapping_sha256")
        or identity.get("mesh") != raw_scientific.get("mesh")
        or not isinstance(identity.get("mesh"), list)
        or len(identity["mesh"]) != 3
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or item <= 0
            for item in identity["mesh"]
        )
        or identity.get("grg2bzg_sha256")
        != raw_scientific.get("grg2bzg_sha256")
        or identity.get("bzg2grg_sha256")
        != raw_scientific.get("bzg2grg_sha256")
        or identity.get("degeneracy_tolerance_thz")
        != raw_scientific.get("degeneracy_tolerance_thz")
        or identity.get("source_identity_sha256")
        != campaign_record["source_identity_sha256"]
        or identity.get("source_authenticated")
        is not campaign_record["source_authenticated"]
        or not isinstance(identity.get("target_grid_points"), list)
        or not identity["target_grid_points"]
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in identity["target_grid_points"]
        )
        or len(set(identity["target_grid_points"]))
        != len(identity["target_grid_points"])
        or not isinstance(identity.get("target_bands"), list)
        or not identity["target_bands"]
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in identity["target_bands"]
        )
        or len(set(identity["target_bands"])) != len(identity["target_bands"])
        or isinstance(identity.get("minimum_promoted_blocks"), bool)
        or not isinstance(identity["minimum_promoted_blocks"], int)
        or not 64 <= identity["minimum_promoted_blocks"] <= 512
        or isinstance(identity.get("maximum_promoted_blocks"), bool)
        or not isinstance(identity["maximum_promoted_blocks"], int)
        or not (
            identity["minimum_promoted_blocks"]
            <= identity["maximum_promoted_blocks"]
            <= 512
        )
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration campaign identity is inconsistent."
        )
    if (
        expected_source_revision is not None
        and identity["source_revision"] != expected_source_revision
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration source revision is not the expected current "
            "candidate."
        )
    if (
        expected_source_tree_sha256 is not None
        and identity["source_tree_sha256"]
        != expected_source_tree_sha256
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration source tree is not the expected current "
            "candidate."
        )
    if (
        expected_environment_sha256 is not None
        and identity["environment_sha256"]
        != expected_environment_sha256
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration environment is not the expected candidate "
            "environment."
        )
    if (
        expected_mesh_tuple is not None
        and tuple(identity["mesh"]) != expected_mesh_tuple
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration mesh is not the expected complete mesh."
        )
    if (
        expected_qpoints is not None
        and tuple(identity["target_grid_points"]) != expected_qpoints
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration target q points do not match the expected "
            "complete target."
        )
    if (
        expected_bands is not None
        and tuple(identity["target_bands"]) != expected_bands
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration target bands do not match the expected "
            "complete target."
        )

    contract = payload["execution_contract"]
    if contract != {
        "parallel_boundary": "one_complete_target_q_hdf5_slab_per_task",
        "mpi_messages": (
            "task_identifiers_status_and_interphase_global_maximum_payload"
        ),
        "interphase_broadcast": {
            "sender": "rank_zero",
            "recipients": "all_ranks",
            "payload": "canonical_global_maximum_promoted_block",
            "floating_point_collective_reduction": False,
        },
        "floating_point_mpi_reductions": False,
        "canonical_merge": (
            "rank_zero_q0_manifest_order_integer_counts_and_fixed_reservoirs"
        ),
        "phase_order": ["maximum", "reservoirs"],
        "worker_thread_limit": 1,
        "scientific_payload_contains_orchestration": False,
    }:
        raise V3SampleMPIArtifactError(
            "V3 orchestration execution contract is not pinned."
        )

    source_path = _validated_evidence_record(
        root,
        payload["source_identity_file"],
        expected_role="source_identity",
    )
    source_document = _strict_json(source_path)
    if set(source_document) != {
        "schema",
        "schema_version",
        "campaign_sha256",
        "source_identity_sha256",
        "source_identity",
    }:
        raise V3SampleMPIArtifactError(
            "V3 source identity evidence schema is invalid."
        )
    try:
        source_identity = _validated_source_identity(
            source_document["source_identity"],
            require_authenticated=require_authenticated_source,
        )
    except V3SampleMPICampaignError as exc:
        raise V3SampleMPIArtifactError(
            "V3 source identity payload is invalid."
        ) from exc
    source_sha256 = _payload_sha256(source_identity)
    if (
        source_document["schema"] != _SOURCE_IDENTITY_SCHEMA
        or source_document["schema_version"] != _SCHEMA_VERSION
        or source_document["campaign_sha256"]
        != campaign_record["campaign_sha256"]
        or source_document["source_identity_sha256"] != source_sha256
        or campaign_record["source_identity_sha256"] != source_sha256
        or campaign_record["source_authenticated"]
        is not source_identity.get("authenticated")
        or source_identity["head_commit"] != identity["source_revision"]
        or source_identity["source_tree_sha256"]
        != identity["source_tree_sha256"]
    ):
        raise V3SampleMPIArtifactError(
            "V3 source identity evidence is inconsistent."
        )
    if (
        normalized_expected_source is not None
        and _canonical_json(source_identity)
        != _canonical_json(normalized_expected_source)
    ):
        raise V3SampleMPIArtifactError(
            "V3 source identity does not match the explicitly supplied "
            "current candidate identity."
        )
    authenticated = source_identity["authenticated"]

    phases = payload["phases"]
    if (
        not isinstance(phases, list)
        or len(phases) != 2
        or [item.get("phase") for item in phases]
        != [_PHASE_MAXIMUM, _PHASE_RESERVOIRS]
        or any(
            not isinstance(item, dict)
            or set(item)
            != {
                "phase",
                "directory",
                "task_manifest_sha256",
                    "selected_execution_id",
                    "selected_execution_sha256",
                    "selected_execution_physical_cores",
                    "selected_execution_origin_lineage",
                    "task_ids",
                "files",
            }
            for item in phases
        )
    ):
        raise V3SampleMPIArtifactError(
            "V3 orchestration phase records are invalid."
        )

    phase_state: dict[str, dict[str, Any]] = {}
    selected_payloads: list[dict[str, Any]] = []
    for phase in phases:
        phase_name = phase["phase"]
        inventory = _validate_phase_file_inventory(root, phase)
        task_manifest = load_task_manifest(inventory["manifest_path"])
        expected_identity_hashes = {
            "campaign_sha256": campaign_record["campaign_sha256"],
            "source_identity_sha256": source_sha256,
        }
        expected_task_ids = [
            f"v3-q0-{int(q0):08d}"
            for q0 in identity["target_grid_points"]
        ]
        metadata = mpi_tasks._normalize_json(  # noqa: SLF001
            task_manifest.metadata
        )
        if phase_name == _PHASE_RESERVOIRS:
            expected_identity_hashes["global_maximum_sha256"] = (
                _payload_sha256(metadata.get("global_maximum_block"))
            )
        if (
            phase["task_manifest_sha256"]
            != task_manifest.manifest_sha256
            or task_manifest.source_revision
            != identity["source_revision"]
            or task_manifest.source_tree_sha256
            != identity["source_tree_sha256"]
            or task_manifest.environment_sha256
            != identity["environment_sha256"]
            or task_manifest.identity_hashes.get("campaign_sha256")
            != campaign_record["campaign_sha256"]
            or task_manifest.identity_hashes.get(
                "source_identity_sha256"
            )
            != source_sha256
            or dict(task_manifest.identity_hashes)
            != expected_identity_hashes
            or phase["task_ids"] != expected_task_ids
            or [task.task_id for task in task_manifest.tasks]
            != expected_task_ids
            or any(
                dict(task.identity_hashes) != expected_identity_hashes
                for task in task_manifest.tasks
            )
            or metadata.get("campaign_schema") != _CAMPAIGN_SCHEMA
            or metadata.get("phase") != phase_name
            or metadata.get("selection_algorithm_version")
            != V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION
            or metadata.get("source_authenticated")
            is not authenticated
            or set(metadata)
            != (
                {
                    "campaign_schema",
                    "phase",
                    "selection_algorithm_version",
                    "source_authenticated",
                }
                | (
                    {"global_maximum_block"}
                    if phase_name == _PHASE_RESERVOIRS
                    else set()
                )
            )
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} task manifest conflicts with orchestration."
            )
        artifact_records = {
            path.name: _file_sha256(path)
            for path in inventory["task_artifacts"]
        }
        checkpoint_records = {
            path.name: _file_sha256(path)
            for path in inventory["task_checkpoints"]
        }
        histories: list[tuple[Path, dict[str, Any], str]] = []
        for path in inventory["executions"]:
            execution, execution_sha = _execution_payload(
                path,
                manifest=task_manifest,
                artifact_records=artifact_records,
                checkpoint_records=checkpoint_records,
            )
            if path.name != f"{execution['execution_id']}.json":
                raise V3SampleMPIArtifactError(
                    "MPI execution filename and execution ID disagree."
                )
            histories.append((path, execution, execution_sha))
        selected = [
            value
            for value in histories
            if value[1]["execution_id"]
            == phase["selected_execution_id"]
        ]
        if (
            len(selected) != 1
            or phase["selected_execution_sha256"] != selected[0][2]
            or selected[0][1]["complete"] is not True
            or selected[0][1]["missing_task_ids"] != []
            or selected[0][1]["infrastructure_errors"] != []
            or [
                assignment["task_id"]
                for assignment in selected[0][1]["assignments"]
            ]
            != expected_task_ids
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} selected execution is missing, incomplete, "
                "or mismatched."
            )
        try:
            origin_lineage = (
                mpi_tasks.task_execution_mpi_origin_lineage_evidence(
                    selected[0][1],
                    [value[1] for value in histories],
                    require_mpi_origin=require_mpi,
                )
            )
        except TaskManifestError as exc:
            raise V3SampleMPIArtifactError(
                f"{phase_name} MPI artifact-origin lineage is invalid."
            ) from exc
        origin_execution_ids = (
            {
                str(record["task_id"]): str(
                    record["origin_execution_id"]
                )
                for record in origin_lineage["lineage"]
            }
            if origin_lineage["passed"]
            else None
        )
        try:
            physical_cores = (
                mpi_tasks.task_assignment_physical_core_evidence(
                    selected[0][1]["assignments"],
                    require_available=require_mpi,
                    origin_execution_ids=origin_execution_ids,
                )
            )
        except TaskManifestError as exc:
            raise V3SampleMPIArtifactError(
                f"{phase_name} physical-core binding evidence is invalid."
            ) from exc
        if (
            mpi_tasks._normalize_json(  # noqa: SLF001
                phase["selected_execution_physical_cores"]
            )
            != mpi_tasks._normalize_json(physical_cores)  # noqa: SLF001
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} physical-core evidence was tampered."
            )
        if (
            mpi_tasks._normalize_json(  # noqa: SLF001
                phase["selected_execution_origin_lineage"]
            )
            != mpi_tasks._normalize_json(origin_lineage)  # noqa: SLF001
        ):
            raise V3SampleMPIArtifactError(
                f"{phase_name} MPI artifact-origin lineage was tampered."
            )
        if require_mpi and selected[0][1]["mode"] != "mpi":
            raise V3SampleMPIArtifactError(
                "release V3 orchestration requires actual multi-rank MPI "
                "executions in both phases."
            )
        selected_payloads.append(selected[0][1])
        phase_state[phase_name] = {
            "manifest": task_manifest,
            "inventory": inventory,
            "histories": histories,
            "physical_cores": physical_cores,
            "origin_lineage": origin_lineage,
        }

    def validate_checkpoint_origins(
        phase_name: str,
        checkpoints: Sequence[Any],
    ) -> None:
        histories = phase_state[phase_name]["histories"]
        by_task = {
            checkpoint.task_id: checkpoint for checkpoint in checkpoints
        }
        for checkpoint in checkpoints:
            execution = checkpoint.execution
            origin_rank = execution.get("rank")
            origin_size = execution.get("communicator_size")
            orphan = execution.get("orphan_artifact_adopted", False)
            if require_mpi and (
                isinstance(origin_rank, bool)
                or not isinstance(origin_rank, int)
                or origin_rank <= 0
                or isinstance(origin_size, bool)
                or not isinstance(origin_size, int)
                or origin_size < 2
                or origin_rank >= origin_size
                or orphan is not False
            ):
                raise V3SampleMPIArtifactError(
                    f"{phase_name} checkpoint lacks nonorphan worker-MPI "
                    "origin."
                )
        for _, history, _ in histories:
            for assignment in history["assignments"]:
                checkpoint = by_task[assignment["task_id"]]
                execution = checkpoint.execution
                thread_cap = execution["thread_cap"]
                if (
                    assignment["rank"] != execution["rank"]
                    or assignment["origin_communicator_size"]
                    != execution["communicator_size"]
                    or assignment["orphan_artifact_adopted"]
                    is not execution.get(
                        "orphan_artifact_adopted",
                        False,
                    )
                    or assignment["cpu_affinity"]
                    != mpi_tasks._normalize_json(  # noqa: SLF001
                        thread_cap.get("cpu_affinity")
                    )
                    or assignment["cpu_affinity_topology"]
                    != mpi_tasks._normalize_json(  # noqa: SLF001
                        thread_cap.get("cpu_affinity_topology")
                    )
                ):
                    raise V3SampleMPIArtifactError(
                        f"{phase_name} execution origin fields conflict with "
                        "the immutable task checkpoint."
                    )

    maximum_manifest = phase_state[_PHASE_MAXIMUM]["manifest"]
    maximum_root = root / "phase-maximum"
    maximum_checkpoints = collect_task_checkpoints(
        maximum_manifest,
        maximum_root / "task-outcomes",
        maximum_root / "task-artifacts",
        artifact_validator=lambda path, task: {
            "schema": _ARTIFACT_SCHEMA,
            "phase": _PHASE_MAXIMUM,
            "q0": _task_q0(task),
            "retained_candidate_slots": (
                1
                if isinstance(
                    _read_task_artifact(
                        path,
                        task,
                        expected_phase=_PHASE_MAXIMUM,
                    ),
                    _PromotedBlock,
                )
                else 0
            ),
        },
        require_complete=True,
    )
    maximum_by_id = {
        value.task_id: value for value in maximum_checkpoints
    }
    validate_checkpoint_origins(
        _PHASE_MAXIMUM,
        maximum_checkpoints,
    )
    maxima = tuple(
        _read_task_artifact(
            maximum_root
            / "task-artifacts"
            / maximum_by_id[task.task_id].artifact_relative_path,
            task,
            expected_phase=_PHASE_MAXIMUM,
        )
        for task in maximum_manifest.tasks
    )
    global_maximum = _merge_maxima(maxima)
    global_payload = _block_payload(global_maximum)
    merge = payload["merge"]
    if (
        not isinstance(merge, dict)
        or set(merge)
        != {
            "global_maximum_block",
            "global_maximum_block_sha256",
            "target_q_order",
            "floating_point_reduction_operator",
        }
        or merge["global_maximum_block"] != global_payload
        or merge["global_maximum_block_sha256"]
        != _payload_sha256(global_payload)
        or merge["target_q_order"] != identity["target_grid_points"]
        or merge["floating_point_reduction_operator"] != "none"
    ):
        raise V3SampleMPIArtifactError(
            "V3 canonical maximum merge evidence is inconsistent."
        )

    reservoir_manifest = phase_state[_PHASE_RESERVOIRS]["manifest"]
    declared_maximum = reservoir_manifest.metadata.get(
        "global_maximum_block"
    )
    if (
        mpi_tasks._normalize_json(declared_maximum)  # noqa: SLF001
        != global_payload
    ):
        raise V3SampleMPIArtifactError(
            "reservoir manifest does not bind the canonical maximum."
        )
    reservoir_root = root / "phase-reservoirs"
    floor = max(1.0e-30, 1.0e-12 * global_maximum.strength_ev2)
    maximum_blocks = int(identity["maximum_promoted_blocks"])
    target_bands = tuple(int(value) for value in identity["target_bands"])

    def reservoir_validator(
        path: Path,
        task: TaskSpec,
    ) -> Mapping[str, Any]:
        value = _read_task_artifact(
            path,
            task,
            expected_phase=_PHASE_RESERVOIRS,
            floor=floor,
            global_maximum=global_maximum.strength_ev2,
            maximum_promoted_blocks=maximum_blocks,
            target_bands=target_bands,
        )
        if not isinstance(value, _BoundedSelectionAccumulator):
            raise V3SampleMPIArtifactError(
                "reservoir task did not contain an accumulator."
            )
        return {
            "schema": _ARTIFACT_SCHEMA,
            "phase": _PHASE_RESERVOIRS,
            "q0": _task_q0(task),
            "retained_candidate_slots": value.retained_candidate_slots,
        }

    reservoir_checkpoints = collect_task_checkpoints(
        reservoir_manifest,
        reservoir_root / "task-outcomes",
        reservoir_root / "task-artifacts",
        artifact_validator=reservoir_validator,
        require_complete=True,
    )
    reservoir_by_id = {
        value.task_id: value for value in reservoir_checkpoints
    }
    validate_checkpoint_origins(
        _PHASE_RESERVOIRS,
        reservoir_checkpoints,
    )
    accumulators = tuple(
        _read_task_artifact(
            reservoir_root
            / "task-artifacts"
            / reservoir_by_id[task.task_id].artifact_relative_path,
            task,
            expected_phase=_PHASE_RESERVOIRS,
            floor=floor,
            global_maximum=global_maximum.strength_ev2,
            maximum_promoted_blocks=maximum_blocks,
            target_bands=target_bands,
        )
        for task in reservoir_manifest.tasks
    )
    merged = _merge_accumulators(
        SimpleNamespace(
            maximum_promoted_blocks=maximum_blocks,
            target_bands=target_bands,
        ),
        accumulators,
        global_maximum_block=global_maximum,
    )
    selected, required_strata, empty_strata = _select_bounded_blocks(
        merged,
        minimum_promoted_blocks=int(identity["minimum_promoted_blocks"]),
    )
    expected_coverage = {
        "cover_all_target_grid_points": False,
        "cover_all_target_bands": True,
        "required_processes": ["normal", "umklapp"],
        "required_strata": list(required_strata),
        "empty_strata": list(empty_strata),
        "minimum_promoted_blocks": int(identity["minimum_promoted_blocks"]),
    }
    if (
        raw_scientific.get("robust_nonzero_floor_ev2") != merged.floor
        or raw_scientific.get("coverage") != expected_coverage
        or raw_scientific.get("samples")
        != [value.json_record() for value in selected]
    ):
        raise V3SampleMPIArtifactError(
            "canonical task-artifact merge does not reproduce the scientific "
            "manifest."
        )

    return V3MPIOrchestrationEvidence(
        path=evidence_path,
        file_sha256=_file_sha256(evidence_path),
        payload_sha256=payload_sha256,
        campaign_sha256=campaign_record["campaign_sha256"],
        source_identity_sha256=source_sha256,
        source_authenticated=authenticated,
        selected_execution_ids=tuple(
            value["execution_id"] for value in selected_payloads
        ),  # type: ignore[arg-type]
        selected_communicator_sizes=tuple(
            int(value["communicator_size"]) for value in selected_payloads
        ),  # type: ignore[arg-type]
        selected_modes=tuple(
            str(value["mode"]) for value in selected_payloads
        ),  # type: ignore[arg-type]
        restart_resume_recorded=any(
            assignment["resumed"]
            for execution in selected_payloads
            for assignment in execution["assignments"]
        ),
        physical_core_binding_passed=all(
            state["physical_cores"]["available"] is True
            for state in phase_state.values()
        ),
        mpi_origin_lineage_passed=all(
            state["origin_lineage"]["passed"] is True
            for state in phase_state.values()
        ),
        source_revision=str(identity["source_revision"]),
        source_tree_sha256=str(identity["source_tree_sha256"]),
        environment_sha256=str(identity["environment_sha256"]),
        mesh=tuple(int(item) for item in identity["mesh"]),
        target_grid_points=tuple(
            int(item) for item in identity["target_grid_points"]
        ),
        target_bands=tuple(
            int(item) for item in identity["target_bands"]
        ),
    )


__all__ = [
    "V3MPIOrchestrationEvidence",
    "V3SampleMPIArtifactError",
    "V3SampleMPICampaign",
    "V3SampleMPICampaignError",
    "V3SampleMPIMergeError",
    "V3SampleMPIResult",
    "current_v3_sampling_source_identity",
    "run_v3_sample_manifest_mpi",
    "validate_v3_mpi_orchestration_evidence",
]
