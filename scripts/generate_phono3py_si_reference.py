#!/usr/bin/env python3
"""Generate a provenance-strict phono3py 4.4.0 silicon reference archive.

This script is intentionally independent of :mod:`negfpy`.  It creates the
external reference data against which NEGFPy's bulk three-phonon implementation
is compared.  It never downloads data, discovers inputs implicitly, changes the
working directory, or overwrites an existing artifact.

The input manifest pins both the source Git commit and the SHA256 digest of
every consumed source file.  A force-constants-only stage can be run first and
resumed later, but only when the source, code, machine, Python environment,
backend, settings, inventory, and checksums still match exactly.

The production implementation is pinned to the public phono3py 4.4.0 APIs.  Two
public module-level helpers, rather than top-level API methods, are unavoidable
and are named in the generated metadata:

* ``phono3py.file_IO.write_fc2_to_hdf5`` and
  ``phono3py.file_IO.write_fc3_to_hdf5``;
* ``phono3py.phonon3.triplets.get_triplets_integration_weights``.

No private (underscore-prefixed) phono3py API is used.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from numbers import Integral, Real
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, Sequence


MANIFEST_SCHEMA_VERSION = 1
REFERENCE_SCHEMA_VERSION = 1
FORCE_CONSTANT_FILENAMES = (
    "fc2.compact.hdf5",
    "fc2.full.hdf5",
    "fc3.compact.hdf5",
    "fc3.full.hdf5",
)
REFERENCE_FILENAME = "si-phono3py-reference.hdf5"

_FORCE_COMMON_FILENAMES = (
    "input-manifest.normalized.json",
    "source-verification.json",
    "force-constants-settings.json",
    "force-constants-environment.json",
)
_FORCE_FINAL_FILENAMES = (
    "force-constants-stage.json",
    "force-constants-inventory.json",
    "force-constants-SHA256SUMS",
)
_FORCE_STAGE_FILENAMES = frozenset(
    FORCE_CONSTANT_FILENAMES + _FORCE_COMMON_FILENAMES + _FORCE_FINAL_FILENAMES
)
_REFERENCE_FILENAMES = (
    "reference-settings.json",
    "reference-environment.json",
    "reference-stage.json",
    "inventory.json",
    "SHA256SUMS",
)

_MAX_MESH_AXIS = 128
_MAX_MESH_POINTS = 262_144
_MAX_TARGET_GRID_POINTS = 4_096
_MAX_BAND_INDEX = 16_383
_MAX_BAND_SELECTION = 4_096
_MAX_TEMPERATURES = 512
_MAX_TEMPERATURE_K = 10_000.0
_MAX_SIGMA_THZ = 100.0
_MAX_SIGMA_CUTOFF = 100.0
_PHONO3PY_CUTOFF_FREQUENCY_THZ = 1.0e-4
# Literal non-mesh points used by the independent FC2 reconstruction gate.
# Keeping the coordinates explicit avoids RNG-version drift in the reference.
FC2_RANDOM_QPOINTS = (
    (0.125, 0.25, -0.375),
    (0.3141592653589793, -0.2718281828459045, 0.1618033988749895),
    (-0.4375, 0.09375, 0.28125),
    (0.03125, -0.21875, 0.40625),
    (-0.1924500897298753, 0.3333333333333333, -0.1178511301977579),
    (0.24609375, 0.37109375, -0.46484375),
    (-0.35546875, -0.14453125, 0.21484375),
    (0.419921875, -0.337890625, -0.072265625),
)
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}.")
        result[key] = value
    return result


def _require_exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    context: str,
) -> None:
    allowed = required | (optional or set())
    missing = required - set(value)
    unknown = set(value) - allowed
    if missing:
        raise ValueError(f"{context} is missing keys: {sorted(missing)!r}.")
    if unknown:
        raise ValueError(f"{context} contains unknown keys: {sorted(unknown)!r}.")


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    return value


def _normalize_relative_path(value: Any, *, context: str) -> Path:
    raw = _require_nonempty_string(value, context=context)
    posix_path = PurePosixPath(raw)
    if posix_path.is_absolute() or ".." in posix_path.parts:
        raise ValueError(f"{context} must be a safe relative path.")
    if any(part in ("", ".") for part in posix_path.parts):
        raise ValueError(f"{context} must be a normalized relative path.")
    # Manifests are platform-independent and therefore always use '/'.
    if "\\" in raw:
        raise ValueError(f"{context} must use POSIX relative-path separators.")
    return Path(*posix_path.parts)


@dataclass(frozen=True)
class SourceFileExpectation:
    """One immutable file expectation from the source manifest."""

    role: str
    relative_path: Path
    sha256: str


@dataclass(frozen=True)
class InputManifest:
    """Detached, immutable input manifest."""

    schema_version: int
    dataset_id: str
    expected_versions: tuple[tuple[str, str], ...]
    source_repository: str
    source_commit: str
    source_files: tuple[SourceFileExpectation, ...]
    input_sha256: str

    def file(self, role: str) -> SourceFileExpectation:
        """Return the source-file expectation for ``role``."""

        for item in self.source_files:
            if item.role == role:
                return item
        raise KeyError(role)

    def normalized_dict(self) -> dict[str, Any]:
        """Return the canonical JSON representation archived with outputs."""

        return {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "versions": dict(self.expected_versions),
            "source": {
                "repository": self.source_repository,
                "commit": self.source_commit,
                "files": {
                    item.role: {
                        "path": item.relative_path.as_posix(),
                        "sha256": item.sha256,
                    }
                    for item in self.source_files
                },
            },
            "input_manifest_sha256": self.input_sha256,
        }


def load_input_manifest(path: str | os.PathLike[str]) -> InputManifest:
    """Read and strictly validate an immutable source manifest."""

    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except UnicodeDecodeError as exc:
        raise ValueError("Input manifest must be UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("Input manifest root must be a JSON object.")
    _require_exact_keys(
        payload,
        required={"schema_version", "dataset_id", "versions", "source"},
        context="input manifest",
    )
    if (
        isinstance(payload["schema_version"], bool)
        or payload["schema_version"] != MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(
            f"schema_version must equal {MANIFEST_SCHEMA_VERSION}."
        )
    dataset_id = _require_nonempty_string(
        payload["dataset_id"],
        context="dataset_id",
    )

    versions = payload["versions"]
    if not isinstance(versions, dict):
        raise ValueError("versions must be a JSON object.")
    _require_exact_keys(
        versions,
        required={"phono3py", "phonopy"},
        context="versions",
    )
    expected_versions = tuple(
        sorted(
            (
                key,
                _require_nonempty_string(
                    versions[key],
                    context=f"versions.{key}",
                ),
            )
            for key in versions
        )
    )

    source = payload["source"]
    if not isinstance(source, dict):
        raise ValueError("source must be a JSON object.")
    _require_exact_keys(
        source,
        required={"repository", "commit", "files"},
        context="source",
    )
    repository = _require_nonempty_string(
        source["repository"],
        context="source.repository",
    )
    commit = _require_nonempty_string(
        source["commit"],
        context="source.commit",
    ).lower()
    if not _COMMIT_RE.fullmatch(commit):
        raise ValueError("source.commit must be an exact 40-hex Git commit.")

    files = source["files"]
    if not isinstance(files, dict) or not files:
        raise ValueError("source.files must be a non-empty JSON object.")
    required_roles = {"phono3py_yaml", "forces_fc3"}
    missing_roles = required_roles - set(files)
    if missing_roles:
        raise ValueError(
            f"source.files is missing roles: {sorted(missing_roles)!r}."
        )

    seen_paths: set[str] = set()
    source_files: list[SourceFileExpectation] = []
    for role in sorted(files):
        if not isinstance(role, str) or not role:
            raise ValueError("Every source.files role must be a non-empty string.")
        entry = files[role]
        if not isinstance(entry, dict):
            raise ValueError(f"source.files.{role} must be a JSON object.")
        _require_exact_keys(
            entry,
            required={"path", "sha256"},
            context=f"source.files.{role}",
        )
        relative_path = _normalize_relative_path(
            entry["path"],
            context=f"source.files.{role}.path",
        )
        normalized_path = relative_path.as_posix()
        if normalized_path in seen_paths:
            raise ValueError(
                f"Source path {normalized_path!r} is assigned to multiple roles."
            )
        seen_paths.add(normalized_path)
        digest = _require_nonempty_string(
            entry["sha256"],
            context=f"source.files.{role}.sha256",
        ).lower()
        if not _SHA256_RE.fullmatch(digest):
            raise ValueError(
                f"source.files.{role}.sha256 must be a 64-hex SHA256 digest."
            )
        source_files.append(
            SourceFileExpectation(
                role=role,
                relative_path=relative_path,
                sha256=digest,
            )
        )

    return InputManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        dataset_id=dataset_id,
        expected_versions=expected_versions,
        source_repository=repository,
        source_commit=commit,
        source_files=tuple(source_files),
        input_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _canonical_int_tuple(
    values: Sequence[int],
    *,
    context: str,
    allow_empty: bool,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of integers.")
    result: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(f"{context} must contain only integers.")
        result.append(int(value))
    if not allow_empty and not result:
        raise ValueError(f"{context} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates.")
    return tuple(sorted(result))


def _canonical_float_tuple(
    values: Sequence[float],
    *,
    context: str,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{context} must be a sequence of real numbers.")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{context} must contain only real numbers.")
        converted = float(value)
        if not math.isfinite(converted):
            raise ValueError(f"{context} must contain only finite values.")
        result.append(converted)
    if not result:
        raise ValueError(f"{context} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{context} must not contain duplicates.")
    return tuple(sorted(result))


def _finite_positive(value: Any, *, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{context} must be a real number.")
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        raise ValueError(f"{context} must be finite and positive.")
    return converted


@dataclass(frozen=True)
class GeneratorConfig:
    """Bounded scientific settings for one reference-generation invocation."""

    mesh: tuple[int, int, int] | Sequence[int] | None = None
    grid_points: tuple[int, ...] | Sequence[int] = (0,)
    band_indices: tuple[int, ...] | Sequence[int] | None = None
    temperatures: tuple[float, ...] | Sequence[float] = (300.0,)
    sigma: float = 0.1
    sigma_cutoff: float | None = None
    backend: str = "Rust"
    make_r0_average: bool = True
    symmetrize_fc: bool = True
    symmetrize_fc3q: bool = False
    force_constants_only: bool = False

    def __post_init__(self) -> None:
        for name in (
            "make_r0_average",
            "symmetrize_fc",
            "symmetrize_fc3q",
            "force_constants_only",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be bool.")
        if self.backend not in ("C", "Rust"):
            raise ValueError("backend must be exactly 'C' or 'Rust'.")

        mesh: tuple[int, int, int] | None
        if self.mesh is None:
            mesh = None
        else:
            if isinstance(self.mesh, (str, bytes)):
                raise TypeError("mesh must be a sequence of integers.")
            if len(self.mesh) != 3:
                raise ValueError("mesh must contain exactly three integers.")
            # A mesh is an axis tuple, so its input ordering must be preserved.
            raw_values: list[int] = []
            for value in self.mesh:
                if isinstance(value, bool) or not isinstance(value, Integral):
                    raise TypeError("mesh must contain only integers.")
                raw_values.append(int(value))
            mesh = tuple(raw_values)  # type: ignore[assignment]
            if any(value <= 0 for value in mesh):
                raise ValueError("mesh entries must be positive.")
            if any(value > _MAX_MESH_AXIS for value in mesh):
                raise ValueError(
                    f"mesh entries may not exceed {_MAX_MESH_AXIS}."
                )
            if math.prod(mesh) > _MAX_MESH_POINTS:
                raise ValueError(
                    f"mesh may not exceed {_MAX_MESH_POINTS} total points."
                )
        if not self.force_constants_only and mesh is None:
            raise ValueError("mesh is required unless force_constants_only=True.")

        grid_points = _canonical_int_tuple(
            self.grid_points,
            context="grid_points",
            allow_empty=False,
        )
        if len(grid_points) > _MAX_TARGET_GRID_POINTS:
            raise ValueError(
                "grid_points exceeds the bounded target-q selection size."
            )
        if any(value < 0 for value in grid_points):
            raise ValueError("grid_points must be nonnegative.")
        if mesh is not None and any(value >= math.prod(mesh) for value in grid_points):
            raise ValueError("grid_points contains an index outside the mesh.")

        band_indices: tuple[int, ...] | None
        if self.band_indices is None:
            band_indices = None
        else:
            band_indices = _canonical_int_tuple(
                self.band_indices,
                context="band_indices",
                allow_empty=False,
            )
            if len(band_indices) > _MAX_BAND_SELECTION:
                raise ValueError("band_indices selection is too large.")
            if any(value < 0 for value in band_indices):
                raise ValueError("band_indices must be nonnegative.")
            if any(value > _MAX_BAND_INDEX for value in band_indices):
                raise ValueError(
                    f"band_indices may not exceed {_MAX_BAND_INDEX}."
                )

        temperatures = _canonical_float_tuple(
            self.temperatures,
            context="temperatures",
        )
        if len(temperatures) > _MAX_TEMPERATURES:
            raise ValueError("temperatures selection is too large.")
        if any(value < 0 for value in temperatures):
            raise ValueError("temperatures must be nonnegative.")
        if any(value > _MAX_TEMPERATURE_K for value in temperatures):
            raise ValueError(
                f"temperatures may not exceed {_MAX_TEMPERATURE_K:g} K."
            )

        sigma = _finite_positive(self.sigma, context="sigma")
        if sigma > _MAX_SIGMA_THZ:
            raise ValueError(f"sigma may not exceed {_MAX_SIGMA_THZ:g} THz.")
        sigma_cutoff = (
            None
            if self.sigma_cutoff is None
            else _finite_positive(self.sigma_cutoff, context="sigma_cutoff")
        )
        if sigma_cutoff is not None and sigma_cutoff > _MAX_SIGMA_CUTOFF:
            raise ValueError(
                f"sigma_cutoff may not exceed {_MAX_SIGMA_CUTOFF:g}."
            )

        object.__setattr__(self, "mesh", mesh)
        object.__setattr__(self, "grid_points", grid_points)
        object.__setattr__(self, "band_indices", band_indices)
        object.__setattr__(self, "temperatures", temperatures)
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "sigma_cutoff", sigma_cutoff)

    def force_constants_settings(self) -> dict[str, Any]:
        """Settings whose identity must survive a force-stage resume."""

        return {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "backend": self.backend,
            "is_nac": False,
            "is_crystal_symmetry": True,
            "is_mesh_symmetry": False,
            "symprec": 1.0e-5,
            "use_grg": False,
            "make_r0_average": self.make_r0_average,
            "symmetrize_fc": self.symmetrize_fc,
            "generated_layouts": ["compact", "full"],
            "fc2_units": "eV/angstrom^2",
            "fc3_units": "eV/angstrom^3",
            "overwrite_policy": "exclusive-create-only",
        }

    def reference_settings(self) -> dict[str, Any]:
        """Canonical settings for the phonon and three-phonon reference."""

        if self.mesh is None:  # Guard for static type checkers and direct callers.
            raise ValueError("mesh is required for reference generation.")
        return {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "mesh": list(self.mesh),
            "grid_point_index_space": "GRG_x_fast",
            "grid_points": list(self.grid_points),
            "band_indices": (
                None
                if self.band_indices is None
                else list(self.band_indices)
            ),
            "temperatures_K": list(self.temperatures),
            "sigma_THz": self.sigma,
            "sigma_cutoff": self.sigma_cutoff,
            "backend": self.backend,
            "is_nac": False,
            "is_crystal_symmetry": True,
            "is_mesh_symmetry": False,
            "symprec": 1.0e-5,
            "use_grg": False,
            "make_r0_average": self.make_r0_average,
            "symmetrize_fc": self.symmetrize_fc,
            "symmetrize_fc3q": self.symmetrize_fc3q,
            "lapack_zheev_uplo": "L",
            "openmp_per_triplets": False,
            "cutoff_frequency_THz": _PHONO3PY_CUTOFF_FREQUENCY_THZ,
            "constant_averaged_interaction": None,
            "frequency_scale_factor": None,
            "frequency_units": "cyclic_THz",
            "gamma_units": "cyclic_THz_HWHM",
            "interaction_strength_units": "eV^2",
            "integration_weight_units": "1/cyclic_THz",
            "triplet_normalization": "explicit_weights_divided_by_Nq_only_in_derived_phase_space",
            "overwrite_policy": "exclusive-create-only",
        }


@dataclass(frozen=True)
class GenerationResult:
    """Summary of a completed force-constant or reference stage."""

    stage: str
    output_dir: Path
    inventory_path: Path
    sha256sums_path: Path


class _Runtime(Protocol):
    phono3py_version: str
    phonopy_version: str

    def verify_versions(
        self,
        expected: Mapping[str, str],
    ) -> Mapping[str, str]:
        ...

    def generate_force_constants(
        self,
        *,
        output_dir: Path,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Mapping[str, Any]:
        ...

    def generate_reference(
        self,
        *,
        output_dir: Path,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Mapping[str, Any]:
        ...


def _git_commit(source_dir: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "--verify", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(
            "The source directory must be a Git checkout at the manifest commit."
        ) from exc
    return completed.stdout.strip()


def _verify_source(
    manifest: InputManifest,
    source_dir: Path,
    *,
    commit_resolver: Callable[[Path], str] | None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    root = source_dir.resolve(strict=True)
    if not root.is_dir():
        raise ValueError("source_dir must resolve to a directory.")
    resolved_commit = (commit_resolver or _git_commit)(root).strip().lower()
    if not _COMMIT_RE.fullmatch(resolved_commit):
        raise ValueError("Resolved source commit is not an exact 40-hex commit.")
    if resolved_commit != manifest.source_commit:
        raise ValueError(
            "Source commit mismatch: "
            f"expected {manifest.source_commit}, got {resolved_commit}."
        )

    source_files: dict[str, Path] = {}
    file_records: dict[str, Any] = {}
    for expectation in manifest.source_files:
        candidate = root.joinpath(expectation.relative_path)
        if candidate.is_symlink():
            raise ValueError(
                f"Source input may not be a symbolic link: "
                f"{expectation.relative_path}."
            )
        try:
            resolved = candidate.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Required source file is missing: {expectation.relative_path}."
            ) from exc
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Source file escapes source_dir: {expectation.relative_path}."
            ) from exc
        if not resolved.is_file():
            raise ValueError(
                f"Source input is not a regular file: {expectation.relative_path}."
            )
        actual_digest = _sha256_file(resolved)
        if actual_digest != expectation.sha256:
            raise ValueError(
                f"SHA256 mismatch for {expectation.relative_path}: "
                f"expected {expectation.sha256}, got {actual_digest}."
            )
        source_files[expectation.role] = resolved
        file_records[expectation.role] = {
            "path_in_source": expectation.relative_path.as_posix(),
            "resolved_path": str(resolved),
            "sha256": actual_digest,
            "size_bytes": resolved.stat().st_size,
        }

    verification = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "dataset_id": manifest.dataset_id,
        "repository": manifest.source_repository,
        "commit": resolved_commit,
        "source_root": str(root),
        "files": file_records,
    }
    return source_files, verification


def _run_git(
    arguments: Sequence[str],
    *,
    cwd: Path,
) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=cwd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def _script_repository_identity() -> dict[str, Any]:
    script_path = Path(__file__).resolve()
    root_text = _run_git(
        ["rev-parse", "--show-toplevel"],
        cwd=script_path.parent,
    )
    identity: dict[str, Any] = {
        "script_path": str(script_path),
        "script_sha256": _sha256_file(script_path),
    }
    if root_text is None:
        identity.update(
            {
                "git_root": None,
                "git_commit": None,
                "tracked_worktree_status": None,
            }
        )
        return identity
    root = Path(root_text).resolve()
    identity.update(
        {
            "git_root": str(root),
            "git_commit": _run_git(["rev-parse", "HEAD"], cwd=root),
            # Untracked artifacts are intentionally omitted so an explicit
            # output directory inside the repository cannot alter resume
            # identity. The generator script itself is independently hashed.
            "tracked_worktree_status": _run_git(
                ["status", "--porcelain=v1", "--untracked-files=no"],
                cwd=root,
            ),
            "untracked_files_omitted": True,
        }
    )
    return identity


def _installed_packages() -> list[dict[str, str]]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return [
        {"name": name, "version": packages[name]}
        for name in sorted(packages)
    ]


def _physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size) * int(page_count)


def _environment_identity(
    *,
    runtime_versions: Mapping[str, str],
    backend: str,
) -> dict[str, Any]:
    uname = platform.uname()
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "machine": uname.machine,
            "processor": uname.processor,
            "hostname": socket.gethostname(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _physical_memory_bytes(),
        },
        "runtime_versions": dict(sorted(runtime_versions.items())),
        "backend": backend,
        "thread_environment": {
            name: os.environ.get(name) for name in _THREAD_ENVIRONMENT_NAMES
        },
        "installed_packages": _installed_packages(),
        "code": _script_repository_identity(),
    }


def _environment_record(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "recorded_at_utc": _utc_now(),
        "identity": identity,
    }


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _publish_bytes_exclusive(path: Path, payload: bytes) -> None:
    """Atomically publish bytes without ever replacing ``path``."""

    path.parent.mkdir(parents=False, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_json_exclusive(path: Path, value: Any) -> None:
    _publish_bytes_exclusive(path, _json_bytes(value))


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )


def _media_type(path: Path) -> str:
    if path.suffix == ".json":
        return "application/json"
    if path.suffix == ".hdf5":
        return "application/x-hdf5"
    if path.name.endswith("SHA256SUMS"):
        return "text/plain"
    return "application/octet-stream"


def _inventory(
    output_dir: Path,
    *,
    exclude: set[str],
) -> dict[str, Any]:
    records = []
    for path in sorted(output_dir.iterdir(), key=lambda item: item.name):
        if path.name in exclude:
            continue
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                f"Output contains unsupported artifact: {path.name!r}."
            )
        records.append(
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
                "media_type": _media_type(path),
            }
        )
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "artifacts": records,
    }


def _write_sha256sums_exclusive(
    path: Path,
    *,
    root: Path,
    names: Sequence[str],
) -> None:
    lines: list[str] = []
    for name in sorted(names):
        relative = _normalize_relative_path(name, context="checksum filename")
        target = root / relative
        if not target.is_file() or target.is_symlink():
            raise ValueError(f"Cannot checksum non-regular artifact: {name!r}.")
        lines.append(f"{_sha256_file(target)}  {relative.as_posix()}\n")
    _publish_bytes_exclusive(path, "".join(lines).encode("utf-8"))


def verify_sha256sums(
    path: str | os.PathLike[str],
    *,
    root: str | os.PathLike[str],
) -> bool:
    """Verify a strict SHA256SUMS file rooted at ``root``."""

    checksum_path = Path(path)
    root_path = Path(root).resolve(strict=True)
    seen: set[str] = set()
    try:
        lines = checksum_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return False
    if not lines:
        return False
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            return False
        digest = line[:64].lower()
        if not _SHA256_RE.fullmatch(digest):
            return False
        name = line[66:]
        try:
            relative = _normalize_relative_path(
                name,
                context="checksum filename",
            )
        except ValueError:
            return False
        normalized = relative.as_posix()
        if normalized in seen:
            return False
        seen.add(normalized)
        try:
            unresolved_target = root_path / relative
            if unresolved_target.is_symlink():
                return False
            target = unresolved_target.resolve(strict=True)
            target.relative_to(root_path)
        except (FileNotFoundError, ValueError, OSError):
            return False
        if not target.is_file() or target.is_symlink():
            return False
        if _sha256_file(target) != digest:
            return False
    return True


def _output_state(output_dir: Path) -> str:
    if not output_dir.exists():
        return "fresh"
    if not output_dir.is_dir():
        raise FileExistsError("output_dir exists and is not a directory.")
    names = frozenset(path.name for path in output_dir.iterdir())
    if not names:
        return "fresh"
    if REFERENCE_FILENAME in names or "reference-stage.json" in names:
        raise FileExistsError(
            "A reference-stage artifact already exists; overwrite is forbidden."
        )
    if names == _FORCE_STAGE_FILENAMES:
        return "force_constants"
    raise FileExistsError(
        "Refusing non-empty output directory with an unknown or incomplete "
        "artifact schema."
    )


def _assert_equal_json(path: Path, expected: Any, *, context: str) -> None:
    actual = _read_json(path)
    if actual != expected:
        raise ValueError(f"{context} does not match the verified invocation.")


def _validate_force_stage(
    output_dir: Path,
    *,
    normalized_manifest: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    force_settings: Mapping[str, Any],
    environment_identity: Mapping[str, Any],
) -> None:
    names = frozenset(path.name for path in output_dir.iterdir())
    if names != _FORCE_STAGE_FILENAMES:
        raise ValueError("Force-constants stage has an unexpected file schema.")
    _assert_equal_json(
        output_dir / "input-manifest.normalized.json",
        normalized_manifest,
        context="Normalized manifest",
    )
    _assert_equal_json(
        output_dir / "source-verification.json",
        source_verification,
        context="Source verification",
    )
    _assert_equal_json(
        output_dir / "force-constants-settings.json",
        force_settings,
        context="Force-constants settings",
    )
    environment = _read_json(output_dir / "force-constants-environment.json")
    if not isinstance(environment, dict) or environment.get("identity") != dict(
        environment_identity
    ):
        raise ValueError(
            "Force-constants environment identity does not match this invocation."
        )
    if not verify_sha256sums(
        output_dir / "force-constants-SHA256SUMS",
        root=output_dir,
    ):
        raise ValueError("Force-constants SHA256 verification failed.")
    inventory = _read_json(output_dir / "force-constants-inventory.json")
    expected_inventory_names = {
        record["path"] for record in inventory.get("artifacts", [])
    }
    actual_inventory_names = (
        set(_FORCE_STAGE_FILENAMES)
        - {"force-constants-inventory.json", "force-constants-SHA256SUMS"}
    )
    if expected_inventory_names != actual_inventory_names:
        raise ValueError("Force-constants inventory is incomplete or ambiguous.")
    for record in inventory["artifacts"]:
        target = output_dir / record["path"]
        if (
            target.stat().st_size != record["size_bytes"]
            or _sha256_file(target) != record["sha256"]
        ):
            raise ValueError("Force-constants inventory verification failed.")


def _finish_force_stage(
    output_dir: Path,
    *,
    runtime_metadata: Mapping[str, Any],
) -> None:
    expected_before_receipt = set(FORCE_CONSTANT_FILENAMES) | set(
        _FORCE_COMMON_FILENAMES
    )
    actual_before_receipt = {path.name for path in output_dir.iterdir()}
    if actual_before_receipt != expected_before_receipt:
        raise RuntimeError(
            "Force-constant runtime created an unexpected output schema: "
            f"{sorted(actual_before_receipt - expected_before_receipt)!r}."
        )
    for name in FORCE_CONSTANT_FILENAMES:
        target = output_dir / name
        if not target.is_file() or target.is_symlink() or target.stat().st_size == 0:
            raise RuntimeError(f"Runtime did not create valid artifact {name!r}.")
    stage_record = {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "stage": "force_constants",
        "completed_at_utc": _utc_now(),
        "status": "complete",
        "runtime": dict(runtime_metadata),
    }
    _write_json_exclusive(
        output_dir / "force-constants-stage.json",
        stage_record,
    )
    _write_json_exclusive(
        output_dir / "force-constants-inventory.json",
        _inventory(
            output_dir,
            exclude={
                "force-constants-inventory.json",
                "force-constants-SHA256SUMS",
            },
        ),
    )
    checksum_names = sorted(
        set(_FORCE_STAGE_FILENAMES) - {"force-constants-SHA256SUMS"}
    )
    _write_sha256sums_exclusive(
        output_dir / "force-constants-SHA256SUMS",
        root=output_dir,
        names=checksum_names,
    )


def _finish_reference_stage(
    output_dir: Path,
    *,
    runtime_metadata: Mapping[str, Any],
) -> None:
    expected_before_receipt = set(_FORCE_STAGE_FILENAMES) | {
        "reference-settings.json",
        "reference-environment.json",
        REFERENCE_FILENAME,
    }
    actual_before_receipt = {path.name for path in output_dir.iterdir()}
    if actual_before_receipt != expected_before_receipt:
        raise RuntimeError(
            "Reference runtime created an unexpected output schema: "
            f"{sorted(actual_before_receipt - expected_before_receipt)!r}."
        )
    target = output_dir / REFERENCE_FILENAME
    if not target.is_file() or target.is_symlink() or target.stat().st_size == 0:
        raise RuntimeError(f"Runtime did not create valid artifact {target.name!r}.")
    _write_json_exclusive(
        output_dir / "reference-stage.json",
        {
            "schema_version": REFERENCE_SCHEMA_VERSION,
            "stage": "reference",
            "completed_at_utc": _utc_now(),
            "status": "complete",
            "runtime": dict(runtime_metadata),
        },
    )
    _write_json_exclusive(
        output_dir / "inventory.json",
        _inventory(
            output_dir,
            exclude={"inventory.json", "SHA256SUMS"},
        ),
    )
    checksum_names = sorted(
        path.name
        for path in output_dir.iterdir()
        if path.name != "SHA256SUMS"
    )
    _write_sha256sums_exclusive(
        output_dir / "SHA256SUMS",
        root=output_dir,
        names=checksum_names,
    )


def run_generator(
    *,
    manifest_path: str | os.PathLike[str],
    source_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config: GeneratorConfig,
    runtime: _Runtime | None = None,
    commit_resolver: Callable[[Path], str] | None = None,
) -> GenerationResult:
    """Run the force-constant stage and, when requested, reference stage."""

    if not isinstance(config, GeneratorConfig):
        raise TypeError("config must be a GeneratorConfig instance.")
    manifest = load_input_manifest(manifest_path)
    source_files, source_verification = _verify_source(
        manifest,
        Path(source_dir),
        commit_resolver=commit_resolver,
    )
    active_runtime: _Runtime = runtime or _Phono3pyRuntime()
    runtime_versions = dict(
        active_runtime.verify_versions(dict(manifest.expected_versions))
    )
    if runtime_versions != dict(manifest.expected_versions):
        raise RuntimeError(
            "Runtime version verifier returned values other than the exact "
            "manifest versions."
        )
    environment_identity = _environment_identity(
        runtime_versions=runtime_versions,
        backend=config.backend,
    )
    normalized_manifest = manifest.normalized_dict()
    force_settings = config.force_constants_settings()

    destination = Path(output_dir).expanduser()
    if not destination.is_absolute():
        destination = (Path.cwd() / destination).resolve()
    state = _output_state(destination)
    if state == "fresh":
        destination.mkdir(parents=True, exist_ok=True)
        _write_json_exclusive(
            destination / "input-manifest.normalized.json",
            normalized_manifest,
        )
        _write_json_exclusive(
            destination / "source-verification.json",
            source_verification,
        )
        _write_json_exclusive(
            destination / "force-constants-settings.json",
            force_settings,
        )
        _write_json_exclusive(
            destination / "force-constants-environment.json",
            _environment_record(environment_identity),
        )
        force_metadata = active_runtime.generate_force_constants(
            output_dir=destination,
            source_files=source_files,
            config=config,
        )
        _finish_force_stage(
            destination,
            runtime_metadata=force_metadata,
        )
    else:
        if config.force_constants_only:
            raise FileExistsError(
                "A complete force-constants stage already exists; overwrite "
                "is forbidden."
            )
        _validate_force_stage(
            destination,
            normalized_manifest=normalized_manifest,
            source_verification=source_verification,
            force_settings=force_settings,
            environment_identity=environment_identity,
        )

    if config.force_constants_only:
        return GenerationResult(
            stage="force_constants",
            output_dir=destination,
            inventory_path=destination / "force-constants-inventory.json",
            sha256sums_path=destination / "force-constants-SHA256SUMS",
        )

    _write_json_exclusive(
        destination / "reference-settings.json",
        config.reference_settings(),
    )
    _write_json_exclusive(
        destination / "reference-environment.json",
        _environment_record(environment_identity),
    )
    reference_metadata = active_runtime.generate_reference(
        output_dir=destination,
        source_files=source_files,
        config=config,
    )
    _finish_reference_stage(
        destination,
        runtime_metadata=reference_metadata,
    )
    return GenerationResult(
        stage="reference",
        output_dir=destination,
        inventory_path=destination / "inventory.json",
        sha256sums_path=destination / "SHA256SUMS",
    )


def _safe_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else [int(item) for item in shape]


def _safe_dtype(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return None if dtype is None else str(dtype)


class _Phono3pyRuntime:
    """Pinned phono3py 4.4.0 implementation loaded only on demand."""

    def __init__(self) -> None:
        try:
            import h5py
            import numpy as np
            import phono3py
            import phonopy
            from phono3py.file_IO import (
                write_fc2_to_hdf5,
                write_fc3_to_hdf5,
            )
            from phono3py.phonon3.triplets import (
                get_triplets_integration_weights,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Reference generation requires h5py, NumPy, phono3py==4.4.0, "
                "and phonopy==4.4.0."
            ) from exc
        self.h5py = h5py
        self.np = np
        self.phono3py = phono3py
        self.phonopy = phonopy
        self.write_fc2_to_hdf5 = write_fc2_to_hdf5
        self.write_fc3_to_hdf5 = write_fc3_to_hdf5
        self.get_triplets_integration_weights = (
            get_triplets_integration_weights
        )
        self.phono3py_version = str(phono3py.__version__)
        self.phonopy_version = str(phonopy.__version__)

    def verify_versions(
        self,
        expected: Mapping[str, str],
    ) -> Mapping[str, str]:
        actual = {
            "phono3py": self.phono3py_version,
            "phonopy": self.phonopy_version,
        }
        if actual != dict(expected):
            raise RuntimeError(
                f"Version mismatch: expected {dict(expected)}, got {actual}."
            )
        return actual

    def _publish_hdf5(
        self,
        target: Path,
        writer: Callable[[Path], None],
    ) -> None:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp.hdf5",
            dir=target.parent,
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        # phono3py's writers open with mode "w"; remove the reserved empty file
        # first. The random name remains unique within this invocation.
        temporary_path.unlink()
        try:
            writer(temporary_path)
            if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
                raise RuntimeError(
                    f"HDF5 writer produced no data for {target.name!r}."
                )
            file_descriptor = os.open(temporary_path, os.O_RDONLY)
            try:
                os.fsync(file_descriptor)
            finally:
                os.close(file_descriptor)
            os.link(temporary_path, target)
            directory_descriptor = os.open(target.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            temporary_path.unlink(missing_ok=True)

    def _load_from_forces(
        self,
        *,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "phono3py_yaml": source_files["phono3py_yaml"],
            "forces_fc3_filename": source_files["forces_fc3"],
            "produce_fc": True,
            "symmetrize_fc": config.symmetrize_fc,
            "is_mesh_symmetry": False,
            "is_compact_fc": True,
            "make_r0_average": config.make_r0_average,
            "is_nac": False,
            "is_symmetry": True,
            "symprec": 1.0e-5,
            "lang": config.backend,
            "log_level": 0,
        }
        if "forces_fc2" in source_files:
            kwargs["forces_fc2_filename"] = source_files["forces_fc2"]
        return self.phono3py.load(**kwargs)

    def generate_force_constants(
        self,
        *,
        output_dir: Path,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Mapping[str, Any]:
        ph3 = self._load_from_forces(
            source_files=source_files,
            config=config,
        )
        if ph3.fc2 is None or ph3.fc3 is None:
            raise RuntimeError("phono3py did not produce both FC2 and FC3.")

        compact_fc2 = self.np.array(ph3.fc2, dtype="double", copy=True, order="C")
        compact_fc3 = self.np.array(ph3.fc3, dtype="double", copy=True, order="C")
        compact_nonzero = (
            None
            if ph3.fc3_nonzero_indices is None
            else self.np.array(
                ph3.fc3_nonzero_indices,
                dtype="byte",
                copy=True,
                order="C",
            )
        )
        fc3_cutoff = (
            None if ph3.fc3_cutoff is None else float(ph3.fc3_cutoff)
        )
        fc2_p2s = self.np.array(
            ph3.phonon_primitive.p2s_map,
            dtype="int64",
            copy=True,
        )
        fc3_p2s = self.np.array(
            ph3.primitive.p2s_map,
            dtype="int64",
            copy=True,
        )

        self._publish_hdf5(
            output_dir / "fc2.compact.hdf5",
            lambda path: self.write_fc2_to_hdf5(
                compact_fc2,
                filename=str(path),
                p2s_map=fc2_p2s,
                physical_unit="eV/angstrom^2",
            ),
        )
        self._publish_hdf5(
            output_dir / "fc3.compact.hdf5",
            lambda path: self.write_fc3_to_hdf5(
                compact_fc3,
                fc3_nonzero_indices=compact_nonzero,
                filename=str(path),
                p2s_map=fc3_p2s,
                fc3_cutoff=fc3_cutoff,
            ),
        )

        ph3.produce_fc2(is_compact_fc=False)
        if config.symmetrize_fc:
            ph3.symmetrize_fc2()
        if ph3.fc2 is None:
            raise RuntimeError("phono3py failed to regenerate full FC2.")
        full_fc2 = self.np.array(ph3.fc2, dtype="double", copy=False, order="C")
        self._publish_hdf5(
            output_dir / "fc2.full.hdf5",
            lambda path: self.write_fc2_to_hdf5(
                full_fc2,
                filename=str(path),
                p2s_map=None,
                physical_unit="eV/angstrom^2",
            ),
        )

        ph3.produce_fc3(is_compact_fc=False)
        if config.symmetrize_fc:
            ph3.symmetrize_fc3()
        if ph3.fc3 is None:
            raise RuntimeError("phono3py failed to regenerate full FC3.")
        full_fc3 = self.np.array(ph3.fc3, dtype="double", copy=False, order="C")
        full_nonzero = (
            None
            if ph3.fc3_nonzero_indices is None
            else self.np.array(
                ph3.fc3_nonzero_indices,
                dtype="byte",
                copy=False,
                order="C",
            )
        )
        self._publish_hdf5(
            output_dir / "fc3.full.hdf5",
            lambda path: self.write_fc3_to_hdf5(
                full_fc3,
                fc3_nonzero_indices=full_nonzero,
                filename=str(path),
                p2s_map=None,
                fc3_cutoff=fc3_cutoff,
            ),
        )

        return {
            "implementation": "phono3py_4_4_0",
            "public_apis": [
                "phono3py.load",
                "phono3py.api_phono3py.Phono3py.produce_fc2",
                "phono3py.api_phono3py.Phono3py.produce_fc3",
                *(
                    [
                        "phono3py.api_phono3py.Phono3py.symmetrize_fc2",
                        "phono3py.api_phono3py.Phono3py.symmetrize_fc3",
                    ]
                    if config.symmetrize_fc
                    else []
                ),
                "phono3py.file_IO.write_fc2_to_hdf5",
                "phono3py.file_IO.write_fc3_to_hdf5",
            ],
            "private_apis": [],
            "phono3py_version": self.phono3py_version,
            "phonopy_version": self.phonopy_version,
            "compact_fc2_shape": _safe_shape(compact_fc2),
            "compact_fc2_dtype": _safe_dtype(compact_fc2),
            "full_fc2_shape": _safe_shape(full_fc2),
            "full_fc2_dtype": _safe_dtype(full_fc2),
            "compact_fc3_shape": _safe_shape(compact_fc3),
            "compact_fc3_dtype": _safe_dtype(compact_fc3),
            "full_fc3_shape": _safe_shape(full_fc3),
            "full_fc3_dtype": _safe_dtype(full_fc3),
            "compact_fc3_nonzero_indices_shape": _safe_shape(compact_nonzero),
            "full_fc3_nonzero_indices_shape": _safe_shape(full_nonzero),
            "fc3_cutoff": fc3_cutoff,
            "fc2_units": "eV/angstrom^2",
            "fc3_units": "eV/angstrom^3",
            "symmetrize_fc": config.symmetrize_fc,
            "make_r0_average": config.make_r0_average,
            "is_nac": False,
            "is_crystal_symmetry": True,
            "is_mesh_symmetry": False,
            "symprec": 1.0e-5,
            "use_grg": False,
            "backend": config.backend,
            "calculator": str(ph3.calculator),
            "frequency_unit_conversion_factor": float(
                ph3.unit_conversion_factor
            ),
        }

    def _load_from_force_constants(
        self,
        *,
        output_dir: Path,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Any:
        return self.phono3py.load(
            phono3py_yaml=source_files["phono3py_yaml"],
            fc2_filename=output_dir / "fc2.compact.hdf5",
            fc3_filename=output_dir / "fc3.compact.hdf5",
            produce_fc=False,
            symmetrize_fc=False,
            is_mesh_symmetry=False,
            is_compact_fc=True,
            make_r0_average=config.make_r0_average,
            is_nac=False,
            is_symmetry=True,
            symprec=1.0e-5,
            lang=config.backend,
            log_level=0,
        )

    def _create_dataset(
        self,
        group: Any,
        name: str,
        data: Any,
        *,
        units: str | None = None,
        description: str | None = None,
    ) -> Any:
        array = self.np.asarray(data)
        kwargs: dict[str, Any] = {}
        if array.ndim > 0 and array.size >= 32:
            kwargs.update(compression="gzip", shuffle=True)
        dataset = group.create_dataset(name, data=array, **kwargs)
        if units is not None:
            dataset.attrs["units"] = units
        if description is not None:
            dataset.attrs["description"] = description
        return dataset

    def _write_atoms(self, group: Any, atoms: Any) -> None:
        self._create_dataset(group, "cell", atoms.cell, units="angstrom")
        self._create_dataset(
            group,
            "scaled_positions",
            atoms.scaled_positions,
            units="fractional",
        )
        self._create_dataset(group, "masses", atoms.masses, units="amu")
        self._create_dataset(group, "atomic_numbers", atoms.numbers)

    def _write_crystal_mapping(self, root: Any, ph3: Any) -> None:
        crystal = root.create_group("crystal")
        self._write_atoms(crystal.create_group("unitcell"), ph3.unitcell)
        self._write_atoms(crystal.create_group("primitive"), ph3.primitive)
        self._write_atoms(crystal.create_group("supercell"), ph3.supercell)
        mapping = crystal.create_group("mapping")
        for name in ("p2s_map", "s2p_map"):
            self._create_dataset(mapping, name, getattr(ph3.primitive, name))
        raw_p2p = ph3.primitive.p2p_map
        if not isinstance(raw_p2p, Mapping):
            raise RuntimeError(
                "phono3py primitive.p2p_map is not the pinned sparse mapping."
            )
        p2p_keys = self.np.array(
            sorted(int(key) for key in raw_p2p),
            dtype="int64",
        )
        p2p_values = self.np.array(
            [int(raw_p2p[int(key)]) for key in p2p_keys],
            dtype="int64",
        )
        p2p_dense = self.np.full(
            len(ph3.supercell),
            -1,
            dtype="int64",
        )
        p2p_dense[p2p_keys] = p2p_values
        self._create_dataset(mapping, "p2p_map_keys", p2p_keys)
        self._create_dataset(mapping, "p2p_map_values", p2p_values)
        self._create_dataset(mapping, "p2p_map_dense", p2p_dense)
        shortest_vectors, multiplicities = ph3.primitive.get_smallest_vectors()
        self._create_dataset(
            mapping,
            "shortest_vectors",
            shortest_vectors,
            units="fractional_supercell",
        )
        self._create_dataset(mapping, "multiplicities", multiplicities)
        self._create_dataset(
            mapping,
            "primitive_matrix",
            ph3.primitive_matrix,
        )
        self._create_dataset(
            mapping,
            "supercell_matrix",
            ph3.supercell_matrix,
        )
        self._create_dataset(
            mapping,
            "translation_permutations",
            ph3.primitive.atomic_permutations,
        )
        self._create_dataset(
            mapping,
            "crystal_symmetry_atomic_permutations",
            ph3.symmetry.atomic_permutations,
        )

    def _write_grid(self, root: Any, bz_grid: Any) -> None:
        grid = root.create_group("grid")
        for name in (
            "D_diag",
            "P",
            "Q",
            "PS",
            "QDinv",
            "addresses",
            "grg2bzg",
            "bzg2grg",
            "gp_map",
            "gp_Gamma",
            "grid_matrix",
            "microzone_lattice",
            "reciprocal_lattice",
            "rotations",
            "reciprocal_operations",
        ):
            value = getattr(bz_grid, name)
            if value is not None:
                self._create_dataset(grid, name, value)
        grid.attrs["GRG_index_order"] = "x_fast"
        grid.attrs["target_grid_point_input_space"] = "GRG"
        grid.attrs["triplet_grid_point_space"] = "BZG"

    def _write_reference_hdf5(
        self,
        path: Path,
        *,
        ph3: Any,
        config: GeneratorConfig,
        frequencies: Any,
        eigenvectors: Any,
        bz_addresses: Any,
        effective_bands: Any,
        grg_targets: Any,
        bzg_targets: Any,
        gamma_values: Any,
        fc2_random_qpoints: Any,
        fc2_random_dynamical_matrices: Any,
    ) -> dict[str, Any]:
        np = self.np
        interaction = ph3.phph_interaction
        if interaction is None:
            raise RuntimeError("phono3py interaction was not initialized.")
        bz_grid = interaction.bz_grid
        n_q = int(np.prod(bz_grid.D_diag))
        gamma = np.asarray(gamma_values.gammas, dtype="double")
        expected_gamma_shape = (
            1,
            len(config.temperatures),
            len(bzg_targets),
            len(effective_bands),
        )
        if gamma.shape != expected_gamma_shape:
            raise RuntimeError(
                "Unexpected phono3py gamma shape: "
                f"expected {expected_gamma_shape}, got {gamma.shape}."
            )

        target_band_frequencies = frequencies[
            np.asarray(bzg_targets, dtype="int64")[:, None],
            np.asarray(effective_bands, dtype="int64")[None, :],
        ]
        target_summaries: list[dict[str, Any]] = []
        with self.h5py.File(path, "w") as h5:
            h5.attrs["schema_name"] = "negfpy_phono3py_si_reference"
            h5.attrs["schema_version"] = REFERENCE_SCHEMA_VERSION
            h5.attrs["phono3py_version"] = self.phono3py_version
            h5.attrs["phonopy_version"] = self.phonopy_version
            h5.attrs["frequency_units"] = "cyclic_THz"
            h5.attrs["gamma_units"] = "cyclic_THz_HWHM"
            h5.attrs["interaction_strength_units"] = "eV^2"
            h5.attrs["is_mesh_symmetry"] = False
            h5.attrs["is_nac"] = False
            h5.attrs["is_crystal_symmetry"] = True
            h5.attrs["symprec"] = 1.0e-5
            h5.attrs["use_grg"] = False
            h5.attrs["make_r0_average"] = config.make_r0_average
            h5.attrs["symmetrize_fc3q"] = config.symmetrize_fc3q
            h5.attrs["backend"] = config.backend
            h5.attrs["calculator"] = str(ph3.calculator)
            h5.attrs["frequency_unit_conversion_factor"] = float(
                ph3.unit_conversion_factor
            )
            h5.attrs["cutoff_frequency_THz"] = float(
                interaction.cutoff_frequency
            )

            settings = h5.create_group("settings")
            self._create_dataset(settings, "mesh", config.mesh)
            self._create_dataset(settings, "grid_points_grg", grg_targets)
            self._create_dataset(settings, "grid_points_bzg", bzg_targets)
            self._create_dataset(settings, "band_indices", effective_bands)
            self._create_dataset(
                settings,
                "temperatures",
                config.temperatures,
                units="K",
            )
            self._create_dataset(
                settings,
                "sigmas",
                [config.sigma],
                units="cyclic_THz",
            )
            settings.attrs["sigma_cutoff"] = (
                "None"
                if config.sigma_cutoff is None
                else float(config.sigma_cutoff)
            )

            self._write_crystal_mapping(h5, ph3)
            self._write_grid(h5, bz_grid)

            phonons = h5.create_group("phonons")
            self._create_dataset(
                phonons,
                "frequencies",
                frequencies,
                units="cyclic_THz",
                description="All solved BZG phonon frequencies.",
            )
            self._create_dataset(
                phonons,
                "eigenvectors",
                eigenvectors,
                description=(
                    "Complex phonon eigenvectors in phono3py's native gauge "
                    "and atom-Cartesian basis."
                ),
            )
            self._create_dataset(
                phonons,
                "bz_grid_addresses",
                bz_addresses,
            )
            self._create_dataset(
                phonons,
                "target_band_frequencies",
                target_band_frequencies,
                units="cyclic_THz",
            )

            fc2_random = h5.create_group("fc2_random_q")
            fc2_random.attrs["qpoint_coordinates"] = (
                "primitive_reciprocal_fractional"
            )
            fc2_random.attrs["dynamical_matrix_units"] = (
                "eV/(angstrom^2*amu)"
            )
            fc2_random.attrs["gauge"] = "phonopy_shortest_vector"
            fc2_random.attrs["hermitianized"] = True
            fc2_random.attrs["point_selection"] = (
                "committed_literal_non_mesh_qpoints"
            )
            self._create_dataset(
                fc2_random,
                "qpoints",
                fc2_random_qpoints,
                units="primitive_reciprocal_fractional",
            )
            self._create_dataset(
                fc2_random,
                "dynamical_matrices",
                fc2_random_dynamical_matrices,
                units="eV/(angstrom^2*amu)",
                description=(
                    "Phonopy DynamicalMatrix.run output before frequency "
                    "conversion, in the native shortest-vector gauge."
                ),
            )

            gamma_group = h5.create_group("gamma")
            gamma_dataset = self._create_dataset(
                gamma_group,
                "values",
                gamma,
                units="cyclic_THz_HWHM",
                description=(
                    "Imaginary bubble self-energy Gamma from "
                    "Phono3py.run_imag_self_energy at band frequencies."
                ),
            )
            gamma_dataset.attrs["axis_order"] = (
                "sigma,temperature,target_grid_point,target_band"
            )
            gamma_group.attrs["cyclic_to_angular_HWHM_factor"] = 2 * math.pi
            gamma_group.attrs["cyclic_to_angular_FWHM_factor"] = 4 * math.pi
            gamma_group.attrs["negfpy_linewidth_comparison"] = (
                "NEGFPy linewidth is angular FWHM; multiply this dataset by 4*pi."
            )
            gamma_group.attrs["frequency_points_at_bands"] = True
            if gamma_values.frequency_points is not None:
                self._create_dataset(
                    gamma_group,
                    "frequency_points",
                    gamma_values.frequency_points,
                    units="cyclic_THz",
                )

            targets_group = h5.create_group("target_q")
            for target_index, (grg, bzg) in enumerate(
                zip(grg_targets, bzg_targets, strict=True)
            ):
                interaction.set_grid_point(
                    int(bzg),
                    store_triplets_map=True,
                )
                interaction.run()
                triplets, weights, triplets_map, ir_map = (
                    interaction.get_triplets_at_q()
                )
                strength = interaction.interaction_strength
                if (
                    triplets is None
                    or weights is None
                    or triplets_map is None
                    or ir_map is None
                    or strength is None
                ):
                    raise RuntimeError(
                        f"Incomplete interaction data at target GRG {int(grg)}."
                    )
                triplets = np.asarray(triplets, dtype="int64")
                weights = np.asarray(weights, dtype="int64")
                if len(triplets) != n_q or int(weights.sum()) != n_q:
                    raise RuntimeError(
                        "is_mesh_symmetry=False did not produce the complete "
                        f"{n_q}-point triplet sum at GRG {int(grg)}."
                    )
                frequency_points = np.asarray(
                    target_band_frequencies[target_index],
                    dtype="double",
                )
                integration_weights, g_zero = (
                    self.get_triplets_integration_weights(
                        interaction,
                        frequency_points,
                        config.sigma,
                        sigma_cutoff=config.sigma_cutoff,
                        is_collision_matrix=False,
                        lang=config.backend,
                    )
                )
                integration_weights = np.asarray(
                    integration_weights,
                    dtype="double",
                )
                cutoff = float(interaction.cutoff_frequency)
                valid_triplet_modes = (
                    frequency_points[None, :, None, None] > cutoff
                ) & (
                    frequencies[triplets[:, 1]][:, None, :, None] > cutoff
                ) & (
                    frequencies[triplets[:, 2]][:, None, None, :] > cutoff
                )
                cutoff_integration_weights = np.where(
                    valid_triplet_modes[None, ...],
                    integration_weights,
                    0.0,
                )
                phase_space_channel_sums = np.einsum(
                    "ctfij,t->cf",
                    cutoff_integration_weights,
                    weights,
                    optimize=True,
                ) / n_q

                target = targets_group.create_group(f"grg_{int(grg):08d}")
                target.attrs["target_ordinal"] = target_index
                target.attrs["grid_point_grg"] = int(grg)
                target.attrs["grid_point_bzg"] = int(bzg)
                target.attrs["number_of_full_mesh_triplets"] = len(triplets)
                target.attrs["sum_of_triplet_weights"] = int(weights.sum())
                target.attrs["Nq"] = n_q
                self._create_dataset(target, "triplets_bzg", triplets)
                self._create_dataset(
                    target,
                    "triplet_addresses",
                    bz_grid.addresses[triplets],
                    description=(
                        "BZ-grid addresses for q0,q1,q2; retained so normal/"
                        "Umklapp classification can be independently audited."
                    ),
                )
                self._create_dataset(target, "triplet_weights", weights)
                self._create_dataset(
                    target,
                    "triplets_map_grg",
                    triplets_map,
                )
                self._create_dataset(target, "ir_map_grg", ir_map)
                strength_dataset = self._create_dataset(
                    target,
                    "interaction_strength",
                    strength,
                    units="eV^2",
                )
                strength_dataset.attrs["axis_order"] = (
                    "triplet,target_band,band1,band2"
                )
                integration_dataset = self._create_dataset(
                    target,
                    "integration_weights",
                    integration_weights,
                    units="1/cyclic_THz",
                )
                integration_dataset.attrs["axis_order"] = (
                    "channel,triplet,target_frequency,band1,band2"
                )
                integration_dataset.attrs["channel_0"] = (
                    "Gaussian delta(omega-f1-f2): decay/sum channel"
                )
                integration_dataset.attrs["channel_1"] = (
                    "Gaussian delta(omega+f1-f2) minus "
                    "delta(omega-f1+f2): signed difference channel"
                )
                if g_zero is not None:
                    self._create_dataset(target, "integration_g_zero", g_zero)
                phase_dataset = self._create_dataset(
                    target,
                    "derived_phase_space_channel_sums",
                    phase_space_channel_sums,
                    units="1/cyclic_THz",
                )
                phase_dataset.attrs["axis_order"] = (
                    "channel,target_frequency"
                )
                phase_dataset.attrs["definition"] = (
                    "sum_triplet,band1,band2("
                    "triplet_weight*integration_weight)/Nq"
                )
                phase_dataset.attrs["cutoff_frequency_THz"] = cutoff
                phase_dataset.attrs["cutoff_rule"] = (
                    "strict_all_three_frequencies_above_cutoff"
                )
                phase_dataset.attrs["warning"] = (
                    "Channel 1 is signed. This explicitly defined diagnostic "
                    "is not relabeled as an official phono3py phase-space API."
                )
                target_summaries.append(
                    {
                        "grid_point_grg": int(grg),
                        "grid_point_bzg": int(bzg),
                        "triplets": int(len(triplets)),
                        "triplet_weight_sum": int(weights.sum()),
                        "interaction_strength_shape": _safe_shape(strength),
                        "integration_weights_shape": _safe_shape(
                            integration_weights
                        ),
                    }
                )

        return {
            "gamma_shape": list(gamma.shape),
            "phonon_frequency_shape": _safe_shape(frequencies),
            "phonon_eigenvector_shape": _safe_shape(eigenvectors),
            "fc2_random_qpoint_count": int(len(fc2_random_qpoints)),
            "fc2_random_dynamical_matrix_shape": _safe_shape(
                fc2_random_dynamical_matrices
            ),
            "number_of_mesh_points": n_q,
            "number_of_bz_grid_points": int(len(bz_addresses)),
            "effective_band_indices": [
                int(value) for value in effective_bands
            ],
            "targets": target_summaries,
        }

    def generate_reference(
        self,
        *,
        output_dir: Path,
        source_files: Mapping[str, Path],
        config: GeneratorConfig,
    ) -> Mapping[str, Any]:
        if config.mesh is None:
            raise ValueError("mesh is required for reference generation.")
        ph3 = self._load_from_force_constants(
            output_dir=output_dir,
            source_files=source_files,
            config=config,
        )
        ph3.mesh_numbers = self.np.asarray(config.mesh, dtype="int64")
        if config.band_indices is not None:
            ph3.band_indices = [
                self.np.asarray(config.band_indices, dtype="int64")
            ]
        ph3.sigmas = [config.sigma]
        ph3.sigma_cutoff = config.sigma_cutoff
        ph3.init_phph_interaction(
            symmetrize_fc3q=config.symmetrize_fc3q,
            lapack_zheev_uplo="L",
            openmp_per_triplets=False,
        )
        ph3.run_phonon_solver()
        frequencies, eigenvectors, bz_addresses = ph3.get_phonon_data()
        if frequencies is None or eigenvectors is None or bz_addresses is None:
            raise RuntimeError("phono3py did not return complete phonon data.")
        number_of_bands = int(frequencies.shape[1])
        effective_bands = (
            self.np.arange(number_of_bands, dtype="int64")
            if config.band_indices is None
            else self.np.asarray(config.band_indices, dtype="int64")
        )
        if len(effective_bands) == 0 or int(effective_bands[-1]) >= number_of_bands:
            raise ValueError(
                f"band_indices must be below the {number_of_bands} phonon bands."
            )
        interaction = ph3.phph_interaction
        if interaction is None:
            raise RuntimeError("phono3py interaction was not initialized.")
        if float(interaction.cutoff_frequency) != (
            _PHONO3PY_CUTOFF_FREQUENCY_THZ
        ):
            raise RuntimeError(
                "Unexpected phono3py cutoff frequency: expected "
                f"{_PHONO3PY_CUTOFF_FREQUENCY_THZ} cyclic THz, got "
                f"{interaction.cutoff_frequency}."
            )
        dynamical_matrix = ph3.dynamical_matrix
        if dynamical_matrix is None:
            raise RuntimeError(
                "phono3py did not initialize the harmonic dynamical matrix."
            )
        fc2_random_qpoints = self.np.asarray(
            FC2_RANDOM_QPOINTS,
            dtype="double",
        )
        fc2_random_dynamical_matrices = []
        for qpoint in fc2_random_qpoints:
            dynamical_matrix.run(qpoint)
            matrix = dynamical_matrix.dynamical_matrix
            if matrix is None:
                raise RuntimeError(
                    "Phonopy returned no random-q dynamical matrix."
                )
            fc2_random_dynamical_matrices.append(
                self.np.array(
                    matrix,
                    dtype="complex128",
                    order="C",
                    copy=True,
                )
            )
        fc2_random_dynamical_matrices = self.np.asarray(
            fc2_random_dynamical_matrices,
            dtype="complex128",
            order="C",
        )
        bz_grid = interaction.bz_grid
        grg_targets = self.np.asarray(config.grid_points, dtype="int64")
        if (
            len(bz_grid.grg2bzg) != math.prod(config.mesh)
            or int(grg_targets[-1]) >= len(bz_grid.grg2bzg)
        ):
            raise RuntimeError(
                "Configured GRG target indices do not match phono3py's grid."
            )
        bzg_targets = self.np.asarray(
            bz_grid.grg2bzg[grg_targets],
            dtype="int64",
        )
        gamma_values = ph3.run_imag_self_energy(
            bzg_targets,
            temperatures=self.np.asarray(config.temperatures, dtype="double"),
            frequency_points_at_bands=True,
            write_txt=False,
            write_gamma_detail=False,
            keep_gamma_detail=False,
        )

        hdf_summary: dict[str, Any] = {}
        self._publish_hdf5(
            output_dir / REFERENCE_FILENAME,
            lambda path: hdf_summary.update(
                self._write_reference_hdf5(
                    path,
                    ph3=ph3,
                    config=config,
                    frequencies=frequencies,
                    eigenvectors=eigenvectors,
                    bz_addresses=bz_addresses,
                    effective_bands=effective_bands,
                    grg_targets=grg_targets,
                    bzg_targets=bzg_targets,
                    gamma_values=gamma_values,
                    fc2_random_qpoints=fc2_random_qpoints,
                    fc2_random_dynamical_matrices=(
                        fc2_random_dynamical_matrices
                    ),
                )
            ),
        )
        return {
            "implementation": "phono3py_4_4_0",
            "public_apis": [
                "phono3py.load",
                "phono3py.api_phono3py.Phono3py.init_phph_interaction",
                "phono3py.api_phono3py.Phono3py.run_phonon_solver",
                "phono3py.api_phono3py.Phono3py.get_phonon_data",
                "phono3py.api_phono3py.Phono3py.dynamical_matrix",
                "phono3py.api_phono3py.Phono3py.run_imag_self_energy",
                "phonopy.harmonic.dynamical_matrix.DynamicalMatrix.run",
                "phono3py.phonon3.interaction.Interaction.set_grid_point",
                "phono3py.phonon3.interaction.Interaction.run",
                "phono3py.phonon3.interaction.Interaction.get_triplets_at_q",
                "phono3py.phonon3.triplets.get_triplets_integration_weights",
            ],
            "private_apis": [],
            "phono3py_version": self.phono3py_version,
            "phonopy_version": self.phonopy_version,
            "mesh": list(config.mesh),
            "grid_points_grg": [int(value) for value in grg_targets],
            "grid_points_bzg": [int(value) for value in bzg_targets],
            "band_indices": [int(value) for value in effective_bands],
            "temperatures_K": list(config.temperatures),
            "sigma_THz": config.sigma,
            "sigma_cutoff": config.sigma_cutoff,
            "gamma_units": "cyclic_THz_HWHM",
            "interaction_strength_units": "eV^2",
            "integration_weight_units": "1/cyclic_THz",
            "is_mesh_symmetry": False,
            "is_nac": False,
            "is_crystal_symmetry": True,
            "symprec": 1.0e-5,
            "use_grg": False,
            "make_r0_average": config.make_r0_average,
            "symmetrize_fc3q": config.symmetrize_fc3q,
            "lapack_zheev_uplo": "L",
            "openmp_per_triplets": False,
            "calculator": str(ph3.calculator),
            "frequency_unit_conversion_factor": float(
                ph3.unit_conversion_factor
            ),
            "cutoff_frequency_THz": float(interaction.cutoff_frequency),
            "constant_averaged_interaction": None,
            "frequency_scale_factor": None,
            "backend": config.backend,
            **hdf_summary,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate checksummed phono3py 4.4.0 Si force constants and "
            "bounded target-q reference data."
        )
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Explicit new/empty output directory; artifacts are never overwritten.",
    )
    parser.add_argument(
        "--mesh",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
    )
    parser.add_argument(
        "--grid-points",
        nargs="+",
        type=int,
        default=[0],
        metavar="GRG_INDEX",
        help="Bounded target q-points in phono3py GRG x-fast index order.",
    )
    parser.add_argument(
        "--band-indices",
        nargs="+",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--temperatures",
        nargs="+",
        type=float,
        default=[300.0],
        metavar="K",
    )
    parser.add_argument("--sigma", type=float, default=0.1, metavar="THZ")
    parser.add_argument("--sigma-cutoff", type=float, default=None)
    parser.add_argument("--backend", choices=("C", "Rust"), default="Rust")
    parser.add_argument(
        "--make-r0-average",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--symmetrize-fc3q",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--force-constants-only",
        action="store_true",
        help="Generate both compact/full FC2/FC3 but no target-q reference.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    config = GeneratorConfig(
        mesh=arguments.mesh,
        grid_points=arguments.grid_points,
        band_indices=arguments.band_indices,
        temperatures=arguments.temperatures,
        sigma=arguments.sigma,
        sigma_cutoff=arguments.sigma_cutoff,
        backend=arguments.backend,
        make_r0_average=arguments.make_r0_average,
        symmetrize_fc=arguments.symmetrize_fc,
        symmetrize_fc3q=arguments.symmetrize_fc3q,
        force_constants_only=arguments.force_constants_only,
    )
    result = run_generator(
        manifest_path=arguments.manifest,
        source_dir=arguments.source_dir,
        output_dir=arguments.output_dir,
        config=config,
    )
    print(
        json.dumps(
            {
                "stage": result.stage,
                "output_dir": str(result.output_dir),
                "inventory": str(result.inventory_path),
                "sha256sums": str(result.sha256sums_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
