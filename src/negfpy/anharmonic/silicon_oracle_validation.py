"""Bounded, provenance-strict silicon bulk-oracle validation.

This module is an orchestration layer around the independent NEGFPy bulk
three-phonon kernel and the pinned phono3py reference comparison.  It does not
introduce another physical approximation, tune a prefactor, or split an
internal q-point sum.  One invocation evaluates the complete internal
``q1``/branch sum for the explicitly archived target q-points.

The driver intentionally consumes the selections stored in the immutable
reference HDF5.  Supplying a second, potentially inconsistent set of mesh,
target, band, temperature, sigma, or cutoff controls is not supported.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, fields, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ._provenance import code_provenance
from .bulk_harmonic import compute_bulk_phonons
from .bulk_three_phonon import (
    BulkThreePhononProblem,
    compute_three_phonon_linewidths,
)
from .io.phono3py import read_phono3py_fc3
from .io.phono3py_crystal import read_phono3py_crystal_mapping
from .io.phono3py_fc2 import read_phono3py_fc2
from .momentum import UniformBZMesh
from .phono3py_reference import (
    InteractionStrengthSample,
    compare_silicon_reference,
    read_phono3py_silicon_reference,
)
from .translational_fc3 import convert_fc3_to_translational
from .translational_fc3_validation import validate_translational_fc3


SILICON_ORACLE_REPORT_SCHEMA = "negfpy.silicon_bulk_oracle_validation.v1"
V3_SAMPLE_MANIFEST_SCHEMA = "negfpy.silicon_v3_sample_manifest.v1"
V3_SAMPLE_SELECTION_ALGORITHM = (
    "deterministic_stratified_reference_invariant_blocks"
)
V3_SAMPLE_SELECTION_ALGORITHM_VERSION = 1
V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM = (
    "deterministic_bounded_streaming_reference_invariant_blocks"
)
V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION = 2
PINNED_PHONO3PY_VERSION = "4.4.0"
PINNED_PHONOPY_VERSION = "4.4.0"
PINNED_SPGLIB_VERSION = "2.7.0"
PINNED_SILICON_SOURCE_COMMIT = (
    "302ad205452ee36e57887b10bbf4a0289dd037bb"
)
PINNED_SILICON_DATASET_ID = "phono3py-4.4.0-Si-PBEsol"
PINNED_SILICON_REPOSITORY = "https://github.com/phonopy/phono3py.git"
PINNED_SILICON_ENVIRONMENT_LOCK_SHA256 = (
    "6d8e017b6f1ead8be3ec9fb8abb1c8c1d02fe0de852aad9cbfc69d08ce341bfc"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MAX_MESH_POINTS = 4096
_MAX_TARGET_GRID_POINTS = 32
_MAX_TARGET_BANDS = 128
_MAX_TEMPERATURES = 64
_MAX_INTERACTION_SAMPLES = 512
_MIN_PROMOTED_BLOCKS = 64
_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_STRATUM_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link.")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file.")
    return resolved


def _directory(path: str | os.PathLike[str], *, name: str) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link.")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir() or resolved.is_symlink():
        raise ValueError(f"{name} must be a non-symlink directory.")
    return resolved


def _reject_duplicate_keys(
    pairs: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}.")
        result[key] = value
    return result


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object.")
    return value


def _parse_sha256sums(path: Path) -> Mapping[str, str]:
    records: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("checksums_path must be readable UTF-8 text.") from exc
    if not lines:
        raise ValueError("checksums_path must not be empty.")
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise ValueError("checksums_path contains a malformed line.")
        digest = line[:64].lower()
        relative_name = line[66:]
        if _SHA256_RE.fullmatch(digest) is None:
            raise ValueError("checksums_path contains a malformed SHA256.")
        candidate = Path(relative_name)
        if (
            not relative_name
            or candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in relative_name
            or candidate.as_posix() != relative_name
            or relative_name in records
        ):
            raise ValueError(
                "checksums_path contains an unsafe or duplicate path."
            )
        records[relative_name] = digest
    return records


def _manifest_identity(path: Path, yaml_sha256: str) -> dict[str, Any]:
    manifest = _read_json_object(path, name="input_manifest_path")
    if manifest.get("schema_version") != 1:
        raise ValueError("The input manifest schema_version must equal 1.")
    if manifest.get("dataset_id") != PINNED_SILICON_DATASET_ID:
        raise ValueError("The input manifest is not the pinned Si-PBEsol set.")
    versions = manifest.get("versions")
    if versions != {
        "phono3py": PINNED_PHONO3PY_VERSION,
        "phonopy": PINNED_PHONOPY_VERSION,
    }:
        raise ValueError("The input manifest has unpinned package versions.")
    source = manifest.get("source")
    if not isinstance(source, dict):
        raise ValueError("The input manifest source must be an object.")
    if source.get("repository") != PINNED_SILICON_REPOSITORY:
        raise ValueError("The input manifest source repository is not pinned.")
    if source.get("commit") != PINNED_SILICON_SOURCE_COMMIT:
        raise ValueError("The input manifest source commit is not pinned.")
    source_files = source.get("files")
    if not isinstance(source_files, dict):
        raise ValueError("The input manifest source.files must be an object.")
    yaml_record = source_files.get("phono3py_yaml")
    if not isinstance(yaml_record, dict):
        raise ValueError("The input manifest lacks phono3py_yaml provenance.")
    expected_yaml_sha = yaml_record.get("sha256")
    if (
        not isinstance(expected_yaml_sha, str)
        or _SHA256_RE.fullmatch(expected_yaml_sha.lower()) is None
        or expected_yaml_sha.lower() != yaml_sha256
    ):
        raise ValueError(
            "The crystal-mapping YAML does not match the pinned manifest."
        )
    return {
        "dataset_id": manifest["dataset_id"],
        "source_repository": source["repository"],
        "source_commit": source["commit"],
        "phono3py_yaml_relative_path": yaml_record.get("path"),
        "phono3py_yaml_sha256": expected_yaml_sha.lower(),
        "input_manifest_sha256": _sha256(path),
    }


def _runtime_identity() -> dict[str, Any]:
    package_versions: dict[str, str] = {}
    for package in (
        "negfpy",
        "numpy",
        "scipy",
        "h5py",
        "phono3py",
        "phonopy",
        "spglib",
        "mpi4py",
    ):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"
    required = {
        "phono3py": PINNED_PHONO3PY_VERSION,
        "phonopy": PINNED_PHONOPY_VERSION,
        "spglib": PINNED_SPGLIB_VERSION,
        "mpi4py": "4.1.2",
    }
    mismatches = {
        name: {
            "expected": expected,
            "actual": package_versions[name],
        }
        for name, expected in required.items()
        if package_versions[name] != expected
    }
    if mismatches:
        raise RuntimeError(
            "The silicon oracle requires the pinned phono3py/phonopy/spglib "
            f"runtime: {mismatches}."
        )
    thread_environment = {
        name: os.environ.get(name) for name in _THREAD_VARIABLES
    }
    invalid_threads = {
        name: value
        for name, value in thread_environment.items()
        if value != "1"
    }
    if invalid_threads:
        raise RuntimeError(
            "The serial silicon oracle requires every BLAS/OpenMP thread "
            f"control to equal '1': {invalid_threads}."
        )
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": package_versions,
        "thread_environment": thread_environment,
    }


def _git(
    repository: Path,
    *arguments: str,
) -> str:
    result = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _repository_identity(
    repository: Path,
    *,
    require_clean: bool,
) -> dict[str, Any]:
    try:
        top_level = Path(_git(repository, "rev-parse", "--show-toplevel"))
        head = _git(repository, "rev-parse", "HEAD")
        tree = _git(repository, "rev-parse", "HEAD^{tree}")
        status = _git(
            repository,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("repository_path is not a readable Git worktree.") from exc
    clean = status == ""
    if require_clean and not clean:
        raise RuntimeError(
            "The silicon-oracle source worktree is dirty; refusing a "
            "provenance-eligible run."
        )
    return {
        "repository": str(top_level.resolve()),
        "head_commit": head,
        "tree_hash": tree,
        "clean": clean,
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_entry_count": 0 if clean else len(status.splitlines()),
    }


def _code_identity(
    repository: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    provenance = dict(code_provenance())
    source_hash = provenance.get("source_tree_sha256")
    revision = provenance.get("git_revision")
    installed_version = provenance.get("installed_package_version")
    runtime_version = runtime.get("packages", {}).get("negfpy")
    checks = {
        "source_tree_sha256_known": bool(
            isinstance(source_hash, str)
            and _SHA256_RE.fullmatch(source_hash) is not None
        ),
        "git_revision_matches_repository": bool(
            isinstance(revision, str)
            and revision == repository.get("head_commit")
        ),
        "git_clean": provenance.get("git_dirty") is False,
        "package_version_consistent": (
            provenance.get("package_version_consistent") is True
        ),
        "runtime_version_matches": bool(
            isinstance(installed_version, str)
            and installed_version != "unknown"
            and installed_version == runtime_version
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "record": provenance,
    }


def _generation_environment_identity(
    config: SiliconOracleValidationConfig,
    repository: Mapping[str, Any],
) -> dict[str, Any]:
    script_path = config.repository_path / "scripts" / (
        "generate_phono3py_si_reference.py"
    )
    if script_path.is_symlink() or not script_path.is_file():
        raise RuntimeError(
            "The validation repository lacks the regular silicon reference "
            "generator script."
        )
    current_script_sha256 = _sha256(script_path)
    records: dict[str, Any] = {}
    all_passed = True
    for label, filename in (
        ("force_constants", "force-constants-environment.json"),
        ("reference", "reference-environment.json"),
    ):
        path = config.checksums_path.parent / filename
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"The generated artifact lacks regular {filename}."
            )
        document = _read_json_object(path, name=filename)
        if set(document) != {
            "schema_version",
            "recorded_at_utc",
            "identity",
        } or document["schema_version"] != 1:
            raise ValueError(f"{filename} has an invalid schema.")
        identity = document["identity"]
        if not isinstance(identity, dict):
            raise ValueError(f"{filename} identity must be an object.")
        code = identity.get("code")
        versions = identity.get("runtime_versions")
        threads = identity.get("thread_environment")
        if not all(isinstance(value, dict) for value in (code, versions, threads)):
            raise ValueError(f"{filename} lacks required identity objects.")
        checks = {
            "git_commit_matches_validator": (
                code.get("git_commit") == repository.get("head_commit")
            ),
            "tracked_worktree_clean": (
                code.get("tracked_worktree_status") == ""
            ),
            "script_sha256_matches_validator": (
                code.get("script_sha256") == current_script_sha256
            ),
            "phono3py_version_pinned": (
                versions.get("phono3py") == PINNED_PHONO3PY_VERSION
            ),
            "phonopy_version_pinned": (
                versions.get("phonopy") == PINNED_PHONOPY_VERSION
            ),
            "threads_exactly_one": all(
                threads.get(name) == "1" for name in _THREAD_VARIABLES
            ),
        }
        passed = all(checks.values())
        all_passed = all_passed and passed
        records[label] = {
            "path": str(path.resolve()),
            "sha256": _sha256(path),
            "passed": passed,
            "checks": checks,
            "record": document,
        }
    return {
        "passed": all_passed,
        "current_generator_script": {
            "path": str(script_path.resolve()),
            "sha256": current_script_sha256,
        },
        "records": records,
    }


def array_sha256(values: Any) -> str:
    """Hash an array with explicit dtype, shape, and C-order byte identity."""

    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(value)) for value in array.shape).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def crystal_mapping_sha256(mapping: Any) -> str:
    """Return a content hash of the scientific crystal-mapping fields."""

    digest = hashlib.sha256()
    digest.update(str(mapping.purpose).encode("utf-8"))
    digest.update(b"\0")
    arrays = (
        mapping.primitive_matrix,
        mapping.supercell_matrix,
        mapping.primitive.lattice,
        mapping.primitive.scaled_positions,
        mapping.primitive.masses,
        mapping.supercell.lattice,
        mapping.supercell.scaled_positions,
        mapping.supercell.masses,
        mapping.p2s,
        mapping.s2p,
        mapping.p2p,
        mapping.shortest_vectors,
        mapping.multiplicity,
        mapping.atomic_permutations,
        mapping.translation_permutations,
    )
    for values in arrays:
        if values is None:
            digest.update(b"none\0")
        else:
            digest.update(array_sha256(values).encode("ascii"))
            digest.update(b"\0")
    for symbols in (mapping.primitive.symbols, mapping.supercell.symbols):
        digest.update(
            json.dumps(
                list(symbols),
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(b"\0")
    return digest.hexdigest()


def _manifest_index(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a nonnegative integer.")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a nonnegative integer.")
    return result


def _manifest_index_tuple(value: Any, *, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a nonempty JSON integer array.")
    result = tuple(
        _manifest_index(item, name=f"{name}[{position}]")
        for position, item in enumerate(value)
    )
    if tuple(sorted(set(result))) != result:
        raise ValueError(f"{name} must be strictly increasing.")
    return result


def _manifest_digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA256 string.")
    result = value.lower()
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{name} must be a 64-hex SHA256.")
    return result


@dataclass(frozen=True, slots=True)
class V3InteractionSampleRecord:
    """One scalar seed and its predeclared gauge-invariant promoted block."""

    q0_grg: int
    q1_grg: int
    expected_q2_grg: int
    b0: int
    b1: int
    b2: int
    target_subspace: tuple[int, ...]
    band1_subspace: tuple[int, ...]
    band2_subspace: tuple[int, ...]
    process: str
    reference_invariant_block_sum_ev2: float
    stratum: str

    @property
    def comparator_sample(self) -> InteractionStrengthSample:
        return InteractionStrengthSample(
            self.q0_grg,
            self.q1_grg,
            self.b0,
            self.b1,
            self.b2,
        )

    @property
    def promoted_block_key(
        self,
    ) -> tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        return (
            self.q0_grg,
            self.q1_grg,
            self.target_subspace,
            self.band1_subspace,
            self.band2_subspace,
        )


@dataclass(frozen=True, slots=True)
class V3InteractionSampleManifest:
    """Detached canonical manifest tied to one exact silicon reference."""

    path: Path
    sha256: str
    inputs: Mapping[str, str]
    mapping_sha256: Mapping[str, str]
    mesh: tuple[int, int, int]
    grg2bzg_sha256: str
    bzg2grg_sha256: str
    degeneracy_tolerance_thz: float
    selection_algorithm: str
    selection_algorithm_version: int
    robust_nonzero_floor_ev2: float
    cover_all_target_grid_points: bool
    cover_all_target_bands: bool
    required_processes: tuple[str, ...]
    required_strata: tuple[str, ...]
    empty_strata: tuple[str, ...]
    minimum_promoted_blocks: int
    samples: tuple[V3InteractionSampleRecord, ...]

    @property
    def comparator_samples(self) -> tuple[InteractionStrengthSample, ...]:
        return tuple(sample.comparator_sample for sample in self.samples)


@dataclass(frozen=True, slots=True)
class SiliconOracleValidationConfig:
    """Inputs and hard resource bounds for one serial validation case."""

    reference_path: Path
    fc2_compact_path: Path
    fc2_full_path: Path
    fc3_compact_path: Path
    fc3_full_path: Path
    phono3py_yaml_path: Path
    input_manifest_path: Path
    checksums_path: Path
    environment_lock_path: Path
    interaction_samples_path: Path
    repository_path: Path
    output_path: Path
    require_clean_source: bool = True
    asr_atol: float = 1.0e-10
    asr_rtol: float = 1.0e-8

    def __post_init__(self) -> None:
        file_names = (
            "reference_path",
            "fc2_compact_path",
            "fc2_full_path",
            "fc3_compact_path",
            "fc3_full_path",
            "phono3py_yaml_path",
            "input_manifest_path",
            "checksums_path",
            "environment_lock_path",
            "interaction_samples_path",
        )
        for name in file_names:
            object.__setattr__(
                self,
                name,
                _regular_file(getattr(self, name), name=name),
            )
        if len(
            {
                self.fc2_compact_path,
                self.fc2_full_path,
                self.fc3_compact_path,
                self.fc3_full_path,
            }
        ) != 4:
            raise ValueError(
                "Compact/full FC2/FC3 paths must identify four distinct files."
            )
        object.__setattr__(
            self,
            "repository_path",
            _directory(self.repository_path, name="repository_path"),
        )
        output = Path(self.output_path).expanduser().resolve()
        if output.exists():
            raise FileExistsError(
                "output_path already exists; validation reports are immutable."
            )
        object.__setattr__(self, "output_path", output)

        if not isinstance(self.require_clean_source, (bool, np.bool_)):
            raise TypeError("require_clean_source must be boolean.")
        read_interaction_sample_manifest(self.interaction_samples_path)
        for name in ("asr_atol", "asr_rtol"):
            if isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a real scalar, not boolean.")
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative.")
            object.__setattr__(self, name, value)


def _input_records(
    config: SiliconOracleValidationConfig,
) -> dict[str, dict[str, Any]]:
    paths = {
        "reference_hdf5": config.reference_path,
        "fc2_compact_hdf5": config.fc2_compact_path,
        "fc2_full_hdf5": config.fc2_full_path,
        "fc3_compact_hdf5": config.fc3_compact_path,
        "fc3_full_hdf5": config.fc3_full_path,
        "phono3py_yaml": config.phono3py_yaml_path,
        "input_manifest": config.input_manifest_path,
        "sha256sums": config.checksums_path,
        "environment_lock": config.environment_lock_path,
        "interaction_samples": config.interaction_samples_path,
    }
    records: dict[str, dict[str, Any]] = {}
    for role, path in paths.items():
        stat = path.stat()
        records[role] = {
            "path": str(path),
            "sha256": _sha256(path),
            "size_bytes": stat.st_size,
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "mtime_ns": stat.st_mtime_ns,
        }
    return records


def _verify_generated_checksums(
    config: SiliconOracleValidationConfig,
    input_records: Mapping[str, Mapping[str, Any]],
) -> None:
    checksums = _parse_sha256sums(config.checksums_path)
    root = config.checksums_path.parent
    for relative_name, expected in checksums.items():
        unresolved = root / relative_name
        if unresolved.is_symlink():
            raise ValueError(
                f"Checksummed artifact {relative_name!r} is a symbolic link."
            )
        try:
            artifact = unresolved.resolve(strict=True)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Checksummed artifact {relative_name!r} is missing."
            ) from exc
        if not artifact.is_relative_to(root) or not artifact.is_file():
            raise ValueError(
                f"Checksummed artifact {relative_name!r} is not a safe "
                "regular file."
            )
        if _sha256(artifact) != expected:
            raise ValueError(
                f"Checksummed artifact {relative_name!r} has changed."
            )
    for role, path in (
        ("reference_hdf5", config.reference_path),
        ("fc2_compact_hdf5", config.fc2_compact_path),
        ("fc2_full_hdf5", config.fc2_full_path),
        ("fc3_compact_hdf5", config.fc3_compact_path),
        ("fc3_full_hdf5", config.fc3_full_path),
    ):
        if path.parent != root:
            raise ValueError(
                f"{role} must be a direct child of the generated artifact "
                "directory containing checksums_path."
            )
        expected = checksums.get(path.name)
        if expected is None:
            raise ValueError(f"checksums_path has no entry for {path.name}.")
        if expected != input_records[role]["sha256"]:
            raise ValueError(f"{role} does not match checksums_path.")


def _verify_inputs_unchanged(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> None:
    for role in before:
        for field in ("sha256", "size_bytes", "device", "inode", "mtime_ns"):
            if before[role][field] != after[role][field]:
                raise RuntimeError(
                    f"Input {role!r} changed during validation ({field})."
                )


def _validate_bounded_reference(reference: Any) -> None:
    n_mesh = int(np.prod(reference.mesh))
    limits = {
        "mesh points": (n_mesh, _MAX_MESH_POINTS),
        "target grid points": (
            int(reference.grid_points_grg.size),
            _MAX_TARGET_GRID_POINTS,
        ),
        "target bands": (
            int(reference.band_indices.size),
            _MAX_TARGET_BANDS,
        ),
        "temperatures": (
            int(reference.temperatures_k.size),
            _MAX_TEMPERATURES,
        ),
    }
    for name, (actual, maximum) in limits.items():
        if actual < 1 or actual > maximum:
            raise ValueError(
                f"Reference {name}={actual} violates the bounded driver "
                f"limit of {maximum}."
            )
    if not reference.make_r0_average:
        raise ValueError(
            "The production silicon oracle requires make_r0_average=True."
        )
    if reference.fc2_random_qpoints is None:
        raise ValueError(
            "The reference lacks the mandatory independent random-q FC2 set."
        )


def read_interaction_sample_manifest(
    path: str | os.PathLike[str],
) -> V3InteractionSampleManifest:
    """Read a provenance-bound manifest of invariant ``|V3|^2`` blocks."""

    source = _regular_file(path, name="interaction_samples_path")
    payload = _read_json_object(source, name="interaction_samples_path")
    expected_root = {
        "schema",
        "schema_version",
        "inputs",
        "mapping_sha256",
        "mesh",
        "grg2bzg_sha256",
        "bzg2grg_sha256",
        "degeneracy_tolerance_thz",
        "selection_algorithm",
        "selection_algorithm_version",
        "robust_nonzero_floor_ev2",
        "coverage",
        "samples",
    }
    if set(payload) != expected_root:
        raise ValueError(
            "The interaction sample manifest must contain exactly "
            f"{sorted(expected_root)}."
        )
    if (
        payload["schema"] != V3_SAMPLE_MANIFEST_SCHEMA
        or payload["schema_version"] != 1
    ):
        raise ValueError(
            "The interaction sample manifest schema identity is unsupported."
        )
    expected_input_keys = {
        "reference_hdf5_sha256",
        "fc2_compact_hdf5_sha256",
        "fc2_full_hdf5_sha256",
        "fc3_compact_hdf5_sha256",
        "fc3_full_hdf5_sha256",
        "phono3py_yaml_sha256",
        "input_manifest_sha256",
    }
    raw_inputs = payload["inputs"]
    if not isinstance(raw_inputs, dict) or set(raw_inputs) != expected_input_keys:
        raise ValueError(
            "The interaction sample manifest inputs have an invalid schema."
        )
    inputs = {
        key: _manifest_digest(value, name=f"inputs.{key}")
        for key, value in raw_inputs.items()
    }
    raw_mapping_hashes = payload["mapping_sha256"]
    if (
        not isinstance(raw_mapping_hashes, dict)
        or set(raw_mapping_hashes) != {"fc2", "fc3"}
    ):
        raise ValueError(
            "mapping_sha256 must contain exactly fc2 and fc3."
        )
    mapping_hashes = {
        key: _manifest_digest(value, name=f"mapping_sha256.{key}")
        for key, value in raw_mapping_hashes.items()
    }
    raw_mesh = payload["mesh"]
    if not isinstance(raw_mesh, list) or len(raw_mesh) != 3:
        raise ValueError("mesh must contain three positive integers.")
    mesh = tuple(
        _manifest_index(value, name=f"mesh[{axis}]")
        for axis, value in enumerate(raw_mesh)
    )
    if any(value <= 0 for value in mesh):
        raise ValueError("mesh must contain three positive integers.")
    degeneracy = float(payload["degeneracy_tolerance_thz"])
    floor = float(payload["robust_nonzero_floor_ev2"])
    if (
        not np.isfinite(degeneracy)
        or degeneracy <= 0.0
        or not np.isfinite(floor)
        or floor <= 0.0
    ):
        raise ValueError(
            "Degeneracy tolerance and robust nonzero floor must be finite "
            "and positive."
        )
    selection_algorithm = payload["selection_algorithm"]
    selection_version = payload["selection_algorithm_version"]
    if (
        not isinstance(selection_algorithm, str)
        or not selection_algorithm
        or isinstance(selection_version, bool)
        or not isinstance(selection_version, int)
    ):
        raise ValueError(
            "The V3 sample selection algorithm/version types are invalid."
        )
    selection_identity = (selection_algorithm, selection_version)
    legacy_selection_identity = (
        V3_SAMPLE_SELECTION_ALGORITHM,
        V3_SAMPLE_SELECTION_ALGORITHM_VERSION,
    )
    bounded_selection_identity = (
        V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM,
        V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION,
    )
    if selection_identity not in (
        legacy_selection_identity,
        bounded_selection_identity,
    ):
        raise ValueError(
            "The V3 sample selection algorithm/version pair is not pinned."
        )

    coverage = payload["coverage"]
    coverage_keys = {
        "cover_all_target_grid_points",
        "cover_all_target_bands",
        "required_processes",
        "required_strata",
        "empty_strata",
        "minimum_promoted_blocks",
    }
    if not isinstance(coverage, dict) or set(coverage) != coverage_keys:
        raise ValueError("The V3 sample coverage schema is invalid.")
    expected_coverage = (
        (True, True)
        if selection_identity == legacy_selection_identity
        else (False, True)
    )
    if any(
        coverage[name] is not expected
        for name, expected in zip(
            (
                "cover_all_target_grid_points",
                "cover_all_target_bands",
            ),
            expected_coverage,
            strict=True,
        )
    ):
        raise ValueError(
            "The V3 sample coverage booleans do not match the pinned "
            "selection algorithm/version contract."
        )
    observed_coverage = expected_coverage
    raw_processes = coverage["required_processes"]
    if (
        not isinstance(raw_processes, list)
        or raw_processes != ["normal", "umklapp"]
    ):
        raise ValueError(
            "required_processes must be the pinned canonical normal/umklapp "
            "pair."
        )
    raw_strata = coverage["required_strata"]
    if (
        not isinstance(raw_strata, list)
        or not raw_strata
        or any(
            not isinstance(value, str)
            or _STRATUM_RE.fullmatch(value) is None
            for value in raw_strata
        )
        or len(set(raw_strata)) != len(raw_strata)
    ):
        raise ValueError(
            "required_strata must be a nonempty unique canonical string list."
        )
    raw_empty_strata = coverage["empty_strata"]
    if (
        not isinstance(raw_empty_strata, list)
        or any(
            not isinstance(value, str)
            or _STRATUM_RE.fullmatch(value) is None
            for value in raw_empty_strata
        )
        or len(set(raw_empty_strata)) != len(raw_empty_strata)
        or set(raw_empty_strata) & set(raw_strata)
    ):
        raise ValueError(
            "empty_strata must be a unique canonical string list disjoint "
            "from required_strata."
        )
    expected_strata = {
        f"{region}.{target}.{internal}.{process}"
        for region in ("gamma", "near", "generic")
        for target in ("acoustic", "optical")
        for internal in ("aa", "ao", "oo")
        for process in ("normal", "umklapp")
    }
    if (
        raw_strata != sorted(raw_strata)
        or raw_empty_strata != sorted(raw_empty_strata)
        or set(raw_strata).union(raw_empty_strata) != expected_strata
    ):
        raise ValueError(
            "required_strata and empty_strata must be the sorted, disjoint "
            "partition of the pinned 36-stratum V3 space."
        )
    minimum_blocks = _manifest_index(
        coverage["minimum_promoted_blocks"],
        name="coverage.minimum_promoted_blocks",
    )
    if minimum_blocks < _MIN_PROMOTED_BLOCKS:
        raise ValueError(
            "minimum_promoted_blocks must be at least "
            f"{_MIN_PROMOTED_BLOCKS}."
        )

    raw_samples = payload["samples"]
    if (
        not isinstance(raw_samples, list)
        or not raw_samples
        or len(raw_samples) > _MAX_INTERACTION_SAMPLES
    ):
        raise ValueError(
            "The interaction sample manifest must contain between 1 and "
            f"{_MAX_INTERACTION_SAMPLES} samples."
        )
    sample_keys = {
        "q0_grg",
        "q1_grg",
        "expected_q2_grg",
        "b0",
        "b1",
        "b2",
        "target_subspace",
        "band1_subspace",
        "band2_subspace",
        "process",
        "reference_invariant_block_sum_ev2",
        "stratum",
    }
    samples: list[V3InteractionSampleRecord] = []
    seen_blocks: set[
        tuple[int, int, tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    ] = set()
    for position, item in enumerate(raw_samples):
        if not isinstance(item, dict) or set(item) != sample_keys:
            raise ValueError(
                "Every interaction sample must contain exactly "
                f"{sorted(sample_keys)} (sample {position})."
            )
        process = item["process"]
        stratum = item["stratum"]
        if process not in {"normal", "umklapp"}:
            raise ValueError(f"Sample {position} has an invalid process.")
        if (
            not isinstance(stratum, str)
            or _STRATUM_RE.fullmatch(stratum) is None
        ):
            raise ValueError(f"Sample {position} has an invalid stratum.")
        if (
            process not in raw_processes
            or stratum not in raw_strata
            or stratum.rsplit(".", 1)[-1] != process
        ):
            raise ValueError(
                f"Sample {position} conflicts with declared process/stratum "
                "coverage."
            )
        block_sum = float(item["reference_invariant_block_sum_ev2"])
        if not np.isfinite(block_sum) or block_sum < floor:
            raise ValueError(
                f"Sample {position} does not exceed the robust nonzero floor."
            )
        sample = V3InteractionSampleRecord(
            q0_grg=_manifest_index(item["q0_grg"], name="q0_grg"),
            q1_grg=_manifest_index(item["q1_grg"], name="q1_grg"),
            expected_q2_grg=_manifest_index(
                item["expected_q2_grg"],
                name="expected_q2_grg",
            ),
            b0=_manifest_index(item["b0"], name="b0"),
            b1=_manifest_index(item["b1"], name="b1"),
            b2=_manifest_index(item["b2"], name="b2"),
            target_subspace=_manifest_index_tuple(
                item["target_subspace"],
                name="target_subspace",
            ),
            band1_subspace=_manifest_index_tuple(
                item["band1_subspace"],
                name="band1_subspace",
            ),
            band2_subspace=_manifest_index_tuple(
                item["band2_subspace"],
                name="band2_subspace",
            ),
            process=process,
            reference_invariant_block_sum_ev2=block_sum,
            stratum=stratum,
        )
        if (
            sample.b0 not in sample.target_subspace
            or sample.b1 not in sample.band1_subspace
            or sample.b2 not in sample.band2_subspace
        ):
            raise ValueError(
                f"Sample {position} seed bands are outside promoted subspaces."
            )
        if sample.promoted_block_key in seen_blocks:
            raise ValueError(
                "The interaction sample manifest contains duplicate promoted "
                f"blocks at sample {position}."
            )
        seen_blocks.add(sample.promoted_block_key)
        samples.append(sample)
    if len(samples) < minimum_blocks:
        raise ValueError(
            "The sample manifest does not meet minimum_promoted_blocks."
        )
    processes = {sample.process for sample in samples}
    strata = {sample.stratum for sample in samples}
    if not set(raw_processes).issubset(processes):
        raise ValueError("The sample manifest omits a required process.")
    if not set(raw_strata).issubset(strata):
        raise ValueError("The sample manifest omits a required stratum.")
    if set(raw_empty_strata) & strata:
        raise ValueError("A declared empty stratum contains a sample.")
    if selection_identity == bounded_selection_identity:
        stratum_counts = {
            name: sum(sample.stratum == name for sample in samples)
            for name in raw_strata
        }
        sparse = {
            name: count
            for name, count in stratum_counts.items()
            if count < 2
        }
        if sparse:
            raise ValueError(
                "Bounded V3 manifests require two distinct promoted blocks "
                f"per populated stratum: {sparse}."
            )
    return V3InteractionSampleManifest(
        path=source,
        sha256=_sha256(source),
        inputs=MappingProxyType(inputs),
        mapping_sha256=MappingProxyType(mapping_hashes),
        mesh=mesh,  # type: ignore[arg-type]
        grg2bzg_sha256=_manifest_digest(
            payload["grg2bzg_sha256"],
            name="grg2bzg_sha256",
        ),
        bzg2grg_sha256=_manifest_digest(
            payload["bzg2grg_sha256"],
            name="bzg2grg_sha256",
        ),
        degeneracy_tolerance_thz=degeneracy,
        selection_algorithm=selection_identity[0],
        selection_algorithm_version=selection_identity[1],
        robust_nonzero_floor_ev2=floor,
        cover_all_target_grid_points=observed_coverage[0],
        cover_all_target_bands=observed_coverage[1],
        required_processes=tuple(raw_processes),
        required_strata=tuple(raw_strata),
        empty_strata=tuple(raw_empty_strata),
        minimum_promoted_blocks=minimum_blocks,
        samples=tuple(samples),
    )


def _degenerate_subspace(
    frequencies: np.ndarray,
    band: int,
    tolerance: float,
) -> tuple[int, ...]:
    if band < 0 or band >= frequencies.size:
        raise ValueError("A sample band is outside the reference spectrum.")
    groups: list[list[int]] = []
    used: set[int] = set()
    for first in range(frequencies.size):
        if first in used:
            continue
        group = [first]
        used.add(first)
        for candidate in range(first + 1, frequencies.size):
            if any(
                abs(frequencies[index] - frequencies[candidate]) < tolerance
                for index in group
            ):
                group.append(candidate)
                used.add(candidate)
        groups.append(group)
    for group in groups:
        if band in group:
            return tuple(group)
    raise AssertionError("Every band belongs to a degeneracy subspace.")


def _validate_sample_manifest_identity(
    manifest: V3InteractionSampleManifest,
    *,
    reference: Any,
    input_records: Mapping[str, Mapping[str, Any]],
    fc2_mapping: Any,
    fc3_mapping: Any,
) -> None:
    expected_inputs = {
        "reference_hdf5_sha256": input_records["reference_hdf5"]["sha256"],
        "fc2_compact_hdf5_sha256": (
            input_records["fc2_compact_hdf5"]["sha256"]
        ),
        "fc2_full_hdf5_sha256": input_records["fc2_full_hdf5"]["sha256"],
        "fc3_compact_hdf5_sha256": (
            input_records["fc3_compact_hdf5"]["sha256"]
        ),
        "fc3_full_hdf5_sha256": input_records["fc3_full_hdf5"]["sha256"],
        "phono3py_yaml_sha256": input_records["phono3py_yaml"]["sha256"],
        "input_manifest_sha256": input_records["input_manifest"]["sha256"],
    }
    if dict(manifest.inputs) != expected_inputs:
        raise ValueError(
            "The V3 sample manifest input hashes do not identify this run."
        )
    expected_mapping_hashes = {
        "fc2": crystal_mapping_sha256(fc2_mapping),
        "fc3": crystal_mapping_sha256(fc3_mapping),
    }
    if dict(manifest.mapping_sha256) != expected_mapping_hashes:
        raise ValueError(
            "The V3 sample manifest mapping hashes do not identify the loaded "
            "crystal mappings."
        )
    if manifest.mesh != tuple(reference.mesh):
        raise ValueError("The V3 sample manifest mesh differs from reference.")
    if manifest.grg2bzg_sha256 != array_sha256(reference.grg2bzg):
        raise ValueError("The V3 manifest GRG-to-BZG map hash differs.")
    if manifest.bzg2grg_sha256 != array_sha256(reference.bzg2grg):
        raise ValueError("The V3 manifest BZG-to-GRG map hash differs.")
    if manifest.degeneracy_tolerance_thz != 1.0e-4:
        raise ValueError(
            "The pinned comparator requires degeneracy_tolerance_thz=1e-4."
        )

    target_set = set(int(value) for value in reference.grid_points_grg)
    target_band_set = set(int(value) for value in reference.band_indices)
    frequencies = reference.frequencies_in_grg_order()
    mesh = UniformBZMesh(reference.mesh)
    observed_targets: set[int] = set()
    observed_target_bands: set[int] = set()
    for sample in manifest.samples:
        if sample.q0_grg not in target_set:
            raise ValueError("A V3 sample q0 is not an archived target.")
        if sample.q1_grg >= reference.n_qpoints:
            raise ValueError("A V3 sample q1 is outside the mesh.")
        if sample.b0 not in target_band_set:
            raise ValueError("A V3 sample b0 is not an archived target band.")
        route = mesh.triplets_at_q(sample.q0_grg)[sample.q1_grg]
        if sample.expected_q2_grg != route.q2_index:
            raise ValueError("A V3 sample expected_q2_grg is inconsistent.")
        expected_subspaces = (
            _degenerate_subspace(
                frequencies[sample.q0_grg],
                sample.b0,
                manifest.degeneracy_tolerance_thz,
            ),
            _degenerate_subspace(
                frequencies[sample.q1_grg],
                sample.b1,
                manifest.degeneracy_tolerance_thz,
            ),
            _degenerate_subspace(
                frequencies[sample.expected_q2_grg],
                sample.b2,
                manifest.degeneracy_tolerance_thz,
            ),
        )
        if expected_subspaces != (
            sample.target_subspace,
            sample.band1_subspace,
            sample.band2_subspace,
        ):
            raise ValueError(
                "A V3 sample promoted subspace differs from the pinned "
                "degeneracy rule."
            )
        observed_targets.add(sample.q0_grg)
        observed_target_bands.add(sample.b0)
    if manifest.cover_all_target_grid_points and observed_targets != target_set:
        raise ValueError("The V3 sample manifest does not cover every target q.")
    if manifest.cover_all_target_bands and (
        observed_target_bands != target_band_set
    ):
        raise ValueError(
            "The V3 sample manifest does not cover every target band."
        )


def _validate_sample_manifest_results(
    manifest: V3InteractionSampleManifest,
    *,
    interactions: Any,
    comparison: Any,
) -> None:
    candidate_target_positions = {
        int(value): position
        for position, value in enumerate(interactions.grid_points)
    }
    comparison_blocks = {
        (
            block.target_grid_point,
            block.q1_grid_point,
            block.target_bands,
            block.bands1,
            block.bands2,
        ): block
        for block in comparison.interactions.blocks
    }
    if len(comparison_blocks) != len(comparison.interactions.blocks):
        raise ValueError("The comparator produced duplicate promoted blocks.")
    if set(comparison_blocks) != {
        sample.promoted_block_key for sample in manifest.samples
    }:
        raise ValueError(
            "Comparator promoted blocks differ from the V3 sample manifest."
        )
    for sample in manifest.samples:
        target_position = candidate_target_positions[sample.q0_grg]
        routes = {
            int(value): position
            for position, value in enumerate(
                interactions.q1_indices[target_position]
            )
        }
        route_position = routes[sample.q1_grg]
        actual_q2 = int(
            interactions.q2_indices[target_position, route_position]
        )
        if actual_q2 != sample.expected_q2_grg:
            raise ValueError("Candidate q2 routing differs from V3 manifest.")
        actual_process = (
            "normal"
            if bool(interactions.normal_mask[target_position, route_position])
            else "umklapp"
        )
        if actual_process != sample.process:
            raise ValueError(
                "Candidate normal/Umklapp classification differs from V3 "
                "sample manifest."
            )
        block = comparison_blocks[sample.promoted_block_key]
        allowed = max(
            1.0e-20,
            manifest.robust_nonzero_floor_ev2 * 1.0e-8,
            abs(sample.reference_invariant_block_sum_ev2) * 1.0e-12,
        )
        if abs(
            block.reference_sum_ev2
            - sample.reference_invariant_block_sum_ev2
        ) > allowed:
            raise ValueError(
                "A reference invariant block sum differs from the pinned V3 "
                "sample manifest."
            )


def _representation_error(
    compact: np.ndarray,
    full: np.ndarray,
    *,
    rtol: float,
    atol: float,
) -> dict[str, Any]:
    if compact.shape != full.shape:
        return {
            "passed": False,
            "shape_equal": False,
            "compact_shape": list(compact.shape),
            "full_shape": list(full.shape),
            "rtol": rtol,
            "atol": atol,
            "maximum_absolute_error": None,
            "maximum_scaled_error": None,
        }
    difference = np.abs(compact - full)
    scale = np.maximum(np.maximum(np.abs(compact), np.abs(full)), 1.0)
    maximum_absolute = float(np.max(difference))
    maximum_scaled = float(np.max(difference / scale))
    return {
        "passed": bool(np.all(difference <= atol + rtol * scale)),
        "shape_equal": True,
        "compact_shape": list(compact.shape),
        "full_shape": list(full.shape),
        "rtol": rtol,
        "atol": atol,
        "maximum_absolute_error": maximum_absolute,
        "maximum_scaled_error": maximum_scaled,
    }


def _deterministic_fc3_triplets(
    reference: Any,
    sample_manifest: V3InteractionSampleManifest,
) -> tuple[tuple[int, int, int], ...]:
    mesh = UniformBZMesh(reference.mesh)
    n_q = mesh.n_qpoints
    q1_seeds = sorted(
        {
            0,
            n_q // 4,
            n_q // 2,
            (3 * n_q) // 4,
            n_q - 1,
        }
    )
    triplets: list[tuple[int, int, int]] = []
    for q0 in sorted(int(value) for value in reference.grid_points_grg):
        routes = mesh.triplets_at_q(q0)
        triplets.extend(routes[q1].indices for q1 in q1_seeds)
    triplets.extend(
        (
            sample.q0_grg,
            sample.q1_grg,
            sample.expected_q2_grg,
        )
        for sample in sample_manifest.samples
    )
    return tuple(dict.fromkeys(triplets))


def _compare_fc3_representations(
    compact: Any,
    full: Any,
    reference: Any,
    sample_manifest: V3InteractionSampleManifest,
) -> dict[str, Any]:
    mesh = UniformBZMesh(reference.mesh)
    triplets = _deterministic_fc3_triplets(reference, sample_manifest)
    compact_values: list[np.ndarray] = []
    full_values: list[np.ndarray] = []
    for indices in triplets:
        qpoints = np.array(
            [
                [
                    component / period
                    for component, period in zip(
                        mesh.address_from_index(index),
                        mesh.shape,
                        strict=True,
                    )
                ]
                for index in indices
            ],
            dtype=np.float64,
        )
        compact_values.append(compact.fourier(qpoints))
        full_values.append(full.fourier(qpoints))
    comparison = _representation_error(
        np.stack(compact_values),
        np.stack(full_values),
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    comparison.update(
        {
            "quantity": "raw reciprocal FC3 tensor",
            "triplet_indices_grg": [list(item) for item in triplets],
            "triplet_selection": (
                "each archived q0 crossed with sorted unique q1 indices "
                "{0,Nq//4,Nq//2,3*Nq//4,Nq-1}, plus every promoted V3 "
                "sample-manifest triplet"
            ),
            "make_r0_average": True,
        }
    )
    return comparison


def _jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return {
                "real": value.real.tolist(),
                "imag": value.imag.tolist(),
            }
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
        }
    return value


def _fc3_validation_summary(report: Any) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "mapping_valid": report.mapping_valid,
        "schedule_complete": report.schedule_complete,
        "gamma_finite": report.gamma_finite,
        "asr_valid": report.asr_valid,
        "asr_valid_by_axis": list(report.asr_valid_by_axis),
        "gamma_asr_max_abs": report.gamma_asr_max_abs.tolist(),
        "gamma_asr_relative": report.gamma_asr_relative.tolist(),
        "gamma_scale": report.gamma_scale,
        "asr_threshold": report.asr_threshold,
        "support_valid": report.support_valid,
        "pairwise_atom_axes": [
            list(pair) for pair in report.pairwise_atom_axes
        ],
        "pairwise_minimum_distances": (
            report.pairwise_minimum_distances.tolist()
        ),
        "pairwise_maximum_distances": (
            report.pairwise_maximum_distances.tolist()
        ),
        "maximum_pairwise_distance": report.maximum_pairwise_distance,
        "distance_unit": report.distance_unit,
        "support_policy": report.support_policy,
        "nonzero_expanded_term_count": report.nonzero_expanded_term_count,
        "exact_zero_expanded_term_count": report.exact_zero_expanded_term_count,
        "nonzero_raw_block_count": report.nonzero_raw_block_count,
        "exact_zero_raw_block_count": report.exact_zero_raw_block_count,
        "make_r0_average": report.make_r0_average,
        "phase_convention": report.phase_convention,
        "anchor_convention": report.anchor_convention,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "metrics": dict(report.metrics),
    }


def write_json_atomic_exclusive(
    path: str | os.PathLike[str],
    value: Mapping[str, Any],
) -> str:
    """Atomically publish canonical JSON and return its SHA256.

    The final path is created with a hard link, so an existing report is never
    replaced even if another process wins a publication race.
    """

    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(payload).hexdigest()


def run_silicon_oracle_validation(
    config: SiliconOracleValidationConfig,
) -> dict[str, Any]:
    """Execute one complete bounded serial validation and publish its report."""

    if not isinstance(config, SiliconOracleValidationConfig):
        raise TypeError("config must be SiliconOracleValidationConfig.")
    started_utc = _utc_now()
    started = time.perf_counter()
    inputs_before = _input_records(config)
    if (
        inputs_before["environment_lock"]["sha256"]
        != PINNED_SILICON_ENVIRONMENT_LOCK_SHA256
    ):
        raise RuntimeError(
            "environment_lock_path does not match the pinned silicon-oracle "
            "environment lock."
        )
    _verify_generated_checksums(config, inputs_before)
    manifest = _manifest_identity(
        config.input_manifest_path,
        str(inputs_before["phono3py_yaml"]["sha256"]),
    )
    runtime = _runtime_identity()
    repository = _repository_identity(
        config.repository_path,
        require_clean=bool(config.require_clean_source),
    )
    imported_code = _code_identity(repository, runtime)
    generation_environment = _generation_environment_identity(
        config,
        repository,
    )

    stage_times: dict[str, float] = {}
    stage_start = time.perf_counter()
    reference = read_phono3py_silicon_reference(
        config.reference_path,
        expected_sha256=str(inputs_before["reference_hdf5"]["sha256"]),
        require_checksum_sidecar=False,
    )
    _validate_bounded_reference(reference)
    sample_manifest = read_interaction_sample_manifest(
        config.interaction_samples_path
    )
    stage_times["read_reference"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    fc2_mapping = read_phono3py_crystal_mapping(
        config.phono3py_yaml_path,
        purpose="fc2",
        is_nac=False,
    )
    fc3_mapping = read_phono3py_crystal_mapping(
        config.phono3py_yaml_path,
        purpose="fc3",
        is_nac=False,
    )
    fc2_compact = read_phono3py_fc2(
        config.fc2_compact_path,
        fc2_mapping.supercell.masses,
        p2s_map=fc2_mapping.p2s,
        s2p_map=fc2_mapping.s2p,
        compact=True,
    )
    fc2_full = read_phono3py_fc2(
        config.fc2_full_path,
        fc2_mapping.supercell.masses,
        p2s_map=fc2_mapping.p2s,
        s2p_map=fc2_mapping.s2p,
        compact=False,
    )
    fc3_compact_raw = read_phono3py_fc3(
        config.fc3_compact_path,
        fc3_mapping.supercell.masses,
        p2s_map=fc3_mapping.p2s,
        s2p_map=fc3_mapping.s2p,
        compact=True,
    )
    fc3_full_raw = read_phono3py_fc3(
        config.fc3_full_path,
        fc3_mapping.supercell.masses,
        p2s_map=fc3_mapping.p2s,
        s2p_map=fc3_mapping.s2p,
        compact=False,
    )
    compact_translational_fc3 = convert_fc3_to_translational(
        fc3_compact_raw,
        fc3_mapping,
        make_r0_average=True,
    )
    full_translational_fc3 = convert_fc3_to_translational(
        fc3_full_raw,
        fc3_mapping,
        make_r0_average=True,
    )
    compact_fc3_validation = validate_translational_fc3(
        compact_translational_fc3,
        fc3_mapping,
        asr_atol=config.asr_atol,
        asr_rtol=config.asr_rtol,
    )
    full_fc3_validation = validate_translational_fc3(
        full_translational_fc3,
        fc3_mapping,
        asr_atol=config.asr_atol,
        asr_rtol=config.asr_rtol,
    )
    _validate_sample_manifest_identity(
        sample_manifest,
        reference=reference,
        input_records=inputs_before,
        fc2_mapping=fc2_mapping,
        fc3_mapping=fc3_mapping,
    )
    stage_times["ingest_map_validate"] = time.perf_counter() - stage_start

    stage_start = time.perf_counter()
    compact_fc2_spectrum = compute_bulk_phonons(
        fc2_compact,
        fc2_mapping,
        reference.fc2_random_qpoints,
        hermitianize=bool(reference.fc2_hermitianized),
    )
    full_fc2_spectrum = compute_bulk_phonons(
        fc2_full,
        fc2_mapping,
        reference.fc2_random_qpoints,
        hermitianize=bool(reference.fc2_hermitianized),
    )
    fc2_representation = _representation_error(
        compact_fc2_spectrum.dynamical_matrices,
        full_fc2_spectrum.dynamical_matrices,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    fc2_representation["quantity"] = "random-q dynamical matrices"
    fc3_representation = _compare_fc3_representations(
        compact_translational_fc3,
        full_translational_fc3,
        reference,
        sample_manifest,
    )
    stage_times["full_compact_equality"] = (
        time.perf_counter() - stage_start
    )

    problem = BulkThreePhononProblem(
        fc2=fc2_compact,
        fc2_mapping=fc2_mapping,
        fc3=compact_translational_fc3,
        fc3_mapping=fc3_mapping,
        mesh=UniformBZMesh(reference.mesh),
        sigma_thz=float(reference.sigmas_thz[0]),
        cutoff_frequency_thz=reference.cutoff_frequency_thz,
        degenerate_tolerance_thz=(
            sample_manifest.degeneracy_tolerance_thz
        ),
        metadata={
            "validation_driver": SILICON_ORACLE_REPORT_SCHEMA,
            "reference_sha256": reference.sha256,
            "fc2_compact_sha256": (
                inputs_before["fc2_compact_hdf5"]["sha256"]
            ),
            "fc2_full_sha256": inputs_before["fc2_full_hdf5"]["sha256"],
            "fc3_compact_sha256": (
                inputs_before["fc3_compact_hdf5"]["sha256"]
            ),
            "fc3_full_sha256": inputs_before["fc3_full_hdf5"]["sha256"],
            "source_commit": manifest["source_commit"],
            "no_fitted_prefactors": True,
            "internal_q_parallelism": "none",
        },
    )

    stage_start = time.perf_counter()
    linewidths = compute_three_phonon_linewidths(
        problem,
        temperatures=reference.temperatures_k,
        grid_points=reference.grid_points_grg,
        band_indices=reference.band_indices,
    )
    stage_times["interactions_phase_linewidths"] = (
        time.perf_counter() - stage_start
    )
    interactions = linewidths.interactions

    stage_start = time.perf_counter()
    comparison = compare_silicon_reference(
        reference,
        fc2_spectrum=compact_fc2_spectrum,
        interactions=interactions,
        linewidths=linewidths,
        interaction_samples=sample_manifest.comparator_samples,
    )
    _validate_sample_manifest_results(
        sample_manifest,
        interactions=interactions,
        comparison=comparison,
    )
    stage_times["strict_comparison"] = time.perf_counter() - stage_start

    inputs_after = _input_records(config)
    _verify_inputs_unchanged(inputs_before, inputs_after)
    _verify_generated_checksums(config, inputs_after)
    representative_nonzero_sample = bool(
        comparison.interactions.blocks
        and any(
            block.reference_sum_ev2
            > comparison.interactions.absolute_tolerance_ev2
            for block in comparison.interactions.blocks
        )
    )
    scientific_passed = bool(
        compact_fc3_validation.passed
        and full_fc3_validation.passed
        and fc2_representation["passed"]
        and fc3_representation["passed"]
        and comparison.passed
        and representative_nonzero_sample
    )
    source_eligible = bool(repository["clean"])
    controls_exact = bool(
        config.asr_atol == 1.0e-10 and config.asr_rtol == 1.0e-8
    )
    completed_utc = _utc_now()
    report: dict[str, Any] = {
        "schema": SILICON_ORACLE_REPORT_SCHEMA,
        "schema_version": 1,
        "status": "passed" if scientific_passed else "failed",
        "validation_passed": scientific_passed,
        "certification_eligible": (
            scientific_passed
            and source_eligible
            and controls_exact
            and imported_code["passed"]
            and generation_environment["passed"]
        ),
        "certification_gates": {
            "scientific_comparison": scientific_passed,
            "source_repository_clean": source_eligible,
            "controls_exact": controls_exact,
            "imported_code_matches_repository": imported_code["passed"],
            "reference_generation_authenticated": (
                generation_environment["passed"]
            ),
            "full_compact_equality": bool(
                fc2_representation["passed"]
                and fc3_representation["passed"]
            ),
        },
        "started_at_utc": started_utc,
        "completed_at_utc": completed_utc,
        "wall_time_seconds": time.perf_counter() - started,
        "stage_times_seconds": stage_times,
        "execution": {
            "method": "deterministic_serial_complete_target_q",
            "target_q_task_parallelism": "none",
            "internal_q_parallelism": "none",
            "floating_point_mpi_reduction": False,
            "fitted_prefactors": False,
            "make_r0_average": True,
            "production_representation": "compact_after_full_equality",
        },
        "controls": {
            "asr_atol": config.asr_atol,
            "asr_rtol": config.asr_rtol,
            "certification_values": {
                "asr_atol": 1.0e-10,
                "asr_rtol": 1.0e-8,
            },
            "controls_exact": controls_exact,
        },
        "bounds": {
            "maximum_mesh_points": _MAX_MESH_POINTS,
            "maximum_target_grid_points": _MAX_TARGET_GRID_POINTS,
            "maximum_target_bands": _MAX_TARGET_BANDS,
            "maximum_temperatures": _MAX_TEMPERATURES,
            "maximum_interaction_samples": _MAX_INTERACTION_SAMPLES,
            "minimum_promoted_interaction_blocks": _MIN_PROMOTED_BLOCKS,
        },
        "selections": {
            "mesh": list(reference.mesh),
            "grid_points_grg": reference.grid_points_grg.tolist(),
            "grid_points_bzg": reference.grid_points_bzg.tolist(),
            "band_indices": reference.band_indices.tolist(),
            "temperatures_k": reference.temperatures_k.tolist(),
            "sigma_thz": float(reference.sigmas_thz[0]),
            "cutoff_frequency_thz": reference.cutoff_frequency_thz,
            "interaction_samples": [
                _jsonable(sample) for sample in sample_manifest.samples
            ],
            "interaction_sample_manifest_sha256": sample_manifest.sha256,
            "selection_source": "immutable_reference_hdf5",
        },
        "inputs": inputs_before,
        "source_manifest": manifest,
        "source_repository": repository,
        "imported_code_provenance": imported_code,
        "reference_generation_environment": generation_environment,
        "runtime": runtime,
        "full_compact_equality": {
            "passed": bool(
                fc2_representation["passed"]
                and fc3_representation["passed"]
            ),
            "fc2": fc2_representation,
            "fc3": fc3_representation,
        },
        "translational_fc3_validation": {
            "compact": _fc3_validation_summary(compact_fc3_validation),
            "full": _fc3_validation_summary(full_fc3_validation),
        },
        "comparison": _jsonable(comparison),
        "interaction_sample_manifest": _jsonable(sample_manifest),
        "interaction_sample_coverage": {
            "all_reference_targets_covered": True,
            "all_reference_target_bands_covered": True,
            "representative_nonzero_block_present": (
                representative_nonzero_sample
            ),
            "nonzero_definition": (
                "reference degenerate-subspace sum exceeds the comparator "
                "absolute tolerance"
            ),
        },
        "result_summary": {
            "fc2_layout": "compact",
            "fc3_layout": compact_translational_fc3.source_layout,
            "fc3_term_count": len(compact_translational_fc3.terms),
            "full_fc3_term_count": len(full_translational_fc3.terms),
            "all_shortest_true_count": int(
                np.count_nonzero(compact_translational_fc3.all_shortest)
            ),
            "interaction_strength_min_ev2": float(
                np.min(interactions.interaction_strengths)
            ),
            "interaction_strength_max_ev2": float(
                np.max(interactions.interaction_strengths)
            ),
            "gamma_min_thz_hwhm": float(
                np.min(linewidths.gamma_thz_hwhm)
            ),
            "gamma_max_thz_hwhm": float(
                np.max(linewidths.gamma_thz_hwhm)
            ),
            "decay_phase_space_max": float(
                np.max(np.abs(linewidths.decay_phase_space))
            ),
            "difference_phase_space_max_abs": float(
                np.max(np.abs(linewidths.difference_phase_space))
            ),
            "gamma_semantics": "cyclic-frequency HWHM",
            "negf_linewidth_semantics": "angular-frequency FWHM",
            "cyclic_hwhm_to_angular_fwhm_factor": "4*pi",
        },
    }
    report_sha256 = write_json_atomic_exclusive(config.output_path, report)
    report["report_artifact"] = {
        "path": str(config.output_path),
        "sha256": report_sha256,
    }
    return report


__all__ = [
    "PINNED_PHONO3PY_VERSION",
    "PINNED_PHONOPY_VERSION",
    "PINNED_SILICON_DATASET_ID",
    "PINNED_SILICON_ENVIRONMENT_LOCK_SHA256",
    "PINNED_SILICON_REPOSITORY",
    "PINNED_SILICON_SOURCE_COMMIT",
    "PINNED_SPGLIB_VERSION",
    "SILICON_ORACLE_REPORT_SCHEMA",
    "SiliconOracleValidationConfig",
    "V3InteractionSampleManifest",
    "V3InteractionSampleRecord",
    "V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM",
    "V3_BOUNDED_SAMPLE_SELECTION_ALGORITHM_VERSION",
    "V3_SAMPLE_MANIFEST_SCHEMA",
    "V3_SAMPLE_SELECTION_ALGORITHM",
    "V3_SAMPLE_SELECTION_ALGORITHM_VERSION",
    "array_sha256",
    "crystal_mapping_sha256",
    "read_interaction_sample_manifest",
    "run_silicon_oracle_validation",
    "write_json_atomic_exclusive",
]
