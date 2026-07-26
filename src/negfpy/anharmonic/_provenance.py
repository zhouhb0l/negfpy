"""Deterministic provenance records for anharmonic calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
import hashlib
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
from typing import Any

import numpy as np

from .types import CubicSCBAProblem, KeldyshSpectrum, SCBAConfig


Array = np.ndarray

_BUILD_PROVENANCE_SCHEMA = "negfpy.build_provenance.v1"
_BUILD_PROVENANCE_FILE = "_build_provenance.json"
_PACKAGE_HASH_DOMAIN = b"negfpy.package-payload.v1\0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_REVISION_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_BUILD_PROVENANCE_KEYS = frozenset(
    {
        "git_dirty",
        "git_revision",
        "package_source_sha256",
        "package_version",
        "project",
        "schema",
    }
)


def _package_payload_entries(
    package_root: Path,
) -> tuple[tuple[str, bytes], ...] | None:
    """Read the complete behavior-bearing package payload."""

    try:
        root = package_root.resolve()
        if package_root.is_symlink() or not root.is_dir():
            return None
        entries: list[tuple[str, bytes]] = []
        for path in sorted(
            root.rglob("*"),
            key=lambda item: item.relative_to(root).as_posix(),
        ):
            relative = path.relative_to(root)
            if (
                relative.as_posix() == _BUILD_PROVENANCE_FILE
                or "__pycache__" in relative.parts
                or relative.suffix in {".pyc", ".pyo"}
            ):
                continue
            if path.is_symlink():
                return None
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                continue
            if not stat.S_ISREG(mode):
                return None
            entries.append((relative.as_posix(), path.read_bytes()))
    except (OSError, RuntimeError, ValueError):
        return None
    return tuple(entries) if entries else None


def _payload_sha256(entries: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_PACKAGE_HASH_DOMAIN)
    digest.update(len(entries).to_bytes(8, "big"))
    for relative, content in entries:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _package_source_sha256(package_root: Path) -> str | None:
    """Hash the framed full installed payload, failing closed on anomalies."""

    entries = _package_payload_entries(package_root)
    return None if entries is None else _payload_sha256(entries)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> Any:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _validated_embedded_build_provenance(
    package_root: Path,
    *,
    package_source_sha256: str,
    installed_version: str,
) -> tuple[str, bool | str] | None:
    """Validate fixed-path wheel metadata against installed payload bytes."""

    path = package_root / _BUILD_PROVENANCE_FILE
    try:
        mode = path.lstat().st_mode
        if not stat.S_ISREG(mode) or path.is_symlink():
            return None
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw or len(raw) > 4096:
        return None
    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(document, dict) or set(document) != _BUILD_PROVENANCE_KEYS:
        return None
    if (
        document.get("schema") != _BUILD_PROVENANCE_SCHEMA
        or document.get("project") != "negfpy"
        or installed_version == "unknown"
        or document.get("package_version") != installed_version
        or _SHA256_RE.fullmatch(package_source_sha256) is None
        or document.get("package_source_sha256") != package_source_sha256
    ):
        return None
    revision = document.get("git_revision")
    dirty = document.get("git_dirty")
    if not isinstance(revision, str):
        return None
    if type(dirty) is bool:
        if _GIT_REVISION_RE.fullmatch(revision) is None:
            return None
    elif dirty == "unknown" and revision == "unknown":
        pass
    else:
        return None
    return revision, dirty


def _embedded_build_provenance_if_present(
    package_root: Path,
    *,
    package_source_sha256: str | None,
    installed_version: str,
) -> tuple[bool, tuple[str, bool | str] | None]:
    """Distinguish an absent wheel record from a present invalid record."""

    path = package_root / _BUILD_PROVENANCE_FILE
    try:
        path.lstat()
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None
    if package_source_sha256 is None:
        return True, None
    return True, _validated_embedded_build_provenance(
        package_root,
        package_source_sha256=package_source_sha256,
        installed_version=installed_version,
    )


def _update_array(digest: Any, values: Any) -> None:
    array = np.ascontiguousarray(np.asarray(values))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))


def _update_value(digest: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        digest.update(b"array:")
        _update_array(digest, value)
    elif isinstance(value, dict) or hasattr(value, "items"):
        digest.update(b"mapping:")
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            _update_value(digest, str(key))
            _update_value(digest, item)
    elif isinstance(value, (tuple, list)):
        digest.update(b"sequence:")
        for item in value:
            _update_value(digest, item)
    elif isinstance(value, (set, frozenset)):
        digest.update(b"set:")
        for item in sorted(value, key=repr):
            _update_value(digest, item)
    elif isinstance(value, np.generic):
        _update_value(digest, value.item())
    else:
        digest.update(type(value).__name__.encode("utf-8"))
        digest.update(b":")
        digest.update(repr(value).encode("utf-8"))


def _array_digest(arrays: Iterable[Any], *, extra: str = "") -> str:
    digest = hashlib.sha256()
    digest.update(extra.encode("utf-8"))
    for values in arrays:
        _update_array(digest, values)
    return digest.hexdigest()


def _indices_token(values: Any) -> str:
    if values is None:
        return "default"
    if isinstance(values, slice):
        return f"slice:{values.start}:{values.stop}:{values.step}"
    return repr(tuple(int(value) for value in values))


def problem_input_hashes(problem: CubicSCBAProblem) -> dict[str, str]:
    """Return SHA256 hashes for every numerical input family."""

    device_hash = _array_digest(
        (*problem.device.onsite_blocks, *problem.device.coupling_blocks),
        extra=(
            f"layers={problem.device.n_layers};"
            f"dof={problem.device.dof_per_layer};"
        ),
    )
    left_hash = _array_digest(
        (
            problem.left.lead.d00,
            problem.left.lead.d01,
            problem.left.lead.d10
            if problem.left.lead.d10 is not None
            else problem.left.lead.d01.conj().T,
            problem.left.resolved_device_to_lead("left"),
        ),
        extra=(
            f"T={problem.left.temperature!r};"
            f"indices={_indices_token(problem.left.device_indices)};"
        ),
    )
    right_hash = _array_digest(
        (
            problem.right.lead.d00,
            problem.right.lead.d01,
            problem.right.lead.d10
            if problem.right.lead.d10 is not None
            else problem.right.lead.d01.conj().T,
            problem.right.resolved_device_to_lead("right"),
        ),
        extra=(
            f"T={problem.right.temperature!r};"
            f"indices={_indices_token(problem.right.device_indices)};"
        ),
    )
    vertex_digest = hashlib.sha256()
    _update_array(vertex_digest, problem.vertex.values)
    _update_value(vertex_digest, problem.vertex.metadata)
    vertex_hash = vertex_digest.hexdigest()
    grid_hash = _array_digest(
        (problem.grid.values,),
        extra=f"step={problem.grid.step!r};n={problem.grid.n_positive};",
    )
    scale_digest = hashlib.sha256()
    _update_value(scale_digest, problem.scale.omega0)
    _update_value(scale_digest, problem.scale.hbar)
    _update_value(scale_digest, problem.scale.metadata)
    scale_hash = scale_digest.hexdigest()

    parts = {
        "device": device_hash,
        "left_reservoir": left_hash,
        "right_reservoir": right_hash,
        "cubic_vertex": vertex_hash,
        "frequency_grid": grid_hash,
        "solver_scale": scale_hash,
    }
    combined = hashlib.sha256()
    for name in sorted(parts):
        combined.update(name.encode("ascii"))
        combined.update(parts[name].encode("ascii"))
    parts["problem"] = combined.hexdigest()
    return parts


def configuration_hash(config: SCBAConfig) -> str:
    """Return a deterministic SHA256 over every SCBA configuration field."""

    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")
    digest = hashlib.sha256()
    for description in fields(config):
        _update_value(digest, description.name)
        _update_value(digest, getattr(config, description.name))
    return digest.hexdigest()


def initial_self_energy_hash(
    problem: CubicSCBAProblem,
    initial_sigma_pp: KeldyshSpectrum | None,
) -> str:
    """Hash the actual lesser/greater continuation state used by SCBA."""

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    digest = hashlib.sha256()
    expected = (problem.grid.n_points, problem.ndof, problem.ndof)
    digest.update(repr(expected).encode("ascii"))
    if initial_sigma_pp is None:
        digest.update(b"explicit_zero_initial_self_energy")
        return digest.hexdigest()
    if not isinstance(initial_sigma_pp, KeldyshSpectrum):
        raise TypeError("initial_sigma_pp must be a KeldyshSpectrum or None.")
    if initial_sigma_pp.shape != expected:
        raise ValueError(
            f"initial_sigma_pp must have shape {expected}; "
            f"got {initial_sigma_pp.shape}."
        )
    # The retarded input is intentionally ignored by solve_scba and rebuilt
    # from this pair, so provenance hashes exactly the state that is consumed.
    digest.update(b"supplied_keldysh_pair")
    _update_array(digest, initial_sigma_pp.lesser)
    _update_array(digest, initial_sigma_pp.greater)
    return digest.hexdigest()


def _head_package_entries(
    repository: Path,
    git_executable: str,
) -> tuple[tuple[str, bytes], ...] | None:
    """Read exact regular package blobs from HEAD, ignoring index flags."""

    try:
        listing = subprocess.run(
            (
                git_executable,
                "ls-tree",
                "-rz",
                "--full-tree",
                "HEAD",
                "--",
                "src/negfpy",
            ),
            cwd=repository,
            check=True,
            capture_output=True,
            timeout=5.0,
        ).stdout
        entries: list[tuple[str, bytes]] = []
        prefix = "src/negfpy/"
        for raw in listing.split(b"\0"):
            if not raw:
                continue
            header, separator, raw_path = raw.partition(b"\t")
            mode, kind, object_id = header.split(b" ")
            source_path = raw_path.decode("utf-8")
            if (
                not separator
                or mode not in {b"100644", b"100755"}
                or kind != b"blob"
                or not source_path.startswith(prefix)
            ):
                return None
            content = subprocess.run(
                (
                    git_executable,
                    "cat-file",
                    "blob",
                    object_id.decode("ascii"),
                ),
                cwd=repository,
                check=True,
                capture_output=True,
                timeout=5.0,
            ).stdout
            entries.append((source_path[len(prefix) :], content))
    except (
        OSError,
        subprocess.SubprocessError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None
    entries.sort(key=lambda item: item[0])
    return tuple(entries) if entries else None


def _source_repository(package_root: Path) -> Path | None:
    """Return only the repository owning the exact ``src/negfpy`` tree."""

    resolved = package_root.resolve()
    for candidate in (resolved, *resolved.parents):
        if not (candidate / ".git").exists():
            continue
        expected = candidate / "src" / "negfpy"
        try:
            if (
                not expected.is_symlink()
                and expected.resolve() == resolved
                and (candidate / "pyproject.toml").is_file()
            ):
                return candidate
        except OSError:
            return None
    return None


def _declared_package_version(project_root: Path) -> str:
    """Read the exact source checkout declaration without consulting cwd."""

    project_file = project_root / "pyproject.toml"
    try:
        project_text = project_file.read_text(encoding="utf-8")
    except OSError:
        return "unknown"
    project_section = re.search(
        r"(?ms)^\[project\][ \t]*(?:\r?\n)"
        r"(?P<body>.*?)"
        r"(?=^\[[^\]]+\][ \t]*(?:\r?\n)|\Z)",
        project_text,
    )
    if project_section is None:
        return "unknown"
    version_match = re.search(
        r"(?m)^\s*version\s*=\s*[\"\x27]([^\"\x27]+)"
        r"[\"\x27]\s*(?:\n|\Z)",
        project_section.group("body"),
    )
    return "unknown" if version_match is None else version_match.group(1)


def code_provenance() -> dict[str, str | bool]:
    """Return package version, source-tree hash, and Git state when available."""

    try:
        installed_version = importlib_metadata.version("negfpy")
    except importlib_metadata.PackageNotFoundError:
        installed_version = "unknown"

    package_root = Path(__file__).resolve().parents[1]
    package_source_sha256 = _package_source_sha256(package_root)
    repository = _source_repository(package_root)
    if repository is None:
        embedded_present, embedded = _embedded_build_provenance_if_present(
            package_root,
            package_source_sha256=package_source_sha256,
            installed_version=installed_version,
        )
    else:
        embedded_present, embedded = False, None
    declared_version = (
        "unknown"
        if repository is None
        else _declared_package_version(repository)
    )
    # Installed wheel metadata is generated from the project declaration.
    # Non-source installs use embedded metadata or remain unknown.
    if declared_version == "unknown" and installed_version != "unknown":
        declared_version = installed_version
    package_version = (
        declared_version
        if declared_version != "unknown"
        else installed_version
    )
    version_consistent = bool(
        installed_version != "unknown"
        and declared_version != "unknown"
        and installed_version == declared_version
    )

    result: dict[str, str | bool] = {
        "package_version": package_version,
        "declared_package_version": declared_version,
        "installed_package_version": installed_version,
        "package_version_consistent": version_consistent,
        "source_tree_sha256": package_source_sha256 or "unknown",
        "git_revision": "unknown",
        "git_dirty": "unknown",
    }
    if embedded_present:
        if embedded is not None:
            revision, dirty = embedded
            result["git_revision"] = revision
            result["git_dirty"] = dirty
        return result

    if repository is None:
        return result

    git_executable = shutil.which("git")
    if git_executable is None:
        executable_ancestors = Path(sys.executable).resolve().parents
        candidates = (
            *(ancestor / "bin" / "git" for ancestor in executable_ancestors),
            Path("/usr/bin/git"),
            Path("/usr/local/bin/git"),
        )
        for candidate in candidates:
            if candidate.is_file():
                git_executable = str(candidate)
                break

    revision = ""
    if git_executable is not None:
        try:
            revision = subprocess.run(
                (git_executable, "rev-parse", "HEAD"),
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=5.0,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            revision = ""
    if not revision:
        git_directory = repository / ".git"
        try:
            if git_directory.is_file():
                pointer = git_directory.read_text(encoding="utf-8").strip()
                if pointer.startswith("gitdir:"):
                    git_directory = (
                        repository / pointer.split(":", 1)[1].strip()
                    ).resolve()
            head = (git_directory / "HEAD").read_text(
                encoding="utf-8"
            ).strip()
            if head.startswith("ref:"):
                reference = head.split(":", 1)[1].strip()
                reference_file = git_directory / reference
                if reference_file.is_file():
                    revision = reference_file.read_text(
                        encoding="utf-8"
                    ).strip()
                else:
                    packed = git_directory / "packed-refs"
                    if packed.is_file():
                        for line in packed.read_text(
                            encoding="utf-8"
                        ).splitlines():
                            if line.startswith(("#", "^")):
                                continue
                            value, _, name = line.partition(" ")
                            if name == reference:
                                revision = value
                                break
            else:
                revision = head
        except OSError:
            revision = ""
    result["git_revision"] = revision or "unknown"

    if git_executable is None:
        return result
    package_pathspec = package_root.relative_to(repository).as_posix()
    pathspecs = [package_pathspec]
    if (repository / "pyproject.toml").is_file():
        pathspecs.append("pyproject.toml")
    try:
        dirty = subprocess.run(
            (
                git_executable,
                "status",
                "--porcelain",
                "--untracked-files=all",
                "--",
                *pathspecs,
            ),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5.0,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return result
    current_entries = _package_payload_entries(package_root)
    head_entries = _head_package_entries(repository, git_executable)
    if current_entries is None or head_entries is None:
        result["git_dirty"] = "unknown"
    else:
        result["git_dirty"] = bool(
            dirty.strip() or current_entries != head_entries
        )
    return result


__all__ = [
    "code_provenance",
    "configuration_hash",
    "initial_self_energy_hash",
    "problem_input_hashes",
]
