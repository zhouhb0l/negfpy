"""Deterministic provenance records for anharmonic calculations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import fields
import hashlib
from importlib import metadata as importlib_metadata
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

import numpy as np

from .types import CubicSCBAProblem, KeldyshSpectrum, SCBAConfig


Array = np.ndarray


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


def code_provenance() -> dict[str, str | bool]:
    """Return package version, source-tree hash, and Git state when available."""

    try:
        installed_version = importlib_metadata.version("negfpy")
    except importlib_metadata.PackageNotFoundError:
        installed_version = "unknown"

    package_root = Path(__file__).resolve().parents[1]
    declared_version = "unknown"
    project_file: Path | None = None
    for candidate in (package_root, *package_root.parents):
        possible = candidate / "pyproject.toml"
        if possible.is_file():
            project_file = possible
            break
    if project_file is not None:
        try:
            project_text = project_file.read_text(encoding="utf-8")
        except OSError:
            pass
        else:
            project_section = re.search(
                r"(?ms)^\[project\]\s*$"
                r"(?P<body>.*?)"
                r"(?=^\[[^\]]+\]\s*$|\Z)",
                project_text,
            )
            if project_section is not None:
                version_match = re.search(
                    r'(?m)^\s*version\s*=\s*["\']([^"\']+)["\']\s*$',
                    project_section.group("body"),
                )
                if version_match is not None:
                    declared_version = version_match.group(1)
    # Installed wheel metadata is generated from the project declaration.
    # Wheels intentionally omit the repository-level pyproject.toml, so use
    # that metadata as the authoritative declaration when no source project
    # file is present.  Exact-commit campaign certification separately
    # requires clean Git and external bundle/environment receipts.
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

    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(path.relative_to(package_root).as_posix().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            continue

    result: dict[str, str | bool] = {
        "package_version": package_version,
        "declared_package_version": declared_version,
        "installed_package_version": installed_version,
        "package_version_consistent": version_consistent,
        "source_tree_sha256": digest.hexdigest(),
        "git_revision": "unknown",
        "git_dirty": "unknown",
    }
    repository = package_root
    for candidate in (package_root, *package_root.parents):
        if (candidate / ".git").exists():
            repository = candidate
            break
    else:
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
    result["git_dirty"] = bool(dirty.strip())
    return result


__all__ = [
    "code_provenance",
    "configuration_hash",
    "initial_self_energy_hash",
    "problem_input_hashes",
]
