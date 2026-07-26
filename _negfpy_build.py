"""Setuptools hook for fail-closed wheel provenance.

This module is build support, not part of the installed ``negfpy`` package.
It deliberately derives repository identity from its own location instead of
the invoking process's working directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Final

from setuptools.command.build_py import build_py as _build_py


_SCHEMA: Final = "negfpy.build_provenance.v1"
_METADATA_NAME: Final = "_build_provenance.json"
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _package_source_sha256(package_root: Path) -> str:
    """Hash the exact Python source bytes copied into a wheel."""

    paths = sorted(
        (path for path in package_root.rglob("*.py") if path.is_file()),
        key=lambda path: path.relative_to(package_root).as_posix(),
    )
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("symlinked Python package sources are not supported")
    if not paths:
        raise RuntimeError(
            f"no Python package sources found below {package_root}"
        )
    digest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_state(project_root: Path) -> tuple[str, bool | str]:
    """Return exact-root Git identity or a deliberately uncertified state."""

    marker = project_root / ".git"
    git = shutil.which("git")
    if not marker.exists() or git is None:
        return "unknown", "unknown"
    try:
        top = subprocess.run(
            (git, "rev-parse", "--show-toplevel"),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip()
        if Path(top).resolve() != project_root.resolve():
            return "unknown", "unknown"
        revision = subprocess.run(
            (git, "rev-parse", "--verify", "HEAD"),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout.strip().lower()
        if _GIT_OBJECT.fullmatch(revision) is None:
            return "unknown", "unknown"
        status = subprocess.run(
            (
                git,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        ).stdout
    except (OSError, subprocess.SubprocessError, ValueError):
        return "unknown", "unknown"
    return revision, bool(status.strip())


def build_provenance_record(
    *,
    project_root: Path,
    package_root: Path,
    package_version: str,
) -> dict[str, object]:
    """Create the canonical record for the copied package source."""

    revision, dirty = _git_state(project_root.resolve())
    return {
        "git_dirty": dirty,
        "git_revision": revision,
        "package_source_sha256": _package_source_sha256(
            package_root.resolve()
        ),
        "package_version": package_version,
        "project": "negfpy",
        "schema": _SCHEMA,
    }


class ProvenanceBuildPy(_build_py):
    """Write build identity after setuptools has copied package sources."""

    def run(self) -> None:
        super().run()
        project_root = Path(__file__).resolve().parent
        package_root = Path(self.build_lib).resolve() / "negfpy"
        record = build_provenance_record(
            project_root=project_root,
            package_root=package_root,
            package_version=self.distribution.get_version(),
        )
        target = package_root / _METADATA_NAME
        target.write_text(
            json.dumps(
                record,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n",
            encoding="ascii",
        )

    def get_outputs(self, include_bytecode: bool = True) -> list[str]:
        outputs = list(super().get_outputs(include_bytecode=include_bytecode))
        metadata = (
            Path(self.build_lib).resolve() / "negfpy" / _METADATA_NAME
        )
        if metadata.is_file() and str(metadata) not in outputs:
            outputs.append(str(metadata))
        return outputs


__all__ = ["ProvenanceBuildPy", "build_provenance_record"]
