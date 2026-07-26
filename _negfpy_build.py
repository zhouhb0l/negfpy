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
import stat
import subprocess
from typing import Final

from setuptools.command.build_py import build_py as _build_py


_SCHEMA: Final = "negfpy.build_provenance.v1"
_METADATA_NAME: Final = "_build_provenance.json"
_HASH_DOMAIN: Final = b"negfpy.package-payload.v1\0"
_GIT_OBJECT = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")


def _payload_entries(package_root: Path) -> tuple[tuple[str, bytes], ...]:
    """Read every behavior-bearing package file with strict path semantics."""

    root = package_root.resolve()
    if package_root.is_symlink() or not root.is_dir():
        raise RuntimeError(f"package root is not a regular directory: {root}")
    entries: list[tuple[str, bytes]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root)
        if (
            relative.as_posix() == _METADATA_NAME
            or "__pycache__" in relative.parts
            or relative.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.is_symlink():
            raise RuntimeError(f"symlinked package payload is forbidden: {path}")
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise RuntimeError(f"special package payload is forbidden: {path}")
        entries.append((relative.as_posix(), path.read_bytes()))
    if not entries:
        raise RuntimeError(f"no package payload found below {root}")
    return tuple(entries)


def _payload_sha256(entries: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(len(entries).to_bytes(8, "big"))
    for relative, content in entries:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _package_source_sha256(package_root: Path) -> str:
    """Hash framed paths and bytes for the complete package payload."""

    return _payload_sha256(_payload_entries(package_root))


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


def _head_package_entries(
    project_root: Path,
) -> tuple[tuple[str, bytes], ...] | None:
    """Read exact regular package blobs from HEAD, independent of index flags."""

    git = shutil.which("git")
    if git is None:
        return None
    try:
        listing = subprocess.run(
            (git, "ls-tree", "-rz", "--full-tree", "HEAD", "--", "src/negfpy"),
            cwd=project_root,
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
        entries: list[tuple[str, bytes]] = []
        prefix = "src/negfpy/"
        for raw in listing.split(b"\0"):
            if not raw:
                continue
            header, separator, raw_path = raw.partition(b"\t")
            mode, kind, object_id = header.split(b" ")
            path = raw_path.decode("utf-8")
            if (
                not separator
                or mode not in {b"100644", b"100755"}
                or kind != b"blob"
                or not path.startswith(prefix)
            ):
                return None
            content = subprocess.run(
                (git, "cat-file", "blob", object_id.decode("ascii")),
                cwd=project_root,
                check=True,
                capture_output=True,
                timeout=10.0,
            ).stdout
            entries.append((path[len(prefix) :], content))
    except (
        OSError,
        subprocess.SubprocessError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None
    entries.sort(key=lambda item: item[0])
    return tuple(entries) if entries else None


def build_provenance_record(
    *,
    project_root: Path,
    package_root: Path,
    package_version: str,
) -> dict[str, object]:
    """Create the canonical record for the copied package payload."""

    project = project_root.resolve()
    build_entries = _payload_entries(package_root)
    build_hash = _payload_sha256(build_entries)
    revision, dirty = _git_state(project)
    source_root = project / "src" / "negfpy"
    try:
        source_entries = _payload_entries(source_root)
    except (OSError, RuntimeError, ValueError):
        source_entries = ()
    head_entries = _head_package_entries(project)
    if type(dirty) is bool and (
        not source_entries
        or head_entries is None
        or source_entries != head_entries
        or build_entries != source_entries
    ):
        dirty = True
    if dirty == "unknown":
        revision = "unknown"
    return {
        "git_dirty": dirty,
        "git_revision": revision,
        "package_source_sha256": build_hash,
        "package_version": package_version,
        "project": "negfpy",
        "schema": _SCHEMA,
    }


def _fresh_package_root(
    build_lib: str | Path,
    *,
    project_root: Path,
) -> Path:
    """Remove only a generated package tree disjoint from canonical source."""

    project = project_root.resolve()
    source_root = (project / "src" / "negfpy").resolve()
    build_root = Path(build_lib).resolve()
    target = build_root / "negfpy"
    unsafe_root = (
        build_root == project
        or build_root in source_root.parents
        or build_root.is_relative_to(source_root)
    )
    overlaps_source = (
        target == source_root
        or target.is_relative_to(source_root)
        or source_root.is_relative_to(target)
    )
    if (
        unsafe_root
        or overlaps_source
        or target.parent != build_root
        or target.name != "negfpy"
    ):
        raise RuntimeError(f"unsafe build package target: {target}")
    if target.is_symlink():
        raise RuntimeError(f"refusing symlinked build package target: {target}")
    if target.exists():
        if not target.is_dir():
            raise RuntimeError(f"build package target is not a directory: {target}")
        shutil.rmtree(target)
    return target


class ProvenanceBuildPy(_build_py):
    """Write build identity after a fresh setuptools package copy."""

    def run(self) -> None:
        project_root = Path(__file__).resolve().parent
        package_root = _fresh_package_root(
            self.build_lib,
            project_root=project_root,
        )
        super().run()
        if package_root.is_symlink() or not package_root.is_dir():
            raise RuntimeError(
                f"setuptools did not create the expected package: {package_root}"
            )
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
