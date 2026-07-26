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
_BUILD_ROOT_SCHEMA: Final = "negfpy.build_root.v1"
_BUILD_ROOT_MARKER: Final = ".negfpy-build-root.json"
_BUILD_ROOT_MARKER_KEYS: Final = frozenset(
    {"build_root", "project", "project_root", "schema"}
)
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


def _regular_file_bytes(path: Path) -> bytes | None:
    """Read a fixed regular non-symlink file, or fail closed."""

    try:
        mode = path.lstat().st_mode
        if path.is_symlink() or not stat.S_ISREG(mode):
            return None
        return path.read_bytes()
    except OSError:
        return None


def _head_regular_blob(project_root: Path, relative: str) -> bytes | None:
    """Read one exact regular blob from HEAD, bypassing index flags."""

    git = shutil.which("git")
    if git is None:
        return None
    try:
        listing = subprocess.run(
            (git, "ls-tree", "-z", "--full-tree", "HEAD", "--", relative),
            cwd=project_root,
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
        records = tuple(item for item in listing.split(b"\0") if item)
        if len(records) != 1:
            return None
        header, separator, raw_path = records[0].partition(b"\t")
        mode, kind, object_id = header.split(b" ")
        if (
            not separator
            or raw_path.decode("utf-8") != relative
            or mode not in {b"100644", b"100755"}
            or kind != b"blob"
        ):
            return None
        return subprocess.run(
            (git, "cat-file", "blob", object_id.decode("ascii")),
            cwd=project_root,
            check=True,
            capture_output=True,
            timeout=10.0,
        ).stdout
    except (
        OSError,
        subprocess.SubprocessError,
        UnicodeDecodeError,
        ValueError,
    ):
        return None


def _declared_package_version(document: bytes | None) -> str | None:
    """Extract the static project version from exact pyproject bytes."""

    if document is None:
        return None
    try:
        project_text = document.decode("utf-8")
    except UnicodeDecodeError:
        return None
    project_section = re.search(
        r"(?ms)^\[project\][ \t]*(?:\r?\n)"
        r"(?P<body>.*?)"
        r"(?=^\[[^\]]+\][ \t]*(?:\r?\n)|\Z)",
        project_text,
    )
    if project_section is None:
        return None
    version_match = re.search(
        r'(?m)^\s*version\s*=\s*["\x27]([^"\x27]+)'
        r'["\x27]\s*(?:\n|\Z)',
        project_section.group("body"),
    )
    return None if version_match is None else version_match.group(1)


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
    current_project = _regular_file_bytes(project / "pyproject.toml")
    head_project = _head_regular_blob(project, "pyproject.toml")
    current_version = _declared_package_version(current_project)
    head_version = _declared_package_version(head_project)
    if type(dirty) is bool and (
        not source_entries
        or head_entries is None
        or source_entries != head_entries
        or build_entries != source_entries
        or current_project is None
        or head_project is None
        or current_project != head_project
        or current_version != package_version
        or head_version != package_version
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


def _build_root_marker_document(
    *,
    project_root: Path,
    build_root: Path,
) -> dict[str, str]:
    return {
        "build_root": str(build_root),
        "project": "negfpy",
        "project_root": str(project_root),
        "schema": _BUILD_ROOT_SCHEMA,
    }


def _valid_build_root_marker(
    marker: Path,
    *,
    project_root: Path,
    build_root: Path,
) -> bool:
    """Authenticate the exact build root before any generated-tree removal."""

    try:
        mode = marker.lstat().st_mode
        if marker.is_symlink() or not stat.S_ISREG(mode):
            return False
        raw = marker.read_bytes()
    except OSError:
        return False
    if not raw or len(raw) > 4096:
        return False

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate marker key: {key}")
            result[key] = value
        return result

    try:
        document = json.loads(
            raw.decode("ascii"),
            object_pairs_hook=reject_duplicates,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite marker value: {value}")
            ),
        )
    except (UnicodeDecodeError, ValueError):
        return False
    return bool(
        isinstance(document, dict)
        and set(document) == _BUILD_ROOT_MARKER_KEYS
        and document
        == _build_root_marker_document(
            project_root=project_root,
            build_root=build_root,
        )
    )


def _create_build_root_marker(
    marker: Path,
    *,
    project_root: Path,
    build_root: Path,
) -> None:
    """Create a project-bound marker without replacing any existing path."""

    payload = (
        json.dumps(
            _build_root_marker_document(
                project_root=project_root,
                build_root=build_root,
            ),
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    )
    try:
        with marker.open("x", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
    except FileExistsError as exc:
        raise RuntimeError(f"build-root marker appeared concurrently: {marker}") from exc
    except OSError as exc:
        raise RuntimeError(f"could not create build-root marker: {marker}") from exc
    if not _valid_build_root_marker(
        marker,
        project_root=project_root,
        build_root=build_root,
    ):
        raise RuntimeError(f"new build-root marker failed validation: {marker}")


def _fresh_package_root(
    build_lib: str | Path,
    *,
    project_root: Path,
) -> Path:
    """Prepare only a package tree owned by this exact project build root."""

    project = project_root.resolve()
    source_root = (project / "src" / "negfpy").resolve()
    requested_build_root = Path(build_lib)
    if requested_build_root.is_symlink():
        raise RuntimeError(
            f"refusing symlinked build root: {requested_build_root}"
        )
    build_root = requested_build_root.resolve()
    target = build_root / "negfpy"
    marker = build_root / _BUILD_ROOT_MARKER
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
        if not _valid_build_root_marker(
            marker,
            project_root=project,
            build_root=build_root,
        ):
            raise RuntimeError(
                f"refusing unowned build package target without valid marker: {target}"
            )
        shutil.rmtree(target)
        return target

    try:
        build_root.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"could not prepare build root: {build_root}") from exc
    if not build_root.is_dir() or build_root.is_symlink():
        raise RuntimeError(f"build root is not a regular directory: {build_root}")

    marker_exists = False
    try:
        marker.lstat()
        marker_exists = True
    except FileNotFoundError:
        pass
    except OSError as exc:
        raise RuntimeError(f"could not inspect build-root marker: {marker}") from exc
    if marker_exists:
        if not _valid_build_root_marker(
            marker,
            project_root=project,
            build_root=build_root,
        ):
            raise RuntimeError(f"invalid build-root ownership marker: {marker}")
    else:
        if target.exists() or target.is_symlink():
            raise RuntimeError(
                f"build package target appeared before marker creation: {target}"
            )
        _create_build_root_marker(
            marker,
            project_root=project,
            build_root=build_root,
        )
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
