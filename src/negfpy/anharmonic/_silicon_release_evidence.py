"""Portable sealing and verification for silicon release evidence.

The scientific release validators use absolute paths and filesystem stat
identities while they are running.  Those identities are useful for detecting
same-process replacement, but they cannot survive a checksum-preserving copy
to another workstation.  This private module creates a separate, canonical
logical archive manifest whose durable identity consists only of:

* normalized paths relative to one recorded artifact root;
* exact regular-file membership;
* file sizes and SHA256 digests; and
* the identity and certification decision of the final aggregate report.

The manifest is deliberately stored outside the artifact root, avoiding a
self-referential inventory.  Verification accepts a relocated artifact root
and remaps absolute file records embedded in strict JSON reports from the
recorded root to that relocated root.  Device numbers, inodes, mtimes, and
absolute current-machine paths never participate in the portable decision.

External path records require a separate explicit staging operation.  Their
verified bytes are copied into a content-addressed subtree, and an immutable
alias manifest is then bound into a version-2 portable seal.  No external path
is resolved implicitly during later sealing or relocated verification.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass, replace
import errno
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
from types import MappingProxyType
from typing import Any
from uuid import uuid4


SILICON_RELEASE_EVIDENCE_SCHEMA = "negfpy.silicon_release_portable_evidence.v1"
SILICON_RELEASE_EVIDENCE_SCHEMA_VERSION = 1
SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA = (
    "negfpy.silicon_release_portable_evidence.v2"
)
SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA_VERSION = 2
SILICON_EXTERNAL_ALIAS_SCHEMA = "negfpy.silicon_external_evidence_aliases.v1"
SILICON_EXTERNAL_ALIAS_SCHEMA_VERSION = 1
SILICON_RELEASE_REPORT_SCHEMA = "negfpy.silicon_0_2_0_release_evidence.v1"
SILICON_RELEASE_VERSION = "0.2.0"

_SHA256_HEX = frozenset("0123456789abcdef")
_HDF5_SUFFIXES = frozenset({".h5", ".hdf5"})
_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_READ_BLOCK_BYTES = 1024 * 1024
_HDF5_TARGET_READ_BYTES = 8 * 1024 * 1024
_EXTERNAL_STAGING_DIRECTORY = ".negfpy-external-evidence"
_EXTERNAL_ALIAS_FILENAME = "aliases.json"
_HANDOFF_CHECKSUM_FILENAME = "SHA256SUMS"
_HANDOFF_MANIFEST_FILENAME = "portable-manifest.json"
_HANDOFF_RELEASE_REPORT_FILENAME = "silicon-release-report.json"


class SiliconReleaseEvidenceError(RuntimeError):
    """Raised when portable release evidence is incomplete or unsafe."""


@dataclass(frozen=True, slots=True)
class PortableSiliconReleaseEvidence:
    """Validated portable evidence identity."""

    manifest_path: Path
    manifest_sha256: str
    payload_sha256: str
    recorded_artifact_root: str
    verified_artifact_root: Path
    release_report_relative_path: str
    release_report_sha256: str
    file_count: int
    total_size_bytes: int
    json_file_count: int
    path_record_count: int
    hdf5_file_count: int
    verification_scope: str
    trusted_manifest_sha256_matched: bool
    trusted_release_report_sha256_matched: bool

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe verification summary."""

        return {
            "manifest_path": str(self.manifest_path),
            "manifest_sha256": self.manifest_sha256,
            "payload_sha256": self.payload_sha256,
            "recorded_artifact_root": self.recorded_artifact_root,
            "verified_artifact_root": str(self.verified_artifact_root),
            "release_report_relative_path": (self.release_report_relative_path),
            "release_report_sha256": self.release_report_sha256,
            "file_count": self.file_count,
            "total_size_bytes": self.total_size_bytes,
            "json_file_count": self.json_file_count,
            "path_record_count": self.path_record_count,
            "hdf5_file_count": self.hdf5_file_count,
            "verification_scope": self.verification_scope,
            "trusted_manifest_sha256_matched": (
                self.trusted_manifest_sha256_matched
            ),
            "trusted_release_report_sha256_matched": (
                self.trusted_release_report_sha256_matched
            ),
        }


@dataclass(frozen=True, slots=True)
class StagedSiliconExternalEvidence:
    """Immutable identity of explicitly staged external evidence files."""

    alias_manifest_path: Path
    alias_manifest_sha256: str
    payload_sha256: str
    artifact_root: Path
    staging_subtree_relative_path: str
    alias_count: int
    object_count: int
    total_size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe staging summary."""

        return {
            "alias_manifest_path": str(self.alias_manifest_path),
            "alias_manifest_sha256": self.alias_manifest_sha256,
            "payload_sha256": self.payload_sha256,
            "artifact_root": str(self.artifact_root),
            "staging_subtree_relative_path": (
                self.staging_subtree_relative_path
            ),
            "alias_count": self.alias_count,
            "object_count": self.object_count,
            "total_size_bytes": self.total_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class SiliconReleaseHandoffEvidence:
    """Validated archival handoff without a new scientific calculation."""

    action: str
    handoff_directory: Path
    checksum_path: Path
    checksum_sha256: str
    release_report_copy_path: Path
    external_evidence_staged: bool
    portable_evidence: PortableSiliconReleaseEvidence

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-safe archival handoff summary."""

        return {
            "action": self.action,
            "handoff_directory": str(self.handoff_directory),
            "checksum_path": str(self.checksum_path),
            "checksum_sha256": self.checksum_sha256,
            "release_report_copy_path": str(
                self.release_report_copy_path
            ),
            "external_evidence_staged": self.external_evidence_staged,
            "scientific_recomputation_performed": False,
            "portable_evidence": self.portable_evidence.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class _FileSnapshot:
    relative_path: str
    path: Path
    size_bytes: int
    sha256: str
    stat_signature: tuple[int, int, int, int, int, int, int]

    def inventory_record(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class _ExternalSource:
    source_path: str
    path: Path
    size_bytes: int
    sha256: str
    stat_signature: tuple[int, int, int, int, int, int, int]
    json_evidence: bool


@dataclass(slots=True)
class _ExternalAliasIndex:
    recorded_artifact_root: str
    alias_manifest_relative_path: str
    manifest_sha256: str
    payload_sha256: str
    aliases: Mapping[str, Mapping[str, Any]]
    objects: Mapping[str, Mapping[str, Any]]
    observed_sources: set[str]

    def resolve(
        self,
        *,
        source_path: str,
        sha256: str,
        size_bytes: int | None,
        location: str,
    ) -> str:
        alias = self.aliases.get(source_path)
        if alias is None:
            raise SiliconReleaseEvidenceError(
                f"{location}.path is external but absent from the bound "
                "external alias manifest."
            )
        if alias["sha256"] != sha256:
            raise SiliconReleaseEvidenceError(
                f"{location} SHA256 differs from its external alias."
            )
        if size_bytes is not None and alias["size_bytes"] != size_bytes:
            raise SiliconReleaseEvidenceError(
                f"{location}.size_bytes differs from its external alias."
            )
        self.observed_sources.add(source_path)
        parent = PurePosixPath(
            self.alias_manifest_relative_path
        ).parent
        return _safe_relative_posix(
            (parent / alias["object_relative_path"]).as_posix(),
            name=f"{location}.staged_path",
        )

    def require_complete(self) -> None:
        declared = set(self.aliases)
        if self.observed_sources != declared:
            missing = sorted(declared - self.observed_sources)
            unexpected = sorted(self.observed_sources - declared)
            raise SiliconReleaseEvidenceError(
                "external alias usage differs from the immutable manifest; "
                f"unused={missing!r}, unexpected={unexpected!r}."
            )


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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SiliconReleaseEvidenceError(
            f"cannot open evidence file without following links: {path}"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise SiliconReleaseEvidenceError(
                f"evidence member is not a regular file: {path}"
            )
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            for block in iter(lambda: stream.read(_READ_BLOCK_BYTES), b""):
                digest.update(block)
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _SHA256_HEX for character in value)
    ):
        raise SiliconReleaseEvidenceError(
            f"{name} must be a lowercase 64-character SHA256."
        )
    return value


def _safe_relative_posix(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise SiliconReleaseEvidenceError(
            f"{name} must be a normalized relative POSIX path."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise SiliconReleaseEvidenceError(
            f"{name} must be a normalized relative POSIX path."
        )
    return value


def _absolute_posix(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise SiliconReleaseEvidenceError(
            f"{name} must be a canonical absolute POSIX path."
        )
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.as_posix() != value
    ):
        raise SiliconReleaseEvidenceError(
            f"{name} must be a canonical absolute POSIX path."
        )
    return value


def _absolute_without_symlink(path: str | os.PathLike[str], *, name: str) -> Path:
    candidate = Path(path).expanduser()
    absolute = Path(os.path.abspath(candidate))
    cursor = Path(absolute.anchor)
    try:
        for component in absolute.parts[1:]:
            cursor = cursor / component
            mode = os.lstat(cursor).st_mode
            if stat.S_ISLNK(mode):
                raise SiliconReleaseEvidenceError(
                    f"{name} traverses a symbolic link: {cursor}"
                )
    except FileNotFoundError as exc:
        raise SiliconReleaseEvidenceError(f"{name} does not exist: {absolute}") from exc
    except OSError as exc:
        raise SiliconReleaseEvidenceError(f"cannot inspect {name}: {absolute}") from exc
    return absolute


def _regular_file_without_symlink(
    path: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    absolute = _absolute_without_symlink(path, name=name)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise SiliconReleaseEvidenceError(f"cannot inspect {name}: {absolute}") from exc
    if not stat.S_ISREG(mode):
        raise SiliconReleaseEvidenceError(f"{name} must be a regular non-symlink file.")
    return absolute


def _directory_without_symlink(
    path: str | os.PathLike[str],
    *,
    name: str,
) -> Path:
    absolute = _absolute_without_symlink(path, name=name)
    try:
        mode = os.lstat(absolute).st_mode
    except OSError as exc:
        raise SiliconReleaseEvidenceError(f"cannot inspect {name}: {absolute}") from exc
    if not stat.S_ISDIR(mode):
        raise SiliconReleaseEvidenceError(
            f"{name} must be a regular non-symlink directory."
        )
    if absolute == Path(absolute.anchor):
        raise SiliconReleaseEvidenceError(f"{name} must not be the filesystem root.")
    return absolute


def _stat_signature(value: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _scan_artifact_tree(root: Path) -> tuple[_FileSnapshot, ...]:
    """Hash every regular member while rejecting ambiguous filesystem objects."""

    snapshots: list[_FileSnapshot] = []
    seen_inodes: dict[tuple[int, int], str] = {}

    def visit(directory: Path, relative_directory: PurePosixPath | None) -> None:
        try:
            with os.scandir(directory) as stream:
                entries = sorted(stream, key=lambda item: item.name)
        except OSError as exc:
            raise SiliconReleaseEvidenceError(
                f"cannot enumerate artifact directory: {directory}"
            ) from exc
        for entry in entries:
            if "\x00" in entry.name or "/" in entry.name:
                raise SiliconReleaseEvidenceError(
                    f"artifact member has an unsafe name: {entry.name!r}"
                )
            relative = (
                PurePosixPath(entry.name)
                if relative_directory is None
                else relative_directory / entry.name
            )
            relative_text = _safe_relative_posix(
                relative.as_posix(),
                name="artifact member",
            )
            try:
                metadata = entry.stat(follow_symlinks=False)
            except OSError as exc:
                raise SiliconReleaseEvidenceError(
                    f"cannot inspect artifact member: {entry.path}"
                ) from exc
            if entry.is_symlink():
                raise SiliconReleaseEvidenceError(
                    f"artifact tree contains a symbolic link: {relative_text}"
                )
            mode = metadata.st_mode
            path = Path(entry.path)
            if stat.S_ISDIR(mode):
                visit(path, relative)
                continue
            if not stat.S_ISREG(mode):
                raise SiliconReleaseEvidenceError(
                    "artifact tree contains a special filesystem object: "
                    f"{relative_text}"
                )
            if metadata.st_nlink != 1:
                raise SiliconReleaseEvidenceError(
                    "artifact tree contains a hard-linked file: " f"{relative_text}"
                )
            inode = (int(metadata.st_dev), int(metadata.st_ino))
            previous = seen_inodes.get(inode)
            if previous is not None:
                raise SiliconReleaseEvidenceError(
                    "artifact members share a filesystem inode: "
                    f"{previous!r} and {relative_text!r}"
                )
            seen_inodes[inode] = relative_text
            digest = _file_sha256(path)
            try:
                after = os.lstat(path)
            except OSError as exc:
                raise SiliconReleaseEvidenceError(
                    f"artifact member disappeared while hashing: {relative_text}"
                ) from exc
            if _stat_signature(metadata) != _stat_signature(after):
                raise SiliconReleaseEvidenceError(
                    f"artifact member changed while hashing: {relative_text}"
                )
            snapshots.append(
                _FileSnapshot(
                    relative_path=relative_text,
                    path=path,
                    size_bytes=int(metadata.st_size),
                    sha256=digest,
                    stat_signature=_stat_signature(metadata),
                )
            )

    visit(root, None)
    snapshots.sort(key=lambda item: item.relative_path)
    if not snapshots:
        raise SiliconReleaseEvidenceError(
            "artifact root contains no regular evidence files."
        )
    return tuple(snapshots)


def _strict_json(path: Path, *, name: str) -> Any:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SiliconReleaseEvidenceError(
                    f"{name} contains duplicate JSON key {key!r}."
                )
            result[key] = value
        return result

    def finite(value: str) -> Any:
        raise SiliconReleaseEvidenceError(
            f"{name} contains nonfinite JSON value {value!r}."
        )

    try:
        raw = path.read_bytes()
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique,
            parse_constant=finite,
        )
    except SiliconReleaseEvidenceError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise SiliconReleaseEvidenceError(
            f"{name} is not strict finite UTF-8 JSON."
        ) from exc


def _inventory_mapping(
    snapshots: Sequence[_FileSnapshot],
) -> Mapping[str, _FileSnapshot]:
    return MappingProxyType(
        {snapshot.relative_path: snapshot for snapshot in snapshots}
    )


def _record_relative_path(
    value: Any,
    *,
    recorded_root: str,
    name: str,
    sha256: str,
    size_bytes: int | None,
    external_aliases: _ExternalAliasIndex | None,
) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise SiliconReleaseEvidenceError(f"{name}.path must be nonempty path text.")
    candidate = PurePosixPath(value)
    if candidate.is_absolute():
        source_path = _absolute_posix(value, name=f"{name}.path")
        root = PurePosixPath(recorded_root)
        try:
            relative = PurePosixPath(source_path).relative_to(root)
        except ValueError:
            if external_aliases is None:
                raise SiliconReleaseEvidenceError(
                    f"{name}.path lies outside the recorded artifact root "
                    "and no external alias manifest was bound."
                )
            return external_aliases.resolve(
                source_path=source_path,
                sha256=sha256,
                size_bytes=size_bytes,
                location=name,
            )
        return _safe_relative_posix(
            relative.as_posix(),
            name=f"{name}.path",
        )
    return _safe_relative_posix(value, name=f"{name}.path")


def _audit_path_records(
    value: Any,
    *,
    recorded_root: str,
    inventory: Mapping[str, _FileSnapshot],
    location: str,
    external_aliases: _ExternalAliasIndex | None,
) -> int:
    count = 0
    if isinstance(value, Mapping):
        if "path" in value:
            if "sha256" not in value:
                raise SiliconReleaseEvidenceError(
                    f"{location} contains path without SHA256 and is "
                    "ambiguous evidence."
                )
            digest = _require_sha256(
                value["sha256"],
                name=f"{location}.sha256",
            )
            size: int | None = None
            if "size_bytes" in value:
                raw_size = value["size_bytes"]
                if (
                    isinstance(raw_size, bool)
                    or not isinstance(raw_size, int)
                    or raw_size < 0
                ):
                    raise SiliconReleaseEvidenceError(
                        f"{location}.size_bytes is invalid."
                    )
                size = raw_size
            relative = _record_relative_path(
                value["path"],
                recorded_root=recorded_root,
                name=location,
                sha256=digest,
                size_bytes=size,
                external_aliases=external_aliases,
            )
            member = inventory.get(relative)
            if member is None:
                raise SiliconReleaseEvidenceError(
                    f"{location}.path is absent from the sealed inventory: "
                    f"{relative}"
                )
            if digest != member.sha256:
                raise SiliconReleaseEvidenceError(
                    f"{location} SHA256 differs from the sealed member."
                )
            if size is not None and size != member.size_bytes:
                raise SiliconReleaseEvidenceError(
                    f"{location}.size_bytes differs from the sealed member."
                )
            count += 1
        for key, child in value.items():
            count += _audit_path_records(
                child,
                recorded_root=recorded_root,
                inventory=inventory,
                location=f"{location}.{key}",
                external_aliases=external_aliases,
            )
        return count
    if isinstance(value, list):
        for index, child in enumerate(value):
            count += _audit_path_records(
                child,
                recorded_root=recorded_root,
                inventory=inventory,
                location=f"{location}[{index}]",
                external_aliases=external_aliases,
            )
    return count


def _read_hdf5_dataset(dataset: Any) -> None:
    if bool(getattr(dataset, "is_virtual", False)):
        raise SiliconReleaseEvidenceError(
            f"HDF5 evidence uses a nonportable virtual dataset: {dataset.name}"
        )
    external = getattr(dataset, "external", None)
    if external:
        raise SiliconReleaseEvidenceError(
            f"HDF5 evidence uses external storage: {dataset.name}"
        )
    if dataset.shape == ():
        dataset[()]
        return
    if dataset.size == 0:
        dataset[...]
        return
    if dataset.chunks is not None:
        for selection in dataset.iter_chunks():
            dataset[selection]
        return
    trailing = math.prod(dataset.shape[1:]) if dataset.ndim > 1 else 1
    bytes_per_row = max(1, trailing * max(1, int(dataset.dtype.itemsize)))
    rows = max(1, _HDF5_TARGET_READ_BYTES // bytes_per_row)
    for start in range(0, dataset.shape[0], rows):
        dataset[start : min(dataset.shape[0], start + rows)]


def _audit_hdf5(path: Path, *, relative_path: str) -> None:
    try:
        import h5py
    except ImportError as exc:
        raise SiliconReleaseEvidenceError(
            "portable silicon evidence containing HDF5 requires h5py."
        ) from exc

    try:
        with h5py.File(path, "r") as handle:
            seen_groups: set[int] = set()

            def visit(group: Any) -> None:
                address = int(h5py.h5o.get_info(group.id).addr)
                if address in seen_groups:
                    return
                seen_groups.add(address)
                for name in sorted(group.keys()):
                    link = group.get(name, getlink=True)
                    if isinstance(link, (h5py.SoftLink, h5py.ExternalLink)):
                        raise SiliconReleaseEvidenceError(
                            "HDF5 evidence contains a soft or external link: "
                            f"{relative_path}:{group.name}/{name}"
                        )
                    child = group[name]
                    if isinstance(child, h5py.Group):
                        visit(child)
                    elif isinstance(child, h5py.Dataset):
                        _read_hdf5_dataset(child)
                    else:
                        raise SiliconReleaseEvidenceError(
                            "HDF5 evidence contains an unsupported object: "
                            f"{relative_path}:{group.name}/{name}"
                        )

            visit(handle)
    except SiliconReleaseEvidenceError:
        raise
    except Exception as exc:
        raise SiliconReleaseEvidenceError(
            f"HDF5 evidence is unreadable or truncated: {relative_path}"
        ) from exc


def _audit_artifact_contents(
    snapshots: Sequence[_FileSnapshot],
    *,
    recorded_root: str,
    external_aliases: _ExternalAliasIndex | None = None,
) -> dict[str, int]:
    inventory = _inventory_mapping(snapshots)
    json_count = 0
    path_record_count = 0
    hdf5_count = 0
    for snapshot in snapshots:
        suffix = snapshot.path.suffix.lower()
        if suffix == ".json":
            value = _strict_json(
                snapshot.path,
                name=f"JSON evidence {snapshot.relative_path}",
            )
            json_count += 1
            path_record_count += _audit_path_records(
                value,
                recorded_root=recorded_root,
                inventory=inventory,
                location=snapshot.relative_path,
                external_aliases=external_aliases,
            )
        if suffix in _HDF5_SUFFIXES:
            _audit_hdf5(
                snapshot.path,
                relative_path=snapshot.relative_path,
            )
            hdf5_count += 1
        elif snapshot.size_bytes >= len(_HDF5_SIGNATURE):
            try:
                with snapshot.path.open("rb") as stream:
                    signature = stream.read(len(_HDF5_SIGNATURE))
            except OSError as exc:
                raise SiliconReleaseEvidenceError(
                    f"cannot inspect evidence signature: {snapshot.relative_path}"
                ) from exc
            if signature == _HDF5_SIGNATURE:
                _audit_hdf5(
                    snapshot.path,
                    relative_path=snapshot.relative_path,
                )
                hdf5_count += 1
    if external_aliases is not None:
        external_aliases.require_complete()
    return {
        "json_file_count": json_count,
        "path_record_count": path_record_count,
        "hdf5_file_count": hdf5_count,
    }


def _certified_release_report(report_path: Path) -> Mapping[str, Any]:
    report = _strict_json(report_path, name="final silicon release report")
    if not isinstance(report, Mapping):
        raise SiliconReleaseEvidenceError(
            "final silicon release report must be a JSON object."
        )
    if (
        report.get("schema") != SILICON_RELEASE_REPORT_SCHEMA
        or report.get("schema_version") != 1
        or report.get("status") != "passed"
        or report.get("validation_passed") is not True
        or report.get("certification_eligible") is not True
        or report.get("release_certified") is not True
        or report.get("release_version") != SILICON_RELEASE_VERSION
    ):
        raise SiliconReleaseEvidenceError(
            "final aggregate report is not an eligible certified silicon "
            "release report."
        )
    return report


def _release_report_identity(
    report_path: Path,
    *,
    relative_path: str,
    snapshot: _FileSnapshot,
) -> dict[str, Any]:
    report = _certified_release_report(report_path)
    return {
        "relative_path": relative_path,
        "sha256": snapshot.sha256,
        "size_bytes": snapshot.size_bytes,
        "schema": report["schema"],
        "schema_version": report["schema_version"],
        "status": report["status"],
        "validation_passed": report["validation_passed"],
        "certification_eligible": report["certification_eligible"],
        "release_certified": report["release_certified"],
        "release_version": report["release_version"],
    }


def _recheck_snapshots(snapshots: Sequence[_FileSnapshot]) -> None:
    for snapshot in snapshots:
        try:
            metadata = os.lstat(snapshot.path)
        except OSError as exc:
            raise SiliconReleaseEvidenceError(
                "artifact member disappeared during evidence audit: "
                f"{snapshot.relative_path}"
            ) from exc
        if _stat_signature(metadata) != snapshot.stat_signature:
            raise SiliconReleaseEvidenceError(
                "artifact member changed during evidence audit: "
                f"{snapshot.relative_path}"
            )


def _manifest_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "payload": dict(payload),
        "payload_sha256": _payload_sha256(payload),
    }


def _write_atomic_exclusive(path: Path, payload: bytes) -> None:
    target = Path(os.path.abspath(path.expanduser()))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.close(descriptor)
        descriptor = None
        try:
            os.link(temporary, target)
        except FileExistsError as exc:
            raise FileExistsError(
                "portable evidence manifest already exists and is immutable: "
                f"{target}"
            ) from exc
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _handoff_checksum_bytes(
    manifest_sha256: str,
    release_report_sha256: str,
) -> bytes:
    manifest_digest = _require_sha256(
        manifest_sha256,
        name="handoff manifest SHA256",
    )
    report_digest = _require_sha256(
        release_report_sha256,
        name="handoff release-report SHA256",
    )
    return (
        f"{manifest_digest}  {_HANDOFF_MANIFEST_FILENAME}\n"
        f"{report_digest}  {_HANDOFF_RELEASE_REPORT_FILENAME}\n"
    ).encode("ascii")


def _handoff_destination(
    artifact_root: Path,
    value: str | os.PathLike[str],
) -> Path:
    target = Path(os.path.abspath(Path(value).expanduser()))
    parent = _directory_without_symlink(
        target.parent,
        name="handoff_directory parent",
    )
    target = parent / target.name
    if target == artifact_root or target.is_relative_to(artifact_root):
        raise SiliconReleaseEvidenceError(
            "handoff_directory must lie outside artifact_root."
        )
    if target.exists() or target.is_symlink():
        raise FileExistsError(
            "handoff_directory already exists and will not be overwritten."
        )
    return target


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one directory without a racy overwrite fallback."""

    if source.parent != target.parent:
        raise SiliconReleaseEvidenceError(
            "atomic handoff publication requires one parent filesystem."
        )
    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:
        raise SiliconReleaseEvidenceError(
            "atomic no-replace directory publication is unavailable."
        ) from exc
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(source),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(
                "handoff_directory appeared during publication and was not "
                "overwritten."
            )
        raise SiliconReleaseEvidenceError(
            "atomic no-replace handoff publication failed: "
            f"{os.strerror(error_number)}"
        )


def _remove_owned_directory(path: Path) -> None:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
        shutil.rmtree(path)


def _verified_handoff_files(
    checksum_path: str | os.PathLike[str],
) -> tuple[Path, Path, Path, Path, str, str, str]:
    checksum = _regular_file_without_symlink(
        checksum_path,
        name="trusted SHA256SUMS handoff",
    )
    if checksum.name != _HANDOFF_CHECKSUM_FILENAME:
        raise SiliconReleaseEvidenceError(
            f"trusted checksum handoff must be named "
            f"{_HANDOFF_CHECKSUM_FILENAME!r}."
        )
    directory = _directory_without_symlink(
        checksum.parent,
        name="trusted handoff directory",
    )
    expected_names = (
        _HANDOFF_CHECKSUM_FILENAME,
        _HANDOFF_MANIFEST_FILENAME,
        _HANDOFF_RELEASE_REPORT_FILENAME,
    )
    try:
        with os.scandir(directory) as stream:
            entries = sorted(stream, key=lambda item: item.name)
    except OSError as exc:
        raise SiliconReleaseEvidenceError(
            "cannot inspect trusted handoff directory."
        ) from exc
    if tuple(item.name for item in entries) != expected_names:
        raise SiliconReleaseEvidenceError(
            "trusted handoff directory membership is not exact."
        )
    if any(
        item.is_symlink() or not item.is_file(follow_symlinks=False)
        for item in entries
    ):
        raise SiliconReleaseEvidenceError(
            "trusted handoff members must be regular non-symlink files."
        )
    snapshots = _scan_artifact_tree(directory)
    by_relative = _inventory_mapping(snapshots)
    if set(by_relative) != set(expected_names):
        raise SiliconReleaseEvidenceError(
            "trusted handoff directory contains nested or ambiguous evidence."
        )
    try:
        raw = checksum.read_bytes()
        text = raw.decode("ascii")
    except (OSError, UnicodeError) as exc:
        raise SiliconReleaseEvidenceError(
            "trusted SHA256SUMS handoff is unreadable."
        ) from exc
    lines = text.splitlines(keepends=True)
    expected_paths = (
        _HANDOFF_MANIFEST_FILENAME,
        _HANDOFF_RELEASE_REPORT_FILENAME,
    )
    if len(lines) != len(expected_paths):
        raise SiliconReleaseEvidenceError(
            "trusted SHA256SUMS handoff must contain exactly two records."
        )
    digests: list[str] = []
    for position, (line, expected_path) in enumerate(
        zip(lines, expected_paths, strict=True)
    ):
        suffix = f"  {expected_path}\n"
        if not line.endswith(suffix):
            raise SiliconReleaseEvidenceError(
                f"trusted SHA256SUMS record {position} is not canonical."
            )
        digests.append(
            _require_sha256(
                line[: -len(suffix)],
                name=f"trusted SHA256SUMS record {position}",
            )
        )
    manifest_digest, report_digest = digests
    if raw != _handoff_checksum_bytes(manifest_digest, report_digest):
        raise SiliconReleaseEvidenceError(
            "trusted SHA256SUMS handoff is not canonical."
        )
    manifest = by_relative[_HANDOFF_MANIFEST_FILENAME]
    report = by_relative[_HANDOFF_RELEASE_REPORT_FILENAME]
    if (
        manifest.sha256 != manifest_digest
        or report.sha256 != report_digest
    ):
        raise SiliconReleaseEvidenceError(
            "trusted handoff member differs from its SHA256SUMS record."
        )
    _certified_release_report(report.path)
    _recheck_snapshots(snapshots)
    return (
        directory,
        checksum,
        manifest.path,
        report.path,
        manifest_digest,
        report_digest,
        by_relative[_HANDOFF_CHECKSUM_FILENAME].sha256,
    )


def _external_record_declarations(
    value: Any,
    *,
    recorded_root: str,
    location: str,
) -> tuple[tuple[str, str, int | None, str], ...]:
    declarations: list[tuple[str, str, int | None, str]] = []
    if isinstance(value, Mapping):
        if "path" in value:
            if "sha256" not in value:
                raise SiliconReleaseEvidenceError(
                    f"{location} contains path without SHA256 and is "
                    "ambiguous evidence."
                )
            raw_path = value["path"]
            digest = _require_sha256(
                value["sha256"],
                name=f"{location}.sha256",
            )
            size: int | None = None
            if "size_bytes" in value:
                raw_size = value["size_bytes"]
                if (
                    isinstance(raw_size, bool)
                    or not isinstance(raw_size, int)
                    or raw_size < 0
                ):
                    raise SiliconReleaseEvidenceError(
                        f"{location}.size_bytes is invalid."
                    )
                size = raw_size
            if (
                not isinstance(raw_path, str)
                or not raw_path
                or "\x00" in raw_path
            ):
                raise SiliconReleaseEvidenceError(
                    f"{location}.path must be nonempty path text."
                )
            candidate = PurePosixPath(raw_path)
            if candidate.is_absolute():
                source_path = _absolute_posix(
                    raw_path,
                    name=f"{location}.path",
                )
                try:
                    PurePosixPath(source_path).relative_to(
                        PurePosixPath(recorded_root)
                    )
                except ValueError:
                    declarations.append(
                        (source_path, digest, size, location)
                    )
            else:
                _safe_relative_posix(
                    raw_path,
                    name=f"{location}.path",
                )
        for key, child in value.items():
            declarations.extend(
                _external_record_declarations(
                    child,
                    recorded_root=recorded_root,
                    location=f"{location}.{key}",
                )
            )
    elif isinstance(value, list):
        for index, child in enumerate(value):
            declarations.extend(
                _external_record_declarations(
                    child,
                    recorded_root=recorded_root,
                    location=f"{location}[{index}]",
                )
            )
    return tuple(declarations)


def _external_source_snapshot(
    source_path: str,
    *,
    declared_sha256: str,
    declared_size: int | None,
) -> _ExternalSource:
    path = _regular_file_without_symlink(
        source_path,
        name=f"external evidence source {source_path}",
    )
    metadata = os.lstat(path)
    if metadata.st_nlink != 1:
        raise SiliconReleaseEvidenceError(
            f"external evidence source is hard-linked: {source_path}"
        )
    signature = _stat_signature(metadata)
    digest = _file_sha256(path)
    after = os.lstat(path)
    if signature != _stat_signature(after):
        raise SiliconReleaseEvidenceError(
            f"external evidence source changed while hashing: {source_path}"
        )
    if digest != declared_sha256:
        raise SiliconReleaseEvidenceError(
            f"external evidence source SHA256 is mismatched: {source_path}"
        )
    if declared_size is not None and metadata.st_size != declared_size:
        raise SiliconReleaseEvidenceError(
            f"external evidence source size is mismatched: {source_path}"
        )
    return _ExternalSource(
        source_path=source_path,
        path=path,
        size_bytes=int(metadata.st_size),
        sha256=digest,
        stat_signature=signature,
        json_evidence=path.suffix.lower() == ".json",
    )


def _discover_external_sources(
    root: Path,
    snapshots: Sequence[_FileSnapshot],
    *,
    require_nonempty: bool = True,
) -> tuple[_ExternalSource, ...]:
    recorded_root = _absolute_posix(
        root.as_posix(),
        name="recorded_artifact_root",
    )
    pending: list[tuple[Path, str]] = [
        (
            snapshot.path,
            snapshot.relative_path,
        )
        for snapshot in snapshots
        if snapshot.path.suffix.lower() == ".json"
    ]
    parsed_documents: set[str] = set()
    sources: dict[str, _ExternalSource] = {}
    declarations: dict[str, tuple[str, int | None]] = {}
    while pending:
        document_path, location = pending.pop(0)
        document_identity = document_path.as_posix()
        if document_identity in parsed_documents:
            continue
        parsed_documents.add(document_identity)
        value = _strict_json(
            document_path,
            name=f"JSON evidence {location}",
        )
        for source_path, digest, size, record_location in (
            _external_record_declarations(
                value,
                recorded_root=recorded_root,
                location=location,
            )
        ):
            previous = declarations.get(source_path)
            if previous is not None:
                previous_digest, previous_size = previous
                if (
                    previous_digest != digest
                    or (
                        previous_size is not None
                        and size is not None
                        and previous_size != size
                    )
                ):
                    raise SiliconReleaseEvidenceError(
                        "external evidence path has conflicting SHA256 or "
                        f"size declarations: {source_path}"
                    )
                if previous_size is None and size is not None:
                    declarations[source_path] = (digest, size)
            else:
                declarations[source_path] = (digest, size)
            source = sources.get(source_path)
            if source is None:
                source = _external_source_snapshot(
                    source_path,
                    declared_sha256=digest,
                    declared_size=size,
                )
                sources[source_path] = source
                if source.json_evidence:
                    pending.append(
                        (
                            source.path,
                            f"external:{source.source_path}",
                        )
                    )
            elif (
                source.sha256 != digest
                or (size is not None and source.size_bytes != size)
            ):
                raise SiliconReleaseEvidenceError(
                    "external evidence declaration changed during discovery: "
                    f"{record_location}"
                )
    if require_nonempty and not sources:
        raise SiliconReleaseEvidenceError(
            "strict JSON evidence contains no external path+SHA256 records "
            "to stage."
        )
    return tuple(sources[name] for name in sorted(sources))


def _source_has_hdf5_signature(source: _ExternalSource) -> bool:
    if source.size_bytes < len(_HDF5_SIGNATURE):
        return False
    try:
        with source.path.open("rb") as stream:
            return stream.read(len(_HDF5_SIGNATURE)) == _HDF5_SIGNATURE
    except OSError as exc:
        raise SiliconReleaseEvidenceError(
            f"cannot inspect external evidence source: {source.source_path}"
        ) from exc


def _external_alias_payload(
    root: Path,
    sources: Sequence[_ExternalSource],
) -> dict[str, Any]:
    by_digest: dict[str, list[_ExternalSource]] = {}
    for source in sources:
        by_digest.setdefault(source.sha256, []).append(source)
    object_path_by_digest: dict[str, str] = {}
    objects: list[dict[str, Any]] = []
    for digest, matches in sorted(by_digest.items()):
        sizes = {source.size_bytes for source in matches}
        if len(sizes) != 1:
            raise SiliconReleaseEvidenceError(
                "equal external SHA256 values have conflicting sizes."
            )
        if any(source.json_evidence for source in matches):
            kind = "json"
        elif _source_has_hdf5_signature(matches[0]):
            kind = "hdf5"
        else:
            kind = "binary"
        suffix = {"json": "json", "hdf5": "hdf5", "binary": "bin"}[kind]
        relative = f"objects/{digest[:2]}/{digest}.{suffix}"
        object_path_by_digest[digest] = relative
        objects.append(
            {
                "object_relative_path": relative,
                "sha256": digest,
                "size_bytes": matches[0].size_bytes,
                "kind": kind,
            }
        )
    aliases = [
        {
            "source_path": source.source_path,
            "sha256": source.sha256,
            "size_bytes": source.size_bytes,
            "object_relative_path": object_path_by_digest[source.sha256],
        }
        for source in sources
    ]
    return {
        "schema": SILICON_EXTERNAL_ALIAS_SCHEMA,
        "schema_version": SILICON_EXTERNAL_ALIAS_SCHEMA_VERSION,
        "recorded_artifact_root": _absolute_posix(
            root.as_posix(),
            name="recorded_artifact_root",
        ),
        "path_semantics": {
            "source_paths": "canonical_absolute_posix_at_staging",
            "object_paths": "relative_posix_from_alias_manifest_directory",
            "object_identity": "sha256_and_size_bytes",
        },
        "aliases": aliases,
        "objects": objects,
        "object_inventory_sha256": _payload_sha256(objects),
        "summary": {
            "alias_count": len(aliases),
            "object_count": len(objects),
            "total_size_bytes": sum(
                int(item["size_bytes"]) for item in objects
            ),
        },
    }


def _copy_external_source(
    source: _ExternalSource,
    target: Path,
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        source_flags |= os.O_NOFOLLOW
    source_descriptor: int | None = None
    target_descriptor: int | None = None
    digest = hashlib.sha256()
    copied = 0
    try:
        source_descriptor = os.open(source.path, source_flags)
        opened = os.fstat(source_descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or _stat_signature(opened) != source.stat_signature
        ):
            raise SiliconReleaseEvidenceError(
                "external evidence source changed before copying: "
                f"{source.source_path}"
            )
        target_descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
        while True:
            block = os.read(source_descriptor, _READ_BLOCK_BYTES)
            if not block:
                break
            digest.update(block)
            copied += len(block)
            view = memoryview(block)
            while view:
                written = os.write(target_descriptor, view)
                if written <= 0:
                    raise SiliconReleaseEvidenceError(
                        "short write while staging external evidence."
                    )
                view = view[written:]
        os.fsync(target_descriptor)
    except Exception:
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        if target_descriptor is not None:
            os.close(target_descriptor)
    if (
        copied != source.size_bytes
        or digest.hexdigest() != source.sha256
        or _stat_signature(os.lstat(source.path)) != source.stat_signature
    ):
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        raise SiliconReleaseEvidenceError(
            f"external evidence source changed while copying: "
            f"{source.source_path}"
        )
    target_metadata = os.lstat(target)
    if (
        not stat.S_ISREG(target_metadata.st_mode)
        or target_metadata.st_nlink != 1
        or target_metadata.st_size != source.size_bytes
        or _file_sha256(target) != source.sha256
    ):
        raise SiliconReleaseEvidenceError(
            f"staged external evidence object failed verification: {target}"
        )
    parent_descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)


def _load_external_alias_manifest(
    root: Path,
    manifest_path: str | os.PathLike[str],
    *,
    expected_recorded_root: str | None,
) -> tuple[_ExternalAliasIndex, dict[str, int]]:
    manifest = _regular_file_without_symlink(
        manifest_path,
        name="external_alias_manifest_path",
    )
    try:
        manifest_relative = manifest.relative_to(root)
    except ValueError as exc:
        raise SiliconReleaseEvidenceError(
            "external alias manifest must lie inside artifact_root."
        ) from exc
    manifest_relative_text = _safe_relative_posix(
        manifest_relative.as_posix(),
        name="external_alias_manifest_path",
    )
    manifest_metadata = os.lstat(manifest)
    if manifest_metadata.st_nlink != 1:
        raise SiliconReleaseEvidenceError(
            "external alias manifest must not be hard-linked."
        )
    manifest_signature = _stat_signature(manifest_metadata)
    parts = PurePosixPath(manifest_relative_text).parts
    if (
        len(parts) != 3
        or parts[0] != _EXTERNAL_STAGING_DIRECTORY
        or parts[2] != _EXTERNAL_ALIAS_FILENAME
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest is outside the pinned staging layout."
        )
    raw = manifest.read_bytes()
    document = _strict_json(
        manifest,
        name="external alias manifest",
    )
    if (
        not isinstance(document, dict)
        or set(document) != {"payload", "payload_sha256"}
        or raw != _canonical_bytes(document)
        or not isinstance(document["payload"], dict)
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest is not a canonical envelope."
        )
    payload = document["payload"]
    payload_digest = _require_sha256(
        document["payload_sha256"],
        name="external alias payload_sha256",
    )
    if (
        payload_digest != _payload_sha256(payload)
        or parts[1] != payload_digest
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest is not content-addressed to its payload."
        )
    expected_keys = {
        "schema",
        "schema_version",
        "recorded_artifact_root",
        "path_semantics",
        "aliases",
        "objects",
        "object_inventory_sha256",
        "summary",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema") != SILICON_EXTERNAL_ALIAS_SCHEMA
        or payload.get("schema_version")
        != SILICON_EXTERNAL_ALIAS_SCHEMA_VERSION
        or payload.get("path_semantics")
        != {
            "source_paths": "canonical_absolute_posix_at_staging",
            "object_paths": (
                "relative_posix_from_alias_manifest_directory"
            ),
            "object_identity": "sha256_and_size_bytes",
        }
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest schema or path semantics are invalid."
        )
    recorded_root = _absolute_posix(
        payload["recorded_artifact_root"],
        name="external alias recorded_artifact_root",
    )
    if (
        expected_recorded_root is not None
        and recorded_root != expected_recorded_root
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest belongs to another recorded artifact "
            "root."
        )
    raw_objects = payload["objects"]
    if not isinstance(raw_objects, list) or not raw_objects:
        raise SiliconReleaseEvidenceError(
            "external alias object inventory must be nonempty."
        )
    objects: dict[str, dict[str, Any]] = {}
    object_records: list[dict[str, Any]] = []
    for position, raw_object in enumerate(raw_objects):
        if not isinstance(raw_object, dict) or set(raw_object) != {
            "object_relative_path",
            "sha256",
            "size_bytes",
            "kind",
        }:
            raise SiliconReleaseEvidenceError(
                f"external alias objects[{position}] is ambiguous."
            )
        digest = _require_sha256(
            raw_object["sha256"],
            name=f"external alias objects[{position}].sha256",
        )
        size = raw_object["size_bytes"]
        kind = raw_object["kind"]
        if (
            isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or kind not in {"json", "hdf5", "binary"}
        ):
            raise SiliconReleaseEvidenceError(
                f"external alias objects[{position}] is invalid."
            )
        suffix = {
            "json": "json",
            "hdf5": "hdf5",
            "binary": "bin",
        }[kind]
        expected_relative = f"objects/{digest[:2]}/{digest}.{suffix}"
        relative = _safe_relative_posix(
            raw_object["object_relative_path"],
            name=(
                f"external alias objects[{position}]."
                "object_relative_path"
            ),
        )
        if relative != expected_relative or relative in objects:
            raise SiliconReleaseEvidenceError(
                "external alias object path is not canonical or unique."
            )
        record = {
            "object_relative_path": relative,
            "sha256": digest,
            "size_bytes": size,
            "kind": kind,
        }
        objects[relative] = record
        object_records.append(record)
    if (
        object_records
        != sorted(
            object_records,
            key=lambda item: item["object_relative_path"],
        )
        or payload["object_inventory_sha256"]
        != _payload_sha256(object_records)
    ):
        raise SiliconReleaseEvidenceError(
            "external alias object inventory is not canonical."
        )
    raw_aliases = payload["aliases"]
    if not isinstance(raw_aliases, list) or not raw_aliases:
        raise SiliconReleaseEvidenceError(
            "external alias list must be nonempty."
        )
    aliases: dict[str, dict[str, Any]] = {}
    for position, raw_alias in enumerate(raw_aliases):
        if not isinstance(raw_alias, dict) or set(raw_alias) != {
            "source_path",
            "sha256",
            "size_bytes",
            "object_relative_path",
        }:
            raise SiliconReleaseEvidenceError(
                f"external aliases[{position}] is ambiguous."
            )
        source_path = _absolute_posix(
            raw_alias["source_path"],
            name=f"external aliases[{position}].source_path",
        )
        try:
            PurePosixPath(source_path).relative_to(
                PurePosixPath(recorded_root)
            )
        except ValueError:
            pass
        else:
            raise SiliconReleaseEvidenceError(
                "external alias source lies inside recorded_artifact_root."
            )
        digest = _require_sha256(
            raw_alias["sha256"],
            name=f"external aliases[{position}].sha256",
        )
        size = raw_alias["size_bytes"]
        relative = _safe_relative_posix(
            raw_alias["object_relative_path"],
            name=(
                f"external aliases[{position}].object_relative_path"
            ),
        )
        object_record = objects.get(relative)
        if (
            source_path in aliases
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or object_record is None
            or object_record["sha256"] != digest
            or object_record["size_bytes"] != size
        ):
            raise SiliconReleaseEvidenceError(
                f"external aliases[{position}] is inconsistent."
            )
        aliases[source_path] = {
            "source_path": source_path,
            "sha256": digest,
            "size_bytes": size,
            "object_relative_path": relative,
        }
    if list(aliases) != sorted(aliases):
        raise SiliconReleaseEvidenceError(
            "external aliases are not in canonical source-path order."
        )
    if {
        alias["object_relative_path"] for alias in aliases.values()
    } != set(objects):
        raise SiliconReleaseEvidenceError(
            "external alias objects are missing or unreferenced."
        )
    summary = payload["summary"]
    expected_summary = {
        "alias_count": len(aliases),
        "object_count": len(objects),
        "total_size_bytes": sum(
            int(item["size_bytes"]) for item in objects.values()
        ),
    }
    if summary != expected_summary:
        raise SiliconReleaseEvidenceError(
            "external alias summary is inconsistent."
        )
    subtree = manifest.parent
    subtree_snapshots = _scan_artifact_tree(subtree)
    by_relative = _inventory_mapping(subtree_snapshots)
    if set(by_relative) != {_EXTERNAL_ALIAS_FILENAME, *objects}:
        raise SiliconReleaseEvidenceError(
            "external staging subtree has missing or extra members."
        )
    alias_snapshot = by_relative[_EXTERNAL_ALIAS_FILENAME]
    if (
        alias_snapshot.sha256 != hashlib.sha256(raw).hexdigest()
        or alias_snapshot.stat_signature != manifest_signature
    ):
        raise SiliconReleaseEvidenceError(
            "external alias manifest changed while being validated."
        )
    for relative, object_record in objects.items():
        observed = by_relative[relative]
        if (
            observed.sha256 != object_record["sha256"]
            or observed.size_bytes != object_record["size_bytes"]
        ):
            raise SiliconReleaseEvidenceError(
                "staged external object differs from its alias identity: "
                f"{relative}"
            )
    _recheck_snapshots(subtree_snapshots)
    manifest_sha256 = alias_snapshot.sha256
    return (
        _ExternalAliasIndex(
            recorded_artifact_root=recorded_root,
            alias_manifest_relative_path=manifest_relative_text,
            manifest_sha256=manifest_sha256,
            payload_sha256=payload_digest,
            aliases=MappingProxyType(aliases),
            objects=MappingProxyType(objects),
            observed_sources=set(),
        ),
        expected_summary,
    )


def stage_external_silicon_release_evidence(
    artifact_root: str | os.PathLike[str],
) -> StagedSiliconExternalEvidence:
    """Copy every external JSON path record into an immutable local subtree."""

    root = _directory_without_symlink(
        artifact_root,
        name="artifact_root",
    )
    staging_root = root / _EXTERNAL_STAGING_DIRECTORY
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(
            "artifact_root already contains an external-evidence staging "
            "directory."
        )
    snapshots = _scan_artifact_tree(root)
    sources = _discover_external_sources(root, snapshots)
    _recheck_snapshots(snapshots)
    payload = _external_alias_payload(root, sources)
    payload_digest = _payload_sha256(payload)
    document = _manifest_document(payload)
    final_subtree = staging_root / payload_digest
    temporary = staging_root / f".stage-{uuid4().hex}.tmp"
    source_by_digest = {
        source.sha256: source for source in sources
    }
    created_staging_root = False
    try:
        os.mkdir(staging_root, 0o755)
        created_staging_root = True
        os.mkdir(temporary, 0o755)
        for object_record in payload["objects"]:
            source = source_by_digest[object_record["sha256"]]
            _copy_external_source(
                source,
                temporary / object_record["object_relative_path"],
            )
        _write_atomic_exclusive(
            temporary / _EXTERNAL_ALIAS_FILENAME,
            _canonical_bytes(document),
        )
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        os.rename(temporary, final_subtree)
        staging_descriptor = os.open(staging_root, os.O_RDONLY)
        try:
            os.fsync(staging_descriptor)
        finally:
            os.close(staging_descriptor)
    except Exception:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
        if created_staging_root:
            try:
                staging_root.rmdir()
            except OSError:
                pass
        raise
    alias_manifest = final_subtree / _EXTERNAL_ALIAS_FILENAME
    index, summary = _load_external_alias_manifest(
        root,
        alias_manifest,
        expected_recorded_root=root.as_posix(),
    )
    return StagedSiliconExternalEvidence(
        alias_manifest_path=alias_manifest,
        alias_manifest_sha256=index.manifest_sha256,
        payload_sha256=index.payload_sha256,
        artifact_root=root,
        staging_subtree_relative_path=_safe_relative_posix(
            final_subtree.relative_to(root).as_posix(),
            name="staging_subtree_relative_path",
        ),
        alias_count=summary["alias_count"],
        object_count=summary["object_count"],
        total_size_bytes=summary["total_size_bytes"],
    )


def _manifest_payload(
    snapshots: Sequence[_FileSnapshot],
    *,
    recorded_root: str,
    release_report: Mapping[str, Any],
    audit: Mapping[str, int],
    external_alias_manifest: Mapping[str, Any] | None,
) -> dict[str, Any]:
    inventory = [snapshot.inventory_record() for snapshot in snapshots]
    alias_enabled = external_alias_manifest is not None
    payload = {
        "schema": (
            SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA
            if alias_enabled
            else SILICON_RELEASE_EVIDENCE_SCHEMA
        ),
        "schema_version": (
            SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA_VERSION
            if alias_enabled
            else SILICON_RELEASE_EVIDENCE_SCHEMA_VERSION
        ),
        "recorded_artifact_root": recorded_root,
        "path_semantics": {
            "inventory_paths": "relative_posix_from_artifact_root",
            "embedded_absolute_paths": (
                "resolve_recorded_root_or_bound_external_alias"
                if alias_enabled
                else "remap_recorded_artifact_root_to_verified_artifact_root"
            ),
            "portable_identity_fields": [
                "relative_path",
                "size_bytes",
                "sha256",
            ],
            "excluded_machine_identity_fields": [
                "absolute_current_path",
                "device",
                "inode",
                "mtime_ns",
            ],
        },
        "release_report": dict(release_report),
        "inventory": inventory,
        "inventory_sha256": _payload_sha256(inventory),
        "content_summary": {
            "file_count": len(inventory),
            "total_size_bytes": sum(int(item["size_bytes"]) for item in inventory),
            **{name: int(value) for name, value in audit.items()},
        },
    }
    if external_alias_manifest is not None:
        payload["external_alias_manifest"] = dict(
            external_alias_manifest
        )
    return payload


def seal_silicon_release_evidence(
    artifact_root: str | os.PathLike[str],
    release_report_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    external_alias_manifest_path: str | os.PathLike[str] | None = None,
) -> PortableSiliconReleaseEvidence:
    """Seal a completed artifact tree into a canonical relocatable manifest."""

    root = _directory_without_symlink(
        artifact_root,
        name="artifact_root",
    )
    report = _regular_file_without_symlink(
        release_report_path,
        name="release_report_path",
    )
    try:
        report_relative = report.relative_to(root)
    except ValueError as exc:
        raise SiliconReleaseEvidenceError(
            "release_report_path must lie inside artifact_root."
        ) from exc
    report_relative_text = _safe_relative_posix(
        report_relative.as_posix(),
        name="release_report_path",
    )
    output = Path(os.path.abspath(Path(manifest_path).expanduser()))
    if output == root or output.is_relative_to(root):
        raise SiliconReleaseEvidenceError(
            "manifest_path must be outside artifact_root to avoid a recursive "
            "inventory."
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(
            "portable evidence manifest already exists and is immutable."
        )

    snapshots = _scan_artifact_tree(root)
    by_relative = _inventory_mapping(snapshots)
    report_snapshot = by_relative.get(report_relative_text)
    if report_snapshot is None:
        raise SiliconReleaseEvidenceError(
            "release report is absent from the complete artifact inventory."
        )
    recorded_root = _absolute_posix(
        root.as_posix(),
        name="recorded_artifact_root",
    )
    external_aliases: _ExternalAliasIndex | None = None
    external_alias_record: dict[str, Any] | None = None
    alias_summary: dict[str, int] | None = None
    if external_alias_manifest_path is not None:
        external_aliases, alias_summary = _load_external_alias_manifest(
            root,
            external_alias_manifest_path,
            expected_recorded_root=recorded_root,
        )
        alias_snapshot = by_relative.get(
            external_aliases.alias_manifest_relative_path
        )
        if (
            alias_snapshot is None
            or alias_snapshot.sha256
            != external_aliases.manifest_sha256
        ):
            raise SiliconReleaseEvidenceError(
                "external alias manifest is absent from the complete "
                "artifact inventory."
            )
        external_alias_record = {
            "relative_path": (
                external_aliases.alias_manifest_relative_path
            ),
            "sha256": external_aliases.manifest_sha256,
            "size_bytes": alias_snapshot.size_bytes,
            "payload_sha256": external_aliases.payload_sha256,
            **alias_summary,
        }
    audit = _audit_artifact_contents(
        snapshots,
        recorded_root=recorded_root,
        external_aliases=external_aliases,
    )
    report_identity = _release_report_identity(
        report,
        relative_path=report_relative_text,
        snapshot=report_snapshot,
    )
    _recheck_snapshots(snapshots)
    payload = _manifest_payload(
        snapshots,
        recorded_root=recorded_root,
        release_report=report_identity,
        audit=audit,
        external_alias_manifest=external_alias_record,
    )
    document = _manifest_document(payload)
    _write_atomic_exclusive(output, _canonical_bytes(document))
    return verify_silicon_release_evidence(root, output)


def _load_manifest(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    document = _strict_json(path, name="portable evidence manifest")
    if (
        not isinstance(document, dict)
        or set(document) != {"payload", "payload_sha256"}
        or raw != _canonical_bytes(document)
        or not isinstance(document["payload"], dict)
    ):
        raise SiliconReleaseEvidenceError(
            "portable evidence manifest is not a canonical envelope."
        )
    digest = _require_sha256(
        document["payload_sha256"],
        name="portable evidence payload_sha256",
    )
    if digest != _payload_sha256(document["payload"]):
        raise SiliconReleaseEvidenceError(
            "portable evidence manifest payload failed SHA256 validation."
        )
    return document["payload"], digest


def _validate_manifest_payload(
    payload: Mapping[str, Any],
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, int],
    dict[str, Any] | None,
]:
    base_keys = {
        "schema",
        "schema_version",
        "recorded_artifact_root",
        "path_semantics",
        "release_report",
        "inventory",
        "inventory_sha256",
        "content_summary",
    }
    alias_enabled = (
        payload.get("schema") == SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA
        and payload.get("schema_version")
        == SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA_VERSION
    )
    legacy = (
        payload.get("schema") == SILICON_RELEASE_EVIDENCE_SCHEMA
        and payload.get("schema_version")
        == SILICON_RELEASE_EVIDENCE_SCHEMA_VERSION
    )
    expected_keys = base_keys | (
        {"external_alias_manifest"} if alias_enabled else set()
    )
    if (
        set(payload) != expected_keys
        or not (legacy or alias_enabled)
    ):
        raise SiliconReleaseEvidenceError(
            "portable evidence payload schema is incomplete or unsupported."
        )
    recorded_root = _absolute_posix(
        payload["recorded_artifact_root"],
        name="recorded_artifact_root",
    )
    expected_semantics = {
        "inventory_paths": "relative_posix_from_artifact_root",
        "embedded_absolute_paths": (
            "resolve_recorded_root_or_bound_external_alias"
            if alias_enabled
            else "remap_recorded_artifact_root_to_verified_artifact_root"
        ),
        "portable_identity_fields": [
            "relative_path",
            "size_bytes",
            "sha256",
        ],
        "excluded_machine_identity_fields": [
            "absolute_current_path",
            "device",
            "inode",
            "mtime_ns",
        ],
    }
    if payload["path_semantics"] != expected_semantics:
        raise SiliconReleaseEvidenceError(
            "portable evidence path semantics are not pinned."
        )

    external_alias_record: dict[str, Any] | None = None
    if alias_enabled:
        raw_alias = payload["external_alias_manifest"]
        alias_keys = {
            "relative_path",
            "sha256",
            "size_bytes",
            "payload_sha256",
            "alias_count",
            "object_count",
            "total_size_bytes",
        }
        if not isinstance(raw_alias, dict) or set(raw_alias) != alias_keys:
            raise SiliconReleaseEvidenceError(
                "portable external-alias identity is ambiguous."
            )
        external_alias_record = dict(raw_alias)
        external_alias_record["relative_path"] = _safe_relative_posix(
            raw_alias["relative_path"],
            name="external_alias_manifest.relative_path",
        )
        external_alias_record["sha256"] = _require_sha256(
            raw_alias["sha256"],
            name="external_alias_manifest.sha256",
        )
        external_alias_record["payload_sha256"] = _require_sha256(
            raw_alias["payload_sha256"],
            name="external_alias_manifest.payload_sha256",
        )
        for name in (
            "size_bytes",
            "alias_count",
            "object_count",
            "total_size_bytes",
        ):
            item = raw_alias[name]
            if (
                isinstance(item, bool)
                or not isinstance(item, int)
                or item < 0
                or (
                    name in {"alias_count", "object_count"}
                    and item == 0
                )
            ):
                raise SiliconReleaseEvidenceError(
                    f"external_alias_manifest.{name} is invalid."
                )
            external_alias_record[name] = item

    raw_inventory = payload["inventory"]
    if not isinstance(raw_inventory, list) or not raw_inventory:
        raise SiliconReleaseEvidenceError(
            "portable evidence inventory must be nonempty."
        )
    inventory: list[dict[str, Any]] = []
    names: list[str] = []
    for position, raw in enumerate(raw_inventory):
        if not isinstance(raw, dict) or set(raw) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise SiliconReleaseEvidenceError(
                f"inventory[{position}] has an ambiguous schema."
            )
        relative = _safe_relative_posix(
            raw["relative_path"],
            name=f"inventory[{position}].relative_path",
        )
        size = raw["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SiliconReleaseEvidenceError(
                f"inventory[{position}].size_bytes is invalid."
            )
        inventory.append(
            {
                "relative_path": relative,
                "size_bytes": size,
                "sha256": _require_sha256(
                    raw["sha256"],
                    name=f"inventory[{position}].sha256",
                ),
            }
        )
        names.append(relative)
    if names != sorted(set(names)):
        raise SiliconReleaseEvidenceError(
            "portable evidence inventory paths are not unique canonical order."
        )
    if payload["inventory_sha256"] != _payload_sha256(inventory):
        raise SiliconReleaseEvidenceError(
            "portable evidence inventory digest is inconsistent."
        )

    report = payload["release_report"]
    expected_report_keys = {
        "relative_path",
        "sha256",
        "size_bytes",
        "schema",
        "schema_version",
        "status",
        "validation_passed",
        "certification_eligible",
        "release_certified",
        "release_version",
    }
    if not isinstance(report, dict) or set(report) != expected_report_keys:
        raise SiliconReleaseEvidenceError(
            "portable release-report identity has an ambiguous schema."
        )
    report = dict(report)
    report["relative_path"] = _safe_relative_posix(
        report["relative_path"],
        name="release_report.relative_path",
    )
    report["sha256"] = _require_sha256(
        report["sha256"],
        name="release_report.sha256",
    )
    if (
        isinstance(report["size_bytes"], bool)
        or not isinstance(report["size_bytes"], int)
        or report["size_bytes"] < 0
        or report["schema"] != SILICON_RELEASE_REPORT_SCHEMA
        or report["schema_version"] != 1
        or report["status"] != "passed"
        or report["validation_passed"] is not True
        or report["certification_eligible"] is not True
        or report["release_certified"] is not True
        or report["release_version"] != SILICON_RELEASE_VERSION
    ):
        raise SiliconReleaseEvidenceError(
            "portable release-report certification identity is invalid."
        )

    summary = payload["content_summary"]
    expected_summary_keys = {
        "file_count",
        "total_size_bytes",
        "json_file_count",
        "path_record_count",
        "hdf5_file_count",
    }
    if (
        not isinstance(summary, dict)
        or set(summary) != expected_summary_keys
        or any(
            isinstance(summary[name], bool)
            or not isinstance(summary[name], int)
            or summary[name] < 0
            for name in expected_summary_keys
        )
        or summary["file_count"] != len(inventory)
        or summary["total_size_bytes"] != sum(item["size_bytes"] for item in inventory)
    ):
        raise SiliconReleaseEvidenceError(
            "portable evidence content summary is inconsistent."
        )
    return (
        recorded_root,
        inventory,
        report,
        dict(summary),
        external_alias_record,
    )


def verify_silicon_release_evidence(
    artifact_root: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    expected_manifest_sha256: str | None = None,
    expected_release_report_sha256: str | None = None,
) -> PortableSiliconReleaseEvidence:
    """Verify sealed evidence at its original or a relocated artifact root.

    Without both expected digests this proves only self-consistent integrity
    and relocation.  Passing digests obtained through the trusted transfer
    channel additionally authenticates the portable manifest and final report
    bytes; it still does not rerun the scientific validators.
    """

    if (expected_manifest_sha256 is None) is not (
        expected_release_report_sha256 is None
    ):
        raise SiliconReleaseEvidenceError(
            "trusted verification requires both the manifest and release-"
            "report SHA256 digests."
        )
    trusted = expected_manifest_sha256 is not None
    if trusted:
        expected_manifest_sha256 = _require_sha256(
            expected_manifest_sha256,
            name="expected_manifest_sha256",
        )
        expected_release_report_sha256 = _require_sha256(
            expected_release_report_sha256,
            name="expected_release_report_sha256",
        )

    root = _directory_without_symlink(
        artifact_root,
        name="artifact_root",
    )
    manifest = _regular_file_without_symlink(
        manifest_path,
        name="manifest_path",
    )
    if manifest == root or manifest.is_relative_to(root):
        raise SiliconReleaseEvidenceError(
            "manifest_path must remain outside the sealed artifact root."
        )
    payload, payload_digest = _load_manifest(manifest)
    (
        recorded_root,
        declared_inventory,
        declared_report,
        declared_summary,
        declared_external_alias,
    ) = _validate_manifest_payload(payload)

    snapshots = _scan_artifact_tree(root)
    observed_inventory = [snapshot.inventory_record() for snapshot in snapshots]
    if observed_inventory != declared_inventory:
        declared_names = {item["relative_path"] for item in declared_inventory}
        observed_names = {item["relative_path"] for item in observed_inventory}
        missing = sorted(declared_names - observed_names)
        extra = sorted(observed_names - declared_names)
        if missing or extra:
            raise SiliconReleaseEvidenceError(
                "artifact membership differs from the portable manifest; "
                f"missing={missing!r}, extra={extra!r}."
            )
        raise SiliconReleaseEvidenceError(
            "artifact size or SHA256 differs from the portable manifest."
        )

    by_relative = _inventory_mapping(snapshots)
    external_aliases: _ExternalAliasIndex | None = None
    if declared_external_alias is not None:
        alias_snapshot = by_relative.get(
            declared_external_alias["relative_path"]
        )
        if (
            alias_snapshot is None
            or alias_snapshot.sha256
            != declared_external_alias["sha256"]
            or alias_snapshot.size_bytes
            != declared_external_alias["size_bytes"]
        ):
            raise SiliconReleaseEvidenceError(
                "bound external alias manifest differs from the portable "
                "inventory."
            )
        external_aliases, alias_summary = (
            _load_external_alias_manifest(
                root,
                alias_snapshot.path,
                expected_recorded_root=recorded_root,
            )
        )
        observed_alias_record = {
            "relative_path": (
                external_aliases.alias_manifest_relative_path
            ),
            "sha256": external_aliases.manifest_sha256,
            "size_bytes": alias_snapshot.size_bytes,
            "payload_sha256": external_aliases.payload_sha256,
            **alias_summary,
        }
        if observed_alias_record != declared_external_alias:
            raise SiliconReleaseEvidenceError(
                "bound external alias identity differs from the portable "
                "manifest."
            )

    audit = _audit_artifact_contents(
        snapshots,
        recorded_root=recorded_root,
        external_aliases=external_aliases,
    )
    observed_summary = {
        "file_count": len(snapshots),
        "total_size_bytes": sum(item.size_bytes for item in snapshots),
        **audit,
    }
    if observed_summary != declared_summary:
        raise SiliconReleaseEvidenceError(
            "relocated evidence audit counts differ from the portable manifest."
        )

    report_snapshot = by_relative.get(declared_report["relative_path"])
    if report_snapshot is None:
        raise SiliconReleaseEvidenceError(
            "sealed release report is absent from the relocated inventory."
        )
    observed_report = _release_report_identity(
        report_snapshot.path,
        relative_path=report_snapshot.relative_path,
        snapshot=report_snapshot,
    )
    if observed_report != declared_report:
        raise SiliconReleaseEvidenceError(
            "relocated final aggregate report differs from the sealed identity."
        )
    _recheck_snapshots(snapshots)

    manifest_sha256 = _file_sha256(manifest)
    if trusted and manifest_sha256 != expected_manifest_sha256:
        raise SiliconReleaseEvidenceError(
            "portable manifest differs from the trusted SHA256."
        )
    if (
        trusted
        and declared_report["sha256"]
        != expected_release_report_sha256
    ):
        raise SiliconReleaseEvidenceError(
            "final release report differs from the trusted SHA256."
        )
    return PortableSiliconReleaseEvidence(
        manifest_path=manifest,
        manifest_sha256=manifest_sha256,
        payload_sha256=payload_digest,
        recorded_artifact_root=recorded_root,
        verified_artifact_root=root,
        release_report_relative_path=declared_report["relative_path"],
        release_report_sha256=declared_report["sha256"],
        file_count=declared_summary["file_count"],
        total_size_bytes=declared_summary["total_size_bytes"],
        json_file_count=declared_summary["json_file_count"],
        path_record_count=declared_summary["path_record_count"],
        hdf5_file_count=declared_summary["hdf5_file_count"],
        verification_scope=(
            "trusted_digest_authenticated_integrity"
            if trusted
            else "self_authenticated_integrity_and_relocation_only"
        ),
        trusted_manifest_sha256_matched=trusted,
        trusted_release_report_sha256_matched=trusted,
    )


def finalize_silicon_release_evidence(
    artifact_root: str | os.PathLike[str],
    release_report_path: str | os.PathLike[str],
    handoff_directory: str | os.PathLike[str],
) -> SiliconReleaseHandoffEvidence:
    """Atomically publish a portable archival handoff.

    This operation stages external evidence, seals and self-verifies the
    artifact tree, and publishes a three-file handoff directory containing
    the portable manifest, an exact release-report copy, and ``SHA256SUMS``.
    It verifies an existing scientific decision; it does not recompute one.
    """

    root = _directory_without_symlink(
        artifact_root,
        name="artifact_root",
    )
    report = _regular_file_without_symlink(
        release_report_path,
        name="release_report_path",
    )
    destination = _handoff_destination(root, handoff_directory)
    staging_root = root / _EXTERNAL_STAGING_DIRECTORY
    if staging_root.exists() or staging_root.is_symlink():
        raise FileExistsError(
            "artifact_root already contains an external-evidence staging "
            "directory; finalize requires a pristine unsealed tree."
        )
    try:
        report_relative = report.relative_to(root).as_posix()
    except ValueError as exc:
        raise SiliconReleaseEvidenceError(
            "release_report_path must lie inside artifact_root."
        ) from exc
    report_relative = _safe_relative_posix(
        report_relative,
        name="release_report_path",
    )
    initial_snapshots = _scan_artifact_tree(root)
    initial_by_relative = _inventory_mapping(initial_snapshots)
    report_snapshot = initial_by_relative.get(report_relative)
    if report_snapshot is None:
        raise SiliconReleaseEvidenceError(
            "release report is absent from the complete artifact inventory."
        )
    _release_report_identity(
        report,
        relative_path=report_relative,
        snapshot=report_snapshot,
    )
    external_sources = _discover_external_sources(
        root,
        initial_snapshots,
        require_nonempty=False,
    )
    _recheck_snapshots(initial_snapshots)

    temporary = destination.parent / (
        f".{destination.name}.{uuid4().hex}.tmp"
    )
    os.mkdir(temporary, 0o700)
    staged: StagedSiliconExternalEvidence | None = None
    published = False
    try:
        if external_sources:
            staged = stage_external_silicon_release_evidence(root)
        temporary_manifest = temporary / _HANDOFF_MANIFEST_FILENAME
        portable = seal_silicon_release_evidence(
            root,
            report,
            temporary_manifest,
            external_alias_manifest_path=(
                None if staged is None else staged.alias_manifest_path
            ),
        )
        if portable.release_report_sha256 != report_snapshot.sha256:
            raise SiliconReleaseEvidenceError(
                "release report changed between finalize preflight and seal."
            )
        report_source = _ExternalSource(
            source_path=report.as_posix(),
            path=report,
            size_bytes=report_snapshot.size_bytes,
            sha256=report_snapshot.sha256,
            stat_signature=report_snapshot.stat_signature,
            json_evidence=True,
        )
        temporary_report = (
            temporary / _HANDOFF_RELEASE_REPORT_FILENAME
        )
        _copy_external_source(report_source, temporary_report)
        checksum_payload = _handoff_checksum_bytes(
            portable.manifest_sha256,
            portable.release_report_sha256,
        )
        temporary_checksum = temporary / _HANDOFF_CHECKSUM_FILENAME
        _write_atomic_exclusive(
            temporary_checksum,
            checksum_payload,
        )
        (
            _,
            _,
            checked_manifest,
            checked_report,
            manifest_digest,
            report_digest,
            checksum_digest,
        ) = _verified_handoff_files(temporary_checksum)
        if (
            checked_manifest != temporary_manifest
            or checked_report != temporary_report
            or manifest_digest != portable.manifest_sha256
            or report_digest != portable.release_report_sha256
        ):
            raise SiliconReleaseEvidenceError(
                "prepared archival handoff differs from the sealed evidence."
            )
        portable = verify_silicon_release_evidence(
            root,
            temporary_manifest,
        )
        if (
            portable.verification_scope
            != "self_authenticated_integrity_and_relocation_only"
        ):
            raise SiliconReleaseEvidenceError(
                "finalize self-verification reported an unexpected scope."
            )
        directory_descriptor = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        final_manifest = destination / _HANDOFF_MANIFEST_FILENAME
        final_report = destination / _HANDOFF_RELEASE_REPORT_FILENAME
        final_checksum = destination / _HANDOFF_CHECKSUM_FILENAME
        result = SiliconReleaseHandoffEvidence(
            action="finalized",
            handoff_directory=destination,
            checksum_path=final_checksum,
            checksum_sha256=checksum_digest,
            release_report_copy_path=final_report,
            external_evidence_staged=staged is not None,
            portable_evidence=replace(
                portable,
                manifest_path=final_manifest,
            ),
        )
        _publish_directory_noreplace(temporary, destination)
        published = True
        parent_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        return result
    except Exception:
        if not published:
            _remove_owned_directory(temporary)
            if staged is not None:
                _remove_owned_directory(staging_root)
        raise


def verify_silicon_release_handoff(
    artifact_root: str | os.PathLike[str],
    trusted_checksum_path: str | os.PathLike[str],
) -> SiliconReleaseHandoffEvidence:
    """Verify a relocated archive using a trusted ``SHA256SUMS`` handoff.

    The checksum file must arrive through an authenticated transfer channel.
    This performs trusted-digest archival verification only and never reruns
    the scientific silicon validators.
    """

    (
        handoff_directory,
        checksum,
        manifest,
        report_copy,
        manifest_digest,
        report_digest,
        checksum_digest,
    ) = _verified_handoff_files(trusted_checksum_path)
    portable = verify_silicon_release_evidence(
        artifact_root,
        manifest,
        expected_manifest_sha256=manifest_digest,
        expected_release_report_sha256=report_digest,
    )
    if (
        portable.verification_scope
        != "trusted_digest_authenticated_integrity"
        or portable.release_report_sha256 != report_digest
    ):
        raise SiliconReleaseEvidenceError(
            "trusted handoff did not authenticate the portable release."
        )
    payload, _ = _load_manifest(manifest)
    return SiliconReleaseHandoffEvidence(
        action="retrieval_verified",
        handoff_directory=handoff_directory,
        checksum_path=checksum,
        checksum_sha256=checksum_digest,
        release_report_copy_path=report_copy,
        external_evidence_staged=(
            "external_alias_manifest" in payload
        ),
        portable_evidence=portable,
    )


__all__ = [
    "PortableSiliconReleaseEvidence",
    "SiliconReleaseHandoffEvidence",
    "StagedSiliconExternalEvidence",
    "SILICON_EXTERNAL_ALIAS_SCHEMA",
    "SILICON_EXTERNAL_ALIAS_SCHEMA_VERSION",
    "SILICON_RELEASE_EVIDENCE_SCHEMA",
    "SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA",
    "SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA_VERSION",
    "SILICON_RELEASE_EVIDENCE_SCHEMA_VERSION",
    "SiliconReleaseEvidenceError",
    "finalize_silicon_release_evidence",
    "seal_silicon_release_evidence",
    "stage_external_silicon_release_evidence",
    "verify_silicon_release_evidence",
    "verify_silicon_release_handoff",
]
