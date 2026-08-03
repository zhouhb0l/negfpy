"""Authenticated silicon Phase-A fixture preparation and verification.

This module implements the evidence-only boundary between the sealed silicon
``0.2.0`` bulk-oracle release and the sparse :math:`N_q=1` work.  It does not
construct a material device and it does not read FC2 or FC3 tensors.  Its job
is narrower: authenticate the sealed archive, copy the exact Phase-A
projection, reconstruct the FC2/FC3 crystal mappings independently, and
publish a deterministic fixture without overwriting an existing directory.

The frozen values below intentionally make the default implementation useful
for exactly one sealed release.  Tests may replace them with a fully synthetic
seal, but production callers cannot provide alternate hashes on the command
line.
"""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
from dataclasses import dataclass
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import tempfile
from typing import Any

from ._canonical import (
    crystal_mapping_content_sha256,
    primitive_crystal_content_sha256,
)
from .io.phono3py_crystal import read_phono3py_crystal_mapping
from .silicon_oracle_validation import crystal_mapping_sha256


SOURCE_COMMIT = "7f23c85711eab4f425c1dcfa3d4538859bba0209"
SOURCE_TREE = "8a7119629b2e40a1893ac2e56114bc8c77188150"
SOURCE_RUN_ID = "si-oracle-005"
PARENT_INPUT_MANIFEST_SHA256 = (
    "eb0058800a458d09630f54ce4883beb20b92391c389290c0674bbba15f4b9642"
)

TERMINAL_REPORT_SHA256 = (
    "da837917d52e79c52af208654d11a8f98bc0e7e8aedd6fd30642fdbd3e639e0a"
)
PORTABLE_MANIFEST_SHA256 = (
    "5a29f9f26f4719eae67828471b74794f4a571a2a43bb12cd0c94e6afb4150902"
)
PORTABLE_PAYLOAD_SHA256 = (
    "847e368ceffeeeecb6d233af9d869034ddab35cb1be60ff405cba4d717fc17cf"
)
HANDOFF_SHA256SUMS_SHA256 = (
    "e114e5ba9a52df2ff47b1d127647631c3cc0f50aa9d0f6a7240d6a549da8b2b4"
)
EXTERNAL_ALIAS_MANIFEST_SHA256 = (
    "c0b3eb256030aa56b7c76521f3b3c5ff9072692411a728ff48c45c992f7757ae"
)
EXTERNAL_ALIAS_PAYLOAD_SHA256 = (
    "206d99ba574c2bcf3f953d62a8aae77e92649568b2a3d58e0be549399415a779"
)

HISTORICAL_MAPPING_SHA256 = {
    "fc2": "e62d4eed227281c2a4a4bf94f7b80db61a9e20a655f3ee7c35083c764b424fa0",
    "fc3": "9f2d489638de7f38aa57b6730772316410d4aaf3db00f0b268ea843c5ffd6ba8",
}

RAW_FORCE_CONSTANT_SHA256 = {
    "fc2.compact.hdf5": (
        "1a67145ab28e792a3aa8a8c432c9fbebc15d0b781f60da24ede582a1581b9cd0"
    ),
    "fc2.full.hdf5": (
        "6b26d489e6a7fffe61530e77b0c2a907eb4e4f13d5b352abb29d9c0c2c6c74f7"
    ),
    "fc3.compact.hdf5": (
        "c7d0902205d9ae48e657de53400c5c6acd0cdd0822804ac8c8dc154b29c15ffd"
    ),
    "fc3.full.hdf5": (
        "e4e297963b4c2a2a98953b140217baa1a7461b18cb05f12126ae545fe572cd73"
    ),
}


@dataclass(frozen=True, slots=True)
class _RoleSpec:
    target_relative_path: str
    source_relative_path: str | None
    origin_kind: str


ROLE_SPECS: dict[str, _RoleSpec] = {
    "fc2_compact": _RoleSpec(
        "fc2.compact.hdf5", "force-constants/fc2.compact.hdf5", "sealed_artifact"
    ),
    "fc2_full": _RoleSpec(
        "fc2.full.hdf5", "force-constants/fc2.full.hdf5", "sealed_artifact"
    ),
    "force_constants_sha256sums": _RoleSpec(
        "force-constants-SHA256SUMS",
        "force-constants/force-constants-SHA256SUMS",
        "sealed_artifact",
    ),
    "force_constants_stage": _RoleSpec(
        "force-constants-stage.json",
        "force-constants/force-constants-stage.json",
        "sealed_artifact",
    ),
    "input_manifest_normalized": _RoleSpec(
        "input-manifest.normalized.json",
        "force-constants/input-manifest.normalized.json",
        "sealed_artifact",
    ),
    "phono3py_yaml": _RoleSpec(
        "phono3py_disp.yaml", None, "sealed_external_alias"
    ),
    "release_report": _RoleSpec(
        "silicon-release-report.json",
        "reports/silicon-release-report.json",
        "sealed_artifact",
    ),
    "source_verification": _RoleSpec(
        "source-verification.json",
        "force-constants/source-verification.json",
        "sealed_artifact",
    ),
    "v3_sample_manifest": _RoleSpec(
        "v3/mesh-3x3x3/samples.json",
        "v3/mesh-3x3x3/samples.json",
        "sealed_artifact",
    ),
}

EXPECTED_FILE_SHA256 = {
    "fc2_compact": RAW_FORCE_CONSTANT_SHA256["fc2.compact.hdf5"],
    "fc2_full": RAW_FORCE_CONSTANT_SHA256["fc2.full.hdf5"],
    "force_constants_sha256sums": (
        "eeabd3ab69b42cd061cd454a29d4271fa2ac71a375b41d8faff66264566cd4da"
    ),
    "force_constants_stage": (
        "ce7786702e6371e93dccce2fee7da76469c96d102035191f6f6fda1db0d722bf"
    ),
    "input_manifest_normalized": (
        "ad1f975a5215c073d3cb63d5b99a6cd2e996394f6ca03242fee392bad763e713"
    ),
    "phono3py_yaml": (
        "1375e3ae19a2c4138a888af8177a3becfcaa3b127ca50755773c405c3e28a2a7"
    ),
    "release_report": TERMINAL_REPORT_SHA256,
    "source_verification": (
        "46e41a246a7cad3195ae2bdc1a12d74037e24b7d2a2cfdd33d2832f6feea6fac"
    ),
    "v3_sample_manifest": (
        "13e19788499dd1ad497bc7aa901f0fbf032397a4f0edd4201286a982fd127845"
    ),
}

HISTORICAL_CONSISTENCY_FILES = {
    "v3/mesh-3x3x3/samples.json": EXPECTED_FILE_SHA256["v3_sample_manifest"],
    "v3/mesh-5x5x5/samples.json": (
        "f58c69fb8fe828c9a910388746f0c1bf05c861eb3290ed15a627fe4372598eed"
    ),
    "v3/mesh-11x11x11/samples.json": (
        "2a26c94172daeb7be1fd7c491e67067cfc3c5934ae5314a6e9448cd35b2ac177"
    ),
    "qualifications/mesh-3x3x3/serial-report.json": (
        "40efc24876ba052610b8de6dafc76f8675702d218758cfecaccb45e443e08ac2"
    ),
    "qualifications/mesh-5x5x5/serial-report.json": (
        "1b525f8999523a24afad1f4d4223f8a1e87ab5f995add9d6ad742b873431181a"
    ),
}

_FIXTURE_MANIFEST = "fixture-manifest.json"
_COPY_RECEIPT = "fixture-copy-receipt.json"
_DERIVATION_RECEIPT = "sealed-input-derivation-receipt.json"
_FIXTURE_SUMS = "SHA256SUMS"
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class PhaseAFixtureError(RuntimeError):
    """Raised when a sealed evidence or fixture invariant does not hold."""


def _fail(message: str) -> None:
    raise PhaseAFixtureError(message)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_payload_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PhaseAFixtureError("JSON value is not canonically serializable.") from exc


def _canonical_json_bytes(value: object) -> bytes:
    return _canonical_payload_bytes(value) + b"\n"


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON token {value!r}")


def _pairs_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load_json(data: bytes, *, name: str, canonical: bool = False) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise PhaseAFixtureError(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        _fail(f"{name} must contain a JSON object.")
    if canonical and data != _canonical_json_bytes(value):
        _fail(f"{name} is not in the frozen canonical JSON representation.")
    return value


def _exact_keys(value: Mapping[str, Any], keys: set[str], *, name: str) -> None:
    actual = set(value)
    if actual != keys:
        _fail(
            f"{name} keys differ from the frozen schema; "
            f"missing={sorted(keys - actual)!r}, extra={sorted(actual - keys)!r}."
        )


def _digest(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        _fail(f"{name} is not a lowercase SHA256 digest.")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{name} must be a nonnegative integer.")
    return value


def _relative_path(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        _fail(f"{name} must be a nonempty relative POSIX path.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        _fail(f"{name} is not a safe relative POSIX path.")
    if path.as_posix() != value:
        _fail(f"{name} is not a normalized relative POSIX path.")
    return value


def _assert_plain_root(path: Path, *, name: str) -> Path:
    absolute = path.expanduser().absolute()
    try:
        mode = absolute.lstat().st_mode
    except FileNotFoundError as exc:
        raise PhaseAFixtureError(f"{name} does not exist: {absolute}") from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        _fail(f"{name} must be a real directory, not a symlink: {absolute}")
    return absolute


def _safe_child(root: Path, relative: str, *, name: str) -> Path:
    relative = _relative_path(relative, name=name)
    current = root
    for part in PurePosixPath(relative).parts:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError as exc:
            raise PhaseAFixtureError(f"{name} is missing: {current}") from exc
        if stat.S_ISLNK(mode):
            _fail(f"{name} traverses a symlink: {current}")
    return current


def _read_regular(path: Path, *, name: str) -> bytes:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PhaseAFixtureError(
            f"Cannot open {name} as a regular file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{name} is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(f"{name} changed while it was being read: {path}")
        data = b"".join(chunks)
        if len(data) != before.st_size:
            _fail(f"{name} byte count changed while it was being read: {path}")
        return data
    finally:
        os.close(descriptor)


def _hash_regular(path: Path, *, name: str) -> tuple[str, int]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise PhaseAFixtureError(
            f"Cannot open {name} as a regular file: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _fail(f"{name} is not a regular file: {path}")
        digest = hashlib.sha256()
        count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            count += len(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            _fail(f"{name} changed while it was being hashed: {path}")
        if count != before.st_size:
            _fail(f"{name} byte count changed while it was being hashed: {path}")
        return digest.hexdigest(), count
    finally:
        os.close(descriptor)


def _verify_bytes(data: bytes, expected: str, *, name: str) -> None:
    actual = _sha256_bytes(data)
    if actual != expected:
        _fail(f"{name} SHA256 mismatch: expected {expected}, found {actual}.")


def _envelope(
    data: bytes,
    *,
    name: str,
    expected_file_sha256: str,
    expected_payload_sha256: str,
) -> dict[str, Any]:
    _verify_bytes(data, expected_file_sha256, name=name)
    document = _load_json(data, name=name, canonical=True)
    _exact_keys(document, {"payload", "payload_sha256"}, name=name)
    payload = document["payload"]
    if not isinstance(payload, dict):
        _fail(f"{name}.payload must be an object.")
    payload_sha256 = _digest(document["payload_sha256"], name=f"{name}.payload_sha256")
    actual_payload_sha256 = _sha256_bytes(_canonical_payload_bytes(payload))
    if payload_sha256 != actual_payload_sha256:
        _fail(f"{name} payload digest does not reproduce.")
    if payload_sha256 != expected_payload_sha256:
        _fail(f"{name} payload does not match the sealed digest.")
    return payload


def _parse_sha256sums(data: bytes, *, name: str) -> dict[str, str]:
    try:
        text = data.decode("ascii")
    except UnicodeDecodeError as exc:
        raise PhaseAFixtureError(f"{name} is not ASCII.") from exc
    if not text or not text.endswith("\n"):
        _fail(f"{name} must be nonempty and end with exactly one LF.")
    records: dict[str, str] = {}
    for line_number, line in enumerate(text.splitlines(), start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            _fail(f"{name} line {line_number} is not standard SHA256SUMS syntax.")
        digest, relative = match.groups()
        relative = _relative_path(relative, name=f"{name} line {line_number} path")
        if relative in records:
            _fail(f"{name} contains duplicate path {relative!r}.")
        records[relative] = digest
    if list(records) != sorted(records):
        _fail(f"{name} paths are not lexicographically sorted.")
    canonical = "".join(f"{records[path]}  {path}\n" for path in sorted(records))
    if canonical.encode("ascii") != data:
        _fail(f"{name} is not in canonical standard two-space form.")
    return records


def _inventory_index(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inventory = payload.get("inventory")
    if not isinstance(inventory, list):
        _fail("portable manifest inventory must be an array.")
    index: dict[str, dict[str, Any]] = {}
    for position, record in enumerate(inventory):
        if not isinstance(record, dict):
            _fail(f"portable manifest inventory[{position}] must be an object.")
        _exact_keys(
            record,
            {"relative_path", "sha256", "size_bytes"},
            name=f"portable manifest inventory[{position}]",
        )
        relative = _relative_path(
            record["relative_path"],
            name=f"portable manifest inventory[{position}].relative_path",
        )
        _digest(record["sha256"], name=f"portable inventory {relative} digest")
        _nonnegative_int(
            record["size_bytes"], name=f"portable inventory {relative} size"
        )
        if relative in index:
            _fail(f"portable manifest has duplicate inventory path {relative!r}.")
        index[relative] = record
    if list(index) != sorted(index):
        _fail("portable manifest inventory is not sorted by relative_path.")
    declared_digest = _digest(
        payload.get("inventory_sha256"),
        name="portable manifest inventory_sha256",
    )
    actual_digest = _sha256_bytes(_canonical_payload_bytes(inventory))
    if declared_digest != actual_digest:
        _fail("portable manifest inventory_sha256 does not reproduce.")
    summary = payload.get("content_summary")
    if not isinstance(summary, dict):
        _fail("portable manifest content_summary must be an object.")
    if _nonnegative_int(summary.get("file_count"), name="portable file_count") != len(
        inventory
    ):
        _fail("portable manifest file_count does not equal inventory length.")
    return index


def _require_inventory(
    index: Mapping[str, Mapping[str, Any]],
    relative: str,
    sha256: str,
    size_bytes: int,
) -> None:
    record = index.get(relative)
    if record is None:
        _fail(f"portable manifest omits required path {relative!r}.")
    if record["sha256"] != sha256 or record["size_bytes"] != size_bytes:
        _fail(f"portable manifest identity mismatch for {relative!r}.")


def _portable_manifest(data: bytes) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    payload = _envelope(
        data,
        name="portable manifest",
        expected_file_sha256=PORTABLE_MANIFEST_SHA256,
        expected_payload_sha256=PORTABLE_PAYLOAD_SHA256,
    )
    _exact_keys(
        payload,
        {
            "content_summary",
            "external_alias_manifest",
            "inventory",
            "inventory_sha256",
            "path_semantics",
            "recorded_artifact_root",
            "release_report",
            "schema",
            "schema_version",
        },
        name="portable manifest payload",
    )
    if payload["schema"] != "negfpy.silicon_release_portable_evidence.v2":
        _fail("portable manifest schema is not the sealed version-2 schema.")
    if payload["schema_version"] != 2:
        _fail("portable manifest schema_version is not 2.")
    recorded_root = payload["recorded_artifact_root"]
    if not isinstance(recorded_root, str) or not recorded_root.startswith("/"):
        _fail("portable manifest recorded_artifact_root must be absolute POSIX.")
    if PurePosixPath(recorded_root).name != SOURCE_RUN_ID:
        _fail("portable manifest recorded root does not end in the sealed run ID.")
    index = _inventory_index(payload)
    return payload, index


def _release_report(data: bytes) -> dict[str, Any]:
    _verify_bytes(data, TERMINAL_REPORT_SHA256, name="terminal release report")
    report = _load_json(data, name="terminal release report")
    required = {
        "schema": "negfpy.silicon_0_2_0_release_evidence.v1",
        "schema_version": 1,
        "release_version": "0.2.0",
        "status": "passed",
        "release_certified": True,
        "validation_passed": True,
        "certification_eligible": True,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            _fail(f"terminal release report field {key!r} is not {expected!r}.")
    source = report.get("source_repository")
    if not isinstance(source, dict):
        _fail("terminal release report source_repository must be an object.")
    if source.get("head_commit") != SOURCE_COMMIT:
        _fail("terminal release report commit differs from the sealed commit.")
    if source.get("tree_hash") != SOURCE_TREE:
        _fail("terminal release report tree differs from the sealed tree.")
    if source.get("clean") is not True or source.get("dirty_entry_count") != 0:
        _fail("terminal release report source repository is not clean.")
    inventory = report.get("evidence_inventory")
    if not isinstance(inventory, dict):
        _fail("terminal release report evidence_inventory must be an object.")
    _exact_keys(inventory, {"file_count", "files"}, name="terminal evidence_inventory")
    files = inventory["files"]
    if not isinstance(files, list):
        _fail("terminal evidence_inventory.files must be an array.")
    if _nonnegative_int(
        inventory["file_count"], name="terminal evidence file_count"
    ) != len(files):
        _fail("terminal evidence file_count does not equal files length.")
    return report


def _terminal_inventory_index(
    report: Mapping[str, Any], *, recorded_root: str
) -> dict[str, Mapping[str, Any]]:
    prefix = recorded_root.rstrip("/") + "/"
    index: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(report["evidence_inventory"]["files"]):
        if not isinstance(record, dict):
            _fail(f"terminal evidence file {position} must be an object.")
        path = record.get("path")
        if not isinstance(path, str) or not path.startswith(prefix):
            continue
        relative = path[len(prefix) :]
        try:
            relative = _relative_path(relative, name="terminal evidence relative path")
        except PhaseAFixtureError:
            continue
        if relative in index:
            _fail(f"terminal evidence inventory duplicates {relative!r}.")
        index[relative] = record
    return index


def _recorded_root_from_terminal(report: Mapping[str, Any]) -> str:
    """Recover one common sealed root from the five authenticated records."""

    files = report["evidence_inventory"]["files"]
    roots: set[str] = set()
    for relative, expected_digest in HISTORICAL_CONSISTENCY_FILES.items():
        suffix = "/" + relative
        matches = [
            record
            for record in files
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and record["path"].endswith(suffix)
            and record.get("sha256") == expected_digest
        ]
        if len(matches) != 1:
            _fail(
                "terminal report must contain exactly one authenticated record "
                f"for {relative!r}."
            )
        roots.add(matches[0]["path"][: -len(suffix)])
    if len(roots) != 1:
        _fail("terminal historical evidence records do not share one artifact root.")
    recorded_root = roots.pop()
    if not recorded_root.startswith("/") or PurePosixPath(recorded_root).name != (
        SOURCE_RUN_ID
    ):
        _fail("terminal historical evidence root does not end in the sealed run ID.")
    return recorded_root


def _mapping_pair(
    document: Mapping[str, Any], *, qualification: bool, name: str
) -> dict[str, str]:
    value: Any = document
    if qualification:
        value = document.get("interaction_sample_manifest")
        if not isinstance(value, dict):
            _fail(f"{name}.interaction_sample_manifest must be an object.")
    value = value.get("mapping_sha256")
    if not isinstance(value, dict):
        _fail(f"{name} mapping_sha256 must be an object.")
    _exact_keys(value, {"fc2", "fc3"}, name=f"{name}.mapping_sha256")
    pair = {
        "fc2": _digest(value["fc2"], name=f"{name} fc2 mapping digest"),
        "fc3": _digest(value["fc3"], name=f"{name} fc3 mapping digest"),
    }
    return pair


def _verify_historical_chain(
    *,
    artifact_root: Path,
    recorded_root: str,
    report: Mapping[str, Any],
    portable_index: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, str], bytes]:
    terminal_index = _terminal_inventory_index(report, recorded_root=recorded_root)
    authoritative_pair: dict[str, str] | None = None
    sample_bytes: bytes | None = None
    for relative, expected_digest in HISTORICAL_CONSISTENCY_FILES.items():
        terminal = terminal_index.get(relative)
        if terminal is None:
            _fail(f"terminal evidence inventory omits {relative!r}.")
        if terminal.get("sha256") != expected_digest:
            _fail(f"terminal evidence digest mismatch for {relative!r}.")
        expected_size = _nonnegative_int(
            terminal.get("size_bytes"), name=f"terminal evidence size for {relative}"
        )
        source_path = _safe_child(artifact_root, relative, name=relative)
        data = _read_regular(source_path, name=relative)
        _verify_bytes(data, expected_digest, name=relative)
        if len(data) != expected_size:
            _fail(f"terminal evidence size mismatch for {relative!r}.")
        _require_inventory(portable_index, relative, expected_digest, len(data))
        document = _load_json(data, name=relative)
        pair = _mapping_pair(
            document,
            qualification=relative.startswith("qualifications/"),
            name=relative,
        )
        if pair != HISTORICAL_MAPPING_SHA256:
            _fail(f"historical mapping values differ in {relative!r}.")
        if authoritative_pair is None:
            authoritative_pair = pair
        elif pair != authoritative_pair:
            _fail("historical mapping manifests do not agree with each other.")
        if relative == "v3/mesh-3x3x3/samples.json":
            sample_bytes = data
    if authoritative_pair is None or sample_bytes is None:
        _fail("historical mapping evidence chain is incomplete.")
    return authoritative_pair, sample_bytes


def _external_alias(
    *,
    artifact_root: Path,
    portable_payload: Mapping[str, Any],
    portable_index: Mapping[str, Mapping[str, Any]],
    yaml_object: Path,
) -> tuple[str, bytes]:
    declaration = portable_payload.get("external_alias_manifest")
    if not isinstance(declaration, dict):
        _fail("portable external_alias_manifest declaration must be an object.")
    for key, expected in {
        "sha256": EXTERNAL_ALIAS_MANIFEST_SHA256,
        "payload_sha256": EXTERNAL_ALIAS_PAYLOAD_SHA256,
    }.items():
        if declaration.get(key) != expected:
            _fail(f"portable external alias declaration {key} mismatch.")
    alias_relative = _relative_path(
        declaration.get("relative_path"), name="external alias manifest path"
    )
    expected_alias_relative = (
        ".negfpy-external-evidence/"
        f"{EXTERNAL_ALIAS_PAYLOAD_SHA256}/aliases.json"
    )
    if alias_relative != expected_alias_relative:
        _fail("external alias manifest path is not content-addressed by its payload.")
    alias_path = _safe_child(
        artifact_root, alias_relative, name="external alias manifest"
    )
    alias_bytes = _read_regular(alias_path, name="external alias manifest")
    if len(alias_bytes) != _nonnegative_int(
        declaration.get("size_bytes"), name="external alias manifest size"
    ):
        _fail("external alias manifest declared size mismatch.")
    _require_inventory(
        portable_index,
        alias_relative,
        EXTERNAL_ALIAS_MANIFEST_SHA256,
        len(alias_bytes),
    )
    payload = _envelope(
        alias_bytes,
        name="external alias manifest",
        expected_file_sha256=EXTERNAL_ALIAS_MANIFEST_SHA256,
        expected_payload_sha256=EXTERNAL_ALIAS_PAYLOAD_SHA256,
    )
    _exact_keys(
        payload,
        {
            "aliases",
            "object_inventory_sha256",
            "objects",
            "path_semantics",
            "recorded_artifact_root",
            "schema",
            "schema_version",
            "summary",
        },
        name="external alias payload",
    )
    if payload["schema"] != "negfpy.silicon_external_evidence_aliases.v1":
        _fail("external alias payload schema mismatch.")
    if payload["schema_version"] != 1:
        _fail("external alias payload schema_version mismatch.")
    objects = payload["objects"]
    aliases = payload["aliases"]
    if not isinstance(objects, list) or not isinstance(aliases, list):
        _fail("external alias objects and aliases must be arrays.")
    object_digest = _sha256_bytes(_canonical_payload_bytes(objects))
    if payload["object_inventory_sha256"] != object_digest:
        _fail("external alias object_inventory_sha256 does not reproduce.")

    yaml_digest = EXPECTED_FILE_SHA256["phono3py_yaml"]
    matches = [
        record
        for record in objects
        if isinstance(record, dict) and record.get("sha256") == yaml_digest
    ]
    alias_matches = [
        record
        for record in aliases
        if isinstance(record, dict) and record.get("sha256") == yaml_digest
    ]
    if len(matches) != 1 or len(alias_matches) != 1:
        _fail("external alias manifest must contain exactly one YAML object and alias.")
    object_record = matches[0]
    alias_record = alias_matches[0]
    _exact_keys(
        object_record,
        {"kind", "object_relative_path", "sha256", "size_bytes"},
        name="YAML external object record",
    )
    _exact_keys(
        alias_record,
        {"object_relative_path", "sha256", "size_bytes", "source_path"},
        name="YAML external alias record",
    )
    if object_record["kind"] != "binary":
        _fail("YAML external object kind must be binary.")
    object_relative = _relative_path(
        object_record["object_relative_path"], name="YAML object_relative_path"
    )
    expected_object_relative = f"objects/{yaml_digest[:2]}/{yaml_digest}.bin"
    if object_relative != expected_object_relative:
        _fail("YAML object path is not content-addressed by its byte digest.")
    if alias_record["object_relative_path"] != object_relative:
        _fail("YAML alias and object paths differ.")
    size = _nonnegative_int(object_record["size_bytes"], name="YAML object size")
    if alias_record["size_bytes"] != size:
        _fail("YAML alias and object sizes differ.")
    alias_directory = PurePosixPath(alias_relative).parent
    artifact_object_relative = (
        alias_directory / PurePosixPath(object_relative)
    ).as_posix()
    expected_object_path = _safe_child(
        artifact_root, artifact_object_relative, name="sealed YAML object"
    )
    explicit = yaml_object.expanduser().absolute()
    if explicit != expected_object_path:
        _fail(
            "Explicit YAML object is not the content-addressed object declared "
            "by the sealed external-alias manifest."
        )
    yaml_bytes = _read_regular(expected_object_path, name="sealed YAML object")
    _verify_bytes(yaml_bytes, yaml_digest, name="sealed YAML object")
    if len(yaml_bytes) != size:
        _fail("sealed YAML object size differs from the alias manifest.")
    _require_inventory(
        portable_index, artifact_object_relative, yaml_digest, len(yaml_bytes)
    )
    return artifact_object_relative, yaml_bytes


def _verify_force_receipt(
    *,
    artifact_root: Path,
    receipt_relative: str,
    receipt_bytes: bytes,
    portable_index: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    records = _parse_sha256sums(receipt_bytes, name="force-constants-SHA256SUMS")
    for filename, expected in RAW_FORCE_CONSTANT_SHA256.items():
        if records.get(filename) != expected:
            _fail(f"force-constants receipt identity mismatch for {filename!r}.")
    base_relative = PurePosixPath(receipt_relative).parent
    for filename, expected in records.items():
        if len(PurePosixPath(filename).parts) != 1:
            _fail("force-constants receipt entries must be plain filenames.")
        relative = (base_relative / filename).as_posix()
        source = _safe_child(artifact_root, relative, name=f"force receipt {filename}")
        actual, size = _hash_regular(source, name=f"force receipt {filename}")
        if actual != expected:
            _fail(f"complete force-constants archive mismatch for {filename!r}.")
        _require_inventory(portable_index, relative, expected, size)
    return records


def _mapping_derivation(yaml_path: Path) -> dict[str, Any]:
    try:
        mappings = {
            purpose: read_phono3py_crystal_mapping(yaml_path, purpose=purpose)
            for purpose in ("fc2", "fc3")
        }
        legacy = {
            purpose: crystal_mapping_sha256(mapping)
            for purpose, mapping in mappings.items()
        }
        primitive = {
            purpose: primitive_crystal_content_sha256(mapping)
            for purpose, mapping in mappings.items()
        }
        canonical = {
            purpose: crystal_mapping_content_sha256(mapping)
            for purpose, mapping in mappings.items()
        }
    except PhaseAFixtureError:
        raise
    except Exception as exc:
        raise PhaseAFixtureError(
            "Independent FC2/FC3 crystal-mapping reconstruction failed."
        ) from exc
    if legacy != HISTORICAL_MAPPING_SHA256:
        _fail("recomputed legacy FC2/FC3 mapping identities do not match 0.2 evidence.")
    if primitive["fc2"] != primitive["fc3"]:
        _fail("FC2 and FC3 canonical primitive-crystal identities differ.")
    if canonical["fc2"] == canonical["fc3"]:
        _fail("purpose-specific FC2 and FC3 mapping identities are not distinct.")
    return {
        "legacy_v0_2": legacy,
        "primitive_crystal_content": {
            "fc2": primitive["fc2"],
            "fc3": primitive["fc3"],
            "common": primitive["fc2"],
        },
        "crystal_mapping_content": canonical,
    }


def _derivation_receipt(derived: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "negfpy.phase-a-sealed-input-derivation.v1",
        "source_evidence": {
            "terminal_release_report_sha256": TERMINAL_REPORT_SHA256,
            "portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
            "portable_manifest_payload_sha256": PORTABLE_PAYLOAD_SHA256,
            "handoff_sha256sums_sha256": HANDOFF_SHA256SUMS_SHA256,
            "external_alias_manifest_sha256": EXTERNAL_ALIAS_MANIFEST_SHA256,
            "external_alias_payload_sha256": EXTERNAL_ALIAS_PAYLOAD_SHA256,
            "v3_sample_manifest_sha256": EXPECTED_FILE_SHA256[
                "v3_sample_manifest"
            ],
            "phono3py_yaml_sha256": EXPECTED_FILE_SHA256["phono3py_yaml"],
            "raw_force_constants_sha256": dict(
                sorted(RAW_FORCE_CONSTANT_SHA256.items())
            ),
        },
        "historical_mapping_sha256": dict(sorted(HISTORICAL_MAPPING_SHA256.items())),
        "derived_mapping_sha256": derived,
        "checks": {
            "terminal_release_certified": True,
            "terminal_evidence_inventory_verified": True,
            "portable_manifest_verified": True,
            "external_alias_manifest_verified": True,
            "force_constants_receipt_complete_verified": True,
            "historical_mapping_consistency_verified": True,
            "legacy_mapping_reproduced": True,
            "primitive_crystal_content_equal": True,
            "purpose_specific_mapping_content_distinct": True,
        },
    }


@dataclass(frozen=True, slots=True)
class _SourceProjection:
    copied_bytes: Mapping[str, bytes]
    origin_relative_paths: Mapping[str, str]
    derivation_receipt: Mapping[str, Any]


def _validate_source(
    *,
    artifact_root: Path,
    yaml_object: Path,
    portable_manifest: Path,
    handoff_sha256sums: Path,
) -> _SourceProjection:
    root = _assert_plain_root(artifact_root, name="artifact root")
    portable_bytes = _read_regular(portable_manifest, name="portable manifest")
    portable_payload, portable_index = _portable_manifest(portable_bytes)
    recorded_root = portable_payload["recorded_artifact_root"]

    sums_bytes = _read_regular(handoff_sha256sums, name="handoff SHA256SUMS")
    _verify_bytes(
        sums_bytes, HANDOFF_SHA256SUMS_SHA256, name="handoff SHA256SUMS"
    )
    handoff_records = _parse_sha256sums(sums_bytes, name="handoff SHA256SUMS")
    expected_handoff = {
        "portable-manifest.json": PORTABLE_MANIFEST_SHA256,
        "silicon-release-report.json": TERMINAL_REPORT_SHA256,
    }
    if handoff_records != expected_handoff:
        _fail("handoff SHA256SUMS does not contain the exact sealed projection.")
    handoff_directory = handoff_sha256sums.expanduser().absolute().parent
    if portable_manifest.expanduser().absolute() != (
        handoff_directory / "portable-manifest.json"
    ):
        _fail("portable manifest is not the file authenticated by handoff SHA256SUMS.")

    report_relative = ROLE_SPECS["release_report"].source_relative_path
    assert report_relative is not None
    report_path = _safe_child(root, report_relative, name="terminal release report")
    report_bytes = _read_regular(report_path, name="terminal release report")
    report = _release_report(report_bytes)
    if _recorded_root_from_terminal(report) != recorded_root:
        _fail(
            "portable and terminal-report evidence do not declare the same "
            "sealed artifact root."
        )
    handoff_report = _read_regular(
        handoff_directory / "silicon-release-report.json",
        name="handoff terminal release report",
    )
    if handoff_report != report_bytes:
        _fail("artifact and handoff terminal release reports are not byte-identical.")

    release_declaration = portable_payload.get("release_report")
    if not isinstance(release_declaration, dict):
        _fail("portable manifest release_report must be an object.")
    if (
        release_declaration.get("relative_path") != report_relative
        or release_declaration.get("sha256") != TERMINAL_REPORT_SHA256
        or release_declaration.get("size_bytes") != len(report_bytes)
        or release_declaration.get("release_certified") is not True
        or release_declaration.get("status") != "passed"
    ):
        _fail("portable release-report projection differs from the sealed report.")
    _require_inventory(
        portable_index, report_relative, TERMINAL_REPORT_SHA256, len(report_bytes)
    )

    historical_pair, sample_bytes = _verify_historical_chain(
        artifact_root=root,
        recorded_root=recorded_root,
        report=report,
        portable_index=portable_index,
    )
    if historical_pair != HISTORICAL_MAPPING_SHA256:
        _fail("authoritative historical mapping pair differs from frozen values.")

    yaml_relative, yaml_bytes = _external_alias(
        artifact_root=root,
        portable_payload=portable_payload,
        portable_index=portable_index,
        yaml_object=yaml_object,
    )

    copied: dict[str, bytes] = {}
    origins: dict[str, str] = {}
    for role, spec in ROLE_SPECS.items():
        if role == "phono3py_yaml":
            copied[role] = yaml_bytes
            origins[role] = yaml_relative
            continue
        if role == "v3_sample_manifest":
            copied[role] = sample_bytes
            assert spec.source_relative_path is not None
            origins[role] = spec.source_relative_path
            continue
        assert spec.source_relative_path is not None
        data = _read_regular(
            _safe_child(root, spec.source_relative_path, name=role), name=role
        )
        _verify_bytes(data, EXPECTED_FILE_SHA256[role], name=role)
        _require_inventory(
            portable_index,
            spec.source_relative_path,
            EXPECTED_FILE_SHA256[role],
            len(data),
        )
        copied[role] = data
        origins[role] = spec.source_relative_path

    receipt_role = "force_constants_sha256sums"
    receipt_relative = origins[receipt_role]
    _verify_force_receipt(
        artifact_root=root,
        receipt_relative=receipt_relative,
        receipt_bytes=copied[receipt_role],
        portable_index=portable_index,
    )

    with tempfile.TemporaryDirectory(prefix="negfpy-phase-a-yaml-") as directory:
        temporary_yaml = Path(directory) / "phono3py_disp.yaml"
        temporary_yaml.write_bytes(yaml_bytes)
        derived = _mapping_derivation(temporary_yaml)
    return _SourceProjection(
        copied_bytes=copied,
        origin_relative_paths=origins,
        derivation_receipt=_derivation_receipt(derived),
    )


def _write_file(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_noreplace(source: Path, target: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        _fail("atomic no-replace directory publication requires Linux renameat2.")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            _fail(f"fixture destination already exists: {target}")
        raise OSError(error, os.strerror(error), str(target))


def prepare_phase_a_fixture(
    *,
    artifact_root: str | Path,
    yaml_object: str | Path,
    output: str | Path,
    portable_manifest: str | Path | None = None,
    handoff_sha256sums: str | Path | None = None,
) -> dict[str, Any]:
    """Authenticate a sealed archive and atomically publish its Phase-A fixture.

    No output directory is created until every source hash, evidence join,
    complete force-constant receipt entry, and mapping identity has passed.
    Existing destinations are never replaced.
    """

    artifact = Path(artifact_root).expanduser().absolute()
    destination = Path(output).expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        _fail(f"fixture destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        _fail("fixture destination parent must be a real directory.")
    handoff_directory = artifact.parent / "handoff"
    portable = (
        handoff_directory / "portable-manifest.json"
        if portable_manifest is None
        else Path(portable_manifest).expanduser().absolute()
    )
    handoff_sums = (
        handoff_directory / "SHA256SUMS"
        if handoff_sha256sums is None
        else Path(handoff_sha256sums).expanduser().absolute()
    )

    source = _validate_source(
        artifact_root=artifact,
        yaml_object=Path(yaml_object),
        portable_manifest=portable,
        handoff_sha256sums=handoff_sums,
    )

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.stage-", dir=parent))
    published = False
    try:
        for role in sorted(ROLE_SPECS):
            spec = ROLE_SPECS[role]
            _write_file(stage / spec.target_relative_path, source.copied_bytes[role])

        manifest_files: list[dict[str, Any]] = []
        for role in sorted(ROLE_SPECS):
            spec = ROLE_SPECS[role]
            data = source.copied_bytes[role]
            origin_manifest = (
                EXTERNAL_ALIAS_MANIFEST_SHA256
                if spec.origin_kind == "sealed_external_alias"
                else PORTABLE_MANIFEST_SHA256
            )
            manifest_files.append(
                {
                    "role": role,
                    "relative_path": spec.target_relative_path,
                    "sha256": EXPECTED_FILE_SHA256[role],
                    "size_bytes": len(data),
                    "origin_kind": spec.origin_kind,
                    "origin_relative_path": source.origin_relative_paths[role],
                    "origin_manifest_sha256": origin_manifest,
                }
            )
        fixture_manifest = {
            "schema": "negfpy.phase-a-silicon-fixture.v1",
            "source_release": {
                "commit": SOURCE_COMMIT,
                "tree": SOURCE_TREE,
                "run_id": SOURCE_RUN_ID,
                "release_report_sha256": TERMINAL_REPORT_SHA256,
                "oracle_v0_2_fc2_mapping_sha256": HISTORICAL_MAPPING_SHA256["fc2"],
                "oracle_v0_2_fc3_mapping_sha256": HISTORICAL_MAPPING_SHA256["fc3"],
                "portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
                "handoff_sha256sums_sha256": HANDOFF_SHA256SUMS_SHA256,
                "external_alias_manifest_sha256": EXTERNAL_ALIAS_MANIFEST_SHA256,
                "external_alias_payload_sha256": EXTERNAL_ALIAS_PAYLOAD_SHA256,
            },
            "parent_input_manifest_sha256": PARENT_INPUT_MANIFEST_SHA256,
            "files": manifest_files,
        }
        manifest_bytes = _canonical_json_bytes(fixture_manifest)
        _write_file(stage / _FIXTURE_MANIFEST, manifest_bytes)

        copy_entries = []
        for record in manifest_files:
            copy_entries.append(
                {
                    "role": record["role"],
                    "source_relative_path": record["origin_relative_path"],
                    "target_relative_path": record["relative_path"],
                    "source_sha256": record["sha256"],
                    "target_sha256": record["sha256"],
                    "size_bytes": record["size_bytes"],
                }
            )
        copy_receipt = {
            "schema": "negfpy.phase-a-silicon-fixture-copy.v1",
            "fixture_manifest_sha256": _sha256_bytes(manifest_bytes),
            "source_release_report_sha256": TERMINAL_REPORT_SHA256,
            "source_portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
            "entries": copy_entries,
            "all_source_hashes_verified": True,
            "all_target_hashes_verified": True,
        }
        _write_file(stage / _COPY_RECEIPT, _canonical_json_bytes(copy_receipt))
        _write_file(
            stage / _DERIVATION_RECEIPT,
            _canonical_json_bytes(source.derivation_receipt),
        )

        covered = sorted(
            path.relative_to(stage).as_posix()
            for path in stage.rglob("*")
            if path.is_file()
        )
        sums = b"".join(
            f"{_hash_regular(stage / relative, name=relative)[0]}  {relative}\n".encode(
                "ascii"
            )
            for relative in covered
        )
        _write_file(stage / _FIXTURE_SUMS, sums)
        verify_phase_a_fixture(stage)
        _publish_noreplace(stage, destination)
        published = True
        return verify_phase_a_fixture(destination)
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def _fixture_tree(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = {"."}
    for entry in root.rglob("*"):
        relative = entry.relative_to(root).as_posix()
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode):
            _fail(f"fixture contains a symlink: {relative}")
        if stat.S_ISREG(mode):
            files.add(relative)
        elif stat.S_ISDIR(mode):
            directories.add(relative)
        else:
            _fail(f"fixture contains a special file: {relative}")
    return files, directories


def _validate_derivation_receipt(
    receipt: Mapping[str, Any], *, yaml_path: Path
) -> dict[str, Any]:
    _exact_keys(
        receipt,
        {
            "schema",
            "source_evidence",
            "historical_mapping_sha256",
            "derived_mapping_sha256",
            "checks",
        },
        name="sealed-input derivation receipt",
    )
    if receipt["schema"] != "negfpy.phase-a-sealed-input-derivation.v1":
        _fail("sealed-input derivation receipt schema mismatch.")
    expected = _derivation_receipt(_mapping_derivation(yaml_path))
    if receipt != expected:
        _fail("sealed-input derivation receipt does not reproduce exactly.")
    return expected["derived_mapping_sha256"]


def verify_phase_a_fixture(fixture_root: str | Path) -> dict[str, Any]:
    """Fail-closed verification of a portable Phase-A fixture directory."""

    root = _assert_plain_root(Path(fixture_root), name="Phase-A fixture")
    expected_files = {
        spec.target_relative_path for spec in ROLE_SPECS.values()
    } | {_FIXTURE_MANIFEST, _COPY_RECEIPT, _DERIVATION_RECEIPT, _FIXTURE_SUMS}
    expected_directories = {"."}
    for relative in expected_files:
        parent = PurePosixPath(relative).parent
        while parent.as_posix() != ".":
            expected_directories.add(parent.as_posix())
            parent = parent.parent
    actual_files, actual_directories = _fixture_tree(root)
    if actual_files != expected_files:
        _fail(
            "fixture file projection differs from the frozen set; "
            f"missing={sorted(expected_files - actual_files)!r}, "
            f"extra={sorted(actual_files - expected_files)!r}."
        )
    if actual_directories != expected_directories:
        _fail(
            "fixture directory projection differs from the frozen set; "
            f"missing={sorted(expected_directories - actual_directories)!r}, "
            f"extra={sorted(actual_directories - expected_directories)!r}."
        )

    sums_data = _read_regular(root / _FIXTURE_SUMS, name="fixture SHA256SUMS")
    sums = _parse_sha256sums(sums_data, name="fixture SHA256SUMS")
    if set(sums) != expected_files - {_FIXTURE_SUMS}:
        _fail("fixture SHA256SUMS coverage is not exact.")
    fixture_bytes: dict[str, bytes] = {}
    for relative in sorted(expected_files - {_FIXTURE_SUMS}):
        data = _read_regular(root / relative, name=f"fixture {relative}")
        fixture_bytes[relative] = data
        if _sha256_bytes(data) != sums[relative]:
            _fail(f"fixture SHA256SUMS mismatch for {relative!r}.")

    manifest = _load_json(
        fixture_bytes[_FIXTURE_MANIFEST], name=_FIXTURE_MANIFEST, canonical=True
    )
    _exact_keys(
        manifest,
        {"schema", "source_release", "parent_input_manifest_sha256", "files"},
        name=_FIXTURE_MANIFEST,
    )
    if manifest["schema"] != "negfpy.phase-a-silicon-fixture.v1":
        _fail("fixture manifest schema mismatch.")
    if manifest["parent_input_manifest_sha256"] != PARENT_INPUT_MANIFEST_SHA256:
        _fail("fixture parent input-manifest digest mismatch.")
    source_release = manifest["source_release"]
    if not isinstance(source_release, dict):
        _fail("fixture source_release must be an object.")
    expected_release = {
        "commit": SOURCE_COMMIT,
        "tree": SOURCE_TREE,
        "run_id": SOURCE_RUN_ID,
        "release_report_sha256": TERMINAL_REPORT_SHA256,
        "oracle_v0_2_fc2_mapping_sha256": HISTORICAL_MAPPING_SHA256["fc2"],
        "oracle_v0_2_fc3_mapping_sha256": HISTORICAL_MAPPING_SHA256["fc3"],
        "portable_manifest_sha256": PORTABLE_MANIFEST_SHA256,
        "handoff_sha256sums_sha256": HANDOFF_SHA256SUMS_SHA256,
        "external_alias_manifest_sha256": EXTERNAL_ALIAS_MANIFEST_SHA256,
        "external_alias_payload_sha256": EXTERNAL_ALIAS_PAYLOAD_SHA256,
    }
    if source_release != expected_release:
        _fail("fixture source_release projection differs from the frozen seal.")

    records = manifest["files"]
    if not isinstance(records, list) or len(records) != len(ROLE_SPECS):
        _fail("fixture manifest must contain exactly nine copied file records.")
    roles = []
    records_by_role: dict[str, Mapping[str, Any]] = {}
    for position, record in enumerate(records):
        if not isinstance(record, dict):
            _fail(f"fixture file record {position} must be an object.")
        _exact_keys(
            record,
            {
                "role",
                "relative_path",
                "sha256",
                "size_bytes",
                "origin_kind",
                "origin_relative_path",
                "origin_manifest_sha256",
            },
            name=f"fixture file record {position}",
        )
        role = record["role"]
        if (
            not isinstance(role, str)
            or role not in ROLE_SPECS
            or role in records_by_role
        ):
            _fail(f"fixture file record {position} has an unknown or duplicate role.")
        roles.append(role)
        records_by_role[role] = record
        spec = ROLE_SPECS[role]
        if record["relative_path"] != spec.target_relative_path:
            _fail(f"fixture target path mismatch for role {role!r}.")
        if record["sha256"] != EXPECTED_FILE_SHA256[role]:
            _fail(f"fixture source digest mismatch for role {role!r}.")
        if record["size_bytes"] != len(fixture_bytes[spec.target_relative_path]):
            _fail(f"fixture source size mismatch for role {role!r}.")
        if _sha256_bytes(fixture_bytes[spec.target_relative_path]) != record["sha256"]:
            _fail(f"fixture copied bytes mismatch for role {role!r}.")
        if record["origin_kind"] != spec.origin_kind:
            _fail(f"fixture origin kind mismatch for role {role!r}.")
        origin = _relative_path(
            record["origin_relative_path"], name=f"fixture {role} origin path"
        )
        if role != "phono3py_yaml" and origin != spec.source_relative_path:
            _fail(f"fixture origin path mismatch for role {role!r}.")
        if role == "phono3py_yaml":
            yaml_digest = EXPECTED_FILE_SHA256["phono3py_yaml"]
            expected_origin = (
                ".negfpy-external-evidence/"
                f"{EXTERNAL_ALIAS_PAYLOAD_SHA256}/objects/"
                f"{yaml_digest[:2]}/{yaml_digest}.bin"
            )
            if origin != expected_origin:
                _fail("fixture YAML origin is not the sealed content-addressed object.")
        expected_origin_manifest = (
            EXTERNAL_ALIAS_MANIFEST_SHA256
            if role == "phono3py_yaml"
            else PORTABLE_MANIFEST_SHA256
        )
        if record["origin_manifest_sha256"] != expected_origin_manifest:
            _fail(f"fixture origin manifest mismatch for role {role!r}.")
    if roles != sorted(ROLE_SPECS):
        _fail("fixture file records are not sorted lexicographically by role.")

    report = _release_report(
        fixture_bytes[ROLE_SPECS["release_report"].target_relative_path]
    )
    recorded_root = _recorded_root_from_terminal(report)
    terminal_files = report["evidence_inventory"]["files"]
    for relative, expected_digest in HISTORICAL_CONSISTENCY_FILES.items():
        matches = [
            record
            for record in terminal_files
            if isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and record["path"].endswith("/" + relative)
        ]
        if len(matches) != 1 or matches[0].get("sha256") != expected_digest:
            _fail(f"fixture report does not authenticate historical file {relative!r}.")
        if matches[0]["path"] != f"{recorded_root}/{relative}":
            _fail(
                "fixture report historical path is outside the sealed run for "
                f"{relative!r}."
            )

    sample = _load_json(
        fixture_bytes[ROLE_SPECS["v3_sample_manifest"].target_relative_path],
        name="fixture v3 sample manifest",
    )
    pair = _mapping_pair(sample, qualification=False, name="fixture v3 sample manifest")
    if pair != HISTORICAL_MAPPING_SHA256:
        _fail("fixture v3 sample historical mapping pair mismatch.")

    force_receipt_bytes = fixture_bytes[
        ROLE_SPECS["force_constants_sha256sums"].target_relative_path
    ]
    force_records = _parse_sha256sums(
        force_receipt_bytes, name="fixture force-constants-SHA256SUMS"
    )
    for filename, expected in RAW_FORCE_CONSTANT_SHA256.items():
        if force_records.get(filename) != expected:
            _fail(f"fixture force receipt mismatch for {filename!r}.")

    copy_receipt = _load_json(
        fixture_bytes[_COPY_RECEIPT], name=_COPY_RECEIPT, canonical=True
    )
    _exact_keys(
        copy_receipt,
        {
            "schema",
            "fixture_manifest_sha256",
            "source_release_report_sha256",
            "source_portable_manifest_sha256",
            "entries",
            "all_source_hashes_verified",
            "all_target_hashes_verified",
        },
        name=_COPY_RECEIPT,
    )
    if copy_receipt["schema"] != "negfpy.phase-a-silicon-fixture-copy.v1":
        _fail("fixture copy receipt schema mismatch.")
    if copy_receipt["fixture_manifest_sha256"] != _sha256_bytes(
        fixture_bytes[_FIXTURE_MANIFEST]
    ):
        _fail("fixture copy receipt manifest digest mismatch.")
    if copy_receipt["source_release_report_sha256"] != TERMINAL_REPORT_SHA256:
        _fail("fixture copy receipt release-report digest mismatch.")
    if copy_receipt["source_portable_manifest_sha256"] != PORTABLE_MANIFEST_SHA256:
        _fail("fixture copy receipt portable-manifest digest mismatch.")
    if copy_receipt["all_source_hashes_verified"] is not True or copy_receipt[
        "all_target_hashes_verified"
    ] is not True:
        _fail("fixture copy receipt verification booleans are not true.")
    expected_copy_entries = []
    for role in sorted(ROLE_SPECS):
        record = records_by_role[role]
        expected_copy_entries.append(
            {
                "role": role,
                "source_relative_path": record["origin_relative_path"],
                "target_relative_path": record["relative_path"],
                "source_sha256": record["sha256"],
                "target_sha256": record["sha256"],
                "size_bytes": record["size_bytes"],
            }
        )
    if copy_receipt["entries"] != expected_copy_entries:
        _fail("fixture copy receipt entries do not reproduce the manifest projection.")

    derivation = _load_json(
        fixture_bytes[_DERIVATION_RECEIPT], name=_DERIVATION_RECEIPT, canonical=True
    )
    derived = _validate_derivation_receipt(
        derivation,
        yaml_path=root / ROLE_SPECS["phono3py_yaml"].target_relative_path,
    )
    return {
        "verified": True,
        "schema": "negfpy.phase-a-silicon-fixture-verification.v1",
        "fixture_manifest_sha256": _sha256_bytes(fixture_bytes[_FIXTURE_MANIFEST]),
        "fixture_sha256sums_sha256": _sha256_bytes(sums_data),
        "copied_file_count": len(ROLE_SPECS),
        "covered_file_count": len(sums),
        "source_commit": SOURCE_COMMIT,
        "source_tree": SOURCE_TREE,
        "source_run_id": SOURCE_RUN_ID,
        "common_primitive_crystal_content_sha256": derived[
            "primitive_crystal_content"
        ]["common"],
        "fc2_crystal_mapping_content_sha256": derived[
            "crystal_mapping_content"
        ]["fc2"],
        "fc3_crystal_mapping_content_sha256": derived[
            "crystal_mapping_content"
        ]["fc3"],
    }


__all__ = [
    "PhaseAFixtureError",
    "prepare_phase_a_fixture",
    "verify_phase_a_fixture",
]
