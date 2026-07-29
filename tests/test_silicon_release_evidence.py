from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
from types import ModuleType

import pytest

from negfpy.anharmonic import _silicon_release_evidence as portable


h5py = pytest.importorskip("h5py")


@dataclass(frozen=True)
class _EvidenceTree:
    root: Path
    report: Path
    manifest: Path
    data: Path
    nested_report: Path
    hdf5: Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_record(path: Path) -> dict[str, object]:
    metadata = path.stat()
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "size_bytes": metadata.st_size,
        # These live-validation values intentionally change after relocation.
        "device": metadata.st_dev,
        "inode": metadata.st_ino,
        "mtime_ns": metadata.st_mtime_ns,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_release_report(tree: _EvidenceTree) -> None:
    _write_json(
        tree.report,
        {
            "schema": portable.SILICON_RELEASE_REPORT_SCHEMA,
            "schema_version": 1,
            "status": "passed",
            "validation_passed": True,
            "certification_eligible": True,
            "release_certified": True,
            "release_version": "0.2.0",
            "evidence": {
                "nested_report": _file_record(tree.nested_report),
                "merged_hdf5": _file_record(tree.hdf5),
            },
        },
    )


def _evidence_tree(tmp_path: Path) -> _EvidenceTree:
    root = tmp_path / "artifact tree"
    root.mkdir()
    data = root / "raw" / "payload.bin"
    data.parent.mkdir()
    data.write_bytes(b"portable silicon evidence\n")

    hdf5 = root / "results" / "merged.hdf5"
    hdf5.parent.mkdir()
    with h5py.File(hdf5, "w") as handle:
        handle.attrs["status"] = "complete"
        handle.create_dataset("values", data=[1.0, 2.0, 3.0])
        group = handle.create_group("nested")
        group.create_dataset("indices", data=[[0, 1], [2, 3]])

    nested = root / "reports" / "nested.json"
    _write_json(
        nested,
        {
            "schema": "test.strict.evidence.v1",
            "artifact": _file_record(data),
        },
    )
    report = root / "release" / "silicon-release.json"
    tree = _EvidenceTree(
        root=root,
        report=report,
        manifest=tmp_path / "portable-manifest.json",
        data=data,
        nested_report=nested,
        hdf5=hdf5,
    )
    _write_release_report(tree)
    return tree


def _external_evidence_tree(
    tmp_path: Path,
    *,
    recursive_json: bool = False,
) -> tuple[_EvidenceTree, Path, tuple[Path, ...]]:
    tree = _evidence_tree(tmp_path)
    source_directory = tmp_path / "external sources"
    source_directory.mkdir()
    payload = source_directory / "payload.bin"
    payload.write_bytes(b"external portable silicon evidence\n")
    sources: tuple[Path, ...]
    if recursive_json:
        external_json = source_directory / "external-report.json"
        _write_json(
            external_json,
            {
                "schema": "test.external.strict.evidence.v1",
                "payload": _file_record(payload),
            },
        )
        record = _file_record(external_json)
        sources = (external_json, payload)
    else:
        record = _file_record(payload)
        sources = (payload,)
    _write_json(
        tree.nested_report,
        {
            "schema": "test.strict.evidence.v1",
            "artifact": record,
        },
    )
    _write_release_report(tree)
    return tree, source_directory, sources


def _seal(tree: _EvidenceTree) -> portable.PortableSiliconReleaseEvidence:
    return portable.seal_silicon_release_evidence(
        tree.root,
        tree.report,
        tree.manifest,
    )


def _load_driver() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "seal_phono3py_si_release_evidence.py"
    )
    specification = importlib.util.spec_from_file_location(
        "negfpy_test_portable_release_cli",
        path,
    )
    if specification is None or specification.loader is None:
        raise AssertionError(f"cannot load CLI driver {path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_seal_and_verify_after_relocation(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    sealed = _seal(tree)
    assert sealed.recorded_artifact_root == str(tree.root.resolve())
    assert sealed.release_report_relative_path == ("release/silicon-release.json")
    assert sealed.release_report_sha256 == _sha256(tree.report)
    assert sealed.file_count == 4
    assert sealed.json_file_count == 2
    assert sealed.path_record_count == 3
    assert sealed.hdf5_file_count == 1
    assert (
        sealed.verification_scope
        == "self_authenticated_integrity_and_relocation_only"
    )

    document = json.loads(tree.manifest.read_text(encoding="ascii"))
    payload = document["payload"]
    assert [set(item) for item in payload["inventory"]] == [
        {"relative_path", "sha256", "size_bytes"}
    ] * 4
    assert payload["release_report"]["release_certified"] is True

    relocated = tmp_path / "retrieved on another workstation"
    shutil.copytree(tree.root, relocated)
    verified = portable.verify_silicon_release_evidence(
        relocated,
        tree.manifest,
    )
    assert verified.payload_sha256 == sealed.payload_sha256
    assert verified.manifest_sha256 == sealed.manifest_sha256
    authenticated = portable.verify_silicon_release_evidence(
        relocated,
        tree.manifest,
        expected_manifest_sha256=sealed.manifest_sha256,
        expected_release_report_sha256=sealed.release_report_sha256,
    )
    assert (
        authenticated.verification_scope
        == "trusted_digest_authenticated_integrity"
    )
    assert authenticated.trusted_manifest_sha256_matched
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="trusted SHA256",
    ):
        portable.verify_silicon_release_evidence(
            relocated,
            tree.manifest,
            expected_manifest_sha256="0" * 64,
            expected_release_report_sha256=sealed.release_report_sha256,
        )
    assert verified.verified_artifact_root == relocated.resolve()
    assert verified.recorded_artifact_root == str(tree.root.resolve())


def test_tampered_member_is_rejected(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    _seal(tree)
    tree.data.write_bytes(b"tampered\n")
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="size or SHA256",
    ):
        portable.verify_silicon_release_evidence(tree.root, tree.manifest)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_missing_or_extra_members_are_rejected(
    tmp_path: Path,
    change: str,
) -> None:
    tree = _evidence_tree(tmp_path)
    _seal(tree)
    if change == "missing":
        tree.data.unlink()
    else:
        (tree.root / "unexpected.log").write_text(
            "not in manifest\n",
            encoding="utf-8",
        )
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="artifact membership differs",
    ):
        portable.verify_silicon_release_evidence(tree.root, tree.manifest)


def test_traversing_embedded_path_record_is_rejected(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside\n")
    _write_json(
        tree.nested_report,
        {
            "schema": "test.strict.evidence.v1",
            "artifact": {
                "path": "../outside.bin",
                "sha256": _sha256(outside),
                "size_bytes": outside.stat().st_size,
            },
        },
    )
    _write_release_report(tree)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="normalized relative POSIX path",
    ):
        _seal(tree)


def test_symbolic_link_is_rejected(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    (tree.root / "payload-link").symlink_to(tree.data)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="symbolic link",
    ):
        _seal(tree)


def test_hard_link_is_rejected(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    os.link(tree.data, tree.root / "payload-hardlink")
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="hard-linked file",
    ):
        _seal(tree)


def test_special_file_is_rejected(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    fifo = tree.root / "unexpected.fifo"
    os.mkfifo(fifo)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="special filesystem object",
    ):
        _seal(tree)


def test_truncated_hdf5_is_rejected_while_sealing(tmp_path: Path) -> None:
    tree = _evidence_tree(tmp_path)
    broken = tree.root / "results" / "truncated.h5"
    broken.write_bytes(portable._HDF5_SIGNATURE + b"\x00" * 32)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="unreadable or truncated",
    ):
        _seal(tree)


def test_manifest_traversal_is_rejected_even_with_recomputed_digest(
    tmp_path: Path,
) -> None:
    tree = _evidence_tree(tmp_path)
    _seal(tree)
    document = json.loads(tree.manifest.read_text(encoding="ascii"))
    inventory = document["payload"]["inventory"]
    inventory[0]["relative_path"] = "../escape"
    document["payload"]["inventory_sha256"] = portable._payload_sha256(inventory)
    document["payload_sha256"] = portable._payload_sha256(document["payload"])
    tree.manifest.write_bytes(portable._canonical_bytes(document))
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="normalized relative POSIX path",
    ):
        portable.verify_silicon_release_evidence(tree.root, tree.manifest)


def test_cli_seals_and_verifies_relocated_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tree = _evidence_tree(tmp_path)
    driver = _load_driver()
    assert (
        driver.main(
            [
                "seal",
                "--artifact-root",
                str(tree.root),
                "--release-report",
                str(tree.report),
                "--output",
                str(tree.manifest),
            ]
        )
        == 0
    )
    sealed_output = json.loads(capsys.readouterr().out)
    assert sealed_output["status"] == "passed"
    assert sealed_output["action"] == "sealed"

    relocated = tmp_path / "relocated"
    shutil.copytree(tree.root, relocated)
    assert (
        driver.main(
            [
                "verify",
                "--artifact-root",
                str(relocated),
                "--manifest",
                str(tree.manifest),
            ]
        )
        == 0
    )
    verified_output = json.loads(capsys.readouterr().out)
    assert verified_output["status"] == "passed"
    assert verified_output["action"] == "verified"
    assert verified_output["verified_artifact_root"] == str(relocated.resolve())


def test_external_staging_seals_and_verifies_without_original_sources(
    tmp_path: Path,
) -> None:
    tree, source_directory, sources = _external_evidence_tree(
        tmp_path,
        recursive_json=True,
    )
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="no external alias manifest was bound",
    ):
        _seal(tree)

    staged = portable.stage_external_silicon_release_evidence(tree.root)
    assert staged.alias_count == 2
    assert staged.object_count == 2
    assert staged.staging_subtree_relative_path.startswith(
        ".negfpy-external-evidence/"
    )
    assert (
        staged.alias_manifest_path.parent.name
        == staged.payload_sha256
    )
    alias_document = json.loads(
        staged.alias_manifest_path.read_text(encoding="ascii")
    )
    alias_payload = alias_document["payload"]
    assert alias_payload["schema"] == portable.SILICON_EXTERNAL_ALIAS_SCHEMA
    assert [item["source_path"] for item in alias_payload["aliases"]] == (
        sorted(str(path.resolve()) for path in sources)
    )
    assert all(
        set(item)
        == {
            "source_path",
            "sha256",
            "size_bytes",
            "object_relative_path",
        }
        for item in alias_payload["aliases"]
    )

    shutil.rmtree(source_directory)
    sealed = portable.seal_silicon_release_evidence(
        tree.root,
        tree.report,
        tree.manifest,
        external_alias_manifest_path=staged.alias_manifest_path,
    )
    portable_document = json.loads(
        tree.manifest.read_text(encoding="ascii")
    )
    assert (
        portable_document["payload"]["schema"]
        == portable.SILICON_RELEASE_EVIDENCE_ALIAS_SCHEMA
    )
    assert (
        portable_document["payload"]["external_alias_manifest"][
            "payload_sha256"
        ]
        == staged.payload_sha256
    )

    relocated = tmp_path / "relocated external evidence"
    shutil.copytree(tree.root, relocated)
    verified = portable.verify_silicon_release_evidence(
        relocated,
        tree.manifest,
    )
    assert verified.payload_sha256 == sealed.payload_sha256
    assert verified.verified_artifact_root == relocated.resolve()


def test_staging_is_explicit_and_immutable(tmp_path: Path) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    staged = portable.stage_external_silicon_release_evidence(tree.root)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="no external alias manifest was bound",
    ):
        _seal(tree)
    with pytest.raises(
        FileExistsError,
        match="already contains",
    ):
        portable.stage_external_silicon_release_evidence(tree.root)
    assert staged.alias_manifest_path.is_file()


def test_staging_records_verified_size_when_record_omits_it(
    tmp_path: Path,
) -> None:
    tree, _, sources = _external_evidence_tree(tmp_path)
    nested = json.loads(tree.nested_report.read_text(encoding="utf-8"))
    nested["artifact"].pop("size_bytes")
    _write_json(tree.nested_report, nested)
    _write_release_report(tree)
    staged = portable.stage_external_silicon_release_evidence(tree.root)
    document = json.loads(
        staged.alias_manifest_path.read_text(encoding="ascii")
    )
    assert document["payload"]["aliases"][0]["size_bytes"] == (
        sources[0].stat().st_size
    )


def test_staging_rejects_tree_without_external_records(
    tmp_path: Path,
) -> None:
    tree = _evidence_tree(tmp_path)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="contains no external",
    ):
        portable.stage_external_silicon_release_evidence(tree.root)
    assert not (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY
    ).exists()


@pytest.mark.parametrize(
    "failure",
    [
        "missing",
        "sha256",
        "size",
        "symlink",
        "symlink-parent",
        "hardlink",
        "traversal",
        "missing-sha256",
        "conflict",
    ],
)
def test_external_staging_rejects_unsafe_or_ambiguous_sources(
    tmp_path: Path,
    failure: str,
) -> None:
    tree, source_directory, sources = _external_evidence_tree(tmp_path)
    source = sources[0]
    record: dict[str, object] = _file_record(source)
    if failure == "missing":
        source.unlink()
    elif failure == "sha256":
        record["sha256"] = "f" * 64
    elif failure == "size":
        record["size_bytes"] = int(record["size_bytes"]) + 1
    elif failure == "symlink":
        real = source_directory / "real.bin"
        source.rename(real)
        source.symlink_to(real)
        record["path"] = str(source.absolute())
        record["sha256"] = _sha256(real)
        record["size_bytes"] = real.stat().st_size
    elif failure == "symlink-parent":
        real_directory = tmp_path / "real external sources"
        source_directory.rename(real_directory)
        source_directory.symlink_to(real_directory, target_is_directory=True)
        linked_source = source_directory / source.name
        record["path"] = str(linked_source.absolute())
        record["sha256"] = _sha256(real_directory / source.name)
        record["size_bytes"] = (
            real_directory / source.name
        ).stat().st_size
    elif failure == "hardlink":
        os.link(source, source_directory / "payload-hardlink.bin")
    elif failure == "traversal":
        record["path"] = "../external sources/payload.bin"
    elif failure == "missing-sha256":
        record.pop("sha256")
    elif failure == "conflict":
        pass
    nested: dict[str, object] = {
        "schema": "test.strict.evidence.v1",
        "artifact": record,
    }
    if failure == "conflict":
        nested["conflicting_artifact"] = {
            **record,
            "sha256": "f" * 64,
        }
    _write_json(tree.nested_report, nested)
    _write_release_report(tree)

    with pytest.raises(portable.SiliconReleaseEvidenceError):
        portable.stage_external_silicon_release_evidence(tree.root)


@pytest.mark.parametrize("target", ["alias", "object"])
def test_seal_rejects_tampered_external_staging(
    tmp_path: Path,
    target: str,
) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    staged = portable.stage_external_silicon_release_evidence(tree.root)
    if target == "alias":
        staged.alias_manifest_path.write_bytes(
            staged.alias_manifest_path.read_bytes() + b" "
        )
    else:
        document = json.loads(
            staged.alias_manifest_path.read_text(encoding="ascii")
        )
        relative = document["payload"]["objects"][0][
            "object_relative_path"
        ]
        (staged.alias_manifest_path.parent / relative).write_bytes(
            b"tampered\n"
        )
    with pytest.raises(portable.SiliconReleaseEvidenceError):
        portable.seal_silicon_release_evidence(
            tree.root,
            tree.report,
            tree.manifest,
            external_alias_manifest_path=staged.alias_manifest_path,
        )


def test_cli_stages_external_evidence_then_seals_and_verifies(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tree, source_directory, _ = _external_evidence_tree(tmp_path)
    driver = _load_driver()
    assert (
        driver.main(
            [
                "stage",
                "--artifact-root",
                str(tree.root),
            ]
        )
        == 0
    )
    staged_output = json.loads(capsys.readouterr().out)
    assert staged_output["status"] == "passed"
    assert staged_output["action"] == "staged"
    shutil.rmtree(source_directory)

    assert (
        driver.main(
            [
                "seal",
                "--artifact-root",
                str(tree.root),
                "--release-report",
                str(tree.report),
                "--external-alias-manifest",
                staged_output["alias_manifest_path"],
                "--output",
                str(tree.manifest),
            ]
        )
        == 0
    )
    sealed_output = json.loads(capsys.readouterr().out)
    assert sealed_output["status"] == "passed"
    assert sealed_output["action"] == "sealed"

    relocated = tmp_path / "cli-relocated"
    shutil.copytree(tree.root, relocated)
    assert (
        driver.main(
            [
                "verify",
                "--artifact-root",
                str(relocated),
                "--manifest",
                str(tree.manifest),
            ]
        )
        == 0
    )
    verified_output = json.loads(capsys.readouterr().out)
    assert verified_output["status"] == "passed"
    assert verified_output["action"] == "verified"


def test_finalize_without_external_records_and_verify_handoff(
    tmp_path: Path,
) -> None:
    tree = _evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"
    finalized = portable.finalize_silicon_release_evidence(
        tree.root,
        tree.report,
        handoff,
    )

    assert finalized.action == "finalized"
    assert not finalized.external_evidence_staged
    assert not finalized.as_dict()["scientific_recomputation_performed"]
    assert finalized.portable_evidence.verification_scope == (
        "self_authenticated_integrity_and_relocation_only"
    )
    assert sorted(path.name for path in handoff.iterdir()) == [
        "SHA256SUMS",
        "portable-manifest.json",
        "silicon-release-report.json",
    ]
    assert finalized.release_report_copy_path.read_bytes() == (
        tree.report.read_bytes()
    )
    assert finalized.checksum_path.read_bytes() == (
        portable._handoff_checksum_bytes(  # noqa: SLF001
            finalized.portable_evidence.manifest_sha256,
            finalized.portable_evidence.release_report_sha256,
        )
    )

    retrieved = portable.verify_silicon_release_handoff(
        tree.root,
        finalized.checksum_path,
    )
    assert retrieved.action == "retrieval_verified"
    assert retrieved.portable_evidence.verification_scope == (
        "trusted_digest_authenticated_integrity"
    )
    assert retrieved.portable_evidence.trusted_manifest_sha256_matched
    assert retrieved.portable_evidence.trusted_release_report_sha256_matched


def test_finalize_stages_external_records_and_verifies_after_relocation(
    tmp_path: Path,
) -> None:
    tree, source_directory, _ = _external_evidence_tree(
        tmp_path,
        recursive_json=True,
    )
    handoff = tmp_path / "release-handoff"
    finalized = portable.finalize_silicon_release_evidence(
        tree.root,
        tree.report,
        handoff,
    )
    assert finalized.external_evidence_staged
    assert (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY  # noqa: SLF001
    ).is_dir()
    shutil.rmtree(source_directory)

    relocated_root = tmp_path / "retrieved-artifacts"
    relocated_handoff = tmp_path / "retrieved-handoff"
    shutil.copytree(tree.root, relocated_root)
    shutil.copytree(handoff, relocated_handoff)
    retrieved = portable.verify_silicon_release_handoff(
        relocated_root,
        relocated_handoff / "SHA256SUMS",
    )
    assert retrieved.external_evidence_staged
    assert retrieved.portable_evidence.verified_artifact_root == (
        relocated_root.resolve()
    )
    assert retrieved.portable_evidence.verification_scope == (
        "trusted_digest_authenticated_integrity"
    )


def test_finalize_rejects_uncertified_report_before_mutation(
    tmp_path: Path,
) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    report = json.loads(tree.report.read_text(encoding="utf-8"))
    report["release_certified"] = False
    _write_json(tree.report, report)
    handoff = tmp_path / "release-handoff"

    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="not an eligible certified",
    ):
        portable.finalize_silicon_release_evidence(
            tree.root,
            tree.report,
            handoff,
        )

    assert not handoff.exists()
    assert not (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY  # noqa: SLF001
    ).exists()
    assert not tuple(tmp_path.glob(".release-handoff.*.tmp"))


def test_finalize_refuses_existing_or_symlink_handoff_without_staging(
    tmp_path: Path,
) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    existing = tmp_path / "existing-handoff"
    existing.mkdir()

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        portable.finalize_silicon_release_evidence(
            tree.root,
            tree.report,
            existing,
        )

    linked = tmp_path / "linked-handoff"
    linked.symlink_to(existing, target_is_directory=True)
    with pytest.raises(FileExistsError, match="will not be overwritten"):
        portable.finalize_silicon_release_evidence(
            tree.root,
            tree.report,
            linked,
        )
    assert not (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY  # noqa: SLF001
    ).exists()


def test_finalize_cleans_staging_and_temporary_handoff_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"

    def fail_seal(*args: object, **kwargs: object) -> object:
        raise RuntimeError("injected sealing failure")

    monkeypatch.setattr(
        portable,
        "seal_silicon_release_evidence",
        fail_seal,
    )
    with pytest.raises(RuntimeError, match="injected sealing failure"):
        portable.finalize_silicon_release_evidence(
            tree.root,
            tree.report,
            handoff,
        )

    assert not handoff.exists()
    assert not (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY  # noqa: SLF001
    ).exists()
    assert not tuple(tmp_path.glob(".release-handoff.*.tmp"))


def test_finalize_atomic_publication_does_not_replace_racing_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree, _, _ = _external_evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"
    publish = portable._publish_directory_noreplace  # noqa: SLF001

    def publish_after_race(source: Path, target: Path) -> None:
        target.mkdir()
        publish(source, target)

    monkeypatch.setattr(
        portable,
        "_publish_directory_noreplace",
        publish_after_race,
    )
    with pytest.raises(FileExistsError, match="not overwritten"):
        portable.finalize_silicon_release_evidence(
            tree.root,
            tree.report,
            handoff,
        )

    assert handoff.is_dir()
    assert not tuple(handoff.iterdir())
    assert not (
        tree.root / portable._EXTERNAL_STAGING_DIRECTORY  # noqa: SLF001
    ).exists()
    assert not tuple(tmp_path.glob(".release-handoff.*.tmp"))


def test_handoff_verification_rejects_uncertified_report_copy(
    tmp_path: Path,
) -> None:
    tree = _evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"
    finalized = portable.finalize_silicon_release_evidence(
        tree.root,
        tree.report,
        handoff,
    )
    copied = json.loads(
        finalized.release_report_copy_path.read_text(encoding="ascii")
    )
    copied["release_certified"] = False
    _write_json(finalized.release_report_copy_path, copied)
    finalized.checksum_path.write_bytes(
        portable._handoff_checksum_bytes(  # noqa: SLF001
            _sha256(finalized.portable_evidence.manifest_path),
            _sha256(finalized.release_report_copy_path),
        )
    )

    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="not an eligible certified",
    ):
        portable.verify_silicon_release_handoff(
            tree.root,
            finalized.checksum_path,
        )


def test_handoff_verification_rejects_symlink_or_extra_member(
    tmp_path: Path,
) -> None:
    tree = _evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"
    finalized = portable.finalize_silicon_release_evidence(
        tree.root,
        tree.report,
        handoff,
    )
    linked_checksum = tmp_path / "trusted-SHA256SUMS"
    linked_checksum.symlink_to(finalized.checksum_path)
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="symbolic link",
    ):
        portable.verify_silicon_release_handoff(
            tree.root,
            linked_checksum,
        )

    (handoff / "unexpected.txt").write_text(
        "not part of the handoff\n",
        encoding="utf-8",
    )
    with pytest.raises(
        portable.SiliconReleaseEvidenceError,
        match="membership is not exact",
    ):
        portable.verify_silicon_release_handoff(
            tree.root,
            finalized.checksum_path,
        )


def test_cli_finalize_and_verify_handoff_after_relocation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tree, source_directory, _ = _external_evidence_tree(tmp_path)
    handoff = tmp_path / "release-handoff"
    driver = _load_driver()
    assert (
        driver.main(
            [
                "finalize",
                "--artifact-root",
                str(tree.root),
                "--release-report",
                str(tree.report),
                "--handoff-directory",
                str(handoff),
            ]
        )
        == 0
    )
    finalized_output = json.loads(capsys.readouterr().out)
    assert finalized_output["status"] == "passed"
    assert finalized_output["action"] == "finalized"
    assert finalized_output["scientific_recomputation_performed"] is False
    shutil.rmtree(source_directory)

    relocated_root = tmp_path / "cli-retrieved-artifacts"
    relocated_handoff = tmp_path / "cli-retrieved-handoff"
    shutil.copytree(tree.root, relocated_root)
    shutil.copytree(handoff, relocated_handoff)
    assert (
        driver.main(
            [
                "verify-handoff",
                "--artifact-root",
                str(relocated_root),
                "--trusted-sha256sums",
                str(relocated_handoff / "SHA256SUMS"),
            ]
        )
        == 0
    )
    retrieved_output = json.loads(capsys.readouterr().out)
    assert retrieved_output["status"] == "passed"
    assert retrieved_output["action"] == "retrieval_verified"
    assert retrieved_output["scientific_recomputation_performed"] is False
    assert (
        retrieved_output["portable_evidence"]["verification_scope"]
        == "trusted_digest_authenticated_integrity"
    )
