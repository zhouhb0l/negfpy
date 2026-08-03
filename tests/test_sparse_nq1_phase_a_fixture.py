from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import negfpy.anharmonic.phase_a_fixture as fixture_module
from negfpy.anharmonic.phase_a_fixture import (
    PhaseAFixtureError,
    prepare_phase_a_fixture,
    verify_phase_a_fixture,
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object, *, trailing_lf: bool = True) -> bytes:
    result = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    return result + (b"\n" if trailing_lf else b"")


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _sums(records: dict[str, str]) -> bytes:
    return "".join(
        f"{records[path]}  {path}\n" for path in sorted(records)
    ).encode("ascii")


def _patch_mapping_functions(
    monkeypatch: pytest.MonkeyPatch,
    *,
    historical: dict[str, str],
    legacy_mismatch: bool,
) -> None:
    monkeypatch.setattr(
        fixture_module,
        "read_phono3py_crystal_mapping",
        lambda _path, *, purpose: SimpleNamespace(purpose=purpose),
    )

    def legacy(mapping: SimpleNamespace) -> str:
        if legacy_mismatch and mapping.purpose == "fc2":
            return _sha256(b"wrong legacy mapping")
        return historical[mapping.purpose]

    monkeypatch.setattr(fixture_module, "crystal_mapping_sha256", legacy)
    monkeypatch.setattr(
        fixture_module,
        "primitive_crystal_content_sha256",
        lambda _mapping: _sha256(b"common primitive"),
    )
    monkeypatch.setattr(
        fixture_module,
        "crystal_mapping_content_sha256",
        lambda mapping: _sha256(f"canonical-{mapping.purpose}".encode()),
    )


def _sealed_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    bad_terminal_inventory: bool = False,
    legacy_mismatch: bool = False,
) -> tuple[Path, Path, Path, Path]:
    attempt = tmp_path / "sealed-attempt"
    artifact = attempt / "artifact-root"
    handoff = attempt / "handoff"
    artifact.mkdir(parents=True)
    handoff.mkdir()

    commit = "c" * 40
    tree = "d" * 40
    run_id = "synthetic-release"
    recorded_root = f"/sealed/archive/{run_id}"
    historical = {
        "fc2": _sha256(b"historical fc2 mapping"),
        "fc3": _sha256(b"historical fc3 mapping"),
    }
    monkeypatch.setattr(fixture_module, "SOURCE_COMMIT", commit)
    monkeypatch.setattr(fixture_module, "SOURCE_TREE", tree)
    monkeypatch.setattr(fixture_module, "SOURCE_RUN_ID", run_id)
    monkeypatch.setattr(
        fixture_module, "PARENT_INPUT_MANIFEST_SHA256", _sha256(b"parent manifest")
    )
    monkeypatch.setattr(fixture_module, "HISTORICAL_MAPPING_SHA256", historical)
    _patch_mapping_functions(
        monkeypatch,
        historical=historical,
        legacy_mismatch=legacy_mismatch,
    )

    raw_force_data = {
        "fc2.compact.hdf5": b"synthetic compact fc2\0",
        "fc2.full.hdf5": b"synthetic full fc2\0",
        "fc3.compact.hdf5": b"synthetic compact fc3\0",
        "fc3.full.hdf5": b"synthetic full fc3\0",
    }
    force_data = {
        **raw_force_data,
        "force-constants-stage.json": _canonical({"stage": "sealed"}),
        "input-manifest.normalized.json": _canonical({"input": "sealed"}),
        "source-verification.json": _canonical({"verified": True}),
    }
    for filename, data in force_data.items():
        _write(artifact / "force-constants" / filename, data)
    force_receipt = _sums(
        {filename: _sha256(data) for filename, data in force_data.items()}
    )
    _write(
        artifact / "force-constants/force-constants-SHA256SUMS", force_receipt
    )
    raw_hashes = {
        filename: _sha256(data) for filename, data in raw_force_data.items()
    }
    monkeypatch.setattr(fixture_module, "RAW_FORCE_CONSTANT_SHA256", raw_hashes)

    consistency_documents: dict[str, bytes] = {}
    consistency_paths = (
        "v3/mesh-3x3x3/samples.json",
        "v3/mesh-5x5x5/samples.json",
        "v3/mesh-11x11x11/samples.json",
        "qualifications/mesh-3x3x3/serial-report.json",
        "qualifications/mesh-5x5x5/serial-report.json",
    )
    for relative in consistency_paths:
        document: dict[str, object]
        if relative.startswith("qualifications/"):
            document = {"interaction_sample_manifest": {"mapping_sha256": historical}}
        else:
            document = {"mapping_sha256": historical}
        data = _canonical(document)
        consistency_documents[relative] = data
        _write(artifact / relative, data)
    consistency_hashes = {
        relative: _sha256(data) for relative, data in consistency_documents.items()
    }
    monkeypatch.setattr(
        fixture_module, "HISTORICAL_CONSISTENCY_FILES", consistency_hashes
    )

    evidence_files = [
        {
            "path": f"{recorded_root}/{relative}",
            "sha256": digest,
            "size_bytes": len(consistency_documents[relative]),
        }
        for relative, digest in sorted(consistency_hashes.items())
    ]
    if bad_terminal_inventory:
        evidence_files = evidence_files[:-1]
    report = {
        "schema": "negfpy.silicon_0_2_0_release_evidence.v1",
        "schema_version": 1,
        "release_version": "0.2.0",
        "status": "passed",
        "release_certified": True,
        "validation_passed": True,
        "certification_eligible": True,
        "source_repository": {
            "head_commit": commit,
            "tree_hash": tree,
            "clean": True,
            "dirty_entry_count": 0,
        },
        "evidence_inventory": {
            "file_count": len(evidence_files),
            "files": evidence_files,
        },
    }
    report_data = _canonical(report)
    report_hash = _sha256(report_data)
    _write(artifact / "reports/silicon-release-report.json", report_data)
    monkeypatch.setattr(fixture_module, "TERMINAL_REPORT_SHA256", report_hash)

    yaml_data = b"synthetic authenticated phono3py YAML\n"
    yaml_hash = _sha256(yaml_data)
    object_relative = f"objects/{yaml_hash[:2]}/{yaml_hash}.bin"
    alias_payload = {
        "schema": "negfpy.silicon_external_evidence_aliases.v1",
        "schema_version": 1,
        "recorded_artifact_root": recorded_root,
        "path_semantics": {"objects": "relative"},
        "aliases": [
            {
                "object_relative_path": object_relative,
                "sha256": yaml_hash,
                "size_bytes": len(yaml_data),
                "source_path": "/sealed/source/phono3py_disp.yaml",
            }
        ],
        "objects": [
            {
                "kind": "binary",
                "object_relative_path": object_relative,
                "sha256": yaml_hash,
                "size_bytes": len(yaml_data),
            }
        ],
        "object_inventory_sha256": "",
        "summary": {
            "alias_count": 1,
            "object_count": 1,
            "total_size_bytes": len(yaml_data),
        },
    }
    alias_payload["object_inventory_sha256"] = _sha256(
        _canonical(alias_payload["objects"], trailing_lf=False)
    )
    alias_payload_hash = _sha256(_canonical(alias_payload, trailing_lf=False))
    alias_document = {"payload": alias_payload, "payload_sha256": alias_payload_hash}
    alias_data = _canonical(alias_document)
    alias_hash = _sha256(alias_data)
    alias_relative = f".negfpy-external-evidence/{alias_payload_hash}/aliases.json"
    yaml_relative = (
        Path(alias_relative).parent / Path(object_relative)
    ).as_posix()
    yaml_object = artifact / yaml_relative
    _write(yaml_object, yaml_data)
    _write(artifact / alias_relative, alias_data)
    monkeypatch.setattr(
        fixture_module, "EXTERNAL_ALIAS_MANIFEST_SHA256", alias_hash
    )
    monkeypatch.setattr(
        fixture_module, "EXTERNAL_ALIAS_PAYLOAD_SHA256", alias_payload_hash
    )

    expected_files = {
        "fc2_compact": raw_hashes["fc2.compact.hdf5"],
        "fc2_full": raw_hashes["fc2.full.hdf5"],
        "force_constants_sha256sums": _sha256(force_receipt),
        "force_constants_stage": _sha256(
            force_data["force-constants-stage.json"]
        ),
        "input_manifest_normalized": _sha256(
            force_data["input-manifest.normalized.json"]
        ),
        "phono3py_yaml": yaml_hash,
        "release_report": report_hash,
        "source_verification": _sha256(force_data["source-verification.json"]),
        "v3_sample_manifest": consistency_hashes[
            "v3/mesh-3x3x3/samples.json"
        ],
    }
    monkeypatch.setattr(fixture_module, "EXPECTED_FILE_SHA256", expected_files)

    inventory = []
    for path in sorted(entry for entry in artifact.rglob("*") if entry.is_file()):
        relative = path.relative_to(artifact).as_posix()
        data = path.read_bytes()
        inventory.append(
            {
                "relative_path": relative,
                "sha256": _sha256(data),
                "size_bytes": len(data),
            }
        )
    portable_payload = {
        "schema": "negfpy.silicon_release_portable_evidence.v2",
        "schema_version": 2,
        "recorded_artifact_root": recorded_root,
        "path_semantics": {"inventory": "relative"},
        "inventory": inventory,
        "inventory_sha256": _sha256(_canonical(inventory, trailing_lf=False)),
        "content_summary": {"file_count": len(inventory)},
        "release_report": {
            "relative_path": "reports/silicon-release-report.json",
            "sha256": report_hash,
            "size_bytes": len(report_data),
            "release_certified": True,
            "status": "passed",
        },
        "external_alias_manifest": {
            "relative_path": alias_relative,
            "sha256": alias_hash,
            "size_bytes": len(alias_data),
            "payload_sha256": alias_payload_hash,
        },
    }
    portable_payload_hash = _sha256(
        _canonical(portable_payload, trailing_lf=False)
    )
    portable_data = _canonical(
        {"payload": portable_payload, "payload_sha256": portable_payload_hash}
    )
    portable_hash = _sha256(portable_data)
    portable_path = handoff / "portable-manifest.json"
    _write(portable_path, portable_data)
    _write(handoff / "silicon-release-report.json", report_data)
    handoff_sums_data = _sums(
        {
            "portable-manifest.json": portable_hash,
            "silicon-release-report.json": report_hash,
        }
    )
    handoff_sums = handoff / "SHA256SUMS"
    _write(handoff_sums, handoff_sums_data)
    monkeypatch.setattr(fixture_module, "PORTABLE_MANIFEST_SHA256", portable_hash)
    monkeypatch.setattr(
        fixture_module, "PORTABLE_PAYLOAD_SHA256", portable_payload_hash
    )
    monkeypatch.setattr(
        fixture_module, "HANDOFF_SHA256SUMS_SHA256", _sha256(handoff_sums_data)
    )
    return artifact, yaml_object, portable_path, handoff_sums


def _fixture_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(entry for entry in root.rglob("*") if entry.is_file())
    }


def test_prepare_is_deterministic_and_verifies_nine_role_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, yaml_object, portable, handoff_sums = _sealed_source(
        tmp_path, monkeypatch
    )
    first = tmp_path / "fixture-one"
    second = tmp_path / "fixture-two"

    first_report = prepare_phase_a_fixture(
        artifact_root=artifact,
        yaml_object=yaml_object,
        output=first,
        portable_manifest=portable,
        handoff_sha256sums=handoff_sums,
    )
    second_report = prepare_phase_a_fixture(
        artifact_root=artifact,
        yaml_object=yaml_object,
        output=second,
        portable_manifest=portable,
        handoff_sha256sums=handoff_sums,
    )

    assert first_report == second_report
    assert first_report["verified"] is True
    assert first_report["copied_file_count"] == 9
    assert first_report["covered_file_count"] == 12
    assert _fixture_bytes(first) == _fixture_bytes(second)
    assert verify_phase_a_fixture(first) == first_report

    manifest = json.loads((first / "fixture-manifest.json").read_bytes())
    assert [record["role"] for record in manifest["files"]] == sorted(
        fixture_module.ROLE_SPECS
    )
    assert manifest["files"][-1]["relative_path"] == (
        "v3/mesh-3x3x3/samples.json"
    )


def test_fixture_verifier_rejects_tampered_copied_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, yaml_object, portable, handoff_sums = _sealed_source(
        tmp_path, monkeypatch
    )
    output = tmp_path / "fixture"
    prepare_phase_a_fixture(
        artifact_root=artifact,
        yaml_object=yaml_object,
        output=output,
        portable_manifest=portable,
        handoff_sha256sums=handoff_sums,
    )

    (output / "fc2.compact.hdf5").write_bytes(b"tampered")

    with pytest.raises(PhaseAFixtureError, match="SHA256SUMS mismatch"):
        verify_phase_a_fixture(output)


def test_mapping_mismatch_fails_before_output_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, yaml_object, portable, handoff_sums = _sealed_source(
        tmp_path, monkeypatch, legacy_mismatch=True
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(PhaseAFixtureError, match="legacy FC2/FC3"):
        prepare_phase_a_fixture(
            artifact_root=artifact,
            yaml_object=yaml_object,
            output=output,
            portable_manifest=portable,
            handoff_sha256sums=handoff_sums,
        )

    assert not output.exists()


def test_terminal_evidence_inventory_mismatch_fails_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, yaml_object, portable, handoff_sums = _sealed_source(
        tmp_path, monkeypatch, bad_terminal_inventory=True
    )
    output = tmp_path / "must-not-exist"

    with pytest.raises(PhaseAFixtureError, match="terminal report must contain"):
        prepare_phase_a_fixture(
            artifact_root=artifact,
            yaml_object=yaml_object,
            output=output,
            portable_manifest=portable,
            handoff_sha256sums=handoff_sums,
        )

    assert not output.exists()


def test_existing_destination_is_never_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, yaml_object, portable, handoff_sums = _sealed_source(
        tmp_path, monkeypatch
    )
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "sentinel"
    sentinel.write_text("keep", encoding="utf-8")

    with pytest.raises(PhaseAFixtureError, match="already exists"):
        prepare_phase_a_fixture(
            artifact_root=artifact,
            yaml_object=yaml_object,
            output=output,
            portable_manifest=portable,
            handoff_sha256sums=handoff_sums,
        )

    assert sentinel.read_text(encoding="utf-8") == "keep"
