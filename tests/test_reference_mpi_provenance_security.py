from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil

import pytest

from negfpy.anharmonic._reference_mpi_provenance import (
    ReferenceMPIProvenanceError,
    validate_reference_mpi_evidence,
)
from tests import test_generate_phono3py_si_reference_mpi as fixtures


@pytest.fixture(autouse=True)
def _pin_all_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in fixtures._THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")


def _replace_json_member(
    *,
    payloads: dict[str, bytes],
    member_record: dict[str, object],
    value: object,
) -> None:
    member = str(member_record["archive_member"])
    payloads[member] = fixtures._json_payload(value)
    member_record.clear()
    member_record.update(fixtures._member_identity(member, payloads[member]))


def _validate(output: Path, document: dict[str, object]) -> dict[str, object]:
    return validate_reference_mpi_evidence(
        output,
        document,
        **fixtures._validation_controls(),
    )


def _scratch_cleanup_removed_fixture() -> list[dict[str, object]]:
    return [
        {
            "directory": ".target-grg-00000000-rank1-aaaaaaaa",
            "files": [
                {
                    "relative_path": "a/leaf.bin",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                },
                {
                    "relative_path": "z.bin",
                    "size_bytes": 2,
                    "sha256": "1" * 64,
                },
            ],
        },
        {
            "directory": ".target-grg-00000001-rank2-bbbbbbbb",
            "files": [],
        },
        {
            "directory": ".",
            "files": [
                {
                    "relative_path": (
                        ".reference-mpi-manifest.json.abcdefgh.tmp"
                    ),
                    "size_bytes": 3,
                    "sha256": "2" * 64,
                }
            ],
        },
        {
            "directory": "tasks",
            "files": [
                {
                    "relative_path": (
                        ".target-grg-00000000.json.12345678.tmp"
                    ),
                    "size_bytes": 4,
                    "sha256": "3" * 64,
                }
            ],
        },
    ]


def _replace_scratch_cleanup_removed(
    *,
    output: Path,
    document: dict[str, object],
    payloads: dict[str, bytes],
    removed: list[dict[str, object]],
) -> None:
    execution_summary = document["executions"][0]
    cleanup_identity = execution_summary["scratch_cleanup"]
    cleanup_member = cleanup_identity["archive_member"]
    cleanup = json.loads(payloads[cleanup_member].decode("utf-8"))
    cleanup["removed"] = removed
    _replace_json_member(
        payloads=payloads,
        member_record=cleanup_identity,
        value=cleanup,
    )

    execution_identity = execution_summary["record"]
    execution_member = execution_identity["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    execution["scratch_cleanup"] = {
        "name": Path(cleanup_identity["archive_member"]).name,
        "size_bytes": cleanup_identity["size_bytes"],
        "sha256": cleanup_identity["sha256"],
    }
    _replace_json_member(
        payloads=payloads,
        member_record=execution_identity,
        value=execution,
    )
    fixtures._replace_archive(output, document, payloads)


def _recovered_mpi_bundle(
    tmp_path: Path,
) -> tuple[Path, dict[str, object]]:
    campaign = fixtures._campaign(tmp_path)
    interrupted = fixtures._publish_task(
        campaign,
        0,
        execution_id="interrupted-001",
    )
    _, interrupted_record_path, _ = fixtures.mpi_reference._task_paths(
        campaign.run_dir,
        0,
    )
    interrupted_record_path.unlink()
    recovered = fixtures.mpi_reference._adopt_or_validate_task(campaign, 0)
    assert recovered is not None
    assert recovered["disposition"] == (
        "computed_after_interrupted_record_publication"
    )
    assert recovered["origin_assignment"] == interrupted["origin_assignment"]

    fresh = fixtures._publish_task(
        campaign,
        1,
        execution_id="resume-001",
    )
    fresh_binding = fixtures._worker_binding(
        rank=2,
        logical_cpu=8,
        hostname=fixtures.mpi_reference.socket.gethostname(),
    )
    assignments = [
        {
            "grid_point_grg": 0,
            "rank": None,
            "status": "resumed",
            "wall_seconds": 0.0,
            "origin_assignment": dict(recovered["origin_assignment"]),
        },
        {
            "grid_point_grg": 1,
            "rank": 2,
            "status": "computed",
            "wall_seconds": 1.0,
            "worker_binding": fresh_binding,
            "artifact": dict(fresh["artifact"]),
            "origin_assignment": dict(fresh["origin_assignment"]),
        },
    ]
    cleanup_path, _ = fixtures.mpi_reference._cleanup_interrupted_scratch(
        campaign,
        execution_id="resume-001",
    )
    execution = fixtures.mpi_reference._execution_record(
        campaign=campaign,
        execution_id="resume-001",
        size=3,
        started_utc="2026-01-01T00:00:00+00:00",
        started=0.0,
        assignments=assignments,
        complete=True,
        scratch_cleanup=fixtures.mpi_reference._file_identity(cleanup_path),
    )
    execution_path = fixtures.mpi_reference._execution_path(
        campaign.run_dir,
        "resume-001",
    )
    fixtures.mpi_reference._write_json_exclusive(
        execution_path,
        execution,
    )
    output = fixtures.mpi_reference._finalize_reference_bundle(
        campaign,
        execution_records=[execution],
    )
    document = json.loads(
        (
            output
            / fixtures.mpi_reference.MPI_ORCHESTRATION_FILENAME
        ).read_text(encoding="utf-8")
    )
    return output, document


def _finalized_mpi_bundle_with_stale_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Path,
    dict[str, object],
    str,
    list[dict[str, object]],
]:
    original_campaign = fixtures._campaign
    nonce = "c" * 32
    payloads_by_path = {
        "alpha.bin": b"alpha final-staging residue",
        "nested/z.json": b"nested final-staging residue",
    }
    staging_evidence_directory = f"output-parent/.output.{nonce}.tmp"

    def campaign_with_stale_staging(path: Path):
        campaign = original_campaign(path)
        staging = (
            campaign.output_dir.parent
            / f".{campaign.output_dir.name}.{nonce}.tmp"
        )
        for relative_path, payload in payloads_by_path.items():
            destination = staging / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        return campaign

    monkeypatch.setattr(
        fixtures,
        "_campaign",
        campaign_with_stale_staging,
    )
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    expected_files = [
        {
            "relative_path": relative_path,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for relative_path, payload in payloads_by_path.items()
    ]
    return (
        output,
        document,
        staging_evidence_directory,
        expected_files,
    )


def test_adopted_tasks_cannot_claim_fresh_mpi_numerical_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)

    for task in document["tasks"]:
        record_identity = task["record"]
        member = record_identity["archive_member"]
        task_record = json.loads(payloads[member].decode("utf-8"))
        task_record["disposition"] = (
            "adopted_after_interrupted_record_publication"
        )
        _replace_json_member(
            payloads=payloads,
            member_record=record_identity,
            value=task_record,
        )

    execution_identity = document["executions"][0]["record"]
    execution_member = execution_identity["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    for assignment in execution["assignments"]:
        assignment["status"] = "adopted"
    _replace_json_member(
        payloads=payloads,
        member_record=execution_identity,
        value=execution,
    )
    fixtures._replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="adopted legacy artifact",
    ):
        _validate(output, document)


@pytest.mark.parametrize(
    "location",
    (
        "execution",
        "assignment",
        "task",
        "task_execution",
        "task_artifact",
        "origin",
    ),
)
def test_reference_records_reject_undeclared_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)

    if location in {"execution", "assignment"}:
        identity = document["executions"][0]["record"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        target = record if location == "execution" else record["assignments"][0]
    elif location == "origin":
        identity = document["tasks"][0]["origin_assignment"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        target = record
    else:
        identity = document["tasks"][0]["record"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        target = {
            "task": record,
            "task_execution": record["execution"],
            "task_artifact": record["artifact"],
        }[location]
    target["undeclared_security_field"] = True
    _replace_json_member(
        payloads=payloads,
        member_record=identity,
        value=record,
    )
    fixtures._replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="ambiguous schema|task record .* inconsistent|origin .* inconsistent",
    ):
        _validate(output, document)


@pytest.mark.parametrize(
    "location",
    ("origin", "task_execution", "execution"),
)
def test_thread_records_reject_seventh_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    location: str,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)

    if location == "origin":
        identity = document["tasks"][0]["origin_assignment"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        threads = record["thread_environment"]
    elif location == "task_execution":
        identity = document["tasks"][0]["record"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        threads = record["execution"]["thread_environment"]
    else:
        identity = document["executions"][0]["record"]
        member = identity["archive_member"]
        record = json.loads(payloads[member].decode("utf-8"))
        threads = record["thread_environment"]
    threads["UNDECLARED_NUM_THREADS"] = "1"
    _replace_json_member(
        payloads=payloads,
        member_record=identity,
        value=record,
    )
    fixtures._replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="exactly the six pinned thread variables",
    ):
        _validate(output, document)


@pytest.mark.parametrize(
    "filename",
    (
        fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME,
        "si-phono3py-reference.hdf5",
        "reference-settings.json",
    ),
)
def test_hard_linked_immutable_evidence_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    source = output / filename
    os.link(source, tmp_path / f"second-link-{filename}")

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="must not be hard-linked",
    ):
        _validate(output, document)


def test_portable_ordinary_file_relocation_remains_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    relocated = tmp_path / "relocated-reference-bundle"
    shutil.copytree(output, relocated)

    validated = _validate(relocated, document)

    assert validated["task_count"] == 2
    assert validated["full_q_selection"] is True


def test_generator_ordered_nested_scratch_cleanup_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)
    _replace_scratch_cleanup_removed(
        output=output,
        document=document,
        payloads=payloads,
        removed=_scratch_cleanup_removed_fixture(),
    )

    validated = _validate(output, document)

    assert validated["execution_count"] == 1


@pytest.mark.parametrize(
    "attack",
    (
        "removed_extra_key",
        "file_extra_key",
        "duplicate_directory",
        "duplicate_file",
        "traversal",
        "noncanonical_path",
        "invalid_directory",
        "worker_after_private",
        "unordered_files",
        "unordered_private_scopes",
        "invalid_private_temporary",
        "empty_private_scope",
        "boolean_size",
        "invalid_sha256",
    ),
)
def test_nested_scratch_cleanup_attacks_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attack: str,
) -> None:
    _, output, document = fixtures._finalized_mpi_bundle(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)
    removed = _scratch_cleanup_removed_fixture()

    if attack == "removed_extra_key":
        removed[0]["undeclared"] = True
    elif attack == "file_extra_key":
        removed[0]["files"][0]["undeclared"] = True
    elif attack == "duplicate_directory":
        removed.insert(1, json.loads(json.dumps(removed[0])))
    elif attack == "duplicate_file":
        removed[0]["files"].append(
            json.loads(json.dumps(removed[0]["files"][0]))
        )
    elif attack == "traversal":
        removed[0]["files"][0]["relative_path"] = "../escape.bin"
    elif attack == "noncanonical_path":
        removed[0]["files"][0]["relative_path"] = "a//leaf.bin"
    elif attack == "invalid_directory":
        removed[0]["directory"] = "target-grg-00000000-rank1-aaaaaaaa"
    elif attack == "worker_after_private":
        removed[0], removed[2] = removed[2], removed[0]
    elif attack == "unordered_files":
        removed[0]["files"].reverse()
    elif attack == "unordered_private_scopes":
        removed[2], removed[3] = removed[3], removed[2]
    elif attack == "invalid_private_temporary":
        removed[2]["files"][0]["relative_path"] = ".unowned.abcdefgh.tmp"
    elif attack == "empty_private_scope":
        removed[2]["files"] = []
    elif attack == "boolean_size":
        removed[0]["files"][0]["size_bytes"] = True
    elif attack == "invalid_sha256":
        removed[0]["files"][0]["sha256"] = "A" * 64
    else:  # pragma: no cover - parameter list and cases are kept aligned.
        raise AssertionError(f"unknown attack fixture {attack!r}")

    _replace_scratch_cleanup_removed(
        output=output,
        document=document,
        payloads=payloads,
        removed=removed,
    )

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match=(
            "scratch cleanup .*"
            "(ambiguous|invalid|noncanonical|generator order|sha256)"
        ),
    ):
        _validate(output, document)


def test_computed_after_recovery_requires_resumed_execution_lineage(
    tmp_path: Path,
) -> None:
    output, document = _recovered_mpi_bundle(tmp_path)

    validated = _validate(output, document)

    assert validated["restart"]["total_resumed_task_assignments"] == 1
    assert validated["restart"]["executions_with_resumed_tasks"] == [
        "resume-001"
    ]

    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)
    execution_identity = document["executions"][0]["record"]
    execution_member = execution_identity["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    task_identity = document["tasks"][0]["record"]
    task_record = json.loads(
        payloads[task_identity["archive_member"]].decode("utf-8")
    )
    origin_identity = document["tasks"][0]["origin_assignment"]
    origin = json.loads(
        payloads[origin_identity["archive_member"]].decode("utf-8")
    )
    assignment = execution["assignments"][0]
    assignment.update(
        {
            "rank": origin["rank"],
            "status": "computed",
            "worker_binding": origin["worker_binding"],
            "artifact": dict(task_record["artifact"]),
        }
    )
    _replace_json_member(
        payloads=payloads,
        member_record=execution_identity,
        value=execution,
    )
    fixtures._replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="fresh task 0 lineage differs",
    ):
        _validate(output, document)


def test_canonical_abandoned_final_staging_round_trip_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (
        output,
        document,
        staging_directory,
        expected_files,
    ) = _finalized_mpi_bundle_with_stale_staging(
        tmp_path,
        monkeypatch,
    )
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)
    cleanup_identity = document["executions"][0]["scratch_cleanup"]
    cleanup = json.loads(
        payloads[cleanup_identity["archive_member"]].decode("utf-8")
    )
    staging_cleanup = next(
        item
        for item in cleanup["removed"]
        if item["directory"] == staging_directory
    )

    assert staging_cleanup["files"] == expected_files
    validated = _validate(output, document)
    assert validated["execution_count"] == 1


@pytest.mark.parametrize(
    "replacement",
    (
        f"output-parentx/.output.{'c' * 32}.tmp",
        f"output-parent/.other-output.{'c' * 32}.tmp",
        f"output-parent/.output.{'c' * 31}.tmp",
    ),
    ids=("prefix", "name", "nonce"),
)
def test_abandoned_final_staging_near_matches_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    replacement: str,
) -> None:
    (
        output,
        document,
        staging_directory,
        _,
    ) = _finalized_mpi_bundle_with_stale_staging(
        tmp_path,
        monkeypatch,
    )
    document = json.loads(json.dumps(document))
    archive = output / fixtures.mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = fixtures._archive_payloads(archive)
    cleanup_identity = document["executions"][0]["scratch_cleanup"]
    cleanup = json.loads(
        payloads[cleanup_identity["archive_member"]].decode("utf-8")
    )
    staging_cleanup = next(
        item
        for item in cleanup["removed"]
        if item["directory"] == staging_directory
    )
    staging_cleanup["directory"] = replacement
    _replace_scratch_cleanup_removed(
        output=output,
        document=document,
        payloads=payloads,
        removed=cleanup["removed"],
    )

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="scratch cleanup .* invalid directory",
    ):
        _validate(output, document)
