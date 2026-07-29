from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from negfpy.anharmonic._bulk_target_q_mpi import (
    BulkTargetQControls,
    build_bulk_target_q_campaign,
)


SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("run_phono3py_si_target_q_mpi")


def _prepared_campaign() -> tuple[dict, object]:
    controls = BulkTargetQControls(
        temperatures=[300.0],
        grid_points=[0, 2],
        band_indices=[0, 1],
        n_qpoints=3,
        n_bands=2,
        problem_sha256="a" * 64,
        input_hashes={
            "target_environment_lock": "b" * 64,
            "target_reference": "c" * 64,
        },
    )
    metadata = {
        "driver_schema": driver.CAMPAIGN_CLI_SCHEMA,
        "internal_q1_mpi": False,
    }
    campaign = build_bulk_target_q_campaign(
        controls,
        source_revision="d" * 40,
        source_tree_sha256="e" * 64,
        environment_sha256="b" * 64,
        metadata=metadata,
    )
    prepared = {
        "schema": driver.CAMPAIGN_CLI_SCHEMA,
        "schema_version": 1,
        "code_identity": {
            "head_commit": "d" * 40,
            "source_tree_sha256": "e" * 64,
        },
        "problem_spec": {},
        "problem_sha256": "a" * 64,
        "controls": controls.as_dict(),
        "campaign_metadata": metadata,
        "campaign": campaign.as_dict(),
        "input_records": {},
        "memory_preflight": {
            "schema": driver.MEMORY_PREFLIGHT_SCHEMA,
            "passed": True,
        },
        "memory_override_used": False,
    }
    return prepared, campaign


def _required_cli(tmp_path: Path) -> list[str]:
    return [
        "--serial-report",
        str(tmp_path / "qualification.json"),
        "--reference",
        str(tmp_path / "target.hdf5"),
        "--fc2-compact",
        str(tmp_path / "fc2.hdf5"),
        "--fc3-compact",
        str(tmp_path / "fc3.hdf5"),
        "--phono3py-yaml",
        str(tmp_path / "phono3py.yaml"),
        "--environment-lock",
        str(tmp_path / "environment.lock"),
        "--target-checksums",
        str(tmp_path / "SHA256SUMS"),
        "--target-input-manifest",
        str(tmp_path / "input-manifest.normalized.json"),
        "--repository",
        str(tmp_path / "repository"),
        "--run-directory",
        str(tmp_path / "run"),
    ]


def test_dry_run_writes_canonical_manifest_without_loading_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prepared, campaign = _prepared_campaign()
    calls = 0

    def fake_prepare(
        arguments: argparse.Namespace,
        *,
        communicator_size: int,
    ) -> tuple[dict, object]:
        nonlocal calls
        calls += 1
        assert communicator_size == 1
        return prepared, object()

    monkeypatch.setattr(driver, "_rank_zero_prepare", fake_prepare)
    monkeypatch.setattr(
        driver,
        "_load_problem",
        lambda spec: pytest.fail("dry-run loaded material data"),
    )
    manifest_path = tmp_path / "manifest.json"
    status = driver.main(
        _required_cli(tmp_path)
        + [
            "--dry-run",
            "--manifest-output",
            str(manifest_path),
            "--grid-points",
            "2",
            "0",
        ]
    )

    assert status == 0
    assert calls == 1
    assert not (tmp_path / "run").exists()
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry-run"
    assert (
        payload["campaign"]["campaign_manifest_sha256"]
        == campaign.campaign_manifest_sha256
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["payload_sha256"] == campaign.campaign_manifest_sha256


def test_one_rank_mpi_is_rejected_before_material_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    communicator = SimpleNamespace(Get_size=lambda: 1)
    monkeypatch.setattr(driver, "_communicator", lambda mode: communicator)
    monkeypatch.setattr(
        driver,
        "_rank_zero_prepare",
        lambda *args, **kwargs: pytest.fail("material preflight started"),
    )

    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="requires at least two MPI ranks",
    ):
        driver.main(_required_cli(tmp_path) + ["--mode", "mpi", "--dry-run"])


def test_passed_serial_report_is_mandatory(tmp_path: Path) -> None:
    path = tmp_path / "serial.json"
    path.write_text(
        json.dumps(
            {
                "schema": driver.SILICON_ORACLE_REPORT_SCHEMA,
                "schema_version": 1,
                "status": "passed",
                "validation_passed": True,
                "certification_eligible": False,
                "certification_gates": {"scientific_comparison": True},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="certification-eligible",
    ):
        driver._load_passed_serial_report(path)


def test_historical_serial_report_cannot_qualify_current_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    head = "1" * 40
    tree = "2" * 40
    source_hash = "3" * 64

    def fake_git(path: Path, *arguments: str) -> str:
        assert path == repository
        if arguments == ("rev-parse", "HEAD"):
            return head
        if arguments == ("rev-parse", "HEAD^{tree}"):
            return tree
        if arguments[:2] == ("status", "--porcelain=v1"):
            return ""
        raise AssertionError(arguments)

    monkeypatch.setattr(driver, "_git", fake_git)
    monkeypatch.setattr(
        driver,
        "code_provenance",
        lambda: {
            "git_revision": head,
            "git_dirty": False,
            "package_version_consistent": True,
            "source_tree_sha256": source_hash,
        },
    )
    report = {
        "source_repository": {
            "repository": repository.as_posix(),
            "head_commit": "0" * 40,
            "tree_hash": tree,
            "clean": True,
        },
        "imported_code_provenance": {
            "passed": True,
            "record": {
                "git_revision": "0" * 40,
                "git_dirty": False,
                "source_tree_sha256": source_hash,
            },
        },
    }

    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="Historical reports cannot authorize",
    ):
        driver._clean_code_identity(repository, report)


@pytest.mark.parametrize("invalid", [[True], [1.5], ["1"]])
def test_integer_selectors_reject_noninteger_values(
    invalid: list[object],
) -> None:
    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="only integers",
    ):
        driver._integer_sequence(
            invalid,
            default=(),
            upper=3,
            name="selection",
        )


def test_target_problem_controls_are_not_locked_to_qualification_mesh() -> None:
    target = SimpleNamespace(
        mesh=(11, 11, 11),
        n_qpoints=1331,
        n_bands=6,
        grid_points_grg=np.array([0, 17, 1330], dtype=np.int64),
        band_indices=np.arange(6, dtype=np.int64),
        temperatures_k=np.array([300.0, 600.0]),
        sigmas_thz=np.array([0.1]),
        cutoff_frequency_thz=1.0e-4,
        make_r0_average=True,
    )
    report = {
        "controls": {"asr_atol": 1.0e-10, "asr_rtol": 1.0e-8},
        "interaction_sample_manifest": {
            "degeneracy_tolerance_thz": 1.0e-4
        },
        "selections": {"mesh": [3, 3, 3]},
    }
    records = {
        "target_bundle_si-phono3py-reference.hdf5": SimpleNamespace(
            sha256="1" * 64
        ),
        "target_bundle_fc2.compact.hdf5": SimpleNamespace(
            sha256="2" * 64
        ),
        "target_bundle_fc3.compact.hdf5": SimpleNamespace(
            sha256="3" * 64
        ),
    }
    spec, temperatures, targets, bands, n_qpoints, n_bands = (
        driver._problem_spec(
            report,
            {
                "fc2_compact": Path("/target/fc2.compact.hdf5"),
                "fc3_compact": Path("/target/fc3.compact.hdf5"),
                "phono3py_yaml": Path("/source/phono3py.yaml"),
            },
            records,
            target,
            {
                "fc2.full.hdf5": "4" * 64,
                "fc3.full.hdf5": "5" * 64,
            },
            driver.PINNED_SILICON_SOURCE_COMMIT,
        )
    )

    assert spec.mesh == (11, 11, 11)
    assert n_qpoints == 1331
    assert n_bands == 6
    assert targets == (0, 17, 1330)
    assert bands == tuple(range(6))
    assert temperatures == (300.0, 600.0)


def test_memory_preflight_accounts_for_complete_canonical_merge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_deep_resident_bytes", lambda problem: 100)
    common = {
        "problem": object(),
        "n_tasks": 5,
        "n_qpoints": 27,
        "n_bands": 6,
        "n_target_bands": 6,
        "n_temperatures": 1,
        "communicator_size": 4,
        "available_memory_bytes": 10**15,
    }
    ordinary = driver._memory_preflight(
        **common,
        compare_monolithic=False,
    )
    comparison = driver._memory_preflight(
        **common,
        compare_monolithic=True,
    )
    full_merge = driver._target_kernel_bytes(
        n_targets=5,
        n_qpoints=27,
        n_target_bands=6,
        n_bands=6,
        n_temperatures=1,
    )

    assert ordinary["active_worker_count"] == 3
    assert (
        ordinary["coordinator_merge_allowance_bytes"]
        == 100 + full_merge + driver.FIXED_PROCESS_OVERHEAD_BYTES
    )
    assert (
        comparison["coordinator_merge_allowance_bytes"]
        == ordinary["coordinator_merge_allowance_bytes"] + full_merge
    )


def test_memory_preflight_marks_over_limit_without_silent_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "_deep_resident_bytes", lambda problem: 1)
    report = driver._memory_preflight(
        object(),
        n_tasks=1,
        n_qpoints=1,
        n_bands=1,
        n_target_bands=1,
        n_temperatures=1,
        communicator_size=1,
        compare_monolithic=False,
        available_memory_bytes=1024,
    )

    assert not report["passed"]
    assert report["aggregate_predicted_bytes"] > report["limit_bytes"]


def test_lazy_problem_factory_loads_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded: list[object] = []
    sentinel = object()
    monkeypatch.setattr(
        driver,
        "_load_problem",
        lambda spec: loaded.append(spec) or sentinel,
    )
    spec = driver.ProblemLoadSpec(
        fc2_compact_path="/fc2",
        fc3_compact_path="/fc3",
        phono3py_yaml_path="/yaml",
        mesh=(1, 1, 1),
        sigma_thz=0.1,
        cutoff_frequency_thz=1.0e-4,
        frequency_factor_to_thz=1.0,
        degenerate_tolerance_thz=1.0e-4,
        make_r0_average=True,
        symprec=1.0e-5,
        translation_atol=1.0e-12,
        translation_rtol=1.0e-10,
        asr_atol=1.0e-10,
        asr_rtol=1.0e-8,
        metadata={},
    )
    factory = driver._LazyProblemFactory(spec)

    assert factory() is sentinel
    assert factory() is sentinel
    assert loaded == [spec]


def test_run_directory_failure_is_broadcast_without_barrier(
    tmp_path: Path,
) -> None:
    class FakeRootCommunicator:
        def bcast(self, value: object, *, root: int) -> object:
            assert root == 0
            return value

    existing = tmp_path / "run"
    existing.mkdir()
    arguments = SimpleNamespace(
        run_directory=existing,
        resume=False,
    )
    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="initialization failed",
    ):
        driver._initialize_run_directory(
            arguments,
            communicator=FakeRootCommunicator(),
            rank=0,
            execution_id="test",
            prepared={},
        )


def test_final_code_identity_is_reauthenticated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    serial_report = tmp_path / "qualification.json"
    serial_report.write_text("{}", encoding="ascii")
    identity = {
        "repository": repository.as_posix(),
        "head_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    prepared = {"code_identity": identity}
    records = {
        "serial_report": driver.FileIdentity.from_path(serial_report)
    }
    monkeypatch.setattr(
        driver,
        "_load_passed_serial_report",
        lambda path: {"path": path.as_posix()},
    )
    monkeypatch.setattr(
        driver,
        "_clean_code_identity",
        lambda path, report: dict(identity),
    )

    driver._code_identity_unchanged(prepared, records)

    monkeypatch.setattr(
        driver,
        "_clean_code_identity",
        lambda path, report: {
            **identity,
            "head_commit": "c" * 40,
        },
    )
    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="code identity changed",
    ):
        driver._code_identity_unchanged(prepared, records)


def test_target_source_manifest_binds_qualified_dataset_and_yaml() -> None:
    yaml_identity = SimpleNamespace(sha256="f" * 64)
    report = {
        "source_manifest": {
            "input_manifest_sha256": "a" * 64,
            "phono3py_yaml_sha256": "f" * 64,
        }
    }
    files = {
        "phono3py_yaml": {
            "path": "example/Si-PBEsol/phono3py_disp.yaml",
            "sha256": "f" * 64,
        }
    }
    normalized = {
        "schema_version": 1,
        "dataset_id": driver.PINNED_SILICON_DATASET_ID,
        "input_manifest_sha256": "a" * 64,
        "versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"},
        "source": {
            "repository": driver.PINNED_SILICON_REPOSITORY,
            "commit": driver.PINNED_SILICON_SOURCE_COMMIT,
            "files": files,
        },
    }
    verified = {
        "schema_version": 1,
        "dataset_id": driver.PINNED_SILICON_DATASET_ID,
        "repository": driver.PINNED_SILICON_REPOSITORY,
        "commit": driver.PINNED_SILICON_SOURCE_COMMIT,
        "files": {
            "phono3py_yaml": {
                "path_in_source": files["phono3py_yaml"]["path"],
                "sha256": "f" * 64,
                "size_bytes": 123,
            }
        },
    }

    assert (
        driver._validate_target_source_records(
            normalized_manifest=normalized,
            source_verification=verified,
            serial_report=report,
            yaml_identity=yaml_identity,
        )
        == driver.PINNED_SILICON_SOURCE_COMMIT
    )
    normalized["source"]["commit"] = "0" * 40
    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="pinned silicon",
    ):
        driver._validate_target_source_records(
            normalized_manifest=normalized,
            source_verification=verified,
            serial_report=report,
            yaml_identity=yaml_identity,
        )


def test_target_sha256sums_rejects_nested_paths(tmp_path: Path) -> None:
    path = tmp_path / "SHA256SUMS"
    path.write_text(f"{'a' * 64}  nested/file\n", encoding="utf-8")

    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="unsafe",
    ):
        driver._parse_sha256sums(path)


def test_final_summary_is_self_contained_and_not_scientifically_certified(
    tmp_path: Path,
) -> None:
    prepared, campaign = _prepared_campaign()
    run = tmp_path / "run"
    (run / "executions").mkdir(parents=True)
    (run / "reports").mkdir()
    (run / "merged").mkdir()
    for path in (
        run / "task-manifest.json",
        run / "executions" / "execution.json",
        run / "reports" / "merge.json",
        run / "merged" / "bulk-target-q.h5",
    ):
        path.write_bytes(path.name.encode("ascii"))
    qualification = tmp_path / "qualification.json"
    reference = tmp_path / "reference.hdf5"
    generation = tmp_path / "reference-stage.json"
    for path in (qualification, reference, generation):
        path.write_bytes(path.name.encode("ascii"))
    prepared["input_records"] = {
        "serial_report": driver.FileIdentity.from_path(
            qualification
        ).as_dict(),
        "target_bundle_si-phono3py-reference.hdf5": (
            driver.FileIdentity.from_path(reference).as_dict()
        ),
        "target_bundle_reference-stage.json": (
            driver.FileIdentity.from_path(generation).as_dict()
        ),
    }
    execution = SimpleNamespace(
        as_dict=lambda: {"payload": {"complete": True}},
        execution_sha256="4" * 64,
        complete=True,
        mode="mpi",
        communicator_size=3,
        assignments=(
            SimpleNamespace(
                task_id=campaign.manifest.tasks[0].task_id,
                resumed=True,
                rank=1,
            ),
            SimpleNamespace(
                task_id=campaign.manifest.tasks[1].task_id,
                resumed=False,
                rank=2,
            ),
        ),
    )
    merge = SimpleNamespace(
        as_dict=lambda: {"report_sha256": "5" * 64},
        report_sha256="5" * 64,
        merged_artifact_path=run / "merged" / "bulk-target-q.h5",
    )

    summary = driver._final_summary(
        execution=execution,
        merge=merge,
        comparison=None,
        campaign=campaign,
        run_directory=run,
        prepared=prepared,
        execution_id="execution",
    )

    assert summary["status"] == "passed"
    assert not summary["complete_target_set"]
    assert summary["complete_band_set"]
    assert not summary["complete_scientific_selection"]
    assert not summary["external_validation_eligible"]
    assert summary["execution_qualification"] == {
        "executed_task_count": 1,
        "resumed_task_count": 1,
        "orphan_artifact_adopted_task_count": 0,
        "executed_task_ids": [campaign.manifest.tasks[1].task_id],
        "resumed_task_ids": [campaign.manifest.tasks[0].task_id],
        "orphan_artifact_adopted_task_ids": [],
        "fresh_complete_task_execution": False,
        "fresh_complete_mpi_execution": False,
        "mpi_equality_eligible": False,
    }
    assert summary["scientific_certification"] == {
        "certified": False,
        "status": "pending_external_comparison",
        "reason": (
            "Campaign completion and optional monolithic equality do not "
            "replace comparison with the independent target phono3py "
            "reference."
        ),
    }
    assert summary["release_certified"] is False
    assert summary["evidence"]["task_manifest"]["path"] == (
        run / "task-manifest.json"
    ).as_posix()
    assert (
        summary["evidence"]["qualification_report"]["sha256"]
        == driver._sha256_file(qualification)
    )
    assert (
        summary["evidence"]["target_generation_records"][
            "target_bundle_reference-stage.json"
        ]["sha256"]
        == driver._sha256_file(generation)
    )

    execution.assignments[0].orphan_artifact_adopted = True
    adopted_summary = driver._final_summary(
        execution=execution,
        merge=merge,
        comparison=None,
        campaign=campaign,
        run_directory=run,
        prepared=prepared,
        execution_id="execution",
    )
    adopted = adopted_summary["execution_qualification"]
    assert adopted["orphan_artifact_adopted_task_count"] == 1
    assert adopted["orphan_artifact_adopted_task_ids"] == [
        campaign.manifest.tasks[0].task_id
    ]
    assert not adopted["fresh_complete_task_execution"]
    assert not adopted["fresh_complete_mpi_execution"]
    assert not adopted["mpi_equality_eligible"]

    execution.assignments[0].resumed = False
    with pytest.raises(
        driver.SiliconTargetQCLIError,
        match="adopted preexisting artifact",
    ):
        driver._final_summary(
            execution=execution,
            merge=merge,
            comparison=None,
            campaign=campaign,
            run_directory=run,
            prepared=prepared,
            execution_id="execution",
        )
