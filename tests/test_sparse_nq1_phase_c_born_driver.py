from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import (
    CurrentReport,
    DiagnosticReport,
    KeldyshSpectrum,
    SignedFrequencyGrid,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("validate_sparse_nq1_phase_c_born")
campaign = importlib.import_module("run_sparse_nq1_phase_c_born_campaign")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: dict[str, object]) -> bytes:
    return driver._canonical_json(value)


def _write_phase_b_set(root: Path) -> tuple[dict[str, object], Path]:
    (root / "outcomes").mkdir(parents=True)
    (root / "logs").mkdir()
    hdf5_path = root / "outcomes/phase-b-mapped-fc3.h5"
    hdf5_path.write_bytes(b"synthetic mapped vertex")
    hdf5_sha = _sha256(hdf5_path)
    manifest = {
        "schema": "negfpy.sparse-nq1-phase-b-campaign.v1",
        "source": {"commit": "a" * 40, "tree": "b" * 40, "dirty": False},
    }
    (root / "campaign-manifest.json").write_bytes(_canonical(manifest))
    manifest_sha = _sha256(root / "campaign-manifest.json")
    outcome = {
        "campaign_manifest_sha256": manifest_sha,
        "phase_b_certified": True,
        "release_certified": False,
        "scba_executed": False,
        "hdf5": {"sha256": hdf5_sha},
    }
    report = {
        "campaign_manifest_sha256": manifest_sha,
        "phase_b_certified": True,
        "release_certified": False,
        "scba_executed": False,
        "mapped_fc3_hdf5_sha256": hdf5_sha,
    }
    status = {
        "campaign_manifest_sha256": manifest_sha,
        "state": "passed",
        "phase_b_certified": True,
        "release_certified": False,
        "scba_executed": False,
    }
    (root / "outcomes/phase-b-mapping.json").write_bytes(_canonical(outcome))
    (root / "phase-b-report.json").write_bytes(_canonical(report))
    (root / "status.json").write_bytes(_canonical(status))
    (root / "logs/phase-b-mapping.json").write_bytes(
        _canonical({"returncode": 0})
    )
    records = (
        root / "campaign-manifest.json",
        root / "logs/phase-b-mapping.json",
        hdf5_path,
        root / "outcomes/phase-b-mapping.json",
        root / "phase-b-report.json",
    )
    sums = "".join(
        f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n"
        for path in sorted(records, key=lambda item: item.relative_to(root).as_posix())
    )
    (root / "SHA256SUMS").write_text(sums, encoding="ascii")
    archive = {
        "schema": driver.phase_b.ARCHIVE_SCHEMA,
        "sha256": hdf5_sha,
        "ndof": 216,
        "nnz": 6,
        "phase_b_certified": True,
        "release_certified": False,
        "scba_executed": False,
        "metadata": {
            "identities": {
                "raw_fc3_sha256": "c" * 64,
                "sparse_vertex_storage_sha256": "d" * 64,
            }
        },
    }
    return archive, hdf5_path


def _symmetric_vertex() -> SparseCubicVertex:
    indices = np.array(
        [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.int64,
    )
    return SparseCubicVertex(
        216,
        indices,
        np.full(6, 0.1),
        metadata={
            "phase_b_archive_sha256": "e" * 64,
            "phase_b_campaign_manifest_sha256": "f" * 64,
            "phase_b_binding_sha256": "0" * 64,
            "cubic_continuation_factor": 1.0,
            "cubic_continuation_applied_count": 1,
        },
    )


def test_fixed_timing_controls_are_not_scba_controls() -> None:
    assert driver.GRID_STEP == 0.6
    assert driver.GRID_N_POSITIVE == 8
    assert driver.GRID_N_POINTS == 17
    assert driver.MEAN_TEMPERATURE == pytest.approx(0.6250985736998272)
    assert driver.TEMPERATURE_BIAS == 0.0
    assert driver.COUPLING_SCALE == 1.0
    config = driver._solver_config()
    assert config.backend == "fft"
    assert config.hilbert_backend == "direct"
    assert config.retarded_strategy == "causal_hilbert"
    assert config.eta_lead == 0.025
    assert config.eta_device == 0.0
    assert config.max_workers == 1
    assert not config.allow_memory_overcommit
    assert campaign.WALL_LIMIT_SECONDS == 7200
    assert campaign.RSS_LIMIT_BYTES == 4 * 1024**3


def test_complete_phase_b_set_is_bound_and_tamper_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, hdf5_path = _write_phase_b_set(tmp_path)
    monkeypatch.setattr(
        driver,
        "EXPECTED_PHASE_B_SHA256SUMS_SHA256",
        _sha256(tmp_path / "SHA256SUMS"),
    )
    monkeypatch.setattr(driver, "EXPECTED_PHASE_B_SOURCE_COMMIT", "a" * 40)
    monkeypatch.setattr(driver, "EXPECTED_PHASE_B_SOURCE_TREE", "b" * 40)
    monkeypatch.setattr(
        driver, "EXPECTED_PHASE_B_MAPPED_HDF5_SHA256", archive["sha256"]
    )
    monkeypatch.setattr(
        driver.phase_b, "verify_phase_b_hdf5", lambda path: archive
    )
    result = driver.verify_phase_b_artifact_set(tmp_path)
    assert result["archive"]["sha256"] == _sha256(hdf5_path)
    assert len(result["artifact_set_sha256"]) == 64
    assert result["campaign_manifest_sha256"] == _sha256(
        tmp_path / "campaign-manifest.json"
    )

    with (tmp_path / "logs/phase-b-mapping.json").open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(driver.PhaseCError, match="checksum mismatch"):
        driver.verify_phase_b_artifact_set(tmp_path)


def _patch_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path, Path, SimpleNamespace]:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    fc3 = tmp_path / "fc3.full.hdf5"
    fc3.write_bytes(b"sealed full fc3")
    phase_b_root = tmp_path / "phase-b"
    phase_b_root.mkdir()
    vertex = _symmetric_vertex()
    phase_b_set = {
        "artifact_set_sha256": "1" * 64,
        "campaign_manifest_sha256": "f" * 64,
        "hdf5_path": str(phase_b_root / "mapped.h5"),
        "archive": {
            "sha256": "e" * 64,
            "metadata": {
                "identities": {
                    "raw_fc3_sha256": _sha256(fc3),
                    "sparse_vertex_storage_sha256": "2" * 64,
                }
            },
        },
    }
    problem = SimpleNamespace(
        grid=SignedFrequencyGrid(step=0.6, n_positive=8),
        ndof=216,
        vertex=vertex,
    )
    from negfpy.anharmonic.sparse_cubic import _compiled_sparse_schedule

    schedule = _compiled_sparse_schedule(vertex)
    monkeypatch.setattr(driver, "EXPECTED_SPARSE_NNZ", vertex.nnz)
    monkeypatch.setattr(driver, "EXPECTED_SCHEDULE_ACTIVE_DOFS", len(schedule.active))
    monkeypatch.setattr(
        driver, "EXPECTED_SCHEDULE_STORAGE_BYTES", schedule.numeric_storage_bytes
    )
    monkeypatch.setattr(
        driver,
        "EXPECTED_PAIR_MULTIPLY_ADD_COUNT",
        schedule.pair_multiply_add_count,
    )
    monkeypatch.setattr(driver, "EXPECTED_RAW_FC3_SHA256", _sha256(fc3))
    monkeypatch.setattr(
        driver,
        "verify_phase_a_fixture",
        lambda _path: {"fixture_manifest_sha256": "3" * 64},
    )
    monkeypatch.setattr(
        driver, "verify_phase_b_artifact_set", lambda _path: phase_b_set
    )
    monkeypatch.setattr(
        driver,
        "_verify_thread_limits",
        lambda: {"environment": dict(driver.THREAD_LIMIT_ENV), "runtime_pools": []},
    )
    monkeypatch.setattr(driver, "_assemble_problem", lambda *_args: problem)
    monkeypatch.setattr(
        driver, "_guard_dense_memory", lambda *_args: driver.EXPECTED_PREDICTED_MEMORY_BYTES
    )
    monkeypatch.setattr(driver, "_available_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(
        driver,
        "problem_input_hashes",
        lambda _problem: {
            "problem": "4" * 64,
            "cubic_vertex": vertex.vertex_storage_sha256,
        },
    )
    monkeypatch.setattr(driver, "configuration_hash", lambda _config: "5" * 64)
    return fixture, fc3, phase_b_root, problem


def test_preflight_assembles_from_phase_b_without_born_or_remap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture, fc3, phase_b_root, problem = _patch_authentication(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        driver,
        "solve_one_shot_born",
        lambda *_args: pytest.fail("preflight must not execute a Born bubble"),
    )
    output_json = tmp_path / "out/outcome.json"
    output_hdf5 = tmp_path / "out/fields.h5"
    result = driver.preflight_phase_c_inputs(
        fixture=fixture,
        fc3_full=fc3,
        phase_b_artifacts=phase_b_root,
        output_json=output_json,
        output_hdf5=output_hdf5,
    )
    assert result["passed"]
    assert not result["one_shot_born_executed"]
    assert not result["scba_executed"]
    assert result["memory"]["predicted_peak_bytes"] == 755_011_856
    assert result["sparse_schedule"]["fft_padding"] == 64
    assert result["sparse_schedule"]["pair_multiply_add_count"] > 0
    assert result["problem_input_hashes"]["problem"] == "4" * 64
    assert not output_json.exists()
    assert not output_hdf5.exists()
    assert problem.grid.cutoff == 4.8


def _fake_result() -> SimpleNamespace:
    grid = SignedFrequencyGrid(step=0.6, n_positive=8)
    shape = (grid.n_points, 2, 2)
    zero = np.zeros(shape, dtype=np.complex128)
    spectrum = KeldyshSpectrum(zero, zero, zero)
    return SimpleNamespace(
        method="born",
        grid=grid,
        green=spectrum,
        sigma_pp=spectrum,
        sigma_left=spectrum,
        sigma_right=spectrum,
        currents=CurrentReport(left=0.0, right=0.0, collision_power=0.0),
        diagnostics=DiagnosticReport(
            metrics={"contact_balance": 0.0},
            failures=("one-shot is nonconserving",),
            passed=False,
            release_certified=False,
        ),
        history=(),
        converged=True,
        release_certified=False,
        metadata={
            "cubic_kernel": "sparse_zero_padded_fft",
            "effective_backend": "fft",
            "worker_count": 1,
            "requested_worker_count": 1,
            "predicted_peak_memory_bytes": driver.EXPECTED_PREDICTED_MEMORY_BYTES,
            "input_hashes": {
                "problem": "3" * 64,
                "configuration": "4" * 64,
            },
        },
    )


def test_successful_born_archive_stores_full_fields_but_never_certifies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "N_DEVICE_DOFS", 2)
    preflight = {
        "phase_b": {
            "artifact_set_sha256": "1" * 64,
            "archive": {"sha256": "2" * 64},
        },
        "problem_input_hashes": {"problem": "3" * 64},
        "configuration_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        driver, "preflight_phase_c_inputs", lambda **_kwargs: preflight
    )
    monkeypatch.setattr(
        driver,
        "authenticate_phase_c_inputs",
        lambda **_kwargs: ({}, object(), object()),
    )
    monkeypatch.setattr(driver, "solve_one_shot_born", lambda *_args: _fake_result())
    output_json = tmp_path / "outcome.json"
    output_hdf5 = tmp_path / "fields.h5"
    result = driver.run_phase_c_born(
        fixture=tmp_path,
        fc3_full=tmp_path / "fc3",
        phase_b_artifacts=tmp_path / "phase-b",
        output_json=output_json,
        output_hdf5=output_hdf5,
        campaign_manifest_sha256="5" * 64,
    )
    assert result["phase_c_gate_passed"]
    assert result["one_shot_born_executed"]
    assert not result["scba_executed"]
    assert not result["release_certified"]
    assert not result["result"]["nongating_observations"]["diagnostics_passed"]
    verified = driver.verify_phase_c_hdf5(output_hdf5)
    assert verified["one_shot_born_executed"]
    with h5py.File(output_hdf5, "r") as handle:
        assert set(handle["green"]) == {"retarded", "lesser", "greater"}
        assert set(handle["sigma_pp"]) == {"retarded", "lesser", "greater"}
    with pytest.raises(driver.PhaseCError, match="already exists"):
        driver._write_hdf5_once(
            output_hdf5,
            metadata=verified["metadata"],
            arrays=None,
        )


def test_numerical_failure_is_archived_as_scientific_not_infrastructure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight = {
        "phase_b": {
            "artifact_set_sha256": "1" * 64,
            "archive": {"sha256": "2" * 64},
        },
        "problem_input_hashes": {"problem": "3" * 64},
        "configuration_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        driver, "preflight_phase_c_inputs", lambda **_kwargs: preflight
    )
    monkeypatch.setattr(
        driver,
        "authenticate_phase_c_inputs",
        lambda **_kwargs: ({}, object(), object()),
    )
    monkeypatch.setattr(
        driver,
        "solve_one_shot_born",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("singular Dyson fixture")),
    )
    output_hdf5 = tmp_path / "failed.h5"
    result = driver.run_phase_c_born(
        fixture=tmp_path,
        fc3_full=tmp_path / "fc3",
        phase_b_artifacts=tmp_path / "phase-b",
        output_json=tmp_path / "failed.json",
        output_hdf5=output_hdf5,
        campaign_manifest_sha256="5" * 64,
    )
    assert not result["phase_c_gate_passed"]
    assert not result["one_shot_born_executed"]
    assert result["scientific_error"]["classification"] == (
        "scientific-numerical-failure"
    )
    verified = driver.verify_phase_c_hdf5(output_hdf5)
    assert not verified["one_shot_born_executed"]


def test_field_tamper_is_detected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(driver, "N_DEVICE_DOFS", 2)
    result = _fake_result()
    summary = driver._result_summary(result)
    metadata = {
        "one_shot_born_executed": True,
        "result": summary,
        "scba_executed": False,
        "release_certified": False,
    }
    path = tmp_path / "fields.h5"
    driver._write_hdf5_once(
        path, metadata=metadata, arrays=driver._result_arrays(result)
    )
    with h5py.File(path, "r+") as handle:
        handle["green/retarded"][0, 0, 0] = 1.0
    with pytest.raises(driver.PhaseCError, match="field hash mismatch"):
        driver.verify_phase_c_hdf5(path)


def test_campaign_archives_fixed_control_scientific_failure_and_never_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(
        campaign,
        "build_manifest",
        lambda **_kwargs: {
            "schema": campaign.SCHEMA,
            "preflight": {"seal": "fixed"},
            "fixed_controls_no_retuning": True,
        },
    )
    monkeypatch.setattr(
        campaign,
        "_stable_preflight_identity",
        lambda value: {"seal": value["seal"]},
    )
    calls = {"count": 0}

    def scientific_failure(
        _command: tuple[str, ...],
        *,
        repo: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, object]:
        del repo
        calls["count"] += 1
        manifest_sha = _sha256(artifacts / "campaign-manifest.json")
        hdf5 = artifacts / "outcomes/phase-c-born-fields.h5"
        metadata = {
            "campaign_manifest_sha256": manifest_sha,
            "one_shot_born_executed": False,
            "result": None,
            "scba_executed": False,
            "release_certified": False,
        }
        hdf5_sha = driver._write_hdf5_once(hdf5, metadata=metadata, arrays=None)
        outcome = {
            "schema": driver.SCHEMA,
            "campaign_manifest_sha256": manifest_sha,
            "hdf5": {"sha256": hdf5_sha},
            "one_shot_born_executed": False,
            "phase_c_gate_passed": False,
            "preflight": {"seal": "fixed"},
            "scba_executed": False,
            "release_certified": False,
        }
        driver._publish_once(
            artifacts / "outcomes/phase-c-born.json", _canonical(outcome)
        )
        stdout_path.write_text("scientific outcome archived\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return {
            "returncode": 2,
            "elapsed_seconds": 1.0,
            "peak_observed_rss_bytes": 1234,
            "watchdog_trigger": None,
            "killed_after_term_grace": False,
        }

    monkeypatch.setattr(campaign, "_supervised_subprocess", scientific_failure)
    kwargs = {
        "repo": tmp_path,
        "fixture": tmp_path / "fixture",
        "fc3_full": tmp_path / "fc3",
        "phase_b_artifacts": tmp_path / "phase-b",
        "artifacts": artifacts,
        "expected_commit": "a" * 40,
        "environment_lock": tmp_path / "lock",
    }
    status = campaign.run_campaign(**kwargs)
    assert status["state"] == "scientific-failure"
    assert not status["phase_c_gate_passed"]
    assert not status["scba_executed"]
    assert (artifacts / "SHA256SUMS").is_file()
    assert (artifacts / "logs/phase-c-born.json").is_file()
    repeated = campaign.run_campaign(**kwargs)
    assert repeated == status
    assert calls["count"] == 1


def test_campaign_archives_infrastructure_failure_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(
        campaign,
        "build_manifest",
        lambda **_kwargs: {"schema": campaign.SCHEMA, "preflight": {"passed": True}},
    )

    def infrastructure_failure(
        _command: tuple[str, ...],
        *,
        repo: Path,
        stdout_path: Path,
        stderr_path: Path,
    ) -> dict[str, object]:
        del repo
        stdout_path.write_text("", encoding="utf-8")
        stderr_path.write_text("killed by watchdog\n", encoding="utf-8")
        return {
            "returncode": -15,
            "elapsed_seconds": 7200.1,
            "peak_observed_rss_bytes": 100,
            "watchdog_trigger": "wall-timeout",
            "killed_after_term_grace": False,
        }

    monkeypatch.setattr(campaign, "_supervised_subprocess", infrastructure_failure)
    kwargs = {
        "repo": tmp_path,
        "fixture": tmp_path / "fixture",
        "fc3_full": tmp_path / "fc3",
        "phase_b_artifacts": tmp_path / "phase-b",
        "artifacts": artifacts,
        "expected_commit": "a" * 40,
        "environment_lock": tmp_path / "lock",
    }
    status = campaign.run_campaign(**kwargs)
    assert status["state"] == "infrastructure-failure"
    assert not status["automatic_retry"]
    failure = json.loads(
        (artifacts / "infrastructure-failure.json").read_text(encoding="ascii")
    )
    assert failure["supervision"]["watchdog_trigger"] == "wall-timeout"
    assert campaign.run_campaign(**kwargs) == status


def test_partial_or_interrupted_campaign_never_auto_reruns(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "campaign-manifest.json").write_bytes(
        campaign._canonical_json({"schema": campaign.SCHEMA})
    )
    with pytest.raises(campaign.CampaignError, match="automatic rerun forbidden"):
        campaign.run_campaign(
            repo=tmp_path,
            fixture=tmp_path,
            fc3_full=tmp_path / "fc3",
            phase_b_artifacts=tmp_path / "phase-b",
            artifacts=artifacts,
            expected_commit="a" * 40,
            environment_lock=tmp_path / "lock",
        )


def test_thread_limits_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in driver.THREAD_LIMIT_ENV:
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(driver.PhaseCError, match="not exact"):
        driver._verify_thread_limits()


def test_nonfinite_diagnostics_have_strict_deterministic_json_encoding() -> None:
    encoded = driver._plain(
        DiagnosticReport(
            metrics={"positive": np.inf, "negative": -np.inf, "missing": np.nan},
            failures=(),
            passed=False,
            release_certified=False,
        )
    )
    assert encoded["metrics"] == {
        "positive": "Infinity",
        "negative": "-Infinity",
        "missing": "NaN",
    }
    assert b"Infinity" in driver._canonical_json({"diagnostics": encoded})


def test_original_symlink_input_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"input")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(driver.PhaseCError, match="must not be a symlink"):
        driver._regular_file(link, name="test input")


def test_finalization_publication_failure_recovers_without_science_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = tmp_path / "artifacts"
    (artifacts / "outcomes").mkdir(parents=True)
    (artifacts / "logs").mkdir()
    manifest = {
        "schema": campaign.SCHEMA,
        "preflight": {"seal": "fixed"},
    }
    manifest_path = artifacts / "campaign-manifest.json"
    manifest_path.write_bytes(campaign._canonical_json(manifest))
    manifest_sha = _sha256(manifest_path)
    running_path = artifacts / "status-running.json"
    running_path.write_bytes(
        campaign._canonical_json(
            {
                "schema": campaign.STATUS_SCHEMA,
                "campaign_manifest_sha256": manifest_sha,
                "state": "running",
            }
        )
    )
    hdf5 = artifacts / "outcomes/phase-c-born-fields.h5"
    metadata = {
        "campaign_manifest_sha256": manifest_sha,
        "one_shot_born_executed": False,
        "result": None,
        "scba_executed": False,
        "release_certified": False,
    }
    hdf5_sha = driver._write_hdf5_once(hdf5, metadata=metadata, arrays=None)
    outcome = {
        "schema": driver.SCHEMA,
        "campaign_manifest_sha256": manifest_sha,
        "preflight": {"seal": "fixed"},
        "hdf5": {"sha256": hdf5_sha},
        "one_shot_born_executed": False,
        "phase_c_gate_passed": False,
        "scba_executed": False,
        "release_certified": False,
    }
    (artifacts / "outcomes/phase-c-born.json").write_bytes(_canonical(outcome))
    log_path = artifacts / "logs/phase-c-born.json"
    log_path.write_bytes(
        campaign._canonical_json(
            {
                "supervision": {
                    "returncode": 2,
                    "watchdog_trigger": None,
                    "elapsed_seconds": 3.0,
                },
                "stdout": "",
                "stderr": "scientific failure archived",
            }
        )
    )
    monkeypatch.setattr(
        campaign,
        "_stable_preflight_identity",
        lambda value: {"seal": value["seal"]},
    )
    original = campaign._write_checksums
    calls = {"count": 0}

    def fail_once(*args: object, **kwargs: object) -> bytes:
        calls["count"] += 1
        if calls["count"] == 1:
            raise campaign.CampaignError("simulated post-solve checksum publication")
        return original(*args, **kwargs)

    monkeypatch.setattr(campaign, "_write_checksums", fail_once)
    finalization_args = {
        "artifacts": artifacts,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "running_path": running_path,
        "output_json": artifacts / "outcomes/phase-c-born.json",
        "output_hdf5": hdf5,
        "log_path": log_path,
    }
    with pytest.raises(campaign.CampaignError, match="post-solve"):
        campaign._finalize_scientific_outcome(**finalization_args)
    assert (artifacts / "phase-c-report.json").is_file()
    assert not (artifacts / "status.json").exists()

    terminal = campaign._finalize_scientific_outcome(**finalization_args)
    assert terminal["state"] == "scientific-failure"
    assert (artifacts / "SHA256SUMS").is_file()
    assert (artifacts / "status.json").is_file()
    assert calls["count"] == 2
