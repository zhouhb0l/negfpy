from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_MPI_CONFIG_FIELDS = (
    ("qualification_report", "qualification_report_path"),
    ("campaign_summary", "campaign_summary_path"),
    ("merge_report", "merge_report_path"),
    ("merged_result", "merged_result_path"),
    ("reference", "reference_path"),
    ("fc2_compact", "fc2_compact_path"),
    ("fc2_full", "fc2_full_path"),
    ("fc3_compact", "fc3_compact_path"),
    ("fc3_full", "fc3_full_path"),
    ("phono3py_yaml", "phono3py_yaml_path"),
    ("input_manifest", "input_manifest_path"),
    ("checksums", "checksums_path"),
    ("environment_lock", "environment_lock_path"),
    ("interaction_samples", "interaction_samples_path"),
    (
        "interaction_samples_orchestration",
        "interaction_samples_orchestration_path",
    ),
    (
        "interaction_samples_run_directory",
        "interaction_samples_run_directory_path",
    ),
    ("repository", "repository_path"),
    ("output", "output_path"),
)


def _load_driver(script_name: str) -> ModuleType:
    path = _SCRIPTS / script_name
    spec = importlib.util.spec_from_file_location(
        f"negfpy_test_{path.stem}",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load validation driver {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _mpi_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        name: tmp_path / filename
        for name, filename in (
            ("qualification_report", "qualification.json"),
            ("campaign_summary", "campaign-summary.json"),
            ("merge_report", "merge-report.json"),
            ("merged_result", "merged-result.h5"),
            ("reference", "reference.hdf5"),
            ("fc2_compact", "fc2.compact.hdf5"),
            ("fc2_full", "fc2.full.hdf5"),
            ("fc3_compact", "fc3.compact.hdf5"),
            ("fc3_full", "fc3.full.hdf5"),
            ("phono3py_yaml", "phono3py.yaml"),
            ("input_manifest", "input-manifest.json"),
            ("checksums", "SHA256SUMS"),
            ("environment_lock", "environment.lock"),
            ("interaction_samples", "interaction-samples.json"),
            (
                "interaction_samples_orchestration",
                "interaction-samples-orchestration.json",
            ),
            (
                "interaction_samples_run_directory",
                "interaction-samples-run",
            ),
            ("repository", "repository"),
            ("output", "mpi-oracle-report.json"),
        )
    }


def _mpi_arguments(paths: dict[str, Path]) -> list[str]:
    return [
        item
        for name, _ in _MPI_CONFIG_FIELDS
        for item in (f"--{name.replace('_', '-')}", str(paths[name]))
    ]


def _release_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "external_oracle_report": tmp_path / "external-oracle.json",
        "equality_3x3_summary": tmp_path / "equality-3x3.json",
        "equality_5x5_summary": tmp_path / "equality-5x5.json",
        "repository": tmp_path / "repository",
        "output": tmp_path / "release-report.json",
    }


def _release_arguments(paths: dict[str, Path]) -> list[str]:
    return [
        item
        for name in (
            "external_oracle_report",
            "equality_3x3_summary",
            "equality_5x5_summary",
            "repository",
            "output",
        )
        for item in (f"--{name.replace('_', '-')}", str(paths[name]))
    ]


def test_mpi_oracle_main_constructs_config_and_prints_passed_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = _load_driver("validate_phono3py_si_mpi_oracle.py")
    paths = _mpi_paths(tmp_path)
    captured: dict[str, object] = {}
    config = object()

    def config_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return config

    def validate(actual_config: object) -> dict[str, object]:
        assert actual_config is config
        return {
            "status": "passed",
            "validation_passed": True,
            "certification_eligible": True,
            "release_certified": False,
            "report_artifact": {
                "path": str(paths["output"].resolve()),
                "sha256": "a" * 64,
            },
        }

    monkeypatch.setattr(
        driver,
        "SiliconMPIOracleValidationConfig",
        config_factory,
    )
    monkeypatch.setattr(
        driver,
        "run_silicon_mpi_oracle_validation",
        validate,
    )

    assert driver.main(_mpi_arguments(paths)) == 0

    assert captured == {
        config_name: paths[argument_name]
        for argument_name, config_name in _MPI_CONFIG_FIELDS
    } | {"require_clean_source": True}
    streams = capsys.readouterr()
    assert streams.err == ""
    assert json.loads(streams.out) == {
        "status": "passed",
        "validation_passed": True,
        "certification_eligible": True,
        "release_certified": False,
        "output": str(paths["output"].resolve()),
        "sha256": "a" * 64,
    }


def test_mpi_oracle_main_returns_one_for_published_failed_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = _load_driver("validate_phono3py_si_mpi_oracle.py")
    paths = _mpi_paths(tmp_path)
    monkeypatch.setattr(
        driver,
        "SiliconMPIOracleValidationConfig",
        lambda **kwargs: object(),
    )
    monkeypatch.setattr(
        driver,
        "run_silicon_mpi_oracle_validation",
        lambda config: {
            "status": "failed",
            "validation_passed": False,
            "certification_eligible": False,
            "release_certified": False,
            "report_artifact": {
                "path": str(paths["output"].resolve()),
                "sha256": "b" * 64,
            },
        },
    )

    assert driver.main(_mpi_arguments(paths)) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "failed"
    assert output["validation_passed"] is False
    assert output["sha256"] == "b" * 64


def test_release_main_constructs_config_and_prints_certification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    driver = _load_driver("validate_phono3py_si_release.py")
    paths = _release_paths(tmp_path)
    captured: dict[str, object] = {}
    config = object()

    def config_factory(**kwargs: object) -> object:
        captured.update(kwargs)
        return config

    def validate(actual_config: object) -> dict[str, object]:
        assert actual_config is config
        return {
            "status": "passed",
            "validation_passed": True,
            "certification_eligible": True,
            "release_certified": True,
            "release_version": "0.2.0",
            "report_artifact": {
                "path": str(paths["output"].resolve()),
                "sha256": "c" * 64,
            },
        }

    monkeypatch.setattr(
        driver,
        "SiliconReleaseValidationConfig",
        config_factory,
    )
    monkeypatch.setattr(
        driver,
        "run_silicon_release_validation",
        validate,
    )

    assert driver.main(_release_arguments(paths)) == 0

    assert captured == {
        f"{name}_path": paths[name]
        for name in (
            "external_oracle_report",
            "equality_3x3_summary",
            "equality_5x5_summary",
            "repository",
            "output",
        )
    }
    streams = capsys.readouterr()
    assert streams.err == ""
    assert json.loads(streams.out) == {
        "status": "passed",
        "validation_passed": True,
        "certification_eligible": True,
        "release_certified": True,
        "release_version": "0.2.0",
        "output": str(paths["output"].resolve()),
        "sha256": "c" * 64,
    }


@pytest.mark.parametrize("kind", ["mpi_oracle", "release"])
def test_validation_exception_publishes_immutable_error_report(
    kind: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    if kind == "mpi_oracle":
        driver = _load_driver("validate_phono3py_si_mpi_oracle.py")
        paths = _mpi_paths(tmp_path)
        arguments = _mpi_arguments(paths)
        requested_input_names = {
            argument_name
            for argument_name, _ in _MPI_CONFIG_FIELDS
            if argument_name != "output"
        }
        config_name = "SiliconMPIOracleValidationConfig"
        validator_name = "run_silicon_mpi_oracle_validation"
        schema_name = "SILICON_MPI_ORACLE_REPORT_SCHEMA"
    else:
        driver = _load_driver("validate_phono3py_si_release.py")
        paths = _release_paths(tmp_path)
        arguments = _release_arguments(paths)
        requested_input_names = set(paths).difference({"output"})
        config_name = "SiliconReleaseValidationConfig"
        validator_name = "run_silicon_release_validation"
        schema_name = "SILICON_RELEASE_REPORT_SCHEMA"

    config = object()
    monkeypatch.setattr(driver, config_name, lambda **kwargs: config)

    def fail(actual_config: object) -> None:
        assert actual_config is config
        raise RuntimeError(f"{kind} validation failed")

    monkeypatch.setattr(driver, validator_name, fail)

    assert driver.main(arguments) == 2

    streams = capsys.readouterr()
    assert streams.out == ""
    summary = json.loads(streams.err)
    published_bytes = paths["output"].read_bytes()
    assert summary["status"] == "error"
    assert summary["error"] == f"{kind} validation failed"
    assert summary["output"] == str(paths["output"].resolve())
    assert summary["sha256"] == hashlib.sha256(published_bytes).hexdigest()

    report = json.loads(published_bytes)
    assert report["schema"] == getattr(driver, schema_name)
    assert report["schema_version"] == 1
    assert report["status"] == "error"
    assert report["validation_passed"] is False
    assert report["certification_eligible"] is False
    assert report["release_certified"] is False
    assert report["error"] == {
        "type": "builtins.RuntimeError",
        "message": f"{kind} validation failed",
    }
    assert report["requested_inputs"] == {
        name: str(path)
        for name, path in paths.items()
        if name in requested_input_names
    }
    if kind == "mpi_oracle":
        assert report["execution"] == {
            "monolithic_negfpy_recomputation": False
        }

    assert driver.main(arguments) == 2
    second_streams = capsys.readouterr()
    publication_failure = json.loads(second_streams.err)
    assert publication_failure["status"] == "error"
    assert (
        publication_failure["error"]
        == f"{kind} validation failed"
    )
    assert publication_failure["report_publication_error"]
    assert paths["output"].read_bytes() == published_bytes
