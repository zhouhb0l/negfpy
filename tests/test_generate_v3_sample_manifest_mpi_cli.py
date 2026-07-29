from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


def _load_driver() -> ModuleType:
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_v3_sample_manifest_mpi.py"
    )
    spec = importlib.util.spec_from_file_location(
        "negfpy_test_generate_v3_sample_manifest_mpi",
        path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load V3 MPI sample-manifest driver.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _required_arguments(tmp_path: Path) -> list[str]:
    return [
        "--reference",
        str(tmp_path / "reference.hdf5"),
        "--fc2-compact",
        str(tmp_path / "fc2-compact.hdf5"),
        "--fc2-full",
        str(tmp_path / "fc2-full.hdf5"),
        "--fc3-compact",
        str(tmp_path / "fc3-compact.hdf5"),
        "--fc3-full",
        str(tmp_path / "fc3-full.hdf5"),
        "--phono3py-yaml",
        str(tmp_path / "phono3py.yaml"),
        "--input-manifest",
        str(tmp_path / "input-manifest.json"),
        "--environment-lock",
        str(tmp_path / "environment.lock"),
        "--repository",
        str(tmp_path / "repository"),
        "--run-directory",
        str(tmp_path / "run"),
        "--output",
        str(tmp_path / "samples.json"),
    ]


def _materialize_required_paths(tmp_path: Path) -> None:
    for name in (
        "reference.hdf5",
        "fc2-compact.hdf5",
        "fc2-full.hdf5",
        "fc3-compact.hdf5",
        "fc3-full.hdf5",
        "phono3py.yaml",
        "input-manifest.json",
        "environment.lock",
    ):
        (tmp_path / name).write_bytes(b"fixture\n")
    (tmp_path / "repository").mkdir()


def test_mpi_manifest_cli_pins_bounded_v2_controls(
    tmp_path: Path,
) -> None:
    driver = _load_driver()
    parsed = driver._parser().parse_args(_required_arguments(tmp_path))
    assert parsed.maximum_promoted_blocks == 512
    assert not parsed.require_mpi
    assert parsed.repository == tmp_path / "repository"

    explicit = driver._parser().parse_args(
        [
            *_required_arguments(tmp_path),
            "--maximum-promoted-blocks",
            "256",
            "--require-mpi",
        ]
    )
    assert explicit.maximum_promoted_blocks == 256
    assert explicit.require_mpi


def test_mpi_manifest_cli_derives_source_identity_and_forwards_campaign(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    driver = _load_driver()
    _materialize_required_paths(tmp_path)
    captured: dict[str, object] = {}
    source_identity = {
        "head_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    monkeypatch.setattr(driver, "_communicator", lambda required: None)
    monkeypatch.setattr(
        driver,
        "read_phono3py_silicon_reference",
        lambda path: ("reference", path),
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_crystal_mapping",
        lambda path, *, purpose: (purpose, path),
    )

    def authenticate(path: Path, *, require_clean: bool):
        captured["authenticated_repository"] = path
        captured["require_clean"] = require_clean
        return source_identity

    monkeypatch.setattr(
        driver,
        "current_v3_sampling_source_identity",
        authenticate,
    )
    monkeypatch.setattr(driver, "_sha256", lambda path: "c" * 64)

    def campaign_factory(**kwargs):
        captured["campaign"] = kwargs
        return object()

    monkeypatch.setattr(driver, "V3SampleMPICampaign", campaign_factory)

    def run(campaign, **kwargs):
        captured["run_campaign"] = campaign
        captured["run"] = kwargs
        return SimpleNamespace(
            manifest=SimpleNamespace(
                path=tmp_path / "samples.json",
                sha256="d" * 64,
                samples=tuple(range(256)),
            ),
            campaign_sha256="e" * 64,
            phase_maximum_execution=SimpleNamespace(
                execution_id="maximum"
            ),
            phase_reservoir_execution=SimpleNamespace(
                execution_id="reservoirs"
            ),
            orchestration_path=tmp_path / "orchestration.json",
            orchestration_file_sha256="f" * 64,
            orchestration_payload_sha256="1" * 64,
        )

    monkeypatch.setattr(driver, "run_v3_sample_manifest_mpi", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_v3_sample_manifest_mpi.py",
            *_required_arguments(tmp_path),
            "--maximum-promoted-blocks",
            "256",
        ],
    )

    assert driver.main() == 0
    assert captured["require_clean"] is True
    assert (
        captured["authenticated_repository"]
        == tmp_path / "repository"
    )
    campaign = captured["campaign"]
    assert campaign["source_revision"] == "a" * 40
    assert campaign["source_tree_sha256"] == "b" * 64
    assert campaign["source_identity"] is source_identity
    assert campaign["require_authenticated_source"] is True
    assert campaign["environment_sha256"] == "c" * 64
    assert campaign["maximum_promoted_blocks"] == 256
    output = json.loads(capsys.readouterr().out)
    assert output == {
        "campaign_sha256": "e" * 64,
        "maximum_phase_execution_id": "maximum",
        "orchestration_file_sha256": "f" * 64,
        "orchestration_path": str(tmp_path / "orchestration.json"),
        "orchestration_payload_sha256": "1" * 64,
        "path": str(tmp_path / "samples.json"),
        "payload_sha256": "d" * 64,
        "promoted_block_count": 256,
        "reservoir_phase_execution_id": "reservoirs",
        "status": "complete",
    }


def test_require_mpi_failure_is_reported_without_starting_work(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    driver = _load_driver()
    monkeypatch.setattr(
        driver,
        "_communicator",
        lambda required: (_ for _ in ()).throw(
            RuntimeError("mpi4py is unavailable")
        ),
    )
    monkeypatch.setattr(
        driver,
        "read_phono3py_silicon_reference",
        lambda path: (_ for _ in ()).throw(
            AssertionError("reference loading started")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_v3_sample_manifest_mpi.py",
            *_required_arguments(tmp_path),
            "--require-mpi",
        ],
    )

    assert driver.main() == 2
    error = json.loads(capsys.readouterr().err)
    assert error == {
        "error": "mpi4py is unavailable",
        "error_type": "RuntimeError",
        "status": "error",
    }


def test_cli_rejects_symlinked_input_before_loading(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    driver = _load_driver()
    _materialize_required_paths(tmp_path)
    reference = tmp_path / "reference.hdf5"
    real_reference = tmp_path / "real-reference.hdf5"
    reference.rename(real_reference)
    reference.symlink_to(real_reference)
    monkeypatch.setattr(driver, "_communicator", lambda required: None)
    monkeypatch.setattr(
        driver,
        "read_phono3py_silicon_reference",
        lambda path: (_ for _ in ()).throw(
            AssertionError("symlinked reference loading started")
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_v3_sample_manifest_mpi.py",
            *_required_arguments(tmp_path),
        ],
    )

    assert driver.main() == 2
    error = json.loads(capsys.readouterr().err)
    assert error["status"] == "error"
    assert error["error_type"] == "V3SampleMPICampaignError"
    assert "symlink" in error["error"]
