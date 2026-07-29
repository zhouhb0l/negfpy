from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import silicon_mpi_oracle_validation as external_validator
from negfpy.anharmonic._reference_mpi_provenance import (
    ReferenceMPIProvenanceError,
    validate_reference_mpi_evidence,
)
from scripts import generate_phono3py_si_reference as serial
from scripts import generate_phono3py_si_reference_mpi as mpi_reference
from scripts import run_phono3py_si_target_q_mpi as target_cli

_THREAD_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_SOURCE_MANIFEST = {
    "schema_version": 1,
    "dataset_id": "synthetic-silicon",
    "versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"},
    "source": {
        "repository": "https://example.invalid/phono3py.git",
        "commit": "a" * 40,
        "files": {},
    },
}
_SOURCE_VERIFICATION = {
    "dataset_id": "synthetic-silicon",
    "resolved_commit": "a" * 40,
    "files": {},
}
_ENVIRONMENT_IDENTITY = {
    "runtime_versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"},
    "thread_environment": _THREAD_ENVIRONMENT,
}


@pytest.fixture(autouse=True)
def _pin_all_thread_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name, value in _THREAD_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)


def _json_payload(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _force_stage_payload(name: str) -> bytes:
    if name == "input-manifest.normalized.json":
        return _json_payload(_SOURCE_MANIFEST)
    if name == "source-verification.json":
        return _json_payload(_SOURCE_VERIFICATION)
    return f"synthetic {name}\n".encode("ascii")


def _force_stage_identity(name: str) -> dict[str, object]:
    payload = _force_stage_payload(name)
    return {
        "name": name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _worker_binding(
    *,
    rank: int,
    logical_cpu: int,
    hostname: str = "worker",
) -> dict[str, object]:
    return {
        "rank": rank,
        "hostname": hostname,
        "cpu_affinity": [logical_cpu, logical_cpu + 128],
        "cpu_affinity_topology": [
            {
                "logical_cpu": logical_cpu,
                "package_id": 0,
                "core_id": logical_cpu,
            },
            {
                "logical_cpu": logical_cpu + 128,
                "package_id": 0,
                "core_id": logical_cpu,
            },
        ],
        "physical_core": {"package_id": 0, "core_id": logical_cpu},
    }


def _identity() -> mpi_reference.CampaignIdentity:
    repository = Path(__file__).resolve().parents[1]
    return mpi_reference.CampaignIdentity.from_payload(
        {
            "schema_version": (
                mpi_reference.MPI_REFERENCE_SCHEMA_VERSION
            ),
            "serial_generator": {
                "script_sha256": mpi_reference._sha256_file(
                    repository
                    / "scripts"
                    / "generate_phono3py_si_reference.py"
                ),
                "git_commit": "a" * 40,
                "tracked_worktree_status": "",
                "untracked_files_omitted": False,
            },
            "mpi_driver": {
                "script_sha256": mpi_reference._sha256_file(
                    repository
                    / "scripts"
                    / "generate_phono3py_si_reference_mpi.py"
                ),
                "git_commit": "a" * 40,
                "tracked_worktree_status": "",
                "untracked_files_omitted": False,
            },
            "algorithm": (
                "one_complete_phono3py_target_q_per_worker_"
                "canonical_rank0_merge"
            ),
            "floating_point_mpi_reductions": False,
            "source_manifest": _SOURCE_MANIFEST,
            "source_verification": _SOURCE_VERIFICATION,
            "force_stage": {
                name: _force_stage_identity(name)
                for name in sorted(serial._FORCE_STAGE_FILENAMES)
            },
            "reference_settings": _config().reference_settings(),
            "runtime_environment": _ENVIRONMENT_IDENTITY,
            "thread_environment": _THREAD_ENVIRONMENT,
            "task_grid_points_grg": [0, 1],
            "final_output_name": "output",
        }
    )


def _config(points=(0, 1)) -> serial.GeneratorConfig:
    return serial.GeneratorConfig(
        mesh=(2, 1, 1),
        grid_points=points,
        band_indices=(0, 1),
        temperatures=(0.0, 300.0),
        sigma=0.1,
    )


class _ForceRuntime:
    phono3py_version = "4.4.0"
    phonopy_version = "4.4.0"

    def verify_versions(self, expected):
        return {
            "phono3py": self.phono3py_version,
            "phonopy": self.phonopy_version,
        }

    def generate_force_constants(self, *, output_dir, source_files, config):
        for name in serial.FORCE_CONSTANT_FILENAMES:
            (output_dir / name).write_bytes(f"{name}\n".encode())
        return {"implementation": "bounded_fake_force_runtime"}


def _source_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source = tmp_path / "source"
    source.mkdir()
    files = {}
    for role, name in (
        ("phono3py_yaml", "phono3py_disp.yaml"),
        ("forces_fc3", "FORCES_FC3"),
        ("readme", "README.md"),
    ):
        path = source / name
        path.write_text(f"{role}\n", encoding="utf-8")
        files[role] = {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": 1,
        "dataset_id": "synthetic-silicon",
        "versions": {"phono3py": "4.4.0", "phonopy": "4.4.0"},
        "source": {
            "repository": "https://example.invalid/phono3py.git",
            "commit": "a" * 40,
            "files": files,
        },
    }
    path = tmp_path / "input-manifest.json"
    path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return source, path


def _runtime_metadata(
    point: int,
    bzg: int,
    config: serial.GeneratorConfig,
) -> dict:
    return {
        "implementation": "synthetic_phono3py_4_4_0",
        "public_apis": ["synthetic"],
        "private_apis": [],
        "phono3py_version": "4.4.0",
        "phonopy_version": "4.4.0",
        "mesh": list(config.mesh),
        "grid_points_grg": [point],
        "grid_points_bzg": [bzg],
        "band_indices": list(config.band_indices),
        "temperatures_K": list(config.temperatures),
        "sigma_THz": config.sigma,
        "sigma_cutoff": config.sigma_cutoff,
        "make_r0_average": True,
        "symmetrize_fc3q": False,
        "gamma_shape": [1, len(config.temperatures), 1, 2],
        "phonon_frequency_shape": [2, 2],
        "phonon_eigenvector_shape": [2, 2, 2],
        "number_of_mesh_points": 2,
        "number_of_bz_grid_points": 2,
        "effective_band_indices": [0, 1],
        "targets": [
            {
                "grid_point_grg": point,
                "grid_point_bzg": bzg,
                "triplets": 2,
                "triplet_weight_sum": 2,
                "interaction_strength_shape": [2, 2, 2, 2],
                "integration_weights_shape": [2, 2, 2, 2, 2],
            }
        ],
    }


def _dataset(group, name, values, **attrs):
    dataset = group.create_dataset(name, data=values)
    for key, value in attrs.items():
        dataset.attrs[key] = value
    return dataset


def _write_one_q(
    path: Path,
    *,
    identity: mpi_reference.CampaignIdentity,
    config: serial.GeneratorConfig,
    point: int,
) -> None:
    bzg = point
    runtime = _runtime_metadata(point, bzg, config)
    with h5py.File(path, "w") as h5:
        h5.attrs.update(
            {
                "schema_name": "negfpy_phono3py_si_reference",
                "schema_version": 1,
                "phono3py_version": "4.4.0",
                "phonopy_version": "4.4.0",
                "make_r0_average": True,
                "symmetrize_fc3q": False,
                "mpi_reference_task_schema": (
                    "negfpy_phono3py_si_reference_mpi_task"
                ),
                "mpi_reference_task_schema_version": 1,
                "mpi_reference_campaign_sha256": identity.sha256,
                "mpi_reference_grid_point_grg": point,
                "mpi_reference_runtime_json": json.dumps(
                    runtime,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
        crystal = h5.create_group("crystal")
        _dataset(crystal, "common", np.arange(6).reshape(2, 3))
        grid = h5.create_group("grid")
        _dataset(grid, "grg2bzg", [0, 1])
        fc2 = h5.create_group("fc2_random_q")
        _dataset(fc2, "qpoints", [[0.125, 0.0, 0.0]])
        _dataset(
            fc2,
            "dynamical_matrices",
            np.eye(2, dtype=np.complex128)[None],
        )

        settings = h5.create_group("settings")
        settings.attrs["sigma_cutoff"] = "None"
        _dataset(settings, "mesh", config.mesh)
        _dataset(settings, "grid_points_grg", [point])
        _dataset(settings, "grid_points_bzg", [bzg])
        _dataset(settings, "band_indices", config.band_indices)
        _dataset(settings, "temperatures", config.temperatures, units="K")
        _dataset(settings, "sigmas", [config.sigma], units="cyclic_THz")

        phonons = h5.create_group("phonons")
        _dataset(
            phonons,
            "frequencies",
            [[1.0, 2.0], [1.5, 2.5]],
            units="cyclic_THz",
        )
        _dataset(
            phonons,
            "eigenvectors",
            np.repeat(np.eye(2, dtype=np.complex128)[None], 2, axis=0),
        )
        _dataset(phonons, "bz_grid_addresses", [[0, 0, 0], [1, 0, 0]])
        _dataset(
            phonons,
            "target_band_frequencies",
            [[1.0 + point / 2, 2.0 + point / 2]],
            units="cyclic_THz",
        )

        gamma = h5.create_group("gamma")
        gamma.attrs["frequency_points_at_bands"] = True
        _dataset(
            gamma,
            "values",
            np.full((1, 2, 1, 2), 0.01 * (point + 1)),
            units="cyclic_THz_HWHM",
        )

        targets = h5.create_group("target_q")
        target = targets.create_group(f"grg_{point:08d}")
        target.attrs.update(
            {
                "target_ordinal": 0,
                "grid_point_grg": point,
                "grid_point_bzg": bzg,
                "number_of_full_mesh_triplets": 2,
                "sum_of_triplet_weights": 2,
                "Nq": 2,
            }
        )
        _dataset(target, "triplets_bzg", [[bzg, 0, point], [bzg, 1, 1 - point]])
        _dataset(target, "triplet_addresses", np.zeros((2, 3, 3), dtype=int))
        _dataset(target, "triplet_weights", [1, 1])
        _dataset(target, "triplets_map_grg", [0, 1])
        _dataset(target, "ir_map_grg", [0, 1])
        _dataset(
            target,
            "interaction_strength",
            np.full((2, 2, 2, 2), point + 1.0),
        )
        _dataset(
            target,
            "integration_weights",
            np.full((2, 2, 2, 2, 2), 0.5 + point),
        )
        _dataset(
            target,
            "derived_phase_space_channel_sums",
            np.full((2, 2), 0.25 + point),
        )


def _campaign(tmp_path: Path) -> mpi_reference.PreparedCampaign:
    force = tmp_path / "force"
    force.mkdir()
    for name in serial._FORCE_STAGE_FILENAMES:
        (force / name).write_bytes(_force_stage_payload(name))
    identity = _identity()
    run_dir = tmp_path / "run"
    mpi_reference._initialize_run_directory(run_dir, identity=identity)
    return mpi_reference.PreparedCampaign(
        identity=identity,
        config=_config(),
        source_files={},
        source_verification=_SOURCE_VERIFICATION,
        environment_identity=_ENVIRONMENT_IDENTITY,
        force_stage_dir=force,
        run_dir=run_dir,
        output_dir=tmp_path / "output",
    )


def _publish_task(
    campaign: mpi_reference.PreparedCampaign,
    point: int,
    *,
    execution_id: str | None = None,
) -> dict:
    hdf5_path, record_path, _ = mpi_reference._task_paths(
        campaign.run_dir,
        point,
    )
    _write_one_q(
        hdf5_path,
        identity=campaign.identity,
        config=campaign.config,
        point=point,
    )
    origin_identity = None
    origin = None
    if execution_id is not None:
        origin_path, origin = mpi_reference._publish_origin_assignment(
            campaign,
            point,
            execution_id=execution_id,
            rank=point + 1,
            communicator_size=3,
            worker_binding=_worker_binding(
                rank=point + 1,
                logical_cpu=7 + point,
                hostname=mpi_reference.socket.gethostname(),
            ),
        )
        origin_identity = mpi_reference._origin_assignment_identity(
            campaign.run_dir,
            origin_path,
        )
        with h5py.File(hdf5_path, "r+") as h5:
            h5.attrs["mpi_reference_origin_assignment_json"] = (
                json.dumps(
                    origin_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            h5.flush()
    validation = mpi_reference._validate_task_hdf5(
        hdf5_path,
        campaign_sha256=campaign.identity.sha256,
        config=campaign.config,
        grid_point=point,
    )
    record = mpi_reference._task_record(
        campaign_sha256=campaign.identity.sha256,
        grid_point=point,
        hdf5_path=hdf5_path,
        validation=validation,
        rank=point + 1,
        hostname=(
            "worker" if origin is None else str(origin["hostname"])
        ),
        started_utc=(
            "2026-01-01T00:00:00+00:00"
            if origin is None
            else str(origin["assigned_at_utc"])
        ),
        completed_utc="2026-01-01T00:00:01+00:00",
        wall_seconds=1.0,
        disposition="computed",
        origin_assignment=origin_identity,
    )
    mpi_reference._write_json_exclusive(record_path, record)
    return record


def _finalized_mpi_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[mpi_reference.PreparedCampaign, Path, dict[str, object]]:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    campaign = _campaign(tmp_path)
    task_records = {
        point: _publish_task(
            campaign,
            point,
            execution_id="run-001",
        )
        for point in campaign.config.grid_points
    }
    assignments = [
        {
            "grid_point_grg": point,
            "rank": point + 1,
            "status": "computed",
            "wall_seconds": 1.0,
            "worker_binding": _worker_binding(
                rank=point + 1,
                logical_cpu=7 + point,
                hostname=mpi_reference.socket.gethostname(),
            ),
            "artifact": dict(task_records[point]["artifact"]),
            "origin_assignment": dict(
                task_records[point]["origin_assignment"]
            ),
        }
        for point in campaign.config.grid_points
    ]
    scratch_cleanup_path, _ = (
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id="run-001",
        )
    )
    execution = mpi_reference._execution_record(
        campaign=campaign,
        execution_id="run-001",
        size=3,
        started_utc="2026-01-01T00:00:00+00:00",
        started=0.0,
        assignments=assignments,
        complete=True,
        scratch_cleanup=mpi_reference._file_identity(
            scratch_cleanup_path
        ),
    )
    execution_path = mpi_reference._execution_path(
        campaign.run_dir, "run-001"
    )
    mpi_reference._write_json_exclusive(execution_path, execution)
    output = mpi_reference._finalize_reference_bundle(
        campaign,
        execution_records=[execution],
    )
    document = json.loads(
        (output / mpi_reference.MPI_ORCHESTRATION_FILENAME).read_text(
            encoding="utf-8"
        )
    )
    return campaign, output, document


def _validation_controls() -> dict[str, object]:
    return {
        "expected_mesh": (2, 1, 1),
        "expected_grid_points": (0, 1),
        "expected_band_indices": (0, 1),
        "expected_n_qpoints": 2,
        "expected_n_bands": 2,
        "require_mpi": True,
    }


def _publish_orchestration(
    output: Path,
    document: dict[str, object],
) -> None:
    path = output / mpi_reference.MPI_ORCHESTRATION_FILENAME
    path.write_bytes(_json_payload(document))
    stage_path = output / "reference-stage.json"
    stage = json.loads(stage_path.read_text(encoding="utf-8"))
    mpi_runtime = stage["runtime"]["mpi_orchestration"]
    mpi_runtime["orchestration_record_sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    mpi_runtime["evidence_archive_sha256"] = document["evidence_archive"][
        "sha256"
    ]
    stage_path.write_bytes(_json_payload(stage))


def _archive_payloads(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, mode="r:") as archive:
        return {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
        }


def _replace_archive(
    output: Path,
    document: dict[str, object],
    payloads: dict[str, bytes],
) -> None:
    archive_path = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    temporary = archive_path.with_suffix(".tmp.tar")
    with tarfile.open(temporary, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name, payload in payloads.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            info.mode = 0o444
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(payload))
    os.replace(temporary, archive_path)
    document["evidence_archive"] = {
        "name": archive_path.name,
        "size_bytes": archive_path.stat().st_size,
        "sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
    }
    _publish_orchestration(output, document)


def _member_identity(member: str, payload: bytes) -> dict[str, object]:
    return {
        "archive_member": member,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _mutate_hdf5_bytes(
    payload: bytes,
    mutator,
) -> bytes:
    descriptor, name = tempfile.mkstemp(suffix=".hdf5")
    os.close(descriptor)
    path = Path(name)
    try:
        path.write_bytes(payload)
        with h5py.File(path, "r+") as h5:
            mutator(h5)
            h5.flush()
        return path.read_bytes()
    finally:
        path.unlink(missing_ok=True)


def test_campaign_identity_and_resume_manifest_are_exact(tmp_path) -> None:
    identity = _identity()
    run_dir = tmp_path / "run"

    mpi_reference._initialize_run_directory(run_dir, identity=identity)
    mpi_reference._initialize_run_directory(run_dir, identity=identity)

    manifest = json.loads(
        (run_dir / mpi_reference.MPI_MANIFEST_FILENAME).read_text()
    )
    assert manifest["campaign_sha256"] == identity.sha256
    changed_payload = dict(identity.payload)
    changed_payload["changed"] = True
    changed = mpi_reference.CampaignIdentity.from_payload(changed_payload)
    with pytest.raises(mpi_reference.ReferenceMPIError, match="different campaign"):
        mpi_reference._initialize_run_directory(run_dir, identity=changed)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "",
        ".",
        "..",
        "nested/output",
        r"nested\output",
        "control\nname",
        "control\x7fname",
        "control\x85name",
    ),
)
def test_campaign_identity_rejects_unsafe_final_output_name(
    unsafe_name,
) -> None:
    payload = dict(_identity().payload)
    payload["final_output_name"] = unsafe_name

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="single safe basename",
    ):
        mpi_reference.CampaignIdentity.from_payload(payload)


def test_cleanup_rejects_output_name_mismatched_to_campaign_identity(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    mismatched = mpi_reference.PreparedCampaign(
        identity=campaign.identity,
        config=campaign.config,
        source_files=campaign.source_files,
        source_verification=campaign.source_verification,
        environment_identity=campaign.environment_identity,
        force_stage_dir=campaign.force_stage_dir,
        run_dir=campaign.run_dir,
        output_dir=tmp_path / "different-output",
    )

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="does not match authenticated",
    ):
        mpi_reference._cleanup_interrupted_scratch(
            mismatched,
            execution_id="mismatched-output",
        )


@pytest.mark.parametrize("unsafe_name", ("control\noutput", r"nested\output"))
def test_prepare_campaign_rejects_unsafe_output_basename(
    tmp_path,
    unsafe_name,
) -> None:
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="output_dir.name must be a single safe basename",
    ):
        mpi_reference.prepare_campaign(
            manifest_path=tmp_path / "missing-manifest.json",
            source_dir=tmp_path / "missing-source",
            force_stage_dir=tmp_path / "missing-force",
            run_dir=tmp_path / "run",
            output_dir=tmp_path / unsafe_name,
            config=_config(),
            require_clean_source=False,
        )


@pytest.mark.parametrize("path_kind", ("run_dir", "output_dir"))
def test_campaign_paths_reject_symlinked_parent_components(
    tmp_path,
    path_kind,
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    run_dir = (
        linked_parent / "run"
        if path_kind == "run_dir"
        else tmp_path / "run"
    )
    output_dir = (
        linked_parent / "output"
        if path_kind == "output_dir"
        else tmp_path / "output"
    )

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="path components must not be symbolic links",
    ):
        mpi_reference.prepare_campaign(
            manifest_path=tmp_path / "missing-manifest.json",
            source_dir=tmp_path / "missing-source",
            force_stage_dir=tmp_path / "missing-force",
            run_dir=run_dir,
            output_dir=output_dir,
            config=_config(),
            require_clean_source=False,
        )


def test_campaign_execution_lock_rejects_overlap_and_restarts(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)

    first = mpi_reference._CampaignExecutionLock.acquire(campaign.run_dir)
    try:
        with pytest.raises(
            mpi_reference.ReferenceMPIError,
            match="already active",
        ):
            mpi_reference._CampaignExecutionLock.acquire(campaign.run_dir)
    finally:
        first.close()

    with mpi_reference._CampaignExecutionLock.acquire(
        campaign.run_dir
    ) as restarted:
        restarted.verify()


@pytest.mark.parametrize("link_kind", ("symbolic", "hard"))
def test_campaign_execution_lock_rejects_linked_lock_file(
    tmp_path,
    link_kind,
) -> None:
    campaign = _campaign(tmp_path)
    lock_path = (
        campaign.run_dir
        / mpi_reference._CAMPAIGN_EXECUTION_LOCK_FILENAME
    )
    lock_path.unlink()
    detached = tmp_path / "detached-lock"
    detached.write_bytes(b"")
    if link_kind == "symbolic":
        lock_path.symlink_to(detached)
    else:
        os.link(detached, lock_path)

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="regular, single-link",
    ):
        mpi_reference._CampaignExecutionLock.acquire(campaign.run_dir)


def test_prepare_campaign_authenticates_exact_serial_force_stage(
    tmp_path,
    monkeypatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    source, manifest = _source_manifest(tmp_path)
    force = tmp_path / "force"
    runtime = _ForceRuntime()
    serial.run_generator(
        manifest_path=manifest,
        source_dir=source,
        output_dir=force,
        config=serial.GeneratorConfig(force_constants_only=True),
        runtime=runtime,
        commit_resolver=lambda _: "a" * 40,
    )

    campaign = mpi_reference.prepare_campaign(
        manifest_path=manifest,
        source_dir=source,
        force_stage_dir=force,
        run_dir=tmp_path / "run",
        output_dir=tmp_path / "output",
        config=_config(),
        runtime=runtime,
        commit_resolver=lambda _: "a" * 40,
        require_clean_source=False,
    )

    assert campaign.identity.sha256 == json.loads(
        (campaign.run_dir / mpi_reference.MPI_MANIFEST_FILENAME).read_text()
    )["campaign_sha256"]
    assert campaign.identity.payload["floating_point_mpi_reductions"] is False
    assert campaign.identity.payload["final_output_name"] == "output"
    assert (
        campaign.identity.payload["thread_environment"]
        == _THREAD_ENVIRONMENT
    )
    (force / "fc3.compact.hdf5").write_bytes(b"mutated\n")
    with pytest.raises(ValueError, match="SHA256"):
        mpi_reference.prepare_campaign(
            manifest_path=manifest,
            source_dir=source,
            force_stage_dir=force,
            run_dir=tmp_path / "other-run",
            output_dir=tmp_path / "other-output",
            config=_config(),
            runtime=runtime,
            commit_resolver=lambda _: "a" * 40,
            require_clean_source=False,
        )


def test_task_record_detects_mutation_and_orphan_is_adopted(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    campaign = _campaign(tmp_path)
    record = _publish_task(campaign, 0)
    hdf5_path, record_path, _ = mpi_reference._task_paths(campaign.run_dir, 0)

    assert (
        mpi_reference._validate_task_record(
            record_path,
            hdf5_path,
            campaign_sha256=campaign.identity.sha256,
            config=campaign.config,
            grid_point=0,
        )
        == record
    )
    record_path.unlink()
    adopted = mpi_reference._adopt_or_validate_task(campaign, 0)
    assert adopted["disposition"] == "adopted_after_interrupted_record_publication"

    with h5py.File(hdf5_path, "r+") as h5:
        h5["gamma/values"][0, 0, 0, 0] += 1
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="changed after publication",
    ):
        mpi_reference._adopt_or_validate_task(campaign, 0)


def test_origin_bound_hdf_recovers_after_hard_record_interruption(
    tmp_path,
    monkeypatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    campaign = _campaign(tmp_path)
    record = _publish_task(
        campaign,
        0,
        execution_id="killed-two-rank",
    )
    _, record_path, _ = mpi_reference._task_paths(campaign.run_dir, 0)
    record_path.unlink()
    mpi_reference._publish_origin_assignment(
        campaign,
        0,
        execution_id="incomplete-retry",
        rank=1,
        communicator_size=4,
        worker_binding=_worker_binding(
            rank=1,
            logical_cpu=11,
            hostname=mpi_reference.socket.gethostname(),
        ),
    )

    recovered = mpi_reference._adopt_or_validate_task(campaign, 0)

    assert recovered is not None
    assert recovered["disposition"] == (
        "computed_after_interrupted_record_publication"
    )
    assert recovered["origin_assignment"] == record["origin_assignment"]


def test_resume_cleans_only_validated_stale_worker_scratch(
    tmp_path,
    monkeypatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    campaign = _campaign(tmp_path)
    stale = (
        campaign.run_dir
        / "scratch"
        / ".target-grg-00000000-rank1-killed"
    )
    stale.mkdir()
    source = campaign.force_stage_dir / "fc2.compact.hdf5"
    os.link(source, stale / source.name)
    (stale / "partial.hdf5").write_bytes(b"partial")

    cleanup_path, cleanup = (
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id="restart-four-rank",
        )
    )

    assert not any((campaign.run_dir / "scratch").iterdir())
    assert source.is_file()
    assert cleanup_path.is_file()
    assert cleanup["removed"][0]["directory"] == stale.name
    assert {item["relative_path"] for item in cleanup["removed"][0]["files"]} == {
        "fc2.compact.hdf5",
        "partial.hdf5",
    }


def test_resume_cleans_and_records_exact_atomic_publication_residue(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    name = (
        ".target-grg-00000000.json."
        f"{'a' * 32}.tmp"
    )
    residue = campaign.run_dir / "tasks" / name
    residue.write_bytes(b"interrupted publication")

    _, cleanup = mpi_reference._cleanup_interrupted_scratch(
        campaign,
        execution_id="resume-private-temp",
    )

    assert not residue.exists()
    private = next(
        item for item in cleanup["removed"] if item["directory"] == "tasks"
    )
    assert private["files"] == [
        {
            "relative_path": name,
            "size_bytes": len(b"interrupted publication"),
            "sha256": hashlib.sha256(
                b"interrupted publication"
            ).hexdigest(),
        }
    ]


def test_resume_rejects_suspicious_atomic_publication_near_match(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    residue = (
        campaign.run_dir
        / "tasks"
        / ".target-grg-00000000.json.short.tmp"
    )
    residue.write_bytes(b"ambiguous")

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="Suspicious atomic-publication residue",
    ):
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id="reject-private-near-match",
        )
    assert residue.is_file()


@pytest.mark.parametrize("link_kind", ("symbolic", "hard"))
def test_resume_rejects_linked_atomic_publication_residue(
    tmp_path,
    link_kind,
) -> None:
    campaign = _campaign(tmp_path)
    detached = tmp_path / "detached-private-temp"
    detached.write_bytes(b"detached")
    residue = (
        campaign.run_dir
        / "executions"
        / f".run-001.json.{'b' * 32}.tmp"
    )
    if link_kind == "symbolic":
        residue.symlink_to(detached)
    else:
        os.link(detached, residue)

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="regular, single-link",
    ):
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id=f"reject-{link_kind}-private-temp",
        )
    assert os.path.lexists(residue)


def test_resume_cleans_canonical_abandoned_final_staging(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    staging = tmp_path / f".output.{'c' * 32}.tmp"
    nested = staging / "nested"
    nested.mkdir(parents=True)
    partial = nested / "partial.json"
    partial.write_bytes(b"partial final bundle")

    cleanup_path, cleanup = mpi_reference._cleanup_interrupted_scratch(
        campaign,
        execution_id="recover-final-staging",
    )

    assert not staging.exists()
    removed = next(
        item
        for item in cleanup["removed"]
        if item["directory"]
        == f"output-parent/{staging.name}"
    )
    assert removed["files"] == [
        {
            "relative_path": "nested/partial.json",
            "size_bytes": len(b"partial final bundle"),
            "sha256": hashlib.sha256(
                b"partial final bundle"
            ).hexdigest(),
        }
    ]
    mpi_reference._validate_scratch_cleanup_record(
        campaign,
        mpi_reference._file_identity(cleanup_path),
        execution_id="recover-final-staging",
    )


def test_resume_rejects_suspicious_final_staging_near_match(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    suspicious = tmp_path / ".output.short.tmp"
    suspicious.mkdir()

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="Suspicious abandoned final-staging near-match",
    ):
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id="reject-final-staging-near-match",
        )
    assert suspicious.is_dir()


@pytest.mark.parametrize("alias_kind", ("symbolic", "hard"))
def test_resume_rejects_unsafe_final_staging_alias(
    tmp_path,
    alias_kind,
) -> None:
    campaign = _campaign(tmp_path)
    staging = tmp_path / f".output.{'d' * 32}.tmp"
    detached = tmp_path / "detached-final-staging"
    if alias_kind == "symbolic":
        detached.mkdir()
        staging.symlink_to(detached, target_is_directory=True)
        match = "non-symlink directory"
    else:
        staging.mkdir()
        detached.write_bytes(b"hard-linked staging content")
        os.link(detached, staging / "partial")
        match = "Unsafe hard-linked owned cleanup entry"

    with pytest.raises(mpi_reference.ReferenceMPIError, match=match):
        mpi_reference._cleanup_interrupted_scratch(
            campaign,
            execution_id=f"reject-{alias_kind}-final-staging",
        )
    assert os.path.lexists(staging)
    assert detached.exists()


@pytest.mark.parametrize("link_kind", ("symbolic", "hard"))
def test_evidence_identity_rejects_linked_source(
    tmp_path,
    link_kind,
) -> None:
    detached = tmp_path / "detached-evidence"
    detached.write_bytes(b"evidence")
    source = tmp_path / "evidence.json"
    if link_kind == "symbolic":
        source.symlink_to(detached)
    else:
        os.link(detached, source)

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="symbolic links|single-link non-symlink",
    ):
        mpi_reference._archive_member_identity(
            source,
            "executions/evidence.json",
        )


def test_canonical_merge_matches_one_q_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    campaign = _campaign(tmp_path)
    records = [_publish_task(campaign, point) for point in (0, 1)]
    destination = tmp_path / "merged.hdf5"

    summary = mpi_reference.merge_task_references(campaign, destination)

    assert summary["grid_points_grg"] == [0, 1]
    assert summary["grid_points_bzg"] == [0, 1]
    assert summary["gamma_shape"] == [1, 2, 2, 2]
    assert {
        item["scientific"]["common_reference_sha256"] for item in records
    } == {summary["common_reference_sha256"]}
    with h5py.File(destination, "r") as h5:
        assert h5["settings/grid_points_grg"][...].tolist() == [0, 1]
        assert h5["phonons/target_band_frequencies"][...].tolist() == [
            [1.0, 2.0],
            [1.5, 2.5],
        ]
        assert h5["gamma/values"].shape == (1, 2, 2, 2)
        assert sorted(h5["target_q"]) == [
            "grg_00000000",
            "grg_00000001",
        ]
        assert h5["target_q/grg_00000000"].attrs["target_ordinal"] == 0
        assert h5["target_q/grg_00000001"].attrs["target_ordinal"] == 1
        assert not any(
            name.startswith("mpi_reference_") for name in h5.attrs
        )


def test_task_filename_and_record_hashes_are_canonical(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "1")
    monkeypatch.setenv("MKL_NUM_THREADS", "1")
    monkeypatch.setenv("NUMEXPR_NUM_THREADS", "1")
    campaign = _campaign(tmp_path)
    _publish_task(campaign, 1)
    hdf5_path, record_path, _ = mpi_reference._task_paths(campaign.run_dir, 1)
    record = json.loads(record_path.read_text())

    assert hdf5_path.name == "target-grg-00000001.hdf5"
    assert record_path.name == "target-grg-00000001.json"
    assert record["artifact"]["sha256"] == hashlib.sha256(
        hdf5_path.read_bytes()
    ).hexdigest()
    assert record["grid_point_grg"] == 1


def test_terminal_task_failure_is_archived_without_retuning(
    tmp_path,
    monkeypatch,
) -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    campaign = _campaign(tmp_path)
    for name in ("fc2.compact.hdf5", "fc3.compact.hdf5"):
        (campaign.force_stage_dir / name).write_bytes(b"force constants\n")

    class FailingRuntime:
        def verify_versions(self, expected):
            return dict(expected)

        def generate_reference(self, **kwargs):
            raise RuntimeError("declared scientific failure")

    with pytest.raises(RuntimeError, match="declared scientific failure"):
        mpi_reference._generate_one_task(
            campaign,
            0,
            rank=1,
            runtime_factory=FailingRuntime,
        )
    _, _, failure_path = mpi_reference._task_paths(campaign.run_dir, 0)
    failure = json.loads(failure_path.read_text())
    assert failure["status"] == "failed"
    assert failure["campaign_sha256"] == campaign.identity.sha256
    assert failure["grid_point_grg"] == 0
    assert failure["exception_type"] == "RuntimeError"
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="archived terminal failure",
    ):
        mpi_reference._adopt_or_validate_task(campaign, 0)


def test_execution_id_rejects_paths(tmp_path) -> None:
    for value in (
        "",
        ".",
        "../run",
        "a/b",
        r"a\b",
        "bad space",
        "a" * 65,
    ):
        with pytest.raises(ValueError):
            mpi_reference._execution_path(tmp_path, value)
    assert mpi_reference._execution_path(tmp_path, "run-001") == (
        tmp_path / "executions" / "run-001.json"
    )


def test_execution_validation_rejects_unknown_top_level_and_thread_keys(
    tmp_path,
    monkeypatch,
) -> None:
    campaign, _, _ = _finalized_mpi_bundle(tmp_path, monkeypatch)
    execution = json.loads(
        mpi_reference._execution_path(
            campaign.run_dir,
            "run-001",
        ).read_text(encoding="utf-8")
    )
    assert (
        mpi_reference._validate_execution_record(campaign, execution)
        == execution
    )

    extra_top_level = json.loads(json.dumps(execution))
    extra_top_level["unreviewed_extension"] = True
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="ambiguous top-level schema",
    ):
        mpi_reference._validate_execution_record(
            campaign,
            extra_top_level,
        )

    extra_thread = json.loads(json.dumps(execution))
    extra_thread["thread_environment"]["UNREVIEWED_THREADS"] = "1"
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="exactly the six pinned thread controls",
    ):
        mpi_reference._validate_execution_record(campaign, extra_thread)


def test_execution_computed_status_rejects_recovered_task_lineage(
    tmp_path,
    monkeypatch,
) -> None:
    campaign, _, _ = _finalized_mpi_bundle(tmp_path, monkeypatch)
    execution = mpi_reference._read_json_object(
        mpi_reference._execution_path(campaign.run_dir, "run-001"),
        name="execution",
    )
    _, record_path, _ = mpi_reference._task_paths(campaign.run_dir, 0)
    task_record = mpi_reference._read_json_object(
        record_path,
        name="task record",
    )
    task_record["disposition"] = (
        "computed_after_interrupted_record_publication"
    )
    record_path.write_bytes(mpi_reference._pretty_json_bytes(task_record))

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="computed lineage differs",
    ):
        mpi_reference._validate_execution_record(campaign, execution)


def test_recovered_computed_tasks_resume_into_release_valid_execution(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    recovered_records = {}
    for point in campaign.config.grid_points:
        _publish_task(
            campaign,
            point,
            execution_id="interrupted-run",
        )
        _, record_path, _ = mpi_reference._task_paths(
            campaign.run_dir,
            point,
        )
        record_path.unlink()
        recovered = mpi_reference._adopt_or_validate_task(campaign, point)
        assert recovered is not None
        assert recovered["disposition"] == (
            "computed_after_interrupted_record_publication"
        )
        recovered_records[point] = recovered

    staging = tmp_path / f".output.{'e' * 32}.tmp"
    staging.mkdir()
    (staging / "partial.json").write_bytes(b"interrupted finalization")
    cleanup_path, cleanup = mpi_reference._cleanup_interrupted_scratch(
        campaign,
        execution_id="recovery-run",
    )
    assert not staging.exists()
    assert any(
        item["directory"] == f"output-parent/{staging.name}"
        for item in cleanup["removed"]
    )
    assignments = [
        {
            "grid_point_grg": point,
            "rank": None,
            "status": "resumed",
            "wall_seconds": 0.0,
            "origin_assignment": dict(
                recovered_records[point]["origin_assignment"]
            ),
        }
        for point in campaign.config.grid_points
    ]
    execution = mpi_reference._execution_record(
        campaign=campaign,
        execution_id="recovery-run",
        size=3,
        started_utc="2026-01-01T00:00:00+00:00",
        started=0.0,
        assignments=assignments,
        complete=True,
        scratch_cleanup=mpi_reference._file_identity(cleanup_path),
    )
    execution_path = mpi_reference._execution_path(
        campaign.run_dir,
        "recovery-run",
    )
    mpi_reference._write_json_exclusive(execution_path, execution)

    assert (
        mpi_reference._validate_execution_record(campaign, execution)
        == execution
    )
    output = mpi_reference._finalize_reference_bundle(
        campaign,
        execution_records=[execution],
    )
    orchestration = mpi_reference._read_json_object(
        output / mpi_reference.MPI_ORCHESTRATION_FILENAME,
        name="orchestration",
    )
    validated = validate_reference_mpi_evidence(
        output,
        orchestration,
        **_validation_controls(),
    )
    assert validated["task_count"] == 2
    assert validated["execution_count"] == 1


def test_execution_assignment_wall_seconds_must_be_finite_nonnegative(
    tmp_path,
    monkeypatch,
) -> None:
    campaign, _, _ = _finalized_mpi_bundle(tmp_path, monkeypatch)
    execution = mpi_reference._read_json_object(
        mpi_reference._execution_path(campaign.run_dir, "run-001"),
        name="execution",
    )

    for invalid in (None, True, -1.0, float("nan"), float("inf")):
        changed = json.loads(json.dumps(execution))
        changed["assignments"][0]["wall_seconds"] = invalid
        with pytest.raises(
            mpi_reference.ReferenceMPIError,
            match="wall_seconds must be finite and nonnegative",
        ):
            mpi_reference._validate_execution_record(campaign, changed)


def test_generator_json_helpers_reject_nonfinite_values(tmp_path) -> None:
    with pytest.raises(ValueError, match="Out of range float values"):
        mpi_reference._canonical_json_bytes({"value": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        mpi_reference._pretty_json_bytes({"value": float("inf")})

    invalid = tmp_path / "nonfinite.json"
    invalid.write_text('{"value": NaN}\n', encoding="utf-8")
    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="contains nonfinite JSON value",
    ):
        mpi_reference._read_json_object(invalid, name="nonfinite fixture")


def test_generator_release_evidence_rejects_legacy_adopted_task(
    tmp_path,
) -> None:
    campaign = _campaign(tmp_path)
    _publish_task(campaign, 0)
    _, record_path, _ = mpi_reference._task_paths(campaign.run_dir, 0)
    record_path.unlink()
    adopted = mpi_reference._adopt_or_validate_task(campaign, 0)
    assert adopted is not None
    assert adopted["disposition"] == (
        "adopted_after_interrupted_record_publication"
    )

    with pytest.raises(
        mpi_reference.ReferenceMPIError,
        match="requires computed tasks",
    ):
        mpi_reference._write_evidence_archive(
            campaign,
            tmp_path / "ineligible-evidence.tar",
            execution_records=[],
        )


def test_cli_target_selection_requires_explicit_points_or_full_mesh() -> None:
    parser = mpi_reference._build_parser()
    common = [
        "--manifest",
        "manifest.json",
        "--source-dir",
        "source",
        "--force-stage-dir",
        "force",
        "--run-dir",
        "run",
        "--output-dir",
        "output",
        "--execution-id",
        "run-001",
        "--mesh",
        "3",
        "3",
        "3",
    ]
    selected = parser.parse_args([*common, "--grid-points", "0", "26"])
    assert selected.grid_points == [0, 26]
    assert not selected.all_grid_points
    complete = parser.parse_args([*common, "--all-grid-points"])
    assert complete.grid_points is None
    assert complete.all_grid_points
    with pytest.raises(SystemExit):
        parser.parse_args(common)


def test_target_consumer_authenticates_both_reference_generators(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    repository = Path(__file__).resolve().parents[1]
    serial_sha = mpi_reference._sha256_file(
        repository / "scripts" / "generate_phono3py_si_reference.py"
    )
    path = output / mpi_reference.MPI_ORCHESTRATION_FILENAME
    stage = json.loads((output / "reference-stage.json").read_text())
    checksums = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        checksums[line[66:]] = line[:64]
    reference = SimpleNamespace(
        mesh=(2, 1, 1),
        grid_points_grg=np.asarray([0, 1], dtype=np.int64),
        band_indices=np.asarray([0, 1], dtype=np.int64),
        n_qpoints=2,
        n_bands=2,
    )

    result = target_cli._validate_reference_mpi_orchestration(
        root=output,
        checksums=checksums,
        reference=reference,
        stage=stage,
        repository=repository,
        code_identity={"head_commit": "a" * 40},
        serial_generator_sha256=serial_sha,
    )

    assert result == document
    with pytest.raises(
        target_cli.SiliconTargetQCLIError,
        match="present or claimed",
    ):
        target_cli._validate_reference_mpi_orchestration(
            root=output,
            checksums={
                key: value
                for key, value in checksums.items()
                if key != path.name
            },
            reference=reference,
            stage=stage,
            repository=repository,
            code_identity={"head_commit": "a" * 40},
            serial_generator_sha256=serial_sha,
        )
    tampered = json.loads(json.dumps(document))
    tampered["generators"]["mpi_task_and_merge_driver"][
        "script_sha256"
    ] = "6" * 64
    path.write_text(
        json.dumps(tampered, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    tampered_sha = mpi_reference._sha256_file(path)
    tampered_checksums = {**checksums, path.name: tampered_sha}
    tampered_stage = json.loads(json.dumps(stage))
    tampered_stage["runtime"]["mpi_orchestration"][
        "orchestration_record_sha256"
    ] = tampered_sha
    with pytest.raises(
        target_cli.SiliconTargetQCLIError,
        match="does not bind the exact",
    ):
        target_cli._validate_reference_mpi_orchestration(
            root=output,
            checksums=tampered_checksums,
            reference=reference,
            stage=tampered_stage,
            repository=repository,
            code_identity={"head_commit": "a" * 40},
            serial_generator_sha256=serial_sha,
        )


def test_external_validator_authenticates_mpi_reference_wrapper(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    repository = Path(__file__).resolve().parents[1]
    reference_path = output / serial.REFERENCE_FILENAME
    reference = SimpleNamespace(
        mesh=(2, 1, 1),
        grid_points_grg=np.asarray([0, 1], dtype=np.int64),
        band_indices=np.asarray([0, 1], dtype=np.int64),
        n_qpoints=2,
        n_bands=2,
    )

    result = (
        external_validator._validate_reference_mpi_generation_provenance(
            SimpleNamespace(
                checksums_path=output / "SHA256SUMS",
                repository_path=repository,
                reference_path=reference_path,
            ),
            repository={"head_commit": "a" * 40},
            checksum_names=frozenset(
                line[66:]
                for line in (output / "SHA256SUMS").read_text().splitlines()
            ),
            reference=reference,
        )
    )

    assert result["campaign_sha256"] == document["campaign_sha256"]
    assert result["validated_evidence"]["task_count"] == 2
    assert result["validated_evidence"]["execution_count"] == 1
    assert result["validated_evidence"]["evidence_members"]


def test_reference_mpi_archive_rejects_minimal_missing_and_traversal_records(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    controls = _validation_controls()
    validated = validate_reference_mpi_evidence(
        output, document, **controls
    )
    assert validated["task_count"] == 2
    binding = validated["worker_bindings_by_execution"]["run-001"]
    assert binding["available"] is True
    assert binding["one_physical_core_per_assignment"] is True
    assert binding["disjoint_physical_cores_across_ranks"] is True
    assert [item["rank"] for item in binding["rank_bindings"]] == [1, 2]

    minimal = {
        "schema_name": document["schema_name"],
        "schema_version": 1,
        "campaign_sha256": document["campaign_sha256"],
        "tasks": [{"grid_point_grg": 0}, {"grid_point_grg": 1}],
    }
    with pytest.raises(
        ReferenceMPIProvenanceError, match="schema is incomplete"
    ):
        validate_reference_mpi_evidence(output, minimal, **controls)

    missing = json.loads(json.dumps(document))
    missing["tasks"].pop()
    with pytest.raises(
        ReferenceMPIProvenanceError, match="task evidence is incomplete"
    ):
        validate_reference_mpi_evidence(output, missing, **controls)

    traversal = json.loads(json.dumps(document))
    traversal["tasks"][0]["hdf5"]["archive_member"] = "../escape.hdf5"
    with pytest.raises(
        ReferenceMPIProvenanceError, match="normalized relative POSIX"
    ):
        validate_reference_mpi_evidence(output, traversal, **controls)


def test_reference_mpi_archive_rejects_tamper_and_restart_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    controls = _validation_controls()
    restart = json.loads(json.dumps(document))
    restart["restart"]["total_resumed_task_assignments"] = 1
    with pytest.raises(
        ReferenceMPIProvenanceError, match="restart facts"
    ):
        validate_reference_mpi_evidence(output, restart, **controls)

    archive = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    with archive.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(
        ReferenceMPIProvenanceError, match="archive identity changed"
    ):
        validate_reference_mpi_evidence(output, document, **controls)


def test_reference_mpi_remerges_archived_numerical_fields(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    archive_path = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = _archive_payloads(archive_path)
    hdf5_member = document["tasks"][0]["hdf5"]["archive_member"]
    record_member = document["tasks"][0]["record"]["archive_member"]
    payloads[hdf5_member] = _mutate_hdf5_bytes(
        payloads[hdf5_member],
        lambda h5: h5["gamma/values"].__setitem__(
            (0, 0, 0, 0),
            h5["gamma/values"][0, 0, 0, 0] + 1.0,
        ),
    )
    task_record = json.loads(payloads[record_member].decode("utf-8"))
    task_record["artifact"]["size_bytes"] = len(payloads[hdf5_member])
    task_record["artifact"]["sha256"] = hashlib.sha256(
        payloads[hdf5_member]
    ).hexdigest()
    payloads[record_member] = _json_payload(task_record)
    execution_member = document["executions"][0]["record"]["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    execution["assignments"][0]["artifact"] = dict(task_record["artifact"])
    payloads[execution_member] = _json_payload(execution)
    document["tasks"][0]["hdf5"] = _member_identity(
        hdf5_member, payloads[hdf5_member]
    )
    document["tasks"][0]["record"] = _member_identity(
        record_member, payloads[record_member]
    )
    document["executions"][0]["record"] = _member_identity(
        execution_member,
        payloads[execution_member],
    )
    _replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="gamma fields do not canonically merge",
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("merge", "declared merge summary differs"),
        ("merged_reference", "merged reference identity differs"),
    ),
)
def test_reference_mpi_rejects_declared_merge_identity_drift(
    tmp_path,
    monkeypatch,
    field,
    message,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    if field == "merge":
        document["merge"]["gamma_shape"][0] += 1
    else:
        document["merged_reference"]["sha256"] = "6" * 64
    _publish_orchestration(output, document)

    with pytest.raises(ReferenceMPIProvenanceError, match=message):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


def test_reference_mpi_rejects_generator_campaign_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    document["generators"]["serial_scientific_generator"][
        "script_sha256"
    ] = "6" * 64
    _publish_orchestration(output, document)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="campaign generators or task settings",
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        (
            "reference-settings.json",
            "source, settings, or runtime differs",
        ),
        (
            "input-manifest.normalized.json",
            "source, settings, or runtime differs|force-stage file",
        ),
    ),
)
def test_reference_mpi_rejects_actual_settings_and_input_drift(
    tmp_path,
    monkeypatch,
    filename,
    message,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    path = output / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    value["tampered"] = True
    path.write_bytes(_json_payload(value))

    with pytest.raises(ReferenceMPIProvenanceError, match=message):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


def test_reference_mpi_rejects_non_physical_core_worker_binding(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    archive_path = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = _archive_payloads(archive_path)
    execution_member = document["executions"][0]["record"]["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    execution["assignments"][0]["worker_binding"][
        "cpu_affinity_topology"
    ][1]["core_id"] += 1
    payloads[execution_member] = _json_payload(execution)
    document["executions"][0]["record"] = _member_identity(
        execution_member,
        payloads[execution_member],
    )
    _replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="exactly one physical core",
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


def test_reference_mpi_rejects_shared_physical_core_across_ranks(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    archive_path = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = _archive_payloads(archive_path)
    execution_member = document["executions"][0]["record"]["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    second = execution["assignments"][1]["worker_binding"]
    second["physical_core"] = {"package_id": 0, "core_id": 7}
    for item in second["cpu_affinity_topology"]:
        item["package_id"] = 0
        item["core_id"] = 7
    payloads[execution_member] = _json_payload(execution)
    document["executions"][0]["record"] = _member_identity(
        execution_member,
        payloads[execution_member],
    )
    _replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="distinct ranks to one physical core",
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


def test_reference_mpi_adopted_tasks_do_not_establish_numerical_origin(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    document = json.loads(json.dumps(document))
    archive_path = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    payloads = _archive_payloads(archive_path)
    for task in document["tasks"]:
        task_member = task["record"]["archive_member"]
        task_record = json.loads(payloads[task_member].decode("utf-8"))
        task_record[
            "disposition"
        ] = "adopted_after_interrupted_record_publication"
        task_record["execution"]["rank"] = None
        task_record["execution"]["started_at_utc"] = None
        task_record["execution"]["wall_seconds"] = None
        payloads[task_member] = _json_payload(task_record)
        task["record"] = _member_identity(
            task_member,
            payloads[task_member],
        )
    execution_member = document["executions"][0]["record"]["archive_member"]
    execution = json.loads(payloads[execution_member].decode("utf-8"))
    for assignment in execution["assignments"]:
        assignment["status"] = "adopted"
    payloads[execution_member] = _json_payload(execution)
    document["executions"][0]["record"] = _member_identity(
        execution_member,
        payloads[execution_member],
    )
    _replace_archive(output, document, payloads)

    with pytest.raises(
        ReferenceMPIProvenanceError,
        match="inconsistent|fresh, bound MPI computation",
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            **_validation_controls(),
        )


def test_reference_mpi_archive_rejects_symbolic_link(
    tmp_path,
    monkeypatch,
) -> None:
    _, output, document = _finalized_mpi_bundle(tmp_path, monkeypatch)
    archive = output / mpi_reference.MPI_EVIDENCE_ARCHIVE_FILENAME
    detached = tmp_path / "detached-evidence.tar"
    detached.write_bytes(archive.read_bytes())
    archive.unlink()
    archive.symlink_to(detached)
    with pytest.raises(
        ReferenceMPIProvenanceError, match="regular non-symlink"
    ):
        validate_reference_mpi_evidence(
            output,
            document,
            expected_mesh=(2, 1, 1),
            expected_grid_points=(0, 1),
            expected_band_indices=(0, 1),
            expected_n_qpoints=2,
            expected_n_bands=2,
            require_mpi=True,
        )


@pytest.mark.nightly
@pytest.mark.skipif(
    os.environ.get("NEGFPY_RUN_MPI_INTEGRATION") != "1",
    reason="set NEGFPY_RUN_MPI_INTEGRATION=1 for real MPI integration",
)
def test_real_mpi_reference_scheduler_and_changed_count_restart(
    tmp_path,
) -> None:
    binary_dir = Path(sys.executable).resolve().parent
    mpiexec = binary_dir / "mpiexec"
    if not mpiexec.is_file() or importlib.util.find_spec("mpi4py") is None:
        pytest.skip("active Python environment lacks local MPI/mpi4py")
    script = (
        Path(__file__).parent / "mpi" / "phono3py_reference_mpi_smoke.py"
    )
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "BLIS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    source_root = Path(__file__).resolve().parents[1] / "src"
    current_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source_root) + (
        "" if not current_pythonpath else os.pathsep + current_pythonpath
    )
    initial = subprocess.run(
        [
            str(mpiexec),
            "--bind-to",
            "core",
            "-n",
            "2",
            sys.executable,
            str(script),
            str(tmp_path),
            "initial",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert initial.returncode == 0, initial.stdout + initial.stderr
    assert '"communicator_size": 2' in initial.stdout
    os.rename(
        tmp_path / "output",
        tmp_path / "initial-output-evidence",
    )
    (
        tmp_path
        / "run"
        / "executions"
        / "smoke-initial.json"
    ).unlink()
    assert '"complete": true' in initial.stdout
    assert '"physical_core_binding_available": true' in initial.stdout
    restarted = subprocess.run(
        [
            str(mpiexec),
            "--bind-to",
            "core",
            "-n",
            "4",
            sys.executable,
            str(script),
            str(tmp_path),
            "restart",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=120,
    )
    assert restarted.returncode == 0, restarted.stdout + restarted.stderr
    assert '"communicator_size": 4' in restarted.stdout
    assert '"all_resumed": true' in restarted.stdout
    assert '"physical_core_binding_available": true' in restarted.stdout
