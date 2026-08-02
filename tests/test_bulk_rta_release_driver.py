from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import h5py
import numpy as np
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_bulk_rta_1_0.py"
_SPEC = importlib.util.spec_from_file_location(
    "negfpy_bulk_rta_release_driver", _SCRIPT
)
assert _SPEC is not None and _SPEC.loader is not None
driver = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = driver
_SPEC.loader.exec_module(driver)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata(mesh: int) -> tuple[dict[str, Any], dict[str, Any]]:
    interaction = {
        "schema": "negfpy.bulk_three_phonon_interactions.v1",
        "mesh": [mesh, mesh, mesh],
        "n_qpoints": mesh**3,
        "full_mesh_normalization": 1.0 / mesh**3,
        "make_r0_average": True,
        "method": "deterministic_serial_full_mesh",
        "phase_convention": "exp(+2π i q·r)",
        "triplet_convention": "q0+q1+q2=G",
    }
    linewidth = {
        "schema": "negfpy.bulk_three_phonon_linewidths.v1",
        "sigma_thz": 0.1,
        "sigma_cutoff": None,
        "cutoff_frequency_thz": driver.CUTOFF_THZ,
        "temperature_units": "K",
        "gamma_thz_semantics": "cyclic-frequency HWHM",
        "gamma_angular_fwhm_conversion": "4*pi*gamma_thz_hwhm",
        "interaction_metadata": interaction,
    }
    return linewidth, interaction


def _passing_harmonic_metrics() -> dict[str, Any]:
    return {
        "acoustic_zero_convention_cutoff_thz": driver.CUTOFF_THZ,
        "raw_uniform_mesh_frequency_max_abs_error_thz": 0.0,
        "uniform_mesh_frequency_max_abs_error_thz": 0.0,
        "group_velocity_max_abs_error_thz_angstrom": 0.0,
        "raw_band_frequency_max_abs_error_thz": 0.0,
        "band_frequency_max_abs_error_thz": 0.0,
        "dos_max_abs_error_states_per_thz_cell": 0.0,
        "dos_projection_sum_max_abs_error": 0.0,
        "dos_integral_states_per_cell": 6.0,
        "dos_expected_states_per_cell": 6,
        "dos_normalization_error": 0.0,
        "dos_normalization_certified": True,
        "fc2_schema_audit_passed": True,
        "fc2_physical_audit_passed": True,
        "passed": True,
    }


def _array_for(name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    if name == "linewidths/temperatures":
        return np.asarray([300.0], dtype=dtype)
    if name.endswith("/grid_points"):
        return np.arange(shape[0], dtype=dtype)
    if name.endswith("/band_indices"):
        return np.arange(6, dtype=dtype)
    if name == "phonons/qpoints":
        from negfpy.anharmonic.momentum import UniformBZMesh

        mesh = round(shape[0] ** (1.0 / 3.0))
        return (
            np.asarray(UniformBZMesh((mesh, mesh, mesh)).addresses, dtype=dtype) / mesh
        )
    if name == "phonons/dynamical_matrices":
        result = np.zeros(shape, dtype=dtype)
        for position in range(shape[0]):
            result[position] = np.eye(6)
        return result
    if name == "phonons/eigenvectors":
        return _array_for("phonons/dynamical_matrices", shape, dtype)
    if name == "phonons/frequencies":
        return np.ones(shape, dtype=dtype)
    if name == "phonons/frequency_factor_to_thz":
        return np.asarray(driver.FREQUENCY_FACTOR_TO_THZ, dtype=dtype)
    if name == "phonons/hermitianized":
        return np.asarray(True, dtype=dtype)
    if name.endswith("gamma_thz_hwhm"):
        return np.full(shape, 0.01, dtype=dtype)
    if name.endswith("gamma_normal_thz_hwhm"):
        return np.full(shape, 0.004, dtype=dtype)
    if name.endswith("gamma_umklapp_thz_hwhm"):
        return np.full(shape, 0.006, dtype=dtype)
    if name.endswith("gamma_angular_fwhm"):
        return np.full(shape, 4.0 * np.pi * 0.01, dtype=dtype)
    return np.zeros(shape, dtype=dtype)


def _write_synthetic_archive(path: Path, *, mesh: int = 1) -> dict[str, Any]:
    shapes = driver._archive_shapes(mesh)
    linewidth_metadata, interaction_metadata = _metadata(mesh)
    metadata = {
        "metadata/linewidths_json": driver._canonical_json(linewidth_metadata).encode(
            "ascii"
        ),
        "metadata/interactions_json": driver._canonical_json(
            interaction_metadata
        ).encode("ascii"),
    }
    root = {
        "problem_sha256": "1" * 64,
        "controls_sha256": "2" * 64,
        "campaign_manifest_sha256": "3" * 64,
        "manifest_run_identity_sha256": "4" * 64,
    }
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_name"] = "negfpy.bulk_target_q_merged"
        handle.attrs["schema_version"] = 1
        handle.attrs["floating_point_mpi_reductions"] = False
        for name, value in root.items():
            handle.attrs[name] = value
        handle.attrs["canonical_task_ids_json"] = driver._canonical_json(
            [f"target-q-{index:08d}" for index in range(mesh**3)]
        )
        handle.attrs["task_artifact_sha256_json"] = driver._canonical_json(
            ["5" * 64 for _ in range(mesh**3)]
        )
        for name in driver.ALL_ARCHIVE_DATASETS:
            if name in metadata:
                values = np.frombuffer(metadata[name], dtype=np.uint8)
            else:
                values = _array_for(
                    name,
                    shapes[name],
                    driver.ARCHIVE_DTYPES[name],
                )
            handle.create_dataset(name, data=values)
        hashes = {}
        for name in driver.ALL_ARCHIVE_DATASETS:
            digest = driver._dataset_sha256(handle[name])
            handle[name].attrs["sha256"] = digest
            hashes[name] = digest
        handle.attrs["dataset_sha256_json"] = driver._canonical_json(hashes)
    return {
        **root,
        "file_sha256": _sha256(path),
        "dataset_sha256": {
            name: hashes[name] for name in driver.SELECTED_ARCHIVE_DATASETS
        },
    }


def _config(tmp_path: Path, *, development: bool) -> driver.ValidationConfig:
    return driver.ValidationConfig(
        artifact_root=tmp_path / "artifact-root",
        crystal_yaml=tmp_path / "crystal.yaml",
        v0_2_release_manifest=tmp_path / "manifest.json",
        environment_lock=tmp_path / "environment.lock",
        environment_lock_sha256=(
            "a" * 64 if development else driver.PINNED_ENVIRONMENT_LOCK_SHA256
        ),
        output_dir=tmp_path / "output",
        meshes=(1,) if development else driver.RELEASE_MESHES,
        end_to_end_meshes=() if development else driver.RELEASE_END_TO_END_MESHES,
        temperatures_k=(300.0,),
        sigma_thz=0.1,
        development=development,
        resume=False,
    )


def test_release_controls_are_immutable_and_development_never_certifies(
    tmp_path: Path,
) -> None:
    release = _config(tmp_path, development=False)
    driver._validate_release_config(release, driver.STRICT_POLICY)
    with pytest.raises(ValueError, match="immutable"):
        driver._validate_release_config(
            replace(release, meshes=(3, 5)), driver.STRICT_POLICY
        )
    with pytest.raises(ValueError, match="immutable"):
        driver._validate_release_config(
            replace(release, sigma_thz=0.2), driver.STRICT_POLICY
        )
    with pytest.raises(ValueError, match="environment_lock_sha256"):
        driver._validate_release_config(
            replace(release, environment_lock_sha256="f" * 64),
            driver.STRICT_POLICY,
        )
    development = _config(tmp_path, development=True)
    driver._validate_release_config(development, driver.STRICT_POLICY)
    assert not driver._release_decision(
        development,
        scientific_passed=True,
        runtime_unchanged=True,
        runtime_after={"clean": True},
    )


def test_sealed_manifest_requires_authorized_exact_v020_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    payload = {
        "schema": "negfpy.silicon_0_2_0_sealed_release_package.v1",
        "phase": "sealed",
        "status": "passed",
        "release_authorization_ready": True,
        "candidate": {
            "commit": driver.PINNED_V0_2_COMMIT,
            "tree": driver.PINNED_V0_2_TREE,
            "version": "0.2.0",
        },
        "run": {
            "attempt_id": driver.PINNED_V0_2_ATTEMPT,
            "run_id": driver.PINNED_V0_2_RUN,
        },
        "visual_inspection": {"result": "PASS"},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    driver._validate_sealed_manifest(
        path, expected_sha256=_sha256(path), strict_release=True
    )
    payload["release_authorization_ready"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="authorization"):
        driver._validate_sealed_manifest(
            path, expected_sha256=_sha256(path), strict_release=True
        )


def test_archive_preflight_hashes_only_selected_fields_and_rejects_dtype(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "merged.h5"
    expected = _write_synthetic_archive(path)
    names = []
    original = driver._dataset_sha256

    def track(dataset: h5py.Dataset) -> str:
        names.append(dataset.name.lstrip("/"))
        return original(dataset)

    monkeypatch.setattr(driver, "_dataset_sha256", track)
    evidence = driver._validate_archive(
        path,
        mesh=1,
        expected=expected,
        temperatures_k=(300.0,),
        sigma_thz=0.1,
    )
    assert evidence["n_qpoints"] == 1
    assert set(names) == set(driver.SELECTED_ARCHIVE_DATASETS)
    assert not any(name.startswith("interactions/") for name in names)

    with h5py.File(path, "r+") as handle:
        del handle["linewidths/gamma_thz_hwhm"]
        dataset = handle.create_dataset(
            "linewidths/gamma_thz_hwhm",
            data=np.full((1, 1, 6), 0.01, dtype=np.float32),
        )
        dataset.attrs["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="wrong dtype"):
        driver._validate_archive(
            path,
            mesh=1,
            expected=expected,
            temperatures_k=(300.0,),
            sigma_thz=0.1,
        )


def test_preflight_failure_does_not_create_output_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, development=True)
    root = config.artifact_root
    (root / "force-constants").mkdir(parents=True)
    archive_dir = root / "target-q/mesh-1x1x1/merged"
    archive_dir.mkdir(parents=True)
    config.crystal_yaml.write_text("crystal", encoding="utf-8")
    (root / "force-constants/fc2.compact.hdf5").write_bytes(b"fc2")
    (root / "force-constants/fc3.compact.hdf5").write_bytes(b"fc3")
    config.v0_2_release_manifest.write_text("{}", encoding="utf-8")
    config.environment_lock.write_text("lock", encoding="utf-8")
    archive = archive_dir / "bulk-target-q.h5"
    expected = _write_synthetic_archive(archive)
    policy = driver.ReleasePolicy(
        manifest_sha256=_sha256(config.v0_2_release_manifest),
        environment_lock_sha256=_sha256(config.environment_lock),
        source_sha256={
            "crystal": _sha256(config.crystal_yaml),
            "fc2": _sha256(root / "force-constants/fc2.compact.hdf5"),
            "fc3": _sha256(root / "force-constants/fc3.compact.hdf5"),
        },
        archives={1: expected},
        runtime_packages=driver.PINNED_RUNTIME_PACKAGES,
        meshes=(1,),
        end_to_end_meshes=(),
    )
    archive.write_bytes(archive.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="whole-file SHA256"):
        driver.preflight_inputs(config, policy=policy, require_runtime=False)
    assert not config.output_dir.exists()

    target = tmp_path / "real-file"
    target.write_text("x", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        driver._regular_file(link, name="test")


def test_campaign_manifest_resume_is_exact(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    document = {"schema": driver.CAMPAIGN_SCHEMA, "input": "a"}
    path, digest = driver._publish_campaign_manifest(output, document, resume=False)
    same_path, same_digest = driver._publish_campaign_manifest(
        output, document, resume=True
    )
    assert same_path == path
    assert same_digest == digest
    with pytest.raises(ValueError, match="differs"):
        driver._publish_campaign_manifest(
            output,
            {"schema": driver.CAMPAIGN_SCHEMA, "input": "b"},
            resume=True,
        )


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"value": 1, "value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        driver._read_json_object(duplicate, name="duplicate fixture")

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(ValueError, match="valid UTF-8 JSON"):
        driver._read_json_object(nonfinite, name="nonfinite fixture")


def test_acoustic_zero_and_dos_stability_conventions_are_explicit() -> None:
    problem = SimpleNamespace(
        fc2_audit=SimpleNamespace(passed=True),
        fc2_physical_audit=SimpleNamespace(passed=True),
    )
    operands = {
        "mesh/frequencies_thz": np.asarray([[2.0e-7, 1.0]]),
        "reference/mesh_frequencies_thz": np.asarray([[-2.0e-7, 1.0]]),
        "mesh/group_velocities_thz_angstrom": np.zeros((1, 2, 3)),
        "reference/group_velocities_thz_angstrom": np.zeros((1, 2, 3)),
        "band/frequencies_thz": np.asarray([[2.0e-7, 1.0]]),
        "reference/band_frequencies_thz": np.asarray([[-2.0e-7, 1.0]]),
        "dos/frequency_points_thz": np.asarray([0.0, 1.0]),
        "dos/total_states_per_thz_cell": np.asarray([2.0, 2.0]),
        "reference/dos_total_states_per_thz_cell": np.asarray([2.0, 2.0]),
        "dos/projected_states_per_thz_cell": np.ones((1, 2, 2)),
    }
    record = driver._derive_harmonic_record(problem, operands)
    assert record["raw_uniform_mesh_frequency_max_abs_error_thz"] == pytest.approx(
        4.0e-7
    )
    assert record["uniform_mesh_frequency_max_abs_error_thz"] == 0.0
    assert record["band_frequency_max_abs_error_thz"] == 0.0
    assert record["passed"] is True
    np.testing.assert_array_equal(
        driver._apply_dos_stability_convention([-2.0e-7, 1.0]),
        [0.0, 1.0],
    )
    with pytest.raises(ValueError, match="dynamically unstable"):
        driver._apply_dos_stability_convention([-2.0 * driver.CUTOFF_THZ, 1.0])


def test_nested_external_hdf5_link_is_rejected(tmp_path: Path) -> None:
    external = tmp_path / "external.hdf5"
    with h5py.File(external, "w") as handle:
        handle.create_dataset("value", data=[1.0])
    payload = tmp_path / "payload.hdf5"
    identity = {"campaign_sha256": "c" * 64}
    with h5py.File(payload, "w") as handle:
        handle.attrs["schema"] = driver.CORE_REFERENCE_SCHEMA
        handle.attrs["campaign_sha256"] = identity["campaign_sha256"]
        handle.attrs["payload_sha256"] = "0" * 64
        nested = handle.create_group("nested")
        nested["value"] = h5py.ExternalLink(str(external), "/value")
    with pytest.raises(TypeError, match="internal hard link"):
        driver._read_checkpoint_payload(
            payload,
            schema=driver.CORE_REFERENCE_SCHEMA,
            identity=identity,
            dataset_names=frozenset({"nested/value"}),
        )


def test_resume_tree_rejects_checkpoint_symlink_before_compute(
    tmp_path: Path,
) -> None:
    config = replace(_config(tmp_path, development=True), resume=True)
    config.output_dir.mkdir()
    external = tmp_path / "external-checkpoints"
    external.mkdir()
    (config.output_dir / "checkpoints").symlink_to(external, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        driver._validate_existing_output_tree(config)


def test_resume_preflight_audits_later_corrupt_checkpoint_before_compute(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(tmp_path, development=True),
        meshes=(1, 2),
        end_to_end_meshes=(),
        resume=True,
    )
    later = config.output_dir / "checkpoints" / "mesh-002"
    later.mkdir(parents=True)
    (later / "rates.hdf5").write_bytes(b"not hdf5")
    snapshot = driver.InputSnapshot(
        paths={},
        sha256={},
        archive_paths={},
        archive_sha256={1: "a" * 64, 2: "b" * 64},
        archive_evidence={},
        sealed_manifest={},
    )
    with pytest.raises(ValueError, match="valid internal-only HDF5"):
        driver._preflight_resume_checkpoints(
            config,
            snapshot,
            campaign_sha256="c" * 64,
        )


def test_release_cli_fails_closed_on_uncertified_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = _config(tmp_path, development=False)
    monkeypatch.setattr(driver, "_parse_arguments", lambda argv=None: release)
    monkeypatch.setattr(
        driver,
        "run_validation",
        lambda config: {
            "scientific_passed": True,
            "release_certified": False,
        },
    )
    assert driver.main([]) == 3

    development = replace(release, development=True)
    monkeypatch.setattr(driver, "_parse_arguments", lambda argv=None: development)
    assert driver.main([]) == 0


def test_runtime_identity_requires_locked_versions_and_tied_import() -> None:
    commit = "c" * 40
    provenance = {
        "package_version_consistent": True,
        "installed_package_version": driver.PINNED_NEGFPY_VERSION,
        "git_revision": commit,
        "git_dirty": False,
        "source_tree_sha256": "d" * 64,
    }
    identity = {
        "clean": True,
        "commit": commit,
        "negfpy_version": driver.PINNED_NEGFPY_VERSION,
        "python_version": driver.PINNED_PYTHON_VERSION,
        "executable_within_prefix": True,
        "package_versions": dict(driver.PINNED_RUNTIME_PACKAGES),
        "module_versions": dict(driver.PINNED_RUNTIME_PACKAGES),
        "imports_within_prefix": {
            name: True for name in driver.PINNED_RUNTIME_PACKAGES
        },
        "pinned_import_paths_owned": {
            name: True for name in driver.PINNED_RUNTIME_PACKAGES
        },
        "mpiexec": {
            "version": driver.PINNED_OPENMPI_VERSION,
            "version_output": f"mpiexec (Open MPI) {driver.PINNED_OPENMPI_VERSION}",
            "within_prefix": True,
        },
        "code_provenance": provenance,
        "negfpy_import_tied_to_repository_payload": True,
        "negfpy_import_from_prefix": True,
        "validation_script_from_repository_scripts": True,
        "PYTHONPATH": None,
        "sys_path": {
            "entries": ["/repository/scripts", "/validated-prefix/lib/python3.12"],
            "entry_count": 2,
            "empty_entry_count": 0,
            "duplicate_entry_count": 0,
            "all_entries_allowed": True,
            "scripts_first": True,
        },
        "import_code_ownership": {"passed": True},
        "prefix": "/validated-prefix",
        "environment_binding": {
            "passed": True,
            "prefix": "/validated-prefix",
        },
        "thread_limits": {
            "environment": {name: "1" for name in driver.THREAD_LIMIT_ENVIRONMENT},
            "threadpools": [],
        },
        "user_site_isolation": {
            "PYTHONNOUSERSITE": "1",
            "no_user_site": True,
        },
    }
    driver._require_release_runtime(identity, policy=driver.STRICT_POLICY)

    wrong_package = {
        **identity,
        "package_versions": {**identity["package_versions"], "numpy": "0.0.0"},
    }
    with pytest.raises(RuntimeError, match="package_versions"):
        driver._require_release_runtime(
            wrong_package,
            policy=driver.STRICT_POLICY,
        )
    with pytest.raises(RuntimeError, match="imported_payload_tied"):
        driver._require_release_runtime(
            {**identity, "negfpy_import_tied_to_repository_payload": False},
            policy=driver.STRICT_POLICY,
        )
    with pytest.raises(RuntimeError, match="thread_environment"):
        driver._require_release_runtime(
            {
                **identity,
                "thread_limits": {
                    **identity["thread_limits"],
                    "environment": {
                        **identity["thread_limits"]["environment"],
                        "OMP_NUM_THREADS": "2",
                    },
                },
            },
            policy=driver.STRICT_POLICY,
        )
    with pytest.raises(RuntimeError, match="python_no_user_site_flag"):
        driver._require_release_runtime(
            {
                **identity,
                "user_site_isolation": {
                    "PYTHONNOUSERSITE": "1",
                    "no_user_site": False,
                },
            },
            policy=driver.STRICT_POLICY,
        )
    with pytest.raises(RuntimeError, match="sys_path_allowed"):
        driver._require_release_runtime(
            {
                **identity,
                "sys_path": {
                    **identity["sys_path"],
                    "all_entries_allowed": False,
                },
            },
            policy=driver.STRICT_POLICY,
        )
    with pytest.raises(RuntimeError, match="import_code_owned"):
        driver._require_release_runtime(
            {**identity, "import_code_ownership": {"passed": False}},
            policy=driver.STRICT_POLICY,
        )


@pytest.mark.parametrize("launcher", ("mpiexec", "mpirun", "orterun"))
def test_openmpi_launcher_banner_variants_are_release_equivalent(
    launcher: str,
) -> None:
    output = f"{launcher} (Open MPI) {driver.PINNED_OPENMPI_VERSION}\n"
    assert (
        driver._openmpi_version_from_banner(output)
        == driver.PINNED_OPENMPI_VERSION
    )
    assert driver._openmpi_version_from_banner("mpirun (MPICH) 5.0.10") == "unknown"


def test_sys_path_identity_rejects_arbitrary_prefix_subdirectory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    scripts = repository / "scripts"
    scripts.mkdir(parents=True)
    prefix = tmp_path / "prefix"
    standard_library = prefix / f"lib/python{sys.version_info.major}.{sys.version_info.minor}"
    standard_library.mkdir(parents=True)
    rogue = prefix / "rogue"
    rogue.mkdir()
    monkeypatch.setattr(sys, "path", [str(scripts), str(standard_library), str(rogue)])
    identity = driver._sys_path_identity(repository, prefix)
    assert identity["scripts_first"] is True
    assert identity["all_entries_allowed"] is False


def test_stable_runtime_identity_ignores_only_owned_generated_pyc_inventory(
    tmp_path: Path,
) -> None:
    before = {
        "import_code_ownership": {
            "passed": True,
            "code_path_count": 10,
            "code_paths_sha256": "a" * 64,
            "generated_pyc_path_count": 1,
            "generated_pyc_paths_sha256": "b" * 64,
            "unowned_code_path_count": 0,
            "unowned_code_paths": [],
        },
        "thread_limits": {"threadpools": []},
    }
    after = {
        **before,
        "import_code_ownership": {
            **before["import_code_ownership"],
            "generated_pyc_path_count": 2,
            "generated_pyc_paths_sha256": "c" * 64,
        },
    }
    assert driver._stable_runtime_identity(before) == driver._stable_runtime_identity(
        after
    )
    contaminated = {
        **after,
        "import_code_ownership": {
            **after["import_code_ownership"],
            "passed": False,
            "unowned_code_path_count": 1,
            "unowned_code_paths": ["site-packages/sitecustomize.py"],
        },
    }
    assert driver._stable_runtime_identity(before) != driver._stable_runtime_identity(
        contaminated
    )
    snapshot = driver.InputSnapshot(
        paths={},
        sha256={"manifest": "m" * 64},
        archive_paths={},
        archive_sha256={1: "a" * 64},
        archive_evidence={},
        sealed_manifest={},
    )
    config = _config(tmp_path, development=True)
    assert driver._campaign_document(
        config, snapshot, before
    ) == driver._campaign_document(config, snapshot, after)


def test_final_publication_contains_harmonic_fields_and_rejects_cross_mismatch(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    checkpoint = output / "checkpoints" / "mesh-001"
    checkpoint.mkdir(parents=True)
    rta_path = checkpoint / "rta.hdf5"
    with h5py.File(rta_path, "w") as handle:
        handle.attrs["schema"] = "synthetic-rta"
        handle.attrs["approximation"] = "three_phonon_smrta"
        handle.attrs["primitive_volume_angstrom3"] = 1.0
        handle.attrs["cutoff_frequency_thz"] = driver.CUTOFF_THZ
        handle.attrs["metadata_json"] = "{}"
        handle.attrs["payload_sha256"] = "a" * 64
        handle.create_dataset("temperatures", data=[300.0])
        handle.create_dataset("kappa_w_mk", data=np.eye(3)[None, :, :])
        handle.create_dataset("qpoints", data=np.zeros((1, 3)))
        handle.create_dataset("frequencies_thz", data=np.ones((1, 6)))
        handle.create_dataset("group_velocities_thz_angstrom", data=np.zeros((1, 6, 3)))
        for name in (
            "heat_capacities_ev_per_k",
            "gamma_thz_hwhm",
            "population_lifetime_ps",
            "mean_free_path_angstrom",
        ):
            handle.create_dataset(name, data=np.ones((1, 1, 6)))
        handle.create_dataset("mode_kappa_w_mk", data=np.zeros((1, 1, 6, 3, 3)))
        handle.create_dataset("band_index_kappa_w_mk", data=np.zeros((1, 6, 3, 3)))
        handle.create_dataset("qpoint_weights", data=np.ones(1))
        handle.create_dataset("included_mask", data=np.ones((1, 6), dtype=np.bool_))
    harmonic_path = checkpoint / "harmonic.hdf5"
    with h5py.File(harmonic_path, "w") as handle:
        handle.attrs["schema"] = driver.HARMONIC_SCHEMA
        handle.attrs["campaign_sha256"] = "c" * 64
        handle.attrs["archive_sha256"] = "a" * 64
        handle.attrs["mesh_json"] = "[1,1,1]"
        handle.attrs["payload_sha256"] = "b" * 64
        handle.create_dataset("band/qpoints", data=np.zeros((2, 3)))
        handle.create_dataset("band/distances_inv_angstrom", data=[0.0, 1.0])
        handle.create_dataset("band/vertex_indices", data=[0, 1])
        handle.create_dataset("band/frequencies_thz", data=np.ones((2, 6)))
        handle.create_dataset("dos/frequency_points_thz", data=[0.0, 1.0])
        handle.create_dataset("dos/total_states_per_thz_cell", data=[1.0, 1.0])
        handle.create_dataset(
            "dos/projected_states_per_thz_cell", data=np.ones((2, 3, 2)) / 6.0
        )
        handle.create_dataset("mesh/frequencies_thz", data=np.ones((1, 6)))
        handle.create_dataset(
            "mesh/group_velocities_thz_angstrom", data=np.zeros((1, 6, 3))
        )
        handle.create_dataset("reference/mesh_frequencies_thz", data=np.ones((1, 6)))
        handle.create_dataset(
            "reference/group_velocities_thz_angstrom", data=np.zeros((1, 6, 3))
        )
        handle.create_dataset("reference/band_frequencies_thz", data=np.ones((2, 6)))
        handle.create_dataset(
            "reference/dos_total_states_per_thz_cell", data=[1.0, 1.0]
        )

    mesh_evidence = [
        {
            "mesh": [1, 1, 1],
            "n_qpoints": 1,
            "passed": True,
            "core_passed": True,
            "end_to_end_required": False,
            "end_to_end_passed": None,
            "end_to_end_metrics": None,
            "harmonic_metrics": _passing_harmonic_metrics(),
            "rate_rta_frequency_max_abs_error_thz": 0.0,
            "rate_rta_frequency_within_tolerance": True,
            "maximum_heat_capacity_absolute_error_ev_per_k": 0.0,
            "mode_kappa_equal": True,
            "total_kappa_diagonal_equal": True,
            "total_kappa_offdiagonal_below_absolute_gate": True,
            "code_and_input_provenance_eligible": False,
            "rta_sha256": _sha256(rta_path),
            "harmonic_sha256": _sha256(harmonic_path),
        }
    ]
    campaign_path = output / "campaign-manifest.json"
    settings = {
        "meshes": [1],
        "end_to_end_meshes": [],
        "temperatures_k": [300.0],
        "sigma_thz": 0.1,
        "cutoff_frequency_thz": driver.CUTOFF_THZ,
        "frequency_factor_to_thz": driver.FREQUENCY_FACTOR_TO_THZ,
    }
    source_sha256 = {"fc2": "a" * 64}
    archive_sha256 = {"1": "b" * 64}
    campaign = {
        "schema": driver.CAMPAIGN_SCHEMA,
        "mode": "development",
        "release_certification_possible": False,
        "settings": settings,
        "inputs": {
            "source_sha256": source_sha256,
            "archive_sha256": archive_sha256,
            "archive_evidence": {},
            "sealed_manifest_sha256": "c" * 64,
            "staged_manifest_sha256": "d" * 64,
            "staged_source_sha256": source_sha256,
            "staged_rate_sha256": archive_sha256,
        },
        "runtime": {},
        "tasks": [
            {
                "task_id": "mesh-1x1x1",
                "mesh": [1, 1, 1],
                "requires_end_to_end": False,
                "dependencies": [],
            }
        ],
    }
    driver._write_json_atomic(campaign_path, campaign)
    campaign_sha256 = _sha256(campaign_path)
    staged_inputs = output / "staged-inputs"
    staged_inputs.mkdir()
    for name in (
        "crystal.yaml",
        "fc2.compact.hdf5",
        "fc3.compact.hdf5",
        "manifest.json",
        "mesh-001-rate-fields.hdf5",
    ):
        (staged_inputs / name).write_bytes(b"synthetic staged input\n")
    for name in (
        "core.json",
        "rates.hdf5",
        "core-reference.hdf5",
        "complete.json",
    ):
        (checkpoint / name).write_bytes(b"synthetic checkpoint\n")
    evidence_document = {
        "schema": driver.FINAL_EVIDENCE_SCHEMA,
        "campaign_sha256": campaign_sha256,
        "scientific_passed": True,
        "candidate_release_gates_passed": False,
        "release_certified": False,
        "certification_authority": "release-completion-receipt.json",
        "settings": settings,
        "source_sha256": source_sha256,
        "archive_sha256": archive_sha256,
        "meshes": mesh_evidence,
    }
    evidence_path = output / "release-evidence.json"
    evidence_sha256 = driver._publish_final_evidence(
        evidence_path,
        evidence_document,
        resume=False,
    )
    results_path = output / "bulk-rta-results.hdf5"
    driver._write_final_hdf5(
        results_path,
        mesh_evidence=mesh_evidence,
        output_dir=output,
        campaign_sha256=campaign_sha256,
        evidence_sha256=evidence_sha256,
    )
    report_path = output / "bulk-rta-validation.json"
    report = {
        "schema": driver.SCHEMA,
        "mode": "development",
        "campaign_sha256": campaign_sha256,
        "scientific_passed": True,
        "candidate_release_gates_passed": False,
        "release_certified": False,
        "certification_authority": "release-completion-receipt.json",
        "physical_mesh_convergence_certified": False,
        "physical_mesh_convergence_note": "Synthetic development fixture.",
        "approximation": "intrinsic_three_phonon_smrta",
        "excluded_physics": [],
        "campaign_manifest": str(campaign_path),
        "settings": settings,
        "runtime_before": {},
        "runtime_after": {},
        "runtime_unchanged": True,
        "inputs": {
            "source_paths": {},
            "source_sha256": source_sha256,
            "archive_paths": {},
            "archive_sha256": archive_sha256,
        },
        "evidence_document": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "results_hdf5": str(results_path),
        "results_hdf5_sha256": _sha256(results_path),
        "mesh_evidence": mesh_evidence,
        "meshes": [],
    }
    driver._write_json_atomic(report_path, report)
    driver._validate_staged_publication(
        report_path,
        results_path,
        evidence_path,
    )
    with pytest.raises(ValueError, match="runtime-stability decision"):
        driver._validate_campaign_publication_contract(
            campaign,
            {**report, "runtime_after": {"unexpected": True}},
            evidence_document,
        )
    bad_mesh = {
        **mesh_evidence[0],
        "harmonic_metrics": {
            **mesh_evidence[0]["harmonic_metrics"],
            "uniform_mesh_frequency_max_abs_error_thz": 1.0,
        },
    }
    with pytest.raises(ValueError, match="mesh decision"):
        driver._validate_campaign_publication_contract(
            campaign,
            {**report, "mesh_evidence": [bad_mesh]},
            {**evidence_document, "meshes": [bad_mesh]},
        )
    checksums_path = driver._write_checksum_manifest(output)
    receipt_path = output / "release-completion-receipt.json"
    driver._write_json_atomic(
        receipt_path,
        driver._completion_receipt_document(
            report_path=report_path,
            hdf5_path=results_path,
            evidence_path=evidence_path,
            checksums_path=checksums_path,
        ),
    )
    receipt = driver._validate_final_publication(
        report_path,
        results_path,
        evidence_path,
        checksums_path,
        receipt_path,
    )
    assert receipt["release_certified"] is False
    unexpected = output / "unexpected.txt"
    unexpected.write_text("not in sealed checksum manifest\n", encoding="utf-8")
    with pytest.raises(ValueError, match="(?:artifact layout|exact artifact tree)"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    unexpected.unlink()
    unexpected_directory = output / "unexpected-empty-directory"
    unexpected_directory.mkdir()
    with pytest.raises(ValueError, match="artifact layout"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    unexpected_directory.rmdir()
    valid_receipt = dict(receipt)
    missing_field = dict(valid_receipt)
    missing_field.pop("scientific_passed")
    driver._write_json_atomic(receipt_path, missing_field)
    with pytest.raises(ValueError, match="receipt_fields"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    driver._write_json_atomic(
        receipt_path,
        {**valid_receipt, "report_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="report_sha256"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    driver._write_json_atomic(
        receipt_path,
        {**valid_receipt, "release_certified": 1},
    )
    with pytest.raises(ValueError, match="receipt_boolean_types"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    driver._write_json_atomic(receipt_path, valid_receipt)
    receipt_path.unlink()
    with pytest.raises(ValueError, match="does not exist"):
        driver._validate_final_publication(
            report_path,
            results_path,
            evidence_path,
            checksums_path,
            receipt_path,
        )
    driver._write_json_atomic(receipt_path, valid_receipt)
    with h5py.File(results_path, "r") as handle:
        group = handle["meshes/mesh-1x1x1"]
        assert group.attrs["n_qpoints"] == 1
        assert "rta/kappa_w_mk" in group
        assert "harmonic/band/frequencies_thz" in group
        assert "harmonic/dos/total_states_per_thz_cell" in group
        assert "harmonic/mesh/group_velocities_thz_angstrom" in group
        assert handle.attrs["evidence_sha256"] == evidence_sha256
    assert "results_hdf5_sha256" not in evidence_document

    driver._write_json_atomic(
        report_path,
        {**report, "evidence_sha256": "0" * 64},
    )
    with pytest.raises(ValueError, match="report_evidence_sha256"):
        driver._validate_staged_publication(
            report_path,
            results_path,
            evidence_path,
        )


def test_explicit_lock_binds_exact_prefix_urls_paths_and_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prefix = tmp_path / "prefix"
    (prefix / "conda-meta").mkdir(parents=True)
    (prefix / "bin").mkdir()
    pycache = prefix / "lib/python3.12/__pycache__"
    pycache.mkdir(parents=True)
    installed = prefix / "bin/tool"
    installed.write_bytes(b"validated executable\n")
    regenerated = pycache / "module.cpython-312.pyc"
    regenerated.write_bytes(b"runtime-regenerated-bytecode\n")
    url = "https://conda.example/linux-64/tool-1.0-0.conda"
    package_cache = tmp_path / "package-cache"
    package_cache.mkdir()
    cached_artifact = package_cache / "tool-1.0-0.conda"
    cached_artifact.write_bytes(b"authenticated conda artifact\n")
    artifact_sha256 = _sha256(cached_artifact)
    record = {
        "url": url,
        "fn": "tool-1.0-0.conda",
        "name": "tool",
        "version": "1.0",
        "build": "0",
        "build_number": 0,
        "subdir": "linux-64",
        "size": cached_artifact.stat().st_size,
        "md5": "b" * 32,
        "sha256": artifact_sha256,
        "extracted_package_dir": str(package_cache / "tool-1.0-0"),
        "paths_data": {
            "paths_version": 1,
            "paths": [
                {
                    "_path": "bin/tool",
                    "path_type": "hardlink",
                    "sha256_in_prefix": _sha256(installed),
                },
                {
                    "_path": ("lib/python3.12/__pycache__/module.cpython-312.pyc"),
                    "path_type": "hardlink",
                    "sha256_in_prefix": hashlib.sha256(
                        b"packaged-bytecode\n"
                    ).hexdigest(),
                },
            ],
        },
    }
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    lock = tmp_path / "explicit.lock"
    lock.write_text(
        f"# release lock\n@EXPLICIT\n{url}#sha256={artifact_sha256}\n"
    )
    binding = driver._validate_conda_prefix_against_lock(lock, prefix)
    assert binding["passed"] is True
    assert binding["explicit_url_count"] == 1
    assert binding["explicit_artifact_sha256_count"] == 1
    assert binding["record_artifact_sha256_count"] == 1
    assert binding["record_artifact_md5_count"] == 1
    assert binding["cache_verified_artifact_sha256_count"] == 0
    assert len(binding["record_artifact_identity_sha256"]) == 64
    assert binding["hashed_file_count"] == 1
    assert binding["hash_declared_file_count"] == 2
    assert binding["skipped_hash_path_count"] == 1
    assert binding["skipped_hash_categories"] == {"python_generated_bytecode": 1}
    assert len(binding["skipped_hash_paths_sha256"]) == 64
    assert driver._mutable_conda_hash_category("lib/python3.12/module.pyc") is None
    assert (
        driver._mutable_conda_hash_category("lib/python3.12/__pycache__/module.py")
        is None
    )
    site_packages = prefix / "lib/python3.12/site-packages"
    site_packages.mkdir()
    (site_packages / "sitecustomize.py").write_text(
        "UNOWNED = True\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        driver,
        "_negfpy_wheel_owned_prefix_paths",
        lambda _prefix: frozenset(),
    )
    ownership = driver._import_code_ownership_identity(prefix)
    assert ownership["passed"] is False
    assert ownership["unowned_code_paths"] == [
        "lib/python3.12/site-packages/sitecustomize.py"
    ]

    record["md5"] = None
    record["sha256"] = None
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    sha_only_binding = driver._validate_conda_prefix_against_lock(lock, prefix)
    assert sha_only_binding["record_artifact_sha256_count"] == 0
    assert sha_only_binding["record_artifact_md5_count"] == 0
    assert sha_only_binding["cache_verified_artifact_sha256_count"] == 1
    cached_payload = cached_artifact.read_bytes()
    cached_artifact.write_bytes(b"tampered artifact payload\n")
    record["size"] = cached_artifact.stat().st_size
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="Cached conda artifact SHA256 conflicts"):
        driver._validate_conda_prefix_against_lock(lock, prefix)
    cached_artifact.write_bytes(cached_payload)
    record["size"] = cached_artifact.stat().st_size
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    lock.write_text(f"@EXPLICIT\n{url}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no authenticated artifact digest"):
        driver._validate_conda_prefix_against_lock(lock, prefix)
    lock.write_text(
        f"@EXPLICIT\n{url}#sha256={artifact_sha256}\n", encoding="utf-8"
    )
    record["md5"] = "b" * 32
    record["sha256"] = artifact_sha256
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )

    installed.write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="SHA256 changed"):
        driver._validate_conda_prefix_against_lock(lock, prefix)
    installed.write_bytes(b"validated executable\n")
    record["sha256"] = "b" * 64
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="conflicts with lock"):
        driver._validate_conda_prefix_against_lock(lock, prefix)
    record["sha256"] = artifact_sha256
    (prefix / "conda-meta/tool-1.0-0.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    lock.write_text("@EXPLICIT\nhttps://conda.example/linux-64/other.conda\n")
    with pytest.raises(ValueError, match="URL set differs"):
        driver._validate_conda_prefix_against_lock(lock, prefix)


def test_authenticated_operand_payload_and_disposable_json_are_fail_closed(
    tmp_path: Path,
) -> None:
    payload = tmp_path / "operands.hdf5"
    identity = {"campaign_sha256": "c" * 64, "mesh_json": "[1,1,1]"}
    driver._write_checkpoint_payload(
        payload,
        schema=driver.CORE_REFERENCE_SCHEMA,
        identity=identity,
        datasets={"value": np.asarray([1.0, 2.0])},
    )
    restored = driver._read_checkpoint_payload(
        payload,
        schema=driver.CORE_REFERENCE_SCHEMA,
        identity=identity,
        dataset_names=frozenset({"value"}),
    )
    np.testing.assert_array_equal(restored["value"], [1.0, 2.0])
    with h5py.File(payload, "r+") as handle:
        handle["value"][0] = 9.0
    with pytest.raises(ValueError, match="payload SHA256 changed"):
        driver._read_checkpoint_payload(
            payload,
            schema=driver.CORE_REFERENCE_SCHEMA,
            identity=identity,
            dataset_names=frozenset({"value"}),
        )

    checkpoint = tmp_path / "core.json"
    original = {
        "schema": driver.CHECKPOINT_SCHEMA,
        "stage": "core",
        "mesh": [1, 1, 1],
        "campaign_sha256": "c" * 64,
        "passed": True,
        "metric": 0.0,
    }
    driver._write_json_atomic(checkpoint, original)
    derived = {**original, "passed": False, "metric": 1.0}
    result = driver._publish_derived_checkpoint(
        checkpoint,
        derived,
        stage="core",
        mesh=1,
        campaign_sha256="c" * 64,
    )
    assert result["passed"] is False
    assert result["metric"] == 1.0

    checkpoint.write_bytes(b'{"truncated":')
    result = driver._publish_derived_checkpoint(
        checkpoint,
        derived,
        stage="core",
        mesh=1,
        campaign_sha256="c" * 64,
    )
    assert result["passed"] is False
    assert list(tmp_path.glob("core.json.invalid-*"))

    driver._write_json_atomic(
        checkpoint,
        {**original, "campaign_sha256": "d" * 64},
    )
    with pytest.raises(ValueError, match="another identity"):
        driver._publish_derived_checkpoint(
            checkpoint,
            derived,
            stage="core",
            mesh=1,
            campaign_sha256="c" * 64,
        )


def test_preflight_only_starts_no_compute_and_creates_no_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(tmp_path, development=True), preflight_only=True)
    snapshot = driver.InputSnapshot(
        paths={},
        sha256={"manifest": "m" * 64},
        archive_paths={},
        archive_sha256={1: "a" * 64},
        archive_evidence={},
        sealed_manifest={},
    )
    monkeypatch.setattr(
        driver,
        "preflight_inputs",
        lambda *args, **kwargs: (snapshot, {"runtime": "validated"}),
    )
    monkeypatch.setattr(
        driver,
        "_stage_scientific_inputs",
        lambda *args, **kwargs: pytest.fail("staging/compute must not start"),
    )
    receipt = driver.run_validation(config)
    assert receipt["preflight_passed"] is True
    assert receipt["release_certified"] is False
    assert receipt["scientific_compute_started"] is False
    assert not config.output_dir.exists()


def test_scientific_staging_copies_sources_and_only_selected_rate_fields(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, development=True)
    config.output_dir.mkdir()
    sources = {}
    hashes = {}
    for name in ("crystal", "fc2", "fc3"):
        path = tmp_path / f"{name}.input"
        path.write_bytes((name + "\n").encode())
        sources[name] = path
        hashes[name] = _sha256(path)
    archive = tmp_path / "bulk-target-q.h5"
    _write_synthetic_archive(archive, mesh=1)
    archive_sha256 = _sha256(archive)
    snapshot = driver.InputSnapshot(
        paths=sources,
        sha256=hashes,
        archive_paths={1: archive},
        archive_sha256={1: archive_sha256},
        archive_evidence={},
        sealed_manifest={},
    )
    staged = driver._stage_scientific_inputs(config, snapshot)
    for name, source in sources.items():
        assert staged.paths[name] != source
        assert not staged.paths[name].is_symlink()
        assert _sha256(staged.paths[name]) == hashes[name]
    operands = driver._read_checkpoint_payload(
        staged.archive_paths[1],
        schema=driver.STAGED_RATE_SCHEMA,
        identity={
            "source_archive_sha256": archive_sha256,
            "mesh_json": driver._canonical_json([1, 1, 1]),
        },
        dataset_names=driver.STAGED_RATE_DATASETS,
    )
    assert set(operands) == set(driver.SELECTED_ARCHIVE_DATASETS)
    assert not any(name.startswith("interactions/") for name in operands)
    driver._verify_staged_inputs(staged, snapshot, meshes=(1,))

    staged.paths["fc2"].write_bytes(b"tampered\n")
    with pytest.raises(RuntimeError, match="Staged input 'fc2'"):
        driver._verify_staged_inputs(staged, snapshot, meshes=(1,))


def test_fresh_core_and_resume_rederive_json_without_heavy_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phonopy.phonon.thermal_properties import mode_cv
    from phonopy.physical_units import get_physical_units

    from negfpy import bulk
    from negfpy.anharmonic.bulk_rta import (
        BulkThreePhononRateSpectrum,
        RTAThermalConductivity,
    )

    frequencies = np.ones((1, 6), dtype=np.float64)
    qpoints = np.zeros((1, 3), dtype=np.float64)
    temperatures = np.asarray([300.0])
    gamma = np.full((1, 1, 6), 0.01)
    rates = BulkThreePhononRateSpectrum(
        temperatures=temperatures,
        qpoints=qpoints,
        frequencies_thz=frequencies,
        grid_points=np.asarray([0]),
        band_indices=np.arange(6),
        gamma_thz_hwhm=gamma,
        gamma_normal_thz_hwhm=gamma * 0.4,
        gamma_umklapp_thz_hwhm=gamma * 0.6,
        gamma_angular_fwhm=4.0 * np.pi * gamma,
        decay_phase_space=np.zeros((1, 6)),
        difference_phase_space=np.zeros((1, 6)),
        cutoff_frequency_thz=driver.CUTOFF_THZ,
        frequency_factor_to_thz=driver.FREQUENCY_FACTOR_TO_THZ,
        physics_input_sha256="a" * 64,
        metadata={"synthetic": True},
    )
    heat_capacity = np.asarray(
        mode_cv(
            temperatures,
            (frequencies * get_physical_units().THzToEv).ravel(),
        )
    ).reshape(1, 1, 6)
    velocities = np.zeros((1, 6, 3))
    lifetime = 1.0 / (4.0 * np.pi * gamma)
    result = RTAThermalConductivity(
        temperatures=temperatures,
        qpoints=qpoints,
        frequencies_thz=frequencies + 2.0e-14,
        group_velocities_thz_angstrom=velocities,
        heat_capacities_ev_per_k=heat_capacity,
        gamma_thz_hwhm=gamma,
        population_lifetime_ps=lifetime,
        mean_free_path_angstrom=np.zeros_like(gamma),
        mode_kappa_w_mk=np.zeros((1, 1, 6, 3, 3)),
        band_index_kappa_w_mk=np.zeros((1, 6, 3, 3)),
        kappa_w_mk=np.zeros((1, 3, 3)),
        qpoint_weights=np.ones(1),
        primitive_volume_angstrom3=10.0,
        cutoff_frequency_thz=driver.CUTOFF_THZ,
        included_mask=np.ones((1, 6), dtype=np.bool_),
        metadata={"code_and_input_provenance_eligible": True},
    )
    problem = SimpleNamespace(
        primitive_volume_angstrom3=10.0,
        fc2_audit=SimpleNamespace(passed=True),
        fc2_physical_audit=SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(
        bulk.BulkTransportProblem,
        "from_phono3py",
        classmethod(lambda cls, *args, **kwargs: problem),
    )
    monkeypatch.setattr(driver, "_make_rate_spectrum", lambda *args, **kwargs: rates)
    monkeypatch.setattr(
        driver,
        "_reference_mode_kappa",
        lambda frequencies, velocities, heat_capacity, gamma_values, volume: np.zeros(
            np.asarray(gamma_values).shape + (3, 3)
        ),
    )
    compute_calls = []

    def compute(*args: Any, **kwargs: Any) -> RTAThermalConductivity:
        compute_calls.append(True)
        return result

    monkeypatch.setattr(bulk, "compute_rta_thermal_conductivity", compute)
    ph3_calls = []

    def ph3_problem(*args: Any, **kwargs: Any) -> SimpleNamespace:
        ph3_calls.append(True)
        return SimpleNamespace()

    monkeypatch.setattr(driver, "_phono3py_reference_problem", ph3_problem)
    monkeypatch.setattr(
        driver, "_initialize_phono_dynamical_matrix", lambda *args: None
    )

    def harmonic_validation(
        supplied_problem: Any,
        archived: Any,
        ph3: Any,
        *,
        mesh: int,
        output_path: Path,
        campaign_sha256: str,
        archive_sha256: str,
    ) -> tuple[dict[str, Any], Any, np.ndarray]:
        points = np.asarray([0.0, 1.0])
        total = np.asarray([6.0, 6.0])
        operands = {
            "band/qpoints": np.zeros((2, 3)),
            "band/distances_inv_angstrom": points,
            "band/vertex_indices": np.asarray([0, 1]),
            "band/frequencies_thz": np.ones((2, 6)),
            "dos/frequency_points_thz": points,
            "dos/total_states_per_thz_cell": total,
            "dos/projected_states_per_thz_cell": np.ones((2, 3, 2)),
            "mesh/frequencies_thz": frequencies,
            "mesh/group_velocities_thz_angstrom": velocities,
            "reference/mesh_frequencies_thz": frequencies,
            "reference/group_velocities_thz_angstrom": velocities,
            "reference/band_frequencies_thz": np.ones((2, 6)),
            "reference/dos_total_states_per_thz_cell": total,
        }
        identity = {
            "campaign_sha256": campaign_sha256,
            "archive_sha256": archive_sha256,
            "mesh_json": driver._canonical_json([mesh, mesh, mesh]),
        }
        driver._write_checkpoint_payload(
            output_path,
            schema=driver.HARMONIC_SCHEMA,
            identity=identity,
            datasets=operands,
        )
        restored = driver._read_checkpoint_payload(
            output_path,
            schema=driver.HARMONIC_SCHEMA,
            identity=identity,
            dataset_names=driver.HARMONIC_CHECKPOINT_DATASETS,
        )
        return (
            driver._derive_harmonic_record(supplied_problem, restored),
            SimpleNamespace(values_thz_angstrom=velocities),
            velocities,
        )

    monkeypatch.setattr(driver, "_harmonic_validation", harmonic_validation)

    raw_archive = tmp_path / "raw.hdf5"
    _write_synthetic_archive(raw_archive, mesh=1)
    source_archive_sha256 = _sha256(raw_archive)
    staged_archive = tmp_path / "staged-rate.hdf5"
    with h5py.File(raw_archive, "r") as handle:
        staged_fields = {
            name: np.asarray(handle[name]) for name in driver.STAGED_RATE_DATASETS
        }
    driver._write_checkpoint_payload(
        staged_archive,
        schema=driver.STAGED_RATE_SCHEMA,
        identity={
            "source_archive_sha256": source_archive_sha256,
            "mesh_json": driver._canonical_json([1, 1, 1]),
        },
        datasets=staged_fields,
    )
    dummy = tmp_path / "dummy"
    dummy.write_text("input", encoding="utf-8")
    snapshot = driver.InputSnapshot(
        paths={"crystal": dummy, "fc2": dummy, "fc3": dummy},
        sha256={
            "crystal": _sha256(dummy),
            "fc2": _sha256(dummy),
            "fc3": _sha256(dummy),
        },
        archive_paths={1: staged_archive},
        archive_sha256={1: source_archive_sha256},
        archive_evidence={},
        sealed_manifest={},
    )
    config = replace(
        _config(tmp_path, development=True),
        end_to_end_meshes=(1,),
        resume=True,
    )
    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    campaign_sha256 = "c" * 64
    fresh = driver._compute_core_stage(
        config,
        snapshot,
        mesh=1,
        checkpoint_dir=checkpoint_dir,
        campaign_sha256=campaign_sha256,
    )
    assert fresh["passed"] is True
    assert fresh["harmonic"]["dos_projection_sum_max_abs_error"] == 0.0
    assert fresh["rate_rta_frequency_within_tolerance"] is True
    assert fresh["rate_rta_frequency_max_abs_error_thz"] == pytest.approx(2.0e-14)

    driver._write_json_atomic(
        checkpoint_dir / "core.json",
        {
            **fresh,
            "passed": False,
            "maximum_heat_capacity_absolute_error_ev_per_k": 9.0,
        },
    )
    resumed = driver._compute_core_stage(
        config,
        snapshot,
        mesh=1,
        checkpoint_dir=checkpoint_dir,
        campaign_sha256=campaign_sha256,
    )
    assert resumed["passed"] is True
    assert resumed["maximum_heat_capacity_absolute_error_ev_per_k"] == 0.0
    assert len(compute_calls) == 2

    (checkpoint_dir / "core-reference.hdf5").unlink()
    before_partial_ph3_calls = len(ph3_calls)
    recovered = driver._compute_core_stage(
        config,
        snapshot,
        mesh=1,
        checkpoint_dir=checkpoint_dir,
        campaign_sha256=campaign_sha256,
    )
    assert recovered["passed"] is True
    assert len(ph3_calls) == before_partial_ph3_calls
    assert len(compute_calls) == 3

    rate_path = checkpoint_dir / "rates.hdf5"
    rta_path = checkpoint_dir / "rta.hdf5"
    e2e_path = checkpoint_dir / "end-to-end-reference.hdf5"
    e2e_identity = {
        "campaign_sha256": campaign_sha256,
        "archive_sha256": source_archive_sha256,
        "mesh_json": driver._canonical_json([1, 1, 1]),
        "sigma_thz": config.sigma_thz,
        "rate_sha256": _sha256(rate_path),
        "rta_sha256": _sha256(rta_path),
    }
    driver._write_checkpoint_payload(
        e2e_path,
        schema=driver.END_TO_END_PAYLOAD_SCHEMA,
        identity=e2e_identity,
        datasets={
            "temperatures": rates.temperatures,
            "grid_points": rates.grid_points,
            "band_indices": rates.band_indices,
            "qpoints": rates.qpoints,
            "gamma_thz_hwhm": rates.gamma_thz_hwhm,
            "kappa_voigt_w_mk": result.kappa_voigt_w_mk,
        },
    )
    before_e2e_calls = len(ph3_calls)
    e2e = driver._compute_end_to_end_stage(
        config,
        snapshot,
        mesh=1,
        checkpoint_dir=checkpoint_dir,
        campaign_sha256=campaign_sha256,
    )
    assert e2e is not None and e2e["passed"] is True
    assert len(ph3_calls) == before_e2e_calls
    driver._write_json_atomic(
        checkpoint_dir / "end-to-end.json", {**e2e, "passed": False}
    )
    resumed_e2e = driver._compute_end_to_end_stage(
        config,
        snapshot,
        mesh=1,
        checkpoint_dir=checkpoint_dir,
        campaign_sha256=campaign_sha256,
    )
    assert resumed_e2e is not None and resumed_e2e["passed"] is True
    assert len(ph3_calls) == before_e2e_calls
    with h5py.File(e2e_path, "r+") as handle:
        handle["gamma_thz_hwhm"][0, 0, 0] *= 2.0
    with pytest.raises(ValueError, match="payload SHA256 changed"):
        driver._compute_end_to_end_stage(
            config,
            snapshot,
            mesh=1,
            checkpoint_dir=checkpoint_dir,
            campaign_sha256=campaign_sha256,
        )
    assert len(ph3_calls) == before_e2e_calls
