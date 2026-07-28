from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.generate_phono3py_si_reference import (
    FC2_RANDOM_QPOINTS,
    FORCE_CONSTANT_FILENAMES,
    REFERENCE_FILENAME,
    GeneratorConfig,
    _Phono3pyRuntime,
    load_input_manifest,
    run_generator,
    verify_sha256sums,
)


PINNED_COMMIT = "a" * 40
PUBLIC_SI_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "materials"
    / "si-pbesol-phono3py-4.4.0"
    / "input-manifest.json"
)
PUBLIC_SI_MPI_LOCK = (
    Path(__file__).resolve().parents[1]
    / "validation"
    / "locks"
    / "phono3py-4.4.0-si-pbesol-openmpi-linux-64.lock"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_source_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    yaml_path = source_dir / "phono3py_disp.yaml"
    forces_path = source_dir / "FORCES_FC3"
    readme_path = source_dir / "README.md"
    yaml_path.write_text("synthetic yaml\n", encoding="utf-8")
    forces_path.write_text("synthetic forces\n", encoding="utf-8")
    readme_path.write_text("synthetic source\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "dataset_id": "phono3py-4.4.0-Si-PBEsol",
        "versions": {
            "phono3py": "4.4.0",
            "phonopy": "4.4.0",
        },
        "source": {
            "repository": "https://github.com/phonopy/phono3py.git",
            "commit": PINNED_COMMIT,
            "files": {
                "phono3py_yaml": {
                    "path": "phono3py_disp.yaml",
                    "sha256": _sha256(yaml_path),
                },
                "forces_fc3": {
                    "path": "FORCES_FC3",
                    "sha256": _sha256(forces_path),
                },
                "readme": {
                    "path": "README.md",
                    "sha256": _sha256(readme_path),
                },
            },
        },
    }
    manifest_path = tmp_path / "input-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return source_dir, manifest_path


class _FakeRuntime:
    def __init__(
        self,
        *,
        phono3py_version: str = "4.4.0",
        phonopy_version: str = "4.4.0",
    ) -> None:
        self.phono3py_version = phono3py_version
        self.phonopy_version = phonopy_version
        self.force_calls = 0
        self.reference_calls = 0

    def verify_versions(self, expected):
        actual = {
            "phono3py": self.phono3py_version,
            "phonopy": self.phonopy_version,
        }
        if actual != dict(expected):
            raise RuntimeError(f"Version mismatch: expected {dict(expected)}, got {actual}")
        return actual

    def generate_force_constants(
        self,
        *,
        output_dir,
        source_files,
        config,
    ):
        self.force_calls += 1
        assert set(source_files) >= {"phono3py_yaml", "forces_fc3"}
        for name in FORCE_CONSTANT_FILENAMES:
            target = output_dir / name
            assert not target.exists()
            target.write_bytes(f"synthetic {name}\n".encode())
        return {
            "api": "fake.generate_force_constants",
            "compact_fc2_shape": [1, 2, 3, 3],
            "full_fc2_shape": [2, 2, 3, 3],
            "compact_fc3_shape": [1, 2, 2, 3, 3, 3],
            "full_fc3_shape": [2, 2, 2, 3, 3, 3],
            "symmetrize_fc": config.symmetrize_fc,
        }

    def generate_reference(
        self,
        *,
        output_dir,
        source_files,
        config,
    ):
        self.reference_calls += 1
        target = output_dir / REFERENCE_FILENAME
        assert not target.exists()
        target.write_bytes(b"synthetic reference hdf5\n")
        return {
            "api": "fake.generate_reference",
            "mesh": list(config.mesh),
            "grid_points": list(config.grid_points),
            "band_indices": (
                None if config.band_indices is None else list(config.band_indices)
            ),
            "gamma_units": "cyclic_THz_HWHM",
        }


def _force_only_config() -> GeneratorConfig:
    return GeneratorConfig(force_constants_only=True)


def _reference_config() -> GeneratorConfig:
    return GeneratorConfig(
        mesh=(3, 3, 3),
        grid_points=(0, 2),
        band_indices=(0, 2, 5),
        temperatures=(0.0, 300.0),
        sigma=0.1,
        backend="Rust",
    )


def test_manifest_is_strict_detached_and_immutable(tmp_path) -> None:
    _, manifest_path = _write_source_and_manifest(tmp_path)

    manifest = load_input_manifest(manifest_path)

    assert manifest.schema_version == 1
    assert manifest.dataset_id == "phono3py-4.4.0-Si-PBEsol"
    assert manifest.source_commit == PINNED_COMMIT
    assert manifest.expected_versions == (
        ("phono3py", "4.4.0"),
        ("phonopy", "4.4.0"),
    )
    assert manifest.file("forces_fc3").relative_path == Path("FORCES_FC3")
    assert manifest.input_sha256 == _sha256(manifest_path)
    with pytest.raises(FrozenInstanceError):
        manifest.dataset_id = "changed"


def test_committed_si_manifest_pins_public_dataset_and_inputs() -> None:
    manifest = load_input_manifest(PUBLIC_SI_MANIFEST)

    assert manifest.dataset_id == "phono3py-4.4.0-Si-PBEsol"
    assert (
        manifest.source_commit
        == "302ad205452ee36e57887b10bbf4a0289dd037bb"
    )
    assert manifest.expected_versions == (
        ("phono3py", "4.4.0"),
        ("phonopy", "4.4.0"),
    )
    assert manifest.file("phono3py_yaml").relative_path == Path(
        "example/Si-PBEsol/phono3py_disp.yaml"
    )
    assert (
        manifest.file("phono3py_yaml").sha256
        == "1375e3ae19a2c4138a888af8177a3becfcaa3b127ca50755773c405c3e28a2a7"
    )
    assert (
        manifest.file("forces_fc3").sha256
        == "d53c5443f028c5d66c70adff20191e00b54af56a81e93d8d20e28f9320c94039"
    )


def test_fc2_random_qpoints_are_fixed_nonmesh_validation_points() -> None:
    points = np.asarray(FC2_RANDOM_QPOINTS, dtype=np.float64)

    assert points.shape == (8, 3)
    assert np.unique(points, axis=0).shape[0] == len(points)
    assert np.all(points >= -0.5)
    assert np.all(points < 0.5)
    assert not np.any(np.all(points == 0.0, axis=1))


def test_committed_material_mpi_lock_pins_oracle_and_runtime() -> None:
    lines = PUBLIC_SI_MPI_LOCK.read_text(encoding="utf-8").splitlines()

    assert lines[2] == "# platform: linux-64"
    assert lines[3] == "@EXPLICIT"
    urls = set(lines[4:])
    required_fragments = {
        "python-3.12.13-",
        "phono3py-4.4.0-",
        "phonopy-4.4.0-",
        "spglib-2.7.0-",
        "openmpi-5.0.10-",
        "mpi4py-4.1.2-",
        "numpy-2.5.1-",
        "scipy-1.18.0-",
        "h5py-3.16.0-",
        "threadpoolctl-3.6.0-",
    }
    for fragment in required_fragments:
        assert sum(fragment in url for url in urls) == 1


def test_manifest_rejects_duplicate_json_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text(
        '{"schema_version":1,"schema_version":1}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate JSON key"):
        load_input_manifest(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("path", "../outside", "relative"),
        ("path", "/absolute", "relative"),
        ("sha256", "not-a-hash", "SHA256"),
    ],
)
def test_manifest_rejects_unsafe_paths_and_bad_hashes(
    tmp_path,
    field: str,
    value: str,
    message: str,
) -> None:
    _, manifest_path = _write_source_and_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["source"]["files"]["forces_fc3"][field] = value
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_input_manifest(manifest_path)


def test_hash_or_commit_failure_occurs_before_output_creation(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    (source_dir / "FORCES_FC3").write_text("changed\n", encoding="utf-8")
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="SHA256"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=_force_only_config(),
            runtime=_FakeRuntime(),
            commit_resolver=lambda _: PINNED_COMMIT,
        )

    assert not output_dir.exists()


def test_exact_runtime_version_failure_precedes_output_creation(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"

    with pytest.raises(RuntimeError, match="Version mismatch"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=_force_only_config(),
            runtime=_FakeRuntime(phono3py_version="4.4.1"),
            commit_resolver=lambda _: PINNED_COMMIT,
        )

    assert not output_dir.exists()


def test_exact_source_commit_failure_precedes_output_creation(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"

    with pytest.raises(ValueError, match="commit mismatch"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=_force_only_config(),
            runtime=_FakeRuntime(),
            commit_resolver=lambda _: "b" * 40,
        )

    assert not output_dir.exists()


def test_force_constants_only_stage_has_strict_schema_and_hashes(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    runtime = _FakeRuntime()

    result = run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=_force_only_config(),
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )

    assert result.stage == "force_constants"
    assert runtime.force_calls == 1
    assert runtime.reference_calls == 0
    for name in FORCE_CONSTANT_FILENAMES:
        assert (output_dir / name).is_file()
    for name in (
        "input-manifest.normalized.json",
        "source-verification.json",
        "force-constants-settings.json",
        "force-constants-environment.json",
        "force-constants-stage.json",
        "force-constants-inventory.json",
        "force-constants-SHA256SUMS",
    ):
        assert (output_dir / name).is_file()
    assert not (output_dir / REFERENCE_FILENAME).exists()
    assert verify_sha256sums(
        output_dir / "force-constants-SHA256SUMS",
        root=output_dir,
    )


def test_unknown_nonempty_output_directory_is_refused(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "unrelated.txt").write_text("do not overwrite\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=_reference_config(),
            runtime=_FakeRuntime(),
            commit_resolver=lambda _: PINNED_COMMIT,
        )

    assert (output_dir / "unrelated.txt").read_text(encoding="utf-8") == (
        "do not overwrite\n"
    )


def test_fresh_full_run_writes_reference_inventory_and_sha256sums(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    runtime = _FakeRuntime()

    result = run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=_reference_config(),
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )

    assert result.stage == "reference"
    assert runtime.force_calls == 1
    assert runtime.reference_calls == 1
    assert (output_dir / REFERENCE_FILENAME).is_file()
    for name in (
        "reference-settings.json",
        "reference-environment.json",
        "reference-stage.json",
        "inventory.json",
        "SHA256SUMS",
    ):
        assert (output_dir / name).is_file()
    assert verify_sha256sums(output_dir / "SHA256SUMS", root=output_dir)

    settings = json.loads(
        (output_dir / "reference-settings.json").read_text(encoding="utf-8")
    )
    assert settings["mesh"] == [3, 3, 3]
    assert settings["grid_points"] == [0, 2]
    assert settings["band_indices"] == [0, 2, 5]
    assert settings["temperatures_K"] == [0.0, 300.0]
    assert settings["sigma_THz"] == 0.1
    assert settings["is_mesh_symmetry"] is False
    assert settings["make_r0_average"] is True
    assert settings["gamma_units"] == "cyclic_THz_HWHM"


def test_force_stage_can_resume_full_stage_without_overwriting_fc_files(
    tmp_path,
) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    runtime = _FakeRuntime()
    run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=_force_only_config(),
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )
    original_hashes = {
        name: _sha256(output_dir / name) for name in FORCE_CONSTANT_FILENAMES
    }

    result = run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=_reference_config(),
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )

    assert result.stage == "reference"
    assert runtime.force_calls == 1
    assert runtime.reference_calls == 1
    assert {
        name: _sha256(output_dir / name) for name in FORCE_CONSTANT_FILENAMES
    } == original_hashes


def test_corrupt_force_stage_is_rejected_instead_of_regenerated(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    runtime = _FakeRuntime()
    run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=_force_only_config(),
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )
    (output_dir / "fc3.compact.hdf5").write_bytes(b"corrupt\n")

    with pytest.raises(ValueError, match="SHA256"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=_reference_config(),
            runtime=runtime,
            commit_resolver=lambda _: PINNED_COMMIT,
        )

    assert runtime.force_calls == 1
    assert runtime.reference_calls == 0


def test_completed_reference_run_is_not_silently_overwritten(tmp_path) -> None:
    source_dir, manifest_path = _write_source_and_manifest(tmp_path)
    output_dir = tmp_path / "output"
    config = _reference_config()
    runtime = _FakeRuntime()
    run_generator(
        manifest_path=manifest_path,
        source_dir=source_dir,
        output_dir=output_dir,
        config=config,
        runtime=runtime,
        commit_resolver=lambda _: PINNED_COMMIT,
    )
    reference_hash = _sha256(output_dir / REFERENCE_FILENAME)

    with pytest.raises(FileExistsError, match="reference"):
        run_generator(
            manifest_path=manifest_path,
            source_dir=source_dir,
            output_dir=output_dir,
            config=config,
            runtime=runtime,
            commit_resolver=lambda _: PINNED_COMMIT,
        )

    assert _sha256(output_dir / REFERENCE_FILENAME) == reference_hash


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mesh": None, "force_constants_only": False},
        {"mesh": (0, 3, 3)},
        {"mesh": (129, 1, 1)},
        {"mesh": (64, 64, 65)},
        {"mesh": (3, 3, 3), "grid_points": (27,)},
        {"mesh": (3, 3, 3), "grid_points": (0, 0)},
        {"mesh": (3, 3, 3), "band_indices": (-1,)},
        {"mesh": (3, 3, 3), "temperatures": (-1.0,)},
        {"mesh": (3, 3, 3), "temperatures": (10_001.0,)},
        {"mesh": (3, 3, 3), "sigma": 0.0},
        {"mesh": (3, 3, 3), "sigma": 101.0},
        {"mesh": (3, 3, 3), "sigma_cutoff": 101.0},
        {"mesh": (3, 3, 3), "backend": "Python"},
    ],
)
def test_generator_config_rejects_unbounded_or_ambiguous_requests(kwargs) -> None:
    with pytest.raises((TypeError, ValueError)):
        GeneratorConfig(**kwargs)


def test_config_canonicalizes_selection_order_deterministically() -> None:
    config = GeneratorConfig(
        mesh=[3, 3, 3],
        grid_points=[7, 0, 3],
        band_indices=[5, 1, 2],
        temperatures=[300, 0, 100],
        sigma=0.2,
    )

    assert config.mesh == (3, 3, 3)
    assert config.grid_points == (0, 3, 7)
    assert config.band_indices == (1, 2, 5)
    assert config.temperatures == (0.0, 100.0, 300.0)


def test_runtime_converts_grg_targets_to_bzg_before_phono3py_calls(
    tmp_path,
    monkeypatch,
) -> None:
    grg2bzg = np.array([4, 0, 5, 1, 3, 2], dtype=np.int64)
    interaction = SimpleNamespace(
        bz_grid=SimpleNamespace(grg2bzg=grg2bzg),
        cutoff_frequency=1.0e-4,
    )

    class _FakeDynamicalMatrix:
        def __init__(self) -> None:
            self.dynamical_matrix = None

        def run(self, qpoint) -> None:
            scale = 1.0 + float(np.dot(qpoint, qpoint))
            self.dynamical_matrix = np.eye(2, dtype=complex) * scale

    class _FakePhono3py:
        calculator = "vasp"
        unit_conversion_factor = 15.633302300230191

        def __init__(self) -> None:
            self.phph_interaction = interaction
            self.gamma_grid_points = None
            self.init_options = None
            self.dynamical_matrix = _FakeDynamicalMatrix()

        def init_phph_interaction(self, **kwargs) -> None:
            self.init_options = kwargs

        def run_phonon_solver(self) -> None:
            pass

        def get_phonon_data(self):
            return (
                np.ones((6, 2), dtype=float),
                np.zeros((6, 2, 2), dtype=complex),
                np.zeros((6, 3), dtype=np.int64),
            )

        def run_imag_self_energy(self, grid_points, **kwargs):
            self.gamma_grid_points = np.array(grid_points, copy=True)
            return SimpleNamespace(
                gammas=np.zeros((1, 1, 3, 2), dtype=float),
                frequency_points=None,
            )

    fake_ph3 = _FakePhono3py()
    runtime = _Phono3pyRuntime.__new__(_Phono3pyRuntime)
    runtime.np = np
    runtime.phono3py_version = "4.4.0"
    runtime.phonopy_version = "4.4.0"
    archived_targets = {}
    monkeypatch.setattr(
        runtime,
        "_load_from_force_constants",
        lambda **_: fake_ph3,
    )
    monkeypatch.setattr(
        runtime,
        "_publish_hdf5",
        lambda target, writer: writer(target),
    )

    def _write_reference(path, **kwargs):
        path.write_bytes(b"mock hdf5\n")
        archived_targets["grg"] = np.array(kwargs["grg_targets"], copy=True)
        archived_targets["bzg"] = np.array(kwargs["bzg_targets"], copy=True)
        archived_targets["random_q"] = np.array(
            kwargs["fc2_random_qpoints"],
            copy=True,
        )
        archived_targets["random_dm"] = np.array(
            kwargs["fc2_random_dynamical_matrices"],
            copy=True,
        )
        return {"number_of_mesh_points": 6}

    monkeypatch.setattr(runtime, "_write_reference_hdf5", _write_reference)
    config = GeneratorConfig(
        mesh=(2, 3, 1),
        grid_points=(0, 2, 5),
        temperatures=(300.0,),
        sigma=0.1,
    )

    metadata = runtime.generate_reference(
        output_dir=tmp_path,
        source_files={},
        config=config,
    )

    np.testing.assert_array_equal(fake_ph3.gamma_grid_points, [4, 5, 2])
    np.testing.assert_array_equal(archived_targets["grg"], [0, 2, 5])
    np.testing.assert_array_equal(archived_targets["bzg"], [4, 5, 2])
    np.testing.assert_array_equal(
        archived_targets["random_q"],
        np.asarray(FC2_RANDOM_QPOINTS),
    )
    assert archived_targets["random_dm"].shape == (8, 2, 2)
    assert metadata["grid_points_grg"] == [0, 2, 5]
    assert metadata["grid_points_bzg"] == [4, 5, 2]
    assert fake_ph3.init_options["openmp_per_triplets"] is False
