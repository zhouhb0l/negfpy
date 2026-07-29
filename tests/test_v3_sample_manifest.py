# ruff: noqa: E402

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

pytest.importorskip("spglib")

from negfpy.anharmonic.momentum import (
    UniformBZMesh,
    niggli_reduced_reciprocal_lattice,
    shortest_bz_address_images,
)
from negfpy.anharmonic.phono3py_reference import (
    read_phono3py_silicon_reference,
)
from negfpy.anharmonic.v3_sample_manifest import (
    LARGE_MESH_BOUNDED_SELECTION,
    V3SampleCoverageError,
    _BoundedSelectionAccumulator,
    _PromotedBlock,
    _coordinate_sha256,
    _production_process,
    _select_bounded_blocks,
    build_v3_sample_manifest,
    create_v3_sample_manifest,
    read_v3_sample_manifest,
    verify_v3_sample_manifest_identity,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_reference(tmp_path: Path) -> Path:
    path = tmp_path / "reference.hdf5"
    mesh_shape = (3, 3, 3)
    mesh = UniformBZMesh(mesh_shape)
    n_q = mesh.n_qpoints
    n_band = 6
    targets_grg = np.asarray([0, 1, 13], dtype=np.int64)
    maps = np.arange(n_q, dtype=np.int64)
    addresses = np.asarray(
        [address.components for address in mesh.addresses],
        dtype=np.int64,
    )
    frequencies = (
        np.arange(1, n_band + 1, dtype=np.float64)[None, :]
        + 0.01 * np.arange(n_q, dtype=np.float64)[:, None]
    )

    with h5py.File(path, "w") as h5:
        h5.attrs["schema_name"] = "negfpy_phono3py_si_reference"
        h5.attrs["schema_version"] = 1
        h5.attrs["phono3py_version"] = "4.4.0"
        h5.attrs["phonopy_version"] = "4.4.0"
        h5.attrs["frequency_units"] = "cyclic_THz"
        h5.attrs["gamma_units"] = "cyclic_THz_HWHM"
        h5.attrs["interaction_strength_units"] = "eV^2"
        h5.attrs["is_mesh_symmetry"] = False
        h5.attrs["is_nac"] = False
        h5.attrs["make_r0_average"] = True
        h5.attrs["cutoff_frequency_THz"] = 1.0e-4
        h5.attrs["frequency_unit_conversion_factor"] = 15.633302300230191
        h5.attrs["symmetrize_fc3q"] = False

        settings = h5.create_group("settings")
        settings.create_dataset("mesh", data=mesh_shape)
        settings.create_dataset("grid_points_grg", data=targets_grg)
        settings.create_dataset("grid_points_bzg", data=targets_grg)
        settings.create_dataset(
            "band_indices",
            data=np.arange(n_band, dtype=np.int64),
        )
        settings.create_dataset("temperatures", data=[300.0])
        settings.create_dataset("sigmas", data=[0.1])
        settings.attrs["sigma_cutoff"] = "None"

        grid = h5.create_group("grid")
        grid.attrs["GRG_index_order"] = "x_fast"
        grid.attrs["target_grid_point_input_space"] = "GRG"
        grid.attrs["triplet_grid_point_space"] = "BZG"
        grid.create_dataset("D_diag", data=mesh_shape)
        grid.create_dataset("Q", data=np.eye(3, dtype=np.int64))
        grid.create_dataset("PS", data=np.zeros(3, dtype=np.int64))
        grid.create_dataset("grg2bzg", data=maps)
        grid.create_dataset("bzg2grg", data=maps)
        grid.create_dataset("addresses", data=addresses)

        phonons = h5.create_group("phonons")
        frequency_dataset = phonons.create_dataset(
            "frequencies",
            data=frequencies,
        )
        frequency_dataset.attrs["units"] = "cyclic_THz"
        phonons.create_dataset(
            "eigenvectors",
            data=np.repeat(
                np.eye(n_band, dtype=np.complex128)[None, :, :],
                n_q,
                axis=0,
            ),
        )
        phonons.create_dataset("bz_grid_addresses", data=addresses)
        target_frequencies = phonons.create_dataset(
            "target_band_frequencies",
            data=frequencies[targets_grg],
        )
        target_frequencies.attrs["units"] = "cyclic_THz"

        gamma = h5.create_group("gamma").create_dataset(
            "values",
            data=np.zeros((1, 1, targets_grg.size, n_band)),
        )
        gamma.attrs["units"] = "cyclic_THz_HWHM"
        gamma.attrs["axis_order"] = (
            "sigma,temperature,target_grid_point,target_band"
        )

        target_root = h5.create_group("target_q")
        for target_ordinal, q0 in enumerate(targets_grg):
            routes = mesh.triplets_at_q(int(q0))
            triplets = np.asarray(
                [[route.q0_index, route.q1_index, route.q2_index] for route in routes],
                dtype=np.int64,
            )
            triplet_addresses = addresses[triplets]
            target = target_root.create_group(f"grg_{int(q0):08d}")
            target.attrs["grid_point_grg"] = int(q0)
            target.attrs["grid_point_bzg"] = int(q0)
            target.attrs["Nq"] = n_q
            target.create_dataset("triplets_bzg", data=triplets)
            target.create_dataset(
                "triplet_addresses",
                data=triplet_addresses,
            )
            target.create_dataset(
                "triplet_weights",
                data=np.ones(n_q, dtype=np.int64),
            )
            target.create_dataset(
                "triplets_map_grg",
                data=np.arange(n_q, dtype=np.int64),
            )
            target.create_dataset(
                "ir_map_grg",
                data=np.arange(n_q, dtype=np.int64),
            )

            route_axis = np.arange(n_q, dtype=np.float64)[:, None, None, None]
            b0_axis = np.arange(n_band, dtype=np.float64)[None, :, None, None]
            b1_axis = np.arange(n_band, dtype=np.float64)[None, None, :, None]
            b2_axis = np.arange(n_band, dtype=np.float64)[None, None, None, :]
            strengths = (
                1.0
                + 1000.0 * target_ordinal
                + 10.0 * route_axis
                + 1.0e-1 * b0_axis
                + 1.0e-2 * b1_axis
                + 1.0e-3 * b2_axis
            )
            strength = target.create_dataset(
                "interaction_strength",
                data=strengths,
            )
            strength.attrs["units"] = "eV^2"
            strength.attrs["axis_order"] = (
                "triplet,target_band,band1,band2"
            )
            integration = target.create_dataset(
                "integration_weights",
                data=np.zeros((2, n_q, n_band, n_band, n_band)),
            )
            integration.attrs["units"] = "1/cyclic_THz"
            integration.attrs["axis_order"] = (
                "channel,triplet,target_frequency,band1,band2"
            )
            integration.attrs["channel_0"] = (
                "Gaussian delta(omega-f1-f2): decay/sum channel"
            )
            integration.attrs["channel_1"] = (
                "Gaussian delta(omega+f1-f2) minus "
                "delta(omega-f1+f2): signed difference channel"
            )
            phase = target.create_dataset(
                "derived_phase_space_channel_sums",
                data=np.zeros((2, n_band)),
            )
            phase.attrs["units"] = "1/cyclic_THz"
            phase.attrs["axis_order"] = "channel,target_frequency"
            phase.attrs["cutoff_frequency_THz"] = 1.0e-4
            phase.attrs["cutoff_rule"] = (
                "strict_all_three_frequencies_above_cutoff"
            )

    (tmp_path / "SHA256SUMS").write_text(
        f"{_sha256(path)}  {path.name}\n",
        encoding="utf-8",
    )
    return path


def _mapping(
    *,
    purpose: str = "fc2",
    primitive_lattice: np.ndarray | None = None,
) -> SimpleNamespace:
    lattice = (
        np.eye(3)
        if primitive_lattice is None
        else np.asarray(primitive_lattice, dtype=float)
    )
    primitive = SimpleNamespace(
        lattice=lattice,
        scaled_positions=np.asarray([[0.0, 0.0, 0.0]]),
        masses=np.asarray([28.0]),
        symbols=("Si",),
    )
    supercell = SimpleNamespace(
        lattice=3.0 * lattice,
        scaled_positions=np.asarray([[0.0, 0.0, 0.0]]),
        masses=np.asarray([28.0]),
        symbols=("Si",),
    )
    return SimpleNamespace(
        purpose=purpose,
        primitive_matrix=np.eye(3),
        supercell_matrix=3 * np.eye(3, dtype=np.int64),
        primitive=primitive,
        supercell=supercell,
        p2s=np.asarray([0], dtype=np.int64),
        s2p=np.asarray([0], dtype=np.int64),
        p2p=np.asarray([0], dtype=np.int64),
        shortest_vectors=np.zeros((1, 1, 1, 3)),
        multiplicity=np.ones((1, 1), dtype=np.int64),
        atomic_permutations=None,
        translation_permutations=np.asarray([[0]], dtype=np.int64),
    )


def _inputs(tmp_path: Path) -> dict[str, object]:
    paths: dict[str, Path] = {}
    for name in (
        "fc2-compact.hdf5",
        "fc2-full.hdf5",
        "fc3-compact.hdf5",
        "fc3-full.hdf5",
        "phono3py.yaml",
        "input-manifest.json",
    ):
        path = tmp_path / name
        path.write_bytes(f"immutable:{name}".encode())
        paths[name] = path
    return {
        "fc2_compact_hdf5_path": paths["fc2-compact.hdf5"],
        "fc2_full_hdf5_path": paths["fc2-full.hdf5"],
        "fc3_compact_hdf5_path": paths["fc3-compact.hdf5"],
        "fc3_full_hdf5_path": paths["fc3-full.hdf5"],
        "phono3py_yaml_path": paths["phono3py.yaml"],
        "input_manifest_path": paths["input-manifest.json"],
        "fc2_mapping": _mapping(purpose="fc2"),
        "fc3_mapping": _mapping(purpose="fc3"),
    }


def test_selection_is_deterministic_stratified_and_nonzero(
    tmp_path: Path,
) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path),
    )
    inputs = _inputs(tmp_path)
    first = build_v3_sample_manifest(reference, **inputs)
    second = build_v3_sample_manifest(reference, **inputs)

    assert first == second
    canonical = json.dumps(
        first,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    # Locks the legacy default payload, not merely repeatability after a
    # selector refactor.
    assert hashlib.sha256(canonical).hexdigest() == (
        "c16867a2710f73ec03d6bccf17be52dab"
        "470f4bf888861e3d0cd060f3439b87b"
    )
    assert len(first["samples"]) >= 64
    keys = {
        (
            item["q0_grg"],
            item["q1_grg"],
            tuple(item["target_subspace"]),
            tuple(item["band1_subspace"]),
            tuple(item["band2_subspace"]),
        )
        for item in first["samples"]
    }
    assert len(keys) == len(first["samples"])
    assert {item["process"] for item in first["samples"]} == {
        "normal",
        "umklapp",
    }
    assert {item["q0_grg"] for item in first["samples"]} == {0, 1, 13}
    assert {item["b0"] for item in first["samples"]} == set(range(6))
    coverage = first["coverage"]
    assert not set(coverage["required_strata"]) & set(
        coverage["empty_strata"]
    )
    assert len(coverage["required_strata"]) + len(
        coverage["empty_strata"]
    ) == 36
    for stratum in coverage["required_strata"]:
        assert sum(
            item["stratum"] == stratum for item in first["samples"]
        ) >= 2
    strengths = [
        item["reference_invariant_block_sum_ev2"]
        for item in first["samples"]
    ]
    assert max(strengths) == pytest.approx(2261.555)
    assert first["robust_nonzero_floor_ev2"] == pytest.approx(
        2261.555e-12
    )
    mesh = UniformBZMesh((3, 3, 3))
    lattice = np.eye(3)
    reduced = niggli_reduced_reciprocal_lattice(lattice)
    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=reduced,
    )
    population: dict[str, list[float]] = {"normal": [], "umklapp": []}
    for target_ordinal, q0 in enumerate((0, 1, 13)):
        for route in mesh.triplets_at_q(q0):
            process = _production_process(
                route,
                address_images=images,
                primitive_lattice=lattice,
                reduced_reciprocal_lattice=reduced,
            )
            for b0 in range(6):
                for b1 in range(6):
                    for b2 in range(6):
                        population[process].append(
                            1.0
                            + 1000.0 * target_ordinal
                            + 10.0 * route.q1_index
                            + 1.0e-1 * b0
                            + 1.0e-2 * b1
                            + 1.0e-3 * b2
                        )
    for process, values in population.items():
        selected_strengths = {
            item["reference_invariant_block_sum_ev2"]
            for item in first["samples"]
            if item["process"] == process
        }
        ordered = sorted(values)
        for quantile in (0.1, 0.5, 0.9):
            position = int(
                np.floor(quantile * (len(ordered) - 1) + 0.5)
            )
            assert ordered[position] in selected_strengths


def test_large_mesh_bounded_selection_is_strict_and_deterministic(
    tmp_path: Path,
) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path),
    )
    inputs = _inputs(tmp_path)
    first = build_v3_sample_manifest(
        reference,
        selection_mode=LARGE_MESH_BOUNDED_SELECTION,
        maximum_promoted_blocks=96,
        **inputs,
    )
    second = build_v3_sample_manifest(
        reference,
        selection_mode=LARGE_MESH_BOUNDED_SELECTION,
        maximum_promoted_blocks=96,
        **inputs,
    )

    assert first == second
    assert first["selection_algorithm"] == (
        "deterministic_bounded_streaming_reference_invariant_blocks"
    )
    assert first["selection_algorithm_version"] == 2
    assert first["coverage"]["cover_all_target_grid_points"] is False
    assert first["coverage"]["cover_all_target_bands"] is True
    assert len(first["samples"]) == 96
    assert len(first["samples"]) <= 512
    assert {sample["process"] for sample in first["samples"]} == {
        "normal",
        "umklapp",
    }
    assert {sample["b0"] for sample in first["samples"]} == set(range(6))
    for stratum in first["coverage"]["required_strata"]:
        assert sum(
            sample["stratum"] == stratum for sample in first["samples"]
        ) >= 2
    assert max(
        sample["reference_invariant_block_sum_ev2"]
        for sample in first["samples"]
    ) == pytest.approx(2261.555)
    assert all(
        sample["reference_invariant_block_sum_ev2"]
        >= first["robust_nonzero_floor_ev2"]
        for sample in first["samples"]
    )

    output = tmp_path / "bounded-v3-samples.json"
    manifest = create_v3_sample_manifest(
        reference,
        output_path=output,
        selection_mode=LARGE_MESH_BOUNDED_SELECTION,
        maximum_promoted_blocks=96,
        **inputs,
    )
    assert manifest.cover_all_target_grid_points is False
    assert manifest.cover_all_target_bands is True
    assert manifest.selection_algorithm_version == 2
    verify_v3_sample_manifest_identity(manifest, reference, **inputs)

    wrong_contract = json.loads(output.read_text(encoding="utf-8"))
    wrong_contract["coverage"]["cover_all_target_grid_points"] = True
    wrong_path = tmp_path / "wrong-v2-contract.json"
    wrong_path.write_text(
        json.dumps(wrong_contract, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage booleans"):
        read_v3_sample_manifest(wrong_path)

    wrong_pair = json.loads(output.read_text(encoding="utf-8"))
    wrong_pair["selection_algorithm_version"] = 1
    wrong_pair_path = tmp_path / "wrong-v2-pair.json"
    wrong_pair_path.write_text(
        json.dumps(wrong_pair, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="algorithm/version pair"):
        read_v3_sample_manifest(wrong_pair_path)

    for invalid_version in (True, 2.0):
        wrong_type = json.loads(output.read_text(encoding="utf-8"))
        wrong_type["selection_algorithm_version"] = invalid_version
        wrong_type_path = tmp_path / (
            "wrong-v2-version-type-"
            f"{type(invalid_version).__name__}.json"
        )
        wrong_type_path.write_text(
            json.dumps(wrong_type, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="version types"):
            read_v3_sample_manifest(wrong_type_path)

    weakened_processes = json.loads(output.read_text(encoding="utf-8"))
    weakened_processes["coverage"]["required_processes"] = ["normal"]
    weakened_processes_path = tmp_path / "weakened-v2-processes.json"
    weakened_processes_path.write_text(
        json.dumps(weakened_processes, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="canonical normal/umklapp"):
        read_v3_sample_manifest(weakened_processes_path)

    incomplete_partition = json.loads(output.read_text(encoding="utf-8"))
    incomplete_partition["coverage"]["required_strata"].pop()
    incomplete_partition_path = tmp_path / "incomplete-v2-partition.json"
    incomplete_partition_path.write_text(
        json.dumps(incomplete_partition, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="36-stratum"):
        read_v3_sample_manifest(incomplete_partition_path)

    sparse_stratum = json.loads(output.read_text(encoding="utf-8"))
    counts = {
        stratum: sum(
            sample["stratum"] == stratum
            for sample in sparse_stratum["samples"]
        )
        for stratum in sparse_stratum["coverage"]["required_strata"]
    }
    target_stratum = next(
        stratum for stratum, count in counts.items() if count == 2
    )
    samples = list(sparse_stratum["samples"])
    remove_index = next(
        index
        for index, sample in enumerate(samples)
        if sample["stratum"] == target_stratum
    )
    del samples[remove_index]
    sparse_stratum["samples"] = samples
    sparse_stratum_path = tmp_path / "sparse-v2-stratum.json"
    sparse_stratum_path.write_text(
        json.dumps(sparse_stratum, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="two distinct promoted blocks"):
        read_v3_sample_manifest(sparse_stratum_path)

    legacy = build_v3_sample_manifest(reference, **inputs)
    legacy["coverage"]["cover_all_target_grid_points"] = False
    weakened_legacy_path = tmp_path / "weakened-v1-contract.json"
    weakened_legacy_path.write_text(
        json.dumps(legacy, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="coverage booleans"):
        read_v3_sample_manifest(weakened_legacy_path)


def test_bounded_accumulator_retention_does_not_scale_with_1331_qpoints() -> None:
    target_bands = tuple(range(6))
    accumulator = _BoundedSelectionAccumulator(
        floor=1.0e-6,
        global_maximum=1.0e6,
        maximum_promoted_blocks=512,
        target_bands=target_bands,
    )
    strata = tuple(
        f"{region}.{target}.{internal}.{process}"
        for region in ("gamma", "near", "generic")
        for target in ("acoustic", "optical")
        for internal in ("aa", "ao", "oo")
        for process in ("normal", "umklapp")
    )
    maximum_retained = 0
    population = 1331 * 64
    for index in range(population):
        stratum = strata[index % len(strata)]
        _, target, internal, process = stratum.split(".")
        target_band = (
            index % 3 if target == "acoustic" else 3 + index % 3
        )
        band1, band2 = {
            "aa": (0, 1),
            "ao": (0, 3),
            "oo": (3, 4),
        }[internal]
        q0 = index % 1331
        q1 = index
        q2 = (q0 + q1) % 1331
        strength = 10.0 ** (-6.0 + 12.0 * ((index % 1001) / 1000.0))
        if index == population - 1:
            strength = 1.0e6
        target_subspace = (target_band,)
        band1_subspace = (band1,)
        band2_subspace = (band2,)
        accumulator.add(
            _PromotedBlock(
                q0=q0,
                q1=q1,
                q2=q2,
                target_subspace=target_subspace,
                band1_subspace=band1_subspace,
                band2_subspace=band2_subspace,
                seed_b0=target_band,
                seed_b1=band1,
                seed_b2=band2,
                process=process,
                strength_ev2=strength,
                stratum=stratum,
                coordinate_sha256=_coordinate_sha256(
                    q0,
                    q1,
                    q2,
                    target_subspace,
                    band1_subspace,
                    band2_subspace,
                ),
            )
        )
        maximum_retained = max(
            maximum_retained,
            accumulator.retained_candidate_slots,
        )

    selected, required_strata, empty_strata = _select_bounded_blocks(
        accumulator,
        minimum_promoted_blocks=64,
    )
    assert accumulator.viable_count == population
    assert maximum_retained <= accumulator.retention_bound
    assert accumulator.retention_bound < population // 100
    assert len(selected) == 512
    assert len({sample.key for sample in selected}) == 512
    assert len({sample.q0 for sample in selected}) < 1331
    assert {sample.process for sample in selected} == {
        "normal",
        "umklapp",
    }
    assert {sample.seed_b0 for sample in selected} == set(target_bands)
    assert {sample.stratum for sample in selected} == set(strata)
    assert set(required_strata) == set(strata)
    assert empty_strata == ()
    assert max(sample.strength_ev2 for sample in selected) == 1.0e6
    for process in ("normal", "umklapp"):
        process_strengths = [
            sample.strength_ev2
            for sample in selected
            if sample.process == process
        ]
        assert min(process_strengths) <= 1.1e-6
        assert max(process_strengths) >= 0.9e6


def test_atomic_roundtrip_hash_tamper_and_input_mismatch(
    tmp_path: Path,
) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path),
    )
    inputs = _inputs(tmp_path)
    output = tmp_path / "v3-samples.json"
    manifest = create_v3_sample_manifest(
        reference,
        output_path=output,
        **inputs,
    )
    assert manifest.sha256 == _sha256(output)
    assert read_v3_sample_manifest(
        output,
        expected_payload_sha256=manifest.sha256,
    ) == manifest
    verify_v3_sample_manifest_identity(
        manifest,
        reference,
        **inputs,
    )
    with pytest.raises(FileExistsError):
        create_v3_sample_manifest(
            reference,
            output_path=output,
            **inputs,
        )

    tampered = tmp_path / "tampered.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["samples"][0]["reference_invariant_block_sum_ev2"] *= 1.01
    tampered.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="expected_payload_sha256"):
        read_v3_sample_manifest(
            tampered,
            expected_payload_sha256=manifest.sha256,
        )

    Path(inputs["fc3_full_hdf5_path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="input hashes"):
        verify_v3_sample_manifest_identity(
            manifest,
            reference,
            **inputs,
        )


def test_insufficient_reference_coverage_is_explicit(tmp_path: Path) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path),
    )
    with pytest.raises(V3SampleCoverageError, match="at least 100000"):
        build_v3_sample_manifest(
            reference,
            minimum_promoted_blocks=100_000,
            **_inputs(tmp_path),
        )


def test_surface_tie_uses_production_normal_classification() -> None:
    mesh = UniformBZMesh((2, 1, 1))
    lattice = np.eye(3)
    reduced = niggli_reduced_reciprocal_lattice(lattice)
    images = shortest_bz_address_images(
        mesh,
        lattice,
        reduced_reciprocal_lattice=reduced,
    )
    boundary = mesh.triplets_at_q(0)[1]

    # The canonical residue representatives sum to a reciprocal lattice
    # vector, but retained +/- boundary images contain a zero-G realization.
    assert not boundary.is_normal
    assert (
        _production_process(
            boundary,
            address_images=images,
            primitive_lattice=lattice,
            reduced_reciprocal_lattice=reduced,
        )
        == "normal"
    )


def test_target_region_uses_skew_cartesian_reciprocal_norm(
    tmp_path: Path,
) -> None:
    reference = read_phono3py_silicon_reference(
        _write_reference(tmp_path),
    )
    skew = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.95, 0.2, 0.0],
            [0.9, 0.1, 0.15],
        ]
    )
    inputs = _inputs(tmp_path)
    inputs["fc2_mapping"] = _mapping(
        purpose="fc2",
        primitive_lattice=skew,
    )
    inputs["fc3_mapping"] = _mapping(
        purpose="fc3",
        primitive_lattice=skew,
    )
    payload = build_v3_sample_manifest(reference, **inputs)
    regions_by_target = {
        item["q0_grg"]: item["stratum"].split(".", 1)[0]
        for item in payload["samples"]
    }

    # Fractional-coordinate Euclidean norms would label q=1 near and q=13
    # generic. The skew reciprocal metric reverses those two shells.
    assert regions_by_target[1] == "generic"
    assert regions_by_target[13] == "near"
