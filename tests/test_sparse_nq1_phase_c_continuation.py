from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import CubicSCBAProblem, HarmonicReservoir, SignedFrequencyGrid
from negfpy.anharmonic._born_initial import write_born_initial_state
from negfpy.anharmonic._checkpoint_v2 import (
    checkpoint_initial_sigma_v2,
    load_scba_checkpoint_v2_self_validating,
    solve_sparse_scba_checkpointed,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("validate_sparse_nq1_phase_c_scba")
mpi_driver = importlib.import_module("run_material_nq1_release_mpi")
strict_validation = importlib.import_module("validate_material_nq1_release")

SOURCE_TREE = "a" * 64
ENVIRONMENT = "b" * 64


def _identity() -> dict[str, object]:
    return {
        "fixture": {"fixture_manifest_sha256": "1" * 64},
        "fc3_full": {"raw_sha256": "2" * 64},
        "phase_b": {
            "artifact_set_sha256": "3" * 64,
            "archive": {"sha256": "4" * 64},
            "campaign_manifest_sha256": "5" * 64,
        },
    }


def _problem(cubic: float) -> CubicSCBAProblem:
    device, lead, vertex = pinned_scalar_chain(1, pinning=1.0, cubic=cubic)
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=SparseCubicVertex.from_dense(vertex),
        grid=SignedFrequencyGrid(step=0.2, n_positive=2),
    )


@pytest.fixture(autouse=True)
def _reconstruct_toy_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    def reconstruct(manifest):
        controls = manifest["numerical_controls"]
        parent_problem = _problem(float(manifest["coupling_scale"]) * 0.1)
        bias = float(controls["temperature_bias"])
        if bias != float(
            parent_problem.left.temperature - parent_problem.right.temperature
        ):
            mean = 0.7
            parent_problem = replace(
                parent_problem,
                left=replace(parent_problem.left, temperature=mean + bias / 2.0),
                right=replace(parent_problem.right, temperature=mean - bias / 2.0),
            )
        parent_config = driver._config(
            eta_lead=controls["eta_lead"],
            mixing=controls["mixing"],
            mixing_floor=controls["mixing_floor"],
        )
        reconstructed_identity = dict(manifest["preflight"]["source"])
        reconstructed_identity["problem_input_hashes"] = (
            driver.problem_input_hashes(parent_problem)
        )
        return reconstructed_identity, parent_problem, parent_config

    monkeypatch.setattr(driver, "_reconstruct_parent_inputs", reconstruct)


def _write_parent(
    root: Path,
    *,
    parent_scale: float = 0.01,
    converged: bool = True,
) -> tuple[CubicSCBAProblem, object]:
    root.mkdir()
    checkpoint = root / "checkpoint" / "sigma-pp.h5"
    config = driver._config()
    if not converged:
        config = replace(
            config,
            max_iterations=1,
            min_iterations=1,
            consecutive_passes=1,
            on_nonconvergence="return",
        )
    problem = _problem(parent_scale * 0.1)
    solve_sparse_scba_checkpointed(
        problem,
        config,
        checkpoint_path=checkpoint,
        run_id="parent",
        environment_sha256=ENVIRONMENT,
        source_tree_sha256=SOURCE_TREE,
        source_hashes=driver._source_hashes(_identity()),
    )
    numerical_controls = {
        "grid_step": float(problem.grid.step),
        "grid_n_positive": int(problem.grid.n_positive),
        "grid_cutoff": float(problem.grid.cutoff),
        "eta_lead": float(config.eta_lead),
        "mixing": float(config.mixing),
        "mixing_floor": float(config.mixing_floor),
        "temperature_bias": float(
            problem.left.temperature - problem.right.temperature
        ),
    }
    outputs = {
        "checkpoint": str(checkpoint.resolve()),
        "json": str((root / "outcomes/scba.json").resolve()),
        "hdf5": str((root / "outcomes/scba-fields.h5").resolve()),
    }
    source = {
        **_identity(),
        "fixture_path": str((root / "inputs/fixture").resolve()),
        "fc3_full": {
            **_identity()["fc3_full"],
            "path": str((root / "inputs/fc3.h5").resolve()),
        },
        "phase_b": {
            **_identity()["phase_b"],
            "root": str((root / "inputs/phase-b").resolve()),
        },
        "problem_input_hashes": driver.problem_input_hashes(problem),
    }
    checkpoint_state = load_scba_checkpoint_v2_self_validating(checkpoint)
    preflight = {
        "schema": driver.PREFLIGHT_SCHEMA,
        "source": source,
        "run_id": "parent",
        "coupling_scale": parent_scale,
        "continuation_parent": None,
        "born_initial": None,
        "source_tree_sha256": SOURCE_TREE,
        "environment_sha256": ENVIRONMENT,
        "configuration_sha256": driver.configuration_hash(config),
        "checkpoint_identity_sha256": checkpoint_state.identity.identity_sha256,
        "checkpoint_schema": driver.CHECKPOINT_SCHEMA,
        "checkpoint_path": outputs["checkpoint"],
        "outputs": {"json": outputs["json"], "hdf5": outputs["hdf5"]},
        "dimensions": {
            "n_frequency": problem.grid.n_points,
            "ndof": problem.ndof,
            "nnz": problem.vertex.nnz,
        },
        "numerical_controls": numerical_controls,
        "controls": driver._plain(config),
    }
    manifest = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-campaign.v2",
        "source_tree_sha256": SOURCE_TREE,
        "environment_lock": {"sha256": ENVIRONMENT},
        "run_id": "parent",
        "coupling_scale": parent_scale,
        "continuation_parent": None,
        "born_initial": None,
        "preflight": preflight,
        "numerical_controls": numerical_controls,
        "outputs": outputs,
    }
    manifest_path = root / "campaign-manifest.json"
    manifest_path.write_bytes(driver._canonical(manifest))
    manifest_sha = driver._sha256_file(manifest_path)
    state = "passed" if converged else "scientific-failure"
    report = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-report.v1",
        "campaign_manifest_sha256": manifest_sha,
        "state": state,
        "scba_converged": converged,
        "checkpoint_sha256": driver._sha256_file(checkpoint),
    }
    report_path = root / "scba-report.json"
    report_path.write_bytes(driver._canonical(report))
    status = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-status.v1",
        "campaign_manifest_sha256": manifest_sha,
        "state": state,
        "scba_converged": converged,
        "report_sha256": driver._sha256_file(report_path),
    }
    (root / "status.json").write_bytes(driver._canonical(status))
    return problem, config


def _rebind_parent_manifest(root: Path, manifest: dict[str, object]) -> None:
    manifest_path = root / "campaign-manifest.json"
    manifest_path.write_bytes(driver._canonical(manifest))
    manifest_sha = driver._sha256_file(manifest_path)
    report_path = root / "scba-report.json"
    report = driver._load_canonical_json(report_path, "test report")
    report["campaign_manifest_sha256"] = manifest_sha
    report_path.write_bytes(driver._canonical(report))
    status_path = root / "status.json"
    status = driver._load_canonical_json(status_path, "test status")
    status["campaign_manifest_sha256"] = manifest_sha
    status["report_sha256"] = driver._sha256_file(report_path)
    status_path.write_bytes(driver._canonical(status))


def test_identity_continuation_is_numerically_unchanged_and_bound(
    tmp_path: Path,
) -> None:
    parent_problem, config = _write_parent(tmp_path / "parent")
    child = replace(
        parent_problem,
        vertex=SparseCubicVertex(
            ndof=parent_problem.vertex.ndof,
            indices=parent_problem.vertex.indices,
            coefficients=2.0 * parent_problem.vertex.coefficients,
            metadata=parent_problem.vertex.metadata,
        ),
    )

    initial, record = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="identity",
    )

    assert initial is not None and record is not None
    assert record["coupling_scale"] == 0.01
    assert record["checkpoint_sha256"] == driver._sha256_file(
        tmp_path / "parent/checkpoint/sigma-pp.h5"
    )
    baseline = checkpoint_initial_sigma_v2(
        load_scba_checkpoint_v2_self_validating(
            tmp_path / "parent/checkpoint/sigma-pp.h5"
        )
    )
    np.testing.assert_array_equal(initial.retarded, baseline.retarded)
    np.testing.assert_array_equal(initial.lesser, baseline.lesser)
    np.testing.assert_array_equal(initial.greater, baseline.greater)
    provenance = record["initialization"]
    assert provenance["predictor"] == "identity"
    assert provenance["factor"] == 1.0
    assert (
        provenance["untransformed_initial_sigma_sha256"]
        == provenance["transformed_consumed_initial_sigma_sha256"]
        == record["initial_sigma_sha256"]
    )
    bound = driver._source_hashes(_identity(), record)
    assert bound["continuation_parent_checkpoint"] == record["checkpoint_sha256"]
    assert bound["continuation_parent_initialization"] == provenance[
        "initialization_sha256"
    ]
    assert set(driver._source_hashes(_identity())).issubset(bound)

    repeated, repeated_record = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="identity",
    )
    np.testing.assert_array_equal(repeated.lesser, initial.lesser)
    np.testing.assert_array_equal(repeated.greater, initial.greater)
    assert repeated_record == record


def test_coupling_square_scales_authenticated_keldysh_state_exactly(
    tmp_path: Path,
) -> None:
    parent_problem, config = _write_parent(tmp_path / "parent")
    child = replace(
        parent_problem,
        vertex=SparseCubicVertex(
            ndof=parent_problem.vertex.ndof,
            indices=parent_problem.vertex.indices,
            coefficients=2.0 * parent_problem.vertex.coefficients,
            metadata=parent_problem.vertex.metadata,
        ),
    )
    baseline = checkpoint_initial_sigma_v2(
        load_scba_checkpoint_v2_self_validating(
            tmp_path / "parent/checkpoint/sigma-pp.h5"
        )
    )
    initial, record = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="coupling-square",
    )

    assert initial is not None and record is not None
    assert initial.shape == baseline.shape
    assert not initial.retarded.flags.writeable
    assert not initial.lesser.flags.writeable
    assert not initial.greater.flags.writeable
    np.testing.assert_array_equal(initial.retarded, np.zeros_like(initial.retarded))
    np.testing.assert_array_equal(initial.lesser, baseline.lesser * 4.0)
    np.testing.assert_array_equal(initial.greater, baseline.greater * 4.0)
    for component in (initial.lesser, initial.greater):
        np.testing.assert_allclose(
            component + np.swapaxes(component.conj(), 1, 2),
            0.0,
            atol=1.0e-15,
        )
    provenance = record["initialization"]
    assert provenance["predictor"] == "coupling-square"
    assert provenance["parent_coupling_scale"] == 0.01
    assert provenance["child_coupling_scale"] == 0.02
    assert provenance["factor"] == 4.0
    assert (
        provenance["untransformed_initial_sigma_sha256"]
        != provenance["transformed_consumed_initial_sigma_sha256"]
    )
    assert (
        record["initial_sigma_sha256"]
        == provenance["transformed_consumed_initial_sigma_sha256"]
    )


def test_release_checkpoint_identity_reconstructs_real_continuation_sources(
    tmp_path: Path,
) -> None:
    parent_root = tmp_path / "parent"
    parent_problem, config = _write_parent(parent_root, parent_scale=0.001)
    child = replace(
        parent_problem,
        vertex=SparseCubicVertex(
            ndof=parent_problem.vertex.ndof,
            indices=parent_problem.vertex.indices,
            coefficients=10.0 * parent_problem.vertex.coefficients,
            metadata=parent_problem.vertex.metadata,
        ),
    )
    scientific_task = {
        "task_id": "coupling-g001",
        "initialization": "continuation",
        "coupling_scale": 0.01,
        "continuation_parent": "parent",
        "continuation_axis": "coupling",
        "continuation_predictor": "coupling-square",
    }
    _initial, parent_record = driver._continuation_initial_state(
        parent_artifacts=parent_root,
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=scientific_task["coupling_scale"],
        continuation_axis=scientific_task["continuation_axis"],
        continuation_predictor=scientific_task["continuation_predictor"],
    )
    assert parent_record is not None
    task_data = tmp_path / "child"
    task_data.mkdir()
    expected, initialization, born = (
        mpi_driver._reconstruct_task_checkpoint_identity(
            scientific_task=scientific_task,
            scba_manifest={
                "continuation_parent": parent_record,
                "born_initial": None,
            },
            task_data=task_data,
            authenticated_task_inputs=(_identity(), child, config),
            authenticated_parent_scba=parent_root,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
        )
    )
    assert initialization == parent_record["initialization"]
    assert born is None
    assert expected.initial_state_sha256 == parent_record["initial_sigma_sha256"]
    expected_parent_sources = {
        "continuation_parent_campaign_manifest": parent_record[
            "campaign_manifest_sha256"
        ],
        "continuation_parent_checkpoint": parent_record["checkpoint_sha256"],
        "continuation_parent_report": parent_record["report_sha256"],
        "continuation_parent_status": parent_record["status_sha256"],
        "continuation_parent_initialization": parent_record[
            "initialization"
        ]["initialization_sha256"],
    }
    for name, digest in expected_parent_sources.items():
        assert expected.source_hashes[name] == digest
    assert expected.source_hashes["cubic_vertex_physics"] == (
        child.vertex.cubic_physics_sha256
    )
    assert expected.source_hashes["cubic_vertex_storage"] == (
        child.vertex.vertex_storage_sha256
    )

    mutations = {
        "campaign_manifest_sha256": "f" * 64,
        "checkpoint_sha256": "f" * 64,
        "report_sha256": "f" * 64,
        "status_sha256": "f" * 64,
    }
    for name, digest in mutations.items():
        changed_parent = dict(parent_record)
        changed_parent[name] = digest
        with pytest.raises(
            mpi_driver.MaterialCampaignError,
            match="differs from authenticated parent state",
        ):
            mpi_driver._reconstruct_task_checkpoint_identity(
                scientific_task=scientific_task,
                scba_manifest={
                    "continuation_parent": changed_parent,
                    "born_initial": None,
                },
                task_data=task_data,
                authenticated_task_inputs=(_identity(), child, config),
                authenticated_parent_scba=parent_root,
                source_tree_sha256=SOURCE_TREE,
                environment_sha256=ENVIRONMENT,
            )

    changed_parent = dict(parent_record)
    changed_initialization = dict(parent_record["initialization"])
    changed_initialization["initialization_sha256"] = "f" * 64
    changed_parent["initialization"] = changed_initialization
    with pytest.raises(
        mpi_driver.MaterialCampaignError,
        match="differs from authenticated parent state",
    ):
        mpi_driver._reconstruct_task_checkpoint_identity(
            scientific_task=scientific_task,
            scba_manifest={
                "continuation_parent": changed_parent,
                "born_initial": None,
            },
            task_data=task_data,
            authenticated_task_inputs=(_identity(), child, config),
            authenticated_parent_scba=parent_root,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
        )


def test_receipt_and_strict_validation_reject_correct_factor_wrong_state(
    tmp_path: Path,
) -> None:
    parent_problem, config = _write_parent(tmp_path / "parent")
    child = replace(
        parent_problem,
        vertex=SparseCubicVertex(
            ndof=parent_problem.vertex.ndof,
            indices=parent_problem.vertex.indices,
            coefficients=2.0 * parent_problem.vertex.coefficients,
            metadata=parent_problem.vertex.metadata,
        ),
    )
    _initial, authentic_parent = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="coupling-square",
    )
    assert authentic_parent is not None
    wrong_parent = dict(authentic_parent)
    wrong_initialization = dict(authentic_parent["initialization"])
    wrong_initialization["transformed_consumed_initial_sigma_sha256"] = (
        "f" * 64
    )
    initialization_payload = {
        name: value
        for name, value in wrong_initialization.items()
        if name != "initialization_sha256"
    }
    wrong_initialization["initialization_sha256"] = driver.hashlib.sha256(
        driver._canonical(initialization_payload)
    ).hexdigest()
    wrong_parent["initialization"] = wrong_initialization
    wrong_parent["initial_sigma_sha256"] = "f" * 64
    assert driver._validate_continuation_initialization(wrong_parent)[
        "factor"
    ] == 4.0
    scientific_task = {
        "coupling_scale": 0.02,
        "continuation_axis": "coupling",
        "continuation_predictor": "coupling-square",
    }
    task = SimpleNamespace(
        coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="coupling-square",
    )
    reproduced, reproduced_state, reproduced_parent = (
        mpi_driver._independently_authenticate_continuation(
        recorded_parent=authentic_parent,
        scientific_task=scientific_task,
        authenticated_parent_scba=tmp_path / "parent",
        child_identity=_identity(),
        child_problem=child,
        child_config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        )
    )
    assert reproduced == authentic_parent["initialization"]
    assert reproduced_state is not None
    assert reproduced_parent == authentic_parent
    strict_reproduced, strict_state, strict_parent = (
        strict_validation._strictly_reproduce_continuation(
        recorded_parent=authentic_parent,
        task=task,
        authenticated_parent_scba=tmp_path / "parent",
        child_inputs=(_identity(), child, config),
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        )
    )
    assert strict_reproduced == authentic_parent["initialization"]
    assert strict_state is not None
    assert strict_parent == authentic_parent

    with pytest.raises(
        mpi_driver.MaterialCampaignError,
        match="differs from authenticated parent state",
    ):
        mpi_driver._independently_authenticate_continuation(
            recorded_parent=wrong_parent,
            scientific_task=scientific_task,
            authenticated_parent_scba=tmp_path / "parent",
            child_identity=_identity(),
            child_problem=child,
            child_config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
        )
    with pytest.raises(
        strict_validation.MaterialReleaseValidationError,
        match="differs from authenticated parent state",
    ):
        strict_validation._strictly_reproduce_continuation(
            recorded_parent=wrong_parent,
            task=task,
            authenticated_parent_scba=tmp_path / "parent",
            child_inputs=(_identity(), child, config),
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
        )


def test_coupling_continuation_rejects_changed_mixing_floor(
    tmp_path: Path,
) -> None:
    problem, config = _write_parent(tmp_path / "parent")
    with pytest.raises(
        driver.SCBACaseError,
        match="cannot change numerical or temperature controls",
    ):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=problem,
            config=replace(config, mixing_floor=0.005),
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="identity",
        )


def test_research_continuation_can_reuse_converged_parent_across_orchestration_tree(
    tmp_path: Path,
) -> None:
    problem, config = _write_parent(tmp_path / "parent")
    child_tree = "c" * 64
    with pytest.raises(driver.SCBACaseError, match="source-tree hash differs"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=problem,
            config=config,
            source_tree_sha256=child_tree,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
        )

    initial, record = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=problem,
        config=config,
        source_tree_sha256=child_tree,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.02,
        continuation_axis="coupling",
        continuation_predictor="coupling-square",
        allow_parent_source_tree_mismatch=True,
    )
    assert initial is not None and record is not None
    assert record["parent_source_tree_sha256"] == SOURCE_TREE
    assert record["child_source_tree_sha256"] == child_tree
    assert record["source_tree_mismatch_allowed"] is True
    assert record["initialization"]["factor"] == 4.0


def test_research_continuation_still_rejects_environment_mismatch(
    tmp_path: Path,
) -> None:
    problem, config = _write_parent(tmp_path / "parent")
    with pytest.raises(driver.SCBACaseError, match="environment hash differs"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=problem,
            config=config,
            source_tree_sha256="c" * 64,
            environment_sha256="d" * 64,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
            allow_parent_source_tree_mismatch=True,
        )


@pytest.mark.parametrize("parent_scale", [0.02, 0.03])
def test_parent_coupling_must_be_strictly_lower(
    tmp_path: Path,
    parent_scale: float,
) -> None:
    problem, config = _write_parent(tmp_path / "parent", parent_scale=parent_scale)
    with pytest.raises(driver.SCBACaseError, match="strictly below"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=problem,
            config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="identity",
        )


def test_failed_or_tampered_parent_is_rejected(tmp_path: Path) -> None:
    failed_problem, failed_config = _write_parent(
        tmp_path / "failed", converged=False
    )
    with pytest.raises(driver.SCBACaseError, match="completed converged"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "failed",
            identity=_identity(),
            problem=failed_problem,
            config=failed_config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
        )

    good_problem, good_config = _write_parent(tmp_path / "tampered")
    status_path = tmp_path / "tampered/status.json"
    status = driver._load_canonical_json(status_path, "test status")
    status["report_sha256"] = "f" * 64
    status_path.write_bytes(driver._canonical(status))
    with pytest.raises(driver.SCBACaseError, match="does not bind its report"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "tampered",
            identity=_identity(),
            problem=good_problem,
            config=good_config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
        )


def test_parent_checkpoint_run_id_must_match_manifest_and_preflight(
    tmp_path: Path,
) -> None:
    problem, config = _write_parent(tmp_path / "parent")
    manifest_path = tmp_path / "parent/campaign-manifest.json"
    manifest = driver._load_canonical_json(manifest_path, "test manifest")
    manifest["run_id"] = "relabeled-parent"
    manifest["preflight"]["run_id"] = "relabeled-parent"
    _rebind_parent_manifest(tmp_path / "parent", manifest)

    with pytest.raises(driver.SCBACaseError, match="checkpoint run id differs"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=replace(
                problem,
                vertex=SparseCubicVertex(
                    ndof=problem.vertex.ndof,
                    indices=problem.vertex.indices,
                    coefficients=2.0 * problem.vertex.coefficients,
                    metadata=problem.vertex.metadata,
                ),
            ),
            config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="identity",
        )


def test_parent_coupling_relabel_cannot_hide_cubic_vertex_physics(
    tmp_path: Path,
) -> None:
    problem, config = _write_parent(tmp_path / "parent")
    manifest_path = tmp_path / "parent/campaign-manifest.json"
    manifest = driver._load_canonical_json(manifest_path, "test manifest")
    manifest["coupling_scale"] = 0.015
    manifest["preflight"]["coupling_scale"] = 0.015
    _rebind_parent_manifest(tmp_path / "parent", manifest)

    with pytest.raises(
        driver.SCBACaseError,
        match="material input identity does not reproduce|problem identity differs",
    ):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=replace(
                problem,
                vertex=SparseCubicVertex(
                    ndof=problem.vertex.ndof,
                    indices=problem.vertex.indices,
                    coefficients=2.0 * problem.vertex.coefficients,
                    metadata=problem.vertex.metadata,
                ),
            ),
            config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.02,
            continuation_axis="coupling",
            continuation_predictor="coupling-square",
        )

def test_temperature_bias_continuation_allows_only_outward_same_branch(
    tmp_path: Path,
) -> None:
    parent, config = _write_parent(tmp_path / "parent", parent_scale=0.01)
    child = replace(
        parent,
        left=replace(parent.left, temperature=0.9),
        right=replace(parent.right, temperature=0.5),
    )
    initial, record = driver._continuation_initial_state(
        parent_artifacts=tmp_path / "parent",
        identity=_identity(),
        problem=child,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        child_coupling_scale=0.01,
        continuation_axis="temperature_bias",
        continuation_predictor="identity",
    )
    assert initial is not None
    assert record is not None
    assert record["continuation_axis"] == "temperature_bias"
    assert record["initialization"]["predictor"] == "identity"
    assert record["initialization"]["factor"] == 1.0

    with pytest.raises(
        driver.SCBACaseError,
        match="temperature-bias continuation requires identity predictor",
    ):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=child,
            config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.01,
            continuation_axis="temperature_bias",
            continuation_predictor="coupling-square",
        )

    opposite = replace(
        parent,
        left=replace(parent.left, temperature=0.5),
        right=replace(parent.right, temperature=0.9),
    )
    with pytest.raises(driver.SCBACaseError, match="same branch"):
        driver._continuation_initial_state(
            parent_artifacts=tmp_path / "parent",
            identity=_identity(),
            problem=opposite,
            config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT,
            child_coupling_scale=0.01,
            continuation_axis="temperature_bias",
            continuation_predictor="identity",
        )


def test_case_driver_authenticates_born_initialization_and_excludes_parent(
    tmp_path: Path,
) -> None:
    problem = _problem(0.001)
    config = driver._config()
    born = tmp_path / "born.h5"
    write_born_initial_state(
        born,
        problem,
        config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        source_hashes=driver._source_hashes(_identity()),
    )
    initial, parent, record = driver._authenticated_initial_state(
        continuation_parent_artifacts=None,
        continuation_axis=None,
        continuation_predictor=None,
        born_initial_hdf5=born,
        identity=_identity(),
        problem=problem,
        config=config,
        source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT,
        coupling_scale=1.0,
    )
    assert initial is not None and parent is None and record is not None
    assert record["file_sha256"]
    assert driver._source_hashes(_identity(), None, record)[
        "born_initial_file"
    ] == record["file_sha256"]
    with pytest.raises(driver.SCBACaseError, match="mutually exclusive"):
        driver._authenticated_initial_state(
            continuation_parent_artifacts=tmp_path,
            continuation_axis="coupling",
            continuation_predictor="identity",
            born_initial_hdf5=born,
            identity=_identity(), problem=problem, config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT, coupling_scale=1.0,
        )
