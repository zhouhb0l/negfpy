from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
import sys

import numpy as np
import pytest

from negfpy.anharmonic import CubicSCBAProblem, HarmonicReservoir, SignedFrequencyGrid
from negfpy.anharmonic._checkpoint_v2 import solve_sparse_scba_checkpointed
from negfpy.anharmonic._born_initial import write_born_initial_state
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("validate_sparse_nq1_phase_c_scba")

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
    problem = _problem(0.001)
    result = solve_sparse_scba_checkpointed(
        problem,
        config,
        checkpoint_path=checkpoint,
        run_id="parent",
        environment_sha256=ENVIRONMENT,
        source_tree_sha256=SOURCE_TREE,
        source_hashes=driver._source_hashes(_identity()),
    )
    manifest = {
        "schema": "negfpy.sparse-nq1-phase-c-scba-campaign.v1",
        "source_tree_sha256": SOURCE_TREE,
        "environment_lock": {"sha256": ENVIRONMENT},
        "run_id": "parent",
        "coupling_scale": parent_scale,
        "numerical_controls": {
            "grid_step": float(problem.grid.step),
            "grid_n_positive": int(problem.grid.n_positive),
            "grid_cutoff": float(problem.grid.cutoff),
            "eta_lead": float(config.eta_lead),
            "mixing": float(config.mixing),
            "mixing_floor": float(config.mixing_floor),
            "temperature_bias": float(
                problem.left.temperature - problem.right.temperature
            ),
        },
        "outputs": {"checkpoint": str(checkpoint.resolve())},
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


def test_completed_lower_coupling_campaign_seeds_and_binds_child(
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
    )

    assert initial is not None and record is not None
    assert record["coupling_scale"] == 0.01
    assert record["checkpoint_sha256"] == driver._sha256_file(
        tmp_path / "parent/checkpoint/sigma-pp.h5"
    )
    assert record["initial_sigma_sha256"]
    bound = driver._source_hashes(_identity(), record)
    assert bound["continuation_parent_checkpoint"] == record["checkpoint_sha256"]
    assert set(driver._source_hashes(_identity())).issubset(bound)


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
    )
    assert initial is not None
    assert record is not None
    assert record["continuation_axis"] == "temperature_bias"

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
            born_initial_hdf5=born,
            identity=_identity(), problem=problem, config=config,
            source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT, coupling_scale=1.0,
        )
