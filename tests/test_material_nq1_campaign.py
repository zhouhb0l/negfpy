from __future__ import annotations

import pytest

from negfpy.anharmonic._material_nq1_campaign import (
    HARMONIC_CONSERVATIVE_BOUND,
    MaterialSCBATask,
    kelvin_to_solver_temperature,
    silicon_nq1_release_manifest,
    silicon_nq1_release_tasks,
)


def test_release_task_graph_is_complete_topological_and_preregistered() -> None:
    tasks = silicon_nq1_release_tasks()
    assert len(tasks) == 19
    assert [task.task_id for task in tasks[:9]] == [
        "coupling-g0001", "coupling-g001", "coupling-g0025",
        "coupling-g005", "coupling-g010",
        "coupling-g025", "coupling-g050", "coupling-g075", "coupling-g100",
    ]
    assert [task.coupling_scale for task in tasks[:9]] == [
        0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0,
    ]
    by_id = {task.task_id: task for task in tasks}
    for index, task in enumerate(tasks):
        if task.continuation_parent is not None:
            assert tasks.index(by_id[task.continuation_parent]) < index
    assert by_id["bias-positive-half"].continuation_parent == "coupling-g100"
    assert by_id["bias-positive-full"].continuation_parent == "bias-positive-half"
    assert by_id["bias-negative-half"].temperature_bias_kelvin == -10.0
    assert by_id["bias-negative-full"].temperature_bias_kelvin == -20.0
    assert by_id["endpoint-zero-start"].initialization == "zero"
    assert by_id["endpoint-born-start"].initialization == "born"
    assert {
        task.task_id for task in tasks if task.role == "numerical-refinement"
    } == {
        "refinement-half-spacing", "refinement-larger-cutoff",
        "refinement-half-eta", "refinement-mixing-010",
    }


def test_every_release_grid_covers_three_harmonic_bounds() -> None:
    for task in silicon_nq1_release_tasks():
        assert task.grid_cutoff >= 3.0 * HARMONIC_CONSERVATIVE_BOUND
    by_id = {task.task_id: task for task in silicon_nq1_release_tasks()}
    assert by_id["coupling-g100"].grid_cutoff == pytest.approx(7.5)
    assert by_id["coupling-g100"].grid_step == pytest.approx(0.1)
    assert 10.0 * by_id["coupling-g100"].grid_step < HARMONIC_CONSERVATIVE_BOUND
    assert by_id["refinement-half-spacing"].grid_step == pytest.approx(0.05)
    assert by_id["refinement-half-spacing"].grid_cutoff == pytest.approx(7.5)
    assert by_id["refinement-larger-cutoff"].grid_cutoff == pytest.approx(9.0)
    assert all(
        task.temperature_bias_kelvin == pytest.approx(20.0)
        for task in silicon_nq1_release_tasks()
        if task.role == "numerical-refinement"
    )


def test_physical_bias_conversion_and_reservoir_temperatures_are_explicit() -> None:
    by_id = {task.task_id: task for task in silicon_nq1_release_tasks()}
    task = by_id["bias-positive-full"]
    assert task.temperatures_kelvin == (310.0, 290.0)
    assert task.temperature_bias_solver == pytest.approx(
        kelvin_to_solver_temperature(20.0)
    )
    negative = by_id["bias-negative-full"]
    assert negative.temperatures_kelvin == (290.0, 310.0)


def test_campaign_manifest_hashes_all_tasks_inputs_and_acceptance() -> None:
    kwargs = {
        "source_revision": "candidate",
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "input_hashes": {"fc2": "c" * 64, "fc3": "d" * 64},
    }
    first = silicon_nq1_release_manifest(**kwargs)
    second = silicon_nq1_release_manifest(**kwargs)
    assert first == second
    assert len(first["campaign_sha256"]) == 64
    assert len(first["tasks"]) == 19
    assert first["acceptance"]["refinement_rtol"] == 5.0e-3
    changed = silicon_nq1_release_manifest(
        **{**kwargs, "input_hashes": {"fc2": "e" * 64, "fc3": "d" * 64}}
    )
    assert changed["campaign_sha256"] != first["campaign_sha256"]


def test_invalid_continuation_and_negative_temperature_fail_closed() -> None:
    base = {
        "task_id": "invalid",
        "role": "test",
        "coupling_scale": 0.1,
        "grid_step": 0.1,
        "grid_n_positive": 75,
        "eta_lead": 0.025,
        "mixing": 0.2,
        "mean_temperature_kelvin": 1.0,
        "temperature_bias_kelvin": 0.0,
    }
    with pytest.raises(ValueError, match="require one parent"):
        MaterialSCBATask(initialization="continuation", **base)
    with pytest.raises(ValueError, match="negative"):
        MaterialSCBATask(
            initialization="zero",
            **{**base, "temperature_bias_kelvin": 3.0},
        )
