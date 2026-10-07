"""Offline attribution must retain normalization and complete coverage."""

import copy
import json

import numpy as np
import pytest

from negfpy.anharmonic.current_comparison import compare_current_quadrature


def fixture():
    # f(w) = [w, -w, 0] before dividing by 2*pi. Split the first
    # original interval; deliberately store the leaves out of order.
    bounds = np.array([[0.5, 1.0], [0.25, 0.5], [0.0, 0.25]])
    scalar = (bounds[:, 1] ** 2 - bounds[:, 0] ** 2) / (4 * np.pi)
    high = np.column_stack((scalar, -scalar, np.zeros(3)))
    low = high.copy()
    low[1, 0] -= 1e-5
    low[2, 0] += 1e-5
    return {
        "positive_frequencies": np.array([0.5, 1.0]),
        "integration_weight": 0.5 / (2 * np.pi),
        "contact_spectra": np.array([[0.5, 1.0], [-0.5, -1.0]]),
        "collision_spectrum": np.zeros(2),
        "current_leaves": {
            "edges": np.array([0.0, 0.5, 1.0]),
            "leaf_bounds": bounds,
            "leaf_parent_indices": np.array([1, 0, 0]),
            "leaf_high": high,
            "leaf_low": low,
            "leaf_error_abs": abs(high - low),
        },
    }


def test_analytic_integral_normalization_and_signed_attribution():
    result = compare_current_quadrature(**fixture())
    np.testing.assert_allclose(
        result["totals"]["refined"], [1 / (4 * np.pi), -1 / (4 * np.pi), 0]
    )
    np.testing.assert_allclose(
        result["totals"]["rectangle"], [3 / (8 * np.pi), -3 / (8 * np.pi), 0]
    )
    np.testing.assert_allclose(
        result["totals"]["refined_minus_rectangle"],
        [-1 / (8 * np.pi), 1 / (8 * np.pi), 0],
    )
    np.testing.assert_allclose(result["totals"]["sampled_errors"], [2e-5, 0, 0])
    assert result["coverage"]["split_parent_indices"] == [0]
    assert result["coverage"]["complete_partition"]
    assert not result["sampled_errors_are_rigorous_bounds"]
    assert not result["changes_solver_acceptance"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "defect", ["gap", "overlap", "wrong_parent", "missing_parent", "zero_width"]
)
def test_rejects_incomplete_or_invalid_partition(defect):
    kwargs = fixture()
    leaves = kwargs["current_leaves"]
    if defect == "gap":
        leaves["leaf_bounds"][1, 0] = 0.3
    elif defect == "overlap":
        leaves["leaf_bounds"][1, 0] = 0.2
    elif defect == "wrong_parent":
        leaves["leaf_parent_indices"][1] = 1
    elif defect == "missing_parent":
        leaves["leaf_parent_indices"][:] = 0
    else:
        leaves["leaf_bounds"][1, 0] = leaves["leaf_bounds"][1, 1]
    with pytest.raises(ValueError):
        compare_current_quadrature(**kwargs)


def test_leaf_order_does_not_change_result_or_mutate_evidence():
    kwargs = fixture()
    original = copy.deepcopy(kwargs)
    first = compare_current_quadrature(**kwargs)
    for key, value in kwargs["current_leaves"].items():
        np.testing.assert_array_equal(value, original["current_leaves"][key])
        if key != "edges":
            kwargs["current_leaves"][key] = value[::-1]
    assert compare_current_quadrature(**kwargs) == first


@pytest.mark.parametrize(
    "defect", ["weight", "node", "nonfinite", "error", "fractional_parent", "complex"]
)
def test_rejects_wrong_normalization_or_corrupt_archive(defect):
    kwargs = fixture()
    if defect == "weight":
        kwargs["integration_weight"] *= 2 * np.pi
    elif defect == "node":
        kwargs["positive_frequencies"][0] += 0.01
    elif defect == "nonfinite":
        kwargs["contact_spectra"][0, 0] = np.nan
    elif defect == "error":
        kwargs["current_leaves"]["leaf_error_abs"][0, 0] = 1.0
    elif defect == "fractional_parent":
        kwargs["current_leaves"]["leaf_parent_indices"] = np.array([1.0, 0.0, 0.0])
    else:
        kwargs["collision_spectrum"] = np.array([1j, 0.0])
    with pytest.raises(ValueError):
        compare_current_quadrature(**kwargs)
