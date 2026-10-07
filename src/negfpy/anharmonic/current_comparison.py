"""Attribute archived current-integration differences without new evaluations."""

from __future__ import annotations

import numpy as np


_COMPONENTS = ("left", "right", "collision_power")


def _real_array(value, name, shape=None):
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite real numbers.")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    return array.astype(float, copy=False)


def compare_current_quadrature(
    *,
    positive_frequencies,
    integration_weight,
    contact_spectra,
    collision_spectrum,
    current_leaves,
):
    """Return JSON-compatible interval attribution of two archived integrals.

    ``contact_spectra`` (2, N) and ``collision_spectrum`` (N,) are the stored
    positive-grid spectra from ``CurrentReport``: they include physical current
    scaling and frequency, but exclude the factor 1/(2*pi). The scalar
    ``integration_weight`` is Delta_omega/(2*pi). Thus original rectangle
    contributions use the *right* endpoint of each interval [0, omega_1], ... .

    ``current_leaves`` is the arrays mapping from
    ``run_positive_current_quadrature``. Its GL8/16 leaf integrals already
    include 1/(2*pi), so no additional weight is applied to them. The component
    order is left, right, collision_power throughout.

    Full, exactly adjacent, nonoverlapping leaf coverage of every original
    interval is required. Summed absolute GL16-minus-GL8 estimates are reported
    as sampled errors, not rigorous bounds. This comparison does not establish
    accuracy of the interpolated field, spatial conservation, or new SCBA
    convergence, and does not change any acceptance threshold.
    """
    edges = _real_array(current_leaves["edges"], "edges")
    if (
        edges.ndim != 1
        or len(edges) < 2
        or edges[0] != 0.0
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("edges must be increasing and start at zero.")
    n = len(edges) - 1
    frequencies = _real_array(positive_frequencies, "positive_frequencies", (n,))
    if not np.array_equal(frequencies, edges[1:]):
        raise ValueError("positive_frequencies must be the exact right edges.")
    weight = float(_real_array(integration_weight, "integration_weight", ()))
    if weight <= 0.0:
        raise ValueError("integration_weight must be positive.")
    uniform_edges = np.arange(n + 1) * (2.0 * np.pi * weight)
    edge_atol = 8.0 * np.finfo(float).eps * edges[-1]
    if not np.allclose(edges, uniform_edges, rtol=0.0, atol=edge_atol):
        raise ValueError("integration_weight must equal uniform Delta_omega/(2*pi).")
    contacts = _real_array(contact_spectra, "contact_spectra", (2, n))
    collision = _real_array(collision_spectrum, "collision_spectrum", (n,))

    bounds = _real_array(current_leaves["leaf_bounds"], "leaf_bounds")
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError("leaf_bounds must have shape (leaves, 2).")
    count = len(bounds)
    parents = np.asarray(current_leaves["leaf_parent_indices"])
    if (
        parents.shape != (count,)
        or parents.dtype.kind not in "iu"
        or np.any(parents < 0)
        or np.any(parents >= n)
    ):
        raise ValueError("leaf_parent_indices must be valid integer interval indices.")
    high = _real_array(current_leaves["leaf_high"], "leaf_high", (count, 3))
    low = _real_array(current_leaves["leaf_low"], "leaf_low", (count, 3))
    errors = _real_array(current_leaves["leaf_error_abs"], "leaf_error_abs", (count, 3))
    if not np.array_equal(errors, np.abs(high - low)):
        raise ValueError("leaf_error_abs must equal abs(leaf_high - leaf_low).")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("Every leaf must have positive width.")

    refined = np.zeros((n, 3))
    sampled_errors = np.zeros((n, 3))
    leaf_counts = np.zeros(n, dtype=int)
    for parent in range(n):
        indices = np.flatnonzero(parents == parent)
        if not len(indices):
            raise ValueError(f"Missing complete coverage of interval {parent}.")
        indices = indices[np.argsort(bounds[indices, 0], kind="stable")]
        group = bounds[indices]
        if (
            group[0, 0] != edges[parent]
            or group[-1, 1] != edges[parent + 1]
            or not np.array_equal(group[:-1, 1], group[1:, 0])
        ):
            raise ValueError(f"Gap, overlap, or wrong bounds in interval {parent}.")
        leaf_counts[parent] = len(indices)
        refined[parent] = high[indices].sum(axis=0)
        sampled_errors[parent] = errors[indices].sum(axis=0)
    rectangle = weight * np.column_stack((contacts.T, collision))
    difference = refined - rectangle
    refined_total = refined.sum(axis=0)
    rectangle_total = rectangle.sum(axis=0)
    scale = max(
        float(np.max(np.abs(refined_total[:2]))),
        float(np.max(np.abs(rectangle_total[:2]))),
        np.finfo(float).tiny,
    )
    return {
        "schema": "negfpy.research.current-comparison.v1",
        "component_order": list(_COMPONENTS),
        "rectangle_rule": "positive-grid-right-endpoint-delta-omega-over-two-pi",
        "leaf_integrals_include_one_over_two_pi": True,
        "coverage": {
            "complete_partition": True,
            "original_intervals": n,
            "leaf_count": count,
            "domain": [float(edges[0]), float(edges[-1])],
            "split_parent_indices": np.flatnonzero(leaf_counts > 1).tolist(),
        },
        "totals": {
            "refined": refined_total.tolist(),
            "rectangle": rectangle_total.tolist(),
            "refined_minus_rectangle": difference.sum(axis=0).tolist(),
            "sum_absolute_interval_difference": np.abs(difference).sum(axis=0).tolist(),
            "sampled_errors": sampled_errors.sum(axis=0).tolist(),
            "contact_relative_change_symmetric_scale": (
                np.abs(refined_total[:2] - rectangle_total[:2]) / scale
            ).tolist(),
        },
        "rankings_by_absolute_difference": {
            name: np.argsort(-np.abs(difference[:, column]), kind="stable").tolist()
            for column, name in enumerate(_COMPONENTS)
        },
        "intervals": [
            {
                "parent_index": parent,
                "bounds": edges[parent : parent + 2].tolist(),
                "leaf_count": int(leaf_counts[parent]),
                "refined": refined[parent].tolist(),
                "rectangle": rectangle[parent].tolist(),
                "refined_minus_rectangle": difference[parent].tolist(),
                "sampled_errors": sampled_errors[parent].tolist(),
            }
            for parent in range(n)
        ],
        "sampled_errors_are_rigorous_bounds": False,
        "global_frequency_accuracy_established": False,
        "changes_solver_acceptance": False,
    }


__all__ = ["compare_current_quadrature"]
