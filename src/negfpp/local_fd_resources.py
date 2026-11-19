"""Dimension-based admission for the retained fixed-block Method 1 engine."""
from __future__ import annotations

from pathlib import Path


def available_memory_bytes():
    """Read Linux available memory; unavailable accounting fails closed."""
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is required for local-FD memory admission")


def resource_plan(centers, nq, positive_nodes, controls, *, available_bytes=None):
    """Apply the existing cache/transient formulas to the declared dimensions.

    This is numeric-array admission with explicit process/runtime reserves,
    not an operating-system RSS limit or a guaranteed wall-time prediction.
    Nothing is pruned or coarsened to satisfy a cap.
    """
    if any(type(v) is not int or v < 1 for v in (centers, nq, positive_nodes)):
        raise ValueError("Positive integer dimensions required")
    tile = min(controls["frequency_tile"], positive_nodes)
    tiles = (positive_nodes + tile - 1) // tile
    workers = min(controls["workers"], nq * tiles)
    cells = centers + 2
    pairs, remainder = divmod(cells, 2)
    diagonal = pairs * 48**2 + remainder * 24**2
    edges = max(0, pairs - 1) * 48**2 + (48*24 if remainder and pairs else 0)
    left = nq * positive_nodes * diagonal * 16
    families = nq * positive_nodes * 4 * 2 * 48**2 * 16
    inputs = nq * positive_nodes * 16 * (2*(2*24**2+36**2)) + nq*tiles*3*24**2*16
    metadata = nq * tiles * (8192*cells + 128*tile + 4096)
    retained = left + families + inputs + metadata
    startup = workers * 24 * tile * 16 * (diagonal + 2*edges)
    action = workers * 8 * tile * 16 * (diagonal + 2*edges)
    numeric = retained + max(startup, action) + controls["parent_reserve_bytes"]
    # Material preparation precedes workers; its declared allocations must fit too.
    preparation = controls["harmonic_max_bytes"] + controls["contact_max_bytes"] + controls["parent_reserve_bytes"]
    requested = max(numeric, preparation) + controls["runtime_reserve_bytes"]
    available = available_memory_bytes() if available_bytes is None else available_bytes
    plan = {
        "schema": "negfpp.local-fd-resource-plan.v1", "center_layers": centers,
        "full_cells": cells, "full_dofs": 24*cells, "q_points": nq,
        "positive_frequency_nodes": positive_nodes, "frequency_tile": tile,
        "workers": workers, "left_factor_bytes": left, "raw_family_bytes": families,
        "input_bytes": inputs, "metadata_allowance_bytes": metadata,
        "retained_estimate_bytes": retained, "startup_workspace_bytes": startup,
        "action_workspace_bytes": action, "numeric_estimate_bytes": numeric,
        "preparation_cap_bytes": preparation, "reserved_total_bytes": requested,
        "available_bytes_at_admission": available, "controls": dict(controls),
        "all_q_frequency_nodes_retained": True, "spatial_work_per_action": "O(center_layers)",
        "runtime_depends_on_solver_iterations": True, "rss_limit_enforced": False,
    }
    if retained > controls["max_cache_bytes"]:
        raise ValueError("Retained cache estimate exceeds max_cache_bytes")
    if max(numeric, preparation) > controls["numeric_cap_bytes"]:
        raise ValueError("Numeric/preparation estimate exceeds numeric_cap_bytes")
    if requested + controls["available_margin_bytes"] > available:
        raise ValueError("Insufficient available memory for the declared local-FD plan")
    plan["admitted"] = True
    return plan
