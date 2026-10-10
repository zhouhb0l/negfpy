"""Postprocess stored finite-memory integrands without changing solver results."""

import numpy as np

from .cubic_cut_current import _real
from .three_point import _lags


def diagnose_current_memory(lag_times, connected_integrand_solver, *, current_scale,
                            harmonic_current, reference_current, window_start,
                            frequency_step=None):
    """Inspect every saved endpoint in an explicitly chosen memory window.

    The connected integrand is in solver units; ``current_scale`` converts its
    lag integral to the units of ``harmonic_current`` and ``reference_current``.
    Lag coordinates can be nonuniform. ``window_start`` must be a stored node
    before the last node. For a uniform signed frequency grid, supplying its
    ``frequency_step`` enforces the same half-recurrence limit as the current
    integrator; omitting it makes no recurrence check.

    Returned curves use cumulative trapezoids from zero, not a fresh integral
    beginning at window_start. The range inspects all stored endpoints in the
    window, including its boundaries. It is sampled sensitivity, not a bound
    on unresolved between-node structure or the omitted infinite-time tail.
    No averaging, damping, extrapolation or convergence decision is applied.
    A zero reference yields None for relative sensitivity, not zero error.
    Input arrays and existing acceptance flags are never modified.
    """
    lag = _lags(lag_times).copy()
    integrand = _real(connected_integrand_solver, lag.shape, "connected_integrand_solver")
    scale = float(_real(current_scale, (), "current_scale"))
    harmonic = float(_real(harmonic_current, (), "harmonic_current"))
    reference = float(_real(reference_current, (), "reference_current"))
    start = float(_real(window_start, (), "window_start"))
    if scale <= 0:
        raise ValueError("current_scale must be positive.")
    locations = np.flatnonzero(lag == start)
    if len(locations) != 1 or locations[0] == len(lag) - 1:
        raise ValueError("window_start must be a saved node before the final lag.")
    first = int(locations[0])
    half_recurrence = None
    if frequency_step is not None:
        step = float(_real(frequency_step, (), "frequency_step"))
        if step <= 0:
            raise ValueError("frequency_step must be positive.")
        half_recurrence = np.pi / step
        if not np.isfinite(half_recurrence) or lag[-1] >= half_recurrence:
            raise ValueError("Lag window must be shorter than half the frequency recurrence.")
    with np.errstate(over="ignore", invalid="ignore"):
        increments = .5 * (integrand[1:] + integrand[:-1]) * np.diff(lag)
        connected = np.concatenate(([0.], np.cumsum(increments))) * scale
        total = harmonic + connected
    if not np.all(np.isfinite(connected)) or not np.all(np.isfinite(total)):
        raise ValueError("Cumulative current overflowed.")
    window = connected[first:]
    low, high = int(np.argmin(window)) + first, int(np.argmax(window)) + first
    # Compute variation before adding the often much larger harmonic offset.
    span = float(connected[high] - connected[low])
    drift = float(connected[-1] - connected[first])
    if not np.isfinite(span) or not np.isfinite(drift):
        raise ValueError("Window current variation overflowed.")
    relative_reference = span / abs(reference) if reference else None
    relative_connected = span / abs(float(connected[-1])) if connected[-1] else None
    if any(value is not None and not np.isfinite(value)
           for value in (relative_reference, relative_connected)):
        raise ValueError("Normalized current variation overflowed.")
    summary = {
        "window_start": start, "window_end": float(lag[-1]),
        "window_points": len(window), "lag_points": len(lag),
        "connected_at_start": float(connected[first]),
        "connected_at_end": float(connected[-1]),
        "total_at_end": float(total[-1]), "harmonic_current": harmonic,
        "connected_min": float(connected[low]), "connected_max": float(connected[high]),
        "minimum_at_lag": float(lag[low]), "maximum_at_lag": float(lag[high]),
        "peak_to_peak_current": span, "endpoint_change": drift,
        "reference_current": reference,
        "peak_to_peak_over_abs_reference": relative_reference,
        "range_over_abs_final_connected": relative_connected,
        "half_recurrence_limit": half_recurrence,
        "recurrence_checked": frequency_step is not None,
        "quadrature": "cumulative-trapezoid-at-stored-lags",
        "convergence_certified": False, "exterior_tail_estimated": False,
        "changes_solver_acceptance": False,
    }
    for array in (lag, connected, total):
        array.setflags(write=False)
    return {"lag_times": lag, "connected_current": connected,
            "total_current": total, "summary": summary}
