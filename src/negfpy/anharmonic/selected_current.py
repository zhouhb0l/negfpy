"""Fixed-field selected spectra for finite-memory cubic cut diagnostics.

No SCBA iteration, static shift or acceptance decision is performed here.
The evaluator owns the physical equations, including the exact-zero limit.
"""

from dataclasses import dataclass
import time

import numpy as np

from .contracted_cubic_current import ContractedCubicCut, _positive_integer
from .grid import SignedFrequencyGrid
from .three_point import _complex


CENTERED_FIXED_FIELD_MEAN_POLICY = "centered-fixed-field-diagnostic"


@dataclass(frozen=True, slots=True)
class SelectedCutSpectra:
    """Union of required greater-G rows and harmonic current spectra.

    Frequencies follow ``grid``. Greater columns follow ``active_indices``;
    rows follow ``rows``. Harmonic spectra are in solver units and have not
    yet been multiplied by the frequency integration weight.
    """

    plans: tuple
    grid: SignedFrequencyGrid
    rows: np.ndarray
    active_indices: np.ndarray
    greater: np.ndarray
    harmonic_spectra: np.ndarray
    metadata: dict

    def integrate(self, cut, lag_times, *, mean_policy, block_size=16):
        """Return harmonic + connected current under an explicit centering assumption.

        The only supported policy is ``centered-fixed-field-diagnostic``.
        This labels a diagnostic of supplied bubble fields; it neither solves
        physical means nor retroactively adds a counterterm to their Hamiltonian.
        """
        if mean_policy != CENTERED_FIXED_FIELD_MEAN_POLICY:
            raise ValueError("Explicit centered-fixed-field-diagnostic mean_policy is required.")
        if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, (int, np.integer)):
            raise ValueError("cut must be an integer present in the selected spectra.")
        matches = [i for i, plan in enumerate(self.plans) if plan.operator.cut == cut]
        if len(matches) != 1:
            raise ValueError("cut is not present in the selected spectra.")
        position = matches[0]
        plan = self.plans[position]
        columns = np.searchsorted(self.active_indices, plan.active_indices)
        j = np.searchsorted(self.rows, plan.displacement_rows)
        i = np.searchsorted(self.rows, plan.momentum_rows)
        displacement = self.greater[:, j[:, None], columns]
        momentum = displacement if np.array_equal(i, j) else self.greater[:, i[:, None], columns]
        result = plan.integrate_from_selected_spectra(
            self.grid, displacement, momentum, lag_times, block_size=block_size)
        harmonic = float(self.grid.integration_weight * np.sum(self.harmonic_spectra[:, position]))
        total = harmonic + result["connected_cubic_solver"]
        result.update({
            "mean_policy": mean_policy,
            "harmonic_solver": harmonic,
            "harmonic": harmonic * plan.operator.current_scale,
            "centered_total_solver": total,
            "centered_total": total * plan.operator.current_scale,
            "harmonic_current_included": True,
            "physical_means_solved": False,
            "displacement_mean_prescribed_zero": True,
            "physical_force_balance_claimed": False,
            "static_feedback_included": False,
            "counterterm_added": False,
            "posthoc_static_mean_allowed": False,
            "full_material_current_validated": False,
            "input_centering_verified": False,
        })
        if not np.isfinite(result["centered_total"]):
            raise ValueError("Selected cut current overflowed.")
        return result


def collect_selected_cut_spectra(plans, grid, evaluate, *, max_evaluations,
                                max_seconds, max_spectrum_bytes=512 * 2**20):
    """Evaluate each signed node once, keeping only a union of selected blocks.

    ``evaluate(omega)`` must return finite full lesser/greater matrices with
    the declared device dimension. Only one frequency's full matrices are
    needed at a time. Exact zero is called explicitly; this function does not
    replace it by zero, an adjacent value or an equilibrium approximation.

    Limits are checked before allocation and between calls. An outer process
    timeout is necessary to interrupt a stalled evaluator. An incomplete
    collection raises and cannot be integrated as a complete grid.
    ``max_spectrum_bytes`` bounds output arrays, not evaluator scratch or RSS.
    """
    plans = tuple(plans)
    if not plans or not all(isinstance(p, ContractedCubicCut) for p in plans):
        raise ValueError("plans must contain at least one contracted cut plan.")
    geometry = {(p.operator.ndof, p.operator.n_layers, p.operator.dof_per_layer,
                 p.operator.current_scale) for p in plans}
    if len(geometry) != 1 or len({p.operator.cut for p in plans}) != len(plans):
        raise ValueError("Plans require common device geometry and current units, and distinct cuts.")
    if not isinstance(grid, SignedFrequencyGrid) or not callable(evaluate):
        raise TypeError("A signed grid and callable evaluator are required.")
    limit = _positive_integer(max_evaluations, "max_evaluations")
    byte_limit = _positive_integer(max_spectrum_bytes, "max_spectrum_bytes")
    if isinstance(max_seconds, (bool, np.bool_)) or not np.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive.")
    if grid.n_points > limit:
        raise ValueError("The complete grid exceeds max_evaluations.")
    active = np.unique(np.concatenate([p.active_indices for p in plans]))
    rows = np.unique(np.concatenate([np.union1d(p.displacement_rows, p.momentum_rows) for p in plans]))
    output_bytes = grid.n_points * (16 * len(rows) * len(active) + 8 * len(plans))
    if output_bytes > byte_limit:
        raise ValueError("Selected spectra exceed max_spectrum_bytes.")
    greater = np.empty((grid.n_points, len(rows), len(active)), dtype=complex)
    harmonic = np.empty((grid.n_points, len(plans)))
    ndof = plans[0].operator.ndof
    start = time.monotonic()
    for index, omega in enumerate(grid.values):
        if time.monotonic() - start >= max_seconds:
            raise TimeoutError(f"Selected collection exhausted max_seconds after {index} frequencies.")
        result = evaluate(float(omega))
        lesser = _complex(result.lesser, (ndof, ndof), "pointwise lesser Green function")
        field = _complex(result.greater, (ndof, ndof), "pointwise greater Green function")
        greater[index] = field[rows[:, None], active]
        for column, plan in enumerate(plans):
            i, j = plan.operator.harmonic_indices.T
            harmonic[index, column] = float(omega * (plan.operator.harmonic_coefficients @ lesser[i, j].real))
        del result, lesser, field
    elapsed = time.monotonic() - start
    if elapsed > max_seconds:
        raise TimeoutError("Selected collection exceeded max_seconds on its final frequency.")
    if not np.all(np.isfinite(harmonic)):
        raise ValueError("Harmonic current spectra overflowed.")
    for array in (active, rows, greater, harmonic):
        array.setflags(write=False)
    metadata = {
        "schema": "negfpy.research.selected-cut-spectra.v1",
        "cuts": [p.operator.cut for p in plans],
        "evaluations": grid.n_points, "elapsed_seconds": elapsed,
        "selected_rows": len(rows), "active_columns": len(active),
        "output_array_bytes": output_bytes,
        "full_green_frequency_history_retained": False,
        "zero_frequency_evaluated_explicitly": True,
        "frequency_quadrature": "stored-signed-grid-rectangle",
        "frequency_accuracy_verified": False,
        "changes_solver_acceptance": False,
    }
    return SelectedCutSpectra(plans, grid, rows, active, greater, harmonic, metadata)


__all__ = [
    "CENTERED_FIXED_FIELD_MEAN_POLICY",
    "SelectedCutSpectra",
    "collect_selected_cut_spectra",
]
