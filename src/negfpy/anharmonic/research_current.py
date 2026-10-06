"""Opt-in final current quadrature for a retained research input.

This diagnostic integrates contact and collision spectra on the positive
frequency domain without evaluating another SCBA map or changing its result.
The GL8/16 difference is a sampled integration estimate for the declared
linear interpolation.  It is not a bound on global frequency accuracy.
"""

from __future__ import annotations

import math
import time

import numpy as np
from threadpoolctl import threadpool_limits

from negfpy.core.surface_gf import surface_gf

from ._moment_quadrature import _Budget, _Evaluator, _leaf
from .acoustic_contacts import _stable_bose_broadening_products
from .interacting_pointwise import PointwiseInteracting
from .research_moment import _field_hashes
from .types import CurrentReport


_COMPONENTS = ("left", "right", "collision_power")


def _validate_controls(
    edges,
    evaluate,
    emit,
    *,
    max_points,
    max_seconds,
    max_workers,
    max_leaves,
    rtol,
    atol,
    clock,
):
    edges = np.asarray(edges, dtype=float)
    if (
        edges.ndim != 1
        or len(edges) < 2
        or edges[0] != 0.0
        or not np.all(np.isfinite(edges))
        or np.any(np.diff(edges) <= 0.0)
    ):
        raise ValueError("Current edges must be finite, increasing, and start at zero.")
    for name, value, minimum in (
        ("max_points", max_points, 0),
        ("max_workers", max_workers, 1),
        ("max_leaves", max_leaves, len(edges) - 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value < minimum
        ):
            raise ValueError(f"{name} must be an integer >= {minimum}.")
    if (
        not math.isfinite(max_seconds)
        or max_seconds <= 0.0
        or not math.isfinite(rtol)
        or rtol < 0.0
        or not math.isfinite(atol)
        or atol < 0.0
    ):
        raise ValueError("Current budgets and tolerances must be finite and valid.")
    if not all(callable(function) for function in (evaluate, emit, clock)):
        raise TypeError("evaluate, emit, and clock must be callable.")
    return edges


def _statistics(high, low, rtol, atol):
    high = np.asarray(high, dtype=float).reshape(-1, 3)
    low = np.asarray(low, dtype=float).reshape(-1, 3)
    total = np.sum(high, axis=0)
    errors = np.sum(np.abs(high - low), axis=0)
    scale = float(max(abs(total[0]), abs(total[1])))
    allowed = float(atol + rtol * scale)
    return {
        "component_order": list(_COMPONENTS),
        "integrals": total.tolist(),
        "component_errors": errors.tolist(),
        "shared_contact_scale": scale,
        "allowed": allowed,
        "rtol": float(rtol),
        "atol": float(atol),
        "passed": bool(np.all(errors <= allowed)),
    }


def run_positive_current_quadrature(
    edges,
    evaluate,
    *,
    max_points=40000,
    max_seconds=1800.0,
    max_workers=8,
    max_leaves=1024,
    rtol=0.002,
    atol=0.0,
    emit=lambda **row: None,
    clock=time.monotonic,
):
    """Integrate left, right, and collision currents with retained final leaves.

    Every original positive-frequency interval is evaluated with GL8 and GL16.
    An atomic split costs another 48 queries.  A panel finishing after the wall
    budget counts toward the budget but cannot replace an accepted leaf.
    Incomplete coverage and exhausted budgets are returned as numerical
    outcomes; pointwise numerical failures propagate.
    """
    edges = _validate_controls(
        edges,
        evaluate,
        emit,
        max_points=max_points,
        max_seconds=max_seconds,
        max_workers=max_workers,
        max_leaves=max_leaves,
        rtol=rtol,
        atol=atol,
        clock=clock,
    )
    bounds = list(zip(edges[:-1], edges[1:]))
    limits = {
        "max_points": int(max_points),
        "max_seconds": float(max_seconds),
        "max_workers": int(max_workers),
        "max_leaves": int(max_leaves),
    }
    runner = _Evaluator(evaluate, (3,), max_points, max_seconds, max_workers, clock)
    leaves = []
    complete_partition = sampled_tolerance_met = False
    initial_completed = 0
    stop_reason = None
    emit(
        event="current_quadrature_start",
        initial_intervals=len(bounds),
        adaptive_orders=[8, 16],
        limits=limits,
        rtol=float(rtol),
        atol=float(atol),
    )
    try:
        for parent, (left, right) in enumerate(bounds):
            low, high = runner.panels([(left, right)], (8, 16))[0]
            if np.any(low.imag != 0.0) or np.any(high.imag != 0.0):
                raise ValueError("Current integrand must be real-valued.")
            leaves.append(_leaf(parent, left, right, low.real, high.real))
            initial_completed += 1
            if initial_completed % 50 == 0 or initial_completed == len(bounds):
                emit(
                    event="current_initial_progress",
                    completed_intervals=initial_completed,
                    total_intervals=len(bounds),
                    point_evaluations=runner.points,
                    elapsed_seconds=runner.clock() - runner.started,
                )
        complete_partition = True
        splits = 0
        while True:
            statistics = _statistics(
                np.stack([leaf.high for leaf in leaves]),
                np.stack([leaf.low for leaf in leaves]),
                rtol,
                atol,
            )
            if splits % 16 == 0 or statistics["passed"]:
                emit(
                    event="current_adaptive_progress",
                    leaves=len(leaves),
                    point_evaluations=runner.points,
                    elapsed_seconds=runner.clock() - runner.started,
                    statistics=statistics,
                )
            if statistics["passed"]:
                sampled_tolerance_met = True
                stop_reason = "sampled-tolerance"
                break
            if len(leaves) >= max_leaves:
                stop_reason = "leaf-budget"
                break
            index = max(
                range(len(leaves)),
                key=lambda item: (
                    float(np.max(leaves[item].error_abs)),
                    -leaves[item].left,
                ),
            )
            old = leaves[index]
            midpoint = 0.5 * (old.left + old.right)
            if midpoint == old.left or midpoint == old.right:
                stop_reason = "floating-point-resolution"
                break
            child_bounds = [(old.left, midpoint), (midpoint, old.right)]
            result = runner.panels(child_bounds, (8, 16))
            if any(np.any(value.imag != 0.0) for pair in result for value in pair):
                raise ValueError("Current integrand must be real-valued.")
            children = [
                _leaf(old.parent, left, right, low.real, high.real)
                for (left, right), (low, high) in zip(child_bounds, result)
            ]
            leaves[index : index + 1] = children
            splits += 1
    except _Budget:
        stop_reason = runner.reason
    finally:
        runner.close()

    leaves.sort(key=lambda leaf: (leaf.parent, leaf.left))
    arrays = {
        "edges": edges.copy(),
        "leaf_bounds": np.array(
            [(leaf.left, leaf.right) for leaf in leaves], dtype=float
        ).reshape(-1, 2),
        "leaf_parent_indices": np.array(
            [leaf.parent for leaf in leaves], dtype=np.int64
        ),
        "leaf_low": np.array([leaf.low for leaf in leaves], dtype=float).reshape(-1, 3),
        "leaf_high": np.array([leaf.high for leaf in leaves], dtype=float).reshape(
            -1, 3
        ),
        "leaf_error_abs": np.array(
            [leaf.error_abs for leaf in leaves], dtype=float
        ).reshape(-1, 3),
    }
    statistics = (
        _statistics(arrays["leaf_high"], arrays["leaf_low"], rtol, atol)
        if complete_partition
        else None
    )
    report = {
        "complete_partition": bool(complete_partition),
        "sampled_tolerance_met": bool(
            complete_partition and sampled_tolerance_met and statistics["passed"]
        ),
        "stop_reason": stop_reason,
        "initial_intervals": len(bounds),
        "initial_completed_intervals": initial_completed,
        "leaf_count": len(leaves),
        "point_evaluations": runner.points,
        "statistics": statistics,
        "elapsed_seconds": float(clock() - runner.started),
        "adaptive_orders": [8, 16],
        "rtol": float(rtol),
        "atol": float(atol),
        "limits": limits,
        "globally_certified": False,
        "evidence_scope": (
            "Every accepted final vector GL8/16 leaf retained; pointwise Green "
            "matrices not retained."
        ),
    }
    return report, arrays


class _CurrentIntegrand:
    """Material-tested three-component linear-Keldysh current integrand."""

    def __init__(self, problem, config, sigma):
        self.pointwise = PointwiseInteracting(
            problem, config, sigma, policy="linear-keldysh"
        )
        self.scale = problem.scale.current_scale

    def __call__(self, omega):
        omega = float(omega)
        if not math.isfinite(omega) or omega <= 0.0:
            raise ValueError("Current queries must be finite and positive.")
        evaluator = self.pointwise
        green = evaluator.evaluate(omega)
        values = []
        for contact in evaluator.contacts:
            surface = surface_gf(
                omega=omega,
                d00=contact.d00,
                d01=contact.forward,
                d10=contact.reverse,
                eta=evaluator.eta_lead,
                method=evaluator.surface_gf_method,
            )
            retarded = contact.coupling @ surface @ contact.coupling.conj().T
            n_gamma, np1_gamma = _stable_bose_broadening_products(
                omega,
                contact.temperature,
                1j * (retarded - retarded.conj().T),
            )
            indices = np.ix_(contact.indices, contact.indices)
            values.append(
                np.einsum("ij,ji->", -1j * np1_gamma, green.lesser[indices])
                - np.einsum("ij,ji->", -1j * n_gamma, green.greater[indices])
            )
        interaction = green.self_energy
        values.append(
            np.einsum("ij,ji->", interaction.greater, green.lesser)
            - np.einsum("ij,ji->", interaction.lesser, green.greater)
        )
        return np.asarray(
            self.scale * omega / (2.0 * np.pi) * np.real(values), dtype=float
        )


def _current_values(currents):
    if not isinstance(currents, CurrentReport):
        raise TypeError("coarse_currents must be a CurrentReport.")
    return np.array(
        [currents.left, currents.right, currents.collision_power], dtype=float
    )


def _interpret(
    problem,
    config,
    quadrature,
    coarse_currents,
    *,
    coarse_agreement_rtol,
):
    coarse = _current_values(coarse_currents)
    spatial_available = coarse_currents.spatial is not None
    if not quadrature["complete_partition"]:
        return {
            "available": False,
            "integration_passed": False,
            "coarse_currents": coarse.tolist(),
            "spatial_validation_available": False,
            "coarse_spatial_validation_available": spatial_available,
            "spatial_validation_refined": False,
        }
    total = np.array(quadrature["statistics"]["integrals"], dtype=float)
    errors = np.array(quadrature["statistics"]["component_errors"], dtype=float)
    shared_scale = float(max(abs(total[0]), abs(total[1])))
    comparison_scale = max(
        shared_scale,
        float(max(abs(coarse[0]), abs(coarse[1]))),
        np.finfo(float).tiny,
    )
    conservation_atol = float(config.current_atol * problem.scale.current_scale)
    conservation_allowed = max(conservation_atol, config.current_rtol * shared_scale)
    if problem.left.temperature > problem.right.temperature:
        direction = bool(total[0] > 0.0 and total[1] < 0.0)
    elif problem.right.temperature > problem.left.temperature:
        direction = bool(total[1] > 0.0 and total[0] < 0.0)
    else:
        direction = None
    return {
        "available": True,
        "integration_passed": bool(quadrature["sampled_tolerance_met"]),
        "component_order": list(_COMPONENTS),
        "currents": total.tolist(),
        "coarse_currents": coarse.tolist(),
        "contact_relative_change": (
            np.abs(total[:2] - coarse[:2]) / comparison_scale
        ).tolist(),
        "coarse_agreement_rtol": float(coarse_agreement_rtol),
        "coarse_contact_agreement": bool(
            np.all(
                np.abs(total[:2] - coarse[:2])
                <= coarse_agreement_rtol * comparison_scale
            )
        ),
        "contact_balance": float(total[0] + total[1]),
        "collision_power": float(total[2]),
        "full_balance": float(np.sum(total)),
        "contact_balance_sampled_error": float(np.sum(errors[:2])),
        "full_balance_sampled_error": float(np.sum(errors)),
        "conservation_rtol": float(config.current_rtol),
        "conservation_atol": conservation_atol,
        "conservation_allowed": float(conservation_allowed),
        "integrated_balance_screen_passed": bool(
            max(abs(total[0] + total[1]), abs(total[2]), abs(np.sum(total)))
            <= conservation_allowed
        ),
        "heat_flows_hot_to_cold": direction,
        "spatial_validation_available": False,
        "coarse_spatial_validation_available": spatial_available,
        "spatial_validation_refined": False,
        "global_frequency_accuracy_established": False,
    }


def _outcome(quadrature):
    if quadrature["sampled_tolerance_met"]:
        return "sampled-tolerance-met"
    reason = quadrature["stop_reason"]
    if not quadrature["complete_partition"]:
        return f"incomplete-{reason or 'partition'}"
    return reason or "sampled-tolerance-not-met"


class ResearchCurrentDiagnostic:
    """Bounded final-map current integration with retained leaf evidence.

    Pass a separate instance to each ``solve_scba(..., research_map=...,
    research_current=...)`` call.  ``report`` is copied into result metadata;
    ``arrays`` holds final accepted leaves for optional external archiving.
    Reusing an instance clears its previous evidence before the solver starts.

    The default absolute quadrature tolerance is ``config.current_atol``
    converted by the problem's physical current scale.  ``passed`` reports
    sampled integration tolerance only.  Coarse agreement, integrated balance,
    direction, and spatial availability remain separate evidence and never
    alter SCBA convergence or release decisions.
    """

    def __init__(
        self,
        *,
        max_points=40000,
        max_seconds=1800.0,
        max_workers=8,
        max_leaves=1024,
        rtol=0.002,
        atol=None,
        coarse_agreement_rtol=0.01,
        emit=None,
    ):
        self._options = {
            "max_points": max_points,
            "max_seconds": max_seconds,
            "max_workers": max_workers,
            "max_leaves": max_leaves,
            "rtol": rtol,
        }
        self._atol = atol
        self._emit = (lambda **row: None) if emit is None else emit
        validation_atol = 0.0 if atol is None else atol
        _validate_controls(
            np.array([0.0, 1.0]),
            lambda omega: None,
            self._emit,
            **self._options,
            atol=validation_atol,
            clock=time.monotonic,
        )
        if not math.isfinite(coarse_agreement_rtol) or coarse_agreement_rtol < 0.0:
            raise ValueError("coarse_agreement_rtol must be finite and non-negative.")
        self._coarse_agreement_rtol = float(coarse_agreement_rtol)
        self.report = None
        self.arrays = None

    @property
    def identity(self):
        return {
            "schema": "negfpy.research.final-current.v1",
            "algorithm": "positive-domain-adaptive-vector-gl8-gl16",
            "interpolation_policy": "linear-keldysh",
            "component_order": list(_COMPONENTS),
            "atol": None if self._atol is None else float(self._atol),
            "atol_policy": (
                "config.current_atol-times-problem-current-scale"
                if self._atol is None
                else "explicit-physical-current"
            ),
            "coarse_agreement_rtol": self._coarse_agreement_rtol,
            **self._options,
        }

    def _resolved_atol(self, problem, config):
        if self._atol is not None:
            return float(self._atol)
        return float(config.current_atol * problem.scale.current_scale)

    def prepare(self, problem, config):
        """Clear old evidence and validate bounds before entering SCBA."""
        self.report = self.arrays = None
        edges = problem.grid.values[problem.grid.zero_index :]
        _validate_controls(
            edges,
            lambda omega: None,
            self._emit,
            **self._options,
            atol=self._resolved_atol(problem, config),
            clock=time.monotonic,
        )
        if config.eta_device != 0.0:
            raise ValueError("Pointwise current diagnostics require eta_device=0.")

    def skip(self, *, coarse_currents, reason):
        coarse = _current_values(coarse_currents)
        self.arrays = None
        self.report = {
            "identity": self.identity,
            "evaluated": False,
            "outcome": "skipped",
            "reason": str(reason),
            "coarse_currents": coarse.tolist(),
            "quadrature": None,
            "interpretation": {
                "available": False,
                "integration_passed": False,
                "coarse_currents": coarse.tolist(),
                "spatial_validation_available": False,
                "coarse_spatial_validation_available": (
                    coarse_currents.spatial is not None
                ),
                "spatial_validation_refined": False,
            },
            "passed": False,
            "changes_solver_acceptance": False,
            "release_certified": False,
            "global_frequency_accuracy_established": False,
            "biased_transport_conservation_established": False,
        }
        return self.report

    def evaluate(self, problem, config, sigma, *, coarse_currents):
        """Integrate one fixed final-map input without an SCBA field update."""
        self.prepare(problem, config)
        coarse = _current_values(coarse_currents)
        before = _field_hashes(sigma)
        integrand = _CurrentIntegrand(problem, config, sigma)
        edges = problem.grid.values[problem.grid.zero_index :]
        atol = self._resolved_atol(problem, config)
        with threadpool_limits(limits=1, user_api="blas"):
            quadrature, arrays = run_positive_current_quadrature(
                edges,
                integrand,
                emit=self._emit,
                atol=atol,
                **self._options,
            )
        after = _field_hashes(sigma)
        if before != after:
            raise RuntimeError(
                "Input self-energy changed during the current diagnostic."
            )
        interpretation = _interpret(
            problem,
            config,
            quadrature,
            coarse_currents,
            coarse_agreement_rtol=self._coarse_agreement_rtol,
        )
        self.arrays = arrays
        self.report = {
            "identity": self.identity,
            "evaluated": True,
            "outcome": _outcome(quadrature),
            "reason": quadrature["stop_reason"],
            "resolved_atol": atol,
            "coarse_currents": coarse.tolist(),
            "input_field_sha256": before,
            "input_field_unchanged": True,
            "quadrature": quadrature,
            "interpretation": interpretation,
            "passed": bool(quadrature["sampled_tolerance_met"]),
            "changes_solver_acceptance": False,
            "release_certified": False,
            "global_frequency_accuracy_established": False,
            "biased_transport_conservation_established": False,
        }
        return self.report


__all__ = ["ResearchCurrentDiagnostic", "run_positive_current_quadrature"]
