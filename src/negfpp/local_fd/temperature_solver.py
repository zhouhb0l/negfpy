"""Matrix-free thermal-probe closure with exact final current residuals.

Only terminal vectors are retained.  Terminals 0 and 1 are the fixed left and
right contacts; probes are 2 onward.  The preconditioner follows physical order
``[left, probe 0, ..., probe M-1, right]``.  Its nearest-neighbour approximation
is *only* a numerical preconditioner: every residual and Jacobian action is
provided by the complete transport calculation, without a range truncation.

At fixed Krylov restart and iteration counts, storage and vector algebra are
O(M).  Each callback must separately provide O(M) work.  No general claim about
iteration counts, and hence no unconditional linear-time theorem, is made.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class SolverOptions:
    flux_rtol: float = 1e-10
    flux_atol_w: float = 1e-25
    linear_rtol: float = 1e-12
    max_newton_steps: int = 12
    max_krylov_iterations: int = 160
    krylov_restart: int = 20
    max_line_search_steps: int = 20
    max_current_calls: int = 160
    max_jacobian_calls: int = 400
    max_seconds: float = 600.

    def __post_init__(self):
        for name in ("flux_rtol", "flux_atol_w", "linear_rtol", "max_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_newton_steps", "max_krylov_iterations", "krylov_restart",
                     "max_line_search_steps", "max_current_calls", "max_jacobian_calls"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.krylov_restart > 20:
            raise ValueError("krylov_restart is capped at 20 to retain O(M) workspace")


class SolverBudgetExceeded(RuntimeError):
    """A declared time, callback, or Krylov-work bound has been reached."""


@dataclass(frozen=True)
class SolverResult:
    temperatures_kelvin: np.ndarray
    currents_w: np.ndarray
    success: bool
    message: str
    max_probe_flux_w: float
    flux_allowance_w: float
    contact_balance_w: float
    total_balance_w: float
    newton_iterations: int
    krylov_iterations: int
    current_calls: int
    jacobian_calls: int
    elapsed_seconds: float
    diagnostics: dict = field(default_factory=dict)

    def scalar_summary(self):
        return {name: getattr(self, name) for name in (
            "success", "message", "max_probe_flux_w", "flux_allowance_w",
            "contact_balance_w", "total_balance_w", "newton_iterations",
            "krylov_iterations", "current_calls", "jacobian_calls", "elapsed_seconds"
        )} | {"diagnostics": self.diagnostics}


@dataclass(frozen=True)
class LinearResponseResult:
    temperature_shifts_kelvin: np.ndarray
    currents_w: np.ndarray
    equilibrium_currents_w: np.ndarray
    success: bool
    message: str
    conductance_w_per_kelvin: float
    max_probe_flux_w: float
    flux_allowance_w: float
    contact_balance_w: float
    total_balance_w: float
    uniform_ward_max_w_per_kelvin: float
    equilibrium_max_current_w: float
    krylov_iterations: int
    current_calls: int
    jacobian_calls: int
    elapsed_seconds: float
    diagnostics: dict = field(default_factory=dict)

    def scalar_summary(self):
        return {name: getattr(self, name) for name in (
            "success", "message", "conductance_w_per_kelvin", "max_probe_flux_w",
            "flux_allowance_w", "contact_balance_w", "total_balance_w",
            "uniform_ward_max_w_per_kelvin", "equilibrium_max_current_w",
            "krylov_iterations", "current_calls", "jacobian_calls", "elapsed_seconds"
        )} | {"diagnostics": self.diagnostics}


def _vector(value, size, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real, without discarding an imaginary part")
    array = np.asarray(raw, dtype=float)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real vector of length {size}")
    return array


def _norm_inf(vector):
    return float(np.max(np.abs(vector))) if len(vector) else 0.


class ChainPreconditioner:
    """O(M) Thomas factors of a positive Dirichlet chain Laplacian.

    ``conductances`` is a positive scalar, or M+1 positive edge conductances in
    physical order.  It never substitutes for the exact transport operator.
    """

    def __init__(self, probe_count, conductances=None):
        if isinstance(probe_count, bool) or not isinstance(probe_count, (int, np.integer)) or probe_count < 0:
            raise ValueError("probe_count must be a nonnegative integer")
        self.probe_count = int(probe_count)
        if conductances is None:
            conductances = 1.
        raw = np.asarray(conductances)
        if np.iscomplexobj(raw):
            raise ValueError("Preconditioner conductances must be real")
        edges = np.asarray(raw, dtype=float)
        if edges.ndim == 0:
            edges = np.full(self.probe_count + 1, float(edges))
        if edges.shape != (self.probe_count + 1,) or not np.isfinite(edges).all() or np.any(edges <= 0):
            raise ValueError("Need M+1 finite positive chain conductances")
        self.edge_conductances = edges.copy()
        self.upper = -edges[1:-1].copy()
        self.pivots = (edges[:-1] + edges[1:]).copy()
        self.multipliers = np.empty(max(0, self.probe_count - 1))
        for i in range(1, self.probe_count):
            self.multipliers[i-1] = -edges[i] / self.pivots[i-1]
            self.pivots[i] -= self.multipliers[i-1] * self.upper[i-1]
        if np.any(self.pivots <= 0) or not np.isfinite(self.pivots).all():
            raise ValueError("Chain factorization is singular or numerically invalid")

    def solve(self, rhs):
        values = _vector(rhs, self.probe_count, "preconditioner rhs").copy()
        for i in range(1, self.probe_count):
            values[i] -= self.multipliers[i-1] * values[i-1]
        if self.probe_count:
            values[-1] /= self.pivots[-1]
        for i in range(self.probe_count-2, -1, -1):
            values[i] = (values[i] - self.upper[i] * values[i+1]) / self.pivots[i]
        return values


class AdditiveProbePreconditioner:
    """Diffusive-chain inverse plus a local high-frequency inverse.

    The two terms cover the smooth and rapidly varying temperature modes.  All
    positivity floors here change *only this numerical preconditioner*.  The
    exact current/Jacobian callbacks and physical stopping criteria are retained.
    """

    def __init__(self, internal_edges, contact_shunts, high_frequency_diagonal, diagnostics=None):
        size = len(contact_shunts)
        self.probe_count = size
        edges = _vector(internal_edges, max(0, size-1), "preconditioner internal edges")
        shunts = _vector(contact_shunts, size, "preconditioner contact shunts")
        self.high_frequency_diagonal = _vector(high_frequency_diagonal, size, "preconditioner high diagonal").copy()
        if np.any(edges <= 0) or np.any(shunts < 0) or np.any(self.high_frequency_diagonal <= 0):
            raise ValueError("Invalid positive numerical preconditioner coefficients")
        if size and not np.any(shunts > 0):
            raise ValueError("No contact anchor is available to the numerical preconditioner")
        self.upper = -edges.copy()
        self.pivots = shunts.copy()
        self.pivots[:-1] += edges
        self.pivots[1:] += edges
        self.multipliers = np.empty(max(0, size-1))
        for i in range(1, size):
            self.multipliers[i-1] = -edges[i-1]/self.pivots[i-1]
            self.pivots[i] -= self.multipliers[i-1]*self.upper[i-1]
        if np.any(self.pivots <= 0) or not np.isfinite(self.pivots).all():
            raise ValueError("Adaptive numerical preconditioner factorization failed")
        self.diagnostics = dict(diagnostics or {})
        self.diagnostics["stored_numeric_bytes"] = sum(a.nbytes for a in (
            self.high_frequency_diagonal, self.upper, self.pivots, self.multipliers))

    def solve(self, rhs):
        raw = _vector(rhs, self.probe_count, "preconditioner rhs")
        values = raw.copy()
        for i in range(1, self.probe_count):
            values[i] -= self.multipliers[i-1]*values[i-1]
        if self.probe_count:
            values[-1] /= self.pivots[-1]
        for i in range(self.probe_count-2, -1, -1):
            values[i] = (values[i]-self.upper[i]*values[i+1])/self.pivots[i]
        return values + raw/self.high_frequency_diagonal


def build_adaptive_preconditioner(jacobian_action, equilibrium_temperatures, *,
                                  max_numeric_bytes=32*1024**2,
                                  max_seconds=120., deadline=None):
    """Build an O(M) additive preconditioner from exactly five complete Jv calls.

    Calls use left/right contact units, an alternating probe mode, the linear
    physical coordinate x, and x squared.  Contact shunts are read from the first
    two actions.  Internal second moments follow the identity
    ``sum_j H_ij (i-j)^2 = 2*i*(Jx)_i - (Jx²)_i`` with contact contributions
    subtracted.  The identity assumes the uniform equilibrium Ward condition;
    solvers independently check the exact Ward/current residuals before success.

    Only the approximate preconditioner coefficients are made positive.  No
    coefficient in the physical operator, self-energy, source or Jv is changed.
    The numeric admission bound is conservative vector payload, not process RSS.
    """
    raw = np.asarray(equilibrium_temperatures)
    if raw.ndim != 1 or len(raw) < 2:
        raise ValueError("Equilibrium temperatures must include both contacts")
    size = len(raw)
    temperatures = _vector(raw, size, "equilibrium temperatures").copy()
    if np.any(temperatures <= 0) or not np.all(temperatures == temperatures[0]):
        raise ValueError("Adaptive preconditioner requires a positive uniform reference")
    if isinstance(max_numeric_bytes, bool) or not isinstance(max_numeric_bytes, int) or max_numeric_bytes < 1:
        raise ValueError("max_numeric_bytes must be a positive integer")
    estimated_bytes = 24*size*np.dtype(float).itemsize
    if estimated_bytes > max_numeric_bytes:
        raise ValueError("Adaptive preconditioner numeric payload exceeds its declared cap")
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("max_seconds must be finite and positive")
    started = time.monotonic()
    if deadline is not None and not math.isfinite(float(deadline)):
        raise ValueError("deadline must be finite when supplied")
    stop = min(started+max_seconds, math.inf if deadline is None else float(deadline))
    calls = 0
    def action(direction):
        nonlocal calls
        if time.monotonic() >= stop:
            raise SolverBudgetExceeded("Adaptive preconditioner time bound reached")
        calls += 1
        result = _vector(jacobian_action(temperatures.copy(), direction.copy()), size, "preconditioner Jv").copy()
        if time.monotonic() >= stop:
            raise SolverBudgetExceeded("Adaptive preconditioner time bound reached")
        return result
    probes = size-2
    left = np.zeros(size)
    left[0] = 1.
    right = np.zeros(size)
    right[1] = 1.
    left_response = action(left)
    right_response = action(right)
    alternate = np.zeros(size)
    alternate[2:] = (-1.)**np.arange(probes)
    alternate_response = action(alternate)
    coordinate = np.empty(size)
    coordinate[:2] = [0., 1.]
    coordinate[2:] = np.arange(1, probes+1)/(probes+1)
    linear_response = action(coordinate)
    quadratic_response = action(coordinate**2)
    left_shunt = -left_response[2:]
    right_shunt = -right_response[2:]
    high_raw = alternate[2:]*alternate_response[2:]
    position = np.arange(1, probes+1, dtype=float)
    second_moment = (2*coordinate[2:]*linear_response[2:]-quadratic_response[2:])*(probes+1)**2
    second_moment -= left_shunt*position**2 + right_shunt*(probes+1-position)**2
    edge_raw = .25*(second_moment[:-1]+second_moment[1:])
    scale = max(_norm_inf(high_raw), _norm_inf(left_shunt+right_shunt))
    if probes and (not math.isfinite(scale) or scale <= 0):
        raise ValueError("Adaptive preconditioner has no finite positive response scale")
    floor = max(np.finfo(float).tiny, scale*1e-12)
    shunts = np.maximum(left_shunt+right_shunt, 0.)
    edges = np.maximum(edge_raw, floor)
    high = np.maximum(high_raw, floor)
    diagnostics = {
        "kind": "additive_second_moment_chain_plus_high_frequency_diagonal",
        "jacobian_calls": calls,
        "elapsed_seconds": time.monotonic()-started,
        "estimated_numeric_bytes": estimated_bytes,
        "numeric_cap_bytes": max_numeric_bytes,
        "reference_temperature_kelvin": float(temperatures[0]),
        "preconditioner_only_approximation": True,
        "physical_operator_or_sources_modified": False,
        "preconditioner_positivity_floor_w_per_kelvin": floor,
        "edge_floor_count": int(np.sum(edge_raw < floor)),
        "high_frequency_floor_count": int(np.sum(high_raw < floor)),
        "negative_contact_shunt_count": int(np.sum(left_shunt+right_shunt < 0)),
        "raw_contact_shunt_min": float(np.min(left_shunt+right_shunt)) if probes else 0.,
        "raw_high_frequency_min": float(np.min(high_raw)) if probes else 0.,
        "raw_internal_second_moment_min": float(np.min(second_moment)) if probes else 0.,
    }
    return AdditiveProbePreconditioner(edges, shunts, high, diagnostics)


def _select_preconditioner(probe_count, local_conductances, preconditioner):
    if preconditioner is None:
        return ChainPreconditioner(probe_count, local_conductances)
    if local_conductances is not None:
        raise ValueError("Specify a preconditioner object or local conductances, not both")
    if getattr(preconditioner, "probe_count", None) != probe_count or not callable(getattr(preconditioner, "solve", None)):
        raise ValueError("Preconditioner object has the wrong probe count or no solve method")
    return preconditioner


class _Budget:
    def __init__(self, options, deadline):
        self.options = options
        self.started = time.monotonic()
        if deadline is not None and not math.isfinite(float(deadline)):
            raise ValueError("deadline must be finite when supplied")
        self.deadline = min(self.started + options.max_seconds,
                            math.inf if deadline is None else float(deadline))
        if not math.isfinite(self.deadline):
            raise ValueError("deadline must be finite when supplied")
        self.current_calls = self.jacobian_calls = self.krylov_iterations = 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise SolverBudgetExceeded("Probe solver time bound reached")

    def currents(self, callback, temperatures):
        self.check()
        if self.current_calls >= self.options.max_current_calls:
            raise SolverBudgetExceeded("Probe current callback cap reached")
        self.current_calls += 1
        output = _vector(callback(temperatures.copy()), len(temperatures), "currents")
        self.check()
        return output.copy()

    def jacobian(self, callback, temperatures, direction):
        self.check()
        if self.jacobian_calls >= self.options.max_jacobian_calls:
            raise SolverBudgetExceeded("Probe Jacobian callback cap reached")
        self.jacobian_calls += 1
        output = _vector(callback(temperatures.copy(), direction.copy()), len(temperatures), "Jacobian action")
        self.check()
        return output.copy()

    def krylov_step(self):
        self.check()
        if self.krylov_iterations >= self.options.max_krylov_iterations:
            raise SolverBudgetExceeded("Probe Krylov iteration cap reached")
        self.krylov_iterations += 1


def _gmres(action, rhs, preconditioner, budget, *, absolute_tolerance):
    """Restarted, left-preconditioned GMRES with true unpreconditioned checks."""
    count = len(rhs)
    if not count:
        return np.empty(0), True
    x = np.zeros(count)
    residual = rhs.copy()
    tolerance = absolute_tolerance + budget.options.linear_rtol * _norm_inf(rhs)
    restart = min(count, budget.options.krylov_restart)
    while _norm_inf(residual) > tolerance:
        budget.check()
        z = preconditioner.solve(residual)
        beta = float(np.linalg.norm(z))
        if not math.isfinite(beta) or beta == 0:
            return x, False
        basis = np.empty((count, restart+1))
        hessenberg = np.zeros((restart+1, restart))
        target = np.zeros(restart+1)
        target[0] = beta
        basis[:, 0] = z/beta
        candidate = x
        for j in range(restart):
            budget.krylov_step()
            w = preconditioner.solve(action(basis[:, j]))
            for _ in range(2):
                coefficients = basis[:, :j+1].T @ w
                hessenberg[:j+1, j] += coefficients
                w -= basis[:, :j+1] @ coefficients
            hessenberg[j+1, j] = np.linalg.norm(w)
            if hessenberg[j+1, j] > 0:
                basis[:, j+1] = w/hessenberg[j+1, j]
            y = np.linalg.lstsq(hessenberg[:j+2, :j+1], target[:j+2], rcond=None)[0]
            estimate = np.linalg.norm(target[:j+2] - hessenberg[:j+2, :j+1] @ y)
            breakdown = hessenberg[j+1, j] <= np.finfo(float).eps * max(
                np.finfo(float).tiny, np.linalg.norm(hessenberg[:j+1, j]))
            if j+1 == restart or estimate <= budget.options.linear_rtol*beta or breakdown:
                candidate = x + basis[:, :j+1] @ y
                true_residual = rhs-action(candidate)
                if _norm_inf(true_residual) <= tolerance:
                    return candidate, True
                if breakdown:
                    return candidate, False
                if j+1 == restart:
                    residual = true_residual
        x = candidate
    return x, True


def _allowance(currents, options):
    return options.flux_atol_w + options.flux_rtol*max(_norm_inf(currents[:2]), options.flux_atol_w)


def _physical_checks(currents, allowance):
    maximum = _norm_inf(currents[2:])
    contact = float(np.sum(currents[:2]))
    total = float(np.sum(currents))
    # Contact and full-source conservation are independent final gates at the
    # same physical allowance; no length-dependent relaxation is introduced.
    contact_limit = allowance
    return maximum <= allowance and abs(contact) <= contact_limit and abs(total) <= allowance, {
        "probe_flux_passed": maximum <= allowance,
        "contact_balance_passed": abs(contact) <= contact_limit,
        "total_balance_passed": abs(total) <= allowance,
        "contact_balance_allowance_w": contact_limit,
        "total_balance_allowance_w": allowance,
    }


def solve_probe_temperatures(currents: Callable, jacobian_action: Callable,
                             initial_temperatures, *, contact_temperatures=(300.5, 299.5),
                             local_conductances=None, preconditioner=None, options=None, deadline=None):
    """Bounded Newton/Krylov zero-current closure using complete exact callbacks.

    ``initial_temperatures`` is the full terminal vector (contacts followed by
    probes).  All Kelvin temperatures must be strictly positive.  Contact values
    are imposed exactly and probes must start within their closed range.  The
    deadline, if supplied, is an absolute ``time.monotonic()`` deadline.
    """
    options = SolverOptions() if options is None else options
    initial = np.asarray(initial_temperatures)
    if initial.ndim != 1 or len(initial) < 2:
        raise ValueError("Initial temperatures must include two contacts")
    temperatures = _vector(initial, len(initial), "initial temperatures").copy()
    contacts = _vector(contact_temperatures, 2, "contact temperatures")
    if np.any(contacts <= 0) or np.any(temperatures <= 0):
        raise ValueError("Kelvin temperatures must be strictly positive")
    lower, upper = float(contacts.min()), float(contacts.max())
    if np.any(temperatures[2:] < lower) or np.any(temperatures[2:] > upper):
        raise ValueError("Initial probes must lie within the contact temperature range")
    temperatures[:2] = contacts
    preconditioner = _select_preconditioner(len(temperatures)-2, local_conductances, preconditioner)
    budget = _Budget(options, deadline)
    values = np.full(len(temperatures), np.nan)
    newton_steps = 0
    message = "Probe Newton step cap reached"
    numerical_success = False
    try:
        values = budget.currents(currents, temperatures)
        for step in range(options.max_newton_steps+1):
            allowance = _allowance(values, options)
            if _norm_inf(values[2:]) <= allowance:
                numerical_success = True
                message = "Exact terminal-current residual passed"
                break
            if step == options.max_newton_steps:
                break
            if lower == upper:
                message = "Uniform-temperature current identity failed"
                break
            def action(direction):
                full = np.zeros(len(temperatures))
                full[2:] = direction
                return budget.jacobian(jacobian_action, temperatures, full)[2:]
            correction, linear_success = _gmres(action, -values[2:], preconditioner, budget,
                                                 absolute_tolerance=.05*allowance)
            if not linear_success:
                message = "Krylov solve did not meet its true Jacobian residual"
                break
            newton_steps += 1
            alpha = 1.
            positive = correction > 0
            negative = correction < 0
            if np.any(positive):
                alpha = min(alpha, float(np.min((upper-temperatures[2:][positive])/correction[positive])))
            if np.any(negative):
                alpha = min(alpha, float(np.min((lower-temperatures[2:][negative])/correction[negative])))
            if alpha <= 0 or not math.isfinite(alpha):
                message = "Newton direction cannot preserve the temperature bounds"
                break
            accepted = False
            old_norm = _norm_inf(values[2:])
            for _ in range(options.max_line_search_steps):
                trial = temperatures.copy()
                trial[2:] += alpha*correction
                # Bounds are checked on the raw iterate, never repaired/clipped.
                if np.any(trial[2:] < lower) or np.any(trial[2:] > upper):
                    alpha *= .5
                    continue
                candidate = budget.currents(currents, trial)
                candidate_norm = _norm_inf(candidate[2:])
                if candidate_norm <= _allowance(candidate, options) or candidate_norm < old_norm*(1-1e-4*alpha):
                    temperatures, values = trial, candidate
                    accepted = True
                    break
                alpha *= .5
            if not accepted:
                message = "Bounded Newton line search did not reduce the exact residual"
                break
    except SolverBudgetExceeded as exc:
        message = str(exc)
    allowance = _allowance(values, options) if np.isfinite(values).all() else math.nan
    checks_passed, checks = _physical_checks(values, allowance)
    success = numerical_success and checks_passed
    if numerical_success and not checks_passed:
        message = "Probe currents passed but independent conservation checks failed"
    return SolverResult(temperatures.copy(), values.copy(), bool(success), message,
                        _norm_inf(values[2:]), allowance, float(np.sum(values[:2])),
                        float(np.sum(values)), newton_steps, budget.krylov_iterations,
                        budget.current_calls, budget.jacobian_calls, time.monotonic()-budget.started,
                        checks | {"restart_cap": options.krylov_restart,
                                  "workspace_vector_length": len(temperatures)-2,
                                  "preconditioner_build": getattr(preconditioner, "diagnostics", {}),
                                  "preconditioner_only_approximation": True,
                                  "numerical_residual_passed": numerical_success})


def solve_linear_response(currents: Callable, jacobian_action: Callable,
                          equilibrium_temperature, probe_count, *, bias_direction=(.5, -.5),
                          local_conductances=None, preconditioner=None, options=None, deadline=None):
    """Exact matrix-free LR closure, uniform Ward gate and positive conductance.

    The derivative callback must be analytic and correspond to the same complete
    currents callback.  Equilibrium currents and the full constant-temperature
    derivative are checked independently; neither is manufactured as zero.
    """
    options = SolverOptions() if options is None else options
    if not math.isfinite(equilibrium_temperature) or equilibrium_temperature <= 0:
        raise ValueError("Equilibrium temperature must be finite and positive")
    preconditioner = _select_preconditioner(probe_count, local_conductances, preconditioner)
    size = int(probe_count)+2
    contacts = _vector(bias_direction, 2, "bias direction")
    bias = float(contacts[0]-contacts[1])
    if bias == 0:
        raise ValueError("A nonzero contact bias direction is required")
    temperatures = np.full(size, float(equilibrium_temperature))
    shifts = np.zeros(size)
    shifts[:2] = contacts
    values = np.full(size, np.nan)
    equilibrium = np.full(size, np.nan)
    ward = np.full(size, np.nan)
    budget = _Budget(options, deadline)
    message = "Linear response did not finish"
    linear_success = False
    refinement_steps = 0
    try:
        equilibrium = budget.currents(currents, temperatures)
        ward = budget.jacobian(jacobian_action, temperatures, np.ones(size))
        drive = budget.jacobian(jacobian_action, temperatures, shifts)
        def action(direction):
            full = np.zeros(size)
            full[2:] = direction
            return budget.jacobian(jacobian_action, temperatures, full)[2:]
        initial_allowance = _allowance(drive, options)
        correction, linear_success = _gmres(action, -drive[2:], preconditioner, budget,
                                             absolute_tolerance=.05*initial_allowance)
        shifts[2:] = correction
        values = budget.jacobian(jacobian_action, temperatures, shifts)
        # Contact conductance can fall as 1/M, so the initial drive scale can be
        # much larger than the final current scale.  Refine against the latter;
        # never certify a drive-normalized residual as the physical residual.
        while linear_success and not _physical_checks(values, _allowance(values, options))[0]:
            if refinement_steps >= options.max_newton_steps or not probe_count:
                break
            residual = -values[2:]
            if _norm_inf(residual) == 0:
                break  # A conservation defect cannot be repaired by probe closure.
            extra, linear_success = _gmres(action, residual, preconditioner, budget,
                                            absolute_tolerance=.05*_allowance(values, options)/max(1, probe_count))
            shifts[2:] += extra
            refinement_steps += 1
            values = budget.jacobian(jacobian_action, temperatures, shifts)
        message = "Exact linear-response residual passed" if linear_success else "Krylov true residual failed"
    except SolverBudgetExceeded as exc:
        message = str(exc)
    allowance = _allowance(values, options) if np.isfinite(values).all() else math.nan
    physical_passed, checks = _physical_checks(values, allowance)
    conductance = float(values[0]/bias)
    ward_max, equilibrium_max = _norm_inf(ward), _norm_inf(equilibrium)
    # Ward is in W/K, normalized to the differential current scale in W/K.
    ward_allowance = options.flux_atol_w + options.flux_rtol*max(abs(conductance), options.flux_atol_w)
    equilibrium_allowance = options.flux_atol_w + options.flux_rtol*max(_norm_inf(values[:2]), options.flux_atol_w)
    ward_passed = ward_max <= ward_allowance
    equilibrium_passed = equilibrium_max <= equilibrium_allowance
    positive = math.isfinite(conductance) and conductance > 0
    success = bool(linear_success and physical_passed and ward_passed and equilibrium_passed and positive)
    checks |= {"uniform_ward_passed": ward_passed,
               "uniform_ward_allowance_w_per_kelvin": ward_allowance,
               "equilibrium_current_passed": equilibrium_passed,
               "equilibrium_current_allowance_w": equilibrium_allowance,
               "finite_positive_conductance": positive,
               "restart_cap": options.krylov_restart,
               "workspace_vector_length": int(probe_count),
               "preconditioner_build": getattr(preconditioner, "diagnostics", {}),
               "linear_refinement_steps": refinement_steps,
               "preconditioner_only_approximation": True}
    if linear_success and not success:
        message = "Linear solve passed but independent flux/Ward/equilibrium/conductance gates failed"
    return LinearResponseResult(shifts.copy(), values.copy(), equilibrium.copy(), success,
                                message, conductance, _norm_inf(values[2:]), allowance,
                                float(np.sum(values[:2])), float(np.sum(values)), ward_max,
                                equilibrium_max, budget.krylov_iterations, budget.current_calls,
                                budget.jacobian_calls, time.monotonic()-budget.started, checks)
