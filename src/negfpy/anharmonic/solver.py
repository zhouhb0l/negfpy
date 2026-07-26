"""Dense one-shot Born and self-consistent Born solvers.

The implementation in this module is intentionally a dense correctness
reference.  It keeps the full signed-frequency state, mixes the lesser and
greater cubic self-energies together, and reconstructs the retarded component
after mixing.  The fixed-point residual is always measured before mixing so a
small mixing parameter cannot create false convergence.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import os
import platform
from typing import Any

import numpy as np
import scipy

from ._harmonic import (
    harmonic_frequency_upper_bound,
    solve_keldysh_green,
    solve_retarded,
)
from ._provenance import (
    code_provenance,
    configuration_hash,
    initial_self_energy_hash,
    problem_input_hashes,
)
from .contacts import build_contact_spectra
from .cubic import (
    cubic_fft_scratch_bytes,
    cubic_lesser_greater,
    cubic_worker_scratch_bytes,
    resolved_worker_count,
)
from .current import (
    _evaluate_current_report,
)
from .diagnostics import check_solution, relative_residual
from .exceptions import SCBAConvergenceError
from .retarded import (
    principal_value_cauchy_fft_scratch_bytes,
    retarded_from_keldysh,
)
from .types import (
    AnharmonicResult,
    CubicSCBAProblem,
    CurrentReport,
    DiagnosticReport,
    KeldyshSpectrum,
    SCBAConfig,
    SCBAIteration,
    estimate_dense_memory_bytes,
)


Array = np.ndarray


def _cubic_execution_details(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> tuple[str, str, int]:
    """Return the actual dense cubic kernel, backend, and worker count."""

    if not np.any(problem.vertex.values):
        return "zero_vertex", "serial", 1
    if problem.ndof == 1:
        return "scalar_np_convolve", "serial", 1
    if config.backend == "serial":
        return "frequency_tiled", "serial", 1
    if config.backend == "fft":
        return "zero_padded_fft", "fft", 1
    return (
        "frequency_tiled",
        "threaded",
        resolved_worker_count(config.backend, config.max_workers),
    )


def _available_memory_bytes() -> int | None:
    """Return currently available physical memory without a hard dependency."""

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return page_size * available_pages


def _guard_dense_memory(problem: CubicSCBAProblem, config: SCBAConfig) -> int:
    """Apply the conservative dense-memory preflight from the release plan."""

    stored_state = estimate_dense_memory_bytes(
        problem.grid.n_points,
        problem.ndof,
    )
    _, _, workers = _cubic_execution_details(problem, config)
    contraction_scratch = (
        cubic_fft_scratch_bytes(problem.grid.n_points, problem.ndof)
        if config.backend == "fft"
        else workers * cubic_worker_scratch_bytes(problem.ndof)
    )
    # Row-wise direct reconstruction holds only O(Nw) weights/masks. The
    # optional padded transform explicitly accounts for its spectral arrays.
    hilbert_scratch = (
        principal_value_cauchy_fft_scratch_bytes(
            problem.grid.n_points,
            problem.ndof,
        )
        if config.hilbert_backend == "fft"
        else (
            2
            * problem.grid.n_points
            * np.dtype(np.float64).itemsize
        )
    )
    predicted = stored_state + contraction_scratch + hilbert_scratch
    available = _available_memory_bytes()
    fraction = float(getattr(config, "max_memory_fraction", 0.70))
    allow = bool(getattr(config, "allow_memory_overcommit", False))
    if available is not None and predicted > fraction * available and not allow:
        raise MemoryError(
            "Dense SCBA preflight predicts "
            f"{predicted / 2**30:.2f} GiB, exceeding {fraction:.0%} of the "
            f"currently available {available / 2**30:.2f} GiB. "
            "Reduce the grid/device size or explicitly allow memory overcommit."
        )
    return predicted


def _zero_spectrum(problem: CubicSCBAProblem) -> KeldyshSpectrum:
    return KeldyshSpectrum.zeros(
        n_frequency=problem.grid.n_points,
        ndof=problem.ndof,
    )


def _validate_initial_sigma(
    problem: CubicSCBAProblem,
    initial_sigma_pp: KeldyshSpectrum | None,
    config: SCBAConfig,
) -> tuple[Array, Array]:
    if initial_sigma_pp is None:
        shape = (problem.grid.n_points, problem.ndof, problem.ndof)
        return (
            np.zeros(shape, dtype=np.complex128),
            np.zeros(shape, dtype=np.complex128),
        )
    if not isinstance(initial_sigma_pp, KeldyshSpectrum):
        raise TypeError("initial_sigma_pp must be a KeldyshSpectrum or None.")
    expected = (problem.grid.n_points, problem.ndof, problem.ndof)
    if initial_sigma_pp.shape != expected:
        raise ValueError(
            f"initial_sigma_pp must have shape {expected}; "
            f"got {initial_sigma_pp.shape}."
        )

    # Retarded is deliberately reconstructed from the Keldysh pair below.  An
    # independently supplied retarded component could violate the spectral
    # identity and make the iteration non-conserving.
    lesser = np.array(initial_sigma_pp.lesser, dtype=np.complex128, copy=True)
    greater = np.array(initial_sigma_pp.greater, dtype=np.complex128, copy=True)
    retarded_from_keldysh(
        lesser,
        greater,
        problem.grid,
        strategy=config.retarded_strategy,
        hilbert_backend=config.hilbert_backend,
    )
    return lesser, greater


def _solve_green(
    problem: CubicSCBAProblem,
    left: KeldyshSpectrum,
    right: KeldyshSpectrum,
    sigma_pp_lesser: Array,
    sigma_pp_greater: Array,
    config: SCBAConfig,
) -> tuple[KeldyshSpectrum, KeldyshSpectrum]:
    sigma_pp_retarded = retarded_from_keldysh(
        sigma_pp_lesser,
        sigma_pp_greater,
        problem.grid,
        strategy=config.retarded_strategy,
        hilbert_backend=config.hilbert_backend,
    )
    sigma_pp = KeldyshSpectrum(
        retarded=sigma_pp_retarded,
        lesser=sigma_pp_lesser,
        greater=sigma_pp_greater,
    )
    try:
        green_retarded = solve_retarded(
            problem,
            left.retarded,
            right.retarded,
            sigma_pp.retarded,
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            "Dense Dyson solve failed. The device may contain a bound mode "
            "that is not broadened by either contact."
        ) from exc

    green = solve_keldysh_green(
        green_retarded,
        left.lesser + right.lesser + sigma_pp.lesser,
        left.greater + right.greater + sigma_pp.greater,
    )
    return green, sigma_pp


def _current_report(
    problem: CubicSCBAProblem,
    left: KeldyshSpectrum,
    right: KeldyshSpectrum,
    sigma_pp: KeldyshSpectrum,
    green: KeldyshSpectrum,
) -> CurrentReport:
    return _evaluate_current_report(problem, left, right, sigma_pp, green)


def _green_residual(
    current: KeldyshSpectrum,
    previous: KeldyshSpectrum | None,
    config: SCBAConfig,
) -> float:
    if previous is None:
        return float("inf")
    floor = (
        config.green_atol / config.green_rtol
        if config.green_rtol > 0.0
        else np.finfo(float).tiny
    )
    return max(
        relative_residual(current.retarded, previous.retarded, floor=floor),
        relative_residual(current.lesser, previous.lesser, floor=floor),
        relative_residual(current.greater, previous.greater, floor=floor),
    )


def _spectral_identity_residual(spectrum: KeldyshSpectrum) -> float:
    advanced = np.swapaxes(spectrum.retarded.conj(), -1, -2)
    return relative_residual(
        spectrum.greater - spectrum.lesser,
        spectrum.retarded - advanced,
    )


def _current_errors(
    problem: CubicSCBAProblem,
    currents: CurrentReport,
    config: SCBAConfig,
) -> tuple[float, float, float]:
    absolute = abs(currents.contact_balance)
    absolute_floor = config.current_atol * problem.scale.current_scale
    current_scale = max(
        abs(currents.left),
        abs(currents.right),
        absolute_floor,
    )
    relative = absolute / current_scale
    allowed = max(absolute_floor, config.current_rtol * current_scale)
    return absolute, relative, allowed


def _empty_diagnostics() -> DiagnosticReport:
    return DiagnosticReport(
        metrics={},
        failures=("final diagnostics have not been evaluated",),
        passed=False,
        release_certified=False,
    )


def _metadata(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    method: str,
    initialization: str,
    initialization_hash: str,
    predicted_memory_bytes: int,
) -> dict[str, Any]:
    provenance = code_provenance()
    input_hashes = problem_input_hashes(problem)
    input_hashes["configuration"] = configuration_hash(config)
    input_hashes["initial_state"] = initialization_hash

    calculation_digest = hashlib.sha256()
    calculation_digest.update(method.encode("ascii"))
    calculation_digest.update(input_hashes["problem"].encode("ascii"))
    calculation_digest.update(input_hashes["configuration"].encode("ascii"))
    calculation_digest.update(input_hashes["initial_state"].encode("ascii"))
    input_hashes["calculation"] = calculation_digest.hexdigest()
    requested_worker_count = resolved_worker_count(
        config.backend,
        config.max_workers,
    )
    cubic_kernel, effective_backend, worker_count = _cubic_execution_details(
        problem,
        config,
    )
    harmonic_upper_bound = harmonic_frequency_upper_bound(problem)
    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover - declared runtime dependency
        blas_runtime: tuple[str, ...] = ("threadpoolctl unavailable",)
    else:
        blas_runtime = tuple(
            (
                f"{item.get('internal_api', 'unknown')}:"
                f"{item.get('prefix', 'unknown')}:"
                f"{item.get('version', 'unknown')}"
            )
            for item in threadpool_info()
        )
    return {
        "formalism": "cubic_bubble_scba_guo_prb102_195412",
        "certification_contract": "dense_v0.1",
        "method": method,
        "initialization": initialization,
        "vertex_convention": "H3=(1/3!)*sum(Psi_abc*u_a*u_b*u_c)",
        "green_lesser_convention": (
            "G^<_{ab}(t,t')=-(i/hbar)<u_b(t')u_a(t)>"
        ),
        "green_greater_convention": (
            "G^>_{ab}(t,t')=-(i/hbar)<u_a(t)u_b(t')>"
        ),
        "fourier_forward": "F(omega)=integral_dt exp(+i*omega*t) F(t)",
        "fourier_inverse": (
            "F(t)=integral_domega/(2*pi) exp(-i*omega*t) F(omega)"
        ),
        "frequency_convention": "angular",
        "quadrature": "signed_uniform_rectangle_zero_extended",
        "reservoir_lesser_convention": "Sigma^<=-i*n_B*Gamma",
        "reservoir_greater_convention": "Sigma^>=-i*(1+n_B)*Gamma",
        "current_positive": "into_device",
        "retarded_strategy": config.retarded_strategy,
        "hilbert_backend": config.hilbert_backend,
        "backend": config.backend,
        "effective_backend": effective_backend,
        "cubic_kernel": cubic_kernel,
        "max_workers_requested": config.max_workers,
        "requested_worker_count": requested_worker_count,
        "worker_count": worker_count,
        "blas_threads_during_threaded_cubic": (
            1 if effective_backend == "threaded" else None
        ),
        "max_iterations": config.max_iterations,
        "min_iterations": config.min_iterations,
        "consecutive_passes": config.consecutive_passes,
        "mixing_initial": config.mixing,
        "mixing_floor": config.mixing_floor,
        "sigma_rtol": config.sigma_rtol,
        "sigma_atol": config.sigma_atol,
        "green_rtol": config.green_rtol,
        "green_atol": config.green_atol,
        "identity_rtol": config.identity_rtol,
        "current_rtol": config.current_rtol,
        "current_atol": config.current_atol,
        "current_atol_physical": (
            config.current_atol * problem.scale.current_scale
        ),
        "tail_rtol": config.tail_rtol,
        "eta_lead": config.eta_lead,
        "eta_device": config.eta_device,
        "surface_gf_method": config.surface_gf_method,
        "dtype": "complex128",
        "grid_step": problem.grid.step,
        "grid_n_positive": problem.grid.n_positive,
        "grid_cutoff": problem.grid.cutoff,
        "harmonic_frequency_upper_bound": harmonic_upper_bound,
        "cutoff_to_harmonic_upper_bound": (
            problem.grid.cutoff / harmonic_upper_bound
            if harmonic_upper_bound > 0.0
            else float("inf")
        ),
        "temperature_left": problem.left.temperature,
        "temperature_right": problem.right.temperature,
        "omega0": problem.scale.omega0,
        "hbar": problem.scale.hbar,
        "current_scale": problem.scale.current_scale,
        "scale_metadata": dict(problem.scale.metadata),
        "vertex_metadata": dict(problem.vertex.metadata),
        "vertex_nonzero": bool(np.any(problem.vertex.values != 0.0)),
        "vertex_max_abs": float(np.max(np.abs(problem.vertex.values))),
        "input_hashes": input_hashes,
        "predicted_peak_memory_bytes": predicted_memory_bytes,
        **provenance,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas_runtime": blas_runtime,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def _finalize_result(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    method: str,
    green: KeldyshSpectrum,
    sigma_pp: KeldyshSpectrum,
    left: KeldyshSpectrum,
    right: KeldyshSpectrum,
    currents: CurrentReport,
    history: tuple[SCBAIteration, ...],
    converged: bool,
    initialization: str,
    initialization_hash: str,
    predicted_memory_bytes: int,
) -> AnharmonicResult:
    provisional = AnharmonicResult(
        method=method,
        grid=problem.grid,
        green=green,
        sigma_pp=sigma_pp,
        sigma_left=left,
        sigma_right=right,
        currents=currents,
        diagnostics=_empty_diagnostics(),
        history=history,
        converged=converged,
        release_certified=False,
        metadata=_metadata(
            problem,
            config,
            method=method,
            initialization=initialization,
            initialization_hash=initialization_hash,
            predicted_memory_bytes=predicted_memory_bytes,
        ),
    )
    diagnostics = check_solution(problem, provisional, config)
    release_certified = bool(
        converged
        and method == "scba"
        and diagnostics.release_certified
        and provisional.has_release_certifying_provenance
    )
    return replace(
        provisional,
        diagnostics=diagnostics,
        release_certified=release_certified,
    )


def solve_one_shot_born(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> AnharmonicResult:
    """Perform one contact-only bubble evaluation and one updated solve.

    The returned method is ``"born"``.  This calculation is useful for
    weak-coupling checks and for comparison to mode-space perturbation theory,
    but is intentionally never marked release-certified because it is not a
    self-consistent conserving approximation.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")

    predicted = _guard_dense_memory(problem, config)
    initialization = "born_contact_only_zero_pp"
    initialization_hash = initial_self_energy_hash(problem, None)
    left, right = build_contact_spectra(
        problem,
        config.eta_lead,
        surface_gf_method=config.surface_gf_method,
    )
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    zero = np.zeros(shape, dtype=np.complex128)
    green_zero, _ = _solve_green(problem, left, right, zero, zero, config)
    sigma_lesser, sigma_greater = cubic_lesser_greater(
        problem.vertex,
        green_zero.lesser,
        green_zero.greater,
        problem.grid,
        backend=config.backend,
        max_workers=config.max_workers,
    )
    green, sigma_pp = _solve_green(
        problem,
        left,
        right,
        sigma_lesser,
        sigma_greater,
        config,
    )
    currents = _current_report(problem, left, right, sigma_pp, green)
    return _finalize_result(
        problem,
        config,
        method="born",
        green=green,
        sigma_pp=sigma_pp,
        left=left,
        right=right,
        currents=currents,
        history=(),
        converged=True,
        initialization=initialization,
        initialization_hash=initialization_hash,
        predicted_memory_bytes=predicted,
    )


def solve_scba(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    initial_sigma_pp: KeldyshSpectrum | None = None,
) -> AnharmonicResult:
    """Solve the full dense cubic self-consistent Born fixed point."""

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")

    predicted = _guard_dense_memory(problem, config)
    initialization = (
        "zero"
        if initial_sigma_pp is None
        else "supplied_keldysh_pair"
    )
    initialization_hash = initial_self_energy_hash(
        problem,
        initial_sigma_pp,
    )
    left, right = build_contact_spectra(
        problem,
        config.eta_lead,
        surface_gf_method=config.surface_gf_method,
    )
    sigma_lesser, sigma_greater = _validate_initial_sigma(
        problem,
        initial_sigma_pp,
        config,
    )

    history: list[SCBAIteration] = []
    previous_green: KeldyshSpectrum | None = None
    previous_sigma_residual: float | None = None
    residual_increase_streak = 0
    mixing = config.mixing
    consecutive = 0
    final_green: KeldyshSpectrum | None = None
    final_sigma: KeldyshSpectrum | None = None
    final_currents: CurrentReport | None = None

    for iteration in range(1, config.max_iterations + 1):
        green, sigma_pp = _solve_green(
            problem,
            left,
            right,
            sigma_lesser,
            sigma_greater,
            config,
        )
        raw_lesser, raw_greater = cubic_lesser_greater(
            problem.vertex,
            green.lesser,
            green.greater,
            problem.grid,
            backend=config.backend,
            max_workers=config.max_workers,
        )
        raw_retarded = retarded_from_keldysh(
            raw_lesser,
            raw_greater,
            problem.grid,
            strategy=config.retarded_strategy,
            hilbert_backend=config.hilbert_backend,
        )
        sigma_floor = (
            config.sigma_atol / config.sigma_rtol
            if config.sigma_rtol > 0.0
            else np.finfo(float).tiny
        )
        sigma_residual = max(
            relative_residual(
                raw_lesser,
                sigma_lesser,
                floor=sigma_floor,
            ),
            relative_residual(
                raw_greater,
                sigma_greater,
                floor=sigma_floor,
            ),
            relative_residual(
                raw_retarded,
                sigma_pp.retarded,
                floor=sigma_floor,
            ),
        )
        green_residual = _green_residual(green, previous_green, config)
        identity_residual = max(
            _spectral_identity_residual(green),
            _spectral_identity_residual(sigma_pp),
        )
        currents = _current_report(problem, left, right, sigma_pp, green)
        current_abs, current_rel, current_allowed = _current_errors(
            problem,
            currents,
            config,
        )
        current_absolute_floor = (
            config.current_atol * problem.scale.current_scale
        )
        collision_allowed = max(
            current_absolute_floor,
            config.current_rtol
            * max(
                abs(currents.left),
                abs(currents.right),
                current_absolute_floor,
            ),
        )
        full_abs = abs(currents.full_balance)
        layer_collision_abs = 0.0
        spatial_identity_abs = 0.0
        corrected_cut_error = 0.0
        block_local_bond_error = 0.0
        if currents.spatial is not None:
            layer_collision_abs = float(
                np.max(np.abs(currents.spatial.layer_collision_power))
            )
            spatial_identity_abs = float(
                np.max(np.abs(currents.spatial.continuity_residual))
            )
            corrected = np.asarray(
                currents.spatial.corrected_cut_currents,
                dtype=float,
            )
            if corrected.size:
                corrected_cut_error = max(
                    float(np.max(np.abs(corrected - currents.left))),
                    float(np.max(np.abs(corrected + currents.right))),
                    float(np.max(corrected) - np.min(corrected)),
                )
            if (
                currents.spatial.sigma_pp_is_block_local
                and currents.bond_currents is not None
                and np.asarray(currents.bond_currents).size
            ):
                bonds = np.asarray(currents.bond_currents, dtype=float)
                block_local_bond_error = max(
                    float(np.max(np.abs(bonds - currents.left))),
                    float(np.max(np.abs(bonds + currents.right))),
                    float(np.max(bonds) - np.min(bonds)),
                )
        direction_valid = True
        if problem.left.temperature > problem.right.temperature:
            direction_valid = bool(
                currents.left >= -current_allowed
                and currents.right <= current_allowed
            )
        elif problem.right.temperature > problem.left.temperature:
            direction_valid = bool(
                currents.right >= -current_allowed
                and currents.left <= current_allowed
            )

        passed = bool(
            iteration >= config.min_iterations
            and sigma_residual <= config.sigma_rtol
            and green_residual <= config.green_rtol
            and identity_residual <= config.identity_rtol
            and current_abs <= current_allowed
            and abs(currents.collision_power) <= collision_allowed
            and full_abs <= current_allowed
            and layer_collision_abs <= collision_allowed
            and spatial_identity_abs <= current_allowed
            and corrected_cut_error <= current_allowed
            and block_local_bond_error <= current_allowed
            and direction_valid
        )
        consecutive = consecutive + 1 if passed else 0

        will_continue = consecutive < config.consecutive_passes
        if will_continue:
            if (
                previous_sigma_residual is not None
                and sigma_residual
                > previous_sigma_residual * (1.0 + 1.0e-12)
            ):
                residual_increase_streak += 1
            else:
                residual_increase_streak = 0
            if residual_increase_streak >= 3:
                mixing = max(config.mixing_floor, 0.5 * mixing)
                residual_increase_streak = 0
            previous_sigma_residual = sigma_residual

        history.append(
            SCBAIteration(
                iteration=iteration,
                sigma_residual=sigma_residual,
                green_residual=green_residual,
                identity_residual=identity_residual,
                current_abs_error=current_abs,
                current_rel_error=current_rel,
                collision_power=currents.collision_power,
                mixing=mixing,
                passed=passed,
                consecutive_passes=consecutive,
            )
        )

        final_green = green
        final_sigma = sigma_pp
        final_currents = currents
        if consecutive >= config.consecutive_passes:
            return _finalize_result(
                problem,
                config,
                method="scba",
                green=green,
                sigma_pp=sigma_pp,
                left=left,
                right=right,
                currents=currents,
                history=tuple(history),
                converged=True,
                initialization=initialization,
                initialization_hash=initialization_hash,
                predicted_memory_bytes=predicted,
            )

        sigma_lesser = (
            (1.0 - mixing) * sigma_lesser + mixing * raw_lesser
        )
        sigma_greater = (
            (1.0 - mixing) * sigma_greater + mixing * raw_greater
        )
        previous_green = green

    assert final_green is not None
    assert final_sigma is not None
    assert final_currents is not None
    result = _finalize_result(
        problem,
        config,
        method="scba",
        green=final_green,
        sigma_pp=final_sigma,
        left=left,
        right=right,
        currents=final_currents,
        history=tuple(history),
        converged=False,
        initialization=initialization,
        initialization_hash=initialization_hash,
        predicted_memory_bytes=predicted,
    )
    if config.on_nonconvergence == "raise":
        last = history[-1]
        raise SCBAConvergenceError(
            "SCBA did not converge within "
            f"{config.max_iterations} iterations "
            f"(sigma residual={last.sigma_residual:.3e}, "
            f"green residual={last.green_residual:.3e}, "
            f"contact imbalance={last.current_abs_error:.3e}).",
            result,
        )
    return result


__all__ = ["solve_one_shot_born", "solve_scba"]
