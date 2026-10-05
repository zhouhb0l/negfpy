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
import json
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


class ResearchMapBudgetExhausted(Exception):
    """A cooperative research-map stop, never a numerical failure or retry."""


def _research_identity(research_map):
    """Validate the explicit, serializable identity without storing a callback."""
    if not isinstance(research_map.identity, dict) or not research_map.identity:
        raise ValueError("research_map.identity must be a nonempty JSON dictionary.")
    for name in ("evaluate", "metadata", "committed"):
        if not callable(getattr(research_map, name, None)):
            raise TypeError(f"research_map.{name} must be callable.")
    return json.loads(json.dumps(research_map.identity, allow_nan=False))


def _research_evaluate(research_map, problem, left, right, lesser, greater,
                       config, *, purpose):
    evaluation = research_map.evaluate(
        problem, left, right, lesser, greater, config, purpose=purpose,
    )
    if not isinstance(evaluation, tuple) or len(evaluation) != 5:
        raise TypeError("Research evaluator must return one complete five-field evaluation.")
    green, sigma, raw_lesser, raw_greater, residual = evaluation
    if not isinstance(green, KeldyshSpectrum) or not isinstance(sigma, KeldyshSpectrum):
        raise TypeError("Research evaluation requires complete Keldysh spectra.")
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    if green.shape != shape or sigma.shape != shape:
        raise ValueError("Research spectra do not match the problem shape.")
    if not np.array_equal(sigma.lesser, lesser) or not np.array_equal(sigma.greater, greater):
        raise ValueError("Research evaluation changed its input self-energy.")
    for value in (raw_lesser, raw_greater):
        if np.shape(value) != shape or not np.all(np.isfinite(value)):
            raise ValueError("Research raw map must contain complete finite arrays.")
    if residual is None or not np.isfinite(residual) or residual < 0:
        raise ValueError("Research evaluation must have a complete finite residual.")
    return evaluation


def _cubic_execution_details(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
) -> tuple[str, str, int]:
    """Return the actual cubic kernel, backend, and worker count."""

    from .sparse_cubic import SparseCubicVertex, cubic_vertex_is_zero

    if cubic_vertex_is_zero(problem.vertex):
        return "zero_vertex", "serial", 1
    if isinstance(problem.vertex, SparseCubicVertex):
        if config.backend == "fft":
            from .sparse_cubic import _sparse_fft_worker_count

            return "sparse_zero_padded_fft", "fft", _sparse_fft_worker_count(config.max_workers)
        # The first sparse integration deliberately routes a threaded request
        # through the deterministic direct oracle.  Metadata records the
        # effective serial execution rather than claiming unused workers.
        return "sparse_grouped_direct", "serial", 1
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

    # Open MPI can expose a low ``SC_AVPHYS_PAGES`` value to ranks even when
    # the host's reclaimable memory is large.  Prefer the Linux kernel's
    # MemAvailable estimate, which is also what the remote qualification guard
    # records, and retain sysconf as a portable fallback.
    try:
        with open("/proc/meminfo", encoding="ascii") as stream:
            for line in stream:
                fields = line.split()
                if len(fields) >= 2 and fields[0] == "MemAvailable:":
                    kilobytes = int(fields[1])
                    if kilobytes > 0:
                        return kilobytes * 1024
                    break
    except (OSError, ValueError):
        pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
    except (AttributeError, OSError, ValueError):
        return None
    if page_size <= 0 or available_pages <= 0:
        return None
    return page_size * available_pages


def _guard_dense_memory(problem: CubicSCBAProblem, config: SCBAConfig) -> int:
    """Apply the conservative full-state memory preflight."""

    from .sparse_cubic import SparseCubicVertex

    stored_state = estimate_dense_memory_bytes(
        problem.grid.n_points,
        problem.ndof,
    )
    if isinstance(problem.vertex, SparseCubicVertex):
        # Green functions, contact spectra, and Dyson operators remain dense;
        # only the cubic Hamiltonian storage changes.  Include its immutable
        # canonical arrays explicitly without materializing D^3 storage.
        stored_state += int(
            problem.vertex.indices.nbytes + problem.vertex.coefficients.nbytes
        )
    _, _, workers = _cubic_execution_details(problem, config)
    contraction_scratch = (
        workers * cubic_fft_scratch_bytes(problem.grid.n_points, problem.ndof)
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
        else (2 * problem.grid.n_points * np.dtype(np.float64).itemsize)
    )
    predicted = stored_state + contraction_scratch + hilbert_scratch
    if config.mixing_method == "anderson":
        # Paired state/residual history, differences and trial/rollback state.
        # Scaled QR additionally keeps up to depth orthogonal field pairs.
        history_arrays = ((10 * config.anderson_depth + 32)
                          if config.anderson_policy == "scaled-trust-v1"
                          else (8 * config.anderson_depth + 28))
        predicted += (
            history_arrays
            * problem.grid.n_points
            * problem.ndof**2
            * np.dtype(np.complex128).itemsize
        )
    available = _available_memory_bytes()
    fraction = float(getattr(config, "max_memory_fraction", 0.70))
    allow = bool(getattr(config, "allow_memory_overcommit", False))
    if available is not None and predicted > fraction * available and not allow:
        storage_label = (
            "Sparse" if isinstance(problem.vertex, SparseCubicVertex) else "Dense"
        )
        raise MemoryError(
            f"{storage_label} SCBA preflight predicts "
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
    from .sparse_cubic import (
        SparseCubicVertex,
        cubic_vertex_is_zero,
        cubic_vertex_max_abs,
    )

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
    sparse_vertex = isinstance(problem.vertex, SparseCubicVertex)
    metadata = {
        "formalism": "cubic_bubble_scba_guo_prb102_195412",
        "certification_contract": (
            "sparse_v0.3_development" if sparse_vertex else "dense_v0.1"
        ),
        "method": method,
        "initialization": initialization,
        "vertex_convention": "H3=(1/3!)*sum(Psi_abc*u_a*u_b*u_c)",
        "green_lesser_convention": ("G^<_{ab}(t,t')=-(i/hbar)<u_b(t')u_a(t)>"),
        "green_greater_convention": ("G^>_{ab}(t,t')=-(i/hbar)<u_a(t)u_b(t')>"),
        "fourier_forward": "F(omega)=integral_dt exp(+i*omega*t) F(t)",
        "fourier_inverse": ("F(t)=integral_domega/(2*pi) exp(-i*omega*t) F(omega)"),
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
            1 if effective_backend == "threaded" or (effective_backend == "fft" and worker_count > 1) else None
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
        "current_atol_physical": (config.current_atol * problem.scale.current_scale),
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
        "vertex_nonzero": not cubic_vertex_is_zero(problem.vertex),
        "vertex_max_abs": cubic_vertex_max_abs(problem.vertex),
        "input_hashes": input_hashes,
        "predicted_peak_memory_bytes": predicted_memory_bytes,
        **provenance,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "blas_runtime": blas_runtime,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if config.mixing_method == "anderson":
        metadata.update(
            mixing_method=config.mixing_method,
            anderson_depth=config.anderson_depth,
            anderson_regularization=config.anderson_regularization,
            anderson_max_step_ratio=config.anderson_max_step_ratio,
            anderson_policy=config.anderson_policy,
            anderson_trust_radius=config.anderson_trust_radius,
            anderson_rank_rtol=config.anderson_rank_rtol,
            anderson_resume_supported=False,
        )
    if sparse_vertex:
        metadata.update(
            {
                "vertex_storage_kind": "sparse",
                "vertex_nnz": problem.vertex.nnz,
                "vertex_support_density": problem.vertex.support_density,
                "vertex_physics_sha256": problem.vertex.cubic_physics_sha256,
                "vertex_storage_sha256": problem.vertex.vertex_storage_sha256,
                "sparse_threaded_execution": (
                    "serial_direct_fallback"
                    if config.backend == "threaded"
                    else "not_requested"
                ),
            }
        )
    return metadata


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
    research_map=None,
    research_evaluation=None,
    research_stop_reason: str | None = None,
    research_moment=None,
) -> AnharmonicResult:
    research_metadata = {}
    research_raw_pair = None
    if research_map is not None:
        if research_evaluation is None or not history:
            raise ValueError("A research result requires a retained evaluated state.")
        final_check_completed = False
        final_stop_reason = None
        try:
            final_evaluation = _research_evaluate(
                research_map, problem, left, right, sigma_pp.lesser,
                sigma_pp.greater, config, purpose="final_diagnostics",
            )
        except ResearchMapBudgetExhausted as exc:
            final_evaluation = research_evaluation
            final_stop_reason = str(exc)
            converged = False
        else:
            final_check_completed = True
            research_map.committed(
                final_evaluation, history[-1], {"step": "final_diagnostics"},
            )
        research_raw_pair = (final_evaluation[2], final_evaluation[3])
        research_metadata = {
            "research_map_identity": _research_identity(research_map),
            "research_map": json.loads(json.dumps(research_map.metadata(), allow_nan=False)),
            "research_stop_reason": research_stop_reason,
            "research_final_stop_reason": final_stop_reason,
            "research_final_check_completed": final_check_completed,
            "research_resume_supported": False,
        }
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
        metadata={**_metadata(
            problem,
            config,
            method=method,
            initialization=initialization,
            initialization_hash=initialization_hash,
            predicted_memory_bytes=predicted_memory_bytes,
        ), **research_metadata},
    )
    diagnostics = (check_solution(problem, provisional, config)
                   if research_map is None else check_solution(
                       problem, provisional, config, research_raw_pair=research_raw_pair,
                   ))
    release_certified = bool(
        converged
        and method == "scba"
        and diagnostics.release_certified
        and provisional.has_release_certifying_provenance
        and provisional.metadata["certification_contract"] == "dense_v0.1"
    )
    result = replace(
        provisional,
        diagnostics=diagnostics,
        release_certified=release_certified and research_map is None,
        converged=converged if research_map is None else converged and diagnostics.passed,
    )
    if research_moment is not None:
        coarse = diagnostics.metrics["green_spectral_moment_residual"]
        if final_check_completed:
            moment_report = research_moment.evaluate(
                problem, config, final_evaluation[1], coarse_residual=coarse,
            )
        else:
            moment_report = research_moment.skip(
                coarse_residual=coarse, reason="fresh_final_map_incomplete",
            )
        result = replace(result, metadata={**result.metadata, "research_moment": moment_report})
    return result


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


def _fixed_point_evaluation(problem, left, right, lesser, greater, config):
    """One full Dyson/bubble evaluation, including the unmixed residual."""

    green, sigma_pp = _solve_green(problem, left, right, lesser, greater, config)
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
    residual = max(
        relative_residual(raw_lesser, lesser, floor=sigma_floor),
        relative_residual(raw_greater, greater, floor=sigma_floor),
        relative_residual(raw_retarded, sigma_pp.retarded, floor=sigma_floor),
    )
    return green, sigma_pp, raw_lesser, raw_greater, residual


def solve_scba(
    problem: CubicSCBAProblem,
    config: SCBAConfig,
    *,
    initial_sigma_pp: KeldyshSpectrum | None = None,
    research_map=None,
    research_moment=None,
) -> AnharmonicResult:
    """Solve the cubic fixed point, optionally with an explicit research map.

    A research object provides JSON-safe ``identity`` and ``metadata()``,
    ``evaluate(problem, left, right, lesser, greater, config, *, purpose)``
    returning the five fields used by ``_fixed_point_evaluation``, and
    ``committed(evaluation, history_record, event)``. It owns the actual map
    evaluation/time budgets and raises ``ResearchMapBudgetExhausted`` for a
    cooperative stop, including an incomplete map. Numerical failures must
    keep their original exceptions. Purpose is ``iteration`` or
    ``final_diagnostics``; all calls must use the same declared map.

    Rejected trials consume that object's budget. A stop retains the last
    controller-committed evaluation, never an unevaluated mixed proposal. A
    final map budget stop still checks its saved raw pair but forbids a
    convergence claim. Research callbacks cannot use resumable checkpoints.

    ``research_moment`` is an optional ``ResearchMomentDiagnostic`` requiring
    an explicit research map. After a completed fresh final map it integrates
    that map's retained input, stores separate metadata and leaf evidence,
    and leaves the historical diagnostics and convergence decision unchanged.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")

    if research_moment is not None:
        from .research_moment import ResearchMomentDiagnostic
        if not isinstance(research_moment, ResearchMomentDiagnostic):
            raise TypeError("research_moment must be a ResearchMomentDiagnostic.")
        if research_map is None:
            raise ValueError("A final research moment diagnostic requires an explicit research_map.")
        research_moment.prepare(problem, config)

    if research_map is not None:
        _research_identity(research_map)
        from ._checkpoint_v2 import _active_solver_checkpoint_session
        if _active_solver_checkpoint_session() is not None:
            raise ValueError("Research maps cannot read or write resumable checkpoints.")
    predicted = _guard_dense_memory(problem, config)
    initialization = "zero" if initial_sigma_pp is None else "supplied_keldysh_pair"
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

    # The v2 sparse-material checkpoint layer is private and context-local.
    # Ordinary calls see ``None`` and retain the established dense/sparse
    # execution path without adding checkpoint controls to this public API.
    from ._checkpoint_v2 import (
        SCBAContinuationState,
        _ANDERSON_DAMPING_CURRENT,
        _active_solver_checkpoint_session,
    )

    checkpoint_session = _active_solver_checkpoint_session()
    resume_state = (
        None
        if checkpoint_session is None
        else checkpoint_session.begin(
            problem,
            config,
            sigma_lesser,
            sigma_greater,
        )
    )

    history: list[SCBAIteration] = []
    previous_green: KeldyshSpectrum | None = None
    previous_sigma_residual: float | None = None
    residual_increase_streak = 0
    mixing = config.mixing
    consecutive = 0
    start_iteration = 1
    if resume_state is not None:
        sigma_lesser = np.array(
            resume_state.sigma_lesser,
            dtype=np.complex128,
            copy=True,
        )
        sigma_greater = np.array(
            resume_state.sigma_greater,
            dtype=np.complex128,
            copy=True,
        )
        history = list(resume_state.history)
        continuation = resume_state.continuation
        mixing = continuation.mixing
        previous_sigma_residual = continuation.previous_sigma_residual
        residual_increase_streak = continuation.residual_increase_streak
        consecutive = continuation.consecutive_passes
        start_iteration = len(history) + 1

        if history:
            # The persisted pair is the mixed input to the last durable
            # iteration. Replaying only its deterministic Dyson/bubble work
            # reconstructs the next mixed pair and previous Green function;
            # the integrity-protected history remains authoritative and is not
            # appended twice.
            replay_green, _ = _solve_green(
                problem,
                left,
                right,
                sigma_lesser,
                sigma_greater,
                config,
            )
            replay_lesser, replay_greater = cubic_lesser_greater(
                problem.vertex,
                replay_green.lesser,
                replay_green.greater,
                problem.grid,
                backend=config.backend,
                max_workers=config.max_workers,
            )
            sigma_lesser = (1.0 - mixing) * sigma_lesser + mixing * replay_lesser
            sigma_greater = (1.0 - mixing) * sigma_greater + mixing * replay_greater
            previous_green = replay_green
        checkpoint_session.replay_complete()

    final_green: KeldyshSpectrum | None = None
    final_sigma: KeldyshSpectrum | None = None
    final_currents: CurrentReport | None = None
    final_evaluation = None
    research_stop_reason = None

    from ._anderson import (
        AndersonHistory, physical_pair_admissible,
        residual_scales, scaled_residual_merit,
    )

    accelerator = (
        AndersonHistory(
            config.anderson_depth,
            config.anderson_regularization,
            config.anderson_max_step_ratio,
            policy=config.anderson_policy,
            trust_radius=config.anderson_trust_radius,
            rank_rtol=config.anderson_rank_rtol,
            absolute_floor=config.sigma_atol,
        )
        if config.mixing_method == "anderson"
        else None
    )
    # Pending proposals keep the previous accepted evaluation, so a rejected
    # trial restores that state without a second expensive map evaluation.
    pending_reference = None
    pending_metric = None
    anderson_events: list[dict[str, Any]] = []

    def algorithm_metadata() -> dict[str, Any]:
        return {
            "mixing_method": "anderson",
            "anderson_resume_supported": False,
            "anderson_damping_controller": _ANDERSON_DAMPING_CURRENT,
            "anderson_policy": config.anderson_policy,
            "anderson": {
                "map_evaluations": len(anderson_events),
                "accepted": sum(
                    e["step"] == "anderson_accepted" for e in anderson_events
                ),
                "rejected": sum(
                    e["step"] == "anderson_rejected" for e in anderson_events
                ),
                "history_records_are_committed_states": True,
                "events": tuple(dict(e) for e in anderson_events),
            },
        }

    for iteration in range(start_iteration, config.max_iterations + 1):
        event: dict[str, Any] = {"iteration": iteration, "step": "linear"}
        rejected_trial = False
        try:
            if research_map is None:
                evaluation = _fixed_point_evaluation(
                    problem, left, right, sigma_lesser, sigma_greater, config
                )
            else:
                evaluation = _research_evaluate(
                    research_map, problem, left, right, sigma_lesser,
                    sigma_greater, config, purpose="iteration",
                )
        except ResearchMapBudgetExhausted as exc:
            if research_map is None or final_evaluation is None:
                raise
            research_stop_reason = str(exc)
            break
        except (
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
            RuntimeError,
        ) as exc:
            if pending_reference is None:
                raise
            if isinstance(exc, RuntimeError) and not isinstance(
                exc.__cause__, np.linalg.LinAlgError
            ):
                raise
            evaluation = None
            event["rejection_reason"] = type(exc).__name__
        if pending_reference is not None:
            trial_residual = None if evaluation is None else evaluation[-1]
            rejected_trial = (
                trial_residual is None
                or not np.isfinite(trial_residual)
                or trial_residual > pending_reference[-1]
            )
            if config.anderson_policy == "scaled-trust-v1":
                scales, reference_merit = pending_metric
                trial_merit = None if evaluation is None else scaled_residual_merit(
                    (evaluation[1].lesser, evaluation[1].greater),
                    (evaluation[2], evaluation[3]), scales,
                )
                if trial_merit is not None and not np.isfinite(trial_merit):
                    trial_merit = None
                rejected_trial = (rejected_trial or trial_merit is None
                                  or trial_merit > reference_merit)
                event.update(reference_merit=reference_merit, trial_merit=trial_merit,
                             merit_scales=list(scales))
            event["trial_sigma_residual"] = (
                trial_residual
                if trial_residual is not None and np.isfinite(trial_residual)
                else None
            )
            event["reference_sigma_residual"] = pending_reference[-1]
            event["step"] = (
                "anderson_rejected" if rejected_trial else "anderson_accepted"
            )
            accelerator.feedback(not rejected_trial)
            if rejected_trial:
                evaluation = pending_reference
                sigma_lesser, sigma_greater = (
                    evaluation[1].lesser,
                    evaluation[1].greater,
                )
                accelerator.clear()
            pending_reference = None
            pending_metric = None
        green, sigma_pp, raw_lesser, raw_greater, sigma_residual = evaluation
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
        current_absolute_floor = config.current_atol * problem.scale.current_scale
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
                currents.left >= -current_allowed and currents.right <= current_allowed
            )
        elif problem.right.temperature > problem.left.temperature:
            direction_valid = bool(
                currents.right >= -current_allowed and currents.left <= current_allowed
            )

        passed = bool(
            not rejected_trial
            and iteration >= config.min_iterations
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
        # A rejected trial restores an evaluation whose damping controller was
        # already advanced when that reference was accepted. Reprocessing its
        # equal residual would erase an increase streak and delay the next
        # reduction, so rollback must leave the controller unchanged.
        if will_continue and not rejected_trial:
            if (
                previous_sigma_residual is not None
                and sigma_residual > previous_sigma_residual * (1.0 + 1.0e-12)
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
        final_evaluation = evaluation
        # Each proposal is evaluated on the next ordinary iteration. Rejected
        # proposals count against the same max_iterations budget, never pass,
        # and leave the durable state at the previous accepted evaluation.
        next_lesser = (1.0 - mixing) * sigma_lesser + mixing * raw_lesser
        next_greater = (1.0 - mixing) * sigma_greater + mixing * raw_greater
        if accelerator is not None:
            if (
                not rejected_trial
                and iteration < config.max_iterations
                and will_continue
            ):
                proposal, reason = accelerator.propose(
                    (sigma_lesser, sigma_greater), (raw_lesser, raw_greater), mixing
                )
                if config.anderson_policy == "scaled-trust-v1":
                    event["proposal_details"] = dict(accelerator.details)
                if proposal is not None:
                    if physical_pair_admissible(proposal, problem.grid):
                        next_lesser, next_greater = proposal
                        pending_reference = evaluation
                        if config.anderson_policy == "scaled-trust-v1":
                            scales = residual_scales(
                                (sigma_lesser, sigma_greater), (raw_lesser, raw_greater),
                                config.sigma_atol,
                            )
                            pending_metric = (scales, scaled_residual_merit(
                                (sigma_lesser, sigma_greater), (raw_lesser, raw_greater),
                                scales,
                            ))
                            event["proposal_details"]["reference_merit"] = pending_metric[1]
                    else:
                        reason = "physical_guard"
                        accelerator.feedback(False)
                        accelerator.clear()
                event["next_proposal"] = reason
            anderson_events.append(event)
            if checkpoint_session is not None:
                checkpoint_session.set_algorithm_metadata(algorithm_metadata())
        if research_map is not None:
            research_map.committed(evaluation, history[-1], dict(event))
        if consecutive >= config.consecutive_passes:
            result = _finalize_result(
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
                **({} if research_map is None else {
                    "research_map": research_map,
                    "research_evaluation": final_evaluation,
                    "research_stop_reason": research_stop_reason,
                    "research_moment": research_moment,
                }),
            )
            if checkpoint_session is not None:
                checkpoint_session.record_terminal(
                    sigma_lesser=sigma_lesser,
                    sigma_greater=sigma_greater,
                    history=tuple(history),
                    continuation=SCBAContinuationState(
                        mixing=mixing,
                        previous_sigma_residual=previous_sigma_residual,
                        residual_increase_streak=residual_increase_streak,
                        consecutive_passes=consecutive,
                    ),
                    status="completed",
                )
            if accelerator is not None:
                result = replace(
                    result, metadata={**result.metadata, **algorithm_metadata()}
                )
            return result

        if checkpoint_session is not None and iteration < config.max_iterations:
            checkpoint_session.record_iteration(
                sigma_lesser=sigma_lesser,
                sigma_greater=sigma_greater,
                history=tuple(history),
                continuation=SCBAContinuationState(
                    mixing=mixing,
                    previous_sigma_residual=previous_sigma_residual,
                    residual_increase_streak=residual_increase_streak,
                    consecutive_passes=consecutive,
                ),
            )

        sigma_lesser, sigma_greater = next_lesser, next_greater
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
        **({} if research_map is None else {
            "research_map": research_map,
            "research_evaluation": final_evaluation,
            "research_stop_reason": research_stop_reason or "iteration_limit",
            "research_moment": research_moment,
        }),
    )
    if accelerator is not None:
        result = replace(result, metadata={**result.metadata, **algorithm_metadata()})
    if checkpoint_session is not None:
        checkpoint_session.record_terminal(
            sigma_lesser=final_sigma.lesser,
            sigma_greater=final_sigma.greater,
            history=tuple(history),
            continuation=SCBAContinuationState(
                mixing=history[-1].mixing,
                previous_sigma_residual=history[-1].sigma_residual,
                residual_increase_streak=residual_increase_streak,
                consecutive_passes=history[-1].consecutive_passes,
            ),
            status="scientific_failure",
            error_type="SCBAConvergenceError",
            error_message=(
                "SCBA exhausted the identically hashed maximum iteration count."
            ),
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


__all__ = ["ResearchMapBudgetExhausted", "solve_one_shot_born", "solve_scba"]
