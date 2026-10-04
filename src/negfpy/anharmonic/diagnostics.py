"""Numerical and physical diagnostics for dense anharmonic NEGF results.

``passed`` and ``release_certified`` deliberately have different meanings.
The former covers algebraic consistency and, for self-consistent Born
calculations, energy conservation.  Certification additionally requires the
symmetry, passivity, cutoff, sum-rule, convergence, and approximation gates
specified for the v0.1 reference solver. This diagnostic certification
is intentionally numerical/physical only; :class:`AnharmonicResult` adds the
clean code-provenance requirement.
"""

from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from ._harmonic import (
    assemble_device_dynamical_matrix,
    harmonic_frequency_upper_bound,
)
from .contacts import bose_occupation_signed, build_contact_spectra
from .cubic import cubic_lesser_greater
from .current import (
    _evaluate_current_report,
    self_energy_nonlocal_ratio,
)
from .grid import SignedFrequencyGrid
from .retarded import retarded_from_keldysh
from .sparse_cubic import cubic_vertex_is_zero
from .types import (
    AnharmonicResult,
    CubicSCBAProblem,
    DiagnosticReport,
    KeldyshSpectrum,
    SCBAConfig,
    SIGMA_PP_BLOCK_LOCAL_RTOL,
)


Array = np.ndarray

_SPECTRAL_MOMENT_RTOL = 5.0e-3
_HIGH_FREQUENCY_RTOL = 5.0e-3
_PASSIVITY_RTOL = 1.0e-10


def _finite_array(values: Array, *, name: str) -> Array:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _matrix_spectrum(
    values: Array,
    *,
    name: str,
    n_frequency: int | None = None,
) -> Array:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError(f"{name} must have shape (n_frequency, ndof, ndof).")
    if n_frequency is not None and array.shape[0] != n_frequency:
        raise ValueError(
            f"{name} frequency dimension must be {n_frequency}; "
            f"got {array.shape[0]}."
        )
    return _finite_array(array, name=name)


def _max_abs(values: Array) -> float:
    array = np.asarray(values)
    if array.size == 0:
        return 0.0
    return float(np.max(np.abs(array)))


def relative_residual(
    actual: Array,
    expected: Array,
    *,
    floor: float = 1.0,
) -> float:
    """Return a scale-aware maximum-norm residual.

    ``floor`` supplies an absolute-error scale when both operands are small.
    With the default dimensionless solver convention, residuals below unity
    are consequently treated as absolute errors rather than being divided by
    numerical noise.
    """

    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError("floor must be finite and strictly positive.")
    left = _finite_array(np.asarray(actual), name="actual")
    right = _finite_array(np.asarray(expected), name="expected")
    if left.shape != right.shape:
        raise ValueError(
            f"actual and expected must have the same shape; "
            f"got {left.shape} and {right.shape}."
        )
    numerator = _max_abs(left - right)
    denominator = max(_max_abs(left), _max_abs(right), float(floor))
    return numerator / denominator


def spectral_function(retarded: Array) -> Array:
    """Return ``A = i (G^r - G^a)`` for a matrix spectrum."""

    values = _matrix_spectrum(retarded, name="retarded")
    advanced = np.swapaxes(values.conj(), -1, -2)
    return 1j * (values - advanced)


def keldysh_identity_residual(spectrum: KeldyshSpectrum) -> float:
    """Measure ``G^> - G^< = G^r - G^a`` (or its self-energy analogue)."""

    if not isinstance(spectrum, KeldyshSpectrum):
        raise TypeError("spectrum must be a KeldyshSpectrum.")
    advanced = np.swapaxes(spectrum.retarded.conj(), -1, -2)
    return relative_residual(
        spectrum.greater - spectrum.lesser,
        spectrum.retarded - advanced,
    )


def spectral_identity_residual(spectrum: KeldyshSpectrum) -> float:
    """Alias emphasizing that the Keldysh difference fixes the spectrum."""

    return keldysh_identity_residual(spectrum)


def antihermitian_residual(values: Array) -> float:
    """Measure the required anti-Hermiticity of a lesser/greater spectrum."""

    matrices = _matrix_spectrum(values, name="values")
    adjoint = np.swapaxes(matrices.conj(), -1, -2)
    return relative_residual(matrices, -adjoint)


def anti_hermitian_residual(values: Array) -> float:
    """Spelling alias for :func:`antihermitian_residual`."""

    return antihermitian_residual(values)


def keldysh_antihermitian_residual(
    spectrum: KeldyshSpectrum,
) -> float:
    """Return the worse anti-Hermiticity residual of ``G^<`` and ``G^>``."""

    if not isinstance(spectrum, KeldyshSpectrum):
        raise TypeError("spectrum must be a KeldyshSpectrum.")
    return max(
        antihermitian_residual(spectrum.lesser),
        antihermitian_residual(spectrum.greater),
    )


def negative_frequency_residual(
    spectrum: KeldyshSpectrum,
    grid: SignedFrequencyGrid,
) -> float:
    """Measure ``[G^>(omega)]^T = G^<(-omega)`` on the signed grid."""

    if not isinstance(spectrum, KeldyshSpectrum):
        raise TypeError("spectrum must be a KeldyshSpectrum.")
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if spectrum.n_frequency != grid.n_points:
        raise ValueError("spectrum frequency dimension does not match grid.")
    lesser_at_negative = spectrum.lesser[grid.negative_partners]
    transposed_greater = np.swapaxes(spectrum.greater, -1, -2)
    return relative_residual(lesser_at_negative, transposed_greater)


def retarded_reality_residual(
    retarded: Array,
    grid: SignedFrequencyGrid,
) -> float:
    """Measure ``G^r(-omega) = G^r(omega)^*`` for real-space IFCs."""

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    values = _matrix_spectrum(
        retarded,
        name="retarded",
        n_frequency=grid.n_points,
    )
    return relative_residual(
        values[grid.negative_partners],
        values.conj(),
    )


def passivity_violation(
    retarded: Array,
    grid: SignedFrequencyGrid,
) -> float:
    """Return the normalized negative-eigenvalue violation of passivity.

    For both a Green function and a retarded self-energy broadening, the
    checked matrix is ``sign(omega) * i * (R - R^dagger)``.  Zero frequency
    is excluded because its sign is zero.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    spectral = spectral_function(
        _matrix_spectrum(
            retarded,
            name="retarded",
            n_frequency=grid.n_points,
        )
    )
    worst_negative = 0.0
    spectral_scale = 0.0
    for index, omega in enumerate(grid.values):
        if omega == 0.0:
            continue
        signed = np.sign(omega) * spectral[index]
        hermitian = 0.5 * (signed + signed.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian)
        spectral_scale = max(
            spectral_scale,
            float(np.max(np.abs(eigenvalues))),
        )
        worst_negative = max(
            worst_negative,
            max(0.0, -float(np.min(eigenvalues))),
        )
    if spectral_scale == 0.0:
        return 0.0
    return worst_negative / spectral_scale


def passivity_residual(
    retarded: Array,
    grid: SignedFrequencyGrid,
) -> float:
    """Alias for :func:`passivity_violation`."""

    return passivity_violation(retarded, grid)


def correlation_positivity_violation(values: Array) -> float:
    """Return the normalized PSD violation of ``i * C``.

    In the displacement convention used here, each lesser or greater
    correlation spectrum is ``-i`` times a positive-semidefinite matrix at
    every signed frequency. This property applies separately to ``C^<`` and
    ``C^>`` even when their difference, and hence an interaction broadening,
    is indefinite in a nonequilibrium steady state.

    The Hermitian part is diagonalized so harmless floating-point
    anti-Hermiticity noise is not mistaken for a negative eigenvalue.
    Anti-Hermiticity itself is checked independently.
    """

    matrices = _matrix_spectrum(values, name="values")
    worst_negative = 0.0
    correlation_scale = 0.0
    for matrix in matrices:
        correlation = 1j * matrix
        hermitian = 0.5 * (correlation + correlation.conj().T)
        eigenvalues = np.linalg.eigvalsh(hermitian)
        correlation_scale = max(
            correlation_scale,
            float(np.max(np.abs(eigenvalues))),
        )
        worst_negative = max(
            worst_negative,
            max(0.0, -float(np.min(eigenvalues))),
        )
    if correlation_scale == 0.0:
        return 0.0
    return worst_negative / correlation_scale


def spectral_moment_residual(
    retarded: Array,
    grid: SignedFrequencyGrid,
) -> float:
    """Measure the phonon sum rule ``integral omega A/(2pi) = I``."""

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    values = _matrix_spectrum(
        retarded,
        name="retarded",
        n_frequency=grid.n_points,
    )
    spectral = spectral_function(values)
    moment = grid.integration_weight * np.einsum(
        "w,wab->ab",
        grid.values,
        spectral,
        optimize=True,
    )
    identity = np.eye(values.shape[1], dtype=np.complex128)
    return relative_residual(moment, identity)


def equilibrium_fdt_residual(
    spectrum: KeldyshSpectrum,
    grid: SignedFrequencyGrid,
    temperature: float,
) -> float:
    """Measure equilibrium fluctuation-dissipation relations off zero.

    With the signed Bose continuation used by the contact implementation,

    ``G^< = n_B (G^r-G^a)`` and
    ``G^> = (1+n_B) (G^r-G^a)``.

    The exactly-zero grid point is excluded because ``n_B(0)`` is singular.
    For the gapped v0.1 problems its Keldysh components are checked separately
    by the spectral identity and anti-Hermiticity diagnostics.
    """

    if not isinstance(spectrum, KeldyshSpectrum):
        raise TypeError("spectrum must be a KeldyshSpectrum.")
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if spectrum.n_frequency != grid.n_points:
        raise ValueError("spectrum frequency dimension does not match grid.")
    if not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("temperature must be finite and non-negative.")

    nonzero = np.arange(grid.n_points) != grid.zero_index
    occupation = np.asarray(
        bose_occupation_signed(grid.values[nonzero], float(temperature)),
        dtype=np.float64,
    )
    retarded = spectrum.retarded[nonzero]
    advanced = np.swapaxes(retarded.conj(), -1, -2)
    discontinuity = retarded - advanced
    expected_lesser = occupation[:, None, None] * discontinuity
    expected_greater = (1.0 + occupation)[:, None, None] * discontinuity
    return max(
        relative_residual(spectrum.lesser[nonzero], expected_lesser),
        relative_residual(spectrum.greater[nonzero], expected_greater),
    )


def high_frequency_residual(
    retarded: Array,
    grid: SignedFrequencyGrid,
    *,
    edge_points: int = 1,
) -> float:
    """Measure the *raw* outer-grid value of ``omega**2 G^r - I``.

    This is a useful descriptive tail-amplitude diagnostic, but it is not an
    asymptotic-coefficient test: a valid Green function has the finite-cutoff
    correction ``D / omega**2 + O(omega**-3)``.  Release certification uses
    :func:`high_frequency_extrapolation_residual` and records this raw value
    separately.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    values = _matrix_spectrum(
        retarded,
        name="retarded",
        n_frequency=grid.n_points,
    )
    if (
        isinstance(edge_points, (bool, np.bool_))
        or not isinstance(edge_points, (int, np.integer))
    ):
        raise TypeError("edge_points must be an integer.")
    edge_points = int(edge_points)
    if edge_points < 1 or edge_points > grid.n_positive:
        raise ValueError(
            "edge_points must be between one and grid.n_positive."
        )

    indices = np.concatenate(
        (
            np.arange(edge_points, dtype=np.int64),
            np.arange(
                grid.n_points - edge_points,
                grid.n_points,
                dtype=np.int64,
            ),
        )
    )
    scaled = (
        grid.values[indices, None, None] ** 2
        * values[indices]
    )
    identity = np.broadcast_to(
        np.eye(values.shape[1], dtype=np.complex128),
        scaled.shape,
    )
    return relative_residual(scaled, identity)


def _high_frequency_extrapolation_metrics(
    retarded: Array,
    grid: SignedFrequencyGrid,
    *,
    fit_fraction: float = 0.125,
    minimum_fit_points: int = 8,
    condition_limit: float = 1.0e6,
    fit_relative_limit: float = 1.0e-2,
    monotonic_rtol: float = 1.0e-8,
) -> dict[str, float]:
    """Fit the high-frequency intercept on both frequency branches.

    The scaled Green function is fitted quadratically against
    ``x = 1 / omega**2``.  The desired asymptotic coefficient is the
    intercept at ``x=0``.  The selected tail must contain enough points,
    approach the identity monotonically, be well conditioned, and be
    represented by the fitted tail model.  Positive and negative branches
    are fitted independently so averaging cannot hide a broken symmetry.
    """

    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    values = _matrix_spectrum(
        retarded,
        name="retarded",
        n_frequency=grid.n_points,
    )
    if (
        not np.isfinite(fit_fraction)
        or fit_fraction <= 0.0
        or fit_fraction > 1.0
    ):
        raise ValueError("fit_fraction must lie in (0, 1].")
    if (
        isinstance(minimum_fit_points, (bool, np.bool_))
        or not isinstance(minimum_fit_points, (int, np.integer))
    ):
        raise TypeError("minimum_fit_points must be an integer.")
    minimum_fit_points = int(minimum_fit_points)
    if minimum_fit_points < 3:
        raise ValueError("minimum_fit_points must be at least three.")
    for name, number in (
        ("condition_limit", condition_limit),
        ("fit_relative_limit", fit_relative_limit),
        ("monotonic_rtol", monotonic_rtol),
    ):
        if not np.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative.")
    if condition_limit <= 1.0:
        raise ValueError("condition_limit must exceed one.")

    n_fit = max(
        minimum_fit_points,
        int(np.ceil(fit_fraction * grid.n_positive)),
    )
    if n_fit > grid.n_positive:
        raise ValueError(
            "high-frequency extrapolation has insufficient positive-frequency "
            f"tail points ({grid.n_positive} available, {n_fit} required)."
        )

    identity = np.eye(values.shape[1], dtype=np.complex128)
    branch_indices = (
        np.asarray(grid.positive_indices[-n_fit:], dtype=np.int64),
        # Reverse the negative edge so |omega| grows along both branches.
        np.arange(n_fit - 1, -1, -1, dtype=np.int64),
    )
    branch_intercepts: list[float] = []
    branch_conditions: list[float] = []
    branch_fit_errors: list[float] = []
    branch_monotonic_violations: list[float] = []

    for indices in branch_indices:
        frequencies = np.abs(grid.values[indices])
        inverse_square = 1.0 / frequencies**2
        # Column scaling preserves the x=0 intercept while avoiding a
        # needlessly ill-conditioned Vandermonde matrix.
        normalized_x = inverse_square / float(np.max(inverse_square))
        design = np.column_stack(
            (
                np.ones(n_fit, dtype=np.float64),
                normalized_x,
                normalized_x**2,
            )
        )
        condition = float(np.linalg.cond(design))
        if not np.isfinite(condition) or condition > condition_limit:
            raise ValueError(
                "high-frequency extrapolation design is ill-conditioned "
                f"(condition={condition:.6e}, limit={condition_limit:.6e})."
            )

        scaled = frequencies[:, None, None] ** 2 * values[indices]
        raw_errors = np.max(
            np.abs(scaled - identity[None, :, :]),
            axis=(1, 2),
        )
        raw_scale = max(float(np.max(raw_errors)), np.finfo(float).tiny)
        increases = np.diff(raw_errors)
        monotonic_violation = max(
            0.0,
            float(np.max(increases, initial=0.0)),
        )
        monotonic_allowance = monotonic_rtol * raw_scale
        if monotonic_violation > monotonic_allowance:
            raise ValueError(
                "high-frequency tail does not approach the identity "
                "monotonically "
                f"(increase={monotonic_violation:.6e}, "
                f"allowance={monotonic_allowance:.6e})."
            )

        flattened = scaled.reshape(n_fit, -1)
        coefficients, _, rank, _ = np.linalg.lstsq(
            design,
            flattened,
            rcond=None,
        )
        if rank != 3:
            raise ValueError(
                "high-frequency extrapolation design is rank-deficient."
            )
        fitted = (design @ coefficients).reshape(scaled.shape)
        fit_absolute = _max_abs(fitted - scaled)
        fit_relative = fit_absolute / raw_scale
        if not np.isfinite(fit_relative) or fit_relative > fit_relative_limit:
            raise ValueError(
                "high-frequency extrapolation does not represent the selected "
                f"tail (relative fit residual={fit_relative:.6e}, "
                f"limit={fit_relative_limit:.6e})."
            )

        intercept = coefficients[0].reshape(identity.shape)
        branch_intercepts.append(relative_residual(intercept, identity))
        branch_conditions.append(condition)
        branch_fit_errors.append(fit_relative)
        branch_monotonic_violations.append(
            monotonic_violation / raw_scale
        )

    return {
        "residual": max(branch_intercepts),
        "condition_number": max(branch_conditions),
        "fit_relative_residual": max(branch_fit_errors),
        "monotonicity_violation": max(branch_monotonic_violations),
        "fit_points_per_branch": float(n_fit),
    }


def high_frequency_extrapolation_residual(
    retarded: Array,
    grid: SignedFrequencyGrid,
    *,
    fit_fraction: float = 0.125,
    minimum_fit_points: int = 8,
) -> float:
    """Return the fitted ``omega -> infinity`` coefficient residual.

    Unlike :func:`high_frequency_residual`, this removes the expected finite
    ``1 / omega**2`` and ``1 / omega**4`` corrections.  Invalid or unresolved
    tails raise :class:`ValueError` rather than returning optimistic evidence.
    """

    metrics = _high_frequency_extrapolation_metrics(
        retarded,
        grid,
        fit_fraction=fit_fraction,
        minimum_fit_points=minimum_fit_points,
    )
    return metrics["residual"]


def tail_ratio(values: Array, *, edge_points: int = 1) -> float:
    """Return the outer-frequency amplitude divided by the global amplitude."""

    array = _finite_array(np.asarray(values), name="values")
    if array.ndim < 1 or array.shape[0] < 1:
        raise ValueError("values must have a non-empty frequency axis.")
    if (
        isinstance(edge_points, (bool, np.bool_))
        or not isinstance(edge_points, (int, np.integer))
    ):
        raise TypeError("edge_points must be an integer.")
    edge_points = int(edge_points)
    if edge_points < 1 or edge_points > array.shape[0]:
        raise ValueError("edge_points must lie within the frequency axis.")

    outer = np.concatenate(
        (array[:edge_points].reshape(-1), array[-edge_points:].reshape(-1))
    )
    scale = _max_abs(array)
    if scale == 0.0:
        return 0.0
    return _max_abs(outer) / scale


def keldysh_tail_ratio(
    spectrum: KeldyshSpectrum,
    *,
    edge_points: int = 1,
) -> float:
    """Return the worse lesser/greater cutoff-tail ratio.

    The retarded Hilbert term is intentionally excluded: its physical
    dispersive tail need not have compact support even when the scattering
    spectrum is fully contained by the frequency grid.
    """

    if not isinstance(spectrum, KeldyshSpectrum):
        raise TypeError("spectrum must be a KeldyshSpectrum.")
    return max(
        tail_ratio(spectrum.lesser, edge_points=edge_points),
        tail_ratio(spectrum.greater, edge_points=edge_points),
    )


def _append_threshold_failure(
    failures: list[str],
    *,
    name: str,
    value: float,
    limit: float,
    release_only: bool,
) -> None:
    if value <= limit:
        return
    prefix = "release: " if release_only else ""
    failures.append(
        f"{prefix}{name}={value:.6e} exceeds {limit:.6e}"
    )


def _configured_field_residual(
    actual: Array,
    expected: Array,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    """Return a residual and threshold equivalent to max(rtol*scale, atol)."""

    if rtol > 0.0:
        floor = max(atol / rtol, np.finfo(float).tiny)
        return relative_residual(actual, expected, floor=floor), rtol
    return _max_abs(np.asarray(actual) - np.asarray(expected)), atol


def _record_field_check(
    metrics: dict[str, float],
    failures: list[str],
    *,
    metric_name: str,
    display_name: str,
    actual: Array,
    expected: Array,
    rtol: float,
    atol: float,
) -> None:
    if np.asarray(actual).shape != np.asarray(expected).shape:
        metrics[metric_name] = float("inf")
        failures.append(
            f"{display_name} shape {np.asarray(actual).shape} does not match "
            f"recomputed shape {np.asarray(expected).shape}"
        )
        return
    residual, limit = _configured_field_residual(
        actual,
        expected,
        rtol=rtol,
        atol=atol,
    )
    metrics[metric_name] = residual
    _append_threshold_failure(
        failures,
        name=display_name,
        value=residual,
        limit=limit,
        release_only=False,
    )


def _record_optional_current_field(
    metrics: dict[str, float],
    failures: list[str],
    *,
    name: str,
    actual: Array | object | None,
    expected: Array | object | None,
    rtol: float,
    atol: float,
) -> None:
    if (actual is None) != (expected is None):
        failures.append(f"stored {name} availability does not match recomputation")
        metrics[f"recomputed_{name}_residual"] = float("inf")
        return
    if actual is None:
        metrics[f"recomputed_{name}_residual"] = 0.0
        return
    if isinstance(actual, Mapping) or isinstance(expected, Mapping):
        failures.append(f"stored {name} has an unsupported mapping representation")
        metrics[f"recomputed_{name}_residual"] = float("inf")
        return
    _record_field_check(
        metrics,
        failures,
        metric_name=f"recomputed_{name}_residual",
        display_name=f"stored {name}",
        actual=np.asarray(actual),
        expected=np.asarray(expected),
        rtol=rtol,
        atol=atol,
    )


def _record_optional_spatial_current(
    metrics: dict[str, float],
    failures: list[str],
    *,
    actual: object | None,
    expected: object | None,
    rtol: float,
    atol: float,
) -> None:
    if (actual is None) != (expected is None):
        failures.append(
            "stored spatial-current availability does not match recomputation"
        )
        metrics["recomputed_spatial_current_residual"] = float("inf")
        return
    if actual is None:
        metrics["recomputed_spatial_current_residual"] = 0.0
        return
    field_names = (
        "sigma_pp_nonlocal_ratio",
        "eom_exchange_spectra",
        "collision_power_spectra",
        "eom_layer_exchange",
        "layer_collision_power",
        "nonlocal_self_energy_eom_cut_correction",
        "corrected_cut_currents",
        "continuity_residual",
    )
    residuals: list[float] = []
    for name in field_names:
        metric_name = f"recomputed_spatial_{name}_residual"
        _record_field_check(
            metrics,
            failures,
            metric_name=metric_name,
            display_name=f"stored spatial current {name}",
            actual=np.asarray(getattr(actual, name)),
            expected=np.asarray(getattr(expected, name)),
            rtol=rtol,
            atol=atol,
        )
        residuals.append(metrics[metric_name])
    metrics["recomputed_spatial_current_residual"] = max(residuals)


def check_solution(
    problem: CubicSCBAProblem,
    result: AnharmonicResult,
    config: SCBAConfig,
) -> DiagnosticReport:
    """Evaluate essential and release-certification gates.

    The existing ``result.diagnostics`` field is ignored.  This permits a
    solver to construct an immutable provisional result, evaluate it here,
    and then return a final result carrying this report.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if not isinstance(result, AnharmonicResult):
        raise TypeError("result must be an AnharmonicResult.")
    if not isinstance(config, SCBAConfig):
        raise TypeError("config must be an SCBAConfig.")
    if not np.array_equal(problem.grid.values, result.grid.values):
        raise ValueError("result and problem must use the same frequency grid.")
    if result.green.ndof != problem.ndof:
        raise ValueError("result spectrum dimension does not match the problem.")

    spectra = {
        "green": result.green,
        "sigma_pp": result.sigma_pp,
        "sigma_left": result.sigma_left,
        "sigma_right": result.sigma_right,
    }
    metrics: dict[str, float] = {}
    essential_failures: list[str] = []
    release_failures: list[str] = []
    method = result.method.strip().lower()
    temperature_scale = max(
        1.0,
        abs(problem.left.temperature),
        abs(problem.right.temperature),
    )
    equal_temperature = (
        abs(problem.left.temperature - problem.right.temperature)
        <= 1.0e-14 * temperature_scale
    )
    zero_vertex = cubic_vertex_is_zero(problem.vertex)
    interacting_retarded_passivity_applicable = (
        equal_temperature or zero_vertex
    )
    metrics["equilibrium_fdt_applicable"] = float(equal_temperature)
    metrics["zero_vertex"] = float(zero_vertex)
    metrics["interacting_retarded_passivity_applicable"] = float(
        interacting_retarded_passivity_applicable
    )

    # Certification never trusts cached contacts, a supplied retarded
    # interaction self-energy, convergence history, or summarized currents.
    # Rebuild all of them from the problem and the stored lesser/greater
    # fixed-point fields.
    rebuilt_left, rebuilt_right = build_contact_spectra(
        problem,
        config.eta_lead,
        surface_gf_method=config.surface_gf_method,
    )
    for label, stored, rebuilt in (
        ("sigma_left", result.sigma_left, rebuilt_left),
        ("sigma_right", result.sigma_right, rebuilt_right),
    ):
        for component in ("retarded", "lesser", "greater"):
            _record_field_check(
                metrics,
                essential_failures,
                metric_name=f"recomputed_{label}_{component}_residual",
                display_name=f"stored {label} {component}",
                actual=getattr(stored, component),
                expected=getattr(rebuilt, component),
                rtol=config.identity_rtol,
                atol=config.identity_rtol,
            )

    rebuilt_sigma_pp_retarded = retarded_from_keldysh(
        result.sigma_pp.lesser,
        result.sigma_pp.greater,
        result.grid,
        strategy=config.retarded_strategy,
        hilbert_backend=config.hilbert_backend,
    )
    _record_field_check(
        metrics,
        essential_failures,
        metric_name="recomputed_sigma_pp_retarded_residual",
        display_name="stored interaction retarded reconstruction",
        actual=result.sigma_pp.retarded,
        expected=rebuilt_sigma_pp_retarded,
        rtol=config.sigma_rtol,
        atol=config.sigma_atol,
    )

    dynamical = assemble_device_dynamical_matrix(problem.device)
    identity = np.eye(problem.ndof, dtype=np.complex128)
    operator = (
        result.grid.values[:, None, None] ** 2 * identity[None, :, :]
        - dynamical[None, :, :]
        - rebuilt_left.retarded
        - rebuilt_right.retarded
        - rebuilt_sigma_pp_retarded
    )
    dyson_product = operator @ result.green.retarded
    dyson_target = np.broadcast_to(identity, dyson_product.shape)
    _record_field_check(
        metrics,
        essential_failures,
        metric_name="recomputed_dyson_residual",
        display_name="retarded Dyson equation",
        actual=dyson_product,
        expected=dyson_target,
        rtol=config.identity_rtol,
        atol=config.identity_rtol,
    )

    advanced = np.swapaxes(result.green.retarded.conj(), -1, -2)
    total_lesser = (
        rebuilt_left.lesser
        + rebuilt_right.lesser
        + result.sigma_pp.lesser
    )
    total_greater = (
        rebuilt_left.greater
        + rebuilt_right.greater
        + result.sigma_pp.greater
    )
    total_self_energy = KeldyshSpectrum(
        rebuilt_left.retarded
        + rebuilt_right.retarded
        + rebuilt_sigma_pp_retarded,
        total_lesser,
        total_greater,
    )
    expected_green_lesser = (
        result.green.retarded @ total_lesser @ advanced
    )
    expected_green_greater = (
        result.green.retarded @ total_greater @ advanced
    )
    for component, actual, expected in (
        (
            "lesser",
            result.green.lesser,
            expected_green_lesser,
        ),
        (
            "greater",
            result.green.greater,
            expected_green_greater,
        ),
    ):
        _record_field_check(
            metrics,
            essential_failures,
            metric_name=f"recomputed_green_{component}_residual",
            display_name=f"Green-function {component} Keldysh equation",
            actual=actual,
            expected=expected,
            rtol=config.green_rtol,
            atol=config.green_atol,
        )

    rebuilt_sigma_pp = KeldyshSpectrum(
        rebuilt_sigma_pp_retarded,
        result.sigma_pp.lesser,
        result.sigma_pp.greater,
    )
    rebuilt_currents = _evaluate_current_report(
        problem,
        rebuilt_left,
        rebuilt_right,
        rebuilt_sigma_pp,
        result.green,
    )
    physical_current_atol = (
        config.current_atol * problem.scale.current_scale
    )
    for name in ("left", "right", "collision_power"):
        _record_field_check(
            metrics,
            essential_failures,
            metric_name=f"recomputed_current_{name}_residual",
            display_name=f"stored current {name}",
            actual=np.asarray(getattr(result.currents, name)),
            expected=np.asarray(getattr(rebuilt_currents, name)),
            rtol=config.current_rtol,
            atol=physical_current_atol,
        )
    for name in (
        "contact_spectra",
        "collision_spectrum",
        "bond_currents",
        "bond_spectra",
    ):
        _record_optional_current_field(
            metrics,
            essential_failures,
            name=name,
            actual=getattr(result.currents, name),
            expected=getattr(rebuilt_currents, name),
            rtol=config.current_rtol,
            atol=physical_current_atol,
        )
    _record_optional_spatial_current(
        metrics,
        essential_failures,
        actual=result.currents.spatial,
        expected=rebuilt_currents.spatial,
        rtol=config.current_rtol,
        atol=physical_current_atol,
    )

    if method == "scba":
        recomputation_backend = (
            "fft" if config.backend == "fft" else "serial"
        )
        raw_lesser, raw_greater = cubic_lesser_greater(
            problem.vertex,
            result.green.lesser,
            result.green.greater,
            result.grid,
            backend=recomputation_backend,
            max_workers=config.max_workers,
        )
        raw_retarded = retarded_from_keldysh(
            raw_lesser,
            raw_greater,
            result.grid,
            strategy=config.retarded_strategy,
            hilbert_backend=config.hilbert_backend,
        )
        for component, actual, expected in (
            ("lesser", result.sigma_pp.lesser, raw_lesser),
            ("greater", result.sigma_pp.greater, raw_greater),
            ("retarded", result.sigma_pp.retarded, raw_retarded),
        ):
            _record_field_check(
                metrics,
                essential_failures,
                metric_name=f"recomputed_scba_{component}_residual",
                display_name=f"SCBA fixed-point {component} self-energy",
                actual=actual,
                expected=expected,
                rtol=config.sigma_rtol,
                atol=config.sigma_atol,
            )

    for label, spectrum in spectra.items():
        keldysh = keldysh_identity_residual(spectrum)
        lesser_anti = antihermitian_residual(spectrum.lesser)
        greater_anti = antihermitian_residual(spectrum.greater)
        negative = negative_frequency_residual(spectrum, result.grid)
        reality = retarded_reality_residual(
            spectrum.retarded,
            result.grid,
        )
        passivity = passivity_violation(
            spectrum.retarded,
            result.grid,
        )
        lesser_positivity = correlation_positivity_violation(
            spectrum.lesser
        )
        greater_positivity = correlation_positivity_violation(
            spectrum.greater
        )
        tails = keldysh_tail_ratio(spectrum)

        metrics[f"{label}_keldysh_residual"] = keldysh
        metrics[f"{label}_spectral_identity_residual"] = keldysh
        metrics[f"{label}_lesser_antihermitian_residual"] = lesser_anti
        metrics[f"{label}_greater_antihermitian_residual"] = greater_anti
        metrics[f"{label}_negative_frequency_residual"] = negative
        metrics[f"{label}_retarded_reality_residual"] = reality
        metrics[f"{label}_passivity_violation"] = passivity
        metrics[
            f"{label}_lesser_correlation_positivity_violation"
        ] = lesser_positivity
        metrics[
            f"{label}_greater_correlation_positivity_violation"
        ] = greater_positivity
        metrics[
            f"{label}_lesser_correlation_positivity_applicable"
        ] = 1.0
        metrics[
            f"{label}_greater_correlation_positivity_applicable"
        ] = 1.0
        metrics[f"{label}_correlation_positivity_applicable"] = 1.0
        metrics[f"{label}_correlation_positivity_violation"] = max(
            lesser_positivity,
            greater_positivity,
        )
        metrics[f"{label}_tail_ratio"] = tails

        for name, value in (
            (f"{label} Keldysh identity", keldysh),
            (f"{label} lesser anti-Hermiticity", lesser_anti),
            (f"{label} greater anti-Hermiticity", greater_anti),
        ):
            _append_threshold_failure(
                essential_failures,
                name=name,
                value=value,
                limit=config.identity_rtol,
                release_only=False,
            )

        for name, value, limit in (
            (
                f"{label} negative-frequency symmetry",
                negative,
                config.identity_rtol,
            ),
            (
                f"{label} retarded reality symmetry",
                reality,
                config.identity_rtol,
            ),
            (
                f"{label} lesser correlation positivity",
                lesser_positivity,
                _PASSIVITY_RTOL,
            ),
            (
                f"{label} greater correlation positivity",
                greater_positivity,
                _PASSIVITY_RTOL,
            ),
        ):
            _append_threshold_failure(
                release_failures,
                name=name,
                value=value,
                limit=limit,
                release_only=True,
            )

        passivity_applicable = (
            label in ("sigma_left", "sigma_right")
            or interacting_retarded_passivity_applicable
        )
        metrics[f"{label}_passivity_applicable"] = float(
            passivity_applicable
        )
        if passivity_applicable:
            _append_threshold_failure(
                release_failures,
                name=f"{label} passivity",
                value=passivity,
                limit=_PASSIVITY_RTOL,
                release_only=True,
            )

    total_passivity = passivity_violation(
        total_self_energy.retarded,
        result.grid,
    )
    total_lesser_positivity = correlation_positivity_violation(
        total_self_energy.lesser
    )
    total_greater_positivity = correlation_positivity_violation(
        total_self_energy.greater
    )
    metrics["sigma_total_passivity_violation"] = total_passivity
    metrics["sigma_total_passivity_applicable"] = float(
        interacting_retarded_passivity_applicable
    )
    metrics[
        "sigma_total_lesser_correlation_positivity_violation"
    ] = total_lesser_positivity
    metrics[
        "sigma_total_greater_correlation_positivity_violation"
    ] = total_greater_positivity
    metrics[
        "sigma_total_lesser_correlation_positivity_applicable"
    ] = 1.0
    metrics[
        "sigma_total_greater_correlation_positivity_applicable"
    ] = 1.0
    metrics["sigma_total_correlation_positivity_applicable"] = 1.0
    metrics["sigma_total_correlation_positivity_violation"] = max(
        total_lesser_positivity,
        total_greater_positivity,
    )
    if interacting_retarded_passivity_applicable:
        _append_threshold_failure(
            release_failures,
            name="sigma_total passivity",
            value=total_passivity,
            limit=_PASSIVITY_RTOL,
            release_only=True,
        )
    for name, value in (
        (
            "sigma_total lesser correlation positivity",
            total_lesser_positivity,
        ),
        (
            "sigma_total greater correlation positivity",
            total_greater_positivity,
        ),
    ):
        _append_threshold_failure(
            release_failures,
            name=name,
            value=value,
            limit=_PASSIVITY_RTOL,
            release_only=True,
        )
    moment = spectral_moment_residual(result.green.retarded, result.grid)
    metrics["green_spectral_moment_residual"] = moment
    _append_threshold_failure(
        release_failures,
        name="Green-function spectral moment",
        value=moment,
        limit=_SPECTRAL_MOMENT_RTOL,
        release_only=True,
    )
    raw_high_frequency = high_frequency_residual(
        result.green.retarded,
        result.grid,
    )
    metrics["green_high_frequency_residual"] = raw_high_frequency
    try:
        extrapolation = _high_frequency_extrapolation_metrics(
            result.green.retarded,
            result.grid,
        )
    except ValueError as error:
        metrics["green_high_frequency_extrapolated_residual"] = float("inf")
        metrics["green_high_frequency_fit_condition"] = float("inf")
        metrics["green_high_frequency_fit_relative_residual"] = float("inf")
        metrics["green_high_frequency_tail_monotonicity_violation"] = float(
            "inf"
        )
        metrics["green_high_frequency_fit_points_per_branch"] = 0.0
        release_failures.append(
            "release: high-frequency extrapolation is invalid: "
            f"{error}"
        )
    else:
        extrapolated_high_frequency = extrapolation["residual"]
        metrics[
            "green_high_frequency_extrapolated_residual"
        ] = extrapolated_high_frequency
        metrics["green_high_frequency_fit_condition"] = extrapolation[
            "condition_number"
        ]
        metrics[
            "green_high_frequency_fit_relative_residual"
        ] = extrapolation["fit_relative_residual"]
        metrics[
            "green_high_frequency_tail_monotonicity_violation"
        ] = extrapolation["monotonicity_violation"]
        metrics[
            "green_high_frequency_fit_points_per_branch"
        ] = extrapolation["fit_points_per_branch"]
        _append_threshold_failure(
            release_failures,
            name="Green-function high-frequency extrapolated coefficient",
            value=extrapolated_high_frequency,
            limit=_HIGH_FREQUENCY_RTOL,
            release_only=True,
        )
    harmonic_upper_bound = harmonic_frequency_upper_bound(problem)
    cutoff_ratio = (
        result.grid.cutoff / harmonic_upper_bound
        if harmonic_upper_bound > 0.0
        else float("inf")
    )
    metrics["harmonic_frequency_upper_bound"] = harmonic_upper_bound
    metrics["cutoff_to_harmonic_upper_bound"] = cutoff_ratio
    if cutoff_ratio < 3.0:
        release_failures.append(
            "release: frequency cutoff / conservative harmonic upper bound="
            f"{cutoff_ratio:.6e} is below 3.000000e+00"
        )
    _append_threshold_failure(
        release_failures,
        name="interaction self-energy cutoff tail",
        value=metrics["sigma_pp_tail_ratio"],
        limit=config.tail_rtol,
        release_only=True,
    )

    current = result.currents
    current_absolute_floor = (
        config.current_atol * problem.scale.current_scale
    )
    current_scale = max(abs(current.left), abs(current.right))
    current_tolerance = max(
        current_absolute_floor,
        config.current_rtol * current_scale,
    )
    current_denominator = max(current_scale, current_absolute_floor)
    contact_abs = abs(current.contact_balance)
    collision_abs = abs(current.collision_power)
    full_abs = abs(current.full_balance)
    metrics["current_contact_abs_error"] = contact_abs
    metrics["current_contact_rel_error"] = contact_abs / current_denominator
    metrics["current_collision_abs_error"] = collision_abs
    metrics["current_collision_rel_error"] = collision_abs / current_denominator
    metrics["current_full_abs_error"] = full_abs
    metrics["current_full_rel_error"] = full_abs / current_denominator
    metrics["current_tolerance"] = current_tolerance
    metrics["current_atol_physical"] = current_absolute_floor
    sigma_pp_nonlocal_ratio = self_energy_nonlocal_ratio(
        problem,
        rebuilt_sigma_pp,
    )
    metrics["sigma_pp_nonlocal_ratio"] = sigma_pp_nonlocal_ratio
    metrics["sigma_pp_block_local_threshold"] = (
        SIGMA_PP_BLOCK_LOCAL_RTOL
    )
    metrics["sigma_pp_is_block_local"] = float(
        sigma_pp_nonlocal_ratio <= SIGMA_PP_BLOCK_LOCAL_RTOL
    )


    # Each ideal contact is internally equilibrated even when the two
    # reservoirs have different temperatures.
    for label, spectrum, temperature in (
        ("sigma_left", result.sigma_left, problem.left.temperature),
        ("sigma_right", result.sigma_right, problem.right.temperature),
    ):
        fdt = equilibrium_fdt_residual(
            spectrum,
            result.grid,
            temperature,
        )
        metrics[f"{label}_equilibrium_fdt_residual"] = fdt
        _append_threshold_failure(
            essential_failures,
            name=f"{label} equilibrium FDT",
            value=fdt,
            limit=config.identity_rtol,
            release_only=False,
        )

    if equal_temperature:
        temperature = 0.5 * (
            problem.left.temperature + problem.right.temperature
        )
        for label, spectrum in (
            ("green", result.green),
            ("sigma_pp", result.sigma_pp),
        ):
            fdt = equilibrium_fdt_residual(
                spectrum,
                result.grid,
                temperature,
            )
            metrics[f"{label}_equilibrium_fdt_residual"] = fdt
            _append_threshold_failure(
                essential_failures,
                name=f"{label} equilibrium FDT",
                value=fdt,
                limit=config.identity_rtol,
                release_only=False,
            )

    if method not in {"scba", "born"}:
        essential_failures.append(
            f"unsupported anharmonic result method {result.method!r}"
        )
    if method == "scba":
        for name, value in (
            ("contact current balance", contact_abs),
            ("interaction collision power", collision_abs),
            ("full energy balance", full_abs),
        ):
            _append_threshold_failure(
                essential_failures,
                name=name,
                value=value,
                limit=current_tolerance,
                release_only=False,
            )

        equilibrium_current = (
            max(abs(current.left), abs(current.right))
            if equal_temperature
            else 0.0
        )
        metrics["equilibrium_current_abs_error"] = equilibrium_current
        if equal_temperature:
            _append_threshold_failure(
                essential_failures,
                name="equilibrium contact current",
                value=equilibrium_current,
                limit=current_absolute_floor,
                release_only=False,
            )

        if not result.history:
            essential_failures.append(
                "SCBA convergence history is empty"
            )
        else:
            last = result.history[-1]
            metrics["history_last_consecutive_passes"] = float(
                last.consecutive_passes
            )
            metrics["history_last_sigma_residual"] = last.sigma_residual
            metrics["history_last_green_residual"] = last.green_residual
            if last.consecutive_passes < config.consecutive_passes:
                essential_failures.append(
                    "SCBA history has "
                    f"{last.consecutive_passes} consecutive passes; "
                    f"{config.consecutive_passes} required"
                )
            _append_threshold_failure(
                essential_failures,
                name="last SCBA sigma residual",
                value=last.sigma_residual,
                limit=config.sigma_rtol,
                release_only=False,
            )
            _append_threshold_failure(
                essential_failures,
                name="last SCBA Green-function residual",
                value=last.green_residual,
                limit=config.green_rtol,
                release_only=False,
            )
            required = config.consecutive_passes
            if len(result.history) < required:
                essential_failures.append(
                    "SCBA history contains only "
                    f"{len(result.history)} records; {required} consecutive "
                    "passing records are required"
                )
            else:
                tail = result.history[-required:]
                expected_iterations = tuple(
                    range(
                        tail[0].iteration,
                        tail[0].iteration + required,
                    )
                )
                actual_iterations = tuple(
                    record.iteration for record in tail
                )
                if actual_iterations != expected_iterations:
                    essential_failures.append(
                        "SCBA terminal history iteration numbers are not sequential"
                    )
                for offset, record in enumerate(tail, start=1):
                    if not record.passed:
                        essential_failures.append(
                            f"SCBA history iteration {record.iteration} is not marked passing"
                        )
                    if record.consecutive_passes != offset:
                        essential_failures.append(
                            "SCBA history iteration "
                            f"{record.iteration} records consecutive_passes="
                            f"{record.consecutive_passes}; expected {offset}"
                        )
                    for field_name, value, limit in (
                        (
                            "sigma residual",
                            record.sigma_residual,
                            config.sigma_rtol,
                        ),
                        (
                            "Green-function residual",
                            record.green_residual,
                            config.green_rtol,
                        ),
                        (
                            "identity residual",
                            record.identity_residual,
                            config.identity_rtol,
                        ),
                        (
                            "contact current error",
                            record.current_abs_error,
                            current_tolerance,
                        ),
                        (
                            "collision power",
                            abs(record.collision_power),
                            current_tolerance,
                        ),
                    ):
                        _append_threshold_failure(
                            essential_failures,
                            name=(
                                f"history iteration {record.iteration} "
                                f"{field_name}"
                            ),
                            value=value,
                            limit=limit,
                            release_only=False,
                        )

        if current.bond_currents is not None:
            bonds = np.asarray(current.bond_currents, dtype=np.float64).reshape(
                -1
            )
            bond_spread = (
                float(np.max(bonds) - np.min(bonds))
                if bonds.size > 1
                else 0.0
            )
            bond_scale = max(
                _max_abs(bonds),
                abs(current.left),
                abs(current.right),
            )
            bond_tolerance = max(
                current_absolute_floor,
                config.current_rtol * bond_scale,
            )
            bond_denominator = max(bond_scale, current_absolute_floor)
            raw_bond_gate_applicable = (
                sigma_pp_nonlocal_ratio <= SIGMA_PP_BLOCK_LOCAL_RTOL
            )
            metrics["harmonic_bond_current_gate_applicable"] = float(
                raw_bond_gate_applicable
            )
            metrics["bond_current_abs_spread"] = bond_spread
            metrics["bond_current_rel_spread"] = (
                bond_spread / bond_denominator
            )
            metrics["bond_current_tolerance"] = bond_tolerance
            metrics["harmonic_bond_current_abs_spread"] = bond_spread
            if bonds.size:
                bond_contact_error = max(
                    float(np.max(np.abs(bonds - current.left))),
                    float(np.max(np.abs(bonds + current.right))),
                )
                metrics["bond_current_contact_abs_error"] = bond_contact_error
                metrics[
                    "harmonic_bond_current_contact_abs_error"
                ] = bond_contact_error
            else:
                bond_contact_error = 0.0
                metrics["bond_current_contact_abs_error"] = 0.0
                metrics["harmonic_bond_current_contact_abs_error"] = 0.0
            if raw_bond_gate_applicable:
                _append_threshold_failure(
                    essential_failures,
                    name="block-local harmonic bond-current spread",
                    value=bond_spread,
                    limit=bond_tolerance,
                    release_only=False,
                )
                _append_threshold_failure(
                    essential_failures,
                    name="block-local harmonic bond-to-contact mismatch",
                    value=bond_contact_error,
                    limit=bond_tolerance,
                    release_only=False,
                )

        spatial = current.spatial
        if spatial is not None:
            metrics["spatial_sigma_pp_nonlocal_ratio"] = (
                spatial.sigma_pp_nonlocal_ratio
            )
            layer_collision = np.asarray(
                spatial.layer_collision_power,
                dtype=np.float64,
            )
            layer_collision_abs = _max_abs(layer_collision)
            eom_continuity_abs = _max_abs(spatial.continuity_residual)
            eom_global_error = abs(
                float(np.sum(spatial.eom_layer_exchange))
                - current.collision_power
            )
            collision_global_error = abs(
                float(np.sum(layer_collision))
                - current.collision_power
            )
            corrected = np.asarray(
                spatial.corrected_cut_currents,
                dtype=np.float64,
            )
            corrected_spread = (
                float(np.max(corrected) - np.min(corrected))
                if corrected.size > 1
                else 0.0
            )
            corrected_contact_error = (
                max(
                    float(np.max(np.abs(corrected - current.left))),
                    float(np.max(np.abs(corrected + current.right))),
                )
                if corrected.size
                else 0.0
            )
            metrics["spatial_layer_collision_abs_error"] = (
                layer_collision_abs
            )
            metrics["spatial_eom_continuity_abs_error"] = (
                eom_continuity_abs
            )
            metrics["spatial_eom_global_collision_abs_error"] = (
                eom_global_error
            )
            metrics["spatial_projected_collision_global_abs_error"] = (
                collision_global_error
            )
            metrics["spatial_corrected_cut_abs_spread"] = corrected_spread
            metrics["spatial_corrected_cut_contact_abs_error"] = (
                corrected_contact_error
            )
            nonlocal_cut_gate_applicable = (
                sigma_pp_nonlocal_ratio > SIGMA_PP_BLOCK_LOCAL_RTOL
            )
            metrics["spatial_nonlocal_eom_cut_gate_applicable"] = float(
                nonlocal_cut_gate_applicable
            )
            for name, value in (
                ("projected EOM continuity", eom_continuity_abs),
                ("projected EOM global collision identity", eom_global_error),
                (
                    "projected greater/lesser global collision identity",
                    collision_global_error,
                ),
            ):
                _append_threshold_failure(
                    essential_failures,
                    name=name,
                    value=value,
                    limit=current_tolerance,
                    release_only=False,
                )
            if nonlocal_cut_gate_applicable:
                for name, value in (
                    (
                        "nonlocal-self-energy EOM corrected cut spread",
                        corrected_spread,
                    ),
                    (
                        "nonlocal-self-energy EOM corrected cut-to-contact mismatch",
                        corrected_contact_error,
                    ),
                ):
                    _append_threshold_failure(
                        essential_failures,
                        name=name,
                        value=value,
                        limit=current_tolerance,
                        release_only=False,
                    )

        direction_violation = 0.0
        if problem.left.temperature > problem.right.temperature:
            direction_violation = max(
                0.0,
                -current.left,
                current.right,
            )
        elif problem.right.temperature > problem.left.temperature:
            direction_violation = max(
                0.0,
                -current.right,
                current.left,
            )
        metrics["current_direction_violation"] = direction_violation
        _append_threshold_failure(
            essential_failures,
            name="hot-to-cold current direction",
            value=direction_violation,
            limit=current_tolerance,
            release_only=False,
        )
    else:
        metrics["equilibrium_current_abs_error"] = 0.0

    if method != "scba":
        release_failures.append(
            "release: only a self-consistent Born result can be certified"
        )
    if not result.converged:
        release_failures.append(
            "release: the SCBA fixed point is not converged"
        )
    if config.retarded_strategy != "causal_hilbert":
        release_failures.append(
            "release: causal_hilbert retarded reconstruction is required"
        )
    for name, value, ceiling in (
        ("sigma_rtol", config.sigma_rtol, 1.0e-8),
        ("sigma_atol", config.sigma_atol, 1.0e-12),
        ("green_rtol", config.green_rtol, 1.0e-8),
        ("green_atol", config.green_atol, 1.0e-12),
        ("identity_rtol", config.identity_rtol, 1.0e-8),
        ("current_rtol", config.current_rtol, 1.0e-5),
        ("current_atol", config.current_atol, 1.0e-10),
        ("tail_rtol", config.tail_rtol, 1.0e-6),
    ):
        if value > ceiling:
            release_failures.append(
                f"release: {name}={value:.6e} exceeds the dense-v0.1 "
                f"certification ceiling {ceiling:.6e}"
            )
    if config.min_iterations < 3:
        release_failures.append(
            "release: min_iterations must be at least 3"
        )
    if config.consecutive_passes < 3:
        release_failures.append(
            "release: at least 3 consecutive passing iterations are required"
        )
    if not bool(
        problem.scale.metadata.get("source_units_declared", False)
    ):
        release_failures.append(
            "release: physical scale source units must be declared"
        )
    # Retain provenance visibility as a descriptive metric. It does not enter
    # this numerical/physics-only diagnostic certification decision.
    package_version_consistent = (
        result.metadata.get("package_version_consistent") is True
    )
    metrics["package_version_consistent"] = float(
        package_version_consistent
    )
    passed = not essential_failures
    release_certified = passed and not release_failures
    return DiagnosticReport(
        metrics=metrics,
        failures=tuple(essential_failures + release_failures),
        passed=passed,
        release_certified=release_certified,
    )


__all__ = [
    "anti_hermitian_residual",
    "antihermitian_residual",
    "check_solution",
    "correlation_positivity_violation",
    "equilibrium_fdt_residual",
    "high_frequency_extrapolation_residual",
    "high_frequency_residual",
    "keldysh_antihermitian_residual",
    "keldysh_identity_residual",
    "keldysh_tail_ratio",
    "negative_frequency_residual",
    "passivity_residual",
    "passivity_violation",
    "relative_residual",
    "retarded_reality_residual",
    "spectral_function",
    "spectral_identity_residual",
    "spectral_moment_residual",
    "tail_ratio",
]
