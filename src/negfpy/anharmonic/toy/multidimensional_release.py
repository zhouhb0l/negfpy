"""Multidimensional dense-toy SCBA benchmark and release-gate harness.

This module supplies one deliberately isolated vertical benchmark for the
three-polarization intralayer cubic model.  It does not alter the production
solver or the generic dense-v0.1 refinement report.

Two execution profiles are provided:

``pr_coarse``
    A small, explicitly non-certifying subset used to exercise the complete
    run graph in pull-request CI.
``full_preregistered``
    The preregistered physical and numerical ladders for a release/nightly
    calculation.  It is intentionally expensive and is never run implicitly.

Every requested point produces an outcome.  Solver failures and complete
nonconverged partial results are retained; no point is omitted, retried with
different controls, or silently replaced by a cold start.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import numpy as np

from .._checkpoint import (
    CheckpointError,
    SCBACheckpoint,
    SCBACheckpointIdentity,
    load_scba_checkpoint,
    make_scba_checkpoint_identity,
    restore_scba_checkpoint_result,
    write_scba_checkpoint,
)
from ..diagnostics import keldysh_tail_ratio
from ..exceptions import SCBAConvergenceError
from ..grid import SignedFrequencyGrid
from ..solver import solve_one_shot_born, solve_scba
from ..types import (
    AnharmonicResult,
    CubicSCBAProblem,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
)
from .cubic_chain import pinned_three_polarization_chain


Array = np.ndarray
RunStatus = Literal[
    "converged",
    "nonconverged_partial",
    "failed",
    "blocked_by_parent",
]
Initialization = Literal[
    "zero",
    "born_seed",
    "continuation",
    "refinement_zero",
]
ComparisonKind = Literal[
    "zero_vs_born",
    "direct_vs_continuation",
    "serial_vs_threaded",
    "numerical_refinement",
]

_WEAK_COUPLING_LIMIT = 0.1
_REQUIRED_INPUT_HASHES = frozenset(
    {
        "device",
        "left_reservoir",
        "right_reservoir",
        "cubic_vertex",
        "frequency_grid",
        "solver_scale",
        "problem",
        "configuration",
        "initial_state",
        "calculation",
    }
)
_PROVENANCE_IDENTITY_KEYS = (
    "declared_package_version",
    "installed_package_version",
    "git_revision",
    "source_tree_sha256",
    "dtype",
    "numpy_version",
    "scipy_version",
    "python",
    "platform",
    "blas_runtime",
)


def _finite(value: Any, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if nonnegative and result < 0.0:
        raise ValueError(f"{name} must be non-negative.")
    return result


def _positive(value: Any, *, name: str) -> float:
    result = _finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one.")
    return result


def _float_tuple(
    values: Any,
    *,
    name: str,
    length: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of real numbers.")
    try:
        raw = tuple(values)
    except TypeError as error:
        raise TypeError(
            f"{name} must be a sequence of real numbers."
        ) from error
    if length is not None and len(raw) != length:
        raise ValueError(f"{name} must contain exactly {length} values.")
    if not raw:
        raise ValueError(f"{name} must not be empty.")
    validator = (
        _positive
        if positive
        else (
            lambda value, *, name: _finite(
                value,
                name=name,
                nonnegative=nonnegative,
            )
        )
    )
    return tuple(
        validator(value, name=f"{name}[{index}]")
        for index, value in enumerate(raw)
    )


def _strict_float(value: Any) -> float | str:
    """Return a JSON-standard representation without NaN/Infinity tokens."""

    result = float(value)
    if np.isnan(result):
        return "NaN"
    if np.isposinf(result):
        return "Infinity"
    if np.isneginf(result):
        return "-Infinity"
    return result


def _slug(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _array_sha256(values: Array) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _spectrum_hashes(
    spectrum: KeldyshSpectrum,
    *,
    prefix: str,
) -> dict[str, str]:
    return {
        f"{prefix}_{component}": _array_sha256(
            getattr(spectrum, component)
        )
        for component in ("retarded", "lesser", "greater")
    }


@dataclass(frozen=True, slots=True)
class MultidimensionalLadderSpec:
    """Immutable preregistration of the three-mode release benchmark."""

    polarization_frequencies: tuple[float, float, float] = (1.5, 0.9, 0.6)
    masses: tuple[float, float, float] = (1.0, 1.0, 1.0)
    springs: tuple[float, float, float] = (0.1, 0.1, 0.1)
    pinnings: tuple[float, float, float] = (2.05, 0.61, 0.16)
    g_values: tuple[float, ...] = (0.0, 0.025, 0.05)
    mean_temperature: float = 0.7
    bias_values: tuple[float, ...] = (0.0, 0.1, 0.2)
    spacings: tuple[float, ...] = (0.05, 0.025, 0.0125)
    cutoffs: tuple[float, ...] = (5.0, 6.0)
    etas: tuple[float, ...] = (0.025, 0.0125, 0.00625)
    mixings: tuple[float, ...] = (0.2, 0.1)

    def __post_init__(self) -> None:
        frequencies = _float_tuple(
            self.polarization_frequencies,
            name="polarization_frequencies",
            length=3,
            positive=True,
        )
        masses = _float_tuple(
            self.masses,
            name="masses",
            length=3,
            positive=True,
        )
        springs = _float_tuple(
            self.springs,
            name="springs",
            length=3,
            positive=True,
        )
        pinnings = _float_tuple(
            self.pinnings,
            name="pinnings",
            length=3,
            positive=True,
        )
        expected_pinnings = tuple(
            frequency**2 - 2.0 * spring
            for frequency, spring in zip(frequencies, springs, strict=True)
        )
        if not np.allclose(
            pinnings,
            expected_pinnings,
            rtol=0.0,
            atol=5.0e-15,
        ):
            raise ValueError(
                "pinnings must equal polarization_frequency**2 - 2*spring."
            )

        g_values = _float_tuple(
            self.g_values,
            name="g_values",
            nonnegative=True,
        )
        biases = _float_tuple(
            self.bias_values,
            name="bias_values",
            nonnegative=True,
        )
        for name, values in (
            ("g_values", g_values),
            ("bias_values", biases),
        ):
            if values[0] != 0.0 or any(
                right <= left for left, right in zip(values, values[1:])
            ):
                raise ValueError(
                    f"{name} must start at zero and be strictly increasing."
                )
        mean_temperature = _positive(
            self.mean_temperature,
            name="mean_temperature",
        )
        if biases[-1] > 2.0 * mean_temperature:
            raise ValueError(
                "the largest bias would make the right temperature negative."
            )

        spacings = _float_tuple(
            self.spacings,
            name="spacings",
            positive=True,
        )
        cutoffs = _float_tuple(
            self.cutoffs,
            name="cutoffs",
            positive=True,
        )
        etas = _float_tuple(self.etas, name="etas", positive=True)
        mixings = _float_tuple(
            self.mixings,
            name="mixings",
            positive=True,
        )
        if any(value > 1.0 for value in mixings):
            raise ValueError("mixings cannot exceed one.")
        for name, values in (
            ("spacings", spacings),
            ("cutoffs", cutoffs),
            ("etas", etas),
            ("mixings", mixings),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"{name} must not contain duplicates.")

        object.__setattr__(self, "polarization_frequencies", frequencies)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "springs", springs)
        object.__setattr__(self, "pinnings", pinnings)
        object.__setattr__(self, "g_values", g_values)
        object.__setattr__(self, "mean_temperature", mean_temperature)
        object.__setattr__(self, "bias_values", biases)
        object.__setattr__(self, "spacings", spacings)
        object.__setattr__(self, "cutoffs", cutoffs)
        object.__setattr__(self, "etas", etas)
        object.__setattr__(self, "mixings", mixings)

    def as_dict(self) -> dict[str, object]:
        return {
            "polarization_order": ["parent_x", "child1_y", "child2_z"],
            "polarization_frequencies": list(self.polarization_frequencies),
            "masses": list(self.masses),
            "springs": list(self.springs),
            "pinnings": list(self.pinnings),
            "interaction": "g*u_x*u_y*u_z on every layer",
            "g_values": list(self.g_values),
            "mean_temperature": self.mean_temperature,
            "bias_values": list(self.bias_values),
            "temperature_definition": (
                "T_left=T_mean+DeltaT/2; T_right=T_mean-DeltaT/2"
            ),
            "spacings": list(self.spacings),
            "cutoffs": list(self.cutoffs),
            "etas": list(self.etas),
            "mixings": list(self.mixings),
            "frequency_kind": "angular",
            "units": "hbar=k_B=omega0=1",
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalNumericalPoint:
    """One exact grid/regulator/mixing point."""

    spacing: float
    cutoff: float
    eta_lead: float
    mixing: float

    def __post_init__(self) -> None:
        spacing = _positive(self.spacing, name="spacing")
        cutoff = _positive(self.cutoff, name="cutoff")
        eta = _positive(self.eta_lead, name="eta_lead")
        mixing = _positive(self.mixing, name="mixing")
        if mixing > 1.0:
            raise ValueError("mixing cannot exceed one.")
        ratio = cutoff / spacing
        nearest = round(ratio)
        if nearest < 1 or not np.isclose(
            ratio,
            nearest,
            rtol=0.0,
            atol=2.0e-12,
        ):
            raise ValueError(
                "cutoff/spacing must be an integer so the signed grid "
                "contains the requested cutoff exactly."
            )
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "cutoff", cutoff)
        object.__setattr__(self, "eta_lead", eta)
        object.__setattr__(self, "mixing", mixing)

    @property
    def n_positive(self) -> int:
        return int(round(self.cutoff / self.spacing))

    @property
    def n_frequency(self) -> int:
        return 2 * self.n_positive + 1

    def as_dict(self) -> dict[str, object]:
        return {
            "spacing": self.spacing,
            "cutoff": self.cutoff,
            "n_positive": self.n_positive,
            "n_frequency": self.n_frequency,
            "eta_lead": self.eta_lead,
            "eta_device": 0.0,
            "mixing": self.mixing,
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalReleaseConfig:
    """Immutable execution controls for one benchmark report."""

    profile: Literal["pr_coarse", "full_preregistered"]
    spec: MultidimensionalLadderSpec
    executed_g_values: tuple[float, ...]
    executed_bias_values: tuple[float, ...]
    numerical: MultidimensionalNumericalPoint
    max_iterations: int
    min_iterations: int
    consecutive_passes: int
    sigma_rtol: float
    sigma_atol: float
    green_rtol: float
    green_atol: float
    identity_rtol: float
    current_rtol: float
    current_atol: float
    tail_rtol: float
    mixing_floor: float = 0.01
    max_workers: int = 2
    field_rtol: float = 1.0e-8
    field_atol: float = 1.0e-12
    backend_field_rtol: float = 1.0e-10
    backend_field_atol: float = 1.0e-12
    current_comparison_rtol: float = 1.0e-10
    current_comparison_atol: float = 1.0e-12
    refinement_rtol: float = 5.0e-3
    line_shape_rtol: float = 5.0e-3
    linewidth_atol: float = 1.0e-12
    shift_atol: float = 1.0e-8
    run_preregistered_refinements: bool = False
    certification_eligible: bool = False

    def __post_init__(self) -> None:
        if self.profile not in {"pr_coarse", "full_preregistered"}:
            raise ValueError(
                "profile must be 'pr_coarse' or 'full_preregistered'."
            )
        if not isinstance(self.spec, MultidimensionalLadderSpec):
            raise TypeError("spec must be a MultidimensionalLadderSpec.")
        if not isinstance(self.numerical, MultidimensionalNumericalPoint):
            raise TypeError(
                "numerical must be a MultidimensionalNumericalPoint."
            )
        executed_g = _float_tuple(
            self.executed_g_values,
            name="executed_g_values",
            nonnegative=True,
        )
        executed_bias = _float_tuple(
            self.executed_bias_values,
            name="executed_bias_values",
            nonnegative=True,
        )
        for name, executed, registered in (
            ("executed_g_values", executed_g, self.spec.g_values),
            (
                "executed_bias_values",
                executed_bias,
                self.spec.bias_values,
            ),
        ):
            if executed[0] != 0.0 or any(
                right <= left
                for left, right in zip(executed, executed[1:])
            ):
                raise ValueError(
                    f"{name} must start at zero and be strictly increasing."
                )
            if any(value not in registered for value in executed):
                raise ValueError(
                    f"{name} must be a subset of the preregistered ladder."
                )
        for name in (
            "max_iterations",
            "min_iterations",
            "consecutive_passes",
            "max_workers",
        ):
            object.__setattr__(
                self,
                name,
                _positive_int(getattr(self, name), name=name),
            )
        if self.min_iterations > self.max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations.")
        for name in (
            "sigma_rtol",
            "sigma_atol",
            "green_rtol",
            "green_atol",
            "identity_rtol",
            "current_rtol",
            "current_atol",
            "tail_rtol",
            "mixing_floor",
            "field_rtol",
            "field_atol",
            "backend_field_rtol",
            "backend_field_atol",
            "current_comparison_rtol",
            "current_comparison_atol",
            "refinement_rtol",
            "line_shape_rtol",
            "linewidth_atol",
            "shift_atol",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name=name),
            )
        if self.mixing_floor > self.numerical.mixing:
            raise ValueError("mixing_floor cannot exceed selected mixing.")
        if self.profile == "pr_coarse" and self.certification_eligible:
            raise ValueError("the coarse PR profile cannot be certifying.")
        if self.certification_eligible:
            if self.profile != "full_preregistered":
                raise ValueError(
                    "certification eligibility requires the full profile."
                )
            if executed_g != self.spec.g_values:
                raise ValueError(
                    "certification eligibility requires the full g ladder."
                )
            if executed_bias != self.spec.bias_values:
                raise ValueError(
                    "certification eligibility requires the full bias ladder."
                )
            if not self.run_preregistered_refinements:
                raise ValueError(
                    "certification eligibility requires numerical "
                    "refinements."
                )
        object.__setattr__(self, "executed_g_values", executed_g)
        object.__setattr__(self, "executed_bias_values", executed_bias)
        object.__setattr__(
            self,
            "run_preregistered_refinements",
            bool(self.run_preregistered_refinements),
        )
        object.__setattr__(
            self,
            "certification_eligible",
            bool(self.certification_eligible),
        )

    @property
    def coarse(self) -> bool:
        return self.profile == "pr_coarse"

    @property
    def full_physical_ladder_selected(self) -> bool:
        return (
            self.executed_g_values == self.spec.g_values
            and self.executed_bias_values == self.spec.bias_values
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "coarse": self.coarse,
            "certification_eligible": self.certification_eligible,
            "spec": self.spec.as_dict(),
            "executed_g_values": list(self.executed_g_values),
            "executed_bias_values": list(self.executed_bias_values),
            "full_physical_ladder_selected": (
                self.full_physical_ladder_selected
            ),
            "physical_execution_path": {
                "kind": "L_shaped_continuation",
                "definition": (
                    "all selected g values at DeltaT=0, followed by all "
                    "remaining selected DeltaT values at max(g)"
                ),
                "cartesian_sweep": False,
                "points": [
                    {"g": g_value, "bias": bias_value}
                    for g_value, bias_value in _physical_path(self)
                ],
            },
            "numerical": self.numerical.as_dict(),
            "solver": {
                "max_iterations": self.max_iterations,
                "min_iterations": self.min_iterations,
                "consecutive_passes": self.consecutive_passes,
                "sigma_rtol": self.sigma_rtol,
                "sigma_atol": self.sigma_atol,
                "green_rtol": self.green_rtol,
                "green_atol": self.green_atol,
                "identity_rtol": self.identity_rtol,
                "current_rtol": self.current_rtol,
                "current_atol": self.current_atol,
                "tail_rtol": self.tail_rtol,
                "mixing_floor": self.mixing_floor,
                "retarded_strategy": "causal_hilbert",
                "eta_device": 0.0,
            },
            "comparison": {
                "field_rtol": self.field_rtol,
                "field_atol": self.field_atol,
                "fixed_point_field_rtol": self.field_rtol,
                "fixed_point_field_atol": self.field_atol,
                "backend_field_rtol": self.backend_field_rtol,
                "backend_field_atol": self.backend_field_atol,
                "current_rtol": self.current_comparison_rtol,
                "current_atol": self.current_comparison_atol,
                "refinement_rtol": self.refinement_rtol,
                "line_shape_rtol": self.line_shape_rtol,
                "linewidth_atol": self.linewidth_atol,
                "shift_atol": self.shift_atol,
                "line_shape_observable": {
                    "dof": "center_layer.parent_x",
                    "target_angular_frequency": (
                        self.spec.polarization_frequencies[0]
                    ),
                    "linewidth": (
                        "-Im[Sigma_pp^r_xx(Omega_p)]/Omega_p"
                    ),
                    "shift": (
                        "Re[Sigma_pp^r_xx(Omega_p)]/(2*Omega_p)"
                    ),
                    "interpolation": (
                        "piecewise-linear on the signed-frequency grid"
                    ),
                },
            },
            "max_workers": self.max_workers,
            "run_preregistered_refinements": (
                self.run_preregistered_refinements
            ),
        }


def pr_multidimensional_release_config() -> MultidimensionalReleaseConfig:
    """Return the bounded, deliberately non-certifying PR profile."""

    return MultidimensionalReleaseConfig(
        profile="pr_coarse",
        spec=MultidimensionalLadderSpec(),
        executed_g_values=(0.0, 0.025),
        executed_bias_values=(0.0, 0.1),
        numerical=MultidimensionalNumericalPoint(
            spacing=0.5,
            cutoff=5.0,
            eta_lead=0.025,
            mixing=0.4,
        ),
        max_iterations=30,
        min_iterations=2,
        consecutive_passes=2,
        sigma_rtol=1.0e-5,
        sigma_atol=1.0e-9,
        green_rtol=1.0e-5,
        green_atol=1.0e-9,
        identity_rtol=1.0e-5,
        current_rtol=5.0e-2,
        current_atol=1.0e-8,
        tail_rtol=2.0e-1,
        mixing_floor=0.05,
        max_workers=2,
        field_rtol=5.0e-4,
        field_atol=1.0e-9,
        current_comparison_rtol=5.0e-4,
        current_comparison_atol=1.0e-9,
        refinement_rtol=5.0e-2,
        line_shape_rtol=5.0e-2,
        linewidth_atol=1.0e-9,
        shift_atol=1.0e-7,
        run_preregistered_refinements=False,
        certification_eligible=False,
    )


def full_multidimensional_release_config() -> MultidimensionalReleaseConfig:
    """Return the exact expensive preregistered release/nightly profile."""

    spec = MultidimensionalLadderSpec()
    return MultidimensionalReleaseConfig(
        profile="full_preregistered",
        spec=spec,
        executed_g_values=spec.g_values,
        executed_bias_values=spec.bias_values,
        numerical=MultidimensionalNumericalPoint(
            spacing=0.025,
            cutoff=5.0,
            eta_lead=0.0125,
            mixing=0.2,
        ),
        max_iterations=200,
        min_iterations=3,
        consecutive_passes=3,
        sigma_rtol=1.0e-8,
        sigma_atol=1.0e-12,
        green_rtol=1.0e-8,
        green_atol=1.0e-12,
        identity_rtol=1.0e-8,
        current_rtol=1.0e-5,
        current_atol=1.0e-10,
        tail_rtol=1.0e-6,
        mixing_floor=0.01,
        max_workers=2,
        field_rtol=1.0e-8,
        field_atol=1.0e-12,
        backend_field_rtol=1.0e-10,
        backend_field_atol=1.0e-12,
        current_comparison_rtol=1.0e-10,
        current_comparison_atol=1.0e-12,
        refinement_rtol=5.0e-3,
        line_shape_rtol=5.0e-3,
        linewidth_atol=1.0e-12,
        shift_atol=1.0e-8,
        run_preregistered_refinements=True,
        certification_eligible=True,
    )


def build_multidimensional_problem(
    config: MultidimensionalReleaseConfig,
    *,
    g: float,
    bias: float,
    numerical: MultidimensionalNumericalPoint | None = None,
) -> CubicSCBAProblem:
    """Build the fixed three-layer/nine-DOF production benchmark."""

    if not isinstance(config, MultidimensionalReleaseConfig):
        raise TypeError("config must be a MultidimensionalReleaseConfig.")
    g_value = _finite(g, name="g", nonnegative=True)
    bias_value = _finite(bias, name="bias", nonnegative=True)
    if g_value not in config.spec.g_values:
        raise ValueError("g must be a preregistered coupling value.")
    if bias_value not in config.spec.bias_values:
        raise ValueError("bias must be a preregistered temperature bias.")
    selected = config.numerical if numerical is None else numerical
    if not isinstance(selected, MultidimensionalNumericalPoint):
        raise TypeError(
            "numerical must be a MultidimensionalNumericalPoint or None."
        )

    left_temperature = config.spec.mean_temperature + 0.5 * bias_value
    right_temperature = config.spec.mean_temperature - 0.5 * bias_value
    if right_temperature < 0.0:
        raise ValueError("bias makes the right temperature negative.")
    device, lead, vertex = pinned_three_polarization_chain(
        3,
        mass=config.spec.masses,
        spring=config.spec.springs,
        pinning=config.spec.pinnings,
        g=g_value,
    )
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, left_temperature),
        right=HarmonicReservoir(lead, right_temperature),
        vertex=vertex,
        grid=SignedFrequencyGrid(
            step=selected.spacing,
            n_positive=selected.n_positive,
        ),
    )


@dataclass(frozen=True, slots=True)
class MultidimensionalRunPoint:
    """Physical and numerical coordinates of one requested solve."""

    g: float
    bias: float
    numerical: MultidimensionalNumericalPoint

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "g",
            _finite(self.g, name="g", nonnegative=True),
        )
        object.__setattr__(
            self,
            "bias",
            _finite(self.bias, name="bias", nonnegative=True),
        )
        if not isinstance(self.numerical, MultidimensionalNumericalPoint):
            raise TypeError(
                "numerical must be a MultidimensionalNumericalPoint."
            )

    def as_dict(self, mean_temperature: float) -> dict[str, object]:
        return {
            "g": self.g,
            "bias": self.bias,
            "mean_temperature": mean_temperature,
            "left_temperature": mean_temperature + 0.5 * self.bias,
            "right_temperature": mean_temperature - 0.5 * self.bias,
            "numerical": self.numerical.as_dict(),
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalRunOutcome:
    """One retained successful, partial, failed, or blocked calculation."""

    run_id: str
    point: MultidimensionalRunPoint
    initialization: Initialization
    backend: Literal["serial", "threaded"]
    continuation_parent_id: str | None
    elapsed_seconds: float
    seed_elapsed_seconds: float
    status: RunStatus
    result: AnharmonicResult | None
    error_type: str | None = None
    error_message: str | None = None
    checkpoint_path: str | None = None
    checkpoint_file_sha256: str | None = None
    checkpoint_identity: SCBACheckpointIdentity | None = None
    resumed_from_checkpoint: bool = False
    reconstruction_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("run_id must be a non-empty string.")
        if not isinstance(self.point, MultidimensionalRunPoint):
            raise TypeError("point must be a MultidimensionalRunPoint.")
        if self.initialization not in {
            "zero",
            "born_seed",
            "continuation",
            "refinement_zero",
        }:
            raise ValueError("unknown initialization.")
        if self.backend not in {"serial", "threaded"}:
            raise ValueError("backend must be serial or threaded.")
        if self.continuation_parent_id is not None and (
            not isinstance(self.continuation_parent_id, str)
            or not self.continuation_parent_id
        ):
            raise ValueError(
                "continuation_parent_id must be a non-empty string or None."
            )
        object.__setattr__(
            self,
            "elapsed_seconds",
            _finite(
                self.elapsed_seconds,
                name="elapsed_seconds",
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "seed_elapsed_seconds",
            _finite(
                self.seed_elapsed_seconds,
                name="seed_elapsed_seconds",
                nonnegative=True,
            ),
        )
        if self.status not in {
            "converged",
            "nonconverged_partial",
            "failed",
            "blocked_by_parent",
        }:
            raise ValueError("unknown run status.")
        if self.result is not None and not isinstance(
            self.result,
            AnharmonicResult,
        ):
            raise TypeError("result must be an AnharmonicResult or None.")
        if self.status == "converged":
            if self.result is None or not self.result.converged:
                raise ValueError(
                    "converged status requires a converged result."
                )
        if self.status == "nonconverged_partial":
            if self.result is None or self.result.converged:
                raise ValueError(
                    "nonconverged_partial requires a nonconverged result."
                )
        if self.status in {"failed", "blocked_by_parent"}:
            if self.result is not None:
                raise ValueError(
                    "failed or blocked outcomes cannot contain a result."
                )
        if self.status != "converged" and (
            not self.error_type or not self.error_message
        ):
            raise ValueError(
                "non-converged outcomes require error type and message."
            )

        object.__setattr__(
            self,
            "reconstruction_seconds",
            _finite(
                self.reconstruction_seconds,
                name="reconstruction_seconds",
                nonnegative=True,
            ),
        )
        checkpoint_values = (
            self.checkpoint_path,
            self.checkpoint_file_sha256,
            self.checkpoint_identity,
        )
        if any(value is not None for value in checkpoint_values):
            if any(value is None for value in checkpoint_values):
                raise ValueError(
                    "checkpoint path, file SHA256, and identity must be "
                    "recorded together."
                )
            assert self.checkpoint_file_sha256 is not None
            if not _is_sha256(self.checkpoint_file_sha256):
                raise ValueError(
                    "checkpoint_file_sha256 must be a lowercase SHA256."
                )
            if not isinstance(
                self.checkpoint_identity,
                SCBACheckpointIdentity,
            ):
                raise TypeError(
                    "checkpoint_identity must be an SCBACheckpointIdentity."
                )
        if self.resumed_from_checkpoint and self.checkpoint_path is None:
            raise ValueError(
                "resumed outcomes require complete checkpoint evidence."
            )
        if self.reconstruction_seconds and not self.resumed_from_checkpoint:
            raise ValueError(
                "reconstruction time is valid only for resumed outcomes."
            )
        object.__setattr__(
            self,
            "resumed_from_checkpoint",
            bool(self.resumed_from_checkpoint),
        )

    @property
    def retained_partial(self) -> bool:
        return self.status == "nonconverged_partial"

    @property
    def successful(self) -> bool:
        return self.status == "converged"

    def as_dict(self, mean_temperature: float) -> dict[str, object]:
        result = self.result
        document: dict[str, object] = {
            "run_id": self.run_id,
            "point": self.point.as_dict(mean_temperature),
            "initialization": self.initialization,
            "backend": self.backend,
            "continuation_parent_id": self.continuation_parent_id,
            "elapsed_seconds": self.elapsed_seconds,
            "seed_elapsed_seconds": self.seed_elapsed_seconds,
            "status": self.status,
            "retained_partial": self.retained_partial,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "checkpoint": {
                "enabled": self.checkpoint_path is not None,
                "path": self.checkpoint_path,
                "file_sha256": self.checkpoint_file_sha256,
                "identity": (
                    None
                    if self.checkpoint_identity is None
                    else {
                        **self.checkpoint_identity.as_dict(),
                        "identity_sha256": (
                            self.checkpoint_identity.identity_sha256
                        ),
                    }
                ),
                "resumed": self.resumed_from_checkpoint,
                "reconstruction_seconds": self.reconstruction_seconds,
            },
            "result": None,
        }
        if result is None:
            return document

        field_hashes = {
            **_spectrum_hashes(result.green, prefix="green"),
            **_spectrum_hashes(result.sigma_pp, prefix="sigma_pp"),
        }
        input_hashes = result.metadata.get("input_hashes", {})
        if not isinstance(input_hashes, Mapping):
            input_hashes = {}
        history = [
            {
                "iteration": item.iteration,
                "sigma_residual": _strict_float(item.sigma_residual),
                "green_residual": _strict_float(item.green_residual),
                "identity_residual": _strict_float(item.identity_residual),
                "current_abs_error": _strict_float(
                    item.current_abs_error
                ),
                "current_rel_error": _strict_float(
                    item.current_rel_error
                ),
                "collision_power": _strict_float(item.collision_power),
                "mixing": item.mixing,
                "passed": item.passed,
                "consecutive_passes": item.consecutive_passes,
            }
            for item in result.history
        ]
        currents = result.currents
        bond_currents = (
            None
            if currents.bond_currents is None
            else np.asarray(currents.bond_currents, dtype=float).tolist()
        )
        document["result"] = {
            "method": result.method,
            "converged": result.converged,
            "diagnostics_passed": result.diagnostics.passed,
            "result_release_certified": result.release_certified,
            "diagnostics_release_certified": (
                result.diagnostics.release_certified
            ),
            "diagnostic_failures": list(result.diagnostics.failures),
            "diagnostic_metrics": {
                key: _strict_float(value)
                for key, value in sorted(
                    result.diagnostics.metrics.items()
                )
            },
            "iterations": len(result.history),
            "history": history,
            "currents": {
                "left": currents.left,
                "right": currents.right,
                "contact_balance": currents.contact_balance,
                "collision_power": currents.collision_power,
                "full_balance": currents.full_balance,
                "bond_currents": bond_currents,
            },
            "sigma_tail_ratio": _strict_float(
                keldysh_tail_ratio(result.sigma_pp)
            ),
            "field_sha256": field_hashes,
            "input_hashes": {
                str(key): str(value)
                for key, value in sorted(input_hashes.items())
            },
            "source_tree_sha256": result.metadata.get(
                "source_tree_sha256"
            ),
            "package_version": result.metadata.get("package_version"),
            "provenance": _result_provenance(result),
            "initial_state_sha256": result.metadata.get(
                "input_hashes",
                {},
            ).get("initial_state")
            if isinstance(result.metadata.get("input_hashes"), Mapping)
            else None,
        }
        return document


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def _result_provenance(result: AnharmonicResult) -> dict[str, object]:
    """Return independently checked source, input, and environment identity."""

    metadata = result.metadata
    raw_hashes = metadata.get("input_hashes")
    input_hashes = (
        {str(key): str(value) for key, value in raw_hashes.items()}
        if isinstance(raw_hashes, Mapping)
        else {}
    )
    input_complete = bool(
        _REQUIRED_INPUT_HASHES.issubset(input_hashes)
        and all(
            _is_sha256(input_hashes[key])
            for key in _REQUIRED_INPUT_HASHES
        )
    )
    revision = metadata.get("git_revision")
    revision_valid = bool(
        isinstance(revision, str)
        and 7 <= len(revision) <= 64
        and all(character in "0123456789abcdef" for character in revision)
    )
    raw_blas = metadata.get("blas_runtime")
    blas_valid = bool(
        isinstance(raw_blas, (tuple, list))
        and all(isinstance(value, str) and value for value in raw_blas)
    )
    identity = {
        "declared_package_version": metadata.get(
            "declared_package_version"
        ),
        "installed_package_version": metadata.get(
            "installed_package_version"
        ),
        "git_revision": revision,
        "source_tree_sha256": metadata.get("source_tree_sha256"),
        "dtype": metadata.get("dtype"),
        "numpy_version": metadata.get("numpy_version"),
        "scipy_version": metadata.get("scipy_version"),
        "python": metadata.get("python"),
        "platform": metadata.get("platform"),
        "blas_runtime": list(raw_blas) if blas_valid else None,
    }
    environment_complete = bool(
        identity["dtype"] == "complex128"
        and blas_valid
        and all(
            isinstance(identity[key], str) and bool(identity[key])
            for key in (
                "numpy_version",
                "scipy_version",
                "python",
                "platform",
            )
        )
    )
    code_complete = bool(
        _is_sha256(identity["source_tree_sha256"])
        and revision_valid
        and metadata.get("git_dirty") is False
        and metadata.get("package_version_consistent") is True
        and all(
            isinstance(identity[key], str) and bool(identity[key])
            for key in (
                "declared_package_version",
                "installed_package_version",
            )
        )
    )
    complete = bool(input_complete and code_complete and environment_complete)
    return {
        "complete": complete,
        "clean": metadata.get("git_dirty") is False,
        "git_dirty": metadata.get("git_dirty"),
        "package_version_consistent": metadata.get(
            "package_version_consistent"
        ),
        "identity": identity,
        "input_hashes": input_hashes,
        "input_hashes_complete": input_complete,
    }


def _multidimensional_provenance_summary(
    outcomes: tuple[MultidimensionalRunOutcome, ...],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    identities: set[str] = set()
    for outcome in outcomes:
        if outcome.result is None:
            records.append(
                {
                    "run_id": outcome.run_id,
                    "result_available": False,
                    "complete": False,
                    "clean": False,
                    "identity": None,
                }
            )
            continue
        record = _result_provenance(outcome.result)
        records.append(
            {
                "run_id": outcome.run_id,
                "result_available": True,
                **record,
            }
        )
        identities.add(
            json.dumps(
                record["identity"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        )
    result_coverage = bool(
        outcomes and len(records) == len(outcomes)
        and all(record["result_available"] for record in records)
    )
    complete = bool(
        result_coverage and all(record["complete"] for record in records)
    )
    clean = bool(
        result_coverage and all(record["clean"] for record in records)
    )
    consistent = bool(result_coverage and len(identities) == 1)
    identity = (
        records[0]["identity"]
        if consistent and records
        else None
    )
    return {
        "result_coverage": result_coverage,
        "complete": complete,
        "clean": clean,
        "consistent": consistent,
        "identity_keys": list(_PROVENANCE_IDENTITY_KEYS),
        "identity": identity,
        "passed": bool(complete and clean and consistent),
        "runs": records,
    }


@dataclass(frozen=True, slots=True)
class MultidimensionalFieldError:
    """Scale-aware difference for one Keldysh field component."""

    field: str
    component: str
    max_absolute_error: float
    scale: float
    max_relative_error: float
    allowed_absolute_error: float
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "component": self.component,
            "max_absolute_error": self.max_absolute_error,
            "scale": self.scale,
            "max_relative_error": self.max_relative_error,
            "allowed_absolute_error": self.allowed_absolute_error,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalLineShapeMeasurement:
    """Predeclared local on-shell cubic self-energy observable."""

    run_id: str
    dof_label: str
    dof_index: int
    target_angular_frequency: float
    sigma_retarded_real: float
    sigma_retarded_imag: float
    linewidth: float
    shift: float

    def __post_init__(self) -> None:
        if not self.run_id or not self.dof_label:
            raise ValueError("line-shape labels must be non-empty.")
        if isinstance(self.dof_index, (bool, np.bool_)) or not isinstance(
            self.dof_index,
            (int, np.integer),
        ):
            raise TypeError("dof_index must be an integer.")
        if int(self.dof_index) < 0:
            raise ValueError("dof_index must be non-negative.")
        object.__setattr__(self, "dof_index", int(self.dof_index))
        for name in (
            "target_angular_frequency",
            "sigma_retarded_real",
            "sigma_retarded_imag",
            "linewidth",
            "shift",
        ):
            object.__setattr__(
                self,
                name,
                _finite(getattr(self, name), name=name),
            )
        if self.target_angular_frequency <= 0.0:
            raise ValueError(
                "target_angular_frequency must be strictly positive."
            )

    def as_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "dof_label": self.dof_label,
            "dof_index": self.dof_index,
            "target_angular_frequency": self.target_angular_frequency,
            "sigma_retarded_real": self.sigma_retarded_real,
            "sigma_retarded_imag": self.sigma_retarded_imag,
            "linewidth": self.linewidth,
            "shift": self.shift,
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalLineShapeComparison:
    """Refinement error for the fixed center-layer parent-mode observable."""

    definition: str
    left: MultidimensionalLineShapeMeasurement
    right: MultidimensionalLineShapeMeasurement
    relative_tolerance: float
    linewidth_absolute_tolerance: float
    shift_absolute_tolerance: float
    linewidth_error: MultidimensionalFieldError
    shift_error: MultidimensionalFieldError
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.definition:
            raise ValueError("line-shape definition must be non-empty.")
        if not isinstance(self.left, MultidimensionalLineShapeMeasurement):
            raise TypeError("left must be a line-shape measurement.")
        if not isinstance(self.right, MultidimensionalLineShapeMeasurement):
            raise TypeError("right must be a line-shape measurement.")
        for name in (
            "relative_tolerance",
            "linewidth_absolute_tolerance",
            "shift_absolute_tolerance",
        ):
            object.__setattr__(
                self,
                name,
                _positive(getattr(self, name), name=name),
            )
        if not isinstance(self.linewidth_error, MultidimensionalFieldError):
            raise TypeError("linewidth_error must be a field error.")
        if not isinstance(self.shift_error, MultidimensionalFieldError):
            raise TypeError("shift_error must be a field error.")
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(
            self,
            "failures",
            tuple(str(value) for value in self.failures),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "definition": self.definition,
            "left": self.left.as_dict(),
            "right": self.right.as_dict(),
            "relative_tolerance": self.relative_tolerance,
            "linewidth_absolute_tolerance": (
                self.linewidth_absolute_tolerance
            ),
            "shift_absolute_tolerance": self.shift_absolute_tolerance,
            "linewidth_error": self.linewidth_error.as_dict(),
            "shift_error": self.shift_error.as_dict(),
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalComparison:
    """One fixed-point, backend, continuation, or refinement comparison."""

    name: str
    kind: ComparisonKind
    left_run_id: str
    right_run_id: str
    field_errors: tuple[MultidimensionalFieldError, ...]
    current_max_absolute_error: float | None
    current_scale: float | None
    current_max_relative_error: float | None
    sigma_tail_left: float | None
    sigma_tail_right: float | None
    elapsed_ratio_right_over_left: float | None
    line_shape: MultidimensionalLineShapeComparison | None
    certification_critical: bool
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("comparison name must be non-empty.")
        if self.kind not in {
            "zero_vs_born",
            "direct_vs_continuation",
            "serial_vs_threaded",
            "numerical_refinement",
        }:
            raise ValueError("unknown comparison kind.")
        if any(
            not isinstance(item, MultidimensionalFieldError)
            for item in self.field_errors
        ):
            raise TypeError(
                "field_errors must contain MultidimensionalFieldError."
            )
        object.__setattr__(self, "field_errors", tuple(self.field_errors))
        if self.line_shape is not None and not isinstance(
            self.line_shape,
            MultidimensionalLineShapeComparison,
        ):
            raise TypeError(
                "line_shape must be a MultidimensionalLineShapeComparison "
                "or None."
            )
        object.__setattr__(
            self,
            "failures",
            tuple(str(value) for value in self.failures),
        )
        object.__setattr__(
            self,
            "certification_critical",
            bool(self.certification_critical),
        )
        object.__setattr__(self, "passed", bool(self.passed))

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "left_run_id": self.left_run_id,
            "right_run_id": self.right_run_id,
            "field_errors": [
                item.as_dict() for item in self.field_errors
            ],
            "current_max_absolute_error": self.current_max_absolute_error,
            "current_scale": self.current_scale,
            "current_max_relative_error": self.current_max_relative_error,
            "sigma_tail_left": self.sigma_tail_left,
            "sigma_tail_right": self.sigma_tail_right,
            "elapsed_ratio_right_over_left": (
                self.elapsed_ratio_right_over_left
            ),
            "line_shape": (
                None
                if self.line_shape is None
                else self.line_shape.as_dict()
            ),
            "certification_critical": self.certification_critical,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalConservationCheck:
    """Explicit equilibrium or nonequilibrium conservation gate."""

    name: str
    run_id: str
    equilibrium: bool
    current_scale: float | None
    allowed_error: float | None
    left_current: float | None
    right_current: float | None
    contact_balance: float | None
    collision_power: float | None
    full_balance: float | None
    bond_current_error: float | None
    passed: bool
    failures: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "equilibrium", bool(self.equilibrium))
        object.__setattr__(self, "passed", bool(self.passed))
        object.__setattr__(
            self,
            "failures",
            tuple(str(value) for value in self.failures),
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "equilibrium": self.equilibrium,
            "current_scale": self.current_scale,
            "allowed_error": self.allowed_error,
            "left_current": self.left_current,
            "right_current": self.right_current,
            "contact_balance": self.contact_balance,
            "collision_power": self.collision_power,
            "full_balance": self.full_balance,
            "bond_current_error": self.bond_current_error,
            "passed": self.passed,
            "failures": list(self.failures),
        }


@dataclass(frozen=True, slots=True)
class MultidimensionalReleaseReport:
    """Complete immutable multidimensional benchmark report."""

    config: MultidimensionalReleaseConfig
    outcomes: tuple[MultidimensionalRunOutcome, ...]
    comparisons: tuple[MultidimensionalComparison, ...]
    conservation_checks: tuple[MultidimensionalConservationCheck, ...]
    failures: tuple[str, ...]
    certification_failures: tuple[str, ...]
    total_elapsed_seconds: float
    execution_passed: bool
    release_certified: bool
    checkpoint_evidence: Mapping[str, object] = field(init=False, repr=False)
    topology_validation: Mapping[str, object] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, MultidimensionalReleaseConfig):
            raise TypeError(
                "config must be a MultidimensionalReleaseConfig."
            )
        for name, values, expected in (
            ("outcomes", self.outcomes, MultidimensionalRunOutcome),
            (
                "comparisons",
                self.comparisons,
                MultidimensionalComparison,
            ),
            (
                "conservation_checks",
                self.conservation_checks,
                MultidimensionalConservationCheck,
            ),
        ):
            normalized = tuple(values)
            if any(not isinstance(item, expected) for item in normalized):
                raise TypeError(f"{name} contains an invalid record.")
            object.__setattr__(self, name, normalized)
        run_ids = tuple(item.run_id for item in self.outcomes)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("outcome run_id values must be unique.")
        object.__setattr__(
            self,
            "failures",
            tuple(str(value) for value in self.failures),
        )
        object.__setattr__(
            self,
            "certification_failures",
            tuple(str(value) for value in self.certification_failures),
        )
        object.__setattr__(
            self,
            "total_elapsed_seconds",
            _finite(
                self.total_elapsed_seconds,
                name="total_elapsed_seconds",
                nonnegative=True,
            ),
        )
        checkpoint_evidence = _multidimensional_checkpoint_summary(
            self.outcomes
        )
        topology_validation = _multidimensional_topology_summary(
            self.config,
            self.outcomes,
            self.comparisons,
            self.conservation_checks,
        )
        object.__setattr__(
            self,
            "checkpoint_evidence",
            checkpoint_evidence,
        )
        object.__setattr__(
            self,
            "topology_validation",
            topology_validation,
        )
        execution_passed = bool(self.execution_passed)
        release_certified = bool(self.release_certified)
        if execution_passed and self.failures:
            raise ValueError(
                "execution_passed cannot be true when failures are recorded."
            )
        if release_certified and (
            not execution_passed
            or not self.config.certification_eligible
            or self.certification_failures
            or not _multidimensional_provenance_summary(self.outcomes)[
                "passed"
            ]
            or not checkpoint_evidence["passed"]
            or not topology_validation["passed"]
            or not _multidimensional_weak_coupling_summary(
                self.outcomes,
                self.config.spec,
            )["passed"]
        ):
            raise ValueError(
                "release certification requires eligible passing execution "
                "without certification failures."
            )
        object.__setattr__(self, "execution_passed", execution_passed)
        object.__setattr__(self, "release_certified", release_certified)

    @property
    def coarse(self) -> bool:
        return self.config.coarse

    def _core_dict(self) -> dict[str, object]:
        refinement_comparisons = {
            item.name: item
            for item in self.comparisons
            if item.kind == "numerical_refinement"
        }
        observed_refinements = set(refinement_comparisons)
        required_refinements = _required_refinement_comparison_names(
            self.config
        )
        expected_refinements = {
            f"numerical_refinement_{label}"
            for label, _ in _refinement_points(self.config)
        }
        provenance = _multidimensional_provenance_summary(self.outcomes)
        weak_coupling = _multidimensional_weak_coupling_summary(
            self.outcomes,
            self.config.spec,
        )
        return {
            "schema": (
                "negfpy.anharmonic.multidimensional_dense_toy_release"
            ),
            "schema_version": 3,
            "config": self.config.as_dict(),
            "outcomes": [
                item.as_dict(self.config.spec.mean_temperature)
                for item in self.outcomes
            ],
            "comparisons": [
                item.as_dict() for item in self.comparisons
            ],
            "conservation_checks": [
                item.as_dict() for item in self.conservation_checks
            ],
            "numerical_refinement_archive": {
                "expected_comparison_names": sorted(expected_refinements),
                "observed_comparison_names": sorted(observed_refinements),
                "all_six_archived": bool(
                    observed_refinements == expected_refinements
                    and len(expected_refinements) == 6
                ),
                "diagnostic_failures": [
                    item.as_dict()
                    for item in refinement_comparisons.values()
                    if not item.certification_critical and not item.passed
                ],
            },
            "required_refinement_coverage": {
                direction: {
                    "comparison_name": comparison_name,
                    "executed": (
                        comparison_name is not None
                        and comparison_name in observed_refinements
                    ),
                    "certification_critical": (
                        comparison_name is not None
                        and comparison_name in refinement_comparisons
                        and refinement_comparisons[
                            comparison_name
                        ].certification_critical
                    ),
                    "passed": (
                        comparison_name is not None
                        and comparison_name in refinement_comparisons
                        and refinement_comparisons[comparison_name].passed
                    ),
                }
                for direction, comparison_name in (
                    required_refinements.items()
                )
            },
            "checkpoint_evidence": self.checkpoint_evidence,
            "topology_validation": self.topology_validation,
            "weak_coupling": weak_coupling,
            "provenance": provenance,
            "failures": list(self.failures),
            "certification_failures": list(
                self.certification_failures
            ),
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "execution_passed": self.execution_passed,
            "release_certified": self.release_certified,
            "coarse": self.coarse,
        }

    def as_dict(self) -> dict[str, object]:
        core = self._core_dict()
        canonical = json.dumps(
            core,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return {
            **core,
            "report_sha256": hashlib.sha256(
                canonical.encode("utf-8")
            ).hexdigest(),
        }

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.as_dict(),
            sort_keys=True,
            indent=indent,
            allow_nan=False,
        )

    @property
    def report_sha256(self) -> str:
        return str(self.as_dict()["report_sha256"])


def _solver_config(
    config: MultidimensionalReleaseConfig,
    numerical: MultidimensionalNumericalPoint,
    *,
    backend: Literal["serial", "threaded"],
) -> SCBAConfig:
    return SCBAConfig(
        eta_lead=numerical.eta_lead,
        max_iterations=config.max_iterations,
        min_iterations=config.min_iterations,
        mixing=numerical.mixing,
        mixing_floor=config.mixing_floor,
        consecutive_passes=config.consecutive_passes,
        sigma_rtol=config.sigma_rtol,
        sigma_atol=config.sigma_atol,
        green_rtol=config.green_rtol,
        green_atol=config.green_atol,
        identity_rtol=config.identity_rtol,
        current_rtol=config.current_rtol,
        current_atol=config.current_atol,
        tail_rtol=config.tail_rtol,
        retarded_strategy="causal_hilbert",
        backend=backend,
        on_nonconvergence="raise",
        eta_device=0.0,
        max_workers=(config.max_workers if backend == "threaded" else None),
    )


def _run_id(
    prefix: str,
    point: MultidimensionalRunPoint,
    *,
    suffix: str = "",
) -> str:
    base = (
        f"{prefix}_g{_slug(point.g)}_bias{_slug(point.bias)}"
        f"_dw{_slug(point.numerical.spacing)}"
        f"_cut{_slug(point.numerical.cutoff)}"
        f"_eta{_slug(point.numerical.eta_lead)}"
        f"_mix{_slug(point.numerical.mixing)}"
    )
    return base if not suffix else f"{base}_{suffix}"



def _checkpoint_target(directory: Path, run_id: str) -> Path:
    """Return a traversal-safe, human-readable checkpoint path."""

    if any(
        not (
            character.isascii()
            and (character.isalnum() or character in "_-")
        )
        for character in run_id
    ):
        raise ValueError(
            "internal run_id is not safe for a checkpoint filename."
        )
    return directory / f"{run_id}.h5"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _blocked_outcome(
    *,
    run_id: str,
    point: MultidimensionalRunPoint,
    initialization: Initialization,
    backend: Literal["serial", "threaded"],
    parent_id: str | None,
    error_type: str,
    error_message: str,
) -> MultidimensionalRunOutcome:
    return MultidimensionalRunOutcome(
        run_id=run_id,
        point=point,
        initialization=initialization,
        backend=backend,
        continuation_parent_id=parent_id,
        elapsed_seconds=0.0,
        seed_elapsed_seconds=0.0,
        status="blocked_by_parent",
        result=None,
        error_type=error_type,
        error_message=error_message,
    )


def _completed_outcome(
    *,
    run_id: str,
    point: MultidimensionalRunPoint,
    initialization: Initialization,
    backend: Literal["serial", "threaded"],
    parent_id: str | None,
    result: AnharmonicResult,
    elapsed_seconds: float,
    seed_elapsed_seconds: float,
    error_type: str | None,
    error_message: str | None,
    checkpoint_path: Path | None,
    checkpoint_identity: SCBACheckpointIdentity | None,
    resumed: bool = False,
    reconstruction_seconds: float = 0.0,
) -> MultidimensionalRunOutcome:
    return MultidimensionalRunOutcome(
        run_id=run_id,
        point=point,
        initialization=initialization,
        backend=backend,
        continuation_parent_id=parent_id,
        elapsed_seconds=elapsed_seconds,
        seed_elapsed_seconds=seed_elapsed_seconds,
        status=("converged" if result.converged else "nonconverged_partial"),
        result=result,
        error_type=(None if result.converged else error_type),
        error_message=(None if result.converged else error_message),
        checkpoint_path=(
            None if checkpoint_path is None else str(checkpoint_path)
        ),
        checkpoint_file_sha256=(
            None if checkpoint_path is None else _file_sha256(checkpoint_path)
        ),
        checkpoint_identity=checkpoint_identity,
        resumed_from_checkpoint=resumed,
        reconstruction_seconds=reconstruction_seconds,
    )


def _execute(
    config: MultidimensionalReleaseConfig,
    *,
    run_id: str,
    point: MultidimensionalRunPoint,
    initialization: Initialization,
    backend: Literal["serial", "threaded"],
    parent: MultidimensionalRunOutcome | None = None,
    checkpoint_dir: Path | None = None,
    environment_sha256: str | None = None,
    resume: bool = True,
) -> MultidimensionalRunOutcome:
    if initialization == "continuation" and (
        parent is None or not parent.successful
    ):
        parent_id = None if parent is None else parent.run_id
        return _blocked_outcome(
            run_id=run_id,
            point=point,
            initialization=initialization,
            backend=backend,
            parent_id=parent_id,
            error_type="ContinuationParentUnavailable",
            error_message=(
                "the preregistered continuation parent did not converge; "
                "this point was not cold-started or seeded from a partial"
            ),
        )

    started = time.perf_counter()
    seed_seconds = 0.0
    parent_id = None if parent is None else parent.run_id
    target: Path | None = None
    identity: SCBACheckpointIdentity | None = None
    try:
        problem = build_multidimensional_problem(
            config,
            g=point.g,
            bias=point.bias,
            numerical=point.numerical,
        )
        solver_config = _solver_config(
            config,
            point.numerical,
            backend=backend,
        )
        initial_sigma: KeldyshSpectrum | None = None
        if initialization == "born_seed":
            seed_started = time.perf_counter()
            born = solve_one_shot_born(problem, solver_config)
            seed_seconds = time.perf_counter() - seed_started
            initial_sigma = born.sigma_pp
        elif initialization == "continuation":
            assert parent is not None and parent.result is not None
            initial_sigma = parent.result.sigma_pp

        if checkpoint_dir is not None:
            assert environment_sha256 is not None
            target = _checkpoint_target(checkpoint_dir, run_id)
            identity = make_scba_checkpoint_identity(
                run_id,
                problem,
                solver_config,
                environment_sha256=environment_sha256,
                initial_sigma_pp=initial_sigma,
            )
            if resume and target.is_file():
                checkpoint = load_scba_checkpoint(
                    target,
                    expected_identity=identity,
                )
                reconstruction_started = time.perf_counter()
                restored = restore_scba_checkpoint_result(
                    checkpoint,
                    problem,
                    solver_config,
                )
                reconstruction_seconds = (
                    time.perf_counter() - reconstruction_started
                )
                stored_error_type = checkpoint.metadata.get("error_type")
                stored_error_message = checkpoint.metadata.get(
                    "error_message"
                )
                return _completed_outcome(
                    run_id=run_id,
                    point=point,
                    initialization=initialization,
                    backend=backend,
                    parent_id=parent_id,
                    result=restored,
                    elapsed_seconds=float(
                        checkpoint.timings.get("elapsed_seconds", 0.0)
                    ),
                    seed_elapsed_seconds=float(
                        checkpoint.timings.get("seed_elapsed_seconds", 0.0)
                    ),
                    error_type=(
                        "CheckpointedNonconvergence"
                        if stored_error_type is None
                        else str(stored_error_type)
                    ),
                    error_message=(
                        "checkpoint contains a nonconverged partial result"
                        if stored_error_message is None
                        else str(stored_error_message)
                    ),
                    checkpoint_path=target,
                    checkpoint_identity=identity,
                    resumed=True,
                    reconstruction_seconds=reconstruction_seconds,
                )

        result = solve_scba(
            problem,
            solver_config,
            initial_sigma_pp=initial_sigma,
        )
    except CheckpointError:
        # Stale, mismatched, and corrupt checkpoints invalidate the run. They
        # must not be downgraded to ordinary archived solver failures.
        raise
    except SCBAConvergenceError as error:
        elapsed = time.perf_counter() - started
        partial = error.partial_result
        if partial is None:
            return MultidimensionalRunOutcome(
                run_id=run_id,
                point=point,
                initialization=initialization,
                backend=backend,
                continuation_parent_id=parent_id,
                elapsed_seconds=elapsed,
                seed_elapsed_seconds=seed_seconds,
                status="failed",
                result=None,
                error_type=type(error).__name__,
                error_message=str(error),
            )
        if target is not None:
            assert identity is not None
            checkpoint = SCBACheckpoint.from_result(
                identity,
                partial,
                timings={
                    "elapsed_seconds": elapsed,
                    "seed_elapsed_seconds": seed_seconds,
                },
                status="nonconverged_partial",
                metadata={
                    "initialization": initialization,
                    "backend": backend,
                    "continuation_parent_id": parent_id,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
            )
            write_scba_checkpoint(target, checkpoint)
        return _completed_outcome(
            run_id=run_id,
            point=point,
            initialization=initialization,
            backend=backend,
            parent_id=parent_id,
            result=partial,
            elapsed_seconds=elapsed,
            seed_elapsed_seconds=seed_seconds,
            error_type=type(error).__name__,
            error_message=str(error),
            checkpoint_path=target,
            checkpoint_identity=identity,
        )
    except Exception as error:  # benchmark reports must retain every failure
        return MultidimensionalRunOutcome(
            run_id=run_id,
            point=point,
            initialization=initialization,
            backend=backend,
            continuation_parent_id=parent_id,
            elapsed_seconds=time.perf_counter() - started,
            seed_elapsed_seconds=seed_seconds,
            status="failed",
            result=None,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    elapsed = time.perf_counter() - started
    result_error_type = None if result.converged else "NonconvergedResult"
    result_error_message = (
        None if result.converged else "solver returned a nonconverged result"
    )
    if target is not None:
        assert identity is not None
        checkpoint = SCBACheckpoint.from_result(
            identity,
            result,
            timings={
                "elapsed_seconds": elapsed,
                "seed_elapsed_seconds": seed_seconds,
            },
            metadata={
                "initialization": initialization,
                "backend": backend,
                "continuation_parent_id": parent_id,
                "error_type": result_error_type,
                "error_message": result_error_message,
            },
        )
        write_scba_checkpoint(target, checkpoint)
    return _completed_outcome(
        run_id=run_id,
        point=point,
        initialization=initialization,
        backend=backend,
        parent_id=parent_id,
        result=result,
        elapsed_seconds=elapsed,
        seed_elapsed_seconds=seed_seconds,
        error_type=result_error_type,
        error_message=result_error_message,
        checkpoint_path=target,
        checkpoint_identity=identity,
    )

def _field_error(
    field: str,
    component: str,
    left: Array,
    right: Array,
    *,
    rtol: float,
    atol: float,
) -> MultidimensionalFieldError:
    difference = float(np.max(np.abs(left - right)))
    scale = max(
        float(np.max(np.abs(left))),
        float(np.max(np.abs(right))),
        atol / rtol,
    )
    allowed = atol + rtol * scale
    return MultidimensionalFieldError(
        field=field,
        component=component,
        max_absolute_error=difference,
        scale=scale,
        max_relative_error=difference / scale,
        allowed_absolute_error=allowed,
        passed=bool(difference <= allowed),
    )


def _line_shape_measurement(
    outcome: MultidimensionalRunOutcome,
    spec: MultidimensionalLadderSpec,
) -> MultidimensionalLineShapeMeasurement:
    if outcome.result is None:
        raise ValueError("line-shape measurement requires a retained result.")
    result = outcome.result
    target = spec.polarization_frequencies[0]
    omega = result.grid.values
    if target < omega[0] or target > omega[-1]:
        raise ValueError(
            "parent-mode target frequency lies outside the result grid."
        )
    dof_index = 3  # parent-x of the center layer in a 3-layer, xyz layout
    diagonal = result.sigma_pp.retarded[:, dof_index, dof_index]
    real = float(np.interp(target, omega, diagonal.real))
    imaginary = float(np.interp(target, omega, diagonal.imag))
    return MultidimensionalLineShapeMeasurement(
        run_id=outcome.run_id,
        dof_label="center_layer.parent_x",
        dof_index=dof_index,
        target_angular_frequency=target,
        sigma_retarded_real=real,
        sigma_retarded_imag=imaginary,
        linewidth=-imaginary / target,
        shift=real / (2.0 * target),
    )


def _multidimensional_weak_coupling_summary(
    outcomes: tuple[MultidimensionalRunOutcome, ...],
    spec: MultidimensionalLadderSpec,
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    parent_frequency = spec.polarization_frequencies[0]
    for outcome in outcomes:
        record: dict[str, object] = {
            "run_id": outcome.run_id,
            "g": outcome.point.g,
            "limit": _WEAK_COUPLING_LIMIT,
            "result_available": outcome.result is not None,
            "linewidth": None,
            "linewidth_ratio": None,
            "shift": None,
            "shift_ratio": None,
            "passed": False,
        }
        if outcome.result is not None:
            try:
                measurement = _line_shape_measurement(outcome, spec)
            except (TypeError, ValueError):
                pass
            else:
                linewidth_ratio = abs(measurement.linewidth) / parent_frequency
                shift_ratio = abs(measurement.shift) / parent_frequency
                record.update(
                    {
                        "linewidth": measurement.linewidth,
                        "linewidth_ratio": linewidth_ratio,
                        "shift": measurement.shift,
                        "shift_ratio": shift_ratio,
                        "passed": bool(
                            linewidth_ratio < _WEAK_COUPLING_LIMIT
                            and shift_ratio < _WEAK_COUPLING_LIMIT
                        ),
                    }
                )
        records.append(record)
    coverage = bool(
        outcomes
        and len(records) == len(outcomes)
        and all(record["result_available"] for record in records)
    )
    passed = bool(coverage and all(record["passed"] for record in records))
    return {
        "definition": (
            "For center-layer parent-x at Omega_p, require both "
            "abs(-Im(Sigma_pp^r)/Omega_p)/Omega_p and "
            "abs(Re(Sigma_pp^r)/(2*Omega_p))/Omega_p below 0.1."
        ),
        "limit": _WEAK_COUPLING_LIMIT,
        "coverage": coverage,
        "passed": passed,
        "runs": records,
    }


def _line_shape_comparison(
    left: MultidimensionalRunOutcome,
    right: MultidimensionalRunOutcome,
    spec: MultidimensionalLadderSpec,
    *,
    rtol: float,
    linewidth_atol: float,
    shift_atol: float,
) -> MultidimensionalLineShapeComparison:
    first = _line_shape_measurement(left, spec)
    second = _line_shape_measurement(right, spec)
    linewidth_error = _field_error(
        "center_layer.parent_x",
        "linewidth",
        np.asarray([first.linewidth]),
        np.asarray([second.linewidth]),
        rtol=rtol,
        atol=linewidth_atol,
    )
    shift_error = _field_error(
        "center_layer.parent_x",
        "frequency_shift",
        np.asarray([first.shift]),
        np.asarray([second.shift]),
        rtol=rtol,
        atol=shift_atol,
    )
    failures: list[str] = []
    if not linewidth_error.passed:
        failures.append("parent-mode linewidth exceeds refinement tolerance")
    if not shift_error.passed:
        failures.append("parent-mode shift exceeds refinement tolerance")
    return MultidimensionalLineShapeComparison(
        definition=(
            f"At Omega_p={first.target_angular_frequency:.17g}, linearly "
            "interpolate the center-layer parent-x diagonal retarded cubic "
            "self-energy; linewidth="
            "-Im(Sigma_pp^r_xx)/Omega_p and shift="
            "Re(Sigma_pp^r_xx)/(2*Omega_p)."
        ),
        left=first,
        right=second,
        relative_tolerance=rtol,
        linewidth_absolute_tolerance=linewidth_atol,
        shift_absolute_tolerance=shift_atol,
        linewidth_error=linewidth_error,
        shift_error=shift_error,
        passed=not failures,
        failures=tuple(failures),
    )


def _current_vector(result: AnharmonicResult) -> Array:
    currents = result.currents
    values = [currents.left, currents.right, currents.collision_power]
    if currents.bond_currents is not None:
        values.extend(
            np.asarray(currents.bond_currents, dtype=float).tolist()
        )
    return np.asarray(values, dtype=float)


def _compare_outcomes(
    name: str,
    kind: ComparisonKind,
    left: MultidimensionalRunOutcome,
    right: MultidimensionalRunOutcome,
    *,
    field_rtol: float,
    field_atol: float,
    current_rtol: float,
    current_atol: float,
    compare_fields: bool,
    certification_critical: bool = True,
    line_shape_spec: MultidimensionalLadderSpec | None = None,
    line_shape_rtol: float = 5.0e-3,
    linewidth_atol: float = 1.0e-12,
    shift_atol: float = 1.0e-8,
) -> MultidimensionalComparison:
    failures: list[str] = []
    if not left.successful:
        failures.append(f"{left.run_id} did not converge")
    if not right.successful:
        failures.append(f"{right.run_id} did not converge")
    field_errors: list[MultidimensionalFieldError] = []
    current_absolute: float | None = None
    current_scale: float | None = None
    current_relative: float | None = None
    tail_left: float | None = None
    tail_right: float | None = None
    line_shape: MultidimensionalLineShapeComparison | None = None

    if left.result is not None and right.result is not None:
        if compare_fields:
            if left.result.grid.n_points != right.result.grid.n_points:
                failures.append(
                    "field comparison requested for different grids"
                )
            else:
                for field in ("green", "sigma_pp"):
                    first_spectrum = getattr(left.result, field)
                    second_spectrum = getattr(right.result, field)
                    for component in ("retarded", "lesser", "greater"):
                        error = _field_error(
                            field,
                            component,
                            getattr(first_spectrum, component),
                            getattr(second_spectrum, component),
                            rtol=field_rtol,
                            atol=field_atol,
                        )
                        field_errors.append(error)
                        if not error.passed:
                            failures.append(
                                f"{field}.{component} exceeds tolerance"
                            )
        first_current = _current_vector(left.result)
        second_current = _current_vector(right.result)
        if first_current.shape != second_current.shape:
            failures.append("current vectors have different shapes")
        else:
            current_absolute = float(
                np.max(np.abs(first_current - second_current))
            )
            current_scale = max(
                float(np.max(np.abs(first_current))),
                float(np.max(np.abs(second_current))),
                current_atol / current_rtol,
            )
            current_relative = current_absolute / current_scale
            if current_absolute > current_atol + current_rtol * current_scale:
                failures.append("current comparison exceeds tolerance")
        tail_left = float(keldysh_tail_ratio(left.result.sigma_pp))
        tail_right = float(keldysh_tail_ratio(right.result.sigma_pp))
        if line_shape_spec is not None:
            try:
                line_shape = _line_shape_comparison(
                    left,
                    right,
                    line_shape_spec,
                    rtol=line_shape_rtol,
                    linewidth_atol=linewidth_atol,
                    shift_atol=shift_atol,
                )
            except (TypeError, ValueError) as error:
                failures.append(
                    f"line-shape comparison unavailable: {error}"
                )
            else:
                failures.extend(line_shape.failures)
    elif line_shape_spec is not None:
        failures.append(
            "line-shape comparison unavailable without two retained results"
        )

    elapsed_ratio = (
        right.elapsed_seconds / left.elapsed_seconds
        if left.elapsed_seconds > 0.0
        else None
    )
    return MultidimensionalComparison(
        name=name,
        kind=kind,
        left_run_id=left.run_id,
        right_run_id=right.run_id,
        field_errors=tuple(field_errors),
        current_max_absolute_error=current_absolute,
        current_scale=current_scale,
        current_max_relative_error=current_relative,
        sigma_tail_left=tail_left,
        sigma_tail_right=tail_right,
        elapsed_ratio_right_over_left=elapsed_ratio,
        line_shape=line_shape,
        certification_critical=certification_critical,
        passed=not failures,
        failures=tuple(failures),
    )


def _conservation_check(
    name: str,
    outcome: MultidimensionalRunOutcome,
    config: MultidimensionalReleaseConfig,
    *,
    equilibrium: bool,
) -> MultidimensionalConservationCheck:
    if outcome.result is None:
        return MultidimensionalConservationCheck(
            name=name,
            run_id=outcome.run_id,
            equilibrium=equilibrium,
            current_scale=None,
            allowed_error=None,
            left_current=None,
            right_current=None,
            contact_balance=None,
            collision_power=None,
            full_balance=None,
            bond_current_error=None,
            passed=False,
            failures=("run has no retained result",),
        )
    result = outcome.result
    currents = result.currents
    physical_floor = config.current_atol * result.metadata.get(
        "current_scale",
        1.0,
    )
    current_scale = max(
        abs(currents.left),
        abs(currents.right),
        float(physical_floor),
    )
    allowed = max(
        float(physical_floor),
        config.current_rtol * current_scale,
    )
    bonds = (
        None
        if currents.bond_currents is None
        else np.asarray(currents.bond_currents, dtype=float)
    )
    bond_error = (
        0.0
        if bonds is None or bonds.size == 0
        else max(
            float(np.max(np.abs(bonds - currents.left))),
            float(np.max(np.abs(bonds + currents.right))),
            float(np.max(bonds) - np.min(bonds)),
        )
    )
    failures: list[str] = []
    if not outcome.successful:
        failures.append("run did not converge")
    if not result.diagnostics.passed:
        failures.append("essential solver diagnostics did not pass")
    if equilibrium:
        equilibrium_allowed = float(physical_floor)
        if abs(currents.left) > equilibrium_allowed:
            failures.append("left equilibrium current exceeds absolute gate")
        if abs(currents.right) > equilibrium_allowed:
            failures.append("right equilibrium current exceeds absolute gate")
    if abs(currents.contact_balance) > allowed:
        failures.append("contact balance exceeds gate")
    if abs(currents.collision_power) > allowed:
        failures.append("collision power exceeds gate")
    if abs(currents.full_balance) > allowed:
        failures.append("full balance exceeds gate")
    if bond_error > allowed:
        failures.append("bond-current error exceeds gate")
    return MultidimensionalConservationCheck(
        name=name,
        run_id=outcome.run_id,
        equilibrium=equilibrium,
        current_scale=current_scale,
        allowed_error=allowed,
        left_current=currents.left,
        right_current=currents.right,
        contact_balance=currents.contact_balance,
        collision_power=currents.collision_power,
        full_balance=currents.full_balance,
        bond_current_error=bond_error,
        passed=not failures,
        failures=tuple(failures),
    )


def _physical_path(
    config: MultidimensionalReleaseConfig,
) -> tuple[tuple[float, float], ...]:
    zero_bias = config.executed_bias_values[0]
    maximum_g = config.executed_g_values[-1]
    return tuple(
        [(g, zero_bias) for g in config.executed_g_values]
        + [
            (maximum_g, bias)
            for bias in config.executed_bias_values[1:]
        ]
    )


def _refinement_points(
    config: MultidimensionalReleaseConfig,
) -> tuple[tuple[str, MultidimensionalNumericalPoint], ...]:
    reference = config.numerical
    candidates: list[tuple[str, MultidimensionalNumericalPoint]] = []
    axes = (
        ("spacing", config.spec.spacings),
        ("cutoff", config.spec.cutoffs),
        ("eta_lead", config.spec.etas),
        ("mixing", config.spec.mixings),
    )
    for axis, values in axes:
        for value in values:
            if value == getattr(reference, axis):
                continue
            point = replace(reference, **{axis: value})
            candidates.append((f"{axis}_{_slug(value)}", point))
    return tuple(candidates)


def _required_refinement_comparison_names(
    config: MultidimensionalReleaseConfig,
) -> dict[str, str | None]:
    """Name the four release directions independently of total point count."""

    reference = config.numerical

    def registered_name(
        axis: str,
        value: float,
        registered: tuple[float, ...],
    ) -> str | None:
        if value not in registered:
            return None
        return f"numerical_refinement_{axis}_{_slug(value)}"

    changed_mixings = tuple(
        value
        for value in config.spec.mixings
        if value != reference.mixing
    )
    changed_mixing = (
        changed_mixings[0] if len(changed_mixings) == 1 else None
    )
    return {
        "half_spacing": registered_name(
            "spacing",
            0.5 * reference.spacing,
            config.spec.spacings,
        ),
        "larger_cutoff": registered_name(
            "cutoff",
            6.0,
            config.spec.cutoffs,
        )
        if reference.cutoff == 5.0
        else None,
        "half_eta": registered_name(
            "eta_lead",
            0.5 * reference.eta_lead,
            config.spec.etas,
        ),
        "changed_mixing": (
            None
            if changed_mixing is None
            else registered_name(
                "mixing",
                changed_mixing,
                config.spec.mixings,
            )
        ),
    }




def _direct_endpoint_unlocks_graph(
    config: MultidimensionalReleaseConfig,
    outcome: MultidimensionalRunOutcome,
) -> bool:
    """Return whether the mandatory endpoint may unlock the remaining graph."""

    result = outcome.result
    return bool(
        outcome.successful
        and result is not None
        and result.diagnostics.passed
        and (not config.certification_eligible or result.release_certified)
    )


def _expected_outcome_topology(
    config: MultidimensionalReleaseConfig,
) -> tuple[dict[str, object], ...]:
    endpoint = MultidimensionalRunPoint(
        g=config.executed_g_values[-1],
        bias=config.executed_bias_values[-1],
        numerical=config.numerical,
    )
    rows: list[dict[str, object]] = [
        {
            "run_id": _run_id("direct_zero", endpoint),
            "point": endpoint,
            "initialization": "zero",
            "backend": "serial",
            "continuation_parent_id": None,
        }
    ]
    previous_path_id: str | None = None
    for index, (g_value, bias_value) in enumerate(_physical_path(config)):
        point = MultidimensionalRunPoint(
            g=g_value,
            bias=bias_value,
            numerical=config.numerical,
        )
        run_id = _run_id("path", point, suffix=f"step{index:02d}")
        rows.append(
            {
                "run_id": run_id,
                "point": point,
                "initialization": (
                    "zero" if index == 0 else "continuation"
                ),
                "backend": "serial",
                "continuation_parent_id": previous_path_id,
            }
        )
        previous_path_id = run_id
    rows.extend(
        (
            {
                "run_id": _run_id("born_seed", endpoint),
                "point": endpoint,
                "initialization": "born_seed",
                "backend": "serial",
                "continuation_parent_id": None,
            },
            {
                "run_id": _run_id("threaded_zero", endpoint),
                "point": endpoint,
                "initialization": "zero",
                "backend": "threaded",
                "continuation_parent_id": None,
            },
        )
    )
    if config.run_preregistered_refinements:
        for label, numerical in _refinement_points(config):
            point = MultidimensionalRunPoint(
                g=endpoint.g,
                bias=endpoint.bias,
                numerical=numerical,
            )
            rows.append(
                {
                    "run_id": _run_id(
                        "refinement",
                        point,
                        suffix=label,
                    ),
                    "point": point,
                    "initialization": "refinement_zero",
                    "backend": "serial",
                    "continuation_parent_id": None,
                }
            )
    return tuple(rows)


def _topology_row_document(
    row: Mapping[str, object],
    mean_temperature: float,
) -> dict[str, object]:
    point = row["point"]
    assert isinstance(point, MultidimensionalRunPoint)
    return {
        "run_id": row["run_id"],
        "point": point.as_dict(mean_temperature),
        "initialization": row["initialization"],
        "backend": row["backend"],
        "continuation_parent_id": row["continuation_parent_id"],
    }


def _multidimensional_topology_summary(
    config: MultidimensionalReleaseConfig,
    outcomes: tuple[MultidimensionalRunOutcome, ...],
    comparisons: tuple[MultidimensionalComparison, ...],
    conservation: tuple[MultidimensionalConservationCheck, ...],
) -> dict[str, object]:
    expected_rows = _expected_outcome_topology(config)
    observed_rows = tuple(
        {
            "run_id": outcome.run_id,
            "point": outcome.point,
            "initialization": outcome.initialization,
            "backend": outcome.backend,
            "continuation_parent_id": outcome.continuation_parent_id,
        }
        for outcome in outcomes
    )
    outcomes_exact = bool(
        len(observed_rows) == len(expected_rows)
        and all(
            observed == expected
            for observed, expected in zip(observed_rows, expected_rows)
        )
    )

    direct_id = str(expected_rows[0]["run_id"])
    path_count = len(_physical_path(config))
    continuation_id = str(expected_rows[path_count]["run_id"])
    born_id = str(expected_rows[path_count + 1]["run_id"])
    threaded_id = str(expected_rows[path_count + 2]["run_id"])
    required_refinement_names = {
        name
        for name in _required_refinement_comparison_names(config).values()
        if name is not None
    }
    expected_comparisons: list[tuple[str, str, str, str, bool]] = [
        (
            "zero_vs_born_endpoint",
            "zero_vs_born",
            direct_id,
            born_id,
            True,
        ),
        (
            "direct_vs_continuation_endpoint",
            "direct_vs_continuation",
            direct_id,
            continuation_id,
            True,
        ),
        (
            "serial_vs_threaded_endpoint",
            "serial_vs_threaded",
            direct_id,
            threaded_id,
            True,
        ),
    ]
    refinement_start = path_count + 3
    for (label, _), row in zip(
        _refinement_points(config),
        expected_rows[refinement_start:],
    ):
        name = f"numerical_refinement_{label}"
        expected_comparisons.append(
            (
                name,
                "numerical_refinement",
                direct_id,
                str(row["run_id"]),
                name in required_refinement_names,
            )
        )
    observed_comparisons = tuple(
        (
            item.name,
            item.kind,
            item.left_run_id,
            item.right_run_id,
            item.certification_critical,
        )
        for item in comparisons
    )
    comparisons_exact = bool(
        observed_comparisons == tuple(expected_comparisons)
    )
    critical_comparisons_passed = bool(
        comparisons_exact
        and all(
            item.passed
            for item in comparisons
            if item.certification_critical
        )
    )

    equilibrium_row = next(
        row
        for row in expected_rows[1 : path_count + 1]
        if (
            isinstance(row["point"], MultidimensionalRunPoint)
            and row["point"].g == config.executed_g_values[-1]
            and row["point"].bias == 0.0
        )
    )
    expected_conservation = (
        (
            "interacting_equilibrium",
            str(equilibrium_row["run_id"]),
            True,
            True,
        ),
        ("interacting_nonequilibrium", direct_id, False, True),
    )
    observed_conservation = tuple(
        (item.name, item.run_id, item.equilibrium, item.passed)
        for item in conservation
    )
    conservation_exact = bool(
        observed_conservation == expected_conservation
    )
    release_outcomes_valid = bool(
        all(
            outcome.successful
            and outcome.result is not None
            and outcome.result.diagnostics.passed
            and (
                not config.certification_eligible
                or outcome.result.release_certified
            )
            for outcome in outcomes
        )
    )
    full_release_count = bool(
        not config.certification_eligible or len(expected_rows) == 14
    )
    passed = bool(
        outcomes_exact
        and release_outcomes_valid
        and comparisons_exact
        and critical_comparisons_passed
        and conservation_exact
        and full_release_count
    )
    return {
        "passed": passed,
        "full_release_count": full_release_count,
        "expected_outcome_count": len(expected_rows),
        "observed_outcome_count": len(outcomes),
        "outcomes_exact": outcomes_exact,
        "comparisons_exact": comparisons_exact,
        "critical_comparisons_passed": critical_comparisons_passed,
        "release_outcomes_valid": release_outcomes_valid,
        "conservation_exact": conservation_exact,
        "expected_outcomes": [
            _topology_row_document(
                row,
                config.spec.mean_temperature,
            )
            for row in expected_rows
        ],
        "observed_outcomes": [
            _topology_row_document(
                row,
                config.spec.mean_temperature,
            )
            for row in observed_rows
        ],
        "expected_comparisons": [
            {
                "name": name,
                "kind": kind,
                "left_run_id": left,
                "right_run_id": right,
                "certification_critical": critical,
            }
            for name, kind, left, right, critical in expected_comparisons
        ],
        "observed_comparisons": [item.as_dict() for item in comparisons],
        "expected_conservation": [
            {
                "name": name,
                "run_id": run_id,
                "equilibrium": equilibrium,
                "passed": check_passed,
            }
            for name, run_id, equilibrium, check_passed in expected_conservation
        ],
        "observed_conservation": [
            item.as_dict() for item in conservation
        ],
    }


def _multidimensional_checkpoint_summary(
    outcomes: tuple[MultidimensionalRunOutcome, ...],
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    environments: set[str] = set()
    completed_count = 0
    all_completed_valid = True
    for outcome in outcomes:
        completed = bool(
            outcome.result is not None
            and outcome.status in {"converged", "nonconverged_partial"}
        )
        if not completed:
            records.append(
                {
                    "run_id": outcome.run_id,
                    "completed": False,
                    "checkpoint_required": False,
                    "valid": None,
                }
            )
            continue
        completed_count += 1
        identity = outcome.checkpoint_identity
        identity_fields_valid = bool(
            isinstance(identity, SCBACheckpointIdentity)
            and identity.run_id == outcome.run_id
            and all(
                _is_sha256(value)
                for value in (
                    identity.configuration_sha256,
                    identity.problem_sha256,
                    identity.initial_state_sha256,
                    identity.code_sha256,
                    identity.environment_sha256,
                    identity.identity_sha256,
                )
            )
        )
        if identity_fields_valid:
            assert identity is not None
            environments.add(identity.environment_sha256)
        result_hashes = (
            outcome.result.metadata.get("input_hashes", {})
            if outcome.result is not None
            else {}
        )
        result_identity_matches = bool(
            identity_fields_valid
            and isinstance(result_hashes, Mapping)
            and result_hashes.get("configuration")
            == identity.configuration_sha256
            and result_hashes.get("problem") == identity.problem_sha256
            and result_hashes.get("initial_state")
            == identity.initial_state_sha256
            and outcome.result is not None
            and outcome.result.metadata.get("source_tree_sha256")
            == identity.code_sha256
        )
        path = (
            None
            if outcome.checkpoint_path is None
            else Path(outcome.checkpoint_path)
        )
        file_record_valid = bool(
            path is not None
            and path.is_file()
            and _is_sha256(outcome.checkpoint_file_sha256)
        )
        actual_file_sha256: str | None = None
        manifest_valid = False
        checkpoint_status: str | None = None
        checkpoint_state_matches = False
        checkpoint_history_matches = False
        if file_record_valid:
            assert path is not None
            try:
                actual_file_sha256 = _file_sha256(path)
                if (
                    actual_file_sha256 == outcome.checkpoint_file_sha256
                    and identity_fields_valid
                ):
                    assert identity is not None
                    loaded = load_scba_checkpoint(
                        path,
                        expected_identity=identity,
                    )
                    checkpoint_status = loaded.status
                    manifest_valid = bool(loaded.status == outcome.status)
                    assert outcome.result is not None
                    checkpoint_state_matches = bool(
                        np.array_equal(
                            loaded.sigma_lesser,
                            outcome.result.sigma_pp.lesser,
                        )
                        and np.array_equal(
                            loaded.sigma_greater,
                            outcome.result.sigma_pp.greater,
                        )
                    )
                    checkpoint_history_matches = bool(
                        loaded.history == outcome.result.history
                    )
            except (CheckpointError, OSError, ValueError, TypeError):
                manifest_valid = False
        valid = bool(
            identity_fields_valid
            and result_identity_matches
            and file_record_valid
            and actual_file_sha256 == outcome.checkpoint_file_sha256
            and manifest_valid
            and checkpoint_state_matches
            and checkpoint_history_matches
        )
        all_completed_valid = bool(all_completed_valid and valid)
        records.append(
            {
                "run_id": outcome.run_id,
                "completed": True,
                "checkpoint_required": True,
                "path": outcome.checkpoint_path,
                "file_sha256": outcome.checkpoint_file_sha256,
                "actual_file_sha256": actual_file_sha256,
                "file_record_valid": file_record_valid,
                "manifest_valid": manifest_valid,
                "checkpoint_state_matches": checkpoint_state_matches,
                "checkpoint_history_matches": checkpoint_history_matches,
                "checkpoint_status": checkpoint_status,
                "identity_sha256": (
                    None if identity is None else identity.identity_sha256
                ),
                "identity_fields_valid": identity_fields_valid,
                "result_identity_matches": result_identity_matches,
                "environment_sha256": (
                    None if identity is None else identity.environment_sha256
                ),
                "resumed": outcome.resumed_from_checkpoint,
                "reconstruction_seconds": outcome.reconstruction_seconds,
                "valid": valid,
            }
        )
    same_environment = bool(
        completed_count > 0
        and len(environments) == 1
        and _is_sha256(next(iter(environments)))
    )
    coverage = bool(completed_count > 0 and all_completed_valid)
    return {
        "passed": bool(coverage and same_environment),
        "checkpointing_required_for_release": True,
        "completed_outcome_count": completed_count,
        "total_outcome_count": len(outcomes),
        "complete_completed_outcome_coverage": coverage,
        "same_environment_sha256": same_environment,
        "environment_sha256": (
            next(iter(environments)) if same_environment else None
        ),
        "records": records,
    }

def run_multidimensional_release_benchmark(
    config: MultidimensionalReleaseConfig,
    *,
    checkpoint_dir: str | Path | None = None,
    environment_sha256: str | None = None,
    resume: bool = True,
) -> MultidimensionalReleaseReport:
    """Execute the preregistered graph, optionally resuming exact checkpoints.

    Checkpointing is private to this benchmark harness.  Enabling it requires
    an externally generated micromamba lock-file SHA256 so a restart cannot
    silently cross source, dependency, machine-environment, grid, or input
    identities.  Existing checkpoints are loaded only on an exact identity
    match and their derived Green functions and diagnostics are recomputed.
    """

    if not isinstance(config, MultidimensionalReleaseConfig):
        raise TypeError("config must be a MultidimensionalReleaseConfig.")
    if config.certification_eligible and checkpoint_dir is None:
        raise ValueError(
            "certification-eligible multidimensional runs require "
            "checkpoint_dir and the explicit micromamba lock-file SHA256."
        )
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool.")
    checkpoint_root: Path | None
    if checkpoint_dir is None:
        if environment_sha256 is not None:
            raise ValueError(
                "environment_sha256 is valid only when checkpoint_dir is set."
            )
        checkpoint_root = None
    else:
        if not _is_sha256(environment_sha256):
            raise ValueError(
                "checkpointing requires environment_sha256 as a lowercase "
                "64-character SHA256."
            )
        checkpoint_root = Path(checkpoint_dir).expanduser().resolve()
        checkpoint_root.mkdir(parents=True, exist_ok=True)

    def execute(
        *,
        run_id: str,
        point: MultidimensionalRunPoint,
        initialization: Initialization,
        backend: Literal["serial", "threaded"],
        parent: MultidimensionalRunOutcome | None = None,
    ) -> MultidimensionalRunOutcome:
        return _execute(
            config,
            run_id=run_id,
            point=point,
            initialization=initialization,
            backend=backend,
            parent=parent,
            checkpoint_dir=checkpoint_root,
            environment_sha256=environment_sha256,
            resume=resume,
        )

    started = time.perf_counter()
    outcomes: list[MultidimensionalRunOutcome] = []
    comparisons: list[MultidimensionalComparison] = []

    endpoint = MultidimensionalRunPoint(
        g=config.executed_g_values[-1],
        bias=config.executed_bias_values[-1],
        numerical=config.numerical,
    )
    direct_zero = execute(
        run_id=_run_id("direct_zero", endpoint),
        point=endpoint,
        initialization="zero",
        backend="serial",
    )
    outcomes.append(direct_zero)

    direct_unlocks_graph = _direct_endpoint_unlocks_graph(
        config,
        direct_zero,
    )
    gate_error = (
        "the hardest direct-zero endpoint did not satisfy the locked "
        "convergence/diagnostic/release gate; every remaining preregistered "
        "point is archived as blocked without altered controls"
    )

    def execute_after_gate(
        *,
        run_id: str,
        point: MultidimensionalRunPoint,
        initialization: Initialization,
        backend: Literal["serial", "threaded"],
        parent: MultidimensionalRunOutcome | None = None,
    ) -> MultidimensionalRunOutcome:
        if not direct_unlocks_graph:
            return _blocked_outcome(
                run_id=run_id,
                point=point,
                initialization=initialization,
                backend=backend,
                parent_id=direct_zero.run_id,
                error_type="HardestEndpointUnavailable",
                error_message=gate_error,
            )
        return execute(
            run_id=run_id,
            point=point,
            initialization=initialization,
            backend=backend,
            parent=parent,
        )

    parent: MultidimensionalRunOutcome | None = None
    equilibrium_endpoint: MultidimensionalRunOutcome | None = None
    path_outcomes: list[MultidimensionalRunOutcome] = []
    for index, (g_value, bias_value) in enumerate(_physical_path(config)):
        point = MultidimensionalRunPoint(
            g=g_value,
            bias=bias_value,
            numerical=config.numerical,
        )
        initialization: Initialization = (
            "zero" if index == 0 else "continuation"
        )
        outcome = execute_after_gate(
            run_id=_run_id(
                "path",
                point,
                suffix=f"step{index:02d}",
            ),
            point=point,
            initialization=initialization,
            backend="serial",
            parent=parent,
        )
        outcomes.append(outcome)
        path_outcomes.append(outcome)
        parent = outcome
        if (
            g_value == config.executed_g_values[-1]
            and bias_value == 0.0
        ):
            equilibrium_endpoint = outcome
    continuation_endpoint = path_outcomes[-1]

    born_seeded = execute_after_gate(
        run_id=_run_id("born_seed", endpoint),
        point=endpoint,
        initialization="born_seed",
        backend="serial",
    )
    outcomes.append(born_seeded)
    threaded = execute_after_gate(
        run_id=_run_id("threaded_zero", endpoint),
        point=endpoint,
        initialization="zero",
        backend="threaded",
    )
    outcomes.append(threaded)

    comparisons.extend(
        (
            _compare_outcomes(
                "zero_vs_born_endpoint",
                "zero_vs_born",
                direct_zero,
                born_seeded,
                field_rtol=config.field_rtol,
                field_atol=config.field_atol,
                current_rtol=config.current_comparison_rtol,
                current_atol=config.current_comparison_atol,
                compare_fields=True,
            ),
            _compare_outcomes(
                "direct_vs_continuation_endpoint",
                "direct_vs_continuation",
                direct_zero,
                continuation_endpoint,
                field_rtol=config.field_rtol,
                field_atol=config.field_atol,
                current_rtol=config.current_comparison_rtol,
                current_atol=config.current_comparison_atol,
                compare_fields=True,
            ),
            _compare_outcomes(
                "serial_vs_threaded_endpoint",
                "serial_vs_threaded",
                direct_zero,
                threaded,
                field_rtol=config.backend_field_rtol,
                field_atol=config.backend_field_atol,
                current_rtol=config.current_comparison_rtol,
                current_atol=config.current_comparison_atol,
                compare_fields=True,
            ),
        )
    )

    if config.run_preregistered_refinements:
        critical_refinement_names = frozenset(
            name
            for name in _required_refinement_comparison_names(config).values()
            if name is not None
        )
        for label, numerical in _refinement_points(config):
            point = MultidimensionalRunPoint(
                g=endpoint.g,
                bias=endpoint.bias,
                numerical=numerical,
            )
            candidate = execute_after_gate(
                run_id=_run_id(
                    "refinement",
                    point,
                    suffix=label,
                ),
                point=point,
                initialization="refinement_zero",
                backend="serial",
            )
            outcomes.append(candidate)
            same_grid = (
                numerical.spacing == config.numerical.spacing
                and numerical.cutoff == config.numerical.cutoff
            )
            comparison_name = f"numerical_refinement_{label}"
            comparison = _compare_outcomes(
                comparison_name,
                "numerical_refinement",
                direct_zero,
                candidate,
                field_rtol=config.field_rtol,
                field_atol=config.field_atol,
                current_rtol=config.refinement_rtol,
                current_atol=config.current_comparison_atol,
                compare_fields=same_grid,
                certification_critical=(
                    comparison_name in critical_refinement_names
                ),
                line_shape_spec=config.spec,
                line_shape_rtol=config.line_shape_rtol,
                linewidth_atol=config.linewidth_atol,
                shift_atol=config.shift_atol,
            )
            refinement_failures = list(comparison.failures)
            if (
                comparison.sigma_tail_left is None
                or comparison.sigma_tail_right is None
                or comparison.sigma_tail_left > config.tail_rtol
                or comparison.sigma_tail_right > config.tail_rtol
            ):
                refinement_failures.append(
                    "interaction self-energy tail exceeds release gate"
                )
            if refinement_failures != list(comparison.failures):
                comparison = replace(
                    comparison,
                    passed=False,
                    failures=tuple(refinement_failures),
                )
            comparisons.append(comparison)

    if equilibrium_endpoint is None:
        equilibrium_endpoint = outcomes[0]
    conservation = (
        _conservation_check(
            "interacting_equilibrium",
            equilibrium_endpoint,
            config,
            equilibrium=True,
        ),
        _conservation_check(
            "interacting_nonequilibrium",
            direct_zero,
            config,
            equilibrium=False,
        ),
    )

    failures: list[str] = []
    for outcome in outcomes:
        if not outcome.successful:
            failures.append(
                f"{outcome.run_id}: {outcome.status}: "
                f"{outcome.error_message}"
            )
        elif outcome.result is not None and not outcome.result.diagnostics.passed:
            failures.append(
                f"{outcome.run_id}: essential diagnostics did not pass"
            )
    for comparison in comparisons:
        if not comparison.passed and comparison.certification_critical:
            failures.append(
                f"{comparison.name}: " + "; ".join(comparison.failures)
            )
    for check in conservation:
        if not check.passed:
            failures.append(
                f"{check.name}: " + "; ".join(check.failures)
            )
    execution_passed = not failures

    certification_failures: list[str] = []
    if not config.certification_eligible:
        certification_failures.append(
            "configuration is explicitly non-certifying"
        )
    if not config.full_physical_ladder_selected:
        certification_failures.append(
            "full preregistered physical ladders were not selected"
        )
    if not config.run_preregistered_refinements:
        certification_failures.append(
            "preregistered numerical refinements were not executed"
        )
    expected_comparison_names = {
        f"numerical_refinement_{label}"
        for label, _ in _refinement_points(config)
    }
    refinement_comparisons = {
        item.name: item
        for item in comparisons
        if item.kind == "numerical_refinement"
    }
    observed_comparison_names = set(refinement_comparisons)
    if observed_comparison_names != expected_comparison_names:
        certification_failures.append(
            "all six numerical refinement points were not archived exactly"
        )
    for direction, required_name in (
        _required_refinement_comparison_names(config).items()
    ):
        if required_name is None:
            certification_failures.append(
                f"preregistered {direction} refinement is not defined"
            )
        elif required_name not in observed_comparison_names:
            certification_failures.append(
                f"required {direction} refinement was not executed"
            )
        else:
            required = refinement_comparisons[required_name]
            if not required.certification_critical:
                certification_failures.append(
                    f"required {direction} refinement is not marked critical"
                )
            if not required.passed:
                certification_failures.append(
                    f"required {direction} refinement did not pass"
                )
    required_names = {
        name
        for name in _required_refinement_comparison_names(config).values()
        if name is not None
    }
    if any(
        item.certification_critical != (item.name in required_names)
        for item in refinement_comparisons.values()
    ):
        certification_failures.append(
            "numerical refinement criticality does not match the locked four directions"
        )
    topology_validation = _multidimensional_topology_summary(
        config,
        tuple(outcomes),
        tuple(comparisons),
        conservation,
    )
    checkpoint_evidence = _multidimensional_checkpoint_summary(
        tuple(outcomes)
    )
    if config.certification_eligible and not topology_validation["passed"]:
        certification_failures.append(
            "exact 14-outcome topology or comparison/conservation references "
            "do not match the locked release graph"
        )
    if config.certification_eligible and not checkpoint_evidence["passed"]:
        certification_failures.append(
            "complete exact-lock HDF5 checkpoint evidence is required for "
            "every completed release outcome"
        )
    weak_coupling = _multidimensional_weak_coupling_summary(
        tuple(outcomes),
        config.spec,
    )
    if not weak_coupling["passed"]:
        certification_failures.append(
            "nine-DOF linewidth/shift weak-coupling gate failed"
        )
    provenance = _multidimensional_provenance_summary(tuple(outcomes))
    if not provenance["complete"]:
        certification_failures.append(
            "multidimensional provenance is incomplete"
        )
    if not provenance["clean"]:
        certification_failures.append(
            "multidimensional source provenance is dirty"
        )
    if not provenance["consistent"]:
        certification_failures.append(
            "multidimensional source/environment identity is inconsistent"
        )
    for outcome in outcomes:
        if outcome.result is None or not outcome.result.release_certified:
            certification_failures.append(
                f"{outcome.run_id} is not release-certified"
            )
    if not execution_passed:
        certification_failures.append("benchmark execution did not pass")
    release_certified = bool(
        config.certification_eligible
        and execution_passed
        and not certification_failures
    )

    return MultidimensionalReleaseReport(
        config=config,
        outcomes=tuple(outcomes),
        comparisons=tuple(comparisons),
        conservation_checks=conservation,
        failures=tuple(failures),
        certification_failures=tuple(certification_failures),
        total_elapsed_seconds=time.perf_counter() - started,
        execution_passed=execution_passed,
        release_certified=release_certified,
    )


__all__ = [
    "MultidimensionalComparison",
    "MultidimensionalConservationCheck",
    "MultidimensionalFieldError",
    "MultidimensionalLadderSpec",
    "MultidimensionalLineShapeComparison",
    "MultidimensionalLineShapeMeasurement",
    "MultidimensionalNumericalPoint",
    "MultidimensionalReleaseConfig",
    "MultidimensionalReleaseReport",
    "MultidimensionalRunOutcome",
    "MultidimensionalRunPoint",
    "build_multidimensional_problem",
    "full_multidimensional_release_config",
    "pr_multidimensional_release_config",
    "run_multidimensional_release_benchmark",
]
