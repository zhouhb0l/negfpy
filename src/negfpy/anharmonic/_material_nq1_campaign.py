"""Frozen task graph for the silicon ``N_q=1`` material-SCBA gate.

This module describes complete SCBA cases only.  MPI orchestration sits above
the solver and may schedule independent tasks concurrently, but it may not
alter these controls or start a continuation child before its parent has a
validated, converged checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Mapping


_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
HBAR_J_S = 1.0545718176461565e-34
KB_J_K = 1.380649e-23
OMEGA0_RAD_S = 2.0 * math.pi * 1.0e13
HARMONIC_CONSERVATIVE_BOUND = 2.463999
MATERIAL_NQ1_MIXING_FLOOR = 0.001
MATERIAL_NQ1_ACCEPTANCE = MappingProxyType({
    "fixed_point_rtol": 1.0e-8,
    "consecutive_passes": 3,
    "field_equality_rtol": 1.0e-8,
    "backend_equality_rtol": 1.0e-10,
    "absolute_field_tolerance": 1.0e-12,
    "conservation_rtol": 1.0e-5,
    "current_atol_j0": 1.0e-10,
    "refinement_rtol": 5.0e-3,
    "tail_rtol": 1.0e-6,
    "weak_coupling_ratio": 0.1,
})
MATERIAL_NQ1_EXCLUDED_CLAIMS = (
    "transverse-momentum convergence",
    "bulk-film prediction",
    "anharmonic leads",
    "quartic and four-phonon diagrams",
)

ContinuationPredictor = Literal["identity", "coupling-square"]


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def kelvin_to_solver_temperature(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("temperature must be finite")
    return result * KB_J_K / (HBAR_J_S * OMEGA0_RAD_S)


@dataclass(frozen=True, slots=True)
class MaterialSCBATask:
    """One immutable complete material-SCBA calculation."""

    task_id: str
    role: str
    coupling_scale: float
    grid_step: float
    grid_n_positive: int
    eta_lead: float
    mixing: float
    mixing_floor: float
    mean_temperature_kelvin: float
    temperature_bias_kelvin: float
    initialization: Literal["zero", "born", "continuation"]
    continuation_parent: str | None = None
    continuation_axis: Literal["coupling", "temperature_bias"] | None = None
    continuation_predictor: ContinuationPredictor | None = None

    schema: ClassVar[str] = "negfpy.material-nq1-scba-task.v3"

    def __post_init__(self) -> None:
        for name in ("task_id", "role"):
            value = getattr(self, name)
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{name} is invalid")
        for name in (
            "coupling_scale", "grid_step", "eta_lead", "mixing",
            "mixing_floor",
            "mean_temperature_kelvin", "temperature_bias_kelvin",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a finite real number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if not 0.0 < self.coupling_scale <= 1.0:
            raise ValueError("coupling_scale must lie in (0, 1]")
        if self.grid_step <= 0.0 or self.eta_lead <= 0.0:
            raise ValueError("grid_step and eta_lead must be positive")
        if isinstance(self.grid_n_positive, bool) or not isinstance(
            self.grid_n_positive, int
        ) or self.grid_n_positive <= 0:
            raise ValueError("grid_n_positive must be a positive integer")
        if not 0.01 <= self.mixing <= 1.0:
            raise ValueError("mixing must lie in [0.01, 1]")
        if not 0.0 < self.mixing_floor <= self.mixing:
            raise ValueError("mixing_floor must lie in (0, mixing]")
        if self.mean_temperature_kelvin < 0.0 or (
            self.mean_temperature_kelvin
            < 0.5 * abs(self.temperature_bias_kelvin)
        ):
            raise ValueError("reservoir temperature would be negative")
        if self.initialization == "continuation":
            if self.continuation_parent is None or self.continuation_axis not in {
                "coupling", "temperature_bias"
            }:
                raise ValueError("continuation tasks require one parent and axis")
            if self.continuation_predictor not in {
                "identity", "coupling-square"
            }:
                raise ValueError(
                    "continuation tasks require an explicit predictor"
                )
            if (
                self.continuation_axis == "temperature_bias"
                and self.continuation_predictor != "identity"
            ):
                raise ValueError(
                    "temperature-bias continuation requires identity predictor"
                )
            if self.continuation_parent == self.task_id:
                raise ValueError("a continuation task cannot be its own parent")
        elif (
            self.continuation_parent is not None
            or self.continuation_axis is not None
            or self.continuation_predictor is not None
        ):
            raise ValueError(
                "non-continuation tasks cannot name a parent, axis, or predictor"
            )

    @property
    def grid_cutoff(self) -> float:
        return self.grid_step * self.grid_n_positive

    @property
    def temperature_bias_solver(self) -> float:
        return kelvin_to_solver_temperature(self.temperature_bias_kelvin)

    @property
    def temperatures_kelvin(self) -> tuple[float, float]:
        half = 0.5 * self.temperature_bias_kelvin
        return self.mean_temperature_kelvin + half, self.mean_temperature_kelvin - half

    def as_dict(self) -> dict[str, Any]:
        left, right = self.temperatures_kelvin
        return {
            "schema": self.schema,
            "task_id": self.task_id,
            "role": self.role,
            "coupling_scale": self.coupling_scale,
            "grid_step": self.grid_step,
            "grid_n_positive": self.grid_n_positive,
            "grid_n_points": 2 * self.grid_n_positive + 1,
            "grid_cutoff": self.grid_cutoff,
            "eta_lead": self.eta_lead,
            "mixing": self.mixing,
            "mixing_floor": self.mixing_floor,
            "mean_temperature_kelvin": self.mean_temperature_kelvin,
            "temperature_bias_kelvin": self.temperature_bias_kelvin,
            "temperature_bias_solver": self.temperature_bias_solver,
            "left_temperature_kelvin": left,
            "right_temperature_kelvin": right,
            "initialization": self.initialization,
            "continuation_parent": self.continuation_parent,
            "continuation_axis": self.continuation_axis,
            "continuation_predictor": self.continuation_predictor,
        }

    @property
    def task_sha256(self) -> str:
        return _sha256_bytes(_canonical(self.as_dict()))


def silicon_nq1_release_tasks() -> tuple[MaterialSCBATask, ...]:
    """Return the preregistered material gate in canonical dependency order."""

    base = {
        # The earlier 0.3 screening spacing cannot certify weak coupling:
        # 10*Delta-omega exceeds the complete silicon harmonic band.  The
        # release grid therefore resolves a nonempty, preregistered mode set.
        "grid_step": 0.1,
        "grid_n_positive": 75,
        "eta_lead": 0.025,
        "mixing": 0.2,
        # This post-7b0c169 development control is part of every task identity.
        # It defines a new candidate and cannot retroactively certify the
        # earlier frozen 0.01-floor campaign.
        "mixing_floor": MATERIAL_NQ1_MIXING_FLOOR,
        "mean_temperature_kelvin": 300.0,
        "temperature_bias_kelvin": 0.0,
    }
    tasks: list[MaterialSCBATask] = []
    previous: str | None = None
    for token, coupling in (
        ("0001", 0.001), ("001", 0.01), ("0025", 0.025),
        ("005", 0.05), ("010", 0.1),
        ("025", 0.25), ("050", 0.5), ("075", 0.75), ("100", 1.0),
    ):
        task_id = f"coupling-g{token}"
        tasks.append(MaterialSCBATask(
            task_id=task_id,
            role="coupling-path",
            coupling_scale=coupling,
            initialization="zero" if previous is None else "continuation",
            continuation_parent=previous,
            continuation_axis=None if previous is None else "coupling",
            continuation_predictor=(
                None if previous is None else "coupling-square"
            ),
            **base,
        ))
        previous = task_id

    equilibrium = "coupling-g100"
    for sign, prefix in ((1.0, "positive"), (-1.0, "negative")):
        half_id = f"bias-{prefix}-half"
        tasks.append(MaterialSCBATask(
            task_id=half_id,
            role="bias-continuation",
            coupling_scale=1.0,
            temperature_bias_kelvin=sign * 10.0,
            initialization="continuation",
            continuation_parent=equilibrium,
            continuation_axis="temperature_bias",
            continuation_predictor="identity",
            **{k: v for k, v in base.items() if k != "temperature_bias_kelvin"},
        ))
        tasks.append(MaterialSCBATask(
            task_id=f"bias-{prefix}-full",
            role="bias-endpoint",
            coupling_scale=1.0,
            temperature_bias_kelvin=sign * 20.0,
            initialization="continuation",
            continuation_parent=half_id,
            continuation_axis="temperature_bias",
            continuation_predictor="identity",
            **{k: v for k, v in base.items() if k != "temperature_bias_kelvin"},
        ))

    tasks.extend((
        MaterialSCBATask(
            task_id="endpoint-zero-start", role="initialization-equality",
            coupling_scale=1.0, initialization="zero", **base,
        ),
        MaterialSCBATask(
            task_id="endpoint-born-start", role="initialization-equality",
            coupling_scale=1.0, initialization="born", **base,
        ),
        MaterialSCBATask(
            task_id="refinement-half-spacing", role="numerical-refinement",
            coupling_scale=1.0, initialization="born",
            **{
                **base, "grid_step": 0.05, "grid_n_positive": 150,
                "temperature_bias_kelvin": 20.0,
            },
        ),
        MaterialSCBATask(
            task_id="refinement-larger-cutoff", role="numerical-refinement",
            coupling_scale=1.0, initialization="born",
            **{
                **base, "grid_n_positive": 90,
                "temperature_bias_kelvin": 20.0,
            },
        ),
        MaterialSCBATask(
            task_id="refinement-half-eta", role="numerical-refinement",
            coupling_scale=1.0, initialization="born",
            **{
                **base, "eta_lead": 0.0125,
                "temperature_bias_kelvin": 20.0,
            },
        ),
        MaterialSCBATask(
            task_id="refinement-mixing-010", role="numerical-refinement",
            coupling_scale=1.0, initialization="born",
            **{
                **base, "mixing": 0.1,
                "temperature_bias_kelvin": 20.0,
            },
        ),
    ))
    result = tuple(tasks)
    by_id = {task.task_id: task for task in result}
    if len(by_id) != len(result):
        raise RuntimeError("material SCBA task identifiers are not unique")
    for index, task in enumerate(result):
        if task.continuation_parent is None:
            continue
        if task.continuation_parent not in by_id:
            raise RuntimeError("material SCBA task names an unknown parent")
        if result.index(by_id[task.continuation_parent]) >= index:
            raise RuntimeError("material SCBA task graph is not topologically ordered")
    return result


def silicon_nq1_release_manifest(
    *,
    source_revision: str,
    source_tree_sha256: str,
    environment_sha256: str,
    input_hashes: Mapping[str, str],
) -> Mapping[str, Any]:
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("source_revision must be nonempty")
    for name, digest in {
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        **dict(input_hashes),
    }.items():
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise ValueError(f"{name} must be a lowercase SHA256")
    tasks = silicon_nq1_release_tasks()
    payload = {
        "schema": "negfpy.material-nq1-scba-campaign.v3",
        "scientific_scope": "silicon N_q=1 algebraic material validation",
        "source_revision": source_revision,
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "input_hashes": dict(sorted(input_hashes.items())),
        "omega0_rad_s": OMEGA0_RAD_S,
        "harmonic_conservative_bound": HARMONIC_CONSERVATIVE_BOUND,
        "minimum_cutoff_ratio": 3.0,
        "tasks": [task.as_dict() | {"task_sha256": task.task_sha256} for task in tasks],
        "acceptance": dict(MATERIAL_NQ1_ACCEPTANCE),
        "excluded": list(MATERIAL_NQ1_EXCLUDED_CLAIMS),
    }
    manifest = dict(payload)
    manifest["campaign_sha256"] = _sha256_bytes(_canonical(payload))
    return MappingProxyType(manifest)


__all__ = [
    "ContinuationPredictor",
    "HARMONIC_CONSERVATIVE_BOUND",
    "MATERIAL_NQ1_ACCEPTANCE",
    "MATERIAL_NQ1_EXCLUDED_CLAIMS",
    "MATERIAL_NQ1_MIXING_FLOOR",
    "MaterialSCBATask",
    "OMEGA0_RAD_S",
    "kelvin_to_solver_temperature",
    "silicon_nq1_release_manifest",
    "silicon_nq1_release_tasks",
]
