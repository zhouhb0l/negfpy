"""Python-only runners for the preregistered production toy gate.

The functions in this module make the release calculation reproducible
without adding an anharmonic command-line interface.  The release runner is
intentionally expensive and never runs implicitly during import or ordinary
pull-request tests.
"""

from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path
import time
from uuid import uuid4

from ..grid import SignedFrequencyGrid
from ..solver import solve_one_shot_born
from .multidimensional_release import (
    MultidimensionalReleaseReport,
    _direct_endpoint_unlocks_graph,
    full_multidimensional_release_config,
    pr_multidimensional_release_config,
    run_multidimensional_release_benchmark,
)
from .open_three_mode import (
    OpenThreeModeFixture,
    open_three_mode_oracle,
)
from .production_gate_report import (
    OpenThreeModeBornRun,
    ProductionToyGateReport,
    build_production_toy_gate_report,
)



class ProductionToyGateBlockedError(RuntimeError):
    """Raised after archiving a direct-endpoint-gated release failure."""

    def __init__(
        self,
        message: str,
        report: MultidimensionalReleaseReport,
        artifact_path: Path,
    ) -> None:
        super().__init__(message)
        self.report = report
        self.artifact_path = artifact_path


def _write_multidimensional_report_atomic(
    report: MultidimensionalReleaseReport,
    path: str | Path,
) -> Path:
    destination = Path(path).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    payload = (report.to_json(indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return destination


def _validate_environment_lock_sha256(value: str | None) -> None:
    if value is None:
        return
    valid = isinstance(value, str) and len(value) == 64
    if valid:
        try:
            int(value, 16)
        except ValueError:
            valid = False
        else:
            valid = value == value.lower()
    if not valid:
        raise ValueError(
            "environment_lock_sha256 must be None or a lowercase "
            "64-character SHA256 digest."
        )

def evaluate_open_three_mode_born(
    name: str,
    fixture: OpenThreeModeFixture,
    *,
    integration_epsabs: float = 1.0e-9,
    integration_epsrel: float = 1.0e-7,
    integration_limit: int = 200,
    environment_lock_sha256: str | None = None,
) -> OpenThreeModeBornRun:
    """Evaluate one independent-oracle/production-Born comparison."""

    _validate_environment_lock_sha256(environment_lock_sha256)

    started = time.perf_counter()
    oracle = open_three_mode_oracle(
        fixture,
        integration_epsabs=integration_epsabs,
        integration_epsrel=integration_epsrel,
        integration_limit=integration_limit,
    )
    oracle_seconds = time.perf_counter() - started

    started = time.perf_counter()
    production = solve_one_shot_born(fixture.problem, fixture.config)
    production_seconds = time.perf_counter() - started
    return OpenThreeModeBornRun(
        name=name,
        fixture=fixture,
        oracle=oracle,
        production=production,
        oracle_wall_seconds=oracle_seconds,
        production_wall_seconds=production_seconds,
        environment_lock_sha256=environment_lock_sha256,
    )


def open_three_mode_release_fixtures(
    *,
    base_grid: SignedFrequencyGrid | None = None,
    base_eta_lead: float = 4.5e-5,
) -> tuple[tuple[str, OpenThreeModeFixture], ...]:
    """Return the locked open-system base matrix and isolated refinements.

    The base matrix is the Cartesian product of temperatures ``(0, 0.7)`` and
    signed couplings ``(-0.02, -0.01, 0, 0.01, 0.02)``.  The three additional
    runs refine spacing, cutoff, and lead eta independently about the
    ``T=0.7, g=0.02`` anchor.
    """

    grid = (
        SignedFrequencyGrid(step=7.0e-5, n_positive=71571)
        if base_grid is None
        else base_grid
    )
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("base_grid must be a SignedFrequencyGrid or None.")
    if grid.cutoff < 3.0 * 1.5:
        raise ValueError(
            "the release base-grid cutoff must be at least three times the "
            "largest harmonic mode frequency."
        )

    specifications: list[tuple[str, OpenThreeModeFixture]] = []
    for temperature in (0.0, 0.7):
        for g_value in (-0.02, -0.01, 0.0, 0.01, 0.02):
            temperature_label = str(temperature).replace(".", "p")
            coupling_label = str(g_value).replace("-", "m").replace(".", "p")
            specifications.append(
                (
                    f"base_T{temperature_label}_g{coupling_label}",
                    OpenThreeModeFixture(
                        grid=grid,
                        temperature=temperature,
                        g=g_value,
                        eta_lead=base_eta_lead,
                        backend="fft",
                        hilbert_backend="fft",
                        surface_gf_method="analytic_diagonal",
                    ),
                )
            )

    anchor_kwargs = {
        "temperature": 0.7,
        "g": 0.02,
    }
    specifications.extend(
        (
            (
                "refine_spacing_half",
                OpenThreeModeFixture(
                    grid=SignedFrequencyGrid(
                        step=grid.step / 2.0,
                        n_positive=2 * grid.n_positive,
                    ),
                    eta_lead=base_eta_lead,
                    backend="fft",
                    hilbert_backend="fft",
                    surface_gf_method="analytic_diagonal",
                    **anchor_kwargs,
                ),
            ),
            (
                "refine_cutoff_6",
                OpenThreeModeFixture(
                    grid=SignedFrequencyGrid(
                        step=grid.step,
                        n_positive=int(round(6.0 / grid.step)),
                    ),
                    eta_lead=base_eta_lead,
                    backend="fft",
                    hilbert_backend="fft",
                    surface_gf_method="analytic_diagonal",
                    **anchor_kwargs,
                ),
            ),
            (
                "refine_eta_half",
                OpenThreeModeFixture(
                    grid=grid,
                    eta_lead=base_eta_lead / 2.0,
                    backend="fft",
                    hilbert_backend="fft",
                    surface_gf_method="analytic_diagonal",
                    **anchor_kwargs,
                ),
            ),
        )
    )
    return tuple(specifications)


def evaluate_open_three_mode_release_matrix(
    fixtures: Sequence[tuple[str, OpenThreeModeFixture]] | None = None,
    *,
    environment_lock_sha256: str | None = None,
) -> tuple[OpenThreeModeBornRun, ...]:
    """Evaluate every preregistered open-system release point serially."""

    _validate_environment_lock_sha256(environment_lock_sha256)

    selected = (
        open_three_mode_release_fixtures()
        if fixtures is None
        else tuple(fixtures)
    )
    return tuple(
        evaluate_open_three_mode_born(
            name,
            fixture,
            environment_lock_sha256=environment_lock_sha256,
        )
        for name, fixture in selected
    )


def run_pull_request_production_toy_gate() -> ProductionToyGateReport:
    """Run the bounded, irrevocably non-certifying pull-request profile."""

    fixture = OpenThreeModeFixture(
        grid=SignedFrequencyGrid(step=0.25, n_positive=21),
        temperature=0.7,
        g=0.0,
        eta_lead=0.05,
    )
    open_run = evaluate_open_three_mode_born(
        "pr_coarse_T0p7_g0",
        fixture,
        integration_epsabs=1.0e-7,
        integration_epsrel=1.0e-5,
        integration_limit=50,
    )
    multidimensional = run_multidimensional_release_benchmark(
        pr_multidimensional_release_config()
    )
    return build_production_toy_gate_report(
        (open_run,),
        multidimensional,
        validation_tier="pull_request",
    )



def run_release_production_toy_gate(
    *,
    open_runs: Sequence[OpenThreeModeBornRun] | None = None,
    multidimensional_report: MultidimensionalReleaseReport | None = None,
    checkpoint_dir: str | Path | None = None,
    environment_sha256: str | None = None,
    resume: bool = True,
    multidimensional_failure_report_path: str | Path | None = None,
) -> ProductionToyGateReport:
    """Run or combine the complete release-tier production toy evidence.

    Internally computed evidence always starts with the checkpointed nine-DOF
    direct-zero endpoint.  The open-system matrix is intentionally not started
    when that mandatory first endpoint fails.  Because the combined report
    requires at least one completed open-system comparison, this failure path
    atomically archives the complete multidimensional strict report and raises
    :class:`ProductionToyGateBlockedError` carrying both report and path.
    """

    selected_multidimensional = (
        run_multidimensional_release_benchmark(
            full_multidimensional_release_config(),
            checkpoint_dir=checkpoint_dir,
            environment_sha256=environment_sha256,
            resume=resume,
        )
        if multidimensional_report is None
        else multidimensional_report
    )

    if open_runs is None:
        direct_endpoint = selected_multidimensional.outcomes[0]
        direct_gate_passed = _direct_endpoint_unlocks_graph(
            selected_multidimensional.config,
            direct_endpoint,
        )
        if not direct_gate_passed:
            failure_path = (
                Path(multidimensional_failure_report_path)
                if multidimensional_failure_report_path is not None
                else (
                    Path(checkpoint_dir).expanduser().parent
                    / "multidimensional_toy_gate_failure.json"
                    if checkpoint_dir is not None
                    else Path(
                        "validation-artifacts/"
                        "multidimensional_toy_gate_failure.json"
                    )
                )
            )
            archived = _write_multidimensional_report_atomic(
                selected_multidimensional,
                failure_path,
            )
            raise ProductionToyGateBlockedError(
                "open-system matrix blocked because the mandatory direct-zero "
                f"endpoint ended with status {direct_endpoint.status!r}; "
                f"multidimensional evidence was archived at {archived}",
                selected_multidimensional,
                archived,
            )
        selected_open_runs = evaluate_open_three_mode_release_matrix(
            environment_lock_sha256=environment_sha256,
        )
    else:
        selected_open_runs = tuple(open_runs)

    return build_production_toy_gate_report(
        selected_open_runs,
        selected_multidimensional,
        validation_tier="release",
    )

__all__ = [
    "ProductionToyGateBlockedError",
    "evaluate_open_three_mode_born",
    "evaluate_open_three_mode_release_matrix",
    "open_three_mode_release_fixtures",
    "run_pull_request_production_toy_gate",
    "run_release_production_toy_gate",
]
