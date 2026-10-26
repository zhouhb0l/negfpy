"""Portable complete-energy SCBA workflows using caller-prepared Python inputs.

Import these functions from an application or notebook.  There are no material
paths, scheduler commands, automatic fits, or calculations at import time.

Prepare a SiteEnergyModel with ``fit_site_energy_model`` or restore its exact
preserved coefficients with ``restore_site_energy_model`` from
``negfpy.anharmonic.complete_energy``.  Keep its measured fit residual explicit.
Supply compatible crystal/device mappings, harmonic matrices, signed-frequency
grid and paired retarded/thermal contacts.  Contacts determine eta and reservoir
temperatures; neither solver below rebuilds or changes them.

For example, after preparing those inputs::

    mapped = build_complete_vertex(
        mesh, model, crystal, device_mapping,
        n_device_layers=n_layers, interaction_layers=center_layers,
        scale=scale, force_constant_units="eV/angstrom^3", mass_units="amu",
        max_bytes=vertex_byte_budget, max_workspace_bytes=workspace_budget,
    )
    config = ChannelSCBAConfig(max_evaluations=16, mixing=0.05)
    result = solve_from_zero(
        grid, mapped.vertex, dynamical, left, right,
        scale=scale, g=0.03, config=config,
    )

This example config retains the library's original physical and convergence
thresholds.  Set resource/time bounds for the actual problem before running.
Inspect ``converged``, ``stop_reason`` and ``failure_reason``: a completed map
alone is not convergence.  Numerical convergence concerns the supplied model
and grid, and does not establish physical-interface or mesh accuracy.
"""

from __future__ import annotations

from negfpy.anharmonic.complete_energy import (
    ChannelAccelerationConfig,
    ChannelAccelerationResult,
    ChannelProposalFeasibility,
    ChannelSCBAConfig,
    ChannelSCBAResult,
    ChannelSelectionPolicy,
    ChannelSpectrum,
    FourierPackedMaterialCubic,
    PackedTransverseCubicVertex,
    SignedFrequencyGrid,
    SiteEnergyModel,
    SolverScale,
    UniformBZMesh,
    build_site_energy_vertex,
    run_channel_accel,
    run_channel_scba,
)
from negfpy.anharmonic.crystal import CrystalMapping
from negfpy.anharmonic.layered import LayeredDeviceMapping


def build_complete_vertex(
    mesh: UniformBZMesh,
    model: SiteEnergyModel,
    crystal: CrystalMapping,
    device_mapping: LayeredDeviceMapping,
    *,
    n_device_layers: int,
    interaction_layers: tuple[int, ...],
    scale: SolverScale,
    force_constant_units: str,
    mass_units: str,
    max_bytes: int,
    max_workspace_bytes: int,
) -> FourierPackedMaterialCubic:
    """Select whole center energies and retain their complete endpoint support.

    The existing builder checks geometry, units and memory admission.  It raises
    if any required endpoint leaves the harmonic device.  The returned metadata
    records model identity and support; keep it with the scientific result.
    """
    return build_site_energy_vertex(
        mesh, model, crystal, device_mapping,
        n_device_layers=n_device_layers, interaction_layers=interaction_layers,
        scale=scale, force_constant_units=force_constant_units, mass_units=mass_units,
        crop_derivatives=False, max_bytes=max_bytes,
        max_workspace_bytes=max_workspace_bytes,
    )


def solve_from_zero(
    grid: SignedFrequencyGrid,
    vertex: PackedTransverseCubicVertex,
    dynamical,
    left,
    right,
    *,
    scale: SolverScale,
    g: float,
    config: ChannelSCBAConfig,
) -> ChannelSCBAResult:
    """Run the existing bounded fixed-mixing solver from exact zero Sigma."""
    return run_channel_scba(
        grid, vertex, dynamical, left, right,
        sigma_initial=None, first_index=1, scale=scale, g=g, config=config,
    )


def continue_with_anderson(
    grid: SignedFrequencyGrid,
    vertex: PackedTransverseCubicVertex,
    dynamical,
    left,
    right,
    *,
    seed_input: ChannelSpectrum,
    seed_mapped: ChannelSpectrum,
    seed_index: int,
    scale: SolverScale,
    g: float,
    config: ChannelSCBAConfig,
) -> ChannelAccelerationResult:
    """Continue a compatible evaluated seed with bounded depth-one Anderson.

    Authenticate a saved seed with ``load_channel_checkpoint`` and its expected
    binding before use, or supply a previously computed input/image pair.  Its
    Hamiltonian, grid, geometry, units, contacts and coupling must match this
    run.  Pass evaluated input and UNEVALUATED mapped image separately.  The
    core mixes them once for a fresh warm-up and counts rejected full maps in
    ``config.max_evaluations``.

    This uses strict original-component/actual-merit nonincrease selection and
    finite-PSD-only proposal backtracking.  Physical guards and all six SCBA
    screens come unchanged from ``config``.  No solver is reimplemented here.
    ``result.state`` is the selected anchor; ``last_evaluated_state`` may be a
    rejected later trial.  Neither role by itself is scientific acceptance.
    """
    return run_channel_accel(
        grid, vertex, dynamical, left, right,
        seed_input=seed_input, seed_mapped=seed_mapped, seed_index=seed_index,
        scale=scale, g=g, config=config,
        acceleration=ChannelAccelerationConfig(),
        selection=ChannelSelectionPolicy(),
        feasibility=ChannelProposalFeasibility(mode="psd-backtracking"),
    )
