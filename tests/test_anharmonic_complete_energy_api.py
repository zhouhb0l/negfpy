"""Public API and portable workflow smoke cases; tiny synthetic inputs only."""

from importlib import import_module, util
from pathlib import Path

import pytest

from negfpy.anharmonic import complete_energy


def _example():
    path = Path(__file__).resolve().parents[1] / "examples" / "complete_energy_scba.py"
    spec = util.spec_from_file_location("complete_energy_scba_example", path)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_exports_are_the_existing_implementations_without_solver_forks():
    assert len(complete_energy.__all__) == len(set(complete_energy.__all__))
    origins = {
        "site_energy_model": ["SiteEnergyModel", "SiteEnergyStar", "fit_site_energy_model"],
        "site_energy_saved": ["restore_site_energy_model"],
        "site_energy_boundary": ["build_site_energy_vertex"],
        "transverse_channel_feedback": [
            "ChannelSpectrum", "ChannelSCBAConfig", "ChannelSCBAResult",
            "evaluate_channel_born_map", "mix_channel_spectra", "run_channel_scba",
        ],
        "transverse_channel_accel": [
            "ChannelAccelerationConfig", "ChannelSelectionPolicy", "run_channel_accel",
        ],
        "transverse_channel_feasibility": ["ChannelProposalFeasibility"],
        "transverse_channel_restart": [
            "ChannelCheckpoint", "load_channel_checkpoint", "save_channel_checkpoint",
        ],
    }
    for source, names in origins.items():
        implementation = import_module("negfpy.anharmonic." + source)
        for name in names:
            assert name in complete_energy.__all__
            assert getattr(complete_energy, name) is getattr(implementation, name)
    for name in complete_energy.__all__:
        assert getattr(complete_energy, name) is not None


def test_portable_complete_vertex_keeps_required_outer_derivatives():
    from test_site_energy_boundary import _inputs

    example = _example()
    mapping, crystal, model = _inputs()
    mapped = example.build_complete_vertex(
        complete_energy.UniformBZMesh((1, 1, 1)), model, crystal, mapping,
        n_device_layers=3, interaction_layers=(1,),
        scale=complete_energy.SolverScale(omega0=2., hbar=3.),
        force_constant_units="J/m^3", mass_units="kg",
        max_bytes=1024**2, max_workspace_bytes=1024**2,
    )
    assert mapped.metadata["all_outer_derivatives_retained"]
    assert not mapped.metadata["crop_derivatives"]
    outer_start = 2 * mapping.n_principal_layer_dofs
    assert any(mapped.vertex.active_dofs >= outer_start)


def test_portable_zero_and_continuation_use_original_six_screens():
    from test_anharmonic_transverse_channel_feedback import _config, _fixture

    example = _example()
    arguments = _fixture(nq=1)
    config = _config(max_evaluations=1, mixing=.05)
    scale = complete_energy.SolverScale(omega0=1., hbar=1e-34)
    # Zero coupling gives an exact, independently understandable fixed point.
    fresh = example.solve_from_zero(*arguments, scale=scale, g=0., config=config)
    assert fresh.converged and fresh.stop_reason == "fixed_point"
    assert len(fresh.history) == 1
    assert fresh.state.sigma_input is None
    assert fresh.state.record.sigma_passed and fresh.state.record.current_passed
    assert len(fresh.state.record.sigma_residuals) == 3
    assert len(fresh.state.record.current_residuals) == 3

    seed_input = complete_energy.mix_channel_spectra(
        None, fresh.state.evaluation.map_sigma, 1.,
    )
    continued = example.continue_with_anderson(
        *arguments, seed_input=seed_input, seed_mapped=fresh.state.evaluation.map_sigma,
        seed_index=1, scale=scale, g=0., config=config,
    )
    assert continued.converged and continued.stop_reason == "fixed_point"
    assert len(continued.history) == 1
    assert continued.state is continued.last_evaluated_state
    assert continued.state.index == 2
    assert continued.state.record.sigma_passed and continued.state.record.current_passed
    assert continued.decisions[0]["merit_evaluated"] is False
    assert config.sigma_rtol == pytest.approx(1e-4)
    assert config.current_rtol == pytest.approx(1e-3)
    assert config.identity_rtol == pytest.approx(1e-8)
