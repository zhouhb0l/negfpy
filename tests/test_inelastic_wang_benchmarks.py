import numpy as np

from negfpy.inelastic import (
    quartic_scmf_current_vs_temperature,
    quartic_scmf_static_self_energy,
    wang_review_fig4_audit,
    wang_review_fig4_paper_convention_spec,
    wang_review_fig4_paper_convention_sweep,
    wang_review_fig4_one_particle_cases,
    wang_review_fig4_one_particle_cases_si,
    wang_review_fig4_two_particle_cases_si,
    wang_review_hbar_effective_si,
    wang_review_kb_effective_si,
    wang_review_fig4_two_particle_cases,
)


def test_wang_one_particle_case_returns_finite_static_self_energy() -> None:
    case = wang_review_fig4_one_particle_cases()["black"]
    sigma, info = quartic_scmf_static_self_energy(
        case,
        temp_left=125.0,
        temp_right=75.0,
        omega_max=150.0,
        n_omega_cov=121,
        kb_effective=0.08617333262,
        max_iter=20,
        tol=1e-5,
    )
    assert sigma.shape == (1, 1)
    assert np.all(np.isfinite(sigma))
    assert bool(info["iterations"]) >= 1
    assert np.isclose(case.bath_left.gamma, 6.0321**2)


def test_wang_two_particle_current_sweep_returns_finite_arrays() -> None:
    case = wang_review_fig4_two_particle_cases()["red"]
    result = quartic_scmf_current_vs_temperature(
        case,
        np.array([200.0, 400.0], dtype=float),
        omega_max=180.0,
        n_omega=121,
        n_omega_cov=121,
        kb_effective=0.08617333262,
        max_iter=20,
        tol=1e-5,
    )
    assert result["current_internal"].shape == (2,)
    assert result["transmission_map"].shape == (2, 121)
    assert np.all(np.isfinite(result["current_internal"]))
    assert np.all(result["current_internal"] >= 0.0)


def test_wang_audit_reports_exact_and_open_items() -> None:
    audit = wang_review_fig4_audit()
    assert audit["paper"]["target_figure"] == "Fig. 4"
    assert "bath_functional_form" in audit["matched_exactly_in_code"]
    assert "absolute_current_scale" in audit["still_under_audit"]


def test_wang_paper_convention_spec_matches_caption_values() -> None:
    spec = wang_review_fig4_paper_convention_spec("two_particle")
    assert spec["bath"]["epsilon_meV_per_A2_u"] == 6.0321
    assert spec["harmonic"]["K12_meV_per_A2_u"] == -30.165
    assert spec["quartic_curves_eV_per_A4_u2"]["blue"]["T1112"] == -2.4


def test_wang_paper_convention_sweep_returns_three_curves() -> None:
    result = wang_review_fig4_paper_convention_sweep(
        "one_particle",
        np.array([300.0], dtype=float),
        omega_max=150.0,
        n_omega=81,
        n_omega_cov=81,
        kb_effective=0.08617333262,
        max_iter=15,
        tol=1e-5,
    )
    assert result["spec"]["family"] == "one_particle"
    assert set(result["results"]) == {"black", "red", "blue"}
    assert np.isfinite(result["results"]["black"]["current_internal"][0])


def test_wang_one_particle_si_case_returns_finite_current() -> None:
    case = wang_review_fig4_one_particle_cases_si()["black"]
    result = quartic_scmf_current_vs_temperature(
        case,
        np.array([300.0, 600.0], dtype=float),
        omega_max=2.0e13,
        n_omega=101,
        n_omega_cov=101,
        kb_effective=wang_review_kb_effective_si(),
        hbar_effective=wang_review_hbar_effective_si(),
        max_iter=20,
        tol=1e-5,
    )
    assert np.all(np.isfinite(result["current_internal"]))
    assert np.all(result["current_internal"] >= 0.0)


def test_wang_two_particle_si_case_returns_finite_current() -> None:
    case = wang_review_fig4_two_particle_cases_si()["red"]
    result = quartic_scmf_current_vs_temperature(
        case,
        np.array([300.0], dtype=float),
        omega_max=2.0e13,
        n_omega=101,
        n_omega_cov=101,
        kb_effective=wang_review_kb_effective_si(),
        hbar_effective=wang_review_hbar_effective_si(),
        max_iter=20,
        tol=1e-5,
    )
    assert np.all(np.isfinite(result["current_internal"]))
    assert np.all(result["current_internal"] >= 0.0)
