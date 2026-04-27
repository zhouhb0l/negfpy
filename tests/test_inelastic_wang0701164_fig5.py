import numpy as np

from negfpy.inelastic import (
    cubic_onsite_effective_transmission,
    cubic_onsite_current,
    cubic_onsite_current_exact,
    cubic_onsite_eq84_numeric_decomposition,
    wang0701164_compare_sweep_to_digitized,
    wang0701164_compare_sweep_to_vector_pdf,
    wang0701164_extract_fig5_vector_curves,
    wang0701164_fig5_lowest_order_conductance_sweep,
    wang0701164_fig5_params,
    wang0701164_paper_unit_factors,
    wang0701164_fig5_spec,
)


def test_wang0701164_fig5_spec_matches_published_values() -> None:
    spec = wang0701164_fig5_spec(unit_system="paper")
    assert spec["target_figure"] == "Fig. 5"
    assert spec["parameters"]["spring"] == 0.625
    assert spec["parameters"]["onsite_spring"] == 0.0625
    assert spec["parameters"]["n_layers"] == 5
    assert spec["parameters"]["cubic_values"] == [0.0, 0.2, 0.5, 0.7, 1.0, 2.0]


def test_cubic_onsite_current_is_finite_for_fig5_model() -> None:
    params = wang0701164_fig5_params(cubic=0.2, unit_system="paper")
    omegas = np.linspace(1e-3, 5.0, 33)
    out = cubic_onsite_current(
        omegas,
        params=params,
        temp_left=300.5,
        temp_right=299.5,
        eta=1e-4,
        kb_effective=0.08617333262,
    )
    assert np.isfinite(out["current_internal"])
    assert np.all(np.isfinite(out["effective_transmission"]))


def test_wang0701164_fig5_sweep_returns_all_published_couplings() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([300.0], dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
    )
    assert set(result["curves"]) == {"0.0", "0.2", "0.5", "0.7", "1.0", "2.0"}
    for curve in result["curves"].values():
        assert np.all(np.isfinite(curve["conductance"]))


def test_wang0701164_fig5_paper_derivative_mode_runs() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([300.0], dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
        conductance_mode="paper_derivative",
    )
    for curve in result["curves"].values():
        assert np.all(np.isfinite(curve["conductance"]))


def test_wang0701164_fig5_paper_derivative_numeric_mode_runs() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([300.0], dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
        conductance_mode="paper_derivative_numeric",
    )
    for curve in result["curves"].values():
        assert np.all(np.isfinite(curve["conductance"]))


def test_wang0701164_fig5_internal_oversample_mode_runs() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([300.0], dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
        conductance_mode="current_over_delta_t",
        internal_oversample=2,
    )
    for curve in result["curves"].values():
        assert np.all(np.isfinite(curve["conductance"]))


def test_wang0701164_effective_transmission_default_uses_tadpole_subtracted_path() -> None:
    fac = wang0701164_paper_unit_factors()
    params = wang0701164_fig5_params(cubic=0.5, unit_system="paper")
    omegas_nonneg = np.linspace(0.0, 2.1, 41)
    omegas = np.concatenate([-omegas_nonneg[:0:-1], omegas_nonneg])
    out = cubic_onsite_effective_transmission(
        omegas,
        params=params,
        temperature=500.0,
        delta_t=0.1,
        eta=1e-4,
        kb_effective=fac["kb_effective_paper"],
    )
    assert out["include_second_graph"] is False
    assert np.all(np.isfinite(out["effective_transmission"]))


def test_wang0701164_fig5_mid_temperature_family_is_in_published_ballpark() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([500.0], dtype=float),
        unit_system="paper",
        n_omega=81,
        delta_t=0.1,
        eta=1e-4,
    )
    curves = {k: 1.0e9 * float(v["conductance"][0]) for k, v in result["curves"].items()}
    assert 0.20 < curves["0.0"] < 0.30
    assert 0.18 < curves["0.2"] < curves["0.0"]
    assert 0.10 < curves["0.5"] < 0.20
    assert 0.05 < curves["0.7"] < 0.15
    assert 0.03 < curves["1.0"] < 0.10
    assert 0.0 < curves["2.0"] < 0.03


def test_wang0701164_second_graph_strongly_suppresses_numeric_derivative_benchmark() -> None:
    off = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([500.0], dtype=float),
        unit_system="paper",
        n_omega=61,
        delta_t=0.02,
        eta=5e-5,
        conductance_mode="paper_derivative_numeric",
        include_second_graph=False,
    )
    on = wang0701164_fig5_lowest_order_conductance_sweep(
        np.array([500.0], dtype=float),
        unit_system="paper",
        n_omega=61,
        delta_t=0.02,
        eta=5e-5,
        conductance_mode="paper_derivative_numeric",
        include_second_graph=True,
    )
    kappa_off = float(off["curves"]["0.5"]["conductance"][0])
    kappa_on = float(on["curves"]["0.5"]["conductance"][0])
    assert kappa_off > 0.0
    assert kappa_on < 0.1 * kappa_off


def test_wang0701164_digitized_comparison_helper_returns_finite_score() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.linspace(50.0, 2000.0, 11, dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
        conductance_mode="paper_derivative_numeric",
    )
    metrics = wang0701164_compare_sweep_to_digitized(
        result,
        "outputs/inelastic/Wang0701164/002_fig5_digitized_curves.tsv",
    )
    assert np.isfinite(float(metrics["score"]))


def test_wang0701164_vector_reference_extraction_returns_expected_ordering() -> None:
    reference = wang0701164_extract_fig5_vector_curves("literature/0701164v1.pdf", page=10)
    peaks = {label: 1.0e9 * float(np.max(values)) for label, values in reference["curves"].items()}
    assert peaks["0.0"] > peaks["0.2"] > peaks["0.5"] > peaks["0.7"] > peaks["1.0"] > peaks["2.0"]
    assert 0.27 < peaks["0.0"] < 0.30
    assert 0.003 < peaks["2.0"] < 0.01


def test_wang0701164_vector_comparison_helper_returns_finite_score() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.linspace(50.0, 2000.0, 11, dtype=float),
        unit_system="paper",
        n_omega=33,
        delta_t=0.1,
        eta=1e-4,
        conductance_mode="paper_derivative_numeric",
    )
    metrics = wang0701164_compare_sweep_to_vector_pdf(
        result,
        "literature/0701164v1.pdf",
        page=10,
    )
    assert np.isfinite(float(metrics["score"]))


def test_wang0701164_current_over_delta_t_post_padding_benchmark_stays_reasonable() -> None:
    result = wang0701164_fig5_lowest_order_conductance_sweep(
        np.linspace(20.0, 2000.0, 11, dtype=float),
        unit_system="paper",
        n_omega=61,
        omega_max=2.7,
        delta_t=0.02,
        eta=2e-5,
        conductance_mode="current_over_delta_t",
        include_second_graph=False,
    )
    metrics = wang0701164_compare_sweep_to_vector_pdf(
        result,
        "literature/0701164v1.pdf",
        page=10,
    )
    assert float(metrics["score"]) < 1.0
    assert 0.003 < float(metrics["curves"]["2.0"]["peak_model_1e-9_W_per_K"]) < 0.007


def test_wang0701164_eq84_numeric_decomposition_is_finite() -> None:
    fac = wang0701164_paper_unit_factors()
    params = wang0701164_fig5_params(cubic=0.5, unit_system="paper")
    omegas_nonneg = np.linspace(0.0, 2.7, 61)
    omegas = np.concatenate([-omegas_nonneg[:0:-1], omegas_nonneg])
    out = cubic_onsite_eq84_numeric_decomposition(
        omegas,
        params=params,
        temperature=500.0,
        eta=1e-4,
        kb_effective=fac["kb_effective_paper"],
        delta_t=0.02,
    )
    assert np.all(np.isfinite(out["delta_spectral_sigma_eq"]))
    assert np.all(np.isfinite(out["spectral_delta_sigma"]))
    assert np.all(np.isfinite(out["lesser_gamma_eq"]))
    assert np.isfinite(float(out["kappa_total"]))


def test_wang0701164_eq84_raw_decomposition_matches_harmonic_current_limit() -> None:
    fac = wang0701164_paper_unit_factors()
    params = wang0701164_fig5_params(cubic=0.0, unit_system="paper")
    delta_t = 0.02
    omegas_nonneg = np.linspace(0.0, 2.7, 121)
    omegas = np.concatenate([-omegas_nonneg[:0:-1], omegas_nonneg])

    decomp = cubic_onsite_eq84_numeric_decomposition(
        omegas,
        params=params,
        temperature=500.0,
        eta=1e-4,
        kb_effective=fac["kb_effective_paper"],
        hbar_effective=fac["hbar_effective_paper"],
        delta_t=delta_t,
    )
    plus = cubic_onsite_current_exact(
        omegas,
        params=params,
        temp_left=500.0 + 0.5 * delta_t,
        temp_right=500.0 - 0.5 * delta_t,
        eta=1e-4,
        kb_effective=fac["kb_effective_paper"],
        hbar_effective=fac["hbar_effective_paper"],
    )
    minus = cubic_onsite_current_exact(
        omegas,
        params=params,
        temp_left=500.0 - 0.5 * delta_t,
        temp_right=500.0 + 0.5 * delta_t,
        eta=1e-4,
        kb_effective=fac["kb_effective_paper"],
        hbar_effective=fac["hbar_effective_paper"],
    )

    current_conductance = (float(plus["current"]) - float(minus["current"])) / (2.0 * delta_t)
    assert np.isclose(float(decomp["kappa_total"]), current_conductance, rtol=1e-3, atol=1e-15)
