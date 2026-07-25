from __future__ import annotations

import numpy as np
import pytest

from negfpy.anharmonic.toy import (
    Wang2006Approximation,
    Wang2006MeanFieldConvergenceError,
    solve_wang_2006_born_t2,
    solve_wang_2006_selective_mean_field,
    wang_2006_approximation_specs,
    wang_2006_author_legacy_figure2_fixture,
    wang_2006_author_odd_grid,
    wang_2006_caption_literal_fixture,
    wang_2006_figure2_reference,
    wang_2006_harmonic_conductance,
    wang_2006_inferred_author_legacy_figure2_fixture,
)
from negfpy.anharmonic.toy.wang2006 import _zero_extended_bubble


def test_wang_caption_and_author_figure_fixtures_are_never_substituted() -> None:
    caption = wang_2006_caption_literal_fixture()
    inferred = wang_2006_inferred_author_legacy_figure2_fixture()
    compatibility_alias = wang_2006_author_legacy_figure2_fixture()

    assert caption.scale.metadata["model_variant"] == "caption_literal"
    assert inferred.scale.metadata["model_variant"] == (
        "inferred_author_legacy_fig2"
    )
    assert compatibility_alias.scale.metadata["model_variant"] == (
        "inferred_author_legacy_fig2"
    )
    provenance = inferred.scale.metadata["harmonic_input_provenance"]
    assert provenance["inferred_values"] is True
    assert provenance["inverse_model_unique"] is False
    assert "source-level" in provenance["main_springs"]
    assert "inferred" in provenance["onsite_terms"]
    np.testing.assert_allclose(caption.left_lead.d01, [[-1.56]])
    np.testing.assert_allclose(
        inferred.left_lead.d01,
        [[-1.5603772686151907]],
        rtol=0.0,
        atol=5.0e-15,
    )
    # The onsite mismatch is the scientifically material difference.  The
    # inferred model assigns 0.01 to the old 10^4 dyn/cm unit; this hypothesis
    # is favored by the EPS curve but is not explicitly archived by the
    # source.
    caption_left_onsite = (
        caption.left_lead.d00[0, 0]
        - 2.0 * abs(caption.left_lead.d01[0, 0])
    )
    inferred_left_onsite = (
        inferred.left_lead.d00[0, 0]
        - 2.0 * abs(inferred.left_lead.d01[0, 0])
    )
    assert caption_left_onsite == pytest.approx(0.01)
    assert inferred_left_onsite == pytest.approx(0.006241509074460763)


def test_wang_approximation_hierarchy_does_not_relabel_guo_scba() -> None:
    specs = wang_2006_approximation_specs()
    historical = specs[Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD]
    production = specs[Wang2006Approximation.GUO_BUBBLE_SCBA]

    source_limited_status = (
        "implemented_inferred_model_curve_reproduced_source_limited"
    )
    assert all(
        specs[approximation].status == source_limited_status
        for approximation in (
            Wang2006Approximation.HARMONIC,
            Wang2006Approximation.WANG_BORN_T2,
            Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD,
        )
    )
    assert historical.includes_tadpole
    assert historical.implemented
    assert historical.bubble_propagator == "dressed G for bubble; G0 for tadpole"
    assert historical.production_operation is None
    assert not production.includes_tadpole
    assert production.implemented
    assert production.production_operation == "solve_scba"
    assert production.figure2_series is None


def test_wang_figure2_vector_reference_has_primary_source_provenance() -> None:
    reference = wang_2006_figure2_reference()

    assert {
        approximation: series.temperature_kelvin.size
        for approximation, series in reference.items()
    } == {
        Wang2006Approximation.HARMONIC: 51,
        Wang2006Approximation.WANG_BORN_T2: 49,
        Wang2006Approximation.WANG_BORN_T4: 15,
        Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD: 56,
    }
    assert all(
        series.source_sha256
        == "0e26c34ca70f3b132af8c6b3446d792c7a1ffd8b12a486f04df0c9db7185432b"
        for series in reference.values()
    )
    assert reference[Wang2006Approximation.HARMONIC].interpolate(
        300.0
    ) == pytest.approx(0.24639073552673266e-9)
    assert reference[Wang2006Approximation.WANG_BORN_T2].interpolate(
        300.0
    ) == pytest.approx(0.2328113465196172e-9)


def test_wang_inferred_model_reproduces_published_ballistic_curve() -> None:
    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    reference = wang_2006_figure2_reference()[
        Wang2006Approximation.HARMONIC
    ]
    temperatures = reference.temperature_kelvin
    calculated = wang_2006_harmonic_conductance(
        temperatures,
        fixture=fixture,
        n_frequency=6001,
        eta_lead=1.0e-5,
    )
    expected = reference.conductance_w_per_k
    absolute_error_nw_per_k = np.abs(calculated - expected) * 1.0e9

    # A scale-aware full-curve gate avoids a misleading relative error at the
    # first nearly-zero vector point.  The inferred onsite-unit hypothesis
    # gives RMS/max errors of about 0.000124/0.000277 nW/K across all 51
    # points.  This is strong curve agreement, not proof of a unique inverse
    # recovery of the source inputs.
    assert np.sqrt(np.mean(absolute_error_nw_per_k**2)) < 1.5e-4
    assert np.max(absolute_error_nw_per_k) < 3.0e-4
    with pytest.raises(ValueError, match="inferred_author_legacy_fig2"):
        wang_2006_harmonic_conductance(
            300.0,
            fixture=wang_2006_caption_literal_fixture(),
        )


def test_wang_zero_padded_fft_matches_direct_convolution() -> None:
    grid = wang_2006_author_odd_grid()
    small_grid = type(grid)(step=grid.step, n_positive=8)
    rng = np.random.default_rng(194)
    vertex = rng.normal(size=(2, 2, 2))
    vertex = sum(
        np.transpose(vertex, axes)
        for axes in (
            (0, 1, 2),
            (0, 2, 1),
            (1, 0, 2),
            (1, 2, 0),
            (2, 0, 1),
            (2, 1, 0),
        )
    ) / 6.0
    green = rng.normal(size=(small_grid.n_points, 2, 2)) + 1j * rng.normal(
        size=(small_grid.n_points, 2, 2)
    )

    direct = _zero_extended_bubble(
        vertex,
        green,
        small_grid,
        convolution="direct",
    )
    transformed = _zero_extended_bubble(
        vertex,
        green,
        small_grid,
        convolution="fft",
    )
    np.testing.assert_allclose(transformed, direct, rtol=2.0e-14, atol=2.0e-14)


def test_wang_historical_solvers_validate_actual_temperature_bias() -> None:
    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    full_grid = wang_2006_author_odd_grid()
    grid = type(full_grid)(step=full_grid.step, n_positive=8)
    factor = fixture.scale.metadata["legacy_frequency_to_solver"]
    problem = fixture.problem(
        grid,
        left_temperature_kelvin=305.0,
        right_temperature_kelvin=295.0,
    )
    with pytest.raises(ValueError, match="does not match"):
        solve_wang_2006_born_t2(
            problem,
            temperature_bias_kelvin=9.0,
            eta_lead=1.0e-4 * factor,
        )

    reversed_problem = fixture.problem(
        grid,
        left_temperature_kelvin=295.0,
        right_temperature_kelvin=305.0,
    )
    with pytest.raises(ValueError, match="strictly hotter"):
        solve_wang_2006_born_t2(
            reversed_problem,
            temperature_bias_kelvin=10.0,
            eta_lead=1.0e-4 * factor,
        )


def test_wang_mean_field_nonconvergence_carries_partial_result() -> None:
    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    full_grid = wang_2006_author_odd_grid()
    grid = type(full_grid)(step=full_grid.step, n_positive=8)
    problem = fixture.problem(
        grid,
        left_temperature_kelvin=305.0,
        right_temperature_kelvin=295.0,
    )
    factor = fixture.scale.metadata["legacy_frequency_to_solver"]

    with pytest.raises(Wang2006MeanFieldConvergenceError) as captured:
        solve_wang_2006_selective_mean_field(
            problem,
            temperature_bias_kelvin=10.0,
            eta_lead=1.0e-4 * factor,
            max_iterations=1,
            min_iterations=1,
            consecutive_passes=2,
        )

    partial = captured.value.partial_result
    assert partial.converged is False
    assert len(partial.history) == 1
    assert partial.release_certified is False
    assert partial.metadata["converged"] is False


@pytest.mark.nightly
def test_wang_born_t2_reproduces_published_temperature_checkpoints() -> None:
    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    grid = wang_2006_author_odd_grid()
    frequency_factor = fixture.scale.metadata["legacy_frequency_to_solver"]
    reference = wang_2006_figure2_reference()[
        Wang2006Approximation.WANG_BORN_T2
    ]

    # The source notebook and final figure do not archive the same output
    # quadrature. Fine-grid current integration and the notebook's separate
    # 0.05 output grid bracket the vector value within 0.5%; 1% is therefore
    # a source-limited validation tolerance, not an adjustable prefactor.
    for temperature in (100.0, 300.0, 500.0):
        problem = fixture.problem(
            grid,
            left_temperature_kelvin=temperature + 5.0,
            right_temperature_kelvin=temperature - 5.0,
        )
        result = solve_wang_2006_born_t2(
            problem,
            temperature_bias_kelvin=10.0,
            eta_lead=1.0e-4 * frequency_factor,
            convolution="fft",
        )
        assert result.through_conductance_w_per_k == pytest.approx(
            reference.interpolate(temperature),
            rel=1.0e-2,
        )
        assert result.metadata["static_tadpole"] is True
        assert result.metadata["dynamic_retarded_reconstruction"] == (
            "Sigma++-Sigma+-"
        )
        assert result.metadata[
            "eta_device_is_unaccounted_legacy_regulator"
        ] is True
        assert result.release_certified is False


@pytest.mark.nightly
@pytest.mark.parametrize("temperature", [300.0, 400.0])
def test_wang_selective_mean_field_reproduces_stable_published_range(
    temperature: float,
) -> None:
    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    grid = wang_2006_author_odd_grid()
    problem = fixture.problem(
        grid,
        left_temperature_kelvin=temperature + 5.0,
        right_temperature_kelvin=temperature - 5.0,
    )
    frequency_factor = fixture.scale.metadata["legacy_frequency_to_solver"]
    result = solve_wang_2006_selective_mean_field(
        problem,
        temperature_bias_kelvin=10.0,
        eta_lead=1.0e-4 * frequency_factor,
        convolution="fft",
        mixing=0.5,
    )
    published = wang_2006_figure2_reference()[
        Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD
    ].interpolate(temperature)

    assert result.converged
    assert result.through_conductance_w_per_k == pytest.approx(
        published,
        rel=1.0e-2,
    )
    assert all(item.passing for item in result.history[-3:])
    assert result.metadata["bubble_propagator"] == "dressed G"
    assert result.metadata["tadpole_propagator"] == "fixed G0"
    assert result.metadata["historical_approximation"] == (
        "selective bubble dressing, not Guo SCBA"
    )
    assert result.static_tadpole.flags.writeable is False
    assert result.release_certified is False
