from __future__ import annotations

import numpy as np

from negfpy.anharmonic import (
    DenseCubicVertex,
    SignedFrequencyGrid,
    cubic_lesser_greater,
)
from negfpy.anharmonic.toy import (
    GUO_2020_PRODUCTION_SCBA,
    MINGO_2006_CAUSAL_NORMALIZED_SURFACE,
    MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY,
    MINGO_2006_LEGACY_EQ27,
    MINGO_2006_UNVERIFIED_VERTEX_X6_SENSITIVITY,
    Mingo2006DiatomicFixture,
    mingo_2006_bare_surface_resolvent,
    mingo_2006_dressed_surface_resolvent,
    mingo_2006_figure4_harmonic_curve,
    mingo_2006_figure8_curve,
    mingo_2006_harmonic_conductance,
    mingo_2006_harmonic_transmission,
    mingo_2006_legacy_lesser_greater,
)


def test_mingo_fixture_maps_printed_km_and_beta_to_production_vertex() -> None:
    fixture = Mingo2006DiatomicFixture()

    np.testing.assert_allclose(
        fixture.central_bond_solver,
        2.0
        * fixture.paper_central_ev_per_angstrom2
        * fixture.force_constant_to_solver,
    )
    lj = fixture.lj_vertex()
    sw = fixture.sw_vertex()
    np.testing.assert_allclose(
        lj.values / sw.values,
        fixture.lj_beta_per_angstrom / fixture.sw_beta_per_angstrom,
    )

    displacement = np.array([0.12, -0.07])
    h3 = np.einsum(
        "abc,a,b,c->",
        lj.values,
        displacement,
        displacement,
        displacement,
    ) / 6.0
    expected = lj.values[0, 0, 0] * (
        displacement[0] - displacement[1]
    ) ** 3 / 6.0
    np.testing.assert_allclose(h3, expected, rtol=1e-14)


def test_mingo_surface_resolvent_is_eta_free_causal_and_normalized() -> None:
    cutoff = 100.0
    omega = 23.0
    positive = mingo_2006_bare_surface_resolvent(omega, cutoff=cutoff)
    negative = mingo_2006_bare_surface_resolvent(-omega, cutoff=cutoff)
    density = 1.5 * omega / cutoff**3

    np.testing.assert_allclose(negative, np.conjugate(positive))
    np.testing.assert_allclose(positive.imag, -np.pi * density)

    high = mingo_2006_bare_surface_resolvent(1.0e6, cutoff=cutoff)
    np.testing.assert_allclose(
        (1.0e6) ** 2 * high.real,
        1.0,
        rtol=2e-8,
    )

    fixture = Mingo2006DiatomicFixture()
    dressed_cutoff = mingo_2006_dressed_surface_resolvent(
        cutoff,
        coupling=fixture.molecule_surface_solver,
        cutoff=cutoff,
    )
    np.testing.assert_allclose(
        dressed_cutoff,
        -1.0 / fixture.molecule_surface_solver,
    )


def test_mingo_harmonic_transmission_is_passive_without_eta() -> None:
    omega = np.arange(0.0, 100.5, 0.5)
    transmission = mingo_2006_harmonic_transmission(
        omega,
        convention="figure4_inferred_half_weight_sensitivity",
    )

    assert np.all(np.isfinite(transmission))
    assert np.min(transmission) >= -1e-14
    assert np.max(transmission) <= 1.0 + 1e-10
    assert transmission[0] == 0.0
    assert transmission[-1] == 0.0


def test_mingo_figure4_half_weight_sensitivity_matches_vector_curve() -> None:
    reference = mingo_2006_figure4_harmonic_curve()
    calculated_pw_per_kelvin = (
        np.asarray(
            mingo_2006_harmonic_conductance(
                reference.temperature_kelvin,
                frequency_step=0.5,
                convention="figure4_inferred_half_weight_sensitivity",
            )
        )
        * 1.0e12
    )
    difference = calculated_pw_per_kelvin - reference.values
    normalized_pw_per_kelvin = (
        np.asarray(
            mingo_2006_harmonic_conductance(
                reference.temperature_kelvin,
                frequency_step=0.5,
                convention="causal_normalized",
            )
        )
        * 1.0e12
    )

    # This only locks a named sensitivity observation.  Eq. (47) normalizes
    # the surface measure to one and does not justify this half weight, so the
    # comparison is not a same-input paper-reproduction gate.
    assert np.sqrt(np.mean(difference**2)) < 0.14
    assert np.max(np.abs(difference)) < 0.24
    assert (
        np.sqrt(
            np.mean((normalized_pw_per_kelvin - reference.values) ** 2)
        )
        > 3.0
    )
    assert MINGO_2006_CAUSAL_NORMALIZED_SURFACE.normalized_spectral_measure
    assert MINGO_2006_CAUSAL_NORMALIZED_SURFACE.derivable_from_printed_equations
    assert not MINGO_2006_CAUSAL_NORMALIZED_SURFACE.release_certifiable
    assert not MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY.release_certifiable
    assert (
        not MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY.derivable_from_printed_equations
    )


def test_mingo_legacy_bubble_is_exactly_pi_over_nine_of_production() -> None:
    grid = SignedFrequencyGrid(step=0.25, n_positive=4)
    vertex = DenseCubicVertex(np.array([[[0.37]]]))
    values = np.arange(1, grid.n_points + 1, dtype=np.float64)
    lesser = (-0.1j * values)[:, None, None]
    greater = (-0.2j * values[::-1])[:, None, None]

    production = cubic_lesser_greater(vertex, lesser, greater, grid)
    legacy = mingo_2006_legacy_lesser_greater(
        vertex,
        lesser,
        greater,
        grid,
    )
    for legacy_component, production_component in zip(legacy, production):
        np.testing.assert_allclose(
            legacy_component,
            (np.pi / 9.0) * production_component,
            rtol=0.0,
            atol=0.0,
        )

    assert GUO_2020_PRODUCTION_SCBA.release_certifiable
    assert not MINGO_2006_LEGACY_EQ27.release_certifiable
    assert (
        MINGO_2006_LEGACY_EQ27.implementation_status
        == "kernel_only_no_mingo_scba_driver"
    )
    assert (
        MINGO_2006_LEGACY_EQ27.cubic_bubble_scale_relative_to_guo
        == np.pi / 9.0
    )
    assert (
        MINGO_2006_UNVERIFIED_VERTEX_X6_SENSITIVITY
        .cubic_bubble_scale_relative_to_guo
        == 4.0 * np.pi
    )
    assert (
        MINGO_2006_UNVERIFIED_VERTEX_X6_SENSITIVITY.implementation_status
        == "not_implemented_hypothesis_only"
    )


def test_mingo_figure8_published_anharmonic_checkpoint_is_explicit() -> None:
    sw_scaled = mingo_2006_figure8_curve("sw_scaled_quantum")
    lj = mingo_2006_figure8_curve("lj_quantum")

    np.testing.assert_array_equal(
        sw_scaled.temperature_kelvin,
        lj.temperature_kelvin,
    )
    assert sw_scaled.values[0] > 0.08
    assert lj.values[-1] > 0.32
    assert lj.values[-1] > sw_scaled.values[-1]
    assert np.all(np.diff(lj.values[4:]) > 0.0)
