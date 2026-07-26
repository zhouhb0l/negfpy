from __future__ import annotations

import numpy as np
import pytest

from negfpy.core import heat_current_from_spectrum, surface_gf, transmission
from negfpy.core.thermal_current import HBAR, KB
from negfpy.models import (
    ChainParams,
    analytic_band_max,
    device_perfect_chain,
    lead_blocks,
)


def _scalar_surface_green_exact(
    omega: float,
    *,
    onsite: complex,
    coupling_forward: complex,
    coupling_reverse: complex,
    eta: float,
) -> complex:
    """Independent retarded root of the scalar surface-GF quadratic."""

    z = (omega + 1.0j * eta) ** 2
    product = coupling_forward * coupling_reverse
    delta = z - onsite
    discriminant = np.sqrt(delta * delta - 4.0 * product)
    roots = (
        (delta + discriminant) / (2.0 * product),
        (delta - discriminant) / (2.0 * product),
    )

    # Positive-frequency retarded response has negative imaginary part. The
    # minimum-magnitude root supplies the decaying solution outside the band.
    causal = tuple(root for root in roots if root.imag < 0.0)
    if len(causal) == 1:
        return causal[0]
    return min(roots, key=abs)


@pytest.mark.parametrize(
    "method",
    ("sancho_rubio", "generalized_eigen", "generalized_eigen_svd"),
)
@pytest.mark.parametrize("omega", (0.2, 0.4, 1.0, 1.6, 2.4))
def test_scalar_surface_green_matches_exact_retarded_root(
    method: str,
    omega: float,
) -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    eta = 1.0e-10
    reverse = lead.d01.conj().T if lead.d10 is None else lead.d10
    expected = _scalar_surface_green_exact(
        omega,
        onsite=lead.d00[0, 0],
        coupling_forward=lead.d01[0, 0],
        coupling_reverse=reverse[0, 0],
        eta=eta,
    )
    actual = surface_gf(
        omega=omega,
        d00=lead.d00,
        d01=lead.d01,
        d10=lead.d10,
        eta=eta,
        method=method,
    )[0, 0]

    np.testing.assert_allclose(actual, expected, rtol=1.0e-10, atol=1.0e-12)


def test_matched_scalar_chain_has_exact_unit_caroli_transmission() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=20, params=params)
    maximum = analytic_band_max(params)

    values = np.array(
        [
            transmission(
                float(omega),
                device,
                lead,
                lead,
                eta=1.0e-10,
                eta_device=0.0,
            )
            for omega in np.linspace(0.1 * maximum, 0.9 * maximum, 17)
        ]
    )

    np.testing.assert_allclose(values, 1.0, rtol=1.0e-10, atol=1.0e-12)


def test_landauer_prefactor_and_current_sign_match_ballistic_quantum() -> None:
    temperature_left = 300.0
    temperature_right = 250.0
    omega_cutoff = 50.0 * KB * temperature_left / HBAR
    omegas = np.linspace(0.0, omega_cutoff, 100_001)
    unit_transmission = np.ones_like(omegas)

    actual = heat_current_from_spectrum(
        omegas,
        unit_transmission,
        temp_left=temperature_left,
        temp_right=temperature_right,
    )
    expected = (
        np.pi
        * KB**2
        * (temperature_left**2 - temperature_right**2)
        / (12.0 * HBAR)
    )
    reversed_current = heat_current_from_spectrum(
        omegas,
        unit_transmission,
        temp_left=temperature_right,
        temp_right=temperature_left,
    )

    assert actual > 0.0
    np.testing.assert_allclose(actual, expected, rtol=1.0e-8, atol=0.0)
    np.testing.assert_allclose(reversed_current, -actual, rtol=1.0e-13, atol=0.0)
