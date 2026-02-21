import numpy as np

from negfpy.core import surface_gf, transmission
from negfpy.models import ChainParams, analytic_band_max, device_perfect_chain, lead_blocks


def test_surface_gf_generalized_eigen_matches_sancho_rubio_chain() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    w = 0.6 * analytic_band_max(params)

    g_sr = surface_gf(omega=w, d00=lead.d00, d01=lead.d01, d10=lead.d10, eta=1e-8, method="sancho_rubio")
    g_ge = surface_gf(omega=w, d00=lead.d00, d01=lead.d01, d10=lead.d10, eta=1e-8, method="generalized_eigen")

    assert np.all(np.isfinite(g_sr))
    assert np.all(np.isfinite(g_ge))
    assert np.linalg.norm(g_sr - g_ge) < 1e-8


def test_transmission_generalized_eigen_matches_sancho_rubio_chain() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=12, params=params)
    w = 0.5 * analytic_band_max(params)

    t_sr = transmission(
        omega=w,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        surface_gf_method="sancho_rubio",
    )
    t_ge = transmission(
        omega=w,
        device=device,
        lead_left=lead,
        lead_right=lead,
        eta=1e-8,
        surface_gf_method="generalized_eigen",
    )

    assert np.isfinite(t_sr)
    assert np.isfinite(t_ge)
    assert abs(t_sr - t_ge) < 1e-7
