import numpy as np

from negfpy.core import transmission
from negfpy.models import ChainParams, analytic_band_max, device_perfect_chain, lead_blocks


def test_transmission_plateau_matched_chain() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=20, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(0.2 * wmax, 0.8 * wmax, 9)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

    assert np.all(np.isfinite(vals))
    assert np.mean(vals) > 0.96
    assert np.max(np.abs(vals - 1.0)) < 0.08


def test_transmission_nonnegative() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=12, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(0.05 * wmax, 1.2 * wmax, 30)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

    assert np.min(vals) > -1e-8


def test_out_of_band_goes_to_zero() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    device = device_perfect_chain(n_layers=16, params=params)

    wmax = analytic_band_max(params)
    omegas = np.linspace(1.1 * wmax, 1.7 * wmax, 10)
    vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

    assert np.max(np.abs(vals)) < 1e-4
