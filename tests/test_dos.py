import numpy as np

from negfpy.core import lead_surface_dos, lead_surface_dos_kavg_adaptive
from negfpy.models import ChainParams, CubicLatticeParams, analytic_band_max, cubic_lattice_lead, lead_blocks


def test_chain_surface_dos_is_inband_larger_than_out_of_band() -> None:
    params = ChainParams(mass=1.0, spring=1.0)
    lead = lead_blocks(params)
    wmax = analytic_band_max(params)

    dos_in = lead_surface_dos(omega=0.5 * wmax, lead=lead, eta=1e-6)
    dos_out = lead_surface_dos(omega=1.2 * wmax, lead=lead, eta=1e-6)

    assert np.isfinite(dos_in)
    assert np.isfinite(dos_out)
    assert dos_in > 0.0
    assert dos_out < dos_in


def test_cubic_kavg_surface_dos_adaptive_is_finite() -> None:
    lead = cubic_lattice_lead(CubicLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.5, spring_z=0.4))
    kvals = np.linspace(-np.pi, np.pi, 4, endpoint=False)
    kpoints = [(float(ky), float(kz)) for ky in kvals for kz in kvals]
    dos, info = lead_surface_dos_kavg_adaptive(
        omega=0.9,
        lead=lead,
        kpoints=kpoints,
        eta_values=(1e-8, 1e-7, 1e-6),
        min_success_fraction=0.7,
    )
    assert np.isfinite(dos)
    assert dos > -1e-8
    assert float(info["success_fraction"]) >= 0.7
