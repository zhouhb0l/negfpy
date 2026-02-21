import numpy as np
import pytest

from negfpy.core import lead_dynamical_matrix, lead_phonon_dispersion_3d, leads_phonon_dispersion_3d
from negfpy.core.types import LeadBlocks
from negfpy.models import CubicLatticeParams, cubic_lattice_lead


def test_cubic_scalar_dispersion_matches_analytic_formula() -> None:
    params = CubicLatticeParams(mass=2.0, spring_x=3.0, spring_y=1.5, spring_z=0.7, onsite_pinning=0.2)
    lead = cubic_lattice_lead(params)

    kx = np.array([-np.pi, -0.4, 0.0, 0.9, np.pi / 2.0])
    ky = np.array([-1.0, 0.0, 0.8])
    kz = np.array([-0.6, 0.0, 1.2])

    omega = lead_phonon_dispersion_3d(lead=lead, kx_points=kx, ky_points=ky, kz_points=kz)
    assert omega.shape == (kx.size, ky.size, kz.size, 1)

    expected = np.zeros((kx.size, ky.size, kz.size), dtype=float)
    for ix, kx_val in enumerate(kx):
        for iy, ky_val in enumerate(ky):
            for iz, kz_val in enumerate(kz):
                omega2 = (
                    2.0 * params.spring_x
                    + 2.0 * params.spring_y
                    + 2.0 * params.spring_z
                    + params.onsite_pinning
                    - 2.0 * params.spring_x * np.cos(kx_val)
                    - 2.0 * params.spring_y * np.cos(ky_val)
                    - 2.0 * params.spring_z * np.cos(kz_val)
                ) / params.mass
                expected[ix, iy, iz] = np.sqrt(max(omega2, 0.0))

    assert np.allclose(omega[..., 0], expected, atol=1e-12, rtol=1e-12)


def test_leads_dispersion_returns_left_and_right() -> None:
    left = cubic_lattice_lead(CubicLatticeParams(mass=1.0, spring_x=1.2, spring_y=0.5, spring_z=0.4))
    right = cubic_lattice_lead(CubicLatticeParams(mass=1.1, spring_x=1.0, spring_y=0.6, spring_z=0.3))

    out = leads_phonon_dispersion_3d(
        lead_left=left,
        lead_right=right,
        kx_points=[0.0, 0.5],
        ky_points=[0.1],
        kz_points=[-0.2, 0.4],
    )
    assert set(out.keys()) == {"left", "right"}
    assert out["left"].shape == (2, 1, 2, 1)
    assert out["right"].shape == (2, 1, 2, 1)
    assert not np.allclose(out["left"], out["right"])


def test_unstable_modes_raise_by_default() -> None:
    lead = LeadBlocks(
        d00=np.array([[0.0]], dtype=np.complex128),
        d01=np.array([[1.0]], dtype=np.complex128),
    )
    with pytest.raises(ValueError):
        lead_phonon_dispersion_3d(lead=lead, kx_points=[np.pi], ky_points=[0.0], kz_points=[0.0])


def test_lead_dynamical_matrix_matches_scalar_formula() -> None:
    lead = cubic_lattice_lead(CubicLatticeParams(mass=1.0, spring_x=2.0, spring_y=1.0, spring_z=0.5))
    kx = 0.7
    ky = -0.3
    kz = 1.2
    dmat = lead_dynamical_matrix(lead=lead, kx=kx, kpar=(ky, kz))
    expected = (
        2.0 * 2.0
        + 2.0 * 1.0
        + 2.0 * 0.5
        - 2.0 * 1.0 * np.cos(ky)
        - 2.0 * 0.5 * np.cos(kz)
        - 2.0 * 2.0 * np.cos(kx)
    )
    assert dmat.shape == (1, 1)
    assert np.allclose(dmat[0, 0].real, expected, atol=1e-12, rtol=1e-12)
    assert abs(dmat[0, 0].imag) < 1e-12
