import numpy as np

from negfpy.core import transmission, transmission_kavg
from negfpy.models import (
    CubicLatticeParams,
    SquareLatticeParams,
    cubic_lattice_device,
    cubic_lattice_lead,
    square_lattice_device,
    square_lattice_lead,
)


def test_square_lattice_kspace_transmission_nonnegative() -> None:
    params = SquareLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.7)
    lead = square_lattice_lead(params)
    device = square_lattice_device(n_layers=40, params=params)

    omega = 0.8
    kpoints = [(-np.pi,), (-np.pi / 2.0,), (0.0,), (np.pi / 2.0,), (np.pi,)]
    vals = np.array(
        [transmission(omega, device=device, lead_left=lead, lead_right=lead, kpar=k) for k in kpoints]
    )

    assert np.all(np.isfinite(vals))
    assert np.min(vals) > -1e-8


def test_square_lattice_kavg_matches_manual_mean() -> None:
    params = SquareLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.5)
    lead = square_lattice_lead(params)
    device = square_lattice_device(n_layers=25, params=params)

    omega = 0.9
    kpoints = [(-np.pi,), (-np.pi / 3.0,), (0.0,), (np.pi / 3.0,), (np.pi,)]
    manual = np.mean(
        [transmission(omega, device=device, lead_left=lead, lead_right=lead, kpar=k) for k in kpoints]
    )
    auto = transmission_kavg(
        omega=omega,
        device=device,
        lead_left=lead,
        lead_right=lead,
        kpoints=kpoints,
    )
    assert abs(manual - auto) < 1e-12


def test_cubic_lattice_kspace_transmission_nonnegative() -> None:
    params = CubicLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.4, spring_z=0.3)
    lead = cubic_lattice_lead(params)
    device = cubic_lattice_device(n_layers=30, params=params)

    omega = 1.0
    kpoints = [(-np.pi, -np.pi), (-np.pi / 2.0, 0.0), (0.0, 0.0), (np.pi / 2.0, np.pi / 3.0)]
    vals = np.array(
        [transmission(omega, device=device, lead_left=lead, lead_right=lead, kpar=k) for k in kpoints]
    )

    assert np.all(np.isfinite(vals))
    assert np.min(vals) > -1e-8
