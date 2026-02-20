"""Compute 1D/2D/3D Landauer phonon heat current quantities."""

import numpy as np

from negfpy.core import (
    heat_current_1d,
    heat_current_density_3d,
    heat_current_per_length_2d,
    transverse_area_from_vectors,
    transverse_length_from_vector,
)
from negfpy.models import (
    ChainParams,
    CubicLatticeParams,
    SquareLatticeParams,
    cubic_lattice_device,
    cubic_lattice_lead,
    device_perfect_chain,
    lead_blocks,
    square_lattice_device,
    square_lattice_lead,
)


def main() -> None:
    # Frequency grid in rad/s (example: 0.02..8 THz converted to angular frequency).
    f_hz = np.linspace(0.02e12, 8.0e12, 240)
    omegas = 2.0 * np.pi * f_hz

    t_left = 310.0
    t_right = 300.0

    # 1D: heat current (W).
    p1 = ChainParams(mass=4.66e-26, spring=50.0)
    lead1 = lead_blocks(p1)
    dev1 = device_perfect_chain(n_layers=100, params=p1)
    iq_1d = heat_current_1d(
        omegas=omegas,
        device=dev1,
        lead_left=lead1,
        lead_right=lead1,
        temp_left=t_left,
        temp_right=t_right,
        eta=1e-6,
    )

    # 2D: heat current per transverse length (W/m).
    p2 = SquareLatticeParams(mass=4.66e-26, spring_x=50.0, spring_y=35.0)
    lead2 = square_lattice_lead(p2)
    dev2 = square_lattice_device(n_layers=100, params=p2)
    nk2 = 41
    ky_vals = np.linspace(-np.pi, np.pi, nk2, endpoint=False)
    k2 = [(float(ky),) for ky in ky_vals]
    a_y = 3.2e-10  # m
    l_perp = transverse_length_from_vector(transverse_vec=(0.0, a_y, 0.0), transport_dir=(1.0, 0.0, 0.0))
    jq_2d = heat_current_per_length_2d(
        omegas=omegas,
        device=dev2,
        lead_left=lead2,
        lead_right=lead2,
        kpoints=k2,
        transverse_length=l_perp,
        temp_left=t_left,
        temp_right=t_right,
        eta=1e-6,
    )

    # 3D: heat current density per transverse area (W/m^2).
    p3 = CubicLatticeParams(mass=4.66e-26, spring_x=50.0, spring_y=35.0, spring_z=30.0)
    lead3 = cubic_lattice_lead(p3)
    dev3 = cubic_lattice_device(n_layers=100, params=p3)
    nk3 = 25
    kyz = np.linspace(-np.pi, np.pi, nk3, endpoint=False)
    k3 = [(float(ky), float(kz)) for ky in kyz for kz in kyz]
    t1 = np.array([0.0, 3.2e-10, 0.0])  # m
    t2 = np.array([0.0, 0.0, 3.2e-10])  # m
    a_perp = transverse_area_from_vectors(t1, t2, transport_dir=(1.0, 0.0, 0.0))
    jq_3d = heat_current_density_3d(
        omegas=omegas,
        device=dev3,
        lead_left=lead3,
        lead_right=lead3,
        kpoints=k3,
        transverse_area=a_perp,
        temp_left=t_left,
        temp_right=t_right,
        eta=1e-6,
    )

    print(f"1D heat current I_Q: {iq_1d:.6e} W")
    print(f"2D heat current density J_Q (per length): {jq_2d:.6e} W/m")
    print(f"3D heat current density J_Q (per area): {jq_3d:.6e} W/m^2")


if __name__ == "__main__":
    main()
