import numpy as np

from negfpy.core import transmission
from negfpy.models import MultiDofCellParams, multidof_device, multidof_lead_blocks


def test_multidof_matched_device_has_channel_count_plateau() -> None:
    n_atoms = 2
    dof_per_atom = 3
    ndof = n_atoms * dof_per_atom

    masses = np.ones(n_atoms)
    k = 1.0
    fc00 = 2.0 * k * np.eye(ndof, dtype=float)
    fc01 = -k * np.eye(ndof, dtype=float)

    params = MultiDofCellParams(masses=masses, fc00=fc00, fc01=fc01, dof_per_atom=dof_per_atom)
    lead = multidof_lead_blocks(params)
    device = multidof_device(n_layers=30, params=params)

    omega = 1.0
    tval = transmission(omega=omega, device=device, lead_left=lead, lead_right=lead, eta=1e-8)

    assert np.isfinite(tval)
    assert tval > 0.95 * ndof
    assert tval < 1.05 * ndof


def test_multidof_device_accepts_per_layer_masses() -> None:
    masses = np.array([1.0, 1.2])
    dof_per_atom = 3
    ndof = masses.size * dof_per_atom
    k = 0.8
    fc00 = 2.0 * k * np.eye(ndof, dtype=float)
    fc01 = -k * np.eye(ndof, dtype=float)

    params = MultiDofCellParams(masses=masses, fc00=fc00, fc01=fc01, dof_per_atom=dof_per_atom)
    lead = multidof_lead_blocks(params)

    layer_masses = np.array(
        [
            [1.0, 1.2],
            [1.0, 1.2],
            [1.1, 1.2],
            [1.0, 1.3],
            [1.0, 1.2],
        ]
    )
    device = multidof_device(n_layers=5, params=params, layer_masses=layer_masses)

    tval = transmission(omega=0.7, device=device, lead_left=lead, lead_right=lead, eta=1e-8)
    assert np.isfinite(tval)
    assert tval > -1e-8
