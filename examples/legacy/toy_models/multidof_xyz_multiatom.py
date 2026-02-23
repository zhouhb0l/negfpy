"""Example: multi-atom unit cell with 3 DOF/atom in 1D transport."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission
from negfpy.models import MultiDofCellParams, multidof_device, multidof_lead_blocks


n_atoms = 2
dof_per_atom = 3
ndof = n_atoms * dof_per_atom

# Simple uncoupled xyz channels for each atom (6 total channels).
masses = np.ones(n_atoms)
k = 1.0
fc00 = 2.0 * k * np.eye(ndof)
fc01 = -k * np.eye(ndof)

params = MultiDofCellParams(
    masses=masses,
    fc00=fc00,
    fc01=fc01,
    dof_per_atom=dof_per_atom,
)
lead = multidof_lead_blocks(params)
device = multidof_device(n_layers=40, params=params)

omegas = np.linspace(0.0, 2.6, 300)
vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

plt.plot(omegas, vals)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$T(\omega)$")
plt.title("Multi-atom, 3 DOF/atom 1D transmission")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
