"""Material-oriented multi-DOF k-space builder example (x-transport, periodic y/z)."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission_kavg
from negfpy.models import MaterialKspaceParams, material_kspace_device, material_kspace_lead


# Example cell: 2 atoms, 3 DOF each (6x6 force-constant blocks).
n_atoms = 2
dof_per_atom = 3
ndof = n_atoms * dof_per_atom
masses = np.array([1.0, 1.2])

# Replace these toy matrices with force constants from your IFC pipeline.
phi00 = 3.0 * np.eye(ndof)
phi00_y = -0.4 * np.eye(ndof)
phi00_z = -0.3 * np.eye(ndof)
phi01 = -0.9 * np.eye(ndof)

params = MaterialKspaceParams(
    masses=masses,
    dof_per_atom=dof_per_atom,
    fc00_terms={
        (0, 0): phi00,
        (1, 0): phi00_y,
        (-1, 0): phi00_y,
        (0, 1): phi00_z,
        (0, -1): phi00_z,
    },
    fc01_terms={(0, 0): phi01},
    onsite_pinning=0,
)

lead = material_kspace_lead(params)
device = material_kspace_device(n_layers=80, params=params)

nk = 15
kvals = np.linspace(-np.pi, np.pi, nk, endpoint=False)
kpoints = [(ky, kz) for ky in kvals for kz in kvals]

omegas = np.linspace(0.0001, 2.2, 220)
vals = np.array([transmission_kavg(w, device, lead, lead, kpoints=kpoints, eta=1e-8) for w in omegas])

plt.plot(omegas, vals)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\langle T(\omega)\rangle_{k_\parallel}$")
plt.title("Material k-space builder: multi-atom, multi-DOF example")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
