"""Plot k_parallel-averaged T(omega) for a 3D cubic lattice toy model."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission_kavg
from negfpy.models import CubicLatticeParams, cubic_lattice_device, cubic_lattice_lead


params = CubicLatticeParams(
    mass=1.0,
    spring_x=1.0,
    spring_y=0.6,
    spring_z=0.4,
    onsite_pinning=0,
)
lead = cubic_lattice_lead(params)
device = cubic_lattice_device(n_layers=120, params=params)

nk = 11
kvals = np.linspace(-np.pi, np.pi, nk)
kpoints = [(ky, kz) for ky in kvals for kz in kvals]

omegas = np.linspace(0.0001, 4.2, 260)
vals = np.array([transmission_kavg(w, device, lead, lead, kpoints=kpoints, eta=1e-8) for w in omegas])

plt.plot(omegas, vals)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$\langle T(\omega)\rangle_{k_\parallel}$")
plt.title("3D cubic lattice k-averaged transmission")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
