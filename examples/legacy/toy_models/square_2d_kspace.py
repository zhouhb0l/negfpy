"""Plot T(omega, ky) slices for a 2D square lattice toy model."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission
from negfpy.models import SquareLatticeParams, square_lattice_device, square_lattice_lead


params = SquareLatticeParams(mass=1.0, spring_x=1.0, spring_y=0.8)
lead = square_lattice_lead(params)
device = square_lattice_device(n_layers=120, params=params)

omegas = np.linspace(0.0, 3.0, 300)
kys = [0.0, np.pi / 3.0, 2.0 * np.pi / 3.0]

for ky in kys:
    vals = np.array([transmission(w, device, lead, lead, kpar=(ky,), eta=1e-8) for w in omegas])
    plt.plot(omegas, vals, label=rf"$k_y={ky:.2f}$")

plt.xlabel(r"$\omega$")
plt.ylabel(r"$T(\omega; k_y)$")
plt.title("2D square lattice transmission slices")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
