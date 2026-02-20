"""Plot T(omega) for a matched 1D monoatomic chain."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission
from negfpy.models import ChainParams, analytic_band_max, device_perfect_chain, lead_blocks


params = ChainParams(mass=1.0, spring=1.0)
lead = lead_blocks(params)
device = device_perfect_chain(n_layers=30, params=params)

wmax = analytic_band_max(params)
omegas = np.linspace(0.0, 1.4 * wmax, 400)
vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

plt.plot(omegas, vals)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$T(\omega)$")
plt.title("1D matched chain transmission")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
