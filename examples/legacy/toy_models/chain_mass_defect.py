"""Plot T(omega) for a 1D chain with a mass defect in the device."""

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission
from negfpy.models import ChainParams, analytic_band_max, device_mass_defect, lead_blocks


params = ChainParams(mass=1.0, spring=1.0)
lead = lead_blocks(params)
device = device_mass_defect(n_layers=31, params=params, defect_index=15, defect_mass=3.0)

wmax = analytic_band_max(params)
omegas = np.linspace(0.0, 1.4 * wmax, 400)
vals = np.array([transmission(w, device, lead, lead, eta=1e-8) for w in omegas])

plt.plot(omegas, vals)
plt.xlabel(r"$\omega$")
plt.ylabel(r"$T(\omega)$")
plt.title("1D chain transmission with central mass defect")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
