"""Validate the isolated inelastic toy model against the ballistic 1D chain.

Step 1 in the inelastic roadmap:
- use the existing harmonic monoatomic chain;
- verify that the new inelastic module reproduces the coherent spectrum when
  phonon-phonon scattering is disabled;
- inspect a simple phenomenological damping model separately.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from negfpy.core import transmission
from negfpy.inelastic import PowerLawPPSelfEnergy, transmission_inelastic
from negfpy.models import ChainParams, analytic_band_max, device_perfect_chain, lead_blocks


params = ChainParams(mass=1.0, spring=1.0)
lead = lead_blocks(params)
device = device_perfect_chain(n_layers=30, params=params)
wmax = analytic_band_max(params)
omegas = np.linspace(0.05 * wmax, 1.10 * wmax, 250)

zero_model = PowerLawPPSelfEnergy(gamma0=0.0, omega_ref=wmax, power=2.0)
damped_model = PowerLawPPSelfEnergy(gamma0=0.05, omega_ref=wmax, power=2.0)

ballistic = np.array([transmission(float(w), device, lead, lead, eta=1e-8) for w in omegas])
inelastic_zero = np.array(
    [
        transmission_inelastic(
            omega=float(w),
            device=device,
            lead_left=lead,
            lead_right=lead,
            eta=1e-8,
            pp_self_energy=zero_model,
        )[0]
        for w in omegas
    ]
)
inelastic_damped = np.array(
    [
        transmission_inelastic(
            omega=float(w),
            device=device,
            lead_left=lead,
            lead_right=lead,
            eta=1e-8,
            pp_self_energy=damped_model,
            mixing=1.0,
        )[0]
        for w in omegas
    ]
)

max_diff = float(np.max(np.abs(inelastic_zero - ballistic)))
print(f"Max |T_inelastic(gamma0=0) - T_ballistic| = {max_diff:.3e}")

fig, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=True)

axes[0].plot(omegas, ballistic, label="Ballistic", linewidth=2.0)
axes[0].plot(omegas, inelastic_zero, "--", label="Inelastic module, gamma0=0", linewidth=1.8)
axes[0].set_ylabel(r"$T(\omega)$")
axes[0].set_title("Toy-model validation: isolated inelastic module")
axes[0].grid(alpha=0.3)
axes[0].legend()

axes[1].plot(omegas, ballistic, label="Ballistic", linewidth=2.0)
axes[1].plot(omegas, inelastic_damped, label="Phenomenological damping", linewidth=1.8)
axes[1].set_xlabel(r"$\omega$")
axes[1].set_ylabel(r"$T(\omega)$")
axes[1].grid(alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.show()
