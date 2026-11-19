"""Existing SI scale and Kelvin conversion, selected without campaign definitions.

Copied verbatim from _material_nq1_campaign.py at source revision
5cf7ea7619c11120064162650dac7ef1ab7bbb72.
"""

import math

HBAR_J_S = 1.0545718176461565e-34
KB_J_K = 1.380649e-23
OMEGA0_RAD_S = 2.0 * math.pi * 1.0e13


def kelvin_to_solver_temperature(value: float) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("temperature must be finite")
    return result * KB_J_K / (HBAR_J_S * OMEGA0_RAD_S)
