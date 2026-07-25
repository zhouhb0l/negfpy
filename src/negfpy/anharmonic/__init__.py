"""Cubic phonon--phonon NEGF in a finite harmonic-contact device.

This subpackage is intentionally imported as ``negfpy.anharmonic`` rather
than re-exported from top-level :mod:`negfpy`.  The dense solver is the
v0.1 correctness reference; later sparse and transverse-momentum backends
must reproduce it on overlapping problems.
"""

from .contacts import (
    bose_occupation_signed,
    broadening,
    build_contact_spectra,
    minimum_lead_frequency_squared,
    reservoir_keldysh_from_retarded,
)
from .cubic import cubic_lesser_greater
from .current import (
    bond_heat_current_spectra,
    bond_heat_currents,
    collision_power,
    contact_heat_current,
    energy_exchange_spectrum,
)
from .diagnostics import (
    antihermitian_residual,
    check_solution,
    equilibrium_fdt_residual,
    high_frequency_extrapolation_residual,
    high_frequency_residual,
    keldysh_identity_residual,
    keldysh_tail_ratio,
    negative_frequency_residual,
    passivity_violation,
    relative_residual,
    retarded_reality_residual,
    spectral_function,
    spectral_moment_residual,
    tail_ratio,
)
from .exceptions import SCBAConvergenceError
from .fc3 import (
    FC3Data,
    FC3ValidationReport,
    MassNormalizedFC3,
    convert_fc3_values_to_si,
    mass_normalize_fc3,
    validate_fc3,
)
from .grid import SignedFrequencyGrid
from .io import read_phono3py_fc3
from .retarded import (
    HilbertBackend,
    principal_value_cauchy,
    principal_value_cauchy_fft,
    retarded_from_keldysh,
)
from .solver import solve_one_shot_born, solve_scba
from .types import (
    AnharmonicResult,
    CubicSCBAProblem,
    CurrentReport,
    DenseCubicVertex,
    DiagnosticReport,
    HarmonicReservoir,
    KeldyshSpectrum,
    SCBAConfig,
    SCBAIteration,
    SolverScale,
    estimate_dense_memory_bytes,
)

__all__ = [
    "AnharmonicResult",
    "CubicSCBAProblem",
    "CurrentReport",
    "DenseCubicVertex",
    "DiagnosticReport",
    "FC3Data",
    "FC3ValidationReport",
    "HarmonicReservoir",
    "HilbertBackend",
    "KeldyshSpectrum",
    "MassNormalizedFC3",
    "SCBAConfig",
    "SCBAConvergenceError",
    "SCBAIteration",
    "SignedFrequencyGrid",
    "SolverScale",
    "bond_heat_current_spectra",
    "bond_heat_currents",
    "antihermitian_residual",
    "bose_occupation_signed",
    "broadening",
    "build_contact_spectra",
    "check_solution",
    "collision_power",
    "contact_heat_current",
    "cubic_lesser_greater",
    "convert_fc3_values_to_si",
    "energy_exchange_spectrum",
    "equilibrium_fdt_residual",
    "estimate_dense_memory_bytes",
    "high_frequency_extrapolation_residual",
    "keldysh_identity_residual",
    "keldysh_tail_ratio",
    "high_frequency_residual",
    "mass_normalize_fc3",
    "minimum_lead_frequency_squared",
    "negative_frequency_residual",
    "passivity_violation",
    "principal_value_cauchy",
    "principal_value_cauchy_fft",
    "relative_residual",
    "read_phono3py_fc3",
    "retarded_reality_residual",
    "reservoir_keldysh_from_retarded",
    "retarded_from_keldysh",
    "solve_one_shot_born",
    "solve_scba",
    "spectral_function",
    "spectral_moment_residual",
    "tail_ratio",
    "validate_fc3",
]
