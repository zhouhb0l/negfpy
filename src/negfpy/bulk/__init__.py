"""Public bulk-phonon and three-phonon SMRTA interface.

This namespace is the stable 1.0 API for periodic material calculations.
The open-device harmonic and anharmonic NEGF interfaces remain separate.
"""

from negfpy.anharmonic.bulk_harmonic import (
    CODATA_FREQUENCY_FACTOR_TO_THZ,
    DEFAULT_FREQUENCY_FACTOR_TO_THZ,
    PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ,
    BulkGroupVelocities,
    BulkPhononSpectrum,
    compute_bulk_group_velocities,
    compute_bulk_phonons,
    dynamical_matrix_derivatives_from_fc2,
    dynamical_matrix_from_fc2,
)
from negfpy.anharmonic.bulk_observables import (
    PhononBandPath,
    PhononBandStructure,
    PhononDOS,
    compute_phonon_band_structure,
    compute_phonon_dos,
    interpolate_band_path,
)
from negfpy.anharmonic.bulk_rta import (
    PHONO3PY_4_4_RTA_CONVERSION,
    BulkMeanFreePaths,
    BulkModeHeatCapacities,
    BulkRelaxationTimes,
    BulkThreePhononRateSpectrum,
    CumulativeThermalConductivity,
    RTAThermalConductivity,
    assemble_rta_thermal_conductivity,
    compute_cumulative_thermal_conductivity,
    compute_mean_free_paths,
    compute_mode_heat_capacities,
    compute_rta_thermal_conductivity,
    compute_three_phonon_lifetimes,
    summarize_three_phonon_linewidths,
)
from negfpy.anharmonic.bulk_three_phonon import (
    BulkThreePhononInteractions,
    BulkThreePhononLinewidths,
    BulkThreePhononProblem,
    compute_three_phonon_interactions,
    compute_three_phonon_linewidths,
)
from negfpy.anharmonic.crystal import (
    CrystalCell,
    CrystalMapping,
    validate_crystal_mapping,
)
from negfpy.anharmonic.fc2 import FC2Data, FC2ValidationReport, validate_fc2
from negfpy.anharmonic.fc3 import FC3Data, validate_fc3
from negfpy.anharmonic.io import (
    read_phono3py_crystal_mapping,
    read_phono3py_fc2,
    read_phono3py_fc3,
)
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.translational_fc3 import (
    TranslationalFC3,
    convert_fc3_to_translational,
)
from negfpy.anharmonic.translational_fc3_validation import (
    TranslationalFC3ValidationReport,
    validate_translational_fc3,
)

from .archive import (
    RTA_RESULT_HDF5_SCHEMA,
    read_rta_result_hdf5,
    write_rta_result_hdf5,
)
from .fc2_audit import MappedFC2AuditReport, audit_mapped_fc2
from .problem import BulkHarmonicProblem, BulkTransportProblem
from .rate_archive import (
    RATE_SPECTRUM_HDF5_SCHEMA,
    read_rate_spectrum_hdf5,
    write_rate_spectrum_hdf5,
)

__all__ = [
    "CODATA_FREQUENCY_FACTOR_TO_THZ",
    "DEFAULT_FREQUENCY_FACTOR_TO_THZ",
    "PHONO3PY_4_4_FREQUENCY_FACTOR_TO_THZ",
    "PHONO3PY_4_4_RTA_CONVERSION",
    "RATE_SPECTRUM_HDF5_SCHEMA",
    "RTA_RESULT_HDF5_SCHEMA",
    "BulkGroupVelocities",
    "BulkHarmonicProblem",
    "BulkMeanFreePaths",
    "BulkModeHeatCapacities",
    "BulkPhononSpectrum",
    "BulkRelaxationTimes",
    "BulkThreePhononInteractions",
    "BulkThreePhononLinewidths",
    "BulkThreePhononProblem",
    "BulkThreePhononRateSpectrum",
    "BulkTransportProblem",
    "CrystalCell",
    "CrystalMapping",
    "CumulativeThermalConductivity",
    "FC2Data",
    "FC2ValidationReport",
    "FC3Data",
    "MappedFC2AuditReport",
    "PhononBandPath",
    "PhononBandStructure",
    "PhononDOS",
    "RTAThermalConductivity",
    "TranslationalFC3",
    "TranslationalFC3ValidationReport",
    "UniformBZMesh",
    "assemble_rta_thermal_conductivity",
    "audit_mapped_fc2",
    "compute_bulk_group_velocities",
    "compute_bulk_phonons",
    "compute_cumulative_thermal_conductivity",
    "compute_mean_free_paths",
    "compute_mode_heat_capacities",
    "compute_phonon_band_structure",
    "compute_phonon_dos",
    "compute_rta_thermal_conductivity",
    "compute_three_phonon_interactions",
    "compute_three_phonon_lifetimes",
    "compute_three_phonon_linewidths",
    "convert_fc3_to_translational",
    "dynamical_matrix_derivatives_from_fc2",
    "dynamical_matrix_from_fc2",
    "interpolate_band_path",
    "read_phono3py_crystal_mapping",
    "read_phono3py_fc2",
    "read_phono3py_fc3",
    "read_rate_spectrum_hdf5",
    "read_rta_result_hdf5",
    "summarize_three_phonon_linewidths",
    "validate_crystal_mapping",
    "validate_fc2",
    "validate_fc3",
    "validate_translational_fc3",
    "write_rate_spectrum_hdf5",
    "write_rta_result_hdf5",
]
