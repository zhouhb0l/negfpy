"""Public Python API for fitted complete-center-energy native-channel SCBA.

These are the existing implementations, exposed through one packaged import
surface.  Fitting creates a declared research surrogate with a measured IFC
residual; it does not recover a unique physical interface from bulk IFCs.
Complete center energies retain all required endpoint derivatives.

Native retarded/statistical/spectral fields remain independent.  An evaluated
input and its unevaluated Born image have different roles, including in saved
checkpoints.  The original three unmixed Sigma and three current screens gate
SCBA convergence; proposal selection and feasibility do not replace them.
"""

from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .site_energy_boundary import build_site_energy_vertex
from .site_energy_model import SiteEnergyModel, SiteEnergyStar, fit_site_energy_model
from .site_energy_saved import restore_site_energy_model
from .transverse_channel_accel import (
    ChannelAccelerationConfig,
    ChannelAccelerationPlan,
    ChannelAccelerationResult,
    ChannelSelectionPolicy,
    plan_channel_accel,
    run_channel_accel,
)
from .transverse_channel_feasibility import (
    ChannelProposalFailure,
    ChannelProposalFeasibility,
)
from .transverse_channel_feedback import (
    ChannelBornMapEvaluation,
    ChannelSCBAConfig,
    ChannelSCBAPlan,
    ChannelSCBARecord,
    ChannelSCBAResult,
    ChannelSCBAState,
    ChannelSigmaResidual,
    ChannelSpectrum,
    evaluate_channel_born_map,
    mix_channel_spectra,
    plan_channel_born_map,
    plan_channel_scba,
    run_channel_scba,
)
from .transverse_channel_restart import (
    ChannelCheckpoint,
    load_channel_checkpoint,
    save_channel_checkpoint,
)
from .transverse_compact_contacts import (
    CompactTransverseContact,
    build_compact_transverse_contact,
)
from .transverse_material_packed import FourierPackedMaterialCubic
from .transverse_packed import PackedTransverseCubicVertex
from .types import SolverScale

__all__ = [
    "ChannelAccelerationConfig",
    "ChannelAccelerationPlan",
    "ChannelAccelerationResult",
    "ChannelBornMapEvaluation",
    "ChannelCheckpoint",
    "ChannelProposalFailure",
    "ChannelProposalFeasibility",
    "ChannelSCBAConfig",
    "ChannelSCBAPlan",
    "ChannelSCBARecord",
    "ChannelSCBAResult",
    "ChannelSCBAState",
    "ChannelSelectionPolicy",
    "ChannelSigmaResidual",
    "ChannelSpectrum",
    "CompactTransverseContact",
    "FourierPackedMaterialCubic",
    "PackedTransverseCubicVertex",
    "SignedFrequencyGrid",
    "SiteEnergyModel",
    "SiteEnergyStar",
    "SolverScale",
    "UniformBZMesh",
    "build_compact_transverse_contact",
    "build_site_energy_vertex",
    "evaluate_channel_born_map",
    "fit_site_energy_model",
    "load_channel_checkpoint",
    "mix_channel_spectra",
    "plan_channel_accel",
    "plan_channel_born_map",
    "plan_channel_scba",
    "restore_site_energy_model",
    "run_channel_accel",
    "run_channel_scba",
    "save_channel_checkpoint",
]
