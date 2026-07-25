"""Validated public data model for dense cubic anharmonic NEGF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import permutations
from types import MappingProxyType
from typing import Any, ClassVar, Literal

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks

from .grid import SignedFrequencyGrid


Array = np.ndarray
Metadata = Mapping[str, Any]
ContactIndices = tuple[int, ...] | slice | None
SIGMA_PP_BLOCK_LOCAL_RTOL = 1.0e-12


def _readonly_array(values: Any, *, dtype: np.dtype[Any]) -> Array:
    array = np.array(values, dtype=dtype, order="C", copy=True)
    array.setflags(write=False)
    return array


def _readonly_complex(values: Any, *, name: str) -> Array:
    array = _readonly_array(values, dtype=np.dtype(np.complex128))
    if not np.all(np.isfinite(array.real)) or not np.all(np.isfinite(array.imag)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _readonly_real(values: Any, *, name: str) -> Array:
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        if not np.allclose(raw.imag, 0.0, rtol=0.0, atol=1e-14):
            raise ValueError(f"{name} must be real-valued.")
        raw = raw.real
    array = _readonly_array(raw, dtype=np.dtype(np.float64))
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _freeze_mapping(values: Metadata | None) -> Metadata:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping):
        raise TypeError("metadata must be a mapping.")
    return MappingProxyType(
        {str(key): _freeze_metadata_value(value) for key, value in values.items()}
    )


def _freeze_metadata_value(value: Any) -> Any:
    """Detach nested provenance so immutable result records stay immutable."""

    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, order="C")
        array.setflags(write=False)
        return array
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_metadata_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return value


def _finite_real(value: Any, *, name: str, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a real number.")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be finite.")
    if positive and number <= 0.0:
        raise ValueError(f"{name} must be positive.")
    return number


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    normalized = int(value)
    if normalized < 1:
        raise ValueError(f"{name} must be at least one.")
    return normalized


def _normalize_indices(values: Any) -> ContactIndices:
    if values is None:
        return None
    if isinstance(values, slice):
        if values.step == 0:
            raise ValueError("device_indices slice step cannot be zero.")
        return values

    raw = np.asarray(values)
    if raw.ndim == 0:
        raw = raw.reshape(1)
    if raw.ndim != 1:
        raise ValueError("device_indices must be a one-dimensional integer sequence.")
    if raw.size == 0:
        raise ValueError("device_indices must be non-empty.")
    if raw.dtype.kind not in "iu" or raw.dtype.kind == "b":
        raise TypeError("device_indices must contain integers.")

    indices = tuple(int(value) for value in raw)
    if any(value < 0 for value in indices):
        raise ValueError("device_indices cannot contain negative indices.")
    if len(set(indices)) != len(indices):
        raise ValueError("device_indices must be unique.")
    return indices


def _freeze_current_spectra(
    values: Array | Mapping[str, Array] | None,
) -> Array | Metadata | None:
    if values is None:
        return None
    if isinstance(values, Mapping):
        frozen: dict[str, Array] = {}
        for key, spectrum in values.items():
            if not isinstance(key, str) or not key:
                raise ValueError("contact_spectra mapping keys must be non-empty strings.")
            frozen[key] = _readonly_real(spectrum, name=f"contact_spectra[{key!r}]")
        return MappingProxyType(frozen)
    return _readonly_real(values, name="contact_spectra")


@dataclass(frozen=True, slots=True, eq=False)
class SolverScale:
    """Reference scale used to make an anharmonic problem dimensionless.

    ``hbar=omega0=1`` is the reduced-unit convention used by the toy models.
    A physical workflow supplies SI ``hbar`` and a physical angular frequency
    ``omega0``.  Temperatures passed to the solver itself are already
    dimensionless. Conversion factors are added to ``metadata``
    automatically. A non-reduced scale must declare ``source_units`` (or the
    legacy alias ``units``) before a result using it can be release-certified.
    """

    omega0: float = 1.0
    hbar: float = 1.0
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "omega0", _finite_real(self.omega0, name="omega0", positive=True))
        object.__setattr__(self, "hbar", _finite_real(self.hbar, name="hbar", positive=True))
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")
        metadata = dict(self.metadata)
        reduced = self.omega0 == 1.0 and self.hbar == 1.0
        source_units = metadata.get("source_units", metadata.get("units"))
        if reduced and source_units is None:
            source_units = "hbar=k_B=omega0=1"

        source_declared = reduced
        if isinstance(source_units, str):
            normalized = source_units.strip().lower()
            source_declared = bool(
                normalized and normalized not in {"unknown", "unspecified"}
            )
            if not source_declared:
                source_units = "unspecified"
        elif isinstance(source_units, Mapping):
            source_declared = bool(source_units)
        elif source_units is not None:
            source_declared = False
            source_units = "unspecified"
        if source_units is None:
            source_units = "unspecified"

        # These fields are reserved and derived from the actual scale. Caller
        # metadata may not override numerical conversion provenance.
        metadata["unit_system"] = (
            "reduced" if reduced else "physical_reference_scale"
        )
        metadata["source_units"] = source_units
        metadata["source_units_declared"] = source_declared
        metadata["problem_units"] = "dimensionless_solver_units"
        metadata["dynamical_matrix_to_solver_factor"] = (
            1.0 / self.omega0**2
        )
        metadata["cubic_vertex_to_solver_factor"] = (
            np.sqrt(self.hbar) / self.omega0**2.5
        )
        metadata["temperature_energy_to_solver_factor"] = (
            1.0 / (self.hbar * self.omega0)
        )
        metadata["current_solver_to_physical_factor"] = (
            self.hbar * self.omega0**2
        )
        object.__setattr__(self, "metadata", _freeze_mapping(metadata))

    @property
    def omega_0(self) -> float:
        """Spelling alias matching the mathematical symbol."""

        return self.omega0

    @property
    def current_scale(self) -> float:
        """Physical current represented by one solver current unit."""

        return self.hbar * self.omega0**2

    def dimensionless_dynamical_matrix(self, values: Any) -> Array:
        return np.asarray(values) / self.omega0**2

    def dimensionless_cubic_vertex(self, values: Any) -> Array:
        return np.asarray(values) * np.sqrt(self.hbar) / self.omega0**2.5

    def dimensionless_temperature(
        self,
        temperature: Any,
        *,
        boltzmann_constant: float = 1.0,
    ) -> Array | float:
        k_b = _finite_real(
            boltzmann_constant,
            name="boltzmann_constant",
            positive=True,
        )
        result = np.asarray(temperature) * k_b / (self.hbar * self.omega0)
        if result.ndim == 0:
            return float(result)
        return result

    def to_physical_current(self, current: Any) -> Array | float:
        result = np.asarray(current) * self.current_scale
        if result.ndim == 0:
            return float(result)
        return result

    def from_physical_current(self, current: Any) -> Array | float:
        result = np.asarray(current) / self.current_scale
        if result.ndim == 0:
            return float(result)
        return result


@dataclass(frozen=True, slots=True, eq=False)
class DenseCubicVertex:
    """Dense, real, mass-normalized cubic vertex in solver units."""

    values: Array
    metadata: Metadata = field(default_factory=dict)

    symmetry_rtol: ClassVar[float] = 1e-10
    symmetry_atol: ClassVar[float] = 1e-12

    def __post_init__(self) -> None:
        if np.iscomplexobj(np.asarray(self.values)):
            raise ValueError("values must be a real cubic tensor.")
        values = _readonly_real(self.values, name="values")
        if values.ndim != 3 or len(set(values.shape)) != 1:
            raise ValueError("values must have shape (ndof, ndof, ndof).")
        if values.shape[0] < 1:
            raise ValueError("values must contain at least one degree of freedom.")

        for axes in permutations(range(3)):
            if not np.allclose(
                values,
                np.transpose(values, axes),
                rtol=self.symmetry_rtol,
                atol=self.symmetry_atol,
            ):
                raise ValueError(
                    "values must be permutation symmetric under all cubic-index exchanges."
                )

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def ndof(self) -> int:
        return int(self.values.shape[0])

    @property
    def support_density(self) -> float:
        return float(np.count_nonzero(self.values) / self.values.size)


@dataclass(frozen=True, slots=True, eq=False)
class HarmonicReservoir:
    """A harmonic lead, its temperature, and its finite-device contact."""

    lead: LeadBlocks
    temperature: float
    device_to_lead: Array | None = None
    device_indices: ContactIndices | Sequence[int] | Array = None

    def __post_init__(self) -> None:
        if not isinstance(self.lead, LeadBlocks):
            raise TypeError("lead must be a LeadBlocks instance.")
        temperature = _finite_real(self.temperature, name="temperature")
        if temperature < 0.0:
            raise ValueError("temperature must be non-negative.")

        coupling: Array | None = None
        if self.device_to_lead is not None:
            coupling = _readonly_complex(self.device_to_lead, name="device_to_lead")
            if coupling.ndim != 2:
                raise ValueError("device_to_lead must be a two-dimensional array.")
            if coupling.shape[1] != self.lead.d00.shape[0]:
                raise ValueError(
                    "device_to_lead column count must equal the lead principal-layer dimension."
                )

        indices = _normalize_indices(self.device_indices)
        if coupling is not None and isinstance(indices, tuple) and len(indices) != coupling.shape[0]:
            raise ValueError(
                "device_indices count must equal the number of device_to_lead rows."
            )

        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "device_to_lead", coupling)
        object.__setattr__(self, "device_indices", indices)

    def resolved_device_to_lead(self, side: Literal["left", "right"]) -> Array:
        """Return the explicit coupling, or the periodic matched-contact default."""

        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        if self.device_to_lead is not None:
            return self.device_to_lead
        if side == "left":
            coupling = self.lead.d10 if self.lead.d10 is not None else self.lead.d01.conj().T
        else:
            coupling = self.lead.d01
        return _readonly_complex(coupling, name="device_to_lead")

    def resolved_device_indices(
        self,
        *,
        dim: int,
        side: Literal["left", "right"],
    ) -> Array:
        """Resolve default/slice contact indices for a device dimension."""

        if side not in {"left", "right"}:
            raise ValueError("side must be 'left' or 'right'.")
        dim = _positive_int(dim, name="dim")
        n_contact = self.resolved_device_to_lead(side).shape[0]

        if self.device_indices is None:
            if n_contact > dim:
                raise ValueError("contact dimension exceeds the finite-device dimension.")
            if side == "left":
                values = np.arange(n_contact, dtype=np.int64)
            else:
                values = np.arange(dim - n_contact, dim, dtype=np.int64)
        elif isinstance(self.device_indices, slice):
            values = np.arange(dim, dtype=np.int64)[self.device_indices]
        else:
            values = np.asarray(self.device_indices, dtype=np.int64)

        if values.size == 0:
            raise ValueError("resolved device_indices must be non-empty.")
        if np.any(values < 0) or np.any(values >= dim):
            raise ValueError("device_indices are outside the finite-device range.")
        if np.unique(values).size != values.size:
            raise ValueError("device_indices must be unique.")
        if values.size != n_contact:
            raise ValueError(
                "resolved device_indices count must equal the contact-coupling row count."
            )
        return _readonly_array(values, dtype=np.dtype(np.int64))


@dataclass(frozen=True, slots=True, eq=False)
class CubicSCBAProblem:
    """Complete dense cubic-SCBA problem in dimensionless solver units."""

    device: Device1D
    left: HarmonicReservoir
    right: HarmonicReservoir
    vertex: DenseCubicVertex
    grid: SignedFrequencyGrid
    scale: SolverScale = field(default_factory=SolverScale)

    def __post_init__(self) -> None:
        if not isinstance(self.device, Device1D):
            raise TypeError("device must be a Device1D instance in v0.1.")
        if not isinstance(self.left, HarmonicReservoir):
            raise TypeError("left must be a HarmonicReservoir instance.")
        if not isinstance(self.right, HarmonicReservoir):
            raise TypeError("right must be a HarmonicReservoir instance.")
        if not isinstance(self.vertex, DenseCubicVertex):
            raise TypeError("vertex must be a DenseCubicVertex instance.")
        if not isinstance(self.grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid instance.")
        if not isinstance(self.scale, SolverScale):
            raise TypeError("scale must be a SolverScale instance.")

        if self.vertex.ndof != self.ndof:
            raise ValueError(
                f"vertex ndof ({self.vertex.ndof}) does not match device ndof ({self.ndof})."
            )
        self.left.resolved_device_indices(dim=self.ndof, side="left")
        self.right.resolved_device_indices(dim=self.ndof, side="right")

    @property
    def ndof(self) -> int:
        return self.device.n_layers * self.device.dof_per_layer


@dataclass(frozen=True, slots=True)
class SCBAConfig:
    """Numerical controls for one-shot Born and conserving SCBA."""

    eta_lead: float
    max_iterations: int = 200
    min_iterations: int = 3
    mixing: float = 0.2
    mixing_floor: float = 0.01
    consecutive_passes: int = 3
    sigma_rtol: float = 1e-8
    sigma_atol: float = 1e-12
    green_rtol: float = 1e-8
    green_atol: float = 1e-12
    identity_rtol: float = 1e-8
    current_rtol: float = 1e-5
    current_atol: float = 1e-10
    tail_rtol: float = 1e-6
    retarded_strategy: Literal["causal_hilbert", "spectral_only"] = "causal_hilbert"
    hilbert_backend: Literal["direct", "fft"] = "direct"
    backend: Literal["serial", "threaded", "fft"] = "serial"
    on_nonconvergence: Literal["raise", "return"] = "raise"
    eta_device: float = 0.0
    surface_gf_method: str = "sancho_rubio"
    max_workers: int | None = None
    max_memory_fraction: float = 0.70
    allow_memory_overcommit: bool = False

    def __post_init__(self) -> None:
        eta_lead = _finite_real(self.eta_lead, name="eta_lead", positive=True)
        eta_device = _finite_real(self.eta_device, name="eta_device")
        if eta_device != 0.0:
            raise ValueError(
                "eta_device must be zero in certified v0.1 calculations; "
                "finite dissipation requires an explicit Keldysh reservoir."
            )

        max_iterations = _positive_int(self.max_iterations, name="max_iterations")
        min_iterations = _positive_int(self.min_iterations, name="min_iterations")
        consecutive = _positive_int(self.consecutive_passes, name="consecutive_passes")
        if min_iterations > max_iterations:
            raise ValueError("min_iterations cannot exceed max_iterations.")
        if consecutive > max_iterations:
            raise ValueError("consecutive_passes cannot exceed max_iterations.")
        if min_iterations + consecutive - 1 > max_iterations:
            raise ValueError(
                "max_iterations must allow the requested consecutive passes "
                "after min_iterations."
            )

        mixing = _finite_real(self.mixing, name="mixing", positive=True)
        mixing_floor = _finite_real(self.mixing_floor, name="mixing_floor", positive=True)
        if mixing > 1.0:
            raise ValueError("mixing cannot exceed one.")
        if mixing_floor > mixing:
            raise ValueError("mixing_floor cannot exceed mixing.")

        for name in (
            "sigma_rtol",
            "sigma_atol",
            "green_rtol",
            "green_atol",
            "identity_rtol",
            "current_rtol",
            "current_atol",
            "tail_rtol",
        ):
            value = _finite_real(getattr(self, name), name=name)
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative.")
            object.__setattr__(self, name, value)

        if self.retarded_strategy not in {"causal_hilbert", "spectral_only"}:
            raise ValueError("unsupported retarded_strategy.")
        if self.hilbert_backend not in {"direct", "fft"}:
            raise ValueError("hilbert_backend must be 'direct' or 'fft'.")
        if self.backend not in {"serial", "threaded", "fft"}:
            raise ValueError(
                "backend must be 'serial', 'threaded', or 'fft'."
            )
        if self.on_nonconvergence not in {"raise", "return"}:
            raise ValueError("on_nonconvergence must be 'raise' or 'return'.")
        if not isinstance(self.surface_gf_method, str) or not self.surface_gf_method:
            raise ValueError("surface_gf_method must be a non-empty string.")
        if self.max_workers is not None:
            object.__setattr__(
                self,
                "max_workers",
                _positive_int(self.max_workers, name="max_workers"),
            )
        max_memory_fraction = _finite_real(
            self.max_memory_fraction,
            name="max_memory_fraction",
            positive=True,
        )
        if max_memory_fraction > 1.0:
            raise ValueError("max_memory_fraction cannot exceed one.")
        if not isinstance(self.allow_memory_overcommit, (bool, np.bool_)):
            raise TypeError("allow_memory_overcommit must be boolean.")

        object.__setattr__(self, "eta_lead", eta_lead)
        object.__setattr__(self, "eta_device", eta_device)
        object.__setattr__(self, "max_iterations", max_iterations)
        object.__setattr__(self, "min_iterations", min_iterations)
        object.__setattr__(self, "consecutive_passes", consecutive)
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "mixing_floor", mixing_floor)
        object.__setattr__(self, "max_memory_fraction", max_memory_fraction)
        object.__setattr__(
            self,
            "allow_memory_overcommit",
            bool(self.allow_memory_overcommit),
        )


@dataclass(frozen=True, slots=True, eq=False)
class KeldyshSpectrum:
    """Retarded, lesser, and greater matrices on one frequency grid."""

    retarded: Array
    lesser: Array
    greater: Array

    def __post_init__(self) -> None:
        retarded = _readonly_complex(self.retarded, name="retarded")
        lesser = _readonly_complex(self.lesser, name="lesser")
        greater = _readonly_complex(self.greater, name="greater")
        if retarded.ndim != 3 or retarded.shape[1] != retarded.shape[2]:
            raise ValueError(
                "Keldysh spectra must have shape (n_frequency, ndof, ndof)."
            )
        if lesser.shape != retarded.shape or greater.shape != retarded.shape:
            raise ValueError("retarded, lesser, and greater spectra must share one shape.")
        object.__setattr__(self, "retarded", retarded)
        object.__setattr__(self, "lesser", lesser)
        object.__setattr__(self, "greater", greater)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.retarded.shape

    @property
    def n_frequency(self) -> int:
        return int(self.retarded.shape[0])

    @property
    def ndof(self) -> int:
        return int(self.retarded.shape[1])

    @classmethod
    def zeros(cls, *, n_frequency: int, ndof: int) -> KeldyshSpectrum:
        n_frequency = _positive_int(n_frequency, name="n_frequency")
        ndof = _positive_int(ndof, name="ndof")
        shape = (n_frequency, ndof, ndof)
        return cls(
            retarded=np.zeros(shape, dtype=np.complex128),
            lesser=np.zeros(shape, dtype=np.complex128),
            greater=np.zeros(shape, dtype=np.complex128),
        )


@dataclass(frozen=True, slots=True)
class SCBAIteration:
    """One immutable SCBA convergence-history record."""

    iteration: int
    sigma_residual: float
    green_residual: float
    identity_residual: float
    current_abs_error: float
    current_rel_error: float
    collision_power: float
    mixing: float
    passed: bool
    consecutive_passes: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.iteration, (bool, np.bool_)) or not isinstance(
            self.iteration, (int, np.integer)
        ):
            raise TypeError("iteration must be an integer.")
        if int(self.iteration) < 0:
            raise ValueError("iteration must be non-negative.")
        if isinstance(self.consecutive_passes, (bool, np.bool_)) or not isinstance(
            self.consecutive_passes, (int, np.integer)
        ):
            raise TypeError("consecutive_passes must be an integer.")
        if int(self.consecutive_passes) < 0:
            raise ValueError("consecutive_passes must be non-negative.")
        for name in (
            "sigma_residual",
            "green_residual",
            "identity_residual",
            "current_abs_error",
            "current_rel_error",
        ):
            value = float(getattr(self, name))
            if np.isnan(value) or value < 0.0:
                raise ValueError(f"{name} must be non-negative and not NaN.")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "collision_power",
            _finite_real(self.collision_power, name="collision_power"),
        )
        mixing = _finite_real(self.mixing, name="mixing", positive=True)
        if mixing > 1.0:
            raise ValueError("mixing cannot exceed one.")
        object.__setattr__(self, "mixing", mixing)
        object.__setattr__(self, "iteration", int(self.iteration))
        object.__setattr__(self, "consecutive_passes", int(self.consecutive_passes))
        object.__setattr__(self, "passed", bool(self.passed))


@dataclass(frozen=True, slots=True, eq=False)
class SpatialCurrentReport:
    """Layer-resolved energy exchange and corrected one-dimensional cuts.

    ``eom_exchange_spectra`` is the projected Dyson-equation source from the
    interaction self-energy. ``collision_power_spectra`` is the distinct
    greater/lesser collision integral. Both have shape
    ``(n_layers, n_positive)``. The remaining arrays are their integrated
    layer values and the one-dimensional cut-current closure.
    ``nonlocal_self_energy_eom_cut_correction`` is an equation-of-motion
    correction induced by off-layer self-energy blocks; it is not a bare
    interaction-current operator for a cubic vertex crossing a cut.
    """

    sigma_pp_nonlocal_ratio: float
    eom_exchange_spectra: Array
    collision_power_spectra: Array
    eom_layer_exchange: Array
    layer_collision_power: Array
    nonlocal_self_energy_eom_cut_correction: Array
    corrected_cut_currents: Array
    continuity_residual: Array

    def __post_init__(self) -> None:
        ratio = _finite_real(
            self.sigma_pp_nonlocal_ratio,
            name="sigma_pp_nonlocal_ratio",
        )
        if ratio < 0.0 or ratio > 1.0 + 1.0e-14:
            raise ValueError("sigma_pp_nonlocal_ratio must lie in [0, 1].")
        object.__setattr__(self, "sigma_pp_nonlocal_ratio", min(ratio, 1.0))
        names = (
            "eom_exchange_spectra",
            "collision_power_spectra",
            "eom_layer_exchange",
            "layer_collision_power",
            "nonlocal_self_energy_eom_cut_correction",
            "corrected_cut_currents",
            "continuity_residual",
        )
        for name in names:
            object.__setattr__(
                self,
                name,
                _readonly_real(getattr(self, name), name=name),
            )

        eom_spectra = self.eom_exchange_spectra
        collision_spectra = self.collision_power_spectra
        if eom_spectra.ndim != 2 or eom_spectra.shape[0] < 1:
            raise ValueError(
                "eom_exchange_spectra must have shape "
                "(n_layers, n_positive)."
            )
        if collision_spectra.shape != eom_spectra.shape:
            raise ValueError(
                "collision_power_spectra must match "
                "eom_exchange_spectra."
            )
        n_layers = eom_spectra.shape[0]
        for name in (
            "eom_layer_exchange",
            "layer_collision_power",
            "continuity_residual",
        ):
            if getattr(self, name).shape != (n_layers,):
                raise ValueError(f"{name} must have shape ({n_layers},).")
        cut_shape = (max(0, n_layers - 1),)
        for name in (
            "nonlocal_self_energy_eom_cut_correction",
            "corrected_cut_currents",
        ):
            if getattr(self, name).shape != cut_shape:
                raise ValueError(f"{name} must have shape {cut_shape}.")

    @property
    def sigma_pp_is_block_local(self) -> bool:
        """Whether off-layer weight is below the fixed structural threshold."""

        return self.sigma_pp_nonlocal_ratio <= SIGMA_PP_BLOCK_LOCAL_RTOL


@dataclass(frozen=True, slots=True, eq=False)
class CurrentReport:
    """Integrated and spectral energy-current diagnostics after scale conversion."""

    left: float
    right: float
    collision_power: float
    contact_spectra: Array | Mapping[str, Array] | None = None
    collision_spectrum: Array | None = None
    bond_currents: Array | None = None
    bond_spectra: Array | None = None
    spatial: SpatialCurrentReport | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "left", _finite_real(self.left, name="left"))
        object.__setattr__(self, "right", _finite_real(self.right, name="right"))
        object.__setattr__(
            self,
            "collision_power",
            _finite_real(self.collision_power, name="collision_power"),
        )
        object.__setattr__(
            self,
            "contact_spectra",
            _freeze_current_spectra(self.contact_spectra),
        )
        for name in ("collision_spectrum", "bond_currents", "bond_spectra"):
            values = getattr(self, name)
            if values is not None:
                object.__setattr__(self, name, _readonly_real(values, name=name))
        if self.spatial is not None and not isinstance(
            self.spatial,
            SpatialCurrentReport,
        ):
            raise TypeError("spatial must be a SpatialCurrentReport or None.")

    @property
    def contact_balance(self) -> float:
        return self.left + self.right

    @property
    def full_balance(self) -> float:
        return self.contact_balance + self.collision_power


@dataclass(frozen=True, slots=True, eq=False)
class DiagnosticReport:
    """Named numerical diagnostics and release-certification state."""

    metrics: Mapping[str, float] = field(default_factory=dict)
    failures: tuple[str, ...] = ()
    passed: bool = False
    release_certified: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping.")
        metrics: dict[str, float] = {}
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name:
                raise ValueError("diagnostic metric names must be non-empty strings.")
            metrics[name] = float(value)
        failures = tuple(str(value) for value in self.failures)
        if any(not value for value in failures):
            raise ValueError("diagnostic failure messages must be non-empty.")
        passed = bool(self.passed)
        release_certified = bool(self.release_certified)
        if release_certified and not passed:
            raise ValueError("a release-certified diagnostic report must pass.")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "failures", failures)
        object.__setattr__(self, "passed", passed)
        object.__setattr__(self, "release_certified", release_certified)

    @property
    def certified(self) -> bool:
        return self.release_certified


@dataclass(frozen=True, slots=True, eq=False)
class AnharmonicResult:
    """Complete immutable output of a one-shot Born or SCBA calculation."""

    method: str
    grid: SignedFrequencyGrid
    green: KeldyshSpectrum
    sigma_pp: KeldyshSpectrum
    sigma_left: KeldyshSpectrum
    sigma_right: KeldyshSpectrum
    currents: CurrentReport
    diagnostics: DiagnosticReport
    history: tuple[SCBAIteration, ...] = ()
    converged: bool = False
    release_certified: bool = False
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.method, str) or not self.method:
            raise ValueError("method must be a non-empty string.")
        if not isinstance(self.grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid instance.")
        spectra = {
            "green": self.green,
            "sigma_pp": self.sigma_pp,
            "sigma_left": self.sigma_left,
            "sigma_right": self.sigma_right,
        }
        expected_shape = self.green.shape
        for name, spectrum in spectra.items():
            if not isinstance(spectrum, KeldyshSpectrum):
                raise TypeError(f"{name} must be a KeldyshSpectrum instance.")
            if spectrum.shape != expected_shape:
                raise ValueError("all Green-function and self-energy spectra must share one shape.")
        if expected_shape[0] != self.grid.n_points:
            raise ValueError("spectrum frequency dimension must match the signed grid.")
        if not isinstance(self.currents, CurrentReport):
            raise TypeError("currents must be a CurrentReport instance.")
        if not isinstance(self.diagnostics, DiagnosticReport):
            raise TypeError("diagnostics must be a DiagnosticReport instance.")
        history = tuple(self.history)
        if any(not isinstance(record, SCBAIteration) for record in history):
            raise TypeError("history must contain only SCBAIteration records.")

        converged = bool(self.converged)
        release_certified = bool(self.release_certified)
        if release_certified and (
            not converged
            or not self.diagnostics.passed
            or not self.diagnostics.release_certified
        ):
            raise ValueError(
                "release_certified requires convergence and certified passing diagnostics."
            )
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "converged", converged)
        object.__setattr__(self, "release_certified", release_certified)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    @property
    def iteration_history(self) -> tuple[SCBAIteration, ...]:
        return self.history


def estimate_dense_memory_bytes(
    n_frequency: int,
    ndof: int,
    *,
    n_q: int = 1,
) -> int:
    """Conservative dense-SCBA peak-memory estimate from the v0.1 plan."""

    n_frequency = _positive_int(n_frequency, name="n_frequency")
    ndof = _positive_int(ndof, name="ndof")
    n_q = _positive_int(n_q, name="n_q")
    return 288 * n_frequency * n_q * ndof**2
