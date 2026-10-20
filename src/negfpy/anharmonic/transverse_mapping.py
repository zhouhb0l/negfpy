"""Explicit real-space input adapters for the bounded coupled-q solver.

The two cell offsets are transverse only; finite transport layers remain in
the DOF indices. Image weights and units are explicit. No permutations, crop,
acoustic sum rule, or missing force constants are inferred by this module.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from negfpy.core.types import DeviceKSpace, LeadKSpace
from negfpy.models.material_kspace import (
    MaterialKspaceParams,
    material_kspace_device,
    material_kspace_lead,
)

from ._harmonic import assemble_device_dynamical_matrix, retarded_contact_self_energy
from .contacts import minimum_lead_frequency_squared, reservoir_keldysh_from_retarded
from .fc2 import fc2_force_constant_to_si_factor, mass_to_si_factor
from .fc3 import fc3_force_constant_to_si_factor
from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .transverse_q import TransverseCubicVertex, difference_q_indices
from .transverse_solver import TransverseSCBAProblem, TransverseSpectrum
from .types import HarmonicReservoir, KeldyshSpectrum, SolverScale

_MAX_BYTES = 64 * 1024**2
ContactBuilder = Callable[[tuple[float, float]], KeldyshSpectrum]


def _integer(value: int, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    return result


def _indices(values: Sequence[int], size: int, name: str, minimum=None) -> tuple[int, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must contain {size} integers.")
    return tuple(_integer(v, name, minimum=minimum) for v in values)


def _real(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim or np.iscomplexobj(value):
        raise TypeError(f"{name} must be a finite real scalar.")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


def _mesh(mesh: UniformBZMesh) -> None:
    if not isinstance(mesh, UniformBZMesh) or mesh.shape[2] != 1:
        raise ValueError("Use a full transverse mesh with third period one.")


def _kpoints(mesh: UniformBZMesh) -> np.ndarray:
    _mesh(mesh)
    return 2 * np.pi * np.asarray(mesh.addresses)[:, :2] / np.asarray(mesh.shape[:2])


@dataclass(frozen=True, slots=True)
class TransverseCubicTerm:
    """One ordered ``Psi[0a,s c,t d]`` contribution, before applying ``weight``.

    ``indices`` refer to DOFs in one finite transport device, including layers.
    Offsets have exactly two integer transverse components. A complete real
    Hamiltonian needs all ordered permutations, including reanchored ones.
    The Fourier mapper takes solver-unit coefficients; use the explicitly
    named normalization helper first when starting from raw force constants.
    """

    indices: tuple[int, int, int]
    second_cell: tuple[int, int]
    third_cell: tuple[int, int]
    coefficient: float
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", _indices(self.indices, 3, "indices", 0))
        object.__setattr__(self, "second_cell", _indices(self.second_cell, 2, "second_cell"))
        object.__setattr__(self, "third_cell", _indices(self.third_cell, 2, "third_cell"))
        object.__setattr__(self, "coefficient", _real(self.coefficient, "coefficient"))
        weight = _real(self.weight, "weight")
        if weight <= 0:
            raise ValueError("Image/anchor weight must be positive.")
        object.__setattr__(self, "weight", weight)


def _terms(values: Sequence[TransverseCubicTerm], ndof: int) -> tuple[TransverseCubicTerm, ...]:
    if not isinstance(values, Sequence):
        raise TypeError("terms must be a bounded sequence.")
    if len(values) > 100_000:
        raise ValueError("At most 100000 cubic contributions are supported.")
    terms = tuple(values)
    if any(not isinstance(term, TransverseCubicTerm) for term in terms):
        raise TypeError("Every contribution must be a TransverseCubicTerm.")
    if any(max(term.indices) >= ndof for term in terms):
        raise ValueError("Cubic DOF index exceeds the finite device dimension.")
    return terms


def fourier_transverse_cubic(
    mesh: UniformBZMesh,
    ndof: int,
    terms: Sequence[TransverseCubicTerm],
    *,
    max_bytes: int = _MAX_BYTES,
    max_phase_evaluations: int = 1_000_000,
) -> TransverseCubicVertex:
    """Sum explicit solver-unit terms with positive cell-periodic phases.

    ``V[q,p,a,c,d] = sum weight*coefficient*exp(i*(p.s+(q-p).t))``.
    Aliased transverse images accumulate. There is no additional 1/Nq, 1/6,
    or symmetrization; the Born kernel supplies its own mesh/bubble factors.
    """
    _mesh(mesh)
    ndof = _integer(ndof, "ndof", minimum=1)
    limit = _integer(max_bytes, "max_bytes", minimum=1)
    budget = _integer(max_phase_evaluations, "max_phase_evaluations", minimum=1)
    terms = _terms(terms, ndof)
    nq = mesh.n_qpoints
    if 16 * nq**2 * ndof**3 > limit:
        raise ValueError("Dense Fourier vertex exceeds max_bytes.")
    if nq**2 * len(terms) > budget:
        raise ValueError("Cubic Fourier sum exceeds max_phase_evaluations.")
    routes = difference_q_indices(mesh, max_bytes=limit)
    kpoints = _kpoints(mesh)
    values = np.zeros((nq, nq, ndof, ndof, ndof), dtype=np.complex128)
    for term in terms:
        # Exact integer folding happens before floating-point phase evaluation.
        # Large translated images therefore do not lose their small residue.
        second = tuple(v % n for v, n in zip(term.second_cell, mesh.shape[:2], strict=True))
        third = tuple(v % n for v, n in zip(term.third_cell, mesh.shape[:2], strict=True))
        phase_p = np.exp(1j * (kpoints @ second))
        phase_partner = np.exp(1j * (kpoints @ third))
        a, c, d = term.indices
        values[:, :, a, c, d] += term.weight * term.coefficient * (
            phase_p[None, :] * phase_partner[routes]
        )
    return TransverseCubicVertex(mesh, values, max_bytes=limit)


def mass_normalize_cubic_terms(
    terms: Sequence[TransverseCubicTerm],
    coordinate_masses: np.ndarray,
    *,
    force_constant_units: str,
    mass_units: str,
    scale: SolverScale,
) -> tuple[TransverseCubicTerm, ...]:
    """Convert raw scalar FC3 to solver units exactly once, leaving weights alone.

    Supply one mass per finite-device displacement coordinate. Units are
    explicit: Phi3*sqrt(hbar)/(sqrt(m_a*m_c*m_d)*omega0**2.5). This helper
    neither maps raw crystal images nor certifies their crop/permutations.
    Do not pass coefficients that have already been mass-normalized.
    """
    if not isinstance(scale, SolverScale):
        raise TypeError("scale must be a SolverScale.")
    raw = np.asarray(coordinate_masses)
    if raw.ndim != 1 or raw.size < 1 or np.iscomplexobj(raw):
        raise ValueError("coordinate_masses must be a nonempty real vector.")
    masses = np.asarray(raw, dtype=float) * mass_to_si_factor(mass_units)
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Coordinate masses must be positive and finite.")
    terms = _terms(terms, len(masses))
    factor = fc3_force_constant_to_si_factor(force_constant_units) * (
        np.sqrt(scale.hbar) / scale.omega0**2.5
    )
    result = []
    for term in terms:
        denominator = np.prod(np.sqrt(masses[list(term.indices)]))
        result.append(replace(term, coefficient=term.coefficient * factor / denominator))
    return tuple(result)


def material_harmonic_providers(
    params: MaterialKspaceParams,
    *,
    n_layers: int,
    force_constant_units: str,
    mass_units: str,
    scale: SolverScale,
    layer_masses: np.ndarray | None = None,
) -> tuple[DeviceKSpace, LeadKSpace]:
    """Adapt declared raw FC2/mass units to existing material k-space builders.

    Existing builders do mass division, but no SI/omega0 conversion. Convert
    FC2 and masses here once and keep layer indices explicit. Pinning, if
    supplied, is in the same raw FC2 units. Returned blocks are solver units.
    The source parameters/arrays are detached and never modified.
    """
    if not isinstance(params, MaterialKspaceParams) or not isinstance(scale, SolverScale):
        raise TypeError("Supply MaterialKspaceParams and SolverScale.")
    n_layers = _integer(n_layers, "n_layers", minimum=1)
    _integer(params.dof_per_atom, "dof_per_atom", minimum=1)
    factor = fc2_force_constant_to_si_factor(force_constant_units) / scale.omega0**2
    mass_factor = mass_to_si_factor(mass_units)
    masses = np.array(params.masses, dtype=float, copy=True) * mass_factor
    if not np.isfinite(masses).all() or np.any(masses <= 0):
        raise ValueError("Atomic masses must be positive and finite.")

    def convert(blocks):
        if blocks is None:
            return None
        result = {}
        for offset, block in blocks.items():
            key = _indices(offset, 2, "harmonic offset")
            values = np.asarray(block)
            if np.iscomplexobj(values) and np.any(values.imag != 0):
                raise ValueError("Real-space harmonic force constants must be real.")
            converted = np.array(values.real, dtype=float, copy=True) * factor
            if not np.isfinite(converted).all():
                raise ValueError("Converted harmonic force constants must be finite.")
            converted.setflags(write=False)
            result[key] = converted
        return result

    fc00 = convert(params.fc00_terms)
    fc01 = convert(params.fc01_terms)
    fc10 = convert(params.fc10_terms)

    def require_reverse(forward, reverse, name):
        # Device1D constructs reverse transport blocks by adjoint. A supplied
        # FC10 must agree before using that representation; never discard an
        # incompatible block. The same test applies to same-layer FC00 pairs.
        keys = set(forward) | {tuple(-v for v in key) for key in reverse}
        zero = np.zeros((params.ndof, params.ndof))
        for key in keys:
            a = forward.get(key, zero)
            b = reverse.get(tuple(-v for v in key), zero).T
            magnitude = max(float(np.max(np.abs(a))), float(np.max(np.abs(b))))
            if not np.allclose(a, b, rtol=1e-10, atol=1e-12 * magnitude):
                raise ValueError(f"{name} violates real-space reverse-transpose symmetry.")

    require_reverse(fc00, fc00, "FC00")
    if fc10 is not None:
        require_reverse(fc01, fc10, "FC01/FC10")
    masses.setflags(write=False)
    converted_params = MaterialKspaceParams(
        masses, fc00, fc01, fc10, params.dof_per_atom,
        _real(params.onsite_pinning, "onsite_pinning") * factor,
    )
    local_masses = None
    if layer_masses is not None:
        raw = np.asarray(layer_masses)
        if np.iscomplexobj(raw):
            raise ValueError("Layer masses must be real.")
        local_masses = np.array(raw, dtype=float, copy=True) * mass_factor
        if not np.isfinite(local_masses).all() or np.any(local_masses <= 0):
            raise ValueError("Layer masses must be positive and finite.")
        local_masses.setflags(write=False)
    return (
        material_kspace_device(n_layers, converted_params, local_masses),
        material_kspace_lead(converted_params),
    )


def gapped_contact_spectrum(
    grid: SignedFrequencyGrid,
    reservoir: HarmonicReservoir,
    ndof: int,
    eta_lead: float,
    *,
    side: str,
) -> KeldyshSpectrum:
    """Build one q contact, with explicit left/right orientation and Bose factors.

    Inputs already use solver units. This reuses the existing *gapped* contact
    convention, including its zero-grid-node product. Acoustic/soft leads are
    rejected; their analytic zero limit must be supplied by another callback.
    """
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(reservoir, HarmonicReservoir):
        raise TypeError("Supply SignedFrequencyGrid and HarmonicReservoir.")
    ndof = _integer(ndof, "ndof", minimum=1)
    eta = _real(eta_lead, "eta_lead")
    if eta <= 0 or side not in ("left", "right"):
        raise ValueError("Positive eta_lead and side='left' or 'right' are required.")
    if 3 * grid.n_points * ndof**2 * 16 > _MAX_BYTES:
        raise ValueError("Contact spectrum exceeds the prototype payload limit.")
    lead = reservoir.lead
    gap = minimum_lead_frequency_squared(lead)
    matrix_scale = max(float(np.max(np.abs(lead.d00))), 1.0)
    if gap <= 100 * np.finfo(float).eps * matrix_scale:
        raise ValueError("Gapped contact helper cannot supply an acoustic/soft zero limit.")
    retarded = np.stack([
        retarded_contact_self_energy(w, reservoir, ndof, eta, side=side)
        for w in grid.values
    ])
    return reservoir_keldysh_from_retarded(grid, retarded, reservoir.temperature)


def build_transverse_problem(
    *,
    grid: SignedFrequencyGrid,
    vertex: TransverseCubicVertex,
    device: DeviceKSpace,
    left: ContactBuilder,
    right: ContactBuilder,
    input_units: str,
    scale: SolverScale | None = None,
) -> TransverseSCBAProblem:
    """Join a finite-device vertex and harmonic/contact providers on its mesh.

    Providers receive two transverse *radian* coordinates, using the positive
    cell-periodic phase convention. ``input_units='solver'`` is mandatory:
    no normalization is silently inferred or repeated here. Returned arrays
    are independent of provider storage. Device geometry must be the same at
    every q, and joint q/-q checks are applied by TransverseSCBAProblem.
    """
    if input_units != "solver":
        raise ValueError("Providers and vertex must explicitly declare input_units='solver'.")
    if not isinstance(grid, SignedFrequencyGrid) or not isinstance(vertex, TransverseCubicVertex):
        raise TypeError("Supply SignedFrequencyGrid and TransverseCubicVertex.")
    if not isinstance(device, DeviceKSpace) or not callable(left) or not callable(right):
        raise TypeError("Supply DeviceKSpace and two contact-spectrum callbacks.")
    if scale is None:
        scale = SolverScale()
    nq, d = vertex.mesh.n_qpoints, vertex.ndof
    if 3 * nq * grid.n_points * d**2 * 16 > _MAX_BYTES:
        raise ValueError("Mapped spectrum exceeds the prototype payload limit.")
    matrices, contacts = [], ([], [])
    geometry = None
    for kpoint in _kpoints(vertex.mesh):
        kpar = tuple(float(v) for v in kpoint)
        finite = device.device(kpar)
        current_geometry = (finite.n_layers, finite.dof_per_layer)
        if geometry is not None and current_geometry != geometry:
            raise ValueError("Finite-device geometry changes across q.")
        geometry = current_geometry
        matrix = assemble_device_dynamical_matrix(finite)
        if matrix.shape != (d, d):
            raise ValueError("Harmonic and cubic finite-device dimensions differ.")
        matrices.append(matrix)
        for callback, destination in zip((left, right), contacts, strict=True):
            spectrum = callback(kpar)
            if not isinstance(spectrum, KeldyshSpectrum):
                raise TypeError("Contact callbacks must return KeldyshSpectrum.")
            if spectrum.retarded.shape != (grid.n_points, d, d):
                raise ValueError("Contact spectrum has the wrong grid/device dimensions.")
            destination.append(spectrum)
    mapped = [
        TransverseSpectrum(**{
            component: np.stack([getattr(s, component) for s in spectra])
            for component in ("retarded", "lesser", "greater")
        })
        for spectra in contacts
    ]
    return TransverseSCBAProblem(grid, vertex, np.stack(matrices), *mapped, scale=scale)


__all__ = [
    "TransverseCubicTerm",
    "build_transverse_problem",
    "fourier_transverse_cubic",
    "gapped_contact_spectrum",
    "mass_normalize_cubic_terms",
    "material_harmonic_providers",
]
