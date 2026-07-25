r"""Audited fixtures for Mingo's 2006 diatomic molecular junction.

The paper contains enough information to reconstruct its harmonic model, but
not enough to silently identify every historical sign and normalization
choice.  This module therefore keeps three things separate:

* a causal, normalized analytic surface resolvent;
* a nonphysical half-weight sensitivity inferred from Fig. 4;
* the :math:`\pi/9` legacy cubic-bubble experiment audited by Guo et al.

The last two are historical-reproduction aids.  Neither changes the
production Guo cubic kernel or qualifies a result for release certification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..cubic import cubic_lesser_greater
from ..grid import SignedFrequencyGrid
from ..types import DenseCubicVertex


Array = np.ndarray
MingoSurfaceConvention = Literal[
    "causal_normalized",
    "figure4_inferred_half_weight_sensitivity",
]

_ELECTRON_VOLT_J = 1.602176634e-19
_ANGSTROM_M = 1.0e-10
_ATOMIC_MASS_KG = 1.66053906660e-27
_HBAR_J_S = 1.0545718176461565e-34
_BOLTZMANN_J_K = 1.380649e-23

_SOURCE_DOI = "10.1103/PhysRevB.74.125402"
_SOURCE_PDF_SHA256 = (
    "ffbeb6111f226c9c7a7296091ef331d22d8ded1df8364f7c71e8a181aa4c3030"
)


def _readonly_real(values: Array | list[float] | tuple[float, ...]) -> Array:
    result = np.array(values, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _positive_finite(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a positive finite real number.")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite real number.")
    return result


@dataclass(frozen=True, slots=True)
class Mingo2006SurfaceInterpretation:
    """A named interpretation of the paper's inconsistent surface formulas."""

    name: str
    spectral_measure_weight: float
    normalized_spectral_measure: bool
    derivable_from_printed_equations: bool
    release_certifiable: bool
    qualification: str


MINGO_2006_CAUSAL_NORMALIZED_SURFACE = Mingo2006SurfaceInterpretation(
    name="causal_normalized",
    spectral_measure_weight=1.0,
    normalized_spectral_measure=True,
    derivable_from_printed_equations=True,
    release_certifiable=False,
    qualification=(
        "Causal resolvent of the Eq. (47) density normalized by "
        "integral rho(omega^2) d(omega^2)=1. The gapless historical "
        "reservoir is not a certified v0.1 SCBA contact."
    ),
)

MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY = (
    Mingo2006SurfaceInterpretation(
        name="figure4_inferred_half_weight_sensitivity",
        spectral_measure_weight=0.5,
        normalized_spectral_measure=False,
        derivable_from_printed_equations=False,
        release_certifiable=False,
        qualification=(
            "Fixed sensitivity inferred from the Fig. 4 vector curve. "
            "Eq. (47) does not justify the half weight; this is not a "
            "same-input reproduction oracle."
        ),
    )
)


def _surface_weight(convention: MingoSurfaceConvention | str) -> float:
    if convention == "causal_normalized":
        return MINGO_2006_CAUSAL_NORMALIZED_SURFACE.spectral_measure_weight
    if convention == "figure4_inferred_half_weight_sensitivity":
        return MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY.spectral_measure_weight
    raise ValueError(
        "convention must be 'causal_normalized' or "
        "'figure4_inferred_half_weight_sensitivity'."
    )


@dataclass(frozen=True, slots=True)
class Mingo2006Approximation:
    """A named, immutable approximation-level descriptor."""

    name: str
    self_consistent: bool
    cubic_bubble_scale_relative_to_guo: float
    retarded_reconstruction: str
    release_certifiable: bool
    implementation_status: str
    source: str
    qualification: str


GUO_2020_PRODUCTION_SCBA = Mingo2006Approximation(
    name="guo_2020_conserving_cubic_scba",
    self_consistent=True,
    cubic_bubble_scale_relative_to_guo=1.0,
    retarded_reconstruction="causal_hilbert",
    release_certifiable=True,
    implementation_status="implemented_by_solve_scba",
    source="PhysRevB.102.195412, Eqs. (7) and (8)",
    qualification=(
        "Production Hamiltonian and bubble convention; no adjustable "
        "diagrammatic prefactor."
    ),
)

MINGO_2006_LEGACY_EQ27 = Mingo2006Approximation(
    name="mingo_2006_eq27_guo_audited_pi_over_9",
    self_consistent=True,
    cubic_bubble_scale_relative_to_guo=float(np.pi / 9.0),
    retarded_reconstruction="causal_hilbert",
    release_certifiable=False,
    implementation_status="kernel_only_no_mingo_scba_driver",
    source=(
        "PhysRevB.74.125402 Eq. (27), mapped using "
        "PhysRevB.102.195412 Table I and Table IV"
    ),
    qualification=(
        "Historical comparison only: after mapping V^(3)=Psi/6, multiply "
        "the production lesser and greater bubbles by pi/9. This does not "
        "alter the production solver."
    ),
)

MINGO_2006_UNVERIFIED_VERTEX_X6_SENSITIVITY = Mingo2006Approximation(
    name="mingo_2006_unverified_v3_input_x6_sensitivity",
    self_consistent=True,
    cubic_bubble_scale_relative_to_guo=float(4.0 * np.pi),
    retarded_reconstruction="causal_hilbert",
    release_certifiable=False,
    implementation_status="not_implemented_hypothesis_only",
    source=(
        "Unverified hypothesis prompted by the unresolved distinction "
        "between the paper's V^(3)=V'''/6 and its numerical V''' input"
    ),
    qualification=(
        "If historical code injected V''' into a slot derived for V^(3), "
        "the vertex is six times too large and the pi/9 legacy bubble is "
        "36*pi/9=4*pi times production. No primary code supports this, so "
        "it must not be used as a validation convention."
    ),
)


@dataclass(frozen=True, slots=True, eq=False)
class Mingo2006PublishedCurve:
    """Vector-extracted values from a named curve in the primary-source PDF."""

    figure: int
    series: str
    temperature_kelvin: Array
    values: Array
    value_units: str
    extraction_note: str

    def __post_init__(self) -> None:
        temperatures = _readonly_real(self.temperature_kelvin)
        values = _readonly_real(self.values)
        if temperatures.ndim != 1 or values.ndim != 1:
            raise ValueError("published-curve arrays must be one-dimensional.")
        if temperatures.shape != values.shape or temperatures.size < 2:
            raise ValueError(
                "temperature_kelvin and values must have the same nontrivial shape."
            )
        if not np.all(np.isfinite(temperatures)) or not np.all(np.isfinite(values)):
            raise ValueError("published-curve arrays must be finite.")
        if np.any(temperatures <= 0.0) or np.any(np.diff(temperatures) <= 0.0):
            raise ValueError("published-curve temperatures must increase.")
        object.__setattr__(self, "temperature_kelvin", temperatures)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True, slots=True)
class Mingo2006DiatomicFixture:
    """Recoverable inputs for the two-Si-atom junction in Figs. 4 and 8."""

    mass_atomic_units: float = 28.0
    omega0_radian_per_second: float = 1.0e12
    cutoff_solver: float = 100.0
    molecule_surface_ev_per_angstrom2: float = 3.16
    paper_central_ev_per_angstrom2: float = 0.316
    lj_beta_per_angstrom: float = 8.94
    sw_beta_per_angstrom: float = 2.46

    def __post_init__(self) -> None:
        for field_name in (
            "mass_atomic_units",
            "omega0_radian_per_second",
            "cutoff_solver",
            "molecule_surface_ev_per_angstrom2",
            "paper_central_ev_per_angstrom2",
            "lj_beta_per_angstrom",
            "sw_beta_per_angstrom",
        ):
            object.__setattr__(
                self,
                field_name,
                _positive_finite(getattr(self, field_name), name=field_name),
            )

    @property
    def mass_kg(self) -> float:
        return self.mass_atomic_units * _ATOMIC_MASS_KG

    @property
    def force_constant_to_solver(self) -> float:
        """Convert eV/angstrom^2 to ``omega0^2`` solver units."""

        return (
            _ELECTRON_VOLT_J
            / _ANGSTROM_M**2
            / (self.mass_kg * self.omega0_radian_per_second**2)
        )

    @property
    def molecule_surface_solver(self) -> float:
        return (
            self.molecule_surface_ev_per_angstrom2
            * self.force_constant_to_solver
        )

    @property
    def central_bond_solver(self) -> float:
        r"""Return the central Hessian spring in solver units.

        Mingo defines the plotted parameter as
        :math:`k_m=\tfrac12 V''(x_0)`.  The force-constant matrix therefore
        contains :math:`V''=2k_m`.
        """

        return 2.0 * self.paper_central_ev_per_angstrom2 * self.force_constant_to_solver

    def cubic_vertex(self, beta_per_angstrom: float) -> DenseCubicVertex:
        r"""Return the mass-normalized bond vertex for ``V'''/V''=beta``."""

        beta = _positive_finite(beta_per_angstrom, name="beta_per_angstrom")
        third_derivative_si = (
            2.0
            * self.paper_central_ev_per_angstrom2
            * beta
            * _ELECTRON_VOLT_J
            / _ANGSTROM_M**3
        )
        vertex_scale = (
            third_derivative_si
            * np.sqrt(_HBAR_J_S)
            / (
                self.mass_kg**1.5
                * self.omega0_radian_per_second**2.5
            )
        )
        signs = np.array([1.0, -1.0], dtype=np.float64)
        values = vertex_scale * np.einsum(
            "a,b,c->abc",
            signs,
            signs,
            signs,
        )
        return DenseCubicVertex(
            values=values,
            metadata={
                "model": "Mingo2006_diatomic_bond",
                "source_doi": _SOURCE_DOI,
                "source_pdf_sha256": _SOURCE_PDF_SHA256,
                "interaction_topology": "bond",
                "harmonic_bond_current_supported": False,
                "beta_per_angstrom": beta,
                "paper_km_ev_per_angstrom2": (
                    self.paper_central_ev_per_angstrom2
                ),
                "paper_definition": "k_m=(1/2)*V''; V'''=beta*V''",
                "vertex_convention": (
                    "H3=(1/3!)*sum(Psi_abc*u_a*u_b*u_c)"
                ),
            },
        )

    def lj_vertex(self) -> DenseCubicVertex:
        return self.cubic_vertex(self.lj_beta_per_angstrom)

    def sw_vertex(self) -> DenseCubicVertex:
        return self.cubic_vertex(self.sw_beta_per_angstrom)


def mingo_2006_bare_surface_resolvent(
    frequency: Array | float,
    *,
    cutoff: float = 100.0,
    convention: MingoSurfaceConvention = "causal_normalized",
) -> Array | complex:
    r"""Return the eta-free retarded resolvent for Mingo's surface DOS.

    For the normalized measure

    .. math::

       \rho(\lambda)=\frac{3\sqrt{\lambda}}{2\omega_D^3},
       \quad 0\leq\lambda\leq\omega_D^2,

    direct integration gives

    .. math::

       g_0^r(\omega)=\frac{3}{\omega_D^3}
       \left[-\omega_D+\frac{\omega}{2}
       \log\frac{\omega+\omega_D}{\omega-\omega_D}\right],

    with the retarded branch.  No numerical ``eta`` is used.  The bare real
    part has a logarithmic singularity at the sharp cutoff; the dressed
    resolvent below has a finite analytic limit there.
    """

    omega_d = _positive_finite(cutoff, name="cutoff")
    raw = np.asarray(frequency)
    if np.iscomplexobj(raw):
        raise ValueError("frequency must be real-valued.")
    omega = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("frequency must contain only finite values.")

    spectral_weight = _surface_weight(convention)
    magnitude = np.abs(omega)
    result = np.empty(omega.shape, dtype=np.complex128)
    below = magnitude < omega_d
    above = magnitude > omega_d
    at_cutoff = ~(below | above)

    if np.any(below):
        selected = magnitude[below]
        log_term = np.zeros_like(selected)
        nonzero = selected > 0.0
        log_term[nonzero] = np.log(
            (omega_d + selected[nonzero])
            / (omega_d - selected[nonzero])
        )
        real = 3.0 / omega_d**3 * (
            -omega_d + 0.5 * selected * log_term
        )
        imaginary = -3.0 * np.pi * selected / (2.0 * omega_d**3)
        result[below] = spectral_weight * (real + 1j * imaginary)

    if np.any(above):
        selected = magnitude[above]
        selected_result = np.empty(selected.shape, dtype=np.complex128)
        asymptotic = selected > 1.0e3 * omega_d
        if np.any(~asymptotic):
            direct = selected[~asymptotic]
            selected_result[~asymptotic] = (
                spectral_weight
                * 3.0
                / omega_d**3
                * (
                    -omega_d
                    + 0.5
                    * direct
                    * np.log((direct + omega_d) / (direct - omega_d))
                )
            )
        if np.any(asymptotic):
            far = selected[asymptotic]
            ratio_squared = (omega_d / far) ** 2
            selected_result[asymptotic] = (
                spectral_weight
                / far**2
                * (
                    1.0
                    + 3.0 / 5.0 * ratio_squared
                    + 3.0 / 7.0 * ratio_squared**2
                    + 1.0 / 3.0 * ratio_squared**3
                )
            )
        result[above] = selected_result

    if np.any(at_cutoff):
        result[at_cutoff] = complex(
            np.inf,
            -spectral_weight * 3.0 * np.pi / (2.0 * omega_d**2),
        )

    result = np.where(omega < 0.0, np.conjugate(result), result)
    if result.ndim == 0:
        return complex(result)
    return result


def mingo_2006_dressed_surface_resolvent(
    frequency: Array | float,
    *,
    coupling: float,
    cutoff: float = 100.0,
    convention: MingoSurfaceConvention = "causal_normalized",
) -> Array | complex:
    """Return the attached surface resolvent in standard causal signs.

    After mapping the paper's force/Green-function signs to the standard
    positive-Hessian NEGFPy convention, attachment of a positive diagonal
    spring gives ``g = g0 / (1 - coupling*g0)``.  At the sharp cutoff the
    analytic limit is ``-1/coupling``.
    """

    spring = _positive_finite(coupling, name="coupling")
    omega_d = _positive_finite(cutoff, name="cutoff")
    raw = np.asarray(frequency)
    if np.iscomplexobj(raw):
        raise ValueError("frequency must be real-valued.")
    omega = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("frequency must contain only finite values.")

    at_cutoff = np.abs(omega) == omega_d
    result = np.empty(omega.shape, dtype=np.complex128)
    regular = ~at_cutoff
    if np.any(regular):
        bare = np.asarray(
            mingo_2006_bare_surface_resolvent(
                omega[regular],
                cutoff=omega_d,
                convention=convention,
            ),
            dtype=np.complex128,
        )
        result[regular] = bare / (1.0 - spring * bare)
    if np.any(at_cutoff):
        result[at_cutoff] = -1.0 / spring
    if result.ndim == 0:
        return complex(result)
    return result


def mingo_2006_harmonic_transmission(
    frequency: Array | float,
    *,
    fixture: Mingo2006DiatomicFixture | None = None,
    convention: MingoSurfaceConvention = "causal_normalized",
) -> Array | float:
    """Return the two-atom Caroli transmission without a lead regulator."""

    model = Mingo2006DiatomicFixture() if fixture is None else fixture
    if not isinstance(model, Mingo2006DiatomicFixture):
        raise TypeError("fixture must be a Mingo2006DiatomicFixture or None.")
    raw = np.asarray(frequency)
    if np.iscomplexobj(raw):
        raise ValueError("frequency must be real-valued.")
    omega = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(omega)):
        raise ValueError("frequency must contain only finite values.")

    k_surface = model.molecule_surface_solver
    k_center = model.central_bond_solver
    surface = np.asarray(
        mingo_2006_dressed_surface_resolvent(
            omega,
            coupling=k_surface,
            cutoff=model.cutoff_solver,
            convention=convention,
        ),
        dtype=np.complex128,
    )
    sigma = k_surface**2 * surface
    gamma = -2.0 * np.imag(sigma)
    diagonal = k_surface + k_center
    inverse_diagonal = omega**2 - diagonal - sigma
    determinant = inverse_diagonal**2 - k_center**2
    with np.errstate(divide="ignore", invalid="ignore"):
        off_diagonal_green = k_center / determinant
        transmission = gamma**2 * np.abs(off_diagonal_green) ** 2
    transmission = np.where(
        (omega == 0.0) | (np.abs(omega) >= model.cutoff_solver),
        0.0,
        transmission,
    )
    transmission = np.real_if_close(transmission).astype(np.float64)
    if transmission.ndim == 0:
        return float(transmission)
    return transmission


def mingo_2006_harmonic_conductance(
    temperature_kelvin: Array | float,
    *,
    fixture: Mingo2006DiatomicFixture | None = None,
    frequency_step: float = 0.5,
    convention: MingoSurfaceConvention = "causal_normalized",
) -> Array | float:
    r"""Return Eq. (51) harmonic conductance in watt/kelvin.

    The integration uses the paper's positive-frequency rectangle grid and
    excludes the logarithmic cutoff endpoint.  Bose derivatives are evaluated
    using negative exponentials, so neither a lead ``eta`` nor a large
    exponential is required.
    """

    model = Mingo2006DiatomicFixture() if fixture is None else fixture
    if not isinstance(model, Mingo2006DiatomicFixture):
        raise TypeError("fixture must be a Mingo2006DiatomicFixture or None.")
    step = _positive_finite(frequency_step, name="frequency_step")
    raw = np.asarray(temperature_kelvin)
    if np.iscomplexobj(raw):
        raise ValueError("temperature_kelvin must be real-valued.")
    temperature = np.asarray(raw, dtype=np.float64)
    if not np.all(np.isfinite(temperature)) or np.any(temperature < 0.0):
        raise ValueError("temperature_kelvin must be finite and nonnegative.")

    frequency = np.arange(
        step,
        model.cutoff_solver,
        step,
        dtype=np.float64,
    )
    transmission = np.asarray(
        mingo_2006_harmonic_transmission(
            frequency,
            fixture=model,
            convention=convention,
        )
    )
    flat_temperature = temperature.reshape(-1)
    result = np.zeros(flat_temperature.shape, dtype=np.float64)
    positive = flat_temperature > 0.0
    for index in np.flatnonzero(positive):
        selected_temperature = flat_temperature[index]
        exponent = (
            _HBAR_J_S
            * model.omega0_radian_per_second
            * frequency
            / (_BOLTZMANN_J_K * selected_temperature)
        )
        exp_negative = np.exp(-exponent)
        denominator = -np.expm1(-exponent)
        bose_derivative = (
            exponent
            / selected_temperature
            * exp_negative
            / denominator**2
        )
        result[index] = (
            _HBAR_J_S
            * model.omega0_radian_per_second**2
            / (2.0 * np.pi)
            * step
            * np.sum(frequency * transmission * bose_derivative)
        )
    result = result.reshape(temperature.shape)
    if result.ndim == 0:
        return float(result)
    return result


def mingo_2006_legacy_lesser_greater(
    vertex: DenseCubicVertex,
    green_lesser: Array,
    green_greater: Array,
    grid: SignedFrequencyGrid,
    *,
    backend: Literal["serial", "threaded"] = "serial",
    max_workers: int | None = None,
) -> tuple[Array, Array]:
    r"""Evaluate the named Mingo/Guo historical :math:`\pi/9` experiment.

    This function deliberately calls the production contraction first and
    applies the fixed, literature-audited ratio afterwards:

    ``legacy_sigma = (pi/9) * production_sigma``.
    """

    lesser, greater = cubic_lesser_greater(
        vertex,
        green_lesser,
        green_greater,
        grid,
        backend=backend,
        max_workers=max_workers,
    )
    scale = MINGO_2006_LEGACY_EQ27.cubic_bubble_scale_relative_to_guo
    return scale * lesser, scale * greater


def mingo_2006_figure4_harmonic_curve() -> Mingo2006PublishedCurve:
    """Return the vector-extracted thick solid curve from Fig. 4."""

    return Mingo2006PublishedCurve(
        figure=4,
        series="harmonic_quantum",
        temperature_kelvin=_FIGURE_TEMPERATURES,
        values=_FIGURE4_HARMONIC_PW_PER_K,
        value_units="pW/K",
        extraction_note=(
            "Coordinates extracted from the vector path in the APS PDF and "
            "mapped using the printed 0..600 K and 0..15 pW/K axes; these "
            "are plotted-curve values, not author-supplied raw data."
        ),
    )


def mingo_2006_figure8_curve(
    series: Literal["sw_scaled_quantum", "lj_quantum"],
) -> Mingo2006PublishedCurve:
    """Return a vector-extracted quantum curve from Fig. 8."""

    if series == "sw_scaled_quantum":
        values = _FIGURE8_SW_SCALED_QUANTUM
    elif series == "lj_quantum":
        values = _FIGURE8_LJ_QUANTUM
    else:
        raise ValueError(
            "series must be 'sw_scaled_quantum' or 'lj_quantum'."
        )
    return Mingo2006PublishedCurve(
        figure=8,
        series=series,
        temperature_kelvin=_FIGURE8_TEMPERATURES,
        values=values,
        value_units="dimensionless",
        extraction_note=(
            "Coordinates extracted from the vector path in the APS PDF and "
            "mapped using the printed axes. The SW values include the "
            "paper's displayed factor 13.207."
        ),
    )


_FIGURE_TEMPERATURES = _readonly_real(
    [
        8.61456,
        13.64182,
        17.14069,
        21.62481,
        27.14469,
        34.21823,
        43.09804,
        54.22623,
        68.28488,
        85.91819,
        108.16193,
        136.21608,
        171.48271,
        215.84388,
        271.73744,
        342.11911,
        430.68988,
        542.14859,
    ]
)

_FIGURE4_HARMONIC_PW_PER_K = _readonly_real(
    [
        0.0,
        0.0,
        0.004476,
        0.033771,
        0.164379,
        0.563120,
        1.444420,
        2.900233,
        4.780014,
        6.774128,
        8.588401,
        10.058048,
        11.152959,
        11.926029,
        12.448869,
        12.795937,
        13.019313,
        13.163348,
    ]
)

_FIGURE8_TEMPERATURES = _readonly_real(
    [
        8.577638,
        10.840293,
        13.614282,
        17.155277,
        21.591102,
        27.164649,
        34.157150,
        43.041589,
        54.175901,
        68.237599,
        85.929775,
        108.198398,
        136.181182,
        171.463268,
        215.859896,
        271.672064,
        342.095619,
        430.645986,
        542.155276,
    ]
)

_FIGURE8_SW_SCALED_QUANTUM = _readonly_real(
    [
        0.08443958,
        0.08267803,
        0.07338668,
        0.05868401,
        0.05326118,
        0.05387139,
        0.05645040,
        0.06126302,
        0.06654769,
        0.07176328,
        0.07894767,
        0.08599390,
        0.09677048,
        0.11133498,
        0.13146048,
        0.15802199,
        0.19291923,
        0.23831673,
        0.29693167,
    ]
)

_FIGURE8_LJ_QUANTUM = _readonly_real(
    [
        0.08511888,
        0.08328824,
        0.07474526,
        0.06140119,
        0.05658857,
        0.05746359,
        0.06112486,
        0.06607564,
        0.07054286,
        0.07603477,
        0.08416326,
        0.09188878,
        0.10428876,
        0.12170859,
        0.14603650,
        0.17957515,
        0.22232456,
        0.27802659,
        0.32451787,
    ]
)


__all__ = [
    "GUO_2020_PRODUCTION_SCBA",
    "MINGO_2006_CAUSAL_NORMALIZED_SURFACE",
    "MINGO_2006_FIGURE4_HALF_WEIGHT_SENSITIVITY",
    "MINGO_2006_LEGACY_EQ27",
    "MINGO_2006_UNVERIFIED_VERTEX_X6_SENSITIVITY",
    "Mingo2006Approximation",
    "Mingo2006DiatomicFixture",
    "Mingo2006PublishedCurve",
    "Mingo2006SurfaceInterpretation",
    "MingoSurfaceConvention",
    "mingo_2006_bare_surface_resolvent",
    "mingo_2006_dressed_surface_resolvent",
    "mingo_2006_figure4_harmonic_curve",
    "mingo_2006_figure8_curve",
    "mingo_2006_harmonic_conductance",
    "mingo_2006_harmonic_transmission",
    "mingo_2006_legacy_lesser_greater",
]
