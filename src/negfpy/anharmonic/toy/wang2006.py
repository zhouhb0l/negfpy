r"""Primary-source reproduction helpers for Wang, Wang, and Zeng (2006).

This module is intentionally separate from the production Guo bubble solver.
Wang et al. Figure 2 compares four different diagram sets:

``harmonic``
    Contact-only ballistic transport.
``wang_born_t2``
    The two leading diagrams of their Eq. (13), evaluated with ``G0``:
    a dynamic bubble and a static tadpole/thermal-expansion self-energy.
``wang_born_t4``
    The two leading diagrams plus the six fourth-order diagrams in their
    Figure 1, all evaluated with ``G0``.
``wang_selective_mean_field``
    Only the dynamic bubble is dressed self-consistently; the tadpole remains
    a ``G0`` contribution.

The production NEGFPy ``solve_one_shot_born`` and ``solve_scba`` operations
are, respectively, bubble-only one-shot Born and conserving bubble-only SCBA.
They are useful modern comparators, but neither is silently relabeled as a
Wang Figure 2 approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy.integrate import simpson
from scipy.signal import fftconvolve

from .._harmonic import (
    assemble_device_dynamical_matrix,
    solve_keldysh_green,
)
from ..contacts import build_contact_spectra
from ..current import _evaluate_current_report
from ..grid import SignedFrequencyGrid
from ..types import (
    CubicSCBAProblem,
    CurrentReport,
    KeldyshSpectrum,
)
from .historical import (
    Wang2006ThreeAtomFixture,
    wang_2006_inferred_author_legacy_figure2_fixture,
)


Array = np.ndarray

_BOLTZMANN_J_K = 1.380649e-23
_EPS_SHA256 = "0e26c34ca70f3b132af8c6b3446d792c7a1ffd8b12a486f04df0c9db7185432b"
_EPS_X_ORIGIN = 0.15
_EPS_X_KELVIN_PER_UNIT = 1000.0
_EPS_Y_ORIGIN = 0.15
_EPS_Y_NW_PER_K_PER_UNIT = 1.0 / 1.4


class Wang2006Approximation(str, Enum):
    """Named, noninterchangeable approximation levels relevant to Figure 2."""

    HARMONIC = "harmonic"
    WANG_BORN_T2 = "wang_born_t2"
    WANG_BORN_T4 = "wang_born_t4"
    WANG_SELECTIVE_MEAN_FIELD = "wang_selective_mean_field"
    GUO_BUBBLE_ONE_SHOT = "guo_bubble_one_shot"
    GUO_BUBBLE_SCBA = "guo_bubble_scba"


@dataclass(frozen=True, slots=True)
class Wang2006ApproximationSpec:
    """Auditable definition of one historical or modern comparator."""

    approximation: Wang2006Approximation
    implemented: bool
    status: str
    bubble_propagator: str | None
    includes_tadpole: bool
    includes_explicit_fourth_order: bool
    self_consistent: bool
    figure2_series: str | None
    production_operation: str | None
    conservation_contract: str


def wang_2006_approximation_specs(
) -> Mapping[Wang2006Approximation, Wang2006ApproximationSpec]:
    """Return the frozen Figure 2 approximation map.

    The entries follow the paper text, Figure 1, Eq. (13), and the archived
    author notebook.  In particular, the historical mean-field curve dresses
    only the first (bubble) graph, not every leading graph.
    """

    specs = {
        Wang2006Approximation.HARMONIC: Wang2006ApproximationSpec(
            approximation=Wang2006Approximation.HARMONIC,
            implemented=True,
            status=(
                "implemented_inferred_model_curve_reproduced_source_limited"
            ),
            bubble_propagator=None,
            includes_tadpole=False,
            includes_explicit_fourth_order=False,
            self_consistent=False,
            figure2_series="ballistic",
            production_operation="harmonic NEGF",
            conservation_contract="two-contact harmonic Landauer conservation",
        ),
        Wang2006Approximation.WANG_BORN_T2: Wang2006ApproximationSpec(
            approximation=Wang2006Approximation.WANG_BORN_T2,
            implemented=True,
            status=(
                "implemented_inferred_model_curve_reproduced_source_limited"
            ),
            bubble_propagator="G0",
            includes_tadpole=True,
            includes_explicit_fourth_order=False,
            self_consistent=False,
            figure2_series="O(T^2)",
            production_operation=None,
            conservation_contract=(
                "legacy truncated contour expansion; not a production "
                "Phi-derivable certification mode"
            ),
        ),
        Wang2006Approximation.WANG_BORN_T4: Wang2006ApproximationSpec(
            approximation=Wang2006Approximation.WANG_BORN_T4,
            implemented=False,
            status="reference_curve_only_not_implemented",
            bubble_propagator="G0",
            includes_tadpole=True,
            includes_explicit_fourth_order=True,
            self_consistent=False,
            figure2_series="O(T^4)",
            production_operation=None,
            conservation_contract=(
                "legacy fourth-order contour expansion; not implemented as "
                "a production certification mode"
            ),
        ),
        Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD: (
            Wang2006ApproximationSpec(
                approximation=Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD,
                implemented=True,
                status=(
                    "implemented_inferred_model_curve_reproduced_source_limited"
                ),
                bubble_propagator="dressed G for bubble; G0 for tadpole",
                includes_tadpole=True,
                includes_explicit_fourth_order=False,
                self_consistent=True,
                figure2_series="mean-field",
                production_operation=None,
                conservation_contract=(
                    "historical selective resummation; no NEGFPy conserving "
                    "SCBA certification claim"
                ),
            )
        ),
        Wang2006Approximation.GUO_BUBBLE_ONE_SHOT: Wang2006ApproximationSpec(
            approximation=Wang2006Approximation.GUO_BUBBLE_ONE_SHOT,
            implemented=True,
            status="implemented_production_diagnostic",
            bubble_propagator="G0",
            includes_tadpole=False,
            includes_explicit_fourth_order=False,
            self_consistent=False,
            figure2_series=None,
            production_operation="solve_one_shot_born",
            conservation_contract="diagnostic; current conservation not required",
        ),
        Wang2006Approximation.GUO_BUBBLE_SCBA: Wang2006ApproximationSpec(
            approximation=Wang2006Approximation.GUO_BUBBLE_SCBA,
            implemented=True,
            status="implemented_production_certification_path",
            bubble_propagator="dressed G",
            includes_tadpole=False,
            includes_explicit_fourth_order=False,
            self_consistent=True,
            figure2_series=None,
            production_operation="solve_scba",
            conservation_contract="conserving bubble SCBA",
        ),
    }
    return MappingProxyType(specs)


@dataclass(frozen=True, slots=True)
class Wang2006Figure2Series:
    """One curve extracted from the author-posted vector EPS."""

    approximation: Wang2006Approximation
    label: str
    temperature_kelvin: Array
    conductance_w_per_k: Array
    source_sha256: str = _EPS_SHA256

    def __post_init__(self) -> None:
        temperature = np.array(self.temperature_kelvin, dtype=float, copy=True)
        conductance = np.array(self.conductance_w_per_k, dtype=float, copy=True)
        if temperature.ndim != 1 or conductance.shape != temperature.shape:
            raise ValueError("Figure 2 reference arrays must be matching vectors.")
        if temperature.size < 2:
            raise ValueError("A Figure 2 series must contain at least two points.")
        if not (
            np.all(np.isfinite(temperature))
            and np.all(np.isfinite(conductance))
            and np.all(np.diff(temperature) > 0.0)
        ):
            raise ValueError("Figure 2 reference data must be finite and ordered.")
        temperature.setflags(write=False)
        conductance.setflags(write=False)
        object.__setattr__(self, "temperature_kelvin", temperature)
        object.__setattr__(self, "conductance_w_per_k", conductance)

    def interpolate(self, temperature_kelvin: float) -> float:
        """Linearly interpolate inside the published path's temperature span."""

        value = float(temperature_kelvin)
        if (
            not np.isfinite(value)
            or value < self.temperature_kelvin[0]
            or value > self.temperature_kelvin[-1]
        ):
            raise ValueError("temperature is outside this Figure 2 series.")
        return float(
            np.interp(
                value,
                self.temperature_kelvin,
                self.conductance_w_per_k,
            )
        )


def _eps_series(
    approximation: Wang2006Approximation,
    label: str,
    text: str,
) -> Wang2006Figure2Series:
    values = np.fromstring(text, sep=" ", dtype=float).reshape(-1, 2)
    return Wang2006Figure2Series(
        approximation=approximation,
        label=label,
        temperature_kelvin=values[:, 0],
        conductance_w_per_k=values[:, 1] * 1.0e-9,
    )


def wang_2006_figure2_reference(
) -> Mapping[Wang2006Approximation, Wang2006Figure2Series]:
    """Return machine-readable paths from the primary-source Figure 2 EPS.

    The Grace EPS uses the plot rectangle ``x=0.15..1.15`` for
    ``0..1000 K`` and ``y=0.15..0.85`` for ``0..0.5 nW/K``.  The values below
    apply those affine maps to the vector path coordinates; they are not
    raster digitization.  The EPS stores four decimal places, corresponding
    to coordinate quantization of roughly ``0.05 K`` and
    ``3.6e-14 W/K``.  That graphical precision is not a claim about the
    underlying numerical integration error.
    """

    ballistic = _eps_series(
        Wang2006Approximation.HARMONIC,
        "ballistic",
        """
        5 0 25.2 .004928571429 45.3 .02014285714 65.5 .03807142857
        85.6 .05664285714 105.8 .0755 125.9 .09435714286
        146.1 .1133571429 166.2 .1322142857 186.4 .1508571429
        206.5 .1692142857 226.7 .1871428571 246.8 .2044285714
        267 .2210714286 287.1 .2368571429 307.3 .2517857143
        327.4 .2659285714 347.6 .2790714286 367.7 .2914285714
        387.9 .303 408 .3137142857 428.2 .3236428571
        448.3 .3328571429 468.5 .3414285714 488.6 .3494285714
        508.8 .3567857143 528.9 .3636428571 549.1 .37
        569.2 .3758571429 589.4 .3813571429 609.5 .3865
        629.7 .3912142857 649.8 .3956428571 670 .3997857143
        690.2 .4036428571 710.3 .4072857143 730.5 .4106428571
        750.6 .4137857143 770.8 .4167857143 790.9 .4195
        811.1 .4221428571 831.2 .4245714286 851.4 .4269285714
        871.5 .4290714286 891.7 .4311428571 911.8 .4330714286
        932 .4348571429 952.1 .4366428571 972.3 .4382142857
        992.4 .4397857143 1000.1 .4403571429
        """,
    )
    born_t2 = _eps_series(
        Wang2006Approximation.WANG_BORN_T2,
        "O(T^2)",
        """
        10 .00007142857143 30.9 .008785714286 51.8 .02592857143
        72.7 .04478571429 93.6 .064 114.5 .08328571429
        135.4 .1025714286 156.3 .1217142857 177.2 .1403571429
        198.1 .1584285714 219 .1757142857 239.9 .192
        260.8 .2072857143 281.7 .2214285714 302.6 .2344285714
        323.5 .2462857143 344.4 .257 365.3 .2667142857
        386.2 .2754285714 407.1 .2832142857 428 .2901428571
        448.9 .2962857143 469.8 .3017142857 490.7 .3065
        511.6 .3107142857 532.5 .3142857143 553.4 .3174285714
        574.3 .3201428571 595.2 .3223571429 616.1 .3242142857
        637 .3257857143 657.9 .327 678.8 .3278571429
        699.7 .3285714286 720.6 .3289285714 741.5 .3291428571
        762.4 .3291428571 783.3 .3289285714 804.2 .3285714286
        825.1 .328 846 .3272857143 866.9 .3264285714
        887.8 .3255 908.7 .3244285714 929.6 .3232142857
        950.5 .3219285714 971.4 .3205 992.3 .319
        1000.1 .3184285714
        """,
    )
    born_t4 = _eps_series(
        Wang2006Approximation.WANG_BORN_T4,
        "O(T^4)",
        """
        72.6 .04435714286 135.3 .1013571429 197.9 .1554285714
        260.5 .2013571429 323.2 .2361428571 385.8 .2602142857
        448.4 .275 511.1 .2825 573.7 .2842142857
        636.3 .2807857143 698.9 .2737857143 761.6 .2629285714
        824.2 .2502857143 886.8 .2334285714 949.5 .2145
        """,
    )
    mean_field = _eps_series(
        Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD,
        "mean-field",
        """
        15.1 .0006428571429 25.1 .004928571429 35.2 .01192857143
        45.2 .02014285714 55.3 .02892857143 65.3 .03792857143
        75.4 .04707142857 85.4 .05628571429 95.5 .06557142857
        105.5 .07478571429 115.6 .08407142857 125.6 .09328571429
        135.7 .1025 145.7 .1115714286 155.8 .1206428571
        165.8 .1295714286 175.9 .1382857143 185.9 .1469285714
        196 .1553571429 206 .1635714286 216.1 .1715
        226.1 .1792142857 236.2 .1867142857 246.2 .1938571429
        256.3 .2007142857 266.3 .2072142857 276.4 .2135
        286.4 .2193571429 296.5 .225 306.5 .2302857143
        316.6 .2352142857 326.6 .2398571429 336.7 .2441428571
        346.7 .2481428571 356.8 .2517857143 366.8 .2552142857
        376.9 .2582142857 386.9 .2609285714 397 .2633571429
        407 .2655 417.1 .2672857143 427.1 .2687142857
        437.2 .2697857143 447.2 .2705714286 457.3 .271
        467.3 .271 477.4 .2706428571 487.4 .2697857143
        497.5 .2683571429 507.5 .2664285714 517.6 .2637142857
        527.6 .2601428571 537.7 .2554285714 547.7 .2489285714
        557.8 .2395714286 567.8 .2241428571
        """,
    )
    return MappingProxyType(
        {
            item.approximation: item
            for item in (ballistic, born_t2, born_t4, mean_field)
        }
    )


def wang_2006_author_odd_grid() -> SignedFrequencyGrid:
    """Return the odd-grid counterpart of the archived notebook grid.

    The notebook requests ``makeGlistSymmetrize[20.5, 1000]``, an even grid
    in its legacy frequency unit.  NEGFPy's numerical contract requires an
    exact zero and an odd point count, so this helper keeps the same converted
    cutoff with 1001 points.  It is an explicitly documented near-reproduction
    grid, not a claim of bitwise equivalence to Mathematica's rounding.
    """

    fixture = wang_2006_inferred_author_legacy_figure2_fixture()
    frequency_factor = float(
        fixture.scale.metadata["legacy_frequency_to_solver"]
    )
    cutoff = 20.5 * frequency_factor
    return SignedFrequencyGrid(step=cutoff / 500.0, n_positive=500)


def _require_inferred_author_legacy_fixture(
    fixture: Wang2006ThreeAtomFixture,
) -> None:
    if not isinstance(fixture, Wang2006ThreeAtomFixture):
        raise TypeError("fixture must be a Wang2006ThreeAtomFixture.")
    variant = fixture.scale.metadata.get("model_variant")
    if variant != "inferred_author_legacy_fig2":
        raise ValueError(
            "Figure 2 reproduction requires the explicit "
            "inferred_author_legacy_fig2 fixture; the caption-literal "
            "fixture is a different physical model."
        )


def _scalar_surface_green(
    omega: Array,
    d00: float,
    hopping: float,
    eta: float,
) -> Array:
    z = (np.asarray(omega, dtype=float) + 1j * float(eta)) ** 2
    delta = z - float(d00)
    hopping_square = float(hopping) ** 2
    discriminant = np.sqrt(delta**2 - 4.0 * hopping_square)
    first = (delta - discriminant) / (2.0 * hopping_square)
    second = (delta + discriminant) / (2.0 * hopping_square)
    choose_first = np.imag(first) < np.imag(second)
    equal_imaginary = np.isclose(
        np.imag(first),
        np.imag(second),
        rtol=0.0,
        atol=32.0 * np.finfo(float).eps,
    )
    choose_first = np.where(
        equal_imaginary,
        np.abs(first) <= np.abs(second),
        choose_first,
    )
    return np.where(choose_first, first, second)


def wang_2006_harmonic_transmission(
    fixture: Wang2006ThreeAtomFixture,
    omega: Array,
    *,
    eta_lead: float = 1.0e-6,
    eta_device: float = 0.0,
) -> Array:
    """Vectorized scalar-lead Caroli transmission for the Figure 2 model."""

    _require_inferred_author_legacy_fixture(fixture)
    frequencies = np.asarray(omega, dtype=float)
    if frequencies.ndim != 1 or np.any(frequencies < 0.0):
        raise ValueError("omega must be a non-negative one-dimensional array.")
    if (
        not np.isfinite(eta_lead)
        or eta_lead <= 0.0
        or not np.isfinite(eta_device)
        or eta_device < 0.0
    ):
        raise ValueError("eta values must be finite with eta_lead > 0.")

    left_d00 = float(np.real(fixture.left_lead.d00[0, 0]))
    right_d00 = float(np.real(fixture.right_lead.d00[0, 0]))
    left_hopping = abs(float(np.real(fixture.left_lead.d01[0, 0])))
    right_hopping = abs(float(np.real(fixture.right_lead.d01[0, 0])))
    left_surface = _scalar_surface_green(
        frequencies,
        left_d00,
        left_hopping,
        eta_lead,
    )
    right_surface = _scalar_surface_green(
        frequencies,
        right_d00,
        right_hopping,
        eta_lead,
    )
    sigma_left = left_hopping**2 * left_surface
    sigma_right = right_hopping**2 * right_surface

    dynamical = assemble_device_dynamical_matrix(fixture.device)
    identity = np.eye(3, dtype=np.complex128)
    operator = (
        (frequencies + 1j * eta_device)[:, None, None] ** 2 * identity
        - dynamical[None, :, :]
    )
    operator[:, 0, 0] -= sigma_left
    operator[:, 2, 2] -= sigma_right
    green = np.linalg.inv(operator)
    gamma_left = -2.0 * np.imag(sigma_left)
    gamma_right = -2.0 * np.imag(sigma_right)
    transmission = (
        gamma_left
        * gamma_right
        * np.abs(green[:, 0, 2]) ** 2
    )
    return np.maximum(np.real(transmission), 0.0)


def wang_2006_harmonic_conductance(
    temperature_kelvin: Array | float,
    *,
    fixture: Wang2006ThreeAtomFixture | None = None,
    n_frequency: int = 6001,
    eta_lead: float = 1.0e-6,
    eta_device: float = 0.0,
) -> Array | float:
    """Return differential harmonic conductance in W/K.

    This is an independent Landauer calculation used for conditional
    curve-level reproduction with the explicitly inferred Figure 2 model. It
    is not the historical interacting quadrature or a unique input recovery.
    """

    selected = (
        wang_2006_inferred_author_legacy_figure2_fixture()
        if fixture is None
        else fixture
    )
    _require_inferred_author_legacy_fixture(selected)
    if (
        isinstance(n_frequency, (bool, np.bool_))
        or not isinstance(n_frequency, (int, np.integer))
        or int(n_frequency) < 257
    ):
        raise ValueError("n_frequency must be an integer of at least 257.")
    temperatures = np.asarray(temperature_kelvin, dtype=float)
    if np.any(~np.isfinite(temperatures)) or np.any(temperatures < 0.0):
        raise ValueError("temperatures must be finite and non-negative.")

    maximum = max(
        float(
            np.sqrt(
                fixture_lead.d00[0, 0].real
                + 2.0 * abs(fixture_lead.d01[0, 0])
            )
        )
        for fixture_lead in (selected.left_lead, selected.right_lead)
    )
    frequencies = np.linspace(
        max(1.0e-9, maximum / (int(n_frequency) * 20.0)),
        maximum * 1.002,
        int(n_frequency),
    )
    transmission = wang_2006_harmonic_transmission(
        selected,
        frequencies,
        eta_lead=eta_lead,
        eta_device=eta_device,
    )
    flattened = temperatures.reshape(-1)
    result = np.zeros(flattened.shape, dtype=float)
    for index, temperature in enumerate(flattened):
        if temperature == 0.0:
            continue
        reduced_temperature = selected.scale.dimensionless_temperature(
            float(temperature),
            boltzmann_constant=_BOLTZMANN_J_K,
        )
        ratio = frequencies / reduced_temperature
        exponential = np.exp(-ratio)
        derivative = (
            frequencies
            / reduced_temperature**2
            * exponential
            / (1.0 - exponential) ** 2
        )
        result[index] = (
            _BOLTZMANN_J_K
            * selected.scale.omega0
            * simpson(
                frequencies * transmission * derivative,
                x=frequencies,
            )
            / (2.0 * np.pi)
        )
    result = result.reshape(temperatures.shape)
    if np.isscalar(temperature_kelvin):
        return float(result)
    return result


def _zero_extended_bubble(
    vertex: Array,
    green: Array,
    grid: SignedFrequencyGrid,
    *,
    convolution: str,
) -> Array:
    """Evaluate a Guo-normalized bubble for one contour component."""

    if convolution not in {"direct", "fft"}:
        raise ValueError("convolution must be 'direct' or 'fft'.")
    values = np.asarray(green, dtype=np.complex128)
    tensor = np.asarray(vertex, dtype=float)
    expected = (grid.n_points, tensor.shape[0], tensor.shape[0])
    if values.shape != expected:
        raise ValueError(f"green must have shape {expected}.")
    result = np.zeros_like(values)
    nonzero = [np.argwhere(tensor[output] != 0.0) for output in range(tensor.shape[0])]
    convolve = np.convolve if convolution == "direct" else fftconvolve
    for first in range(tensor.shape[0]):
        for second in range(tensor.shape[0]):
            accumulated = np.zeros(2 * grid.n_points - 1, dtype=np.complex128)
            for c, d in nonzero[first]:
                for e, f in nonzero[second]:
                    accumulated += (
                        tensor[first, c, d]
                        * tensor[second, e, f]
                        * convolve(
                            values[:, c, e],
                            values[:, d, f],
                            mode="full",
                        )
                    )
            start = grid.zero_index
            result[:, first, second] = (
                0.5j
                * grid.integration_weight
                * accumulated[start : start + grid.n_points]
            )
    return result


def _legacy_retarded_green(
    problem: CubicSCBAProblem,
    sigma_left: Array,
    sigma_right: Array,
    sigma_pp: Array,
    *,
    eta_device: float,
) -> Array:
    dynamical = assemble_device_dynamical_matrix(problem.device)
    identity = np.eye(problem.ndof, dtype=np.complex128)
    result = np.empty_like(sigma_pp)
    for index, omega in enumerate(problem.grid.values):
        operator = (
            (float(omega) + 1j * eta_device) ** 2 * identity
            - dynamical
            - sigma_left[index]
            - sigma_right[index]
            - sigma_pp[index]
        )
        result[index] = np.linalg.solve(operator, identity)
    return result


def _symmetrized_contour_components(green: KeldyshSpectrum) -> tuple[Array, ...]:
    advanced = np.swapaxes(green.retarded.conj(), -1, -2)
    components = np.stack(
        (
            np.stack((green.lesser + green.retarded, green.lesser)),
            np.stack((green.greater, green.lesser - advanced)),
        )
    )
    result = np.empty_like(components)
    for first in range(2):
        for second in range(2):
            negative_partner = np.transpose(
                components[second, first, ::-1],
                (0, 2, 1),
            )
            result[first, second] = 0.5 * (
                components[first, second] + negative_partner
            )
    return (
        result[0, 0],
        result[0, 1],
        result[1, 0],
        result[1, 1],
    )


def wang_2006_static_tadpole(
    problem: CubicSCBAProblem,
    green_zero: KeldyshSpectrum,
) -> Array:
    """Return the real, frequency-independent second term of Wang Eq. (13)."""

    time_ordered, lesser, _, anti_time_ordered = (
        _symmetrized_contour_components(green_zero)
    )
    paper_tensor = problem.vertex.values / 2.0
    zero = problem.grid.zero_index
    integrated_time_ordered = (
        problem.grid.integration_weight * np.sum(time_ordered, axis=0)
    )
    integrated_anti_time_ordered = (
        problem.grid.integration_weight * np.sum(anti_time_ordered, axis=0)
    )
    plus = np.einsum(
        "jkl,lm,mrs,rs->jk",
        paper_tensor,
        time_ordered[zero],
        paper_tensor,
        integrated_time_ordered,
        optimize=True,
    )
    minus = np.einsum(
        "jkl,lm,mrs,rs->jk",
        paper_tensor,
        lesser[zero],
        paper_tensor,
        integrated_anti_time_ordered,
        optimize=True,
    )
    # The author notebook explicitly takes Re after the contour contraction;
    # the analytic static term is real.  This also removes finite-grid/eta
    # contour noise without modifying the dynamic bubble.
    return np.asarray(np.real(2.0j * (plus - minus)), dtype=np.complex128)


@dataclass(frozen=True, slots=True)
class Wang2006BornT2Result:
    """One explicitly legacy, non-certifying Wang ``O(T^2)`` calculation."""

    problem: CubicSCBAProblem
    green: KeldyshSpectrum
    sigma_pp: KeldyshSpectrum
    sigma_left: KeldyshSpectrum
    sigma_right: KeldyshSpectrum
    currents: CurrentReport
    through_conductance_w_per_k: float
    metadata: Mapping[str, object]
    release_certified: bool = False

    def __post_init__(self) -> None:
        if self.release_certified:
            raise ValueError(
                "Historical Wang O(T^2) results cannot be release-certified."
            )


@dataclass(frozen=True, slots=True)
class Wang2006MeanFieldIteration:
    """One selective mean-field fixed-point record."""

    iteration: int
    mixing: float
    sigma_residual: float
    green_residual: float
    contact_balance_w: float
    collision_power_w: float
    passing: bool
    consecutive_passes: int


@dataclass(frozen=True, slots=True)
class Wang2006SelectiveMeanFieldResult:
    """Historical selective mean-field result, never production-certified."""

    problem: CubicSCBAProblem
    green: KeldyshSpectrum
    sigma_pp: KeldyshSpectrum
    sigma_left: KeldyshSpectrum
    sigma_right: KeldyshSpectrum
    currents: CurrentReport
    through_conductance_w_per_k: float
    static_tadpole: Array
    history: tuple[Wang2006MeanFieldIteration, ...]
    converged: bool
    metadata: Mapping[str, object]
    release_certified: bool = False

    def __post_init__(self) -> None:
        if self.release_certified:
            raise ValueError(
                "Historical Wang mean-field results cannot be "
                "release-certified."
            )
        tadpole = np.array(
            self.static_tadpole,
            dtype=np.complex128,
            order="C",
            copy=True,
        )
        expected = (self.problem.ndof, self.problem.ndof)
        if tadpole.shape != expected:
            raise ValueError(
                f"static_tadpole must have shape {expected}, got "
                f"{tadpole.shape}."
            )
        if not (
            np.all(np.isfinite(tadpole.real))
            and np.all(np.isfinite(tadpole.imag))
        ):
            raise ValueError("static_tadpole must contain only finite values.")
        tadpole.setflags(write=False)
        object.__setattr__(self, "static_tadpole", tadpole)


class Wang2006MeanFieldConvergenceError(RuntimeError):
    """Raised with the last complete historical mean-field result."""

    def __init__(self, result: Wang2006SelectiveMeanFieldResult):
        self.partial_result = result
        last = result.history[-1] if result.history else None
        detail = (
            "without an iteration record"
            if last is None
            else (
                f"at iteration {last.iteration}: "
                f"sigma residual={last.sigma_residual:.3e}, "
                f"Green residual={last.green_residual:.3e}"
            )
        )
        super().__init__(
            "Wang 2006 selective mean-field iteration did not converge "
            + detail
        )


def _relative_field_residual(candidate: Array, current: Array) -> float:
    difference = float(np.max(np.abs(candidate - current)))
    scale = max(
        float(np.max(np.abs(candidate))),
        float(np.max(np.abs(current))),
        1.0e-14,
    )
    return difference / scale


def _wang_result_metadata(
    *,
    method: Wang2006Approximation,
    eta_lead: float,
    eta_device: float,
    temperature_bias_kelvin: float,
    convolution: str,
    extra: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    values: dict[str, object] = {
        "method": method.value,
        "dynamic_retarded_reconstruction": "Sigma++-Sigma+-",
        "static_tadpole": True,
        "convolution": convolution,
        "eta_lead": float(eta_lead),
        "eta_device": float(eta_device),
        "eta_device_is_unaccounted_legacy_regulator": bool(
            eta_device > 0.0
        ),
        "temperature_bias_kelvin": float(temperature_bias_kelvin),
        "current_combination": "(J_left-J_right)/2",
        "release_certified": False,
        "figure2_eps_sha256": _EPS_SHA256,
    }
    if extra is not None:
        values.update(extra)
    return MappingProxyType(values)


def _validated_temperature_bias(
    problem: CubicSCBAProblem,
    temperature_bias_kelvin: float,
) -> float:
    """Validate the declared bias against the actual reservoir state."""

    if (
        not np.isfinite(temperature_bias_kelvin)
        or temperature_bias_kelvin <= 0.0
    ):
        raise ValueError("temperature_bias_kelvin must be positive.")
    solver_bias = float(problem.left.temperature - problem.right.temperature)
    if solver_bias <= 0.0:
        raise ValueError(
            "Wang Figure 2 conductance requires the left reservoir to be "
            "strictly hotter than the right reservoir."
        )
    actual_kelvin = (
        solver_bias
        * problem.scale.hbar
        * problem.scale.omega0
        / _BOLTZMANN_J_K
    )
    if not np.isclose(
        actual_kelvin,
        float(temperature_bias_kelvin),
        rtol=1.0e-10,
        atol=1.0e-10,
    ):
        raise ValueError(
            "temperature_bias_kelvin does not match the scale-converted "
            f"reservoir difference (declared {temperature_bias_kelvin:.12g} K, "
            f"actual {actual_kelvin:.12g} K)."
        )
    return float(actual_kelvin)


def solve_wang_2006_born_t2(
    problem: CubicSCBAProblem,
    *,
    temperature_bias_kelvin: float,
    eta_lead: float,
    eta_device: float | None = None,
    convolution: str = "fft",
) -> Wang2006BornT2Result:
    r"""Evaluate both leading diagrams using Wang's contour prescription.

    The dynamic retarded bubble is reconstructed as
    ``Sigma++ - Sigma+-``, not by NEGFPy's finite-grid Hilbert transform.  The
    static tadpole is then added to the retarded component.  This function is
    a historical validation layer and always returns
    ``release_certified=False``.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if (
        problem.scale.metadata.get("model_variant")
        != "inferred_author_legacy_fig2"
    ):
        raise ValueError(
            "Wang O(T^2) reproduction requires model_variant="
            "'inferred_author_legacy_fig2'."
        )
    validated_bias = _validated_temperature_bias(
        problem,
        temperature_bias_kelvin,
    )
    if not np.isfinite(eta_lead) or eta_lead <= 0.0:
        raise ValueError("eta_lead must be positive.")
    selected_eta_device = eta_lead if eta_device is None else float(eta_device)
    if not np.isfinite(selected_eta_device) or selected_eta_device < 0.0:
        raise ValueError("eta_device must be finite and non-negative.")

    left, right = build_contact_spectra(problem, eta_lead)
    zero = np.zeros(
        (problem.grid.n_points, problem.ndof, problem.ndof),
        dtype=np.complex128,
    )
    green_zero_retarded = _legacy_retarded_green(
        problem,
        left.retarded,
        right.retarded,
        zero,
        eta_device=selected_eta_device,
    )
    green_zero = solve_keldysh_green(
        green_zero_retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )
    time_ordered, lesser, greater, _ = _symmetrized_contour_components(
        green_zero
    )
    sigma_lesser = _zero_extended_bubble(
        problem.vertex.values,
        lesser,
        problem.grid,
        convolution=convolution,
    )
    sigma_greater = _zero_extended_bubble(
        problem.vertex.values,
        greater,
        problem.grid,
        convolution=convolution,
    )
    sigma_time_ordered = _zero_extended_bubble(
        problem.vertex.values,
        time_ordered,
        problem.grid,
        convolution=convolution,
    )
    tadpole = wang_2006_static_tadpole(problem, green_zero)
    sigma_retarded = (
        sigma_time_ordered
        - sigma_lesser
        + tadpole[None, :, :]
    )
    sigma_pp = KeldyshSpectrum(
        retarded=sigma_retarded,
        lesser=sigma_lesser,
        greater=sigma_greater,
    )
    green_retarded = _legacy_retarded_green(
        problem,
        left.retarded,
        right.retarded,
        sigma_retarded,
        eta_device=selected_eta_device,
    )
    green = solve_keldysh_green(
        green_retarded,
        left.lesser + right.lesser + sigma_lesser,
        left.greater + right.greater + sigma_greater,
    )
    currents = _evaluate_current_report(
        problem,
        left,
        right,
        sigma_pp,
        green,
    )
    conductance = (
        (currents.left - currents.right)
        / 2.0
        / validated_bias
    )
    return Wang2006BornT2Result(
        problem=problem,
        green=green,
        sigma_pp=sigma_pp,
        sigma_left=left,
        sigma_right=right,
        currents=currents,
        through_conductance_w_per_k=float(conductance),
        metadata=_wang_result_metadata(
            method=Wang2006Approximation.WANG_BORN_T2,
            eta_lead=eta_lead,
            eta_device=selected_eta_device,
            temperature_bias_kelvin=validated_bias,
            convolution=convolution,
            extra={"bubble_propagator": "G0"},
        ),
    )


def solve_wang_2006_selective_mean_field(
    problem: CubicSCBAProblem,
    *,
    temperature_bias_kelvin: float,
    eta_lead: float,
    eta_device: float | None = None,
    convolution: str = "fft",
    max_iterations: int = 200,
    min_iterations: int = 3,
    mixing: float = 0.5,
    mixing_floor: float = 0.01,
    sigma_rtol: float = 1.0e-8,
    green_rtol: float = 1.0e-8,
    consecutive_passes: int = 3,
    on_nonconvergence: str = "raise",
) -> Wang2006SelectiveMeanFieldResult:
    r"""Solve Wang's selective mean-field approximation.

    The static second diagram of Wang Eq. (13) is evaluated once from
    contact-only ``G0`` and then frozen.  Only the first, dynamic bubble is
    reevaluated from the full contour Green function.  Lesser, greater, and
    time-ordered bubble components are mixed with the same coefficient;
    the dynamic retarded component is reconstructed after mixing as
    ``Sigma++ - Sigma+-``.

    This historical resummation is not Guo bubble-only SCBA and is never
    release-certified.  A finite ``eta_device`` reproduces the legacy
    notebook but represents an unaccounted bath, another reason this path is
    validation-only.
    """

    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if (
        problem.scale.metadata.get("model_variant")
        != "inferred_author_legacy_fig2"
    ):
        raise ValueError(
            "Wang selective mean-field reproduction requires "
            "model_variant='inferred_author_legacy_fig2'."
        )
    validated_bias = _validated_temperature_bias(
        problem,
        temperature_bias_kelvin,
    )
    if not np.isfinite(eta_lead) or eta_lead <= 0.0:
        raise ValueError("eta_lead must be positive.")
    selected_eta_device = eta_lead if eta_device is None else float(eta_device)
    if not np.isfinite(selected_eta_device) or selected_eta_device < 0.0:
        raise ValueError("eta_device must be finite and non-negative.")
    if convolution not in {"direct", "fft"}:
        raise ValueError("convolution must be 'direct' or 'fft'.")
    for name, value, lower in (
        ("max_iterations", max_iterations, 1),
        ("min_iterations", min_iterations, 1),
        ("consecutive_passes", consecutive_passes, 1),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < lower
        ):
            raise ValueError(f"{name} must be an integer of at least {lower}.")
    max_iterations = int(max_iterations)
    min_iterations = int(min_iterations)
    consecutive_passes = int(consecutive_passes)
    if min_iterations > max_iterations:
        raise ValueError("min_iterations cannot exceed max_iterations.")
    for name, value in (
        ("mixing", mixing),
        ("mixing_floor", mixing_floor),
    ):
        if not np.isfinite(value) or not 0.0 < float(value) <= 1.0:
            raise ValueError(f"{name} must be in (0, 1].")
    if mixing_floor > mixing:
        raise ValueError("mixing_floor cannot exceed mixing.")
    for name, value in (
        ("sigma_rtol", sigma_rtol),
        ("green_rtol", green_rtol),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive.")
    if on_nonconvergence not in {"raise", "return"}:
        raise ValueError("on_nonconvergence must be 'raise' or 'return'.")

    left, right = build_contact_spectra(problem, eta_lead)
    shape = (problem.grid.n_points, problem.ndof, problem.ndof)
    zero = np.zeros(shape, dtype=np.complex128)
    green_zero_retarded = _legacy_retarded_green(
        problem,
        left.retarded,
        right.retarded,
        zero,
        eta_device=selected_eta_device,
    )
    green_zero = solve_keldysh_green(
        green_zero_retarded,
        left.lesser + right.lesser,
        left.greater + right.greater,
    )
    tadpole = wang_2006_static_tadpole(problem, green_zero)

    # The first raw bubble is built from G0, matching the historical
    # perturbative starting point. The state itself starts at zero so mixing
    # semantics are explicit and reproducible.
    current_green = green_zero
    sigma_lesser = np.zeros(shape, dtype=np.complex128)
    sigma_greater = np.zeros(shape, dtype=np.complex128)
    sigma_time_ordered = np.zeros(shape, dtype=np.complex128)
    history: list[Wang2006MeanFieldIteration] = []
    residual_sequence: list[float] = []
    selected_mixing = float(mixing)
    consecutive = 0
    final_sigma = KeldyshSpectrum(
        retarded=np.broadcast_to(tadpole, shape).copy(),
        lesser=sigma_lesser,
        greater=sigma_greater,
    )
    final_currents = _evaluate_current_report(
        problem,
        left,
        right,
        final_sigma,
        current_green,
    )

    for iteration in range(1, max_iterations + 1):
        time_ordered, lesser, greater, _ = _symmetrized_contour_components(
            current_green
        )
        raw_lesser = _zero_extended_bubble(
            problem.vertex.values,
            lesser,
            problem.grid,
            convolution=convolution,
        )
        raw_greater = _zero_extended_bubble(
            problem.vertex.values,
            greater,
            problem.grid,
            convolution=convolution,
        )
        raw_time_ordered = _zero_extended_bubble(
            problem.vertex.values,
            time_ordered,
            problem.grid,
            convolution=convolution,
        )
        raw_retarded = raw_time_ordered - raw_lesser
        current_retarded = sigma_time_ordered - sigma_lesser
        sigma_residual = max(
            _relative_field_residual(raw_lesser, sigma_lesser),
            _relative_field_residual(raw_greater, sigma_greater),
            _relative_field_residual(raw_retarded, current_retarded),
        )
        residual_sequence.append(sigma_residual)
        if (
            len(residual_sequence) >= 3
            and residual_sequence[-1] > residual_sequence[-2]
            and residual_sequence[-2] > residual_sequence[-3]
        ):
            selected_mixing = max(
                float(mixing_floor),
                0.5 * selected_mixing,
            )

        sigma_lesser = (
            (1.0 - selected_mixing) * sigma_lesser
            + selected_mixing * raw_lesser
        )
        sigma_greater = (
            (1.0 - selected_mixing) * sigma_greater
            + selected_mixing * raw_greater
        )
        sigma_time_ordered = (
            (1.0 - selected_mixing) * sigma_time_ordered
            + selected_mixing * raw_time_ordered
        )
        dynamic_retarded = sigma_time_ordered - sigma_lesser
        sigma_retarded = dynamic_retarded + tadpole[None, :, :]
        final_sigma = KeldyshSpectrum(
            retarded=sigma_retarded,
            lesser=sigma_lesser,
            greater=sigma_greater,
        )
        green_retarded = _legacy_retarded_green(
            problem,
            left.retarded,
            right.retarded,
            sigma_retarded,
            eta_device=selected_eta_device,
        )
        next_green = solve_keldysh_green(
            green_retarded,
            left.lesser + right.lesser + sigma_lesser,
            left.greater + right.greater + sigma_greater,
        )
        green_residual = max(
            _relative_field_residual(
                next_green.retarded,
                current_green.retarded,
            ),
            _relative_field_residual(
                next_green.lesser,
                current_green.lesser,
            ),
            _relative_field_residual(
                next_green.greater,
                current_green.greater,
            ),
        )
        current_green = next_green
        final_currents = _evaluate_current_report(
            problem,
            left,
            right,
            final_sigma,
            current_green,
        )
        passing = bool(
            iteration >= min_iterations
            and sigma_residual <= sigma_rtol
            and green_residual <= green_rtol
        )
        consecutive = consecutive + 1 if passing else 0
        history.append(
            Wang2006MeanFieldIteration(
                iteration=iteration,
                mixing=selected_mixing,
                sigma_residual=float(sigma_residual),
                green_residual=float(green_residual),
                contact_balance_w=float(final_currents.contact_balance),
                collision_power_w=float(final_currents.collision_power),
                passing=passing,
                consecutive_passes=consecutive,
            )
        )
        if consecutive >= consecutive_passes:
            break

    converged = consecutive >= consecutive_passes
    conductance = (
        (final_currents.left - final_currents.right)
        / 2.0
        / validated_bias
    )
    result = Wang2006SelectiveMeanFieldResult(
        problem=problem,
        green=current_green,
        sigma_pp=final_sigma,
        sigma_left=left,
        sigma_right=right,
        currents=final_currents,
        through_conductance_w_per_k=float(conductance),
        static_tadpole=np.array(tadpole, copy=True),
        history=tuple(history),
        converged=converged,
        metadata=_wang_result_metadata(
            method=Wang2006Approximation.WANG_SELECTIVE_MEAN_FIELD,
            eta_lead=eta_lead,
            eta_device=selected_eta_device,
            temperature_bias_kelvin=validated_bias,
            convolution=convolution,
            extra={
                "bubble_propagator": "dressed G",
                "tadpole_propagator": "fixed G0",
                "mixing_initial": float(mixing),
                "mixing_final": float(selected_mixing),
                "mixing_floor": float(mixing_floor),
                "sigma_rtol": float(sigma_rtol),
                "green_rtol": float(green_rtol),
                "max_iterations": max_iterations,
                "consecutive_passes": consecutive_passes,
                "converged": converged,
                "historical_approximation": (
                    "selective bubble dressing, not Guo SCBA"
                ),
            },
        ),
    )
    if not converged and on_nonconvergence == "raise":
        raise Wang2006MeanFieldConvergenceError(result)
    return result


__all__ = [
    "Wang2006Approximation",
    "Wang2006ApproximationSpec",
    "Wang2006BornT2Result",
    "Wang2006Figure2Series",
    "Wang2006MeanFieldConvergenceError",
    "Wang2006MeanFieldIteration",
    "Wang2006SelectiveMeanFieldResult",
    "solve_wang_2006_born_t2",
    "solve_wang_2006_selective_mean_field",
    "wang_2006_approximation_specs",
    "wang_2006_author_odd_grid",
    "wang_2006_figure2_reference",
    "wang_2006_harmonic_conductance",
    "wang_2006_harmonic_transmission",
    "wang_2006_static_tadpole",
]
