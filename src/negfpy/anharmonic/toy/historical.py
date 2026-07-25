r"""Historical toy-model fixtures with explicit convention conversions.

The fixtures in this module encode model inputs documented by primary sources
or, where named explicitly, an inference tested against primary-source vector
curves. They do not, by themselves, constitute reproduced paper results.

Wang et al.'s published Figure 2 needs two deliberately separate fixtures.
``wang_2006_caption_literal_fixture`` follows the final caption literally.
``wang_2006_inferred_author_legacy_figure2_fixture`` converts the three main
springs from the legacy values retained in the author TeX and tests the
additional hypothesis that the two onsite numbers also used that legacy unit.
The onsite unit assignment is inferred from agreement with the vector EPS; it
is not explicitly archived and is not a unique recovery of the source input.
Neither function silently substitutes one interpretation for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks

from ..grid import SignedFrequencyGrid
from ..types import (
    CubicSCBAProblem,
    DenseCubicVertex,
    HarmonicReservoir,
    SolverScale,
)


Array = np.ndarray

# SI definitions.  The electron volt and angstrom are exact; the atomic mass
# and reduced Planck constant use the CODATA values underlying SciPy 1.15.
_ELECTRON_VOLT_J = 1.602176634e-19
_ANGSTROM_M = 1.0e-10
_ATOMIC_MASS_KG = 1.66053906892e-27
_HBAR_J_S = 1.0545718176461565e-34
_BOLTZMANN_J_K = 1.380649e-23

_DYNAMICAL_UNIT_SI = (
    _ELECTRON_VOLT_J / (_ANGSTROM_M**2 * _ATOMIC_MASS_KG)
)
_CUBIC_UNIT_SI = (
    _ELECTRON_VOLT_J / (_ANGSTROM_M**3 * _ATOMIC_MASS_KG**1.5)
)
_LEGACY_DYNAMICAL_TO_EV_ANGSTROM2 = (
    10.0 / (_ELECTRON_VOLT_J / _ANGSTROM_M**2)
)
_LEGACY_FREQUENCY_TO_SOLVER = np.sqrt(
    _LEGACY_DYNAMICAL_TO_EV_ANGSTROM2
)


@dataclass(frozen=True, slots=True)
class Wang2006ThreeAtomFixture:
    """Mass-normalized three-atom FPU-alpha model from Wang et al. (2006).

    The stored harmonic matrices and cubic vertex are dimensionless solver
    inputs.  Temperatures are deliberately not fixed: the paper plots a
    temperature sweep but does not publish the exact bias protocol used for
    every curve.
    """

    device: Device1D
    left_lead: LeadBlocks
    right_lead: LeadBlocks
    vertex: DenseCubicVertex
    scale: SolverScale

    def problem(
        self,
        grid: SignedFrequencyGrid,
        *,
        left_temperature_kelvin: float,
        right_temperature_kelvin: float,
    ) -> CubicSCBAProblem:
        """Build a solver problem after converting kelvin to solver units."""

        left_temperature = self.scale.dimensionless_temperature(
            left_temperature_kelvin,
            boltzmann_constant=_BOLTZMANN_J_K,
        )
        right_temperature = self.scale.dimensionless_temperature(
            right_temperature_kelvin,
            boltzmann_constant=_BOLTZMANN_J_K,
        )
        left = HarmonicReservoir(
            lead=self.left_lead,
            temperature=float(left_temperature),
            device_to_lead=np.array(self.left_lead.d01, dtype=np.float64),
            device_indices=(0,),
        )
        right = HarmonicReservoir(
            lead=self.right_lead,
            temperature=float(right_temperature),
            device_to_lead=np.array(self.right_lead.d01, dtype=np.float64),
            device_indices=(2,),
        )
        return CubicSCBAProblem(
            device=self.device,
            left=left,
            right=right,
            vertex=self.vertex,
            grid=grid,
            scale=self.scale,
        )


def _wang_paper_vertex(*, paper_t: float) -> Array:
    r"""Return ``Psi`` in the paper's mass-normalized physical units.

    Wang et al. write

    ``H_n = (t / 3) * sum_i (u_i - u_{i+1})**3``.

    NEGFPy uses ``H_3 = sum(Psi_abc u_a u_b u_c) / 3!``.  Consequently
    each oriented bond contributes ``Psi = 2*t*s_a*s_b*s_c``, with signs
    ``(+1, -1)`` on the left and right sites.  This is also consistent with
    the authors' archived ``noneq.nb`` notebook: its code variable is
    ``t_code=t/3`` and its tensor entries are ``T=3*t_code``.
    """

    values = np.zeros((3, 3, 3), dtype=np.float64)
    for left, right in ((0, 1), (1, 2)):
        signs = {left: 1.0, right: -1.0}
        for indices in product((left, right), repeat=3):
            values[indices] += (
                2.0
                * paper_t
                * signs[indices[0]]
                * signs[indices[1]]
                * signs[indices[2]]
            )
    return values


def _build_wang_2006_fixture(
    *,
    k_left: float,
    k_right: float,
    k_center: float,
    pinning_left: float,
    pinning_right: float,
    variant: str,
    harmonic_source_units: str,
    harmonic_provenance: dict[str, object] | None = None,
) -> Wang2006ThreeAtomFixture:
    """Build one explicit interpretation of Wang et al.'s Figure 2 model."""

    paper_t = 1.8

    omega0 = float(np.sqrt(_DYNAMICAL_UNIT_SI))
    provenance = {} if harmonic_provenance is None else dict(
        harmonic_provenance
    )
    scale = SolverScale(
        omega0=omega0,
        hbar=_HBAR_J_S,
        metadata={
            "model": "Wang2006_three_atom_FPU_alpha",
            "model_variant": variant,
            "source": "PhysRevB.74.033408",
            "source_doi": "10.1103/PhysRevB.74.033408",
            "source_code": (
                "https://phyweb.physics.nus.edu.sg/~phywjs/NEGF/noneq.nb"
            ),
            "source_preprint": "arXiv:cond-mat/0605028v1",
            "source_audit_date": "2026-07-24",
            "source_preprint_pdf_sha256": (
                "e238818b720bc368e45298d9ab11e04c78564353e59d3f56677021f2574f345b"
            ),
            "source_preprint_archive_sha256": (
                "5b9485665edb217fd67a4c9ef7c7ffacdd36a7ce497827698af56a7568da0ae0"
            ),
            "source_code_sha256": (
                "001aed1ef40d55dcc034244f054d21603a7603a95fb8d7113bdd1456b595f688"
            ),
            "source_units": {
                "dynamical_matrix": "eV/(angstrom^2*amu)",
                "cubic_vertex": "eV/(angstrom^3*amu^(3/2))",
                "temperature": "K",
            },
            "harmonic_source_units": harmonic_source_units,
            "harmonic_input_provenance": provenance,
            "legacy_dynamical_to_eV_per_angstrom2": (
                _LEGACY_DYNAMICAL_TO_EV_ANGSTROM2
            ),
            "legacy_frequency_to_solver": _LEGACY_FREQUENCY_TO_SOLVER,
            "frequency_kind": "angular_frequency",
        },
    )

    left_lead = LeadBlocks(
        d00=np.array([[2.0 * k_left + pinning_left]], dtype=np.float64),
        d01=np.array([[-k_left]], dtype=np.float64),
    )
    right_lead = LeadBlocks(
        d00=np.array([[2.0 * k_right + pinning_right]], dtype=np.float64),
        d01=np.array([[-k_right]], dtype=np.float64),
    )
    device = Device1D(
        onsite_blocks=[
            np.array([[k_left + k_center]], dtype=np.float64),
            np.array([[2.0 * k_center]], dtype=np.float64),
            np.array([[k_center + k_right]], dtype=np.float64),
        ],
        coupling_blocks=[
            np.array([[-k_center]], dtype=np.float64),
            np.array([[-k_center]], dtype=np.float64),
        ],
    )

    paper_vertex = _wang_paper_vertex(paper_t=paper_t)
    cubic_to_solver = (
        _CUBIC_UNIT_SI * np.sqrt(scale.hbar) / scale.omega0**2.5
    )
    vertex = DenseCubicVertex(
        values=paper_vertex * cubic_to_solver,
        metadata={
            "model": "Wang2006_three_atom_FPU_alpha",
            "model_variant": variant,
            "interaction_topology": "bond",
            "harmonic_bond_current_supported": False,
            "mass_weighted": True,
            "cubic_convention": "H3=(1/3!)*sum_abc(Psi_abc*u_a*u_b*u_c)",
            "paper_hamiltonian": (
                "Hn=(t/3)*sum_i(u_i-u_(i+1))^3"
            ),
            "paper_t": paper_t,
            "paper_t_units": "eV/(angstrom^3*amu^(3/2))",
            "paper_vertex_to_solver_factor": cubic_to_solver,
            "source": "PhysRevB.74.033408 and archived noneq.nb",
        },
    )
    return Wang2006ThreeAtomFixture(
        device=device,
        left_lead=left_lead,
        right_lead=right_lead,
        vertex=vertex,
        scale=scale,
    )


def wang_2006_caption_literal_fixture() -> Wang2006ThreeAtomFixture:
    r"""Return the literal numerical values printed in the Figure 2 caption.

    This interpretation uses ``k_L=1.56``, ``k_R=1.44``, ``k_C=1.38``,
    ``k_L^onsite=0.01``, and ``k_R^onsite=0.02`` in
    eV/(angstrom\ :sup:`2` amu).  It is a deterministic convention fixture,
    but it does **not** reproduce the author-posted ballistic EPS curve at low
    temperature.
    """

    return _build_wang_2006_fixture(
        k_left=1.56,
        k_right=1.44,
        k_center=1.38,
        pinning_left=0.01,
        pinning_right=0.02,
        variant="caption_literal",
        harmonic_source_units="literal Figure 2 caption",
        harmonic_provenance={
            "main_springs": "active final Figure 2 caption",
            "onsite_terms": "active final Figure 2 caption",
            "inferred_values": False,
        },
    )


def wang_2006_inferred_author_legacy_figure2_fixture(
) -> Wang2006ThreeAtomFixture:
    r"""Return an explicitly inferred interpretation of Wang Figure 2.

    The author TeX retains the original three main-spring inputs
    ``(2.5, 2.3, 2.2)`` in units of ``10^4 dyn/(cm*amu)``.  It does not
    explicitly archive the onsite values in that unit.  This fixture converts
    the main springs and applies the *inferred* hypothesis that the printed
    onsite numbers ``(0.01, 0.02)`` were also legacy-unit inputs.  That
    hypothesis reproduces the author-posted vector curve, but EPS agreement
    alone does not make the inverse model unique.

    The cubic coefficient remains the paper value ``t=1.8`` in
    eV/(angstrom\ :sup:`3` amu\ :sup:`3/2`).  In the archived notebook this
    is represented by ``t_code=0.6`` and tensor entries
    ``T=3*t_code=1.8``.
    """

    factor = _LEGACY_DYNAMICAL_TO_EV_ANGSTROM2
    return _build_wang_2006_fixture(
        k_left=2.5 * factor,
        k_right=2.3 * factor,
        k_center=2.2 * factor,
        pinning_left=0.01 * factor,
        pinning_right=0.02 * factor,
        variant="inferred_author_legacy_fig2",
        harmonic_source_units=(
            "main springs: 10^4 dyn/(cm*amu), author TeX commented inputs; "
            "onsite unit: inferred from EPS agreement"
        ),
        harmonic_provenance={
            "main_springs": (
                "source-level commented legacy values in author TeX"
            ),
            "onsite_terms": (
                "legacy-unit assignment inferred from vector-EPS agreement"
            ),
            "inferred_values": True,
            "inverse_model_unique": False,
        },
    )


def wang_2006_author_legacy_figure2_fixture() -> Wang2006ThreeAtomFixture:
    """Compatibility alias for the explicitly inferred legacy fixture.

    The historical name must not be interpreted as proof that all five
    harmonic inputs were archived in legacy units.
    """

    return wang_2006_inferred_author_legacy_figure2_fixture()


def wang_2006_three_atom_fixture() -> Wang2006ThreeAtomFixture:
    """Backward-compatible alias for the explicit caption-literal fixture."""

    return wang_2006_caption_literal_fixture()


__all__ = [
    "Wang2006ThreeAtomFixture",
    "wang_2006_author_legacy_figure2_fixture",
    "wang_2006_caption_literal_fixture",
    "wang_2006_inferred_author_legacy_figure2_fixture",
    "wang_2006_three_atom_fixture",
]
