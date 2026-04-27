"""General quartic SCMF benchmark utilities plus Wang Fig. 4 paper presets.

This module provides a small-system benchmark engine for the quartic
self-consistent mean-field (SCMF) setup discussed in:

- J.-S. Wang et al., review Fig. 4
- ``Nonequilibrium Green's function method for thermal transport in junctions``

The implementation here is intentionally isolated from the main block-lead NEGF
solver because the literature benchmark uses analytic Lorentz-Drude bath
self-energies rather than periodic principal-layer leads.

Design note:
    The solver pieces in this module are kept general so they can later be
    driven by material-derived small benchmark models as well as toy models.
    The Wang Fig. 4 case builders are deliberately kept separate as paper-
    specific presets. This lets us audit exact paper inputs without coupling the
    core inelastic theory to one literature convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray
EV_J = 1.602_176_634e-19
HBAR_SI = 1.054_571_817e-34
KB_SI = 1.380_649e-23
AMU_KG = 1.660_539_066_60e-27
ANGSTROM_M = 1.0e-10
MEV_FORCE_TO_SI = 1.0e-3 * EV_J / ((ANGSTROM_M**2) * AMU_KG)
EV_QUARTIC_TO_SI = EV_J / ((ANGSTROM_M**4) * (AMU_KG**2))


def _symmetrize_matrix(matrix: Array) -> Array:
    arr = np.asarray(matrix, dtype=np.complex128)
    return 0.5 * (arr + arr.conj().T)


def _stable_bose_occupation(omega: Array, temperature: float, *, kb_effective: float) -> Array:
    w = np.asarray(omega, dtype=float)
    if temperature <= 0.0:
        return np.where(w < 0.0, -1.0, 0.0)
    x = w / (float(kb_effective) * float(temperature))
    out = np.zeros_like(x)
    small = np.abs(x) < 1e-6
    regular = (~small) & (np.abs(x) < 700.0)
    positive_large = x >= 700.0
    negative_large = x <= -700.0
    out[small] = 1.0 / np.where(np.abs(x[small]) > 1e-12, x[small], np.sign(x[small]) * 1e-12 + 1e-12)
    out[regular] = 1.0 / np.expm1(x[regular])
    out[positive_large] = 0.0
    out[negative_large] = -1.0
    return out


@dataclass(frozen=True)
class LorentzDrudeBath:
    """Single-channel Lorentz-Drude bath attached through a projector matrix.

    The spectral density is

        J(omega) = gamma * omega / (1 + (omega / omega_d)^2)

    and the retarded bath self-energy is chosen so that

        Im Sigma^r(omega) = -J(omega).
    """

    gamma: float
    omega_d: float
    projector: Array

    def __post_init__(self) -> None:
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        if self.omega_d <= 0.0:
            raise ValueError("omega_d must be positive.")
        proj = np.asarray(self.projector, dtype=np.complex128)
        if proj.ndim != 2 or proj.shape[0] != proj.shape[1]:
            raise ValueError("projector must be a square 2D array.")
        object.__setattr__(self, "projector", _symmetrize_matrix(proj))

    @property
    def dim(self) -> int:
        return int(self.projector.shape[0])

    def retarded_self_energy(self, omega: float) -> Array:
        scalar = (
            float(self.gamma)
            * float(self.omega_d)
            * float(omega)
            / (float(omega) + 1j * float(self.omega_d))
        )
        return scalar * self.projector

    def lesser_self_energy(self, omega: float, *, temperature: float, kb_effective: float) -> Array:
        sigma_r = self.retarded_self_energy(float(omega))
        sigma_a = sigma_r.conj().T
        n = float(_stable_bose_occupation(np.array([omega], dtype=float), temperature, kb_effective=kb_effective)[0])
        return n * (sigma_r - sigma_a)


@dataclass(frozen=True)
class QuarticSCMFBenchmarkCase:
    """Small quartic benchmark system with analytic baths."""

    label: str
    harmonic: Array
    quartic: Array
    bath_left: LorentzDrudeBath
    bath_right: LorentzDrudeBath
    literature_note: str

    def __post_init__(self) -> None:
        harmonic = np.asarray(self.harmonic, dtype=np.complex128)
        quartic = np.asarray(self.quartic, dtype=np.complex128)
        if harmonic.ndim != 2 or harmonic.shape[0] != harmonic.shape[1]:
            raise ValueError("harmonic must be a square 2D array.")
        dim = harmonic.shape[0]
        if quartic.shape != (dim, dim, dim, dim):
            raise ValueError("quartic must have shape (dim, dim, dim, dim).")
        if self.bath_left.dim != dim or self.bath_right.dim != dim:
            raise ValueError("bath projector dimensions must match the harmonic system size.")
        object.__setattr__(self, "harmonic", _symmetrize_matrix(harmonic))
        object.__setattr__(self, "quartic", quartic)

    @property
    def dim(self) -> int:
        return int(self.harmonic.shape[0])


def wang_lorentz_drude_bath_from_epsilon(
    *,
    epsilon: float,
    omega_d: float,
    projector: Array,
) -> LorentzDrudeBath:
    """Build a Lorentz-Drude bath from Wang's paper notation.

    In the review caption for Fig. 4 the bath spectral density is written as

        J(omega) = epsilon^2 * omega / (1 + omega^2 / omega_d^2).

    The generic ``LorentzDrudeBath`` class stores the prefactor multiplying
    ``omega / (1 + omega^2 / omega_d^2)`` as ``gamma``. This helper keeps the
    paper-facing builder exact while preserving a reusable generic bath class.
    """

    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    return LorentzDrudeBath(gamma=float(epsilon) ** 2, omega_d=float(omega_d), projector=projector)


def wang_review_fig4_audit() -> dict[str, object]:
    """Return a structured audit of what is matched exactly for Wang Fig. 4."""

    return {
        "paper": {
            "title": "Nonequilibrium Green's function method for quantum thermal transport",
            "identifier": "arXiv:1303.7317v1",
            "target_figure": "Fig. 4",
        },
        "matched_exactly_in_code": {
            "quartic_scmf_closure_eq_126": (
                "Sigma_n(j,j') = 3 i hbar delta(t,t') sum_{k,l} T_{j j' k l} G_{k l}(t,t)"
            ),
            "covariance_eq_129_structure": "<u u^T> = i hbar integral G^< d omega / (2 pi)",
            "bath_functional_form": "J_alpha(omega) = epsilon^2 * omega / (1 + omega^2 / omega_D^2)",
            "temperature_protocol": "T_L = 1.25 T, T_R = 0.75 T",
            "one_particle_coefficients": {
                "Omega2_meV_per_A2_u": 60.321,
                "T1111_eV_per_A4_u2": [0.241, 1.2, 2.4],
            },
            "two_particle_coefficients": {
                "K11_K22_meV_per_A2_u": 60.321,
                "K12_K21_meV_per_A2_u": -30.165,
                "quartic_curves_eV_per_A4_u2": {
                    "black": {"T1111": 0.483, "T1112": -0.241, "T1122": 0.241},
                    "red": {"T1111": 2.4, "T1112": -1.2, "T1122": 1.2},
                    "blue": {"T1111": 4.8, "T1112": -2.4, "T1122": 2.4},
                },
            },
            "current_evaluation_strategy": "Caroli/Landauer with Sigma_n^r incorporated into G^r",
        },
        "still_under_audit": {
            "paper_native_unit_to_si_mapping": (
                "Exact mapping from the paper's mass-normalized units to SI dynamic units "
                "is not yet proven in the code."
            ),
            "absolute_current_scale": (
                "Current ordering is sensible, but the magnitude is still below the published nW scale."
            ),
            "analytic_bath_normalization_prefactor": (
                "The functional form is matched, but the absolute normalization in SI remains under audit."
            ),
        },
    }


def wang_review_fig4_paper_convention_spec(family: str) -> dict[str, object]:
    """Return the exact Fig. 4 benchmark specification for one Wang family.

    This helper is intentionally paper-specific. It lets us benchmark against
    the published toy model without baking Wang-only conventions into the
    general inelastic solver.
    """

    family_name = str(family).strip().lower()
    if family_name not in {"one_particle", "two_particle"}:
        raise ValueError("family must be either 'one_particle' or 'two_particle'.")

    shared = {
        "paper": "J.-S. Wang et al., arXiv:1303.7317v1",
        "target_figure": "Fig. 4",
        "approximation": "quartic SCMF",
        "temperature_protocol": {"T_left": "1.25 * T", "T_right": "0.75 * T"},
        "bath": {
            "model": "Lorentz-Drude",
            "spectral_density": "J_alpha(omega) = epsilon^2 * omega / (1 + omega^2 / omega_D^2)",
            "epsilon_meV_per_A2_u": 6.0321,
            "hbar_omega_D_eV": 10.0,
        },
        "published_current_axis": "10^-9 W",
    }

    if family_name == "one_particle":
        return {
            **shared,
            "family": "one_particle",
            "harmonic": {"Omega2_meV_per_A2_u": 60.321},
            "quartic_curves_eV_per_A4_u2": {
                "black": {"T1111": 0.241},
                "red": {"T1111": 1.2},
                "blue": {"T1111": 2.4},
            },
        }

    return {
        **shared,
        "family": "two_particle",
        "harmonic": {
            "K11_meV_per_A2_u": 60.321,
            "K22_meV_per_A2_u": 60.321,
            "K12_meV_per_A2_u": -30.165,
            "K21_meV_per_A2_u": -30.165,
        },
        "quartic_curves_eV_per_A4_u2": {
            "black": {"T1111": 0.483, "T1112": -0.241, "T1122": 0.241},
            "red": {"T1111": 2.4, "T1112": -1.2, "T1122": 1.2},
            "blue": {"T1111": 4.8, "T1112": -2.4, "T1122": 2.4},
        },
        "index_permutation_note": "Curly-bracket subscripts indicate all permutations.",
    }


def wang_review_fig4_paper_convention_sweep(
    family: str,
    temperatures: Array,
    *,
    omega_max: float,
    n_omega: int = 1201,
    n_omega_cov: int = 1201,
    eta: float = 1e-8,
    kb_effective: float = 1.0,
    hbar_effective: float = 1.0,
    max_iter: int = 100,
    mixing: float = 0.5,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Run the Wang Fig. 4 benchmark in the paper's native toy-model convention.

    This wrapper is deliberately separate from the more flexible benchmark
    pieces. It locks the published parameter family and temperature protocol so
    we can benchmark exact paper conventions step by step.
    """

    family_name = str(family).strip().lower()
    if family_name == "one_particle":
        cases = wang_review_fig4_one_particle_cases()
    elif family_name == "two_particle":
        cases = wang_review_fig4_two_particle_cases()
    else:
        raise ValueError("family must be either 'one_particle' or 'two_particle'.")

    results: dict[str, object] = {}
    for color in ("black", "red", "blue"):
        results[color] = quartic_scmf_current_vs_temperature(
            cases[color],
            temperatures,
            temp_left_factor=1.25,
            temp_right_factor=0.75,
            omega_max=float(omega_max),
            n_omega=int(n_omega),
            n_omega_cov=int(n_omega_cov),
            eta=float(eta),
            kb_effective=float(kb_effective),
            hbar_effective=float(hbar_effective),
            max_iter=int(max_iter),
            mixing=float(mixing),
            tol=float(tol),
        )

    return {
        "spec": wang_review_fig4_paper_convention_spec(family_name),
        "results": results,
    }


def _retarded_green_function(
    omega: float,
    *,
    harmonic: Array,
    sigma_static: Array,
    bath_left: LorentzDrudeBath,
    bath_right: LorentzDrudeBath,
    eta: float,
) -> tuple[Array, Array, Array]:
    sigma_l = bath_left.retarded_self_energy(float(omega))
    sigma_r = bath_right.retarded_self_energy(float(omega))
    dim = harmonic.shape[0]
    system = (
        ((float(omega) ** 2) + 1j * float(eta)) * np.eye(dim, dtype=np.complex128)
        - np.asarray(harmonic, dtype=np.complex128)
        - np.asarray(sigma_static, dtype=np.complex128)
        - sigma_l
        - sigma_r
    )
    g = np.linalg.inv(system)
    return g, sigma_l, sigma_r


def _covariance_from_open_system(
    *,
    case: QuarticSCMFBenchmarkCase,
    temp_left: float,
    temp_right: float,
    omega_max: float,
    n_omega_cov: int,
    sigma_static: Array,
    eta: float,
    kb_effective: float,
    hbar_effective: float,
) -> Array:
    if n_omega_cov < 8:
        raise ValueError("n_omega_cov must be at least 8.")
    omega_eps = max(float(omega_max) / max(10.0 * float(n_omega_cov), 1.0), 1e-10)
    omega_pos = np.linspace(omega_eps, float(omega_max), int(n_omega_cov))
    omega_neg = -omega_pos[::-1]
    omegas = np.concatenate([omega_neg, omega_pos])

    cov_omega = np.zeros((omegas.size, case.dim, case.dim), dtype=np.complex128)
    for i, omega in enumerate(omegas):
        g, _, _ = _retarded_green_function(
            omega=float(omega),
            harmonic=case.harmonic,
            sigma_static=sigma_static,
            bath_left=case.bath_left,
            bath_right=case.bath_right,
            eta=eta,
        )
        sigma_less = (
            case.bath_left.lesser_self_energy(float(omega), temperature=float(temp_left), kb_effective=kb_effective)
            + case.bath_right.lesser_self_energy(float(omega), temperature=float(temp_right), kb_effective=kb_effective)
        )
        g_less = g @ sigma_less @ g.conj().T
        cov_omega[i] = 1j * g_less

    covariance = float(hbar_effective) * np.trapezoid(cov_omega, omegas, axis=0) / (2.0 * np.pi)
    return _symmetrize_matrix(covariance)


def quartic_scmf_static_self_energy(
    case: QuarticSCMFBenchmarkCase,
    *,
    temp_left: float,
    temp_right: float,
    omega_max: float,
    n_omega_cov: int = 1201,
    eta: float = 1e-8,
    kb_effective: float = 1.0,
    hbar_effective: float = 1.0,
    max_iter: int = 100,
    mixing: float = 0.5,
    tol: float = 1e-8,
    raise_on_nonconvergence: bool = False,
) -> tuple[Array, dict[str, object]]:
    """Return the quartic SCMF static self-energy for a small analytic-bath model."""

    if max_iter <= 0:
        raise ValueError("max_iter must be positive.")
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must be in (0, 1].")
    if tol < 0.0:
        raise ValueError("tol must be non-negative.")
    if omega_max <= 0.0:
        raise ValueError("omega_max must be positive.")

    sigma = np.zeros((case.dim, case.dim), dtype=np.complex128)
    covariance = np.zeros_like(sigma)
    converged = False
    residual = float("inf")
    iterations = 0

    for it in range(max_iter):
        iterations = it + 1
        covariance_new = _covariance_from_open_system(
            case=case,
            temp_left=float(temp_left),
            temp_right=float(temp_right),
            omega_max=float(omega_max),
            n_omega_cov=int(n_omega_cov),
            sigma_static=sigma,
            eta=float(eta),
            kb_effective=float(kb_effective),
            hbar_effective=float(hbar_effective),
        )
        sigma_new = 3.0 * np.einsum("ijkl,kl->ij", case.quartic, covariance_new, optimize=True)
        sigma_next = mixing * sigma_new + (1.0 - mixing) * sigma
        residual = float(np.linalg.norm(sigma_next - sigma) / max(np.linalg.norm(sigma_next), 1e-30))
        sigma = _symmetrize_matrix(sigma_next)
        covariance = covariance_new
        if residual <= tol:
            converged = True
            break

    if not converged and raise_on_nonconvergence:
        raise RuntimeError(
            "Quartic SCMF benchmark solve did not converge within max_iter "
            f"(iterations={iterations}, residual={residual:.3e}, tol={tol:.3e})."
        )

    info = {
        "converged": bool(converged),
        "iterations": int(iterations),
        "residual": float(residual),
        "sigma_static_norm": float(np.linalg.norm(sigma)),
        "covariance_norm": float(np.linalg.norm(covariance)),
    }
    return sigma, info


def quartic_scmf_current_vs_temperature(
    case: QuarticSCMFBenchmarkCase,
    temperatures: Array,
    *,
    temp_left_factor: float = 1.25,
    temp_right_factor: float = 0.75,
    omega_max: float | None = None,
    n_omega: int = 1201,
    n_omega_cov: int = 1201,
    eta: float = 1e-8,
    kb_effective: float = 1.0,
    hbar_effective: float = 1.0,
    max_iter: int = 100,
    mixing: float = 0.5,
    tol: float = 1e-8,
) -> dict[str, object]:
    """Return a Wang-style quartic SCMF current sweep for one literature case."""

    temps = np.asarray(temperatures, dtype=float)
    if temps.ndim != 1 or temps.size == 0 or np.any(temps <= 0.0):
        raise ValueError("temperatures must be a non-empty 1D array of positive values.")
    if n_omega < 8:
        raise ValueError("n_omega must be at least 8.")
    if temp_left_factor <= temp_right_factor:
        raise ValueError("temp_left_factor must exceed temp_right_factor.")

    if omega_max is None:
        w_harm = float(np.sqrt(max(np.max(np.linalg.eigvalsh(np.asarray(case.harmonic, dtype=float))), 1e-12)))
        omega_max = 6.0 * max(w_harm, float(case.bath_left.omega_d), float(case.bath_right.omega_d))

    omegas = np.linspace(max(float(omega_max) / max(1000.0, float(n_omega)), 1e-10), float(omega_max), int(n_omega))
    temp_left = float(temp_left_factor) * temps
    temp_right = float(temp_right_factor) * temps
    current = np.zeros_like(temps)
    transmission_map = np.zeros((temps.size, omegas.size), dtype=float)
    converged = np.zeros_like(temps, dtype=bool)
    iterations = np.zeros_like(temps, dtype=int)
    residuals = np.zeros_like(temps)

    for i, temp_avg in enumerate(temps):
        sigma_static, info = quartic_scmf_static_self_energy(
            case,
            temp_left=float(temp_left[i]),
            temp_right=float(temp_right[i]),
            omega_max=float(omega_max),
            n_omega_cov=int(n_omega_cov),
            eta=float(eta),
            kb_effective=float(kb_effective),
            hbar_effective=float(hbar_effective),
            max_iter=int(max_iter),
            mixing=float(mixing),
            tol=float(tol),
        )
        converged[i] = bool(info["converged"])
        iterations[i] = int(info["iterations"])
        residuals[i] = float(info["residual"])

        tvals = np.zeros_like(omegas)
        nleft = _stable_bose_occupation(omegas, float(temp_left[i]), kb_effective=float(kb_effective))
        nright = _stable_bose_occupation(omegas, float(temp_right[i]), kb_effective=float(kb_effective))

        for iw, omega in enumerate(omegas):
            g, sigma_l, sigma_r = _retarded_green_function(
                omega=float(omega),
                harmonic=case.harmonic,
                sigma_static=sigma_static,
                bath_left=case.bath_left,
                bath_right=case.bath_right,
                eta=float(eta),
            )
            gamma_l = 1j * (sigma_l - sigma_l.conj().T)
            gamma_r = 1j * (sigma_r - sigma_r.conj().T)
            tvals[iw] = max(float(np.real_if_close(np.trace(gamma_l @ g @ gamma_r @ g.conj().T)).real), 0.0)

        transmission_map[i, :] = tvals
        integrand = omegas * tvals * (nleft - nright)
        current[i] = float(float(hbar_effective) * np.trapezoid(integrand, omegas) / (2.0 * np.pi))

    return {
        "label": case.label,
        "temperature_avg": temps,
        "temperature_left": temp_left,
        "temperature_right": temp_right,
        "omegas": omegas,
        "transmission_map": transmission_map,
        "current_internal": current,
        "converged": converged,
        "iterations": iterations,
        "residual": residuals,
        "literature_note": case.literature_note,
    }


def _quartic_tensor_with_permutations(
    dim: int,
    *,
    diagonal: dict[int, float],
    mixed_1112: float | None = None,
    mixed_1122: float | None = None,
) -> Array:
    phi4 = np.zeros((dim, dim, dim, dim), dtype=np.complex128)
    for idx, value in diagonal.items():
        phi4[idx, idx, idx, idx] = complex(value)

    if dim == 2:
        if mixed_1112 is not None:
            for inds in [(0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0),
                         (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1)]:
                phi4[inds] = complex(mixed_1112)
        if mixed_1122 is not None:
            for inds in [(0, 0, 1, 1), (0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0), (1, 1, 0, 0)]:
                phi4[inds] = complex(mixed_1122)
    return phi4


def wang_review_fig4_one_particle_cases() -> dict[str, QuarticSCMFBenchmarkCase]:
    """Return one-particle Wang Fig. 4 cases in the paper's native coefficient units."""

    cases: dict[str, QuarticSCMFBenchmarkCase] = {}
    harmonic = np.array([[60.321]], dtype=np.complex128)
    projector = np.array([[1.0]], dtype=np.complex128)
    bath_l = wang_lorentz_drude_bath_from_epsilon(epsilon=6.0321, omega_d=1.0e4, projector=projector)
    bath_r = wang_lorentz_drude_bath_from_epsilon(epsilon=6.0321, omega_d=1.0e4, projector=projector)

    for label, t1111 in {
        "black": 0.241,
        "red": 1.2,
        "blue": 2.4,
    }.items():
        quartic = np.zeros((1, 1, 1, 1), dtype=np.complex128)
        quartic[0, 0, 0, 0] = complex(t1111)
        cases[label] = QuarticSCMFBenchmarkCase(
            label=f"wang_fig4_one_particle_{label}",
            harmonic=harmonic,
            quartic=quartic,
            bath_left=bath_l,
            bath_right=bath_r,
            literature_note=(
                "One-particle quartic SCMF benchmark using the published Wang Fig. 4 coefficients "
                "in the paper's native units, including the bath written through J(omega)=epsilon^2 omega/(1+omega^2/omega_D^2)."
            ),
        )
    return cases


def wang_review_fig4_two_particle_cases() -> dict[str, QuarticSCMFBenchmarkCase]:
    """Return two-particle Wang Fig. 4 cases in the paper's native coefficient units."""

    cases: dict[str, QuarticSCMFBenchmarkCase] = {}
    harmonic = np.array([[60.321, -30.165], [-30.165, 60.321]], dtype=np.complex128)
    proj_left = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    proj_right = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    bath_l = wang_lorentz_drude_bath_from_epsilon(epsilon=6.0321, omega_d=1.0e4, projector=proj_left)
    bath_r = wang_lorentz_drude_bath_from_epsilon(epsilon=6.0321, omega_d=1.0e4, projector=proj_right)

    presets = {
        "black": (0.483, -0.241, 0.241),
        "red": (2.4, -1.2, 1.2),
        "blue": (4.8, -2.4, 2.4),
    }
    for label, (diag, mixed_1112, mixed_1122) in presets.items():
        quartic = _quartic_tensor_with_permutations(
            2,
            diagonal={0: diag, 1: diag},
            mixed_1112=mixed_1112,
            mixed_1122=mixed_1122,
        )
        cases[label] = QuarticSCMFBenchmarkCase(
            label=f"wang_fig4_two_particle_{label}",
            harmonic=harmonic,
            quartic=quartic,
            bath_left=bath_l,
            bath_right=bath_r,
            literature_note=(
                "Two-particle quartic SCMF benchmark using the published Wang Fig. 4 coefficients "
                "in the paper's native units, including the bath written through J(omega)=epsilon^2 omega/(1+omega^2/omega_D^2)."
            ),
        )
    return cases


def _mev_force_constant_to_si(value_mev_per_a2_u: float) -> float:
    return float(value_mev_per_a2_u) * MEV_FORCE_TO_SI


def _ev_quartic_to_si(value_ev_per_a4_u2: float) -> float:
    return float(value_ev_per_a4_u2) * EV_QUARTIC_TO_SI


def wang_review_kb_effective_si() -> float:
    """Return the effective ``k_B / ħ`` prefactor for ``omega`` in rad/s and ``T`` in K."""

    return KB_SI / HBAR_SI


def wang_review_hbar_effective_si() -> float:
    """Return ``ħ`` in SI for converting the benchmark current to Watts."""

    return HBAR_SI


def wang_review_fig4_one_particle_cases_si() -> dict[str, QuarticSCMFBenchmarkCase]:
    """Return one-particle Wang Fig. 4 cases with a heuristic SI conversion."""

    cases: dict[str, QuarticSCMFBenchmarkCase] = {}
    harmonic = np.array([[_mev_force_constant_to_si(60.321)]], dtype=np.complex128)
    projector = np.array([[1.0]], dtype=np.complex128)
    epsilon_force = _mev_force_constant_to_si(6.0321)
    omega_d = (10.0 * EV_J) / HBAR_SI
    bath_l = wang_lorentz_drude_bath_from_epsilon(epsilon=epsilon_force, omega_d=omega_d, projector=projector)
    bath_r = wang_lorentz_drude_bath_from_epsilon(epsilon=epsilon_force, omega_d=omega_d, projector=projector)

    for label, t1111 in {
        "black": 0.241,
        "red": 1.2,
        "blue": 2.4,
    }.items():
        quartic = np.zeros((1, 1, 1, 1), dtype=np.complex128)
        quartic[0, 0, 0, 0] = complex(_ev_quartic_to_si(t1111))
        cases[label] = QuarticSCMFBenchmarkCase(
            label=f"wang_fig4_one_particle_{label}_si",
            harmonic=harmonic,
            quartic=quartic,
            bath_left=bath_l,
            bath_right=bath_r,
            literature_note=(
                "Heuristic SI-converted one-particle quartic SCMF target based on Wang review Fig. 4. "
                "The functional form and published coefficients are preserved, but the exact paper-native "
                "unit mapping into SI is still under audit."
            ),
        )
    return cases


def wang_review_fig4_two_particle_cases_si() -> dict[str, QuarticSCMFBenchmarkCase]:
    """Return two-particle Wang Fig. 4 cases with a heuristic SI conversion."""

    cases: dict[str, QuarticSCMFBenchmarkCase] = {}
    harmonic = np.array(
        [
            [_mev_force_constant_to_si(60.321), -_mev_force_constant_to_si(30.165)],
            [-_mev_force_constant_to_si(30.165), _mev_force_constant_to_si(60.321)],
        ],
        dtype=np.complex128,
    )
    proj_left = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    proj_right = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.complex128)
    epsilon_force = _mev_force_constant_to_si(6.0321)
    omega_d = (10.0 * EV_J) / HBAR_SI
    bath_l = wang_lorentz_drude_bath_from_epsilon(epsilon=epsilon_force, omega_d=omega_d, projector=proj_left)
    bath_r = wang_lorentz_drude_bath_from_epsilon(epsilon=epsilon_force, omega_d=omega_d, projector=proj_right)

    presets = {
        "black": (0.483, -0.241, 0.241),
        "red": (2.4, -1.2, 1.2),
        "blue": (4.8, -2.4, 2.4),
    }
    for label, (diag, mixed_1112, mixed_1122) in presets.items():
        quartic = _quartic_tensor_with_permutations(
            2,
            diagonal={0: _ev_quartic_to_si(diag), 1: _ev_quartic_to_si(diag)},
            mixed_1112=_ev_quartic_to_si(mixed_1112),
            mixed_1122=_ev_quartic_to_si(mixed_1122),
        )
        cases[label] = QuarticSCMFBenchmarkCase(
            label=f"wang_fig4_two_particle_{label}_si",
            harmonic=harmonic,
            quartic=quartic,
            bath_left=bath_l,
            bath_right=bath_r,
            literature_note=(
                "Heuristic SI-converted two-particle quartic SCMF target based on Wang review Fig. 4. "
                "The functional form and published coefficients are preserved, but the exact paper-native "
                "unit mapping into SI is still under audit."
            ),
        )
    return cases
