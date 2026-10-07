"""Hamiltonian heat current for a real-space cubic device and a declared cut.

Each potential monomial is shared equally among its coordinate legs, counting
repeated indices. The resulting Hermitian operator is compiled into sparse
coefficients of Weyl-ordered p*u and p*u*u moments. This module supplies the
operator, not an SCBA approximation for the required three-point moments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._harmonic import assemble_device_dynamical_matrix
from .grid import SignedFrequencyGrid
from .sparse_cubic import SparseCubicVertex
from .types import CubicSCBAProblem, DenseCubicVertex, KeldyshSpectrum


def _real(values, shape, name):
    array = np.asarray(values)
    if array.shape != shape or array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real array with shape {shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite.")
    with np.errstate(over="ignore"):
        result = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be representable as finite float64 values.")
    return result


def _collect(indices, coefficients, width):
    """Collect commuting monomials by summing every supplied coefficient."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1, width)
    coefficients = np.asarray(coefficients, dtype=float).reshape(-1)
    keys, inverse = np.unique(indices, axis=0, return_inverse=True)
    totals = np.zeros(len(keys))
    np.add.at(totals, inverse, coefficients)
    if not np.all(np.isfinite(totals)):
        raise ValueError("Cut-current coefficient accumulation overflowed.")
    keep = totals != 0.0
    return keys[keep], totals[keep]


@dataclass(frozen=True, slots=True, eq=False)
class CubicCutCurrentOperator:
    """Sparse complete internal cut operator in real canonical coordinates.

    ``cut`` is the number of layers on the left (1 through n_layers-1).
    Indices identify momentum first, then one or two displacements. Each
    coefficient multiplies the symmetrized expectation
    ``< {p_i, u_j}/2 >`` or ``< {p_i, u_j*u_k}/2 >``. The latter equals full
    Weyl ordering, including repeated indices. Coefficients are in solver
    units; evaluated physical currents include ``current_scale`` once.
    """

    ndof: int
    n_layers: int
    dof_per_layer: int
    cut: int
    current_scale: float
    harmonic_indices: np.ndarray
    harmonic_coefficients: np.ndarray
    cubic_indices: np.ndarray
    cubic_coefficients: np.ndarray

    def __post_init__(self):
        for name in ("ndof", "n_layers", "dof_per_layer", "cut"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
        if (self.dof_per_layer < 1 or self.ndof != self.n_layers * self.dof_per_layer
                or not 1 <= self.cut < self.n_layers):
            raise ValueError("Invalid device dimensions or cut.")
        if not np.isfinite(self.current_scale) or self.current_scale <= 0.0:
            raise ValueError("current_scale must be finite and positive.")
        for prefix, width in (("harmonic", 2), ("cubic", 3)):
            raw = np.asarray(getattr(self, prefix + "_indices"))
            if (raw.ndim != 2 or raw.shape[1] != width or raw.dtype.kind not in "iu"
                    or np.any(raw < 0) or np.any(raw >= self.ndof)):
                raise ValueError(f"Invalid {prefix} moment indices.")
            indices = np.array(raw, dtype=np.int64, copy=True)
            coefficients = _real(getattr(self, prefix + "_coefficients"),
                                 (len(raw),), prefix + "_coefficients").copy()
            indices.setflags(write=False)
            coefficients.setflags(write=False)
            object.__setattr__(self, prefix + "_indices", indices)
            object.__setattr__(self, prefix + "_coefficients", coefficients)

    @property
    def metadata(self):
        return {
            "schema": "negfpy.research.cubic-cut-current.v1",
            "partition": "equal-coordinate-leg",
            "orientation": "left-to-right",
            "cut": int(self.cut),
            "ndof": int(self.ndof),
            "harmonic_moment_count": len(self.harmonic_indices),
            "cubic_moment_count": len(self.cubic_indices),
            "current_scale": float(self.current_scale),
            "moment_ordering": "one-half-anticommutator-momentum-with-coordinate-product",
            "coordinate_basis": "real-canonical-mass-normalized-solver",
            "scba_three_point_closure_implemented": False,
        }

    def evaluate_moments(self, harmonic_moments, cubic_moments):
        """Contract explicitly supplied real symmetrized moments at this cut.

        Arrays must follow this operator's index rows. In particular, cubic
        moments are required; zero values are never inferred from two-point
        Green functions. The caller owns the state approximation/provenance.
        Supplying moments does not establish transport conservation.
        """
        two = _real(harmonic_moments, (len(self.harmonic_indices),), "harmonic_moments")
        three = _real(cubic_moments, (len(self.cubic_indices),), "cubic_moments")
        harmonic = float(self.harmonic_coefficients @ two)
        cubic = float(self.cubic_coefficients @ three)
        total = harmonic + cubic
        values = np.array([harmonic, cubic, total])
        physical = self.current_scale * values
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(physical)):
            raise ValueError("Cut-current evaluation overflowed.")
        return {
            **self.metadata,
            "evaluation": "symmetrized-moments",
            "harmonic_solver": harmonic,
            "cubic_solver": cubic,
            "total_solver": total,
            "harmonic": float(physical[0]),
            "cubic": float(physical[1]),
            "total": float(physical[2]),
            "full_operator_included": True,
            "state_moments_verified": False,
            "spatial_validation_passed": False,
            "changes_solver_acceptance": False,
        }

    def evaluate_phase_space(self, displacement, velocity):
        """Evaluate the classical instantaneous polynomial on one state.

        Both vectors are real, dimensionless solver coordinates; velocity is
        du/d(t*omega0), which equals canonical momentum in these units. This
        is not a quantum or steady-state material current evaluation.
        """
        u = _real(displacement, (self.ndof,), "displacement")
        p = _real(velocity, (self.ndof,), "velocity")
        i, j = self.harmonic_indices.T
        a, b, c = self.cubic_indices.T
        result = self.evaluate_moments(p[i] * u[j], p[a] * u[b] * u[c])
        result["evaluation"] = "classical-phase-space"
        return result

    def harmonic_moments_from_green(self, grid, green):
        r"""Extract symmetrized p*u moments from the stored signed lesser G.

        With G< = -i <u_j(t')u_i(t)> in solver units,
        <W(p_i u_j)> = Re integral d omega/(2*pi) omega G<_ij(omega).
        This uses the existing rectangle grid and makes no frequency-accuracy
        claim. There is deliberately no Gaussian substitute for cubic moments.
        """
        if not isinstance(grid, SignedFrequencyGrid) or not isinstance(green, KeldyshSpectrum):
            raise TypeError("grid and green must be signed-grid and Keldysh spectrum objects.")
        if green.shape != (grid.n_points, self.ndof, self.ndof):
            raise ValueError("Green function shape does not match the operator and grid.")
        i, j = self.harmonic_indices.T
        moments = grid.integration_weight * np.real(
            np.einsum("w,wt->t", grid.values, green.lesser[:, i, j])
        )
        if not np.all(np.isfinite(moments)):
            raise ValueError("Harmonic moment integration overflowed.")
        return moments


def build_cubic_cut_current_operator(problem, cut):
    """Compile -dH_left/dt with equal-leg partition for an internal device cut.

    ``cut=1`` means the cut after the first layer. Real-space canonical
    coordinates are required; complex Bloch matrices are rejected. The sparse
    vertex path never materializes an ndof**3 tensor. All ordered coefficients
    enter their potential monomials as Psi/6; this is algebraic collection,
    not replacement or averaging of the input permutation entries.
    """
    if not isinstance(problem, CubicSCBAProblem):
        raise TypeError("problem must be a CubicSCBAProblem.")
    if isinstance(cut, (bool, np.bool_)) or not isinstance(cut, (int, np.integer)):
        raise TypeError("cut must be an integer number of left layers.")
    if not 1 <= cut < problem.device.n_layers:
        raise ValueError("cut must lie between 1 and n_layers-1.")
    dynamical = assemble_device_dynamical_matrix(problem.device)
    if np.any(dynamical.imag != 0.0):
        raise ValueError("Cubic cut current requires a real-space dynamical matrix.")
    dynamical = dynamical.real
    inside = (np.arange(problem.ndof) < cut * problem.device.dof_per_layer).astype(int)
    harmonic = 0.5 * (inside[:, None] - inside[None, :]) * dynamical
    h_indices = np.argwhere(harmonic != 0.0)
    h_coefficients = harmonic[tuple(h_indices.T)]

    vertex = problem.vertex
    if isinstance(vertex, SparseCubicVertex):
        indices, coefficients = vertex.indices, vertex.coefficients
    elif isinstance(vertex, DenseCubicVertex):
        indices = np.argwhere(vertex.values != 0.0)
        coefficients = vertex.values[tuple(indices.T)]
    else:
        raise TypeError("Unsupported cubic vertex type.")
    # A commuting coordinate monomial can have 1, 3 or 6 ordered entries.
    # Sum all of them; never assume a common representative coefficient.
    monomials, potential_coefficients = _collect(np.sort(indices, axis=1), coefficients / 6.0, 3)
    left_count = inside[monomials].sum(axis=1)
    crosses = (left_count > 0) & (left_count < 3)
    monomials = monomials[crosses]
    left_count = left_count[crosses]
    potential_coefficients = potential_coefficients[crosses]
    terms, weights = [], []
    for leg in range(3):
        other = [slot for slot in range(3) if slot != leg]
        moment = np.column_stack((monomials[:, leg], monomials[:, other]))
        coefficient = potential_coefficients * (
            3 * inside[monomials[:, leg]] - left_count
        ) / 3.0
        terms.append(moment)
        weights.append(coefficient)
    c_indices, c_coefficients = _collect(np.concatenate(terms), np.concatenate(weights), 3)
    return CubicCutCurrentOperator(
        ndof=problem.ndof, n_layers=problem.device.n_layers,
        dof_per_layer=problem.device.dof_per_layer, cut=int(cut),
        current_scale=problem.scale.current_scale,
        harmonic_indices=h_indices, harmonic_coefficients=h_coefficients,
        cubic_indices=c_indices, cubic_coefficients=c_coefficients,
    )


__all__ = ["CubicCutCurrentOperator", "build_cubic_cut_current_operator"]
