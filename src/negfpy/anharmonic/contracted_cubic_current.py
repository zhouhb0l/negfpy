"""Stream a selected connected cubic cut current without individual moments.

This is an algebraic rearrangement of the one-vertex kernel in three_point,
not a new state approximation or an EOM substitute for the cubic operator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .cubic_cut_current import CubicCutCurrentOperator, build_cubic_cut_current_operator
from .grid import SignedFrequencyGrid
from .three_point import _complex, _lags, _potential
from .types import KeldyshSpectrum


def _readonly(values):
    result = np.asarray(values)
    result.setflags(write=False)
    return result


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


@dataclass(frozen=True, slots=True, eq=False)
class ContractedCubicCut:
    """Selected index plan; construct with build_contracted_cubic_cut.

    Provider C rows follow displacement_rows, P rows follow momentum_rows;
    both column sets follow active_indices. All indices refer to original
    device coordinates. The returned current is connected cubic only.
    """

    operator: CubicCutCurrentOperator
    active_indices: np.ndarray
    displacement_rows: np.ndarray
    momentum_rows: np.ndarray
    _response: np.ndarray
    _momentum_positions: np.ndarray
    _displacement_positions: np.ndarray

    @property
    def metadata(self):
        a, j, i = len(self.active_indices), len(self.displacement_rows), len(self.momentum_rows)
        targets = len(self.operator.cubic_indices)
        return {
            "schema": "negfpy.research.contracted-cubic-cut.v1",
            "cut": self.operator.cut,
            "ndof": self.operator.ndof,
            "active_coordinates": a,
            "displacement_rows": j,
            "momentum_rows": i,
            "operator_terms": targets,
            "dense_response_bytes": self._response.nbytes,
            # Excludes BLAS-internal scratch, provider buffers, input spectra
            # and allocator overhead. Only these explicit arrays are counted.
            "transform_and_gather_array_bytes": 16 * (a * a * j + a * j * j
                                                       + 2 * a * min(256, targets)),
            "coordinate_basis": "real-canonical-mass-normalized-solver",
            "approximation": "connected-one-cubic-vertex-with-supplied-two-point-lines",
            "full_three_point_tensor_constructed": False,
            "disconnected_means_included": False,
            "harmonic_current_included": False,
            "tail_error_estimated": False,
            "scba_self_consistency_verified": False,
            "spatial_validation_passed": False,
            "changes_solver_acceptance": False,
        }

    def _kernel(self, displacement, momentum):
        # V[a,b,c] C[k,c] -> T[a,b,k]; then C[j,b] -> X[a,j,k].
        # No conjugate transpose: these are Wick products, not inner products.
        first = self._response @ displacement.T
        transformed = displacement[None, :, :] @ first
        del first
        result = 0j
        for start in range(0, len(self.operator.cubic_indices), 256):
            stop = start + 256
            i = self._momentum_positions[start:stop]
            j, k = self._displacement_positions[start:stop].T
            products = np.einsum("ta,at->t", momentum[i], transformed[:, j, k])
            result += self.operator.cubic_coefficients[start:stop] @ products
        return 2.0 * result.imag

    def integrate(self, lag_times, correlation_provider, *, block_size=16):
        """Integrate finite-lag current from bounded selected correlation blocks.

        provider(times) must return C(nt,J,A), P(nt,I,A) in the declared row
        and column order. Both are centered connected two-point correlations.
        No three-point moments or full lag/matrix history
        are retained. Scalar integrands are retained so trapezoids cross block
        boundaries exactly. Provider means and state consistency are not inferred.
        """
        lag = _lags(lag_times)
        block_size = _positive_integer(block_size, "block_size")
        if not callable(correlation_provider):
            raise TypeError("correlation_provider must be callable.")
        a, j, i = len(self.active_indices), len(self.displacement_rows), len(self.momentum_rows)
        integrand = np.zeros(len(lag))
        largest = 0
        if len(self.operator.cubic_indices):
            for start in range(0, len(lag), block_size):
                stop = min(start + block_size, len(lag))
                count = stop - start
                displacement, momentum = correlation_provider(lag[start:stop])
                displacement = _complex(displacement, (count, j, a), "selected displacement correlation")
                momentum = _complex(momentum, (count, i, a), "selected momentum correlation")
                largest = max(largest, count)
                for local in range(count):
                    integrand[start + local] = self._kernel(displacement[local], momentum[local])
        value = float(np.trapezoid(integrand, lag))
        physical = value * self.operator.current_scale
        if not np.all(np.isfinite(integrand)) or not np.isfinite(value) or not np.isfinite(physical):
            raise ValueError("Contracted cubic current overflowed.")
        return {
            **self.metadata, "connected_cubic_solver": value, "connected_cubic": physical,
            "integrand": integrand, "endpoint_integrand": float(integrand[-1]),
            "lag_end": float(lag[-1]), "lag_points": len(lag),
            "max_provider_block_points": largest,
            "selected_correlation_block_bytes": 16 * largest * a * (i + j),
            "quadrature": "finite-lag-trapezoid",
            "quadrature_convergence_verified": False,
        }

    def integrate_from_selected_spectra(self, grid, greater_displacement, greater_momentum,
                                        lag_times, *, block_size=16):
        r"""Stream C=iG> and P=dC/dt from explicitly selected signed spectra.

        Spectra have shapes (Nw,J,A), (Nw,I,A) in this plan's index order.
        Both inputs are centered displacement G>, without a mean delta at zero.
        greater_momentum selects the rows where momentum will be taken; it is
        not a pre-differentiated momentum Green function. This method applies
        that time derivative itself through the factor omega.
        This interface also supports selected fixed-field Green blocks without
        retaining full device Green matrices. It performs only Fourier sums.
        """
        if not isinstance(grid, SignedFrequencyGrid):
            raise TypeError("grid must be a SignedFrequencyGrid.")
        lag = _lags(lag_times)
        recurrence = 2.0 * np.pi / grid.step
        if lag[-1] >= recurrence / 2.0:
            raise ValueError("lag window must be shorter than half the discrete-frequency recurrence.")
        a, j, i = len(self.active_indices), len(self.displacement_rows), len(self.momentum_rows)
        displacement = _complex(greater_displacement, (grid.n_points, j, a), "selected G> displacement")
        momentum = _complex(greater_momentum, (grid.n_points, i, a), "selected G> momentum")
        displacement = displacement.reshape(grid.n_points, j * a)
        momentum = momentum.reshape(grid.n_points, i * a)

        def provider(times):
            phase = np.exp(-1j * np.outer(times, grid.values))
            c = (1j * grid.integration_weight) * (phase @ displacement)
            p = grid.integration_weight * ((phase * grid.values) @ momentum)
            return c.reshape(len(times), j, a), p.reshape(len(times), i, a)

        report = self.integrate(lag, provider, block_size=block_size)
        report.update({"frequency_quadrature": "stored-signed-grid-rectangle",
                       "recurrence_period": float(recurrence),
                       "input_centering_verified": False, "frequency_accuracy_verified": False})
        return report

    def integrate_from_green(self, grid, green, lag_times, *, block_size=16):
        """Select the needed G> entries once, then stream the Fourier blocks."""
        if not isinstance(grid, SignedFrequencyGrid) or not isinstance(green, KeldyshSpectrum):
            raise TypeError("grid and green must be signed-grid and Keldysh spectrum objects.")
        if green.shape != (grid.n_points, self.operator.ndof, self.operator.ndof):
            raise ValueError("Green function shape does not match the operator and grid.")
        a = self.active_indices
        displacement = green.greater[:, self.displacement_rows[:, None], a[None, :]]
        if np.array_equal(self.displacement_rows, self.momentum_rows):
            momentum = displacement
        else:
            momentum = green.greater[:, self.momentum_rows[:, None], a[None, :]]
        return self.integrate_from_selected_spectra(grid, displacement, momentum,
                                                     lag_times, block_size=block_size)


def build_contracted_cubic_cut(problem, cut, *, max_dense_tensor_bytes=256 * 2**20):
    """Compile one cut with a dense tensor only on active vertex coordinates.

    max_dense_tensor_bytes caps the persistent complex response tensor before
    allocation. It is not a process-RSS cap; transform/provider buffers and
    BLAS-internal scratch are separate. The sparse input is never densified
    on the full device. The polynomial derivative sums all ordered entries.
    """
    limit = _positive_integer(max_dense_tensor_bytes, "max_dense_tensor_bytes")
    operator = build_cubic_cut_current_operator(problem, cut)
    _, monomials, coefficients = _potential(problem.vertex)
    if len(operator.cubic_indices):
        active = np.unique(monomials)
        displacement_rows = np.unique(operator.cubic_indices[:, 1:])
        momentum_rows = np.unique(operator.cubic_indices[:, 0])
    else:
        active = displacement_rows = momentum_rows = np.empty(0, dtype=np.int64)
    a = len(active)
    needed = 16 * a**3
    if needed > limit:
        raise MemoryError(f"Active cubic response tensor needs {needed} bytes; limit is {limit}.")
    response = np.zeros((a, a, a), dtype=np.complex128)
    if a:
        local = np.searchsorted(active, monomials)
        for order in permutations(range(3)):
            np.add.at(response, tuple(local[:, order].T), coefficients)
    if not np.all(np.isfinite(response)):
        raise ValueError("Response tensor coefficient accumulation overflowed.")
    momentum_positions = np.searchsorted(momentum_rows, operator.cubic_indices[:, 0])
    displacement_positions = np.searchsorted(displacement_rows, operator.cubic_indices[:, 1:])
    return ContractedCubicCut(operator, *map(_readonly, (
        active, displacement_rows, momentum_rows, response,
        momentum_positions, displacement_positions)))


__all__ = ["ContractedCubicCut", "build_contracted_cubic_cut"]
