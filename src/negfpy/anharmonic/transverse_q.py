r"""Direct coupled-q cubic bubble for small transverse-periodic reference models.

With ``u_R = Nq**(-1/2) sum_q exp(i*q.R) u_q``, store the reduced
vertex ``V[q,p,a,c,d] = sum_s,t Psi[0a,sc,td] exp(i*(p.s+(q-p).t))``.
It contains no mesh normalization. The bubble is

``Sigma[q,omega] = i/(2*Nq) sum_p integral V[q,p] G[p,omega']
                  G[q-p,omega-omega'] conj(V[q,p]) domega'/(2*pi)``.

Momentum differences wrap on the transverse mesh; frequency differences are
zero outside the signed grid. This deterministic dense reference is a Born
contraction, not a coupled-q SCBA solver or a material IFC Fourier mapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh

_MAX_ARRAY_BYTES = 64 * 1024**2


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be a positive integer.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _mesh(mesh: UniformBZMesh) -> None:
    if not isinstance(mesh, UniformBZMesh):
        raise TypeError("mesh must be a UniformBZMesh.")
    if mesh.shape[2] != 1:
        raise ValueError("The transverse prototype requires mesh.shape[2] == 1.")


def difference_q_indices(
    mesh: UniformBZMesh, *, max_bytes: int = _MAX_ARRAY_BYTES,
) -> np.ndarray:
    """Return immutable X-fast indices ``result[q,p] = index(q-p)``.

    The first two mesh axes are periodic transverse directions; the third has
    period one. Routing uses exact integer addresses, including Nyquist points.
    """
    _mesh(mesh)
    limit = _positive_integer(max_bytes, "max_bytes")
    n_q = mesh.n_qpoints
    if 8 * n_q**2 > limit:
        raise ValueError("Momentum routing table exceeds max_bytes.")
    result = np.empty((n_q, n_q), dtype=np.int64)
    addresses = mesh.addresses
    for q, output in enumerate(addresses):
        for p, internal in enumerate(addresses):
            result[q, p] = mesh.index_from_address(
                tuple(a - b for a, b in zip(output, internal, strict=True))
            )
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True, eq=False)
class TransverseCubicVertex:
    """Reduced complex vertex of a real, permutation-symmetric cubic potential.

    ``coefficients`` has shape ``(Nq, Nq, D, D, D)`` and is indexed by
    ``[q_output, q_internal, a, c, d]``. The second internal momentum is
    ``q_output-q_internal``. Reality and two generating leg exchanges are
    checked without averaging, clipping or modifying the supplied coefficients.
    All stored values use the existing dimensionless solver-unit convention.
    """

    mesh: UniformBZMesh
    coefficients: np.ndarray = field(repr=False)
    max_bytes: int = field(default=_MAX_ARRAY_BYTES, repr=False)
    symmetry_rtol: ClassVar[float] = 1e-10
    symmetry_atol: ClassVar[float] = 1e-12

    def __post_init__(self) -> None:
        _mesh(self.mesh)
        limit = _positive_integer(self.max_bytes, "max_bytes")
        raw = np.asarray(self.coefficients)
        n_q = self.mesh.n_qpoints
        if (raw.ndim != 5 or raw.shape[:2] != (n_q, n_q)
                or len(set(raw.shape[2:])) != 1 or raw.shape[2] < 1):
            raise ValueError("coefficients must have shape (Nq, Nq, D, D, D), D >= 1.")
        if raw.size * np.dtype(np.complex128).itemsize > limit:
            raise ValueError("Cubic coefficients exceed max_bytes.")
        values = np.array(raw, dtype=np.complex128, order="C", copy=True)
        if not np.isfinite(values).all():
            raise ValueError("coefficients must contain only finite values.")
        routes = difference_q_indices(self.mesh, max_bytes=limit)
        negatives = [self.mesh.negative_index(q) for q in range(n_q)]
        for q in range(n_q):
            for p in range(n_q):
                block = values[q, p]
                checks = (
                    ("reality", values[negatives[q], negatives[p]].conj()),
                    ("internal-leg exchange", values[q, routes[q, p]].transpose(0, 2, 1)),
                    ("external-leg exchange", values[negatives[p], negatives[q]].transpose(1, 0, 2)),
                )
                for name, expected in checks:
                    if not np.allclose(block, expected, rtol=self.symmetry_rtol, atol=self.symmetry_atol):
                        raise ValueError(f"coefficients violate {name} at q={q}, p={p}.")
        values.setflags(write=False)
        object.__setattr__(self, "coefficients", values)
        object.__setattr__(self, "max_bytes", limit)

    @property
    def ndof(self) -> int:
        return self.coefficients.shape[2]


def _green(values: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.complex128)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}; got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def cubic_born_self_energy_q(
    grid: SignedFrequencyGrid,
    vertex: TransverseCubicVertex,
    g_lesser: np.ndarray,
    g_greater: np.ndarray,
    *,
    max_pair_evaluations: int = 100_000,
    max_output_bytes: int = _MAX_ARRAY_BYTES,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate both components on shape ``(Nq, Nomega, D, D)``.

    Sum every q' with weight 1/Nq and every supported frequency pair with the
    existing rectangle weight delta-omega/(2*pi). No Green matrix is conjugated
    or transposed in the contraction. The *second vertex* is conjugated.

    Budgets are checked before allocating outputs or doing contractions. The
    work limit counts (q,q',omega,omega') tuples per component; both components
    use that same partition. ``max_output_bytes`` bounds their combined payload,
    not process RSS. Inputs are not modified. Physical Green-function identities
    are not imposed here, allowing comparison with arbitrary algebraic fixtures.
    """
    if not isinstance(grid, SignedFrequencyGrid):
        raise TypeError("grid must be a SignedFrequencyGrid.")
    if not isinstance(vertex, TransverseCubicVertex):
        raise TypeError("vertex must be a TransverseCubicVertex.")
    pair_limit = _positive_integer(max_pair_evaluations, "max_pair_evaluations")
    byte_limit = _positive_integer(max_output_bytes, "max_output_bytes")
    n_q, n_w, d = vertex.mesh.n_qpoints, grid.n_points, vertex.ndof
    zero = grid.zero_index
    frequency_pairs = 3 * zero**2 + 3 * zero + 1
    if n_q**2 * frequency_pairs > pair_limit:
        raise ValueError("Direct coupled-q contraction exceeds max_pair_evaluations.")
    shape = (n_q, n_w, d, d)
    if 2 * n_q * n_w * d**2 * np.dtype(np.complex128).itemsize > byte_limit:
        raise ValueError("Lesser/greater output payload exceeds max_output_bytes.")
    greens = (_green(g_lesser, shape, "g_lesser"), _green(g_greater, shape, "g_greater"))
    routes = difference_q_indices(vertex.mesh)
    outputs = (np.zeros(shape, dtype=np.complex128), np.zeros(shape, dtype=np.complex128))
    prefactor = 0.5j * grid.integration_weight / n_q
    for q in range(n_q):
        for p in range(n_q):
            partner = routes[q, p]
            first = vertex.coefficients[q, p]
            second = first.conj()
            for omega in range(n_w):
                start = max(0, omega + zero - n_w + 1)
                stop = min(n_w - 1, omega + zero) + 1
                for internal in range(start, stop):
                    difference = omega - internal + zero
                    for green, output in zip(greens, outputs, strict=True):
                        left = np.einsum("acd,ce->aed", first, green[p, internal], optimize=True)
                        both = np.einsum("aed,df->aef", left, green[partner, difference], optimize=True)
                        output[q, omega] += prefactor * np.einsum(
                            "aef,bef->ab", both, second, optimize=True,
                        )
    if not all(np.isfinite(output).all() for output in outputs):
        raise ValueError("Coupled-q contraction produced nonfinite output.")
    return outputs


__all__ = ["TransverseCubicVertex", "cubic_born_self_energy_q", "difference_q_indices"]
