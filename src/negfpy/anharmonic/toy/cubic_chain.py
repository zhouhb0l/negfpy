r"""Pinned cubic-chain fixtures for validating anharmonic NEGF.

The harmonic blocks use mass-weighted coordinates and describe

.. math::

   V_2 = \sum_i \frac{p x_i^2}{2}
       + \sum_{\langle i,j\rangle}\frac{k(x_j-x_i)^2}{2}.

Device end sites retain the bulk onsite term because the missing neighbors are
supplied by their harmonic reservoirs.

Three deliberately small cubic interactions are provided:

* an onsite scalar term ``cubic * x_i**3 / 3!``;
* an intralayer mass-weighted term ``g * u_x * u_y * u_z``;
* selected FPU-alpha bonds ``alpha * (x_j - x_i)**3 / 3!``.

All returned :class:`~negfpy.anharmonic.types.DenseCubicVertex` objects are
mass-normalized and use the convention
``H3 = sum(Psi[a,b,c] * u[a] * u[b] * u[c]) / 3!``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import permutations, product
from typing import Any

import numpy as np

from negfpy.core.types import Device1D, LeadBlocks

from ..types import DenseCubicVertex


Array = np.ndarray
ToyCubicModel = tuple[Device1D, LeadBlocks, DenseCubicVertex]
Bond = tuple[int, int]


def _positive_int(value: Any, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be at least one.")
    return result


def _finite_scalar(
    value: Any,
    *,
    name: str,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)) or np.asarray(value).ndim != 0:
        raise TypeError(f"{name} must be a scalar real number.")
    if np.iscomplexobj(value):
        raise TypeError(f"{name} must be real.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if strictly_positive and result <= 0.0:
        raise ValueError(f"{name} must be strictly positive.")
    return result


def _positive_three_vector(values: Any, *, name: str) -> Array:
    if isinstance(values, (bool, np.bool_)) or np.iscomplexobj(values):
        raise TypeError(f"{name} must be real.")
    raw = np.asarray(values)
    if raw.ndim == 0:
        result = np.full(3, float(raw), dtype=np.float64)
    elif raw.shape == (3,):
        result = np.asarray(raw, dtype=np.float64)
    else:
        raise ValueError(f"{name} must be a scalar or a length-three sequence.")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} entries must be finite and strictly positive.")
    return result


def _scalar_harmonic_chain(
    *,
    n_layers: int,
    mass: float,
    spring: float,
    pinning: float,
) -> tuple[Device1D, LeadBlocks]:
    onsite_value = (2.0 * spring + pinning) / mass
    coupling_value = -spring / mass
    onsite = np.array([[onsite_value]], dtype=np.float64)
    coupling = np.array([[coupling_value]], dtype=np.float64)
    lead = LeadBlocks(d00=onsite.copy(), d01=coupling.copy())
    device = Device1D(
        onsite_blocks=[onsite.copy() for _ in range(n_layers)],
        coupling_blocks=[coupling.copy() for _ in range(n_layers - 1)],
    )
    return device, lead


def _common_metadata(
    *,
    model: str,
    topology: str,
    n_layers: int,
    harmonic_bond_current_supported: bool,
) -> dict[str, Any]:
    return {
        "model": model,
        "interaction_topology": topology,
        "n_layers": n_layers,
        "mass_weighted": True,
        "cubic_convention": "H3=(1/3!)*sum_abc(Psi_abc*u_a*u_b*u_c)",
        "harmonic_bond_current_supported": harmonic_bond_current_supported,
    }


def pinned_scalar_chain(
    n_layers: int,
    *,
    mass: float = 1.0,
    spring: float = 1.0,
    pinning: float = 0.25,
    cubic: float = 0.0,
) -> ToyCubicModel:
    r"""Return a pinned scalar chain with onsite cubic interactions.

    ``cubic`` is the coefficient in physical displacement coordinates,
    ``cubic * x_i**3 / 3!``.  The returned mass-weighted vertex therefore has
    ``Psi[i,i,i] = cubic / mass**(3/2)``.
    """

    n_layers = _positive_int(n_layers, name="n_layers")
    mass = _finite_scalar(mass, name="mass", strictly_positive=True)
    spring = _finite_scalar(spring, name="spring", strictly_positive=True)
    pinning = _finite_scalar(pinning, name="pinning", strictly_positive=True)
    cubic = _finite_scalar(cubic, name="cubic")

    device, lead = _scalar_harmonic_chain(
        n_layers=n_layers,
        mass=mass,
        spring=spring,
        pinning=pinning,
    )
    mass_weighted_cubic = cubic / mass**1.5
    values = np.zeros((n_layers, n_layers, n_layers), dtype=np.float64)
    diagonal = np.arange(n_layers)
    values[diagonal, diagonal, diagonal] = mass_weighted_cubic

    metadata = _common_metadata(
        model="pinned_scalar_onsite_cubic",
        topology="onsite",
        n_layers=n_layers,
        harmonic_bond_current_supported=True,
    )
    metadata.update(
        {
            "active_sites": tuple(range(n_layers)),
            "coordinate_cubic": cubic,
            "mass_weighted_cubic": mass_weighted_cubic,
            "mass": mass,
            "spring": spring,
            "pinning": pinning,
        }
    )
    return device, lead, DenseCubicVertex(values=values, metadata=metadata)


def pinned_three_polarization_chain(
    n_layers: int,
    *,
    mass: float | Sequence[float] | Array = 1.0,
    spring: float | Sequence[float] | Array = 1.0,
    pinning: float | Sequence[float] | Array = 0.25,
    g: float = 0.0,
) -> ToyCubicModel:
    r"""Return a pinned chain with three independent harmonic polarizations.

    The layer-major degree-of-freedom order is ``(x, y, z)``.  ``mass``,
    ``spring``, and ``pinning`` may be scalars or length-three sequences.

    The interaction on every layer is already specified in mass-weighted
    coordinates as ``g * u_x * u_y * u_z``.  Accordingly, each of the six
    permutations of ``Psi[x,y,z]`` equals ``g``.
    """

    n_layers = _positive_int(n_layers, name="n_layers")
    masses = _positive_three_vector(mass, name="mass")
    springs = _positive_three_vector(spring, name="spring")
    pinnings = _positive_three_vector(pinning, name="pinning")
    g = _finite_scalar(g, name="g")

    onsite = np.diag((2.0 * springs + pinnings) / masses).astype(np.float64)
    coupling = np.diag(-springs / masses).astype(np.float64)
    lead = LeadBlocks(d00=onsite.copy(), d01=coupling.copy())
    device = Device1D(
        onsite_blocks=[onsite.copy() for _ in range(n_layers)],
        coupling_blocks=[coupling.copy() for _ in range(n_layers - 1)],
    )

    ndof = 3 * n_layers
    values = np.zeros((ndof, ndof, ndof), dtype=np.float64)
    xyz_permutations = tuple(permutations((0, 1, 2)))
    for layer in range(n_layers):
        offset = 3 * layer
        for local_indices in xyz_permutations:
            indices = tuple(offset + value for value in local_indices)
            values[indices] = g

    metadata = _common_metadata(
        model="pinned_three_polarization_intralayer_cubic",
        topology="intralayer",
        n_layers=n_layers,
        harmonic_bond_current_supported=True,
    )
    metadata.update(
        {
            "dof_order": ("x", "y", "z"),
            "active_layers": tuple(range(n_layers)),
            "mass_weighted_xyz_coupling": g,
            "mass": tuple(float(value) for value in masses),
            "spring": tuple(float(value) for value in springs),
            "pinning": tuple(float(value) for value in pinnings),
        }
    )
    return device, lead, DenseCubicVertex(values=values, metadata=metadata)


def _normalize_bonds(
    interaction_bonds: Sequence[Bond] | None,
    *,
    n_layers: int,
) -> tuple[Bond, ...]:
    if interaction_bonds is None:
        return tuple((left, left + 1) for left in range(n_layers - 1))
    if isinstance(interaction_bonds, (str, bytes)):
        raise TypeError("interaction_bonds must be a sequence of integer pairs.")

    normalized: list[Bond] = []
    for position, bond in enumerate(interaction_bonds):
        if isinstance(bond, (str, bytes)):
            raise TypeError(
                f"interaction_bonds[{position}] must be an integer pair."
            )
        try:
            left, right = bond
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"interaction_bonds[{position}] must contain exactly two indices."
            ) from error
        if (
            isinstance(left, (bool, np.bool_))
            or isinstance(right, (bool, np.bool_))
            or not isinstance(left, (int, np.integer))
            or not isinstance(right, (int, np.integer))
        ):
            raise TypeError("interaction bond indices must be integers.")
        pair = (int(left), int(right))
        if pair[0] < 0 or pair[1] >= n_layers:
            raise ValueError("interaction bond indices are outside the finite device.")
        if pair[1] != pair[0] + 1:
            raise ValueError(
                "FPU-alpha interaction bonds must be oriented nearest-neighbor pairs (i, i+1)."
            )
        normalized.append(pair)

    if len(set(normalized)) != len(normalized):
        raise ValueError("interaction_bonds cannot contain duplicate pairs.")
    return tuple(normalized)


def fpu_alpha_chain(
    n_layers: int,
    *,
    mass: float = 1.0,
    spring: float = 1.0,
    pinning: float = 0.25,
    alpha: float = 0.0,
    interaction_bonds: Sequence[Bond] | None = None,
) -> ToyCubicModel:
    r"""Return a pinned scalar chain with selected FPU-alpha device bonds.

    Every selected pair contributes

    ``alpha * (x_j - x_i)**3 / 3!``.

    Since ``x_i = u_i / sqrt(mass)``, the ordered vertex contribution is
    ``alpha * s[a] * s[b] * s[c] / mass**(3/2)``, where ``s_i=-1`` and
    ``s_j=+1``.  Overlapping bonds are accumulated exactly.
    """

    n_layers = _positive_int(n_layers, name="n_layers")
    if n_layers < 2:
        raise ValueError("an FPU-alpha chain must contain at least two layers.")
    mass = _finite_scalar(mass, name="mass", strictly_positive=True)
    spring = _finite_scalar(spring, name="spring", strictly_positive=True)
    pinning = _finite_scalar(pinning, name="pinning", strictly_positive=True)
    alpha = _finite_scalar(alpha, name="alpha")
    bonds = _normalize_bonds(interaction_bonds, n_layers=n_layers)

    device, lead = _scalar_harmonic_chain(
        n_layers=n_layers,
        mass=mass,
        spring=spring,
        pinning=pinning,
    )
    mass_weighted_alpha = alpha / mass**1.5
    values = np.zeros((n_layers, n_layers, n_layers), dtype=np.float64)
    for left, right in bonds:
        signs = {left: -1.0, right: 1.0}
        for indices in product((left, right), repeat=3):
            values[indices] += (
                mass_weighted_alpha
                * signs[indices[0]]
                * signs[indices[1]]
                * signs[indices[2]]
            )

    metadata = _common_metadata(
        model="pinned_fpu_alpha_bond_cubic",
        topology="bond",
        n_layers=n_layers,
        harmonic_bond_current_supported=False,
    )
    metadata.update(
        {
            "interaction_bonds": bonds,
            "coordinate_alpha": alpha,
            "mass_weighted_alpha": mass_weighted_alpha,
            "mass": mass,
            "spring": spring,
            "pinning": pinning,
        }
    )
    return device, lead, DenseCubicVertex(values=values, metadata=metadata)


__all__ = [
    "ToyCubicModel",
    "fpu_alpha_chain",
    "pinned_scalar_chain",
    "pinned_three_polarization_chain",
]
