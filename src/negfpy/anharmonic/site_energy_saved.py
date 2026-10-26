"""Restore a declared site-energy model from its saved coefficients, without fitting."""
from __future__ import annotations

import json
from copy import deepcopy
from itertools import combinations_with_replacement
from math import comb
from time import monotonic

import numpy as np

from .local_energy import _classes, _finite, _integer, _stars
from .site_energy_model import SiteEnergyModel, SiteEnergyStar, _readonly


def _canonical(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Preserved site-energy metadata must be finite JSON data.") from exc


def restore_site_energy_model(coefficients, positions, lattice, center_classes, *,
        fit_report, geometry, neighbors=4, max_atoms=128, max_bytes=2*1024**3,
        identity_rtol=1e-8, identity_atol=1e-11, deadline_monotonic=None):
    """Rebuild only the deterministic basis around the exact saved parameters.

    The caller authenticates the coefficient file and historical report first.
    Geometry must equal the preserved fit geometry exactly; it is never refitted,
    rephased or repaired. The old fit report remains historical evidence. No FC3
    tensor, sparse derivative design, or least-squares solver is used here.
    """
    positions, lattice = np.asarray(positions, dtype=float), np.asarray(lattice, dtype=float)
    if positions.ndim != 2 or positions.shape[1:] != (3,) or lattice.shape != (3, 3):
        raise ValueError("Positions and lattice require fractional[n,3] and row-vector[3,3] shapes.")
    n = len(positions)
    if n < 2 or n > _integer(max_atoms, "max_atoms", 2):
        raise ValueError("Saved-model atom-count admission exceeded.")
    neighbors = _integer(neighbors, "neighbors")
    if neighbors >= n:
        raise ValueError("Neighbors must be fewer than atoms.")
    classes, labels, counts = _classes(center_classes, n)
    monomial_count = comb(3*neighbors+2, 3)
    if (not isinstance(coefficients, np.ndarray) or coefficients.dtype != np.dtype('float64')
            or coefficients.shape != (len(labels), monomial_count)):
        raise ValueError("Saved coefficients require finite float64[class,monomial] shape.")
    _finite(coefficients, "saved coefficients")
    _finite(positions, "positions")
    _finite(lattice, "lattice")
    for value, name in ((identity_rtol, "identity_rtol"), (identity_atol, "identity_atol")):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    # Conservative numeric workspace for the existing bounded image enumeration;
    # Python object overhead and library scratch are separate, not an RSS bound.
    owned = 512*n*n+256*n*(neighbors+1)+256*len(labels)*monomial_count+2**20
    if owned > _integer(max_bytes, "max_bytes"):
        raise ValueError("Saved-model geometry workspace admission exceeded.")
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):
        raise ValueError("deadline_monotonic must be finite when supplied.")

    def admit():
        if deadline_monotonic is not None and monotonic() >= deadline_monotonic:
            raise TimeoutError("Saved-model restoration reached its bounded deadline.")

    admit()
    report_text = _canonical(fit_report)
    if (not isinstance(fit_report, dict)
            or fit_report.get('schema') != 'negfpy.compute-only-site-energy-model-fit.v1'
            or fit_report.get('fit', {}).get('parameter_count') != coefficients.size
            or fit_report.get('fit', {}).get('bounded_solver_converged') is not True
            or fit_report.get('source', {}).get('center_class_labels') != labels.tolist()
            or fit_report.get('source', {}).get('center_class_counts') != counts.tolist()
            or fit_report.get('basis', {}).get('neighbors_per_site') != neighbors
            or fit_report.get('basis', {}).get('monomials_per_class') != monomial_count):
        raise ValueError("Preserved fit report differs from the saved coefficient basis/classes.")
    _, rebuilt = _stars(positions, lattice, classes, neighbors,
                        identity_rtol, identity_atol, admit)
    if _canonical(geometry) != _canonical(rebuilt):
        raise ValueError("Rebuilt site-star geometry differs from the preserved fit geometry.")
    stars = tuple(SiteEnergyStar(center=int(item['center']), class_index=int(item['class_index']),
        neighbors=_readonly(item['neighbors'], np.int64),
        neighbor_image_integers=_readonly(item['neighbor_image_integers'], np.int64),
        neighbor_cartesian_vectors=_readonly(item['neighbor_cartesian_vectors'], np.float64))
        for item in rebuilt['stars'])
    admit()
    return SiteEnergyModel(coefficients=_readonly(coefficients),
        monomials=tuple(combinations_with_replacement(range(3*neighbors), 3)),
        classes=_readonly(classes, np.int64), class_labels=_readonly(labels, np.int64),
        class_counts=_readonly(counts, np.int64), stars=stars,
        positions=_readonly(positions, np.float64), lattice=_readonly(lattice, np.float64),
        geometry=deepcopy(rebuilt), fit_report=json.loads(report_text))


__all__ = ['restore_site_energy_model']
