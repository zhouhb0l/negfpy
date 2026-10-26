"""Tiny saved-parameter reuse cases; no material tensor or numerical replay."""
from copy import deepcopy
from math import comb
from time import monotonic

import numpy as np
import pytest

from negfpy.anharmonic import local_energy, site_energy_model
from negfpy.anharmonic.local_energy import _classes, _stars
from negfpy.anharmonic.site_energy_saved import restore_site_energy_model


def _case():
    positions = np.array([[.95,.2,.2],[.05,.2,.2],[.45,.2,.2],[.55,.2,.2]])
    lattice, labels = np.eye(3)*10, np.array([7,13,7,13])
    classes, unique, counts = _classes(labels, 4)
    _, geometry = _stars(positions, lattice, classes, 1, 1e-8, 1e-11, lambda: None)
    monomials = comb(5,3)
    coefficients = np.arange(2*monomials, dtype=np.float64).reshape(2, monomials)
    coefficients[0,0] = -0.
    report = {'schema':'negfpy.compute-only-site-energy-model-fit.v1',
        'fit':{'parameter_count':coefficients.size,'bounded_solver_converged':True},
        'source':{'center_class_labels':unique.tolist(),'center_class_counts':counts.tolist()},
        'basis':{'neighbors_per_site':1,'monomials_per_class':monomials},
        'residual':{'relative_frobenius':.0214,'included_in_surrogate_vertex':False}}
    return coefficients, positions, lattice, labels, report, geometry


def _restore(case, **kwargs):
    coefficients, positions, lattice, labels, report, geometry = case
    return restore_site_energy_model(coefficients, positions, lattice, labels,
        fit_report=report, geometry=geometry, neighbors=1, **kwargs)


def test_saved_coefficients_keep_exact_bits_classes_and_geometry_without_any_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Restoration must not build a derivative design or fit parameters')
    monkeypatch.setattr(local_energy, '_design', forbidden)
    monkeypatch.setattr(site_energy_model, '_design', forbidden)
    monkeypatch.setattr(site_energy_model, 'lsmr', forbidden)
    case = _case()
    model = _restore(case)
    assert model.coefficients.tobytes() == case[0].tobytes()
    np.testing.assert_array_equal(model.class_labels, [7,13])
    assert model.geometry == case[-1] and model.fit_report == case[-2]
    assert model.stars[0].neighbor_image_integers.tolist() == [[1,0,0]]
    for value in (model.coefficients, model.classes, model.positions, model.lattice,
                  model.stars[0].neighbor_cartesian_vectors):
        assert not value.flags.writeable
    case[0][0,1] = 999.
    case[-2]['residual']['relative_frobenius'] = 99.
    assert model.coefficients[0,1] == 1.
    assert model.fit_report['residual']['relative_frobenius'] == .0214


def test_altered_preserved_geometry_or_class_basis_is_rejected():
    case = list(_case())
    case[-1] = deepcopy(case[-1])
    case[-1]['stars'][0]['neighbor_image_integers'][0][0] = 0
    with pytest.raises(ValueError, match='geometry differs'):
        _restore(case)
    case = list(_case())
    case[-2]['source']['center_class_labels'] = [0,1]
    with pytest.raises(ValueError, match='basis/classes'):
        _restore(case)


@pytest.mark.parametrize('bad', [np.zeros((1,10)), np.zeros((2,10), dtype=np.float32),
                               np.full((2,10), np.nan)])
def test_saved_coefficient_shape_dtype_and_finiteness_are_not_coerced(bad):
    case = list(_case())
    case[0] = bad
    with pytest.raises(ValueError, match='coefficients'):
        _restore(case)


def test_geometry_work_admission_and_deadline_stop_before_restoration():
    with pytest.raises(ValueError, match='workspace admission'):
        _restore(_case(), max_bytes=1)
    with pytest.raises(TimeoutError, match='deadline'):
        _restore(_case(), deadline_monotonic=monotonic()-1)
