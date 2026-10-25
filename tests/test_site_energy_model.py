from itertools import permutations

import numpy as np

from negfpy.anharmonic.local_energy import _design, _stars
from negfpy.anharmonic.site_energy_model import fit_site_energy_model


def test_compute_model_coefficients_reconstruct_analytic_periodic_bond_energy():
    n = 4
    positions = np.array([[.95, .2, .2], [.05, .2, .2], [.45, .2, .2], [.55, .2, .2]])
    cell, labels = np.eye(3)*10, np.array([7, 13, 7, 13])
    source = np.zeros((n, n, n, 3, 3, 3))
    for first, second in ((0, 1), (2, 3)):
        x, y = np.zeros((n, 3)), np.zeros((n, 3))
        x[first, 0], x[second, 0], y[first, 1], y[second, 1] = -1., 1., -1., 1.
        vectors = (x.ravel(), y.ravel(), y.ravel())
        flat = sum((np.einsum("i,j,k->ijk", *(vectors[index] for index in order)) for order in permutations(range(3))), np.zeros((n*3,)*3))
        source += .7/6*flat.reshape(n, 3, n, 3, n, 3).transpose(0, 2, 4, 1, 3, 5)
    original = source.copy()
    model = fit_site_energy_model(source, positions, cell, labels, neighbors=1)
    assert model.coefficients.shape == (2, 10)
    assert model.fit_report["fit"]["source_reconstruction_passed"]
    assert model.stars[0].neighbor_image_integers.tolist() == [[1, 0, 0]]
    np.testing.assert_array_equal(model.class_labels, [7, 13])
    np.testing.assert_array_equal(source, original)
    assert not model.coefficients.flags.writeable
    assert not model.stars[0].neighbor_cartesian_vectors.flags.writeable
    stars, _ = _stars(positions, cell, model.classes, 1, 1e-8, 1e-11, lambda: None)
    design, rows, _ = _design(n, stars, model.classes, 1, lambda: None)
    reconstructed = np.zeros(source.shape)
    reconstructed.ravel()[rows] = design@model.coefficients.ravel()
    np.testing.assert_allclose(reconstructed, source, atol=1e-13)
    displacement = np.array([[.03, -.01, .02], [.01, .04, -.02], [-.02, .05, .01], [.02, -.03, .04]])
    energy = 0.
    for star in model.stars:
        relative = (displacement[star.neighbors]-displacement[star.center]).ravel()
        for coefficient, (p, q, r) in zip(model.coefficients[star.class_index], model.monomials):
            divisor = 6 if p == r else (2 if p == q or q == r else 1)
            energy += coefficient*relative[p]*relative[q]*relative[r]/divisor
    expected = sum(.7/6*(displacement[b, 0]-displacement[a, 0])*(displacement[b, 1]-displacement[a, 1])**2 for a, b in ((0, 1), (2, 3)))
    np.testing.assert_allclose(energy, expected, atol=1e-18)


def test_model_reports_omitted_remainder_without_adoption():
    source = np.zeros((2, 2, 2, 3, 3, 3))
    source[0, 0, 0, 2, 2, 2] = 1.
    model = fit_site_energy_model(source, [[.1, 0, 0], [.2, 0, 0]], np.eye(3)*10, [0, 1], neighbors=1)
    assert not model.fit_report["fit"]["source_reconstruction_passed"]
    assert model.fit_report["residual"]["frobenius"] > 0.
    assert model.fit_report["residual"]["sum_closure"]["passed"]
    assert model.fit_report["scope"]["surrogate_omits_measured_remainder"]
    assert not model.fit_report["residual"]["included_in_surrogate_vertex"]
    assert not model.fit_report["scope"]["physical_model_adopted"]
