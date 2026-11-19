"""Generated controls for the archived-Hessian conversion operation order.

The oracle below follows the original ImageFC2Harmonics.from_force_constants
and its independent cell-pair assembly. It reads no campaign data or potential.
"""
import numpy as np
import pytest

from negfpy.anharmonic import image_harmonic as h


def _raw_images():
    # Mixed masses and non-diagonal endpoint blocks expose accidental changes
    # to unit association, basis order, image sign or Fourier accumulation.
    b = np.arange(36, dtype=float).reshape(6, 6) / 37. - .23
    onsite = b + b.T + 5.9*np.eye(6)
    one = b*.13 - .37*np.eye(6)
    two = b.T*.071 + .029*np.eye(6)
    return {(2, -1, 1): two, (0, 0, 0): onsite,
            (-1, -1, 0): one.T, (1, 1, 0): one,
            (-2, 1, -1): two.T}


def _native_images(images, masses, frequency):
    # Original producer sequence, including a separate sqrt for each mass.
    masses_si = np.repeat(np.asarray(masses, dtype=float)*h.AMU_KG, 3)
    denominator = (np.sqrt(masses_si[:, None])*np.sqrt(masses_si[None, :])
                   * np.float64(frequency)**2)
    force_factor = h.EV_J/h.ANGSTROM_M**2
    return {key: np.asarray(images[key], dtype=np.float64)*force_factor/denominator
            for key in sorted(images)}


def _native_at_q(images, point, cells):
    dofs = next(iter(images.values())).shape[0]
    blocks = {shift: np.zeros((dofs, dofs), complex) for shift in range(-2, 3)}
    for (dx, dy, dz), values in images.items():
        blocks[dx] += values*np.exp(1j*(point[0]*dy+point[1]*dz))
    def assemble(rows, columns):
        result = np.zeros((len(rows)*dofs, len(columns)*dofs), complex)
        for i, row in enumerate(rows):
            for j, column in enumerate(columns):
                if column-row in blocks:
                    result[i*dofs:(i+1)*dofs, j*dofs:(j+1)*dofs] = blocks[column-row]
        return result
    return (assemble(tuple(range(cells)), tuple(range(cells))),
            assemble((0, 1), (0, 1)), assemble((0, 1), (2, 3)),
            assemble((0, 1), (-2, -1)), assemble((0, 1), (-2, -1)),
            assemble((cells-2, cells-1), (cells, cells+1)))


def test_default_unit_conversion_preserves_original_material_test_expression():
    images, masses = _raw_images(), np.array([12.34567, 95.95123])
    frequency = 6.283185307179586e13
    default = h.ImageHarmonics(images, masses, np.eye(3), omega0_rad_s=frequency)
    explicit = h.ImageHarmonics(images, masses, np.eye(3), omega0_rad_s=frequency,
                               unit_conversion="material-test")
    denominator = np.sqrt(np.repeat(masses, 3)[:, None]*np.repeat(masses, 3)[None, :])
    factor = h.EV_J/h.ANGSTROM_M**2/h.AMU_KG/frequency**2
    for key, raw in images.items():
        np.testing.assert_array_equal(default.image_dynamical[key], raw/denominator*factor)
        np.testing.assert_array_equal(default.image_dynamical[key], explicit.image_dynamical[key])
    assert default.unit_conversion == "material-test"


@pytest.mark.parametrize("cells", [3, 9])
@pytest.mark.parametrize("point", [(0., 0.), (.37, -.61), (-.37, .61)])
def test_archived_hessian_conversion_full_device_and_leads_match_original_order(cells, point):
    images, masses = _raw_images(), np.array([28.0855, 95.95123])
    frequency = 6.283185307179586e13
    original = _native_images(images, masses, frequency)
    model = h.ImageHarmonics(images, masses, np.eye(3), omega0_rad_s=frequency,
                            unit_conversion="native-diagonal-mass")
    assert tuple(model.image_dynamical) == tuple(sorted(images))
    for key, value in original.items():
        np.testing.assert_array_equal(model.image_dynamical[key], value)
        assert not model.image_dynamical[key].flags.writeable
    observed = model.at_q(*point, cells)
    values = (observed.dynamical, observed.lead.d00, observed.lead.d01,
              observed.lead.d10, observed.device_to_lead_left, observed.device_to_lead_right)
    for actual, expected in zip(values, _native_at_q(original, point, cells), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(observed.contact_left_indices, np.arange(12))
    np.testing.assert_array_equal(observed.contact_right_indices, np.arange((cells-2)*6, cells*6))
    if cells == 3:
        assert set(observed.contact_left_indices) & set(observed.contact_right_indices)


def test_native_conversion_retains_transverse_gauge_and_opposite_q_relations():
    model = h.ImageHarmonics(_raw_images(), [28.0855, 95.95123], np.eye(3),
                            unit_conversion="native-diagonal-mass")
    q = (.37, -.61)
    point = model.at_q(*q, 3)
    negative = model.at_q(-q[0], -q[1], 3)
    periodic = model.at_q(q[0]+2*np.pi, q[1]-2*np.pi, 3)
    for extract in (lambda p:p.dynamical, lambda p:p.lead.d00, lambda p:p.lead.d01,
                    lambda p:p.lead.d10, lambda p:p.device_to_lead_left,
                    lambda p:p.device_to_lead_right):
        np.testing.assert_array_equal(extract(point).conj(), extract(negative))
        np.testing.assert_allclose(extract(point), extract(periodic), rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize("value", ["native", "", None, 17, [], {}])
def test_unknown_conversion_fails_closed(value):
    with pytest.raises(ValueError, match="unit_conversion"):
        h.ImageHarmonics(_raw_images(), [28.0855, 95.95123], np.eye(3), unit_conversion=value)


def test_native_mass_conversion_rejects_unrepresentable_denominator():
    with pytest.raises(ValueError, match="not representable"):
        h.ImageHarmonics(_raw_images(), [1e-300, 1e-300], np.eye(3),
                        unit_conversion="native-diagonal-mass")
