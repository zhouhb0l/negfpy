"""Independent tiny fixed-input checks; no saved material or checkpoint loads."""

from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest

import negfpy.anharmonic.transverse_native_zero_diagnostic as diagnostic
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_contacts import TransverseContact
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex
from negfpy.anharmonic.transverse_solver import TransverseSpectrum

ELEMENTS = tuple(product(range(2), repeat=2))


def _fixture(nq=3, npositive=2, huge=False):
    mesh, grid = UniformBZMesh((nq, 1, 1)), SignedFrequencyGrid(.17, npositive)
    factors = np.empty((nq, 2), complex)
    for q, a in product(range(nq), range(2)):
        factors[q, a] = (a+1)/3 + np.exp(2j*np.pi*q/nq)*(a+2)/7
    coefficients = np.empty((nq, nq, 2, 2, 2), complex)
    for q, p, a, c, d in product(range(nq), range(nq), range(2), range(2), range(2)):
        coefficients[q, p, a, c, d] = (
            factors[(-q) % nq, a]*factors[p, c]*factors[(q-p) % nq, d])
    vertex = PackedTransverseCubicVertex(mesh, coefficients, np.array([2, 0]), 3)
    dynamical = np.tile(np.diag([1.3, 1.6, 1.9]), (nq, 1, 1)).astype(complex)
    for q in range(nq):
        phase = np.exp(2j*np.pi*q/nq)
        dynamical[q, 0, 1] = -.12-.035*phase
        dynamical[q, 1, 0] = dynamical[q, 0, 1].conjugate()
        dynamical[q, 0, 2] = -.04*phase
        dynamical[q, 2, 0] = dynamical[q, 0, 2].conjugate()
    contacts = []
    for indices, side, temp, damping in (((0, 1), "left", .91, .31),
                                          ((1, 2), "right", .57, .23)):
        weight = np.empty((nq, 2, 2), complex)
        for q in range(nq):
            phase = np.exp(2j*np.pi*q/nq)
            weight[q] = damping*np.array([[1.2, .19*phase], [.19*phase.conjugate(), .8]])
        r = -1j*grid.values[None, :, None, None]*weight[:, None]
        c = np.broadcast_to(-2j*temp*weight[:, None], r.shape)
        d = r-r.conj().swapaxes(-1, -2)
        spectrum = TransverseSpectrum(r, c-d/2, c+d/2)
        contact = TransverseContact(mesh, grid, spectrum, 2*weight,
                                    tuple({} for _ in range(nq)), {}, side, .0015625, temp)
        contacts.append(CompactTransverseContact(contact, 3, indices))
    shape = (nq, grid.n_points, 2, 2)
    r, c, d = (np.zeros(shape, complex) for _ in range(3))
    for q, w in product(range(nq), range(grid.n_points)):
        phase = np.exp(2j*np.pi*q/nq)
        weight = np.array([[1, .17*phase], [.17*phase.conjugate(), .7]])
        r[q, w] = -.003j*grid.values[w]*weight
        c[q, w] = -1j*(2**56 if huge else .02)*weight
        d[q, w] = -.002j*grid.values[w]*weight + .0002*np.array([[0, 1], [-1, 0]])
    sigma = ChannelSpectrum(r, c, d)
    mapped = ChannelSpectrum(*(np.full(shape, (i+1)*(.3+.2j), complex) for i in range(3)))
    return grid, vertex, dynamical, *contacts, sigma, mapped


def _run(args, **kwargs):
    return diagnostic.study_native_zero_diagnostic(
        *args, selected_elements=ELEMENTS, workers=1, external_tile=1, **kwargs)


def _matrix(summary):
    result = np.empty((2, 2), complex)
    for row in summary["selected_elements"]:
        result[row["row"], row["column"]] = row["real"]+1j*row["imag"]
    return result


def _literal(args, g, width):
    """Independent direct Green assembly and six-index native Born contraction."""
    grid, vertex, dynamical, left, right, sigma, _ = args
    nq, z = vertex.mesh.n_qpoints, grid.zero_index
    indices = list(range(z-width, z+width+1))
    c, d = (np.zeros((nq, len(indices), 2, 2), complex) for _ in range(2))
    greens = {}
    for q, local in product(range(nq), range(len(indices))):
        w = indices[local]
        operator = grid.values[w]**2*np.eye(3)-dynamical[q]
        cs, ds = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
        for contact in (left, right):
            support = contact.device_indices
            for i, j in product(range(2), repeat=2):
                target = support[i], support[j]
                operator[target] -= contact.spectrum.retarded[q, w, i, j]
                cs[target] += (contact.spectrum.lesser[q, w, i, j]
                               + contact.spectrum.greater[q, w, i, j])/2
                ds[target] += (contact.spectrum.greater[q, w, i, j]
                               - contact.spectrum.lesser[q, w, i, j])
        for i, j in product(range(2), repeat=2):
            target = vertex.active_dofs[i], vertex.active_dofs[j]
            operator[target] -= sigma.retarded[q, w, i, j]
            cs[target] += sigma.statistical[q, w, i, j]
            ds[target] += sigma.spectral[q, w, i, j]
        inverse = np.linalg.inv(operator)
        full_c, full_d = inverse@cs@inverse.conj().T, inverse@ds@inverse.conj().T
        for i, j in product(range(2), repeat=2):
            c[q, local, i, j] = full_c[vertex.active_dofs[i], vertex.active_dofs[j]]
            d[q, local, i, j] = full_d[vertex.active_dofs[i], vertex.active_dofs[j]]
        greens[q, w] = operator, inverse, full_c, full_d
    groups = {"zero" if k == 0 else f"near{k}": {
        term: np.zeros((2, 2), complex) for term in ("CC", "DD_over_4", "CD", "DC")}
        for k in range(width+1)}
    factor = .5j*grid.integration_weight/nq*g**2
    for p, local, a, b, i, j, k, ell in product(
            range(nq), range(len(indices)), *([range(2)]*6)):
        partner_q, partner_local = (-p) % nq, len(indices)-1-local
        v = vertex.coefficients[0, p]
        coefficient = factor*v[a, i, j]*v[b, k, ell].conjugate()
        key = "zero" if indices[local] == z else f"near{abs(indices[local]-z)}"
        c1, c2 = c[p, local, i, k], c[partner_q, partner_local, j, ell]
        d1, d2 = d[p, local, i, k], d[partner_q, partner_local, j, ell]
        for term, value in (("CC", c1*c2), ("DD_over_4", d1*d2/4),
                            ("CD", c1*d2), ("DC", d1*c2)):
            groups[key][term][a, b] += coefficient*value
    return groups, greens


@pytest.mark.parametrize("nq", [1, 3, 4])
@pytest.mark.parametrize("width", [0, 1, 2])
def test_partial_native_born_matches_literal_q_frequency_routing(nq, width):
    args = _fixture(nq)
    result = _run(args, g=.023, half_width=width)
    expected, greens = _literal(args, .023, width)
    assert result["dyson_evaluations"] == nq*(2*width+1)
    assert result["plan"]["contraction_work"] == 12*nq*(2*width+1)*2**4
    assert result["born"]["partner_q_indices"] == [(-p) % nq for p in range(nq)]
    sums = {term: np.zeros((2, 2), complex) for term in ("CC", "DD_over_4", "CD", "DC")}
    for partition, values in expected.items():
        actual = result["born"]["partitions"][partition]
        for term, matrix in values.items():
            np.testing.assert_allclose(_matrix(actual["terms"][term]), matrix, rtol=3e-13, atol=1e-18)
            assert actual["terms"][term]["frobenius"] == pytest.approx(np.linalg.norm(matrix))
            sums[term] += matrix
        np.testing.assert_allclose(_matrix(actual["statistical"]), values["CC"]+values["DD_over_4"],
                                   rtol=3e-13, atol=1e-18)
        np.testing.assert_allclose(_matrix(actual["spectral"]), values["CD"]+values["DC"],
                                   rtol=3e-13, atol=1e-18)
    for name, matrix in (("statistical", sums["CC"]+sums["DD_over_4"]),
                         ("spectral", sums["CD"]+sums["DC"])):
        born = result["born"]
        np.testing.assert_allclose(_matrix(born["partial_total"][name]), matrix,
                                   rtol=3e-13, atol=1e-18)
        saved = getattr(args[-1], name)[0, args[0].zero_index]
        np.testing.assert_allclose(_matrix(born["inferred_remainder"][name]), saved-matrix)
    assert result["born"]["inferred_remainder"]["direct_other_frequency_evaluation"] is False
    assert result["born"]["inferred_remainder"]["oracle"] is False
    for point in result["points"]:
        operator, inverse, _, _ = greens[point["q_index"], point["frequency_index"]]
        np.testing.assert_allclose(point["operator_singular_values"], np.linalg.svd(operator, compute_uv=False))
        assert point["context"]["norm_inf_condition_estimate"] == pytest.approx(
            np.linalg.norm(operator, np.inf)*np.linalg.norm(inverse, np.inf))
    json.dumps(result, allow_nan=False)


def test_independent_raw_spectral_zero_survives_huge_statistical_channel():
    args = _fixture(nq=1, huge=True)
    result = _run(args)
    z = args[0].zero_index
    native = args[-2]
    # Deriving D through two rounded exported contours would lose this signal.
    exported_delta = ((native.statistical+native.spectral/2)
                      -(native.statistical-native.spectral/2))
    assert np.max(abs(exported_delta.imag)) == 0
    assert np.max(abs(native.spectral.imag)) > 0
    zero = next(point for point in result["points"] if point["frequency_index"] == z)
    assert zero["spectral_source"]["max_abs"] > 0
    assert zero["spectral_green_active"]["max_abs"] > 0
    assert result["born"]["partitions"]["zero"]["terms"]["DD_over_4"]["max_abs"] > 0
    assert result["native_spectral_channel_retained"] is True
    assert zero["context"]["raw_source_identity"]["diagnostic_only"] is True
    assert zero["context"]["raw_source_identity"]["acceptance"] is False
    assert zero["context"]["raw_source_identity"]["pointwise_allowance_ratio"] > 1


def test_inputs_unchanged_and_no_pv_or_full_fft_path(monkeypatch):
    args = _fixture()
    arrays = [args[1].coefficients, args[2]]
    arrays += [getattr(contact.spectrum, name) for contact in args[3:5]
               for name in ("retarded", "lesser", "greater")]
    arrays += [getattr(spectrum, name) for spectrum in args[-2:]
               for name in ("retarded", "statistical", "spectral")]
    before = [value.copy() for value in arrays]
    import negfpy.anharmonic.transverse_zero_reference as old_reference
    monkeypatch.setattr(old_reference, "_pv_zero", lambda *a, **k: pytest.fail("PV forbidden"))
    monkeypatch.setattr(np.fft, "fft", lambda *a, **k: pytest.fail("full FFT forbidden"))
    result = _run(args)
    for actual, original in zip(arrays, before, strict=True):
        np.testing.assert_array_equal(actual, original)
    assert result["accepted_restart"] is False
    assert result["scba_evaluation"] is False
    assert result["full_born_map"] is False
    references = [point["extended_reference"] for point in result["points"]
                  if "extended_reference" in point]
    assert len(references) == 1
    if references[0]["available"]:
        assert references[0]["refinement"]["iterations"] == 4
        assert references[0]["certified_forward_accuracy"] is False
        assert references[0]["used_in_partial_born"] is False
        assert references[0]["operator_and_sources"] == "promoted fixed encoded complex128 assembly"
        assert "extended_assembly_comparison" not in references[0]
    assert result["extended_reference_scope"] == {
        "q_index": 0, "frequency_index": args[0].zero_index,
        "other_points_use_complex128_contexts_only": True}


@pytest.mark.parametrize("kwargs", [
    {"max_bytes": 1}, {"max_contraction_work": 1}, {"max_dyson_evaluations": 1},
    {"max_refinements": 5}, {"half_width": 3}, {"external_q": 1},
    {"g": float("nan")}, {"g": -1}, {"time_reserve_seconds": -1},
])
def test_bound_failures_precede_inverse(monkeypatch, kwargs):
    args = _fixture()
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("inverse started before bound"))
    with pytest.raises((ValueError, TypeError)):
        _run(args, **kwargs)


def test_deadline_exhaustion_precedes_inverse(monkeypatch):
    args = _fixture()
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("inverse after deadline"))
    with pytest.raises(TimeoutError):
        _run(args, deadline_monotonic=0)


@pytest.mark.parametrize("which", ["input", "saved", "dynamical"])
def test_nonfinite_selected_input_stops_before_inverse(monkeypatch, which):
    args = list(_fixture())
    if which == "input":
        replacement = args[-2].spectral.copy()
        replacement[0, args[0].zero_index] = np.nan
        args[-2] = ChannelSpectrum(args[-2].retarded, args[-2].statistical, replacement)
    elif which == "saved":
        replacement = args[-1].statistical.copy()
        replacement[0, args[0].zero_index] = np.inf
        args[-1] = ChannelSpectrum(args[-1].retarded, replacement, args[-1].spectral)
    else:
        args[2][0, 0, 0] = np.nan
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("inverse before finite validation"))
    with pytest.raises(ValueError, match="nonfinite"):
        _run(args)


def test_material_dimension_plan_is_bounded_without_allocations(monkeypatch):
    monkeypatch.setattr(np, "empty", lambda *a, **k: pytest.fail("planner allocated"))
    plan = diagnostic.plan_native_zero_diagnostic_dimensions(4, 601, 216, 120)
    assert plan.dyson_evaluations == 20
    assert plan.contraction_work == 49_766_400_000
    assert plan.peak_owned_bytes < 2*1024**3
    assert plan.retained_green_bytes < 100*1024**2
    assert plan.caller_payload_excluded is True


def test_zero_coupling_partial_born_is_exact_zero():
    result = _run(_fixture(), g=0)
    assert result["born"]["partial_total"]["statistical"]["max_abs"] == 0
    assert result["born"]["partial_total"]["spectral"]["max_abs"] == 0


def test_only_declared_five_bins_and_saved_zero_are_read():
    args = list(_fixture(npositive=3))
    z = args[0].zero_index
    # The raw mapped retarded field is never used computationally. Neither are
    # unrequested input bins or mapped nonzero frequencies.
    sigma_fields = [getattr(args[-2], name).copy() for name in
                    ("retarded", "statistical", "spectral")]
    for value in sigma_fields:
        value[:, 0] = np.nan
        value[:, -1] = np.nan
    args[-2] = ChannelSpectrum(*sigma_fields)
    mapped_fields = [getattr(args[-1], name).copy() for name in
                     ("retarded", "statistical", "spectral")]
    mapped_fields[0][:] = np.nan
    for value in mapped_fields[1:]:
        value[:, :z] = np.nan
        value[:, z+1:] = np.nan
    args[-1] = ChannelSpectrum(*mapped_fields)
    result = _run(args)
    assert result["dyson_evaluations"] == 15
    assert result["born"]["internal_frequency_indices"] == list(range(z-2, z+3))
    json.dumps(result, allow_nan=False)


def test_original_singular_operator_is_fatal_without_repair():
    args = list(_fixture(nq=1))
    args[2][:] = 0
    z = args[0].zero_index
    # Zero retarded contacts/interaction at omega=0 leave an exactly singular A.
    assert np.max(abs(args[-2].retarded[:, z])) == 0
    with pytest.raises(np.linalg.LinAlgError):
        _run(args)


@pytest.mark.parametrize("kwargs", [{"g": 1j}, {"g": True}, {"g": [1]},
                                    {"max_refinements": True}, {"half_width": 1.5}])
def test_non_scalar_or_non_integer_configuration_is_rejected(kwargs):
    with pytest.raises((TypeError, ValueError)):
        _run(_fixture(), **kwargs)
