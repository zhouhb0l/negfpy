"""Independent tiny eta-reference semantics and rigid-translation contractions."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest
from test_anharmonic_transverse_native_zero_diagnostic import ELEMENTS, _fixture

from negfpy.anharmonic import transverse_eta_zero_diagnostic as study
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_compact_contacts import CompactTransverseContact
from negfpy.anharmonic.transverse_packed import PackedTransverseCubicVertex


def _eta_fixture(eta):
    args = list(_fixture(nq=1, npositive=1))
    for index in (3, 4):
        old = args[index]
        args[index] = CompactTransverseContact(replace(old.contact, eta_lead=eta),
                                               old.full_ndof, old.device_indices)
    return args


@pytest.mark.parametrize("eta", [.001, .005, .01])
def test_changed_eta_keeps_baseline_comparison_distinct_from_far_bins(eta):
    args = _eta_fixture(eta)
    originals = [getattr(spectrum, name).copy() for spectrum in args[-2:]
                 for name in ("retarded", "statistical", "spectral")]
    result = study.study_eta_zero_diagnostic(
        *args, eta_lead=eta, half_width=1, workers=1, external_tile=1,
        selected_elements=ELEMENTS, max_refinements=0)
    assert result["schema"] == "negfpy.eta-zero-diagnostic.v1"
    assert result["eta_lead"] == eta
    assert result["baseline_eta_lead"] == .0015625
    assert result["eta_device"] == 0
    assert result["physical_inputs_unchanged"] is False
    assert result["fixed_interaction_inputs_unchanged"] is True
    assert result["physical_guards_unchanged"] is True
    assert result["fresh_contact_eta_tags_verified"] is True
    assert result["accepted_restart"] is False
    assert result["scba_evaluation"] is False
    assert result["full_born_map"] is False
    born = result["born"]
    assert "saved_mapped" not in born and "inferred_remainder" not in born
    assert born["full_current_eta_total_evaluated"] is False
    assert born["baseline_saved_mapped_used_for_green_or_born"] is False
    gap = born["cross_eta_reference_gap"]
    assert gap["far_frequency_remainder"] is False
    assert gap["direct_other_frequency_evaluation"] is False
    assert gap["same_eta_as_baseline"] is False
    assert gap["current_eta_lead"] == eta
    assert gap["baseline_eta_lead"] == .0015625
    assert "current-eta partial" in gap["status"]
    assert born["baseline_saved_mapped"]["statistical"]["max_abs"] == pytest.approx(
        np.max(abs(args[-1].statistical[0, args[0].zero_index])))
    actuals = [getattr(spectrum, name) for spectrum in args[-2:]
               for name in ("retarded", "statistical", "spectral")]
    for actual, original in zip(actuals, originals, strict=True):
        np.testing.assert_array_equal(actual, original)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("eta", [0, -.01, float("nan"), float("inf"), True, .01j])
def test_invalid_eta_stops_before_inverse(monkeypatch, eta):
    args = _eta_fixture(.005)
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("inverse before eta validation"))
    with pytest.raises(ValueError):
        study.study_eta_zero_diagnostic(*args, eta_lead=eta)


def test_old_contact_eta_cannot_be_relabelled(monkeypatch):
    monkeypatch.setattr(np.linalg, "inv", lambda *a: pytest.fail("stale eta reached inverse"))
    with pytest.raises(ValueError, match="eta tag"):
        study.study_eta_zero_diagnostic(*_fixture(), eta_lead=.005)


def _relative_bond_vertex(crop=False, nq=3):
    """Cubic relative-displacement potential with nontrivial q phases.

    At q=0 the factors are the difference of the two atoms' x displacements.
    At other q their common component is allowed: only a zero-momentum leg is
    constrained by the global translation.  Thus wrong q routing is detectable.
    """
    mesh = UniformBZMesh((nq, 1, 1))
    difference = np.array([1., 0, 0, -1., 0, 0])
    common = np.array([.2, .3, .4, .7, .8, .9])
    factors = [difference+(np.exp(2j*np.pi*q/nq)-1)*common for q in range(nq)]
    coefficients = np.empty((nq, nq, 6, 6, 6), complex)
    for q in range(nq):
        for p in range(nq):
            coefficients[q, p] = np.einsum(
                "a,c,d->acd", factors[(-q) % nq], factors[p], factors[(q-p) % nq])
    translations = np.tile(np.eye(3), (2, 1))/np.sqrt(2)
    indices = np.arange(3 if crop else 6)
    if crop:
        coefficients = coefficients[:, :, :3, :3, :3]
    vertex = PackedTransverseCubicVertex(mesh, coefficients, indices, 6)
    return vertex, translations


def test_complete_relative_displacement_vertex_annihilates_rigid_leg_at_correct_q():
    vertex, translations = _relative_bond_vertex()
    before = vertex.coefficients.copy()
    result = study.audit_cropped_vertex_acoustic(vertex, translations)
    assert result["contraction_count"] == 27
    assert result["planned_multiply_accumulate_work"] == 9*3*6**3
    assert result["diagnostic_only"] is True
    assert result["acceptance"] is False
    assert result["modifies_vertex"] is False
    assert result["all_gamma_modes_are_translations"] is False
    assert max(row["max_abs"] for row in result["records"]) < 2e-14
    for row in result["records"]:
        axis, q, p = row["zero_momentum_tensor_axis"], row["q_output"], row["q_internal"]
        assert q == 0 if axis == 0 else p == 0 if axis == 1 else q == p
    # Nonzero internal momentum does not annihilate a common translation.
    wrong_routing = np.einsum("c,acd->ad", translations[:, 0], vertex.coefficients[0, 1])
    assert np.max(abs(wrong_routing)) > .1
    np.testing.assert_array_equal(vertex.coefficients, before)
    json.dumps(result, allow_nan=False)


def test_all_leg_crop_can_break_asr_and_active_translation_is_not_renormalized():
    vertex, translations = _relative_bond_vertex(crop=True, nq=1)
    result = study.audit_cropped_vertex_acoustic(vertex, translations)
    np.testing.assert_allclose(result["full_translation_gram"], np.eye(3))
    np.testing.assert_allclose(result["active_translation_gram"], np.eye(3)/2)
    assert result["contraction_count"] == 9
    for row in result["records"]:
        assert row["active_translation_norm"] == pytest.approx(1/np.sqrt(2))
        if row["translation_direction"] == 0:
            assert row["max_abs"] == pytest.approx(1/np.sqrt(2))
            assert row["frobenius"] == pytest.approx(1/np.sqrt(2))
            assert row["relative_frobenius_to_scale"] == pytest.approx(1)
        else:
            assert row["frobenius"] == 0


def test_unequal_mass_translations_annihilate_massweighted_relative_vertex():
    masses = np.array([2., 7.])
    translations = np.repeat(np.sqrt(masses), 3)[:, None]*np.tile(np.eye(3), (2, 1))
    translations /= np.sqrt(masses.sum())
    relative = np.array([1/np.sqrt(2), 0, 0, -1/np.sqrt(7), 0, 0])
    block = np.einsum("a,c,d->acd", relative, relative, relative)
    vertex = PackedTransverseCubicVertex(UniformBZMesh((1, 1, 1)),
                                         block[None, None], np.arange(6), 6)
    result = study.audit_cropped_vertex_acoustic(vertex, translations)
    assert max(row["frobenius"] for row in result["records"]) < 1e-16


@pytest.mark.parametrize("kwargs", [
    {"max_bytes": 1}, {"max_contraction_work": 1},
    {"deadline_monotonic": 0}, {"time_reserve_seconds": -1},
])
def test_acoustic_audit_admission_before_contraction(monkeypatch, kwargs):
    vertex, translations = _relative_bond_vertex()
    monkeypatch.setattr(np, "einsum", lambda *a, **k: pytest.fail("contraction before admission"))
    with pytest.raises((ValueError, TimeoutError)):
        study.audit_cropped_vertex_acoustic(vertex, translations, **kwargs)


@pytest.mark.parametrize("kind", ["complex", "nonfinite", "unnormalized", "wrongshape"])
def test_translation_contract_is_validated(kind):
    vertex, translations = _relative_bond_vertex()
    if kind == "complex":
        translations = translations.astype(complex)
    elif kind == "nonfinite":
        translations[0, 0] = np.nan
    elif kind == "unnormalized":
        translations *= 2
    else:
        translations = translations[:, :2]
    with pytest.raises(ValueError):
        study.audit_cropped_vertex_acoustic(vertex, translations)


def test_zero_vertex_reports_zero_scale_without_nan():
    vertex, translations = _relative_bond_vertex()
    zero = PackedTransverseCubicVertex(vertex.mesh, np.zeros_like(vertex.coefficients),
                                       vertex.active_dofs, vertex.full_ndof)
    result = study.audit_cropped_vertex_acoustic(zero, translations)
    assert all(row["relative_frobenius_to_scale"] == 0 for row in result["records"])
    json.dumps(result, allow_nan=False)
