"""Small independent algebra/reference cases; no saved material arrays."""

import json
from decimal import Decimal, localcontext

import numpy as np
import pytest

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.transverse_zero_reference import (
    _pv_zero,
    _refine_inverse,
    study_zero_frequency,
    zero_reference_workspace_bytes,
)


def _fixture(dimension=1, n_positive=2):
    grid = SignedFrequencyGrid(.2, n_positive)
    shape = (grid.n_points, dimension, dimension)
    spectrum = {name: np.zeros(shape, complex) for name in ("retarded", "lesser", "greater")}
    point = {"support_dofs": list(range(dimension)),
             **{name: np.zeros((dimension, dimension), complex)
                for name in ("retarded", "lesser", "greater")}}
    return {"grid": grid, "dynamical_q": np.eye(dimension, dtype=complex)*2,
            "active_dofs": np.arange(dimension), "left_point": point, "right_point": point,
            "sigma_spectrum_q": spectrum,
            "max_bytes": zero_reference_workspace_bytes(dimension, dimension, grid.n_points)}


def _zero_values(arguments, retarded, lesser, greater):
    z = arguments["grid"].zero_index
    for name, field in zip(("retarded", "lesser", "greater"), (retarded, lesser, greater)):
        arguments["sigma_spectrum_q"][name][z] = field


def test_scalar_source_defect_and_propagation_match_independent_formula():
    args = _fixture()
    _zero_values(args, -.2j, -.7j, -1.100004j)
    result = study_zero_frequency(**args)
    original = result["original_candidate"]
    encoded_delta = np.longdouble(-1.100004)-np.longdouble(-.7)
    defect = encoded_delta+np.longdouble(.4)
    expected = abs(defect)/(np.longdouble(4)+np.longdouble(.2)**2)
    raw = original["raw_sources"]["defect_contributions"]["interaction"]
    assert raw["raw_defect"]["max_abs"] == pytest.approx(float(abs(defect)), rel=1e-13)
    assert raw["propagated_defect"]["max_abs"] == pytest.approx(float(expected), rel=1e-13)
    ext = original["extended_reference"]
    assert ext["stable_observed_minus_propagated"]["max_abs"] < 1e-18
    assert result["identity_forced"] is False
    assert result["symmetry_projected"] is False


def test_harmonic_nonhermiticity_sign_is_measured_without_projection():
    args = _fixture()
    args["dynamical_q"][0, 0] += 3e-5j
    result = study_zero_frequency(**args)
    harmonic = result["original_candidate"]["raw_sources"]["defect_contributions"]["harmonic"]
    assert harmonic["raw_defect"]["value_at_max_abs"]["imag"] == pytest.approx(-6e-5)
    assert harmonic["propagated_defect"]["value_at_max_abs"]["imag"] < 0


def test_pv_sign_and_finite_domain_match_decimal_scalar_oracle():
    args = _fixture(n_positive=3)
    grid = args["grid"]
    spectrum = args["sigma_spectrum_q"]
    gamma = .37
    spectrum["greater"][:, 0, 0] = -2j*gamma*grid.values
    actual = _pv_zero(tuple(spectrum[name] for name in ("retarded", "lesser", "greater")), grid)
    with localcontext() as context:
        context.prec = 70
        total = Decimal(0)
        for index, omega in enumerate(grid.values):
            if omega:
                encoded_delta_imag = spectrum["greater"][index, 0, 0].imag
                total += (Decimal.from_float(grid.integration_weight)
                          *Decimal.from_float(encoded_delta_imag)/-Decimal.from_float(omega))
        expected = -total
        assert abs(Decimal(str(actual[0, 0].real))-expected) < Decimal("2e-19")
    assert actual[0, 0].real < 0
    assert actual[0, 0].imag == 0


def test_nonzero_real_antisymmetric_zero_spectral_is_retained():
    args = _fixture(dimension=2)
    antisymmetric = np.array([[0, .125], [-.125, 0]], complex)
    _zero_values(args, antisymmetric, -1j*np.eye(2)-antisymmetric,
                 -1j*np.eye(2)+antisymmetric)
    result = study_zero_frequency(**args)
    pv = result["principal_value"]
    assert pv["raw_zero_spectral"]["max_abs_real"] == .25
    assert pv["raw_zero_spectral"]["max_abs_imag"] == 0
    assert pv["candidate_direct_pv"]["max_abs"] == .125
    assert pv["candidate_stored_minus_direct_pv"]["error"] == 0
    ext = result["original_candidate"]["extended_reference"]
    assert ext["stable_spectral_product"]["max_abs_real"] > 0
    assert ext["stable_keldysh"]["error"] < 1e-18
    assert ext["retarded"]["reality_defect"] == 0


def test_extended_inverse_refinement_matches_decimal_complex_division():
    real, imag = .123456789, .987654321
    operator = np.array([[complex(real, imag)]], dtype=np.clongdouble)
    inverse = np.linalg.inv(operator.astype(complex))
    reference, quality = _refine_inverse(operator, inverse, 4)
    with localcontext() as context:
        context.prec = 70
        a, b = Decimal.from_float(real), Decimal.from_float(imag)
        denominator = a*a+b*b
        assert abs(Decimal(str(reference[0, 0].real))-a/denominator) < Decimal("2e-19")
        assert abs(Decimal(str(reference[0, 0].imag))+b/denominator) < Decimal("2e-19")
    assert quality["iterations"] == 4
    assert len(quality["history"]) == 5
    assert quality["measured_backward_error_adequate"] is True
    assert quality["certified_forward_accuracy"] is False
    assert quality["history"][-1]["left_absolute_residual"] <= np.finfo(np.longdouble).eps


def test_bad_preconditioner_is_not_labelled_adequate_or_exact():
    operator = np.array([[2]], dtype=np.clongdouble)
    _, quality = _refine_inverse(operator, np.array([[0]], complex), 4)
    assert quality["measured_backward_error_adequate"] is False
    assert quality["certified_forward_accuracy"] is False
    assert quality["history"][-1]["left_absolute_residual"] == 1


def test_statistical_spectral_split_exposes_double_separate_product_cancellation():
    args = _fixture()
    _zero_values(args, -.25j, -1e12j, -1e12j-.5j)
    result = study_zero_frequency(**args)
    case = result["original_candidate"]
    baseline = case["ordinary_complex128"]
    reference = case["extended_reference"]
    assert baseline["cancellation_magnitude_ratio"] > 1e12
    assert baseline["separate_minus_stable"]["error"] > 1e-8
    assert reference["stable_keldysh"]["error"] < 1e-18
    assert case["ordinary_versus_extended"]["stable_difference"]["error"] < 1e-16
    assert case["ordinary_versus_extended"]["separate_difference"]["error"] > 1e-8


def test_baseline_reproduces_ordinary_operation_order_with_overlapping_sources():
    args = _fixture(dimension=2)
    args["dynamical_q"][:] = [[2, .23], [.23, 3]]
    args["left_point"] = {
        "support_dofs": [1, 0], "retarded": np.diag([-.3j, -.4j]),
        "lesser": np.diag([-.7j, -.9j]), "greater": np.diag([-1.3j, -1.7j])}
    _zero_values(args, np.diag([-.1j, -.2j]), np.diag([-.3j, -.4j]), np.diag([-.5j, -.8j]))
    a = 0*np.eye(2, dtype=complex)-args["dynamical_q"]
    sl, sg = np.zeros((2, 2), complex), np.zeros((2, 2), complex)
    z = args["grid"].zero_index
    for contact in (args["left_point"], args["right_point"]):
        support = np.ix_(contact["support_dofs"], contact["support_dofs"])
        a[support] -= contact["retarded"]
        sl[support] += contact["lesser"]
        sg[support] += contact["greater"]
    a -= args["sigma_spectrum_q"]["retarded"][z]
    sl += args["sigma_spectrum_q"]["lesser"][z]
    sg += args["sigma_spectrum_q"]["greater"][z]
    gr = np.linalg.inv(a)
    gl, gg = gr@sl@gr.conj().T, gr@sg@gr.conj().T
    expected = float(np.max(np.abs(gg-gl-(gr-gr.conj().T))))
    result = study_zero_frequency(**args)
    assert result["original_candidate"]["ordinary_complex128"]["separate_keldysh"]["error"] == expected


def test_saved_input_and_map_are_decomposed_without_changing_candidate():
    args = _fixture()
    grid = args["grid"]
    saved = {key: value.copy() for key, value in args["sigma_spectrum_q"].items()}
    mapped = {key: value.copy() for key, value in saved.items()}
    for index, name in enumerate(("retarded", "lesser", "greater")):
        saved[name][:] = (index+1)*.01j
        mapped[name][:] = (index+2)*.03j
        args["sigma_spectrum_q"][name][:] = .9*saved[name]+.1*mapped[name]
    before = {key: value.tobytes() for key, value in args["sigma_spectrum_q"].items()}
    result = study_zero_frequency(**args, sigma_input_spectrum_q=saved, mapped_sigma_spectrum_q=mapped)
    decomposition = result["principal_value"]["mixing_decomposition"]
    assert decomposition["mixing"] == .1
    assert decomposition["candidate_pv_minus_weighted_saved_pv"]["error"] < 1e-17
    assert all(value.tobytes() == before[key] for key, value in args["sigma_spectrum_q"].items())
    assert result["frequency_index"] == grid.zero_index
    assert result["direct_pv_retarded_trial"]["changed_candidate"] is True
    assert result["direct_pv_retarded_trial"]["accepted"] is False


def test_two_inversions_four_refinements_bound_and_json_scalar_output(monkeypatch):
    args = _fixture()
    inv = np.linalg.inv
    calls = []
    def counted(value):
        calls.append(value.dtype)
        return inv(value)
    monkeypatch.setattr(np.linalg, "inv", counted)
    result = study_zero_frequency(**args)
    assert calls == [np.dtype(complex), np.dtype(complex)]
    assert result["resources"]["zero_dyson_cases"] == 2
    assert result["resources"]["inverse_refinements_per_case"] == 4
    assert result["resources"]["frequency_stack_allocated"] is False
    json.dumps(result, allow_nan=False)
    def scalar(value):
        if isinstance(value, dict):
            return all(scalar(child) for child in value.values())
        if isinstance(value, list):
            return all(scalar(child) for child in value)
        return value is None or type(value) in (str, bool, int, float)
    assert scalar(result)
    assert result["nonfinite_fields"] == []
    assert result["born_evaluated"] is False
    assert result["scba_state_completed"] is False


@pytest.mark.parametrize("change", [
    {"max_bytes": 1}, {"max_refinements": 5}, {"max_refinements": True},
    {"mixing": 0}, {"mixing": np.nan}, {"mixing": True},
    {"active_dofs": np.array([1])}, {"active_dofs": np.array([0.0])},
])
def test_invalid_bounds_and_inputs_fail_before_inverse(monkeypatch, change):
    args = _fixture()
    args.update(change)
    monkeypatch.setattr(np.linalg, "inv", lambda *a, **k: pytest.fail("inverse ran before guard"))
    with pytest.raises((TypeError, ValueError)):
        study_zero_frequency(**args)


def test_longdouble_platform_without_extra_precision_fails_before_inverse(monkeypatch):
    args = _fixture()
    finfo = np.finfo
    monkeypatch.setattr(np, "finfo", lambda dtype: finfo(np.float64) if dtype is np.longdouble else finfo(dtype))
    monkeypatch.setattr(np.linalg, "inv", lambda *a, **k: pytest.fail("inverse ran before precision guard"))
    with pytest.raises(RuntimeError, match="additional"):
        study_zero_frequency(**args)


@pytest.mark.parametrize("which", ["candidate", "contact", "dynamical", "pv"])
def test_nonfinite_raw_fields_rejected_without_projection(monkeypatch, which):
    args = _fixture()
    if which == "candidate":
        args["sigma_spectrum_q"]["retarded"][2, 0, 0] = np.nan
    elif which == "contact":
        args["left_point"]["lesser"][0, 0] = np.inf
    elif which == "dynamical":
        args["dynamical_q"][0, 0] = np.nan
    else:
        args["sigma_spectrum_q"]["greater"][0, 0, 0] = np.inf
    monkeypatch.setattr(np.linalg, "inv", lambda *a, **k: pytest.fail("inverse ran before finite guard"))
    with pytest.raises(ValueError, match="finite"):
        study_zero_frequency(**args)


def test_optional_saved_spectra_must_be_a_pair(monkeypatch):
    args = _fixture()
    monkeypatch.setattr(np.linalg, "inv", lambda *a, **k: pytest.fail("inverse ran before paired guard"))
    with pytest.raises(ValueError, match="both"):
        study_zero_frequency(**args, sigma_input_spectrum_q=args["sigma_spectrum_q"])


@pytest.mark.parametrize("dimensions", [(0, 1, 3), (513, 1, 3), (2, 3, 3), (1, 1, 4099)])
def test_workspace_count_caps_are_explicit(dimensions):
    with pytest.raises(ValueError):
        zero_reference_workspace_bytes(*dimensions)
