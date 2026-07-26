from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.core import (
    transmission_kavg_adaptive,
    transmission_kavg_adaptive_global_eta,
)
from negfpy.modeling import BuildConfig
from negfpy.modeling.builders import build_material_kspace_params
from negfpy.modeling.schema import IFCData, IFCTerm
from negfpy.models import material_kspace_device
import negfpy.workflows.ifc_bulk as ifc_bulk


def _safe_config(*, requested: bool = True) -> dict[str, object]:
    return {
        "run": {
            "calculation": "transmission",
            "release_preflight": requested,
            "progress_omega": False,
        },
        "kmesh": {
            "dimension": 1,
            "mode": "centered",
            "nk": 3,
            "nk_low": 3,
        },
        "solver": {
            "eta_scheme": "adaptive_global",
            "eta_values": [1.0e-8],
            "eta_device": None,
            "min_success_fraction": 1.0,
        },
    }


def _complete_results() -> dict[str, object]:
    return {
        "summary": {
            "complete_frequency_grid": True,
            "complete_kmesh": True,
            "complete_q_grid": True,
            "release_safe": True,
        }
    }


def _passing_ifc_audit(*, geometry_promised: bool = False) -> dict[str, object]:
    return {
        "source": {
            "source_format": "analytic_fixture",
            "geometry_promised": geometry_promised,
        },
        "final": {
            "passed": True,
            "mapping_valid": True,
            "rotational_geometry_complete": geometry_promised,
            "rotational_applicable": geometry_promised,
            "rotational_valid": True,
            "metrics": {
                "geometry_metadata_complete": (
                    1.0 if geometry_promised else 0.0
                )
            },
        },
    }


def _passing_operator_audit() -> dict[str, object]:
    return {
        "schema": "negfpy.material_kspace_operator_audit",
        "schema_version": 1,
        "passed": True,
    }


def _audited_scalar_chain(*, mass: float = 1.0) -> IFCData:
    return IFCData(
        masses=np.array([mass]),
        dof_per_atom=1,
        terms=(
            IFCTerm(-1, 0, 0, np.array([[-1.0]])),
            IFCTerm(0, 0, 0, np.array([[2.0]])),
            IFCTerm(1, 0, 0, np.array([[-1.0]])),
        ),
        metadata={
            "source_format": "analytic_fixture",
            "nr": (3, 1, 1),
        },
    )


def test_per_q_adaptive_metadata_is_explicitly_noncertifying(monkeypatch) -> None:
    monkeypatch.setattr(
        "negfpy.core.negf.transmission",
        lambda **_kwargs: 1.0,
    )
    value, info = transmission_kavg_adaptive(
        omega=1.0,
        device=object(),
        lead_left=object(),
        lead_right=object(),
        kpoints=[(0.0,), (0.5,)],
        eta_values=(1.0e-8,),
        min_success_fraction=1.0,
    )

    assert value == pytest.approx(1.0)
    assert info["complete_kmesh"] is True
    assert info["complete_q_grid"] is True
    assert info["release_safe"] is False
    assert "independently" in str(info["release_unsafe_reason"])


def test_global_eta_requires_complete_q_grid_by_default(monkeypatch) -> None:
    def fake_transmission(*, kpar, **_kwargs):
        if kpar == (1.0,):
            raise RuntimeError("synthetic failed q-point")
        return 1.0

    monkeypatch.setattr("negfpy.core.negf.transmission", fake_transmission)
    common = {
        "omega": 1.0,
        "device": object(),
        "lead_left": object(),
        "lead_right": object(),
        "kpoints": [(0.0,), (1.0,)],
        "eta_values": (1.0e-8,),
    }

    with pytest.raises(RuntimeError, match="No eta value"):
        transmission_kavg_adaptive_global_eta(**common)

    value, info = transmission_kavg_adaptive_global_eta(
        **common,
        min_success_fraction=0.5,
    )
    assert value == pytest.approx(1.0)
    assert info["complete_kmesh"] is False
    assert info["complete_q_grid"] is False
    assert info["release_safe"] is False


def test_generic_templates_enable_release_safe_controls() -> None:
    generated = ifc_bulk._default_template()
    committed_path = (
        Path(__file__).parents[1]
        / "examples"
        / "configs"
        / "ifc_bulk_template.json"
    )
    committed = json.loads(committed_path.read_text(encoding="utf-8"))

    for template in (generated, committed):
        assert template["run"]["release_preflight"] is True
        assert template["solver"]["eta_scheme"] == "adaptive_global"
        assert template["solver"]["min_success_fraction"] == 1.0
        assert template["solver"]["eta_device"] is None
        assert template["model"]["infer_fc01_from_negative_dx"] is False
        assert template["model"]["enforce_hermitian_fc00"] is False
        assert template["model"]["dtype"] == "complex128"


def test_release_preflight_certifies_only_complete_enforced_result() -> None:
    cfg = _safe_config()
    mesh = ifc_bulk._prepare_kmesh(cfg)

    controls = ifc_bulk._harmonic_release_preflight(
        cfg,
        calculation="transmission",
        kmesh=mesh,
        ifc_audit=_passing_ifc_audit(),
        operator_audit=_passing_operator_audit(),
    )
    result = ifc_bulk._harmonic_release_preflight(
        cfg,
        calculation="transmission",
        kmesh=mesh,
        results=_complete_results(),
        ifc_audit=_passing_ifc_audit(),
        operator_audit=_passing_operator_audit(),
    )

    assert controls["passed"] is True
    assert controls["release_certified"] is False
    assert result["passed"] is True
    assert result["release_certified"] is True
    assert result["failure_reasons"] == []
    assert all(result["checks"].values())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("eta_scheme", "adaptive", "per-q adaptive"),
        ("min_success_fraction", 0.999, "must equal 1.0"),
        ("eta_device", 1.0e-8, "unaccounted bath"),
    ],
)
def test_release_preflight_rejects_unsafe_solver_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    cfg = copy.deepcopy(_safe_config())
    cfg["solver"][field] = value
    mesh = ifc_bulk._prepare_kmesh(cfg)

    with pytest.raises(ValueError, match=message):
        ifc_bulk._harmonic_release_preflight(
            cfg,
            calculation="transmission",
            kmesh=mesh,
        )


def test_release_preflight_rejects_legacy_endpoint_duplicate_mesh() -> None:
    cfg = copy.deepcopy(_safe_config())
    cfg["kmesh"].update({"mode": "legacy_endpoint", "nk": 5, "nk_low": 5})
    mesh = ifc_bulk._prepare_kmesh(cfg)

    assert ifc_bulk._periodic_kmesh_has_duplicates(mesh["low"]["kpoints"])
    with pytest.raises(ValueError, match="legacy_endpoint"):
        ifc_bulk._harmonic_release_preflight(
            cfg,
            calculation="transmission",
            kmesh=mesh,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("complete_frequency_grid", "incomplete frequency"),
        ("complete_q_grid", "omits one or more"),
        ("complete_kmesh", "omits one or more"),
        ("release_safe", "does not mark"),
    ],
)
def test_release_preflight_rejects_incomplete_results(
    field: str,
    message: str,
) -> None:
    cfg = _safe_config()
    mesh = ifc_bulk._prepare_kmesh(cfg)
    results = _complete_results()
    results["summary"][field] = False

    with pytest.raises(ValueError, match=message):
        ifc_bulk._harmonic_release_preflight(
            cfg,
            calculation="transmission",
            kmesh=mesh,
            results=results,
        )


def test_absent_release_flag_is_recorded_and_never_certifies() -> None:
    cfg = _safe_config(requested=False)
    mesh = ifc_bulk._prepare_kmesh(cfg)
    report = ifc_bulk._harmonic_release_preflight(
        cfg,
        calculation="transmission",
        kmesh=mesh,
        results=_complete_results(),
    )

    assert report["requested"] is False
    assert report["passed"] is False
    assert report["release_certified"] is False
    assert any("not enabled" in reason for reason in report["failure_reasons"])


def test_transmission_workflow_defaults_to_complete_global_eta(monkeypatch) -> None:
    def fake_global_eta(**kwargs):
        assert kwargs["min_success_fraction"] == 1.0
        assert kwargs["eta_device"] is None
        n_total = len(kwargs["kpoints"])
        eta = float(kwargs["eta_values"][0])
        return 0.75, {
            "n_total": n_total,
            "n_success": n_total,
            "n_failed": 0,
            "success_fraction": 1.0,
            "eta_selected": eta,
            "eta_histogram": {eta: n_total},
            "complete_kmesh": True,
            "complete_q_grid": True,
            "release_safe": True,
        }

    monkeypatch.setattr(
        ifc_bulk,
        "transmission_kavg_adaptive_global_eta",
        fake_global_eta,
    )
    monkeypatch.setattr(
        ifc_bulk,
        "transmission_kavg_adaptive",
        lambda **_kwargs: pytest.fail("per-q adaptive path must not be selected"),
    )
    cfg = {"run": {"progress_omega": False}, "solver": {"eta_values": [1.0e-8]}}
    kpoints = [(0.0,), (0.5,)]
    kmesh = {
        "low_cm1": 1.0e30,
        "low": {"kpoints": kpoints},
        "high": {"kpoints": kpoints},
    }

    result = ifc_bulk._run_transmission(
        cfg,
        device=object(),
        lead_left=object(),
        lead_right=object(),
        omegas=np.array([0.0, 1.0]),
        kmesh=kmesh,
    )

    summary = result["summary"]
    assert summary["eta_scheme"] == "adaptive_global"
    assert summary["min_success_fraction"] == 1.0
    assert summary["complete_frequency_grid"] is True
    assert summary["complete_q_grid"] is True
    assert summary["release_safe"] is True


def test_fcs_summary_requires_complete_frequency_and_q_grids(monkeypatch) -> None:
    monkeypatch.setattr(
        ifc_bulk,
        "_compute_fcs_k_moments",
        lambda *_args, **_kwargs: {
            "t_mean": np.array([0.0, 0.5]),
            "t2_mean": np.array([0.0, 0.25]),
            "eta_selected": np.array([np.nan, 1.0e-8]),
            "n_k_success": np.array([0, 2]),
            "n_k_total": np.array([0, 2]),
            "omega_failures": [],
        },
    )
    monkeypatch.setattr(
        ifc_bulk,
        "heat_current_cumulants_from_k_moments",
        lambda **_kwargs: SimpleNamespace(c1_j_per_s=1.0, c2_j2_per_s=2.0),
    )
    monkeypatch.setattr(
        ifc_bulk,
        "heat_current_uncertainty",
        lambda *_args, **_kwargs: SimpleNamespace(std_current_j_per_s=0.1),
    )
    cfg = {
        "fcs": {
            "y": "conductance",
            "temperature_min_k": 100.0,
            "temperature_max_k": 110.0,
            "n_points": 2,
            "delta_t_k": 10.0,
        }
    }

    result = ifc_bulk._run_fcs(
        cfg,
        omegas=np.array([0.0, 1.0]),
        kmesh={},
        device=object(),
        lead_left=object(),
        lead_right=object(),
        device_to_lead_left=None,
        device_to_lead_right=None,
        area_m2=None,
    )

    assert result["summary"]["complete_frequency_grid"] is True
    assert result["summary"]["complete_q_grid"] is True
    assert result["summary"]["release_safe"] is True


def test_dos_workflow_defaults_to_complete_global_eta(monkeypatch) -> None:
    def fake_global_eta(*, kpoints, min_success_fraction, **_kwargs):
        assert min_success_fraction == 1.0
        n_total = len(kpoints)
        eta = 1.0e-8
        return 0.5, {
            "n_total": n_total,
            "n_success": n_total,
            "n_failed": 0,
            "success_fraction": 1.0,
            "eta_selected": eta,
            "eta_histogram": {eta: n_total},
            "complete_kmesh": True,
            "complete_q_grid": True,
            "release_safe": True,
        }

    monkeypatch.setattr(
        ifc_bulk,
        "_lead_surface_dos_kavg_adaptive_global_eta",
        fake_global_eta,
    )
    monkeypatch.setattr(
        ifc_bulk,
        "lead_surface_dos_kavg_adaptive",
        lambda **_kwargs: pytest.fail("per-q adaptive path must not be selected"),
    )
    cfg = {"run": {"progress_omega": False}}
    kpoints = [(0.0,), (0.5,)]
    kmesh = {
        "low_cm1": 1.0e30,
        "low": {"kpoints": kpoints},
        "high": {"kpoints": kpoints},
    }

    result = ifc_bulk._run_dos(
        cfg,
        lead=object(),
        omegas=np.array([0.0, 1.0]),
        kmesh=kmesh,
    )

    summary = result["summary"]
    assert summary["eta_scheme"] == "adaptive_global"
    assert summary["min_success_fraction"] == 1.0
    assert summary["complete_frequency_grid"] is True
    assert summary["complete_q_grid"] is True
    assert summary["release_safe"] is True


def test_thermal_conductance_uses_warning_clean_trapezoid() -> None:
    conductance = ifc_bulk._thermal_conductance_from_spectrum(
        np.array([0.0, 0.5, 1.0]),
        np.ones(3),
        np.array([0.0, 300.0]),
    )

    assert conductance[0] == 0.0
    assert np.isfinite(conductance[1])
    assert conductance[1] > 0.0


@pytest.mark.parametrize(
    ("solver", "message"),
    [
        ({"eta_scheme": "fixed", "eta_fixed": 0.0}, "eta_fixed"),
        (
            {"eta_scheme": "adaptive_global", "eta_values": []},
            "non-empty",
        ),
        (
            {
                "eta_scheme": "adaptive_global",
                "eta_values": [1.0e-8, 1.0e-8],
            },
            "nonduplicate",
        ),
        (
            {"eta_scheme": "fixed", "eta_device": -1.0e-8},
            "non-negative",
        ),
    ],
)
def test_eta_controls_reject_noncausal_values_before_solve(
    solver: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        ifc_bulk._validated_eta_controls(solver)


def test_promised_geometry_requires_mapping_and_rotational_evidence() -> None:
    cfg = _safe_config()
    mesh = ifc_bulk._prepare_kmesh(cfg)
    incomplete = _passing_ifc_audit(geometry_promised=True)
    incomplete["final"]["mapping_valid"] = False
    incomplete["final"]["rotational_geometry_complete"] = False
    incomplete["final"]["rotational_applicable"] = False
    incomplete["final"]["metrics"]["geometry_metadata_complete"] = 0.0

    with pytest.raises(ValueError, match="mapping geometry is incomplete"):
        ifc_bulk._harmonic_release_preflight(
            cfg,
            calculation="transmission",
            kmesh=mesh,
            ifc_audit=incomplete,
            operator_audit=_passing_operator_audit(),
        )


def test_final_material_operator_audit_detects_mass_and_support_mutations() -> None:
    ifc = _audited_scalar_chain(mass=2.0)
    build = BuildConfig(
        infer_fc01_from_negative_dx=False,
        enforce_hermitian_fc00=False,
        dtype="complex128",
    )
    params = build_material_kspace_params(ifc, config=build)
    device = material_kspace_device(n_layers=3, params=params)

    first = ifc_bulk._audit_material_kspace_operator(ifc, params, device)
    second = ifc_bulk._audit_material_kspace_operator(ifc, params, device)
    assert first == second
    assert first["passed"] is True
    assert first["expected_hash"] == first["actual_hash"]

    wrong_masses = replace(params, masses=np.ones_like(params.masses))
    mass_report = ifc_bulk._audit_material_kspace_operator(
        ifc,
        wrong_masses,
        material_kspace_device(n_layers=3, params=wrong_masses),
    )
    assert mass_report["passed"] is False
    assert mass_report["checks"]["masses_exact"] is False

    missing_reverse = replace(params, fc10_terms=None)
    support_report = ifc_bulk._audit_material_kspace_operator(
        ifc,
        missing_reverse,
        material_kspace_device(n_layers=3, params=missing_reverse),
    )
    assert support_report["passed"] is False
    assert support_report["checks"]["support_exact"] is False
    assert support_report["support_mismatches"]["fc10"]["missing"]


def _write_release_run_config(
    tmp_path: Path,
    *,
    mutation: tuple[str, str, object] | None = None,
) -> Path:
    source_path = tmp_path / "analytic.ifc"
    source_path.write_text("analytic fixture\n", encoding="utf-8")
    cfg: dict[str, object] = {
        "run": {
            "name": "release_path",
            "calculation": "transmission",
            "output_mode": "flat",
            "output_dir": str(tmp_path / "output"),
            "release_preflight": True,
            "write_input_snapshot": False,
            "write_plot": False,
            "write_data": False,
            "write_report": False,
            "progress_omega": False,
        },
        "ifc": {
            "path": source_path.name,
            "reader": "qe",
            "enforce_asr": False,
            "dy_cutoff": None,
            "dz_cutoff": None,
        },
        "model": {
            "transport_direction": 1,
            "n_layers": 3,
            "principal_layer_size": 1,
            "nyquist_split_half": False,
            "dtype": "complex128",
            "mass_mode": "ifc",
            "drop_nyquist_transverse": False,
            "n1_transport_unwrap": None,
            "onsite_pinning": 0.0,
        },
        "kmesh": {"dimension": 0, "mode": "gamma", "nk": 1},
        "omega": {
            "min": 0.0,
            "max": 1.0,
            "n_points": 2,
            "auto_max": False,
        },
        "solver": {
            "eta_scheme": "fixed",
            "eta_fixed": 1.0e-4,
            "eta_values": [1.0e-8],
            "eta_device": None,
            "min_success_fraction": 1.0,
        },
    }
    if mutation is not None:
        section, field, value = mutation
        cfg[section][field] = value
    config_path = tmp_path / "run.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    return config_path


def _mock_complete_transmission(_cfg, *, omegas, **_kwargs) -> dict[str, object]:
    return {
        "values": np.ones_like(omegas, dtype=float),
        "eta_records": [],
        "omega_failures": [],
        "n_rejected_total": 0,
        "summary": {
            "complete_frequency_grid": True,
            "complete_kmesh": True,
            "complete_q_grid": True,
            "release_safe": True,
        },
    }


def test_run_ifc_bulk_release_path_certifies_exact_operator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(ifc_bulk, "read_ifc", lambda *_args, **_kwargs: _audited_scalar_chain())
    monkeypatch.setattr(ifc_bulk, "_run_transmission", _mock_complete_transmission)

    report = ifc_bulk.run_ifc_bulk(_write_release_run_config(tmp_path))

    preflight = report["provenance"]["release_preflight"]
    assert preflight["release_certified"] is True
    assert preflight["checks"]["final_operator_matches_ifc"] is True
    assert report["ifc"]["operator_audit"]["passed"] is True
    assert report["model"]["enforce_transverse_pm_symmetry"] is False


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("model", "onsite_pinning", 1.0e-4), "onsite_pinning"),
        (("model", "infer_fc01_from_negative_dx", True), "infer_fc01"),
        (("model", "enforce_hermitian_fc00", True), "enforce_hermitian"),
        (("model", "mass_mode", "unit"), "mass_mode"),
        (("model", "drop_nyquist_transverse", True), "drop_nyquist"),
        (("model", "nyquist_split_half", True), "nyquist_split_half"),
        (("model", "dtype", "complex64"), "dtype"),
        (("ifc", "dy_cutoff", 0), "dy_cutoff"),
        (("solver", "eta_fixed", -1.0e-4), "eta_fixed"),
        (("solver", "eta_values", []), "eta_values"),
    ],
)
def test_run_ifc_bulk_release_path_rejects_adversarial_mutations(
    tmp_path: Path,
    monkeypatch,
    mutation: tuple[str, str, object],
    message: str,
) -> None:
    monkeypatch.setattr(ifc_bulk, "read_ifc", lambda *_args, **_kwargs: _audited_scalar_chain())
    monkeypatch.setattr(ifc_bulk, "_run_transmission", _mock_complete_transmission)

    with pytest.raises(ValueError, match=message):
        ifc_bulk.run_ifc_bulk(
            _write_release_run_config(tmp_path, mutation=mutation)
        )
