from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.core import (
    transmission_kavg_adaptive,
    transmission_kavg_adaptive_global_eta,
)
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


def test_release_preflight_certifies_only_complete_enforced_result() -> None:
    cfg = _safe_config()
    mesh = ifc_bulk._prepare_kmesh(cfg)

    controls = ifc_bulk._harmonic_release_preflight(
        cfg,
        calculation="transmission",
        kmesh=mesh,
    )
    result = ifc_bulk._harmonic_release_preflight(
        cfg,
        calculation="transmission",
        kmesh=mesh,
        results=_complete_results(),
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
