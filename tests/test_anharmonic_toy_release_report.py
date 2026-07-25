from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from negfpy.anharmonic.toy.cubic_chain import pinned_scalar_chain
from negfpy.anharmonic.toy.refinement import (
    RefinementSpec,
    RefinementStudy,
    run_refinement_study,
)
from negfpy.anharmonic.toy.release_report import (
    DenseV01Acceptance,
    DenseV01PeakMeasurement,
    build_dense_v01_refinement_report,
)
from negfpy.anharmonic.types import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)


@pytest.fixture(scope="module")
def coarse_mixing_study() -> RefinementStudy:
    """A fast stable study that deliberately fails full release diagnostics."""

    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.0,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=15),
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=2,
        mixing=0.3,
        sigma_rtol=3.0e-6,
        green_rtol=3.0e-6,
        identity_rtol=1.0e-7,
        current_rtol=1.0e-4,
        current_atol=1.0e-9,
    )
    return run_refinement_study(
        problem,
        config,
        (RefinementSpec("smaller_mixing", mixing=0.15),),
        max_frequency_points=40,
    )


def test_report_is_strict_deterministic_json_with_complete_provenance(
    coarse_mixing_study: RefinementStudy,
    tmp_path: Path,
) -> None:
    report = build_dense_v01_refinement_report(
        coarse_mixing_study,
        selected_peak_measurements={
            "smaller_mixing": DenseV01PeakMeasurement(
                observable="left_contact_spectrum_peak_height",
            )
        },
    )

    serialized = report.to_json()
    assert serialized == report.to_json()
    document = json.loads(serialized)
    assert document == report.as_dict()
    assert document["schema"] == "negfpy.anharmonic.dense_v0_1_refinement"
    assert document["schema_version"] == 2
    assert document["summary"]["all_input_provenance_complete"]

    required_hashes = {
        "device",
        "left_reservoir",
        "right_reservoir",
        "cubic_vertex",
        "frequency_grid",
        "solver_scale",
        "problem",
        "configuration",
        "initial_state",
        "calculation",
    }
    for observation in [document["reference"], *document["candidates"]]:
        assert set(observation["input_hashes"]) == required_hashes
        assert all(
            len(value) == 64 for value in observation["input_hashes"].values()
        )
        assert observation["result_metadata"]["current_positive"] == "into_device"
        assert observation["result_metadata"]["retarded_strategy"] == (
            "causal_hilbert"
        )
        assert observation["code_provenance_complete"]
        assert observation["code_provenance"]["package_version_consistent"]
        assert observation["study_summary_consistent"]

    peak = document["comparisons"][0]["gates"][
        "selected_peak_refinement"
    ]["measurements"][0]
    expected_reference_peak = float(
        np.max(
            np.abs(
                coarse_mixing_study.reference.result.currents.contact_spectra[
                    0
                ]
            )
        )
    )
    assert peak["reference_value"] == pytest.approx(expected_reference_peak)
    assert len(peak["reference_source_sha256"]) == 64
    assert len(peak["candidate_source_sha256"]) == 64

    digest_document = dict(document)
    digest = digest_document.pop("report_sha256")
    canonical = json.dumps(
        digest_document,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == report.sha256

    destination = tmp_path / "dense-v0.1-refinement.json"
    assert report.write_json(destination) == destination
    archived = json.loads(destination.read_text(encoding="utf-8"))
    assert archived == document


def test_report_records_all_controls_thresholds_and_separate_outcomes(
    coarse_mixing_study: RefinementStudy,
) -> None:
    report = build_dense_v01_refinement_report(
        coarse_mixing_study,
        selected_peak_measurements={
            "smaller_mixing": DenseV01PeakMeasurement(
                observable="left_contact_spectrum_peak_height",
            )
        },
    )
    document = report.as_dict()
    comparison = document["comparisons"][0]

    assert set(comparison["controls"]) == {
        "spacing",
        "cutoff",
        "eta_lead",
        "mixing",
    }
    assert not comparison["controls"]["spacing"]["changed"]
    assert not comparison["controls"]["cutoff"]["changed"]
    assert not comparison["controls"]["eta_lead"]["changed"]
    assert comparison["controls"]["mixing"]["candidate"] == 0.15
    assert comparison["controls"]["mixing"]["changed"]
    assert comparison["controls"]["mixing"][
        "qualifies_for_release_coverage"
    ]
    assert comparison["controls"]["mixing"]["ratio"] == 0.5
    assert comparison["controls"]["mixing"]["reference"] == 0.3
    assert document["acceptance"] == {
        "max_current_relative_change": 5.0e-3,
        "max_selected_peak_relative_change": 5.0e-3,
        "max_sigma_tail_ratio": 1.0e-6,
        "require_converged": True,
        "require_selected_peak_changes": True,
        "release_contract_eligible": True,
    }
    assert comparison["gates"]["current_refinement"]["passed"]
    assert comparison["gates"]["selected_peak_refinement"]["passed"]
    assert comparison["gates"]["selected_peak_refinement"][
        "evidence_verified"
    ]
    assert comparison["gates"]["self_energy_tail"]["passed"]

    # Numerical stability is useful evidence, but these deliberately coarse
    # spectra fail independent solver certification and must not be promoted.
    assert comparison["refinement_passed"]
    assert not comparison["release_certified"]
    assert not report.refinement_passed
    assert not report.release_certified
    assert not document["summary"]["full_control_coverage"]
    assert document["summary"]["control_coverage"] == {
        "cutoff": False,
        "eta_lead": False,
        "mixing": True,
        "spacing": False,
    }
    assert not document["summary"]["all_results_release_certified"]
    assert any(
        "independently release-certified" in failure
        for failure in document["summary"]["failures"]
    )


def test_missing_peak_measurement_is_explicitly_unevaluated_and_fails(
    coarse_mixing_study: RefinementStudy,
) -> None:
    report = build_dense_v01_refinement_report(coarse_mixing_study)
    peak_gate = report.as_dict()["comparisons"][0]["gates"][
        "selected_peak_refinement"
    ]

    assert peak_gate["value"] is None
    assert not peak_gate["evaluated"]
    assert not peak_gate["passed"]
    assert not report.refinement_passed
    assert not report.release_certified


def test_report_validates_acceptance_and_selected_peak_names(
    coarse_mixing_study: RefinementStudy,
) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DenseV01Acceptance(max_current_relative_change=-1.0)
    with pytest.raises(TypeError, match="boolean"):
        DenseV01Acceptance(require_converged=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown candidates"):
        build_dense_v01_refinement_report(
            coarse_mixing_study,
            selected_peak_relative_changes={"not_a_candidate": 0.0},
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        build_dense_v01_refinement_report(
            coarse_mixing_study,
            selected_peak_relative_changes={"smaller_mixing": float("nan")},
        )


def test_legacy_peak_delta_is_diagnostic_only_and_cannot_certify(
    coarse_mixing_study: RefinementStudy,
) -> None:
    report = build_dense_v01_refinement_report(
        coarse_mixing_study,
        selected_peak_relative_changes={"smaller_mixing": 0.0},
    )
    peak_gate = report.as_dict()["comparisons"][0]["gates"][
        "selected_peak_refinement"
    ]

    assert peak_gate["evaluated"]
    assert peak_gate["passed"]
    assert not peak_gate["evidence_verified"]
    assert peak_gate["legacy_relative_delta_only"]
    assert not report.release_certified


def test_peak_measurement_computes_delta_and_rejects_zero_scale() -> None:
    measurement = DenseV01PeakMeasurement(
        observable="left_contact_spectrum_peak_height",
    )
    assert measurement.observable == "left_contact_spectrum_peak_height"
    with pytest.raises(ValueError, match="observable must be one of"):
        DenseV01PeakMeasurement(
            observable="caller_supplied_peak",
        )


def test_release_control_coverage_rejects_wrong_refinement_directions() -> None:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.0,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=15),
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )
    study = run_refinement_study(
        problem,
        config,
        (
            RefinementSpec("coarser_spacing", step_factor=2.0),
            RefinementSpec("smaller_cutoff", cutoff_factor=0.8),
            RefinementSpec("larger_eta", eta_factor=2.0),
            RefinementSpec("changed_mixing", mixing=0.1),
        ),
        max_frequency_points=40,
    )
    measurements = {
        item.name: DenseV01PeakMeasurement()
        for item in study.candidates
    }
    report = build_dense_v01_refinement_report(
        study,
        selected_peak_measurements=measurements,
    )

    assert report.as_dict()["summary"]["control_coverage"] == {
        "cutoff": False,
        "eta_lead": False,
        "mixing": True,
        "spacing": False,
    }
    assert not report.refinement_passed
    assert not report.release_certified


def test_report_detects_spoofed_refinement_summary(
    coarse_mixing_study: RefinementStudy,
) -> None:
    tampered = RefinementStudy(
        reference=replace(
            coarse_mixing_study.reference,
            left_current=coarse_mixing_study.reference.left_current + 1.0,
        ),
        candidates=coarse_mixing_study.candidates,
        deltas=coarse_mixing_study.deltas,
    )
    report = build_dense_v01_refinement_report(
        tampered,
        selected_peak_measurements={
            "smaller_mixing": DenseV01PeakMeasurement()
        },
    )

    document = report.as_dict()
    assert not document["reference"]["study_summary_consistent"]
    assert not document["summary"]["all_study_summaries_consistent"]
    assert not report.release_certified


def test_one_candidate_cannot_cover_all_independent_controls() -> None:
    device, lead, vertex = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=0.0,
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=15),
    )
    config = SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=8,
        min_iterations=3,
        consecutive_passes=3,
    )
    study = run_refinement_study(
        problem,
        config,
        (
            RefinementSpec(
                "all_at_once",
                step_factor=0.5,
                cutoff_factor=1.2,
                eta_factor=0.5,
                mixing=0.1,
            ),
        ),
        max_frequency_points=100,
    )
    report = build_dense_v01_refinement_report(
        study,
        selected_peak_measurements={
            "all_at_once": DenseV01PeakMeasurement()
        },
    )

    assert report.as_dict()["summary"]["control_coverage"] == {
        "cutoff": False,
        "eta_lead": False,
        "mixing": False,
        "spacing": False,
    }
    assert not report.refinement_passed
