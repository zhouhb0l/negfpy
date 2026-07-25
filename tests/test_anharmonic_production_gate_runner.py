from __future__ import annotations

import os
from pathlib import Path

import pytest

import negfpy.anharmonic.toy.production_gate_runner as production_runner

from negfpy.anharmonic.toy.production_gate_runner import (
    open_three_mode_release_fixtures,
    ProductionToyGateBlockedError,
    run_release_production_toy_gate,
)


def test_open_release_fixture_matrix_is_complete_and_one_control_refined() -> None:
    specifications = open_three_mode_release_fixtures()
    names = [name for name, _ in specifications]
    fixtures = {name: fixture for name, fixture in specifications}

    assert len(specifications) == 13
    assert len(set(names)) == len(names)
    base = [
        fixture
        for name, fixture in specifications
        if name.startswith("base_")
    ]
    assert {
        (fixture.temperature, fixture.g)
        for fixture in base
    } == {
        (temperature, g_value)
        for temperature in (0.0, 0.7)
        for g_value in (-0.02, -0.01, 0.0, 0.01, 0.02)
    }

    anchor = fixtures["base_T0p7_g0p02"]
    spacing = fixtures["refine_spacing_half"]
    cutoff = fixtures["refine_cutoff_6"]
    eta = fixtures["refine_eta_half"]

    assert spacing.grid.step == pytest.approx(anchor.grid.step / 2.0)
    assert spacing.grid.cutoff == pytest.approx(anchor.grid.cutoff)
    assert spacing.eta_lead == anchor.eta_lead

    assert cutoff.grid.step == anchor.grid.step
    assert cutoff.grid.cutoff > anchor.grid.cutoff
    assert cutoff.eta_lead == anchor.eta_lead

    assert eta.grid.step == anchor.grid.step
    assert eta.grid.cutoff == anchor.grid.cutoff
    assert eta.eta_lead == pytest.approx(anchor.eta_lead / 2.0)



def test_release_runner_orders_multidimensional_before_open_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    valid_result = type(
        "Result",
        (),
        {
            "diagnostics": type("Diagnostics", (), {"passed": True})(),
            "release_certified": True,
        },
    )()
    direct = type(
        "Direct",
        (),
        {
            "successful": True,
            "status": "converged",
            "result": valid_result,
        },
    )()
    selected_config = type(
        "Config",
        (),
        {"certification_eligible": True},
    )()
    multidimensional = type(
        "Report",
        (),
        {"outcomes": (direct,), "config": selected_config},
    )()

    monkeypatch.setattr(
        production_runner,
        "full_multidimensional_release_config",
        lambda: "locked-config",
    )

    def run_multidimensional(config: object, **kwargs: object) -> object:
        events.append("multidimensional")
        assert config == "locked-config"
        assert kwargs["checkpoint_dir"] == tmp_path / "checkpoints"
        assert kwargs["environment_sha256"] == "d" * 64
        return multidimensional

    def run_open_matrix(**kwargs: object) -> tuple[str]:
        events.append("open")
        assert kwargs["environment_lock_sha256"] == "d" * 64
        return ("open-evidence",)

    monkeypatch.setattr(
        production_runner,
        "run_multidimensional_release_benchmark",
        run_multidimensional,
    )
    monkeypatch.setattr(
        production_runner,
        "evaluate_open_three_mode_release_matrix",
        run_open_matrix,
    )
    monkeypatch.setattr(
        production_runner,
        "build_production_toy_gate_report",
        lambda runs, report, **kwargs: (
            events.append("build")
            or (runs, report, kwargs["validation_tier"])
        ),
    )

    combined = run_release_production_toy_gate(
        checkpoint_dir=tmp_path / "checkpoints",
        environment_sha256="d" * 64,
    )
    assert events == ["multidimensional", "open", "build"]
    assert combined[0] == ("open-evidence",)
    assert combined[1] is multidimensional
    assert combined[2] == "release"



def test_open_matrix_propagates_environment_lock_to_every_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, str | None]] = []
    with pytest.raises(ValueError, match="environment_lock_sha256"):
        production_runner.evaluate_open_three_mode_release_matrix(
            (),
            environment_lock_sha256="invalid",
        )

    def evaluate(
        name: str,
        fixture: object,
        **kwargs: str | None,
    ) -> str:
        calls.append((name, fixture, kwargs["environment_lock_sha256"]))
        return name

    monkeypatch.setattr(
        production_runner,
        "evaluate_open_three_mode_born",
        evaluate,
    )
    first = object()
    second = object()
    result = production_runner.evaluate_open_three_mode_release_matrix(
        (("first", first), ("second", second)),  # type: ignore[arg-type]
        environment_lock_sha256="f" * 64,
    )

    assert result == ("first", "second")
    assert calls == [
        ("first", first, "f" * 64),
        ("second", second, "f" * 64),
    ]

def test_release_runner_archives_converged_invalid_direct_and_skips_open_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    invalid_result = type(
        "Result",
        (),
        {
            "diagnostics": type("Diagnostics", (), {"passed": True})(),
            "release_certified": False,
        },
    )()
    direct = type(
        "Direct",
        (),
        {
            "successful": True,
            "status": "converged",
            "result": invalid_result,
        },
    )()

    class FailedReport:
        outcomes = (direct,)
        config = type("Config", (), {"certification_eligible": True})()

        @staticmethod
        def to_json(*, indent: int | None = 2) -> str:
            assert indent == 2
            return '{"release_certified":false,"strict":true}'

    failed = FailedReport()
    monkeypatch.setattr(
        production_runner,
        "run_multidimensional_release_benchmark",
        lambda *args, **kwargs: failed,
    )

    def forbidden_open_matrix() -> object:
        raise AssertionError("open matrix must be blocked")

    monkeypatch.setattr(
        production_runner,
        "evaluate_open_three_mode_release_matrix",
        forbidden_open_matrix,
    )
    artifact = tmp_path / "artifacts" / "multidimensional_failure.json"
    with pytest.raises(ProductionToyGateBlockedError) as raised:
        run_release_production_toy_gate(
            checkpoint_dir=tmp_path / "checkpoints",
            environment_sha256="e" * 64,
            multidimensional_failure_report_path=artifact,
        )
    assert raised.value.report is failed
    assert raised.value.artifact_path == artifact
    assert artifact.read_text(encoding="utf-8") == (
        '{"release_certified":false,"strict":true}\n'
    )
    assert not tuple(artifact.parent.glob(".*.tmp"))

@pytest.mark.nightly
@pytest.mark.skipif(
    os.environ.get("NEGFPY_RUN_FULL_ANHARMONIC_TOY_GATE") != "1",
    reason="full production toy gate is enabled only by the nightly job",
)
def test_full_production_toy_gate() -> None:
    """Execute, archive, and enforce the complete preregistered release gate."""

    checkpoint_dir = Path(os.environ["NEGFPY_TOY_GATE_CHECKPOINT_DIR"])
    environment_sha256 = os.environ["NEGFPY_VALIDATION_ENV_SHA256"]
    report = run_release_production_toy_gate(
        checkpoint_dir=checkpoint_dir,
        environment_sha256=environment_sha256,
        multidimensional_failure_report_path=(
            checkpoint_dir.parent / "multidimensional_toy_gate_failure.json"
        ),
    )
    output = Path(
        os.environ.get(
            "NEGFPY_TOY_GATE_REPORT",
            "validation-artifacts/production_toy_gate.json",
        )
    )
    report.write_json(output, indent=2)
    assert report.release_certified, report.as_dict()["summary"]["failures"]
