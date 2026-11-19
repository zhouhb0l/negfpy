"""Failure and bounded-result reporting without input assembly or SCBA."""

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from negfpp.input import load_input
import negfpp.workflow as workflow
from negfpy.anharmonic.complete_energy import ChannelSCBARecord
from negfpy.anharmonic.transverse_solver import TransverseCurrents


@dataclass(frozen=True)
class Ratio:
    ratio: float


@pytest.fixture
def config(tmp_path):
    example = Path(__file__).resolve().parents[1] / "examples" / "silicon24" / "input.json"
    return replace(load_input(example), output=tmp_path / "run")


@pytest.fixture
def prepared(monkeypatch):
    calls = []
    def prepare(config, deadline):
        calls.append((config, deadline))
        return (
            object(), object(), object(), (object(), object()),
            SimpleNamespace(vertex=object()), object(), {"full_atoms": 24},
        )
    monkeypatch.setattr(workflow, "prepare", prepare)
    return calls


def _state(index):
    sigma = {name: Ratio(2.0) for name in ("retarded", "lesser", "greater")}
    current = {name: Ratio(0.5) for name in ("contact_balance", "collision_power", "full_balance")}
    record = ChannelSCBARecord(index, 0.01, 0.01, sigma, current, {}, False, True, False)
    return SimpleNamespace(
        index=index, record=record,
        evaluation=SimpleNamespace(
            input_currents=TransverseCurrents(1.0, -1.0, 0.0),
            checks={"synthetic_test_guard": {"passed": True}},
        ),
    )


def _saved_report(output):
    report = json.loads((output / "report.json").read_text())
    hashes = {}
    for line in (output / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        hashes[name] = digest
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert "report.json" in hashes and "input.json" in hashes
    return report


def test_solver_exception_preserves_completed_counts_and_scalar_anchor(config, prepared, monkeypatch):
    def fail_after_callbacks(*args, **kwargs):
        kwargs["on_start"](1, None, None)
        kwargs["on_evaluation"](_state(1), None)
        kwargs["on_start"](2, object(), {"kind": "anderson1"})
        kwargs["on_evaluation"](_state(2), {"accepted_as_anchor": False})
        raise RuntimeError("controlled solver interruption")
    monkeypatch.setattr(workflow, "solve_from_zero", fail_after_callbacks)

    report, output = workflow.run(config)
    saved = _saved_report(output)
    assert len(prepared) == 1
    assert report == saved
    assert saved["status"] == "failed" and not saved["converged"]
    assert saved["failure_reason"] == "RuntimeError: controlled solver interruption"
    assert saved["counts"] == {"attempted": 2, "completed": 2, "selected": 1, "rejected": 1}
    assert saved["selected_anchor"]["index"] == 1
    assert saved["last_evaluated"]["index"] == 2
    assert saved["selected_anchor"]["currents_w"]["left"] == 1.0
    rows = [json.loads(line) for line in (output / "iterations.jsonl").read_text().splitlines()]
    assert [row["index"] for row in rows] == [1, 2]
    assert [row["accepted_as_anchor"] for row in rows] == [True, False]
    assert not (output / "fields.h5").exists()
    assert saved["scientific_release_certified"] is False


def test_zero_map_bounded_stop_does_not_advertise_nonexistent_outputs(config, prepared, monkeypatch):
    def bounded(*args, **kwargs):
        return SimpleNamespace(
            result=SimpleNamespace(
                state=None, converged=False, failure_reason=None,
                stop_reason="max_seconds", plan={"synthetic_plan": True},
            ),
            attempted=0, completed=0, selected=0, rejected=0, records=(),
        )
    monkeypatch.setattr(workflow, "solve_from_zero", bounded)

    report, output = workflow.run(config)
    saved = _saved_report(output)
    assert len(prepared) == 1
    assert report == saved
    assert saved["status"] == "not_converged" and saved["stop_reason"] == "max_seconds"
    assert not saved["converged"] and saved["failure_reason"] is None
    assert all(value == 0 for value in saved["counts"].values())
    assert "iterations" not in saved["outputs"] and "fields" not in saved["outputs"]
    assert not (output / "iterations.jsonl").exists()
    assert all((output / name).is_file() for name in saved["outputs"].values())
    assert len((output / "convergence.csv").read_text().splitlines()) == 1


def test_existing_output_directory_is_refused_without_overwrite(config, monkeypatch):
    config.output.mkdir()
    marker = config.output / "keep.txt"
    marker.write_bytes(b"existing result must remain unchanged\n")
    before = {path.name: path.read_bytes() for path in config.output.iterdir()}
    def unexpected(*args, **kwargs):
        raise AssertionError("Calculation preparation must not start")
    monkeypatch.setattr(workflow, "prepare", unexpected)
    monkeypatch.setattr(workflow, "solve_from_zero", unexpected)

    with pytest.raises(FileExistsError):
        workflow.run(config)
    assert {path.name: path.read_bytes() for path in config.output.iterdir()} == before


@pytest.mark.parametrize("formatting", ["pretty", "compact-noncanonical"])
def test_original_input_bytes_are_preserved_with_matching_report_hash(tmp_path, monkeypatch, formatting):
    """Hash the captured source bytes, not a normalized JSON serialization."""
    import shutil
    example = Path(__file__).resolve().parents[1] / "examples" / "silicon24"
    case = tmp_path / "input-case"
    case.mkdir()
    document = json.loads((example / "input.json").read_text())
    document["run"]["output_dir"] = "output"
    for name in document["inputs_sha256"]:
        shutil.copy2(example / name, case / name)
    if formatting == "pretty":
        source_bytes = ("\n " + json.dumps(document, sort_keys=True, indent=7) + "\n\n").encode()
    else:
        source_bytes = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    source = case / "input.json"
    source.write_bytes(source_bytes)
    loaded = load_input(source)

    def stop_before_science(*args, **kwargs):
        raise RuntimeError("controlled stop before input assembly")
    monkeypatch.setattr(workflow, "prepare", stop_before_science)
    report, output = workflow.run(loaded)
    saved = _saved_report(output)
    expected = hashlib.sha256(source_bytes).hexdigest()
    assert source.read_bytes() == (output / "input.json").read_bytes() == source_bytes
    assert saved["source_input_sha256"] == report["source_input_sha256"] == expected
    assert saved["status"] == "failed"
    assert saved["failure_reason"] == "RuntimeError: controlled stop before input assembly"
    assert saved["counts"] == {"attempted": 0, "completed": 0, "selected": 0, "rejected": 0}


def test_input_mutation_after_validation_is_rejected_before_output_or_preparation(tmp_path, monkeypatch):
    import shutil
    example = Path(__file__).resolve().parents[1] / "examples" / "silicon24"
    case = tmp_path / "input-case"
    case.mkdir()
    document = json.loads((example / "input.json").read_text())
    document["run"]["output_dir"] = "output"
    for name in document["inputs_sha256"]:
        shutil.copy2(example / name, case / name)
    source = case / "input.json"
    source.write_text(json.dumps(document))
    loaded = load_input(source)
    document["scba"]["g"] = .2
    source.write_text(json.dumps(document))

    def unexpected(*args, **kwargs):
        raise AssertionError("Changed input must not reach preparation")
    monkeypatch.setattr(workflow, "prepare", unexpected)
    with pytest.raises(ValueError, match="Input JSON changed after validation"):
        workflow.run(loaded)
    assert not loaded.output.exists()
