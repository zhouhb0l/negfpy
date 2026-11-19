"""Portable run-input and hash checks; no material calculation is launched."""

import json
from pathlib import Path
import shutil

import pytest

from negfpp.input import load_input


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "silicon24"


@pytest.fixture
def case(tmp_path):
    target = tmp_path / "case"
    target.mkdir()
    document = json.loads((EXAMPLE / "input.json").read_text())
    shutil.copy2(EXAMPLE / "input.json", target / "input.json")
    for name in document["inputs_sha256"]:
        shutil.copy2(EXAMPLE / name, target / name)
    return target / "input.json", document


def _save(case):
    path, document = case
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_bundled_json_and_all_four_input_hashes_pass():
    config = load_input(EXAMPLE / "input.json")
    assert set(config.files) == {"fc2", "crystal", "coefficients", "metadata"}
    assert all(path.is_file() for path in config.files.values())
    assert config.document["geometry"]["n_device_layers"] == 3
    assert config.document["scba"]["g"] == 0.1
    assert all(len(record["sha256"]) == 64 for record in config.hashes.values())


@pytest.mark.parametrize("section", [None, "scba", "resources"])
def test_unknown_keys_are_rejected(case, section):
    document = case[1]
    (document if section is None else document[section])["unexpected"] = 1
    with pytest.raises(ValueError, match="unknown"):
        load_input(_save(case))


@pytest.mark.parametrize("text", ["{", "[]", "null", '{"schema": "negfpp.run.v1"}'])
def test_malformed_or_incomplete_input_rejected(case, text):
    path = case[0]
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_input(path)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_values_rejected(case, value):
    case[1]["scba"]["g"] = value
    with pytest.raises(ValueError, match="nonfinite"):
        load_input(_save(case))


@pytest.mark.parametrize("section", [None, "scba"])
def test_duplicate_keys_rejected(case, section):
    path = case[0]
    text = path.read_text()
    if section is None:
        text = text.replace('"schema":', '"schema": "negfpp.run.v1", "schema":', 1)
    else:
        text = text.replace('"g":', '"g": 0.1, "g":', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_input(path)


def test_huge_json_integer_is_cleanly_rejected(case):
    case[1]["scba"]["max_evaluations"] = 10**400
    with pytest.raises(ValueError, match="finite number"):
        load_input(_save(case))


def test_nonzero_device_regulator_rejected(case):
    case[1]["reservoirs"]["eta_device"] = 1e-5
    with pytest.raises(ValueError, match="eta_device must be zero"):
        load_input(_save(case))


def test_changed_coefficient_bytes_fail_before_numerical_loading(case):
    path, document = case
    coefficients = path.parent / document["model"]["coefficients"]
    coefficients.write_bytes(coefficients.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="input hash mismatch"):
        load_input(path)


def test_inputs_and_outputs_resolve_from_json_directory(case, tmp_path, monkeypatch):
    path, document = case
    document["run"]["output_dir"] = "results/session"
    _save(case)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    config = load_input(path)
    assert config.output == path.parent / "results" / "session"
    assert all(file.parent == path.parent for file in config.files.values())
    assert not config.output.exists() and not list(elsewhere.iterdir())
