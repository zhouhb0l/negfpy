"""Reference-method input compatibility and limits, without calculations."""

import json
from pathlib import Path
import shutil

import pytest

from negfpp.input import load_input


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture
def reference_input(tmp_path):
    old = tmp_path / "silicon24"
    new = tmp_path / "silicon5layer"
    old.mkdir()
    new.mkdir()
    source = EXAMPLES / "silicon24"
    document = json.loads((source / "input.json").read_text())
    for name in document["inputs_sha256"]:
        shutil.copy2(source / name, old / name)
    path = new / "input.json"
    document = json.loads((EXAMPLES / "silicon5layer" / "input.json").read_text())
    path.write_text(json.dumps(document))
    return path, document


def _write(reference_input):
    path, document = reference_input
    path.write_text(json.dumps(document))
    return path


def test_original_input_remains_compatible_without_reference_controls():
    config = load_input(EXAMPLES / "silicon24" / "input.json")
    assert config.document["scba"]["method"] == "anderson1"
    assert config.document["scba"]["max_evaluations"] == 60
    assert "reference" not in config.document


def test_five_layer_example_hashes_and_reference_controls_pass():
    config = load_input(EXAMPLES / "silicon5layer" / "input.json")
    assert config.document["geometry"]["n_device_layers"] == 5
    assert config.document["geometry"]["interaction_layers"] == [1, 2, 3]
    assert config.document["scba"]["method"] == "reference-seeded"
    assert config.document["scba"]["max_evaluations"] == 18
    assert config.document["reference"]["max_tangent_directions"] == 6
    assert all(file.parent.name == "silicon24" for file in config.files.values())


def test_reference_controls_can_be_omitted_to_use_existing_defaults(reference_input):
    reference_input[1].pop("reference")
    assert load_input(_write(reference_input)).document["scba"]["method"] == "reference-seeded"


def test_reference_full_map_cap_is_eighteen(reference_input):
    reference_input[1]["scba"]["max_evaluations"] = 19
    with pytest.raises(ValueError, match="at most 18"):
        load_input(_write(reference_input))


@pytest.mark.parametrize("value", [7, -1, True, 1.5])
def test_invalid_tangent_caps_are_rejected(reference_input, value):
    reference_input[1]["reference"]["max_tangent_directions"] = value
    with pytest.raises(ValueError):
        load_input(_write(reference_input))


def test_zero_tangent_cap_is_a_supported_bounded_control(reference_input):
    reference_input[1]["reference"]["max_tangent_directions"] = 0
    assert load_input(_write(reference_input)).document["reference"]["max_tangent_directions"] == 0


@pytest.mark.parametrize("method", ["anderson1", "linear"])
def test_reference_controls_require_reference_method(reference_input, method):
    reference_input[1]["scba"]["method"] = method
    with pytest.raises(ValueError, match="reference controls require"):
        load_input(_write(reference_input))


@pytest.mark.parametrize("value", [None, [], "external-checkpoint"])
def test_reference_controls_must_be_an_object(reference_input, value):
    reference_input[1]["reference"] = value
    with pytest.raises(ValueError, match="reference must be an object"):
        load_input(_write(reference_input))


@pytest.mark.parametrize(("key", "value"), [("warmup_evaluations", 1), ("tangent_interval", 2)])
def test_existing_warmup_and_tangent_interval_are_fixed(reference_input, key, value):
    reference_input[1]["reference"][key] = value
    with pytest.raises(ValueError, match="two warmups, interval three"):
        load_input(_write(reference_input))


def test_unknown_reference_keys_are_rejected(reference_input):
    reference_input[1]["reference"]["checkpoint"] = "foreign-state.h5"
    with pytest.raises(ValueError, match="unknown"):
        load_input(_write(reference_input))


def test_incomplete_reference_object_is_rejected(reference_input):
    reference_input[1]["reference"].pop("initial_tangent_estimate_seconds")
    with pytest.raises(ValueError, match="missing"):
        load_input(_write(reference_input))
