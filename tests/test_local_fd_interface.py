import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from negfpp.input import load_input
from negfpp.local_fd_resources import resource_plan
from negfpp import local_fd_workflow as workflow


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples/silicon1um/input.json"


def test_portable_example_qualified_without_long_calculation():
    config = load_input(EXAMPLE)
    assert config.donor.qualification["all_original_six_passed"]
    assert config.document["geometry"]["center_layers"] == 1840
    assert config.donor.sigmaR.shape == (1, 101, 36, 36)
    assert not config.donor.sigmaR.flags.writeable


@pytest.mark.parametrize("section,key,value", [
    ("geometry", "center_layers", True),
    ("resources", "workers", 17),
    ("resources", "frequency_tile", 0),
    ("reservoirs", "equilibrium_temperature_k", 0),
    ("solver", "krylov_restart", 21),
    ("solver", "flux_rtol", 1e-4),
    ("resources", "exit_reserve_seconds", 2400),
])
def test_invalid_controls_rejected_before_donor_access(tmp_path, section, key, value):
    doc = json.loads(EXAMPLE.read_text())
    doc[section][key] = value
    file = tmp_path / "input.json"
    file.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_input(file)


def test_plan_includes_simultaneous_startup_and_every_frequency():
    controls = json.loads(EXAMPLE.read_text())["resources"]
    plan = resource_plan(1840, 1, 50, controls, available_bytes=100*1024**3)
    assert plan["workers"] == 4 and plan["full_dofs"] == 44208
    assert plan["positive_frequency_nodes"] == 50
    assert plan["startup_workspace_bytes"] > plan["action_workspace_bytes"]
    assert plan["numeric_estimate_bytes"] == (plan["retained_estimate_bytes"]
        + plan["startup_workspace_bytes"] + controls["parent_reserve_bytes"])
    for key in ("max_cache_bytes", "numeric_cap_bytes"):
        small = copy.deepcopy(controls)
        small[key] = 1
        with pytest.raises(ValueError):
            resource_plan(1840, 1, 50, small, available_bytes=100*1024**3)
    with pytest.raises(ValueError, match="available memory"):
        resource_plan(1840, 1, 50, controls, available_bytes=1)


def test_admission_failure_retains_report_and_never_starts_engine(tmp_path, monkeypatch):
    doc = json.loads(EXAMPLE.read_text())
    donor = SimpleNamespace(mesh=SimpleNamespace(n_qpoints=1), grid=SimpleNamespace(positive_indices=range(50)))
    config = SimpleNamespace(document=doc, source=EXAMPLE, manifest=EXAMPLE.parent/"reference.json", output=tmp_path/"run")
    monkeypatch.setattr(workflow, "load_frozen_source", lambda *a, **kw: donor)
    def reject(*args, **kwargs):
        raise ValueError("controlled memory admission failure")
    monkeypatch.setattr(workflow, "resource_plan", reject)
    def unexpected(*args, **kwargs):
        raise AssertionError("No long preparation or worker allowed")
    monkeypatch.setattr(workflow, "prepare_material", unexpected)
    monkeypatch.setattr(workflow, "GOnlyParallelCurrents", unexpected)
    report, output = workflow.run_local_fd(config)
    assert not report["converged"] and report["new_scba_maps"] == 0
    assert "controlled memory admission" in report["failure_reason"]
    assert report == json.loads((output/"report.json").read_text())
    for line in (output/"SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ")
        assert hashlib.sha256((output/name).read_bytes()).hexdigest() == digest
    before = {p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        workflow.run_local_fd(config)
    assert before == {p.name:p.read_bytes() for p in output.iterdir()}
