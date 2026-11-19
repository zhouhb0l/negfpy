"""Compact donor integrity and evaluated-state admission, without transport."""

import hashlib
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from negfpp.local_fd import source


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
MANIFEST_SHA = "feb67f886337acb5b8366a689542463c90fa74c6bed4a0683174f3b660946b68"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def portable_copy(tmp_path):
    root = tmp_path / "examples"
    shutil.copytree(EXAMPLES / "silicon1um", root / "silicon1um")
    (root / "silicon24").mkdir()
    for name in ("fc2.full.hdf5", "phono3py_disp.yaml", "coefficients.npy", "model.json"):
        shutil.copyfile(EXAMPLES / "silicon24" / name, root / "silicon24" / name)
    return root / "silicon1um" / "reference.json"


def write_document(manifest_path, manifest, role, document):
    entry = manifest["documents"][role]
    path = manifest_path.parent / entry["path"]
    path.write_text(json.dumps(document, indent=2) + "\n")
    entry.update(bytes=path.stat().st_size, sha256=sha(path))


def test_real_compact_donor_loads_without_campaign_tree(portable_copy):
    donor = source.load_frozen_source(portable_copy, expected_sha256=MANIFEST_SHA)
    assert donor.sigmaR.shape == (1, 101, 36, 36)
    assert not donor.sigmaR.flags.writeable
    assert not donor.active_dofs.flags.writeable
    assert sha(portable_copy) == MANIFEST_SHA
    assert hashlib.sha256(donor.sigmaR.tobytes()).hexdigest() == (
        "17765664e4be5f7058b3261e6ac91148db3eb0852ab667c6b3c26bd4817dd80f")
    np.testing.assert_array_equal(donor.grid.values, -donor.grid.values[::-1])
    assert donor.qualification["evaluated_input"] is True
    assert donor.qualification["all_original_six_passed"] is True
    for gate in ("psd", "retarded_reality", "acoustic_asr"):
        assert donor.qualification["validation"][gate]["passed"] is True
    assert all(path.is_file() and path.is_absolute() for path in donor.files.values())
    with h5py.File(portable_copy.parent / "reference.h5") as handle:
        assert "sigma_born_image_unevaluated" not in handle
        assert "green_evaluated" not in handle


@pytest.mark.parametrize("failure", ["unconverged", "sixth_gate", "physical_gate", "exit", "image", "bytes"])
def test_bad_donors_are_rejected_before_array_access(portable_copy, monkeypatch, failure):
    manifest = json.loads(portable_copy.read_text())
    report = json.loads((portable_copy.parent / "source-report.json").read_text())
    receipt = json.loads((portable_copy.parent / "source-qualification.json").read_text())
    if failure == "unconverged":
        report["converged"] = False
    elif failure == "sixth_gate":
        report["selected_anchor"]["record"]["current_residuals"]["full_balance"]["ratio"] = 1.01
    elif failure == "physical_gate":
        report["selected_anchor"]["physical_checks"]["input_causality"]["causal_reconstruction"]["ratio"] = 1.01
    elif failure == "exit":
        receipt["actual_exit_code"] = 1
    elif failure == "image":
        manifest["retarded"]["role"] = "sigma_born_image_unevaluated/retarded"
    elif failure == "bytes":
        with (portable_copy.parent / "reference.h5").open("ab") as handle:
            handle.write(b"changed")
    if failure in {"unconverged", "sixth_gate", "physical_gate"}:
        write_document(portable_copy, manifest, "report", report)
        manifest["origin"]["report_sha256"] = manifest["documents"]["report"]["sha256"]
        receipt["report_sha256"] = manifest["origin"]["report_sha256"]
    write_document(portable_copy, manifest, "qualification", receipt)
    portable_copy.write_text(json.dumps(manifest, indent=2) + "\n")

    def arrays_forbidden(*args, **kwargs):
        pytest.fail("Invalid donor reached HDF5 array access")

    monkeypatch.setattr(source.h5py, "File", arrays_forbidden)
    with pytest.raises(ValueError):
        source.load_frozen_source(portable_copy)


def test_manifest_and_material_hashes_are_required_before_arrays(portable_copy, monkeypatch):
    def arrays_forbidden(*args, **kwargs):
        pytest.fail("A hash mismatch reached HDF5 array access")

    monkeypatch.setattr(source.h5py, "File", arrays_forbidden)
    with pytest.raises(ValueError, match="manifest hash"):
        source.load_frozen_source(portable_copy, expected_sha256="0" * 64)
    model = portable_copy.parent.parent / "silicon24" / "model.json"
    model.write_bytes(model.read_bytes() + b" ")
    with pytest.raises(ValueError, match="size/hash"):
        source.load_frozen_source(portable_copy, expected_sha256=MANIFEST_SHA)
