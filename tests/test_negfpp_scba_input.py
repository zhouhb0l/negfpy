"""Public checkpoint and saved-complete-energy JSON contracts, without a solver."""
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from negfpp.input import load_input

EXAMPLE = Path(__file__).resolve().parents[1]/"examples"/"silicon24"

@pytest.fixture
def case(tmp_path):
    doc=json.loads((EXAMPLE/"input.json").read_text())
    for name in doc["inputs_sha256"]:
        shutil.copy2(EXAMPLE/name,tmp_path/name)
    return tmp_path/"input.json",doc

def load(case):
    path,doc=case
    path.write_text(json.dumps(doc))
    return load_input(path)

@pytest.mark.parametrize("name",["self-energy-latest.h5","checkpoint-001.h5"])
def test_checkpoint_filename_resolves_inside_fresh_output(case,name):
    case[1]["checkpoint"]={"file":name}
    assert load(case).document["checkpoint"]["file"]==name

@pytest.mark.parametrize("value",[{},None,{"file":"../outside.h5"},{"file":"/tmp/out.h5"},
 {"file":"fields.h5"},{"file":"report.json"},{"file":"a/b.h5"},{"file":"x.h5","extra":True}])
def test_checkpoint_rejects_ambiguous_or_overwriting_paths(case,value):
    case[1]["checkpoint"]=value
    with pytest.raises(ValueError):
        load(case)

def test_saved_complete_energy_binds_exact_three_material_assets(case):
    path,doc=case
    names={"ifc2":"ifc2.hdf5","archive":"complete.hdf5","metadata":"nep-model.json"}
    for role,name in names.items():
        (path.parent/name).write_bytes((role+" authentic fixture bytes").encode())
    doc["harmonic"]={"ifc2":names["ifc2"]}
    doc["model"]={"type":"saved_complete_energy","archive":names["archive"],"metadata":names["metadata"]}
    doc["inputs_sha256"]={name:hashlib.sha256((path.parent/name).read_bytes()).hexdigest() for name in names.values()}
    config=load(case)
    assert set(config.files)==set(names)
    (path.parent/names["archive"]).write_bytes(b"changed cubic asset")
    with pytest.raises(ValueError,match="input hash mismatch"):
        load(case)


def test_complete_energy_optional_original_archive_is_independently_hashed(case):
    path,doc=case
    names={"ifc2":"ifc2.hdf5","archive":"complete.hdf5","metadata":"nep-model.json","source_archive":"original.hdf5"}
    for role,name in names.items():
        (path.parent/name).write_bytes((role+" fixture").encode())
    doc["harmonic"]={"ifc2":names["ifc2"]}
    doc["model"]={"type":"saved_complete_energy",**{k:v for k,v in names.items() if k!="ifc2"}}
    doc["inputs_sha256"]={name:hashlib.sha256((path.parent/name).read_bytes()).hexdigest() for name in names.values()}
    assert set(load(case).files)==set(names)
    del doc["inputs_sha256"][names["source_archive"]]
    with pytest.raises(ValueError,match="exactly"):
        load(case)
