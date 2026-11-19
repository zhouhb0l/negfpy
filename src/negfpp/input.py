"""Strict, portable JSON run inputs and external-file authentication."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re


_SECTIONS = {
    "model": {"type", "coefficients", "metadata"},
    "harmonic": {"fc2", "crystal"},
    "geometry": {"transport_supercell_matrix", "n_device_layers", "interaction_layers"},
    "grid": {"mesh", "step", "n_positive"},
    "reservoirs": {"left_temperature_k", "right_temperature_k", "eta_lead", "eta_device"},
    "scale": {"omega0_rad_s", "hbar_j_s"},
    "scba": {"g", "method", "max_evaluations", "max_seconds", "mixing", "workers",
             "external_tile", "bin_tile", "max_bytes", "max_contraction_work",
             "max_dyson_evaluations", "sigma_rtol", "sigma_atol", "current_rtol",
             "current_atol_w", "identity_rtol", "identity_atol", "time_reserve_seconds",
             "initial_map_estimate_seconds", "next_map_safety_factor"},
    "resources": {"model_max_bytes", "vertex_max_bytes", "vertex_workspace_bytes",
                  "harmonic_max_bytes", "contact_max_bytes", "q_samples",
                  "max_base_evaluations", "max_geometry_evaluations", "max_phase_evaluations"},
    "run": {"name", "output_dir"},
}


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=reject)


def read_json(path):
    return parse_json(Path(path).read_text(encoding="utf-8"))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _object(value, keys, name):
    if not isinstance(value, dict) or set(value) != keys:
        found = set(value) if isinstance(value, dict) else set()
        raise ValueError(f"{name}: missing {sorted(keys-found)}; unknown {sorted(found-keys)}")


def _number(value, name, minimum=0, strict=False, integer=False):
    try:
        finite = isinstance(value, (int, float)) and math.isfinite(value)
    except OverflowError:
        finite = False
    if isinstance(value, bool) or not finite:
        raise ValueError(f"{name} must be a finite number")
    if integer and not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < minimum or (strict and value == minimum):
        raise ValueError(f"{name} is below its allowed minimum")


@dataclass(frozen=True)
class RunInput:
    source: Path
    document: dict
    files: dict
    hashes: dict
    output: Path
    restart: dict | None = None


def load_input(path):
    source = Path(path).resolve(strict=True)
    doc = read_json(source)
    if not isinstance(doc, dict):
        raise ValueError("input must be a JSON object")
    if doc.get("schema") == "negfpp.local-fd.v1":
        from .local_fd_input import load_local_fd_input
        return load_local_fd_input(source, doc)
    _object({key:value for key,value in doc.items() if key not in {"reference", "initial_self_energy"}}, set(_SECTIONS) | {"schema", "inputs_sha256"}, "input")
    if doc["schema"] != "negfpp.run.v1":
        raise ValueError("schema must be negfpp.run.v1")
    for name, keys in _SECTIONS.items():
        _object(doc[name], keys, name)
    if doc["model"]["type"] != "saved_site_energy":
        raise ValueError("version 1.0 supports model.type=saved_site_energy")
    for section in ("grid", "reservoirs", "scale", "resources"):
        for key, value in doc[section].items():
            if key == "mesh":
                continue
            integer = section == "resources" or key == "n_positive"
            strict = section != "reservoirs" or key == "eta_lead"
            _number(value, f"{section}.{key}", strict=strict, integer=integer)
    if doc["reservoirs"]["eta_device"] != 0:
        raise ValueError("eta_device must be zero; an implicit device bath is unsupported")
    mesh = doc["grid"]["mesh"]
    if not isinstance(mesh, list) or len(mesh) != 3 or mesh[2] != 1:
        raise ValueError("grid.mesh must be [n1,n2,1]")
    for value in mesh:
        _number(value, "mesh", strict=True, integer=True)
    geometry = doc["geometry"]
    _number(geometry["n_device_layers"], "n_device_layers", strict=True, integer=True)
    matrix = geometry["transport_supercell_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 3 or any(not isinstance(row,list) or len(row)!=3 for row in matrix):
        raise ValueError("transport_supercell_matrix must be a 3 by 3 integer matrix")
    for row in matrix:
        for value in row:
            if isinstance(value,bool) or not isinstance(value,int):
                raise ValueError("transport_supercell_matrix requires integers")
    centers = geometry["interaction_layers"]
    if not isinstance(centers, list) or not centers or len(set(centers)) != len(centers):
        raise ValueError("interaction_layers must contain unique zero-based layer indices")
    for value in centers:
        _number(value, "interaction_layers", integer=True)
        if value >= geometry["n_device_layers"]:
            raise ValueError("interaction layer is outside the device")
    scba = doc["scba"]
    if scba["method"] not in ("anderson1", "linear", "reference-seeded"):
        raise ValueError("scba.method must be anderson1, linear or reference-seeded")
    reference = doc.get("reference")
    if "reference" in doc and not isinstance(reference,dict):
        raise ValueError("reference must be an object when provided")
    if scba["method"] != "reference-seeded" and reference is not None:
        raise ValueError("reference controls require scba.method=reference-seeded")
    if scba["method"] == "reference-seeded":
        if reference is not None:
            _object(reference, {"warmup_evaluations", "tangent_interval", "max_tangent_directions", "initial_tangent_estimate_seconds"}, "reference")
            for key,value in reference.items():
                _number(value, "reference."+key, integer=key!="initial_tangent_estimate_seconds")
            if reference["warmup_evaluations"]!=2 or reference["tangent_interval"]!=3 or reference["max_tangent_directions"]>6:
                raise ValueError("reference SCBA requires two warmups, interval three and at most six tangent directions")
        if isinstance(scba["max_evaluations"], (int,float)) and scba["max_evaluations"]>18:
            raise ValueError("reference-seeded SCBA permits at most 18 nonlinear map attempts")
    integer_keys = {"max_evaluations", "workers", "external_tile", "bin_tile", "max_bytes", "max_contraction_work", "max_dyson_evaluations"}
    for key, value in scba.items():
        if key != "method":
            _number(value, f"scba.{key}", strict=key in integer_keys or key in {"mixing", "max_seconds", "initial_map_estimate_seconds"}, integer=key in integer_keys)
    # Validate all native convergence/resource controls before any input preparation.
    from negfpy.anharmonic.complete_energy import ChannelSCBAConfig
    ChannelSCBAConfig(**{key:value for key,value in scba.items() if key not in {"g", "method"}})
    files, hashes = {}, {}
    expected = doc["inputs_sha256"]
    if not isinstance(expected, dict):
        raise ValueError("inputs_sha256 must map input filenames to SHA-256 values")
    references = {"fc2":doc["harmonic"]["fc2"], "crystal":doc["harmonic"]["crystal"],
                  "coefficients":doc["model"]["coefficients"], "metadata":doc["model"]["metadata"]}
    if any(not isinstance(value,str) or not value for value in references.values()) or set(expected) != set(references.values()):
        raise ValueError("inputs_sha256 must cover exactly the four referenced input files")
    for key, value in references.items():
        if not isinstance(expected[value],str) or not re.fullmatch(r"[0-9a-f]{64}", expected[value]):
            raise ValueError(f"invalid SHA-256 for {value}")
        file = (source.parent / value).resolve(strict=True)
        if not file.is_file():
            raise ValueError(f"input is not a regular file: {value}")
        actual = sha256_file(file)
        if actual != expected[value]:
            raise ValueError(f"input hash mismatch: {value}")
        files[key] = file
        hashes[key] = {"file":value, "bytes":file.stat().st_size, "sha256":actual}
    if not isinstance(doc["run"]["name"],str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", doc["run"]["name"]):
        raise ValueError("run.name must be a short filename-safe identifier")
    output_name = doc["run"]["output_dir"]
    if not isinstance(output_name,str) or not output_name:
        raise ValueError("run.output_dir must be a nonempty path")
    output = (source.parent / output_name).resolve()
    restart = None
    if "initial_self_energy" in doc:
        from .restart import resolve_restart
        restart = resolve_restart(source, doc["initial_self_energy"])
    return RunInput(source, doc, files, hashes, output, restart)
