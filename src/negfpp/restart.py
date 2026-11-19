"""Authenticated evaluated self-energy inputs for an independent SCBA run.

A saved result admits an initial guess only. Its mapped image, Green fields,
controller history and convergence are never imported into the new solve.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

_KEYS = {"fields", "fields_sha256", "input", "input_sha256", "report", "report_sha256"}
_ROLES = ("fields", "input", "report")
_CHECKSUM_KEYS = {"checksums", "checksums_sha256"}
_LEGACY_VERSIONS = {"1.0.0", "1.0.1", "1.0.2"}
_COMPONENTS = ("retarded", "statistical", "spectral")
_PHYSICAL_GROUPS = {"harmonic", "input_sigma", "input_causality", "left", "right",
                    "raw_source", "full_green_local", "active_green", "mapped_sigma"}
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _stat(path):
    value = path.stat()
    if not path.is_file():
        raise ValueError("Restart input is not a regular file: " + str(path))
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _authenticate(path, expected):
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if before != _stat(path):
        raise ValueError("Restart file changed during authentication: " + str(path))
    if digest.hexdigest() != expected:
        raise ValueError("Restart SHA256 mismatch: " + str(path))
    return before


def resolve_restart(source_path, spec):
    """Authenticate three files, optionally with the original legacy checksum file."""
    if not isinstance(spec, dict) or set(spec) not in (_KEYS, _KEYS | _CHECKSUM_KEYS):
        raise ValueError("initial_self_energy requires fields/input/report hashes and optionally both checksums keys")
    result = {}
    base = Path(source_path).resolve().parent
    for role in _ROLES + (("checksums",) if "checksums" in spec else ()):
        value, digest = spec[role], spec[role + "_sha256"]
        if (not isinstance(value, str) or not value.strip()
                or Path(value).is_absolute() or "\\" in value):
            raise ValueError("initial_self_energy." + role + " must be a relative filename")
        if not isinstance(digest, str) or not _SHA.fullmatch(digest):
            raise ValueError("initial_self_energy." + role + "_sha256 requires lowercase SHA256")
        path = (base / value).resolve(strict=True)
        _authenticate(path, digest)
        result[role], result[role + "_sha256"] = path, digest
    return result


def _json(path):
    from .input import read_json
    if path.stat().st_size > 8 * 1024**2:
        raise ValueError("Restart metadata exceeds the bounded JSON size")
    value = read_json(path)
    if not isinstance(value, dict):
        raise ValueError("Restart metadata must be a JSON object")
    return value


def _ratio(value):
    return (not isinstance(value, bool) and isinstance(value, (int, float))
            and math.isfinite(value) and 0 <= value <= 1)


def _six(record):
    groups = {"sigma_residuals": {"retarded", "lesser", "greater"},
              "current_residuals": {"contact_balance", "collision_power", "full_balance"}}
    return (all(record.get(key) is True for key in ("converged", "sigma_passed", "current_passed"))
            and all(set(record.get(key, {})) == names for key, names in groups.items())
            and all(_ratio(row.get("ratio")) for key in groups for row in record[key].values()))


def _ratios(value):
    if isinstance(value, dict):
        if "ratio" in value:
            yield value["ratio"]
        else:
            for item in value.values():
                yield from _ratios(item)


def _physical(checks, index):
    required = _PHYSICAL_GROUPS - ({"input_sigma", "input_causality"} if index == 1 else set())
    if not isinstance(checks, dict) or not required <= checks.keys():
        raise ValueError("Restart source is missing original physical checks")
    for name in required:
        values = list(_ratios(checks[name]))
        if not values or not all(_ratio(value) for value in values):
            raise ValueError("Restart source physical check failed: " + name)
    # inverse and *_exports remain original diagnostics, not acceptance gates.


def _cross_binding(report, restart):
    """Bind the selected bytes to one source result, not three independent pins."""
    checksum_proof = False
    if "checksums" in restart:
        path = restart["checksums"]
        if path.stat().st_size > 8 * 1024**2:
            raise ValueError("Restart checksum metadata exceeds its bounded size")
        entries = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2 or not _SHA.fullmatch(parts[0]):
                raise ValueError("Invalid original restart SHA256SUMS entry")
            name = parts[1].removeprefix("*")
            if not name or name in entries or Path(name).is_absolute() or "\\" in name:
                raise ValueError("Invalid or duplicate original restart checksum filename")
            entries[name] = parts[0]
        for role in _ROLES:
            selected = restart[role]
            try:
                name = selected.relative_to(path.parent).as_posix()
            except ValueError as exc:
                raise ValueError("Legacy restart files must remain with their original checksum file") from exc
            if entries.get(name) != restart[role + "_sha256"]:
                raise ValueError("Original restart SHA256SUMS does not bind selected " + role)
        checksum_proof = True
    fields = report.get("fields_file")
    if fields is not None:
        if (not isinstance(fields, dict) or set(fields) != {"sha256", "bytes"}
                or fields["sha256"] != restart["fields_sha256"]
                or isinstance(fields["bytes"], bool) or not isinstance(fields["bytes"], int)
                or fields["bytes"] != restart["fields"].stat().st_size):
            raise ValueError("Restart report does not bind the selected fields file")
        kind = "report_fields_file"
    elif report.get("version") in _LEGACY_VERSIONS and checksum_proof:
        kind = "original_legacy_checksums"
    else:
        raise ValueError("Restart needs report.fields_file or authenticated original legacy SHA256SUMS")
    original = report.get("source_input_sha256")
    if not isinstance(original, str) or not _SHA.fullmatch(original):
        raise ValueError("Restart report requires its original source-input SHA256")
    reserialized = original != restart["input_sha256"]
    if reserialized and not (report.get("version") in _LEGACY_VERSIONS and checksum_proof):
        raise ValueError("Restart source input hash differs without authenticated legacy writer proof")
    return dict(source_file_binding=kind, legacy_checksums_authenticated=checksum_proof,
                source_original_input_sha256=original,
                selected_saved_input_sha256=restart["input_sha256"],
                legacy_input_reserialized=reserialized)


def _qualify(report, source_input, config, vertex, info):
    if (report.get("schema") != "negfpp.result.v1"
            or report.get("status") != "converged" or report.get("converged") is not True
            or report.get("stop_reason") != "fixed_point" or report.get("failure_reason") is not None):
        raise ValueError("Restart requires a bound converged negfpp fixed-point result")
    if source_input.get("schema") != "negfpp.run.v1":
        raise ValueError("Restart source must be a finite-device negfpp run")
    anchor, last = report["selected_anchor"], report["last_evaluated"]
    index = anchor["index"]
    if (isinstance(index, bool) or not isinstance(index, int) or index < 1
            or last["index"] != index or anchor["record"]["index"] != index
            or last["record"] != anchor["record"]
            or anchor.get("input_sigma_evaluated") is not True
            or anchor.get("mapped_image_dyson_evaluated") is not False
            or anchor.get("accepted_as_anchor") is not True or not _six(anchor["record"])):
        raise ValueError("Restart source is not a selected evaluated six-screen fixed point")
    _physical(anchor["physical_checks"], index)
    target = config.document
    for name in ("geometry", "grid", "reservoirs", "scale"):
        if source_input[name] != target[name]:
            raise ValueError("Restart source " + name + " differs from target")
    if (source_input["model"].get("type") != "saved_site_energy"
            or target["model"].get("type") != "saved_site_energy"
            or source_input["scba"]["g"] != target["scba"]["g"]):
        raise ValueError("Restart source cubic model type or coupling differs")
    roles = {"fc2": source_input["harmonic"]["fc2"], "crystal": source_input["harmonic"]["crystal"],
             "coefficients": source_input["model"]["coefficients"], "metadata": source_input["model"]["metadata"]}
    if set(report["inputs"]) != set(roles) or set(source_input["inputs_sha256"]) != set(roles.values()):
        raise ValueError("Restart source requires exactly four bound material input roles")
    for role, filename in roles.items():
        saved, current = report["inputs"][role], config.hashes[role]
        if (saved["file"] != filename or source_input["inputs_sha256"][filename] != saved["sha256"]
                or saved["sha256"] != current["sha256"] or saved["bytes"] != current["bytes"]):
            raise ValueError("Restart source material input differs: " + role)
    old = report["model"]
    dimensions = {"full_dofs": vertex.full_ndof, "active_dofs": vertex.ndof,
                  "full_atoms": vertex.full_ndof // 3, "active_atoms": vertex.ndof // 3,
                  "q_points": info["q_points"], "frequency_nodes": info["frequency_nodes"],
                  "energy_centers": info["energy_centers"]}
    if any(old.get(key) != value or info.get(key) != value for key, value in dimensions.items()):
        raise ValueError("Restart source device/support dimensions differ")
    if (old.get("source_units") != info["source_units"]
            or old.get("fitting_repeated") is not False or info.get("fitting_repeated") is not False):
        raise ValueError("Restart source units or restored fitted model differ")
    for support in (old["support"], info["support"]):
        if (support.get("all_outer_derivatives_retained") is not True
                or support.get("cropped_local_derivative_blocks") != 0
                or support.get("full_ndof") != vertex.full_ndof
                or support.get("active_dofs") != vertex.ndof):
            raise ValueError("Restart requires complete uncropped cubic endpoint support")
    for name in ("energy_identity_sha256", "geometry_sha256", "model_identity_sha256"):
        if name in {"energy_identity_sha256", "geometry_sha256"} and name not in info["support"]:
            raise ValueError("Fresh target is missing its model/support identity: " + name)
        if name in info["support"] and old["support"].get(name) != info["support"][name]:
            raise ValueError("Restart source model/support identity differs: " + name)
    return index


def _safe_tree(group, seen):
    """Inspect link/storage metadata only, including unused Green/mapped groups."""
    import h5py
    for name in group:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ValueError("Restart HDF5 contains a nonlocal or symbolic link")
        item = group[name]
        address = h5py.h5o.get_info(item.id).addr
        if address in seen:
            raise ValueError("Restart HDF5 contains aliased/cyclic storage")
        seen.add(address)
        if isinstance(item, h5py.Group):
            _safe_tree(item, seen)
        elif isinstance(item, h5py.Dataset):
            if item.is_virtual or item.external:
                raise ValueError("Restart HDF5 contains virtual or external storage")
        else:
            raise ValueError("Unsupported restart HDF5 object")


def _dataset(group, name, shape, dtype=None, units=None):
    import h5py
    item = group.get(name)
    if not isinstance(item, h5py.Dataset) or item.shape != shape:
        raise ValueError("Restart dataset is missing or has the wrong shape: " + name)
    if dtype is not None and item.dtype != dtype:
        raise ValueError("Restart dataset has the wrong native dtype: " + name)
    if units is not None and item.attrs.get("units") != units:
        raise ValueError("Restart dataset has the wrong units: " + name)
    return item


def load_self_energy(config, grid, mesh, vertex, scale, info):
    """Return only copied evaluated native R/C/D plus a source-admission receipt."""
    import h5py
    import numpy as np
    from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
    from negfpy.anharmonic.transverse_mapping import _kpoints

    restart = config.restart
    if restart is None:
        raise ValueError("No initial_self_energy descriptor was provided")
    roles = _ROLES + (("checksums",) if "checksums" in restart else ())
    before = {role: _authenticate(restart[role], restart[role + "_sha256"]) for role in roles}
    try:
        source_input, report = _json(restart["input"]), _json(restart["report"])
        binding_receipt = _cross_binding(report, restart)
        index = _qualify(report, source_input, config, vertex, info)
        shape = (mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
        payload_bytes = 3 * math.prod(shape) * np.dtype("complex128").itemsize
        if payload_bytes > config.document["scba"]["max_bytes"]:
            raise ValueError("Restart self-energy payload exceeds scba.max_bytes")
        if (grid.n_points != info["frequency_nodes"] or mesh.n_qpoints != info["q_points"]
                or list(mesh.shape) != config.document["grid"]["mesh"]
                or grid.step != config.document["grid"]["step"]
                or grid.n_positive != config.document["grid"]["n_positive"]
                or scale.omega0 != config.document["scale"]["omega0_rad_s"]
                or scale.hbar != config.document["scale"]["hbar_j_s"]):
            raise ValueError("Fresh grid/scale differs from target input")
        with h5py.File(restart["fields"], "r") as handle:
            _safe_tree(handle, {h5py.h5o.get_info(handle.id).addr})
            saved_index = handle.attrs.get("evaluated_state_index")
            mapped = handle.attrs.get("mapped_image_dyson_evaluated")
            if (handle.attrs.get("schema") != "negfpp.fields.v1"
                    or isinstance(saved_index, (bool, np.bool_))
                    or not isinstance(saved_index, (int, np.integer)) or saved_index != index
                    or not isinstance(mapped, (bool, np.bool_)) or bool(mapped)
                    or handle.attrs.get("omega0_rad_s") != scale.omega0
                    or handle.attrs.get("hbar_j_s") != scale.hbar):
                raise ValueError("Restart fields role/index/physical scale differs")
            phases = np.asarray(_kpoints(mesh))
            coordinates = (
                ("omega_solver", grid.values, "dimensionless angular frequency"),
                ("omega_rad_s", grid.values * scale.omega0, "rad/s"),
                ("q_phase_rad", phases, "rad"),
                ("q_reduced", phases / (2 * np.pi), "reciprocal lattice fractions"),
            )
            for name, expected, units in coordinates:
                dataset = _dataset(handle, name, expected.shape, np.dtype("float64"), units)
                if not np.array_equal(dataset[...], expected):
                    raise ValueError("Restart coordinate ordering differs: " + name)
            active = _dataset(handle, "active_device_indices", (vertex.ndof,))
            if active.dtype.kind not in "iu" or not np.array_equal(active[...], vertex.active_dofs):
                raise ValueError("Restart active endpoint ordering differs")
            group = handle.get("sigma_input_evaluated")
            if not isinstance(group, h5py.Group):
                raise ValueError("Restart requires sigma_input_evaluated, not a mapped image")
            zero = group.attrs.get("exact_zero")
            if (not isinstance(zero, (bool, np.bool_))
                    or group.attrs.get("units") != "dimensionless solver fields; see SolverScale in input.json"
                    or bool(zero) != (index == 1)):
                raise ValueError("Restart input role/units or exact-zero flag is invalid")
            if zero:
                if index != 1 or len(group) != 0:
                    raise ValueError("Exact-zero input has unexpected index/components")
                arrays = [np.zeros(shape, dtype=np.complex128) for _ in _COMPONENTS]
            else:
                if set(group) != set(_COMPONENTS):
                    raise ValueError("Restart evaluated input requires all three native components")
                arrays = []
                for name in _COMPONENTS:
                    dataset = _dataset(group, name, shape, np.dtype("complex128"))
                    # No arithmetic/casting: preserve signed zero and every source bit.
                    values = np.empty(shape, dtype=np.complex128, order="C")
                    dataset.read_direct(values)
                    if any(not np.isfinite(values[point]).all() for point in np.ndindex(shape[:2])):
                        raise ValueError("Restart input component contains nonfinite values: " + name)
                    arrays.append(values)
            for values in arrays:
                values.setflags(write=False)
        for role in roles:
            if before[role] != _authenticate(restart[role], restart[role + "_sha256"]):
                raise ValueError("Restart file changed while loading: " + str(restart[role]))
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Incomplete or invalid restart source metadata: " + str(exc)) from exc
    receipt = {
        **binding_receipt,
        "kind": "external_evaluated_self_energy_input",
        "source_index": index, "source_input_sha256": restart["input_sha256"],
        "source_report_sha256": restart["report_sha256"], "source_fields_sha256": restart["fields_sha256"],
        "source_schema": "negfpp.result.v1", "source_convergence_admitted": True,
        "source_exact_zero": bool(zero), "input_channels": list(_COMPONENTS),
        "shape": list(shape), "payload_bytes": payload_bytes,
        "source_input_evaluated": True, "mapped_image_or_green_loaded": False,
        "controller_history_loaded": False, "source_convergence_inherited": False,
        "new_target_evaluation_required": True,
    }
    return ChannelSpectrum(*arrays), receipt
