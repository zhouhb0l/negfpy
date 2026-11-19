"""Atomic rolling evaluated-input SCBA checkpoints, independent of convergence.

A completed physical map is evidence for an initial guess, never imported
convergence. Only its evaluated R/C/D input is stored; no Green function,
unevaluated Born image, proposal vector or controller history is retained.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

import h5py
import numpy as np

_SCHEMA = "negfpp.scba-checkpoint.v1"
_BINDING_SCHEMA = "negfpp.scba-checkpoint-binding.v1"
_CHANNELS = ("retarded", "statistical", "spectral")
_PHYSICAL = {"harmonic", "input_sigma", "input_causality", "left", "right",
             "raw_source", "full_green_local", "active_green", "mapped_sigma"}
_DIMENSIONS = ("full_atoms", "full_dofs", "active_atoms", "active_dofs",
               "energy_centers", "q_points", "frequency_nodes")
_SUPPORT = ("energy_identity_sha256", "geometry_sha256", "full_ndof", "active_dofs",
            "all_outer_derivatives_retained", "cropped_local_derivative_blocks")
_PHYSICS = ("geometry", "grid", "reservoirs", "scale")
_FIELD_UNITS = "dimensionless solver fields; see source_input_json"
_JSON_CAP = 8 * 1024**2
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _plain(value):
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, np.ndarray):
        raise TypeError("Checkpoint scalar metadata cannot contain field arrays")
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _json_bytes(value):
    payload = (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                          allow_nan=False) + "\n").encode("utf-8")
    if len(payload) > _JSON_CAP:
        raise ValueError("Checkpoint scalar metadata exceeds its bounded size")
    return payload


def _positive_integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(name + " must be a positive integer")
    return value


def _digest(value, name):
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(name + " requires lowercase SHA256")
    return value


def _physical_groups(checks, exact_zero):
    required = _PHYSICAL - ({"input_sigma", "input_causality"} if exact_zero else set())
    if not isinstance(checks, dict) or not required <= checks.keys():
        raise ValueError("Checkpoint evaluation is missing original physical checks")
    def ratios(value):
        if isinstance(value, dict):
            if "ratio" in value:
                yield value["ratio"]
            else:
                for item in value.values():
                    yield from ratios(item)
    for name in required:
        values = list(ratios(checks[name]))
        if not values or not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                 for v in values):
            raise ValueError("Checkpoint physical check failed: " + name)


def _record_flags(record):
    groups = {"sigma_residuals": {"retarded", "lesser", "greater"},
              "current_residuals": {"contact_balance", "collision_power", "full_balance"}}
    passed = {}
    for name, keys in groups.items():
        rows = record.get(name)
        if not isinstance(rows, dict) or set(rows) != keys:
            raise ValueError("Checkpoint requires all original six convergence screens")
        ratios = [row.get("ratio") if isinstance(row, dict) else None for row in rows.values()]
        if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value) or value < 0)
               for value in ratios):
            raise ValueError("Checkpoint convergence screen ratio is invalid")
        passed[name] = all(value is not None and value <= 1 for value in ratios)
    expected = dict(sigma_passed=passed["sigma_residuals"], current_passed=passed["current_residuals"],
                    converged=all(passed.values()))
    if any(type(record.get(name)) is not bool or record[name] != value for name, value in expected.items()):
        raise ValueError("Checkpoint convergence flags differ from its original six screens")


def make_binding(config, info):
    """Bind both model backends to physical inputs, units and complete support.

    Filenames and numerical iteration controls are deliberately absent. All
    authenticated material input roles participate, including NEP derivative
    and qualification assets when that backend supplies them.
    """
    doc = config.document
    support = info["support"]
    dimensions = {name: _positive_integer(info[name], name) for name in _DIMENSIONS}
    if (dimensions["full_dofs"] != 3 * dimensions["full_atoms"]
            or dimensions["active_dofs"] != 3 * dimensions["active_atoms"]
            or dimensions["active_dofs"] > dimensions["full_dofs"]):
        raise ValueError("Checkpoint full/active dimensions differ")
    selected_support = {name: support[name] for name in _SUPPORT}
    if (selected_support["all_outer_derivatives_retained"] is not True
            or selected_support["cropped_local_derivative_blocks"] != 0
            or selected_support["full_ndof"] != dimensions["full_dofs"]
            or selected_support["active_dofs"] != dimensions["active_dofs"]):
        raise ValueError("Checkpoint requires complete uncropped endpoint support")
    for name in ("energy_identity_sha256", "geometry_sha256", "model_identity_sha256"):
        if name in support:
            selected_support[name] = _digest(support[name], name)
    roles = {}
    for role, record in config.hashes.items():
        if not isinstance(role, str) or not role:
            raise ValueError("Checkpoint input roles require nonempty names")
        roles[role] = dict(sha256=_digest(record["sha256"], role),
                           bytes=_positive_integer(record["bytes"], role + " bytes"))
    if not roles or not isinstance(info["source_units"], dict) or not info["source_units"]:
        raise ValueError("Checkpoint material inputs and source units are required")
    result = dict(schema=_BINDING_SCHEMA, model_type=doc["model"]["type"],
                  input_roles=roles, physical={name: doc[name] for name in _PHYSICS},
                  g=doc["scba"]["g"], dimensions=dimensions,
                  support=selected_support, source_units=info["source_units"])
    # Detached ordinary JSON types: caller mutations cannot alter a saved binding.
    return json.loads(_json_bytes(result))


def _source_input(payload, binding):
    from .input import parse_json
    if not isinstance(payload, bytes) or not payload or len(payload) > _JSON_CAP:
        raise ValueError("Checkpoint requires bounded original input bytes")
    document = parse_json(payload)
    if (not isinstance(document, dict) or document.get("schema") != "negfpp.run.v1"
            or document.get("model", {}).get("type") != binding["model_type"]
            or document.get("scba", {}).get("g") != binding["g"]
            or {name: document.get(name) for name in _PHYSICS} != binding["physical"]):
        raise ValueError("Checkpoint input bytes and physical binding differ")
    hashes = document.get("inputs_sha256")
    if (not isinstance(hashes, dict)
            or sorted(hashes.values()) != sorted(row["sha256"] for row in binding["input_roles"].values())):
        raise ValueError("Checkpoint input bytes and material digests differ")
    return document


def _metadata(state, decision, binding, source_input_bytes, completed_ordinal, selected_anchor_index):
    index = _positive_integer(state.index, "evaluated state index")
    ordinal = _positive_integer(completed_ordinal, "completed ordinal")
    anchor = _positive_integer(selected_anchor_index, "selected anchor index")
    record = _plain(state.record)
    exact_zero = state.sigma_input is None
    if (ordinal > index or anchor > index or record.get("index") != index
            or exact_zero != (index == 1)):
        raise ValueError("Checkpoint state/record/ordinal/exact-zero identity differs")
    accepted = True if decision is None else decision.get("accepted_as_anchor")
    if (type(accepted) is not bool or (accepted and anchor != index)
            or (not accepted and anchor >= index)
            or (decision is not None and decision.get("selected_anchor_index", anchor) != anchor)):
        raise ValueError("Checkpoint latest trial and selected anchor identities differ")
    checks = _plain(state.evaluation.checks)
    _physical_groups(checks, exact_zero)
    _record_flags(record)
    if record["converged"] and not accepted:
        raise ValueError("A converged evaluated trial cannot be marked rejected")
    _source_input(source_input_bytes, binding)
    return dict(schema=_SCHEMA, complete=True, binding=binding,
                source_input_sha256=hashlib.sha256(source_input_bytes).hexdigest(),
                evaluated_state_index=index, completed_ordinal=ordinal,
                selected_anchor_index=anchor, accepted_as_anchor=accepted,
                record=record, physical_checks=checks,
                currents_w=_plain(state.evaluation.input_currents), decision=_plain(decision),
                evaluated_input_saved=True, exact_zero=exact_zero,
                mapped_image_dyson_evaluated=False, mapped_image_saved=False,
                green_saved=False, solver_history_saved=False,
                source_convergence_is_imported=False)


def _coordinates(grid, mesh, vertex, scale):
    from negfpy.anharmonic.transverse_mapping import _kpoints
    phases = np.asarray(_kpoints(mesh), dtype=np.float64)
    return dict(omega_solver=(grid.values, "dimensionless angular frequency"),
                omega_rad_s=(grid.values * scale.omega0, "rad/s"),
                q_reduced=(phases / (2 * np.pi), "reciprocal lattice fractions"),
                q_phase_rad=(phases, "rad"),
                active_device_indices=(vertex.active_dofs, "zero-based full-device coordinate"))


def _shape(grid, mesh, vertex, binding, scale):
    shape = (mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
    dimensions = binding["dimensions"]
    physical = binding["physical"]
    if (dimensions["full_dofs"] != vertex.full_ndof or dimensions["active_dofs"] != vertex.ndof
            or dimensions["q_points"] != mesh.n_qpoints or dimensions["frequency_nodes"] != grid.n_points
            or physical["grid"] != dict(mesh=list(mesh.shape), step=grid.step, n_positive=grid.n_positive)
            or physical["scale"] != dict(omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar)):
        raise ValueError("Checkpoint binding and fresh dimensions/grid/scale differ")
    return shape


def _payload_size(shape, maximum):
    _positive_integer(maximum, "checkpoint max_bytes")
    size = 3 * math.prod(shape) * np.dtype("complex128").itemsize
    if size > maximum:
        raise ValueError("Checkpoint evaluated R/C/D payload exceeds max_bytes")
    return size


def _chunks(shape):
    # Bounded contiguous I/O slices, including noncontiguous caller-owned fields.
    tile = max(1, min(8, (8 * 1024**2) // (16 * shape[-1]**2)))
    for q in range(shape[0]):
        for start in range(0, shape[1], tile):
            yield q, slice(start, min(shape[1], start + tile))


def _stat(path):
    value = path.stat()
    if not path.is_file():
        raise ValueError("Checkpoint is not a regular file")
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _authenticate(path, expected):
    _digest(expected, "checkpoint_sha256")
    before = _stat(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    if before != _stat(path):
        raise ValueError("Checkpoint changed during authentication")
    if digest.hexdigest() != expected:
        raise ValueError("Checkpoint SHA256 mismatch")
    return before


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_latest(path, state, decision, *, binding, source_input_bytes, grid, mesh,
                 vertex, scale, completed_ordinal, selected_anchor_index, max_bytes):
    """Commit one complete physical evaluated input by atomic single-file replace.

    A failed precommit write leaves the previous checkpoint untouched. The
    temporary file coexists with that checkpoint, bounding retained storage to
    two payloads during publication rather than growing with the map count.
    """
    shape = _shape(grid, mesh, vertex, binding, scale)
    payload_bytes = _payload_size(shape, max_bytes)
    metadata = _metadata(state, decision, binding, source_input_bytes,
                         completed_ordinal, selected_anchor_index)
    metadata_bytes = _json_bytes(metadata)
    if state.sigma_input is not None:
        for name in _CHANNELS:
            values = getattr(state.sigma_input, name)
            if not isinstance(values, np.ndarray) or values.dtype != np.dtype("complex128") or values.shape != shape:
                raise ValueError("Checkpoint input component shape/dtype differs: " + name)
    path = Path(path)
    if path.is_symlink() or not path.parent.is_dir():
        raise ValueError("Checkpoint requires a real file in an existing output directory")
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs.update(schema=_SCHEMA, complete=True,
                                evaluated_state_index=state.index,
                                mapped_image_dyson_evaluated=False,
                                mapped_image_saved=False, green_saved=False, solver_history_saved=False)
            handle.create_dataset("metadata_json", data=np.frombuffer(metadata_bytes, dtype=np.uint8))
            handle.create_dataset("source_input_json", data=np.frombuffer(source_input_bytes, dtype=np.uint8))
            for name, (values, units) in _coordinates(grid, mesh, vertex, scale).items():
                handle.create_dataset(name, data=values).attrs["units"] = units
            group = handle.create_group("sigma_input_evaluated")
            group.attrs.update(exact_zero=state.sigma_input is None, units=_FIELD_UNITS)
            if state.sigma_input is not None:
                for name in _CHANNELS:
                    values = getattr(state.sigma_input, name)
                    dataset = group.create_dataset(name, shape=shape, dtype=np.complex128)
                    for q, block in _chunks(shape):
                        chunk = values[q, block]
                        if not np.isfinite(chunk).all():
                            raise ValueError("Checkpoint input component is nonfinite: " + name)
                        dataset[q, block] = chunk
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        # Hash the complete new inode before replacing the old publication.
        digest = hashlib.sha256()
        with temporary.open("rb") as stream:
            for block in iter(lambda: stream.read(1024**2), b""):
                digest.update(block)
        size = temporary.stat().st_size
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(path=str(path), schema=_SCHEMA, sha256=digest.hexdigest(), bytes=size,
                payload_bytes=payload_bytes, evaluated_state_index=state.index,
                completed_ordinal=completed_ordinal, selected_anchor_index=selected_anchor_index,
                accepted_as_anchor=metadata["accepted_as_anchor"], evaluated_input_saved=True,
                mapped_image_saved=False, green_saved=False, solver_history_saved=False)


def _safe_tree(group, seen):
    for name in group:
        if not isinstance(group.get(name, getlink=True), h5py.HardLink):
            raise ValueError("Checkpoint external/soft links are unsupported")
        item = group[name]
        address = h5py.h5o.get_info(item.id).addr
        if address in seen:
            raise ValueError("Checkpoint aliased/cyclic objects are unsupported")
        seen.add(address)
        if isinstance(item, h5py.Dataset):
            if item.is_virtual or item.external:
                raise ValueError("Checkpoint virtual/external storage is unsupported")
        elif isinstance(item, h5py.Group):
            _safe_tree(item, seen)
        else:
            raise ValueError("Checkpoint HDF5 object is unsupported")


def _json_dataset(handle, name):
    from .input import parse_json
    dataset = handle.get(name)
    if (not isinstance(dataset, h5py.Dataset) or len(dataset.shape) != 1
            or not 0 < dataset.shape[0] <= _JSON_CAP or dataset.dtype != np.dtype("uint8")):
        raise ValueError("Checkpoint JSON dataset shape/dtype/size differs: " + name)
    payload = dataset[...].tobytes()
    return payload, parse_json(payload)


def load_checkpoint(path, expected_sha256, *, binding, grid, mesh, vertex, scale, max_bytes):
    """Load only authenticated evaluated R/C/D as a fresh, unevaluated guess.

    Source convergence and anchor selection are receipts, never admission
    requirements. Every source must nevertheless be a completed physical map
    under the original guards, bound to this exact target model and grid.
    """
    from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
    path = Path(path)
    before = _authenticate(path, expected_sha256)
    shape = _shape(grid, mesh, vertex, binding, scale)
    payload_bytes = _payload_size(shape, max_bytes)
    try:
        with h5py.File(path, "r") as handle:
            _safe_tree(handle, {h5py.h5o.get_info(handle.id).addr})
            expected_names = {"metadata_json", "source_input_json", "sigma_input_evaluated", *_coordinates(grid, mesh, vertex, scale)}
            if set(handle) != expected_names or handle.attrs.get("schema") != _SCHEMA:
                raise ValueError("Checkpoint schema or saved field roles differ")
            for name, expected in (("complete", True), ("mapped_image_dyson_evaluated", False),
                                   ("mapped_image_saved", False), ("green_saved", False), ("solver_history_saved", False)):
                value = handle.attrs.get(name)
                if not isinstance(value, (bool, np.bool_)) or bool(value) != expected:
                    raise ValueError("Checkpoint completion/field-role flag differs: " + name)
            _, metadata = _json_dataset(handle, "metadata_json")
            source_bytes, _ = _json_dataset(handle, "source_input_json")
            if metadata.get("schema") != _SCHEMA or metadata.get("complete") is not True or metadata.get("binding") != binding:
                raise ValueError("Checkpoint physical binding differs from target")
            _source_input(source_bytes, binding)
            if hashlib.sha256(source_bytes).hexdigest() != metadata.get("source_input_sha256"):
                raise ValueError("Checkpoint original input bytes digest differs")
            index = _positive_integer(metadata["evaluated_state_index"], "evaluated state index")
            if isinstance(handle.attrs.get("evaluated_state_index"), (bool, np.bool_)) or handle.attrs.get("evaluated_state_index") != index:
                raise ValueError("Checkpoint field and record indices differ")
            exact_zero = metadata.get("exact_zero")
            if type(exact_zero) is not bool or exact_zero != (index == 1):
                raise ValueError("Checkpoint exact-zero role differs")
            if (metadata.get("evaluated_input_saved") is not True
                    or any(metadata.get(name) is not False for name in
                           ("mapped_image_dyson_evaluated", "mapped_image_saved", "green_saved",
                            "solver_history_saved", "source_convergence_is_imported"))):
                raise ValueError("Checkpoint evaluated-input role is invalid")
            ordinal = _positive_integer(metadata["completed_ordinal"], "completed ordinal")
            anchor = _positive_integer(metadata["selected_anchor_index"], "selected anchor index")
            accepted = metadata.get("accepted_as_anchor")
            if (ordinal > index or anchor > index or type(accepted) is not bool
                    or (accepted and anchor != index) or (not accepted and anchor >= index)
                    or metadata["record"].get("index") != index):
                raise ValueError("Checkpoint latest/selected scalar identities differ")
            _physical_groups(metadata["physical_checks"], exact_zero)
            _record_flags(metadata["record"])
            if metadata["record"]["converged"] and not accepted:
                raise ValueError("Checkpoint converged trial is marked rejected")
            for name, (expected, units) in _coordinates(grid, mesh, vertex, scale).items():
                dataset = handle[name]
                if (not isinstance(dataset, h5py.Dataset) or dataset.shape != expected.shape
                        or dataset.attrs.get("units") != units
                        or (dataset.dtype.kind not in "iu" if name == "active_device_indices" else dataset.dtype != np.dtype("float64"))
                        or not np.array_equal(dataset[...], expected)):
                    raise ValueError("Checkpoint coordinate/ordering/units differ: " + name)
            group = handle["sigma_input_evaluated"]
            if not isinstance(group, h5py.Group):
                raise ValueError("Checkpoint evaluated input group is missing")
            zero_flag = group.attrs.get("exact_zero")
            if (not isinstance(zero_flag, (bool, np.bool_)) or bool(zero_flag) != exact_zero
                    or group.attrs.get("units") != _FIELD_UNITS
                    or set(group) != (set() if exact_zero else set(_CHANNELS))):
                raise ValueError("Checkpoint evaluated input components/units/zero flag differ")
            arrays = []
            for name in _CHANNELS:
                if exact_zero:
                    values = np.zeros(shape, dtype=np.complex128)
                else:
                    dataset = group[name]
                    if not isinstance(dataset, h5py.Dataset) or dataset.shape != shape or dataset.dtype != np.dtype("complex128"):
                        raise ValueError("Checkpoint input component shape/dtype differs: " + name)
                    values = np.empty(shape, dtype=np.complex128, order="C")
                    dataset.read_direct(values)
                    for q, block in _chunks(shape):
                        if not np.isfinite(values[q, block]).all():
                            raise ValueError("Checkpoint input component is nonfinite: " + name)
                values.setflags(write=False)
                arrays.append(values)
        if before != _authenticate(path, expected_sha256):
            raise ValueError("Checkpoint changed while loading")
    except (KeyError, AttributeError, TypeError) as error:
        raise ValueError("Incomplete or invalid checkpoint metadata") from error
    receipt = dict(source_format=_SCHEMA, path=str(path), sha256=expected_sha256,
                   bytes=before[2], payload_bytes=payload_bytes,
                   source_input_sha256=metadata["source_input_sha256"],
                   source_evaluated_state_index=index, source_completed_ordinal=ordinal,
                   source_selected_anchor_index=anchor, source_accepted_as_anchor=accepted,
                   source_reported_converged=metadata["record"]["converged"],
                   source_physical_checks_passed=True,
                   loaded_field_role="sigma_input_evaluated", loaded_components=list(_CHANNELS),
                   initial_guess_is_unevaluated_on_target=True,
                   convergence_history_loaded=False, acceleration_history_loaded=False,
                   green_loaded=False, mapped_image_loaded=False,
                   source_convergence_is_imported=False, fresh_full_map_required=True)
    return ChannelSpectrum(*arrays), receipt
