"""Authenticated historical R/C/D inputs, admitted only as a new-run guess.

Portable schema ``negfpp.historical-initial-self-energy.v1`` contains:
``source={format, evidence={name: {file, sha256, bytes}}}``, ``identity``,
``identity_bindings={identity_JSON_pointer: {evidence, pointer}}``, and
``arrays={retarded|statistical|spectral: {file,sha256,bytes,shape,dtype}}``.
The evidence named ``state`` is an unchanged native channel manifest or a
completed native HDF publication result. Every physical identity leaf is
cross-bound to authenticated original JSON; only the public model-type label
is normalized. Extraction metadata is not a physical asset identity.

The source frequency grid, scale, model, complete support, Cartesian endpoint
ordering and geometry must match. Only q sampling and reservoir temperatures
may differ. Periodic tensor-product linear interpolation has convex weights
and commutes with q inversion; it is an initializer, never a target result.
No bias scaling, dimensional interpolation, projection, or history is loaded.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path

_SCHEMA = "negfpp.historical-initial-self-energy.v1"
_MODE = "same_model_closest_q_temperature_guess"
_COMPONENTS = ("retarded", "statistical", "spectral")
_IDENTITY_KEYS = {
    "model_type", "g", "full_dofs", "active_device_indices", "n_device_layers",
    "interaction_layers", "mesh", "frequency_nodes", "frequency_step",
    "omega0_rad_s", "hbar_j_s", "temperatures_kelvin", "eta_lead", "eta_device",
    "geometry_sha256", "energy_identity_sha256", "all_outer_derivatives_retained",
    "cropped_local_derivative_blocks", "source_units", "physical_inputs",
}
_NATIVE_SCHEMAS = {
    "negfpy.transverse-channel-scba-checkpoint.v1",
    "negfpy.nep-seeded-fitted-input-channel-checkpoint.v1",
    "negfpy.nep3-mesh-warm-input-channel-checkpoint.v1",
}


def _integer(value, minimum=0):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _number(value, minimum=0):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and value >= minimum)


def _pointer(value, pointer):
    if not isinstance(pointer, str) or not pointer.startswith("/"):
        raise ValueError("Historical evidence requires an absolute JSON pointer")
    for token in pointer[1:].split("/"):
        token = token.replace("~1", "/").replace("~0", "~")
        try:
            value = value[int(token)] if isinstance(value, list) else value[token]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ValueError("Historical evidence pointer is absent: " + pointer) from exc
    return value


def _leaves(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            escaped = key.replace("~", "~0").replace("/", "~1")
            yield from _leaves(item, prefix + "/" + escaped)
    else:
        yield prefix, value


def _relative(base, name):
    if (not isinstance(name, str) or not name or "\\" in name
            or Path(name).is_absolute() or ".." in Path(name).parts):
        raise ValueError("Historical evidence/array files must stay inside the portable bundle")
    candidate = base / name
    if candidate.is_symlink():
        raise ValueError("Historical evidence/array files must not be symbolic links")
    path = candidate.resolve(strict=True)
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("Historical evidence/array path escapes the portable bundle")
    return path


def _pin(base, row, *, max_bytes):
    from .restart import _authenticate, _SHA
    if (not isinstance(row, dict) or set(row) != {"file", "sha256", "bytes"}
            or not isinstance(row["sha256"], str) or not _SHA.fullmatch(row["sha256"])
            or not _integer(row["bytes"], 1) or row["bytes"] > max_bytes):
        raise ValueError("Invalid or excessive historical file pin")
    path = _relative(base, row["file"])
    if path.stat().st_size != row["bytes"]:
        raise ValueError("Historical file size differs from its pin")
    return path, _authenticate(path, row["sha256"])


def _state(source, evidence):
    from .restart import _physical, _six
    state = evidence["state"]
    kind = source["format"]
    if kind == "native_channel_manifest":
        if (state.get("schema") not in _NATIVE_SCHEMAS
                or state.get("sigma_input_evaluated") is not True
                or state.get("mapped_sigma_dyson_evaluated") is not False
                or state.get("input_sigma_is_zero") is not False
                or state.get("accepted_restart") is not False
                or state.get("representation", state.get("binding", {}).get("representation"))
                != "retarded-statistical-spectral-v1"):
            raise ValueError("Historical native state is not an evaluated R/C/D input publication")
        index, record = state["iteration"], state["record"]
        selected = record.get("accepted_as_anchor", record.get("controller_decision", {}).get("accepted_as_anchor"))
        if selected is not True:
            raise ValueError("Historical native state was not selected as an anchor")
        checks = record.get("physical_checks", record.get("checks"))
        expected = {prefix + name for prefix in ("sigma_input_", "mapped_sigma_") for name in _COMPONENTS}
        if set(state["artifacts"]) != expected:
            raise ValueError("Historical native publication requires all six original artifact pins")
    elif kind == "native_hdf_result":
        if (state.get("schema") not in {"negfpy.materialTest.scba-result.v2", "negfpp.validation.native-independent-result.v1"}
                or state.get("complete") is not True or state.get("passed") is not True
                or state.get("converged") is not True or state.get("failure_reason") is not None
                or state.get("stop_reason") != "fixed_point"
                or state.get("self_energy", {}).get("all_original_six_passed") is not True):
            raise ValueError("Historical HDF source must be a completed converged native result")
        index = state["active_map"]
        rows = [row for row in state["records"] if row.get("record", {}).get("index") == index]
        if len(rows) != 1:
            raise ValueError("Historical HDF source does not bind one selected evaluated record")
        record, checks = rows[0]["record"], rows[0]["physical_checks"]
        if rows[0].get("currents_w") != state.get("selected_current_w"):
            raise ValueError("Historical HDF selected currents do not bind the final evaluated input")
        expected = {role + "/" + name for role in
                    ("selected_evaluated_input", "separate_unevaluated_mapped_image") for name in _COMPONENTS}
        if set(state["self_energy"]["components"]) != expected:
            raise ValueError("Historical HDF result requires all six original component hashes")
    else:
        raise ValueError("Unsupported historical evidence format")
    if not _integer(index, 2) or record.get("index") != index or not _six(record):
        raise ValueError("Historical input requires an original selected six-screen fixed point")
    _physical(checks, index)
    return state, index, record


def _hdf_evidence(path, state):
    """Read bounded original coordinates/attributes, never a self-energy field."""
    import h5py
    import numpy as np
    from .restart import _safe_tree
    publication = state["self_energy"]
    with h5py.File(path, "r") as handle:
        _safe_tree(handle, {h5py.h5o.get_info(handle.id).addr})
        attrs = handle.attrs
        if (attrs.get("schema") != "negfpy.materialTest.converged-self-energy.v2"
                or not isinstance(attrs.get("complete"), (bool, np.bool_)) or not bool(attrs["complete"])
                or not isinstance(attrs.get("all_original_six_passed"), (bool, np.bool_)) or not bool(attrs["all_original_six_passed"])
                or not isinstance(attrs.get("mapped_image_has_been_Dyson_evaluated"), (bool, np.bool_))
                or bool(attrs["mapped_image_has_been_Dyson_evaluated"])
                or attrs.get("selected_evaluated_input_index") != state["active_map"]):
            raise ValueError("Historical HDF original evaluated-input publication flags differ")
        raw = attrs.get("source_json")
        if not isinstance(raw, str) or len(raw.encode()) > 2 * 1024**2:
            raise ValueError("Historical HDF source metadata exceeds its bound")
        source = json.loads(raw)
        selected = next(row for row in state["records"] if row["record"]["index"] == state["active_map"])
        if (source["original_record"] != selected["record"]
                or source["physical_checks"] != selected["physical_checks"]
                or source["evaluated_input_currents"] != selected["currents_w"]):
            raise ValueError("Historical HDF original proofs do not bind the selected result")
        result = dict(source_json=source, full_dofs=int(attrs["full_device_dofs"]),
                      frequency_step=float(attrs["grid_step"]))
        for name, key in (("active_dof_indices", "active_device_indices"), ("mesh_shape", "mesh"),
                          ("negative_q_indices", "negative_q_indices"), ("mesh_addresses", "mesh_addresses"),
                          ("grid_values", "grid_values")):
            item = handle[name]
            if not isinstance(item, h5py.Dataset) or item.size > 1000000 or item.dtype.kind not in "iuf":
                raise ValueError("Historical HDF coordinate is invalid or excessive")
            result[key] = item[...].tolist()
        for role in ("selected_evaluated_input", "separate_unevaluated_mapped_image"):
            group = handle[role]
            if (group.attrs.get("role") != role
                    or not isinstance(group.attrs.get("Dyson_evaluated"), (bool, np.bool_))
                    or bool(group.attrs["Dyson_evaluated"]) != (role == "selected_evaluated_input")):
                raise ValueError("Historical HDF original field roles differ")
            for name in _COMPONENTS:
                item = group[name]
                expected_shape = (math.prod(result["mesh"]), len(result["grid_values"]),
                                  len(result["active_device_indices"]), len(result["active_device_indices"]))
                if (not isinstance(item, h5py.Dataset) or item.dtype != np.dtype("complex128")
                        or item.shape != expected_shape
                        or item.attrs.get("sha256_float64_complex_c_order") != publication["components"][role + "/" + name]):
                    raise ValueError("Historical HDF component proof differs from the original result")
        return result


def _source_identity(source, state, evidence, identity):
    """Prevent a different authenticated profile from relabelling the selected state."""
    if source["format"] == "native_channel_manifest":
        binding, physics = state["binding"], state["binding"]["physics"]
        expected = dict(full_dofs=binding["full_device_dofs"], active_device_indices=binding["feedback_dofs"],
            energy_identity_sha256=binding["energy_identity_sha256"],
            geometry_sha256=binding.get("geometry_sha256", physics.get("expected_geometry_sha256")),
            mesh=physics["mesh"], frequency_nodes=physics.get("frequency_nodes", 2 * physics.get("n_positive", 0) + 1),
            frequency_step=physics.get("frequency_step", physics.get("grid_step")),
            n_device_layers=physics.get("device_layers", physics.get("n_device_layers")),
            interaction_layers=physics.get("center_layers", physics.get("interaction_layers")),
            temperatures_kelvin=physics["temperatures_kelvin"], g=physics["g"],
            eta_lead=physics["eta_lead"], eta_device=physics["eta_device"],
            model_type="saved_complete_energy" if "direct_nep_derivatives" in binding else "saved_site_energy")
        if "scale" in physics:
            expected.update(omega0_rad_s=physics["scale"]["omega0"], hbar_j_s=physics["scale"]["hbar"])
    else:
        metadata, fields = state["input_metadata"], evidence["fields"]
        physics = metadata["physical_contract"]
        expected = dict(full_dofs=fields["full_dofs"], active_device_indices=fields["active_device_indices"],
            energy_identity_sha256=metadata["energy_identity_sha256"], geometry_sha256=metadata["geometry_sha256"],
            mesh=fields["mesh"], frequency_nodes=len(fields["grid_values"]), frequency_step=fields["frequency_step"],
            temperatures_kelvin=physics["temperatures_kelvin"], g=physics["coupling_g"],
            omega0_rad_s=physics["omega0_rad_s"], hbar_j_s=physics["hbar_j_s"],
            eta_lead=physics["eta_lead"], eta_device=physics["eta_device"],
            n_device_layers=metadata["geometry_descriptor"]["device_cells"],
            interaction_layers=metadata["geometry_descriptor"]["center_layers"])
        shape = _mesh(fields["mesh"])
        if (fields["mesh_addresses"] != _addresses(shape)
                or fields["negative_q_indices"] != [_negative(index, shape) for index in range(math.prod(shape))]):
            raise ValueError("Historical HDF source q-coordinate ordering differs")
        expected_grid = [(index - (len(fields["grid_values"]) // 2)) * fields["frequency_step"] for index in range(len(fields["grid_values"]))]
        if fields["grid_values"] != expected_grid:
            raise ValueError("Historical HDF source signed frequency coordinates differ")
    different = [key for key, value in expected.items() if identity[key] != value]
    if different:
        raise ValueError("Historical identity does not describe its original selected state: " + ", ".join(different))


def _fresh_auxiliary(source, state, evidence, identity):
    """A later proof may fill metadata gaps only when it re-evaluated these bytes."""
    names = {"fresh_validation", "fresh_validation_manifest"}
    if not names.intersection(evidence):
        return
    from .restart import _physical, _six
    if not names <= evidence.keys() or source["format"] != "native_channel_manifest":
        raise ValueError("Fresh auxiliary proof requires both the result and selected-final manifest")
    report, manifest = (evidence[name] for name in ("fresh_validation", "fresh_validation_manifest"))
    seed = dict(mode="own-final-evaluated-input-only", source_index=state["iteration"],
                manifest_sha256=source["evidence"]["state"]["sha256"])
    if (report.get("schema") != "negfpp.validation.fitted-scba-result.v1"
            or report.get("passed") is not True or report.get("converged") is not True
            or report.get("failure_reason") is not None or report.get("stop_reason") != "fixed_point"
            or report.get("seed") != seed or manifest.get("seed") != seed
            or manifest.get("schema") != "negfpp.validation.fitted-selected-final.v1"
            or manifest.get("sigma_input_evaluated") is not True
            or manifest.get("mapped_sigma_dyson_evaluated") is not False
            or manifest.get("old_history_imported") is not False
            or manifest.get("all_original_six_passed") is not True
            or manifest.get("original_physical_guards_passed") is not True
            or manifest.get("selected_record") != report.get("selected_record")
            or not _six(report["selected_record"])):
        raise ValueError("Fresh auxiliary result is not bound to this original evaluated source")
    selected = [row for row in report["evaluations"] if row["record"] == report["selected_record"]]
    if len(selected) != 1 or selected[0].get("accepted_as_anchor") is not True:
        raise ValueError("Fresh auxiliary result is missing its selected evaluated physical proof")
    _physical(selected[0]["physical_checks"], report["selected_record"]["index"])
    physics, target = report["physics"], report["target"]
    expected = dict(mesh=physics["mesh"], temperatures_kelvin=physics["temperatures_kelvin"],
        frequency_nodes=physics["frequency_nodes"], frequency_step=physics["frequency_step"],
        g=physics["g"], n_device_layers=physics["device_layers"], interaction_layers=physics["center_layers"],
        eta_lead=physics["eta_lead"], eta_device=physics["eta_device"],
        omega0_rad_s=physics["scale"]["omega0"], hbar_j_s=physics["scale"]["hbar"],
        full_dofs=target["full_dofs"], active_device_indices=manifest["feedback_dofs"])
    if (any(identity[key] != value for key, value in expected.items())
            or manifest["physics"] != physics or manifest["target"] != target
            or manifest["selected_new_native_index"] != report["selected_record"]["index"]
            or manifest["full_device_dofs"] != identity["full_dofs"]):
        raise ValueError("Fresh auxiliary proof describes a different physical source")
    # The original and new NPY writers retained the identical input bytes;
    # this also excludes a proof of a later, changed nonlinear candidate.
    for name in _COMPONENTS:
        original = state["artifacts"]["sigma_input_" + name]
        actual = manifest["artifacts"]["sigma_input_" + name + ".npy"]
        if actual["sha256"] != original["file_sha256"] or actual["bytes"] != original["file_bytes"]:
            raise ValueError("Fresh auxiliary proof does not retain the original selected INPUT bytes")


def _mesh(value):
    if (not isinstance(value, list) or len(value) != 3
            or any(not _integer(item, 1) for item in value) or value[2] != 1
            or math.prod(value) > 65536):
        raise ValueError("Historical interpolation requires a bounded full transverse mesh")
    return tuple(value)


def _negative(index, shape):
    nx, ny, nz = shape
    return (-index % nx) % nx + nx * ((-(index // nx) % ny) % ny
               + ny * ((-(index // (nx * ny)) % nz) % nz))


def _addresses(shape):
    rows = []
    for index in range(math.prod(shape)):
        row = []
        for axis, size in enumerate(shape):
            value = (index // math.prod(shape[:axis])) % size
            row.append(value - size if 2 * value >= size else value)
        rows.append(row)
    return rows


def periodic_linear_weights(source_mesh, target_mesh):
    """Return convex X-fast full-mesh rows, checking exact inversion routing.

    Fractions construct weights without nearest-node tie decisions. Exact
    source nodes have a single unit-weight entry and require no arithmetic.
    """
    source, target = _mesh(list(source_mesh)), _mesh(list(target_mesh))
    rows = []
    for index in range(math.prod(target)):
        axes = []
        for axis, (ns, nt) in enumerate(zip(source, target)):
            stride = math.prod(target[:axis])
            coordinate = (index // stride) % nt
            low, remainder = divmod(coordinate * ns, nt)
            options = {low % ns: Fraction(nt - remainder, nt)}
            if remainder:
                high = (low + 1) % ns
                options[high] = options.get(high, Fraction()) + Fraction(remainder, nt)
            axes.append(options.items())
        row = {}
        for entries in itertools.product(*axes):
            source_index = entries[0][0] + source[0] * (entries[1][0] + source[1] * entries[2][0])
            weight = math.prod(item[1] for item in entries)
            row[source_index] = row.get(source_index, Fraction()) + weight
        if sum(row.values()) != 1 or any(value <= 0 for value in row.values()):
            raise ValueError("Invalid convex periodic interpolation weights")
        rows.append(row)
    for index, row in enumerate(rows):
        opposite = {_negative(key, source): value for key, value in row.items()}
        if rows[_negative(index, target)] != opposite:
            raise ValueError("Historical interpolation must commute with opposite-q routing")
    return [[(key, float(value)) for key, value in sorted(row.items())] for row in rows]


def _identity(document, evidence):
    from .restart import _SHA
    identity, bindings = document["identity"], document["identity_bindings"]
    if not isinstance(identity, dict) or set(identity) != _IDENTITY_KEYS:
        raise ValueError("Historical source identity is incomplete or contains unknown fields")
    leaves = dict(_leaves(identity))
    # Public model-type names are normalized namespaces, not native source bytes.
    leaves.pop("/model_type")
    if not isinstance(bindings, dict) or set(bindings) != set(leaves):
        raise ValueError("Every historical physical identity leaf needs an original evidence binding")
    for pointer, value in leaves.items():
        binding = bindings[pointer]
        if (not isinstance(binding, dict) or set(binding) != {"evidence", "pointer"}
                or binding["evidence"] not in evidence
                or _pointer(evidence[binding["evidence"]], binding["pointer"]) != value):
            raise ValueError("Historical normalized identity differs from original evidence: " + pointer)
    _mesh(identity["mesh"])
    for key in ("full_dofs", "n_device_layers", "frequency_nodes"):
        if not _integer(identity[key], 1):
            raise ValueError("Historical identity has an invalid positive dimension")
    active = identity["active_device_indices"]
    if (not isinstance(active, list) or not active or len(set(active)) != len(active)
            or any(not _integer(item) or item >= identity["full_dofs"] for item in active)):
        raise ValueError("Historical active endpoint ordering is invalid")
    layers = identity["interaction_layers"]
    if (not isinstance(layers, list) or not layers or len(set(layers)) != len(layers)
            or any(not _integer(item) or item >= identity["n_device_layers"] for item in layers)):
        raise ValueError("Historical selected energy-center layers are invalid")
    for key in ("g", "frequency_step", "omega0_rad_s", "hbar_j_s", "eta_lead", "eta_device"):
        if not _number(identity[key]) or (key in {"frequency_step", "omega0_rad_s", "hbar_j_s", "eta_lead"} and identity[key] == 0):
            raise ValueError("Historical physical scalar is invalid: " + key)
    if (not isinstance(identity["temperatures_kelvin"], list)
            or len(identity["temperatures_kelvin"]) != 2
            or any(not _number(value) for value in identity["temperatures_kelvin"])):
        raise ValueError("Historical reservoir temperatures are invalid")
    for key in ("geometry_sha256", "energy_identity_sha256"):
        if not isinstance(identity[key], str) or not _SHA.fullmatch(identity[key]):
            raise ValueError("Historical geometry/energy identity hash is invalid")
    if (identity["all_outer_derivatives_retained"] is not True
            or not _integer(identity["cropped_local_derivative_blocks"])
            or identity["cropped_local_derivative_blocks"] != 0):
        raise ValueError("Historical source must retain complete uncropped endpoint support")
    return identity


def _target(identity, config, grid, mesh, vertex, scale, info):
    import numpy as np
    doc, support = config.document, info["support"]
    expected = dict(model_type=doc["model"]["type"], g=doc["scba"]["g"],
        full_dofs=vertex.full_ndof, active_device_indices=vertex.active_dofs.tolist(),
        n_device_layers=doc["geometry"]["n_device_layers"],
        interaction_layers=doc["geometry"]["interaction_layers"],
        frequency_nodes=grid.n_points, frequency_step=grid.step,
        omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar,
        eta_lead=doc["reservoirs"]["eta_lead"], eta_device=doc["reservoirs"]["eta_device"],
        geometry_sha256=support["geometry_sha256"],
        energy_identity_sha256=support["energy_identity_sha256"],
        all_outer_derivatives_retained=support["all_outer_derivatives_retained"],
        cropped_local_derivative_blocks=support["cropped_local_derivative_blocks"],
        source_units=info["source_units"])
    roles = ({"fc2", "crystal", "coefficients"} if doc["model"]["type"] == "saved_site_energy"
             else {"ifc2", "archive"} if doc["model"]["type"] == "saved_complete_energy" else set())
    if "initialization_physical_input_roles" in info:
        # This receipt is emitted only after the complete-energy backend has
        # independently compared exported IFC2/cubic payloads to the original
        # archive. It is not an input-JSON role override.
        if (doc["model"]["type"] != "saved_complete_energy"
                or info["initialization_physical_input_roles"] != ["source_archive"]
                or support.get("source_model") != "native-complete-site-derivatives-lossless-export"):
            raise ValueError("Unsupported historical physical-asset equivalence receipt")
        roles = {"source_archive"}
    if not roles or not roles <= config.hashes.keys():
        raise ValueError("Historical initialization needs supported authenticated physical asset roles")
    expected["physical_inputs"] = {role: {key: config.hashes[role][key] for key in ("sha256", "bytes")} for role in roles}
    if any(identity[key] != value for key, value in expected.items()):
        different = [key for key, value in expected.items() if identity[key] != value]
        raise ValueError("Historical source differs from fresh target physical identity: " + ", ".join(different))
    if (grid.n_points != info["frequency_nodes"] or mesh.n_qpoints != info["q_points"]
            or list(mesh.shape) != doc["grid"]["mesh"] or grid.step != doc["grid"]["step"]
            or grid.n_positive != doc["grid"]["n_positive"]
            or not np.array_equal(grid.values, np.arange(-grid.n_positive, grid.n_positive + 1) * grid.step)
            or scale.omega0 != doc["scale"]["omega0_rad_s"] or scale.hbar != doc["scale"]["hbar_j_s"]):
        raise ValueError("Fresh grid/scale differs from the declared target input")
    if (not np.array_equal(mesh.addresses, _addresses(tuple(mesh.shape)))
            or any(mesh.negative_index(index) != _negative(index, tuple(mesh.shape)) for index in range(mesh.n_qpoints))):
        raise ValueError("Fresh target does not use the canonical authenticated X-fast Bloch basis")
    return expected


def load_initial_guess(descriptor, *, config, grid, mesh, vertex, scale, info):
    """Return read-only ChannelSpectrum and an explicit unevaluated-guess receipt."""
    import numpy as np
    from negfpy.anharmonic.transverse_channel_feedback import ChannelSpectrum
    from .input import read_json
    from .restart import _authenticate, _SHA

    if (not isinstance(descriptor, dict) or set(descriptor) != {"manifest", "manifest_sha256", "mode"}
            or descriptor["mode"] != _MODE or not isinstance(descriptor["manifest_sha256"], str)
            or not _SHA.fullmatch(descriptor["manifest_sha256"])):
        raise ValueError("Invalid historical initial-self-energy descriptor")
    manifest_path = Path(descriptor["manifest"]).resolve(strict=True)
    if manifest_path.stat().st_size > 8 * 1024**2:
        raise ValueError("Historical portable manifest exceeds its bounded size")
    manifest_before = _authenticate(manifest_path, descriptor["manifest_sha256"])
    document = read_json(manifest_path)
    if (not isinstance(document, dict) or set(document) != {"schema", "source", "identity", "identity_bindings", "arrays"}
            or document["schema"] != _SCHEMA):
        raise ValueError("Invalid historical portable manifest schema")
    base, source = manifest_path.parent, document["source"]
    if (not isinstance(source, dict) or set(source) != {"format", "evidence"}
            or not isinstance(source["evidence"], dict) or "state" not in source["evidence"]
            or not 1 <= len(source["evidence"]) <= 16):
        raise ValueError("Historical source requires bounded original state evidence")
    if sum(row.get("bytes", 0) for name, row in source["evidence"].items()
           if name != "fields" and isinstance(row, dict) and _integer(row.get("bytes"))) > 32 * 1024**2:
        raise ValueError("Historical original evidence exceeds the total metadata bound")
    evidence, pins = {}, []
    for name, row in source["evidence"].items():
        is_fields = source["format"] == "native_hdf_result" and name == "fields"
        path, before = _pin(base, row, max_bytes=64 * 1024**3 if is_fields else 8 * 1024**2)
        evidence[name] = path if is_fields else read_json(path)
        pins.append((path, row["sha256"], before))
    try:
        state, index, record = _state(source, evidence)
        if source["format"] == "native_hdf_result":
            original = source["evidence"]["fields"]
            if original["sha256"] != state["self_energy"]["sha256"] or original["bytes"] != state["self_energy"]["bytes"]:
                raise ValueError("Historical HDF evidence does not bind the original published file")
            evidence["fields"] = _hdf_evidence(evidence["fields"], state)
        identity = _identity(document, evidence)
        _source_identity(source, state, evidence, identity)
        _fresh_auxiliary(source, state, evidence, identity)
        _target(identity, config, grid, mesh, vertex, scale, info)
        rows = periodic_linear_weights(identity["mesh"], list(mesh.shape))
        source_shape = (math.prod(identity["mesh"]), grid.n_points, vertex.ndof, vertex.ndof)
        target_shape = (mesh.n_qpoints, grid.n_points, vertex.ndof, vertex.ndof)
        payload = 3 * math.prod(target_shape) * 16
        source_field = math.prod(source_shape) * 16
        # One mapped source component plus destination fields and bounded row arithmetic.
        bound = payload + source_field + 32 * 1024**2
        if bound > config.document["scba"]["max_bytes"]:
            raise ValueError("Historical source/target loading exceeds scba.max_bytes")
        if not isinstance(document["arrays"], dict) or set(document["arrays"]) != set(_COMPONENTS):
            raise ValueError("Historical input requires exactly three independent INPUT components")
        output, adopted = [], {}
        for component in _COMPONENTS:
            row = document["arrays"][component]
            if (not isinstance(row, dict) or set(row) != {"file", "sha256", "bytes", "shape", "dtype"}
                    or row["shape"] != list(source_shape) or row["dtype"] != "complex128"):
                raise ValueError("Historical INPUT array shape/dtype declaration differs")
            pin = {key: row[key] for key in ("file", "sha256", "bytes")}
            path, before = _pin(base, pin, max_bytes=source_field + 65536)
            if source["format"] == "native_channel_manifest":
                original = state["artifacts"]["sigma_input_" + component]
                if (original["file_sha256"] != row["sha256"] or original["file_bytes"] != row["bytes"]
                        or original["shape"] != list(source_shape) or original["dtype"] != "complex128"):
                    raise ValueError("Portable array is not the original selected native INPUT artifact")
            values = np.load(path, mmap_mode="r", allow_pickle=False, max_header_size=16384)
            if values.dtype != np.dtype("complex128") or values.shape != source_shape:
                raise ValueError("Historical INPUT array is not native complex128 in the source basis")
            digest = hashlib.sha256()
            for q, w in np.ndindex(source_shape[:2]):
                matrix = values[q, w]
                if not np.isfinite(matrix).all():
                    raise ValueError("Historical INPUT array contains nonfinite values")
                if source["format"] == "native_hdf_result":
                    digest.update(matrix.tobytes(order="C"))
            if source["format"] == "native_hdf_result":
                expected_sha = state["self_energy"]["components"]["selected_evaluated_input/" + component]
                if digest.hexdigest() != expected_sha:
                    raise ValueError("Portable array payload differs from the original evaluated HDF INPUT")
            result = np.empty(target_shape, dtype=np.complex128)
            for q, entries in enumerate(rows):
                if len(entries) == 1 and entries[0][1] == 1:
                    # Exact nodes, including same-q/T-only transfers, preserve every bit.
                    np.copyto(result[q], values[entries[0][0]], casting="no")
                else:
                    for w in range(grid.n_points):
                        first, weight = entries[0]
                        np.multiply(values[first, w], weight, out=result[q, w])
                        for origin, weight in entries[1:]:
                            result[q, w] += values[origin, w] * weight
            if any(not np.isfinite(result[q, w]).all() for q, w in np.ndindex(target_shape[:2])):
                raise ValueError("Historical interpolated INPUT contains nonfinite values")
            result.setflags(write=False)
            output.append(result)
            adopted[component] = dict(sha256=row["sha256"], bytes=row["bytes"])
            del values
            pins.append((path, row["sha256"], before))
        for path, sha, before in pins:
            if before != _authenticate(path, sha):
                raise ValueError("Historical evidence/array changed during admission")
        if manifest_before != _authenticate(manifest_path, descriptor["manifest_sha256"]):
            raise ValueError("Historical portable manifest changed during admission")
    except (KeyError, TypeError, AttributeError, IndexError) as exc:
        raise ValueError("Incomplete historical initial-guess metadata: " + str(exc)) from exc
    target_temperatures = [config.document["reservoirs"][side + "_temperature_k"] for side in ("left", "right")]
    receipt = dict(kind="historical_evaluated_input_initial_guess", source_schema=state["schema"],
        source_manifest_sha256=descriptor["manifest_sha256"], source_index=index,
        source_convergence_admitted=True, source_convergence_inherited=False,
        source_input_evaluated=True, new_target_evaluation_required=True,
        target_input_evaluated=False, controller_history_loaded=False, mapped_image_or_green_loaded=False,
        source_identity=identity, source_evidence=source["evidence"], input_components=adopted,
        source_mesh=identity["mesh"], target_mesh=list(mesh.shape), interpolation="periodic_linear",
        interpolation_rows=[dict(source_q_indices=[item[0] for item in row], weights=[item[1] for item in row]) for row in rows],
        opposite_q_routing_preserved=True, weights_are_not_quadrature=True,
        target_quadrature_rebuilt=True, source_temperatures_kelvin=identity["temperatures_kelvin"],
        target_temperatures_kelvin=target_temperatures, temperature_scaling_applied=False,
        shape=list(target_shape), payload_bytes=payload, loader_numeric_bound_bytes=bound,
        exact_node_copies_preserve_bits=True, copied_components_read_only=True)
    return ChannelSpectrum(*output), receipt
