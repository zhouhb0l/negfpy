"""Authenticated material-map reuse and immutable evaluated SCBA checkpoints.

Authentication is byte based and precedes array mapping. Saved fields are input
data, not an accepted restart or evidence of self-consistent convergence. No raw
IFCs or completed numerical workflows are read or repeated here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np

from ._material_nq1_campaign import kelvin_to_solver_temperature
from .grid import SignedFrequencyGrid
from .momentum import UniformBZMesh
from .transverse_compact_contacts import CompactTransverseContact
from .transverse_contacts import TransverseContact
from .transverse_material_map import PackedMapSpectrum
from .transverse_packed import PackedTransverseCubicVertex, _finite, _integer
from .transverse_solver import TransverseSpectrum

_COMPONENTS = ("retarded", "lesser", "greater")
_CHECKPOINT_SCHEMA = "negfpy.transverse-material-scba-checkpoint.v1"
_DESCRIPTOR_KEYS = {"relative_path", "shape", "dtype", "array_bytes", "file_bytes", "file_sha256"}


@dataclass(frozen=True, slots=True)
class MaterialMapFields:
    grid: SignedFrequencyGrid
    mesh: UniformBZMesh
    dynamical: np.ndarray
    vertex: PackedTransverseCubicVertex
    left: CompactTransverseContact
    right: CompactTransverseContact
    input_green: PackedMapSpectrum
    mapped_sigma: PackedMapSpectrum
    result: dict
    profile: dict
    binding: dict


@dataclass(frozen=True, slots=True)
class MaterialCheckpoint:
    sigma_input: PackedMapSpectrum
    mapped_sigma: PackedMapSpectrum
    record: dict
    binding: dict
    manifest: dict
    manifest_sha256: str


def _digest(value, *, size=64):
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{size}}}", value) is None:
        raise ValueError(f"Expected a lowercase {size}-character digest.")
    return value


def _plain(value):
    """Detach finite JSON metadata; reject arrays, NaNs and custom objects."""
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _safe_path(root, relative):
    relative = PurePosixPath(relative)
    if relative.is_absolute() or any(part in ("..", ".") for part in relative.parts):
        raise ValueError("Artifact paths must remain below their root.")
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("Artifact root must not be a symlink.")
    path = root
    for part in relative.parts:
        path = path / part
        if path.is_symlink():
            raise ValueError("Artifact paths must not contain symlinks.")
    return path


def _identity(stream):
    value = os.fstat(stream.fileno())
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def _open_verified(stack, path, digest, size=None):
    _digest(digest)
    stream = stack.enter_context(os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb"))
    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
        raise ValueError("Artifact must be a regular file.")
    before = _identity(stream)
    if size is not None and before[2] != size:
        raise ValueError(f"Artifact file size mismatch: {path.name}.")
    if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
        raise ValueError(f"Artifact SHA256 mismatch: {path.name}.")
    if _identity(stream) != before:
        raise ValueError("Artifact changed during authentication.")
    stream.seek(0)
    return stream, before


def _json_verified(stack, root, relative, digest):
    stream, _ = _open_verified(stack, _safe_path(root, relative), digest)
    if os.fstat(stream.fileno()).st_size > 16 * 1024**2:
        raise ValueError("Metadata exceeds the 16 MiB bound.")
    return json.load(stream)


def _expected_artifacts(nq, nw, d, active, left, right):
    shapes = {"dynamical": ((nq, d, d), "complex128"),
              "vertex_before_g": ((nq, nq, active, active, active), "complex128"),
              "active_dofs": ((active,), "int64")}
    for side, width in (("left", left), ("right", right)):
        shapes[side + "_device_indices"] = ((width,), "int64")
        shapes[side + "_zero_limits"] = ((nq, width, width), "complex128")
        for component in _COMPONENTS:
            shapes[side + "_" + component] = ((nq, nw, width, width), "complex128")
    for label in ("input_green_active", "mapped_sigma"):
        for component in _COMPONENTS:
            shapes[label + "_" + component] = ((nq, nw, active, active), "complex128")
    return shapes


def _arrays_verified(stack, root, descriptors, expected, max_total_bytes, authentication=None):
    """Hash every file first, then map those same open file descriptors."""
    limit = _integer(max_total_bytes, "max_total_bytes")
    if set(descriptors) != set(expected):
        raise ValueError("Artifact names differ from the required exact set.")
    opened = {}
    total = 0
    for name, (shape, dtype) in expected.items():
        row = descriptors[name]
        numeric_bytes = int(np.prod(shape, dtype=object)) * np.dtype(dtype).itemsize
        if (set(row) != _DESCRIPTOR_KEYS or row["relative_path"] != f"arrays/{name}.npy"
                or row["shape"] != list(shape) or row["dtype"] != dtype
                or row["array_bytes"] != numeric_bytes):
            raise ValueError(f"Artifact descriptor mismatch: {name}.")
        file_bytes = _integer(row["file_bytes"], "file_bytes")
        if file_bytes < numeric_bytes or file_bytes > numeric_bytes + 65536:
            raise ValueError("Array file has an invalid header/data size.")
        total += file_bytes
        if total > limit:
            raise ValueError("Saved fields exceed max_total_bytes.")
    for name, row in descriptors.items():
        opened[name] = _open_verified(
            stack, _safe_path(root, row["relative_path"]), row["file_sha256"], row["file_bytes"],
        )
    arrays = {}
    for name, (stream, identity) in opened.items():
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Only numeric NPY version 1 or 2 is supported.")
        expected_shape, expected_dtype = expected[name]
        if shape != expected_shape or dtype != np.dtype(expected_dtype) or fortran:
            raise ValueError(f"NPY header differs from the authenticated descriptor: {name}.")
        if stream.tell() + descriptors[name]["array_bytes"] != identity[2]:
            raise ValueError("NPY contains truncated or trailing data.")
        values = np.memmap(stream, mode="r", dtype=dtype, offset=stream.tell(), shape=shape)
        if _identity(stream) != identity:
            raise ValueError("Artifact changed between authentication and mapping.")
        arrays[name] = values
        if authentication is not None:
            authentication[name] = {
                **descriptors[name], "path": str(_safe_path(root, descriptors[name]["relative_path"])),
                "stat_before": dict(zip(("device", "inode", "bytes", "mtime_ns", "ctime_ns"), identity, strict=True)),
                "stat_after": dict(zip(("device", "inode", "bytes", "mtime_ns", "ctime_ns"), _identity(stream), strict=True)),
            }
    return arrays


def _spectrum(arrays, label):
    return PackedMapSpectrum(*(arrays[label + "_" + name] for name in _COMPONENTS))


def load_material_map_fields(
    parent_root, *, expected_result_sha256, expected_profile_sha256,
    max_total_bytes=8 * 1024**3, expected_source_commit=None,
    expected_source_tree=None, expected_settings=None,
):
    """Reuse the pinned completed map's 19 fields, freshly authenticated.

    File descriptors stay bound through hashing and read-only mmap creation.
    The source directory must remain immutable while mapped arrays are in use.
    Contact and vertex constructors retain their existing validation and caps.
    """
    with ExitStack() as stack:
        profile = _json_verified(stack, parent_root, "profile.json", expected_profile_sha256)
        result = _json_verified(
            stack, parent_root, "runtime/material-map-result.json", expected_result_sha256,
        )
        if (result.get("schema") != "negfpy.transverse-material-born-map.v1"
                or profile.get("schema") != "negfpy.transverse-material-born-map.profile.v1"
                or result.get("passed") is not True):
            raise ValueError("Parent is not a passed material-map result.")
        cfg = profile["map"]
        if result.get("settings") != cfg or result.get("source_inputs") != profile["runtime_inputs"]:
            raise ValueError("Parent result/profile settings or inputs differ.")
        for key, pinned in (("source_commit", expected_source_commit),
                            ("source_tree", expected_source_tree)):
            _digest(profile[key], size=40)
            if result.get(key) != profile[key] or (pinned is not None and profile[key] != pinned):
                raise ValueError("Parent source binding differs.")
        if expected_settings is not None and cfg != expected_settings:
            raise ValueError("Parent settings differ from the caller's pinned settings.")
        scope = result["scope"]
        for key, value in {
            "actual_material_one_born_map": True, "input_sigma": "exact-zero",
            "input_state_currents": True, "mapped_sigma_dyson_evaluated": False,
            "scba": False, "converged": False, "accepted_field_update": False,
            "vertex_solver_units_before_g": True,
            "coupling_applied_to_image_once_as_g_squared": True,
        }.items():
            if scope.get(key) != value:
                raise ValueError(f"Parent scope differs: {key}.")
        if result["artifact_state_semantics"].get("accepted_restart") is not False:
            raise ValueError("One-map fields cannot be claimed as accepted restart state.")
        mesh = UniformBZMesh(tuple(cfg["mesh"]))
        grid = SignedFrequencyGrid(cfg["grid_step"], cfg["n_positive"])
        d = _integer(result["geometry"]["ndof"], "ndof")
        active = len(result["cubic"]["active_dofs"])
        if active < 1 or result["cubic"].get("full_ndof") != d:
            raise ValueError("Invalid full/active material dimensions.")
        if (result["geometry"]["geometry_sha256"] != cfg["expected_geometry_sha256"]
                or result["cubic"].get("coupling_applied") is not False
                or result["cubic"].get("solver_units_before_g") is not True
                or result["cubic"].get("all_joint_vertex_checks_passed") is not True
                or scope.get("contact_frequency_count") != grid.n_points):
            raise ValueError("Parent geometry, vertex or frequency binding differs.")
        widths = [len(result["contacts"][side]["device_indices"]) for side in ("left", "right")]
        expected = _expected_artifacts(mesh.n_qpoints, grid.n_points, d, active, *widths)
        authentication = {}
        arrays = _arrays_verified(stack, Path(parent_root) / "runtime", result["artifacts"],
                                  expected, max_total_bytes, authentication)
        if arrays["active_dofs"].tolist() != result["cubic"]["active_dofs"]:
            raise ValueError("Packed addresses differ from the parent result.")
        vertex = PackedTransverseCubicVertex(
            mesh, arrays["vertex_before_g"], arrays["active_dofs"], d,
            max_bytes=cfg["vertex_max_bytes"],
        )
        contacts = []
        for index, side in enumerate(("left", "right")):
            report = result["contacts"][side]
            indices = arrays[side + "_device_indices"].tolist()
            if indices != report["device_indices"] or report["temperature_kelvin"] != cfg["temperatures_kelvin"][index]:
                raise ValueError("Contact embedding or temperature differs.")
            if len(report["q_reports"]) != mesh.n_qpoints or any(
                row.get("classification") not in ("acoustic", "gapped") for row in report["q_reports"]
            ):
                raise ValueError("Contact classifications were not accepted.")
            spectrum = TransverseSpectrum(*(arrays[side + "_" + name] for name in _COMPONENTS))
            contact = TransverseContact(
                mesh, grid, spectrum, arrays[side + "_zero_limits"],
                tuple(report["q_reports"]), report["checks"], side, cfg["eta_lead"],
                kelvin_to_solver_temperature(report["temperature_kelvin"]),
            )
            contacts.append(CompactTransverseContact(contact, d, tuple(indices)))
        binding = {"parent_result_sha256": expected_result_sha256,
                   "parent_profile_sha256": expected_profile_sha256,
                   "parent_source_commit": profile["source_commit"],
                   "parent_source_tree": profile["source_tree"],
                   "artifacts_freshly_authenticated": 19, "accepted_restart": False,
                   "artifacts": authentication}
        return MaterialMapFields(grid, mesh, arrays["dynamical"], vertex, *contacts,
                                 _spectrum(arrays, "input_green_active"),
                                 _spectrum(arrays, "mapped_sigma"), result, profile, binding)


def _binding(binding):
    binding = _plain(binding)
    for name in ("parent_result_sha256", "parent_profile_sha256", "profile_sha256"):
        _digest(binding[name])
    for name in ("source_commit", "source_tree"):
        _digest(binding[name], size=40)
    if not isinstance(binding["physics"], dict) or not binding["physics"]:
        raise ValueError("Checkpoint requires explicit physics settings.")
    return binding


def _record(record, iteration):
    record = _plain(record)
    if record.get("index") != iteration or type(record.get("converged")) is not bool:
        raise ValueError("Checkpoint index/convergence record is invalid.")
    if (record.get("sigma_input_evaluated") is not True
            or record.get("mapped_sigma_dyson_evaluated") is not False
            or not isinstance(record.get("currents"), dict)):
        raise ValueError("Checkpoint must identify currents with its evaluated input.")
    if record.get("status") != ("converged" if record["converged"] else "evaluated"):
        raise ValueError("Checkpoint status differs from evaluated convergence.")
    for key, required, flag in (
        ("sigma_residuals", set(_COMPONENTS), "sigma_passed"),
        ("current_residuals", {"contact_balance", "collision_power", "full_balance"}, "current_passed"),
    ):
        residuals = record.get(key)
        if not isinstance(residuals, dict) or set(residuals) != required:
            raise ValueError("Checkpoint requires all three sigma and all three current residuals.")
        ratios = []
        for row in residuals.values():
            ratio = row.get("ratio") if isinstance(row, dict) else None
            if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or ratio < 0:
                raise ValueError("Residual ratios must be finite nonnegative real numbers.")
            ratios.append(ratio)
        if type(record.get(flag)) is not bool or record[flag] != all(value <= 1 for value in ratios):
            raise ValueError("Checkpoint pass flags differ from its complete residual diagnostics.")
    if record["converged"] != (record["sigma_passed"] and record["current_passed"]):
        raise ValueError("Checkpoint convergence differs from evaluated residual convergence.")
    return record


def save_material_checkpoint(
    root, *, iteration, sigma_input, mapped_sigma, record, binding,
    max_bytes=8 * 1024**3, max_checkpoints=12,
):
    """Save six fields in an immutable iteration directory; publish manifest last.

    Incomplete writes remain visible but have no manifest and cannot be resumed.
    No prior state is overwritten or deleted. The manifest authenticates the
    exact evaluated input, its unevaluated Born image, currents and residuals.
    """
    iteration = _integer(iteration, "iteration")
    limit = _integer(max_bytes, "max_bytes")
    count = _integer(max_checkpoints, "max_checkpoints")
    record, binding = _record(record, iteration), _binding(binding)
    arrays = {}
    shape = None
    for label, spectrum in (("sigma_input", sigma_input), ("mapped_sigma", mapped_sigma)):
        if not isinstance(spectrum, PackedMapSpectrum):
            raise TypeError("Checkpoint fields must be PackedMapSpectrum.")
        for component in _COMPONENTS:
            values = getattr(spectrum, component)
            if (not isinstance(values, np.ndarray) or values.dtype != np.dtype("complex128")
                    or values.ndim != 4 or min(values.shape) < 1 or values.shape[-1] != values.shape[-2]):
                raise ValueError("Checkpoint fields must be complex128 (Nq,Nw,A,A) arrays.")
            shape = values.shape if shape is None else shape
            if values.shape != shape:
                raise ValueError("Checkpoint field shapes differ.")
            _finite(values, label + "." + component)
            arrays[label + "_" + component] = values
    if sum(value.nbytes + 65536 for value in arrays.values()) > limit:
        raise ValueError("Checkpoint fields exceed max_bytes.")
    root = Path(root)
    if root.is_symlink():
        raise ValueError("Checkpoint root must not be a symlink.")
    root.mkdir(parents=True, exist_ok=True)
    if sum(path.is_dir() for path in root.glob("iteration-*")) >= count:
        raise ValueError("Checkpoint count exceeds max_checkpoints.")
    directory = root / f"iteration-{iteration:06d}"
    directory.mkdir(exist_ok=False)
    (directory / "arrays").mkdir()
    descriptors = {}
    for name, values in arrays.items():
        path = directory / "arrays" / (name + ".npy")
        with path.open("xb") as stream:
            np.save(stream, values, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        descriptors[name] = {"relative_path": "arrays/" + path.name,
                             "shape": list(values.shape), "dtype": str(values.dtype),
                             "array_bytes": values.nbytes, "file_bytes": path.stat().st_size,
                             "file_sha256": digest}
    manifest = {"schema": _CHECKPOINT_SCHEMA, "iteration": iteration,
                "binding": binding, "record": record, "artifacts": descriptors,
                "accepted_restart": False, "sigma_input_evaluated": True,
                "mapped_sigma_dyson_evaluated": False}
    payload = (json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    temporary = directory / "manifest.json.partial"
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, directory / "manifest.json")
    temporary.unlink()
    for target in (directory / "arrays", directory, root):
        descriptor = os.open(target, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {"path": str(directory), "manifest_sha256": hashlib.sha256(payload).hexdigest(),
            "manifest": manifest}


def load_material_checkpoint(
    path, *, expected_manifest_sha256, expected_binding, max_total_bytes=8 * 1024**3,
):
    """Authenticate a complete checkpoint without treating its image as evaluated."""
    with ExitStack() as stack:
        manifest = _json_verified(stack, path, "manifest.json", expected_manifest_sha256)
        if (manifest.get("schema") != _CHECKPOINT_SCHEMA
                or manifest.get("accepted_restart") is not False
                or manifest.get("sigma_input_evaluated") is not True
                or manifest.get("mapped_sigma_dyson_evaluated") is not False):
            raise ValueError("Invalid evaluated-state checkpoint manifest.")
        if _binding(manifest["binding"]) != _binding(expected_binding):
            raise ValueError("Checkpoint source, physics or parent binding differs.")
        iteration = _integer(manifest["iteration"], "iteration")
        record = _record(manifest["record"], iteration)
        shape = tuple(manifest["artifacts"]["sigma_input_retarded"]["shape"])
        if len(shape) != 4 or any(type(v) is not int or v < 1 for v in shape) or shape[-1] != shape[-2]:
            raise ValueError("Invalid checkpoint field shape.")
        expected = {label + "_" + name: (shape, "complex128")
                    for label in ("sigma_input", "mapped_sigma") for name in _COMPONENTS}
        arrays = _arrays_verified(stack, path, manifest["artifacts"], expected, max_total_bytes)
        for name, values in arrays.items():
            _finite(values, name)
        return MaterialCheckpoint(_spectrum(arrays, "sigma_input"), _spectrum(arrays, "mapped_sigma"),
                                  record, manifest["binding"], manifest, expected_manifest_sha256)
