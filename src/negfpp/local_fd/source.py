"""Read an authenticated, evaluated short-device retarded donor.

The compact format contains only the selected input retarded self-energy,
its coordinates, and the original result/input/exit records. It contains no
Born image and cannot initiate SCBA. Metadata and all bound files are checked
before scientific arrays are read. Hashes establish provenance and integrity;
they do not certify the underlying material model or its convergence grid.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh
from negfpy.anharmonic.transverse_mapping import _kpoints
from negfpy.anharmonic.types import SolverScale

from .kernel_transfer import SOURCE_ACTIVE_DOFS, make_frozen_kernel


_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ROLES = ("fc2", "crystal", "coefficients", "metadata")
_PHYSICAL_GATES = (
    "harmonic", "input_sigma", "input_causality", "left", "right",
    "raw_source", "full_green_local", "active_green", "mapped_sigma",
)
_MAX_METADATA_BYTES = 4 * 1024**2
_MAX_ARRAY_BYTES = 256 * 1024**2


@dataclass(frozen=True, slots=True)
class FrozenSource:
    """Qualified donor, preserving the evaluated input field and its order."""

    sigmaR: np.ndarray
    grid: SignedFrequencyGrid
    mesh: UniformBZMesh
    scale: SolverScale
    active_dofs: np.ndarray
    report: dict[str, Any]
    inputdoc: dict[str, Any]
    files: dict[str, Path]
    qualification: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(name + " must be a lowercase SHA256 digest")
    return value


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _json(path: Path) -> dict:
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise ValueError("Donor metadata exceeds its bounded size: " + path.name)
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object,
                       parse_constant=lambda value: (_ for _ in ()).throw(
                           ValueError("Non-finite JSON value: " + value)))
    if not isinstance(value, dict):
        raise ValueError("Donor metadata must be a JSON object")
    return value


def _bound(base: Path, record: dict, *, limit: int) -> Path:
    if not isinstance(record, dict) or set(record) != {"path", "bytes", "sha256"}:
        raise ValueError("Each donor file requires path, bytes and sha256")
    raw = record["path"]
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute() or "\\" in raw:
        raise ValueError("Donor file paths must be portable relative paths")
    size = record["bytes"]
    if type(size) is not int or not 0 < size <= limit:
        raise ValueError("Donor file has an invalid or excessive declared size")
    expected = _digest(record["sha256"], "File sha256")
    path = (base / raw).resolve(strict=True)
    if not path.is_file() or path.stat().st_size != size or _sha256(path) != expected:
        raise ValueError("Donor file size/hash mismatch: " + raw)
    return path


def _ratios(values: dict, expected: set[str], label: str) -> None:
    if not isinstance(values, dict) or set(values) != expected:
        raise ValueError("Missing original convergence gates: " + label)
    for value in values.values():
        ratio = value.get("ratio") if isinstance(value, dict) else None
        if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                or not math.isfinite(ratio) or not 0 <= ratio <= 1):
            raise ValueError("Original convergence gate failed: " + label)


def _qualify(report: dict, inputdoc: dict, receipt: dict, manifest: dict) -> int:
    """Reject incomplete/unevaluated donors before opening their arrays."""
    if (report.get("schema") != "negfpp.result.v1"
            or report.get("status") != "converged"
            or report.get("converged") is not True
            or report.get("stop_reason") != "fixed_point"
            or report.get("failure_reason") is not None):
        raise ValueError("A converged fixed-point donor is required")
    documents = manifest["documents"]
    origin = manifest["origin"]
    if (report.get("source_input_sha256") != documents["input"]["sha256"]
            or origin.get("report_sha256") != documents["report"]["sha256"]
            or origin.get("input_sha256") != documents["input"]["sha256"]
            or receipt.get("schema") != "negfpp.silicon24-check.v1"
            or receipt.get("passed") is not True
            or type(receipt.get("actual_exit_code")) is not int
            or receipt["actual_exit_code"] != 0
            or receipt.get("report_sha256") != origin["report_sha256"]
            or receipt.get("input_hashes_verified") is not True
            or receipt.get("result_hashes_verified") is not True):
        raise ValueError("The original input, result and actual exit receipt must agree")
    anchor = report["selected_anchor"]
    index = anchor["index"]
    if (type(index) is not int or index < 1 or index != report["last_evaluated"]["index"]
            or anchor.get("input_sigma_evaluated") is not True
            or anchor.get("mapped_image_dyson_evaluated") is not False
            or anchor.get("accepted_as_anchor") is not True
            or index != anchor["record"]["index"]):
        raise ValueError("The selected donor must be a Dyson-evaluated input state")
    record = anchor["record"]
    if any(record.get(key) is not True for key in ("converged", "sigma_passed", "current_passed")):
        raise ValueError("The original six convergence gates must all pass")
    _ratios(record["sigma_residuals"], {"retarded", "lesser", "greater"}, "self-energy")
    _ratios(record["current_residuals"], {"contact_balance", "collision_power", "full_balance"}, "current")
    for group in _PHYSICAL_GATES:
        checks = anchor["physical_checks"][group]
        if not isinstance(checks, dict) or not checks:
            raise ValueError("Missing original physical checks: " + group)
        _ratios(checks, set(checks), group)
    model, geometry = report["model"], inputdoc["geometry"]
    support = model["support"]
    if (model["full_atoms"] != 24 or model["full_dofs"] != 72
            or model["active_atoms"] != 12 or model["active_dofs"] != 36
            or model["energy_centers"] != 8
            or geometry["n_device_layers"] != 3 or geometry["interaction_layers"] != [1]
            or support["center_layers"] != [1] or support["n_device_layers"] != 3
            or support["full_ndof"] != 72 or support["active_dofs"] != 36
            or support["selected_center_count"] != 8
            or support["all_outer_derivatives_retained"] is not True
            or support["crop_derivatives"] is not False
            or support["cropped_local_derivative_blocks"] != 0):
        raise ValueError("The donor must retain the complete original one-center 72/36-DOF support")
    return index


def load_frozen_source(manifest_path: str | Path, *, expected_sha256: str | None = None) -> FrozenSource:
    """Load a portable donor with all bindings checked before array access.

    The caller can pin the manifest itself using ``expected_sha256``. Its
    bound source documents and four unchanged material inputs are verified
    regardless. Only the evaluated input Sigma-R dataset is accepted; the
    unequal, unevaluated Born-map image is never a permissible donor.
    """
    path = Path(manifest_path).resolve(strict=True)
    actual_manifest_sha = _sha256(path)
    if expected_sha256 is not None and actual_manifest_sha != _digest(expected_sha256, "Manifest sha256"):
        raise ValueError("Frozen donor manifest hash mismatch")
    manifest = _json(path)
    if manifest.get("schema") != "negfpp.frozen-source.v1":
        raise ValueError("Unsupported frozen donor manifest schema")
    try:
        docs = manifest["documents"]
        if set(docs) != {"report", "input", "qualification", "original_checksums"}:
            raise ValueError("Incomplete original donor documents")
        documents = {role: _bound(path.parent, value, limit=_MAX_METADATA_BYTES)
                     for role, value in docs.items()}
        if set(manifest["inputs"]) != set(_ROLES):
            raise ValueError("The four original donor input roles are required")
        files = {role: _bound(path.parent, manifest["inputs"][role], limit=_MAX_ARRAY_BYTES)
                 for role in _ROLES}
        fields = _bound(path.parent, manifest["retarded"]["file"], limit=_MAX_ARRAY_BYTES)
        report, inputdoc, receipt = (_json(documents[key]) for key in ("report", "input", "qualification"))
        index = _qualify(report, inputdoc, receipt, manifest)
        original_checksums = {}
        for line in documents["original_checksums"].read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            name = name.lstrip("*")
            if name in original_checksums:
                raise ValueError("Duplicate original checksum filename")
            original_checksums[name] = _digest(digest, "Original checksum")
        origin = manifest["origin"]
        for name, expected in (("report.json", docs["report"]["sha256"]),
                               ("input.json", docs["input"]["sha256"]),
                               ("fields.h5", origin["fields_sha256"])):
            if original_checksums.get(name) != expected:
                raise ValueError("Original result checksum binding differs: " + name)
        for role in _ROLES:
            record = manifest["inputs"][role]
            expected = report["inputs"][role]
            if (record["sha256"] != expected["sha256"] or record["bytes"] != expected["bytes"]
                    or inputdoc["inputs_sha256"].get(expected["file"]) != expected["sha256"]):
                raise ValueError("Original material input binding differs: " + role)
        grid = SignedFrequencyGrid(inputdoc["grid"]["step"], inputdoc["grid"]["n_positive"])
        mesh = UniformBZMesh(tuple(inputdoc["grid"]["mesh"]))
        scale = SolverScale(omega0=inputdoc["scale"]["omega0_rad_s"],
                            hbar=inputdoc["scale"]["hbar_j_s"], metadata={"source_units": "SI"})
        shape = (mesh.n_qpoints, grid.n_points, 36, 36)
        retarded = manifest["retarded"]
        payload_sha = _digest(retarded["payload_sha256"], "Retarded payload sha256")
        if (retarded["role"] != "sigma_input_evaluated/retarded"
                or retarded["evaluated_state_index"] != index
                or retarded["shape"] != list(shape)
                or retarded["dtype"] != "complex128"
                or retarded["payload_order"] != "C-little-endian"
                or math.prod(shape) * 16 > _MAX_ARRAY_BYTES
                or report["model"]["q_points"] != mesh.n_qpoints
                or report["model"]["frequency_nodes"] != grid.n_points
                or receipt["field_shape"] != list(shape)
                or receipt["input_sigma_nonzero"] is not True
                or receipt["mapped_image_unevaluated"] is not True):
            raise ValueError("Invalid evaluated retarded payload role, grid or size")
        # All document qualification and byte checks above precede HDF5 access.
        with h5py.File(fields, "r") as handle:
            names = {"sigma_retarded", "active_device_indices", "omega_solver", "omega_rad_s", "q_phase_rad", "q_reduced"}
            if set(handle) != names or any(not isinstance(handle.get(n, getlink=True), h5py.HardLink) for n in names):
                raise ValueError("Compact donor must contain only its own evaluated field and coordinates")
            if (handle.attrs.get("schema") != "negfpp.frozen-retarded.v1"
                    or handle.attrs.get("source_role") != retarded["role"]
                    or handle.attrs.get("evaluated_state_index") != index
                    or bool(handle.attrs.get("mapped_image_dyson_evaluated", True))
                    or handle.attrs.get("origin_fields_sha256") != origin["fields_sha256"]
                    or handle.attrs.get("omega0_rad_s") != scale.omega0
                    or handle.attrs.get("hbar_j_s") != scale.hbar):
                raise ValueError("Compact donor field attributes differ from the qualified source")
            expected_shapes = {"sigma_retarded": shape, "active_device_indices": (36,),
                               "omega_solver": (grid.n_points,), "omega_rad_s": (grid.n_points,),
                               "q_phase_rad": (mesh.n_qpoints, 2), "q_reduced": (mesh.n_qpoints, 2)}
            for name, expected_shape in expected_shapes.items():
                dataset = handle[name]
                if not isinstance(dataset, h5py.Dataset) or dataset.shape != expected_shape or dataset.is_virtual:
                    raise ValueError("Invalid compact donor dataset: " + name)
                if dataset.external:
                    raise ValueError("External HDF5 storage is not a portable compact donor")
            if handle["sigma_retarded"].dtype != np.dtype("<c16"):
                raise ValueError("The donor field must retain complex128 precision")
            sigma = np.asarray(handle["sigma_retarded"][:], dtype="<c16", order="C")
            active = handle["active_device_indices"][:]
            omega, physical = handle["omega_solver"][:], handle["omega_rad_s"][:]
            phases, reduced = handle["q_phase_rad"][:], handle["q_reduced"][:]
        if hashlib.sha256(sigma.tobytes(order="C")).hexdigest() != payload_sha:
            raise ValueError("Selected evaluated retarded payload checksum differs")
        if (not np.array_equal(active, np.asarray(SOURCE_ACTIVE_DOFS))
                or not np.array_equal(omega, grid.values)
                or not np.array_equal(physical, grid.values * scale.omega0)
                or not np.array_equal(phases, _kpoints(mesh))
                or not np.array_equal(reduced, _kpoints(mesh) / (2 * np.pi))):
            raise ValueError("Compact donor coordinates or endpoint order differ from the original input")
        negative_q = np.asarray([mesh.negative_index(q) for q in range(mesh.n_qpoints)], dtype=np.int64)
        kernel = make_frozen_kernel(sigma, grid.values, active,
                                    negative_q_indices=negative_q,
                                    gamma_q_index=mesh.index_from_address((0, 0, 0)),
                                    source_translation_vectors=np.tile(np.eye(3), (24, 1)))
        qualification = dict(passed=True, evaluated_input=True, all_original_six_passed=True,
                             actual_exit_code=0, manifest_sha256=actual_manifest_sha,
                             original_report_sha256=origin["report_sha256"],
                             original_fields_sha256=origin["fields_sha256"],
                             compact_fields_sha256=retarded["file"]["sha256"],
                             selected_payload_sha256=payload_sha, source_role=retarded["role"],
                             evaluated_state_index=index, validation=kernel.validation)
        return FrozenSource(kernel.retarded, grid, mesh, scale, kernel.active_dofs,
                            report, inputdoc, files, qualification)
    except (KeyError, TypeError, IndexError) as error:
        raise ValueError("Incomplete or invalid frozen donor metadata") from error
