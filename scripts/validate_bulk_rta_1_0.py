#!/usr/bin/env python3
"""Fail-closed NEGFPy 1.0 bulk-SMRTA release validation.

The release path consumes the exact sealed 0.2.0 silicon target-q archives.
It never regenerates the 11x11x11 interaction tensor.  Before creating an
output directory it authenticates the sealed release manifest, every source
file, every merged archive (including its whole-file digest), its exact HDF5
schema, selectors, metadata, and the selected scientific datasets used here.

The 3x3x3 and 5x5x5 phono3py end-to-end checks remain mandatory in release
mode.  They are separately checkpointed, so an infrastructure interruption
reruns only the unfinished stage.  ``--development`` permits different
settings for bounded investigation but can never produce a certified report.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urldefrag, urlparse

import h5py
import numpy as np

SCHEMA = "negfpy.bulk_rta_release_validation.v2"
CAMPAIGN_SCHEMA = "negfpy.bulk_rta_release_campaign.v1"
CHECKPOINT_SCHEMA = "negfpy.bulk_rta_release_checkpoint.v1"
HARMONIC_SCHEMA = "negfpy.bulk_rta_harmonic_observables.v2"
CORE_REFERENCE_SCHEMA = "negfpy.bulk_rta_core_reference.v1"
END_TO_END_PAYLOAD_SCHEMA = "negfpy.bulk_rta_end_to_end_reference.v1"
FINAL_EVIDENCE_SCHEMA = "negfpy.bulk_rta_release_evidence.v1"
FINAL_HDF5_SCHEMA = "negfpy.bulk_rta_release_results.hdf5.v1"
FINAL_RECEIPT_SCHEMA = "negfpy.bulk_rta_release_completion_receipt.v1"
PREFLIGHT_RECEIPT_SCHEMA = "negfpy.bulk_rta_release_preflight.v1"
STAGED_INPUT_SCHEMA = "negfpy.bulk_rta_staged_inputs.v1"
STAGED_RATE_SCHEMA = "negfpy.bulk_rta_staged_rate_fields.v1"

THREAD_LIMIT_ENVIRONMENT = (
    "BLIS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)

RELEASE_MESHES = (3, 5, 11)
RELEASE_END_TO_END_MESHES = (3, 5)
RELEASE_TEMPERATURES_K = (300.0,)
RELEASE_SIGMA_THZ = 0.1
CUTOFF_THZ = 1.0e-4
FREQUENCY_FACTOR_TO_THZ = 15.633302300230191
PINNED_NEGFPY_VERSION = "1.0.0"
PINNED_PHONO3PY_VERSION = "4.4.0"
PINNED_ENVIRONMENT_LOCK_SHA256 = (
    "a1fb0e7f7ae092106ec4b62911b86cf372da8264ab28aba6c8880aa0489f5eb1"
)
PINNED_PYTHON_VERSION = "3.12.13"
PINNED_OPENMPI_VERSION = "5.0.10"
PINNED_RUNTIME_PACKAGES = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "h5py": "3.16.0",
    "spglib": "2.7.0",
    "phonopy": "4.4.0",
    "phono3py": "4.4.0",
    "mpi4py": "4.1.2",
    "threadpoolctl": "3.6.0",
}

GV_ATOL_THZ_ANGSTROM = 1.0e-10
FREQUENCY_ATOL_THZ = 1.0e-10
CV_ATOL_EV_K = 1.0e-12
FIELD_RTOL = 1.0e-10
FIELD_ATOL = 1.0e-12
KAPPA_OFFDIAGONAL_ATOL_W_MK = 1.0e-8
DOS_NORMALIZATION_ATOL = 5.0e-4
MINIMUM_OUTPUT_FREE_BYTES = 256 * 1024 * 1024

PINNED_V0_2_MANIFEST_SHA256 = (
    "faed3294ba0dfec93638b982f623bcc6c5a603283178e316e40847671f970aba"
)
PINNED_V0_2_COMMIT = "7f23c85711eab4f425c1dcfa3d4538859bba0209"
PINNED_V0_2_TREE = "8a7119629b2e40a1893ac2e56114bc8c77188150"
PINNED_V0_2_ATTEMPT = "attempt-005"
PINNED_V0_2_RUN = "si-oracle-005"

PINNED_SOURCE_SHA256 = {
    "crystal": "1375e3ae19a2c4138a888af8177a3becfcaa3b127ca50755773c405c3e28a2a7",
    "fc2": "1a67145ab28e792a3aa8a8c432c9fbebc15d0b781f60da24ede582a1581b9cd0",
    "fc3": "c7d0902205d9ae48e657de53400c5c6acd0cdd0822804ac8c8dc154b29c15ffd",
}

PINNED_ARCHIVES: dict[int, dict[str, Any]] = {
    3: {
        "file_sha256": "c37d5b1fa13271b2be1b9d359a81b0f73dd1fcf31916975e8cb36fe4aa74ba4c",
        "problem_sha256": "557b0b50c2b6a522479ac2a335db32d05b0bbf30b4f137fbdc17a8f323ce6806",
        "controls_sha256": "99175ae331a7c3e6a060105672d60b32a779e2d75e2afca5397376d41be8e846",
        "campaign_manifest_sha256": "f609b494144f4bd52892b1cd3a9f752ab0c12cec7e248b5406cea232efe22935",
        "manifest_run_identity_sha256": "123e07b7df2cbb6e715641d0a07f711c420debef4c57a00643eb2dba8b0d0920",
        "dataset_sha256": {
            "linewidths/temperatures": "3649ddad1e709a65a2ed7a78ed2a43b672dc0d5f3ed16655ac8bd6aec561c58c",
            "linewidths/grid_points": "72c13eb892b0c4f33eb55293fc560cb36a9e412a6c29ba06e75958b45f793438",
            "linewidths/band_indices": "70b6d6ef8ad2571ff26ade9edb1f5a576f77e76f6a5181bb1c106b4ad6d1ecd2",
            "linewidths/gamma_thz_hwhm": "6cd6d2d76a587e8a5f2ae5c92b45218bebaaec03072fc7c53e135a46bfb46de4",
            "linewidths/gamma_normal_thz_hwhm": "032dece28bb1e67bd5d1a1cf3c8c7debcc74bcdd3cf9d92e6bf24888d1ba5bf6",
            "linewidths/gamma_umklapp_thz_hwhm": "8497568a3adeccc0c23bb889289da38534fc788dcbf03a2bf9da3c660aa9d0b7",
            "linewidths/gamma_angular_fwhm": "399989e118e26b2dd9973358b65d1be3933956e7dcc8833aaac55d9eea334d5b",
            "linewidths/decay_phase_space": "e7b2e0a466fbc12e53595d79b451fe67529869559414ee03826a6727758f805c",
            "linewidths/difference_phase_space": "afd8a3af0c2ca6c1c3d5a02183edf698b3c9c445343544aa810f739e04e7c070",
            "phonons/qpoints": "320f13550149be900f50de349f7d45f15552d998b8200d6250ed05a124f2be42",
            "phonons/frequencies": "4a3be7242ec1768c3ccedf0c6837c8f341a5f4f18f348ff94a3e965f94dbab2d",
            "phonons/dynamical_matrices": "1ec936dde58d413737264043c8b8939a9f46cb275d47f511bee493af54dbc62b",
            "phonons/eigenvectors": "f74b5c606be629403b9f29e3b53ae53e15c5c7ccfd336171da174a7792ae5bac",
            "phonons/frequency_factor_to_thz": "18bc8a36961dee7534ece622b985cdf8826152fe5833806310db1f003476eca3",
            "phonons/hermitianized": "98f4a2e19cfe8b66aa8ea97fd24253b1f525b059bb82731565acc934e5a10816",
            "metadata/linewidths_json": "4841056ea197c7411c47572f2a553532a5e2d9e1ec44d4b307c4c9ff1628bbdd",
            "metadata/interactions_json": "a864fcd53a630d8939dbbce0d2e44a4ee85934bc0543a7a8809fa5f5e3fdf5d5",
        },
    },
    5: {
        "file_sha256": "3e68e02c9c0dd484df3bc9caafd27715875dd3c5dade282765bad01aad422f29",
        "problem_sha256": "342c84de2353aa350c47ddd8bbb1880dd0f2db7ed82278ce3f16292d9ebc93ab",
        "controls_sha256": "d16fd9fcbc4bb90dacc7cb1c61c91ab1689a339a4e9825b7309cdcf93cd974d5",
        "campaign_manifest_sha256": "3ff02dcd080a3deca8af78e1a235e999547cda5107ff7c82759447221ff8ef20",
        "manifest_run_identity_sha256": "3fe53b30e9028b656b31905645ee748ff792149bc0d01bccdbdae6e9a688d157",
        "dataset_sha256": {
            "linewidths/temperatures": "3649ddad1e709a65a2ed7a78ed2a43b672dc0d5f3ed16655ac8bd6aec561c58c",
            "linewidths/grid_points": "cf8845d552ad98529cf1daf62b5ae7fa4198ac6a35d42af7ff5f5331a4b8fd23",
            "linewidths/band_indices": "70b6d6ef8ad2571ff26ade9edb1f5a576f77e76f6a5181bb1c106b4ad6d1ecd2",
            "linewidths/gamma_thz_hwhm": "e1dbe5dfbc710dbf984395816148486344cb990db7277697f661a688f7f06b16",
            "linewidths/gamma_normal_thz_hwhm": "996a0d09f5f7da0d0b4f7a6aaeaa52c8e9f42f0b2831ffa784c2cb497401ae84",
            "linewidths/gamma_umklapp_thz_hwhm": "b8ad3fb4a9e988051185a1a91b90ff6dc7e4a777f545b7c44f01a87b6795fe5f",
            "linewidths/gamma_angular_fwhm": "f7df239b1b9608d8bef89aa4eb2949a6f4f1c3d8d951de0583a955718b11b044",
            "linewidths/decay_phase_space": "00c46cb56715b3fffb09a7bd7ce0604c78d7f3f991aa46c0634a0059c0a7cbd2",
            "linewidths/difference_phase_space": "5e0353fbd06a5a3529e1ec9472a9683d4256ec5f88c7e925396bc9bf3fb37187",
            "phonons/qpoints": "7de317aec7fdd24af73b911c6cd5b70096250db5097e125d536f6a54070aa735",
            "phonons/frequencies": "b7b8dfc2713e193938b6c4ad78d594773e346fcd0df9086df747979dee01f76e",
            "phonons/dynamical_matrices": "6bb7ff1ae91d90eca5acdeaa0d4d23f38a7800e0661b98ba27f74c36da886fe0",
            "phonons/eigenvectors": "a1c043210cbdf76d031098990df10c25273a11dab328d0005fc319ef366e1e47",
            "phonons/frequency_factor_to_thz": "18bc8a36961dee7534ece622b985cdf8826152fe5833806310db1f003476eca3",
            "phonons/hermitianized": "98f4a2e19cfe8b66aa8ea97fd24253b1f525b059bb82731565acc934e5a10816",
            "metadata/linewidths_json": "d3faa1e147b66d8e49d113900f8b07ea95562455e929b9867f3683bf03a1e840",
            "metadata/interactions_json": "04e7c88aa8f1f7a385ab027e983f6b001c4f3f04d67aef240d3e6c0b9a14d509",
        },
    },
    11: {
        "file_sha256": "0b104ee20f5d16e78a195c99b851ca56005da9a26161401a1b20465b924411fe",
        "problem_sha256": "f77f081e1c4df79b24d11eedfaf5994162fe9ee6491bc1ba56e35df8f26c045d",
        "controls_sha256": "39734b7a49e2d231652d509399600d1c4f068b3c7b0be074812704fb02c7b6e1",
        "campaign_manifest_sha256": "d189c76ee43fca8eab15237e652b75539f4eb079e20a9128b1ef2e61219f42e0",
        "manifest_run_identity_sha256": "961667a707a260447c4fe06a3f8294e8dd07a26108252422aa5cb227efb34569",
        "dataset_sha256": {
            "linewidths/temperatures": "3649ddad1e709a65a2ed7a78ed2a43b672dc0d5f3ed16655ac8bd6aec561c58c",
            "linewidths/grid_points": "be19422b49536d3cf679d57469f6b0774f5df4957e19894f938db39426526f26",
            "linewidths/band_indices": "70b6d6ef8ad2571ff26ade9edb1f5a576f77e76f6a5181bb1c106b4ad6d1ecd2",
            "linewidths/gamma_thz_hwhm": "38aaa56589f63bb6623c39fbe5f05ad9e1e9b95aa910bff65caa721500d855c7",
            "linewidths/gamma_normal_thz_hwhm": "a1a8afe0edd33a6b734dc8c080e7fdbad2e12a4a0fc332d2950cea9847f36f98",
            "linewidths/gamma_umklapp_thz_hwhm": "37849548e6f87f210dfd68e43b4ca0a71d683c92ddbc77d394e4fa1eac3ba570",
            "linewidths/gamma_angular_fwhm": "55c904c394645bb551ce3846a2e0a2a5d089584ac1c31b523685b467242fd3c9",
            "linewidths/decay_phase_space": "f3ebd616306d0de761cfd23d2bd639affa48e7cd52aa7fa6c46d012566c62f44",
            "linewidths/difference_phase_space": "3cbd1c0d024e2154fea0af626711c80ba88ceffa786723bacd0897896d7470cb",
            "phonons/qpoints": "a8530264126d95dca94d8fa9ac9e2f35c639577c4c8fd16f201af3f7531d799d",
            "phonons/frequencies": "7ec3647d1a91809785f647c13459f768566922ffab700f82b6f994466bafb9f1",
            "phonons/dynamical_matrices": "0e12023dff53ea72236d7d9135ccc501e6647fae34e67b43178008d57db5418a",
            "phonons/eigenvectors": "9001317cca2d34fee53a302134011b54c5361aeee0d6382084c699b17852ce43",
            "phonons/frequency_factor_to_thz": "18bc8a36961dee7534ece622b985cdf8826152fe5833806310db1f003476eca3",
            "phonons/hermitianized": "98f4a2e19cfe8b66aa8ea97fd24253b1f525b059bb82731565acc934e5a10816",
            "metadata/linewidths_json": "75ce12b0ff0b6a1074e3b49ff2fd859c4579c0c9fcae06791fcb8235851f9cd5",
            "metadata/interactions_json": "4eec5f4aa12e966c1d395dc4db592781f44b47ecb717e894c7580a349b770c3d",
        },
    },
}

_LINEWIDTH_FIELDS = (
    "temperatures",
    "grid_points",
    "band_indices",
    "gamma_thz_hwhm",
    "gamma_normal_thz_hwhm",
    "gamma_umklapp_thz_hwhm",
    "gamma_angular_fwhm",
    "decay_phase_space",
    "difference_phase_space",
)
_INTERACTION_FIELDS = (
    "grid_points",
    "band_indices",
    "q1_indices",
    "q2_indices",
    "reciprocal_vectors",
    "normal_mask",
    "matrix_elements",
    "interaction_strengths",
)
_PHONON_FIELDS = (
    "qpoints",
    "dynamical_matrices",
    "frequencies",
    "eigenvectors",
    "frequency_factor_to_thz",
    "hermitianized",
)
_METADATA_FIELDS = ("linewidths_json", "interactions_json")
ALL_ARCHIVE_DATASETS = tuple(
    [f"linewidths/{name}" for name in _LINEWIDTH_FIELDS]
    + [f"interactions/{name}" for name in _INTERACTION_FIELDS]
    + [f"phonons/{name}" for name in _PHONON_FIELDS]
    + [f"metadata/{name}" for name in _METADATA_FIELDS]
)
SELECTED_ARCHIVE_DATASETS = tuple(
    next(iter(PINNED_ARCHIVES.values()))["dataset_sha256"]
)

ARCHIVE_DTYPES = {
    **{f"linewidths/{name}": np.dtype(np.float64) for name in _LINEWIDTH_FIELDS},
    "linewidths/grid_points": np.dtype(np.int64),
    "linewidths/band_indices": np.dtype(np.int64),
    **{f"interactions/{name}": np.dtype(np.int64) for name in _INTERACTION_FIELDS},
    "interactions/normal_mask": np.dtype(np.bool_),
    "interactions/matrix_elements": np.dtype(np.complex128),
    "interactions/interaction_strengths": np.dtype(np.float64),
    "phonons/qpoints": np.dtype(np.float64),
    "phonons/dynamical_matrices": np.dtype(np.complex128),
    "phonons/frequencies": np.dtype(np.float64),
    "phonons/eigenvectors": np.dtype(np.complex128),
    "phonons/frequency_factor_to_thz": np.dtype(np.float64),
    "phonons/hermitianized": np.dtype(np.bool_),
    "metadata/linewidths_json": np.dtype(np.uint8),
    "metadata/interactions_json": np.dtype(np.uint8),
}

EXPECTED_ROOT_ATTRIBUTES = frozenset(
    {
        "schema_name",
        "schema_version",
        "campaign_manifest_sha256",
        "manifest_run_identity_sha256",
        "problem_sha256",
        "controls_sha256",
        "canonical_task_ids_json",
        "floating_point_mpi_reductions",
        "dataset_sha256_json",
        "task_artifact_sha256_json",
    }
)

HARMONIC_CHECKPOINT_DATASETS = frozenset(
    {
        "band/qpoints",
        "band/distances_inv_angstrom",
        "band/vertex_indices",
        "band/frequencies_thz",
        "dos/frequency_points_thz",
        "dos/total_states_per_thz_cell",
        "dos/projected_states_per_thz_cell",
        "mesh/frequencies_thz",
        "mesh/group_velocities_thz_angstrom",
        "reference/mesh_frequencies_thz",
        "reference/group_velocities_thz_angstrom",
        "reference/band_frequencies_thz",
        "reference/dos_total_states_per_thz_cell",
    }
)
CORE_REFERENCE_DATASETS = frozenset(
    {
        "heat_capacities_ev_per_k",
        "mode_kappa_w_mk",
        "kappa_w_mk",
    }
)
END_TO_END_REFERENCE_DATASETS = frozenset(
    {
        "temperatures",
        "grid_points",
        "band_indices",
        "qpoints",
        "gamma_thz_hwhm",
        "kappa_voigt_w_mk",
    }
)
STAGED_RATE_DATASETS = frozenset(SELECTED_ARCHIVE_DATASETS)


@dataclass(frozen=True, slots=True)
class ReleasePolicy:
    """Exact inputs accepted by a validation campaign."""

    manifest_sha256: str
    environment_lock_sha256: str
    source_sha256: Mapping[str, str]
    archives: Mapping[int, Mapping[str, Any]]
    runtime_packages: Mapping[str, str]
    meshes: tuple[int, ...] = RELEASE_MESHES
    end_to_end_meshes: tuple[int, ...] = RELEASE_END_TO_END_MESHES
    temperatures_k: tuple[float, ...] = RELEASE_TEMPERATURES_K
    sigma_thz: float = RELEASE_SIGMA_THZ
    negfpy_version: str = PINNED_NEGFPY_VERSION
    phono3py_version: str = PINNED_PHONO3PY_VERSION
    python_version: str = PINNED_PYTHON_VERSION
    openmpi_version: str = PINNED_OPENMPI_VERSION


STRICT_POLICY = ReleasePolicy(
    manifest_sha256=PINNED_V0_2_MANIFEST_SHA256,
    environment_lock_sha256=PINNED_ENVIRONMENT_LOCK_SHA256,
    source_sha256=PINNED_SOURCE_SHA256,
    archives=PINNED_ARCHIVES,
    runtime_packages=PINNED_RUNTIME_PACKAGES,
)


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    artifact_root: Path
    crystal_yaml: Path
    v0_2_release_manifest: Path
    environment_lock: Path
    environment_lock_sha256: str
    output_dir: Path
    meshes: tuple[int, ...]
    end_to_end_meshes: tuple[int, ...]
    temperatures_k: tuple[float, ...]
    sigma_thz: float
    development: bool
    resume: bool
    preflight_only: bool = False


@dataclass(frozen=True, slots=True)
class InputSnapshot:
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]
    archive_paths: Mapping[int, Path]
    archive_sha256: Mapping[int, str]
    archive_evidence: Mapping[int, Mapping[str, Any]]
    sealed_manifest: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class StagedScientificInputs:
    paths: Mapping[str, Path]
    sha256: Mapping[str, str]
    archive_paths: Mapping[int, Path]
    archive_sha256: Mapping[int, str]
    manifest_path: Path
    manifest_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported JSON value {type(value).__name__!r}.")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _dataset_sha256(dataset: h5py.Dataset) -> str:
    header = _canonical_json(
        {"dtype": np.dtype(dataset.dtype).str, "shape": list(dataset.shape)}
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    if dataset.shape:
        for first in range(dataset.shape[0]):
            values = np.ascontiguousarray(dataset[first])
            digest.update(memoryview(values).cast("B"))
    else:
        values = np.ascontiguousarray(dataset[()])
        digest.update(memoryview(values).cast("B"))
    return digest.hexdigest()


def _regular_file(value: Path, *, name: str) -> Path:
    candidate = value.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{name} must not be a symbolic link.")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{name} does not exist.") from exc
    if not resolved.is_file():
        raise ValueError(f"{name} must be a regular file.")
    return resolved


def _read_json_object(path: Path, *, name: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_nonfinite_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_nonfinite_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{name} is not valid UTF-8 JSON.") from exc
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - malformed file content
            f"{name} must contain one JSON object."
        )
    return value


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_bytes_atomic(
        path,
        (
            json.dumps(_json_value(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8"),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _archive_shapes(mesh: int) -> dict[str, tuple[int, ...]]:
    n_q = mesh**3
    n_b = 6
    return {
        "linewidths/temperatures": (1,),
        "linewidths/grid_points": (n_q,),
        "linewidths/band_indices": (n_b,),
        "linewidths/gamma_thz_hwhm": (1, n_q, n_b),
        "linewidths/gamma_normal_thz_hwhm": (1, n_q, n_b),
        "linewidths/gamma_umklapp_thz_hwhm": (1, n_q, n_b),
        "linewidths/gamma_angular_fwhm": (1, n_q, n_b),
        "linewidths/decay_phase_space": (n_q, n_b),
        "linewidths/difference_phase_space": (n_q, n_b),
        "interactions/grid_points": (n_q,),
        "interactions/band_indices": (n_b,),
        "interactions/q1_indices": (n_q, n_q),
        "interactions/q2_indices": (n_q, n_q),
        "interactions/reciprocal_vectors": (n_q, n_q, 3),
        "interactions/normal_mask": (n_q, n_q),
        "interactions/matrix_elements": (n_q, n_q, n_b, n_b, n_b),
        "interactions/interaction_strengths": (n_q, n_q, n_b, n_b, n_b),
        "phonons/qpoints": (n_q, 3),
        "phonons/dynamical_matrices": (n_q, n_b, n_b),
        "phonons/frequencies": (n_q, n_b),
        "phonons/eigenvectors": (n_q, n_b, n_b),
        "phonons/frequency_factor_to_thz": (),
        "phonons/hermitianized": (),
    }


def _validate_sealed_manifest(
    path: Path,
    *,
    expected_sha256: str,
    strict_release: bool,
) -> dict[str, Any]:
    if _sha256(path) != expected_sha256:
        raise ValueError("The v0.2.0 sealed release manifest SHA256 is not pinned.")
    value = _read_json_object(path, name="v0.2.0 sealed release manifest")
    if strict_release:
        checks = {
            "schema": value.get("schema")
            == "negfpy.silicon_0_2_0_sealed_release_package.v1",
            "phase": value.get("phase") == "sealed",
            "status": value.get("status") == "passed",
            "authorization": value.get("release_authorization_ready") is True,
            "commit": value.get("candidate", {}).get("commit") == PINNED_V0_2_COMMIT,
            "tree": value.get("candidate", {}).get("tree") == PINNED_V0_2_TREE,
            "version": value.get("candidate", {}).get("version") == "0.2.0",
            "attempt": value.get("run", {}).get("attempt_id") == PINNED_V0_2_ATTEMPT,
            "run": value.get("run", {}).get("run_id") == PINNED_V0_2_RUN,
            "visual_inspection": value.get("visual_inspection", {}).get("result")
            == "PASS",
        }
        failed = sorted(name for name, passed in checks.items() if not passed)
        if failed:
            raise ValueError("The v0.2.0 sealed manifest failed: " + ", ".join(failed))
    return value


def _expected_archive_dataset_keys() -> frozenset[str]:
    return frozenset(ALL_ARCHIVE_DATASETS)


def _read_metadata_dataset(handle: h5py.File, name: str) -> dict[str, Any]:
    try:
        raw = bytes(np.asarray(handle[name], dtype=np.uint8)).decode("ascii")
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"Archive metadata {name!r} is invalid.") from exc
    if not isinstance(value, dict) or _canonical_json(value) != raw:
        raise ValueError(f"Archive metadata {name!r} is not canonical JSON.")
    return value


def _validate_archive(
    path: Path,
    *,
    mesh: int,
    expected: Mapping[str, Any],
    temperatures_k: tuple[float, ...],
    sigma_thz: float,
) -> dict[str, Any]:
    """Authenticate one merged archive without reading FC3 interaction tensors."""

    shapes = _archive_shapes(mesh)
    with h5py.File(path, "r") as handle:
        if frozenset(str(name) for name in handle.attrs) != EXPECTED_ROOT_ATTRIBUTES:
            raise ValueError(f"Mesh {mesh} archive root attributes are not exact.")
        root_checks = {
            "schema_name": str(handle.attrs["schema_name"])
            == "negfpy.bulk_target_q_merged",
            "schema_version": int(handle.attrs["schema_version"]) == 1,
            "problem_sha256": str(handle.attrs["problem_sha256"])
            == expected["problem_sha256"],
            "controls_sha256": str(handle.attrs["controls_sha256"])
            == expected["controls_sha256"],
            "campaign_manifest_sha256": str(handle.attrs["campaign_manifest_sha256"])
            == expected["campaign_manifest_sha256"],
            "manifest_run_identity_sha256": str(
                handle.attrs["manifest_run_identity_sha256"]
            )
            == expected["manifest_run_identity_sha256"],
            "no_float_mpi_reductions": bool(
                handle.attrs["floating_point_mpi_reductions"]
            )
            is False,
        }
        failed = sorted(name for name, passed in root_checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Mesh {mesh} archive root identity failed: " + ", ".join(failed)
            )

        paths: list[str] = []

        def collect(name: str, value: Any) -> None:
            if isinstance(value, h5py.Dataset):
                paths.append(name)

        handle.visititems(collect)
        if frozenset(paths) != _expected_archive_dataset_keys():
            raise ValueError(f"Mesh {mesh} archive dataset schema is not exact.")
        declared_hashes = json.loads(str(handle.attrs["dataset_sha256_json"]))
        if not isinstance(declared_hashes, dict) or frozenset(declared_hashes) != (
            _expected_archive_dataset_keys()
        ):
            raise ValueError(f"Mesh {mesh} dataset SHA256 manifest is not exact.")
        expected_hashes = expected["dataset_sha256"]
        if frozenset(expected_hashes) != frozenset(SELECTED_ARCHIVE_DATASETS):
            raise ValueError("Internal release policy lacks selected dataset hashes.")
        for name in ALL_ARCHIVE_DATASETS:
            if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                raise ValueError(  # noqa: TRY004 - malformed archive content
                    f"Archive field {name!r} is not an internal hard link."
                )
            dataset = handle[name]
            if np.dtype(dataset.dtype) != ARCHIVE_DTYPES[name]:
                raise ValueError(f"Archive field {name!r} has the wrong dtype.")
            if {str(key) for key in dataset.attrs} != {"sha256"}:
                raise ValueError(f"Archive field {name!r} attributes are not exact.")
            if str(dataset.attrs["sha256"]) != declared_hashes[name]:
                raise ValueError(
                    f"Archive field {name!r} digest is not declared exactly."
                )
            expected_shape = shapes.get(name)
            if expected_shape is not None and tuple(dataset.shape) != expected_shape:
                raise ValueError(f"Archive field {name!r} has the wrong shape.")
        for name in SELECTED_ARCHIVE_DATASETS:
            if declared_hashes[name] != expected_hashes[name]:
                raise ValueError(f"Pinned archive digest for {name!r} changed.")
            if _dataset_sha256(handle[name]) != expected_hashes[name]:
                raise ValueError(f"Archive field {name!r} failed SHA256 validation.")

        n_q = mesh**3
        task_ids = json.loads(str(handle.attrs["canonical_task_ids_json"]))
        expected_tasks = [f"target-q-{index:08d}" for index in range(n_q)]
        if task_ids != expected_tasks:
            raise ValueError(f"Mesh {mesh} canonical task selectors are incomplete.")
        task_hashes = json.loads(str(handle.attrs["task_artifact_sha256_json"]))
        if (
            not isinstance(task_hashes, list)
            or len(task_hashes) != n_q
            or any(
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in task_hashes
            )
        ):
            raise ValueError(f"Mesh {mesh} task artifact hashes are incomplete.")

        grid_points = np.asarray(handle["linewidths/grid_points"])
        bands = np.asarray(handle["linewidths/band_indices"])
        temperatures = np.asarray(handle["linewidths/temperatures"])
        if not np.array_equal(grid_points, np.arange(n_q, dtype=np.int64)):
            raise ValueError(f"Mesh {mesh} does not select the full q mesh.")
        if not np.array_equal(bands, np.arange(6, dtype=np.int64)):
            raise ValueError(f"Mesh {mesh} does not select all silicon bands.")
        if not np.array_equal(temperatures, np.asarray(temperatures_k)):
            raise ValueError(f"Mesh {mesh} temperatures are not pinned.")
        if not bool(np.asarray(handle["phonons/hermitianized"])):
            raise ValueError(f"Mesh {mesh} phonons were not Hermitianized.")
        if float(np.asarray(handle["phonons/frequency_factor_to_thz"])) != (
            FREQUENCY_FACTOR_TO_THZ
        ):
            raise ValueError(f"Mesh {mesh} frequency factor is not pinned.")

        from negfpy.anharmonic.momentum import UniformBZMesh

        uniform_mesh = UniformBZMesh((mesh, mesh, mesh))
        expected_qpoints = np.asarray(uniform_mesh.addresses, dtype=np.float64) / mesh
        qpoints = np.asarray(handle["phonons/qpoints"])
        difference = qpoints - expected_qpoints
        difference -= np.rint(difference)
        if not np.allclose(difference, 0.0, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"Mesh {mesh} q-point order is not canonical.")

        linewidth_metadata = _read_metadata_dataset(handle, "metadata/linewidths_json")
        interaction_metadata = _read_metadata_dataset(
            handle, "metadata/interactions_json"
        )
        metadata_checks = {
            "linewidth_schema": linewidth_metadata.get("schema")
            == "negfpy.bulk_three_phonon_linewidths.v1",
            "interaction_schema": interaction_metadata.get("schema")
            == "negfpy.bulk_three_phonon_interactions.v1",
            "sigma": linewidth_metadata.get("sigma_thz") == sigma_thz,
            "sigma_cutoff": linewidth_metadata.get("sigma_cutoff") is None,
            "cutoff": linewidth_metadata.get("cutoff_frequency_thz") == CUTOFF_THZ,
            "temperature_units": linewidth_metadata.get("temperature_units") == "K",
            "gamma_semantics": linewidth_metadata.get("gamma_thz_semantics")
            == "cyclic-frequency HWHM",
            "gamma_conversion": linewidth_metadata.get("gamma_angular_fwhm_conversion")
            == "4*pi*gamma_thz_hwhm",
            "mesh": interaction_metadata.get("mesh") == [mesh, mesh, mesh],
            "n_qpoints": interaction_metadata.get("n_qpoints") == n_q,
            "normalization": interaction_metadata.get("full_mesh_normalization")
            == 1.0 / n_q,
            "r0_average": interaction_metadata.get("make_r0_average") is True,
            "method": interaction_metadata.get("method")
            == "deterministic_serial_full_mesh",
            "phase": interaction_metadata.get("phase_convention") == "exp(+2π i q·r)",
            "triplet": interaction_metadata.get("triplet_convention") == "q0+q1+q2=G",
        }
        failed = sorted(name for name, passed in metadata_checks.items() if not passed)
        if failed:
            raise ValueError(
                f"Mesh {mesh} archive metadata failed: " + ", ".join(failed)
            )

        gamma = np.asarray(handle["linewidths/gamma_thz_hwhm"])
        gamma_n = np.asarray(handle["linewidths/gamma_normal_thz_hwhm"])
        gamma_u = np.asarray(handle["linewidths/gamma_umklapp_thz_hwhm"])
        angular = np.asarray(handle["linewidths/gamma_angular_fwhm"])
        if not np.allclose(gamma, gamma_n + gamma_u, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError(f"Mesh {mesh} archived N/U linewidth identity failed.")
        if not np.allclose(angular, 4.0 * np.pi * gamma, rtol=1.0e-13, atol=1.0e-15):
            raise ValueError(f"Mesh {mesh} archived 4*pi linewidth identity failed.")
        return {
            "problem_sha256": str(handle.attrs["problem_sha256"]),
            "controls_sha256": str(handle.attrs["controls_sha256"]),
            "dataset_sha256": {
                name: declared_hashes[name] for name in SELECTED_ARCHIVE_DATASETS
            },
            "n_qpoints": n_q,
            "n_bands": 6,
            "temperature_count": len(temperatures_k),
            "metadata_sha256": {
                name: declared_hashes[name]
                for name in ("metadata/linewidths_json", "metadata/interactions_json")
            },
        }


def _validate_release_config(config: ValidationConfig, policy: ReleasePolicy) -> None:
    if config.preflight_only and config.resume:
        raise ValueError("--preflight-only and --resume are mutually exclusive.")
    if config.development:
        if not config.meshes or len(set(config.meshes)) != len(config.meshes):
            raise ValueError("Development meshes must be nonempty and unique.")
        if any(mesh <= 0 for mesh in config.meshes):
            raise ValueError("Every mesh size must be positive.")
        if any(mesh not in config.meshes for mesh in config.end_to_end_meshes):
            raise ValueError("End-to-end meshes must be a subset of meshes.")
        if config.sigma_thz <= 0.0 or not np.isfinite(config.sigma_thz):
            raise ValueError("sigma_thz must be positive.")
        return
    checks = {
        "meshes": config.meshes == policy.meshes,
        "end_to_end_meshes": config.end_to_end_meshes == policy.end_to_end_meshes,
        "temperatures": config.temperatures_k == policy.temperatures_k,
        "sigma": config.sigma_thz == policy.sigma_thz,
        "environment_lock_sha256": config.environment_lock_sha256
        == policy.environment_lock_sha256,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Release settings are immutable; changed: " + ", ".join(failed)
        )


def _parse_explicit_lock(path: Path) -> tuple[tuple[str, str | None], ...]:
    """Return package URLs and any authenticated SHA256 URL fragments."""

    source = _regular_file(path, name="explicit environment lock")
    meaningful = [
        line.strip()
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not meaningful or meaningful[0] != "@EXPLICIT":
        raise ValueError("Environment lock must start with @EXPLICIT.")
    entries: list[tuple[str, str | None]] = []
    for line in meaningful[1:]:
        url, fragment = urldefrag(line)
        if not url:
            raise ValueError("Explicit environment lock contains an empty URL.")
        if fragment:
            prefix = "sha256="
            if not fragment.startswith(prefix):
                raise ValueError("Explicit lock URL fragment must declare sha256.")
            artifact_sha256 = fragment.removeprefix(prefix)
            if len(artifact_sha256) != 64:
                raise ValueError("Explicit lock SHA256 fragment is malformed.")
            try:
                int(artifact_sha256, 16)
            except ValueError as exc:
                raise ValueError("Explicit lock SHA256 fragment is malformed.") from exc
            artifact_sha256 = artifact_sha256.lower()
        else:
            artifact_sha256 = None
        entries.append((url, artifact_sha256))
    if not entries:
        raise ValueError("Explicit environment lock contains no package URLs.")
    urls = [url for url, _artifact_sha256 in entries]
    if len(set(urls)) != len(urls):
        raise ValueError("Explicit environment lock contains duplicate package URLs.")
    return tuple(entries)


def _safe_prefix_member(prefix: Path, relative: str, *, context: str) -> Path:
    candidate_relative = Path(relative)
    if (
        not relative
        or candidate_relative.is_absolute()
        or ".." in candidate_relative.parts
    ):
        raise ValueError(f"{context} contains an unsafe prefix path {relative!r}.")
    candidate = prefix.joinpath(candidate_relative)
    try:
        candidate.parent.resolve(strict=True).relative_to(prefix)
    except (FileNotFoundError, ValueError) as exc:
        raise ValueError(f"{context} path {relative!r} escapes the prefix.") from exc
    return candidate


def _mutable_conda_hash_category(relative: str) -> str | None:
    """Classify the sole authenticated mutable conda path family."""

    path = PurePosixPath(relative)
    if path.suffix == ".pyc" and path.parent.name == "__pycache__":
        return "python_generated_bytecode"
    return None


def _validate_conda_prefix_against_lock(
    lock_path: Path,
    prefix: Path,
) -> dict[str, Any]:
    """Authenticate the active conda prefix against an exact explicit lock."""

    lock_entries = _parse_explicit_lock(lock_path)
    lock_urls = tuple(url for url, _artifact_sha256 in lock_entries)
    lock_artifact_sha256 = dict(lock_entries)
    root = prefix.expanduser().resolve(strict=True)
    metadata_dir = root / "conda-meta"
    if metadata_dir.is_symlink() or not metadata_dir.is_dir():
        raise ValueError("Active prefix has no regular conda-meta directory.")
    metadata_paths = sorted(metadata_dir.glob("*.json"))
    if not metadata_paths:
        raise ValueError("Active prefix contains no conda package records.")
    records: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    record_identities: list[dict[str, Any]] = []
    record_artifact_sha256_count = 0
    record_artifact_md5_count = 0
    cache_verified_artifact_sha256_count = 0
    hashed_file_count = 0
    hash_declared_file_count = 0
    skipped_declared_hash_count = 0
    verified_path_count = 0
    skipped_hash_paths: list[tuple[str, str]] = []
    recognized_mutable_paths: list[tuple[str, str]] = []
    for metadata_path in metadata_paths:
        source = _regular_file(metadata_path, name="conda package record")
        record = _read_json_object(source, name="conda package record")
        raw_url = record.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            raise ValueError(f"Conda package record {source.name!r} has no URL.")
        url = urldefrag(raw_url)[0]
        if url in records:
            raise ValueError(f"Active prefix contains duplicate URL {url!r}.")
        records[url] = (source, record)

    lock_set = set(lock_urls)
    record_set = set(records)
    if lock_set != record_set:
        missing = sorted(lock_set - record_set)
        extra = sorted(record_set - lock_set)
        raise ValueError(
            "Active conda prefix URL set differs from explicit lock: "
            f"missing={missing}, extra={extra}."
        )

    for url in lock_urls:
        source, record = records[url]
        filename = PurePosixPath(urlparse(url).path).name
        subdir = PurePosixPath(urlparse(url).path).parent.name
        expected_record_filename = (
            f"{record.get('name')}-{record.get('version')}-{record.get('build')}.json"
        )
        if (
            not filename
            or record.get("fn") != filename
            or not isinstance(record.get("name"), str)
            or not record.get("name")
            or not isinstance(record.get("version"), str)
            or not record.get("version")
            or not isinstance(record.get("build"), str)
            or not record.get("build")
            or isinstance(record.get("build_number"), bool)
            or not isinstance(record.get("build_number"), int)
            or int(record["build_number"]) < 0
            or record.get("subdir") != subdir
            or source.name != expected_record_filename
            or isinstance(record.get("size"), bool)
            or not isinstance(record.get("size"), int)
            or int(record["size"]) <= 0
        ):
            raise ValueError(
                f"Conda package record {source.name!r} identity conflicts with its URL."
            )
        expected_artifact_sha256 = lock_artifact_sha256[url]
        record_md5 = record.get("md5")
        if record_md5 is not None:
            if not isinstance(record_md5, str) or len(record_md5) != 32:
                raise ValueError(
                    f"Conda package record {source.name!r} has an invalid "
                    "artifact MD5."
                )
            try:
                int(record_md5, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Conda package record {source.name!r} has an invalid "
                    "artifact MD5."
                ) from exc
            record_md5 = record_md5.lower()
            record_artifact_md5_count += 1
        elif expected_artifact_sha256 is None:
            raise ValueError(
                f"Conda package record {source.name!r} has no authenticated "
                "artifact digest."
            )
        record_sha256 = record.get("sha256")
        if record_sha256 is not None and (
            not isinstance(record_sha256, str) or len(record_sha256) != 64
        ):
            raise ValueError(
                f"Conda package record {source.name!r} has an invalid artifact SHA256."
            )
        if isinstance(record_sha256, str):
            try:
                int(record_sha256, 16)
            except ValueError as exc:
                raise ValueError(
                    f"Conda package record {source.name!r} has an invalid artifact SHA256."
                ) from exc
            record_sha256 = record_sha256.lower()
            record_artifact_sha256_count += 1
        if (
            expected_artifact_sha256 is not None
            and record_sha256 is not None
            and record_sha256 != expected_artifact_sha256
        ):
            raise ValueError(
                f"Conda package record artifact SHA256 conflicts with lock for {url!r}."
            )
        artifact_sha256_verification = (
            "record_sha256"
            if expected_artifact_sha256 is not None
            else "record_md5"
        )
        if expected_artifact_sha256 is not None and record_sha256 is None:
            extracted_package_dir = record.get("extracted_package_dir")
            if (
                not isinstance(extracted_package_dir, str)
                or not extracted_package_dir
                or not Path(extracted_package_dir).is_absolute()
            ):
                raise ValueError(
                    f"Conda package record {source.name!r} has no usable "
                    "artifact-cache location."
                )
            cached_artifact = Path(extracted_package_dir).parent / filename
            if cached_artifact.is_symlink() or not cached_artifact.is_file():
                raise ValueError(
                    f"Conda package record {source.name!r} has no regular "
                    "cached artifact."
                )
            if cached_artifact.stat().st_size != int(record["size"]):
                raise ValueError(
                    f"Cached conda artifact size changed for {source.name!r}."
                )
            if _sha256(cached_artifact) != expected_artifact_sha256:
                raise ValueError(
                    f"Cached conda artifact SHA256 conflicts with lock for {url!r}."
                )
            artifact_sha256_verification = "package_cache_sha256"
            cache_verified_artifact_sha256_count += 1
        record_identities.append(
            {
                "url": url,
                "name": record["name"],
                "version": record["version"],
                "build": record["build"],
                "build_number": record["build_number"],
                "subdir": record["subdir"],
                "filename": filename,
                "size": record["size"],
                "md5": record_md5,
                "sha256": record_sha256,
                "authenticated_sha256": expected_artifact_sha256,
                "sha256_verification": artifact_sha256_verification,
            }
        )

    for url in lock_urls:
        source, record = records[url]
        paths_data = record.get("paths_data")
        if isinstance(paths_data, Mapping) and isinstance(
            paths_data.get("paths"), list
        ):
            path_records = paths_data["paths"]
        else:
            files = record.get("files")
            if not isinstance(files, list):
                raise TypeError(
                    f"Conda package record {source.name!r} has no path inventory."
                )
            path_records = [
                {"_path": relative, "path_type": "hardlink"} for relative in files
            ]
        for item in path_records:
            if not isinstance(item, Mapping):
                raise TypeError(f"Conda package record {source.name!r} is malformed.")
            relative = item.get("_path")
            path_type = item.get("path_type")
            if not isinstance(relative, str) or not isinstance(path_type, str):
                raise TypeError(
                    f"Conda package record {source.name!r} has an invalid path entry."
                )
            candidate = _safe_prefix_member(
                root,
                relative,
                context=f"Conda package record {source.name!r}",
            )
            normalized_type = path_type.lower()
            mutable_category = _mutable_conda_hash_category(relative)
            if normalized_type in {
                "hardlink",
                "pyc_file",
                "unix_python_entry_point",
            }:
                if candidate.is_symlink() or not candidate.is_file():
                    raise ValueError(
                        f"Conda prefix file path {relative!r} is not a regular file."
                    )
                if normalized_type == "pyc_file" and mutable_category is None:
                    raise ValueError(
                        f"Conda pyc_file path {relative!r} is outside __pycache__."
                    )
            elif normalized_type == "softlink":
                if not candidate.is_symlink():
                    raise ValueError(
                        f"Conda prefix softlink path {relative!r} is not a symlink."
                    )
                target = Path(os.readlink(candidate))
                if target.is_absolute():
                    raise ValueError(
                        f"Conda prefix softlink {relative!r} has an absolute target."
                    )
                try:
                    candidate.resolve(strict=True).relative_to(root)
                except (FileNotFoundError, ValueError) as exc:
                    raise ValueError(
                        f"Conda prefix softlink {relative!r} escapes the prefix."
                    ) from exc
            elif normalized_type == "directory":
                if candidate.is_symlink() or not candidate.is_dir():
                    raise ValueError(
                        f"Conda prefix directory path {relative!r} is invalid."
                    )
            else:
                raise ValueError(
                    f"Conda package record has unsupported path type {path_type!r}."
                )
            expected_sha256 = item.get("sha256_in_prefix")
            if expected_sha256 is not None:
                if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                    raise ValueError(
                        f"Conda prefix SHA256 metadata is invalid for {relative!r}."
                    )
                try:
                    int(expected_sha256, 16)
                except ValueError as exc:
                    raise ValueError(
                        f"Conda prefix SHA256 metadata is invalid for {relative!r}."
                    ) from exc
                hash_declared_file_count += 1
                if normalized_type == "directory":
                    raise ValueError(
                        f"Conda directory {relative!r} declares a file SHA256."
                    )
                if (
                    normalized_type == "softlink"
                    and candidate.resolve(strict=True).is_dir()
                ):
                    actual_sha256 = hashlib.sha256(b"").hexdigest()
                else:
                    actual_sha256 = _sha256(candidate)
                if actual_sha256 == expected_sha256:
                    hashed_file_count += 1
                elif mutable_category is None:
                    raise ValueError(f"Conda prefix SHA256 changed for {relative!r}.")
                else:
                    skipped_declared_hash_count += 1
                    skipped_hash_paths.append((mutable_category, relative))
            if mutable_category is not None:
                recognized_mutable_paths.append((mutable_category, relative))
            verified_path_count += 1
    skipped_categories: dict[str, int] = {}
    for category, _relative in skipped_hash_paths:
        skipped_categories[category] = skipped_categories.get(category, 0) + 1
    skipped_hash_paths.sort()
    recognized_mutable_paths.sort()
    return {
        "passed": True,
        "prefix": str(root),
        "lock_sha256": _sha256(lock_path),
        "explicit_url_count": len(lock_urls),
        "explicit_artifact_sha256_count": sum(
            value is not None for value in lock_artifact_sha256.values()
        ),
        "explicit_artifact_sha256_digest": hashlib.sha256(
            _canonical_json(
                sorted(
                    (url, value)
                    for url, value in lock_artifact_sha256.items()
                    if value is not None
                )
            ).encode("utf-8")
        ).hexdigest(),
        "package_record_count": len(records),
        "record_artifact_sha256_count": record_artifact_sha256_count,
        "record_artifact_md5_count": record_artifact_md5_count,
        "cache_verified_artifact_sha256_count": (
            cache_verified_artifact_sha256_count
        ),
        "record_artifact_identity_sha256": hashlib.sha256(
            _canonical_json(record_identities).encode("utf-8")
        ).hexdigest(),
        "verified_path_count": verified_path_count,
        "hash_declared_file_count": hash_declared_file_count,
        "hashed_file_count": hashed_file_count,
        "skipped_declared_hash_count": skipped_declared_hash_count,
        "skipped_hash_policy": "immediate___pycache___pyc_only.v1",
        "skipped_hash_categories": skipped_categories,
        "skipped_hash_path_count": len(skipped_hash_paths),
        "skipped_hash_paths_sha256": hashlib.sha256(
            _canonical_json(skipped_hash_paths).encode("utf-8")
        ).hexdigest(),
        "recognized_mutable_path_count": len(recognized_mutable_paths),
        "recognized_mutable_paths_sha256": hashlib.sha256(
            _canonical_json(recognized_mutable_paths).encode("utf-8")
        ).hexdigest(),
        "explicit_urls_sha256": hashlib.sha256(
            _canonical_json(lock_urls).encode("utf-8")
        ).hexdigest(),
    }


def _thread_runtime_identity(prefix: Path) -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    root = prefix.resolve()
    pools: list[dict[str, Any]] = []
    for item in threadpool_info():
        filepath = item.get("filepath")
        within_prefix = False
        if isinstance(filepath, str) and filepath:
            try:
                Path(filepath).resolve(strict=True).relative_to(root)
            except (FileNotFoundError, ValueError):
                pass
            else:
                within_prefix = True
        pools.append(
            {
                "filepath": filepath,
                "internal_api": item.get("internal_api"),
                "prefix": item.get("prefix"),
                "num_threads": item.get("num_threads"),
                "version": item.get("version"),
                "within_prefix": within_prefix,
            }
        )
    return {
        "environment": {
            name: os.environ.get(name) for name in THREAD_LIMIT_ENVIRONMENT
        },
        "threadpools": sorted(pools, key=lambda item: str(item["filepath"])),
    }


def _conda_owned_prefix_paths(prefix: Path) -> frozenset[str]:
    owned: set[str] = set()
    for metadata_path in sorted((prefix / "conda-meta").glob("*.json")):
        record = _read_json_object(
            _regular_file(metadata_path, name="conda package record"),
            name="conda package record",
        )
        paths_data = record.get("paths_data")
        if isinstance(paths_data, Mapping) and isinstance(
            paths_data.get("paths"), list
        ):
            items = paths_data["paths"]
            for item in items:
                if not isinstance(item, Mapping) or not isinstance(
                    item.get("_path"), str
                ):
                    raise TypeError(
                        f"Conda package record {metadata_path.name!r} is malformed."
                    )
                owned.add(str(PurePosixPath(item["_path"])))
        else:
            files = record.get("files")
            if not isinstance(files, list) or not all(
                isinstance(value, str) for value in files
            ):
                raise TypeError(
                    f"Conda package record {metadata_path.name!r} has no path inventory."
                )
            owned.update(str(PurePosixPath(value)) for value in files)
    return frozenset(owned)


def _negfpy_wheel_owned_prefix_paths(prefix: Path) -> frozenset[str]:
    distribution = importlib.metadata.distribution("negfpy")
    files = distribution.files
    if files is None:
        raise RuntimeError("Installed NEGFPy distribution has no RECORD inventory.")
    owned: set[str] = set()
    for item in files:
        path = Path(distribution.locate_file(item)).resolve(strict=False)
        try:
            relative = path.relative_to(prefix)
        except ValueError as exc:
            raise RuntimeError(
                "Installed NEGFPy RECORD contains a path outside the active prefix."
            ) from exc
        owned.add(relative.as_posix())
    if not owned:
        raise RuntimeError("Installed NEGFPy RECORD inventory is empty.")
    return frozenset(owned)


def _generated_pyc_source(relative: str) -> str | None:
    path = PurePosixPath(relative)
    if path.suffix != ".pyc" or path.parent.name != "__pycache__":
        return None
    stem = path.name.split(".", maxsplit=1)[0]
    return str(path.parent.parent / f"{stem}.py")


def _import_code_ownership_identity(prefix: Path) -> dict[str, Any]:
    """Prove import-bearing files are conda-owned or in the NEGFPy wheel RECORD."""

    conda_owned = _conda_owned_prefix_paths(prefix)
    try:
        wheel_owned = _negfpy_wheel_owned_prefix_paths(prefix)
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        wheel_owned = frozenset()
        wheel_inventory_available = False
    else:
        wheel_inventory_available = True
    code_paths: list[str] = []
    generated_pyc_paths: list[str] = []
    unowned: list[str] = []
    python_directory = f"python{sys.version_info.major}.{sys.version_info.minor}"
    for site_packages in (prefix / "lib" / python_directory / "site-packages",):
        if site_packages.is_symlink() or not site_packages.is_dir():
            raise RuntimeError("Active prefix site-packages path is not a directory.")
        for path in site_packages.rglob("*"):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = path.relative_to(prefix).as_posix()
            basename = path.name.lower()
            code_bearing = bool(
                path.suffix.lower() in {".py", ".pyc", ".pth", ".so"}
                or basename.startswith(("sitecustomize", "usercustomize"))
            )
            if not code_bearing:
                continue
            owned = relative in conda_owned or relative in wheel_owned
            generated_source = _generated_pyc_source(relative)
            if generated_source is not None:
                generated_pyc_paths.append(relative)
                owned = (
                    owned
                    or generated_source in conda_owned
                    or generated_source in wheel_owned
                )
            else:
                code_paths.append(relative)
            if not owned:
                unowned.append(relative)
    code_paths.sort()
    generated_pyc_paths.sort()
    unowned.sort()
    return {
        "passed": bool(wheel_inventory_available and not unowned),
        "code_path_count": len(code_paths),
        "code_paths_sha256": hashlib.sha256(
            _canonical_json(code_paths).encode("utf-8")
        ).hexdigest(),
        "generated_pyc_path_count": len(generated_pyc_paths),
        "generated_pyc_paths_sha256": hashlib.sha256(
            _canonical_json(generated_pyc_paths).encode("utf-8")
        ).hexdigest(),
        "unowned_code_path_count": len(unowned),
        "unowned_code_paths": unowned,
        "conda_owned_path_count": len(conda_owned),
        "negfpy_wheel_owned_path_count": len(wheel_owned),
        "negfpy_wheel_inventory_available": wheel_inventory_available,
    }


def _sys_path_identity(repository: Path, prefix: Path) -> dict[str, Any]:
    scripts = (repository / "scripts").resolve()
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    exact_allowed = {
        scripts,
        (prefix / "lib" / f"python{sys.version_info.major}{sys.version_info.minor}.zip").resolve(
            strict=False
        ),
        (prefix / "lib" / version).resolve(strict=False),
        (prefix / "lib" / version / "lib-dynload").resolve(strict=False),
        (prefix / "lib" / version / "site-packages").resolve(strict=False),
    }
    normalized: list[str] = []
    empty_entry_count = 0
    allowed: list[bool] = []
    for raw in sys.path:
        if not raw:
            empty_entry_count += 1
            normalized.append("")
            allowed.append(False)
            continue
        path = Path(raw).resolve(strict=False)
        normalized.append(str(path))
        allowed.append(path in exact_allowed)
    return {
        "entries": normalized,
        "entry_count": len(normalized),
        "empty_entry_count": empty_entry_count,
        "duplicate_entry_count": len(normalized) - len(set(normalized)),
        "all_entries_allowed": all(allowed),
        "scripts_first": bool(normalized and normalized[0] == str(scripts)),
        "allowed_entries": sorted(str(path) for path in exact_allowed),
    }


def _pinned_import_ownership(
    prefix: Path,
    import_paths: Mapping[str, str],
) -> dict[str, bool]:
    owned = set(_conda_owned_prefix_paths(prefix))
    try:
        owned.update(_negfpy_wheel_owned_prefix_paths(prefix))
    except (importlib.metadata.PackageNotFoundError, RuntimeError):
        pass
    result: dict[str, bool] = {}
    for name, raw_path in import_paths.items():
        path = Path(raw_path).resolve(strict=True)
        try:
            relative = path.relative_to(prefix).as_posix()
        except ValueError:
            result[name] = False
            continue
        generated_source = _generated_pyc_source(relative)
        result[name] = bool(
            relative in owned
            or (generated_source is not None and generated_source in owned)
        )
    return result


def _openmpi_version_from_banner(output: str) -> str:
    """Parse official Open MPI launcher names, including resolved symlinks."""

    match = re.search(
        r"(?im)^(?:mpiexec|mpirun|orterun)\s+\(Open MPI\)\s+"
        r"([0-9]+\.[0-9]+\.[0-9]+)\s*$",
        output,
    )
    return "unknown" if match is None else match.group(1)


def _runtime_identity(repository: Path) -> dict[str, Any]:
    from negfpy.anharmonic._provenance import (
        _package_source_sha256,
        code_provenance,
    )

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    provenance = code_provenance()
    negfpy_version = importlib.metadata.version("negfpy")
    prefix = Path(sys.prefix).resolve()
    executable = Path(sys.executable).resolve()
    try:
        executable.relative_to(prefix)
    except ValueError:
        executable_within_prefix = False
    else:
        executable_within_prefix = True
    package_versions: dict[str, str] = {}
    module_versions: dict[str, str] = {}
    import_paths: dict[str, str] = {}
    imports_within_prefix: dict[str, bool] = {}
    for package_name in PINNED_RUNTIME_PACKAGES:
        package_versions[package_name] = importlib.metadata.version(package_name)
        module = importlib.import_module(package_name)
        module_versions[package_name] = str(getattr(module, "__version__", "unknown"))
        module_file = Path(module.__file__).resolve()
        import_paths[package_name] = str(module_file)
        try:
            module_file.relative_to(prefix)
        except ValueError:
            imports_within_prefix[package_name] = False
        else:
            imports_within_prefix[package_name] = True

    mpiexec = shutil.which("mpiexec")
    if mpiexec is None:
        mpiexec_record = {
            "path": "unknown",
            "version": "unknown",
            "version_output": "",
            "within_prefix": False,
        }
    else:
        mpiexec_path = Path(mpiexec).resolve()
        completed = subprocess.run(
            [str(mpiexec_path), "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
        output = (completed.stdout + "\n" + completed.stderr).strip()
        try:
            mpiexec_path.relative_to(prefix)
        except ValueError:
            within_prefix = False
        else:
            within_prefix = True
        mpiexec_record = {
            "path": str(mpiexec_path),
            "version": _openmpi_version_from_banner(output),
            "version_output": output,
            "within_prefix": within_prefix,
        }

    imported_negfpy = Path(sys.modules["negfpy"].__file__).resolve()
    repository_package = repository / "src" / "negfpy"
    repository_package_sha256 = _package_source_sha256(repository_package)
    try:
        imported_negfpy.relative_to(repository_package)
    except ValueError:
        import_from_repository = False
    else:
        import_from_repository = True
    try:
        imported_negfpy.relative_to(prefix)
    except ValueError:
        import_from_prefix = False
    else:
        import_from_prefix = True
    imported_payload_tied = bool(
        repository_package_sha256 is not None
        and provenance.get("source_tree_sha256") == repository_package_sha256
        and (import_from_repository or import_from_prefix)
    )
    script_path = Path(__file__).resolve()
    return {
        "repository": str(Path(git("rev-parse", "--show-toplevel")).resolve()),
        "commit": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "clean": status == "",
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "dirty_entry_count": 0 if not status else len(status.splitlines()),
        "negfpy_version": negfpy_version,
        "python_version": platform.python_version(),
        "python_build": sys.version,
        "executable": str(executable),
        "executable_within_prefix": executable_within_prefix,
        "prefix": str(prefix),
        "package_versions": package_versions,
        "module_versions": module_versions,
        "import_paths": import_paths,
        "imports_within_prefix": imports_within_prefix,
        "pinned_import_paths_owned": _pinned_import_ownership(
            prefix, import_paths
        ),
        "mpiexec": mpiexec_record,
        "code_provenance": provenance,
        "imported_negfpy": str(imported_negfpy),
        "repository_package_source_sha256": repository_package_sha256 or "unknown",
        "negfpy_import_from_repository": import_from_repository,
        "negfpy_import_from_prefix": import_from_prefix,
        "negfpy_import_tied_to_repository_payload": imported_payload_tied,
        "validation_script": str(script_path),
        "validation_script_from_repository_scripts": script_path.parent
        == (repository / "scripts").resolve(),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
        "sys_path": _sys_path_identity(repository, prefix),
        "import_code_ownership": _import_code_ownership_identity(prefix),
        "thread_limits": _thread_runtime_identity(prefix),
        "user_site_isolation": {
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "no_user_site": bool(sys.flags.no_user_site),
        },
    }


def _require_release_runtime(
    identity: Mapping[str, Any],
    *,
    policy: ReleasePolicy,
) -> None:
    provenance = identity["code_provenance"]
    thread_limits = identity.get("thread_limits", {})
    thread_environment = thread_limits.get("environment", {})
    threadpools = thread_limits.get("threadpools", [])
    environment_binding = identity.get("environment_binding", {})
    user_site_isolation = identity.get("user_site_isolation", {})
    sys_path_identity = identity.get("sys_path", {})
    import_code_ownership = identity.get("import_code_ownership", {})
    checks = {
        "repository_clean": identity.get("clean") is True,
        "negfpy_version": identity.get("negfpy_version") == policy.negfpy_version,
        "python_version": identity.get("python_version") == policy.python_version,
        "python_executable_within_prefix": identity.get("executable_within_prefix")
        is True,
        "package_versions": identity.get("package_versions")
        == dict(policy.runtime_packages),
        "module_versions": identity.get("module_versions")
        == dict(policy.runtime_packages),
        "imports_within_prefix": identity.get("imports_within_prefix")
        == {name: True for name in policy.runtime_packages},
        "pinned_import_paths_owned": identity.get("pinned_import_paths_owned")
        == {name: True for name in policy.runtime_packages},
        "mpiexec_version": identity.get("mpiexec", {}).get("version")
        == policy.openmpi_version,
        "mpiexec_within_prefix": identity.get("mpiexec", {}).get("within_prefix")
        is True,
        "version_consistent": provenance.get("package_version_consistent") is True,
        "provenance_version": provenance.get("installed_package_version")
        == policy.negfpy_version,
        "provenance_revision": provenance.get("git_revision") == identity.get("commit"),
        "provenance_clean": provenance.get("git_dirty") is False,
        "source_hash_known": isinstance(provenance.get("source_tree_sha256"), str)
        and len(str(provenance.get("source_tree_sha256"))) == 64,
        "imported_payload_tied": identity.get(
            "negfpy_import_tied_to_repository_payload"
        )
        is True,
        "negfpy_wheel_import": identity.get("negfpy_import_from_prefix") is True,
        "validation_script": identity.get(
            "validation_script_from_repository_scripts"
        )
        is True,
        "pythonpath_empty": identity.get("PYTHONPATH") in (None, ""),
        "sys_path_allowed": sys_path_identity.get("all_entries_allowed") is True,
        "sys_path_scripts_first": sys_path_identity.get("scripts_first") is True,
        "sys_path_no_empty": sys_path_identity.get("empty_entry_count") == 0,
        "sys_path_no_duplicates": sys_path_identity.get("duplicate_entry_count") == 0,
        "import_code_owned": import_code_ownership.get("passed") is True,
        "explicit_environment_binding": environment_binding.get("passed") is True,
        "explicit_environment_prefix": environment_binding.get("prefix")
        == identity.get("prefix"),
        "thread_environment": thread_environment
        == {name: "1" for name in THREAD_LIMIT_ENVIRONMENT},
        "threadpool_threads": isinstance(threadpools, list)
        and all(pool.get("num_threads") == 1 for pool in threadpools),
        "threadpool_prefix": isinstance(threadpools, list)
        and all(pool.get("within_prefix") is True for pool in threadpools),
        "python_no_user_site_environment": user_site_isolation.get("PYTHONNOUSERSITE")
        == "1",
        "python_no_user_site_flag": user_site_isolation.get("no_user_site") is True,
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise RuntimeError("Release runtime provenance failed: " + ", ".join(failed))


def _stable_runtime_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize separately validated lazy runtime state for exact restart."""

    stable = _json_value(identity)
    thread_limits = stable.get("thread_limits")
    if isinstance(thread_limits, dict):
        thread_limits["threadpools"] = "validated_separately"
    ownership = stable.get("import_code_ownership")
    if isinstance(ownership, dict):
        ownership["generated_pyc_path_count"] = "validated_separately"
        ownership["generated_pyc_paths_sha256"] = "validated_separately"
    return stable


def preflight_inputs(
    config: ValidationConfig,
    *,
    policy: ReleasePolicy = STRICT_POLICY,
    repository: Path | None = None,
    require_runtime: bool = True,
) -> tuple[InputSnapshot, dict[str, Any]]:
    """Authenticate every input before any validation output is created."""

    _validate_release_config(config, policy)
    strict = not config.development
    repository = (
        Path(__file__).resolve().parents[1]
        if repository is None
        else repository.resolve()
    )
    if config.output_dir.exists():
        if config.output_dir.is_symlink() or not config.output_dir.is_dir():
            raise ValueError("Existing output_dir must be a regular directory.")
        if not config.resume:
            raise FileExistsError(config.output_dir)
        _validate_existing_output_tree(config)
    try:
        config.output_dir.resolve().relative_to(repository)
    except ValueError:
        pass
    else:
        if strict:
            raise ValueError(
                "Release output must be outside the clean source repository."
            )

    paths = {
        "crystal": _regular_file(config.crystal_yaml, name="crystal_yaml"),
        "manifest": _regular_file(
            config.v0_2_release_manifest, name="v0_2_release_manifest"
        ),
        "environment_lock": _regular_file(
            config.environment_lock, name="environment_lock"
        ),
    }
    artifact_root = config.artifact_root.expanduser()
    if artifact_root.is_symlink():
        raise ValueError("artifact_root must not be a symbolic link.")
    artifact_root = artifact_root.resolve(strict=True)
    if not artifact_root.is_dir():
        raise ValueError("artifact_root must be a directory.")
    paths["fc2"] = _regular_file(
        artifact_root / "force-constants/fc2.compact.hdf5", name="fc2"
    )
    paths["fc3"] = _regular_file(
        artifact_root / "force-constants/fc3.compact.hdf5", name="fc3"
    )
    source_hashes = {name: _sha256(path) for name, path in paths.items()}
    if strict:
        for name in ("crystal", "fc2", "fc3"):
            if source_hashes[name] != policy.source_sha256[name]:
                raise ValueError(f"Pinned {name} SHA256 changed.")
        if source_hashes["environment_lock"] != policy.environment_lock_sha256:
            raise ValueError("Environment lock SHA256 does not match the release pin.")
    sealed = _validate_sealed_manifest(
        paths["manifest"],
        expected_sha256=policy.manifest_sha256,
        strict_release=strict,
    )

    archive_paths: dict[int, Path] = {}
    archive_hashes: dict[int, str] = {}
    evidence: dict[int, Mapping[str, Any]] = {}
    for mesh in config.meshes:
        archive = _regular_file(
            artifact_root
            / f"target-q/mesh-{mesh}x{mesh}x{mesh}/merged/bulk-target-q.h5",
            name=f"mesh-{mesh} archive",
        )
        if mesh not in policy.archives:
            raise ValueError(f"Mesh {mesh} has no authenticated archive policy.")
        digest = _sha256(archive)
        if digest != policy.archives[mesh]["file_sha256"]:
            raise ValueError(f"Mesh {mesh} whole-file SHA256 changed.")
        archive_paths[mesh] = archive
        archive_hashes[mesh] = digest
        evidence[mesh] = _validate_archive(
            archive,
            mesh=mesh,
            expected=policy.archives[mesh],
            temperatures_k=config.temperatures_k,
            sigma_thz=config.sigma_thz,
        )

    parent = config.output_dir.expanduser().resolve().parent
    if not parent.is_dir():
        raise ValueError("output_dir parent must already exist and be a directory.")
    if shutil.disk_usage(parent).free < MINIMUM_OUTPUT_FREE_BYTES:
        raise RuntimeError("Insufficient free space for atomic validation outputs.")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".negfpy-write-test.", dir=parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(b"negfpy-write-preflight\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        temporary.unlink(missing_ok=True)

    runtime = _runtime_identity(repository) if require_runtime else {}
    if strict and require_runtime:
        runtime["environment_binding"] = _validate_conda_prefix_against_lock(
            paths["environment_lock"], Path(sys.prefix)
        )
    if strict and require_runtime:
        _require_release_runtime(runtime, policy=policy)
    return (
        InputSnapshot(
            paths=paths,
            sha256=source_hashes,
            archive_paths=archive_paths,
            archive_sha256=archive_hashes,
            archive_evidence=evidence,
            sealed_manifest=sealed,
        ),
        runtime,
    )


def _campaign_document(
    config: ValidationConfig,
    snapshot: InputSnapshot,
    runtime: Mapping[str, Any],
    staged: StagedScientificInputs | None = None,
) -> dict[str, Any]:
    return {
        "schema": CAMPAIGN_SCHEMA,
        "mode": "development" if config.development else "release",
        "release_certification_possible": not config.development,
        "settings": {
            "meshes": config.meshes,
            "end_to_end_meshes": config.end_to_end_meshes,
            "temperatures_k": config.temperatures_k,
            "sigma_thz": config.sigma_thz,
            "cutoff_frequency_thz": CUTOFF_THZ,
            "frequency_factor_to_thz": FREQUENCY_FACTOR_TO_THZ,
        },
        "inputs": {
            "source_sha256": snapshot.sha256,
            "archive_sha256": snapshot.archive_sha256,
            "archive_evidence": snapshot.archive_evidence,
            "sealed_manifest_sha256": snapshot.sha256["manifest"],
            "staged_manifest_sha256": (
                None if staged is None else staged.manifest_sha256
            ),
            "staged_source_sha256": None if staged is None else staged.sha256,
            "staged_rate_sha256": (None if staged is None else staged.archive_sha256),
        },
        "runtime": _stable_runtime_identity(runtime),
        "tasks": [
            {
                "task_id": f"mesh-{mesh}x{mesh}x{mesh}",
                "mesh": (mesh, mesh, mesh),
                "requires_end_to_end": mesh in config.end_to_end_meshes,
                "dependencies": (),
            }
            for mesh in config.meshes
        ],
    }


def _publish_campaign_manifest(
    output_dir: Path,
    document: Mapping[str, Any],
    *,
    resume: bool,
) -> tuple[Path, str]:
    payload = (
        json.dumps(_json_value(document), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    path = output_dir / "campaign-manifest.json"
    if path.exists():
        if not resume or path.is_symlink() or not path.is_file():
            raise FileExistsError(path)
        if path.read_bytes() != payload:
            raise ValueError("Existing campaign manifest differs from exact inputs.")
    else:
        _write_bytes_atomic(path, payload)
    return path, digest


def _validate_existing_output_tree(config: ValidationConfig) -> None:
    """Reject unsafe or unknown resume content before any staged write/compute."""

    output_dir = config.output_dir
    if not output_dir.exists():
        return
    allowed_directories = {"staged-inputs", "checkpoints"}
    allowed_files = {
        "campaign-manifest.json",
        "release-evidence.json",
        "bulk-rta-results.hdf5",
        "bulk-rta-validation.json",
        "SHA256SUMS.json",
        "release-completion-receipt.json",
        "staged-inputs/crystal.yaml",
        "staged-inputs/fc2.compact.hdf5",
        "staged-inputs/fc3.compact.hdf5",
        "staged-inputs/manifest.json",
    }
    invalid_checkpoint_pattern = re.compile(
        r"^checkpoints/mesh-[0-9]{3}/"
        r"(?:core|end-to-end)\.json\.invalid-[0-9a-f]{12}$"
    )
    for mesh in config.meshes:
        checkpoint = f"checkpoints/mesh-{mesh:03d}"
        allowed_directories.add(checkpoint)
        allowed_files.add(f"staged-inputs/mesh-{mesh:03d}-rate-fields.hdf5")
        allowed_files.update(
            {
                f"{checkpoint}/core.json",
                f"{checkpoint}/rates.hdf5",
                f"{checkpoint}/rta.hdf5",
                f"{checkpoint}/harmonic.hdf5",
                f"{checkpoint}/core-reference.hdf5",
                f"{checkpoint}/complete.json",
            }
        )
        if mesh in config.end_to_end_meshes:
            allowed_files.update(
                {
                    f"{checkpoint}/end-to-end.json",
                    f"{checkpoint}/end-to-end-reference.hdf5",
                }
            )
    for path in output_dir.rglob("*"):
        relative = str(path.relative_to(output_dir))
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"Resume artifact tree contains symlink {path}.")
        if stat.S_ISDIR(mode):
            if relative not in allowed_directories:
                raise ValueError(
                    f"Resume artifact tree contains unknown directory {relative!r}."
                )
        elif stat.S_ISREG(mode):
            if (
                relative not in allowed_files
                and invalid_checkpoint_pattern.fullmatch(relative) is None
            ):
                raise ValueError(
                    f"Resume artifact tree contains unknown file {relative!r}."
                )
        else:
            raise ValueError(f"Resume artifact tree contains special file {path}.")


def _validate_internal_hdf5_file(path: Path, *, name: str) -> None:
    source = _regular_file(path, name=name)
    before = _sha256(source)
    try:
        with h5py.File(source, "r") as handle:
            _internal_hdf5_dataset_paths(
                handle,
                require_empty_group_attributes=False,
            )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a valid internal-only HDF5 file.") from exc
    if _sha256(source) != before:
        raise RuntimeError(f"{name} changed while it was audited.")


def _validate_derived_checkpoint_identity(
    path: Path,
    *,
    stage: str,
    mesh: int,
    campaign_sha256: str,
) -> None:
    record = _read_json_object(
        _regular_file(path, name=f"mesh {mesh} {stage} checkpoint"),
        name=f"mesh {mesh} {stage} checkpoint",
    )
    expected = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": stage,
        "mesh": [mesh, mesh, mesh],
        "campaign_sha256": campaign_sha256,
    }
    if any(record.get(name) != value for name, value in expected.items()):
        raise ValueError(f"Mesh {mesh} {stage} checkpoint identity changed.")


def _preflight_resume_checkpoints(
    config: ValidationConfig,
    snapshot: InputSnapshot,
    *,
    campaign_sha256: str,
) -> None:
    """Audit every existing checkpoint before the first mesh compute starts."""

    from negfpy.bulk import read_rate_spectrum_hdf5, read_rta_result_hdf5

    _validate_existing_output_tree(config)
    for mesh in config.meshes:
        checkpoint_dir = config.output_dir / "checkpoints" / f"mesh-{mesh:03d}"
        if not checkpoint_dir.exists():
            continue
        if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
            raise ValueError(f"Mesh {mesh} checkpoint path is not a regular directory.")
        rate_path = checkpoint_dir / "rates.hdf5"
        rta_path = checkpoint_dir / "rta.hdf5"
        harmonic_path = checkpoint_dir / "harmonic.hdf5"
        reference_path = checkpoint_dir / "core-reference.hdf5"
        e2e_path = checkpoint_dir / "end-to-end-reference.hdf5"
        harmonic_identity = {
            "campaign_sha256": campaign_sha256,
            "archive_sha256": snapshot.archive_sha256[mesh],
            "mesh_json": _canonical_json([mesh, mesh, mesh]),
        }
        if rate_path.exists():
            _validate_internal_hdf5_file(
                rate_path, name=f"mesh {mesh} rate checkpoint"
            )
            read_rate_spectrum_hdf5(rate_path)
        if rta_path.exists():
            _validate_internal_hdf5_file(rta_path, name=f"mesh {mesh} RTA checkpoint")
            read_rta_result_hdf5(rta_path)
        if harmonic_path.exists():
            _read_checkpoint_payload(
                harmonic_path,
                schema=HARMONIC_SCHEMA,
                identity=harmonic_identity,
                dataset_names=HARMONIC_CHECKPOINT_DATASETS,
            )
        if reference_path.exists():
            if not all(path.exists() for path in (rate_path, rta_path, harmonic_path)):
                raise ValueError(
                    f"Mesh {mesh} core reference exists without all dependencies."
                )
            _read_checkpoint_payload(
                reference_path,
                schema=CORE_REFERENCE_SCHEMA,
                identity={
                    **harmonic_identity,
                    "rate_sha256": _sha256(rate_path),
                    "rta_sha256": _sha256(rta_path),
                    "harmonic_sha256": _sha256(harmonic_path),
                },
                dataset_names=CORE_REFERENCE_DATASETS,
            )
        if e2e_path.exists():
            if mesh not in config.end_to_end_meshes or not all(
                path.exists() for path in (rate_path, rta_path)
            ):
                raise ValueError(
                    f"Mesh {mesh} end-to-end reference has invalid dependencies."
                )
            _read_checkpoint_payload(
                e2e_path,
                schema=END_TO_END_PAYLOAD_SCHEMA,
                identity={
                    **harmonic_identity,
                    "sigma_thz": config.sigma_thz,
                    "rate_sha256": _sha256(rate_path),
                    "rta_sha256": _sha256(rta_path),
                },
                dataset_names=END_TO_END_REFERENCE_DATASETS,
            )
        for filename, stage in (
            ("core.json", "core"),
            ("end-to-end.json", "end_to_end"),
        ):
            path = checkpoint_dir / filename
            if path.exists():
                _validate_derived_checkpoint_identity(
                    path,
                    stage=stage,
                    mesh=mesh,
                    campaign_sha256=campaign_sha256,
                )
        complete_path = checkpoint_dir / "complete.json"
        if complete_path.exists():
            _read_json_object(
                _regular_file(complete_path, name=f"mesh {mesh} completion record"),
                name=f"mesh {mesh} completion record",
            )
    for filename, name in (
        ("bulk-rta-validation.json", "provisional report"),
        ("release-evidence.json", "provisional evidence"),
        ("SHA256SUMS.json", "provisional checksum manifest"),
    ):
        path = config.output_dir / filename
        if path.exists():
            _read_json_object(_regular_file(path, name=name), name=name)
    final_hdf5 = config.output_dir / "bulk-rta-results.hdf5"
    if final_hdf5.exists():
        _validate_internal_hdf5_file(final_hdf5, name="provisional final HDF5")


def _ensure_regular_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Required directory {path} is not a regular directory.")
        return
    path.mkdir(parents=False)


def _assert_array_fields_equal(first: Any, second: Any, names: Sequence[str]) -> None:
    for name in names:
        if not np.array_equal(
            np.asarray(getattr(first, name)), np.asarray(getattr(second, name))
        ):
            raise ValueError(f"HDF5 round trip changed field {name!r}.")


def _phono3py_reference_problem(
    crystal: Path,
    fc2: Path,
    fc3: Path,
) -> Any:
    import phono3py

    return phono3py.load(
        phono3py_yaml=crystal,
        fc2_filename=fc2,
        fc3_filename=fc3,
        produce_fc=False,
        is_mesh_symmetry=False,
        is_nac=False,
        log_level=0,
    )


def _initialize_phono_dynamical_matrix(ph3: Any, mesh: int) -> Any:
    ph3.mesh_numbers = [mesh, mesh, mesh]
    ph3.init_phph_interaction()
    if ph3.dynamical_matrix is None:
        raise RuntimeError("phono3py did not initialize its dynamical matrix.")
    return ph3.dynamical_matrix


def _reference_frequencies(ph3: Any, qpoints: np.ndarray) -> np.ndarray:
    dynamical = ph3.dynamical_matrix
    if dynamical is None:
        raise RuntimeError("phono3py dynamical matrix is not initialized.")
    result = np.empty((qpoints.shape[0], 6), dtype=np.float64)
    for position, qpoint in enumerate(qpoints):
        dynamical.run(qpoint)
        eigenvalues = np.linalg.eigvalsh(dynamical.dynamical_matrix)
        result[position] = (
            np.sqrt(np.abs(eigenvalues))
            * np.where(eigenvalues < 0.0, -1.0, 1.0)
            * ph3.unit_conversion_factor
        )
    return result


def _reference_mode_kappa(
    frequencies: np.ndarray,
    velocities: np.ndarray,
    heat_capacity: np.ndarray,
    gamma: np.ndarray,
    volume: float,
) -> np.ndarray:
    from phono3py.conductivity.utils import get_unit_to_WmK

    outer = np.einsum("qbi,qbj->qbij", velocities, velocities)
    mode = np.zeros(gamma.shape + (3, 3), dtype=np.float64)
    valid = (frequencies > CUTOFF_THZ)[None, :, :] & (gamma > 0.0)
    numerator = (
        heat_capacity[:, :, :, None, None]
        * outer[None, :, :, :, :]
        * get_unit_to_WmK()
        / volume
        / frequencies.shape[0]
    )
    mode[valid] = numerator[valid] / (2.0 * gamma[valid][:, None, None])
    return mode


def _voigt(tensor: np.ndarray) -> np.ndarray:
    pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
    return np.stack([tensor[..., first, second] for first, second in pairs], axis=-1)


def _match_qpoints(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    positions: list[int] = []
    for qpoint in source:
        difference = qpoint[None, :] - target
        difference -= np.rint(difference)
        norms = np.linalg.norm(difference, axis=1)
        position = int(np.argmin(norms))
        if norms[position] > 1.0e-12:
            raise ValueError("Could not match a phono3py reference q-point.")
        positions.append(position)
    result = np.asarray(positions, dtype=np.int64)
    if np.unique(result).size != source.shape[0]:
        raise ValueError("Reference q-point mapping is not one-to-one.")
    return result


def _internal_hdf5_dataset_paths(
    group: h5py.Group,
    *,
    require_empty_group_attributes: bool = True,
) -> tuple[str, ...]:
    """Return datasets only after proving every descendant link is internal."""

    paths: list[str] = []

    def walk(parent: h5py.Group, prefix: str) -> None:
        for name in parent:
            link = parent.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise TypeError(
                    f"HDF5 field {(prefix + name)!r} is not an internal hard link."
                )
            value = parent.get(name, getlink=False)
            path = prefix + name
            if isinstance(value, h5py.Dataset):
                paths.append(path)
            elif isinstance(value, h5py.Group):
                if require_empty_group_attributes and {str(key) for key in value.attrs}:
                    raise ValueError(
                        f"HDF5 intermediate group {path!r} has unexpected attributes."
                    )
                walk(value, path + "/")
            else:
                raise TypeError(f"HDF5 field {path!r} has an unsupported object type.")

    walk(group, "")
    return tuple(sorted(paths))


def _checkpoint_payload_sha256(handle: h5py.File) -> str:
    identity = {
        str(name): _json_value(value)
        for name, value in handle.attrs.items()
        if str(name) != "payload_sha256"
    }
    digest = hashlib.sha256()
    digest.update(b"negfpy.bulk-rta-checkpoint.v1\0")
    digest.update(_canonical_json(identity).encode("utf-8"))
    dataset_paths = _internal_hdf5_dataset_paths(handle)
    for name in dataset_paths:
        dataset = handle[name]
        if {str(key) for key in dataset.attrs}:
            raise ValueError(f"Checkpoint field {name!r} has unexpected attributes.")
        encoded_name = name.encode("utf-8")
        encoded_digest = _dataset_sha256(dataset).encode("ascii")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(encoded_digest)
    return digest.hexdigest()


def _write_checkpoint_payload(
    path: Path,
    *,
    schema: str,
    identity: Mapping[str, Any],
    datasets: Mapping[str, Any],
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["schema"] = schema
            for name, value in identity.items():
                handle.attrs[str(name)] = value
            for name, values in datasets.items():
                handle.create_dataset(name, data=values)
            handle.attrs["payload_sha256"] = _checkpoint_payload_sha256(handle)
            handle.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_checkpoint_payload(
    path: Path,
    *,
    schema: str,
    identity: Mapping[str, Any],
    dataset_names: frozenset[str],
) -> dict[str, np.ndarray]:
    source = _regular_file(path, name=f"{schema} checkpoint")
    before = _sha256(source)
    with h5py.File(source, "r") as handle:
        expected_attributes = {"schema", "payload_sha256", *identity}
        if {str(name) for name in handle.attrs} != expected_attributes:
            raise ValueError(f"{schema} checkpoint attributes are not exact.")
        if str(handle.attrs["schema"]) != schema:
            raise ValueError(f"{schema} checkpoint schema changed.")
        for name, expected in identity.items():
            actual = handle.attrs[name]
            if _json_value(actual) != _json_value(expected):
                raise ValueError(f"{schema} checkpoint identity {name!r} changed.")
        paths = _internal_hdf5_dataset_paths(handle)
        if frozenset(paths) != dataset_names:
            raise ValueError(f"{schema} checkpoint dataset schema is not exact.")
        actual_digest = _checkpoint_payload_sha256(handle)
        if str(handle.attrs["payload_sha256"]) != actual_digest:
            raise ValueError(f"{schema} checkpoint payload SHA256 changed.")
        arrays = {name: np.asarray(handle[name]) for name in sorted(dataset_names)}
    if _sha256(source) != before:
        raise RuntimeError(f"{schema} checkpoint changed while it was read.")
    return arrays


def _copy_input_atomic(source: Path, destination: Path, *, sha256: str) -> None:
    """Copy one authenticated input without a mutable hard-link shortcut."""

    if _sha256(source) != sha256:
        raise RuntimeError(f"Input {source} changed before staging.")
    if destination.exists():
        staged = _regular_file(destination, name="staged scientific input")
        if _sha256(staged) != sha256:
            raise ValueError(f"Existing staged input {destination.name!r} changed.")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with (
            source.open("rb") as input_stream,
            os.fdopen(descriptor, "wb") as output_stream,
        ):
            shutil.copyfileobj(input_stream, output_stream, length=8 * 1024 * 1024)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if _sha256(source) != sha256 or _sha256(temporary) != sha256:
            raise RuntimeError(f"Input {source} changed while it was staged.")
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _stage_scientific_inputs(
    config: ValidationConfig,
    snapshot: InputSnapshot,
) -> StagedScientificInputs:
    """Create immutable compact scientific inputs after the full preflight."""

    staged_root = config.output_dir / "staged-inputs"
    if staged_root.exists():
        if staged_root.is_symlink() or not staged_root.is_dir():
            raise ValueError("staged-inputs must be a regular directory.")
        if not config.resume:
            raise FileExistsError(staged_root)
    else:
        staged_root.mkdir(parents=False)
    source_paths: dict[str, Path] = {}
    source_hashes: dict[str, str] = {}
    filenames = {
        "crystal": "crystal.yaml",
        "fc2": "fc2.compact.hdf5",
        "fc3": "fc3.compact.hdf5",
    }
    for name, filename in filenames.items():
        destination = staged_root / filename
        _copy_input_atomic(
            snapshot.paths[name], destination, sha256=snapshot.sha256[name]
        )
        source_paths[name] = destination
        source_hashes[name] = _sha256(destination)

    staged_archives: dict[int, Path] = {}
    staged_archive_hashes: dict[int, str] = {}
    for mesh in config.meshes:
        original = snapshot.archive_paths[mesh]
        original_sha256 = snapshot.archive_sha256[mesh]
        if _sha256(original) != original_sha256:
            raise RuntimeError(f"Mesh {mesh} archive changed before staging.")
        with h5py.File(original, "r") as handle:
            fields = {name: np.asarray(handle[name]) for name in STAGED_RATE_DATASETS}
        if _sha256(original) != original_sha256:
            raise RuntimeError(f"Mesh {mesh} archive changed while staging.")
        destination = staged_root / f"mesh-{mesh:03d}-rate-fields.hdf5"
        identity = {
            "source_archive_sha256": original_sha256,
            "mesh_json": _canonical_json([mesh, mesh, mesh]),
        }
        if destination.exists():
            _read_checkpoint_payload(
                destination,
                schema=STAGED_RATE_SCHEMA,
                identity=identity,
                dataset_names=STAGED_RATE_DATASETS,
            )
        else:
            _write_checkpoint_payload(
                destination,
                schema=STAGED_RATE_SCHEMA,
                identity=identity,
                datasets=fields,
            )
        staged_archives[mesh] = destination
        staged_archive_hashes[mesh] = _sha256(destination)

    for name in filenames:
        if _sha256(snapshot.paths[name]) != snapshot.sha256[name]:
            raise RuntimeError(f"Input {name!r} changed during staging.")
    for mesh, original in snapshot.archive_paths.items():
        if _sha256(original) != snapshot.archive_sha256[mesh]:
            raise RuntimeError(f"Mesh {mesh} archive changed during staging.")

    manifest = {
        "schema": STAGED_INPUT_SCHEMA,
        "source_sha256": {name: snapshot.sha256[name] for name in filenames},
        "staged_source_sha256": source_hashes,
        "source_archive_sha256": snapshot.archive_sha256,
        "staged_rate_sha256": staged_archive_hashes,
        "staged_rate_datasets": sorted(STAGED_RATE_DATASETS),
        "interaction_tensor_copied": False,
    }
    manifest_path = staged_root / "manifest.json"
    payload = (
        json.dumps(_json_value(manifest), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    if manifest_path.exists():
        existing = _regular_file(manifest_path, name="staged input manifest")
        if existing.read_bytes() != payload:
            raise ValueError("Existing staged input manifest changed.")
    else:
        _write_bytes_atomic(manifest_path, payload)
    return StagedScientificInputs(
        paths=source_paths,
        sha256=source_hashes,
        archive_paths=staged_archives,
        archive_sha256=staged_archive_hashes,
        manifest_path=manifest_path,
        manifest_sha256=_sha256(manifest_path),
    )


def _verify_staged_inputs(
    staged: StagedScientificInputs,
    snapshot: InputSnapshot,
    *,
    meshes: Sequence[int],
) -> None:
    for name, path in staged.paths.items():
        if (
            _sha256(path) != staged.sha256[name]
            or staged.sha256[name] != snapshot.sha256[name]
        ):
            raise RuntimeError(f"Staged input {name!r} changed during validation.")
    for mesh in meshes:
        path = staged.archive_paths[mesh]
        if _sha256(path) != staged.archive_sha256[mesh]:
            raise RuntimeError(f"Staged mesh {mesh} rate payload changed.")
        _read_staged_rate_fields(
            path,
            mesh=mesh,
            source_archive_sha256=snapshot.archive_sha256[mesh],
        )
    if _sha256(staged.manifest_path) != staged.manifest_sha256:
        raise RuntimeError("Staged input manifest changed during validation.")


def _read_staged_rate_fields(
    path: Path,
    *,
    mesh: int,
    source_archive_sha256: str,
) -> dict[str, np.ndarray | float]:
    operands = _read_checkpoint_payload(
        path,
        schema=STAGED_RATE_SCHEMA,
        identity={
            "source_archive_sha256": source_archive_sha256,
            "mesh_json": _canonical_json([mesh, mesh, mesh]),
        },
        dataset_names=STAGED_RATE_DATASETS,
    )
    return {
        "temperatures": operands["linewidths/temperatures"],
        "qpoints": operands["phonons/qpoints"],
        "dynamical_matrices": operands["phonons/dynamical_matrices"],
        "frequencies": operands["phonons/frequencies"],
        "eigenvectors": operands["phonons/eigenvectors"],
        "frequency_factor": float(
            np.asarray(operands["phonons/frequency_factor_to_thz"])
        ),
        "grid_points": operands["linewidths/grid_points"],
        "band_indices": operands["linewidths/band_indices"],
        "gamma": operands["linewidths/gamma_thz_hwhm"],
        "gamma_normal": operands["linewidths/gamma_normal_thz_hwhm"],
        "gamma_umklapp": operands["linewidths/gamma_umklapp_thz_hwhm"],
        "gamma_angular": operands["linewidths/gamma_angular_fwhm"],
        "decay_phase_space": operands["linewidths/decay_phase_space"],
        "difference_phase_space": operands["linewidths/difference_phase_space"],
    }


def _publish_derived_checkpoint(
    path: Path,
    record: Mapping[str, Any],
    *,
    stage: str,
    mesh: int,
    campaign_sha256: str,
) -> dict[str, Any]:
    """Validate checkpoint identity, then replace all derived scientific fields."""

    if path.exists():
        source = _regular_file(path, name=f"mesh {mesh} {stage} checkpoint")
        try:
            existing = _read_json_object(source, name=f"mesh {mesh} {stage} checkpoint")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            archived = path.with_name(f"{path.name}.invalid-{_sha256(path)[:12]}")
            os.replace(path, archived)
            _fsync_directory(path.parent)
        else:
            identity_fields = {
                "schema": CHECKPOINT_SCHEMA,
                "stage": stage,
                "mesh": [mesh, mesh, mesh],
                "campaign_sha256": campaign_sha256,
            }
            present = {name for name in identity_fields if name in existing}
            if present != set(identity_fields):
                archived = path.with_name(f"{path.name}.invalid-{_sha256(path)[:12]}")
                os.replace(path, archived)
                _fsync_directory(path.parent)
            elif any(
                existing[name] != expected for name, expected in identity_fields.items()
            ):
                raise ValueError(
                    f"Mesh {mesh} {stage} checkpoint belongs to another identity."
                )
    _write_json_atomic(path, record)
    return _read_json_object(path, name=f"mesh {mesh} {stage} checkpoint")


def _write_harmonic_archive(
    path: Path,
    *,
    band: Any,
    dos: Any,
    mesh_frequencies: np.ndarray,
    group_velocities: np.ndarray,
    reference_mesh_frequencies: np.ndarray,
    reference_group_velocities: np.ndarray,
    reference_band_frequencies: np.ndarray,
    reference_dos_total: np.ndarray,
    campaign_sha256: str,
    archive_sha256: str,
    mesh: int,
) -> None:
    _write_checkpoint_payload(
        path,
        schema=HARMONIC_SCHEMA,
        identity={
            "campaign_sha256": campaign_sha256,
            "archive_sha256": archive_sha256,
            "mesh_json": _canonical_json([mesh, mesh, mesh]),
        },
        datasets={
            "band/qpoints": band.path.qpoints,
            "band/distances_inv_angstrom": band.path.distances_inv_angstrom,
            "band/vertex_indices": band.path.vertex_indices,
            "band/frequencies_thz": band.frequencies_thz,
            "dos/frequency_points_thz": dos.frequency_points_thz,
            "dos/total_states_per_thz_cell": dos.total_states_per_thz_cell,
            "dos/projected_states_per_thz_cell": dos.projected_states_per_thz_cell,
            "mesh/frequencies_thz": mesh_frequencies,
            "mesh/group_velocities_thz_angstrom": group_velocities,
            "reference/mesh_frequencies_thz": reference_mesh_frequencies,
            "reference/group_velocities_thz_angstrom": (reference_group_velocities),
            "reference/band_frequencies_thz": reference_band_frequencies,
            "reference/dos_total_states_per_thz_cell": reference_dos_total,
        },
    )


def _apply_acoustic_zero_convention(values: Any) -> np.ndarray:
    frequencies = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("Harmonic frequencies must be finite.")
    frequencies[np.abs(frequencies) <= CUTOFF_THZ] = 0.0
    return frequencies


def _apply_dos_stability_convention(values: Any) -> np.ndarray:
    frequencies = np.array(values, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("DOS frequencies must be finite.")
    if np.any(frequencies < -CUTOFF_THZ):
        minimum = float(np.min(frequencies))
        raise ValueError(
            "DOS reference rejects a dynamically unstable spectrum; "
            f"minimum frequency is {minimum:.16g} THz."
        )
    frequencies[(frequencies < 0.0) & (frequencies >= -CUTOFF_THZ)] = 0.0
    return frequencies


def _harmonic_validation(
    problem: Any,
    archived: Mapping[str, Any],
    ph3: Any,
    *,
    mesh: int,
    output_path: Path | None,
    campaign_sha256: str,
    archive_sha256: str,
) -> tuple[dict[str, Any], Any, Any]:
    from phonopy.phonon.group_velocity import GroupVelocity

    from negfpy.bulk import (
        compute_bulk_group_velocities,
        compute_phonon_band_structure,
        compute_phonon_dos,
    )

    harmonic = problem.harmonic
    spectrum = harmonic.compute_uniform_mesh_phonons((mesh, mesh, mesh))
    archived_frequencies = np.asarray(archived["frequencies"])
    velocities = compute_bulk_group_velocities(
        harmonic.fc2,
        harmonic.mapping,
        spectrum.qpoints,
        frequency_factor_to_thz=FREQUENCY_FACTOR_TO_THZ,
        cutoff_frequency_thz=CUTOFF_THZ,
        symmetrize=True,
    )
    reference = GroupVelocity(
        ph3.dynamical_matrix
        if ph3.dynamical_matrix is not None
        else _initialize_phono_dynamical_matrix(ph3, mesh),
        symmetry=ph3.primitive_symmetry,
        frequency_factor_to_THz=ph3.unit_conversion_factor,
        cutoff_frequency=CUTOFF_THZ,
    )
    reference.run(spectrum.qpoints)
    reference_velocities = np.asarray(reference.group_velocities)

    vertices = np.asarray(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.5], [0.0, 0.0, 0.0]]
    )
    band = compute_phonon_band_structure(
        harmonic.fc2,
        harmonic.mapping,
        vertices,
        points_per_segment=17,
        labels=("Γ", "X", "L", "Γ"),
        frequency_factor_to_thz=FREQUENCY_FACTOR_TO_THZ,
    )
    band_reference = _reference_frequencies(ph3, band.path.qpoints)

    dos = compute_phonon_dos(
        spectrum,
        sigma_thz=RELEASE_SIGMA_THZ,
        n_frequency_points=2001,
        normalization_tolerance=DOS_NORMALIZATION_ATOL,
    )
    points = dos.frequency_points_thz
    frequencies = _apply_dos_stability_convention(spectrum.frequencies)
    kernel = np.exp(
        -0.5 * ((points[None, None, :] - frequencies[:, :, None]) / dos.sigma_thz) ** 2
    ) / (np.sqrt(2.0 * np.pi) * dos.sigma_thz)
    dos_reference = np.mean(np.sum(kernel, axis=1), axis=0)
    operands: Mapping[str, np.ndarray] = {
        "band/qpoints": np.asarray(band.path.qpoints),
        "band/distances_inv_angstrom": np.asarray(band.path.distances_inv_angstrom),
        "band/vertex_indices": np.asarray(band.path.vertex_indices),
        "band/frequencies_thz": np.asarray(band.frequencies_thz),
        "dos/frequency_points_thz": np.asarray(dos.frequency_points_thz),
        "dos/total_states_per_thz_cell": np.asarray(dos.total_states_per_thz_cell),
        "dos/projected_states_per_thz_cell": np.asarray(
            dos.projected_states_per_thz_cell
        ),
        "mesh/frequencies_thz": np.asarray(spectrum.frequencies),
        "mesh/group_velocities_thz_angstrom": np.asarray(
            velocities.values_thz_angstrom
        ),
        "reference/mesh_frequencies_thz": archived_frequencies,
        "reference/group_velocities_thz_angstrom": reference_velocities,
        "reference/band_frequencies_thz": band_reference,
        "reference/dos_total_states_per_thz_cell": dos_reference,
    }
    if output_path is not None:
        _write_harmonic_archive(
            output_path,
            band=band,
            dos=dos,
            mesh_frequencies=spectrum.frequencies,
            group_velocities=velocities.values_thz_angstrom,
            reference_mesh_frequencies=archived_frequencies,
            reference_group_velocities=reference_velocities,
            reference_band_frequencies=band_reference,
            reference_dos_total=dos_reference,
            campaign_sha256=campaign_sha256,
            archive_sha256=archive_sha256,
            mesh=mesh,
        )
        operands = _read_checkpoint_payload(
            output_path,
            schema=HARMONIC_SCHEMA,
            identity={
                "campaign_sha256": campaign_sha256,
                "archive_sha256": archive_sha256,
                "mesh_json": _canonical_json([mesh, mesh, mesh]),
            },
            dataset_names=HARMONIC_CHECKPOINT_DATASETS,
        )
    record = _derive_harmonic_record(problem, operands)
    return record, velocities, reference_velocities


def _derive_harmonic_record(
    problem: Any,
    operands: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    mesh_frequencies = np.asarray(operands["mesh/frequencies_thz"])
    reference_mesh_frequencies = np.asarray(operands["reference/mesh_frequencies_thz"])
    raw_frequency_error = float(
        np.max(np.abs(mesh_frequencies - reference_mesh_frequencies))
    )
    frequency_error = float(
        np.max(
            np.abs(
                _apply_acoustic_zero_convention(mesh_frequencies)
                - _apply_acoustic_zero_convention(reference_mesh_frequencies)
            )
        )
    )
    velocity_error = float(
        np.max(
            np.abs(
                operands["mesh/group_velocities_thz_angstrom"]
                - operands["reference/group_velocities_thz_angstrom"]
            )
        )
    )
    band_frequencies = np.asarray(operands["band/frequencies_thz"])
    reference_band_frequencies = np.asarray(operands["reference/band_frequencies_thz"])
    raw_band_error = float(
        np.max(np.abs(band_frequencies - reference_band_frequencies))
    )
    band_error = float(
        np.max(
            np.abs(
                _apply_acoustic_zero_convention(band_frequencies)
                - _apply_acoustic_zero_convention(reference_band_frequencies)
            )
        )
    )
    dos_error = float(
        np.max(
            np.abs(
                operands["dos/total_states_per_thz_cell"]
                - operands["reference/dos_total_states_per_thz_cell"]
            )
        )
    )
    projected_error = float(
        np.max(
            np.abs(
                np.sum(operands["dos/projected_states_per_thz_cell"], axis=(0, 1))
                - operands["dos/total_states_per_thz_cell"]
            )
        )
    )
    integral = float(
        np.trapezoid(
            operands["dos/total_states_per_thz_cell"],
            operands["dos/frequency_points_thz"],
        )
    )
    expected_states = int(operands["mesh/frequencies_thz"].shape[1])
    normalization_error = abs(integral - expected_states)
    normalization_certified = normalization_error <= DOS_NORMALIZATION_ATOL
    passed = bool(
        frequency_error <= FREQUENCY_ATOL_THZ
        and velocity_error <= GV_ATOL_THZ_ANGSTROM
        and band_error <= FREQUENCY_ATOL_THZ
        and dos_error <= FIELD_ATOL
        and projected_error <= FIELD_ATOL
        and normalization_certified
        and problem.fc2_audit.passed
        and problem.fc2_physical_audit.passed
    )
    return {
        "acoustic_zero_convention_cutoff_thz": CUTOFF_THZ,
        "raw_uniform_mesh_frequency_max_abs_error_thz": raw_frequency_error,
        "uniform_mesh_frequency_max_abs_error_thz": frequency_error,
        "group_velocity_max_abs_error_thz_angstrom": velocity_error,
        "raw_band_frequency_max_abs_error_thz": raw_band_error,
        "band_frequency_max_abs_error_thz": band_error,
        "dos_max_abs_error_states_per_thz_cell": dos_error,
        "dos_projection_sum_max_abs_error": projected_error,
        "dos_integral_states_per_cell": integral,
        "dos_expected_states_per_cell": expected_states,
        "dos_normalization_error": normalization_error,
        "dos_normalization_certified": normalization_certified,
        "fc2_schema_audit_passed": problem.fc2_audit.passed,
        "fc2_physical_audit_passed": problem.fc2_physical_audit.passed,
        "passed": passed,
    }


def _derive_core_record(
    config: ValidationConfig,
    snapshot: InputSnapshot,
    problem: Any,
    rates: Any,
    result: Any,
    harmonic_operands: Mapping[str, np.ndarray],
    reference_operands: Mapping[str, np.ndarray],
    *,
    mesh: int,
    campaign_sha256: str,
    rate_path: Path,
    rta_path: Path,
    harmonic_path: Path,
    reference_path: Path,
) -> dict[str, Any]:
    for rate_name, result_name in (
        ("temperatures", "temperatures"),
        ("qpoints", "qpoints"),
        ("gamma_thz_hwhm", "gamma_thz_hwhm"),
    ):
        if not np.array_equal(
            np.asarray(getattr(rates, rate_name)),
            np.asarray(getattr(result, result_name)),
        ):
            raise ValueError(
                f"Mesh {mesh} rate and RTA checkpoint field {rate_name!r} differ."
            )
    frequency_discrepancy = float(
        np.max(
            np.abs(
                np.asarray(rates.frequencies_thz) - np.asarray(result.frequencies_thz)
            )
        )
    )
    frequency_consistent = bool(frequency_discrepancy <= FREQUENCY_ATOL_THZ)
    harmonic = _derive_harmonic_record(problem, harmonic_operands)
    reference_cv = reference_operands["heat_capacities_ev_per_k"]
    reference_mode = reference_operands["mode_kappa_w_mk"]
    reference_kappa = reference_operands["kappa_w_mk"]
    cv_error = float(np.max(np.abs(result.heat_capacities_ev_per_k - reference_cv)))
    mode_equal = bool(
        np.allclose(
            result.mode_kappa_w_mk,
            reference_mode,
            rtol=FIELD_RTOL,
            atol=FIELD_ATOL,
        )
    )
    diagonal_equal = bool(
        np.allclose(
            np.diagonal(result.kappa_w_mk, axis1=1, axis2=2),
            np.diagonal(reference_kappa, axis1=1, axis2=2),
            rtol=FIELD_RTOL,
            atol=FIELD_ATOL,
        )
    )
    offdiagonal = bool(
        np.max(np.abs(result.kappa_voigt_w_mk[:, 3:])) <= KAPPA_OFFDIAGONAL_ATOL_W_MK
        and np.max(np.abs(_voigt(reference_kappa)[:, 3:]))
        <= KAPPA_OFFDIAGONAL_ATOL_W_MK
    )
    passed = bool(
        harmonic["passed"]
        and frequency_consistent
        and cv_error <= CV_ATOL_EV_K
        and mode_equal
        and diagonal_equal
        and offdiagonal
        and (
            config.development
            or result.metadata.get("code_and_input_provenance_eligible") is True
        )
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "core",
        "mesh": (mesh, mesh, mesh),
        "campaign_sha256": campaign_sha256,
        "archive_sha256": snapshot.archive_sha256[mesh],
        "rate_sha256": _sha256(rate_path),
        "rta_sha256": _sha256(rta_path),
        "harmonic_sha256": _sha256(harmonic_path),
        "core_reference_sha256": _sha256(reference_path),
        "harmonic": harmonic,
        "rate_rta_frequency_max_abs_error_thz": frequency_discrepancy,
        "rate_rta_frequency_within_tolerance": frequency_consistent,
        "maximum_heat_capacity_absolute_error_ev_per_k": cv_error,
        "mode_kappa_equal": mode_equal,
        "total_kappa_diagonal_equal": diagonal_equal,
        "total_kappa_offdiagonal_below_absolute_gate": offdiagonal,
        "negfpy_kappa_voigt_w_mk": result.kappa_voigt_w_mk,
        "reference_kappa_voigt_w_mk": _voigt(reference_kappa),
        "code_and_input_provenance_eligible": result.metadata.get(
            "code_and_input_provenance_eligible"
        ),
        "passed": passed,
    }


def _make_rate_spectrum(
    problem: Any,
    archived: Mapping[str, Any],
    *,
    archive_sha256: str,
) -> Any:
    from negfpy.anharmonic._bulk_identity import (
        bulk_three_phonon_physics_sha256,
    )
    from negfpy.bulk import BulkThreePhononRateSpectrum

    return BulkThreePhononRateSpectrum(
        temperatures=archived["temperatures"],
        qpoints=archived["qpoints"],
        frequencies_thz=archived["frequencies"],
        grid_points=archived["grid_points"],
        band_indices=archived["band_indices"],
        gamma_thz_hwhm=archived["gamma"],
        gamma_normal_thz_hwhm=archived["gamma_normal"],
        gamma_umklapp_thz_hwhm=archived["gamma_umklapp"],
        gamma_angular_fwhm=archived["gamma_angular"],
        decay_phase_space=archived["decay_phase_space"],
        difference_phase_space=archived["difference_phase_space"],
        cutoff_frequency_thz=CUTOFF_THZ,
        frequency_factor_to_thz=float(archived["frequency_factor"]),
        physics_input_sha256=bulk_three_phonon_physics_sha256(problem.three_phonon),
        metadata={
            "schema": "negfpy.bulk_rta_release_rate_input.v1",
            "sealed_v0_2_archive_sha256": archive_sha256,
        },
    )


def _compute_core_stage(
    config: ValidationConfig,
    snapshot: InputSnapshot,
    *,
    mesh: int,
    checkpoint_dir: Path,
    campaign_sha256: str,
) -> dict[str, Any]:
    from phonopy.phonon.thermal_properties import mode_cv
    from phonopy.physical_units import get_physical_units

    from negfpy.bulk import (
        BulkTransportProblem,
        compute_rta_thermal_conductivity,
        read_rate_spectrum_hdf5,
        read_rta_result_hdf5,
        write_rate_spectrum_hdf5,
        write_rta_result_hdf5,
    )

    record_path = checkpoint_dir / "core.json"
    rate_path = checkpoint_dir / "rates.hdf5"
    rta_path = checkpoint_dir / "rta.hdf5"
    harmonic_path = checkpoint_dir / "harmonic.hdf5"
    reference_path = checkpoint_dir / "core-reference.hdf5"
    problem = BulkTransportProblem.from_phono3py(
        snapshot.paths["crystal"],
        snapshot.paths["fc2"],
        snapshot.paths["fc3"],
        mesh=(mesh, mesh, mesh),
        sigma_thz=config.sigma_thz,
        cutoff_frequency_thz=CUTOFF_THZ,
        frequency_factor_to_thz=FREQUENCY_FACTOR_TO_THZ,
        make_r0_average=True,
        metadata={"release_validation": SCHEMA},
    )
    archived = _read_staged_rate_fields(
        snapshot.archive_paths[mesh],
        mesh=mesh,
        source_archive_sha256=snapshot.archive_sha256[mesh],
    )
    expected_rates = _make_rate_spectrum(
        problem,
        archived,
        archive_sha256=snapshot.archive_sha256[mesh],
    )
    harmonic_identity = {
        "campaign_sha256": campaign_sha256,
        "archive_sha256": snapshot.archive_sha256[mesh],
        "mesh_json": _canonical_json([mesh, mesh, mesh]),
    }
    rta_fields = (
        "temperatures",
        "qpoints",
        "frequencies_thz",
        "group_velocities_thz_angstrom",
        "heat_capacities_ev_per_k",
        "gamma_thz_hwhm",
        "population_lifetime_ps",
        "mean_free_path_angstrom",
        "mode_kappa_w_mk",
        "band_index_kappa_w_mk",
        "kappa_w_mk",
        "qpoint_weights",
        "included_mask",
    )
    rate_fields = (
        "temperatures",
        "qpoints",
        "frequencies_thz",
        "grid_points",
        "band_indices",
        "gamma_thz_hwhm",
        "gamma_normal_thz_hwhm",
        "gamma_umklapp_thz_hwhm",
        "gamma_angular_fwhm",
        "decay_phase_space",
        "difference_phase_space",
    )
    rates = expected_rates
    if rate_path.exists():
        stored_rates = read_rate_spectrum_hdf5(rate_path)
    else:
        write_rate_spectrum_hdf5(rate_path, rates)
        stored_rates = read_rate_spectrum_hdf5(rate_path)
    _assert_array_fields_equal(
        rates,
        stored_rates,
        rate_fields,
    )
    computed_result = compute_rta_thermal_conductivity(
        problem,
        temperatures=config.temperatures_k,
        linewidths=stored_rates,
    )
    if rta_path.exists():
        stored_result = read_rta_result_hdf5(rta_path)
    else:
        write_rta_result_hdf5(rta_path, computed_result)
        stored_result = read_rta_result_hdf5(rta_path)
    _assert_array_fields_equal(computed_result, stored_result, rta_fields)

    fresh_harmonic: Mapping[str, Any] | None = None
    if harmonic_path.exists():
        harmonic_operands = _read_checkpoint_payload(
            harmonic_path,
            schema=HARMONIC_SCHEMA,
            identity=harmonic_identity,
            dataset_names=HARMONIC_CHECKPOINT_DATASETS,
        )
    else:
        ph3 = _phono3py_reference_problem(
            snapshot.paths["crystal"],
            snapshot.paths["fc2"],
            snapshot.paths["fc3"],
        )
        _initialize_phono_dynamical_matrix(ph3, mesh)
        fresh_harmonic, _velocities, _reference_velocities = _harmonic_validation(
            problem,
            archived,
            ph3,
            mesh=mesh,
            output_path=harmonic_path,
            campaign_sha256=campaign_sha256,
            archive_sha256=snapshot.archive_sha256[mesh],
        )
        harmonic_operands = _read_checkpoint_payload(
            harmonic_path,
            schema=HARMONIC_SCHEMA,
            identity=harmonic_identity,
            dataset_names=HARMONIC_CHECKPOINT_DATASETS,
        )

    reference_identity = {
        "campaign_sha256": campaign_sha256,
        "archive_sha256": snapshot.archive_sha256[mesh],
        "mesh_json": _canonical_json([mesh, mesh, mesh]),
        "rate_sha256": _sha256(rate_path),
        "rta_sha256": _sha256(rta_path),
        "harmonic_sha256": _sha256(harmonic_path),
    }
    if reference_path.exists():
        reference_operands = _read_checkpoint_payload(
            reference_path,
            schema=CORE_REFERENCE_SCHEMA,
            identity=reference_identity,
            dataset_names=CORE_REFERENCE_DATASETS,
        )
    else:
        reference_cv = np.asarray(
            mode_cv(
                np.asarray(config.temperatures_k),
                (
                    np.asarray(archived["frequencies"])
                    * get_physical_units().THzToEv
                ).ravel(),
            )
        ).reshape(
            len(config.temperatures_k),
            *np.asarray(archived["frequencies"]).shape,
        )
        reference_cv[:, np.asarray(archived["frequencies"]) <= CUTOFF_THZ] = 0.0
        reference_mode = _reference_mode_kappa(
            np.asarray(archived["frequencies"]),
            np.asarray(
                harmonic_operands["reference/group_velocities_thz_angstrom"]
            ),
            reference_cv,
            np.asarray(archived["gamma"]),
            problem.primitive_volume_angstrom3,
        )
        _write_checkpoint_payload(
            reference_path,
            schema=CORE_REFERENCE_SCHEMA,
            identity=reference_identity,
            datasets={
                "heat_capacities_ev_per_k": reference_cv,
                "mode_kappa_w_mk": reference_mode,
                "kappa_w_mk": np.sum(reference_mode, axis=(1, 2)),
            },
        )
        reference_operands = _read_checkpoint_payload(
            reference_path,
            schema=CORE_REFERENCE_SCHEMA,
            identity=reference_identity,
            dataset_names=CORE_REFERENCE_DATASETS,
        )
    derived = _derive_core_record(
        config,
        snapshot,
        problem,
        stored_rates,
        stored_result,
        harmonic_operands,
        reference_operands,
        mesh=mesh,
        campaign_sha256=campaign_sha256,
        rate_path=rate_path,
        rta_path=rta_path,
        harmonic_path=harmonic_path,
        reference_path=reference_path,
    )
    if (
        fresh_harmonic is not None
        and derived["harmonic"] != _json_value(fresh_harmonic)
    ):
        raise ValueError("Fresh harmonic metrics disagree with stored operands.")
    return _publish_derived_checkpoint(
        record_path,
        derived,
        stage="core",
        mesh=mesh,
        campaign_sha256=campaign_sha256,
    )


def _derive_end_to_end_record(
    rates: Any,
    result: Any,
    operands: Mapping[str, np.ndarray],
    *,
    mesh: int,
    campaign_sha256: str,
    payload_path: Path,
) -> dict[str, Any]:
    selectors_equal = bool(
        np.array_equal(operands["temperatures"], rates.temperatures)
        and np.array_equal(operands["grid_points"], rates.grid_points)
        and np.array_equal(operands["band_indices"], rates.band_indices)
        and np.array_equal(operands["qpoints"], rates.qpoints)
    )
    reference_gamma = operands["gamma_thz_hwhm"]
    reference_kappa = operands["kappa_voigt_w_mk"]
    gamma_equal = bool(
        selectors_equal
        and np.allclose(
            rates.gamma_thz_hwhm,
            reference_gamma,
            rtol=FIELD_RTOL,
            atol=FIELD_ATOL,
        )
    )
    diagonal_equal = bool(
        np.allclose(
            np.diagonal(result.kappa_w_mk, axis1=1, axis2=2),
            reference_kappa[:, :3],
            rtol=FIELD_RTOL,
            atol=FIELD_ATOL,
        )
    )
    offdiagonal = bool(
        np.max(np.abs(result.kappa_voigt_w_mk[:, 3:])) <= KAPPA_OFFDIAGONAL_ATOL_W_MK
        and np.max(np.abs(reference_kappa[:, 3:])) <= KAPPA_OFFDIAGONAL_ATOL_W_MK
    )
    return {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "end_to_end",
        "mesh": (mesh, mesh, mesh),
        "campaign_sha256": campaign_sha256,
        "reference_payload_sha256": _sha256(payload_path),
        "selectors_equal": selectors_equal,
        "gamma_equal": gamma_equal,
        "diagonal_kappa_equal": diagonal_equal,
        "offdiagonal_kappa_below_absolute_gate": offdiagonal,
        "phono3py_kappa_voigt_w_mk": reference_kappa,
        "passed": bool(
            selectors_equal and gamma_equal and diagonal_equal and offdiagonal
        ),
    }


def _compute_end_to_end_stage(
    config: ValidationConfig,
    snapshot: InputSnapshot,
    *,
    mesh: int,
    checkpoint_dir: Path,
    campaign_sha256: str,
) -> dict[str, Any] | None:
    if mesh not in config.end_to_end_meshes:
        return None
    from phonopy.phonon.grid import get_qpoints_from_bz_grid_points

    from negfpy.bulk import read_rate_spectrum_hdf5, read_rta_result_hdf5

    record_path = checkpoint_dir / "end-to-end.json"
    payload_path = checkpoint_dir / "end-to-end-reference.hdf5"
    rate_path = checkpoint_dir / "rates.hdf5"
    rta_path = checkpoint_dir / "rta.hdf5"
    rates = read_rate_spectrum_hdf5(rate_path)
    result = read_rta_result_hdf5(rta_path)
    identity = {
        "campaign_sha256": campaign_sha256,
        "archive_sha256": snapshot.archive_sha256[mesh],
        "mesh_json": _canonical_json([mesh, mesh, mesh]),
        "sigma_thz": config.sigma_thz,
        "rate_sha256": _sha256(rate_path),
        "rta_sha256": _sha256(rta_path),
    }
    if payload_path.exists():
        operands = _read_checkpoint_payload(
            payload_path,
            schema=END_TO_END_PAYLOAD_SCHEMA,
            identity=identity,
            dataset_names=END_TO_END_REFERENCE_DATASETS,
        )
    else:
        ph3 = _phono3py_reference_problem(
            snapshot.paths["crystal"],
            snapshot.paths["fc2"],
            snapshot.paths["fc3"],
        )
        ph3.mesh_numbers = [mesh, mesh, mesh]
        ph3.sigmas = [config.sigma_thz]
        ph3.init_phph_interaction()
        ph3.run_thermal_conductivity(
            temperatures=np.asarray(config.temperatures_k),
            is_kappa_star=False,
            log_level=0,
        )
        conductivity = ph3.thermal_conductivity
        if conductivity is None:
            raise RuntimeError("phono3py did not produce a conductivity result.")
        q_reference = get_qpoints_from_bz_grid_points(
            conductivity.grid_points,
            ph3.phph_interaction.bz_grid,
        )
        permutation = _match_qpoints(rates.qpoints, q_reference)
        reference_gamma = np.asarray(conductivity.gamma)[0][:, permutation, :]
        reference_kappa = np.asarray(conductivity.kappa)[0]
        _write_checkpoint_payload(
            payload_path,
            schema=END_TO_END_PAYLOAD_SCHEMA,
            identity=identity,
            datasets={
                "temperatures": rates.temperatures,
                "grid_points": rates.grid_points,
                "band_indices": rates.band_indices,
                "qpoints": rates.qpoints,
                "gamma_thz_hwhm": reference_gamma,
                "kappa_voigt_w_mk": reference_kappa,
            },
        )
        operands = _read_checkpoint_payload(
            payload_path,
            schema=END_TO_END_PAYLOAD_SCHEMA,
            identity=identity,
            dataset_names=END_TO_END_REFERENCE_DATASETS,
        )
    derived = _derive_end_to_end_record(
        rates,
        result,
        operands,
        mesh=mesh,
        campaign_sha256=campaign_sha256,
        payload_path=payload_path,
    )
    return _publish_derived_checkpoint(
        record_path,
        derived,
        stage="end_to_end",
        mesh=mesh,
        campaign_sha256=campaign_sha256,
    )


def _verify_input_snapshot(snapshot: InputSnapshot) -> None:
    for name, path in snapshot.paths.items():
        if _sha256(path) != snapshot.sha256[name]:
            raise RuntimeError(f"Input {name!r} changed during validation.")
    for mesh, path in snapshot.archive_paths.items():
        if _sha256(path) != snapshot.archive_sha256[mesh]:
            raise RuntimeError(f"Mesh {mesh} archive changed during validation.")


def _write_final_hdf5(
    path: Path,
    *,
    mesh_evidence: Sequence[Mapping[str, Any]],
    output_dir: Path,
    campaign_sha256: str,
    evidence_sha256: str,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with h5py.File(temporary, "w") as target:
            target.attrs["schema"] = FINAL_HDF5_SCHEMA
            target.attrs["campaign_sha256"] = campaign_sha256
            target.attrs["evidence_schema"] = FINAL_EVIDENCE_SCHEMA
            target.attrs["evidence_sha256"] = evidence_sha256
            meshes_group = target.create_group("meshes")
            for evidence in mesh_evidence:
                mesh = int(evidence["mesh"][0])
                rta_path = output_dir / "checkpoints" / f"mesh-{mesh:03d}" / "rta.hdf5"
                harmonic_path = (
                    output_dir / "checkpoints" / f"mesh-{mesh:03d}" / "harmonic.hdf5"
                )
                rta_source = _regular_file(rta_path, name=f"mesh {mesh} RTA source")
                harmonic_source = _regular_file(
                    harmonic_path, name=f"mesh {mesh} harmonic source"
                )
                source_hashes = {
                    "rta": _sha256(rta_source),
                    "harmonic": _sha256(harmonic_source),
                }
                if (
                    source_hashes["rta"] != evidence["rta_sha256"]
                    or source_hashes["harmonic"] != evidence["harmonic_sha256"]
                ):
                    raise ValueError(
                        f"Mesh {mesh} final HDF5 source hashes conflict with evidence."
                    )
                group = meshes_group.create_group(f"mesh-{mesh}x{mesh}x{mesh}")
                group.attrs["n_qpoints"] = int(evidence["n_qpoints"])
                group.attrs["passed"] = bool(evidence["passed"])
                group.attrs["source_rta_sha256"] = source_hashes["rta"]
                group.attrs["source_harmonic_sha256"] = source_hashes["harmonic"]
                group.attrs["metrics_json"] = _canonical_json(evidence)
                with h5py.File(rta_source, "r") as source:
                    source_paths = _internal_hdf5_dataset_paths(
                        source,
                        require_empty_group_attributes=False,
                    )
                    rta_group = group.create_group("rta")
                    for name in source:
                        source.copy(name, rta_group)
                    for name, value in source.attrs.items():
                        rta_group.attrs[f"source_{name}"] = value
                    if any(
                        _dataset_sha256(source[name])
                        != _dataset_sha256(rta_group[name])
                        for name in source_paths
                    ):
                        raise ValueError(f"Mesh {mesh} RTA HDF5 copy changed data.")
                if _sha256(rta_source) != source_hashes["rta"]:
                    raise RuntimeError(f"Mesh {mesh} RTA source changed during copy.")
                with h5py.File(harmonic_source, "r") as source:
                    source_paths = _internal_hdf5_dataset_paths(
                        source,
                        require_empty_group_attributes=False,
                    )
                    harmonic_group = group.create_group("harmonic")
                    for name in source:
                        source.copy(name, harmonic_group)
                    for name, value in source.attrs.items():
                        harmonic_group.attrs[f"source_{name}"] = value
                    if any(
                        _dataset_sha256(source[name])
                        != _dataset_sha256(harmonic_group[name])
                        for name in source_paths
                    ):
                        raise ValueError(f"Mesh {mesh} harmonic HDF5 copy changed data.")
                if _sha256(harmonic_source) != source_hashes["harmonic"]:
                    raise RuntimeError(
                        f"Mesh {mesh} harmonic source changed during copy."
                    )
            target.flush()
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _final_mesh_evidence(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return the stable direct-metric schema shared by JSON and HDF5."""

    result: list[dict[str, Any]] = []
    for record in records:
        mesh = tuple(int(value) for value in record["mesh"])
        core = record["core"]
        end_to_end = record["end_to_end"]
        result.append(
            {
                "mesh": mesh,
                "n_qpoints": int(np.prod(mesh)),
                "passed": bool(record["passed"]),
                "core_passed": bool(core["passed"]),
                "end_to_end_required": end_to_end is not None,
                "end_to_end_passed": (
                    None if end_to_end is None else bool(end_to_end["passed"])
                ),
                "archive_sha256": core["archive_sha256"],
                "rate_sha256": core["rate_sha256"],
                "rta_sha256": core["rta_sha256"],
                "harmonic_sha256": core["harmonic_sha256"],
                "harmonic_metrics": core["harmonic"],
                "rate_rta_frequency_max_abs_error_thz": core[
                    "rate_rta_frequency_max_abs_error_thz"
                ],
                "rate_rta_frequency_within_tolerance": core[
                    "rate_rta_frequency_within_tolerance"
                ],
                "maximum_heat_capacity_absolute_error_ev_per_k": core[
                    "maximum_heat_capacity_absolute_error_ev_per_k"
                ],
                "mode_kappa_equal": core["mode_kappa_equal"],
                "total_kappa_diagonal_equal": core["total_kappa_diagonal_equal"],
                "total_kappa_offdiagonal_below_absolute_gate": core[
                    "total_kappa_offdiagonal_below_absolute_gate"
                ],
                "negfpy_kappa_voigt_w_mk": core["negfpy_kappa_voigt_w_mk"],
                "reference_kappa_voigt_w_mk": core["reference_kappa_voigt_w_mk"],
                "code_and_input_provenance_eligible": core[
                    "code_and_input_provenance_eligible"
                ],
                "end_to_end_metrics": (
                    None
                    if end_to_end is None
                    else {
                        "selectors_equal": end_to_end["selectors_equal"],
                        "gamma_equal": end_to_end["gamma_equal"],
                        "diagonal_kappa_equal": end_to_end["diagonal_kappa_equal"],
                        "offdiagonal_kappa_below_absolute_gate": end_to_end[
                            "offdiagonal_kappa_below_absolute_gate"
                        ],
                    }
                ),
            }
        )
    return result


def _publish_final_evidence(
    path: Path,
    document: Mapping[str, Any],
    *,
    resume: bool,
) -> str:
    payload = (
        json.dumps(_json_value(document), sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    if path.exists():
        if not resume or path.is_symlink() or not path.is_file():
            raise FileExistsError(path)
        if path.read_bytes() != payload:
            raise ValueError("Existing final evidence differs from exact checkpoints.")
    else:
        _write_bytes_atomic(path, payload)
    return digest


def _hdf5_dataset_paths(group: h5py.Group) -> frozenset[str]:
    return frozenset(_internal_hdf5_dataset_paths(group))


def _validate_campaign_publication_contract(
    campaign: Mapping[str, Any],
    report: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    expected_campaign_keys = {
        "schema",
        "mode",
        "release_certification_possible",
        "settings",
        "inputs",
        "runtime",
        "tasks",
    }
    if (
        set(campaign) != expected_campaign_keys
        or campaign.get("schema") != CAMPAIGN_SCHEMA
    ):
        raise ValueError("Campaign document schema is not exact.")
    mode = campaign.get("mode")
    if mode not in {"development", "release"} or report.get("mode") != mode:
        raise ValueError("Campaign/report mode binding failed.")
    settings = campaign.get("settings")
    expected_setting_keys = {
        "meshes",
        "end_to_end_meshes",
        "temperatures_k",
        "sigma_thz",
        "cutoff_frequency_thz",
        "frequency_factor_to_thz",
    }
    if not isinstance(settings, Mapping) or set(settings) != expected_setting_keys:
        raise ValueError("Campaign settings schema is not exact.")
    try:
        meshes = tuple(int(value) for value in settings["meshes"])
        end_to_end_meshes = tuple(int(value) for value in settings["end_to_end_meshes"])
        temperatures = tuple(float(value) for value in settings["temperatures_k"])
        sigma = float(settings["sigma_thz"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Campaign numerical settings are malformed.") from exc
    if (
        not meshes
        or len(meshes) != len(set(meshes))
        or any(mesh <= 0 for mesh in meshes)
        or any(mesh not in meshes for mesh in end_to_end_meshes)
        or not temperatures
        or any(not np.isfinite(value) or value < 0.0 for value in temperatures)
        or not np.isfinite(sigma)
        or sigma <= 0.0
        or settings.get("cutoff_frequency_thz") != CUTOFF_THZ
        or settings.get("frequency_factor_to_thz") != FREQUENCY_FACTOR_TO_THZ
    ):
        raise ValueError("Campaign numerical settings are invalid.")
    expected_tasks = [
        {
            "task_id": f"mesh-{mesh}x{mesh}x{mesh}",
            "mesh": [mesh, mesh, mesh],
            "requires_end_to_end": mesh in end_to_end_meshes,
            "dependencies": [],
        }
        for mesh in meshes
    ]
    if campaign.get("tasks") != expected_tasks:
        raise ValueError("Campaign task graph conflicts with its settings.")
    inputs = campaign.get("inputs")
    if not isinstance(inputs, Mapping):
        raise TypeError("Campaign inputs schema is invalid.")
    report_runtime_before = report.get("runtime_before")
    if not isinstance(report_runtime_before, Mapping):
        raise TypeError("Final pre-compute runtime identity is missing.")
    if (
        inputs.get("source_sha256") != evidence.get("source_sha256")
        or inputs.get("archive_sha256") != evidence.get("archive_sha256")
        or campaign.get("runtime")
        != _stable_runtime_identity(report_runtime_before)
        or report.get("settings") != settings
        or evidence.get("settings") != settings
    ):
        raise ValueError("Campaign inputs/runtime are not bound to final evidence.")
    mesh_evidence = evidence.get("meshes")
    if not isinstance(mesh_evidence, list) or len(mesh_evidence) != len(meshes):
        raise ValueError("Final mesh evidence is incomplete.")
    recomputed_mesh_passes: list[bool] = []
    for mesh, item in zip(meshes, mesh_evidence, strict=True):
        requires_end_to_end = mesh in end_to_end_meshes
        if (
            not isinstance(item, Mapping)
            or item.get("mesh") != [mesh, mesh, mesh]
            or item.get("n_qpoints") != mesh**3
            or item.get("end_to_end_required") is not requires_end_to_end
        ):
            raise ValueError("Final mesh evidence conflicts with campaign tasks.")
        harmonic = item.get("harmonic_metrics")
        if not isinstance(harmonic, Mapping):
            raise TypeError("Final harmonic evidence schema is invalid.")
        try:
            harmonic_passed = bool(
                float(harmonic["uniform_mesh_frequency_max_abs_error_thz"])
                <= FREQUENCY_ATOL_THZ
                and float(harmonic["group_velocity_max_abs_error_thz_angstrom"])
                <= GV_ATOL_THZ_ANGSTROM
                and float(harmonic["band_frequency_max_abs_error_thz"])
                <= FREQUENCY_ATOL_THZ
                and float(harmonic["dos_max_abs_error_states_per_thz_cell"])
                <= FIELD_ATOL
                and float(harmonic["dos_projection_sum_max_abs_error"])
                <= FIELD_ATOL
                and float(harmonic["dos_normalization_error"])
                <= DOS_NORMALIZATION_ATOL
                and harmonic["dos_normalization_certified"] is True
                and harmonic["fc2_schema_audit_passed"] is True
                and harmonic["fc2_physical_audit_passed"] is True
            )
            core_passed = bool(
                harmonic_passed
                and item["rate_rta_frequency_within_tolerance"] is True
                and float(item["rate_rta_frequency_max_abs_error_thz"])
                <= FREQUENCY_ATOL_THZ
                and float(item["maximum_heat_capacity_absolute_error_ev_per_k"])
                <= CV_ATOL_EV_K
                and item["mode_kappa_equal"] is True
                and item["total_kappa_diagonal_equal"] is True
                and item["total_kappa_offdiagonal_below_absolute_gate"] is True
                and (
                    mode == "development"
                    or item["code_and_input_provenance_eligible"] is True
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Final core evidence schema is invalid.") from exc
        end_to_end_metrics = item.get("end_to_end_metrics")
        if requires_end_to_end:
            if not isinstance(end_to_end_metrics, Mapping):
                raise TypeError("Required end-to-end evidence is missing.")
            end_to_end_passed = bool(
                end_to_end_metrics.get("selectors_equal") is True
                and end_to_end_metrics.get("gamma_equal") is True
                and end_to_end_metrics.get("diagonal_kappa_equal") is True
                and end_to_end_metrics.get("offdiagonal_kappa_below_absolute_gate")
                is True
            )
        else:
            if end_to_end_metrics is not None:
                raise ValueError("Unexpected end-to-end evidence is present.")
            end_to_end_passed = True
        mesh_passed = bool(core_passed and end_to_end_passed)
        if (
            harmonic.get("passed") is not harmonic_passed
            or item.get("core_passed") is not core_passed
            or item.get("end_to_end_passed")
            is not (end_to_end_passed if requires_end_to_end else None)
            or item.get("passed") is not mesh_passed
        ):
            raise ValueError("Final mesh decision is not derived from scalar gates.")
        recomputed_mesh_passes.append(mesh_passed)
    scientific_passed = bool(recomputed_mesh_passes) and all(recomputed_mesh_passes)
    if (
        report.get("scientific_passed") is not scientific_passed
        or evidence.get("scientific_passed") is not scientific_passed
    ):
        raise ValueError("Final scientific decision is not derived from all meshes.")
    if not isinstance(report.get("runtime_unchanged"), bool):
        raise TypeError("Final runtime-stability decision is missing.")
    runtime_before = report_runtime_before
    runtime_after = report.get("runtime_after")
    if not isinstance(runtime_before, Mapping) or not isinstance(
        runtime_after, Mapping
    ):
        raise TypeError("Final runtime identities are missing.")
    runtime_unchanged = bool(
        _stable_runtime_identity(runtime_before)
        == _stable_runtime_identity(runtime_after)
    )
    if report.get("runtime_unchanged") is not runtime_unchanged:
        raise ValueError("Final runtime-stability decision was not recomputed.")

    strict_settings = bool(
        meshes == RELEASE_MESHES
        and end_to_end_meshes == RELEASE_END_TO_END_MESHES
        and temperatures == RELEASE_TEMPERATURES_K
        and sigma == RELEASE_SIGMA_THZ
    )
    source_sha256 = evidence.get("source_sha256")
    archive_sha256 = evidence.get("archive_sha256")
    strict_inputs = bool(
        isinstance(source_sha256, Mapping)
        and source_sha256.get("crystal") == PINNED_SOURCE_SHA256["crystal"]
        and source_sha256.get("fc2") == PINNED_SOURCE_SHA256["fc2"]
        and source_sha256.get("fc3") == PINNED_SOURCE_SHA256["fc3"]
        and source_sha256.get("manifest") == PINNED_V0_2_MANIFEST_SHA256
        and source_sha256.get("environment_lock") == PINNED_ENVIRONMENT_LOCK_SHA256
        and isinstance(archive_sha256, Mapping)
        and archive_sha256
        == {str(mesh): PINNED_ARCHIVES[mesh]["file_sha256"] for mesh in RELEASE_MESHES}
    )
    runtime_eligible = False
    if mode == "release":
        try:
            _require_release_runtime(
                report.get("runtime_after", {}),
                policy=STRICT_POLICY,
            )
        except (KeyError, RuntimeError, TypeError, ValueError):
            pass
        else:
            runtime_eligible = True
    candidate = bool(
        mode == "release"
        and campaign.get("release_certification_possible") is True
        and scientific_passed
        and runtime_unchanged
        and strict_settings
        and strict_inputs
        and runtime_eligible
    )
    if (
        report.get("candidate_release_gates_passed") is not candidate
        or evidence.get("candidate_release_gates_passed") is not candidate
    ):
        raise ValueError("Final candidate decision is not derived from release policy.")


def _validate_release_artifact_layout(
    output_dir: Path,
    campaign: Mapping[str, Any],
) -> None:
    settings = campaign["settings"]
    meshes = tuple(int(value) for value in settings["meshes"])
    end_to_end_meshes = {int(value) for value in settings["end_to_end_meshes"]}
    expected_directories = {"staged-inputs", "checkpoints"}
    expected_files = {
        "campaign-manifest.json",
        "release-evidence.json",
        "bulk-rta-results.hdf5",
        "bulk-rta-validation.json",
        "staged-inputs/crystal.yaml",
        "staged-inputs/fc2.compact.hdf5",
        "staged-inputs/fc3.compact.hdf5",
        "staged-inputs/manifest.json",
    }
    optional_root_files = {"SHA256SUMS.json", "release-completion-receipt.json"}
    for mesh in meshes:
        checkpoint = f"checkpoints/mesh-{mesh:03d}"
        expected_directories.add(checkpoint)
        expected_files.add(f"staged-inputs/mesh-{mesh:03d}-rate-fields.hdf5")
        expected_files.update(
            {
                f"{checkpoint}/core.json",
                f"{checkpoint}/rates.hdf5",
                f"{checkpoint}/rta.hdf5",
                f"{checkpoint}/harmonic.hdf5",
                f"{checkpoint}/core-reference.hdf5",
                f"{checkpoint}/complete.json",
            }
        )
        if mesh in end_to_end_meshes:
            expected_files.update(
                {
                    f"{checkpoint}/end-to-end.json",
                    f"{checkpoint}/end-to-end-reference.hdf5",
                }
            )
    actual_directories: set[str] = set()
    actual_files: set[str] = set()
    invalid_checkpoint_pattern = re.compile(
        r"^checkpoints/mesh-[0-9]{3}/"
        r"(?:core|end-to-end)\.json\.invalid-[0-9a-f]{12}$"
    )
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Release artifact layout contains symlink {path}.")
        relative = str(path.relative_to(output_dir))
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            actual_directories.add(relative)
        elif stat.S_ISREG(mode):
            actual_files.add(relative)
        else:
            raise ValueError(f"Release artifact layout contains special file {path}.")
    unexpected_directories = actual_directories - expected_directories
    unexpected_files = actual_files - expected_files - optional_root_files
    unexpected_files = {
        name
        for name in unexpected_files
        if invalid_checkpoint_pattern.fullmatch(name) is None
    }
    if (
        not expected_directories.issubset(actual_directories)
        or not expected_files.issubset(actual_files)
        or unexpected_directories
        or unexpected_files
    ):
        raise ValueError(
            "Release artifact layout is not exact: "
            f"unexpected_directories={sorted(unexpected_directories)}, "
            f"unexpected_files={sorted(unexpected_files)}."
        )


def _validate_staged_publication(
    report_path: Path,
    hdf5_path: Path,
    evidence_path: Path,
) -> None:
    """Verify provisional JSON/evidence/HDF5 before certification is sealed."""

    report = _read_json_object(
        _regular_file(report_path, name="final JSON report"),
        name="final JSON report",
    )
    evidence_file = _regular_file(evidence_path, name="final evidence document")
    evidence = _read_json_object(evidence_file, name="final evidence document")
    evidence_sha256 = _sha256(evidence_file)
    results_file = _regular_file(hdf5_path, name="final HDF5 results")
    campaign_file = _regular_file(
        report_path.parent / "campaign-manifest.json",
        name="campaign manifest",
    )
    campaign = _read_json_object(campaign_file, name="campaign manifest")
    expected_report_keys = {
        "schema",
        "mode",
        "scientific_passed",
        "candidate_release_gates_passed",
        "release_certified",
        "certification_authority",
        "physical_mesh_convergence_certified",
        "physical_mesh_convergence_note",
        "approximation",
        "excluded_physics",
        "campaign_manifest",
        "campaign_sha256",
        "evidence_document",
        "evidence_sha256",
        "results_hdf5",
        "results_hdf5_sha256",
        "runtime_before",
        "runtime_after",
        "runtime_unchanged",
        "inputs",
        "settings",
        "mesh_evidence",
        "meshes",
    }
    expected_evidence_keys = {
        "schema",
        "campaign_sha256",
        "scientific_passed",
        "candidate_release_gates_passed",
        "release_certified",
        "certification_authority",
        "settings",
        "source_sha256",
        "archive_sha256",
        "meshes",
    }
    if set(report) != expected_report_keys or set(evidence) != expected_evidence_keys:
        raise ValueError("Final JSON/evidence document schemas are not exact.")
    report_inputs = report.get("inputs")
    if not isinstance(report_inputs, Mapping) or set(report_inputs) != {
        "source_paths",
        "source_sha256",
        "archive_paths",
        "archive_sha256",
    }:
        raise ValueError("Final report input schema is not exact.")
    _validate_campaign_publication_contract(campaign, report, evidence)
    _validate_release_artifact_layout(report_path.parent, campaign)
    checks = {
        "report_schema": report.get("schema") == SCHEMA,
        "evidence_schema": evidence.get("schema") == FINAL_EVIDENCE_SCHEMA,
        "report_scientific_decision": isinstance(report.get("scientific_passed"), bool),
        "evidence_scientific_decision": isinstance(
            evidence.get("scientific_passed"), bool
        ),
        "scientific_decision": report.get("scientific_passed")
        == evidence.get("scientific_passed"),
        "report_candidate_decision": isinstance(
            report.get("candidate_release_gates_passed"), bool
        ),
        "evidence_candidate_decision": isinstance(
            evidence.get("candidate_release_gates_passed"), bool
        ),
        "candidate_decision": report.get("candidate_release_gates_passed")
        == evidence.get("candidate_release_gates_passed"),
        "report_provisional_not_certified": report.get("release_certified") is False,
        "evidence_provisional_not_certified": evidence.get("release_certified")
        is False,
        "certification_authority": report.get("certification_authority")
        == evidence.get("certification_authority")
        == "release-completion-receipt.json",
        "report_evidence_sha256": report.get("evidence_sha256") == evidence_sha256,
        "report_hdf5_sha256": report.get("results_hdf5_sha256")
        == _sha256(results_file),
        "campaign": report.get("campaign_sha256") == evidence.get("campaign_sha256"),
        "campaign_file_sha256": report.get("campaign_sha256") == _sha256(campaign_file),
        "candidate_requires_science": not bool(
            report.get("candidate_release_gates_passed")
        )
        or report.get("scientific_passed") is True,
        "settings": report.get("settings") == evidence.get("settings"),
        "source_sha256": report.get("inputs", {}).get("source_sha256")
        == evidence.get("source_sha256"),
        "archive_sha256": report.get("inputs", {}).get("archive_sha256")
        == evidence.get("archive_sha256"),
        "mesh_metrics": report.get("mesh_evidence") == evidence.get("meshes"),
        "physical_mesh_not_certified": report.get(
            "physical_mesh_convergence_certified"
        )
        is False,
        "approximation": report.get("approximation")
        == "intrinsic_three_phonon_smrta",
        "campaign_path": report.get("campaign_manifest") == str(campaign_file),
        "evidence_path": report.get("evidence_document") == str(evidence_file),
        "results_path": report.get("results_hdf5") == str(results_file),
    }
    with h5py.File(results_file, "r") as handle:
        try:
            _internal_hdf5_dataset_paths(
                handle,
                require_empty_group_attributes=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Final HDF5 must contain only internal hard links."
            ) from exc
        root_schema_exact = bool(
            {str(name) for name in handle.attrs}
            == {
                "schema",
                "campaign_sha256",
                "evidence_schema",
                "evidence_sha256",
            }
            and set(handle.keys()) == {"meshes"}
            and isinstance(handle.get("meshes"), h5py.Group)
        )
        checks.update(
            {
                "hdf5_root_schema_exact": root_schema_exact,
                "hdf5_schema": str(handle.attrs.get("schema", "")) == FINAL_HDF5_SCHEMA,
                "hdf5_evidence_schema": str(handle.attrs.get("evidence_schema", ""))
                == FINAL_EVIDENCE_SCHEMA,
                "hdf5_evidence_sha256": str(handle.attrs.get("evidence_sha256", ""))
                == evidence_sha256,
                "hdf5_campaign": str(handle.attrs.get("campaign_sha256", ""))
                == evidence.get("campaign_sha256"),
            }
        )
        meshes = evidence.get("meshes")
        if not isinstance(meshes, list):
            checks["mesh_schema"] = False
        else:
            required_rta = {
                "temperatures",
                "qpoints",
                "frequencies_thz",
                "group_velocities_thz_angstrom",
                "heat_capacities_ev_per_k",
                "gamma_thz_hwhm",
                "population_lifetime_ps",
                "mean_free_path_angstrom",
                "mode_kappa_w_mk",
                "band_index_kappa_w_mk",
                "kappa_w_mk",
                "qpoint_weights",
                "included_mask",
            }
            mesh_schema = True
            mesh_keys: list[tuple[int, int, int]] = []
            for item in meshes:
                if not isinstance(item, Mapping) or not {
                    "mesh",
                    "n_qpoints",
                    "passed",
                    "rta_sha256",
                    "harmonic_sha256",
                }.issubset(item):
                    mesh_schema = False
                    continue
                try:
                    mesh_key = tuple(int(value) for value in item["mesh"])
                except (TypeError, ValueError):
                    mesh_schema = False
                    continue
                if len(mesh_key) != 3 or len(set(mesh_key)) != 1:
                    mesh_schema = False
                    continue
                mesh = mesh_key[0]
                mesh_keys.append(mesh_key)
                group_name = f"meshes/mesh-{mesh}x{mesh}x{mesh}"
                if group_name not in handle:
                    mesh_schema = False
                    continue
                group = handle[group_name]
                harmonic = group.get("harmonic")
                rta = group.get("rta")
                if not isinstance(harmonic, h5py.Group) or not isinstance(
                    rta, h5py.Group
                ):
                    mesh_schema = False
                    continue
                harmonic_paths = _hdf5_dataset_paths(harmonic)
                rta_paths = _hdf5_dataset_paths(rta)
                try:
                    n_qpoints = int(item["n_qpoints"])
                    n_temperatures = int(rta["temperatures"].shape[0])
                    n_bands = int(rta["frequencies_thz"].shape[1])
                    shape_dtype_valid = bool(
                        rta["temperatures"].shape == (n_temperatures,)
                        and rta["qpoints"].shape == (n_qpoints, 3)
                        and rta["frequencies_thz"].shape == (n_qpoints, n_bands)
                        and rta["group_velocities_thz_angstrom"].shape
                        == (n_qpoints, n_bands, 3)
                        and all(
                            rta[name].shape == (n_temperatures, n_qpoints, n_bands)
                            for name in (
                                "heat_capacities_ev_per_k",
                                "gamma_thz_hwhm",
                                "population_lifetime_ps",
                                "mean_free_path_angstrom",
                            )
                        )
                        and rta["mode_kappa_w_mk"].shape
                        == (n_temperatures, n_qpoints, n_bands, 3, 3)
                        and rta["band_index_kappa_w_mk"].shape
                        == (n_temperatures, n_bands, 3, 3)
                        and rta["kappa_w_mk"].shape == (n_temperatures, 3, 3)
                        and rta["qpoint_weights"].shape == (n_qpoints,)
                        and rta["included_mask"].shape == (n_qpoints, n_bands)
                        and harmonic["mesh/frequencies_thz"].shape
                        == (n_qpoints, n_bands)
                        and harmonic["mesh/group_velocities_thz_angstrom"].shape
                        == (n_qpoints, n_bands, 3)
                        and np.dtype(rta["included_mask"].dtype) == np.dtype(np.bool_)
                        and all(
                            np.dtype(rta[name].dtype) == np.dtype(np.float64)
                            for name in required_rta - {"included_mask"}
                        )
                        and np.dtype(harmonic["band/vertex_indices"].dtype)
                        == np.dtype(np.int64)
                        and all(
                            np.dtype(harmonic[name].dtype) == np.dtype(np.float64)
                            for name in HARMONIC_CHECKPOINT_DATASETS
                            - {"band/vertex_indices"}
                        )
                        and all(not rta[name].attrs for name in rta_paths)
                        and all(not harmonic[name].attrs for name in harmonic_paths)
                    )
                except (KeyError, IndexError, TypeError, ValueError):
                    shape_dtype_valid = False
                mesh_schema = bool(
                    mesh_schema
                    and harmonic_paths == HARMONIC_CHECKPOINT_DATASETS
                    and rta_paths == required_rta
                    and shape_dtype_valid
                    and set(rta.attrs)
                    == {
                        "source_schema",
                        "source_approximation",
                        "source_primitive_volume_angstrom3",
                        "source_cutoff_frequency_thz",
                        "source_metadata_json",
                        "source_payload_sha256",
                    }
                    and set(harmonic.attrs)
                    == {
                        "source_schema",
                        "source_campaign_sha256",
                        "source_archive_sha256",
                        "source_mesh_json",
                        "source_payload_sha256",
                    }
                    and set(group.attrs)
                    == {
                        "n_qpoints",
                        "passed",
                        "source_rta_sha256",
                        "source_harmonic_sha256",
                        "metrics_json",
                    }
                    and bool(group.attrs.get("passed", False)) == bool(item["passed"])
                    and str(group.attrs.get("source_rta_sha256", ""))
                    == item["rta_sha256"]
                    and str(group.attrs.get("source_harmonic_sha256", ""))
                    == item["harmonic_sha256"]
                    and int(group.attrs.get("n_qpoints", -1)) == int(item["n_qpoints"])
                    and json.loads(str(group.attrs.get("metrics_json", "{}"))) == item
                )
            expected_group_names = {
                f"mesh-{mesh[0]}x{mesh[1]}x{mesh[2]}" for mesh in mesh_keys
            }
            meshes_group = handle.get("meshes")
            checks["mesh_schema"] = bool(
                mesh_schema
                and isinstance(meshes_group, h5py.Group)
                and len(mesh_keys) == len(set(mesh_keys))
                and list(mesh_keys) == sorted(mesh_keys)
                and set(meshes_group.keys()) == expected_group_names
            )
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(
            "Final JSON/evidence/HDF5 binding failed: " + ", ".join(failed)
        )


def _checksum_manifest_payload(output_dir: Path) -> dict[str, str]:
    excluded = {"SHA256SUMS.json", "release-completion-receipt.json"}
    payload: dict[str, str] = {}
    for path in output_dir.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Release artifact tree contains symlink {path}.")
        relative = str(path.relative_to(output_dir))
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            key = relative + "/"
            payload[key] = hashlib.sha256(
                b"negfpy.release-directory.v1\0" + relative.encode("utf-8")
            ).hexdigest()
        elif stat.S_ISREG(mode):
            if relative not in excluded:
                payload[relative] = _sha256(path)
        else:
            raise ValueError(f"Release artifact tree contains special file {path}.")
    return {name: payload[name] for name in sorted(payload)}


def _write_checksum_manifest(output_dir: Path) -> Path:
    path = output_dir / "SHA256SUMS.json"
    _write_json_atomic(path, _checksum_manifest_payload(output_dir))
    return path


def _validate_checksum_manifest(output_dir: Path, path: Path) -> dict[str, str]:
    declared = _read_json_object(
        _regular_file(path, name="SHA256 manifest"), name="SHA256 manifest"
    )
    if any(
        not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
        for name, digest in declared.items()
    ):
        raise ValueError("SHA256 manifest has an invalid entry schema.")
    actual = _checksum_manifest_payload(output_dir)
    if declared != actual:
        raise ValueError("SHA256 manifest differs from the exact artifact tree.")
    return {str(name): str(digest) for name, digest in declared.items()}


def _completion_receipt_document(
    *,
    report_path: Path,
    hdf5_path: Path,
    evidence_path: Path,
    checksums_path: Path,
) -> dict[str, Any]:
    report = _read_json_object(report_path, name="provisional JSON report")
    evidence = _read_json_object(evidence_path, name="provisional evidence")
    return {
        "schema": FINAL_RECEIPT_SCHEMA,
        "certification_authority": "sole",
        "campaign_sha256": report["campaign_sha256"],
        "scientific_passed": report["scientific_passed"],
        "candidate_release_gates_passed": report["candidate_release_gates_passed"],
        "provisional_release_certified": False,
        "release_certified": report["candidate_release_gates_passed"],
        "report_sha256": _sha256(report_path),
        "evidence_sha256": _sha256(evidence_path),
        "results_hdf5_sha256": _sha256(hdf5_path),
        "sha256sums_sha256": _sha256(checksums_path),
        "evidence_scientific_passed": evidence["scientific_passed"],
        "evidence_candidate_release_gates_passed": evidence[
            "candidate_release_gates_passed"
        ],
    }


def _validate_final_publication(
    report_path: Path,
    hdf5_path: Path,
    evidence_path: Path,
    checksums_path: Path,
    receipt_path: Path,
) -> dict[str, Any]:
    """Validate the completion receipt, the sole certification authority."""

    _validate_staged_publication(report_path, hdf5_path, evidence_path)
    _validate_checksum_manifest(receipt_path.parent, checksums_path)
    receipt = _read_json_object(
        _regular_file(receipt_path, name="release completion receipt"),
        name="release completion receipt",
    )
    expected_keys = {
        "schema",
        "certification_authority",
        "campaign_sha256",
        "scientific_passed",
        "candidate_release_gates_passed",
        "provisional_release_certified",
        "release_certified",
        "report_sha256",
        "evidence_sha256",
        "results_hdf5_sha256",
        "sha256sums_sha256",
        "evidence_scientific_passed",
        "evidence_candidate_release_gates_passed",
    }
    report = _read_json_object(report_path, name="provisional JSON report")
    evidence = _read_json_object(evidence_path, name="provisional evidence")
    checks = {
        "receipt_fields": set(receipt) == expected_keys,
        "receipt_schema": receipt.get("schema") == FINAL_RECEIPT_SCHEMA,
        "receipt_authority": receipt.get("certification_authority") == "sole",
        "receipt_boolean_types": all(
            isinstance(receipt.get(name), bool)
            for name in (
                "scientific_passed",
                "candidate_release_gates_passed",
                "provisional_release_certified",
                "release_certified",
                "evidence_scientific_passed",
                "evidence_candidate_release_gates_passed",
            )
        ),
        "receipt_campaign": receipt.get("campaign_sha256")
        == report.get("campaign_sha256")
        == evidence.get("campaign_sha256"),
        "receipt_scientific": receipt.get("scientific_passed")
        == receipt.get("evidence_scientific_passed")
        == report.get("scientific_passed")
        == evidence.get("scientific_passed"),
        "receipt_candidate": receipt.get("candidate_release_gates_passed")
        == receipt.get("evidence_candidate_release_gates_passed")
        == report.get("candidate_release_gates_passed")
        == evidence.get("candidate_release_gates_passed"),
        "receipt_provisional_false": receipt.get("provisional_release_certified")
        is False,
        "receipt_release_decision": receipt.get("release_certified")
        == receipt.get("candidate_release_gates_passed"),
        "report_sha256": receipt.get("report_sha256") == _sha256(report_path),
        "evidence_sha256": receipt.get("evidence_sha256") == _sha256(evidence_path),
        "results_hdf5_sha256": receipt.get("results_hdf5_sha256") == _sha256(hdf5_path),
        "sha256sums_sha256": receipt.get("sha256sums_sha256")
        == _sha256(checksums_path),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError("Release completion receipt failed: " + ", ".join(failed))
    return receipt


def _parse_arguments(argv: Sequence[str] | None = None) -> ValidationConfig:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--crystal-yaml", type=Path, required=True)
    parser.add_argument("--v0-2-release-manifest", type=Path, required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--environment-lock-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meshes", type=int, nargs="+", default=list(RELEASE_MESHES))
    parser.add_argument(
        "--end-to-end-meshes",
        type=int,
        nargs="*",
        default=list(RELEASE_END_TO_END_MESHES),
    )
    parser.add_argument(
        "--temperatures-k", type=float, nargs="+", default=list(RELEASE_TEMPERATURES_K)
    )
    parser.add_argument("--sigma-thz", type=float, default=RELEASE_SIGMA_THZ)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args(argv)
    return ValidationConfig(
        artifact_root=arguments.artifact_root,
        crystal_yaml=arguments.crystal_yaml,
        v0_2_release_manifest=arguments.v0_2_release_manifest,
        environment_lock=arguments.environment_lock,
        environment_lock_sha256=str(arguments.environment_lock_sha256).lower(),
        output_dir=arguments.output_dir.expanduser().resolve(),
        meshes=tuple(arguments.meshes),
        end_to_end_meshes=tuple(arguments.end_to_end_meshes),
        temperatures_k=tuple(arguments.temperatures_k),
        sigma_thz=float(arguments.sigma_thz),
        development=bool(arguments.development),
        resume=bool(arguments.resume),
        preflight_only=bool(arguments.preflight_only),
    )


def _release_decision(
    config: ValidationConfig,
    *,
    scientific_passed: bool,
    runtime_unchanged: bool,
    runtime_after: Mapping[str, Any],
) -> bool:
    """Return the sole release decision; development is unconditionally false."""

    return bool(
        not config.development
        and scientific_passed
        and runtime_unchanged
        and runtime_after.get("clean") is True
        and tuple(config.meshes) == RELEASE_MESHES
        and tuple(config.end_to_end_meshes) == RELEASE_END_TO_END_MESHES
        and tuple(config.temperatures_k) == RELEASE_TEMPERATURES_K
        and config.sigma_thz == RELEASE_SIGMA_THZ
    )


def run_validation(
    config: ValidationConfig,
    *,
    policy: ReleasePolicy = STRICT_POLICY,
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    snapshot, runtime_before = preflight_inputs(
        config, policy=policy, repository=repository
    )
    if config.preflight_only:
        return {
            "schema": PREFLIGHT_RECEIPT_SCHEMA,
            "mode": "preflight_only",
            "preflight_passed": True,
            "scientific_compute_started": False,
            "scientific_passed": None,
            "release_certified": False,
            "output_created": False,
            "output_dir": config.output_dir,
            "source_sha256": snapshot.sha256,
            "archive_sha256": snapshot.archive_sha256,
            "runtime": runtime_before,
        }
    # No output existed or was modified before this point.
    config.output_dir.mkdir(parents=False, exist_ok=config.resume)
    staged = _stage_scientific_inputs(config, snapshot)
    scientific_paths = dict(snapshot.paths)
    scientific_paths.update(staged.paths)
    scientific_snapshot = InputSnapshot(
        paths=scientific_paths,
        sha256={**snapshot.sha256, **staged.sha256},
        archive_paths=staged.archive_paths,
        archive_sha256=snapshot.archive_sha256,
        archive_evidence=snapshot.archive_evidence,
        sealed_manifest=snapshot.sealed_manifest,
    )
    campaign = _campaign_document(config, snapshot, runtime_before, staged)
    campaign_path, campaign_sha256 = _publish_campaign_manifest(
        config.output_dir, campaign, resume=config.resume
    )
    report_path = config.output_dir / "bulk-rta-validation.json"
    evidence_path = config.output_dir / "release-evidence.json"
    final_hdf5 = config.output_dir / "bulk-rta-results.hdf5"
    checksums_path = config.output_dir / "SHA256SUMS.json"
    receipt_path = config.output_dir / "release-completion-receipt.json"
    if receipt_path.exists():
        if not config.resume:
            raise FileExistsError(receipt_path)
        receipt = _validate_final_publication(
            report_path,
            final_hdf5,
            evidence_path,
            checksums_path,
            receipt_path,
        )
        report = _read_json_object(report_path, name="provisional JSON report")
        return {
            **report,
            "authoritative_release_certified": receipt["release_certified"],
            "completion_receipt": str(receipt_path),
            "completion_receipt_sha256": _sha256(receipt_path),
        }

    _preflight_resume_checkpoints(
        config,
        scientific_snapshot,
        campaign_sha256=campaign_sha256,
    )
    checkpoints_root = config.output_dir / "checkpoints"
    _ensure_regular_directory(checkpoints_root)
    records: list[dict[str, Any]] = []
    for mesh in config.meshes:
        checkpoint_dir = checkpoints_root / f"mesh-{mesh:03d}"
        _ensure_regular_directory(checkpoint_dir)
        core = _compute_core_stage(
            config,
            scientific_snapshot,
            mesh=mesh,
            checkpoint_dir=checkpoint_dir,
            campaign_sha256=campaign_sha256,
        )
        end_to_end = _compute_end_to_end_stage(
            config,
            scientific_snapshot,
            mesh=mesh,
            checkpoint_dir=checkpoint_dir,
            campaign_sha256=campaign_sha256,
        )
        passed = bool(core["passed"] and (end_to_end is None or end_to_end["passed"]))
        record = {
            "mesh": (mesh, mesh, mesh),
            "core": core,
            "end_to_end": end_to_end,
            "passed": passed,
        }
        _write_json_atomic(checkpoint_dir / "complete.json", record)
        records.append(record)

    _verify_input_snapshot(snapshot)
    _verify_staged_inputs(staged, snapshot, meshes=config.meshes)
    runtime_after = _runtime_identity(repository)
    if not config.development:
        runtime_after["environment_binding"] = _validate_conda_prefix_against_lock(
            snapshot.paths["environment_lock"], Path(sys.prefix)
        )
    runtime_unchanged = _stable_runtime_identity(
        runtime_after
    ) == _stable_runtime_identity(runtime_before)
    if not config.development:
        _require_release_runtime(runtime_after, policy=policy)
    scientific_passed = all(record["passed"] for record in records)
    candidate_release_gates_passed = _release_decision(
        config,
        scientific_passed=scientific_passed,
        runtime_unchanged=runtime_unchanged,
        runtime_after=runtime_after,
    )
    mesh_evidence = _final_mesh_evidence(records)
    evidence_document = {
        "schema": FINAL_EVIDENCE_SCHEMA,
        "campaign_sha256": campaign_sha256,
        "scientific_passed": scientific_passed,
        "candidate_release_gates_passed": candidate_release_gates_passed,
        "release_certified": False,
        "certification_authority": "release-completion-receipt.json",
        "settings": campaign["settings"],
        "source_sha256": snapshot.sha256,
        "archive_sha256": snapshot.archive_sha256,
        "meshes": mesh_evidence,
    }
    evidence_sha256 = _publish_final_evidence(
        evidence_path,
        evidence_document,
        resume=config.resume,
    )
    _write_final_hdf5(
        final_hdf5,
        mesh_evidence=mesh_evidence,
        output_dir=config.output_dir,
        campaign_sha256=campaign_sha256,
        evidence_sha256=evidence_sha256,
    )
    report = {
        "schema": SCHEMA,
        "mode": "development" if config.development else "release",
        "scientific_passed": scientific_passed,
        "candidate_release_gates_passed": candidate_release_gates_passed,
        "release_certified": False,
        "certification_authority": "release-completion-receipt.json",
        "physical_mesh_convergence_certified": False,
        "physical_mesh_convergence_note": (
            "This exact-input implementation validation is not a mesh- and "
            "smearing-converged silicon conductivity prediction."
        ),
        "approximation": "intrinsic_three_phonon_smrta",
        "excluded_physics": [
            "isotope_scattering",
            "boundary_scattering",
            "four_phonon_scattering",
            "iterative_lbte",
            "nac_polar_materials",
            "wigner_interband_transport",
        ],
        "campaign_manifest": str(campaign_path),
        "campaign_sha256": campaign_sha256,
        "evidence_document": str(evidence_path),
        "evidence_sha256": evidence_sha256,
        "results_hdf5": str(final_hdf5),
        "results_hdf5_sha256": _sha256(final_hdf5),
        "runtime_before": runtime_before,
        "runtime_after": runtime_after,
        "runtime_unchanged": runtime_unchanged,
        "inputs": {
            "source_paths": snapshot.paths,
            "source_sha256": snapshot.sha256,
            "archive_paths": snapshot.archive_paths,
            "archive_sha256": snapshot.archive_sha256,
        },
        "settings": campaign["settings"],
        "mesh_evidence": mesh_evidence,
        "meshes": records,
    }
    _write_json_atomic(report_path, report)
    _validate_staged_publication(report_path, final_hdf5, evidence_path)
    checksums_path = _write_checksum_manifest(config.output_dir)
    _validate_checksum_manifest(config.output_dir, checksums_path)
    receipt_document = _completion_receipt_document(
        report_path=report_path,
        hdf5_path=final_hdf5,
        evidence_path=evidence_path,
        checksums_path=checksums_path,
    )
    # Commit-last: this atomic rename is the sole operation allowed to assert
    # certification.  Every previously published artifact remains provisional.
    _write_json_atomic(receipt_path, receipt_document)
    receipt = _validate_final_publication(
        report_path,
        final_hdf5,
        evidence_path,
        checksums_path,
        receipt_path,
    )
    return {
        **report,
        "authoritative_release_certified": receipt["release_certified"],
        "completion_receipt": str(receipt_path),
        "completion_receipt_sha256": _sha256(receipt_path),
    }


def main(argv: Sequence[str] | None = None) -> int:
    config = _parse_arguments(argv)
    report = run_validation(config)
    print(json.dumps(_json_value(report), indent=2, sort_keys=True, allow_nan=False))
    if config.preflight_only:
        return 0 if report.get("preflight_passed") is True else 4
    if config.development:
        return 0 if report["scientific_passed"] else 2
    return 0 if report.get("authoritative_release_certified") is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
