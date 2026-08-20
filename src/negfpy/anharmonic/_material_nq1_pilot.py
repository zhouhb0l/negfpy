"""Immutable non-release pilot for the material ``N_q=1`` predictor.

The pilot is the exact first three tasks of the full release graph:
``g=.001`` from zero, followed by coupling-square continuations at ``g=.01``
and ``g=.025``.  The first two results must reproduce their immutable
release-018 field anchors.  The new ``g=.025`` result must converge under the
unchanged 800-iteration contract.  Passing this pilot can qualify a predictor
for a *new* candidate; it cannot certify or release a material result.

This module owns the byte-level manifest, receipt, comparison, report, and
audit contracts.  Scheduler submission is deliberately outside its scope.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import stat
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

import h5py
import numpy as np

from ._checkpoint_v2 import load_scba_checkpoint_v2_self_validating
from ._material_nq1_campaign import (
    MATERIAL_NQ1_ACCEPTANCE,
    MATERIAL_NQ1_EXCLUDED_CLAIMS,
    MaterialSCBATask,
    silicon_nq1_release_tasks,
)
from .current import _evaluate_current_report
from .diagnostics import check_solution
from .types import AnharmonicResult, DiagnosticReport, KeldyshSpectrum

PILOT_SCHEMA = "negfpy.material-nq1-coupling-square-pilot.v1"
PREFLIGHT_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-preflight.v1"
TASK_RECEIPT_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-task-receipt.v1"
COMPARISON_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-comparison.v1"
REPORT_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-report.v1"
AUDIT_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-audit.v1"
SCHEDULER_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-scheduler-terminal.v1"
SUBMISSION_SCHEMA = "negfpy.material-nq1-coupling-square-pilot-submission.v1"
PRE_SUBMISSION_SCHEMA = (
    "negfpy.material-nq1-coupling-square-pilot-pre-submission.v1"
)
QUALIFICATION_SCHEMA = "negfpy.material-nq1-candidate-qualification.v3"
CONTINUATION_INITIALIZATION_SCHEMA = (
    "negfpy.material-nq1-continuation-initialization.v1"
)

PILOT_TASK_IDS = (
    "coupling-g0001",
    "coupling-g001",
    "coupling-g0025",
)
FIELD_DATASETS = (
    "green/retarded",
    "green/lesser",
    "green/greater",
    "sigma_pp/retarded",
    "sigma_pp/lesser",
    "sigma_pp/greater",
)
FIELD_RTOL = 1.0e-8
FIELD_ATOL = 1.0e-12
CURRENT_RTOL = 1.0e-8
CURRENT_ATOL = 1.0e-12
EXPECTED_FIELD_SHAPE = (151, 216, 216)
EXPECTED_FIELD_DTYPE = "complex128"
PILOT_WALL_LIMIT_SECONDS = 48 * 60 * 60
PILOT_RSS_LIMIT_BYTES = 8 * 1024**3
PILOT_PBS_NCPUS = 2
PILOT_PBS_MEMORY = "16gb"
PILOT_PBS_WALLTIME = "50:00:00"
PILOT_PBS_PROJECT = "13004417"
PILOT_PBS_QUEUE = "normal"
# Aspire2A exposes /app as a symlink; use its canonical lexical path so the
# fail-closed ancestry checks do not reject the approved scheduler runtime.
PILOT_GIT_BIN = Path("/scratch/GPFS/app/apps/git/2.39.2/bin")
PILOT_PYTHON_VERSION = "3.12.13"
PILOT_DISTRIBUTIONS = MappingProxyType({
    "h5py": "3.16.0",
    "numpy": "2.5.1",
    "phonopy": "4.4.0",
    "phono3py": "4.4.0",
    "scipy": "1.18.0",
    "spglib": "2.7.0",
    "threadpoolctl": "3.6.0",
})
PILOT_THREAD_LIMITS = MappingProxyType({
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
})
PILOT_SCHEDULER_JOB_NAMES = MappingProxyType({
    "coupling-g0001": "nq1p-g0001",
    "coupling-g001": "nq1p-g001",
    "coupling-g0025": "nq1p-g0025",
})

RELEASE_018_ANCHOR_ROOT = Path(
    "/scratch/users/astar/ihpc/zhouhb/NEGFPy/0.3.0/"
    "nq1-iter800-957c3cb/runs/material-nq1-release-018"
)
RELEASE_018_COMMIT = "0d1f5bb298b7c267f58b5f444be1e7bedeca01cc"
RELEASE_018_TREE = "5f583806a5dd9090a7e4f248a90a31c0eb864abc"
RELEASE_018_CAMPAIGN_PAYLOAD_SHA256 = (
    "16483760bd51e74dc31dbeb28a7602c5aa40d430133b2afd5bae5687566ff1aa"
)

_ANCHOR_RECORDS = {
    "coupling-g0001": {
        "receipt": {
            "relative_path": "waves/000/task-artifacts/000000-f08a0fb1860940fb.json",
            "sha256": "69428ca804b7c047a81252a0992de8edd7bde453fa6986301a8bd5269ef2bb3b",
        },
        "fields": {
            "relative_path": (
                "waves/000/task-artifacts/000000-f08a0fb1860940fb.data/"
                "scba/outcomes/scba-fields.h5"
            ),
            "sha256": "73cdf3354e8d40c5f3c637b153ad3dc4b46e06cb81237b05ffed96543b2eb64a",
        },
        "outcome": {
            "relative_path": (
                "waves/000/task-artifacts/000000-f08a0fb1860940fb.data/"
                "scba/outcomes/scba.json"
            ),
            "sha256": "2c47037568abfd857c9f347ac7b4db8ac431e083420f7220c391a07e58a1ee0a",
        },
        "status": {
            "relative_path": (
                "waves/000/task-artifacts/000000-f08a0fb1860940fb.data/"
                "scba/status.json"
            ),
            "sha256": "401eda3579ce277ea553c321418522d6c6589e5b4a9cea6141d93e608c900217",
        },
        "report": {
            "relative_path": (
                "waves/000/task-artifacts/000000-f08a0fb1860940fb.data/"
                "scba/scba-report.json"
            ),
            "sha256": "177ab5602194b48b39ba6016fa14dacd48127c07a8f355264517e0e16310808d",
        },
    },
    "coupling-g001": {
        "receipt": {
            "relative_path": "waves/001/task-artifacts/000000-3193c021823f1778.json",
            "sha256": "e059ddaefe80670a1966d60655fb58792422ddd4db15bb32fabb1538a8b53960",
        },
        "fields": {
            "relative_path": (
                "waves/001/task-artifacts/000000-3193c021823f1778.data/"
                "scba/outcomes/scba-fields.h5"
            ),
            "sha256": "f7d0321e0f638428568c640b14ccc3d01d92f2f911b532fed501acedee9bd1e4",
        },
        "outcome": {
            "relative_path": (
                "waves/001/task-artifacts/000000-3193c021823f1778.data/"
                "scba/outcomes/scba.json"
            ),
            "sha256": "7d0d09fe9b85f7db5e10269f62d06af14d3d924809482c0be266d1b5b6f9a4f3",
        },
        "status": {
            "relative_path": (
                "waves/001/task-artifacts/000000-3193c021823f1778.data/"
                "scba/status.json"
            ),
            "sha256": "a4bb495278e6a5bd2f71e9327cdc71b5cc5cf1f95daedbedb5a007760bafb4b4",
        },
        "report": {
            "relative_path": (
                "waves/001/task-artifacts/000000-3193c021823f1778.data/"
                "scba/scba-report.json"
            ),
            "sha256": "1fceaf5547ff90447af56659e67e799806dcac1baa71f67505c7e1442092fe20",
        },
    },
}
_ANCHOR_TASK_SHA256 = {
    "coupling-g0001": "d76dad06c7fd83ce92c76a9a10e29f1f296eb5dabb85d1b89f2903394d29e189",
    "coupling-g001": "dd1adba3735db1deeb52e480ef28b94eaa43e936c225ae21975f8ed721afa260",
}
RELEASE_018_ANCHORS: Mapping[str, Mapping[str, Mapping[str, str]]] = (
    MappingProxyType({
        task_id: MappingProxyType({
            name: MappingProxyType(dict(record))
            for name, record in records.items()
        })
        for task_id, records in _ANCHOR_RECORDS.items()
    })
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_EXECUTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_PBS_JOB_ID = re.compile(r"^[0-9]+\.pbs101$")
_CORE_FILE_SHA256 = MappingProxyType({
    "src/negfpy/anharmonic/solver.py": (
        "51ef18cdb65e106c176631347e22bb543002ce9204e1703873d8d0016082c6a6"
    ),
    "src/negfpy/anharmonic/cubic.py": (
        "05105b95eaeca8ded94b1eab61b4a39ad41e85ea75e2593eaabf9f6d1a81bc42"
    ),
    "src/negfpy/anharmonic/_checkpoint_v2.py": (
        "0887298a9aa2021b52843082b51ffc998ba230376af3aae86be382fd97282a94"
    ),
})
_PROPERTY_ORACLE_SHA256 = (
    "74c893fc6974d4bd22036e539e4ab7ec11bb11c8a8e3805ac76e32cb373fb21b"
)
_QUALIFICATION_DRIVER_PATHS = (
    "pyproject.toml",
    "scripts/qualify_material_nq1_candidate.py",
    "scripts/qualify_material_nq1_predictor_candidate.py",
    "scripts/run_material_nq1_release_mpi.py",
    "scripts/validate_material_nq1_release.py",
    "scripts/validate_sparse_nq1_phase_c_scba.py",
    "tests/mpi/run_material_nq1_mpi_smoke.py",
)
_BASE_QUALIFICATION_CHECKS = frozenset({
    "source_clean_and_unchanged",
    "python_310_full_suite",
    "python_312_full_suite",
    "mpi_two_rank_smoke",
    "mpi_four_rank_smoke",
    "material_inputs_authenticated",
    "resolved_mode_set_nonempty",
    "harmonic_embedding_buffer",
    "memory_guard",
    "disk_guard",
    "thread_limits_one",
    "candidate_pythonpath_first",
})
_PREDICTOR_QUALIFICATION_CHECKS = frozenset({
    "full_scientific_controls_v2",
    "full_task_count_19",
    "predictor_mapping_8_square_4_identity_7_none",
    "pilot_exact_first_three_tasks",
    "pilot_anchor_contract_frozen",
    "pilot_anchor_files_deferred_to_preflight",
    "unchanged_core_solver",
    "unchanged_core_cubic",
    "unchanged_core_checkpoint_v2",
    "predictor_property_oracle_in_both_full_suites",
    "source_clean_and_unchanged_after_wrapper",
})


class MaterialNQ1PilotError(RuntimeError):
    """The pilot evidence is incomplete, mutable, or scientifically failed."""


def canonical_json(value: Any) -> bytes:
    """Return the repository's canonical newline-terminated JSON bytes."""

    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    selected = _regular_file(path, name="SHA256 input")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise MaterialNQ1PilotError(
            f"{name} must be a lowercase 64-character SHA256"
        )
    return value


def _non_symlink_directory_chain(
    path: str | Path, *, name: str, create: bool
) -> Path:
    lexical = Path(os.path.abspath(Path(path).expanduser()))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError:
            if not create:
                raise MaterialNQ1PilotError(f"{name} is unavailable: {lexical}")
            try:
                cursor.mkdir()
                mode = cursor.lstat().st_mode
            except OSError as error:
                raise MaterialNQ1PilotError(
                    f"cannot create safe {name}: {cursor}"
                ) from error
        except OSError as error:
            raise MaterialNQ1PilotError(f"{name} is unavailable: {cursor}") from error
        if stat.S_ISLNK(mode):
            raise MaterialNQ1PilotError(f"{name} traverses a symlink: {cursor}")
        if not stat.S_ISDIR(mode):
            raise MaterialNQ1PilotError(
                f"{name} contains a non-directory component: {cursor}"
            )
    return lexical


def _regular_file(path: str | Path, *, name: str) -> Path:
    selected = Path(path)
    _non_symlink_directory_chain(selected.parent, name=f"{name} parent", create=False)
    try:
        status = selected.lstat()
    except OSError as error:
        raise MaterialNQ1PilotError(f"{name} is unavailable: {selected}") from error
    if not stat.S_ISREG(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise MaterialNQ1PilotError(
            f"{name} must be a non-symlink regular file: {selected}"
        )
    return selected


def _directory(path: str | Path, *, name: str) -> Path:
    selected = Path(path)
    _non_symlink_directory_chain(selected.parent, name=f"{name} parent", create=False)
    try:
        status = selected.lstat()
    except OSError as error:
        raise MaterialNQ1PilotError(f"{name} is unavailable: {selected}") from error
    if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(status.st_mode):
        raise MaterialNQ1PilotError(
            f"{name} must be a non-symlink directory: {selected}"
        )
    return selected


def _require_fixed_absolute_path(
    path: str | Path, expected: str | Path, *, name: str
) -> Path:
    """Reject relative, normalized, or otherwise aliased evidence paths."""

    selected = Path(path)
    fixed = Path(expected)
    if (
        not fixed.is_absolute()
        or not selected.is_absolute()
        or os.fspath(selected) != os.fspath(fixed)
        or Path(os.path.abspath(selected)) != fixed
    ):
        raise MaterialNQ1PilotError(f"{name} path is not fixed")
    return selected


def load_canonical_json(path: str | Path, *, name: str) -> dict[str, Any]:
    selected = _regular_file(path, name=name)
    try:
        raw = selected.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MaterialNQ1PilotError(f"{name} is invalid JSON") from error
    if not isinstance(value, dict) or raw != canonical_json(value):
        raise MaterialNQ1PilotError(f"{name} is not canonical JSON")
    return value


def publish_json_once(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable canonical document, or verify exact idempotence."""

    target = Path(path)
    payload = canonical_json(dict(value))
    _non_symlink_directory_chain(
        target.parent, name="immutable JSON parent", create=True
    )
    if target.exists() or target.is_symlink():
        selected = _regular_file(target, name="immutable JSON")
        if selected.read_bytes() != payload:
            raise MaterialNQ1PilotError(f"immutable JSON differs: {target}")
        if stat.S_IMODE(selected.stat().st_mode) != 0o444:
            raise MaterialNQ1PilotError(
                f"immutable JSON must have mode 0444: {target}"
            )
        return target
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, target)
        except FileExistsError:
            selected = _regular_file(target, name="immutable JSON")
            if selected.read_bytes() != payload:
                raise MaterialNQ1PilotError(f"immutable JSON differs: {target}")
            if stat.S_IMODE(selected.stat().st_mode) != 0o444:
                raise MaterialNQ1PilotError(
                    f"immutable JSON must have mode 0444: {target}"
                )
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def predictor_pilot_tasks() -> tuple[MaterialSCBATask, ...]:
    """Return the exact three-task predictor pilot in dependency order."""

    tasks = tuple(silicon_nq1_release_tasks()[:3])
    if tuple(task.task_id for task in tasks) != PILOT_TASK_IDS:
        raise RuntimeError("pilot is not the exact first three release tasks")
    if (
        tasks[0].initialization != "zero"
        or tasks[1].continuation_parent != tasks[0].task_id
        or tasks[2].continuation_parent != tasks[1].task_id
        or tasks[1].continuation_predictor != "coupling-square"
        or tasks[2].continuation_predictor != "coupling-square"
    ):
        raise RuntimeError("release prefix does not use the frozen predictor chain")
    return tasks


def anchor_task_id(task_id: str) -> str | None:
    if task_id in PILOT_TASK_IDS[:2]:
        return task_id
    if task_id == PILOT_TASK_IDS[2]:
        return None
    raise MaterialNQ1PilotError(f"unknown pilot task: {task_id!r}")


def build_pilot_manifest(
    *,
    source_revision: str,
    source_git_tree: str,
    source_package_file: str | Path,
    source_tree_sha256: str,
    environment_sha256: str,
    qualification_sha256: str,
    input_hashes: Mapping[str, str],
    execution_id: str,
    anchor_root: str | Path = RELEASE_018_ANCHOR_ROOT,
) -> dict[str, Any]:
    """Build the frozen non-release pilot identity."""

    if not isinstance(source_revision, str) or _COMMIT.fullmatch(source_revision) is None:
        raise MaterialNQ1PilotError("source_revision must be a lowercase Git commit")
    if not isinstance(source_git_tree, str) or _COMMIT.fullmatch(source_git_tree) is None:
        raise MaterialNQ1PilotError("source_git_tree must be a lowercase Git tree")
    package_file = Path(source_package_file).expanduser()
    if (
        not package_file.is_absolute()
        or package_file.parts[-3:] != ("src", "negfpy", "__init__.py")
        or ".." in package_file.parts
    ):
        raise MaterialNQ1PilotError("source_package_file is not the exact package path")
    if not isinstance(execution_id, str) or _EXECUTION_ID.fullmatch(execution_id) is None:
        raise MaterialNQ1PilotError("execution_id is invalid")
    _validate_sha256(source_tree_sha256, name="source_tree_sha256")
    _validate_sha256(environment_sha256, name="environment_sha256")
    _validate_sha256(qualification_sha256, name="qualification_sha256")
    normalized_inputs: dict[str, str] = {}
    for name, digest in sorted(input_hashes.items()):
        if not isinstance(name, str) or not name:
            raise MaterialNQ1PilotError("input hash names must be nonempty strings")
        normalized_inputs[name] = _validate_sha256(
            digest, name=f"input_hashes[{name!r}]"
        )
    if not normalized_inputs:
        raise MaterialNQ1PilotError("input_hashes cannot be empty")
    required_inputs = {
        "phase_a_fixture",
        "raw_fc3",
        "phase_b_artifact_set",
        "phase_b_mapped_hdf5",
        "phase_b_campaign_manifest",
        "scba_case_driver",
    }
    if set(normalized_inputs) != required_inputs:
        raise MaterialNQ1PilotError("input_hashes do not match the exact pilot input set")
    root = Path(anchor_root).expanduser()
    if not root.is_absolute() or ".." in root.parts:
        raise MaterialNQ1PilotError("anchor_root must be an absolute lexical path")
    tasks = predictor_pilot_tasks()
    payload: dict[str, Any] = {
        "schema": PILOT_SCHEMA,
        "execution_id": execution_id,
        "scientific_scope": (
            "silicon N_q=1 coupling-square continuation pilot, g=.001 to .025"
        ),
        "classification": "development-pilot-not-release",
        "source_revision": source_revision,
        "source_git_tree": source_git_tree,
        "source_package_file": os.fspath(package_file),
        "source_tree_sha256": source_tree_sha256,
        "environment_sha256": environment_sha256,
        "qualification_sha256": qualification_sha256,
        "input_hashes": normalized_inputs,
        "solver_contract": {
            "backend": "fft",
            "hilbert_backend": "fft",
            "eta_device": 0.0,
            "max_workers": 1,
            "max_iterations": 800,
            "min_iterations": 3,
            "consecutive_passes": 3,
            "on_nonconvergence": "raise",
            "automatic_retry": False,
            "controls_changed_after_outcome": False,
        },
        "tasks": [
            task.as_dict()
            | {
                "task_sha256": task.task_sha256,
                "dependencies": (
                    []
                    if task.continuation_parent is None
                    else [task.continuation_parent]
                ),
                "anchor_task_id": anchor_task_id(task.task_id),
                "scheduler_job_name": PILOT_SCHEDULER_JOB_NAMES[task.task_id],
            }
            for task in tasks
        ],
        "acceptance": {
            **dict(MATERIAL_NQ1_ACCEPTANCE),
            "anchor_field_rtol": FIELD_RTOL,
            "anchor_field_atol": FIELD_ATOL,
            "anchor_current_rtol": CURRENT_RTOL,
            "anchor_current_atol": CURRENT_ATOL,
        },
        "anchor_source": {
            "root": os.fspath(root),
            "source_revision": RELEASE_018_COMMIT,
            "source_tree_sha256": RELEASE_018_TREE,
            "campaign_payload_sha256": RELEASE_018_CAMPAIGN_PAYLOAD_SHA256,
            "records": {
                task_id: {
                    name: dict(record)
                    for name, record in records.items()
                }
                for task_id, records in RELEASE_018_ANCHORS.items()
            },
            "use": "comparison-only-never-continuation-parent",
        },
        "excluded_claims": list(MATERIAL_NQ1_EXCLUDED_CLAIMS),
        "release_certified": False,
    }
    return payload | {"pilot_sha256": sha256_bytes(canonical_json(payload))}


def validate_pilot_manifest(document: Mapping[str, Any]) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "execution_id",
        "scientific_scope",
        "classification",
        "source_revision",
        "source_git_tree",
        "source_package_file",
        "source_tree_sha256",
        "environment_sha256",
        "qualification_sha256",
        "input_hashes",
        "solver_contract",
        "tasks",
        "acceptance",
        "anchor_source",
        "excluded_claims",
        "release_certified",
        "pilot_sha256",
    }
    if set(document) != expected_keys or document.get("schema") != PILOT_SCHEMA:
        raise MaterialNQ1PilotError("pilot manifest schema is invalid")
    if document.get("classification") != "development-pilot-not-release":
        raise MaterialNQ1PilotError("pilot classification changed")
    if document.get("release_certified") is not False:
        raise MaterialNQ1PilotError("a predictor pilot cannot certify a release")
    pilot_sha = _validate_sha256(document.get("pilot_sha256"), name="pilot_sha256")
    payload = {name: document[name] for name in expected_keys - {"pilot_sha256"}}
    if sha256_bytes(canonical_json(payload)) != pilot_sha:
        raise MaterialNQ1PilotError("pilot manifest payload hash changed")
    rebuilt = build_pilot_manifest(
        source_revision=str(document.get("source_revision")),
        source_git_tree=str(document.get("source_git_tree")),
        source_package_file=str(document.get("source_package_file")),
        source_tree_sha256=str(document.get("source_tree_sha256")),
        environment_sha256=str(document.get("environment_sha256")),
        qualification_sha256=str(document.get("qualification_sha256")),
        input_hashes=document.get("input_hashes", {}),
        execution_id=str(document.get("execution_id")),
        anchor_root=str(document.get("anchor_source", {}).get("root")),
    )
    if dict(document) != rebuilt:
        raise MaterialNQ1PilotError("pilot manifest does not reproduce")
    return rebuilt


def _anchor_path(
    root: Path,
    record: Mapping[str, Any],
    *,
    name: str,
) -> Path:
    relative = record.get("relative_path")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise MaterialNQ1PilotError(f"{name} has an invalid relative path")
    path = root / relative
    _regular_file(path, name=name)
    expected = _validate_sha256(record.get("sha256"), name=f"{name} SHA256")
    if sha256_file(path) != expected:
        raise MaterialNQ1PilotError(f"{name} SHA256 changed")
    return path


def authenticate_anchor_source(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Authenticate every fixed release-018 comparison anchor."""

    validated = validate_pilot_manifest(manifest)
    source = validated["anchor_source"]
    root = _directory(source["root"], name="release-018 anchor root")
    results: dict[str, dict[str, Any]] = {}
    for anchor_id in ("coupling-g0001", "coupling-g001"):
        records = source["records"][anchor_id]
        paths = {
            name: _anchor_path(root, record, name=f"{anchor_id} {name}")
            for name, record in records.items()
        }
        receipt = load_canonical_json(paths["receipt"], name=f"{anchor_id} receipt")
        outcome = load_canonical_json(paths["outcome"], name=f"{anchor_id} outcome")
        status = load_canonical_json(paths["status"], name=f"{anchor_id} status")
        report = load_canonical_json(paths["report"], name=f"{anchor_id} report")
        if (
            receipt.get("task_id") != anchor_id
            or receipt.get("state") != "passed"
            or receipt.get("campaign_manifest_sha256")
            != RELEASE_018_CAMPAIGN_PAYLOAD_SHA256
            or receipt.get("scientific_task_sha256")
            != _ANCHOR_TASK_SHA256[anchor_id]
        ):
            raise MaterialNQ1PilotError(f"{anchor_id} receipt is not a passed anchor")
        if (
            status.get("state") != "passed"
            or status.get("scba_converged") is not True
            or report.get("state") != "passed"
            or report.get("scba_converged") is not True
            or outcome.get("result", {}).get("converged") is not True
            or outcome.get("release_certified") is not False
            or outcome.get("result", {}).get("release_certified") is not False
        ):
            raise MaterialNQ1PilotError(f"{anchor_id} SCBA evidence is not converged")
        if (
            receipt.get("release_certified") is not False
            or status.get("release_certified") is not False
            or report.get("release_certified") is not False
        ):
            raise MaterialNQ1PilotError(
                f"{anchor_id} evidence misstates task-level release certification"
            )
        if report.get("fields_hdf5_sha256") != records["fields"]["sha256"]:
            raise MaterialNQ1PilotError(f"{anchor_id} report does not bind fields")
        if report.get("outcome_sha256") != records["outcome"]["sha256"]:
            raise MaterialNQ1PilotError(f"{anchor_id} report does not bind outcome")
        if status.get("report_sha256") != records["report"]["sha256"]:
            raise MaterialNQ1PilotError(f"{anchor_id} status does not bind report")
        tree_records = receipt.get("task_tree_records")
        expected_tree_records = {
            "scba/outcomes/scba-fields.h5": records["fields"]["sha256"],
            "scba/outcomes/scba.json": records["outcome"]["sha256"],
            "scba/status.json": records["status"]["sha256"],
            "scba/scba-report.json": records["report"]["sha256"],
        }
        if (
            not isinstance(tree_records, Mapping)
            or any(tree_records.get(name) != digest for name, digest in expected_tree_records.items())
        ):
            raise MaterialNQ1PilotError(
                f"{anchor_id} receipt tree does not bind every frozen anchor"
            )
        scba_root = paths["outcome"].parents[1]
        scba_manifest_path = _regular_file(
            scba_root / "campaign-manifest.json",
            name=f"{anchor_id} SCBA campaign manifest",
        )
        scba_manifest_sha = sha256_file(scba_manifest_path)
        if tree_records.get("scba/campaign-manifest.json") != scba_manifest_sha:
            raise MaterialNQ1PilotError(
                f"{anchor_id} receipt does not bind its SCBA manifest"
            )
        if (
            outcome.get("campaign_manifest_sha256") != scba_manifest_sha
            or status.get("campaign_manifest_sha256") != scba_manifest_sha
            or report.get("campaign_manifest_sha256") != scba_manifest_sha
            or outcome.get("hdf5", {}).get("sha256")
            != records["fields"]["sha256"]
            or report.get("checkpoint_sha256")
            != receipt.get("checkpoint_sha256")
        ):
            raise MaterialNQ1PilotError(
                f"{anchor_id} nested campaign bindings disagree"
            )
        results[anchor_id] = {
            "paths": {name: os.fspath(path) for name, path in paths.items()},
            "hashes": {
                name: records[name]["sha256"] for name in sorted(records)
            },
            "conservation": conservation_record(outcome),
            "passed": True,
        }
    return results


def conservation_record(outcome: Mapping[str, Any]) -> dict[str, Any]:
    """Evaluate the frozen contact/collision conservation criterion."""

    try:
        result = outcome["result"]
        currents = result["currents"]
        left = float(currents["left"])
        right = float(currents["right"])
        collision = float(currents["collision_power"])
        current_scale = float(result["solver_metadata"]["current_scale"])
        diagnostics_passed = result["diagnostics"]["passed"] is True
    except (KeyError, TypeError, ValueError) as error:
        raise MaterialNQ1PilotError("SCBA outcome lacks conservation evidence") from error
    if not all(math.isfinite(value) for value in (left, right, collision, current_scale)):
        raise MaterialNQ1PilotError("SCBA current evidence is non-finite")
    if current_scale <= 0.0:
        raise MaterialNQ1PilotError("SCBA current scale must be positive")
    tolerance = max(
        float(MATERIAL_NQ1_ACCEPTANCE["current_atol_j0"]) * current_scale,
        float(MATERIAL_NQ1_ACCEPTANCE["conservation_rtol"])
        * max(abs(left), abs(right)),
    )
    contact_error = abs(left + right)
    collision_error = abs(collision)
    passed = bool(
        diagnostics_passed
        and contact_error <= tolerance
        and collision_error <= tolerance
    )
    return {
        "left_current_w": left,
        "right_current_w": right,
        "collision_power_w": collision,
        "current_scale_w": current_scale,
        "tolerance_w": tolerance,
        "contact_imbalance_w": contact_error,
        "collision_power_abs_w": collision_error,
        "diagnostics_passed": diagnostics_passed,
        "passed": passed,
    }


def compare_fields(
    first: str | Path,
    second: str | Path,
    *,
    rtol: float = FIELD_RTOL,
    atol: float = FIELD_ATOL,
) -> dict[str, Any]:
    """Stream and compare the exact grid and all six Keldysh fields."""

    left_path = _regular_file(first, name="first field archive")
    right_path = _regular_file(second, name="second field archive")
    if not math.isfinite(rtol) or not math.isfinite(atol) or rtol < 0 or atol < 0:
        raise MaterialNQ1PilotError("field tolerances must be finite and nonnegative")
    datasets: dict[str, Any] = {}
    passed = True
    try:
        with h5py.File(left_path, "r") as left, h5py.File(right_path, "r") as right:
            if "grid/omega" not in left or "grid/omega" not in right:
                raise MaterialNQ1PilotError("field archive lacks grid/omega")
            left_grid = np.asarray(left["grid/omega"])
            right_grid = np.asarray(right["grid/omega"])
            grid_equal = bool(
                left_grid.shape == right_grid.shape
                and left_grid.dtype == right_grid.dtype
                and np.array_equal(left_grid, right_grid)
            )
            passed = passed and grid_equal
            for name in FIELD_DATASETS:
                if name not in left or name not in right:
                    datasets[name] = {"present": False, "passed": False}
                    passed = False
                    continue
                a = left[name]
                b = right[name]
                shape_equal = a.shape == b.shape == EXPECTED_FIELD_SHAPE
                dtype_equal = (
                    a.dtype == b.dtype == np.dtype(EXPECTED_FIELD_DTYPE)
                )
                if not shape_equal or not dtype_equal:
                    datasets[name] = {
                        "present": True,
                        "shape_equal": shape_equal,
                        "dtype_equal": dtype_equal,
                        "first_shape": list(a.shape),
                        "second_shape": list(b.shape),
                        "first_dtype": str(a.dtype),
                        "second_dtype": str(b.dtype),
                        "passed": False,
                    }
                    passed = False
                    continue
                maximum_absolute = 0.0
                maximum_allowed_ratio = 0.0
                finite = True
                for index in range(a.shape[0]):
                    av = np.asarray(a[index])
                    bv = np.asarray(b[index])
                    finite = finite and bool(
                        np.all(np.isfinite(av)) and np.all(np.isfinite(bv))
                    )
                    difference = np.abs(av - bv)
                    allowed = atol + rtol * np.maximum(np.abs(av), np.abs(bv))
                    maximum_absolute = max(
                        maximum_absolute, float(np.max(difference))
                    )
                    maximum_allowed_ratio = max(
                        maximum_allowed_ratio,
                        float(np.max(difference / allowed)),
                    )
                item_passed = bool(finite and maximum_allowed_ratio <= 1.0)
                datasets[name] = {
                    "present": True,
                    "shape_equal": True,
                    "dtype_equal": True,
                    "finite": finite,
                    "maximum_absolute_error": maximum_absolute,
                    "maximum_allowed_ratio": maximum_allowed_ratio,
                    "passed": item_passed,
                }
                passed = passed and item_passed
    except OSError as error:
        raise MaterialNQ1PilotError("cannot read field archive") from error
    return {
        "grid_equal": grid_equal,
        "rtol": rtol,
        "atol": atol,
        "datasets": datasets,
        "passed": bool(passed),
    }


def compare_currents(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    rtol: float = CURRENT_RTOL,
    atol: float = CURRENT_ATOL,
) -> dict[str, Any]:
    first_conservation = conservation_record(first)
    second_conservation = conservation_record(second)
    comparisons: dict[str, Any] = {}
    passed = first_conservation["passed"] and second_conservation["passed"]
    for name, key in (
        ("left", "left_current_w"),
        ("right", "right_current_w"),
        ("collision_power", "collision_power_w"),
    ):
        left = float(first_conservation[key])
        right = float(second_conservation[key])
        difference = abs(left - right)
        allowed = atol + rtol * max(abs(left), abs(right))
        ratio = difference / allowed
        item_passed = ratio <= 1.0
        comparisons[name] = {
            "first": left,
            "second": right,
            "absolute_error": difference,
            "maximum_allowed_ratio": ratio,
            "passed": item_passed,
        }
        passed = passed and item_passed
    return {
        "rtol": rtol,
        "atol": atol,
        "values": comparisons,
        "first_conservation": first_conservation,
        "second_conservation": second_conservation,
        "passed": bool(passed),
    }


def _tree_records(root: Path) -> dict[str, str]:
    _directory(root, name="SCBA evidence directory")
    records: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        status_result = path.lstat()
        if stat.S_ISLNK(status_result.st_mode):
            raise MaterialNQ1PilotError(f"SCBA evidence contains a symlink: {path}")
        if stat.S_ISREG(status_result.st_mode):
            records[path.relative_to(root).as_posix()] = sha256_file(path)
        elif not stat.S_ISDIR(status_result.st_mode):
            raise MaterialNQ1PilotError(
                f"SCBA evidence contains a special file: {path}"
            )
    if not records:
        raise MaterialNQ1PilotError("SCBA evidence directory is empty")
    return records


def _tree_sha256(records: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json(dict(sorted(records.items()))))


def _initial_pair_sha256(lesser: Any, greater: Any) -> str:
    """Reproduce ``initial_self_energy_hash`` without a mutable problem object."""

    left = np.ascontiguousarray(np.asarray(lesser))
    right = np.ascontiguousarray(np.asarray(greater))
    if left.shape != right.shape or len(left.shape) != 3:
        raise MaterialNQ1PilotError("continuation checkpoint pair shape is invalid")
    if left.dtype != np.dtype(np.complex128) or right.dtype != np.dtype(np.complex128):
        raise MaterialNQ1PilotError("continuation checkpoint pair dtype is invalid")
    digest = hashlib.sha256()
    digest.update(repr(tuple(int(value) for value in left.shape)).encode("ascii"))
    digest.update(b"supplied_keldysh_pair")
    for array in (left, right):
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(repr(tuple(int(value) for value in array.shape)).encode("ascii"))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _explicit_zero_initial_sha256(shape: Sequence[int]) -> str:
    normalized = tuple(int(value) for value in shape)
    digest = hashlib.sha256()
    digest.update(repr(normalized).encode("ascii"))
    digest.update(b"explicit_zero_initial_self_energy")
    return digest.hexdigest()


def _task_from_manifest(manifest: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["tasks"] if item.get("task_id") == task_id]
    if len(matches) != 1:
        raise MaterialNQ1PilotError(f"pilot manifest lacks one task {task_id!r}")
    return dict(matches[0])


def _stable_material_identity(value: Any, *, name: str) -> dict[str, Any]:
    """Normalize only the independently observed available-memory value."""

    if not isinstance(value, Mapping):
        raise MaterialNQ1PilotError(f"{name} is not a material identity")
    try:
        stable = json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise MaterialNQ1PilotError(f"{name} is not JSON-compatible") from error
    memory = stable.get("memory")
    if not isinstance(memory, dict) or set(memory) != {
        "available_bytes", "predicted_peak_bytes", "maximum_fraction",
        "guard_passed",
    }:
        raise MaterialNQ1PilotError(f"{name} memory identity is incomplete")
    available = memory.get("available_bytes")
    if isinstance(available, bool) or not isinstance(available, int) or available <= 0:
        raise MaterialNQ1PilotError(
            f"{name} available-memory observation is invalid"
        )
    memory["available_bytes"] = "runtime-observation"
    return stable


def qualified_python312_runtime(
    manifest: Mapping[str, Any], qualification: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the exact P312 runtime and lock from an authorized v3 report."""

    validated = validate_pilot_manifest(manifest)
    if (
        sha256_bytes(canonical_json(dict(qualification)))
        != validated["qualification_sha256"]
        or not qualification_authorizes_pilot(validated, qualification)
    ):
        raise MaterialNQ1PilotError("qualification does not authorize this pilot")
    try:
        base = qualification["evidence"]["base_qualification_evidence"]
        runtime = base["records"]["python312_runtime"]
        context = base["qualification_context"]
        context_runtime = context["runtimes"]["python312"]
        environment_lock = context["environment_lock"]
    except (KeyError, TypeError) as error:
        raise MaterialNQ1PilotError(
            "qualification lacks the bound P312 runtime"
        ) from error
    runtime_payload = {
        name: runtime[name]
        for name in (
            "schema", "label", "version", "executable",
            "executable_sha256", "packages_sha256",
        )
    } if isinstance(runtime, Mapping) and all(
        name in runtime for name in (
            "schema", "label", "version", "executable",
            "executable_sha256", "packages_sha256",
        )
    ) else None
    if (
        not isinstance(runtime, Mapping)
        or runtime != context_runtime
        or runtime_payload is None
        or runtime.get("schema") != "negfpy.material-nq1-python-runtime.v1"
        or runtime.get("label") != "python312"
        or runtime.get("version") != PILOT_PYTHON_VERSION
        or sha256_bytes(canonical_json(runtime_payload))
        != runtime.get("runtime_sha256")
        or not isinstance(environment_lock, Mapping)
        or set(environment_lock) != {"path", "sha256", "format"}
        or environment_lock.get("sha256") != validated["environment_sha256"]
        or environment_lock.get("format") != "@EXPLICIT"
    ):
        raise MaterialNQ1PilotError("qualified P312 runtime contract changed")
    executable = _regular_file(
        runtime.get("executable"), name="qualified P312 executable"
    )
    if (
        os.fspath(executable) != runtime.get("executable")
        or sha256_file(executable) != runtime.get("executable_sha256")
    ):
        raise MaterialNQ1PilotError("qualified P312 executable changed")
    lock_path = _regular_file(
        environment_lock.get("path"), name="qualified environment lock"
    )
    if (
        os.fspath(lock_path) != environment_lock.get("path")
        or sha256_file(lock_path) != environment_lock.get("sha256")
    ):
        raise MaterialNQ1PilotError("qualified environment lock changed")
    return dict(runtime), dict(environment_lock)


def _fixed_task_layout(scba_root: str | Path, task_id: str) -> tuple[Path, Path]:
    root = _directory(scba_root, name=f"{task_id} SCBA root")
    root = Path(os.path.abspath(root))
    if root.name != "scba" or root.parent.name != task_id:
        raise MaterialNQ1PilotError("SCBA root is outside the fixed pilot task layout")
    pilot_root = root.parent.parent.parent
    expected = pilot_root / "tasks" / task_id / "scba"
    if root != expected:
        raise MaterialNQ1PilotError("SCBA root is outside the fixed pilot task layout")
    _directory(pilot_root, name="pilot root")
    return root, pilot_root


def _bound_qualification_contract(
    manifest: Mapping[str, Any], pilot_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the qualification only through the fixed, hash-bound preflight."""

    validated = validate_pilot_manifest(manifest)
    manifest_path = _regular_file(
        pilot_root / "pilot-manifest.json", name="bound pilot manifest"
    )
    if (
        stat.S_IMODE(manifest_path.stat().st_mode) != 0o444
        or load_canonical_json(manifest_path, name="bound pilot manifest") != validated
    ):
        raise MaterialNQ1PilotError("fixed pilot manifest binding changed")
    preflight_path = _regular_file(
        pilot_root / "pilot-preflight.json", name="bound pilot preflight"
    )
    if stat.S_IMODE(preflight_path.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("bound pilot preflight must have mode 0444")
    preflight = validate_preflight_record(validated, preflight_path)
    manifest_record = preflight.get("manifest")
    qualification_record = preflight.get("qualification")
    if (
        preflight.get("schema") != PREFLIGHT_SCHEMA
        or preflight.get("pilot_sha256") != validated["pilot_sha256"]
        or preflight.get("passed") is not True
        or preflight.get("release_certified") is not False
        or not isinstance(manifest_record, Mapping)
        or manifest_record != {
            "path": os.fspath(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
        or not isinstance(qualification_record, Mapping)
        or qualification_record.get("sha256") != validated["qualification_sha256"]
    ):
        raise MaterialNQ1PilotError("fixed pilot preflight binding changed")
    qualification_path = _regular_file(
        qualification_record.get("path"), name="bound qualification report"
    )
    if (
        sha256_file(qualification_path) != qualification_record.get("sha256")
        or stat.S_IMODE(qualification_path.stat().st_mode) != 0o444
    ):
        raise MaterialNQ1PilotError("bound qualification report changed")
    qualification = load_canonical_json(
        qualification_path, name="bound qualification report"
    )
    return qualified_python312_runtime(validated, qualification)


def _expected_scba_attempt_command(
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    scba_manifest: Mapping[str, Any],
    root: Path,
    runtime: Mapping[str, Any],
) -> list[str]:
    source = scba_manifest["preflight"]["source"]
    repo = Path(manifest["source_package_file"]).parents[2]
    outputs = scba_manifest["outputs"]
    numerical = scba_manifest["numerical_controls"]
    command = [
        str(runtime["executable"]),
        os.fspath(repo / "scripts/validate_sparse_nq1_phase_c_scba.py"),
        "--fixture", str(source["fixture_path"]),
        "--fc3-full", str(source["fc3_full"]["path"]),
        "--phase-b-artifacts", str(source["phase_b"]["root"]),
        "--checkpoint", str(outputs["checkpoint"]),
        "--output-json", str(outputs["json"]),
        "--output-hdf5", str(outputs["hdf5"]),
        "--campaign-manifest-sha256", sha256_file(root / "campaign-manifest.json"),
        "--source-tree-sha256", str(scba_manifest["source_tree_sha256"]),
        "--environment-sha256", str(scba_manifest["environment_lock"]["sha256"]),
        "--run-id", str(scba_manifest["run_id"]),
        "--coupling-scale", str(scba_manifest["coupling_scale"]),
        "--grid-step", str(numerical["grid_step"]),
        "--grid-n-positive", str(numerical["grid_n_positive"]),
        "--eta-lead", str(numerical["eta_lead"]),
        "--mixing", str(numerical["mixing"]),
        "--mixing-floor", str(numerical["mixing_floor"]),
        "--temperature-bias", str(numerical["temperature_bias"]),
    ]
    parent = scba_manifest.get("continuation_parent")
    if parent is not None:
        command.extend([
            "--continuation-parent-artifacts", str(parent["artifacts"]),
            "--continuation-axis", str(task["continuation_axis"]),
            "--continuation-predictor", str(task["continuation_predictor"]),
        ])
    command.append("--no-resume")
    return command


def _expected_scba_sums(root: Path) -> bytes:
    names = (
        "campaign-manifest.json",
        "status-running.json",
        "outcomes/scba.json",
        "outcomes/scba-fields.h5",
        "checkpoint/sigma-pp.h5",
        "scba-report.json",
        "logs/scba.json",
    )
    return "".join(
        f"{sha256_file(root / name)}  {name}\n" for name in sorted(names)
    ).encode("ascii")


def _hdf5_metadata(path: Path) -> dict[str, Any]:
    try:
        with h5py.File(path, "r") as handle:
            if (
                handle.attrs.get("schema")
                != "negfpy.sparse-nq1-phase-c-scba-fields.v1"
                or bool(handle.attrs.get("scba_executed")) is not True
                or bool(handle.attrs.get("release_certified")) is not False
                or "metadata_json" not in handle
            ):
                raise MaterialNQ1PilotError("SCBA field archive metadata is invalid")
            raw = np.asarray(handle["metadata_json"], dtype=np.uint8).tobytes()
    except OSError as error:
        raise MaterialNQ1PilotError("cannot read SCBA field archive metadata") from error
    try:
        value = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MaterialNQ1PilotError("SCBA HDF5 metadata is invalid JSON") from error
    if not isinstance(value, dict) or raw != canonical_json(value):
        raise MaterialNQ1PilotError("SCBA HDF5 metadata is not canonical JSON")
    return value


def _array_sha256(value: Any) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(value)).tobytes(order="C")
    ).hexdigest()


def _validate_result_arrays(
    path: Path,
    outcome: Mapping[str, Any],
    checkpoint: Any,
    task: Mapping[str, Any],
) -> None:
    result = outcome.get("result")
    hashes = result.get("array_hashes") if isinstance(result, Mapping) else None
    expected_names = {"grid/omega"} | {
        f"{spectrum}/{component}"
        for spectrum in ("green", "sigma_pp", "sigma_left", "sigma_right")
        for component in ("retarded", "lesser", "greater")
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        raise MaterialNQ1PilotError("SCBA result array-hash inventory changed")
    try:
        with h5py.File(path, "r") as handle:
            for name in sorted(expected_names):
                if name not in handle:
                    raise MaterialNQ1PilotError(
                        f"SCBA field archive lacks result array {name!r}"
                    )
                values = np.asarray(handle[name])
                if _array_sha256(values) != hashes[name]:
                    raise MaterialNQ1PilotError(
                        f"SCBA result array hash changed for {name!r}"
                    )
                if name == "grid/omega":
                    expected_n_positive = (EXPECTED_FIELD_SHAPE[0] - 1) // 2
                    expected_grid = float(task["grid_step"]) * np.arange(
                        -expected_n_positive,
                        expected_n_positive + 1,
                        dtype=np.int64,
                    )
                    if (
                        values.shape != (EXPECTED_FIELD_SHAPE[0],)
                        or values.dtype != np.dtype(np.float64)
                        or not np.array_equal(values, expected_grid)
                    ):
                        raise MaterialNQ1PilotError("SCBA frequency grid shape changed")
                elif (
                    values.shape != EXPECTED_FIELD_SHAPE
                    or values.dtype != np.dtype(EXPECTED_FIELD_DTYPE)
                    or not np.all(np.isfinite(values))
                ):
                    raise MaterialNQ1PilotError(
                        f"SCBA result array contract changed for {name!r}"
                    )
            if (
                not np.array_equal(
                    np.asarray(handle["sigma_pp/lesser"]),
                    np.asarray(checkpoint.sigma_lesser),
                )
                or not np.array_equal(
                    np.asarray(handle["sigma_pp/greater"]),
                    np.asarray(checkpoint.sigma_greater),
                )
            ):
                raise MaterialNQ1PilotError(
                    "terminal checkpoint and field archive self-energies disagree"
                )
    except OSError as error:
        raise MaterialNQ1PilotError("cannot authenticate SCBA result arrays") from error


def _replay_scientific_diagnostics(
    *,
    scba_manifest: Mapping[str, Any],
    outcome: Mapping[str, Any],
    fields_path: Path,
    checkpoint: Any,
) -> None:
    """Rebuild the exact problem/result and independently rerun diagnostics."""

    try:
        import validate_sparse_nq1_phase_c_scba as case

        reconstructed_identity, problem, config = case._reconstruct_parent_inputs(
            scba_manifest
        )
        recorded_source = scba_manifest["preflight"]["source"]
        if case._stable_material_identity(
            reconstructed_identity
        ) != case._stable_material_identity(recorded_source):
            raise MaterialNQ1PilotError(
                "SCBA material identity does not independently reconstruct"
            )
        if (
            case.configuration_hash(config)
            != scba_manifest["preflight"]["configuration_sha256"]
            or case.problem_input_hashes(problem)
            != recorded_source["problem_input_hashes"]
            or case._plain(config) != scba_manifest["preflight"]["controls"]
        ):
            raise MaterialNQ1PilotError(
                "SCBA problem/configuration does not independently reconstruct"
            )
        with h5py.File(fields_path, "r") as handle:
            spectra = {
                name: KeldyshSpectrum(
                    retarded=np.asarray(handle[f"{name}/retarded"]),
                    lesser=np.asarray(handle[f"{name}/lesser"]),
                    greater=np.asarray(handle[f"{name}/greater"]),
                )
                for name in ("green", "sigma_pp", "sigma_left", "sigma_right")
            }
        currents = _evaluate_current_report(
            problem,
            spectra["sigma_left"],
            spectra["sigma_right"],
            spectra["sigma_pp"],
            spectra["green"],
        )
        summary = outcome["result"]
        provisional = AnharmonicResult(
            method=summary["method"],
            grid=problem.grid,
            green=spectra["green"],
            sigma_pp=spectra["sigma_pp"],
            sigma_left=spectra["sigma_left"],
            sigma_right=spectra["sigma_right"],
            currents=currents,
            diagnostics=DiagnosticReport(
                metrics={},
                failures=("independent replay pending",),
                passed=False,
                release_certified=False,
            ),
            history=tuple(checkpoint.history),
            converged=bool(summary["converged"]),
            release_certified=False,
            metadata=dict(summary["solver_metadata"]),
        )
        replayed = check_solution(problem, provisional, config)
    except MaterialNQ1PilotError:
        raise
    except Exception as error:
        raise MaterialNQ1PilotError(
            "SCBA problem/result could not be independently reconstructed"
        ) from error
    converged = summary.get("converged") is True
    expected_failure_semantics = (
        replayed.passed is True
        and all(str(failure).startswith("release:") for failure in replayed.failures)
        if converged
        else replayed.passed is False
        and any(
            not str(failure).startswith("release:") for failure in replayed.failures
        )
    )
    if (
        case._plain(currents) != summary.get("currents")
        or case._plain(replayed) != summary.get("diagnostics")
        or not expected_failure_semantics
    ):
        raise MaterialNQ1PilotError(
            "independently replayed scientific diagnostics do not pass or reproduce"
        )


def _validate_supervision(
    scba_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    converged: bool,
) -> dict[str, Any]:
    watchdog = scba_manifest.get("watchdog")
    supervision = report.get("supervision")
    if not isinstance(watchdog, Mapping) or not isinstance(supervision, Mapping):
        raise MaterialNQ1PilotError("SCBA supervision evidence is missing")
    if (
        set(watchdog) != {
            "wall_limit_seconds", "rss_limit_bytes", "term_grace_seconds",
            "automatic_retry",
        }
        or set(supervision) != {
            "returncode", "elapsed_seconds", "peak_observed_rss_bytes",
            "watchdog_trigger", "killed_after_term_grace",
            "wall_limit_seconds", "rss_limit_bytes",
        }
        or watchdog.get("wall_limit_seconds") != PILOT_WALL_LIMIT_SECONDS
        or watchdog.get("rss_limit_bytes") != PILOT_RSS_LIMIT_BYTES
        or watchdog.get("term_grace_seconds") != 60
        or watchdog.get("automatic_retry") is not False
        or supervision.get("wall_limit_seconds") != PILOT_WALL_LIMIT_SECONDS
        or supervision.get("rss_limit_bytes") != PILOT_RSS_LIMIT_BYTES
        or supervision.get("watchdog_trigger") is not None
        or supervision.get("returncode") != (0 if converged else 2)
        or supervision.get("killed_after_term_grace") is not False
    ):
        raise MaterialNQ1PilotError("SCBA supervision contract changed or failed")
    elapsed = supervision.get("elapsed_seconds")
    peak = supervision.get("peak_observed_rss_bytes")
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or not 0.0 <= float(elapsed) <= PILOT_WALL_LIMIT_SECONDS
        or isinstance(peak, bool)
        or not isinstance(peak, int)
        or not 0 <= peak <= PILOT_RSS_LIMIT_BYTES
    ):
        raise MaterialNQ1PilotError("SCBA supervision resources exceed the pilot limits")
    return dict(supervision)


def _validate_scientific_result(
    outcome: Mapping[str, Any],
    checkpoint: Any,
    task: Mapping[str, Any],
    *,
    converged: bool,
) -> None:
    result = outcome.get("result")
    if not isinstance(result, Mapping):
        raise MaterialNQ1PilotError("SCBA outcome lacks a terminal result")
    metadata = result.get("solver_metadata")
    diagnostics = result.get("diagnostics")
    currents = result.get("currents")
    if (
        set(result) != {
            "method", "converged", "history_length", "release_certified",
            "array_hashes", "currents", "diagnostics", "solver_metadata",
            "scientific_error",
        }
        or not isinstance(metadata, Mapping)
        or not isinstance(diagnostics, Mapping)
        or set(diagnostics) != {"metrics", "failures", "passed", "release_certified"}
        or not isinstance(currents, Mapping)
        or set(currents) != {
            "left", "right", "collision_power", "contact_spectra",
            "collision_spectrum", "bond_currents", "bond_spectra", "spatial",
        }
    ):
        raise MaterialNQ1PilotError("SCBA result lacks metadata or diagnostics")
    expected_metadata = {
        "max_iterations": 800,
        "min_iterations": 3,
        "consecutive_passes": 3,
        "mixing_initial": task["mixing"],
        "mixing_floor": task["mixing_floor"],
        "eta_lead": task["eta_lead"],
        "eta_device": 0.0,
        "backend": "fft",
        "effective_backend": "fft",
        "hilbert_backend": "fft",
        "grid_step": task["grid_step"],
        "grid_n_positive": task["grid_n_positive"],
        "grid_cutoff": task["grid_cutoff"],
        "dtype": "complex128",
    }
    if any(metadata.get(name) != value for name, value in expected_metadata.items()):
        raise MaterialNQ1PilotError("SCBA terminal solver metadata changed")
    if result.get("method") != "scba" or result.get("release_certified") is not False:
        raise MaterialNQ1PilotError("SCBA terminal result semantics changed")
    if result.get("history_length") != len(checkpoint.history):
        raise MaterialNQ1PilotError("SCBA outcome/checkpoint history lengths disagree")
    failures = diagnostics.get("failures")
    metrics = diagnostics.get("metrics")
    if (
        not isinstance(failures, list)
        or not all(isinstance(item, str) and item for item in failures)
        or not isinstance(metrics, Mapping)
        or not metrics
        or not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            for value in metrics.values()
        )
    ):
        raise MaterialNQ1PilotError("SCBA diagnostic evidence is malformed")
    required_metrics = {
        "history_last_consecutive_passes",
        "history_last_green_residual",
        "history_last_sigma_residual",
        "current_contact_abs_error",
        "current_collision_abs_error",
        "current_tolerance",
        "sigma_pp_tail_ratio",
        "equilibrium_fdt_applicable",
        "package_version_consistent",
    }
    if not required_metrics.issubset(metrics):
        raise MaterialNQ1PilotError("SCBA diagnostics lack frozen essential metrics")
    if converged:
        final = tuple(checkpoint.history[-3:])
        last = final[-1] if final else None
        conservation = conservation_record(outcome)
        if (
            result.get("scientific_error") is not None
            or diagnostics.get("passed") is not True
            or diagnostics.get("release_certified") is not False
            or not all(item.startswith("release:") for item in failures)
            or len(final) != 3
            or any(item.passed is not True for item in final)
            or tuple(item.consecutive_passes for item in final) != (1, 2, 3)
            or float(metrics["history_last_consecutive_passes"]) != 3.0
            or float(metrics["history_last_sigma_residual"])
            != float(last.sigma_residual)
            or float(metrics["history_last_green_residual"])
            != float(last.green_residual)
            or float(metrics["current_contact_abs_error"])
            != conservation["contact_imbalance_w"]
            or float(metrics["current_collision_abs_error"])
            != conservation["collision_power_abs_w"]
            or float(metrics["current_tolerance"])
            != conservation["tolerance_w"]
            or float(metrics["equilibrium_fdt_applicable"]) != 1.0
            or float(metrics["package_version_consistent"]) != 1.0
        ):
            raise MaterialNQ1PilotError("SCBA result lacks the final three passing rows")
    else:
        if (
            not isinstance(result.get("scientific_error"), Mapping)
            or len(checkpoint.history) != 800
            or diagnostics.get("passed") is not False
            or diagnostics.get("release_certified") is not False
            or not any(not item.startswith("release:") for item in failures)
        ):
            raise MaterialNQ1PilotError("scientific failure did not exhaust frozen budget")


def _preflight_input_hashes(source: Mapping[str, Any]) -> dict[str, Any]:
    try:
        phase_b = source["phase_b"]
        return {
            "phase_a_fixture": source["fixture"]["fixture_manifest_sha256"],
            "raw_fc3": source["fc3_full"]["raw_sha256"],
            "phase_b_artifact_set": phase_b["artifact_set_sha256"],
            "phase_b_mapped_hdf5": phase_b["archive"]["sha256"],
            "phase_b_campaign_manifest": phase_b["campaign_manifest_sha256"],
        }
    except (KeyError, TypeError) as error:
        raise MaterialNQ1PilotError("SCBA outcome preflight sources are incomplete") from error


def _expected_scba_controls(task: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "eta_lead": task["eta_lead"],
        "max_iterations": 800,
        "min_iterations": 3,
        "mixing": task["mixing"],
        "mixing_floor": task["mixing_floor"],
        "consecutive_passes": 3,
        "sigma_rtol": 1.0e-8,
        "sigma_atol": 1.0e-12,
        "green_rtol": 1.0e-8,
        "green_atol": 1.0e-12,
        "identity_rtol": 1.0e-8,
        "current_rtol": 1.0e-5,
        "current_atol": 1.0e-10,
        "tail_rtol": 1.0e-6,
        "retarded_strategy": "causal_hilbert",
        "hilbert_backend": "fft",
        "backend": "fft",
        "on_nonconvergence": "raise",
        "eta_device": 0.0,
        "surface_gf_method": "sancho_rubio",
        "max_workers": 1,
        "max_memory_fraction": 0.7,
        "allow_memory_overcommit": False,
    }


def _validate_continuation_binding(
    *,
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    scba_manifest: Mapping[str, Any],
    outcome: Mapping[str, Any],
    hdf5_metadata: Mapping[str, Any],
    parent_receipt: Mapping[str, Any] | None,
    checkpoint: Any,
) -> dict[str, Any] | None:
    continuation = scba_manifest.get("continuation_parent")
    expected_parent = task["continuation_parent"]
    if expected_parent is None:
        if (
            continuation is not None
            or outcome.get("continuation_parent") is not None
            or hdf5_metadata.get("continuation_parent") is not None
        ):
            raise MaterialNQ1PilotError("zero-start task contains continuation evidence")
        if any(
            str(name).startswith("continuation_parent_")
            for name in checkpoint.identity.source_hashes
        ):
            raise MaterialNQ1PilotError(
                "zero-start checkpoint unexpectedly binds continuation inputs"
            )
        if checkpoint.identity.initial_state_sha256 != _explicit_zero_initial_sha256(
            checkpoint.sigma_lesser.shape
        ):
            raise MaterialNQ1PilotError("zero-start checkpoint initial state changed")
        return None
    if parent_receipt is None or not isinstance(continuation, Mapping):
        raise MaterialNQ1PilotError("continuation task lacks an authenticated parent")
    parent_task = _task_from_manifest(manifest, expected_parent)
    expected_factor = (
        float(task["coupling_scale"]) / float(parent_task["coupling_scale"])
    ) ** 2
    initialization = continuation.get("initialization")
    initialization_keys = {
        "schema",
        "predictor",
        "parent_coupling_scale",
        "child_coupling_scale",
        "factor",
        "untransformed_initial_sigma_sha256",
        "transformed_consumed_initial_sigma_sha256",
        "initialization_sha256",
    }
    if (
        not isinstance(initialization, Mapping)
        or set(initialization) != initialization_keys
        or initialization.get("schema") != CONTINUATION_INITIALIZATION_SCHEMA
        or initialization.get("predictor") != task["continuation_predictor"]
        or initialization.get("parent_coupling_scale")
        != parent_task["coupling_scale"]
        or initialization.get("child_coupling_scale") != task["coupling_scale"]
        or initialization.get("factor") != expected_factor
    ):
        raise MaterialNQ1PilotError("continuation initialization identity changed")
    for name in (
        "untransformed_initial_sigma_sha256",
        "transformed_consumed_initial_sigma_sha256",
        "initialization_sha256",
    ):
        _validate_sha256(initialization.get(name), name=name)
    initialization_payload = {
        name: initialization[name]
        for name in initialization_keys - {"initialization_sha256"}
    }
    if (
        sha256_bytes(canonical_json(initialization_payload))
        != initialization["initialization_sha256"]
        or continuation.get("initial_sigma_sha256")
        != initialization["transformed_consumed_initial_sigma_sha256"]
    ):
        raise MaterialNQ1PilotError("continuation initialization hash is invalid")
    parent_hashes = parent_receipt["terminal_hashes"]
    parent_checkpoint_path = (
        Path(str(parent_receipt["scba_root"])) / "checkpoint/sigma-pp.h5"
    )
    if sha256_file(parent_checkpoint_path) != parent_hashes["checkpoint"]:
        raise MaterialNQ1PilotError("continuation parent checkpoint hash changed")
    try:
        parent_checkpoint = load_scba_checkpoint_v2_self_validating(
            parent_checkpoint_path
        )
    except Exception as error:
        raise MaterialNQ1PilotError(
            "continuation parent checkpoint is not self-validating"
        ) from error
    if parent_checkpoint.status != "completed":
        raise MaterialNQ1PilotError("continuation parent checkpoint is not completed")
    untransformed_hash = _initial_pair_sha256(
        parent_checkpoint.sigma_lesser, parent_checkpoint.sigma_greater
    )
    transformed_hash = _initial_pair_sha256(
        np.asarray(parent_checkpoint.sigma_lesser) * expected_factor,
        np.asarray(parent_checkpoint.sigma_greater) * expected_factor,
    )
    if (
        initialization["untransformed_initial_sigma_sha256"]
        != untransformed_hash
        or initialization["transformed_consumed_initial_sigma_sha256"]
        != transformed_hash
    ):
        raise MaterialNQ1PilotError(
            "continuation predictor state does not derive from parent checkpoint"
        )
    if (
        continuation.get("run_id") != expected_parent
        or continuation.get("artifacts") != parent_receipt["scba_root"]
        or continuation.get("coupling_scale") != parent_task["coupling_scale"]
        or continuation.get("continuation_axis") != "coupling"
        or continuation.get("campaign_manifest_sha256") != parent_hashes["manifest"]
        or continuation.get("status_sha256") != parent_hashes["status"]
        or continuation.get("report_sha256") != parent_hashes["report"]
        or continuation.get("checkpoint_sha256") != parent_hashes["checkpoint"]
        or continuation.get("checkpoint_identity_sha256")
        != parent_receipt["checkpoint_identity_sha256"]
        or outcome.get("continuation_parent") != continuation
        or hdf5_metadata.get("continuation_parent") != continuation
    ):
        raise MaterialNQ1PilotError("continuation parent provenance bindings disagree")
    identity = checkpoint.identity
    sources = identity.source_hashes
    if (
        identity.initial_state_sha256
        != initialization["transformed_consumed_initial_sigma_sha256"]
        or sources.get("continuation_parent_campaign_manifest")
        != continuation["campaign_manifest_sha256"]
        or sources.get("continuation_parent_checkpoint")
        != continuation["checkpoint_sha256"]
        or sources.get("continuation_parent_report") != continuation["report_sha256"]
        or sources.get("continuation_parent_status") != continuation["status_sha256"]
        or sources.get("continuation_parent_initialization")
        != initialization["initialization_sha256"]
    ):
        raise MaterialNQ1PilotError("checkpoint does not bind the consumed predictor state")
    return dict(continuation)


def build_task_receipt(
    *,
    manifest: Mapping[str, Any],
    task_id: str,
    scba_root: str | Path,
    parent_receipt: str | Path | None,
) -> dict[str, Any]:
    """Authenticate a terminal task tree and build its create-only receipt."""

    validated = validate_pilot_manifest(manifest)
    task = _task_from_manifest(validated, task_id)
    root, pilot_root = _fixed_task_layout(scba_root, task_id)
    qualified_runtime, qualified_lock = _bound_qualification_contract(
        validated, pilot_root
    )
    paths = {
        "manifest": root / "campaign-manifest.json",
        "status": root / "status.json",
        "report": root / "scba-report.json",
        "outcome": root / "outcomes/scba.json",
        "fields": root / "outcomes/scba-fields.h5",
        "checkpoint": root / "checkpoint/sigma-pp.h5",
    }
    for name, path in paths.items():
        _regular_file(path, name=f"{task_id} {name}")
    scba_manifest = load_canonical_json(paths["manifest"], name="SCBA manifest")
    status_document = load_canonical_json(paths["status"], name="SCBA status")
    report_document = load_canonical_json(paths["report"], name="SCBA report")
    outcome = load_canonical_json(paths["outcome"], name="SCBA outcome")
    hdf5_metadata = _hdf5_metadata(paths["fields"])
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    manifest_keys = {
        "schema", "scientific_scope", "source", "source_tree_sha256",
        "environment_lock", "driver", "preflight", "run_id", "coupling_scale",
        "continuation_parent", "continuation_axis", "continuation_predictor",
        "born_initial", "numerical_controls", "fixed_controls_no_retuning",
        "restart", "watchdog", "outputs", "python", "distributions",
        "thread_limits",
    }
    status_keys = {
        "schema", "campaign_manifest_sha256", "state", "scba_converged",
        "release_certified", "report_sha256", "sha256sums_sha256",
        "automatic_retry",
    }
    report_keys = {
        "schema", "campaign_manifest_sha256", "state", "scba_converged",
        "checkpoint_sha256", "outcome_sha256", "fields_hdf5_sha256",
        "log_sha256", "supervision", "automatic_retry", "attempt_logs",
        "infrastructure_failures", "controls_changed", "release_certified",
        "next_gate",
    }
    outcome_keys = {
        "schema", "campaign_manifest_sha256", "preflight_source", "checkpoint",
        "checkpoint_sha256", "checkpoint_status", "continuation_parent",
        "born_initial", "result", "hdf5", "scba_executed",
        "release_certified", "not_certified",
    }
    preflight_keys = {
        "schema", "source", "run_id", "coupling_scale",
        "continuation_parent", "born_initial", "source_tree_sha256",
        "environment_sha256", "configuration_sha256",
        "checkpoint_identity_sha256", "checkpoint_schema", "checkpoint_path",
        "outputs", "dimensions", "predicted_peak_memory_bytes",
        "numerical_controls", "controls", "fixed_controls_no_retuning",
        "resume_exact_identity", "passed", "scba_executed",
        "release_certified",
    }
    if (
        set(scba_manifest) != manifest_keys
        or scba_manifest.get("schema")
        != "negfpy.sparse-nq1-phase-c-scba-campaign.v2"
        or set(status_document) != status_keys
        or status_document.get("schema")
        != "negfpy.sparse-nq1-phase-c-scba-status.v1"
        or set(report_document) != report_keys
        or report_document.get("schema")
        != "negfpy.sparse-nq1-phase-c-scba-report.v1"
        or set(outcome) != outcome_keys
        or outcome.get("schema") != "negfpy.sparse-nq1-phase-c-scba-case.v1"
    ):
        raise MaterialNQ1PilotError("SCBA terminal document schemas are invalid")
    if scba_manifest.get("run_id") != task_id:
        raise MaterialNQ1PilotError("SCBA manifest names another task")
    if scba_manifest.get("scientific_scope") != (
        "authenticated silicon Nq=1 sparse material SCBA, fixed Nw=17 "
        "development gate"
    ):
        raise MaterialNQ1PilotError("SCBA scientific scope changed")
    if scba_manifest.get("coupling_scale") != task["coupling_scale"]:
        raise MaterialNQ1PilotError("SCBA coupling changed")
    if scba_manifest.get("continuation_axis") != task["continuation_axis"]:
        raise MaterialNQ1PilotError("SCBA continuation axis changed")
    if scba_manifest.get("continuation_predictor") != task["continuation_predictor"]:
        raise MaterialNQ1PilotError("SCBA continuation predictor changed")
    source_record = scba_manifest.get("source")
    driver_record = scba_manifest.get("driver")
    if (
        not isinstance(source_record, Mapping)
        or source_record != {
            "commit": validated["source_revision"],
            "tree": validated["source_git_tree"],
            "dirty": False,
            "package_file": validated["source_package_file"],
        }
        or scba_manifest.get("source_tree_sha256") != validated["source_tree_sha256"]
        or scba_manifest.get("environment_lock") != qualified_lock
        or not isinstance(driver_record, Mapping)
        or driver_record != {
            "relative_path": "scripts/validate_sparse_nq1_phase_c_scba.py",
            "sha256": validated["input_hashes"]["scba_case_driver"],
        }
        or scba_manifest.get("python") != {
            "executable": qualified_runtime["executable"],
            "version": PILOT_PYTHON_VERSION,
        }
        or scba_manifest.get("distributions") != dict(PILOT_DISTRIBUTIONS)
        or scba_manifest.get("thread_limits") != dict(PILOT_THREAD_LIMITS)
    ):
        raise MaterialNQ1PilotError("SCBA candidate identity changed")
    expected_controls = {
        "grid_step": task["grid_step"],
        "grid_n_positive": task["grid_n_positive"],
        "grid_cutoff": task["grid_cutoff"],
        "eta_lead": task["eta_lead"],
        "mixing": task["mixing"],
        "mixing_floor": task["mixing_floor"],
        "temperature_bias": task["temperature_bias_solver"],
    }
    if scba_manifest.get("numerical_controls") != expected_controls:
        raise MaterialNQ1PilotError("SCBA numerical controls changed")
    expected_outputs = {
        "checkpoint": os.fspath(paths["checkpoint"].resolve()),
        "json": os.fspath(paths["outcome"].resolve()),
        "hdf5": os.fspath(paths["fields"].resolve()),
    }
    preflight = scba_manifest.get("preflight")
    if (
        scba_manifest.get("outputs") != expected_outputs
        or scba_manifest.get("fixed_controls_no_retuning") is not True
        or scba_manifest.get("restart") != {
            "checkpoint_schema": "negfpy_sparse_material_scba_checkpoint",
            "resume_exact_identity": True,
            "automatic_retry": False,
        }
        or scba_manifest.get("born_initial") is not None
        or not isinstance(preflight, Mapping)
        or set(preflight) != preflight_keys
        or preflight.get("schema")
        != "negfpy.sparse-nq1-phase-c-scba-preflight.v1"
        or preflight.get("run_id") != task_id
        or preflight.get("coupling_scale") != task["coupling_scale"]
        or preflight.get("source_tree_sha256") != validated["source_tree_sha256"]
        or preflight.get("environment_sha256") != validated["environment_sha256"]
        or preflight.get("checkpoint_identity_sha256")
        != hdf5_metadata.get("checkpoint_identity_sha256")
        or preflight.get("checkpoint_schema")
        != "negfpy_sparse_material_scba_checkpoint"
        or preflight.get("checkpoint_path") != expected_outputs["checkpoint"]
        or preflight.get("outputs")
        != {"json": expected_outputs["json"], "hdf5": expected_outputs["hdf5"]}
        or not isinstance(preflight.get("dimensions"), Mapping)
        or set(preflight["dimensions"]) != {"n_frequency", "ndof", "nnz"}
        or preflight["dimensions"].get("n_frequency") != EXPECTED_FIELD_SHAPE[0]
        or preflight["dimensions"].get("ndof") != EXPECTED_FIELD_SHAPE[1]
        or not isinstance(preflight["dimensions"].get("nnz"), int)
        or preflight["dimensions"].get("nnz", 0) <= 0
        or preflight.get("predicted_peak_memory_bytes")
        != 288 * EXPECTED_FIELD_SHAPE[0] * EXPECTED_FIELD_SHAPE[1] ** 2
        or preflight.get("numerical_controls") != expected_controls
        or preflight.get("fixed_controls_no_retuning") is not True
        or preflight.get("resume_exact_identity") is not True
        or preflight.get("scba_executed") is not False
        or preflight.get("continuation_parent")
        != scba_manifest.get("continuation_parent")
        or preflight.get("born_initial") is not None
        or preflight.get("release_certified") is not False
        or preflight.get("passed") is not True
    ):
        raise MaterialNQ1PilotError("SCBA immutable campaign contract changed")
    if _stable_material_identity(
        preflight.get("source"), name="SCBA manifest preflight source"
    ) != _stable_material_identity(
        outcome.get("preflight_source"), name="SCBA outcome preflight source"
    ):
        raise MaterialNQ1PilotError("SCBA preflight material identity changed")
    controls = preflight.get("controls", {})
    if controls != _expected_scba_controls(task):
        raise MaterialNQ1PilotError("SCBA full solver controls changed")
    state = status_document.get("state")
    if state not in {"passed", "scientific-failure"}:
        raise MaterialNQ1PilotError("SCBA task is not scientifically terminal")
    converged = state == "passed"
    try:
        checkpoint = load_scba_checkpoint_v2_self_validating(paths["checkpoint"])
    except Exception as error:
        raise MaterialNQ1PilotError("SCBA checkpoint is not self-validating") from error
    checkpoint_identity = checkpoint.identity
    expected_base_inputs = {
        name: digest
        for name, digest in validated["input_hashes"].items()
        if name != "scba_case_driver"
    }
    observed_base_inputs = _preflight_input_hashes(outcome.get("preflight_source", {}))
    expected_source_hashes = dict(expected_base_inputs)
    continuation_for_sources = scba_manifest.get("continuation_parent")
    if continuation_for_sources is not None:
        expected_source_hashes.update({
            "continuation_parent_campaign_manifest": continuation_for_sources.get(
                "campaign_manifest_sha256"
            ),
            "continuation_parent_checkpoint": continuation_for_sources.get(
                "checkpoint_sha256"
            ),
            "continuation_parent_report": continuation_for_sources.get(
                "report_sha256"
            ),
            "continuation_parent_status": continuation_for_sources.get(
                "status_sha256"
            ),
            "continuation_parent_initialization": continuation_for_sources.get(
                "initialization", {}
            ).get("initialization_sha256"),
        })
    hdf5_keys = {
        "schema", "campaign_manifest_sha256", "checkpoint_identity_sha256",
        "problem_sha256", "configuration_sha256", "continuation_parent",
        "born_initial", "result", "timings_seconds", "scba_executed",
        "release_certified",
    }
    if (
        observed_base_inputs != expected_base_inputs
        or checkpoint_identity.source_hashes != expected_source_hashes
        or status_document.get("campaign_manifest_sha256") != hashes["manifest"]
        or report_document.get("campaign_manifest_sha256") != hashes["manifest"]
        or status_document.get("report_sha256") != hashes["report"]
        or report_document.get("outcome_sha256") != hashes["outcome"]
        or report_document.get("fields_hdf5_sha256") != hashes["fields"]
        or report_document.get("checkpoint_sha256") != hashes["checkpoint"]
        or outcome.get("checkpoint_sha256") != hashes["checkpoint"]
        or outcome.get("checkpoint") != expected_outputs["checkpoint"]
        or outcome.get("born_initial") is not None
        or outcome.get("not_certified") != [
            "converged material conductance", "0.3.0 release", "transverse SCBA"
        ]
        or outcome.get("hdf5") != {
            "path": expected_outputs["hdf5"],
            "sha256": hashes["fields"],
            "schema": "negfpy.sparse-nq1-phase-c-scba-fields.v1",
        }
        or status_document.get("state") != report_document.get("state")
        or status_document.get("scba_converged") is not converged
        or report_document.get("scba_converged") is not converged
        or outcome.get("result", {}).get("converged") is not converged
        or outcome.get("release_certified") is not False
        or outcome.get("result", {}).get("release_certified") is not False
        or outcome.get("scba_executed") is not True
        or set(hdf5_metadata) != hdf5_keys
        or hdf5_metadata.get("schema")
        != "negfpy.sparse-nq1-phase-c-scba-fields.v1"
        or hdf5_metadata.get("campaign_manifest_sha256") != hashes["manifest"]
        or hdf5_metadata.get("checkpoint_identity_sha256")
        != checkpoint_identity.identity_sha256
        or hdf5_metadata.get("problem_sha256")
        != outcome["preflight_source"].get("problem_input_hashes", {}).get("problem")
        or hdf5_metadata.get("configuration_sha256")
        != preflight.get("configuration_sha256")
        or hdf5_metadata.get("born_initial") is not None
        or hdf5_metadata.get("result") != outcome.get("result")
        or not isinstance(hdf5_metadata.get("timings_seconds"), Mapping)
        or set(hdf5_metadata["timings_seconds"]) != {"total"}
        or not isinstance(hdf5_metadata["timings_seconds"]["total"], (int, float))
        or isinstance(hdf5_metadata["timings_seconds"]["total"], bool)
        or not math.isfinite(float(hdf5_metadata["timings_seconds"]["total"]))
        or float(hdf5_metadata["timings_seconds"]["total"]) < 0.0
        or hdf5_metadata.get("scba_executed") is not True
        or hdf5_metadata.get("release_certified") is not False
        or checkpoint_identity.run_id != task_id
        or checkpoint_identity.source_tree_sha256 != validated["source_tree_sha256"]
        or checkpoint_identity.environment_sha256 != validated["environment_sha256"]
    ):
        raise MaterialNQ1PilotError("SCBA terminal provenance bindings disagree")
    if (
        status_document.get("release_certified") is not False
        or report_document.get("release_certified") is not False
        or report_document.get("next_gate") != (
            "refine frequency/cutoff/eta and validate conservation"
            if converged
            else "inspect fixed-control scientific failure"
        )
    ):
        raise MaterialNQ1PilotError("task evidence misstates release certification")
    _validate_supervision(scba_manifest, report_document, converged=converged)
    if (
        report_document.get("automatic_retry") is not False
        or report_document.get("controls_changed") is not False
        or status_document.get("automatic_retry") is not False
    ):
        raise MaterialNQ1PilotError("task evidence permits retry or control changes")
    if converged:
        if (
            checkpoint.status != "completed"
            or not checkpoint.history
            or checkpoint.history[-1].consecutive_passes < 3
            or checkpoint.solver_state.consecutive_passes < 3
            or outcome.get("checkpoint_status") != "completed"
        ):
            raise MaterialNQ1PilotError(
                "passed task checkpoint lacks three consecutive convergence passes"
            )
    elif checkpoint.status != "scientific_failure":
        raise MaterialNQ1PilotError("failed task checkpoint is not scientifically terminal")
    _validate_result_arrays(paths["fields"], outcome, checkpoint, task)
    _validate_scientific_result(outcome, checkpoint, task, converged=converged)
    _replay_scientific_diagnostics(
        scba_manifest=scba_manifest,
        outcome=outcome,
        fields_path=paths["fields"],
        checkpoint=checkpoint,
    )
    expected_parent = task["continuation_parent"]
    if expected_parent is None:
        if parent_receipt is not None or scba_manifest.get("continuation_parent") is not None:
            raise MaterialNQ1PilotError("zero-start task cannot consume a parent")
        parent_record = None
    else:
        if parent_receipt is None:
            raise MaterialNQ1PilotError("continuation task lacks its pilot parent receipt")
        parent_path = _regular_file(parent_receipt, name="pilot parent receipt")
        expected_parent_path = (
            pilot_root / "tasks" / expected_parent / "task-receipt.json"
        )
        if (
            Path(os.path.abspath(parent_path)) != expected_parent_path
            or os.fspath(parent_path) != os.fspath(expected_parent_path)
        ):
            raise MaterialNQ1PilotError("pilot parent receipt path is not fixed")
        parent_document = validate_task_receipt(validated, parent_path)
        if (
            parent_document.get("task_id") != expected_parent
            or parent_document.get("state") != "passed"
        ):
            raise MaterialNQ1PilotError("pilot parent receipt is not a passed dependency")
        continuation = scba_manifest.get("continuation_parent")
        if (
            not isinstance(continuation, Mapping)
            or continuation.get("run_id") != expected_parent
        ):
            raise MaterialNQ1PilotError("SCBA manifest consumed another continuation parent")
        parent_record = {
            "relative_path": os.fspath(parent_path),
            "sha256": sha256_file(parent_path),
            "task_id": expected_parent,
        }
    continuation_record = _validate_continuation_binding(
        manifest=validated,
        task=task,
        scba_manifest=scba_manifest,
        outcome=outcome,
        hdf5_metadata=hdf5_metadata,
        parent_receipt=(None if expected_parent is None else parent_document),
        checkpoint=checkpoint,
    )
    records = _tree_records(root)
    forbidden_paths = [
        name
        for name in records
        if "infrastructure-failure" in name
        or re.search(r"(?:^|[-_/])resume(?:[-_.]|$)", name) is not None
    ]
    if forbidden_paths:
        raise MaterialNQ1PilotError(
            "task evidence is not one immutable initial attempt without "
            "infrastructure history"
        )
    expected_tree_names = {
        "campaign-manifest.json",
        "status-running.json",
        "outcomes/scba.json",
        "outcomes/scba-fields.h5",
        "checkpoint/sigma-pp.h5",
        "scba-report.json",
        "logs/scba.json",
        "SHA256SUMS",
        "status.json",
    }
    observed_directories = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_dir()
    }
    expected_directories = {"checkpoint", "logs", "outcomes"}
    running = load_canonical_json(
        root / "status-running.json", name="SCBA running marker"
    )
    sums_path = _regular_file(root / "SHA256SUMS", name="SCBA SHA256SUMS")
    if (
        set(records) != expected_tree_names
        or observed_directories != expected_directories
        or running != {
            "schema": "negfpy.sparse-nq1-phase-c-scba-status.v1",
            "campaign_manifest_sha256": hashes["manifest"],
            "state": "running",
            "automatic_retry": False,
            "release_certified": False,
        }
        or sums_path.read_bytes() != _expected_scba_sums(root)
        or status_document.get("sha256sums_sha256")
        != records.get("SHA256SUMS")
    ):
        raise MaterialNQ1PilotError("SCBA initial/terminal checksum evidence changed")
    expected_attempts = [{
        "path": "logs/scba.json",
        "sha256": records.get("logs/scba.json"),
    }]
    if (
        report_document.get("attempt_logs") != expected_attempts
        or report_document.get("infrastructure_failures") != []
        or report_document.get("log_sha256") != records.get("logs/scba.json")
    ):
        raise MaterialNQ1PilotError(
            "task evidence is not one immutable initial attempt without infrastructure history"
        )
    attempt_log = load_canonical_json(root / "logs/scba.json", name="SCBA attempt log")
    expected_attempt_command = _expected_scba_attempt_command(
        validated, task, scba_manifest, root, qualified_runtime
    )
    if (
        set(attempt_log)
        != {"attempt", "resume", "command", "supervision", "stdout", "stderr"}
        or attempt_log.get("attempt") != 1
        or attempt_log.get("resume") is not False
        or attempt_log.get("supervision") != report_document.get("supervision")
        or attempt_log.get("command") != expected_attempt_command
        or attempt_log.get("stdout") != canonical_json(outcome).decode("ascii")
        or attempt_log.get("stderr") != ""
    ):
        raise MaterialNQ1PilotError("SCBA attempt log does not prove a fresh initial run")
    conservation = conservation_record(outcome)
    if converged and conservation["passed"] is not True:
        raise MaterialNQ1PilotError("converged task failed conservation")
    return {
        "schema": TASK_RECEIPT_SCHEMA,
        "pilot_sha256": validated["pilot_sha256"],
        "execution_id": validated["execution_id"],
        "task_id": task_id,
        "scientific_task_sha256": task["task_sha256"],
        "state": state,
        "scba_root": os.fspath(root),
        "scba_tree_records": records,
        "scba_tree_sha256": _tree_sha256(records),
        "terminal_hashes": hashes,
        "parent_receipt": parent_record,
        "conservation": conservation,
        "continuation": continuation_record,
        "checkpoint_identity_sha256": checkpoint_identity.identity_sha256,
        "release_certified": False,
    }


def validate_task_receipt(
    manifest: Mapping[str, Any],
    receipt_path: str | Path,
) -> dict[str, Any]:
    selected = _regular_file(receipt_path, name="pilot task receipt")
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("pilot task receipt must have mode 0444")
    receipt = load_canonical_json(selected, name="pilot task receipt")
    expected_keys = {
        "schema",
        "pilot_sha256",
        "execution_id",
        "task_id",
        "scientific_task_sha256",
        "state",
        "scba_root",
        "scba_tree_records",
        "scba_tree_sha256",
        "terminal_hashes",
        "parent_receipt",
        "conservation",
        "continuation",
        "checkpoint_identity_sha256",
        "release_certified",
    }
    if set(receipt) != expected_keys or receipt.get("schema") != TASK_RECEIPT_SCHEMA:
        raise MaterialNQ1PilotError("pilot task receipt schema is invalid")
    task_id = str(receipt.get("task_id"))
    scba_root, pilot_root = _fixed_task_layout(
        str(receipt.get("scba_root")), task_id
    )
    expected_receipt = pilot_root / "tasks" / task_id / "task-receipt.json"
    if (
        Path(os.path.abspath(selected)) != expected_receipt
        or os.fspath(selected) != os.fspath(expected_receipt)
    ):
        raise MaterialNQ1PilotError("pilot task receipt path is not fixed")
    rebuilt = build_task_receipt(
        manifest=manifest,
        task_id=task_id,
        scba_root=scba_root,
        parent_receipt=(
            None
            if receipt.get("parent_receipt") is None
            else str(receipt["parent_receipt"]["relative_path"])
        ),
    )
    if receipt != rebuilt:
        raise MaterialNQ1PilotError("pilot task receipt does not reproduce")
    return rebuilt


def build_comparison(
    *,
    manifest: Mapping[str, Any],
    comparison_id: str,
    first_receipt_path: str | Path,
    second_kind: str,
    second_id: str,
    second_fields_path: str | Path,
    second_outcome_path: str | Path,
    second_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Compare one passed pilot receipt with an anchor or another route."""

    validated = validate_pilot_manifest(manifest)
    first_path = _regular_file(first_receipt_path, name="first pilot receipt")
    first_receipt = validate_task_receipt(validated, first_path)
    task = _task_from_manifest(validated, first_receipt["task_id"])
    expected_anchor_id = task.get("anchor_task_id")
    if expected_anchor_id is None:
        raise MaterialNQ1PilotError("this pilot task has no historical comparison anchor")
    expected_comparison_id = f"anchor-{first_receipt['task_id']}"
    anchor_records = validated["anchor_source"]["records"][expected_anchor_id]
    anchor_root = Path(validated["anchor_source"]["root"])
    expected_fields_path = anchor_root / anchor_records["fields"]["relative_path"]
    expected_outcome_path = anchor_root / anchor_records["outcome"]["relative_path"]
    if (
        comparison_id != expected_comparison_id
        or second_kind != "release-018-anchor"
        or second_id != expected_anchor_id
        or Path(second_fields_path).absolute() != expected_fields_path.absolute()
        or Path(second_outcome_path).absolute() != expected_outcome_path.absolute()
        or second_hashes.get("fields") != anchor_records["fields"]["sha256"]
        or second_hashes.get("outcome") != anchor_records["outcome"]["sha256"]
    ):
        raise MaterialNQ1PilotError("comparison does not name the frozen release-018 anchor")
    first_fields = Path(first_receipt["scba_root"]) / "outcomes/scba-fields.h5"
    first_outcome_path = Path(first_receipt["scba_root"]) / "outcomes/scba.json"
    second_fields = _regular_file(second_fields_path, name="comparison fields")
    second_outcome_selected = _regular_file(
        second_outcome_path, name="comparison outcome"
    )
    expected_second_fields = _validate_sha256(
        second_hashes.get("fields"), name="comparison fields SHA256"
    )
    expected_second_outcome = _validate_sha256(
        second_hashes.get("outcome"), name="comparison outcome SHA256"
    )
    if sha256_file(second_fields) != expected_second_fields:
        raise MaterialNQ1PilotError("comparison field archive hash changed")
    if sha256_file(second_outcome_selected) != expected_second_outcome:
        raise MaterialNQ1PilotError("comparison outcome hash changed")
    first_outcome = load_canonical_json(first_outcome_path, name="first outcome")
    second_outcome = load_canonical_json(second_outcome_selected, name="second outcome")
    fields = compare_fields(first_fields, second_fields)
    currents = compare_currents(first_outcome, second_outcome)
    passed = bool(
        first_receipt["state"] == "passed"
        and fields["passed"]
        and currents["passed"]
    )
    return {
        "schema": COMPARISON_SCHEMA,
        "pilot_sha256": validated["pilot_sha256"],
        "comparison_id": comparison_id,
        "first": {
            "kind": "pilot-task",
            "task_id": first_receipt["task_id"],
            "receipt_path": os.fspath(first_path),
            "receipt_sha256": sha256_file(first_path),
            "fields_sha256": sha256_file(first_fields),
            "outcome_sha256": sha256_file(first_outcome_path),
        },
        "second": {
            "kind": second_kind,
            "id": second_id,
            "fields_path": os.fspath(second_fields),
            "fields_sha256": expected_second_fields,
            "outcome_path": os.fspath(second_outcome_selected),
            "outcome_sha256": expected_second_outcome,
        },
        "fields": fields,
        "currents_and_conservation": currents,
        "passed": passed,
        "release_certified": False,
    }


def validate_comparison(
    manifest: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    selected = _regular_file(path, name="pilot comparison")
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("pilot comparison must have mode 0444")
    record = load_canonical_json(selected, name="pilot comparison")
    expected_keys = {
        "schema",
        "pilot_sha256",
        "comparison_id",
        "first",
        "second",
        "fields",
        "currents_and_conservation",
        "passed",
        "release_certified",
    }
    first = record.get("first")
    second = record.get("second")
    if (
        set(record) != expected_keys
        or record.get("schema") != COMPARISON_SCHEMA
        or not isinstance(first, Mapping)
        or not isinstance(second, Mapping)
    ):
        raise MaterialNQ1PilotError("pilot comparison schema is invalid")
    first_receipt_path = Path(str(first.get("receipt_path")))
    first_receipt = validate_task_receipt(manifest, first_receipt_path)
    fixed_first_receipt = Path(first_receipt["scba_root"]).parent / "task-receipt.json"
    pilot_root = fixed_first_receipt.parent.parent.parent
    expected_path = (
        pilot_root / "comparisons" / f"{record['comparison_id']}.json"
    )
    _require_fixed_absolute_path(
        selected, expected_path, name="pilot comparison"
    )
    rebuilt = build_comparison(
        manifest=manifest,
        comparison_id=str(record.get("comparison_id")),
        first_receipt_path=str(first.get("receipt_path")),
        second_kind=str(second.get("kind")),
        second_id=str(second.get("id")),
        second_fields_path=str(second.get("fields_path")),
        second_outcome_path=str(second.get("outcome_path")),
        second_hashes={
            "fields": str(second.get("fields_sha256")),
            "outcome": str(second.get("outcome_sha256")),
        },
    )
    if record != rebuilt:
        raise MaterialNQ1PilotError("pilot comparison does not reproduce")
    return rebuilt


def build_preflight_record(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    qualification_path: str | Path,
    source_clean_and_exact: bool,
    input_hashes_reproduced: bool,
    anchors: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_pilot_manifest(manifest)
    selected_manifest = _regular_file(manifest_path, name="pilot manifest")
    qualification = _regular_file(qualification_path, name="qualification report")
    if sha256_file(qualification) != validated["qualification_sha256"]:
        raise MaterialNQ1PilotError("qualification report hash changed")
    qualification_document = load_canonical_json(
        qualification, name="qualification report"
    )
    qualification_passed = qualification_authorizes_pilot(
        validated, qualification_document
    )
    anchor_passed = bool(
        set(anchors) == {"coupling-g0001", "coupling-g001"}
        and all(item.get("passed") is True for item in anchors.values())
    )
    check_record = {
        "manifest_reproduced": (
            sha256_file(selected_manifest)
            == sha256_bytes(canonical_json(validated))
        ),
        "source_clean_and_exact": source_clean_and_exact is True,
        "qualification_passed": qualification_passed,
        "input_hashes_reproduced": input_hashes_reproduced is True,
        "release_018_anchors_authenticated": anchor_passed,
        "pilot_is_non_release": validated["release_certified"] is False,
    }
    return {
        "schema": PREFLIGHT_SCHEMA,
        "pilot_sha256": validated["pilot_sha256"],
        "manifest": {
            "path": os.fspath(selected_manifest),
            "sha256": sha256_file(selected_manifest),
        },
        "qualification": {
            "path": os.fspath(qualification),
            "sha256": sha256_file(qualification),
        },
        "anchor_authentication": dict(anchors),
        "checks": check_record,
        "passed": bool(check_record and all(check_record.values())),
        "release_certified": False,
    }


def _qualification_file_records_match(
    repo: Path, records: Any, *, exact_names: set[str] | None = None
) -> bool:
    if not isinstance(records, Mapping) or not records:
        return False
    if exact_names is not None and set(records) != exact_names:
        return False
    for relative, item in records.items():
        if (
            not isinstance(relative, str)
            or not isinstance(item, Mapping)
            or set(item) != {"relative_path", "sha256", "size"}
            or item.get("relative_path") != relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            return False
        try:
            selected = _regular_file(repo / relative, name="qualification context file")
        except MaterialNQ1PilotError:
            return False
        if (
            sha256_file(selected) != item.get("sha256")
            or selected.stat().st_size != item.get("size")
        ):
            return False
    return True


def _qualification_context_valid(
    manifest: Mapping[str, Any],
    context: Any,
    base_records: Any,
    *,
    expected_task_set_sha256: str,
) -> bool:
    if not isinstance(context, Mapping) or not isinstance(base_records, Mapping):
        return False
    expected_keys = {
        "schema", "repo_root", "source", "source_tree_sha256",
        "environment_lock", "scientific_task_set_sha256",
        "material_input_hashes", "runtimes", "mpiexec",
        "tracked_test_files", "driver_files", "qualification_context_sha256",
    }
    if set(context) != expected_keys or context.get("schema") != (
        "negfpy.material-nq1-qualification-context.v1"
    ):
        return False
    payload = {name: context[name] for name in expected_keys - {"qualification_context_sha256"}}
    if sha256_bytes(canonical_json(payload)) != context.get(
        "qualification_context_sha256"
    ):
        return False
    try:
        repo = _directory(context.get("repo_root"), name="qualified repository")
    except MaterialNQ1PilotError:
        return False
    expected_source = {
        "commit": manifest["source_revision"],
        "tree": manifest["source_git_tree"],
        "dirty": False,
        "package_file": manifest["source_package_file"],
    }
    environment_lock = context.get("environment_lock")
    runtimes = context.get("runtimes")
    mpiexec = context.get("mpiexec")
    expected_inputs = {
        name: digest for name, digest in manifest["input_hashes"].items()
        if name != "scba_case_driver"
    }
    if (
        context.get("source") != expected_source
        or context.get("source_tree_sha256") != manifest["source_tree_sha256"]
        or not isinstance(environment_lock, Mapping)
        or environment_lock.get("sha256") != manifest["environment_sha256"]
        or context.get("material_input_hashes") != expected_inputs
        or context.get("scientific_task_set_sha256")
        != expected_task_set_sha256
        or not isinstance(runtimes, Mapping)
        or set(runtimes) != {"python310", "python312"}
        or runtimes.get("python310") != base_records.get("python310_runtime")
        or runtimes.get("python312") != base_records.get("python312_runtime")
        or not isinstance(mpiexec, Mapping)
        or set(mpiexec) != {"path", "sha256"}
    ):
        return False
    # The task-set value is cross-checked by the caller after reconstructing
    # the exact frozen scientific controls.
    try:
        selected_mpiexec = _regular_file(mpiexec.get("path"), name="qualified mpiexec")
    except MaterialNQ1PilotError:
        return False
    if sha256_file(selected_mpiexec) != mpiexec.get("sha256"):
        return False
    for runtime in runtimes.values():
        if not isinstance(runtime, Mapping):
            return False
        try:
            executable = _regular_file(
                runtime.get("executable"), name="qualified Python runtime"
            )
        except MaterialNQ1PilotError:
            return False
        if sha256_file(executable) != runtime.get("executable_sha256"):
            return False
    return bool(
        _qualification_file_records_match(
            repo,
            context.get("driver_files"),
            exact_names=set(_QUALIFICATION_DRIVER_PATHS),
        )
        and _qualification_file_records_match(
            repo, context.get("tracked_test_files")
        )
        and all(
            name == item.get("relative_path")
            and (
                name == "tests"
                or name.startswith(
                    ("tests/", "docs/validation/bulk-rta-report/tests/")
                )
            )
            for name, item in context["tracked_test_files"].items()
        )
    )


def qualification_authorizes_pilot(
    manifest: Mapping[str, Any], document: Mapping[str, Any]
) -> bool:
    """Return whether the exact v3 qualification authorizes only this pilot."""

    validated = validate_pilot_manifest(manifest)
    expected_keys = {
        "schema",
        "source",
        "source_tree_sha256",
        "environment_sha256",
        "scientific_controls",
        "predictor_pilot_controls",
        "checks",
        "evidence",
        "candidate_qualification_passed",
        "pilot_submission_eligible",
        "release_submission_eligible",
        "next_gate",
    }
    checks = document.get("checks")
    controls = document.get("predictor_pilot_controls")
    full_controls = document.get("scientific_controls")
    evidence = document.get("evidence")
    source = document.get("source")
    expected_pilot_tasks = [
        task.as_dict() | {"task_sha256": task.task_sha256}
        for task in predictor_pilot_tasks()
    ]
    expected_full_tasks = [
        task.as_dict() | {"task_sha256": task.task_sha256}
        for task in silicon_nq1_release_tasks()
    ]
    full_payload = {
        "schema": "negfpy.material-nq1-scientific-controls.v2",
        "acceptance": dict(MATERIAL_NQ1_ACCEPTANCE),
        "excluded_claims": list(MATERIAL_NQ1_EXCLUDED_CLAIMS),
        "task_count": 19,
        "tasks": expected_full_tasks,
    }
    expected_full_controls = full_payload | {
        "task_set_sha256": sha256_bytes(canonical_json(full_payload))
    }
    anchor_contract = dict(validated["anchor_source"])
    anchor_contract_sha = sha256_bytes(canonical_json(anchor_contract))
    pilot_payload = {
        "schema": "negfpy.material-nq1-predictor-pilot-controls.v1",
        "classification": "development-pilot-not-release",
        "task_count": 3,
        "tasks": expected_pilot_tasks,
        "solver_contract": dict(validated["solver_contract"]),
        "acceptance": dict(validated["acceptance"]),
        "anchor_contract": anchor_contract,
        "anchor_contract_sha256": anchor_contract_sha,
        "anchor_authentication_gate": "three-task-pilot-preflight",
        "excluded_claims": list(validated["excluded_claims"]),
        "release_certified": False,
    }
    expected_pilot_controls = pilot_payload | {
        "task_set_sha256": sha256_bytes(canonical_json(pilot_payload))
    }
    expected_checks = _BASE_QUALIFICATION_CHECKS | _PREDICTOR_QUALIFICATION_CHECKS
    evidence_valid = False
    if isinstance(evidence, Mapping) and set(evidence) == {
        "base_qualification_schema",
        "base_qualification_evidence",
        "core_files",
        "predictor_counts",
        "property_oracle",
    }:
        core = evidence.get("core_files")
        oracle = evidence.get("property_oracle")
        base_evidence = evidence.get("base_qualification_evidence")
        base_records = (
            base_evidence.get("records")
            if isinstance(base_evidence, Mapping)
            else None
        )
        expected_base_records = {
            "python310_runtime", "python310_pytest",
            "python312_runtime", "python312_pytest",
            "mpi_ranks_2", "mpi_ranks_4",
            "material_input_preflight", "resources",
        }
        base_valid = bool(
            isinstance(base_evidence, Mapping)
            and set(base_evidence)
            == {"qualification_context", "python_versions", "records"}
            and isinstance(base_evidence.get("python_versions"), Mapping)
            and set(base_evidence["python_versions"]) == {"python310", "python312"}
            and str(base_evidence["python_versions"]["python310"]).startswith("3.10.")
            and str(base_evidence["python_versions"]["python312"]).startswith("3.12.")
            and isinstance(base_records, Mapping)
            and set(base_records) == expected_base_records
            and _qualification_context_valid(
                validated,
                base_evidence.get("qualification_context"),
                base_records,
                expected_task_set_sha256=expected_full_controls[
                    "task_set_sha256"
                ],
            )
        )
        if base_valid:
            for label in ("python310", "python312"):
                runtime = base_records[f"{label}_runtime"]
                suite = base_records[f"{label}_pytest"]
                base_valid = bool(
                    base_valid
                    and isinstance(runtime, Mapping)
                    and set(runtime) == {
                        "schema", "label", "version", "executable",
                        "executable_sha256", "packages_sha256",
                        "runtime_sha256", "package_manifest",
                    }
                    and runtime.get("schema")
                    == "negfpy.material-nq1-python-runtime.v1"
                    and runtime.get("label") == label
                    and runtime.get("version")
                    == base_evidence["python_versions"][label]
                    and _SHA256.fullmatch(str(runtime.get("executable_sha256")))
                    is not None
                    and _SHA256.fullmatch(str(runtime.get("packages_sha256")))
                    is not None
                    and _SHA256.fullmatch(str(runtime.get("runtime_sha256")))
                    is not None
                    and isinstance(runtime.get("package_manifest"), Mapping)
                    and set(runtime["package_manifest"]) == {"path", "sha256", "size"}
                    and runtime["package_manifest"].get("sha256")
                    == runtime.get("packages_sha256")
                    and isinstance(runtime["package_manifest"].get("size"), int)
                    and runtime["package_manifest"].get("size", 0) > 0
                    and isinstance(suite, Mapping)
                    and set(suite) == {
                        "command", "returncode", "path", "sha256", "size",
                        "receipt_path", "receipt_sha256", "identity_sha256",
                        "attempt", "reused", "scope", "excluded_tests",
                    }
                    and suite.get("returncode") == 0
                    and suite.get("scope")
                    == ["tests", "docs/validation/bulk-rta-report/tests"]
                    and suite.get("excluded_tests") == []
                    and _SHA256.fullmatch(str(suite.get("sha256"))) is not None
                    and _SHA256.fullmatch(str(suite.get("receipt_sha256"))) is not None
                    and isinstance(suite.get("size"), int)
                    and suite.get("size", 0) >= 0
                    and isinstance(suite.get("reused"), bool)
                    and isinstance(suite.get("attempt"), int)
                    and suite.get("attempt", 0) >= 1
                    and _SHA256.fullmatch(str(suite.get("identity_sha256")))
                    is not None
                )
            for ranks in (2, 4):
                mpi = base_records[f"mpi_ranks_{ranks}"]
                base_valid = bool(
                    base_valid
                    and isinstance(mpi, Mapping)
                    and set(mpi) == {
                        "command", "returncode", "path", "sha256", "size",
                        "receipt_path", "receipt_sha256", "identity_sha256",
                        "attempt", "reused", "interpreter",
                    }
                    and mpi.get("returncode") == 0
                    and mpi.get("interpreter") == "python312"
                    and isinstance(mpi.get("command"), list)
                    and "-n" in mpi["command"]
                    and str(ranks) in mpi["command"]
                    and _SHA256.fullmatch(str(mpi.get("sha256"))) is not None
                    and _SHA256.fullmatch(str(mpi.get("receipt_sha256"))) is not None
                    and isinstance(mpi.get("reused"), bool)
                    and isinstance(mpi.get("attempt"), int)
                    and mpi.get("attempt", 0) >= 1
                    and _SHA256.fullmatch(str(mpi.get("identity_sha256")))
                    is not None
                )
            material = base_records["material_input_preflight"]
            resources = base_records["resources"]
            expected_input_hashes = {
                name: digest
                for name, digest in validated["input_hashes"].items()
                if name != "scba_case_driver"
            }
            base_valid = bool(
                base_valid
                and isinstance(material, Mapping)
                and set(material)
                == {"source_hashes", "observation_modes", "harmonic_buffer"}
                and material.get("source_hashes") == expected_input_hashes
                and isinstance(material.get("observation_modes"), Mapping)
                and bool(material["observation_modes"].get("subspaces"))
                and isinstance(material.get("harmonic_buffer"), Mapping)
                and material["harmonic_buffer"].get("passed") is True
                and isinstance(resources, Mapping)
                and resources.get("memory_guard_passed") is True
                and resources.get("disk_guard_passed") is True
            )
        evidence_valid = bool(
            evidence.get("base_qualification_schema")
            == "negfpy.material-nq1-candidate-qualification.v2"
            and base_valid
            and isinstance(core, Mapping)
            and set(core) == set(_CORE_FILE_SHA256)
            and all(
                isinstance(item, Mapping)
                and set(item) == {"expected_sha256", "observed_sha256", "passed"}
                and item.get("passed") is True
                and item.get("expected_sha256") == _CORE_FILE_SHA256[name]
                and item.get("observed_sha256") == _CORE_FILE_SHA256[name]
                for name, item in core.items()
            )
            and evidence.get("predictor_counts")
            == {"coupling-square": 8, "identity": 4, "none": 7}
            and isinstance(oracle, Mapping)
            and oracle.get("relative_path")
            == "tests/test_material_nq1_predictor_oracle.py"
            and set(oracle) == {
                "relative_path", "sha256", "covered_by", "suite_records"
            }
            and oracle.get("sha256") == _PROPERTY_ORACLE_SHA256
            and oracle.get("covered_by")
            == ["python310_full_suite", "python312_full_suite"]
            and isinstance(oracle.get("suite_records"), Mapping)
            and set(oracle["suite_records"])
            == {"python310_pytest", "python312_pytest"}
            and all(
                isinstance(item, Mapping)
                and "tests" in item.get("scope", [])
                and item.get("excluded_tests") == []
                for item in oracle["suite_records"].values()
            )
        )
    return bool(
        set(document) == expected_keys
        and document.get("schema") == QUALIFICATION_SCHEMA
        and isinstance(source, Mapping)
        and set(source) == {"commit", "tree", "dirty", "package_file"}
        and source.get("commit") == validated["source_revision"]
        and source.get("tree") == validated["source_git_tree"]
        and source.get("dirty") is False
        and source.get("package_file") == validated["source_package_file"]
        and document.get("source_tree_sha256") == validated["source_tree_sha256"]
        and document.get("environment_sha256") == validated["environment_sha256"]
        and isinstance(checks, Mapping)
        and set(checks) == expected_checks
        and all(value is True for value in checks.values())
        and full_controls == expected_full_controls
        and controls == expected_pilot_controls
        and evidence_valid
        and document.get("candidate_qualification_passed") is True
        and document.get("pilot_submission_eligible") is True
        and document.get("release_submission_eligible") is False
        and document.get("next_gate") == "three-task-pilot"
    )


def validate_preflight_record(
    manifest: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    validated = validate_pilot_manifest(manifest)
    selected = _regular_file(path, name="pilot preflight")
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("pilot preflight must have mode 0444")
    record = load_canonical_json(selected, name="pilot preflight")
    expected_record_keys = {
        "schema", "pilot_sha256", "manifest", "qualification",
        "anchor_authentication", "checks", "passed", "release_certified",
    }
    expected_checks = {
        "manifest_reproduced",
        "source_clean_and_exact",
        "qualification_passed",
        "input_hashes_reproduced",
        "release_018_anchors_authenticated",
        "pilot_is_non_release",
    }
    manifest_record = record.get("manifest")
    qualification_record = record.get("qualification")
    manifest_record_path = (
        Path(str(manifest_record.get("path")))
        if isinstance(manifest_record, Mapping)
        else Path()
    )
    qualification_record_path = (
        Path(str(qualification_record.get("path")))
        if isinstance(qualification_record, Mapping)
        else Path()
    )
    if (
        set(record) != expected_record_keys
        or record.get("schema") != PREFLIGHT_SCHEMA
        or record.get("pilot_sha256") != validated["pilot_sha256"]
        or record.get("passed") is not True
        or record.get("release_certified") is not False
        or not isinstance(record.get("checks"), Mapping)
        or set(record["checks"]) != expected_checks
        or not all(value is True for value in record["checks"].values())
        or not isinstance(manifest_record, Mapping)
        or set(manifest_record) != {"path", "sha256"}
        or not manifest_record_path.is_absolute()
        or manifest_record_path != Path(os.path.abspath(manifest_record_path))
        or sha256_file(manifest_record_path) != manifest_record.get("sha256")
        or load_canonical_json(manifest_record_path, name="bound pilot manifest")
        != validated
        or not isinstance(qualification_record, Mapping)
        or set(qualification_record) != {"path", "sha256"}
        or not qualification_record_path.is_absolute()
        or qualification_record_path
        != Path(os.path.abspath(qualification_record_path))
        or sha256_file(qualification_record_path)
        != qualification_record.get("sha256")
        or qualification_record.get("sha256") != validated["qualification_sha256"]
    ):
        raise MaterialNQ1PilotError("pilot preflight did not pass")
    manifest_selected = _regular_file(
        manifest_record_path, name="bound pilot manifest"
    )
    if stat.S_IMODE(manifest_selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("bound pilot manifest must have mode 0444")
    qualification_selected = _regular_file(
        qualification_record_path, name="bound qualification"
    )
    if stat.S_IMODE(qualification_selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("bound qualification must have mode 0444")
    qualification = load_canonical_json(
        qualification_selected, name="bound qualification"
    )
    if not qualification_authorizes_pilot(validated, qualification):
        raise MaterialNQ1PilotError("bound qualification no longer authorizes pilot")
    anchors = authenticate_anchor_source(validated)
    if record.get("anchor_authentication") != anchors:
        raise MaterialNQ1PilotError("preflight anchor authentication does not reproduce")
    return record


def _scheduler_workload_contract(
    manifest: Mapping[str, Any],
    task: Mapping[str, Any],
    task_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    scba_root, pilot_root = _fixed_task_layout(
        task_receipt["scba_root"], task["task_id"]
    )
    outcome = load_canonical_json(
        scba_root / "outcomes/scba.json", name="scheduler-bound SCBA outcome"
    )
    preflight = validate_preflight_record(
        manifest, pilot_root / "pilot-preflight.json"
    )
    qualification = load_canonical_json(
        preflight["qualification"]["path"], name="scheduler-bound qualification"
    )
    runtime, environment_lock_record = qualified_python312_runtime(
        manifest, qualification
    )
    source = outcome.get("preflight_source")
    if not isinstance(source, Mapping):
        raise MaterialNQ1PilotError("scheduler workload lacks authenticated input paths")
    try:
        repo = Path(manifest["source_package_file"]).parents[2]
        python = runtime["executable"]
        environment_lock = environment_lock_record["path"]
        qualification_path = preflight["qualification"]["path"]
        fixture = source["fixture_path"]
        fc3_full = source["fc3_full"]["path"]
        phase_b = source["phase_b"]["root"]
    except (KeyError, TypeError, IndexError) as error:
        raise MaterialNQ1PilotError(
            "scheduler workload identity is incomplete"
        ) from error
    driver = repo / "scripts/run_material_nq1_predictor_pilot.py"
    _regular_file(driver, name="pilot workload driver")
    command = [
        str(python),
        os.fspath(driver),
        "run-task",
        "--root", os.fspath(pilot_root),
        "--repo-root", os.fspath(repo),
        "--expected-commit", manifest["source_revision"],
        "--environment-lock", str(environment_lock),
        "--qualification-report", str(qualification_path),
        "--fixture", str(fixture),
        "--fc3-full", str(fc3_full),
        "--phase-b-artifacts", str(phase_b),
        "--task-id", task["task_id"],
    ]
    return {
        "pilot_root": os.fspath(pilot_root),
        "repo_root": os.fspath(repo),
        "driver_path": os.fspath(driver),
        "driver_sha256": sha256_file(driver),
        "command": command,
        "command_sha256": sha256_bytes(canonical_json(command)),
    }


def _canonical_pbs_script(
    *,
    task: Mapping[str, Any],
    stdout_path: Path,
    stderr_path: Path,
    workload: Mapping[str, Any],
) -> bytes:
    def literal_path(value: Any, *, name: str) -> str:
        text = str(value)
        selected = Path(text)
        if (
            not selected.is_absolute()
            or ".." in selected.parts
            or any(character in text for character in "\r\n\0")
            or any(character.isspace() for character in text)
        ):
            raise MaterialNQ1PilotError(f"{name} is not a safe absolute PBS path")
        return text

    repo = literal_path(workload["repo_root"], name="PBS repository")
    stdout = literal_path(stdout_path, name="PBS stdout")
    stderr = literal_path(stderr_path, name="PBS stderr")
    command = workload.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(
            isinstance(item, str)
            and item
            and not any(character in item for character in "\r\n\0")
            for item in command
        )
    ):
        raise MaterialNQ1PilotError("PBS workload command contains unsafe bytes")
    python = literal_path(command[0], name="qualified P312 executable")
    staged_path = os.pathsep.join((
        os.fspath(PILOT_GIT_BIN),
        os.fspath(Path(python).parent),
        "/usr/bin",
        "/bin",
    ))
    lines = [
        "#!/bin/bash",
        f"#PBS -N {task['scheduler_job_name']}",
        f"#PBS -P {PILOT_PBS_PROJECT}",
        f"#PBS -q {PILOT_PBS_QUEUE}",
        "#PBS -r n",
        f"#PBS -o {stdout}",
        f"#PBS -e {stderr}",
        (
            "#PBS -l select=1:"
            f"ncpus={PILOT_PBS_NCPUS}:mem={PILOT_PBS_MEMORY}"
        ),
        f"#PBS -l walltime={PILOT_PBS_WALLTIME}",
        "set -euo pipefail",
        "umask 077",
        f"export PATH={shlex.quote(staged_path)}",
        "export GIT_CONFIG_NOSYSTEM=1",
        "export GIT_CONFIG_GLOBAL=/dev/null",
        "export LC_ALL=C",
        "export TZ=UTC",
        "export PYTHONDONTWRITEBYTECODE=1",
        "export PYTHONNOUSERSITE=1",
        "export PYTHONHASHSEED=0",
        *(f"export {name}=1" for name in PILOT_THREAD_LIMITS),
        f"export PYTHONPATH={shlex.quote(repo + '/src:' + repo + '/scripts')}",
        f"cd -- {shlex.quote(repo)}",
        f"exec {shlex.join(command)}",
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


def validate_scheduler_receipt(
    manifest: Mapping[str, Any],
    task_id: str,
    path: str | Path,
) -> dict[str, Any]:
    """Authenticate one externally captured terminal PBS ``F``/0 receipt."""

    validated = validate_pilot_manifest(manifest)
    task = _task_from_manifest(validated, task_id)
    selected = _regular_file(path, name=f"{task_id} scheduler receipt")
    lexical_selected = selected
    selected = Path(os.path.abspath(selected))
    scheduler_root = selected.parent
    pilot_root = scheduler_root.parent.parent
    expected_scheduler_root = pilot_root / "scheduler" / task_id
    if (
        scheduler_root != expected_scheduler_root
        or selected != scheduler_root / "scheduler-terminal-receipt.json"
        or os.fspath(lexical_selected) != os.fspath(selected)
    ):
        raise MaterialNQ1PilotError("scheduler receipt path is not fixed")
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("scheduler receipt must have mode 0444")
    record = load_canonical_json(selected, name=f"{task_id} scheduler receipt")
    expected_keys = {
        "schema",
        "pilot_sha256",
        "task_id",
        "job_id",
        "job_name",
        "terminal_state",
        "exit_status",
        "raw_qstat_path",
        "raw_qstat_sha256",
        "task_receipt_path",
        "task_receipt_sha256",
        "pre_submission_path",
        "pre_submission_sha256",
        "submission_path",
        "submission_sha256",
        "pbs_script_path",
        "pbs_script_sha256",
        "pbs_stdout_path",
        "pbs_stdout_sha256",
        "pbs_stderr_path",
        "pbs_stderr_sha256",
        "submission_command_sha256",
        "workload_command_sha256",
        "release_certified",
    }
    if (
        set(record) != expected_keys
        or record.get("schema") != SCHEDULER_SCHEMA
        or record.get("pilot_sha256") != validated["pilot_sha256"]
        or record.get("task_id") != task_id
        or record.get("job_name") != task["scheduler_job_name"]
        or record.get("terminal_state") != "F"
        or record.get("exit_status") != 0
        or record.get("release_certified") is not False
        or not isinstance(record.get("job_id"), str)
        or _PBS_JOB_ID.fullmatch(record["job_id"]) is None
    ):
        raise MaterialNQ1PilotError("scheduler receipt is not exact terminal F/0 evidence")
    if record.get("raw_qstat_path") != os.fspath(
        scheduler_root / "qstat-terminal.txt"
    ):
        raise MaterialNQ1PilotError("raw qstat evidence path is not fixed")
    raw_path = _regular_file(record["raw_qstat_path"], name="raw qstat evidence")
    if stat.S_IMODE(raw_path.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("raw qstat evidence must have mode 0444")
    expected_raw = _validate_sha256(
        record.get("raw_qstat_sha256"), name="raw qstat SHA256"
    )
    if sha256_file(raw_path) != expected_raw:
        raise MaterialNQ1PilotError("raw qstat evidence hash changed")
    try:
        text = raw_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise MaterialNQ1PilotError("raw qstat evidence is unreadable") from error
    observed_lines = {
        "job_id": re.findall(r"^Job Id:\s*(\S+)\s*$", text, flags=re.MULTILINE),
        "job_name": re.findall(
            r"^\s*Job_Name\s*=\s*(\S+)\s*$", text, flags=re.MULTILINE
        ),
        "state": re.findall(
            r"^\s*job_state\s*=\s*(\S+)\s*$", text, flags=re.MULTILINE
        ),
        "exit": re.findall(
            r"^\s*Exit_status\s*=\s*(-?\d+)\s*$", text, flags=re.MULTILINE
        ),
    }
    if observed_lines != {
        "job_id": [record["job_id"]],
        "job_name": [record["job_name"]],
        "state": ["F"],
        "exit": ["0"],
    }:
        raise MaterialNQ1PilotError("raw qstat evidence does not reproduce F/0 identity")

    task_receipt_path = _regular_file(
        record.get("task_receipt_path"), name="scheduler-bound task receipt"
    )
    if (
        Path(os.path.abspath(task_receipt_path))
        != pilot_root / "tasks" / task_id / "task-receipt.json"
        or record.get("task_receipt_path")
        != os.fspath(pilot_root / "tasks" / task_id / "task-receipt.json")
        or stat.S_IMODE(task_receipt_path.stat().st_mode) != 0o444
        or sha256_file(task_receipt_path) != record.get("task_receipt_sha256")
    ):
        raise MaterialNQ1PilotError("scheduler receipt task binding changed")
    task_receipt = validate_task_receipt(validated, task_receipt_path)
    if task_receipt.get("task_id") != task_id or task_receipt.get("state") != "passed":
        raise MaterialNQ1PilotError("scheduler receipt binds another workload")
    workload = _scheduler_workload_contract(validated, task, task_receipt)

    pre_path = _regular_file(
        record.get("pre_submission_path"), name="pilot pre-submission record"
    )
    submission_path = _regular_file(
        record.get("submission_path"), name="pilot submission record"
    )
    pbs_path = _regular_file(record.get("pbs_script_path"), name="pilot PBS script")
    pbs_stdout_path = _regular_file(
        record.get("pbs_stdout_path"), name="pilot PBS stdout"
    )
    pbs_stderr_path = _regular_file(
        record.get("pbs_stderr_path"), name="pilot PBS stderr"
    )
    fixed_paths = {
        pre_path: scheduler_root / "pre-submission.json",
        submission_path: scheduler_root / "submission.json",
        pbs_path: scheduler_root / "task.pbs",
        pbs_stdout_path: scheduler_root / "task.stdout",
        pbs_stderr_path: scheduler_root / "task.stderr",
    }
    if any(
        Path(os.path.abspath(observed)) != expected
        or os.fspath(observed) != os.fspath(expected)
        for observed, expected in fixed_paths.items()
    ):
        raise MaterialNQ1PilotError("scheduler-bound artifact path is not fixed")
    for selected_record in (
        pre_path, submission_path, pbs_path, pbs_stdout_path, pbs_stderr_path
    ):
        if stat.S_IMODE(selected_record.stat().st_mode) != 0o444:
            raise MaterialNQ1PilotError("scheduler-bound artifacts must have mode 0444")
    if (
        sha256_file(pre_path) != record.get("pre_submission_sha256")
        or sha256_file(submission_path) != record.get("submission_sha256")
        or sha256_file(pbs_path) != record.get("pbs_script_sha256")
        or sha256_file(pbs_stdout_path) != record.get("pbs_stdout_sha256")
        or sha256_file(pbs_stderr_path) != record.get("pbs_stderr_sha256")
    ):
        raise MaterialNQ1PilotError("scheduler-bound artifact hash changed")
    pre = load_canonical_json(pre_path, name="pilot pre-submission record")
    submission = load_canonical_json(submission_path, name="pilot submission record")
    pre_keys = {
        "schema", "pilot_sha256", "task_id", "scientific_task_sha256",
        "job_name", "pbs_script_path", "pbs_script_sha256",
        "pilot_driver_path", "pilot_driver_sha256",
        "workload_command", "workload_command_sha256",
        "submission_command", "submission_command_sha256", "release_certified",
    }
    submission_keys = pre_keys | {
        "job_id", "pre_submission_path", "pre_submission_sha256",
        "qsub_stdout_path", "qsub_stdout_sha256",
        "qsub_stderr_path", "qsub_stderr_sha256",
    }
    command = ["qsub", os.fspath(pbs_path)]
    command_sha = sha256_bytes(canonical_json(command))
    common = {
        "pilot_sha256": validated["pilot_sha256"],
        "task_id": task_id,
        "scientific_task_sha256": task["task_sha256"],
        "job_name": task["scheduler_job_name"],
        "pbs_script_path": os.fspath(pbs_path),
        "pbs_script_sha256": sha256_file(pbs_path),
        "pilot_driver_path": workload["driver_path"],
        "pilot_driver_sha256": workload["driver_sha256"],
        "workload_command": workload["command"],
        "workload_command_sha256": workload["command_sha256"],
        "submission_command": command,
        "submission_command_sha256": command_sha,
        "release_certified": False,
    }
    qsub_stdout_path = _regular_file(
        submission.get("qsub_stdout_path"), name="qsub stdout"
    )
    qsub_stderr_path = _regular_file(
        submission.get("qsub_stderr_path"), name="qsub stderr"
    )
    expected_task_stdout = canonical_json({
        "task_id": task_id,
        "state": "passed",
        "anchor_comparison_passed": (
            True if task.get("anchor_task_id") is not None else None
        ),
        "release_certified": False,
    })
    if (
        Path(os.path.abspath(qsub_stdout_path)) != scheduler_root / "qsub.stdout"
        or Path(os.path.abspath(qsub_stderr_path)) != scheduler_root / "qsub.stderr"
        or os.fspath(qsub_stdout_path) != os.fspath(scheduler_root / "qsub.stdout")
        or os.fspath(qsub_stderr_path) != os.fspath(scheduler_root / "qsub.stderr")
        or stat.S_IMODE(qsub_stdout_path.stat().st_mode) != 0o444
        or stat.S_IMODE(qsub_stderr_path.stat().st_mode) != 0o444
        or sha256_file(qsub_stdout_path) != submission.get("qsub_stdout_sha256")
        or sha256_file(qsub_stderr_path) != submission.get("qsub_stderr_sha256")
        or qsub_stdout_path.read_bytes() != (record["job_id"] + "\n").encode("ascii")
        or qsub_stderr_path.read_bytes() != b""
        or pbs_stdout_path.read_bytes() != expected_task_stdout
        or pbs_stderr_path.read_bytes() != b""
        or set(pre) != pre_keys
        or pre != ({"schema": PRE_SUBMISSION_SCHEMA} | common)
        or set(submission) != submission_keys
        or submission != (
            {"schema": SUBMISSION_SCHEMA}
            | common
            | {
                "job_id": record["job_id"],
                "pre_submission_path": os.fspath(pre_path),
                "pre_submission_sha256": sha256_file(pre_path),
                "qsub_stdout_path": os.fspath(qsub_stdout_path),
                "qsub_stdout_sha256": sha256_file(qsub_stdout_path),
                "qsub_stderr_path": os.fspath(qsub_stderr_path),
                "qsub_stderr_sha256": sha256_file(qsub_stderr_path),
            }
        )
        or record.get("pbs_script_path") != os.fspath(pbs_path)
        or record.get("submission_command_sha256") != command_sha
        or record.get("workload_command_sha256") != workload["command_sha256"]
    ):
        raise MaterialNQ1PilotError("scheduler submission identity does not reproduce")
    expected_pbs = _canonical_pbs_script(
        task=task,
        stdout_path=pbs_stdout_path,
        stderr_path=pbs_stderr_path,
        workload=workload,
    )
    if pbs_path.read_bytes() != expected_pbs:
        raise MaterialNQ1PilotError("pilot PBS script is not the canonical workload")
    return record


def build_final_report(
    *,
    manifest: Mapping[str, Any],
    manifest_path: str | Path,
    preflight_path: str | Path,
    task_receipts: Mapping[str, str | Path],
    comparisons: Mapping[str, str | Path],
    scheduler_receipts: Mapping[str, str | Path],
) -> dict[str, Any]:
    validated = validate_pilot_manifest(manifest)
    manifest_selected = _regular_file(manifest_path, name="pilot manifest")
    pilot_root = Path(os.path.abspath(manifest_selected)).parent
    _require_fixed_absolute_path(
        manifest_selected,
        pilot_root / "pilot-manifest.json",
        name="pilot manifest",
    )
    if (
        stat.S_IMODE(manifest_selected.stat().st_mode) != 0o444
        or load_canonical_json(manifest_selected, name="pilot manifest") != validated
    ):
        raise MaterialNQ1PilotError("fixed pilot manifest binding changed")
    preflight_selected = _regular_file(preflight_path, name="pilot preflight")
    _require_fixed_absolute_path(
        preflight_selected,
        pilot_root / "pilot-preflight.json",
        name="pilot preflight",
    )
    validate_preflight_record(validated, preflight_selected)
    if set(task_receipts) != set(PILOT_TASK_IDS):
        raise MaterialNQ1PilotError("final report requires exactly three task receipts")
    receipt_records: dict[str, Any] = {}
    all_tasks_passed = True
    for task_id in PILOT_TASK_IDS:
        path = _regular_file(task_receipts[task_id], name=f"{task_id} receipt")
        _require_fixed_absolute_path(
            path,
            pilot_root / "tasks" / task_id / "task-receipt.json",
            name=f"{task_id} receipt",
        )
        receipt = validate_task_receipt(validated, path)
        if receipt["task_id"] != task_id:
            raise MaterialNQ1PilotError("task receipt map names another task")
        all_tasks_passed = all_tasks_passed and receipt["state"] == "passed"
        receipt_records[task_id] = {
            "path": os.fspath(path),
            "sha256": sha256_file(path),
            "state": receipt["state"],
            "scientific_task_sha256": receipt["scientific_task_sha256"],
        }
    expected_comparisons = {
        "anchor-coupling-g0001",
        "anchor-coupling-g001",
    }
    if set(comparisons) != expected_comparisons:
        raise MaterialNQ1PilotError("final report comparison set is incomplete")
    comparison_records: dict[str, Any] = {}
    all_comparisons_passed = True
    for comparison_id in sorted(comparisons):
        path = _regular_file(comparisons[comparison_id], name="pilot comparison")
        _require_fixed_absolute_path(
            path,
            pilot_root / "comparisons" / f"{comparison_id}.json",
            name="pilot comparison",
        )
        record = validate_comparison(validated, path)
        if (
            record.get("schema") != COMPARISON_SCHEMA
            or record.get("pilot_sha256") != validated["pilot_sha256"]
            or record.get("comparison_id") != comparison_id
            or record.get("release_certified") is not False
        ):
            raise MaterialNQ1PilotError("pilot comparison identity changed")
        all_comparisons_passed = (
            all_comparisons_passed and record.get("passed") is True
        )
        comparison_records[comparison_id] = {
            "path": os.fspath(path),
            "sha256": sha256_file(path),
            "passed": record.get("passed"),
        }
    if set(scheduler_receipts) != set(PILOT_TASK_IDS):
        raise MaterialNQ1PilotError(
            "final report requires one scheduler receipt for every pilot task"
        )
    scheduler_records: dict[str, Any] = {}
    for task_id in PILOT_TASK_IDS:
        path = _regular_file(
            scheduler_receipts[task_id], name=f"{task_id} scheduler receipt"
        )
        _require_fixed_absolute_path(
            path,
            pilot_root
            / "scheduler"
            / task_id
            / "scheduler-terminal-receipt.json",
            name=f"{task_id} scheduler receipt",
        )
        record = validate_scheduler_receipt(validated, task_id, path)
        scheduler_records[task_id] = {
            "path": os.fspath(path),
            "sha256": sha256_file(path),
            "job_id": record["job_id"],
            "job_name": record["job_name"],
            "terminal_state": "F",
            "exit_status": 0,
        }
    if len({item["job_id"] for item in scheduler_records.values()}) != len(
        PILOT_TASK_IDS
    ):
        raise MaterialNQ1PilotError("pilot tasks do not have distinct scheduler jobs")
    passed = bool(all_tasks_passed and all_comparisons_passed)
    return {
        "schema": REPORT_SCHEMA,
        "pilot_sha256": validated["pilot_sha256"],
        "execution_id": validated["execution_id"],
        "manifest": {
            "path": os.fspath(manifest_selected),
            "sha256": sha256_file(manifest_selected),
        },
        "preflight": {
            "path": os.fspath(preflight_selected),
            "sha256": sha256_file(preflight_selected),
        },
        "task_receipts": receipt_records,
        "comparisons": comparison_records,
        "scheduler_receipts": scheduler_records,
        "all_tasks_completed": all_tasks_passed,
        "all_scheduler_jobs_terminal_success": True,
        "strict_anchor_reproduction": all_comparisons_passed,
        "coupling_g0025_completed": (
            receipt_records["coupling-g0025"]["state"] == "passed"
        ),
        "pilot_passed": passed,
        "authorizes": {
            "scope": "consideration-and-submission-gating-only",
            "next_campaign": "separately-identified-full-19-task-campaign",
            "required_source_revision": validated["source_revision"],
            "required_source_tree_sha256": validated["source_tree_sha256"],
            "requires_identical_qualified_commit_and_tree": True,
            "source_or_scientific_control_changes_allowed": False,
        },
        "release_certified": False,
    }


def validate_final_report(
    manifest: Mapping[str, Any], path: str | Path
) -> dict[str, Any]:
    """Recursively reproduce a final report from every referenced artifact."""

    validated = validate_pilot_manifest(manifest)
    selected = _regular_file(path, name="pilot final report")
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("pilot final report must have mode 0444")
    report = load_canonical_json(selected, name="pilot final report")
    expected_keys = {
        "schema",
        "pilot_sha256",
        "execution_id",
        "manifest",
        "preflight",
        "task_receipts",
        "comparisons",
        "scheduler_receipts",
        "all_tasks_completed",
        "all_scheduler_jobs_terminal_success",
        "strict_anchor_reproduction",
        "coupling_g0025_completed",
        "pilot_passed",
        "authorizes",
        "release_certified",
    }
    if (
        set(report) != expected_keys
        or report.get("schema") != REPORT_SCHEMA
        or not isinstance(report.get("manifest"), Mapping)
        or not isinstance(report.get("preflight"), Mapping)
        or not isinstance(report.get("task_receipts"), Mapping)
        or not isinstance(report.get("comparisons"), Mapping)
        or not isinstance(report.get("scheduler_receipts"), Mapping)
    ):
        raise MaterialNQ1PilotError("pilot final report schema is invalid")
    manifest_record_path = Path(str(report["manifest"].get("path")))
    pilot_root = manifest_record_path.parent
    _require_fixed_absolute_path(
        manifest_record_path,
        pilot_root / "pilot-manifest.json",
        name="pilot report manifest",
    )
    _require_fixed_absolute_path(
        selected, pilot_root / "pilot-report.json", name="pilot final report"
    )
    task_paths = {
        task_id: str(item.get("path"))
        for task_id, item in report["task_receipts"].items()
        if isinstance(item, Mapping)
    }
    comparison_paths = {
        comparison_id: str(item.get("path"))
        for comparison_id, item in report["comparisons"].items()
        if isinstance(item, Mapping)
    }
    scheduler_paths = {
        task_id: str(item.get("path"))
        for task_id, item in report["scheduler_receipts"].items()
        if isinstance(item, Mapping)
    }
    rebuilt = build_final_report(
        manifest=validated,
        manifest_path=str(report["manifest"].get("path")),
        preflight_path=str(report["preflight"].get("path")),
        task_receipts=task_paths,
        comparisons=comparison_paths,
        scheduler_receipts=scheduler_paths,
    )
    if report != rebuilt:
        raise MaterialNQ1PilotError("pilot final report does not reproduce")
    return rebuilt


def _inventory(root: Path, *, excluded: Sequence[Path] = ()) -> dict[str, Any]:
    excluded_resolved = {path.absolute() for path in excluded}
    records: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if path.absolute() in excluded_resolved:
            continue
        status_result = path.lstat()
        if stat.S_ISLNK(status_result.st_mode):
            raise MaterialNQ1PilotError(f"pilot tree contains a symlink: {path}")
        if stat.S_ISREG(status_result.st_mode):
            size = status_result.st_size
            total += size
            records.append({
                "relative_path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size": size,
            })
        elif not stat.S_ISDIR(status_result.st_mode):
            raise MaterialNQ1PilotError(f"pilot tree contains a special file: {path}")
    digest = sha256_bytes(canonical_json(records))
    return {
        "file_count": len(records),
        "total_bytes": total,
        "records": records,
        "inventory_sha256": digest,
    }


def build_audit(
    *,
    manifest: Mapping[str, Any],
    root: str | Path,
    report_path: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    validated = validate_pilot_manifest(manifest)
    selected_root = _directory(root, name="pilot root")
    fixed_root = Path(os.path.abspath(selected_root))
    _require_fixed_absolute_path(selected_root, fixed_root, name="pilot root")
    report_selected = _regular_file(report_path, name="pilot report")
    _require_fixed_absolute_path(
        report_selected, fixed_root / "pilot-report.json", name="pilot report"
    )
    fixed_audit_path = fixed_root / "pilot-audit.json"
    _require_fixed_absolute_path(audit_path, fixed_audit_path, name="pilot audit")
    report = validate_final_report(validated, report_selected)
    if (
        report.get("schema") != REPORT_SCHEMA
        or report.get("pilot_sha256") != validated["pilot_sha256"]
        or report.get("pilot_passed") is not True
        or report.get("release_certified") is not False
    ):
        raise MaterialNQ1PilotError("pilot report is not a passed non-release report")
    # The audit excludes its own not-yet-created path; every other byte is
    # covered.  Re-running after publication reproduces the same inventory.
    inventory = _inventory(selected_root, excluded=(fixed_audit_path,))
    return {
        "schema": AUDIT_SCHEMA,
        "pilot_sha256": validated["pilot_sha256"],
        "report": {
            "path": os.fspath(report_selected),
            "sha256": sha256_file(report_selected),
        },
        "inventory": inventory,
        "checks": {
            "pilot_report_passed": True,
            "all_files_hashed": True,
            "no_symlinks_or_special_files": True,
            "release_claim_absent": report["release_certified"] is False,
        },
        "audit_passed": True,
        "release_certified": False,
    }


def validate_audit(
    manifest: Mapping[str, Any],
    *,
    root: str | Path,
    audit_path: str | Path,
) -> dict[str, Any]:
    """Reproduce the recursively authenticated final audit exactly."""

    validated = validate_pilot_manifest(manifest)
    selected_root = _directory(root, name="pilot root")
    fixed_root = Path(os.path.abspath(selected_root))
    _require_fixed_absolute_path(selected_root, fixed_root, name="pilot root")
    selected = _regular_file(audit_path, name="pilot audit")
    _require_fixed_absolute_path(
        selected, fixed_root / "pilot-audit.json", name="pilot audit"
    )
    if stat.S_IMODE(selected.stat().st_mode) != 0o444:
        raise MaterialNQ1PilotError("pilot audit must have mode 0444")
    record = load_canonical_json(selected, name="pilot audit")
    if (
        record.get("schema") != AUDIT_SCHEMA
        or record.get("pilot_sha256") != validated["pilot_sha256"]
        or not isinstance(record.get("report"), Mapping)
        or record.get("audit_passed") is not True
        or record.get("release_certified") is not False
    ):
        raise MaterialNQ1PilotError("pilot audit schema is invalid")
    rebuilt = build_audit(
        manifest=validated,
        root=selected_root,
        report_path=str(record["report"].get("path")),
        audit_path=selected,
    )
    if record != rebuilt:
        raise MaterialNQ1PilotError("pilot audit does not reproduce")
    return rebuilt


__all__ = [
    "AUDIT_SCHEMA",
    "COMPARISON_SCHEMA",
    "FIELD_ATOL",
    "FIELD_DATASETS",
    "FIELD_RTOL",
    "PILOT_PBS_MEMORY",
    "PILOT_PBS_NCPUS",
    "PILOT_PBS_WALLTIME",
    "PILOT_RSS_LIMIT_BYTES",
    "PILOT_SCHEMA",
    "PILOT_TASK_IDS",
    "PILOT_WALL_LIMIT_SECONDS",
    "PREFLIGHT_SCHEMA",
    "PRE_SUBMISSION_SCHEMA",
    "QUALIFICATION_SCHEMA",
    "RELEASE_018_ANCHORS",
    "RELEASE_018_ANCHOR_ROOT",
    "REPORT_SCHEMA",
    "SCHEDULER_SCHEMA",
    "SUBMISSION_SCHEMA",
    "TASK_RECEIPT_SCHEMA",
    "MaterialNQ1PilotError",
    "anchor_task_id",
    "authenticate_anchor_source",
    "build_audit",
    "build_comparison",
    "build_final_report",
    "build_pilot_manifest",
    "build_preflight_record",
    "build_task_receipt",
    "canonical_json",
    "compare_currents",
    "compare_fields",
    "conservation_record",
    "load_canonical_json",
    "predictor_pilot_tasks",
    "publish_json_once",
    "qualification_authorizes_pilot",
    "sha256_file",
    "validate_audit",
    "validate_comparison",
    "validate_final_report",
    "validate_pilot_manifest",
    "validate_preflight_record",
    "validate_scheduler_receipt",
    "validate_task_receipt",
]
