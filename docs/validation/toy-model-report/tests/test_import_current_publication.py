"""Current-attempt topology tests for the report artifact importer."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "import_production_toy_gate.py"
)
SPEC = importlib.util.spec_from_file_location(
    "negfpy_report_current_publication_importer", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
IMPORTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = IMPORTER
SPEC.loader.exec_module(IMPORTER)

COMMIT = "a" * 40


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )


def _canonical_bytes(value: object) -> bytes:
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


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value).rstrip(b"\n")).hexdigest()


def _embedded_manifest(value: dict[str, object]) -> dict[str, object]:
    manifest = dict(value)
    manifest["manifest_sha256"] = _canonical_sha256(manifest)
    return manifest


def _write_hashed_manifest(path: Path, payload: dict[str, object]) -> dict[str, object]:
    document = {
        "payload": payload,
        "payload_sha256": _canonical_sha256(payload),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(document))
    return document


def _source_and_prefix_manifests() -> tuple[dict[str, object], dict[str, object]]:
    records = [
        {
            "path": path,
            "git_tracked": True,
            "kind": "regular_file",
            "size": 100 + index,
            "mode": 0o644,
            "sha256": hashlib.sha256(path.encode("ascii")).hexdigest(),
        }
        for index, path in enumerate(IMPORTER._ORCHESTRATION_SCRIPT_PATHS)
    ]
    tracked = _embedded_manifest(
        {
            "schema": "negfpy.remote_validation.tracked_worktree.v1",
            "records": records,
        }
    )
    worktree = _embedded_manifest(
        {
            "schema": "negfpy.remote_validation.full_worktree.v1",
            "records": records,
        }
    )
    orchestration = _embedded_manifest(
        {
            "schema": "negfpy.remote_validation.orchestration_sources.v1",
            "records": records,
        }
    )
    source = _embedded_manifest(
        {
            "schema": "negfpy.remote_validation.execution_source.v1",
            "git_revision": COMMIT,
            "git_tree": "b" * 40,
            "package_source_sha256": "1" * 64,
            "tracked_file_count": len(records),
            "tracked_files_manifest_sha256": tracked["manifest_sha256"],
            "tracked_files": records,
            "worktree_file_count": len(records),
            "worktree_files_manifest_sha256": worktree["manifest_sha256"],
            "worktree_files": records,
            "orchestration_manifest_sha256": orchestration["manifest_sha256"],
            "orchestration_scripts": records,
        }
    )
    prefix = "/home/hangbo/negfpy-validation/synthetic/environment"
    environment_files = [
        {"path": "bin/mpiexec", "size": 11, "sha256": "2" * 64},
        {"path": "bin/python", "size": 12, "sha256": "3" * 64},
    ]
    environment = _embedded_manifest(
        {
            "schema": "negfpy.remote_validation.environment_prefix.v1",
            "prefix": prefix,
            "directory_count": 3,
            "regular_file_count": len(environment_files),
            "symlink_count": 0,
            "regular_files": environment_files,
            "symlinks": [],
        }
    )
    return source, environment


def _release_chain(
    current_root: Path,
    *,
    source: dict[str, object],
    environment: dict[str, object],
) -> dict[str, object]:
    prefix = str(environment["prefix"])
    remote_root = "/home/hangbo/negfpy-validation/synthetic"
    remote_attempt = f"{remote_root}/artifacts/attempts/attempt-002"
    runner_record = next(
        record
        for record in source["orchestration_scripts"]
        if record["path"] == "scripts/remote_toy_mpi_runner.py"
    )
    launcher = {
        "path": f"{prefix}/bin/mpiexec",
        "resolved_path": f"{prefix}/bin/mpiexec",
        "sha256": "2" * 64,
    }
    python = {
        "path": f"{prefix}/bin/python",
        "resolved_path": f"{prefix}/bin/python",
        "sha256": "3" * 64,
    }
    runner = {
        "path": f"{remote_root}/worktree/scripts/remote_toy_mpi_runner.py",
        "resolved_path": f"{remote_root}/worktree/scripts/remote_toy_mpi_runner.py",
        "size": runner_record["size"],
        "sha256": runner_record["sha256"],
    }
    binding_arguments = [
        "--nooversubscribe",
        "--bind-to",
        "core",
        "--map-by",
        "ppr:2:socket:PE=1",
        "--report-bindings",
    ]
    argv = [
        launcher["path"],
        *binding_arguments,
        "-n",
        "4",
        python["path"],
        runner["path"],
        "release",
        "--run-directory",
        f"{remote_root}/artifacts/toy-release",
        "--source-revision",
        COMMIT,
        "--source-tree-sha256",
        "1" * 64,
        "--environment-sha256",
        "4" * 64,
        "--execution-id",
        "release-attempt-002",
        "--expected-ranks",
        "4",
    ]
    launch_environment = {
        "HOME": "/home/hangbo",
        "USER": "hangbo",
        "LOGNAME": "hangbo",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "PATH": f"{prefix}/bin:/usr/local/bin:/usr/bin:/bin",
        "PYTHONPATH": f"{remote_root}/worktree/src",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONPYCACHEPREFIX": f"{remote_attempt}/runtime-cache/pycache",
        "TMPDIR": f"{remote_attempt}/runtime-tmp",
        "XDG_CACHE_HOME": f"{remote_attempt}/runtime-cache",
        "MPLCONFIGDIR": f"{remote_attempt}/runtime-config/matplotlib",
        **{name: "1" for name in IMPORTER._THREAD_LIMIT_KEYS},
    }
    launch_environment = dict(sorted(launch_environment.items()))
    binding = {
        "schema": "negfpy.remote_validation.release_launch_binding.v1",
        "source_revision": COMMIT,
        "source_tree_sha256": "1" * 64,
        "environment_sha256": "4" * 64,
        "execution_id": "release-attempt-002",
        "selected_rank_count": 4,
        "argv": argv,
        "binding_arguments": binding_arguments,
        "thread_limits": {name: "1" for name in IMPORTER._THREAD_LIMIT_KEYS},
        "launcher": launcher,
        "python": python,
        "runner": runner,
        "execution_source_identity_sha256": source["manifest_sha256"],
        "tracked_files_manifest_sha256": source["tracked_files_manifest_sha256"],
        "worktree_files_manifest_sha256": source["worktree_files_manifest_sha256"],
        "orchestration_manifest_sha256": source["orchestration_manifest_sha256"],
        "orchestration_scripts": source["orchestration_scripts"],
        "environment_prefix_manifest_sha256": environment["manifest_sha256"],
        "environment_regular_file_count": environment["regular_file_count"],
        "environment_symlink_count": environment["symlink_count"],
        "launch_environment": launch_environment,
        "launch_environment_sha256": _canonical_sha256(launch_environment),
    }
    intent_path = current_root / "mpi-release-launch-intent.json"
    intent = _write_hashed_manifest(
        intent_path,
        {
            "schema": "negfpy.remote_validation.release_launch_intent.v1",
            "launch_binding": binding,
            "stdout_relative_path": "mpi-release.stdout.log",
            "stderr_relative_path": "mpi-release.stderr.log",
        },
    )
    stdout = current_root / "mpi-release.stdout.log"
    stderr = current_root / "mpi-release.stderr.log"
    stdout.write_text("synthetic release stdout\n", encoding="ascii")
    stderr.write_text("", encoding="ascii")
    receipt_path = current_root / "mpi-release-launch.json"
    receipt = _write_hashed_manifest(
        receipt_path,
        {
            "schema": "negfpy.remote_validation.release_launch.v2",
            "launch_binding": binding,
            "intent": {
                "relative_path": intent_path.name,
                "sha256": _sha(intent_path),
            },
            "intent_payload_sha256": intent["payload_sha256"],
            "started_utc": "2026-07-26T00:00:00Z",
            "finished_utc": "2026-07-26T00:00:01Z",
            "returncode": 0,
            "post_run_verification": {
                "passed": True,
                "checked_utc": "2026-07-26T00:00:01Z",
                "execution_source_identity_sha256": source["manifest_sha256"],
                "environment_prefix_manifest_sha256": environment["manifest_sha256"],
            },
            "stdout": {"relative_path": stdout.name, "sha256": _sha(stdout)},
            "stderr": {"relative_path": stderr.name, "sha256": _sha(stderr)},
        },
    )
    launch_facts = {
        "schema": "negfpy.remote_validation.release_launch_facts.v1",
        "receipt_relative_path": receipt_path.name,
        "receipt_file_sha256": _sha(receipt_path),
        "receipt_payload_sha256": receipt["payload_sha256"],
        "intent_relative_path": intent_path.name,
        "intent_file_sha256": _sha(intent_path),
        "intent_payload_sha256": intent["payload_sha256"],
        "launcher_file_sha256": launcher["sha256"],
        "python_file_sha256": python["sha256"],
        "runner_file_sha256": runner["sha256"],
        "execution_source_identity_sha256": source["manifest_sha256"],
        "tracked_files_manifest_sha256": source["tracked_files_manifest_sha256"],
        "worktree_files_manifest_sha256": source["worktree_files_manifest_sha256"],
        "orchestration_manifest_sha256": source["orchestration_manifest_sha256"],
        "environment_prefix_manifest_sha256": environment["manifest_sha256"],
        "launch_environment_sha256": binding["launch_environment_sha256"],
        "stdout_relative_path": stdout.name,
        "stdout_file_sha256": _sha(stdout),
        "stderr_relative_path": stderr.name,
        "stderr_file_sha256": _sha(stderr),
        "selected_rank_count": 4,
        "binding_arguments": binding_arguments,
        "post_run_verified": True,
        "returncode": 0,
    }
    return launch_facts


def _current_publication(root: Path) -> dict[str, object]:
    """Create the dc60+ topology, including a reusable prior scaling attempt."""

    prior = root / "attempts" / "attempt-001"
    current_root = root / "attempts" / "attempt-002"
    scaling = prior / "mpi-scaling" / "scaling-selection.json"
    _write_json(scaling, {"selected_rank": 4})

    source, environment = _source_and_prefix_manifests()
    _write_json(
        current_root / "execution-identity.json",
        {
            "schema": "negfpy.remote_validation.remote_environment.v1",
            "commit": COMMIT,
            "tree": "b" * 40,
            "source_tree_sha256": "1" * 64,
            "explicit_lock_sha256": "4" * 64,
            "execution_source_identity": source,
            "environment_prefix_manifest": environment,
        },
    )
    report = current_root / "production_toy_gate.json"
    _write_json(report, {"release_certified": True})
    launch_facts = _release_chain(
        current_root, source=source, environment=environment
    )
    mpi_facts = {
        "release_certified": True,
        "communicator_size": 4,
        "execution_id": "release-attempt-002",
        "launch_receipt": launch_facts,
    }
    facts = {
        "schema": "negfpy.remote_validation.report_facts.v1",
        "report_relative_path": (
            "attempts/attempt-002/production_toy_gate.json"
        ),
        "report_file_sha256": _sha(report),
        "commit": COMMIT,
        "mpi_execution": mpi_facts,
    }
    _write_json(current_root / "report-facts.json", facts)
    _write_json(
        current_root / "checkpoint-index.json",
        {
            "schema": "negfpy.remote_validation.checkpoint_index.v1",
            "attempt": 2,
            "checkpoint_root": IMPORTER._REMOTE_CHECKPOINT_ROOT,
            "count": 13,
            "records": [],
        },
    )
    (current_root / "production_toy_gate.log").write_text(
        "completed\n", encoding="ascii"
    )
    named_files = {
        "mpi_release_launch_intent": "mpi-release-launch-intent.json",
        "mpi_release_launch": "mpi-release-launch.json",
        "mpi_release_stdout": "mpi-release.stdout.log",
        "mpi_release_stderr": "mpi-release.stderr.log",
        "mpi_runner_result": "mpi-release-runner-result.json",
        "mpi_execution_manifest": "mpi-execution-manifest.json",
        "mpi_campaign_manifest": "mpi-campaign-manifest.json",
    }
    for key, name in named_files.items():
        target = current_root / name
        if target.is_file():
            continue
        if target.suffix == ".log":
            target.write_text(f"{key}\n", encoding="ascii")
        else:
            _write_json(target, {"artifact": key})

    status = {
        "schema": "negfpy.remote_validation.job_status.v2",
        "state": "completed",
        "commit": COMMIT,
        "attempt": 2,
        "attempt_path": "attempts/attempt-002",
        "returncode": 0,
        "report_facts": facts,
        "scaling_facts": {
            "selection_relative_path": (
                "attempts/attempt-001/mpi-scaling/scaling-selection.json"
            )
        },
        "mpi_execution_facts": mpi_facts,
    }
    _write_json(current_root / "status.json", status)
    _write_json(root / "job-status.json", status)

    publication: dict[str, object] = {
        "schema": "negfpy.remote_validation.attempt_publication.v1",
        "attempt": 2,
        "state": "completed",
        "commit": COMMIT,
        "attempt_path": "attempts/attempt-002",
        "status": "attempts/attempt-002/status.json",
        "status_sha256": _sha(current_root / "status.json"),
        "log": "attempts/attempt-002/production_toy_gate.log",
        "log_sha256": _sha(current_root / "production_toy_gate.log"),
        "report": "attempts/attempt-002/production_toy_gate.json",
        "report_sha256": _sha(report),
        "report_facts": facts,
        "scaling_selection": (
            "attempts/attempt-001/mpi-scaling/scaling-selection.json"
        ),
        "scaling_selection_sha256": _sha(scaling),
    }
    for key, name in named_files.items():
        path = current_root / name
        publication[key] = f"attempts/attempt-002/{name}"
        publication[f"{key}_sha256"] = _sha(path)
    _write_json(current_root / "publication.json", publication)
    _write_json(root / "current-attempt.json", publication)
    return publication


def _refresh_publication_hashes(root: Path, *keys: str) -> None:
    publication_path = root / "attempts/attempt-002/publication.json"
    publication = json.loads(publication_path.read_text(encoding="ascii"))
    for key in keys:
        relative = str(publication[key])
        publication[f"{key}_sha256"] = _sha(root / relative)
    _write_json(publication_path, publication)
    _write_json(root / "current-attempt.json", publication)


def test_current_publication_accepts_prior_attempt_scaling_and_full_mpi_chain(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    publication = _current_publication(root)
    result = IMPORTER._validate_publication(
        root.resolve(strict=True), expected_commit=COMMIT
    )
    assert result[0] == COMMIT
    assert result[1] == 2
    assert result[2].relative_to(root).as_posix() == publication["report"]
    assert result[4]["mpi_execution"]["release_certified"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("post_run_failed", "post-run source/environment revalidation did not pass"),
        ("source_drift", "execution source identity differs from the release source"),
        (
            "environment_drift",
            "MPI launcher or Python differs from the environment-prefix manifest",
        ),
        (
            "launch_environment_tamper",
            "scientific launch environment is not the exact scrubbed environment",
        ),
    ),
)
def test_v2_release_provenance_tampering_is_rejected(
    tmp_path: Path, mutation: str, message: str
) -> None:
    root = tmp_path / "artifacts"
    _current_publication(root)
    attempt = root / "attempts/attempt-002"
    if mutation == "post_run_failed":
        receipt_path = attempt / "mpi-release-launch.json"
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        receipt["payload"]["post_run_verification"]["passed"] = False
        _write_hashed_manifest(receipt_path, receipt["payload"])
        _refresh_publication_hashes(root, "mpi_release_launch")
    elif mutation == "source_drift":
        identity_path = attempt / "execution-identity.json"
        identity = json.loads(identity_path.read_text(encoding="ascii"))
        source = dict(identity["execution_source_identity"])
        source["git_revision"] = "c" * 40
        source.pop("manifest_sha256")
        identity["execution_source_identity"] = _embedded_manifest(source)
        _write_json(identity_path, identity)
    elif mutation == "environment_drift":
        identity_path = attempt / "execution-identity.json"
        identity = json.loads(identity_path.read_text(encoding="ascii"))
        environment = dict(identity["environment_prefix_manifest"])
        files = [dict(record) for record in environment["regular_files"]]
        files[0]["sha256"] = "f" * 64
        environment["regular_files"] = files
        environment.pop("manifest_sha256")
        identity["environment_prefix_manifest"] = _embedded_manifest(environment)
        _write_json(identity_path, identity)
    elif mutation == "launch_environment_tamper":
        intent_path = attempt / "mpi-release-launch-intent.json"
        intent = json.loads(intent_path.read_text(encoding="ascii"))
        binding = intent["payload"]["launch_binding"]
        binding["launch_environment"]["HOME"] = "/home/intruder"
        rewritten_intent = _write_hashed_manifest(intent_path, intent["payload"])
        receipt_path = attempt / "mpi-release-launch.json"
        receipt = json.loads(receipt_path.read_text(encoding="ascii"))
        receipt["payload"]["launch_binding"] = binding
        receipt["payload"]["intent"] = {
            "relative_path": intent_path.name,
            "sha256": _sha(intent_path),
        }
        receipt["payload"]["intent_payload_sha256"] = rewritten_intent[
            "payload_sha256"
        ]
        _write_hashed_manifest(receipt_path, receipt["payload"])
        _refresh_publication_hashes(
            root, "mpi_release_launch_intent", "mpi_release_launch"
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(IMPORTER.ArtifactImportError, match=message):
        IMPORTER._validate_publication(
            root.resolve(strict=True), expected_commit=COMMIT
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("failed", "not completed"),
        ("current_mismatch", "differs from the selected attempt publication"),
        ("newer_attempt", "not the newest immutable attempt"),
        ("stale_root_report", "stale root publication artifact"),
        ("mpi_tamper", "mpi_runner_result SHA256 differs"),
    ),
)
def test_stale_failed_or_mismatched_current_publication_is_rejected(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    root = tmp_path / "artifacts"
    publication = _current_publication(root)
    if mutation == "failed":
        publication["state"] = "failed"
        _write_json(root / "current-attempt.json", publication)
        _write_json(root / "attempts/attempt-002/publication.json", publication)
    elif mutation == "current_mismatch":
        publication["extra"] = "stale pointer"
        _write_json(root / "current-attempt.json", publication)
    elif mutation == "newer_attempt":
        (root / "attempts/attempt-003").mkdir()
    elif mutation == "stale_root_report":
        _write_json(root / "production_toy_gate.json", {"stale": True})
    elif mutation == "mpi_tamper":
        (root / "attempts/attempt-002/mpi-release-runner-result.json").write_text(
            "tampered\n", encoding="ascii"
        )
    else:  # pragma: no cover
        raise AssertionError(mutation)
    with pytest.raises(IMPORTER.ArtifactImportError, match=message):
        IMPORTER._validate_publication(
            root.resolve(strict=True), expected_commit=COMMIT
        )


def test_exhaustive_fetched_index_preserves_every_path_size_and_hash(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    (root / "nested").mkdir(parents=True)
    (root / "a.txt").write_bytes(b"alpha\n")
    (root / "nested/b.bin").write_bytes(b"\x00\x01\x02")
    entries = {
        "a.txt": _sha(root / "a.txt"),
        "nested/b.bin": _sha(root / "nested/b.bin"),
    }
    raw = IMPORTER._fetched_artifact_index(
        root=root,
        entries=entries,
        sha256sums_sha256="1" * 64,
        fetch_receipt_sha256="2" * 64,
    )
    document = json.loads(raw)
    assert document["artifact_count"] == 2
    assert [row["relative_path"] for row in document["artifacts"]] == [
        "a.txt",
        "nested/b.bin",
    ]
    assert {row["sha256"] for row in document["artifacts"]} == set(
        entries.values()
    )
    assert document["artifacts"][1]["bytes"] == 3


def _seal_remote_fetch(root: Path) -> tuple[str, str]:
    files = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "fetch-receipt.json"}
    )
    manifest = root / "SHA256SUMS"
    manifest.write_text(
        "".join(f"{_sha(root / name)}  {name}\n" for name in files),
        encoding="ascii",
    )
    receipt = root / "fetch-receipt.json"
    _write_json(
        receipt,
        {
            "schema": "negfpy.remote_validation.fetch_receipt.v1",
            "fetched_utc": "2026-07-26T00:00:00Z",
            "commit": COMMIT,
            "remote": "~/negfpy-validation/synthetic/artifacts",
            "verified_files": files,
        },
    )
    return _sha(receipt), _sha(manifest)


def test_outer_manifest_accepts_and_retains_the_exhaustive_current_topology(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _current_publication(root)
    fetch_sha, manifest_sha = _seal_remote_fetch(root)
    entries, receipt = IMPORTER._verify_outer_sha256sums(
        root.resolve(strict=True),
        expected_commit=COMMIT,
        expected_environment_lock_sha256="3" * 64,
        expected_fetch_receipt_sha256=fetch_sha,
        expected_sha256sums_sha256=manifest_sha,
    )
    expected = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
        and path.name not in {"SHA256SUMS", "fetch-receipt.json"}
    }
    assert set(entries) == expected
    assert tuple(receipt["verified_files"]) == tuple(sorted(expected))


def test_outer_manifest_rejects_an_unlisted_artifact_after_fetch(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    _current_publication(root)
    fetch_sha, manifest_sha = _seal_remote_fetch(root)
    (root / "attempts/attempt-002/unlisted.log").write_text(
        "not in remote publication barrier\n", encoding="ascii"
    )
    with pytest.raises(IMPORTER.ArtifactImportError, match="cover exactly"):
        IMPORTER._verify_outer_sha256sums(
            root.resolve(strict=True),
            expected_commit=COMMIT,
            expected_environment_lock_sha256="3" * 64,
            expected_fetch_receipt_sha256=fetch_sha,
            expected_sha256sums_sha256=manifest_sha,
        )


def test_importer_source_version_is_fixed_and_hashable() -> None:
    assert IMPORTER._IMPORTER_VERSION == "3"
    assert IMPORTER._sha256_file(SCRIPT) == hashlib.sha256(
        SCRIPT.read_bytes()
    ).hexdigest()


def test_import_receipt_binds_staged_and_remote_report_paths() -> None:
    receipt = IMPORTER._build_receipt(
        commit=COMMIT,
        attempt=2,
        report_relative_path=(
            "attempts/attempt-002/production_toy_gate.json"
        ),
        report_file_sha256="1" * 64,
        report_body_sha256="2" * 64,
        nested_body_sha256="3" * 64,
        outer_manifest_sha256="4" * 64,
        fetch_receipt={
            "schema": "negfpy.remote_validation.fetch_receipt.v1",
            "fetched_utc": "2026-07-26T00:00:00Z",
        },
        expected_commit=COMMIT,
        expected_environment_lock_sha256="5" * 64,
        expected_fetch_receipt_sha256="6" * 64,
        expected_sha256sums_sha256="4" * 64,
        facts={
            "open_system_run_count": 13,
            "multidimensional_outcome_count": 13,
            "checkpoint_count": 13,
        },
        identity_bundle={
            "git_tree": "b" * 40,
            "source_tree_sha256": "7" * 64,
            "explicit_lock_sha256": "5" * 64,
        },
        checkpoint_summaries=[],
        generated_files={"production_toy_gate.json": b"{}\n"},
    )
    source = receipt["source"]
    assert source["report_relative_path"] == "production_toy_gate.json"
    assert source["remote_report_relative_path"] == (
        "attempts/attempt-002/production_toy_gate.json"
    )
    assert receipt["implementation"] == {
        "name": SCRIPT.name,
        "version": IMPORTER._IMPORTER_VERSION,
        "source_sha256": hashlib.sha256(SCRIPT.read_bytes()).hexdigest(),
    }
