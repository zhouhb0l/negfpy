#!/usr/bin/env python3
"""Seal and run exactly one remote silicon Phase-B mapping outcome.

The coordinator verifies a clean exact worktree, import origin, the pinned
material environment lock, authenticated inputs, and output mechanics before
starting the expensive full-FC3 mapping subprocess.  A complete scientific
outcome is immutable and is never rerun; partial or mismatched output pairs
fail closed for explicit operator inspection.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import negfpy

import validate_sparse_nq1_phase_b as phase_b


SCHEMA = "negfpy.sparse-nq1-phase-b-campaign.v1"
MATERIAL_LOCK_SHA256 = (
    "6d8e017b6f1ead8be3ec9fb8abb1c8c1d02fe0de852aad9cbfc69d08ce341bfc"
)
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class CampaignError(RuntimeError):
    """A fail-closed Phase-B orchestration or provenance error."""


def _next_gate(phase_b_certified: bool) -> str:
    if phase_b_certified:
        return "remote one-bubble/one-shot timing and memory preflight"
    return (
        "resolve the Phase-B scientific failure on a new exact commit before "
        "one-bubble work"
    )


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_once(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != content:
            raise CampaignError(f"immutable artifact mismatch: {path}")
        return
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise CampaignError(f"immutable artifact raced: {path}")
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(repo), *arguments),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise CampaignError(f"git {' '.join(arguments)} failed: {detail}")
    return completed.stdout.strip()


def _source_identity(repo: Path, expected_commit: str) -> dict[str, Any]:
    expected = expected_commit.strip().lower()
    if (
        len(expected) != 40
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise CampaignError("--expected-commit must be a full lowercase SHA-1.")
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != expected:
        raise CampaignError(f"source commit mismatch: {commit} != {expected}")
    dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise CampaignError("remote scientific worktree is not clean.")
    package_path = Path(negfpy.__file__).resolve()
    expected_package = (repo / "src" / "negfpy").resolve()
    try:
        package_path.relative_to(expected_package)
    except ValueError as error:
        raise CampaignError(
            f"negfpy imported from {package_path}, not {expected_package}"
        ) from error
    return {
        "commit": commit,
        "tree": _git(repo, "rev-parse", "HEAD^{tree}"),
        "dirty": False,
        "package_file": str(package_path),
    }


def _validate_lock(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CampaignError(f"material environment lock is not a regular file: {path}")
    first = next(
        (
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ),
        None,
    )
    if first != "@EXPLICIT":
        raise CampaignError("material environment lock is not an explicit lock.")
    digest = _sha256_file(path)
    if digest != MATERIAL_LOCK_SHA256:
        raise CampaignError(
            f"material environment lock SHA256 mismatch: {digest} != {MATERIAL_LOCK_SHA256}"
        )
    return {"path": str(path.resolve()), "sha256": digest, "format": "@EXPLICIT"}


def _versions() -> dict[str, str]:
    names = (
        "h5py",
        "numpy",
        "phonopy",
        "phono3py",
        "scipy",
        "spglib",
        "threadpoolctl",
    )
    try:
        versions = {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise CampaignError(f"missing material-lock distribution: {error.name}") from error
    for name in ("phonopy", "phono3py"):
        if versions[name] != "4.4.0":
            raise CampaignError(f"{name} version {versions[name]} is not pinned 4.4.0.")
    return versions


def build_manifest(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    artifacts = artifacts.resolve()
    output_json = artifacts / "outcomes" / "phase-b-mapping.json"
    output_hdf5 = artifacts / "outcomes" / "phase-b-mapped-fc3.h5"
    source = _source_identity(repo, expected_commit)
    lock = _validate_lock(environment_lock.resolve())
    preflight = phase_b.preflight_phase_b_inputs(
        fixture=fixture.resolve(),
        fc3_full=fc3_full.resolve(),
        output_json=output_json,
        output_hdf5=output_hdf5,
    )
    driver = (repo / "scripts" / "validate_sparse_nq1_phase_b.py").resolve()
    if not driver.is_file():
        raise CampaignError(f"missing Phase-B driver: {driver}")
    return {
        "schema": SCHEMA,
        "scientific_scope": (
            "authenticated nine-layer full-FC3 r0 mapping, real acoustic "
            "lead audit, and sparse scaling; explicitly no SCBA"
        ),
        "source": source,
        "environment_lock": lock,
        "driver": {
            "relative_path": "scripts/validate_sparse_nq1_phase_b.py",
            "sha256": _sha256_file(driver),
        },
        "preflight": preflight,
        "outputs": {
            "json": str(output_json),
            "hdf5": str(output_hdf5),
        },
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "prefix": str(Path(sys.prefix).resolve()),
        },
        "distributions": _versions(),
        "platform": platform.platform(),
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
    }


def _validate_resume_manifest(
    manifest: Mapping[str, Any],
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> None:
    """Reauthenticate every external identity before reusing an outcome."""

    if manifest.get("schema") != SCHEMA:
        raise CampaignError("existing Phase-B campaign schema mismatch.")
    source = _source_identity(repo, expected_commit)
    lock = _validate_lock(environment_lock.resolve())
    if manifest.get("source") != source or manifest.get("environment_lock") != lock:
        raise CampaignError("existing Phase-B campaign provenance mismatch.")
    if manifest.get("driver", {}).get("sha256") != _sha256_file(
        repo / "scripts" / "validate_sparse_nq1_phase_b.py"
    ):
        raise CampaignError("existing Phase-B campaign driver mismatch.")
    preflight = manifest.get("preflight", {})
    if preflight.get("fixture_path") != str(fixture):
        raise CampaignError("existing Phase-B campaign fixture path mismatch.")
    if preflight.get("fc3_full", {}).get("path") != str(fc3_full):
        raise CampaignError("existing Phase-B campaign FC3 path mismatch.")
    current_fixture = phase_b.verify_phase_a_fixture(fixture)
    if preflight.get("fixture") != current_fixture:
        raise CampaignError("existing Phase-B campaign fixture identity mismatch.")
    if (
        not fc3_full.is_file()
        or fc3_full.is_symlink()
        or phase_b._sha256_file(fc3_full)
        != preflight.get("fc3_full", {}).get("raw_sha256")
    ):
        raise CampaignError("existing Phase-B campaign FC3 identity mismatch.")
    current_python = {
        "executable": str(Path(sys.executable).resolve()),
        "implementation": platform.python_implementation(),
        "version": platform.python_version(),
        "prefix": str(Path(sys.prefix).resolve()),
    }
    if manifest.get("python") != current_python or manifest.get(
        "distributions"
    ) != _versions():
        raise CampaignError("existing Phase-B campaign environment mismatch.")
    expected_outputs = {
        "json": str(artifacts / "outcomes" / "phase-b-mapping.json"),
        "hdf5": str(artifacts / "outcomes" / "phase-b-mapped-fc3.h5"),
    }
    if manifest.get("outputs") != expected_outputs:
        raise CampaignError("existing Phase-B campaign output path mismatch.")


def _load_outcome(
    *, output_json: Path, output_hdf5: Path, manifest_sha256: str
) -> dict[str, Any]:
    if output_json.exists() != output_hdf5.exists():
        raise CampaignError(
            "partial Phase-B outcome pair exists; refusing an automatic heavy rerun."
        )
    if not output_json.is_file() or output_json.is_symlink():
        raise CampaignError("Phase-B JSON outcome is not a regular file.")
    try:
        value = json.loads(output_json.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid Phase-B outcome JSON: {error}") from error
    if output_json.read_bytes() != _canonical_json(value):
        raise CampaignError("Phase-B outcome JSON is not canonical.")
    if value.get("schema") != phase_b.SCHEMA:
        raise CampaignError("Phase-B outcome schema mismatch.")
    if value.get("campaign_manifest_sha256") != manifest_sha256:
        raise CampaignError("Phase-B outcome is not bound to this campaign.")
    archive = phase_b.verify_phase_b_hdf5(output_hdf5)
    if value.get("hdf5", {}).get("sha256") != archive["sha256"]:
        raise CampaignError("Phase-B JSON/HDF5 digest mismatch.")
    if bool(value.get("phase_b_certified")) != bool(archive["phase_b_certified"]):
        raise CampaignError("Phase-B JSON/HDF5 certification mismatch.")
    if value.get("release_certified") or value.get("scba_executed"):
        raise CampaignError("Phase-B outcome overclaims its scientific scope.")
    return value


def _run_science(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    output_json: Path,
    output_hdf5: Path,
    manifest_sha256: str,
    log_path: Path,
) -> dict[str, Any]:
    command = (
        sys.executable,
        str(repo / "scripts" / "validate_sparse_nq1_phase_b.py"),
        "--fixture",
        str(fixture),
        "--fc3-full",
        str(fc3_full),
        "--output-json",
        str(output_json),
        "--output-hdf5",
        str(output_hdf5),
        "--campaign-manifest-sha256",
        manifest_sha256,
    )
    environment = os.environ.copy()
    environment.update(THREAD_LIMIT_ENV)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str((repo / "src").resolve())
    completed = subprocess.run(
        command,
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    _publish_once(
        log_path,
        _canonical_json(
            {
                "command": list(command),
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        ),
    )
    if completed.returncode not in {0, 2}:
        if output_json.exists() or output_hdf5.exists():
            raise CampaignError(
                "Phase-B infrastructure failure left partial output; it is preserved "
                "and no automatic heavy rerun is allowed."
            )
        raise CampaignError(
            f"Phase-B infrastructure failed before an outcome (code {completed.returncode})."
        )
    return _load_outcome(
        output_json=output_json,
        output_hdf5=output_hdf5,
        manifest_sha256=manifest_sha256,
    )


def run_campaign(
    *,
    repo: Path,
    fixture: Path,
    fc3_full: Path,
    artifacts: Path,
    expected_commit: str,
    environment_lock: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    fixture = fixture.resolve()
    fc3_full = fc3_full.resolve()
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    output_json = artifacts / "outcomes" / "phase-b-mapping.json"
    output_hdf5 = artifacts / "outcomes" / "phase-b-mapped-fc3.h5"
    manifest_path = artifacts / "campaign-manifest.json"

    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CampaignError(f"invalid existing campaign manifest: {error}") from error
        if manifest_path.read_bytes() != _canonical_json(manifest):
            raise CampaignError("existing campaign manifest is not canonical.")
        _validate_resume_manifest(
            manifest,
            repo=repo,
            fixture=fixture,
            fc3_full=fc3_full,
            artifacts=artifacts,
            expected_commit=expected_commit,
            environment_lock=environment_lock,
        )
    else:
        manifest = build_manifest(
            repo=repo,
            fixture=fixture,
            fc3_full=fc3_full,
            artifacts=artifacts,
            expected_commit=expected_commit,
            environment_lock=environment_lock,
        )
        _publish_once(manifest_path, _canonical_json(manifest))
    manifest_sha256 = _sha256_file(manifest_path)
    _atomic_replace(
        artifacts / "status.json",
        _canonical_json(
            {
                "schema": "negfpy.sparse-nq1-phase-b-status.v1",
                "campaign_manifest_sha256": manifest_sha256,
                "state": "running",
                "release_certified": False,
                "scba_executed": False,
            }
        ),
    )
    if output_json.exists() or output_hdf5.exists():
        outcome = _load_outcome(
            output_json=output_json,
            output_hdf5=output_hdf5,
            manifest_sha256=manifest_sha256,
        )
    else:
        outcome = _run_science(
            repo=repo,
            fixture=fixture,
            fc3_full=fc3_full,
            output_json=output_json,
            output_hdf5=output_hdf5,
            manifest_sha256=manifest_sha256,
            log_path=artifacts / "logs" / "phase-b-mapping.json",
        )
    certified = bool(outcome["phase_b_certified"])
    report = {
        "schema": "negfpy.sparse-nq1-phase-b-report.v1",
        "campaign_manifest_sha256": manifest_sha256,
        "outcome_json_sha256": _sha256_file(output_json),
        "mapped_fc3_hdf5_sha256": _sha256_file(output_hdf5),
        "phase_b_certified": certified,
        "scba_executed": False,
        "release_certified": False,
        "next_gate": _next_gate(certified),
    }
    report_path = artifacts / "phase-b-report.json"
    _publish_once(report_path, _canonical_json(report))
    log_path = artifacts / "logs" / "phase-b-mapping.json"
    if not log_path.is_file() or log_path.is_symlink():
        raise CampaignError("completed Phase-B outcome is missing its immutable log.")
    checksummed = (manifest_path, output_json, output_hdf5, report_path, log_path)
    sums = "".join(
        f"{_sha256_file(path)}  {path.relative_to(artifacts).as_posix()}\n"
        for path in sorted(checksummed, key=lambda item: item.relative_to(artifacts).as_posix())
    ).encode("ascii")
    _publish_once(artifacts / "SHA256SUMS", sums)
    terminal = {
        "schema": "negfpy.sparse-nq1-phase-b-status.v1",
        "campaign_manifest_sha256": manifest_sha256,
        "state": "passed" if certified else "scientific-failure",
        "phase_b_certified": certified,
        "release_certified": False,
        "scba_executed": False,
        "report_sha256": _sha256_file(report_path),
        "sha256sums_sha256": _sha256_bytes(sums),
    }
    _atomic_replace(artifacts / "status.json", _canonical_json(terminal))
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--fc3-full", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--environment-lock", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    try:
        if arguments.preflight_only:
            result = build_manifest(
                repo=arguments.repo_root,
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                artifacts=arguments.artifacts,
                expected_commit=arguments.expected_commit,
                environment_lock=arguments.environment_lock,
            )
        else:
            result = run_campaign(
                repo=arguments.repo_root,
                fixture=arguments.fixture,
                fc3_full=arguments.fc3_full,
                artifacts=arguments.artifacts,
                expected_commit=arguments.expected_commit,
                environment_lock=arguments.environment_lock,
            )
    except (CampaignError, phase_b.PhaseBError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    print(_canonical_json(result).decode("ascii"), end="")
    passed = bool(
        result.get("phase_b_certified", result.get("preflight", {}).get("passed"))
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
