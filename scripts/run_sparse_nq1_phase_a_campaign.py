#!/usr/bin/env python3
"""Run the bounded silicon Phase-A gate with immutable per-case outcomes.

This is deliberately a serial orchestration layer.  Each scientific case is
small, deterministic, and independently restartable; MPI would add no useful
work sharing at this stage.  A completed outcome is terminal, including a
scientific failure.  Only an interrupted case without a valid outcome may be
started again with the identical manifest.
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

from negfpy.anharmonic.phase_a_fixture import verify_phase_a_fixture


SCHEMA = "negfpy.sparse-nq1-phase-a-campaign.v1"
CASES = ("cosets", "compact", "full", "full-compact", "omega0")
THREAD_LIMIT_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


class CampaignError(RuntimeError):
    """A fail-closed orchestration or provenance error."""


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
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _publish_once(path: Path, content: bytes) -> None:
    """Publish immutable bytes, or verify an identical existing object."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != content:
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
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != expected_commit:
        raise CampaignError(
            f"source commit mismatch: {commit} != {expected_commit}"
        )
    dirty = _git(repo, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise CampaignError("remote scientific worktree is not clean")
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


def _distribution_versions() -> dict[str, str]:
    names = ("h5py", "numpy", "scipy", "spglib", "threadpoolctl")
    try:
        return {name: importlib.metadata.version(name) for name in names}
    except importlib.metadata.PackageNotFoundError as error:
        raise CampaignError(f"missing locked distribution: {error.name}") from error


def build_manifest(
    *, repo: Path, fixture: Path, expected_commit: str
) -> dict[str, Any]:
    """Build the stable identity used by both fresh and resumed campaigns."""

    expected = expected_commit.strip().lower()
    if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
        raise CampaignError("--expected-commit must be a full lowercase SHA-1")
    source = _source_identity(repo, expected)
    fixture_report = verify_phase_a_fixture(fixture)
    driver = (repo / "scripts" / "validate_sparse_nq1_phase_a.py").resolve()
    if not driver.is_file():
        raise CampaignError(f"missing Phase-A case driver: {driver}")
    return {
        "schema": SCHEMA,
        "phase": "harmonic_mapping_only",
        "scientific_scope": (
            "Silicon FC2 layered mapping, full/compact equivalence, exact "
            "cosets, and solver-scale invariance; not anharmonic SCBA"
        ),
        "source": source,
        "fixture": fixture_report,
        "driver": {
            "relative_path": "scripts/validate_sparse_nq1_phase_a.py",
            "sha256": _sha256_file(driver),
        },
        "cases": list(CASES),
        "python": {
            "executable": str(Path(sys.executable).resolve()),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "platform": platform.platform(),
        "distributions": _distribution_versions(),
        "thread_limits": dict(sorted(THREAD_LIMIT_ENV.items())),
    }


def _load_outcome(path: Path, *, case: str, manifest_sha256: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CampaignError(f"invalid completed outcome {path}: {error}") from error
    if (
        value.get("schema") != "negfpy.sparse-nq1-phase-a-case.v1"
        or value.get("case") != case
    ):
        raise CampaignError(f"completed outcome identity mismatch: {path}")
    binding = value.get("campaign_manifest_sha256")
    if binding != manifest_sha256:
        raise CampaignError(
            f"completed outcome is not bound to this campaign: {path}"
        )
    return value


def _run_case(
    *,
    repo: Path,
    fixture: Path,
    outcome: Path,
    case: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    temporary_outcome = outcome.with_name(f".{outcome.name}.worker-{os.getpid()}")
    temporary_outcome.unlink(missing_ok=True)
    command = (
        sys.executable,
        str(repo / "scripts" / "validate_sparse_nq1_phase_a.py"),
        "--fixture",
        str(fixture),
        "--case",
        case,
        "--output",
        str(temporary_outcome),
    )
    environment = os.environ.copy()
    environment.update(THREAD_LIMIT_ENV)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str((repo / "src").resolve())
    completed = subprocess.run(
        command,
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    log = {
        "case": case,
        "command": list(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    _atomic_replace(
        outcome.parent.parent / "logs" / f"{case}.json", _canonical_json(log)
    )
    if not temporary_outcome.is_file():
        raise CampaignError(
            f"case {case!r} ended without an atomic scientific outcome "
            f"(return code {completed.returncode})"
        )
    try:
        value = json.loads(temporary_outcome.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CampaignError(f"case {case!r} produced invalid JSON") from error
    value["campaign_manifest_sha256"] = manifest_sha256
    content = _canonical_json(value)
    _publish_once(outcome, content)
    temporary_outcome.unlink(missing_ok=True)
    if completed.returncode not in {0, 2}:
        raise CampaignError(
            f"case {case!r} returned infrastructure code {completed.returncode}"
        )
    return _load_outcome(outcome, case=case, manifest_sha256=manifest_sha256)


def _sha256sums(artifacts: Path, paths: list[Path]) -> bytes:
    lines = []
    for path in sorted(paths, key=lambda item: str(item.relative_to(artifacts))):
        relative = path.relative_to(artifacts)
        lines.append(f"{_sha256_file(path)}  {relative.as_posix()}")
    return ("\n".join(lines) + "\n").encode("ascii")


def run_campaign(
    *, repo: Path, fixture: Path, artifacts: Path, expected_commit: str
) -> dict[str, Any]:
    repo = repo.resolve()
    fixture = fixture.resolve()
    artifacts = artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        repo=repo, fixture=fixture, expected_commit=expected_commit
    )
    manifest_content = _canonical_json(manifest)
    manifest_sha256 = _sha256_bytes(manifest_content)
    manifest_path = artifacts / "campaign-manifest.json"
    _publish_once(manifest_path, manifest_content)
    outcomes: list[dict[str, Any]] = []
    outcome_paths: list[Path] = []
    for index, case in enumerate(CASES):
        status = {
            "schema": "negfpy.sparse-nq1-phase-a-status.v1",
            "campaign_manifest_sha256": manifest_sha256,
            "state": "running",
            "current_case": case,
            "completed_cases": [item["case"] for item in outcomes],
            "case_index": index,
            "case_count": len(CASES),
        }
        _atomic_replace(artifacts / "status.json", _canonical_json(status))
        outcome = artifacts / "outcomes" / f"{index:02d}-{case}.json"
        if outcome.exists():
            value = _load_outcome(
                outcome, case=case, manifest_sha256=manifest_sha256
            )
        else:
            value = _run_case(
                repo=repo,
                fixture=fixture,
                outcome=outcome,
                case=case,
                manifest_sha256=manifest_sha256,
            )
        outcomes.append(value)
        outcome_paths.append(outcome)
    certified = all(bool(item.get("certification_passed")) for item in outcomes)
    report = {
        "schema": "negfpy.sparse-nq1-phase-a-report.v1",
        "campaign_manifest_sha256": manifest_sha256,
        "scientific_scope": manifest["scientific_scope"],
        "case_order": list(CASES),
        "outcomes": [
            {
                "case": item["case"],
                "sha256": _sha256_file(path),
                "certification_passed": bool(item.get("certification_passed")),
            }
            for item, path in zip(outcomes, outcome_paths, strict=True)
        ],
        "release_certified": False,
        "phase_a_certified": certified,
        "not_certified": [
            "anharmonic FC3 device mapping",
            "sparse SCBA",
            "acoustic contacts",
            "N_q=1 material-device current",
        ],
    }
    report_content = _canonical_json(report)
    report_path = artifacts / "phase-a-report.json"
    _publish_once(report_path, report_content)
    checksummed = [manifest_path, *outcome_paths, report_path]
    sums = _sha256sums(artifacts, checksummed)
    _publish_once(artifacts / "SHA256SUMS", sums)
    terminal = {
        "schema": "negfpy.sparse-nq1-phase-a-status.v1",
        "campaign_manifest_sha256": manifest_sha256,
        "state": "passed" if certified else "scientific-failure",
        "current_case": None,
        "completed_cases": list(CASES),
        "case_index": len(CASES),
        "case_count": len(CASES),
        "phase_a_certified": certified,
        "release_certified": False,
        "report_sha256": _sha256_bytes(report_content),
        "sha256sums_sha256": _sha256_bytes(sums),
    }
    _atomic_replace(artifacts / "status.json", _canonical_json(terminal))
    return terminal


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.preflight_only:
        manifest = build_manifest(
            repo=arguments.repo_root.resolve(),
            fixture=arguments.fixture.resolve(),
            expected_commit=arguments.expected_commit,
        )
        print(_canonical_json(manifest).decode("utf-8"), end="")
        return 0
    try:
        result = run_campaign(
            repo=arguments.repo_root,
            fixture=arguments.fixture,
            artifacts=arguments.artifacts,
            expected_commit=arguments.expected_commit,
        )
    except CampaignError as error:
        print(f"error: {error}", file=sys.stderr)
        return 3
    print(_canonical_json(result).decode("utf-8"), end="")
    return 0 if result["phase_a_certified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
