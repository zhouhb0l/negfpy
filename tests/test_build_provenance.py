from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from _negfpy_build import build_provenance_record
import negfpy.anharmonic._provenance as provenance


def _run_git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ("git", *arguments),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_repository(root: Path) -> str:
    root.mkdir()
    (root / ".gitignore").write_text("build/\n", encoding="ascii")
    (root / "tracked.txt").write_text("exact source\n", encoding="ascii")
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.name", "NEGFPy test")
    _run_git(root, "config", "user.email", "negfpy-test@example.invalid")
    _run_git(root, "add", ".gitignore", "tracked.txt")
    _run_git(root, "commit", "--quiet", "-m", "fixture")
    return _run_git(root, "rev-parse", "HEAD")


def _package(root: Path) -> Path:
    package = root / "build" / "lib" / "negfpy"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes(b"VALUE = 1\n")
    nested = package / "anharmonic"
    nested.mkdir()
    (nested / "kernel.py").write_bytes(b"VALUE = 2\n")
    return package


def _source_hash(package: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _record(source_hash: str, **updates: object) -> dict[str, object]:
    record: dict[str, object] = {
        "git_dirty": False,
        "git_revision": "a" * 40,
        "package_source_sha256": source_hash,
        "package_version": "0.1.0rc1",
        "project": "negfpy",
        "schema": "negfpy.build_provenance.v1",
    }
    record.update(updates)
    return record


def _write_record(package: Path, record: dict[str, object]) -> None:
    (package / "_build_provenance.json").write_text(
        json.dumps(record, sort_keys=True) + "\n",
        encoding="ascii",
    )


def test_build_record_uses_exact_clean_repository_and_copied_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkout"
    revision = _init_repository(project)
    package = _package(project)

    record = build_provenance_record(
        project_root=project,
        package_root=package,
        package_version="0.1.0rc1",
    )

    assert record == {
        "git_dirty": False,
        "git_revision": revision,
        "package_source_sha256": _source_hash(package),
        "package_version": "0.1.0rc1",
        "project": "negfpy",
        "schema": "negfpy.build_provenance.v1",
    }


def test_dirty_and_unknown_build_records_are_explicitly_nonclean(
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkout"
    revision = _init_repository(project)
    package = _package(project)
    (project / "tracked.txt").write_text("dirty source\n", encoding="ascii")

    dirty = build_provenance_record(
        project_root=project,
        package_root=package,
        package_version="0.1.0rc1",
    )
    assert dirty["git_revision"] == revision
    assert dirty["git_dirty"] is True

    outside_git = tmp_path / "source-archive"
    outside_git.mkdir()
    unknown = build_provenance_record(
        project_root=outside_git,
        package_root=package,
        package_version="0.1.0rc1",
    )
    assert unknown["git_revision"] == "unknown"
    assert unknown["git_dirty"] == "unknown"


def test_embedded_build_record_requires_exact_installed_python_bytes(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    source_hash = provenance._package_source_sha256(package)
    assert source_hash == _source_hash(package)
    _write_record(package, _record(source_hash))

    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    ) == ("a" * 40, False)

    (package / "anharmonic" / "kernel.py").write_bytes(b"FORGED = True\n")
    changed_hash = provenance._package_source_sha256(package)
    assert changed_hash != source_hash
    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=changed_hash,
        installed_version="0.1.0rc1",
    ) is None


@pytest.mark.parametrize(
    "update",
    (
        {"package_source_sha256": "0" * 64},
        {"package_version": "9.9"},
        {"git_revision": "short"},
        {"git_dirty": 1},
        {"schema": "forged"},
        {"extra": "not allowed"},
    ),
)
def test_embedded_build_record_rejects_forged_fields(
    tmp_path: Path,
    update: dict[str, object],
) -> None:
    package = _package(tmp_path)
    source_hash = _source_hash(package)
    _write_record(package, _record(source_hash, **update))

    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    ) is None


def test_embedded_build_record_rejects_duplicate_and_nonfinite_json(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    source_hash = _source_hash(package)
    path = package / "_build_provenance.json"
    valid = json.dumps(_record(source_hash), sort_keys=True)

    path.write_text(
        valid[:-1] + ',"git_dirty":false}',
        encoding="ascii",
    )
    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    ) is None

    path.write_text(
        valid.replace('"git_dirty": false', '"git_dirty": NaN'),
        encoding="ascii",
    )
    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    ) is None


def test_embedded_build_record_rejects_symlink(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    source_hash = _source_hash(package)
    outside = tmp_path / "record.json"
    outside.write_text(json.dumps(_record(source_hash)), encoding="ascii")
    (package / "_build_provenance.json").symlink_to(outside)

    assert provenance._validated_embedded_build_provenance(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    ) is None


def test_source_checkout_prefers_live_git_without_generated_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_loaded(*args: object, **kwargs: object) -> None:
        raise AssertionError("source checkout consulted wheel metadata")

    monkeypatch.setattr(
        provenance,
        "_validated_embedded_build_provenance",
        fail_if_loaded,
    )
    result = provenance.code_provenance()
    assert result["git_revision"] == _run_git(
        Path(__file__).resolve().parents[1], "rev-parse", "HEAD"
    )
    assert type(result["git_dirty"]) is bool


def test_present_invalid_record_blocks_ancestor_git_fallback(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    source_hash = _source_hash(package)
    (package / "_build_provenance.json").write_text(
        "{}\n",
        encoding="ascii",
    )

    present, record = provenance._embedded_build_provenance_if_present(
        package,
        package_source_sha256=source_hash,
        installed_version="0.1.0rc1",
    )
    assert present
    assert record is None


def test_pep517_clean_wheel_contains_bound_record_and_record_entry(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    if not (repository / ".git").exists():
        pytest.skip("clean-wheel integration test requires a Git checkout")
    status = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if status:
        pytest.skip("clean-wheel integration test requires a clean checkout")
    revision = _run_git(repository, "rev-parse", "HEAD")
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()

    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-deps",
            "--wheel-dir",
            str(wheel_dir),
            str(repository),
        ),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(wheel_dir.glob("negfpy-*.whl"))
    assert len(wheels) == 1

    with zipfile.ZipFile(wheels[0]) as archive:
        names = archive.namelist()
        metadata_name = "negfpy/_build_provenance.json"
        assert names.count(metadata_name) == 1
        record_name = next(
            name for name in names if name.endswith(".dist-info/RECORD")
        )
        wheel_record = archive.read(record_name).decode("utf-8")
        assert any(
            line.startswith(metadata_name + ",sha256=")
            for line in wheel_record.splitlines()
        )
        document = json.loads(archive.read(metadata_name).decode("ascii"))
        digest = hashlib.sha256()
        sources = sorted(
            name
            for name in names
            if name.startswith("negfpy/") and name.endswith(".py")
        )
        assert sources
        for name in sources:
            digest.update(name.removeprefix("negfpy/").encode("utf-8"))
            digest.update(archive.read(name))

    assert document["git_revision"] == revision
    assert document["git_dirty"] is False
    assert document["package_source_sha256"] == digest.hexdigest()


def test_present_embedded_identity_precedes_ancestor_project_and_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def embedded(*args: object, **kwargs: object) -> tuple[bool, tuple[str, bool]]:
        return True, ("c" * 40, False)

    def ancestor_scan_forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("wheel identity consulted an ancestor project")

    monkeypatch.setattr(
        provenance,
        "_embedded_build_provenance_if_present",
        embedded,
    )
    monkeypatch.setattr(
        provenance,
        "_declared_package_version",
        ancestor_scan_forbidden,
    )
    result = provenance.code_provenance()
    assert result["git_revision"] == "c" * 40
    assert result["git_dirty"] is False
    assert result["declared_package_version"] == "0.1.0rc1"
