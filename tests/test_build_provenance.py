from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

import _negfpy_build as build_support
from _negfpy_build import build_provenance_record
import negfpy.anharmonic._provenance as provenance


_HASH_DOMAIN = b"negfpy.package-payload.v1\0"


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
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"negfpy\"\nversion = \"0.1.0rc1\"\n",
        encoding="ascii",
    )
    source = root / "src" / "negfpy"
    (source / "anharmonic").mkdir(parents=True)
    (source / "__init__.py").write_bytes(b"VALUE = 1\n")
    (source / "anharmonic" / "kernel.py").write_bytes(b"VALUE = 2\n")
    (source / "anharmonic" / "_provenance.py").write_bytes(
        b"SOURCE = True\n"
    )
    _run_git(root, "init", "--quiet")
    _run_git(root, "config", "user.name", "NEGFPy test")
    _run_git(root, "config", "user.email", "negfpy-test@example.invalid")
    _run_git(
        root, "add", ".gitignore", "tracked.txt", "pyproject.toml", "src"
    )
    _run_git(root, "commit", "--quiet", "-m", "fixture")
    return _run_git(root, "rev-parse", "HEAD")


def _package(root: Path) -> Path:
    package = root / "build" / "lib" / "negfpy"
    source = root / "src" / "negfpy"
    if source.is_dir():
        shutil.copytree(source, package)
    else:
        package.mkdir(parents=True)
        (package / "__init__.py").write_bytes(b"VALUE = 1\n")
        nested = package / "anharmonic"
        nested.mkdir()
        (nested / "kernel.py").write_bytes(b"VALUE = 2\n")
    return package


def _hash_items(items: tuple[tuple[str, bytes], ...]) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_DOMAIN)
    digest.update(len(items).to_bytes(8, "big"))
    for relative, content in items:
        encoded = relative.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _source_hash(package: Path) -> str:
    items = tuple(
        (path.relative_to(package).as_posix(), path.read_bytes())
        for path in sorted(
            (item for item in package.rglob("*") if item.is_file()),
            key=lambda item: item.relative_to(package).as_posix(),
        )
        if path.name != "_build_provenance.json"
        and "__pycache__" not in path.relative_to(package).parts
        and path.suffix not in {".pyc", ".pyo"}
    )
    return _hash_items(items)


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
        {"git_dirty": "unknown", "git_revision": "a" * 40},
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


def _clone_clean_repository(repository: Path, destination: Path) -> None:
    subprocess.run(
        ("git", "clone", "--quiet", "--no-local", str(repository), str(destination)),
        check=True,
        capture_output=True,
        text=True,
    )


def _build_wheel(source: Path, destination: Path, *, epoch: str) -> Path:
    destination.mkdir()
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = epoch
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-build-isolation",
            "--no-cache-dir",
            "--no-deps",
            "--wheel-dir",
            str(destination),
            str(source),
        ),
        cwd=destination.parent,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    wheels = tuple(destination.glob("negfpy-*.whl"))
    assert len(wheels) == 1
    return wheels[0]


def _wheel_build_record(wheel: Path) -> dict[str, object]:
    with zipfile.ZipFile(wheel) as archive:
        return json.loads(
            archive.read("negfpy/_build_provenance.json").decode("ascii")
        )


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
    epoch = _run_git(repository, "show", "-s", "--format=%ct", "HEAD")
    source = tmp_path / "source"
    _clone_clean_repository(repository, source)
    stale = source / "build" / "lib" / "negfpy"
    stale.mkdir(parents=True)
    (stale / "stale_ignored.py").write_text(
        "STALE = True\n", encoding="ascii"
    )

    first = _build_wheel(source, tmp_path / "wheel-first", epoch=epoch)
    second = _build_wheel(source, tmp_path / "wheel-second", epoch=epoch)
    assert first.read_bytes() == second.read_bytes()
    assert _run_git(source, "status", "--porcelain=v1", "--untracked-files=all") == ""

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
        metadata_name = "negfpy/_build_provenance.json"
        assert names.count(metadata_name) == 1
        assert "negfpy/stale_ignored.py" not in names
        record_name = next(
            name for name in names if name.endswith(".dist-info/RECORD")
        )
        wheel_record = archive.read(record_name).decode("utf-8")
        assert any(
            line.startswith(metadata_name + ",sha256=")
            for line in wheel_record.splitlines()
        )
        document = json.loads(archive.read(metadata_name).decode("ascii"))
        payload = tuple(
            (name.removeprefix("negfpy/"), archive.read(name))
            for name in sorted(names)
            if name.startswith("negfpy/")
            and not name.endswith("/")
            and name != metadata_name
            and "__pycache__" not in name.split("/")
            and not name.endswith((".pyc", ".pyo"))
        )
        assert payload
        payload_hash = _hash_items(payload)

    assert document["git_revision"] == revision
    assert document["git_dirty"] is False
    assert document["package_source_sha256"] == payload_hash

    install_target = tmp_path / "installed"
    subprocess.run(
        (
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-compile",
            "--target",
            str(install_target),
            str(first),
        ),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    installed = json.loads(
        subprocess.run(
            (
                sys.executable,
                "-c",
                (
                    "import json,sys; "
                    f"sys.path.insert(0, {str(install_target)!r}); "
                    "from negfpy.anharmonic._provenance import code_provenance; "
                    "print(json.dumps(code_provenance(), sort_keys=True))"
                ),
            ),
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    assert installed["git_revision"] == revision
    assert installed["git_dirty"] is False
    assert installed["source_tree_sha256"] == payload_hash


def test_pep517_dirty_and_no_git_builds_are_noncertifying(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).resolve().parents[1]
    if not (repository / ".git").exists():
        pytest.skip("wheel integration test requires a Git checkout")
    if _run_git(repository, "status", "--porcelain=v1", "--untracked-files=all"):
        pytest.skip("wheel integration test requires a clean checkout")
    epoch = _run_git(repository, "show", "-s", "--format=%ct", "HEAD")

    dirty_source = tmp_path / "dirty-source"
    _clone_clean_repository(repository, dirty_source)
    dirty_file = dirty_source / "src" / "negfpy" / "anharmonic" / "_provenance.py"
    dirty_file.write_bytes(dirty_file.read_bytes() + b"\n# dirty build fixture\n")
    dirty_wheel = _build_wheel(
        dirty_source, tmp_path / "dirty-wheel", epoch=epoch
    )
    dirty = _wheel_build_record(dirty_wheel)
    assert len(str(dirty["git_revision"])) == 40
    assert dirty["git_dirty"] is True

    sdist_source = tmp_path / "sdist-source"
    _clone_clean_repository(repository, sdist_source)
    sdist_dir = tmp_path / "sdist"
    sdist_dir.mkdir()
    environment = dict(os.environ)
    environment["SOURCE_DATE_EPOCH"] = epoch
    subprocess.run(
        (
            sys.executable,
            "setup.py",
            "--quiet",
            "sdist",
            "--dist-dir",
            str(sdist_dir),
        ),
        cwd=sdist_source,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    sdists = tuple(sdist_dir.glob("negfpy-*.tar.gz"))
    assert len(sdists) == 1
    archive_wheel = _build_wheel(
        sdists[0], tmp_path / "archive-wheel", epoch=epoch
    )
    unknown = _wheel_build_record(archive_wheel)
    assert unknown["git_revision"] == "unknown"
    assert unknown["git_dirty"] == "unknown"


def test_present_embedded_identity_precedes_ancestor_project_and_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def embedded(*args: object, **kwargs: object) -> tuple[bool, tuple[str, bool]]:
        return True, ("c" * 40, False)

    def ancestor_scan_forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("wheel identity consulted an ancestor project")

    monkeypatch.setattr(provenance, "_source_repository", lambda root: None)
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


def test_framed_payload_hash_separates_ambiguous_path_byte_streams(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.py").write_bytes(b"b.pyX")
    (second / "a.pyb.py").write_bytes(b"X")
    old_first = b"a.py" + b"b.pyX"
    old_second = b"a.pyb.py" + b"X"
    assert old_first == old_second
    assert _source_hash(first) != _source_hash(second)
    assert build_support._package_source_sha256(first) == _source_hash(first)
    assert provenance._package_source_sha256(second) == _source_hash(second)


def test_runtime_hash_binds_non_python_payload_and_rejects_symlink(
    tmp_path: Path,
) -> None:
    package = _package(tmp_path)
    before = provenance._package_source_sha256(package)
    assert before is not None
    (package / "native-extension.so").write_bytes(b"native bytes")
    after = provenance._package_source_sha256(package)
    assert after is not None and after != before
    outside = tmp_path / "outside.dat"
    outside.write_bytes(b"outside")
    (package / "linked.dat").symlink_to(outside)
    assert provenance._package_source_sha256(package) is None


def test_fresh_build_root_removes_ignored_stale_payload(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    build_lib = project / "build" / "lib"
    target = build_lib / "negfpy"
    target.mkdir(parents=True)
    (target / "stale_ignored.py").write_text("STALE = True\n", encoding="ascii")
    assert build_support._fresh_package_root(
        build_lib, project_root=project
    ) == target
    assert not target.exists()

    outside = tmp_path / "outside"
    outside.mkdir()
    target.symlink_to(outside, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlinked build package target"):
        build_support._fresh_package_root(build_lib, project_root=project)


def test_fresh_build_root_cannot_overlap_canonical_source(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    source = project / "src" / "negfpy"
    source.mkdir(parents=True)
    sentinel = source / "sentinel.py"
    sentinel.write_bytes(b"PRESERVE = True\n")

    for unsafe in (project, project / "src", source):
        with pytest.raises(RuntimeError, match="unsafe build package target"):
            build_support._fresh_package_root(unsafe, project_root=project)
        assert sentinel.read_bytes() == b"PRESERVE = True\n"


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_head_blob_comparison_defeats_hidden_index_mutation(
    tmp_path: Path,
    index_flag: str,
) -> None:
    project = tmp_path / "checkout"
    _init_repository(project)
    source_file = project / "src" / "negfpy" / "anharmonic" / "kernel.py"
    _run_git(project, "update-index", index_flag, str(source_file.relative_to(project)))
    source_file.write_bytes(b"HIDDEN = True\n")
    package = _package(project)
    assert _run_git(project, "status", "--porcelain=v1") == ""

    record = build_provenance_record(
        project_root=project,
        package_root=package,
        package_version="0.1.0rc1",
    )
    assert len(str(record["git_revision"])) == 40
    assert record["git_dirty"] is True


def test_absent_record_under_ancestor_git_remains_unknown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkout"
    _init_repository(project)
    package = (
        project / ".venv" / "lib" / "python" / "site-packages" / "negfpy"
    )
    (package / "anharmonic").mkdir(parents=True)
    fake_module = package / "anharmonic" / "_provenance.py"
    fake_module.write_bytes(b"INSTALLED = True\n")
    (package / "__init__.py").write_bytes(b"INSTALLED = True\n")
    monkeypatch.setattr(provenance, "__file__", str(fake_module))

    assert provenance._source_repository(package) is None
    result = provenance.code_provenance()
    assert result["git_revision"] == "unknown"
    assert result["git_dirty"] == "unknown"


def test_exact_source_layout_ignores_forged_embedded_record(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkout"
    revision = _init_repository(project)
    package = project / "src" / "negfpy"
    fake_module = package / "anharmonic" / "_provenance.py"
    source_hash = provenance._package_source_sha256(package)
    assert source_hash is not None
    _write_record(package, _record(source_hash, git_revision="f" * 40))
    monkeypatch.setattr(provenance, "__file__", str(fake_module))

    result = provenance.code_provenance()
    assert result["git_revision"] == revision
    assert result["git_revision"] != "f" * 40
    assert result["git_dirty"] is True


@pytest.mark.parametrize("index_flag", ("--assume-unchanged", "--skip-worktree"))
def test_live_source_head_comparison_defeats_hidden_index_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    index_flag: str,
) -> None:
    project = tmp_path / "checkout"
    revision = _init_repository(project)
    package = project / "src" / "negfpy"
    source_file = package / "anharmonic" / "kernel.py"
    _run_git(project, "update-index", index_flag, str(source_file.relative_to(project)))
    source_file.write_bytes(b"HIDDEN_RUNTIME = True\n")
    assert _run_git(project, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(
        provenance,
        "__file__",
        str(package / "anharmonic" / "_provenance.py"),
    )

    result = provenance.code_provenance()
    assert result["git_revision"] == revision
    assert result["git_dirty"] is True


def test_live_source_head_comparison_detects_ignored_extra_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project = tmp_path / "checkout"
    revision = _init_repository(project)
    package = project / "src" / "negfpy"
    exclude = project / ".git" / "info" / "exclude"
    exclude.write_text("src/negfpy/ignored-native.so\n", encoding="ascii")
    (package / "ignored-native.so").write_bytes(b"ignored executable payload")
    assert _run_git(project, "status", "--porcelain=v1") == ""
    monkeypatch.setattr(
        provenance,
        "__file__",
        str(package / "anharmonic" / "_provenance.py"),
    )

    result = provenance.code_provenance()
    assert result["git_revision"] == revision
    assert result["git_dirty"] is True
