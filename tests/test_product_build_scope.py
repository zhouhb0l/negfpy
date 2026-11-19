from pathlib import Path

import pytest

from _negfpy_build import _build_root_marker_path, _fresh_package_root, _fresh_product_package_roots


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    for name in ("negfpp", "negfpy"):
        source = project / "src" / name
        source.mkdir(parents=True)
        (source / "__init__.py").write_text("PRESERVE = True\n")
    return project


def test_fresh_product_build_removes_stale_files_in_both_owned_namespaces(tmp_path):
    project = _project(tmp_path)
    build = project / "build" / "lib"
    roots = _fresh_product_package_roots(build, project_root=project)
    for root in roots.values():
        root.mkdir()
        (root / "stale.py").write_text("STALE = True\n")
    refreshed = _fresh_product_package_roots(build, project_root=project)
    assert set(refreshed) == {"negfpp", "negfpy"}
    assert all(not root.exists() for root in refreshed.values())
    assert all((project / "src" / name / "__init__.py").read_text() == "PRESERVE = True\n"
               for name in refreshed)


@pytest.mark.parametrize("name", ["negfpp", "negfpy"])
def test_existing_unowned_namespace_prevents_claim_and_preserves_files(tmp_path, name):
    project = _project(tmp_path)
    build = project / "build" / "lib"
    target = build / name
    target.mkdir(parents=True)
    sentinel = target / "preserve.txt"
    sentinel.write_text("existing unowned data")
    with pytest.raises(RuntimeError, match="unowned"):
        _fresh_product_package_roots(build, project_root=project)
    assert sentinel.read_text() == "existing unowned data"
    assert not _build_root_marker_path(build).exists()


@pytest.mark.parametrize("name", ["negfpp", "negfpy"])
def test_build_target_cannot_overlap_either_source_namespace(tmp_path, name):
    project = _project(tmp_path)
    for source_name in ("negfpp", "negfpy"):
        with pytest.raises(RuntimeError, match="unsafe build package target"):
            _fresh_package_root(project / "src" / source_name,
                                project_root=project, package_name=name)
    assert (project / "src" / name / "__init__.py").read_text() == "PRESERVE = True\n"
