from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import sys

import pytest


SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
qualification = importlib.import_module("qualify_material_nq1_candidate")
from negfpy.anharmonic._launch_environment import build_launch_environment


def _environment(root: Path | None = None) -> dict[str, str]:
    root = Path(__file__).parents[1] if root is None else root
    return build_launch_environment(
        repo=root,
        executable=Path(sys.executable).resolve(),
        home=root,
        tmpdir=root,
        git_bin=Path("/usr/bin"),
    ) | {"NEGFPY_QUALIFICATION_RUNTIME_SHA256": "a" * 64}


def _context(root: Path) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": qualification.QUALIFICATION_CONTEXT_SCHEMA,
        "repo_root": str(root),
        "source": {
            "commit": "1" * 40,
            "tree": "2" * 40,
            "dirty": False,
            "package_file": str(root / "src/negfpy/__init__.py"),
        },
        "source_tree_sha256": "3" * 64,
        "environment_lock": {
            "path": str(root / "lock.txt"),
            "sha256": "4" * 64,
            "format": "@EXPLICIT",
        },
        "scientific_task_set_sha256": "5" * 64,
        "material_input_hashes": {"phase_a_fixture": "6" * 64},
        "runtimes": {"python310": {}, "python312": {}},
        "mpiexec": {"path": "/usr/bin/true", "sha256": "7" * 64},
        "tracked_test_files": {
            "tests/test_example.py": {
                "relative_path": "tests/test_example.py",
                "sha256": "8" * 64,
                "size": 1,
            },
        },
        "driver_files": {
            "scripts/driver.py": {
                "relative_path": "scripts/driver.py",
                "sha256": "9" * 64,
                "size": 1,
            },
        },
    }
    return payload | {
        "qualification_context_sha256": qualification._context_digest(payload)
    }


def _changed_context(
    context: dict[str, object], mutate: object,
) -> dict[str, object]:
    changed = copy.deepcopy(context)
    mutate(changed)
    changed.pop("qualification_context_sha256")
    changed["qualification_context_sha256"] = qualification._context_digest(changed)
    return changed


def test_passed_qualification_check_is_reused_without_rerun(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('stable qualification')"]
    first = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "stable.log",
        label="stable check",
        qualification_context=_context(tmp_path),
    )
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "stable.log",
        label="stable check",
        qualification_context=_context(tmp_path),
    )
    assert not first["reused"]
    assert second["reused"]
    assert second["receipt_sha256"] == first["receipt_sha256"]
    assert len(list(tmp_path.glob("stable*.log"))) == 1


def test_failed_qualification_attempts_are_append_only(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(7)"]
    for _ in range(2):
        with pytest.raises(qualification.QualificationError, match="failed"):
            qualification._run(
                command,
                cwd=tmp_path,
                environment=_environment(),
                log=tmp_path / "failure.log",
                label="failing check",
                qualification_context=_context(tmp_path),
            )
    assert (tmp_path / "failure.log").is_file()
    assert (tmp_path / "failure-attempt-002.log").is_file()
    assert len(list(tmp_path.glob("failure*.receipt.json"))) == 2


def test_passed_receipt_reruns_for_runtime_package_identity_drift(
    tmp_path: Path,
) -> None:
    command = [sys.executable, "-c", "print('runtime-bound')"]
    first_environment = _environment() | {
        "NEGFPY_QUALIFICATION_RUNTIME_SHA256": "a" * 64
    }
    first = qualification._run(
        command,
        cwd=tmp_path,
        environment=first_environment,
        log=tmp_path / "runtime.log",
        label="runtime check",
        qualification_context=_context(tmp_path),
    )
    changed_environment = _environment() | {
        "NEGFPY_QUALIFICATION_RUNTIME_SHA256": "b" * 64
    }
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=changed_environment,
        log=tmp_path / "runtime.log",
        label="runtime check",
        qualification_context=_context(tmp_path),
    )
    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert second["reused"] is False


def test_passed_receipt_reruns_for_tool_path_identity_drift(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('tool-path-bound')"]
    first_environment = _environment() | {"PATH": "/validated/tools:/usr/bin"}
    qualification._run(
        command,
        cwd=tmp_path,
        environment=first_environment,
        log=tmp_path / "tool-path.log",
        label="tool path check",
        qualification_context=_context(tmp_path),
    )
    changed_environment = _environment() | {"PATH": "/usr/bin"}
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=changed_environment,
        log=tmp_path / "tool-path.log",
        label="tool path check",
        qualification_context=_context(tmp_path),
    )
    assert second["attempt"] == 2
    assert second["reused"] is False


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPAL_PREFIX", "/different/mpi"),
        ("OMPI_MCA_pml", "ucx"),
        ("OMPI_MCA_btl", "self,tcp"),
        ("OMPI_MCA_rmaps_base_oversubscribe", "0"),
        ("HOME", "/different/home"),
        ("TMPDIR", "/different/tmp"),
        ("LD_LIBRARY_PATH", "/different/libraries"),
        ("UNEXPECTED_LAUNCH_CONTROL", "changed"),
    ],
)
def test_passed_receipt_cannot_reuse_any_changed_launch_environment(
    tmp_path: Path, name: str, value: str,
) -> None:
    command = [sys.executable, "-c", "print('complete-environment-bound')"]
    environment = _environment(tmp_path)
    first = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "environment.log", label="environment check",
        qualification_context=_context(tmp_path),
    )
    second = qualification._run(
        command, cwd=tmp_path, environment=environment | {name: value},
        log=tmp_path / "environment.log", label="environment check",
        qualification_context=_context(tmp_path),
    )
    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert second["reused"] is False
    assert second["identity_sha256"] != first["identity_sha256"]


@pytest.mark.parametrize("name", ["OPAL_PREFIX", "OMPI_MCA_btl", "HOME", "TMPDIR"])
def test_passed_receipt_cannot_reuse_dropped_launch_environment_key(
    tmp_path: Path, name: str,
) -> None:
    command = [sys.executable, "-c", "print('no-projection')"]
    environment = _environment(tmp_path)
    qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "dropped.log", label="dropped check",
        qualification_context=_context(tmp_path),
    )
    del environment[name]
    result = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "dropped.log", label="dropped check",
        qualification_context=_context(tmp_path),
    )
    assert result["attempt"] == 2
    assert result["reused"] is False


def test_executed_environment_exactly_matches_complete_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INHERITED_SECRET_FOR_TEST", "must-not-reach-child")
    environment = _environment(tmp_path)
    environment["ADDITIONAL_EXPLICIT_VALUE"] = "authenticated in full"
    command = [
        sys.executable, "-c",
        "import json, os; print(json.dumps(dict(os.environ), sort_keys=True))",
    ]
    result = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "actual.log", label="actual environment",
        qualification_context=_context(tmp_path),
    )
    receipt = json.loads((tmp_path / result["receipt_path"]).read_text())
    executed = json.loads((tmp_path / result["path"]).read_text())
    assert receipt["schema"] == "negfpy.material-nq1-qualification-check.v3"
    assert receipt["identity"]["schema"] == (
        "negfpy.material-nq1-qualification-check-identity.v3"
    )
    assert receipt["identity"]["environment"] == environment == executed
    assert "INHERITED_SECRET_FOR_TEST" not in executed


def test_legacy_projected_environment_receipt_is_not_reused(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "print('legacy receipt')"]
    environment = _environment(tmp_path)
    context = _context(tmp_path)
    result = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "legacy.log", label="legacy check",
        qualification_context=context,
    )
    receipt_path = tmp_path / result["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["schema"] = "negfpy.material-nq1-qualification-check.v2"
    receipt["identity"]["schema"] = (
        "negfpy.material-nq1-qualification-check-identity.v2"
    )
    receipt["identity"]["environment"].pop("OPAL_PREFIX")
    receipt["identity_sha256"] = hashlib.sha256(
        qualification.release._canonical(receipt["identity"])
    ).hexdigest()
    receipt_path.write_bytes(qualification.release._canonical(receipt))
    second = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "legacy.log", label="legacy check",
        qualification_context=context,
    )
    assert second["attempt"] == 2
    assert second["reused"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(
            lambda value: value["source"].update(commit="a" * 40),
            id="candidate-commit",
        ),
        pytest.param(
            lambda value: value["source"].update(tree="b" * 40),
            id="git-tree",
        ),
        pytest.param(
            lambda value: value["material_input_hashes"].update(
                phase_a_fixture="c" * 64
            ),
            id="material-input",
        ),
        pytest.param(
            lambda value: value["tracked_test_files"]["tests/test_example.py"].update(
                sha256="d" * 64
            ),
            id="test-bytes",
        ),
        pytest.param(
            lambda value: value["driver_files"]["scripts/driver.py"].update(
                sha256="e" * 64
            ),
            id="driver-bytes",
        ),
    ],
)
def test_passed_receipt_reruns_for_stale_qualification_context(
    tmp_path: Path, mutate: object,
) -> None:
    command = [sys.executable, "-c", "print('context-bound')"]
    context = _context(tmp_path)
    first = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "context.log",
        label="context check",
        qualification_context=context,
    )
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "context.log",
        label="context check",
        qualification_context=_changed_context(context, mutate),
    )
    assert first["attempt"] == 1
    assert second["attempt"] == 2
    assert second["reused"] is False


@pytest.mark.parametrize("target", ["log", "receipt"])
def test_tampered_log_or_receipt_is_never_reused(
    tmp_path: Path, target: str,
) -> None:
    command = [sys.executable, "-c", "print('evidence-bound')"]
    context = _context(tmp_path)
    first = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "evidence.log",
        label="evidence check",
        qualification_context=context,
    )
    selected = tmp_path / first["path" if target == "log" else "receipt_path"]
    if target == "log":
        selected.write_bytes(selected.read_bytes() + b"tampered\n")
    else:
        receipt = json.loads(selected.read_bytes())
        receipt["log_size"] += 1
        selected.write_bytes(qualification.release._canonical(receipt))
    second = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(),
        log=tmp_path / "evidence.log",
        label="evidence check",
        qualification_context=context,
    )
    assert second["attempt"] == 2
    assert second["reused"] is False


@pytest.mark.parametrize("target", ["log", "receipt"])
def test_recursive_check_validation_rejects_tampered_evidence(
    tmp_path: Path, target: str,
) -> None:
    command = [sys.executable, "-c", "print('recursive')"]
    context = _context(tmp_path)
    record = qualification._run(
        command,
        cwd=tmp_path,
        environment=_environment(tmp_path),
        log=tmp_path / "recursive.log",
        label="recursive check",
        qualification_context=context,
    )
    selected = tmp_path / record["path" if target == "log" else "receipt_path"]
    selected.write_bytes(selected.read_bytes() + b"tampered\n")
    with pytest.raises(qualification.QualificationError):
        qualification._validate_check_record(
            record,
            repo=tmp_path,
            evidence_root=tmp_path,
            context_sha256=context["qualification_context_sha256"],
            expected_label="recursive check",
            expected_runtime_sha256="a" * 64,
            expected_environment=_environment(tmp_path),
            expected_command=command,
            expected_log_name="recursive.log",
            expected_record_keys=qualification.CHECK_RECORD_KEYS,
        )


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda env: env.update(OPAL_PREFIX="/unqualified"), id="opal"),
        pytest.param(lambda env: env.update(OMPI_MCA_pml="ucx"), id="mpi-pml"),
        pytest.param(lambda env: env.update(OMPI_MCA_btl="self,tcp"), id="mpi-btl"),
        pytest.param(lambda env: env.update(TMPDIR="/unqualified"), id="tmpdir"),
        pytest.param(lambda env: env.update(HOME="/unqualified"), id="home"),
        pytest.param(lambda env: env.update(EXTRA_CONTROL="injected"), id="extra-key"),
        pytest.param(lambda env: env.pop("OPAL_PREFIX"), id="missing-key"),
    ],
)
def test_recursive_validation_rejects_rehashed_environment_forgery(
    tmp_path: Path, mutate: object,
) -> None:
    command = [sys.executable, "-c", "print('validator environment binding')"]
    context = _context(tmp_path)
    environment = _environment(tmp_path)
    record = qualification._run(
        command, cwd=tmp_path, environment=environment,
        log=tmp_path / "forgery.log", label="forgery check",
        qualification_context=context,
    )
    qualification._validate_check_record(
        record, repo=tmp_path, evidence_root=tmp_path,
        context_sha256=context["qualification_context_sha256"],
        expected_label="forgery check", expected_runtime_sha256="a" * 64,
        expected_environment=environment,
        expected_command=command, expected_log_name="forgery.log",
        expected_record_keys=qualification.CHECK_RECORD_KEYS,
    )
    receipt_path = tmp_path / record["receipt_path"]
    receipt = json.loads(receipt_path.read_text())
    mutate(receipt["identity"]["environment"])
    receipt["identity_sha256"] = hashlib.sha256(
        qualification.release._canonical(receipt["identity"])
    ).hexdigest()
    receipt_path.write_bytes(qualification.release._canonical(receipt))
    record["receipt_sha256"] = qualification.release._sha256(receipt_path)
    record["identity_sha256"] = receipt["identity_sha256"]
    with pytest.raises(qualification.QualificationError, match="authenticate"):
        qualification._validate_check_record(
            record, repo=tmp_path, evidence_root=tmp_path,
            context_sha256=context["qualification_context_sha256"],
            expected_label="forgery check",
            expected_runtime_sha256="a" * 64,
            expected_environment=environment,
            expected_command=command, expected_log_name="forgery.log",
            expected_record_keys=qualification.CHECK_RECORD_KEYS,
        )


def test_qualification_output_paths_reject_symlink_ancestry_before_writes(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(qualification.QualificationError, match="symlink"):
        qualification._qualification_output_paths(alias / "qualification.json")
    assert not (outside / "qualification-evidence").exists()


def test_qualification_output_paths_reject_final_symlink(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.json"
    target.write_text("preserve\n", encoding="utf-8")
    output = tmp_path / "qualification.json"
    output.symlink_to(target)
    with pytest.raises(
        qualification.QualificationError, match="non-symlink regular file"
    ):
        qualification._qualification_output_paths(output)
    assert target.read_text(encoding="utf-8") == "preserve\n"
    assert not (tmp_path / "qualification-evidence").exists()


def test_qualification_output_paths_reject_writable_existing_report(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    output.write_text("preserve\n", encoding="utf-8")
    output.chmod(0o644)
    with pytest.raises(qualification.QualificationError, match="mode 0444"):
        qualification._qualification_output_paths(output)
    assert output.read_text(encoding="utf-8") == "preserve\n"
    assert not (tmp_path / "qualification-evidence").exists()
