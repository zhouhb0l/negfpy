from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from negfpy.anharmonic._launch_environment import (
    build_launch_environment,
    qualification_environment,
    validate_launch_environment,
)


@pytest.fixture
def launch_paths(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "repo": tmp_path / "candidate",
        "executable": tmp_path / "runtime/bin/python3.12",
        "home": tmp_path / "job-home",
        "tmpdir": tmp_path / "node-local",
        "git_bin": tmp_path / "git-tools/bin",
    }
    for name, path in paths.items():
        if name == "executable":
            path.parent.mkdir(parents=True)
            path.write_text("runtime fixture\n", encoding="utf-8")
        else:
            path.mkdir(parents=True)
    return paths


def test_sanitizer_ignores_ambient_injection_and_binds_explicit_paths(
    launch_paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = build_launch_environment(**launch_paths)
    for name, value in {
        "PYTHONPATH": "/untrusted/modules",
        "LD_PRELOAD": "/untrusted/loader.so",
        "LD_LIBRARY_PATH": "/untrusted/libraries",
        "API_SECRET": "private-test-value",
        "OPENAI_API_KEY": "private-test-value",
        "PATH": "/untrusted/tools",
        "OPAL_PREFIX": "/untrusted/openmpi",
        "OMPI_MCA_pml": "ucx",
        "OMPI_MCA_btl": "self,tcp",
        "KMP_DUPLICATE_LIB_OK": "False",
        "HOME": "/untrusted/home",
        "TMPDIR": "/untrusted/tmp",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.sshCommand",
        "GIT_CONFIG_VALUE_0": "/untrusted/ssh",
    }.items():
        monkeypatch.setenv(name, value)
    environment = build_launch_environment(**launch_paths)
    assert environment == baseline
    assert environment["PATH"] == ":".join((
        str(launch_paths["executable"].parent),
        str(launch_paths["git_bin"]), "/usr/bin", "/bin",
    ))
    assert environment["PYTHONPATH"] == ":".join((
        str(launch_paths["repo"] / "src"),
        str(launch_paths["repo"] / "scripts"),
    ))
    assert environment["HOME"] == str(launch_paths["home"])
    assert environment["TMPDIR"] == str(launch_paths["tmpdir"])
    assert environment["OPAL_PREFIX"] == str(launch_paths["executable"].parent.parent)
    assert environment["OMPI_MCA_pml"] == "ob1"
    assert environment["OMPI_MCA_btl"] == "self,sm"
    assert environment["OMPI_MCA_rmaps_base_oversubscribe"] == "1"
    assert environment["KMP_DUPLICATE_LIB_OK"] == "True"
    for name in (
        "LD_PRELOAD", "LD_LIBRARY_PATH", "API_SECRET", "OPENAI_API_KEY",
        "GIT_CONFIG_KEY_0", "GIT_CONFIG_VALUE_0",
    ):
        assert name not in environment
    assert "NEGFPY_QUALIFICATION_RUNTIME_SHA256" not in environment


def test_sanitized_environment_round_trips_and_qualification_adds_only_runtime_hash(
    launch_paths: dict[str, Path],
) -> None:
    environment = build_launch_environment(**launch_paths)
    original = dict(environment)
    validated = validate_launch_environment(
        environment,
        repo=launch_paths["repo"], executable=launch_paths["executable"],
    )
    runtime = {
        "executable": str(launch_paths["executable"]),
        "launch_environment": environment,
        "runtime_sha256": "a" * 64,
    }
    launch = qualification_environment(runtime, repo=launch_paths["repo"])
    assert validated == original
    assert environment == original
    assert launch == original | {"NEGFPY_QUALIFICATION_RUNTIME_SHA256": "a" * 64}


@pytest.mark.parametrize(
    "name,value",
    [
        ("OPAL_PREFIX", "/unqualified/runtime"),
        ("OMPI_MCA_pml", "ucx"),
        ("OMPI_MCA_btl", "self,tcp"),
        ("OMPI_MCA_rmaps_base_oversubscribe", "0"),
        ("KMP_DUPLICATE_LIB_OK", "False"),
        ("KMP_DUPLICATE_LIB_OK", "true"),
        ("PYTHONPATH", "/untrusted/modules"),
        ("OMP_NUM_THREADS", "8"),
        ("PYTHONHASHSEED", "random"),
        ("LD_PRELOAD", "/untrusted/loader.so"),
        ("API_SECRET", "must-not-be-accepted"),
        ("NEGFPY_QUALIFICATION_RUNTIME_SHA256", "b" * 64),
    ],
)
def test_validator_rejects_changed_controls_and_extra_variables(
    launch_paths: dict[str, Path], name: str, value: str,
) -> None:
    environment = build_launch_environment(**launch_paths) | {name: value}
    with pytest.raises(ValueError):
        validate_launch_environment(
            environment,
            repo=launch_paths["repo"], executable=launch_paths["executable"],
        )


@pytest.mark.parametrize("name", ["HOME", "TMPDIR", "PATH", "OPAL_PREFIX", "OMPI_MCA_btl", "KMP_DUPLICATE_LIB_OK"])
def test_validator_rejects_missing_launch_variables(
    launch_paths: dict[str, Path], name: str,
) -> None:
    environment = build_launch_environment(**launch_paths)
    del environment[name]
    with pytest.raises(ValueError):
        validate_launch_environment(
            environment,
            repo=launch_paths["repo"], executable=launch_paths["executable"],
        )


@pytest.mark.parametrize("name", ["HOME", "TMPDIR", "PATH", "OPAL_PREFIX"])
def test_validator_rejects_non_string_environment_values(
    launch_paths: dict[str, Path], name: str,
) -> None:
    environment = build_launch_environment(**launch_paths)
    environment[name] = None
    with pytest.raises(ValueError):
        validate_launch_environment(
            environment,
            repo=launch_paths["repo"], executable=launch_paths["executable"],
        )


@pytest.mark.parametrize("value", ["relative/bin:/git/bin:/usr/bin:/bin", "/bin", "/a:/b:/usr/bin:/bin:/extra"])
def test_validator_rejects_unsafe_or_expanded_path(
    launch_paths: dict[str, Path], value: str,
) -> None:
    environment = build_launch_environment(**launch_paths) | {"PATH": value}
    with pytest.raises(ValueError):
        validate_launch_environment(
            environment,
            repo=launch_paths["repo"], executable=launch_paths["executable"],
        )


def test_qualification_refuses_unvalidated_environment(
    launch_paths: dict[str, Path],
) -> None:
    runtime = {
        "executable": str(launch_paths["executable"]),
        "launch_environment": build_launch_environment(**launch_paths) | {
            "OMPI_MCA_pml": "ucx",
        },
        "runtime_sha256": "a" * 64,
    }
    with pytest.raises(ValueError):
        qualification_environment(runtime, repo=launch_paths["repo"])


@pytest.mark.parametrize("digest", [None, "", "a" * 63, "g" * 64, "A" * 64])
def test_qualification_rejects_invalid_runtime_marker(
    launch_paths: dict[str, Path], digest: object,
) -> None:
    runtime = {
        "executable": str(launch_paths["executable"]),
        "launch_environment": build_launch_environment(**launch_paths),
        "runtime_sha256": digest,
    }
    with pytest.raises(ValueError):
        qualification_environment(runtime, repo=launch_paths["repo"])


def test_pmix_relocation_is_bound_and_cannot_drift(tmp_path):
    environment = build_launch_environment(
        repo=tmp_path, executable=tmp_path / "runtime/bin/python3.12",
        home=tmp_path, tmpdir=tmp_path, git_bin=Path("/usr/bin"),
    )
    assert environment["PMIX_PREFIX"] == str(tmp_path / "runtime")
    for changed in (dict(environment, PMIX_PREFIX="/old/build/prefix"),
                    {k:v for k,v in environment.items() if k != "PMIX_PREFIX"}):
        with pytest.raises(ValueError, match="sanitized policy"):
            validate_launch_environment(changed, repo=tmp_path,
                executable=tmp_path / "runtime/bin/python3.12")


@pytest.mark.parametrize("entry", ["import", "main-help"])
def test_fresh_pilot_startup_preserves_the_entire_qualified_environment(
    tmp_path: Path, entry: str,
) -> None:
    """Exercise real dependency imports outside pytest's populated modules."""
    repo = Path(__file__).resolve().parents[1]
    executable = Path(sys.executable).resolve()
    runtime = {
        "executable": str(executable),
        "launch_environment": build_launch_environment(
            repo=repo, executable=executable, home=tmp_path,
            tmpdir=tmp_path, git_bin=Path("/usr/bin"),
        ),
        "runtime_sha256": "a" * 64,
    }
    environment = qualification_environment(runtime, repo=repo)
    driver = repo / "scripts/run_material_nq1_predictor_pilot.py"
    # Capture immediately on interpreter entry, before the real driver loads
    # threadpoolctl. Do not replace os.environ or pre-import candidate modules.
    code = """
import importlib
import json
import os
import runpy
import sys
before = dict(os.environ)
driver, entry = sys.argv[1:]
if entry == 'import':
    module = importlib.import_module('run_material_nq1_predictor_pilot')
    assert module.__file__ == driver
else:
    sys.argv = [driver, '--help']
    try:
        runpy.run_path(driver, run_name='__main__')
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError('pilot CLI help did not terminate')
print(json.dumps({'before': before, 'after': dict(os.environ),
                  'threadpoolctl_loaded': 'threadpoolctl' in sys.modules},
                 sort_keys=True))
"""
    completed = subprocess.run(
        [str(executable), "-B", "-c", code, str(driver), entry],
        cwd=repo, env=environment, capture_output=True, text=True,
        check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["threadpoolctl_loaded"] is True
    assert observed["before"] == environment
    assert observed["after"] == environment
    assert observed["after"]["KMP_DUPLICATE_LIB_OK"] == "True"
