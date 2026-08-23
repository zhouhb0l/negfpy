from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
qualification = importlib.import_module(
    "qualify_material_nq1_predictor_candidate"
)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    contents = {
        "src/negfpy/anharmonic/solver.py": b"frozen solver\n",
        "src/negfpy/anharmonic/cubic.py": b"frozen cubic\n",
        "src/negfpy/anharmonic/_checkpoint_v2.py": b"frozen checkpoint\n",
        qualification.PROPERTY_ORACLE: b"def test_oracle(): pass\n",
    }
    for relative, payload in contents.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    monkeypatch.setattr(
        qualification,
        "CORE_FILE_SHA256",
        {
            relative: _sha(contents[relative])
            for relative in contents
            if relative != qualification.PROPERTY_ORACLE
        },
    )
    monkeypatch.setattr(
        qualification,
        "PROPERTY_ORACLE_SHA256",
        _sha(contents[qualification.PROPERTY_ORACLE]),
    )
    return repo


def _source(repo: Path) -> dict[str, object]:
    return {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "dirty": False,
        "package_file": str(repo / "src/negfpy/__init__.py"),
    }


def _base_report(repo: Path) -> dict[str, object]:
    checks = {
        name: True for name in qualification.release.EXPECTED_QUALIFICATION_CHECKS
    }
    suite = {"scope": ["tests", "docs/validation/bulk-rta-report/tests"],
             "excluded_tests": [], "returncode": 0}
    context_payload: dict[str, object] = {
        "schema": qualification.base.QUALIFICATION_CONTEXT_SCHEMA,
        "repo_root": str(repo),
        "source": _source(repo),
        "source_tree_sha256": "3" * 64,
        "environment_lock": {
            "path": str(repo / "lock"),
            "sha256": "4" * 64,
            "format": "@EXPLICIT",
        },
        "scientific_task_set_sha256": qualification.release.
        _scientific_control_record()["task_set_sha256"],
        "material_input_hashes": {
            "phase_a_fixture": "5" * 64,
            "raw_fc3": "6" * 64,
            "phase_b_artifact_set": "7" * 64,
            "phase_b_mapped_hdf5": "8" * 64,
            "phase_b_campaign_manifest": "9" * 64,
        },
        "runtimes": {"python310": {}, "python312": {}},
        "mpiexec": {"path": "/usr/bin/true", "sha256": "a" * 64},
        "tracked_test_files": {
            "tests/test.py": {
                "relative_path": "tests/test.py", "sha256": "b" * 64, "size": 1,
            },
        },
        "driver_files": {
            "scripts/driver.py": {
                "relative_path": "scripts/driver.py", "sha256": "c" * 64, "size": 1,
            },
        },
    }
    context = context_payload | {
        "qualification_context_sha256": qualification.base._context_digest(
            context_payload
        )
    }
    return {
        "schema": qualification.BASE_QUALIFICATION_SCHEMA,
        "source": _source(repo),
        "source_tree_sha256": "3" * 64,
        "environment_sha256": "4" * 64,
        "scientific_controls": qualification.release._scientific_control_record(),
        "checks": checks,
        "evidence": {
            "qualification_context": context,
            "python_versions": {"python310": "3.10.0", "python312": "3.12.0"},
            "records": {
                "python310_pytest": dict(suite),
                "python312_pytest": dict(suite),
            },
        },
        "release_preflight_passed": True,
    }


def _args(repo: Path) -> argparse.Namespace:
    return argparse.Namespace(
        repo_root=repo,
        fixture=Path("fixture"),
        fc3_full=Path("fc3"),
        phase_b_artifacts=Path("phase-b"),
        expected_commit="1" * 40,
        environment_lock=Path("lock"),
        python310=Path("python3.10"),
        python312=Path("python3.12"),
        mpiexec=Path("mpiexec"),
        output=repo / "qualification.json",
    )


def _patch_base(
    monkeypatch: pytest.MonkeyPatch,
    repo: Path,
    report: dict[str, object],
) -> None:
    monkeypatch.setattr(qualification.base, "qualify", lambda _args: report)
    monkeypatch.setattr(
        qualification.base,
        "validate_qualification_evidence",
        lambda document, **_kwargs: dict(document),
    )
    monkeypatch.setattr(
        qualification.base.common,
        "_source_identity",
        lambda _repo, _commit: _source(repo),
    )
    monkeypatch.setattr(
        qualification.base,
        "code_provenance",
        lambda: {"source_tree_sha256": "3" * 64},
    )


def test_wrapper_authorizes_only_the_exact_three_task_pilot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    _patch_base(monkeypatch, repo, base_report)

    report = qualification.qualify(_args(repo))

    assert report["schema"] == qualification.QUALIFICATION_SCHEMA
    assert report["candidate_qualification_passed"] is True
    assert report["pilot_submission_eligible"] is True
    assert report["release_submission_eligible"] is False
    assert report["next_gate"] == "staged-five-task-pilot"
    assert "release_preflight_passed" not in report
    assert report["scientific_controls"]["schema"].endswith("controls.v2")
    pilot_controls = report["predictor_pilot_controls"]
    assert pilot_controls["task_count"] == 5
    assert tuple(task["task_id"] for task in pilot_controls["tasks"]) == (
        "coupling-g0001", "coupling-g001", "coupling-g0015",
        "coupling-g002", "coupling-g0025",
    )
    assert report["evidence"]["predictor_counts"] == {
        "coupling-square": 8, "identity": 4, "none": 7,
    }
    assert pilot_controls["anchor_authentication_gate"] == (
        "staged-five-task-pilot-preflight"
    )
    context = report["evidence"]["base_qualification_evidence"][
        "qualification_context"
    ]
    assert qualification.base._validate_context_digest(context) == context[
        "qualification_context_sha256"
    ]


def test_tampered_full_scientific_controls_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    tampered = copy.deepcopy(base_report)
    tampered["scientific_controls"]["tasks"][1]["continuation_predictor"] = (
        "identity"
    )
    _patch_base(monkeypatch, repo, tampered)

    with pytest.raises(
        qualification.PredictorQualificationError,
        match="base candidate qualification is invalid",
    ):
        qualification.qualify(_args(repo))


def test_changed_frozen_core_code_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    _patch_base(monkeypatch, repo, base_report)
    (repo / "src/negfpy/anharmonic/solver.py").write_bytes(b"changed\n")

    with pytest.raises(
        qualification.PredictorQualificationError,
        match="frozen core file SHA256 changed",
    ):
        qualification.qualify(_args(repo))


def test_dirty_or_changed_source_after_base_suite_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    _patch_base(monkeypatch, repo, base_report)
    dirty = _source(repo) | {"dirty": True}
    monkeypatch.setattr(
        qualification.base.common,
        "_source_identity",
        lambda _repo, _commit: dirty,
    )

    with pytest.raises(
        qualification.PredictorQualificationError,
        match="source changed after",
    ):
        qualification.qualify(_args(repo))


def test_property_oracle_must_be_in_both_unexcluded_full_suites(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    base_report["evidence"]["records"]["python312_pytest"][
        "excluded_tests"
    ] = [qualification.PROPERTY_ORACLE]
    _patch_base(monkeypatch, repo, base_report)

    with pytest.raises(
        qualification.PredictorQualificationError,
        match="property oracle was not covered",
    ):
        qualification.qualify(_args(repo))


@pytest.mark.parametrize("target", ["core", "oracle"])
def test_recursive_v3_validation_rejects_tampered_frozen_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    _patch_base(monkeypatch, repo, base_report)
    report = qualification.qualify(_args(repo))
    tampered = copy.deepcopy(report)
    if target == "core":
        first = next(iter(tampered["evidence"]["core_files"].values()))
        first["observed_sha256"] = "f" * 64
    else:
        tampered["evidence"]["property_oracle"]["sha256"] = "f" * 64

    with pytest.raises(
        qualification.PredictorQualificationError,
        match="evidence did not authenticate",
    ):
        qualification.validate_predictor_qualification_report(
            tampered,
            repo=repo,
            evidence_root=repo / "qualification-evidence",
        )


def test_handwritten_all_true_v3_cannot_bypass_recursive_base_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    base_report = _base_report(repo)
    _patch_base(monkeypatch, repo, base_report)
    report = qualification.qualify(_args(repo))

    def reject_missing_evidence(*_args: object, **_kwargs: object) -> object:
        raise qualification.base.QualificationError("qualification log is unavailable")

    monkeypatch.setattr(
        qualification.base, "validate_qualification_evidence", reject_missing_evidence
    )
    with pytest.raises(
        qualification.PredictorQualificationError,
        match="base qualification evidence did not authenticate",
    ):
        qualification.validate_predictor_qualification_report(
            report,
            repo=repo,
            evidence_root=repo / "qualification-evidence",
        )


def test_report_publication_is_canonical_mode_0444_and_idempotent(
    tmp_path: Path,
) -> None:
    output = tmp_path / "qualification.json"
    report = {"schema": qualification.QUALIFICATION_SCHEMA, "passed": True}

    qualification._publish_report_once(output, report)
    assert output.read_bytes() == qualification.release._canonical(report)
    assert output.stat().st_mode & 0o777 == 0o444
    qualification._publish_report_once(output, report)

    with pytest.raises(
        qualification.PredictorQualificationError, match="differs"
    ):
        qualification._publish_report_once(output, report | {"passed": False})


def test_report_publication_rejects_symlink_and_special_output(
    tmp_path: Path,
) -> None:
    report = {"schema": qualification.QUALIFICATION_SCHEMA, "passed": True}
    target = tmp_path / "target.json"
    target.write_text("not qualification evidence\n", encoding="ascii")
    symlink = tmp_path / "symlink.json"
    symlink.symlink_to(target)
    with pytest.raises(
        qualification.PredictorQualificationError,
        match="non-symlink regular file",
    ):
        qualification._publish_report_once(symlink, report)

    fifo = tmp_path / "qualification.fifo"
    os.mkfifo(fifo)
    with pytest.raises(
        qualification.PredictorQualificationError,
        match="non-symlink regular file",
    ):
        qualification._publish_report_once(fifo, report)


def test_symlink_repository_root_is_rejected_before_base_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    linked_repo = tmp_path / "linked-repo"
    linked_repo.symlink_to(repo, target_is_directory=True)
    called = False

    def unexpected_base(_args: argparse.Namespace) -> dict[str, object]:
        nonlocal called
        called = True
        return _base_report(repo)

    monkeypatch.setattr(qualification.base, "qualify", unexpected_base)
    with pytest.raises(
        qualification.PredictorQualificationError,
        match="symlink",
    ):
        qualification.qualify(_args(linked_repo))
    assert called is False


def test_symlink_output_ancestry_is_rejected_before_base_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = _repo(tmp_path, monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "output-alias"
    alias.symlink_to(outside, target_is_directory=True)
    args = _args(repo)
    args.output = alias / "qualification.json"
    called = False

    def unexpected_base(_args: argparse.Namespace) -> dict[str, object]:
        nonlocal called
        called = True
        return _base_report(repo)

    monkeypatch.setattr(qualification.base, "qualify", unexpected_base)
    with pytest.raises(
        qualification.PredictorQualificationError, match="symlink"
    ):
        qualification.qualify(args)
    assert called is False
    assert not (outside / "qualification-evidence").exists()
