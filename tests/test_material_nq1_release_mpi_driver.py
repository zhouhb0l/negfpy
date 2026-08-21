from __future__ import annotations

import ast
import importlib
import inspect
import sys
import textwrap
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
)
from negfpy.anharmonic._material_nq1_mpi import (
    build_material_nq1_mpi_campaign,
)
from negfpy.anharmonic._mpi_tasks import TaskSpec
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain

SCRIPTS = Path(__file__).parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
driver = importlib.import_module("run_material_nq1_release_mpi")
_REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY = (
    driver._reconstruct_task_checkpoint_identity
)
_REAL_LOAD_CHECKPOINT = driver.case.load_scba_checkpoint_v2_self_validating


_COMPLETE_SOURCE_HASHES = {
    "phase_a_fixture": "1" * 64,
    "raw_fc3": "2" * 64,
    "phase_b_artifact_set": "3" * 64,
    "phase_b_mapped_hdf5": "4" * 64,
    "phase_b_campaign_manifest": "5" * 64,
    "cubic_vertex_physics": "6" * 64,
    "cubic_vertex_storage": "7" * 64,
    "vertex_metadata.phase_b.archive_sha256": "8" * 64,
}


def _checkpoint_identity(task_id: str, **changes: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "run_id": task_id,
        "identity_sha256": "d" * 64,
        "source_hashes": dict(_COMPLETE_SOURCE_HASHES),
        "initial_state_sha256": "9" * 64,
    }
    values.update(changes)
    return SimpleNamespace(**values)


def _checkpoint_identity_from_manifest(
    manifest: Mapping[str, object],
) -> SimpleNamespace:
    source_hashes = dict(_COMPLETE_SOURCE_HASHES)
    initial_state_sha256 = "9" * 64
    born = manifest.get("born_initial")
    if isinstance(born, Mapping):
        source_hashes.update({
            "born_initial_file": born["file_sha256"],
            "born_initial_identity": born["identity_sha256"],
            "born_initial_manifest": born["manifest_sha256"],
            "born_initial_state": born["initial_state_sha256"],
        })
        initial_state_sha256 = str(born["initial_state_sha256"])
    return _checkpoint_identity(
        str(manifest["run_id"]),
        source_hashes=source_hashes,
        initial_state_sha256=initial_state_sha256,
    )


def _small_authenticated_inputs():
    device, lead, _dense = pinned_scalar_chain(1, pinning=1.0, cubic=0.2)
    vertex = SparseCubicVertex(
        ndof=1,
        indices=np.asarray([[0, 0, 0]], dtype=np.int64),
        coefficients=np.asarray([0.2], dtype=np.float64),
        metadata={"phase_b": {"archive_sha256": "8" * 64}},
    )
    problem = CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=vertex,
        grid=SignedFrequencyGrid(step=0.2, n_positive=5),
    )
    config = SCBAConfig(
        eta_lead=0.025,
        max_iterations=800,
        min_iterations=3,
        consecutive_passes=3,
    )
    material_identity = {
        "fixture": {"fixture_manifest_sha256": "1" * 64},
        "fc3_full": {"raw_sha256": "2" * 64},
        "phase_b": {
            "artifact_set_sha256": "3" * 64,
            "archive": {"sha256": "4" * 64},
            "campaign_manifest_sha256": "5" * 64,
        },
    }
    return material_identity, problem, config


@pytest.fixture(autouse=True)
def _bounded_checkpoint_identity_reconstruction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reconstruct(**kwargs: object):
        scientific = kwargs["scientific_task"]
        manifest = kwargs["scba_manifest"]
        assert isinstance(scientific, Mapping)
        assert isinstance(manifest, Mapping)
        initialization = manifest.get("continuation_parent")
        continuation = (
            None
            if initialization is None
            else initialization.get("initialization")
        )
        return (
            _checkpoint_identity_from_manifest(manifest),
            continuation,
            manifest.get("born_initial"),
        )

    def load(path: Path):
        manifest = driver._load_canonical(
            Path(path).parents[1] / "campaign-manifest.json",
            "test SCBA manifest",
        )
        return SimpleNamespace(identity=_checkpoint_identity_from_manifest(manifest))

    monkeypatch.setattr(driver, "_reconstruct_task_checkpoint_identity", reconstruct)
    monkeypatch.setattr(
        driver.case, "load_scba_checkpoint_v2_self_validating", load
    )


def _write(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(driver._canonical(value))


def _fixture(tmp_path: Path, *, task_id: str = "coupling-g0001"):
    campaign = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )
    scientific = campaign.task(task_id)
    spec = campaign.manifest.task(scientific.task_id)
    phase_module = importlib.import_module("negfpy.anharmonic._material_nq1_mpi")
    phase = phase_module._phase_manifest(
        campaign, campaign.wave_index(scientific.task_id)
    )
    task_identity = phase.task_identity_sha256(spec)
    receipt_path = tmp_path / "000000-0123456789abcdef.json"
    data = receipt_path.with_suffix(".data")
    scba = data / "scba"
    checkpoint = scba / "checkpoint" / "sigma-pp.h5"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"authenticated checkpoint fixture\n")
    born_record = None
    if scientific.initialization == "born":
        born_path = data / "born-initial.h5"
        born_path.write_bytes(b"authenticated Born fixture\n")
        born_record = {
            "path": str(born_path.resolve()),
            "file_sha256": driver._sha256(born_path),
            "identity_sha256": "a" * 64,
            "manifest_sha256": "b" * 64,
            "initial_state_sha256": "c" * 64,
        }
    controls = {
        "grid_step": scientific.grid_step,
        "grid_n_positive": scientific.grid_n_positive,
        "grid_cutoff": scientific.grid_cutoff,
        "eta_lead": scientific.eta_lead,
        "mixing": scientific.mixing,
        "mixing_floor": scientific.mixing_floor,
        "temperature_bias": scientific.temperature_bias_solver,
    }
    manifest = {
        "schema": driver.single_campaign.SCHEMA,
        "run_id": scientific.task_id,
        "coupling_scale": scientific.coupling_scale,
        "continuation_parent": None,
        "continuation_axis": None,
        "continuation_predictor": None,
        "born_initial": born_record,
        "preflight": {"continuation_parent": None},
        "numerical_controls": controls,
        "watchdog": {"automatic_retry": False},
    }
    _write(scba / "campaign-manifest.json", manifest)
    manifest_sha = driver._sha256(scba / "campaign-manifest.json")
    _write(
        scba / "logs/scba.json",
        {"attempt": 1, "resume": False, "campaign_manifest_sha256": manifest_sha},
    )
    report = {
        "campaign_manifest_sha256": manifest_sha,
        "state": "passed",
        "scba_converged": True,
        "checkpoint_sha256": driver._sha256(checkpoint),
        "automatic_retry": False,
        "controls_changed": False,
        "attempt_logs": [
            {
                "path": "logs/scba.json",
                "sha256": driver._sha256(scba / "logs/scba.json"),
            }
        ],
        "infrastructure_failures": [],
    }
    _write(scba / "scba-report.json", report)
    status = {
        "campaign_manifest_sha256": manifest_sha,
        "state": "passed",
        "scba_converged": True,
        "report_sha256": driver._sha256(scba / "scba-report.json"),
        "automatic_retry": False,
    }
    _write(scba / "status.json", status)
    records = driver._tree_records(data)
    receipt = {
        "schema": driver.RECEIPT_SCHEMA,
        "campaign_manifest_sha256": campaign.manifest.manifest_sha256,
        "task_id": scientific.task_id,
        "task_identity_sha256": task_identity,
        "scientific_task_sha256": scientific.task_sha256,
        "state": "passed",
        "task_directory": data.name,
        "task_tree_records": records,
        "task_tree_sha256": driver._tree_sha256(records),
        "scba_status_sha256": driver._sha256(scba / "status.json"),
        "scba_report_sha256": driver._sha256(scba / "scba-report.json"),
        "checkpoint_sha256": driver._sha256(checkpoint),
        "born_initial": born_record,
        "continuation_parent_task_id": None,
        "continuation_initialization": None,
        "release_certified": False,
    }
    _write(receipt_path, receipt)
    bindings = {
        "expected_campaign_manifest_sha256": campaign.manifest.manifest_sha256,
        "expected_task_identity_sha256": task_identity,
        "authenticated_child_inputs": ({"authenticated": True}, object(), object()),
        "expected_source_tree_sha256": "a" * 64,
        "expected_environment_sha256": "b" * 64,
    }
    return receipt_path, spec, scba, checkpoint, bindings


@pytest.mark.parametrize("task_id", ("coupling-g0001", "endpoint-born-start"))
def test_material_receipt_authenticates_entire_terminal_tree(
    tmp_path: Path, task_id: str,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(
        tmp_path, task_id=task_id
    )
    evidence = driver._receipt_evidence(receipt, spec, **bindings)
    assert evidence["task_id"] == task_id
    assert evidence["state"] == "passed"


@pytest.mark.parametrize("task_id", ("coupling-g0001", "endpoint-born-start"))
@pytest.mark.parametrize(
    "mutation",
    (
        "missing-derived",
        "altered-derived",
        "extra-derived",
        "altered-explicit",
        "altered-initial-state",
    ),
)
def test_material_receipt_rejects_inexact_complete_checkpoint_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_id: str,
    mutation: str,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(
        tmp_path, task_id=task_id
    )
    manifest = driver._load_canonical(
        _scba / "campaign-manifest.json", "test SCBA manifest"
    )
    expected = _checkpoint_identity_from_manifest(manifest)
    source_hashes = dict(expected.source_hashes)
    initial_state_sha256 = expected.initial_state_sha256
    if mutation == "missing-derived":
        source_hashes.pop("cubic_vertex_physics")
    elif mutation == "altered-derived":
        source_hashes["cubic_vertex_storage"] = "f" * 64
    elif mutation == "extra-derived":
        source_hashes["vertex_metadata.unexpected.source_sha256"] = "f" * 64
    elif mutation == "altered-explicit":
        source_hashes["phase_a_fixture"] = "f" * 64
    else:
        initial_state_sha256 = "f" * 64
    observed = _checkpoint_identity(
        task_id,
        source_hashes=source_hashes,
        initial_state_sha256=initial_state_sha256,
    )
    monkeypatch.setattr(
        driver.case,
        "load_scba_checkpoint_v2_self_validating",
        lambda _path: SimpleNamespace(identity=observed),
    )

    with pytest.raises(
        driver.MaterialCampaignError,
        match="complete identity did not reproduce",
    ):
        driver._receipt_evidence(receipt, spec, **bindings)


@pytest.mark.parametrize(
    "source_name",
    (
        "born_initial_file",
        "born_initial_identity",
        "born_initial_manifest",
        "born_initial_state",
    ),
)
def test_born_receipt_rejects_each_checkpoint_source_binding_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_name: str,
) -> None:
    receipt, spec, scba, _checkpoint, bindings = _fixture(
        tmp_path, task_id="endpoint-born-start"
    )
    manifest = driver._load_canonical(
        scba / "campaign-manifest.json", "test SCBA manifest"
    )
    expected = _checkpoint_identity_from_manifest(manifest)
    source_hashes = dict(expected.source_hashes)
    source_hashes[source_name] = "f" * 64
    observed = _checkpoint_identity(
        "endpoint-born-start",
        source_hashes=source_hashes,
        initial_state_sha256=expected.initial_state_sha256,
    )
    monkeypatch.setattr(
        driver.case,
        "load_scba_checkpoint_v2_self_validating",
        lambda _path: SimpleNamespace(identity=observed),
    )

    with pytest.raises(
        driver.MaterialCampaignError,
        match="complete identity did not reproduce",
    ):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_material_receipt_rejects_plain_text_checkpoint_even_when_hash_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    monkeypatch.setattr(
        driver.case,
        "load_scba_checkpoint_v2_self_validating",
        _REAL_LOAD_CHECKPOINT,
    )

    with pytest.raises(driver.MaterialCampaignError, match="checkpoint is invalid"):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_checkpoint_reconstruction_authenticates_zero_and_born_modes(
    tmp_path: Path,
) -> None:
    campaign = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )
    authenticated = _small_authenticated_inputs()
    material_identity, problem, config = authenticated
    explicit = driver.case._source_hashes(material_identity)
    zero = campaign.task("coupling-g0001")
    zero_data = tmp_path / "zero"
    zero_data.mkdir()
    zero_identity, zero_continuation, zero_born = (
        _REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY(
            scientific_task=zero.as_dict(),
            scba_manifest={"continuation_parent": None, "born_initial": None},
            task_data=zero_data,
            authenticated_task_inputs=authenticated,
            authenticated_parent_scba=None,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )
    )
    expected_derived = {
        "cubic_vertex_physics": problem.vertex.cubic_physics_sha256,
        "cubic_vertex_storage": problem.vertex.vertex_storage_sha256,
        "vertex_metadata.phase_b.archive_sha256": "8" * 64,
        "vertex_metadata.permutation_symmetry.cubic_physics_sha256": (
            problem.vertex.cubic_physics_sha256
        ),
    }
    assert dict(zero_identity.source_hashes) == explicit | expected_derived
    assert zero_continuation is None
    assert zero_born is None

    born = campaign.task("endpoint-born-start")
    born_data = tmp_path / "born"
    born_data.mkdir()
    born_path = born_data / "born-initial.h5"
    driver.write_born_initial_state(
        born_path,
        problem,
        config,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        source_hashes=explicit,
    )
    _initial, born_record = driver.load_born_initial_state(
        born_path,
        problem,
        config,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        source_hashes=explicit,
    )
    born_identity, born_continuation, derived_born = (
        _REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY(
            scientific_task=born.as_dict(),
            scba_manifest={
                "continuation_parent": None,
                "born_initial": born_record,
            },
            task_data=born_data,
            authenticated_task_inputs=authenticated,
            authenticated_parent_scba=None,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )
    )
    assert born_continuation is None
    assert derived_born == born_record
    assert born_identity.initial_state_sha256 == born_record[
        "initial_state_sha256"
    ]
    for name, digest in {
        "born_initial_file": born_record["file_sha256"],
        "born_initial_identity": born_record["identity_sha256"],
        "born_initial_manifest": born_record["manifest_sha256"],
        "born_initial_state": born_record["initial_state_sha256"],
    }.items():
        assert born_identity.source_hashes[name] == digest


def test_checkpoint_reconstruction_rejects_unexpected_zero_or_changed_born(
    tmp_path: Path,
) -> None:
    campaign = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    )
    authenticated = _small_authenticated_inputs()
    material_identity, problem, config = authenticated
    zero_data = tmp_path / "zero"
    zero_data.mkdir()
    (zero_data / "born-initial.h5").write_bytes(b"unexpected\n")
    with pytest.raises(driver.MaterialCampaignError, match="zero.*changed"):
        _REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY(
            scientific_task=campaign.task("coupling-g0001").as_dict(),
            scba_manifest={"continuation_parent": None, "born_initial": None},
            task_data=zero_data,
            authenticated_task_inputs=authenticated,
            authenticated_parent_scba=None,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )

    born_data = tmp_path / "born"
    born_data.mkdir()
    born_path = born_data / "born-initial.h5"
    explicit = driver.case._source_hashes(material_identity)
    driver.write_born_initial_state(
        born_path,
        problem,
        config,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        source_hashes=explicit,
    )
    _initial, born_record = driver.load_born_initial_state(
        born_path,
        problem,
        config,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        source_hashes=explicit,
    )
    changed_born = dict(born_record)
    changed_born["file_sha256"] = "f" * 64
    with pytest.raises(driver.MaterialCampaignError, match="recorded Born state"):
        _REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY(
            scientific_task=campaign.task("endpoint-born-start").as_dict(),
            scba_manifest={
                "continuation_parent": None,
                "born_initial": changed_born,
            },
            task_data=born_data,
            authenticated_task_inputs=authenticated,
            authenticated_parent_scba=None,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )

    born_path.write_bytes(b"tampered Born state\n")
    with pytest.raises(
        driver.MaterialCampaignError,
        match="cannot independently authenticate Born state",
    ):
        _REAL_RECONSTRUCT_TASK_CHECKPOINT_IDENTITY(
            scientific_task=campaign.task("endpoint-born-start").as_dict(),
            scba_manifest={
                "continuation_parent": None,
                "born_initial": born_record,
            },
            task_data=born_data,
            authenticated_task_inputs=authenticated,
            authenticated_parent_scba=None,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


def test_material_receipt_rejects_checkpoint_mutation(tmp_path: Path) -> None:
    receipt, spec, _scba, checkpoint, bindings = _fixture(tmp_path)
    checkpoint.write_bytes(b"changed\n")
    with pytest.raises(driver.MaterialCampaignError, match="task tree"):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_release_executor_never_infers_resume_from_failed_artifacts() -> None:
    tree = ast.parse(textwrap.dedent(inspect.getsource(driver.run)))
    run_campaign_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "run_campaign"
    ]
    assert len(run_campaign_calls) == 1
    resume_values = [
        keyword.value
        for keyword in run_campaign_calls[0].keywords
        if keyword.arg == "resume"
    ]
    assert len(resume_values) == 1
    assert isinstance(resume_values[0], ast.Constant)
    assert resume_values[0].value is False
    assert not any(
        isinstance(node, ast.Attribute) and node.attr == "_failure_records"
        for node in ast.walk(tree)
    )
    named_calls = {
        node.func.id: node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id in {
            "_collective_release_authorization", "_prepare_campaign_root"
        }
    }
    assert named_calls["_collective_release_authorization"] < named_calls[
        "_prepare_campaign_root"
    ]


class _SerialCommunicator:
    def Get_size(self) -> int:
        return 1


class _CollectiveCommunicator:
    def __init__(self, observations: list[dict[str, object]]) -> None:
        self.observations = observations
        self.local: dict[str, object] | None = None

    def Get_size(self) -> int:
        return len(self.observations)

    def allgather(self, value: dict[str, object]) -> list[dict[str, object]]:
        self.local = value
        return self.observations


def test_release_authorization_is_authenticated_identically_on_every_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorization = {"passed": True, "qualification": {"sha256": "a" * 64}}
    digest = driver.hashlib.sha256(driver._canonical(authorization)).hexdigest()
    monkeypatch.setattr(
        driver,
        "_validate_release_authorization",
        lambda **_kwargs: authorization,
    )
    communicator = _CollectiveCommunicator([
        {"rank": 0, "authorization_sha256": digest, "error": None},
        {"rank": 1, "authorization_sha256": digest, "error": None},
    ])
    observed = driver._collective_release_authorization(
        communicator=communicator,
        rank=0,
        repo=tmp_path,
        qualification_path=tmp_path / "qualification.json",
        pilot_audit_path=tmp_path / "pilot-audit.json",
        source={},
        source_tree_sha256="b" * 64,
        environment_sha256="c" * 64,
        scientific_input_hashes={},
    )
    assert observed == authorization
    assert communicator.local == {
        "rank": 0,
        "authorization_sha256": digest,
        "error": None,
    }

    communicator.observations[1] = {
        "rank": 1,
        "authorization_sha256": None,
        "error": "MaterialCampaignError: wrong runtime",
    }
    with pytest.raises(driver.MaterialCampaignError, match="before campaign creation"):
        driver._collective_release_authorization(
            communicator=communicator,
            rank=0,
            repo=tmp_path,
            qualification_path=tmp_path / "qualification.json",
            pilot_audit_path=tmp_path / "pilot-audit.json",
            source={},
            source_tree_sha256="b" * 64,
            environment_sha256="c" * 64,
            scientific_input_hashes={},
        )


def test_release_campaign_root_must_be_new_and_cannot_follow_symlinks(
    tmp_path: Path,
) -> None:
    communicator = _SerialCommunicator()
    campaign = tmp_path / "runs/material-nq1-release-new"
    campaign.parent.mkdir()
    created = driver._prepare_campaign_root(
        campaign, communicator=communicator, rank=0
    )
    assert created == campaign
    assert created.is_dir()
    marker = created / "preserved-evidence"
    marker.write_text("preserve\n", encoding="utf-8")
    with pytest.raises(driver.MaterialCampaignError, match="already exists"):
        driver._prepare_campaign_root(
            campaign, communicator=communicator, rank=0
        )
    assert marker.read_text(encoding="utf-8") == "preserve\n"

    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(driver.MaterialCampaignError, match="symlink"):
        driver._prepare_campaign_root(
            alias / "escaped-campaign", communicator=communicator, rank=0
        )
    assert not (outside / "escaped-campaign").exists()


def test_release_atomic_json_is_mode_0444_and_rejects_symlink_targets(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "campaign"
    parent.mkdir()
    published = driver._atomic_json(parent / "record.json", {"passed": True})
    assert published.stat().st_mode & 0o777 == 0o444
    assert driver._atomic_json(parent / "record.json", {"passed": True}) == published

    outside = tmp_path / "outside.json"
    outside.write_bytes(driver._canonical({"passed": True}))
    alias = parent / "alias.json"
    alias.symlink_to(outside)
    with pytest.raises(driver.MaterialCampaignError, match="differs"):
        driver._atomic_json(alias, {"passed": True})
    assert alias.is_symlink()


def test_material_receipt_rejects_prior_infrastructure_history(
    tmp_path: Path,
) -> None:
    receipt, spec, scba, _checkpoint, bindings = _fixture(tmp_path)
    failure_path = scba / "infrastructure-failure.json"
    _write(
        failure_path,
        {
            "schema": "negfpy.sparse-nq1-phase-c-scba-infrastructure-failure.v1",
            "classification": "infrastructure-failure",
        },
    )
    report = driver._load_canonical(scba / "scba-report.json", "fixture report")
    report["infrastructure_failures"] = [
        {
            "path": "infrastructure-failure.json",
            "sha256": driver._sha256(failure_path),
        }
    ]
    _write(scba / "scba-report.json", report)
    status = driver._load_canonical(scba / "status.json", "fixture status")
    status["report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(scba / "status.json", status)
    records = driver._tree_records(receipt.with_suffix(".data"))
    payload = driver._load_canonical(receipt, "fixture receipt")
    payload["task_tree_records"] = records
    payload["task_tree_sha256"] = driver._tree_sha256(records)
    payload["scba_status_sha256"] = driver._sha256(scba / "status.json")
    payload["scba_report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(receipt, payload)
    with pytest.raises(driver.MaterialCampaignError, match="infrastructure history"):
        driver._receipt_evidence(receipt, spec, **bindings)


@pytest.mark.parametrize(
    ("control", "changed_value"),
    (("mixing", 0.1), ("mixing_floor", 0.005)),
)
def test_material_receipt_rejects_changed_numerical_controls(
    tmp_path: Path, control: str, changed_value: float,
) -> None:
    receipt, spec, scba, _checkpoint, bindings = _fixture(tmp_path)
    manifest = driver._load_canonical(scba / "campaign-manifest.json", "fixture")
    manifest["numerical_controls"][control] = changed_value
    _write(scba / "campaign-manifest.json", manifest)
    manifest_sha = driver._sha256(scba / "campaign-manifest.json")
    report = driver._load_canonical(scba / "scba-report.json", "fixture report")
    report["campaign_manifest_sha256"] = manifest_sha
    _write(scba / "scba-report.json", report)
    status = driver._load_canonical(scba / "status.json", "fixture status")
    status["campaign_manifest_sha256"] = manifest_sha
    status["report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(scba / "status.json", status)
    records = driver._tree_records(receipt.with_suffix(".data"))
    payload = driver._load_canonical(receipt, "fixture receipt")
    payload["task_tree_records"] = records
    payload["task_tree_sha256"] = driver._tree_sha256(records)
    payload["scba_status_sha256"] = driver._sha256(scba / "status.json")
    payload["scba_report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(receipt, payload)
    with pytest.raises(driver.MaterialCampaignError, match="numerical controls"):
        driver._receipt_evidence(receipt, spec, **bindings)


@pytest.mark.parametrize(
    "binding",
    ("expected_campaign_manifest_sha256", "expected_task_identity_sha256"),
)
def test_material_receipt_rejects_cross_campaign_identity(
    tmp_path: Path, binding: str,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    bindings[binding] = "f" * 64
    with pytest.raises(driver.MaterialCampaignError, match="campaign|identity"):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_v3_material_receipt_requires_serialized_mixing_floor(
    tmp_path: Path,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    payload = {
        "schema": spec.payload["schema"],
        "dependencies": list(spec.payload["dependencies"]),
        "scientific_task": dict(spec.payload["scientific_task"]),
    }
    payload["scientific_task"].pop("mixing_floor")
    malformed = TaskSpec(
        task_id=spec.task_id,
        identity_hashes=spec.identity_hashes,
        payload=payload,
    )
    with pytest.raises(driver.MaterialCampaignError, match="lacks mixing_floor"):
        driver._receipt_evidence(receipt, malformed, **bindings)


def test_v3_material_receipt_requires_serialized_predictor(
    tmp_path: Path,
) -> None:
    receipt, spec, _scba, _checkpoint, bindings = _fixture(tmp_path)
    payload = {
        "schema": spec.payload["schema"],
        "dependencies": list(spec.payload["dependencies"]),
        "scientific_task": dict(spec.payload["scientific_task"]),
    }
    payload["scientific_task"].pop("continuation_predictor")
    malformed = TaskSpec(
        task_id=spec.task_id,
        identity_hashes=spec.identity_hashes,
        payload=payload,
    )
    with pytest.raises(
        driver.MaterialCampaignError, match="lacks continuation_predictor"
    ):
        driver._receipt_evidence(receipt, malformed, **bindings)


def test_material_receipt_rejects_unexpected_continuation_predictor(
    tmp_path: Path,
) -> None:
    receipt, spec, scba, _checkpoint, bindings = _fixture(tmp_path)
    manifest = driver._load_canonical(scba / "campaign-manifest.json", "fixture")
    manifest["continuation_predictor"] = "identity"
    _write(scba / "campaign-manifest.json", manifest)
    manifest_sha = driver._sha256(scba / "campaign-manifest.json")
    report = driver._load_canonical(scba / "scba-report.json", "fixture report")
    report["campaign_manifest_sha256"] = manifest_sha
    _write(scba / "scba-report.json", report)
    status = driver._load_canonical(scba / "status.json", "fixture status")
    status["campaign_manifest_sha256"] = manifest_sha
    status["report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(scba / "status.json", status)
    records = driver._tree_records(receipt.with_suffix(".data"))
    payload = driver._load_canonical(receipt, "fixture receipt")
    payload["task_tree_records"] = records
    payload["task_tree_sha256"] = driver._tree_sha256(records)
    payload["scba_status_sha256"] = driver._sha256(scba / "status.json")
    payload["scba_report_sha256"] = driver._sha256(scba / "scba-report.json")
    _write(receipt, payload)
    with pytest.raises(
        driver.MaterialCampaignError, match="continuation controls changed"
    ):
        driver._receipt_evidence(receipt, spec, **bindings)


def test_task_input_authentication_forwards_serialized_mixing_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    task = build_material_nq1_mpi_campaign(
        source_revision="candidate",
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        input_hashes={"fc2": "c" * 64, "fc3": "d" * 64},
    ).task("coupling-g100")
    captured: dict[str, object] = {}

    def authenticate(**kwargs):
        captured.update(kwargs)
        return "identity", "problem", "config"

    monkeypatch.setattr(driver.case, "authenticate_scba_inputs", authenticate)
    result = driver._authenticate_task_inputs(
        fixture=tmp_path / "fixture",
        fc3_full=tmp_path / "fc3.full.hdf5",
        phase_b_artifacts=tmp_path / "phase-b",
        task=task,
    )
    assert result == ("identity", "problem", "config")
    assert captured["mixing"] == task.mixing
    assert captured["mixing_floor"] == task.mixing_floor
    assert captured["coupling_scale"] == task.coupling_scale

    with pytest.raises(driver.MaterialCampaignError, match="mean temperature"):
        driver._authenticate_task_inputs(
            fixture=tmp_path / "fixture",
            fc3_full=tmp_path / "fc3.full.hdf5",
            phase_b_artifacts=tmp_path / "phase-b",
            task=replace(task, mean_temperature_kelvin=301.0),
        )


def _v3_qualification(repo: Path, source: dict[str, object]) -> dict[str, object]:
    base_records: dict[str, dict[str, object]] = {
        name: {} for name in (
            "python310_runtime", "python310_pytest",
            "python312_runtime", "python312_pytest",
            "mpi_ranks_2", "mpi_ranks_4",
            "material_input_preflight", "resources",
        )
    }
    for name in ("python310_pytest", "python312_pytest", "mpi_ranks_2", "mpi_ranks_4"):
        base_records[name]["returncode"] = 0
    base_records["resources"].update({
        "memory_guard_passed": True,
        "disk_guard_passed": True,
    })
    oracle_path = repo / driver.PREDICTOR_ORACLE_PATH
    return {
        "schema": driver.PREDICTOR_QUALIFICATION_SCHEMA,
        "source": source,
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "scientific_controls": driver._scientific_control_record(),
        "checks": {"all": True},
        "evidence": {
            "base_qualification_schema": driver.QUALIFICATION_SCHEMA,
            "base_qualification_evidence": {
                "qualification_context": {},
                "python_versions": {"python310": "3.10", "python312": "3.12"},
                "records": base_records,
            },
            "core_files": {
                name: {
                    "expected_sha256": digest,
                    "observed_sha256": digest,
                    "passed": True,
                }
                for name, digest in driver.PREDICTOR_CORE_SHA256.items()
            },
            "predictor_counts": {
                "coupling-square": 8, "identity": 4, "none": 7,
            },
            "property_oracle": {
                "relative_path": driver.PREDICTOR_ORACLE_PATH,
                "sha256": driver._sha256(oracle_path),
                "covered_by": ["python310_full_suite", "python312_full_suite"],
            },
        },
        "candidate_qualification_passed": True,
        "pilot_submission_eligible": True,
        "release_submission_eligible": False,
        "next_gate": "three-task-pilot",
        "predictor_pilot_controls": {},
    }


def _write_immutable(path: Path, value: dict[str, object]) -> None:
    _write(path, value)
    path.chmod(0o444)


def test_release_qualification_accepts_exact_v3_and_rejects_v2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(__file__).parents[1]
    source = {"commit": "candidate", "tree": "tree", "dirty": False}
    monkeypatch.setattr(
        driver.predictor_pilot,
        "qualification_authorizes_pilot",
        lambda _manifest, document: (
            document.get("schema") == driver.PREDICTOR_QUALIFICATION_SCHEMA
        ),
    )
    evidence_root = tmp_path / "qualification-evidence"
    evidence_root.mkdir()
    recursively_validated: list[tuple[object, Path, Path]] = []

    def recursive_validator(
        document: dict[str, object], *, repo: Path, evidence_root: Path,
    ) -> dict[str, object]:
        recursively_validated.append((document, repo, evidence_root))
        return dict(document)

    monkeypatch.setattr(
        driver, "_recursively_validate_predictor_qualification",
        recursive_validator,
    )
    runtime_record = {
        "schema": "negfpy.material-nq1-python-runtime.v1",
        "label": "python312",
        "version": "3.12.13",
        "executable": "/qualified/python3.12",
        "executable_sha256": "d" * 64,
        "packages_sha256": "e" * 64,
        "runtime_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        driver,
        "_authenticate_current_python312_runtime",
        lambda *_args, **_kwargs: dict(runtime_record),
    )
    exact = tmp_path / "qualification-v3.json"
    payload = _v3_qualification(repo, source)
    _write_immutable(exact, payload)
    record = driver._validate_qualification(
        exact,
        repo=repo,
        pilot_manifest={"pilot": "exact"},
        source=source,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
    )
    assert record["sha256"] == driver._sha256(exact)
    assert record["python312_runtime"] == runtime_record
    assert recursively_validated == [(payload, repo, evidence_root)]

    legacy = tmp_path / "qualification-v2.json"
    payload["schema"] = driver.QUALIFICATION_SCHEMA
    _write_immutable(legacy, payload)
    with pytest.raises(driver.MaterialCampaignError, match="exact v3"):
        driver._validate_qualification(
            legacy,
            repo=repo,
            pilot_manifest={"pilot": "exact"},
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


def test_release_runtime_replays_exact_qualified_python_and_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = tmp_path / "python3.12"
    executable.write_bytes(b"qualified executable\n")
    packages = b"numpy==2.5.1\n"
    runtime = {
        "schema": "negfpy.material-nq1-python-runtime.v1",
        "label": "python312",
        "version": "3.12.13",
        "executable": str(executable),
        "executable_sha256": driver._sha256(executable),
        "packages_sha256": driver.hashlib.sha256(packages).hexdigest(),
        "runtime_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        driver.predictor_pilot,
        "qualified_python312_runtime",
        lambda *_args: (dict(runtime), {"path": "/qualified/lock"}),
    )
    monkeypatch.setattr(driver.sys, "executable", str(executable))
    monkeypatch.setattr(driver.platform, "python_version", lambda: "3.12.13")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def replay(command: list[str], **kwargs: object) -> object:
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout=packages)

    monkeypatch.setattr(driver.subprocess, "run", replay)
    observed = driver._authenticate_current_python312_runtime(
        {"pilot": "exact"}, {"qualification": "exact"}, repo=tmp_path
    )
    assert observed == runtime
    assert calls == [(
        [str(executable), "-m", "pip", "freeze", "--all"],
        {
            "cwd": tmp_path,
            "stdout": driver.subprocess.PIPE,
            "stderr": driver.subprocess.STDOUT,
            "check": False,
            "timeout": driver.RUNTIME_REPLAY_TIMEOUT_SECONDS,
        },
    )]

    runtime["version"] = "3.12.12"
    with pytest.raises(driver.MaterialCampaignError, match="exact qualified"):
        driver._authenticate_current_python312_runtime(
            {}, {}, repo=tmp_path
        )
    runtime["version"] = "3.12.13"
    runtime["packages_sha256"] = "0" * 64
    with pytest.raises(driver.MaterialCampaignError, match="package inventory"):
        driver._authenticate_current_python312_runtime(
            {}, {}, repo=tmp_path
        )

    runtime["packages_sha256"] = driver.hashlib.sha256(packages).hexdigest()
    alias = tmp_path / "python-alias"
    alias.symlink_to(executable)
    runtime["executable"] = str(alias)
    monkeypatch.setattr(driver.sys, "executable", str(alias))
    with pytest.raises(driver.MaterialCampaignError, match="unavailable"):
        driver._authenticate_current_python312_runtime(
            {}, {}, repo=tmp_path
        )


def test_release_campaign_hashes_bind_qualified_runtime_and_packages() -> None:
    runtime = {
        "schema": "negfpy.material-nq1-python-runtime.v1",
        "label": "python312",
        "version": "3.12.13",
        "executable": "/qualified/python3.12",
        "executable_sha256": "1" * 64,
        "packages_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
    }
    authorization = {
        "qualification": {"sha256": "4" * 64, "python312_runtime": runtime},
        "pilot_manifest": {"sha256": "5" * 64},
        "pilot_report": {"sha256": "6" * 64},
        "pilot_audit": {"sha256": "7" * 64},
    }
    hashes = driver._release_campaign_input_hashes(
        scientific_input_hashes={"phase_a_fixture": "8" * 64},
        observation_modes_sha256="9" * 64,
        harmonic_buffer_sha256="a" * 64,
        release_authorization=authorization,
    )
    assert hashes["qualified_python312_executable"] == "1" * 64
    assert hashes["qualified_python312_packages"] == "2" * 64
    assert hashes["qualified_python312_runtime"] == "3" * 64

    del runtime["runtime_sha256"]
    with pytest.raises(driver.MaterialCampaignError, match="qualified P312"):
        driver._release_campaign_input_hashes(
            scientific_input_hashes={"phase_a_fixture": "8" * 64},
            observation_modes_sha256="9" * 64,
            harmonic_buffer_sha256="a" * 64,
            release_authorization=authorization,
        )


def test_release_task_manifest_records_exact_qualified_runtime() -> None:
    runtime = {
        "executable": "/qualified/python3.12",
        "version": "3.12.13",
    }
    manifest = {
        "python": dict(runtime),
        "distributions": dict(driver.common.EXPECTED_DISTRIBUTIONS),
        "thread_limits": dict(sorted(driver.case.phase_c.THREAD_LIMIT_ENV.items())),
    }
    driver._validate_scba_python312_runtime(manifest, runtime)

    manifest["python"]["executable"] = "/unqualified/python3.12"
    with pytest.raises(driver.MaterialCampaignError, match="qualified P312"):
        driver._validate_scba_python312_runtime(manifest, runtime)


def test_release_qualification_rejects_recursive_evidence_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = Path(__file__).parents[1]
    source = {"commit": "candidate", "tree": "tree", "dirty": False}
    (tmp_path / "qualification-evidence").mkdir()
    report = tmp_path / "qualification-v3.json"
    _write_immutable(report, _v3_qualification(repo, source))
    monkeypatch.setattr(
        driver.predictor_pilot,
        "qualification_authorizes_pilot",
        lambda *_args: True,
    )

    def reject(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("qualification receipt is unavailable")

    monkeypatch.setattr(
        driver, "_recursively_validate_predictor_qualification", reject
    )
    with pytest.raises(
        driver.MaterialCampaignError, match="recursively authenticate"
    ):
        driver._validate_qualification(
            report,
            repo=repo,
            pilot_manifest={"pilot": "exact"},
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


@pytest.mark.parametrize("target", ["qualification-ancestor", "evidence-root"])
def test_release_qualification_rejects_symlinked_evidence_ancestry(
    tmp_path: Path, target: str,
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    report = real / "qualification-v3.json"
    _write_immutable(report, {"schema": driver.PREDICTOR_QUALIFICATION_SCHEMA})
    real_evidence = real / "real-evidence"
    real_evidence.mkdir()
    if target == "qualification-ancestor":
        linked = tmp_path / "linked"
        linked.symlink_to(real, target_is_directory=True)
        selected = linked / report.name
        (real / "qualification-evidence").mkdir()
    else:
        selected = report
        (real / "qualification-evidence").symlink_to(
            real_evidence, target_is_directory=True
        )

    with pytest.raises(driver.MaterialCampaignError, match="symlink"):
        driver._validate_qualification(
            selected,
            repo=Path(__file__).parents[1],
            pilot_manifest={},
            source={},
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
        )


def _authorization_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, object], dict[str, object], dict[str, str], Path, Path]:
    source = {
        "commit": "1" * 40,
        "tree": "2" * 40,
        "dirty": False,
        "package_file": "/synthetic/src/negfpy/__init__.py",
    }
    inputs = {
        "phase_a_fixture": "3" * 64,
        "raw_fc3": "4" * 64,
        "phase_b_artifact_set": "5" * 64,
        "phase_b_mapped_hdf5": "6" * 64,
        "phase_b_campaign_manifest": "7" * 64,
        "scba_case_driver": "8" * 64,
    }
    qualification = tmp_path / "qualification.json"
    _write_immutable(qualification, {"schema": "synthetic-v3"})
    report_path = tmp_path / "pilot-report.json"
    _write_immutable(report_path, {"schema": "synthetic-passed-report"})
    audit_path = tmp_path / "pilot-audit.json"
    _write_immutable(audit_path, {"schema": "synthetic-passed-audit"})
    manifest = {
        "source_revision": source["commit"],
        "source_git_tree": source["tree"],
        "source_package_file": source["package_file"],
        "source_tree_sha256": "a" * 64,
        "environment_sha256": "b" * 64,
        "input_hashes": dict(inputs),
        "qualification_sha256": driver._sha256(qualification),
        "pilot_sha256": "9" * 64,
    }
    _write_immutable(tmp_path / driver.PILOT_MANIFEST_NAME, manifest)
    monkeypatch.setattr(
        driver.predictor_pilot, "validate_pilot_manifest", lambda value: dict(value)
    )
    monkeypatch.setattr(
        driver,
        "_validate_qualification",
        lambda *_args, **_kwargs: {
            "path": str(qualification),
            "sha256": driver._sha256(qualification),
            "checks": {"all": True},
            "scientific_task_set_sha256": "c" * 64,
        },
    )
    authorization = {
        "scope": "consideration-and-submission-gating-only",
        "next_campaign": "separately-identified-full-19-task-campaign",
        "required_source_revision": source["commit"],
        "required_source_tree_sha256": "a" * 64,
        "requires_identical_qualified_commit_and_tree": True,
        "source_or_scientific_control_changes_allowed": False,
    }
    audit = {
        "report": {"path": str(report_path), "sha256": driver._sha256(report_path)},
        "audit_passed": True,
        "release_certified": False,
    }
    report = {
        "pilot_passed": True,
        "all_tasks_completed": True,
        "all_scheduler_jobs_terminal_success": True,
        "strict_anchor_reproduction": True,
        "coupling_g0025_completed": True,
        "authorizes": authorization,
        "release_certified": False,
    }
    monkeypatch.setattr(
        driver.predictor_pilot, "validate_audit", lambda *_args, **_kwargs: audit
    )
    monkeypatch.setattr(
        driver.predictor_pilot,
        "validate_final_report",
        lambda *_args, **_kwargs: report,
    )
    return source, manifest, inputs, qualification, audit_path


def test_exact_passing_qualification_and_recursive_pilot_audit_authorize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _manifest, inputs, qualification, audit = _authorization_fixture(
        tmp_path, monkeypatch
    )
    record = driver._validate_release_authorization(
        repo=Path(__file__).parents[1],
        qualification_path=qualification,
        pilot_audit_path=audit,
        source=source,
        source_tree_sha256="a" * 64,
        environment_sha256="b" * 64,
        scientific_input_hashes=inputs,
    )
    assert record["passed"] is True
    assert record["identical_controls"] is True
    assert record["pilot_manifest"]["sha256"] == driver._sha256(
        tmp_path / driver.PILOT_MANIFEST_NAME
    )
    assert record["pilot_report"]["sha256"] == driver._sha256(
        tmp_path / "pilot-report.json"
    )
    assert record["pilot_audit"]["sha256"] == driver._sha256(audit)


@pytest.mark.parametrize(
    "foreign",
    ("commit", "tree", "environment", "inputs", "qualification"),
)
def test_foreign_pilot_audit_identity_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, foreign: str,
) -> None:
    source, manifest, inputs, qualification, audit = _authorization_fixture(
        tmp_path, monkeypatch
    )
    if foreign == "commit":
        manifest["source_revision"] = "f" * 40
    elif foreign == "tree":
        manifest["source_git_tree"] = "e" * 40
    elif foreign == "environment":
        manifest["environment_sha256"] = "d" * 64
    elif foreign == "inputs":
        manifest["input_hashes"] = dict(inputs) | {"raw_fc3": "c" * 64}
    else:
        manifest["qualification_sha256"] = "b" * 64
    manifest_path = tmp_path / driver.PILOT_MANIFEST_NAME
    manifest_path.chmod(0o644)
    _write_immutable(manifest_path, manifest)

    with pytest.raises(driver.MaterialCampaignError, match="identical-control"):
        driver._validate_release_authorization(
            repo=Path(__file__).parents[1],
            qualification_path=qualification,
            pilot_audit_path=audit,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
            scientific_input_hashes=inputs,
        )


def test_missing_or_failed_pilot_audit_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, _manifest, inputs, qualification, audit = _authorization_fixture(
        tmp_path, monkeypatch
    )
    audit.unlink()
    with pytest.raises((driver.MaterialCampaignError, OSError)):
        driver._validate_release_authorization(
            repo=Path(__file__).parents[1],
            qualification_path=qualification,
            pilot_audit_path=audit,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
            scientific_input_hashes=inputs,
        )

    _write_immutable(audit, {"schema": "synthetic-failed-audit"})
    monkeypatch.setattr(
        driver.predictor_pilot,
        "validate_audit",
        lambda *_args, **_kwargs: {
            "report": {
                "path": str(tmp_path / "pilot-report.json"),
                "sha256": driver._sha256(tmp_path / "pilot-report.json"),
            },
            "audit_passed": False,
            "release_certified": False,
        },
    )
    with pytest.raises(driver.MaterialCampaignError, match="identical-control"):
        driver._validate_release_authorization(
            repo=Path(__file__).parents[1],
            qualification_path=qualification,
            pilot_audit_path=audit,
            source=source,
            source_tree_sha256="a" * 64,
            environment_sha256="b" * 64,
            scientific_input_hashes=inputs,
        )


def test_release_cli_requires_pilot_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", [
        "run_material_nq1_release_mpi.py",
        "--repo-root", "repo",
        "--fixture", "fixture",
        "--fc3-full", "fc3",
        "--phase-b-artifacts", "phase-b",
        "--artifacts", "campaign",
        "--expected-commit", "1" * 40,
        "--environment-lock", "lock",
        "--qualification-report", "qualification",
    ])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2
