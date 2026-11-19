"""Synthetic checkpoint publication/admission; no material or SCBA runs."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from negfpp import checkpoint
from negfpy.anharmonic.complete_energy import ChannelSCBARecord, SignedFrequencyGrid, SolverScale, UniformBZMesh
from negfpy.anharmonic.transverse_channel_feedback import ChannelSCBAState, ChannelSpectrum
from negfpy.anharmonic.transverse_solver import TransverseCurrents


@pytest.fixture
def case(tmp_path):
    grid, mesh, scale = SignedFrequencyGrid(.1, 1), UniformBZMesh((1, 1, 1)), SolverScale()
    vertex = SimpleNamespace(full_ndof=6, ndof=3, active_dofs=np.arange(3, dtype=np.int64))
    doc = dict(schema="negfpp.run.v1", model=dict(type="saved_site_energy"),
               inputs_sha256={"asset": "a"*64}, geometry=dict(n_device_layers=2, interaction_layers=[1]),
               grid=dict(mesh=[1, 1, 1], step=.1, n_positive=1),
               reservoirs=dict(left_temperature_k=300.5, right_temperature_k=299.5, eta_lead=.0015625, eta_device=0.),
               scale=dict(omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar), scba=dict(g=1.))
    config = SimpleNamespace(document=doc, hashes=dict(asset=dict(file="asset", bytes=7, sha256="a"*64)))
    info = dict(full_atoms=2, full_dofs=6, active_atoms=1, active_dofs=3, energy_centers=1,
                q_points=1, frequency_nodes=3, source_units=dict(force_constants="eV/angstrom^3", mass="amu"),
                support=dict(energy_identity_sha256="b"*64, geometry_sha256="c"*64,
                             full_ndof=6, active_dofs=3, all_outer_derivatives_retained=True,
                             cropped_local_derivative_blocks=0))
    return SimpleNamespace(config=config, info=info, grid=grid, mesh=mesh, vertex=vertex, scale=scale,
                           path=tmp_path/"latest.h5")


def _state(case, index=2, *, zero=False, converged=False, transpose=False):
    shape = (1, 3, 3, 3)
    retarded = np.zeros(shape, dtype=np.complex128)
    retarded.view(np.float64).reshape(-1)[::2] = -0.
    statistical = np.full(shape, 1e100j, dtype=np.complex128)
    spectral = np.full(shape, 1e-200j, dtype=np.complex128)
    if transpose:
        retarded, statistical, spectral = (v.swapaxes(-1, -2) for v in (retarded, statistical, spectral))
    fields = None if zero else ChannelSpectrum(retarded, statistical, spectral)
    sigma = {name: dict(ratio=.5 if converged else 2.) for name in ("retarded", "lesser", "greater")}
    current = {name: dict(ratio=.5) for name in ("contact_balance", "collision_power", "full_balance")}
    record = ChannelSCBARecord(index, .01, .02, sigma, current, {}, converged, True, converged)
    checks = {name: dict(synthetic_fixture=dict(ratio=.5)) for name in checkpoint._PHYSICAL}
    evaluation = SimpleNamespace(checks=checks, input_currents=TransverseCurrents(1., -1., 0.))
    return ChannelSCBAState(index, fields, evaluation, record)


def _options(case):
    return dict(binding=checkpoint.make_binding(case.config, case.info), grid=case.grid,
                mesh=case.mesh, vertex=case.vertex, scale=case.scale, max_bytes=100000)


def _write(case, state=None, decision=None, *, selected=None, **overrides):
    state = state or _state(case)
    options = _options(case)
    options.update(source_input_bytes=("\n " + json.dumps(case.config.document, indent=3) + "\n").encode(),
                   completed_ordinal=state.index-1 if state.index>1 else 1,
                   selected_anchor_index=state.index if selected is None else selected)
    options.update(overrides)
    return checkpoint.write_latest(case.path, state, decision, **options)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(case, **overrides):
    options = _options(case)
    options.update(overrides)
    return checkpoint.load_checkpoint(case.path, _sha(case.path), **options)


def _mutate_metadata(path, change):
    with h5py.File(path, "r+") as handle:
        value = json.loads(handle["metadata_json"][...].tobytes())
        change(value)
        del handle["metadata_json"]
        handle.create_dataset("metadata_json", data=np.frombuffer(json.dumps(value).encode(), dtype=np.uint8))


@pytest.mark.parametrize("model", ["saved_site_energy", "saved_nep_site_derivatives"])
@pytest.mark.parametrize("transpose", [False, True])
def test_checkpoint_roundtrip_preserves_independent_native_bits_and_loads_no_other_state(case, model, transpose):
    case.config.document["model"]["type"] = model
    state = _state(case, transpose=transpose)
    saved = _write(case, state)
    fields, receipt = _load(case)
    for name in checkpoint._CHANNELS:
        assert getattr(fields, name).tobytes() == getattr(state.sigma_input, name).tobytes()
        assert not getattr(fields, name).flags.writeable
        assert not np.shares_memory(getattr(fields, name), getattr(state.sigma_input, name))
    assert np.signbit(fields.retarded.real).all()
    assert np.any(fields.spectral != 0)
    assert receipt["loaded_field_role"] == "sigma_input_evaluated"
    assert receipt["source_reported_converged"] is False
    assert receipt["initial_guess_is_unevaluated_on_target"] and receipt["fresh_full_map_required"]
    for name in ("green_loaded", "mapped_image_loaded", "convergence_history_loaded", "acceleration_history_loaded"):
        assert receipt[name] is False
    assert saved["sha256"] == _sha(case.path)
    with h5py.File(case.path) as handle:
        assert "green_evaluated" not in handle and "sigma_born_image_unevaluated" not in handle
        raw = handle["source_input_json"][...].tobytes()
        assert receipt["source_input_sha256"] == hashlib.sha256(raw).hexdigest()
        assert raw.startswith(b"\n ")


def test_rejected_completed_trial_overwrites_latest_without_promoting_selected_anchor(case):
    _write(case)
    state = _state(case, 3)
    state.sigma_input.spectral[...] = 7e-200j
    _write(case, state, dict(accepted_as_anchor=False, selected_anchor_index=2), selected=2)
    fields, receipt = _load(case)
    assert receipt["source_evaluated_state_index"] == 3
    assert receipt["source_selected_anchor_index"] == 2
    assert receipt["source_accepted_as_anchor"] is False
    assert np.array_equal(fields.spectral, state.sigma_input.spectral)
    assert len(list(case.path.parent.iterdir())) == 1


def test_exact_zero_checkpoint_has_explicit_role_and_loads_readonly_zero_fields(case):
    _write(case, _state(case, 1, zero=True))
    fields, receipt = _load(case)
    with h5py.File(case.path) as handle:
        assert len(handle["sigma_input_evaluated"]) == 0
    assert receipt["source_evaluated_state_index"] == 1
    for name in checkpoint._CHANNELS:
        assert not np.any(getattr(fields, name)) and not getattr(fields, name).flags.writeable


def test_failed_atomic_replace_keeps_last_complete_checkpoint_and_cleans_temp(case, monkeypatch):
    _write(case)
    original = case.path.read_bytes()
    def fail_replace(*args):
        raise OSError("generated precommit filesystem failure")
    monkeypatch.setattr(checkpoint.os, "replace", fail_replace)
    with pytest.raises(OSError, match="precommit"):
        _write(case, _state(case, 3))
    assert case.path.read_bytes() == original
    assert list(case.path.parent.iterdir()) == [case.path]
    assert _load(case)[1]["source_evaluated_state_index"] == 2


def test_nonfinite_new_input_keeps_old_checkpoint(case):
    _write(case)
    original = case.path.read_bytes()
    state = _state(case, 3)
    state.sigma_input.statistical[0, 1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        _write(case, state)
    assert case.path.read_bytes() == original
    assert list(case.path.parent.iterdir()) == [case.path]


def test_writer_payload_admission_precedes_file_creation(case):
    with pytest.raises(ValueError, match="max_bytes"):
        _write(case, max_bytes=1)
    assert not list(case.path.parent.iterdir())


def test_loader_payload_admission_precedes_numerical_reads(case, monkeypatch):
    _write(case)
    monkeypatch.setattr(h5py.Dataset, "read_direct", lambda *args: pytest.fail("No numerical field read admitted"))
    with pytest.raises(ValueError, match="max_bytes"):
        _load(case, max_bytes=1)


@pytest.mark.parametrize("change", ["temperature", "g", "geometry", "model", "asset", "support"])
def test_loader_rejects_physical_binding_mismatch_before_numerical_reads(case, monkeypatch, change):
    _write(case)
    if change == "temperature": case.config.document["reservoirs"]["left_temperature_k"] = 305.
    elif change == "g": case.config.document["scba"]["g"] = .9
    elif change == "geometry": case.config.document["geometry"]["interaction_layers"] = [0]
    elif change == "model": case.config.document["model"]["type"] = "saved_nep_site_derivatives"
    elif change == "asset": case.config.hashes["asset"]["sha256"] = "d"*64
    else: case.info["support"]["energy_identity_sha256"] = "d"*64
    monkeypatch.setattr(h5py.Dataset, "read_direct", lambda *args: pytest.fail("Mismatched input must precede fields"))
    with pytest.raises(ValueError, match="physical binding"):
        _load(case)


@pytest.mark.parametrize("defect", ["missing", "dtype", "nonfinite", "units", "ordering", "role", "index", "physical", "six"])
def test_structural_and_physical_tampering_rejected_even_with_updated_file_digest(case, defect):
    _write(case)
    if defect in ("physical", "six"):
        def change(metadata):
            if defect == "physical": metadata["physical_checks"]["mapped_sigma"]["synthetic_fixture"]["ratio"] = 2.
            else: del metadata["record"]["sigma_residuals"]["lesser"]
        _mutate_metadata(case.path, change)
    else:
        with h5py.File(case.path, "r+") as handle:
            group = handle["sigma_input_evaluated"]
            if defect == "missing": del group["spectral"]
            elif defect == "dtype":
                data = group["spectral"][...].astype(np.complex64)
                del group["spectral"]
                group.create_dataset("spectral", data=data)
            elif defect == "nonfinite": group["spectral"][0, 0, 0, 0] = np.inf
            elif defect == "units": group.attrs["units"] = "wrong"
            elif defect == "ordering": handle["active_device_indices"][...] = [2, 1, 0]
            elif defect == "role": handle.attrs["mapped_image_saved"] = True
            elif defect == "index": handle.attrs["evaluated_state_index"] = 99
    with pytest.raises(ValueError):
        _load(case)


@pytest.mark.parametrize("link", ["external", "soft", "virtual", "external-storage"])
def test_unsafe_hdf5_links_and_storage_rejected(case, tmp_path, link):
    _write(case)
    with h5py.File(case.path, "r+") as handle:
        group = handle["sigma_input_evaluated"]
        shape = group["spectral"].shape
        del group["spectral"]
        if link == "external": group["spectral"] = h5py.ExternalLink("absent.h5", "/data")
        elif link == "soft": group["spectral"] = h5py.SoftLink("/sigma_input_evaluated/retarded")
        elif link == "virtual":
            layout = h5py.VirtualLayout(shape, dtype=np.complex128)
            layout[...] = h5py.VirtualSource("absent.h5", "data", shape=shape)
            group.create_virtual_dataset("spectral", layout)
        else:
            group.create_dataset("spectral", shape=shape, dtype=np.complex128,
                                 external=[(str(tmp_path/"outside.bin"), 0, h5py.h5f.UNLIMITED)])
    with pytest.raises(ValueError, match="unsupported"):
        _load(case)


def test_wrong_digest_rejected_before_hdf5_open(case, monkeypatch):
    _write(case)
    monkeypatch.setattr(checkpoint.h5py, "File", lambda *args: pytest.fail("Wrong digest cannot reach HDF5"))
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        checkpoint.load_checkpoint(case.path, "0"*64, **_options(case))


def test_make_binding_detaches_values_and_excludes_path_and_solver_controls(case):
    bound = checkpoint.make_binding(case.config, case.info)
    case.config.document["reservoirs"]["left_temperature_k"] = 305.
    case.config.hashes["asset"]["file"] = "elsewhere"
    case.config.document["scba"]["method"] = "reference-seeded"
    assert bound["physical"]["reservoirs"]["left_temperature_k"] == 300.5
    assert "file" not in bound["input_roles"]["asset"]
    assert "scba" not in bound


def test_writer_rejects_mismatched_raw_source_input_before_creating_file(case):
    payload = json.dumps(dict(case.config.document, scba=dict(g=.9))).encode()
    with pytest.raises(ValueError, match="input bytes"):
        _write(case, source_input_bytes=payload)
    assert not list(case.path.parent.iterdir())


def test_inconsistent_selected_anchor_or_fabricated_convergence_cannot_be_published(case):
    with pytest.raises(ValueError, match="anchor"):
        _write(case, decision=dict(accepted_as_anchor=False), selected=2)
    state = _state(case)
    state = replace(state, record=replace(state.record, converged=True))
    with pytest.raises(ValueError, match="six screens"):
        _write(case, state)
    assert not list(case.path.parent.iterdir())


def _workflow_case(case, tmp_path, monkeypatch):
    """Build normal authenticated JSON, replacing only numerical preparation."""
    import shutil
    from negfpp.input import load_input
    import negfpp.workflow as workflow
    example = Path(__file__).resolve().parents[1]/"examples/silicon24"
    root = tmp_path/"public-input"
    root.mkdir()
    document = json.loads((example/"input.json").read_text())
    document["run"].update(name="checkpoint-unit", output_dir="output")
    document["checkpoint"] = dict(file="self-energy-latest.h5")
    document["grid"] = case.config.document["grid"]
    document["scale"] = case.config.document["scale"]
    document["geometry"].update(case.config.document["geometry"])
    document["scba"].update(g=1., max_evaluations=3, max_bytes=100000)
    for name in document["inputs_sha256"]:
        shutil.copy2(example/name, root/name)
    source = root/"input.json"
    source.write_text(json.dumps(document, indent=3))
    case.config = load_input(source)
    monkeypatch.setattr(workflow, "prepare", lambda config, deadline:
        (case.grid, case.mesh, np.zeros((1, 6, 6)), (object(), object()),
         SimpleNamespace(vertex=case.vertex), case.scale, case.info))
    return workflow, source


def _workflow_state(case, index, *, zero=False):
    from dataclasses import dataclass
    @dataclass(frozen=True)
    class Ratio:
        ratio: float
    state = _state(case, index, zero=zero)
    record = replace(state.record,
                     sigma_residuals={key: Ratio(value["ratio"]) for key, value in state.record.sigma_residuals.items()},
                     current_residuals={key: Ratio(value["ratio"]) for key, value in state.record.current_residuals.items()})
    return replace(state, record=record)


def test_workflow_publishes_every_completed_trial_and_preserves_rejected_latest_on_interruption(case, tmp_path, monkeypatch):
    workflow, _ = _workflow_case(case, tmp_path, monkeypatch)
    callbacks = []
    def interrupted(*args, **kwargs):
        for index in (1, 2, 3):
            state = _workflow_state(case, index, zero=index==1)
            kwargs["on_start"](index, state.sigma_input, None)
            decision = None if index<3 else dict(accepted_as_anchor=False, selected_anchor_index=2)
            kwargs["on_evaluation"](state, decision)
            path = case.config.output/"self-energy-latest.h5"
            with h5py.File(path) as handle:
                assert handle.attrs["evaluated_state_index"] == index
            progress = json.loads((case.config.output/"report.json").read_text())
            assert progress["last_evaluated"]["index"] == index
            assert progress["checkpoint"]["evaluated_state_index"] == index
            callbacks.append(index)
        raise RuntimeError("generated interruption after complete rejected trial")
    monkeypatch.setattr(workflow, "solve_from_zero", interrupted)
    report, output = workflow.run(case.config)
    assert callbacks == [1, 2, 3]
    assert report["status"] == "failed" and report["converged"] is False
    assert report["counts"] == dict(attempted=3, completed=3, selected=2, rejected=1)
    assert report["selected_anchor"]["index"] == 2 and report["last_evaluated"]["index"] == 3
    assert report["checkpoint"]["selected_anchor_index"] == 2
    assert "currents_w" not in report and "finite_bias_conductance_w_per_k" not in report
    assert not (output/"fields.h5").exists()
    case.path = output/"self-energy-latest.h5"
    _, receipt = _load(case)
    assert receipt["source_evaluated_state_index"] == 3 and receipt["source_accepted_as_anchor"] is False
    assert len(list(output.glob("*latest*.h5"))) == 1


def test_workflow_checkpoint_failure_stops_without_losing_previous_publication(case, tmp_path, monkeypatch):
    workflow, _ = _workflow_case(case, tmp_path, monkeypatch)
    original_write = checkpoint.write_latest
    committed = []
    def fail_second(path, state, decision, **kwargs):
        if state.index == 2:
            raise OSError("generated checkpoint publication failure")
        saved = original_write(path, state, decision, **kwargs)
        committed.append(saved["evaluated_state_index"])
        return saved
    monkeypatch.setattr(checkpoint, "write_latest", fail_second)
    starts = []
    def interrupted(*args, **kwargs):
        for index in (1, 2, 3):
            starts.append(index)
            state = _workflow_state(case, index, zero=index==1)
            kwargs["on_start"](index, state.sigma_input, None)
            kwargs["on_evaluation"](state, None)
        pytest.fail("Publication failure must stop before the next map")
    monkeypatch.setattr(workflow, "solve_from_zero", interrupted)
    report, output = workflow.run(case.config)
    assert starts == [1, 2] and committed == [1]
    assert report["converged"] is False and "publication failure" in report["failure_reason"]
    assert report["checkpoint"]["evaluated_state_index"] == 1
    case.path = output/"self-energy-latest.h5"
    _, receipt = _load(case)
    assert receipt["source_evaluated_state_index"] == 1


def test_public_restart_dispatch_admits_nonconverged_rejected_checkpoint_only_as_fresh_guess(case, tmp_path, monkeypatch):
    workflow, source = _workflow_case(case, tmp_path, monkeypatch)
    state = _workflow_state(case, 3)
    saved = _write(case, state, dict(accepted_as_anchor=False, selected_anchor_index=2), selected=2)
    document = json.loads(source.read_text())
    document["initial_self_energy"] = dict(checkpoint="../latest.h5", checkpoint_sha256=saved["sha256"])
    source.write_text(json.dumps(document, indent=5))
    from negfpp.input import load_input
    case.config = load_input(source)
    calls = []
    def fresh(*args, **kwargs):
        guess = kwargs["sigma_initial"]
        assert guess.spectral.tobytes() == state.sigma_input.spectral.tobytes()
        assert not guess.spectral.flags.writeable
        # The next physical evaluation is newly counted; no accepted result or
        # source decision is carried into the new run.
        new = _workflow_state(case, 2)
        kwargs["on_start"](2, guess, None)
        kwargs["on_evaluation"](new, None)
        calls.append(2)
        return SimpleNamespace(result=SimpleNamespace(state=None, converged=False,
            failure_reason=None, stop_reason="max_evaluations", plan={}),
            attempted=1, completed=1, selected=1, rejected=0, records=(), diagnostics=None)
    monkeypatch.setattr(workflow, "solve_from_self_energy", fresh)
    monkeypatch.setattr(workflow, "solve_from_zero", lambda *a, **k: pytest.fail("Loaded fields must reach seeded API"))
    report, _ = workflow.run(case.config)
    assert calls == [2] and report["converged"] is False
    assert report["initialization"]["source_reported_converged"] is False
    assert report["initialization"]["source_accepted_as_anchor"] is False
    assert report["initialization"]["fresh_full_map_required"] is True
    assert report["counts"] == dict(attempted=1, completed=1, selected=1, rejected=0)
    assert "currents_w" not in report
