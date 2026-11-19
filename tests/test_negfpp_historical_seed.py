import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from negfpp.historical_seed import load_initial_guess, periodic_linear_weights, _leaves
from negfpy.anharmonic.grid import SignedFrequencyGrid
from negfpy.anharmonic.momentum import UniformBZMesh


def _dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def bundle(tmp_path):
    grid = SignedFrequencyGrid(.1, 1)
    mesh = UniformBZMesh((2, 2, 1))
    vertex = SimpleNamespace(full_ndof=3, ndof=3, active_dofs=np.arange(3))
    physical = {role: {"sha256": str(index) * 64, "bytes": 100 + index}
                for index, role in enumerate(("fc2", "crystal", "coefficients"), 1)}
    identity = dict(model_type="saved_site_energy", g=1, full_dofs=3,
        active_device_indices=[0, 1, 2], n_device_layers=1, interaction_layers=[0],
        mesh=[2, 2, 1], frequency_nodes=3, frequency_step=.1, omega0_rad_s=2., hbar_j_s=3.,
        temperatures_kelvin=[305, 295], eta_lead=.01, eta_device=0,
        geometry_sha256="a" * 64, energy_identity_sha256="b" * 64,
        all_outer_derivatives_retained=True, cropped_local_derivative_blocks=0,
        source_units={"force_constants": "eV/angstrom^3", "mass": "amu"}, physical_inputs=physical)
    proof = {"ratio": .1}
    record = dict(index=2, converged=True, sigma_passed=True, current_passed=True,
        accepted_as_anchor=True,
        sigma_residuals={name: proof.copy() for name in ("retarded", "lesser", "greater")},
        current_residuals={name: proof.copy() for name in ("contact_balance", "collision_power", "full_balance")},
        physical_checks={name: proof.copy() for name in ("harmonic", "input_sigma", "input_causality", "left",
            "right", "raw_source", "full_green_local", "active_green", "mapped_sigma")})
    shape = (4, 3, 3, 3)
    source = np.empty(shape, dtype=np.complex128)
    source.real = np.arange(source.size).reshape(shape)
    source.imag = -np.arange(source.size).reshape(shape)
    source.real[0, 0, 0, 0] = -0.
    source.imag[0, 0, 0, 0] = -0.
    arrays, artifacts = {}, {}
    for prefix in ("sigma_input_", "mapped_sigma_"):
        for component in ("retarded", "statistical", "spectral"):
            path = tmp_path / (prefix + component + ".npy")
            np.save(path, source)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts[prefix + component] = dict(relative_path=path.name, file_sha256=digest,
                file_bytes=path.stat().st_size, shape=list(shape), dtype="complex128")
            if prefix == "sigma_input_":
                arrays[component] = dict(file=path.name, sha256=digest, bytes=path.stat().st_size,
                    shape=list(shape), dtype="complex128")
    state = dict(schema="negfpy.transverse-channel-scba-checkpoint.v1", iteration=2,
        accepted_restart=False, sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False,
        input_sigma_is_zero=False, representation="retarded-statistical-spectral-v1",
        record=record, artifacts=artifacts, original_identity=identity,
        binding=dict(full_device_dofs=3, feedback_dofs=[0, 1, 2],
            energy_identity_sha256=identity["energy_identity_sha256"], geometry_sha256=identity["geometry_sha256"],
            physics=dict(mesh=[2, 2, 1], frequency_nodes=3, frequency_step=.1,
                device_layers=1, center_layers=[0], temperatures_kelvin=[305, 295],
                g=1, eta_lead=.01, eta_device=0, scale=dict(omega0=2., hbar=3.))))
    state_path = tmp_path / "original-state.json"
    state_sha = _dump(state_path, state)
    document = dict(schema="negfpp.historical-initial-self-energy.v1",
        source=dict(format="native_channel_manifest", evidence={"state": dict(file=state_path.name,
            sha256=state_sha, bytes=state_path.stat().st_size)}), identity=copy.deepcopy(identity),
        identity_bindings={pointer: dict(evidence="state", pointer="/original_identity" + pointer)
            for pointer, value in _leaves(identity) if pointer != "/model_type"}, arrays=arrays)
    manifest = tmp_path / "manifest.json"
    descriptor = dict(manifest=manifest, manifest_sha256=_dump(manifest, document),
        mode="same_model_closest_q_temperature_guess")
    config = SimpleNamespace(hashes=physical, document=dict(model={"type": "saved_site_energy"},
        scba={"g": 1, "max_bytes": 64 * 1024**2}, geometry={"n_device_layers": 1, "interaction_layers": [0]},
        grid={"mesh": [2, 2, 1], "step": .1, "n_positive": 1},
        scale={"omega0_rad_s": 2., "hbar_j_s": 3.},
        reservoirs=dict(left_temperature_k=300.5, right_temperature_k=299.5, eta_lead=.01, eta_device=0)))
    info = dict(frequency_nodes=3, q_points=4, source_units=identity["source_units"],
        support={key: identity[key] for key in ("geometry_sha256", "energy_identity_sha256",
            "all_outer_derivatives_retained", "cropped_local_derivative_blocks")})
    kwargs = dict(config=config, grid=grid, mesh=mesh, vertex=vertex,
        scale=SimpleNamespace(omega0=2., hbar=3.), info=info)
    return SimpleNamespace(path=manifest, descriptor=descriptor, document=document, state_path=state_path,
        state=state, source=source, kwargs=kwargs)


def _update(bundle):
    bundle.descriptor["manifest_sha256"] = _dump(bundle.path, bundle.document)


def test_exact_input_temperature_change_is_only_a_guess_and_preserves_bits(bundle):
    spectrum, receipt = load_initial_guess(bundle.descriptor, **bundle.kwargs)
    for component in (spectrum.retarded, spectrum.statistical, spectrum.spectral):
        assert component.tobytes() == bundle.source.tobytes()
        assert not component.flags.writeable
    assert receipt["source_temperatures_kelvin"] == [305, 295]
    assert receipt["target_temperatures_kelvin"] == [300.5, 299.5]
    assert receipt["source_convergence_admitted"]
    assert not receipt["source_convergence_inherited"]
    assert not receipt["mapped_image_or_green_loaded"]
    assert not receipt["controller_history_loaded"]
    assert not receipt["temperature_scaling_applied"]
    assert receipt["new_target_evaluation_required"]


@pytest.mark.parametrize("change", ["asset", "basis", "energy", "geometry", "frequency", "g"])
def test_target_physics_cannot_be_changed_silently(bundle, change):
    if change == "asset": bundle.kwargs["config"].hashes["coefficients"]["sha256"] = "f" * 64
    if change == "basis": bundle.kwargs["vertex"].active_dofs = np.array([2, 1, 0])
    if change in {"energy", "geometry"}: bundle.kwargs["info"]["support"][change + "_identity_sha256" if change == "energy" else "geometry_sha256"] = "f" * 64
    if change == "frequency": bundle.kwargs["grid"] = SignedFrequencyGrid(.2, 1)
    if change == "g": bundle.kwargs["config"].document["scba"]["g"] = .9
    with pytest.raises(ValueError, match="physical identity"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_normalized_identity_must_match_original_evidence(bundle):
    bundle.document["identity"]["temperatures_kelvin"] = [300.5, 299.5]
    _update(bundle)
    with pytest.raises(ValueError, match="original evidence"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_source_six_screen_failure_is_not_a_qualified_seed(bundle):
    bundle.state["record"]["sigma_residuals"]["retarded"]["ratio"] = 2
    pin = bundle.document["source"]["evidence"]["state"]
    pin["sha256"] = _dump(bundle.state_path, bundle.state)
    pin["bytes"] = bundle.state_path.stat().st_size
    _update(bundle)
    with pytest.raises(ValueError, match="six-screen"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_failed_source_physical_guard_is_rejected(bundle):
    bundle.state["record"]["physical_checks"]["raw_source"]["ratio"] = 2
    pin = bundle.document["source"]["evidence"]["state"]
    pin.update(sha256=_dump(bundle.state_path, bundle.state), bytes=bundle.state_path.stat().st_size)
    _update(bundle)
    with pytest.raises(ValueError, match="physical check failed"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_unrelated_authenticated_profile_cannot_relabel_selected_state(bundle):
    bundle.state["original_identity"]["temperatures_kelvin"] = [300.5, 299.5]
    bundle.document["identity"]["temperatures_kelvin"] = [300.5, 299.5]
    pin = bundle.document["source"]["evidence"]["state"]
    pin.update(sha256=_dump(bundle.state_path, bundle.state), bytes=bundle.state_path.stat().st_size)
    _update(bundle)
    with pytest.raises(ValueError, match="does not describe its original selected state"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_mapped_image_cannot_replace_input_even_with_a_valid_portable_hash(bundle):
    mapped = bundle.path.parent / "mapped_sigma_retarded.npy"
    changed = bundle.source.copy()
    changed.real += 1
    np.save(mapped, changed)
    row = bundle.document["arrays"]["retarded"]
    row.update(file=mapped.name, sha256=hashlib.sha256(mapped.read_bytes()).hexdigest(), bytes=mapped.stat().st_size)
    _update(bundle)
    with pytest.raises(ValueError, match="original selected native INPUT"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_array_tamper_is_detected(bundle):
    path = bundle.path.parent / bundle.document["arrays"]["retarded"]["file"]
    raw = bytearray(path.read_bytes()); raw[-1] ^= 1; path.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_loader_memory_is_admitted_before_array_load(bundle, monkeypatch):
    bundle.kwargs["config"].document["scba"]["max_bytes"] = 1
    monkeypatch.setattr(np, "load", lambda *a, **k: pytest.fail("array load before resource admission"))
    with pytest.raises(ValueError, match="exceeds"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_only_backend_validated_complete_energy_asset_equivalence_is_allowed(bundle):
    bundle.kwargs["info"]["initialization_physical_input_roles"] = ["source_archive"]
    with pytest.raises(ValueError, match="Unsupported historical physical-asset equivalence"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


def test_fresh_auxiliary_proof_cross_binds_original_manifest_and_input_bytes(bundle):
    identity = bundle.document["identity"]
    record = bundle.state["record"]
    seed = dict(mode="own-final-evaluated-input-only", source_index=2,
        manifest_sha256=bundle.document["source"]["evidence"]["state"]["sha256"])
    physics = dict(mesh=identity["mesh"], temperatures_kelvin=identity["temperatures_kelvin"],
        frequency_nodes=3, frequency_step=.1, g=1, device_layers=1, center_layers=[0],
        eta_lead=.01, eta_device=0, scale=dict(omega0=2., hbar=3.))
    target = dict(full_dofs=3, active_dofs=3)
    report = dict(schema="negfpp.validation.fitted-scba-result.v1", passed=True, converged=True,
        failure_reason=None, stop_reason="fixed_point", seed=copy.deepcopy(seed),
        selected_record=record, evaluations=[dict(record=record, accepted_as_anchor=True,
            physical_checks=record["physical_checks"])], physics=physics, target=target)
    manifest = dict(schema="negfpp.validation.fitted-selected-final.v1", seed=copy.deepcopy(seed),
        sigma_input_evaluated=True, mapped_sigma_dyson_evaluated=False, old_history_imported=False,
        all_original_six_passed=True, original_physical_guards_passed=True, selected_record=record,
        physics=physics, target=target, selected_new_native_index=2, feedback_dofs=[0, 1, 2], full_device_dofs=3,
        artifacts={"sigma_input_" + component + ".npy": dict(sha256=bundle.document["arrays"][component]["sha256"],
            bytes=bundle.document["arrays"][component]["bytes"]) for component in ["retarded", "statistical", "spectral"]})
    for name, value in [("fresh_validation", report), ("fresh_validation_manifest", manifest)]:
        path = bundle.path.parent / (name + ".json")
        sha = _dump(path, value)
        bundle.document["source"]["evidence"][name] = dict(file=path.name, sha256=sha, bytes=path.stat().st_size)
    _update(bundle)
    load_initial_guess(bundle.descriptor, **bundle.kwargs)
    report["seed"]["manifest_sha256"] = "f" * 64
    path = bundle.path.parent / "fresh_validation.json"
    pin = bundle.document["source"]["evidence"]["fresh_validation"]
    pin.update(sha256=_dump(path, report), bytes=path.stat().st_size)
    _update(bundle)
    with pytest.raises(ValueError, match="not bound to this original"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)


@pytest.mark.parametrize("source,target", [([4, 1, 1], [8, 8, 1]), ([2, 2, 1], [4, 4, 1]), ([8, 8, 1], [2, 2, 1])])
def test_periodic_convex_rows_commute_with_q_inversion_and_preserve_psd(source, target):
    rows = periodic_linear_weights(source, target)
    sm, tm = UniformBZMesh(tuple(source)), UniformBZMesh(tuple(target))
    matrices = [np.array([[1 + i, .1j], [-.1j, 2 + i]]) for i in range(sm.n_qpoints)]
    for index, row in enumerate(rows):
        assert sum(weight for _, weight in row) == 1
        assert all(weight > 0 for _, weight in row)
        opposite = dict(rows[tm.negative_index(index)])
        assert opposite == {sm.negative_index(origin): weight for origin, weight in row}
        combined = sum(matrices[origin] * weight for origin, weight in row)
        assert np.linalg.eigvalsh(combined).min() >= 0
    if source == [8, 8, 1]:
        assert [row[0][0] for row in rows] == [0, 4, 32, 36]
        assert all(len(row) == 1 and row[0][1] == 1 for row in rows)


def test_hdf_payload_is_bound_to_original_input_hashes(bundle):
    import h5py
    record = bundle.state["record"]
    checks = record["physical_checks"]
    currents = {"left": 1., "right": -1., "collision_power": 0.}
    identity = bundle.state["original_identity"]
    fields = bundle.path.parent / "original.hdf5"
    payload_sha = hashlib.sha256(bundle.source.tobytes()).hexdigest()
    with h5py.File(fields, "w") as handle:
        handle.attrs.update(schema="negfpy.materialTest.converged-self-energy.v2", complete=True,
            all_original_six_passed=True, mapped_image_has_been_Dyson_evaluated=False,
            selected_evaluated_input_index=2, full_device_dofs=3, grid_step=.1,
            source_json=json.dumps(dict(original_record=record, physical_checks=checks, evaluated_input_currents=currents)))
        for name, values in dict(active_dof_indices=[0, 1, 2], mesh_shape=[2, 2, 1],
                negative_q_indices=[0, 1, 2, 3], mesh_addresses=bundle.kwargs["mesh"].addresses,
                grid_values=[-.1, 0, .1]).items():
            handle.create_dataset(name, data=values)
        for role in ("selected_evaluated_input", "separate_unevaluated_mapped_image"):
            group = handle.create_group(role)
            group.attrs.update(role=role, Dyson_evaluated=(role == "selected_evaluated_input"))
            for component in ("retarded", "statistical", "spectral"):
                data = group.create_dataset(component, data=bundle.source)
                data.attrs["sha256_float64_complex_c_order"] = payload_sha
    source = dict(schema="negfpy.materialTest.scba-result.v2", complete=True, passed=True,
        converged=True, failure_reason=None, stop_reason="fixed_point", active_map=2,
        records=[dict(record=record, physical_checks=checks, currents_w=currents)], selected_current_w=currents,
        original_identity=bundle.state["original_identity"],
        input_metadata=dict(geometry_sha256=identity["geometry_sha256"], energy_identity_sha256=identity["energy_identity_sha256"],
            physical_contract=dict(temperatures_kelvin=[305, 295], coupling_g=1, omega0_rad_s=2., hbar_j_s=3., eta_lead=.01, eta_device=0),
            geometry_descriptor=dict(device_cells=1, center_layers=[0])),
        self_energy=dict(all_original_six_passed=True, sha256=hashlib.sha256(fields.read_bytes()).hexdigest(),
            bytes=fields.stat().st_size, components={
            role + "/" + name: hashlib.sha256(bundle.source.tobytes()).hexdigest()
            for role in ("selected_evaluated_input", "separate_unevaluated_mapped_image")
            for name in ("retarded", "statistical", "spectral")}))
    bundle.document["source"]["format"] = "native_hdf_result"
    bundle.document["source"]["evidence"]["fields"] = dict(file=fields.name,
        sha256=source["self_energy"]["sha256"], bytes=fields.stat().st_size)
    pin = bundle.document["source"]["evidence"]["state"]
    pin.update(sha256=_dump(bundle.state_path, source), bytes=bundle.state_path.stat().st_size)
    _update(bundle)
    result, _ = load_initial_guess(bundle.descriptor, **bundle.kwargs)
    assert result.retarded.tobytes() == bundle.source.tobytes()
    row = bundle.document["arrays"]["retarded"]
    altered = bundle.source.copy(); altered.real += 1
    path = bundle.path.parent / row["file"]
    np.save(path, altered)
    row.update(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    _update(bundle)
    with pytest.raises(ValueError, match="original evaluated HDF INPUT"):
        load_initial_guess(bundle.descriptor, **bundle.kwargs)
