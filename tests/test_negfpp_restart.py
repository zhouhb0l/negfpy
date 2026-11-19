"""Tiny saved-output fixtures exercise restart admission, never SCBA evaluation."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from negfpp.restart import load_self_energy, resolve_restart
from negfpy.anharmonic.complete_energy import SignedFrequencyGrid, SolverScale, UniformBZMesh
from negfpy.anharmonic.transverse_mapping import _kpoints


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, allow_nan=False))
    return path


def residuals(names):
    return {name: {"ratio": .25} for name in names}


@pytest.fixture
def saved(tmp_path):
    grid = SignedFrequencyGrid(.5, 1)
    mesh = UniformBZMesh((1, 1, 1))
    scale = SolverScale(omega0=2e12, hbar=1e-34)
    vertex = SimpleNamespace(full_ndof=6, ndof=3, active_dofs=np.array([0, 1, 2]))
    support = dict(full_ndof=6, active_dofs=3, all_outer_derivatives_retained=True,
                   cropped_local_derivative_blocks=0, energy_identity_sha256="energy",
                   model_identity_sha256="model", geometry_sha256="geometry")
    info = dict(full_dofs=6, active_dofs=3, full_atoms=2, active_atoms=1,
                q_points=1, frequency_nodes=3, energy_centers=1, support=support,
                source_units={"force_constants": "eV/angstrom^3", "mass": "amu"},
                fitting_repeated=False)
    hashes = {name: {"file": name + ".dat", "sha256": str(i) * 64, "bytes": i}
              for i, name in enumerate(("fc2", "crystal", "coefficients", "metadata"), 1)}
    doc = dict(schema="negfpp.run.v1", harmonic={k: hashes[k]["file"] for k in ("fc2", "crystal")},
               model=dict(type="saved_site_energy", coefficients=hashes["coefficients"]["file"],
                          metadata=hashes["metadata"]["file"]),
               geometry=dict(transport_supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                             n_device_layers=3, interaction_layers=[1]),
               grid=dict(mesh=[1, 1, 1], step=.5, n_positive=1),
               reservoirs=dict(left_temperature_k=300.5, right_temperature_k=299.5,
                               eta_lead=.0015625, eta_device=0),
               scale=dict(omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar),
               scba=dict(g=.1, max_bytes=1024**2),
               inputs_sha256={row["file"]: row["sha256"] for row in hashes.values()})
    source = dump(tmp_path / "source-input.json", doc)
    record = dict(index=2, converged=True, sigma_passed=True, current_passed=True,
                  sigma_residuals=residuals(("retarded", "lesser", "greater")),
                  current_residuals=residuals(("contact_balance", "collision_power", "full_balance")))
    from negfpp.restart import _PHYSICAL_GROUPS
    anchor = dict(index=2, record=record, input_sigma_evaluated=True,
                  mapped_image_dyson_evaluated=False, accepted_as_anchor=True,
                  physical_checks={name: {"identity": {"ratio": .2}} for name in _PHYSICAL_GROUPS})
    # These source export/inverse diagnostics are deliberately above one.
    anchor["physical_checks"]["full_green_exports"] = {"diagnostic": {"ratio": 50}}
    report = dict(schema="negfpp.result.v1", version="1.1.0", status="converged", converged=True,
                  stop_reason="fixed_point", failure_reason=None,
                  source_input_sha256=digest(source), inputs=deepcopy(hashes),
                  model=deepcopy(info), selected_anchor=anchor, last_evaluated=deepcopy(anchor))
    report_path = dump(tmp_path / "source-report.json", report)
    fields = tmp_path / "source-fields.h5"
    values = np.arange(27, dtype=float).reshape((1, 3, 3, 3)).astype(np.complex128)
    # Preserve both real and imaginary negative-zero bit patterns.
    values.view(np.uint64).flat[0] = 1 << 63
    values.view(np.uint64).flat[1] = 1 << 63
    with h5py.File(fields, "x") as h:
        h.attrs.update(schema="negfpp.fields.v1", evaluated_state_index=2,
                       mapped_image_dyson_evaluated=False,
                       omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar)
        phases = np.asarray(_kpoints(mesh))
        for name, data, units in (
            ("omega_solver", grid.values, "dimensionless angular frequency"),
            ("omega_rad_s", grid.values * scale.omega0, "rad/s"),
            ("q_phase_rad", phases, "rad"),
            ("q_reduced", phases / (2 * np.pi), "reciprocal lattice fractions"),
        ):
            h.create_dataset(name, data=data).attrs["units"] = units
        h.create_dataset("active_device_indices", data=vertex.active_dofs)
        group = h.create_group("sigma_input_evaluated")
        group.attrs.update(exact_zero=False, units="dimensionless solver fields; see SolverScale in input.json")
        for name in ("retarded", "statistical", "spectral"):
            group.create_dataset(name, data=values)
        for name in ("green_evaluated", "sigma_born_image_unevaluated"):
            # Invalid *unused* scientific values prove these are not read/admitted.
            h.create_group(name).create_dataset("unused", data=np.array([np.nan]))
    report["fields_file"] = dict(sha256=digest(fields), bytes=fields.stat().st_size)
    dump(report_path, report)
    target_path = dump(tmp_path / "new-input.json", doc)
    spec = dict(fields=fields.name, fields_sha256=digest(fields),
                input=source.name, input_sha256=digest(source),
                report=report_path.name, report_sha256=digest(report_path))
    cfg = SimpleNamespace(document=deepcopy(doc), hashes=hashes,
                          restart=resolve_restart(target_path, spec))
    return SimpleNamespace(config=cfg, grid=grid, mesh=mesh, vertex=vertex, scale=scale,
                           info=info, fields=fields, report=report, report_path=report_path,
                           source=source, target=target_path, spec=spec, values=values)


def load(saved):
    return load_self_energy(saved.config, saved.grid, saved.mesh, saved.vertex, saved.scale, saved.info)


def repin_fields(saved):
    saved.spec["fields_sha256"] = digest(saved.fields)
    saved.report["fields_file"] = dict(sha256=digest(saved.fields), bytes=saved.fields.stat().st_size)
    dump(saved.report_path, saved.report)
    saved.spec["report_sha256"] = digest(saved.report_path)
    saved.config.restart = resolve_restart(saved.target, saved.spec)


def repin_report(saved):
    dump(saved.report_path, saved.report)
    saved.spec["report_sha256"] = digest(saved.report_path)
    saved.config.restart = resolve_restart(saved.target, saved.spec)


def test_reads_only_evaluated_input_preserving_bits_and_new_qualification(saved, monkeypatch):
    original = h5py.Dataset.__getitem__

    def guarded(dataset, key):
        assert not dataset.name.startswith(("/green_evaluated", "/sigma_born_image_unevaluated"))
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    spectrum, receipt = load(saved)
    for name in ("retarded", "statistical", "spectral"):
        array = getattr(spectrum, name)
        assert not array.flags.writeable
        assert array.tobytes() == saved.values.tobytes()
    assert receipt["new_target_evaluation_required"]
    assert not receipt["source_convergence_inherited"]
    assert not receipt["mapped_image_or_green_loaded"]


@pytest.mark.parametrize("mutate", [
    lambda spec: spec.update(unexpected="key"),
    lambda spec: spec.pop("report_sha256"),
    lambda spec: spec.update(fields="/absolute/fields.h5"),
    lambda spec: spec.update(input_sha256="INVALID"),
])
def test_strict_descriptor_rejected_before_field_read(saved, mutate):
    mutate(saved.spec)
    with pytest.raises(ValueError):
        resolve_restart(saved.target, saved.spec)


def test_whole_file_tampering_is_rejected(saved):
    saved.fields.write_bytes(saved.fields.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load(saved)


@pytest.mark.parametrize("section,key,value", [
    ("reservoirs", "left_temperature_k", 305),
    ("grid", "step", .25),
    ("scba", "g", 1),
])
def test_physical_target_change_rejected(saved, section, key, value):
    saved.config.document[section][key] = value
    with pytest.raises(ValueError, match="differs"):
        load(saved)


def test_failed_physical_proof_and_missing_six_screen_rejected(saved):
    saved.report["selected_anchor"]["physical_checks"]["input_causality"]["identity"]["ratio"] = 1.5
    repin_report(saved)
    with pytest.raises(ValueError, match="physical check failed"):
        load(saved)
    saved.report["selected_anchor"]["physical_checks"]["input_causality"]["identity"]["ratio"] = .2
    saved.report["selected_anchor"]["record"]["sigma_residuals"].pop("lesser")
    saved.report["last_evaluated"]["record"] = deepcopy(saved.report["selected_anchor"]["record"])
    repin_report(saved)
    with pytest.raises(ValueError, match="six-screen"):
        load(saved)


@pytest.mark.parametrize("problem", ["missing", "wrong-role", "dtype", "nonfinite", "index", "active-order", "units"])
def test_invalid_fields_rejected(saved, problem):
    with h5py.File(saved.fields, "r+") as h:
        g = h["sigma_input_evaluated"]
        if problem == "missing":
            del g["statistical"]
        elif problem == "wrong-role":
            del h["sigma_born_image_unevaluated"]
            h.move("sigma_input_evaluated", "sigma_born_image_unevaluated")
        elif problem == "dtype":
            del g["retarded"]
            g.create_dataset("retarded", data=saved.values.astype(">c16"))
        elif problem == "nonfinite":
            g["retarded"][0, 0, 0, 0] = np.nan
        elif problem == "index":
            h.attrs["evaluated_state_index"] = 3
        elif problem == "active-order":
            h["active_device_indices"][...] = [2, 1, 0]
        elif problem == "units":
            h["omega_solver"].attrs["units"] = "THz"
    repin_fields(saved)
    with pytest.raises(ValueError):
        load(saved)


@pytest.mark.parametrize("problem", ["external-link", "virtual", "external-storage"])
def test_nonlocal_hdf_storage_rejected_even_in_unused_group(saved, problem):
    with h5py.File(saved.fields, "r+") as h:
        if problem == "external-link":
            h["unused_foreign"] = h5py.ExternalLink("missing.h5", "/field")
        elif problem == "virtual":
            layout = h5py.VirtualLayout(shape=(1,), dtype="f8")
            layout[:] = h5py.VirtualSource("missing.h5", "field", shape=(1,))
            h.create_virtual_dataset("unused_virtual", layout)
        else:
            h.create_dataset("unused_external", shape=(1,), dtype="f8",
                             external=[("missing.raw", 0, h5py.h5f.UNLIMITED)])
    repin_fields(saved)
    with pytest.raises(ValueError, match="link|virtual or external"):
        load(saved)


def test_seed_payload_memory_admission(saved):
    saved.config.document["scba"]["max_bytes"] = 1
    with pytest.raises(ValueError, match="payload exceeds"):
        load(saved)


def test_qualified_exact_zero_becomes_explicit_readonly_channels(saved):
    for row in (saved.report["selected_anchor"], saved.report["last_evaluated"]):
        row["index"] = row["record"]["index"] = 1
        row["physical_checks"].pop("input_sigma")
        row["physical_checks"].pop("input_causality")
    repin_report(saved)
    with h5py.File(saved.fields, "r+") as h:
        h.attrs["evaluated_state_index"] = 1
        g = h["sigma_input_evaluated"]
        for name in list(g):
            del g[name]
        g.attrs["exact_zero"] = True
    repin_fields(saved)
    spectrum, receipt = load(saved)
    assert receipt["source_exact_zero"]
    for name in ("retarded", "statistical", "spectral"):
        assert not np.any(getattr(spectrum, name))
        assert not getattr(spectrum, name).flags.writeable


def test_hash_valid_file_replacement_during_load_is_rejected(saved, monkeypatch):
    import negfpp.restart as restart
    original = restart._authenticate
    seen = 0

    def replaced(path, expected):
        nonlocal seen
        seen += 1
        if seen == 4:
            # Same payload/hash, different inode: source changed while arrays lived.
            temporary = saved.fields.with_suffix(".replacement")
            temporary.write_bytes(saved.fields.read_bytes())
            temporary.replace(saved.fields)
        return original(path, expected)

    monkeypatch.setattr(restart, "_authenticate", replaced)
    with pytest.raises(ValueError, match="changed while loading"):
        load(saved)


def test_public_field_writer_loader_to_fresh_tiny_kernel_roundtrip(tmp_path):
    """Existing five-bin algebra fixture only; no material or campaign inputs."""
    from test_anharmonic_transverse_channel_feedback import _fixture, _config
    from negfpp.solver import solve_from_zero, solve_from_self_energy
    from negfpp.workflow import save_fields, scalar_state

    grid, vertex, dynamical, left, right = _fixture(nq=1)
    mesh, scale = vertex.mesh, SolverScale(omega0=1.7, hbar=.6)
    controls = _config(max_evaluations=30, mixing=.5)
    source = solve_from_zero(grid, vertex, dynamical, left, right,
                             scale=scale, g=.01, config=controls, method="linear")
    assert source.result.converged and source.result.state is not None
    assert source.result.state.sigma_input is not None
    fields = tmp_path / "fields.h5"
    save_fields(fields, source.result.state, grid, mesh, vertex, scale)
    hashes = {role: dict(file=role + ".dat", bytes=1, sha256=str(i) * 64)
              for i, role in enumerate(("fc2", "crystal", "coefficients", "metadata"), 1)}
    doc = dict(schema="negfpp.run.v1",
        model=dict(type="saved_site_energy", coefficients="coefficients.dat", metadata="metadata.dat"),
        harmonic=dict(fc2="fc2.dat", crystal="crystal.dat"),
        geometry=dict(transport_supercell_matrix=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      n_device_layers=1, interaction_layers=[0]),
        grid=dict(mesh=list(mesh.shape), step=grid.step, n_positive=grid.n_positive),
        reservoirs=dict(left_temperature_k=300.5, right_temperature_k=299.5,
                        eta_lead=.0015625, eta_device=0),
        scale=dict(omega0_rad_s=scale.omega0, hbar_j_s=scale.hbar),
        scba=dict(g=.01, max_bytes=controls.max_bytes),
        inputs_sha256={row["file"]: row["sha256"] for row in hashes.values()})
    source_input = dump(tmp_path / "source-input.json", doc)
    support = dict(full_ndof=vertex.full_ndof, active_dofs=vertex.ndof,
        all_outer_derivatives_retained=True, cropped_local_derivative_blocks=0,
        energy_identity_sha256="synthetic-unit-fixture", geometry_sha256="synthetic-unit-fixture")
    info = dict(full_dofs=vertex.full_ndof, active_dofs=vertex.ndof,
        full_atoms=vertex.full_ndof // 3, active_atoms=vertex.ndof // 3,
        q_points=mesh.n_qpoints, frequency_nodes=grid.n_points, energy_centers=1,
        source_units={"force_constants": "synthetic-unit-algebra", "mass": "synthetic-unit-algebra"},
        support=support, fitting_repeated=False)
    anchor = scalar_state(source.result.state)
    anchor["accepted_as_anchor"] = True
    source_report = dump(tmp_path / "report.json",
        dict(schema="negfpp.result.v1", version="1.1.0",
             fields_file=dict(sha256=digest(fields), bytes=fields.stat().st_size),
             status="converged", converged=True,
             stop_reason="fixed_point", failure_reason=None,
             source_input_sha256=digest(source_input), inputs=hashes, model=info,
             selected_anchor=anchor, last_evaluated=anchor))
    target_input = dump(tmp_path / "input.json", doc)
    descriptor = dict(fields=fields.name, fields_sha256=digest(fields),
                      input=source_input.name, input_sha256=digest(source_input),
                      report=source_report.name, report_sha256=digest(source_report))
    cfg = SimpleNamespace(document=doc, hashes=hashes, restart=resolve_restart(target_input, descriptor))
    guess, receipt = load_self_energy(cfg, grid, mesh, vertex, scale, info)
    for name in ("retarded", "statistical", "spectral"):
        assert getattr(guess, name).tobytes() == getattr(source.result.state.sigma_input, name).tobytes()
    starts = []
    repeated = solve_from_self_energy(grid, vertex, dynamical, left, right,
        sigma_initial=guess, scale=scale, g=.01, config=controls, method="linear",
        on_start=lambda index, sigma, proposal: starts.append(index))
    assert starts == [2]
    assert repeated.attempted == repeated.completed == 1
    assert repeated.result.converged
    assert repeated.result.state.sigma_input is not guess
    assert receipt["source_convergence_inherited"] is False


def legacy_bundle(saved, *, reserialized=False):
    """Emulate the authenticated three-file capture of the known old writer."""
    saved.report.pop("fields_file")
    saved.report["version"] = "1.0.2"
    if reserialized:
        original_raw = ("\n" + json.dumps(json.loads(saved.source.read_text()), indent=7) + "\n").encode()
        saved.report["source_input_sha256"] = hashlib.sha256(original_raw).hexdigest()
    dump(saved.report_path, saved.report)
    saved.spec["report_sha256"] = digest(saved.report_path)
    path = saved.source.parent / "SHA256SUMS"
    path.write_text("".join(saved.spec[role + "_sha256"] + "  " + Path(saved.spec[role]).name + "\n"
                            for role in ("fields", "input", "report")))
    saved.spec.update(checksums=path.name, checksums_sha256=digest(path))
    saved.config.restart = resolve_restart(saved.target, saved.spec)
    return path


def test_finite_same_shape_field_swap_cannot_borrow_converged_report(saved):
    with h5py.File(saved.fields, "r+") as h:
        h["sigma_input_evaluated/retarded"][0, 0, 0, 0] = 42 + 7j
    # Only updating the caller's field pin cannot rebind the old source result.
    saved.spec["fields_sha256"] = digest(saved.fields)
    saved.config.restart = resolve_restart(saved.target, saved.spec)
    with pytest.raises(ValueError, match="report does not bind"):
        load(saved)


def test_new_result_requires_report_fields_binding(saved):
    saved.report.pop("fields_file")
    repin_report(saved)
    with pytest.raises(ValueError, match="report.fields_file"):
        load(saved)


def test_legacy_original_checksum_file_binds_all_three_files(saved):
    legacy_bundle(saved)
    spectrum, receipt = load(saved)
    assert spectrum.retarded.tobytes() == saved.values.tobytes()
    assert receipt["source_file_binding"] == "original_legacy_checksums"
    assert receipt["legacy_checksums_authenticated"]
    assert not receipt["legacy_input_reserialized"]


def test_legacy_reserialized_saved_input_has_explicit_original_and_saved_hashes(saved):
    legacy_bundle(saved, reserialized=True)
    _, receipt = load(saved)
    assert receipt["legacy_input_reserialized"]
    assert receipt["source_original_input_sha256"] == saved.report["source_input_sha256"]
    assert receipt["selected_saved_input_sha256"] == digest(saved.source)
    assert receipt["source_original_input_sha256"] != receipt["selected_saved_input_sha256"]
    assert not receipt["source_convergence_inherited"]


def test_new_result_cannot_use_legacy_raw_input_hash_exception(saved):
    saved.report["source_input_sha256"] = "e" * 64
    repin_report(saved)
    with pytest.raises(ValueError, match="source input hash differs"):
        load(saved)


def test_old_result_without_original_checksum_file_is_rejected(saved):
    saved.report.pop("fields_file")
    saved.report["version"] = "1.0.2"
    repin_report(saved)
    with pytest.raises(ValueError, match="SHA256SUMS"):
        load(saved)


def test_original_checksum_digest_tampering_is_rejected(saved):
    path = legacy_bundle(saved)
    path.write_text(path.read_text() + "# changed\n")
    with pytest.raises(ValueError, match="SHA256"):
        load(saved)


@pytest.mark.parametrize("role", ["fields", "input", "report"])
def test_legacy_checksum_crossbinding_rejects_each_selected_file_swap(saved, role):
    legacy_bundle(saved)
    path = saved.config.restart[role]
    if role == "fields":
        with h5py.File(path, "r+") as h:
            h["sigma_input_evaluated/retarded"][0, 0, 0, 0] = 42 + 7j
    elif role == "input":
        # Even semantically equivalent reserialization must match the capture.
        path.write_text(json.dumps(json.loads(path.read_text()), indent=7))
    else:
        data = json.loads(path.read_text())
        data["name"] = "another-finite-result"
        dump(path, data)
    saved.spec[role + "_sha256"] = digest(path)
    saved.config.restart = resolve_restart(saved.target, saved.spec)
    with pytest.raises(ValueError, match="SHA256SUMS does not bind"):
        load(saved)


@pytest.mark.parametrize("keys", [{"checksums": "SHA256SUMS"}, {"checksums_sha256": "a" * 64}])
def test_checksum_descriptor_pair_is_atomic(saved, keys):
    saved.spec.update(keys)
    with pytest.raises(ValueError, match="both checksums"):
        resolve_restart(saved.target, saved.spec)


def test_native_payload_reads_directly_without_full_getitem_temporary(saved, monkeypatch):
    original_getitem = h5py.Dataset.__getitem__
    original_direct = h5py.Dataset.read_direct
    direct = []

    def guarded(dataset, key):
        assert not dataset.name.startswith("/sigma_input_evaluated/")
        return original_getitem(dataset, key)

    def read_direct(dataset, destination, *args, **kwargs):
        if dataset.name.startswith("/sigma_input_evaluated/"):
            direct.append(dataset.name)
            assert destination.dtype == np.dtype("complex128") and destination.flags.c_contiguous
        return original_direct(dataset, destination, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", guarded)
    monkeypatch.setattr(h5py.Dataset, "read_direct", read_direct)
    spectrum, _ = load(saved)
    assert len(direct) == 3
    assert spectrum.retarded.tobytes() == saved.values.tobytes()
