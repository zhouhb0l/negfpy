from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

import negfpy.anharmonic.solver as solver_module
from negfpy.anharmonic import (
    CubicSCBAProblem,
    HarmonicReservoir,
    SCBAConfig,
    SignedFrequencyGrid,
    solve_one_shot_born,
    solve_scba,
)
from negfpy.anharmonic._provenance import problem_input_hashes
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain


def _bounded_code_provenance() -> dict[str, str | bool]:
    """Avoid filesystem/git discovery in bounded numerical unit tests."""

    return {
        "package_version": "2.0.0.dev0",
        "declared_package_version": "2.0.0.dev0",
        "installed_package_version": "2.0.0.dev0",
        "package_version_consistent": True,
        "source_tree_sha256": "b" * 64,
        "git_revision": "a" * 40,
        "git_dirty": True,
    }


@pytest.fixture(autouse=True)
def _fixed_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        solver_module,
        "code_provenance",
        _bounded_code_provenance,
    )


def _problem_pair(*, cubic: float) -> tuple[CubicSCBAProblem, CubicSCBAProblem]:
    device, lead, dense = pinned_scalar_chain(
        1,
        pinning=1.0,
        cubic=cubic,
    )
    common: dict[str, Any] = {
        "device": device,
        "left": HarmonicReservoir(lead, 0.8),
        "right": HarmonicReservoir(lead, 0.6),
        "grid": SignedFrequencyGrid(step=0.35, n_positive=7),
    }
    dense_problem = CubicSCBAProblem(vertex=dense, **common)
    sparse_problem = CubicSCBAProblem(
        vertex=SparseCubicVertex.from_dense(dense),
        **common,
    )
    return dense_problem, sparse_problem


def _config(*, backend: str = "serial") -> SCBAConfig:
    return SCBAConfig(
        eta_lead=1.0e-4,
        max_iterations=50,
        min_iterations=3,
        consecutive_passes=2,
        mixing=0.5,
        sigma_rtol=1.0e-5,
        green_rtol=1.0e-5,
        identity_rtol=1.0e-6,
        current_rtol=1.0e-3,
        current_atol=1.0e-8,
        backend=backend,  # type: ignore[arg-type]
        max_workers=2 if backend == "threaded" else None,
    )


def _assert_full_fields_equal(first: Any, second: Any) -> None:
    for spectrum_name in ("green", "sigma_pp"):
        first_spectrum = getattr(first, spectrum_name)
        second_spectrum = getattr(second, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_allclose(
                getattr(second_spectrum, component),
                getattr(first_spectrum, component),
                rtol=1.0e-10,
                atol=1.0e-12,
            )


def _assert_currents_equal(first: Any, second: Any) -> None:
    for name in (
        "left",
        "right",
        "collision_power",
        "contact_balance",
        "full_balance",
    ):
        np.testing.assert_allclose(
            getattr(second.currents, name),
            getattr(first.currents, name),
            rtol=1.0e-10,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        second.currents.contact_spectra,
        first.currents.contact_spectra,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        second.currents.collision_spectrum,
        first.currents.collision_spectrum,
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_dense_sparse_one_shot_born_full_fields_and_provenance() -> None:
    dense_problem, sparse_problem = _problem_pair(cubic=0.002)
    config = _config()

    dense = solve_one_shot_born(dense_problem, config)
    sparse = solve_one_shot_born(sparse_problem, config)

    _assert_full_fields_equal(dense, sparse)
    _assert_currents_equal(dense, sparse)
    assert dense.diagnostics.passed == sparse.diagnostics.passed
    assert dense.converged == sparse.converged
    assert "vertex_storage_kind" not in dense.metadata
    assert dense.metadata["certification_contract"] == "dense_v0.1"
    assert sparse.metadata["certification_contract"] == "sparse_v0.3_development"
    assert sparse.metadata["vertex_storage_kind"] == "sparse"
    assert sparse.metadata["vertex_nnz"] == sparse_problem.vertex.nnz
    assert (
        sparse.metadata["vertex_physics_sha256"]
        == sparse_problem.vertex.cubic_physics_sha256
    )
    assert (
        sparse.metadata["vertex_storage_sha256"]
        == sparse_problem.vertex.vertex_storage_sha256
    )
    hashes = problem_input_hashes(sparse_problem)
    assert hashes["cubic_vertex"] == sparse_problem.vertex.vertex_storage_sha256
    assert (
        hashes["cubic_vertex_physics"]
        == sparse_problem.vertex.cubic_physics_sha256
    )
    assert (
        hashes["cubic_vertex_storage"]
        == sparse_problem.vertex.vertex_storage_sha256
    )
    assert all(len(value) == 64 for value in hashes.values())


def test_dense_sparse_converged_scba_full_fields() -> None:
    dense_problem, sparse_problem = _problem_pair(cubic=0.002)
    config = _config()

    dense = solve_scba(dense_problem, config)
    sparse = solve_scba(sparse_problem, config)

    assert dense.converged and sparse.converged
    assert dense.diagnostics.passed and sparse.diagnostics.passed
    assert len(dense.history) == len(sparse.history)
    assert tuple(item.passed for item in dense.history) == tuple(
        item.passed for item in sparse.history
    )
    _assert_full_fields_equal(dense, sparse)
    _assert_currents_equal(dense, sparse)


def test_sparse_zero_vertex_matches_dense_exactly() -> None:
    dense_problem, sparse_problem = _problem_pair(cubic=0.0)
    dense = solve_scba(dense_problem, _config())
    sparse = solve_scba(sparse_problem, _config())

    for spectrum_name in ("green", "sigma_pp"):
        first_spectrum = getattr(dense, spectrum_name)
        second_spectrum = getattr(sparse, spectrum_name)
        for component in ("retarded", "lesser", "greater"):
            np.testing.assert_array_equal(
                getattr(second_spectrum, component),
                getattr(first_spectrum, component),
            )
    assert sparse_problem.vertex.is_zero
    assert sparse.metadata["cubic_kernel"] == "zero_vertex"
    assert sparse.metadata["vertex_nonzero"] is False
    assert sparse.metadata["vertex_max_abs"] == 0.0


@pytest.mark.parametrize(
    ("backend", "kernel", "effective", "fallback"),
    (
        ("serial", "sparse_grouped_direct", "serial", "not_requested"),
        (
            "threaded",
            "sparse_grouped_direct",
            "serial",
            "serial_direct_fallback",
        ),
        ("fft", "sparse_zero_padded_fft", "fft", "not_requested"),
    ),
)
def test_sparse_solver_backend_dispatch_is_explicit(
    backend: str,
    kernel: str,
    effective: str,
    fallback: str,
) -> None:
    _, sparse_problem = _problem_pair(cubic=0.002)
    result = solve_one_shot_born(sparse_problem, _config(backend=backend))
    direct = solve_one_shot_born(sparse_problem, _config(backend="serial"))

    _assert_full_fields_equal(direct, result)
    _assert_currents_equal(direct, result)
    assert result.metadata["backend"] == backend
    assert result.metadata["cubic_kernel"] == kernel
    assert result.metadata["effective_backend"] == effective
    assert result.metadata["worker_count"] == 1
    assert result.metadata["sparse_threaded_execution"] == fallback


def test_sparse_problem_replace_preserves_public_solver_signature() -> None:
    dense_problem, sparse_problem = _problem_pair(cubic=0.002)
    rebuilt = replace(dense_problem, vertex=sparse_problem.vertex)

    assert isinstance(rebuilt, CubicSCBAProblem)
    assert rebuilt.vertex is sparse_problem.vertex
    assert rebuilt.ndof == dense_problem.ndof
