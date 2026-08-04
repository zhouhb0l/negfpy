from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest

from negfpy.anharmonic import CubicSCBAProblem, HarmonicReservoir, SCBAConfig, SignedFrequencyGrid
from negfpy.anharmonic._born_initial import (
    BornInitialError,
    load_born_initial_state,
    write_born_initial_state,
)
from negfpy.anharmonic.sparse_cubic import SparseCubicVertex
from negfpy.anharmonic.toy import pinned_scalar_chain


SOURCE_TREE = "a" * 64
ENVIRONMENT = "b" * 64
SOURCES = {"fc2": "c" * 64, "fc3": "d" * 64}


def _problem() -> CubicSCBAProblem:
    device, lead, dense = pinned_scalar_chain(1, pinning=1.0, cubic=0.01)
    return CubicSCBAProblem(
        device=device,
        left=HarmonicReservoir(lead, 0.8),
        right=HarmonicReservoir(lead, 0.6),
        vertex=SparseCubicVertex.from_dense(dense),
        grid=SignedFrequencyGrid(step=0.2, n_positive=3),
    )


def _config() -> SCBAConfig:
    return SCBAConfig(eta_lead=0.025, backend="fft")


def test_born_initial_round_trip_is_exact_and_bound(tmp_path: Path) -> None:
    path = tmp_path / "born.h5"
    record = write_born_initial_state(
        path, _problem(), _config(), source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
    )
    spectrum, loaded = load_born_initial_state(
        path, _problem(), _config(), source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
    )
    assert record["identity_sha256"] == loaded["identity_sha256"]
    assert record["initial_state_sha256"] == loaded["initial_state_sha256"]
    assert spectrum.shape == (7, 1, 1)
    assert np.count_nonzero(spectrum.retarded) == 0
    assert np.count_nonzero(spectrum.lesser) > 0
    assert np.count_nonzero(spectrum.greater) > 0
    with pytest.raises(FileExistsError):
        write_born_initial_state(
            path, _problem(), _config(), source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
        )


def test_born_initial_rejects_problem_config_source_and_field_tampering(
    tmp_path: Path,
) -> None:
    path = tmp_path / "born.h5"
    problem = _problem()
    config = _config()
    write_born_initial_state(
        path, problem, config, source_tree_sha256=SOURCE_TREE,
        environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
    )
    with pytest.raises(BornInitialError, match="does not match"):
        load_born_initial_state(
            path, problem, replace(config, eta_lead=0.0125),
            source_tree_sha256=SOURCE_TREE, environment_sha256=ENVIRONMENT,
            source_hashes=SOURCES,
        )
    with pytest.raises(BornInitialError, match="does not match"):
        load_born_initial_state(
            path, problem, config, source_tree_sha256="e" * 64,
            environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
        )
    with h5py.File(path, "r+") as handle:
        handle["sigma_lesser"][0, 0, 0] += 1.0j
    with pytest.raises(BornInitialError, match="field hash"):
        load_born_initial_state(
            path, problem, config, source_tree_sha256=SOURCE_TREE,
            environment_sha256=ENVIRONMENT, source_hashes=SOURCES,
        )
