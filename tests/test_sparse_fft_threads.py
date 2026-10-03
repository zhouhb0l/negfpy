from __future__ import annotations

import itertools
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest
from threadpoolctl import threadpool_info, threadpool_limits

from negfpy.anharmonic.cubic import cubic_lesser_greater
from negfpy.anharmonic.grid import SignedFrequencyGrid
import negfpy.anharmonic.sparse_cubic as sparse


def _case():
    entries = {}
    for orbit, value in [((0, 1, 2), 0.25), ((1, 2, 4), -0.125), ((4, 4, 4), 0.2)]:
        for index in set(itertools.permutations(orbit)):
            entries[index] = value
    ordered = sorted(entries.items())
    vertex = sparse.SparseCubicVertex(
        6, [index for index, _ in ordered], [value for _, value in ordered]
    )
    grid = SignedFrequencyGrid(step=0.125, n_positive=5)
    rng = np.random.default_rng(7304)
    shape = (grid.n_points, vertex.ndof, vertex.ndof)
    lesser = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    greater = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    return vertex, grid, lesser, greater


@pytest.mark.parametrize("workers", [2, 4, 8])
def test_threaded_slices_match_serial_and_direct_without_input_mutation(workers):
    vertex, grid, lesser, greater = _case()
    originals = lesser.copy(), greater.copy(), vertex.coefficients.copy()
    with threadpool_limits(limits=1, user_api="blas"):
        serial = cubic_lesser_greater(vertex, lesser, greater, grid, backend="fft")
        threaded = cubic_lesser_greater(
            vertex, lesser, greater, grid, backend="fft", max_workers=workers
        )
        direct = cubic_lesser_greater(vertex, lesser, greater, grid, backend="serial")
    for reference, observed, oracle in zip(serial, threaded, direct, strict=True):
        np.testing.assert_array_equal(observed, reference)
        np.testing.assert_allclose(observed, oracle, rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(observed[:, (3, 5), :], 0)
        np.testing.assert_array_equal(observed[:, :, (3, 5)], 0)
    for actual, original in zip(
        (lesser, greater, vertex.coefficients), originals, strict=True
    ):
        np.testing.assert_array_equal(actual, original)


@pytest.mark.parametrize("workers", [None, 1])
def test_serial_default_does_not_create_executor(workers, monkeypatch):
    vertex, grid, lesser, greater = _case()

    def unexpected_executor(*args, **kwargs):
        raise AssertionError("serial path created a thread executor")

    monkeypatch.setattr(sparse, "ThreadPoolExecutor", unexpected_executor)
    sparse.cubic_sparse_lesser_greater(
        vertex, lesser, greater, grid, backend="fft", max_workers=workers
    )


def test_opt_in_uses_concurrent_threads_with_one_blas_thread(monkeypatch):
    vertex, grid, lesser, greater = _case()
    original = sparse._sparse_pair_contraction
    barrier = threading.Barrier(2)
    seen = set()
    lock = threading.Lock()

    def observed(*args):
        ident = threading.get_ident()
        with lock:
            first = ident not in seen
            seen.add(ident)
        if first:
            barrier.wait(timeout=5)
        assert all(
            pool["num_threads"] == 1
            for pool in threadpool_info()
            if pool["user_api"] == "blas"
        )
        return original(*args)

    monkeypatch.setattr(sparse, "_sparse_pair_contraction", observed)
    sparse._sparse_bubble_fft(
        vertex, sparse._compiled_sparse_schedule(vertex), lesser, grid, max_workers=2
    )
    assert len(seen) >= 2
    assert threading.get_ident() not in seen


@pytest.mark.parametrize(
    "workers,error",
    [
        (0, ValueError),
        (-1, ValueError),
        (9, ValueError),
        (128, ValueError),
        (True, TypeError),
        (2.5, TypeError),
    ],
)
def test_fft_workers_are_bounded(workers, error):
    vertex, grid, lesser, greater = _case()
    with pytest.raises(error, match="max_workers"):
        sparse.cubic_sparse_lesser_greater(
            vertex, lesser, greater, grid, backend="fft", max_workers=workers
        )


def test_worker_exception_reaches_caller(monkeypatch):
    vertex, grid, lesser, greater = _case()

    def failed(*args):
        raise RuntimeError("slice failed")

    monkeypatch.setattr(sparse, "_sparse_pair_contraction", failed)
    with pytest.raises(RuntimeError, match="slice failed"):
        sparse.cubic_sparse_lesser_greater(
            vertex, lesser, greater, grid, backend="fft", max_workers=2
        )


def _benchmark_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_sparse_fft_threads.py"
    )
    spec = importlib.util.spec_from_file_location("fft_thread_benchmark_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_benchmark_reports_equality_and_actual_cpu_timing():
    module = _benchmark_module()
    vertex, grid, lesser, greater = _case()
    events = []
    records = module.benchmark_cases(
        vertex, grid, lesser, greater, [1, 2], emit=events.append
    )
    assert records == events
    assert [record["workers"] for record in records] == [1, 2]
    assert records[0]["speedup_vs_serial"] == 1.0
    for record in records:
        assert record["passed"] is True
        assert record["wall_seconds"] > 0
        assert record["cpu_seconds"] > 0
        assert all(item["exactly_equal"] for item in record["components"])


def test_benchmark_timeout_preserves_partial_evidence(tmp_path, monkeypatch):
    module = _benchmark_module()
    output = tmp_path / "benchmark.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(module.__file__),
            "--parent-artifacts",
            str(tmp_path),
            "--expected-manifest-sha256",
            "a" * 64,
            "--expected-status-sha256",
            "b" * 64,
            "--wall-seconds",
            "1",
            "--output",
            str(output),
        ],
    )

    def timeout(command, **kwargs):
        assert command[-1] == "--benchmark-child"
        assert kwargs["timeout"] == 1
        raise subprocess.TimeoutExpired(
            command, 1, output=b'{"event":"inputs","passed":true}\n'
        )

    monkeypatch.setattr(module.subprocess, "run", timeout)
    assert module.main() == 2
    report = json.loads(output.read_text())
    assert report["timed_out"] is True
    assert report["passed"] is False
    assert report["events"] == [{"event": "inputs", "passed": True}]
    assert report["scientific_solve_executed"] is False
