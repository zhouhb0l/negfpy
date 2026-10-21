"""One-call-per-configuration dense contraction-slice timing; no material solve."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_info

from negfpy.anharmonic.transverse_packed import (
    contract_packed_fft_bins,
    plan_packed_born_fft_dimensions,
)


def array_hash(value):
    contiguous = np.ascontiguousarray(value)
    return hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest()


def fixture(active_dofs, bins, seed, coupling):
    """Dense rank-three polynomial; execution does not exploit its factorization."""
    rng = np.random.default_rng(seed)
    factors = rng.normal(size=(3, active_dofs))
    factors /= np.linalg.norm(factors, axis=1)[:, None]
    strengths = coupling * np.array([0.8, -0.5, 0.3])
    vertex = np.einsum("r,ra,rc,rd->acd", strengths, factors, factors, factors).astype(complex)
    shape = (bins, active_dofs, active_dofs)
    left = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    right = rng.normal(size=shape) + 1j*rng.normal(size=shape)
    # Independent factorized oracle: only rank-by-rank matrix elements, not
    # the three dense contractions used by the executor being timed.
    expected = np.zeros_like(left)
    for k in range(bins):
        for r in range(3):
            for s in range(3):
                amplitude = strengths[r] * strengths[s]
                amplitude *= factors[r] @ left[k] @ factors[s]
                amplitude *= factors[r] @ right[k] @ factors[s]
                expected[k] += amplitude*np.outer(factors[r], factors[s])
    for value in (vertex, left, right, expected):
        value.setflags(write=False)
    return vertex, left, right, expected


def run(profile, result, emit):
    cfg = profile["benchmark"]
    active, bins = cfg["active_dofs"], cfg["bins"]
    if type(active) is not int or not 1 <= active <= 120:
        raise ValueError("active_dofs must be an integer in 1..120.")
    if type(bins) is not int or not 1 <= bins <= 8:
        raise ValueError("bins must be an integer in 1..8.")
    if cfg["g"] != 0.03 or len(cfg["cases"]) != 2:
        raise ValueError("Fixed g=.03 and exactly two distinct worker cases required.")
    if [case["workers"] for case in cfg["cases"]] != [1, min(8, bins)]:
        raise ValueError("Expected one serial case and one bounded parallel case.")
    if bins < 2:
        raise ValueError("At least two bins are needed for distinct worker cases.")
    result.update(
        source_commit=profile["source_commit"], source_tree=profile["source_tree"],
        settings=cfg, cases=[], material_inputs=False,
        scope={"dense_contraction_slice":True, "synthetic":True, "material_dofs":120,
               "full_born_map":False, "fft_transforms_timed":False, "scba":False,
               "currents":False, "material_speedup_claim":False,
               "warmups":0, "calls_per_configuration":1},
    )
    result["material_dimension_plan"] = asdict(plan_packed_born_fft_dimensions(
        4, 601, 120, external_tile=16, workers=8, bin_tile=1,
    ))
    emit("fixture_start", {"active_dofs":active, "bins":bins})
    vertex, left, right, expected = fixture(active, bins, cfg["seed"], cfg["g"])
    before = {name:array_hash(value) for name,value in
              [("vertex",vertex),("left",left),("right",right),("oracle",expected)]}
    result["input_sha256_before"] = before
    result["fixture_payload_bytes"] = sum(v.nbytes for v in (vertex,left,right,expected))
    emit("fixture_complete", {"payload_bytes":result["fixture_payload_bytes"], "input_sha256":before})
    for index,case in enumerate(cfg["cases"]):
        if case["external_tile"] != min(16,active) or case["bin_tile"] != 1:
            raise ValueError("Benchmark requires external tile min(16,A) and bin tile 1.")
        emit("case_start", {"index":index, **case})
        begin = time.perf_counter()
        output = contract_packed_fft_bins(
            vertex, left, right, **case,
            max_bytes=cfg["memory_budget_bytes"],
            max_contraction_work=cfg["max_contraction_work"],
        )
        elapsed = time.perf_counter()-begin
        difference = output-expected
        maximum = float(np.max(np.abs(difference)))
        scale = float(np.max(np.abs(expected)))
        relative_frobenius = float(np.linalg.norm(difference)/np.linalg.norm(expected))
        passed = bool(np.isfinite(output).all() and maximum <= 3e-12*scale+1e-24
                      and relative_frobenius <= 3e-12)
        record = {"index":index, **case, "active_dofs":active, "bins":bins,
                  "elapsed_seconds":elapsed, "complex_mac_terms":3*bins*active**4,
                  "maximum_absolute_difference":maximum,
                  "maximum_reference_magnitude":scale,
                  "relative_frobenius_difference":relative_frobenius,
                  "output_sha256":array_hash(output), "passed":passed}
        result["cases"].append(record)
        emit("case_complete", record)
        if not passed:
            raise ValueError(f"Contraction slice case {index} disagrees with factorized oracle.")
        del output, difference
    after = {name:array_hash(value) for name,value in
             [("vertex",vertex),("left",left),("right",right),("oracle",expected)]}
    result["input_sha256_after"] = after
    if before != after:
        raise ValueError("Benchmark inputs changed.")
    result["threadpools_after"] = threadpool_info()
    result["single_observation_serial_over_parallel_ratio"] = (
        result["cases"][0]["elapsed_seconds"]/result["cases"][1]["elapsed_seconds"]
    )
    result["passed"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    result = {"passed":False, "schema":"negfpy.transverse-packed-slice-benchmark.v1"}
    started = time.monotonic()

    def emit(event, details=None):
        record = {"event":event, "time":datetime.now(timezone.utc).isoformat(),
                  "elapsed_seconds":time.monotonic()-started, "details":details or {}}
        with (args.output.parent/"events.jsonl").open("a") as f:
            f.write(json.dumps(record,sort_keys=True,allow_nan=False)+"\n")
        print(json.dumps(record,sort_keys=True,allow_nan=False),flush=True)

    try:
        run(profile,result,emit)
    except Exception as exc:  # noqa: BLE001 - retain the failed benchmark evidence
        result["failure"] = {"type":type(exc).__name__,"message":str(exc)}
        traceback.print_exc()
        emit("benchmark_failed",result["failure"])
    result["elapsed_seconds"] = time.monotonic()-started
    result["maximum_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with args.output.open("x") as f:
        json.dump(result,f,indent=2,sort_keys=True,allow_nan=False);f.write("\n")
    emit("benchmark_closed", {"passed":result["passed"],"elapsed_seconds":result["elapsed_seconds"]})
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
