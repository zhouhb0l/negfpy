"""Small new smoke case for the planned material-size timing driver."""

import runpy
from pathlib import Path


def test_packed_benchmark_driver_completes_distinct_worker_cases():
    driver = runpy.run_path(str(Path(__file__).parents[1]/"scripts/benchmark_transverse_packed.py"))
    config = {
        "active_dofs":4, "bins":3, "seed":481209, "g":0.03,
        "memory_budget_bytes":32*1024**2, "max_contraction_work":100_000,
        "cases":[{"workers":1,"external_tile":4,"bin_tile":1},
                 {"workers":3,"external_tile":4,"bin_tile":1}],
    }
    result, events = {}, []
    driver["run"]({"benchmark":config,"source_commit":"test","source_tree":"test"},
                  result, lambda kind,details: events.append((kind,details)))
    assert result["passed"]
    assert result["input_sha256_before"] == result["input_sha256_after"]
    assert len(result["cases"]) == 2 and all(v["passed"] for v in result["cases"])
    assert [kind for kind,_ in events] == ["fixture_start","fixture_complete",
                                         "case_start","case_complete","case_start","case_complete"]
    assert result["scope"]["full_born_map"] is False
    assert result["scope"]["material_speedup_claim"] is False
