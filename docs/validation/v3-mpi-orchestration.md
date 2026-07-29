# V3 bounded-sampling MPI orchestration contract

The bounded-v2 scientific sample manifest deliberately remains byte-identical
to the serial builder. MPI provenance is bound through a content-addressed
adjacent sidecar named

```text
<samples.json>.mpi-orchestration.<payload-sha256>.json
```

The sidecar binds the scientific-manifest SHA256, complete campaign and source
identity, the two task manifests, all immutable task artifacts and checkpoints,
all execution records present at publication, and the selected execution from
each phase. It also pins the one-complete-target-\(q\)-slab task boundary,
one-thread worker contract, canonical rank-zero merge, and the absence of
floating-point MPI reductions. The contract truthfully records the one
scientific MPI message outside task scheduling: after the maximum phase, rank
zero broadcasts the canonical global-maximum promoted block to all ranks for
the reservoir phase. This is a broadcast, not a floating-point reduction.

Release consumers must be given the sample manifest, sidecar, and preserved
task run directory. Before accepting a bounded-v2 MPI sample set, call:

```python
from negfpy.anharmonic._v3_sample_manifest_mpi import (
    validate_v3_mpi_orchestration_evidence,
)

evidence = validate_v3_mpi_orchestration_evidence(
    sample_manifest_path,
    orchestration_path,
    run_directory,
    require_authenticated_source=True,
    require_mpi=True,
    expected_source_revision=current_commit,
    expected_source_tree_sha256=current_source_tree_sha256,
    expected_environment_sha256=current_environment_sha256,
    expected_source_identity=current_source_identity,
    expected_mesh=(11, 11, 11),
    expected_target_grid_points=tuple(range(11**3)),
    expected_target_bands=tuple(range(number_of_bands)),
)
```

The helper reparses and rehashes every recorded file, rejects missing or extra
task files, symlinks, and path traversal, validates both task/checkpoint
graphs, proves that every accepted artifact originated from an earlier fresh,
non-orphan worker-MPI assignment (including no-op restarts), verifies one
physical core per numerical rank with disjoint rank bindings, and repeats the
canonical maximum/reservoir merge to reproduce the scientific samples.
Authenticated validation fails closed unless the consumer supplies the exact
candidate commit, source tree, environment, full source identity, complete
mesh, every target q point, and every target band. The consuming strict report
must record
`evidence.file_sha256`, `evidence.payload_sha256`,
`evidence.campaign_sha256`, `evidence.source_identity_sha256`,
`evidence.physical_core_binding_passed`, and
`evidence.mpi_origin_lineage_passed`.

Historical serial v1 manifests do not use this contract. A bounded-v2 release
run must not be accepted from the scientific JSON alone.
