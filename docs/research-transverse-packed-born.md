# Exact-support packed and tiled transverse Born execution

The `transverse_packed` module is a separate execution path for the next material
integration step. It stores cubic coefficients only on explicitly listed active
device coordinates and returns self-energies on that same support. The existing
direct and FFT kernels, prototype solver and their limits remain unchanged.

`PackedTransverseCubicVertex(mesh, coefficients, active_dofs, full_ndof)` accepts
coefficients of shape `(Nq,Nq,A,A,A)`. The integer addresses can be in any unique
order within the full harmonic device. Reality and both generating leg-exchange
checks are inherited from `TransverseCubicVertex`; construction does not average
or repair the vertex. Empty support is permitted. Stored coefficients and
addresses are detached and read-only.

`PackedTransverseCubicVertex.from_dense(vertex)` finds the exact union of all
nonzero coordinates on all three legs, retaining arbitrarily small coefficients.
This utility still needs the caller's dense vertex and a temporary packed copy.
Direct packed construction avoids that full input, but a raw-material streaming
mapper into this representation remains subsequent work. Neither path removes
harmonic buffers, changes the interaction crop or applies coupling g implicitly.

Use `project_transverse_green(vertex, full_green)` to obtain the exact active
principal block. Use `embed_packed_self_energy(vertex, packed_array)` to restore
full device dimensions; inactive rows and columns are exactly zero. Both use
bounded row copies and explicit memory limits, with finite-value validation
before and after dtype conversion. They do not infer full-device Dyson physics
from a reduced harmonic matrix.

The execution call is:

```python
sigma_less, sigma_greater = cubic_born_self_energy_q_packed_fft(
    grid, packed_vertex, packed_green_less, packed_green_greater,
    external_tile=16, workers=8, bin_tile=1,
    max_bytes=declared_owned_array_budget,
    max_contraction_work=declared_complex_mac_budget,
)
```

Inputs and outputs have shape `(Nq,Nomega,A,A)`. The same zero-extended rectangle
convolution, exact momentum routing, conjugated second vertex and prefactor
`0.5j * grid.integration_weight / Nq` as the existing FFT reference apply. The
inverse FFT is cropped at `grid.zero_index : grid.zero_index + grid.n_points`.
There is no extra FFT or mesh normalization. The Green matrices themselves are
neither conjugated nor transposed in the mathematical contraction.

Each three-stage contraction is tiled over the first external coordinate.
Intermediate arrays have shape `(tile,A,A)`, `(tile,A,A)` and `(tile,A)`; no
rank-four coordinate tensor is created. Contour components, output q and internal
q accumulation are sequential. Independent FFT-bin chunks write disjoint output
regions, with at most the effective worker count of submitted futures. A restoring
`threadpoolctl` context sets BLAS to one thread during execution. Worker count
and bin size are explicit, so a caller can avoid nested CPU oversubscription.

`contract_packed_fft_bins(V, Gleft, Gright, ...)` exposes this same contraction
on a bounded set of transformed bins without FFT, mesh summation or prefactors.
It accepts arbitrary finite complex inputs for independent algebraic checks and
dimension-controlled timing. It does not produce a complete Born map on its own.

## Explicit resource plan

`plan_packed_born_fft` accepts a vertex and signed grid.
`plan_packed_born_fft_dimensions(Q,W,A,...)` computes the same plan without
allocating or reading material arrays. With FFT length `L`, external tile `t`
and effective workers `P`, the main additional payloads are:

- Both returned arrays: `2 * 16 * Q * W * A**2` bytes.
- One all-q transformed Green component: `16 * Q * L * A**2` bytes.
- One output-q transform: `16 * L * A**2` bytes.
- Shared conjugate vertex: `16 * A**3` bytes.
- Worker buffers: `16 * P * (2*t*A**2 + t*A)` bytes.

The peak combines both outputs and the all-q transform with the larger of
contraction workspace and the two coexisting single-q arrays needed for inverse
FFT, plus auxiliary routing/masks. `max_bytes` bounds this additional owned
numeric payload as a whole. Caller-owned vertex and input Greens are reported
separately. Vertex bytes include its address array; nominal input Green bytes
assume complex128. Construction/projection/embedding have their own bounds.
FFT/BLAS internal scratch and Python objects are excluded, so this is not an RSS
guarantee. Full-device input Greens held by an upstream solver also remain live.

Work is counted as complex multiply-accumulate terms, not floating-point
operations or a runtime estimate. A slice costs `3 * bins * A**4`; both complete
contour components cost `6 * Q**2 * L * A**4`. For Q=4, W=601, A=120 and L=2048,
the complete map has **40,768,634,880,000** such terms. Packing reduces the cubic
payload from about 2.403 GiB at D=216 to 0.412 GiB at A=120, but does not remove
the fourth-power contraction cost.

## Bounded verification and timing

New focused tests compare packed execution with the existing direct and FFT
kernels on small one-q, complex four-q and two-by-two meshes. They exercise
nonordered embedding, exact-zero inactive entries, tiny nonzero support,
frequency-edge zero extension and mesh weight. A factorized complex polynomial
provides a separate oracle for the tiled slice; input/resource guards and worker
thread restoration are checked separately. Historical suites are not rerun merely
for packaging.

All 38 new checks passed on the first run in 1.70 seconds on 21 October 2026,
with no failures, errors or skips. Production, driver and test hashes stayed
unchanged across execution. The new small driver smoke case also passed. Ruff
and whitespace checks pass. Evidence is in the workspace report directory
`multiq-packed-born-development-20261021`; no material arrays were read locally.

`scripts/benchmark_transverse_packed.py` prepares a new synthetic dense
120-coordinate, eight-bin slice with an independent rank-three polynomial oracle.
It times each of two worker configurations once (one and eight workers), without
warm-up or repetition. The dense executor does not exploit the factorization.
This measures slice execution at the material active dimension; it does not
measure FFT, full q summation, real IFC data, a complete material map or SCBA.
No extrapolated material speedup or runtime claim is justified by it alone.

Full-grid contacts, blocked Dyson/validation, direct raw-material packed mapping,
FFT retarded reconstruction, restart and complete evaluated-state solver
integration remain open. Keep the accepted material g=.03, eta_lead=.0015625,
eta_device=0 and existing geometry/crop unchanged.
