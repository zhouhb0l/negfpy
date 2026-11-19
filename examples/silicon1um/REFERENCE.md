# Compact silicon donor

`reference.json` authenticates the short-device donor used by the local-FD
example. Its SHA256 is
`feb67f886337acb5b8366a689542463c90fa74c6bed4a0683174f3b660946b68`.
All file paths are relative to that manifest, including the four unchanged
material inputs shared with `../silicon24/`. The original campaign tree is
not required.

The source is the converged negfpp 1.0.0 direct-command silicon24 calculation:
24 atoms, 72 full DOFs, one complete eight-center interaction layer, and 36
active endpoint DOFs. It used Gamma only, 101 signed frequencies with solver
step 0.15, g = 0.1, and reservoir temperatures 300.5/299.5 K. The selected
state is 12, with all three self-energy and three current convergence gates
passed, unchanged original physical checks passed, and recorded actual exit
code 0. The original left current was 4.4708627069550947e-10 W.

`source-report.json`, `source-input.json`, `source-qualification.json`, and
`source-SHA256SUMS` preserve their original bytes. The compact `reference.h5`
contains only `sigma_input_evaluated/retarded` from the original fields file,
renamed `sigma_retarded`, and its original endpoint and frequency/q
coordinates. The Green functions, statistical/spectral self-energies, and
unevaluated Born-map image are omitted. No matrix entry is clipped, projected,
rescaled, refitted, or normalized. Its C-order little-endian complex128
payload SHA256 is
`17765664e4be5f7058b3261e6ac91148db3eb0852ab667c6b3c26bd4817dd80f`.

The loader checks all metadata, file hashes, original convergence gates,
state roles and actual-exit binding before reading scientific arrays. It then
checks the unchanged retarded field for passivity, signed q/frequency reality,
and acoustic translation invariance. The exported donor passed these guards;
their largest normalized errors were approximately 0.000258, 0.0000738, and
0.0000748, respectively, with tolerance ratios required to be at most one.

This compact donor supports a coarse organization and execution check. It is
not the historical 4x4-q, 601-frequency, g = 1 micron reference and does not
establish a converged silicon conductivity. Method 1 freezes and overlaps the
short-device retarded kernel; it does not evaluate a long-device Born map.
