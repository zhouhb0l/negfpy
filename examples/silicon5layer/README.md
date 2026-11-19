# Five-layer silicon example

Run the installed command from the repository root:

```sh
negfpp examples/silicon5layer/input.json
```

This device has **five full layers: 40 Si atoms and 120 harmonic DOF**.
The three complete interacting center layers, `[1,2,3]`, contain 24 energy
centers. Their expected endpoint support is 28 atoms (84 DOF); the existing
complete-center builder must verify the actual support and retain every
required derivative.

The JSON references the unchanged IFC, crystal and saved cubic-model files
in `../silicon24/`. The files are shared between examples to avoid duplicate
input payloads. Paths resolve relative to this JSON, and its input hashes
authenticate all four selected files. Outputs are written below
`generated/silicon5layer-1.0.1`.

`reference-seeded` uses a fresh reference for this same five-layer problem,
then applies the existing exact reference/increment SCBA and bounded analytic
feedback. It does not load fields from the three-layer example or another
historical calculation.

The grid remains Gamma-only with 101 signed frequencies, coupling `g=0.10`,
and reservoirs at 300.5/299.5 K. Limits are 18 nonlinear attempts, including
rejected trials, six tangent directions, four workers and a shared 900-second
time-admission budget. An already running numerical call is not interrupted.
A budget limit is an incomplete result, and must not be reported as convergence.

No previous five-layer current or successful convergence is asserted by this
example. The model retains its documented relative bulk-FC3 fit residual
`0.021414695219655902`. This coarse check does not establish frequency,
wavevector or broadening convergence, or physical-interface accuracy.
