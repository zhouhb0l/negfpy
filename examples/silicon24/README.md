# Silicon24 example

Run the installed command from the repository root:

```sh
negfpp examples/silicon24/input.json
```

The JSON contains paths, input hashes, the device geometry, grid, reservoirs,
solver controls and resource limits. Paths resolve relative to the JSON file.
Results are written below `generated/silicon24` in the repository.

This example restores the existing saved cubic-energy coefficients and uses
the supplied harmonic force constants. It performs no fit and reads no
historical Green functions, self-energies or checkpoints. `model.json` retains
the original fit report, fit geometry and material identities;
`input-provenance.json` records the selected sources, hashes and transformation.

The device contains 24 Si atoms in three layers, eight cubic-energy centers,
and 12 active endpoint atoms. It uses one Gamma wavevector, 101 signed
frequencies, 300.5/299.5 K reservoirs and coupling `g=0.10`. The previous run
converged in 12 maps, about 21 seconds, with a left heat current of
`4.4708627069550947e-10 W`. Timing depends on the machine.

The saved cubic model has a relative bulk-FC3 Frobenius fit residual of
`0.021414695219655902`; its omitted remainder is recorded in the fit report.
The coarse grid checks execution and numerical solver behavior. It does not
establish frequency, wavevector or broadening convergence, or physical
interface accuracy.

A successful calculation must report convergence with all three original
unmixed self-energy screens, all three current screens and the physical
guards passing. A time or map limit must be reported as an incomplete result.
