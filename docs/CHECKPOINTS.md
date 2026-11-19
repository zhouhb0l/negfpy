# Rolling self-energy checkpoints

Add the optional top-level object to a finite-device SCBA input:

```json
"checkpoint": {"file": "self-energy-latest.h5"}
```

Run the installed `negfpp INPUT.json` command. The filename is a simple `.h5`
name inside the fresh output directory. The solver saves after every completed
physical map, including a trial rejected as an anchor. It writes a temporary
file in that directory, flushes it, then atomically replaces the previous file.
An interrupted write leaves the previous completed checkpoint intact. Reserve
disk space for the previous file and one temporary replacement.

The `negfpp.scba-checkpoint.v1` file contains the evaluated INPUT self-energy
R/C/D in native complex128, exact-zero metadata where applicable, ordered
frequency/q/active-DOF coordinates, the original input bytes and their hash,
physical model binding, completed-map ordinal, native state index, convergence
screens, physical checks and separate selected-anchor index. It contains no
Green function, unevaluated Born image or controller history. Tangent actions,
failed physical maps and interrupted maps are not completed-map checkpoints.
The scalar report records the checkpoint digest and byte count.

A checkpoint may represent a nonconverged or rejected trial. Saving it does not
accept that trial for transport. Final acceptance still requires a fresh
completed map, all original physical guards and all six convergence screens.
The final `fields.h5` remains the selected evaluated anchor, which may differ
from the latest rolling checkpoint.

To start a new invocation from the latest checkpoint, authenticate its bytes:

```json
"initial_self_energy": {
  "checkpoint": "../previous-output/self-energy-latest.h5",
  "checkpoint_sha256": "<64-character lowercase SHA-256>"
}
```

The physical model, material inputs, device, endpoint ordering, mesh/frequency
grid, units, reservoirs, broadening and coupling must match. Controls and output
paths may change. Only evaluated INPUT channels are read, with fresh history,
new map counts and fresh physical/convergence checks. Use a new output folder.

# Explicit historical initial guesses

Saved native campaigns can supply authenticated evaluated INPUT fields through
an explicit portable manifest. The release reads this data directly; it never
imports campaign Python modules or executes a native driver. This interface
requires `negfpp.historical-initial-self-energy.v1` provenance and field roles;
an arbitrary NPY/HDF5 file is not an admitted source.

```json
"initial_self_energy": {
  "manifest": "../seeds/initial-guess.json",
  "manifest_sha256": "<64-character lowercase SHA-256>",
  "mode": "same_model_closest_q_temperature_guess"
}
```

Only q mesh and reservoir temperatures may differ. Material parameters,
harmonic/cubic input identities, device geometry, complete endpoint support,
ordered active DOFs, signed frequencies, normalization, broadening and coupling
must agree. Source evidence and original field hashes remain authenticated.
The importer copies exact matching q nodes and otherwise uses convex periodic
linear interpolation of whole channel matrices in the same DOF basis. This
commutes with q negation and introduces no DOF embedding, cropping, projection,
quadrature factor or temperature/bias amplitude rescaling. Interpolation weights
and source/target temperatures are recorded in the initialization receipt.

The source convergence does not qualify the target. Target contacts and Born
maps are rebuilt and must pass all original physical and six convergence
screens. A mesh migration is only an initial guess and does not establish mesh
convergence or physical accuracy. Source files are never overwritten.
