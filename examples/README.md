# NEGFPP example

The [Silicon24 example](silicon24/README.md) restores the existing saved
cubic-energy model and runs a small phonon-phonon SCBA calculation:

```sh
negfpp examples/silicon24/input.json
```

It includes one JSON input and compact selected model/crystal data. Results
are written under `generated/silicon24`. The example uses a coarse grid for
execution and solver checks; it does not establish material convergence.
