# Outputs

The SCBA outputs below apply to `negfpp.run.v1`. The Method 1 input uses a distinct `negfpp.local-fd-result.v1` report and small thermal profiles, described in [METHOD1.md](METHOD1.md). Donor SCBA qualification and target probe closure are recorded separately.

`report.json` is the primary result. `status: converged`, `converged: true` and no `failure_reason` indicate that the selected evaluated state passed all six original SCBA screens and the solver's physical guards. Exit status is then zero. A software release version identifies the package; the bounded silicon check is not a new full scientific certification campaign (`scientific_release_certified` remains false).

`selected_anchor` and `last_evaluated` can differ when a completed trial is rejected. Their currents always belong to their **evaluated input** self-energy. The mapped Born image is an unevaluated fixed-point update. It has not been used in the reported Dyson solution. `harmonic_reference_currents_w` comes from the first, exact-zero-scattering map on the same model/grid/contacts.

`currents_w` contains `left`, `right`, `collision_power`, `contact_balance` and `full_balance`, all in watts per repeated transverse cell. Both contacts are positive into the device. Contact balance is `left + right`; full balance adds collision power. When converged at nonzero temperature bias, `finite_bias_conductance_w_per_k` is `J_left / (T_left - T_right)`, not a zero-bias derivative.

`convergence.csv` and `iterations.jsonl` retain every completed trial. Three Sigma ratios correspond to retarded/lesser/greater; three current ratios correspond to contact balance/collision power/full balance. Small mixed steps do not substitute for these unmixed residuals.

For `scba.method: "reference-seeded"`, `reference.json` and the identical
`report.json` section `reference_scba` retain the reference/increment controller
diagnostics:

- `reference_kind: "same_run_selected_anchor"` identifies references built
  from evaluated states in this invocation. It does not identify an external
  checkpoint or a previously saved calculation.
- `map_records` records each completed nonlinear map, including its direct
  baseline or exact reference/increment arithmetic and available Dyson,
  cache-consistency and increment diagnostics.
- `decisions` records trial selection, the selected/reference anchor indices,
  proposal kind and the actual full-map merit, residual and current checks.
- `proposal_screens` records candidate PSD feasibility checks, damping and
  fallback decisions before full-map evaluation. A feasible proposal is not
  an evaluated or converged state.
- `tangent_count` counts attempted signed tangent directions separately from
  the nonlinear map `counts`. `tangent_records` retains start, completion and
  skipped-admission records, with available arithmetic work, timing and
  bounded-coefficient diagnostics. Several records can describe one tangent;
  skipped directions do not consume the tangent count.

The nonlinear `counts` still distinguish attempted, completed, selected and
rejected maps. Rejected trials consume their attempt; signed tangents do not
enter `convergence.csv` as nonlinear evaluations. Neither reference arithmetic
diagnostics nor a tangent's analytic proposal establishes convergence. The
selected evaluated state must pass the same six screens and physical guards
described above.

`fields.h5` contains:

- `omega_solver`: signed dimensionless angular frequencies;
- `omega_rad_s`: physical signed angular frequencies;
- `q_reduced` and `q_phase_rad`: transverse reciprocal fractions and equivalent Bloch phases;
- `active_device_indices`: zero-based full-device coordinate addresses;
- `green_evaluated`, `sigma_input_evaluated`, `sigma_born_image_unevaluated`: independent `retarded`, `statistical` and `spectral` complex128 arrays with axes `(q,frequency,active_dof,active_dof)`.

The field arrays use the existing dimensionless solver normalization. For a field `X`, statistical `C=(X^>+X^<)/2` and spectral `D=X^>-X^<`, hence `X^<=C-D/2` and `X^>=C+D/2`. An `exact_zero` input group has no stored arrays and denotes exact zero. The Born image must not be reported as an evaluated solution.

A preparation failure can produce only the input/report/checksums. A bounded run can have no evaluated state. If acceleration stops before its first new evaluation, warm-up scalar records remain available even when no final field payload exists. Missing fields do not imply convergence. Every output directory is exclusive to one invocation.

## Rolling evaluated-input self-energy (1.2.0)

When requested, the simple HDF5 filename in `checkpoint.file` is replaced after
each completed physical map. It records the latest evaluated input, which may
be a rejected or nonconverged trial; the selected anchor remains separate.
It contains no mapped image, Green functions or controller history. The scalar
report records a checkpoint receipt with a digest and byte count. A failed or
bounded invocation retains its completed-map checkpoint without accepting its
transport. See [checkpoint semantics](CHECKPOINTS.md).
