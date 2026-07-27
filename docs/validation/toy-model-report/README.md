# NEGFPy anharmonic toy-model validation report

This directory contains the maintained, fail-closed LaTeX source for the
NEGFPy 0.1.0-rc1 harmonic-plus-anharmonic toy validation report. It contains
no campaign staging, rendered figures, caches, or compiled PDFs.

The report has no default commit or artifact path. It requires a generated
`release_config.tex`, an exact imported production staging tree, three
production figures, and the separately labeled historical Wang/Mingo figure.
Missing or noncertifying evidence raises a named `NEGFPy toy-gate report`
error before the manuscript body is processed.

## Reproducible artifact pipeline

The report transformations are maintained in `scripts/` and execute no
physics. They are fail-closed, atomic consumers of an already fetched remote
publication:

1. `import_production_toy_gate.py` accepts only the dc60+ remote
   `current-attempt.json` publication whose state is `completed`. It requires
   the selected attempt to be the newest immutable attempt, permits an exact
   retry to reuse a checksum-bound scaling selection from a prior attempt,
   and reopens the status, report, launch intent/receipt and logs, runner result,
   execution manifest, campaign manifest, report facts, and checkpoints.
2. The importer preserves the raw source `SHA256SUMS`, fetch receipt, current
   pointer, and selected publication. It also stages a deterministic
   `source/fetched-artifact-index.json` containing the path, byte count, and
   SHA256 of every fetched artifact. Large HDF5 files remain in the externally
   anchored fetched tree rather than being duplicated.
3. `render_production_toy_gate_figures.py` re-derives every plotted CSV and
   manuscript TeX input from the raw strict report, then emits three native
   vector-PDF/450-DPI-RGB-PNG pairs.
4. `render_historical_paper_benchmarks.py` consumes only the five pinned CSVs
   in `historical-data/`. It emits the explicitly noncertifying Wang/Mingo
   figure and `historical_paper_benchmarks_metrics.json`, which binds the
   recomputed numerical metrics, selected input hashes, and renderer identity.

Every importer/renderer receipt records an explicit implementation version
and SHA256 of its source file. Every output directory contains an exhaustive,
path-sorted `SHA256SUMS` and is published with no-replace semantics.

Typical commands after `remote_heavy_cli.py fetch` are:

```bash
python3 scripts/import_production_toy_gate.py \
  /verified/artifacts production-staging-<exact-commit> \
  --expected-commit <exact-commit> \
  --expected-environment-lock-sha256 <external-lock-sha256> \
  --expected-fetch-receipt-sha256 <external-fetch-receipt-sha256> \
  --expected-sha256sums-sha256 <external-fetched-SHA256SUMS-sha256>

python3 scripts/render_production_toy_gate_figures.py \
  production-staging-<exact-commit> production-figures-<exact-commit> \
  --expected-import-receipt-sha256 <external-import-receipt-sha256>

python3 scripts/render_historical_paper_benchmarks.py \
  --data-dir historical-data \
  --benchmark-manifest historical-data/benchmark_manifest.json \
  --expected-benchmark-manifest-sha256 <external-manifest-sha256> \
  --regular-font-path /path/to/sans-regular.ttf \
  --bold-font-path /path/to/sans-bold.ttf \
  --output-dir historical-figures
```

Caller anchors must be captured outside the directory being validated; do not
derive an expected hash from a possibly modified input in the same command.
The production renderer structurally rejects raster image records and
unembedded/Type-3 fonts in its PDFs. Both renderers fully decode their PNGs
and require `(3150, 2250)`, RGB, and 450-DPI metadata. Release-config
validation also reopens the exact recorded fonts and deterministically rerenders
every PDF and PNG; manifest-rehashed substitute artwork is rejected unless its
bytes equal the committed renderer output.

Run the complete artifact tests with:

```bash
make artifact-test
```

The release-config generator reopens the historical metrics manifest,
recomputes every cited value from the exact committed CSVs, verifies the
committed renderer identity and structural PDF/PNG records, and emits the
manuscript numeric macros only after those noncertifying checks pass.

## Generate the release configuration

`generate_release_config.py` requires the `report` extra because it dynamically
loads the exact committed renderers and repeats their semantic and structural
validators. It consumes the
checksum-complete fetched artifact root and its byte-identical
`current-attempt.json` / immutable selected publication. The caller-supplied
scaling selection and release-runner paths must exactly equal the paths named
by that publication; a manifest-listed alternate or stale success is rejected.
Four raw `SHA256SUMS` hashes must be supplied independently by the caller: the
fetched artifacts, imported staging, production figures, and historical
figures. The generator independently checks:

- byte-identical current/selected publication records, a completed current
  attempt, matching successful root and attempt status (`returncode=0`), and
  every checksum-bound publication reference;
- identical source revision, source-tree hash, and environment hash;
- exhaustive path-sorted checksum coverage with no symlinks, traversal,
  special files, unlisted files, or stale entries;
- all five sibling rank envelopes and worker-result chains, including their
  campaign, 28 exact tasks, eight dependency waves, execution outcomes,
  strict reports, and canonical physics projections;
- all five fresh rank measurements, the exact timing-selection policy, and
  canonical physics-projection equality;
- one distinct physical core per rank, equal socket distribution for even
  multi-rank launches, all BLAS/OpenMP thread limits fixed to one, and the
  recorded runner-validated aggregate 70% RAM preflight;
- the selected release rank, complete outcome topology, exact per-task
  artifact-validation schema/identity/completed status, all mandatory strict
  summary gates, an empty failure list, and `release_certified=true`;
- the immutable pre-launch intent and terminal MPI launch-v2 envelope,
  deployed launcher/Python/runner identities, exact argv and Open MPI binding,
  the clean tracked/full-worktree/orchestration manifests, exhaustive environment
  prefix manifest, exact scrubbed child environment, post-run revalidation with
  `started_utc <= checked_utc <= finished_utc`,
  thread caps, stdout/stderr hashes, and equality with terminal facts; and
- receipt-bound exact coverage of every generated TeX row and every PDF/PNG
  figure byte used by the manuscript.

Example:

```bash
python3 generate_release_config.py \
  --artifact-root /verified/artifacts \
  --expected-artifact-sha256sums-sha256 <external-raw-sha256> \
  --scaling-selection-relative attempts/attempt-001/mpi-scaling/scaling-selection.json \
  --release-runner-relative attempts/attempt-001/mpi-release-runner-result.json \
  --staging-dir production-staging-<exact-commit> \
  --expected-staging-sha256sums-sha256 <external-raw-sha256> \
  --production-figure-dir production-figures-<exact-commit> \
  --expected-production-figure-sha256sums-sha256 <external-raw-sha256> \
  --historical-figure-dir historical-figures \
  --expected-historical-figure-sha256sums-sha256 <external-raw-sha256> \
  --report-date "26 July 2026" \
  --output release_config.tex
```

The three evidence-directory arguments are normalized paths relative to the
output configuration's directory; the artifact references are normalized
paths inside the caller-anchored artifact root. The generator writes
`release_config.tex` atomically. It includes exact SHA256 macros for the
current pointer, selected publication, pre-launch intent, terminal launch-v2
envelope, execution-source identity, environment-prefix manifest, and scrubbed
launch environment, in addition to report, execution, scaling, staging, and
figure anchors.
Do not hand-edit it. `release_config.example.tex` is only a macro inventory; its
false gates are deliberately noncompilable.

## Imported evidence

The configured staging tree must contain generated rows for the harmonic
preflight, all 13 open-system runs, all 13 SCBA outcomes, all 8 comparisons,
conservation, scoped passivity, weak coupling, cost/accuracy, and provenance.
The production figure tree must contain:

- `figure_open_oracle_born.{pdf,png}`;
- `figure_scba_validation.{pdf,png}`; and
- `figure_passivity_backend_timing.{pdf,png}`.

The historical figure tree contains
`historical_paper_benchmarks.{pdf,png}`. Its receipt must explicitly retain
the noncertifying historical-sensitivity status. The historical
820d402 campaign is retained in the narrative as prior evidence only; it is
never substituted for the exact current candidate.

## Build and checks

Run the focused config tests in the exact NEGFPy environment. One test imports
the installed production `production_toy_physics_sha256` implementation and
requires byte-for-byte digest equality with the report verifier's projection:

```bash
python3 -m unittest -v test_generate_release_config.py
```

Then build only from verified artifacts:

```bash
make pdf
```

The output is
`output/pdf/NEGFPy_toy_model_validation_report.pdf`. After compilation,
render every page with Poppler and inspect the images before mirroring the PDF
and figures to the external report folder.

The manuscript does not claim a same-input Wang or Mingo reproduction.
Wang \(O(T^2)\) remains a source-limited historical regression, Wang
\(O(T^4)\) remains unimplemented, and the unresolved Mingo normalization is
not used to calibrate a production prefactor. Real-material validation
remains blocked pending the pinned phono3py Si-PBEsol FC3 oracle.
