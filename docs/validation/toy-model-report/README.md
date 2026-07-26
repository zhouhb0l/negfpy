# NEGFPy anharmonic toy-model validation report

This directory contains the maintained, fail-closed LaTeX source for the
NEGFPy 0.1.0-rc1 harmonic-plus-anharmonic toy validation report. It contains
no campaign staging, rendered figures, caches, or compiled PDFs.

The report has no default commit or artifact path. It requires a generated
`release_config.tex`, an exact imported production staging tree, three
production figures, and the separately labeled historical Wang/Mingo figure.
Missing or noncertifying evidence raises a named `NEGFPy toy-gate report`
error before the manuscript body is processed.

## Generate the release configuration

`generate_release_config.py` is standard-library only. It consumes the
checksum-complete fetched artifact root, the canonical hashed
`scaling-selection.json` from the complete 1/2/4/8/16 rank ladder, and the
canonical hashed release-runner result from the locked production campaign.
Four raw `SHA256SUMS` hashes must be supplied independently by the caller: the
fetched artifacts, imported staging, production figures, and historical
figures. The generator independently checks:

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
- the selected release rank, complete outcome topology and scientific digest,
  nested strict-report body, and `release_certified=true`; and
- receipt-bound exact coverage of every generated TeX row and every PDF/PNG
  figure byte used by the manuscript.

Example:

```bash
python3 generate_release_config.py \
  --artifact-root /verified/artifacts \
  --expected-artifact-sha256sums-sha256 <external-raw-sha256> \
  --scaling-selection-relative mpi-scaling/scaling-selection.json \
  --release-runner-relative mpi-release/runner-results/release.json \
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
`release_config.tex` atomically. Do not hand-edit it.
`release_config.example.tex` is only a macro inventory; its
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
