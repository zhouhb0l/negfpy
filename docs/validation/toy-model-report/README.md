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
canonical hashed `scaling-selection.json` from the complete 1/2/4/8/16 rank
ladder and the canonical hashed release-runner result from the locked
production campaign. It independently checks:

- identical source revision, source-tree hash, and environment hash;
- all five fresh rank measurements, exact timing selection, and canonical
  physics equality;
- one distinct physical core per rank, equal socket distribution for even
  multi-rank launches, and all BLAS/OpenMP thread limits fixed to one;
- live `/proc/meminfo` `MemAvailable` records and the aggregate 70% RAM
  limit;
- the selected release rank, complete 28-task topology, and
  `release_certified=true`; and
- the presence of every generated TeX row and required PDF figure.

Example:

```bash
python3 generate_release_config.py \
  --scaling-selection /verified/artifacts/mpi-scaling/scaling-selection.json \
  --release-runner /verified/artifacts/mpi-release-runner-result.json \
  --staging-dir production-staging-<exact-commit> \
  --production-figure-dir production-figures-<exact-commit> \
  --historical-figure-dir historical-figures \
  --report-date "26 July 2026" \
  --output release_config.tex
```

The three directory arguments are normalized paths relative to this report
directory. The generator writes `release_config.tex` atomically. Do not
hand-edit it. `release_config.example.tex` is only a macro inventory; its
false gates are deliberately noncompilable.

## Imported evidence

The configured staging tree must contain generated rows for the harmonic
preflight, all 13 open-system runs, all 13 SCBA outcomes, all 8 comparisons,
conservation, scoped passivity, weak coupling, cost/accuracy, and provenance.
The production figure tree must contain:

- `figure_open_oracle_born.pdf`;
- `figure_scba_validation.pdf`; and
- `figure_passivity_backend_timing.pdf`.

The historical figure tree contains
`historical_paper_benchmarks.pdf`. It remains noncertifying. The historical
820d402 campaign is retained in the narrative as prior evidence only; it is
never substituted for the exact current candidate.

## Build and checks

Run the focused config tests without project dependencies:

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
