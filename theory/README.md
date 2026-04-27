# Theory Workspace

This directory collects theory-facing material for `negfpy`.

## Layout

- `literature/`
  - Local paper copies, benchmark audits, and theory-oriented notes copied or moved from the old repo root.
  - The repo root path `literature/` is preserved as a compatibility symlink so existing scripts and tests continue to work.
- `derivations/`
  - LaTeX source for detailed theoretical derivations.
  - `sections/` contains the sectioned source files.
  - `build/` contains LaTeX build artifacts.

## Main Theory Document

The main derivation note is:

- `derivations/negf_phonon_theory_notes.tex`

The compiled PDF is written to:

- `derivations/negf_phonon_theory_notes.pdf`

## Goal

This workspace is meant to stay general enough for future material-based phonon NEGF implementations, while still recording the detailed toy-model and literature benchmarks that we use to validate the formalism step by step.
