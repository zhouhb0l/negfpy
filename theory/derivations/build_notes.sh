#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build negf_phonon_theory_notes.tex >/tmp/negfpy_theory_pdflatex_1.log
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build negf_phonon_theory_notes.tex >/tmp/negfpy_theory_pdflatex_2.log
cp build/negf_phonon_theory_notes.pdf ./negf_phonon_theory_notes.pdf
