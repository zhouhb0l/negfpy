LATEXMK ?= latexmk
PYTHON ?= python3
OUTDIR := output/pdf
REPORT := report.tex
CONFIG := release_config.tex
FINAL_PDF := $(OUTDIR)/NEGFPy_toy_model_validation_report.pdf

.PHONY: all artifact-test pdf test

all: pdf

test: artifact-test
	$(PYTHON) -m unittest -v test_generate_release_config.py

artifact-test:
	$(PYTHON) -m pytest -W error -q \
		tests/test_import_current_publication.py \
		tests/test_render_production_toy_gate_figures.py \
		tests/test_render_historical_paper_benchmarks.py

pdf: $(CONFIG)
	mkdir -p $(OUTDIR)
	$(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode \
		-outdir=$(OUTDIR) $(REPORT)
	cp $(OUTDIR)/report.pdf $(FINAL_PDF)

$(CONFIG):
	@echo "Missing $(CONFIG); generate it from verified remote artifacts." >&2
	@false
