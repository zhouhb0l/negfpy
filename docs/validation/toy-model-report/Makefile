LATEXMK ?= latexmk
PYTHON ?= python3
OUTDIR := output/pdf
REPORT := report.tex
CONFIG := release_config.tex
FINAL_PDF := $(OUTDIR)/NEGFPy_toy_model_validation_report.pdf

.PHONY: all pdf test

all: pdf

test:
	$(PYTHON) -m unittest -v test_generate_release_config.py

pdf: $(CONFIG)
	mkdir -p $(OUTDIR)
	$(LATEXMK) -pdf -halt-on-error -interaction=nonstopmode \
		-outdir=$(OUTDIR) $(REPORT)
	cp $(OUTDIR)/report.pdf $(FINAL_PDF)

$(CONFIG):
	@echo "Missing $(CONFIG); generate it from verified remote artifacts." >&2
	@false
