# ----------------------------------------------------------------------------
# Convenience targets. Run `make help` for a listing.
# ----------------------------------------------------------------------------
.PHONY: help install smoke phase-a phase-b1 phase-b2 phase-c phase-d figures all clean-results clean-all

PYTHON        ?= python
CONFIG        ?= configs/main.yaml
SMOKE_CONFIG  ?= configs/smoke.yaml

help:
	@echo "Targets:"
	@echo "  install       pip install -e . (installs the fraud_fl package)"
	@echo "  smoke         run the CPU-friendly smoke test (~15 min)"
	@echo "  phase-a       full 10-run main campaign (~13-15 h on H100)"
	@echo "  phase-b1      synthetic sample-size ablation (~2-3 h)"
	@echo "  phase-b2      privacy-budget sensitivity (~2 h)"
	@echo "  phase-c       privacy attacks (~1.5 h)"
	@echo "  phase-d       fidelity metrics (~0.5 h, CPU)"
	@echo "  figures       regenerate every paper figure"
	@echo "  all           run smoke + every phase + figures in order"
	@echo "  clean-results delete workspace/results (keeps checkpoints)"
	@echo "  clean-all     delete the whole workspace/ directory"
	@echo ""
	@echo "Override the config with:  make phase-a CONFIG=path/to/other.yaml"

install:
	pip install -e .

smoke:
	$(PYTHON) scripts/smoke_test.py --config $(SMOKE_CONFIG)

phase-a:
	$(PYTHON) scripts/run_phase_a.py --config $(CONFIG)

phase-b1:
	$(PYTHON) scripts/run_phase_b1.py --config $(CONFIG)

phase-b2:
	$(PYTHON) scripts/run_phase_b2.py --config $(CONFIG)

phase-c:
	$(PYTHON) scripts/run_phase_c.py --config $(CONFIG)

phase-d:
	$(PYTHON) scripts/run_phase_d.py --config $(CONFIG)

figures:
	$(PYTHON) scripts/generate_figures.py --config $(CONFIG)

all: smoke phase-a phase-b1 phase-b2 phase-c phase-d figures

clean-results:
	rm -rf workspace/results workspace/figures workspace/logs

clean-all:
	rm -rf workspace
