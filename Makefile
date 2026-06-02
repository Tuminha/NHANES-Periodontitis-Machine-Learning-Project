SHELL := /bin/bash
PYTHON ?= ./venv/bin/python

.PHONY: help setup setup-lock download process train reproduce temporal test consistency verify-submission reproduce-full notebook clean figures lock dirs manuscript

help:
	@echo "NHANES Periodontitis ML Project - Make Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup        - Create venv and install dependencies from requirements.txt"
	@echo "  make setup-lock   - Create venv and install dependencies from requirements.lock.txt"
	@echo "  make test         - Run pytest unit tests"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download     - Download NHANES XPT files"
	@echo "  make process      - Merge components and apply CDC/AAP labels"
	@echo ""
	@echo "Modeling:"
	@echo "  make train        - Train all models"
	@echo "  make reproduce    - Run primary-model reproduction workflow"
	@echo "  make temporal     - Run same-source temporal validation workflow"
	@echo "  make notebook     - Launch Jupyter notebook"
	@echo ""
	@echo "Publication:"
	@echo "  make consistency  - Check result and manuscript consistency"
	@echo "  make verify-submission - Run lightweight submission-readiness checks"
	@echo "  make reproduce-full - Run full local reproduction workflow"
	@echo "  make manuscript   - Render PDF manuscript if pandoc is installed"
	@echo "  make figures      - Generate publication figures from saved results"
	@echo ""
	@echo "Reproducibility:"
	@echo "  make lock         - Save current venv package versions to requirements.lock.txt"
	@echo "  make clean        - Remove generated local artifacts"

setup:
	@echo "Setting up Python environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source venv/bin/activate"

setup-lock:
	@echo "Setting up Python environment from lock file..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.lock.txt
	@echo "Setup complete. Activate with: source venv/bin/activate"

test:
	@echo "Running pytest unit tests..."
	$(PYTHON) -m pytest tests/ -v --tb=short
	@echo "Tests complete"

consistency:
	@echo "Checking publication consistency..."
	$(PYTHON) scripts/check_publication_consistency.py
	@echo "Publication consistency checks passed"

verify-submission:
	@echo "Running submission-readiness checks..."
	$(MAKE) test
	$(MAKE) consistency
	$(PYTHON) scripts/verify_submission.py
	$(PYTHON) scripts/05_number_manuscript_lines.py
	@echo "Submission-readiness checks complete"

download:
	@echo "Downloading NHANES data..."
	$(PYTHON) scripts/01_download_nhanes_data.py
	@echo "Download complete"

process:
	@echo "Processing and merging NHANES components..."
	$(PYTHON) scripts/02_process_nhanes_data.py
	@echo "Processing complete"

train:
	@echo "Training models..."
	$(PYTHON) scripts/03_train_models.py
	@echo "Training complete"

reproduce:
	@echo "Running primary-model reproduction workflow..."
	$(PYTHON) scripts/reproduce_v13_primary.py

temporal:
	@echo "Running same-source temporal validation workflow..."
	$(PYTHON) scripts/run_temporal_validation.py

reproduce-full:
	@mkdir -p logs
	@set -euo pipefail; \
	LOG="logs/full_reproduction_$$(date -u +%Y%m%dT%H%M%SZ).log"; \
	echo "Writing full reproduction log to $$LOG"; \
	{ \
		$(MAKE) download; \
		$(MAKE) process; \
		$(MAKE) reproduce; \
		$(MAKE) temporal; \
		$(PYTHON) scripts/04_publication_analyses.py \
			--input data/processed/publication_predictions.parquet \
			--feature-cols age bmi waist_cm waist_height height_cm systolic_bp diastolic_bp glucose triglycerides hdl; \
		$(MAKE) consistency; \
		$(MAKE) verify-submission; \
	} 2>&1 | tee "$$LOG"

notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb

figures:
	@echo "Generating publication figures..."
	@echo "Figure regeneration remains notebook-backed; see notebooks/00_nhanes_periodontitis_end_to_end.ipynb"

manuscript:
	@echo "Rendering manuscript if pandoc is installed..."
	$(PYTHON) scripts/05_number_manuscript_lines.py
	@if command -v pandoc >/dev/null 2>&1; then \
		mkdir -p reports; \
		pandoc docs/publication/ARTICLE_DRAFT.md \
			--number-sections \
			--pdf-engine=xelatex \
			-V geometry:margin=1in \
			-o reports/manuscript_publication_repair.pdf; \
		echo "Rendered reports/manuscript_publication_repair.pdf"; \
	else \
		echo "pandoc is not installed; manuscript source is docs/publication/ARTICLE_DRAFT.md"; \
	fi

clean:
	@echo "Cleaning generated local artifacts..."
	rm -rf models/*.pkl
	rm -rf models/*.json
	rm -rf artifacts/*.pkl
	rm -rf artifacts/*.npy
	rm -rf artifacts/*.db
	rm -rf logs/*.log
	rm -rf reports/*.pdf reports/*.html
	@echo "Clean complete"

lock:
	@echo "Freezing package versions..."
	./venv/bin/pip freeze > requirements.lock.txt
	@echo "requirements.lock.txt updated"

dirs:
	@echo "Creating project directories..."
	mkdir -p configs figures models results artifacts logs reports data/raw data/processed notebooks src tests
	@echo "Directories created"
