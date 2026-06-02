.PHONY: help setup setup-lock download process train reproduce temporal test consistency notebook clean figures lock dirs manuscript

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
	./venv/bin/python -m pytest tests/ -v --tb=short
	@echo "Tests complete"

consistency:
	@echo "Checking publication consistency..."
	python3 scripts/check_publication_consistency.py
	@echo "Publication consistency checks passed"

download:
	@echo "Downloading NHANES data..."
	python3 scripts/01_download_nhanes_data.py
	@echo "Download complete"

process:
	@echo "Processing and merging NHANES components..."
	python3 scripts/02_process_nhanes_data.py
	@echo "Processing complete"

train:
	@echo "Training models..."
	python3 scripts/03_train_models.py
	@echo "Training complete"

reproduce:
	@echo "Running primary-model reproduction workflow..."
	bash scripts/run_v13_primary.sh

temporal:
	@echo "Running same-source temporal validation workflow..."
	bash scripts/run_external_validation.sh

notebook:
	@echo "Launching Jupyter notebook..."
	jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb

figures:
	@echo "Generating publication figures..."
	@echo "Figure regeneration remains notebook-backed; see notebooks/00_nhanes_periodontitis_end_to_end.ipynb"

manuscript:
	@echo "Rendering manuscript if pandoc is installed..."
	python3 scripts/05_number_manuscript_lines.py
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
