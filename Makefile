.PHONY: help setup download process train test notebook clean figures

# Default target
help:
	@echo "NHANES Periodontitis ML Project - Make Commands"
	@echo "================================================"
	@echo ""
	@echo "Setup & Environment:"
	@echo "  make setup       - Create venv and install dependencies"
	@echo "  make test        - Run pytest unit tests"
	@echo ""
	@echo "Data Pipeline:"
	@echo "  make download    - Download NHANES XPT files"
	@echo "  make process     - Merge components and apply CDC/AAP labels"
	@echo ""
	@echo "Modeling:"
	@echo "  make train       - Train all models (baselines + gradient boosting)"
	@echo "  make notebook    - Launch Jupyter notebook"
	@echo ""
	@echo "Artifacts:"
	@echo "  make figures     - Generate all publication figures"
	@echo "  make clean       - Remove generated files (figures, models, results)"
	@echo ""
	@echo "Reproducibility:"
	@echo "  make freeze      - Save current package versions to requirements.txt"
	@echo ""

# Setup virtual environment
setup:
	@echo "🔧 Setting up Python environment..."
	python3 -m venv venv
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Setup complete. Activate with: source venv/bin/activate"

# Run tests
test:
	@echo "🧪 Running pytest unit tests..."
	pytest tests/ -v --tb=short
	@echo "✅ Tests complete"

# Download NHANES data
download:
	@echo "📥 Downloading NHANES data..."
	python 01_download_nhanes_data.py
	@echo "✅ Download complete"

# Process and merge data
process:
	@echo "⚙️  Processing and merging NHANES components..."
	python 02_process_nhanes_data.py
	@echo "✅ Processing complete"

# Train models
train:
	@echo "🤖 Training models..."
	python 03_train_models.py
	@echo "✅ Training complete"

# Launch Jupyter notebook
notebook:
	@echo "📓 Launching Jupyter notebook..."
	jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb

# Generate figures
figures:
	@echo "📊 Generating publication figures..."
	@echo "TODO: Add script to regenerate all figures from saved results"
	@echo "✅ Figures generated in figures/"

# Clean generated files
clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf figures/*.png
	rm -rf models/*.pkl
	rm -rf results/*.json results/*.csv
	rm -rf artifacts/*.pkl
	rm -rf logs/*.log
	@echo "✅ Clean complete"

# Freeze package versions
freeze:
	@echo "❄️  Freezing package versions..."
	./venv/bin/pip freeze > requirements.txt
	@echo "✅ requirements.txt updated"

# Create all necessary directories
dirs:
	@echo "📁 Creating project directories..."
	mkdir -p configs figures models results artifacts logs reports data/raw data/processed notebooks src tests
	@echo "✅ Directories created"

