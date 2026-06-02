#!/bin/bash
# =============================================================================
# run_v13_primary.sh
# Reproduce v1.3 Primary Model Results (Non-Interactive)
# =============================================================================

set -e  # Exit on error

echo "============================================================"
echo "NHANES Periodontitis v1.3 Primary Model Reproduction"
echo "============================================================"
echo ""

# Set random seeds for reproducibility
export PYTHONHASHSEED=42

# Navigate to project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check Python version
echo "Python version:"
python3 --version
echo ""

# Check dependencies
echo "Checking dependencies..."
python3 -c "
import pandas, numpy, sklearn, xgboost, catboost, lightgbm, optuna, shap
print('  pandas:', pandas.__version__)
print('  numpy:', numpy.__version__)
print('  scikit-learn:', sklearn.__version__)
print('  xgboost:', xgboost.__version__)
print('  catboost:', catboost.__version__)
print('  lightgbm:', lightgbm.__version__)
print('  optuna:', optuna.__version__)
print('  shap:', shap.__version__)
print('All dependencies installed')
"
echo ""

# Run notebook sections via papermill (if available) or jupyter nbconvert
echo "Running notebook..."
echo ""

if command -v papermill &> /dev/null; then
    echo "Using papermill..."
    papermill notebooks/00_nhanes_periodontitis_end_to_end.ipynb \
               notebooks/00_nhanes_periodontitis_end_to_end_executed.ipynb \
               --no-progress-bar
else
    echo "Using jupyter nbconvert..."
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=3600 \
        --output 00_nhanes_periodontitis_end_to_end_executed.ipynb \
        notebooks/00_nhanes_periodontitis_end_to_end.ipynb
fi

echo ""
echo "============================================================"
echo "REPRODUCTION COMPLETE"
echo "============================================================"
echo ""
echo "Results saved to:"
echo "   - results/v13_primary_norc_summary.json"
echo "   - results/v13_secondary_full_summary.json"
echo "   - results/v13_operating_points.json"
echo "   - results/v13_featuredrop.json"
echo "   - results/v13_nan_ablation.json"
echo "   - results/v13_shap_summary.json"
echo ""
echo "Figures saved to:"
echo "   - figures/14_v13_operating_points.png"
echo "   - figures/15_shap_beeswarm.png"
echo "   - figures/16_shap_importance.png"
echo "   - figures/17_shap_dependence.png"
echo "   - figures/18_nan_ablation.png"
echo ""
echo "Git tags available:"
echo "   - v1.3-primary-norc"
echo "   - v1.3-secondary-full"
echo ""
