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
"${PYTHON:-python3}" --version
echo ""

# Check dependencies
echo "Checking dependencies..."
"${PYTHON:-python3}" -c "
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

echo "Running maintained Python reproduction script..."
"${PYTHON:-python3}" scripts/reproduce_v13_primary.py

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
