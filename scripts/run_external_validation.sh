#!/bin/bash
# ==============================================================================
# Same-source temporal validation script for NHANES 2009-2010
# ==============================================================================
# 
# This script runs the temporal validation notebook non-interactively,
# generating all figures and result files.
#
# Usage:
#   bash scripts/run_external_validation.sh
#
# Outputs:
#   - results/external_summary.json
#   - results/external_0910_metrics.json
#   - results/prevalence_check.json
#   - results/decision_curve_external.json
#   - figures/external_roc_pr_calibration.png
#   - figures/decision_curve_external.png
# ==============================================================================

set -e  # Exit on error

echo "=============================================="
echo "NHANES Same-Source Temporal Validation (2009-2010)"
echo "=============================================="

# Navigate to project root
cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

echo "Project root: $PROJECT_ROOT"

# Check if notebook exists
NOTEBOOK="$PROJECT_ROOT/notebooks/01_external_validation.ipynb"
if [ ! -f "$NOTEBOOK" ]; then
    echo "Error: Notebook not found at $NOTEBOOK"
    exit 1
fi

echo "Running notebook: $NOTEBOOK"

# Execute notebook
jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=3600 \
    --ExecutePreprocessor.kernel_name=python3 \
    --output="01_external_validation_executed.ipynb" \
    "$NOTEBOOK"

echo ""
echo "Notebook executed successfully."
echo ""

# Check outputs
echo "Checking output files..."

FILES=(
    "results/external_summary.json"
    "results/external_0910_metrics.json"
    "results/prevalence_check.json"
    "results/decision_curve_external.json"
    "figures/external_roc_pr_calibration.png"
    "figures/decision_curve_external.png"
)

for file in "${FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        echo "   OK $file"
    else
        echo "   MISSING $file"
    fi
done

echo ""
echo "=============================================="
echo "Same-source temporal validation complete."
echo "=============================================="
