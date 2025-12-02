# Model Card: Periodontitis Risk Prediction (v1.3-primary)

## Model Details

| Field | Value |
|-------|-------|
| **Model Name** | v1.3-primary-norc |
| **Version** | 1.3 |
| **Type** | Soft-voting ensemble (CatBoost + XGBoost + LightGBM) |
| **Task** | Binary classification (periodontitis vs. no periodontitis) |
| **Training Data** | NHANES 2011-2014, adults ≥30 years (n=9,379) |
| **Features** | 29 predictors (excludes reverse-causality features) |
| **Calibration** | Isotonic regression |

---

## Intended Use

### Primary Intended Use
- **Population screening** to identify individuals who may benefit from clinical periodontal examination
- **Risk stratification** in public health settings
- **Research applications** comparing ML methods for periodontitis prediction

### Out-of-Scope Uses
- ❌ **NOT for clinical diagnosis** - requires professional examination
- ❌ **NOT for treatment planning** - needs detailed clinical assessment
- ❌ **NOT validated outside US adults 30+**
- ❌ **NOT validated on clinic-collected data** (only NHANES survey data)

---

## Operating Thresholds

### Rule-Out Threshold (Screening)
| Metric | Value |
|--------|-------|
| **Threshold** | 0.35 |
| **Sensitivity (Recall)** | 99.9% |
| **Specificity** | 12.4% |
| **NPV** | 96% |
| **Use Case** | Initial screening; negative result rules out disease |

**Interpretation:** A predicted probability < 0.35 indicates low risk. With 96% NPV, a negative result provides strong evidence against periodontitis.

### Balanced Threshold (Clinical Decision)
| Metric | Value |
|--------|-------|
| **Threshold** | 0.65 |
| **Sensitivity (Recall)** | 72.8% |
| **Specificity** | 59.2% |
| **Youden's J** | 0.32 |
| **Use Case** | Clinical decision support; balanced tradeoff |

**Interpretation:** Optimal threshold for maximizing Youden's J index. Use when balanced sensitivity/specificity is desired.

---

## Performance Metrics

| Metric | Value |
|--------|-------|
| **AUC-ROC** | 0.7172 |
| **PR-AUC** | 0.8157 |
| **Brier Score** | 0.1783 (calibrated) |

### Comparison with Secondary Model (full 33 features)
| Model | AUC-ROC | PR-AUC |
|-------|---------|--------|
| Primary (29 features) | 0.7172 | 0.8157 |
| Secondary (33 features) | 0.7255 | 0.8207 |

The ~1% AUC difference reflects the exclusion of reverse-causality features (dental_visit, floss_days, mobile_teeth).

---

## Features

### Included (29 features)
| Category | Features |
|----------|----------|
| **Demographics** | age, sex, education |
| **Behaviors** | smoke_current, smoke_former, alcohol_current |
| **Metabolic** | bmi, waist_cm, waist_height, height_cm, systolic_bp, diastolic_bp, glucose, triglycerides, hdl |
| **Missingness Indicators** | bmi_missing, systolic_bp_missing, diastolic_bp_missing, glucose_missing, triglycerides_missing, hdl_missing, smoking_missing, alcohol_missing, waist_cm_missing, waist_height_missing, height_cm_missing, alcohol_current_missing |

### Excluded (reverse-causality)
- `dental_visit` - may reflect treatment-seeking behavior
- `floss_days` - sicker patients may floss more
- `mobile_teeth` - is a consequence, not predictor, of disease
- `floss_days_missing`

---

## Monotonic Constraints

Clinical priors enforced during training:

| Constraint | Features | Rationale |
|------------|----------|-----------|
| **Increasing (+1)** | age, bmi, waist_cm, waist_height, systolic_bp, diastolic_bp, glucose, triglycerides | Higher values → increased risk |
| **Decreasing (-1)** | hdl | Higher HDL → reduced risk |
| **Unconstrained (0)** | All others | Allow model flexibility |

---

## Calibration Notes

- **Method:** Isotonic regression
- **Leakage Prevention:** Fit on each fold's validation predictions, applied only to that fold
- **Brier Improvement:** -1.6%
- **Reliability:** Predicted probabilities can be interpreted as approximate risk estimates

---

## External Validation

### NHANES 2009-2010 Results

| Metric | Value | 95% CI |
|--------|-------|--------|
| **N (test)** | 5,177 | — |
| **Prevalence** | 67.2% | — |
| **AUC-ROC** | 0.677 | [0.661, 0.693] |
| **PR-AUC** | 0.773 | [0.757, 0.789] |
| **Brier Score** | 0.200 | [0.194, 0.207] |

### Operating Points on External Data

| Threshold | Sensitivity | Specificity | PPV | NPV |
|-----------|-------------|-------------|-----|-----|
| Rule-Out (0.35) | 97.1% | 18.1% | 70.8% | 75.2% |
| Balanced (0.65) | 82.6% | 43.3% | 74.9% | 54.9% |

### Transportability and Recalibration

When applied to NHANES 2009–2010, the model achieved AUC 0.677, a realistic ~4% drop from internal validation (0.717). Calibration showed drift at lower predicted probabilities (underestimation below 0.3) with reasonable alignment above 0.5.

**Recommendations for deployment:**
- Perform local recalibration on a small validation sample before clinical use
- Consider setting thresholds using cohort-specific data
- Monitor missingness patterns—if they differ substantially from NHANES, retrain missing indicators

---

## Limitations and Risks

### Known Limitations
1. **Moderate external generalization** - AUC dropped ~4% on NHANES 2009-2010
2. **High prevalence** (67-68%) vs CDC estimates (47%) - reflects full-mouth exam inclusion criteria
3. **NHANES-specific missingness patterns** - may not transfer to clinical data
4. **Low specificity at rule-out** (18%) - high false positive rate for screening
5. **Calibration drift** - underestimation at low probabilities on external cohort

### Ethical Considerations
- Model may perpetuate biases in NHANES sampling
- Not validated across all demographic subgroups
- Should not replace clinical judgment

### Failure Modes
- May underperform on populations with different risk factor distributions
- Missingness indicators may not be informative in settings with complete data
- Monotonic constraints may limit flexibility for non-linear relationships

---

## Training Details

| Parameter | Value |
|-----------|-------|
| **Validation** | Stratified 5-fold CV |
| **Hyperparameter Tuning** | Optuna (100 trials per model) |
| **Ensemble Weights** | CatBoost 34%, XGBoost 33%, LightGBM 33% |
| **Random Seed** | 42 |

---

## Reproducibility

```bash
# Clone repository
git clone https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project.git

# Install dependencies
pip install -r requirements.txt

# Run reproduction script
bash scripts/run_v13_primary.sh

# Or run notebook interactively
jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb
```

---

## Citation

```bibtex
@article{barbosa2025gradient,
  title={Evaluating Modern Gradient Boosting Methods for Periodontitis Prediction},
  author={Barbosa, Francisco Teixeira},
  journal={medRxiv preprint},
  year={2025}
}
```

---

## Contact

**Author:** Francisco Teixeira Barbosa  
**Email:** cisco@periospot.com  
**GitHub:** [@Tuminha](https://github.com/Tuminha)

---

**⚠️ DISCLAIMER:** This model is for research and screening purposes only. It is NOT a diagnostic tool and should NOT be used for clinical decision-making without professional medical evaluation.

