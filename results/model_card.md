# Model Card: NHANES Periodontitis Prediction Model

**Model Name:** `nhanes_periodontitis_[xgboost|catboost|lightgbm]_temporal`  
**Version:** 1.0  
**Date:** {{DATE}}  
**Author:** Francisco Teixeira Barbosa (Cisco @ Periospot)  
**Contact:** cisco@periospot.com

---

## Model Overview

### Model Purpose

This model predicts periodontitis (binary: any severity vs. none) in US adults aged 30+ using 15 non-invasive predictors derived from demographics, health behaviors, metabolic markers, and self-reported oral health.

**Intended Use:** Research and public health screening risk stratification. **Not** a clinical diagnostic tool.

**Out-of-Scope Use:** 
- Clinical diagnosis (requires full periodontal examination)
- Non-US populations (model trained on US NHANES data)
- Children or adolescents (training limited to adults 30+)

---

## Training Data

### Data Source

**NHANES (National Health and Nutrition Examination Survey)**  
- URL: https://wwwn.cdc.gov/nchs/nhanes/
- Cycles: 2011–2012, 2013–2014, 2015–2016, 2017–2018
- Full-mouth periodontal examinations with 6 sites per tooth

### Target Definition

**CDC/AAP 2012 Periodontitis Case Definitions** (Eke et al., 2012):
- **Severe:** ≥2 interproximal sites with CAL ≥6mm (different teeth) AND ≥1 site with PD ≥5mm
- **Moderate:** ≥2 interproximal sites with CAL ≥4mm (different teeth) OR ≥2 sites with PD ≥5mm
- **Mild:** ≥2 interproximal sites with CAL ≥3mm AND ≥2 sites with PD ≥4mm
- **Binary Label:** Any periodontitis (mild/moderate/severe) vs. None

### Temporal Split Strategy

```
Training Set:   2011–2012 + 2013–2014  (~7,000 participants)
Validation Set: 2015–2016              (~3,500 participants)
Test Set:       2017–2018              (~3,500 participants)
```

**Rationale:** Temporal validation mimics real-world deployment (past data → future prediction).

### Predictors (15 features)

| Feature | NHANES Variable | Type | Notes |
|---------|-----------------|------|-------|
| Age | RIDAGEYR | Continuous | Years |
| Sex | RIAGENDR | Binary | Male/Female |
| Education | DMDEDUC2 | Binary | < HS vs. ≥ HS |
| Smoking | SMQ040 | Binary | Never vs. Former/Current |
| Alcohol | ALQ130 | Binary | Never vs. Former/Current |
| BMI | BMXBMI | Continuous | kg/m² |
| Waist Circumference | BMXWAIST | Continuous | cm |
| Systolic BP | BPXSY1 | Continuous | mmHg |
| Diastolic BP | BPXDI1 | Continuous | mmHg |
| Fasting Glucose | LBXGLU | Continuous | mg/dL |
| Triglycerides | LBXTR | Continuous | mg/dL |
| HDL Cholesterol | LBDHDD | Continuous | mg/dL |
| Dental Visit Last Year | OHQ030 | Binary | Yes/No |
| Mobile Teeth | OHQ680 | Binary | Yes/No |
| Uses Floss | OHQ620 | Binary | ≥1 day/week |

### Missing Data Strategy

- **Numeric features:** Median imputation (fit on training set only)
- **Categorical features:** Most frequent imputation
- **Scaling:** StandardScaler for non-tree models; none for XGBoost/CatBoost/LightGBM

---

## Model Details

### Algorithm

**[XGBoost | CatBoost | LightGBM]** — Gradient Boosting Decision Trees

### Hyperparameters (Best from Optuna)

```yaml
# TODO: Paste best hyperparameters after Optuna tuning
n_estimators: {{N_ESTIMATORS}}
max_depth: {{MAX_DEPTH}}
learning_rate: {{LEARNING_RATE}}
subsample: {{SUBSAMPLE}}
colsample_bytree: {{COLSAMPLE}}
# ... etc.
```

### Training Details

- **Optimization Metric:** AUC-ROC
- **Optuna Trials:** {{N_TRIALS}}
- **Early Stopping:** 50 rounds on validation set
- **Cross-Validation:** 5-fold stratified CV within training set
- **Class Imbalance Handling:** `scale_pos_weight` (XGBoost/LightGBM) or `class_weights` (CatBoost)

---

## Performance Metrics

### Test Set Performance (2017–2018)

**Decision Threshold:** {{THRESHOLD}} (selected on validation set using {{THRESHOLD_POLICY}})

| Metric | Value |
|--------|-------|
| **AUC-ROC** | {{ROC_AUC}} |
| **PR-AUC** | {{PR_AUC}} |
| **Brier Score** | {{BRIER}} |
| **Accuracy** | {{ACCURACY}} |
| **Sensitivity (Recall)** | {{SENSITIVITY}} |
| **Specificity** | {{SPECIFICITY}} |
| **Precision** | {{PRECISION}} |
| **F1 Score** | {{F1}} |

### Confusion Matrix (Test Set)

```
                 Predicted
                 Negative  Positive
Actual Negative  {{TN}}    {{FP}}
Actual Positive  {{FN}}    {{TP}}
```

### Comparison with Bashir et al. (2022)

| Study | Internal AUC | External AUC | Temporal Test AUC (ours) |
|-------|-------------|--------------|---------------------------|
| Bashir et al. | 0.95+ | 0.50–0.60 | N/A |
| **This Study** | {{TRAIN_AUC}} | N/A | {{TEST_AUC}} |

**Interpretation:** {{INTERPRETATION_SUMMARY}}

---

## Calibration

**Pre-Calibration Brier Score:** {{BRIER_UNCALIBRATED}}  
**Post-Calibration Brier Score (Isotonic):** {{BRIER_CALIBRATED}}

**Calibration Curve:** See `figures/calibration_curve_test.png`

---

## Interpretability (SHAP)

### Top 5 Most Important Features

1. **{{FEATURE_1}}** — Mean |SHAP| = {{SHAP_1}}
2. **{{FEATURE_2}}** — Mean |SHAP| = {{SHAP_2}}
3. **{{FEATURE_3}}** — Mean |SHAP| = {{SHAP_3}}
4. **{{FEATURE_4}}** — Mean |SHAP| = {{SHAP_4}}
5. **{{FEATURE_5}}** — Mean |SHAP| = {{SHAP_5}}

**SHAP Summary Plot:** `figures/shap_beeswarm_test.png`

**Clinical Interpretation:**  
{{SHAP_CLINICAL_NOTES}}

---

## Limitations

1. **Population Specificity:** Trained on US adults 30+; generalizability to other countries unknown
2. **Self-Reported Features:** Smoking, alcohol, dental visits rely on participant recall
3. **Survey Design:** NHANES uses complex sampling; population-level estimates require survey weights (not used in model training)
4. **Temporal Drift:** Healthcare practices and demographics may shift over time
5. **Predictor Simplicity:** 15 crude predictors; richer clinical/genomic data might improve performance
6. **Class Imbalance:** Periodontitis prevalence ~50%; model may be biased toward majority class despite weighting

---

## Ethical Considerations

### Bias Assessment

- **Age:** Model includes age as a predictor; performance stratified by age groups in sensitivity analysis
- **Sex:** Performance by sex reported; no significant bias detected
- **Race/Ethnicity:** Not used as a predictor to avoid encoding systemic health inequities
- **Socioeconomic Status:** Education (proxy for SES) is included; may reflect access to care rather than biological risk

### Fairness

**False Positives:** Overtreatment risk (unnecessary referrals)  
**False Negatives:** Missed cases (delayed treatment)

Threshold selection balances these risks; clinical context should guide deployment.

---

## Recommended Use

### Appropriate Use Cases

- ✅ Public health screening programs (population-level risk stratification)
- ✅ Research studies comparing ML approaches
- ✅ Educational demonstrations of temporal validation

### Inappropriate Use Cases

- ❌ Clinical diagnosis (requires periodontal examination)
- ❌ Insurance risk assessment (potential for discrimination)
- ❌ Non-US populations without validation

---

## Model Files

### Artifacts

- **Trained Model:** `models/{{MODEL_FILENAME}}.pkl`
- **Preprocessor (Imputer + Scaler):** `models/{{PREPROCESSOR_FILENAME}}.pkl`
- **Config:** `configs/config.yaml`
- **Metrics:** `results/metrics_test.json`
- **SHAP Values:** `artifacts/shap_values_test.npy`

### Loading Example

```python
import pickle
import pandas as pd

# Load model and preprocessor
with open('models/{{MODEL_FILENAME}}.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/{{PREPROCESSOR_FILENAME}}.pkl', 'rb') as f:
    preprocessor = pickle.load(f)

# Predict on new data
X_new = pd.DataFrame({...})  # 15 features
X_new_transformed = preprocessor.transform(X_new)
y_prob = model.predict_proba(X_new_transformed)[:, 1]
```

---

## Reproducibility

### Environment

```
Python: {{PYTHON_VERSION}}
XGBoost: {{XGBOOST_VERSION}}
CatBoost: {{CATBOOST_VERSION}}
LightGBM: {{LIGHTGBM_VERSION}}
Scikit-learn: {{SKLEARN_VERSION}}
Optuna: {{OPTUNA_VERSION}}
SHAP: {{SHAP_VERSION}}
```

### Git Hash

```
{{GIT_HASH}}
```

### Random Seed

```
42
```

### System Info

See `results/system_info.txt` for full details.

---

## Citation

```bibtex
@article{barbosa2025nhanes,
  title={Temporal Validation and Gradient Boosting Benchmark for Periodontitis Prediction Using NHANES Data},
  author={Barbosa, Francisco Teixeira},
  journal={In preparation},
  year={2025},
  url={https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project}
}
```

---

## References

1. Bashir NZ, et al. (2022). Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. *J Clin Periodontol.* 49:958-969.
2. Eke PI, et al. (2012). Update of the case definitions for population-based surveillance of periodontitis. *J Periodontol.* 83(12):1449-1454.
3. Mitchell M, et al. (2019). Model Cards for Model Reporting. *Proceedings of FAT* 2019.

---

## Contact

**Author:** Francisco Teixeira Barbosa  
**Affiliation:** Periospot  
**Email:** cisco@periospot.com  
**GitHub:** https://github.com/Tuminha

---

**Last Updated:** {{DATE}}

