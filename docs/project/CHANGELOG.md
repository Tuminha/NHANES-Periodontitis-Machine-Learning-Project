# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.3-primary-norc] - 2025-12-02

### Summary: Feature-Drop Decision for Publication

After comprehensive analysis, we established a **dual-model strategy** for publication:

1. **Primary Model (v1.3-primary-norc):** 29 features, excludes `dental_visit`, `floss_days`, `mobile_teeth`, and `floss_days_missing` to reduce treatment-seeking bias.

2. **Secondary Model (v1.3-secondary-full):** 33 features, includes all features for comparison.

### Rationale

The feature-drop test showed that removing reverse-causality features:
- Reduces AUC by only ~1.1% (0.7255 → 0.7172)
- Slightly improves balanced specificity (57.7% → 59.2%)
- Results in a more clinically plausible model
- Core risk factors (age, BP, glucose, smoking) drive majority of signal

### Key Findings

| Model Variant | AUC-ROC | PR-AUC | Rule-out Sens | Rule-out Spec | Balanced Sens | Balanced Spec |
|---------------|---------|--------|---------------|---------------|---------------|---------------|
| Primary (no RC) | 0.7172 | 0.8157 | 99.9% | 12.4% | 72.8% | 59.2% |
| Secondary (full) | 0.7255 | 0.8207 | 98.8% | 16.8% | 75.4% | 57.7% |

### Added
- `results/v13_primary_norc_summary.json` - Primary model metrics
- `results/v13_secondary_full_summary.json` - Secondary model metrics
- `results/v13_featuredrop.json` - Feature-drop experiment results
- `results/v13_nan_ablation.json` - Missing data ablation results
- `results/v13_shap_summary.json` - SHAP feature importance
- `figures/15_shap_beeswarm.png` - SHAP beeswarm plot
- `figures/16_shap_importance.png` - SHAP bar plot
- `figures/17_shap_dependence.png` - SHAP dependence plots
- `figures/18_nan_ablation.png` - NaN ablation comparison

### Changed
- Updated `results/v13_operating_points.json` to include both model variants
- Updated README.md with final metrics tables
- Updated ARTICLE_DRAFT.md with publication-ready methods/results

---

## [v1.3-light] - 2025-12-01

### Added
- Monotonic constraints for biological plausibility
- Enhanced features: `waist_height`, `smoke_current`, `smoke_former`, `alcohol_current`
- Dual operating-point policy (Rule-Out and Balanced)
- SHAP analysis for interpretability
- NaN ablation experiments

### Changed
- Model now enforces: risk ↑ with age/BP/glucose, risk ↓ with HDL
- Constraints cost ~0.006 AUC but improve interpretability

---

## [v1.2-quickwins] - 2025-11-30

### Added
- Soft-voting ensemble of CatBoost + XGBoost + LightGBM
- Threshold tuning for recall ≥95%
- Isotonic calibration for probability estimates

### Results
- Ensemble AUC: 0.7277 (+0.0009 from best single)
- Calibrated AUC: 0.7302 (+0.35% boost)
- Brier score improved by 1.6%

---

## [v1.1-native-nan] - 2025-11-29

### Added
- Native NaN handling for tree models (no imputation)
- 9 missing indicator features (`*_missing` columns)
- Demonstrated "missingness is informative" hypothesis

### Results
- AUC improved from 0.7071 → 0.7267 (+2.8%)
- All tree models benefited from native NaN handling

---

## [v1.0-baseline] - 2025-11-28

### Added
- Initial project setup
- CDC/AAP periodontitis labeling
- 14 predictors from Bashir et al.
- Baseline models: Logistic Regression, Random Forest
- XGBoost, CatBoost, LightGBM with Optuna tuning
- Stratified 5-fold cross-validation

### Results
- Best model: CatBoost (AUC 0.7071)
- Significant improvement over Logistic Regression (+13%)

