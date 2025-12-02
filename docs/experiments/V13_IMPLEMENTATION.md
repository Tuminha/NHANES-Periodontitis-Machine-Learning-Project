# v1.3-Light Implementation Checklist

**Status:** In Progress  
**Date:** December 2025

---

## GPT's Requirements Checklist

### 1. Feature Engineering ✅ DONE
- [x] `waist_height` = waist_cm / height_cm
- [x] `smoke_current`, `smoke_former` (3-level smoking)
- [x] `alcohol_current` (from ALQ110)
- [x] Keep native NaN + `_missing` indicators

### 2. Monotonic Constraints ⏳ IN PROGRESS
- [ ] XGBoost: monotone_constraints parameter
- [ ] LightGBM: monotone_constraints parameter
- [ ] CatBoost: monotone_constraints parameter

**Constraint Mapping:**
```python
# Monotone INCREASING (+1): Higher value → higher risk
MONO_INCREASING = ['age', 'bmi', 'waist_cm', 'waist_height', 
                   'systolic_bp', 'diastolic_bp', 'glucose', 'triglycerides']

# Monotone DECREASING (-1): Higher value → lower risk
MONO_DECREASING = ['hdl']

# Unconstrained (0):
UNCONSTRAINED = ['sex', 'education', 'smoking', 'alcohol', 'alcohol_current',
                 'smoke_current', 'smoke_former', 'dental_visit', 'mobile_teeth',
                 'floss_days', 'height_cm', '*_missing']
```

### 3. Training & Evaluation ⏳ PENDING
- [ ] 5-fold stratified CV
- [ ] Report per-fold metrics
- [ ] Soft-voting ensemble
- [ ] Isotonic calibration

### 4. Operating-Point Policy ⏳ PENDING
- [ ] Constrained threshold search
- [ ] Target A: Recall ≥90%, Specificity ≥35%
- [ ] If infeasible:
  - [ ] Rule-out: max Recall with Specificity ≥20%
  - [ ] Balanced: max Youden's J
- [ ] Annotate both on ROC/PR curves

### 5. NaN Sensitivity Ablation ⏳ PENDING
- [ ] ABLATION_1: No _missing indicators
- [ ] ABLATION_2: Indicator-only (median impute values)
- [ ] ABLATION_3: Complete-case only
- [ ] STRAT_AVAIL: Lab-complete subset
- [ ] Summary table with deltas

### 6. Interpretability ⏳ PENDING
- [ ] SHAP beeswarm plot
- [ ] SHAP mean |SHAP| bar chart
- [ ] Feature-drop test (dental_visit, floss_*)
- [ ] Reverse-causality discussion

### 7. External-Cycle Scaffold ⏳ PENDING
- [ ] 2009-2010 data loader
- [ ] Direct evaluation mode
- [ ] Recalibrated evaluation mode

### 8. Documentation ⏳ PENDING
- [ ] README: Clinical operating points table
- [ ] ARTICLE_DRAFT.md: Limitations section
- [ ] results/v1_3_summary.json
- [ ] All figures with Periospot palette

---

## Implementation Order

1. **NOW:** Add monotonic constraints to existing model cells
2. **NEXT:** Create Section 18: v1.3 Ablation Experiments
3. **THEN:** Update Section 17: SHAP with reverse-causality
4. **THEN:** Create Section 19: Operating Point Policy
5. **THEN:** Create Section 20: External Validation Scaffold
6. **FINALLY:** Update documentation

---

## Acceptance Criteria

- [ ] v1_3_summary.json with all metrics
- [ ] Updated README with operating points table
- [ ] Updated ARTICLE_DRAFT.md with limitations
- [ ] Figures: ROC/PR with dual thresholds, SHAP, ablation comparison, calibration
- [ ] Clear answer: Can we achieve Recall ≥90% + Specificity ≥35%?

