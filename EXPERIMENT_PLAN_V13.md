# v1.3 Experiment Plan: Addressing Gemini & GPT Feedback

**Created:** December 2025  
**Status:** Planning Phase  
**Goal:** Make research publication-ready with rigorous validation

---

## 📋 Executive Summary

Based on feedback from GPT and Gemini, we need to:
1. **Prove the NaN signal is real** (not just survey artifacts)
2. **Check specificity** (ensure we're not flagging everyone)
3. **Add external validation** (2009-2010 NHANES)
4. **Complete interpretability** (SHAP + Decision Curve)

---

## 🔬 Experiment Suite

### Experiment 1: Ablation Tests (NaN Hypothesis)

**Purpose:** Prove missing data signal is biological, not just survey design.

| Test | Description | Expected Result |
|------|-------------|-----------------|
| **ABLATION_1** | Models without `_missing` indicators (impute only) | AUC should drop ~0.01-0.02 |
| **ABLATION_2** | Models with indicators only (impute all values) | AUC should be lower than full |
| **ABLATION_3** | Complete-case analysis (drop rows with any NaN) | AUC ~0.70 if signal is real |
| **STRAT_AVAIL** | Evaluate only on participants with all labs | Tests if signal generalizes |

**Code Template:**
```python
# ABLATION_1: No missing indicators
X_ablation1 = df_features[ALL_FEATURES_LINEAR]  # Base features only
X_ablation1_imputed = SimpleImputer(strategy='median').fit_transform(X_ablation1)

# ABLATION_2: Indicators only (values imputed)
X_ablation2 = df_features[ALL_FEATURES_TREE].copy()
for col in CONTINUOUS_FEATURES + BINARY_FEATURES + ORDINAL_FEATURES:
    X_ablation2[col] = X_ablation2[col].fillna(X_ablation2[col].median())

# ABLATION_3: Complete case
mask_complete = df_features[ALL_FEATURES_LINEAR].notna().all(axis=1)
X_ablation3 = df_features.loc[mask_complete, ALL_FEATURES_LINEAR]
y_ablation3 = df_features.loc[mask_complete, TARGET]
```

---

### Experiment 2: Specificity Check

**Gemini's Concern:** "If you classify 90% of the population as positive to catch 98% of the sick people, the tool is useless."

**Current Status:**
- Recall: 97.97%
- Specificity: **UNKNOWN** ⚠️

**Required Analysis:**
```python
from sklearn.metrics import confusion_matrix

# At our current threshold (0.49)
y_pred = (proba_calibrated >= 0.49).astype(int)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

specificity = tn / (tn + fp)
print(f"Specificity at threshold 0.49: {specificity:.2%}")

# If specificity < 30%, we're flagging too many healthy people!
```

**Target:** Find threshold where Recall ≥ 90% AND Specificity ≥ 35%

**Code Template:**
```python
def find_clinical_threshold(y_true, proba, min_recall=0.90, min_specificity=0.35):
    """Find threshold meeting clinical constraints."""
    best = {'f1': -1, 'threshold': 0.5}
    
    for t in np.linspace(0.05, 0.95, 181):
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        recall = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        precision = tp / (tp + fp + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        
        if recall >= min_recall and specificity >= min_specificity:
            if f1 > best['f1']:
                best = {
                    'f1': f1, 'threshold': t,
                    'recall': recall, 'specificity': specificity,
                    'precision': precision
                }
    
    return best
```

---

### Experiment 3: External Validation (2009-2010)

**Purpose:** Prove model generalizes across NHANES cycles.

**Steps:**
1. Download NHANES 2009-2010 data (same components)
2. Apply same preprocessing pipeline
3. Apply same CDC/AAP labeling
4. Evaluate frozen model (no retraining)
5. Evaluate with 10% recalibration split

**Expected Results:**
- AUC drop ≤ 0.02 = **SUCCESS** (model generalizes)
- AUC drop > 0.05 = **CONCERN** (overfitting to 2011-2014)
- AUC drop due to NaN pattern change = **Validates Gemini's concern**

---

### Experiment 4: Reverse Causality Sensitivity

**Concern:** `dental_visit` and `floss_days` may reflect treatment, not risk.

**Analysis:**
```python
# Train model WITHOUT dental_visit and floss_days
FEATURES_NO_REVERSE = [f for f in ALL_FEATURES_TREE 
                       if f not in ['dental_visit', 'floss_days', 'floss_days_missing']]

# Compare AUC with and without
# If AUC drops significantly → these features carry real signal
# If AUC unchanged → model doesn't rely on reverse-causality features
```

---

## 📊 New Notebook Sections to Add

### Section 16: SHAP Analysis

**Purpose:** Model interpretability for clinical trust.

**Components:**
1. SHAP summary plot (beeswarm)
2. SHAP importance bar chart
3. SHAP dependence plots for top features
4. Flag reverse-causality candidates

### Section 17: Decision Curve Analysis

**Purpose:** Evaluate clinical utility across threshold range.

**Components:**
1. Net benefit curves
2. Comparison to "treat all" and "treat none" strategies
3. Threshold range recommendations

### Section 18: Ablation Experiments

**Purpose:** Prove NaN signal validity.

**Components:**
1. ABLATION_1 through ABLATION_3 results
2. STRAT_AVAIL analysis
3. Statistical comparison

### Section 19: Specificity Analysis

**Purpose:** Ensure clinical utility.

**Components:**
1. ROC curve with operating points
2. Threshold sweep with recall/specificity
3. Clinical operating point selection

### Section 20: External Validation (2009-2010)

**Purpose:** Prove generalization.

**Components:**
1. Data loading and preprocessing
2. Frozen model evaluation
3. Recalibration experiment
4. Comparison figures

---

## 📝 Documentation Updates

### README Updates
- [ ] Add v1.3 experiment results
- [ ] Add external validation results
- [ ] Add specificity analysis
- [ ] Update version comparison table

### ARTICLE_DRAFT.md (Preprint)
- [ ] Fill in results tables
- [ ] Add ablation results
- [ ] Add interpretation

### ARTICLE_PEER_REVIEW.md (JCP/JDR)
- [ ] Complete all tables
- [ ] Add external validation section
- [ ] Address reverse causality
- [ ] Full TRIPOD compliance

---

## 🎯 Priority Order

### Phase 1: Critical Checks (Before Any Publishing)
1. **Specificity check** - Takes 5 minutes, critical safety check
2. **Ablation tests** - Prove NaN signal is real

### Phase 2: Model Improvement
3. **Run v1.3 features** - New smoking/alcohol/waist variables
4. **Monotonic constraints** - Better generalization

### Phase 3: External Validation
5. **Download 2009-2010** - External test set
6. **Evaluate and recalibrate** - Prove generalization

### Phase 4: Interpretability
7. **SHAP analysis** - Feature importance
8. **Decision curve** - Clinical utility

### Phase 5: Publication
9. **Complete preprint** - medRxiv
10. **Prepare peer review** - JCP/JDR submission

---

## ⏱️ Time Estimates

| Task | Estimated Time |
|------|---------------|
| Specificity check | 10 minutes |
| Ablation experiments | 1-2 hours |
| Run v1.3 features | 30 minutes (training already done?) |
| Download 2009-2010 | 30 minutes |
| External validation | 1-2 hours |
| SHAP analysis | 1 hour |
| Decision curve | 30 minutes |
| Complete preprint | 2-3 hours |
| Peer review draft | 4-6 hours |

**Total: ~12-16 hours of focused work**

---

## 🚦 Decision Points

### After Specificity Check:
- If Specificity < 20% at Recall 98% → **STOP**, retune threshold
- If Specificity ≥ 35% at Recall 90% → **PROCEED**

### After Ablation Tests:
- If Complete-case AUC ≈ Full AUC → NaN signal is survey artifact
- If Complete-case AUC < Full AUC but stable → NaN carries real info

### After External Validation:
- If AUC drop ≤ 2% → Model ready for publication
- If AUC drop > 5% → Need recalibration analysis
- If AUC drop correlates with NaN pattern → Gemini was right

---

## 📎 Files to Create/Modify

### New Files:
- [ ] `ARTICLE_PEER_REVIEW.md` ✅ Created
- [ ] `EXPERIMENT_PLAN_V13.md` ✅ Creating now
- [ ] `results/ablation_results.json`
- [ ] `results/external_validation_results.json`
- [ ] `figures/13_shap_summary.png`
- [ ] `figures/14_decision_curve.png`
- [ ] `figures/15_external_validation.png`

### Modified Files:
- [ ] `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` (add sections 16-20)
- [ ] `README.md` (update with results)
- [ ] `ARTICLE_DRAFT.md` (fill tables)

---

**Next Action:** Start with Phase 1 - Specificity Check (10 minutes)

