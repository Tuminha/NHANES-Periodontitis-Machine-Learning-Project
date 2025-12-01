# 🏁 Baseline v1 Checkpoint

**Date:** December 1, 2025  
**Status:** ✅ Complete and Production-Ready  
**Performance:** AUC 0.7071 (CatBoost winner)

---

## 📊 Current Results Summary

### Model Performance (5-Fold CV)

| Model | AUC | PR-AUC | Precision | Recall | F1 |
|-------|-----|--------|-----------|--------|-----|
| **CatBoost** | **0.7071** | 0.815 | 0.768 | **0.960** | **0.853** |
| LightGBM | 0.7062 | 0.813 | 0.735 | 0.957 | 0.834 |
| XGBoost | 0.7056 | 0.813 | 0.722 | 0.942 | 0.819 |
| Random Forest | 0.6953 | 0.806 | 0.766 | 0.808 | 0.781 |
| Logistic Regression | 0.6430 | 0.771 | 0.766 | 0.594 | 0.671 |

---

## 🎯 What We Accomplished

### Scientific Achievements:
1. ✅ **Proved hypothesis:** Gradient boosting > traditional ML (p < 0.001)
2. ✅ **Filled research gap:** First systematic XGB/CatBoost/LightGBM comparison for periodontitis
3. ✅ **Rigorous methodology:** 5-fold CV, Optuna tuning, statistical testing
4. ✅ **Clinical utility:** 96% recall suitable for screening
5. ✅ **Full reproducibility:** Open code, documented decisions, version control

### Technical Achievements:
1. ✅ Fixed zero-variance variables (alcohol, floss)
2. ✅ Improved floss from binary → ordinal (22.5x variance boost)
3. ✅ Cleaned outliers (diastolic BP, triglycerides)
4. ✅ Removed multicollinearity (waist_cm excluded)
5. ✅ Proper cross-validation setup
6. ✅ Optuna hyperparameter optimization (100 trials × 3 models)

---

## 📁 Current Feature Set (14 Predictors)

### Demographics (3):
- age (continuous, 0% missing)
- sex (binary, 0% missing)
- education (binary, 0% missing)

### Behaviors (2):
- smoking (binary, 54.5% missing)
- alcohol (binary via ALQ101, 44.1% missing)

### Metabolic (6):
- bmi (continuous, 5.2% missing)
- systolic_bp (continuous, 12.0% missing)
- diastolic_bp (continuous, 12.0% missing)
- glucose (continuous, 55.0% missing)
- triglycerides (continuous, 55.5% missing)
- hdl (continuous, 9.4% missing)

### Oral Health (3):
- dental_visit (binary, 0% missing)
- mobile_teeth (binary, 0% missing)
- floss_days (ordinal 1-5, 0.1% missing)

---

## 🔍 Performance Analysis

### Strengths:
- ✅ **Exceptional recall:** 96% (CatBoost) - catches almost all cases
- ✅ **Statistically significant:** All gradient boosting > baselines (p<0.001)
- ✅ **Stable estimates:** Low variance across folds
- ✅ **Good PR-AUC:** 0.81+ (accounts for 68% class imbalance)

### Weaknesses:
- ⚠️ **Moderate AUC:** 0.71 (good, not excellent)
- ⚠️ **Precision trade-off:** 77% precision = 23% false positives
- ⚠️ **Gap vs Bashir:** 0.71 vs 0.95 (24 point difference)

### Root Causes of Moderate Performance:
1. **Weak correlations:** Age (r=0.16) strongest, most features r < 0.10
2. **High missing data:** 55% in fasting labs (imputation noise)
3. **Feature limitations:** No genetics, inflammation markers, detailed behaviors
4. **High prevalence:** 68% makes discrimination harder
5. **Alcohol variable:** ALQ101 (ever drinker) weaker than ALQ130 (current drinks)

---

## 🚀 Improvement Opportunities (Feature Engineering v2)

### Strategy A: Enhanced Feature Engineering

**1. Age Groups (Non-linear Effects):**
```python
# Instead of linear age, create age bins
df['age_group'] = pd.cut(df['age'], 
                         bins=[30, 40, 50, 60, 70, 80],
                         labels=['30-39', '40-49', '50-59', '60-69', '70+'])
```

**2. Metabolic Syndrome Score:**
```python
# Composite risk score
metabolic_syndrome_count = (
    (df['bmi'] >= 30).astype(int) +           # Obesity
    (df['glucose'] >= 100).astype(int) +      # Prediabetes
    (df['triglycerides'] >= 150).astype(int) + # High triglycerides
    (df['hdl'] < 40).astype(int) +            # Low HDL (male)
    (df['systolic_bp'] >= 130).astype(int)    # Hypertension
)
df['metabolic_syndrome_score'] = metabolic_syndrome_count
```

**3. Interaction Features:**
```python
# Age × Smoking (older smokers highest risk)
df['age_smoking_interaction'] = df['age'] * df['smoking']

# BMI × Diabetes (obesity-diabetes synergy)
df['bmi_diabetes'] = df['bmi'] * (df['glucose'] >= 126).astype(int)
```

**4. Polynomial Features:**
```python
from sklearn.preprocessing import PolynomialFeatures
# Age^2, Age^3 to capture non-linear aging effects
```

**5. Alternative Smoking Variable:**
```python
# Check if SMQ020 (smoked 100+ cigarettes lifetime) available
# Might have less missing data than SMQ040
```

**6. Better Alcohol Variable:**
```python
# Investigate ALQ120Q (drinking frequency)
# Might capture current behavior better than ALQ101 (ever drinker)
```

---

### Strategy B: Advanced Imputation

**Current:** Simple median/mode imputation  
**Upgrade to:** IterativeImputer (MICE) or KNN imputation

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# MICE: Multiple Imputation by Chained Equations
imputer = IterativeImputer(random_state=42, max_iter=10)
# Imputes each feature using all other features (smarter than median)
```

---

### Strategy C: Feature Selection

**Current:** Using all 14 features  
**Upgrade to:** Select top N features by importance

```python
from sklearn.feature_selection import SelectFromModel

# Use RandomForest or XGBoost to select features
selector = SelectFromModel(XGBClassifier(), threshold='median')
# Keep only features with above-median importance
```

---

### Strategy D: Ensemble Methods

**Current:** Individual models  
**Upgrade to:** Stacked ensemble

```python
# Combine XGBoost + CatBoost + LightGBM predictions
# Use LogReg as meta-learner
ensemble_pred = 0.33*xgb_pred + 0.33*cb_pred + 0.34*lgb_pred
```

---

### Strategy E: Cost-Sensitive Learning

**Current:** Equal cost for FP and FN  
**Upgrade to:** Penalize false negatives more

```python
# XGBoost with scale_pos_weight
xgb_model = XGBClassifier(
    scale_pos_weight=2.0  # Penalize missing periodontitis 2x more
)
```

---

## 📋 Experiment Tracking Plan

### Version Control Strategy:

```
main branch (protected)
├─ v1.0-baseline (THIS CHECKPOINT) ← We can always return here
│  └─ Results: AUC 0.7071, features=14, imputation=simple
│
├─ experiment/age-groups (try Strategy A.1)
│  └─ Results: AUC 0.72? Compare to v1.0
│
├─ experiment/metabolic-syndrome (try Strategy A.2)
│  └─ Results: AUC 0.73? Compare to v1.0
│
└─ experiment/advanced-imputation (try Strategy B)
   └─ Results: AUC 0.71? (might not help)
```

---

## ✅ Files to Preserve (Baseline v1)

### Data:
- `data/processed/features_full.parquet` (before cleaning)
- `data/processed/features_cleaned.parquet` (after cleaning)

### Models:
- `results/baseline_results.json`
- `results/xgboost_results.json`
- `results/catboost_results.json`
- `results/lightgbm_results.json`
- `results/model_comparison_detailed.json`

### Figures:
- `figures/01_periodontitis_classification_summary.png`
- `figures/02-07_*.png` (EDA)
- `figures/08-11_model_comparison_*.png` (Results)

### Code:
- `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` (Sections 1-14 complete)
- `src/*.py` (all modules)
- `configs/config.yaml`

---

## 🎯 Next Experiment Recommendations

### **Experiment 1: Age Group Encoding (HIGH PRIORITY)** ⭐

**Rationale:**
- Age has r=0.16 (strongest predictor)
- But relationship might be non-linear (risk accelerates after 60?)
- Categorical age groups might capture threshold effects

**Expected Impact:** +0.01-0.03 AUC (1-3 points)

---

### **Experiment 2: Metabolic Syndrome Score (MEDIUM PRIORITY)**

**Rationale:**
- Individual metabolic features weak (r<0.10)
- But **combined** might have synergistic effect
- Clinical reality: Multiple risk factors compound

**Expected Impact:** +0.01-0.02 AUC

---

### **Experiment 3: Better Alcohol Variable (MEDIUM PRIORITY)**

**Rationale:**
- ALQ101 ("ever drinker") is weak
- ALQ120Q (frequency) might be better if available
- Current drinking more predictive than lifetime history

**Expected Impact:** +0.02-0.05 AUC (if variable available)

---

### **Experiment 4: Advanced Imputation (LOW PRIORITY)**

**Rationale:**
- 55% missing data in fasting labs
- Median imputation is simple but loses information
- MICE might capture feature relationships

**Expected Impact:** +0.00-0.01 AUC (probably minimal)  
**Risk:** Might actually hurt performance (overfit to imputed values)

---

## 📝 Reporting Strategy

### **Option A: Report Baseline v1 (Current Results)**

**Use if:** No improvements exceed v1.0 performance

**Message:** 
- "We achieved AUC 0.71 with modern gradient boosting"
- "96% recall demonstrates clinical screening utility"
- "Performance gap vs Bashir (0.95) reveals prediction challenges"
- "Honest, rigorous science"

---

### **Option B: Report Best Improved Version**

**Use if:** Improvements significantly beat v1.0 (e.g., AUC 0.75+)

**Message:**
- "Enhanced feature engineering improved AUC to 0.75"
- "Baseline v1 (0.71) → v2 with age groups (0.75)"
- "Still below Bashir (0.95), but more realistic"

---

### **Option C: Report Both (Recommended)** ⭐

**Structure:**
- **Main results:** Best performing version
- **Supplementary:** Baseline v1 and all experiments
- **Ablation study:** Show impact of each feature engineering step

**Benefits:**
- Full transparency
- Shows iterative scientific process
- Readers can reproduce any version

---

## 🏷️ Git Tags for Easy Rollback

```bash
# Tag current state
git tag -a v1.0-baseline -m "Baseline v1: 14 predictors, AUC 0.7071"

# Future: Tag each experiment
git tag -a v1.1-age-groups -m "Experiment: Age groups, AUC 0.72"
git tag -a v1.2-metabolic-score -m "Experiment: Metabolic syndrome, AUC 0.73"

# Rollback if needed
git checkout v1.0-baseline
```

---

## ✅ Checklist Before Next Experiment

- [x] Results documented in README
- [x] All figures saved (08-11 model comparison)
- [x] JSON results files saved
- [x] Notebook Sections 1-14 complete
- [x] Code committed to git
- [ ] Git tag created (v1.0-baseline)
- [ ] Feature engineering strategies defined
- [ ] Ready to experiment

---

## 🎯 Success Metrics for v2.0

**Minimum viable improvement:**
- AUC gain: ≥ 0.01 (1 percentage point)
- Statistical significance: p < 0.05 vs v1.0
- No loss in recall: Must maintain ≥ 95%

**Target improvement:**
- AUC gain: ≥ 0.03 (3 percentage points)
- New AUC: 0.74+ (crossing into "good" territory)

**Stretch goal:**
- AUC: 0.80+ (major improvement)
- Close Bashir gap significantly

---

## 📖 Lessons Learned

### What Worked:
1. ✅ Ordinal floss (1-5 days) >> binary
2. ✅ ALQ101 for alcohol (proper variance)
3. ✅ Optuna optimization (found good hyperparameters)
4. ✅ Stratified CV (fair evaluation)

### What Didn't Work:
1. ⚠️ Simple features alone can't reach AUC 0.90+
2. ⚠️ High missing data is a real limitation
3. ⚠️ Waist circumference redundant (excluded)

### What to Try Next:
1. 🔬 Age groups (non-linear effects)
2. 🔬 Metabolic syndrome composite score
3. 🔬 Feature interactions
4. 🔬 Better alcohol variable (ALQ120Q)
5. 🔬 Advanced imputation (MICE)

---

**This checkpoint ensures we can always return to these solid baseline results while exploring improvements!** ✅
