# 🔬 Feature Engineering v2.0 Strategy

**Current Performance (Baseline v1):** AUC 0.7071 (CatBoost)  
**Goal:** Improve to AUC ≥ 0.74 (+3 points minimum)  
**Approach:** Systematic feature engineering experiments

---

## 🎯 Prioritized Experiments (Ranked by Expected Impact)

### 🥇 **Experiment 1: Age Group Encoding + Polynomial Features**

**Priority:** ⭐⭐⭐ HIGH  
**Expected Impact:** +0.02-0.04 AUC  
**Rationale:** Age is strongest predictor (r=0.16) but relationship likely non-linear

**Implementation:**

```python
# A. Categorical age groups (clinical thresholds)
df['age_group'] = pd.cut(df['age'], 
                         bins=[30, 45, 55, 65, 80],
                         labels=['30-44', '45-54', '55-64', '65+'])
# One-hot encode
age_dummies = pd.get_dummies(df['age_group'], prefix='age')

# B. Polynomial features (capture U-shaped or accelerating effects)
df['age_squared'] = df['age'] ** 2
df['age_cubed'] = df['age'] ** 3

# C. Spline basis (smooth non-linear)
from sklearn.preprocessing import SplineTransformer
spline = SplineTransformer(n_knots=5, degree=3)
age_spline = spline.fit_transform(df[['age']])
```

**Comparison Plan:**
- Baseline v1 (linear age): AUC 0.7071
- v2a (age groups): AUC ?
- v2b (age polynomial): AUC ?
- v2c (age spline): AUC ?
→ Keep best version

---

### 🥈 **Experiment 2: Metabolic Syndrome Composite Score**

**Priority:** ⭐⭐ MEDIUM-HIGH  
**Expected Impact:** +0.01-0.03 AUC  
**Rationale:** Individual metabolic features weak, but combined might show synergy

**Implementation:**

```python
# Clinical metabolic syndrome criteria (AHA/NHLBI)
df['metabolic_syndrome_score'] = (
    (df['bmi'] >= 30).astype(int) +                      # Obesity
    (df['glucose'] >= 100).astype(int) +                 # Impaired fasting glucose
    (df['triglycerides'] >= 150).astype(int) +           # High triglycerides
    (df['hdl'] < 40).astype(int) +                       # Low HDL (males)
    ((df['systolic_bp'] >= 130) | 
     (df['diastolic_bp'] >= 85)).astype(int)             # Hypertension
)
# Score: 0-5 (ordinal)

# Binary version
df['has_metabolic_syndrome'] = (df['metabolic_syndrome_score'] >= 3).astype(int)
```

**Comparison Plan:**
- Baseline v1: AUC 0.7071
- v2 + metabolic_score (ordinal): AUC ?
- v2 + has_metabolic_syndrome (binary): AUC ?
→ Test which encoding works better

---

### 🥉 **Experiment 3: Feature Interactions (Age × Risk Factors)**

**Priority:** ⭐⭐ MEDIUM  
**Expected Impact:** +0.01-0.02 AUC  
**Rationale:** Certain risk factors amplify with age (e.g., older smokers)

**Implementation:**

```python
# High-priority interactions
df['age_smoking'] = df['age'] * df['smoking']
df['age_diabetes'] = df['age'] * (df['glucose'] >= 126).astype(int)
df['age_bmi'] = df['age'] * df['bmi']
df['bmi_diabetes'] = df['bmi'] * (df['glucose'] >= 126).astype(int)

# Oral health interactions
df['floss_visit'] = df['floss_days'] * df['dental_visit']
df['age_mobile_teeth'] = df['age'] * df['mobile_teeth']
```

**Risk:** Too many interactions might cause overfitting  
**Mitigation:** Use L1 regularization or feature selection

---

### 4️⃣ **Experiment 4: Better Alcohol Variable**

**Priority:** ⭐ MEDIUM-LOW  
**Expected Impact:** +0.02-0.04 AUC (IF variable available and better)  
**Rationale:** ALQ101 ("ever drinker") is weak; current drinking more relevant

**Implementation:**

```python
# Check if ALQ120Q (drinking frequency) available
if 'ALQ120Q' in df_full.columns:
    # ALQ120Q: 1=Daily, 2=Weekly, 3=Monthly, etc.
    # Create ordinal or binary from frequency
    df['drinks_frequently'] = df_full['ALQ120Q'].apply(
        lambda x: 1 if x in [1, 2] else (0 if x >= 3 else np.nan)
    )
    
# Or check ALQ130 distribution more carefully
# Maybe we can recover non-drinkers from missing pattern?
```

**Action:** Run Cell 12 from notebook to explore ALQ variables

---

### 5️⃣ **Experiment 5: Advanced Imputation (MICE)**

**Priority:** ⭐ LOW  
**Expected Impact:** +0.00-0.01 AUC (uncertain, might hurt)  
**Rationale:** 55% missing data; MICE uses feature relationships for imputation

**Implementation:**

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Replace SimpleImputer with IterativeImputer
mice_imputer = IterativeImputer(
    estimator=None,  # Uses BayesianRidge by default
    max_iter=10,
    random_state=42
)
```

**Risk:** 
- MICE can overfit to imputed values
- Slower training (10x longer)
- Might not improve performance

**Recommendation:** Try last (after other strategies)

---

### 6️⃣ **Experiment 6: Feature Selection**

**Priority:** ⭐ LOW  
**Expected Impact:** +0.00-0.01 AUC (simplification, not improvement)  
**Rationale:** Remove weak features to reduce noise

**Implementation:**

```python
# Use XGBoost to select features
from sklearn.feature_selection import SelectFromModel

selector = SelectFromModel(
    XGBClassifier(),
    threshold='median'  # Keep top 50% of features
)

# Or use SHAP values
# Keep only features with mean(|SHAP|) > threshold
```

**Risk:** Might remove features with weak marginal effects but strong interaction effects

---

## 📊 Experiment Execution Protocol

### For Each Experiment:

**Step 1: Create Branch**
```bash
git checkout -b experiment/age-groups
```

**Step 2: Implement Changes**
- Modify Section 6 (Build Predictors)
- Add new features to ALL_FEATURES list
- Update preprocessing pipeline if needed

**Step 3: Train Models**
- Re-run Sections 9-14
- Save results to `results/experiment_age_groups_*.json`

**Step 4: Compare to Baseline**
```python
baseline_auc = 0.7071
new_auc = catboost_scores['auc'].mean()
improvement = new_auc - baseline_auc

if improvement > 0.01 and p_value < 0.05:
    print("✅ Significant improvement! Keep changes.")
else:
    print("❌ No improvement. Revert to baseline.")
```

**Step 5: Decide**
- **If better:** Merge to main, create new tag (v1.1, v1.2, etc.)
- **If worse:** Discard branch, keep baseline v1.0

---

## 🔬 Ablation Study (For Final Paper)

**Goal:** Understand contribution of each improvement

```
Model 0 (Baseline v1):  AUC = 0.7071  [14 features, simple imputation]
↓
Model 1 (+ age groups):  AUC = 0.72?   [+4 features, Δ = +0.01]
↓
Model 2 (+ metabolic):   AUC = 0.73?   [+1 feature, Δ = +0.01]
↓
Model 3 (+ interactions): AUC = 0.74?  [+6 features, Δ = +0.01]
↓
Final Model:             AUC = 0.74    [25 features total]

Ablation Table:
| Model | Features Added | AUC | Δ from Previous |
|-------|----------------|-----|-----------------|
| v1.0  | Baseline (14)  | 0.7071 | - |
| v1.1  | Age groups     | 0.72   | +0.01 |
| v1.2  | Metabolic score| 0.73   | +0.01 |
| v1.3  | Interactions   | 0.74   | +0.01 |
```

---

## ⚠️ Overfitting Prevention

### Safeguards:

1. **Always use 5-fold CV** (never train/test split)
2. **Monitor train vs val AUC gap:**
   - Gap < 0.05: Good generalization ✅
   - Gap > 0.10: Overfitting ⚠️
3. **Feature count limit:** Max 25 features (avoid feature explosion)
4. **Regularization:** Increase reg_alpha, reg_lambda if overfitting
5. **Early stopping:** Already enabled in all models

---

## 📝 Documentation Requirements

### For Each Experiment:

1. **Create Markdown Cell:**
```markdown
## Experiment: Age Group Encoding

**Hypothesis:** Non-linear age effects improve prediction
**Changes:** Added age_30_44, age_45_54, age_55_64, age_65+ (one-hot)
**Expected:** +0.02 AUC
**Result:** AUC 0.72 (+0.01, p=0.03) ✅ Significant improvement
```

2. **Update Config:**
```yaml
experiments:
  v1.0_baseline:
    date: "2025-12-01"
    auc: 0.7071
    features: 14
    
  v1.1_age_groups:
    date: "2025-12-02"
    auc: 0.72
    features: 17
    improvement: +0.01
```

3. **Save Comparison Plot:**
```python
# Bar chart: Baseline vs Experiment
plt.bar(['v1.0 Baseline', 'v1.1 Age Groups'], 
        [0.7071, 0.72])
plt.title('Experiment 1: Impact of Age Group Encoding')
```

---

## 🎯 Decision Tree for Experiments

```
Start: Baseline v1.0 (AUC 0.7071)
│
├─ Try Experiment 1 (Age Groups)
│  ├─ If AUC > 0.72 → KEEP, tag v1.1, continue to Exp 2
│  └─ If AUC ≤ 0.71 → DISCARD, try Exp 2 from v1.0
│
├─ Try Experiment 2 (Metabolic Score)
│  ├─ If AUC > previous + 0.01 → KEEP, tag v1.2
│  └─ If AUC ≤ previous → DISCARD
│
├─ Try Experiment 3 (Interactions)
│  ├─ If AUC > previous + 0.01 → KEEP, tag v1.3
│  └─ If AUC ≤ previous → DISCARD
│
└─ Final: Report best version (v1.0, v1.1, v1.2, or v1.3)
```

---

## ✅ Ready to Experiment Checklist

- [x] Baseline v1.0 results saved
- [x] Git checkpoint created (pending tag)
- [x] Feature engineering strategies defined
- [x] Experiment protocol established
- [x] Success criteria clear (≥+0.01 AUC, p<0.05)
- [x] Rollback plan ready (git tags)
- [ ] Choose first experiment to try

---

**Recommendation:** Start with **Experiment 1 (Age Groups)** - highest expected impact, lowest risk! 🚀
