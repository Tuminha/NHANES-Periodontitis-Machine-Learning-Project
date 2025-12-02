# 🚀 Quick Reference Guide

**Current Version:** v1.0-baseline (AUC 0.7071)

---

## 📊 View Current Results

```bash
# Summary results
cat results/model_comparison_report.txt

# Detailed JSON
cat results/model_comparison_detailed.json

# Checkpoint document
cat BASELINE_V1_CHECKPOINT.md
```

---

## 🔄 Version Control Commands

### View all versions:
```bash
git tag -l
```

### Restore to Baseline v1.0:
```bash
git checkout v1.0-baseline
```

### Return to latest:
```bash
git checkout main
```

### Compare current to baseline:
```bash
git diff v1.0-baseline..main
```

---

## 🧪 Run New Experiment

### Step 1: Create experiment branch
```bash
git checkout -b experiment/age-groups
```

### Step 2: Implement changes
- Edit notebook Section 6 (Build Predictors)
- Add new features
- Re-run Sections 9-14

### Step 3: Compare results
```python
baseline_auc = 0.7071
new_auc = catboost_scores['auc'].mean()
print(f"Improvement: {new_auc - baseline_auc:+.4f}")
```

### Step 4: Decision
**If BETTER (≥+0.01 AUC):**
```bash
git checkout main
git merge experiment/age-groups
git tag -a v1.1-age-groups -m "Age groups: AUC 0.72"
```

**If WORSE:**
```bash
git checkout main
git branch -D experiment/age-groups  # Discard
```

---

## 📊 View All Figures

```bash
open figures/08_model_comparison_auc.png
open figures/09_model_comparison_metrics.png
open figures/10_model_comparison_boxplot.png
open figures/11_model_comparison_significance.png
```

---

## 🎯 Next Experiments (Priority Order)

1. **Age Groups** (HIGH)
   - Expected: +0.02 AUC
   - File: `FEATURE_ENGINEERING_V2_STRATEGY.md` (Experiment 1)

2. **Metabolic Syndrome Score** (MEDIUM)
   - Expected: +0.01 AUC
   - File: `FEATURE_ENGINEERING_V2_STRATEGY.md` (Experiment 2)

3. **Feature Interactions** (MEDIUM)
   - Expected: +0.01 AUC
   - File: `FEATURE_ENGINEERING_V2_STRATEGY.md` (Experiment 3)

---

## 📝 Push to GitHub (Manual)

Due to SSL certificate issue in sandbox, push manually:

```bash
# In your regular terminal (not Cursor sandbox)
cd /Users/franciscoteixeirabarbosa/Dropbox/Random_scripts/nhanes_periodontitis_ml

# Push commits
git push origin main

# Push tags
git push origin v1.0-baseline

# Or push all tags
git push origin --tags
```

---

## 🏷️ Tag Naming Convention

```
v1.0-baseline       # Current baseline
v1.1-age-groups     # First improvement (if successful)
v1.2-metabolic      # Second improvement
v1.3-interactions   # Third improvement
v2.0-final          # Best final version for paper
```

---

## ✅ Checklist Before Experimenting

- [x] Baseline results documented
- [x] Git committed and tagged (v1.0-baseline)
- [x] Checkpoint file created
- [x] Strategy document ready
- [x] Ready to experiment safely

**Status:** 🟢 Ready to improve! Start with Experiment 1 (Age Groups) 🚀
