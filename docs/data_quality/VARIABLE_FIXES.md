# 🔧 Variable Recoding Fixes

**Date:** December 1, 2025  
**Issue:** EDA revealed zero-variance problems in alcohol and floss variables

---

## 🚨 Problems Identified

### 1. **Alcohol (ALQ130) - Zero Variance**
**Symptom:** All non-missing values = 1 (100% drinkers)

**Root Cause:**
- ALQ130 = "Avg # alcoholic drinks/day - past 12 months"
- Values: 1-25+ (actual drinks per day)
- **NO zero values** because this question is only asked to people who answered "Yes" to ALQ101 ("Ever had 12+ drinks in lifetime?")
- Non-drinkers never get ALQ130 → missing values

**Original Code (WRONG):**
```python
df_features['alcohol'] = df_full['ALQ130'].apply(
    lambda x: 1 if x > 0 else (0 if x == 0 else np.nan)
)
# Result: Everyone with ALQ130 data = 1 (drinker)
```

**Fix Applied:**
```python
# Use ALQ101 instead (true binary variable)
df_features['alcohol'] = df_full['ALQ101'].apply(
    lambda x: 1 if x == 1 else (0 if x == 2 else np.nan)
)
# ALQ101: 1=Yes (ever had 12+ drinks), 2=No (never)
```

---

### 2. **Floss (OHQ620) - Near-Zero Variance**
**Symptom:** All non-missing values = 1 (100% floss users)

**Root Cause:**
- OHQ620 = "Days used floss/dental device past 7 days"
- Raw values: 1, 2, 3, 4, 5, 7, 9
- **NO zero values** in the data
- Special codes: 7 = Refused, 9 = Don't know
- Most common value: 5 (5,581 responses - suggesting "5+ days" or "always")

**Original Code (WRONG):**
```python
df_features['uses_floss'] = df_full['OHQ620'].apply(
    lambda x: 1 if x >= 1 else (0 if x == 0 else np.nan)
)
# Result: Everyone with OHQ620 data ≥ 1 → all = 1
```

**Fix Applied:**
```python
# Treat special codes (7, 9) as missing
df_features['uses_floss'] = df_full['OHQ620'].apply(
    lambda x: 1 if (pd.notna(x) and x <= 5) else (np.nan if pd.isna(x) or x >= 7 else 0)
)
# 1-5 = uses floss (1), 7/9 = missing, no 0 values in data
```

**Note:** This variable may still have low variance due to skip pattern or social desirability bias. May need to be excluded from modeling.

---

## 📋 Changes Made

| File | Change | Reason |
|------|--------|--------|
| `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` | Cell 12: Added variable exploration | Investigate ALQ/OHQ variables |
| `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` | Cell 15: Changed `ALQ130` → `ALQ101` | Fix zero-variance alcohol |
| `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` | Cell 15: Updated OHQ620 logic | Handle special codes properly |
| `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` | Cell 15: Added variance check | Verify binary variables have variance |
| `notebooks/00_nhanes_periodontitis_end_to_end.ipynb` | Cell 13: Updated `bashir_vars` | Use ALQ101 instead of ALQ130 |
| `configs/config.yaml` | Updated alcohol predictor | Document ALQ101 vs ALQ130 |

---

## ✅ Expected Outcome

**Before Fix:**
- `alcohol`: 100% = 1 (zero variance) ❌
- `uses_floss`: 100% = 1 (zero variance) ❌

**After Fix:**
- `alcohol`: Mix of 0/1 values (proper variance) ✅
- `uses_floss`: Mostly 1, but special codes handled ⚠️

---

## ⚠️ Remaining Concerns

### **Floss Variable:**
- May still have very low variance (most people report flossing)
- Could be affected by:
  1. Social desirability bias (over-reporting)
  2. Skip pattern (only flossers answer)
  3. Question wording (knowledge vs. behavior)

**Decision Point:** 
If variance remains < 0.01 after fixing, consider **excluding from modeling**.

---

## 🔍 Verification Steps

Run these checks after re-running Section 6:

```python
# 1. Check alcohol variance
print(df_features['alcohol'].value_counts())
print(f"Variance: {df_features['alcohol'].var():.3f}")
# Expected: ~0.25 (balanced binary)

# 2. Check floss variance
print(df_features['uses_floss'].value_counts())
print(f"Variance: {df_features['uses_floss'].var():.3f}")
# Expected: > 0.01 (at least some variance)

# 3. Verify no 100% values
for var in ['alcohol', 'uses_floss']:
    prop_ones = (df_features[var] == 1).mean()
    if prop_ones > 0.99:
        print(f"⚠️ {var}: {prop_ones:.1%} are 1 (near-zero variance)")
```

---

## 📖 NHANES Codebook References

- **ALQ101:** "Ever had 12+ drinks in lifetime?" (1=Yes, 2=No)
- **ALQ130:** "Avg # alcoholic drinks/day - past 12 months" (continuous, only for drinkers)
- **OHQ620:** "Days used floss/dental device past 7 days" (0-7, but 7/9 are special codes)

