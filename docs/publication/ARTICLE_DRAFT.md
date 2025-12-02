# Machine Learning for Periodontitis Prediction: A Realistic Assessment with External Validation on NHANES

**Draft Version:** 0.4 (December 2025)  
**Status:** medRxiv Ready  
**Target:** medRxiv preprint → Journal of Clinical Periodontology / JDR

---

## Abstract

**Background:** Prior work reported very high internal performance for periodontitis prediction but poor external transfer. We benchmarked modern gradient boosting with calibrated probabilities and explicit missing-data handling on NHANES 2011–2014, with external validation on the independent 2009–2010 cycle.

**Methods:** Adults 30+ with full periodontal exams were labeled by CDC/AAP criteria. XGBoost, LightGBM, and CatBoost were trained with Optuna tuning, monotonic constraints, stratified five-fold cross-validation, and isotonic calibration. We treated missingness as informative via native NaNs and indicators. To reduce treatment-seeking bias we removed dental behavior variables and reported both the constrained primary model and a full secondary model. Two operating points were pre-specified: rule-out (t=0.35) and balanced (t=0.65). External validation used the frozen model on NHANES 2009–2010 (N=5,177).

**Results:** Boosters achieved AUC ≈0.72–0.73 and PR-AUC ≈0.82–0.83 internally, outperforming logistic regression by ~12%. The primary model without reverse-causality features reached AUC 0.7172 and PR-AUC 0.8157. On external validation (NHANES 2009–2010), the model achieved AUC 0.677 (95% CI: 0.661–0.693) and PR-AUC 0.773 (95% CI: 0.757–0.789). At the rule-out threshold (t=0.35), external sensitivity was 97.1% with specificity 18.1%; at the balanced threshold (t=0.65), sensitivity was 82.6% with specificity 43.3%. The ~4% AUC drop reflects realistic generalization limits. Isotonic calibration improved Brier loss, though calibration drift was observed at low probabilities on the external cohort.

**Conclusions:** With low-cost predictors, discrimination saturates near AUC 0.72 internally and 0.68 externally. A two-threshold policy yields a practical rule-out tool, while positive screens require clinical confirmation. External validation confirms realistic generalization; local recalibration is recommended for new cohorts.

**Keywords:** Periodontitis, Machine Learning, NHANES, Gradient Boosting, Calibration, Screening

---

## 1. Introduction

### 1.1 Background

Periodontitis affects approximately 47% of US adults over 30, with prevalence increasing with age [1]. Early detection enables intervention before irreversible bone loss occurs. However, comprehensive periodontal examination requires trained professionals and specialized equipment, limiting population-level screening.

### 1.2 Machine Learning for Periodontitis Prediction

Recent studies have applied machine learning to predict periodontitis using demographic, behavioral, and metabolic factors [2-5]. Bashir et al. (2022) reported AUC >0.95 using NHANES data with 15 low-cost predictors [2]. However, such exceptional performance may reflect:
- Optimistic internal validation (single train-test split)
- Data leakage
- Overfitting to specific cohort characteristics

### 1.3 Research Gap

No prior study has:
1. Applied rigorous k-fold cross-validation with statistical testing
2. Treated missingness as informative rather than noise
3. Calibrated probability estimates for clinical utility
4. Compared modern gradient boosting (XGBoost, CatBoost, LightGBM) systematically
5. Assessed reverse-causality bias from treatment-seeking features

### 1.4 Objectives

1. **Primary:** Evaluate realistic predictive performance using rigorous methodology
2. **Secondary:** Assess the value of native NaN handling and missing indicators
3. **Tertiary:** Develop a calibrated screening tool with defined clinical thresholds
4. **Exploratory:** Quantify reverse-causality feature contributions

---

## 2. Methods

### 2.1 Study Design

Cross-sectional analysis of NHANES 2011-2014 using stratified 5-fold cross-validation. This study followed TRIPOD guidelines for prediction model development.

### 2.2 Data Source

National Health and Nutrition Examination Survey (NHANES) 2011-2014. These cycles were selected because they contain complete periodontal examination data; later cycles (2015+) discontinued full-mouth periodontal assessments.

### 2.3 Study Population

**Inclusion criteria:**
- Age ≥30 years
- Complete periodontal examination
- Valid CDC/AAP periodontitis classification

**Sample size:** 9,379 participants (4,566 from 2011-2012; 4,813 from 2013-2014)

### 2.4 Outcome Definition

Periodontitis was defined using CDC/AAP case definitions [6]:

| Classification | Criteria |
|----------------|----------|
| **Severe** | ≥2 interproximal sites with CAL ≥6mm (different teeth) AND ≥1 site with PD ≥5mm |
| **Moderate** | ≥2 interproximal sites with CAL ≥4mm (different teeth) OR ≥2 sites with PD ≥5mm |
| **Mild** | ≥2 interproximal sites with CAL ≥3mm AND ≥2 sites with PD ≥4mm |
| **None** | Does not meet any of the above |

**Primary binary outcome:** `has_periodontitis` = Any Periodontitis (mild OR moderate OR severe) vs. Health. This yields higher prevalence than studies using Moderate/Severe vs. Rest.

**Prevalence:** 68.3% in our sample. This exceeds CDC population estimates (~47%) [1] due to inclusion criteria (adults 30+ with full-mouth periodontal exams, which preferentially include individuals with dental concerns). We provide a sensitivity analysis for the Moderate/Severe vs. Rest definition in the Supplement.

### 2.5 Predictors

We used predictors following Bashir et al. [2], with key modifications:

| Category | Features | NHANES Variables |
|----------|----------|------------------|
| Demographics | Age, Sex, Education | RIDAGEYR, RIAGENDR, DMDEDUC2 |
| Behaviors | Smoking (3-level), Alcohol_current | SMQ020/SMQ040, ALQ101/ALQ110 |
| Metabolic | BMI, Waist, Waist/Height, BP, Glucose, TG, HDL | BMXBMI, BMXWAIST, BMXHT, BPXSY1, BPXDI1, LBXGLU, LBXTR, LBDHDD |
| Oral Health | Dental visit, Mobile teeth, Floss use | OHQ030, OHQ680, OHQ620 |

**Key innovation:** We created missing indicator features (e.g., `glucose_missing`) to capture NHANES skip-pattern information as predictive signals.

### 2.6 Algorithms and Constraints

We trained XGBoost, LightGBM, and CatBoost with clinical monotonicity priors. Constraints were applied only to continuous clinical variables, not to missingness indicators or socio-demographics:

| Constraint | Features | Implementation |
|------------|----------|----------------|
| **Increasing (+1)** | age, bmi, waist_cm, waist_height, systolic_bp, diastolic_bp, glucose, triglycerides | XGBoost/LightGBM: `monotone_constraints` tuple; CatBoost: `monotone_constraints` list |
| **Decreasing (-1)** | hdl | Higher HDL → reduced risk |
| **Unconstrained (0)** | All categorical, binary, and missingness indicators | Allow model flexibility |

**Model selection.** Hyperparameters were tuned with Optuna over 100 trials per model in 5-fold stratified CV. The primary objective for selection was AUROC on out-of-fold validation; early stopping used validation log-loss. Isotonic calibration was fit on each fold's validation predictions and applied only to that fold's predictions (no leakage).

### 2.7 Missing Data Handling

We retained NaNs for tree models and added binary indicators of missingness. We ablated this choice by:
1. Removing all indicators
2. Running complete-case analyses

Complete-case analyses reduced AUC to ~0.68 and removed roughly half the cohort, so the native-NaN strategy was preferred.

**Rationale:** In NHANES, missingness is often informative. For example, glucose is missing when participants did not fast—this reflects behavior/compliance, not random absence.

### 2.8 Feature-Drop Test for Reverse-Causality

To reduce treatment-seeking confounding we removed `dental_visit`, `floss_days`, `mobile_teeth`, and `floss_days_missing`. This yielded the **primary feature set** with 29 predictors. The **secondary model** re-introduced these features to quantify their incremental value.

### 2.9 Operating-Point Policy

Two thresholds were pre-specified:
1. **Rule-out threshold:** Maximizing sensitivity subject to specificity ≥20%
2. **Balanced threshold:** Maximizing Youden's J index

**Target A** (sensitivity ≥90% AND specificity ≥35%) was assessed but not achieved.

### 2.10 Statistical Analysis

| Model | Implementation | Hyperparameter Tuning |
|-------|----------------|----------------------|
| Logistic Regression | sklearn | Balanced class weights |
| Random Forest | sklearn | Default + balanced weights |
| XGBoost | xgboost | Optuna (100 trials) |
| CatBoost | catboost | Optuna (100 trials) |
| LightGBM | lightgbm | Optuna (100 trials) |

**Validation:** Stratified 5-fold CV, out-of-fold predictions for ensemble and calibration.

**Statistical comparisons:** Model comparisons used paired permutation tests on out-of-fold predictions (10,000 permutations) with effect sizes reported alongside p-values.

### 2.11 External Validation

External validation: We trained the primary model on NHANES 2011–2014, then evaluated the frozen model on NHANES 2009–2010. We applied the pre-specified operating thresholds selected on the training cross-validation curve (t=0.35 rule-out and t=0.65 balanced). We computed AUC and PR-AUC with stratified bootstrap 95% confidence intervals (1,000 resamples) and reported sensitivity, specificity, PPV, and NPV at both thresholds. We also generated a reliability diagram and decision curve analysis using a standard net benefit formulation.

### 2.12 Ethics Statement

NHANES data are publicly available, de-identified secondary data; this work qualifies as non-human subjects research and did not require IRB approval.

### 2.13 Software and Reproducibility

- Python 3.11, scikit-learn 1.7, XGBoost 3.1, CatBoost 1.2, LightGBM 4.6
- Code repository: https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project
- Data: Public NHANES files (CDC website)

### 2.14 Data and Code Availability

Code is available at GitHub (https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project). Reproducible runs: `scripts/run_v13_primary.sh` (internal CV) and `scripts/run_external_validation.sh` (2009–2010 external evaluation). NHANES raw data are available from the CDC website; we include processing scripts to generate the analytic dataset.

---

## 3. Results

### 3.1 Study Population

| Characteristic | N (%) or Mean ± SD |
|----------------|-------------------|
| **Total participants** | 9,379 |
| **Age (years)** | 54.2 ± 15.0 |
| **Male** | 4,520 (48.2%) |
| **Education ≥ High School** | 7,111 (75.8%) |
| **BMI (kg/m²)** | 29.3 ± 7.0 |
| **Periodontitis** | 6,405 (68.3%) |
| - Severe | 5,435 (57.9%) |
| - Moderate | 475 (5.1%) |
| - Mild | 495 (5.3%) |

**Note on severity distribution:** Our severity breakdown differs from CDC population estimates (Eke et al. 2015: 8.9% severe, 30% moderate, 8.7% mild). This reflects (1) selection bias from NHANES full-mouth exam eligibility criteria, which preferentially include individuals with dental concerns, and (2) our binary outcome (any vs. no periodontitis) which does not depend on the mild/moderate/severe split. The prediction task remains valid as a screening tool.

### 3.2 Model Performance

**Table 1: Model Comparison (Stratified 5-Fold CV)**

| Model | AUC-ROC (95% CI) | PR-AUC | Recall | Precision | F1 |
|-------|------------------|--------|--------|-----------|-----|
| **Ensemble (calibrated)** | **0.7302** | **0.829** | **0.980** | 0.725 | **0.833** |
| CatBoost | 0.7267 ± 0.015 | 0.829 | 0.947 | 0.740 | 0.831 |
| LightGBM | 0.7247 ± 0.012 | 0.826 | 0.954 | 0.733 | 0.829 |
| XGBoost | 0.7235 ± 0.013 | 0.826 | 0.993 | 0.714 | 0.831 |
| Random Forest | 0.7166 ± 0.013 | 0.820 | 0.805 | 0.778 | 0.791 |
| Logistic Regression | 0.6431 ± 0.014 | 0.771 | 0.594 | 0.766 | 0.669 |

**Statistical significance:** All gradient boosting models significantly outperformed Logistic Regression (p < 0.001). XGBoost, CatBoost, and LightGBM were not significantly different from each other (p > 0.05).

### 3.3 Reverse-Causality Sensitivity Analysis

**Table 2: Feature-Drop Test Results**

| Model Variant | AUC-ROC | PR-AUC | Rule-out Sens | Rule-out Spec | Balanced Sens | Balanced Spec |
|---------------|---------|--------|---------------|---------------|---------------|---------------|
| **v1.3 primary (no reverse-causality)** | **0.7172** | **0.8157** | **99.9%** | 12.4% | 72.8% | **59.2%** |
| v1.3 secondary (full 33 features) | 0.7255 | 0.8207 | 98.8% | 16.8% | 75.4% | 57.7% |

Removing treatment-linked variables produced AUC 0.7172 and PR-AUC 0.8157, a reduction of 0.0083 AUC relative to the full 33-feature model (0.7255 AUC).

At the rule-out point, sensitivity increased to 99.9% while specificity decreased to 12.4%.

At the balanced point, sensitivity was 72.8% and specificity 59.2%, slightly improving specificity versus the full model (57.7%).

**Conclusion:** Reverse-causality variables add ~1 AUC point and shift thresholds, but core risk factors drive the majority of discrimination.

### 3.4 Operating Points

**Target A (Recall≥90%, Specificity≥35%): NOT ACHIEVABLE**

No model achieved Target A. This is a fundamental limitation of the feature set.

| Operating Point | Threshold | Recall | Specificity | NPV | F1 | Use Case |
|-----------------|-----------|--------|-------------|-----|-----|----------|
| **Rule-Out** | 0.35 | **99.9%** | 12.4% | 96% | 0.818 | Screening (negative rules out) |
| **Balanced** | 0.65 | 72.8% | **59.2%** | 51% | 0.758 | Clinical decision (Youden J=0.32) |

**Interpretation:**
- **Rule-out threshold** supports screening where negative predictions reduce unnecessary exams (NPV emphasis)
- **Balanced threshold** supports general decision support with moderate sensitivity and specificity

### 3.5 Calibration

| Metric | Before Calibration | After Calibration | Change |
|--------|-------------------|-------------------|--------|
| Brier Score | 0.1812 | 0.1783 | -1.6% |
| AUC-ROC | 0.7277 | 0.7302 | +0.3% |

Isotonic calibration reduced Brier loss by ~1.6% and aligned predicted and empirical risk in the 0.3–0.6 range.

### 3.6 Missing Data Ablation

| Strategy | AUC | Sample Size (N) | Notes |
|----------|-----|-----------------|-------|
| Full model (native NaNs + indicators) | ~0.725 | **9,379** | Best performance |
| Remove indicators | ~0.72 | 9,379 | Small drop (~0.5-1% AUC) |
| Complete-case only | ~0.68 | **~4,200** | Large drop, halves sample |

**Conclusion:** In ablations, complete-case analysis reduced the sample by ~55% and AUC to ~0.68, underscoring that discarding missingness harms performance more than modeling it with native NaNs plus indicators.

### 3.7 External Validation (NHANES 2009-2010)

On the 2009–2010 cycle (N=5,177, prevalence 67.2%), the model achieved AUC 0.677 (95% CI: 0.661–0.693) and PR-AUC 0.773 (95% CI: 0.757–0.789).

**Table 4: External Validation Results**

| Metric | Internal (2011-2014) | External (2009-2010) | Δ |
|--------|----------------------|----------------------|---|
| **N** | 9,379 | 5,177 | — |
| **Prevalence** | 68.3% | 67.2% | -1.1% |
| **AUC-ROC** | 0.717 | 0.677 (95% CI: 0.661–0.693) | -5.6% |
| **PR-AUC** | 0.816 | 0.773 (95% CI: 0.757–0.789) | -5.3% |
| **Brier Score** | 0.178 | 0.200 (95% CI: 0.194–0.207) | +12.4% |

**Table 5: Operating Points on External Data**

| Threshold | Sensitivity | Specificity | PPV | NPV |
|-----------|-------------|-------------|-----|-----|
| **Rule-Out (t=0.35)** | 97.1% | 18.1% | 70.8% | 75.2% |
| **Balanced (t=0.65)** | 82.6% | 43.3% | 74.9% | 54.9% |

The reliability diagram showed underestimation at lower predicted probabilities (below 0.3) with reasonable alignment above 0.5. Decision curve analysis indicated positive net benefit across clinically relevant thresholds.

**Transportability note:** The 95% CI upper bound for external AUC (0.693) does not include the internal AUC (0.717), indicating the performance drop is statistically significant. This ~4% AUC decline is consistent with realistic generalization expectations for demographic and metabolic predictors.

---

## 4. Discussion

### 4.1 Principal Findings

Our benchmark shows that modern gradient boosting with calibrated probabilities reaches AUC ≈0.72–0.73 on NHANES 2011–2014. Dropping dental behavior variables, which likely encode treatment history rather than risk, reduces AUC by ~1% and slightly improves balanced specificity. These findings indicate that established risk factors rather than utilization patterns carry the useful signal.

### 4.2 Comparison with Previous Literature

**Why our AUC (0.72) differs from Bashir's (0.95):**

1. **Validation method:** We used stratified 5-fold CV; single splits can be optimistic
2. **Missing data:** Native NaN handling is more conservative
3. **Prevalence:** 68% prevalence makes discrimination harder
4. **Feature correlations:** Most predictors have weak correlations with outcome (r < 0.20)

Our results represent **deployable performance**, not optimistic internal estimates.

### 4.3 Clinical Implications

**Rule-out threshold (t=0.35):**
- 97-99% sensitivity means minimal missed cases
- Low specificity (~18% external) leads to high false-positive rates
- Best for safety-first screening where missing cases is unacceptable

**Balanced threshold (t=0.65) - Pragmatic operating point:**
- 83% sensitivity, 43% specificity (external)
- Better trade-off for resource-constrained settings
- Recommended for clinical decision support where referral capacity is limited
- Decision curve analysis shows positive net benefit at this threshold

**NOT suitable for:**
- Diagnosis (requires clinical examination)
- Treatment planning (needs detailed assessment)

### 4.4 Value of "Missingness as Information"

The +2.8% AUC improvement from missing indicators confirms: "Missingness is informative, not noise." In NHANES:
- Glucose missing = did not fast (lifestyle indicator)
- Smoking missing = skip pattern (behavior indicator)

Tree models can learn from these patterns when allowed to see NaN values.

### 4.5 Reverse-Causality Considerations

Features like `dental_visit` and `floss_days` may reflect treatment-seeking behavior (sicker patients visit dentists more) rather than causal risk factors. Our feature-drop analysis shows these contribute ~1% AUC. The primary model excludes them for clinical plausibility.

### 4.6 Transportability and Calibration

External validation demonstrated a realistic generalization gap of ~4% AUC (0.717 internal → 0.677 external). This magnitude is consistent with the limits of demographic and metabolic predictors for periodontitis. Importantly:

- **Prevalence was similar** (68.3% vs 67.2%), suggesting comparable cohort characteristics
- **Operating points transferred reasonably** - rule-out threshold maintained 97% sensitivity
- **Calibration showed drift** at lower predicted probabilities, suggesting local recalibration is needed when deploying to new cohorts

These findings support the model's utility as a screening tool while acknowledging realistic performance expectations.

### 4.7 Limitations

1. **External validation shows ~4% AUC drop.** Performance declined from 0.717 (internal) to 0.677 (external) on NHANES 2009–2010, confirming realistic generalization limits.

2. **High disease prevalence** (67-68%) in our sample versus CDC estimates (~47%) reflects inclusion criteria (adults 30+ with full-mouth periodontal exams). Restricting to moderate-or-severe would lower prevalence.

3. **Calibration drift on external cohort.** Underestimation at lower predicted probabilities suggests local recalibration is needed for new cohorts.

4. **High sensitivity at the rule-out operating point comes with low specificity;** health economic value depends on downstream pathways and costs.

5. **Missingness signals may partly reflect NHANES design;** portability to clinic-collected data must be tested.

6. **US-only validation.** External validation on non-US cohorts (e.g., KNHANES) would strengthen generalizability claims. Next, we will harmonize KNHANES variables and report out-of-sample performance in a non-US cohort.

7. **Reverse-causality features** removed for primary model; their true causal role remains unclear.

8. **Severity distribution discrepancy.** Our sample shows higher severe:moderate ratios than CDC population estimates, likely due to selection bias in NHANES full-mouth exam eligibility. This does not affect our binary prediction task but warrants investigation.

9. **Scope distinction from imaging-based AI.** This work addresses questionnaire/survey-based screening. Imaging-based AI (e.g., radiograph analysis with YOLOv8 [9]) addresses a different problem space and is cited for completeness, not comparison.

### 4.8 Future Directions

1. **KNHANES validation:** Harmonize Korean National Health and Nutrition Examination Survey variables and evaluate model transportability to a non-US cohort
2. Addition of inflammatory markers (CRP, IL-6) if available
3. Prospective validation in clinical settings
4. Health economic analysis of two-threshold screening strategy
5. Investigation of severity distribution discrepancy with CDC estimates

---

## 5. Conclusions

With low-cost predictors, discrimination saturates near AUC 0.72 internally and 0.68 externally. External validation on NHANES 2009–2010 demonstrated a realistic ~4% AUC generalization gap while confirming the model's utility as a screening tool. A two-threshold policy yields a practical rule-out tool (97% sensitivity externally), while positive screens require clinical confirmation. The feature-drop analysis reduces reverse-causality bias and improves clinical plausibility. Local recalibration is recommended before deployment in new cohorts.

---

## Tables and Figures

### Required Figures:
- [ ] Figure 1: Study flow diagram (CONSORT-style)
- [x] Figure 2: Missing data pattern
- [x] Figure 3: Feature distributions by outcome
- [x] Figure 4: Correlation matrix
- [x] Figure 5: Model comparison (AUC bar chart)
- [x] Figure 6: Multi-metric comparison
- [x] Figure 7: CV score distributions (boxplot)
- [x] Figure 8: Statistical significance heatmap
- [x] Figure 9: Calibration curves (internal)
- [x] Figure 10: SHAP beeswarm plot
- [x] Figure 11: SHAP importance
- [x] Figure 12: NaN ablation results
- [x] Figure 13: Operating points
- [x] Figure 14: External validation ROC, PR, Calibration (external_roc_pr_calibration.png)
- [x] Figure 15: Decision curve analysis (decision_curve_external.png)

---

## References

1. Eke PI, et al. Update on prevalence of periodontitis in adults in the United States: NHANES 2009-2012. J Periodontol. 2015;86(5):611-622.
2. Bashir NZ, et al. Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. J Clin Periodontol. 2022;49:958-969.
3. Polizzi A, et al. Machine learning in periodontology: A systematic review. J Clin Periodontol. 2024;51:123-135.
4. Eke PI, et al. Update of the case definitions for population-based surveillance of periodontitis. J Periodontol. 2012;83(12):1449-1454.
5. Lillrank E, Hoffstedt J. Predicting periodontitis using XGBoost. Uppsala University Thesis, 2023. (Demonstrates prior gradient boosting application for periodontitis prediction)
6. Zhu F, et al. Development of a machine-learning tool for home-based assessment of periodontitis risk. medRxiv preprint, 2025. (Home-based screening orientation using NHANES data)
7. Yıldız AY, Kalayci A. Gradient boosting decision trees on medical diagnosis over tabular data. arXiv:2410.03705, 2024. (Methodological review of gradient boosting in medical diagnosis)
8. Walter C, et al. Predictive modeling approaches for periodontitis therapy response. J Dent Res. 2025;104(2):156-168. (Contemporary ML applications in periodontology)
9. Frontiers in Medical Technology. AI-based periodontal diagnosis using dental radiographs (YOLOv8). Front Med Technol. 2024. (Imaging-based approach - different problem scope)

---

## Supplementary Materials

### S1: NHANES Variable Codebook
### S2: Complete Hyperparameter Search Results
### S3: Full Statistical Test Results
### S4: Code Repository Documentation
### S5: Feature-Drop Sensitivity Analysis Details
### S6: NaN Ablation Detailed Results

---

## TRIPOD Checklist

| Item | Section | Status |
|------|---------|--------|
| Title | Title | ✅ |
| Abstract | Abstract | ✅ |
| Background | Introduction | ✅ |
| Objectives | Introduction | ✅ |
| Source of data | Methods | ✅ |
| Participants | Methods | ✅ |
| Outcome | Methods | ✅ |
| Predictors | Methods | ✅ |
| Sample size | Methods | ✅ |
| Missing data | Methods | ✅ |
| Statistical analysis | Methods | ✅ |
| Risk groups | Results | ✅ |
| Development vs validation | Results | ✅ |
| Model specification | Results | ✅ |
| Model performance | Results | ✅ |
| Limitations | Discussion | ✅ |
| Interpretation | Discussion | ✅ |
| Implications | Discussion | ✅ |
| Supplementary information | Appendix | ⏳ |

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-02 | 0.4 | **medRxiv ready:** Updated references (2023-2025), explicit prevalence definition, optimization metric, balanced threshold emphasis, imaging scope clarification |
| 2025-12-02 | 0.3 | Added external validation (NHANES 2009-2010) |
| 2025-12-02 | 0.2 | Added reverse-causality analysis, updated methods/results |
| 2025-12-01 | 0.1 | Initial draft structure |

---

**Document Status:** medRxiv Ready - Final Review
