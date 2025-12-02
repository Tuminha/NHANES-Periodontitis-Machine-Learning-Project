# Re-evaluating Machine Learning Benchmarks for Periodontitis Prediction: The Importance of Calibration, Rigorous Validation, and Missing Data Handling

**Target Journal:** Journal of Clinical Periodontology (JCP) / Journal of Periodontology / JDR  
**Status:** Draft v0.1 (December 2025)  
**Type:** Original Research / Methodological Study

---

## Abstract (Structured - 300 words max)

**Aim:** To evaluate the realistic predictive performance of modern gradient boosting methods for periodontitis screening using rigorous cross-validation, probability calibration, and external validation on NHANES data.

**Materials and Methods:** We analyzed 9,379 US adults aged ≥30 from NHANES 2011-2014 with complete periodontal examinations (CDC/AAP case definitions). Five machine learning models (Logistic Regression, Random Forest, XGBoost, CatBoost, LightGBM) were compared using stratified 5-fold cross-validation with Optuna hyperparameter optimization. Key methodological innovations included: (1) native handling of missing values as informative features, (2) isotonic probability calibration, and (3) external validation on NHANES 2009-2010. We performed ablation experiments to distinguish genuine predictive signal from survey design artifacts.

**Results:** The calibrated soft-voting ensemble achieved AUC 0.7302 (95% CI: X.XX-X.XX) with 97.97% recall at the optimized threshold. Modern gradient boosting significantly outperformed logistic regression (Δ AUC = +0.087, p<0.001). Missing data indicators contributed meaningfully to prediction (Δ AUC = +0.020), though ablation experiments revealed [X% of the signal persisted in complete-case analysis / was partially attributable to survey design]. External validation on NHANES 2009-2010 yielded AUC X.XX, demonstrating [robustness / need for recalibration]. At the clinical operating point (recall ≥90%, specificity ≥35%), the model achieved F1 = X.XX.

**Conclusions:** Gradient boosting with calibration provides a realistic periodontitis screening tool with AUC ~0.73-0.75. Performance is substantially lower than previously reported internal validation results (~0.95), suggesting prior studies may have been overfit. The gap between internal and external validation underscores the importance of rigorous methodology for clinical translation.

**Keywords:** Periodontitis, Machine Learning, NHANES, Gradient Boosting, Calibration, External Validation, Missing Data

---

## 1. Introduction

### 1.1 Clinical Context

Periodontitis is a chronic inflammatory disease affecting approximately 47% of US adults over 30 years old, with prevalence increasing to 70% in adults over 65 [Eke et al., 2015]. The disease progresses silently, often without symptoms until significant bone loss has occurred. Early identification of at-risk individuals through population-level screening could enable timely intervention and reduce the substantial morbidity and healthcare costs associated with advanced periodontitis.

### 1.2 Machine Learning for Periodontitis Prediction

Recent studies have applied machine learning (ML) to predict periodontitis using demographic, behavioral, and metabolic factors available without specialized dental examination [Bashir et al., 2022; List others]. These studies report impressive performance metrics, with some achieving area under the receiver operating characteristic curve (AUC) exceeding 0.95 using data from the National Health and Nutrition Examination Survey (NHANES).

However, several methodological concerns limit the clinical applicability of these findings:

1. **Validation Strategy:** Many studies use single train-test splits, which can produce optimistic performance estimates due to random favorable partitioning.

2. **Probability Calibration:** Raw ML predictions are often miscalibrated, meaning a predicted probability of 0.70 does not correspond to 70% observed risk. This limits clinical utility for shared decision-making.

3. **Missing Data Handling:** NHANES data contain substantial missingness due to skip-pattern questionnaires and fasting subsample selection. Simple imputation may discard predictive information, while native handling may learn survey artifacts rather than biological signal.

4. **External Validation:** Cross-validation on a single dataset does not guarantee generalization to other populations or time periods.

### 1.3 Research Gap

No prior study has:
1. Systematically compared modern gradient boosting methods (XGBoost, CatBoost, LightGBM) for periodontitis prediction
2. Applied probability calibration for clinical utility
3. Investigated whether missing data patterns contain genuine predictive signal or survey artifacts
4. Provided external temporal validation across NHANES cycles

### 1.4 Objectives

**Primary:** Evaluate realistic predictive performance of gradient boosting methods using rigorous cross-validation and external validation.

**Secondary:** 
- Assess the contribution of missing data indicators to prediction
- Develop calibrated probability estimates for clinical decision-making
- Establish clinically relevant operating thresholds balancing sensitivity and specificity

---

## 2. Materials and Methods

### 2.1 Study Design and TRIPOD Adherence

This study follows the Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis (TRIPOD) guidelines [Collins et al., 2015]. We report a prediction model development study with internal validation (stratified 5-fold cross-validation) and external temporal validation.

**Ethical Considerations:** NHANES data are publicly available and de-identified. This secondary data analysis did not require institutional review board approval.

### 2.2 Data Source

We used data from the National Health and Nutrition Examination Survey (NHANES), a nationally representative cross-sectional survey of the US civilian non-institutionalized population conducted by the National Center for Health Statistics.

**Development Cohort:** NHANES 2011-2012 and 2013-2014 cycles (n = 9,379)
**External Validation Cohort:** NHANES 2009-2010 cycle (n = TBD)

**Rationale for Cycle Selection:** NHANES discontinued full-mouth periodontal examinations after 2013-2014. The 2015-2018 cycles lack the probing depth (PD) and clinical attachment loss (CAL) measurements required for CDC/AAP periodontitis classification.

### 2.3 Study Population

**Inclusion Criteria:**
- Age ≥ 30 years (consistent with CDC/AAP case definition requirements)
- Complete periodontal examination with valid PD and CAL measurements
- Valid assignment of CDC/AAP periodontitis classification

**Exclusion Criteria:**
- Edentulous individuals
- Incomplete periodontal examination

### 2.4 Outcome Definition

Periodontitis was defined using the CDC/AAP case definitions [Eke et al., 2012]:

| Classification | Definition |
|----------------|------------|
| **Severe** | ≥2 interproximal sites with CAL ≥6mm (on different teeth) AND ≥1 interproximal site with PD ≥5mm |
| **Moderate** | ≥2 interproximal sites with CAL ≥4mm (on different teeth) OR ≥2 interproximal sites with PD ≥5mm (on different teeth) |
| **Mild** | ≥2 interproximal sites with CAL ≥3mm AND ≥2 interproximal sites with PD ≥4mm (on different teeth) |
| **None** | Does not meet criteria for mild, moderate, or severe periodontitis |

**Binary Outcome:** has_periodontitis = 1 if mild, moderate, or severe; 0 if none.

### 2.5 Predictors

We selected 15 predictors based on Bashir et al. [2022] and clinical relevance:

| Category | Variable | NHANES Code | Handling |
|----------|----------|-------------|----------|
| **Demographics** | Age (years) | RIDAGEYR | Continuous |
| | Sex | RIAGENDR | Binary (male=1) |
| | Education | DMDEDUC2 | Binary (≥HS=1) |
| **Behaviors** | Smoking status | SMQ020, SMQ040 | 3-level (never/former/current) |
| | Current alcohol use | ALQ110 | Binary |
| **Metabolic** | BMI (kg/m²) | BMXBMI | Continuous |
| | Waist circumference (cm) | BMXWAIST | Continuous |
| | Waist-to-height ratio | BMXWAIST/BMXHT | Continuous |
| | Systolic BP (mmHg) | BPXSY1 | Continuous |
| | Diastolic BP (mmHg) | BPXDI1 | Continuous |
| | Fasting glucose (mg/dL) | LBXGLU | Continuous |
| | Triglycerides (mg/dL) | LBXTR | Continuous |
| | HDL cholesterol (mg/dL) | LBDHDD | Continuous |
| **Oral Health** | Dental visit (≤1 year) | OHQ030 | Binary |
| | Mobile/loose teeth | OHQ680 | Binary |
| | Flossing frequency | OHQ620 | Ordinal (1-5 days/week) |

### 2.6 Missing Data Strategy

**Missing Data Patterns:**
NHANES data exhibit non-random missingness:
- Fasting laboratory tests (glucose, triglycerides, HDL) are obtained only from participants assigned to the morning session who successfully fasted
- Questionnaire items follow skip-pattern logic (e.g., smoking details skipped for never-smokers)

**Approaches Compared:**

| Approach | Description | Use Case |
|----------|-------------|----------|
| **Median/Mode Imputation** | Replace missing values with median (continuous) or mode (categorical) | Linear models (Logistic Regression) |
| **Native NaN Handling** | Pass missing values directly to tree-based models | XGBoost, CatBoost, LightGBM |
| **Missing Indicators** | Create binary flags (feature_missing = 1 if NaN) | All models |

**Ablation Experiments:**
To distinguish genuine predictive signal from survey artifacts, we conducted:
1. **ABLATION_1:** Models trained with imputation only (no missing indicators)
2. **ABLATION_2:** Models trained with missing indicators only (imputed feature values)
3. **ABLATION_3:** Complete-case analysis (participants with all features present)
4. **STRAT_AVAIL:** Evaluation stratified by feature availability

### 2.7 Model Development

**Algorithms:**

| Model | Implementation | Hyperparameter Optimization |
|-------|----------------|----------------------------|
| Logistic Regression | scikit-learn 1.7 | Balanced class weights |
| Random Forest | scikit-learn 1.7 | Default parameters + balanced weights |
| XGBoost | xgboost 3.1 | Optuna (100 trials) |
| CatBoost | catboost 1.2 | Optuna (100 trials) |
| LightGBM | lightgbm 4.6 | Optuna (100 trials) |

**Hyperparameter Search:**
We used Optuna [Akiba et al., 2019] with Tree-structured Parzen Estimator (TPE) sampling for Bayesian hyperparameter optimization. Search spaces included:
- Learning rate: 0.01–0.3 (log scale)
- Tree depth: 3–10
- Number of estimators: 100–500
- Regularization parameters: model-specific

**Monotonic Constraints:**
For gradient boosting models, we enforced clinically plausible monotonic constraints:
- Positive monotonicity (+1): age, BMI, waist measures, blood pressure, glucose, triglycerides
- Negative monotonicity (-1): HDL cholesterol
- No constraint (0): sex, education, behaviors, oral health

### 2.8 Probability Calibration

Raw ML predictions were calibrated using isotonic regression fitted on out-of-fold predictions during cross-validation. Calibration quality was assessed using:
- **Brier Score:** Mean squared error of probabilistic predictions
- **Reliability Diagrams:** Visual comparison of predicted vs. observed probabilities

### 2.9 Ensemble Construction

A soft-voting ensemble was constructed by averaging calibrated probabilities:
- CatBoost: 34%
- XGBoost: 33%
- LightGBM: 33%

Weights were selected based on cross-validation performance.

### 2.10 Threshold Selection

**Clinical Operating Point:**
For screening applications, we prioritized high sensitivity while maintaining clinically acceptable specificity:
- **Target:** Recall ≥ 90%, Specificity ≥ 35%
- **Optimization:** Maximize F1-score subject to constraints

### 2.11 Internal Validation

Stratified 5-fold cross-validation with:
- Stratification on outcome to maintain class balance
- Out-of-fold predictions for calibration and ensemble training
- Statistical comparison using paired t-tests with Bonferroni correction

### 2.12 External Validation

**NHANES 2009-2010:**
- Same predictor extraction and preprocessing pipeline
- Evaluation with frozen model weights from 2011-2014 training
- Recalibration experiment: 10% calibration split + 90% test

### 2.13 Performance Metrics

**Discrimination:**
- AUC-ROC: Area under receiver operating characteristic curve
- PR-AUC: Area under precision-recall curve (appropriate for imbalanced data)

**Calibration:**
- Brier Score: Lower is better
- Expected Calibration Error (ECE)

**Clinical Utility:**
- Sensitivity (Recall), Specificity, Precision, F1-Score at selected threshold
- Decision Curve Analysis

### 2.14 Interpretability

**SHAP Analysis:**
SHapley Additive exPlanations [Lundberg & Lee, 2017] were computed for the best-performing model to:
- Rank feature importance
- Identify potential reverse-causality (e.g., dental visit, flossing)
- Assess biological plausibility

### 2.15 Software and Reproducibility

All analyses were performed in Python 3.11. Code is available at [GitHub URL]. NHANES data files are publicly available at https://wwwn.cdc.gov/nchs/nhanes/.

---

## 3. Results

### 3.1 Study Population

**[TABLE 1: Participant Characteristics]**

| Characteristic | Development (2011-2014) | External (2009-2010) | p-value |
|----------------|-------------------------|---------------------|---------|
| N | 9,379 | TBD | - |
| Age, years (mean ± SD) | TBD | TBD | TBD |
| Male, n (%) | TBD | TBD | TBD |
| Periodontitis, n (%) | 6,405 (68.3%) | TBD | TBD |
| - Severe | TBD | TBD | TBD |
| - Moderate | TBD | TBD | TBD |
| - Mild | TBD | TBD | TBD |

### 3.2 Missing Data Patterns

**[TABLE 2: Missing Data Summary]**

| Feature | N Missing | % Missing | Mechanism |
|---------|-----------|-----------|-----------|
| Glucose | 5,154 | 55.0% | Fasting subsample |
| Triglycerides | 5,203 | 55.5% | Fasting subsample |
| Smoking (detailed) | 5,116 | 54.5% | Skip pattern |
| ... | ... | ... | ... |

### 3.3 Model Performance (Internal Validation)

**[TABLE 3: Cross-Validation Performance]**

| Model | AUC-ROC (95% CI) | PR-AUC | Brier | Recall | Specificity | F1 |
|-------|------------------|--------|-------|--------|-------------|-----|
| Logistic Regression | 0.643 (X-X) | 0.771 | TBD | TBD | TBD | TBD |
| Random Forest | 0.717 (X-X) | 0.820 | TBD | TBD | TBD | TBD |
| XGBoost | 0.724 (X-X) | 0.826 | TBD | TBD | TBD | TBD |
| CatBoost | 0.727 (X-X) | 0.829 | TBD | TBD | TBD | TBD |
| LightGBM | 0.725 (X-X) | 0.826 | TBD | TBD | TBD | TBD |
| **Ensemble (calibrated)** | **0.730 (X-X)** | **0.829** | **0.178** | **TBD** | **TBD** | **TBD** |

### 3.4 Ablation Experiments (Missing Data)

**[TABLE 4: Contribution of Missing Data Handling]**

| Configuration | Best Model AUC | Δ vs Full | Interpretation |
|---------------|----------------|-----------|----------------|
| Full (NaN + indicators) | 0.727 | - | Baseline |
| Imputation only | TBD | TBD | Signal loss from imputation |
| Indicators only | TBD | TBD | Pure survey artifact? |
| Complete-case | TBD | TBD | True biological signal |

### 3.5 External Validation (2009-2010)

**[TABLE 5: External Validation Results]**

| Configuration | AUC-ROC | Δ from Internal | Brier |
|---------------|---------|-----------------|-------|
| Frozen model | TBD | TBD | TBD |
| Recalibrated (10% split) | TBD | TBD | TBD |

### 3.6 Clinical Operating Point

**[TABLE 6: Threshold Analysis]**

At threshold = X.XX:
- Sensitivity: TBD
- Specificity: TBD
- PPV: TBD
- NPV: TBD
- F1: TBD

### 3.7 Feature Importance (SHAP)

**[FIGURE X: SHAP Summary Plot]**

Top 5 features:
1. Age
2. TBD
3. TBD
4. TBD
5. TBD

**Reverse-Causality Candidates:**
- Dental visit: SHAP = TBD (may reflect treatment-seeking)
- Flossing: SHAP = TBD (may reflect dentist advice post-diagnosis)

---

## 4. Discussion

### 4.1 Principal Findings

This study provides a rigorous benchmark of modern gradient boosting methods for periodontitis prediction. Key findings include:

1. **Realistic Performance:** Calibrated ensemble achieved AUC 0.730, substantially lower than previously reported values (~0.95) but consistent with theoretical expectations given the feature set.

2. **Missing Data Signal:** Missing indicators contributed +0.02 AUC. Ablation experiments revealed [X% attributable to survey design vs. biological signal].

3. **External Generalization:** Performance on NHANES 2009-2010 was [maintained/reduced by X%], demonstrating [robustness/limitations].

### 4.2 Comparison with Previous Literature

**Why Our AUC (0.73) Differs from Bashir (0.95):**

| Factor | Our Study | Bashir et al. |
|--------|-----------|---------------|
| Validation | 5-fold CV + External | Single split |
| Calibration | Isotonic regression | None reported |
| Missing data | Native NaN + indicators | Not specified |
| Sample size | 9,379 + external | TBD |

We propose that our results represent deployable performance estimates, while higher internal validation AUCs in the literature reflect overfitting to specific data partitions.

### 4.3 Clinical Implications

**As a Screening Tool:**
- Sensitivity TBD: [Interpretation]
- Specificity TBD: [Interpretation]
- Suitable for [identifying high-risk individuals for dental referral / population-level screening]

**NOT Suitable For:**
- Diagnosis (requires clinical examination)
- Treatment planning

**Risk Communication:**
Calibrated probabilities enable statements like: "You have an estimated 70% probability of having periodontitis based on your health profile."

### 4.4 The Missingness Question

Gemini [AI collaborator] raised an important concern: Are we learning survey artifacts or biological reality?

**Evidence for Biological Signal:**
- [Results from complete-case analysis]
- [Stability across NHANES cycles]

**Evidence for Survey Artifact:**
- [Results from indicator-only ablation]
- [Differences between cycles]

**Clinical Implication:**
If deploying in a setting where all patients receive laboratory tests (no NaNs), the model may perform differently. We recommend [recalibration / using the complete-case model variant].

### 4.5 Limitations

1. **Cross-Sectional Design:** Cannot assess incident risk prediction
2. **US Population:** May not generalize internationally
3. **2011-2014 Data:** NHANES discontinued periodontal exams
4. **Reverse Causality:** Dental visit and flossing may reflect response to disease
5. **Feature Set:** Limited to low-cost predictors; no genetics or inflammatory markers

### 4.6 Future Directions

1. Prospective validation in clinical settings
2. International replication (KNHANES, European surveys)
3. Integration of inflammatory markers (CRP, IL-6)
4. Development of patient-facing risk calculator

---

## 5. Conclusions

Modern gradient boosting with native missing data handling and probability calibration achieves AUC ~0.73 for periodontitis prediction using low-cost NHANES predictors. This performance is substantially lower than previously reported (~0.95), likely reflecting more rigorous validation methodology. The model demonstrates [robust/limited] external generalization and provides calibrated probabilities suitable for clinical decision support. Future studies should validate in prospective clinical cohorts and address the reverse-causality limitations of self-reported oral health behaviors.

---

## Tables and Figures Checklist

### Required for Peer Review:

- [ ] **Table 1:** Participant characteristics (development vs. external)
- [ ] **Table 2:** Missing data summary
- [ ] **Table 3:** Model performance comparison (internal CV)
- [ ] **Table 4:** Ablation experiments (missing data contribution)
- [ ] **Table 5:** External validation results
- [ ] **Table 6:** Threshold analysis (clinical operating point)

- [ ] **Figure 1:** Study flow diagram (CONSORT-style)
- [ ] **Figure 2:** Calibration curves (before/after isotonic regression)
- [ ] **Figure 3:** ROC curves (all models + ensemble)
- [ ] **Figure 4:** SHAP summary plot
- [ ] **Figure 5:** Decision curve analysis
- [ ] **Figure 6:** External validation comparison (internal vs. 2009-2010)

### Supplementary Materials:

- [ ] **S1:** Full hyperparameter search results
- [ ] **S2:** TRIPOD checklist
- [ ] **S3:** Complete statistical tests
- [ ] **S4:** NHANES variable codebook
- [ ] **S5:** Code repository documentation

---

## TRIPOD Checklist (Abbreviated)

| Item | Section | Status |
|------|---------|--------|
| Title (D;V) | Title | ✅ |
| Abstract | Abstract | ✅ |
| Background/objectives | Introduction | ✅ |
| Source of data | Methods 2.2 | ✅ |
| Participants | Methods 2.3 | ✅ |
| Outcome | Methods 2.4 | ✅ |
| Predictors | Methods 2.5 | ✅ |
| Sample size | Methods 2.3 | ✅ |
| Missing data | Methods 2.6 | ✅ |
| Statistical analysis | Methods 2.7-2.14 | ✅ |
| Model specification | Methods 2.7 | ✅ |
| Model performance | Results 3.3 | ⏳ |
| External validation | Results 3.5 | ⏳ |
| Limitations | Discussion 4.5 | ✅ |
| Interpretation | Discussion 4.1-4.4 | ✅ |
| Supplementary info | Supplementary | ⏳ |

---

## Authorship Notes

**Francisco Teixeira Barbosa** - Conceptualization, Methodology, Software, Formal Analysis, Writing (Original Draft), Visualization

**AI Collaborators:**
- Claude (Anthropic): Code development, methodology
- GPT-4 (OpenAI): Methodological consultation, publication strategy
- Gemini (Google): Critical review, scientific rigor

*Note: AI contribution will be disclosed per journal guidelines*

---

## References (To Be Completed)

1. Eke PI, et al. Update on prevalence of periodontitis in adults in the United States: NHANES 2009-2012. J Periodontol. 2015.
2. Eke PI, et al. CDC/AAP case definitions for surveillance of periodontitis. J Periodontol. 2012;83(12):1449-1454.
3. Bashir NZ, et al. Systematic comparison of machine learning algorithms... J Clin Periodontol. 2022;49:958-969.
4. Collins GS, et al. Transparent Reporting of a multivariable prediction model for Individual Prognosis or Diagnosis (TRIPOD). Ann Intern Med. 2015;162(1):55-63.
5. Lundberg SM, Lee SI. A unified approach to interpreting model predictions. NeurIPS 2017.
6. Akiba T, et al. Optuna: A next-generation hyperparameter optimization framework. KDD 2019.

---

## Changelog

| Date | Version | Changes |
|------|---------|---------|
| 2025-12-02 | 0.1 | Initial peer-review draft structure |

---

**Document Status:** Draft - For internal review only

