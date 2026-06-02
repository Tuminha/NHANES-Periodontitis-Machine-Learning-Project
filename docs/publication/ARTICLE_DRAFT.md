# Machine Learning for Periodontitis Prediction: A Realistic Benchmark with Same-Source Temporal Validation on NHANES

**Draft version:** publication-readiness repair, June 2026
**Article type:** prediction-model methodology benchmark
**Reporting target:** TRIPOD+AI-aligned development and validation report

## Abstract

**Background:** Machine-learning studies for periodontitis prediction have reported very high internal discrimination using low-cost NHANES predictors. Such estimates require careful reassessment because internal validation, missing-data handling, calibration, and treatment-seeking predictors can materially change apparent performance.

**Methods:** We analyzed NHANES 2011-2014 adults age 30 years or older with full periodontal examinations (`n=9,379`). Periodontitis was classified using CDC/AAP case definitions. XGBoost, LightGBM, and CatBoost were compared with logistic regression and random forest using stratified 5-fold cross-validation. The primary calibrated ensemble excluded treatment-seeking variables and used 29 predictors. A secondary 33-feature model restored dental visit, flossing, loose teeth, and floss-missing variables to estimate their incremental contribution. The frozen primary model was evaluated on NHANES 2009-2010 (`n=5,177`) as same-source temporal validation. Missingness ablations and deployment-ready feature analysis were used to assess whether NHANES missingness patterns contributed survey-specific signal.

**Results:** The primary 29-feature model achieved internal AUC-ROC 0.7172 and PR-AUC 0.8157. The secondary 33-feature model achieved AUC-ROC 0.7255 and PR-AUC 0.8207, indicating that treatment-seeking variables added only 0.0083 AUC. In same-source temporal validation, the frozen primary model achieved AUC-ROC 0.6771, PR-AUC 0.7735, and Brier score 0.2003. At threshold 0.35, sensitivity was 97.1% and specificity was 18.1%; at threshold 0.65, sensitivity was 82.6% and specificity was 43.3%. These operating points are not diagnostic rules and require periodontal examination for confirmation.

**Conclusions:** Low-cost NHANES predictors appear to support realistic discrimination around 0.72 internally and 0.68 under same-source temporal validation. The study is best interpreted as a benchmark and cautionary methods report, not as evidence that a deployable diagnostic screening model has been established. Geographic validation, prospective clinical validation, local recalibration, subgroup calibration, and survey-weighted sensitivity analyses remain necessary before implementation claims.

**Keywords:** periodontitis; NHANES; prediction model; gradient boosting; calibration; missing data; TRIPOD+AI

## 1. Introduction

Periodontitis is common among adults and is often identified only after irreversible tissue destruction has occurred. Low-cost risk stratification is attractive because full periodontal examination requires trained personnel, examination time, and access to dental care. However, a model built from demographic, behavioral, anthropometric, and metabolic predictors is not equivalent to clinical examination.

Prior machine-learning work using NHANES has reported very high apparent discrimination. This manuscript reassesses the likely performance ceiling under stricter validation and reporting expectations. The central question is not whether a model can replace periodontal examination, but whether low-cost NHANES predictors contain enough signal to support reliable risk stratification and what methodological choices inflate or constrain that estimate.

The objectives were to:

1. Estimate realistic internal performance for modern gradient boosting models.
2. Evaluate performance on an earlier NHANES cycle using a frozen model.
3. Quantify the effect of treatment-seeking variables that may reflect existing disease.
4. Assess missingness indicators and a deployment-ready feature set without NHANES-specific missingness flags.
5. Reframe the model using prediction-model reporting standards and explicit limitations.

## 2. Methods

### 2.1 Study Design and Data Source

This was a prediction-model benchmark using public NHANES data. The development cohort comprised NHANES 2011-2012 and 2013-2014. The same-source temporal validation cohort comprised NHANES 2009-2010. Later NHANES cycles were not used for temporal validation because full-mouth periodontal measurements required for CDC/AAP classification were discontinued.

NHANES data are publicly available and de-identified. This secondary analysis did not require institutional review board approval.

### 2.2 Participants

Eligible participants were adults age 30 years or older with full periodontal examination data and sufficient information for CDC/AAP periodontitis classification. The development cohort included 9,379 participants. The same-source temporal validation cohort included 5,177 participants.

The analytic prevalence of periodontitis was approximately 67-68%, higher than general-population CDC estimates. This reflects the restricted full-examination analytic sample and should not be interpreted as the expected prevalence in a lower-risk screening population.

### 2.3 Outcome Definition

The binary outcome was any periodontitis versus no periodontitis using CDC/AAP definitions. Severe, moderate, and mild classifications were assigned hierarchically from interproximal pocket depth and clinical attachment loss measurements, excluding third molars and enforcing different-teeth criteria where specified.

The repository now includes synthetic tests for severe, moderate, mild, and no-periodontitis cases, third-molar exclusion, and different-teeth logic.

### 2.4 Predictors and Feature Sets

The primary model used 29 predictors after excluding treatment-seeking variables that can be downstream of disease:

- `dental_visit`
- `floss_days`
- `mobile_teeth`
- `floss_days_missing`

The secondary model used 33 predictors by restoring those variables. The secondary model is reported only as an upper-bound sensitivity analysis, not as the preferred screening model.

Predictor categories included demographics, smoking and alcohol variables, anthropometric measures, blood pressure, fasting glucose, triglycerides, HDL, and missingness indicators.

### 2.5 Missing Data

NHANES missingness is not purely random. Fasting laboratory variables are collected in subsamples, and questionnaire items may follow skip-pattern logic. Tree models were allowed to handle missing values natively, and missingness indicators were included in the primary model. A deployment-ready 15-feature model without missingness indicators was retained as a conservative lower-bound benchmark because NHANES-specific missingness may not transfer to clinical datasets.

### 2.6 Model Development and Calibration

Gradient boosting models were tuned using Optuna and evaluated with stratified 5-fold cross-validation. Monotonic constraints were applied to selected continuous variables where clinical priors were clear. Isotonic calibration was used for probability calibration in the cross-validation workflow.

The final temporal evaluation used the frozen primary model and pre-specified thresholds from the development workflow. Thresholds were not re-optimized on the temporal validation cohort.

### 2.7 Statistical Analysis

Discrimination was summarized with AUC-ROC and PR-AUC. Calibration was summarized with Brier score and reliability plots. Operating points were reported using sensitivity, specificity, PPV, and NPV. Decision-curve analysis was included as a descriptive utility analysis only.

Survey-weighted prevalence and subgroup performance are generated by `scripts/04_publication_analyses.py` when processed prediction tables are available. These analyses should be regenerated immediately before journal submission.

## 3. Results

### 3.1 Cohorts

The development cohort included 9,379 adults from NHANES 2011-2014. The temporal validation cohort included 5,177 adults from NHANES 2009-2010. Periodontitis prevalence was 68.3% in development data and 67.2% in temporal validation data.

### 3.2 Internal Model Performance

| Model variant | Features | AUC-ROC | PR-AUC | Interpretation |
|---|---:|---:|---:|---|
| Primary model without treatment-seeking variables | 29 | 0.7172 | 0.8157 | Preferred benchmark model |
| Secondary full-feature model | 33 | 0.7255 | 0.8207 | Upper-bound sensitivity analysis |
| Deployment-ready core model | 15 | 0.6692 | 0.8137 | Conservative feature set without missingness indicators |

The full-feature model improved AUC by 0.0083 over the primary model. This small difference supports excluding treatment-seeking variables from the primary benchmark.

### 3.3 Same-Source Temporal Validation

| Metric | Development estimate | Temporal validation estimate |
|---|---:|---:|
| N | 9,379 | 5,177 |
| Prevalence | 68.3% | 67.2% |
| AUC-ROC | 0.7172 | 0.6771 (95% CI 0.6612-0.6934) |
| PR-AUC | 0.8157 | 0.7735 (95% CI 0.7571-0.7892) |
| Brier score | 0.1812 | 0.2003 (95% CI 0.1935-0.2073) |

The drop from internal AUC 0.7172 to temporal AUC 0.6771 is consistent with a modest but meaningful generalization gap.

### 3.4 Operating Points

| Threshold | Sensitivity | Specificity | PPV | NPV |
|---:|---:|---:|---:|---:|
| 0.35 | 97.1% | 18.1% | 70.8% | 75.2% |
| 0.65 | 82.6% | 43.3% | 74.9% | 54.9% |

The 0.35 threshold prioritizes sensitivity but has very low specificity and an NPV of only 75.2% in this high-prevalence cohort. It should be described as a high-sensitivity triage operating point, not a reliable disease-exclusion rule.

### 3.5 Missingness and Survey-Design Sensitivity

Missingness indicators contributed limited but measurable predictive signal. Missingness patterns were broadly comparable between the development and temporal validation cohorts, but the signal may still be NHANES-specific. The deployment-ready 15-feature model remains the most conservative estimate for settings where NHANES missingness patterns are unavailable.

Survey-weighted prevalence and subgroup-performance tables should be generated from processed prediction data before submission. The repository now includes reusable code for these analyses, but the current manuscript should not claim population-level transportability until those tables are complete and reviewed.

## 4. Discussion

This repair changes the interpretation of the study. The main contribution is not a clinically ready model. The contribution is a reproducible benchmark showing that realistic performance for low-cost periodontitis predictors is far below highly optimistic internal estimates reported in some prior work.

The same-source temporal validation result is important but limited. Because both cohorts come from NHANES, performance does not establish geographic transportability, prospective clinical performance, or behavior in lower-prevalence screening populations.

The operating points also require conservative interpretation. High sensitivity at threshold 0.35 comes at the cost of low specificity, and the NPV does not support reassuring individual patients without periodontal examination.

## 5. Limitations

1. The validation cohort is temporally distinct but not independent by geography, health system, or measurement program.
2. The analytic cohort is restricted to adults with full periodontal examination data and has high disease prevalence.
3. NHANES missingness indicators may encode survey logistics rather than transportable clinical information.
4. Survey-weighted and subgroup calibration tables must be regenerated before submission and interpreted carefully.
5. The model predicts current case status, not future incident periodontitis.
6. The manuscript does not establish patient benefit, treatment impact, or workflow safety.

## 6. Conclusions

The primary 29-feature model achieved AUC-ROC 0.7172 internally and 0.6771 under same-source temporal validation. A secondary 33-feature model provided only a small apparent gain, supporting exclusion of treatment-seeking variables from the preferred benchmark. These results support a cautious methodological conclusion: low-cost NHANES predictors provide moderate discrimination and are useful for benchmarking, but they do not establish a clinically ready periodontitis screening system.

## Data and Code Availability

Code and saved result artifacts are available at: <https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project>. Raw NHANES data are available from the CDC.

## Funding

No external funding was reported for this analysis.

## Conflicts of Interest

The author declares no conflicts of interest.

## AI-Use Disclosure

AI systems were used for drafting support, code review, and critique generation during manuscript development. The author reviewed and remains responsible for all analysis decisions, code changes, interpretation, and final claims.

## References

1. Eke PI, Page RC, Wei L, Thornton-Evans G, Genco RJ. Update of the case definitions for population-based surveillance of periodontitis. J Periodontol. 2012;83(12):1449-1454.
2. Eke PI, Dye BA, Wei L, et al. Update on prevalence of periodontitis in adults in the United States: NHANES 2009-2012. J Periodontol. 2015;86(5):611-622.
3. Bashir NZ, Rahman Z, Chen SLS. Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. J Clin Periodontol. 2022;49:958-969.
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.
5. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505.
