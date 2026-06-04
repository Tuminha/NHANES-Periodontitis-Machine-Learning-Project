# Machine Learning for Periodontitis Prediction: A Realistic Benchmark with Same-Source Temporal Validation on NHANES

**Authors:** Francisco Teixeira Barbosa^1,2*, Aritza Brizuela-Velasco^2, Daniel Robles Cantero^2

**Affiliations:**

1. Foundation for Oral Rehabilitation (FOR), Werftestrasse 4, 6002 Luzern, Switzerland.
2. DENS-ia Research Group, Faculty of Health Sciences, Miguel de Cervantes European University (UEMC), Padre Julio Chevalier 2, 47012 Valladolid, Spain.

**Corresponding author:** Francisco Teixeira Barbosa, cisco@periospot.com

- **Draft version:** BMC Oral Health submission preparation, June 2026
- **Target journal:** BMC Oral Health
- **Article type:** Research article
- **Reporting target:** TRIPOD+AI-aligned development and validation report

## Abstract

**Background:** Machine-learning studies for periodontitis prediction have reported very high internal discrimination using low-cost NHANES predictors. Such estimates require careful reassessment because internal validation, missing-data handling, calibration, and treatment-seeking predictors can materially change apparent performance.

**Methods:** We analyzed NHANES 2011-2014 adults age 30 years or older with full periodontal examinations (`n=9,034`). Periodontitis was classified using CDC/AAP case definitions. XGBoost, LightGBM, and CatBoost were compared using stratified 5-fold cross-validation. The primary calibrated ensemble excluded treatment-seeking variables and used 29 predictors. A secondary 33-feature model restored dental visit, flossing, loose teeth, and floss-missing variables to estimate their incremental contribution. The frozen primary model was evaluated on NHANES 2009-2010 (`n=5,037`) as same-source temporal validation. Missingness ablations and deployment-ready feature analysis were used to assess whether NHANES missingness patterns contributed survey-specific signal.

**Results:** The primary 29-feature model achieved internal AUC-ROC 0.6896 and PR-AUC 0.8240. The secondary 33-feature model achieved AUC-ROC 0.6996 and PR-AUC 0.8295, indicating that treatment-seeking variables added 0.0100 AUC. In same-source temporal validation, the frozen primary model achieved AUC-ROC 0.6495, PR-AUC 0.7727, and Brier score 0.2023. At threshold 0.35, sensitivity was 98.9% and specificity was 5.5%; at threshold 0.65, sensitivity was 77.7% and specificity was 45.2%. These operating points are not diagnostic rules and require periodontal examination for confirmation.

**Conclusions:** Low-cost NHANES predictors appear to support realistic discrimination around 0.69 internally and 0.65 under same-source temporal validation. The study is best interpreted as a benchmark and cautionary methods report, not as evidence that a deployable diagnostic screening model has been established. Geographic validation, prospective clinical validation, local recalibration, and subgroup calibration remain necessary before implementation claims.

**Trial registration:** Not applicable.

**Keywords:** periodontitis; NHANES; prediction model; gradient boosting; calibration; missing data; TRIPOD+AI

## Background

Periodontitis is common among adults and is often identified only after irreversible tissue destruction has occurred. Low-cost risk stratification is attractive because full periodontal examination requires trained personnel, examination time, and access to dental care. However, a model built from demographic, behavioral, anthropometric, and metabolic predictors is not equivalent to clinical examination.

Prior machine-learning work using NHANES has reported very high apparent discrimination. This manuscript reassesses the likely performance ceiling under stricter validation and reporting expectations. The central question is not whether a model can replace periodontal examination, but whether low-cost NHANES predictors contain enough signal to support reliable risk stratification and what methodological choices inflate or constrain that estimate.

The objectives were to:

1. Estimate realistic internal performance for modern gradient boosting models.
2. Evaluate performance on an earlier NHANES cycle using a frozen model.
3. Quantify the effect of treatment-seeking variables that may reflect existing disease.
4. Assess missingness indicators and a deployment-ready feature set without NHANES-specific missingness flags.
5. Reframe the model using prediction-model reporting standards and explicit limitations.

## Methods

### Study Design and Data Source

This was a prediction-model benchmark using public NHANES data. The development cohort comprised NHANES 2011-2012 and 2013-2014. The same-source temporal validation cohort comprised NHANES 2009-2010. Later NHANES cycles were not used for temporal validation because full-mouth periodontal measurements required for CDC/AAP classification were discontinued.

NHANES data are publicly available and de-identified. This secondary analysis did not require institutional review board approval.

### Participants

Eligible participants were adults age 30 years or older with full periodontal examination data and sufficient information for CDC/AAP periodontitis classification. The development cohort included 9,034 participants. The same-source temporal validation cohort included 5,037 participants.

The analytic prevalence of periodontitis was approximately 69-72% unweighted and approximately 66% after applying examination weights by cycle, higher than general-population CDC estimates. This reflects the restricted full-examination analytic sample and should not be interpreted as the expected prevalence in a lower-risk screening population.

### Outcome Definition

The binary outcome was any periodontitis versus no periodontitis using CDC/AAP definitions. Severe, moderate, and mild classifications were assigned hierarchically from interproximal pocket depth and clinical attachment loss measurements, excluding third molars and enforcing different-teeth criteria where specified.

The repository now includes synthetic tests for severe, moderate, mild, and no-periodontitis cases, third-molar exclusion, and different-teeth logic.

### Predictors and Feature Sets

The primary model used 29 predictors after excluding treatment-seeking variables that can be downstream of disease:

- `dental_visit`
- `floss_days`
- `mobile_teeth`
- `floss_days_missing`

The secondary model used 33 predictors by restoring those variables. The secondary model is reported only as an upper-bound sensitivity analysis, not as the preferred screening model.

Predictor categories included demographics, smoking and alcohol variables, anthropometric measures, blood pressure, fasting glucose, triglycerides, HDL, and missingness indicators.

### Missing Data

NHANES missingness is not purely random. Fasting laboratory variables are collected in subsamples, and questionnaire items may follow skip-pattern logic. Tree models were allowed to handle missing values natively, and missingness indicators were included in the primary model. A deployment-ready 15-feature model without missingness indicators was retained as a conservative lower-bound benchmark because NHANES-specific missingness may not transfer to clinical datasets.

### Model Development and Calibration

Gradient boosting models were tuned using Optuna and evaluated with stratified 5-fold cross-validation. Monotonic constraints were applied to selected continuous variables where clinical priors were clear. Isotonic calibration was used for probability calibration in the cross-validation workflow.

The final temporal evaluation used the frozen primary model and pre-specified thresholds from the development workflow. Thresholds were not re-optimized on the temporal validation cohort.

### Statistical Analysis

Discrimination was summarized with AUC-ROC and PR-AUC. Calibration was summarized with Brier score and reliability plots. Operating points were reported using sensitivity, specificity, PPV, and NPV. Decision-curve analysis was included as a descriptive utility analysis only.

Survey-weighted prevalence and subgroup performance are generated by `scripts/04_publication_analyses.py` when processed prediction tables are available. The current regenerated summary is saved in `results/publication_sensitivity_tables.md`.

## Results

### Cohorts

The development cohort included 9,034 adults from NHANES 2011-2014. The temporal validation cohort included 5,037 adults from NHANES 2009-2010. Periodontitis prevalence was 70.9% in development data and 69.1% in temporal validation data before survey weighting.

### Internal Model Performance

| Model variant | Features | AUC-ROC | PR-AUC | Interpretation |
|---|---:|---:|---:|---|
| Primary model without treatment-seeking variables | 29 | 0.6896 | 0.8240 | Preferred benchmark model |
| Secondary full-feature model | 33 | 0.6996 | 0.8295 | Upper-bound sensitivity analysis |
| Deployment-ready core model | 15 | 0.6896 | 0.8237 | Conservative feature set without missingness indicators |

The full-feature model improved AUC by 0.0100 over the primary model. This small difference supports excluding treatment-seeking variables from the primary benchmark.

### Same-Source Temporal Validation

| Metric | Development estimate | Temporal validation estimate |
|---|---:|---:|
| N | 9,034 | 5,037 |
| Prevalence | 70.9% | 69.1% |
| AUC-ROC | 0.6896 | 0.6495 (95% CI 0.6315-0.6664) |
| PR-AUC | 0.8240 | 0.7727 (95% CI 0.7570-0.7885) |
| Brier score | 0.1871 | 0.2023 (95% CI 0.1955-0.2085) |

The drop from internal AUC 0.6896 to temporal AUC 0.6495 is consistent with a modest but meaningful generalization gap.

### Operating Points

| Threshold | Sensitivity | Specificity | PPV | NPV |
|---:|---:|---:|---:|---:|
| 0.35 | 98.9% | 5.5% | 70.0% | 69.1% |
| 0.65 | 77.7% | 45.2% | 76.0% | 47.5% |

The 0.35 threshold prioritizes sensitivity but has very low specificity and an NPV of only 69.1% in this high-prevalence cohort. It should be described as a high-sensitivity triage operating point, not a reliable disease-exclusion rule.

Figure 1 summarizes the discrimination, calibration error, temporal operating points, and primary-versus-secondary feature-set comparison.

![Figure 1. Model performance summary. Panel A shows internal and same-source temporal discrimination for the primary and secondary model variants. Panel B shows Brier score, where lower values indicate lower calibration error. Panel C shows sensitivity, specificity, PPV, and NPV for the frozen primary model in NHANES 2009-2010 at thresholds 0.35 and 0.65. Panel D shows that adding treatment-seeking variables increased the feature count from 29 to 33 and changed internal AUC by 0.0100.](figures/19_publication_performance_summary.png)

### Missingness and Survey-Design Sensitivity

Missingness indicators contributed limited but measurable predictive signal. Missingness patterns were broadly comparable between the development and temporal validation cohorts, but the signal may still be NHANES-specific. The deployment-ready 15-feature model remains the most conservative estimate for settings where NHANES missingness patterns are unavailable.

Survey-weighted prevalence and subgroup-performance tables were generated from processed prediction data and are saved in `results/publication_sensitivity_tables.md`. Weighted prevalence was approximately 65.6% in 2009-2010 and 66.2-66.3% in 2011-2014. Subgroup analyses are descriptive and should not be overinterpreted as evidence of transportability.

Figure 2 summarizes weighted prevalence, selected subgroup AUCs, and the highest missingness proportions among predictors.

![Figure 2. Survey sensitivity summary. Panel A compares unweighted and survey-weighted periodontitis prevalence by NHANES cycle. Panel B shows temporal AUC-ROC across selected age, sex, smoking, and metabolic-risk subgroups, with the overall temporal AUC shown as a dashed reference line. Panel C shows the predictors with the highest missingness proportions, highlighting the fasting laboratory variables as the dominant source of missingness.](figures/20_publication_sensitivity_summary.png)

## Discussion

This repair changes the interpretation of the study. The main contribution is not a clinically ready model. The contribution is a reproducible benchmark showing that realistic performance for low-cost periodontitis predictors is far below highly optimistic internal estimates reported in some prior work.

The same-source temporal validation result is important but limited. Because both cohorts come from NHANES, performance does not establish geographic transportability, prospective clinical performance, or behavior in lower-prevalence screening populations.

The operating points also require conservative interpretation. High sensitivity at threshold 0.35 comes at the cost of low specificity, and the NPV does not support reassuring individual patients without periodontal examination.

## Limitations

1. The validation cohort is temporally distinct but not independent by geography, health system, or measurement program.
2. The analytic cohort is restricted to adults with full periodontal examination data and has high disease prevalence.
3. NHANES missingness indicators may encode survey logistics rather than transportable clinical information.
4. Survey-weighted and subgroup calibration tables are descriptive and should be interpreted carefully.
5. The model predicts current case status, not future incident periodontitis.
6. The manuscript does not establish patient benefit, treatment impact, or workflow safety.

## Conclusions

The primary 29-feature model achieved AUC-ROC 0.6896 internally and 0.6495 under same-source temporal validation. A secondary 33-feature model provided only a small apparent gain, supporting exclusion of treatment-seeking variables from the preferred benchmark. These results support a cautious methodological conclusion: low-cost NHANES predictors provide moderate discrimination and are useful for benchmarking, but they do not establish a clinically ready periodontitis screening system.

## List of abbreviations

AUC-ROC: area under the receiver operating characteristic curve; Brier: Brier score; CDC: Centers for Disease Control and Prevention; CI: confidence interval; FOR: Foundation for Oral Rehabilitation; NHANES: National Health and Nutrition Examination Survey; NCHS: National Center for Health Statistics; NPV: negative predictive value; PPV: positive predictive value; PR-AUC: area under the precision-recall curve; PROBAST+AI: Prediction model Risk Of Bias ASsessment Tool plus artificial intelligence; TRIPOD+AI: Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis plus artificial intelligence; UEMC: Miguel de Cervantes European University.

## Declarations

### Ethics approval and consent to participate

NHANES protocols were approved by the NCHS Research Ethics Review Board, and NHANES participants provided informed consent [8]. This secondary analysis used publicly available de-identified NHANES data and did not require additional local ethics approval.

### Consent for publication

Not applicable. The manuscript does not include individual person-level identifying data, images, or videos.

### Availability of data and materials

The datasets analyzed during the current study are publicly available from the CDC/NCHS NHANES website [7]. Code, tests, scripts, generated figures, and saved result artifacts are available in the project repository [6].

### Competing interests

FTB is Executive Director of the Foundation for Oral Rehabilitation and founder/editor-in-chief of PerioSpot. ABV and DRC are affiliated with DENS-ia Research Group, Faculty of Health Sciences, Miguel de Cervantes European University. The authors report no financial competing interests directly related to the NHANES data, code, or periodontitis prediction model.

### Funding

No external funding was reported for this analysis.

### Authors' contributions

FTB conceived the study, implemented and verified the reproducible analysis workflow, generated figures, interpreted results, and drafted the manuscript. ABV contributed clinical and methodological interpretation, reviewed the oral-health framing, and reviewed the manuscript. DRC contributed clinical interpretation, reviewed the oral-health framing, and reviewed the manuscript. All authors approved the submitted manuscript.

### Acknowledgements

The authors acknowledge the Centers for Disease Control and Prevention, the National Center for Health Statistics, and the NHANES participants and staff who made the public-use data available.

### Authors' information

Not applicable.

### AI-use disclosure

AI systems were used for drafting support, code review, figure-label review, and critique generation during manuscript development. No AI system is listed as an author. The authors reviewed and remain responsible for all analysis decisions, code changes, interpretation, and final claims.

## References

1. Eke PI, Page RC, Wei L, Thornton-Evans G, Genco RJ. Update of the case definitions for population-based surveillance of periodontitis. J Periodontol. 2012;83(12):1449-1454.
2. Eke PI, Dye BA, Wei L, et al. Update on prevalence of periodontitis in adults in the United States: NHANES 2009-2012. J Periodontol. 2015;86(5):611-622.
3. Bashir NZ, Rahman Z, Chen SLS. Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. J Clin Periodontol. 2022;49:958-969.
4. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. BMJ. 2024;385:e078378.
5. Moons KGM, Damen JAA, Kaul T, et al. PROBAST+AI: an updated quality, risk of bias, and applicability assessment tool for prediction models using regression or artificial intelligence methods. BMJ. 2025;388:e082505.
6. Barbosa FT, Brizuela-Velasco A, Robles Cantero D. NHANES Periodontitis Prediction Benchmark. GitHub. 2026. https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project. Accessed 4 Jun 2026.
7. National Center for Health Statistics. National Health and Nutrition Examination Survey. Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/. Accessed 4 Jun 2026.
8. National Center for Health Statistics. Ethics Review Board Approval. Centers for Disease Control and Prevention. https://www.cdc.gov/nchs/nhanes/about/erb.html. Accessed 4 Jun 2026.
