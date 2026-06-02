# Model Card: NHANES Periodontitis Benchmark v1.3

## Model Details

| Field | Value |
|---|---|
| Model name | `v1.3_primary_no_reverse_causality` |
| Model type | Calibrated soft-voting ensemble of CatBoost, XGBoost, and LightGBM |
| Development data | NHANES 2011-2014 adults age 30+ with full periodontal examination, `n=9,034` |
| Same-source temporal validation data | NHANES 2009-2010, `n=5,037` |
| Outcome | Any CDC/AAP periodontitis versus no periodontitis |
| Primary feature count | 29 predictors |
| Secondary feature count | 33 predictors |
| Calibration | Isotonic calibration in the cross-validation workflow |

## Intended Use

This model is intended for research benchmarking, methods comparison, and risk-stratification experiments using NHANES-like tabular predictors. It is not a diagnostic system. It should not be used for treatment planning, insurance decisions, or patient-facing screening without independent validation, local recalibration, and governance review.

## Performance

| Evaluation | AUC-ROC | PR-AUC | Brier | Notes |
|---|---:|---:|---:|---|
| Internal 5-fold CV, primary 29-feature model | 0.6896 | 0.8240 | 0.1871 | Excludes treatment-seeking variables |
| Internal 5-fold CV, secondary 33-feature model | 0.6996 | 0.8295 | 0.1844 | Includes treatment-seeking variables |
| Same-source temporal validation, frozen primary model | 0.6495 | 0.7727 | 0.2023 | NHANES 2009-2010 |

Temporal operating points for the frozen primary model:

| Threshold | Sensitivity | Specificity | PPV | NPV | Appropriate interpretation |
|---:|---:|---:|---:|---:|---|
| 0.35 | 98.9% | 5.5% | 70.0% | 69.1% | High-sensitivity triage threshold; many false positives and some false negatives remain |
| 0.65 | 77.7% | 45.2% | 76.0% | 47.5% | More balanced threshold; still not sufficient for diagnosis |

## Feature Sets

The primary model includes 29 predictors after removing treatment-seeking variables that may be consequences of existing disease.

Included categories:

- Demographics: age, sex, education.
- Behaviors: smoking and alcohol variables retained as low-cost risk predictors.
- Metabolic and anthropometric variables: BMI, height, waist measures, blood pressure, glucose, triglycerides, HDL.
- Missingness indicators for variables where NHANES skip patterns or fasting subsamples create informative missingness.

Excluded from the primary model:

- `dental_visit`
- `floss_days`
- `mobile_teeth`
- `floss_days_missing`

The secondary 33-feature model restores these variables only to estimate the upper-bound performance contribution from treatment-seeking signals.

## Validation and Applicability

The temporal validation cohort is useful because the model is frozen and evaluated on a different NHANES cycle. It is still the same survey program, country, and broad measurement system. Geographic validation and prospective clinical validation remain unperformed.

Known applicability limits:

- High analytic-sample prevalence, around 66-72% depending on cycle and weighting, limits direct PPV/NPV transfer to lower-prevalence populations.
- Missingness indicators may learn survey logistics, so the deployment-ready no-indicator model should be reported as a conservative benchmark.
- Subgroup calibration and discrimination should be regenerated before journal submission using `scripts/04_publication_analyses.py`.
- Any implementation outside NHANES-like research data requires local recalibration and independent safety assessment.

## Reproducibility

```bash
make setup-lock
source venv/bin/activate
make test
make consistency
make verify-submission
make reproduce-full
```

The consistency check enforces agreement between result artifacts, README, this model card, and the manuscript source.

## AI-Use Disclosure

AI systems were used as drafting and code-review aids during project development. The author remains responsible for study design, code, analysis decisions, interpretation, and manuscript claims.
