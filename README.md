# NHANES Periodontitis Prediction Benchmark

This repository contains a reproducible benchmark of low-cost predictors for periodontitis classification in NHANES. The current manuscript framing is methodological: it estimates realistic performance bounds, checks calibration and missingness sensitivity, and documents why questionnaire/metabolic predictors should not be presented as a stand-alone diagnostic replacement for periodontal examination.

## Current Study Framing

- Development cohort: NHANES 2011-2014 adults age 30+ with full periodontal examination, `n=9,379`.
- Same-source temporal validation cohort: NHANES 2009-2010, `n=5,177`.
- Outcome: any periodontitis versus no periodontitis using CDC/AAP case definitions.
- Primary model: calibrated soft-voting ensemble with 29 predictors after excluding treatment-seeking/reverse-causality variables.
- Secondary model: 33 predictors with the treatment-seeking variables restored for upper-bound sensitivity analysis.
- Scope: research benchmarking and risk stratification only; not diagnosis, treatment planning, or unvalidated use in non-NHANES clinical settings.

## Canonical Results

These values are the source-of-truth values enforced by `scripts/check_publication_consistency.py`.

| Analysis | Model | Features | AUC-ROC | PR-AUC | Notes |
|---|---:|---:|---:|---:|---|
| Internal 5-fold CV | Primary no reverse-causality | 29 | 0.7172 | 0.8157 | Main development estimate |
| Internal 5-fold CV | Secondary full-feature | 33 | 0.7255 | 0.8207 | Adds dental visit, flossing, loose teeth, and floss-missing flag |
| Same-source temporal validation | Frozen primary model on 2009-2010 | 29 | 0.6771 | 0.7735 | Same survey system, earlier cycle |

Temporal operating points for the frozen primary model:

| Threshold | Sensitivity | Specificity | PPV | NPV | Interpretation |
|---:|---:|---:|---:|---:|---|
| 0.35 | 97.1% | 18.1% | 70.8% | 75.2% | High-sensitivity triage; negative screens are not definitive |
| 0.65 | 82.6% | 43.3% | 74.9% | 54.9% | More balanced but still requires clinical confirmation |

The key conclusion is deliberately modest: with these low-cost predictors, discrimination is around 0.72 internally and around 0.68 under same-source temporal validation. The observed performance is useful as a benchmark, not as proof of readiness for clinical implementation.

## Reproducibility

Set up a development environment:

```bash
make setup
source venv/bin/activate
```

Set up the pinned publication target:

```bash
make setup-lock
source venv/bin/activate
```

Run lightweight checks that do not require NHANES data:

```bash
make test
make consistency
```

Run the full workflows after NHANES data are available:

```bash
make download
make process
make reproduce
make temporal
python3 scripts/04_publication_analyses.py \
  --input data/processed/publication_predictions.parquet \
  --feature-cols age bmi waist_cm systolic_bp diastolic_bp glucose triglycerides hdl
```

The legacy notebooks are retired as source-of-truth artifacts. The maintained publication surface is the script targets, result artifacts, model card, tests, and manuscript source, with consistency checks to prevent silent drift across those files.

## Repository Structure

| Path | Purpose |
|---|---|
| `src/labels.py` | CDC/AAP case-definition implementation and synthetic test fixtures |
| `src/evaluation.py` | Metrics, threshold selection, calibration, and plotting helpers |
| `src/publication_analysis.py` | Survey-weighted prevalence, subgroup performance, and missingness tables |
| `scripts/check_publication_consistency.py` | Guards canonical values and conservative publication wording |
| `scripts/04_publication_analyses.py` | Generates publication sensitivity tables from processed predictions |
| `results/` | Saved result artifacts used by the manuscript and model card |
| `docs/publication/ARTICLE_DRAFT.md` | Current manuscript source |

## Known Limitations

- The validation cohort is temporally distinct but comes from the same NHANES survey system; this is not geographic or prospective clinical validation.
- The analytic cohort has high periodontitis prevalence because it is restricted to adults with full periodontal examination data.
- NHANES missingness patterns may encode survey logistics. The deployment-ready model without missingness indicators is therefore emphasized as a more realistic lower-bound benchmark.
- Subgroup and survey-weighted analyses require processed prediction tables and should be regenerated before journal submission.
- The repository is research software. Any clinical use would require independent validation, local recalibration, governance review, and workflow-specific safety assessment.

## Citation

```bibtex
@software{barbosa_nhanes_periodontitis_benchmark,
  author = {Barbosa, Francisco Teixeira},
  title = {NHANES Periodontitis Prediction Benchmark},
  year = {2026},
  url = {https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project}
}
```
