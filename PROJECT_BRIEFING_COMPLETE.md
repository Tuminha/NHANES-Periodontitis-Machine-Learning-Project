# NHANES Periodontitis Machine Learning Project
## Complete Project Briefing for Independent Development

**Author**: Francisco Teixeira Barbosa (Tuminha)  
**Date**: November 2025  
**Purpose**: This document provides complete context for continuing development with Cursor AI, ChatGPT, or any AI assistant.

---

## TABLE OF CONTENTS

1. [Project Overview](#1-project-overview)
2. [Scientific Rationale](#2-scientific-rationale)
3. [Reference Papers](#3-reference-papers)
4. [Data Source: NHANES](#4-data-source-nhanes)
5. [Methodology](#5-methodology)
6. [Technical Implementation](#6-technical-implementation)
7. [Publication Strategy](#7-publication-strategy)
8. [Step-by-Step Development Plan](#8-step-by-step-development-plan)
9. [Key Code Files Already Created](#9-key-code-files-already-created)
10. [Questions for AI Assistants](#10-questions-for-ai-assistants)

---

## 1. PROJECT OVERVIEW

### What We're Building

A machine learning pipeline to predict periodontitis (gum disease) using publicly available NHANES data from the US CDC. The goal is to:

1. **Replicate** the methodology from Bashir et al. (2022) published in *Journal of Clinical Periodontology*
2. **Improve upon it** by adding modern gradient boosting algorithms (XGBoost, CatBoost, LightGBM) that were NOT tested in the original paper
3. **Use better validation** (temporal validation instead of geographic cross-validation)
4. **Publish the results** first as a preprint, then in a peer-reviewed journal

### Why This Project Matters

- Periodontitis affects ~50% of US adults over 30
- Early prediction could enable preventive interventions
- The original paper showed a critical problem: models that looked great internally (AUC > 0.95) failed completely on external validation (AUC dropped to 0.50-0.60)
- Modern algorithms and better validation strategies might solve this

### The Gap We're Filling

From our analysis of existing literature:
- **Bashir et al. (2022)** tested 10 algorithms but did NOT include XGBoost, CatBoost, or LightGBM
- **Polizzi et al. (2024)** systematic review confirmed: only ONE study ever used XGBoost, ZERO used CatBoost or LightGBM
- This is a clear methodological gap we can fill

---

## 2. SCIENTIFIC RATIONALE

### The Problem with Bashir et al. (2022)

Bashir's team achieved amazing internal validation results (AUC > 0.95) but when they tested their models on a different population:
- **Taiwan cohort → US cohort**: AUC dropped to ~0.50-0.60
- **US cohort → Taiwan cohort**: Same problem

This is a classic case of **overfitting to population-specific characteristics**.

### Our Hypothesis

The poor external validation might be due to:
1. **Cross-population validation** being too strict (different healthcare systems, diagnostic criteria)
2. **Algorithm limitations** - older algorithms may not capture complex feature interactions
3. **Feature quality** - the 15 predictors used are "crude" (demographics, self-reported data)

### Our Approach

1. **Temporal validation** instead of geographic: Train on 2011-2014 data, validate on 2015-2016, test on 2017-2018
2. **Same population** (US NHANES) across all splits - removes confounding from different healthcare systems
3. **Modern gradient boosting** algorithms with hyperparameter optimization
4. **SHAP interpretability** to understand which features actually matter

### Realistic Expectations

We must be honest: if the predictors themselves are the bottleneck (which Bashir's paper suggests), better algorithms won't magically fix this. However:
- We might achieve better temporal generalization
- We can provide interpretability insights (SHAP)
- We can establish a methodological benchmark for future research

---

## 3. REFERENCE PAPERS

### Primary Reference (to replicate and improve)

**Bashir NZ, Gill S, Tawse-Smith A, Torkzaban P, Graf D, Gary MT. Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. J Clin Periodontol. 2022;49:958-969.**

Key details:
- Used NHANES 2011-2012 data (US) and Taiwan cohort
- Tested 10 algorithms: AdaBoost, ANN, Decision Trees, Gaussian Process, KNN, Linear SVC, LDA, Logistic Regression, Random Forest, Naïve Bayes
- **DID NOT TEST**: XGBoost, CatBoost, LightGBM
- Used 15 predictors (demographics, metabolic health, oral health behaviors)
- Followed TRIPOD 2015 guidelines
- Internal AUC > 0.95, External AUC ~0.50-0.60

### Secondary Reference (systematic review confirming the gap)

**Talal Al-Shammari A, Alsulaiman AA, Existing research and their findings regarding the use of artificial intelligence and machine learning in predicting periodontitis: A systematic review. JDR Clinical & Translational Research. 2024.**

Also: **Polizzi A, et al. (2024)** - Confirmed that the field is dominated by basic algorithms (ANNs, SVMs, decision trees)

Key finding: "None of the included articles used more powerful networks" - explicitly states XGBoost/CatBoost/LightGBM are underutilized

### Methodological Reference (CDC/AAP Case Definitions)

**Eke PI, Page RC, Wei L, Thornton-Evans G, Genco RJ. Update of the case definitions for population-based surveillance of periodontitis. J Periodontol. 2012;83(12):1449-1454.**

This paper defines how to classify periodontitis severity (mild/moderate/severe) from clinical measurements.

---

## 4. DATA SOURCE: NHANES

### What is NHANES?

The **National Health and Nutrition Examination Survey** is conducted by the US CDC. It's:
- Free and publicly available
- Nationally representative of the US population
- Conducted in 2-year cycles
- Includes comprehensive health examinations

### Data Access

**Main portal**: https://wwwn.cdc.gov/nchs/nhanes/

Data is provided as SAS transport files (.XPT) that can be read directly with pandas:
```python
import pandas as pd
df = pd.read_sas("https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT")
```

### Available Periodontal Data Cycles

| Cycle | Status | Sample Size (adults 30+) |
|-------|--------|--------------------------|
| 2009-2010 | ✅ Available | ~3,500 |
| 2011-2012 | ✅ Available | ~3,500 |
| 2013-2014 | ✅ Available | ~3,500 |
| 2015-2016 | ✅ Available | ~3,500 |
| 2017-2018 | ✅ Available | ~3,500 |
| 2017-March 2020 | ✅ Combined (COVID interrupted) | ~5,000 |

**Note**: Full-mouth periodontal examination (the gold standard) started in 2009-2010.

### Files Needed Per Cycle

For each cycle, download and merge these components:

| Component | File Prefix | Variables |
|-----------|-------------|-----------|
| Demographics | DEMO | Age, sex, education, race |
| Body Measures | BMX | BMI, waist circumference |
| Blood Pressure | BPX | Systolic/diastolic BP |
| Smoking | SMQ | Smoking history |
| Alcohol | ALQ | Alcohol consumption |
| Oral Health Questionnaire | OHQ | Dental visits, flossing |
| Periodontal Exam | OHXPER | Pocket depth, attachment loss |
| Fasting Glucose | GLU | Blood glucose |
| Lipids | TRIGLY, HDL | Triglycerides, HDL cholesterol |

File naming convention:
- 2011-2012: `DEMO_G.XPT`, `BMX_G.XPT`, etc.
- 2013-2014: `DEMO_H.XPT`, `BMX_H.XPT`, etc.
- 2015-2016: `DEMO_I.XPT`, `BMX_I.XPT`, etc.
- 2017-2018: `DEMO_J.XPT`, `BMX_J.XPT`, etc.

---

## 5. METHODOLOGY

### Periodontitis Case Definition (CDC/AAP)

From the periodontal examination data, classify participants using these criteria:

**Severe periodontitis**:
- ≥2 interproximal sites with Clinical Attachment Loss (CAL) ≥6 mm (on different teeth) AND
- ≥1 interproximal site with Pocket Depth (PD) ≥5 mm

**Moderate periodontitis**:
- ≥2 interproximal sites with CAL ≥4 mm (on different teeth) OR
- ≥2 interproximal sites with PD ≥5 mm (on different teeth)

**Mild periodontitis**:
- ≥2 interproximal sites with CAL ≥3 mm AND
- ≥2 interproximal sites with PD ≥4 mm

**Binary outcome**: Any periodontitis (mild + moderate + severe) vs. No periodontitis

### The 15 Predictors (from Bashir et al.)

**Demographics (3)**:
1. Age (years)
2. Sex (male/female)
3. Education (less than high school vs. high school or above)

**Health Behaviors (2)**:
4. Smoking status (never vs. former/current)
5. Alcohol consumption (never vs. former/current)

**Metabolic Health (7)**:
6. Body Mass Index (BMI, kg/m²)
7. Waist circumference (cm)
8. Systolic blood pressure (mmHg)
9. Diastolic blood pressure (mmHg)
10. Fasting plasma glucose (mg/dL)
11. Serum triglycerides (mg/dL)
12. HDL cholesterol (mg/dL)

**Oral Health (3)**:
13. Dental visit in last year (yes/no)
14. Noticed mobile/loose teeth (yes/no)
15. Uses dental floss (yes/no)

### Data Splitting Strategy (Temporal Validation)

This is DIFFERENT from Bashir's approach and is our methodological improvement:

```
TRAIN SET:     2011-2012 + 2013-2014  (~7,000 participants)
VALIDATION SET: 2015-2016              (~3,500 participants)
TEST SET:       2017-2018              (~3,500 participants)
```

**Why temporal validation?**
- Mimics real-world deployment: "Can a model trained on past data predict future patients?"
- Avoids data leakage
- Same population (US), same methodology across years
- More realistic than random splits

### Algorithms to Compare

**Baseline (Bashir's algorithms)**:
1. Logistic Regression
2. Random Forest
3. Decision Tree
4. K-Nearest Neighbors
5. Naive Bayes
6. AdaBoost
7. Support Vector Machine (SVM)
8. Linear Discriminant Analysis (LDA)
9. Artificial Neural Network (MLP)

**Our additions (the gap we're filling)**:
10. XGBoost (with Optuna hyperparameter tuning)
11. CatBoost (handles categorical features natively)
12. LightGBM (fast gradient boosting)
13. Ensemble/Stacking (combine best models)

### Hyperparameter Optimization

Use **Optuna** for Bayesian hyperparameter optimization:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        # ... more parameters
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

### Evaluation Metrics

Report all of these for each model:

1. **AUC-ROC** (primary metric) - Area Under the Receiver Operating Characteristic Curve
2. **Accuracy**
3. **Sensitivity (Recall)** - True positive rate
4. **Specificity** - True negative rate
5. **Precision** - Positive predictive value
6. **F1-Score** - Harmonic mean of precision and recall
7. **Brier Score** - Calibration metric

Report metrics for:
- Internal validation (cross-validation on training set)
- Temporal validation (validation set: 2015-2016)
- Final test (test set: 2017-2018)

### Interpretability with SHAP

Use SHAP (SHapley Additive exPlanations) to understand feature importance:

```python
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, feature_names=feature_names)

# Bar plot (mean absolute SHAP values)
shap.summary_plot(shap_values, X_test, plot_type='bar')
```

This adds clinical value by showing which predictors are most important.

---

## 6. TECHNICAL IMPLEMENTATION

### Python Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy pyarrow scikit-learn xgboost catboost lightgbm optuna shap matplotlib seaborn plotly jupyter
```

### Project Structure

```
nhanes_periodontitis_ml/
├── data/
│   ├── raw/                    # Downloaded NHANES XPT files
│   │   ├── 2011_2012/
│   │   ├── 2013_2014/
│   │   ├── 2015_2016/
│   │   └── 2017_2018/
│   └── processed/              # Cleaned, merged parquet files
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_comparison.ipynb
├── src/
│   ├── data_download.py
│   ├── data_processing.py
│   ├── periodontitis_labels.py
│   ├── feature_engineering.py
│   ├── models.py
│   └── evaluation.py
├── models/                     # Saved trained models
├── results/                    # Outputs, figures, tables
├── requirements.txt
└── README.md
```

### Key Technical Challenges

1. **NHANES variable names are cryptic**: e.g., `OHX02PCM` = Pocket depth, tooth 02, mesial site
2. **Periodontal exam has 6 sites per tooth × 28 teeth**: Need to aggregate correctly
3. **Missing data**: Use median imputation or model-based imputation
4. **Sample weights**: NHANES uses complex survey design - for publication, should use weights
5. **Merging multiple files**: All files share `SEQN` (participant ID)

### NHANES Periodontal Variable Pattern

```
OHXxxPCM = Pocket depth, tooth xx, Mesial site
OHXxxPCD = Pocket depth, tooth xx, Distal site
OHXxxPCS = Pocket depth, tooth xx, mid-buccal/lingual Site
OHXxxLAM = Loss of attachment (CAL), tooth xx, Mesial
OHXxxLAD = Loss of attachment (CAL), tooth xx, Distal
OHXxxLAS = Loss of attachment (CAL), tooth xx, mid-Site

Where xx = tooth number (02-15 upper jaw, 18-31 lower jaw)
Third molars (01, 16, 17, 32) are excluded
```

---

## 7. PUBLICATION STRATEGY

### Phase 1: Preprint (Immediate)

**Target**: arXiv, medRxiv, or bioRxiv

Benefits:
- Establishes priority
- Gets feedback from community
- Can be done in 2-4 weeks
- Creates citable DOI

### Phase 2: Peer-Reviewed Journal

**Primary targets** (in order of preference):

1. **Journal of Clinical Periodontology** (IF: ~6.0)
   - Same journal as Bashir et al. - direct comparison
   - Accepts methodological papers
   
2. **Journal of Periodontology** (IF: ~4.0)
   - Official journal of AAP
   - Good fit for ML applications

3. **Journal of Dental Research** (IF: ~5.0)
   - High impact, broad readership
   - Accepts innovative methodologies

4. **Dentomaxillofacial Radiology** (IF: ~2.5)
   - Accepts AI/ML papers
   - Less competitive

### Paper Structure (TRIPOD Guidelines)

Follow TRIPOD (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis):

1. **Title**: Include "development and validation" and "NHANES"
2. **Abstract**: Structured with Background, Methods, Results, Conclusions
3. **Introduction**: Gap in literature, rationale for gradient boosting
4. **Methods**: Data source, participants, predictors, outcome, algorithms, validation
5. **Results**: Model performance tables, SHAP figures, ROC curves
6. **Discussion**: Comparison with Bashir, limitations, clinical implications
7. **Conclusion**: Summary and future directions

### Key Selling Points for the Paper

1. **First systematic comparison of gradient boosting methods** for periodontitis prediction
2. **Temporal validation** approach (more robust than cross-population)
3. **Larger dataset** than previous studies (4 NHANES cycles)
4. **Interpretability** with SHAP analysis
5. **Reproducible** - all code available on GitHub

---

## 8. STEP-BY-STEP DEVELOPMENT PLAN

### Week 1: Data Acquisition and Processing

- [ ] Download all NHANES cycles (2011-2018)
- [ ] Merge components by SEQN
- [ ] Implement CDC/AAP periodontitis case definitions
- [ ] Create the 15 Bashir predictors
- [ ] Handle missing data
- [ ] Create train/validation/test splits

### Week 2: Baseline Models

- [ ] Implement all 9 Bashir baseline algorithms
- [ ] Run 5-fold cross-validation on training set
- [ ] Evaluate on validation set
- [ ] Document performance metrics

### Week 3: Gradient Boosting Models

- [ ] Implement XGBoost, CatBoost, LightGBM
- [ ] Run Optuna hyperparameter optimization
- [ ] Compare with baseline models
- [ ] Evaluate on test set

### Week 4: Analysis and Interpretability

- [ ] Generate SHAP analysis for best model
- [ ] Create ROC curves and comparison plots
- [ ] Statistical significance testing
- [ ] Prepare results tables

### Week 5: Writing and Publication

- [ ] Write methods section
- [ ] Write results section
- [ ] Create figures (SHAP plots, ROC curves, comparison tables)
- [ ] Write discussion
- [ ] Submit to preprint server

### Week 6+: Journal Submission

- [ ] Revise based on preprint feedback
- [ ] Format for target journal
- [ ] Submit to peer-reviewed journal
- [ ] Respond to reviewer comments

---

## 9. KEY CODE FILES ALREADY CREATED

The following files have been created and are ready to use:

### 01_download_nhanes_data.py
- Downloads NHANES XPT files from CDC
- Saves as parquet for faster loading
- Handles multiple cycles

### 02_process_nhanes_data.py
- Merges NHANES components by SEQN
- Creates readable variable names
- Filters to adults 30+
- Implements CDC/AAP periodontitis classification (partial)

### 03_train_models.py
- Defines baseline and gradient boosting models
- Optuna hyperparameter optimization
- Evaluation metrics
- SHAP analysis (partial)

### requirements.txt
- All Python dependencies

### README.md
- Project overview

---

## 10. QUESTIONS FOR AI ASSISTANTS

When working with Cursor AI or ChatGPT, use these prompts:

### For Data Processing

```
"Help me download NHANES 2011-2012 periodontal data from the CDC website 
and merge it with the demographics file. The periodontal file is OHXPER_G.XPT 
and demographics is DEMO_G.XPT. They share the SEQN participant ID."
```

### For Periodontitis Classification

```
"Implement the CDC/AAP case definitions for periodontitis using NHANES 
periodontal exam data. I need to classify participants as having 
mild/moderate/severe periodontitis based on pocket depth (OHXxxPCx) 
and clinical attachment loss (OHXxxLAx) at interproximal sites."
```

### For Model Training

```
"Create an XGBoost classifier for periodontitis prediction with Optuna 
hyperparameter optimization. The features are: age, sex, education, 
smoking status, BMI, waist circumference, blood pressure, fasting glucose, 
triglycerides, HDL, dental visits, loose teeth, and flossing. 
Use AUC-ROC as the optimization metric."
```

### For SHAP Analysis

```
"Generate SHAP summary plots for my XGBoost periodontitis prediction model. 
I want to understand which of the 15 features are most important for 
predicting periodontitis. Create both a beeswarm plot and a bar plot."
```

### For Publication

```
"Help me write the Methods section for a paper on machine learning 
prediction of periodontitis using NHANES data. I need to follow 
TRIPOD guidelines and describe: data source, study population, 
predictors, outcome definition, algorithms compared, and validation strategy."
```

---

## SUMMARY

**In one sentence**: We are building machine learning models to predict periodontitis using free NHANES data, specifically to test whether modern gradient boosting algorithms (XGBoost, CatBoost, LightGBM) can outperform the algorithms tested by Bashir et al. (2022), with the goal of publishing in a peer-reviewed periodontal journal.

**Key differentiators**:
1. First to systematically test gradient boosting methods for this task
2. Temporal validation (2011-2014 → 2015-2016 → 2017-2018) instead of cross-population
3. SHAP interpretability analysis
4. Larger combined dataset

**Realistic outcome**: Even if we don't dramatically improve on Bashir's results, demonstrating that gradient boosting methods don't solve the external validation problem is itself a publishable finding that advances the field.

---

*Document created for project continuity. Last updated: November 2025*
