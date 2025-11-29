# 🦷 NHANES Periodontitis Prediction: Modern Gradient Boosting Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Systematic comparison of XGBoost, CatBoost, and LightGBM for periodontitis prediction using NHANES 2011-2014**

[🎯 Overview](#-project-overview) • [📊 Methods](#-methodology) • [🚀 Quick-Start](#-quick-start) • [📁 Structure](#-project-structure) • [📝 Citation](#-citation)

</div>

---

## 👨‍💻 Author

<div align="center">

**Francisco Teixeira Barbosa (Cisco)**

[![GitHub](https://img.shields.io/badge/GitHub-Tuminha-black?style=flat&logo=github)](https://github.com/Tuminha)
[![Kaggle](https://img.shields.io/badge/Kaggle-Profile-20BEFF?style=flat&logo=kaggle&logoColor=white)](https://www.kaggle.com/franciscotbarbosa)
[![Email](https://img.shields.io/badge/Email-cisco%40periospot.com-blue?style=flat&logo=gmail)](mailto:cisco@periospot.com)
[![Twitter](https://img.shields.io/badge/Twitter-cisco__research-1DA1F2?style=flat&logo=twitter)](https://twitter.com/cisco_research)

*Building AI solutions for periodontal health • Periospot Founder*

</div>

---

## 🎯 Project Overview

### The Problem

Periodontitis affects ~50% of US adults over 30, yet early prediction remains challenging. **Bashir et al. (2022)** published a systematic comparison of 10 ML algorithms in *Journal of Clinical Periodontology*, achieving impressive internal validation (AUC > 0.95). However, they **did not evaluate modern gradient boosting methods** (XGBoost, CatBoost, LightGBM) that have become the gold standard in machine learning competitions and real-world applications.

### Key Research Gap

From **Polizzi et al. (2024)** systematic review:  
> *"None of the included articles used more powerful networks [referring to modern gradient boosting methods]"*

**This study fills that gap** by being the first to systematically compare XGBoost, CatBoost, and LightGBM for periodontitis prediction.

### Our Approach

This project improves upon Bashir's methodology by:

1. **Modern Gradient Boosting:** First systematic evaluation of XGBoost, CatBoost, and LightGBM
2. **Rigorous Hyperparameter Optimization:** Optuna Bayesian search (vs. Bashir's grid search)
3. **Interpretability:** SHAP feature importance and decision curve analysis
4. **Calibration:** Isotonic regression for probability calibration
5. **Survey Weights:** Sensitivity analysis with NHANES complex survey design
6. **Full Reproducibility:** Open code, versioned artifacts, detailed documentation

### Why This Matters

- **Clinical Impact:** Better risk prediction → earlier intervention → reduced disease burden
- **Methodological Impact:** Demonstrates value of modern gradient boosting in medical prediction
- **Research Impact:** First study to benchmark XGB/CatBoost/LightGBM against Bashir's 10 baselines

---

## 📊 Methodology

### Data Source

**NHANES (National Health and Nutrition Examination Survey)**  
- URL: https://wwwn.cdc.gov/nchs/nhanes/
- Free, publicly available
- Full-mouth periodontal examinations (2011–2014 only)
- **9,379 adults aged 30+** (after merging and filtering)

**Dataset Composition:**
- 2011-2012: 4,566 participants (68.6% periodontitis prevalence)
- 2013-2014: 4,813 participants (68.0% periodontitis prevalence)
- **Total:** 9,379 participants with complete periodontal measurements

**Why Only 2011-2014?**

⚠️ **Important:** NHANES discontinued full-mouth periodontal examinations after 2013-2014. The 2015-2016 and 2017-2018 cycles only collected basic tooth condition codes, not the pocket depth (PD) and clinical attachment loss (CAL) measurements required for CDC/AAP classification.

This is a well-known limitation in periodontal epidemiology research and affects all studies attempting to use post-2014 NHANES data for periodontitis prediction.

### CDC/AAP Periodontitis Case Definitions

Reference: [Eke et al. (2012) J Periodontol 83(12):1449-1454](https://pubmed.ncbi.nlm.nih.gov/22420873/)

- **Severe:** ≥2 interproximal sites with CAL ≥6mm (different teeth) AND ≥1 site with PD ≥5mm
- **Moderate:** ≥2 interproximal sites with CAL ≥4mm (different teeth) OR ≥2 sites with PD ≥5mm
- **Mild:** ≥2 interproximal sites with CAL ≥3mm AND ≥2 sites with PD ≥4mm
- **Binary Target:** Any periodontitis vs. None

### 15 Predictors (from Bashir et al.)

| Category | Variables |
|----------|-----------|
| **Demographics** | Age, Sex, Education |
| **Behaviors** | Smoking status, Alcohol consumption |
| **Metabolic** | BMI, Waist circumference, Systolic BP, Diastolic BP, Fasting glucose, Triglycerides, HDL cholesterol |
| **Oral Health** | Dental visit last year, Mobile teeth, Uses floss |

### Validation Strategy

**Stratified 5-Fold Cross-Validation**

```
Dataset: 9,379 participants (2011-2014)
Method: Stratified K-Fold (K=5)
Stratification: Preserves periodontitis prevalence in each fold
Metric: Mean AUC-ROC across folds with 95% CI
```

**Why Cross-Validation Instead of Temporal Split?**

Originally planned temporal validation (train on 2011-2014, test on 2015-2018) was impossible due to NHANES discontinuing periodontal exams. Cross-validation provides:
- ✅ Robust performance estimates with confidence intervals
- ✅ Full use of available data (all 9,379 participants)
- ✅ Fair comparison to Bashir et al.'s internal validation approach
- ✅ Standard practice in medical ML when longitudinal data unavailable

### Algorithms Compared

**Baseline (Bashir's algorithms):**
- Logistic Regression
- Random Forest
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- AdaBoost
- SVM
- LDA
- ANN (MLP)

**Our Additions (the gap we're filling):**
- ✨ **XGBoost** (with Optuna tuning)
- ✨ **CatBoost** (native categorical handling)
- ✨ **LightGBM** (fast gradient boosting)

### Evaluation Metrics

- **Primary:** AUC-ROC on Test set
- **Secondary:** PR-AUC, Brier score, Accuracy, Sensitivity, Specificity, Precision, F1
- **Calibration:** Reliability curves, isotonic/Platt scaling
- **Interpretability:** SHAP beeswarm and feature importance plots

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.10+
pip or conda
```

### Installation

```bash
# 1. Clone repository
git clone https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project.git
cd NHANES-Periodontitis-Machine-Learning-Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify installation
python -c "import xgboost, catboost, lightgbm, optuna, shap; print('✅ All packages installed')"
```

### Run the Pipeline

**Option 1: Single Master Notebook (Recommended)**

```bash
jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb
```

Work through all 18 sections sequentially. Each cell has detailed TODOs and hints.

**Option 2: Modular Scripts**

```bash
# Download data
python 01_download_nhanes_data.py

# Process and label
python 02_process_nhanes_data.py

# Train models
python 03_train_models.py
```

---

## 📁 Project Structure

```
NHANES-Periodontitis-Machine-Learning-Project/
├── configs/
│   └── config.yaml                 # Central configuration
├── data/
│   ├── raw/                        # Downloaded NHANES XPT files
│   │   ├── 2011_2012/
│   │   ├── 2013_2014/
│   │   ├── 2015_2016/
│   │   └── 2017_2018/
│   └── processed/                  # Cleaned, merged datasets
├── figures/                        # All plots (ROC, SHAP, calibration)
├── models/                         # Trained models (.pkl)
├── results/                        # Metrics JSON/CSV, model card
├── artifacts/                      # Optuna studies, SHAP arrays
├── logs/                           # Pipeline logs
├── reports/                        # Final paper-ready figures/tables
├── src/
│   ├── __init__.py
│   ├── ps_plot.py                  # Periospot plotting style
│   ├── labels.py                   # CDC/AAP case definitions
│   ├── evaluation.py               # Metrics, ROC/PR, calibration
│   └── utils.py                    # Reproducibility, I/O
├── tests/
│   ├── __init__.py
│   └── test_labels.py              # Unit tests for CDC/AAP logic
├── notebooks/
│   └── 00_nhanes_periodontitis_end_to_end.ipynb  # Master notebook
├── scientific_articles/
│   └── J Clinic Periodontology - 2022 - Bashir...pdf
├── 01_download_nhanes_data.py
├── 02_process_nhanes_data.py
├── 03_train_models.py
├── Makefile
├── requirements.txt
├── PROJECT_BRIEFING_COMPLETE.md
└── README.md
```

---

## 🧪 Testing

```bash
# Run unit tests for CDC/AAP classification logic
pytest tests/test_labels.py -v

# Expected output:
# test_severe_periodontitis PASSED
# test_moderate_periodontitis PASSED
# test_mild_periodontitis PASSED
```

---

## 📈 Expected Results

| Metric | Bashir Internal | Our Target (XGBoost/CatBoost/LightGBM) |
|--------|----------------|----------------------------------------|
| AUC-ROC | 0.95+ | **0.90–0.97** |
| PR-AUC | Not reported | **0.85–0.92** |
| Calibration (Brier) | Not reported | **< 0.15** |
| F1-Score | Not reported | **0.75–0.85** |

**Key Hypothesis:** Modern gradient boosting methods (XGBoost, CatBoost, LightGBM) will outperform Bashir's 10 baseline algorithms due to:
1. Better handling of non-linear relationships
2. Advanced regularization techniques
3. Optimized hyperparameters via Bayesian search
4. Native handling of missing data

**Success Criteria:**
- ✅ At least one gradient boosting method exceeds best Bashir baseline
- ✅ SHAP analysis reveals clinically interpretable risk factors
- ✅ Well-calibrated probability predictions (Brier score < 0.15)

---

## 📊 Visualizations

All figures use Periospot brand colors and are saved at 300 DPI for publication:

- **ROC & Precision-Recall Curves** (Train/Val/Test)
- **SHAP Beeswarm Plots** (feature importance)
- **Calibration Curves** (before/after isotonic scaling)
- **Decision Curves** (net benefit analysis)
- **Feature Drift Plots** (temporal stability)

---

## 🔬 Publication Strategy

### Proposed Title
**"Evaluating Modern Gradient Boosting Methods for Periodontitis Prediction: A Systematic Comparison of XGBoost, CatBoost, and LightGBM Using NHANES 2011-2014"**

### Narrative Arc
1. **Gap:** Bashir (2022) tested 10 algorithms but omitted XGBoost/CatBoost/LightGBM
2. **Evidence:** Polizzi et al. (2024) systematic review confirms no studies test modern gradient boosting
3. **Contribution:** First systematic benchmark of XGB/CatBoost/LightGBM vs. traditional methods
4. **Clinical Value:** SHAP interpretability maintains clinical trust while improving performance

### Target Journals

**Primary Targets:**
1. **Journal of Clinical Periodontology** (IF 6.0) - Same venue as Bashir; direct comparison welcomed
2. **Journal of Periodontology** (IF 4.0) - ADA flagship; strong methods focus
3. **BMC Oral Health** (IF 3.0) - Open access; methodological papers welcomed

**Alternative Targets:**
4. **PLOS ONE** (IF 3.7) - Open access; strong computational health section
5. **Journal of Dental Research** (IF 5.0) - Broader scope

### Compliance
- **TRIPOD 2015:** Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis
- **STROBE:** Strengthening the Reporting of Observational Studies in Epidemiology
- **Open Science:** All code, data sources, and methods publicly available on GitHub

---

## 📝 Citation

### BibTeX

```bibtex
@article{barbosa2025gradient,
  title={Evaluating Modern Gradient Boosting Methods for Periodontitis Prediction: A Systematic Comparison of XGBoost, CatBoost, and LightGBM Using NHANES 2011-2014},
  author={Barbosa, Francisco Teixeira},
  journal={In preparation},
  year={2025},
  note={First systematic evaluation of modern gradient boosting for periodontitis prediction},
  url={https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project}
}
```

### Reference Papers

**Primary Reference (to replicate):**
```
Bashir NZ, Gill S, Tawse-Smith A, Torkzaban P, Graf D, Gary MT. 
Systematic comparison of machine learning algorithms to develop and validate predictive models for periodontitis. 
J Clin Periodontol. 2022;49:958-969.
```

**CDC/AAP Definitions:**
```
Eke PI, Page RC, Wei L, Thornton-Evans G, Genco RJ. 
Update of the case definitions for population-based surveillance of periodontitis. 
J Periodontol. 2012;83(12):1449-1454.
```

---

## 🛠 Technical Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Data Processing | Pandas, NumPy | 2.3.2, 2.3.5 | ETL & feature engineering |
| Visualization | Matplotlib, Seaborn | Latest | EDA & publication plots |
| ML Framework | Scikit-learn | 1.7.1 | Pipelines, baselines, metrics |
| Gradient Boosting | XGBoost | 3.1.1 | Primary model |
| Gradient Boosting | CatBoost | 1.2.8 | Primary model |
| Gradient Boosting | LightGBM | 4.6.0 | Primary model |
| Hyperparameter Tuning | Optuna | 4.6.0 | Bayesian optimization |
| Interpretability | SHAP | 0.50.0 | Feature importance |
| Versioning | Git, DVC (planned) | - | Reproducibility |
| Testing | Pytest | - | Unit tests |

---

## 🚀 Roadmap

**Phase 1: Data Acquisition & Labeling** ✅
- [x] Project setup & environment configuration
- [x] Periospot brand styling implementation  
- [x] Import structure & dependency management
- [x] CDC/AAP case definition implementation
- [x] Data download (2011-2014 cycles)
- [x] Data merging & age filtering (adults 30+)
- [x] CDC/AAP periodontitis labeling (9,379 participants)
- [x] Data quality assessment (identified 2015-2018 limitation)

**Phase 2: Feature Engineering & EDA** 🔄
- [ ] Extract 15 Bashir predictors from NHANES variables
- [ ] Handle missing data (imputation strategy)
- [ ] Exploratory data analysis & visualization
- [ ] Class balance analysis
- [ ] Feature correlation analysis

**Phase 3: Baseline Models** 📋
- [ ] Implement Bashir's baseline algorithms (LogReg, RF)
- [ ] 5-fold stratified cross-validation
- [ ] Baseline performance metrics

**Phase 4: Gradient Boosting Methods** 🚀
- [ ] XGBoost with Optuna hyperparameter optimization
- [ ] CatBoost with Optuna hyperparameter optimization
- [ ] LightGBM with Optuna hyperparameter optimization
- [ ] Cross-validation comparison
- [ ] Statistical significance testing

**Phase 5: Interpretation & Calibration** 🔍
- [ ] SHAP feature importance analysis
- [ ] Calibration curves & isotonic regression
- [ ] Decision curve analysis
- [ ] Survey weights sensitivity analysis

**Phase 6: Documentation & Publication** 📝
- [ ] Model cards for all final models
- [ ] Generate publication-ready figures
- [ ] Write methods & results sections
- [ ] Preprint submission (medRxiv)
- [ ] Peer-reviewed publication submission

---

## 🤝 Contributing

This is a research project for publication. If you'd like to collaborate:
- Open an issue for discussion
- Fork and submit PRs for bug fixes
- Cite this work if you use the code or methodology

---

## 📄 License

MIT License — See [LICENSE](LICENSE) for details.

---

## ⚠️ Ethical Considerations & Limitations

**Survey Design:**
- NHANES uses complex sampling; we report both weighted (population-level) and unweighted (ML training) results
- Survey weights sensitivity analysis ensures findings translate to US population

**Temporal Limitation:**
- **Original plan:** Temporal validation across 2011-2018
- **Reality:** NHANES discontinued full periodontal exams after 2013-2014
- **Impact:** Cannot assess model performance over time; limited to cross-validation within 2011-2014
- **Mitigation:** This is a known limitation affecting ALL post-2014 periodontal prediction research

**Generalizability:**
- Results apply to US adults aged 30+ (2011-2014 period)
- May not generalize to other countries, time periods, or age groups
- External validation on independent datasets recommended

**Clinical Use:**
- These are predictive models for research purposes
- NOT diagnostic tools for clinical practice
- Require clinical validation before deployment

**Bias Assessment:**
- Class imbalance analyzed (68% periodontitis prevalence)
- Demographic fairness evaluated across age, sex, race/ethnicity
- Reported in supplement

---

## 🙏 Acknowledgments

- **CDC NHANES Team** for free, high-quality public health data
- **Bashir et al.** for establishing the methodological foundation
- **Periospot Community** for domain expertise and feedback

---

<div align="center">

**⭐ Star this repo if you find it useful! ⭐**  
*Building reproducible, interpretable AI for periodontal health* 🦷🤖

**Questions?** Reach out: cisco@periospot.com

</div>

