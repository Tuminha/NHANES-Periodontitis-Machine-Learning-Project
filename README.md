# 🦷 NHANES Periodontitis Prediction: Temporal Validation & Gradient Boosting Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Replicating and improving upon Bashir et al. (2022) with temporal validation and modern gradient boosting methods**

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

Periodontitis affects ~50% of US adults over 30, yet early prediction remains challenging. **Bashir et al. (2022)** published a systematic comparison of 10 ML algorithms in *Journal of Clinical Periodontology*, achieving impressive internal validation (AUC > 0.95) but **poor external validation** (AUC ~0.50–0.60) when applied to different populations.

### Our Approach

This project replicates Bashir's methodology and improves it by:

1. **Temporal Validation:** Train on 2011–2014, validate on 2015–2016, test on 2017–2018 (same population, different time)
2. **Modern Gradient Boosting:** XGBoost, CatBoost, and LightGBM with Optuna hyperparameter optimization
3. **Rigorous Interpretation:** SHAP analysis, calibration curves, and decision curve analysis
4. **Survey Weights:** Sensitivity analysis with NHANES complex survey design
5. **Full Reproducibility:** Open code, versioned artifacts, and detailed documentation

### Key Research Gap

From Polizzi et al. (2024) systematic review: **No studies have systematically tested XGBoost, CatBoost, or LightGBM for periodontitis prediction.** This project fills that gap.

---

## 📊 Methodology

### Data Source

**NHANES (National Health and Nutrition Examination Survey)**  
- URL: https://wwwn.cdc.gov/nchs/nhanes/
- Free, publicly available
- Full-mouth periodontal examinations (2011–2018)
- ~14,000 adults aged 30+

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

### Temporal Validation Strategy

```
TRAIN:      2011-2012 + 2013-2014  (~7,000 participants)
VALIDATION: 2015-2016               (~3,500 participants)
TEST:       2017-2018               (~3,500 participants)
```

**Why temporal?** Mimics real-world deployment: "Can a model trained on past data predict future patients?"

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

| Metric | Bashir Internal | Bashir External | Our Target |
|--------|----------------|-----------------|------------|
| AUC-ROC | 0.95+ | 0.50–0.60 | 0.75–0.85 |
| PR-AUC | Not reported | Not reported | 0.60–0.75 |
| Temporal generalization | N/A | Poor | **Better** |

**Key Insight:** Even if we don't dramatically improve AUC, demonstrating that gradient boosting doesn't solve external validation is a publishable finding.

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

### Phase 1: Preprint (Immediate)
- **Target:** medRxiv or bioRxiv
- **Timeline:** 2–4 weeks after results

### Phase 2: Peer-Reviewed Journal
**Primary Targets:**
1. *Journal of Clinical Periodontology* (IF ~6.0) — same as Bashir
2. *Journal of Periodontology* (IF ~4.0)
3. *Journal of Dental Research* (IF ~5.0)

### TRIPOD Compliance
This study follows [TRIPOD 2015 guidelines](https://www.tripod-statement.org/) for transparent reporting of prediction models.

---

## 📝 Citation

### BibTeX

```bibtex
@article{barbosa2025nhanes,
  title={Temporal Validation and Gradient Boosting Benchmark for Periodontitis Prediction Using NHANES Data},
  author={Barbosa, Francisco Teixeira},
  journal={In preparation},
  year={2025},
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

- [x] Project setup & environment configuration
- [x] Periospot brand styling implementation
- [x] Import structure & dependency management
- [x] CDC/AAP case definition implementation
- [x] Temporal split strategy
- [ ] Data download & preprocessing
- [ ] Feature engineering (15 Bashir predictors)
- [ ] Exploratory data analysis
- [ ] Baseline model comparison (LogReg, RF)
- [ ] Gradient boosting with Optuna (XGBoost, CatBoost, LightGBM)
- [ ] SHAP analysis & interpretability
- [ ] Calibration & decision curves
- [ ] Survey weights sensitivity analysis
- [ ] Model cards & documentation
- [ ] Preprint submission
- [ ] Peer-reviewed publication

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

## ⚠️ Ethical Considerations

- **Survey Design:** NHANES uses complex sampling; population-level estimates require survey weights
- **Generalizability:** Results apply to US adults 30+; may not generalize to other populations
- **Clinical Use:** These are predictive models for research, not diagnostic tools for clinical practice
- **Bias:** We assess class imbalance and potential demographic biases in sensitivity analyses

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

