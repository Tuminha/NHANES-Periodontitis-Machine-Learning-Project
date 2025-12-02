# 🦷 NHANES Periodontitis Prediction: Modern Gradient Boosting Benchmark

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Status](https://img.shields.io/badge/Status-Publication%20Ready-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Systematic comparison of XGBoost, CatBoost, and LightGBM for periodontitis prediction using NHANES 2011-2014**

[🎯 Overview](#-project-overview) • [📊 Results](#-results) • [🚀 Quick-Start](#-quick-start) • [📁 Structure](#-project-structure) • [📝 Citation](#-citation)

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

---

## 📊 Summary

**Models compared:** Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost.

**Cohort:** NHANES 2011–2014, adults 30+, CDC/AAP case definition.

**Training:** Stratified 5-fold CV, Optuna-tuned, monotonic constraints consistent with clinical priors.

**Calibration:** Isotonic on out-of-fold predictions.

**Missing data:** Kept natively for boosters, plus missingness indicators.

---

### 🏆 Model Selection

| Model | Description | Use Case |
|-------|-------------|----------|
| **Primary model** | v1.3-constrained without reverse-causality features (29 predictors) | Publication, clinical deployment |
| **Secondary model** | v1.3-constrained with reverse-causality features (33 predictors) | Supplementary analysis |

---

### 📈 Headline Results

- ✅ **Boosters are tied** at AUC ≈0.72–0.73 and PR-AUC ≈0.82–0.83
- ✅ **Calibration improves Brier** and yields clinically usable probabilities
- ✅ **Rule-out threshold** gives very high sensitivity with low specificity
- ✅ **Balanced threshold** gives moderate sensitivity and moderate specificity
- ✅ **Removing dental behavior variables** (reverse-causality) reduces AUC by ~1.1% and slightly improves balanced specificity

---

### 📋 Feature Sets

**Core clinical and demographic predictors (kept):**
- Age, sex, education
- Smoking (3-level: never/former/current), alcohol_current
- BMI, height_cm, waist_cm, waist_height ratio
- Systolic BP, diastolic BP, glucose, triglycerides, HDL
- Lab missingness indicators (`*_missing` flags)

**Dropped for primary model (treatment-linked signals):**
- `dental_visit`, `floss_days`, `mobile_teeth`, `floss_days_missing`

---

## 📊 Results

### Final Metrics

| Model Variant | AUC-ROC | PR-AUC | Rule-out Sens | Rule-out Spec | Balanced Sens | Balanced Spec |
|---------------|---------|--------|---------------|---------------|---------------|---------------|
| **v1.3 primary (no reverse-causality)** | **0.7172** | **0.8157** | **99.9%** | 12.4% | 72.8% | **59.2%** |
| v1.3 secondary (full 33 features) | 0.7255 | 0.8207 | 98.8% | 16.8% | 75.4% | 57.7% |

**Interpretation:** Reverse-causality features contribute ~0.008 AUC and a few points of rule-out specificity, but they are not essential. The balanced operating point slightly benefits from dropping them.

---

### 🎯 Clinical Operating Points

**❌ Target A NOT Achievable:** Cannot achieve Recall ≥90% AND Specificity ≥35% simultaneously (fundamental feature set limitation)

| Operating Point | Threshold | Recall | Specificity | NPV | F1 | Use Case |
|-----------------|-----------|--------|-------------|-----|-----|----------|
| **📍 Rule-Out** | 0.35 | **99.9%** | 12.4% | 96% | 0.818 | Screening (negative = likely healthy) |
| **📍 Balanced** | 0.65 | 72.8% | **59.2%** | 51% | 0.758 | Diagnosis (optimal Youden J=0.32) |

<div align="center">
<img src="figures/14_v13_operating_points.png" alt="v1.3 Operating Points" width="800"/>
</div>

**Clinical Interpretation:**
- **Rule-Out (t=0.35):** If test is negative, 96% chance patient is truly healthy. Use for initial screening.
- **Balanced (t=0.65):** Best tradeoff between sensitivity and specificity. Use for clinical decisions.

---

### 🔬 Missing Data Ablation (v1.3)

| Strategy | AUC | Sample Size | Notes |
|----------|-----|-------------|-------|
| **Full model (native NaNs + indicators)** | **~0.725** | 9,379 | Best performance |
| Remove indicators | ~0.72 | 9,379 | Small drop (~0.5-1% AUC) |
| Complete-case only | ~0.68 | ~4,500 | Large drop, halves sample |

**Conclusion:** Treat missingness as information rather than noise.

<div align="center">
<img src="figures/18_nan_ablation.png" alt="NaN Ablation Results" width="800"/>
</div>

---

### 📐 Calibration

Isotonic calibration improved Brier by ~1.5–2% and corrected S-curve bias in mid-probability bins. Report calibrated probabilities everywhere.

<div align="center">
<img src="figures/12_calibration_analysis.png" alt="Calibration Analysis" width="800"/>
</div>

---

### 🔍 SHAP Feature Importance (Primary Model)

<div align="center">
<img src="figures/15_shap_beeswarm.png" alt="SHAP Beeswarm" width="800"/>
</div>

<div align="center">
<img src="figures/16_shap_importance.png" alt="SHAP Importance" width="800"/>
</div>

---

### 📈 Version Evolution

| Version | Key Change | AUC | Δ from Baseline |
|---------|------------|-----|-----------------|
| v1.0 | Baseline (imputation) | 0.7071 | - |
| v1.1 | Native NaN + missing indicators | 0.7267 | +2.8% |
| v1.2 | Ensemble + calibration | 0.7302 | +3.3% |
| **v1.3 primary** | **Monotonic + no reverse-causality** | **0.7172** | **+1.4%** |

---

### 📊 Model Comparison Visualizations

<div align="center">
<img src="figures/08_model_comparison_auc.png" alt="Model AUC Comparison" width="800"/>
</div>

<div align="center">
<img src="figures/09_model_comparison_metrics.png" alt="Multi-Metric Comparison" width="800"/>
</div>

<div align="center">
<img src="figures/11_model_comparison_significance.png" alt="Statistical Significance" width="800"/>
</div>

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

### How to Reproduce v1.3 Primary Model

```bash
# Run the master notebook
jupyter notebook notebooks/00_nhanes_periodontitis_end_to_end.ipynb

# Execute all cells through Section 22
# Primary model results saved to: results/v13_primary_norc_summary.json
```

---

## 📁 Project Structure

```
NHANES-Periodontitis-Machine-Learning-Project/
├── configs/
│   └── config.yaml                 # Central configuration
├── data/
│   ├── raw/                        # Downloaded NHANES XPT files
│   └── processed/                  # Cleaned, merged datasets
├── figures/                        # All plots (ROC, SHAP, calibration)
├── models/                         # Trained models (.pkl)
├── results/                        # Metrics JSON/CSV
│   ├── v13_primary_norc_summary.json   # Primary model
│   ├── v13_secondary_full_summary.json # Secondary model
│   ├── v13_operating_points.json       # Operating points
│   ├── v13_featuredrop.json            # Feature-drop analysis
│   └── v13_nan_ablation.json           # NaN ablation results
├── src/
│   ├── ps_plot.py                  # Periospot plotting style
│   ├── labels.py                   # CDC/AAP case definitions
│   ├── evaluation.py               # Metrics, ROC/PR, calibration
│   └── utils.py                    # Reproducibility, I/O
├── notebooks/
│   └── 00_nhanes_periodontitis_end_to_end.ipynb
├── ARTICLE_DRAFT.md                # Publication draft
├── CHANGELOG.md                    # Version history
└── README.md
```

---

## 📋 Decisions Log (Reproducibility)

### Feature Selection Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **Drop dental_visit, floss_days, mobile_teeth** | Reverse-causality (treatment-seeking) | -1.1% AUC, +1.5% balanced spec |
| **Keep waist_cm, waist_height** | Trees handle multicollinearity | +1-2 features |
| **3-level smoking** | Never/former/current more informative | Richer signal |
| **Native NaN handling** | "Missingness is informative" | +2.8% AUC |

### Modeling Decisions

| Decision | Rationale | Impact |
|----------|-----------|--------|
| **Monotonic constraints** | Biological plausibility | -0.8% AUC (acceptable) |
| **Isotonic calibration** | Better probability estimates | -1.6% Brier |
| **Soft-voting ensemble** | Combine 3 models | +0.0009 AUC |
| **Dual operating points** | Target A unachievable | Practical deployment |

---

## 🔬 Publication Strategy

### Proposed Title
**"Evaluating Modern Gradient Boosting Methods for Periodontitis Prediction: A Systematic Comparison Using NHANES 2011-2014"**

### Target Journals
1. **Journal of Clinical Periodontology** (IF 6.0) - Same venue as Bashir
2. **Journal of Periodontology** (IF 4.0) - ADA flagship
3. **BMC Oral Health** (IF 3.0) - Open access

### Compliance
- **TRIPOD 2015:** Transparent Reporting
- **STROBE:** Observational Studies
- **Open Science:** All code public on GitHub

---

## 📝 Citation

### BibTeX

```bibtex
@article{barbosa2025gradient,
  title={Evaluating Modern Gradient Boosting Methods for Periodontitis Prediction: A Systematic Comparison Using NHANES 2011-2014},
  author={Barbosa, Francisco Teixeira},
  journal={In preparation},
  year={2025},
  note={First systematic evaluation of modern gradient boosting for periodontitis prediction},
  url={https://github.com/Tuminha/NHANES-Periodontitis-Machine-Learning-Project}
}
```

---

## ⚠️ Limitations

1. **Single cohort with cross-validation only.** External validation on NHANES 2009–2010 or another national survey is required.

2. **High disease prevalence** in our sample versus CDC estimates mandates careful reconciliation of CDC/AAP coding pipeline and exam inclusion criteria.

3. **High sensitivity at rule-out operating point comes with low specificity;** health economic value depends on downstream pathways and costs.

4. **Missingness signals may partly reflect NHANES design;** portability to clinic-collected data must be tested.

5. **Reverse-causality features** (dental_visit, floss_days) may encode treatment history rather than risk. Primary model excludes these.

---

## 🛠 Technical Stack

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| Data Processing | Pandas, NumPy | 2.3.2, 2.3.5 | ETL & feature engineering |
| ML Framework | Scikit-learn | 1.7.1 | Pipelines, metrics |
| Gradient Boosting | XGBoost, CatBoost, LightGBM | 3.1, 1.2, 4.6 | Primary models |
| Hyperparameter Tuning | Optuna | 4.6.0 | Bayesian optimization |
| Interpretability | SHAP | 0.50.0 | Feature importance |

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
