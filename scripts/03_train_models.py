"""
NHANES Periodontitis Prediction Project
Step 3: Train and Compare Models

This script:
1. Loads processed NHANES data
2. Creates temporal train/validation/test splits
3. Trains baseline models (Bashir's algorithms)
4. Trains gradient boosting models (XGBoost, CatBoost, LightGBM)
5. Compares performance with multiple metrics
6. Generates SHAP interpretability analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report, roc_curve
)
from sklearn.pipeline import Pipeline

# Baseline models (Bashir's)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Gradient Boosting (our additions)
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not installed. Run: pip install catboost")

# Hyperparameter optimization
import optuna
from optuna.samplers import TPESampler

# Interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not installed. Run: pip install shap")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
DATA_DIR = Path("data")
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path("results")
MODELS_DIR = Path("models")

for d in [RESULTS_DIR, MODELS_DIR]:
    d.mkdir(exist_ok=True, parents=True)


# =============================================================================
# Data Loading and Splitting
# =============================================================================

def load_and_split_data(target_col: str = 'periodontitis_binary'):
    """
    Load processed data and create temporal train/val/test splits.
    
    Split strategy:
    - Train: 2011-2012 + 2013-2014
    - Validation: 2015-2016
    - Test: 2017-2018
    """
    df = pd.read_parquet(PROCESSED_DIR / "nhanes_combined.parquet")
    
    print(f"Loaded {len(df)} participants")
    print(f"Cycles: {df['cycle'].value_counts().to_dict()}")
    
    # Define splits
    train_cycles = ['2011-2012', '2013-2014']
    val_cycles = ['2015-2016']
    test_cycles = ['2017-2018']
    
    df_train = df[df['cycle'].isin(train_cycles)].copy()
    df_val = df[df['cycle'].isin(val_cycles)].copy()
    df_test = df[df['cycle'].isin(test_cycles)].copy()
    
    print(f"\nSplit sizes:")
    print(f"  Train: {len(df_train)} ({df_train['cycle'].value_counts().to_dict()})")
    print(f"  Val:   {len(df_val)} ({df_val['cycle'].value_counts().to_dict()})")
    print(f"  Test:  {len(df_test)} ({df_test['cycle'].value_counts().to_dict()})")
    
    return df_train, df_val, df_test


def prepare_features(df: pd.DataFrame, feature_cols: list, target_col: str):
    """
    Prepare features and target for modeling.
    Handle missing values and ensure correct dtypes.
    """
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    # Remove rows with missing target
    valid_idx = ~y.isna()
    X = X[valid_idx]
    y = y[valid_idx].astype(int)
    
    return X, y


# =============================================================================
# Model Definitions
# =============================================================================

def get_baseline_models():
    """
    Get baseline models from Bashir et al. (2022).
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'LDA': LinearDiscriminantAnalysis(),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42),
    }
    return models


def get_gradient_boosting_models():
    """
    Get gradient boosting models (our additions).
    """
    models = {
        'XGBoost': xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42,
            verbose=-1
        ),
    }
    
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=42,
            verbose=0
        )
    
    return models


# =============================================================================
# Hyperparameter Optimization with Optuna
# =============================================================================

def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimize XGBoost hyperparameters using Optuna.
    """
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }
        
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest XGBoost AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


def optimize_catboost(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Optimize CatBoost hyperparameters using Optuna.
    """
    if not CATBOOST_AVAILABLE:
        return None
        
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 50, 300),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_state': 42,
            'verbose': 0
        }
        
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, y_pred_proba)
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest CatBoost AUC: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    
    return study.best_params


# =============================================================================
# Training and Evaluation
# =============================================================================

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name):
    """
    Train model and evaluate on all splits.
    """
    # Train
    model.fit(X_train, y_train)
    
    results = {'model': model_name}
    
    for split_name, X, y in [('train', X_train, y_train), 
                              ('val', X_val, y_val), 
                              ('test', X_test, y_test)]:
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        results[f'{split_name}_auc'] = roc_auc_score(y, y_pred_proba)
        results[f'{split_name}_accuracy'] = accuracy_score(y, y_pred)
        results[f'{split_name}_precision'] = precision_score(y, y_pred)
        results[f'{split_name}_recall'] = recall_score(y, y_pred)
        results[f'{split_name}_f1'] = f1_score(y, y_pred)
    
    return results, model


def run_all_models(X_train, y_train, X_val, y_val, X_test, y_test, 
                   optimize_hyperparams=False, n_trials=30):
    """
    Train and evaluate all models.
    """
    all_results = []
    trained_models = {}
    
    # Baseline models
    print("\n" + "="*60)
    print("Training Baseline Models (Bashir et al.)")
    print("="*60)
    
    for name, model in get_baseline_models().items():
        print(f"\n  Training {name}...")
        try:
            results, fitted_model = evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test, name
            )
            all_results.append(results)
            trained_models[name] = fitted_model
            print(f"    Val AUC: {results['val_auc']:.4f}, Test AUC: {results['test_auc']:.4f}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Gradient boosting models
    print("\n" + "="*60)
    print("Training Gradient Boosting Models (Our Additions)")
    print("="*60)
    
    for name, model in get_gradient_boosting_models().items():
        print(f"\n  Training {name}...")
        try:
            results, fitted_model = evaluate_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test, name
            )
            all_results.append(results)
            trained_models[name] = fitted_model
            print(f"    Val AUC: {results['val_auc']:.4f}, Test AUC: {results['test_auc']:.4f}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Optimized models
    if optimize_hyperparams:
        print("\n" + "="*60)
        print("Optimizing Hyperparameters with Optuna")
        print("="*60)
        
        # XGBoost
        print("\n  Optimizing XGBoost...")
        best_xgb_params = optimize_xgboost(X_train, y_train, X_val, y_val, n_trials)
        best_xgb_params['random_state'] = 42
        best_xgb_params['use_label_encoder'] = False
        best_xgb_params['eval_metric'] = 'logloss'
        xgb_optimized = xgb.XGBClassifier(**best_xgb_params)
        results, fitted_model = evaluate_model(
            xgb_optimized, X_train, y_train, X_val, y_val, X_test, y_test, 'XGBoost (Optimized)'
        )
        all_results.append(results)
        trained_models['XGBoost (Optimized)'] = fitted_model
        
        # CatBoost
        if CATBOOST_AVAILABLE:
            print("\n  Optimizing CatBoost...")
            best_cat_params = optimize_catboost(X_train, y_train, X_val, y_val, n_trials)
            if best_cat_params:
                best_cat_params['random_state'] = 42
                best_cat_params['verbose'] = 0
                cat_optimized = CatBoostClassifier(**best_cat_params)
                results, fitted_model = evaluate_model(
                    cat_optimized, X_train, y_train, X_val, y_val, X_test, y_test, 'CatBoost (Optimized)'
                )
                all_results.append(results)
                trained_models['CatBoost (Optimized)'] = fitted_model
    
    return pd.DataFrame(all_results), trained_models


# =============================================================================
# SHAP Analysis
# =============================================================================

def shap_analysis(model, X_test, feature_names, model_name, save_dir=RESULTS_DIR):
    """
    Generate SHAP analysis for model interpretability.
    """
    if not SHAP_AVAILABLE:
        print("SHAP not available. Skipping interpretability analysis.")
        return
    
    print(f"\nGenerating SHAP analysis for {model_name}...")
    
    # Create explainer
    if 'XGBoost' in model_name or 'LightGBM' in model_name or 'CatBoost' in model_name:
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 100))
    
    shap_values = explainer.shap_values(X_test)
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, take class 1
    
    # Summary plot
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(save_dir / f'shap_summary_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type='bar', show=False)
    plt.tight_layout()
    plt.savefig(save_dir / f'shap_bar_{model_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved SHAP plots to {save_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("="*60)
    print("NHANES Periodontitis ML Project - Model Training")
    print("="*60)
    
    # This is a placeholder - you need to run the data processing first
    # to create the processed data file with the periodontitis labels
    
    print("""
    Before running this script, ensure you have:
    
    1. Downloaded NHANES data (01_download_nhanes_data.py)
    2. Processed the data (02_process_nhanes_data.py)
    3. Created periodontitis labels using CDC/AAP definitions
    
    The processed file should be at:
    data/processed/nhanes_combined.parquet
    
    With columns including:
    - periodontitis_binary (0/1)
    - The 15 predictor features
    - cycle (for temporal splitting)
    """)
    
    # Feature columns (Bashir's 15 predictors)
    FEATURE_COLS = [
        'age', 'sex_male', 'education_hs_plus',
        'ever_smoker', 'ever_drinker',
        'bmi', 'waist_circumference',
        'systolic_bp', 'diastolic_bp',
        'fasting_glucose', 'triglycerides', 'hdl',
        'dental_visit_last_year', 'has_loose_teeth', 'uses_floss'
    ]
    
    TARGET_COL = 'periodontitis_binary'
    
    # Check if data exists
    data_path = PROCESSED_DIR / "nhanes_combined.parquet"
    if not data_path.exists():
        print(f"\n✗ Data file not found: {data_path}")
        print("  Run 01_download_nhanes_data.py and 02_process_nhanes_data.py first.")
        return
    
    # Load and split data
    df_train, df_val, df_test = load_and_split_data()
    
    # Prepare features
    # Note: You'll need to run create_bashir_predictors first
    # X_train, y_train = prepare_features(df_train, FEATURE_COLS, TARGET_COL)
    # X_val, y_val = prepare_features(df_val, FEATURE_COLS, TARGET_COL)
    # X_test, y_test = prepare_features(df_test, FEATURE_COLS, TARGET_COL)
    
    # Impute missing values
    # imputer = SimpleImputer(strategy='median')
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(imputer.fit_transform(X_train))
    # X_val = scaler.transform(imputer.transform(X_val))
    # X_test = scaler.transform(imputer.transform(X_test))
    
    # Run all models
    # results_df, trained_models = run_all_models(
    #     X_train, y_train, X_val, y_val, X_test, y_test,
    #     optimize_hyperparams=True, n_trials=30
    # )
    
    # Save results
    # results_df.to_csv(RESULTS_DIR / 'model_comparison.csv', index=False)
    
    # SHAP analysis for best model
    # best_model_name = results_df.loc[results_df['test_auc'].idxmax(), 'model']
    # shap_analysis(trained_models[best_model_name], X_test, FEATURE_COLS, best_model_name)
    
    print("\n" + "="*60)
    print("Done! Check results/ directory for outputs.")
    print("="*60)


if __name__ == "__main__":
    main()
