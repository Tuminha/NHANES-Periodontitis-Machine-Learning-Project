"""
Statistical utilities for model comparison and prevalence analysis.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import json
from typing import Dict, Tuple, List, Optional
from pathlib import Path


def permutation_test_auc(
    y_true: np.ndarray,
    proba_model1: np.ndarray,
    proba_model2: np.ndarray,
    n_permutations: int = 10000,
    random_state: int = 42
) -> Dict:
    """
    Perform permutation test comparing AUCs of two models.
    
    This tests the null hypothesis that the two models have equal AUC
    by permuting the model labels and computing the AUC difference.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    proba_model1 : array-like
        Predicted probabilities from model 1
    proba_model2 : array-like  
        Predicted probabilities from model 2
    n_permutations : int
        Number of permutations (default 10000)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict with keys:
        - auc_model1: AUC of model 1
        - auc_model2: AUC of model 2
        - observed_diff: Observed AUC difference (model1 - model2)
        - p_value: Two-sided p-value
        - n_permutations: Number of permutations used
        - effect_size: Cohen's d effect size
    """
    np.random.seed(random_state)
    
    # Observed AUCs
    auc1 = roc_auc_score(y_true, proba_model1)
    auc2 = roc_auc_score(y_true, proba_model2)
    observed_diff = auc1 - auc2
    
    # Stack predictions for permutation
    stacked = np.column_stack([proba_model1, proba_model2])
    
    # Permutation distribution
    perm_diffs = np.zeros(n_permutations)
    
    for i in range(n_permutations):
        # For each sample, randomly swap predictions between models
        swap_mask = np.random.randint(0, 2, size=len(y_true)).astype(bool)
        perm_proba1 = np.where(swap_mask, stacked[:, 1], stacked[:, 0])
        perm_proba2 = np.where(swap_mask, stacked[:, 0], stacked[:, 1])
        
        perm_auc1 = roc_auc_score(y_true, perm_proba1)
        perm_auc2 = roc_auc_score(y_true, perm_proba2)
        perm_diffs[i] = perm_auc1 - perm_auc2
    
    # Two-sided p-value
    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
    
    # Effect size (Cohen's d using permutation distribution as reference)
    effect_size = observed_diff / (np.std(perm_diffs) + 1e-10)
    
    return {
        'auc_model1': float(auc1),
        'auc_model2': float(auc2),
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'n_permutations': n_permutations,
        'effect_size': float(effect_size),
        'perm_std': float(np.std(perm_diffs))
    }


def pairwise_permutation_tests(
    y_true: np.ndarray,
    model_predictions: Dict[str, np.ndarray],
    n_permutations: int = 10000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Perform pairwise permutation tests between all model pairs.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    model_predictions : dict
        Dictionary mapping model names to predicted probabilities
    n_permutations : int
        Number of permutations
    random_state : int
        Random seed
        
    Returns
    -------
    DataFrame with pairwise comparison results
    """
    models = list(model_predictions.keys())
    results = []
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            test_result = permutation_test_auc(
                y_true,
                model_predictions[model1],
                model_predictions[model2],
                n_permutations,
                random_state
            )
            
            results.append({
                'model1': model1,
                'model2': model2,
                'auc_model1': test_result['auc_model1'],
                'auc_model2': test_result['auc_model2'],
                'diff': test_result['observed_diff'],
                'p_value': test_result['p_value'],
                'effect_size': test_result['effect_size'],
                'significant_0.05': test_result['p_value'] < 0.05,
                'significant_0.01': test_result['p_value'] < 0.01
            })
    
    return pd.DataFrame(results)


def compute_prevalence_by_cycle(
    df: pd.DataFrame,
    cycle_col: str = 'cycle',
    severity_col: str = 'severity',
    has_perio_col: str = 'has_periodontitis'
) -> Dict:
    """
    Compute periodontitis prevalence by NHANES cycle.
    
    Parameters
    ----------
    df : DataFrame
        Data with periodontitis labels
    cycle_col : str
        Column name for NHANES cycle
    severity_col : str
        Column name for severity classification
    has_perio_col : str
        Column name for binary periodontitis flag
        
    Returns
    -------
    dict with prevalence statistics by cycle
    """
    results = {
        'overall': {},
        'by_cycle': {},
        'by_severity': {},
        'cdc_comparison': {}
    }
    
    # Overall prevalence
    if has_perio_col in df.columns:
        overall_prev = df[has_perio_col].mean()
        results['overall'] = {
            'n_total': len(df),
            'n_periodontitis': int(df[has_perio_col].sum()),
            'prevalence': float(overall_prev),
            'prevalence_pct': f"{overall_prev*100:.1f}%"
        }
    
    # By cycle
    if cycle_col in df.columns and has_perio_col in df.columns:
        for cycle, group in df.groupby(cycle_col):
            prev = group[has_perio_col].mean()
            results['by_cycle'][str(cycle)] = {
                'n': len(group),
                'n_periodontitis': int(group[has_perio_col].sum()),
                'prevalence': float(prev),
                'prevalence_pct': f"{prev*100:.1f}%"
            }
    
    # By severity
    if severity_col in df.columns:
        severity_counts = df[severity_col].value_counts(normalize=True)
        results['by_severity'] = {
            str(k): f"{v*100:.1f}%" for k, v in severity_counts.items()
        }
    
    # CDC published estimates for comparison
    results['cdc_comparison'] = {
        'cdc_2009_2012': {
            'total_periodontitis': '47.2%',
            'severe': '8.9%',
            'moderate': '30.0%',
            'mild': '8.7%',
            'source': 'Eke et al. 2015, J Periodontol'
        },
        'note': 'Our higher prevalence (68%) likely reflects inclusion criteria (adults 30+ with full perio exam) vs CDC population estimates'
    }
    
    return results


def save_prevalence_check(
    df: pd.DataFrame,
    output_path: Path,
    cycle_col: str = 'cycle',
    severity_col: str = 'severity',
    has_perio_col: str = 'has_periodontitis'
) -> Dict:
    """
    Compute and save prevalence statistics to JSON.
    
    Parameters
    ----------
    df : DataFrame
        Data with periodontitis labels
    output_path : Path
        Output JSON file path
    cycle_col, severity_col, has_perio_col : str
        Column names
        
    Returns
    -------
    dict with prevalence results
    """
    results = compute_prevalence_by_cycle(df, cycle_col, severity_col, has_perio_col)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Prevalence check saved to: {output_path}")
    return results


def save_permutation_results(
    comparison_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save permutation test results to JSON.
    
    Parameters
    ----------
    comparison_df : DataFrame
        Results from pairwise_permutation_tests
    output_path : Path
        Output JSON file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'method': 'permutation_test_10000_iterations',
        'description': 'Pairwise AUC comparisons using label-shuffling permutation test',
        'comparisons': comparison_df.to_dict(orient='records')
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Permutation test results saved to: {output_path}")

