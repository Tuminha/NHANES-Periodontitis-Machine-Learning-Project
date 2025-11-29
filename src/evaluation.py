"""
Model Evaluation Utilities
Author: Francisco Teixeira Barbosa (Cisco)

Purpose: Compute classification metrics, ROC/PR curves, calibration plots,
         decision curves, and threshold selection strategies.

Usage:
    from src.evaluation import compute_metrics, plot_roc_pr, select_threshold
    
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    select_threshold(y_val, p_val, policy="recall_0.80")
    plot_roc_pr(y_test, p_test, save_path="figures/roc_pr_test.png")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from typing import Dict, Tuple, Optional


# =============================================================================
# Core Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: True binary labels (0/1)
        y_prob: Predicted probabilities (0-1)
        threshold: Decision threshold for binary classification
    
    Returns:
        Dict with keys:
            - roc_auc: Area under ROC curve
            - pr_auc: Area under precision-recall curve
            - brier_score: Brier score (calibration)
            - accuracy: Accuracy at given threshold
            - sensitivity: Recall / True Positive Rate
            - specificity: True Negative Rate
            - precision: Positive Predictive Value
            - f1_score: Harmonic mean of precision and recall
            - tn, fp, fn, tp: Confusion matrix counts
    
    TODO: Compute each metric using sklearn functions
    TODO: Compute specificity from confusion matrix
    TODO: Handle edge cases (e.g., all one class)
    TODO: Return as dict
    """
    # TODO: y_pred = (y_prob >= threshold).astype(int)
    # TODO: roc_auc = roc_auc_score(y_true, y_prob)
    # TODO: pr_auc = average_precision_score(y_true, y_prob)
    # TODO: brier = brier_score_loss(y_true, y_prob)
    # TODO: accuracy = accuracy_score(y_true, y_pred)
    # TODO: sensitivity = recall_score(y_true, y_pred)
    # TODO: precision_val = precision_score(y_true, y_pred, zero_division=0)
    # TODO: f1 = f1_score(y_true, y_pred, zero_division=0)
    # TODO: tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    # TODO: specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    # TODO: return dict with all metrics
    pass


def compute_metrics_at_multiple_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """
    Compute metrics across a range of thresholds.
    Useful for threshold sensitivity analysis.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        thresholds: Array of thresholds to test (default: np.linspace(0.1, 0.9, 17))
    
    Returns:
        DataFrame with columns: threshold, accuracy, sensitivity, specificity, precision, f1
    
    TODO: Loop over thresholds
    TODO: Call compute_metrics for each threshold
    TODO: Collect results into DataFrame
    """
    # TODO: if thresholds is None: thresholds = np.linspace(0.1, 0.9, 17)
    # TODO: results = []
    # TODO: for thr in thresholds:
    #           metrics = compute_metrics(y_true, y_prob, threshold=thr)
    #           results.append({...})
    # TODO: return pd.DataFrame(results)
    pass


# =============================================================================
# Threshold Selection
# =============================================================================

def select_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    policy: str = "youden"
) -> float:
    """
    Select optimal decision threshold using a specified policy.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        policy: Selection strategy:
            - "youden": Maximize Youden's J statistic (sensitivity + specificity - 1)
            - "f1_max": Maximize F1 score
            - "recall_0.80": Threshold that gives sensitivity >= 0.80 with max specificity
            - "recall_0.90": Threshold that gives sensitivity >= 0.90
    
    Returns:
        threshold: Selected threshold value
    
    Freeze this threshold on VALIDATION set, then apply to TEST set.
    
    TODO: Implement each policy
    TODO: For Youden: use roc_curve, find max(tpr - fpr)
    TODO: For F1: loop over thresholds, find max F1
    TODO: For recall constraint: find threshold where recall >= target
    """
    # TODO: if policy == "youden":
    #           fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    #           j_statistic = tpr - fpr
    #           best_idx = np.argmax(j_statistic)
    #           return thresholds[best_idx]
    
    # TODO: elif policy == "f1_max":
    #           precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)
    #           f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    #           best_idx = np.argmax(f1_scores)
    #           return thresholds[best_idx]
    
    # TODO: elif policy.startswith("recall_"):
    #           target_recall = float(policy.split("_")[1])
    #           # Find threshold where recall >= target_recall and maximize specificity
    
    # TODO: else: raise ValueError(f"Unknown policy: {policy}")
    pass


# =============================================================================
# Visualization
# =============================================================================

def plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None,
    title_prefix: str = ""
) -> None:
    """
    Plot ROC and Precision-Recall curves side by side.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        save_path: Path to save figure (e.g., "figures/roc_pr_test.png")
        title_prefix: Optional prefix for titles (e.g., "Test Set")
    
    TODO: Compute ROC and PR curves
    TODO: Compute AUCs
    TODO: Create 1x2 subplot
    TODO: Plot both curves with AUC in legend
    TODO: Add diagonal reference line to ROC plot
    TODO: Apply Periospot styling (call ps_plot helpers)
    TODO: Save figure if save_path provided
    """
    # TODO: from src.ps_plot import set_style, get_palette, save_figure
    # TODO: set_style()
    # TODO: palette = get_palette()
    
    # TODO: fpr, tpr, _ = roc_curve(y_true, y_prob)
    # TODO: roc_auc = roc_auc_score(y_true, y_prob)
    # TODO: precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    # TODO: pr_auc = average_precision_score(y_true, y_prob)
    
    # TODO: fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # TODO: axes[0].plot ROC curve
    # TODO: axes[1].plot PR curve
    # TODO: if save_path: save_figure(fig, save_path)
    pass


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
    normalize: bool = False
) -> None:
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels (after thresholding)
        save_path: Path to save figure
        normalize: If True, show proportions instead of counts
    
    TODO: Compute confusion matrix
    TODO: Create heatmap with seaborn or matplotlib
    TODO: Apply Periospot colors
    TODO: Save if save_path provided
    """
    # TODO: cm = confusion_matrix(y_true, y_pred)
    # TODO: if normalize: cm = cm / cm.sum(axis=1, keepdims=True)
    # TODO: import seaborn as sns
    # TODO: sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap="Blues")
    # TODO: plt.xlabel("Predicted"), plt.ylabel("True")
    # TODO: if save_path: save_figure(plt.gcf(), save_path)
    pass


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot reliability diagram (calibration curve).
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve
        save_path: Path to save figure
    
    TODO: Use sklearn.calibration.calibration_curve
    TODO: Plot predicted vs observed probabilities
    TODO: Add perfect calibration diagonal line
    TODO: Compute Brier score and show on plot
    TODO: Apply Periospot styling
    """
    # TODO: prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy='uniform')
    # TODO: brier = brier_score_loss(y_true, y_prob)
    # TODO: fig, ax = plt.subplots(figsize=(7, 7))
    # TODO: ax.plot(prob_pred, prob_true, marker='o', label=f"Model (Brier={brier:.3f})")
    # TODO: ax.plot([0, 1], [0, 1], 'k--', label="Perfect calibration")
    # TODO: ax.set_xlabel("Mean predicted probability")
    # TODO: ax.set_ylabel("Fraction of positives")
    # TODO: if save_path: save_figure(fig, save_path)
    pass


def plot_decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: Optional[str] = None
) -> None:
    """
    Plot decision curve analysis.
    
    Shows net benefit across range of threshold probabilities.
    Useful for clinical decision-making.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        save_path: Path to save figure
    
    TODO: Compute net benefit for model
    TODO: Compute net benefit for treat-all and treat-none strategies
    TODO: Plot across threshold range
    TODO: Reference: Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making. 2006
    """
    # TODO: thresholds = np.linspace(0.01, 0.99, 100)
    # TODO: net_benefits = []
    # TODO: for thr in thresholds:
    #           # Net benefit = (TP/N) - (FP/N) * (thr / (1 - thr))
    # TODO: Plot model, treat-all, treat-none strategies
    # TODO: if save_path: save_figure(fig, save_path)
    pass


# =============================================================================
# Model Calibration
# =============================================================================

def calibrate_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "isotonic"
):
    """
    Calibrate model probabilities using isotonic regression or Platt scaling.
    
    Args:
        model: Trained sklearn-compatible classifier
        X_train: Training features (typically use validation set for calibration)
        y_train: Training labels
        method: "isotonic" or "sigmoid" (Platt scaling)
    
    Returns:
        calibrated_model: CalibratedClassifierCV wrapper
    
    TODO: Use sklearn.calibration.CalibratedClassifierCV
    TODO: Fit on provided data
    TODO: Return calibrated model
    """
    # TODO: calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    # TODO: calibrated.fit(X_train, y_train)
    # TODO: return calibrated
    pass


# =============================================================================
# Results Export
# =============================================================================

def export_metrics_table(
    metrics_dict: Dict[str, Dict[str, float]],
    save_path: str = "results/metrics_table.csv"
) -> pd.DataFrame:
    """
    Export metrics from multiple models to a formatted table.
    
    Args:
        metrics_dict: Dict of {model_name: {metric_name: value}}
        save_path: Path to save CSV
    
    Returns:
        DataFrame with models as rows, metrics as columns
    
    TODO: Convert nested dict to DataFrame
    TODO: Round values to 4 decimal places
    TODO: Save to CSV
    TODO: Also save as formatted Markdown table
    """
    # TODO: df = pd.DataFrame(metrics_dict).T
    # TODO: df = df.round(4)
    # TODO: df.to_csv(save_path)
    # TODO: print(f"Metrics table saved to {save_path}")
    # TODO: return df
    pass

