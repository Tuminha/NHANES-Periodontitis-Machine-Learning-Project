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
    
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    labels = np.array([0, 1])
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    if len(np.unique(y_true)) == 2:
        roc_auc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        roc_auc = np.nan
        pr_auc = np.nan

    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


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
    
    """
    if thresholds is None:
        thresholds = np.linspace(0.1, 0.9, 17)

    rows = [compute_metrics(y_true, y_prob, threshold=float(thr)) for thr in thresholds]
    return pd.DataFrame(rows)


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
    
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    if len(np.unique(y_true)) < 2:
        raise ValueError("Threshold selection requires both outcome classes.")

    if policy == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        best_idx = int(np.argmax(tpr - fpr))
        return float(thresholds[best_idx])

    if policy == "f1_max":
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_prob)
        if len(thresholds) == 0:
            return 0.5
        precision_vals = precision_vals[:-1]
        recall_vals = recall_vals[:-1]
        f1_scores = 2 * precision_vals * recall_vals / (precision_vals + recall_vals + 1e-10)
        return float(thresholds[int(np.argmax(f1_scores))])

    if policy.startswith("recall_"):
        target_recall = float(policy.split("_", 1)[1])
        thresholds = np.unique(y_prob)
        rows = compute_metrics_at_multiple_thresholds(y_true, y_prob, thresholds)
        eligible = rows[rows["sensitivity"] >= target_recall]
        if eligible.empty:
            return float(rows.sort_values("sensitivity", ascending=False).iloc[0]["threshold"])
        best = eligible.sort_values(["specificity", "f1_score"], ascending=False).iloc[0]
        return float(best["threshold"])

    raise ValueError(f"Unknown threshold policy: {policy}")


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
    
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].set_xlabel("False positive rate")
    axes[0].set_ylabel("True positive rate")
    axes[0].set_title(f"{title_prefix} ROC".strip())
    axes[0].legend(loc="lower right")

    axes[1].plot(recall_vals, precision_vals, label=f"PR-AUC = {pr_auc:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} Precision-recall".strip())
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    
    """
    import seaborn as sns
    from pathlib import Path

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    values = cm
    fmt = "d"
    if normalize:
        values = cm.astype(float)
        denom = values.sum(axis=1, keepdims=True)
        values = np.divide(values, denom, out=np.zeros_like(values), where=denom != 0)
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(values, annot=True, fmt=fmt, cmap="Blues", cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_xticklabels(["No periodontitis", "Periodontitis"], rotation=20, ha="right")
    ax.set_yticklabels(["No periodontitis", "Periodontitis"], rotation=0)
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    
    """
    from pathlib import Path

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="uniform"
    )
    brier = brier_score_loss(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(prob_pred, prob_true, marker="o", label=f"Model (Brier={brier:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event fraction")
    ax.legend(loc="upper left")
    ax.set_title("Calibration")
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    
    """
    from pathlib import Path

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    thresholds = np.linspace(0.01, 0.99, 99)
    model_nb = []
    treat_all_nb = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        weight = thr / (1 - thr)
        model_nb.append((tp / n) - (fp / n) * weight)
        treat_all_nb.append(prevalence - (1 - prevalence) * weight)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(thresholds, model_nb, label="Model")
    ax.plot(thresholds, treat_all_nb, "--", label="Treat all")
    ax.axhline(0, color="black", linestyle=":", linewidth=1, label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.legend()
    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


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
    
    """
    try:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    except TypeError:
        calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(X_train, y_train)
    return calibrated


# =============================================================================
# Results Export
# =============================================================================

def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows available._"

    display = df.astype(object).where(pd.notna(df), "")
    columns = [str(col) for col in display.columns]
    rows = [[str(value) for value in row] for row in display.to_numpy()]
    widths = [
        max(len(columns[idx]), *(len(row[idx]) for row in rows))
        for idx in range(len(columns))
    ]
    header = "| " + " | ".join(col.ljust(widths[idx]) for idx, col in enumerate(columns)) + " |"
    divider = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


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
    
    """
    from pathlib import Path

    df = pd.DataFrame(metrics_dict).T.round(4)
    output = Path(save_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output)
    output.with_suffix(".md").write_text(_markdown_table(df), encoding="utf-8")
    return df
