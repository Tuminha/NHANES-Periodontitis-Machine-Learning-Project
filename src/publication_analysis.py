"""Publication sensitivity analyses for NHANES prediction reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


@dataclass(frozen=True)
class BinaryOperatingPoint:
    threshold: float
    sensitivity: float
    specificity: float
    ppv: float
    npv: float
    n: int


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & (weights > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(values[mask].astype(float), weights=weights[mask].astype(float)))


def prevalence_table(
    df: pd.DataFrame,
    outcome_col: str,
    weight_col: str | None = None,
    by: Iterable[str] = ("cycle",),
) -> pd.DataFrame:
    rows = []
    group_cols = [col for col in by if col in df.columns]
    grouped = df.groupby(group_cols, dropna=False) if group_cols else [(("overall",), df)]

    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: value for col, value in zip(group_cols, key)}
        row["n"] = int(len(group))
        row["unweighted_prevalence"] = float(group[outcome_col].mean())
        if weight_col and weight_col in group.columns:
            row["weighted_prevalence"] = weighted_mean(group[outcome_col], group[weight_col])
        rows.append(row)

    return pd.DataFrame(rows)


def operating_point(y_true: pd.Series, y_prob: pd.Series, threshold: float) -> BinaryOperatingPoint:
    y = y_true.astype(int).to_numpy()
    p = y_prob.astype(float).to_numpy()
    pred = (p >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return BinaryOperatingPoint(threshold, sensitivity, specificity, ppv, npv, len(y))


def subgroup_performance_table(
    df: pd.DataFrame,
    outcome_col: str,
    probability_col: str,
    subgroup_cols: Iterable[str],
    thresholds: Iterable[float] = (0.35, 0.65),
) -> pd.DataFrame:
    rows = []
    for subgroup_col in subgroup_cols:
        if subgroup_col not in df.columns:
            continue
        for subgroup, group in df.groupby(subgroup_col, dropna=False):
            group = group[[outcome_col, probability_col]].join(group[[subgroup_col]]).dropna(
                subset=[outcome_col, probability_col]
            )
            if group.empty:
                continue
            y = group[outcome_col].astype(int)
            p = group[probability_col].astype(float)
            row = {
                "subgroup_variable": subgroup_col,
                "subgroup": subgroup,
                "n": int(len(group)),
                "prevalence": float(y.mean()),
                "brier": float(brier_score_loss(y, p)),
            }
            if y.nunique() == 2:
                row["auc"] = float(roc_auc_score(y, p))
                row["pr_auc"] = float(average_precision_score(y, p))
            else:
                row["auc"] = np.nan
                row["pr_auc"] = np.nan

            for threshold in thresholds:
                op = operating_point(y, p, threshold)
                suffix = str(threshold).replace(".", "_")
                row[f"sensitivity_t_{suffix}"] = op.sensitivity
                row[f"specificity_t_{suffix}"] = op.specificity
                row[f"ppv_t_{suffix}"] = op.ppv
                row[f"npv_t_{suffix}"] = op.npv
            rows.append(row)

    return pd.DataFrame(rows)


def missingness_table(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in feature_cols:
        if col not in df.columns:
            continue
        rows.append(
            {
                "feature": col,
                "n": int(len(df)),
                "missing_n": int(df[col].isna().sum()),
                "missing_pct": float(df[col].isna().mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)
