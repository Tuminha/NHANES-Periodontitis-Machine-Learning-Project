#!/usr/bin/env python3
"""Generate manuscript figures from canonical publication result artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"

BLUE = "#15365a"
LIGHT_BLUE = "#2f80b7"
RED = "#8f2d2d"
GOLD = "#c28b12"
GREEN = "#2f7d42"
GRAY = "#6b7280"
LIGHT_GRAY = "#e5e7eb"
PDF_METADATA = {
    "CreationDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
    "ModDate": datetime(2026, 6, 4, tzinfo=timezone.utc),
}


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": LIGHT_GRAY,
            "grid.linewidth": 0.7,
        }
    )


def label_bars(ax: plt.Axes, bars, fmt: str = "{:.3f}", offset: float = 0.01) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def label_horizontal_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}%") -> None:
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + 0.01,
            bar.get_y() + bar.get_height() / 2,
            fmt.format(width * 100),
            ha="left",
            va="center",
            fontsize=8,
        )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", metadata=PDF_METADATA)
    plt.close(fig)


def model_rows() -> list[dict]:
    primary = load_json("results/v13_primary_norc_summary.json")
    secondary = load_json("results/v13_secondary_full_summary.json")
    temporal = load_json("results/external_0910_metrics.json")
    return [
        {
            "label": "Primary internal\n29 features",
            "auc": primary["metrics"]["auc_roc"],
            "pr_auc": primary["metrics"]["pr_auc"],
            "brier": primary["metrics"]["brier_score"],
            "features": primary["n_features"],
        },
        {
            "label": "Secondary internal\n33 features",
            "auc": secondary["metrics"]["auc_roc"],
            "pr_auc": secondary["metrics"]["pr_auc"],
            "brier": secondary["metrics"]["brier_score"],
            "features": secondary["n_features"],
        },
        {
            "label": "Same-source temporal\nfrozen primary",
            "auc": temporal["metrics"]["auc"]["mean"],
            "pr_auc": temporal["metrics"]["prauc"]["mean"],
            "brier": temporal["metrics"]["brier"]["mean"],
            "features": primary["n_features"],
        },
    ]


def plot_performance_summary() -> None:
    rows = model_rows()
    temporal = load_json("results/external_0910_metrics.json")
    operating = temporal["operating_points"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Publication model performance summary", fontsize=15, fontweight="bold", y=0.99)

    labels = [row["label"] for row in rows]
    x = np.arange(len(labels))
    width = 0.36

    ax = axes[0, 0]
    auc_bars = ax.bar(x - width / 2, [row["auc"] for row in rows], width, label="AUC-ROC", color=BLUE)
    pr_bars = ax.bar(x + width / 2, [row["pr_auc"] for row in rows], width, label="PR-AUC", color=LIGHT_BLUE)
    ax.set_title("A. Discrimination")
    ax.set_ylabel("Metric value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    label_bars(ax, auc_bars, offset=0.015)
    label_bars(ax, pr_bars, offset=0.015)

    ax = axes[0, 1]
    brier_bars = ax.bar(x, [row["brier"] for row in rows], color=[BLUE, LIGHT_BLUE, RED])
    ax.set_title("B. Calibration error")
    ax.set_ylabel("Brier score, lower is better")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 0.25)
    label_bars(ax, brier_bars, offset=0.004)

    ax = axes[1, 0]
    metrics = ["sensitivity", "specificity", "ppv", "npv"]
    metric_labels = ["Sensitivity", "Specificity", "PPV", "NPV"]
    op_labels = ["t = 0.35", "t = 0.65"]
    values = [
        [operating["rule_out_t_0.35"][metric] for metric in metrics],
        [operating["balanced_t_0.65"][metric] for metric in metrics],
    ]
    metric_x = np.arange(len(metrics))
    op_width = 0.36
    bars_035 = ax.bar(metric_x - op_width / 2, values[0], op_width, label=op_labels[0], color=GREEN)
    bars_065 = ax.bar(metric_x + op_width / 2, values[1], op_width, label=op_labels[1], color=GOLD)
    ax.set_title("C. Same-source temporal operating points")
    ax.set_ylabel("Proportion")
    ax.set_xticks(metric_x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.08)
    ax.legend(loc="upper right")
    label_bars(ax, bars_035, "{:.2f}", 0.015)
    label_bars(ax, bars_065, "{:.2f}", 0.015)

    ax = axes[1, 1]
    feature_counts = [rows[0]["features"], rows[1]["features"]]
    auc_gain = rows[1]["auc"] - rows[0]["auc"]
    bars = ax.bar(["Primary", "Secondary"], feature_counts, color=[BLUE, LIGHT_BLUE])
    ax.set_title("D. Feature-set comparison")
    ax.set_ylabel("Number of predictors")
    ax.set_ylim(0, 40)
    for bar, count in zip(bars, feature_counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 1,
            str(count),
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.text(
        0.5,
        0.18,
        f"Adding treatment-seeking variables changed AUC by {auc_gain:.4f}.",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f8fafc", "edgecolor": LIGHT_GRAY},
    )

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_figure(fig, "19_publication_performance_summary")


def subgroup_label(row: pd.Series) -> str:
    variable = row["subgroup_variable"]
    subgroup = row["subgroup"]
    if variable == "sex":
        subgroup = "Male" if float(subgroup) == 1.0 else "Female"
        return f"Sex: {subgroup}"
    if variable == "age_group":
        return f"Age: {subgroup}"
    if variable == "smoking":
        subgroup = str(subgroup).replace("never/unknown", "never or unknown")
        return f"Smoking: {subgroup}"
    elif variable == "metabolic_risk":
        subgroup = str(subgroup).replace("_", " ")
        return f"Metabolic risk: {subgroup}"
    return f"{variable.replace('_', ' ').title()}: {subgroup}"


def plot_sensitivity_summary() -> None:
    payload = load_json("results/publication_sensitivity_tables.json")
    temporal = load_json("results/external_0910_metrics.json")
    prevalence = pd.DataFrame(payload["prevalence_by_cycle"])
    subgroup = pd.DataFrame(payload["subgroup_performance"])
    missingness = pd.DataFrame(payload["missingness"]).sort_values("missing_pct", ascending=False).head(7)

    selected_variables = {"age_group", "sex", "smoking", "metabolic_risk"}
    subgroup = subgroup[subgroup["subgroup_variable"].isin(selected_variables)].copy()
    subgroup["label"] = subgroup.apply(subgroup_label, axis=1)
    variable_order = {"age_group": 0, "sex": 1, "smoking": 2, "metabolic_risk": 3}
    subgroup["variable_order"] = subgroup["subgroup_variable"].map(variable_order)
    subgroup = subgroup.sort_values(["variable_order", "subgroup"])

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Survey and subgroup sensitivity summary", fontsize=15, fontweight="bold", y=1.03)

    ax = axes[0]
    x = np.arange(len(prevalence))
    width = 0.36
    unweighted = ax.bar(
        x - width / 2,
        prevalence["unweighted_prevalence"],
        width,
        color=BLUE,
        label="Unweighted",
    )
    weighted = ax.bar(
        x + width / 2,
        prevalence["weighted_prevalence"],
        width,
        color=LIGHT_BLUE,
        label="Survey-weighted",
    )
    ax.set_title("A. Periodontitis prevalence")
    ax.set_ylabel("Prevalence")
    ax.set_xticks(x)
    ax.set_xticklabels(prevalence["cycle"], rotation=20, ha="right")
    ax.set_ylim(0, 0.82)
    ax.legend(loc="upper right")
    label_bars(ax, unweighted, "{:.2f}", 0.015)
    label_bars(ax, weighted, "{:.2f}", 0.015)

    ax = axes[1]
    y = np.arange(len(subgroup))
    bars = ax.barh(y, subgroup["auc"], color=BLUE)
    ax.axvline(temporal["metrics"]["auc"]["mean"], color=RED, linestyle="--", linewidth=1.2, label="Overall temporal AUC")
    ax.set_title("B. Temporal AUC by subgroup")
    ax.set_xlabel("AUC-ROC")
    ax.set_yticks(y)
    ax.set_yticklabels(subgroup["label"])
    ax.set_xlim(0.5, 0.75)
    ax.invert_yaxis()
    ax.legend(loc="lower right")
    for bar in bars:
        ax.text(bar.get_width() + 0.004, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}", va="center", fontsize=8)

    ax = axes[2]
    missingness = missingness.iloc[::-1]
    bars = ax.barh(np.arange(len(missingness)), missingness["missing_pct"], color=GRAY)
    ax.set_title("C. Highest feature missingness")
    ax.set_xlabel("Missing proportion")
    ax.set_yticks(np.arange(len(missingness)))
    ax.set_yticklabels(missingness["feature"])
    ax.set_xlim(0, max(0.65, float(missingness["missing_pct"].max()) + 0.08))
    label_horizontal_bars(ax, bars)

    fig.tight_layout()
    save_figure(fig, "20_publication_sensitivity_summary")


def main() -> None:
    set_style()
    plot_performance_summary()
    plot_sensitivity_summary()
    print("Generated publication figures:")
    for name in [
        "19_publication_performance_summary.png",
        "19_publication_performance_summary.pdf",
        "20_publication_sensitivity_summary.png",
        "20_publication_sensitivity_summary.pdf",
    ]:
        print(f"  figures/{name}")


if __name__ == "__main__":
    main()
