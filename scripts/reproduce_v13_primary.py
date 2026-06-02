#!/usr/bin/env python3
"""Reproduce the v1.3 internal benchmark from processed NHANES data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.reproduction import (
    BALANCED_THRESHOLD,
    CORE_FEATURES,
    PRIMARY_FEATURES,
    RULE_OUT_THRESHOLD,
    SECONDARY_BALANCED_THRESHOLD,
    SECONDARY_FEATURES,
    SECONDARY_RULE_OUT_THRESHOLD,
    TREATMENT_SEEKING_FEATURES,
    assert_feature_contract,
    build_modeling_frame,
    cross_validated_predictions,
    feature_sets,
    fit_calibrated_ensemble,
    summarize_predictions,
    summary_artifact,
    timestamp,
    write_json,
)


def load_modeling_frame(input_path: Path) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Processed data not found: {input_path}. Run `make download && make process` first.")

    raw = pd.read_parquet(input_path)
    frame = build_modeling_frame(raw)
    frame = frame.dropna(subset=["has_periodontitis"])
    assert_feature_contract(frame)
    return frame.reset_index(drop=True)


def development_subset(frame: pd.DataFrame, max_rows: int | None = None) -> pd.DataFrame:
    frame = frame[frame["cycle"].isin(["2011-2012", "2013-2014"])].copy()
    if max_rows:
        per_class = max(1, max_rows // max(frame["has_periodontitis"].nunique(), 1))
        sampled = [
            group.sample(min(len(group), per_class), random_state=42)
            for _, group in frame.groupby("has_periodontitis")
        ]
        frame = pd.concat(sampled, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    return frame.reset_index(drop=True)


def write_internal_artifacts(frame: pd.DataFrame, out_dir: Path, folds: int, seed: int) -> dict:
    y = frame["has_periodontitis"].astype(int)
    prevalence = float(y.mean())

    predictions = {}
    summaries = {}
    for name, features, ro_threshold, balanced_threshold in [
        ("deployment_ready", CORE_FEATURES, RULE_OUT_THRESHOLD, BALANCED_THRESHOLD),
        ("primary", PRIMARY_FEATURES, RULE_OUT_THRESHOLD, BALANCED_THRESHOLD),
        ("secondary", SECONDARY_FEATURES, SECONDARY_RULE_OUT_THRESHOLD, SECONDARY_BALANCED_THRESHOLD),
    ]:
        print(f"Running {folds}-fold CV for {name} ({len(features)} features)...")
        prob = cross_validated_predictions(frame, features, n_folds=folds, seed=seed)
        predictions[name] = prob
        summaries[name] = summarize_predictions(
            y,
            prob,
            n_features=len(features),
            name=name,
            rule_out_threshold=ro_threshold,
            balanced_threshold=balanced_threshold,
        )

    primary = summaries["primary"]
    secondary = summaries["secondary"]
    deployment = summaries["deployment_ready"]

    featuredrop = {
        "v1.3_full": {
            key: secondary[key]
            for key in [
                "name",
                "n_features",
                "auc",
                "pr_auc",
                "rule_out_recall",
                "rule_out_specificity",
                "balanced_recall",
                "balanced_specificity",
            ]
        },
        "v1.3_no_reverse_causality": {
            key: primary[key]
            for key in [
                "name",
                "n_features",
                "auc",
                "pr_auc",
                "rule_out_recall",
                "rule_out_specificity",
                "balanced_recall",
                "balanced_specificity",
            ]
        },
        "dropped_features": TREATMENT_SEEKING_FEATURES,
        "deltas": {
            "auc": primary["auc"] - secondary["auc"],
            "pr_auc": primary["pr_auc"] - secondary["pr_auc"],
            "rule_out_recall": primary["rule_out_recall"] - secondary["rule_out_recall"],
            "balanced_specificity": primary["balanced_specificity"] - secondary["balanced_specificity"],
        },
        "timestamp": timestamp(),
    }
    write_json(out_dir / "v13_featuredrop.json", featuredrop)

    write_json(
        out_dir / "v13_primary_norc_summary.json",
        summary_artifact(
            primary,
            model_name="v1.3_primary_no_reverse_causality",
            description="Primary benchmark model excluding treatment-seeking variables",
            dataset="NHANES 2011-2014",
            prevalence=prevalence,
            threshold_rule_out=RULE_OUT_THRESHOLD,
            threshold_balanced=BALANCED_THRESHOLD,
            extra={
                "dropped_features": TREATMENT_SEEKING_FEATURES,
                "rationale": "Removes treatment-seeking variables with limited discrimination cost.",
            },
        ),
    )
    write_json(
        out_dir / "v13_secondary_full_summary.json",
        summary_artifact(
            secondary,
            model_name="v1.3_secondary_full_features",
            description="Secondary upper-bound model including treatment-seeking variables",
            dataset="NHANES 2011-2014",
            prevalence=prevalence,
            threshold_rule_out=SECONDARY_RULE_OUT_THRESHOLD,
            threshold_balanced=SECONDARY_BALANCED_THRESHOLD,
            extra={
                "additional_features": TREATMENT_SEEKING_FEATURES,
                "rationale": "Upper-bound sensitivity analysis for treatment-seeking variables.",
            },
        ),
    )

    nan_ablation = [
        slim_summary("v1.3_full", secondary),
        slim_summary("v1.3_no_reverse_causality", primary),
        slim_summary("deployment_ready_core", deployment),
    ]
    complete = frame.dropna(subset=CORE_FEATURES)
    if len(complete) and complete["has_periodontitis"].nunique() == 2:
        complete_prob = cross_validated_predictions(complete.reset_index(drop=True), CORE_FEATURES, n_folds=folds, seed=seed)
        complete_summary = summarize_predictions(
            complete["has_periodontitis"].astype(int),
            complete_prob,
            len(CORE_FEATURES),
            "complete_case_core",
        )
        nan_ablation.append(slim_summary("complete_case_core", complete_summary, n_samples=len(complete)))
    write_json(out_dir / "v13_nan_ablation.json", nan_ablation)

    write_json(
        out_dir / "v13_operating_points.json",
        {
            "version": "v1.3",
            "dataset": "NHANES 2011-2014",
            "n_samples": int(len(frame)),
            "prevalence": prevalence,
            "calibration": "nested_isotonic_regression",
            "feature_sets": {key: len(value) for key, value in feature_sets().items()},
            "models": {
                "primary": model_operating_payload(primary, RULE_OUT_THRESHOLD, BALANCED_THRESHOLD),
                "secondary": model_operating_payload(secondary, SECONDARY_RULE_OUT_THRESHOLD, SECONDARY_BALANCED_THRESHOLD),
            },
            "feature_drop_analysis": {
                "dropped_features": TREATMENT_SEEKING_FEATURES,
                "auc_delta": primary["auc"] - secondary["auc"],
                "pr_auc_delta": primary["pr_auc"] - secondary["pr_auc"],
                "interpretation": "Treatment-seeking features are reported as an upper-bound sensitivity analysis.",
            },
            "timestamp": timestamp(),
        },
    )

    importance = feature_importance_summary(frame, SECONDARY_FEATURES, seed)
    write_json(out_dir / "v13_shap_summary.json", importance)
    return summaries


def slim_summary(name: str, summary: dict, n_samples: int | None = None) -> dict:
    return {
        "name": name,
        "n_samples": int(n_samples) if n_samples is not None else None,
        "n_features": int(summary["n_features"]),
        "auc": summary["auc"],
        "pr_auc": summary["pr_auc"],
        "brier_score": summary["brier_score"],
        "rule_out_recall": summary["rule_out_recall"],
        "rule_out_specificity": summary["rule_out_specificity"],
        "balanced_recall": summary["balanced_recall"],
        "balanced_specificity": summary["balanced_specificity"],
    }


def model_operating_payload(summary: dict, rule_threshold: float, balanced_threshold: float) -> dict:
    return {
        "name": summary["name"],
        "n_features": summary["n_features"],
        "metrics": {
            "auc_roc": summary["auc"],
            "pr_auc": summary["pr_auc"],
            "brier_score": summary["brier_score"],
        },
        "operating_points": {
            "rule_out": {
                "threshold": rule_threshold,
                "recall": summary["rule_out"]["sensitivity"],
                "specificity": summary["rule_out"]["specificity"],
                "precision": summary["rule_out"]["precision"],
                "f1": summary["rule_out"]["f1_score"],
            },
            "balanced": {
                "threshold": balanced_threshold,
                "recall": summary["balanced"]["sensitivity"],
                "specificity": summary["balanced"]["specificity"],
                "precision": summary["balanced"]["precision"],
                "f1": summary["balanced"]["f1_score"],
            },
        },
    }


def feature_importance_summary(frame: pd.DataFrame, features: list[str], seed: int) -> dict:
    fitted = fit_calibrated_ensemble(frame, frame["has_periodontitis"].astype(int), features, seed=seed)
    importances = []
    for model in fitted.models:
        if hasattr(model, "feature_importances_"):
            importances.append(model.feature_importances_)
        elif hasattr(model, "get_feature_importance"):
            importances.append(model.get_feature_importance())
    if importances:
        mean_importance = pd.Series(sum(importances) / len(importances), index=features)
        top = [
            {"feature": feature, "mean_model_importance": float(value)}
            for feature, value in mean_importance.sort_values(ascending=False).head(10).items()
        ]
    else:
        top = []
    return {
        "model": "v1.3_secondary_full_features",
        "importance_method": "mean_tree_feature_importance",
        "n_samples": int(len(frame)),
        "n_features": int(len(features)),
        "top_10_features": top,
        "treatment_seeking_features": TREATMENT_SEEKING_FEATURES,
        "timestamp": timestamp(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/nhanes_combined.parquet")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--modeling-frame", default="data/processed/nhanes_modeling_frame.parquet")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional smoke-test row cap.")
    args = parser.parse_args()

    full_frame = load_modeling_frame(Path(args.input))
    Path(args.modeling_frame).parent.mkdir(parents=True, exist_ok=True)
    full_frame.to_parquet(args.modeling_frame)
    frame = development_subset(full_frame, max_rows=args.max_rows)
    summaries = write_internal_artifacts(frame, Path(args.out_dir), args.folds, args.seed)
    print("Internal reproduction complete:")
    for name, summary in summaries.items():
        print(f"  {name}: AUC={summary['auc']:.4f}, PR-AUC={summary['pr_auc']:.4f}, Brier={summary['brier_score']:.4f}")


if __name__ == "__main__":
    main()
