"""Script-backed reproduction utilities for the NHANES benchmark."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.evaluation import compute_metrics


SEED = 42
DEVELOPMENT_CYCLES = ("2011-2012", "2013-2014")
TEMPORAL_CYCLE = "2009-2010"

CORE_FEATURES = [
    "age",
    "sex",
    "education",
    "bmi",
    "waist_cm",
    "waist_height",
    "height_cm",
    "systolic_bp",
    "diastolic_bp",
    "glucose",
    "triglycerides",
    "hdl",
    "smoke_current",
    "smoke_former",
    "alcohol_current",
]
MISSING_INDICATOR_FEATURES = [f"{feature}_missing" for feature in CORE_FEATURES if feature != "age"]
TREATMENT_SEEKING_FEATURES = ["dental_visit", "floss_days", "mobile_teeth", "floss_days_missing"]
PRIMARY_FEATURES = CORE_FEATURES + MISSING_INDICATOR_FEATURES
SECONDARY_FEATURES = PRIMARY_FEATURES + TREATMENT_SEEKING_FEATURES

RULE_OUT_THRESHOLD = 0.35
BALANCED_THRESHOLD = 0.65
SECONDARY_RULE_OUT_THRESHOLD = 0.37
SECONDARY_BALANCED_THRESHOLD = 0.67


@dataclass
class FittedEnsemble:
    models: list
    calibrator: IsotonicRegression
    features: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = ensemble_raw_probability(self.models, X[self.features])
        return np.asarray(self.calibrator.predict(raw), dtype=float)


def write_json(path: Path | str, payload) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def clean_numeric(series: pd.Series, missing_codes: Iterable[float] = ()) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values.isin(list(missing_codes)))


def first_existing(df: pd.DataFrame, columns: Iterable[str], default=np.nan) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column]
    return pd.Series(default, index=df.index)


def average_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return pd.Series(np.nan, index=df.index)
    return df[existing].apply(pd.to_numeric, errors="coerce").mean(axis=1)


def build_modeling_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Build the maintained 29/33-feature modeling frame from processed NHANES data."""
    frame = pd.DataFrame(index=df.index)
    frame["participant_id"] = first_existing(df, ["participant_id", "SEQN"])
    frame["cycle"] = df["cycle"]
    frame["age"] = clean_numeric(first_existing(df, ["age", "RIDAGEYR"]))
    frame["sex"] = clean_numeric(first_existing(df, ["sex", "RIAGENDR"]), missing_codes=(7, 9)).map({1: 1.0, 2: 0.0})
    frame["education"] = clean_numeric(first_existing(df, ["education", "DMDEDUC2"]), missing_codes=(7, 9))

    frame["bmi"] = clean_numeric(first_existing(df, ["bmi", "BMXBMI"]))
    frame["waist_cm"] = clean_numeric(first_existing(df, ["waist_cm", "waist_circumference", "BMXWAIST"]))
    frame["height_cm"] = clean_numeric(first_existing(df, ["height_cm", "height", "BMXHT"]))
    frame["waist_height"] = frame["waist_cm"] / frame["height_cm"].replace(0, np.nan)

    frame["systolic_bp"] = clean_numeric(first_existing(df, ["systolic_bp"]))
    if frame["systolic_bp"].isna().all():
        frame["systolic_bp"] = average_columns(df, ["systolic_bp_1", "systolic_bp_2", "systolic_bp_3"])
    frame["diastolic_bp"] = clean_numeric(first_existing(df, ["diastolic_bp"]))
    if frame["diastolic_bp"].isna().all():
        frame["diastolic_bp"] = average_columns(df, ["diastolic_bp_1", "diastolic_bp_2", "diastolic_bp_3"])

    frame["glucose"] = clean_numeric(first_existing(df, ["glucose", "fasting_glucose", "LBXGLU"]))
    frame["triglycerides"] = clean_numeric(first_existing(df, ["triglycerides", "LBXTR"]))
    frame["hdl"] = clean_numeric(first_existing(df, ["hdl", "LBDHDD"]))

    smoking_now = clean_numeric(first_existing(df, ["smoking_now", "SMQ040"]), missing_codes=(7, 9))
    smoked_100 = clean_numeric(first_existing(df, ["smoked_100_cigs", "SMQ020"]), missing_codes=(7, 9))
    frame["smoke_current"] = smoking_now.isin([1, 2]).astype(float).mask(smoking_now.isna())
    frame["smoke_former"] = ((smoked_100 == 1) & (smoking_now == 3)).astype(float).mask(
        smoked_100.isna() & smoking_now.isna()
    )

    alcohol_year = clean_numeric(first_existing(df, ["ever_12_drinks_year", "ALQ101"]), missing_codes=(7, 9))
    alcohol_lifetime = clean_numeric(first_existing(df, ["ever_12_drinks_lifetime", "ALQ110"]), missing_codes=(7, 9))
    alcohol_source = alcohol_year.where(alcohol_year.notna(), alcohol_lifetime)
    frame["alcohol_current"] = (alcohol_source == 1).astype(float).mask(alcohol_source.isna())

    dental_visit = clean_numeric(first_existing(df, ["time_since_dental_visit", "OHQ030"]), missing_codes=(7, 9))
    floss_days = clean_numeric(first_existing(df, ["floss_days_per_week", "OHQ620"]), missing_codes=(77, 99))
    loose_teeth = clean_numeric(first_existing(df, ["loose_teeth", "mobile_teeth", "OHQ845", "OHQ680"]), missing_codes=(7, 9))
    frame["dental_visit"] = (dental_visit <= 2).astype(float).mask(dental_visit.isna())
    frame["floss_days"] = floss_days.where(floss_days.between(0, 7))
    frame["mobile_teeth"] = (loose_teeth == 1).astype(float).mask(loose_teeth.isna())

    for feature in CORE_FEATURES:
        if feature != "age":
            frame[f"{feature}_missing"] = frame[feature].isna().astype(int)
    frame["floss_days_missing"] = frame["floss_days"].isna().astype(int)

    outcome = first_existing(df, ["has_periodontitis", "periodontitis_binary"])
    frame["has_periodontitis"] = outcome.astype(int)
    frame["perio_class"] = first_existing(df, ["perio_class"], default="")
    for column in ["exam_weight", "fasting_weight", "survey_psu", "survey_strata"]:
        if column in df.columns:
            frame[column] = df[column]

    frame["age_group"] = pd.cut(
        frame["age"],
        bins=[29, 44, 64, np.inf],
        labels=["30-44", "45-64", "65+"],
        include_lowest=True,
    ).astype(object)
    frame["smoking"] = np.select(
        [frame["smoke_current"] == 1, frame["smoke_former"] == 1],
        ["current", "former"],
        default="never/unknown",
    )
    frame["metabolic_risk"] = np.where(
        (frame["bmi"] >= 30)
        | (frame["glucose"] >= 126)
        | (frame["triglycerides"] >= 150)
        | (frame["systolic_bp"] >= 130)
        | (frame["diastolic_bp"] >= 80),
        "elevated",
        "not_elevated",
    )

    return frame


def feature_sets() -> dict[str, list[str]]:
    return {
        "deployment_ready": CORE_FEATURES,
        "primary": PRIMARY_FEATURES,
        "secondary": SECONDARY_FEATURES,
    }


def assert_feature_contract(frame: pd.DataFrame) -> None:
    missing = sorted(set(SECONDARY_FEATURES + ["has_periodontitis", "cycle"]) - set(frame.columns))
    if missing:
        raise KeyError(f"Modeling frame missing required columns: {missing}")
    if len(PRIMARY_FEATURES) != 29 or len(SECONDARY_FEATURES) != 33:
        raise AssertionError("Feature contract must remain 29 primary and 33 secondary predictors.")


def build_base_models(seed: int = SEED) -> list:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    return [
        CatBoostClassifier(
            iterations=200,
            depth=5,
            learning_rate=0.05,
            loss_function="Logloss",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        ),
        XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=seed,
        ),
        LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=seed,
            verbose=-1,
        ),
    ]


def ensemble_raw_probability(models: list, X: pd.DataFrame) -> np.ndarray:
    probabilities = [model.predict_proba(X)[:, 1] for model in models]
    return np.mean(probabilities, axis=0)


def fit_calibrated_ensemble(
    X: pd.DataFrame,
    y: pd.Series,
    features: list[str],
    seed: int = SEED,
    calibration_fraction: float = 0.2,
) -> FittedEnsemble:
    y = y.astype(int)
    if y.nunique() < 2:
        raise ValueError("Model fitting requires both outcome classes.")

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X[features],
        y,
        test_size=calibration_fraction,
        random_state=seed,
        stratify=y,
    )
    models = build_base_models(seed)
    for model in models:
        model.fit(X_fit, y_fit)
    raw_cal = ensemble_raw_probability(models, X_cal)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(raw_cal, y_cal)
    return FittedEnsemble(models=models, calibrator=calibrator, features=features)


def cross_validated_predictions(
    frame: pd.DataFrame,
    features: list[str],
    n_folds: int = 5,
    seed: int = SEED,
) -> np.ndarray:
    y = frame["has_periodontitis"].astype(int)
    predictions = np.zeros(len(frame), dtype=float)
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(frame[features], y), start=1):
        fitted = fit_calibrated_ensemble(
            frame.iloc[train_idx],
            y.iloc[train_idx],
            features,
            seed=seed + fold_idx,
        )
        predictions[test_idx] = fitted.predict_proba(frame.iloc[test_idx])
    return predictions


def summarize_predictions(
    y_true: pd.Series,
    y_prob: np.ndarray,
    n_features: int,
    name: str,
    rule_out_threshold: float = RULE_OUT_THRESHOLD,
    balanced_threshold: float = BALANCED_THRESHOLD,
) -> dict:
    rule_out = compute_metrics(y_true, y_prob, threshold=rule_out_threshold)
    balanced = compute_metrics(y_true, y_prob, threshold=balanced_threshold)
    return {
        "name": name,
        "n_features": int(n_features),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "rule_out_recall": rule_out["sensitivity"],
        "rule_out_specificity": rule_out["specificity"],
        "balanced_recall": balanced["sensitivity"],
        "balanced_specificity": balanced["specificity"],
        "rule_out": rule_out,
        "balanced": balanced,
    }


def summary_artifact(
    summary: dict,
    model_name: str,
    description: str,
    dataset: str,
    prevalence: float,
    threshold_rule_out: float,
    threshold_balanced: float,
    extra: dict | None = None,
) -> dict:
    payload = {
        "model": model_name,
        "description": description,
        "dataset": dataset,
        "n_samples": int(summary["rule_out"]["tn"] + summary["rule_out"]["fp"] + summary["rule_out"]["fn"] + summary["rule_out"]["tp"]),
        "n_features": int(summary["n_features"]),
        "prevalence": float(prevalence),
        "metrics": {
            "auc_roc": summary["auc"],
            "pr_auc": summary["pr_auc"],
            "brier_score": summary["brier_score"],
        },
        "operating_points": {
            "rule_out": operating_point_payload(summary["rule_out"], threshold_rule_out, "High-sensitivity triage"),
            "balanced": operating_point_payload(summary["balanced"], threshold_balanced, "Balanced triage"),
        },
        "confusion_matrix": {
            "rule_out": confusion_payload(summary["rule_out"], threshold_rule_out),
            "balanced": confusion_payload(summary["balanced"], threshold_balanced),
        },
        "calibration": "nested_isotonic_regression",
        "timestamp": timestamp(),
    }
    if extra:
        payload.update(extra)
    return payload


def operating_point_payload(metrics: dict, threshold: float, use_case: str) -> dict:
    return {
        "name": use_case,
        "threshold": float(threshold),
        "recall": metrics["sensitivity"],
        "specificity": metrics["specificity"],
        "precision": metrics["precision"],
        "npv": npv(metrics),
        "f1": metrics["f1_score"],
        "use_case": "Triage operating point; periodontal examination remains required.",
    }


def confusion_payload(metrics: dict, threshold: float) -> dict:
    return {
        "threshold": float(threshold),
        "tp": int(metrics["tp"]),
        "fp": int(metrics["fp"]),
        "tn": int(metrics["tn"]),
        "fn": int(metrics["fn"]),
    }


def npv(metrics: dict) -> float:
    denominator = metrics["tn"] + metrics["fn"]
    return float(metrics["tn"] / denominator) if denominator else 0.0


def bootstrap_ci(
    y_true: pd.Series,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 500,
    seed: int = SEED,
) -> tuple[float, list[float]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_prob).astype(float)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        scores.append(float(metric_fn(y[idx], p[idx])))
    mean = float(metric_fn(y, p))
    if not scores:
        return mean, [float("nan"), float("nan")]
    return mean, [float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))]


def feature_missingness_shift(train: pd.DataFrame, temporal: pd.DataFrame, features: Iterable[str]) -> dict:
    rates = {}
    deltas = {}
    for feature in features:
        train_rate = float(train[feature].isna().mean() * 100)
        temporal_rate = float(temporal[feature].isna().mean() * 100)
        delta = temporal_rate - train_rate
        rates[feature] = {
            "train_2011_2014": train_rate,
            "temporal_2009_2010": temporal_rate,
            "delta": delta,
        }
        deltas[feature] = abs(delta)
    max_feature = max(deltas, key=deltas.get) if deltas else None
    return {
        "description": "Per-feature NaN rates: NHANES 2011-2014 training data vs 2009-2010 same-source temporal validation data",
        "feature_nan_rates": rates,
        "flags": {
            "delta_exceeds_10pct": [feature for feature, delta in deltas.items() if delta > 10],
            "max_delta_feature": max_feature,
            "max_delta_value": deltas.get(max_feature, 0.0) if max_feature else 0.0,
        },
        "interpretation": "Missingness shift is reported descriptively; transportability remains limited to NHANES-like data.",
        "timestamp": timestamp(),
    }
