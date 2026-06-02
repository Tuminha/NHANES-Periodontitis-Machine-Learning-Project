#!/usr/bin/env python3
"""Check that publication-facing files use the same current result artifacts."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DOCS = [
    ROOT / "README.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "docs/publication/ARTICLE_DRAFT.md",
]

BANNED_PHRASES = [
    "Publication Ready",
    "publication ready",
    "clinical deployment",
    "negative result rules out",
    "practical rule-out tool",
    "External Validation",
    "external validation",
    "27 features",
    "31 features",
    "AUC >0.95 were driven",
]


def load_json(path: str) -> dict:
    with (ROOT / path).open("r", encoding="utf-8") as f:
        return json.load(f)


def fmt4(value: float) -> str:
    return f"{float(value):.4f}"


def pct1(value: float) -> str:
    return f"{float(value) * 100:.1f}%"


def canonical_values() -> dict[str, float | int | str]:
    featuredrop = load_json("results/v13_featuredrop.json")
    temporal = load_json("results/external_0910_metrics.json")
    primary = featuredrop["v1.3_no_reverse_causality"]
    secondary = featuredrop["v1.3_full"]
    rule_out = temporal["operating_points"]["rule_out_t_0.35"]
    balanced = temporal["operating_points"]["balanced_t_0.65"]
    return {
        "primary_features": int(primary["n_features"]),
        "secondary_features": int(secondary["n_features"]),
        "primary_auc": float(primary["auc"]),
        "primary_pr_auc": float(primary["pr_auc"]),
        "secondary_auc": float(secondary["auc"]),
        "secondary_pr_auc": float(secondary["pr_auc"]),
        "temporal_auc": float(temporal["metrics"]["auc"]["mean"]),
        "temporal_pr_auc": float(temporal["metrics"]["prauc"]["mean"]),
        "temporal_brier": float(temporal["metrics"]["brier"]["mean"]),
        "temporal_rule_out_sensitivity": float(rule_out["sensitivity"]),
        "temporal_rule_out_specificity": float(rule_out["specificity"]),
        "temporal_balanced_sensitivity": float(balanced["sensitivity"]),
        "temporal_balanced_specificity": float(balanced["specificity"]),
    }


def check_result_files() -> None:
    values = canonical_values()
    if values["primary_features"] != 29:
        raise AssertionError(f"Primary feature count must be 29, got {values['primary_features']}")
    if values["secondary_features"] != 33:
        raise AssertionError(f"Secondary feature count must be 33, got {values['secondary_features']}")
    for key in [
        "primary_auc",
        "primary_pr_auc",
        "secondary_auc",
        "secondary_pr_auc",
        "temporal_auc",
        "temporal_pr_auc",
        "temporal_brier",
    ]:
        value = float(values[key])
        if not 0 <= value <= 1:
            raise AssertionError(f"{key} must be in [0, 1], got {value}")


def required_strings(values: dict) -> list[str]:
    return [
        str(values["primary_features"]),
        str(values["secondary_features"]),
        fmt4(values["primary_auc"]),
        fmt4(values["primary_pr_auc"]),
        fmt4(values["secondary_auc"]),
        fmt4(values["secondary_pr_auc"]),
        fmt4(values["temporal_auc"]),
        fmt4(values["temporal_pr_auc"]),
    ]


def check_docs() -> None:
    values = canonical_values()
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for phrase in BANNED_PHRASES:
            if phrase in text:
                rel = path.relative_to(ROOT)
                raise AssertionError(f"{rel} contains banned phrase: {phrase}")

        missing = [value for value in required_strings(values) if value not in text]
        if missing:
            rel = path.relative_to(ROOT)
            raise AssertionError(f"{rel} is missing current result values: {missing}")

        if "same-source temporal validation" not in text.lower():
            rel = path.relative_to(ROOT)
            raise AssertionError(f"{rel} must use same-source temporal validation framing")


def main() -> None:
    check_result_files()
    check_docs()
    print("Publication consistency checks passed.")


if __name__ == "__main__":
    main()
