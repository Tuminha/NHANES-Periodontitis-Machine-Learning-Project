#!/usr/bin/env python3
"""Check that publication-facing files use the same canonical results."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "primary_features": 29,
    "secondary_features": 33,
    "primary_auc": 0.717245742046474,
    "primary_pr_auc": 0.8157447372867956,
    "secondary_auc": 0.7255326805774952,
    "temporal_auc": 0.6771141964954918,
    "temporal_pr_auc": 0.7734533687334428,
    "temporal_brier": 0.20025236260487186,
    "temporal_rule_out_sensitivity": 0.970968669157804,
    "temporal_rule_out_specificity": 0.18080094228504123,
    "temporal_balanced_sensitivity": 0.8263868927852831,
    "temporal_balanced_specificity": 0.4334511189634865,
}

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


def close(actual: float, expected: float, tol: float = 1e-6) -> bool:
    return math.isclose(float(actual), float(expected), rel_tol=tol, abs_tol=tol)


def assert_close(label: str, actual: float, expected: float) -> None:
    if not close(actual, expected):
        raise AssertionError(f"{label}: expected {expected}, got {actual}")


def check_result_files() -> None:
    featuredrop = load_json("results/v13_featuredrop.json")
    temporal = load_json("results/external_0910_metrics.json")

    primary = featuredrop["v1.3_no_reverse_causality"]
    secondary = featuredrop["v1.3_full"]

    assert primary["n_features"] == EXPECTED["primary_features"]
    assert secondary["n_features"] == EXPECTED["secondary_features"]
    assert_close("primary AUC", primary["auc"], EXPECTED["primary_auc"])
    assert_close("primary PR-AUC", primary["pr_auc"], EXPECTED["primary_pr_auc"])
    assert_close("secondary AUC", secondary["auc"], EXPECTED["secondary_auc"])

    assert_close("temporal AUC", temporal["metrics"]["auc"]["mean"], EXPECTED["temporal_auc"])
    assert_close("temporal PR-AUC", temporal["metrics"]["prauc"]["mean"], EXPECTED["temporal_pr_auc"])
    assert_close("temporal Brier", temporal["metrics"]["brier"]["mean"], EXPECTED["temporal_brier"])

    rule_out = temporal["operating_points"]["rule_out_t_0.35"]
    balanced = temporal["operating_points"]["balanced_t_0.65"]
    assert_close(
        "temporal rule-out sensitivity",
        rule_out["sensitivity"],
        EXPECTED["temporal_rule_out_sensitivity"],
    )
    assert_close(
        "temporal rule-out specificity",
        rule_out["specificity"],
        EXPECTED["temporal_rule_out_specificity"],
    )
    assert_close(
        "temporal balanced sensitivity",
        balanced["sensitivity"],
        EXPECTED["temporal_balanced_sensitivity"],
    )
    assert_close(
        "temporal balanced specificity",
        balanced["specificity"],
        EXPECTED["temporal_balanced_specificity"],
    )


def check_docs() -> None:
    for path in DOCS:
        text = path.read_text(encoding="utf-8")
        for phrase in BANNED_PHRASES:
            if phrase in text:
                rel = path.relative_to(ROOT)
                raise AssertionError(f"{rel} contains banned phrase: {phrase}")

        if "29" not in text or "33" not in text:
            rel = path.relative_to(ROOT)
            raise AssertionError(f"{rel} must mention canonical 29/33 feature counts")

        if "same-source temporal validation" not in text.lower():
            rel = path.relative_to(ROOT)
            raise AssertionError(f"{rel} must use same-source temporal validation framing")


def main() -> None:
    check_result_files()
    check_docs()
    print("Publication consistency checks passed.")


if __name__ == "__main__":
    main()
