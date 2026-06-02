#!/usr/bin/env python3
"""Generate publication sensitivity tables from processed NHANES predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.publication_analysis import (
    missingness_table,
    prevalence_table,
    subgroup_performance_table,
)


DEFAULT_SUBGROUPS = [
    "age_group",
    "sex",
    "race_ethnicity",
    "education",
    "smoking",
    "metabolic_risk",
]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path.suffix}")


def dataframe_payload(df: pd.DataFrame) -> list[dict]:
    clean = df.astype(object).where(pd.notna(df), None)
    return clean.to_dict(orient="records")


def markdown_table(df: pd.DataFrame) -> str:
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


def resolve_weight_col(df: pd.DataFrame, requested: str) -> str | None:
    if requested in df.columns:
        return requested
    aliases = {
        "exam_weight": "WTMEC2YR",
        "WTMEC2YR": "exam_weight",
    }
    fallback = aliases.get(requested)
    if fallback in df.columns:
        return fallback
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/processed/publication_predictions.parquet",
        help="Processed data with outcome, optional survey weights, and optional predictions.",
    )
    parser.add_argument("--out-json", default="results/publication_sensitivity_tables.json")
    parser.add_argument("--out-md", default="results/publication_sensitivity_tables.md")
    parser.add_argument("--outcome-col", default="has_periodontitis")
    parser.add_argument("--probability-col", default="predicted_probability")
    parser.add_argument("--weight-col", default="exam_weight")
    parser.add_argument("--feature-cols", nargs="*", default=[])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input table not found: {input_path}. Run `make process && make temporal` first."
        )

    df = load_table(input_path)
    if args.outcome_col not in df.columns:
        raise KeyError(f"Outcome column missing from input table: {args.outcome_col}")
    weight_col = resolve_weight_col(df, args.weight_col)

    payload = {
        "input": str(input_path),
        "prevalence_by_cycle": dataframe_payload(
            prevalence_table(df, args.outcome_col, weight_col, by=("cycle",))
        ),
    }

    subgroup_cols = [col for col in DEFAULT_SUBGROUPS if col in df.columns]
    if args.probability_col in df.columns and subgroup_cols:
        payload["subgroup_performance"] = dataframe_payload(
            subgroup_performance_table(
                df,
                args.outcome_col,
                args.probability_col,
                subgroup_cols,
            )
        )
    else:
        payload["subgroup_performance_note"] = (
            "Prediction probabilities or subgroup columns were unavailable."
        )

    if args.feature_cols:
        payload["missingness"] = dataframe_payload(missingness_table(df, args.feature_cols))

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    out_md = Path(args.out_md)
    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Publication Sensitivity Tables\n\n")
        f.write("## Prevalence by Cycle\n\n")
        f.write(markdown_table(pd.DataFrame(payload["prevalence_by_cycle"])))
        f.write("\n\n")
        if "subgroup_performance" in payload:
            f.write("## Subgroup Performance\n\n")
            f.write(markdown_table(pd.DataFrame(payload["subgroup_performance"])))
            f.write("\n")
        else:
            f.write(payload["subgroup_performance_note"] + "\n")

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
