#!/usr/bin/env python3
"""Lightweight submission-readiness checks that do not run the full reproduction."""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.check_publication_consistency import check_docs, check_result_files
from scripts import download_nhanes


PUBLICATION_FILES = [
    ROOT / "README.md",
    ROOT / "MODEL_CARD.md",
    ROOT / "docs/publication/ARTICLE_DRAFT.md",
    ROOT / "docs/publication/ARTICLE_PEER_REVIEW.md",
    ROOT / "results/v13_primary_norc_summary.json",
    ROOT / "results/v13_secondary_full_summary.json",
    ROOT / "results/external_0910_metrics.json",
    ROOT / "results/publication_sensitivity_tables.md",
]


def check_json_files() -> None:
    for path in sorted((ROOT / "results").glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            json.load(f)


def check_yaml_files() -> None:
    for path in [ROOT / "CITATION.cff", ROOT / "configs/config.yaml"]:
        with path.open("r", encoding="utf-8") as f:
            if yaml.safe_load(f) is None:
                raise AssertionError(f"{path.relative_to(ROOT)} parsed to empty YAML")


def check_reproduction_hooks() -> None:
    for script in ["scripts/run_v13_primary.sh", "scripts/run_external_validation.sh"]:
        text = (ROOT / script).read_text(encoding="utf-8")
        if "nbconvert" in text or "papermill" in text:
            raise AssertionError(f"{script} still uses notebook execution.")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    for target in ["verify-submission", "reproduce-full", "scripts/reproduce_v13_primary.py", "scripts/run_temporal_validation.py"]:
        if target not in makefile:
            raise AssertionError(f"Makefile missing expected submission target or script reference: {target}")


def check_temporal_metric_shape() -> None:
    metrics = json.loads((ROOT / "results/external_0910_metrics.json").read_text(encoding="utf-8"))
    for metric in ["auc", "prauc", "brier"]:
        payload = metrics["metrics"][metric]
        if "mean" not in payload or "ci95" not in payload or len(payload["ci95"]) != 2:
            raise AssertionError(f"Temporal metric missing mean/ci95 shape: {metric}")
    for point in ["rule_out_t_0.35", "balanced_t_0.65"]:
        payload = metrics["operating_points"][point]
        for field in ["sensitivity", "specificity", "ppv", "npv"]:
            if field not in payload:
                raise AssertionError(f"Temporal operating point {point} missing {field}")


def check_publication_analysis_outputs() -> None:
    path = ROOT / "results/publication_sensitivity_tables.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    prevalence = payload.get("prevalence_by_cycle", [])
    subgroup = payload.get("subgroup_performance", [])
    if not prevalence:
        raise AssertionError("Publication sensitivity output is missing prevalence_by_cycle rows.")
    if not subgroup:
        raise AssertionError("Publication sensitivity output is missing subgroup_performance rows.")
    for row in prevalence:
        if "weighted_prevalence" not in row:
            raise AssertionError("Publication prevalence row missing weighted_prevalence.")
    subgroup_variables = {row.get("subgroup_variable") for row in subgroup}
    required = {"age_group", "sex", "education", "smoking", "metabolic_risk"}
    missing = required - subgroup_variables
    if missing:
        raise AssertionError(f"Publication subgroup output missing strata: {sorted(missing)}")


def check_publication_wording() -> None:
    banned = ["Publication Ready", "External Validation", "clinical deployment", "negative result rules out"]
    for path in PUBLICATION_FILES:
        text = path.read_text(encoding="utf-8")
        for phrase in banned:
            if phrase in text:
                raise AssertionError(f"{path.relative_to(ROOT)} contains banned phrase: {phrase}")


def check_nhanes_urls() -> None:
    urls = [
        download_nhanes.NHANES_FILES["2009-2010"]["demographics"],
        download_nhanes.NHANES_FILES["2011-2012"]["periodontal"],
        download_nhanes.NHANES_FILES["2013-2014"]["hdl"],
    ]
    for url in urls:
        with urllib.request.urlopen(url, timeout=20) as response:
            head = response.read(20)
        if not head.startswith(b"HEADER RECORD"):
            raise AssertionError(f"CDC URL did not return an XPT header: {url}")


def check_manuscript_render_support() -> None:
    if shutil.which("pandoc") is None:
        print("pandoc not installed; PDF render check skipped.")
    else:
        print("pandoc installed; `make manuscript` can render PDF if a PDF engine is available.")


def main() -> None:
    check_json_files()
    check_yaml_files()
    check_result_files()
    check_docs()
    check_reproduction_hooks()
    check_temporal_metric_shape()
    check_publication_analysis_outputs()
    check_publication_wording()
    check_nhanes_urls()
    check_manuscript_render_support()
    print("Submission-readiness checks passed.")


if __name__ == "__main__":
    main()
