import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "04_publication_analyses.py"


def load_publication_script():
    spec = importlib.util.spec_from_file_location("publication_analyses", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dataframe_payload_converts_nan_to_json_null():
    module = load_publication_script()
    payload = module.dataframe_payload(pd.DataFrame({"value": [1.0, np.nan]}))

    assert payload == [{"value": 1.0}, {"value": None}]
    assert json.dumps(payload, allow_nan=False)


def test_resolve_weight_col_accepts_processed_and_raw_names():
    module = load_publication_script()

    assert module.resolve_weight_col(pd.DataFrame({"exam_weight": [1]}), "exam_weight") == "exam_weight"
    assert module.resolve_weight_col(pd.DataFrame({"WTMEC2YR": [1]}), "exam_weight") == "WTMEC2YR"


def test_publication_analysis_cli_runs_from_outside_repo(tmp_path):
    input_path = tmp_path / "publication.csv"
    out_json = tmp_path / "tables.json"
    out_md = tmp_path / "tables.md"
    input_path.write_text(
        "cycle,has_periodontitis,exam_weight,predicted_probability,sex\n"
        "2011-2012,1,2.0,0.8,M\n"
        "2011-2012,0,1.0,0.2,F\n",
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--input",
            str(input_path),
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
        ],
        cwd=tmp_path,
        check=True,
    )

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["prevalence_by_cycle"][0]["weighted_prevalence"] == 2 / 3
    assert out_md.read_text(encoding="utf-8").startswith("# Publication Sensitivity Tables")
