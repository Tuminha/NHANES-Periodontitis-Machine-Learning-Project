import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labels import (  # noqa: E402
    INTERPROXIMAL_SITES,
    VALID_TEETH,
    classify_mild,
    classify_moderate,
    classify_severe,
    create_synthetic_test_cases,
    label_periodontitis,
)


def healthy_row() -> pd.Series:
    row = {}
    for tooth in VALID_TEETH:
        for site in INTERPROXIMAL_SITES:
            row[f"OHX{tooth:02d}LA{site}"] = 1.0
            row[f"OHX{tooth:02d}PC{site}"] = 2.0
    return pd.Series(row)


def test_valid_teeth_exclude_third_molars():
    assert 1 not in VALID_TEETH
    assert 16 not in VALID_TEETH
    assert 17 not in VALID_TEETH
    assert 32 not in VALID_TEETH
    assert len(VALID_TEETH) == 28


def test_severe_periodontitis_requires_cal_on_two_teeth_and_pd_site():
    row = healthy_row()
    row["OHX02LAM"] = 7.0
    row["OHX03LAM"] = 7.0
    row["OHX02PCM"] = 6.0

    assert classify_severe(row) is True
    assert classify_moderate(row) is True
    assert classify_mild(row) is True


def test_moderate_periodontitis_cal_based_without_severe():
    row = healthy_row()
    row["OHX04LAM"] = 5.0
    row["OHX05LAM"] = 5.0

    assert classify_severe(row) is False
    assert classify_moderate(row) is True
    assert classify_mild(row) is False


def test_moderate_periodontitis_requires_pd_on_different_teeth():
    row = healthy_row()
    row["OHX06PCM"] = 5.0
    row["OHX06PCD"] = 5.0

    assert classify_moderate(row) is False
    assert classify_mild(row) is True


def test_mild_periodontitis_from_cal_and_pd_on_two_teeth():
    row = healthy_row()
    row["OHX06LAM"] = 3.0
    row["OHX07LAM"] = 3.0
    row["OHX06PCM"] = 4.0
    row["OHX07PCM"] = 4.0

    assert classify_severe(row) is False
    assert classify_moderate(row) is False
    assert classify_mild(row) is True


def test_third_molar_like_columns_do_not_affect_classification():
    row = healthy_row()
    row["OHX01LAM"] = 9.0
    row["OHX16LAM"] = 9.0
    row["OHX17PCM"] = 9.0
    row["OHX32PCD"] = 9.0

    assert classify_severe(row) is False
    assert classify_moderate(row) is False
    assert classify_mild(row) is False


def test_full_label_pipeline_hierarchical_classes():
    df = create_synthetic_test_cases()
    labeled = label_periodontitis(df.copy())

    assert labeled["perio_class"].tolist() == ["severe", "moderate", "mild", "none"]
    assert labeled["has_periodontitis"].tolist() == [True, True, True, False]
