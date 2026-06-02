import pandas as pd

from src.reproduction import (
    PRIMARY_FEATURES,
    SECONDARY_FEATURES,
    assert_feature_contract,
    build_modeling_frame,
    feature_sets,
)


def toy_processed_frame():
    return pd.DataFrame(
        {
            "participant_id": [1, 2],
            "cycle": ["2011-2012", "2009-2010"],
            "age": [45, 67],
            "sex": [1, 2],
            "education": [4, 2],
            "bmi": [26.0, None],
            "waist_circumference": [90.0, 101.0],
            "height_cm": [170.0, 165.0],
            "systolic_bp_1": [120, 140],
            "systolic_bp_2": [122, 142],
            "diastolic_bp_1": [75, 85],
            "diastolic_bp_2": [76, 86],
            "fasting_glucose": [95, None],
            "triglycerides": [130, 180],
            "hdl": [55, 42],
            "smoked_100_cigs": [1, 2],
            "smoking_now": [3, None],
            "ever_12_drinks_year": [1, 2],
            "time_since_dental_visit": [2, 4],
            "floss_days_per_week": [5, None],
            "loose_teeth": [2, 1],
            "has_periodontitis": [1, 0],
            "perio_class": ["moderate", "none"],
            "exam_weight": [1.2, 0.8],
        }
    )


def test_feature_contract_is_29_primary_and_33_secondary():
    sets = feature_sets()

    assert len(PRIMARY_FEATURES) == 29
    assert len(SECONDARY_FEATURES) == 33
    assert len(sets["deployment_ready"]) == 15
    assert set(sets["primary"]).issubset(set(sets["secondary"]))


def test_build_modeling_frame_produces_required_columns_and_subgroups():
    frame = build_modeling_frame(toy_processed_frame())

    assert_feature_contract(frame)
    assert frame.loc[0, "waist_height"] == 90.0 / 170.0
    assert frame.loc[1, "bmi_missing"] == 1
    assert frame.loc[0, "floss_days_missing"] == 0
    assert {"age_group", "smoking", "metabolic_risk", "exam_weight"}.issubset(frame.columns)
