"""
CDC/AAP Periodontitis Case Definitions
Author: Francisco Teixeira Barbosa (Cisco)

Reference: Eke PI, Page RC, Wei L, Thornton-Evans G, Genco RJ. 
           Update of the case definitions for population-based surveillance 
           of periodontitis. J Periodontol. 2012;83(12):1449-1454.

Purpose: Implement CDC/AAP 2012 case definitions to classify periodontitis
         severity (severe, moderate, mild, none) from NHANES full-mouth
         periodontal examination data.

NHANES Periodontal Variables:
    - Pocket Depth (PD): OHXxxPCM (mesial), OHXxxPCD (distal), OHXxxPCS (mid)
    - Clinical Attachment Loss (CAL): OHXxxLAM (mesial), OHXxxLAD (distal), OHXxxLAS (mid)
    - Where xx = tooth number (02-15 upper jaw, 18-31 lower jaw)
    - Third molars (01, 16, 17, 32) are EXCLUDED from definitions

Key Constraints:
    - "Interproximal" means MESIAL + DISTAL only (exclude mid-facial/mid-lingual)
    - "On different teeth" must be enforced where specified
    - Count sites meeting thresholds, then check teeth uniqueness

Usage:
    from src.labels import label_periodontitis
    
    df_with_labels = label_periodontitis(df_nhanes_merged)
    # Returns df with columns: 'perio_class' and 'has_periodontitis'
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


# =============================================================================
# Tooth and Site Definitions
# =============================================================================

# Valid tooth numbers (exclude third molars: 01, 16, 17, 32)
VALID_TEETH = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,  # Upper jaw
    18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31  # Lower jaw
]

# Interproximal sites only (M = Mesial, D = Distal)
INTERPROXIMAL_SITES = ['M', 'D']


def build_variable_lists() -> Tuple[List[str], List[str]]:
    """
    Build lists of NHANES variable names for PD and CAL measurements.
    
    Returns:
        pd_vars: List of pocket depth variable names (e.g., ['OHX02PCM', 'OHX02PCD', ...])
        cal_vars: List of CAL variable names (e.g., ['OHX02LAM', 'OHX02LAD', ...])
    
    Note: Only includes interproximal sites (M, D) for valid teeth.
    
    TODO: Loop over VALID_TEETH and INTERPROXIMAL_SITES
    TODO: Construct variable names following NHANES convention:
          - PD: OHXxxPCM, OHXxxPCD (where xx is zero-padded tooth number)
          - CAL: OHXxxLAM, OHXxxLAD
    TODO: Return two lists
    """
    # TODO: Initialize pd_vars = [], cal_vars = []
    # TODO: for tooth in VALID_TEETH:
    #           for site in ['M', 'D']:
    #               pd_var = f"OHX{tooth:02d}PC{site}"
    #               cal_var = f"OHX{tooth:02d}LA{site}"
    #               pd_vars.append(pd_var)
    #               cal_vars.append(cal_var)
    # TODO: Return pd_vars, cal_vars
    pass


def count_sites_meeting_threshold(
    row: pd.Series,
    variables: List[str],
    threshold: float,
    comparison: str = ">="
) -> int:
    """
    Count how many sites meet a given threshold for one participant.
    
    Args:
        row: Single row from DataFrame (one participant)
        variables: List of variable names to check
        threshold: Numeric threshold value (e.g., 4, 5, 6)
        comparison: String ">=" or ">" or "<=" etc.
    
    Returns:
        count: Number of sites meeting the condition
    
    TODO: Extract values from row[variables]
    TODO: Drop NaN values
    TODO: Apply comparison (e.g., >= threshold)
    TODO: Return count of True values
    """
    # TODO: values = row[variables].dropna()
    # TODO: if comparison == ">=":
    #           mask = values >= threshold
    # TODO: return mask.sum()
    pass


def count_teeth_with_any_site_meeting_threshold(
    row: pd.Series,
    measurement_type: str,  # "PD" or "CAL"
    threshold: float,
    comparison: str = ">="
) -> int:
    """
    Count how many DIFFERENT TEETH have at least one interproximal site
    meeting the threshold.
    
    This enforces the "on different teeth" constraint in CDC/AAP definitions.
    
    Args:
        row: Single row from DataFrame
        measurement_type: "PD" or "CAL"
        threshold: Numeric threshold
        comparison: String ">=" or ">"
    
    Returns:
        count: Number of unique teeth with >= 1 site meeting threshold
    
    TODO: Loop over VALID_TEETH
    TODO: For each tooth, check both mesial and distal sites
    TODO: If ANY interproximal site >= threshold, count that tooth
    TODO: Return total count of teeth
    """
    # TODO: affected_teeth = 0
    # TODO: for tooth in VALID_TEETH:
    #           if measurement_type == "PD":
    #               var_m = f"OHX{tooth:02d}PCM"
    #               var_d = f"OHX{tooth:02d}PCD"
    #           elif measurement_type == "CAL":
    #               var_m = f"OHX{tooth:02d}LAM"
    #               var_d = f"OHX{tooth:02d}LAD"
    #           val_m = row.get(var_m, np.nan)
    #           val_d = row.get(var_d, np.nan)
    #           if (not pd.isna(val_m) and val_m >= threshold) or \
    #              (not pd.isna(val_d) and val_d >= threshold):
    #               affected_teeth += 1
    # TODO: return affected_teeth
    pass


# =============================================================================
# CDC/AAP Case Definitions (Eke et al. 2012)
# =============================================================================

def classify_severe(row: pd.Series) -> bool:
    """
    Severe periodontitis:
        >= 2 interproximal sites with CAL >= 6 mm (on different teeth) AND
        >= 1 interproximal site with PD >= 5 mm
    
    TODO: Use count_teeth_with_any_site_meeting_threshold for CAL >= 6
    TODO: Use count_sites_meeting_threshold for PD >= 5
    TODO: Return True if BOTH conditions met
    """
    # TODO: cal_6_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 6)
    # TODO: pd_5_sites = count_sites_meeting_threshold(row, build_variable_lists()[0], 5)
    # TODO: return (cal_6_teeth >= 2) and (pd_5_sites >= 1)
    pass


def classify_moderate(row: pd.Series) -> bool:
    """
    Moderate periodontitis:
        >= 2 interproximal sites with CAL >= 4 mm (on different teeth) OR
        >= 2 interproximal sites with PD >= 5 mm (on different teeth)
    
    TODO: Check CAL >= 4 on >= 2 different teeth
    TODO: Check PD >= 5 on >= 2 different teeth
    TODO: Return True if EITHER condition met
    """
    # TODO: cal_4_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 4)
    # TODO: pd_5_teeth = count_teeth_with_any_site_meeting_threshold(row, "PD", 5)
    # TODO: return (cal_4_teeth >= 2) or (pd_5_teeth >= 2)
    pass


def classify_mild(row: pd.Series) -> bool:
    """
    Mild periodontitis:
        (>= 2 interproximal sites with CAL >= 3 mm AND 
         >= 2 interproximal sites with PD >= 4 mm on different teeth)
        OR
        one site with PD >= 5 mm
    
    Note: This is the most complex definition. Implement carefully.
    
    TODO: Check CAL >= 3 on >= 2 different teeth
    TODO: Check PD >= 4 on >= 2 different teeth
    TODO: Check PD >= 5 on >= 1 site (any site)
    TODO: Return True if (CAL AND PD condition) OR (PD >= 5)
    """
    # TODO: cal_3_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 3)
    # TODO: pd_4_teeth = count_teeth_with_any_site_meeting_threshold(row, "PD", 4)
    # TODO: pd_5_sites = count_sites_meeting_threshold(row, build_variable_lists()[0], 5)
    # TODO: condition_a = (cal_3_teeth >= 2) and (pd_4_teeth >= 2)
    # TODO: condition_b = (pd_5_sites >= 1)
    # TODO: return condition_a or condition_b
    pass


def label_periodontitis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Main function: Apply CDC/AAP case definitions to classify periodontitis.
    
    Args:
        df: DataFrame with merged NHANES data including OHXxxPCM, OHXxxPCD, 
            OHXxxLAM, OHXxxLAD columns for all valid teeth
    
    Returns:
        df: Same DataFrame with two new columns:
            - 'perio_class': str ("none", "mild", "moderate", "severe")
            - 'has_periodontitis': bool (True if any severity, False if none)
    
    Classification hierarchy (mutually exclusive):
        1. Check severe first
        2. If not severe, check moderate
        3. If not moderate, check mild
        4. Otherwise, none
    
    TODO: Apply classification functions row by row
    TODO: Create perio_class column
    TODO: Create binary has_periodontitis column
    TODO: Add assertion checks (e.g., sum of each class > 0)
    TODO: Print summary statistics (prevalence by class)
    """
    # TODO: Check that required columns exist in df
    # TODO: pd_vars, cal_vars = build_variable_lists()
    # TODO: missing_vars = set(pd_vars + cal_vars) - set(df.columns)
    # TODO: if missing_vars:
    #           raise ValueError(f"Missing variables: {missing_vars}")
    
    # TODO: Apply classification in hierarchy
    # TODO: df['is_severe'] = df.apply(classify_severe, axis=1)
    # TODO: df['is_moderate'] = df.apply(classify_moderate, axis=1)
    # TODO: df['is_mild'] = df.apply(classify_mild, axis=1)
    
    # TODO: Create perio_class using nested logic
    # TODO: def assign_class(row):
    #           if row['is_severe']: return "severe"
    #           elif row['is_moderate']: return "moderate"
    #           elif row['is_mild']: return "mild"
    #           else: return "none"
    # TODO: df['perio_class'] = df.apply(assign_class, axis=1)
    
    # TODO: Create binary label
    # TODO: df['has_periodontitis'] = df['perio_class'] != "none"
    
    # TODO: Drop intermediate columns
    # TODO: df.drop(columns=['is_severe', 'is_moderate', 'is_mild'], inplace=True)
    
    # TODO: Print summary
    # TODO: print("Periodontitis Classification Summary:")
    # TODO: print(df['perio_class'].value_counts())
    # TODO: print(f"Prevalence: {df['has_periodontitis'].mean():.2%}")
    
    # TODO: Return df with new columns
    pass


# =============================================================================
# Unit Testing Helpers
# =============================================================================

def create_synthetic_test_cases() -> pd.DataFrame:
    """
    Create synthetic test rows to validate classification logic.
    
    Returns:
        df_test: Small DataFrame with known cases:
            - Row 0: Severe periodontitis (clear case)
            - Row 1: Moderate periodontitis (CAL-based)
            - Row 2: Mild periodontitis (minimal criteria)
            - Row 3: No periodontitis (healthy)
    
    Use this to write unit tests in tests/test_labels.py
    
    TODO: Create synthetic rows with explicit PD and CAL values
    TODO: Ensure each row meets known classification criteria
    TODO: Return as DataFrame with proper NHANES variable names
    """
    # TODO: Build a dict or list of dicts with synthetic data
    # TODO: For example:
    #       Row 0 (severe): CAL=7 at teeth 2,3 (mesial), PD=6 at tooth 2 (mesial)
    #       Row 1 (moderate): CAL=5 at teeth 4,5 (mesial)
    #       Row 2 (mild): CAL=3.5 at teeth 6,7 (mesial), PD=4.5 at teeth 6,7 (distal)
    #       Row 3 (none): CAL=1, PD=2 everywhere
    # TODO: Return pd.DataFrame with these rows
    pass

