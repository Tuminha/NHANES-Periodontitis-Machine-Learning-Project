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
    """
    pd_vars = []
    cal_vars = []
    
    for tooth in VALID_TEETH:
        for site in INTERPROXIMAL_SITES:  # M and D only
            pd_var = f"OHX{tooth:02d}PC{site}"
            cal_var = f"OHX{tooth:02d}LA{site}"
            pd_vars.append(pd_var)
            cal_vars.append(cal_var)
    
    return pd_vars, cal_vars


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
    """
    # Filter to only variables that exist in this row
    existing_vars = [v for v in variables if v in row.index]
    
    if not existing_vars:
        return 0
    
    # Extract values and drop NaN
    try:
        values = row[existing_vars].dropna()
    except KeyError:
        return 0
    
    if len(values) == 0:
        return 0
    
    # Apply comparison
    if comparison == ">=":
        mask = values >= threshold
    elif comparison == ">":
        mask = values > threshold
    elif comparison == "<=":
        mask = values <= threshold
    elif comparison == "<":
        mask = values < threshold
    else:
        raise ValueError(f"Unknown comparison: {comparison}")
    
    return int(mask.sum())


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
    """
    affected_teeth = 0
    
    for tooth in VALID_TEETH:
        # Build variable names for mesial and distal sites
        if measurement_type == "PD":
            var_m = f"OHX{tooth:02d}PCM"
            var_d = f"OHX{tooth:02d}PCD"
        elif measurement_type == "CAL":
            var_m = f"OHX{tooth:02d}LAM"
            var_d = f"OHX{tooth:02d}LAD"
        else:
            raise ValueError(f"Unknown measurement_type: {measurement_type}")
        
        # Get values (use .get() to safely handle missing columns)
        # This returns NaN if the column doesn't exist
        val_m = row.get(var_m, np.nan)
        val_d = row.get(var_d, np.nan)
        
        # Check if ANY interproximal site meets threshold
        tooth_affected = False
        if comparison == ">=":
            tooth_affected = (not pd.isna(val_m) and val_m >= threshold) or \
                           (not pd.isna(val_d) and val_d >= threshold)
        elif comparison == ">":
            tooth_affected = (not pd.isna(val_m) and val_m > threshold) or \
                           (not pd.isna(val_d) and val_d > threshold)
        
        if tooth_affected:
            affected_teeth += 1
    
    return affected_teeth


# =============================================================================
# CDC/AAP Case Definitions (Eke et al. 2012)
# =============================================================================

def classify_severe(row: pd.Series) -> bool:
    """
    Severe periodontitis:
        >= 2 interproximal sites with CAL >= 6 mm (on different teeth) AND
        >= 1 interproximal site with PD >= 5 mm
    """
    # Count teeth with CAL >= 6mm (must be on different teeth)
    cal_6_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 6)
    
    # Count sites with PD >= 5mm (any sites)
    pd_vars, _ = build_variable_lists()
    pd_5_sites = count_sites_meeting_threshold(row, pd_vars, 5)
    
    # Both conditions must be met
    return (cal_6_teeth >= 2) and (pd_5_sites >= 1)


def classify_moderate(row: pd.Series) -> bool:
    """
    Moderate periodontitis:
        >= 2 interproximal sites with CAL >= 4 mm (on different teeth) OR
        >= 2 interproximal sites with PD >= 5 mm (on different teeth)
    """
    # Count teeth with CAL >= 4mm (on different teeth)
    cal_4_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 4)
    
    # Count teeth with PD >= 5mm (on different teeth)
    pd_5_teeth = count_teeth_with_any_site_meeting_threshold(row, "PD", 5)
    
    # Either condition satisfies moderate
    return (cal_4_teeth >= 2) or (pd_5_teeth >= 2)


def classify_mild(row: pd.Series) -> bool:
    """
    Mild periodontitis:
        (>= 2 interproximal sites with CAL >= 3 mm AND 
         >= 2 interproximal sites with PD >= 4 mm on different teeth)
        OR
        one site with PD >= 5 mm
    
    Note: This is the most complex definition. Implement carefully.
    """
    # Condition A: CAL >= 3mm on >= 2 teeth AND PD >= 4mm on >= 2 teeth
    cal_3_teeth = count_teeth_with_any_site_meeting_threshold(row, "CAL", 3)
    pd_4_teeth = count_teeth_with_any_site_meeting_threshold(row, "PD", 4)
    condition_a = (cal_3_teeth >= 2) and (pd_4_teeth >= 2)
    
    # Condition B: At least one site with PD >= 5mm
    pd_vars, _ = build_variable_lists()
    pd_5_sites = count_sites_meeting_threshold(row, pd_vars, 5)
    condition_b = (pd_5_sites >= 1)
    
    # Either condition satisfies mild
    return condition_a or condition_b


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
    """
    print("🦷 Applying CDC/AAP periodontitis case definitions...")
    
    # Check that required periodontal variables exist
    pd_vars, cal_vars = build_variable_lists()
    missing_vars = set(pd_vars + cal_vars) - set(df.columns)
    if missing_vars:
        print(f"⚠️  Warning: {len(missing_vars)} periodontal variables missing from dataset")
        print(f"   First 10 missing: {list(missing_vars)[:10]}")
        print("   Proceeding with available data...")
    
    # Apply classification functions (in hierarchical order)
    print("   Classifying severe cases...")
    df['is_severe'] = df.apply(classify_severe, axis=1)
    
    print("   Classifying moderate cases...")
    df['is_moderate'] = df.apply(classify_moderate, axis=1)
    
    print("   Classifying mild cases...")
    df['is_mild'] = df.apply(classify_mild, axis=1)
    
    # Create perio_class using hierarchical logic (mutually exclusive)
    def assign_class(row):
        if row['is_severe']:
            return "severe"
        elif row['is_moderate']:
            return "moderate"
        elif row['is_mild']:
            return "mild"
        else:
            return "none"
    
    print("   Assigning final classifications...")
    df['perio_class'] = df.apply(assign_class, axis=1)
    
    # Create binary label
    df['has_periodontitis'] = df['perio_class'] != "none"
    
    # Drop intermediate classification columns
    df.drop(columns=['is_severe', 'is_moderate', 'is_mild'], inplace=True)
    
    # Print summary statistics
    print("\n📊 Periodontitis Classification Summary:")
    print(df['perio_class'].value_counts().sort_index())
    print(f"\n   Overall Prevalence: {df['has_periodontitis'].mean():.2%}")
    print(f"   Sample Size: {len(df)} participants\n")
    
    return df


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

