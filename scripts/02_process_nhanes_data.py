"""
NHANES Periodontitis Prediction Project
Step 2: Merge and Process NHANES Data

This script:
1. Loads downloaded NHANES components
2. Merges them by SEQN (participant ID)
3. Creates periodontitis labels using CDC/AAP definitions
4. Extracts the 15 predictors from Bashir et al.
5. Creates temporal train/val/test splits
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# =============================================================================
# Variable Mappings - NHANES variable names to readable names
# =============================================================================

# Demographics (DEMO)
DEMO_VARS = {
    'SEQN': 'participant_id',
    'RIDAGEYR': 'age',
    'RIAGENDR': 'sex',  # 1=Male, 2=Female
    'DMDEDUC2': 'education',  # Education level - adults 20+
    'RIDRETH3': 'race_ethnicity',
}

# Body Measures (BMX)
BMX_VARS = {
    'SEQN': 'participant_id',
    'BMXBMI': 'bmi',
    'BMXWAIST': 'waist_circumference',
}

# Blood Pressure (BPX)
BPX_VARS = {
    'SEQN': 'participant_id',
    'BPXSY1': 'systolic_bp_1',
    'BPXSY2': 'systolic_bp_2', 
    'BPXSY3': 'systolic_bp_3',
    'BPXDI1': 'diastolic_bp_1',
    'BPXDI2': 'diastolic_bp_2',
    'BPXDI3': 'diastolic_bp_3',
}

# Smoking (SMQ)
SMQ_VARS = {
    'SEQN': 'participant_id',
    'SMQ020': 'smoked_100_cigs',  # 1=Yes, 2=No (lifetime)
    'SMQ040': 'smoking_now',  # 1=Every day, 2=Some days, 3=Not at all
}

# Oral Health Questionnaire (OHQ)
OHQ_VARS = {
    'SEQN': 'participant_id',
    'OHQ030': 'time_since_dental_visit',  # 1=6mo, 2=1yr, 3=2yr, etc.
    'OHQ620': 'floss_days_per_week',
    'OHQ845': 'loose_teeth',  # 1=Yes, 2=No
}

# =============================================================================
# CDC/AAP Periodontitis Case Definitions
# =============================================================================

def calculate_cal(pocket_depth: float, recession: float) -> float:
    """
    Calculate Clinical Attachment Loss (CAL).
    CAL = Pocket Depth + Recession (if recession is positive/gingival margin below CEJ)
    
    Note: In NHANES, recession is measured as distance from gingival margin to CEJ.
    Positive values = recession, Negative values = gingival overgrowth
    """
    if pd.isna(pocket_depth) or pd.isna(recession):
        return np.nan
    return pocket_depth + max(0, recession)


def apply_cdc_aap_definition(df_perio: pd.DataFrame) -> pd.DataFrame:
    """
    Apply CDC/AAP case definitions for periodontitis surveillance.
    
    From Eke et al. (2012) J Periodontol:
    
    Severe periodontitis:
    - ≥2 interproximal sites with CAL ≥6 mm (not on same tooth) AND
    - ≥1 interproximal site with PD ≥5 mm
    
    Moderate periodontitis:
    - ≥2 interproximal sites with CAL ≥4 mm (not on same tooth) OR
    - ≥2 interproximal sites with PD ≥5 mm (not on same tooth)
    
    Mild periodontitis:
    - ≥2 interproximal sites with CAL ≥3 mm AND
    - ≥2 interproximal sites with PD ≥4 mm
    
    Returns DataFrame with columns:
    - periodontitis_binary: 0=No, 1=Yes (any level)
    - periodontitis_severity: 0=None, 1=Mild, 2=Moderate, 3=Severe
    """
    
    # NHANES periodontal exam variables follow pattern:
    # OHXxxPCM = Pocket depth, site xx, mesial
    # OHXxxPCD = Pocket depth, site xx, distal  
    # OHXxxPCS = Pocket depth, site xx, mid-buccal/mid-lingual
    # OHXxxLAM = Loss of attachment, site xx, mesial
    # OHXxxLAD = Loss of attachment, site xx, distal
    # OHXxxLAS = Loss of attachment, site xx, mid-site
    
    # We need interproximal sites (mesial and distal), not mid-sites
    
    results = []
    
    for idx, row in df_perio.iterrows():
        seqn = row['SEQN']
        
        # Collect all interproximal measurements
        # Teeth are numbered 02-15 (upper) and 18-31 (lower), excluding third molars
        interprox_cal = []
        interprox_pd = []
        teeth_with_cal_6 = set()
        teeth_with_cal_4 = set()
        teeth_with_cal_3 = set()
        teeth_with_pd_5 = set()
        teeth_with_pd_4 = set()
        
        for tooth_num in list(range(2, 16)) + list(range(18, 32)):
            tooth_str = f"{tooth_num:02d}"
            
            # Mesial and Distal sites (interproximal)
            for site in ['M', 'D']:  # Mesial, Distal
                pd_col = f'OHX{tooth_str}PC{site}'
                cal_col = f'OHX{tooth_str}LA{site}'
                
                if pd_col in row.index and cal_col in row.index:
                    pd_val = row[pd_col]
                    cal_val = row[cal_col]
                    
                    if not pd.isna(pd_val):
                        interprox_pd.append((tooth_num, site, pd_val))
                        if pd_val >= 5:
                            teeth_with_pd_5.add(tooth_num)
                        if pd_val >= 4:
                            teeth_with_pd_4.add(tooth_num)
                    
                    if not pd.isna(cal_val):
                        interprox_cal.append((tooth_num, site, cal_val))
                        if cal_val >= 6:
                            teeth_with_cal_6.add(tooth_num)
                        if cal_val >= 4:
                            teeth_with_cal_4.add(tooth_num)
                        if cal_val >= 3:
                            teeth_with_cal_3.add(tooth_num)
        
        # Apply case definitions
        severe = (len(teeth_with_cal_6) >= 2) and (len(teeth_with_pd_5) >= 1)
        
        moderate = (len(teeth_with_cal_4) >= 2) or (len(teeth_with_pd_5) >= 2)
        
        mild = (len(teeth_with_cal_3) >= 2) and (len(teeth_with_pd_4) >= 2)
        
        # Determine severity (hierarchical)
        if severe:
            severity = 3
        elif moderate:
            severity = 2
        elif mild:
            severity = 1
        else:
            severity = 0
        
        results.append({
            'participant_id': seqn,
            'periodontitis_severity': severity,
            'periodontitis_binary': 1 if severity > 0 else 0,
            'n_interprox_sites': len(interprox_cal),
            'n_teeth_cal_6': len(teeth_with_cal_6),
            'n_teeth_cal_4': len(teeth_with_cal_4),
            'n_teeth_pd_5': len(teeth_with_pd_5),
        })
    
    return pd.DataFrame(results)


# =============================================================================
# Feature Engineering for Bashir's 15 Predictors
# =============================================================================

def create_bashir_predictors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create the 15 predictors used in Bashir et al. (2022).
    """
    features = pd.DataFrame()
    features['participant_id'] = df['participant_id']
    
    # 1. Age (already numeric)
    features['age'] = df['age']
    
    # 2. Sex (binary: 1=Male, 0=Female)
    features['sex_male'] = (df['sex'] == 1).astype(int)
    
    # 3. Education (binary: 1=High school or above, 0=Less than high school)
    # DMDEDUC2: 1=Less than 9th, 2=9-11th, 3=HS grad, 4=Some college, 5=College+
    features['education_hs_plus'] = (df['education'] >= 3).astype(int)
    
    # 4. Smoking status (binary: 1=Former or current, 0=Never)
    # SMQ020: 1=Yes smoked 100+ cigs, 2=No
    features['ever_smoker'] = (df['smoked_100_cigs'] == 1).astype(int)
    
    # 5. Alcohol consumption (need to add - check ALQ component)
    # Placeholder - will need to download ALQ component
    features['ever_drinker'] = np.nan
    
    # 6. BMI
    features['bmi'] = df['bmi']
    
    # 7. Waist circumference
    features['waist_circumference'] = df['waist_circumference']
    
    # 8-9. Blood pressure (average of multiple readings)
    bp_sys_cols = ['systolic_bp_1', 'systolic_bp_2', 'systolic_bp_3']
    bp_dia_cols = ['diastolic_bp_1', 'diastolic_bp_2', 'diastolic_bp_3']
    
    available_sys = [c for c in bp_sys_cols if c in df.columns]
    available_dia = [c for c in bp_dia_cols if c in df.columns]
    
    if available_sys:
        features['systolic_bp'] = df[available_sys].mean(axis=1)
    if available_dia:
        features['diastolic_bp'] = df[available_dia].mean(axis=1)
    
    # 10-12. Metabolic markers (glucose, triglycerides, HDL)
    if 'fasting_glucose' in df.columns:
        features['fasting_glucose'] = df['fasting_glucose']
    if 'triglycerides' in df.columns:
        features['triglycerides'] = df['triglycerides']
    if 'hdl' in df.columns:
        features['hdl'] = df['hdl']
    
    # 13. Dental visit in last year (binary)
    # OHQ030: 1=6mo or less, 2=More than 6mo but not more than 1yr, ...
    if 'time_since_dental_visit' in df.columns:
        features['dental_visit_last_year'] = (df['time_since_dental_visit'] <= 2).astype(int)
    
    # 14. Mobile teeth (binary)
    if 'loose_teeth' in df.columns:
        features['has_loose_teeth'] = (df['loose_teeth'] == 1).astype(int)
    
    # 15. Uses floss (binary)
    if 'floss_days_per_week' in df.columns:
        features['uses_floss'] = (df['floss_days_per_week'] > 0).astype(int)
    
    return features


# =============================================================================
# Main Processing Pipeline
# =============================================================================

def process_cycle(cycle: str) -> pd.DataFrame:
    """
    Process a single NHANES cycle: load, merge, and create features.
    """
    print(f"\nProcessing {cycle}...")
    cycle_dir = RAW_DIR / cycle.replace("-", "_")
    
    # Load all components
    dfs = {}
    for component in ['demographics', 'body_measures', 'blood_pressure', 
                      'smoking', 'oral_health_questionnaire', 'periodontal']:
        fpath = cycle_dir / f"{component}.parquet"
        if fpath.exists():
            dfs[component] = pd.read_parquet(fpath)
            print(f"  Loaded {component}: {len(dfs[component])} rows")
    
    if 'demographics' not in dfs:
        print(f"  ✗ Missing demographics for {cycle}")
        return None
    
    # Start with demographics
    df = dfs['demographics'][['SEQN', 'RIDAGEYR', 'RIAGENDR', 'DMDEDUC2']].copy()
    df.columns = ['participant_id', 'age', 'sex', 'education']
    
    # Filter to adults 30+ (standard for periodontitis surveillance)
    df = df[df['age'] >= 30].copy()
    print(f"  Adults 30+: {len(df)}")
    
    # Merge other components
    if 'body_measures' in dfs:
        bm = dfs['body_measures'][['SEQN', 'BMXBMI', 'BMXWAIST']].copy()
        bm.columns = ['participant_id', 'bmi', 'waist_circumference']
        df = df.merge(bm, on='participant_id', how='left')
    
    if 'blood_pressure' in dfs:
        bp_cols = ['SEQN'] + [c for c in dfs['blood_pressure'].columns 
                              if c.startswith('BPXSY') or c.startswith('BPXDI')]
        bp = dfs['blood_pressure'][bp_cols].copy()
        bp.columns = ['participant_id'] + [c.replace('BPXSY', 'systolic_bp_').replace('BPXDI', 'diastolic_bp_') 
                                           for c in bp_cols[1:]]
        df = df.merge(bp, on='participant_id', how='left')
    
    if 'smoking' in dfs:
        sm = dfs['smoking'][['SEQN', 'SMQ020']].copy()
        sm.columns = ['participant_id', 'smoked_100_cigs']
        df = df.merge(sm, on='participant_id', how='left')
    
    if 'oral_health_questionnaire' in dfs:
        ohq_cols = ['SEQN']
        for c in ['OHQ030', 'OHQ620', 'OHQ845']:
            if c in dfs['oral_health_questionnaire'].columns:
                ohq_cols.append(c)
        ohq = dfs['oral_health_questionnaire'][ohq_cols].copy()
        col_map = {'SEQN': 'participant_id', 'OHQ030': 'time_since_dental_visit',
                   'OHQ620': 'floss_days_per_week', 'OHQ845': 'loose_teeth'}
        ohq.columns = [col_map.get(c, c) for c in ohq.columns]
        df = df.merge(ohq, on='participant_id', how='left')
    
    # Add cycle identifier
    df['cycle'] = cycle
    
    print(f"  Final merged dataset: {len(df)} rows, {len(df.columns)} columns")
    
    return df


def main():
    """
    Main processing pipeline.
    """
    print("="*60)
    print("NHANES Periodontitis ML Project - Data Processing")
    print("="*60)
    
    # Process each cycle
    cycles = ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]
    all_data = []
    
    for cycle in cycles:
        df = process_cycle(cycle)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        # Combine all cycles
        df_combined = pd.concat(all_data, ignore_index=True)
        print(f"\n{'='*60}")
        print(f"Combined dataset: {len(df_combined)} participants")
        print(f"Cycles: {df_combined['cycle'].value_counts().to_dict()}")
        
        # Save
        output_path = PROCESSED_DIR / "nhanes_combined.parquet"
        df_combined.to_parquet(output_path)
        print(f"\nSaved to {output_path}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("Data Summary:")
        print("="*60)
        print(df_combined.describe())
    
    return df_combined


if __name__ == "__main__":
    main()
