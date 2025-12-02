"""
NHANES Periodontitis Prediction Project
Step 1: Download and Prepare NHANES Data

Author: Tuminha (Francisco Teixeira Barbosa)
Goal: Replicate and improve upon Bashir et al. (2022) using gradient boosting methods

Data source: CDC NHANES - https://wwwn.cdc.gov/nchs/nhanes/
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# =============================================================================
# NHANES Data URLs - XPT format (SAS transport files)
# =============================================================================

# Define cycles we'll use
CYCLES = {
    "2009-2010": "I",
    "2011-2012": "G", 
    "2013-2014": "H",
    "2015-2016": "I",
    "2017-2018": "J",
}

# Base URL for NHANES data
BASE_URL = "https://wwwn.cdc.gov/Nchs/Nhanes"

# Files needed for each cycle (suffix changes per cycle)
# Format: (component_name, file_prefix)
COMPONENTS = {
    "demographics": "DEMO",
    "body_measures": "BMX", 
    "blood_pressure": "BPX",
    "smoking": "SMQ",
    "oral_health_questionnaire": "OHQ",
    "periodontal": "OHXPER",  # Periodontal exam data
    "glucose": "GLU",
    "triglycerides": "TRIGLY",
    "hdl": "HDL",
}

def get_nhanes_url(cycle: str, component: str) -> str:
    """
    Generate NHANES download URL for a specific cycle and component.
    
    NHANES uses suffixes like _G, _H, _I, _J for different cycles.
    """
    cycle_suffix = {
        "2009-2010": "",  # No suffix for 2009-2010
        "2011-2012": "_G",
        "2013-2014": "_H", 
        "2015-2016": "_I",
        "2017-2018": "_J",
    }
    
    suffix = cycle_suffix.get(cycle, "")
    file_prefix = COMPONENTS.get(component, component.upper())
    
    # Special handling for some components
    if component == "periodontal":
        if cycle == "2009-2010":
            return f"{BASE_URL}/2009-2010/OHXPER_F.XPT"
        else:
            return f"{BASE_URL}/{cycle}/OHXPER{suffix}.XPT"
    
    if cycle == "2009-2010":
        return f"{BASE_URL}/{cycle}/{file_prefix}_F.XPT"
    
    return f"{BASE_URL}/{cycle}/{file_prefix}{suffix}.XPT"


def download_nhanes_file(url: str, save_path: Path) -> pd.DataFrame:
    """
    Download NHANES XPT file and return as DataFrame.
    Uses pandas to read SAS transport format directly from URL.
    """
    try:
        print(f"  Downloading: {url}")
        df = pd.read_sas(url)
        df.to_parquet(save_path)
        print(f"  ✓ Saved to {save_path} ({len(df)} rows)")
        return df
    except Exception as e:
        print(f"  ✗ Error downloading {url}: {e}")
        return None


def download_all_data():
    """
    Download all required NHANES components for all cycles.
    """
    cycles_to_download = ["2011-2012", "2013-2014", "2015-2016", "2017-2018"]
    
    for cycle in cycles_to_download:
        print(f"\n{'='*60}")
        print(f"Downloading NHANES {cycle}")
        print('='*60)
        
        cycle_dir = RAW_DIR / cycle.replace("-", "_")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        
        for component_name, file_prefix in COMPONENTS.items():
            save_path = cycle_dir / f"{component_name}.parquet"
            
            if save_path.exists():
                print(f"  {component_name}: Already downloaded, skipping...")
                continue
                
            url = get_nhanes_url(cycle, component_name)
            download_nhanes_file(url, save_path)


# =============================================================================
# CDC/AAP Case Definition for Periodontitis
# =============================================================================

def calculate_periodontitis_status(df_perio: pd.DataFrame) -> pd.DataFrame:
    """
    Apply CDC/AAP case definitions for periodontitis.
    
    Severe periodontitis:
    - ≥2 interproximal sites with CAL ≥6 mm (not on same tooth) AND
    - ≥1 interproximal site with PD ≥5 mm
    
    Moderate periodontitis:
    - ≥2 interproximal sites with CAL ≥4 mm (not on same tooth) OR
    - ≥2 interproximal sites with PD ≥5 mm (not on same tooth)
    
    Mild periodontitis:
    - ≥2 interproximal sites with CAL ≥3 mm AND
    - ≥2 interproximal sites with PD ≥4 mm
    
    Reference: Eke et al. (2012) J Periodontol
    """
    # This function will be implemented after we examine the actual data structure
    # The periodontal exam has measurements at 6 sites per tooth
    pass


# =============================================================================
# Predictors from Bashir et al. (2022)
# =============================================================================

BASHIR_PREDICTORS = """
The 15 predictors used in Bashir et al. (2022):

DEMOGRAPHICS:
1. Age (years)
2. Sex (male/female)  
3. Education (less than high school / high school or above)

HEALTH BEHAVIORS:
4. Smoking status (never / former or current)
5. Alcohol consumption (never / former or current)

METABOLIC HEALTH:
6. Body mass index (kg/m²)
7. Waist circumference (cm)
8. Systolic blood pressure (mmHg)
9. Diastolic blood pressure (mmHg)
10. Fasting plasma glucose (mg/dL)
11. Serum triglycerides (mg/dL)
12. HDL cholesterol (mg/dL)

ORAL HEALTH:
13. Dental visit in last year (yes/no)
14. Noticed mobile teeth (yes/no)
15. Uses floss (yes/no)
"""

print(BASHIR_PREDICTORS)


# =============================================================================
# Main execution
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NHANES Periodontitis ML Project - Data Download")
    print("="*60)
    
    # Download the data
    download_all_data()
    
    print("\n" + "="*60)
    print("Next steps:")
    print("="*60)
    print("""
    1. Run 02_merge_and_clean.py to merge all components
    2. Apply CDC/AAP periodontitis case definitions
    3. Extract the 15 predictors from Bashir et al.
    4. Create train/validation/test splits by year
    5. Train models (XGBoost, CatBoost, LightGBM)
    6. Compare with Bashir's results
    """)
