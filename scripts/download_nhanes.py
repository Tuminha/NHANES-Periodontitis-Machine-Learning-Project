"""
Simple NHANES Data Downloader
Downloads the exact files needed for periodontitis prediction
"""

import pandas as pd
from pathlib import Path

# Create directories
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# NHANES file URLs - these are the EXACT URLs from CDC
# Format: https://wwwn.cdc.gov/Nchs/Nhanes/{CYCLE}/{FILENAME}.XPT

NHANES_FILES = {
    "2011-2012": {
        "demographics": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT",
        "body_measures": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BMX_G.XPT",
        "blood_pressure": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/BPX_G.XPT",
        "smoking": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/SMQ_G.XPT",
        "alcohol": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/ALQ_G.XPT",
        "oral_health_exam": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHXPER_G.XPT",
        "oral_health_questionnaire": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHQ_G.XPT",
        "glucose": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/GLU_G.XPT",
        "triglycerides": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/TRIGLY_G.XPT",
        "hdl": "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/HDL_G.XPT",
    },
    "2013-2014": {
        "demographics": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT",
        "body_measures": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BMX_H.XPT",
        "blood_pressure": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/BPX_H.XPT",
        "smoking": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/SMQ_H.XPT",
        "alcohol": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/ALQ_H.XPT",
        "oral_health_exam": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHXPER_H.XPT",
        "oral_health_questionnaire": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHQ_H.XPT",
        "glucose": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/GLU_H.XPT",
        "triglycerides": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/TRIGLY_H.XPT",
        "hdl": "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/HDL_H.XPT",
    },
    "2015-2016": {
        "demographics": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT",
        "body_measures": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BMX_I.XPT",
        "blood_pressure": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/BPX_I.XPT",
        "smoking": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/SMQ_I.XPT",
        "alcohol": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/ALQ_I.XPT",
        "oral_health_exam": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHXPER_I.XPT",
        "oral_health_questionnaire": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHQ_I.XPT",
        "glucose": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/GLU_I.XPT",
        "triglycerides": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/TRIGLY_I.XPT",
        "hdl": "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/HDL_I.XPT",
    },
    "2017-2018": {
        "demographics": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT",
        "body_measures": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BMX_J.XPT",
        "blood_pressure": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/BPX_J.XPT",
        "smoking": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/SMQ_J.XPT",
        "alcohol": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/ALQ_J.XPT",
        "oral_health_exam": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXPER_J.XPT",
        "oral_health_questionnaire": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHQ_J.XPT",
        "glucose": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/GLU_J.XPT",
        "triglycerides": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/TRIGLY_J.XPT",
        "hdl": "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/HDL_J.XPT",
    },
}


def download_nhanes_data(cycles=None, components=None):
    """
    Download NHANES data files.
    
    Args:
        cycles: List of cycles to download, e.g., ["2011-2012", "2013-2014"]
                If None, downloads all cycles.
        components: List of components to download, e.g., ["demographics", "oral_health_exam"]
                   If None, downloads all components.
    """
    if cycles is None:
        cycles = list(NHANES_FILES.keys())
    
    all_data = {}
    
    for cycle in cycles:
        print(f"\n{'='*60}")
        print(f"Downloading NHANES {cycle}")
        print('='*60)
        
        cycle_dir = DATA_DIR / cycle.replace("-", "_")
        cycle_dir.mkdir(exist_ok=True, parents=True)
        
        all_data[cycle] = {}
        
        if components is None:
            components_to_download = NHANES_FILES[cycle].keys()
        else:
            components_to_download = components
        
        for component in components_to_download:
            if component not in NHANES_FILES[cycle]:
                print(f"  ⚠ {component} not available for {cycle}")
                continue
                
            url = NHANES_FILES[cycle][component]
            save_path = cycle_dir / f"{component}.parquet"
            
            if save_path.exists():
                print(f"  ✓ {component}: Already downloaded")
                df = pd.read_parquet(save_path)
            else:
                print(f"  ↓ {component}: Downloading from {url}...")
                try:
                    df = pd.read_sas(url)
                    df.to_parquet(save_path)
                    print(f"    ✓ Saved {len(df)} rows")
                except Exception as e:
                    print(f"    ✗ Error: {e}")
                    continue
            
            all_data[cycle][component] = df
    
    return all_data


def summarize_data(all_data):
    """Print summary of downloaded data."""
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for cycle, components in all_data.items():
        print(f"\n{cycle}:")
        for comp, df in components.items():
            print(f"  {comp}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    # Download just 2011-2012 first to test
    print("Starting NHANES download...")
    print("This will download data directly from CDC servers.")
    
    # Download 2011-2012 first (the cycle Bashir used)
    data = download_nhanes_data(
        cycles=["2011-2012"],
        components=["demographics", "oral_health_exam", "body_measures", 
                   "blood_pressure", "smoking", "oral_health_questionnaire"]
    )
    
    summarize_data(data)
    
    print("\n" + "="*60)
    print("Next: Run this again with all cycles for full dataset")
    print("="*60)
