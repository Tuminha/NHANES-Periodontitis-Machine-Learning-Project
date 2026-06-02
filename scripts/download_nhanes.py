"""
Simple NHANES Data Downloader
Downloads the exact files needed for periodontitis prediction
"""

import pandas as pd
from pathlib import Path

# Create directories
DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# NHANES file URLs from the current CDC public data-file endpoint.

BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public"
CYCLES = {
    "2009-2010": ("2009", "F"),
    "2011-2012": ("2011", "G"),
    "2013-2014": ("2013", "H"),
}
COMPONENTS = {
    "demographics": "DEMO",
    "body_measures": "BMX",
    "blood_pressure": "BPX",
    "smoking": "SMQ",
    "alcohol": "ALQ",
    "periodontal": "OHXPER",
    "oral_health_questionnaire": "OHQ",
    "glucose": "GLU",
    "triglycerides": "TRIGLY",
    "hdl": "HDL",
}


def nhanes_url(cycle: str, prefix: str) -> str:
    start_year, suffix = CYCLES[cycle]
    return f"{BASE_URL}/{start_year}/DataFiles/{prefix}_{suffix}.xpt"


NHANES_FILES = {
    cycle: {component: nhanes_url(cycle, prefix) for component, prefix in COMPONENTS.items()}
    for cycle in CYCLES
}


def download_nhanes_data(cycles=None, components=None):
    """
    Download NHANES data files.
    
    Args:
        cycles: List of cycles to download, e.g., ["2011-2012", "2013-2014"]
                If None, downloads all cycles.
        components: List of components to download, e.g., ["demographics", "periodontal"]
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
                print(f"  Warning: {component} not available for {cycle}")
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
        components=["demographics", "periodontal", "body_measures",
                   "blood_pressure", "smoking", "oral_health_questionnaire"]
    )
    
    summarize_data(data)
    
    print("\n" + "="*60)
    print("Next: Run this again with all cycles for full dataset")
    print("="*60)
