"""
General Utility Functions
Author: Francisco Teixeira Barbosa (Cisco)

Purpose: Helper functions for logging, saving/loading, reproducibility,
         and data processing.

Usage:
    from src.utils import set_seed, save_json, log_versions
    
    set_seed(42)
    save_json({"auc": 0.85}, "results/metrics.json")
    log_versions("results/system_info.txt")
"""

import os
import json
import yaml
import pickle
import hashlib
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, and ML libraries.
    
    Args:
        seed: Random seed value
    
    TODO: Set seed for numpy, random, and any other libraries used
    """
    # TODO: np.random.seed(seed)
    # TODO: import random; random.seed(seed)
    # TODO: If using scikit-learn, it respects np random state
    # TODO: If using torch: torch.manual_seed(seed)
    # TODO: print(f"Random seed set to {seed}")
    pass


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash for reproducibility tracking.
    
    Returns:
        Git commit hash string, or None if not in a git repo
    
    TODO: Use subprocess to run 'git rev-parse HEAD'
    TODO: Handle case where git is not available or not a repo
    """
    # TODO: try:
    #           result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
    #                                   capture_output=True, text=True, check=True)
    #           return result.stdout.strip()
    # TODO: except Exception:
    #           return None
    pass


def log_versions(output_path: str = "results/system_info.txt") -> None:
    """
    Log package versions, hardware info, and git hash for reproducibility.
    
    Args:
        output_path: Path to save system info text file
    
    TODO: Collect versions of key packages (pandas, sklearn, xgboost, etc.)
    TODO: Get hardware info (CPU, RAM if possible)
    TODO: Get git hash
    TODO: Get timestamp
    TODO: Write all to output_path
    """
    # TODO: import platform, sys
    # TODO: info = {
    #           "timestamp": datetime.now().isoformat(),
    #           "python_version": sys.version,
    #           "platform": platform.platform(),
    #           "git_hash": get_git_hash(),
    #       }
    # TODO: Get package versions: pd.__version__, sklearn.__version__, etc.
    # TODO: Write to file
    # TODO: print(f"System info logged to {output_path}")
    pass


# =============================================================================
# File I/O
# =============================================================================

def save_json(obj: Any, filepath: str, indent: int = 2) -> None:
    """
    Save Python object as JSON.
    
    Args:
        obj: Object to serialize (dict, list, etc.)
        filepath: Output path
        indent: JSON indentation for readability
    
    TODO: Create parent directory if missing
    TODO: Convert numpy types to native Python types
    TODO: Write JSON
    """
    # TODO: Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # TODO: def convert_types(o):
    #           if isinstance(o, np.integer): return int(o)
    #           elif isinstance(o, np.floating): return float(o)
    #           elif isinstance(o, np.ndarray): return o.tolist()
    #           raise TypeError
    # TODO: with open(filepath, 'w') as f:
    #           json.dump(obj, f, indent=indent, default=convert_types)
    # TODO: print(f"JSON saved to {filepath}")
    pass


def load_json(filepath: str) -> Any:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded object
    
    TODO: Open and parse JSON
    """
    # TODO: with open(filepath, 'r') as f:
    #           return json.load(f)
    pass


def load_yaml(filepath: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Dict containing configuration
    
    TODO: Open and parse YAML
    TODO: Handle FileNotFoundError with helpful message
    """
    # TODO: with open(filepath, 'r') as f:
    #           return yaml.safe_load(f)
    pass


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save Python object as pickle.
    
    Args:
        obj: Object to serialize
        filepath: Output path
    
    TODO: Create parent directory
    TODO: Save with pickle
    """
    # TODO: Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    # TODO: with open(filepath, 'wb') as f:
    #           pickle.dump(obj, f)
    # TODO: print(f"Pickle saved to {filepath}")
    pass


def load_pickle(filepath: str) -> Any:
    """
    Load pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded object
    
    TODO: Open and unpickle
    """
    # TODO: with open(filepath, 'rb') as f:
    #           return pickle.load(f)
    pass


# =============================================================================
# Model Management
# =============================================================================

def save_model(
    model,
    model_name: str,
    output_dir: str = "models",
    include_timestamp: bool = True
) -> str:
    """
    Save trained model with versioned filename.
    
    Args:
        model: Trained sklearn-compatible model
        model_name: Base name (e.g., "xgboost_best")
        output_dir: Directory to save models
        include_timestamp: If True, append timestamp to filename
    
    Returns:
        filepath: Full path where model was saved
    
    TODO: Generate versioned filename with timestamp and short hash
    TODO: Save model as pickle
    TODO: Also save metadata JSON (timestamp, git hash, model type)
    """
    # TODO: timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # TODO: if include_timestamp:
    #           filename = f"{model_name}_{timestamp}.pkl"
    # TODO: else:
    #           filename = f"{model_name}.pkl"
    # TODO: filepath = Path(output_dir) / filename
    # TODO: save_pickle(model, filepath)
    # TODO: Save metadata
    # TODO: return str(filepath)
    pass


def create_model_hash(model_params: Dict) -> str:
    """
    Create short hash of model parameters for versioning.
    
    Args:
        model_params: Dict of model hyperparameters
    
    Returns:
        Short hash string (first 8 chars of MD5)
    
    TODO: Serialize params to string
    TODO: Compute MD5 hash
    TODO: Return first 8 characters
    """
    # TODO: param_str = json.dumps(model_params, sort_keys=True)
    # TODO: hash_obj = hashlib.md5(param_str.encode())
    # TODO: return hash_obj.hexdigest()[:8]
    pass


# =============================================================================
# Data Processing Helpers
# =============================================================================

def print_missing_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Print summary of missing data in DataFrame.
    
    Args:
        df: Pandas DataFrame
        top_n: Show top N columns by missing percentage
    
    TODO: Compute missing count and percentage per column
    TODO: Sort by percentage descending
    TODO: Print formatted table
    """
    # TODO: missing = df.isnull().sum()
    # TODO: missing_pct = 100 * missing / len(df)
    # TODO: missing_df = pd.DataFrame({'count': missing, 'percent': missing_pct})
    # TODO: missing_df = missing_df[missing_df['count'] > 0].sort_values('percent', ascending=False)
    # TODO: print(f"\nMissing Data Summary (top {top_n}):")
    # TODO: print(missing_df.head(top_n))
    pass


def check_data_drift(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    features: list,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Check for feature distribution drift between train and test sets.
    
    Args:
        df_train: Training DataFrame
        df_test: Test DataFrame
        features: List of feature names to check
        save_path: Optional path to save drift report
    
    Returns:
        DataFrame with drift statistics per feature
    
    TODO: For each feature, compute mean/std in train vs test
    TODO: Compute standardized difference
    TODO: Flag features with large drift
    TODO: Optionally create visualization
    """
    # TODO: drift_report = []
    # TODO: for feat in features:
    #           mean_train = df_train[feat].mean()
    #           mean_test = df_test[feat].mean()
    #           std_train = df_train[feat].std()
    #           # Standardized difference = (mean_test - mean_train) / std_train
    #           drift_report.append({...})
    # TODO: df_drift = pd.DataFrame(drift_report)
    # TODO: if save_path: df_drift.to_csv(save_path)
    # TODO: return df_drift
    pass


# =============================================================================
# Progress and Logging
# =============================================================================

def log_step(step_name: str, log_file: str = "logs/pipeline.log") -> None:
    """
    Log a pipeline step with timestamp.
    
    Args:
        step_name: Description of step
        log_file: Path to log file
    
    TODO: Append timestamped message to log file
    TODO: Also print to console
    """
    # TODO: timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # TODO: message = f"[{timestamp}] {step_name}"
    # TODO: print(message)
    # TODO: Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    # TODO: with open(log_file, 'a') as f:
    #           f.write(message + '\n')
    pass

