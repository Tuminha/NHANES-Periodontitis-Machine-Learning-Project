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
    
    """
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_git_hash() -> Optional[str]:
    """
    Get current git commit hash for reproducibility tracking.
    
    Returns:
        Git commit hash string, or None if not in a git repo
    
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def log_versions(output_path: str = "results/system_info.txt") -> None:
    """
    Log package versions, hardware info, and git hash for reproducibility.
    
    Args:
        output_path: Path to save system info text file
    
    """
    import importlib
    import platform
    import sys

    packages = [
        "pandas",
        "numpy",
        "sklearn",
        "xgboost",
        "catboost",
        "lightgbm",
        "optuna",
        "shap",
        "scipy",
        "statsmodels",
    ]
    versions = {}
    for package in packages:
        try:
            module = importlib.import_module(package)
            versions[package] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[package] = "not installed"

    info = {
        "timestamp": datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "git_hash": get_git_hash(),
        "packages": versions,
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(info, indent=2), encoding="utf-8")


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
    
    """
    def convert_types(value):
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, pd.Series):
            return value.tolist()
        if isinstance(value, pd.DataFrame):
            return value.to_dict(orient="records")
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    output = Path(filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=convert_types)


def load_json(filepath: str) -> Any:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
    
    Returns:
        Loaded object
    
    """
    with Path(filepath).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(filepath: str) -> Dict:
    """
    Load YAML configuration file.
    
    Args:
        filepath: Path to YAML file
    
    Returns:
        Dict containing configuration
    
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_pickle(obj: Any, filepath: str) -> None:
    """
    Save Python object as pickle.
    
    Args:
        obj: Object to serialize
        filepath: Output path
    
    """
    output = Path(filepath)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath: str) -> Any:
    """
    Load pickle file.
    
    Args:
        filepath: Path to pickle file
    
    Returns:
        Loaded object
    
    """
    with Path(filepath).open("rb") as f:
        return pickle.load(f)


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
    
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{model_name}_{timestamp}.pkl" if include_timestamp else f"{model_name}.pkl"
    filepath = Path(output_dir) / filename
    save_pickle(model, filepath)

    metadata = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "saved_at": datetime.now().isoformat(),
        "git_hash": get_git_hash(),
        "filepath": str(filepath),
    }
    save_json(metadata, str(filepath.with_suffix(".metadata.json")))
    return str(filepath)


def create_model_hash(model_params: Dict) -> str:
    """
    Create short hash of model parameters for versioning.
    
    Args:
        model_params: Dict of model hyperparameters
    
    Returns:
        Short hash string (first 8 chars of MD5)
    
    """
    param_str = json.dumps(model_params, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode("utf-8")).hexdigest()[:8]


# =============================================================================
# Data Processing Helpers
# =============================================================================

def print_missing_summary(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Print summary of missing data in DataFrame.
    
    Args:
        df: Pandas DataFrame
        top_n: Show top N columns by missing percentage
    
    """
    missing = df.isnull().sum()
    missing_pct = 100 * missing / max(len(df), 1)
    missing_df = pd.DataFrame({"count": missing, "percent": missing_pct})
    missing_df = missing_df[missing_df["count"] > 0].sort_values(
        "percent", ascending=False
    )
    print(f"\nMissing Data Summary (top {top_n}):")
    print(missing_df.head(top_n).round(2))


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
    
    """
    rows = []
    for feature in features:
        if feature not in df_train.columns or feature not in df_test.columns:
            rows.append({"feature": feature, "status": "missing"})
            continue

        train = pd.to_numeric(df_train[feature], errors="coerce")
        test = pd.to_numeric(df_test[feature], errors="coerce")
        train_std = train.std()
        pooled_std = np.sqrt((train.var() + test.var()) / 2)
        std_diff = (
            (test.mean() - train.mean()) / pooled_std
            if pooled_std and not np.isnan(pooled_std)
            else np.nan
        )
        rows.append(
            {
                "feature": feature,
                "status": "ok",
                "train_mean": float(train.mean()),
                "test_mean": float(test.mean()),
                "train_std": float(train_std),
                "test_std": float(test.std()),
                "standardized_difference": float(std_diff),
                "large_drift": bool(abs(std_diff) >= 0.1) if not np.isnan(std_diff) else False,
            }
        )

    drift = pd.DataFrame(rows)
    if save_path:
        output = Path(save_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        drift.to_csv(output, index=False)
    return drift


# =============================================================================
# Progress and Logging
# =============================================================================

def log_step(step_name: str, log_file: str = "logs/pipeline.log") -> None:
    """
    Log a pipeline step with timestamp.
    
    Args:
        step_name: Description of step
        log_file: Path to log file
    
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"[{timestamp}] {step_name}"
    print(message)
    output = Path(log_file)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as f:
        f.write(message + "\n")
