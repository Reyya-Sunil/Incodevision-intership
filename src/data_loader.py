"""
data_loader.py - Load and provide initial overview of the dataset.

This module handles:
  • Loading the Titanic dataset from seaborn's built-in datasets
  • Saving a raw copy for reproducibility
  • Printing an initial summary of the dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
import sys
import os

# Add parent directory to path for config imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RAW_CSV, RAW_DATA_DIR


def load_dataset() -> pd.DataFrame:
    """
    Load the Titanic dataset.
    First tries to load from local raw CSV; if not found,
    downloads from seaborn's built-in datasets and saves locally.
    
    Returns:
        pd.DataFrame: Raw Titanic dataset.
    """
    if os.path.exists(RAW_CSV):
        print(f"✅ Loading dataset from local file: {RAW_CSV}")
        df = pd.read_csv(RAW_CSV)
    else:
        print("📥 Downloading Titanic dataset from seaborn...")
        df = sns.load_dataset("titanic")
        # Rename columns to match standard Titanic dataset naming
        df.columns = [col.title().replace("_", "") for col in df.columns]
        # Re-map to standard column names
        column_map = {
            "Survived": "Survived",
            "Pclass": "Pclass",
            "Sex": "Sex",
            "Age": "Age",
            "Sibsp": "SibSp",
            "Parch": "Parch",
            "Fare": "Fare",
            "Embarked": "Embarked",
            "Class": "Class",
            "Who": "Who",
            "Adultmale": "AdultMale",
            "Deck": "Cabin",
            "Embarktown": "EmbarkTown",
            "Alive": "Alive",
            "Alone": "Alone",
        }
        df.rename(columns=column_map, inplace=True)
        
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        df.to_csv(RAW_CSV, index=False)
        print(f"💾 Raw dataset saved to: {RAW_CSV}")
    
    return df


def dataset_overview(df: pd.DataFrame) -> dict:
    """
    Generate a comprehensive overview of the dataset.
    
    Args:
        df: Input DataFrame.
    
    Returns:
        dict: Summary statistics and metadata about the dataset.
    """
    overview = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "memory_usage_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    return overview


def print_overview(overview: dict):
    """Pretty-print the dataset overview."""
    print("\n" + "=" * 70)
    print("📊  DATASET OVERVIEW")
    print("=" * 70)
    
    rows, cols = overview["shape"]
    print(f"\n📏 Shape: {rows:,} rows × {cols} columns")
    print(f"💾 Memory Usage: {overview['memory_usage_mb']:.3f} MB")
    print(f"🔁 Duplicate Rows: {overview['duplicates']}")
    
    print(f"\n📋 Columns & Data Types:")
    print("-" * 45)
    for col, dtype in overview["dtypes"].items():
        missing = overview["missing_values"][col]
        pct = overview["missing_pct"][col]
        status = "✅" if missing == 0 else "⚠️"
        print(f"  {status} {col:<18} {str(dtype):<12} Missing: {missing:>4} ({pct:.1f}%)")
    
    print("-" * 45)
    total_missing = sum(overview["missing_values"].values())
    total_cells = rows * cols
    print(f"  📊 Total missing cells: {total_missing:,} / {total_cells:,} "
          f"({total_missing/total_cells*100:.2f}%)")
    print("=" * 70)


if __name__ == "__main__":
    df = load_dataset()
    overview = dataset_overview(df)
    print_overview(overview)
    print("\n🔍 First 5 rows:")
    print(df.head().to_string())
    print("\n📈 Statistical Summary:")
    print(df.describe().to_string())
