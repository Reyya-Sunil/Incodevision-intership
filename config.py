"""
Configuration file for the Data Cleaning & Visualization Project.
Contains all paths, constants, and settings used across the project.
"""

import os

# ─── Project Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DATA_DIR = os.path.join(DATA_DIR, "cleaned")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
DASHBOARD_DIR = os.path.join(OUTPUT_DIR, "dashboard")

# Create directories if they don't exist
for d in [RAW_DATA_DIR, CLEANED_DATA_DIR, PLOTS_DIR, REPORTS_DIR, DASHBOARD_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Dataset Configuration ────────────────────────────────────────
DATASET_NAME = "titanic"
RAW_CSV = os.path.join(RAW_DATA_DIR, "titanic_raw.csv")
CLEANED_CSV = os.path.join(CLEANED_DATA_DIR, "titanic_cleaned.csv")
CLEANING_REPORT = os.path.join(REPORTS_DIR, "cleaning_report.txt")

# ─── Visualization Settings ──────────────────────────────────────
FIGURE_DPI = 150
FIGURE_SIZE = (12, 8)
COLOR_PALETTE = "viridis"
SEABORN_STYLE = "whitegrid"

# ─── Column Configurations ────────────────────────────────────────
# Columns to drop (too many missing values or not useful for analysis)
COLUMNS_TO_DROP = ["Cabin", "Ticket"]

# Numerical columns for outlier detection
NUMERICAL_COLS = ["Age", "Fare", "SibSp", "Parch"]

# Categorical columns for encoding/analysis
CATEGORICAL_COLS = ["Survived", "Pclass", "Sex", "Embarked"]

# Target variable
TARGET_COL = "Survived"
