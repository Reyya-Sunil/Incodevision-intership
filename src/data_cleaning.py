"""
data_cleaning.py - Comprehensive Data Cleaning Pipeline.
Handles: missing values, duplicates, data types, outliers, feature engineering.
"""
import pandas as pd
import numpy as np
from scipy import stats
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (COLUMNS_TO_DROP, NUMERICAL_COLS, CLEANED_CSV, CLEANING_REPORT, REPORTS_DIR)


class DataCleaner:
    def __init__(self, df):
        self.df = df.copy()
        self.original_shape = df.shape
        self.log = []
        self._log(f"Pipeline initialized: {df.shape[0]} rows x {df.shape[1]} cols")

    def _log(self, msg):
        self.log.append(msg)
        print(msg)

    def drop_columns(self, columns=None):
        cols = columns or COLUMNS_TO_DROP
        existing = [c for c in cols if c in self.df.columns]
        if existing:
            for col in existing:
                pct = self.df[col].isnull().sum() / len(self.df) * 100
                self._log(f"  Dropping '{col}' ({pct:.1f}% missing)")
            self.df.drop(columns=existing, inplace=True)
            self._log(f"  Dropped {len(existing)} columns. Shape: {self.df.shape}")
        return self

    def remove_duplicates(self):
        n = self.df.duplicated().sum()
        if n > 0:
            self.df.drop_duplicates(inplace=True)
            self.df.reset_index(drop=True, inplace=True)
            self._log(f"  Removed {n} duplicate rows")
        else:
            self._log("  No duplicates found")
        return self

    def handle_missing_values(self):
        self._log("--- Handling Missing Values ---")
        if "Age" in self.df.columns and self.df["Age"].isnull().any():
            n = self.df["Age"].isnull().sum()
            if "Pclass" in self.df.columns and "Sex" in self.df.columns:
                self.df["Age"] = self.df.groupby(["Pclass","Sex"])["Age"].transform(lambda x: x.fillna(x.median()))
            self.df["Age"].fillna(self.df["Age"].median(), inplace=True)
            self._log(f"  Age: imputed {n} values (group median)")

        if "Embarked" in self.df.columns and self.df["Embarked"].isnull().any():
            n = self.df["Embarked"].isnull().sum()
            self.df["Embarked"].fillna(self.df["Embarked"].mode()[0], inplace=True)
            self._log(f"  Embarked: imputed {n} values (mode)")

        if "Fare" in self.df.columns and self.df["Fare"].isnull().any():
            n = self.df["Fare"].isnull().sum()
            if "Pclass" in self.df.columns:
                self.df["Fare"] = self.df.groupby("Pclass")["Fare"].transform(lambda x: x.fillna(x.median()))
            self.df["Fare"].fillna(self.df["Fare"].median(), inplace=True)
            self._log(f"  Fare: imputed {n} values (group median)")

        for col in self.df.columns[self.df.isnull().any()]:
            n = self.df[col].isnull().sum()
            pct = n / len(self.df) * 100
            if pct > 50:
                self.df.drop(columns=[col], inplace=True)
                self._log(f"  {col}: dropped ({pct:.1f}% missing)")
            elif self.df[col].dtype in ["object","category"]:
                self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                self._log(f"  {col}: imputed {n} with mode")
            else:
                self.df[col].fillna(self.df[col].median(), inplace=True)
                self._log(f"  {col}: imputed {n} with median")

        self._log(f"  Missing after cleaning: {self.df.isnull().sum().sum()}")
        return self

    def fix_data_types(self):
        self._log("--- Fixing Data Types ---")
        if "Survived" in self.df.columns:
            self.df["Survived"] = self.df["Survived"].astype(int)
        if "Pclass" in self.df.columns:
            self.df["Pclass"] = self.df["Pclass"].astype(int)
        for col in ["Sex","Embarked","EmbarkTown","Who","Class","Alive"]:
            if col in self.df.columns and self.df[col].dtype == "object":
                self.df[col] = self.df[col].astype("category")
        for col in ["AdultMale","Alone"]:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(bool)
        if "Age" in self.df.columns:
            self.df["Age"] = self.df["Age"].round(0).astype(int)
        self._log(f"  Data types fixed. Memory: {self.df.memory_usage(deep=True).sum()/1024**2:.3f} MB")
        return self

    def handle_outliers(self, method="iqr", threshold=1.5):
        self._log(f"--- Handling Outliers ({method.upper()}) ---")
        num_cols = [c for c in NUMERICAL_COLS if c in self.df.columns]
        total = 0
        for col in num_cols:
            Q1, Q3 = self.df[col].quantile(0.25), self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - threshold*IQR, Q3 + threshold*IQR
            n = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if n > 0:
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                self._log(f"  {col}: {n} outliers capped to [{lower:.2f}, {upper:.2f}]")
            total += n
        self._log(f"  Total outliers handled: {total}")
        return self

    def feature_engineering(self):
        self._log("--- Feature Engineering ---")
        if "Age" in self.df.columns:
            self.df["AgeGroup"] = pd.cut(self.df["Age"], bins=[0,12,18,35,55,100],
                                         labels=["Child","Teen","YoungAdult","Adult","Senior"])
            self._log("  Created AgeGroup")
        if "SibSp" in self.df.columns and "Parch" in self.df.columns:
            self.df["FamilySize"] = self.df["SibSp"] + self.df["Parch"] + 1
            self.df["IsAlone"] = (self.df["FamilySize"] == 1).astype(int)
            self._log(f"  Created FamilySize & IsAlone")
        if "Fare" in self.df.columns and "FamilySize" in self.df.columns:
            self.df["FarePerPerson"] = (self.df["Fare"] / self.df["FamilySize"]).round(2)
            self._log("  Created FarePerPerson")
        if "Fare" in self.df.columns:
            self.df["FareCategory"] = pd.qcut(self.df["Fare"], q=4,
                                              labels=["Budget","Economy","Business","Premium"])
            self._log("  Created FareCategory")
        self._log(f"  Total columns: {len(self.df.columns)}")
        return self

    def generate_report(self):
        self._log(f"\n{'='*50}\nCLEANING REPORT\n{'='*50}")
        self._log(f"  Original: {self.original_shape}")
        self._log(f"  Cleaned:  {self.df.shape}")
        self._log(f"  Missing:  {self.df.isnull().sum().sum()}")
        self._log(f"  Dupes:    {self.df.duplicated().sum()}")
        os.makedirs(REPORTS_DIR, exist_ok=True)
        with open(CLEANING_REPORT, "w", encoding="utf-8") as f:
            f.write("\n".join(self.log))
        self._log(f"  Report saved: {CLEANING_REPORT}")
        return self

    def save(self, filepath=None):
        path = filepath or CLEANED_CSV
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.df.to_csv(path, index=False)
        self._log(f"  Saved: {path}")
        return self

    def get_dataframe(self):
        return self.df

    def run_pipeline(self):
        return (self.drop_columns().remove_duplicates().handle_missing_values()
                .fix_data_types().handle_outliers().feature_engineering()
                .generate_report().save())


if __name__ == "__main__":
    from data_loader import load_dataset
    df = load_dataset()
    cleaner = DataCleaner(df)
    cleaner.run_pipeline()
    print(cleaner.get_dataframe().head(10).to_string())
