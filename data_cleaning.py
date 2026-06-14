"""
Step 1: Data Cleaning
- Load the Titanic dataset
- Handle missing values
- Remove duplicates
- Fix data types
- Handle outliers
- Save the cleaned dataset
"""

import pandas as pd
import numpy as np
import seaborn as sns
import os

# ── Load Dataset ─────────────────────────────────────────────────
print("=" * 50)
print("  STEP 1: LOADING THE DATASET")
print("=" * 50)

df = sns.load_dataset("titanic")
print(f"\nDataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# ── Initial Data Overview ────────────────────────────────────────
print("\n" + "=" * 50)
print("  DATA OVERVIEW (Before Cleaning)")
print("=" * 50)

print(f"\nData Types:\n{df.dtypes}")
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")
print(f"\nBasic Stats:\n{df.describe()}")

# ── Handle Missing Values ────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 2: HANDLING MISSING VALUES")
print("=" * 50)

# Age → fill with median
print(f"\nAge: {df['age'].isnull().sum()} missing → filling with median")
df['age'] = df['age'].fillna(df['age'].median())

# Embarked → fill with mode (most common value)
print(f"Embarked: {df['embarked'].isnull().sum()} missing → filling with mode")
df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])

# Embark_town → fill with mode
print(f"Embark_town: {df['embark_town'].isnull().sum()} missing → filling with mode")
df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

# Deck → too many missing (>70%), drop it
print(f"Deck: {df['deck'].isnull().sum()} missing (77%) → dropping column")
df = df.drop(columns=['deck'])

print(f"\nMissing values after cleaning:\n{df.isnull().sum()}")

# ── Remove Duplicates ────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 3: REMOVING DUPLICATES")
print("=" * 50)

before = len(df)
df = df.drop_duplicates()
after = len(df)
print(f"Before: {before} rows → After: {after} rows → Removed: {before - after}")

# ── Fix Data Types ───────────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 4: FIXING DATA TYPES")
print("=" * 50)

df['survived'] = df['survived'].astype(int)
df['pclass'] = df['pclass'].astype(int)
df['age'] = df['age'].astype(int)
df['sex'] = df['sex'].astype('category')
df['embarked'] = df['embarked'].astype('category')

print("Converted: survived→int, pclass→int, age→int, sex→category, embarked→category")
print(f"\nUpdated Data Types:\n{df.dtypes}")

# ── Handle Outliers (IQR Method) ─────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 5: HANDLING OUTLIERS (IQR Method)")
print("=" * 50)

def cap_outliers(df, column):
    """Cap outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df[column] < lower) | (df[column] > upper)).sum()
    df[column] = df[column].clip(lower=lower, upper=upper)
    print(f"  {column}: {outliers} outliers capped to [{lower:.2f}, {upper:.2f}]")
    return df

for col in ['age', 'fare', 'sibsp', 'parch']:
    df = cap_outliers(df, col)

# ── Create New Features ──────────────────────────────────────────
print("\n" + "=" * 50)
print("  STEP 6: FEATURE ENGINEERING")
print("=" * 50)

# Family size
df['family_size'] = df['sibsp'] + df['parch'] + 1
print("Created 'family_size' = sibsp + parch + 1")

# Age group
df['age_group'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 55, 100],
                         labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])
print("Created 'age_group' with bins: Child, Teen, Young Adult, Adult, Senior")

# Fare category
df['fare_category'] = pd.qcut(df['fare'], q=4,
                               labels=['Budget', 'Economy', 'Business', 'Premium'])
print("Created 'fare_category' with 4 quartile bins")

# ── Save Cleaned Dataset ─────────────────────────────────────────
print("\n" + "=" * 50)
print("  SAVING CLEANED DATASET")
print("=" * 50)

os.makedirs("data", exist_ok=True)
df.to_csv("data/titanic_cleaned.csv", index=False)
print(f"Saved to: data/titanic_cleaned.csv")
print(f"Final shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nCleaned Data Sample:")
print(df.head(10))
