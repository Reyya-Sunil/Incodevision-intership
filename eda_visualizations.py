"""
Step 2: EDA & Visualization (matplotlib + seaborn)
- Histograms, bar charts, box plots, heatmaps
- Survival analysis by class, gender, age, fare, etc.
- All plots saved to outputs/ folder
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ── Setup ─────────────────────────────────────────────────────────
sns.set_style("whitegrid")
os.makedirs("outputs", exist_ok=True)

# Load cleaned data
df = pd.read_csv("data/titanic_cleaned.csv")
print(f"Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns\n")

# ── Plot 1: Survival Count (Bar Chart) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Survival Overview", fontsize=16, fontweight="bold")

# Count plot
sns.countplot(data=df, x="survived", palette=["#e74c3c", "#2ecc71"], ax=axes[0])
axes[0].set_xticklabels(["Did Not Survive", "Survived"])
axes[0].set_title("Survival Count")
axes[0].set_ylabel("Count")

# Pie chart
survived_counts = df["survived"].value_counts()
axes[1].pie(survived_counts, labels=["Did Not Survive", "Survived"],
            autopct="%1.1f%%", colors=["#e74c3c", "#2ecc71"],
            explode=(0.05, 0.05))
axes[1].set_title("Survival Percentage")

plt.tight_layout()
plt.savefig("outputs/01_survival_overview.png", dpi=150)
plt.close()
print("Saved: 01_survival_overview.png")

# ── Plot 2: Age Distribution (Histogram) ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Age Analysis", fontsize=16, fontweight="bold")

sns.histplot(data=df, x="age", hue="survived", kde=True,
             palette=["#e74c3c", "#2ecc71"], ax=axes[0], alpha=0.7)
axes[0].set_title("Age Distribution by Survival")

sns.boxplot(data=df, x="survived", y="age",
            palette=["#e74c3c", "#2ecc71"], ax=axes[1])
axes[1].set_xticklabels(["Did Not Survive", "Survived"])
axes[1].set_title("Age Box Plot by Survival")

plt.tight_layout()
plt.savefig("outputs/02_age_analysis.png", dpi=150)
plt.close()
print("Saved: 02_age_analysis.png")

# ── Plot 3: Class Analysis (Bar Chart) ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Passenger Class Analysis", fontsize=16, fontweight="bold")

sns.countplot(data=df, x="pclass", hue="survived",
              palette=["#e74c3c", "#2ecc71"], ax=axes[0])
axes[0].set_title("Survival by Class")
axes[0].set_xticklabels(["1st Class", "2nd Class", "3rd Class"])

# Survival rate by class
surv_rate = df.groupby("pclass")["survived"].mean() * 100
bars = axes[1].bar(["1st", "2nd", "3rd"], surv_rate.values,
                   color=["#3498db", "#e67e22", "#9b59b6"])
for bar, val in zip(bars, surv_rate.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12)
axes[1].set_title("Survival Rate by Class (%)")
axes[1].set_ylabel("Survival Rate (%)")

plt.tight_layout()
plt.savefig("outputs/03_class_analysis.png", dpi=150)
plt.close()
print("Saved: 03_class_analysis.png")

# ── Plot 4: Gender Analysis ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Gender & Survival", fontsize=16, fontweight="bold")

sns.countplot(data=df, x="sex", hue="survived",
              palette=["#e74c3c", "#2ecc71"], ax=axes[0])
axes[0].set_title("Survival Count by Gender")

gender_surv = df.groupby("sex")["survived"].mean() * 100
bars = axes[1].bar(gender_surv.index, gender_surv.values,
                   color=["#e67e22", "#3498db"])
for bar, val in zip(bars, gender_surv.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=13)
axes[1].set_title("Survival Rate by Gender (%)")
axes[1].set_ylabel("Survival Rate (%)")

plt.tight_layout()
plt.savefig("outputs/04_gender_analysis.png", dpi=150)
plt.close()
print("Saved: 04_gender_analysis.png")

# ── Plot 5: Fare Distribution ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Fare Analysis", fontsize=16, fontweight="bold")

sns.histplot(data=df, x="fare", hue="survived", kde=True,
             palette=["#e74c3c", "#2ecc71"], ax=axes[0], bins=30)
axes[0].set_title("Fare Distribution by Survival")

sns.boxplot(data=df, x="pclass", y="fare",
            palette=["#3498db", "#e67e22", "#9b59b6"], ax=axes[1])
axes[1].set_xticklabels(["1st Class", "2nd Class", "3rd Class"])
axes[1].set_title("Fare by Passenger Class")

plt.tight_layout()
plt.savefig("outputs/05_fare_analysis.png", dpi=150)
plt.close()
print("Saved: 05_fare_analysis.png")

# ── Plot 6: Correlation Heatmap ──────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))
fig.suptitle("Correlation Heatmap", fontsize=16, fontweight="bold")

numeric_cols = df.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.5)
ax.set_title("Feature Correlations")

plt.tight_layout()
plt.savefig("outputs/06_correlation_heatmap.png", dpi=150)
plt.close()
print("Saved: 06_correlation_heatmap.png")

# ── Plot 7: Embarkation Port ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Embarkation Port Analysis", fontsize=16, fontweight="bold")

sns.countplot(data=df, x="embarked", hue="survived",
              palette=["#e74c3c", "#2ecc71"], ax=axes[0])
axes[0].set_title("Survival by Port")

port_surv = df.groupby("embarked")["survived"].mean() * 100
bars = axes[1].bar(port_surv.index, port_surv.values,
                   color=["#3498db", "#e67e22", "#9b59b6"])
for bar, val in zip(bars, port_surv.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=12)
axes[1].set_title("Survival Rate by Port (%)")
axes[1].set_ylabel("Survival Rate (%)")

plt.tight_layout()
plt.savefig("outputs/07_embarked_analysis.png", dpi=150)
plt.close()
print("Saved: 07_embarked_analysis.png")

# ── Plot 8: Family Size Analysis ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Family Size & Age Group Analysis", fontsize=16, fontweight="bold")

fam_surv = df.groupby("family_size")["survived"].mean() * 100
axes[0].bar(fam_surv.index.astype(str), fam_surv.values, color="#3498db")
axes[0].set_title("Survival Rate by Family Size")
axes[0].set_xlabel("Family Size")
axes[0].set_ylabel("Survival Rate (%)")

age_surv = df.groupby("age_group")["survived"].mean() * 100
bars = axes[1].bar(age_surv.index.astype(str), age_surv.values,
                   color=["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db"])
for bar, val in zip(bars, age_surv.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", fontsize=10)
axes[1].set_title("Survival Rate by Age Group")
axes[1].set_ylabel("Survival Rate (%)")

plt.tight_layout()
plt.savefig("outputs/08_family_age_analysis.png", dpi=150)
plt.close()
print("Saved: 08_family_age_analysis.png")

# ── Plot 9: Multi-Variable Heatmap ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Survival Rate: Class × Gender", fontsize=16, fontweight="bold")

pivot = df.pivot_table(values="survived", index="pclass",
                       columns="sex", aggfunc="mean") * 100
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=ax,
            linewidths=2, annot_kws={"size": 16, "fontweight": "bold"},
            cbar_kws={"label": "Survival Rate (%)"})
ax.set_yticklabels(["1st Class", "2nd Class", "3rd Class"])
ax.set_title("Survival Heatmap")

plt.tight_layout()
plt.savefig("outputs/09_survival_heatmap.png", dpi=150)
plt.close()
print("Saved: 09_survival_heatmap.png")

print("\n✅ All 9 plots saved in the 'outputs/' folder!")
