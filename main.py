"""
Run all steps in one go:
  1. Data Cleaning
  2. EDA Visualizations
  3. Interactive Dashboard
"""

print("=" * 60)
print("  TITANIC DATASET — DATA CLEANING & VISUALIZATION PROJECT")
print("=" * 60)

# Step 1
print("\n>>> Running Step 1: Data Cleaning...")
exec(open("data_cleaning.py").read())

# Step 2
print("\n>>> Running Step 2: EDA Visualizations...")
exec(open("eda_visualizations.py").read())

# Step 3
print("\n>>> Running Step 3: Interactive Dashboard...")
exec(open("interactive_dashboard.py").read())

print("\n" + "=" * 60)
print("  ALL DONE!")
print("=" * 60)
print("""
Project outputs:
  📁 data/titanic_cleaned.csv    — Cleaned dataset
  📁 outputs/*.png               — Static plots (9 charts)
  📁 outputs/*.html              — Interactive Plotly charts (5 charts)

To view interactive charts, open the .html files in your browser.
""")
