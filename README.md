# Data Cleaning & Visualization Project

A simple Python project that demonstrates data cleaning and exploratory data analysis (EDA) using the **Titanic dataset**.

## What This Project Does

| Step | Script | Description |
|------|--------|-------------|
| 1 | `data_cleaning.py` | Loads data, handles missing values, removes duplicates, fixes data types, caps outliers, creates new features |
| 2 | `eda_visualizations.py` | Creates 9 static plots — histograms, bar charts, box plots, heatmaps, pie charts |
| 3 | `interactive_dashboard.py` | Creates 5 interactive Plotly charts (HTML) |
| All | `main.py` | Runs all 3 steps at once |

## Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything
python main.py

# Or run individual steps:
python data_cleaning.py
python eda_visualizations.py
python interactive_dashboard.py
```

## Project Structure

```
data-cleaning-visualization/
├── data_cleaning.py            # Step 1: Clean the data
├── eda_visualizations.py       # Step 2: Static plots (matplotlib/seaborn)
├── interactive_dashboard.py    # Step 3: Interactive plots (Plotly)
├── main.py                     # Run all steps
├── requirements.txt            # Python dependencies
├── data/
│   └── titanic_cleaned.csv     # Cleaned dataset (generated)
└── outputs/
    ├── 01_survival_overview.png
    ├── 02_age_analysis.png
    ├── ...
    └── interactive_dashboard.html
```

## Tools & Libraries Used

- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **matplotlib** — Static plots
- **seaborn** — Statistical visualizations
- **plotly** — Interactive charts & dashboards

## Data Cleaning Steps

1. **Missing Values** — Age filled with median, Embarked with mode, Deck column dropped (77% missing)
2. **Duplicates** — Removed duplicate rows
3. **Data Types** — Converted to proper types (int, category)
4. **Outliers** — Capped using IQR method
5. **Feature Engineering** — Created family_size, age_group, fare_category

## Visualizations Created

### Static Plots (matplotlib/seaborn)
1. Survival Overview (bar + pie)
2. Age Distribution (histogram + box plot)
3. Class Analysis (bar charts)
4. Gender Analysis (bar charts)
5. Fare Analysis (histogram + box plot)
6. Correlation Heatmap
7. Embarkation Port Analysis
8. Family & Age Group Analysis
9. Survival Heatmap (Class × Gender)

### Interactive Plots (Plotly)
1. Survival by Class & Gender
2. Age Distribution
3. Age vs Fare Scatter
4. Sunburst Chart (Class → Gender → Survival)
5. Combined Dashboard
