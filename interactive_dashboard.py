"""
Step 3: Interactive Dashboard (Plotly)
- Interactive plots that you can hover, zoom, and filter
- Opens in your browser automatically
- Saves as an HTML file you can share
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# Load cleaned data
df = pd.read_csv("data/titanic_cleaned.csv")
print(f"Loaded: {df.shape[0]} rows, {df.shape[1]} columns\n")

os.makedirs("outputs", exist_ok=True)

# ── Chart 1: Survival by Class & Gender (Interactive Bar) ────────
fig1 = px.histogram(df, x="pclass", color="survived", barmode="group",
                    facet_col="sex", title="Survival by Class & Gender",
                    labels={"pclass": "Passenger Class", "survived": "Survived"},
                    color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                    category_orders={"pclass": [1, 2, 3]})
fig1.update_layout(template="plotly_dark")
fig1.write_html("outputs/interactive_01_class_gender.html")
print("Saved: interactive_01_class_gender.html")

# ── Chart 2: Age Distribution (Interactive Histogram) ────────────
fig2 = px.histogram(df, x="age", color="survived", nbins=30,
                    title="Age Distribution by Survival",
                    labels={"age": "Age", "survived": "Survived"},
                    color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                    marginal="box", opacity=0.7)
fig2.update_layout(template="plotly_dark")
fig2.write_html("outputs/interactive_02_age.html")
print("Saved: interactive_02_age.html")

# ── Chart 3: Fare vs Age Scatter ─────────────────────────────────
fig3 = px.scatter(df, x="age", y="fare", color="survived",
                  size="family_size", hover_data=["sex", "pclass", "embarked"],
                  title="Age vs Fare (size = family size)",
                  labels={"age": "Age", "fare": "Fare", "survived": "Survived"},
                  color_discrete_map={0: "#e74c3c", 1: "#2ecc71"},
                  opacity=0.7)
fig3.update_layout(template="plotly_dark")
fig3.write_html("outputs/interactive_03_scatter.html")
print("Saved: interactive_03_scatter.html")

# ── Chart 4: Sunburst Chart (Class → Gender → Survived) ─────────
df_sunburst = df.copy()
df_sunburst["survived_label"] = df_sunburst["survived"].map({0: "Died", 1: "Survived"})
df_sunburst["class_label"] = df_sunburst["pclass"].map({1: "1st Class", 2: "2nd Class", 3: "3rd Class"})

fig4 = px.sunburst(df_sunburst, path=["class_label", "sex", "survived_label"],
                   title="Passenger Breakdown: Class → Gender → Survival",
                   color="survived_label",
                   color_discrete_map={"Died": "#e74c3c", "Survived": "#2ecc71"})
fig4.update_layout(template="plotly_dark")
fig4.write_html("outputs/interactive_04_sunburst.html")
print("Saved: interactive_04_sunburst.html")

# ── Chart 5: Full Interactive Dashboard ──────────────────────────
fig5 = make_subplots(
    rows=2, cols=2,
    subplot_titles=("Survival Count", "Age Distribution",
                    "Fare by Class", "Survival by Gender"),
    specs=[[{"type": "bar"}, {"type": "histogram"}],
           [{"type": "box"}, {"type": "bar"}]]
)

# Survival count
surv_counts = df["survived"].value_counts().sort_index()
fig5.add_trace(go.Bar(x=["Died", "Survived"], y=surv_counts.values,
                      marker_color=["#e74c3c", "#2ecc71"], name="Survival"),
               row=1, col=1)

# Age histogram
fig5.add_trace(go.Histogram(x=df[df["survived"]==0]["age"], name="Died",
                            marker_color="#e74c3c", opacity=0.7), row=1, col=2)
fig5.add_trace(go.Histogram(x=df[df["survived"]==1]["age"], name="Survived",
                            marker_color="#2ecc71", opacity=0.7), row=1, col=2)

# Fare by class
for pclass, color in zip([1, 2, 3], ["#3498db", "#e67e22", "#9b59b6"]):
    fig5.add_trace(go.Box(y=df[df["pclass"]==pclass]["fare"],
                          name=f"Class {pclass}", marker_color=color),
                   row=2, col=1)

# Gender survival rate
gender_surv = df.groupby("sex")["survived"].mean() * 100
fig5.add_trace(go.Bar(x=gender_surv.index, y=gender_surv.values,
                      marker_color=["#e67e22", "#3498db"],
                      name="Gender Survival", text=[f"{v:.1f}%" for v in gender_surv.values],
                      textposition="outside"), row=2, col=2)

fig5.update_layout(height=700, title_text="Titanic Dataset — Interactive Dashboard",
                   template="plotly_dark", showlegend=False)
fig5.write_html("outputs/interactive_dashboard.html")
print("Saved: interactive_dashboard.html")

print("\n✅ All interactive charts saved in 'outputs/' folder!")
print("Open the .html files in your browser to explore them.")
