"""
eda_visualizations.py - Exploratory Data Analysis & Static Visualizations.
Creates: histograms, bar charts, box plots, heatmaps, count plots, pair plots.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PLOTS_DIR, FIGURE_DPI, SEABORN_STYLE

# Set global style
sns.set_style(SEABORN_STYLE)
plt.rcParams.update({
    "figure.dpi": FIGURE_DPI, "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "figure.facecolor": "#0D1117", "axes.facecolor": "#161B22",
    "text.color": "#E6EDF3", "axes.labelcolor": "#E6EDF3",
    "xtick.color": "#8B949E", "ytick.color": "#8B949E",
    "axes.edgecolor": "#30363D", "grid.color": "#21262D",
})

PALETTE = ["#58A6FF", "#F78166", "#3FB950", "#D2A8FF", "#79C0FF",
           "#FFA657", "#7EE787", "#FF7B72", "#D29922", "#A5D6FF"]


class EDAVisualizer:
    def __init__(self, df, output_dir=None):
        self.df = df
        self.output_dir = output_dir or PLOTS_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.output_dir, f"{name}.png")
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)
        print(f"  Saved: {name}.png")
        return path

    # 1. Survival Distribution
    def plot_survival_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Survival Distribution", fontsize=18, fontweight="bold", color="#58A6FF")

        # Pie chart
        survived = self.df["Survived"].value_counts()
        colors = [PALETTE[1], PALETTE[2]]
        explode = (0.05, 0.05)
        axes[0].pie(survived, labels=["Did Not Survive", "Survived"],
                    autopct="%1.1f%%", colors=colors, explode=explode,
                    textprops={"color": "#E6EDF3", "fontsize": 13},
                    wedgeprops={"edgecolor": "#30363D", "linewidth": 2})
        axes[0].set_title("Overall Survival Rate", color="#E6EDF3", fontsize=14)

        # Bar chart
        sns.countplot(data=self.df, x="Survived", palette=colors, ax=axes[1],
                      edgecolor="#30363D", linewidth=1.5)
        axes[1].set_xticklabels(["Did Not Survive", "Survived"])
        axes[1].set_title("Survival Count", color="#E6EDF3", fontsize=14)
        for p in axes[1].patches:
            axes[1].annotate(f'{int(p.get_height())}',
                            (p.get_x() + p.get_width()/2., p.get_height()),
                            ha='center', va='bottom', fontsize=12, color="#E6EDF3")
        return self._save(fig, "01_survival_distribution")

    # 2. Age Distribution
    def plot_age_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Age Distribution Analysis", fontsize=18, fontweight="bold", color="#58A6FF")

        sns.histplot(data=self.df, x="Age", hue="Survived", kde=True,
                     palette=[PALETTE[1], PALETTE[2]], ax=axes[0],
                     alpha=0.7, edgecolor="#30363D")
        axes[0].set_title("Age by Survival", fontsize=14)
        axes[0].legend(["Survived","Did Not Survive"])

        sns.boxplot(data=self.df, x="Survived", y="Age",
                    palette=[PALETTE[1], PALETTE[2]], ax=axes[1],
                    linewidth=1.5, flierprops={"marker":"o","markersize":4})
        axes[1].set_xticklabels(["Did Not Survive", "Survived"])
        axes[1].set_title("Age Box Plot by Survival", fontsize=14)
        return self._save(fig, "02_age_distribution")

    # 3. Passenger Class Analysis
    def plot_class_analysis(self):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle("Passenger Class Analysis", fontsize=18, fontweight="bold", color="#58A6FF")

        sns.countplot(data=self.df, x="Pclass", hue="Survived",
                      palette=[PALETTE[1], PALETTE[2]], ax=axes[0], edgecolor="#30363D")
        axes[0].set_title("Survival by Class", fontsize=14)
        axes[0].set_xticklabels(["1st","2nd","3rd"])

        survival_rate = self.df.groupby("Pclass")["Survived"].mean() * 100
        bars = axes[1].bar(["1st","2nd","3rd"], survival_rate.values,
                           color=[PALETTE[2], PALETTE[0], PALETTE[1]], edgecolor="#30363D")
        for bar, val in zip(bars, survival_rate.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{val:.1f}%", ha="center", color="#E6EDF3", fontsize=12)
        axes[1].set_title("Survival Rate by Class", fontsize=14)
        axes[1].set_ylabel("Survival Rate (%)")

        sns.boxplot(data=self.df, x="Pclass", y="Fare",
                    palette=PALETTE[:3], ax=axes[2], linewidth=1.5)
        axes[2].set_xticklabels(["1st","2nd","3rd"])
        axes[2].set_title("Fare by Class", fontsize=14)
        return self._save(fig, "03_class_analysis")

    # 4. Gender Analysis
    def plot_gender_analysis(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Gender & Survival Analysis", fontsize=18, fontweight="bold", color="#58A6FF")

        sns.countplot(data=self.df, x="Sex", hue="Survived",
                      palette=[PALETTE[1], PALETTE[2]], ax=axes[0], edgecolor="#30363D")
        axes[0].set_title("Survival by Gender", fontsize=14)

        gender_surv = self.df.groupby("Sex")["Survived"].mean() * 100
        bars = axes[1].bar(gender_surv.index, gender_surv.values,
                           color=[PALETTE[3], PALETTE[0]], edgecolor="#30363D")
        for bar, val in zip(bars, gender_surv.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{val:.1f}%", ha="center", color="#E6EDF3", fontsize=13)
        axes[1].set_title("Survival Rate by Gender", fontsize=14)
        axes[1].set_ylabel("Survival Rate (%)")
        return self._save(fig, "04_gender_analysis")

    # 5. Fare Distribution
    def plot_fare_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Fare Distribution", fontsize=18, fontweight="bold", color="#58A6FF")

        sns.histplot(data=self.df, x="Fare", hue="Survived", kde=True,
                     palette=[PALETTE[1], PALETTE[2]], ax=axes[0], alpha=0.7, bins=30)
        axes[0].set_title("Fare by Survival", fontsize=14)

        sns.violinplot(data=self.df, x="Pclass", y="Fare", hue="Survived",
                       split=True, palette=[PALETTE[1], PALETTE[2]], ax=axes[1])
        axes[1].set_xticklabels(["1st","2nd","3rd"])
        axes[1].set_title("Fare Violin Plot", fontsize=14)
        return self._save(fig, "05_fare_distribution")

    # 6. Correlation Heatmap
    def plot_correlation_heatmap(self):
        fig, ax = plt.subplots(figsize=(12, 10))
        fig.suptitle("Correlation Heatmap", fontsize=18, fontweight="bold", color="#58A6FF")

        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, ax=ax, linewidths=0.5, linecolor="#30363D",
                    annot_kws={"size": 10, "color": "#E6EDF3"},
                    cbar_kws={"shrink": 0.8})
        ax.set_title("Feature Correlations", fontsize=14, pad=20)
        return self._save(fig, "06_correlation_heatmap")

    # 7. Embarked Analysis
    def plot_embarked_analysis(self):
        if "Embarked" not in self.df.columns:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Embarkation Port Analysis", fontsize=18, fontweight="bold", color="#58A6FF")

        port_map = {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}
        df_plot = self.df.copy()
        df_plot["Port"] = df_plot["Embarked"].map(port_map).fillna(df_plot["Embarked"])

        sns.countplot(data=df_plot, x="Port", hue="Survived",
                      palette=[PALETTE[1], PALETTE[2]], ax=axes[0], edgecolor="#30363D")
        axes[0].set_title("Survival by Port", fontsize=14)

        port_surv = df_plot.groupby("Port")["Survived"].mean() * 100
        bars = axes[1].bar(port_surv.index, port_surv.values,
                           color=PALETTE[:3], edgecolor="#30363D")
        for bar, val in zip(bars, port_surv.values):
            axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                        f"{val:.1f}%", ha="center", color="#E6EDF3", fontsize=12)
        axes[1].set_title("Survival Rate by Port", fontsize=14)
        axes[1].set_ylabel("Survival Rate (%)")
        return self._save(fig, "07_embarked_analysis")

    # 8. Family Size Analysis
    def plot_family_analysis(self):
        if "FamilySize" not in self.df.columns:
            return None
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Family Size & Survival", fontsize=18, fontweight="bold", color="#58A6FF")

        fam_surv = self.df.groupby("FamilySize")["Survived"].mean() * 100
        bars = axes[0].bar(fam_surv.index.astype(str), fam_surv.values,
                           color=PALETTE[0], edgecolor="#30363D")
        axes[0].set_title("Survival Rate by Family Size", fontsize=14)
        axes[0].set_xlabel("Family Size")
        axes[0].set_ylabel("Survival Rate (%)")

        if "AgeGroup" in self.df.columns:
            age_surv = self.df.groupby("AgeGroup")["Survived"].mean() * 100
            bars2 = axes[1].bar(age_surv.index.astype(str), age_surv.values,
                               color=PALETTE[3], edgecolor="#30363D")
            for bar, val in zip(bars2, age_surv.values):
                axes[1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
                            f"{val:.1f}%", ha="center", color="#E6EDF3", fontsize=11)
            axes[1].set_title("Survival Rate by Age Group", fontsize=14)
            axes[1].set_ylabel("Survival Rate (%)")
        return self._save(fig, "08_family_analysis")

    # 9. Multi-variable Analysis
    def plot_multivariate(self):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle("Multi-Variable Analysis", fontsize=18, fontweight="bold", color="#58A6FF")

        sns.scatterplot(data=self.df, x="Age", y="Fare", hue="Survived",
                        style="Pclass", palette=[PALETTE[1], PALETTE[2]],
                        alpha=0.7, ax=axes[0], s=60, edgecolor="#30363D")
        axes[0].set_title("Age vs Fare (by Survival & Class)", fontsize=14)

        pivot = self.df.pivot_table(values="Survived", index="Pclass",
                                    columns="Sex", aggfunc="mean") * 100
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", ax=axes[1],
                    linewidths=2, linecolor="#30363D",
                    annot_kws={"size": 16, "fontweight": "bold"},
                    cbar_kws={"label": "Survival Rate (%)"})
        axes[1].set_title("Survival Rate: Class × Gender", fontsize=14)
        axes[1].set_yticklabels(["1st","2nd","3rd"])
        return self._save(fig, "09_multivariate_analysis")

    # 10. Summary Dashboard
    def plot_summary_dashboard(self):
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle("TITANIC DATASET — EDA DASHBOARD",
                     fontsize=24, fontweight="bold", color="#58A6FF", y=0.98)
        gs = gridspec.GridSpec(3, 3, hspace=0.35, wspace=0.3)

        # Survival pie
        ax1 = fig.add_subplot(gs[0, 0])
        survived = self.df["Survived"].value_counts()
        ax1.pie(survived, labels=["Died","Survived"], autopct="%1.1f%%",
                colors=[PALETTE[1], PALETTE[2]], textprops={"color":"#E6EDF3"},
                wedgeprops={"edgecolor":"#30363D"})
        ax1.set_title("Survival Rate", fontsize=13)

        # Age hist
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(data=self.df, x="Age", hue="Survived", kde=True,
                     palette=[PALETTE[1], PALETTE[2]], ax=ax2, alpha=0.6)
        ax2.set_title("Age Distribution", fontsize=13)

        # Class survival
        ax3 = fig.add_subplot(gs[0, 2])
        sr = self.df.groupby("Pclass")["Survived"].mean()*100
        ax3.bar(["1st","2nd","3rd"], sr.values, color=PALETTE[:3], edgecolor="#30363D")
        ax3.set_title("Survival by Class", fontsize=13)
        ax3.set_ylabel("%")

        # Gender
        ax4 = fig.add_subplot(gs[1, 0])
        gs2 = self.df.groupby("Sex")["Survived"].mean()*100
        ax4.bar(gs2.index, gs2.values, color=[PALETTE[3], PALETTE[0]], edgecolor="#30363D")
        ax4.set_title("Survival by Gender", fontsize=13)
        ax4.set_ylabel("%")

        # Fare box
        ax5 = fig.add_subplot(gs[1, 1])
        sns.boxplot(data=self.df, x="Pclass", y="Fare", palette=PALETTE[:3], ax=ax5)
        ax5.set_xticklabels(["1st","2nd","3rd"])
        ax5.set_title("Fare by Class", fontsize=13)

        # Correlation
        ax6 = fig.add_subplot(gs[1, 2])
        num = self.df.select_dtypes(include=[np.number])
        corr = num.corr()
        sns.heatmap(corr, annot=True, fmt=".1f", cmap="coolwarm", center=0,
                    ax=ax6, annot_kws={"size":7}, linewidths=0.5, cbar=False)
        ax6.set_title("Correlations", fontsize=13)

        # Embarked
        ax7 = fig.add_subplot(gs[2, 0])
        if "Embarked" in self.df.columns:
            sns.countplot(data=self.df, x="Embarked", hue="Survived",
                          palette=[PALETTE[1], PALETTE[2]], ax=ax7, edgecolor="#30363D")
        ax7.set_title("Embarkation", fontsize=13)

        # Family size
        ax8 = fig.add_subplot(gs[2, 1])
        if "FamilySize" in self.df.columns:
            fs = self.df.groupby("FamilySize")["Survived"].mean()*100
            ax8.bar(fs.index.astype(str), fs.values, color=PALETTE[4], edgecolor="#30363D")
        ax8.set_title("Family Size Survival", fontsize=13)
        ax8.set_ylabel("%")

        # Scatter
        ax9 = fig.add_subplot(gs[2, 2])
        sns.scatterplot(data=self.df, x="Age", y="Fare", hue="Survived",
                        palette=[PALETTE[1], PALETTE[2]], alpha=0.6, ax=ax9, s=30)
        ax9.set_title("Age vs Fare", fontsize=13)

        return self._save(fig, "10_summary_dashboard")

    def generate_all_plots(self):
        print("\n" + "="*50)
        print("GENERATING EDA VISUALIZATIONS")
        print("="*50)
        self.plot_survival_distribution()
        self.plot_age_distribution()
        self.plot_class_analysis()
        self.plot_gender_analysis()
        self.plot_fare_distribution()
        self.plot_correlation_heatmap()
        self.plot_embarked_analysis()
        self.plot_family_analysis()
        self.plot_multivariate()
        self.plot_summary_dashboard()
        print(f"\nAll plots saved to: {self.output_dir}")


if __name__ == "__main__":
    from data_loader import load_dataset
    from data_cleaning import DataCleaner
    df = load_dataset()
    cleaner = DataCleaner(df)
    cleaner.run_pipeline()
    viz = EDAVisualizer(cleaner.get_dataframe())
    viz.generate_all_plots()
