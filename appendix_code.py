"""
Library Justification:
- pandas: for data import, manipulation, and cleaning.
- numpy: for handling missing values and numerical calculations.
- matplotlib & seaborn: for creating data visualisations (histograms, boxplots, etc.).
- scipy.stats: for statistical testing (normality, Mann–Whitney U, Chi-square).
- os & subprocess: for managing output folders and viewing results.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------
# 1. Directory setup (adjust paths if needed)
# ------------------------------------------------------------
BASE_DIR = r"C:\Users\USER\Desktop\Arden"
DATA_PATH = os.path.join(BASE_DIR, "Career data_PDA_4053.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURE_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------
df_raw = pd.read_excel(DATA_PATH)
print("Dataset loaded successfully.")
print("Shape:", df_raw.shape)

# Keep a copy for reference
df = df_raw.copy()

# ------------------------------------------------------------
# 3. Handle anomalies and data type corrections
# ------------------------------------------------------------
for col in ["Career Change Interest", "Certifications", "Geographic Mobility"]:
    if col in df.columns:
        df[col] = df[col].replace("A", np.nan)

# Convert numeric columns properly
if "Salary" in df.columns:
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

# Ordinal encoding for ordered categories
edu_map = {"High School": 0, "Bachelor's": 1, "Master's": 2, "PhD": 3}
igr_map = {"Low": 0, "Medium": 1, "High": 2}
if "Education Level" in df.columns:
    df["Education Level (ord)"] = df["Education Level"].map(edu_map)
if "Industry Growth Rate" in df.columns:
    df["Industry Growth Rate (ord)"] = df["Industry Growth Rate"].map(igr_map)

# Convert binary-like columns to numeric
for col in ["Career Change Interest", "Certifications", "Geographic Mobility"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ------------------------------------------------------------
# 4. Remove duplicates and handle missing values
# ------------------------------------------------------------
print("Duplicates before:", df.duplicated().sum())
df = df.drop_duplicates()

num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(exclude=[np.number]).columns

# Fill missing numeric values with median; categorical with mode
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    if df[c].isna().any():
        df[c] = df[c].fillna(df[c].mode()[0])

# ------------------------------------------------------------
# 5. Outlier treatment (Winsorisation)
# ------------------------------------------------------------
def winsorize(s):
    """Winsorisation caps extreme outliers at 1st and 99th percentiles."""
    lo, hi = s.quantile(0.01), s.quantile(0.99)
    return s.clip(lo, hi)

for col in ["Salary", "Age", "Years of Experience", "Job Opportunities"]:
    if col in df.columns:
        df[col] = winsorize(df[col])

# ------------------------------------------------------------
# 6. Save cleaned dataset
# ------------------------------------------------------------
CLEAN_PATH = os.path.join(OUTPUT_DIR, "Career_data_clean.csv")
df.to_csv(CLEAN_PATH, index=False)
print(f"Cleaned dataset saved -> {CLEAN_PATH}")

# ------------------------------------------------------------
# 7. Descriptive statistics (before vs after)
# ------------------------------------------------------------
salary_before = pd.to_numeric(df_raw["Salary"], errors="coerce")
print("\n--- Salary (Before Cleaning) ---")
print(salary_before.describe())

print("\n--- Salary (After Cleaning) ---")
print(df["Salary"].describe())

# ------------------------------------------------------------
# 8. Plot saving function
# ------------------------------------------------------------
def save_plot(fig, filename):
    """Save figure to output folder."""
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_DIR, filename), bbox_inches="tight", dpi=300)
    plt.close(fig)

# ------------------------------------------------------------
# 9. Visualisations
# ------------------------------------------------------------
sns.set(style="whitegrid")

# --- Salary distributions ---
fig, ax = plt.subplots(figsize=(6,4))
ax.hist(salary_before.dropna(), bins=30, color='lightcoral', edgecolor='black')
ax.set_title("Salary Distribution (Before Cleaning)")
ax.set_xlabel("Salary")
ax.set_ylabel("Frequency")
save_plot(fig, "salary_before_hist.png")

fig, ax = plt.subplots(figsize=(6,4))
ax.hist(df["Salary"], bins=30, color='mediumseagreen', edgecolor='black')
ax.set_title("Salary Distribution (After Cleaning)")
ax.set_xlabel("Salary")
ax.set_ylabel("Frequency")
save_plot(fig, "salary_after_hist.png")

fig, ax = plt.subplots(figsize=(6,4))
ax.boxplot(df["Salary"].dropna())
ax.set_title("Salary Boxplot (After Cleaning)")
save_plot(fig, "salary_after_box.png")

# --- Gender distribution ---
if "Gender" in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    df["Gender"].value_counts().plot(kind="bar", ax=ax, color='skyblue')
    ax.set_title("Gender Distribution")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    save_plot(fig, "gender_distribution.png")

# --- Field of Study distribution ---
if "Field of Study" in df.columns:
    fig, ax = plt.subplots(figsize=(8,4))
    df["Field of Study"].value_counts().head(10).plot(kind="bar", ax=ax, color='steelblue')
    ax.set_title("Top 10 Fields of Study")
    ax.set_xlabel("Field of Study")
    ax.set_ylabel("Count")
    save_plot(fig, "field_study_distribution.png")

# --- Age distribution
if "Age" in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    ax.hist(df["Age"], bins=20, color='cornflowerblue', edgecolor='black')
    ax.set_title("Age Distribution of Respondents")
    ax.set_xlabel("Age")
    ax.set_ylabel("Count")
    save_plot(fig, "age_distribution.png")

# --- Correlation heatmap ---
corr = df.select_dtypes(include=[np.number]).corr()
fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
ax.set_title("Correlation Heatmap (Numeric Features)")
save_plot(fig, "corr_heatmap.png")

# --- Education vs Career Change ---
target = "Career Change Interest"
if "Education Level" in df.columns and target in df.columns:
    edu_order = ["High School","Bachelor's","Master's","PhD"]
    edu_rates = df.groupby("Education Level")[target].mean().reindex(edu_order)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.bar(edu_rates.index, edu_rates.values, color='teal')
    ax.set_xticklabels(edu_rates.index, rotation=45)
    ax.set_title("Mean Likelihood of Career Change by Education Level")
    ax.set_ylabel("Mean Likelihood (0–1)")
    save_plot(fig, "career_change_by_education.png")

# --- Experience vs Career Change ---
if target in df.columns and "Years of Experience" in df.columns:
    groups = [df.loc[df[target]==g, "Years of Experience"].dropna()
              for g in sorted(df[target].unique())]
    fig, ax = plt.subplots(figsize=(6,4))
    ax.boxplot(groups)
    ax.set_title("Years of Experience by Career Change Likelihood")
    ax.set_xlabel("Career Change Interest (0=No, 1=Yes)")
    ax.set_ylabel("Years of Experience")
    save_plot(fig, "experience_by_target_box.png")

# --- Job Satisfaction vs Career Change 
if target in df.columns and "Job Satisfaction" in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    df.boxplot(column="Job Satisfaction", by=target, ax=ax)
    plt.suptitle("")
    ax.set_title("Job Satisfaction by Career Change Interest")
    ax.set_xlabel("Career Change Interest (0=No, 1=Yes)")
    ax.set_ylabel("Job Satisfaction")
    save_plot(fig, "job_satisfaction_by_target.png")

# --- Gender vs Career Change ---
if "Gender" in df.columns and target in df.columns:
    fig, ax = plt.subplots(figsize=(6,4))
    df.groupby("Gender")[target].mean().plot(kind="bar", ax=ax, color='orchid')
    ax.set_title("Mean Career Change Likelihood by Gender")
    ax.set_ylabel("Mean Likelihood (0–1)")
    ax.set_xlabel("Gender")
    save_plot(fig, "career_change_by_gender.png")

print(f"Figures saved -> {FIGURE_DIR}")

# ------------------------------------------------------------
# 10. Statistical tests
# ------------------------------------------------------------
if target in df.columns:
    # Normality test for Salary
    norm_stat, norm_p = stats.normaltest(df["Salary"].dropna())
    print(f"\nNormality test (Salary): stat={norm_stat:.2f}, p={norm_p:.4f}")

    # Mann–Whitney U test: Years of Experience vs Career Change
    y0 = df.loc[df[target]==0, "Years of Experience"].dropna()
    y1 = df.loc[df[target]==1, "Years of Experience"].dropna()
    mw_stat, mw_p = stats.mannwhitneyu(y0, y1)
    print(f"Mann–Whitney U (YoE by Career Change): stat={mw_stat:.2f}, p={mw_p:.4f}")

    # Chi-square: Education vs Career Change
    if "Education Level" in df.columns:
        ct = pd.crosstab(df["Education Level"], df[target])
        chi2_stat, chi2_p, _, _ = stats.chi2_contingency(ct)
        print(f"Chi-square (Education vs Career Change): stat={chi2_stat:.2f}, p={chi2_p:.4f}")


print("\n--- Summary ---")
print(f"Total observations after cleaning: {len(df)}")
print("Figures and CSV outputs saved successfully.")
print("Statistical tests complete (Normality, Mann–Whitney, Chi-Square).")
print("All analysis steps completed as per COM7024 Guidelines.")
print("\n=== Processing Complete ===")

subprocess.Popen(r'explorer.exe "C:\Users\USER\Desktop\Arden\figures"')


