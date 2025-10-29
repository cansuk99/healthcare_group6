"""
Step 3: Exploratory Data Analysis (EDA)
Diabetes 130-US Hospitals Dataset

This script:
1. Loads cleaned data from Step 2
2. Performs univariate analysis (single variable distributions)
3. Performs bivariate analysis (relationships with target variable)
4. Analyzes correlations between features
5. Creates comprehensive visualizations
6. Identifies key insights for modeling
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plt.rcParams["font.size"] = 10


# ============================================================================
# 1. LOAD CLEANED DATA
# ============================================================================

print("\n[1] Loading cleaned data from Step 2...")
try:
    df = pd.read_csv("../data/processed/diabetes_cleaned_data.csv")
    print(f"✓ Data loaded: {df.shape}")
    print(f"  - Patients: {len(df):,}")
    print(f"  - Features: {len(df.columns)}")
except FileNotFoundError:
    print("✗ Error: 'diabetes_cleaned_data.csv' not found.")
    print("  Please run Step 2 first to generate the cleaned data.")
    # exit()

# Quick verification
print("\n[1.1] Data Quality Check:")
print(f"  - Missing values: {df.isnull().sum().sum()}")
print(f"  - Duplicate rows: {df.duplicated().sum()}")

if "readmitted_binary" in df.columns:
    target_dist = df["readmitted_binary"].value_counts()
    print("\n[1.2] Target Variable:")
    print(f"  - Class 0: {target_dist[0]:,} ({target_dist[0]/len(df)*100:.1f}%)")
    print(f"  - Class 1: {target_dist[1]:,} ({target_dist[1]/len(df)*100:.1f}%)")

# ============================================================================
# 2. UNIVARIATE ANALYSIS - NUMERICAL FEATURES
# ============================================================================


# Identify numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# Remove target variable from list
if "readmitted_binary" in numerical_cols:
    numerical_cols.remove("readmitted_binary")

print(f"\n[2.1] Found {len(numerical_cols)} numerical features")

# Select key numerical features for detailed analysis
key_numerical = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]

# Filter to only those that exist
key_numerical = [col for col in key_numerical if col in df.columns]

print(f"\n[2.2] Analyzing {len(key_numerical)} key numerical features:")
for col in key_numerical:
    print(f"\n  {col}:")
    print(f"    Mean: {df[col].mean():.2f}")
    print(f"    Median: {df[col].median():.2f}")
    print(f"    Std: {df[col].std():.2f}")
    print(f"    Min: {df[col].min():.0f}, Max: {df[col].max():.0f}")
    print(f"    Skewness: {df[col].skew():.2f}")

# Create distribution plots
print("\n[2.3] Creating distribution visualizations...")
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle("Distribution of Key Numerical Features", fontsize=16, fontweight="bold")

for idx, col in enumerate(key_numerical[:9]):  # Plot first 9
    ax = axes[idx // 3, idx % 3]

    # Histogram with KDE
    df[col].hist(bins=30, ax=ax, alpha=0.7, color="skyblue", edgecolor="black")
    ax.set_xlabel(col.replace("_", " ").title())
    ax.set_ylabel("Frequency")
    ax.set_title(
        f'{col.replace("_", " ").title()}\n(Mean: {df[col].mean():.1f}, Median: {df[col].median():.1f})'
    )

    # Add median line
    ax.axvline(
        df[col].median(), color="red", linestyle="--", linewidth=2, label="Median"
    )
    ax.legend()

plt.tight_layout()
plt.savefig("../figures/03_numerical_distributions.png", dpi=300, bbox_inches="tight")
print("  ✓ Saved: 03_numerical_distributions.png")

# ============================================================================
# 3. UNIVARIATE ANALYSIS - CATEGORICAL FEATURES
# ============================================================================


# Identify categorical columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()

print(f"\n[3.1] Found {len(categorical_cols)} categorical features")

# Key categorical features
key_categorical = [
    "age",
    "gender",
    "race",
    "admission_type_id",
    "diag_1_category",
    "diag_2_category",
    "diag_3_category",
]
key_categorical = [col for col in key_categorical if col in df.columns]

print(f"\n[3.2] Analyzing {len(key_categorical)} key categorical features:")
for col in key_categorical:
    print(f"\n  {col}:")
    value_counts = df[col].value_counts()
    print(f"    Unique values: {df[col].nunique()}")
    print("    Top 3 values:")
    for val, count in value_counts.head(3).items():
        pct = (count / len(df)) * 100
        print(f"      {val}: {count:,} ({pct:.1f}%)")

# Create categorical distribution plots
print("\n[3.3] Creating categorical visualizations...")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Distribution of Key Categorical Features", fontsize=16, fontweight="bold")

# Age distribution
if "age" in df.columns:
    ax = axes[0, 0]
    age_counts = df["age"].value_counts().sort_index()
    age_counts.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Age Distribution", fontweight="bold")
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

# Gender distribution
if "gender" in df.columns:
    ax = axes[0, 1]
    gender_counts = df["gender"].value_counts()
    gender_counts.plot(kind="bar", ax=ax, color=["lightcoral", "skyblue", "lightgray"])
    ax.set_title("Gender Distribution", fontweight="bold")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=0)

# Primary diagnosis category
if "diag_1_category" in df.columns:
    ax = axes[1, 0]
    diag_counts = df["diag_1_category"].value_counts().head(9)
    diag_counts.plot(kind="barh", ax=ax, color="mediumseagreen")
    ax.set_title("Primary Diagnosis Categories", fontweight="bold")
    ax.set_xlabel("Count")
    ax.set_ylabel("Diagnosis Category")

# Race distribution
if "race" in df.columns:
    ax = axes[1, 1]
    race_counts = df["race"].value_counts()
    race_counts.plot(kind="bar", ax=ax, color="orchid")
    ax.set_title("Race Distribution", fontweight="bold")
    ax.set_xlabel("Race")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("../figures/03_categorical_distributions.png", dpi=300, bbox_inches="tight")
print("  ✓ Saved: 03_categorical_distributions.png")

# ============================================================================
# 4. BIVARIATE ANALYSIS - RELATIONSHIP WITH TARGET
# ============================================================================


if "readmitted_binary" not in df.columns:
    print(
        "  ✗ Target variable 'readmitted_binary' not found. Skipping bivariate analysis."
    )
else:
    print("\n[4.1] Readmission rates by numerical features:")

    # Analyze numerical features by target
    for col in key_numerical[:5]:  # Top 5 numerical features
        readmit_yes = df[df["readmitted_binary"] == 1][col].mean()
        readmit_no = df[df["readmitted_binary"] == 0][col].mean()
        diff = readmit_yes - readmit_no
        pct_diff = (diff / readmit_no) * 100

        # T-test
        stat, pval = stats.ttest_ind(
            df[df["readmitted_binary"] == 1][col], df[df["readmitted_binary"] == 0][col]
        )

        print(f"\n  {col}:")
        print(f"    Readmitted <30: {readmit_yes:.2f}")
        print(f"    Not readmitted <30: {readmit_no:.2f}")
        print(f"    Difference: {diff:+.2f} ({pct_diff:+.1f}%)")
        print(
            f"    Significance: p={pval:.4f} {'***' if pval < 0.001 else '**' if pval < 0.01 else '*' if pval < 0.05 else 'ns'}"
        )

    print("\n[4.2] Readmission rates by categorical features:")

    # Analyze categorical features by target
    for col in key_categorical[:3]:  # Top 3 categorical features
        print(f"\n  {col}:")
        crosstab = (
            pd.crosstab(df[col], df["readmitted_binary"], normalize="index") * 100
        )
        readmit_rates = crosstab[1].sort_values(ascending=False).head(5)

        for val, rate in readmit_rates.items():
            count = df[df[col] == val].shape[0]
            print(f"    {val}: {rate:.1f}% readmission rate (n={count:,})")

    # Create bivariate visualizations
    print("\n[4.3] Creating bivariate visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Readmission Rates by Key Features", fontsize=16, fontweight="bold")

    # Time in hospital vs readmission
    if "time_in_hospital" in df.columns:
        ax = axes[0, 0]
        time_readmit = df.groupby("time_in_hospital")["readmitted_binary"].mean() * 100
        time_readmit.plot(kind="line", ax=ax, marker="o", color="crimson", linewidth=2)
        ax.set_title("Readmission Rate by Time in Hospital", fontweight="bold")
        ax.set_xlabel("Days in Hospital")
        ax.set_ylabel("Readmission Rate (%)")
        ax.grid(True, alpha=0.3)

    # Number of medications vs readmission
    if "num_medications" in df.columns:
        ax = axes[0, 1]
        # Group into bins for cleaner visualization
        df["med_bins"] = pd.cut(
            df["num_medications"],
            bins=[0, 5, 10, 15, 20, 100],
            labels=["1-5", "6-10", "11-15", "16-20", "20+"],
        )
        med_readmit = df.groupby("med_bins")["readmitted_binary"].mean() * 100
        med_readmit.plot(kind="bar", ax=ax, color="teal")
        ax.set_title("Readmission Rate by Number of Medications", fontweight="bold")
        ax.set_xlabel("Number of Medications")
        ax.set_ylabel("Readmission Rate (%)")
        ax.tick_params(axis="x", rotation=45)

    # Age group vs readmission
    if "age" in df.columns:
        ax = axes[1, 0]
        age_readmit = df.groupby("age")["readmitted_binary"].mean() * 100
        age_readmit.sort_index().plot(kind="bar", ax=ax, color="darkorange")
        ax.set_title("Readmission Rate by Age Group", fontweight="bold")
        ax.set_xlabel("Age Group")
        ax.set_ylabel("Readmission Rate (%)")
        ax.tick_params(axis="x", rotation=45)
        ax.axhline(
            df["readmitted_binary"].mean() * 100,
            color="red",
            linestyle="--",
            label="Overall Mean",
        )
        ax.legend()

    # Diagnosis category vs readmission
    if "diag_1_category" in df.columns:
        ax = axes[1, 1]
        diag_readmit = df.groupby("diag_1_category")["readmitted_binary"].mean() * 100
        diag_readmit.sort_values(ascending=True).plot(
            kind="barh", ax=ax, color="mediumseagreen"
        )
        ax.set_title("Readmission Rate by Primary Diagnosis", fontweight="bold")
        ax.set_xlabel("Readmission Rate (%)")
        ax.set_ylabel("Diagnosis Category")
        ax.axvline(
            df["readmitted_binary"].mean() * 100,
            color="red",
            linestyle="--",
            label="Overall Mean",
        )
        ax.legend()

    plt.tight_layout()
    plt.savefig("../figures/03_bivariate_analysis.png", dpi=300, bbox_inches="tight")
    print("  ✓ Saved: 03_bivariate_analysis.png")

# ============================================================================
# 5. CORRELATION ANALYSIS
# ============================================================================


print("\n[5.1] Computing correlations for numerical features...")

# Select only numerical columns
numerical_df = df[numerical_cols + ["readmitted_binary"]]
corr_matrix = numerical_df.corr()

# Find top correlations with target
if "readmitted_binary" in corr_matrix.columns:
    target_corr = (
        corr_matrix["readmitted_binary"]
        .drop("readmitted_binary")
        .abs()
        .sort_values(ascending=False)
    )

    print("\n[5.2] Top 10 features correlated with readmission:")
    for idx, (feature, corr) in enumerate(target_corr.head(10).items(), 1):
        print(f"  {idx}. {feature}: {corr:.4f}")

    # Create correlation heatmap
    print("\n[5.3] Creating correlation heatmap...")
    fig, ax = plt.subplots(figsize=(14, 12))

    # Select top correlated features for cleaner visualization
    top_features = target_corr.head(15).index.tolist() + ["readmitted_binary"]
    corr_subset = numerical_df[top_features].corr()

    sns.heatmap(
        corr_subset,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(
        "Correlation Matrix - Top Features with Readmission",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.tight_layout()
    plt.savefig("../figures/03_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    print("  ✓ Saved: 03_correlation_heatmap.png")

# ============================================================================
# 6. KEY INSIGHTS SUMMARY
# ============================================================================


insights = []

# Insight 1: Class imbalance
if "readmitted_binary" in df.columns:
    imbalance_ratio = target_dist[0] / target_dist[1]
    insights.append(
        f"1. Class imbalance ratio: 1:{imbalance_ratio:.1f} - SMOTE recommended"
    )

# Insight 2: Top 3 correlated features
if "readmitted_binary" in corr_matrix.columns:
    top_3_features = target_corr.head(3).index.tolist()
    insights.append(f"2. Top 3 predictive features: {', '.join(top_3_features)}")

# Insight 3: Skewed distributions
skewed_features = [col for col in key_numerical if abs(df[col].skew()) > 1]
if skewed_features:
    insights.append(
        f"3. Skewed features requiring transformation: {', '.join(skewed_features[:3])}"
    )

# Insight 4: Age trends
if "age" in df.columns and "readmitted_binary" in df.columns:
    age_readmit = df.groupby("age")["readmitted_binary"].mean() * 100
    highest_risk_age = age_readmit.idxmax()
    insights.append(f"4. Highest readmission risk age group: {highest_risk_age}")

# Insight 5: Diagnosis impact
if "diag_1_category" in df.columns and "readmitted_binary" in df.columns:
    diag_readmit = df.groupby("diag_1_category")["readmitted_binary"].mean() * 100
    highest_risk_diag = diag_readmit.idxmax()
    insights.append(f"5. Highest readmission risk diagnosis: {highest_risk_diag}")

print("\n")
for insight in insights:
    print(f"  {insight}")

# ============================================================================
# 7. SAVE EDA REPORT
# ============================================================================


# Create summary report
eda_summary = {
    "Total_Patients": len(df),
    "Numerical_Features": len(numerical_cols),
    "Categorical_Features": len(categorical_cols),
    "Target_Class_0": target_dist[0] if "readmitted_binary" in df.columns else "N/A",
    "Target_Class_1": target_dist[1] if "readmitted_binary" in df.columns else "N/A",
    "Imbalance_Ratio": (
        f"1:{imbalance_ratio:.1f}" if "readmitted_binary" in df.columns else "N/A"
    ),
}

summary_df = pd.DataFrame(list(eda_summary.items()), columns=["Metric", "Value"])
summary_df.to_csv("../reports/03_eda_summary.csv", index=False)
print("\n✓ EDA summary saved as '03_eda_summary.csv'")

# Save insights
insights_df = pd.DataFrame({"Insight": insights})
insights_df.to_csv("../reports/03_key_insights.csv", index=False)
print("✓ Key insights saved as '03_key_insights.csv'")

print("\n" + "=" * 80)
print("STEP 3 COMPLETE!")
print("=" * 80)
print("\nGenerated Files:")
print("  - 03_numerical_distributions.png")
print("  - 03_categorical_distributions.png")
print("  - 03_bivariate_analysis.png")
print("  - 03_correlation_heatmap.png")
print("  - 03_eda_summary.csv")
print("  - 03_key_insights.csv")
print("\nNext Steps:")
print("  1. Review all visualizations")
print("  2. Check key insights CSV")
print("  3. Proceed to Step 4: Feature Engineering")
print("=" * 80)
