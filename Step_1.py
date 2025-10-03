"""
Step 1: Data Acquisition and Initial Exploration
Diabetes 130-US Hospitals Dataset

This script:
1. Loads the dataset from UCI ML Repository
2. Performs initial data exploration
3. Identifies data quality issues
4. Generates summary statistics and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
sns.set_style("whitegrid")

print("="*80)
print("DIABETES HOSPITAL READMISSION - DATA ACQUISITION & EXPLORATION")
print("="*80)

# === STEP 1: Load dataset ===
# Fetch dataset (ID: 296)

print("\n[1] Loading dataset from UCI ML Repository...")
try:
    diabetes_data = fetch_ucirepo(id=296)
    
    # Extract features and targets
    X = diabetes_data.data.features
    y = diabetes_data.data.targets
    
    # Combine into single dataframe
    df = pd.concat([X, y], axis=1)
    
    print("✓ Dataset loaded successfully!")
    print(f"  - Shape: {df.shape}")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Records: {len(df)}")
    
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    print("\nNote: Make sure you have installed ucimlrepo:")
    print("  pip install ucimlrepo")
    exit()

# ============================================================================
# 2. INITIAL DATA INSPECTION
# ============================================================================

print("\n" + "="*80)
print("[2] INITIAL DATA INSPECTION")
print("="*80)

# Display first few rows
print("\n[2.1] First 5 rows:")
print(df.head())

# Display column names and types
print("\n[2.2] Column Information:")
print(f"\nTotal Columns: {len(df.columns)}")
print("\nColumn Names and Types:")
print(df.dtypes)

# Display basic statistics
print("\n[2.3] Numerical Features - Summary Statistics:")
print(df.describe())

print("\n[2.4] Categorical Features - Unique Values:")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols[:10]:  # Show first 10 categorical columns
    print(f"  {col}: {df[col].nunique()} unique values")
    if df[col].nunique() <= 10:
        print(f"    Values: {df[col].unique()}")

# ============================================================================
# 3. TARGET VARIABLE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("[3] TARGET VARIABLE ANALYSIS")
print("="*80)

if 'readmitted' in df.columns:
    print("\n[3.1] Readmission Distribution:")
    readmit_counts = df['readmitted'].value_counts()
    readmit_pct = df['readmitted'].value_counts(normalize=True) * 100
    
    readmit_summary = pd.DataFrame({
        'Count': readmit_counts,
        'Percentage': readmit_pct.round(2)
    })
    print(readmit_summary)
    
    # Create binary target for <30 days readmission
    print("\n[3.2] Creating Binary Target (<30 days readmission):")
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    
    binary_counts = df['readmitted_binary'].value_counts()
    binary_pct = df['readmitted_binary'].value_counts(normalize=True) * 100
    
    binary_summary = pd.DataFrame({
        'Count': binary_counts,
        'Percentage': binary_pct.round(2)
    })
    print(binary_summary)
    
    print("\n  ⚠ Class Imbalance Detected:")
    print(f"    - Class 0 (Not readmitted <30): {binary_pct[0]:.2f}%")
    print(f"    - Class 1 (Readmitted <30): {binary_pct[1]:.2f}%")
    print(f"    - Imbalance Ratio: 1:{(binary_pct[0]/binary_pct[1]):.1f}")


# ============================================================================
# 4. MISSING VALUES ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("[4] MISSING VALUES ANALYSIS")
print("="*80)

# Calculate missing values
missing_counts = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df)) * 100

missing_df = pd.DataFrame({
    'Missing_Count': missing_counts,
    'Missing_Percentage': missing_pct
})

# Filter columns with missing values
missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
    'Missing_Percentage', ascending=False
)

if len(missing_df) > 0:
    print(f"\n[4.1] Columns with Missing Values ({len(missing_df)} total):")
    print(missing_df)
else:
    print("\n✓ No missing values detected!")

# Check for '?' as missing indicator
print("\n[4.2] Checking for '?' as missing value indicator:")
question_mark_cols = []
for col in df.select_dtypes(include=['object']).columns:
    if '?' in df[col].values:
        count = (df[col] == '?').sum()
        pct = (count / len(df)) * 100
        question_mark_cols.append({
            'Column': col,
            'Count': count,
            'Percentage': pct
        })

if question_mark_cols:
    qm_df = pd.DataFrame(question_mark_cols).sort_values('Percentage', ascending=False)
    print(qm_df)
    print("\n  ⚠ Found '?' values - these need to be treated as missing!")
else:
    print("✓ No '?' values found")

# ============================================================================
# 5. DIAGNOSIS CODES ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("[5] DIAGNOSIS CODES ANALYSIS")
print("="*80)

diag_cols = ['diag_1', 'diag_2', 'diag_3']
for col in diag_cols:
    if col in df.columns:
        print(f"\n[5.{diag_cols.index(col)+1}] {col}:")
        print(f"  - Unique codes: {df[col].nunique()}")
        print("  - Most common codes:")
        print(df[col].value_counts().head(10))

print("\n  ℹ Note: These ICD-9 codes will be mapped to 9 disease categories in Step 2")
print("  Categories: Circulatory, Diabetes, Digestive, Genitourinary,")
print("             Injury, Musculoskeletal, Neoplasms, Other, Respiratory")
print("  Mapping based on: Strack et al. (2014), Table 2")

# ============================================================================
# 6. DUPLICATE PATIENT ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("[6] DUPLICATE PATIENT ANALYSIS")
print("="*80)

if 'patient_nbr' in df.columns:
    unique_patients = df['patient_nbr'].nunique()
    total_encounters = len(df)
    duplicate_encounters = total_encounters - unique_patients
    
    print("\n[6.1] Patient Encounter Summary:")
    print(f"  - Total encounters: {total_encounters:,}")
    print(f"  - Unique patients: {unique_patients:,}")
    print(f"  - Duplicate encounters: {duplicate_encounters:,}")
    print(f"  - Average encounters per patient: {total_encounters/unique_patients:.2f}")
    
    # Find patients with multiple encounters
    encounter_counts = df['patient_nbr'].value_counts()
    multiple_encounters = encounter_counts[encounter_counts > 1]
    
    print("\n[6.2] Patients with Multiple Encounters:")
    print(f"  - Patients with >1 encounter: {len(multiple_encounters):,}")
    print(f"  - Max encounters for one patient: {encounter_counts.max()}")
    
    print("\n  ⚠ Action Required: Keep only one encounter per patient")
    print("    Strategy: Select encounter with longest time_in_hospital")

# ============================================================================
# 7. BASIC VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("[7] GENERATING VISUALIZATIONS")
print("="*80)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Diabetes Hospital Readmission - Initial Data Exploration', 
             fontsize=16, fontweight='bold')

# Plot 1: Target Variable Distribution
if 'readmitted' in df.columns:
    ax1 = axes[0, 0]
    readmit_counts.plot(kind='bar', ax=ax1, color=['#2ecc71', '#e74c3c', '#f39c12'])
    ax1.set_title('Readmission Status Distribution', fontweight='bold')
    ax1.set_xlabel('Readmission Category')
    ax1.set_ylabel('Count')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add percentage labels
    for i, (idx, val) in enumerate(readmit_counts.items()):
        pct = (val / len(df)) * 100
        ax1.text(i, val, f'{pct:.1f}%', ha='center', va='bottom')

# Plot 2: Age Distribution
if 'age' in df.columns:
    ax2 = axes[0, 1]
    age_order = sorted(df['age'].unique())
    df['age'].value_counts()[age_order].plot(kind='bar', ax=ax2, color='#3498db')
    ax2.set_title('Age Distribution', fontweight='bold')
    ax2.set_xlabel('Age Group')
    ax2.set_ylabel('Count')
    ax2.tick_params(axis='x', rotation=45)

# Plot 3: Time in Hospital Distribution
if 'time_in_hospital' in df.columns:
    ax3 = axes[1, 0]
    df['time_in_hospital'].hist(bins=14, ax=ax3, color='#9b59b6', edgecolor='black')
    ax3.set_title('Time in Hospital Distribution', fontweight='bold')
    ax3.set_xlabel('Days')
    ax3.set_ylabel('Frequency')
    ax3.axvline(df['time_in_hospital'].median(), color='red', 
                linestyle='--', linewidth=2, label=f'Median: {df["time_in_hospital"].median():.1f}')
    ax3.legend()

# Plot 4: Number of Medications Distribution
if 'num_medications' in df.columns:
    ax4 = axes[1, 1]
    df['num_medications'].hist(bins=30, ax=ax4, color='#e67e22', edgecolor='black')
    ax4.set_title('Number of Medications Distribution', fontweight='bold')
    ax4.set_xlabel('Number of Medications')
    ax4.set_ylabel('Frequency')
    ax4.axvline(df['num_medications'].median(), color='red', 
                linestyle='--', linewidth=2, label=f'Median: {df["num_medications"].median():.1f}')
    ax4.legend()

plt.tight_layout()
plt.savefig('01_initial_exploration.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as '01_initial_exploration.png'")


# ============================================================================
# 8. SAVE PROCESSED DATA
# ============================================================================

print("\n" + "="*80)
print("[8] SAVING DATA")
print("="*80)

# Save raw data
df.to_csv('diabetes_raw_data.csv', index=False)
print("\n✓ Raw data saved as 'diabetes_raw_data.csv'")

# Create summary report
summary_stats = {
    'Total Records': len(df),
    'Total Features': len(df.columns),
    'Numerical Features': len(df.select_dtypes(include=[np.number]).columns),
    'Categorical Features': len(df.select_dtypes(include=['object']).columns),
    'Missing Values': df.isnull().sum().sum(),
    'Duplicate Patients': duplicate_encounters if 'patient_nbr' in df.columns else 'N/A',
    'Readmission <30 days (%)': binary_pct[1] if 'readmitted' in df.columns else 'N/A'
}

summary_df = pd.DataFrame(list(summary_stats.items()), columns=['Metric', 'Value'])
summary_df.to_csv('01_data_summary.csv', index=False)
print("✓ Summary statistics saved as '01_data_summary.csv'")

print("\n" + "="*80)
print("STEP 1 COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("  1. Review the generated visualizations")
print("  2. Examine '01_data_summary.csv' for quick reference")
print("  3. Proceed to Step 2: Data Preprocessing")
print("="*80)