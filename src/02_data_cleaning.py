"""
Step 2: Data Preprocessing
Diabetes 130-US Hospitals Dataset

This script:
1. Loads the raw data from Step 1
2. Handles missing values ('?' indicators)
3. Maps ICD-9 diagnosis codes to 9 disease categories
4. Maps ICD diabetes diagnoses codes to related information:
    - 34 diabetes diagnoses descriptions
    - diabetes type
    - diabetes control status
    - diabetes complication (binary and categories)
5. Creation of per day features
6. Handles duplicate patients (multiple encounters)
7. Creates binary target variable
8. Handles outliers and data quality issues
9. Saves cleaned data for analysis
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.api.types import is_numeric_dtype



# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================

print("\n[1] Loading raw data from Step 1...")
try:
    df = pd.read_csv('../data/raw/diabetes_data.csv')
    print(f"✓ Data loaded: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'diabetes_raw_data.csv' not found.")
    print("  Please run Step 1 first to generate the data file.")
    #exit()

initial_rows = len(df)
initial_cols = len(df.columns)


# ============================================================================
# 2. HANDLE MISSING VALUE INDICATORS
# ============================================================================


# Replace '?' with NaN
print("\n[2.1] Converting '?' to NaN...")
question_mark_count = (df == '?').sum().sum()
df.replace('?', np.nan, inplace=True)
print(f"  - Converted {question_mark_count:,} '?' values to NaN")

# Canonicalization for A1Cresult: convert NaN to 'NoTest'
# Check if column exists
if 'A1Cresult' in df.columns:
    print(f"\n[2.2] Canonicalization of A1Cresult:")
    df['A1Cresult'] = df['A1Cresult'].fillna('NoTest')
    no_test_count = (df['A1Cresult'] == 'NoTest').sum()
    print(f"  - Canonicalized A1Cresult: filled NaN to 'NoTest' for {no_test_count:,} rows")

# Display missing values after conversion
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

print(f"\n[2.3] Columns with missing values ({len(missing_summary)} total):")
for col, count in missing_summary.items():
    pct = (count / len(df)) * 100
    print(f"  {col}: {count:,} ({pct:.2f}%)")

# Decision: Drop columns with >50% missing
threshold = 0.39
cols_to_drop = []
for col, count in missing_summary.items():
    pct = count / len(df)
    if pct > threshold:
        cols_to_drop.append(col)

if cols_to_drop:
    print(f"\n[2.4] Dropping columns with >{threshold*100}% missing:")
    for col in cols_to_drop:
        pct = (missing_summary[col] / len(df)) * 100
        print(f"  - {col}: {pct:.2f}% missing")
    df.drop(columns=cols_to_drop, inplace=True)
    print(f"  ✓ Dropped {len(cols_to_drop)} columns")


# ============================================================================
# 3. ICD-9 DIAGNOSIS CODE MAPPING
# ============================================================================



def map_icd9_to_category(code):
    """
    Map ICD-9 diagnosis codes to 9 disease categories.
    Based on Strack et al. (2014) methodology.
    
    Categories:
    - Circulatory: 390-459, 785
    - Respiratory: 460-519, 786
    - Digestive: 520-579, 787
    - Diabetes: 250.xx
    - Injury: 800-999
    - Musculoskeletal: 710-739
    - Genitourinary: 580-629, 788
    - Neoplasms: 140-239
    - Other: Everything else
    """
    
    if pd.isna(code):
        return '0'
    
    # Convert to string
    code_str = str(code).strip()
    
    # Try to extract numeric value
    try:
        # Handle decimal codes like "250.01"
        if '.' in code_str:
            code_num = float(code_str)
        else:
            code_num = float(code_str)
    except (ValueError, TypeError):
        return 'Other'
    
    # Apply ICD-9 range mapping
    if 390 <= code_num <= 459 or code_num == 785:
        return 'Circulatory'
    elif 460 <= code_num <= 519 or code_num == 786:
        return 'Respiratory'
    elif 520 <= code_num <= 579 or code_num == 787:
        return 'Digestive'
    elif 250 <= code_num < 251:  # All 250.xx codes
        return 'Diabetes'
    elif 800 <= code_num <= 999:
        return 'Injury'
    elif 710 <= code_num <= 739:
        return 'Musculoskeletal'
    elif 580 <= code_num <= 629 or code_num == 788:
        return 'Genitourinary'
    elif 140 <= code_num <= 239:
        return 'Neoplasms'
    else:
        return 'Other'

# Apply mapping to all three diagnosis columns
diag_cols = ['diag_1', 'diag_2', 'diag_3']
for col in diag_cols:
    if col in df.columns:
        new_col = f"{col}_category"
        print(f"\n[3.{diag_cols.index(col)+1}] Mapping {col}...")
        
        # Show before mapping
        unique_before = df[col].nunique()
        print(f"  Before: {unique_before} unique codes")
        
        # Apply mapping
        df[new_col] = df[col].apply(map_icd9_to_category)
        
        # Show after mapping
        unique_after = df[new_col].nunique()
        print(f"  After: {unique_after} categories")
        
        # Show category distribution
        print(f"  Category distribution:")
        category_counts = df[new_col].value_counts()
        for cat, count in category_counts.items():
            pct = (count / len(df)) * 100
            print(f"    {cat}: {count:,} ({pct:.2f}%)")

print("\n✓ ICD-9 mapping complete: 700+ codes → 9 categories per diagnosis")


# ============================================================================
# 4. ICD DIABETES DIAGNOSIS CODE MAPPING
# ============================================================================
# codes retrived from https://www.aapc.com/codes/icd9-codes/250.93


def create_get_value_fn(mapping):
    def get_value(row, default_value:str = "250"):
        for val in [row['diag_1'], row['diag_2'], row['diag_3']]:
            if val in list(mapping.keys()):
                return val
        return default_value
    return get_value

# get diabetes diagnosis description based on diagnosis code
with open('../data/preprocessing-codes-mapping/diabetes_description_based_on_code.json', 'r') as fp:
    diab_descr_dict = json.load(fp)

# get diabetes type based on diagnosis code
with open('../data/preprocessing-codes-mapping/diabetes_type_based_on_code.json', 'r') as fp:
    diab_type_dict = json.load(fp)

# get diabetes control status based on diagnosis code
# 1: diabetes not stated as uncontrolled
# 0: diabetes stated as uncontrolled
diab_control_dict = {key: 1 if "not stated as uncontrolled" in value else 0
                     for key, value in diab_descr_dict.items()}

# get diabetes complication existence based on diagnosis code
# 1: there is a complication
# 0: there is no complication
diab_complications_binary_dict = {key: 1 if " with " in value else 0
                                  for key, value in diab_descr_dict.items()}

# get diabetes complication category based on diagnosis code
# if no complication is mentionned, "None"
diab_complications_categories_dict = {
    key: value.split("with")[1].split(", ")[0].strip() if " with " in value else "None"
    for key, value in diab_descr_dict.items()
    }

# apply maps for all additionnal features
df['diab_code'] = df[["diag_1", 'diag_2', "diag_3"]].apply(create_get_value_fn(diab_descr_dict), axis=1)

df['diab_type'] = df['diab_code'].map(diab_type_dict.copy()).astype(int)

df['diab_control'] = df['diab_code'].map(diab_control_dict).astype(int)

df['diab_complication_binary'] = df['diab_code'].map(diab_complications_binary_dict).astype(int)

df['diab_complication_categories'] = df['diab_code'].map(diab_complications_categories_dict).astype(str)

print("\n✓ ICD-diabetes information mapping complete")


# ============================================================================
# 5. PER DAY FEATURES
# ============================================================================
# Interaction 1: Medications × Time in Hospital
if 'num_medications' in df.columns and 'time_in_hospital' in df.columns:
    df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'].apply(lambda x: max(x, 1)))  # +1 to avoid division by zero
    print("  ✓ Created: meds_per_day (medications per hospital day)")

# Interaction 2: Procedures × Time in Hospital
if 'num_procedures' in df.columns and 'time_in_hospital' in df.columns:
    df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'].apply(lambda x: max(x, 1)))
    print("  ✓ Created: procedures_per_day")

# Interaction 3: Lab tests × Time in Hospital
if 'num_lab_procedures' in df.columns and 'time_in_hospital' in df.columns:
    df['labs_per_day'] = df['num_lab_procedures'] / (df['time_in_hospital'].apply(lambda x: max(x, 1)))
    print("  ✓ Created: labs_per_day")


# ============================================================================
# 6. CREATE BINARY TARGET VARIABLE
# ============================================================================



if 'readmitted' in df.columns:
    print("\n[5.1] Converting to binary classification...")
    print("  Readmitted <30 days = 1, Otherwise = 0")
    
    # Create binary target
    df['readmitted_binary'] = (df['readmitted'] == '<30').astype(int)
    
    # Show distribution
    target_counts = df['readmitted_binary'].value_counts()
    print(f"\n  Class distribution:")
    for cls, count in target_counts.items():
        pct = (count / len(df)) * 100
        label = "Readmitted <30" if cls == 1 else "Not readmitted <30"
        print(f"    Class {cls} ({label}): {count:,} ({pct:.2f}%)")
    
    # Calculate imbalance ratio
    if 1 in target_counts.index and 0 in target_counts.index:
        ratio = target_counts[0] / target_counts[1]
        print(f"\n  ⚠ Imbalance ratio: 1:{ratio:.1f}")
        print(f"    → Will need SMOTE or class weighting in modeling phase")

# ============================================================================
# 7. HANDLE REMAINING MISSING VALUES
# ============================================================================

# Define known categorical columns (use this list instead of relying solely on dtype)
likely_cats = {
    'race','gender','age','admission_type_id','discharge_disposition_id','admission_source_id',
    'medical_specialty','A1Cresult','change','diabetesMed','insulin',
    'diab_type','diab_control','diab_complication_binary','diab_complication_categories',
    'metformin','repaglinide','nateglinide','chlorpropamide','glimepiride',
    'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone','rosiglitazone',
    'acarbose','miglitol','troglitazone','tolazamide','examide','citoglipton'
}

# Check remaining missing values
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if len(remaining_missing) > 0:
    print(f"\n[6.1] Columns still with missing values: {len(remaining_missing)}")
    
    for col in remaining_missing.index:
        count = remaining_missing[col]
        pct = (count / len(df)) * 100
        
        print(f"\n  Processing: {col} ({count:,} missing, {pct:.2f}%)")
        
        if col.startswith('diag_'):
            # For diagnostic columns: replace missing with "0" (no diagnosis)
            df[col] = df[col].where(df[col].notna(), "0")
            diag_cols = [c for c in df.columns if c.startswith('diag_')]
            print('Number of no diagnoses for diag_1, diag_2, diag_3 in total: ', df[diag_cols].isna().sum())

        # Treat explicitly listed categorical columns as categorical even if dtype is numeric
        elif col in likely_cats:
            # Categorical: impute with mode or 'Unknown'
            if df[col].mode().empty:
                df[col].fillna('Unknown', inplace=True)
                print(f"    → Imputed '{col}' with: 'Unknown'")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"    → Imputed '{col}' with mode: '{mode_val}'")

        # Numeric columns: use median imputation
        elif is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    → '{col}' imputed with median: {median_val}")

        else:
            # Fallback: treat as categorical
            if df[col].mode().empty:
                df[col].fillna('Unknown', inplace=True)
                print(f"    → Imputed '{col}' with: 'Unknown' (fallback)")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"    → Imputed '{col}' with mode: '{mode_val}' (fallback)")
    
    print(f"\n✓ All missing values handled")
else:
    print("\n✓ No remaining missing values")


# ============================================================================
# 7. Remove Duplicate Patient Encounters 
# ============================================================================
# we need independent variables for our models
df = df.sort_values(["patient_nbr", "encounter_id"])
#  Keep only the first encounter per patient 
df = df.groupby("patient_nbr", as_index=False).first()

# Drop identifier columns that add no information for modeling
drop_ids = [c for c in ['patient_nbr', 'encounter_id'] if c in df.columns]
if drop_ids:
    df.drop(columns=drop_ids, inplace=True)
    print(f"\n✓ Dropped identifier columns: {', '.join(drop_ids)}")


# ============================================================================
# DATA QUALITY SUMMARY
# ============================================================================



print(f"\nInitial dataset:")
print(f"  - Rows: {initial_rows:,}")
print(f"  - Columns: {initial_cols}")

print(f"\nFinal dataset:")
print(f"  - Rows: {len(df):,} ({((len(df)-initial_rows)/initial_rows*100):+.2f}%)")
print(f"  - Columns: {len(df.columns)} ({len(df.columns)-initial_cols:+d})")

print(f"\nChanges made:")
print(f"  ✓ Converted '?' to NaN")
print(f"  ✓ Dropped {len(cols_to_drop)} high-missing columns")
print(f"  ✓ Mapped diagnosis codes to categories")
print(f"  ✓ Removed duplicate patient encounters")
print(f"  ✓ Created binary target variable")
print(f"  ✓ Imputed remaining missing values")

# Verify no missing values remain
total_missing = df.isnull().sum().sum()
print(f"\nFinal missing values: {total_missing}")

# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================



# Save cleaned dataset
df.to_csv('../data/processed/diabetes_cleaned_data.csv', index=False)
print("\n✓ Cleaned data saved as 'diabetes_cleaned_data.csv'")

# Save preprocessing report
preprocessing_report = {
    'Initial_Rows': initial_rows,
    'Final_Rows': len(df),
    'Rows_Removed': initial_rows - len(df),
    'Initial_Columns': initial_cols,
    'Final_Columns': len(df.columns),
    'Columns_Dropped': len(cols_to_drop),
    'Missing_Values_Imputed': question_mark_count,
    'Final_Missing_Values': total_missing
}

report_df = pd.DataFrame(list(preprocessing_report.items()), 
                         columns=['Metric', 'Value'])
report_df.to_csv('../reports/02_preprocessing_report.csv', index=False)
print("✓ Preprocessing report saved as '02_preprocessing_report.csv'")
print("\n" + "="*80)
print("STEP 2 COMPLETE!")
print("="*80)
print("\nNext Steps:")
print("  1. Review 'diabetes_cleaned_data.csv'")
print("  2. Check '02_preprocessing_report.csv' for summary")
print("  3. Proceed to Step 3: Exploratory Data Analysis")
print("="*80)
