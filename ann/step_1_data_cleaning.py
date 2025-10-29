"""
Step 2: Data Preprocessing
Diabetes 130-US Hospitals Dataset

This script:
1. Loads the raw data from Step 1
1.1 Adding dictionary from IDS mapping
1.2 Dropping patients that do not return
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
import pickle


# ============================================================================
# 1. LOAD RAW DATA
# ============================================================================


df = pd.read_csv("data/raw/diabetes_data.csv")

print(df.shape)
print(df.info())


# ============================================================================
# 1.1 Mapping Dicts from IDS_mapping
# ============================================================================


# --- 1. Build mapping dicts (from IDS_mapping.csv content) ---

df["admission_type_id"] = df["admission_type_id"].replace({5: 9, 6: 9, 8: 9})

admission_type_map = {
    1: "Emergency",
    2: "Urgent",
    3: "Elective",
    4: "Newborn",
    7: "Trauma Center",
    9: "Not available",
}

df["admission_type_desc"] = df["admission_type_id"].map(admission_type_map)
# the admission ID will be removed for modeling


discharge_disposition_map = {
    1: "Home",
    2: "Transfer Short-term Hospital",
    3: "Transfer SNF",
    4: "Transfer ICF",
    5: "Transfer Other Inpatient Facility",
    6: "Home Health Service",
    7: "Left AMA",
    8: "Home IV Care",
    9: "Admitted as Inpatient Here",
    10: "Neonate Aftercare Transfer",
    11: "Expired",
    12: "Still Patient / Outpatient Followup Planned",
    13: "Hospice Home",
    14: "Hospice Medical Facility",
    15: "Swing Bed (Medicare)",
    16: "Transfer for Outpatient Services (Other Institution)",
    17: "Transfer for Outpatient Services (This Institution)",
    18: "NULL",
    19: "Expired at Home (Hospice)",
    20: "Expired in Facility (Hospice)",
    21: "Expired - Place Unknown (Hospice)",
    22: "Rehab Facility",
    23: "Long Term Care Hospital",
    24: "Nursing Facility (Medicaid only)",
    25: "Not Mapped",
    26: "Unknown/Invalid",
    27: "Federal Health Care Facility",
    28: "Psych Hospital / Psych Unit",
    29: "Critical Access Hospital",
    30: "Other Health Care Institution",
}

df["discharge_disposition_desc"] = df["discharge_disposition_id"].map(
    discharge_disposition_map
)
# discharge will be removed all toghether so no need to replace


df["admission_source_id"] = df["admission_source_id"].replace(
    {9: 27, 15: 27, 17: 27, 20: 27}
)

admission_source_map = {
    1: "Physician Referral",
    2: "Clinic Referral",
    3: "HMO Referral",
    4: "Transfer from Hospital",
    5: "Transfer from SNF",
    6: "Transfer from Other Facility",
    7: "Emergency Room",
    8: "Court/Law Enforcement",
    10: "Transfer from critical access hospital",
    11: "Normal Delivery",
    12: "Premature Delivery",
    13: "Sick Baby",
    14: "Extramural Birth",
    18: "Transfer From Another Home Health Agency",
    19: "Readmission to Same Home Health Agency",
    21: "Unknown/Invalid",
    22: "Transfer from Hospital Inpatient / Same Facility",
    23: "Born Inside This Hospital",
    24: "Born Outside This Hospital",
    25: "Transfer from Ambulatory Surgery Center",
    26: "Transfer from Hospice",
    27: "Not available",
}


df["admission_source_desc"] = df["admission_source_id"].map(admission_source_map)

# --- 3. Create a flag for rows that should be excluded from training
# (patients who died or went to hospice cannot be "readmitted")

dead_or_hospice_codes = [11, 13, 14, 19, 20, 21]
df["died_or_hospice"] = (
    df["discharge_disposition_id"].isin(dead_or_hospice_codes).astype(int)
)

# ============================================================================
# 1.2. Droping:  expired or hospice
# ============================================================================


dead_or_hospice_codes = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(dead_or_hospice_codes)].copy()


# ============================================================================
# 1.3. Droping citoglipton and examide -- all values are the same
# ============================================================================
# these 2 variables have only 1 value "no"

columns_to_drop = []

columns_to_drop.extend(["citoglipton", "examide"])

# droping ID columns admision-id, admision source-id, discharge

# ============================================================================
# 1.3. # droping ID columns admision-id, admision source-id, discharge
# ============================================================================

columns_to_drop.extend(
    [
        "admission_source_id",
        "admission_type_id",
        "discharge_disposition_desc",
        "discharge_disposition_id",
    ]
)


# ============================================================================
# 2. HANDLE MISSING VALUE INDICATORS
# ============================================================================


# Replace '?' with NaN

question_mark_count = (df == "?").sum().sum()
df.replace("?", np.nan, inplace=True)
print(f"  - Converted {question_mark_count:,} '?' values to NaN")

# Canonicalization for A1Cresult: convert NaN to 'NoTest'
# Check if column exists

df["A1Cresult"] = df["A1Cresult"].fillna("NoTest")
# no_test_count = (df['A1Cresult'] == 'NoTest').sum()
#    print(f"filled NaN to 'NoTest' for {no_test_count:,} rows")

# Display missing values after conversion
missing_summary = df.isnull().sum()
missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)

print(f"\n[2.3] Columns with missing values ({len(missing_summary)} total):")
for col, count in missing_summary.items():
    pct = (count / len(df)) * 100
    print(f"  {col}: {count:,} ({pct:.2f}%)")

# Decision: Drop columns with >39% missing


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
        return "0"

    # Convert to string
    code_str = str(code).strip()

    # Try to extract numeric value
    try:
        # Handle decimal codes like "250.01"
        if "." in code_str:
            code_num = float(code_str)
        else:
            code_num = float(code_str)
    except (ValueError, TypeError):
        return "Other"

    # Apply ICD-9 range mapping
    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    elif 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    elif 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    elif 250 <= code_num < 251:  # All 250.xx codes
        return "Diabetes"
    elif 800 <= code_num <= 999:
        return "Injury"
    elif 710 <= code_num <= 739:
        return "Musculoskeletal"
    elif 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    elif 140 <= code_num <= 239:
        return "Neoplasms"
    else:
        return "Other"


# Apply mapping to all three diagnosis columns
diag_cols = ["diag_1", "diag_2", "diag_3"]
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
    def get_value(row, default_value: str = "250"):
        for val in [row["diag_1"], row["diag_2"], row["diag_3"]]:
            if val in list(mapping.keys()):
                return val
        return default_value

    return get_value


# get diabetes diagnosis description based on diagnosis code
with open(
    "data/preprocessing-codes-mapping/diabetes_description_based_on_code.json", "r"
) as fp:
    diab_descr_dict = json.load(fp)

# get diabetes type based on diagnosis code
with open(
    "data/preprocessing-codes-mapping/diabetes_type_based_on_code.json", "r"
) as fp:
    diab_type_dict = json.load(fp)

# get diabetes control status based on diagnosis code
# 1: diabetes not stated as uncontrolled
# 0: diabetes stated as uncontrolled
diab_control_dict = {
    key: 1 if "not stated as uncontrolled" in value else 0
    for key, value in diab_descr_dict.items()
}

# get diabetes complication existence based on diagnosis code
# 1: there is a complication
# 0: there is no complication
diab_complications_binary_dict = {
    key: 1 if " with " in value else 0 for key, value in diab_descr_dict.items()
}

# get diabetes complication category based on diagnosis code
# if no complication is mentionned, "None"
diab_complications_categories_dict = {
    key: value.split("with")[1].split(", ")[0].strip() if " with " in value else "None"
    for key, value in diab_descr_dict.items()
}

# apply maps for all additionnal features
df["diab_code"] = df[["diag_1", "diag_2", "diag_3"]].apply(
    create_get_value_fn(diab_descr_dict), axis=1
)

df["diab_type"] = df["diab_code"].map(diab_type_dict.copy()).astype(int)

df["diab_control"] = df["diab_code"].map(diab_control_dict).astype(int)

df["diab_complication_binary"] = (
    df["diab_code"].map(diab_complications_binary_dict).astype(int)
)

df["diab_complication_categories"] = (
    df["diab_code"].map(diab_complications_categories_dict).astype(str)
)

print("\n✓ ICD-diabetes information mapping complete")


# ============================================================================
# 5. PER DAY FEATURES
# ============================================================================
# Interaction 1: Medications × Time in Hospital

df["meds_per_day"] = df["num_medications"] / (
    df["time_in_hospital"].apply(lambda x: max(x, 1))
)  # +1 to avoid division by zero
print("  ✓ Created: meds_per_day (medications per hospital day)")

# Interaction 2: Procedures × Time in Hospital

df["procedures_per_day"] = df["num_procedures"] / (
    df["time_in_hospital"].apply(lambda x: max(x, 1))
)
print("  ✓ Created: procedures_per_day")

# Interaction 3: Lab tests × Time in Hospital

df["labs_per_day"] = df["num_lab_procedures"] / (
    df["time_in_hospital"].apply(lambda x: max(x, 1))
)
print("  ✓ Created: labs_per_day")


# ============================================================================
# 7. HANDLE REMAINING MISSING VALUES
# ============================================================================


# Check remaining missing values
remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]


if len(remaining_missing) > 0:
    print(f"\n[6.1] Columns still with missing values: {len(remaining_missing)}")

    for col in remaining_missing.index:
        count = remaining_missing[col]
        pct = (count / len(df)) * 100

        print(f"\n  Processing: {col} ({count:,} missing, {pct:.2f}%)")

        if col.startswith("diag_"):
            # For diagnostic columns: replace '?' with None (already handled above)
            df[col] = df[col].where(df[col].notna(), "0")
            diag_cols = [c for c in df.columns if c.startswith("diag_")]
            print(
                "Number of no diagnoses for diag_1, diag_2, diag_3 in total: ",
                df[diag_cols].isna().sum(),
            )

        elif df[col].dtype in ["int64", "float64"]:
            # Numerical: impute with median
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"    → '{col}' imputed with median: {median_val}")
        else:
            # Categorical: impute with mode or 'Unknown'
            if df[col].mode().empty:
                df[col].fillna("Unknown", inplace=True)
                print(f"    → Imputed with: 'Unknown'")
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val, inplace=False)
                print(f"    → Imputed with mode: '{mode_val}'")

    print(f"\n✓ All missing values handled")
else:
    print("\n✓ No remaining missing values")


# save data for the initial trial
# df.to_pickle("data/processed/allrows_diabetes_cleaned.pkl")

# ============================================================================
# 7. Remove Duplicate Patient Encounters
# ============================================================================
# we need independent variables for our models
# df = df.sort_values(["patient_nbr", "encounter_id"])
# #  Keep only the first encounter per patient
# df = df.groupby("patient_nbr", as_index=False).first()

# ============================================================================
# 7A. AND CLEAN SOME MORE
# ============================================================================
import os

out_dir = "figures/outliers"
os.makedirs(out_dir, exist_ok=True)

# remove patients under 18  and over 90
# ---------------------------------AGE - outliers

plt.boxplot(df["age"].str.extract("(\d+)").astype(int))
plt.title("Age distribution")
plt.ylabel("Age lower bound")
plt.show()
plt.savefig(os.path.join(out_dir, "encounters_boxplotboxplot_age.png"), dpi=300)
plt.close()


print(df["age"].value_counts().sort_index())
df = df[~df["age"].isin(["[0-10)", "[10-20)", "[90-100)"])].copy()

# ----------------------------encountes < 13 - keep

encounters_per_patient = df.groupby("patient_nbr")["encounter_id"].nunique()

# Create a summary: how many patients have 1, 2, 3, ... encounters
encounter_distribution = encounters_per_patient.value_counts().sort_index()

print(encounter_distribution)

encounter_distribution.plot(kind="bar", figsize=(10, 4))
plt.xlabel("Number of encounters per patient")
plt.ylabel("Number of patients")
plt.title("Distribution of patient encounter counts")
plt.tight_layout()
plt.show()
plt.savefig(os.path.join(out_dir, "barplot encounters_age.png"), dpi=300)
plt.close()


plt.figure(figsize=(6, 4))
plt.boxplot(encounters_per_patient, vert=True)
plt.ylabel("Number of encounters per patient")
plt.title("Boxplot of patient encounter counts")
plt.show()
plt.savefig(os.path.join(out_dir, "boxplot_encounters_age.png"), dpi=300)
plt.close()

# keep patients with < 13 encounters (remove outliers)

encounters_per_patient_grouped = df.groupby("patient_nbr")["encounter_id"].nunique()
patients_to_keep = encounters_per_patient_grouped[encounters_per_patient <= 13].index

df = df[df["patient_nbr"].isin(patients_to_keep)].copy()


# -----------------------time in the hospital--------- good - no putliers

hospital_days_count = df["time_in_hospital"].value_counts().sort_index()

plt.figure(figsize=(7, 4))
hospital_days_count.plot(kind="bar")
plt.xlabel("Days in hospital")
plt.ylabel("Number of patients")
plt.title("Distribution of hospital stay duration")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "time_in_hospital_distribution.png"), dpi=300)
plt.close()

plt.figure(figsize=(6, 4))
plt.boxplot(df["time_in_hospital"], vert=True)
plt.ylabel("Days in hospital")
plt.title("Boxplot of hospital stay duration")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "time_in_hospital_boxplot.png"), dpi=300)
plt.close()

# -------------------------------------multiple cols  --------


cols_main = ["num_lab_procedures", "num_procedures", "num_medications"]
cols_ratio = ["meds_per_day", "labs_per_day", "procedures_per_day"]
cols_extra = ["number_diagnoses"]

procedure_counts = df["num_procedures"].value_counts().sort_index()
procedure_summary = procedure_counts.reset_index()
procedure_summary.columns = ["procedures", "patients"]
print(procedure_summary)
# nothing to drop


procedure_counts = df["num_lab_procedures"].value_counts().sort_index()
procedure_summary = procedure_counts.reset_index()
procedure_summary.columns = ["procedures", "patients"]
print(procedure_summary)
# will drop all lab procedures after 99(outliers)
df = df[df["num_lab_procedures"] <= 99].copy()


procedure_counts = df["num_medications"].value_counts().sort_index()
procedure_summary = procedure_counts.reset_index()
procedure_summary.columns = ["procedures", "patients"]
# will drop mode than 63 medications -
df = df[df["num_medications"] <= 63].copy()

procedure_counts = df["number_diagnoses"].value_counts().sort_index()
procedure_summary = procedure_counts.reset_index()
procedure_summary.columns = ["procedures", "patients"]
# will drop mode than 9 and less than 2
df = df[df["number_diagnoses"] <= 9].copy()
df = df[df["number_diagnoses"] >= 2].copy()

# ----------------- plot figures with outlier in case needed for presentation


for col in cols_main:
    plt.figure(figsize=(6, 4))
    plt.boxplot(df[col])
    plt.title(f"Outliers: {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{col}_boxplot.png"), dpi=300)
    plt.close()


for col in cols_ratio:
    if col in df.columns:
        plt.figure(figsize=(6, 4))
        plt.boxplot(df[col])
        plt.title(f"Outliers: {col}")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{col}_boxplot.png"), dpi=300)
        plt.close()


plt.figure(figsize=(6, 4))
plt.boxplot(df["number_diagnoses"])
plt.title("Outliers: Number of diagnoses")
plt.ylabel("Diagnoses count")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "number_diagnoses_boxplot.png"), dpi=300)
plt.close()

len(df)


# save data for trial nuber 2
df.to_pickle("data/processed/trimr_diabetes_cleaned.pkl")

print(df.shape)
print(df.info())


# ============================================================================
# 9. SAVE CLEANED DATA
# ============================================================================

columns_to_drop

# save data for trial nuber 2
df.to_pickle("data/processed/trimr_diabetes_cleaned.pkl")


# Save cleaned dataset
# df.to_csv('../data/processed/diabetes_cleaned_data.csv', index=False)
# print("\n✓ Cleaned data saved as 'diabetes_cleaned_data.csv'")
