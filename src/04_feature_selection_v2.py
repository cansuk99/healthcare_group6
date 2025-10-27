"""
Step 4: Feature Engineering and Selection
Diabetes 130-US Hospitals Dataset

This script:
1. Loads cleaned data from Step 2
2. Creates interaction features
3. Creates polynomial features for key variables
4. Creates binary indicator features
5. Encodes categorical variables
6. Performs feature selection using multiple methods
7. Prepares final feature set for modeling
"""

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.3f} seconds")
        return result
    return wrapper



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier
import re


# ============================================================================
# 1. LOAD CLEANED DATA
# ============================================================================

print("\n[1] Loading cleaned data...")
# try:
#     df = pd.read_csv('../data/processed/diabetes_cleaned_data.csv')
#     print(f"✓ Data loaded: {df.shape}")
#     ids_raw = pd.read_csv('../data/raw/IDS_mapping.csv', header=None, dtype=str, keep_default_na=False)
#     print(f"✓ IDs mapping loaded: {ids_raw.shape}")
# except FileNotFoundError:
#     print("✗ Error: 'diabetes_cleaned_data.csv' not found.")
#     print("  Please run Step 2 first.")
#     #exit()

# if 'readmitted_binary' not in df.columns:
#     print("✗ Error: Target variable 'readmitted_binary' not found.")
   # exit()

df = pd.read_pickle("data/processed/diabetes_cleaned.pkl")

original_features = len(df.columns) - 1
original_features
# ============================================================================
# 2. CREATE INTERACTION FEATURES
# ============================================================================
# per day feature was moved to STEP 2
#                 - medications per hospital day
#                 - procedures_per_day
#                 - Created: labs_per_day

print("\n[2.1] Creating clinically meaningful interactions...")

# Interaction 4: Total prior visits
prior_visit_cols = ['number_outpatient', 'number_emergency', 'number_inpatient']
if all(col in df.columns for col in prior_visit_cols):
    df['total_prior_visits'] = df[prior_visit_cols].sum(axis=1)
    print("  ✓ Created: total_prior_visits (sum of all prior visits)")

# Interaction 5: High utilization flag
if 'total_prior_visits' in df.columns:
    df['high_utilizer'] = (df['total_prior_visits'] > df['total_prior_visits'].quantile(0.75)).astype(int)
    print("  ✓ Created: high_utilizer (top 25% of prior visits)")

interaction_features = ['meds_per_day', 'procedures_per_day', 'labs_per_day', 
                        'total_prior_visits', 'high_utilizer']
interaction_features = [f for f in interaction_features if f in df.columns]
print(f"\n  Total interaction features created: {len(interaction_features)}")

# ============================================================================
# 3. CREATE POLYNOMIAL FEATURES
# ============================================================================



print("\n[3.1] Creating squared terms for key continuous variables...")

# Select variables for polynomial features (based on EDA insights)
poly_candidates = ['time_in_hospital', 'num_medications', 'num_lab_procedures']
poly_candidates = [col for col in poly_candidates if col in df.columns]

poly_features = []
for col in poly_candidates:
    new_col = f"{col}_squared"
    df[new_col] = df[col] ** 2
    poly_features.append(new_col)
    print(f"  ✓ Created: {new_col}")

print(f"\n  Total polynomial features created: {len(poly_features)}")

# why we are adding squared features:
# Linear regression assumes a straight-line relationship:
# Y=β0​+β1​X+ε
# real-world data (especially in healthcare) often show curved or non-linear patterns.
#The length of hospital stay might increase health costs rapidly at first but then plateau.
# - Number of medications might improve recovery up to a point, then cause side effects.
# - A simple linear term (num_medications) cannot capture that curvature.

# To model curvature, we include the square of x
# Y=β0​+β1​X+β2​X^2+ε
# This creates a quadratic relationship, allowing the regression line to bend.
# If 𝛽2 is:
# # - positive → the curve opens upward (U-shape)
# # - negative → the curve opens downward (∩-shape)



# ============================================================================
# 4. CREATE BINARY INDICATOR FEATURES
# ============================================================================



print("\n[4.1] Creating binary flags for important conditions...")
df.columns

binary_features = []


# Flag: HbA1c tested
df['A1C_tested'] = (df['A1Cresult'] != 'NoTest').astype(int)
binary_features.append('A1C_tested')

# flag: a1c_bad
df['a1c_flag_bad'] = df['A1Cresult'].isin(['>8', '>9']).astype(int)
binary_features.append('a1c_flag_bad')
# look into literature there my be bad with changes and bad no changes


# Flag: Change in diabetes medication
df['diabetes_med_changed'] = (df['change'] == 'Ch').astype(int)
binary_features.append('diabetes_med_changed')


# Flag: Diabetes medication prescribed
df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
binary_features.append('on_diabetes_med')


# Flag: Long hospital stay (>7 days)
df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
binary_features.append('long_stay')
 

# Flag: Many medications (>15)

df['many_medications'] = (df['num_medications'] > 15).astype(int)
binary_features.append('many_medications')
 

# Flag: Multiple diagnoses (>7)

df['complex_case'] = (df['number_diagnoses'] > 7).astype(int)
binary_features.append('complex_case')


# Flag: prior Impatient > 0
df['had_prior_inpatient'] = (df['number_inpatient'] > 0).astype(int)
binary_features.append('had_prior_inpatient')

# Flag: frequent ER suer
df['frequent_ER_user'] = (df['number_emergency'] >= 2).astype(int)
binary_features.append('frequent_ER_user')
# High Utilization Flag was done above in chapter 2 using quantile 0.75

# home-like
home_codes = [1, 6, 8]  # home, home health, home with provider
df['discharge_home'] = df['discharge_disposition_id'].isin(home_codes).astype(int)
binary_features.append('discharge_home')

# post-acute facility (SNF / rehab / nursing facility)
post_acute_codes = [3, 4, 5, 15, 22, 23, 24, 29, 30]
df['discharge_postacute'] = df['discharge_disposition_id'].isin(post_acute_codes).astype(int)
binary_features.append('discharge_postacute')

# psych/behavioral
psych_codes = [28]
df['discharge_psych'] = df['discharge_disposition_id'].isin(psych_codes).astype(int)
binary_features.append('discharge_psych')


emergency_like = [1, 2, 7]  # Emergency, Urgent, Trauma Center
elective_like  = [3]        # Elective
transfer_like  = [4, 5, 6]  # transferred from other facility/SNF
newborn_like   = [4]        # depending on mapping, you might drop these later anyway

df['admit_emergent']  = df['admission_type_id'].isin([1,2,7]).astype(int)
df['admit_elective']  = df['admission_type_id'].isin([3]).astype(int)
df['admit_transfer']  = df['admission_source_id'].isin([4,5,6]).astype(int)

binary_features.extend(['admit_emergent','admit_elective','admit_transfer'])

interaction_features = [
    'meds_per_day',        # Medications per hospital day
    'procedures_per_day',  # Procedures per hospital day
    'labs_per_day',        # Lab tests per hospital day
    'total_prior_visits',  # Total prior visits (outpatient + emergency + inpatient)
    'high_utilizer'        # Flag: top 25% of total prior visits
]



engineered_cols = interaction_features + poly_features + binary_features
engineered_cols = list(dict.fromkeys(engineered_cols))  # dedupe, keep order



#df = df.copy()

#  after testing the models we are adding NEW FEATURE HOPING TO GET A BETTER RECALL ------------------

#Cross-intensity interactions (medications × labs × visits)
df['meds_x_labs']   = df['num_medications'] * df['num_lab_procedures']
df['meds_x_visits'] = df['num_medications'] * df['total_prior_visits']
df['stay_x_meds']   = df['time_in_hospital'] * df['num_medications']
df['stay_x_labs']   = df['time_in_hospital'] * df['num_lab_procedures']
df['labs_x_visits'] = df['num_lab_procedures'] * df['total_prior_visits']

interaction_features.extend(['meds_x_labs', 'meds_x_visits', 'stay_x_meds', 'stay_x_labs', 'labs_x_visits'])

# Maybe the combination of these two factors matters more than each one individually.
# encoding nonlinear relationships that  ANN can use to pick up complex patterns 


#Binary interaction strength (chronic condition × intensity)
df['meds_x_diab_comp'] = df['num_medications'] * df['diab_complication_binary']
df['labs_x_diab_comp'] = df['num_lab_procedures'] * df['diab_complication_binary']


binary_features.extend(['meds_x_diab_comp', 'labs_x_diab_comp'])


print("\n[SUMMARY] Engineered feature count:", len(engineered_cols))
print("Engineered features:", engineered_cols)


#let me doublecheck
df[['meds_x_labs','meds_x_visits','stay_x_meds','stay_x_labs',
    'labs_x_visits','meds_x_diab_comp','labs_x_diab_comp']].head()

df.shape

pd.to_pickle(df, 'data/selected-features/feature_all_not_onehot.pkl')

# ============================================================================
# 5. MAPPING VALUES
# ---------------------------------------------------------------------------
# this was moved in step2 as i did not know that it exists here, anyway was needed to remove some rows


# ============================================================================
# 6. ENCODE CATEGORICAL VARIABLES
# ============================================================================

y = df['readmitted_binary'].astype(int)

cols_to_drop = [
    'encounter_id', 'patient_nbr', 'discharge_disposition_id',
    'admission_type_id', 'discharge_disposition_desc', 'admission_source_id',
    'readmitted', 'readmited_binary'
]

df = df.drop(columns=cols_to_drop, errors='ignore')

noise = ['diag_1', 'diag_2', 'diag_3']

df = df.drop(columns=noise, errors='ignore')



print("\n[5.1] One-hot encoding categorical features...")

# Identify categorical columns for encoding
categorical_to_encode = [col for col in df.columns if df[col].dtype == 'object']
# One-hot encode
df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

print(f"  ✓ Encoded {len(categorical_to_encode)} categorical features")
print(f"  ✓ Dataset shape after encoding: {df_encoded.shape}")

data_to_save = {'X': df_encoded, 'y': y}
save_path = "data/selected-features/feature_all_onehot.pkl"
pd.to_pickle(data_to_save, save_path)

# ============================================================================
# 7. PREPARE FEATURE MATRIX
# ============================================================================

df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True, dtype=int)
X = df_encoded.drop('readmitted_binary', axis=1)
y = df_encoded['readmitted_binary']

# print(f"Shape: {X.shape}")
# print(f"Target distribution: {y.value_counts().to_dict()}")


non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\n[6.1] Removing {len(non_numeric)} non-numeric columns: {non_numeric}")
    X = X.drop(columns=non_numeric)


print(f"Shape: {X.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")
# ============================================================================
# 8. FEATURE SELECTION - CORRELATION METHOD 
# ============================================================================

#Pearson correlation coefficient - between each colun in X and Y


# Calculate correlation with target
correlations = X.corrwith(y).abs().sort_values(ascending=False)

print("\n  Top 15 features by correlation:")
for idx, (feature, corr) in enumerate(correlations.head(15).items(), 1):
    print(f"    {idx}. {feature}: {corr:.4f}")

# Select features with correlation > threshold
corr_threshold = 0.01
corr_selected = correlations[correlations > corr_threshold].index.tolist()
print(f"\n  Features with |correlation| > {corr_threshold}: {len(corr_selected)}")

# ============================================================================
# 9. FEATURE SELECTION - MUTUAL INFORMATION Scores
# ============================================================================

#This uses scikit-learn’s mutual_info_classif,
#  which measures how much knowing a feature reduces uncertainty about the target.

#In plain English:
        # from module import namesIf a feature helps you predict the target, it will have a higher MI score.
       # If it’s unrelated noise, it will have a score near 0.
#Unlike correlation, mutual information detects nonlinear relationships — it’s not limited to straight-line effects.
import multiprocessing
import platform
import psutil

print(platform.processor())
print(f"CPU cores: {multiprocessing.cpu_count()}")
print(f"Available memory: {psutil.virtual_memory().available / 1e9:.2f} GB")



from sklearn.feature_selection import mutual_info_classif

start = time.perf_counter()
# start time
mi_scores = mutual_info_classif(X, y, random_state=42, n_jobs=-1)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
# time
print(f"It took {time.perf_counter() - start:.2f} seconds")

print("\n  Top 15 features by mutual information:")
for idx, (feature, score) in enumerate(mi_scores.head(15).items(), 1):
    print(f"    {idx}. {feature}: {score:.4f}")

# Select top N features
mi_selected = mi_scores.head(75).index.tolist()
print(f"\n  Selected top 50 features by mutual information")

# ============================================================================
# 10. FEATURE SELECTION - RANDOM FOREST IMPORTANCE
# ============================================================================



print("\n[9.1] Training Random Forest for feature importance...")


start = time.perf_counter()
# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X, y)
#time
print(f"It took {time.perf_counter() - start:.2f} seconds")


# Get feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n  Top 15 features by Random Forest importance:")
for idx, (feature, importance) in enumerate(importances.head(15).items(), 1):
    print(f"    {idx}. {feature}: {importance:.4f}")

# Select features with importance > threshold
importance_threshold = 0.001
rf_selected = importances[importances > importance_threshold].index.tolist()
print(f"\n  Features with importance > {importance_threshold}: {len(rf_selected)}")

# ============================================================================
# 11. COMBINE SELECTION METHODS
# ============================================================================



print("\n[10.1] Finding consensus features...")

# Features selected by at least 2 methods
all_methods = [set(corr_selected), set(mi_selected), set(rf_selected)]

# Count how many methods selected each feature
feature_votes = {}
for feature in X.columns:
    votes = sum(feature in method for method in all_methods)
    if votes >= 2:  # Selected by at least 2 methods
        feature_votes[feature] = votes

# Sort by votes
feature_votes = dict(sorted(feature_votes.items(), key=lambda x: x[1], reverse=True))

print(f"  Features selected by 2+ methods: {len(feature_votes)}")
print(f"\n  Features selected by all 3 methods: {sum(v==3 for v in feature_votes.values())}")
print(f"  Features selected by 2 methods: {sum(v==2 for v in feature_votes.values())}")

# Final feature set
final_features = list(feature_votes.keys())

print(f"\n[10.2] Final feature set: {len(final_features)} features")

# ============================================================================
# 12. CREATE FINAL DATASETS
# ============================================================================



# Create final feature matrix
X_final = X[final_features].copy()
y_final = y.copy()

# --- Save final datasets (PKL format) ---
print(f"\n[11.1] Final dataset dimensions: {X_final.shape[0]} samples, {X_final.shape[1]} features")

# Save combined dataset as pickle
final_data = {'X': X_final, 'y': y_final}
pd.to_pickle(final_data, 'data/selected-features/feature_selection_onehot.pkl')


# Save feature info
feature_info = pd.DataFrame({'Feature': final_features,
                             'Votes': [feature_votes.get(f, 0) for f in final_features]})
feature_info.sort_values('Votes', ascending=False).to_csv('../reports/04_selected_features.csv', index=False)



# ============================================================================
# 13. FEATURE ENGINEERING SUMMARY
# ============================================================================



summary = {
    'Original_Features': original_features,
    'Interaction_Features': len(interaction_features),
    'Polynomial_Features': len(poly_features),
    'Binary_Indicator_Features': len(binary_features),
    'After_Encoding': X.shape[1],
    'After_Selection': len(final_features),
    'Reduction_Percentage': ((X.shape[1] - len(final_features)) / X.shape[1] * 100)
}

print("\n")
for key, value in summary.items():
    if isinstance(value, float):
        print(f"  {key.replace('_', ' ')}: {value:.1f}%")
    else:
        print(f"  {key.replace('_', ' ')}: {value}")

# Create visualization of feature importance
print("\n[12.1] Creating feature importance visualization...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Feature Selection Methods Comparison', fontsize=16, fontweight='bold')

# Plot 1: Top features by correlation
ax = axes[0]
correlations.head(15).plot(kind='barh', ax=ax, color='steelblue')
ax.set_title('Top 15 Features by Correlation', fontweight='bold')
ax.set_xlabel('|Correlation| with Target')
ax.invert_yaxis()

# Plot 2: Top features by mutual information
ax = axes[1]
mi_scores.head(15).plot(kind='barh', ax=ax, color='darkorange')
ax.set_title('Top 15 Features by Mutual Information', fontweight='bold')
ax.set_xlabel('Mutual Information Score')
ax.invert_yaxis()

# Plot 3: Top features by Random Forest
ax = axes[2]
importances.head(15).plot(kind='barh', ax=ax, color='mediumseagreen')
ax.set_title('Top 15 Features by RF Importance', fontweight='bold')
ax.set_xlabel('Feature Importance')
ax.invert_yaxis()

plt.tight_layout()
plt.savefig('../figures/04_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 04_feature_importance_comparison.png")

# Save summary
summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
summary_df.to_csv('../reports/04_feature_engineering_summary.csv', index=False)
print("✓ Summary saved as '04_feature_engineering_summary.csv'")




print("\n" + "="*80)
print("\nGenerated Files:")
print("  - 04_X_features.csv (final feature matrix)")
print("  - 04_y_target.csv (target variable)")
print("  - 04_selected_features.csv (feature list)")
print("  - 04_feature_importance_comparison.png")
print("  - 04_feature_engineering_summary.csv")
print("\nNext Steps:")
print("  1. Review selected features")
print("  2. Check feature importance visualization")
print("  3. Proceed to Step 5: Model Development and Training")
print("="*80)