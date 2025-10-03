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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_classif, SelectKBest, chi2, f_classif
from sklearn.ensemble import RandomForestClassifier

print("="*80)
print("STEP 4: FEATURE ENGINEERING AND SELECTION")
print("="*80)

# ============================================================================
# 1. LOAD CLEANED DATA
# ============================================================================

print("\n[1] Loading cleaned data...")
try:
    df = pd.read_csv('diabetes_cleaned_data.csv')
    print(f"✓ Data loaded: {df.shape}")
except FileNotFoundError:
    print("✗ Error: 'diabetes_cleaned_data.csv' not found.")
    print("  Please run Step 2 first.")
    exit()

# Verify target exists
if 'readmitted_binary' not in df.columns:
    print("✗ Error: Target variable 'readmitted_binary' not found.")
    exit()

# Store original shape
original_features = len(df.columns) - 1  # Exclude target

# ============================================================================
# 2. CREATE INTERACTION FEATURES
# ============================================================================

print("\n" + "="*80)
print("[2] CREATING INTERACTION FEATURES")
print("="*80)

print("\n[2.1] Creating clinically meaningful interactions...")

# Interaction 1: Medications × Time in Hospital
if 'num_medications' in df.columns and 'time_in_hospital' in df.columns:
    df['meds_per_day'] = df['num_medications'] / (df['time_in_hospital'] + 1)  # +1 to avoid division by zero
    print("  ✓ Created: meds_per_day (medications per hospital day)")

# Interaction 2: Procedures × Time in Hospital
if 'num_procedures' in df.columns and 'time_in_hospital' in df.columns:
    df['procedures_per_day'] = df['num_procedures'] / (df['time_in_hospital'] + 1)
    print("  ✓ Created: procedures_per_day")

# Interaction 3: Lab tests × Time in Hospital
if 'num_lab_procedures' in df.columns and 'time_in_hospital' in df.columns:
    df['labs_per_day'] = df['num_lab_procedures'] / (df['time_in_hospital'] + 1)
    print("  ✓ Created: labs_per_day")

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

print("\n" + "="*80)
print("[3] CREATING POLYNOMIAL FEATURES")
print("="*80)

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

# ============================================================================
# 4. CREATE BINARY INDICATOR FEATURES
# ============================================================================

print("\n" + "="*80)
print("[4] CREATING BINARY INDICATOR FEATURES")
print("="*80)

print("\n[4.1] Creating binary flags for important conditions...")

binary_features = []

# Flag: HbA1c tested
if 'A1Cresult' in df.columns:
    df['A1C_tested'] = (df['A1Cresult'] != 'None').astype(int)
    binary_features.append('A1C_tested')
    print("  ✓ Created: A1C_tested")

# Flag: Change in diabetes medication
if 'change' in df.columns:
    df['diabetes_med_changed'] = (df['change'] == 'Ch').astype(int)
    binary_features.append('diabetes_med_changed')
    print("  ✓ Created: diabetes_med_changed")

# Flag: Diabetes medication prescribed
if 'diabetesMed' in df.columns:
    df['on_diabetes_med'] = (df['diabetesMed'] == 'Yes').astype(int)
    binary_features.append('on_diabetes_med')
    print("  ✓ Created: on_diabetes_med")

# Flag: Long hospital stay (>7 days)
if 'time_in_hospital' in df.columns:
    df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
    binary_features.append('long_stay')
    print("  ✓ Created: long_stay")

# Flag: Many medications (>15)
if 'num_medications' in df.columns:
    df['many_medications'] = (df['num_medications'] > 15).astype(int)
    binary_features.append('many_medications')
    print("  ✓ Created: many_medications")

# Flag: Multiple diagnoses (>7)
if 'number_diagnoses' in df.columns:
    df['complex_case'] = (df['number_diagnoses'] > 7).astype(int)
    binary_features.append('complex_case')
    print("  ✓ Created: complex_case")

print(f"\n  Total binary indicator features created: {len(binary_features)}")

# ============================================================================
# 5. ENCODE CATEGORICAL VARIABLES
# ============================================================================

print("\n" + "="*80)
print("[5] ENCODING CATEGORICAL VARIABLES")
print("="*80)

print("\n[5.1] One-hot encoding categorical features...")

# Identify categorical columns for encoding
categorical_to_encode = []
for col in df.columns:
    if df[col].dtype == 'object' and col != 'readmitted':  # Exclude original target
        categorical_to_encode.append(col)

# One-hot encode
df_encoded = pd.get_dummies(df, columns=categorical_to_encode, drop_first=True)

print(f"  ✓ Encoded {len(categorical_to_encode)} categorical features")
print(f"  ✓ Dataset shape after encoding: {df_encoded.shape}")

# ============================================================================
# 6. PREPARE FEATURE MATRIX
# ============================================================================

print("\n" + "="*80)
print("[6] PREPARING FEATURE MATRIX")
print("="*80)

# Separate features and target
X = df_encoded.drop(['readmitted_binary'], axis=1)
y = df_encoded['readmitted_binary']

# Drop any remaining non-numeric columns
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    print(f"\n[6.1] Removing {len(non_numeric)} non-numeric columns: {non_numeric}")
    X = X.drop(columns=non_numeric)

print(f"\n[6.2] Feature matrix prepared:")
print(f"  - Shape: {X.shape}")
print(f"  - Features: {X.shape[1]}")
print(f"  - Samples: {X.shape[0]}")
print(f"  - Target distribution: {y.value_counts().to_dict()}")

# ============================================================================
# 7. FEATURE SELECTION - CORRELATION METHOD
# ============================================================================

print("\n" + "="*80)
print("[7] FEATURE SELECTION - CORRELATION METHOD")
print("="*80)

print("\n[7.1] Computing correlations with target...")

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
# 8. FEATURE SELECTION - MUTUAL INFORMATION
# ============================================================================

print("\n" + "="*80)
print("[8] FEATURE SELECTION - MUTUAL INFORMATION")
print("="*80)

print("\n[8.1] Computing mutual information scores...")

# Calculate mutual information
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

print("\n  Top 15 features by mutual information:")
for idx, (feature, score) in enumerate(mi_scores.head(15).items(), 1):
    print(f"    {idx}. {feature}: {score:.4f}")

# Select top N features
mi_selected = mi_scores.head(50).index.tolist()
print(f"\n  Selected top 50 features by mutual information")

# ============================================================================
# 9. FEATURE SELECTION - RANDOM FOREST IMPORTANCE
# ============================================================================

print("\n" + "="*80)
print("[9] FEATURE SELECTION - RANDOM FOREST IMPORTANCE")
print("="*80)

print("\n[9.1] Training Random Forest for feature importance...")

# Train Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X, y)

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
# 10. COMBINE SELECTION METHODS
# ============================================================================

print("\n" + "="*80)
print("[10] COMBINING FEATURE SELECTION METHODS")
print("="*80)

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
# 11. CREATE FINAL DATASETS
# ============================================================================

print("\n" + "="*80)
print("[11] CREATING FINAL DATASETS")
print("="*80)

# Create final feature matrix
X_final = X[final_features].copy()
y_final = y.copy()

print(f"\n[11.1] Final dataset dimensions:")
print(f"  - Features: {X_final.shape[1]}")
print(f"  - Samples: {X_final.shape[0]}")
print(f"  - Target: {y_final.shape[0]}")

# Save feature names
feature_info = pd.DataFrame({
    'Feature': final_features,
    'Votes': [feature_votes.get(f, 0) for f in final_features]
})
feature_info = feature_info.sort_values('Votes', ascending=False)
feature_info.to_csv('04_selected_features.csv', index=False)
print("\n✓ Feature list saved as '04_selected_features.csv'")

# Save final datasets
X_final.to_csv('04_X_features.csv', index=False)
y_final.to_csv('04_y_target.csv', index=False)
print("✓ Feature matrix saved as '04_X_features.csv'")
print("✓ Target variable saved as '04_y_target.csv'")

# ============================================================================
# 12. FEATURE ENGINEERING SUMMARY
# ============================================================================

print("\n" + "="*80)
print("[12] FEATURE ENGINEERING SUMMARY")
print("="*80)

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
plt.savefig('04_feature_importance_comparison.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 04_feature_importance_comparison.png")

# Save summary
summary_df = pd.DataFrame(list(summary.items()), columns=['Metric', 'Value'])
summary_df.to_csv('04_feature_engineering_summary.csv', index=False)
print("✓ Summary saved as '04_feature_engineering_summary.csv'")

print("\n" + "="*80)
print("STEP 4 COMPLETE!")
print("="*80)
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