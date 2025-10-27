
# not working for now but i want to do another approach to the data.
# If that works Then will look into RF

# Random Forest   ==========================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import resample
from sklearn.model_selection import train_test_split



# 1. We start from the full X_model, y_final
X_train, X_test, y_train, y_test = train_test_split(
    X_model,
    y_final,
    test_size=0.2,
    random_state=42,
    stratify=y_final
)

# 2. Build a balanced training set by undersampling the majority class
train_df = pd.concat([X_train, y_train], axis=1)
target_col = y_train.name if hasattr(y_train, "name") and y_train.name is not None else "target_tmp"
if target_col == "target_tmp":
    train_df[target_col] = y_train.values  # ensure column exists

majority_df = train_df[train_df[target_col] == 0]
minority_df = train_df[train_df[target_col] == 1]

# downsample majority to the same size as minority
majority_downsampled = resample(
    majority_df,
    replace=False,
    n_samples=len(minority_df),
    random_state=42
)

balanced_df = pd.concat([majority_downsampled, minority_df])

# shuffle rows
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# split back into X_balanced, y_balanced
X_balanced = balanced_df.drop(columns=[target_col])
y_balanced = balanced_df[target_col]

print("Balanced training set shape:", X_balanced.shape, y_balanced.shape)
print("Class counts in balanced training set:")
print(y_balanced.value_counts())

# 3. Train RandomForest on the balanced data (no class_weight now)
rf_balanced = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,
    random_state=42
)

rf_balanced.fit(X_balanced, y_balanced)

# 4. Evaluate on the ORIGINAL untouched test set
y_pred_rf = rf_balanced.predict(X_test)

print("\n=== Classification report (Random Forest on undersampled data) ===")
print(classification_report(y_test, y_pred_rf))

print("\n=== Confusion matrix (Random Forest on undersampled data) ===")
print(confusion_matrix(y_test, y_pred_rf))


# RAndom Forest with SMOTE -------

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE



# 1. Build SMOTE + Random Forest pipeline
smote_rf = ImbPipeline(steps=[
    ('smote', SMOTE(random_state=42, sampling_strategy='auto')),
    ('model', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    ))
])

# 2. Fit model (SMOTE is applied only to X_train/y_train internally)
smote_rf.fit(X_train, y_train)

# 3. Evaluate on untouched test data
y_pred_smote = smote_rf.predict(X_test)

print("\n=== Classification report (SMOTE + Random Forest) ===")
print(classification_report(y_test, y_pred_smote))

print("\n=== Confusion matrix (SMOTE + Random Forest) ===")
print(confusion_matrix(y_test, y_pred_smote))
