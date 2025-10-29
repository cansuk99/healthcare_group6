import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek


# ============================================================================
# 1. LOAD FINAL FEATURE SET - SELECTED
# ============================================================================
#          50/50--      Results - feature selection
# print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       0.89      0.96      0.92     17606
#            1       0.22      0.10      0.14      2263

#     accuracy                           0.86     19869
#    macro avg       0.56      0.53      0.53     19869
# weighted avg       0.82      0.86      0.83     19869

# Best threshold = 0.157 with F1 = 0.237
#               precision    recall  f1-score   support

#            0       0.92      0.53      0.67     17606
#            1       0.15      0.63      0.24      2263

#     accuracy                           0.54     19869
#    macro avg       0.53      0.58      0.45     19869
# weighted avg       0.83      0.54      0.62     19869

# The model looks “good” on paper (86% accuracy), but this is fake comfort.
# Clinically: this is not acceptable. You miss 90% of people who come back in 30 days.


#           SMOTE    Results - feature selection
#               precision    recall  f1-score   support

#            0       0.89      0.98      0.93     17606
#            1       0.19      0.04      0.07      2263

#     accuracy                           0.87     19869
#    macro avg       0.54      0.51      0.50     19869
# weighted avg       0.81      0.87      0.83     19869

# Best threshold = 0.094 with F1 = 0.226
#               precision    recall  f1-score   support

#            0       0.91      0.49      0.64     17606
#            1       0.14      0.63      0.23      2263

#     accuracy                           0.51     19869
#    macro avg       0.52      0.56      0.43     19869
# weighted avg       0.82      0.51      0.59     19869

# This is interesting: Even after SMOTE, when you evaluate on real-world (imbalanced) test data with threshold=0.5, recall for class 1 is even worse (4% vs 10% before).
# SMOTE alone did not “solve” the imbalance in deployment.

# --------------------------------------------------------------------------------
# something is up if i fit all columns I am not sure why maybe overfitting ----


# print(classification_report(y_test, y_pred))
#               precision    recall  f1-score   support

#            0       1.00      1.00      1.00     17606
#            1       1.00      1.00      1.00      2263

#     accuracy                           1.00     19869
#    macro avg       1.00      1.00      1.00     19869
# weighted avg       1.00      1.00      1.00     19869


# =======================================================
# 1. RESAMPLING FUNCTION
# =======================================================
def resample_data(strategy, X, y):
    """Resample dataset according to selected strategy."""
    if strategy == "smote":
        sampler = SMOTE(random_state=42, n_jobs=-1)
    elif strategy == "under":
        sampler = RandomUnderSampler(random_state=42)
    elif strategy == "combo_50_50":
        sampler = SMOTETomek(random_state=42, n_jobs=-1)
    else:
        raise ValueError(
            "Unknown strategy. Choose from ['smote', 'under', 'combo_50_50']"
        )

    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


# =======================================================
# 2. LOAD DATA & SPLIT
# =======================================================
data = pd.read_pickle("data/selected-features/feature_selection_onehot.pkl")
X, y = data["X"], data["y"]

# Something is up with this dataset and i do not knowwhat is the problem
# data = pd.read_pickle("data/selected-features/feature_all_onehot.pkl")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Test shape : {X_test.shape}, {y_test.shape}")
print("\nClass balance in TRAIN before resampling:")
print(y_train.value_counts(normalize=True).rename("proportion"))


# =======================================================
# 3. RESAMPLE TRAINING DATA
# =======================================================
strategy = "combo_50_50"  # choose: 'smote', 'under', or 'combo_50_50'
X_train_res, y_train_res = resample_data(strategy, X_train, y_train)

print("\nClass balance AFTER resampling:")
print(y_train_res.value_counts(normalize=True).rename("proportion"))


# =======================================================
# 4. DEFINE MODEL PIPELINE
# =======================================================
logreg_pipe = Pipeline(
    [
        ("scaler", StandardScaler(with_mean=False)),
        (
            "model",
            LogisticRegression(
                solver="saga",
                penalty="l2",
                class_weight="balanced",
                max_iter=2000,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
)


# =======================================================
# 5. TRAIN & EVALUATE MODEL
# =======================================================
logreg_pipe.fit(X_train_res, y_train_res)
y_prob = logreg_pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

print("\n=== BASE THRESHOLD RESULTS (0.5) ===")
print(classification_report(y_test, y_pred))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")


# =======================================================
# 6. OPTIMAL THRESHOLD (BASED ON F1)
# =======================================================
precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-9)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx]

print(f"\nBest threshold = {best_thresh:.3f} with F1 = {f1_scores[best_idx]:.3f}")

y_pred_opt = (y_prob >= best_thresh).astype(int)

print("\n=== OPTIMAL THRESHOLD RESULTS ===")
print(classification_report(y_test, y_pred_opt))
print(f"ROC AUC: {roc_auc_score(y_test, y_prob):.3f}")


# 2. ROC Curve
y_prob = logreg_clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (1 - Specificity)")
plt.ylabel("True Positive Rate (Sensitivity)")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


# 3. Precision-Recall Curve
precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall_vals, precision_vals)

plt.figure()
plt.plot(recall_vals, precision_vals, label=f"PR curve (AUC = {pr_auc:.3f})")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("Recall (Sensitivity)")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve (Positive Class)")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()
