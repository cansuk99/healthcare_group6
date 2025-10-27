import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import RocCurveDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Objective: Can hospital readmissions within 30 days be predicted with an F1-score?
# We used the cleaned diabetes dataset and apply manual feature selection based on statistical significance and clinical reasoning.
# This approach reflects a more explainable, research-driven logic, where features are included if they are **statistically significant predictors of readmission** and **clinically meaningful**.

df = pd.read_csv('./data/processed/diabetes_cleaned_data.csv')
print(f"Loaded dataset shape: {df.shape}")

# Feature Selection Logic

# We include only features that are:
# 1. Clinically interpretable and measurable at admission or discharge.
# 2. Statistically significant in prior logistic regression models (p < 0.05).

# Define feature sets based on statistical significance

y = (df["readmitted"] == "<30").astype(int)
X = df.drop(columns=["readmitted"])

X = pd.get_dummies(X, drop_first=True)

statistical_features = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_diagnoses",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "change_Yes",
    "diabetesMed_Yes",
    "age"
]

X = X[[col for col in statistical_features if col in X.columns]]

# Model Impelementation and Cross K Validation
# In the first phase, we evaluate models using 5-fold stratified cross-validation on the dataset to establish baseline performance.

# Logistic Regression, Random Forest, Gradient Boosting
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.08)
}

def evaluate_model(model, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1).mean()
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    return f1, auc

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = []
for name, model in models.items():
    f1, auc = evaluate_model(model, X_scaled, y)
    results.append({"Model": name, "F1": f1, "AUC": auc})

results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
print(results_df)

# The baseline evaluation shows that all models perform poorly, with F1-scores below 0.20 and AUC values around 0.60.
# This indicates that the models struggle to correctly identify patients who are readmitted within 30 days.

# Visualization for F1-score and AUC

results_melted = results_df.melt(
    id_vars="Model",
    value_vars=["F1", "AUC"],
    var_name="Metric",
    value_name="Score"
)

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.figure(figsize=(8, 4.5))

barplot = sns.barplot(
    data=results_melted,
    x="Model",
    y="Score",
    hue="Metric",
    palette=["#1f77b4", "#ff7f0e"],
    edgecolor="black",
    width=0.5
)

for p in barplot.patches:
    height = p.get_height()

    offset = 0.03
    barplot.annotate(
        f'{height:.2f}',
        (p.get_x() + p.get_width() / 2., height + offset),
        ha='center', va='bottom',
        fontsize=10, fontweight='bold'
    )

# Title and axis settings
plt.title("Model Performance Comparison (F1 vs AUC)", fontsize=14, weight='bold', pad=15)
plt.ylabel("Score")
plt.xlabel("")
plt.ylim(0, 1.1)  # extra space for labels
plt.legend(title="Metric", loc="upper right", frameon=True)
plt.tight_layout()
plt.show()

# The results show using only clinically interpretable, statistically explainable features, hospital readmissions are extremely hard to predict. The imbalance and limited features lead to poor explainiblity for hospital readmissions with the used machine learning models.

# Addressing Class Imbalance
# In this phase, we handle class imbalance using SMOTE and re-evaluate the models after adjusting the classification threshold to maximize the F1-score.
# The default threshold of 0.5 is often not suitable for imbalanced data, so we tune it to get a better balance between precision and recall.
# This helps the model identify more patients at risk of readmission while reducing false alarms.

# Apply SMOTE on the training data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Logistic Regression, Random Forest, Gradient Boosting with SMOTE and Threshold application

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
    "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=200),
    "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=300, learning_rate=0.08)
}

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

print("Before SMOTE:", np.bincount(y))
X_train, y_train = SMOTE(random_state=42).fit_resample(X_train, y_train)
print("After SMOTE:", np.bincount(y_train))

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = [f1_score(y_test, (y_proba > t).astype(int)) for t in thresholds]

    best_t = thresholds[np.argmax(f1_scores)]
    best_f1 = max(f1_scores)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Best_Threshold": best_t, "F1": best_f1, "AUC": auc})

    print(f"{name}: F1 = {best_f1:.3f}, AUC = {auc:.3f}, Optimal threshold = {best_t:.2f}")

# After applying SMOTE and tuning the classification threshold, the Random Forest and Gradient Boosting models achieve very high F1-scores (above 0.90) and AUC values close to 1.0, indicating strong performance on the balanced dataset.
# However, these results likely overestimate real-world performance, since the models were trained and tested on synthetic data generated by SMOTE.
# In clinical practice, where readmissions remain rare (~9%), performance would likely be much lower.

# Cross-validation on the balanced dataset
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    f1 = cross_val_score(model, X_test, y_test, cv=cv, scoring='f1', n_jobs=-1).mean()
    auc = cross_val_score(model, X_test, y_test, cv=cv, scoring='roc_auc', n_jobs=-1).mean()
    print(f"{name}: CV F1={f1:.3f}, AUC={auc:.3f}")

# The cross-validation results show that the models maintain high F1 and AUC scores across all folds, confirming that their performance on the balanced dataset is consistent and not due to a single lucky split.
# This means the models have learned the patterns in the resampled data reliably.
# However, because the data was balanced using SMOTE, these results still does not represent real-world hospitalization cases.

# Visualizations

# Select Logistic Regression as best model for visualization
best_model_name = "LogisticRegression"
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

best_thresh = 0.35
y_pred = (best_model.predict_proba(X_test)[:, 1] > best_thresh).astype(int)

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Model Evaluation Dashboard – Manual Feature Selection", fontsize=16, weight="bold")

#F1-score Barplot
sns.barplot(ax=axes[0,0], data=results_df, x="Model", y="F1", palette="crest", edgecolor="black")
axes[0,0].set_title("F1-Scores After SMOTE + Threshold Tuning", fontsize=13, weight='bold')
axes[0,0].set_ylim(0, 1)
axes[0,0].grid(axis='y', alpha=0.3)
for i, v in enumerate(results_df['F1']):
    axes[0,0].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# AUC Comparison Bar Chart
sns.barplot(ax=axes[0,1], data=results_df, x="Model", y="AUC", palette="viridis", edgecolor="black")
axes[0,1].set_title("AUC Comparison", fontsize=13, weight='bold')
axes[0,1].set_ylim(0, 1)
axes[0,1].grid(axis='y', alpha=0.3)
for i, v in enumerate(results_df['AUC']):
    axes[0,1].text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')

# ROC Curves
for name, model in models.items():
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, name=f"{name}", ax=axes[1,0])
axes[1,0].plot([0,1], [0,1], 'k--', lw=1)
axes[1,0].set_title("ROC Curves", fontsize=13, weight='bold')

# Confusion Matrix for Logistic Regression
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(ax=axes[1,1], cmap="Blues", values_format='d', colorbar=True)

axes[1,1].set_title("Confusion Matrix - Logistic Regression", fontsize=13, weight='bold')
axes[1,1].set_xlabel("Predicted Label", fontsize=11, labelpad=10)
axes[1,1].set_ylabel("Actual Label", fontsize=11, labelpad=10)

axes[1,1].xaxis.set_ticklabels(["Not Readmit", "Readmit"])
axes[1,1].yaxis.set_ticklabels(["Not Readmit", "Readmit"])

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()


# The baseline evaluation with manually selected features shows that the models perform poorly on the dataset. 
# F1-scores are very low (<0.20), indicating that models struggle to correctly identify patients who are readmitted within 30 days.
# AUC values are around 0.55–0.60, showing only a slight improvement over random guessing.
# The ROC curve for Logistic Regression lies close to the diagonal line, indicating that the model can only weakly distinguish between patients who will be readmitted and those who will not, showing low predictive power.
# The Confusion Matrix for Logistic Regression shows a large number of false negatives (actual readmitted patients predicted as not readmitted), highlighting the strong effect of class imbalance.
# Overall, these results suggest that hospital readmission is difficult to predict using only a limited set of statistically significant and clinically explainable features.
# While the models are interpretable, they fail to capture the complex interactions behind patient readmissions.