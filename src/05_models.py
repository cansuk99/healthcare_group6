"""
Step 5: Model Development and Training
Diabetes 130-US Hospitals Dataset

This script:
1. Loads the final feature set from Step 4
2. Splits data into train/test sets
3. Handles class imbalance with SMOTE
4. Trains multiple classification models
5. Performs hyperparameter tuning
6. Evaluates model performance
7. Interprets feature importance
8. Saves the best model
9. aditional line
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, precision_recall_curve, f1_score, 
                             accuracy_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("STEP 5: MODEL DEVELOPMENT AND TRAINING")
print("="*80)

# ============================================================================
# 1. LOAD FINAL FEATURE SET
# ============================================================================

print("\n[1] Loading final feature set from Step 4...")
try:
    X = pd.read_csv('../data/selected-features/04_X_features.csv')
    y = pd.read_csv('../data/selected-features/04_y_target.csv').values.ravel()  # Convert to 1D array
    print(f"✓ Data loaded successfully")
    print(f"  - Features: {X.shape[1]}")
    print(f"  - Samples: {X.shape[0]}")
    print(f"  - Target shape: {y.shape}")
except FileNotFoundError:
    print("✗ Error: Feature files not found.")
    print("  Please run Step 4 first.")
    exit()

# Verify class distribution
unique, counts = np.unique(y, return_counts=True)
print(f"\n[1.1] Target class distribution:")
for cls, count in zip(unique, counts):
    pct = (count / len(y)) * 100
    print(f"  Class {cls}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 2. TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*80)
print("[2] SPLITTING DATA INTO TRAIN AND TEST SETS")
print("="*80)

# Split with stratification to maintain class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[2.1] Split completed:")
print(f"  Training set: {X_train.shape[0]:,} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"  Test set: {X_test.shape[0]:,} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

print(f"\n[2.2] Class distribution in training set:")
unique_train, counts_train = np.unique(y_train, return_counts=True)
for cls, count in zip(unique_train, counts_train):
    pct = (count / len(y_train)) * 100
    print(f"  Class {cls}: {count:,} ({pct:.2f}%)")

# ============================================================================
# 3. FEATURE SCALING
# ============================================================================

print("\n" + "="*80)
print("[3] SCALING FEATURES")
print("="*80)

print("\n[3.1] Applying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"  ✓ Features scaled to mean=0, std=1")
print(f"  Training mean: {X_train_scaled.mean():.4f}")
print(f"  Training std: {X_train_scaled.std():.4f}")

# ============================================================================
# 4. HANDLE CLASS IMBALANCE WITH SMOTE
# ============================================================================

print("\n" + "="*80)
print("[4] HANDLING CLASS IMBALANCE WITH SMOTE")
print("="*80)

print("\n[4.1] Applying SMOTE to training set...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"\n[4.2] Before SMOTE:")
unique_before, counts_before = np.unique(y_train, return_counts=True)
for cls, count in zip(unique_before, counts_before):
    print(f"  Class {cls}: {count:,}")

print(f"\n[4.3] After SMOTE:")
unique_after, counts_after = np.unique(y_train_resampled, return_counts=True)
for cls, count in zip(unique_after, counts_after):
    print(f"  Class {cls}: {count:,}")

print(f"\n  ✓ Classes balanced from 1:{counts_before[0]/counts_before[1]:.1f} to 1:1")

# ============================================================================
# 5. TRAIN BASELINE MODELS
# ============================================================================

print("\n" + "="*80)
print("[5] TRAINING BASELINE MODELS")
print("="*80)

# Dictionary to store models and results
models = {}
results = {}

# Define models
baseline_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, 
                                           max_depth=10, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42,
                                                     max_depth=5),
    'LightGBM': LGBMClassifier(random_state=42, n_jobs=-1, n_estimators=100)
}

print("\n[5.1] Training baseline models...")

for name, model in baseline_models.items():
    print(f"\n  Training {name}...")
    
    # Train model
    model.fit(X_train_resampled, y_train_resampled)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store model and results
    models[name] = model
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"    Accuracy: {accuracy:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall: {recall:.4f}")
    print(f"    F1-Score: {f1:.4f}")
    print(f"    ROC-AUC: {roc_auc:.4f}")

# ============================================================================
# 6. HYPERPARAMETER TUNING FOR BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("[6] HYPERPARAMETER TUNING")
print("="*80)

# Find best baseline model by F1-score
best_baseline = max(results.items(), key=lambda x: x[1]['f1_score'])[0]
print(f"\n[6.1] Best baseline model: {best_baseline}")
print(f"  F1-Score: {results[best_baseline]['f1_score']:.4f}")

# Tune Random Forest (assuming it's one of the best)
if best_baseline == 'Random Forest' or True:  # Always tune RF for this example
    print("\n[6.2] Tuning Random Forest hyperparameters...")
    
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Use StratifiedKFold for cross-validation
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='f1',
        n_jobs=-1, verbose=1
    )
    
    print("  Running grid search (this may take several minutes)...")
    grid_search.fit(X_train_resampled, y_train_resampled)
    
    print(f"\n[6.3] Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    # Evaluate tuned model
    best_model = grid_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test_scaled)
    y_pred_proba_tuned = best_model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    tuned_results = {
        'accuracy': accuracy_score(y_test, y_pred_tuned),
        'precision': precision_score(y_test, y_pred_tuned),
        'recall': recall_score(y_test, y_pred_tuned),
        'f1_score': f1_score(y_test, y_pred_tuned),
        'roc_auc': roc_auc_score(y_test, y_pred_proba_tuned),
        'y_pred': y_pred_tuned,
        'y_pred_proba': y_pred_proba_tuned
    }
    
    print(f"\n[6.4] Tuned model performance:")
    print(f"  Accuracy: {tuned_results['accuracy']:.4f}")
    print(f"  Precision: {tuned_results['precision']:.4f}")
    print(f"  Recall: {tuned_results['recall']:.4f}")
    print(f"  F1-Score: {tuned_results['f1_score']:.4f}")
    print(f"  ROC-AUC: {tuned_results['roc_auc']:.4f}")
    
    # Add tuned model to results
    models['Random Forest (Tuned)'] = best_model
    results['Random Forest (Tuned)'] = tuned_results

# ============================================================================
# 7. MODEL COMPARISON
# ============================================================================

print("\n" + "="*80)
print("[7] MODEL PERFORMANCE COMPARISON")
print("="*80)

# Create comparison dataframe
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [r['accuracy'] for r in results.values()],
    'Precision': [r['precision'] for r in results.values()],
    'Recall': [r['recall'] for r in results.values()],
    'F1-Score': [r['f1_score'] for r in results.values()],
    'ROC-AUC': [r['roc_auc'] for r in results.values()]
})

comparison_df = comparison_df.sort_values('F1-Score', ascending=False)

print("\n[7.1] Performance metrics for all models:")
print(comparison_df.to_string(index=False))

# Identify best model
best_model_name = comparison_df.iloc[0]['Model']
print(f"\n[7.2] Best model: {best_model_name}")
print(f"  F1-Score: {comparison_df.iloc[0]['F1-Score']:.4f}")

# Save comparison
comparison_df.to_csv('../reports/05_model_comparison.csv', index=False)
print("\n✓ Model comparison saved as '05_model_comparison.csv'")

# ============================================================================
# 8. DETAILED EVALUATION OF BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("[8] DETAILED EVALUATION OF BEST MODEL")
print("="*80)

best_model = models[best_model_name]
best_results = results[best_model_name]

# Classification report
print(f"\n[8.1] Classification Report for {best_model_name}:")
print(classification_report(y_test, best_results['y_pred'], 
                          target_names=['Not Readmitted', 'Readmitted']))

# Confusion matrix
cm = confusion_matrix(y_test, best_results['y_pred'])
print(f"\n[8.2] Confusion Matrix:")
print(f"                  Predicted")
print(f"                  0       1")
print(f"Actual   0     {cm[0,0]:6d}  {cm[0,1]:6d}")
print(f"         1     {cm[1,0]:6d}  {cm[1,1]:6d}")

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("[9] CREATING VISUALIZATIONS")
print("="*80)

print("\n[9.1] Generating performance visualizations...")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Model Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
x = np.arange(len(comparison_df))
width = 0.15

for i, metric in enumerate(metrics):
    ax1.bar(x + i*width, comparison_df[metric], width, label=metric)

ax1.set_xlabel('Models')
ax1.set_ylabel('Score')
ax1.set_title('Model Performance Comparison', fontweight='bold')
ax1.set_xticks(x + width * 2)
ax1.set_xticklabels(comparison_df['Model'], rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Plot 2: Confusion Matrix Heatmap
ax2 = plt.subplot(2, 3, 2)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
            xticklabels=['Not Readmit', 'Readmit'],
            yticklabels=['Not Readmit', 'Readmit'])
ax2.set_title(f'Confusion Matrix - {best_model_name}', fontweight='bold')
ax2.set_ylabel('Actual')
ax2.set_xlabel('Predicted')

# Plot 3: ROC Curves
ax3 = plt.subplot(2, 3, 3)
for name, result in results.items():
    fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
    ax3.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")

ax3.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('ROC Curves', fontweight='bold')
ax3.legend(loc='lower right')
ax3.grid(alpha=0.3)

# Plot 4: Precision-Recall Curve
ax4 = plt.subplot(2, 3, 4)
precision_vals, recall_vals, _ = precision_recall_curve(
    y_test, best_results['y_pred_proba']
)
ax4.plot(recall_vals, precision_vals, linewidth=2)
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.set_title(f'Precision-Recall Curve - {best_model_name}', fontweight='bold')
ax4.grid(alpha=0.3)

# Plot 5: Feature Importance (if available)
ax5 = plt.subplot(2, 3, 5)
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[-15:]  # Top 15
    
    ax5.barh(range(len(indices)), importances[indices], color='steelblue')
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([X.columns[i] for i in indices])
    ax5.set_xlabel('Feature Importance')
    ax5.set_title('Top 15 Feature Importances', fontweight='bold')
else:
    ax5.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
            ha='center', va='center', fontsize=12)
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')

# Plot 6: Metrics Comparison
ax6 = plt.subplot(2, 3, 6)
best_metrics = [best_results['accuracy'], best_results['precision'],
                best_results['recall'], best_results['f1_score'], best_results['roc_auc']]
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

bars = ax6.bar(metric_names, best_metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
ax6.set_ylabel('Score')
ax6.set_title(f'{best_model_name} - Final Metrics', fontweight='bold')
ax6.set_ylim(0, 1)
ax6.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, best_metrics):
    height = bar.get_height()
    ax6.text(bar.get_x() + bar.get_width()/2., height,
            f'{value:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('../figures/05_model_evaluation.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: 05_model_evaluation.png")

# ============================================================================
# 10. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "="*80)
print("[10] FEATURE IMPORTANCE ANALYSIS")
print("="*80)

if hasattr(best_model, 'feature_importances_'):
    print(f"\n[10.1] Top 20 most important features for {best_model_name}:")
    
    importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance_df.head(20).to_string(index=False))
    
    # Save feature importance
    feature_importance_df.to_csv('../reports/05_feature_importance.csv', index=False)
    print("\n✓ Feature importance saved as '05_feature_importance.csv'")
else:
    print(f"\n  Feature importance not available for {best_model_name}")

# ============================================================================
# 11. SAVE BEST MODEL
# ============================================================================

print("\n" + "="*80)
print("[11] SAVING BEST MODEL")
print("="*80)

# Save model
with open('../models/05_best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"\n✓ Model saved as '05_best_model.pkl'")

# Save scaler
with open('../models/05_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved as '05_scaler.pkl'")

# Save model metadata
metadata = {
    'model_name': best_model_name,
    'model_type': type(best_model).__name__,
    'n_features': X.shape[1],
    'feature_names': X.columns.tolist(),
    'performance': {
        'accuracy': best_results['accuracy'],
        'precision': best_results['precision'],
        'recall': best_results['recall'],
        'f1_score': best_results['f1_score'],
        'roc_auc': best_results['roc_auc']
    },
    'training_samples': len(X_train_resampled),
    'test_samples': len(X_test),
    'class_balance_method': 'SMOTE'
}


with open('../models/05_model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print("✓ Model metadata saved as '05_model_metadata.json'")

# ============================================================================
# 13. PER MODEL RESULTS FIGURES
# ============================================================================

print("\n" + "=" * 80)
print("[13] PER MODEL RESULTS FIGURES")
print("=" * 80)

for model in models.keys():
    selected_model = models[model]
    selected_results = results[model]

    # Confusion matrix
    cm = confusion_matrix(y_test, selected_results['y_pred'])

    fig = plt.figure(figsize=(20, 6))

    # Plot 1: Confusion Matrix Heatmap
    ax1 = plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Readmit', 'Readmit'],
                yticklabels=['Not Readmit', 'Readmit'])
    ax1.set_title(f'Confusion Matrix - {model}', fontweight='bold')
    ax1.set_ylabel('Actual')
    ax1.set_xlabel('Predicted')


    # Plot 2: Precision-Recall Curve
    ax2 = plt.subplot(1, 3, 2)
    precision_vals, recall_vals, _ = precision_recall_curve(
        y_test, selected_results['y_pred_proba']
    )
    ax2.plot(recall_vals, precision_vals, linewidth=2)
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title(f'Precision-Recall Curve - {model}', fontweight='bold')
    ax2.grid(alpha=0.3)

    # Plot 3: Metrics Comparison
    ax3 = plt.subplot(1, 3, 3)
    best_metrics = [selected_results['accuracy'], selected_results['precision'],
                    selected_results['recall'], selected_results['f1_score'], selected_results['roc_auc']]
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    bars = ax3.bar(metric_names, best_metrics, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
    ax3.set_ylabel('Score')
    ax3.set_title(f'{model} - Final Metrics', fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, value in zip(bars, best_metrics):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.3f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f'../figures/05_model_evaluation_{model}.png', dpi=300, bbox_inches='tight')
    print("  ✓ Saved: 05_model_evaluation.png")

# ============================================================================
# 14. FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("[12] PROJECT SUMMARY")
print("="*80)

print(f"""
✓ Data Processing Complete:
  - Original dataset: 101,766 encounters
  - After preprocessing: 71,518 unique patients
  - Final features: {X.shape[1]} (selected from 155)
  - Target: Binary classification (readmitted <30 days)

✓ Model Development Complete:
  - Models trained: {len(models)}
  - Best model: {best_model_name}
  - Evaluation: {len(X_test):,} test samples

✓ Performance Achieved:
  - Accuracy: {best_results['accuracy']:.1%}
  - Precision: {best_results['precision']:.1%}
  - Recall: {best_results['recall']:.1%}
  - F1-Score: {best_results['f1_score']:.3f}
  - ROC-AUC: {best_results['roc_auc']:.3f}

✓ Key Insights:
  - Class imbalance addressed with SMOTE
  - Feature selection reduced noise and improved generalization
  - Model can identify high-risk patients for targeted intervention
""")

print("\n" + "="*80)
print("STEP 5 COMPLETE!")
print("="*80)
print("\nGenerated Files:")
print("  - 05_model_comparison.csv")
print("  - 05_feature_importance.csv")
print("  - 05_model_evaluation.png")
print("  - 05_best_model.pkl")
print("  - 05_scaler.pkl")
print("  - 05_model_metadata.json")
print("\nNext Steps:")
print("  1. Review model evaluation visualizations")
print("  2. Analyze feature importance")
print("  3. Prepare final presentation")
print("  4. Document findings in final report")
print("="*80)
