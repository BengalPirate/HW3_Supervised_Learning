"""
Fast XGBoost Classification with Validation Split
Optimized for speed - trains only XGBoost with a single train/val split
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("Fast XGBoost Classification Pipeline")
print("="*70)

# Load data
print("\nLoading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

# Preprocess
print("Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='median')
scaler = RobustScaler()

X_full = imputer.fit_transform(train_data.values)
X_full = scaler.fit_transform(X_full)
y_full = train_labels['label'].values

X_test_processed = imputer.transform(test_data.values)
X_test_processed = scaler.transform(X_test_processed)

print(f"Training data: {X_full.shape}")
print(f"Test data: {X_test_processed.shape}")
print(f"Class distribution: {np.bincount(y_full)}")

# Train/validation split
print("\nCreating train/validation split (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# Train XGBoost for each class (One-vs-Rest)
print("\n" + "="*70)
print("Training XGBoost Classifiers (One-vs-Rest)")
print("="*70)

models = {}
val_probas = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    print(f"\nTraining Class {cls} vs Rest...")
    y_binary_train = (y_train == cls).astype(int)
    y_binary_val = (y_val == cls).astype(int)

    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_binary_train == 0)
    pos_count = np.sum(y_binary_train == 1)
    scale_pos_weight = neg_count / pos_count

    print(f"  Positive samples: {pos_count}, Negative: {neg_count}, Scale: {scale_pos_weight:.2f}")

    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=20,
        verbose=False
    )

    # Train with early stopping
    clf.fit(
        X_train, y_binary_train,
        eval_set=[(X_val, y_binary_val)],
        verbose=False
    )

    models[cls] = clf
    val_probas[:, cls-1] = clf.predict_proba(X_val)[:, 1]

    # Validation metrics
    auc = roc_auc_score(y_binary_val, val_probas[:, cls-1])
    print(f"  Validation AUC: {auc:.4f}")
    print(f"  Best iteration: {clf.best_iteration}")

# Overall validation performance
print("\n" + "="*70)
print("Validation Performance")
print("="*70)

val_preds = np.argmax(val_probas, axis=1) + 1
acc = accuracy_score(y_val, val_preds)

print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

for cls in [1, 2, 3, 4]:
    y_binary = (y_val == cls).astype(int)
    auc = roc_auc_score(y_binary, val_probas[:, cls-1])
    print(f"Class {cls} AUC: {auc:.4f}")

avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), val_probas[:, cls-1])
                    for cls in [1, 2, 3, 4]])
print(f"Average AUC: {avg_auc:.4f}")

# Plot ROC curves
print("\nGenerating ROC curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for cls_idx, cls in enumerate([1, 2, 3, 4]):
    y_binary = (y_val == cls).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, val_probas[:, cls_idx])
    auc_score = roc_auc_score(y_binary, val_probas[:, cls_idx])

    ax = axes[cls_idx]
    ax.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc_score:.3f})', color='blue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - Class {cls} vs Rest', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_xgboost.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_xgboost.png")
plt.close()

# Feature importance
print("\nExtracting feature importance...")
avg_importance = np.mean([models[cls].feature_importances_ for cls in [1, 2, 3, 4]], axis=0)
top_features_idx = np.argsort(avg_importance)[::-1][:20]

print("\nTop 20 Most Important Features:")
for i, feat_idx in enumerate(top_features_idx):
    print(f"  {i+1}. Feature {feat_idx}: {avg_importance[feat_idx]:.6f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(20), avg_importance[top_features_idx], color='steelblue', edgecolor='black')
plt.xlabel('Feature Rank', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importance - XGBoost', fontsize=14, fontweight='bold')
plt.xticks(range(20), [f'F{i}' for i in top_features_idx], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance_xgboost.png")
plt.close()

# Save feature importance to CSV
importance_df = pd.DataFrame({
    'feature_index': range(len(avg_importance)),
    'importance': avg_importance
}).sort_values('importance', ascending=False)
importance_df.to_csv('feature_importance_xgboost.csv', index=False)
print("Saved: feature_importance_xgboost.csv")

# Retrain on full training data
print("\n" + "="*70)
print("Retraining on Full Training Data for Test Predictions")
print("="*70)

final_models = {}
for cls in [1, 2, 3, 4]:
    print(f"\nTraining Class {cls} vs Rest on full data...")
    y_binary = (y_full == cls).astype(int)

    neg_count = np.sum(y_binary == 0)
    pos_count = np.sum(y_binary == 1)
    scale_pos_weight = neg_count / pos_count

    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbose=False
    )

    clf.fit(X_full, y_binary)
    final_models[cls] = clf
    print(f"  Training complete")

# Generate test predictions
print("\nGenerating test predictions...")
test_probas = np.zeros((len(X_test_processed), 4))
for cls in [1, 2, 3, 4]:
    test_probas[:, cls-1] = final_models[cls].predict_proba(X_test_processed)[:, 1]

test_preds = np.argmax(test_probas, axis=1) + 1

# Save predictions in required format
output = np.column_stack([test_probas, test_preds])
np.savetxt('testLabel.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
print("Saved: testLabel.txt")

# Save results summary
with open('results_summary_xgboost.txt', 'w') as f:
    f.write("CSC 621 - HW3 XGBoost Classification Results\n")
    f.write("="*70 + "\n\n")
    f.write("Validation Performance (20% holdout):\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"Average AUC: {avg_auc:.4f}\n\n")
    for cls in [1, 2, 3, 4]:
        y_binary = (y_val == cls).astype(int)
        auc = roc_auc_score(y_binary, val_probas[:, cls-1])
        f.write(f"Class {cls} AUC: {auc:.4f}\n")

print("\n" + "="*70)
print("PIPELINE COMPLETE!")
print("="*70)
print(f"\nValidation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Average AUC: {avg_auc:.4f}")
print("\nGenerated Files:")
print("  - testLabel.txt: Test predictions (5-column format)")
print("  - roc_curves_xgboost.png: ROC curves")
print("  - feature_importance_xgboost.png: Top 20 features")
print("  - feature_importance_xgboost.csv: All feature importance")
print("  - results_summary_xgboost.txt: Detailed results")
