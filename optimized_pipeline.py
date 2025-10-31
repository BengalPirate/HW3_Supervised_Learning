"""
Optimized Classification Pipeline - All Improvements
Implements: hyperparameter tuning, feature engineering, ensemble methods,
SMOTE for class imbalance, and more estimators
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("Optimized Classification Pipeline - All Improvements")
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

X_base = imputer.fit_transform(train_data.values)
X_base = scaler.fit_transform(X_base)
y_full = train_labels['label'].values

X_test_base = imputer.transform(test_data.values)
X_test_base = scaler.transform(X_test_base)

print(f"Training data: {X_base.shape}")
print(f"Test data: {X_test_base.shape}")

# Feature Engineering - Create interaction features from top features
print("\nFeature Engineering...")
top_features = [85, 357, 336, 86, 356, 337, 84, 225, 124, 236, 144, 354, 143, 205, 323]

def create_interaction_features(X_data, top_feat):
    """Create polynomial and interaction features from top features"""
    X_interact = []

    # Original features
    X_interact.append(X_data)

    # Squared features for top features
    for feat in top_feat[:10]:  # Top 10
        X_interact.append((X_data[:, feat] ** 2).reshape(-1, 1))

    # Pairwise interactions for top 5 features
    for i in range(min(5, len(top_feat))):
        for j in range(i+1, min(5, len(top_feat))):
            interaction = (X_data[:, top_feat[i]] * X_data[:, top_feat[j]]).reshape(-1, 1)
            X_interact.append(interaction)

    # Statistical features for top features
    top_data = X_data[:, top_feat[:10]]
    X_interact.append(np.mean(top_data, axis=1).reshape(-1, 1))
    X_interact.append(np.std(top_data, axis=1).reshape(-1, 1))
    X_interact.append(np.max(top_data, axis=1).reshape(-1, 1))
    X_interact.append(np.min(top_data, axis=1).reshape(-1, 1))

    return np.hstack(X_interact)

X_full = create_interaction_features(X_base, top_features)
X_test_processed = create_interaction_features(X_test_base, top_features)

print(f"After feature engineering: {X_full.shape}")

# Train/validation split
print("\nCreating train/validation split (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# Ensemble: XGBoost + LightGBM
print("\n" + "="*70)
print("Training Ensemble Models (XGBoost + LightGBM)")
print("="*70)

xgb_models = {}
lgb_models = {}
val_probas_xgb = np.zeros((len(X_val), 4))
val_probas_lgb = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    print(f"\n{'='*70}")
    print(f"Class {cls} vs Rest")
    print(f"{'='*70}")

    y_binary_train = (y_train == cls).astype(int)
    y_binary_val = (y_val == cls).astype(int)

    # Apply SMOTE for class imbalance
    print("  Applying SMOTE for class balance...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_binary_train)

    print(f"  Before SMOTE: {np.bincount(y_binary_train)}")
    print(f"  After SMOTE: {np.bincount(y_train_balanced)}")

    # XGBoost with more estimators
    print("  Training XGBoost (500 estimators)...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.15,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        early_stopping_rounds=30,
        verbose=False
    )

    xgb_clf.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_val, y_binary_val)],
        verbose=False
    )

    xgb_models[cls] = xgb_clf
    val_probas_xgb[:, cls-1] = xgb_clf.predict_proba(X_val)[:, 1]

    xgb_auc = roc_auc_score(y_binary_val, val_probas_xgb[:, cls-1])
    print(f"  XGBoost AUC: {xgb_auc:.4f}, Best iteration: {xgb_clf.best_iteration}")

    # LightGBM with more estimators
    print("  Training LightGBM (500 estimators)...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=12,
        num_leaves=60,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    lgb_clf.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_val, y_binary_val)],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )

    lgb_models[cls] = lgb_clf
    val_probas_lgb[:, cls-1] = lgb_clf.predict_proba(X_val)[:, 1]

    lgb_auc = roc_auc_score(y_binary_val, val_probas_lgb[:, cls-1])
    print(f"  LightGBM AUC: {lgb_auc:.4f}, Best iteration: {lgb_clf.best_iteration_}")

# Ensemble predictions - weighted average
print("\n" + "="*70)
print("Creating Ensemble Predictions")
print("="*70)

# Find optimal weights
weights = [0.5, 0.5]  # Start with equal weights
best_acc = 0
best_weights = weights

print("\nOptimizing ensemble weights...")
for w1 in np.arange(0.3, 0.8, 0.05):
    w2 = 1 - w1
    val_probas_ensemble = w1 * val_probas_xgb + w2 * val_probas_lgb
    val_preds_ensemble = np.argmax(val_probas_ensemble, axis=1) + 1
    acc = accuracy_score(y_val, val_preds_ensemble)

    if acc > best_acc:
        best_acc = acc
        best_weights = [w1, w2]

print(f"Best weights: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}")

val_probas_ensemble = best_weights[0] * val_probas_xgb + best_weights[1] * val_probas_lgb
val_preds_ensemble = np.argmax(val_probas_ensemble, axis=1) + 1

# Validation performance
print("\n" + "="*70)
print("Validation Performance - Ensemble")
print("="*70)

acc = accuracy_score(y_val, val_preds_ensemble)
print(f"\nAccuracy: {acc:.4f} ({acc*100:.2f}%)")

for cls in [1, 2, 3, 4]:
    y_binary = (y_val == cls).astype(int)
    auc = roc_auc_score(y_binary, val_probas_ensemble[:, cls-1])
    print(f"Class {cls} AUC: {auc:.4f}")

avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), val_probas_ensemble[:, cls-1])
                    for cls in [1, 2, 3, 4]])
print(f"Average AUC: {avg_auc:.4f}")

# Compare individual models
print("\n" + "="*70)
print("Model Comparison")
print("="*70)

xgb_preds = np.argmax(val_probas_xgb, axis=1) + 1
lgb_preds = np.argmax(val_probas_lgb, axis=1) + 1

print(f"XGBoost alone:  {accuracy_score(y_val, xgb_preds):.4f} ({accuracy_score(y_val, xgb_preds)*100:.2f}%)")
print(f"LightGBM alone: {accuracy_score(y_val, lgb_preds):.4f} ({accuracy_score(y_val, lgb_preds)*100:.2f}%)")
print(f"Ensemble:       {acc:.4f} ({acc*100:.2f}%)")

# Plot ROC curves
print("\nGenerating ROC curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for cls_idx, cls in enumerate([1, 2, 3, 4]):
    y_binary = (y_val == cls).astype(int)

    # XGBoost ROC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_binary, val_probas_xgb[:, cls_idx])
    auc_xgb = roc_auc_score(y_binary, val_probas_xgb[:, cls_idx])

    # LightGBM ROC
    fpr_lgb, tpr_lgb, _ = roc_curve(y_binary, val_probas_lgb[:, cls_idx])
    auc_lgb = roc_auc_score(y_binary, val_probas_lgb[:, cls_idx])

    # Ensemble ROC
    fpr_ens, tpr_ens, _ = roc_curve(y_binary, val_probas_ensemble[:, cls_idx])
    auc_ens = roc_auc_score(y_binary, val_probas_ensemble[:, cls_idx])

    ax = axes[cls_idx]
    ax.plot(fpr_xgb, tpr_xgb, linewidth=2, label=f'XGBoost (AUC = {auc_xgb:.3f})', color='blue', alpha=0.7)
    ax.plot(fpr_lgb, tpr_lgb, linewidth=2, label=f'LightGBM (AUC = {auc_lgb:.3f})', color='green', alpha=0.7)
    ax.plot(fpr_ens, tpr_ens, linewidth=2, label=f'Ensemble (AUC = {auc_ens:.3f})', color='red')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - Class {cls} vs Rest', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_optimized.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_optimized.png")
plt.close()

# Feature importance (from XGBoost)
print("\nExtracting feature importance...")
avg_importance = np.mean([xgb_models[cls].feature_importances_ for cls in [1, 2, 3, 4]], axis=0)
top_features_idx = np.argsort(avg_importance)[::-1][:20]

print("\nTop 20 Most Important Features:")
for i, feat_idx in enumerate(top_features_idx):
    print(f"  {i+1}. Feature {feat_idx}: {avg_importance[feat_idx]:.6f}")

# Plot feature importance
plt.figure(figsize=(12, 6))
plt.bar(range(20), avg_importance[top_features_idx], color='steelblue', edgecolor='black')
plt.xlabel('Feature Rank', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.title('Top 20 Feature Importance - Optimized Ensemble', fontsize=14, fontweight='bold')
plt.xticks(range(20), [f'F{i}' for i in top_features_idx], rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_optimized.png', dpi=300, bbox_inches='tight')
print("Saved: feature_importance_optimized.png")
plt.close()

# Retrain on full training data
print("\n" + "="*70)
print("Retraining on Full Training Data for Test Predictions")
print("="*70)

final_xgb_models = {}
final_lgb_models = {}

for cls in [1, 2, 3, 4]:
    print(f"\nClass {cls} vs Rest...")
    y_binary = (y_full == cls).astype(int)

    # Apply SMOTE
    print("  Applying SMOTE...")
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_full_balanced, y_full_balanced = smote.fit_resample(X_full, y_binary)

    # XGBoost
    print("  Training XGBoost...")
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=9,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.15,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbose=False
    )
    xgb_clf.fit(X_full_balanced, y_full_balanced)
    final_xgb_models[cls] = xgb_clf

    # LightGBM
    print("  Training LightGBM...")
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=12,
        num_leaves=60,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.15,
        reg_lambda=1.2,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_clf.fit(X_full_balanced, y_full_balanced)
    final_lgb_models[cls] = lgb_clf

# Generate test predictions
print("\nGenerating test predictions...")
test_probas_xgb = np.zeros((len(X_test_processed), 4))
test_probas_lgb = np.zeros((len(X_test_processed), 4))

for cls in [1, 2, 3, 4]:
    test_probas_xgb[:, cls-1] = final_xgb_models[cls].predict_proba(X_test_processed)[:, 1]
    test_probas_lgb[:, cls-1] = final_lgb_models[cls].predict_proba(X_test_processed)[:, 1]

# Ensemble test predictions
test_probas = best_weights[0] * test_probas_xgb + best_weights[1] * test_probas_lgb
test_preds = np.argmax(test_probas, axis=1) + 1

# Save predictions in required format
output = np.column_stack([test_probas, test_preds])
np.savetxt('testLabel_optimized.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
print("Saved: testLabel_optimized.txt")

# Save results summary
with open('results_summary_optimized.txt', 'w') as f:
    f.write("CSC 621 - HW3 Optimized Classification Results\n")
    f.write("="*70 + "\n\n")
    f.write("Improvements Applied:\n")
    f.write("  - Feature engineering (interactions, polynomials, statistics)\n")
    f.write("  - SMOTE for class imbalance\n")
    f.write("  - XGBoost + LightGBM ensemble\n")
    f.write("  - 500 estimators per model\n")
    f.write("  - Optimized ensemble weights\n\n")
    f.write("Validation Performance (20% holdout):\n")
    f.write("-"*70 + "\n")
    f.write(f"Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
    f.write(f"Average AUC: {avg_auc:.4f}\n\n")
    f.write(f"XGBoost alone:  {accuracy_score(y_val, xgb_preds):.4f} ({accuracy_score(y_val, xgb_preds)*100:.2f}%)\n")
    f.write(f"LightGBM alone: {accuracy_score(y_val, lgb_preds):.4f} ({accuracy_score(y_val, lgb_preds)*100:.2f}%)\n")
    f.write(f"Ensemble:       {acc:.4f} ({acc*100:.2f}%)\n\n")
    f.write(f"Ensemble weights: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}\n\n")
    for cls in [1, 2, 3, 4]:
        y_binary = (y_val == cls).astype(int)
        auc = roc_auc_score(y_binary, val_probas_ensemble[:, cls-1])
        f.write(f"Class {cls} AUC: {auc:.4f}\n")

print("\n" + "="*70)
print("OPTIMIZED PIPELINE COMPLETE!")
print("="*70)
print(f"\nValidation Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print(f"Target: 0.9282 (92.82%)")
if acc >= 0.9282:
    print(f"SUCCESS! Beat the benchmark by {(acc - 0.9282)*100:.2f}%")
else:
    print(f"Gap: {(0.9282 - acc)*100:.2f}% below target")
print(f"\nAverage AUC: {avg_auc:.4f}")
print(f"\nEnsemble weights: XGBoost={best_weights[0]:.2f}, LightGBM={best_weights[1]:.2f}")
print("\nGenerated Files:")
print("  - testLabel_optimized.txt: Test predictions (5-column format)")
print("  - roc_curves_optimized.png: ROC curves comparison")
print("  - feature_importance_optimized.png: Top 20 features")
print("  - results_summary_optimized.txt: Detailed results")
