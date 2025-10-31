"""
Final Best Pipeline - Combining Best Strategies
Uses: LightGBM with aggressive hyperparameters, enhanced feature engineering,
and stacked ensemble
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("Final Best Pipeline - Stacked Ensemble with Feature Engineering")
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

# Enhanced Feature Engineering
print("\nAdvanced Feature Engineering...")
top_features = [85, 357, 336, 86, 356, 337, 84, 225, 124, 236, 144, 354, 143, 205, 323, 165, 185, 248, 105, 194]

def create_advanced_features(X_data, top_feat):
    """Create comprehensive feature set"""
    features = [X_data]

    # Squared and cubed features
    for feat in top_feat[:20]:
        features.append((X_data[:, feat] ** 2).reshape(-1, 1))
        features.append((X_data[:, feat] ** 3).reshape(-1, 1))

    # Pairwise interactions (more combinations)
    for i in range(min(10, len(top_feat))):
        for j in range(i+1, min(10, len(top_feat))):
            interaction = (X_data[:, top_feat[i]] * X_data[:, top_feat[j]]).reshape(-1, 1)
            features.append(interaction)
            # Ratio features
            ratio = np.divide(X_data[:, top_feat[i]], X_data[:, top_feat[j]] + 1e-8).reshape(-1, 1)
            features.append(ratio)

    # Statistical features on different subsets
    for subset_size in [10, 15, 20]:
        top_data = X_data[:, top_feat[:subset_size]]
        features.append(np.mean(top_data, axis=1).reshape(-1, 1))
        features.append(np.std(top_data, axis=1).reshape(-1, 1))
        features.append(np.max(top_data, axis=1).reshape(-1, 1))
        features.append(np.min(top_data, axis=1).reshape(-1, 1))
        features.append(np.median(top_data, axis=1).reshape(-1, 1))
        features.append(np.percentile(top_data, 25, axis=1).reshape(-1, 1))
        features.append(np.percentile(top_data, 75, axis=1).reshape(-1, 1))

    return np.hstack(features)

X_full = create_advanced_features(X_base, top_features)
X_test_processed = create_advanced_features(X_test_base, top_features)

print(f"After feature engineering: {X_full.shape}")

# Train/validation split
print("\nCreating train/validation split (85/15 for more training data)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.15, random_state=42, stratify=y_full
)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}")

# Train base models
print("\n" + "="*70)
print("Training Base Models")
print("="*70)

# Model 1: LightGBM with aggressive params
print("\n[1/3] Training LightGBM-1...")
lgb1_models = {}
val_probas_lgb1 = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    y_binary_train = (y_train == cls).astype(int)
    y_binary_val = (y_val == cls).astype(int)

    neg_count = np.sum(y_binary_train == 0)
    pos_count = np.sum(y_binary_train == 1)
    scale_pos_weight = neg_count / pos_count

    clf = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=20,
        num_leaves=120,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_binary_train)
    lgb1_models[cls] = clf
    val_probas_lgb1[:, cls-1] = clf.predict_proba(X_val)[:, 1]

acc_lgb1 = accuracy_score(y_val, np.argmax(val_probas_lgb1, axis=1) + 1)
print(f"LightGBM-1 Accuracy: {acc_lgb1:.4f} ({acc_lgb1*100:.2f}%)")

# Model 2: LightGBM with different params
print("\n[2/3] Training LightGBM-2...")
lgb2_models = {}
val_probas_lgb2 = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    y_binary_train = (y_train == cls).astype(int)

    neg_count = np.sum(y_binary_train == 0)
    pos_count = np.sum(y_binary_train == 1)
    scale_pos_weight = neg_count / pos_count

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=18,
        num_leaves=100,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=43,  # Different seed
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_train, y_binary_train)
    lgb2_models[cls] = clf
    val_probas_lgb2[:, cls-1] = clf.predict_proba(X_val)[:, 1]

acc_lgb2 = accuracy_score(y_val, np.argmax(val_probas_lgb2, axis=1) + 1)
print(f"LightGBM-2 Accuracy: {acc_lgb2:.4f} ({acc_lgb2*100:.2f}%)")

# Model 3: XGBoost
print("\n[3/3] Training XGBoost...")
xgb_models = {}
val_probas_xgb = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    y_binary_train = (y_train == cls).astype(int)

    neg_count = np.sum(y_binary_train == 0)
    pos_count = np.sum(y_binary_train == 1)
    scale_pos_weight = neg_count / pos_count

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=11,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        reg_alpha=0.01,
        reg_lambda=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbose=False
    )
    clf.fit(X_train, y_binary_train)
    xgb_models[cls] = clf
    val_probas_xgb[:, cls-1] = clf.predict_proba(X_val)[:, 1]

acc_xgb = accuracy_score(y_val, np.argmax(val_probas_xgb, axis=1) + 1)
print(f"XGBoost Accuracy: {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")

# Stacking - train meta-learner
print("\n" + "="*70)
print("Training Stacked Meta-Learner")
print("="*70)

# Prepare stacking features
stacked_features_val = np.column_stack([val_probas_lgb1, val_probas_lgb2, val_probas_xgb])

# Train meta-learner for each class
meta_models = {}
final_val_probas = np.zeros((len(X_val), 4))

for cls in [1, 2, 3, 4]:
    y_binary_val = (y_val == cls).astype(int)

    meta_clf = LogisticRegression(max_iter=1000, random_state=42, C=0.1)
    meta_clf.fit(stacked_features_val, y_binary_val)
    meta_models[cls] = meta_clf

    final_val_probas[:, cls-1] = meta_clf.predict_proba(stacked_features_val)[:, 1]

final_val_preds = np.argmax(final_val_probas, axis=1) + 1
final_acc = accuracy_score(y_val, final_val_preds)

print("\n" + "="*70)
print("Validation Results")
print("="*70)
print(f"\nLightGBM-1:  {acc_lgb1:.4f} ({acc_lgb1*100:.2f}%)")
print(f"LightGBM-2:  {acc_lgb2:.4f} ({acc_lgb2*100:.2f}%)")
print(f"XGBoost:     {acc_xgb:.4f} ({acc_xgb*100:.2f}%)")
print(f"STACKED:     {final_acc:.4f} ({final_acc*100:.2f}%)")

for cls in [1, 2, 3, 4]:
    y_binary = (y_val == cls).astype(int)
    auc = roc_auc_score(y_binary, final_val_probas[:, cls-1])
    print(f"Class {cls} AUC: {auc:.4f}")

avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), final_val_probas[:, cls-1])
                    for cls in [1, 2, 3, 4]])
print(f"Average AUC: {avg_auc:.4f}")

# Retrain all models on full data
print("\n" + "="*70)
print("Retraining All Models on Full Training Data")
print("="*70)

# Retrain LGB1
print("\nRetraining LightGBM-1...")
final_lgb1_models = {}
for cls in [1, 2, 3, 4]:
    y_binary = (y_full == cls).astype(int)
    neg_count = np.sum(y_binary == 0)
    pos_count = np.sum(y_binary == 1)
    scale_pos_weight = neg_count / pos_count

    clf = lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.015,
        max_depth=20,
        num_leaves=120,
        min_child_samples=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.01,
        reg_lambda=0.5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_full, y_binary)
    final_lgb1_models[cls] = clf

# Retrain LGB2
print("Retraining LightGBM-2...")
final_lgb2_models = {}
for cls in [1, 2, 3, 4]:
    y_binary = (y_full == cls).astype(int)
    neg_count = np.sum(y_binary == 0)
    pos_count = np.sum(y_binary == 1)
    scale_pos_weight = neg_count / pos_count

    clf = lgb.LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=18,
        num_leaves=100,
        min_child_samples=8,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=43,
        n_jobs=-1,
        verbose=-1
    )
    clf.fit(X_full, y_binary)
    final_lgb2_models[cls] = clf

# Retrain XGB
print("Retraining XGBoost...")
final_xgb_models = {}
for cls in [1, 2, 3, 4]:
    y_binary = (y_full == cls).astype(int)
    neg_count = np.sum(y_binary == 0)
    pos_count = np.sum(y_binary == 1)
    scale_pos_weight = neg_count / pos_count

    clf = xgb.XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=11,
        min_child_weight=1,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        reg_alpha=0.01,
        reg_lambda=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss',
        verbose=False
    )
    clf.fit(X_full, y_binary)
    final_xgb_models[cls] = clf

# Generate test predictions
print("\nGenerating test predictions...")
test_probas_lgb1 = np.zeros((len(X_test_processed), 4))
test_probas_lgb2 = np.zeros((len(X_test_processed), 4))
test_probas_xgb = np.zeros((len(X_test_processed), 4))

for cls in [1, 2, 3, 4]:
    test_probas_lgb1[:, cls-1] = final_lgb1_models[cls].predict_proba(X_test_processed)[:, 1]
    test_probas_lgb2[:, cls-1] = final_lgb2_models[cls].predict_proba(X_test_processed)[:, 1]
    test_probas_xgb[:, cls-1] = final_xgb_models[cls].predict_proba(X_test_processed)[:, 1]

# Stack test predictions
stacked_test_features = np.column_stack([test_probas_lgb1, test_probas_lgb2, test_probas_xgb])

final_test_probas = np.zeros((len(X_test_processed), 4))
for cls in [1, 2, 3, 4]:
    final_test_probas[:, cls-1] = meta_models[cls].predict_proba(stacked_test_features)[:, 1]

final_test_preds = np.argmax(final_test_probas, axis=1) + 1

# Save predictions
output = np.column_stack([final_test_probas, final_test_preds])
np.savetxt('testLabel_final.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
print("Saved: testLabel_final.txt")

# Copy as testLabel.txt (the required filename)
import shutil
shutil.copy('testLabel_final.txt', 'testLabel.txt')
print("Saved: testLabel.txt")

print("\n" + "="*70)
print("FINAL BEST PIPELINE COMPLETE!")
print("="*70)
print(f"\nValidation Accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
print(f"Target: 0.9282 (92.82%)")
if final_acc >= 0.9282:
    print(f"SUCCESS! Beat the benchmark by {(final_acc - 0.9282)*100:.2f}%")
else:
    print(f"Gap: {(0.9282 - final_acc)*100:.2f}% below target")
print(f"Average AUC: {avg_auc:.4f}")
print("\nApproach: Stacked ensemble (LightGBM x2 + XGBoost + Logistic meta-learner)")
print("Features: Advanced engineering with 452+ features")
print("\nGenerated Files:")
print("  - testLabel.txt: Final test predictions")
print("  - testLabel_final.txt: Backup of predictions")
