"""
Weighted Ensemble Optimization - Quick Win Strategy
Optimizes ensemble weights on validation set instead of using fixed weights.
Expected gain: 0.3-0.5%
Time: ~5 minutes
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import mutual_info_classif
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("WEIGHTED ENSEMBLE OPTIMIZATION - QUICK WIN")
print("="*80)

# Load data
print("\n[1/7] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"Training: {train_data.shape}, Test: {test_data.shape}")

# Preprocess
print("\n[2/7] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Selection
print("\n[3/7] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")

X_base_selected = X_base[:, selected_features]
X_test_base_selected = X_test_base[:, selected_features]

# Feature Engineering
print("\n[4/7] Advanced feature engineering...")
top_mi_features = np.argsort(mi_scores)[::-1][:15]

def create_advanced_features(X_data, top_feat_idx):
    features = [X_data]

    # Polynomial features for top 15
    for feat in top_feat_idx[:15]:
        if feat < X_data.shape[1]:
            features.append((X_data[:, feat] ** 2).reshape(-1, 1))
            features.append((X_data[:, feat] ** 3).reshape(-1, 1))

    # Interactions for top 8
    for i in range(min(8, len(top_feat_idx))):
        for j in range(i+1, min(8, len(top_feat_idx))):
            if top_feat_idx[i] < X_data.shape[1] and top_feat_idx[j] < X_data.shape[1]:
                interaction = (X_data[:, top_feat_idx[i]] * X_data[:, top_feat_idx[j]]).reshape(-1, 1)
                features.append(interaction)
                ratio = np.divide(X_data[:, top_feat_idx[i]],
                                 X_data[:, top_feat_idx[j]] + 1e-8).reshape(-1, 1)
                features.append(ratio)

    # Statistical features
    top_data = X_data[:, [f for f in top_feat_idx[:15] if f < X_data.shape[1]]]
    features.append(np.mean(top_data, axis=1).reshape(-1, 1))
    features.append(np.std(top_data, axis=1).reshape(-1, 1))
    features.append(np.max(top_data, axis=1).reshape(-1, 1))
    features.append(np.min(top_data, axis=1).reshape(-1, 1))
    features.append(np.median(top_data, axis=1).reshape(-1, 1))
    features.append(np.percentile(top_data, 25, axis=1).reshape(-1, 1))
    features.append(np.percentile(top_data, 75, axis=1).reshape(-1, 1))

    return np.hstack(features)

X_full = create_advanced_features(X_base_selected, top_mi_features)
X_test = create_advanced_features(X_test_base_selected, top_mi_features)

print(f"After feature engineering: {X_full.shape}")

# Scaling
scaler = RobustScaler()
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

y_zero_based = y - 1

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5

# Best hyperparameters from ultimate_pipeline
best_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.025,
    'num_leaves': 70,
    'max_depth': 14,
    'min_data_in_leaf': 12,
    'feature_fraction': 0.88,
    'bagging_fraction': 0.88,
    'bagging_freq': 2,
    'lambda_l1': 0.1,
    'lambda_l2': 0.4,
    'min_gain_to_split': 0.003,
    'verbose': -1,
    'is_unbalance': is_imbalanced,
}

second_best_params = {
    'objective': 'multiclass',
    'num_class': 4,
    'metric': 'multi_logloss',
    'learning_rate': 0.02,
    'num_leaves': 45,
    'max_depth': 10,
    'min_data_in_leaf': 18,
    'feature_fraction': 0.85,
    'bagging_fraction': 0.85,
    'bagging_freq': 2,
    'lambda_l1': 0.15,
    'lambda_l2': 0.6,
    'min_gain_to_split': 0.008,
    'verbose': -1,
    'is_unbalance': is_imbalanced,
}

# 5-fold CV to collect validation predictions
print("\n[5/7] Training ensemble models and collecting validation predictions...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Store models and validation predictions
ensemble_seeds_best = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777,
                       1111, 2222, 5555, 8888, 9999]
ensemble_seeds_second = [42, 123, 456]

all_fold_models_best = []
all_fold_models_second = []
val_predictions_per_model = []
val_true_labels = []
val_indices_per_fold = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
    print(f"\nFold {fold+1}/{n_folds}")
    X_train, X_val = X_full[train_idx], X_full[val_idx]
    y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]

    train_dataset = lgb.Dataset(X_train, label=y_train)
    valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

    # Train best config model
    model_best = lgb.train(
        best_params,
        train_dataset,
        num_boost_round=2000,
        valid_sets=[valid_dataset],
        callbacks=[
            early_stopping(stopping_rounds=120),
            log_evaluation(period=0)
        ]
    )

    # Train second-best config model
    model_second = lgb.train(
        second_best_params,
        train_dataset,
        num_boost_round=2000,
        valid_sets=[valid_dataset],
        callbacks=[
            early_stopping(stopping_rounds=120),
            log_evaluation(period=0)
        ]
    )

    all_fold_models_best.append(model_best)
    all_fold_models_second.append(model_second)
    val_indices_per_fold.append(val_idx)

    # Store validation predictions for this fold
    fold_val_preds = []

    # Best config predictions
    val_pred = model_best.predict(X_val, num_iteration=model_best.best_iteration)
    fold_val_preds.append(val_pred)

    # Second-best config predictions
    val_pred = model_second.predict(X_val, num_iteration=model_second.best_iteration)
    fold_val_preds.append(val_pred)

    val_predictions_per_model.append(fold_val_preds)
    val_true_labels.append(y_val)

# Concatenate all validation predictions
print("\n[6/7] Optimizing ensemble weights on validation set...")

# Reshape: [n_models, n_samples, n_classes]
n_models = 2  # best + second-best
all_val_preds = []
all_val_y = np.concatenate(val_true_labels)

for model_idx in range(n_models):
    model_preds = np.vstack([fold_preds[model_idx] for fold_preds in val_predictions_per_model])
    all_val_preds.append(model_preds)

all_val_preds = np.array(all_val_preds)  # Shape: [n_models, n_samples, n_classes]

print(f"Validation predictions shape: {all_val_preds.shape}")
print(f"Validation labels shape: {all_val_y.shape}")

# Define optimization objective
def ensemble_loss(weights):
    # weights shape: [n_models]
    # all_val_preds shape: [n_models, n_samples, n_classes]
    weighted_pred = np.tensordot(weights, all_val_preds, axes=([0], [0]))  # [n_samples, n_classes]
    pred_labels = np.argmax(weighted_pred, axis=1)
    accuracy = accuracy_score(all_val_y, pred_labels)
    return -accuracy  # Negative because we minimize

# Initial weights (equal)
initial_weights = np.ones(n_models) / n_models

print(f"\nInitial weights: {initial_weights}")
print(f"Initial validation accuracy: {-ensemble_loss(initial_weights):.4f}")

# Optimize weights
result = minimize(
    ensemble_loss,
    initial_weights,
    method='SLSQP',
    bounds=[(0, 1)] * n_models,
    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
)

optimized_weights = result.x
optimized_val_acc = -result.fun

print(f"\nOptimized weights: {optimized_weights}")
print(f"Optimized validation accuracy: {optimized_val_acc:.4f}")
print(f"Improvement: {(optimized_val_acc - (-ensemble_loss(initial_weights)))*100:.2f}%")

# Generate test predictions with optimized weights
print("\n[7/7] Generating test predictions with optimized weights...")

# Train final models on full dataset
test_preds_best = []
test_preds_second = []

avg_best_iter = int(np.mean([m.best_iteration for m in all_fold_models_best]))
avg_second_iter = int(np.mean([m.best_iteration for m in all_fold_models_second]))

print(f"\nTraining {len(ensemble_seeds_best)} models with best config (avg iter: {avg_best_iter})")
full_train = lgb.Dataset(X_full, label=y_zero_based)

for seed_idx, seed in enumerate(ensemble_seeds_best):
    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed

    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    test_pred = model.predict(X_test)
    test_preds_best.append(test_pred)
    print(f"  Model {seed_idx+1}/{len(ensemble_seeds_best)} complete")

print(f"\nTraining {len(ensemble_seeds_second)} models with second-best config (avg iter: {avg_second_iter})")

for seed_idx, seed in enumerate(ensemble_seeds_second):
    params_with_seed = second_best_params.copy()
    params_with_seed['seed'] = seed

    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_second_iter
    )

    test_pred = model.predict(X_test)
    test_preds_second.append(test_pred)
    print(f"  Model {seed_idx+1}/{len(ensemble_seeds_second)} complete")

# Apply optimized weights
# Weight for best config models
weight_best_per_model = optimized_weights[0] / len(test_preds_best)
# Weight for second-best config models
weight_second_per_model = optimized_weights[1] / len(test_preds_second)

print(f"\nPer-model weight (best config): {weight_best_per_model:.4f}")
print(f"Per-model weight (second-best config): {weight_second_per_model:.4f}")

# Weighted average
test_pred_final = np.zeros_like(test_preds_best[0])

for pred in test_preds_best:
    test_pred_final += weight_best_per_model * pred

for pred in test_preds_second:
    test_pred_final += weight_second_per_model * pred

test_labels = np.argmax(test_pred_final, axis=1) + 1

# Save
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel_weighted_optimized.txt', output, delimiter='\\t',
           fmt='%.6f\\t%.6f\\t%.6f\\t%.6f\\t%d')

print("\\n" + "="*80)
print("✓ WEIGHTED ENSEMBLE OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\\nOptimized Validation Accuracy: {optimized_val_acc:.4f} ({optimized_val_acc*100:.2f}%)")
print(f"Baseline (equal weights): {-ensemble_loss(initial_weights):.4f}")
print(f"Improvement: +{(optimized_val_acc - (-ensemble_loss(initial_weights)))*100:.2f}%")

print(f"\\nOptimized Weights:")
print(f"  Best config (15 models): {optimized_weights[0]:.4f}")
print(f"  Second-best config (3 models): {optimized_weights[1]:.4f}")

print(f"\\nFile saved: testLabel_weighted_optimized.txt")
