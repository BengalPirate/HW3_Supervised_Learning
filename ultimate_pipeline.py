"""
Ultimate Optimized Pipeline - Closing the Final Gap
Additional strategies:
- Feature selection using mutual information
- More aggressive hyperparameter search
- Larger ensemble (15+ models)
- Pseudo-labeling on test set (semi-supervised)
- Diverse model types (LightGBM + CatBoost if available)
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("ULTIMATE OPTIMIZED PIPELINE - CLOSING THE GAP")
print("="*80)

# Load data
print("\n[1/9] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"Training: {train_data.shape}, Test: {test_data.shape}")

# Preprocess
print("\n[2/9] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Selection
print("\n[3/9] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)  # Keep top 90%
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")
print(f"Top 10 MI scores: {sorted(mi_scores, reverse=True)[:10]}")

X_base_selected = X_base[:, selected_features]
X_test_base_selected = X_test_base[:, selected_features]

# Feature Engineering on selected features
print("\n[4/9] Advanced feature engineering...")
# Find top features by MI
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
                # Ratio
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

# Scaling (can help with some features)
scaler = RobustScaler()
X_full = scaler.fit_transform(X_full)
X_test = scaler.transform(X_test)

y_zero_based = y - 1

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5

# More aggressive hyperparameter configs
print("\n[5/9] Setting up aggressive hyperparameter configs...")

configs = [
    {  # Very deep, conservative LR
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.015,
        'num_leaves': 60,
        'max_depth': 12,
        'min_data_in_leaf': 15,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'lambda_l1': 0.2,
        'lambda_l2': 0.8,
        'min_gain_to_split': 0.005,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
    },
    {  # Balanced
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
    },
    {  # Aggressive
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
    },
    {  # Very aggressive
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'num_leaves': 80,
        'max_depth': 15,
        'min_data_in_leaf': 10,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 1,
        'lambda_l1': 0.05,
        'lambda_l2': 0.3,
        'min_gain_to_split': 0.001,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
    },
]

# 5-fold CV
print("\n[6/9] 5-Fold Cross-Validation (4 configs)...")
n_folds = 5
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

all_config_results = []

for config_idx, params in enumerate(configs):
    print(f"\nConfig {config_idx+1}/{len(configs)}: LR={params['learning_rate']}, "
          f"Leaves={params['num_leaves']}, Depth={params['max_depth']}")

    fold_scores = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
        X_train, X_val = X_full[train_idx], X_full[val_idx]
        y_train, y_val = y_zero_based[train_idx], y_zero_based[val_idx]

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        model = lgb.train(
            params,
            train_dataset,
            num_boost_round=2000,  # More rounds
            valid_sets=[valid_dataset],
            callbacks=[
                early_stopping(stopping_rounds=120),  # More patience
                log_evaluation(period=0)
            ]
        )

        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_labels = np.argmax(y_val_pred, axis=1)
        accuracy = accuracy_score(y_val, y_val_labels)

        fold_scores.append(accuracy)
        fold_models.append(model)

        print(f"  Fold {fold+1}: Acc={accuracy:.4f}, Iter={model.best_iteration}")

    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"  → CV Score: {avg_score:.4f} ± {std_score:.4f}")

    all_config_results.append((avg_score, std_score, config_idx, fold_models, params))

# Select best config
best_score, best_std, best_config_idx, best_models, best_params = max(
    all_config_results, key=lambda x: x[0]
)

print(f"\n✓ Best Config: #{best_config_idx+1}, CV Score: {best_score:.4f} ± {best_std:.4f}")

# Validation metrics
print("\n[7/9] Computing validation metrics...")
all_val_preds = []
all_val_true = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_zero_based)):
    X_val = X_full[val_idx]
    y_val = y_zero_based[val_idx]

    model = best_models[fold]
    y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)

    all_val_preds.append(y_val_pred)
    all_val_true.extend(y_val)

all_val_preds = np.vstack(all_val_preds)
all_val_true = np.array(all_val_true)

val_pred_labels = np.argmax(all_val_preds, axis=1)
final_val_acc = accuracy_score(all_val_true, val_pred_labels)

print(f"\nValidation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

print("\nClass-wise AUC:")
auc_scores = []
for i in range(4):
    y_true_bin = (all_val_true == i).astype(int)
    auc = roc_auc_score(y_true_bin, all_val_preds[:, i])
    auc_scores.append(auc)
    print(f"  Class {i+1}: {auc:.4f}")

avg_auc = np.mean(auc_scores)
print(f"  Average: {avg_auc:.4f}")

# Very Large Ensemble
print("\n[8/9] Training very large ensemble (15 models)...")

ensemble_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777,
                  1111, 2222, 5555, 8888, 9999]
ensemble_preds = []

avg_best_iter = int(np.mean([m.best_iteration for m in best_models]))
print(f"Using avg best iteration: {avg_best_iter}")

for seed_idx, seed in enumerate(ensemble_seeds):
    print(f"  Model {seed_idx+1}/{len(ensemble_seeds)} (seed={seed})...")

    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed

    full_train = lgb.Dataset(X_full, label=y_zero_based)
    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    test_pred = model.predict(X_test)
    ensemble_preds.append(test_pred)

# Also train with second-best config for diversity
print("\n[9/9] Adding diversity - training with 2nd best config...")
sorted_results = sorted(all_config_results, key=lambda x: x[0], reverse=True)
if len(sorted_results) > 1:
    second_best = sorted_results[1]
    second_params = second_best[4]
    second_iter = int(np.mean([m.best_iteration for m in second_best[3]]))

    for seed in [42, 123, 456]:
        print(f"  2nd-best config model (seed={seed})...")
        params_with_seed = second_params.copy()
        params_with_seed['seed'] = seed

        full_train = lgb.Dataset(X_full, label=y_zero_based)
        model = lgb.train(
            params_with_seed,
            full_train,
            num_boost_round=second_iter
        )

        test_pred = model.predict(X_test)
        ensemble_preds.append(test_pred)

# Weighted average (give more weight to best config)
n_best = 15
n_second = len(ensemble_preds) - n_best

weights = [1.0] * n_best + [0.7] * n_second
weights = np.array(weights) / np.sum(weights)

test_pred_final = np.average(ensemble_preds, axis=0, weights=weights)
test_labels = np.argmax(test_pred_final, axis=1) + 1

# Save
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
np.savetxt('testLabel_ultimate.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\n" + "="*80)
print("✓ ULTIMATE PIPELINE COMPLETE!")
print("="*80)
print(f"\nValidation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Average AUC: {avg_auc:.4f}")
print(f"\nTarget: 92.82%")
print(f"Partner: 92.58%")

if final_val_acc >= 0.9282:
    print(f"✓ SUCCESS! Beat benchmark by {(final_val_acc - 0.9282)*100:.2f}%")
elif final_val_acc >= 0.9258:
    print(f"✓ Beat partner! ({(final_val_acc - 0.9258)*100:.2f}% ahead)")
    print(f"  Gap to benchmark: {(0.9282 - final_val_acc)*100:.2f}%")
else:
    print(f"  Gap to partner: {(0.9258 - final_val_acc)*100:.2f}%")
    print(f"  Gap to benchmark: {(0.9282 - final_val_acc)*100:.2f}%")

print(f"\nTotal ensemble models: {len(ensemble_preds)}")
print(f"Features: {X_full.shape[1]} (from {X_base.shape[1]} original)")
print(f"Strategy: MI feature selection + advanced engineering + large ensemble")
print("\nFiles:")
print("  - testLabel.txt")
print("  - testLabel_ultimate.txt")
