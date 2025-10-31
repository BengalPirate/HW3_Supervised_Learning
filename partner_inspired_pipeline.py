"""
Partner-Inspired Pipeline - Combining Best Strategies
Based on partner's 92.58% approach + feature engineering
Key strategies:
- 5-fold CV for better generalization
- Multiple configs tested
- Large ensemble (8+ models)
- Noise-robust hyperparameters
- Feature engineering for extra boost
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm.callback import early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*80)
print("PARTNER-INSPIRED OPTIMIZED PIPELINE")
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

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5
print(f"Class imbalance detected: {is_imbalanced}")

# Median imputation (robust to noise)
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Engineering - add engineered features
print("\n[3/7] Feature engineering...")
top_features = [85, 357, 336, 86, 356, 337, 84, 225, 124, 236]

def add_features(X_data):
    features = [X_data]

    # Squared features (top 10)
    for feat in top_features:
        features.append((X_data[:, feat] ** 2).reshape(-1, 1))

    # Key interactions (top 5)
    for i in range(5):
        for j in range(i+1, 5):
            interaction = (X_data[:, top_features[i]] * X_data[:, top_features[j]]).reshape(-1, 1)
            features.append(interaction)

    # Statistical features
    top_data = X_data[:, top_features]
    features.append(np.mean(top_data, axis=1).reshape(-1, 1))
    features.append(np.std(top_data, axis=1).reshape(-1, 1))
    features.append(np.max(top_data, axis=1).reshape(-1, 1))
    features.append(np.min(top_data, axis=1).reshape(-1, 1))

    return np.hstack(features)

X_full = add_features(X_base)
X_test = add_features(X_test_base)

print(f"After feature engineering: {X_full.shape}")

# Convert labels to 0-based for LightGBM multiclass
y_zero_based = y - 1

# Noise-robust hyperparameter configs (inspired by partner)
print("\n[4/7] Configuring noise-robust hyperparameters...")

configs = [
    {  # Conservative - high regularization
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 7,
        'min_data_in_leaf': 30,
        'feature_fraction': 0.75,
        'bagging_fraction': 0.75,
        'bagging_freq': 5,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
    },
    {  # Moderate
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.025,
        'num_leaves': 40,
        'max_depth': 9,
        'min_data_in_leaf': 20,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 3,
        'lambda_l1': 0.3,
        'lambda_l2': 0.7,
        'min_gain_to_split': 0.01,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
    },
    {  # Aggressive
        'objective': 'multiclass',
        'num_class': 4,
        'metric': 'multi_logloss',
        'learning_rate': 0.03,
        'num_leaves': 50,
        'max_depth': 10,
        'min_data_in_leaf': 15,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.9,
        'bagging_freq': 2,
        'lambda_l1': 0.1,
        'lambda_l2': 0.5,
        'min_gain_to_split': 0.005,
        'verbose': -1,
        'is_unbalance': is_imbalanced,
    },
]

# 5-fold Cross-Validation
print("\n[5/7] 5-Fold Cross-Validation...")
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
            num_boost_round=1500,
            valid_sets=[valid_dataset],
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(period=0)
            ]
        )

        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_labels = np.argmax(y_val_pred, axis=1)
        accuracy = accuracy_score(y_val, y_val_labels)

        fold_scores.append(accuracy)
        fold_models.append(model)

        print(f"  Fold {fold+1}: Acc={accuracy:.4f}, Best iter={model.best_iteration}")

    avg_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)
    print(f"  → CV Score: {avg_score:.4f} ± {std_score:.4f}")

    all_config_results.append((avg_score, std_score, config_idx, fold_models, params))

# Select best config
best_score, best_std, best_config_idx, best_models, best_params = max(
    all_config_results, key=lambda x: x[0]
)

print(f"\n✓ Best Config: #{best_config_idx+1}, CV Score: {best_score:.4f} ± {best_std:.4f}")

# Detailed validation metrics
print("\n[6/7] Validation metrics...")
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

print(f"\nFinal Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")

print("\nClass-wise AUC scores:")
for i in range(4):
    y_true_bin = (all_val_true == i).astype(int)
    auc = roc_auc_score(y_true_bin, all_val_preds[:, i])
    print(f"  Class {i+1} AUC: {auc:.4f}")

# Large Ensemble for Test Predictions
print("\n[7/7] Generating large ensemble predictions...")

# Use 10 different seeds for ensemble
ensemble_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777]
ensemble_preds = []

avg_best_iter = int(np.mean([m.best_iteration for m in best_models]))
print(f"Using avg best iteration: {avg_best_iter}")

for seed_idx, seed in enumerate(ensemble_seeds):
    print(f"  Ensemble model {seed_idx+1}/{len(ensemble_seeds)} (seed={seed})...")

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

# Average ensemble predictions
test_pred_final = np.mean(ensemble_preds, axis=0)
test_labels = np.argmax(test_pred_final, axis=1) + 1  # Convert back to 1-based

# Save in required format
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
np.savetxt('testLabel_partner_inspired.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\n" + "="*80)
print("✓ COMPLETED!")
print("="*80)
print(f"\nValidation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"Target: 0.9282 (92.82%)")
if final_val_acc >= 0.9282:
    print(f"SUCCESS! Beat benchmark by {(final_val_acc - 0.9282)*100:.2f}%")
else:
    print(f"Gap: {(0.9282 - final_val_acc)*100:.2f}% below target")

print(f"\nPartner's accuracy: 92.58%")
if final_val_acc >= 0.9258:
    print(f"SUCCESS! Beat partner by {(final_val_acc - 0.9258)*100:.2f}%")
else:
    print(f"Gap: {(0.9258 - final_val_acc)*100:.2f}% below partner")

print(f"\nEnsemble: {len(ensemble_seeds)} models")
print(f"Strategy: Noise-robust LightGBM + Feature Engineering + 5-Fold CV")
print("\nFiles saved:")
print("  - testLabel.txt (required filename)")
print("  - testLabel_partner_inspired.txt (backup)")
