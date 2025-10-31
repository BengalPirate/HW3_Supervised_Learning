"""
Pseudo-Labeling Pipeline - Semi-Supervised Learning
Uses high-confidence test predictions to expand training data iteratively.
Expected gain: 0.5-1.0%
Time: ~30 minutes
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
print("PSEUDO-LABELING PIPELINE - SEMI-SUPERVISED LEARNING")
print("="*80)

# Load data
print("\n[1/8] Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"Training: {train_data.shape}, Test: {test_data.shape}")

# Preprocess
print("\n[2/8] Preprocessing...")
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

y = train_labels['label'].values

# Median imputation
imputer = SimpleImputer(strategy='median')
X_base = imputer.fit_transform(train_data.values)
X_test_base = imputer.transform(test_data.values)

# Feature Selection
print("\n[3/8] Feature selection using mutual information...")
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)
selected_features = np.where(mi_scores > mi_threshold)[0]

print(f"Selected {len(selected_features)} features from {X_base.shape[1]}")

X_base_selected = X_base[:, selected_features]
X_test_base_selected = X_test_base[:, selected_features]

# Feature Engineering
print("\n[4/8] Advanced feature engineering...")
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

X_train_full = create_advanced_features(X_base_selected, top_mi_features)
X_test_full = create_advanced_features(X_test_base_selected, top_mi_features)

print(f"After feature engineering: {X_train_full.shape}")

# Scaling
scaler = RobustScaler()
X_train_full = scaler.fit_transform(X_train_full)
X_test_full = scaler.transform(X_test_full)

y_zero_based = y - 1

# Check class imbalance
class_counts = pd.Series(y).value_counts()
is_imbalanced = (class_counts.max() / class_counts.min()) > 1.5

# Best hyperparameters
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

# Pseudo-labeling configuration
confidence_threshold = 0.95  # High confidence threshold
n_iterations = 3
n_folds = 5

print(f"\n[5/8] Pseudo-labeling with {n_iterations} iterations...")
print(f"Confidence threshold: {confidence_threshold}")

# Keep track of expanded training set
X_train_expanded = X_train_full.copy()
y_train_expanded = y_zero_based.copy()

iteration_results = []

for iteration in range(n_iterations):
    print(f"\n{'='*60}")
    print(f"Iteration {iteration+1}/{n_iterations}")
    print(f"{'='*60}")
    print(f"Current training size: {X_train_expanded.shape[0]} samples")

    # Train model on current expanded training set using CV
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42+iteration)

    cv_scores = []
    fold_models = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_expanded, y_train_expanded)):
        X_train, X_val = X_train_expanded[train_idx], X_train_expanded[val_idx]
        y_train, y_val = y_train_expanded[train_idx], y_train_expanded[val_idx]

        train_dataset = lgb.Dataset(X_train, label=y_train)
        valid_dataset = lgb.Dataset(X_val, label=y_val, reference=train_dataset)

        model = lgb.train(
            best_params,
            train_dataset,
            num_boost_round=2000,
            valid_sets=[valid_dataset],
            callbacks=[
                early_stopping(stopping_rounds=120),
                log_evaluation(period=0)
            ]
        )

        y_val_pred = model.predict(X_val, num_iteration=model.best_iteration)
        y_val_labels = np.argmax(y_val_pred, axis=1)
        accuracy = accuracy_score(y_val, y_val_labels)

        cv_scores.append(accuracy)
        fold_models.append(model)

        print(f"  Fold {fold+1}: Acc={accuracy:.4f}, Iter={model.best_iteration}")

    avg_cv_score = np.mean(cv_scores)
    std_cv_score = np.std(cv_scores)
    print(f"\nIteration {iteration+1} CV Score: {avg_cv_score:.4f} ± {std_cv_score:.4f}")

    iteration_results.append({
        'iteration': iteration + 1,
        'train_size': X_train_expanded.shape[0],
        'cv_score': avg_cv_score,
        'cv_std': std_cv_score
    })

    # Predict on test set with ensemble
    test_preds = []
    for model in fold_models:
        pred = model.predict(X_test_full, num_iteration=model.best_iteration)
        test_preds.append(pred)

    # Average predictions
    test_pred_avg = np.mean(test_preds, axis=0)

    # Get max probability for each sample
    max_probs = np.max(test_pred_avg, axis=1)
    pred_labels = np.argmax(test_pred_avg, axis=1)

    # Select high-confidence predictions
    high_conf_mask = max_probs > confidence_threshold
    n_high_conf = np.sum(high_conf_mask)

    print(f"\nHigh-confidence predictions: {n_high_conf} / {len(test_pred_avg)} ({n_high_conf/len(test_pred_avg)*100:.1f}%)")

    if n_high_conf == 0:
        print("No high-confidence predictions found. Stopping iteration.")
        break

    # Distribution of high-confidence predictions
    conf_class_dist = pd.Series(pred_labels[high_conf_mask] + 1).value_counts().sort_index()
    print("High-confidence label distribution:")
    for cls, count in conf_class_dist.items():
        print(f"  Class {cls}: {count} ({count/n_high_conf*100:.1f}%)")

    # Add pseudo-labeled samples to training set
    if iteration < n_iterations - 1:  # Don't expand on last iteration
        X_train_expanded = np.vstack([X_train_expanded, X_test_full[high_conf_mask]])
        y_train_expanded = np.concatenate([y_train_expanded, pred_labels[high_conf_mask]])
        print(f"\nAdded {n_high_conf} pseudo-labeled samples")
        print(f"New training size: {X_train_expanded.shape[0]} samples")

print("\n[6/8] Pseudo-labeling iteration summary:")
print(f"{'Iter':<6} {'Train Size':<12} {'CV Score':<12} {'CV Std':<10}")
print("-" * 42)
for result in iteration_results:
    print(f"{result['iteration']:<6} {result['train_size']:<12} {result['cv_score']:<12.4f} {result['cv_std']:<10.4f}")

# Select best iteration based on CV score
best_iteration_idx = np.argmax([r['cv_score'] for r in iteration_results])
best_iteration_result = iteration_results[best_iteration_idx]

print(f"\nBest iteration: {best_iteration_result['iteration']}")
print(f"Best CV score: {best_iteration_result['cv_score']:.4f} ± {best_iteration_result['cv_std']:.4f}")

# Train final ensemble on best expanded training set
print(f"\n[7/8] Training final ensemble with pseudo-labeled data...")

# Use the fully expanded training set (from all iterations)
print(f"Final training size: {X_train_expanded.shape[0]} samples")
print(f"Original training size: {X_train_full.shape[0]} samples")
print(f"Added samples: {X_train_expanded.shape[0] - X_train_full.shape[0]}")

# Train final ensemble (15 models with different seeds)
ensemble_seeds = [42, 123, 456, 789, 2024, 2025, 3141, 9876, 1337, 7777,
                  1111, 2222, 5555, 8888, 9999]

ensemble_preds = []
avg_best_iter = int(np.mean([m.best_iteration for model_list in [fold_models] for m in model_list]))

print(f"Training {len(ensemble_seeds)} models (avg iter: {avg_best_iter})")

full_train = lgb.Dataset(X_train_expanded, label=y_train_expanded)

for seed_idx, seed in enumerate(ensemble_seeds):
    params_with_seed = best_params.copy()
    params_with_seed['seed'] = seed

    model = lgb.train(
        params_with_seed,
        full_train,
        num_boost_round=avg_best_iter
    )

    test_pred = model.predict(X_test_full)
    ensemble_preds.append(test_pred)
    print(f"  Model {seed_idx+1}/{len(ensemble_seeds)} complete")

# Average ensemble predictions
test_pred_final = np.mean(ensemble_preds, axis=0)
test_labels = np.argmax(test_pred_final, axis=1) + 1

print(f"\n[8/8] Saving predictions...")

# Save
output = np.column_stack([test_pred_final, test_labels])
np.savetxt('testLabel_pseudo_labeled.txt', output, delimiter='\\t',
           fmt='%.6f\\t%.6f\\t%.6f\\t%.6f\\t%d')

print("\\n" + "="*80)
print("✓ PSEUDO-LABELING PIPELINE COMPLETE!")
print("="*80)

print(f"\\nBest CV Score: {best_iteration_result['cv_score']:.4f} ({best_iteration_result['cv_score']*100:.2f}%)")
print(f"Baseline (no pseudo-labeling): ~91.58%")

if best_iteration_result['cv_score'] > 0.9158:
    improvement = (best_iteration_result['cv_score'] - 0.9158) * 100
    print(f"✓ Improvement: +{improvement:.2f}%")
else:
    gap = (0.9158 - best_iteration_result['cv_score']) * 100
    print(f"⚠ Performance decreased: -{gap:.2f}%")

print(f"\\nPseudo-labeled samples added: {X_train_expanded.shape[0] - X_train_full.shape[0]}")
print(f"Final training set size: {X_train_expanded.shape[0]} samples")
print(f"\\nFile saved: testLabel_pseudo_labeled.txt")

print(f"\\nTarget: 92.82%")
print(f"Partner: 92.58%")
if best_iteration_result['cv_score'] >= 0.9282:
    print(f"✓✓ EXCEEDED TARGET!")
elif best_iteration_result['cv_score'] >= 0.9258:
    print(f"✓ EXCEEDED PARTNER!")
else:
    gap_to_partner = (0.9258 - best_iteration_result['cv_score']) * 100
    gap_to_target = (0.9282 - best_iteration_result['cv_score']) * 100
    print(f"Gap to partner: {gap_to_partner:.2f}%")
    print(f"Gap to target: {gap_to_target:.2f}%")
