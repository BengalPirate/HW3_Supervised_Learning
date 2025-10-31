"""
Hyperparameter Tuned Pipeline - Without SMOTE
Focus on better hyperparameters and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("Hyperparameter Tuned Pipeline")
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

# Feature Engineering
print("\nFeature Engineering...")
top_features = [85, 357, 336, 86, 356, 337, 84, 225, 124, 236, 144, 354, 143, 205, 323]

def create_features(X_data, top_feat):
    """Create enhanced features"""
    features = [X_data]

    # Squared features
    for feat in top_feat[:15]:
        features.append((X_data[:, feat] ** 2).reshape(-1, 1))

    # Pairwise interactions
    for i in range(min(7, len(top_feat))):
        for j in range(i+1, min(7, len(top_feat))):
            interaction = (X_data[:, top_feat[i]] * X_data[:, top_feat[j]]).reshape(-1, 1)
            features.append(interaction)

    # Statistical features
    top_data = X_data[:, top_feat[:15]]
    features.append(np.mean(top_data, axis=1).reshape(-1, 1))
    features.append(np.std(top_data, axis=1).reshape(-1, 1))
    features.append(np.max(top_data, axis=1).reshape(-1, 1))
    features.append(np.min(top_data, axis=1).reshape(-1, 1))
    features.append(np.median(top_data, axis=1).reshape(-1, 1))

    return np.hstack(features)

X_full = create_features(X_base, top_features)
X_test_processed = create_features(X_test_base, top_features)

print(f"After feature engineering: {X_full.shape}")

# Train/validation split
print("\nCreating train/validation split (80/20)...")
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

# Try multiple model configurations
print("\n" + "="*70)
print("Testing Multiple Configurations")
print("="*70)

configs = [
    {
        'name': 'XGBoost-Config1',
        'model': 'xgb',
        'params': {
            'n_estimators': 800,
            'learning_rate': 0.02,
            'max_depth': 10,
            'min_child_weight': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'gamma': 0.1,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
        }
    },
    {
        'name': 'XGBoost-Config2',
        'model': 'xgb',
        'params': {
            'n_estimators': 1000,
            'learning_rate': 0.015,
            'max_depth': 11,
            'min_child_weight': 1,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'gamma': 0.05,
            'reg_alpha': 0.01,
            'reg_lambda': 0.8,
        }
    },
    {
        'name': 'LightGBM-Config1',
        'model': 'lgb',
        'params': {
            'n_estimators': 800,
            'learning_rate': 0.02,
            'max_depth': 15,
            'num_leaves': 80,
            'min_child_samples': 10,
            'subsample': 0.9,
            'colsample_bytree': 0.9,
            'reg_alpha': 0.05,
            'reg_lambda': 1.0,
        }
    },
    {
        'name': 'LightGBM-Config2',
        'model': 'lgb',
        'params': {
            'n_estimators': 1000,
            'learning_rate': 0.015,
            'max_depth': 18,
            'num_leaves': 100,
            'min_child_samples': 8,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.01,
            'reg_lambda': 0.8,
        }
    },
]

best_accuracy = 0
best_config_name = None
best_models = None
best_val_probas = None

for config in configs:
    print(f"\n{'='*70}")
    print(f"Testing: {config['name']}")
    print(f"{'='*70}")

    models = {}
    val_probas = np.zeros((len(X_val), 4))

    for cls in [1, 2, 3, 4]:
        y_binary_train = (y_train == cls).astype(int)
        y_binary_val = (y_val == cls).astype(int)

        # Calculate scale_pos_weight
        neg_count = np.sum(y_binary_train == 0)
        pos_count = np.sum(y_binary_train == 1)
        scale_pos_weight = neg_count / pos_count

        if config['model'] == 'xgb':
            clf = xgb.XGBClassifier(
                **config['params'],
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                early_stopping_rounds=50,
                verbose=False
            )
            clf.fit(
                X_train, y_binary_train,
                eval_set=[(X_val, y_binary_val)],
                verbose=False
            )
        else:  # lgb
            clf = lgb.LGBMClassifier(
                **config['params'],
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            clf.fit(
                X_train, y_binary_train,
                eval_set=[(X_val, y_binary_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )

        models[cls] = clf
        val_probas[:, cls-1] = clf.predict_proba(X_val)[:, 1]

    val_preds = np.argmax(val_probas, axis=1) + 1
    acc = accuracy_score(y_val, val_preds)

    print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    for cls in [1, 2, 3, 4]:
        y_binary = (y_val == cls).astype(int)
        auc = roc_auc_score(y_binary, val_probas[:, cls-1])
        print(f"  Class {cls} AUC: {auc:.4f}")

    avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), val_probas[:, cls-1])
                        for cls in [1, 2, 3, 4]])
    print(f"  Average AUC: {avg_auc:.4f}")

    if acc > best_accuracy:
        best_accuracy = acc
        best_config_name = config['name']
        best_config = config
        best_models = models
        best_val_probas = val_probas

print("\n" + "="*70)
print(f"BEST CONFIGURATION: {best_config_name}")
print(f"Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("="*70)

# Plot ROC curves for best model
print("\nGenerating ROC curves...")
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.ravel()

for cls_idx, cls in enumerate([1, 2, 3, 4]):
    y_binary = (y_val == cls).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, best_val_probas[:, cls_idx])
    auc_score = roc_auc_score(y_binary, best_val_probas[:, cls_idx])

    ax = axes[cls_idx]
    ax.plot(fpr, tpr, linewidth=2, label=f'{best_config_name} (AUC = {auc_score:.3f})', color='blue')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - Class {cls} vs Rest', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_hypertuned.png', dpi=300, bbox_inches='tight')
print("Saved: roc_curves_hypertuned.png")
plt.close()

# Feature importance
print("\nExtracting feature importance...")
avg_importance = np.mean([best_models[cls].feature_importances_ for cls in [1, 2, 3, 4]], axis=0)
top_features_idx = np.argsort(avg_importance)[::-1][:20]

print("\nTop 20 Most Important Features:")
for i, feat_idx in enumerate(top_features_idx):
    print(f"  {i+1}. Feature {feat_idx}: {avg_importance[feat_idx]:.6f}")

# Retrain on full training data
print("\n" + "="*70)
print("Retraining on Full Training Data")
print("="*70)

final_models = {}
for cls in [1, 2, 3, 4]:
    print(f"\nClass {cls} vs Rest...")
    y_binary = (y_full == cls).astype(int)

    neg_count = np.sum(y_binary == 0)
    pos_count = np.sum(y_binary == 1)
    scale_pos_weight = neg_count / pos_count

    if best_config['model'] == 'xgb':
        clf = xgb.XGBClassifier(
            **best_config['params'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss',
            verbose=False
        )
    else:
        clf = lgb.LGBMClassifier(
            **best_config['params'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )

    clf.fit(X_full, y_binary)
    final_models[cls] = clf

# Generate test predictions
print("\nGenerating test predictions...")
test_probas = np.zeros((len(X_test_processed), 4))
for cls in [1, 2, 3, 4]:
    test_probas[:, cls-1] = final_models[cls].predict_proba(X_test_processed)[:, 1]

test_preds = np.argmax(test_probas, axis=1) + 1

# Save predictions
output = np.column_stack([test_probas, test_preds])
np.savetxt('testLabel_hypertuned.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
print("Saved: testLabel_hypertuned.txt")

# Results summary
avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), best_val_probas[:, cls-1])
                    for cls in [1, 2, 3, 4]])

print("\n" + "="*70)
print("HYPERTUNED PIPELINE COMPLETE!")
print("="*70)
print(f"\nBest Configuration: {best_config_name}")
print(f"Validation Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"Target: 0.9282 (92.82%)")
if best_accuracy >= 0.9282:
    print(f"SUCCESS! Beat the benchmark by {(best_accuracy - 0.9282)*100:.2f}%")
else:
    print(f"Gap: {(0.9282 - best_accuracy)*100:.2f}% below target")
print(f"Average AUC: {avg_auc:.4f}")
print("\nGenerated Files:")
print("  - testLabel_hypertuned.txt: Test predictions")
print("  - roc_curves_hypertuned.png: ROC curves")
