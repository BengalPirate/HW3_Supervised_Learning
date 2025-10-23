"""
Quick test script - faster version for initial testing
Uses single train/val split instead of full cross-validation
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

np.random.seed(42)

print("Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

# Preprocess
train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

imputer = SimpleImputer(strategy='median')
scaler = RobustScaler()

X_full = imputer.fit_transform(train_data.values)
X_full = scaler.fit_transform(X_full)
y_full = train_labels['label'].values

X_test = imputer.transform(test_data.values)
X_test = scaler.transform(X_test)

# Single train/val split
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

# Quick Random Forest test
print("\n" + "="*50)
print("Testing Random Forest...")
print("="*50)

models = {}
for cls in [1, 2, 3, 4]:
    print(f"Training class {cls} vs rest...")
    y_binary = (y_train == cls).astype(int)

    if HAS_XGB:
        clf = xgb.XGBClassifier(n_estimators=100, max_depth=8, learning_rate=0.1, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)

    clf.fit(X_train, y_binary)
    models[cls] = clf

# Validate
val_probas = np.zeros((len(X_val), 4))
for i, cls in enumerate([1, 2, 3, 4]):
    val_probas[:, i] = models[cls].predict_proba(X_val)[:, 1]

val_preds = np.argmax(val_probas, axis=1) + 1

# Metrics
acc = accuracy_score(y_val, val_preds)
print(f"\nValidation Accuracy: {acc:.4f}")

for i, cls in enumerate([1, 2, 3, 4]):
    y_binary = (y_val == cls).astype(int)
    auc = roc_auc_score(y_binary, val_probas[:, i])
    print(f"Class {cls} AUC: {auc:.4f}")

avg_auc = np.mean([roc_auc_score((y_val == cls).astype(int), val_probas[:, i]) for i, cls in enumerate([1, 2, 3, 4])])
print(f"Average AUC: {avg_auc:.4f}")

# Train on full data and predict test
print("\n" + "="*50)
print("Training on full data for test predictions...")
print("="*50)

final_models = {}
for cls in [1, 2, 3, 4]:
    print(f"Training class {cls} vs rest...")
    y_binary = (y_full == cls).astype(int)

    if HAS_XGB:
        clf = xgb.XGBClassifier(n_estimators=150, max_depth=8, learning_rate=0.05, random_state=42, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42, n_jobs=-1)

    clf.fit(X_full, y_binary)
    final_models[cls] = clf

# Predict test
test_probas = np.zeros((len(X_test), 4))
for i, cls in enumerate([1, 2, 3, 4]):
    test_probas[:, i] = final_models[cls].predict_proba(X_test)[:, 1]

test_preds = np.argmax(test_probas, axis=1) + 1

# Save
output = np.column_stack([test_probas, test_preds])
np.savetxt('testLabel_quick.txt', output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')

print("\nPredictions saved to: testLabel_quick.txt")
print("Quick test complete!")
