"""
CSC 621 - HW3 Classification Project
Advanced Ensemble Methods

This script implements advanced ensemble techniques:
- Voting Classifier (combines multiple models)
- Stacking Classifier
- Weighted averaging of predictions

Run this after classification_pipeline.py for potentially better results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

np.random.seed(42)


def load_and_preprocess_data():
    """Load and preprocess data"""
    print("Loading data...")
    train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
    train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
    test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

    # Replace empty strings with NaN and convert to numeric
    train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
    test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

    # Impute and scale
    imputer = SimpleImputer(strategy='median')
    scaler = RobustScaler()

    X_train = imputer.fit_transform(train_data.values)
    X_train = scaler.fit_transform(X_train)

    X_test = imputer.transform(test_data.values)
    X_test = scaler.transform(X_test)

    y_train = train_labels['label'].values

    return X_train, y_train, X_test, imputer, scaler


class EnsembleClassifier:
    """Advanced ensemble classifier using multiple strategies"""

    def __init__(self, strategy='voting'):
        """
        strategy: 'voting', 'stacking', or 'weighted'
        """
        self.strategy = strategy
        self.models = {}
        self.classes = [1, 2, 3, 4]

        # Define base estimators
        self.base_estimators = self._get_base_estimators()

    def _get_base_estimators(self):
        """Get list of base estimators"""
        estimators = []

        # Random Forest
        estimators.append(('rf', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )))

        # Gradient Boosting
        estimators.append(('gb', GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=10,
            random_state=42
        )))

        # XGBoost
        if HAS_XGB:
            estimators.append(('xgb', xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            )))

        # LightGBM
        if HAS_LGB:
            estimators.append(('lgb', lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=10,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))

        return estimators

    def fit(self, X_train, y_train):
        """Train ensemble for each class (One-vs-Rest)"""
        print(f"\nTraining {self.strategy} ensemble...")

        for cls in self.classes:
            print(f"  Class {cls} vs rest...")
            y_binary = (y_train == cls).astype(int)

            if self.strategy == 'voting':
                # Soft voting (average probabilities)
                model = VotingClassifier(
                    estimators=self.base_estimators,
                    voting='soft',
                    n_jobs=-1
                )

            elif self.strategy == 'stacking':
                # Stacking with logistic regression as meta-learner
                model = StackingClassifier(
                    estimators=self.base_estimators,
                    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
                    cv=5,
                    n_jobs=-1
                )

            else:  # weighted
                # Train individual models and use weighted average
                model = VotingClassifier(
                    estimators=self.base_estimators,
                    voting='soft',
                    weights=[2, 1, 3, 3] if len(self.base_estimators) == 4 else None,  # Higher weight for XGB and LGB
                    n_jobs=-1
                )

            model.fit(X_train, y_binary)
            self.models[cls] = model

        return self

    def predict_proba(self, X_test):
        """Predict probabilities for each class"""
        probas = np.zeros((len(X_test), len(self.classes)))

        for i, cls in enumerate(self.classes):
            probas[:, i] = self.models[cls].predict_proba(X_test)[:, 1]

        return probas

    def predict(self, X_test):
        """Predict class labels"""
        probas = self.predict_proba(X_test)
        return np.argmax(probas, axis=1) + 1


def evaluate_ensemble(ensemble, X_train, y_train, cv=5):
    """Evaluate ensemble using cross-validation"""
    print(f"\nEvaluating {ensemble.strategy} ensemble...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = {1: [], 2: [], 3: [], 4: []}
    accuracy_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold + 1}/{cv}...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        ensemble.fit(X_tr, y_tr)
        y_proba = ensemble.predict_proba(X_val)
        y_pred = ensemble.predict(X_val)

        # Calculate AUC for each class
        for i, cls in enumerate([1, 2, 3, 4]):
            y_binary = (y_val == cls).astype(int)
            auc = roc_auc_score(y_binary, y_proba[:, i])
            auc_scores[cls].append(auc)

        # Accuracy
        acc = accuracy_score(y_val, y_pred)
        accuracy_scores.append(acc)

    # Print results
    print(f"\n{ensemble.strategy.upper()} Ensemble Results:")
    print(f"  Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    for cls in [1, 2, 3, 4]:
        mean_auc = np.mean(auc_scores[cls])
        std_auc = np.std(auc_scores[cls])
        print(f"  Class {cls} AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    avg_auc = np.mean([np.mean(auc_scores[cls]) for cls in [1, 2, 3, 4]])
    print(f"  Average AUC: {avg_auc:.4f}")

    return avg_auc


def save_predictions(predictions_proba, predictions_class, filename):
    """Save predictions in required format"""
    output = np.column_stack([predictions_proba, predictions_class])
    np.savetxt(filename, output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
    print(f"Predictions saved to: {filename}")


def main():
    """Main execution"""
    print("="*70)
    print("Advanced Ensemble Methods")
    print("="*70)

    # Load data
    X_train, y_train, X_test, imputer, scaler = load_and_preprocess_data()

    # Test different ensemble strategies
    strategies = ['voting', 'weighted', 'stacking']
    results = {}

    for strategy in strategies:
        ensemble = EnsembleClassifier(strategy=strategy)
        avg_auc = evaluate_ensemble(ensemble, X_train, y_train, cv=3)  # Use 3 folds for speed
        results[strategy] = avg_auc

    # Find best strategy
    best_strategy = max(results.keys(), key=lambda k: results[k])
    print(f"\n{'='*70}")
    print(f"Best Ensemble Strategy: {best_strategy.upper()}")
    print(f"Average AUC: {results[best_strategy]:.4f}")
    print("="*70)

    # Train best ensemble on full data
    print("\nTraining best ensemble on full training set...")
    best_ensemble = EnsembleClassifier(strategy=best_strategy)
    best_ensemble.fit(X_train, y_train)

    # Generate predictions
    print("\nGenerating predictions...")
    test_proba = best_ensemble.predict_proba(X_test)
    test_pred = best_ensemble.predict(X_test)

    # Save
    save_predictions(test_proba, test_pred, 'testLabel_ensemble.txt')

    print("\n" + "="*70)
    print("Advanced ensemble complete!")
    print("Output: testLabel_ensemble.txt")
    print("="*70)


if __name__ == "__main__":
    main()
