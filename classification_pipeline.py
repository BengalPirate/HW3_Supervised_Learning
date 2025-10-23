"""
CSC 621 - HW3 Classification Project
Main Classification Pipeline

This script implements multiple classification algorithms for the 4-class problem:
- Random Forest
- XGBoost
- LightGBM
- Neural Network (MLP)
- Logistic Regression with feature engineering

The script handles:
1. Missing value imputation
2. Feature scaling
3. Cross-validation
4. Hyperparameter tuning
5. Feature importance extraction
6. ROC curve generation
7. Predictions in required format
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, classification_report
from sklearn.multiclass import OneVsRestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced libraries
try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

# Set random seed
np.random.seed(42)

class MultiClassClassifier:
    """
    Wrapper for multi-class classification using One-vs-Rest strategy
    """
    def __init__(self, base_classifier, name="Classifier"):
        self.name = name
        self.base_classifier = base_classifier
        self.classifiers = {}
        self.classes = [1, 2, 3, 4]
        self.scaler = None
        self.imputer = None

    def preprocess_data(self, X_train, fit=True):
        """Handle missing values and scale features"""
        if fit:
            # Use median imputation for missing values (only column 410 has missing values)
            self.imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            X_imputed = self.imputer.fit_transform(X_train)

            # Use RobustScaler to handle outliers
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X_train)
            X_scaled = self.scaler.transform(X_imputed)

        return X_scaled

    def fit(self, X_train, y_train):
        """Train one classifier per class (One-vs-Rest)"""
        X_processed = self.preprocess_data(X_train, fit=True)

        for cls in self.classes:
            print(f"  Training classifier for class {cls} vs rest...")
            # Create binary labels
            y_binary = (y_train == cls).astype(int)

            # Clone and fit classifier
            from sklearn.base import clone
            clf = clone(self.base_classifier)
            clf.fit(X_processed, y_binary)
            self.classifiers[cls] = clf

        return self

    def predict_proba(self, X_test):
        """Get probability scores for each class"""
        X_processed = self.preprocess_data(X_test, fit=False)

        probas = np.zeros((len(X_test), len(self.classes)))
        for i, cls in enumerate(self.classes):
            if hasattr(self.classifiers[cls], 'predict_proba'):
                probas[:, i] = self.classifiers[cls].predict_proba(X_processed)[:, 1]
            else:
                probas[:, i] = self.classifiers[cls].decision_function(X_processed)

        return probas

    def predict(self, X_test):
        """Predict class labels"""
        probas = self.predict_proba(X_test)
        return np.argmax(probas, axis=1) + 1  # Add 1 because classes are 1-4

    def get_feature_importance(self):
        """Extract feature importance if available"""
        importance_dict = {}
        for cls in self.classes:
            clf = self.classifiers[cls]
            if hasattr(clf, 'feature_importances_'):
                importance_dict[cls] = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                importance_dict[cls] = np.abs(clf.coef_[0])

        return importance_dict


def load_data():
    """Load training and test data"""
    print("Loading data...")
    train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
    train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
    test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

    # Replace empty strings with NaN
    train_data = train_data.replace('', np.nan)
    test_data = test_data.replace('', np.nan)

    # Convert to numeric
    train_data = train_data.apply(pd.to_numeric, errors='coerce')
    test_data = test_data.apply(pd.to_numeric, errors='coerce')

    X_train = train_data.values
    y_train = train_labels['label'].values
    X_test = test_data.values

    print(f"Training set: {X_train.shape}, Labels: {y_train.shape}")
    print(f"Test set: {X_test.shape}")

    return X_train, y_train, X_test


def evaluate_cv(classifier, X_train, y_train, cv=5):
    """Evaluate classifier using cross-validation"""
    print(f"\nCross-validating {classifier.name}...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = {1: [], 2: [], 3: [], 4: []}
    accuracy_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold + 1}/{cv}...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        # Train
        classifier.fit(X_tr, y_tr)

        # Predict probabilities
        y_proba = classifier.predict_proba(X_val)
        y_pred = classifier.predict(X_val)

        # Calculate AUC for each class
        for i, cls in enumerate([1, 2, 3, 4]):
            y_binary = (y_val == cls).astype(int)
            auc = roc_auc_score(y_binary, y_proba[:, i])
            auc_scores[cls].append(auc)

        # Calculate accuracy
        acc = accuracy_score(y_val, y_pred)
        accuracy_scores.append(acc)

    # Print results
    print(f"\n{classifier.name} Cross-Validation Results:")
    print(f"  Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
    for cls in [1, 2, 3, 4]:
        mean_auc = np.mean(auc_scores[cls])
        std_auc = np.std(auc_scores[cls])
        print(f"  Class {cls} AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

    avg_auc = np.mean([np.mean(auc_scores[cls]) for cls in [1, 2, 3, 4]])
    print(f"  Average AUC: {avg_auc:.4f}")

    return {
        'accuracy': np.mean(accuracy_scores),
        'auc_scores': {cls: np.mean(auc_scores[cls]) for cls in [1, 2, 3, 4]},
        'avg_auc': avg_auc
    }


def plot_roc_curves(classifiers, X_train, y_train, cv=5):
    """Generate ROC curves for all classifiers"""
    print("\nGenerating ROC curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    for cls_idx, cls in enumerate([1, 2, 3, 4]):
        ax = axes[cls_idx]

        # Use stratified k-fold
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for classifier in classifiers:
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                classifier.fit(X_tr, y_tr)
                y_proba = classifier.predict_proba(X_val)

                # Binary labels for this class
                y_binary = (y_val == cls).astype(int)

                # ROC curve
                fpr, tpr, _ = roc_curve(y_binary, y_proba[:, cls_idx])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc_score(y_binary, y_proba[:, cls_idx]))

            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)

            ax.plot(mean_fpr, mean_tpr,
                   label=f'{classifier.name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})',
                   linewidth=2)

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - Class {cls} vs Rest', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved ROC curves to: roc_curves_comparison.png")
    plt.close()


def save_predictions(predictions_proba, predictions_class, filename):
    """Save predictions in required format (5-column tab-delimited)"""
    # predictions_proba shape: (n_samples, 4) for class 1,2,3,4 probabilities
    # predictions_class shape: (n_samples,) for final class labels

    output = np.column_stack([predictions_proba, predictions_class])

    # Save with tab delimiter
    np.savetxt(filename, output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
    print(f"Predictions saved to: {filename}")


def main():
    """Main execution pipeline"""
    print("="*70)
    print("CSC 621 - HW3 Classification Pipeline")
    print("="*70)

    # Load data
    X_train, y_train, X_test = load_data()

    # Define classifiers to test
    classifiers = []

    # 1. Random Forest
    print("\n" + "="*70)
    print("Setting up Random Forest Classifier")
    print("="*70)
    rf_clf = MultiClassClassifier(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ),
        name="Random Forest"
    )
    classifiers.append(rf_clf)

    # 2. Gradient Boosting
    print("\n" + "="*70)
    print("Setting up Gradient Boosting Classifier")
    print("="*70)
    gb_clf = MultiClassClassifier(
        GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=7,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        ),
        name="Gradient Boosting"
    )
    classifiers.append(gb_clf)

    # 3. XGBoost (if available)
    if HAS_XGB:
        print("\n" + "="*70)
        print("Setting up XGBoost Classifier")
        print("="*70)
        xgb_clf = MultiClassClassifier(
            xgb.XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            name="XGBoost"
        )
        classifiers.append(xgb_clf)

    # 4. LightGBM (if available)
    if HAS_LGB:
        print("\n" + "="*70)
        print("Setting up LightGBM Classifier")
        print("="*70)
        lgb_clf = MultiClassClassifier(
            lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=10,
                num_leaves=50,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            name="LightGBM"
        )
        classifiers.append(lgb_clf)

    # Evaluate all classifiers
    results = {}
    for clf in classifiers:
        results[clf.name] = evaluate_cv(clf, X_train, y_train, cv=5)

    # Plot ROC curves
    plot_roc_curves(classifiers, X_train, y_train, cv=3)  # Use 3 folds for speed

    # Find best classifier
    best_clf_name = max(results.keys(), key=lambda k: results[k]['avg_auc'])
    best_clf = [clf for clf in classifiers if clf.name == best_clf_name][0]

    print("\n" + "="*70)
    print(f"Best Classifier: {best_clf_name}")
    print(f"Average AUC: {results[best_clf_name]['avg_auc']:.4f}")
    print("="*70)

    # Train best classifier on full training set
    print("\nTraining best classifier on full training set...")
    best_clf.fit(X_train, y_train)

    # Extract and save feature importance
    print("\nExtracting feature importance...")
    importance_dict = best_clf.get_feature_importance()

    if importance_dict:
        # Average importance across all classes
        avg_importance = np.mean([importance_dict[cls] for cls in [1, 2, 3, 4]], axis=0)
        top_features_idx = np.argsort(avg_importance)[::-1][:20]

        print("\nTop 20 Most Important Features:")
        for i, feat_idx in enumerate(top_features_idx):
            print(f"  {i+1}. Feature {feat_idx}: {avg_importance[feat_idx]:.6f}")

        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(20), avg_importance[top_features_idx])
        plt.xlabel('Feature Rank')
        plt.ylabel('Importance Score')
        plt.title(f'Top 20 Feature Importance - {best_clf_name}')
        plt.xticks(range(20), [f'F{i}' for i in top_features_idx], rotation=45)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved feature importance plot to: feature_importance.png")
        plt.close()

        # Save to CSV
        importance_df = pd.DataFrame({
            'feature_index': range(len(avg_importance)),
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        importance_df.to_csv('feature_importance.csv', index=False)
        print("Saved feature importance to: feature_importance.csv")

    # Generate predictions on test set
    print("\nGenerating predictions on test set...")
    test_proba = best_clf.predict_proba(X_test)
    test_pred = best_clf.predict(X_test)

    # Save predictions
    save_predictions(test_proba, test_pred, 'testLabel.txt')

    print("\n" + "="*70)
    print("Classification pipeline complete!")
    print("="*70)
    print("\nGenerated files:")
    print("  - roc_curves_comparison.png: ROC curves for all classifiers")
    print("  - feature_importance.png: Top features visualization")
    print("  - feature_importance.csv: Feature importance scores")
    print("  - testLabel.txt: Test set predictions (5-column format)")

    return best_clf, results


if __name__ == "__main__":
    best_clf, results = main()
