"""
CSC 621 - HW3 Final Classification Pipeline (Optimized)

This is an optimized version that:
- Uses 3-fold CV instead of 5-fold for speed
- Tests multiple algorithms
- Generates ROC curves
- Extracts feature importance
- Creates final predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
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

class MultiClassClassifier:
    """Multi-class classifier using One-vs-Rest"""

    def __init__(self, base_classifier, name="Classifier"):
        self.name = name
        self.base_classifier = base_classifier
        self.classifiers = {}
        self.classes = [1, 2, 3, 4]
        self.scaler = None
        self.imputer = None

    def preprocess_data(self, X_train, fit=True):
        if fit:
            self.imputer = SimpleImputer(strategy='median', missing_values=np.nan)
            X_imputed = self.imputer.fit_transform(X_train)
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X_imputed)
        else:
            X_imputed = self.imputer.transform(X_train)
            X_scaled = self.scaler.transform(X_imputed)
        return X_scaled

    def fit(self, X_train, y_train):
        X_processed = self.preprocess_data(X_train, fit=True)

        for cls in self.classes:
            y_binary = (y_train == cls).astype(int)
            from sklearn.base import clone
            clf = clone(self.base_classifier)
            clf.fit(X_processed, y_binary)
            self.classifiers[cls] = clf

        return self

    def predict_proba(self, X_test):
        X_processed = self.preprocess_data(X_test, fit=False)
        probas = np.zeros((len(X_test), len(self.classes)))

        for i, cls in enumerate(self.classes):
            if hasattr(self.classifiers[cls], 'predict_proba'):
                probas[:, i] = self.classifiers[cls].predict_proba(X_processed)[:, 1]
            else:
                probas[:, i] = self.classifiers[cls].decision_function(X_processed)

        return probas

    def predict(self, X_test):
        probas = self.predict_proba(X_test)
        return np.argmax(probas, axis=1) + 1

    def get_feature_importance(self):
        importance_dict = {}
        for cls in self.classes:
            clf = self.classifiers[cls]
            if hasattr(clf, 'feature_importances_'):
                importance_dict[cls] = clf.feature_importances_
            elif hasattr(clf, 'coef_'):
                importance_dict[cls] = np.abs(clf.coef_[0])
        return importance_dict


def load_data():
    print("Loading data...")
    train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
    train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
    test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

    train_data = train_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')
    test_data = test_data.replace('', np.nan).apply(pd.to_numeric, errors='coerce')

    X_train = train_data.values
    y_train = train_labels['label'].values
    X_test = test_data.values

    print(f"Training: {X_train.shape}, Labels: {y_train.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_test


def evaluate_cv(classifier, X_train, y_train, cv=3):
    print(f"\nCross-validating {classifier.name}...")

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    auc_scores = {1: [], 2: [], 3: [], 4: []}
    accuracy_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold + 1}/{cv}...")
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        classifier.fit(X_tr, y_tr)
        y_proba = classifier.predict_proba(X_val)
        y_pred = classifier.predict(X_val)

        for i, cls in enumerate([1, 2, 3, 4]):
            y_binary = (y_val == cls).astype(int)
            auc = roc_auc_score(y_binary, y_proba[:, i])
            auc_scores[cls].append(auc)

        acc = accuracy_score(y_val, y_pred)
        accuracy_scores.append(acc)

    print(f"\n{classifier.name} Results:")
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


def plot_roc_curves(classifiers, X_train, y_train, cv=3):
    print("\nGenerating ROC curves...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()

    colors = ['blue', 'red', 'green', 'orange', 'purple']

    for cls_idx, cls in enumerate([1, 2, 3, 4]):
        ax = axes[cls_idx]
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

        for clf_idx, classifier in enumerate(classifiers):
            tprs = []
            aucs = []
            mean_fpr = np.linspace(0, 1, 100)

            for train_idx, val_idx in skf.split(X_train, y_train):
                X_tr, X_val = X_train[train_idx], X_train[val_idx]
                y_tr, y_val = y_train[train_idx], y_train[val_idx]

                classifier.fit(X_tr, y_tr)
                y_proba = classifier.predict_proba(X_val)

                y_binary = (y_val == cls).astype(int)
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
                   linewidth=2, color=colors[clf_idx])

        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'ROC Curve - Class {cls} vs Rest', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('roc_curves_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: roc_curves_comparison.png")
    plt.close()


def save_predictions(predictions_proba, predictions_class, filename):
    output = np.column_stack([predictions_proba, predictions_class])
    np.savetxt(filename, output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
    print(f"Saved: {filename}")


def main():
    print("="*70)
    print("CSC 621 - HW3 Final Classification Pipeline")
    print("="*70)

    X_train, y_train, X_test = load_data()

    # Define classifiers
    classifiers = []

    print("\n" + "="*70)
    print("Setting up classifiers...")
    print("="*70)

    # Random Forest
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

    # Gradient Boosting
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

    # XGBoost
    if HAS_XGB:
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

    # LightGBM
    if HAS_LGB:
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
        results[clf.name] = evaluate_cv(clf, X_train, y_train, cv=3)

    # Plot ROC curves
    plot_roc_curves(classifiers, X_train, y_train, cv=3)

    # Find best classifier
    best_clf_name = max(results.keys(), key=lambda k: results[k]['avg_auc'])
    best_clf = [clf for clf in classifiers if clf.name == best_clf_name][0]

    print("\n" + "="*70)
    print(f"BEST CLASSIFIER: {best_clf_name}")
    print(f"Average AUC: {results[best_clf_name]['avg_auc']:.4f}")
    print(f"Accuracy: {results[best_clf_name]['accuracy']:.4f}")
    print("="*70)

    # Train best classifier on full data
    print("\nTraining best classifier on full training set...")
    best_clf.fit(X_train, y_train)

    # Extract feature importance
    print("\nExtracting feature importance...")
    importance_dict = best_clf.get_feature_importance()

    if importance_dict:
        avg_importance = np.mean([importance_dict[cls] for cls in [1, 2, 3, 4]], axis=0)
        top_features_idx = np.argsort(avg_importance)[::-1][:20]

        print("\nTop 20 Most Important Features:")
        for i, feat_idx in enumerate(top_features_idx):
            print(f"  {i+1}. Feature {feat_idx}: {avg_importance[feat_idx]:.6f}")

        # Save feature importance plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(20), avg_importance[top_features_idx], color='steelblue', edgecolor='black')
        plt.xlabel('Feature Rank', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title(f'Top 20 Feature Importance - {best_clf_name}', fontsize=14, fontweight='bold')
        plt.xticks(range(20), [f'F{i}' for i in top_features_idx], rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("Saved: feature_importance.png")
        plt.close()

        # Save to CSV
        importance_df = pd.DataFrame({
            'feature_index': range(len(avg_importance)),
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        importance_df.to_csv('feature_importance.csv', index=False)
        print("Saved: feature_importance.csv")

    # Generate test predictions
    print("\nGenerating test predictions...")
    test_proba = best_clf.predict_proba(X_test)
    test_pred = best_clf.predict(X_test)

    save_predictions(test_proba, test_pred, 'testLabel.txt')

    # Save results summary
    with open('results_summary.txt', 'w') as f:
        f.write("CSC 621 - HW3 Classification Results\n")
        f.write("="*70 + "\n\n")
        f.write("All Models Performance (3-fold CV):\n")
        f.write("-"*70 + "\n")
        for name, res in results.items():
            f.write(f"\n{name}:\n")
            f.write(f"  Accuracy: {res['accuracy']:.4f}\n")
            f.write(f"  Average AUC: {res['avg_auc']:.4f}\n")
            for cls in [1, 2, 3, 4]:
                f.write(f"  Class {cls} AUC: {res['auc_scores'][cls]:.4f}\n")
        f.write("\n" + "="*70 + "\n")
        f.write(f"BEST MODEL: {best_clf_name}\n")
        f.write(f"Average AUC: {results[best_clf_name]['avg_auc']:.4f}\n")
        f.write(f"Accuracy: {results[best_clf_name]['accuracy']:.4f}\n")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nGenerated Files:")
    print("  - testLabel.txt: Test predictions (5-column format)")
    print("  - roc_curves_comparison.png: ROC curves for all models")
    print("  - feature_importance.png: Top 20 features visualization")
    print("  - feature_importance.csv: All feature importance scores")
    print("  - results_summary.txt: Detailed results summary")

    return best_clf, results


if __name__ == "__main__":
    best_clf, results = main()
