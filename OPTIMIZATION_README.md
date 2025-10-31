# Optimization Branch - Advanced Strategies

This branch contains advanced optimization strategies to close the gap from 91.58% to 92.82%+ accuracy.

## Current Best (Main Branch)

- **Validation Accuracy**: 91.58%
- **Average AUC**: 0.9885
- **Strategy**: LightGBM with MI feature selection + 18-model ensemble
- **File**: `testLabel_lightgbm_mi_18ensemble.txt`

## Goals

- **Target**: 92.82% (benchmark)
- **Partner**: 92.58%
- **Gap to Close**: 1.24%

## Optimization Strategies

### 1. Weighted Ensemble Optimization ⚡ (Quick Win)

**Expected Gain**: 0.3-0.5%
**Time**: ~5 minutes

Instead of fixed weights (1.0 for best, 0.7 for second-best), optimize weights on validation set.

**Approach**:
```python
from scipy.optimize import minimize

def ensemble_loss(weights, predictions, y_true):
    weighted_pred = np.average(predictions, axis=0, weights=weights)
    pred_labels = np.argmax(weighted_pred, axis=1)
    return -accuracy_score(y_true, pred_labels)

# Optimize weights
result = minimize(ensemble_loss, initial_weights,
                  constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
                  bounds=[(0, 1)] * n_models)
```

**Files**:
- `weighted_ensemble_optimizer.py` - Optimizes weights for existing models

---

### 2. Pseudo-Labeling (Semi-Supervised)

**Expected Gain**: 0.5-1.0%
**Time**: ~30 minutes

Use high-confidence test predictions as additional training data.

**Approach**:
1. Train initial model on training data
2. Predict on test set
3. Select high-confidence predictions (e.g., max probability > 0.95)
4. Add pseudo-labeled samples to training set
5. Retrain model on expanded dataset
6. Iterate 2-3 times

**Algorithm**:
```python
confidence_threshold = 0.95
for iteration in range(3):
    model.fit(X_train_expanded, y_train_expanded)
    test_probs = model.predict(X_test)

    # Select high-confidence predictions
    max_probs = np.max(test_probs, axis=1)
    high_conf_mask = max_probs > confidence_threshold

    # Add to training set
    X_train_expanded = np.vstack([X_train_expanded, X_test[high_conf_mask]])
    y_train_expanded = np.concatenate([y_train_expanded,
                                       np.argmax(test_probs[high_conf_mask], axis=1)])
```

**Key Parameters**:
- Confidence threshold: 0.95 (start conservative)
- Iterations: 2-3
- Can add 1000s of samples per iteration

**Files**:
- `pseudo_labeling_pipeline.py` - Implements semi-supervised learning

---

### 3. Stacking Ensemble (Meta-Learner)

**Expected Gain**: 0.4-0.7%
**Time**: ~45 minutes

Train diverse base models, then use meta-learner to combine predictions.

**Approach**:
```python
# Layer 1: Diverse base models
base_models = [
    LGBMClassifier(params1),
    XGBClassifier(params2),
    CatBoostClassifier(params3),
    MLPClassifier(params4)  # Neural net for diversity
]

# Generate meta-features via CV
meta_features = []
for model in base_models:
    cv_preds = cross_val_predict(model, X_train, y_train,
                                 cv=5, method='predict_proba')
    meta_features.append(cv_preds)

meta_X = np.hstack(meta_features)

# Layer 2: Meta-learner
meta_model = LogisticRegression(C=0.1, multi_class='multinomial')
meta_model.fit(meta_X, y_train)
```

**Base Models**:
1. LightGBM (current best config)
2. XGBoost (complementary boosting)
3. CatBoost (different tree algorithm)
4. Neural Network (captures non-linear patterns)

**Meta-Learner Options**:
- Logistic Regression (simple, fast)
- Small Neural Network (more capacity)
- LightGBM (can learn complex relationships)

**Files**:
- `stacking_ensemble_pipeline.py` - Implements 2-layer stacking

---

### 4. Optuna Hyperparameter Tuning

**Expected Gain**: 0.3-0.6%
**Time**: 1-2 hours

Systematic Bayesian optimization instead of manual config search.

**Approach**:
```python
import optuna

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('lr', 0.01, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 40, 100),
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'min_data_in_leaf': trial.suggest_int('min_data', 8, 30),
        'feature_fraction': trial.suggest_float('feat_frac', 0.7, 0.95),
        'bagging_fraction': trial.suggest_float('bag_frac', 0.7, 0.95),
        'lambda_l1': trial.suggest_float('l1', 0.0, 1.0),
        'lambda_l2': trial.suggest_float('l2', 0.0, 1.5),
    }

    cv_scores = cross_val_score(LGBMClassifier(**params), X, y, cv=5)
    return cv_scores.mean()

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

**Search Space**:
- 50-100 trials
- 8-10 hyperparameters
- Pruning for early stopping

**Files**:
- `optuna_tuning_pipeline.py` - Bayesian optimization with Optuna

---

## Implementation Order

### Phase 1: Quick Wins (Today)
1. ✅ Weighted Ensemble Optimization (~5 min)
   - Immediate improvement without retraining
   - Low risk, fast validation

### Phase 2: Semi-Supervised (Today)
2. 🔄 Pseudo-Labeling (~30 min)
   - Leverages test set effectively
   - Can iterate quickly

### Phase 3: Advanced Ensemble (Today/Tomorrow)
3. 🔄 Stacking Ensemble (~45 min)
   - More sophisticated than averaging
   - Requires diverse base models

### Phase 4: Systematic Search (Optional)
4. ⏰ Optuna Tuning (~1-2 hours)
   - If still short of 92.82%
   - Most time-intensive

## Expected Results

| Strategy | Expected Gain | Cumulative | Total Expected |
|----------|--------------|------------|----------------|
| Baseline (Ultimate) | - | - | 91.58% |
| + Weighted Ensemble | +0.3-0.5% | +0.4% | 91.98% |
| + Pseudo-Labeling | +0.5-1.0% | +1.1% | 92.68% |
| + Stacking | +0.4-0.7% | +1.6% | 93.18% |
| + Optuna (if needed) | +0.3-0.6% | +2.0% | 93.58% |

**Note**: Gains are not necessarily additive - some strategies overlap.

## Risk Assessment

### Low Risk ✅
- **Weighted Ensemble**: Can only improve, uses existing models
- **Optuna**: Systematic search, well-tested

### Medium Risk ⚠️
- **Pseudo-Labeling**: Could amplify errors if confidence threshold too low
- **Stacking**: Depends on base model diversity

### Mitigation
- Always validate on held-out CV folds
- Keep baseline predictions for comparison
- Use conservative thresholds initially

## Files in This Branch

### Optimization Scripts
- `weighted_ensemble_optimizer.py` - Quick win optimization
- `pseudo_labeling_pipeline.py` - Semi-supervised learning
- `stacking_ensemble_pipeline.py` - Meta-learner stacking
- `optuna_tuning_pipeline.py` - Bayesian hyperparameter search

### Output Predictions
- `testLabel_weighted_optimized.txt` - After weight optimization
- `testLabel_pseudo_labeled.txt` - After semi-supervised
- `testLabel_stacked.txt` - After stacking ensemble
- `testLabel_optuna_tuned.txt` - After Optuna tuning

### Documentation
- `OPTIMIZATION_README.md` - This file
- `RESULTS.md` - Will track results of each optimization

## Progress Tracking

- [x] Create optimization branch
- [x] Document optimization strategies
- [ ] Implement weighted ensemble optimization
- [ ] Implement pseudo-labeling
- [ ] Implement stacking ensemble
- [ ] Implement Optuna tuning
- [ ] Compare all results and select best
- [ ] Push optimized predictions to main

## Next Steps

1. Start with weighted ensemble optimization (5 min)
2. If successful (>92.0%), try pseudo-labeling
3. If reaches 92.5%+, submit and document
4. Otherwise, implement stacking and Optuna

## References

- Pseudo-Labeling: Lee, D. (2013). "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning"
- Stacking: Wolpert, D. (1992). "Stacked Generalization"
- Optuna: Akiba et al. (2019). "Optuna: A Next-generation Hyperparameter Optimization Framework"
- LightGBM Advanced Tuning: https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html
