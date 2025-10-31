# CSC 621 - HW3 Classification Project

## Team Members
Brandon 
Mounika

## Project Overview
This project implements a multi-class classification system for a 4-class problem using various machine learning algorithms. The solution employs a One-vs-Rest strategy to handle the multi-class classification task.

## Dataset
- **Training set**: 27,617 samples × 411 features
- **Test set**: 13,082 samples × 411 features
- **Classes**: 1, 2, 3, 4
- **Missing values**: 138 samples (0.50%) in feature column 410

### Class Distribution
- Class 1: 8,874 samples (32.13%)
- Class 2: 6,127 samples (22.19%)
- Class 3: 8,483 samples (30.72%)
- Class 4: 4,133 samples (14.97%)

Note: Class 4 is underrepresented, requiring class balancing strategies.

## Approach

### 1. Missing Value Handling
- **Strategy**: Median imputation
- **Rationale**: Only 0.50% of data has missing values (column 410 only)
- Median imputation is robust to outliers and preserves the distribution
- Alternative approaches tested: mean imputation, KNN imputation (median performed best)

### 2. Preprocessing
- **Feature scaling**: RobustScaler
  - More robust to outliers than StandardScaler
  - Important given the feature value range (-2.69 to 3.26)
- **No feature engineering**: Features appear to be already engineered/normalized

### 3. Classification Algorithms Tested

#### Random Forest
- Ensemble of 200 decision trees
- Hyperparameters: max_depth=20, min_samples_split=10
- Class weighting to handle imbalanced classes
- Provides feature importance scores

#### Gradient Boosting
- 150 boosting stages
- Learning rate: 0.1
- Hyperparameters: max_depth=7, subsample=0.8

#### XGBoost
- 200 estimators with early stopping
- Learning rate: 0.05
- Regularization: L1=0.1, L2=1.0
- Advanced gradient boosting with better performance

#### LightGBM
- 200 estimators
- Leaf-wise tree growth
- Optimized for large datasets
- Hyperparameters: num_leaves=50, max_depth=10

### 4. Model Selection Strategy
- **One-vs-Rest approach**: Train separate binary classifier for each class
- **Cross-validation**: 5-fold stratified CV to preserve class distribution
- **Evaluation metric**: ROC-AUC for each class (as specified)
- **Final prediction**: Argmax of class probabilities

### 5. Important Features
The top features were extracted using feature importance from the best performing model:
- Features are ranked by average importance across all 4 class classifiers
- Top 20 features account for majority of predictive power
- See `feature_importance.csv` for complete rankings

## Results

### Best Pipeline Performance: **91.58% Validation Accuracy**

**File**: `testLabel_lightgbm_mi_18ensemble.txt`
**Pipeline**: `ultimate_pipeline.py`
**Average AUC**: 0.9885

#### Pipeline Evolution
1. **Fast XGBoost** (`fast_xgboost.py`): 90.01% - Baseline
2. **Hypertuned LightGBM** (`hypertuned_pipeline.py`): 90.42% - Config search
3. **Partner-Inspired** (`partner_inspired_pipeline.py`): 90.84% - Noise-robust
4. **Ultimate Pipeline** (`ultimate_pipeline.py`): **91.58%** ⭐ - MI + 18-model ensemble

#### Class-wise AUC (Ultimate Pipeline)
- Class 1: 0.9901
- Class 2: 0.9804
- Class 3: 0.9911
- Class 4: 0.8843 (lowest due to class imbalance)

#### Benchmarks
- **Target**: 92.82%
- **Partner**: 92.58%
- **Our Result**: 91.58% (Gap: -1.24% to target)

### Final Classifier Selection
LightGBM with Mutual Information feature selection and 18-model weighted ensemble.

## Files

### Code - Pipeline Evolution
- `explore_data.py`: Initial exploratory data analysis
- `classification_pipeline.py`: Original multi-algorithm pipeline
- `fast_xgboost.py`: Quick XGBoost baseline (90.01%)
- `optimized_pipeline.py`: Early SMOTE attempt (87.36% - failed)
- `hypertuned_pipeline.py`: LightGBM with 4 configs (90.42%)
- `partner_inspired_pipeline.py`: Noise-robust approach (90.84%)
- `ultimate_pipeline.py`: **BEST** - MI + 18-ensemble (91.58%)
- `final_comprehensive_pipeline.py`: Multi-algo attempt (crashed)

### Predictions - Test Set
- `testLabel_lightgbm_mi_18ensemble.txt`: **PRIMARY SUBMISSION** (91.58%)
- `testLabel_ultimate.txt`: Same as above (alternate name)
- `testLabel.txt`: Generic testLabel output
- `testLabel_partner_inspired.txt`: From partner-inspired pipeline
- `testLabel_hypertuned.txt`: From hypertuned pipeline
- `testLabel_optimized.txt`: From SMOTE attempt
- `testLabel_final.txt`: From fast baseline
- `testLabel_quick.txt`: Early quick test

### Visualizations
- `roc_curves_xgboost.png`: XGBoost baseline ROC curves
- `roc_curves_optimized.png`: Multi-model comparison
- `roc_curves_hypertuned.png`: LightGBM config comparison
- `eda_results.png`: Exploratory data analysis

## How to Reproduce Results

### Requirements
```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
```

### Step 1: Exploratory Data Analysis
```bash
python explore_data.py
```
This generates `eda_results.png` with data distribution visualizations.

### Step 2: Train and Evaluate Models
```bash
python classification_pipeline.py
```
This script:
1. Loads and preprocesses the data
2. Trains all classification algorithms
3. Performs 5-fold cross-validation
4. Generates ROC curves
5. Extracts feature importance
6. Trains the best model on full training set
7. Generates test set predictions in required format

Output: `testLabel.txt`, `roc_curves_comparison.png`, `feature_importance.png`

### Step 3: Generate Blind Predictions (when blind data available)
```bash
python generate_blind_predictions.py
```
Loads the trained best model and generates predictions for blind data.
Output: `blindLabel.txt`

## Prediction Format
All prediction files follow the required 5-column tab-delimited format:
- Column 1: Class 1 vs rest probability
- Column 2: Class 2 vs rest probability
- Column 3: Class 3 vs rest probability
- Column 4: Class 4 vs rest probability
- Column 5: Final predicted class (1, 2, 3, or 4)

Each row corresponds to a sample in the test/blind data.

## Key Insights

### Data Characteristics
1. Relatively balanced dataset except for class 4 (15% of data)
2. Very few missing values (0.5%)
3. Features are already normalized/standardized
4. No obvious outliers requiring removal

### Model Performance
1. Tree-based ensemble methods outperform linear models
2. Boosting methods (XGBoost, LightGBM) show best performance
3. Class 4 (minority class) has lower AUC than other classes
4. Cross-validation reveals stable performance across folds

### Feature Importance
1. Approximately 50-100 features contribute significantly to predictions
2. Top 20 features capture majority of discriminative power
3. Feature importance varies across different class classifiers

## Ultimate Pipeline Strategy (91.58%)

### Key Techniques

#### 1. Mutual Information Feature Selection
```python
mi_scores = mutual_info_classif(X_base, y, random_state=42)
mi_threshold = np.percentile(mi_scores, 10)  # Keep top 90%
selected_features = np.where(mi_scores > mi_threshold)[0]
# Result: 411 → 314 features
```

#### 2. Advanced Feature Engineering
- Polynomial features (x², x³) for top 15 MI features
- Interaction features (multiplication, ratios) for top 8 features
- Statistical aggregations (mean, std, median, Q1, Q3, min, max)
- Final: 314 → 347 features

#### 3. 18-Model Weighted Ensemble
- 15 models from best hyperparameter configuration
- 3 models from second-best configuration (diversity)
- Different random seeds for each model
- Weighted averaging (0.7 weight for second-best models)

#### 4. Hyperparameter Configuration (Best)
```python
{
    'learning_rate': 0.025,
    'num_leaves': 70,
    'max_depth': 14,
    'min_data_in_leaf': 12,
    'feature_fraction': 0.88,
    'bagging_fraction': 0.88,
    'lambda_l1': 0.1,
    'lambda_l2': 0.4,
}
```

## Next Steps - Optimization Branch

Planned enhancements to close 1.24% gap to 92.82% target:

1. **Weighted Ensemble Optimization** (Quick Win)
   - Optimize weights on validation set instead of equal/fixed weights
   - Expected gain: 0.3-0.5%

2. **Pseudo-Labeling** (Semi-Supervised Learning)
   - Use high-confidence test predictions as training data
   - Iteratively retrain model with expanded dataset
   - Can add 1000s of quality training samples

3. **Stacking Ensemble**
   - Train meta-learner on base model predictions
   - Use diverse models (LightGBM, XGBoost, CatBoost, Neural Net)
   - Capture non-linear relationships

4. **Optuna Hyperparameter Tuning**
   - Bayesian optimization instead of manual configs
   - Search 10-20 configurations systematically
   - Per-class hyperparameter tuning

## References
- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- LightGBM documentation: https://lightgbm.readthedocs.io/
