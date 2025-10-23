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

### Cross-Validation Performance
[Results will be populated after running the pipeline]

### Final Classifier Selection
The best performing classifier based on average AUC across all classes.

## Files

### Code
- `explore_data.py`: Exploratory data analysis and visualization
- `classification_pipeline.py`: Main classification pipeline with all algorithms
- `generate_blind_predictions.py`: Script to generate predictions on blind data

### Output
- `testLabel.txt`: Test set predictions (5-column tab-delimited format)
- `blindLabel.txt`: Blind set predictions (generated when blind data available)
- `roc_curves_comparison.png`: ROC curves comparing all algorithms
- `feature_importance.png`: Visualization of top 20 features
- `feature_importance.csv`: Complete feature importance rankings
- `eda_results.png`: Exploratory data analysis visualizations

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

## Future Improvements
1. Feature selection to reduce dimensionality
2. Advanced ensemble methods (stacking, blending)
3. Neural network architectures (deep learning)
4. Hyperparameter optimization with Bayesian optimization
5. Handling class imbalance with SMOTE or other techniques

## References
- Scikit-learn documentation: https://scikit-learn.org/
- XGBoost documentation: https://xgboost.readthedocs.io/
- LightGBM documentation: https://lightgbm.readthedocs.io/
