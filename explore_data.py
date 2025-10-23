"""
CSC 621 - HW3 Classification Project
Exploratory Data Analysis Script

This script explores the training data to understand:
- Data dimensions
- Missing values
- Class distribution
- Feature statistics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Load the data
print("Loading data...")
train_data = pd.read_csv('HW3-dataset-1/trainingData.txt', header=None)
train_labels = pd.read_csv('HW3-dataset-1/trainingTruth.txt', header=None, names=['label'])
test_data = pd.read_csv('HW3-dataset-1/testData.txt', header=None)

print(f"\nTraining data shape: {train_data.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test data shape: {test_data.shape}")

# Analyze missing values
print("\n" + "="*60)
print("MISSING VALUES ANALYSIS")
print("="*60)

# Replace empty strings with NaN
train_data_clean = train_data.replace('', np.nan)

# Count missing values per row
missing_per_row = train_data_clean.isnull().sum(axis=1)
print(f"\nRows with missing values: {(missing_per_row > 0).sum()} / {len(train_data_clean)}")
print(f"Percentage of rows with missing values: {(missing_per_row > 0).sum() / len(train_data_clean) * 100:.2f}%")

# Count missing values per column
missing_per_col = train_data_clean.isnull().sum(axis=0)
print(f"\nColumns with missing values: {(missing_per_col > 0).sum()} / {len(train_data_clean.columns)}")
print(f"Total missing values: {train_data_clean.isnull().sum().sum()}")
print(f"Percentage of total values that are missing: {train_data_clean.isnull().sum().sum() / (train_data_clean.shape[0] * train_data_clean.shape[1]) * 100:.2f}%")

# Top columns with most missing values
print("\nTop 10 columns with most missing values:")
missing_sorted = missing_per_col.sort_values(ascending=False).head(10)
for idx, count in missing_sorted.items():
    print(f"  Column {idx}: {count} missing ({count/len(train_data_clean)*100:.2f}%)")

# Class distribution
print("\n" + "="*60)
print("CLASS DISTRIBUTION")
print("="*60)
class_counts = train_labels['label'].value_counts().sort_index()
print("\nTraining set class distribution:")
for label, count in class_counts.items():
    print(f"  Class {label}: {count} samples ({count/len(train_labels)*100:.2f}%)")

# Basic statistics
print("\n" + "="*60)
print("BASIC STATISTICS")
print("="*60)
print(f"\nNumber of features: {train_data.shape[1]}")
print(f"Number of training samples: {train_data.shape[0]}")
print(f"Number of test samples: {test_data.shape[0]}")

# Convert to numeric for statistics (excluding missing values)
train_numeric = train_data_clean.apply(pd.to_numeric, errors='coerce')
print(f"\nFeature value ranges:")
print(f"  Min value across all features: {train_numeric.min().min():.4f}")
print(f"  Max value across all features: {train_numeric.max().max():.4f}")
print(f"  Mean value across all features: {train_numeric.mean().mean():.4f}")
print(f"  Median value across all features: {train_numeric.median().median():.4f}")

# Save exploration results
print("\n" + "="*60)
print("Saving exploration results...")
print("="*60)

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Class distribution
axes[0, 0].bar(class_counts.index, class_counts.values)
axes[0, 0].set_xlabel('Class')
axes[0, 0].set_ylabel('Count')
axes[0, 0].set_title('Class Distribution in Training Set')
axes[0, 0].set_xticks([1, 2, 3, 4])

# 2. Missing values per row histogram
axes[0, 1].hist(missing_per_row, bins=50, edgecolor='black')
axes[0, 1].set_xlabel('Number of Missing Values')
axes[0, 1].set_ylabel('Number of Rows')
axes[0, 1].set_title('Distribution of Missing Values per Row')

# 3. Missing values per column (top 50)
top_missing_cols = missing_per_col.sort_values(ascending=False).head(50)
axes[1, 0].bar(range(len(top_missing_cols)), top_missing_cols.values)
axes[1, 0].set_xlabel('Column Index (sorted by missing count)')
axes[1, 0].set_ylabel('Missing Count')
axes[1, 0].set_title('Top 50 Columns with Most Missing Values')

# 4. Feature value distribution (sample 10 random features)
sample_features = np.random.choice(train_numeric.columns, size=min(10, len(train_numeric.columns)), replace=False)
for col in sample_features:
    axes[1, 1].hist(train_numeric[col].dropna(), bins=30, alpha=0.5, label=f'F{col}')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Sample Feature Distributions (10 random features)')

plt.tight_layout()
plt.savefig('eda_results.png', dpi=300, bbox_inches='tight')
print("Saved visualization to: eda_results.png")

print("\nExploration complete!")
