"""
CSC 621 - HW3 Classification Project
Generate Blind Data Predictions

This script loads the trained best classifier and generates predictions
for the blind dataset when it becomes available.

Usage:
    python generate_blind_predictions.py
"""

import numpy as np
import pandas as pd
import pickle
import sys
from classification_pipeline import MultiClassClassifier, load_data

def load_blind_data(blind_data_path='HW3-dataset-1/blindData.txt'):
    """Load blind data"""
    try:
        print(f"Loading blind data from: {blind_data_path}")
        blind_data = pd.read_csv(blind_data_path, header=None)

        # Replace empty strings with NaN (though blind data shouldn't have any)
        blind_data = blind_data.replace('', np.nan)

        # Convert to numeric
        blind_data = blind_data.apply(pd.to_numeric, errors='coerce')

        X_blind = blind_data.values
        print(f"Blind data shape: {X_blind.shape}")

        return X_blind
    except FileNotFoundError:
        print(f"Error: Blind data file not found at {blind_data_path}")
        print("Please ensure blindData.txt is in the HW3-dataset-1 directory")
        sys.exit(1)


def save_predictions(predictions_proba, predictions_class, filename):
    """Save predictions in required format (5-column tab-delimited)"""
    output = np.column_stack([predictions_proba, predictions_class])

    # Save with tab delimiter
    np.savetxt(filename, output, delimiter='\t', fmt='%.6f\t%.6f\t%.6f\t%.6f\t%d')
    print(f"Predictions saved to: {filename}")


def main():
    """Generate blind predictions using the trained best model"""
    print("="*70)
    print("Generating Blind Data Predictions")
    print("="*70)

    # Check if trained model exists
    try:
        with open('best_model.pkl', 'rb') as f:
            best_clf = pickle.load(f)
        print("Loaded trained model from: best_model.pkl")
    except FileNotFoundError:
        print("\nError: Trained model not found!")
        print("Please run classification_pipeline.py first to train the model.")
        print("\nAlternatively, training model from scratch...")

        # Train the model
        from classification_pipeline import main as train_main
        best_clf, _ = train_main()

        # Save the model for future use
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(best_clf, f)
        print("Model saved to: best_model.pkl")

    # Load blind data
    X_blind = load_blind_data()

    # Generate predictions
    print("\nGenerating predictions...")
    blind_proba = best_clf.predict_proba(X_blind)
    blind_pred = best_clf.predict(X_blind)

    # Display prediction statistics
    unique, counts = np.unique(blind_pred, return_counts=True)
    print("\nPrediction distribution:")
    for cls, count in zip(unique, counts):
        print(f"  Class {cls}: {count} samples ({count/len(blind_pred)*100:.2f}%)")

    # Save predictions
    save_predictions(blind_proba, blind_pred, 'blindLabel.txt')

    print("\n" + "="*70)
    print("Blind predictions complete!")
    print("="*70)


if __name__ == "__main__":
    main()
