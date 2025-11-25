#!/usr/bin/env python3
"""
Test the Hybrid Model with a Dataset

This script allows users to test the trained hybrid model (LSTM + Random Forest + Meta-Learner)
with their own dataset to get prediction results.

Usage:
    python test_model.py --static_data path/to/static.csv --temporal_data path/to/temporal.csv

Author: Master's Thesis Research
Date: November 2024
"""

import os
import argparse
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow import keras

# Default model paths (using the latest trained model)
DEFAULT_MODEL_TIMESTAMP = "20251104_171930"
DEFAULT_RESULTS_DIR = "results"
DEFAULT_LSTM_PATH = os.path.join(DEFAULT_RESULTS_DIR, f"lstm_model_{DEFAULT_MODEL_TIMESTAMP}.h5")
DEFAULT_RF_PATH = os.path.join(DEFAULT_RESULTS_DIR, f"rf_model_{DEFAULT_MODEL_TIMESTAMP}.pkl")
DEFAULT_META_LEARNER_PATH = os.path.join(DEFAULT_RESULTS_DIR, f"meta_learner_{DEFAULT_MODEL_TIMESTAMP}.pkl")
DEFAULT_PREPROCESSING_PATH = os.path.join(DEFAULT_RESULTS_DIR, f"preprocessing_objects_{DEFAULT_MODEL_TIMESTAMP}.pkl")


def load_models(lstm_path=None, rf_path=None, meta_learner_path=None, preprocessing_path=None):
    """
    Load all trained models.
    
    Parameters
    ----------
    lstm_path : str, optional
        Path to LSTM model file
    rf_path : str, optional
        Path to Random Forest model file
    meta_learner_path : str, optional
        Path to Meta-learner model file
    preprocessing_path : str, optional
        Path to preprocessing objects file
    
    Returns
    -------
    tuple
        (lstm_model, rf_model, meta_learner, preprocessing_objects)
    """
    # Use defaults if not provided
    lstm_path = lstm_path or DEFAULT_LSTM_PATH
    rf_path = rf_path or DEFAULT_RF_PATH
    meta_learner_path = meta_learner_path or DEFAULT_META_LEARNER_PATH
    preprocessing_path = preprocessing_path or DEFAULT_PREPROCESSING_PATH
    
    print(f"Loading LSTM model from: {lstm_path}")
    # Try to load the model with compile=False to avoid compatibility issues
    try:
        lstm_model = keras.models.load_model(lstm_path)
    except (ValueError, TypeError) as e:
        print(f"Note: Loading model with compile=False due to version compatibility")
        lstm_model = keras.models.load_model(lstm_path, compile=False)
        # Recompile with compatible settings
        lstm_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    print(f"Loading Random Forest model from: {rf_path}")
    rf_model = joblib.load(rf_path)
    
    print(f"Loading Meta-learner from: {meta_learner_path}")
    meta_learner = joblib.load(meta_learner_path)
    
    print(f"Loading preprocessing objects from: {preprocessing_path}")
    preprocessing = joblib.load(preprocessing_path)
    
    print("✓ All models loaded successfully!\n")
    
    return lstm_model, rf_model, meta_learner, preprocessing


def preprocess_static_data(df_static, preprocessing):
    """
    Preprocess static student data.
    
    Parameters
    ----------
    df_static : pd.DataFrame
        Static student data
    preprocessing : dict
        Preprocessing objects (scalers, encoders, etc.)
    
    Returns
    -------
    np.array
        Preprocessed static features
    """
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    
    # Get feature lists from preprocessing
    actual_numerical = preprocessing.get('actual_numerical', [])
    actual_categorical = preprocessing.get('actual_categorical', [])
    actual_binary = preprocessing.get('actual_binary', [])
    static_feature_cols = preprocessing.get('static_feature_cols', [])
    label_encoders = preprocessing.get('label_encoders', {})
    scaler = preprocessing.get('scaler')
    
    df_processed = df_static.copy()
    
    # Handle numerical features
    for col in actual_numerical:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
            median_val = df_processed[col].median()
            df_processed[col].fillna(median_val, inplace=True)
    
    # Handle categorical features
    for col in actual_categorical:
        if col in df_processed.columns:
            df_processed[col].fillna('Unknown', inplace=True)
            # Encode categorical
            if col in label_encoders:
                le = label_encoders[col]
                # Handle unseen labels by mapping them to a known label
                known_labels = set(le.classes_)
                # Get the most common known label as fallback
                fallback_label = le.classes_[0]
                df_processed[col] = df_processed[col].apply(
                    lambda x: str(x) if str(x) in known_labels else fallback_label
                )
                df_processed[col + '_encoded'] = le.transform(df_processed[col].astype(str))
            else:
                # Create new encoder if not in preprocessing
                le = LabelEncoder()
                df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
    
    # Handle binary features
    for col in actual_binary:
        if col in df_processed.columns:
            df_processed[col].fillna(0, inplace=True)
    
    # Scale numerical features
    if scaler is not None and len(actual_numerical) > 0:
        numerical_cols_present = [c for c in actual_numerical if c in df_processed.columns]
        if numerical_cols_present:
            df_processed[numerical_cols_present] = scaler.transform(df_processed[numerical_cols_present])
    
    # Select features
    available_cols = [c for c in static_feature_cols if c in df_processed.columns]
    X_static = df_processed[available_cols].values if available_cols else df_processed.values
    
    return X_static


def create_temporal_sequences(df_temporal, df_static, sequence_length=32):
    """
    Create temporal sequences aligned with static data.
    
    Parameters
    ----------
    df_temporal : pd.DataFrame
        Temporal student data
    df_static : pd.DataFrame
        Static student data
    sequence_length : int
        Number of weeks in sequence
    
    Returns
    -------
    np.array
        Temporal sequences
    """
    temporal_features = [
        'weekly_engagement', 'weekly_attendance',
        'weekly_assignments_submitted', 'weekly_quiz_attempts'
    ]
    
    # Check which temporal features are available
    available_temporal = [f for f in temporal_features if f in df_temporal.columns]
    if not available_temporal:
        print("Warning: No temporal features found. Using zeros.")
        return np.zeros((len(df_static), sequence_length, len(temporal_features)))
    
    sequences_dict = {}
    
    # Group by student and create sequences
    for student_id, group in df_temporal.groupby('student_id'):
        group = group.sort_values('week_index')
        
        # Get feature values
        feature_data = group[available_temporal].values
        
        # Pad or truncate to sequence_length
        if len(feature_data) < sequence_length:
            padding = np.zeros((sequence_length - len(feature_data), len(available_temporal)))
            feature_data = np.vstack([padding, feature_data])
        elif len(feature_data) > sequence_length:
            feature_data = feature_data[-sequence_length:]
        
        sequences_dict[student_id] = feature_data
    
    # Create sequences array aligned with static data
    sequences = []
    for student_id in df_static['student_id']:
        if student_id in sequences_dict:
            sequences.append(sequences_dict[student_id])
        else:
            sequences.append(np.zeros((sequence_length, len(available_temporal))))
    
    sequences_array = np.array(sequences)
    
    # Normalize temporal features
    for i in range(sequences_array.shape[2]):
        feature_vals = sequences_array[:, :, i].flatten()
        non_zero = feature_vals[feature_vals != 0]
        if len(non_zero) > 0:
            mean_val = non_zero.mean()
            std_val = non_zero.std()
            if std_val > 0:
                mask = sequences_array[:, :, i] != 0
                sequences_array[:, :, i][mask] = (sequences_array[:, :, i][mask] - mean_val) / std_val
    
    return sequences_array


def predict_with_hybrid_model(lstm_model, rf_model, meta_learner, preprocessing,
                               X_static, X_temporal):
    """
    Make predictions using the hybrid model.
    
    Parameters
    ----------
    lstm_model : keras.Model
        Trained LSTM model
    rf_model : sklearn model
        Trained Random Forest model
    meta_learner : sklearn model
        Trained meta-learner
    preprocessing : dict
        Preprocessing objects
    X_static : np.array
        Preprocessed static features
    X_temporal : np.array
        Preprocessed temporal sequences
    
    Returns
    -------
    tuple
        (probabilities, predictions, risk_levels)
    """
    # Get base model predictions
    print("Getting LSTM predictions...")
    lstm_pred = lstm_model.predict(X_temporal, verbose=0).flatten()
    
    print("Getting Random Forest predictions...")
    # Check if RF model is a regressor or classifier
    if hasattr(rf_model, 'predict_proba'):
        rf_pred = rf_model.predict_proba(X_static)[:, 1]
    else:
        # It's a regressor, use predict directly
        rf_pred = rf_model.predict(X_static)
        # Clip to [0, 1] range for probability-like output
        rf_pred = np.clip(rf_pred, 0, 1)
    
    # Create meta features
    meta_features = np.column_stack([lstm_pred, rf_pred])
    
    # Get hybrid predictions
    print("Getting hybrid meta-learner predictions...")
    hybrid_proba = meta_learner.predict_proba(meta_features)[:, 1]
    
    threshold = preprocessing.get('threshold', 0.5)
    hybrid_pred = (hybrid_proba > threshold).astype(int)
    
    # Determine risk levels
    risk_levels = []
    for prob in hybrid_proba:
        if prob < 0.33:
            risk_levels.append('Low Risk')
        elif prob < 0.66:
            risk_levels.append('Medium Risk')
        else:
            risk_levels.append('High Risk')
    
    return hybrid_proba, hybrid_pred, risk_levels, lstm_pred, rf_pred


def test_model(static_data_path, temporal_data_path=None, output_path=None,
               lstm_path=None, rf_path=None, meta_learner_path=None, preprocessing_path=None):
    """
    Test the hybrid model with a dataset.
    
    Parameters
    ----------
    static_data_path : str
        Path to static student data CSV
    temporal_data_path : str, optional
        Path to temporal student data CSV
    output_path : str, optional
        Path for output results
    lstm_path, rf_path, meta_learner_path, preprocessing_path : str, optional
        Paths to model files
    
    Returns
    -------
    pd.DataFrame
        Results dataframe with predictions
    """
    print("=" * 60)
    print("HYBRID MODEL TESTING")
    print("=" * 60)
    
    # Load models
    lstm_model, rf_model, meta_learner, preprocessing = load_models(
        lstm_path, rf_path, meta_learner_path, preprocessing_path
    )
    
    # Load data
    print(f"Loading static data from: {static_data_path}")
    df_static = pd.read_csv(static_data_path)
    print(f"  Total students: {len(df_static)}")
    
    # Load temporal data if provided
    if temporal_data_path and os.path.exists(temporal_data_path):
        print(f"Loading temporal data from: {temporal_data_path}")
        df_temporal = pd.read_csv(temporal_data_path)
        print(f"  Total temporal records: {len(df_temporal)}")
    else:
        print("No temporal data provided. Using zeros for temporal features.")
        df_temporal = pd.DataFrame()
    
    # Preprocess data
    print("\nPreprocessing data...")
    X_static = preprocess_static_data(df_static, preprocessing)
    
    if not df_temporal.empty:
        X_temporal = create_temporal_sequences(df_temporal, df_static)
    else:
        sequence_length = preprocessing.get('sequence_length', 32) or 32
        X_temporal = np.zeros((len(df_static), sequence_length, 4))
    
    print(f"  Static features shape: {X_static.shape}")
    print(f"  Temporal sequences shape: {X_temporal.shape}")
    
    # Make predictions
    print("\nMaking predictions...")
    hybrid_proba, hybrid_pred, risk_levels, lstm_pred, rf_pred = predict_with_hybrid_model(
        lstm_model, rf_model, meta_learner, preprocessing, X_static, X_temporal
    )
    
    # Create results dataframe
    results = df_static.copy()
    results['predicted_success_proba'] = hybrid_proba
    results['predicted_risk_level'] = risk_levels
    results['predicted_at_risk'] = hybrid_pred
    results['lstm_contribution'] = lstm_pred
    results['rf_contribution'] = rf_pred
    
    # Add success label
    results['success_label'] = results['predicted_success_proba'].apply(
        lambda x: 'Success' if x > 0.5 else 'At Risk'
    )
    
    # Print summary
    print("\n" + "=" * 60)
    print("PREDICTION RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nTotal students analyzed: {len(results)}")
    print(f"\nSuccess Probability Statistics:")
    print(f"  Mean:   {results['predicted_success_proba'].mean():.4f}")
    print(f"  Median: {results['predicted_success_proba'].median():.4f}")
    print(f"  Std:    {results['predicted_success_proba'].std():.4f}")
    print(f"  Min:    {results['predicted_success_proba'].min():.4f}")
    print(f"  Max:    {results['predicted_success_proba'].max():.4f}")
    
    print(f"\nRisk Level Distribution:")
    risk_counts = results['predicted_risk_level'].value_counts()
    for level, count in risk_counts.items():
        pct = count / len(results) * 100
        print(f"  {level}: {count} ({pct:.1f}%)")
    
    print(f"\nPrediction Distribution:")
    pred_counts = results['success_label'].value_counts()
    for label, count in pred_counts.items():
        pct = count / len(results) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    
    # Save results
    if output_path:
        results.to_csv(output_path, index=False)
        print(f"\n✓ Results saved to: {output_path}")
    else:
        default_output = "results/prediction_results.csv"
        os.makedirs("results", exist_ok=True)
        results.to_csv(default_output, index=False)
        print(f"\n✓ Results saved to: {default_output}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TESTING COMPLETE")
    print("=" * 60)
    
    return results


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Test the Hybrid Model with a Dataset'
    )
    parser.add_argument(
        '--static_data', '-s',
        required=True,
        help='Path to static student data CSV file'
    )
    parser.add_argument(
        '--temporal_data', '-t',
        default=None,
        help='Path to temporal student data CSV file (optional)'
    )
    parser.add_argument(
        '--output', '-o',
        default=None,
        help='Path for output results CSV file'
    )
    parser.add_argument(
        '--lstm_path',
        default=None,
        help='Path to LSTM model file'
    )
    parser.add_argument(
        '--rf_path',
        default=None,
        help='Path to Random Forest model file'
    )
    parser.add_argument(
        '--meta_learner_path',
        default=None,
        help='Path to Meta-learner model file'
    )
    parser.add_argument(
        '--preprocessing_path',
        default=None,
        help='Path to preprocessing objects file'
    )
    
    args = parser.parse_args()
    
    # Run model testing
    results = test_model(
        static_data_path=args.static_data,
        temporal_data_path=args.temporal_data,
        output_path=args.output,
        lstm_path=args.lstm_path,
        rf_path=args.rf_path,
        meta_learner_path=args.meta_learner_path,
        preprocessing_path=args.preprocessing_path
    )
    
    return results


if __name__ == '__main__':
    main()
