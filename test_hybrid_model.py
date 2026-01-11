#!/usr/bin/env python3
"""
Comprehensive Test Script for Hybrid Model
==========================================
Tests the hybrid LSTM + RF/GB model with meta-learner

Usage:
    python test_hybrid_model.py --timestamp 20260111_023557
    python test_hybrid_model.py --timestamp 20260111_023557 --static uploads/synthetic_static_20260111_023313.csv --temporal uploads/synthetic_temporal_20260111_023313.csv
    python test_hybrid_model.py --generate  # Generate new test data
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
import pickle
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(text.center(70))
    print("=" * 70)


def print_subheader(text):
    """Print formatted subheader"""
    print("\n" + "-" * 70)
    print(text)
    print("-" * 70)


def load_models(timestamp):
    """Load all model components"""
    print_subheader("Step 1: Loading Models")

    models = {}
    try:
        # Load LSTM model
        lstm_path = f'results/lstm_model_{timestamp}.h5'
        print(f"Loading LSTM from: {lstm_path}")
        models['lstm'] = keras.models.load_model(lstm_path)
        print("  ✓ LSTM model loaded")

        # Load RF model
        rf_path = f'results/rf_model_{timestamp}.pkl'
        print(f"Loading Random Forest from: {rf_path}")
        models['rf'] = joblib.load(rf_path)
        print("  ✓ Random Forest loaded")

        # Load GB model
        gb_path = f'results/gb_model_{timestamp}.pkl'
        print(f"Loading Gradient Boosting from: {gb_path}")
        models['gb'] = joblib.load(gb_path)
        print("  ✓ Gradient Boosting loaded")

        # Load meta-learner
        meta_path = f'results/meta_learner_{timestamp}.pkl'
        print(f"Loading Meta-learner from: {meta_path}")
        models['meta'] = joblib.load(meta_path)
        print("  ✓ Meta-learner loaded")

        # Load preprocessing objects
        prep_path = f'results/preprocessing_objects_{timestamp}.pkl'
        print(f"Loading preprocessing objects from: {prep_path}")
        models['preprocessing'] = joblib.load(prep_path)
        print("  ✓ Preprocessing objects loaded")

        print("\n✓ All models loaded successfully!")
        return models

    except FileNotFoundError as e:
        print(f"\n✗ Error: Model file not found - {e}")
        print("\nAvailable model timestamps:")
        list_available_models()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error loading models: {e}")
        sys.exit(1)


def list_available_models():
    """List available model timestamps in results directory"""
    if not os.path.exists('results'):
        print("  No results directory found")
        return

    files = os.listdir('results')
    timestamps = set()
    for f in files:
        if f.startswith('lstm_model_') and f.endswith('.h5'):
            timestamp = f.replace('lstm_model_', '').replace('.h5', '')
            timestamps.add(timestamp)

    if timestamps:
        for ts in sorted(timestamps, reverse=True):
            print(f"  - {ts}")
    else:
        print("  No trained models found in results/")


def load_data(static_path, temporal_path):
    """Load test data"""
    print_subheader("Step 2: Loading Test Data")

    try:
        df_static = pd.read_csv(static_path)
        df_temporal = pd.read_csv(temporal_path)

        print(f"✓ Loaded static data: {len(df_static)} students")
        print(f"  Features: {list(df_static.columns[:5])}... ({len(df_static.columns)} total)")

        print(f"✓ Loaded temporal data: {len(df_temporal)} records")
        print(f"  Features: {list(df_temporal.columns[:5])}... ({len(df_temporal.columns)} total)")

        return df_static, df_temporal

    except FileNotFoundError as e:
        print(f"\n✗ Error: Data file not found - {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df_static, df_temporal, preprocessing_objects):
    """Preprocess test data"""
    print_subheader("Step 3: Preprocessing Test Data")

    try:
        # Extract preprocessing objects
        scaler_static = preprocessing_objects['scaler_static']
        scaler_temporal = preprocessing_objects['scaler_temporal']
        label_encoders = preprocessing_objects.get('label_encoders', {})

        # Separate features and target
        feature_cols = [col for col in df_static.columns
                       if col not in ['student_id', 'success_label', 'risk_level']]
        X_static = df_static[feature_cols].copy()
        y = df_static['success_label'].values

        print(f"Static features: {X_static.shape}")
        print(f"Target: {y.shape} (Success rate: {y.mean():.1%})")

        # Encode categorical features
        categorical_cols = X_static.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding {len(categorical_cols)} categorical columns...")
            for col in categorical_cols:
                if col in label_encoders:
                    try:
                        X_static[col] = label_encoders[col].transform(X_static[col])
                    except ValueError:
                        # Handle unseen categories
                        print(f"  ⚠ Warning: New categories in '{col}', creating new encoder")
                        le = LabelEncoder()
                        X_static[col] = le.fit_transform(X_static[col].astype(str))
                else:
                    le = LabelEncoder()
                    X_static[col] = le.fit_transform(X_static[col].astype(str))

        # Scale static features
        X_static_scaled = scaler_static.transform(X_static)
        print(f"✓ Static features scaled: {X_static_scaled.shape}")

        # Prepare temporal sequences
        print("Creating temporal sequences...")
        temporal_feature_cols = ['weekly_engagement', 'weekly_attendance',
                                'weekly_assignments_submitted', 'weekly_quiz_attempts']

        temporal_sequences = []
        for student_id in df_static['student_id']:
            student_temporal = df_temporal[df_temporal['student_id'] == student_id]
            temporal_features = student_temporal[temporal_feature_cols].values
            temporal_sequences.append(temporal_features)

        X_temporal = np.array(temporal_sequences)
        print(f"Temporal sequences created: {X_temporal.shape}")

        # Scale temporal data
        original_shape = X_temporal.shape
        X_temporal_reshaped = X_temporal.reshape(-1, X_temporal.shape[-1])
        X_temporal_scaled_reshaped = scaler_temporal.transform(X_temporal_reshaped)
        X_temporal_scaled = X_temporal_scaled_reshaped.reshape(original_shape)
        print(f"✓ Temporal sequences scaled: {X_temporal_scaled.shape}")

        print("\n✓ Preprocessing complete!")
        return X_static_scaled, X_temporal_scaled, y

    except Exception as e:
        print(f"\n✗ Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_predictions(models, X_static, X_temporal):
    """Generate predictions from all models"""
    print_subheader("Step 4: Generating Predictions")

    try:
        # LSTM predictions
        print("Generating LSTM predictions...")
        lstm_pred_proba = models['lstm'].predict(X_temporal, verbose=0).flatten()
        print(f"  ✓ LSTM predictions: {lstm_pred_proba.shape}")

        # Random Forest predictions
        print("Generating Random Forest predictions...")
        rf_pred_proba = models['rf'].predict_proba(X_static)[:, 1]
        print(f"  ✓ RF predictions: {rf_pred_proba.shape}")

        # Gradient Boosting predictions
        print("Generating Gradient Boosting predictions...")
        gb_pred_proba = models['gb'].predict_proba(X_static)[:, 1]
        print(f"  ✓ GB predictions: {gb_pred_proba.shape}")

        # Stack predictions for meta-learner
        print("Combining predictions with meta-learner...")
        meta_features = np.column_stack([lstm_pred_proba, rf_pred_proba, gb_pred_proba])
        print(f"  Meta-features shape: {meta_features.shape}")

        # Final hybrid predictions
        hybrid_pred_proba = models['meta'].predict_proba(meta_features)[:, 1]
        hybrid_pred = (hybrid_pred_proba >= 0.5).astype(int)
        print(f"  ✓ Hybrid predictions: {hybrid_pred.shape}")

        print("\n✓ All predictions generated!")

        return {
            'lstm_proba': lstm_pred_proba,
            'rf_proba': rf_pred_proba,
            'gb_proba': gb_pred_proba,
            'hybrid_proba': hybrid_pred_proba,
            'hybrid_pred': hybrid_pred
        }

    except Exception as e:
        print(f"\n✗ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def evaluate_models(y_true, predictions):
    """Evaluate all models"""
    print_header("MODEL EVALUATION RESULTS")

    results = {}

    # Evaluate each model
    models_to_eval = ['lstm', 'rf', 'gb', 'hybrid']
    model_names = ['LSTM', 'Random Forest', 'Gradient Boost', 'HYBRID']

    for model_key, model_name in zip(models_to_eval, model_names):
        proba_key = f"{model_key}_proba"
        pred = (predictions[proba_key] >= 0.5).astype(int)

        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)
        auc = roc_auc_score(y_true, predictions[proba_key])

        results[model_name] = {
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'auc': auc
        }

    # Print comparison table
    print_subheader("Performance Comparison")
    print(f"\n{'Model':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print("-" * 88)

    for model_name in model_names:
        r = results[model_name]
        highlight = ">>> " if model_name == "HYBRID" else "    "
        print(f"{highlight}{model_name:<14} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
              f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}")

    # Detailed hybrid results
    print_subheader("Detailed HYBRID Model Results")

    cm = confusion_matrix(y_true, predictions['hybrid_pred'])
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted")
    print(f"                  Fail    Success")
    print(f"Actual  Fail    [{cm[0,0]:4d}]   [{cm[0,1]:4d}]")
    print(f"        Success [{cm[1,0]:4d}]   [{cm[1,1]:4d}]")

    print(f"\nMetric Breakdown:")
    print(f"  True Negatives:  {cm[0,0]:4d} (Correctly predicted failures)")
    print(f"  False Positives: {cm[0,1]:4d} (Incorrectly predicted success)")
    print(f"  False Negatives: {cm[1,0]:4d} (Missed at-risk students)")
    print(f"  True Positives:  {cm[1,1]:4d} (Correctly predicted success)")

    print("\nClassification Report:")
    print(classification_report(y_true, predictions['hybrid_pred'],
                               target_names=['Failure', 'Success']))

    return results


def threshold_analysis(y_true, pred_proba):
    """Analyze performance at different thresholds"""
    print_header("THRESHOLD SENSITIVITY ANALYSIS")

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    print(f"\n{'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 60)

    for thresh in thresholds:
        pred = (pred_proba >= thresh).astype(int)
        acc = accuracy_score(y_true, pred)
        prec = precision_score(y_true, pred, zero_division=0)
        rec = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        highlight = " *" if thresh == 0.5 else "  "
        print(f"{thresh:<12.1f} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}{highlight}")

    print("\n* Current threshold used in model")


def save_visualizations(y_true, predictions, timestamp):
    """Create and save visualization plots"""
    print_subheader("Creating Visualizations")

    try:
        # 1. ROC Curves
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        models = [('LSTM', predictions['lstm_proba']),
                 ('Random Forest', predictions['rf_proba']),
                 ('Gradient Boost', predictions['gb_proba']),
                 ('HYBRID', predictions['hybrid_proba'])]

        for name, proba in models:
            fpr, tpr, _ = roc_curve(y_true, proba)
            auc = roc_auc_score(y_true, proba)
            linewidth = 3 if name == 'HYBRID' else 2
            ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})', linewidth=linewidth)

        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        roc_path = f'test_results_roc_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(roc_path, dpi=150)
        print(f"  ✓ ROC curve saved: {roc_path}")
        plt.close()

        # 2. Confusion Matrix
        cm = confusion_matrix(y_true, predictions['hybrid_pred'])
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Failure', 'Success'],
                   yticklabels=['Failure', 'Success'])
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix - Hybrid Model', fontsize=14, fontweight='bold')

        cm_path = f'test_results_confusion_matrix_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(cm_path, dpi=150)
        print(f"  ✓ Confusion matrix saved: {cm_path}")
        plt.close()

        # 3. Prediction distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        models = [('LSTM', predictions['lstm_proba']),
                 ('Random Forest', predictions['rf_proba']),
                 ('Gradient Boost', predictions['gb_proba']),
                 ('HYBRID', predictions['hybrid_proba'])]

        for idx, (name, proba) in enumerate(models):
            ax = axes[idx]
            ax.hist(proba[y_true == 0], bins=30, alpha=0.6, label='Failure', color='red')
            ax.hist(proba[y_true == 1], bins=30, alpha=0.6, label='Success', color='green')
            ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
            ax.set_xlabel('Predicted Probability', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(f'{name}', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        dist_path = f'test_results_distributions_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(dist_path, dpi=150)
        print(f"  ✓ Prediction distributions saved: {dist_path}")
        plt.close()

        print("\n✓ All visualizations created!")

    except Exception as e:
        print(f"  ⚠ Warning: Could not create visualizations: {e}")


def save_test_report(results, timestamp, static_path, temporal_path):
    """Save test results to a text file"""
    print_subheader("Saving Test Report")

    try:
        report_path = f'test_results_report_{timestamp}.txt'

        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("HYBRID MODEL TEST REPORT\n")
            f.write("="*70 + "\n\n")

            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Timestamp: {timestamp}\n")
            f.write(f"Static Data: {static_path}\n")
            f.write(f"Temporal Data: {temporal_path}\n\n")

            f.write("-"*70 + "\n")
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-"*70 + "\n\n")

            f.write(f"{'Model':<18} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<12}\n")
            f.write("-" * 88 + "\n")

            for model_name in ['LSTM', 'Random Forest', 'Gradient Boost', 'HYBRID']:
                r = results[model_name]
                f.write(f"{model_name:<18} {r['accuracy']:<12.4f} {r['precision']:<12.4f} "
                       f"{r['recall']:<12.4f} {r['f1']:<12.4f} {r['auc']:<12.4f}\n")

            f.write("\n" + "="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")

        print(f"  ✓ Test report saved: {report_path}")

    except Exception as e:
        print(f"  ⚠ Warning: Could not save report: {e}")


def main():
    parser = argparse.ArgumentParser(description='Test Hybrid Model')
    parser.add_argument('--timestamp', type=str, help='Model timestamp (e.g., 20260111_023557)')
    parser.add_argument('--static', type=str, help='Path to static CSV file')
    parser.add_argument('--temporal', type=str, help='Path to temporal CSV file')
    parser.add_argument('--generate', action='store_true', help='Generate new test data')
    parser.add_argument('--list', action='store_true', help='List available models')

    args = parser.parse_args()

    # List models and exit
    if args.list:
        print_header("AVAILABLE MODELS")
        list_available_models()
        return

    # Check timestamp
    if not args.timestamp:
        print("\n✗ Error: --timestamp is required")
        print("\nAvailable models:")
        list_available_models()
        print("\nUsage: python test_hybrid_model.py --timestamp 20260111_023557")
        sys.exit(1)

    timestamp = args.timestamp

    # Generate test data if requested
    if args.generate:
        print_header("GENERATING NEW TEST DATA")
        from generate_synthetic_data import generate_synthetic_student_data
        static_path, temporal_path, _, _ = generate_synthetic_student_data(
            n_students=500,
            n_weeks=32,
            output_dir='./uploads'
        )
    else:
        # Use provided paths or default
        if not args.static or not args.temporal:
            print("\n✗ Error: --static and --temporal paths required (or use --generate)")
            sys.exit(1)
        static_path = args.static
        temporal_path = args.temporal

    # Main testing workflow
    print_header("HYBRID MODEL TESTING")
    print(f"\nModel: {timestamp}")
    print(f"Static Data: {static_path}")
    print(f"Temporal Data: {temporal_path}")

    # Step 1: Load models
    models = load_models(timestamp)

    # Step 2: Load data
    df_static, df_temporal = load_data(static_path, temporal_path)

    # Step 3: Preprocess
    X_static, X_temporal, y_true = preprocess_data(
        df_static, df_temporal, models['preprocessing']
    )

    # Step 4: Generate predictions
    predictions = generate_predictions(models, X_static, X_temporal)

    # Step 5: Evaluate
    results = evaluate_models(y_true, predictions)

    # Step 6: Threshold analysis
    threshold_analysis(y_true, predictions['hybrid_proba'])

    # Step 7: Save visualizations
    save_visualizations(y_true, predictions, timestamp)

    # Step 8: Save report
    save_test_report(results, timestamp, static_path, temporal_path)

    # Final summary
    print_header("TEST COMPLETED SUCCESSFULLY")
    print(f"\nHybrid Model Performance:")
    print(f"  Accuracy:  {results['HYBRID']['accuracy']:.4f}")
    print(f"  F1-Score:  {results['HYBRID']['f1']:.4f}")
    print(f"  ROC-AUC:   {results['HYBRID']['auc']:.4f}")
    print(f"\nResults saved with timestamp: {timestamp}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    main()
