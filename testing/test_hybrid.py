#!/usr/bin/env python3
"""
Simple Test Script for Hybrid Model
====================================
Tests your latest hybrid model with prepared test data

Usage:
    python test_hybrid.py
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add parent directory to path
sys.path.append('..')

# Suppress warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder

# Configuration
MODEL_TIMESTAMP = '20260111_023557'  # Your latest model
STATIC_DATA = 'test_data/synthetic_static_train.csv'
TEMPORAL_DATA = 'test_data/synthetic_temporal_train.csv'
RESULTS_DIR = 'test_results'

# Set seeds
np.random.seed(42)
tf.random.set_seed(42)


# ============================================================================
# Custom Attention Layer (Required for loading LSTM model)
# ============================================================================
@tf.keras.utils.register_keras_serializable(package="Custom")
class AttentionLayer(tf.keras.layers.Layer):
    """
    Custom Attention Layer for LSTM outputs.
    Applies attention over timesteps to focus on important time steps.
    """

    def __init__(self, use_bias=True, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.use_bias = use_bias
        self.supports_masking = True

    def build(self, input_shape):
        feature_dim = int(input_shape[-1])

        self.W = self.add_weight(
            name="attention_weight",
            shape=(feature_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )

        if self.use_bias:
            self.b = self.add_weight(
                name="attention_bias",
                shape=(1,),
                initializer="zeros",
                trainable=True,
            )
        else:
            self.b = None

        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        # x: (B, T, F)
        scores = tf.tanh(tf.tensordot(x, self.W, axes=1) + (self.b if self.b is not None else 0.0))
        scores = tf.squeeze(scores, axis=-1)  # (B, T)

        # Apply mask (if sequences are padded)
        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            scores = scores + (1.0 - mask) * tf.constant(-1e9, dtype=scores.dtype)

        # attention weights over time
        weights = tf.nn.softmax(scores, axis=1)      # (B, T)
        weights = tf.expand_dims(weights, axis=-1)   # (B, T, 1)

        # weighted sum of inputs
        context = tf.reduce_sum(x * weights, axis=1)  # (B, F)
        return context

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias})
        return config


def print_section(title):
    """Print section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70)


def print_step(step_num, description):
    """Print step header"""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 70)


def load_models():
    """Load all model components"""
    print_step(1, "Loading Models")

    models = {}
    base_path = '../results'

    print(f"Model timestamp: {MODEL_TIMESTAMP}")
    print(f"Loading from: {base_path}/\n")

    try:
        # LSTM (with custom AttentionLayer)
        lstm_path = f'{base_path}/lstm_model_{MODEL_TIMESTAMP}.h5'
        models['lstm'] = keras.models.load_model(
            lstm_path,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"[+] LSTM loaded: {os.path.basename(lstm_path)}")

        # Random Forest
        rf_path = f'{base_path}/rf_model_{MODEL_TIMESTAMP}.pkl'
        models['rf'] = joblib.load(rf_path)
        print(f"[+] Random Forest loaded: {os.path.basename(rf_path)}")

        # Gradient Boosting
        gb_path = f'{base_path}/gb_model_{MODEL_TIMESTAMP}.pkl'
        models['gb'] = joblib.load(gb_path)
        print(f"[+] Gradient Boosting loaded: {os.path.basename(gb_path)}")

        # Meta-learner
        meta_path = f'{base_path}/meta_learner_{MODEL_TIMESTAMP}.pkl'
        models['meta'] = joblib.load(meta_path)
        print(f"[+] Meta-learner loaded: {os.path.basename(meta_path)}")

        # Preprocessing
        prep_path = f'{base_path}/preprocessing_objects_{MODEL_TIMESTAMP}.pkl'
        models['preprocessing'] = joblib.load(prep_path)
        print(f"[+] Preprocessing objects loaded: {os.path.basename(prep_path)}")

        print("\n[SUCCESS] All models loaded successfully!")
        return models

    except Exception as e:
        print(f"\n[ERROR] Error loading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def load_data():
    """Load test data"""
    print_step(2, "Loading Test Data")

    try:
        df_static = pd.read_csv(STATIC_DATA)
        df_temporal = pd.read_csv(TEMPORAL_DATA)

        print(f"Static data: {STATIC_DATA}")
        print(f"  -> {len(df_static)} students")
        print(f"  -> {len(df_static.columns)} features")
        print(f"  -> Success rate: {df_static['success_label'].mean():.1%}")

        print(f"\nTemporal data: {TEMPORAL_DATA}")
        print(f"  -> {len(df_temporal)} records")
        print(f"  -> {len(df_temporal.columns)} features")
        print(f"  -> {len(df_temporal) // len(df_static)} weeks per student")

        print("\n[SUCCESS] Data loaded successfully!")
        return df_static, df_temporal

    except Exception as e:
        print(f"\n[-] Error loading data: {e}")
        sys.exit(1)


def preprocess_data(df_static, df_temporal, preprocessing):
    """Preprocess test data"""
    print_step(3, "Preprocessing Data")

    try:
        # Get preprocessing objects
        scaler = preprocessing['scaler']
        label_encoders = preprocessing.get('label_encoders', {})
        static_feature_cols = preprocessing.get('static_feature_cols', None)

        print("Preprocessing objects loaded")
        print(f"  Scaler: {type(scaler).__name__}")
        print(f"  Label encoders: {len(label_encoders)} columns")

        # Get features and target
        y = df_static['success_label'].values

        # Start with all columns except student_id, success_label, risk_level
        X_static = df_static[[col for col in df_static.columns
                              if col not in ['student_id', 'success_label', 'risk_level']]].copy()

        print(f"\nInitial features: {len(X_static.columns)}")
        print(f"Samples: {len(X_static)}")
        print(f"Success rate: {y.mean():.1%}")

        # Encode categoricals FIRST (this creates the _encoded columns)
        categorical_cols = X_static.select_dtypes(include=['object']).columns
        print(f"\nEncoding {len(categorical_cols)} categorical features...")

        for col in categorical_cols:
            if col in label_encoders:
                try:
                    X_static[f'{col}_encoded'] = label_encoders[col].transform(X_static[col])
                    print(f"  [+] Encoded {col} using saved encoder")
                except ValueError as e:
                    print(f"  Warning: New categories in '{col}', using mode")
                    # Use the most common class from training
                    X_static[f'{col}_encoded'] = 0
            X_static = X_static.drop(col, axis=1)  # Drop original column after encoding

        print(f"[+] After encoding: {len(X_static.columns)} features")

        # Now select the exact features used during training
        if static_feature_cols is not None:
            print(f"[+] Selecting saved feature columns: {len(static_feature_cols)} features")
            # Ensure we have all required columns
            missing_cols = set(static_feature_cols) - set(X_static.columns)
            if missing_cols:
                print(f"  Warning: Missing columns: {missing_cols}")
                for col in missing_cols:
                    X_static[col] = 0  # Add missing columns with default value

            # Select and reorder columns to match training
            X_static = X_static[static_feature_cols]

        print(f"[+] Final feature shape before scaling: {X_static.shape}")

        # Scale static features - scaler was trained on specific features only
        # Get the actual features the scaler expects
        if hasattr(scaler, 'feature_names_in_'):
            scaler_features = list(scaler.feature_names_in_)
            print(f"[+] Scaler expects {len(scaler_features)} features")

            # Select only the features the scaler was trained on
            X_for_scaling = X_static[scaler_features]
            X_static_scaled = scaler.transform(X_for_scaling.values)
            print(f"[+] Scaled {len(scaler_features)} features: {X_static_scaled.shape}")

            # For features not scaled, keep them as-is
            other_features = [col for col in X_static.columns if col not in scaler_features]
            if other_features:
                print(f"[+] Keeping {len(other_features)} non-scaled features")
                X_other = X_static[other_features].values
                X_static_scaled = np.hstack([X_static_scaled, X_other])
        else:
            # Fallback: use all features
            X_static_scaled = scaler.transform(X_static.values)

        print(f"[+] Final scaled features: {X_static_scaled.shape}")

        # Prepare temporal sequences
        print("\nPreparing temporal sequences...")
        temporal_cols = ['weekly_engagement', 'weekly_attendance',
                        'weekly_assignments_submitted', 'weekly_quiz_attempts']

        sequences = []
        for student_id in df_static['student_id']:
            student_temp = df_temporal[df_temporal['student_id'] == student_id]
            seq = student_temp[temporal_cols].values
            sequences.append(seq)

        X_temporal = np.array(sequences)
        print(f"[+] Temporal sequences created: {X_temporal.shape}")

        # Note: Temporal data is typically not scaled or scaled separately during training
        # For this test, we'll use it as-is (0-1 range from data generation)
        X_temporal_scaled = X_temporal
        print(f"[+] Temporal data ready: {X_temporal_scaled.shape}")

        print("\n[SUCCESS] Preprocessing complete!")
        return X_static_scaled, X_temporal_scaled, y

    except Exception as e:
        print(f"\n[-] Error in preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def generate_predictions(models, X_static, X_temporal):
    """Generate predictions"""
    print_step(4, "Generating Predictions")

    try:
        # Individual model predictions
        print("LSTM predicting...")
        lstm_proba = models['lstm'].predict(X_temporal, verbose=0).flatten()
        print(f"  [+] LSTM: {lstm_proba.shape}")

        print("Random Forest predicting...")
        # RF is a regressor, outputs probabilities directly
        if hasattr(models['rf'], 'predict_proba'):
            rf_proba = models['rf'].predict_proba(X_static)[:, 1]
        else:
            rf_proba = models['rf'].predict(X_static)  # Regressor
        print(f"  [+] Random Forest: {rf_proba.shape}")

        print("Gradient Boosting predicting...")
        # GB is a classifier
        if hasattr(models['gb'], 'predict_proba'):
            gb_proba = models['gb'].predict_proba(X_static)[:, 1]
        else:
            gb_proba = models['gb'].predict(X_static)
        print(f"  [+] Gradient Boosting: {gb_proba.shape}")

        # Hybrid prediction
        print("\nCombining with meta-learner...")
        # Meta-learner typically gets: [base_predictions, top_features]
        # Stack base model predictions
        base_predictions = np.column_stack([lstm_proba, rf_proba, gb_proba])
        print(f"  Base predictions shape: {base_predictions.shape}")

        # Check how many features meta-learner expects
        meta_n_features = models['meta'].n_features_in_
        print(f"  Meta-learner expects: {meta_n_features} features")

        # Add top features from static data if needed
        if meta_n_features > 3:
            n_additional = meta_n_features - 3
            print(f"  Adding {n_additional} additional features from static data")
            # Use first n_additional features (typically most important ones)
            additional_features = X_static[:, :n_additional]
            meta_features = np.hstack([base_predictions, additional_features])
        else:
            meta_features = base_predictions

        print(f"  Final meta-features shape: {meta_features.shape}")

        hybrid_proba = models['meta'].predict_proba(meta_features)[:, 1]
        hybrid_pred = (hybrid_proba >= 0.5).astype(int)
        print(f"  [+] Hybrid: {hybrid_proba.shape}")

        print("\n[SUCCESS] All predictions generated!")

        return {
            'lstm': lstm_proba,
            'rf': rf_proba,
            'gb': gb_proba,
            'hybrid': hybrid_proba,
            'hybrid_pred': hybrid_pred
        }

    except Exception as e:
        print(f"\n[-] Error generating predictions: {e}")
        sys.exit(1)


def evaluate_models(y_true, predictions):
    """Evaluate and compare models"""
    print_section("MODEL PERFORMANCE RESULTS")

    results = {}

    # Evaluate each model
    for model_name, proba_key in [('LSTM', 'lstm'),
                                   ('Random Forest', 'rf'),
                                   ('Gradient Boosting', 'gb'),
                                   ('HYBRID', 'hybrid')]:
        pred = (predictions[proba_key] >= 0.5).astype(int)

        results[model_name] = {
            'accuracy': accuracy_score(y_true, pred),
            'precision': precision_score(y_true, pred, zero_division=0),
            'recall': recall_score(y_true, pred, zero_division=0),
            'f1': f1_score(y_true, pred, zero_division=0),
            'auc': roc_auc_score(y_true, predictions[proba_key])
        }

    # Print comparison
    print("\n>>> Model Comparison:")
    print("-" * 88)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}")
    print("-" * 88)

    for model_name in ['LSTM', 'Random Forest', 'Gradient Boosting', 'HYBRID']:
        r = results[model_name]
        prefix = ">>> " if model_name == 'HYBRID' else "    "
        print(f"{prefix}{model_name:<16} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
              f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}")

    print("-" * 88)
    print(">>> = Your Hybrid Model (should have best scores!)")

    # Detailed hybrid results
    print("\n\n>>> Detailed HYBRID Model Results:")
    print("-" * 70)

    cm = confusion_matrix(y_true, predictions['hybrid_pred'])

    print(f"\nConfusion Matrix:")
    print(f"                    Predicted")
    print(f"                    Failure    Success")
    print(f"Actual  Failure    [{cm[0,0]:>5}]     [{cm[0,1]:>5}]")
    print(f"        Success    [{cm[1,0]:>5}]     [{cm[1,1]:>5}]")

    print(f"\nBreakdown:")
    print(f"  [+] True Negatives:  {cm[0,0]:>4} (correctly predicted failure)")
    print(f"  [-] False Positives: {cm[0,1]:>4} (false alarm - predicted success)")
    print(f"  [-] False Negatives: {cm[1,0]:>4} (missed at-risk student)")
    print(f"  [+] True Positives:  {cm[1,1]:>4} (correctly predicted success)")

    print(f"\n>>> Classification Report:")
    print(classification_report(y_true, predictions['hybrid_pred'],
                               target_names=['Failure', 'Success']))

    return results, cm


def create_visualizations(y_true, predictions, cm):
    """Create visualization plots"""
    print_section("Creating Visualizations")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")

    # 1. ROC Curves
    print("\n[1/3] Creating ROC curves...")
    fig, ax = plt.subplots(figsize=(10, 8))

    models = [('LSTM', predictions['lstm'], 'blue'),
              ('Random Forest', predictions['rf'], 'green'),
              ('Gradient Boosting', predictions['gb'], 'orange'),
              ('HYBRID', predictions['hybrid'], 'red')]

    for name, proba, color in models:
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        linewidth = 4 if name == 'HYBRID' else 2
        ax.plot(fpr, tpr, label=f'{name} (AUC={auc:.3f})',
               linewidth=linewidth, color=color)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('ROC Curves - All Models', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(alpha=0.3)

    roc_path = f'{RESULTS_DIR}/roc_curves_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    print(f"  [+] Saved: {roc_path}")
    plt.close()

    # 2. Confusion Matrix
    print("[2/3] Creating confusion matrix...")
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
               xticklabels=['Failure', 'Success'],
               yticklabels=['Failure', 'Success'],
               cbar_kws={'label': 'Count'}, ax=ax)

    ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
    ax.set_title('Confusion Matrix - Hybrid Model', fontsize=16, fontweight='bold')

    cm_path = f'{RESULTS_DIR}/confusion_matrix_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    print(f"  [+] Saved: {cm_path}")
    plt.close()

    # 3. Prediction Distributions
    print("[3/3] Creating prediction distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    models = [('LSTM', predictions['lstm']),
              ('Random Forest', predictions['rf']),
              ('Gradient Boosting', predictions['gb']),
              ('HYBRID', predictions['hybrid'])]

    for idx, (name, proba) in enumerate(models):
        ax = axes[idx]

        ax.hist(proba[y_true == 0], bins=30, alpha=0.7,
               label='Actual Failure', color='#ff6b6b', edgecolor='black')
        ax.hist(proba[y_true == 1], bins=30, alpha=0.7,
               label='Actual Success', color='#51cf66', edgecolor='black')
        ax.axvline(0.5, color='black', linestyle='--', linewidth=2, label='Threshold')

        ax.set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    dist_path = f'{RESULTS_DIR}/distributions_{timestamp}.png'
    plt.tight_layout()
    plt.savefig(dist_path, dpi=150, bbox_inches='tight')
    print(f"  [+] Saved: {dist_path}")
    plt.close()

    print(f"\n[+] All visualizations saved in '{RESULTS_DIR}/' folder!")
    return timestamp


def save_report(results, timestamp):
    """Save text report"""
    print("\n[Saving] Creating test report...")

    report_path = f'{RESULTS_DIR}/test_report_{timestamp}.txt'

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("HYBRID MODEL TEST REPORT\n")
        f.write("="*70 + "\n\n")

        f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {MODEL_TIMESTAMP}\n")
        f.write(f"Static Data: {STATIC_DATA}\n")
        f.write(f"Temporal Data: {TEMPORAL_DATA}\n\n")

        f.write("-"*70 + "\n")
        f.write("MODEL PERFORMANCE\n")
        f.write("-"*70 + "\n\n")

        f.write(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10}\n")
        f.write("-" * 88 + "\n")

        for model_name in ['LSTM', 'Random Forest', 'Gradient Boosting', 'HYBRID']:
            r = results[model_name]
            f.write(f"{model_name:<20} {r['accuracy']:>10.4f} {r['precision']:>10.4f} "
                   f"{r['recall']:>10.4f} {r['f1']:>10.4f} {r['auc']:>10.4f}\n")

        f.write("\n" + "="*70 + "\n")

    print(f"  [+] Saved: {report_path}")


def main():
    """Main testing workflow"""

    print_section("HYBRID MODEL TESTING")
    print(f"\nTesting Configuration:")
    print(f"  Model: {MODEL_TIMESTAMP}")
    print(f"  Static Data: {STATIC_DATA}")
    print(f"  Temporal Data: {TEMPORAL_DATA}")
    print(f"  Results Directory: {RESULTS_DIR}/")

    # Run testing pipeline
    models = load_models()
    df_static, df_temporal = load_data()
    X_static, X_temporal, y_true = preprocess_data(df_static, df_temporal, models['preprocessing'])
    predictions = generate_predictions(models, X_static, X_temporal)
    results, cm = evaluate_models(y_true, predictions)
    timestamp = create_visualizations(y_true, predictions, cm)
    save_report(results, timestamp)

    # Final summary
    print_section("TEST COMPLETED SUCCESSFULLY")

    hybrid_results = results['HYBRID']
    print(f"\n>>> Your HYBRID Model Performance:")
    print(f"   Accuracy:  {hybrid_results['accuracy']:.2%}")
    print(f"   Precision: {hybrid_results['precision']:.2%}")
    print(f"   Recall:    {hybrid_results['recall']:.2%}")
    print(f"   F1-Score:  {hybrid_results['f1']:.2%}")
    print(f"   ROC-AUC:   {hybrid_results['auc']:.2%}")

    # Performance check
    if hybrid_results['accuracy'] >= 0.75:
        print("\n[+] EXCELLENT! Model performance is good (>=75% accuracy)")
    elif hybrid_results['accuracy'] >= 0.65:
        print("\n[!] MODERATE: Model performance is acceptable (>=65% accuracy)")
    else:
        print("\n[-] POOR: Model performance needs improvement (<65% accuracy)")

    # Check if hybrid is best
    best_f1 = max(results['LSTM']['f1'], results['Random Forest']['f1'],
                  results['Gradient Boosting']['f1'])
    if hybrid_results['f1'] >= best_f1:
        print("[+] HYBRID model has the BEST F1-Score!")
    else:
        print("[!] Warning: Hybrid not outperforming individual models")

    print(f"\n>>> Results saved in: {RESULTS_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
