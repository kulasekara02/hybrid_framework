# Hybrid Model Testing Guide

## Overview
This guide provides comprehensive steps to test your hybrid model (LSTM + Random Forest/Gradient Boosting with Meta-Learner) for student success prediction.

---

## 1. Model Architecture

Your hybrid model consists of:
- **LSTM Model**: Captures temporal patterns from weekly student engagement data
- **Tree-based Models**: Random Forest (RF) or Gradient Boosting (GB) for static features
- **Meta-Learner**: Combines predictions from both models for final prediction

---

## 2. Required Files for Testing

### 2.1 Model Files (in `results/` directory)
Choose the latest trained model set (e.g., `20260111_023557`):
```
✓ lstm_model_20260111_023557.h5              # LSTM weights
✓ rf_model_20260111_023557.pkl               # Random Forest model
✓ gb_model_20260111_023557.pkl               # Gradient Boosting model
✓ meta_learner_20260111_023557.pkl           # Meta-learner (stacks predictions)
✓ preprocessing_objects_20260111_023557.pkl  # Scalers, encoders, etc.
✓ hybrid_model_complete_20260111_023557.pkl  # Complete model wrapper
```

### 2.2 Test Data Files
```
✓ Static CSV: uploads/synthetic_static_YYYYMMDD_HHMMSS.csv
✓ Temporal CSV: uploads/synthetic_temporal_YYYYMMDD_HHMMSS.csv
```

### 2.3 Required Python Scripts
```
✓ generate_synthetic_data.py  # Generate new test data if needed
✓ test2.ipynb                  # Main training/testing notebook
```

---

## 3. Testing Steps

### Step 1: Prepare Test Environment

```python
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Set random seed for reproducibility
np.random.seed(42)
```

### Step 2: Load Trained Models

```python
# Specify the timestamp of your trained model
MODEL_TIMESTAMP = '20260111_023557'  # Change to your latest model

# Load all model components
lstm_model = load_model(f'results/lstm_model_{MODEL_TIMESTAMP}.h5')
rf_model = joblib.load(f'results/rf_model_{MODEL_TIMESTAMP}.pkl')
gb_model = joblib.load(f'results/gb_model_{MODEL_TIMESTAMP}.pkl')
meta_learner = joblib.load(f'results/meta_learner_{MODEL_TIMESTAMP}.pkl')
preprocessing_objects = joblib.load(f'results/preprocessing_objects_{MODEL_TIMESTAMP}.pkl')

# Extract preprocessing objects
scaler_static = preprocessing_objects['scaler_static']
scaler_temporal = preprocessing_objects['scaler_temporal']
label_encoders = preprocessing_objects.get('label_encoders', {})

print("✓ All models loaded successfully")
```

### Step 3: Prepare Test Data

#### Option A: Use Existing Test Data
```python
# Load the data you already have
static_path = 'uploads/synthetic_static_20260111_023313.csv'
temporal_path = 'uploads/synthetic_temporal_20260111_023313.csv'

df_static = pd.read_csv(static_path)
df_temporal = pd.read_csv(temporal_path)

print(f"Loaded {len(df_static)} students with {len(df_temporal)} temporal records")
```

#### Option B: Generate Fresh Test Data
```python
from generate_synthetic_data import generate_synthetic_student_data

# Generate new synthetic test data
static_path, temporal_path, df_static, df_temporal = generate_synthetic_student_data(
    n_students=500,  # Use different size for testing
    n_weeks=32,
    output_dir='./uploads'
)

print(f"Generated {len(df_static)} test students")
```

### Step 4: Preprocess Test Data

```python
# Separate features and target
X_static_test = df_static.drop(['student_id', 'success_label', 'risk_level'], axis=1, errors='ignore')
y_test = df_static['success_label']

# Encode categorical features
categorical_cols = X_static_test.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if col in label_encoders:
        X_static_test[col] = label_encoders[col].transform(X_static_test[col])
    else:
        # If new categories, use mode or create new encoder
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        X_static_test[col] = le.fit_transform(X_static_test[col].astype(str))

# Scale static features
X_static_scaled = scaler_static.transform(X_static_test)

# Prepare temporal data (reshape for LSTM)
# Group by student and create sequences
temporal_sequences = []
for student_id in df_static['student_id']:
    student_temporal = df_temporal[df_temporal['student_id'] == student_id]
    temporal_features = student_temporal[['weekly_engagement', 'weekly_attendance',
                                          'weekly_assignments_submitted', 'weekly_quiz_attempts']].values
    temporal_sequences.append(temporal_features)

X_temporal = np.array(temporal_sequences)
# Scale temporal data (reshape for scaler)
original_shape = X_temporal.shape
X_temporal_reshaped = X_temporal.reshape(-1, X_temporal.shape[-1])
X_temporal_scaled_reshaped = scaler_temporal.transform(X_temporal_reshaped)
X_temporal_scaled = X_temporal_scaled_reshaped.reshape(original_shape)

print(f"✓ Test data preprocessed")
print(f"  Static shape: {X_static_scaled.shape}")
print(f"  Temporal shape: {X_temporal_scaled.shape}")
```

### Step 5: Generate Predictions

```python
# Get predictions from each base model
lstm_pred_proba = lstm_model.predict(X_temporal_scaled)
rf_pred_proba = rf_model.predict_proba(X_static_scaled)[:, 1]
gb_pred_proba = gb_model.predict_proba(X_static_scaled)[:, 1]

# Stack predictions for meta-learner
meta_features = np.column_stack([lstm_pred_proba, rf_pred_proba, gb_pred_proba])

# Get final hybrid prediction
hybrid_pred_proba = meta_learner.predict_proba(meta_features)[:, 1]
hybrid_pred = (hybrid_pred_proba >= 0.5).astype(int)

print("✓ Predictions generated")
```

### Step 6: Evaluate Model Performance

```python
# Calculate metrics
accuracy = accuracy_score(y_test, hybrid_pred)
precision = precision_score(y_test, hybrid_pred)
recall = recall_score(y_test, hybrid_pred)
f1 = f1_score(y_test, hybrid_pred)
roc_auc = roc_auc_score(y_test, hybrid_pred_proba)

# Print results
print("\n" + "="*60)
print("HYBRID MODEL TEST RESULTS")
print("="*60)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print("="*60)

# Confusion Matrix
cm = confusion_matrix(y_test, hybrid_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nTrue Negatives:  {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives:  {cm[1,1]}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, hybrid_pred, target_names=['Failure', 'Success']))
```

### Step 7: Compare Individual Model Performance

```python
# Evaluate LSTM alone
lstm_pred = (lstm_pred_proba >= 0.5).astype(int)
lstm_acc = accuracy_score(y_test, lstm_pred)
lstm_f1 = f1_score(y_test, lstm_pred)

# Evaluate RF alone
rf_pred = (rf_pred_proba >= 0.5).astype(int)
rf_acc = accuracy_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

# Evaluate GB alone
gb_pred = (gb_pred_proba >= 0.5).astype(int)
gb_acc = accuracy_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"{'Model':<15} {'Accuracy':<12} {'F1-Score':<12}")
print("-"*60)
print(f"{'LSTM':<15} {lstm_acc:<12.4f} {lstm_f1:<12.4f}")
print(f"{'Random Forest':<15} {rf_acc:<12.4f} {rf_f1:<12.4f}")
print(f"{'Gradient Boost':<15} {gb_acc:<12.4f} {gb_f1:<12.4f}")
print(f"{'HYBRID':<15} {accuracy:<12.4f} {f1:<12.4f}")
print("="*60)
```

---

## 4. Validation Tests

### 4.1 Cross-Validation Test
```python
from sklearn.model_selection import cross_val_score

# Test with cross-validation on meta-learner
cv_scores = cross_val_score(meta_learner, meta_features, y_test, cv=5, scoring='f1')
print(f"\n5-Fold CV F1-Scores: {cv_scores}")
print(f"Mean CV F1-Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
```

### 4.2 Threshold Sensitivity Test
```python
# Test different probability thresholds
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
print("\n" + "="*60)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("="*60)
for thresh in thresholds:
    pred = (hybrid_pred_proba >= thresh).astype(int)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"Threshold {thresh:.1f}: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, F1={f1:.3f}")
```

### 4.3 Feature Importance Test
```python
# Check feature importance from tree models
import matplotlib.pyplot as plt

rf_importance = rf_model.feature_importances_
feature_names = X_static_test.columns

# Plot top 15 important features
indices = np.argsort(rf_importance)[-15:]
plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), rf_importance[indices])
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 15 Important Features (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✓ Feature importance plot saved: feature_importance.png")
```

---

## 5. Expected Results

### 5.1 Performance Benchmarks
Based on your model architecture, expected test performance:
- **Accuracy**: 75-85%
- **F1-Score**: 70-80%
- **ROC-AUC**: 80-90%

### 5.2 Model Comparison
The hybrid model should outperform individual models:
- Hybrid > Individual models by 3-7% in F1-score
- Meta-learner effectively combines strengths

### 5.3 Red Flags
Watch out for:
- **Accuracy < 70%**: Model may need retraining
- **Large train/test gap**: Possible overfitting
- **Recall < 60%**: Missing too many at-risk students
- **Precision < 60%**: Too many false alarms

---

## 6. Testing Checklist

### Pre-Test Checklist
- [ ] All model files (.h5, .pkl) are present in results/
- [ ] Test data files (static + temporal CSV) are available
- [ ] Same timestamp used for all model components
- [ ] Python environment has all dependencies installed
- [ ] Test data has same structure as training data

### During Test Checklist
- [ ] Models load without errors
- [ ] Preprocessing applies correctly
- [ ] Predictions generate without warnings
- [ ] Shapes match expected dimensions
- [ ] No NaN/Inf values in predictions

### Post-Test Checklist
- [ ] All metrics calculated successfully
- [ ] Results documented and saved
- [ ] Comparison with baseline established
- [ ] Visualizations generated (confusion matrix, ROC curve)
- [ ] Model performance meets requirements

---

## 7. Quick Test Script

Create `test_hybrid_model.py`:

```python
#!/usr/bin/env python3
"""Quick test script for hybrid model"""

import sys
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def test_model(model_timestamp, static_path, temporal_path):
    """Test the hybrid model with given data"""

    print(f"Testing model: {model_timestamp}")
    print("="*60)

    # Load models
    lstm = load_model(f'results/lstm_model_{model_timestamp}.h5')
    rf = joblib.load(f'results/rf_model_{model_timestamp}.pkl')
    gb = joblib.load(f'results/gb_model_{model_timestamp}.pkl')
    meta = joblib.load(f'results/meta_learner_{model_timestamp}.pkl')
    prep = joblib.load(f'results/preprocessing_objects_{model_timestamp}.pkl')

    # Load data
    df_static = pd.read_csv(static_path)
    df_temporal = pd.read_csv(temporal_path)

    # TODO: Add preprocessing and prediction logic here
    # (Use code from Step 4-5 above)

    print("✓ Test completed successfully")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python test_hybrid_model.py <timestamp> <static_csv> <temporal_csv>")
        print("Example: python test_hybrid_model.py 20260111_023557 uploads/static.csv uploads/temporal.csv")
        sys.exit(1)

    test_model(sys.argv[1], sys.argv[2], sys.argv[3])
```

Run with:
```bash
python test_hybrid_model.py 20260111_023557 uploads/synthetic_static_20260111_023313.csv uploads/synthetic_temporal_20260111_023313.csv
```

---

## 8. Troubleshooting

### Issue 1: Model Loading Errors
**Error**: `OSError: Unable to open file`
**Solution**: Verify timestamp matches exactly, check file permissions

### Issue 2: Shape Mismatch
**Error**: `ValueError: Input shape mismatch`
**Solution**: Ensure test data has same features as training data

### Issue 3: Encoding Errors
**Error**: `ValueError: y contains previously unseen labels`
**Solution**: Check for new categorical values in test data

### Issue 4: Low Performance
**Error**: Accuracy < 60%
**Solution**: Check if test data distribution matches training data

---

## 9. Advanced Testing

### 9.1 Time-based Validation
Test on data from different time periods to check temporal robustness.

### 9.2 Subgroup Analysis
Evaluate performance across different student subgroups:
- By country
- By study level
- By language proficiency

### 9.3 Explainability Testing
Use SHAP or LIME to explain individual predictions.

---

## 10. Next Steps

After testing:
1. Document all results in a testing report
2. Compare with baseline/previous models
3. Identify areas for improvement
4. Plan model updates if needed
5. Deploy to production if performance is satisfactory

---

**Last Updated**: January 11, 2026
**Model Version**: Hybrid LSTM + RF/GB with Meta-Learner
**Contact**: [Your Email/Contact Info]
