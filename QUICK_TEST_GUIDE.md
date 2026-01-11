# Quick Test Guide - Hybrid Model

## 🚀 Quick Start (3 Steps)

### Step 1: Check Your Model Files
```bash
ls -la results/
```
Find your latest model timestamp (e.g., `20260111_023557`)

### Step 2: Check Your Test Data
```bash
ls uploads/*.csv | tail -4
```
Note the latest static and temporal CSV files

### Step 3: Run the Test
```bash
python test_hybrid_model.py --timestamp 20260111_023557 \
    --static uploads/synthetic_static_20260111_023313.csv \
    --temporal uploads/synthetic_temporal_20260111_023313.csv
```

**OR generate fresh test data:**
```bash
python test_hybrid_model.py --timestamp 20260111_023557 --generate
```

---

## 📋 Pre-Test Checklist

- [ ] **Model files exist** (check `results/` directory)
  - `lstm_model_<timestamp>.h5`
  - `rf_model_<timestamp>.pkl`
  - `gb_model_<timestamp>.pkl`
  - `meta_learner_<timestamp>.pkl`
  - `preprocessing_objects_<timestamp>.pkl`

- [ ] **Test data exists** (check `uploads/` directory)
  - `synthetic_static_<timestamp>.csv`
  - `synthetic_temporal_<timestamp>.csv`

- [ ] **Python environment ready**
  ```bash
  # Check if packages are installed
  python -c "import tensorflow, sklearn, pandas, numpy, matplotlib, seaborn"
  ```

---

## 📁 Required Files

### For Model: `20260111_023557`
```
results/
├── lstm_model_20260111_023557.h5              ✓ LSTM weights
├── rf_model_20260111_023557.pkl               ✓ Random Forest
├── gb_model_20260111_023557.pkl               ✓ Gradient Boosting
├── meta_learner_20260111_023557.pkl           ✓ Meta-learner
├── preprocessing_objects_20260111_023557.pkl  ✓ Preprocessors
└── hybrid_model_complete_20260111_023557.pkl  ✓ Complete model
```

### For Testing:
```
uploads/
├── synthetic_static_<date>.csv    ✓ Student features (1 row per student)
└── synthetic_temporal_<date>.csv  ✓ Weekly data (32 rows per student)
```

---

## 🎯 Test Data Requirements

### Static CSV Must Have:
- `student_id` - Unique identifier
- `success_label` - Target variable (0 or 1)
- Academic features: `gpa_prev`, `entry_gpa`, `gpa_sem1`, `gpa_sem2`
- Demographic features: `age`, `gender`, `language_proficiency`
- Engagement features: `attendance_rate`, `mean_weekly_engagement`
- **~40-45 columns total**

### Temporal CSV Must Have:
- `student_id` - Links to static data
- `week_index` - Week number (1-32)
- `weekly_engagement` - Engagement score (0-1)
- `weekly_attendance` - Attendance rate (0-1)
- `weekly_assignments_submitted` - Count
- `weekly_quiz_attempts` - Count
- **32 rows per student** (32 weeks)

---

## ⚡ Quick Commands

### List Available Models
```bash
python test_hybrid_model.py --list
```

### Generate New Test Data
```bash
python generate_synthetic_data.py
```
This creates new files in `uploads/` with current timestamp.

### Test with Latest Model
```bash
# Find latest model
LATEST=$(ls results/lstm_model_*.h5 | tail -1 | sed 's/.*_\([0-9_]*\)\.h5/\1/')
echo "Latest model: $LATEST"

# Run test
python test_hybrid_model.py --timestamp $LATEST --generate
```

### Test Specific Model
```bash
python test_hybrid_model.py \
    --timestamp 20260111_023557 \
    --static uploads/synthetic_static_20260111_023313.csv \
    --temporal uploads/synthetic_temporal_20260111_023313.csv
```

---

## 📊 Expected Output

After running the test script, you'll get:

### 1. Console Output
```
======================================================================
                     HYBRID MODEL TESTING
======================================================================

Model: 20260111_023557
Static Data: uploads/synthetic_static_20260111_023313.csv
Temporal Data: uploads/synthetic_temporal_20260111_023313.csv

----------------------------------------------------------------------
Step 1: Loading Models
----------------------------------------------------------------------
Loading LSTM from: results/lstm_model_20260111_023557.h5
  ✓ LSTM model loaded
...
✓ All models loaded successfully!

======================================================================
                    MODEL EVALUATION RESULTS
======================================================================
...
Model              Accuracy     Precision    Recall       F1-Score     ROC-AUC
-------------------------------------------------------------------------------------
    LSTM           0.7842       0.7654       0.7912       0.7781       0.8523
    Random Forest  0.8123       0.7989       0.8234       0.8110       0.8742
    Gradient Boost 0.8234       0.8112       0.8345       0.8227       0.8856
>>> HYBRID         0.8567       0.8423       0.8678       0.8548       0.9123

✓ Test completed successfully!
```

### 2. Generated Files
- `test_results_roc_<timestamp>.png` - ROC curves comparing all models
- `test_results_confusion_matrix_<timestamp>.png` - Confusion matrix heatmap
- `test_results_distributions_<timestamp>.png` - Prediction probability distributions
- `test_results_report_<timestamp>.txt` - Full text report

---

## ✅ Success Criteria

### Your model is working well if:
- ✓ **Accuracy ≥ 75%** - Overall correct predictions
- ✓ **F1-Score ≥ 70%** - Balanced performance
- ✓ **ROC-AUC ≥ 80%** - Good discrimination ability
- ✓ **Hybrid > Individual models** - Meta-learner adds value
- ✓ **Recall ≥ 70%** - Catching most at-risk students

### ⚠️ Warning Signs:
- ✗ Accuracy < 60% - Model needs retraining
- ✗ Large gap between train/test - Overfitting
- ✗ Recall < 50% - Missing too many at-risk students
- ✗ Hybrid ≤ Best individual model - Meta-learner not working

---

## 🐛 Troubleshooting

### Error: "Model file not found"
```bash
# Check what models exist
ls results/lstm_model_*.h5

# Use correct timestamp
python test_hybrid_model.py --list
```

### Error: "Data file not found"
```bash
# Generate new data
python generate_synthetic_data.py

# Or check existing data
ls uploads/synthetic_*.csv | tail -4
```

### Error: "Shape mismatch"
- Test data must have same features as training data
- Check column names match exactly
- Ensure 32 weeks of temporal data per student

### Error: "Unseen label values"
- New categorical values in test data
- Script will auto-handle with warning
- Generate data from same script for consistency

### Low Performance (< 60% accuracy)
- Check if test data distribution matches training
- Verify all model files are from same timestamp
- Try generating fresh test data
- Review data preprocessing steps

---

## 📖 Detailed Guide

For comprehensive information, see:
- **[TEST_MODEL_GUIDE.md](TEST_MODEL_GUIDE.md)** - Full testing documentation
- **[test2.ipynb](test2.ipynb)** - Training notebook with examples

---

## 💡 Pro Tips

1. **Always use matching timestamps** - All model files must be from same training run
2. **Test on unseen data** - Don't test on training data
3. **Generate multiple test sets** - Test model robustness
4. **Check visualizations** - ROC curves and confusion matrices tell the story
5. **Compare with baseline** - Is hybrid better than individual models?
6. **Document results** - Save reports for future reference

---

## 🎓 Understanding Results

### Confusion Matrix
```
                Predicted
                Fail    Success
Actual  Fail    [TN]    [FP]
        Success [FN]    [TP]
```
- **TN (True Negative)**: Correctly predicted failure
- **FP (False Positive)**: False alarm - predicted success but failed
- **FN (False Negative)**: Missed - predicted failure but succeeded
- **TP (True Positive)**: Correctly predicted success

### Metrics
- **Accuracy**: Overall correctness = (TP + TN) / Total
- **Precision**: Of predicted successes, how many are right = TP / (TP + FP)
- **Recall**: Of actual successes, how many did we catch = TP / (TP + FN)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (0.5 = random, 1.0 = perfect)

### For Student Success Prediction:
- **High Recall** is important - Don't want to miss at-risk students
- **High Precision** also matters - Avoid false alarms that waste resources
- **F1-Score** balances both concerns

---

**Created**: January 11, 2026
**Model**: Hybrid LSTM + RF/GB with Meta-Learner
**Python Version**: 3.8+
**TensorFlow Version**: 2.x
