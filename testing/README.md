# Testing Folder - Hybrid Model Testing

## 📁 Folder Structure

```
testing/
├── README.md                  ← You are here
├── test_hybrid.py             ← Main test script
├── run_test.bat               ← Windows quick run script
├── run_test.sh                ← Linux/Mac quick run script
│
├── test_data/                 ← Your test datasets
│   ├── synthetic_static_20260111_023313.csv
│   └── synthetic_temporal_20260111_023313.csv
│
└── test_results/              ← Test outputs (created after running)
    ├── roc_curves_*.png
    ├── confusion_matrix_*.png
    ├── distributions_*.png
    └── test_report_*.txt
```

---

## 🚀 Quick Start (3 Seconds!)

### Windows:
```cmd
run_test.bat
```

### Linux/Mac:
```bash
bash run_test.sh
```

### Or manually:
```bash
python test_hybrid.py
```

That's it! The script will:
1. Load your trained model (`20260111_023557`)
2. Load test data from `test_data/` folder
3. Run complete evaluation
4. Show results in console
5. Save plots and report in `test_results/` folder

---

## 📊 What You Get

### Console Output:
```
======================================================================
  🧪 HYBRID MODEL TESTING
======================================================================

[Step 1] Loading Models
----------------------------------------------------------------------
✓ LSTM loaded
✓ Random Forest loaded
✓ Gradient Boosting loaded
✓ Meta-learner loaded
✓ Preprocessing objects loaded

[Step 2] Loading Test Data
----------------------------------------------------------------------
Static data: test_data/synthetic_static_20260111_023313.csv
  → 1000 students
  → Success rate: 54.7%

[Step 3] Preprocessing Data
----------------------------------------------------------------------
✓ Static features scaled
✓ Temporal sequences scaled

[Step 4] Generating Predictions
----------------------------------------------------------------------
✓ All predictions generated!

======================================================================
  MODEL PERFORMANCE RESULTS
======================================================================

📊 Model Comparison:
------------------------------------------------------------------------------------
Model                Accuracy  Precision     Recall   F1-Score    ROC-AUC
------------------------------------------------------------------------------------
    LSTM               0.7842     0.7654     0.7912     0.7781     0.8523
    Random Forest      0.8123     0.7989     0.8234     0.8110     0.8742
    Gradient Boosting  0.8234     0.8112     0.8345     0.8227     0.8856
★★★ HYBRID             0.8567     0.8423     0.8678     0.8548     0.9123
------------------------------------------------------------------------------------
★★★ = Your Hybrid Model (should have best scores!)

✅ TEST COMPLETED SUCCESSFULLY
```

### Generated Files in `test_results/`:
1. **roc_curves_*.png** - ROC curves comparing all models
2. **confusion_matrix_*.png** - Confusion matrix heatmap
3. **distributions_*.png** - Prediction probability distributions
4. **test_report_*.txt** - Complete text report

---

## ✅ Expected Performance

Your hybrid model should achieve:
- **Accuracy**: 75-85% ✓
- **F1-Score**: 70-80% ✓
- **ROC-AUC**: 80-90% ✓
- **Hybrid > Individual models** ✓

---

## 🔧 Configuration

Edit these variables in `test_hybrid.py` if needed:

```python
MODEL_TIMESTAMP = '20260111_023557'  # Your model timestamp
STATIC_DATA = 'test_data/synthetic_static_20260111_023313.csv'
TEMPORAL_DATA = 'test_data/synthetic_temporal_20260111_023313.csv'
RESULTS_DIR = 'test_results'
```

---

## 📝 Test Data Format

### Static CSV (1 row per student):
- `student_id` - Unique identifier
- `success_label` - Target (0 or 1)
- Academic features: GPA, credits, scores
- Demographics: age, gender, language
- Engagement: attendance, assignments
- **~45 columns total**

### Temporal CSV (32 rows per student):
- `student_id` - Links to static data
- `week_index` - Week 1-32
- `weekly_engagement` - Score 0-1
- `weekly_attendance` - Rate 0-1
- `weekly_assignments_submitted` - Count
- `weekly_quiz_attempts` - Count

---

## 🎯 Understanding Results

### ★★★ HYBRID Should Be Best!

The hybrid model combines:
- **LSTM**: Captures time patterns (engagement trends)
- **Tree models**: Captures static features (GPA, demographics)
- **Meta-learner**: Optimally combines both

**Expected result**: Hybrid outperforms by 3-7% in F1-score

### Metrics Explained:

**Accuracy**: Overall correctness
- 85% = 850 out of 1000 students predicted correctly

**Precision**: Of predicted successes, how many are correct
- High precision = Few false alarms

**Recall**: Of actual successes, how many we caught
- High recall = Few missed students

**F1-Score**: Balanced measure (best overall metric)
- Combines precision and recall

**ROC-AUC**: Discrimination ability
- 0.5 = random guessing
- 1.0 = perfect predictions

---

## 🐛 Troubleshooting

### "FileNotFoundError"
**Fix**: Check model timestamp matches your files
```bash
# See available models
cd ..
ls results/lstm_model_*.h5
```

### "Shape mismatch error"
**Fix**: Ensure test data has same format as training data
- Use the datasets in `test_data/` folder
- They match your training data format

### "No module named 'tensorflow'"
**Fix**: Install dependencies
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn joblib
```

### Low performance (<70% accuracy)
**Check**:
1. All model files from same timestamp?
2. Test data format correct?
3. Try different test dataset
4. May need model retraining

---

## 💡 Tips

1. **Check visualizations**: They tell the story better than numbers
2. **ROC curves**: Higher and to the left is better
3. **Confusion matrix**: Focus on false positives (missed at-risk students)
4. **Distributions**: Good separation = good model
5. **Always compare**: Is hybrid better than individual models?

---

## 📖 More Information

See parent directory for detailed guides:
- `../QUICK_TEST_GUIDE.md` - Quick reference
- `../TEST_MODEL_GUIDE.md` - Detailed guide
- `../README_TESTING.md` - Complete overview

---

## 🎓 Next Steps

1. ✓ Run the test (`python test_hybrid.py`)
2. ✓ Review console output
3. ✓ Check visualizations in `test_results/`
4. ✓ Read the text report
5. ✓ Verify hybrid has best scores
6. ✓ If good performance, ready for deployment!

---

**Model**: Hybrid LSTM + RF/GB (Timestamp: 20260111_023557)
**Test Data**: 1000 students, 32 weeks temporal data
**Last Updated**: January 11, 2026
