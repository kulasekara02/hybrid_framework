# Hybrid Model Testing Guide

This guide explains how to test the trained hybrid model (LSTM + Random Forest + Meta-Learner) with your own dataset to get prediction results.

## Prerequisites

Before testing the model, ensure you have the following:

1. **Python 3.8+** installed
2. Required packages:
   - numpy
   - pandas
   - tensorflow (2.x)
   - scikit-learn
   - joblib

Install dependencies:
```bash
pip install numpy pandas tensorflow scikit-learn joblib
```

## Trained Model Files

The trained models are stored in the `results/` folder:
- `lstm_model_*.h5` - LSTM model for temporal patterns
- `rf_model_*.pkl` - Random Forest model for static features  
- `meta_learner_*.pkl` - Meta-learner that combines predictions
- `preprocessing_objects_*.pkl` - Preprocessing objects (scalers, encoders)
- `hybrid_model_complete_*.pkl` - Complete model package

## Data Format

### Static Data (Required)
A CSV file with student information including:
- `student_id` - Unique student identifier
- Academic features: `gpa_sem1`, `gpa_sem2`, `gpa_prev`, `entry_gpa`
- Engagement features: `attendance_rate`, `mean_weekly_engagement`
- Performance features: `avg_assignment_score`, `avg_exam_score`
- Course info: `failed_courses_sem1`, `failed_courses_sem2`
- Demographics: `age`, `gender`, `country_home`, etc.

### Temporal Data (Optional)
A CSV file with weekly engagement data:
- `student_id` - Matches static data
- `week_index` - Week number (1-32)
- `weekly_engagement` - Engagement score
- `weekly_attendance` - Attendance rate
- `weekly_assignments_submitted` - Assignments submitted
- `weekly_quiz_attempts` - Quiz attempts

## Usage

### Command Line

```bash
# Basic usage with static data only
python test_model.py --static_data path/to/static_data.csv

# With temporal data
python test_model.py --static_data path/to/static.csv --temporal_data path/to/temporal.csv

# Specify output file
python test_model.py -s static.csv -t temporal.csv -o results/my_predictions.csv

# Use custom model paths
python test_model.py -s data.csv --lstm_path results/lstm_model_20251104_171930.h5
```

### Python Script

```python
from test_model import test_model, load_models, predict_with_hybrid_model

# Test with your dataset
results = test_model(
    static_data_path='your_static_data.csv',
    temporal_data_path='your_temporal_data.csv',  # Optional
    output_path='results/predictions.csv'
)

# Access predictions
print(results['predicted_success_proba'])
print(results['predicted_risk_level'])
```

## Output

The script generates a CSV file with the following prediction columns:
- `predicted_success_proba` - Success probability (0-1)
- `predicted_risk_level` - Risk level (Low Risk, Medium Risk, High Risk)
- `predicted_at_risk` - Binary prediction (0 or 1)
- `lstm_contribution` - LSTM model's prediction
- `rf_contribution` - Random Forest model's prediction
- `success_label` - Human-readable label (Success or At Risk)

## Example

Using the sample data in `uploads/`:

```bash
python test_model.py \
    --static_data uploads/international_students_static_latvia.csv \
    --temporal_data uploads/international_students_temporal_latvia.csv \
    --output results/latvia_predictions.csv
```

## Interpreting Results

### Risk Levels
- **Low Risk** (probability < 0.33): Student likely to succeed
- **Medium Risk** (0.33 - 0.66): Monitor student progress
- **High Risk** (probability > 0.66): Immediate intervention needed

### Model Contributions
- `lstm_contribution`: Captures temporal engagement patterns
- `rf_contribution`: Based on static features and academic performance
- The meta-learner combines both for the final prediction

## Troubleshooting

1. **Model not found**: Ensure model files exist in `results/` folder
2. **Missing columns**: Check your data has required columns
3. **Memory issues**: Process data in batches for large datasets
