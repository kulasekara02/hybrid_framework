# Student Success Prediction - Hybrid ML Framework

A machine learning system that predicts whether international students will succeed or fail in their studies. It combines two different approaches: one that looks at how students engage week-by-week over 32 weeks (using LSTM), and another that analyzes their background and personal factors (using Random Forest). Together they work better than either one alone.

## What This Does

- Takes student data (both weekly engagement patterns and personal info)
- Predicts if they'll pass or fail their studies
- Tells you WHY the model thinks they might fail (which factors matter most)
- Identifies at-risk students early so advisors can help them

## How It Works (Simple Version)

1. **LSTM Part**: Looks at 32 weeks of student activity (attendance, assignments, quiz attempts). Learns if engagement is getting worse or staying steady.

2. **Random Forest Part**: Looks at student background (age, country, language level, cultural background, GPA). Learns patterns about who typically succeeds.

3. **Combine Both**: Uses a simple meta-learner to blend the predictions from both models for the final result.

The cool thing about this is it actually considers cultural factors - how different a student's home country culture is from the teaching style they're experiencing. This hasn't been done much in similar systems before.

## What You Need

```
Python 3.8+
TensorFlow 2.x
scikit-learn
pandas, numpy
matplotlib, seaborn
```

## How to Use

1. **Get the code**
   ```bash
   git clone https://github.com/kulasekara02/hybrid_framework.git
   cd hybrid_framework
   pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
   ```

2. **Run it**
   ```bash
   jupyter notebook test2.ipynb
   ```
   
   Just run all the cells top to bottom. It will:
   - Load the student data
   - Clean and prepare it
   - Train the LSTM model on 32-week patterns
   - Train the Random Forest model on background info
   - Combine them for final predictions
   - Show you visualizations and results

   This takes about 15-20 minutes.

3. **What you get**
   - Risk predictions for each student (High/Medium/Low)
   - Which students are most at risk
   - What factors make them at risk (why)
   - Recommendations for what to do

## Files Explained

```
hybrid_framework/
├── test2.ipynb                    # Main notebook - run this
├── uploads/                       # Student data goes here
│   ├── synthetic_static_*.csv    # Background info (age, country, etc)
│   └── synthetic_temporal_*.csv  # Weekly engagement data
├── outputs/                       # Results saved here
│   ├── students_high_risk.csv    # At-risk students
│   ├── root_cause_frequency.csv  # Why students fail
│   └── cluster_*.csv             # Groups of similar students
└── results/                       # Trained models saved here
    ├── lstm_model_*.h5           # The LSTM model
    ├── rf_model_*.pkl            # The Random Forest model
    └── meta_learner_*.pkl        # The combining model
```

## Quick Example

Want to predict for one student?

```python
import numpy as np
import keras
import joblib

# Load the trained models
lstm = keras.models.load_model('results/lstm_model_latest.h5')
rf = joblib.load('results/rf_model_latest.pkl')
meta = joblib.load('results/meta_learner_latest.pkl')

# Prepare data: 32 weeks × 4 metrics (engagement, attendance, assignments, quizzes)
temporal_data = np.random.rand(1, 32, 4)

# Prepare background data: age, country, gpa, language level, etc (40+ features)
static_data = np.random.rand(1, 40)

# Get prediction
lstm_pred = lstm.predict(temporal_data)
rf_pred = rf.predict(static_data)
final = meta.predict_proba(np.column_stack([lstm_pred, rf_pred]))[0, 1]

print(f"Success chance: {final:.0%}")
if final < 0.33:
    print("Status: HIGH RISK - Student needs help")
elif final < 0.67:
    print("Status: MEDIUM RISK - Monitor closely")
else:
    print("Status: LOW RISK - Student is doing fine")
```

## The Data

Currently using realistic simulated data with ~1,800 students and 32 weeks of tracking each. This is for testing the system before using real university data.

The idea is to eventually get real data from universities, but for now synthetic data lets us:
- Test everything works
- Avoid privacy issues
- Validate the approach

Plan to add real data later by surveying actual international students.

## Results You Get

When you run the notebook, it creates several outputs:

### What the model tells you:
- Accuracy, Precision, Recall scores (how well it predicts)
- Risk classifications for each student (High/Medium/Low)
- Which features/factors matter most in predictions
- Confusion matrix (true positives, false positives, etc)

### Charts it creates:
- Line graphs showing model performance
- Heat maps of feature correlations
- ROC curves comparing models
- Student risk distribution
- Feature importance bar charts

### Data files it saves:
- `students_high_risk.csv` - Students at risk
- `root_cause_frequency.csv` - What factors cause failures
- `rf_feature_importance.csv` - Which factors matter most
- Confusion matrices and performance reports

## Why This Works Better

Detecting at-risk students in week 8-12 gives advisors time to help. Other systems detect them in week 14+, leaving only 2-4 weeks. That's a big advantage.

Also, most student prediction systems ignore cultural factors. This one includes:
- How different the home culture is from teaching style
- Language proficiency level
- Cultural distance from home country

This actually matters for international students - it's one of the top factors in the model.

## Plan for Real Data

Right now it uses realistic simulated data. Next steps would be:

1. Survey real international students (55 questions ready)
2. Use public OULAD dataset (32K+ students available)
3. Get data from universities if they agree

The system should work the same way with real data, just with real patterns instead of simulated ones.

## Want to Use This?

The code is set up to work with any similar student data. You'd need:
- Student IDs
- Weekly engagement data (attendance, assignments, quiz attempts) 
- Background info (age, country, GPA, language level, etc)
- Whether they passed or failed

Then feed it into the notebook and it'll train models and make predictions.

## Testing

There's a `testing/` folder with scripts to validate the models work correctly. Run:
```bash
python testing/test_hybrid.py
```

This generates test reports and visualizations to make sure everything's working.

## License

MIT - Use it however you want.

---

**Last Updated**: January 2026
