# Publicly Available Datasets for Student Success Prediction
## Real Data Alternatives for Your Hybrid Framework

**Date**: November 22, 2025  
**Purpose**: Replace synthetic data with real, publicly available educational datasets

---

## 🌟 Top Recommendation: OULAD (Open University Learning Analytics Dataset)

### Overview
The **Open University Learning Analytics Dataset (OULAD)** is the best publicly available option for your thesis.

**Source**: https://analyse.kmi.open.ac.uk/open_dataset

**Published by**: Open University (UK) & Knowledge Media Institute

**Citation**: Kuzilek J., Hlosta M., Zdrahal Z. (2017) Open University Learning Analytics dataset. Scientific Data 4:170171

### Dataset Details

| Attribute | Value |
|-----------|-------|
| **Students** | 32,593 unique students |
| **Courses** | 7 courses (modules) |
| **Presentations** | 22 course presentations (semesters) |
| **Time Period** | 2013-2014 academic year |
| **Total Records** | 10,655,280 interactions |
| **Format** | CSV files |
| **Size** | ~200 MB compressed |
| **License** | CC BY 4.0 (free to use with attribution) |

### Files Included

1. **studentInfo.csv** (32,593 records)
   - Demographics: age, gender, region, disability
   - Education: highest education level
   - Socioeconomic: IMD (Index of Multiple Deprivation)
   - Registration: studied credits, previous attempts
   - **Outcome**: final_result (Pass/Distinction/Fail/Withdrawn)

2. **studentVle.csv** (10,655,280 records)
   - Daily VLE (Virtual Learning Environment) interactions
   - Activity type: resource views, forum posts, assignments
   - **Temporal data**: date and sum of clicks per activity
   - **Perfect for your LSTM**: Week-by-week engagement patterns

3. **studentAssessment.csv** (173,912 records)
   - Assessment scores (0-100)
   - Submission dates
   - Assessment types: TMA (Tutor Marked Assessment), CMA (Computer Marked)
   - Weight in final grade

4. **studentRegistration.csv** (32,593 records)
   - Registration dates
   - Unregistration dates (for withdrawals)

5. **courses.csv** (7 records)
   - Module information
   - Duration, credits, subject domain

6. **assessments.csv** (206 records)
   - Assessment metadata
   - Type, date, weight

7. **vle.csv** (6,364 records)
   - VLE activity types
   - Week availability

### Why OULAD is Perfect for Your Thesis

#### ✅ Matches Your Framework Requirements

| Your Framework Needs | OULAD Provides |
|---------------------|----------------|
| Temporal engagement data | ✅ 10M+ VLE interactions (daily → weekly) |
| Static demographic features | ✅ Age, gender, education, region, socioeconomic |
| Academic performance | ✅ Assessment scores, previous attempts |
| Success/failure outcomes | ✅ Pass/Distinction/Fail/Withdrawn |
| Large sample size | ✅ 32,593 students (18× your synthetic dataset) |
| Multi-course validation | ✅ 7 different courses |

#### ✅ Advantages Over Synthetic Data

1. **Real Student Behavior**: Actual engagement patterns, not simulated
2. **Peer-Reviewed**: Dataset validated in multiple academic publications (400+ citations)
3. **Widely Used**: Benchmark dataset in learning analytics research
4. **No Privacy Concerns**: Already anonymized and approved for research
5. **Free & Immediate**: Download today, use immediately
6. **Publication-Ready**: Using OULAD strengthens your thesis credibility

#### ⚠️ Limitations (Minor)

1. **UK-Focused**: All students from UK (less international diversity)
   - **Solution**: Position as "comparative study" - UK vs. Latvia context
   
2. **No Cultural Factors**: Missing language proficiency, cultural distance
   - **Solution**: Use placeholders or focus on temporal/academic patterns
   
3. **Distance Learning**: Open University uses distance education model
   - **Solution**: Highlight online learning relevance (post-pandemic context)

---

## 🎓 Other Publicly Available Educational Datasets

### Option 2: UCI Student Performance Dataset

**Source**: https://archive.ics.uci.edu/ml/datasets/Student+Performance

**Size**: 
- Mathematics: 395 students
- Portuguese: 649 students

**Features** (33 variables):
- Demographics: age, sex, family size, parent education
- Social: free time, going out, alcohol consumption
- Academic: past grades, study time, absences
- **Outcome**: G3 (final grade 0-20)

**Pros**:
- ✅ Easy to access (direct download)
- ✅ Well-documented features
- ✅ Multiple studies use it (benchmark)

**Cons**:
- ❌ Small sample (< 1,000 students)
- ❌ No temporal data (snapshot only)
- ❌ Secondary school (not university)

**Use Case**: Good for static feature analysis, not for LSTM temporal modeling

---

### Option 3: HarvardX-MITx MOOC Dataset

**Source**: https://dataverse.harvard.edu/dataverse/moodatasets

**Size**: 290,000+ students, 13 million records

**Features**:
- Demographics: country, education level, age
- Engagement: video views, forum posts, problem attempts
- **Temporal**: Daily activity logs
- **Outcome**: Certification status

**Pros**:
- ✅ Massive sample size
- ✅ Temporal engagement data
- ✅ International students (multiple countries)

**Cons**:
- ⚠️ Requires data use agreement
- ⚠️ MOOCs different from traditional universities
- ⚠️ Very low completion rates (~5%)

**Use Case**: If you want international diversity + large scale

---

### Option 4: Kaggle Education Datasets

**Source**: https://www.kaggle.com/datasets

#### Popular Datasets:

1. **Students Performance in Exams**
   - Link: kaggle.com/datasets/spscientist/students-performance-in-exams
   - Size: 1,000 students
   - Features: Demographics, test scores
   - ❌ No temporal data

2. **Online Education Dataset**
   - Link: kaggle.com/datasets/mharis9/online-education-dataset
   - Size: Varies
   - Features: Virtual classroom engagement
   - ✅ Some temporal features

3. **Student Dropout Prediction**
   - Various datasets available
   - Size: 500-5,000 students typically
   - Features: Academic + demographic

**Pros**:
- ✅ Easy download (Kaggle API)
- ✅ Community support/kernels
- ✅ Variety of options

**Cons**:
- ⚠️ Variable quality
- ⚠️ Often smaller samples
- ⚠️ Limited temporal data

---

### Option 5: Moodle Research Datasets

**Source**: Various universities release anonymized Moodle logs

**Examples**:
- TU Delft: Open educational data
- Canvas/Blackboard: Some institutions share data

**Pros**:
- ✅ Real LMS (Learning Management System) data
- ✅ Temporal engagement logs
- ✅ Assessment scores

**Cons**:
- ⚠️ Requires institutional partnerships
- ⚠️ Not always publicly available
- ⚠️ Variable formats

---

## 📊 Dataset Comparison Table

| Dataset | Students | Temporal Data | International | Free | Publication Quality |
|---------|----------|---------------|---------------|------|---------------------|
| **OULAD** ⭐ | 32,593 | ✅ Daily VLE | ❌ UK only | ✅ Yes | 🌟🌟🌟🌟🌟 |
| UCI Student Perf | 649 | ❌ No | ❌ Portugal only | ✅ Yes | 🌟🌟🌟 |
| HarvardX-MITx | 290,000+ | ✅ Daily logs | ✅ Global | ⚠️ Agreement | 🌟🌟🌟🌟 |
| Kaggle Datasets | 500-5K | ⚠️ Limited | ⚠️ Varies | ✅ Yes | 🌟🌟 |
| Moodle Logs | Varies | ✅ Yes | ⚠️ Varies | ⚠️ Restricted | 🌟🌟🌟 |

---

## 🚀 How to Use OULAD in Your Framework

### Step 1: Download OULAD (5 minutes)

```bash
# Visit: https://analyse.kmi.open.ac.uk/open_dataset
# Click "Download" → Register (free) → Download ZIP file
# Extract to: ./uploads/oulad/
```

Or use the automatic download code in your notebook!

### Step 2: Run Preprocessing (Already in Your Notebook)

The code I added to your notebook will:
1. Load 7 OULAD CSV files
2. Map OULAD features to your framework format
3. Create temporal sequences (32 weeks)
4. Add placeholder features (cultural distance, etc.)
5. Validate data quality

### Step 3: Use Your Existing Framework

After preprocessing, your framework will work **exactly the same** because the data structure matches:

```python
# Your existing code works as-is!
X_static = df_static[static_feature_cols].values
X_temporal = temporal_sequences
y = df_static['success_probability'].values

# Train models
lstm_model.fit(X_temporal_train, y_train, ...)
rf_model.fit(X_static_train, y_train)
meta_learner.fit(meta_train, y_train_binary)
```

### Step 4: Compare Results

You can now compare:
- **Synthetic data performance** (proof of concept)
- **OULAD performance** (real-world validation)
- **Feature importance** (which factors matter most in real data)

---

## 📝 How to Present This in Your Thesis

### Option A: Primary Dataset (Replace Synthetic)

**Approach**: Use OULAD as your main dataset

**Thesis Structure**:
1. Introduction: Student success prediction problem
2. Literature Review: Existing approaches + OULAD usage in literature
3. Methodology: Your hybrid LSTM+RF framework
4. **Dataset**: OULAD (32,593 students, 7 courses)
5. Results: Performance on OULAD
6. Discussion: Feature importance, early detection
7. Conclusion: Contributions + future work

**Advantage**: Strongest validation (real data from start)

**Professor Response**: "You used real data!" ✅

---

### Option B: Dual Validation (Synthetic + OULAD)

**Approach**: Keep synthetic, add OULAD for validation

**Thesis Structure**:
1. Introduction: Student success prediction problem
2. Literature Review: Existing approaches
3. Methodology: Your hybrid LSTM+RF framework
4. **Dataset 1**: Synthetic data (framework development)
5. **Dataset 2**: OULAD (real-world validation)
6. Results: 
   - Synthetic: Proof of concept
   - OULAD: Real-world performance
   - Comparison: Framework generalizability
7. Discussion: Feature importance across datasets
8. Conclusion: Validated framework

**Advantage**: Demonstrates generalizability (works on multiple data sources)

**Professor Response**: "You validated on real data!" ✅✅

---

### Option C: Comparative Study (Synthetic vs. OULAD vs. Survey)

**Approach**: Use all three data sources

**Thesis Structure**:
1. Introduction: Student success prediction problem
2. Literature Review: Existing approaches
3. Methodology: Your hybrid LSTM+RF framework
4. **Datasets**:
   - Synthetic: Framework development (1,783 students)
   - OULAD: Large-scale validation (32,593 students)
   - Survey: Latvia-specific validation (200-300 students)
5. Results: 
   - Synthetic: Proof of concept
   - OULAD: Real-world large-scale
   - Survey: Context-specific (Latvia, cultural factors)
6. Discussion: Framework robustness across contexts
7. Conclusion: Generalizable, validated framework

**Advantage**: Most comprehensive validation (3 data sources)

**Professor Response**: "This is publication-quality work!" ✅✅✅

---

## ⏱️ Timeline Comparison

| Approach | Time to Complete | Risk | Publication Strength |
|----------|------------------|------|----------------------|
| **Keep Synthetic Only** | 0 days | 🔴 High (questioned) | ⭐⭐ |
| **Use OULAD** | 1-2 days | 🟢 Low | ⭐⭐⭐⭐ |
| **Synthetic + OULAD** | 3-5 days | 🟢 Low | ⭐⭐⭐⭐⭐ |
| **Survey Only** | 4-6 weeks | 🟡 Medium (delays) | ⭐⭐⭐⭐ |
| **All Three** | 6-8 weeks | 🟡 Medium (complex) | ⭐⭐⭐⭐⭐⭐ |

---

## 🎯 Recommendation

### **Use Option B: Synthetic + OULAD (Dual Validation)**

**Why?**
1. ✅ **Quick**: 3-5 days to integrate OULAD
2. ✅ **Strong**: Real data validation (32K students)
3. ✅ **Smart**: Shows generalizability
4. ✅ **Safe**: If OULAD doesn't work perfectly, you have synthetic
5. ✅ **Publishable**: Demonstrates framework robustness

**Timeline**:
- **Day 1**: Download OULAD, run preprocessing
- **Day 2**: Train models on OULAD, compare performance
- **Day 3**: Analyze feature importance differences
- **Day 4-5**: Update thesis with OULAD results
- **Week 2**: Thesis defense ready! 🎓

---

## 📚 Citation Information

### OULAD Citation (Required if you use it):

```bibtex
@article{kuzilek2017open,
  title={Open University Learning Analytics dataset},
  author={Kuzilek, Jakub and Hlosta, Martin and Zdrahal, Zdenek},
  journal={Scientific data},
  volume={4},
  number={1},
  pages={1--8},
  year={2017},
  publisher={Nature Publishing Group}
}
```

### Your Thesis Citation (Updated):

```bibtex
@mastersthesis{yourname2025,
  title={Predicting International Student Academic Success Using a Hybrid 
         LSTM-Random Forest Framework with Cultural Adaptation Modeling},
  author={Your Name},
  school={Your University},
  year={2025},
  note={Framework validated on synthetic data and Open University Learning 
        Analytics Dataset (32,593 students)}
}
```

---

## 🔗 Useful Links

### OULAD Resources:
- **Official Website**: https://analyse.kmi.open.ac.uk/open_dataset
- **Documentation**: https://analyse.kmi.open.ac.uk/open_dataset/documentation
- **Paper**: https://www.nature.com/articles/sdata2017171
- **GitHub Examples**: https://github.com/oulad

### Other Dataset Resources:
- **UCI ML Repository**: https://archive.ics.uci.edu/ml/datasets
- **Kaggle Education**: https://www.kaggle.com/datasets?tags=13207-Education
- **Harvard Dataverse**: https://dataverse.harvard.edu/dataverse/moodatasets
- **Google Dataset Search**: https://datasetsearch.research.google.com/

---

## ❓ FAQ

### Q1: Is OULAD data similar enough to international students in Latvia?

**A**: While OULAD is UK-focused, the **temporal engagement patterns** and **academic performance predictors** are universal. Position your thesis as:
- Framework development on diverse data sources
- Methodology applicable across contexts
- Future work: Deploy in Latvian institutions

### Q2: Will my cultural adaptation features work with OULAD?

**A**: OULAD doesn't have cultural distance/language proficiency. You have two options:
1. **Use placeholders** (all students = low cultural distance, high language proficiency)
2. **Remove cultural features** for OULAD, compare with/without cultural features on synthetic data

### Q3: How long does OULAD download take?

**A**: 
- Download: 5-10 minutes (200 MB compressed)
- Extract: 1-2 minutes
- Preprocessing: 2-5 minutes (code provided in notebook)
- **Total: ~15-20 minutes to be ready**

### Q4: Do I need permission to use OULAD?

**A**: No! OULAD is released under CC BY 4.0 license. You only need to:
1. Register on their website (free)
2. Cite their paper in your thesis
3. Acknowledge the source

### Q5: What if my model performs worse on OULAD than synthetic?

**A**: This is actually **good for your thesis**! It shows:
- Framework sensitivity to data characteristics
- Need for domain adaptation
- Real-world challenges (more realistic)
- Opportunity to discuss improvement strategies

---

**Document Status**: ✅ Ready to Use  
**Last Updated**: November 22, 2025  
**Next Step**: Run the OULAD cells in your notebook!
