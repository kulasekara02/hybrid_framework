# Master Defense - Answers to Review Questions

**Candidate**: Master's Student  
**Program**: Computer Science / Data Science  
**Date**: February 2026  
**Defense Committee Feedback Response**

---

## Question 1: Presentation Issues - Figures and Equations

### Problem Statement
Several figures and equations in your thesis are neither numbered nor referenced in the text, some figures are difficult to read, and one figure is inverted.

### Answer

| **Aspect** | **Impact** | **Correction Steps** |
|------------|-----------|---------------------|
| **Missing Figure Numbers** | **Interpretability**: Readers cannot identify which figure the text discusses. This creates confusion and makes it hard to follow the research results. | 1. Add sequential numbering to all figures (Figure 1, Figure 2, etc.)<br>2. Reference each figure in the text (e.g., "as shown in Figure 3")<br>3. Create a List of Figures in the front matter |
| **Unnumbered Equations** | **Reproducibility**: Without equation numbers, other researchers cannot cite specific formulas or replicate the mathematical methods used in the study. | 1. Number all equations sequentially (Eq. 1, Eq. 2, etc.)<br>2. Reference equations in the text when discussing methods<br>3. Use consistent numbering format (e.g., chapter.equation: 3.1, 3.2) |
| **Poor Figure Quality** | **Scientific Validity**: Low-resolution or small text in figures makes it impossible to verify the data visualization. This reduces trust in the results. | 1. Regenerate all figures at minimum 300 DPI resolution<br>2. Increase font sizes in charts to at least 10pt<br>3. Use vector graphics (SVG/PDF) instead of raster images<br>4. Ensure color schemes are colorblind-friendly |
| **Inverted Figure** | **Credibility**: An upside-down figure suggests careless preparation and may lead reviewers to question the accuracy of all research outputs. | 1. Locate and correct the inverted figure immediately<br>2. Verify orientation of all figures systematically<br>3. Add a quality control checklist for final submission |

### Summary of Actions
To fully comply with the thesis template and academic standards, I will:

1. **Complete Figure Audit** (1 day)
   - Review all 15-20 figures in the thesis
   - Ensure proper numbering (Figure 1, Figure 2, etc.)
   - Add captions with clear descriptions

2. **Equation Standardization** (1 day)
   - Number all mathematical formulas
   - Cross-reference equations in the text
   - Verify mathematical notation consistency

3. **Quality Enhancement** (2 days)
   - Regenerate low-quality figures
   - Fix the inverted figure
   - Test all figures for readability when printed

4. **Template Compliance Check** (1 day)
   - Match figure formatting to thesis guidelines
   - Ensure consistent spacing and alignment
   - Create complete List of Figures and List of Tables

**Expected Result**: All figures and equations will be properly numbered, referenced, and meet academic publication standards, ensuring full reproducibility and scientific validity.

---

## Question 2: Late Fusion Methodology Justification

### Problem Statement
Your thesis adopts a late-fusion weighted averaging strategy to combine LSTM and Random Forest outputs, but early and intermediate fusion were not implemented or empirically evaluated.

### Answer

| **Fusion Strategy** | **Description** | **Advantages** | **Disadvantages** | **Why Not Used** |
|---------------------|----------------|----------------|-------------------|-----------------|
| **Early Fusion** | Combine all features (temporal + static) into single input for one model | - Simpler architecture<br>- Single model training<br>- Features interact early | - Cannot leverage specialized models<br>- LSTM not optimized for static data<br>- RF not optimized for sequences | LSTM requires 3D input (time × features), while RF needs 2D static data. Combining them early loses model-specific strengths. |
| **Intermediate Fusion** | Merge feature representations at middle layers of neural networks | - Learns shared representations<br>- Balances early and late fusion | - Complex architecture<br>- Requires deep neural networks<br>- Difficult to interpret | Random Forest is not a neural network, so it has no "middle layers" to fuse. This approach only works with multiple neural networks. |
| **Late Fusion** **(Used)** | Train separate models, then combine predictions at final stage | - Uses optimal model for each data type<br>- Easy to interpret contributions<br>- Simple to implement and debug<br>- Can weight models by performance | - Models don't learn joint representations<br>- May miss cross-modal interactions | **Chosen because**: LSTM specializes in temporal patterns, RF specializes in static features, and meta-learner learns optimal combination. |

### Justification for Late Fusion

**Methodological Reasons**:

1. **Data Type Compatibility**
   - Temporal data (32 weeks × 4 features) requires sequential processing → LSTM is specialized for this
   - Static data (40+ features) requires feature interaction analysis → Random Forest excels here
   - Late fusion respects the natural structure of both data types

2. **Interpretability Requirements**
   - Educational domain needs explainable predictions for ethical use
   - Late fusion allows separate feature importance analysis (RF) and temporal pattern analysis (LSTM)
   - Clear contribution percentages (e.g., LSTM: 55%, RF: 45%) help advisors understand predictions

3. **Existing Research Precedent**
   - Literature review of 50 papers shows late fusion is the standard approach for multi-modal educational data
   - Studies combining clickstream + demographic data typically use late fusion (Gašević et al., 2016; Whitehill et al., 2017)

**Specific Risks and Limitations**:

| **Risk/Limitation** | **Description** | **Mitigation Strategy** |
|---------------------|----------------|------------------------|
| **Suboptimal Feature Interaction** | Models train independently, so they may miss important relationships between temporal and static features (e.g., how cultural distance affects engagement trends) | Conduct correlation analysis between static features and temporal patterns before modeling to understand key relationships |
| **Weight Selection Dependency** | Meta-learner weights depend on validation set performance, which may not generalize to all student types | Use stratified validation ensuring all student subgroups are represented; test weights on external dataset |
| **Limited Cross-Modal Learning** | LSTM cannot learn from static features, and RF cannot learn from temporal patterns | Accept this limitation as a trade-off for interpretability and specialized model performance |
| **Comparative Validation Gap** | Without early/intermediate fusion experiments, claims of "optimal approach" lack empirical support | **Future Work**: Implement early fusion baseline (concat features → single LSTM) to demonstrate late fusion superiority through ablation study |

### Proposed Additional Validation

To strengthen the methodology section, I will add:

1. **Baseline Comparison** (2 days of work)
   - Implement simple early fusion: concatenate temporal + static features → single LSTM
   - Compare performance: Early Fusion vs. Late Fusion
   - Expected result: Late fusion outperforms by 5-10% due to model specialization

2. **Ablation Study** (already partially done)
   - Test LSTM-only performance
   - Test RF-only performance
   - Test hybrid (late fusion) performance
   - Show that hybrid exceeds individual models

3. **Literature Justification** (1 day)
   - Add section citing 5-7 papers that use late fusion for educational data
   - Explain why intermediate fusion is uncommon for LSTM+RF combinations
   - Reference meta-learning theory (Wolpert, 1992)

**Conclusion**: Late fusion is appropriate because it respects data structure, maintains interpretability, and follows domain best practices. The main limitation is lack of comparative experiments, which I will address by adding a simple early fusion baseline.

---

## Question 3: Generalizability to Other Populations

### Problem Statement
Your dataset focuses on international students in Latvian higher education institutions. To what extent can your results be generalized to other student populations or institutional contexts?

### Answer

### Current Scope and Limitations

| **Population Characteristic** | **Current Study** | **Generalizability** |
|-------------------------------|------------------|---------------------|
| **Geographic Context** | Latvia (Baltic state, EU member) | **Limited**: Results may not apply to universities in USA, Asia, or Africa due to different educational systems |
| **Student Type** | International students only | **Limited**: Framework may not work for domestic students who lack cultural adaptation challenges |
| **Institution Type** | 5 universities/colleges in Latvia | **Moderate**: Can likely transfer to other small-country European universities, but not to large-scale institutions |
| **Academic Level** | 60% Bachelor, 40% Master students | **Moderate**: Framework should work for undergraduate/graduate levels, but PhD students have different needs |
| **Time Period** | 32-week academic calendar | **Good**: Most institutions use similar semester structures (30-36 weeks) |
| **Cultural Diversity** | 15+ countries, focus on India, Nigeria, China, Bangladesh | **Moderate**: Results apply to South Asian and African students, but may not generalize to other regions |

### Extent of Generalizability

**Where Results LIKELY Generalize**:
- ✅ Other European universities with similar international student populations
- ✅ Small-to-medium universities (5,000-15,000 students)
- ✅ Institutions using 30-36 week academic calendars
- ✅ Countries with significant international student enrollment from South Asia and Africa

**Where Results PROBABLY DO NOT Generalize**:
- ❌ Domestic student populations (different challenges)
- ❌ USA universities (different education system, larger scale)
- ❌ Online-only institutions (no campus integration)
- ❌ Professional certification programs (short duration, different goals)
- ❌ Countries with minimal international student enrollment

### Required Additional Validation

| **Validation Type** | **Purpose** | **Required Data** | **Expected Timeline** | **Success Criteria** |
|---------------------|------------|-------------------|---------------------|---------------------|
| **Cross-Institutional Validation** | Test if model works at other universities | Data from 2-3 universities in different countries (e.g., Estonia, Poland, Germany) | 6-12 months | Model achieves 85%+ of original accuracy without retraining |
| **Cross-Population Testing** | Evaluate performance on domestic students | Data from 300+ domestic students at same institution | 3-6 months | Identify which features need adjustment for domestic population |
| **Temporal Validation** | Check if model remains accurate over time | Test on students from different academic years (2024, 2025, 2026) | 2-3 years | Model maintains 90%+ accuracy across years |
| **External Dataset Validation** | Test on publicly available data | Apply framework to OULAD dataset (UK, 32K students) | 1-2 months | Model transfers to different educational context |
| **Cultural Subgroup Analysis** | Verify model works for all regions | Separate testing for Asian, African, European, American students | 1 month | Model performs equally well (±5%) across all subgroups |

### Concrete Steps to Justify Generalization

**Step 1: Immediate Validation (1-2 months)**
- Apply framework to OULAD dataset (Open University, UK)
- Compare performance on UK data vs. Latvia data
- Identify which features transfer and which need recalibration

**Step 2: Survey-Based External Validation (3-6 months)**
- Collect survey data from 200-300 students in 2-3 other countries
- Test if cultural distance, language proficiency, and engagement patterns remain strong predictors
- Calculate cross-context validation accuracy

**Step 3: Sensitivity Analysis (1 month)**
- Test how model performs when features are changed slightly
- Example: What if cultural distance is measured differently?
- Example: What if attendance is measured biweekly instead of weekly?
- This shows which assumptions are critical vs. flexible

**Step 4: Subgroup Performance Analysis (2 weeks)**
- Calculate accuracy separately for:
  - Asian students vs. African students vs. European students
  - Male vs. Female students
  - STEM majors vs. Non-STEM majors
  - Bachelor vs. Master students
- Identify if model has bias or works equally well for all groups

**Step 5: Documentation of Limitations (1 week)**
- Add "Generalizability Limitations" section to thesis
- Clearly state: "Results apply to international students in small European universities"
- List all assumptions and their validity ranges
- Suggest adaptations needed for other contexts

### Thesis Updates Required

I will add the following sections:

1. **Limitations Chapter** (new section, 3-4 pages)
   - Geographic limitations
   - Population scope
   - Temporal limitations
   - Institutional context boundaries

2. **Future Work Section** (expand existing section)
   - Cross-institutional validation plan
   - Adaptation guidelines for other contexts
   - Feature engineering for different populations

3. **Discussion Enhancement** (revise existing section)
   - Explicitly state generalizability boundaries
   - Compare with literature from other countries
   - Explain which findings are universal vs. context-specific

**Conclusion**: Results are most generalizable to similar European universities with international student populations. Full generalization requires validation on 2-3 external datasets from different countries and institutional contexts. I will conduct OULAD validation (1-2 months) and clearly document limitations in the revised thesis.

---

## Summary: Key Points for Defense Presentation

### Slide 1: Presentation Quality Issues
- **Acknowledged**: Figures and equations were not properly numbered
- **Impact**: Reduced interpretability and reproducibility
- **Solution**: Complete audit and reformatting (5 days)
- **Commitment**: All academic standards will be met in final version

### Slide 2: Late Fusion Methodology
- **Choice**: Late fusion combines LSTM (temporal) + RF (static) at prediction stage
- **Justification**: Respects data structure, maintains interpretability, follows domain best practices
- **Limitation**: Lack of comparative experiments with early fusion
- **Action**: Will add early fusion baseline comparison

### Slide 3: Generalizability
- **Current Scope**: International students in Latvian universities
- **Generalizable To**: Similar European institutions with international populations
- **Not Generalizable To**: Domestic students, USA universities, online institutions
- **Required Validation**: OULAD testing (1-2 months) + cross-institutional data
- **Commitment**: Clear documentation of limitations in revised thesis

---

**Document Purpose**: Professional answers for master defense presentation  
**Language Level**: B1 English (Intermediate)  
**Format**: Tables and structured responses suitable for slides  
**Date Prepared**: February 8, 2026  
**Status**: Ready for presentation slide conversion
