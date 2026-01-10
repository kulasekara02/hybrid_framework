import pandas as pd
import numpy as np
from datetime import datetime
import os

# Set random seed for reproducibility (change this for different datasets)
np.random.seed(42)

def generate_synthetic_student_data(n_students=1000, n_weeks=32, output_dir='./uploads'):
    """
    Generate synthetic student performance dataset with complex relationships.

    Parameters:
    - n_students: Number of students to generate
    - n_weeks: Number of weeks of temporal data (default: 32)
    - output_dir: Directory to save generated CSV files

    Returns:
    - Paths to generated static and temporal CSV files
    """

    print(f'Generating synthetic dataset for {n_students} students over {n_weeks} weeks...')
    print('=' * 70)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # ============================================================================
    # 1. GENERATE STATIC FEATURES
    # ============================================================================

    # Basic identifiers
    student_ids = [f'SYN{str(i).zfill(4)}' for i in range(1, n_students + 1)]
    institutions = np.random.choice(['Latvia_Uni_A', 'Latvia_Uni_B', 'Latvia_Uni_C'], n_students)
    program_ids = [f'PRG_{np.random.randint(100, 999)}' for _ in range(n_students)]

    # Countries with realistic distribution
    countries_home = np.random.choice(
        ['India', 'China', 'Nigeria', 'Kenya', 'Bangladesh', 'Pakistan', 'Vietnam', 'Indonesia'],
        n_students, p=[0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07, 0.05]
    )
    countries_host = ['Latvia'] * n_students

    # Academic info
    cohort_years = np.random.choice([2018, 2019, 2020, 2021, 2022], n_students)
    subject_fields = np.random.choice(
        ['Computer Science', 'Engineering', 'Medicine', 'Business', 'Mathematics'],
        n_students, p=[0.30, 0.25, 0.20, 0.15, 0.10]
    )
    study_levels = np.random.choice(['Bachelor', 'Master', 'PhD'], n_students, p=[0.50, 0.35, 0.15])
    study_modes = ['Full-time'] * n_students

    # Demographics
    genders = np.random.choice(['Male', 'Female'], n_students, p=[0.55, 0.45])
    ages = np.random.normal(24, 3.5, n_students).clip(18, 35).astype(int)
    marital_status = np.random.choice(['Single', 'Married'], n_students, p=[0.70, 0.30])

    # Language proficiency (1-5 scale) - this will heavily influence success
    language_proficiency = np.random.choice([1, 2, 3, 4, 5], n_students, p=[0.05, 0.15, 0.30, 0.35, 0.15])

    # Cultural factors (0-1 continuous)
    teaching_style_difference = np.random.beta(2, 2, n_students)  # Bell curve around 0.5
    cultural_distance = np.random.choice([0.3, 0.6, 0.75, 0.9], n_students, p=[0.20, 0.30, 0.30, 0.20])

    # Support programs
    support_programs = np.random.choice(['None', 'Mentoring', 'StudySkills'], n_students, p=[0.40, 0.35, 0.25])
    participates_in_buddy = np.random.choice([0, 1], n_students, p=[0.65, 0.35])
    participates_in_language = np.random.choice([0, 1], n_students, p=[0.70, 0.30])

    # Work-study balance
    works_while_studying = np.random.choice([0, 1], n_students, p=[0.45, 0.55])
    work_hours_per_week = np.where(
        works_while_studying == 1,
        np.random.choice([10, 15, 18, 20, 25], n_students),
        0
    )

    # Scholarship
    scholarship_status = np.random.choice(['None', 'Partial', 'Full'], n_students, p=[0.40, 0.45, 0.15])

    # ============================================================================
    # 2. GENERATE COMPLEX LATENT FACTORS (hidden variables that drive success)
    # ============================================================================

    # Latent Factor 1: Intrinsic Motivation (not directly observed)
    intrinsic_motivation = np.random.beta(2, 2, n_students)

    # Latent Factor 2: Prior Academic Foundation
    prior_foundation = np.random.beta(3, 2, n_students)  # Slightly skewed toward better

    # Latent Factor 3: Adaptation Capability
    adaptation_capability = np.random.beta(2.5, 2, n_students)

    # Latent Factor 4: Time Management Skills
    time_management = np.random.beta(2, 3, n_students)  # Slightly skewed toward weaker

    # ============================================================================
    # 3. GENERATE ACADEMIC PERFORMANCE FEATURES (influenced by latent factors)
    # ============================================================================

    # Entry GPA (0-10 scale) - influenced by prior foundation
    entry_gpa = (prior_foundation * 6 + np.random.normal(2, 0.8, n_students)).clip(0.5, 10)

    # Generate semester GPAs with complex relationships
    # GPA influenced by: prior foundation, motivation, language proficiency, work hours
    language_effect = (language_proficiency - 1) / 4  # Normalize to 0-1
    work_penalty = (work_hours_per_week / 30) * 0.3  # More work = lower GPA
    support_boost = np.where(support_programs != 'None', 0.15, 0)

    base_gpa = (
        prior_foundation * 0.4 +
        intrinsic_motivation * 0.3 +
        language_effect * 0.2 +
        adaptation_capability * 0.1
        - work_penalty
        + support_boost
    ) * 8  # Scale to 0-8 range

    # Add realistic variation and trends across semesters
    gpa_sem1 = (base_gpa + np.random.normal(0, 0.5, n_students) - 0.5).clip(0.5, 10)  # Slightly lower in sem1
    gpa_sem2 = (base_gpa + np.random.normal(0, 0.5, n_students) + adaptation_capability * 0.5).clip(0.5, 10)  # Improvement
    gpa_prev = (base_gpa + np.random.normal(0, 0.4, n_students)).clip(0.5, 10)

    # Credits
    credits_attempted_sem1 = np.full(n_students, 30)
    credits_attempted_sem2 = np.full(n_students, 30)

    # Credits earned - influenced by GPA and time management
    earn_rate_sem1 = (gpa_sem1 / 10 * 0.7 + time_management * 0.3).clip(0.6, 1.0)
    earn_rate_sem2 = (gpa_sem2 / 10 * 0.7 + time_management * 0.3).clip(0.6, 1.0)

    credits_earned_sem1 = (credits_attempted_sem1 * earn_rate_sem1).round().astype(int)
    credits_earned_sem2 = (credits_attempted_sem2 * earn_rate_sem2).round().astype(int)

    # Failed courses
    failed_courses_sem1 = ((1 - earn_rate_sem1) * 5).round().astype(int).clip(0, 3)
    failed_courses_sem2 = ((1 - earn_rate_sem2) * 5).round().astype(int).clip(0, 3)

    # Attendance rate - influenced by motivation, work hours, cultural adaptation
    attendance_rate = (
        intrinsic_motivation * 0.5 +
        adaptation_capability * 0.3 +
        (1 - work_hours_per_week / 30) * 0.2
    ).clip(0.3, 1.0)
    attendance_rate += np.random.normal(0, 0.05, n_students)
    attendance_rate = attendance_rate.clip(0.3, 1.0)

    # Engagement metrics - will be derived from temporal data
    # For now, create placeholders (will be filled from temporal data)
    mean_weekly_engagement = np.zeros(n_students)
    std_weekly_engagement = np.zeros(n_students)
    low_engagement_weeks = np.zeros(n_students, dtype=int)
    engagement_trend = np.zeros(n_students)

    # Assignment and exam scores
    academic_capability = (prior_foundation * 0.6 + intrinsic_motivation * 0.4)
    avg_assignment_score = (academic_capability * 60 + 30 + np.random.normal(0, 8, n_students)).clip(20, 100)
    avg_exam_score = (academic_capability * 60 + 25 + np.random.normal(0, 10, n_students)).clip(15, 100)

    # Submission behavior - influenced by time management
    late_submission_rate = ((1 - time_management) * 0.5 + np.random.uniform(0, 0.2, n_students)).clip(0, 0.8)
    missing_assignments_count = ((1 - time_management) * 10 + np.random.poisson(2, n_students)).clip(0, 15).astype(int)

    # Final GPA
    final_gpa_sem3_or_year = (
        gpa_prev * 0.4 +
        gpa_sem1 * 0.3 +
        gpa_sem2 * 0.3 +
        np.random.normal(0, 0.3, n_students)
    ).clip(0.5, 10)

    # ============================================================================
    # 4. GENERATE TARGET VARIABLE (SUCCESS) - COMPLEX NON-LINEAR RELATIONSHIP
    # ============================================================================

    # Success is determined by a complex weighted combination with interactions
    success_score = (
        # Academic factors (40%)
        (gpa_prev / 10) * 0.20 +
        (avg_assignment_score / 100) * 0.10 +
        (avg_exam_score / 100) * 0.10 +

        # Engagement factors (25%)
        intrinsic_motivation * 0.15 +
        attendance_rate * 0.10 +

        # Adaptation factors (20%)
        language_effect * 0.12 +
        adaptation_capability * 0.08 +

        # Support factors (15%)
        (1 - cultural_distance) * 0.08 +
        (support_boost * 2) * 0.07
    )

    # Add interaction effects (makes it harder for linear models)
    interaction_boost = (
        (language_proficiency >= 4) & (gpa_prev >= 6.0)
    ).astype(float) * 0.08

    interaction_penalty = (
        (work_hours_per_week > 20) & (gpa_prev < 5.0)
    ).astype(float) * 0.12

    success_score = success_score + interaction_boost - interaction_penalty

    # Add some noise to make it realistic
    success_score += np.random.normal(0, 0.08, n_students)
    success_score = success_score.clip(0, 1)

    # Convert to binary with threshold (creates ~40-50% success rate)
    success_threshold = np.percentile(success_score, 45)  # Adjust this to control class balance
    success_label = (success_score > success_threshold).astype(int)

    # Risk level based on success score quartiles
    risk_level = pd.cut(
        success_score,
        bins=[0, 0.33, 0.66, 1.0],
        labels=['High', 'Medium', 'Low']
    )

    print(f"Generated static features")
    print(f"  Success rate: {success_label.mean():.1%}")
    print(f"  Risk distribution - High: {(risk_level=='High').sum()}, Medium: {(risk_level=='Medium').sum()}, Low: {(risk_level=='Low').sum()}")

    # ============================================================================
    # 5. GENERATE TEMPORAL DATA (32 weeks)
    # ============================================================================

    temporal_data = []

    for i, student_id in enumerate(student_ids):
        # Get student's characteristics
        motivation = intrinsic_motivation[i]
        foundation = prior_foundation[i]
        adapt = adaptation_capability[i]
        time_mgmt = time_management[i]
        lang_prof = language_proficiency[i]
        work_hrs = work_hours_per_week[i]

        # Base engagement level for this student
        base_engagement = (motivation * 0.5 + foundation * 0.3 + adapt * 0.2)

        # Generate weekly pattern with realistic trends
        weeks = []
        for week in range(1, n_weeks + 1):
            # Semester (assuming 16 weeks per semester)
            semester = 1 if week <= 16 else 2
            week_in_semester = week if semester == 1 else week - 16

            # Weekly engagement with patterns:
            # - Initial high engagement (enthusiasm)
            # - Mid-semester slump
            # - End-of-semester stress boost
            cycle_effect = np.sin(week_in_semester / 16 * np.pi) * 0.15

            # Burnout effect (gradual decrease over time)
            burnout = -(week / n_weeks) * 0.1 * (1 - motivation)

            # Exam pressure (weeks 8, 16, 24, 32)
            exam_weeks = [8, 16, 24, 32]
            exam_boost = 0.2 if week in exam_weeks else 0

            # Work interference (random spikes for students who work)
            work_interference = -0.15 if (work_hrs > 15 and np.random.random() < 0.3) else 0

            # Calculate weekly engagement
            weekly_eng = base_engagement + cycle_effect + burnout + exam_boost + work_interference
            weekly_eng += np.random.normal(0, 0.08)
            weekly_eng = np.clip(weekly_eng, 0, 1)

            # Weekly attendance (correlated with engagement but with its own variance)
            weekly_att = attendance_rate[i] + (weekly_eng - base_engagement) * 0.5
            weekly_att += np.random.normal(0, 0.06)
            weekly_att = np.clip(weekly_att, 0, 1)

            # Assignments and quizzes (Poisson-like distribution)
            assignments = np.random.poisson(1.5 if week_in_semester not in [1, 16] else 0.5)
            quizzes = np.random.poisson(1.0 if week % 4 == 0 else 0.3)

            weeks.append({
                'student_id': student_id,
                'institution': institutions[i],
                'country_host': 'Latvia',
                'week_index': week,
                'semester_index': semester,
                'weekly_engagement': weekly_eng,
                'weekly_attendance': weekly_att,
                'weekly_assignments_submitted': assignments,
                'weekly_quiz_attempts': quizzes
            })

        temporal_data.extend(weeks)

        # Calculate aggregate engagement metrics for static data
        week_engagements = [w['weekly_engagement'] for w in weeks]
        mean_weekly_engagement[i] = np.mean(week_engagements)
        std_weekly_engagement[i] = np.std(week_engagements)
        low_engagement_weeks[i] = sum(1 for e in week_engagements if e < 0.3)

        # Engagement trend (linear regression slope)
        week_indices = np.arange(1, n_weeks + 1)
        engagement_trend[i] = np.polyfit(week_indices, week_engagements, 1)[0]

    print(f"Generated temporal data ({len(temporal_data)} records)")

    # ============================================================================
    # 6. CREATE DATAFRAMES
    # ============================================================================

    # Static DataFrame
    df_static = pd.DataFrame({
        'student_id': student_ids,
        'institution': institutions,
        'program_id': program_ids,
        'country_home': countries_home,
        'country_host': countries_host,
        'cohort_year': cohort_years,
        'subject_field': subject_fields,
        'study_level': study_levels,
        'study_mode': study_modes,
        'gender': genders,
        'age': ages,
        'marital_status': marital_status,
        'language_proficiency': language_proficiency,
        'teaching_style_difference': teaching_style_difference,
        'cultural_distance': cultural_distance,
        'support_program': support_programs,
        'participates_in_buddy_program': participates_in_buddy,
        'participates_in_language_course': participates_in_language,
        'works_while_studying': works_while_studying,
        'work_hours_per_week': work_hours_per_week,
        'scholarship_status': scholarship_status,
        'entry_gpa': entry_gpa,
        'gpa_sem1': gpa_sem1,
        'gpa_sem2': gpa_sem2,
        'gpa_prev': gpa_prev,
        'credits_attempted_sem1': credits_attempted_sem1,
        'credits_earned_sem1': credits_earned_sem1,
        'credits_attempted_sem2': credits_attempted_sem2,
        'credits_earned_sem2': credits_earned_sem2,
        'failed_courses_sem1': failed_courses_sem1,
        'failed_courses_sem2': failed_courses_sem2,
        'attendance_rate': attendance_rate,
        'mean_weekly_engagement': mean_weekly_engagement,
        'std_weekly_engagement': std_weekly_engagement,
        'low_engagement_weeks': low_engagement_weeks,
        'engagement_trend': engagement_trend,
        'avg_assignment_score': avg_assignment_score,
        'avg_exam_score': avg_exam_score,
        'late_submission_rate': late_submission_rate,
        'missing_assignments_count': missing_assignments_count,
        'final_gpa_sem3_or_year': final_gpa_sem3_or_year,
        'success_label': success_label,
        'risk_level': risk_level
    })

    # Temporal DataFrame
    df_temporal = pd.DataFrame(temporal_data)

    # ============================================================================
    # 7. SAVE TO CSV FILES
    # ============================================================================

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    static_path = f"{output_dir}/synthetic_static_{timestamp}.csv"
    temporal_path = f"{output_dir}/synthetic_temporal_{timestamp}.csv"

    df_static.to_csv(static_path, index=False)
    df_temporal.to_csv(temporal_path, index=False)

    print(f"\n{'='*70}")
    print(f"SYNTHETIC DATASET GENERATED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"Static data:    {static_path}")
    print(f"   - {len(df_static)} students")
    print(f"   - {len(df_static.columns)} features")
    print(f"\nTemporal data:  {temporal_path}")
    print(f"   - {len(df_temporal)} records ({n_weeks} weeks per student)")
    print(f"   - {len(df_temporal.columns)} features")
    print(f"\nTo use this data, update the PATHS dictionary in the data loading cell:")
    print(f"   'latvia_static': '{static_path}'")
    print(f"   'latvia_temporal': '{temporal_path}'")
    print(f"{'='*70}\n")

    return static_path, temporal_path, df_static, df_temporal


if __name__ == "__main__":
    # Generate the dataset
    static_path, temporal_path, df_static, df_temporal = generate_synthetic_student_data(
        n_students=1000,
        n_weeks=32,
        output_dir='./uploads'
    )

    # Display sample
    print('\nSample of generated STATIC data:')
    print(df_static.head(3).to_string())

    print('\n\nSample of generated TEMPORAL data:')
    print(df_temporal.head(10).to_string())

    print('\n\nStatistical Summary:')
    print(f'Success rate: {df_static["success_label"].mean():.1%}')
    print(f'Average GPA: {df_static["gpa_prev"].mean():.2f} (±{df_static["gpa_prev"].std():.2f})')
    print(f'Average engagement: {df_static["mean_weekly_engagement"].mean():.3f}')
    print(f'Average attendance: {df_static["attendance_rate"].mean():.1%}')
