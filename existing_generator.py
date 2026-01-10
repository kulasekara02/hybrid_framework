# ============================================================================
# SYNTHETIC DATA GENERATOR
# Creates randomized datasets matching your data structure
# but with different distributions to make modeling non-trivial
# ============================================================================

import pandas as pd
import numpy as np
from datetime import datetime
import os

class SyntheticStudentDataGenerator:
    """
    Generates synthetic student data with realistic correlations
    that make the prediction task challenging
    """
    
    def __init__(self, num_students=1000, num_weeks=32, random_seed=42):
        np.random.seed(random_seed)
        self.num_students = num_students
        self.num_weeks = num_weeks
        self.student_ids = [f"SYN{str(i).zfill(6)}" for i in range(num_students)]
        
        # Define possible values for categorical variables
        self.institutions = ['Synthetic_Uni_A', 'Synthetic_Uni_B', 'Synthetic_College_C', 
                            'Synthetic_Tech_D', 'Synthetic_Institute_E']
        self.program_ids = ['COMP', 'ENG', 'BUS', 'MED', 'LAW', 'ART', 'SCI', 'DATA', 'MIS']
        self.countries_home = ['India', 'China', 'Nigeria', 'Brazil', 'Pakistan', 'Bangladesh',
                              'Kenya', 'Vietnam', 'Philippines', 'Indonesia', 'Mexico', 'Iran']
        self.countries_host = ['Country_A', 'Country_B', 'Country_C']
        self.cohort_years = [2018, 2019, 2020, 2021, 2022, 2023]
        self.subject_fields = ['Computer Science', 'Engineering', 'Business', 'Medicine', 
                              'Social Sciences', 'Humanities', 'Information Systems']
        self.study_levels = ['Bachelor', 'Master']
        self.study_modes = ['Full-time', 'Part-time']
        self.genders = ['Male', 'Female', 'Other']
        self.marital_statuses = ['Single', 'Married']
        self.teaching_styles = ['Low', 'Medium', 'High']
        self.scholarship_statuses = ['None', 'Partial', 'Full']
    
    def generate_static_data(self):
        """Generate static student data with realistic correlations"""
        data = []
        
        for student_id in self.student_ids:
            # Basic demographics
            institution = np.random.choice(self.institutions)
            program_id = np.random.choice(self.program_ids)
            country_home = np.random.choice(self.countries_home)
            country_host = np.random.choice(self.countries_host)
            cohort_year = np.random.choice(self.cohort_years)
            subject_field = np.random.choice(self.subject_fields)
            study_level = np.random.choice(self.study_levels)
            study_mode = np.random.choice(self.study_modes, p=[0.75, 0.25])
            gender = np.random.choice(self.genders, p=[0.45, 0.45, 0.1])
            age = np.random.randint(18, 45)
            marital_status = np.random.choice(self.marital_statuses, p=[0.7, 0.3])
            
            # Language and cultural factors
            language_proficiency = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.25, 0.15])
            teaching_style_diff = np.random.choice(self.teaching_styles, p=[0.3, 0.4, 0.3])
            
            # Cultural distance (higher for certain countries)
            if country_home in ['China', 'India', 'Nigeria', 'Bangladesh']:
                cultural_distance = np.random.uniform(0.5, 0.8)
            elif country_home in ['Brazil', 'Mexico']:
                cultural_distance = np.random.uniform(0.3, 0.6)
            else:
                cultural_distance = np.random.uniform(0.1, 0.4)
            
            # Support systems
            support_program = np.random.choice([0, 1])
            buddy_program = np.random.choice([0, 1], p=[0.6, 0.4])
            language_course = np.random.choice([0, 1], p=[0.7, 0.3])
            
            # Work and financial factors
            works_while_studying = np.random.choice([0, 1], p=[0.5, 0.5])
            work_hours = np.random.choice([0, 10, 15, 18, 20, 25, 30]) if works_while_studying else 0
            scholarship_status = np.random.choice(self.scholarship_statuses, p=[0.4, 0.4, 0.2])
            
            # Academic performance - entry GPA
            entry_gpa = np.random.uniform(1.5, 4.0)
            
            # Create base ability factor (latent variable affecting performance)
            base_ability = np.random.normal(0, 1)
            
            # Language proficiency affects performance
            lang_effect = (language_proficiency - 3) * 0.3
            
            # Work affects performance negatively
            work_effect = -work_hours * 0.02 if works_while_studying else 0
            
            # Support programs help
            support_effect = 0.2 if support_program else 0
            support_effect += 0.15 if buddy_program else 0
            
            # Cultural distance creates challenges
            cultural_effect = -cultural_distance * 0.5
            
            # Calculate performance metrics with noise
            performance_factor = base_ability + lang_effect + work_effect + support_effect + cultural_effect
            
            # GPAs with realistic ranges and noise
            gpa_sem1 = np.clip(2.5 + performance_factor * 0.8 + np.random.normal(0, 0.5), 0, 4)
            gpa_sem2 = np.clip(gpa_sem1 + np.random.normal(0, 0.3), 0, 4)
            gpa_prev = (gpa_sem1 + gpa_sem2) / 2
            
            # Credits
            if study_mode == 'Full-time':
                credits_attempted_sem1 = np.random.choice([24, 27, 30, 33, 35])
                credits_attempted_sem2 = np.random.choice([24, 27, 30, 33, 35])
            else:
                credits_attempted_sem1 = np.random.choice([18, 21, 24])
                credits_attempted_sem2 = np.random.choice([18, 21, 24])
            
            # Credits earned related to performance
            pass_rate_sem1 = np.clip(0.6 + performance_factor * 0.15, 0.5, 1.0)
            pass_rate_sem2 = np.clip(0.6 + performance_factor * 0.15 + np.random.normal(0, 0.05), 0.5, 1.0)
            
            credits_earned_sem1 = int(credits_attempted_sem1 * pass_rate_sem1)
            credits_earned_sem2 = int(credits_attempted_sem2 * pass_rate_sem2)
            
            failed_courses_sem1 = np.random.poisson(max(0, 2 - performance_factor))
            failed_courses_sem2 = np.random.poisson(max(0, 2 - performance_factor))
            
            # Engagement metrics
            base_engagement = np.clip(0.5 + performance_factor * 0.15, 0, 1)
            mean_weekly_engagement = np.clip(base_engagement + np.random.normal(0, 0.1), 0.1, 1.0)
            std_weekly_engagement = np.random.uniform(0.05, 0.15)
            low_engagement_weeks = int(self.num_weeks * (1 - base_engagement) * np.random.uniform(0.3, 0.7))
            engagement_trend = np.random.uniform(-0.2, 0.2)
            
            # Attendance related to engagement
            attendance_rate = np.clip(base_engagement + np.random.normal(0, 0.1), 0.2, 1.0)
            
            # Assignment and exam scores
            avg_assignment_score = np.clip((gpa_sem1/4) * 100 + np.random.normal(0, 10), 0, 100)
            avg_exam_score = np.clip((gpa_sem2/4) * 100 + np.random.normal(0, 10), 0, 100)
            
            # Late submissions and missing assignments (inversely related to performance)
            late_submission_rate = np.clip(0.6 - performance_factor * 0.1 + np.random.normal(0, 0.1), 0, 1)
            missing_assignments_count = np.random.poisson(max(0, 3 - performance_factor * 1.5))
            
            # Final GPA (sem3 or year)
            gpa_sem3 = np.clip(gpa_prev + np.random.normal(0, 0.3) + performance_factor * 0.1, 0, 4)
            
            # Success determination with complex logic
            success_criteria = [
                gpa_sem3 >= 2.5,
                attendance_rate >= 0.6,
                (credits_earned_sem1 + credits_earned_sem2) >= (credits_attempted_sem1 + credits_attempted_sem2) * 0.7,
                failed_courses_sem1 + failed_courses_sem2 <= 3
            ]
            
            success_label = 1 if sum(success_criteria) >= 3 else 0
            
            # Risk category based on multiple factors
            risk_score = (
                (4 - gpa_prev) * 0.25 + 
                (1 - attendance_rate) * 0.2 + 
                (failed_courses_sem1 + failed_courses_sem2) * 0.1 + 
                late_submission_rate * 0.15 + 
                missing_assignments_count * 0.05 + 
                (1 - mean_weekly_engagement) * 0.15
            )
            
            if risk_score < 0.4:
                risk_category = 'Low'
            elif risk_score < 0.7:
                risk_category = 'Medium'
            else:
                risk_category = 'High'
            
            # Create row
            row = {
                'student_id': student_id,
                'institution': institution,
                'program_id': program_id,
                'country_home': country_home,
                'country_host': country_host,
                'cohort_year': cohort_year,
                'subject_field': subject_field,
                'study_level': study_level,
                'study_mode': study_mode,
                'gender': gender,
                'age': age,
                'marital_status': marital_status,
                'language_proficiency': language_proficiency,
                'teaching_style_difference': teaching_style_diff,
                'cultural_distance': cultural_distance,
                'support_program': support_program,
                'participates_in_buddy_program': buddy_program,
                'participates_in_language_course': language_course,
                'works_while_studying': works_while_studying,
                'work_hours_per_week': work_hours,
                'scholarship_status': scholarship_status,
                'entry_gpa': round(entry_gpa, 2),
                'gpa_sem1': round(gpa_sem1, 2),
                'gpa_sem2': round(gpa_sem2, 2),
                'gpa_prev': round(gpa_prev, 2),
                'credits_attempted_sem1': credits_attempted_sem1,
                'credits_earned_sem1': credits_earned_sem1,
                'credits_attempted_sem2': credits_attempted_sem2,
                'credits_earned_sem2': credits_earned_sem2,
                'failed_courses_sem1': failed_courses_sem1,
                'failed_courses_sem2': failed_courses_sem2,
                'attendance_rate': round(attendance_rate, 3),
                'mean_weekly_engagement': round(mean_weekly_engagement, 3),
                'std_weekly_engagement': round(std_weekly_engagement, 3),
                'low_engagement_weeks': low_engagement_weeks,
                'engagement_trend': round(engagement_trend, 3),
                'avg_assignment_score': round(avg_assignment_score, 1),
                'avg_exam_score': round(avg_exam_score, 1),
                'late_submission_rate': round(late_submission_rate, 3),
                'missing_assignments_count': missing_assignments_count,
                'final_gpa_sem3_or_year': round(gpa_sem3, 2),
                'success_label': success_label,
                'risk_level': risk_category
            }
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def generate_temporal_data(self, static_df):
        """Generate temporal (weekly) data for each student"""
        temporal_data = []
        
        for _, student in static_df.iterrows():
            student_id = student['student_id']
            institution = student['institution']
            country_host = student['country_host']
            
            # Get base engagement from static data
            base_engagement = student['mean_weekly_engagement']
            std_engagement = student['std_weekly_engagement']
            trend = student['engagement_trend']
            
            # Generate weekly data
            for week in range(1, self.num_weeks + 1):
                semester = 1 if week <= 16 else 2
                
                # Add trend and weekly variation
                week_factor = (week - 1) / self.num_weeks
                weekly_engagement = np.clip(
                    base_engagement + trend * week_factor + np.random.normal(0, std_engagement),
                    0.05, 1.0
                )
                
                # Attendance correlated with engagement
                weekly_attendance = np.clip(
                    weekly_engagement + np.random.normal(0, 0.1),
                    0, 1.0
                )
                
                # Assignments and quizzes vary by week
                is_busy_week = week % 4 == 0  # Every 4th week is busier
                
                if is_busy_week:
                    avg_assignments = 3
                    avg_quizzes = 2
                else:
                    avg_assignments = 1
                    avg_quizzes = 1
                
                # Performance affects submission behavior
                if weekly_engagement > 0.7:
                    assignments_submitted = np.random.poisson(avg_assignments)
                    quiz_attempts = np.random.poisson(avg_quizzes)
                elif weekly_engagement > 0.4:
                    assignments_submitted = np.random.poisson(avg_assignments * 0.7)
                    quiz_attempts = np.random.poisson(avg_quizzes * 0.8)
                else:
                    assignments_submitted = np.random.poisson(avg_assignments * 0.4)
                    quiz_attempts = np.random.poisson(avg_quizzes * 0.5)
                
                row = {
                    'student_id': student_id,
                    'institution': institution,
                    'country_host': country_host,
                    'week_index': week,
                    'semester_index': semester,
                    'weekly_engagement': round(weekly_engagement, 6),
                    'weekly_attendance': round(weekly_attendance, 6),
                    'weekly_assignments_submitted': max(0, assignments_submitted),
                    'weekly_quiz_attempts': max(0, quiz_attempts)
                }
                
                temporal_data.append(row)
        
        return pd.DataFrame(temporal_data)
    
    def generate_datasets(self, output_dir='synthetic_data'):
        """Generate both static and temporal datasets"""
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating synthetic data for {self.num_students} students...")
        
        # Generate static data
        print("Creating static dataset...")
        static_df = self.generate_static_data()
        
        # Generate temporal data
        print("Creating temporal dataset (this may take a moment)...")
        temporal_df = self.generate_temporal_data(static_df)
        
        # Save datasets
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        static_filename = f"{output_dir}/synthetic_static_students_{timestamp}.csv"
        temporal_filename = f"{output_dir}/synthetic_temporal_students_{timestamp}.csv"
        
        static_df.to_csv(static_filename, index=False)
        temporal_df.to_csv(temporal_filename, index=False)
        
        print(f"\n✓ Static dataset saved: {static_filename}")
        print(f"  - Shape: {static_df.shape}")
        print(f"  - Success rate: {static_df['success_label'].mean():.1%}")
        print(f"  - Risk distribution: {dict(static_df['risk_level'].value_counts())}")
        
        print(f"\n✓ Temporal dataset saved: {temporal_filename}")
        print(f"  - Shape: {temporal_df.shape}")
        print(f"  - Weeks per student: {self.num_weeks}")
        
        return static_df, temporal_df, static_filename, temporal_filename


# ============================================================================
# USAGE: Run this cell to generate synthetic data
# ============================================================================

# Configuration
NUM_STUDENTS = 1000      # Number of students to generate
NUM_WEEKS = 32           # Number of weeks of temporal data
RANDOM_SEED = 42         # Change this for different random data
OUTPUT_DIR = 'synthetic_data'

# Generate the data
generator = SyntheticStudentDataGenerator(
    num_students=NUM_STUDENTS, 
    num_weeks=NUM_WEEKS,
    random_seed=RANDOM_SEED
)

synthetic_static_df, synthetic_temporal_df, static_file, temporal_file = generator.generate_datasets(OUTPUT_DIR)

print("\n" + "="*70)
print("SYNTHETIC DATA GENERATION COMPLETE!")
print("="*70)
print("\nYou can now use these datasets for experiments.")
print("To generate different data, change RANDOM_SEED and re-run this cell.")