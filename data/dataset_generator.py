"""
Dataset Generator for Academic Risk & Engagement Prediction System
Attributes:
- Age: int (18-26)
- Gender: categorical ('Male', 'Female')
- Socio_Economic_Status: categorical ('Low', 'Medium', 'High')
- GPA: float (0.00 - 4.00)
- Attendance_Rate: float (0.0 - 100.0%)
- Assignments_Submitted: int (0 - 10)
- Online_Learning_Hours: float (1.0 - 35.0 hrs/week)
- Forum_Participation: int (0 - 50 posts/queries)
- Lab_Performance: float (0.0 - 100.0)
- Extracurricular_Participation: categorical ('Yes', 'No')
- At_Risk: binary int (0 = Not at Risk / Safe, 1 = At Risk / Dropout Hazard)
"""

import numpy as np
import pandas as pd
import os

def generate_academic_risk_dataset(n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Age (Student cohort typically 18 to 26)
    age = np.random.choice([18, 19, 20, 21, 22, 23, 24, 25, 26], 
                           size=n_samples, 
                           p=[0.15, 0.25, 0.25, 0.15, 0.10, 0.04, 0.03, 0.02, 0.01])
    
    # 2. Gender ('Male', 'Female')
    gender = np.random.choice(['Male', 'Female'], size=n_samples, p=[0.52, 0.48])
    
    # 3. Socio_Economic_Status ('Low', 'Medium', 'High')
    ses = np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.28, 0.52, 0.20])
    
    # Base academic capability latent factor z_academic ~ N(0, 1)
    # SES has subtle environmental influence on resources/time availability
    ses_effect = np.where(ses == 'Low', -0.25, np.where(ses == 'High', 0.25, 0.0))
    z_capability = np.random.normal(loc=0.0, scale=1.0, size=n_samples) + ses_effect
    
    # 4. GPA (Scale 0.0 to 4.0)
    gpa = 2.95 + 0.52 * z_capability + np.random.normal(0, 0.25, n_samples)
    gpa = np.clip(np.round(gpa, 2), 1.20, 4.00)
    
    # 5. Attendance_Rate (Scale 0.0% to 100.0%)
    # Correlated with capability and discipline
    attendance = 78.0 + 11.0 * z_capability + np.random.normal(0, 6.5, n_samples)
    attendance = np.clip(np.round(attendance, 1), 32.0, 100.0)
    
    # 6. Assignments_Submitted (0 to 10)
    # Strongly related to attendance and diligence
    assignment_prob = np.clip(0.72 + 0.15 * z_capability + (attendance - 75) / 150.0, 0.1, 0.98)
    assignments = np.random.binomial(n=10, p=assignment_prob)
    
    # 7. Online_Learning_Hours (Weekly LMS hours: 1 to 35)
    online_hours = 12.0 + 4.5 * z_capability + (attendance / 10.0) * 0.5 + np.random.exponential(scale=3.0, size=n_samples)
    online_hours = np.clip(np.round(online_hours, 1), 1.0, 35.0)
    
    # 8. Forum_Participation (0 to 50 interactions)
    forum_lambda = np.clip(6.0 + 3.0 * z_capability + (online_hours / 4.0), 0.5, 35.0)
    forum_posts = np.random.poisson(lam=forum_lambda)
    forum_posts = np.clip(forum_posts, 0, 50)
    
    # 9. Lab_Performance (0.0 to 100.0)
    lab_score = 72.0 + 10.5 * z_capability + (attendance / 100.0) * 8.0 + np.random.normal(0, 6.0, n_samples)
    lab_score = np.clip(np.round(lab_score, 1), 20.0, 100.0)
    
    # 10. Extracurricular_Participation ('Yes', 'No')
    extra_prob = np.clip(0.40 + 0.10 * (attendance / 100.0) + np.where(ses == 'High', 0.1, 0.0), 0.15, 0.75)
    extracurricular = np.where(np.random.rand(n_samples) < extra_prob, 'Yes', 'No')
    
    # 11. Target Variable: At_Risk (0 = Safe, 1 = At Risk)
    # Realistic educational risk equation:
    # High risk when: GPA < 2.5, Attendance < 70%, Assignments < 6, Low LMS hours, Low Lab
    z_risk = (
        -1.85 * (gpa - 2.8)
        - 0.065 * (attendance - 75.0)
        - 0.42 * (assignments - 7.0)
        - 0.09 * (online_hours - 12.0)
        - 0.045 * (lab_score - 70.0)
        - 0.05 * (forum_posts - 8.0)
        + np.where(ses == 'Low', 0.35, np.where(ses == 'High', -0.25, 0.0))
        + np.where(extracurricular == 'No', 0.20, -0.15)
        # Non-linear penalty for extreme low attendance (<60%) or low GPA (<2.0)
        + np.where(attendance < 60.0, 1.25, 0.0)
        + np.where(gpa < 2.0, 1.40, 0.0)
        + np.random.normal(0, 0.55, n_samples)  # Realistic stochastic noise
    )
    
    # Sigmoidal mapping to probability
    prob_risk = 1.0 / (1.0 + np.exp(-z_risk))
    at_risk = (prob_risk >= 0.50).astype(int)
    
    df = pd.DataFrame({
        'Age': age,
        'Gender': gender,
        'Socio_Economic_Status': ses,
        'GPA': gpa,
        'Attendance_Rate': attendance,
        'Assignments_Submitted': assignments,
        'Online_Learning_Hours': online_hours,
        'Forum_Participation': forum_posts,
        'Lab_Performance': lab_score,
        'Extracurricular_Participation': extracurricular,
        'At_Risk': at_risk
    })
    
    return df

if __name__ == '__main__':
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, 'academic_risk_data.csv')
    
    df = generate_academic_risk_dataset(n_samples=1500, random_state=42)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} records successfully saved to: {output_path}")
    print("\nDataset Shape:", df.shape)
    print("\nAttribute Data Types & Null Counts:")
    print(df.info())
    print("\nTarget Class Distribution (At_Risk):")
    print(df['At_Risk'].value_counts(normalize=True).round(4) * 100)
    print("\nFirst 5 Records:")
    print(df.head())
