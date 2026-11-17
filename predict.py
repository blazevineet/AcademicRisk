"""
Interactive Prediction & Early Intervention Advisor
Academic Risk & Engagement Prediction System (Units 1 - 5)
Leverages Champion Bayesian Logistic Regression (with epistemic uncertainty)
and HMM Viterbi temporal trajectory decoding.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')
PREPROCESSOR_PATH = os.path.join(BASE_DIR, 'models', 'preprocessor.joblib')

from src.unit4_hmm_viterbi import HiddenMarkovModel

def load_inference_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(
            "Trained model artifacts not found! Please run 'python run_experiments.py' first."
        )
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    return model, preprocessor

def compute_engagement_index(student: dict) -> float:
    """Estimates continuous composite Engagement Index (0-100)."""
    att = (student['Attendance_Rate'] - 30.0) / 70.0 * 30.0
    assign = (student['Assignments_Submitted'] / 10.0) * 25.0
    lms = min(student['Online_Learning_Hours'] / 30.0, 1.0) * 20.0
    forum = min(student['Forum_Participation'] / 35.0, 1.0) * 15.0
    extra = 10.0 if student['Extracurricular_Participation'] == 'Yes' else 0.0
    score = np.clip(att + assign + lms + forum + extra, 0.0, 100.0)
    return round(float(score), 2)

def generate_intervention_plan(student: dict, risk_prob: float, earliest_risk_week: int) -> list:
    """Generates targeted pedagogical interventions based on cross-sectional & temporal dynamics."""
    actions = []
    if risk_prob >= 0.50:
        actions.append(f"[URGENT INTERVENTION] High attrition hazard. Schedule 1-on-1 Academic Advisory before Week {earliest_risk_week + 1 if earliest_risk_week > 0 else 5}.")
    if student['Attendance_Rate'] < 75.0:
        actions.append(f"[ATTENDANCE DEFICIT] Attendance is {student['Attendance_Rate']}%. Enforce mandatory check-ins and lecture capture review.")
    if student['Assignments_Submitted'] < 7:
        actions.append(f"[COURSEWORK DEFICIT] Only {student['Assignments_Submitted']}/10 assignments completed. Offer assignment clinic and tutoring.")
    if student['GPA'] < 6.0:
        actions.append(f"[GRADE MONITORING] Cumulative GPA is {student['GPA']}/10. Recommend peer tutoring and remedial study groups.")
    if student['Online_Learning_Hours'] < 8.0:
        actions.append(f"[LMS DISENGAGEMENT] Only {student['Online_Learning_Hours']} hrs/week active on LMS. Issue portal reminder.")
    if student['Lab_Performance'] < 60.0:
        actions.append(f"[LAB PRACTICAL HAZARD] Practical lab score is {student['Lab_Performance']}/100. Assign teaching assistant support.")
        
    if not actions:
        actions.append("[OPTIMAL PROGRESS] Student maintains strong engagement and mastery. Encourage honors/research involvement.")
        
    return actions

def sanitize_student_input(raw: dict) -> dict:
    """Sanitizes and enforces strict types for inference input."""
    return {
        'Age': int(raw.get('Age', 20)),
        'Gender': str(raw.get('Gender', 'Male')).capitalize().strip(),
        'Socio_Economic_Status': str(raw.get('Socio_Economic_Status', 'Medium')).capitalize().strip(),
        'GPA': round(float(raw.get('GPA', 7.5)), 2),
        'Attendance_Rate': round(float(raw.get('Attendance_Rate', 80.0)), 2),
        'Assignments_Submitted': int(raw.get('Assignments_Submitted', 8)),
        'Online_Learning_Hours': round(float(raw.get('Online_Learning_Hours', 15.0)), 2),
        'Forum_Participation': int(raw.get('Forum_Participation', 10)),
        'Lab_Performance': round(float(raw.get('Lab_Performance', 75.0)), 2),
        'Extracurricular_Participation': str(raw.get('Extracurricular_Participation', 'No')).capitalize().strip()
    }

def predict_student(student_dict: dict):
    student_dict = sanitize_student_input(student_dict)
    model, preprocessor = load_inference_artifacts()
    df = pd.DataFrame([student_dict])
    X_proc = preprocessor.transform(df)
    
    # 1. Unit 2 Bayesian Predictive Probability & Epistemic Uncertainty
    if hasattr(model, 'predict_proba'):
        prob = float(model.predict_proba(X_proc)[0, 1])
    else:
        prob = float(model.predict(X_proc)[0])
        
    # Epistemic variance if Bayesian model
    uncertainty = None
    if hasattr(model, 'get_predictive_uncertainty'):
        uncertainty = round(float(model.get_predictive_uncertainty(X_proc)[0]), 4)
        
    is_risk = int(prob >= 0.50)
    engagement_idx = compute_engagement_index(student_dict)
    
    # 2. Unit 4 Temporal Trajectory Simulation & Viterbi Decoding
    hmm = HiddenMarkovModel()
    att = float(student_dict['Attendance_Rate'])
    lms = float(student_dict['Online_Learning_Hours'])
    
    # Synthesize 10-week observational tokens
    tokens = []
    for w in range(10):
        drop = (w / 10.0) * (0.35 if is_risk else 0.05)
        p_high = np.clip((att / 100.0) * 0.7 + (lms / 35.0) * 0.3 - drop, 0.05, 0.90)
        p_low = np.clip((1.0 - p_high) * (0.75 if is_risk else 0.25), 0.05, 0.85)
        p_med = np.clip(1.0 - p_high - p_low, 0.05, 0.90)
        probs = np.array([p_high, p_med, p_low]) / (p_high + p_med + p_low)
        tokens.append(int(np.random.choice([0, 1, 2], p=probs)))
        
    decoded_path, decoded_names, _, _ = hmm.viterbi(tokens)
    risk_weeks = [w + 1 for w, s in enumerate(decoded_path) if s == 2]
    earliest_risk_week = risk_weeks[0] if risk_weeks else -1
    
    actions = generate_intervention_plan(student_dict, prob, earliest_risk_week)
    
    return {
        'at_risk_status': 'AT RISK (Intervention Needed)' if is_risk == 1 else 'SAFE (Satisfactory Progress)',
        'is_at_risk': bool(is_risk == 1),
        'risk_probability': round(float(prob * 100), 2),
        'epistemic_uncertainty': uncertainty,
        'engagement_index': float(engagement_idx),
        'viterbi_decoded_trajectory': decoded_names,
        'earliest_risk_detected_week': earliest_risk_week if earliest_risk_week > 0 else 'None (Remained in Safe/Warning band)',
        'intervention_plan': actions
    }

def run_demo():
    print("=" * 80)
    print(" ACADEMIC RISK & ENGAGEMENT EARLY ALERT PREDICTOR")
    print(" Powered by Bayesian Inference (Unit 2) & Viterbi Decoding (Unit 4)")
    print("=" * 80)
    
    sample_students = [
        {
            'name': 'Student A (High Hazard Profile)',
            'data': {
                'Age': 21,
                'Gender': 'Male',
                'Socio_Economic_Status': 'Low',
                'GPA': 5.38,
                'Attendance_Rate': 58.0,
                'Assignments_Submitted': 4,
                'Online_Learning_Hours': 5.2,
                'Forum_Participation': 2,
                'Lab_Performance': 52.0,
                'Extracurricular_Participation': 'No'
            }
        },
        {
            'name': 'Student B (High Performing / Safe Profile)',
            'data': {
                'Age': 20,
                'Gender': 'Female',
                'Socio_Economic_Status': 'High',
                'GPA': 9.62,
                'Attendance_Rate': 94.0,
                'Assignments_Submitted': 10,
                'Online_Learning_Hours': 22.5,
                'Forum_Participation': 24,
                'Lab_Performance': 91.0,
                'Extracurricular_Participation': 'Yes'
            }
        },
        {
            'name': 'Student C (Borderline Vulnerable Profile)',
            'data': {
                'Age': 22,
                'Gender': 'Male',
                'Socio_Economic_Status': 'Medium',
                'GPA': 6.75,
                'Attendance_Rate': 72.5,
                'Assignments_Submitted': 6,
                'Online_Learning_Hours': 9.0,
                'Forum_Participation': 5,
                'Lab_Performance': 68.0,
                'Extracurricular_Participation': 'No'
            }
        }
    ]
    
    for item in sample_students:
        name = item['name']
        data = item['data']
        res = predict_student(data)
        
        print(f"\nEvaluating: {name}")
        print("-" * 65)
        print(f"  Inputs: GPA={data['GPA']} | Attendance={data['Attendance_Rate']}% | Assignments={data['Assignments_Submitted']}/10 | LMS={data['Online_Learning_Hours']} hrs")
        print(f"  Risk Status:                {res['at_risk_status']}")
        print(f"  Bayesian Risk Probability:  {res['risk_probability']}% (Epistemic Variance: {res['epistemic_uncertainty']})")
        print(f"  Engagement Index:           {res['engagement_index']} / 100")
        print(f"  Viterbi Early Alert Week:   Week {res['earliest_risk_detected_week']}")
        print(f"  Terminal Decoded State:     {res['viterbi_decoded_trajectory'][-1]}")
        print(f"  Actionable Pedagogical Plan:")
        for action in res['intervention_plan']:
            print(f"    -> {action}")

if __name__ == '__main__':
    run_demo()
