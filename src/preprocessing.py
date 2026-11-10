"""
Data Preprocessing and Pipeline Management
Module handles data loading, exploratory statistics, categorical encoding,
feature scaling, and train-test partitioning for the Academic Risk & Engagement System.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

NUMERICAL_COLS = [
    'Age', 'GPA', 'Attendance_Rate', 'Assignments_Submitted',
    'Online_Learning_Hours', 'Forum_Participation', 'Lab_Performance'
]

CATEGORICAL_COLS = [
    'Gender', 'Socio_Economic_Status', 'Extracurricular_Participation'
]

TARGET_COL = 'At_Risk'

def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Computes a continuous composite Engagement Score (0 to 100)
    combining behavioral indicators: Attendance, LMS hours, Assignments, Forums, Extracurricular.
    Used for Unit 1 Regression modeling.
    """
    att_norm = (df['Attendance_Rate'] - 30.0) / 70.0 * 30.0  # max 30 pts
    assign_norm = (df['Assignments_Submitted'] / 10.0) * 25.0  # max 25 pts
    lms_norm = np.clip(df['Online_Learning_Hours'] / 30.0, 0, 1) * 20.0  # max 20 pts
    forum_norm = np.clip(df['Forum_Participation'] / 35.0, 0, 1) * 15.0  # max 15 pts
    extra_norm = np.where(df['Extracurricular_Participation'] == 'Yes', 10.0, 0.0)  # max 10 pts
    
    score = att_norm + assign_norm + lms_norm + forum_norm + extra_norm
    return pd.Series(np.clip(np.round(score, 2), 0.0, 100.0), name='Engagement_Score')

def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from path."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    df = pd.read_csv(filepath)
    # Validate expected columns
    expected = NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET_COL]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    return df

def get_preprocessor():
    """Build scikit-learn ColumnTransformer for scaling and one-hot encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_COLS),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), CATEGORICAL_COLS)
        ]
    )
    return preprocessor

def prepare_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """
    Prepares train and test splits with fitted preprocessors.
    Returns:
        X_train_raw, X_test_raw: Raw DataFrames
        X_train_proc, X_test_proc: Scaled and encoded numpy arrays
        y_train, y_test: Binary labels for At_Risk
        y_train_eng, y_test_eng: Continuous Engagement Scores
        feature_names: Transformed feature column names
        preprocessor: Fitted ColumnTransformer
    """
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df[TARGET_COL]
    y_eng = compute_engagement_score(df)
    
    X_train_raw, X_test_raw, y_train, y_test, y_train_eng, y_test_eng = train_test_split(
        X, y, y_eng, test_size=test_size, random_state=random_state, stratify=y
    )
    
    preprocessor = get_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_test_proc = preprocessor.transform(X_test_raw)
    
    # Extract feature names
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(CATEGORICAL_COLS).tolist()
    feature_names = NUMERICAL_COLS + cat_feature_names
    
    return {
        'X_train_raw': X_train_raw,
        'X_test_raw': X_test_raw,
        'X_train_proc': X_train_proc,
        'X_test_proc': X_test_proc,
        'y_train': y_train.values,
        'y_test': y_test.values,
        'y_train_eng': y_train_eng.values,
        'y_test_eng': y_test_eng.values,
        'feature_names': feature_names,
        'preprocessor': preprocessor
    }

def get_eda_summary(df: pd.DataFrame) -> dict:
    """Computes comprehensive dataset summary statistics."""
    summary = {
        'total_samples': len(df),
        'num_features': len(df.columns) - 1,
        'at_risk_count': int((df[TARGET_COL] == 1).sum()),
        'safe_count': int((df[TARGET_COL] == 0).sum()),
        'at_risk_ratio': float(round((df[TARGET_COL] == 1).mean() * 100, 2)),
        'numerical_stats': df[NUMERICAL_COLS].describe().round(2).to_dict(),
        'categorical_distributions': {
            col: df[col].value_counts().to_dict() for col in CATEGORICAL_COLS
        },
        'correlations_with_risk': df[NUMERICAL_COLS].apply(lambda c: c.corr(df[TARGET_COL])).round(4).to_dict()
    }
    return summary
