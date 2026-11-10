"""
Model Evaluation, Cross-Comparison, and Champion Identification Module
Consolidates results across Units 1 to 5, computes comparative metrics,
and ranks models to identify the top performer for Academic Risk & Engagement Prediction.
"""

import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate

def benchmark_model(model, X_train, y_train, X_test, y_test, cv_folds: int = 5, random_state: int = 42):
    """
    Fits and evaluates a model, recording latency, test metrics, confusion matrix, and K-Fold CV.
    """
    # Measure Training Time
    t_start = time.perf_counter()
    model.fit(X_train, y_train)
    train_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    
    # Measure Inference Time
    t_start = time.perf_counter()
    y_pred = model.predict(X_test)
    inference_time_ms = round((time.perf_counter() - t_start) * 1000, 2)
    
    # Compute Probabilities
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        df_vals = model.decision_function(X_test)
        y_prob = 1.0 / (1.0 + np.exp(-df_vals))
    else:
        y_prob = y_pred.astype(float)
        
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Efficient Stratified K-Fold CV (fits model only once per fold)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    X_full = np.vstack((X_train, X_test))
    y_full = np.concatenate((y_train, y_test))
    
    cv_res = cross_validate(
        model, X_full, y_full, cv=cv,
        scoring=['accuracy', 'roc_auc', 'f1'],
        n_jobs=1
    )
    
    # Metrics
    metrics = {
        'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
        'Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        'Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        'F1_Score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        'ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
        'Log_Loss': round(float(log_loss(y_test, y_prob)), 4) if y_prob is not None else None,
        'TP': int(tp),
        'FP': int(fp),
        'TN': int(tn),
        'FN': int(fn),
        'FNR': round(float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0, 4),  # False Negative Rate (critical)
        'Train_Time_ms': train_time_ms,
        'Infer_Time_ms': inference_time_ms,
        'CV_Acc_Mean': round(float(cv_res['test_accuracy'].mean()), 4),
        'CV_Acc_Std': round(float(cv_res['test_accuracy'].std()), 4),
        'CV_ROC_Mean': round(float(cv_res['test_roc_auc'].mean()), 4),
        'CV_ROC_Std': round(float(cv_res['test_roc_auc'].std()), 4),
        'CV_F1_Mean': round(float(cv_res['test_f1'].mean()), 4),
    }
    return metrics

def run_comprehensive_comparison(models_dict, X_train, y_train, X_test, y_test, random_state: int = 42):
    """
    Runs benchmarking across all models from Units 1 to 5 and creates a ranked comparative DataFrame.
    """
    rows = []
    for (unit_label, model_name), model in models_dict.items():
        metrics = benchmark_model(model, X_train, y_train, X_test, y_test, random_state=random_state)
        entry = {
            'Unit': unit_label,
            'Algorithm': model_name,
            **metrics
        }
        rows.append(entry)
        
    df_results = pd.DataFrame(rows)
    
    # Composite Score for Ranking in Educational Risk:
    # 40% ROC-AUC + 35% F1-Score + 25% CV_ROC_Mean (emphasizing early alert reliability)
    df_results['Rank_Score'] = (
        0.40 * df_results['ROC_AUC'] +
        0.35 * df_results['F1_Score'] +
        0.25 * df_results['CV_ROC_Mean']
    ).round(4)
    
    df_results = df_results.sort_values(by='Rank_Score', ascending=False).reset_index(drop=True)
    df_results['Rank'] = range(1, len(df_results) + 1)
    
    # Rearrange columns
    cols_order = [
        'Rank', 'Unit', 'Algorithm', 'Accuracy', 'Precision', 'Recall',
        'F1_Score', 'ROC_AUC', 'CV_Acc_Mean', 'CV_ROC_Mean', 'FNR',
        'Train_Time_ms', 'Infer_Time_ms', 'Rank_Score'
    ]
    df_display = df_results[cols_order]
    
    champion = df_results.iloc[0].to_dict()
    return df_display, df_results, champion
