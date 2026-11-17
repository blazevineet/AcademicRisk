"""
Master Experimentation & Benchmarking Orchestrator
Academic Risk & Engagement Prediction System
Explicit Units 1 to 5 Syllabus:
  Unit 1: Polynomial Curve Fitting
  Unit 2: Bayesian Logistic Regression, SVM
  Unit 3: K-Means, PCA, Hierarchical Clustering
  Unit 4: HMM, Viterbi
  Unit 5: CART, Ensemble Learning
"""

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import warnings
warnings.filterwarnings('ignore')
import sys
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import (
    load_data, prepare_data, get_eda_summary, compute_engagement_score
)
from data.dataset_generator import generate_academic_risk_dataset
from src.unit1_polynomial_curve_fitting import Unit1PolynomialStudy
from src.unit2_bayesian_lr_svm import Unit2BayesianLrSvmModels
from src.unit3_clustering_pca import Unit3ClusteringPcaModels
from src.unit4_hmm_viterbi import evaluate_hmm_viterbi_pipeline, HiddenMarkovModel
from src.unit5_cart_ensemble import Unit5CartEnsembleModels
from src.evaluate import run_comprehensive_comparison

from src.dashboard_template import generate_full_dashboard

def generate_html_dashboard(df_ranked, eda_summary, poly_df, cluster_profiles, pca_summary, hmm_metrics, top_features, champion):
    """Generates visual HTML dashboard with integrated real-time interactive predictor."""
    return generate_full_dashboard(
        df_ranked, eda_summary, poly_df, cluster_profiles, pca_summary, hmm_metrics, top_features, champion
    )

def main():
    print("=" * 80)
    print(" ACADEMIC RISK & ENGAGEMENT PREDICTION SYSTEM")
    print(" Explicit Units 1 to 5 Syllabus Implementation & Comparative Benchmark")
    print("=" * 80)

    # 1. Dataset Verification
    data_path = os.path.join(BASE_DIR, 'data', 'academic_risk_data.csv')
    if not os.path.exists(data_path):
        df = generate_academic_risk_dataset(n_samples=1500, random_state=42)
        df.to_csv(data_path, index=False)
    else:
        df = load_data(data_path)

    eda = get_eda_summary(df)
    print(f"\n--- REQUIREMENT 3: DATASET DETAILS ---")
    print(f"Total Student Cohort: {eda['total_samples']} records")
    print(f"Attributes: {list(df.columns)}")
    print(f"Target Distribution: {eda['at_risk_ratio']}% At Risk ({eda['at_risk_count']} At Risk / {eda['safe_count']} Safe)")
    
    splits = prepare_data(df, test_size=0.2, random_state=42)
    X_train_proc = splits['X_train_proc']
    X_test_proc = splits['X_test_proc']
    y_train = splits['y_train']
    y_test = splits['y_test']
    y_train_eng = splits['y_train_eng']
    y_test_eng = splits['y_test_eng']
    feature_names = splits['feature_names']
    
    # Save preprocessor
    joblib.dump(splits['preprocessor'], os.path.join(BASE_DIR, 'models', 'preprocessor.joblib'))

    # =========================================================================
    # UNIT 1: POLYNOMIAL CURVE FITTING
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENT 1: UNIT 1 (POLYNOMIAL CURVE FITTING) ---")
    # Fit student Engagement vs Online Learning Hours & Attendance
    poly_study = Unit1PolynomialStudy()
    x_tr_eng = splits['X_train_raw']['Online_Learning_Hours'].values
    x_te_eng = splits['X_test_raw']['Online_Learning_Hours'].values
    poly_results = poly_study.run_study(x_tr_eng, y_train_eng, x_te_eng, y_test_eng)
    print("\nPolynomial Curve Fitting (Bishop PRML Ch. 1) Results across Degree M:")
    print(poly_results.to_string(index=False))

    # =========================================================================
    # UNIT 2: BAYESIAN LOGISTIC REGRESSION & SVM
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENT 1: UNIT 2 (BAYESIAN LOGISTIC REGRESSION & SVM) ---")
    u2 = Unit2BayesianLrSvmModels(random_state=42)
    u2.train_all(X_train_proc, y_train)
    u2_res = u2.evaluate_all(X_test_proc, y_test)
    for name, r in u2_res.items():
        print(f"  [{name}] Acc: {r['Accuracy']:.4f} | Recall: {r['Recall']:.4f} | ROC-AUC: {r['ROC_AUC']:.4f}")

    # =========================================================================
    # UNIT 3: K-MEANS, PCA, HIERARCHICAL CLUSTERING
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENT 1: UNIT 3 (K-MEANS, PCA & HIERARCHICAL CLUSTERING) ---")
    u3 = Unit3ClusteringPcaModels(n_clusters=3, random_state=42)
    clust_labels = u3.fit_clustering(X_train_proc)
    u3_eval = u3.evaluate_clustering(X_train_proc, clust_labels, y_true=y_train)
    for name, r in u3_eval.items():
        print(f"  [{name}] Silhouette: {r['Silhouette_Score']:.4f} | Davies-Bouldin: {r['Davies_Bouldin_Index']:.4f} | ARI: {r['Adjusted_Rand_Index']:.4f}")
    
    X_pca_2d, pca_summary = u3.fit_pca(X_train_proc)
    print(f"  PCA 2D Variance Retained: {pca_summary['total_2d_variance_retained']*100:.2f}% (PC1: {pca_summary['pc1_variance_ratio']*100:.2f}%, PC2: {pca_summary['pc2_variance_ratio']*100:.2f}%)")
    cluster_profiles = u3.profile_clusters(splits['X_train_raw'].assign(At_Risk=y_train), clust_labels['KMeans'])

    # =========================================================================
    # UNIT 4: HMM & VITERBI ALGORITHM
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENT 1: UNIT 4 (HMM & VITERBI DECODING) ---")
    hmm_obj, hmm_metrics, earliest_weeks = evaluate_hmm_viterbi_pipeline(df)
    print(f"  Viterbi Sequence Accuracy:        {hmm_metrics['Viterbi_Sequence_Accuracy']*100:.2f}%")
    print(f"  Viterbi At-Risk Recall:           {hmm_metrics['Viterbi_Recall']*100:.2f}%")
    print(f"  Viterbi F1-Score:                 {hmm_metrics['Viterbi_F1_Score']*100:.2f}%")
    print(f"  Average Earliest Alert Week:      Week {hmm_metrics['Avg_Earliest_Detection_Week']}")
    print(f"  Mean Sequence Log-Likelihood:     {hmm_metrics['Mean_Sequence_Log_Likelihood']}")

    # =========================================================================
    # UNIT 5: CART & ENSEMBLE LEARNING
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENT 1: UNIT 5 (CART & ENSEMBLE LEARNING) ---")
    u5 = Unit5CartEnsembleModels(random_state=42)
    u5.train_all(X_train_proc, y_train)
    u5_res = u5.evaluate_all(X_test_proc, y_test)
    for name, r in u5_res.items():
        print(f"  [{name}] Acc: {r['Accuracy']:.4f} | Recall: {r['Recall']:.4f} | ROC-AUC: {r['ROC_AUC']:.4f}")

    # =========================================================================
    # MASTER BENCHMARK & RANKING (REQUIREMENTS 2 & 4)
    # =========================================================================
    print("\n" + "=" * 80)
    print("--- REQUIREMENTS 2 & 4: MASTER ALGORITHM COMPARISON & RANKING ---")
    
    all_models = {
        ('Unit 2', 'Bayesian Logistic Regression'): u2.bayes_lr,
        ('Unit 2', 'SVM (Linear Kernel)'): u2.svm_linear,
        ('Unit 2', 'SVM (RBF Kernel)'): u2.svm_rbf,
        ('Unit 2', 'SVM (Poly Kernel)'): u2.svm_poly,
        ('Unit 5', 'CART (Gini Impurity)'): u5.cart_gini,
        ('Unit 5', 'CART (Entropy)'): u5.cart_entropy,
        ('Unit 5', 'Random Forest'): u5.rf,
        ('Unit 5', 'AdaBoost'): u5.adaboost,
        ('Unit 5', 'Gradient Boosting'): u5.gbdt,
        ('Unit 5', 'XGBoost'): u5.xgb,
        ('Unit 5', 'Voting Classifier'): u5.voting,
    }

    df_display, df_full, champion = run_comprehensive_comparison(
        all_models, X_train_proc, y_train, X_test_proc, y_test, random_state=42
    )

    # Save CSV and JSON
    comparison_csv = os.path.join(BASE_DIR, 'results', 'model_comparison_summary.csv')
    df_display.to_csv(comparison_csv, index=False)
    
    with open(os.path.join(BASE_DIR, 'results', 'classification_report.json'), 'w') as f:
        json.dump(df_full.to_dict(orient='records'), f, indent=4)

    pd.set_option('display.max_columns', 15)
    pd.set_option('display.width', 125)
    print("\n" + df_display.to_string(index=False))

    # Champion model persistence
    champ_unit = champion['Unit']
    champ_algo = champion['Algorithm']
    champion_model_obj = all_models[(champ_unit, champ_algo)]
    joblib.dump(champion_model_obj, os.path.join(BASE_DIR, 'models', 'best_model.joblib'))

    # Extract feature importances
    top_features = u5.get_rf_importances(feature_names)

    # Generate High-Resolution Visual Graphs (PNGs)
    try:
        from src.generate_plots import main as generate_all_plots
        generate_all_plots()
    except Exception as e:
        print(f"Plot generation note: {e}")

    # Generate HTML Dashboard
    html = generate_html_dashboard(
        df_display, eda, poly_results, cluster_profiles, pca_summary, hmm_metrics, top_features, champion
    )
    with open(os.path.join(BASE_DIR, 'results', 'report_dashboard.html'), 'w', encoding='utf-8') as f:
        f.write(html)

    print("\n" + "=" * 80)
    print("--- REQUIREMENT 4: CHAMPION ALGORITHM IDENTIFICATION ---")
    print(f"Champion Algorithm: {champ_algo} ({champ_unit})")
    print(f"  * Accuracy:             {champion['Accuracy']*100:.2f}%")
    print(f"  * Recall / Sensitivity: {champion['Recall']*100:.2f}% (Critical for early intervention)")
    print(f"  * Precision:            {champion['Precision']*100:.2f}%")
    print(f"  * F1-Score:             {champion['F1_Score']*100:.2f}%")
    print(f"  * ROC-AUC:              {champion['ROC_AUC']*100:.2f}%")
    print(f"  * 5-Fold CV ROC-AUC:    {champion['CV_ROC_Mean']*100:.2f}% (Std: +/-{champion['CV_ROC_Std']*100:.2f}%)")
    print(f"  * False Negative Rate:  {champion['FNR']*100:.2f}% ({champion['FN']} students missed)")
    print("=" * 80)

if __name__ == '__main__':
    main()
