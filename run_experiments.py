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

def generate_html_dashboard(df_ranked, eda_summary, poly_df, cluster_profiles, pca_summary, hmm_metrics, top_features, champion):
    """Generates standalone visual HTML dashboard."""
    table_rows = ""
    for _, row in df_ranked.iterrows():
        is_champ = row['Algorithm'] == champion['Algorithm']
        row_class = 'class="champion-row"' if is_champ else ''
        badge = '<span class="badge champion-badge">CHAMPION</span> ' if is_champ else ''
        table_rows += f"""
        <tr {row_class}>
            <td><strong>#{row['Rank']}</strong></td>
            <td><span class="unit-tag">{row['Unit']}</span></td>
            <td><strong>{badge}{row['Algorithm']}</strong></td>
            <td>{row['Accuracy']*100:.2f}%</td>
            <td>{row['Precision']*100:.2f}%</td>
            <td>{row['Recall']*100:.2f}%</td>
            <td><strong>{row['F1_Score']*100:.2f}%</strong></td>
            <td><strong>{row['ROC_AUC']*100:.2f}%</strong></td>
            <td>{row['CV_ROC_Mean']*100:.2f}%</td>
            <td>{row['FNR']*100:.2f}%</td>
            <td>{row['Train_Time_ms']:.1f} ms</td>
        </tr>
        """
        
    poly_rows = ""
    for _, row in poly_df.iterrows():
        poly_rows += f"""
        <tr>
            <td><strong>M = {row['Degree_M']}</strong></td>
            <td>{row['Reg_Lambda']}</td>
            <td>{row['Train_E_RMS']:.4f}</td>
            <td>{row['Test_E_RMS']:.4f}</td>
            <td>{row['Test_R2']:.4f}</td>
            <td>{row['Weight_Norm_||w||']}</td>
            <td><span class="status-tag">{row['Status']}</span></td>
        </tr>
        """

    feature_svg_bars = ""
    max_imp = max(f['importance'] for f in top_features) if top_features else 1.0
    for i, feat in enumerate(top_features[:8]):
        y_pos = 35 + i * 40
        width = int((feat['importance'] / max_imp) * 380)
        pct = feat['importance'] * 100
        feature_svg_bars += f"""
        <text x="10" y="{y_pos + 15}" font-size="13" font-family="Segoe UI, sans-serif" fill="#2d3748">{feat['feature']}</text>
        <rect x="220" y="{y_pos}" width="{width}" height="22" rx="4" fill="#3182ce" opacity="0.85" />
        <text x="{230 + width}" y="{y_pos + 16}" font-size="12" font-family="Segoe UI, sans-serif" font-weight="bold" fill="#2b6cb0">{pct:.1f}%</text>
        """

    # Dynamically rank clusters by mean GPA descending (Highest GPA -> Tier 1, Lowest GPA -> Tier 3)
    sorted_clusters = sorted(
        cluster_profiles.items(),
        key=lambda item: item[1].get('GPA', 0.0),
        reverse=True
    )
    
    tier_meta = [
        ("Tier 1: High Engagement / Low Risk", "#2f855a"),
        ("Tier 2: Moderate Engagement / Warning Zone", "#dd6b20"),
        ("Tier 3: Severe Disengagement / Critical Risk", "#e53e3e")
    ]
    
    cluster_cards = ""
    for i, (cid, metrics) in enumerate(sorted_clusters):
        name, color = tier_meta[i % len(tier_meta)]
        cluster_cards += f"""
        <div class="card" style="border-top: 4px solid {color};">
            <h4 style="color: {color}; margin-top:0;">{name}</h4>
            <div style="font-size: 13px; color: #4a5568;">
                <p><strong>Avg GPA:</strong> {metrics.get('GPA', 'N/A')}</p>
                <p><strong>Avg Attendance:</strong> {metrics.get('Attendance_Rate', 'N/A')}%</p>
                <p><strong>Avg Assignments:</strong> {metrics.get('Assignments_Submitted', 'N/A')}/10</p>
                <p><strong>Online LMS Hours:</strong> {metrics.get('Online_Learning_Hours', 'N/A')} hrs/wk</p>
                <p><strong>Actual At-Risk Rate:</strong> <span style="font-size:16px; font-weight:bold; color:{color};">{metrics.get('At_Risk_Rate', 'N/A')}%</span></p>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Academic Risk & Engagement Prediction - Units 1-5 Benchmark</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f7fafc;
            color: #2d3748;
            margin: 0;
            padding: 24px;
        }}
        .container {{ max-width: 1250px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1a365d 0%, #2b6cb0 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .champion-banner {{
            background: #ebf8ff;
            border: 2px solid #3182ce;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .champion-badge {{
            background: #3182ce;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}
        .unit-tag {{
            background: #edf2f7;
            color: #4a5568;
            padding: 3px 7px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .status-tag {{
            background: #feebc8;
            color: #7b341e;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .card {{
            background: white;
            padding: 22px;
            border-radius: 10px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
            border: 1px solid #e2e8f0;
            margin-bottom: 24px;
        }}
        .card h3 {{ margin-top: 0; font-size: 17px; color: #1a202c; border-bottom: 2px solid #edf2f7; padding-bottom: 8px; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        .grid-3 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background-color: #edf2f7;
            color: #4a5568;
            padding: 10px 8px;
            font-weight: 600;
            border-bottom: 2px solid #cbd5e0;
        }}
        td {{
            padding: 9px 8px;
            border-bottom: 1px solid #e2e8f0;
        }}
        tr:hover {{ background-color: #f8fafc; }}
        .champion-row {{
            background-color: #ebf8ff !important;
            font-weight: 500;
        }}
        .stat-box {{
            display: inline-block;
            background: #f7fafc;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
            margin-right: 10px;
            margin-bottom: 10px;
            min-width: 120px;
        }}
        .stat-label {{ font-size: 10px; color: #718096; text-transform: uppercase; font-weight: 600; }}
        .stat-val {{ font-size: 20px; font-weight: bold; color: #2d3748; margin-top: 2px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Academic Risk & Engagement Prediction System</h1>
            <p>Syllabus Units 1 to 5: Polynomial Curve Fitting | Bayesian Logistic Regression & SVM | Clustering & PCA | HMM & Viterbi | CART & Ensembles</p>
        </div>

        <div class="champion-banner">
            <div>
                <span class="champion-badge">WINNING ALGORITHM</span>
                <h2 style="margin: 8px 0 4px 0; color: #2b6cb0;">{champion['Algorithm']} ({champion['Unit']})</h2>
                <p style="margin: 0; color: #4a5568; font-size: 13px;">
                    Attained <strong>{champion['Accuracy']*100:.2f}% Accuracy</strong>, 
                    <strong>{champion['Recall']*100:.2f}% Sensitivity</strong>, and <strong>{champion['ROC_AUC']*100:.2f}% ROC-AUC</strong> 
                    across 5-Fold Stratified Cross-Validation.
                </p>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 34px; font-weight: bold; color: #2b6cb0;">{champion['ROC_AUC']*100:.1f}%</div>
                <div style="font-size: 11px; color: #718096; font-weight: bold;">TEST ROC-AUC</div>
            </div>
        </div>

        <div class="card">
            <h3>Requirement 2 & 4: Master Algorithm Leaderboard (Units 2, 4, 5 Classification)</h3>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Unit</th>
                            <th>Algorithm</th>
                            <th>Accuracy</th>
                            <th>Precision</th>
                            <th>Recall</th>
                            <th>F1-Score</th>
                            <th>ROC-AUC</th>
                            <th>5-Fold CV ROC</th>
                            <th>False Neg. Rate</th>
                            <th>Train Latency</th>
                        </tr>
                    </thead>
                    <tbody>
                        {table_rows}
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <img src="algorithm_comparison.png" alt="Algorithm Comparison Chart" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            </div>
        </div>

        <div class="card">
            <h3>Unit 1: Polynomial Curve Fitting (Bishop PRML Ch. 1)</h3>
            <p style="font-size: 13px; color: #718096;">
                Analysis of polynomial degree M, Root-Mean-Square Error (E_RMS), weight vector norm ||w||, and regularization (weight decay lambda):
            </p>
            <div style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Degree M</th>
                            <th>Reg. Lambda</th>
                            <th>Train E_RMS</th>
                            <th>Test E_RMS</th>
                            <th>Test R²</th>
                            <th>Weight Norm ||w||</th>
                            <th>Empirical Diagnosis</th>
                        </tr>
                    </thead>
                    <tbody>
                        {poly_rows}
                    </tbody>
                </table>
            </div>
            <div style="margin-top: 20px; text-align: center;">
                <img src="polynomial_curve_fitting.png" alt="Polynomial Curve Fitting Graph" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            </div>
        </div>

        <div class="card">
            <h3>Error Analysis: Confusion Matrices Across Key Models</h3>
            <div style="margin-top: 10px; text-align: center;">
                <img src="confusion_matrices.png" alt="Confusion Matrices" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h3>Unit 4: HMM & Viterbi Dynamic Decoding</h3>
                <div class="stat-box">
                    <div class="stat-label">Sequence Accuracy</div>
                    <div class="stat-val">{hmm_metrics['Viterbi_Sequence_Accuracy']*100:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Viterbi Recall</div>
                    <div class="stat-val">{hmm_metrics['Viterbi_Recall']*100:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Viterbi F1</div>
                    <div class="stat-val">{hmm_metrics['Viterbi_F1_Score']*100:.1f}%</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Avg Alert Week</div>
                    <div class="stat-val">Week {hmm_metrics['Avg_Earliest_Detection_Week']}</div>
                </div>
                <div style="margin-top: 15px; text-align: center;">
                    <img src="viterbi_trajectories.png" alt="Viterbi Weekly Trajectories" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
            </div>

            <div class="card">
                <h3>Unit 5: Feature Relevance Ranking (CART & Ensembles)</h3>
                <div style="margin-top: 10px; text-align: center;">
                    <img src="feature_importance.png" alt="Feature Importance Ranking" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
                </div>
            </div>
        </div>

        <div class="card">
            <h3>Unit 3: K-Means Student Persona Profiling (Unsupervised Discovery & PCA)</h3>
            <p style="font-size: 13px; color: #718096; margin-bottom: 16px;">
                PCA retains <strong>{pca_summary['total_2d_variance_retained']*100:.1f}%</strong> of total multi-attribute variance in 2D. K-Means segments the cohort into 3 tiers:
            </p>
            <div style="margin-top: 10px; margin-bottom: 20px; text-align: center;">
                <img src="pca_clusters.png" alt="PCA Clusters" style="max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #e2e8f0;">
            </div>
            <div class="grid-3">
                {cluster_cards}
            </div>
        </div>
    </div>
</body>
</html>
    """
    return html

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
