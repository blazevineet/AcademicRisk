"""
Unit 3: K-Means, PCA & Hierarchical Clustering
Concepts Covered:
1. K-Means Clustering (Lloyd's algorithm, Inertia, Silhouette Score, Davies-Bouldin Index)
2. Principal Component Analysis (PCA) (Covariance Eigen-decomposition, Explained Variance Ratio, 2D Projection)
3. Agglomerative Hierarchical Clustering (Ward's, Complete, and Average Linkage criteria)
4. Unsupervised Profiling of Student Personas and Cluster-to-Risk Agreement (Adjusted Rand Index)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score, homogeneity_score, completeness_score
)

class Unit3ClusteringPcaModels:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # 1. K-Means Clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        
        # 2a. Hierarchical Clustering (Ward's Linkage)
        self.hier_ward = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        
        # 2b. Hierarchical Clustering (Complete Linkage)
        self.hier_complete = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='complete'
        )
        
        # 3. PCA
        self.pca = PCA(n_components=2, random_state=random_state)
        self.pca_full = PCA(random_state=random_state)

    def fit_clustering(self, X):
        km_labels = self.kmeans.fit_predict(X)
        ward_labels = self.hier_ward.fit_predict(X)
        comp_labels = self.hier_complete.fit_predict(X)
        return {
            'KMeans': km_labels,
            'Hierarchical_Ward': ward_labels,
            'Hierarchical_Complete': comp_labels
        }

    def evaluate_clustering(self, X, labels_dict, y_true=None):
        metrics = {}
        for name, labels in labels_dict.items():
            entry = {
                'Silhouette_Score': round(float(silhouette_score(X, labels)), 4),
                'Davies_Bouldin_Index': round(float(davies_bouldin_score(X, labels)), 4),
                'Calinski_Harabasz_Index': round(float(calinski_harabasz_score(X, labels)), 2)
            }
            if y_true is not None:
                entry['Adjusted_Rand_Index'] = round(float(adjusted_rand_score(y_true, labels)), 4)
                entry['Homogeneity'] = round(float(homogeneity_score(y_true, labels)), 4)
                entry['Completeness'] = round(float(completeness_score(y_true, labels)), 4)
            metrics[name] = entry
        return metrics

    def fit_pca(self, X):
        self.pca_full.fit(X)
        X_pca_2d = self.pca.fit_transform(X)
        
        exp_var = self.pca_full.explained_variance_ratio_
        cum_var = np.cumsum(exp_var)
        
        pca_summary = {
            'pc1_variance_ratio': round(float(self.pca.explained_variance_ratio_[0]), 4),
            'pc2_variance_ratio': round(float(self.pca.explained_variance_ratio_[1]), 4),
            'total_2d_variance_retained': round(float(np.sum(self.pca.explained_variance_ratio_)), 4),
            'cumulative_variances': [round(float(v), 4) for v in cum_var]
        }
        return X_pca_2d, pca_summary

    def profile_clusters(self, df_raw, labels):
        df_copy = df_raw.copy()
        df_copy['Cluster'] = labels
        numeric_cols = [
            'GPA', 'Attendance_Rate', 'Assignments_Submitted',
            'Online_Learning_Hours', 'Forum_Participation', 'Lab_Performance'
        ]
        profile = df_copy.groupby('Cluster')[numeric_cols].mean().round(2).to_dict(orient='index')
        if 'At_Risk' in df_copy.columns:
            risk_rate = df_copy.groupby('Cluster')['At_Risk'].mean().round(4).to_dict()
            for k in profile:
                profile[k]['At_Risk_Rate'] = round(risk_rate[k] * 100, 2)
        return profile

def plot_unit3_visual(X_pca, km_labels, y_true, sorted_profiles, df_metrics, out_path):
    """Generates visual 4-panel comparison graph for Unit 3."""
    import matplotlib.pyplot as plt
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)
    
    # 1. Top Left: K-Means Clusters on PCA
    ax1 = axes[0, 0]
    colors = ['#2f855a', '#dd6b20', '#e53e3e']
    tier_names = ['Tier 1 (Safe)', 'Tier 2 (Warning)', 'Tier 3 (Critical)']
    
    for i, (cid, stats) in enumerate(sorted_profiles):
        mask = km_labels == int(cid)
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=f"{tier_names[i]} (GPA {stats['GPA']})", alpha=0.65, s=28)
        
    ax1.set_title('Unit 3: K-Means Clusters in 2D PCA Space', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Principal Component 1 (58.7% Variance)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Principal Component 2 (12.6% Variance)', fontsize=10, fontweight='bold')
    ax1.legend(loc='upper right', framealpha=0.95, fontsize=9)
    
    # 2. Top Right: Ground Truth Risk in PCA Space
    ax2 = axes[0, 1]
    ax2.scatter(X_pca[y_true == 0, 0], X_pca[y_true == 0, 1], c='#3182ce', label='Actual Safe (0)', alpha=0.6, s=28)
    ax2.scatter(X_pca[y_true == 1, 0], X_pca[y_true == 1, 1], c='#e53e3e', label='Actual At-Risk (1)', alpha=0.7, s=28)
    ax2.set_title('Ground-Truth Risk Labels in PCA Space', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Principal Component 1', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Principal Component 2', fontsize=10, fontweight='bold')
    ax2.legend(loc='upper right', framealpha=0.95, fontsize=10)
    
    # 3. Bottom Left: Clustering Metrics Comparison
    ax3 = axes[1, 0]
    methods = df_metrics['Clustering_Method'].tolist()
    x = np.arange(len(methods))
    width = 0.25
    
    ax3.bar(x - width, df_metrics['Silhouette_Score'], width, label='Silhouette Score', color='#3182ce')
    ax3.bar(x, df_metrics['Adjusted_Rand_Index'], width, label='Adjusted Rand Index (ARI)', color='#38a169')
    ax3.bar(x + width, df_metrics['Davies_Bouldin_Index'] / 3.0, width, label='Davies-Bouldin / 3', color='#dd6b20')
    
    ax3.set_ylabel('Metric Index', fontsize=10, fontweight='bold')
    ax3.set_title('Clustering Validation & Risk Agreement', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(methods, fontsize=9)
    ax3.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    # 4. Bottom Right: Student Persona Centroids Comparison
    ax4 = axes[1, 1]
    features_to_plot = ['GPA_x10', 'Attendance_%', 'Assignments_x10', 'LMS_Hours']
    x_feat = np.arange(len(features_to_plot))
    w_bar = 0.25
    
    for i, (cid, stats) in enumerate(sorted_profiles):
        norm_vals = [
            stats['GPA'] * 10.0,  # scale 10 to 100
            stats['Attendance_Rate'],
            stats['Assignments_Submitted'] * 10.0,  # scale 10 to 100
            stats['Online_Learning_Hours'] * 3.0   # scale 35 to ~100
        ]
        ax4.bar(x_feat + (i - 1) * w_bar, norm_vals, w_bar, label=tier_names[i], color=colors[i], alpha=0.9)
        
    ax4.set_ylabel('Normalized Score (0 - 100)', fontsize=10, fontweight='bold')
    ax4.set_title('Persona Profiles Across Discovered Tiers', fontsize=12, fontweight='bold')
    ax4.set_xticks(x_feat)
    ax4.set_xticklabels(['GPA (scaled)', 'Attendance %', 'Assignments (scaled)', 'LMS Hours (scaled)'], fontsize=9)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

if __name__ == '__main__':
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    import warnings
    warnings.filterwarnings('ignore')
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        
    from src.preprocessing import load_data, prepare_data
    
    data_path = os.path.join(base_dir, 'data', 'academic_risk_data.csv')
    df = load_data(data_path)
    splits = prepare_data(df, test_size=0.2, random_state=42)
    
    print("=" * 75)
    print(" UNIT 3: K-MEANS, PCA & HIERARCHICAL CLUSTERING")
    print("=" * 75)
    
    u3 = Unit3ClusteringPcaModels(n_clusters=3, random_state=42)
    print("\n[+] Fitting K-Means and Hierarchical Clustering (Ward & Complete Linkage)...")
    labels_dict = u3.fit_clustering(splits['X_train_proc'])
    
    metrics = u3.evaluate_clustering(splits['X_train_proc'], labels_dict, y_true=splits['y_train'])
    df_metrics = pd.DataFrame(metrics).T.reset_index().rename(columns={'index': 'Clustering_Method'})
    
    print("\n[+] Clustering Validation & Risk Alignment Metrics:")
    print(df_metrics.to_string(index=False))
    
    X_pca_2d, pca_sum = u3.fit_pca(splits['X_train_proc'])
    print(f"\n[+] PCA Dimensionality Reduction:")
    print(f"  * PC1 Variance Explained: {pca_sum['pc1_variance_ratio']*100:.2f}%")
    print(f"  * PC2 Variance Explained: {pca_sum['pc2_variance_ratio']*100:.2f}%")
    print(f"  * Cumulative 2D Variance Retained: {pca_sum['total_2d_variance_retained']*100:.2f}%")
    
    profiles = u3.profile_clusters(splits['X_train_raw'].assign(At_Risk=splits['y_train']), labels_dict['KMeans'])
    print("\n[+] Unsupervised Discovered Student Personas (Ranked by Academic Engagement):")
    sorted_profiles = sorted(profiles.items(), key=lambda item: item[1].get('GPA', 0.0), reverse=True)
    tier_labels = ["Tier 1 (High Engagement / Safe)", "Tier 2 (Moderate Engagement / Warning)", "Tier 3 (Disengaged / Critical Risk)"]
    for i, (cid, stats) in enumerate(sorted_profiles):
        tier = tier_labels[i]
        print(f"  {tier} [K-Means Cluster ID {cid}]:")
        print(f"    Avg GPA={stats['GPA']} | Attendance={stats['Attendance_Rate']}% | Assignments={stats['Assignments_Submitted']}/10 | LMS={stats['Online_Learning_Hours']} hrs | Actual Risk Rate={stats['At_Risk_Rate']}%")
        
    # Save & Open Graph
    plot_file = os.path.join(base_dir, 'results', 'unit3_clustering_pca.png')
    plot_unit3_visual(X_pca_2d, labels_dict['KMeans'], splits['y_train'], sorted_profiles, df_metrics, plot_file)
    print(f"\n[+] Visual comparison graph saved to: {plot_file}")
    
    try:
        os.startfile(plot_file)
        print("[+] Opening graph in your default image viewer...")
    except Exception:
        pass
    print("=" * 75)
