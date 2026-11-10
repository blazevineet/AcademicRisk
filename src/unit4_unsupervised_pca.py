"""
Unit 4: Unsupervised Learning & Dimensionality Reduction
Concepts Covered:
1. K-Means Clustering
   - Centroid-based clustering, Lloyd's algorithm
   - Objective: Minimize Within-Cluster Sum of Squares (Inertia)
   - Optimal cluster evaluation: Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index
   - Profiling student behavioral personas (e.g. High Engaged, Moderate, Severely Disengaged)
2. Agglomerative Hierarchical Clustering
   - Bottom-up hierarchical merging with Ward's linkage criterion
   - Adjusted Rand Index (ARI) comparing unsupervised clusters against ground-truth At_Risk
3. Principal Component Analysis (PCA)
   - Covariance matrix spectral decomposition (Eigenvalues & Eigenvectors)
   - Explained Variance Ratio & Cumulative Variance
   - 2D orthogonal projection for student engagement mapping
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score

class Unit4UnsupervisedPcaModels:
    def __init__(self, n_clusters: int = 3, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        # K-Means clustering
        self.kmeans = KMeans(
            n_clusters=n_clusters,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=random_state
        )
        
        # Agglomerative Hierarchical Clustering
        self.hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='euclidean',
            linkage='ward'
        )
        
        # Principal Component Analysis
        self.pca = PCA(n_components=2, random_state=random_state)
        self.pca_full = PCA(random_state=random_state)

    def fit_clustering(self, X):
        """Fit clustering algorithms on student feature matrix."""
        kmeans_labels = self.kmeans.fit_predict(X)
        hierarchical_labels = self.hierarchical.fit_predict(X)
        return kmeans_labels, hierarchical_labels

    def evaluate_clustering(self, X, y_true=None):
        """Evaluate clustering quality metrics."""
        km_labels = self.kmeans.labels_
        hier_labels = self.hierarchical.labels_
        
        metrics = {
            'KMeans': {
                'Inertia': round(float(self.kmeans.inertia_), 2),
                'Silhouette_Score': round(float(silhouette_score(X, km_labels)), 4),
                'Davies_Bouldin_Index': round(float(davies_bouldin_score(X, km_labels)), 4),
                'Calinski_Harabasz_Index': round(float(calinski_harabasz_score(X, km_labels)), 2),
            },
            'Hierarchical': {
                'Silhouette_Score': round(float(silhouette_score(X, hier_labels)), 4),
                'Davies_Bouldin_Index': round(float(davies_bouldin_score(X, hier_labels)), 4),
                'Calinski_Harabasz_Index': round(float(calinski_harabasz_score(X, hier_labels)), 2),
            }
        }
        
        if y_true is not None:
            metrics['KMeans']['Adjusted_Rand_Index'] = round(float(adjusted_rand_score(y_true, km_labels)), 4)
            metrics['Hierarchical']['Adjusted_Rand_Index'] = round(float(adjusted_rand_score(y_true, hier_labels)), 4)
            
        return metrics

    def fit_pca(self, X):
        """Fit PCA and compute explained variance ratios."""
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
        """
        Profiles the behavioral characteristics of student cohorts discovered via unsupervised clustering.
        """
        df_copy = df_raw.copy()
        df_copy['Cluster'] = labels
        
        numeric_cols = [
            'GPA', 'Attendance_Rate', 'Assignments_Submitted',
            'Online_Learning_Hours', 'Forum_Participation', 'Lab_Performance'
        ]
        
        profile = df_copy.groupby('Cluster')[numeric_cols].mean().round(2).to_dict(orient='index')
        # Add risk proportion in each cluster if available
        if 'At_Risk' in df_copy.columns:
            risk_rate = df_copy.groupby('Cluster')['At_Risk'].mean().round(4).to_dict()
            for k in profile:
                profile[k]['At_Risk_Rate'] = round(risk_rate[k] * 100, 2)
        return profile
