"""
Unit 2: Bayesian Logistic Regression & Support Vector Machines (SVM)
Concepts Covered:
1. Bayesian Logistic Regression (Bishop PRML Chapter 4.5)
   - Prior: p(w) ~ N(0, alpha^(-1) * I)
   - MAP Estimation: w_MAP = argmin E(w) = -log p(t|w) + (alpha/2) * w^T w
   - Laplace Approximation: Posterior p(w|t) ~ N(w_MAP, S_N) where S_N = H^(-1)
   - Predictive Distribution via Probit Approximation: p(t=1|x) = sigmoid(kappa(sigma_a^2) * mu_a)
   - Epistemic uncertainty quantification (sigma_a^2 = x^T S_N x)
2. Support Vector Machines (SVM)
   - Maximal margin hyperplane & dual Lagrangian formulation
   - Linear Kernel, RBF Kernel, and Polynomial Kernel
   - Slack variable C trade-off and Support Vector identification
"""

import numpy as np
import scipy.optimize as opt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha  # Prior precision
        self.w_map = None
        self.S_N = None     # Posterior covariance matrix
        self.classes_ = np.array([0, 1])

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        return tags

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit Bayesian Logistic Regression via MAP and Laplace approximation."""
        # Add bias column
        N, D = X.shape
        Phi = np.hstack([np.ones((N, 1)), X])
        y = np.asarray(y, dtype=float)
        
        # Define negative log posterior objective E(w)
        def objective(w):
            z = Phi @ w
            y_pred = self._sigmoid(z)
            # Binary cross entropy + L2 prior regularization
            loss = -np.sum(y * np.log(y_pred + 1e-12) + (1.0 - y) * np.log(1.0 - y_pred + 1e-12))
            reg = 0.5 * self.alpha * np.dot(w[1:], w[1:])  # do not penalize bias
            return loss + reg

        def gradient(w):
            z = Phi @ w
            y_pred = self._sigmoid(z)
            grad = Phi.T @ (y_pred - y)
            grad[1:] += self.alpha * w[1:]
            return grad

        # Optimize for MAP weights
        w_init = np.zeros(D + 1)
        res = opt.minimize(objective, w_init, jac=gradient, method='L-BFGS-B')
        self.w_map = res.x

        # Compute Hessian at w_map for Laplace approximation
        z_map = Phi @ self.w_map
        y_map = self._sigmoid(z_map)
        R = y_map * (1.0 - y_map)  # N weights
        
        # H = Phi^T diag(R) Phi + alpha * I
        H = (Phi.T * R) @ Phi
        reg_I = self.alpha * np.eye(D + 1)
        reg_I[0, 0] = 1e-4  # uninformative prior for bias
        H += reg_I
        
        # Posterior covariance S_N = H^(-1)
        self.S_N = np.linalg.pinv(H)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Computes predictive distribution via probit convolution approximation:
        mu_a = w_map^T phi
        sigma_a^2 = phi^T S_N phi
        kappa(sigma^2) = (1 + pi * sigma^2 / 8)^(-1/2)
        p(t=1|x) = sigmoid(kappa * mu_a)
        """
        N, D = X.shape
        Phi = np.hstack([np.ones((N, 1)), X])
        
        mu_a = Phi @ self.w_map
        # Variances sigma_a^2 = diag(Phi S_N Phi^T)
        sigma_a_sq = np.sum((Phi @ self.S_N) * Phi, axis=1)
        
        # Probit scaling factor kappa
        kappa = 1.0 / np.sqrt(1.0 + (np.pi * sigma_a_sq) / 8.0)
        p_risk = self._sigmoid(kappa * mu_a)
        
        return np.column_stack([1.0 - p_risk, p_risk])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.50).astype(int)

    def get_predictive_uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Returns epistemic variance sigma_a^2 for each test sample."""
        N, D = X.shape
        Phi = np.hstack([np.ones((N, 1)), X])
        sigma_a_sq = np.sum((Phi @ self.S_N) * Phi, axis=1)
        return sigma_a_sq

class Unit2BayesianLrSvmModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # 1. Bayesian Logistic Regression
        self.bayes_lr = BayesianLogisticRegression(alpha=1.0)
        
        # 2a. Support Vector Machine - Linear Kernel
        self.svm_linear = SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=random_state
        )
        
        # 2b. Support Vector Machine - RBF Kernel
        self.svm_rbf = SVC(
            kernel='rbf',
            C=1.5,
            gamma='scale',
            probability=True,
            random_state=random_state
        )
        
        # 2c. Support Vector Machine - Polynomial Kernel (Degree 3)
        self.svm_poly = SVC(
            kernel='poly',
            degree=3,
            C=1.0,
            probability=True,
            random_state=random_state
        )

    def train_all(self, X_train, y_train):
        self.bayes_lr.fit(X_train, y_train)
        self.svm_linear.fit(X_train, y_train)
        self.svm_rbf.fit(X_train, y_train)
        self.svm_poly.fit(X_train, y_train)

    def evaluate_all(self, X_test, y_test):
        models = {
            'Bayesian_Logistic_Regression': self.bayes_lr,
            'SVM_Linear_Kernel': self.svm_linear,
            'SVM_RBF_Kernel': self.svm_rbf,
            'SVM_Poly_Kernel': self.svm_poly
        }
        
        results = {}
        for name, clf in models.items():
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            
            results[name] = {
                'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                'Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                'F1_Score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                'ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
                'Log_Loss': round(float(log_loss(y_test, y_prob)), 4)
            }
        return results

def plot_unit2_visual(u2_models, X_test, y_test, results_df, out_path):
    """Generates visual 4-panel comparison graph for Unit 2."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=300)
    
    # 1. Top Left: Metric Comparison Bar Chart
    ax1 = axes[0, 0]
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']
    models = results_df['Model'].tolist()
    x = np.arange(len(models))
    width = 0.16
    colors = ['#3182ce', '#38a169', '#e53e3e', '#dd6b20', '#805ad5']
    
    for i, (m, col) in enumerate(zip(metrics, colors)):
        vals = [results_df.loc[results_df['Model'] == mod, m].values[0] * 100 for mod in models]
        ax1.bar(x + (i - 2) * width, vals, width, label=m, color=col, alpha=0.9)
        
    ax1.set_ylabel('Score (%)', fontsize=11, fontweight='bold')
    ax1.set_title('Unit 2: Performance Comparison Across Kernels & Bayes LR', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', '\n') for m in models], fontsize=9)
    ax1.set_ylim(85, 102)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # 2. Top Right: ROC Curves
    ax2 = axes[0, 1]
    clfs = {
        'Bayesian LR': u2_models.bayes_lr,
        'SVM Linear': u2_models.svm_linear,
        'SVM RBF': u2_models.svm_rbf,
        'SVM Poly': u2_models.svm_poly
    }
    roc_colors = ['#2b6cb0', '#2f855a', '#dd6b20', '#805ad5']
    for (name, clf), col in zip(clfs.items(), roc_colors):
        probs = clf.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probs)
        score = auc(fpr, tpr)
        ax2.plot(fpr, tpr, label=f"{name} (AUC = {score:.4f})", color=col, linewidth=2.2)
        
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
    ax2.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax2.set_title('Unit 2: ROC Curves (Receiver Operating Characteristic)', fontsize=12, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.95)
    
    # 3. Bottom Left: Confusion Matrix Heatmap (Bayesian Logistic Regression)
    ax3 = axes[1, 0]
    y_pred_blr = u2_models.bayes_lr.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_blr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax3, cbar=False,
                xticklabels=['Safe', 'At Risk'], yticklabels=['Safe', 'At Risk'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax3.set_title('Bayesian Logistic Regression Confusion Matrix', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Predicted Risk Class', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Actual Ground Truth', fontsize=10, fontweight='bold')
    
    # 4. Bottom Right: Epistemic Uncertainty Distribution
    ax4 = axes[1, 1]
    variances = u2_models.bayes_lr.get_predictive_uncertainty(X_test)
    ax4.hist(variances, bins=25, color='#3182ce', edgecolor='#1a365d', alpha=0.85)
    ax4.axvline(np.mean(variances), color='red', linestyle='--', linewidth=2, label=f'Mean Variance = {np.mean(variances):.3f}')
    ax4.set_xlabel('Predictive Epistemic Variance sigma_a^2', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Student Density / Count', fontsize=11, fontweight='bold')
    ax4.set_title('Epistemic Uncertainty Distribution (Laplace Approximation)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=10, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return out_path

if __name__ == '__main__':
    import os
    import sys
    import pandas as pd
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        
    from src.preprocessing import load_data, prepare_data
    
    data_path = os.path.join(base_dir, 'data', 'academic_risk_data.csv')
    df = load_data(data_path)
    splits = prepare_data(df, test_size=0.2, random_state=42)
    
    print("=" * 75)
    print(" UNIT 2: BAYESIAN LOGISTIC REGRESSION & SUPPORT VECTOR MACHINES (SVM)")
    print("=" * 75)
    
    u2 = Unit2BayesianLrSvmModels(random_state=42)
    print("\n[+] Fitting Bayesian Logistic Regression (MAP + Laplace + Probit)...")
    print("[+] Fitting SVM (Linear, RBF, and Polynomial Kernels)...")
    u2.train_all(splits['X_train_proc'], splits['y_train'])
    
    results = u2.evaluate_all(splits['X_test_proc'], splits['y_test'])
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)
    print("\n[+] Unit 2 Test Evaluation Results:")
    print(df_res.to_string(index=False))
    
    # Showcase epistemic uncertainty on a sample student
    sample_x = splits['X_test_proc'][:3]
    var_a = u2.bayes_lr.get_predictive_uncertainty(sample_x)
    probs = u2.bayes_lr.predict_proba(sample_x)[:, 1]
    print("\n[+] Sample Bayesian Predictions with Epistemic Uncertainty Quantification:")
    for i in range(3):
        print(f"  Student #{i+1}: Predicted Risk Prob = {probs[i]*100:.2f}% | Latent Variance sigma_a^2 = {var_a[i]:.4f}")
        
    # Save & Open Graph
    plot_file = os.path.join(base_dir, 'results', 'unit2_bayesian_lr_svm.png')
    plot_unit2_visual(u2, splits['X_test_proc'], splits['y_test'], df_res, plot_file)
    print(f"\n[+] Visual comparison graph saved to: {plot_file}")
    
    try:
        os.startfile(plot_file)
        print("[+] Opening graph in your default image viewer...")
    except Exception:
        pass
    print("=" * 75)
