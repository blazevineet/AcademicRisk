"""
Unit 1: Linear Models & Optimization Foundations
Concepts Covered:
1. Linear Regression (Engagement Score Prediction)
2. Ridge Regression (L2 Regularization) & Lasso Regression (L1 Regularization)
3. Logistic Regression (Binary Academic Risk Classification)
4. Sigmoid Activation & Binary Cross-Entropy Loss
5. Coefficient Interpretation and Odds Ratios
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class Unit1LinearModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # Regression models for Engagement Score
        self.linear_reg = LinearRegression()
        self.ridge_reg = Ridge(alpha=1.0, random_state=random_state)
        self.lasso_reg = Lasso(alpha=0.1, random_state=random_state)
        
        # Classification model for Academic Risk
        self.logistic_clf = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=random_state
        )

    def train_regression(self, X_train, y_train_eng):
        """Train Linear, Ridge, and Lasso regressors."""
        self.linear_reg.fit(X_train, y_train_eng)
        self.ridge_reg.fit(X_train, y_train_eng)
        self.lasso_reg.fit(X_train, y_train_eng)

    def evaluate_regression(self, X_test, y_test_eng):
        """Evaluate regression models on Engagement Score."""
        results = {}
        for name, model in [('Linear_Regression', self.linear_reg),
                            ('Ridge_Regression', self.ridge_reg),
                            ('Lasso_Regression', self.lasso_reg)]:
            y_pred = model.predict(X_test)
            results[name] = {
                'MSE': round(float(mean_squared_error(y_test_eng, y_pred)), 4),
                'RMSE': round(float(np.sqrt(mean_squared_error(y_test_eng, y_pred))), 4),
                'MAE': round(float(mean_absolute_error(y_test_eng, y_pred)), 4),
                'R2_Score': round(float(r2_score(y_test_eng, y_pred)), 4)
            }
        return results

    def train_classification(self, X_train, y_train):
        """Train Logistic Regression for Academic Risk classification."""
        self.logistic_clf.fit(X_train, y_train)

    def evaluate_classification(self, X_test, y_test):
        """Evaluate Logistic Regression on Academic Risk."""
        y_pred = self.logistic_clf.predict(X_test)
        y_prob = self.logistic_clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
            'Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            'Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            'F1_Score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
            'ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
            'Log_Loss': round(float(log_loss(y_test, y_prob)), 4)
        }
        return metrics

    def get_feature_impact(self, feature_names):
        """Calculate coefficients and odds ratios for Logistic Regression."""
        coefs = self.logistic_clf.coef_[0]
        odds_ratios = np.exp(coefs)
        impact = []
        for name, coef, odds in zip(feature_names, coefs, odds_ratios):
            impact.append({
                'feature': name,
                'coefficient': round(float(coef), 4),
                'odds_ratio': round(float(odds), 4),
                'risk_effect': 'Increases Risk' if coef > 0 else 'Protective / Decreases Risk'
            })
        impact.sort(key=lambda x: abs(x['coefficient']), reverse=True)
        return impact
