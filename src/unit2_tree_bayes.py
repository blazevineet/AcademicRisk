"""
Unit 2: Decision Trees & Probabilistic Bayesian Models
Concepts Covered:
1. Decision Tree Classifier (CART Algorithm)
   - Impurity Criteria: Gini Impurity & Information Gain (Entropy)
   - Tree Pruning parameters: max_depth, min_samples_split, min_samples_leaf
   - Non-linear decision surfaces & Mean Decrease in Impurity (MDI)
2. Gaussian Naive Bayes Classifier
   - Bayes' Theorem: P(Risk|X) = [P(X|Risk) * P(Risk)] / P(X)
   - Conditional independence assumption
   - Continuous feature likelihood via Gaussian density estimation
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class Unit2TreeBayesModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # Decision Tree with Gini Impurity
        self.dt_gini = DecisionTreeClassifier(
            criterion='gini',
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        )
        
        # Decision Tree with Entropy (Information Gain)
        self.dt_entropy = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=random_state
        )
        
        # Gaussian Naive Bayes
        self.naive_bayes = GaussianNB()

    def train(self, X_train, y_train):
        """Train Decision Trees and Naive Bayes."""
        self.dt_gini.fit(X_train, y_train)
        self.dt_entropy.fit(X_train, y_train)
        self.naive_bayes.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate Unit 2 models on Academic Risk prediction."""
        models = {
            'Decision_Tree_Gini': self.dt_gini,
            'Decision_Tree_Entropy': self.dt_entropy,
            'Gaussian_Naive_Bayes': self.naive_bayes
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

    def get_feature_importances(self, feature_names):
        """Return Gini-based feature importances."""
        importances = self.dt_gini.feature_importances_
        ranking = [
            {'feature': f, 'importance': round(float(imp), 4)}
            for f, imp in zip(feature_names, importances)
        ]
        ranking.sort(key=lambda x: x['importance'], reverse=True)
        return ranking

    def get_decision_rules(self, feature_names, max_depth: int = 3):
        """Export human-interpretable pedagogical decision rules."""
        return export_text(self.dt_gini, feature_names=feature_names, max_depth=max_depth)
