"""
Unit 5: Ensemble Learning & Advanced Predictive Modeling
Concepts Covered:
1. Bagging:
   - Random Forest Classifier (Bootstrap aggregation, random subspace projection, OOB error)
2. Boosting:
   - AdaBoost Classifier (Adaptive sample reweighting)
   - Gradient Boosting Classifier (GBDT - sequential functional gradient descent)
   - Extreme Gradient Boosting (XGBoost - 2nd-order Taylor expansion, regularization)
3. Meta-Ensemble:
   - Soft Voting Classifier (aggregating predicted class probabilities)
4. Cross-Validation & Model Evaluation:
   - 5-Fold Stratified Cross-Validation (Mean & Std Dev of Accuracy & ROC-AUC)
   - Detailed Confusion Matrix & Error Analysis
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix
)

class Unit5EnsembleModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # 1. Random Forest Classifier
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state
        )
        
        # 2. AdaBoost Classifier
        self.adaboost = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=random_state
        )
        
        # 3. Gradient Boosting Classifier
        self.gbdt = GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        )
        
        # 4. XGBoost Classifier
        self.xgb = XGBClassifier(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=random_state
        )
        
        # 5. Soft Voting Classifier
        self.voting = VotingClassifier(
            estimators=[
                ('rf', self.rf),
                ('xgb', self.xgb),
                ('lr', LogisticRegression(max_iter=500, random_state=random_state))
            ],
            voting='soft'
        )

    def train_all(self, X_train, y_train):
        """Train all ensemble models."""
        self.rf.fit(X_train, y_train)
        self.adaboost.fit(X_train, y_train)
        self.gbdt.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        self.voting.fit(X_train, y_train)

    def evaluate_all(self, X_test, y_test):
        """Evaluate all Unit 5 ensemble models."""
        models = {
            'Random_Forest': self.rf,
            'AdaBoost': self.adaboost,
            'Gradient_Boosting': self.gbdt,
            'XGBoost': self.xgb,
            'Voting_Classifier': self.voting
        }
        
        results = {}
        for name, clf in models.items():
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            results[name] = {
                'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                'Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                'F1_Score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                'ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
                'Log_Loss': round(float(log_loss(y_test, y_prob)), 4),
                'Confusion_Matrix': {
                    'TP': int(tp), 'FP': int(fp), 'TN': int(tn), 'FN': int(fn)
                }
            }
        return results

    def cross_validate(self, X, y, cv_folds: int = 5):
        """Execute Stratified K-Fold Cross Validation for ensembles."""
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        models = {
            'Random_Forest': self.rf,
            'AdaBoost': self.adaboost,
            'Gradient_Boosting': self.gbdt,
            'XGBoost': self.xgb,
            'Voting_Classifier': self.voting
        }
        
        cv_scores = {}
        for name, clf in models.items():
            acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
            roc_scores = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
            f1_scores = cross_val_score(clf, X, y, cv=cv, scoring='f1')
            
            cv_scores[name] = {
                'CV_Accuracy_Mean': round(float(acc_scores.mean()), 4),
                'CV_Accuracy_Std': round(float(acc_scores.std()), 4),
                'CV_ROC_AUC_Mean': round(float(roc_scores.mean()), 4),
                'CV_ROC_AUC_Std': round(float(roc_scores.std()), 4),
                'CV_F1_Mean': round(float(f1_scores.mean()), 4),
                'CV_F1_Std': round(float(f1_scores.std()), 4),
            }
        return cv_scores

    def get_rf_feature_importances(self, feature_names):
        """Extract MDI feature importances from Random Forest."""
        importances = self.rf.feature_importances_
        ranking = [
            {'feature': f, 'importance': round(float(imp), 4)}
            for f, imp in zip(feature_names, importances)
        ]
        ranking.sort(key=lambda x: x['importance'], reverse=True)
        return ranking

    def get_xgb_feature_importances(self, feature_names):
        """Extract gain-based feature importances from XGBoost."""
        importances = self.xgb.feature_importances_
        ranking = [
            {'feature': f, 'importance': round(float(imp), 4)}
            for f, imp in zip(feature_names, importances)
        ]
        ranking.sort(key=lambda x: x['importance'], reverse=True)
        return ranking
