"""
Unit 5: CART (Classification & Regression Trees) & Ensemble Learning
Concepts Covered:
1. CART Algorithm (Breiman et al.)
   - Binary recursive recursive partitioning
   - Splitting criteria: Gini Impurity & Information Gain (Entropy)
   - Cost-complexity pruning & tree regularization
2. Ensemble Learning Paradigms:
   - Bagging: Random Forest Classifier (Bootstrap aggregation & random feature subsets)
   - Boosting: AdaBoost Classifier (Exponential loss reweighting)
   - Gradient Boosting (GBDT): Sequential functional gradient descent
   - Extreme Gradient Boosting (XGBoost): Regularized second-order Taylor trees
   - Meta-Ensemble: Soft Voting Classifier (Probabilistic model blending)
"""

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class Unit5CartEnsembleModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # 1a. CART - Gini Impurity
        self.cart_gini = DecisionTreeClassifier(
            criterion='gini',
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state
        )
        
        # 1b. CART - Entropy / Information Gain
        self.cart_entropy = DecisionTreeClassifier(
            criterion='entropy',
            max_depth=6,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=random_state
        )
        
        # 2a. Random Forest (Bagging)
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=7,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=random_state
        )
        
        # 2b. AdaBoost (Sequential Adaptive Boosting)
        self.adaboost = AdaBoostClassifier(
            n_estimators=100,
            learning_rate=0.8,
            random_state=random_state
        )
        
        # 2c. Gradient Boosting Decision Trees (GBDT)
        self.gbdt = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.85,
            random_state=random_state
        )
        
        # 2d. XGBoost (Extreme Gradient Boosting)
        self.xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric='logloss',
            random_state=random_state
        )
        
        # 2e. Soft Voting Meta-Ensemble
        self.voting = VotingClassifier(
            estimators=[
                ('cart', self.cart_gini),
                ('rf', self.rf),
                ('xgb', self.xgb)
            ],
            voting='soft'
        )

    def train_all(self, X_train, y_train):
        self.cart_gini.fit(X_train, y_train)
        self.cart_entropy.fit(X_train, y_train)
        self.rf.fit(X_train, y_train)
        self.adaboost.fit(X_train, y_train)
        self.gbdt.fit(X_train, y_train)
        self.xgb.fit(X_train, y_train)
        self.voting.fit(X_train, y_train)

    def evaluate_all(self, X_test, y_test):
        models = {
            'CART_Gini': self.cart_gini,
            'CART_Entropy': self.cart_entropy,
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
            
            results[name] = {
                'Accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                'Precision': round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
                'Recall': round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
                'F1_Score': round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                'ROC_AUC': round(float(roc_auc_score(y_test, y_prob)), 4),
                'Log_Loss': round(float(log_loss(y_test, y_prob)), 4)
            }
        return results

    def get_cart_rules(self, feature_names, max_depth: int = 3):
        return export_text(self.cart_gini, feature_names=feature_names, max_depth=max_depth)

    def get_rf_importances(self, feature_names):
        importances = self.rf.feature_importances_
        ranking = [
            {'feature': f, 'importance': round(float(imp), 4)}
            for f, imp in zip(feature_names, importances)
        ]
        ranking.sort(key=lambda x: x['importance'], reverse=True)
        return ranking

def plot_unit5_visual(u5, X_test, y_test, df_res, feature_names, out_path):
    """Generates a comprehensive 4-panel visual report for Unit 5 CART & Ensemble Learning."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, auc, confusion_matrix
    import numpy as np
    import os
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.patch.set_facecolor('#ffffff')
    
    # 1. Performance Comparison Bar Chart
    models = df_res['Model'].tolist()
    accs = df_res['Accuracy'].tolist()
    recalls = df_res['Recall'].tolist()
    aucs = df_res['ROC_AUC'].tolist()
    
    x = np.arange(len(models))
    width = 0.26
    
    axes[0, 0].bar(x - width, accs, width, label='Accuracy', color='#3498db', alpha=0.9)
    axes[0, 0].bar(x, recalls, width, label='Recall (Sensitivity)', color='#e67e22', alpha=0.9)
    axes[0, 0].bar(x + width, aucs, width, label='ROC-AUC', color='#2ecc71', alpha=0.9)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=25, ha='right', fontsize=9)
    axes[0, 0].set_ylim([0.7, 1.05])
    axes[0, 0].set_title("Unit 5: Tree & Ensemble Benchmark Metrics", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Score", fontsize=10)
    axes[0, 0].legend(loc='lower right', fontsize=9)
    axes[0, 0].grid(True, linestyle=':', alpha=0.6)
    
    # 2. Multi-Model ROC Curves
    for name, model in [('CART (Gini)', u5.cart_gini), ('Random Forest', u5.rf), 
                        ('AdaBoost', u5.adaboost), ('GBDT', u5.gbdt), ('XGBoost', u5.xgb)]:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    axes[0, 1].plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Chance')
    axes[0, 1].set_title("ROC Curves (Tree vs. Boosting vs. Bagging)", fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel("False Positive Rate", fontsize=10)
    axes[0, 1].set_ylabel("True Positive Rate", fontsize=10)
    axes[0, 1].legend(loc='lower right', fontsize=9)
    axes[0, 1].grid(True, linestyle=':', alpha=0.6)
    
    # 3. Confusion Matrix for Random Forest
    rf_preds = u5.rf.predict(X_test)
    cm = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0],
                xticklabels=['Safe (0)', 'At-Risk (1)'],
                yticklabels=['Safe (0)', 'At-Risk (1)'])
    axes[1, 0].set_title("Random Forest Confusion Matrix (Test Set N=300)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Predicted Class", fontsize=10)
    axes[1, 0].set_ylabel("Ground Truth Class", fontsize=10)
    
    # 4. Feature Importance Horizontal Bar Chart
    top_feats = u5.get_rf_importances(feature_names)[:8]
    feats = [item['feature'] for item in reversed(top_feats)]
    scores = [item['importance'] * 100 for item in reversed(top_feats)]
    
    axes[1, 1].barh(feats, scores, color='#1abc9c', edgecolor='black', alpha=0.85)
    axes[1, 1].set_title("Top Academic Risk Drivers (RF MDI Importance %)", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("Relative Importance (%)", fontsize=10)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    for i, v in enumerate(scores):
        axes[1, 1].text(v + 0.5, i, f"{v:.1f}%", va='center', fontsize=9, fontweight='bold')
    axes[1, 1].set_xlim([0, max(scores) + 8])
    
    plt.suptitle("Unit 5: Decision Trees (CART) & Ensemble Learning Architecture", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import os
    os.environ["LOKY_MAX_CPU_COUNT"] = "4"
    import warnings
    warnings.filterwarnings('ignore')
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
    print(" UNIT 5: CART (DECISION TREES) & ENSEMBLE LEARNING")
    print("=" * 75)
    
    u5 = Unit5CartEnsembleModels(random_state=42)
    print("\n[+] Training CART (Gini & Entropy) and Ensembles (RF, AdaBoost, GBDT, XGBoost, Voting)...")
    u5.train_all(splits['X_train_proc'], splits['y_train'])
    
    results = u5.evaluate_all(splits['X_test_proc'], splits['y_test'])
    df_res = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})
    
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)
    print("\n[+] Unit 5 Test Evaluation Results:")
    print(df_res.to_string(index=False))
    
    print("\n[+] CART Pedagogical Decision Rules (First 2 Levels):")
    print(u5.get_cart_rules(splits['feature_names'], max_depth=2))
    
    print("\n[+] Random Forest MDI Feature Importances (Top 5 Drivers):")
    top_feats = u5.get_rf_importances(splits['feature_names'])[:5]
    for i, item in enumerate(top_feats):
        print(f"  {i+1}. {item['feature']:<25} : {item['importance']*100:.2f}% contribution")
    
    # Save & Open Graph
    plot_file = os.path.join(base_dir, 'results', 'unit5_cart_ensemble.png')
    plot_unit5_visual(u5, splits['X_test_proc'], splits['y_test'], df_res, splits['feature_names'], plot_file)
    print(f"\n[+] Visual comparison graph saved to: {plot_file}")
    
    try:
        os.startfile(plot_file)
        print("[+] Opening graph in your default image viewer...")
    except Exception:
        pass
    print("=" * 75)
