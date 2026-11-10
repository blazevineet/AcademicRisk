"""
Unit 3: Instance-Based Learning, Support Vector Machines & Neural Networks
Concepts Covered:
1. K-Nearest Neighbors (k-NN) Classifier
   - Non-parametric lazy learning, Euclidean distance metric
   - Neighborhood vote weighting (uniform vs distance-weighted)
2. Support Vector Machines (SVM / SVC)
   - Maximal Margin Hyperplane, Support Vectors, Slack Variables
   - Linear Kernel vs Non-linear Radial Basis Function (RBF) Kernel
   - Margin penalty C and kernel coefficient gamma
3. Multi-Layer Perceptron (MLP Classifier / Artificial Neural Network)
   - Feedforward architecture: Input -> Hidden Layers -> Output
   - Non-linear Activation (ReLU), Cross-Entropy Loss
   - Backpropagation via Stochastic Gradient Descent / Adam optimizer
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss

class Unit3InstanceSvmAnnModels:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        
        # 1. K-Nearest Neighbors (k=5, Euclidean metric)
        self.knn = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='euclidean'
        )
        
        # 2a. Support Vector Classifier - Linear Kernel
        self.svm_linear = SVC(
            kernel='linear',
            C=1.0,
            probability=True,
            random_state=random_state
        )
        
        # 2b. Support Vector Classifier - RBF Kernel
        self.svm_rbf = SVC(
            kernel='rbf',
            C=1.5,
            gamma='scale',
            probability=True,
            random_state=random_state
        )
        
        # 3. Multi-Layer Perceptron (Neural Network: 64-32 architecture)
        self.mlp = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate_init=0.01,
            max_iter=500,
            early_stopping=True,
            n_iter_no_change=15,
            random_state=random_state
        )

    def train(self, X_train, y_train):
        """Train k-NN, Linear SVM, RBF SVM, and MLP Neural Network."""
        self.knn.fit(X_train, y_train)
        self.svm_linear.fit(X_train, y_train)
        self.svm_rbf.fit(X_train, y_train)
        self.mlp.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        """Evaluate Unit 3 models on Academic Risk prediction."""
        models = {
            'K_Nearest_Neighbors': self.knn,
            'SVM_Linear_Kernel': self.svm_linear,
            'SVM_RBF_Kernel': self.svm_rbf,
            'MLP_Neural_Network': self.mlp
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
