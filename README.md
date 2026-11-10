# Academic Risk & Engagement Prediction System

A comprehensive Machine Learning Early Alert and Student Analytics System built according to the specific curriculum:
* **Unit 1:** Polynomial Curve Fitting
* **Unit 2:** Bayesian Logistic Regression, Support Vector Machines (SVM)
* **Unit 3:** K-Means, Principal Component Analysis (PCA), Hierarchical Clustering
* **Unit 4:** Hidden Markov Models (HMM), Viterbi Algorithm
* **Unit 5:** CART (Classification & Regression Trees), Ensemble Learning

---

## 1. Dataset Schema (1,500 Student Records)

The system operates on an academic cohort dataset matching the exact 11 target attributes:

| Attribute | Data Type | Range / Domain | Pedagogical Meaning |
|:---|:---:|:---:|:---|
| `Age` | Numerical (int) | 18 – 26 | Student chronological age |
| `Gender` | Categorical | `Male`, `Female` | Self-reported gender |
| `Socio_Economic_Status` | Categorical | `Low`, `Medium`, `High` | Household environmental background |
| `GPA` | Numerical (float) | 1.20 – 4.00 | Prior academic mastery ($r = -0.7239$) |
| `Attendance_Rate` | Numerical (float) | 32.0% – 100.0% | Lecture session attendance ($r = -0.7139$) |
| `Assignments_Submitted` | Numerical (int) | 0 – 10 | Completed coursework deliverables ($r = -0.7515$) |
| `Online_Learning_Hours` | Numerical (float) | 1.0 – 35.0 hrs/wk | Weekly LMS interaction duration ($r = -0.6698$) |
| `Forum_Participation` | Numerical (int) | 0 – 50 posts | Collaborative Q&A activity ($r = -0.6100$) |
| `Lab_Performance` | Numerical (float) | 20.0 – 100.0 | Practical examination score ($r = -0.7191$) |
| `Extracurricular_Participation` | Categorical | `Yes`, `No` | Clubs, sports, or academic society involvement |
| **`At_Risk`** | **Binary Target** | **0 = Safe (63.53%), 1 = At Risk (36.47%)** | Target academic attrition hazard |

---

## 2. Curriculum Implementation Breakdown

### **Unit 1: Polynomial Curve Fitting (Bishop PRML Ch. 1)**
* Fits $y(x, w) = \sum_{j=0}^M w_j x^j$ across degrees $M \in \{1, 2, 3, 5, 9\}$.
* Compares Train and Test Root-Mean-Square Error ($E_{RMS}$).
* Demonstrates underfitting ($M=1$, $E_{RMS}=9.28$), optimal fit ($M=2, 3$, $E_{RMS}=8.98$), severe overfitting ($M=9$ with $\|w\|_2 = 811,796$), and regularization recovery ($\lambda = 10^{-2}$, $\|w\|_2 = 101.76$, $E_{RMS} = 8.9769$).

### **Unit 2: Bayesian Logistic Regression & SVM**
* **Bayesian Logistic Regression:** Full MAP estimation with Laplace approximation, computing posterior covariance $S_N = H^{-1}$ and predictive probit integration with epistemic uncertainty estimation $\sigma_a^2$.
* **Support Vector Machines (SVM):** Maximum margin optimization across **Linear**, **RBF**, and **Polynomial (Degree 3)** kernels.

### **Unit 3: K-Means, PCA & Hierarchical Clustering**
* **K-Means Clustering:** Discovers 3 student behavioral tiers (Silhouette: 0.2045, Davies-Bouldin: 1.5001).
* **Agglomerative Hierarchical Clustering:** Evaluates Ward's linkage (ARI = 0.3496) and Complete linkage (ARI = 0.4588).
* **PCA:** Dimensionality reduction retaining **71.24%** of variance in 2D ($\text{PC}_1$: 58.67%, $\text{PC}_2$: 12.57%).

### **Unit 4: Hidden Markov Models (HMM) & Viterbi Decoding**
* **3 Hidden States:** $S_1$ (Safe/Engaged), $S_2$ (Warning/Struggle), $S_3$ (Critical At-Risk).
* **Observable Emissions:** High, Moderate, Low weekly engagement tokens.
* **Forward Algorithm:** Computes observation sequence likelihood $P(O|\lambda)$.
* **Viterbi Dynamic Programming:** Traces student weekly progression and pinpoints the earliest critical risk entry week (average alert at **Week 4.68**).

### **Unit 5: CART & Ensemble Learning**
* **CART:** Decision trees evaluating **Gini Impurity** and **Entropy Information Gain**.
* **Ensembles:** **Random Forest** (Bagging), **AdaBoost**, **Gradient Boosting (GBDT)**, **XGBoost**, and **Soft Voting Classifier**.

---

## 3. Master Benchmark Results (Units 2, 4, 5 Classification)

Evaluated on $N=300$ independent held-out students and 5-Fold Stratified Cross-Validation on $N=1,500$:

| Rank | Unit | Algorithm | Test Accuracy | Precision | Recall | F1-Score | Test ROC-AUC | 5-Fold CV ROC | False Neg. Rate | Train Latency |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 1 | Unit 2 | **Bayesian Logistic Regression** | **96.67%** | **98.06%** | **92.66%** | **95.28%** | **99.76%** | **99.50%** | **7.34%** | **7.0 ms** |
| 🥈 2 | Unit 2 | **SVM (Linear Kernel)** | **96.67%** | **98.06%** | **92.66%** | **95.28%** | 99.73% | 99.49% | 7.34% | 42.4 ms |
| 🥉 3 | Unit 2 | **SVM (RBF Kernel)** | 96.33% | 97.12% | 92.66% | 94.84% | 99.63% | 99.32% | 7.34% | 54.3 ms |
| 4 | Unit 5 | Random Forest | 96.00% | 96.19% | 92.66% | 94.39% | 99.51% | 99.01% | 7.34% | 273.1 ms |
| 5 | Unit 5 | AdaBoost | 95.67% | 95.28% | 92.66% | 93.95% | 99.54% | 99.24% | 7.34% | 213.4 ms |
| 6 | Unit 2 | SVM (Poly Kernel) | 95.67% | **100.0%** | 88.07% | 93.66% | 99.66% | 99.21% | 11.93% | 50.2 ms |
| 7 | Unit 5 | XGBoost | 95.67% | 96.15% | 91.74% | 93.90% | 99.52% | 99.00% | 8.26% | 75.0 ms |
| 8 | Unit 5 | Gradient Boosting | 95.67% | 96.15% | 91.74% | 93.90% | 99.37% | 98.93% | 8.26% | 491.7 ms |
| 9 | Unit 5 | Voting Classifier | 95.33% | 93.58% | **93.58%** | 93.58% | 99.56% | 98.87% | **6.42%** | 253.0 ms |
| 10 | Unit 5 | CART (Gini Impurity) | 94.00% | 90.27% | **93.58%** | 91.89% | 96.26% | 94.87% | **6.42%** | **5.5 ms** |
| 11 | Unit 5 | CART (Entropy) | 94.33% | 95.10% | 88.99% | 91.94% | 95.58% | 95.51% | 11.01% | **4.0 ms** |

---

## 4. Execution Commands

1. **Run Full Experiments & Cross-Validation:**
   ```bash
   python run_experiments.py
   ```
2. **Run Interactive Early Warning Predictor:**
   ```bash
   python predict.py
   ```
3. **Open Visual HTML Dashboard:**
   Open [`results/report_dashboard.html`](results/report_dashboard.html) in any browser.
