# ACADEMIC RISK & ENGAGEMENT PREDICTION SYSTEM
## Formal Machine Learning Project Report: Units 1 to 5

---

### Executive Summary

Early identification of undergraduate students at risk of academic failure or drop-out is a paramount objective for higher education institutions. This report presents an empirical, theoretical, and algorithmic investigation of the **Academic Risk & Engagement Prediction System**, structured according to the syllabus:
* **Unit 1: Polynomial Curve Fitting** (Bishop PRML Chapter 1)
* **Unit 2: Bayesian Logistic Regression, Support Vector Machines (SVM)**
* **Unit 3: K-Means, Principal Component Analysis (PCA), Hierarchical Clustering**
* **Unit 4: Hidden Markov Models (HMM), Viterbi Decoding Algorithm**
* **Unit 5: CART (Classification & Regression Trees), Ensemble Learning**

Experiments were conducted on a collegiate cohort dataset of **1,500 students** structured across **11 attributes**:
`Age, Gender, Socio_Economic_Status, GPA, Attendance_Rate, Assignments_Submitted, Online_Learning_Hours, Forum_Participation, Lab_Performance, Extracurricular_Participation, At_Risk`.

---

## 1. Implementation of Concepts and Algorithms (Units 1 to 5)

```
+----------------------------------------------------------------------------------------------------+
|                                    CURRICULUM IMPLEMENTATION MAP                                   |
+----------------------------------------------------------------------------------------------------+
|  UNIT 1: Polynomial Curve Fitting                                                                  |
|          -> Model: y(x, w) = sum_{j=0}^M w_j * x^j                                                 |
|          -> Error: Sum-of-Squares E(w) & Root-Mean-Square Error E_RMS                              |
|          -> Investigation: Degrees M in {1, 2, 3, 5, 9}, Underfitting vs Overfitting              |
|          -> Regularization: Weight decay (Ridge penalty lambda), shrinkage of ||w||_2              |
+----------------------------------------------------------------------------------------------------+
|  UNIT 2: Bayesian Logistic Regression & Support Vector Machines (SVM)                              |
|          -> Bayesian Logistic Regression: Prior p(w) ~ N(0, alpha^-1 * I), MAP Estimation          |
|             Laplace Approximation: S_N = H^-1, Predictive Probit Convolution p(t=1|x)              |
|             Epistemic Uncertainty Quantification: sigma_a^2 = x^T S_N x                            |
|          -> Support Vector Machines (SVM): Maximum margin hyperplane, Dual Lagrangian              |
|             Linear Kernel, Radial Basis Function (RBF) Kernel, Polynomial Kernel (Degree 3)        |
+----------------------------------------------------------------------------------------------------+
|  UNIT 3: K-Means, PCA & Hierarchical Clustering                                                    |
|          -> K-Means: Inertia minimization, Silhouette Coefficient, Davies-Bouldin Index            |
|          -> Agglomerative Hierarchical: Ward's Minimum Variance Linkage vs Complete Linkage       |
|          -> Principal Component Analysis (PCA): Covariance Eigen-decomposition, 2D Projection      |
|          -> Unsupervised Discovery: 3 Student Persona Cohorts (Safe, Warning, Critical)            |
+----------------------------------------------------------------------------------------------------+
|  UNIT 4: Hidden Markov Models (HMM) & Viterbi Decoding                                             |
|          -> 3 Latent States: S0 (Safe/Engaged), S1 (Warning/Struggle), S2 (Critical At-Risk)       |
|          -> 3 Observable Emissions: High, Moderate, Low weekly engagement tokens                   |
|          -> Forward Algorithm: Sequence log-likelihood log P(O | lambda)                           |
|          -> Backward Algorithm: Posterior marginal gamma_t(i)                                      |
|          -> Viterbi Algorithm: Dynamic programming lattice & optimal hidden path backtracking     |
|          -> Earliest Risk Detection: Average warning issued at Week 4.68                           |
+----------------------------------------------------------------------------------------------------+
|  UNIT 5: CART & Ensemble Learning                                                                  |
|          -> CART: Recursive partitioning, Gini Impurity vs Entropy Information Gain                |
|          -> Bagging: Random Forest Classifier (Bootstrap aggregating, random feature subspace)     |
|          -> Boosting: AdaBoost (Adaptive reweighting), GBDT (Functional gradient descent)          |
|          -> Extreme Gradient Boosting (XGBoost): 2nd-order Taylor expansion & tree regularization  |
|          -> Meta-Ensemble: Soft Voting Classifier (Probabilistic model blending)                   |
|          -> 5-Fold Stratified Cross-Validation & Metric Evaluation                                 |
+----------------------------------------------------------------------------------------------------+
```

---

### Unit 1: Polynomial Curve Fitting (Bishop PRML Chapter 1)

#### Mathematical Formulation:
1. **Model Hypothesis:**
   Models continuous student engagement $y(x, w)$ as a linear combination of non-linear polynomial powers of an input feature $x$ (e.g., Online Learning Hours or Attendance Rate):
   $$y(x, w) = w_0 + w_1 x + w_2 x^2 + \dots + w_M x^M = \sum_{j=0}^M w_j x^j$$
   Where $M$ is the polynomial degree.

2. **Error Functions:**
   * **Sum-of-Squares Error:**
     $$E(w) = \frac{1}{2} \sum_{n=1}^N \left( y(x_n, w) - t_n \right)^2$$
   * **Root-Mean-Square Error ($E_{RMS}$):**
     $$E_{RMS} = \sqrt{\frac{2 E(w^*)}{N}} = \sqrt{\frac{1}{N} \sum_{n=1}^N \left( y(x_n, w^*) - t_n \right)^2}$$
     $E_{RMS}$ allows comparison across datasets of different sample sizes $N$ on the original scale.

3. **Closed-Form Solution via Regularized Normal Equations:**
   Constructing the Vandermonde design matrix $\Phi \in \mathbb{R}^{N \times (M+1)}$, where $\Phi_{nj} = x_n^j$:
   $$w^* = \left( \Phi^T \Phi + \lambda I \right)^{-1} \Phi^T t$$
   Where $\lambda$ is the regularization weight decay parameter ($I_{00} = 0$ to prevent penalizing the bias term $w_0$).

#### Empirical Study Across Degrees $M \in \{1, 2, 3, 5, 9\}$:
Fitting student Engagement Score on the cohort ($N_{\text{train}}=1200, N_{\text{test}}=300$):

| Degree $M$ | Regularization $\lambda$ | Train $E_{RMS}$ | Test $E_{RMS}$ | Test $R^2$ | Weight Norm $\|w\|_2$ | Empirical Diagnosis |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **$M = 1$** | $0.0$ | $9.4038$ | $9.2829$ | $0.7123$ | $80.83$ | **Underfitting** (Inflexible linear line fails to capture curvature) |
| **$M = 2$** | $0.0$ | $8.9795$ | $8.9969$ | $0.7298$ | $157.44$ | **Optimal Generalization** (Captures natural engagement saturation) |
| **$M = 3$** | $0.0$ | $8.9217$ | $8.9825$ | $0.7306$ | $138.48$ | **Optimal Generalization** (Balanced bias and variance) |
| **$M = 5$** | $0.0$ | $8.9216$ | $8.9785$ | $0.7309$ | $160.73$ | Mild Overfitting |
| **$M = 9$** | $0.0$ | $8.9164$ | $8.9954$ | $0.7299$ | **$811,796.28$** | **Severe Overfitting** (Exploding weights & oscillation) |
| **$M = 9$** | $10^{-4}$ ($\ln \lambda = -9.2$) | $8.9211$ | $8.9820$ | $0.7307$ | $209.53$ | Regularized (Weights contained) |
| **$M = 9$** | $10^{-2}$ ($\ln \lambda = -4.6$) | $8.9218$ | **$8.9769$** | **$0.7310$** | **$101.76$** | **Optimal Regularized Fit** (Best test performance) |
| **$M = 9$** | $1.0$ ($\ln \lambda = 0.0$) | $8.9903$ | $9.0229$ | $0.7282$ | $80.14$ | Over-regularized (Approaching linear shrinkage) |

---

### Unit 2: Bayesian Logistic Regression & Support Vector Machines (SVM)

#### 1. Bayesian Logistic Regression (Bishop PRML 4.5):
* **Prior:** Zero-mean Gaussian prior over parameter vector: $p(w) = \mathcal{N}(w | 0, \alpha^{-1} I)$.
* **Likelihood:**
  $$p(t | w) = \prod_{n=1}^N y_n^{t_n} (1 - y_n)^{1 - t_n}, \quad y_n = \sigma(w^T \phi_n) = \frac{1}{1 + e^{-w^T \phi_n}}$$
* **MAP Optimization:** Find $w_{\text{MAP}} = \arg\max \ln p(w | t)$:
  $$E(w) = -\sum_{n=1}^N \left[ t_n \ln y_n + (1 - t_n) \ln(1 - y_n) \right] + \frac{\alpha}{2} w^T w$$
* **Laplace Approximation:**
  The posterior distribution is approximated by a Gaussian centered at $w_{\text{MAP}}$:
  $$p(w | t) \approx \mathcal{N}(w | w_{\text{MAP}}, S_N), \quad \text{where } S_N = H^{-1} = \left( \nabla \nabla E(w_{\text{MAP}}) \right)^{-1}$$
  The Hessian is computed analytically:
  $$H = \alpha I + \sum_{n=1}^N y_n (1 - y_n) \phi_n \phi_n^T$$
* **Predictive Distribution via Probit Approximation:**
  For a new query student $x^*$, the predictive distribution integrates over parameter uncertainty:
  $$p(t=1 | x^*, \mathcal{D}) = \int \sigma(w^T x^*) \mathcal{N}(w | w_{\text{MAP}}, S_N) dw \approx \sigma\left( \kappa(\sigma_a^2) \mu_a \right)$$
  Where:
  $$\mu_a = w_{\text{MAP}}^T x^*, \quad \sigma_a^2 = {x^*}^T S_N x^*, \quad \kappa(\sigma_a^2) = \left( 1 + \frac{\pi \sigma_a^2}{8} \right)^{-1/2}$$
  $\sigma_a^2$ quantifies **epistemic model uncertainty**; when evaluating an unfamiliar student profile, $\sigma_a^2$ expands, moderating the predicted probability towards $0.5$.

#### 2. Support Vector Machines (SVM):
Finds the maximal margin hyperplane $w^T \phi(x) + b = 0$ in primal form:
$$\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{n=1}^N \xi_n \quad \text{s.t. } t_n(w^T \phi(x_n) + b) \ge 1 - \xi_n, \quad \xi_n \ge 0$$
* **Linear Kernel:** $K(x, z) = x^T z$
* **Radial Basis Function (RBF) Kernel:** $K(x, z) = \exp(-\gamma \|x - z\|^2)$
* **Polynomial Kernel (Degree 3):** $K(x, z) = (\gamma x^T z + r)^3$

---

### Unit 3: K-Means, PCA & Hierarchical Clustering

#### 1. K-Means Clustering:
Minimizes the Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{k=1}^K \sum_{x \in C_k} \|x - \mu_k\|^2, \quad \mu_k = \frac{1}{|C_k|} \sum_{x \in C_k} x$$
Evaluated with $k=3$ clusters: **Inertia = 7445.8**, **Silhouette Score = 0.2045**, **Davies-Bouldin Index = 1.5001**.

#### 2. Agglomerative Hierarchical Clustering:
Evaluated bottom-up tree structures under two distinct linkage metrics:
* **Ward's Minimum Variance Linkage:** Merges clusters that minimize the increase in total within-cluster variance ($\Delta \text{ESS}_{AB} = \frac{n_A n_B}{n_A + n_B} \|\mu_A - \mu_B\|^2$). Achieved **Adjusted Rand Index (ARI) = 0.3496** with ground-truth risk.
* **Complete Linkage:** Merges based on maximum pairwise distance ($d(A, B) = \max_{x \in A, y \in B} \|x - y\|$). Achieved **ARI = 0.4588**.

#### 3. Principal Component Analysis (PCA):
Eigenvalue decomposition of the feature covariance matrix $\Sigma$:
$$\Sigma v_i = \lambda_i v_i$$
* **First Principal Component ($\text{PC}_1$):** Explains **58.67%** of total variance, loading heavily on continuous engagement metrics (Attendance, Assignments, GPA).
* **Second Principal Component ($\text{PC}_2$):** Explains **12.57%** of total variance, capturing demographic and practical lab variance.
* **Aggregate 2D Information Retained:** **71.24%**.

---

### Unit 4: Hidden Markov Models (HMM) & Viterbi Decoding Algorithm

#### Formal Model Specification $\lambda = (\pi, A, B)$:
Academic risk is dynamic. A student's internal cognitive state evolves across a 10-week semester:
1. **Hidden States $S = \{S_0, S_1, S_2\}$:**
   * $S_0$: **Safe & Deeply Engaged**
   * $S_1$: **Moderate Struggle / Early Warning Transition**
   * $S_2$: **Critical Academic Hazard / Attrition State**
2. **Observable Emissions $V = \{V_0, V_1, V_2\}$:**
   * $V_0$: High Weekly Activity (full attendance, on-time assignments, active LMS hours)
   * $V_1$: Moderate Weekly Activity (intermittent attendance, partial submissions)
   * $V_2$: Low / Inactive (missed sessions, overdue submissions, minimal LMS login)
3. **Transition Probability Matrix $A$ ($3 \times 3$):**
   $$A = \begin{pmatrix} 0.82 & 0.15 & 0.03 \\ 0.15 & 0.65 & 0.20 \\ 0.03 & 0.17 & 0.80 \end{pmatrix}$$
4. **Emission Probability Matrix $B$ ($3 \times 3$):**
   $$B = \begin{pmatrix} 0.75 & 0.20 & 0.05 \\ 0.15 & 0.60 & 0.25 \\ 0.04 & 0.21 & 0.75 \end{pmatrix}$$
5. **Initial State Prior:** $\pi = [0.72, 0.22, 0.06]^T$.

#### Algorithms:
* **Forward Algorithm ($\alpha$ pass):** Recursively computes sequence likelihood:
  $$\alpha_t(j) = \left[ \sum_{i=1}^N \alpha_{t-1}(i) a_{ij} \right] b_j(O_t), \quad P(O | \lambda) = \sum_{i=1}^N \alpha_T(i)$$
  Mean sequence log-likelihood achieved: **$-9.4537$**.
* **Viterbi Dynamic Programming Algorithm:** Computes the maximal probability hidden state path in log-space:
  $$\delta_t(j) = \max_{1 \le i \le N} \left[ \delta_{t-1}(i) + \ln a_{ij} \right] + \ln b_j(O_t)$$
  $$\psi_t(j) = \arg\max_{1 \le i \le N} \left[ \delta_{t-1}(i) + \ln a_{ij} \right]$$
  Backtracking from $q_T^* = \arg\max_i \delta_T(i)$ yields the complete latent state trajectory $q_1^*, \dots, q_T^*$.

#### Empirical Viterbi Sequence Performance:
* **Sequence Classification Accuracy:** **83.13%**
* **At-Risk State Recall:** **67.82%**
* **Viterbi F1-Score:** **74.57%**
* **Average Earliest Detection Week:** **Week 4.68** (identifies transition into critical risk prior to mid-term examinations).

---

### Unit 5: CART & Ensemble Learning

#### 1. CART (Classification and Regression Trees):
* **Gini Impurity Criterion:**
  $$I_G(p) = 1 - \sum_{k=1}^K p_k^2$$
* **Entropy Information Gain Criterion:**
  $$IG(S, A) = H(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} H(S_v)$$
Cost-complexity pruning constrained tree depth to prevent memorization.

#### 2. Ensemble Learning Architectures:
* **Random Forest (Bagging):** An ensemble of 100 decorrelated decision trees with bootstrap sampling and random feature sub-selection ($m_{\text{try}} = \sqrt{p}$).
* **AdaBoost (Adaptive Boosting):** Iteratively updates student sample weights $w_t(i)$ focusing sequentially on misclassified instances.
* **Gradient Boosted Decision Trees (GBDT):** Sequential functional gradient descent minimizing log-loss pseudo-residuals.
* **Extreme Gradient Boosting (XGBoost):** Regularized boosted trees leveraging exact second-order Taylor gradients:
  $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
* **Soft Voting Meta-Classifier:** Blends probabilistic class predictions across diverse tree and margin models.

---

## 2. Comparative Analysis of the Implemented Algorithms

| Algorithm | Unit | Theoretical Paradigm | Training Time Complexity | Inference Time Complexity | Interpretability | Outlier Sensitivity | Feature Scaling Required? | Bias-Variance Trade-Off |
|:---|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Polynomial Curve Fitting** | Unit 1 | Parametric Regression | $O(N \cdot M^2 + M^3)$ | $O(M)$ | **High** | High (High $M$) | **Yes** (Numerical stability) | Underfit at low $M$, Overfit at high $M$ |
| **Bayesian Logistic Reg.** | Unit 2 | Bayesian Parametric | $O(N \cdot d^2 + d^3)$ | $O(d^2)$ | **High (Odds + Variance)**| Moderate | **Yes** (Priors & Hessian) | Balanced (Controlled by $\alpha$) |
| **SVM (Linear Kernel)** | Unit 2 | Margin-based Hyperplane | $O(N^2 \cdot d)$ | $O(d \cdot |SV|)$ | Moderate (Weights) | Moderate (Slack bounds) | **Strictly Yes** | Low Bias, Low Variance |
| **SVM (RBF Kernel)** | Unit 2 | Kernelized Manifold | $O(N^2 \cdot d)$ to $O(N^3)$| $O(d \cdot |SV|)$ | Low (Dual weights) | Moderate | **Strictly Yes** | Very Low Bias, Moderate Var |
| **SVM (Poly Kernel)** | Unit 2 | Polynomial Kernel | $O(N^2 \cdot d)$ | $O(d \cdot |SV|)$ | Low | Moderate | **Strictly Yes** | Low Bias, Moderate Var |
| **K-Means Clustering** | Unit 3 | Centroid Partitioning | $O(I \cdot K \cdot N \cdot d)$ | $O(K \cdot d)$ | Moderate (Centroids) | High (Mean centroid) | **Strictly Yes** | Unsupervised |
| **Agglomerative Cluster** | Unit 3 | Hierarchical Dendrogram | $O(N^2 \log N)$ | N/A (Transductive) | Moderate (Dendrogram) | Moderate | **Strictly Yes** | Unsupervised |
| **PCA Reduction** | Unit 3 | Spectral Projection | $O(d^3 + N d^2)$ | $O(d \cdot k)$ | Moderate (Loadings) | High (Covariance) | **Strictly Yes** | Unsupervised |
| **HMM & Viterbi** | Unit 4 | Probabilistic Graphical | $O(N \cdot S^2)$ | $O(T \cdot S^2)$ (Viterbi) | **High (Latent Path)** | Low (Transition smoothed)| **No** (Discrete tokens) | Temporal Structural Bias |
| **CART (Gini / Entropy)** | Unit 5 | Non-parametric Recursive | $O(d \cdot N \log N)$ | $O(\text{depth})$ | **High (Visual Tree)** | Low | **No** | Low Bias, High Variance |
| **Random Forest** | Unit 5 | Ensemble (Bagging) | $O(B \cdot d \cdot N \log N)$| $O(B \cdot \text{depth})$ | Moderate (MDI Gain) | **Very Low** | **No** | Low Bias, **Very Low Variance** |
| **AdaBoost** | Unit 5 | Ensemble (Boosting) | $O(T \cdot N \cdot d)$ | $O(T)$ | Moderate | High (Focus on outliers) | **No** | Very Low Bias, Low Variance |
| **Gradient Boosting** | Unit 5 | Functional Gradient | $O(T \cdot d \cdot N \log N)$| $O(T \cdot \text{depth})$ | Moderate | Moderate (Shrinkage) | **No** | Very Low Bias, Low Variance |
| **XGBoost** | Unit 5 | Regularized 2nd-Order | $O(T \cdot d \cdot N \log N)$| $O(T \cdot \text{depth})$ | Moderate | **Very Low** | **No** | Very Low Bias, **Lowest Variance** |
| **Voting Classifier** | Unit 5 | Meta-Ensemble Blending | $O(\sum \text{Models})$ | $O(\sum \text{Infer})$ | Low | **Very Low** | Mixed | **Optimal Compromise** |

---

## 3. Details of Datasets Used for Experimentation

### Schema and Attribute Definitions:
The dataset consists of **1,500 undergraduate student records** with **11 attributes**:

1. **`Age`** (Integer): Chronological age, range: 18 – 26 (Mean: 20.47, Std: 1.68). Uncorrelated with risk ($r = +0.0166$).
2. **`Gender`** (Categorical): Binary self-report (`Male`: 52.0%, `Female`: 48.0%).
3. **`Socio_Economic_Status`** (Categorical): Household tier (`Low`: 28.0%, `Medium`: 52.0%, `High`: 20.0%).
4. **`GPA`** (Float): Cumulative GPA (0.00 to 4.00, Mean: 2.94, Std: 0.53). Pearson $r = -0.7239$.
5. **`Attendance_Rate`** (Float): Class attendance percentage, range: 32.0% – 100.0% (Mean: 77.8%). Pearson $r = -0.7139$.
6. **`Assignments_Submitted`** (Integer): Coursework deliverables completed on time out of 10 (Mean: 7.2). Strongest predictor: **$r = -0.7515$**.
7. **`Online_Learning_Hours`** (Float): Weekly hours on LMS platform, range: 1.0 – 35.0 (Mean: 13.1 hrs/wk). Pearson $r = -0.6698$.
8. **`Forum_Participation`** (Integer): Queries and discussion contributions, range: 0 – 50 (Mean: 8.8). Pearson $r = -0.6100$.
9. **`Lab_Performance`** (Float): Hands-on laboratory examination score, range: 20.0 – 100.0 (Mean: 74.2). Pearson $r = -0.7191$.
10. **`Extracurricular_Participation`** (Categorical): Involvement in athletic/academic clubs (`Yes`: 45.3%, `No`: 54.7%).
11. **`At_Risk`** (Binary Target): Student attrition status (`0` = Safe / Satisfactory progress, `1` = At Risk). Class ratio: **63.53% Safe (953 students) vs. 36.47% At Risk (547 students)**.

### Preprocessing Protocol:
* Categorical features transformed using `OneHotEncoder(drop='first')`.
* Numerical features standardized via `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
* Data partitioned into **80% Training ($N=1,200$)** and **20% Independent Test ($N=300$)** using **Stratified Sampling**.

---

## 4. Empirical Comparison of Results and Best Algorithm Identification

### Master Classification Leaderboard (Units 2, 4, 5):

| Rank | Unit | Algorithm | Accuracy | Precision | Recall (Sensitivity) | F1-Score | ROC-AUC | 5-Fold CV ROC (±Std) | False Neg. Rate | Train Time | Inference Time | Rank Score |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 **1** | **Unit 2** | **Bayesian Logistic Regression** | **96.67%** | **98.06%** | **92.66%** | **95.28%** | **99.76%** | **99.50% (±0.16%)** | **7.34%** | **6.96 ms** | **0.17 ms** | **0.9813** |
| 🥈 **2** | **Unit 2** | **SVM (Linear Kernel)** | **96.67%** | **98.06%** | **92.66%** | **95.28%** | **99.73%** | **99.49% (±0.17%)** | **7.34%** | 42.36 ms | 1.08 ms | **0.9811** |
| 🥉 **3** | **Unit 2** | **SVM (RBF Kernel)** | 96.33% | 97.12% | 92.66% | 94.84% | 99.63% | 99.32% (±0.22%) | 7.34% | 54.27 ms | 5.09 ms | 0.9788 |
| 4 | Unit 5 | Random Forest | 96.00% | 96.19% | 92.66% | 94.39% | 99.51% | 99.01% (±0.36%) | 7.34% | 273.11 ms | 8.76 ms | 0.9759 |
| 5 | Unit 5 | AdaBoost | 95.67% | 95.28% | 92.66% | 93.95% | 99.54% | 99.24% (±0.31%) | 7.34% | 213.36 ms | 8.54 ms | 0.9751 |
| 6 | Unit 2 | SVM (Poly Kernel) | 95.67% | **100.0%** | 88.07% | 93.66% | 99.66% | 99.21% (±0.32%) | 11.93% | 50.18 ms | 2.14 ms | 0.9745 |
| 7 | Unit 5 | XGBoost | 95.67% | 96.15% | 91.74% | 93.90% | 99.52% | 99.00% (±0.35%) | 8.26% | 74.96 ms | 1.85 ms | 0.9742 |
| 8 | Unit 5 | Gradient Boosting | 95.67% | 96.15% | 91.74% | 93.90% | 99.37% | 98.93% (±0.34%) | 8.26% | 491.69 ms | 0.89 ms | 0.9735 |
| 9 | Unit 5 | Voting Classifier | 95.33% | 93.58% | **93.58%** | 93.58% | 99.56% | 98.87% (±0.38%) | **6.42%** | 253.01 ms | 14.34 ms | 0.9729 |
| 10 | Unit 5 | CART (Gini Impurity) | 94.00% | 90.27% | **93.58%** | 91.89% | 96.26% | 94.87% (±0.68%) | **6.42%** | **5.54 ms** | 0.28 ms | 0.9438 |
| 11 | Unit 5 | CART (Entropy) | 94.33% | 95.10% | 88.99% | 91.94% | 95.58% | 95.51% (±0.57%) | 11.01% | **4.02 ms** | 0.23 ms | 0.9429 |

---

### Identification and Pedagogical Justification of the Best Algorithm

#### **The Champion Algorithm:** `Bayesian Logistic Regression` (Unit 2)
Ranked **#1 Overall** with a composite score of **0.9813**.

#### **Justification:**
1. **Optimal Generalization & Discrimination:**
   Bayesian Logistic Regression achieved the top **ROC-AUC of 99.76%** on the test set and **99.50% ($\pm 0.16\%$)** on 5-Fold Cross-Validation, outperforming both complex tree ensembles and standard margin hyperplanes.
2. **Epistemic Uncertainty Quantification:**
   Unlike deterministic models, Bayesian Logistic Regression provides the predictive variance $\sigma_a^2 = {x^*}^T S_N x^*$. If a student displays anomalous or unseen behavioral patterns, the model acknowledges its uncertainty and softens extreme confidence, preventing disastrous false alarms.
3. **Execution Latency:**
   * **Training Time:** `6.96 ms` (39$\times$ faster than Random Forest and 70$\times$ faster than GBDT).
   * **Inference Latency:** `0.17 ms`, making it optimal for large-scale institutional monitoring systems.
4. **Actionable Institutional Odds Ratios:**
   MAP parameters provide direct, interpretable odds multipliers ($e^{w_{\text{MAP}}}$):
   * Assignments: Each drop of 1 standard deviation increases failure odds by **$3.1\times$**.
   * Attendance: Low attendance ($<70\%$) increases failure odds by **$2.8\times$**.
   * GPA: GPA $<2.50$ increases failure odds by **$2.6\times$**.

#### **Temporal Synergy with Unit 4 (Viterbi Decoding):**
In production, **Bayesian Logistic Regression** is paired with the **Unit 4 Hidden Markov Model**:
* **HMM Viterbi** continuously tracks the student's weekly engagement trajectory, generating an early warning alert as early as **Week 4.68**.
* **Bayesian Logistic Regression** computes cross-sectional confirmatory risk probabilities and generates personalized academic intervention plans before mid-terms.
