"""
Unit 4: Hidden Markov Models (HMM) & Viterbi Decoding Algorithm
Concepts Covered:
1. Hidden Markov Model formulation: lambda = (pi, A, B)
   - Hidden States S: {S0: Safe/Engaged, S1: Moderate Warning, S2: Critical At-Risk}
   - Observable Emissions V: {V0: High Weekly Activity, V1: Moderate Activity, V2: Low/Inactive}
   - Initial State Distribution pi
   - State Transition Matrix A: P(q_{t+1} = Sj | q_t = Si)
   - Emission Probability Matrix B: P(O_t = Vk | q_t = Sj)
2. Evaluation Problem: Forward Algorithm (alpha pass) & Log-Likelihood log P(O|lambda)
3. Posterior State Estimation: Backward Algorithm (beta pass) & Gamma_t(i) marginals
4. Decoding Problem: The Viterbi Dynamic Programming Algorithm
   - Delta recursion & Backtracking path pointers (psi)
   - Tracing weekly academic state evolution and detecting earliest risk entry week
"""

import numpy as np
import pandas as pd

class HiddenMarkovModel:
    def __init__(self, states=None, observations=None, pi=None, A=None, B=None):
        self.states = states or ['Safe_Engaged', 'Moderate_Warning', 'Critical_Risk']
        self.observations = observations or ['High_Activity', 'Moderate_Activity', 'Low_Inactive']
        self.N = len(self.states)
        self.M = len(self.observations)
        
        # Default pedagogical transition and emission parameters
        # Pi: Initial distribution (start of semester)
        self.pi = np.array([0.72, 0.22, 0.06]) if pi is None else np.asarray(pi, dtype=float)
        
        # A: Transition matrix P(State_t+1 | State_t)
        self.A = np.array([
            [0.82, 0.15, 0.03],  # Safe -> Safe (82%), Warning (15%), Critical (3%)
            [0.15, 0.65, 0.20],  # Warning -> Safe (15%), Warning (65%), Critical (20%)
            [0.03, 0.17, 0.80]   # Critical -> Safe (3%), Warning (17%), Critical (80%)
        ]) if A is None else np.asarray(A, dtype=float)
        
        # B: Emission matrix P(Obs_k | State_j)
        self.B = np.array([
            [0.75, 0.20, 0.05],  # Safe emits High (75%), Mod (20%), Low (5%)
            [0.15, 0.60, 0.25],  # Warning emits High (15%), Mod (60%), Low (25%)
            [0.04, 0.21, 0.75]   # Critical emits High (4%), Mod (21%), Low (75%)
        ]) if B is None else np.asarray(B, dtype=float)

    def forward(self, obs_seq: list) -> tuple:
        """
        Forward Algorithm: Computes alpha_t(i) = P(O_1...O_t, q_t = S_i | lambda)
        Returns:
            alpha: matrix of shape (T, N)
            log_likelihood: float
        """
        T = len(obs_seq)
        alpha = np.zeros((T, self.N))
        
        # Initialization
        alpha[0, :] = self.pi * self.B[:, obs_seq[0]]
        
        # Induction
        for t in range(1, T):
            for j in range(self.N):
                alpha[t, j] = np.sum(alpha[t - 1, :] * self.A[:, j]) * self.B[j, obs_seq[t]]
                
        p_obs = np.sum(alpha[T - 1, :])
        log_likelihood = float(np.log(p_obs + 1e-300))
        return alpha, log_likelihood

    def backward(self, obs_seq: list) -> np.ndarray:
        """
        Backward Algorithm: Computes beta_t(i) = P(O_{t+1}...O_T | q_t = S_i, lambda)
        Returns:
            beta: matrix of shape (T, N)
        """
        T = len(obs_seq)
        beta = np.zeros((T, self.N))
        beta[T - 1, :] = 1.0
        
        for t in range(T - 2, -1, -1):
            for i in range(self.N):
                beta[t, i] = np.sum(self.A[i, :] * self.B[:, obs_seq[t + 1]] * beta[t + 1, :])
                
        return beta

    def posterior_gamma(self, obs_seq: list) -> np.ndarray:
        """
        Computes marginal posterior probabilities gamma_t(i) = P(q_t = S_i | O, lambda).
        """
        alpha, _ = self.forward(obs_seq)
        beta = self.backward(obs_seq)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-12
        return gamma

    def viterbi(self, obs_seq: list) -> tuple:
        """
        The Viterbi Dynamic Programming Algorithm (performed in log-space for numerical stability).
        Finds the globally optimal hidden state sequence q* = argmax P(Q, O | lambda).
        Returns:
            best_path: list of decoded state indices
            best_state_names: list of decoded state names
            best_prob: float (probability of optimal sequence)
            delta: dynamic programming lattice
        """
        T = len(obs_seq)
        log_pi = np.log(self.pi + 1e-12)
        log_A = np.log(self.A + 1e-12)
        log_B = np.log(self.B + 1e-12)
        
        delta = np.zeros((T, self.N))
        psi = np.zeros((T, self.N), dtype=int)
        
        # 1. Initialization (t = 0)
        delta[0, :] = log_pi + log_B[:, obs_seq[0]]
        psi[0, :] = 0
        
        # 2. Recursion (t = 1 ... T-1)
        for t in range(1, T):
            for j in range(self.N):
                candidate_probs = delta[t - 1, :] + log_A[:, j]
                psi[t, j] = int(np.argmax(candidate_probs))
                delta[t, j] = candidate_probs[psi[t, j]] + log_B[j, obs_seq[t]]
                
        # 3. Termination
        best_path_prob = float(np.max(delta[T - 1, :]))
        best_last_state = int(np.argmax(delta[T - 1, :]))
        
        # 4. Path Backtracking
        best_path = [best_last_state] * T
        for t in range(T - 2, -1, -1):
            best_path[t] = psi[t + 1, best_path[t + 1]]
            
        best_state_names = [self.states[s] for s in best_path]
        return best_path, best_state_names, np.exp(best_path_prob), delta

def generate_weekly_student_traces(df: pd.DataFrame, num_weeks: int = 10, random_state: int = 42):
    """
    Transforms cross-sectional student metrics into realistic 10-week observational sequences
    reflecting their Attendance, LMS hours, and assignment progress.
    Returns:
        sequences: list of list of token indices [0, 1, 2]
        ground_truth_states: list of expected terminal states [0, 1, 2]
    """
    np.random.seed(random_state)
    sequences = []
    terminal_states = []
    
    for _, row in df.iterrows():
        is_risk = int(row['At_Risk'])
        att = float(row['Attendance_Rate'])
        lms = float(row['Online_Learning_Hours'])
        
        seq = []
        for w in range(num_weeks):
            # Mid-term fatigue / drop factor for struggling students
            fatigue = (w / num_weeks) * (0.35 if is_risk else 0.05)
            
            p_high = np.clip((att / 100.0) * 0.7 + (lms / 35.0) * 0.3 - fatigue, 0.05, 0.90)
            p_low = np.clip((1.0 - p_high) * (0.75 if is_risk else 0.25), 0.05, 0.85)
            p_med = np.clip(1.0 - p_high - p_low, 0.05, 0.90)
            
            # Normalize
            probs = np.array([p_high, p_med, p_low])
            probs /= np.sum(probs)
            
            obs_token = np.random.choice([0, 1, 2], p=probs)
            seq.append(int(obs_token))
            
        sequences.append(seq)
        # Expected state
        terminal_states.append(2 if is_risk == 1 else 0)
        
    return sequences, terminal_states

def evaluate_hmm_viterbi_pipeline(df: pd.DataFrame):
    """
    Evaluates HMM Forward and Viterbi algorithms on student trajectory sequences.
    """
    hmm = HiddenMarkovModel()
    sequences, true_terminals = generate_weekly_student_traces(df, num_weeks=10)
    
    decoded_terminal_states = []
    log_likelihoods = []
    earliest_risk_weeks = []
    
    for seq in sequences:
        _, log_lik = hmm.forward(seq)
        path, _, _, _ = hmm.viterbi(seq)
        
        log_likelihoods.append(log_lik)
        terminal_state = path[-1]
        decoded_terminal_states.append(terminal_state)
        
        # Identify first week student entered Critical Risk (State 2)
        risk_weeks = [w + 1 for w, s in enumerate(path) if s == 2]
        earliest_risk_weeks.append(risk_weeks[0] if risk_weeks else -1)
        
    # Map binary risk: State 2 (Critical) or State 1 (Warning) -> 1, State 0 (Safe) -> 0
    pred_binary_risk = [1 if s >= 1 else 0 for s in decoded_terminal_states]
    actual_binary_risk = [1 if s == 2 else 0 for s in true_terminals]
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    metrics = {
        'Viterbi_Sequence_Accuracy': round(float(accuracy_score(actual_binary_risk, pred_binary_risk)), 4),
        'Viterbi_Precision': round(float(precision_score(actual_binary_risk, pred_binary_risk, zero_division=0)), 4),
        'Viterbi_Recall': round(float(recall_score(actual_binary_risk, pred_binary_risk, zero_division=0)), 4),
        'Viterbi_F1_Score': round(float(f1_score(actual_binary_risk, pred_binary_risk, zero_division=0)), 4),
        'Mean_Sequence_Log_Likelihood': round(float(np.mean(log_likelihoods)), 4),
        'Avg_Earliest_Detection_Week': round(float(np.mean([w for w in earliest_risk_weeks if w > 0])), 2)
    }
    
    return hmm, metrics, earliest_risk_weeks

def plot_unit4_visual(hmm, earliest_weeks, out_path):
    """Generates a comprehensive 4-panel visual report for Unit 4 HMM & Viterbi."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#ffffff')
    
    # 1. State Transition Matrix A
    sns.heatmap(hmm.A, annot=True, fmt='.2f', cmap='Blues', ax=axes[0, 0],
                xticklabels=['Safe', 'Warning', 'Critical'],
                yticklabels=['Safe', 'Warning', 'Critical'], cbar=False)
    axes[0, 0].set_title("HMM State Transition Matrix A (P(S_t+1 | S_t))", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Current State", fontsize=10)
    axes[0, 0].set_xlabel("Next State", fontsize=10)
    
    # 2. Emission Probability Matrix B
    sns.heatmap(hmm.B, annot=True, fmt='.2f', cmap='Greens', ax=axes[0, 1],
                xticklabels=['High Act.', 'Mod Act.', 'Low/Inact.'],
                yticklabels=['Safe', 'Warning', 'Critical'], cbar=False)
    axes[0, 1].set_title("HMM Emission Matrix B (P(Emission | State))", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Hidden Psychological State", fontsize=10)
    axes[0, 1].set_xlabel("Observable Weekly LMS Behavior", fontsize=10)
    
    # 3. Sample Viterbi Decoded Trajectories over 10 Weeks
    weeks = list(range(1, 11))
    seq_a = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]  # High Engaged
    path_a, _, _, _ = hmm.viterbi(seq_a)
    seq_b = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # Slipping
    path_b, _, _, _ = hmm.viterbi(seq_b)
    seq_c = [1, 2, 2, 2, 2, 2, 2, 2, 2, 2]  # Chronic At-Risk
    path_c, _, _, _ = hmm.viterbi(seq_c)
    
    axes[1, 0].plot(weeks, path_a, marker='o', color='#2ecc71', lw=2.5, label='Student 1: High Engaged (Safe)')
    axes[1, 0].plot(weeks, path_b, marker='s', color='#e67e22', lw=2.5, label='Student 2: Slipping (Warning -> Critical)')
    axes[1, 0].plot(weeks, path_c, marker='^', color='#e74c3c', lw=2.5, label='Student 3: Chronic At-Risk')
    axes[1, 0].axvline(x=4.68, color='#c0392b', linestyle='--', lw=1.8, label='Avg Alert Week (W4.68)')
    axes[1, 0].set_yticks([0, 1, 2])
    axes[1, 0].set_yticklabels(['Safe (0)', 'Warning (1)', 'Critical (2)'])
    axes[1, 0].set_xticks(weeks)
    axes[1, 0].set_title("Viterbi Dynamic Latent State Trajectories (10 Weeks)", fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel("Semester Week", fontsize=10)
    axes[1, 0].set_ylabel("Decoded Hidden State", fontsize=10)
    axes[1, 0].grid(True, linestyle=':', alpha=0.6)
    axes[1, 0].legend(loc='upper left', fontsize=9)
    
    # 4. Earliest Hazard Detection Week Distribution
    valid_weeks = [w for w in earliest_weeks if w > 0]
    if valid_weeks:
        axes[1, 1].hist(valid_weeks, bins=np.arange(0.5, 11.5, 1), color='#3498db', edgecolor='black', rwidth=0.8)
        axes[1, 1].axvline(np.mean(valid_weeks), color='#e74c3c', linestyle='--', lw=2,
                           label=f'Mean Early Detection: Week {np.mean(valid_weeks):.2f}')
    axes[1, 1].set_xticks(weeks)
    axes[1, 1].set_title("Cohort Earliest Risk Detection Week Distribution", fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel("First Week Flagged At-Risk", fontsize=10)
    axes[1, 1].set_ylabel("Number of Students", fontsize=10)
    axes[1, 1].grid(True, linestyle=':', alpha=0.6)
    axes[1, 1].legend(fontsize=9)
    
    plt.suptitle("Unit 4: Hidden Markov Models (HMM) & Viterbi Dynamic Decoding Analysis", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    import os
    import sys
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)
        
    from src.preprocessing import load_data
    
    data_path = os.path.join(base_dir, 'data', 'academic_risk_data.csv')
    df = load_data(data_path)
    
    print("=" * 75)
    print(" UNIT 4: HIDDEN MARKOV MODELS (HMM) & VITERBI DECODING ALGORITHM")
    print("=" * 75)
    
    hmm, metrics, earliest_weeks = evaluate_hmm_viterbi_pipeline(df)
    
    print("\n[+] Hidden Markov Model Architecture:")
    print("  * Hidden States S:       ['Safe_Engaged', 'Moderate_Warning', 'Critical_Risk']")
    print("  * Observable Emissions V: ['High_Activity', 'Moderate_Activity', 'Low_Inactive']")
    print("\n[+] State Transition Probability Matrix A (Rows: Current State, Cols: Next State):")
    print(np.round(hmm.A, 2))
    print("\n[+] Emission Probability Matrix B (Rows: Hidden State, Cols: Emission Token):")
    print(np.round(hmm.B, 2))
    
    print("\n[+] Forward Algorithm & Viterbi Sequence Evaluation on Student Cohort:")
    print(f"  * Sequence Classification Accuracy:   {metrics['Viterbi_Sequence_Accuracy']*100:.2f}%")
    print(f"  * Sensitivity / Recall on At-Risk:    {metrics['Viterbi_Recall']*100:.2f}%")
    print(f"  * Sequence F1-Score:                  {metrics['Viterbi_F1_Score']*100:.2f}%")
    print(f"  * Average Earliest Warning Week:      Week {metrics['Avg_Earliest_Detection_Week']}")
    print(f"  * Average Sequence Log-Likelihood:    {metrics['Mean_Sequence_Log_Likelihood']}")
    
    # Showcase Viterbi dynamic decoding on a sample student sequence
    sample_obs = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]  # Dropping activity
    path, path_names, prob, lattice = hmm.viterbi(sample_obs)
    print("\n[+] Sample Viterbi Trace for a Student with Dropping Activity [High, High, Mod, Mod, Mod, Low...]:")
    for w, (token, state) in enumerate(zip(sample_obs, path_names)):
        t_name = hmm.observations[token]
        print(f"  Week {w+1:2d}: Observed = {t_name:<18} -> Viterbi Decoded Hidden State = {state}")
    
    # Save & Open Graph
    plot_file = os.path.join(base_dir, 'results', 'unit4_hmm_viterbi.png')
    plot_unit4_visual(hmm, earliest_weeks, plot_file)
    print(f"\n[+] Visual comparison graph saved to: {plot_file}")
    
    try:
        os.startfile(plot_file)
        print("[+] Opening graph in your default image viewer...")
    except Exception:
        pass
    print("=" * 75)
