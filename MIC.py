# MIC 
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
lgb.basic._log_warning = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# --- MIC Implementation ---
def mic(X, Y, alpha=0.6, c=15):
    # Normalize to [0,1]
    X = (X - np.min(X)) / np.ptp(X)
    Y = (Y - np.min(Y)) / np.ptp(Y)

    B = int(len(X)**alpha)
    max_mi = 0

    def compute_mi(Pxy, Px, Py):
        nz = Pxy > 0
        i_idx, j_idx = np.where(nz)
        return np.sum(
            Pxy[i_idx, j_idx] * np.log2(Pxy[i_idx, j_idx] / (Px[i_idx] * Py[j_idx]))
        )

    for x_bins in range(2, c + 1):
        for y_bins in range(2, c + 1):
            if x_bins * y_bins > B:
                continue

            x_binned = np.floor(X * x_bins).astype(int)
            y_binned = np.floor(Y * y_bins).astype(int)
            x_binned[x_binned == x_bins] = x_bins - 1
            y_binned[y_binned == y_bins] = y_bins - 1

            joint_counts = np.zeros((x_bins, y_bins))
            for xb, yb in zip(x_binned, y_binned):
                joint_counts[xb, yb] += 1

            Pxy = joint_counts / len(X)
            Px = Pxy.sum(axis=1)
            Py = Pxy.sum(axis=0)

            mi = compute_mi(Pxy, Px, Py)
            norm_factor = np.log2(min(x_bins, y_bins))
            nmi = mi / norm_factor if norm_factor > 0 else 0

            if nmi > max_mi:
                max_mi = nmi

    return max_mi


# --- Run LightGBM as before ---
def run_lgb(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    start_time = time.time()
    eval_history = {}

    lgb_model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1
    )

    def record_eval(env):
        if 'l2' in env.evaluation_result_list[0][1]:
            epoch = env.iteration
            l2_loss = env.evaluation_result_list[0][2]
            eval_history[epoch] = l2_loss

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric='l2',
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            record_eval
        ]
    )

    runtime = time.time() - start_time
    y_pred = lgb_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    keys = list(eval_history.keys())
    if len(keys) > 5:
        early_vals = np.mean([eval_history[k] for k in keys[:5]])
        final_vals = np.mean([eval_history[k] for k in keys[-5:]])
        convergence_ratio = (1 - final_vals / early_vals) * 100
        convergence_percent = min(round(convergence_ratio, 2), 100.0)
    else:
        convergence_percent = np.nan

    return {
        'r2': r2,
        'accuracy_percent': round(r2 * 100, 2),
        'runtime_sec': round(runtime, 2),
        'avg_convergence_percent': convergence_percent
    }

def process_dataset(name, df, label_col):
    print(f"\n==========  MIC Results for {name} Dataset ==========")

    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    if label_col not in df_numeric.columns:
        label_col = df_numeric.columns[-1]  # fallback

    X = df_numeric.drop(columns=[label_col])
    y = df_numeric[label_col]

    # Feature Engineering (existing)
    if X.shape[1] >= 3:
        cols = X.columns
        X['feat1_x_feat2'] = X[cols[0]] * X[cols[1]]
        X['feat2_x_feat3'] = X[cols[1]] * X[cols[2]]
        X['feat1_x_feat3'] = X[cols[0]] * X[cols[2]]

    # --- MIC-based feature selection ---
    mic_scores = []
    for col in X.columns:
        score = mic(X[col].values, y.values)
        mic_scores.append((col, score))
    mic_scores.sort(key=lambda x: x[1], reverse=True)

    print("\nTop features by MIC score:")
    for f, s in mic_scores[:5]:
        print(f"  {f}: {s:.4f}")

    # Select top 5 features by MIC score
    top_features = [f for f, s in mic_scores[:5]]
    X_selected = X[top_features]

    # Scale selected features
    X_scaled = StandardScaler().fit_transform(X_selected)

    # Run LightGBM on selected features
    full_results = run_lgb(X_scaled, y)
    print("------------------------------------------------------------")
    print(f"Test R2 Score: {full_results['r2']:.4f}")
    print(f"Test Accuracy Percentage: {full_results['accuracy_percent']}%")
    print(f"Computational Cost (seconds): {full_results['runtime_sec']}")
    print(f"Average Convergence Rate: {full_results['avg_convergence_percent']}%")

    # Scalability Evaluation
    print("\n Scalability Test Results:")
    print("-------------------------------------------------------------------")
    print(f"{'Data %':<10}{'Samples':<10}{'Runtime (s)':<15}{'Accuracy (%)':<17}{'Conv. Rate (%)'}")
    print("-------------------------------------------------------------------")

    fractions = [0.1, 0.3, 0.5, 0.7, 1.0]
    for frac in fractions:
        n_samples = int(len(X_scaled) * frac)
        X_sub = X_scaled[:n_samples]
        y_sub = y[:n_samples]
        res = run_lgb(X_sub, y_sub)
        print(f"{int(frac*100):<10}{n_samples:<10}{res['runtime_sec']:<15}{res['accuracy_percent']:<17}{res['avg_convergence_percent']}")

# === Load Boston Housing Dataset ===
df_boston = pd.read_csv("/content/sample_data/BostonHousing.csv")
process_dataset("Boston Housing", df_boston, label_col="medv")

# === Load Parkinson’s Dataset ===
df_parkinsons = pd.read_csv("/content/parkinsons_updrs.data")
process_dataset("Parkinson's", df_parkinsons, label_col="total_UPDRS")

# === Load California Housing Dataset ===
df_california = pd.read_csv("/content/sample_data/california_housing_test.csv")
process_dataset("California Housing", df_california, label_col="total_rooms")
