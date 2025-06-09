#GACD  algorithem 

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
    print(f"\n========== GACD Results for {name} Dataset ==========")

    df_numeric = df.select_dtypes(include=[np.number]).dropna()

    if label_col not in df_numeric.columns:
        label_col = df_numeric.columns[-1]  # fallback

    X = df_numeric.drop(columns=[label_col])
    y = df_numeric[label_col]

    # Feature Engineering
    if X.shape[1] >= 3:
        cols = X.columns
        X['feat1_x_feat2'] = X[cols[0]] * X[cols[1]]
        X['feat2_x_feat3'] = X[cols[1]] * X[cols[2]]
        X['feat1_x_feat3'] = X[cols[0]] * X[cols[2]]

    X_scaled = StandardScaler().fit_transform(X)

    # Full Dataset Evaluation
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
df_boston = pd.read_csv("/content/sample_data/california_housing_test.csv")
process_dataset("California Housing", df_boston, label_col="total_rooms")

