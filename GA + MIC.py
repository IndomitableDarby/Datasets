# GA + MIC  
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
lgb.basic._log_warning = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# --- MIC Implementation ---
def mic(X, Y, alpha=0.6, c=15):
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

# --- GA + MIC ---
def ga_mic(X, y, P=20, G=30, c_r=0.7, m_r=0.1, alpha=0.6, c=15):
    X_scaled = StandardScaler().fit_transform(X)
    n_features = X.shape[1]
    mic_scores = np.array([mic(X_scaled[:, i], y, alpha, c) for i in range(n_features)])

    population = np.random.randint(2, size=(P, n_features))
    best_individual = None
    best_fitness = -np.inf

    for gen in range(G):
        fitness_scores = []
        for individual in population:
            selected = np.where(individual == 1)[0]
            if len(selected) == 0:
                fitness_scores.append(0)
                continue
            X_sel = X_scaled[:, selected]
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
            model = lgb.LGBMRegressor()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            mic_sum = np.sum(mic_scores[selected])
            fitness = mic_sum * (1 / (1 + mse))
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        fitness_scores = np.array(fitness_scores)
        sorted_idx = np.argsort(fitness_scores)[::-1]
        top_half = population[sorted_idx[:P // 2]]

        children = []
        while len(children) < P:
            parents = top_half[np.random.choice(len(top_half), 2, replace=False)]
            if np.random.rand() < c_r:
                point = np.random.randint(1, n_features - 1)
                child1 = np.concatenate((parents[0][:point], parents[1][point:]))
                child2 = np.concatenate((parents[1][:point], parents[0][point:]))
            else:
                child1, child2 = parents[0].copy(), parents[1].copy()
            for child in [child1, child2]:
                for i in range(n_features):
                    if np.random.rand() < m_r:
                        child[i] = 1 - child[i]
                children.append(child)

        population = np.array(children[:P])

    selected_features = np.where(best_individual == 1)[0]
    return selected_features, mic_scores[selected_features]

# --- LightGBM Runner ---
def run_lgb(X_data, y_data):
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    model = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.03, max_depth=7, num_leaves=31,
                              subsample=0.8, colsample_bytree=0.8, random_state=42)
    start = time.time()
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='l2', callbacks=[lgb.early_stopping(50)])
    duration = time.time() - start
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return round(r2 * 100, 2), round(duration, 2)

# --- Full Evaluation ---
def process_dataset(name, df, label_col):
    print(f"\n========== 🚀 GA+MIC Results for {name} Dataset ==========")

    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    if label_col not in df_numeric.columns:
        label_col = df_numeric.columns[-1]

    X = df_numeric.drop(columns=[label_col])
    y = df_numeric[label_col].values

    selected_idx, selected_scores = ga_mic(X, y)
    print("\nSelected Features by GA+MIC:")
    for idx, score in zip(selected_idx, selected_scores):
        print(f"  {X.columns[idx]}: MIC={score:.4f}")

    X_selected = X.iloc[:, selected_idx]
    X_scaled = StandardScaler().fit_transform(X_selected)
    acc, cost = run_lgb(X_scaled, y)
    print("\n----------------------")
    print(f"Test Accuracy: {acc}%")
    print(f"Runtime: {cost} seconds")

# === Load and Evaluate ===
df_boston = pd.read_csv("/content/sample_data/BostonHousing.csv")
process_dataset("Boston Housing", df_boston, label_col="median_house_value")

df_parkinsons = pd.read_csv("/content/parkinsons_updrs.data")
process_dataset("Parkinson's", df_parkinsons, label_col="total_UPDRS")

df_california = pd.read_csv("/content/sample_data/california_housing_test.csv")
process_dataset("California Housing", df_california, label_col="total_rooms")
