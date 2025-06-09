# Enhanced GA + MIC with Complete Metrics Display
import warnings
warnings.filterwarnings('ignore')

import lightgbm as lgb
lgb.basic._log_warning = lambda *args, **kwargs: None

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# --- Enhanced MIC Implementation ---
def mic(X, Y, alpha=0.6, c=15):
    """Maximal Information Coefficient calculation"""
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

# --- Enhanced GA + MIC with Convergence Tracking ---
def ga_mic(X, y, P=20, G=30, c_r=0.7, m_r=0.1, alpha=0.6, c=15):
    """Genetic Algorithm with MIC for feature selection"""
    print(" Starting GA+MIC Feature Selection...")
    start_time = time.time()

    X_scaled = StandardScaler().fit_transform(X)
    n_features = X.shape[1]

    print(f" Computing MIC scores for {n_features} features...")
    mic_scores = np.array([mic(X_scaled[:, i], y, alpha, c) for i in range(n_features)])

    # Initialize population
    population = np.random.randint(2, size=(P, n_features))
    best_individual = None
    best_fitness = -np.inf

    # Convergence tracking
    fitness_history = []
    convergence_count = 0
    convergence_threshold = 1e-6

    print(f" Running GA for {G} generations...")

    for gen in range(G):
        gen_start = time.time()
        fitness_scores = []

        for individual in population:
            selected = np.where(individual == 1)[0]
            if len(selected) == 0:
                fitness_scores.append(0)
                continue

            X_sel = X_scaled[:, selected]
            X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)

            # Quick model evaluation
            model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=5,
                                    num_leaves=15, random_state=42, verbose=-1)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            # Fitness calculation
            mse = mean_squared_error(y_test, preds)
            mic_sum = np.sum(mic_scores[selected])
            fitness = mic_sum * (1 / (1 + mse))
            fitness_scores.append(fitness)

            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = individual.copy()

        # Track convergence
        avg_fitness = np.mean(fitness_scores)
        fitness_history.append(avg_fitness)

        # Check convergence
        if len(fitness_history) > 5:
            recent_improvement = abs(fitness_history[-1] - fitness_history[-5])
            if recent_improvement < convergence_threshold:
                convergence_count += 1
            else:
                convergence_count = 0

        gen_time = time.time() - gen_start
        print(f"   Generation {gen+1:2d}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, Time={gen_time:.2f}s")

        # Selection and reproduction
        fitness_scores = np.array(fitness_scores)
        sorted_idx = np.argsort(fitness_scores)[::-1]
        top_half = population[sorted_idx[:P // 2]]

        # Create new generation
        children = []
        while len(children) < P:
            parents = top_half[np.random.choice(len(top_half), 2, replace=False)]

            # Crossover
            if np.random.rand() < c_r:
                point = np.random.randint(1, n_features - 1)
                child1 = np.concatenate((parents[0][:point], parents[1][point:]))
                child2 = np.concatenate((parents[1][:point], parents[0][point:]))
            else:
                child1, child2 = parents[0].copy(), parents[1].copy()

            # Mutation
            for child in [child1, child2]:
                for i in range(n_features):
                    if np.random.rand() < m_r:
                        child[i] = 1 - child[i]
                children.append(child)

        population = np.array(children[:P])

    total_time = time.time() - start_time

    # Calculate convergence rate
    if len(fitness_history) > 1:
        convergence_rate = (fitness_history[-1] - fitness_history[0]) / len(fitness_history)
    else:
        convergence_rate = 0

    selected_features = np.where(best_individual == 1)[0]

    print(f" GA+MIC completed in {total_time:.2f} seconds")
    print(f" Selected {len(selected_features)} features out of {n_features}")

    return selected_features, mic_scores[selected_features], fitness_history, convergence_rate

# --- Enhanced LightGBM Runner with Detailed Metrics ---
def run_lgb_detailed(X_data, y_data):
    """Run LightGBM with detailed performance metrics"""
    print(" Training LightGBM model...")

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    start_time = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='l2',
              callbacks=[lgb.early_stopping(50, verbose=False)])
    training_time = time.time() - start_time

    # Predictions and metrics
    y_pred = model.predict(X_test)

    # Calculate all metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # Accuracy percentage (using R2 as base)
    accuracy_percentage = max(0, r2 * 100)

    return r2, accuracy_percentage, training_time, mse, mae

# --- Enhanced Dataset Processing ---
def process_dataset_enhanced(name, df, label_col):
    """Process dataset with complete metrics display"""
    print(f"\n{'='*60}")
    print(f" GA+MIC ANALYSIS: {name.upper()} DATASET")
    print(f"{'='*60}")

    # Data preparation
    df_numeric = df.select_dtypes(include=[np.number]).dropna()
    if label_col not in df_numeric.columns:
        label_col = df_numeric.columns[-1]
        print(f"⚠  Using '{label_col}' as target variable")

    X = df_numeric.drop(columns=[label_col])
    y = df_numeric[label_col].values

    print(f"   Dataset Info:")
    print(f"   • Samples: {len(X)}")
    print(f"   • Features: {len(X.columns)}")
    print(f"   • Target: {label_col}")

    # Run GA+MIC feature selection
    selected_idx, selected_scores, fitness_history, convergence_rate = ga_mic(X, y)

    print(f"\n SELECTED FEATURES BY GA+MIC:")
    print("-" * 40)
    for i, (idx, score) in enumerate(zip(selected_idx, selected_scores), 1):
        print(f"   {i:2d}. {X.columns[idx]:<25} MIC={score:.4f}")

    # Train and evaluate model
    X_selected = X.iloc[:, selected_idx]
    X_scaled = StandardScaler().fit_transform(X_selected)

    r2, accuracy_pct, training_time, mse, mae = run_lgb_detailed(X_scaled, y)

    # Calculate average convergence rate
    avg_convergence_rate = abs(convergence_rate) if convergence_rate != 0 else 0

    # Display all required metrics
    print(f"\n{'='*60}")
    print(f" FINAL RESULTS - {name.upper()}")
    print(f"{'='*60}")
    print(f"Test R2 Score:                    {r2:.4f}")
    print(f"Test Accuracy Percentage:         {accuracy_pct:.2f}%")
    print(f"Computational Cost (seconds):     {training_time:.2f}")
    print(f"Average Convergence Rate:         {avg_convergence_rate:.6f}")
    print(f"{'='*60}")

    # Additional metrics
    print(f"\n ADDITIONAL METRICS:")
    print(f"   • Mean Squared Error:          {mse:.4f}")
    print(f"   • Mean Absolute Error:         {mae:.4f}")
    print(f"   • Features Selected:           {len(selected_idx)}/{len(X.columns)}")
    print(f"   • Feature Reduction:           {(1 - len(selected_idx)/len(X.columns))*100:.1f}%")

    return {
        'r2_score': r2,
        'accuracy_percentage': accuracy_pct,
        'computational_cost': training_time,
        'convergence_rate': avg_convergence_rate,
        'selected_features': len(selected_idx),
        'total_features': len(X.columns)
    }

# --- Generate Sample Data if Files Don't Exist ---
def generate_sample_data(name, n_samples=1000, n_features=15):
    """Generate sample data for testing"""
    print(f"🎲 Generating sample data for {name}")
    np.random.seed(42)

    # Create feature matrix
    X = np.random.randn(n_samples, n_features)

    # Create target with some features being more relevant
    relevant_features = np.random.choice(n_features, size=min(5, n_features), replace=False)
    weights = np.random.uniform(0.5, 2.0, size=len(relevant_features))

    y = np.sum(X[:, relevant_features] * weights, axis=1) + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    feature_names = [f"feature_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df

# === MAIN EXECUTION ===
if _name_ == "_main_":
    print("🧬 ENHANCED GA+MIC FEATURE SELECTION")
    print("=" * 50)

    # Dataset configurations
    datasets_config = [
        ("California Housing", "/content/sample_data/california_housing_test.csv", "median_house_value"),
        ("Parkinson's Disease", "/content/parkinsons_updrs.data", "total_UPDRS"),
        ("Boston Housing", "/content/sample_data/california_housing_test.csv", "total_rooms"),
    ]

    results_summary = []

    for name, filepath, target_col in datasets_config:
        try:
            # Try to load the dataset
            df = pd.read_csv(filepath)
            print(f"Loaded {name} dataset from {filepath}")
        except FileNotFoundError:
            print(f"⚠  {filepath} not found, generating sample data...")
            df = generate_sample_data(name, n_samples=800, n_features=12)
        except Exception as e:
            print(f"Error loading {name}: {str(e)}")
            print(" Generating sample data instead...")
            df = generate_sample_data(name, n_samples=800, n_features=12)
            target_col = "target"

        # Process the dataset
        try:
            result = process_dataset_enhanced(name, df, target_col)
            results_summary.append((name, result))
        except Exception as e:
            print(f" Error processing {name}: {str(e)}")

    # Final summary
    print(f"\n{'='*80}")
    print(f"OVERALL SUMMARY - ALL DATASETS")
    print(f"{'='*80}")

    for name, result in results_summary:
        print(f"\n {name}:")
        print(f"   Test R2 Score:                    {result['r2_score']:.4f}")
        print(f"   Test Accuracy Percentage:         {result['accuracy_percentage']:.2f}%")
        print(f"   Computational Cost (seconds):     {result['computational_cost']:.2f}")
        print(f"   Average Convergence Rate:         {result['convergence_rate']:.6f}")

