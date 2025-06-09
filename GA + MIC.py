# Enhanced GA + MIC with Complete Metrics Display and Fixed Computational Cost
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

# ---  MIC Implementation ---
def mic(X, Y, alpha=0.6, c=15):
    """Maximal Information Coefficient calculation"""
    # Handle edge cases
    if len(X) != len(Y) or len(X) == 0:
        return 0
    
    # Normalize data to [0, 1] range
    X_range = np.ptp(X)
    Y_range = np.ptp(Y)
    
    if X_range == 0 or Y_range == 0:
        return 0
    
    X = (X - np.min(X)) / X_range
    Y = (Y - np.min(Y)) / Y_range
    
    B = max(int(len(X)**alpha), 4)  # Ensure minimum B value
    max_mi = 0

    def compute_mi(Pxy, Px, Py):
        """Compute mutual information"""
        nz = Pxy > 0
        if not np.any(nz):
            return 0
        
        i_idx, j_idx = np.where(nz)
        mi_values = Pxy[i_idx, j_idx] * np.log2(Pxy[i_idx, j_idx] / (Px[i_idx] * Py[j_idx]))
        return np.sum(mi_values)

    for x_bins in range(2, min(c + 1, len(X) // 2)):
        for y_bins in range(2, min(c + 1, len(Y) // 2)):
            if x_bins * y_bins > B:
                continue

            # Create bins
            x_binned = np.floor(X * (x_bins - 1e-10)).astype(int)
            y_binned = np.floor(Y * (y_bins - 1e-10)).astype(int)
            
            # Ensure values are within valid range
            x_binned = np.clip(x_binned, 0, x_bins - 1)
            y_binned = np.clip(y_binned, 0, y_bins - 1)

            # Count joint occurrences
            joint_counts = np.zeros((x_bins, y_bins))
            for xb, yb in zip(x_binned, y_binned):
                joint_counts[xb, yb] += 1

            # Calculate probabilities
            Pxy = joint_counts / len(X)
            Px = Pxy.sum(axis=1)
            Py = Pxy.sum(axis=0)

            # Compute mutual information
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
    mic_scores = []
    for i in range(n_features):
        try:
            score = mic(X_scaled[:, i], y, alpha, c)
            mic_scores.append(score)
        except Exception as e:
            print(f"   Warning: Error computing MIC for feature {i}: {e}")
            mic_scores.append(0.0)
    
    mic_scores = np.array(mic_scores)

    # Initialize population
    population = []
    for _ in range(P):
        individual = np.random.randint(2, size=n_features)
        # Ensure at least one feature is selected
        if np.sum(individual) == 0:
            individual[np.random.randint(n_features)] = 1
        population.append(individual)
    population = np.array(population)

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
            
            # Check if we have enough samples for train/test split
            if len(X_sel) < 10:
                fitness_scores.append(0)
                continue
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_sel, y, test_size=0.3, random_state=42, stratify=None
                )

                # Quick model evaluation
                model = lgb.LGBMRegressor(
                    n_estimators=50,  # Reduced for speed
                    learning_rate=0.1, 
                    max_depth=5,
                    num_leaves=15, 
                    random_state=42, 
                    verbose=-1
                )
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                # Fitness calculation
                mse = mean_squared_error(y_test, preds)
                mic_sum = np.sum(mic_scores[selected])
                fitness = mic_sum * (1 / (1 + mse)) if mse > 0 else mic_sum
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    
            except Exception as e:
                print(f"   Warning: Error in fitness evaluation: {e}")
                fitness_scores.append(0)

        # Track convergence
        if fitness_scores:
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
            if np.max(fitness_scores) > 0:
                # Tournament selection
                new_population = []
                for _ in range(P):
                    # Select parents
                    tournament_size = 3
                    candidates = np.random.choice(len(population), tournament_size, replace=False)
                    winner = candidates[np.argmax(fitness_scores[candidates])]
                    new_population.append(population[winner].copy())
                
                # Apply crossover and mutation
                children = []
                for i in range(0, len(new_population), 2):
                    parent1 = new_population[i]
                    parent2 = new_population[min(i+1, len(new_population)-1)]
                    
                    # Crossover
                    if np.random.rand() < c_r and len(parent1) > 1:
                        point = np.random.randint(1, len(parent1))
                        child1 = np.concatenate((parent1[:point], parent2[point:]))
                        child2 = np.concatenate((parent2[:point], parent1[point:]))
                    else:
                        child1, child2 = parent1.copy(), parent2.copy()

                    # Mutation
                    for child in [child1, child2]:
                        for j in range(len(child)):
                            if np.random.rand() < m_r:
                                child[j] = 1 - child[j]
                        # Ensure at least one feature is selected
                        if np.sum(child) == 0:
                            child[np.random.randint(len(child))] = 1
                        children.append(child)

                population = np.array(children[:P])
            else:
                # If all fitness scores are 0, reinitialize population
                population = []
                for _ in range(P):
                    individual = np.random.randint(2, size=n_features)
                    if np.sum(individual) == 0:
                        individual[np.random.randint(n_features)] = 1
                    population.append(individual)
                population = np.array(population)

    total_time = time.time() - start_time

    # Calculate convergence rate
    if len(fitness_history) > 1:
        convergence_rate = (fitness_history[-1] - fitness_history[0]) / len(fitness_history)
    else:
        convergence_rate = 0

    # Ensure we have a valid best individual
    if best_individual is None:
        best_individual = population[0]
    
    selected_features = np.where(best_individual == 1)[0]
    
    # Ensure at least one feature is selected
    if len(selected_features) == 0:
        selected_features = np.array([0])  # Select first feature as fallback

    print(f" GA+MIC completed in {total_time:.2f} seconds")
    print(f" Selected {len(selected_features)} features out of {n_features}")

    return selected_features, mic_scores[selected_features], fitness_history, convergence_rate, total_time

# --- Enhanced LightGBM Runner with Detailed Metrics ---
def run_lgb_detailed(X_data, y_data):
    """Run LightGBM with detailed performance metrics"""
    print(" Training LightGBM model...")

    # Ensure we have enough data for train/test split
    if len(X_data) < 10:
        print(" Warning: Not enough data for proper evaluation")
        return 0, 0, 0, float('inf'), float('inf')

    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42
    )

    model = lgb.LGBMRegressor(
        n_estimators=500,  # Reduced for faster execution
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )

    start_time = time.time()
    try:
        model.fit(X_train, y_train,
                  eval_set=[(X_test, y_test)],
                  eval_metric='l2',
                  callbacks=[lgb.early_stopping(30, verbose=False)])
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
    
    except Exception as e:
        print(f" Error in model training: {e}")
        return 0, 0, time.time() - start_time, float('inf'), float('inf')

# --- Enhanced Dataset Processing with Correct Computational Cost ---
def process_dataset_enhanced(name, df, label_col):
    """Process dataset with complete metrics display and CORRECT computational cost"""
    print(f"\n{'='*60}")
    print(f" GA+MIC ANALYSIS: {name.upper()} DATASET")
    print(f"{'='*60}")
    
    # Start total timing
    total_start_time = time.time()

    try:
        # Data preparation
        preprocessing_start = time.time()
        df_numeric = df.select_dtypes(include=[np.number]).dropna()
        
        if label_col not in df_numeric.columns:
            print(f"  Warning: '{label_col}' not found. Available columns: {list(df_numeric.columns)}")
            label_col = df_numeric.columns[-1]
            print(f"  Using '{label_col}' as target variable")

        X = df_numeric.drop(columns=[label_col])
        y = df_numeric[label_col].values
        preprocessing_time = time.time() - preprocessing_start

        print(f" Dataset Info:")
        print(f"   • Samples: {len(X)}")
        print(f"   • Features: {len(X.columns)}")
        print(f"   • Target: {label_col}")

        # Check if we have enough data
        if len(X) < 50:
            print(" Warning: Dataset too small for reliable analysis")

        # Run GA+MIC feature selection
        selected_idx, selected_scores, fitness_history, convergence_rate, feature_selection_time = ga_mic(X, y)

        print(f"\n SELECTED FEATURES BY GA+MIC:")
        print("-" * 40)
        for i, (idx, score) in enumerate(zip(selected_idx, selected_scores), 1):
            feature_name = X.columns[idx] if idx < len(X.columns) else f"Feature_{idx}"
            print(f"   {i:2d}. {feature_name:<25} MIC={score:.4f}")

        # Train and evaluate model
        model_start_time = time.time()
        X_selected = X.iloc[:, selected_idx]
        X_scaled = StandardScaler().fit_transform(X_selected)
        r2, accuracy_pct, lgb_training_time, mse, mae = run_lgb_detailed(X_scaled, y)
        model_total_time = time.time() - model_start_time

        # Calculate COMPLETE computational cost
        total_computational_cost = time.time() - total_start_time
        
        # Calculate average convergence rate
        avg_convergence_rate = abs(convergence_rate) if convergence_rate != 0 else 0

        # Display all required metrics with CORRECT computational cost
        print(f"\n{'='*60}")
        print(f" FINAL RESULTS - {name.upper()}")
        print(f"{'='*60}")
        print(f"Test R2 Score:                    {r2:.4f}")
        print(f"Test Accuracy Percentage:         {accuracy_pct:.2f}%")
        print(f"Computational Cost (seconds):     {total_computational_cost:.2f}")
        print(f"Average Convergence Rate:         {avg_convergence_rate:.6f}")
        print(f"{'='*60}")

        # Additional metrics with detailed timing breakdown
        print(f"\n ADDITIONAL METRICS:")
        print(f"   • Mean Squared Error:          {mse:.4f}")
        print(f"   • Mean Absolute Error:         {mae:.4f}")
        print(f"   • Features Selected:           {len(selected_idx)}/{len(X.columns)}")
        print(f"   • Feature Reduction:           {(1 - len(selected_idx)/len(X.columns))*100:.1f}%")
        
        # Detailed timing breakdown
        print(f"\n COMPUTATIONAL COST BREAKDOWN:")
        print(f"   • Data Preprocessing:          {preprocessing_time:.2f}s ({preprocessing_time/total_computational_cost*100:.1f}%)")
        print(f"   • Feature Selection (GA+MIC):  {feature_selection_time:.2f}s ({feature_selection_time/total_computational_cost*100:.1f}%)")
        print(f"   • Model Training (LightGBM):   {model_total_time:.2f}s ({model_total_time/total_computational_cost*100:.1f}%)")
        print(f"   • Total Process Time:          {total_computational_cost:.2f}s (100%)")

        return {
            'r2_score': r2,
            'accuracy_percentage': accuracy_pct,
            'computational_cost': total_computational_cost,
            'convergence_rate': avg_convergence_rate,
            'selected_features': len(selected_idx),
            'total_features': len(X.columns),
            'feature_selection_time': feature_selection_time,
            'model_training_time': model_total_time,
            'preprocessing_time': preprocessing_time
        }
        
    except Exception as e:
        print(f" Error processing dataset: {e}")
        total_time = time.time() - total_start_time
        return {
            'r2_score': 0,
            'accuracy_percentage': 0,
            'computational_cost': total_time,
            'convergence_rate': 0,
            'selected_features': 0,
            'total_features': 0,
            'feature_selection_time': 0,
            'model_training_time': 0,
            'preprocessing_time': 0
        }

# --- Generate Sample Data if Files Don't Exist ---
def generate_sample_data(name, n_samples=1000, n_features=15):
    """Generate sample data for testing"""
    print(f" Generating sample data for {name}")
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
    print("  GA+MIC Algo ")
    print("=" * 70)

    # Dataset configurations
    datasets_config = [
        ("California Housing", "/content/sample_data/california_housing_test.csv", "median_house_value"),
        ("Parkinson's Disease", "/content/parkinsons_updrs.data", "total_UPDRS"),
        ("Boston Housing", "/content/BostonHousing.csv", "total_rooms"),
    ]

    results_summary = []

    for name, filepath, target_col in datasets_config:
        try:
            # Try to load the dataset
            df = pd.read_csv(filepath)
            print(f"Loaded {name} dataset from {filepath}")
        except FileNotFoundError:
            print(f" {filepath} not found, generating sample data...")
            df = generate_sample_data(name, n_samples=800, n_features=12)
            target_col = "target"
        except Exception as e:
            print(f" Error loading {name}: {str(e)}")
            print(" Generating sample data instead...")
            df = generate_sample_data(name, n_samples=800, n_features=12)
            target_col = "target"

        # Process the dataset
        try:
            result = process_dataset_enhanced(name, df, target_col)
            results_summary.append((name, result))
        except Exception as e:
            print(f" Error processing {name}: {str(e)}")

    # Final summary with CORRECT computational costs
    print(f"\n{'='*80}")
    print(f" OVERALL SUMMARY - ALL DATASETS (FIXED COMPUTATIONAL COSTS)")
    print(f"{'='*80}")

    total_computation_time = 0
    for name, result in results_summary:
        total_computation_time += result['computational_cost']
        print(f"\n {name}:")  # FIXED: Removed the erroneous int() call
        print(f"   Test R2 Score:                    {result['r2_score']:.4f}")
        print(f"   Test Accuracy Percentage:         {result['accuracy_percentage']:.2f}%")
        print(f"   Computational Cost (seconds):     {result['computational_cost']:.2f}")
        print(f"   Average Convergence Rate:         {result['convergence_rate']:.6f}")
        print(f"   Feature Selection Time:           {result['feature_selection_time']:.2f}s")
        print(f"   Model Training Time:              {result['model_training_time']:.2f}s")

    print(f"\n TOTAL COMPUTATIONAL COST: {total_computation_time:.2f} seconds")
    print(f" Analysis Complete!")
