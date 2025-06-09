import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def fast_pso_selection(X, y, max_features=10, S=8, T=5):
    print(f"   Starting PSO with {X.shape[1]} features...")

    # Track convergence for average convergence rate
    convergence_history = []

    # Pre-compute correlations once
    n_features = X.shape[1]

    # Handle correlation computation safely
    try:
        # Add small noise to avoid perfect correlations
        X_noisy = X + np.random.normal(0, 1e-10, X.shape)
        combined = np.column_stack([X_noisy, y.reshape(-1, 1)])
        corr_matrix = np.corrcoef(combined.T)

        # Extract target correlations safely
        target_corr = np.abs(corr_matrix[:-1, -1])
        target_corr = np.nan_to_num(target_corr, nan=0.0)

        # Feature correlation matrix
        feature_corr = np.abs(corr_matrix[:-1, :-1])
        feature_corr = np.nan_to_num(feature_corr, nan=0.0)

    except Exception as e:
        print(f"    Correlation computation failed: {e}")
        # Fallback: use simple variance-based selection
        feature_var = np.var(X, axis=0)
        top_features = np.argsort(feature_var)[-min(max_features, n_features):]
        return top_features, 0.0  # Return 0 convergence rate for fallback

    def fast_fitness(position):
        selected = np.where(position == 1)[0]
        if len(selected) == 0 or len(selected) > max_features:
            return 0

        # Relevance: average correlation with target
        relevance = np.mean(target_corr[selected]) if len(selected) > 0 else 0

        # Redundancy: average inter-feature correlation
        if len(selected) > 1:
            selected_corr = feature_corr[np.ix_(selected, selected)]
            redundancy = (np.sum(selected_corr) - np.trace(selected_corr)) / (len(selected) * (len(selected) - 1))
        else:
            redundancy = 0

        # Simple fitness: relevance - redundancy penalty
        fitness = relevance - 0.3 * redundancy
        return max(fitness, 0)

    # Initialize swarm with better strategy
    swarm_pos = np.random.randint(2, size=(S, n_features))

    # Ensure each particle has at least one feature selected
    for i in range(S):
        if np.sum(swarm_pos[i]) == 0:
            # Select top correlated features
            top_idx = np.argsort(target_corr)[-min(3, n_features):]
            swarm_pos[i][top_idx] = 1

    swarm_vel = np.random.uniform(-1, 1, size=(S, n_features))

    # Personal and global best
    pbest_pos = swarm_pos.copy()
    pbest_fit = np.array([fast_fitness(pos) for pos in swarm_pos])

    if np.max(pbest_fit) == 0:
        print("     All fitness values are 0, using correlation fallback")
        top_features = np.argsort(target_corr)[-min(max_features, n_features):]
        return top_features, 0.0

    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()

    # Initialize convergence tracking
    initial_fitness = pbest_fit[gbest_idx]
    convergence_history.append(initial_fitness)

    print(f"     Initial best fitness: {initial_fitness:.4f}")

    # Quick PSO iterations
    w, c1, c2 = 0.5, 1.2, 1.2
    for iteration in range(T):
        for i in range(S):
            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            swarm_vel[i] = (w * swarm_vel[i] +
                           c1 * r1 * (pbest_pos[i] - swarm_pos[i]) +
                           c2 * r2 * (gbest_pos - swarm_pos[i]))

            # Update position using sigmoid with clipping
            sigmoid = 1 / (1 + np.exp(-np.clip(swarm_vel[i], -10, 10)))
            swarm_pos[i] = (sigmoid > np.random.rand(n_features)).astype(int)

            # Ensure at least one feature is selected
            if np.sum(swarm_pos[i]) == 0:
                best_feature = np.argmax(target_corr)
                swarm_pos[i][best_feature] = 1

            # Update personal best
            fit = fast_fitness(swarm_pos[i])
            if fit > pbest_fit[i]:
                pbest_fit[i] = fit
                pbest_pos[i] = swarm_pos[i].copy()

                # Update global best
                if fit > pbest_fit[gbest_idx]:
                    gbest_idx = i
                    gbest_pos = swarm_pos[i].copy()

        # Track convergence after each iteration
        current_best_fitness = pbest_fit[gbest_idx]
        convergence_history.append(current_best_fitness)

        if iteration % 2 == 0:  # Progress update every 2 iterations
            print(f"     Iteration {iteration+1}/{T}, best fitness: {current_best_fitness:.4f}")

    # Calculate average convergence rate
    if len(convergence_history) > 1:
        convergence_improvements = []
        for i in range(1, len(convergence_history)):
            if convergence_history[i-1] != 0:  # Avoid division by zero
                improvement = (convergence_history[i] - convergence_history[i-1]) / convergence_history[i-1]
                convergence_improvements.append(max(0, improvement))  # Only positive improvements

        avg_convergence_rate = np.mean(convergence_improvements) if convergence_improvements else 0.0
    else:
        avg_convergence_rate = 0.0

    selected_features = np.where(gbest_pos == 1)[0]
    print(f"   PSO completed, selected {len(selected_features)} features")
    return selected_features, avg_convergence_rate

def comprehensive_model_test(X_data, y_data):
    """Comprehensive model evaluation with detailed metrics"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.25, random_state=42)

        # Use Ridge regression for evaluation
        model = Ridge(alpha=0.1)

        # Measure computational cost
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time

        # Test predictions
        y_pred = model.predict(X_test)

        # Calculate R² Score
        r2 = r2_score(y_test, y_pred)

        # Calculate Test Accuracy Percentage (for regression, we'll use 1 - normalized RMSE)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        y_range = np.max(y_test) - np.min(y_test)

        # Accuracy as percentage (100% - normalized error percentage)
        if y_range != 0:
            normalized_rmse = rmse / y_range
            accuracy_percentage = max(0, (1 - normalized_rmse) * 100)
        else:
            accuracy_percentage = 0.0

        return {
            'r2_score': round(r2, 4),
            'test_accuracy_percentage': round(accuracy_percentage, 2),
            'computational_cost': round(training_time, 4),
            'mse': round(mse, 4),
            'rmse': round(rmse, 4)
        }

    except Exception as e:
        print(f"     Model test failed: {e}")
        return {
            'r2_score': 0.0,
            'test_accuracy_percentage': 0.0,
            'computational_cost': 0.0,
            'mse': 0.0,
            'rmse': 0.0
        }

def process_dataset_fast(name, df, label_col):
    """Fast dataset processing with comprehensive metrics"""
    print(f"\n Fast PSO Analysis: {name}")
    print("=" * 50)

    total_start_time = time.time()

    try:
        print("   Preparing data...")

        # Quick data prep
        df_numeric = df.select_dtypes(include=[np.number])
        print(f"    Found {len(df_numeric.columns)} numeric columns")

        df_clean = df_numeric.dropna()
        print(f"    After cleaning: {len(df_clean)} rows")

        if label_col not in df_clean.columns:
            label_col = df_clean.columns[-1]
            print(f"     Using '{label_col}' as target")

        X = df_clean.drop(columns=[label_col])
        y = df_clean[label_col].values

        print(f"   Data shape: {X.shape[0]} samples, {X.shape[1]} features")

        if X.shape[0] < 10:
            print("     Not enough samples for analysis")
            return

        # Limit features for very large datasets
        if X.shape[1] > 50:
            print("  Too many features, selecting top 50 by correlation...")
            try:
                corr_with_target = np.abs(np.corrcoef(X.T, y)[:-1, -1])
                corr_with_target = np.nan_to_num(corr_with_target, nan=0.0)
                top_indices = np.argsort(corr_with_target)[-50:]
                X = X.iloc[:, top_indices]
                print(f"     Reduced to {X.shape[1]} features")
            except Exception as e:
                print(f"     Feature reduction failed: {e}")
                X = X.iloc[:, :50]  # Just take first 50

        # Scale features
        print("   Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Check for issues
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("    Scaling produced NaN/Inf values, using robust scaling")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        # Fast PSO selection with convergence tracking
        print("   Running fast PSO...")
        pso_start_time = time.time()
        selected_idx, avg_convergence_rate = fast_pso_selection(X_scaled, y, max_features=min(10, X.shape[1]//2))
        pso_time = time.time() - pso_start_time

        if len(selected_idx) == 0:
            print("   PSO failed, using correlation fallback...")
            corr_with_target = np.abs(np.corrcoef(X_scaled.T, y)[:-1, -1])
            corr_with_target = np.nan_to_num(corr_with_target, nan=0.0)
            selected_idx = np.argsort(corr_with_target)[-min(5, X.shape[1]):]
            avg_convergence_rate = 0.0

        # Show selected features
        print(f"\n   Selected {len(selected_idx)} features:")
        for i, idx in enumerate(selected_idx[:8]):  # Show max 8 features
            feature_name = X.columns[idx] if hasattr(X, 'columns') else f"Feature_{idx}"
            print(f"     {i+1}. {feature_name}")
        if len(selected_idx) > 8:
            print(f"     ... and {len(selected_idx)-8} more")

        # Comprehensive model evaluation
        print("   Testing model performance...")
        X_selected = X_scaled[:, selected_idx]
        model_results = comprehensive_model_test(X_selected, y)

        total_time = time.time() - total_start_time

        # Display results with requested metrics
        print(f"\n   PERFORMANCE METRICS:")
        print(f"     Test R2 Score: {model_results['r2_score']}")
        print(f"     Test Accuracy Percentage: {model_results['test_accuracy_percentage']}%")
        print(f"     Computational Cost (seconds): {model_results['computational_cost']}")
        print(f"     Average Convergence Rate: {avg_convergence_rate:.6f}")

        print(f"\n   Additional Metrics:")
        print(f"     Features Selected: {len(selected_idx)}/{X.shape[1]}")
        print(f"     PSO Time: {pso_time:.2f}s")
        print(f"     Total Processing Time: {total_time:.2f}s")
        print(f"     Mean Squared Error: {model_results['mse']}")
        print(f"     Root Mean Squared Error: {model_results['rmse']}")

        # Summary line with key metrics
        print(f"\n   SUMMARY: R²={model_results['r2_score']} | Accuracy={model_results['test_accuracy_percentage']}% | Cost={model_results['computational_cost']}s | Convergence={avg_convergence_rate:.6f}")

    except Exception as e:
        print(f"   Error: {str(e)}")
        import traceback
        print(f"   Details: {traceback.format_exc()}")

def generate_sample_data(name, n_samples=1000, n_features=20):
    """Generate sample data for testing"""
    print(f"   Generating sample data for {name}")
    np.random.seed(42)

    # Create correlated features
    X = np.random.randn(n_samples, n_features)

    # Make some features relevant to target
    relevant_features = np.random.choice(n_features, size=5, replace=False)
    weights = np.random.uniform(0.5, 2.0, size=5)

    y = np.sum(X[:, relevant_features] * weights, axis=1) + np.random.randn(n_samples) * 0.1

    # Create DataFrame
    feature_names = [f"feature_{i}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    return df

if _name_ == '_main_':
    print(" Enhanced PSO Feature Selection with Performance Metrics")
    print("=" * 60)

    # Test with generated data first
    print("\n  Testing with generated data...")
    df_sample = generate_sample_data("Sample Dataset", 500, 15)
    process_dataset_fast("Sample Dataset", df_sample, "target")

    # Try real datasets if available
    datasets_to_test = [
        ("California Housing", "/content/sample_data/california_housing_test.csv", "median_house_value"),
        ("Parkinson's UPDRS", "parkinsons_updrs.data", "total_UPDRS"),
        ("Boston Housing", "BostonHousing.csv", "medv"),
    ]

    for name, filepath, target_col in datasets_to_test:
        try:
            print(f"\n Trying to load {filepath}...")
            df = pd.read_csv(filepath)
            print(f"   Loaded successfully: {df.shape}")
            process_dataset_fast(name, df, target_col)
        except FileNotFoundError:
            print(f"   {filepath} not found, skipping...")
        except Exception as e:
            print(f"   Error with {name}: {str(e)}")

    print(f"\n Analysis complete with comprehensive metrics!")
