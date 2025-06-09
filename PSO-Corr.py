import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

def fast_pso_selection(X, y, max_features=10, S=8, T=5):
    """Ultra-fast PSO feature selection using correlation-based fitness"""
    print(f"  🔍 Starting PSO with {X.shape[1]} features...")
    
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
        print(f"    ⚠️ Correlation computation failed: {e}")
        # Fallback: use simple variance-based selection
        feature_var = np.var(X, axis=0)
        top_features = np.argsort(feature_var)[-min(max_features, n_features):]
        return top_features

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
        print("    ⚠️ All fitness values are 0, using correlation fallback")
        top_features = np.argsort(target_corr)[-min(max_features, n_features):]
        return top_features
    
    gbest_idx = np.argmax(pbest_fit)
    gbest_pos = pbest_pos[gbest_idx].copy()

    print(f"    🎯 Initial best fitness: {pbest_fit[gbest_idx]:.4f}")

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
        
        if iteration % 2 == 0:  # Progress update every 2 iterations
            print(f"    📊 Iteration {iteration+1}/{T}, best fitness: {pbest_fit[gbest_idx]:.4f}")

    selected_features = np.where(gbest_pos == 1)[0]
    print(f"  ✅ PSO completed, selected {len(selected_features)} features")
    return selected_features

def quick_model_test(X_data, y_data):
    """Fast model evaluation"""
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_data, y_data, test_size=0.25, random_state=42)

        # Use simpler model for speed
        model = Ridge(alpha=0.1)
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        return round(r2 * 100, 2), round(duration, 4)
    except Exception as e:
        print(f"    ⚠️ Model test failed: {e}")
        return 0.0, 0.0

def process_dataset_fast(name, df, label_col):
    """Fast dataset processing with better error handling"""
    print(f"\n🚀 Fast PSO Analysis: {name}")
    print("-" * 40)

    start_time = time.time()

    try:
        print("  📝 Preparing data...")
        
        # Quick data prep
        df_numeric = df.select_dtypes(include=[np.number])
        print(f"    Found {len(df_numeric.columns)} numeric columns")
        
        df_clean = df_numeric.dropna()
        print(f"    After cleaning: {len(df_clean)} rows")

        if label_col not in df_clean.columns:
            label_col = df_clean.columns[-1]
            print(f"    ⚠️ Using '{label_col}' as target")

        X = df_clean.drop(columns=[label_col])
        y = df_clean[label_col].values

        print(f"  📊 Data shape: {X.shape[0]} samples, {X.shape[1]} features")

        if X.shape[0] < 10:
            print("    ❌ Not enough samples for analysis")
            return

        # Limit features for very large datasets
        if X.shape[1] > 50:
            print("  🔥 Too many features, selecting top 50 by correlation...")
            try:
                corr_with_target = np.abs(np.corrcoef(X.T, y)[:-1, -1])
                corr_with_target = np.nan_to_num(corr_with_target, nan=0.0)
                top_indices = np.argsort(corr_with_target)[-50:]
                X = X.iloc[:, top_indices]
                print(f"    ✅ Reduced to {X.shape[1]} features")
            except Exception as e:
                print(f"    ⚠️ Feature reduction failed: {e}")
                X = X.iloc[:, :50]  # Just take first 50

        # Scale features
        print("  🔧 Scaling features...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Check for issues
        if np.any(np.isnan(X_scaled)) or np.any(np.isinf(X_scaled)):
            print("    ⚠️ Scaling produced NaN/Inf values, using robust scaling")
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=-1.0)

        # Fast PSO selection
        print("  ⚡ Running fast PSO...")
        selected_idx = fast_pso_selection(X_scaled, y, max_features=min(10, X.shape[1]//2))

        if len(selected_idx) == 0:
            print("  🔄 PSO failed, using correlation fallback...")
            corr_with_target = np.abs(np.corrcoef(X_scaled.T, y)[:-1, -1])
            corr_with_target = np.nan_to_num(corr_with_target, nan=0.0)
            selected_idx = np.argsort(corr_with_target)[-min(5, X.shape[1]):]

        # Show selected features
        print(f"\n  ✅ Selected {len(selected_idx)} features:")
        for i, idx in enumerate(selected_idx[:8]):  # Show max 8 features
            feature_name = X.columns[idx] if hasattr(X, 'columns') else f"Feature_{idx}"
            print(f"     {i+1}. {feature_name}")
        if len(selected_idx) > 8:
            print(f"     ... and {len(selected_idx)-8} more")

        # Quick model evaluation
        print("  🧪 Testing model performance...")
        X_selected = X_scaled[:, selected_idx]
        r2_score_val, train_time = quick_model_test(X_selected, y)

        total_time = time.time() - start_time

        print(f"\n  📈 Results:")
        print(f"     R² Score: {r2_score_val}%")
        print(f"     Features: {len(selected_idx)}/{X.shape[1]}")
        print(f"     Training Time: {train_time}s")
        print(f"     Total Time: {total_time:.2f}s")

    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        import traceback
        print(f"  🔍 Details: {traceback.format_exc()}")

def generate_sample_data(name, n_samples=1000, n_features=20):
    """Generate sample data for testing"""
    print(f"  🎲 Generating sample data for {name}")
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

if __name__ == '__main__':
    print("⚡  PSO  ALgo ")
    print("=" * 45)

    # Test with generated data first
    print("\n🧪 Testing with generated data...")
    df_sample = generate_sample_data("Sample Dataset", 500, 15)
    process_dataset_fast("Sample Dataset", df_sample, "target")

    # Try real datasets if available
    datasets_to_test = [
        ("California Housing", "/content/sample_data/california_housing_test.csv", "median_house_value"),
        ("Parkinson's UPDRS", "parkinsons_updrs.data", "total_UPDRS"),
        ("Boston Housing ","BostonHousing.csv","medv"),
    ]

    for name, filepath, target_col in datasets_to_test:
        try:
            print(f"\n📂 Trying to load {filepath}...")
            df = pd.read_csv(filepath)
            print(f"  ✅ Loaded successfully: {df.shape}")
            process_dataset_fast(name, df, target_col)
        except FileNotFoundError:
            print(f"  📂 {filepath} not found, skipping...")
        except Exception as e:
            print(f"  ❌ Error with {name}: {str(e)}")

    print(f"\n🎉 Analysis complete!")
