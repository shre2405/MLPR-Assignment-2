import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ============================================================================
# DATA GENERATION FUNCTIONS (from provided hw2q2.py)
# ============================================================================

def hw2q2():
    """Generate training and validation datasets as specified"""
    Ntrain = 100
    data = generateData(Ntrain)
    plot3(data[0,:], data[1,:], data[2,:], title='Training Dataset')
    xTrain = data[0:2,:]
    yTrain = data[2,:]
    
    Nvalidate = 1000
    data = generateData(Nvalidate)
    plot3(data[0,:], data[1,:], data[2,:], title='Validation Dataset')
    xValidate = data[0:2,:]
    yValidate = data[2,:]
    
    return xTrain, yTrain, xValidate, yValidate

def generateData(N):
    """Generate data from Gaussian Mixture Model"""
    gmmParameters = {}
    gmmParameters['priors'] = [.3, .4, .3]
    gmmParameters['meanVectors'] = np.array([[-10, 0, 10], [0, 0, 0], [10, 0, -10]])
    gmmParameters['covMatrices'] = np.zeros((3, 3, 3))
    gmmParameters['covMatrices'][:,:,0] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    gmmParameters['covMatrices'][:,:,1] = np.array([[8, 0, 0], [0, .5, 0], [0, 0, .5]])
    gmmParameters['covMatrices'][:,:,2] = np.array([[1, 0, -3], [0, 1, 0], [-3, 0, 15]])
    x, labels = generateDataFromGMM(N, gmmParameters)
    return x

def generateDataFromGMM(N, gmmParameters):
    """
    Generates N vector samples from the specified mixture of Gaussians
    Returns samples and their component labels
    """
    priors = gmmParameters['priors']
    meanVectors = gmmParameters['meanVectors']
    covMatrices = gmmParameters['covMatrices']
    n = meanVectors.shape[0]  # Data dimensionality
    C = len(priors)  # Number of components
    x = np.zeros((n, N))
    labels = np.zeros((1, N))
    
    # Decide randomly which samples will come from each component
    u = np.random.random((1, N))
    thresholds = np.zeros((1, C+1))
    thresholds[:,0:C] = np.cumsum(priors)
    thresholds[:,C] = 1
    
    for l in range(C):
        indl = np.where(u <= float(thresholds[:,l]))
        Nl = len(indl[1])
        labels[indl] = (l+1)*1
        u[indl] = 1.1
        x[:,indl[1]] = np.transpose(np.random.multivariate_normal(
            meanVectors[:,l], covMatrices[:,:,l], Nl))
    
    return x, labels

def plot3(a, b, c, mark="o", col="b", title='Dataset'):
    """Plot 3D scatter"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a, b, c, marker=mark, color=col, alpha=0.6, s=40)
    ax.set_xlabel("x₁", fontsize=12)
    ax.set_ylabel("x₂", fontsize=12)
    ax.set_zlabel("y", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ============================================================================
# CUBIC POLYNOMIAL MODEL - ML AND MAP ESTIMATORS
# ============================================================================

def create_cubic_features(X):
    """
    Create cubic polynomial feature matrix for 2D input
    
    For x = [x₁, x₂]ᵀ, cubic polynomial includes:
    φ(x) = [1, x₁, x₂, x₁², x₁x₂, x₂², x₁³, x₁²x₂, x₁x₂², x₂³]ᵀ
    
    Args:
        X: (2, N) array where each column is [x₁, x₂]ᵀ
    Returns:
        Phi: (N, 10) design matrix where each row is φ(xᵢ)ᵀ
    """
    x1 = X[0, :]
    x2 = X[1, :]
    
    Phi = np.column_stack([
        np.ones_like(x1),      # 1
        x1,                     # x₁
        x2,                     # x₂
        x1**2,                  # x₁²
        x1*x2,                  # x₁x₂
        x2**2,                  # x₂²
        x1**3,                  # x₁³
        (x1**2)*x2,            # x₁²x₂
        x1*(x2**2),            # x₁x₂²
        x2**3                   # x₂³
    ])
    
    return Phi

def ml_estimator(X, y):
    """
    Maximum Likelihood (ML) Estimator
    
    Derivation:
    Given: y = wᵀφ(x) + v, where v ~ N(0, σ²)
    Likelihood: p(y|x,w) = N(y | wᵀφ(x), σ²)
    Log-likelihood: ℓ(w) = -1/(2σ²) Σᵢ(yᵢ - wᵀφ(xᵢ))² + const
    
    Maximize ℓ(w) ⟺ Minimize Σᵢ(yᵢ - wᵀφ(xᵢ))²
    
    Solution (Normal Equations):
    ∇ℓ(w) = 0 ⟹ ΦᵀΦw = Φᵀy
    
    w_ML = (ΦᵀΦ)⁻¹Φᵀy
    
    Args:
        X: (2, N) input features
        y: (N,) output values
    Returns:
        w_ML: (10,) ML parameter estimates
    """
    Phi = create_cubic_features(X)
    
    # Solve normal equations: (ΦᵀΦ)w = Φᵀy
    w_ML = np.linalg.solve(Phi.T @ Phi, Phi.T @ y)
    
    return w_ML

def map_estimator(X, y, gamma, sigma_sq=1.0):
    """
    Maximum A Posteriori (MAP) Estimator
    
    Derivation:
    Prior: p(w) = N(0, γI)
    Posterior: p(w|D) ∝ p(D|w)p(w)
    
    Log-posterior:
    log p(w|D) = -1/(2σ²) Σᵢ(yᵢ - wᵀφ(xᵢ))² - 1/(2γ)||w||² + const
    
    Maximize log p(w|D) ⟺ Minimize [Σᵢ(yᵢ - wᵀφ(xᵢ))² + (σ²/γ)||w||²]
    
    Solution:
    ∇[log p(w|D)] = 0 ⟹ (ΦᵀΦ + λI)w = Φᵀy, where λ = σ²/γ
    
    w_MAP = (ΦᵀΦ + (σ²/γ)I)⁻¹Φᵀy
    
    This is equivalent to Ridge Regression with regularization λ = σ²/γ
    
    Args:
        X: (2, N) input features
        y: (N,) output values
        gamma: prior variance hyperparameter (controls regularization strength)
        sigma_sq: noise variance (assumed known, default=1.0)
    Returns:
        w_MAP: (10,) MAP parameter estimates
    """
    Phi = create_cubic_features(X)
    n_features = Phi.shape[1]
    
    # Regularization parameter: λ = σ²/γ
    lambda_reg = sigma_sq / gamma
    
    # Solve regularized normal equations: (ΦᵀΦ + λI)w = Φᵀy
    w_MAP = np.linalg.solve(Phi.T @ Phi + lambda_reg * np.eye(n_features), 
                            Phi.T @ y)
    
    return w_MAP

def predict(X, w):
    """
    Predict y values using learned parameters
    
    Args:
        X: (2, N) input features
        w: (10,) parameter vector
    Returns:
        y_pred: (N,) predicted values
    """
    Phi = create_cubic_features(X)
    return Phi @ w

def mean_squared_error(y_true, y_pred):
    """Calculate Mean Squared Error (MSE)"""
    return np.mean((y_true - y_pred)**2)

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_predictions_comparison(X, y_true, y_pred_ML, y_pred_MAP, title_suffix=""):
    """Plot predictions vs actual in 3D space"""
    fig = plt.figure(figsize=(16, 6))
    
    # Plot 1: ML Predictions
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(X[0, :], X[1, :], y_pred_ML, c='blue', marker='o', 
               s=30, alpha=0.6, label='ML Predictions')
    ax1.scatter(X[0, :], X[1, :], y_true, c='red', marker='x', 
               s=50, alpha=0.8, label='Actual')
    ax1.set_xlabel('x₁', fontsize=11)
    ax1.set_ylabel('x₂', fontsize=11)
    ax1.set_zlabel('y', fontsize=11)
    ax1.set_title(f'ML Predictions {title_suffix}', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: MAP Predictions
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(X[0, :], X[1, :], y_pred_MAP, c='green', marker='o', 
               s=30, alpha=0.6, label='MAP Predictions')
    ax2.scatter(X[0, :], X[1, :], y_true, c='red', marker='x', 
               s=50, alpha=0.8, label='Actual')
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('x₂', fontsize=11)
    ax2.set_zlabel('y', fontsize=11)
    ax2.set_title(f'MAP Predictions {title_suffix}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Comparison
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(X[0, :], X[1, :], y_true, c='red', marker='x', 
               s=50, alpha=0.8, label='Actual', zorder=3)
    ax3.scatter(X[0, :], X[1, :], y_pred_ML, c='blue', marker='o', 
               s=20, alpha=0.4, label='ML', zorder=1)
    ax3.scatter(X[0, :], X[1, :], y_pred_MAP, c='green', marker='^', 
               s=20, alpha=0.4, label='MAP', zorder=2)
    ax3.set_xlabel('x₁', fontsize=11)
    ax3.set_ylabel('x₂', fontsize=11)
    ax3.set_zlabel('y', fontsize=11)
    ax3.set_title(f'Comparison {title_suffix}', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_analysis_results(gamma_values, mse_train_MAP, mse_val_MAP, mse_val_ML, 
                         optimal_gamma, w_MAP_all, w_ML):
    """Create comprehensive analysis plots"""
    fig = plt.figure(figsize=(18, 5))
    
    # Plot 1: MSE vs Gamma
    ax1 = fig.add_subplot(131)
    ax1.semilogx(gamma_values, mse_train_MAP, 'b-', linewidth=2.5, label='Training MSE')
    ax1.semilogx(gamma_values, mse_val_MAP, 'r-', linewidth=2.5, label='Validation MSE')
    ax1.axhline(y=mse_val_ML, color='g', linestyle='--', linewidth=2, label='ML Validation MSE')
    ax1.axvline(x=optimal_gamma, color='k', linestyle=':', linewidth=2.5, label=f'Optimal γ={optimal_gamma:.2e}')
    ax1.scatter([optimal_gamma], [mse_val_MAP[np.argmin(mse_val_MAP)]], 
               color='red', s=200, zorder=10, marker='*', edgecolors='black', linewidth=2)
    ax1.set_xlabel('γ (Prior Variance)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Squared Error', fontsize=13, fontweight='bold')
    ax1.set_title('MAP Performance vs Hyperparameter γ', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(labelsize=11)
    
    # Plot 2: Parameter Evolution
    ax2 = fig.add_subplot(132)
    feature_names = ['w₀', 'w₁', 'w₂', 'w₃', 'w₄', 'w₅', 'w₆', 'w₇', 'w₈', 'w₉']
    for i in range(10):
        ax2.semilogx(gamma_values, w_MAP_all[:, i], label=feature_names[i], 
                    alpha=0.8, linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='-', linewidth=1)
    ax2.axvline(x=optimal_gamma, color='k', linestyle=':', linewidth=2.5)
    ax2.set_xlabel('γ (Prior Variance)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Parameter Value', fontsize=13, fontweight='bold')
    ax2.set_title('MAP Parameter Evolution vs γ', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, ncol=2, loc='best')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.tick_params(labelsize=11)
    
    # Plot 3: Parameter Shrinkage (L2 Norm)
    ax3 = fig.add_subplot(133)
    param_norms = np.linalg.norm(w_MAP_all, axis=1)
    ax3.semilogx(gamma_values, param_norms, 'b-', linewidth=3, label='||w_MAP||₂')
    ax3.axhline(y=np.linalg.norm(w_ML), color='r', linestyle='--', linewidth=3, label='||w_ML||₂')
    ax3.axvline(x=optimal_gamma, color='k', linestyle=':', linewidth=2.5)
    ax3.fill_between(gamma_values, 0, param_norms, alpha=0.2)
    ax3.set_xlabel('γ (Prior Variance)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('L₂ Norm of Parameters', fontsize=13, fontweight='bold')
    ax3.set_title('Parameter Shrinkage Effect', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=12)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.tick_params(labelsize=11)
    
    plt.tight_layout()
    plt.show()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducibility (optional - comment out for different results)
    np.random.seed(42)
    
    print("="*80)
    print("EECE5644 - ASSIGNMENT 2 - QUESTION 2")
    print("ML and MAP Estimation for Cubic Polynomial Regression")
    print("="*80)
    
    # ========================================================================
    # STEP 1: Generate Data
    # ========================================================================
    print("\n[STEP 1] Generating datasets using provided hw2q2() function...")
    xTrain, yTrain, xValidate, yValidate = hw2q2()
    
    print(f"\nDataset sizes:")
    print(f"  Training samples:   {xTrain.shape[1]}")
    print(f"  Validation samples: {xValidate.shape[1]}")
    print(f"  Input dimensions:   {xTrain.shape[0]}")
    print(f"  Feature dimensions: 10 (cubic polynomial)")
    
    # ========================================================================
    # STEP 2: Maximum Likelihood (ML) Estimation
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 2] MAXIMUM LIKELIHOOD (ML) ESTIMATION")
    print("="*80)
    print("\nTheory: w_ML = (ΦᵀΦ)⁻¹Φᵀy")
    print("Minimizes: Σᵢ(yᵢ - wᵀφ(xᵢ))²")
    
    w_ML = ml_estimator(xTrain, yTrain)
    
    print(f"\nML Parameter Estimates:")
    feature_names = ['1', 'x₁', 'x₂', 'x₁²', 'x₁x₂', 'x₂²', 'x₁³', 'x₁²x₂', 'x₁x₂²', 'x₂³']
    for i, (name, w) in enumerate(zip(feature_names, w_ML)):
        print(f"  w[{i}] ({name:6s}): {w:12.6f}")
    
    # Evaluate ML estimator
    y_pred_train_ML = predict(xTrain, w_ML)
    y_pred_val_ML = predict(xValidate, w_ML)
    
    mse_train_ML = mean_squared_error(yTrain, y_pred_train_ML)
    mse_val_ML = mean_squared_error(yValidate, y_pred_val_ML)
    
    print(f"\nML Performance:")
    print(f"  Training MSE:   {mse_train_ML:.8f}")
    print(f"  Validation MSE: {mse_val_ML:.8f}")
    print(f"  Parameter norm: ||w_ML||₂ = {np.linalg.norm(w_ML):.6f}")
    
    if mse_val_ML > mse_train_ML:
        print(f"  ⚠ Overfitting detected (Val MSE > Train MSE by {((mse_val_ML/mse_train_ML - 1)*100):.2f}%)")
    
    # ========================================================================
    # STEP 3: Maximum A Posteriori (MAP) Estimation
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 3] MAXIMUM A POSTERIORI (MAP) ESTIMATION")
    print("="*80)
    print("\nTheory: w_MAP = (ΦᵀΦ + (σ²/γ)I)⁻¹Φᵀy")
    print("Minimizes: Σᵢ(yᵢ - wᵀφ(xᵢ))² + (σ²/γ)||w||²")
    print("Equivalent to Ridge Regression with λ = σ²/γ")
    
    # Test various gamma values
    m, n = 4, 4
    gamma_values = np.logspace(-m, n, 100)
    
    print(f"\nTesting {len(gamma_values)} γ values from 10⁻⁴ to 10⁴...")
    
    mse_train_MAP = []
    mse_val_MAP = []
    w_MAP_all = []
    
    for gamma in gamma_values:
        w_MAP = map_estimator(xTrain, yTrain, gamma, sigma_sq=1.0)
        w_MAP_all.append(w_MAP)
        
        y_pred_train = predict(xTrain, w_MAP)
        y_pred_val = predict(xValidate, w_MAP)
        
        mse_train_MAP.append(mean_squared_error(yTrain, y_pred_train))
        mse_val_MAP.append(mean_squared_error(yValidate, y_pred_val))
    
    w_MAP_all = np.array(w_MAP_all)
    
    # Find optimal gamma
    optimal_idx = np.argmin(mse_val_MAP)
    optimal_gamma = gamma_values[optimal_idx]
    optimal_mse_train = mse_train_MAP[optimal_idx]
    optimal_mse_val = mse_val_MAP[optimal_idx]
    w_MAP_optimal = w_MAP_all[optimal_idx]
    
    print(f"\nOptimal Hyperparameter: γ = {optimal_gamma:.6e}")
    print(f"Regularization strength: λ = σ²/γ = {1.0/optimal_gamma:.6e}")
    
    print(f"\nOptimal MAP Parameter Estimates:")
    for i, (name, w) in enumerate(zip(feature_names, w_MAP_optimal)):
        print(f"  w[{i}] ({name:6s}): {w:12.6f}")
    
    print(f"\nMAP Performance (at optimal γ):")
    print(f"  Training MSE:   {optimal_mse_train:.8f}")
    print(f"  Validation MSE: {optimal_mse_val:.8f}")
    print(f"  Parameter norm: ||w_MAP||₂ = {np.linalg.norm(w_MAP_optimal):.6f}")
    
    improvement = ((mse_val_ML - optimal_mse_val) / mse_val_ML) * 100
    print(f"\nImprovement over ML: {improvement:.2f}%")
    
    # ========================================================================
    # STEP 4: Analysis and Visualization
    # ========================================================================
    print("\n" + "="*80)
    print("[STEP 4] ANALYSIS AND VISUALIZATION")
    print("="*80)
    
    # Create visualizations
    print("\nGenerating plots...")
    
    # Plot 1: Predictions comparison
    y_pred_val_MAP = predict(xValidate, w_MAP_optimal)
    plot_predictions_comparison(xValidate, yValidate, y_pred_val_ML, 
                               y_pred_val_MAP, "(Validation Set)")
    
    # Plot 2: Comprehensive analysis
    plot_analysis_results(gamma_values, mse_train_MAP, mse_val_MAP, mse_val_ML,
                         optimal_gamma, w_MAP_all, w_ML)
    
    # ========================================================================
    # STEP 5: Summary and Interpretation
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY AND INTERPRETATION")
    print("="*80)
    
    print(f"\n{'Estimator':<20} {'Train MSE':<15} {'Val MSE':<15} {'||w||₂':<15}")
    print("-"*65)
    print(f"{'ML':<20} {mse_train_ML:<15.6f} {mse_val_ML:<15.6f} {np.linalg.norm(w_ML):<15.6f}")
    print(f"{'MAP (optimal γ)':<20} {optimal_mse_train:<15.6f} {optimal_mse_val:<15.6f} {np.linalg.norm(w_MAP_optimal):<15.6f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"""
1. RELATIONSHIP BETWEEN ML AND MAP:
   • As γ → ∞: MAP → ML (prior variance increases, regularization weakens)
   • As γ → 0:  MAP → 0 (prior variance decreases, regularization strengthens)
   • Optimal γ = {optimal_gamma:.2e} provides best bias-variance tradeoff

2. REGULARIZATION EFFECT:
   • MAP introduces L₂ penalty with strength λ = σ²/γ = {1.0/optimal_gamma:.2e}
   • This shrinks parameters: ||w_ML|| = {np.linalg.norm(w_ML):.4f} → ||w_MAP|| = {np.linalg.norm(w_MAP_optimal):.4f}
   • Parameter shrinkage reduces model complexity and prevents overfitting

3. PERFORMANCE COMPARISON:
   • ML Validation MSE:  {mse_val_ML:.6f}
   • MAP Validation MSE: {optimal_mse_val:.6f}
   • Improvement: {improvement:.2f}%
   • {'✓ MAP successfully reduces overfitting!' if improvement > 0 else '⚠ ML performs similarly or better'}

4. BIAS-VARIANCE TRADEOFF:
   • Small γ (strong reg): High bias, Low variance → May underfit
   • Large γ (weak reg):   Low bias, High variance → May overfit
   • Optimal γ selected via validation set minimizes generalization error

5. PRACTICAL IMPLICATIONS:
   • With only {xTrain.shape[1]} training samples and 10 parameters, regularization is crucial
   • The cubic polynomial model has sufficient capacity to overfit noise
   • MAP estimation with proper γ selection improves generalization
   • Cross-validation is essential for hyperparameter selection
    """)
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)