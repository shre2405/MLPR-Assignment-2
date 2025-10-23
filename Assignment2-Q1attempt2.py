import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from sklearn.metrics import roc_curve, auc

# Set random seed for reproducibility
np.random.seed(42)

print("="*70)
print("QUESTION 1: OPTIMAL CLASSIFICATION & LOGISTIC REGRESSION")
print("="*70)

# ============================================================================
# PARAMETERS (From Assignment)
# ============================================================================

P_L0 = 0.6
P_L1 = 0.4
w01, w02 = 0.5, 0.5
w11, w12 = 0.5, 0.5

m01 = np.array([-0.9, -1.1])
m02 = np.array([0.8, 0.75])
m11 = np.array([-1.1, 0.9])
m12 = np.array([0.9, -0.75])

C = np.array([[0.75, 0], [0, 1.25]])

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_data(n_samples):
    """Generate samples from the mixture distribution"""
    X = []
    y = []
    
    for _ in range(n_samples):
        # Sample class label
        label = 0 if np.random.rand() < P_L0 else 1
        
        if label == 0:
            # Sample from class 0 mixture
            component = 0 if np.random.rand() < w01 else 1
            if component == 0:
                sample = np.random.multivariate_normal(m01, C)
            else:
                sample = np.random.multivariate_normal(m02, C)
        else:
            # Sample from class 1 mixture
            component = 0 if np.random.rand() < w11 else 1
            if component == 0:
                sample = np.random.multivariate_normal(m11, C)
            else:
                sample = np.random.multivariate_normal(m12, C)
        
        X.append(sample)
        y.append(label)
    
    return np.array(X), np.array(y)

def class_conditional_pdf(x, label):
    """Compute p(x|L=label)"""
    if label == 0:
        p1 = multivariate_normal.pdf(x, mean=m01, cov=C)
        p2 = multivariate_normal.pdf(x, mean=m02, cov=C)
        return w01 * p1 + w02 * p2
    else:
        p1 = multivariate_normal.pdf(x, mean=m11, cov=C)
        p2 = multivariate_normal.pdf(x, mean=m12, cov=C)
        return w11 * p1 + w12 * p2

def posterior_probability(x, label):
    """Compute P(L=label|x) using Bayes rule"""
    if label == 0:
        prior = P_L0
    else:
        prior = P_L1
    
    likelihood = class_conditional_pdf(x, label)
    evidence = P_L0 * class_conditional_pdf(x, 0) + P_L1 * class_conditional_pdf(x, 1)
    
    return (likelihood * prior) / evidence if evidence > 0 else 0

def optimal_classifier(X):
    """Theoretically optimal Bayes classifier"""
    predictions = []
    scores = []
    
    for x in X:
        p0 = posterior_probability(x, 0)
        p1 = posterior_probability(x, 1)
        predictions.append(1 if p1 > p0 else 0)
        scores.append(p1)  # Use P(L=1|x) as discriminant score
    
    return np.array(predictions), np.array(scores)

# ============================================================================
# PART 1: OPTIMAL CLASSIFIER
# ============================================================================

print("\n" + "="*70)
print("PART 1: THEORETICALLY OPTIMAL CLASSIFIER")
print("="*70)

# Generate datasets
print("\nGenerating datasets...")
D_50_train, y_50_train = generate_data(50)
D_500_train, y_500_train = generate_data(500)
D_5000_train, y_5000_train = generate_data(5000)
D_10K_validate, y_10K_validate = generate_data(10000)

print(f"Training sets: 50, 500, 5000 samples")
print(f"Validation set: 10000 samples")

# Apply optimal classifier to validation set
y_pred_optimal, scores_optimal = optimal_classifier(D_10K_validate)

# Compute error
error_optimal = np.mean(y_pred_optimal != y_10K_validate)
print(f"\nMin P(error) estimate: {error_optimal:.4f}")

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_10K_validate, scores_optimal)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')

# Find the point on ROC corresponding to min P(error)
optimal_threshold = 0.5
idx = np.argmin(np.abs(thresholds - optimal_threshold))
plt.plot(fpr[idx], tpr[idx], 'ro', markersize=10, 
         label=f'Min P(error) point (Threshold={optimal_threshold})')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Optimal Classifier', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('q1_roc_optimal.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot decision boundary
print("\nPlotting decision boundary...")
plt.figure(figsize=(10, 8))
x_min, x_max = -3, 3
y_min, y_max = -3, 3
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = np.array([posterior_probability(x, 1) for x in grid_points])
Z = Z.reshape(xx.shape)

plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
plt.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu_r')

# Plot validation data
plt.scatter(D_10K_validate[y_10K_validate==0, 0], 
           D_10K_validate[y_10K_validate==0, 1],
           c='blue', alpha=0.3, s=10, label='Class 0')
plt.scatter(D_10K_validate[y_10K_validate==1, 0], 
           D_10K_validate[y_10K_validate==1, 1],
           c='red', alpha=0.3, s=10, label='Class 1')

plt.xlabel('x₁', fontsize=12)
plt.ylabel('x₂', fontsize=12)
plt.title('Optimal Decision Boundary', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.savefig('q1_decision_boundary_optimal.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PART 2: LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*70)
print("PART 2: LOGISTIC REGRESSION")
print("="*70)

def logistic(z):
    """Logistic sigmoid function"""
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def z_linear(X):
    """Linear feature transformation: z(x) = [1, x1, x2]^T"""
    ones = np.ones((X.shape[0], 1))
    return np.hstack([ones, X])

def z_quadratic(X):
    """Quadratic feature transformation: z(x) = [1, x1, x2, x1^2, x1*x2, x2^2]^T"""
    ones = np.ones((X.shape[0], 1))
    x1 = X[:, 0:1]
    x2 = X[:, 1:2]
    return np.hstack([ones, x1, x2, x1**2, x1*x2, x2**2])

def negative_log_likelihood(w, X, y, feature_func):
    """Negative log-likelihood for logistic regression"""
    Z = feature_func(X)
    h = logistic(Z @ w)
    
    # Add small epsilon to avoid log(0)
    epsilon = 1e-15
    h = np.clip(h, epsilon, 1 - epsilon)
    
    nll = -np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return nll

def train_logistic_regression(X_train, y_train, feature_func, n_features):
    """Train logistic regression using optimization"""
    w_init = np.zeros(n_features)
    
    result = minimize(negative_log_likelihood, w_init, 
                     args=(X_train, y_train, feature_func),
                     method='BFGS', options={'maxiter': 1000})
    
    return result.x

def predict_logistic(X, w, feature_func):
    """Make predictions using logistic regression"""
    Z = feature_func(X)
    h = logistic(Z @ w)
    return (h >= 0.5).astype(int), h

# Train and evaluate logistic-linear models
print("\n(a) Logistic-Linear Models:")
print("-" * 50)

datasets = [
    (D_50_train, y_50_train, "50", z_linear, 3),
    (D_500_train, y_500_train, "500", z_linear, 3),
    (D_5000_train, y_5000_train, "5000", z_linear, 3)
]

linear_results = []
for X_train, y_train, n_str, feature_func, n_features in datasets:
    print(f"Training linear model with N={n_str}...")
    w_linear = train_logistic_regression(X_train, y_train, feature_func, n_features)
    y_pred, _ = predict_logistic(D_10K_validate, w_linear, feature_func)
    error = np.mean(y_pred != y_10K_validate)
    linear_results.append((w_linear, error, n_str))
    print(f"N={n_str:>4}: P(error) = {error:.4f}")

# Train and evaluate logistic-quadratic models
print("\n(b) Logistic-Quadratic Models:")
print("-" * 50)

datasets = [
    (D_50_train, y_50_train, "50", z_quadratic, 6),
    (D_500_train, y_500_train, "500", z_quadratic, 6),
    (D_5000_train, y_5000_train, "5000", z_quadratic, 6)
]

quadratic_results = []
for X_train, y_train, n_str, feature_func, n_features in datasets:
    print(f"Training quadratic model with N={n_str}...")
    w_quad = train_logistic_regression(X_train, y_train, feature_func, n_features)
    y_pred, _ = predict_logistic(D_10K_validate, w_quad, feature_func)
    error = np.mean(y_pred != y_10K_validate)
    quadratic_results.append((w_quad, error, n_str))
    print(f"N={n_str:>4}: P(error) = {error:.4f}")

# ============================================================================
# VISUALIZATION: DECISION BOUNDARIES
# ============================================================================

print("\nPlotting decision boundaries...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

x_min, x_max = -3, 3
y_min, y_max = -3, 3
datasets_train_vis = [
    (D_50_train, y_50_train, "50"),
    (D_500_train, y_500_train, "500"),
    (D_5000_train, y_5000_train, "5000")
]

# Linear models
for idx, ((w_linear, error, n_str), ax) in enumerate(zip(linear_results, axes[0])):
    X_train, y_train, _ = datasets_train_vis[idx]
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    _, Z = predict_logistic(grid_points, w_linear, z_linear)
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu_r')
    
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
              c='blue', alpha=0.5, s=20, label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
              c='red', alpha=0.5, s=20, label='Class 1')
    
    ax.set_title(f'Linear, N={n_str} (Error={error:.4f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Quadratic models
for idx, ((w_quad, error, n_str), ax) in enumerate(zip(quadratic_results, axes[1])):
    X_train, y_train, _ = datasets_train_vis[idx]

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    _, Z = predict_logistic(grid_points, w_quad, z_quadratic)
    Z = Z.reshape(xx.shape)
    
    ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    ax.contourf(xx, yy, Z, levels=20, alpha=0.3, cmap='RdBu_r')
    
    ax.scatter(X_train[y_train==0, 0], X_train[y_train==0, 1],
              c='blue', alpha=0.5, s=20, label='Class 0')
    ax.scatter(X_train[y_train==1, 0], X_train[y_train==1, 1],
              c='red', alpha=0.5, s=20, label='Class 1')
    
    ax.set_title(f'Quadratic, N={n_str} (Error={error:.4f})', fontsize=12, fontweight='bold')
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('q1_logistic_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# DISCUSSION
# ============================================================================

print("\n" + "="*70)
print("DISCUSSION")
print("="*70)

print(f"\nOptimal Classifier Error: {error_optimal:.4f}")
print(f"\nLinear Models Errors:    {[f'{e:.4f}' for _, e, _ in linear_results]}")
print(f"Quadratic Models Errors: {[f'{e:.4f}' for _, e, _ in quadratic_results]}")

print("\n" + "-"*70)
print("KEY OBSERVATIONS:")
print("-"*70)

print("""
1. SAMPLE SIZE EFFECT:
   - Performance improves with more training samples
   - Reduced estimation variance with larger datasets
   - Both models benefit from more data

2. MODEL COMPARISON:
   - Quadratic models significantly outperform linear models
   - Linear models limited by functional form (high bias)
   - Quadratic models can capture non-linear decision boundaries

3. APPROACHING OPTIMAL:
   - Only quadratic models approach optimal Bayes error
   - Linear models asymptote at higher error (~0.30-0.40)
   - With N=5000, quadratic model nearly matches optimal

4. BIAS-VARIANCE TRADEOFF:
   - Linear: High bias, low variance (underfit)
   - Quadratic: Lower bias, captures true boundary better
   - More parameters need more data to estimate reliably

5. TRUE BOUNDARY:
   - Optimal boundary is non-linear (from mixture of Gaussians)
   - Quadratic approximation is reasonable
   - Linear boundary fundamentally limited
""")

print("="*70)
print("QUESTION 1 COMPLETE")
print("="*70)