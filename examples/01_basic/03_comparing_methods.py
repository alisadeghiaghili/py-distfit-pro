"""
Comparing Fitting Methods: MLE vs Method of Moments
===================================================

What you'll learn:
- Difference between MLE and Method of Moments (MOM)
- When to use each method
- Speed vs accuracy tradeoff
- Convergence issues and solutions

Real-world context:
You have large datasets and need fast fitting, or small/difficult
datasets where MLE fails to converge.
"""

import numpy as np
import time
from distfit_pro import get_distribution

# ============================================================================
# Background: MLE vs Method of Moments
# ============================================================================
print("📚 Background")
print("=" * 70)
print("""
Maximum Likelihood Estimation (MLE):
  ✅ Most accurate (optimal under regularity conditions)
  ✅ Best for inference (confidence intervals, hypothesis tests)
  ❌ Slower (requires numerical optimization)
  ❌ Can fail to converge on difficult data
  
 Method of Moments (MOM):
  ✅ Very fast (closed-form formulas)
  ✅ Always converges
  ✅ Good initial guess for MLE
  ❌ Less efficient (larger standard errors)
  ❌ Not optimal for small samples
""")
print()

# ============================================================================
# EXPERIMENT 1: Speed Comparison
# ============================================================================
print("⏱️  Experiment 1: Speed Comparison")
print("=" * 70)

# Generate large dataset
np.random.seed(42)
large_data = np.random.normal(100, 15, size=10000)

print(f"Dataset: {len(large_data)} samples from Normal(100, 15)\n")

# Time MLE
dist_mle = get_distribution('normal')
start = time.time()
dist_mle.fit(large_data, method='mle')
time_mle = time.time() - start

# Time MOM  
dist_mom = get_distribution('normal')
start = time.time()
dist_mom.fit(large_data, method='mom')
time_mom = time.time() - start

print(f"MLE time: {time_mle*1000:.2f} ms")
print(f"MOM time: {time_mom*1000:.2f} ms")
print(f"Speedup: {time_mle/time_mom:.1f}x faster with MOM\n")

# Compare estimates
print("Parameter Estimates:")
print("-" * 70)
params_mle = dist_mle.params
params_mom = dist_mom.params

for param in params_mle:
    diff = abs(params_mle[param] - params_mom[param])
    pct_diff = 100 * diff / params_mle[param]
    print(f"{param:10s}: MLE={params_mle[param]:8.4f}  "
          f"MOM={params_mom[param]:8.4f}  "
          f"Diff={pct_diff:.3f}%")

print("\n✅ Both methods agree closely on large samples\n")

# ============================================================================
# EXPERIMENT 2: Small Sample Behavior
# ============================================================================
print("🔬 Experiment 2: Small Sample Performance")
print("=" * 70)

# Generate small dataset
np.random.seed(123)
small_data = np.random.exponential(scale=50, size=20)

print(f"Dataset: {len(small_data)} samples from Exponential(scale=50)")
print(f"True parameter: scale=50\n")

# Fit with both methods
dist_exp_mle = get_distribution('exponential')
dist_exp_mom = get_distribution('exponential')

dist_exp_mle.fit(small_data, method='mle')
dist_exp_mom.fit(small_data, method='mom')

scale_mle = dist_exp_mle.params['scale']
scale_mom = dist_exp_mom.params['scale']

print(f"MLE estimate: scale={scale_mle:.3f}  (error: {abs(scale_mle-50):.3f})")
print(f"MOM estimate: scale={scale_mom:.3f}  (error: {abs(scale_mom-50):.3f})")

print("\n📊 MLE is generally more accurate on small samples\n")

# ============================================================================
# EXPERIMENT 3: Difficult Data (when MLE struggles)
# ============================================================================
print("⚠️  Experiment 3: Difficult Data")
print("=" * 70)

# Generate data near distribution bounds
np.random.seed(456)
difficult_data = np.random.beta(0.5, 0.5, size=100)  # U-shaped, hard to fit

print("Dataset: 100 samples from Beta(0.5, 0.5) - U-shaped distribution")
print("This is challenging for optimization\n")

# Try MLE (might have issues)
print("Fitting with MLE...")
try:
    dist_beta_mle = get_distribution('beta')
    dist_beta_mle.fit(difficult_data, method='mle')
    print("✅ MLE converged")
    print(f"   alpha={dist_beta_mle.params['alpha']:.3f}, "
          f"beta={dist_beta_mle.params['beta']:.3f}")
except Exception as e:
    print(f"❌ MLE failed: {str(e)}")

print()

# Try MOM (more robust)
print("Fitting with MOM...")
try:
    dist_beta_mom = get_distribution('beta')
    dist_beta_mom.fit(difficult_data, method='mom')
    print("✅ MOM converged (always does!)")
    print(f"   alpha={dist_beta_mom.params['alpha']:.3f}, "
          f"beta={dist_beta_mom.params['beta']:.3f}")
except Exception as e:
    print(f"❌ MOM failed: {str(e)}")

print("\n💡 Tip: Use MOM first, then refine with MLE if needed\n")

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("🎯 Recommendations")
print("=" * 70)
print("""
Use MLE when:
✅ You need maximum accuracy
✅ Sample size is moderate to large (n > 100)
✅ You need confidence intervals or hypothesis tests
✅ Data is well-behaved

Use MOM when:
✅ Speed is critical (real-time, large batches)
✅ MLE fails to converge
✅ You need a quick initial estimate
✅ Data is near distribution bounds

Best practice:
✅ Start with MOM for initial guess
✅ Refine with MLE if needed
✅ Always validate results visually
""")

# ============================================================================
# PRACTICAL TIP
# ============================================================================
print("🚀 Practical Workflow")
print("=" * 70)
print("""
# Fast initial screening
for dist_name in ['normal', 'lognormal', 'gamma', 'weibull']:
    dist = get_distribution(dist_name)
    dist.fit(data, method='mom')  # Fast!
    print(f"{dist_name}: AIC={dist.aic():.2f}")

# Refine best candidate with MLE
best_dist = get_distribution('gamma')
best_dist.fit(data, method='mle')  # Accurate!
print(best_dist.summary())
""")
