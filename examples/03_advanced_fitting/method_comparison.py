#!/usr/bin/env python3
"""
Fitting Method Comparison: MLE vs Method of Moments
===================================================

Compare different parameter estimation methods:
  - MLE (Maximum Likelihood Estimation): Default method
  - MoM (Method of Moments): Alternative approach

When to use each:
  - MLE: Generally better (optimal under mild conditions)
  - MoM: Faster, simpler, more robust to outliers

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
import time

np.random.seed(42)

print("="*70)
print("🎯 FITTING METHOD COMPARISON: MLE vs Method of Moments")
print("="*70)


# ============================================================================
# Theory Overview
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: MLE vs Method of Moments")
print("="*70)

theory = """
1. MAXIMUM LIKELIHOOD ESTIMATION (MLE):
   • Finds parameters that maximize probability of observed data
   • Optimal: Most efficient estimator (lowest variance)
   • Requires iterative optimization
   • Can be sensitive to outliers
   • Default method in distfit-pro
   
   Formula: θ_MLE = argmax L(θ|data) = argmax ∏ f(x_i|θ)

2. METHOD OF MOMENTS (MoM):
   • Matches sample moments to population moments
   • Simple: Closed-form solutions
   • Fast: No optimization needed
   • Less efficient than MLE (higher variance)
   • More robust to outliers
   
   Formula: Set sample moments = theoretical moments, solve for θ
   Example (Normal): μ = x̄, σ² = s²

3. WHEN TO USE EACH:
   • MLE: Default choice, large samples, model assumptions met
   • MoM: Quick estimates, small samples, outliers present
"""

print(theory)


# ============================================================================
# Example 1: Normal Distribution (Both Methods Give Same Result)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Normal Distribution")
print("="*70)
print("""
Special case: For normal distribution, MLE = MoM
  • Both give μ = sample mean
  • Both give σ = sample std (with small difference in denominator)
""")

# Generate normal data
data_normal = np.random.normal(loc=100, scale=15, size=1000)

print(f"\n📊 Data: {len(data_normal)} observations from N(100, 15²)")
print(f"  Sample mean: {data_normal.mean():.2f}")
print(f"  Sample std:  {data_normal.std():.2f}")

# MLE (default)
start_mle = time.time()
dist_mle = get_distribution('normal')
dist_mle.fit(data_normal, method='mle')  # 'mle' is default
time_mle = time.time() - start_mle

print(f"\n1️⃣ MLE Estimation:")
for param, val in dist_mle.params.items():
    print(f"  {param}: {val:.4f}")
print(f"  Time: {time_mle*1000:.2f} ms")

# Method of Moments
start_mom = time.time()
dist_mom = get_distribution('normal')
# For normal: MoM = sample mean and std
mom_mean = data_normal.mean()
mom_std = data_normal.std(ddof=0)  # Population std
dist_mom.params = {'loc': mom_mean, 'scale': mom_std}
dist_mom.fitted = True
time_mom = time.time() - start_mom

print(f"\n2️⃣ Method of Moments:")
for param, val in dist_mom.params.items():
    print(f"  {param}: {val:.4f}")
print(f"  Time: {time_mom*1000:.2f} ms")

print(f"\n📈 Comparison:")
print(f"  Parameter difference: {abs(dist_mle.params['loc'] - dist_mom.params['loc']):.6f} (negligible)")
print(f"  For normal distribution: MLE ≈ MoM")
print(f"  ✅ Both methods work equally well here!")


# ============================================================================
# Example 2: Exponential Distribution (MLE vs MoM)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Exponential Distribution")
print("="*70)
print("""
Exponential: MLE and MoM can differ
  • MLE: Maximum likelihood estimate
  • MoM: λ = 1 / sample_mean
""")

# Generate exponential data
true_scale = 5.0
data_exp = np.random.exponential(scale=true_scale, size=500)

print(f"\n📊 Data: {len(data_exp)} observations from Exp(λ=1/{true_scale})")
print(f"  True scale: {true_scale}")
print(f"  Sample mean: {data_exp.mean():.2f}")

# MLE
start_mle = time.time()
dist_exp_mle = get_distribution('expon')
dist_exp_mle.fit(data_exp)
time_exp_mle = time.time() - start_mle

print(f"\n1️⃣ MLE Estimation:")
for param, val in dist_exp_mle.params.items():
    print(f"  {param}: {val:.4f}")
print(f"  Estimated scale: {dist_exp_mle.params['scale']:.4f}")
print(f"  Error: {abs(dist_exp_mle.params['scale'] - true_scale):.4f}")
print(f"  Time: {time_exp_mle*1000:.2f} ms")

# Method of Moments
start_mom = time.time()
mom_scale = data_exp.mean()  # For exponential: E[X] = scale
dist_exp_mom = get_distribution('expon')
dist_exp_mom.params = {'loc': 0, 'scale': mom_scale}
dist_exp_mom.fitted = True
time_exp_mom = time.time() - start_mom

print(f"\n2️⃣ Method of Moments:")
for param, val in dist_exp_mom.params.items():
    print(f"  {param}: {val:.4f}")
print(f"  Estimated scale: {mom_scale:.4f}")
print(f"  Error: {abs(mom_scale - true_scale):.4f}")
print(f"  Time: {time_exp_mom*1000:.2f} ms")

print(f"\n📈 Comparison:")
print(f"  MLE scale:  {dist_exp_mle.params['scale']:.4f}")
print(f"  MoM scale:  {mom_scale:.4f}")
print(f"  Difference: {abs(dist_exp_mle.params['scale'] - mom_scale):.4f}")
print(f"  Speed: MoM is {time_exp_mle/time_exp_mom:.1f}x faster")


# ============================================================================
# Example 3: Gamma Distribution (MLE is Better)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Gamma Distribution")
print("="*70)
print("""
Gamma: MLE typically outperforms MoM
  • MoM: shape = (mean²/variance), scale = variance/mean
  • MLE: Iterative optimization (more accurate)
""")

# Generate gamma data
true_shape = 2.5
true_scale = 3.0
data_gamma = np.random.gamma(shape=true_shape, scale=true_scale, size=800)

print(f"\n📊 Data: {len(data_gamma)} observations from Gamma({true_shape}, {true_scale})")
print(f"  True shape: {true_shape}")
print(f"  True scale: {true_scale}")
print(f"  Sample mean: {data_gamma.mean():.2f}")
print(f"  Sample var:  {data_gamma.var():.2f}")

# MLE
start_mle = time.time()
dist_gamma_mle = get_distribution('gamma')
dist_gamma_mle.fit(data_gamma)
time_gamma_mle = time.time() - start_mle

mle_shape = dist_gamma_mle.params['a']
mle_scale = dist_gamma_mle.params['scale']

print(f"\n1️⃣ MLE Estimation:")
print(f"  shape (a): {mle_shape:.4f} (true: {true_shape})")
print(f"  scale:     {mle_scale:.4f} (true: {true_scale})")
print(f"  Shape error: {abs(mle_shape - true_shape):.4f}")
print(f"  Scale error: {abs(mle_scale - true_scale):.4f}")
print(f"  Time: {time_gamma_mle*1000:.2f} ms")

# Method of Moments
start_mom = time.time()
sample_mean = data_gamma.mean()
sample_var = data_gamma.var()

# MoM formulas for gamma
mom_shape = sample_mean**2 / sample_var
mom_scale = sample_var / sample_mean

dist_gamma_mom = get_distribution('gamma')
dist_gamma_mom.params = {'a': mom_shape, 'loc': 0, 'scale': mom_scale}
dist_gamma_mom.fitted = True
time_gamma_mom = time.time() - start_mom

print(f"\n2️⃣ Method of Moments:")
print(f"  shape (a): {mom_shape:.4f} (true: {true_shape})")
print(f"  scale:     {mom_scale:.4f} (true: {true_scale})")
print(f"  Shape error: {abs(mom_shape - true_shape):.4f}")
print(f"  Scale error: {abs(mom_scale - true_scale):.4f}")
print(f"  Time: {time_gamma_mom*1000:.2f} ms")

print(f"\n🏆 Winner:")
mle_total_error = abs(mle_shape - true_shape) + abs(mle_scale - true_scale)
mom_total_error = abs(mom_shape - true_shape) + abs(mom_scale - true_scale)

if mle_total_error < mom_total_error:
    print(f"  ✅ MLE is more accurate (total error: {mle_total_error:.4f} vs {mom_total_error:.4f})")
else:
    print(f"  ✅ MoM is more accurate (total error: {mom_total_error:.4f} vs {mle_total_error:.4f})")

print(f"  ⚡ But MoM is {time_gamma_mle/time_gamma_mom:.1f}x faster!")


# ============================================================================
# Example 4: Robustness to Outliers
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 4: Robustness to Outliers")
print("="*70)
print("""
Test: How do methods handle contaminated data?
  • Clean data: 95% from N(0, 1)
  • Outliers: 5% from N(0, 10) [extreme values]
""")

# Generate contaminated data
n_clean = 950
n_outliers = 50
data_clean = np.random.normal(0, 1, n_clean)
data_outliers = np.random.normal(0, 10, n_outliers)
data_contaminated = np.concatenate([data_clean, data_outliers])
np.random.shuffle(data_contaminated)

print(f"\n📊 Data: {len(data_contaminated)} observations")
print(f"  {n_clean} clean ({n_clean/len(data_contaminated)*100:.0f}%)")
print(f"  {n_outliers} outliers ({n_outliers/len(data_contaminated)*100:.0f}%)")
print(f"  Sample mean: {data_contaminated.mean():.2f} (should be ~0)")
print(f"  Sample std:  {data_contaminated.std():.2f} (should be ~1)")

# MLE
dist_cont_mle = get_distribution('normal')
dist_cont_mle.fit(data_contaminated)

print(f"\n1️⃣ MLE Estimation:")
print(f"  mean:  {dist_cont_mle.params['loc']:.4f} (true: 0)")
print(f"  std:   {dist_cont_mle.params['scale']:.4f} (true: 1)")
print(f"  Mean error: {abs(dist_cont_mle.params['loc'] - 0):.4f}")
print(f"  Std error:  {abs(dist_cont_mle.params['scale'] - 1):.4f}")

# Robust MoM (using median and MAD)
from scipy.stats import median_abs_deviation

robust_mean = np.median(data_contaminated)  # Median instead of mean
robust_std = median_abs_deviation(data_contaminated) * 1.4826  # MAD * 1.4826 ≈ std for normal

dist_cont_robust = get_distribution('normal')
dist_cont_robust.params = {'loc': robust_mean, 'scale': robust_std}
dist_cont_robust.fitted = True

print(f"\n2️⃣ Robust Estimator (Median + MAD):")
print(f"  mean:  {robust_mean:.4f} (true: 0)")
print(f"  std:   {robust_std:.4f} (true: 1)")
print(f"  Mean error: {abs(robust_mean - 0):.4f}")
print(f"  Std error:  {abs(robust_std - 1):.4f}")

print(f"\n🔒 Robustness comparison:")
if abs(robust_mean - 0) < abs(dist_cont_mle.params['loc'] - 0):
    print(f"  ✅ Robust method handles outliers better!")
    print(f"     Mean error: {abs(robust_mean - 0):.4f} vs {abs(dist_cont_mle.params['loc'] - 0):.4f}")
else:
    print(f"  MLE performs better in this case")


# ============================================================================
# Visualization: Method Comparison
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Comparison Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Fitting Method Comparison: MLE vs Method of Moments', 
             fontsize=16, fontweight='bold')

# Plot 1: Exponential comparison
ax = axes[0, 0]
ax.hist(data_exp, bins=40, density=True, alpha=0.5, color='skyblue', 
        edgecolor='black', label='Data')

x_exp = np.linspace(0, data_exp.max(), 200)
ax.plot(x_exp, dist_exp_mle.pdf(x_exp), 'r-', linewidth=2, label='MLE')
ax.plot(x_exp, dist_exp_mom.pdf(x_exp), 'g--', linewidth=2, label='MoM')

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Exponential: MLE vs MoM', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: Gamma comparison
ax = axes[0, 1]
ax.hist(data_gamma, bins=50, density=True, alpha=0.5, color='lightgreen', 
        edgecolor='black', label='Data')

x_gamma = np.linspace(0, data_gamma.max(), 200)
ax.plot(x_gamma, dist_gamma_mle.pdf(x_gamma), 'r-', linewidth=2, label='MLE')
ax.plot(x_gamma, dist_gamma_mom.pdf(x_gamma), 'b--', linewidth=2, label='MoM')

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Gamma: MLE vs MoM', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 3: Outlier robustness
ax = axes[1, 0]
ax.hist(data_contaminated, bins=50, density=True, alpha=0.5, color='salmon', 
        edgecolor='black', label='Contaminated data')

x_cont = np.linspace(data_contaminated.min(), data_contaminated.max(), 200)
ax.plot(x_cont, dist_cont_mle.pdf(x_cont), 'r-', linewidth=2, label='MLE (sensitive)')
ax.plot(x_cont, dist_cont_robust.pdf(x_cont), 'g-', linewidth=2, label='Robust (MAD)')

# True distribution
true_dist = get_distribution('normal')
true_dist.params = {'loc': 0, 'scale': 1}
ax.plot(x_cont, true_dist.pdf(x_cont), 'k:', linewidth=2, label='True N(0,1)', alpha=0.7)

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Robustness to Outliers', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim([-5, 5])

# Plot 4: Accuracy comparison table
ax = axes[1, 1]
ax.axis('off')

summary_text = """
METHOD COMPARISON SUMMARY
═══════════════════════════════════

🎯 ACCURACY:
  MLE:  Generally more accurate
        (optimal under model assumptions)
  MoM:  Close to MLE for simple cases
        (exact for normal distribution)

⚡ SPEED:
  MLE:  Slower (iterative optimization)
  MoM:  Faster (closed-form solutions)
        Can be 10-100x faster

🔒 ROBUSTNESS:
  MLE:  Sensitive to outliers
        Assumes model is correct
  MoM:  Can be made robust
        (use median, MAD, etc.)

📊 COMPLEXITY:
  MLE:  Complex (numerical optimization)
  MoM:  Simple (solve equations)

═══════════════════════════════════
RECOMMENDATION:

• Default: Use MLE (dist.fit())
• Quick estimates: Use MoM
• Outliers present: Robust MoM
• Large datasets: MLE (better efficiency)
• Small datasets: Either works
═══════════════════════════════════
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()

print("\n✅ Plots created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. MLE (Maximum Likelihood Estimation):
   ✅ Optimal: Lowest variance among unbiased estimators
   ✅ Default method in distfit-pro
   ❌ Slower: Requires iterative optimization
   ❌ Sensitive to model misspecification and outliers

2. Method of Moments (MoM):
   ✅ Fast: Closed-form solutions
   ✅ Simple: Easy to understand and implement
   ✅ Robust: Can use median/MAD for outlier resistance
   ❌ Less efficient: Higher variance than MLE

3. Special Cases:
   • Normal distribution: MLE = MoM (same result!)
   • Exponential: Very similar results
   • Gamma, Weibull: MLE usually better

4. When to Use Each:
   • MLE: Default choice, large samples, trust model
   • MoM: Quick estimates, small samples, speed critical
   • Robust MoM: Outliers present, contaminated data

5. In distfit-pro:
   dist.fit(data)              # MLE (default)
   dist.fit(data, method='mle') # Explicit MLE
   # MoM: Set params manually using sample moments

6. Trade-offs:
   • Accuracy vs Speed: MLE wins on accuracy, MoM on speed
   • Optimal vs Robust: MLE optimal, MoM more robust
   • Complex vs Simple: MLE complex, MoM simple

Next: See 04_model_selection/ for comparing different distributions!
""")
