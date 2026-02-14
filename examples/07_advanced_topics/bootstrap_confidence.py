#!/usr/bin/env python3
"""
Bootstrap Confidence Intervals
==============================

Quantify parameter uncertainty using bootstrap:
  - Parametric bootstrap
  - Non-parametric bootstrap
  - Confidence intervals for parameters
  - Distribution of estimates

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("🥾 BOOTSTRAP CONFIDENCE INTERVALS")
print("="*70)


# ============================================================================
# Theory: Bootstrap
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: Bootstrap Method")
print("="*70)

theory = """
1. WHAT IS BOOTSTRAP?
   • Resampling technique to estimate uncertainty
   • Create many "pseudo-datasets" from original data
   • Fit model to each bootstrap sample
   • Distribution of estimates → confidence intervals

2. NON-PARAMETRIC BOOTSTRAP:
   • Resample data with replacement
   • No distributional assumptions
   • Each bootstrap sample: same size as original
   • B bootstrap samples (e.g., B = 1000)

3. PARAMETRIC BOOTSTRAP:
   • Fit distribution to data
   • Generate new data from fitted distribution
   • Refit distribution to generated data
   • Assumes fitted distribution is correct

4. CONFIDENCE INTERVALS:
   • Percentile method: 2.5th and 97.5th percentiles of bootstrap estimates
   • 95% CI means parameter is in this range with 95% confidence
   • Wider CI = more uncertainty

5. WHEN TO USE:
   • Want to quantify parameter uncertainty
   • Small samples (where theory is unreliable)
   • Complex statistics (no closed-form CI)
   • Non-normal distributions

6. ADVANTAGES:
   • Simple and general
   • No assumptions about distribution
   • Works for any statistic
   • Provides full sampling distribution
"""

print(theory)


# ============================================================================
# Example 1: Non-Parametric Bootstrap
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Non-Parametric Bootstrap")
print("="*70)

print("""
Scenario: Small sample (n=50), estimate mean and std with CI
""")

# Generate small sample
data = np.random.normal(100, 15, 50)

print(f"\n📊 Original Data: {len(data)} observations")
print(f"  Sample mean: {data.mean():.2f}")
print(f"  Sample std:  {data.std():.2f}")

# Non-parametric bootstrap
n_bootstrap = 1000
bootstrap_means = []
bootstrap_stds = []

print(f"\n🔄 Running non-parametric bootstrap ({n_bootstrap} iterations)...")

for _ in range(n_bootstrap):
    # Resample with replacement
    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
    bootstrap_means.append(bootstrap_sample.mean())
    bootstrap_stds.append(bootstrap_sample.std())

bootstrap_means = np.array(bootstrap_means)
bootstrap_stds = np.array(bootstrap_stds)

# Calculate 95% confidence intervals
ci_mean = np.percentile(bootstrap_means, [2.5, 97.5])
ci_std = np.percentile(bootstrap_stds, [2.5, 97.5])

print(f"\n📈 Bootstrap Results:")
print(f"\n  Mean:")
print(f"    Point estimate: {data.mean():.2f}")
print(f"    95% CI: [{ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
print(f"    Bootstrap SE: {bootstrap_means.std():.2f}")

print(f"\n  Standard Deviation:")
print(f"    Point estimate: {data.std():.2f}")
print(f"    95% CI: [{ci_std[0]:.2f}, {ci_std[1]:.2f}]")
print(f"    Bootstrap SE: {bootstrap_stds.std():.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Bootstrap distribution of mean
ax = axes[0]
ax.hist(bootstrap_means, bins=40, density=True, alpha=0.6, color='skyblue',
        edgecolor='black', label='Bootstrap Distribution')

# Mark CI
ax.axvline(ci_mean[0], color='red', linestyle='--', linewidth=2, label='95% CI')
ax.axvline(ci_mean[1], color='red', linestyle='--', linewidth=2)
ax.axvline(data.mean(), color='green', linestyle='-', linewidth=2.5, label='Observed')

# Add normal approximation
x_mean = np.linspace(bootstrap_means.min(), bootstrap_means.max(), 200)
ax.plot(x_mean, stats.norm(bootstrap_means.mean(), bootstrap_means.std()).pdf(x_mean),
        'b-', linewidth=2, label='Normal Approx')

ax.set_xlabel('Mean', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution of Mean', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Bootstrap distribution of std
ax = axes[1]
ax.hist(bootstrap_stds, bins=40, density=True, alpha=0.6, color='lightcoral',
        edgecolor='black', label='Bootstrap Distribution')

ax.axvline(ci_std[0], color='red', linestyle='--', linewidth=2, label='95% CI')
ax.axvline(ci_std[1], color='red', linestyle='--', linewidth=2)
ax.axvline(data.std(), color='green', linestyle='-', linewidth=2.5, label='Observed')

ax.set_xlabel('Standard Deviation', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution of Std Dev', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Non-parametric bootstrap plots created!")
plt.savefig('/tmp/bootstrap_nonparametric.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Parametric Bootstrap for Distribution Parameters
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Parametric Bootstrap for Weibull Parameters")
print("="*70)

print("""
Scenario: Estimate Weibull shape and scale with confidence intervals
""")

# Generate Weibull data
true_shape = 2.5
true_scale = 100
data_weibull = np.random.weibull(true_shape, 100) * true_scale

print(f"\n📊 Data: {len(data_weibull)} observations from Weibull({true_shape}, {true_scale})")

# Fit Weibull
print(f"\n🔬 Fitting Weibull distribution...")
dist_weibull = get_distribution('weibull_min')
dist_weibull.fit(data_weibull)

shape_fit = dist_weibull.params['c']
scale_fit = dist_weibull.params['scale']

print(f"  Fitted parameters:")
print(f"    Shape: {shape_fit:.3f} (true: {true_shape})")
print(f"    Scale: {scale_fit:.2f} (true: {true_scale})")

# Parametric bootstrap
n_bootstrap_param = 1000
bootstrap_shapes = []
bootstrap_scales = []

print(f"\n🔄 Running parametric bootstrap ({n_bootstrap_param} iterations)...")

for _ in range(n_bootstrap_param):
    # Generate data from fitted distribution
    bootstrap_data = dist_weibull.rvs(len(data_weibull))
    
    # Fit to bootstrap data
    dist_boot = get_distribution('weibull_min')
    dist_boot.fit(bootstrap_data)
    
    bootstrap_shapes.append(dist_boot.params['c'])
    bootstrap_scales.append(dist_boot.params['scale'])

bootstrap_shapes = np.array(bootstrap_shapes)
bootstrap_scales = np.array(bootstrap_scales)

# Calculate 95% confidence intervals
ci_shape = np.percentile(bootstrap_shapes, [2.5, 97.5])
ci_scale = np.percentile(bootstrap_scales, [2.5, 97.5])

print(f"\n📈 Parametric Bootstrap Results:")
print(f"\n  Shape Parameter:")
print(f"    Point estimate: {shape_fit:.3f}")
print(f"    95% CI: [{ci_shape[0]:.3f}, {ci_shape[1]:.3f}]")
print(f"    True value: {true_shape:.3f}")
print(f"    Bootstrap SE: {bootstrap_shapes.std():.3f}")

print(f"\n  Scale Parameter:")
print(f"    Point estimate: {scale_fit:.2f}")
print(f"    95% CI: [{ci_scale[0]:.2f}, {ci_scale[1]:.2f}]")
print(f"    True value: {true_scale:.2f}")
print(f"    Bootstrap SE: {bootstrap_scales.std():.2f}")

# Check if true values are in CI
shape_in_ci = ci_shape[0] <= true_shape <= ci_shape[1]
scale_in_ci = ci_scale[0] <= true_scale <= ci_scale[1]

print(f"\n✅ Validation:")
print(f"  True shape in CI: {'YES ✓' if shape_in_ci else 'NO ✗'}")
print(f"  True scale in CI: {'YES ✓' if scale_in_ci else 'NO ✗'}")

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Parametric Bootstrap: Weibull Parameters', fontsize=16, fontweight='bold')

# Plot 1: Shape parameter distribution
ax = axes[0, 0]
ax.hist(bootstrap_shapes, bins=40, density=True, alpha=0.6, color='skyblue',
        edgecolor='black')
ax.axvline(ci_shape[0], color='red', linestyle='--', linewidth=2, label='95% CI')
ax.axvline(ci_shape[1], color='red', linestyle='--', linewidth=2)
ax.axvline(shape_fit, color='green', linestyle='-', linewidth=2.5, label='Estimate')
ax.axvline(true_shape, color='blue', linestyle=':', linewidth=2.5, label='True')
ax.set_xlabel('Shape Parameter', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution: Shape', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Scale parameter distribution
ax = axes[0, 1]
ax.hist(bootstrap_scales, bins=40, density=True, alpha=0.6, color='lightcoral',
        edgecolor='black')
ax.axvline(ci_scale[0], color='red', linestyle='--', linewidth=2, label='95% CI')
ax.axvline(ci_scale[1], color='red', linestyle='--', linewidth=2)
ax.axvline(scale_fit, color='green', linestyle='-', linewidth=2.5, label='Estimate')
ax.axvline(true_scale, color='blue', linestyle=':', linewidth=2.5, label='True')
ax.set_xlabel('Scale Parameter', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Bootstrap Distribution: Scale', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Joint distribution (scatter)
ax = axes[1, 0]
ax.scatter(bootstrap_shapes, bootstrap_scales, alpha=0.3, s=10, color='blue')
ax.scatter([shape_fit], [scale_fit], s=200, marker='*', color='green',
           edgecolors='black', linewidth=2, label='Estimate', zorder=5)
ax.scatter([true_shape], [true_scale], s=200, marker='x', color='red',
           linewidths=3, label='True', zorder=5)
ax.set_xlabel('Shape Parameter', fontsize=11, fontweight='bold')
ax.set_ylabel('Scale Parameter', fontsize=11, fontweight='bold')
ax.set_title('Joint Distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Fitted distribution with uncertainty
ax = axes[1, 1]
ax.hist(data_weibull, bins=30, density=True, alpha=0.4, color='gray',
        edgecolor='black', label='Data')

x_wei = np.linspace(0, data_weibull.max(), 200)

# Plot sample of bootstrap distributions (uncertainty)
for i in range(0, len(bootstrap_shapes), 50):
    d_boot = get_distribution('weibull_min')
    d_boot.params = {'c': bootstrap_shapes[i], 'scale': bootstrap_scales[i], 'loc': 0}
    ax.plot(x_wei, d_boot.pdf(x_wei), 'b-', alpha=0.05, linewidth=1)

# Plot fitted distribution
ax.plot(x_wei, dist_weibull.pdf(x_wei), 'r-', linewidth=3, label='Fitted')

ax.set_xlabel('Value', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Fitted Distribution with Uncertainty', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Parametric bootstrap plots created!")
plt.savefig('/tmp/bootstrap_parametric.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Bootstrap")
print("="*70)
print("""
1. BOOTSTRAP METHODS:
   • Non-parametric: Resample data directly
   • Parametric: Generate from fitted distribution
   • Both provide uncertainty estimates

2. WHEN TO USE EACH:
   Non-parametric:
     + No distributional assumptions
     + Robust to model misspecification
     - Requires more data
   
   Parametric:
     + More efficient (if model correct)
     + Works with smaller samples
     - Assumes model is correct

3. CONFIDENCE INTERVALS:
   • 95% CI: Parameter likely in this range
   • Wider CI = more uncertainty
   • Use percentile method (2.5th, 97.5th)
   • Can do any confidence level (90%, 99%, etc.)

4. NUMBER OF BOOTSTRAP SAMPLES:
   • B = 1000: Standard for CI
   • B = 10000: For publication
   • More is better but slower

5. APPLICATIONS:
   • Parameter uncertainty quantification
   • Complex statistics (no formula)
   • Small sample inference
   • Model comparison

6. IMPLEMENTATION:
   # Non-parametric
   for _ in range(n_bootstrap):
       sample = np.random.choice(data, len(data), replace=True)
       # Compute statistic
   
   # Parametric
   dist.fit(data)
   for _ in range(n_bootstrap):
       sample = dist.rvs(len(data))
       # Refit and store parameters

7. BEST PRACTICES:
   ✓ Use B ≥ 1000 for stable CI
   ✓ Check bootstrap distribution shape
   ✓ Report both point estimate and CI
   ✓ Visualize uncertainty (uncertainty bands)
   ✓ Set random seed for reproducibility

8. LIMITATIONS:
   • Computationally intensive
   • Assumes sample represents population
   • Doesn't fix fundamental data issues
   • Parametric assumes correct model

Next: See custom_distributions.py to create your own!
""")
