#!/usr/bin/env python3
"""
Continuous Distributions Showcase
=================================

Explore all continuous distributions available in distfit-pro.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("📊 CONTINUOUS DISTRIBUTIONS SHOWCASE")
print("="*70)


# ============================================================================
# 1. NORMAL (Gaussian) - Most Common
# ============================================================================

print("\n" + "="*70)
print("1. NORMAL DISTRIBUTION (Gaussian)")
print("="*70)
print("""
Use when:
  ✓ Data is symmetric around mean
  ✓ Natural measurement errors
  ✓ Heights, weights, test scores
  ✓ Most common distribution in statistics
""")

data_normal = np.random.normal(loc=100, scale=15, size=1000)
dist_normal = get_distribution('normal')
dist_normal.fit(data_normal)

print("Fitted parameters:")
for param, val in dist_normal.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean: {dist_normal.mean():.2f}")
print(f"Std:  {dist_normal.std():.2f}")


# ============================================================================
# 2. EXPONENTIAL - Time Between Events
# ============================================================================

print("\n" + "="*70)
print("2. EXPONENTIAL DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Time between arrivals/events
  ✓ Failure times (constant hazard rate)
  ✓ Service times
  ✓ Radioactive decay
""")

data_exp = np.random.exponential(scale=5.0, size=1000)
dist_exp = get_distribution('expon')
dist_exp.fit(data_exp)

print("Fitted parameters:")
for param, val in dist_exp.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean (λ⁻¹): {dist_exp.mean():.2f}")
print(f"Median:     {dist_exp.median():.2f}")


# ============================================================================
# 3. LOGNORMAL - Multiplicative Processes
# ============================================================================

print("\n" + "="*70)
print("3. LOGNORMAL DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Data is strictly positive
  ✓ Right-skewed with long tail
  ✓ Income/wealth distributions
  ✓ Stock prices, file sizes
  ✓ Product of many random factors
""")

data_lognorm = np.random.lognormal(mean=3, sigma=0.5, size=1000)
dist_lognorm = get_distribution('lognormal')
dist_lognorm.fit(data_lognorm)

print("Fitted parameters:")
for param, val in dist_lognorm.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean:   {dist_lognorm.mean():.2f}")
print(f"Median: {dist_lognorm.median():.2f}")
print(f"Mode exists but varies with parameters")


# ============================================================================
# 4. GAMMA - Waiting Times
# ============================================================================

print("\n" + "="*70)
print("4. GAMMA DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Sum of exponential random variables
  ✓ Waiting times for multiple events
  ✓ Rainfall amounts
  ✓ Insurance claims
  ✓ Flexible shape (can be skewed or near-normal)
""")

data_gamma = np.random.gamma(shape=2, scale=3, size=1000)
dist_gamma = get_distribution('gamma')
dist_gamma.fit(data_gamma)

print("Fitted parameters:")
for param, val in dist_gamma.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean: {dist_gamma.mean():.2f}")
print(f"Std:  {dist_gamma.std():.2f}")


# ============================================================================
# 5. WEIBULL - Reliability Analysis
# ============================================================================

print("\n" + "="*70)
print("5. WEIBULL DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Reliability engineering (failure analysis)
  ✓ Wind speed modeling
  ✓ Lifetime data
  ✓ Can model increasing/decreasing hazard rates
  ✓ Shape parameter controls behavior
""")

data_weibull = np.random.weibull(a=1.5, size=1000) * 10
dist_weibull = get_distribution('weibull_min')
dist_weibull.fit(data_weibull)

print("Fitted parameters:")
for param, val in dist_weibull.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean: {dist_weibull.mean():.2f}")
shape_param = dist_weibull.params.get('c', dist_weibull.params.get('shape', 0))
if shape_param < 1:
    print("  → Decreasing hazard rate (early failures)")
elif shape_param > 1:
    print("  → Increasing hazard rate (wear-out failures)")
else:
    print("  → Constant hazard rate (random failures)")


# ============================================================================
# 6. BETA - Bounded [0, 1] Data
# ============================================================================

print("\n" + "="*70)
print("6. BETA DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Data is bounded between 0 and 1
  ✓ Probabilities, proportions, percentages
  ✓ Project completion estimates
  ✓ Bayesian priors
  ✓ Very flexible shapes
""")

data_beta = np.random.beta(a=2, b=5, size=1000)
dist_beta = get_distribution('beta')
dist_beta.fit(data_beta)

print("Fitted parameters:")
for param, val in dist_beta.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean:   {dist_beta.mean():.4f}")
print(f"Median: {dist_beta.median():.4f}")


# ============================================================================
# 7. UNIFORM - Equal Probability
# ============================================================================

print("\n" + "="*70)
print("7. UNIFORM DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ All values equally likely in range [a, b]
  ✓ Random number generation
  ✓ Lack of information (maximum entropy)
  ✓ Simple baseline model
""")

data_uniform = np.random.uniform(low=10, high=50, size=1000)
dist_uniform = get_distribution('uniform')
dist_uniform.fit(data_uniform)

print("Fitted parameters:")
for param, val in dist_uniform.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean:   {dist_uniform.mean():.2f}")
print(f"Range:  [{data_uniform.min():.2f}, {data_uniform.max():.2f}]")


# ============================================================================
# 8. CHI-SQUARE - Sum of Squared Normals
# ============================================================================

print("\n" + "="*70)
print("8. CHI-SQUARE DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Sum of squared standard normal variables
  ✓ Variance estimation
  ✓ Goodness-of-fit tests
  ✓ Degrees of freedom = shape parameter
""")

data_chi2 = np.random.chisquare(df=5, size=1000)
dist_chi2 = get_distribution('chi2')
dist_chi2.fit(data_chi2)

print("Fitted parameters:")
for param, val in dist_chi2.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean: {dist_chi2.mean():.2f} (equals df)")
print(f"Std:  {dist_chi2.std():.2f}")


# ============================================================================
# 9. STUDENT'S T - Heavy Tails
# ============================================================================

print("\n" + "="*70)
print("9. STUDENT'S T DISTRIBUTION")
print("="*70)
print("""
Use when:
  ✓ Small sample sizes
  ✓ Heavier tails than normal
  ✓ Outliers present
  ✓ Robust alternative to normal
""")

data_t = np.random.standard_t(df=10, size=1000)
dist_t = get_distribution('t')
dist_t.fit(data_t)

print("Fitted parameters:")
for param, val in dist_t.params.items():
    print(f"  {param}: {val:.4f}")

print(f"\nMean: {dist_t.mean():.2f}")
print(f"Std:  {dist_t.std():.2f}")


# ============================================================================
# Visual Comparison
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Visual Comparison...")
print("="*70)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle('Continuous Distributions Gallery', fontsize=16, fontweight='bold')

distributions = [
    (data_normal, dist_normal, 'Normal'),
    (data_exp, dist_exp, 'Exponential'),
    (data_lognorm, dist_lognorm, 'Lognormal'),
    (data_gamma, dist_gamma, 'Gamma'),
    (data_weibull, dist_weibull, 'Weibull'),
    (data_beta, dist_beta, 'Beta'),
    (data_uniform, dist_uniform, 'Uniform'),
    (data_chi2, dist_chi2, 'Chi-Square'),
    (data_t, dist_t, "Student's t"),
]

for idx, (data, dist, name) in enumerate(distributions):
    ax = axes[idx // 3, idx % 3]
    
    # Histogram
    ax.hist(data, bins=40, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Fitted PDF
    x = np.linspace(data.min(), data.max(), 200)
    y = dist.pdf(x)
    ax.plot(x, y, 'r-', linewidth=2)
    
    ax.set_title(f'{name}', fontweight='bold')
    ax.set_xlabel('Value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=8)

plt.tight_layout()

print("\n✅ Visual comparison created!")
print("   Close the plot window to continue...")

plt.show()

print("\n" + "="*70)
print("🎓 Summary")
print("="*70)
print("""
You've explored 9 continuous distributions!

Quick Selection Guide:
  • Symmetric data           → Normal
  • Right-skewed, positive   → Lognormal, Gamma, Weibull
  • Time between events      → Exponential
  • Bounded [0,1]            → Beta
  • Heavy tails/outliers     → Student's t
  • Bounded [a,b]            → Uniform
  • Sum of squared normals   → Chi-Square

Next: See discrete_distributions.py for count data!
""")
