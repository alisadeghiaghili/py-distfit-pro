#!/usr/bin/env python3
"""
Goodness-of-Fit Tests
====================

Statistical tests to check if data follows a distribution.

Tests covered:
  - Kolmogorov-Smirnov (KS): General purpose
  - Anderson-Darling (AD): More sensitive to tails
  - Chi-Square: For grouped/discrete data

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
from scipy import stats
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("✅ GOODNESS-OF-FIT TESTS")
print("="*70)


# ============================================================================
# Understanding GoF Tests
# ============================================================================

print("\n" + "="*70)
print("📚 Understanding Goodness-of-Fit Tests")
print("="*70)
print("""
GoF tests answer: "Does my data follow this distribution?"

NULL HYPOTHESIS (H₀): Data follows the specified distribution
ALTERNATIVE (H₁): Data does NOT follow the distribution

P-VALUE INTERPRETATION:
  • p > 0.05: Cannot reject H₀ (data could follow distribution) ✓
  • p < 0.05: Reject H₀ (data does NOT follow distribution) ✗

TEST COMPARISON:

  Kolmogorov-Smirnov (KS):
    • Measures max distance between CDFs
    • Good for continuous distributions
    • Less sensitive to tails
    • Most commonly used
  
  Anderson-Darling (AD):
    • Weighted KS (emphasizes tails)
    • More powerful for tail differences
    • Better for detecting outliers
  
  Chi-Square (χ²):
    • Compares observed vs expected frequencies
    • Good for discrete/grouped data
    • Requires sufficient data in each bin
""")


# ============================================================================
# Example 1: Good Fit (Normal Data)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Good Fit - Normal Distribution")
print("="*70)

# Generate truly normal data
data_good = np.random.normal(loc=100, scale=15, size=500)

print(f"\n📊 Data: {len(data_good)} samples from N(100, 15)")

# Fit distribution
dist_good = get_distribution('normal')
dist_good.fit(data_good)

print(f"\nFitted parameters:")
for param, val in dist_good.params.items():
    print(f"  {param}: {val:.4f}")

# Kolmogorov-Smirnov Test
ks_stat, ks_pval = stats.kstest(data_good, dist_good.cdf)

print(f"\n1️⃣ Kolmogorov-Smirnov Test:")
print(f"  Test statistic: {ks_stat:.6f}")
print(f"  P-value:        {ks_pval:.6f}")

if ks_pval > 0.05:
    print(f"  ✅ Cannot reject H₀ (p > 0.05)")
    print(f"     → Data is consistent with Normal distribution")
else:
    print(f"  ✗ Reject H₀ (p < 0.05)")
    print(f"     → Data does NOT follow Normal distribution")

# Anderson-Darling Test
ad_result = stats.anderson(data_good, dist='norm')

print(f"\n2️⃣ Anderson-Darling Test:")
print(f"  Test statistic: {ad_result.statistic:.6f}")
print(f"  Critical values: {ad_result.critical_values}")
print(f"  Significance levels: {ad_result.significance_level}%")

if ad_result.statistic < ad_result.critical_values[2]:  # 5% level (index 2)
    print(f"  ✅ Cannot reject H₀ (stat < critical value at 5%)")
    print(f"     → Data is consistent with Normal distribution")
else:
    print(f"  ✗ Reject H₀ (stat > critical value at 5%)")
    print(f"     → Data does NOT follow Normal distribution")


# ============================================================================
# Example 2: Bad Fit (Normal fit to Lognormal data)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Bad Fit - Normal on Lognormal Data")
print("="*70)

# Generate lognormal data (right-skewed)
data_bad = np.random.lognormal(mean=4, sigma=0.5, size=500)

print(f"\n📊 Data: {len(data_bad)} samples from Lognormal(4, 0.5)")
print(f"  Skewness: {stats.skew(data_bad):.2f} (right-skewed)")

# Incorrectly fit Normal distribution
dist_bad = get_distribution('normal')
dist_bad.fit(data_bad)

print(f"\nFitted Normal parameters (WRONG model):")
for param, val in dist_bad.params.items():
    print(f"  {param}: {val:.4f}")

# KS Test
ks_stat_bad, ks_pval_bad = stats.kstest(data_bad, dist_bad.cdf)

print(f"\n1️⃣ Kolmogorov-Smirnov Test:")
print(f"  Test statistic: {ks_stat_bad:.6f}")
print(f"  P-value:        {ks_pval_bad:.6f}")

if ks_pval_bad > 0.05:
    print(f"  ✅ Cannot reject H₀")
else:
    print(f"  ✗ Reject H₀ (p < 0.05)")
    print(f"     → Normal distribution is NOT a good fit!")

# Now fit correct distribution (Lognormal)
print(f"\n🔧 Trying correct distribution (Lognormal)...")

dist_correct = get_distribution('lognormal')
dist_correct.fit(data_bad)

ks_stat_correct, ks_pval_correct = stats.kstest(data_bad, dist_correct.cdf)

print(f"\n3️⃣ KS Test with Lognormal:")
print(f"  Test statistic: {ks_stat_correct:.6f}")
print(f"  P-value:        {ks_pval_correct:.6f}")

if ks_pval_correct > 0.05:
    print(f"  ✅ Cannot reject H₀ (p > 0.05)")
    print(f"     → Lognormal is a good fit!")

print(f"\n🎯 Comparison:")
print(f"  Normal fit:    p = {ks_pval_bad:.6f} {'(BAD)' if ks_pval_bad < 0.05 else ''}")
print(f"  Lognormal fit: p = {ks_pval_correct:.6f} {'(GOOD)' if ks_pval_correct > 0.05 else ''}")


# ============================================================================
# Example 3: Chi-Square Test for Discrete Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Chi-Square Test - Poisson Data")
print("="*70)

# Generate Poisson data
lambda_true = 5.0
data_poisson = np.random.poisson(lam=lambda_true, size=500)

print(f"\n📊 Data: {len(data_poisson)} samples from Poisson(λ={lambda_true})")
print(f"  Mean: {data_poisson.mean():.2f}")

# Fit Poisson
dist_poisson = get_distribution('poisson')
dist_poisson.fit(data_poisson)

print(f"\nFitted λ: {dist_poisson.mean():.4f}")

# Chi-Square Test
# Group data into bins
unique_vals, observed_counts = np.unique(data_poisson, return_counts=True)

# Calculate expected frequencies
expected_counts = len(data_poisson) * dist_poisson.pdf(unique_vals)

# Remove bins with expected < 5 (chi-square requirement)
valid_mask = expected_counts >= 5
observed_valid = observed_counts[valid_mask]
expected_valid = expected_counts[valid_mask]

chi2_stat, chi2_pval = stats.chisquare(observed_valid, expected_valid, 
                                        ddof=1)  # 1 parameter estimated

print(f"\n4️⃣ Chi-Square Test:")
print(f"  Test statistic: {chi2_stat:.6f}")
print(f"  P-value:        {chi2_pval:.6f}")
print(f"  Degrees of freedom: {len(observed_valid) - 1 - 1}")

if chi2_pval > 0.05:
    print(f"  ✅ Cannot reject H₀ (p > 0.05)")
    print(f"     → Data is consistent with Poisson distribution")
else:
    print(f"  ✗ Reject H₀ (p < 0.05)")
    print(f"     → Data does NOT follow Poisson distribution")


# ============================================================================
# Visualization: Q-Q Plots
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Q-Q Plot Diagnostics...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Good fit Q-Q plot
ax = axes[0, 0]
theoretical_quantiles = dist_good.ppf(np.linspace(0.01, 0.99, len(data_good)))
empirical_quantiles = np.sort(data_good)

ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=20)
min_val = min(theoretical_quantiles.min(), empirical_quantiles.min())
max_val = max(theoretical_quantiles.max(), empirical_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
ax.set_ylabel('Sample Quantiles', fontsize=10)
ax.set_title(f'Q-Q Plot: Good Fit (p={ks_pval:.4f})', fontweight='bold', color='green')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 2: Bad fit Q-Q plot
ax = axes[0, 1]
theoretical_bad = dist_bad.ppf(np.linspace(0.01, 0.99, len(data_bad)))
empirical_bad = np.sort(data_bad)

ax.scatter(theoretical_bad, empirical_bad, alpha=0.5, s=20, color='orange')
min_val = min(theoretical_bad.min(), empirical_bad.min())
max_val = max(theoretical_bad.max(), empirical_bad.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')

ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
ax.set_ylabel('Sample Quantiles', fontsize=10)
ax.set_title(f'Q-Q Plot: Bad Fit (p={ks_pval_bad:.4f})', fontweight='bold', color='red')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: CDF comparison (good fit)
ax = axes[1, 0]
data_sorted = np.sort(data_good)
empirical_cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
theoretical_cdf = dist_good.cdf(data_sorted)

ax.plot(data_sorted, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF', alpha=0.7)
ax.plot(data_sorted, theoretical_cdf, 'r--', linewidth=2, label='Fitted CDF')

ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Cumulative Probability', fontsize=10)
ax.set_title('CDF Comparison: Good Fit', fontweight='bold', color='green')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 4: CDF comparison (bad fit)
ax = axes[1, 1]
data_sorted_bad = np.sort(data_bad)
empirical_cdf_bad = np.arange(1, len(data_sorted_bad)+1) / len(data_sorted_bad)
theoretical_cdf_bad = dist_bad.cdf(data_sorted_bad)

ax.plot(data_sorted_bad, empirical_cdf_bad, 'b-', linewidth=2, label='Empirical CDF', alpha=0.7)
ax.plot(data_sorted_bad, theoretical_cdf_bad, 'r--', linewidth=2, label='Fitted CDF (Wrong)')

# Also plot correct fit
theoretical_correct = dist_correct.cdf(data_sorted_bad)
ax.plot(data_sorted_bad, theoretical_correct, 'g:', linewidth=2, label='Fitted CDF (Correct)')

ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Cumulative Probability', fontsize=10)
ax.set_title('CDF Comparison: Bad vs Correct Fit', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Q-Q and CDF plots created!")
print("   Q-Q plots: Points on line = good fit")
print("   CDF plots: Lines overlap = good fit")

plt.show()


print("\n" + "="*70)
print("🎓 GoF Test Guidelines")
print("="*70)
print("""
📊 TEST SELECTION:

  Kolmogorov-Smirnov:
    ✅ Continuous distributions
    ✅ Any sample size
    ✅ General-purpose test
    ⚠️  Less sensitive to tails
  
  Anderson-Darling:
    ✅ When tail behavior matters
    ✅ Detecting outliers
    ✅ More powerful than KS
    ⚠️  Limited to specific distributions
  
  Chi-Square:
    ✅ Discrete data
    ✅ Grouped/binned data
    ⚠️  Need ≥5 expected per bin
    ⚠️  Loses information from binning

💡 PRACTICAL TIPS:

  1. ALWAYS visualize (Q-Q plots, CDF comparison)
  2. P-value is not everything (check visual fit)
  3. Large samples: tests may reject "good enough" fits
  4. Small samples: tests may not detect poor fits
  5. Use multiple tests when possible
  6. Consider practical significance, not just statistical

⚠️  COMMON PITFALLS:

  ✗ Relying only on p-values
  ✗ Not checking assumptions (e.g., parameters known vs estimated)
  ✗ Ignoring visual diagnostics
  ✗ Testing too many distributions (multiple testing problem)

RECOMMENDATION:
  🎯 Use GoF tests + AIC/BIC + visual inspection together!

Next: See auto_selection.py for automated distribution selection!
""")
