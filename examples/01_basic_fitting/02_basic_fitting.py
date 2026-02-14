"""Basic Distribution Fitting - Comprehensive Walkthrough

This example shows the complete workflow for fitting distributions:
- Data preparation and validation
- Fitting with MLE (Maximum Likelihood)
- Parameter interpretation
- Statistical summaries
- Model diagnostics

Perfect for: Understanding the fitting process in detail
Time: ~10 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution
from distfit_pro.utils.config import config

print("="*70)
print("📚 BASIC FITTING: Complete Workflow")
print("="*70)

# ============================================================================
# PART 1: Data Preparation
# ============================================================================
print("\n" + "="*70)
print("PART 1: Data Preparation")
print("="*70)

# Simulate real-world scenario: product lifetimes (exponential distribution)
np.random.seed(123)
true_scale = 500  # Mean lifetime = 500 hours
data = np.random.exponential(scale=true_scale, size=200)

print(f"\n📊 Data Summary:")
print(f"  Sample size: {len(data)}")
print(f"  Mean: {data.mean():.2f}")
print(f"  Std Dev: {data.std():.2f}")
print(f"  Min: {data.min():.2f}")
print(f"  Max: {data.max():.2f}")
print(f"  Median: {np.median(data):.2f}")

# Data validation
print("\n🔍 Data Validation:")
has_nan = np.any(np.isnan(data))
has_inf = np.any(np.isinf(data))
has_negative = np.any(data < 0)

print(f"  NaN values: {has_nan} ✓" if not has_nan else "  NaN values: {has_nan} ✗")
print(f"  Infinite values: {has_inf} ✓" if not has_inf else "  Infinite values: {has_inf} ✗")
print(f"  Negative values: {has_negative} ✓" if not has_negative else "  Negative values: {has_negative} ✗")

# ============================================================================
# PART 2: Fitting with MLE
# ============================================================================
print("\n" + "="*70)
print("PART 2: Maximum Likelihood Estimation (MLE)")
print("="*70)

# Create distribution
dist = get_distribution('exponential')
print(f"\n📈 Distribution: {dist.info.display_name}")
print(f"   Description: {dist.info.description}")
print(f"   Parameters: {dist.info.parameters}")
print(f"   Support: {dist.info.support}")

# Fit using MLE (default)
print("\n🔧 Fitting with Maximum Likelihood Estimation...")
dist.fit(data, method='mle')
print("✓ Fitting completed!")

# ============================================================================
# PART 3: Parameter Interpretation
# ============================================================================
print("\n" + "="*70)
print("PART 3: Parameter Interpretation")
print("="*70)

params = dist.params
print("\n📊 Estimated Parameters:")
for name, value in params.items():
    print(f"  {name:<10} = {value:>12.4f}")

print("\n🎯 Comparison with True Values:")
print(f"  {'Parameter':<15} {'Estimated':<15} {'True':<15} {'Error %'}")
print("-" * 60)
estimated_scale = params['scale']
error_pct = abs(estimated_scale - true_scale) / true_scale * 100
print(f"  {'scale':<15} {estimated_scale:<15.4f} {true_scale:<15.4f} {error_pct:.2f}%")

print("\n💡 Interpretation:")
print(f"  Mean lifetime = {estimated_scale:.2f} hours")
print(f"  Half-life (median) = {estimated_scale * np.log(2):.2f} hours")
print(f"  99% fail by: {dist.ppf(0.99):.2f} hours")

# ============================================================================
# PART 4: Statistical Summary
# ============================================================================
print("\n" + "="*70)
print("PART 4: Statistical Summary")
print("="*70)

print("\n📋 Distribution Statistics:")
print(f"  Mean (E[X]): {dist.mean():.4f}")
print(f"  Variance: {dist.var():.4f}")
print(f"  Std Dev: {dist.std():.4f}")
print(f"  Median: {dist.median():.4f}")
print(f"  Skewness: {dist.skewness():.4f}")
print(f"  Kurtosis: {dist.kurtosis():.4f}")

print("\n📊 Quantiles:")
for q in [0.25, 0.5, 0.75, 0.9, 0.95, 0.99]:
    val = dist.ppf(q)
    print(f"  {int(q*100):>3}th percentile: {val:>10.2f}")

# ============================================================================
# PART 5: Model Diagnostics
# ============================================================================
print("\n" + "="*70)
print("PART 5: Model Diagnostics")
print("="*70)

# Goodness of fit metrics
ll = dist.log_likelihood()
aic = dist.aic()
bic = dist.bic()

print("\n✅ Goodness of Fit:")
print(f"  Log-Likelihood: {ll:.2f}")
print(f"  AIC: {aic:.2f}")
print(f"  BIC: {bic:.2f}")
print("\n  Note: Lower AIC/BIC indicates better fit")

# Kolmogorov-Smirnov test
from scipy import stats
ks_stat, ks_pvalue = stats.kstest(data, lambda x: dist.cdf(x))
print(f"\n📊 Kolmogorov-Smirnov Test:")
print(f"  Statistic: {ks_stat:.4f}")
print(f"  P-value: {ks_pvalue:.4f}")
if ks_pvalue > 0.05:
    print("  ✓ Cannot reject H0: Data follows exponential distribution (p > 0.05)")
else:
    print("  ✗ Reject H0: Data does not follow exponential distribution (p ≤ 0.05)")

# ============================================================================
# PART 6: Visualization
# ============================================================================
print("\n" + "="*70)
print("PART 6: Visualization")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Histogram + PDF
ax1 = axes[0, 0]
ax1.hist(data, bins=30, density=True, alpha=0.7, color='lightblue', 
         edgecolor='black', label='Observed Data')
x = np.linspace(0, data.max(), 300)
ax1.plot(x, dist.pdf(x), 'r-', linewidth=2, label='Fitted Exponential')
ax1.set_xlabel('Lifetime (hours)', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('Histogram + Fitted PDF', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: CDF Comparison
ax2 = axes[0, 1]
sorted_data = np.sort(data)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, empirical_cdf, 'o', markersize=3, alpha=0.5, label='Empirical CDF')
ax2.plot(x, dist.cdf(x), 'r-', linewidth=2, label='Fitted CDF')
ax2.set_xlabel('Lifetime (hours)', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('CDF Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Plot 3: Q-Q Plot
ax3 = axes[1, 0]
theoretical_quantiles = dist.ppf(np.linspace(0.01, 0.99, len(data)))
sample_quantiles = np.sort(data)
ax3.scatter(theoretical_quantiles, sample_quantiles, alpha=0.5, s=20)
ax3.plot([0, max(theoretical_quantiles)], [0, max(theoretical_quantiles)], 
         'r--', linewidth=2, label='Perfect Fit')
ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
ax3.set_ylabel('Sample Quantiles', fontsize=11)
ax3.set_title('Q-Q Plot', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: Residuals
ax4 = axes[1, 1]
expected_probs = dist.cdf(sorted_data)
residuals = empirical_cdf - expected_probs
ax4.scatter(sorted_data, residuals, alpha=0.5, s=20)
ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Lifetime (hours)', fontsize=11)
ax4.set_ylabel('CDF Residual', fontsize=11)
ax4.set_title('CDF Residual Plot', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('basic_fitting_diagnostics.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: basic_fitting_diagnostics.png")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 COMPLETE! Basic Fitting Workflow Finished.")
print("="*70)
print("\nKey Takeaways:")
print("  1. Always validate data before fitting (NaN, Inf, range)")
print("  2. MLE provides optimal parameter estimates")
print("  3. Compare estimated vs true parameters when known")
print("  4. Use multiple diagnostics (AIC, BIC, K-S test, Q-Q plot)")
print("  5. Visualize both PDF and CDF for complete picture")
print("\nNext Steps:")
print("  → Try 03_comparing_distributions.py to compare multiple models")
print("  → Try 04_method_of_moments.py for faster estimation")
print("="*70)
