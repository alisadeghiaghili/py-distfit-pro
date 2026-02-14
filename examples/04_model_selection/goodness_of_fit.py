#!/usr/bin/env python3
"""
Goodness-of-Fit Tests
====================

Statistical tests to assess how well a distribution fits data:
  - Kolmogorov-Smirnov (KS) test: Distribution comparison
  - Chi-square test: Binned data comparison
  - Anderson-Darling (AD) test: More sensitive in tails

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
# Theory: GoF Tests
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: Goodness-of-Fit Tests")
print("="*70)

theory = """
1. KOLMOGOROV-SMIRNOV (KS) TEST:
   • Compares empirical CDF with theoretical CDF
   • Test statistic: D = max|F_empirical(x) - F_theoretical(x)|
   • Sensitive to differences in middle of distribution
   • Null hypothesis: Data follows specified distribution
   • p-value > 0.05 → Don't reject (distribution fits)
   • p-value < 0.05 → Reject (distribution doesn't fit)

2. CHI-SQUARE (χ²) TEST:
   • Compares observed vs expected frequencies in bins
   • Works with binned data
   • Test statistic: χ² = Σ(Observed - Expected)²/Expected
   • Requires adequate bin counts (≥5 per bin)
   • p-value > 0.05 → Distribution fits
   • p-value < 0.05 → Distribution doesn't fit

3. ANDERSON-DARLING (AD) TEST:
   • Modified KS test, more sensitive in tails
   • Weighted test: emphasizes tail differences
   • Better for detecting departures in extremes
   • Critical values depend on distribution
   • Statistic < Critical value → Distribution fits

4. CHOOSING A TEST:
   • KS: General purpose, any distribution
   • Chi-square: Discrete data, histograms
   • AD: When tails matter (risk, reliability)
   • Use multiple tests for robustness!
"""

print(theory)


# ============================================================================
# Example 1: Good Fit (Data from Fitted Distribution)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Good Fit Scenario")
print("="*70)
print("""
Test: Does normal data fit normal distribution?
(Expected result: YES - p-value > 0.05)
""")

# Generate normal data
data_good = np.random.normal(loc=50, scale=10, size=1000)

print(f"\n📊 Data: {len(data_good)} samples from N(50, 10²)")
print(f"  Mean: {data_good.mean():.2f}")
print(f"  Std:  {data_good.std():.2f}")

# Fit normal distribution
dist_good = get_distribution('normal')
dist_good.fit(data_good)

print(f"\n✅ Fitted Normal Distribution:")
for param, val in dist_good.params.items():
    print(f"  {param}: {val:.4f}")

# 1. Kolmogorov-Smirnov Test
ks_stat, ks_pvalue = stats.kstest(data_good, dist_good.cdf)

print(f"\n1️⃣ Kolmogorov-Smirnov Test:")
print(f"  Test statistic: {ks_stat:.6f}")
print(f"  p-value:        {ks_pvalue:.6f}")
print(f"  Result: {'✅ PASS (distribution fits)' if ks_pvalue > 0.05 else '❌ FAIL (distribution does not fit)'}")

# 2. Chi-square Test (binned)
n_bins = 20
observed_freq, bin_edges = np.histogram(data_good, bins=n_bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
bin_width = bin_edges[1] - bin_edges[0]

# Expected frequencies
expected_prob = np.diff(dist_good.cdf(bin_edges))
expected_freq = expected_prob * len(data_good)

# Remove bins with low expected counts
mask = expected_freq >= 5
observed_filtered = observed_freq[mask]
expected_filtered = expected_freq[mask]

chi2_stat, chi2_pvalue = stats.chisquare(observed_filtered, expected_filtered)

print(f"\n2️⃣ Chi-Square Test ({np.sum(mask)} bins):")
print(f"  Test statistic: {chi2_stat:.4f}")
print(f"  p-value:        {chi2_pvalue:.6f}")
print(f"  Result: {'✅ PASS (distribution fits)' if chi2_pvalue > 0.05 else '❌ FAIL (distribution does not fit)'}")

# 3. Anderson-Darling Test
ad_result = stats.anderson(data_good, dist='norm')

print(f"\n3️⃣ Anderson-Darling Test:")
print(f"  Test statistic: {ad_result.statistic:.4f}")
print(f"  Critical values: {ad_result.critical_values}")
print(f"  Significance:    {ad_result.significance_level}%")
if ad_result.statistic < ad_result.critical_values[2]:  # 5% level
    print(f"  Result: ✅ PASS (statistic < critical value at 5%)")
else:
    print(f"  Result: ❌ FAIL (statistic > critical value at 5%)")

print(f"\n🎯 Conclusion: All tests suggest good fit!")


# ============================================================================
# Example 2: Poor Fit (Wrong Distribution)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Poor Fit Scenario")
print("="*70)
print("""
Test: Does exponential data fit normal distribution?
(Expected result: NO - p-value < 0.05)
""")

# Generate exponential data (right-skewed)
data_poor = np.random.exponential(scale=10, size=1000)

print(f"\n📊 Data: {len(data_poor)} samples from Exp(λ=1/10)")
print(f"  Mean:     {data_poor.mean():.2f}")
print(f"  Median:   {np.median(data_poor):.2f}")
print(f"  Skewness: {stats.skew(data_poor):.2f}")

# WRONG: Fit normal distribution to exponential data
dist_wrong = get_distribution('normal')
dist_wrong.fit(data_poor)

print(f"\n❌ Incorrectly Fitted Normal Distribution:")
for param, val in dist_wrong.params.items():
    print(f"  {param}: {val:.4f}")

# Kolmogorov-Smirnov Test
ks_stat_wrong, ks_pvalue_wrong = stats.kstest(data_poor, dist_wrong.cdf)

print(f"\n1️⃣ Kolmogorov-Smirnov Test:")
print(f"  Test statistic: {ks_stat_wrong:.6f}")
print(f"  p-value:        {ks_pvalue_wrong:.6f}")
print(f"  Result: {'✅ PASS' if ks_pvalue_wrong > 0.05 else '❌ FAIL (normal distribution does NOT fit!)'}")

print(f"\n🎯 Conclusion: Tests correctly reject normal distribution!")

# Now fit CORRECT distribution
print(f"\n✅ Fitting CORRECT Distribution (Exponential):")
dist_correct = get_distribution('expon')
dist_correct.fit(data_poor)

ks_stat_correct, ks_pvalue_correct = stats.kstest(data_poor, dist_correct.cdf)

print(f"\n1️⃣ KS Test with Exponential:")
print(f"  p-value: {ks_pvalue_correct:.6f}")
print(f"  Result: {'✅ PASS (exponential fits well!)' if ks_pvalue_correct > 0.05 else '❌ FAIL'}")


# ============================================================================
# Example 3: Comparing Multiple Distributions
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Compare Multiple Distributions")
print("="*70)
print("""
Scenario: Test which distribution fits best
Data: Gamma-distributed (skewed, positive)
""")

# Generate gamma data
data_gamma = np.random.gamma(shape=3, scale=2, size=800)

print(f"\n📊 Data: {len(data_gamma)} samples from Gamma(3, 2)")

# Test multiple distributions
candidates = ['normal', 'lognormal', 'gamma', 'weibull_min', 'expon']

test_results = []

for dist_name in candidates:
    try:
        dist = get_distribution(dist_name)
        dist.fit(data_gamma)
        
        # KS test
        ks_stat, ks_pval = stats.kstest(data_gamma, dist.cdf)
        
        test_results.append({
            'name': dist_name,
            'dist': dist,
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'aic': dist.aic(),
        })
    except Exception as e:
        print(f"  Failed {dist_name}: {e}")

print("\n" + "="*70)
print("Goodness-of-Fit Results")
print("="*70)

print(f"\n{'Distribution':<15} {'KS Stat':<10} {'p-value':<12} {'Result':<15} {'AIC':<10}")
print("-"*70)

for r in sorted(test_results, key=lambda x: x['ks_stat']):
    pass_fail = "✅ PASS" if r['ks_pval'] > 0.05 else "❌ FAIL"
    print(f"{r['name']:<15} {r['ks_stat']:<10.6f} {r['ks_pval']:<12.6f} {pass_fail:<15} {r['aic']:<10.2f}")

# Best by p-value
best_pval = max(test_results, key=lambda x: x['ks_pval'])
print(f"\n🏆 Best by p-value: {best_pval['name']} (p = {best_pval['ks_pval']:.6f})")

# Best by KS statistic
best_ks = min(test_results, key=lambda x: x['ks_stat'])
print(f"🏆 Best by KS stat:  {best_ks['name']} (D = {best_ks['ks_stat']:.6f})")

# Best by AIC
best_aic = min(test_results, key=lambda x: x['aic'])
print(f"🏆 Best by AIC:      {best_aic['name']} (AIC = {best_aic['aic']:.2f})")


# ============================================================================
# Visualization: GoF Assessment
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Goodness-of-Fit Tests Visualization', fontsize=16, fontweight='bold')

# Plot 1: Good fit - Histogram + PDF
ax = axes[0, 0]
ax.hist(data_good, bins=40, density=True, alpha=0.6, color='skyblue', 
        edgecolor='black', label='Data')
x = np.linspace(data_good.min(), data_good.max(), 200)
ax.plot(x, dist_good.pdf(x), 'r-', linewidth=2, label='Fitted Normal')
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title(f'Good Fit: p={ks_pvalue:.4f}', fontweight='bold', color='green')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Good fit - QQ plot
ax = axes[0, 1]
theoretical_quantiles = dist_good.ppf(np.linspace(0.01, 0.99, len(data_good)))
empirical_quantiles = np.sort(data_good)
ax.scatter(theoretical_quantiles, empirical_quantiles, alpha=0.5, s=10)
min_val, max_val = min(theoretical_quantiles.min(), empirical_quantiles.min()), \
                   max(theoretical_quantiles.max(), empirical_quantiles.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles', fontsize=10)
ax.set_ylabel('Empirical Quantiles', fontsize=10)
ax.set_title('Q-Q Plot: Good Fit', fontweight='bold', color='green')
ax.grid(True, alpha=0.3)

# Plot 3: Good fit - CDF comparison
ax = axes[0, 2]
data_sorted = np.sort(data_good)
empirical_cdf = np.arange(1, len(data_sorted)+1) / len(data_sorted)
ax.plot(data_sorted, empirical_cdf, 'b-', linewidth=1, label='Empirical CDF', alpha=0.7)
ax.plot(data_sorted, dist_good.cdf(data_sorted), 'r-', linewidth=2, label='Theoretical CDF')
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Cumulative Probability', fontsize=10)
ax.set_title(f'CDF: KS stat={ks_stat:.4f}', fontweight='bold', color='green')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Poor fit - Histogram + PDF
ax = axes[1, 0]
ax.hist(data_poor, bins=40, density=True, alpha=0.6, color='salmon', 
        edgecolor='black', label='Exp Data')
x_poor = np.linspace(data_poor.min(), data_poor.max(), 200)
ax.plot(x_poor, dist_wrong.pdf(x_poor), 'r-', linewidth=2, label='Wrong: Normal')
ax.plot(x_poor, dist_correct.pdf(x_poor), 'g-', linewidth=2, label='Correct: Exp')
ax.set_xlabel('Value', fontsize=10)
ax.set_ylabel('Density', fontsize=10)
ax.set_title(f'Poor Fit: p={ks_pvalue_wrong:.4f}', fontweight='bold', color='red')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 5: Poor fit - QQ plot
ax = axes[1, 1]
theoretical_q_wrong = dist_wrong.ppf(np.linspace(0.01, 0.99, len(data_poor)))
empirical_q_poor = np.sort(data_poor)
ax.scatter(theoretical_q_wrong, empirical_q_poor, alpha=0.5, s=10, color='red')
min_v = min(theoretical_q_wrong.min(), empirical_q_poor.min())
max_v = max(theoretical_q_wrong.max(), empirical_q_poor.max())
ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10)
ax.set_ylabel('Empirical Quantiles', fontsize=10)
ax.set_title('Q-Q Plot: Poor Fit', fontweight='bold', color='red')
ax.grid(True, alpha=0.3)

# Plot 6: p-value comparison
ax = axes[1, 2]
names = [r['name'] for r in sorted(test_results, key=lambda x: x['ks_pval'], reverse=True)]
pvals = [r['ks_pval'] for r in sorted(test_results, key=lambda x: x['ks_pval'], reverse=True)]
colors = ['green' if p > 0.05 else 'red' for p in pvals]
ax.barh(names, pvals, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(0.05, color='black', linestyle='--', linewidth=2, label='α=0.05')
ax.set_xlabel('p-value', fontsize=10)
ax.set_title('KS Test p-values (Example 3)', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

plt.tight_layout()

print("\n✅ Plots created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. GOODNESS-OF-FIT TESTS:
   • Statistical tests to assess distribution fit
   • Complement visual inspection (QQ-plots)
   • Use p-value > 0.05 as threshold (typically)

2. THREE MAIN TESTS:
   • KS: General purpose, works for any distribution
   • Chi-square: For binned data, discrete distributions
   • Anderson-Darling: Sensitive to tails

3. INTERPRETING RESULTS:
   • p-value > 0.05: Don't reject (distribution may fit)
   • p-value < 0.05: Reject (distribution doesn't fit)
   • Lower p-value = stronger evidence against fit

4. IMPORTANT NOTES:
   ⚠️  "Don't reject" ≠ "distribution is correct"
   ⚠️  Tests lose power with small samples
   ⚠️  Tests are sensitive with large samples
   ⚠️  Always use multiple tests + visual checks

5. BEST PRACTICES:
   ✓ Use GoF tests + information criteria (AIC/BIC)
   ✓ Check QQ-plots visually
   ✓ Test multiple candidate distributions
   ✓ Consider domain knowledge
   ✓ Use highest p-value (or lowest KS stat)

6. IN scipy:
   from scipy import stats
   stats.kstest(data, dist.cdf)      # KS test
   stats.chisquare(obs, exp)         # Chi-square
   stats.anderson(data, dist='norm') # AD test

Next: See auto_selection.py for automatic distribution selection!
""")
