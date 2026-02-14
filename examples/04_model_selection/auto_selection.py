#!/usr/bin/env python3
"""
Automatic Distribution Selection
================================

Complete workflow to automatically find the best distribution.

Combines:
  1. Try multiple candidate distributions
  2. Fit each using MLE
  3. Compare using AIC/BIC
  4. Validate with GoF tests
  5. Visual diagnostics

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
from scipy import stats
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

print("="*70)
print("🤖 AUTOMATIC DISTRIBUTION SELECTION")
print("="*70)


# ============================================================================
# Automatic Selection Function
# ============================================================================

def select_best_distribution(data, candidates=None, criterion='aic', 
                            return_top_n=3, verbose=True):
    """
    Automatically select best distribution for data.
    
    Parameters:
    -----------
    data : array-like
        Data to fit
    candidates : list, optional
        List of distribution names to try
    criterion : str
        'aic' or 'bic' for model selection
    return_top_n : int
        Number of top distributions to return
    verbose : bool
        Print progress
    
    Returns:
    --------
    list of dicts with distribution info
    """
    
    if candidates is None:
        # Default candidates for continuous data
        candidates = [
            'normal', 'lognormal', 'gamma', 'weibull_min', 
            'expon', 'beta', 't', 'chi2', 'uniform'
        ]
    
    results = []
    
    if verbose:
        print(f"\n🔍 Testing {len(candidates)} distributions...\n")
        print(f"{'Distribution':<15} {'Status':<10} {criterion.upper():<10} {'KS p-value':<12}")
        print("  " + "-"*50)
    
    for name in candidates:
        try:
            # Fit distribution
            dist = get_distribution(name)
            dist.fit(data)
            
            # Calculate metrics
            if criterion == 'aic':
                ic_value = dist.aic()
            else:
                ic_value = dist.bic()
            
            # Goodness-of-fit test
            ks_stat, ks_pval = stats.kstest(data, dist.cdf)
            
            results.append({
                'name': name,
                'dist': dist,
                'ic': ic_value,
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pval,
                'params': dist.params,
                'log_likelihood': dist.log_likelihood()
            })
            
            if verbose:
                status = "✅ Good" if ks_pval > 0.05 else "⚠️  Weak"
                print(f"  {name:<15} {status:<10} {ic_value:<10.2f} {ks_pval:<12.6f}")
            
        except Exception as e:
            if verbose:
                print(f"  {name:<15} ✗ Failed   {str(e)[:30]}")
    
    # Sort by information criterion
    results.sort(key=lambda x: x['ic'])
    
    if verbose:
        print(f"\n✅ Successfully fitted {len(results)}/{len(candidates)} distributions")
    
    return results[:return_top_n]


# ============================================================================
# Example 1: Unknown Distribution (Lognormal)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Identify Unknown Distribution")
print("="*70)

# Generate data (user doesn't know it's lognormal)
data_unknown = np.random.lognormal(mean=3, sigma=0.5, size=800)

print(f"\n📊 Analyzing {len(data_unknown)} data points...")
print(f"  Mean:     {data_unknown.mean():.2f}")
print(f"  Median:   {np.median(data_unknown):.2f}")
print(f"  Std:      {data_unknown.std():.2f}")
print(f"  Skewness: {stats.skew(data_unknown):.2f}")
print(f"  Range:    [{data_unknown.min():.2f}, {data_unknown.max():.2f}]")

# Automatic selection
top_distributions = select_best_distribution(data_unknown, criterion='aic')

print("\n" + "="*70)
print("🏆 TOP 3 DISTRIBUTIONS")
print("="*70)

for rank, result in enumerate(top_distributions, 1):
    print(f"\n{rank}. {result['name'].upper()}")
    print(f"   AIC: {result['ic']:.2f}")
    print(f"   KS test p-value: {result['ks_pvalue']:.6f}")
    
    if result['ks_pvalue'] > 0.05:
        print(f"   ✅ Good fit (p > 0.05)")
    else:
        print(f"   ⚠️  Questionable fit (p < 0.05)")
    
    print(f"   Parameters:")
    for param, val in result['params'].items():
        print(f"     {param}: {val:.4f}")

best = top_distributions[0]
print(f"\n🎯 RECOMMENDED: {best['name']}")
print(f"   (True distribution was: lognormal)")


# ============================================================================
# Example 2: Compare AIC vs BIC Selection
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: AIC vs BIC Selection")
print("="*70)

# Generate data
data_compare = np.random.gamma(shape=2, scale=3, size=500)

print(f"\n📊 Data: {len(data_compare)} samples from Gamma(2, 3)")

# Select by AIC
print("\n1️⃣ Selection by AIC:")
top_aic = select_best_distribution(data_compare, criterion='aic', 
                                   return_top_n=3, verbose=False)

print(f"\n  Top 3:")
for i, r in enumerate(top_aic, 1):
    print(f"    {i}. {r['name']:<15} AIC={r['ic']:.2f}")

# Select by BIC
print("\n2️⃣ Selection by BIC:")
top_bic = select_best_distribution(data_compare, criterion='bic', 
                                   return_top_n=3, verbose=False)

print(f"\n  Top 3:")
for i, r in enumerate(top_bic, 1):
    print(f"    {i}. {r['name']:<15} BIC={r['ic']:.2f}")

if top_aic[0]['name'] != top_bic[0]['name']:
    print(f"\n⚠️  AIC and BIC chose different models!")
    print(f"   AIC winner: {top_aic[0]['name']}")
    print(f"   BIC winner: {top_bic[0]['name']}")
    print(f"   → Both are likely good fits; choose based on goal")
else:
    print(f"\n✅ AIC and BIC agree: {top_aic[0]['name']}")


# ============================================================================
# Example 3: Custom Candidate Set
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: Custom Candidate Set (Symmetric Only)")
print("="*70)

# Generate symmetric data
data_sym = np.random.normal(50, 10, 400)

print(f"\n📊 Testing only symmetric distributions...")

# Only test symmetric distributions
symmetric_candidates = ['normal', 't', 'logistic', 'uniform']

top_sym = select_best_distribution(data_sym, candidates=symmetric_candidates,
                                   criterion='aic', return_top_n=2,
                                   verbose=True)

print(f"\n🎯 Best symmetric distribution: {top_sym[0]['name']}")


# ============================================================================
# Visualization: Top 3 Models
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Comparison Visualization...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Histogram with top 3 fits
ax = axes[0, 0]
ax.hist(data_unknown, bins=50, density=True, alpha=0.3, color='gray',
        edgecolor='black', label='Data')

x = np.linspace(data_unknown.min(), data_unknown.max(), 200)
colors = ['red', 'blue', 'green']
labels = ['🥇 1st', '🥈 2nd', '🥉 3rd']

for i, result in enumerate(top_distributions[:3]):
    y = result['dist'].pdf(x)
    ax.plot(x, y, color=colors[i], linewidth=2, 
            label=f"{labels[i]}: {result['name']} (AIC={result['ic']:.0f})")

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Top 3 Fitted Distributions', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Q-Q plot for best fit
ax = axes[0, 1]
best_dist = top_distributions[0]['dist']
theoretical = best_dist.ppf(np.linspace(0.01, 0.99, len(data_unknown)))
empirical = np.sort(data_unknown)

ax.scatter(theoretical, empirical, alpha=0.5, s=20, color='blue')
min_val = min(theoretical.min(), empirical.min())
max_val = max(theoretical.max(), empirical.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

ax.set_xlabel(f'Theoretical Quantiles ({best["name"]})', fontsize=10)
ax.set_ylabel('Sample Quantiles', fontsize=10)
ax.set_title(f'Q-Q Plot: Best Fit ({best["name"]})', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Plot 3: AIC comparison
ax = axes[1, 0]
names = [r['name'] for r in top_distributions]
aics = [r['ic'] for r in top_distributions]
colors_bar = ['green', 'orange', 'red']
ax.barh(names, aics, color=colors_bar, edgecolor='black', alpha=0.7)
ax.set_xlabel('AIC (lower is better)', fontsize=11)
ax.set_title('AIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 4: Summary table
ax = axes[1, 1]
ax.axis('off')

summary = f"""
AUTOMATIC SELECTION SUMMARY
{'='*40}

Best Distribution: {best['name'].upper()}
{'-'*40}

Parameters:
"""

for param, val in best['params'].items():
    summary += f"  {param:<10s} = {val:>10.4f}\n"

summary += f"""
{'='*40}

Model Fit Quality:
{'-'*40}
  AIC:             {best['ic']:>10.2f}
  Log-Likelihood:  {best['log_likelihood']:>10.2f}
  KS Statistic:    {best['ks_statistic']:>10.6f}
  KS P-value:      {best['ks_pvalue']:>10.6f}

{'='*40}

Interpretation:
{'-'*40}
"""

if best['ks_pvalue'] > 0.05:
    summary += f"  ✅ GOOD FIT\n  (p > 0.05, cannot reject H₀)\n"
else:
    summary += f"  ⚠️  WEAK FIT\n  (p < 0.05, reject H₀)\n"

summary += f"""
{'='*40}

Recommendation:
{'-'*40}
  Use {best['name']} distribution
  for modeling this data.
"""

ax.text(0.05, 0.95, summary, transform=ax.transAxes,
        fontsize=9, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.tight_layout()

print("\n✅ Visualization created!")
plt.show()


print("\n" + "="*70)
print("🎓 Automatic Selection Workflow")
print("="*70)
print("""
🔄 RECOMMENDED WORKFLOW:

1️⃣ EXPLORE DATA:
   • Plot histogram
   • Check skewness, range
   • Look for outliers

2️⃣ CHOOSE CANDIDATES:
   • Based on data characteristics
   • Right-skewed → lognormal, gamma, weibull
   • Symmetric → normal, t, logistic
   • Bounded → beta, uniform

3️⃣ FIT & COMPARE:
   • Fit all candidates
   • Compare AIC/BIC
   • Check GoF tests

4️⃣ VALIDATE:
   • Q-Q plots
   • CDF comparison
   • Residual analysis

5️⃣ DECIDE:
   • Balance fit quality vs simplicity
   • Consider domain knowledge
   • Check practical implications

💡 PRO TIPS:

  • Don't blindly trust automatic selection
  • Always visualize the fit
  • Consider multiple "good" distributions
  • Domain knowledge > statistical tests
  • Simpler models often better for prediction

⚠️  WARNINGS:

  • No distribution may fit perfectly
  • Multiple testing increases false positives
  • Large samples: tests reject good fits
  • Small samples: tests miss poor fits

Next: Check 05_visualization/ for advanced plotting techniques!
""")
