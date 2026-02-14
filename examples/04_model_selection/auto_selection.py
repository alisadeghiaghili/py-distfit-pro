#!/usr/bin/env python3
"""
Automatic Distribution Selection
================================

Automatically find the best distribution for your data.

Combines:
  - Information criteria (AIC, BIC)
  - Goodness-of-fit tests (KS test)
  - Visual assessment

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
print("✨ AUTOMATIC DISTRIBUTION SELECTION")
print("="*70)


# ============================================================================
# Function: Auto-Select Best Distribution
# ============================================================================

def auto_select_distribution(data, candidates=None, verbose=True):
    """
    Automatically select the best distribution for data.
    
    Parameters:
    -----------
    data : array-like
        Data to fit
    candidates : list, optional
        List of distribution names to try. If None, uses common distributions.
    verbose : bool
        Print progress and results
    
    Returns:
    --------
    results : list of dict
        Sorted results with fit metrics
    best : dict
        Best distribution info
    """
    
    # Default candidates if not specified
    if candidates is None:
        # Choose candidates based on data characteristics
        data_min = data.min()
        data_mean = data.mean()
        data_median = np.median(data)
        
        if data_min >= 0:
            # Positive data
            if data_mean > data_median * 1.5:
                # Right-skewed
                candidates = ['lognormal', 'gamma', 'weibull_min', 'expon']
            else:
                # Symmetric or mild skew
                candidates = ['normal', 'lognormal', 'gamma']
        else:
            # Can be negative
            candidates = ['normal', 't', 'logistic']
    
    if verbose:
        print(f"\n🔍 Testing {len(candidates)} distributions...")
        print(f"   Candidates: {', '.join(candidates)}")
    
    results = []
    
    for dist_name in candidates:
        try:
            # Fit distribution
            dist = get_distribution(dist_name)
            dist.fit(data)
            
            # Calculate metrics
            aic = dist.aic()
            bic = dist.bic()
            log_lik = dist.log_likelihood()
            
            # KS test
            ks_stat, ks_pval = stats.kstest(data, dist.cdf)
            
            # Store results
            results.append({
                'name': dist_name,
                'dist': dist,
                'aic': aic,
                'bic': bic,
                'log_lik': log_lik,
                'ks_stat': ks_stat,
                'ks_pval': ks_pval,
            })
            
            if verbose:
                print(f"   ✓ {dist_name:<15} AIC={aic:8.2f}  p-value={ks_pval:.4f}")
            
        except Exception as e:
            if verbose:
                print(f"   ✗ {dist_name:<15} Failed: {str(e)[:40]}")
    
    if not results:
        raise ValueError("No distributions could be fitted!")
    
    # Rank by composite score
    # Normalize metrics to [0, 1] and combine
    aic_vals = np.array([r['aic'] for r in results])
    ks_vals = np.array([r['ks_stat'] for r in results])
    
    aic_normalized = (aic_vals - aic_vals.min()) / (aic_vals.max() - aic_vals.min() + 1e-10)
    ks_normalized = ks_vals / (ks_vals.max() + 1e-10)
    
    # Composite score (lower is better)
    for i, r in enumerate(results):
        r['score'] = aic_normalized[i] + ks_normalized[i]
    
    # Sort by score
    results = sorted(results, key=lambda x: x['score'])
    
    best = results[0]
    
    if verbose:
        print(f"\n🏆 Best distribution: {best['name']}")
        print(f"   AIC:       {best['aic']:.2f}")
        print(f"   BIC:       {best['bic']:.2f}")
        print(f"   KS p-val:  {best['ks_pval']:.6f}")
        print(f"   KS stat:   {best['ks_stat']:.6f}")
    
    return results, best


# ============================================================================
# Example 1: Auto-Select for Skewed Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Auto-Selection for Right-Skewed Data")
print("="*70)

# Generate lognormal data
data1 = np.random.lognormal(mean=2.5, sigma=0.7, size=1000)

print(f"\n📊 Data: {len(data1)} observations")
print(f"  Mean:     {data1.mean():.2f}")
print(f"  Median:   {np.median(data1):.2f}")
print(f"  Skewness: {stats.skew(data1):.2f}")
print(f"  Range:    [{data1.min():.2f}, {data1.max():.2f}]")

# Auto-select
results1, best1 = auto_select_distribution(data1)

print("\n" + "="*70)
print("Top 3 Candidates:")
print("="*70)
print(f"\n{'Rank':<6} {'Distribution':<15} {'AIC':<10} {'BIC':<10} {'KS p-val':<12}")
print("-"*70)

for rank, r in enumerate(results1[:3], 1):
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
    print(f"{marker} {rank:<4} {r['name']:<15} {r['aic']:<10.2f} {r['bic']:<10.2f} {r['ks_pval']:<12.6f}")


# ============================================================================
# Example 2: Auto-Select for Symmetric Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Auto-Selection for Symmetric Data")
print("="*70)

# Generate normal data with some outliers
data2_clean = np.random.normal(50, 10, 950)
data2_outliers = np.random.normal(50, 30, 50)
data2 = np.concatenate([data2_clean, data2_outliers])
np.random.shuffle(data2)

print(f"\n📊 Data: {len(data2)} observations (with outliers)")
print(f"  Mean:     {data2.mean():.2f}")
print(f"  Median:   {np.median(data2):.2f}")
print(f"  Skewness: {stats.skew(data2):.2f}")

# Auto-select with specific candidates
candidates2 = ['normal', 't', 'logistic', 'laplace']
results2, best2 = auto_select_distribution(data2, candidates=candidates2)

print("\n" + "="*70)
print("Results (Symmetric Distributions):")
print("="*70)
print(f"\n{'Distribution':<15} {'AIC':<10} {'KS p-val':<12} {'Status':<10}")
print("-"*70)

for r in results2:
    status = "✅ PASS" if r['ks_pval'] > 0.05 else "❌ FAIL"
    print(f"{r['name']:<15} {r['aic']:<10.2f} {r['ks_pval']:<12.6f} {status:<10}")


# ============================================================================
# Example 3: One-Line Solution (Quick & Easy)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 3: One-Line Quick Solution")
print("="*70)

print("""
For quick analysis, just use auto_select_distribution():
""")

# Generate some data
data3 = np.random.gamma(shape=2, scale=3, size=500)

print("\n# Your code:")
print("results, best = auto_select_distribution(data3)")
print("print(f'Best: {best[\'name\']} with AIC={best[\'aic\']:.2f}')")
print()

# Execute
results3, best3 = auto_select_distribution(data3, verbose=False)
print(f"\n✅ Best: {best3['name']} with AIC={best3['aic']:.2f}")


# ============================================================================
# Visualization: Compare Top Candidates
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Comparison Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Automatic Distribution Selection Results', fontsize=16, fontweight='bold')

# Plot 1: Example 1 - PDF overlay
ax = axes[0, 0]
ax.hist(data1, bins=50, density=True, alpha=0.4, color='skyblue', 
        edgecolor='black', label='Data')

x1 = np.linspace(data1.min(), data1.max(), 200)
for i, r in enumerate(results1[:3]):
    linestyle = '-' if i == 0 else '--' if i == 1 else ':'
    ax.plot(x1, r['dist'].pdf(x1), linewidth=2, linestyle=linestyle, 
            label=f"{r['name']} (rank {i+1})")

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 1: Right-Skewed Data', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Example 1 - QQ plot for best
ax = axes[0, 1]
theoretical_q = best1['dist'].ppf(np.linspace(0.01, 0.99, len(data1)))
empirical_q = np.sort(data1)
ax.scatter(theoretical_q, empirical_q, alpha=0.5, s=10, color='green')
min_v = min(theoretical_q.min(), empirical_q.min())
max_v = max(theoretical_q.max(), empirical_q.max())
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Fit')
ax.set_xlabel(f'Theoretical Quantiles ({best1["name"]})', fontsize=10)
ax.set_ylabel('Empirical Quantiles', fontsize=10)
ax.set_title(f'Q-Q Plot: Best Model ({best1["name"]})', fontweight='bold', color='green')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Example 2 - AIC comparison
ax = axes[1, 0]
names = [r['name'] for r in results2]
aic_values = [r['aic'] for r in results2]
colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' for i in range(len(names))]
ax.barh(names, aic_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('AIC (lower = better)', fontsize=11)
ax.set_title('Example 2: AIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 4: Example 2 - PDF overlay
ax = axes[1, 1]
ax.hist(data2, bins=50, density=True, alpha=0.4, color='salmon', 
        edgecolor='black', label='Data (with outliers)')

x2 = np.linspace(data2.min(), data2.max(), 200)
for i, r in enumerate(results2[:2]):
    linestyle = '-' if i == 0 else '--'
    ax.plot(x2, r['dist'].pdf(x2), linewidth=2, linestyle=linestyle, 
            label=f"{r['name']} (rank {i+1})")

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title(f"Example 2: Best = {best2['name']}", fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Plots created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. AUTOMATIC SELECTION:
   ✓ Saves time - no manual trial and error
   ✓ Combines multiple criteria (AIC, KS test)
   ✓ Works for most common scenarios
   ❌ May miss domain-specific requirements

2. HOW IT WORKS:
   1. Select candidates based on data characteristics
   2. Fit all candidate distributions
   3. Calculate AIC, BIC, KS test for each
   4. Rank by composite score
   5. Return best + all results

3. WHEN TO USE:
   • Quick exploratory analysis
   • Benchmark for manual selection
   • When unsure which distribution to try
   • Batch processing multiple datasets

4. LIMITATIONS:
   • Assumes standard distributions
   • May not respect domain knowledge
   • Statistical "best" ≠ practical "best"
   • Always validate results visually!

5. CUSTOMIZATION:
   • Provide custom candidate list
   • Adjust composite scoring weights
   • Add domain-specific constraints
   • Combine with expert judgment

6. BEST PRACTICES:
   ✓ Use as starting point, not final answer
   ✓ Check top 2-3 candidates, not just #1
   ✓ Visualize results (QQ-plot, histogram)
   ✓ Consider interpretability vs fit
   ✓ Validate on holdout data if available

7. USAGE:
   from distfit_pro import get_distribution
   
   # Quick usage
   results, best = auto_select_distribution(data)
   best_dist = best['dist']
   
   # Custom candidates
   results, best = auto_select_distribution(
       data, 
       candidates=['normal', 'lognormal', 'gamma']
   )

Next: See 05_visualization/ for advanced plotting techniques!
""")
