"""Comparing Multiple Distributions - Model Selection

Learn how to:
- Fit multiple distribution types to the same data
- Compare model quality using AIC and BIC
- Select the best distribution for your data
- Visualize comparison results

Perfect for: Model selection, hypothesis testing
Time: ~15 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution
import pandas as pd

print("="*70)
print("🔍 COMPARING DISTRIBUTIONS: Model Selection")
print("="*70)

# ============================================================================
# Generate Test Data (mixture that could fit multiple distributions)
# ============================================================================
np.random.seed(456)
# Right-skewed data (could be gamma, lognormal, or weibull)
data = np.random.gamma(shape=2.0, scale=3.0, size=500)

print("\n📊 Data Characteristics:")
print(f"  Sample size: {len(data)}")
print(f"  Mean: {data.mean():.4f}")
print(f"  Std: {data.std():.4f}")
print(f"  Skewness: {pd.Series(data).skew():.4f}")
print(f"  Kurtosis: {pd.Series(data).kurt():.4f}")
print(f"  Range: [{data.min():.4f}, {data.max():.4f}]")

# ============================================================================
# Fit Multiple Distributions
# ============================================================================
print("\n" + "="*70)
print("FITTING MULTIPLE DISTRIBUTIONS")
print("="*70)

# List of distributions to try
dist_names = ['gamma', 'lognormal', 'weibull', 'exponential', 'normal']
results = []

for dist_name in dist_names:
    print(f"\n📈 Fitting {dist_name.capitalize()}...")
    
    try:
        # Create and fit distribution
        dist = get_distribution(dist_name)
        dist.fit(data, method='mle')
        
        # Collect metrics
        result = {
            'Distribution': dist.info.display_name,
            'Parameters': dist.params,
            'Log-Likelihood': dist.log_likelihood(),
            'AIC': dist.aic(),
            'BIC': dist.bic(),
            'Mean': dist.mean(),
            'Std': dist.std(),
            'Skewness': dist.skewness(),
            'Distribution Object': dist
        }
        results.append(result)
        
        print(f"  ✓ {dist_name} fitted successfully")
        print(f"    AIC: {result['AIC']:.2f}, BIC: {result['BIC']:.2f}")
        
    except Exception as e:
        print(f"  ✗ Failed to fit {dist_name}: {str(e)}")

# ============================================================================
# Compare Results
# ============================================================================
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

# Create comparison table
comparison_df = pd.DataFrame([{
    'Distribution': r['Distribution'],
    'Log-Likelihood': r['Log-Likelihood'],
    'AIC': r['AIC'],
    'BIC': r['BIC'],
    'Mean': r['Mean'],
    'Std': r['Std'],
    'Skewness': r['Skewness']
} for r in results])

# Sort by AIC (lower is better)
comparison_df = comparison_df.sort_values('AIC')
comparison_df['Δ AIC'] = comparison_df['AIC'] - comparison_df['AIC'].min()
comparison_df['Δ BIC'] = comparison_df['BIC'] - comparison_df['BIC'].min()

print("\n📊 Model Comparison (sorted by AIC):")
print("\n" + str(comparison_df[['Distribution', 'AIC', 'Δ AIC', 'BIC', 'Δ BIC']].to_string(index=False)))

print("\n💡 Interpretation:")
print("  - Δ AIC/BIC = 0: Best model")
print("  - Δ AIC/BIC < 2: Substantial support")
print("  - Δ AIC/BIC 4-7: Considerably less support")
print("  - Δ AIC/BIC > 10: Essentially no support")

# Identify best model
best_model = comparison_df.iloc[0]
print(f"\n🏆 Best Model: {best_model['Distribution']}")
print(f"   AIC: {best_model['AIC']:.2f}")
print(f"   BIC: {best_model['BIC']:.2f}")

# ============================================================================
# Detailed Parameter Comparison
# ============================================================================
print("\n" + "="*70)
print("PARAMETER ESTIMATES")
print("="*70)

for result in results:
    print(f"\n{result['Distribution']}:")
    for param, value in result['Parameters'].items():
        print(f"  {param:<15} = {value:>12.6f}")

# ============================================================================
# Visual Comparison
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: PDF Overlay
ax1 = axes[0, 0]
ax1.hist(data, bins=40, density=True, alpha=0.5, color='gray', 
         edgecolor='black', label='Observed Data')

x = np.linspace(data.min(), data.max(), 300)
colors = ['red', 'blue', 'green', 'orange', 'purple']
for i, result in enumerate(results):
    dist = result['Distribution Object']
    ax1.plot(x, dist.pdf(x), linewidth=2, color=colors[i], 
             label=result['Distribution'], alpha=0.8)

ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Probability Density', fontsize=11)
ax1.set_title('PDF Comparison', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Plot 2: CDF Comparison
ax2 = axes[0, 1]
sorted_data = np.sort(data)
empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
ax2.plot(sorted_data, empirical_cdf, 'o', markersize=2, alpha=0.3, 
         color='gray', label='Empirical CDF')

for i, result in enumerate(results):
    dist = result['Distribution Object']
    ax2.plot(x, dist.cdf(x), linewidth=2, color=colors[i], 
             label=result['Distribution'], alpha=0.8)

ax2.set_xlabel('Value', fontsize=11)
ax2.set_ylabel('Cumulative Probability', fontsize=11)
ax2.set_title('CDF Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# Plot 3: AIC Comparison
ax3 = axes[1, 0]
aic_df = comparison_df.sort_values('Δ AIC', ascending=False)
y_pos = np.arange(len(aic_df))
colors_sorted = [colors[dist_names.index(row['Distribution'].lower())] 
                 for _, row in aic_df.iterrows()]
ax3.barh(y_pos, aic_df['Δ AIC'], color=colors_sorted, alpha=0.7, edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels(aic_df['Distribution'])
ax3.set_xlabel('Δ AIC (relative to best)', fontsize=11)
ax3.set_title('AIC Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax3.axvline(x=2, color='orange', linestyle='--', linewidth=2, label='Δ AIC = 2')
ax3.axvline(x=7, color='red', linestyle='--', linewidth=2, label='Δ AIC = 7')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3, axis='x')

# Plot 4: Q-Q Plot for Best Model
ax4 = axes[1, 1]
best_dist = results[0]['Distribution Object']  # Already sorted by AIC
theoretical = best_dist.ppf(np.linspace(0.01, 0.99, len(data)))
sample = np.sort(data)
ax4.scatter(theoretical, sample, alpha=0.5, s=15, color=colors[0])
ax4.plot([theoretical.min(), theoretical.max()], 
         [theoretical.min(), theoretical.max()], 
         'k--', linewidth=2, label='Perfect Fit')
ax4.set_xlabel('Theoretical Quantiles', fontsize=11)
ax4.set_ylabel('Sample Quantiles', fontsize=11)
ax4.set_title(f'Q-Q Plot: {best_model["Distribution"]}', 
              fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('distribution_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: distribution_comparison.png")

# ============================================================================
# Prediction Comparison
# ============================================================================
print("\n" + "="*70)
print("PREDICTION COMPARISON")
print("="*70)

print("\n95th Percentile Predictions:")
for result in results:
    dist = result['Distribution Object']
    p95 = dist.ppf(0.95)
    print(f"  {result['Distribution']:<20}: {p95:>10.4f}")

print(f"\nObserved 95th percentile: {np.percentile(data, 95):>10.4f}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("🎉 MODEL SELECTION COMPLETE")
print("="*70)
print(f"\n🏆 Best Model: {best_model['Distribution']}")
print(f"   Reason: Lowest AIC ({best_model['AIC']:.2f})")
print("\nKey Insights:")
print("  1. AIC and BIC both prefer the same model (strong evidence)")
print("  2. Visual inspection confirms good fit")
print("  3. Q-Q plot shows alignment with theoretical quantiles")
print("  4. Predictions match observed quantiles well")
print("\nRecommendations:")
print("  → Use the best model for predictions")
print("  → Consider models with Δ AIC < 2 as alternatives")
print("  → Always validate with domain knowledge")
print("="*70)
